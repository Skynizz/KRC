//! Driver of the fully GPU KRC pipeline (krc_gpu.wgsl). Per frame, the CPU only sends the
//! scene (a few hundred bytes) and, when needed, the description of the event; it only
//! reads back the image and a few counters, for evaluation (outside GPU timing).
use crate::deps::{Coarse, SUPER, WORDS};
use crate::gpu::{f3, lights_to_gpu, scene_to_gpu, GBox, GLight, Gpu, MAX_BOXES, MAX_LIGHTS};
use crate::grid::Grid;
use crate::math::{v3, Rng, V3};
use crate::methods::Event;
use crate::render::World;
use crate::rt::{Accel, TriScene};
use crate::scene::{cosine_dir, Scene};

/// Camera and resolution.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub w: usize,
    pub h: usize,
    pub eye: V3,
    pub fwd: V3,
    pub right: V3,
    pub up: V3,
    pub th: f64,
    pub tv: f64,
}

impl View {
    pub fn of(world: &World) -> View {
        let (eye, fwd, right, up, th, tv) = world.camera();
        View { w: world.w, h: world.h, eye, fwd, right, up, th, tv }
    }
    pub fn len(&self) -> usize {
        self.w * self.h
    }
    pub fn px_angle(&self) -> f64 {
        2.0 * self.th / self.w as f64
    }
}

/// Description of an event for the GPU (words of `ev`), if the scene changed.
#[derive(Clone)]
pub struct GpuEvent {
    pub words: Vec<u32>,
    pub geometry: bool,
    pub lights: u32,
    pub light_w: Vec<f64>,
    pub probe_rays: usize,
}

const PROBES: usize = 1024;

const K: usize = 4;
const ACC: usize = 52;
const CELL_BYTES: usize = 160;
const CTR: usize = 256;

// counter indices (must follow krc_gpu.wgsl)
const C_CELLS: usize = 0;
const C_TASKS: usize = 1;
const C_SURF: usize = 2;
const C_SUSPECTS: usize = 3;
const C_DPATHS: usize = 4;
const C_DOLD: usize = 5;
const C_DZERO: usize = 6;
const C_HARD: usize = 7;
const C_SAFETY: usize = 8;
const C_REG0: usize = 12;
const C_VISIBLE: usize = 13;
const C_OVERFLOW: usize = 15;
const C_FREE: usize = 20;
const C_TOMB: usize = 21;
const C_EVICT: usize = 22;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameU {
    w: u32,
    h: u32,
    n_boxes: u32,
    n_lights: u32,
    frame: u32,
    seed: u32,
    flags: u32,
    table_mask: u32,
    cell_cap: u32,
    round: u32,
    ev_lights: u32,
    pad0: u32,
    eye: [f32; 3],
    px_angle: f32,
    fwd: [f32; 3],
    th: f32,
    right: [f32; 3],
    tv: f32,
    up: [f32; 3],
    ppp: f32,
    clock: u32,
    rel_cap: f32,
    eps_energy: f32,
    n_post: f32,
    sh_cam_prev: [f32; 3],
    sh_scene_scale: f32,
    sh_rscale: f32,
    sh_cap: u32,
    sh_accum: u32,
    sh_stale: u32,
    sh_down: u32,
    cell_px: f32,
    rec_every: u32,
    tau: f32,
    prev_eye: [f32; 3],
    m_light: u32,
    prev_fwd: [f32; 3],
    debt_rate: f32,
    prev_right: [f32; 3],
    reset_ratio: f32,
    prev_up: [f32; 3],
    dn_gamma: f32,
    sh_spp: u32,
    krc_spp: u32,
    dbg: u32,
    off: u32,
}

/// Pipeline method.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Method {
    Krc,
    /// home-made SHaRC-like cache (KRC cells, running average, 1 path per pixel)
    SharcLike,
    /// WGSL port of NVIDIA's official SHaRC (sharc.wgsl)
    Sharc,
}

/// Parameters of the official SHaRC: default values of NVIDIA RTXGI's Pathtracer sample
/// (UI.h: sceneScale 50, accumulation 20 frames, eviction after 60, 1 pixel per 5×5 block;
/// Pathtracer.cpp: 4 M entries).
#[derive(Clone, Copy, Debug)]
pub struct ShParams {
    pub scene_scale: f32,
    pub accum_frames: u32,
    pub stale_frames: u32,
    /// radiance quantisation (u32): the sample uses 1e3 for radiances ~1; our
    /// scenes are ~30× darker (Sponza), hence 1e4 so that truncation stays < 0.2 %
    pub radiance_scale: f32,
    pub downscale: u32,
    /// paths per pixel of the Render pass (the sample: samplesPerPixel, 1 by default)
    pub render_spp: u32,
    /// final image: Render pass (path per pixel, cache at secondary hits) instead of
    /// the cache view at the first hit
    pub render: bool,
}

impl Default for ShParams {
    fn default() -> Self {
        ShParams {
            scene_scale: 50.0,
            accum_frames: 20,
            stale_frames: 60,
            radiance_scale: 1e4,
            downscale: 5,
            render_spp: 1,
            render: false,
        }
    }
}

const SH_CAPACITY: usize = 1 << 22;

const F_EVENT: u32 = 1;
const F_GEOMETRY: u32 = 2;
const F_SHARC: u32 = 4;
const F_PENDING: u32 = 8;
const F_AMORTIZE: u32 = 16;
const F_S2HIST: u32 = 32;
const F_ZCAP: u32 = 64;
const F_RESAMPLE: u32 = 128;
const F_REUSE: u32 = 256;
const F_DEMOD: u32 = 512;
const F_SEED: u32 = 1024;
const F_SUPPORT: u32 = 2048;
const F_MULTI: u32 = 4096;
const F_INDIRECT: u32 = 8192;
const F_DSPLIT: u32 = 16384;
const F_GAME: u32 = 32768;
const F_DRS: u32 = 65536;

/// Result of a frame.
pub struct GpuFrame {
    pub image: Vec<V3>,
    pub cells: u32,
    pub visible: u32,
    pub surf: u32,
    pub regular_paths: u32,
    pub suspects: u32,
    pub delta_paths: u32,
    pub delta_old: u32,
    pub delta_zero: u32,
    pub hard: u32,
    pub safety: u32,
    pub overflow: u32,
    /// cells still waiting for their correlated paths (amortised event)
    pub pending: u32,
    pub max_n: f32,
    pub debt: u32,
    /// cells evicted in the frame, free cells, tombstones (moving camera)
    pub evicted: u32,
    pub free: u32,
    pub _tombs: u32,
    /// segments traced by ordinary paths, and paths stopped on the cache
    pub segs: u32,
    pub rs_hits: u32,
    /// kind of each pixel (0 empty, 1 lamp, 2 surface), if the image was read back
    pub kinds: Vec<u8>,
    pub probe_rays: usize,
    /// GPU duration of each pass (timestamps), in ms
    pub passes: Vec<(&'static str, f64)>,
}

impl GpuFrame {
    pub fn gpu_ms(&self) -> f64 {
        self.passes.iter().map(|p| p.1).sum()
    }
}

pub struct GpuKrc<'a> {
    gpu: &'a Gpu,
    view: View,
    grid: Grid,
    sharc: bool,
    pipes: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
    bufs: Vec<wgpu::Buffer>,
    /// one binding group (boxes); two for triangles (current/previous TLAS alternating)
    binds: Vec<wgpu::BindGroup>,
    bind_idx: usize,
    accel: Option<Accel>,
    probe_buf: wgpu::Buffer,
    probe_read: wgpu::Buffer,
    read_img: wgpu::Buffer,
    read_ctr: wgpu::Buffer,
    ts: wgpu::QuerySet,
    ts_resolve: wgpu::Buffer,
    ts_read: wgpu::Buffer,
    period_ns: f32,
    cell_cap: u32,
    table_mask: u32,
    seed: u32,
    /// read the image back every frame (evaluation); the speed bench does without
    pub readback: bool,
    /// budget: paths per surface pixel per frame
    pub ppp: f32,
    /// amortise lamp events (correlated paths capped per frame)
    pub amortize: bool,
    /// cap of the relative allocation variance, in multiples of the median (0: none)
    pub rel_cap: f32,
    /// threshold of the expected energy effect (fraction of the cell's luminance)
    pub eps_energy: f32,
    /// max weight of a corrected estimate (0: the shader's N_POST)
    pub n_post: f32,
    /// variance after correction: at least the history's
    pub s2_hist: bool,
    /// lamp event: history capped for cells without a measured difference too
    pub zero_cap: bool,
    /// cache resampling: ordinary paths stopped at the 1st or 2nd bounce on a cell
    pub resample: bool,
    /// width of a cell, in pixels (at the surface's distance)
    pub cell_px: f32,
    /// one ordinary path in `rec_every` records its dependencies (1: all)
    pub rec_every: u32,
    /// target quality (relative error of a cell's mean); 0: fixed budget
    pub tau: f32,
    /// vertex reuse: the 1st bounce also feeds the visible cell it hits
    pub reuse: bool,
    /// material demodulation (radiance / albedo in the cells)
    pub demod: bool,
    /// new cell initialised from the neighbouring level
    pub seed_levels: bool,
    /// off-screen support cells for resampling
    pub support: bool,
    /// two-resolution cache (fine cells and 2× larger ones)
    pub multi: bool,
    /// "game-like" pipeline: per-pixel final image (KRC: per-pixel direct + indirect from the
    /// cells; SHaRC: Render pass), then a shared denoiser (game.wgsl)
    pub game: bool,
    /// denoiser of the game pipeline
    pub denoise: bool,
    /// game mode, old variant: "indirect only" cells
    pub indirect_only: bool,
    /// denoiser with 5 à-trous passes (steps 1 to 16) instead of 3
    pub dn_wide: bool,
    /// game mode: per-pixel estimates of direct lighting and reflection (KRC)
    pub krc_spp: u32,
    /// diagnostic of KRC's final image: 1 cells' indirect alone, 2 direct + reflection alone
    pub dbg: u32,
    /// protections disabled (diagnostic): 1 anti-firefly, 2 truncated mean, 4 cell
    /// clamping, 8 path regularisation
    pub off: u32,
    /// regularised reference (same paths as game mode beyond the first vertex)
    pub ref_reg: bool,
    /// correlated paths stopped on the cache (cheaper events)
    pub delta_resample: bool,
    /// correlated paths per cell for a lamp event (0: 8)
    pub m_light: u32,
    /// share of the budget that pays back event paths (< 0: 0.5)
    pub debt_rate: f32,
    /// relative change beyond which a corrected cell restarts from zero (0: never)
    pub reset_ratio: f32,
    /// denoiser history clamp, in standard deviations (0: 1.5)
    pub dn_gamma: f32,
    prev_view: View,
    pending: Option<GpuEvent>,
    _free_buf: wgpu::Buffer,
    _vis_buf: wgpu::Buffer,
    _sh_bufs: [wgpu::Buffer; 4],
    _dn_bufs: [wgpu::Buffer; 4],
    _spec_bufs: [wgpu::Buffer; 3],
    _sampler: wgpu::Sampler,
    /// monotonic frame counter (eviction of invisible cells)
    clock: u32,
    /// the hash table has too many tombstones: rebuild at the start of the next frame
    rebuild: bool,
    method: Method,
    /// parameters of the official SHaRC (Sharc method)
    pub sh: ShParams,
    sh_cap: u32,
    sh_cam_prev: V3,
}

const ENTRIES: [&str; 50] = [
    "krc_refl",
    "game_combine",
    "dn_spec",
    "dn_spec_store",
    "dn_atrous3b",
    "dn_atrous4",
    "dn_atrous5",
    "krc_final",
    "dn_temporal",
    "dn_store",
    "dn_atrous1",
    "dn_atrous2",
    "dn_atrous3",
    "prep_vis2",
    "evict_sweep",
    "sharc_update",
    "sharc_resolve",
    "sharc_query",
    "sharc_render",
    "rebuild_clear",
    "rebuild_insert",
    "pending_mark",
    "probe_trace",
    "reference_acc",
    "reset_ctr",
    "prep_cells",
    "gbuffer",
    "insert",
    "lookup",
    "event_mark",
    "prep_d1",
    "delta_trace1",
    "delta_more",
    "prep_d2",
    "delta_trace2",
    "delta_apply",
    "alloc_hist",
    "alloc_prep",
    "alloc_scan1",
    "alloc_pick1",
    "alloc_scan2",
    "alloc_pick2",
    "alloc_scan3",
    "alloc_pick3",
    "alloc_emit",
    "sharc_emit",
    "prep_reg",
    "trace",
    "resolve",
    "display",
];
const MAX_PASSES: u32 = 48;

impl<'a> GpuKrc<'a> {
    /// Rooms in a row (axis-aligned boxes), fixed grid.
    pub fn new(gpu: &'a Gpu, world: World, sharc: bool, seed: u32) -> GpuKrc<'a> {
        GpuKrc::with(gpu, View::of(&world), Grid::boxes(), None, sharc, seed)
    }

    /// General case: `tri` = triangle scene (hardware ray query) or None for boxes.
    pub fn with(gpu: &'a Gpu, view: View, grid: Grid, tri: Option<&TriScene>, sharc: bool, seed: u32) -> GpuKrc<'a> {
        GpuKrc::with_method(gpu, view, grid, tri, if sharc { Method::SharcLike } else { Method::Krc }, seed)
    }

    pub fn with_method(gpu: &'a Gpu, view: View, grid: Grid, tri: Option<&TriScene>, method: Method, seed: u32) -> GpuKrc<'a> {
        let sharc = method != Method::Krc;
        let sh_cap = if method == Method::Sharc { SH_CAPACITY } else { 16 };
        let dev = &gpu.device;
        let len = view.len();
        // a cell covers 2 to 4 pixels in width; with the jitter, ≈ len/6 distinct cells per
        // view; a moving camera creates ~3,000 per frame at 480×270 and keeps them 60 frames before
        // evicting them (~250,000 live cells measured in Sponza), hence 2 per pixel
        let cell_cap = (2 * len).next_power_of_two().max(1 << 14) as u32;
        let table = (cell_cap as usize * 2).next_power_of_two();
        let task_cap = cell_cap as usize * 10 + len * 9;
        let storage = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let mut entries = Vec::new();
        for b in 0..38u32 {
            let ty = match b {
                0 => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                1 | 3 if tri.is_some() => wgpu::BindingType::AccelerationStructure,
                18..=20 | 32..=34 if tri.is_none() => continue,
                32 => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                33 => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                34 => storage(true),
                1..=4 | 14 | 18..=20 => storage(true),
                _ => storage(false),
            };
            entries.push(wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None });
        }
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("krc"), entries: &entries });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("krc"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let mut src = grid.template(include_str!("krc_gpu.wgsl")) + include_str!("sharc.wgsl") + include_str!("game.wgsl");
        if tri.is_some() {
            let (a, b) = (src.find("// <geometry>").unwrap(), src.find("// </geometry>").unwrap());
            src = format!("{}{}{}", &src[..a], crate::rt::TRI_GEOMETRY, &src[b..]);
        }
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("krc_gpu"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let mut pipes = std::collections::HashMap::new();
        for e in ENTRIES {
            pipes.entry(e).or_insert_with(|| {
                dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(e),
                    compilation_options: Default::default(),
                    cache: None,
                })
            });
        }
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let sizes: [(usize, wgpu::BufferUsages); 18] = [
            (std::mem::size_of::<FrameU>(), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            (MAX_BOXES * std::mem::size_of::<GBox>(), st),
            (MAX_LIGHTS * std::mem::size_of::<GLight>(), st),
            (MAX_BOXES * std::mem::size_of::<GBox>(), st),
            (MAX_LIGHTS * std::mem::size_of::<GLight>(), st),
            (len * 48, st),
            (len * K * 4, st),
            (table * 4, st),
            (table * 4, st),
            (CTR * 4, st),
            (cell_cap as usize * CELL_BYTES, st),
            (cell_cap as usize * ACC * 4, st),
            (task_cap * 16, st),
            (len * 16, st),
            (162 * 4, st),
            (32, st),
            (64, st | wgpu::BufferUsages::INDIRECT),
            (cell_cap as usize * 16, st),
        ];
        let bufs: Vec<wgpu::Buffer> = sizes
            .iter()
            .enumerate()
            .map(|(i, &(size, usage))| {
                dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("b{i}")),
                    size: size as u64,
                    usage,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let mk = |size: u64, usage: wgpu::BufferUsages| {
            dev.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false })
        };
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let probe_buf = mk((PROBES * 32) as u64, st);
        let free_buf = mk(cell_cap as u64 * 4, st);
        let vis_buf = mk(cell_cap as u64 * 4, st);
        // denoiser: history, two work buffers, previous G-buffer (2 vec4 per pixel)
        let dn_bufs = [16u64, 16, 16, 32].map(|b| mk(len as u64 * b, st));
        // SHaRC: 64-bit keys, locks, accumulation (uint4), resolved (half4 + 2 u32)
        let sh_bufs = [8u64, 4, 16, 16].map(|b| mk(sh_cap as u64 * b, st));
        let accel = tri.map(|t| Accel::new(gpu, t));
        let sampler = dev.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("textures"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // game mode: specular part of the image (current, history, work)
        let spec_bufs = [16u64, 16, 16].map(|b| mk(len as u64 * b, st));
        let n_binds = if accel.is_some() { 2 } else { 1 };
        let binds = (0..n_binds)
            .map(|k| {
                let mut e: Vec<wgpu::BindGroupEntry> = Vec::new();
                for (i, b) in bufs.iter().enumerate() {
                    let resource = match (&accel, i) {
                        (Some(a), 1) => a.tlas[k].as_binding(),
                        (Some(a), 3) => a.tlas[1 - k].as_binding(),
                        _ => b.as_entire_binding(),
                    };
                    e.push(wgpu::BindGroupEntry { binding: i as u32, resource });
                }
                if let Some(a) = &accel {
                    e.push(wgpu::BindGroupEntry { binding: 18, resource: a.verts.as_entire_binding() });
                    e.push(wgpu::BindGroupEntry { binding: 19, resource: a.idx.as_entire_binding() });
                    e.push(wgpu::BindGroupEntry { binding: 20, resource: a.mesh_table.as_entire_binding() });
                    e.push(wgpu::BindGroupEntry { binding: 32, resource: wgpu::BindingResource::TextureView(&a.tex_view) });
                    e.push(wgpu::BindGroupEntry { binding: 33, resource: wgpu::BindingResource::Sampler(&sampler) });
                    e.push(wgpu::BindGroupEntry { binding: 34, resource: a.uvs.as_entire_binding() });
                }
                for (i, b) in spec_bufs.iter().enumerate() {
                    e.push(wgpu::BindGroupEntry { binding: 35 + i as u32, resource: b.as_entire_binding() });
                }
                e.push(wgpu::BindGroupEntry { binding: 21, resource: probe_buf.as_entire_binding() });
                e.push(wgpu::BindGroupEntry { binding: 22, resource: free_buf.as_entire_binding() });
                e.push(wgpu::BindGroupEntry { binding: 27, resource: vis_buf.as_entire_binding() });
                for (i, b) in dn_bufs.iter().enumerate() {
                    e.push(wgpu::BindGroupEntry { binding: 28 + i as u32, resource: b.as_entire_binding() });
                }
                for (i, b) in sh_bufs.iter().enumerate() {
                    e.push(wgpu::BindGroupEntry { binding: 23 + i as u32, resource: b.as_entire_binding() });
                }
                dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("krc"), layout: &layout, entries: &e })
            })
            .collect();
        GpuKrc {
            gpu,
            view,
            grid,
            sharc,
            pipes,
            probe_read: mk((PROBES * 32) as u64, read),
            probe_buf,
            accel,
            binds,
            bind_idx: 0,
            read_img: mk((len * 16) as u64, read),
            read_ctr: mk((CTR * 4) as u64, read),
            ts: dev.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Timestamp,
                count: MAX_PASSES * 2,
            }),
            ts_resolve: mk(MAX_PASSES as u64 * 16, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            ts_read: mk(MAX_PASSES as u64 * 16, read),
            period_ns: gpu.queue.get_timestamp_period(),
            bufs,
            cell_cap,
            table_mask: table as u32 - 1,
            seed,
            readback: true,
            ppp: 1.0,
            amortize: false,
            rel_cap: 1.0,
            eps_energy: 0.01,
            n_post: 16.0,
            s2_hist: true,
            zero_cap: true,
            // KRC v2 (RESULTS §13): resampling, prior from the neighbouring level, demodulation,
            // 8 px cells, target quality 5 %
            resample: true,
            cell_px: 8.0,
            rec_every: 1,
            tau: 0.05,
            reuse: false,
            demod: true,
            seed_levels: true,
            support: false,
            multi: false,
            game: false,
            denoise: true,
            indirect_only: false,
            dn_wide: false,
            krc_spp: 1,
            dbg: 0,
            off: 0,
            ref_reg: false,
            delta_resample: false,
            m_light: 0,
            debt_rate: -1.0,
            reset_ratio: 0.0,
            dn_gamma: 0.0,
            prev_view: view,
            pending: None,
            clock: 0,
            rebuild: false,
            _free_buf: free_buf,
            _vis_buf: vis_buf,
            _sh_bufs: sh_bufs,
            _dn_bufs: dn_bufs,
            _spec_bufs: spec_bufs,
            _sampler: sampler,
            method,
            sh: ShParams::default(),
            sh_cap: sh_cap as u32,
            sh_cam_prev: view.eye,
        }
    }

    /// Changes the camera (same resolution). Cells are in world space: they stay.
    pub fn set_view(&mut self, v: View) {
        assert!(v.w == self.view.w && v.h == self.view.h);
        self.view = v;
    }

    fn uniform(&self, n_boxes: usize, n_lights: usize, t: u32, flags: u32, round: u32, ev_lights: u32, pad0: u32) -> FrameU {
        let v = &self.view;
        FrameU {
            w: v.w as u32,
            h: v.h as u32,
            n_boxes: n_boxes as u32,
            n_lights: n_lights as u32,
            frame: t,
            seed: self.seed,
            flags,
            table_mask: self.table_mask,
            cell_cap: self.cell_cap,
            round,
            ev_lights,
            pad0,
            eye: f3(v.eye),
            px_angle: v.px_angle() as f32,
            fwd: f3(v.fwd),
            th: v.th as f32,
            right: f3(v.right),
            tv: v.tv as f32,
            up: f3(v.up),
            ppp: self.ppp,
            clock: self.clock,
            rel_cap: self.rel_cap,
            eps_energy: self.eps_energy,
            n_post: self.n_post,
            sh_cam_prev: f3(self.sh_cam_prev),
            sh_scene_scale: self.sh.scene_scale,
            sh_rscale: self.sh.radiance_scale,
            sh_cap: self.sh_cap,
            sh_accum: self.sh.accum_frames,
            sh_stale: self.sh.stale_frames,
            sh_down: self.sh.downscale,
            cell_px: self.cell_px,
            rec_every: self.rec_every,
            tau: self.tau,
            prev_eye: f3(self.prev_view.eye),
            m_light: self.m_light,
            prev_fwd: f3(self.prev_view.fwd),
            debt_rate: self.debt_rate,
            prev_right: f3(self.prev_view.right),
            reset_ratio: self.reset_ratio,
            prev_up: f3(self.prev_view.up),
            dn_gamma: self.dn_gamma,
            sh_spp: self.sh.render_spp,
            krc_spp: self.krc_spp,
            dbg: self.dbg,
            off: self.off,
        }
    }

    /// A full frame, box scene. `t`: frame index (seeds, exploration).
    pub fn frame(&mut self, scene: &Scene, prev: &Scene, ev: Option<&Event>, t: u32) -> GpuFrame {
        let gev = ev.map(|e| GpuEvent {
            words: event_words(e),
            geometry: e.geometry,
            lights: e.lights as u32,
            light_w: e.light_w.clone(),
            probe_rays: e.probe_rays,
        });
        let w_ev: Vec<f64> =
            ev.map(|e| e.light_w.clone()).or(self.pending.as_ref().map(|e| e.light_w.clone())).unwrap_or_default();
        let (b, l) = scene_to_gpu(scene, &w_ev);
        let (bp, lp) = scene_to_gpu(prev, &w_ev);
        let q = &self.gpu.queue;
        q.write_buffer(&self.bufs[1], 0, bytemuck::cast_slice(&b));
        q.write_buffer(&self.bufs[3], 0, bytemuck::cast_slice(&bp));
        self.run_frame(b.len(), &l, &lp, gev.as_ref(), t)
    }

    /// A full frame, triangle scene: the frame's TLAS rebuilt (the previous one
    /// is kept for correlated paths), event detected and described on the spot.
    pub fn frame_tri(&mut self, scene: &TriScene, prev: &TriScene, t: u32) -> GpuFrame {
        let accel = self.accel.as_mut().expect("triangle scene");
        self.bind_idx = 1 - self.bind_idx;
        accel.build_tlas(self.gpu, scene, self.bind_idx);
        let ev = if self.sharc { None } else { self.tri_event(prev, scene, t) };
        let w_ev: Vec<f64> = ev.as_ref().or(self.pending.as_ref()).map(|e| e.light_w.clone()).unwrap_or_default();
        let l = lights_to_gpu(&scene.lights, &w_ev);
        let lp = lights_to_gpu(&prev.lights, &w_ev);
        self.run_frame(0, &l, &lp, ev.as_ref(), t)
    }

    /// Event between two states of a triangle scene: moved instances (swept AABB),
    /// switched lamps (probes traced on the GPU from the scene where the lamp is on).
    fn tri_event(&mut self, prev: &TriScene, cur: &TriScene, t: u32) -> Option<GpuEvent> {
        let g = self.grid;
        let mut words = vec![0u32; 162];
        let mut geometry = false;
        for i in 0..cur.instances.len() {
            if (cur.instances[i].offset - prev.instances[i].offset).len() > 0.0 {
                geometry = true;
                let swept = prev.instance_aabb(i).union(cur.instance_aabb(i)).expand(0.05);
                g.mark_aabb(&mut words[..108], &swept);
            }
        }
        let mut lights = 0u32;
        let mut light_w = vec![0.0; cur.lights.len()];
        let mut mag = [0.0f32; 36];
        let mut probe_rays = 0;
        for j in 0..cur.lights.len() {
            let (was, is) = (prev.lights[j].on, cur.lights[j].on);
            if was == is {
                continue;
            }
            lights |= 1 << j;
            light_w[j] = was as u8 as f64 - is as u8 as f64;
            let l = if is { &cur.lights[j] } else { &prev.lights[j] };
            let hits = self.probes(l, if is { 0 } else { 1 }, mix_seed(t, j));
            probe_rays += PROBES;
            let flux = std::f64::consts::PI * l.emission.lum() * l.area() / PROBES as f64;
            let area = g.region * g.region;
            g.mark_aabb(&mut words[..108], &l.aabb());
            for (p, albedo) in hits {
                let c = g.cell_of(p);
                let i = g.index(c);
                words[i >> 5] |= 1 << (i & 31);
                let s = g.super_of(c);
                mag[s] = mag[s].max((albedo * flux / (std::f64::consts::PI * area)) as f32);
            }
        }
        if !geometry && lights == 0 {
            return None;
        }
        let l2 = g.supers_of_bits(&words[..108]);
        words[108] = l2 as u32;
        words[109] = (l2 >> 32) as u32;
        for s in 0..36 {
            words[110 + s] = mag[s].to_bits();
        }
        for (j, &w) in light_w.iter().enumerate().take(16) {
            words[146 + j] = (w as f32).to_bits();
        }
        Some(GpuEvent { words, geometry, lights, light_w, probe_rays })
    }

    /// Probe rays from a lamp, traced on the GPU (`sel` 0: current scene, 1: previous).
    /// Returns (hit point, albedo luminance) of the probes that hit the geometry.
    pub fn probes(&mut self, l: &crate::scene::Light, sel: u32, seed: u64) -> Vec<(V3, f64)> {
        let mut rng = Rng::new(seed);
        let rays: Vec<(V3, V3)> = (0..PROBES)
            .map(|_| {
                let o = l.center + v3((2.0 * rng.f64() - 1.0) * l.hx, -1e-3, (2.0 * rng.f64() - 1.0) * l.hz);
                (o, cosine_dir(v3(0.0, -1.0, 0.0), &mut rng))
            })
            .collect();
        self.trace_rays(&rays, sel).into_iter().filter_map(|h| h.map(|(p, a)| (p, a.lum()))).collect()
    }

    /// First hit (point, albedo) of arbitrary rays (at most PROBES), on the GPU.
    pub fn trace_rays(&mut self, rays_in: &[(V3, V3)], sel: u32) -> Vec<Option<(V3, V3)>> {
        assert!(rays_in.len() <= PROBES);
        let mut rays = vec![[0.0f32, -1e6, 0.0, 0.0]; PROBES * 2];
        for (i, (o, d)) in rays_in.iter().enumerate() {
            rays[2 * i] = [o.x as f32, o.y as f32, o.z as f32, 0.0];
            rays[2 * i + 1] = [d.x as f32, d.y as f32, d.z as f32, 0.0];
        }
        let q = &self.gpu.queue;
        q.write_buffer(&self.probe_buf, 0, bytemuck::cast_slice(&rays));
        let fu = self.uniform(0, 0, 0, 0, sel, 0, 0);
        q.write_buffer(&self.bufs[0], 0, bytemuck::bytes_of(&fu));
        let dev = &self.gpu.device;
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipes["probe_trace"]);
            pass.set_bind_group(0, &self.binds[self.bind_idx], &[]);
            pass.dispatch_workgroups((PROBES as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.probe_buf, 0, &self.probe_read, 0, (PROBES * 32) as u64);
        q.submit(Some(enc.finish()));
        self.probe_read.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("lecture sondes"));
        dev.poll(wgpu::Maintain::Wait);
        let v: Vec<[f32; 4]> = bytemuck::cast_slice(&self.probe_read.slice(..).get_mapped_range()).to_vec();
        self.probe_read.unmap();
        (0..rays_in.len())
            .map(|i| {
                let a = v[2 * i + 1];
                (v[2 * i][3] > 0.5).then(|| {
                    (v3(v[2 * i][0] as f64, v[2 * i][1] as f64, v[2 * i][2] as f64), v3(a[0] as f64, a[1] as f64, a[2] as f64))
                })
            })
            .collect()
    }

    /// Reference image (triangles): `spp` paths per pixel, without a cache.
    pub fn reference_tri(&mut self, scene: &TriScene, spp: u32) -> (Vec<V3>, Vec<u8>) {
        let accel = self.accel.as_mut().expect("triangle scene");
        accel.build_tlas(self.gpu, scene, self.bind_idx);
        let l = lights_to_gpu(&scene.lights, &[]);
        let q = &self.gpu.queue;
        let dev = &self.gpu.device;
        q.write_buffer(&self.bufs[2], 0, bytemuck::cast_slice(&l));
        let len = self.view.len() as u32;
        let chunk = 64u32;
        for pass in 0..spp.div_ceil(chunk) {
            // ref_reg: reference with the game mode's path regularisation
            let fu = self.uniform(0, l.len(), pass, if self.ref_reg { F_GAME } else { 0 }, 0, 0, chunk.min(spp - pass * chunk));
            q.write_buffer(&self.bufs[0], 0, bytemuck::bytes_of(&fu));
            let mut enc = dev.create_command_encoder(&Default::default());
            if pass == 0 {
                enc.clear_buffer(&self.bufs[13], 0, None);
            }
            for name in if pass == 0 { vec!["gbuffer", "reference_acc"] } else { vec!["reference_acc"] } {
                let mut p = enc.begin_compute_pass(&Default::default());
                p.set_pipeline(&self.pipes[name]);
                p.set_bind_group(0, &self.binds[self.bind_idx], &[]);
                p.dispatch_workgroups(len.div_ceil(64), 1, 1);
            }
            q.submit(Some(enc.finish()));
            dev.poll(wgpu::Maintain::Wait);
        }
        // kind of pixels: display pass on an empty cache (lamps: emission)
        let fu = self.uniform(0, l.len(), 0, 0, 0, 0, 0);
        q.write_buffer(&self.bufs[0], 0, bytemuck::bytes_of(&fu));
        let mut enc = dev.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.bufs[13], 0, &self.read_img, 0, len as u64 * 16);
        q.submit(Some(enc.finish()));
        self.read_img.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("lecture"));
        dev.poll(wgpu::Maintain::Wait);
        let img: Vec<f32> = bytemuck::cast_slice(&self.read_img.slice(..).get_mapped_range()).to_vec();
        self.read_img.unmap();
        let gb = self.read_gbuffer_kinds();
        let px = (0..len as usize)
            .map(|k| match gb[k] {
                2 => v3(img[4 * k] as f64, img[4 * k + 1] as f64, img[4 * k + 2] as f64) / spp as f64,
                _ => V3::default(),
            })
            .collect();
        (px, gb)
    }

    /// Kind of the current G-buffer's pixels (0 empty, 1 lamp, 2 surface).
    fn read_gbuffer_kinds(&self) -> Vec<u8> {
        let dev = &self.gpu.device;
        let len = self.view.len();
        let rb = dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (len * 48) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.bufs[5], 0, &rb, 0, (len * 48) as u64);
        self.gpu.queue.submit(Some(enc.finish()));
        rb.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("lecture"));
        dev.poll(wgpu::Maintain::Wait);
        let v: Vec<u32> = bytemuck::cast_slice(&rb.slice(..).get_mapped_range()).to_vec();
        (0..len).map(|k| v[12 * k + 3] as u8).collect()
    }

    fn run_frame(&mut self, n_boxes: usize, l: &[GLight], lp: &[GLight], ev: Option<&GpuEvent>, t: u32) -> GpuFrame {
        let q = &self.gpu.queue;
        let len = self.view.len() as u32;
        let mut flags = if self.sharc { F_SHARC } else { 0 };
        let mut ev_lights = 0;
        if self.amortize {
            flags |= F_AMORTIZE;
        }
        if self.s2_hist {
            flags |= F_S2HIST;
        }
        if self.zero_cap {
            flags |= F_ZCAP;
        }
        if self.resample {
            flags |= F_RESAMPLE;
        }
        if self.delta_resample {
            flags |= F_DRS;
        }
        if self.reuse {
            flags |= F_REUSE;
        }
        if self.demod {
            flags |= F_DEMOD;
        }
        if self.seed_levels {
            flags |= F_SEED;
        }
        if self.support {
            flags |= F_SUPPORT;
        }
        if self.multi {
            flags |= F_MULTI;
        }
        if self.game {
            // F_INDIRECT ("indirect only" cells) is still available but no longer used:
            // it recomputed the direct lighting at the read-back vertex, noisily (see JOURNAL v14)
            flags |= if self.indirect_only { F_INDIRECT } else { F_DSPLIT };
            flags |= F_GAME;
        }
        if let Some(e) = ev {
            if !self.sharc {
                flags |= F_EVENT;
                if e.geometry {
                    flags |= F_GEOMETRY;
                }
                ev_lights = e.lights;
                q.write_buffer(&self.bufs[14], 0, bytemuck::cast_slice(&e.words));
            }
        } else if let Some(p) = &self.pending {
            // no new event: the amortised lamp event goes on (words already on the GPU)
            flags |= F_PENDING;
            ev_lights = p.lights;
        }
        let fu = self.uniform(n_boxes, l.len(), t, flags, 0, ev_lights, 0);
        q.write_buffer(&self.bufs[0], 0, bytemuck::bytes_of(&fu));
        q.write_buffer(&self.bufs[2], 0, bytemuck::cast_slice(l));
        q.write_buffer(&self.bufs[4], 0, bytemuck::cast_slice(lp));

        let mut plan: Vec<(&'static str, Dispatch)> = Vec::new();
        let px = Dispatch::Direct(len.div_ceil(64));
        if self.rebuild {
            plan.push(("rebuild_clear", Dispatch::Direct((self.table_mask + 1).div_ceil(256))));
            plan.push(("rebuild_insert", Dispatch::Direct(self.cell_cap.div_ceil(64))));
            self.rebuild = false;
        }
        let slots = Dispatch::Direct((len * K as u32).div_ceil(256));
        let cells = Dispatch::Indirect(36);
        plan.push(("reset_ctr", Dispatch::Direct((CTR as u32).div_ceil(64))));
        plan.push(("gbuffer", px));
        if self.method == Method::Sharc {
            // Update (sparse pixels) → Resolve (every entry) → cache view or Render
            let d = self.sh.downscale;
            let blocks = self.view.w.div_ceil(d as usize) as u32 * self.view.h.div_ceil(d as usize) as u32;
            plan.push(("sharc_update", Dispatch::Direct(blocks.div_ceil(64))));
            plan.push(("sharc_resolve", Dispatch::Direct(self.sh_cap.div_ceil(256))));
            plan.push((if self.sh.render || self.game { "sharc_render" } else { "sharc_query" }, px));
        } else {
            plan.push(("insert", slots));
            plan.push(("lookup", slots));
            plan.push(("prep_cells", Dispatch::Direct(1)));
            if flags & F_EVENT != 0 {
                plan.push(("event_mark", cells));
                plan.push(("prep_d1", Dispatch::Direct(1)));
                plan.push(("delta_trace1", Dispatch::Indirect(0)));
                if flags & F_GEOMETRY != 0 {
                    plan.push(("delta_more", cells));
                    plan.push(("prep_d2", Dispatch::Direct(1)));
                    plan.push(("delta_trace2", Dispatch::Indirect(12)));
                }
                plan.push(("delta_apply", cells));
            } else if flags & F_PENDING != 0 {
                plan.push(("pending_mark", cells));
                plan.push(("prep_d1", Dispatch::Direct(1)));
                plan.push(("delta_trace1", Dispatch::Indirect(0)));
                plan.push(("delta_apply", cells));
            }
            let vis = Dispatch::Indirect(48);
            if self.sharc {
                plan.push(("sharc_emit", px));
            } else {
                plan.push(("alloc_hist", vis));
                plan.push(("alloc_prep", Dispatch::Direct(1)));
                plan.push(("alloc_scan1", vis));
                plan.push(("alloc_pick1", Dispatch::Direct(1)));
                plan.push(("alloc_scan2", vis));
                plan.push(("alloc_pick2", Dispatch::Direct(1)));
                plan.push(("alloc_scan3", vis));
                plan.push(("alloc_pick3", Dispatch::Direct(1)));
                plan.push(("alloc_emit", vis));
            }
            plan.push(("prep_reg", Dispatch::Direct(1)));
            plan.push(("trace", Dispatch::Indirect(24)));
            plan.push(("prep_vis2", Dispatch::Direct(1)));
            plan.push(("evict_sweep", Dispatch::Direct(self.cell_cap.div_ceil(8 * 64))));
            if self.game {
                // reflections: feed off-screen cells, resolved in the same frame
                plan.push(("krc_refl", px));
                plan.push(("prep_vis2", Dispatch::Direct(1)));
            }
            plan.push(("resolve", vis));
            plan.push((if self.game { "krc_final" } else { "display" }, px));
        }

        if self.game && !self.denoise {
            plan.push(("game_combine", px));
        }
        if self.game && self.denoise {
            let passes: &[&'static str] = if self.dn_wide {
                &[
                    "dn_temporal",
                    "dn_spec",
                    "dn_store",
                    "dn_spec_store",
                    "dn_atrous1",
                    "dn_atrous2",
                    "dn_atrous3b",
                    "dn_atrous4",
                    "dn_atrous5",
                ]
            } else {
                &["dn_temporal", "dn_spec", "dn_store", "dn_spec_store", "dn_atrous1", "dn_atrous2", "dn_atrous3"]
            };
            for name in passes {
                plan.push((name, px));
            }
        }
        let dev = &self.gpu.device;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("krc frame") });
        for (i, (name, d)) in plan.iter().enumerate() {
            let tw = wgpu::ComputePassTimestampWrites {
                query_set: &self.ts,
                beginning_of_pass_write_index: Some(2 * i as u32),
                end_of_pass_write_index: Some(2 * i as u32 + 1),
            };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(name), timestamp_writes: Some(tw) });
            pass.set_pipeline(&self.pipes[name]);
            pass.set_bind_group(0, &self.binds[self.bind_idx], &[]);
            match *d {
                Dispatch::Direct(n) => pass.dispatch_workgroups(n, 1, 1),
                Dispatch::Indirect(off) => pass.dispatch_workgroups_indirect(&self.bufs[16], off),
            }
        }
        let np = plan.len() as u32;
        enc.resolve_query_set(&self.ts, 0..2 * np, &self.ts_resolve, 0);
        enc.copy_buffer_to_buffer(&self.ts_resolve, 0, &self.ts_read, 0, np as u64 * 16);
        if self.readback {
            enc.copy_buffer_to_buffer(&self.bufs[13], 0, &self.read_img, 0, len as u64 * 16);
        }
        enc.copy_buffer_to_buffer(&self.bufs[9], 0, &self.read_ctr, 0, (CTR * 4) as u64);
        q.submit(Some(enc.finish()));
        let reads: Vec<&wgpu::Buffer> =
            if self.readback { vec![&self.read_img, &self.read_ctr, &self.ts_read] } else { vec![&self.read_ctr, &self.ts_read] };
        for b in &reads {
            b.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("lecture GPU"));
        }
        dev.poll(wgpu::Maintain::Wait);
        let img: Vec<f32> =
            if self.readback { bytemuck::cast_slice(&self.read_img.slice(..).get_mapped_range()).to_vec() } else { Vec::new() };
        let ctr: Vec<u32> = bytemuck::cast_slice(&self.read_ctr.slice(..).get_mapped_range()).to_vec();
        let ts: Vec<u64> = bytemuck::cast_slice(&self.ts_read.slice(..).get_mapped_range()).to_vec();
        for b in &reads {
            b.unmap();
        }
        let ms = self.period_ns as f64 * 1e-6;
        // lamp event with cells left to process: it is kept for the next frame
        self.pending = if ctr[18] > 0 { ev.cloned().or(self.pending.take()) } else { None };
        self.rebuild = ctr[C_TOMB] as u64 * 4 > self.table_mask as u64 + 1;
        self.clock += 1;
        self.sh_cam_prev = self.view.eye;
        self.prev_view = self.view;
        GpuFrame {
            image: (0..img.len() / 4).map(|k| v3(img[4 * k] as f64, img[4 * k + 1] as f64, img[4 * k + 2] as f64)).collect(),
            kinds: (0..img.len() / 4).map(|k| img[4 * k + 3] as u8).collect(),
            probe_rays: ev.map_or(0, |e| e.probe_rays),
            cells: ctr[C_CELLS],
            visible: ctr[C_VISIBLE],
            surf: ctr[C_SURF],
            regular_paths: ctr[C_TASKS] - ctr[C_REG0],
            suspects: ctr[C_SUSPECTS],
            delta_paths: ctr[C_DPATHS],
            delta_old: ctr[C_DOLD],
            delta_zero: ctr[C_DZERO],
            hard: ctr[C_HARD],
            safety: ctr[C_SAFETY],
            overflow: ctr[C_OVERFLOW],
            pending: ctr[18],
            max_n: f32::from_bits(ctr[17]),
            debt: ctr[9],
            evicted: ctr[C_EVICT],
            free: ctr[C_FREE],
            _tombs: ctr[C_TOMB],
            segs: ctr[25],
            rs_hits: ctr[26],
            passes: plan
                .iter()
                .enumerate()
                .map(|(i, (name, _))| (*name, ts[2 * i + 1].saturating_sub(ts[2 * i]) as f64 * ms))
                .collect(),
        }
    }
}

/// `krc gpubench`: GPU time per frame of the full pipeline, increasing the resolution.
pub fn bench(gpu: &Gpu) {
    use crate::methods::event;
    use crate::scene::{build, state_at, FRAMES};
    let mut csv = String::from(
        "scale,width,height,method,phase,mean_ms,max_ms,paths_per_frame
",
    );
    for s in [1usize, 2, 5, 10] {
        let world = World::scaled(2, s);
        for sharc in [false, true] {
            let name = if sharc { "gpu_sharc" } else { "gpu_krc" };
            let mut krc = GpuKrc::new(gpu, world, sharc, 1);
            krc.readback = false;
            let mut prev_state = state_at(0);
            let mut prev_scene = build(&prev_state, 2);
            let mut per_frame: Vec<(u32, f64, u32)> = Vec::new();
            let mut passes: Vec<(&'static str, f64, usize)> = Vec::new();
            // a warm-up sequence (GPU clocks, caches), then the measured sequence
            for t in 0..2 * FRAMES {
                let warm = t < FRAMES;
                let t = t % FRAMES;
                let state = state_at(t);
                let scene = build(&state, 2);
                let ev = event(&prev_state, &state, &prev_scene, &scene, t);
                let f = krc.frame(&scene, &prev_scene, ev.as_ref(), t);
                assert_eq!(f.overflow, 0, "table full ({} cells)", f.cells);
                prev_state = state;
                prev_scene = scene;
                if warm {
                    continue;
                }
                per_frame.push((t, f.gpu_ms(), f.regular_paths + f.delta_paths));
                for (n, ms) in &f.passes {
                    match passes.iter_mut().find(|p| p.0 == *n) {
                        Some(p) => {
                            p.1 += ms;
                            p.2 += 1;
                        }
                        None => passes.push((n, *ms, 1)),
                    }
                }
            }
            let mut line = format!("{}×{} {name:<9}", world.w, world.h);
            for (phase, a, b) in
                [("static", 20u32, 40u32), ("box", 40, 64), ("lamp_on", 80, 81), ("lamp_off", 100, 101), ("all", 0, 120)]
            {
                let sel: Vec<&(u32, f64, u32)> = per_frame.iter().filter(|p| p.0 >= a && p.0 < b).collect();
                // median: robust to the GPU's clock fluctuations
                let mut v: Vec<f64> = sel.iter().map(|p| p.1).collect();
                v.sort_by(|x, y| x.total_cmp(y));
                let mean = v[v.len() / 2];
                let max = sel.iter().map(|p| p.1).fold(0.0, f64::max);
                let paths = sel.iter().map(|p| p.2 as f64).sum::<f64>() / sel.len() as f64;
                csv += &format!(
                    "{s},{},{},{name},{phase},{mean:.4},{max:.4},{paths:.0}
",
                    world.w, world.h
                );
                line += &format!(" | {phase} {mean:6.3} ms");
            }
            println!("{line}");
            if s == 10 {
                let mut p: Vec<String> = passes.iter().map(|(n, ms, c)| format!("{n} {:.3}", ms / *c as f64)).collect();
                p.sort();
                println!("   passes (ms, mean over the frames where they run): {}", p.join(", "));
            }
        }
    }
    std::fs::create_dir_all("results/gpu").unwrap();
    std::fs::write("results/gpu/bench_pipeline.csv", csv).unwrap();
}

/// `krc drift [cycles]`: the 120-frame sequence repeated; error measured at the same moments
/// of each cycle to detect a drift (f32 stagnation, saturation, debt, table).
pub fn drift(gpu: &Gpu, cycles: u32, refs: &std::collections::HashMap<String, Vec<V3>>) {
    use crate::methods::event;
    use crate::render::{errors, gbuffer, Px};
    use crate::scene::{build, state_at, FRAMES};
    let world = World::new(2);
    let probe_frames = [39u32, 63, 79, 81, 99, 101, 119];
    let mut csv = String::from(
        "method,cycle,frame,relmse,cells,max_n,debt,overflow
",
    );
    for sharc in [false, true] {
        let name = if sharc { "gpu_sharc" } else { "gpu_krc" };
        let mut krc = GpuKrc::new(gpu, world, sharc, 7);
        let mut prev_state = state_at(0);
        let mut prev_scene = build(&prev_state, 2);
        let mut first: Vec<f64> = Vec::new();
        let mut last: Vec<f64> = Vec::new();
        for cycle in 0..cycles {
            let mut row = Vec::new();
            for t in 0..FRAMES {
                let state = state_at(t);
                let scene = build(&state, 2);
                let ev = event(&prev_state, &state, &prev_scene, &scene, t);
                krc.readback = probe_frames.contains(&t);
                let f = krc.frame(&scene, &prev_scene, ev.as_ref(), cycle * FRAMES + t);
                if krc.readback {
                    let gb = gbuffer(&scene, &world);
                    let mut img = f.image;
                    for k in 0..world.len() {
                        if let Px::Light(e) = gb[k] {
                            img[k] = e;
                        }
                    }
                    let (_, rel) = errors(&img, &refs[&state.id()], &gb);
                    csv += &format!(
                        "{name},{cycle},{t},{rel:.6e},{},{},{},{}
",
                        f.cells, f.max_n, f.debt, f.overflow
                    );
                    row.push(rel);
                    if t == 119 && (cycle % 10 == 9 || cycle == 0) {
                        println!(
                            "{name} cycle {cycle:>3}: relMSE at frames {probe_frames:?} = {} | cells {} | max n {:.0} | debt {} | overflows {}",
                            row.iter().map(|r| format!("{r:.2e}")).collect::<Vec<_>>().join(" "),
                            f.cells,
                            f.max_n,
                            f.debt,
                            f.overflow
                        );
                    }
                }
                prev_state = state;
                prev_scene = scene;
            }
            if cycle < 10 {
                first.push(row.iter().sum::<f64>() / row.len() as f64);
            }
            if cycle >= cycles.saturating_sub(10) {
                last.push(row.iter().sum::<f64>() / row.len() as f64);
            }
        }
        let m = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        println!(
            "{name}: mean relMSE at the probed moments, first 10 cycles {:.3e} -> last 10 {:.3e}",
            m(&first),
            m(&last)
        );
    }
    std::fs::create_dir_all("results/gpu").unwrap();
    std::fs::write("results/gpu/drift.csv", csv).unwrap();
}

#[derive(Clone, Copy)]
enum Dispatch {
    Direct(u32),
    /// offset (bytes) of the arguments in the indirect buffer
    Indirect(u64),
}

fn mix_seed(t: u32, j: usize) -> u64 {
    crate::math::mix64((t as u64) << 8 | j as u64)
}

/// Description of an event for the GPU: fine regions, super-regions, magnitudes, lamp weights.
fn event_words(e: &Event) -> Vec<u32> {
    let mut v = vec![0u32; 162];
    for k in 0..WORDS {
        v[2 * k] = e.regions.0[k] as u32;
        v[2 * k + 1] = (e.regions.0[k] >> 32) as u32;
    }
    let l2 = Coarse::from(&e.regions).l2;
    v[108] = l2 as u32;
    v[109] = (l2 >> 32) as u32;
    let mut mag = [0.0f32; SUPER];
    for &(r, m) in &e.mag {
        let s = crate::deps::super_of(r);
        mag[s] = mag[s].max(m as f32);
    }
    for s in 0..SUPER {
        v[110 + s] = mag[s].to_bits();
    }
    for (j, &w) in e.light_w.iter().enumerate().take(16) {
        v[146 + j] = (w as f32).to_bits();
    }
    v
}
