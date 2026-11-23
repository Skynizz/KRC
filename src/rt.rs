//! Triangle scenes and hardware acceleration structures (Vulkan ray query through wgpu).
use crate::gpu::Gpu;
use crate::math::{v3, V3};
use crate::scene::{Aabb, Light};

/// Material: colour (× texture `tex` if ≥ 0), GGX roughness, metal, alpha mask.
/// `rough >= 1` and `metal == 0`: pure Lambertian (the scenes from before the showcase).
#[derive(Clone, Copy, Debug)]
pub struct Mat {
    pub tex: i32,
    pub rough: f32,
    pub metal: f32,
    pub mask: bool,
}

impl Mat {
    pub const DIFFUSE: Mat = Mat { tex: -1, rough: 1.0, metal: 0.0, mask: false };
}

/// Indexed mesh, one material per mesh.
#[derive(Clone)]
pub struct Mesh {
    pub pos: Vec<[f32; 3]>,
    pub idx: Vec<u32>,
    pub albedo: [f32; 3],
    /// texture coordinates (one per vertex)
    pub uv: Vec<[f32; 2]>,
    pub mat: Mat,
}

/// Colour textures, all brought to `size`², with their mipmaps: `mips[l]` holds the
/// layers one after the other (RGBA8 sRGB).
pub struct TexArray {
    pub size: u32,
    pub layers: u32,
    pub mips: Vec<Vec<u8>>,
}

impl Mesh {
    fn new(albedo: [f32; 3]) -> Mesh {
        Mesh { pos: Vec::new(), idx: Vec::new(), albedo, uv: Vec::new(), mat: Mat::DIFFUSE }
    }
    /// Quad a, b, c, d (in outline order).
    fn quad(&mut self, a: V3, b: V3, c: V3, d: V3) {
        let i = self.pos.len() as u32;
        for p in [a, b, c, d] {
            self.pos.push([p.x as f32, p.y as f32, p.z as f32]);
        }
        self.uv.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        self.idx.extend_from_slice(&[i, i + 1, i + 2, i, i + 2, i + 3]);
    }
    /// UV sphere of radius r, centred at the origin.
    pub fn sphere(albedo: [f32; 3], r: f64, mat: Mat) -> Mesh {
        let mut m = Mesh::new(albedo);
        m.mat = mat;
        let (nu, nv) = (384u32, 192u32); // fine: a mirror sphere's reflection must not show facets
        for j in 0..=nv {
            let th = std::f64::consts::PI * j as f64 / nv as f64;
            for i in 0..=nu {
                let ph = 2.0 * std::f64::consts::PI * i as f64 / nu as f64;
                m.pos.push([(r * th.sin() * ph.cos()) as f32, (r * th.cos()) as f32, (r * th.sin() * ph.sin()) as f32]);
                m.uv.push([i as f32 / nu as f32, j as f32 / nv as f32]);
            }
        }
        for j in 0..nv {
            for i in 0..nu {
                let a = j * (nu + 1) + i;
                let b = a + nu + 1;
                m.idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        m
    }
    /// Axis-aligned box (6 faces).
    pub fn cuboid(albedo: [f32; 3], b: &Aabb) -> Mesh {
        let mut m = Mesh::new(albedo);
        let (l, h) = (b.min, b.max);
        let p = |x: f64, y: f64, z: f64| v3(x, y, z);
        m.quad(p(l.x, l.y, l.z), p(h.x, l.y, l.z), p(h.x, l.y, h.z), p(l.x, l.y, h.z));
        m.quad(p(l.x, h.y, l.z), p(l.x, h.y, h.z), p(h.x, h.y, h.z), p(h.x, h.y, l.z));
        m.quad(p(l.x, l.y, l.z), p(l.x, h.y, l.z), p(h.x, h.y, l.z), p(h.x, l.y, l.z));
        m.quad(p(l.x, l.y, h.z), p(h.x, l.y, h.z), p(h.x, h.y, h.z), p(l.x, h.y, h.z));
        m.quad(p(l.x, l.y, l.z), p(l.x, l.y, h.z), p(l.x, h.y, h.z), p(l.x, h.y, l.z));
        m.quad(p(h.x, l.y, l.z), p(h.x, h.y, l.z), p(h.x, h.y, h.z), p(h.x, l.y, h.z));
        m
    }
}

/// Instance: mesh + translation (moving objects are displaced instances).
#[derive(Clone, Copy)]
pub struct Instance {
    pub mesh: usize,
    pub offset: V3,
}

pub struct TriScene {
    pub meshes: Vec<Mesh>,
    pub instances: Vec<Instance>,
    pub lights: Vec<Light>,
    pub bounds: Aabb,
    /// colour textures (showcase); None: no texture
    pub tex: Option<&'static TexArray>,
}

impl TriScene {
    pub fn instance_aabb(&self, i: usize) -> Aabb {
        let inst = self.instances[i];
        let m = &self.meshes[inst.mesh];
        let mut b = Aabb { min: V3::splat(f64::INFINITY), max: V3::splat(f64::NEG_INFINITY) };
        for p in &m.pos {
            let q = v3(p[0] as f64, p[1] as f64, p[2] as f64) + inst.offset;
            b = Aabb { min: b.min.min(q), max: b.max.max(q) };
        }
        b
    }
    pub fn triangles(&self) -> usize {
        self.instances.iter().map(|i| self.meshes[i.mesh].idx.len() / 3).sum()
    }
}

/// Cornell box (5.5 m), fixed tall box, moving short box (instance 1), ceiling light (lamp 0)
/// and a second warm lamp at the front left (lamp 1): the same events as the rooms
/// sequence (box that moves, lamp that switches on, lamp that switches off).
pub fn cornell(step: u32, light_on: bool, lamp2_on: bool) -> TriScene {
    let white = [0.73, 0.73, 0.73];
    let red = [0.65, 0.05, 0.05];
    let green = [0.12, 0.45, 0.15];
    let s = 5.5;
    let mut room = Mesh::new(white);
    let p = |x: f64, y: f64, z: f64| v3(x, y, z);
    room.quad(p(0.0, 0.0, 0.0), p(0.0, 0.0, s), p(s, 0.0, s), p(s, 0.0, 0.0)); // sol
    room.quad(p(0.0, s, 0.0), p(s, s, 0.0), p(s, s, s), p(0.0, s, s)); // plafond
    room.quad(p(0.0, 0.0, 0.0), p(s, 0.0, 0.0), p(s, s, 0.0), p(0.0, s, 0.0)); // fond
    let mut left = Mesh::new(red);
    left.quad(p(0.0, 0.0, 0.0), p(0.0, s, 0.0), p(0.0, s, s), p(0.0, 0.0, s));
    let mut right = Mesh::new(green);
    right.quad(p(s, 0.0, 0.0), p(s, 0.0, s), p(s, s, s), p(s, s, 0.0));
    let tall = Mesh::cuboid(white, &Aabb { min: v3(0.9, 0.0, 1.0), max: v3(2.4, 3.3, 2.5) });
    let short = Mesh::cuboid(white, &Aabb { min: v3(0.0, 0.0, 0.0), max: v3(1.5, 1.65, 1.5) });
    let meshes = vec![room, left, right, tall, short];
    let shift = v3(3.3 - step as f64 * 0.1, 0.0, 2.6);
    let mut instances: Vec<Instance> = (0..4).map(|m| Instance { mesh: m, offset: V3::default() }).collect();
    instances.insert(1, Instance { mesh: 4, offset: shift }); // instance 1 = moving box
    let lights = vec![
        Light { center: v3(2.75, s - 0.01, 2.75), hx: 0.65, hz: 0.55, emission: V3::splat(15.0), on: light_on },
        Light { center: v3(1.0, s - 0.01, 4.4), hx: 0.35, hz: 0.35, emission: v3(40.0, 30.0, 18.0), on: lamp2_on },
    ];
    TriScene { meshes, instances, lights, bounds: Aabb { min: V3::splat(-0.1), max: V3::splat(s + 0.1) }, tex: None }
}

/// GPU acceleration structures of a scene: one BLAS per mesh, one TLAS of instances.
pub struct Accel {
    /// two alternating TLAS: the current frame's and the previous one's
    pub tlas: [wgpu::TlasPackage; 2],
    pub blas: Vec<wgpu::Blas>,
    /// vertices (aligned vec4) and indices of every mesh, for the normals in the shader
    pub verts: wgpu::Buffer,
    pub idx: wgpu::Buffer,
    /// per mesh: vertex offset, index offset, material
    pub mesh_table: wgpu::Buffer,
    /// texture coordinates of every vertex
    pub uvs: wgpu::Buffer,
    /// array of colour textures (1×1 white if the scene has none)
    pub tex_view: wgpu::TextureView,
    vbufs: Vec<wgpu::Buffer>,
    ibufs: Vec<wgpu::Buffer>,
    sizes: Vec<wgpu::BlasTriangleGeometrySizeDescriptor>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GMesh {
    pub first_vertex: u32,
    pub first_index: u32,
    pub tex: i32,
    /// bit 0 : masque alpha
    pub flags: u32,
    pub albedo: [f32; 3],
    pub rough: f32,
    pub metal: f32,
    pub pad: [f32; 3],
}

/// View of a texture array on the GPU (mipmaps included); 1×1 white if `tex` is None.
pub fn tex_view(gpu: &Gpu, tex: Option<&TexArray>) -> wgpu::TextureView {
    let white = TexArray { size: 1, layers: 1, mips: vec![vec![255; 4]] };
    let t = tex.unwrap_or(&white);
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("textures"),
        size: wgpu::Extent3d { width: t.size, height: t.size, depth_or_array_layers: t.layers },
        mip_level_count: t.mips.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (l, data) in t.mips.iter().enumerate() {
        let s = (t.size >> l).max(1);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: l as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * s), rows_per_image: Some(s) },
            wgpu::Extent3d { width: s, height: s, depth_or_array_layers: t.layers },
        );
    }
    texture
        .create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
}

fn transform(o: V3) -> [f32; 12] {
    [1.0, 0.0, 0.0, o.x as f32, 0.0, 1.0, 0.0, o.y as f32, 0.0, 0.0, 1.0, o.z as f32]
}

impl Accel {
    pub fn new(gpu: &Gpu, scene: &TriScene) -> Accel {
        use wgpu::util::DeviceExt;
        let dev = &gpu.device;
        let mut vbufs = Vec::new();
        let mut ibufs = Vec::new();
        let mut sizes = Vec::new();
        let mut blas = Vec::new();
        let (mut all_v, mut all_i, mut table) = (Vec::<[f32; 4]>::new(), Vec::<u32>::new(), Vec::new());
        let mut all_uv: Vec<[f32; 2]> = Vec::new();
        for m in &scene.meshes {
            table.push(GMesh {
                first_vertex: all_v.len() as u32,
                first_index: all_i.len() as u32,
                tex: m.mat.tex,
                flags: m.mat.mask as u32,
                albedo: m.albedo,
                rough: m.mat.rough,
                metal: m.mat.metal,
                pad: [0.0; 3],
            });
            all_v.extend(m.pos.iter().map(|p| [p[0], p[1], p[2], 0.0]));
            assert!(m.uv.is_empty() || m.uv.len() == m.pos.len());
            all_uv.extend((0..m.pos.len()).map(|k| m.uv.get(k).copied().unwrap_or([0.0; 2])));
            all_i.extend_from_slice(&m.idx);
            vbufs.push(dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blas v"),
                contents: bytemuck::cast_slice(&m.pos),
                usage: wgpu::BufferUsages::BLAS_INPUT,
            }));
            ibufs.push(dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blas i"),
                contents: bytemuck::cast_slice(&m.idx),
                usage: wgpu::BufferUsages::BLAS_INPUT,
            }));
            let size = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: m.pos.len() as u32,
                index_format: Some(wgpu::IndexFormat::Uint32),
                index_count: Some(m.idx.len() as u32),
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };
            blas.push(dev.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some("blas"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
            ));
            sizes.push(size);
        }
        let tlas = || {
            wgpu::TlasPackage::new(dev.create_tlas(&wgpu::CreateTlasDescriptor {
                label: Some("tlas"),
                max_instances: scene.instances.len() as u32,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            }))
        };
        let st = wgpu::BufferUsages::STORAGE;
        let mut a = Accel {
            tlas: [tlas(), tlas()],
            verts: dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("verts"),
                contents: bytemuck::cast_slice(&all_v),
                usage: st,
            }),
            idx: dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("idx"),
                contents: bytemuck::cast_slice(&all_i),
                usage: st,
            }),
            mesh_table: dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("meshes"),
                contents: bytemuck::cast_slice(&table),
                usage: st,
            }),
            uvs: dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uvs"),
                contents: bytemuck::cast_slice(&all_uv),
                usage: st,
            }),
            tex_view: tex_view(gpu, scene.tex),
            blas,
            vbufs,
            ibufs,
            sizes,
        };
        a.build(gpu, scene, 0, true);
        a.build(gpu, scene, 1, false);
        a
    }

    /// Rebuilds TLAS `k` with the instances' current positions.
    pub fn build_tlas(&mut self, gpu: &Gpu, scene: &TriScene, k: usize) {
        self.build(gpu, scene, k, false);
    }

    fn build(&mut self, gpu: &Gpu, scene: &TriScene, k: usize, with_blas: bool) {
        for (i, inst) in scene.instances.iter().enumerate() {
            self.tlas[k][i] =
                Some(wgpu::TlasInstance::new(&self.blas[inst.mesh], transform(inst.offset), inst.mesh as u32, 0xff));
        }
        let entries: Vec<wgpu::BlasBuildEntry> = if with_blas {
            (0..self.blas.len())
                .map(|m| wgpu::BlasBuildEntry {
                    blas: &self.blas[m],
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                        size: &self.sizes[m],
                        vertex_buffer: &self.vbufs[m],
                        first_vertex: 0,
                        vertex_stride: 12,
                        index_buffer: Some(&self.ibufs[m]),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    }]),
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("as build") });
        enc.build_acceleration_structures(entries.iter(), std::iter::once(&self.tlas[k]));
        gpu.queue.submit(Some(enc.finish()));
    }
}

/// `krc rtsmoke`: normals of a Cornell box seen through hardware ray query (API validation).
pub fn smoke(gpu: &Gpu) {
    use wgpu::util::DeviceExt;
    let scene = cornell(0, true, false);
    let accel = Accel::new(gpu, &scene);
    let (w, h) = (512u32, 512u32);
    let dev = &gpu.device;
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rtsmoke"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct GMesh { first_vertex: u32, first_index: u32, tex: i32, flags: u32, albedo: vec3<f32>, rough: f32, metal: f32, p0: f32, p1: f32, p2: f32 };
@group(0) @binding(0) var acc: acceleration_structure;
@group(0) @binding(1) var<storage, read> verts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read> meshes: array<GMesh>;
@group(0) @binding(4) var<storage, read_write> img: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 512u || gid.y >= 512u) { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / 512.0 * 2.0 - 1.0;
    let o = vec3<f32>(2.75, 2.75, 13.0);
    let d = normalize(vec3<f32>(uv.x * 0.36, -uv.y * 0.36, -1.0));
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xffu, 0.001, 1e30, o, d));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    var c = vec3<f32>(0.0);
    if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
        let m = meshes[hit.instance_custom_index];
        let t = m.first_index + 3u * hit.primitive_index;
        let a = verts[m.first_vertex + idx[t]].xyz;
        let b = verts[m.first_vertex + idx[t + 1u]].xyz;
        let e = verts[m.first_vertex + idx[t + 2u]].xyz;
        var n = normalize((hit.object_to_world * vec4<f32>(cross(b - a, e - a), 0.0)).xyz);
        if (dot(n, d) > 0.0) { n = -n; }
        c = m.albedo * (0.35 + 0.65 * max(dot(n, normalize(vec3<f32>(0.3, 0.8, 0.5))), 0.0));
    }
    img[gid.y * 512u + gid.x] = vec4<f32>(c, 1.0);
}
"#
            .into(),
        ),
    });
    let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rtsmoke"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let img = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (w * h * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (w * h * 16) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: accel.tlas[0].as_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: accel.verts.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: accel.idx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: accel.mesh_table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: img.as_entire_binding() },
        ],
    });
    let _ = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &[0u8; 4],
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut enc = dev.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(w / 8, h / 8, 1);
    }
    enc.copy_buffer_to_buffer(&img, 0, &read, 0, (w * h * 16) as u64);
    gpu.queue.submit(Some(enc.finish()));
    read.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    dev.poll(wgpu::Maintain::Wait);
    let data: Vec<f32> = bytemuck::cast_slice(&read.slice(..).get_mapped_range()).to_vec();
    let px: Vec<V3> =
        (0..(w * h) as usize).map(|k| v3(data[4 * k] as f64, data[4 * k + 1] as f64, data[4 * k + 2] as f64)).collect();
    std::fs::create_dir_all("results/rt").unwrap();
    crate::image::save("results/rt/smoke_cornell.png", w as usize, h as usize, &px);
    println!("ray query OK: {} triangles, image results/rt/smoke_cornell.png", scene.triangles());
}

/// Geometry block of the shader for triangles: hardware ray query on the current TLAS
/// (binding 1) or the previous one (binding 3); triangle normal, mesh albedo.
pub const TRI_GEOMETRY: &str = r#"// <geometry> triangles: hardware ray query
struct GMesh { first_vertex: u32, first_index: u32, tex: i32, flags: u32, albedo: vec3<f32>, rough: f32, metal: f32, p0: f32, p1: f32, p2: f32 };
@group(0) @binding(1) var acc_cur: acceleration_structure;
@group(0) @binding(3) var acc_prev: acceleration_structure;
@group(0) @binding(18) var<storage, read> verts: array<vec4<f32>>;
@group(0) @binding(19) var<storage, read> tidx: array<u32>;
@group(0) @binding(20) var<storage, read> meshes: array<GMesh>;
@group(0) @binding(32) var tex_arr: texture_2d_array<f32>;
@group(0) @binding(33) var tex_samp: sampler;
@group(0) @binding(34) var<storage, read> uvs: array<vec2<f32>>;

// base colour (and alpha) at the hit point: texture × factor; mip level from the
// distance travelled by the ray
fn surface_color(r: RayIntersection) -> vec4<f32> {
    let m = meshes[r.instance_custom_index];
    if (m.tex < 0) { return vec4<f32>(m.albedo, 1.0); }
    let ti = m.first_index + 3u * r.primitive_index;
    let b = r.barycentrics;
    let uv = uvs[m.first_vertex + tidx[ti]] * (1.0 - b.x - b.y) + uvs[m.first_vertex + tidx[ti + 1u]] * b.x
        + uvs[m.first_vertex + tidx[ti + 2u]] * b.y;
    let lod = clamp(log2(max(r.t * fr.px_angle * 256.0, 1.0)), 0.0, 9.0);
    let c = textureSampleLevel(tex_arr, tex_samp, uv, m.tex, lod);
    return vec4<f32>(c.rgb * m.albedo, c.a);
}

// closest hit, going through the transparent texels of alpha-masked meshes
fn rq_closest(o: vec3<f32>, d: vec3<f32>, tmax: f32) -> RayIntersection {
    var t0 = 0.0;
    var r = rq_trace(o, d, tmax, 0u);
    for (var k = 0; k < 8; k++) {
        if (r.kind == RAY_QUERY_INTERSECTION_NONE) { break; }
        let m = meshes[r.instance_custom_index];
        if ((m.flags & 1u) == 0u || surface_color(r).a >= 0.5) { break; }
        t0 += r.t + 1e-4;
        if (t0 >= tmax) {
            r.kind = RAY_QUERY_INTERSECTION_NONE;
            break;
        }
        r = rq_trace(o + d * t0, d, tmax - t0, 0u);
    }
    r.t += t0;
    return r;
}

fn rq_trace(o: vec3<f32>, d: vec3<f32>, tmax: f32, flags: u32) -> RayIntersection {
    var q: ray_query;
    let desc = RayDesc(flags, 0xffu, EPS, tmax, o, d);
    if (sel == 0u) { rayQueryInitialize(&q, acc_cur, desc); } else { rayQueryInitialize(&q, acc_prev, desc); }
    rayQueryProceed(&q);
    return rayQueryGetCommittedIntersection(&q);
}

fn scene_hit(o: vec3<f32>, d: vec3<f32>) -> Hit {
    var h: Hit;
    h.found = false;
    h.t = 1e30;
    h.light = -1;
    h.rough = 1.0;
    h.metal = 0.0;
    let r = rq_closest(o, d, 1e30);
    if (r.kind != RAY_QUERY_INTERSECTION_NONE) {
        let m = meshes[r.instance_custom_index];
        let ti = m.first_index + 3u * r.primitive_index;
        let a = verts[m.first_vertex + tidx[ti]].xyz;
        let b = verts[m.first_vertex + tidx[ti + 1u]].xyz;
        let c = verts[m.first_vertex + tidx[ti + 2u]].xyz;
        var n = normalize((r.object_to_world * vec4<f32>(cross(b - a, c - a), 0.0)).xyz);
        if (dot(n, d) > 0.0) { n = -n; }
        h.found = true;
        h.t = r.t;
        h.n = n;
        h.albedo = surface_color(r).rgb;
        h.rough = m.rough;
        h.metal = m.metal;
    }
    return h;
}

fn occluder(o: vec3<f32>, d: vec3<f32>, tmax: f32) -> Occ {
    var r: Occ;
    // any first hit (fast); if it falls on an alpha-masked mesh, the closest one is searched
    // again going through transparent texels
    let i = rq_trace(o, d, tmax, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    var found = i.kind != RAY_QUERY_INTERSECTION_NONE;
    var t = i.t;
    if (found && (meshes[i.instance_custom_index].flags & 1u) != 0u) {
        let c = rq_closest(o, d, tmax);
        found = c.kind != RAY_QUERY_INTERSECTION_NONE;
        t = c.t;
    }
    r.found = found;
    r.t = select(tmax, t, r.found);
    return r;
}
"#;

/// View of the Cornell box (square).
pub fn cornell_view(size: usize) -> crate::gpu_krc::View {
    crate::gpu_krc::View {
        w: size,
        h: size,
        eye: v3(2.75, 2.75, 13.0),
        fwd: v3(0.0, 0.0, -1.0),
        right: v3(1.0, 0.0, 0.0),
        up: v3(0.0, 1.0, 0.0),
        th: 0.36,
        tv: 0.36,
    }
}

pub fn cornell_at(frame: u32) -> TriScene {
    let s = crate::scene::state_at(frame);
    cornell(s.box_step, s.l0_on, s.l2_on)
}

/// A triangle scene for the benches: name, view, scene at each frame of the sequence.
pub struct TriSetup {
    pub name: String,
    pub view: crate::gpu_krc::View,
    pub at: fn(u32) -> TriScene,
    pub ref_spp: u32,
    /// moving camera: (eye, target) at frame t; None = fixed camera `view`
    pub cam: Option<fn(u32) -> (V3, V3)>,
}

impl TriSetup {
    /// View of frame t (same resolution and same field as `view`).
    pub fn view_at(&self, t: u32) -> crate::gpu_krc::View {
        let mut v = self.view;
        if let Some(cam) = self.cam {
            let (eye, target) = cam(t);
            v.eye = eye;
            v.fwd = (target - eye).norm();
            v.right = v.fwd.cross(v3(0.0, 1.0, 0.0)).norm();
            v.up = v.right.cross(v.fwd);
        }
        v
    }

    /// Name of frame t's reference: the scene's state if the camera is fixed, otherwise the frame.
    pub fn ref_key(&self, t: u32) -> String {
        match self.cam {
            Some(_) => format!("f{t:03}"),
            None => crate::scene::state_at(t).id(),
        }
    }
}

pub fn cornell_setup(size: usize) -> TriSetup {
    TriSetup { name: format!("cornell_{size}"), view: cornell_view(size), at: cornell_at, ref_spp: 4096, cam: None }
}

/// Cornell box, moving camera: forward dolly (z 13 → 9) with a sideways sway.
pub fn cornell_cam_setup(size: usize) -> TriSetup {
    fn cam(t: u32) -> (V3, V3) {
        let u = t as f64 / 119.0;
        let a = std::f64::consts::PI * t as f64 / 60.0;
        (v3(2.75 + 1.5 * a.sin(), 2.75 + 0.4 * (2.0 * a).sin(), 13.0 - 4.0 * u), v3(2.75, 2.5, 2.0))
    }
    TriSetup { name: format!("cornell_cam_{size}"), cam: Some(cam), ..cornell_setup(size) }
}

/// relMSE of a frame over surface pixels: raw, and without the worst 0.1 % of
/// pixels (common practice in path guiding: an isolated firefly must not decide alone).
/// Third value: total image luminance / reference, to see a bias relMSE hides
/// (with y² + 0.01 in the denominator, a dark scene rewards a too-dark image).
pub fn frame_err(img: &[V3], kinds: &[u8], r: &[V3]) -> (f64, f64, f64) {
    let mut px: Vec<f64> = Vec::with_capacity(img.len());
    let (mut li, mut lr) = (0.0, 0.0);
    for i in 0..img.len() {
        if kinds[i] != 2 {
            continue;
        }
        li += img[i].lum();
        lr += r[i].lum();
        let mut e = 0.0;
        for c in 0..3 {
            let (x, y) = (img[i].get(c), r[i].get(c));
            e += (x - y) * (x - y) / (y * y + 0.01);
        }
        px.push(e / 3.0);
    }
    let n = px.len().max(1) as f64;
    let raw = px.iter().sum::<f64>() / n;
    let keep = px.len() - px.len() / 1000;
    if keep < px.len() {
        px.select_nth_unstable_by(keep, |a, b| a.total_cmp(b));
    }
    let trim = px[..keep].iter().sum::<f64>() / keep.max(1) as f64;
    (raw, trim, li / lr.max(1e-12))
}

/// `krc cornell|sponza`: the 120-frame sequence (object that moves, lamp switched on, lamp
/// switched off) on a triangle scene, hardware ray tracing, fully GPU pipeline.
pub fn run_tri(gpu: &Gpu, seeds: u64, setup: &TriSetup) {
    use crate::gpu_krc::GpuKrc;
    use crate::grid::Grid;
    use crate::render::{load_raw, save_raw};
    use crate::scene::FRAMES;
    use std::collections::HashMap;
    let view = setup.view;
    let at = setup.at;
    let base = at(0);
    let grid = Grid::fit(&base.bounds);
    eprintln!(
        "grid: regions {:.3} m, {:?} ({} regions), super-regions {:?}",
        grid.region,
        grid.dims,
        grid.dims.iter().product::<i64>(),
        grid.sdims()
    );
    let dir = format!("results/ref_{}", setup.name);
    std::fs::create_dir_all(&dir).unwrap();
    let mut refs: HashMap<String, Vec<V3>> = HashMap::new();
    let mut rk = GpuKrc::with(gpu, view, grid, Some(&base), false, 0);
    for f in 0..FRAMES {
        let id = setup.ref_key(f);
        if refs.contains_key(&id) {
            continue;
        }
        let path = format!("{dir}/{id}.bin");
        let img = load_raw(&path, view.len()).unwrap_or_else(|| {
            let t = std::time::Instant::now();
            rk.set_view(setup.view_at(f));
            let (img, _) = rk.reference_tri(&at(f), setup.ref_spp);
            save_raw(&path, &img);
            crate::image::save(&format!("{dir}/{id}.png"), view.w, view.h, &img);
            eprintln!("reference {id}: {:.1} s", t.elapsed().as_secs_f64());
            img
        });
        refs.insert(id, img);
    }
    let phases: [(&str, u32, u32); 6] = [
        ("all", 0, 120),
        ("static", 30, 40),
        ("box", 40, 64),
        ("after_box", 64, 80),
        ("lamp_on", 80, 100),
        ("lamp_off", 100, 120),
    ];
    let mut summary = String::from("method,phase,relmse_mean,relmse_std,gpu_ms,paths_per_frame,trim_mean,trim_std,lum_ratio\n");
    use crate::gpu_krc::Method;
    // tri_sharc: home-made SHaRC-like cache; tri_sharc_nv: port of NVIDIA's official SHaRC,
    // cache view at the first hit; tri_sharc_nv_render: its Render pass (1 path/px)
    let all = [
        ("tri_krc", Method::Krc, false),
        ("tri_sharc", Method::SharcLike, false),
        ("tri_sharc_nv", Method::Sharc, false),
        ("tri_sharc_nv_render", Method::Sharc, true),
    ];
    let only = std::env::var("KRC_METHODS").ok();
    for (name, method, render) in all {
        if only.as_ref().is_some_and(|m| !m.split(',').any(|x| x == name)) {
            continue;
        }
        let out = format!("results/{}/{name}", setup.name);
        std::fs::create_dir_all(&out).unwrap();
        let mut runs: Vec<Vec<(f64, f64, u32, f64, f64)>> = Vec::new();
        for seed in 0..seeds {
            let mut k = GpuKrc::with_method(gpu, view, grid, Some(&base), method, 1 + seed as u32);
            k.sh.render = render;
            let mut prev = at(0);
            let mut rows = Vec::new();
            let mut csv = String::from(
                "frame,relmse,gpu_ms,paths,suspects,delta_paths,relmse_trim,lum_ratio,cells_live,evicted,overflow\n",
            );
            for t in 0..FRAMES {
                let scene = at(t);
                k.set_view(setup.view_at(t));
                let f = k.frame_tri(&scene, &prev, t);
                if f.overflow > 0 {
                    eprintln!("{name} seed {seed} frame {t}: {} insertions lost (table full)", f.overflow);
                }
                let (rel, trim, lr) = frame_err(&f.image, &f.kinds, &refs[&setup.ref_key(t)]);
                let paths = f.regular_paths + f.delta_paths + f.probe_rays as u32;
                csv += &format!(
                    "{t},{rel:.6e},{:.4},{paths},{},{},{trim:.6e},{lr:.4},{},{},{}\n",
                    f.gpu_ms(),
                    f.suspects,
                    f.delta_paths,
                    f.cells - f.free,
                    f.evicted,
                    f.overflow
                );
                if seed == 0 && crate::methods::SAVE_FRAMES.contains(&t) {
                    crate::image::save(&format!("{out}/f{t:03}.png"), view.w, view.h, &f.image);
                }
                rows.push((rel, f.gpu_ms(), paths, trim, lr));
                prev = scene;
            }
            let file = if seed == 0 { "metrics.csv".to_string() } else { format!("metrics_s{seed}.csv") };
            std::fs::write(format!("{out}/{file}"), csv).unwrap();
            runs.push(rows);
        }
        let mut line = format!("{name:<20}");
        let mut line_t = format!("{:<20}", format!("{name}*"));
        let stat = |v: Vec<f64>| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (m, (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len().max(2) - 1) as f64).sqrt())
        };
        for (phase, a, b) in phases {
            let (m, sd) =
                stat(runs.iter().map(|r| r[a as usize..b as usize].iter().map(|x| x.0).sum::<f64>() / (b - a) as f64).collect());
            let (mt, sdt) =
                stat(runs.iter().map(|r| r[a as usize..b as usize].iter().map(|x| x.3).sum::<f64>() / (b - a) as f64).collect());
            let (lr, _) =
                stat(runs.iter().map(|r| r[a as usize..b as usize].iter().map(|x| x.4).sum::<f64>() / (b - a) as f64).collect());
            let ms: f64 =
                runs.iter().map(|r| r[a as usize..b as usize].iter().map(|x| x.1).sum::<f64>() / (b - a) as f64).sum::<f64>()
                    / runs.len() as f64;
            let paths: f64 = runs
                .iter()
                .map(|r| r[a as usize..b as usize].iter().map(|x| x.2 as f64).sum::<f64>() / (b - a) as f64)
                .sum::<f64>()
                / runs.len() as f64;
            summary += &format!("{name},{phase},{m:.6e},{sd:.6e},{ms:.4},{paths:.0},{mt:.6e},{sdt:.6e},{lr:.4}\n");
            line += &format!(" | {phase} {m:.2e}±{sd:.0e} {ms:.2}ms");
            line_t += &format!(" | {phase} {mt:.2e}±{sdt:.0e} lum {lr:.3}");
        }
        println!("{line}");
        println!("{line_t}");
    }
    let file = if only.is_some() { "summary_partial.csv" } else { "summary.csv" };
    std::fs::write(format!("results/{}/{file}", setup.name), summary).unwrap();
}

/// `krc pareto [size]`: error against GPU time, KRC and SHaRC, for several budgets
/// (paths per pixel per frame). Error: one pass with image readback; time: two
/// passes without readback (warm-up, then measurement; median per frame).
pub fn pareto(gpu: &Gpu, setup: &TriSetup) {
    use crate::gpu_krc::GpuKrc;
    use crate::grid::Grid;
    use crate::render::load_raw;
    use crate::scene::FRAMES;
    let view = setup.view;
    let at = setup.at;
    let base = at(0);
    let grid = Grid::fit(&base.bounds);
    let dir = format!("results/ref_{}", setup.name);
    let refs: std::collections::HashMap<String, Vec<V3>> = (0..FRAMES)
        .map(|f| setup.ref_key(f))
        .map(|id| {
            let img =
                load_raw(&format!("{dir}/{id}.bin"), view.len()).expect("missing references: run the sequence first");
            (id, img)
        })
        .collect();
    let mut csv =
        String::from("method,ppp,relmse,relmse_static,relmse_dynamic,ms_median,ms_mean,ms_p95,trim,trim_static,trim_dynamic\n");
    use crate::gpu_krc::{Method, ShParams};
    // (method, budget "paths per pixel", SHaRC parameters). For the official SHaRC, the budget
    // is its update density: 1 pixel per d×d block, i.e. 1/d² path per pixel.
    let mut variants: Vec<(String, Method, f32, ShParams)> = Vec::new();
    let d = ShParams::default();
    for ppp in [0.25f32, 0.5, 1.0, 2.0] {
        variants.push(("tri_krc".into(), Method::Krc, ppp, d));
    }
    for scale in [50.0f32] {
        for down in [5u32, 3, 2, 1] {
            let name = format!("tri_sharc_nv_s{scale}");
            variants.push((
                name,
                Method::Sharc,
                1.0 / (down * down) as f32,
                ShParams { scene_scale: scale, downscale: down, ..d },
            ));
        }
    }
    let only = std::env::var("KRC_METHODS").ok();
    for (name, method, ppp, sh) in variants {
        if only.as_ref().is_some_and(|m| !m.split(',').any(|x| name.starts_with(x))) {
            continue;
        }
        {
            let name = name.as_str();
            let mut k = GpuKrc::with_method(gpu, view, grid, Some(&base), method, 3);
            k.ppp = ppp;
            k.sh = sh;
            let mut prev = at(0);
            let (mut rel, mut rel_s, mut rel_d) = (Vec::new(), Vec::new(), Vec::new());
            let (mut tr, mut tr_s, mut tr_d) = (Vec::new(), Vec::new(), Vec::new());
            let mut ms = Vec::new();
            for pass in 0..3 {
                k.readback = pass == 0;
                for t in 0..FRAMES {
                    let scene = at(t);
                    k.set_view(setup.view_at(t));
                    let f = k.frame_tri(&scene, &prev, t);
                    prev = scene;
                    if pass == 0 {
                        let (e, et, _) = frame_err(&f.image, &f.kinds, &refs[&setup.ref_key(t)]);
                        rel.push(e);
                        tr.push(et);
                        if (30..40).contains(&t) {
                            rel_s.push(e);
                            tr_s.push(et);
                        }
                        if t >= 40 {
                            rel_d.push(e);
                            tr_d.push(et);
                        }
                    }
                    if pass == 2 {
                        ms.push(f.gpu_ms());
                    }
                }
            }
            let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
            ms.sort_by(|a, b| a.total_cmp(b));
            let (med, p95) = (ms[ms.len() / 2], ms[ms.len() * 95 / 100]);
            println!(
                "{name:<14} {ppp:>5} path/px: relMSE {:.3e} (static {:.2e}, dynamic {:.2e}) | GPU median {med:.2} ms, mean {:.2} ms, p95 {p95:.2} ms",
                mean(&rel),
                mean(&rel_s),
                mean(&rel_d),
                mean(&ms)
            );
            println!(
                "{:<14} {ppp:>5} path/px: relMSE without 0.1 % {:.3e} (static {:.2e}, dynamic {:.2e})",
                format!("{name}*"),
                mean(&tr),
                mean(&tr_s),
                mean(&tr_d)
            );
            csv += &format!(
                "{name},{ppp},{:.6e},{:.6e},{:.6e},{med:.4},{:.4},{p95:.4},{:.6e},{:.6e},{:.6e}\n",
                mean(&rel),
                mean(&rel_s),
                mean(&rel_d),
                mean(&ms),
                mean(&tr),
                mean(&tr_s),
                mean(&tr_d)
            );
        }
    }
    std::fs::create_dir_all(format!("results/{}", setup.name)).unwrap();
    let file = if only.is_some() { "pareto_partial.csv" } else { "pareto.csv" };
    std::fs::write(format!("results/{}/{file}", setup.name), csv).unwrap();
}

/// `krc spikes [size]`: breakdown of the GPU time of event frames (KRC, 1 path/px).
pub fn spikes(gpu: &Gpu, setup: &TriSetup) {
    use crate::gpu_krc::GpuKrc;
    use crate::grid::Grid;
    use crate::scene::FRAMES;
    let view = setup.view;
    let at = setup.at;
    let base = at(0);
    let mut k = GpuKrc::with(gpu, view, Grid::fit(&base.bounds), Some(&base), false, 3);
    k.readback = false;
    k.amortize = std::env::var("KRC_AMORT").is_ok();
    let mut prev = at(0);
    for pass in 0..2 {
        for t in 0..FRAMES {
            let scene = at(t);
            let f = k.frame_tri(&scene, &prev, t);
            prev = scene;
            if pass == 1 && [38u32, 40, 41, 80, 81, 82, 83, 100, 101, 102].contains(&t) {
                let mut p: Vec<String> = f.passes.iter().filter(|x| x.1 > 0.05).map(|(n, ms)| format!("{n} {ms:.2}")).collect();
                p.sort();
                println!(
                    "frame {t:>3}: {:.2} ms | suspects {} | on hold {} | correlated paths {} (retraced {}) | ordinary {} | {}",
                    f.gpu_ms(),
                    f.suspects,
                    f.pending,
                    f.delta_paths,
                    f.delta_old,
                    f.regular_paths,
                    p.join(", ")
                );
            }
        }
    }
}

/// Sponza geometry (Khronos glTF-Sample-Assets, CC-BY: Crytek / Frank Meinl), loaded
/// once: one primitive = one mesh; albedo = base colour (textures ignored),
/// slightly tinted per material to tell surfaces apart.
fn sponza_meshes() -> &'static (Vec<Mesh>, Aabb) {
    static CELL: std::sync::OnceLock<(Vec<Mesh>, Aabb)> = std::sync::OnceLock::new();
    CELL.get_or_init(|| {
        let dir = "assets/sponza";
        let gltf =
            gltf::Gltf::open(format!("{dir}/Sponza.gltf")).expect("assets/sponza/Sponza.gltf: run tools/fetch_sponza.sh");
        let bin = std::fs::read(format!("{dir}/Sponza.bin")).expect("Sponza.bin");
        let scale = 0.008f32;
        let mut meshes = Vec::new();
        let mut b = Aabb { min: V3::splat(f64::INFINITY), max: V3::splat(f64::NEG_INFINITY) };
        for mesh in gltf.meshes() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|_| Some(&bin[..]));
                let pos: Vec<[f32; 3]> =
                    reader.read_positions().unwrap().map(|p| [p[0] * scale, p[1] * scale, p[2] * scale]).collect();
                let idx: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();
                let mat = prim.material();
                let c = mat.pbr_metallic_roughness().base_color_factor();
                let m = mat.index().unwrap_or(0) as u64;
                let h = crate::math::mix64(m ^ 0x5b0);
                let tint = |k: u32| 0.85 + 0.3 * ((h >> (8 * k)) & 0xff) as f32 / 255.0;
                let albedo = [(c[0] * tint(0)).min(0.9), (c[1] * tint(1)).min(0.9), (c[2] * tint(2)).min(0.9)];
                for p in &pos {
                    let q = v3(p[0] as f64, p[1] as f64, p[2] as f64);
                    b = Aabb { min: b.min.min(q), max: b.max.max(q) };
                }
                let uv: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                    Some(t) => t.into_f32().collect(),
                    None => vec![[0.0; 2]; pos.len()],
                };
                meshes.push(Mesh { pos, idx, albedo, uv, mat: Mat::DIFFUSE });
            }
        }
        (meshes, b.expand(0.1))
    })
}

/// Sponza lit by three lamps hanging in the atrium (lamp 0: central, switches off) and a
/// warm fill lamp in the lower gallery (lamp 1, switches on); a box moves on the floor.
pub fn sponza(step: u32, l0_on: bool, l2_on: bool) -> TriScene {
    let (meshes0, bounds) = sponza_meshes();
    let mut meshes = meshes0.clone();
    let mut instances: Vec<Instance> = (0..meshes.len()).map(|m| Instance { mesh: m, offset: V3::default() }).collect();
    meshes.push(Mesh::cuboid([0.25, 0.35, 0.8], &Aabb { min: v3(0.0, 0.0, 0.0), max: v3(1.4, 1.4, 1.4) }));
    let floor = 0.0;
    instances.insert(1, Instance { mesh: meshes.len() - 1, offset: v3(3.0 - step as f64 * 0.5, floor, -0.7) });
    let lamp = |x: f64, on: bool| Light { center: v3(x, 9.0, -0.3), hx: 0.8, hz: 0.8, emission: V3::splat(30.0), on };
    let lights = vec![
        lamp(0.0, l0_on),
        Light { center: v3(-6.0, 2.5, 4.0), hx: 0.4, hz: 0.4, emission: v3(60.0, 45.0, 27.0), on: l2_on },
        lamp(-8.5, true),
        lamp(8.5, true),
    ];
    TriScene { meshes, instances, lights, bounds: *bounds, tex: None }
}

pub fn sponza_at(frame: u32) -> TriScene {
    let s = crate::scene::state_at(frame);
    sponza(s.box_step, s.l0_on, s.l2_on)
}

pub fn sponza_setup(w: usize, h: usize) -> TriSetup {
    // in the atrium (open to the sky for |z| < ~1.9 m), in front of the back wall (x ≈ −12)
    let eye = v3(-10.0, 1.6, -0.3);
    let fwd = (v3(10.0, 4.0, -0.3) - eye).norm();
    let right = fwd.cross(v3(0.0, 1.0, 0.0)).norm();
    let tv = 0.5;
    TriSetup {
        name: format!("sponza_{w}x{h}"),
        view: crate::gpu_krc::View { w, h, eye, fwd, right, up: right.cross(fwd), th: tv * w as f64 / h as f64, tv },
        at: sponza_at,
        // small bright lamps close to the walls: fireflies at 4096 spp
        ref_spp: 16384,
        cam: None,
    }
}

/// Sponza, moving camera: walking through the atrium (9 m in 120 frames, ~4.5 m/s at 60 Hz)
/// looking left and right (yaw ±17°) and swaying sideways.
pub fn sponza_cam_setup(w: usize, h: usize) -> TriSetup {
    fn cam(t: u32) -> (V3, V3) {
        let u = t as f64 / 119.0;
        let tau = 2.0 * std::f64::consts::PI;
        let eye = v3(-10.0 + 9.0 * u, 1.6, -0.3 + 0.8 * (tau * t as f64 / 120.0).sin());
        let yaw = 0.3 * (tau * t as f64 / 90.0).sin();
        (eye, eye + v3(yaw.cos(), 0.12, yaw.sin()) * 10.0)
    }
    TriSetup { name: format!("sponza_cam_{w}x{h}"), cam: Some(cam), ..sponza_setup(w, h) }
}

/// `krc sponzadebug`: what the camera sees and where each lamp's probes land.
pub fn sponza_debug(gpu: &Gpu) {
    use crate::gpu_krc::GpuKrc;
    use crate::grid::Grid;
    let setup = sponza_setup(480, 270);
    let scene = sponza_at(0);
    println!("emprise {:?} -> {:?}, {} triangles", scene.bounds.min, scene.bounds.max, scene.triangles());
    let mut k = GpuKrc::with(gpu, setup.view, Grid::fit(&scene.bounds), Some(&scene), false, 1);
    let f = k.frame_tri(&scene, &scene, 0);
    let n = f.kinds.len() as f64;
    let count = |v: u8| f.kinds.iter().filter(|&&x| x == v).count() as f64 / n;
    println!("pixels: empty {:.2}, lamps {:.2}, surfaces {:.2}", count(0), count(1), count(2));
    let v = setup.view;
    let mut rays = Vec::new();
    for (sx, sy) in [(0.0, 0.0), (-0.9, 0.9), (0.9, 0.9), (-0.9, -0.9), (0.9, -0.9), (0.0, -0.9), (0.0, 0.9)] {
        rays.push((v.eye, (v.fwd + v.right * (sx * v.th) + v.up * (sy * v.tv)).norm()));
    }
    for ((_, d), h) in rays.iter().zip(k.trace_rays(&rays, 0)) {
        match h {
            Some((p, a)) => println!("camera ray {:?}: hit at {:.2} m at {:?}, albedo {:?}", d, (p - v.eye).len(), p, a),
            None => println!("camera ray {:?}: nothing", d),
        }
    }
    for z in [-8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0] {
        let eye = v3(-5.0, 1.8, z);
        let rays = vec![(eye, v3(0.0, 1.0, 0.0)), (eye, v3(0.0, -1.0, 0.0))];
        let h = k.trace_rays(&rays, 0);
        let d: Vec<String> = h.iter().map(|x| x.map_or("∞".to_string(), |(p, _)| format!("{:.2}", (p - eye).len()))).collect();
        println!("x=-5, z={z:>5}: above {}, below {}", d[0], d[1]);
    }
    for ex in [-11.5, -10.5, -9.5, -8.0] {
        let eye = v3(ex, 1.8, -0.3);
        let dirs = [
            v3(1.0, 0.1, 0.0),
            v3(1.0, 0.4, 0.4),
            v3(1.0, 0.4, -0.4),
            v3(1.0, -0.2, 0.4),
            v3(1.0, -0.2, -0.4),
            v3(0.0, 1.0, 0.0),
        ];
        let rays: Vec<(V3, V3)> = dirs.iter().map(|d| (eye, d.norm())).collect();
        let h = k.trace_rays(&rays, 0);
        let dist: Vec<String> = h.iter().map(|x| x.map_or("∞".to_string(), |(p, _)| format!("{:.1}", (p - eye).len()))).collect();
        println!(
            "eye x = {ex}: free distances (forward, up-left, up-right, down-left, down-right, zenith) = {:?}",
            dist
        );
    }
    for (j, l) in scene.lights.iter().enumerate() {
        let hits = k.probes(l, 0, 7);
        let d: Vec<f64> = hits.iter().map(|(p, _)| (*p - l.center).len()).collect();
        let mean = d.iter().sum::<f64>() / d.len().max(1) as f64;
        let near = d.iter().filter(|&&x| x < 0.5).count();
        println!(
            "lamp {j} at {:?}: {} probes hit, mean distance {:.2} m, {} closer than 0.5 m",
            l.center,
            hits.len(),
            mean,
            near
        );
    }
}

/// `krc bias [cornell|sponza]`: mean image luminance / reference (Σ over surface pixels),
/// to separate a bias from noise. (a) final static scene from an empty cache,
/// (b) full sequence, KRC and SHaRC.
pub fn bias(gpu: &Gpu, setup: &TriSetup) {
    use crate::gpu_krc::GpuKrc;
    use crate::grid::Grid;
    use crate::render::load_raw;
    use crate::scene::FRAMES;
    let (view, at) = (setup.view, setup.at);
    let grid = Grid::fit(&at(0).bounds);
    let dir = format!("results/ref_{}", setup.name);
    let refs: std::collections::HashMap<String, Vec<V3>> = (0..FRAMES)
        .map(|f| setup.ref_key(f))
        .map(|id| (id.clone(), load_raw(&format!("{dir}/{id}.bin"), view.len()).expect("missing references")))
        .collect();
    let ratio = |f: &crate::gpu_krc::GpuFrame, r: &[V3]| {
        let (mut a, mut b) = (0.0, 0.0);
        for i in 0..view.len() {
            if f.kinds[i] == 2 {
                a += f.image[i].lum();
                b += r[i].lum();
            }
        }
        a / b
    };
    let fs: u32 = std::env::var("KRC_BIAS_FRAME").ok().and_then(|s| s.parse().ok()).unwrap_or(FRAMES - 1);
    let last = at(fs);
    // (name, sharc, variance cap, n_post, history variance, cap of no-difference cells)
    let variants = [
        ("krc_avant", false, 0.0f32, 64.0f32, false, false),
        ("+plafond_var", false, 1.0, 64.0, false, false),
        ("+zero_cap", false, 1.0, 64.0, false, true),
        ("krc", false, 1.0, 16.0, true, true),
        ("sharc", true, 0.0, 0.0, false, false),
    ];
    for (name, sharc, cap, np, h, z) in variants {
        let mut k = GpuKrc::with(gpu, view, grid, Some(&last), sharc, 7);
        k.set_view(setup.view_at(fs));
        k.rel_cap = cap;
        k.n_post = np;
        k.s2_hist = h;
        k.zero_cap = z;
        let r = &refs[&setup.ref_key(fs)];
        let mut line = format!("(a) {name:<13} static, final state:");
        for t in 0..200 {
            let f = k.frame_tri(&last, &last, t);
            if [0, 4, 9, 19, 39, 59, 99, 199].contains(&t) {
                line += &format!(" f{t} {:.3}", ratio(&f, r));
            }
        }
        println!("{line}");
    }
    for (name, sharc, cap, np, h, z) in variants {
        let mut k = GpuKrc::with(gpu, view, grid, Some(&at(0)), sharc, 7);
        k.rel_cap = cap;
        k.n_post = np;
        k.s2_hist = h;
        k.zero_cap = z;
        let mut prev = at(0);
        let mut line = format!("(b) {name:<13} sequence:");
        let mut err = 0.0;
        for t in 0..FRAMES {
            let scene = at(t);
            k.set_view(setup.view_at(t));
            let f = k.frame_tri(&scene, &prev, t);
            prev = scene;
            let (_, trim, _) = frame_err(&f.image, &f.kinds, &refs[&setup.ref_key(t)]);
            err += trim / FRAMES as f64;
            if [39, 63, 79, 80, 81, 85, 99, 100, 101, 105, 119].contains(&t) {
                line += &format!(" f{t} {:.3}", ratio(&f, &refs[&setup.ref_key(t)]));
            }
            if t == 80 || t == 100 {
                line += &format!(" [susp {} zero {}]", f.suspects, f.delta_zero);
            }
        }
        println!("{line} | relMSE* {err:.3e}");
    }
}

/// `krc sharccheck [cornell|sponza]`: validation of the SHaRC port. Static scene (frame 39),
/// fixed camera, 300 frames; with generous settings (every pixel updated, long
/// accumulation, fine voxels), the cache must converge to the reference.
pub fn sharc_check(gpu: &Gpu, setup: &TriSetup) {
    use crate::gpu_krc::{GpuKrc, Method, ShParams};
    use crate::grid::Grid;
    use crate::render::load_raw;
    let (view, at) = (setup.view, setup.at);
    let scene = at(39);
    let grid = Grid::fit(&scene.bounds);
    let r =
        load_raw(&format!("results/ref_{}/{}.bin", setup.name, setup.ref_key(39)), view.len()).expect("missing references");
    let d = ShParams::default();
    let variants = [
        ("NVIDIA default", d),
        ("every pixel", ShParams { downscale: 1, ..d }),
        ("+ accumulation 1024", ShParams { downscale: 1, accum_frames: 1024, ..d }),
        ("+ voxels 8× plus fins", ShParams { downscale: 1, accum_frames: 1024, scene_scale: 400.0, ..d }),
    ];
    for (name, sh) in variants {
        let mut k = GpuKrc::with_method(gpu, view, grid, Some(&scene), Method::Sharc, 5);
        k.sh = sh;
        let mut line = format!("{name:<24}");
        for t in 0..300 {
            let f = k.frame_tri(&scene, &scene, t);
            if [0, 9, 49, 299].contains(&t) {
                let (_, trim, lr) = frame_err(&f.image, &f.kinds, &r);
                line += &format!(" | f{t} relMSE* {trim:.2e} lum {lr:.3}");
            }
        }
        println!("{line}");
    }
    let mut k = GpuKrc::with_method(gpu, view, grid, Some(&scene), Method::Krc, 5);
    let mut line = format!("{:<24}", "KRC 1 path/px");
    for t in 0..300 {
        let f = k.frame_tri(&scene, &scene, t);
        if [0, 9, 49, 299].contains(&t) {
            let (_, trim, lr) = frame_err(&f.image, &f.kinds, &r);
            line += &format!(" | f{t} relMSE* {trim:.2e} lum {lr:.3}");
        }
    }
    println!("{line}");
}

/// `krc passes [sponzacam|sponza|cornellcam] [ppp]`: mean GPU time per pass (frames 0–119,
/// without image readback), KRC then official SHaRC, and number of live cells.
pub fn passes(gpu: &Gpu, setup: &TriSetup, ppp: f32) {
    use crate::gpu_krc::{GpuKrc, Method};
    use crate::grid::Grid;
    use crate::scene::FRAMES;
    let at = setup.at;
    let grid = Grid::fit(&at(0).bounds);
    for method in [Method::Krc, Method::Sharc] {
        let mut k = GpuKrc::with_method(gpu, setup.view, grid, Some(&at(0)), method, 3);
        k.ppp = ppp;
        k.readback = false;
        // optional KRC settings: KRC_RS=1 (resampling), KRC_CELL=8, KRC_SEED=1
        let env = |v: &str| std::env::var(v).ok();
        k.resample = env("KRC_RS").is_some();
        k.seed_levels = env("KRC_SEED").is_some();
        if let Some(c) = env("KRC_CELL").and_then(|c| c.parse().ok()) {
            k.cell_px = c;
        }
        if let Some(r) = env("KRC_REC").and_then(|c| c.parse().ok()) {
            k.rec_every = r;
        }
        k.demod = env("KRC_DM").is_some();
        k.support = env("KRC_SUP").is_some();
        k.game = env("KRC_GAME").is_some();
        k.delta_resample = env("KRC_DRS").is_some();
        if let Some(r) = env("KRC_SPP").and_then(|c| c.parse().ok()) {
            k.krc_spp = r;
        }
        k.m_light = std::env::var("KRC_ML").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
        let mut sum: Vec<(&'static str, f64)> = Vec::new();
        let (mut live, mut total) = (0u32, 0.0);
        let (mut segs, mut paths, mut hits) = (0u64, 0u64, 0u64);
        let mut prev = at(0);
        let mut worst: (f64, u32, Vec<(&'static str, f64)>) = (0.0, 0, Vec::new());
        for t in 0..FRAMES {
            let scene = at(t);
            k.set_view(setup.view_at(t));
            let f = k.frame_tri(&scene, &prev, t);
            prev = scene;
            if f.gpu_ms() > worst.0 {
                worst = (f.gpu_ms(), t, f.passes.clone());
            }
            total += f.gpu_ms() / FRAMES as f64;
            live = live.max(f.cells.saturating_sub(f.free));
            segs += f.segs as u64;
            paths += f.regular_paths as u64;
            hits += f.rs_hits as u64;
            for (name, ms) in &f.passes {
                match sum.iter_mut().find(|e| e.0 == *name) {
                    Some(e) => e.1 += ms / FRAMES as f64,
                    None => sum.push((name, ms / FRAMES as f64)),
                }
            }
        }
        sum.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("{method:?}: {total:.2} ms/frame on average, max live cells {live}");
        if method == Method::Krc {
            println!(
                "    {:.2} segments per ordinary path, {:.0} % stopped on the cache",
                segs as f64 / paths.max(1) as f64,
                100.0 * hits as f64 / paths.max(1) as f64
            );
        }
        for (name, ms) in sum.iter().take(12) {
            println!("    {name:<16} {ms:.3} ms");
        }
        let mut wp = worst.2.clone();
        wp.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!(
            "    slowest frame: f{} {:.2} ms — {}",
            worst.1,
            worst.0,
            wp.iter().take(5).map(|(n, m)| format!("{n} {m:.2}")).collect::<Vec<_>>().join(", ")
        );
    }
}

/// A configuration of the KRC / SHaRC duel.
#[derive(Clone, Copy)]
pub struct DuelCfg {
    pub name: &'static str,
    pub method: crate::gpu_krc::Method,
    pub ppp: f32,
    pub resample: bool,
    pub cell_px: f32,
    pub reuse: bool,
    pub demod: bool,
    pub seed: bool,
    pub support: bool,
    pub tau: f32,
    pub multi: bool,
    pub sh: crate::gpu_krc::ShParams,
    /// settings of the fixes (see GpuKrc): lamp correlated paths, payback,
    /// reset, denoiser clamp
    pub m_light: u32,
    pub debt_rate: f32,
    pub reset_ratio: f32,
    pub dn_gamma: f32,
    /// correlated paths stopped on the cache
    pub drs: bool,
}

impl Default for DuelCfg {
    fn default() -> Self {
        DuelCfg {
            name: "",
            method: crate::gpu_krc::Method::Krc,
            ppp: 1.0,
            resample: true,
            cell_px: 8.0,
            reuse: false,
            demod: true,
            seed: true,
            support: false,
            tau: 0.05,
            multi: false,
            sh: crate::gpu_krc::ShParams::default(),
            m_light: 0,
            debt_rate: -1.0,
            reset_ratio: 0.0,
            dn_gamma: 0.0,
            drs: false,
        }
    }
}

/// `krc duel [sponzacam|sponza|cornellcam|cornell]`: short test, 1 seed, 120 frames.
/// Pass 1: error (read back); pass 2: GPU time without readback. One line per configuration.
pub fn duel(gpu: &Gpu, setup: &TriSetup, cfgs: &[DuelCfg]) {
    use crate::gpu_krc::GpuKrc;
    use crate::grid::Grid;
    use crate::render::load_raw;
    use crate::scene::FRAMES;
    let at = setup.at;
    let grid = Grid::fit(&at(0).bounds);
    let dir = format!("results/ref_{}", setup.name);
    let refs: Vec<Vec<V3>> = (0..FRAMES)
        .map(|f| load_raw(&format!("{dir}/{}.bin", setup.ref_key(f)), setup.view.len()).expect("missing references"))
        .collect();
    let mut out = String::new();
    let seeds: u32 = std::env::var("KRC_DUEL_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    for c in cfgs {
        let (mut all, mut stat, mut dynm, mut lum_min, mut ms) = (0.0, 0.0, 0.0, 9.0f64, Vec::new());
        for seed in 0..seeds {
            let mut k = GpuKrc::with_method(gpu, setup.view, grid, Some(&at(0)), c.method, 3 + seed);
            k.ppp = c.ppp;
            k.resample = c.resample;
            k.cell_px = c.cell_px;
            k.reuse = c.reuse;
            k.demod = c.demod;
            k.seed_levels = c.seed;
            k.support = c.support;
            k.tau = c.tau;
            k.multi = c.multi;
            k.sh = c.sh;
            for pass in 0..2 {
                k.readback = pass == 0;
                let mut prev = at(0);
                for t in 0..FRAMES {
                    let scene = at(t);
                    k.set_view(setup.view_at(t));
                    let f = k.frame_tri(&scene, &prev, t);
                    prev = scene;
                    if pass == 0 {
                        let (_, e, lr) = frame_err(&f.image, &f.kinds, &refs[t as usize]);
                        let w = 1.0 / seeds as f64;
                        all += w * e / FRAMES as f64;
                        if (30..40).contains(&t) {
                            stat += w * e / 10.0;
                        }
                        if t >= 40 {
                            dynm += w * e / 80.0;
                        }
                        if t >= 20 {
                            lum_min = lum_min.min(lr);
                        }
                    } else {
                        ms.push(f.gpu_ms());
                    }
                }
            }
        }
        ms.sort_by(|a, b| a.total_cmp(b));
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        let line = format!(
            "{:<28} relMSE* {all:.3e} (static {stat:.2e}, dynamic {dynm:.2e}) lum min {lum_min:.2} | GPU mean {mean:.2} ms, p95 {:.2} ms",
            c.name,
            ms[ms.len() * 95 / 100]
        );
        println!("{line}");
        out += &line;
        out += "\n";
    }
    std::fs::create_dir_all("results/duel").unwrap();
    let path = format!("results/duel/{}.txt", setup.name);
    let prev = std::fs::read_to_string(&path).unwrap_or_default();
    std::fs::write(path, prev + &out).unwrap();
}

/// `krc static [sponza|cornell]`: convergence in a static scene, fixed camera, 300 frames.
/// Error at frames 9, 29, 99, 299 and mean GPU time (without readback, second pass).
pub fn static_conv(gpu: &Gpu, setup: &TriSetup, cfgs: &[DuelCfg]) {
    use crate::gpu_krc::GpuKrc;
    use crate::grid::Grid;
    use crate::render::load_raw;
    let (view, at) = (setup.view, setup.at);
    let scene = at(39);
    let grid = Grid::fit(&scene.bounds);
    let r =
        load_raw(&format!("results/ref_{}/{}.bin", setup.name, setup.ref_key(39)), view.len()).expect("missing references");
    let mut out = String::new();
    for c in cfgs {
        let mut k = GpuKrc::with_method(gpu, view, grid, Some(&scene), c.method, 5);
        k.ppp = c.ppp;
        k.resample = c.resample;
        k.cell_px = c.cell_px;
        k.reuse = c.reuse;
        k.demod = c.demod;
        k.seed_levels = c.seed;
        k.support = c.support;
        k.tau = c.tau;
        k.multi = c.multi;
        k.sh = c.sh;
        let mut line = format!("{:<28}", c.name);
        for t in 0..300 {
            let f = k.frame_tri(&scene, &scene, t);
            if [9, 29, 99, 299].contains(&t) {
                let (_, trim, lr) = frame_err(&f.image, &f.kinds, &r);
                line += &format!(" | f{t} {trim:.2e} (lum {lr:.2})");
            }
        }
        k.readback = false;
        let mut ms = 0.0;
        for t in 300..360 {
            ms += k.frame_tri(&scene, &scene, t).gpu_ms() / 60.0;
        }
        line += &format!(" | {ms:.2} ms");
        println!("{line}");
        out += &line;
        out += "\n";
    }
    std::fs::create_dir_all("results/duel").unwrap();
    let path = format!("results/duel/static_{}.txt", setup.name);
    let prev = std::fs::read_to_string(&path).unwrap_or_default();
    std::fs::write(path, prev + &out).unwrap();
}

/// `krc game [scene]`: game-like pipeline (per-pixel final image + shared denoiser),
/// 3 seeds. Error of the final image, flicker (frames 31–39, relative difference
/// from one frame to the next in the static phase), luminance after the large
/// lamp switches off (lag), full GPU time (denoiser included).
pub fn game_eval(gpu: &Gpu, setup: &TriSetup, cfgs: &[(DuelCfg, bool)]) {
    use crate::gpu_krc::GpuKrc;
    use crate::grid::Grid;
    use crate::render::load_raw;
    use crate::scene::FRAMES;
    let at = setup.at;
    let grid = Grid::fit(&at(0).bounds);
    let dir = format!("results/ref_{}", setup.name);
    let refs: Vec<Vec<V3>> = (0..FRAMES)
        .map(|f| load_raw(&format!("{dir}/{}.bin", setup.ref_key(f)), setup.view.len()).expect("missing references"))
        .collect();
    let seeds: u32 = std::env::var("KRC_DUEL_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let out_dir = format!("results/game/{}", setup.name);
    std::fs::create_dir_all(&out_dir).unwrap();
    let mut out = String::new();
    for (c, denoise) in cfgs {
        let w = 1.0 / seeds as f64;
        let (mut all, mut stat, mut dynm, mut flick) = (0.0, 0.0, 0.0, 0.0);
        let mut lum = [0.0f64; 4];
        let mut ms = Vec::new();
        for seed in 0..seeds {
            let mut k = GpuKrc::with_method(gpu, setup.view, grid, Some(&at(0)), c.method, 3 + seed);
            k.ppp = c.ppp;
            k.cell_px = c.cell_px;
            k.tau = c.tau;
            k.sh = c.sh;
            k.game = true;
            k.denoise = *denoise;
            k.m_light = c.m_light;
            k.debt_rate = c.debt_rate;
            k.reset_ratio = c.reset_ratio;
            k.dn_gamma = c.dn_gamma;
            k.delta_resample = c.drs;
            k.dn_wide = std::env::var("KRC_DN5").is_ok();
            for pass in 0..2 {
                k.readback = pass == 0;
                let mut prev = at(0);
                let mut last_img: Vec<V3> = Vec::new();
                for t in 0..FRAMES {
                    let scene = at(t);
                    k.set_view(setup.view_at(t));
                    let f = k.frame_tri(&scene, &prev, t);
                    prev = scene;
                    if pass == 1 {
                        ms.push(f.gpu_ms());
                        continue;
                    }
                    let r = &refs[t as usize];
                    let (_, e, lr) = frame_err(&f.image, &f.kinds, r);
                    all += w * e / FRAMES as f64;
                    if (30..40).contains(&t) {
                        stat += w * e / 10.0;
                    }
                    if t >= 40 {
                        dynm += w * e / 80.0;
                    }
                    for (i, ft) in [101u32, 103, 105, 110].iter().enumerate() {
                        if t == *ft {
                            lum[i] += w * lr;
                        }
                    }
                    if (31..40).contains(&t) && !last_img.is_empty() {
                        let (mut d, mut n) = (0.0f64, 0.0f64);
                        for i in 0..r.len() {
                            if f.kinds[i] == 2 {
                                d += (f.image[i].lum() - last_img[i].lum()).abs() / (r[i].lum() + 0.02);
                                n += 1.0;
                            }
                        }
                        flick += w * d / n.max(1.0) / 9.0;
                    }
                    if seed == 0 && [39u32, 101, 119].contains(&t) {
                        let tag = c.name.replace([' ', '/'], "_");
                        let suffix = if *denoise { "" } else { "_brut" };
                        crate::image::save(&format!("{out_dir}/{tag}{suffix}_f{t:03}.png"), setup.view.w, setup.view.h, &f.image);
                    }
                    last_img = f.image.clone();
                }
            }
        }
        ms.sort_by(|a, b| a.total_cmp(b));
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        let name = format!("{}{}", c.name, if *denoise { "" } else { " (brut)" });
        let line = format!(
            "{name:<30} relMSE* {all:.3e} (static {stat:.2e}, dynamic {dynm:.2e}) | flicker {flick:.3} | lum f101 {:.2} f103 {:.2} f105 {:.2} f110 {:.2} | GPU mean {mean:.2} ms, p95 {:.2} ms",
            lum[0],
            lum[1],
            lum[2],
            lum[3],
            ms[ms.len() * 95 / 100]
        );
        println!("{line}");
        out += &line;
        out += "\n";
    }
    let path = format!("{out_dir}/resultats.txt");
    let prev = std::fs::read_to_string(&path).unwrap_or_default();
    std::fs::write(path, prev + &out).unwrap();
}

/// Showcase Sponza: real colour textures (512², mipmaps), PBR materials chosen by
/// hand (the glTF has no names): almost-mirror polished stone floor, satin stone, matte
/// fabrics, metal (chain, hangers), chrome posts; alpha-masked foliage.
pub fn sponza_pbr() -> &'static (Vec<Mesh>, Aabb, TexArray) {
    static CELL: std::sync::OnceLock<(Vec<Mesh>, Aabb, TexArray)> = std::sync::OnceLock::new();
    CELL.get_or_init(|| {
        let dir = "assets/sponza";
        let gltf = gltf::Gltf::open(format!("{dir}/Sponza.gltf")).expect("assets/sponza: run tools/fetch_sponza.sh");
        let bin = std::fs::read(format!("{dir}/Sponza.bin")).expect("Sponza.bin");
        let scale = 0.008f32;
        const SIZE: u32 = 512;
        // one texture layer per material (colour texture)
        let mut layers: Vec<image::RgbaImage> = Vec::new();
        let mut layer_of = vec![-1i32; gltf.materials().len()];
        for mat in gltf.materials() {
            let Some(i) = mat.index() else { continue };
            if let Some(info) = mat.pbr_metallic_roughness().base_color_texture() {
                let gltf::image::Source::Uri { uri, .. } = info.texture().source().source() else { continue };
                let img = image::open(format!("{dir}/{uri}")).expect("texture Sponza").to_rgba8();
                layers.push(image::imageops::resize(&img, SIZE, SIZE, image::imageops::FilterType::Triangle));
                layer_of[i] = layers.len() as i32 - 1;
            }
        }
        let mut mips = Vec::new();
        let mut cur = layers;
        loop {
            let s = cur[0].width();
            mips.push(cur.iter().flat_map(|l| l.as_raw().iter().copied()).collect::<Vec<u8>>());
            if s == 1 {
                break;
            }
            cur = cur.iter().map(|l| image::imageops::resize(l, s / 2, s / 2, image::imageops::FilterType::Triangle)).collect();
        }
        let tex = TexArray { size: SIZE, layers: layer_of.iter().filter(|&&l| l >= 0).count() as u32, mips };
        let mut meshes = Vec::new();
        let mut b = Aabb { min: V3::splat(f64::INFINITY), max: V3::splat(f64::NEG_INFINITY) };
        for mesh in gltf.meshes() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|_| Some(&bin[..]));
                let pos: Vec<[f32; 3]> =
                    reader.read_positions().unwrap().map(|p| [p[0] * scale, p[1] * scale, p[2] * scale]).collect();
                let idx: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();
                let uv: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                    Some(t) => t.into_f32().collect(),
                    None => vec![[0.0; 2]; pos.len()],
                };
                let mat = prim.material();
                let mi = mat.index().unwrap_or(0);
                let c = mat.pbr_metallic_roughness().base_color_factor();
                // floor: upward faces at floor level
                let (mut up, mut area) = (0.0f32, 0.0f32);
                for t in idx.chunks(3) {
                    let (p0, p1, p2) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
                    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                    let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                    let a = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                    area += a;
                    if cr[1].abs() > 0.9 * a && p0[1] < 0.3 {
                        up += a;
                    }
                }
                let floor = area > 0.0 && up > 0.6 * area;
                let (rough, metal) = match mi {
                    _ if floor => (0.06, 0.0),
                    14..=19 | 0 | 3 | 24 => (1.0, 0.0), // tissus, feuillages, tuiles : mats
                    20 | 21 => (0.3, 1.0),              // chain, hangers: metal
                    2 => (0.15, 1.0),                   // poteaux : chrome
                    _ => (0.4, 0.0),                    // satin stone
                };
                let alpha_mask = matches!(mat.alpha_mode(), gltf::material::AlphaMode::Mask);
                for p in &pos {
                    let q = v3(p[0] as f64, p[1] as f64, p[2] as f64);
                    b = Aabb { min: b.min.min(q), max: b.max.max(q) };
                }
                meshes.push(Mesh {
                    pos,
                    idx,
                    albedo: [c[0], c[1], c[2]],
                    uv,
                    mat: Mat { tex: layer_of[mi], rough, metal, mask: alpha_mask },
                });
            }
        }
        (meshes, b.expand(0.1), tex)
    })
}

/// Showcase: textured Sponza (PBR materials), a chrome sphere rolling on the floor (instance 1),
/// a fixed golden sphere, a sky panel above the roof (lamp 0: it "sets" at frame
/// 100), the gallery's warm lamp (lamp 1: switches on at frame 80), two warm lanterns.
pub fn showcase(step: u32, sky_on: bool, l2_on: bool) -> TriScene {
    let (meshes0, bounds, tex) = sponza_pbr();
    let mut meshes = meshes0.clone();
    let mut instances: Vec<Instance> = (0..meshes.len()).map(|m| Instance { mesh: m, offset: V3::default() }).collect();
    meshes.push(Mesh::sphere([0.95, 0.95, 0.97], 0.7, Mat { tex: -1, rough: 0.04, metal: 1.0, mask: false }));
    instances.insert(1, Instance { mesh: meshes.len() - 1, offset: v3(3.0 - step as f64 * 0.5, 0.7, -0.7) });
    meshes.push(Mesh::sphere([1.0, 0.78, 0.34], 0.6, Mat { tex: -1, rough: 0.18, metal: 1.0, mask: false }));
    instances.push(Instance { mesh: meshes.len() - 1, offset: v3(-4.5, 0.6, 0.9) });
    let top = bounds.max.y + 0.5;
    let lights = vec![
        Light { center: v3(0.0, top, -0.3), hx: 11.0, hz: 2.2, emission: v3(3.2, 3.4, 3.8), on: sky_on },
        Light { center: v3(-6.0, 2.5, 4.0), hx: 0.4, hz: 0.4, emission: v3(60.0, 45.0, 27.0), on: l2_on },
        Light { center: v3(-8.5, 6.0, -0.3), hx: 0.3, hz: 0.3, emission: v3(80.0, 52.0, 26.0), on: true },
        Light { center: v3(8.5, 6.0, -0.3), hx: 0.3, hz: 0.3, emission: v3(80.0, 52.0, 26.0), on: true },
    ];
    TriScene { meshes, instances, lights, bounds: *bounds, tex: Some(tex) }
}

pub fn showcase_at(frame: u32) -> TriScene {
    let s = crate::scene::state_at(frame);
    showcase(s.box_step, s.l0_on, s.l2_on)
}

/// Showcase: walking through the atrium looking slightly toward the floor (reflections).
pub fn showcase_setup(w: usize, h: usize) -> TriSetup {
    fn cam(t: u32) -> (V3, V3) {
        let u = t as f64 / 119.0;
        let tau = 2.0 * std::f64::consts::PI;
        let eye = v3(-10.0 + 9.0 * u, 1.7, -0.3 + 0.8 * (tau * t as f64 / 120.0).sin());
        let yaw = 0.3 * (tau * t as f64 / 90.0).sin();
        (eye, eye + v3(yaw.cos(), -0.06, yaw.sin()) * 10.0)
    }
    let tv = 0.5;
    let mut view = sponza_setup(w, h).view;
    view.tv = tv;
    view.th = tv * w as f64 / h as f64;
    TriSetup { name: format!("showcase_{w}x{h}"), view, at: showcase_at, ref_spp: 8192, cam: Some(cam) }
}

/// PSNR (dB) between two 8-bit images.
pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0 * 255.0 / mse.max(1e-12)).log10()
}

/// Filmic tone mapping (ACES, Narkowicz approximation) then 8-bit sRGB.
pub fn tonemap(img: &[V3], exposure: f64) -> Vec<u8> {
    let aces = |x: f64| {
        let x = x * exposure;
        ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
    };
    let srgb = |c: f64| {
        let v = if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
        (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    };
    img.iter().flat_map(|v| [srgb(aces(v.x)), srgb(aces(v.y)), srgb(aces(v.z))]).collect()
}

/// `krc showcasetest [w]`: one reference image and one image of each method (game mode,
/// frame 30) to check the showcase's look.
pub fn showcase_test(gpu: &Gpu, w: usize) {
    use crate::gpu_krc::{GpuKrc, Method};
    use crate::grid::Grid;
    let setup = showcase_setup(w, w * 9 / 16);
    let (view, at) = (setup.view, setup.at);
    let scene0 = at(0);
    eprintln!("showcase: {} triangles, box {:?} → {:?}", scene0.triangles(), scene0.bounds.min, scene0.bounds.max);
    let grid = Grid::fit(&scene0.bounds);
    std::fs::create_dir_all("results/showcase").unwrap();
    let mut rk = GpuKrc::with(gpu, view, grid, Some(&scene0), false, 0);
    rk.set_view(setup.view_at(30));
    let t = std::time::Instant::now();
    let (r, kinds) = rk.reference_tri(&at(30), 512);
    eprintln!("reference 512 spp: {:.1} s", t.elapsed().as_secs_f64());
    let mean = r.iter().zip(&kinds).filter(|(_, k)| **k == 2).map(|(v, _)| v.lum()).sum::<f64>()
        / kinds.iter().filter(|k| **k == 2).count().max(1) as f64;
    let exposure = 0.25 / mean.max(1e-6);
    eprintln!("mean luminance {mean:.4}, exposure {exposure:.2}");
    crate::image::write_png("results/showcase/test_reference.png", view.w, view.h, &tonemap(&r, exposure)).unwrap();
    for (name, method) in [("krc", Method::Krc), ("sharc", Method::Sharc)] {
        let mut k = GpuKrc::with_method(gpu, view, grid, Some(&scene0), method, 1);
        k.game = true;
        k.ppp = 0.5;
        let mut prev = at(0);
        let mut last = Vec::new();
        for f in 0..=30 {
            let scene = at(f);
            k.set_view(setup.view_at(f));
            let fr = k.frame_tri(&scene, &prev, f);
            prev = scene;
            last = fr.image;
            if f == 30 {
                eprintln!("{name} : {:.2} ms", fr.passes.iter().map(|p| p.1).sum::<f64>());
            }
        }
        crate::image::write_png(&format!("results/showcase/test_{name}.png"), view.w, view.h, &tonemap(&last, exposure)).unwrap();
    }
}

/// `krc showcasems [w]`: mean GPU time (frames 10–40, without readback) of a few game-mode
/// settings on the showcase, to choose configurations at equal time.
pub fn showcase_ms(gpu: &Gpu, w: usize) {
    use crate::gpu_krc::{GpuKrc, Method, ShParams};
    use crate::grid::Grid;
    let setup = showcase_setup(w, w * 9 / 16);
    let at = setup.at;
    let scene0 = at(0);
    let grid = Grid::fit(&scene0.bounds);
    let d = ShParams::default();
    let cfgs: Vec<(&str, Method, f32, ShParams)> = vec![
        ("krc 0.25", Method::Krc, 0.25, d),
        ("krc 0.5", Method::Krc, 0.5, d),
        ("krc 1", Method::Krc, 1.0, d),
        ("krc 0.25 spp2", Method::Krc, 0.25, ShParams { render_spp: 2, ..d }),
        ("krc 0.25 spp3", Method::Krc, 0.25, ShParams { render_spp: 3, ..d }),
        ("krc 0.5 spp2", Method::Krc, 0.5, ShParams { render_spp: 2, ..d }),
        ("krc 0.5 spp3", Method::Krc, 0.5, ShParams { render_spp: 3, ..d }),
        ("sharc 5x5 spp1", Method::Sharc, 1.0, d),
        ("sharc 5x5 spp2", Method::Sharc, 1.0, ShParams { render_spp: 2, ..d }),
        ("sharc 2x2 spp2", Method::Sharc, 1.0, ShParams { render_spp: 2, downscale: 2, ..d }),
        ("sharc 5x5 spp3", Method::Sharc, 1.0, ShParams { render_spp: 3, ..d }),
        ("sharc 2x2 spp3", Method::Sharc, 1.0, ShParams { render_spp: 3, downscale: 2, ..d }),
    ];
    for (name, method, ppp, sh) in cfgs {
        let mut k = GpuKrc::with_method(gpu, setup.view, grid, Some(&scene0), method, 1);
        k.game = true;
        k.ppp = ppp;
        k.sh = sh;
        k.krc_spp = sh.render_spp;
        k.off = std::env::var("KRC_OFF").ok().and_then(|c| c.parse().ok()).unwrap_or(0); // for KRC, render_spp is the number of estimates per pixel
        k.readback = false;
        let mut prev = at(0);
        let mut ms = 0.0;
        for f in 0..=40 {
            let scene = at(f);
            k.set_view(setup.view_at(f));
            let fr = k.frame_tri(&scene, &prev, f);
            prev = scene;
            if f >= 10 {
                ms += fr.gpu_ms() / 31.0;
            }
        }
        println!("{name:<16} {ms:.2} ms");
    }
}

/// `krc showcase [w]`: comparison video KRC | SHaRC (NVIDIA) | reference on the showcase, same
/// game pipeline (per-pixel direct, reflections, shared denoiser), at equal GPU time.
/// Outputs: results/showcase/{krc,sharc,reference}/fNNN.png, comparison.mp4 (24 fps),
/// comparison_slow.mp4 (12 fps), stills at key moments, measurements.txt.
pub fn showcase_video(gpu: &Gpu, w: usize) {
    use crate::gpu_krc::{GpuKrc, Method, ShParams};
    use crate::grid::Grid;
    use crate::render::{load_raw, save_raw};
    use crate::scene::FRAMES;
    let h = w * 9 / 16;
    let setup = showcase_setup(w, h);
    let (view, at) = (setup.view, setup.at);
    let scene0 = at(0);
    let grid = Grid::fit(&scene0.bounds);
    let ref_spp: u32 = std::env::var("KRC_REF_SPP").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let out = "results/showcase";
    let refdir = format!("results/ref_{}_{}", setup.name, ref_spp);
    for d in [format!("{out}/krc"), format!("{out}/sharc"), format!("{out}/reference"), refdir.clone()] {
        std::fs::create_dir_all(d).unwrap();
    }
    // references (one per frame, kept on disk)
    let mut rk = GpuKrc::with(gpu, view, grid, Some(&scene0), false, 0);
    let mut refs = Vec::new();
    for f in 0..FRAMES {
        let path = format!("{refdir}/f{f:03}.bin");
        let img = load_raw(&path, view.len()).unwrap_or_else(|| {
            let t = std::time::Instant::now();
            rk.set_view(setup.view_at(f));
            let (img, _) = rk.reference_tri(&at(f), ref_spp);
            save_raw(&path, &img);
            eprintln!("reference f{f:03}: {:.1} s", t.elapsed().as_secs_f64());
            img
        });
        refs.push(img);
    }
    drop(rk);
    let (_, kinds0) = {
        let mut k = GpuKrc::with(gpu, view, grid, Some(&scene0), false, 0);
        k.set_view(setup.view_at(0));
        k.reference_tri(&at(0), 1)
    };
    let mean = refs[0].iter().zip(&kinds0).filter(|(_, k)| **k == 2).map(|(v, _)| v.lum()).sum::<f64>()
        / kinds0.iter().filter(|k| **k == 2).count().max(1) as f64;
    let exposure = 0.25 / mean.max(1e-6);
    let ref_png: Vec<Vec<u8>> = refs.iter().map(|r| tonemap(r, exposure)).collect();
    for (f, r) in ref_png.iter().enumerate() {
        crate::image::write_png(&format!("{out}/reference/f{f:03}.png"), w, h, r).unwrap();
    }
    let d = ShParams::default();
    let methods = [
        (
            "krc",
            "KRC",
            Method::Krc,
            std::env::var("KRC_PPP").ok().and_then(|c| c.parse().ok()).unwrap_or(0.4f32),
            ShParams { render_spp: 2, ..d },
        ),
        ("sharc", "SHaRC NVIDIA", Method::Sharc, 1.0, ShParams { downscale: 2, render_spp: 3, ..d }),
    ];
    let mut report = String::new();
    let mut labels = Vec::new();
    for (dir, label, method, ppp, sh) in methods {
        // GPU time: a separate pass, without image readback or writing (otherwise the GPU waits for the
        // CPU and slows down)
        let mut k = GpuKrc::with_method(gpu, view, grid, Some(&scene0), method, 1);
        k.game = true;
        k.ppp = ppp;
        k.sh = sh;
        k.krc_spp = sh.render_spp;
        k.off = std::env::var("KRC_OFF").ok().and_then(|c| c.parse().ok()).unwrap_or(0);
        if method == Method::Krc {
            // 16 px cells: fewer low-frequency blotches (noise specific to each cell) than
            // 8 px in this textured scene, and cheaper; sweep 6/8/12/16 px in RESULTS §17
            k.cell_px = std::env::var("KRC_CELL").ok().and_then(|c| c.parse().ok()).unwrap_or(16.0);
        }
        k.m_light = std::env::var("KRC_ML").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        k.readback = false;
        let mut prev = at(0);
        let mut ms = Vec::new();
        for f in 0..FRAMES {
            let scene = at(f);
            k.set_view(setup.view_at(f));
            ms.push(k.frame_tri(&scene, &prev, f).gpu_ms());
            prev = scene;
        }
        drop(k);
        ms.sort_by(|a, b| a.total_cmp(b));
        let mean_ms = ms.iter().sum::<f64>() / ms.len() as f64;
        let p95 = ms[ms.len() * 95 / 100];
        let mut k = GpuKrc::with_method(gpu, view, grid, Some(&scene0), method, 1);
        k.game = true;
        k.ppp = ppp;
        k.sh = sh;
        k.krc_spp = sh.render_spp;
        k.off = std::env::var("KRC_OFF").ok().and_then(|c| c.parse().ok()).unwrap_or(0);
        if method == Method::Krc {
            // 16 px cells: fewer low-frequency blotches (noise specific to each cell) than
            // 8 px in this textured scene, and cheaper; sweep 6/8/12/16 px in RESULTS §17
            k.cell_px = std::env::var("KRC_CELL").ok().and_then(|c| c.parse().ok()).unwrap_or(16.0);
        }
        k.m_light = std::env::var("KRC_ML").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        let mut prev = at(0);
        let (mut err, mut psnr_m) = (0.0, 0.0);
        for f in 0..FRAMES {
            let scene = at(f);
            k.set_view(setup.view_at(f));
            let fr = k.frame_tri(&scene, &prev, f);
            prev = scene;
            let (_, e, lr) = frame_err(&fr.image, &fr.kinds, &refs[f as usize]);
            err += e / FRAMES as f64;
            // error "to the eye": PSNR of the displayed images (after tone mapping, 8 bits); relMSE*
            // penalises a bright pixel on a dark background of the reference very heavily (edge of a sky
            // reflection), a missing reflection much less
            let img = tonemap(&fr.image, exposure);
            let p = psnr(&img, &ref_png[f as usize]);
            psnr_m += p / FRAMES as f64;
            if [39u32, 60, 81, 101, 110, 119].contains(&f) {
                report += &format!("{label} f{f:03}: relMSE* {e:.3e}, lum {lr:.2}, displayed PSNR {p:.2} dB\n");
            }
            crate::image::write_png(&format!("{out}/{dir}/f{f:03}.png"), w, h, &img).unwrap();
        }
        report += &format!(
            "{label}: mean relMSE* {err:.3e}, mean displayed PSNR {psnr_m:.2} dB, mean GPU {mean_ms:.2} ms, p95 {p95:.2} ms\n"
        );
        labels.push(format!("{label} - {mean_ms:.1} ms per frame"));
        eprintln!("{label} : relMSE* {err:.3e}, {mean_ms:.2} ms");
    }
    labels.push(format!("Reference - {ref_spp} paths per pixel"));
    std::fs::write(format!("{out}/measurements.txt"), &report).unwrap();
    // side-by-side video, labelled
    let font = "C\\:/Windows/Fonts/arial.ttf";
    let lab = |i: usize| {
        format!(
            "drawtext=fontfile='{font}':text='{}':x=16:y=16:fontsize={}:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8",
            labels[i],
            (h / 22).max(14)
        )
    };
    let filter = format!("[0:v]{}[a];[1:v]{}[b];[2:v]{}[c];[a][b][c]hstack=inputs=3,format=yuv420p[v]", lab(0), lab(1), lab(2));
    for (fps, name) in [(24, "comparison.mp4"), (12, "comparison_slow.mp4")] {
        let mut cmd = std::process::Command::new("ffmpeg");
        cmd.args(["-y", "-loglevel", "error"]);
        for dir in ["krc", "sharc", "reference"] {
            cmd.args(["-framerate", &fps.to_string(), "-i", &format!("{out}/{dir}/f%03d.png")]);
        }
        cmd.args(["-filter_complex", &filter, "-map", "[v]", "-c:v", "libx264", "-crf", "16", "-preset", "slow"]);
        cmd.arg(format!("{out}/{name}"));
        let st = cmd.status().expect("ffmpeg");
        eprintln!("{name} : {st}");
    }
    // stills: static, rolling sphere, gallery lit, sky set
    for f in [39u32, 60, 90, 110] {
        let mut cmd = std::process::Command::new("ffmpeg");
        cmd.args(["-y", "-loglevel", "error"]);
        for dir in ["krc", "sharc", "reference"] {
            cmd.args(["-i", &format!("{out}/{dir}/f{f:03}.png")]);
        }
        let still = format!("[0:v]{}[a];[1:v]{}[b];[2:v]{}[c];[a][b][c]hstack=inputs=3[v]", lab(0), lab(1), lab(2));
        cmd.args(["-filter_complex", &still, "-map", "[v]", "-frames:v", "1"]);
        cmd.arg(format!("{out}/comparison_f{f:03}.png"));
        cmd.status().expect("ffmpeg");
    }
}

/// `krc showcaseref f [spp]`: reference of one showcase frame (check).
pub fn showcase_ref(gpu: &Gpu, f: u32, spp: u32) {
    use crate::gpu_krc::GpuKrc;
    use crate::grid::Grid;
    let setup = showcase_setup(960, 540);
    let scene0 = (setup.at)(0);
    let mut rk = GpuKrc::with(gpu, setup.view, Grid::fit(&scene0.bounds), Some(&scene0), false, 0);
    rk.set_view(setup.view_at(f));
    let (img, _) = rk.reference_tri(&(setup.at)(f), spp);
    crate::image::write_png(&format!("results/showcase/test_ref_f{f:03}.png"), 960, 540, &tonemap(&img, 9.3)).unwrap();
    // reference built directly with frame f's scene
    let scf = (setup.at)(f);
    let mut rk2 = GpuKrc::with(gpu, setup.view, Grid::fit(&scene0.bounds), Some(&scf), false, 0);
    rk2.set_view(setup.view_at(f));
    let (img3, _) = rk2.reference_tri(&scf, spp);
    crate::image::write_png(&format!("results/showcase/test_ref_direct_f{f:03}.png"), 960, 540, &tonemap(&img3, 9.3)).unwrap();
    // same frame, KRC from cold (a single frame, without a sequence)
    let mut k = GpuKrc::with(gpu, setup.view, Grid::fit(&scene0.bounds), Some(&scene0), false, 1);
    k.set_view(setup.view_at(f));
    let sc = (setup.at)(f);
    let mut img2 = Vec::new();
    for _ in 0..8 {
        img2 = k.frame_tri(&sc, &sc, f).image;
    }
    crate::image::write_png(&format!("results/showcase/test_krc_cold_f{f:03}.png"), 960, 540, &tonemap(&img2, 9.3)).unwrap();
}

/// `krc showcasedbg f`: components of KRC's final image at frame f (sequence played from 0):
/// cells' indirect alone, direct + reflection alone, full image; with and without the denoiser.
pub fn showcase_dbg(gpu: &Gpu, f: u32) {
    use crate::gpu_krc::{GpuKrc, Method, ShParams};
    use crate::grid::Grid;
    let setup = showcase_setup(960, 540);
    let at = setup.at;
    let scene0 = at(0);
    let grid = Grid::fit(&scene0.bounds);
    for (dbg, name) in [(1u32, "indirect"), (2, "direct"), (0, "complet")] {
        for denoise in [false, true] {
            let mut k = GpuKrc::with_method(gpu, setup.view, grid, Some(&scene0), Method::Krc, 1);
            k.game = true;
            k.ppp = 0.25;
            k.krc_spp = 2;
            k.m_light = 16;
            k.sh = ShParams::default();
            k.dbg = dbg;
            if let Some(c) = std::env::var("KRC_CELL").ok().and_then(|c| c.parse().ok()) {
                k.cell_px = c;
            }
            k.denoise = denoise;
            k.off = std::env::var("KRC_OFF").ok().and_then(|c| c.parse().ok()).unwrap_or(0);
            let mut prev = at(0);
            let mut img = Vec::new();
            for t in 0..=f {
                let scene = at(t);
                k.set_view(setup.view_at(t));
                k.readback = t == f;
                let fr = k.frame_tri(&scene, &prev, t);
                prev = scene;
                if t == f {
                    img = fr.image;
                }
            }
            let tag = if denoise { "debruite" } else { "brut" };
            crate::image::write_png(&format!("results/showcase/dbg_f{f:03}_{name}_{tag}.png"), 960, 540, &tonemap(&img, 9.3))
                .unwrap();
        }
    }
}

/// `krc showcaselum`: share of each protection in the darkening (KRC and SHaRC, game mode,
/// frames 0–45, relMSE* and luminance against the reference at frames 30–45).
pub fn showcase_lum(gpu: &Gpu) {
    use crate::gpu_krc::{GpuKrc, Method, ShParams};
    use crate::grid::Grid;
    use crate::render::load_raw;
    let setup = showcase_setup(960, 540);
    let at = setup.at;
    let scene0 = at(0);
    let grid = Grid::fit(&scene0.bounds);
    let refdir = format!("results/ref_{}_2048", setup.name);
    let d = ShParams::default();
    let cases: Vec<(&str, Method, u32, bool, f32)> = vec![
        ("krc", Method::Krc, 0, true, 0.0),
        ("krc without anti-firefly", Method::Krc, 1, true, 0.0),
        ("krc without truncated mean", Method::Krc, 2, true, 0.0),
        ("krc without cell clamping", Method::Krc, 4, true, 0.0),
        ("krc without regularisation", Method::Krc, 8, true, 0.0),
        ("krc without anything", Method::Krc, 15, true, 0.0),
        ("krc without anything, clamp 4", Method::Krc, 15, true, 4.0),
        ("krc raw (no denoiser)", Method::Krc, 15, false, 0.0),
        ("krc without temporal", Method::Krc, 15 | 16, true, 0.0),
        ("krc without a-trous", Method::Krc, 15 | 32, true, 0.0),
        ("krc raw specular", Method::Krc, 15 | 64, true, 0.0),
        ("sharc without temporal", Method::Sharc, 15 | 16, true, 0.0),
        ("sharc without a-trous", Method::Sharc, 15 | 32, true, 0.0),
        ("sharc raw specular", Method::Sharc, 15 | 64, true, 0.0),
        ("sharc", Method::Sharc, 0, true, 0.0),
        ("sharc without anything", Method::Sharc, 15, true, 0.0),
        ("sharc brut", Method::Sharc, 15, false, 0.0),
    ];
    for (name, method, off, denoise, gamma) in cases {
        let mut k = GpuKrc::with_method(gpu, setup.view, grid, Some(&scene0), method, 1);
        k.game = true;
        k.off = off;
        k.denoise = denoise;
        k.dn_gamma = gamma;
        k.m_light = 16;
        if method == Method::Krc {
            k.ppp = 0.25;
            k.krc_spp = 2;
        } else {
            k.sh = ShParams { downscale: 2, render_spp: 3, ..d };
        }
        let mut prev = at(0);
        let (mut e, mut lr, mut n) = (0.0, 0.0, 0.0);
        for t in 0..=45u32 {
            let scene = at(t);
            k.set_view(setup.view_at(t));
            k.readback = t >= 30;
            let f = k.frame_tri(&scene, &prev, t);
            prev = scene;
            if t >= 30 {
                let r = load_raw(&format!("{refdir}/f{t:03}.bin"), setup.view.len()).expect("reference");
                let (_, et, l) = frame_err(&f.image, &f.kinds, &r);
                e += et;
                lr += l;
                n += 1.0;
            }
        }
        println!("{name:<30} relMSE* {:.3e}  lum {:.3}", e / n, lr / n);
    }
}

/// `krc showcaseregref f`: luminance of a regularised reference against the exact reference.
pub fn showcase_regref(gpu: &Gpu, f: u32) {
    use crate::gpu_krc::GpuKrc;
    use crate::grid::Grid;
    use crate::render::load_raw;
    let setup = showcase_setup(960, 540);
    let scene0 = (setup.at)(0);
    let mut rk = GpuKrc::with(gpu, setup.view, Grid::fit(&scene0.bounds), Some(&scene0), false, 0);
    rk.set_view(setup.view_at(f));
    rk.ref_reg = true;
    let (img, kinds) = rk.reference_tri(&(setup.at)(f), 2048);
    let r = load_raw(&format!("results/ref_{}_2048/f{f:03}.bin", setup.name), setup.view.len()).expect("reference");
    let (_, e, lr) = frame_err(&img, &kinds, &r);
    println!("regularised / exact reference, f{f}: lum {lr:.3}, relMSE* {e:.3e}");
    crate::image::write_png(&format!("results/showcase/ref_reg_f{f:03}.png"), 960, 540, &tonemap(&img, 9.3)).unwrap();
}
