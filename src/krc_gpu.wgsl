// KRC entirely on the GPU: a pipeline of compute passes per frame.
// Same rules as methods.rs (krc_energy, L2 dependencies), with two documented
// simplifications: Neyman allocation by bisection of the gain threshold (continuous equivalent of the
// greedy one) and one reference pixel per cell per frame (drawn uniformly).

// <grid> (replaced by the scene's grid, see Grid::wgsl)
const REGION: f32 = 0.6;
const ORIGIN: vec3<f32> = vec3<f32>(-0.2, -0.2, -0.2);
const DIMS: vec3<i32> = vec3<i32>(72, 6, 8);
const SDIV: vec3<i32> = vec3<i32>(4, 6, 4);
const SDIMS: vec3<i32> = vec3<i32>(18, 1, 2);
// </grid>
const MAX_DEPTH: u32 = 8u;
const MISS_LEN: f32 = 30.0;
const EPS: f32 = 1e-4;
const OFFSET: f32 = 1e-4;
const PI: f32 = 3.14159265358979;
const LIGHT_ID_BASE: u32 = 64u;

const K: u32 = 4u;
const M_DELTA0: u32 = 2u;
const M_DELTA: u32 = 8u;
const SAFETY_WINDOW: f32 = 32.0;
const Z_SAFETY: f32 = 4.0;
const EXPLORE_PERIOD: u32 = 16u;
const KAPPA: f32 = 4.0;
const PRIOR_FLOOR: f32 = 0.1;
const REL_EPS: f32 = 0.01;
const DEP_WINDOW: u32 = 64u;
const DEBT_RATE: f32 = 0.5;
const EMA_MIN: f32 = 0.015625;
const MAX_PER_CELL: u32 = 64u;
const N_MAX: f32 = 1024.0; // maximum history of a cell (drift test)
const N_POST: f32 = 64.0;  // max weight of a corrected estimate (its variance comes from 8 draws); tuned by the drift test

// compteurs (ctr)
const C_CELLS: u32 = 0u;
const C_TASKS: u32 = 1u;     // write cursor in `tasks`
const C_SURF: u32 = 2u;
const C_SUSPECTS: u32 = 3u;
const C_DPATHS: u32 = 4u;
const C_DOLD: u32 = 5u;
const C_DZERO: u32 = 6u;
const C_HARD: u32 = 7u;
const C_SAFETY: u32 = 8u;
const C_DEBT: u32 = 9u;
const C_D1: u32 = 10u;       // delta tasks of the 1st round: [0, C_D1)
const C_D2: u32 = 11u;       // end of the 2nd round
const C_REG0: u32 = 12u;     // start of the ordinary tasks
const C_VISIBLE: u32 = 13u;
const C_MATURE: u32 = 14u;
const C_OVERFLOW: u32 = 15u;
const C_EXPLORE: u32 = 16u;
const C_MAXN: u32 = 17u;     // max of the cells' n (positive f32 bits: order preserved)
const C_PENDING: u32 = 18u;  // cells still waiting for their correlated paths
const C_DUSED: u32 = 19u;    // correlated paths reserved in the frame (amortisation cap)
const C_FREE: u32 = 20u;     // free cells in free_list (persistent)
const C_TOMB: u32 = 21u;     // tombstones in the hash table (persistent)
const C_EVICT: u32 = 22u;    // cells evicted in the frame
const C_POP: u32 = 23u;      // cells taken from the free list during insertion
const C_NVIS: u32 = 24u;     // visible cells in the frame (vis_list)
const C_SEGS: u32 = 25u;     // segments traced by ordinary paths
const C_RSHIT: u32 = 26u;    // ordinary paths stopped on the cache
const TOMB: u32 = 0xffffffffu;
// Moving camera: a cell invisible for STALE_FRAMES frames is evicted (same rule as
// the staleFrameNumMax of NVIDIA's SHaRC sample, 60).
const STALE_FRAMES: u32 = 60u;
const C_HIST: u32 = 32u;     // 64 cases
const C_SCAN: u32 = 96u;     // 3 × 32 candidate quota sums (three sweeps)

// atomic accumulators per cell (acc), in words
const ACC: u32 = 52u;
const A_FOOT: u32 = 0u;
const A_REF: u32 = 1u;
const A_FN: u32 = 2u;
const A_FR: u32 = 3u;        // 3, 4, 5 : Σ rgb
const A_FL: u32 = 6u;
const A_FL2: u32 = 7u;
const A_DEPS: u32 = 8u;      // 8, 9: super-regions crossed (current window)
const A_NS: u32 = 10u;       // samples in the current window
const A_FREQ: u32 = 11u;     // 11..28: 36 u16 vertex counters per super-region
const A_PM: u32 = 29u;
const A_PDIFF: u32 = 30u;
const A_PAFF: u32 = 31u;
const A_PX: u32 = 32u;       // 32, 33, 34 : Σx rgb
const A_PY: u32 = 35u;       // 35, 36, 37 : Σy rgb
const A_PXL: u32 = 38u;
const A_PYL: u32 = 39u;
const A_PXL2: u32 = 40u;
const A_PYL2: u32 = 41u;
const A_PXYL: u32 = 42u;
const A_SUSP: u32 = 43u;
const A_FD: u32 = 44u;        // 44, 45, 46: Σ direct share at the first vertex (game mode)
const A_PD: u32 = 47u;        // 47, 48, 49: Σ (new direct − old direct) of the correlated paths

// fr.flags
const F_EVENT: u32 = 1u;
const F_GEOMETRY: u32 = 2u;
const F_SHARC: u32 = 4u;
const F_PENDING: u32 = 8u;   // frame without a new event, but with cells waiting
const F_AMORTIZE: u32 = 16u;
const F_S2HIST: u32 = 32u;
const F_ZCAP: u32 = 64u;
// cache resampling (like SHaRC): an ordinary path stops at the 1st or 2nd bounce
// on an existing cell and takes its mean
const F_RESAMPLE: u32 = 128u;
const RESAMPLE_N_MIN: f32 = 2.0;
// vertex reuse (like SHaRC): the radiance of the rest of the path at the 1st bounce is
// also a sample of the visible cell located there
const F_REUSE: u32 = 256u;
// material demodulation (like SHARC_MATERIAL_DEMODULATION): cells store the radiance
// divided by the albedo of the starting point, the display multiplies it back by the pixel's albedo
const F_DEMOD: u32 = 512u;
// new cell initialised from the cell of the neighbouring level at the same place (a camera
// moving forward makes surfaces change level; like SHARC_BLEND_ADJACENT_LEVELS)
const F_SEED: u32 = 1024u;
// off-screen support cells (like SHaRC voxels at secondary vertices): an ordinary
// path that finds no mature cell at the resampling vertex creates one (or
// feeds the young cell found) with the radiance of the rest of its path
const F_SUPPORT: u32 = 2048u;
// two-resolution cache: a pixel's first K/2 slots target fine cells,
// the next K/2 cells 2× larger; each path feeds both levels;
// the display moves from coarse to fine when the fine one has enough samples
const F_MULTI: u32 = 4096u;
// cells store only indirect light (direct is computed per pixel)
const F_INDIRECT: u32 = 8192u;
// game mode: cells keep the total radiance (read back by resampling, like
// SHaRC) and, separately, the mean of their direct share; the final image takes the indirect
// = total − mean direct, and the per-pixel direct
const F_DSPLIT: u32 = 16384u;
const F_GAME: u32 = 32768u;   // game pipeline (separate diffuse / specular parts, game.wgsl)
// correlated paths also stopped on the cache at the 1st or 2nd bounce: both paths of the
// pair read the same value, which cancels out in the difference; only the effect of
// the event beyond the stopping point is missing (second order, corrected over the next frames)
const F_DRS: u32 = 65536u;
const FIREFLY_K: f32 = 100.0;
const FIREFLY_K_YOUNG: f32 = 20.0;
const MULTI_N: f32 = 32.0;

fn slot_dist(dist: f32, slot: u32) -> f32 {
    if ((fr.flags & F_MULTI) != 0u && slot >= K / 2u) { return dist * 2.0; }
    return dist;
}
const SEED_N: f32 = 8.0;

fn demod(a: vec3<f32>) -> vec3<f32> {
    if ((fr.flags & F_DEMOD) == 0u) { return vec3<f32>(1.0); }
    return max(a, vec3<f32>(0.05));
}     // lamp: cell without a measured difference → capped history   // per-sample variance after correction: at least the history's // correlated paths capped at DELTA_CAP path/pixel per frame
const DELTA_CAP: f32 = 1.0;

struct Frame {
    w: u32, h: u32, n_boxes: u32, n_lights: u32,
    frame: u32, seed: u32, flags: u32, table_mask: u32,
    cell_cap: u32, round: u32, ev_lights: u32, pad0: u32,
    eye: vec3<f32>, px_angle: f32,
    fwd: vec3<f32>, th: f32,
    right: vec3<f32>, tv: f32,
    up: vec3<f32>, ppp: f32,  // budget: paths per surface pixel per frame
    clock: u32,      // monotonic frame counter (eviction), independent of `frame` (seeds)
    rel_cap: f32,    // relative allocation variance capped at rel_cap × median (0: no cap)
    eps_energy: f32, // threshold of the expected energy effect (fraction of the cell's luminance)
    n_post: f32,     // max weight of a corrected estimate (0: N_POST)
    // official SHaRC (sharc.wgsl)
    sh_cam_prev: vec3<f32>, sh_scene_scale: f32,
    sh_rscale: f32, sh_cap: u32, sh_accum: u32, sh_stale: u32,
    sh_down: u32, cell_px: f32, rec_every: u32, tau: f32, // cell_px: width of a cell in pixels; rec_every: 1 ordinary path in N records its dependencies
    // previous frame's camera (denoiser reprojection, game.wgsl)
    prev_eye: vec3<f32>, m_light: u32,     // m_light: correlated paths per cell for a lamp (0: M_DELTA)
    prev_fwd: vec3<f32>, debt_rate: f32,   // share of the budget that pays back event paths (< 0: DEBT_RATE)
    prev_right: vec3<f32>, reset_ratio: f32, // relative change beyond which it restarts from zero (0: never)
    prev_up: vec3<f32>, dn_gamma: f32,     // denoiser history clamp (0: DN_GAMMA)
    sh_spp: u32, krc_spp: u32, dbg: u32, off: u32, // off: protections disabled (1 anti-firefly, 2 truncated mean, 4 cell clamping, 8 regularisation) // SHaRC: paths per pixel of the Render pass; KRC: direct estimates per pixel (game mode)
};
struct Box { mn: vec3<f32>, p0: f32, mx: vec3<f32>, p1: f32, albedo: vec3<f32>, p2: f32 };
struct Light { center: vec3<f32>, hx: f32, hz: f32, on: f32, power: f32, toggle_w: f32, emission: vec3<f32>, p0: f32 };
struct Gpx { p: vec3<f32>, kind: u32, n: vec3<f32>, dist: f32, albedo: vec3<f32>, p0: u32 };
struct Cell {
    n: f32, lmean: f32, m2: f32, last: u32,
    mean: vec3<f32>, rn: f32,
    rl: f32, rm2: f32, ns_prev: u32, seen: u32, // seen: last frame (clock) where the cell was visible
    deps_prev: vec2<u32>, key: u32, slot: u32,   // slot: index in the table + 1 (0: free cell)
    dmean: vec3<f32>, p2: f32,                   // mean of the direct share at the first vertex (game mode)
    freq_prev: array<u32, 18>,
};

@group(0) @binding(0) var<uniform> fr: Frame;
@group(0) @binding(2) var<storage, read> lights: array<Light>;
@group(0) @binding(4) var<storage, read> lights_prev: array<Light>;
@group(0) @binding(5) var<storage, read_write> gbuf: array<Gpx>;
@group(0) @binding(6) var<storage, read_write> pix_cells: array<u32>;
@group(0) @binding(7) var<storage, read_write> keys: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> vals: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> ctr: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(11) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> tasks: array<vec4<u32>>;
@group(0) @binding(13) var<storage, read_write> image: array<vec4<f32>>;
// event: [0,108) fine regions, [108,110) super-regions, [110,146) magnitude per
// super-region (f32), [146,162) lamp weights (f32)
@group(0) @binding(14) var<storage, read> ev: array<u32>;
@group(0) @binding(15) var<storage, read_write> scal: array<f32>; // 0 : λ, 1 : a priori, 2 : budget, 3-4 : bornes log2 λ
@group(0) @binding(16) var<storage, read_write> ind: array<u32>; // arguments de dispatch indirect
// compact summary per cell for allocation: (footprint, n, relative variance, draw)
@group(0) @binding(17) var<storage, read_write> qbuf: array<vec4<f32>>;
@group(0) @binding(22) var<storage, read_write> free_list: array<u32>;
// visible cells of the frame: allocation and resolve walk only those (with a
// moving camera, most live cells are off screen)
@group(0) @binding(27) var<storage, read_write> vis_list: array<u32>;

// ---------------------------------------------------------------- outils
var<private> rng: u32;
var<private> sel: u32;        // 0: current scene, 1: previous scene
var<private> rec_on: bool;
var<private> uni: bool;
var<private> split: bool;
var<private> m_trav: vec2<u32>;
var<private> m_vsup: vec2<u32>;
var<private> m_lights: u32;
var<private> m_aff: bool;     // does the path touch a region of the event?
var<private> resample: bool;  // ordinary paths only (never correlated paths)
var<private> rs_depth: u32;
var<private> d0_l: vec3<f32>;   // direct contribution at the first vertex (indirect mode)
var<private> d0_dl: vec3<f32>;
var<private> rs_stop: bool;     // indirect mode: cell read, stop after the vertex's direct lighting
var<private> pm_wo: vec3<f32>;      // first vertex of radiance(): outgoing direction, material
var<private> pm_rough: f32;
var<private> pm_metal: f32;
var<private> pm_diffuse_only: bool; // KRC cells: only the diffuse lobe at the first vertex
var<private> nee_spec_only: bool;   // vertex read back from the cache: only the direct specular is missing
var<private> n_segs: u32;
var<private> sp_cell: u32;
var<private> sp_prefix: vec3<f32>;
var<private> sp_beta: vec3<f32>;
var<private> sp_albedo: vec3<f32>;
var<private> rs_hit: bool;
var<private> reuse: bool;
var<private> ru_cell: u32;       // cell of the 1st bounce (0xffffffff: none)
var<private> ru_prefix: vec3<f32>;
var<private> ru_beta: vec3<f32>;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rnd() -> f32 {
    rng = rng * 747796405u + 2891336453u;
    let s = rng;
    var w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    w = (w >> 22u) ^ w;
    return f32(w >> 8u) * (1.0 / 16777216.0);
}

fn seed_rng(a: u32, b: u32, c: u32) {
    rng = pcg(a ^ pcg(b ^ pcg(c + fr.seed)));
}

// radiance sums per cell in fixed point (a plain atomicAdd, like SHaRC): float CAS
// loops jostle when dozens of paths target the same cell
const FX: f32 = 16384.0;
fn add_fx(i: u32, v: f32) {
    let q = u32(max(v, 0.0) * FX + 0.5);
    if (q != 0u) { atomicAdd(&acc[i], q); }
}
fn ld_fx(i: u32) -> f32 { return f32(atomicLoad(&acc[i])) / FX; }

fn add_f32(i: u32, v: f32) {
    var old = atomicLoad(&acc[i]);
    loop {
        let r = atomicCompareExchangeWeak(&acc[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}

fn ld_f32(i: u32) -> f32 {
    return bitcast<f32>(atomicLoad(&acc[i]));
}

fn n_lights() -> u32 { return fr.n_lights; }
fn light_at(j: u32) -> Light { if (sel == 0u) { return lights[j]; } return lights_prev[j]; }

// ---------------------------------------------------------------- regions
fn cell_of(p: vec3<f32>) -> vec3<i32> {
    return clamp(vec3<i32>(floor((p - ORIGIN) / REGION)), vec3<i32>(0), DIMS - vec3<i32>(1));
}
fn index_of(c: vec3<i32>) -> u32 { return u32((c.z * DIMS.y + c.y) * DIMS.x + c.x); }
fn super_of(c: vec3<i32>) -> u32 {
    return u32(((c.z / SDIV.z) * SDIMS.y + c.y / SDIV.y) * SDIMS.x + c.x / SDIV.x);
}
fn set2(m: vec2<u32>, s: u32) -> vec2<u32> {
    if (s < 32u) { return vec2<u32>(m.x | (1u << s), m.y); }
    return vec2<u32>(m.x, m.y | (1u << (s - 32u)));
}

fn mark_vertex(p: vec3<f32>) {
    if (rec_on) { m_vsup = set2(m_vsup, super_of(cell_of(p))); }
}

// Super-region grid (stored dependencies): blocks of SDIV regions.
const SUP: vec3<f32> = vec3<f32>(SDIV) * REGION;
var<private> fine_dda: bool;

// DDA: super-regions crossed (L2 dependencies). On the fine grid (0.6 m) only when the exact
// contact test with the event is needed too (correlated paths); otherwise on the coarse grid,
// ~4× fewer steps.
fn mark_segment(a: vec3<f32>, b: vec3<f32>) {
    if (!rec_on) { return; }
    let d = b - a;
    let cs = select(SUP, vec3<f32>(REGION), fine_dda);
    let dims = select(SDIMS, DIMS, fine_dda);
    let lo = ORIGIN;
    let hi = ORIGIN + vec3<f32>(dims) * cs;
    var t0 = 0.0;
    var t1 = 1.0;
    for (var ax = 0; ax < 3; ax++) {
        if (abs(d[ax]) < 1e-12) {
            if (a[ax] < lo[ax] || a[ax] > hi[ax]) { return; }
        } else {
            let ta = (lo[ax] - a[ax]) / d[ax];
            let tb = (hi[ax] - a[ax]) / d[ax];
            t0 = max(t0, min(ta, tb));
            t1 = min(t1, max(ta, tb));
        }
    }
    if (t0 > t1) { return; }
    let p = a + d * t0;
    var c = clamp(vec3<i32>(floor((p - ORIGIN) / cs)), vec3<i32>(0), dims - vec3<i32>(1));
    var stp = vec3<i32>(0);
    var tmax = vec3<f32>(1e30);
    var tdelta = vec3<f32>(1e30);
    for (var ax = 0; ax < 3; ax++) {
        if (abs(d[ax]) >= 1e-12) {
            let pos = d[ax] > 0.0;
            stp[ax] = select(-1, 1, pos);
            let boundary = ORIGIN[ax] + f32(c[ax] + select(0, 1, pos)) * cs[ax];
            tmax[ax] = t0 + (boundary - p[ax]) / d[ax];
            tdelta[ax] = cs[ax] / abs(d[ax]);
        }
    }
    let test_ev = fine_dda && (fr.flags & F_EVENT) != 0u;
    for (var it = 0; it < 256; it++) {
        m_trav = set2(m_trav, select(u32((c.z * SDIMS.y + c.y) * SDIMS.x + c.x), super_of(c), fine_dda));
        if (test_ev && !m_aff) {
            let idx = index_of(c);
            if ((ev[idx >> 5u] >> (idx & 31u) & 1u) == 1u) { m_aff = true; }
        }
        var ax = 2;
        if (tmax.x < tmax.y) {
            if (tmax.x < tmax.z) { ax = 0; }
        } else if (tmax.y < tmax.z) {
            ax = 1;
        }
        if (tmax[ax] > t1) { break; }
        c[ax] += stp[ax];
        if (c[ax] < 0 || c[ax] >= dims[ax]) { break; }
        tmax[ax] += tdelta[ax];
    }
}

// ---------------------------------------------------------------- geometry
// albedo: base colour; rough ≥ 1 and metal = 0: pure Lambertian (scenes from before the showcase)
struct Hit { found: bool, t: f32, p: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>, light: i32, rough: f32, metal: f32 };
struct Occ { found: bool, t: f32 };

// <geometry> (axis-aligned boxes; replaced by the triangles + ray query version, see rt.rs)
@group(0) @binding(1) var<storage, read> boxes: array<Box>;
@group(0) @binding(3) var<storage, read> boxes_prev: array<Box>;

fn box_at(i: u32) -> Box { if (sel == 0u) { return boxes[i]; } return boxes_prev[i]; }

fn safe_dir(d: vec3<f32>) -> vec3<f32> {
    let s = select(vec3<f32>(1.0), vec3<f32>(-1.0), d < vec3<f32>(0.0));
    return select(d, s * 1e-12, abs(d) < vec3<f32>(1e-12));
}

fn slab(mn: vec3<f32>, mx: vec3<f32>, o: vec3<f32>, inv: vec3<f32>) -> vec3<f32> {
    let ta = (mn - o) * inv;
    let tb = (mx - o) * inv;
    let tlo = min(ta, tb);
    let thi = max(ta, tb);
    var t0 = tlo.x;
    var axis = 0.0;
    if (tlo.y > t0) { t0 = tlo.y; axis = 1.0; }
    if (tlo.z > t0) { t0 = tlo.z; axis = 2.0; }
    return vec3<f32>(t0, min(thi.x, min(thi.y, thi.z)), axis);
}

// first hit on the geometry (without the lamps)
fn scene_hit(o: vec3<f32>, d: vec3<f32>) -> Hit {
    var h: Hit;
    h.found = false;
    h.t = 1e30;
    h.light = -1;
    h.rough = 1.0;
    h.metal = 0.0;
    let inv = 1.0 / safe_dir(d);
    var axis = 0u;
    var id = 0u;
    for (var i = 0u; i < fr.n_boxes; i++) {
        let b = box_at(i);
        let s = slab(b.mn, b.mx, o, inv);
        if (s.x <= s.y && s.x > EPS && s.x < h.t) {
            h.t = s.x;
            id = i;
            axis = u32(s.z);
            h.found = true;
        }
    }
    if (h.found) {
        var nn = vec3<f32>(0.0);
        nn[axis] = select(1.0, -1.0, d[axis] > 0.0);
        h.n = nn;
        h.albedo = box_at(id).albedo;
    }
    return h;
}

fn occluder(o: vec3<f32>, d: vec3<f32>, tmax: f32) -> Occ {
    var r: Occ;
    r.found = false;
    r.t = tmax;
    let inv = 1.0 / safe_dir(d);
    for (var i = 0u; i < fr.n_boxes; i++) {
        let b = box_at(i);
        let s = slab(b.mn, b.mx, o, inv);
        if (s.x <= s.y && s.x > EPS && s.x < r.t && s.y > 0.0) {
            r.found = true;
            r.t = s.x;
        }
    }
    return r;
}
// </geometry>

// first hit, lamps included
fn intersect(o: vec3<f32>, d: vec3<f32>) -> Hit {
    var h = scene_hit(o, d);
    for (var j = 0u; j < n_lights(); j++) {
        if (abs(d.y) < 1e-12) { continue; }
        let l = light_at(j);
        let t = (l.center.y - o.y) / d.y;
        if (t > EPS && t < h.t) {
            let p = o + d * t;
            if (abs(p.x - l.center.x) <= l.hx && abs(p.z - l.center.z) <= l.hz) {
                h.t = t;
                h.light = i32(j);
                h.found = true;
                h.n = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0), d.y > 0.0);
                h.albedo = vec3<f32>(0.0);
                h.rough = 1.0;
                h.metal = 0.0;
            }
        }
    }
    if (h.found) { h.p = o + d * h.t; }
    return h;
}

// ---------------------------------------------------------------- materials (showcase)
// BRDF: Lambertian (1 − metal)·base/π + GGX (Smith, Schlick, F0 = mix(0.04, base, metal)).
// A pure Lambertian material (rough ≥ 1, metal = 0) follows exactly the old code path,
// random draws included.
fn g_rough(g: Gpx) -> f32 { return select(1.0, unpack2x16unorm(g.p0).x, g.kind == 2u); }
fn g_metal(g: Gpx) -> f32 { return select(0.0, unpack2x16unorm(g.p0).y, g.kind == 2u); }
fn is_diffuse(rough: f32, metal: f32) -> bool { return rough >= 0.999 && metal <= 0.0; }
// path regularisation (game mode, both methods; the reference stays exact): beyond
// the first vertex, a glossy surface is treated as rough of at least PATH_REG; the
// caustics (diffuse → mirror → small lamp), almost impossible to sample, disappear
const PATH_REG: f32 = 0.3;
fn reg_rough(r: f32) -> f32 {
    if ((fr.flags & F_GAME) == 0u || (fr.off & 8u) != 0u) { return r; }
    return max(r, PATH_REG);
}
fn diff_albedo(base: vec3<f32>, metal: f32) -> vec3<f32> { return base * (1.0 - metal); }
fn f0_of(base: vec3<f32>, metal: f32) -> vec3<f32> { return mix(vec3<f32>(0.04), base, metal); }
fn fresnel(f0: vec3<f32>, vh: f32) -> vec3<f32> { return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - clamp(vh, 0.0, 1.0), 5.0); }
fn ggx_a2(rough: f32) -> f32 { let a = max(rough * rough, 0.002); return a * a; }
fn ggx_d(nh: f32, a2: f32) -> f32 { let t = nh * nh * (a2 - 1.0) + 1.0; return a2 / (PI * t * t); }
fn smith_g1(nv: f32, a2: f32) -> f32 { return 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv)); }

// BRDF × cos, split into diffuse and specular parts
struct Brdf2 { d: vec3<f32>, s: vec3<f32> };
fn brdf_cos(n: vec3<f32>, wo: vec3<f32>, wi: vec3<f32>, base: vec3<f32>, rough: f32, metal: f32) -> Brdf2 {
    var r: Brdf2;
    r.d = vec3<f32>(0.0);
    r.s = vec3<f32>(0.0);
    let nl = dot(n, wi);
    if (nl <= 0.0) { return r; }
    r.d = diff_albedo(base, metal) / PI * nl;
    if (is_diffuse(rough, metal)) { return r; }
    let nv = dot(n, wo);
    if (nv <= 0.0) { return r; }
    let h = normalize(wo + wi);
    let a2 = ggx_a2(rough);
    let d = ggx_d(max(dot(n, h), 0.0), a2);
    let g = smith_g1(nv, a2) * smith_g1(nl, a2);
    r.s = fresnel(f0_of(base, metal), dot(wo, h)) * (d * g / (4.0 * nv));
    return r;
}

fn tangent_frame(n: vec3<f32>) -> mat3x3<f32> {
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    let t = normalize(cross(n, a));
    return mat3x3<f32>(t, cross(n, t), n);
}

// probability of drawing the specular lobe
fn spec_prob(n: vec3<f32>, wo: vec3<f32>, base: vec3<f32>, rough: f32, metal: f32) -> f32 {
    if (is_diffuse(rough, metal)) { return 0.0; }
    let fs = lum(fresnel(f0_of(base, metal), max(dot(n, wo), 0.0)));
    let fd = lum(diff_albedo(base, metal));
    return clamp(fs / max(fs + fd, 1e-4), 0.1, 0.95);
}

// GGX direction (half-vector drawn according to D), reflected around the half-vector
fn ggx_dir(n: vec3<f32>, wo: vec3<f32>, rough: f32) -> vec3<f32> {
    let a2 = ggx_a2(rough);
    let u1 = rnd();
    let u2 = rnd();
    let ct = sqrt((1.0 - u2) / (1.0 + (a2 - 1.0) * u2));
    let st = sqrt(max(0.0, 1.0 - ct * ct));
    let phi = 2.0 * PI * u1;
    let h = tangent_frame(n) * vec3<f32>(st * cos(phi), st * sin(phi), ct);
    return reflect(-wo, h);
}

fn pdf_spec(n: vec3<f32>, wo: vec3<f32>, wi: vec3<f32>, rough: f32) -> f32 {
    let h = normalize(wo + wi);
    let vh = dot(wo, h);
    if (vh <= 0.0) { return 0.0; }
    return ggx_d(max(dot(n, h), 0.0), ggx_a2(rough)) * max(dot(n, h), 0.0) / (4.0 * vh);
}

// BSDF sample: direction, weight f·cos/pdf (diffuse + specular parts), lobe drawn
struct Bs { ok: bool, wi: vec3<f32>, w: vec3<f32>, spec: bool, lobe_rough: f32 };
fn bsdf_sample(n: vec3<f32>, wo: vec3<f32>, base: vec3<f32>, rough: f32, metal: f32, diffuse_only: bool) -> Bs {
    var b: Bs;
    b.ok = true;
    b.spec = false;
    b.lobe_rough = 1.0;
    if (is_diffuse(rough, metal) || diffuse_only) {
        b.wi = cosine_dir(n);
        b.w = diff_albedo(base, metal);
        return b;
    }
    let ps = spec_prob(n, wo, base, rough, metal);
    if (rnd() < ps) {
        b.wi = ggx_dir(n, wo, rough);
        b.spec = true;
        b.lobe_rough = rough;
    } else {
        b.wi = cosine_dir(n);
    }
    let nl = dot(n, b.wi);
    if (nl <= 0.0) {
        b.ok = false;
        return b;
    }
    let pdf = (1.0 - ps) * nl / PI + ps * pdf_spec(n, wo, b.wi, rough);
    let f = brdf_cos(n, wo, b.wi, base, rough, metal);
    b.w = (f.d + f.s) / max(pdf, 1e-8);
    return b;
}

// sample of the specular lobe only (per-pixel reflections in game mode)
fn spec_sample(n: vec3<f32>, wo: vec3<f32>, base: vec3<f32>, rough: f32, metal: f32) -> Bs {
    var b: Bs;
    b.spec = true;
    b.lobe_rough = rough;
    b.wi = ggx_dir(n, wo, rough);
    b.ok = dot(n, b.wi) > 0.0;
    if (!b.ok) { return b; }
    let f = brdf_cos(n, wo, b.wi, base, rough, metal);
    b.w = f.s / max(pdf_spec(n, wo, b.wi, rough), 1e-8);
    return b;
}

// MIS (balance heuristic) at the first vertex in game mode, between sampling a lamp and
// sampling the BSDF: a mirror seeing an extended lamp (the sky) almost never reaches it through
// lamp sampling (tiny lobe), hence white dots; the reflection ray hitting the
// lamp then carries most of it. mis_mode 0: no MIS; 1: specular lobe only (KRC's per-pixel
// reflections, only the specular part is weighted); 2: full BSDF (SHaRC Render).
var<private> mis_mode: u32;

fn mis_bsdf_pdf(n: vec3<f32>, wo: vec3<f32>, wi: vec3<f32>, base: vec3<f32>, rough: f32, metal: f32) -> f32 {
    let nl = max(dot(n, wi), 0.0);
    if (mis_mode == 1u) { return pdf_spec(n, wo, wi, rough); }
    if (is_diffuse(rough, metal)) { return nl / PI; }
    let ps = spec_prob(n, wo, base, rough, metal);
    return (1.0 - ps) * nl / PI + ps * pdf_spec(n, wo, wi, rough);
}

// density (solid angle) of lamp sampling for a ray hitting lamp j at distance t
fn light_hit_pdf(j: u32, t: f32, wi: vec3<f32>) -> f32 {
    var total = 0.0;
    for (var i = 0u; i < n_lights(); i++) { total += light_power(i); }
    let li = light_at(j);
    if (total <= 0.0 || wi.y <= 0.0) { return 0.0; }
    return light_power(j) / total * t * t / (wi.y * 4.0 * li.hx * li.hz);
}

// light of a lamp hit by the BSDF ray at the first vertex, weighted by MIS
fn mis_light_hit(j: u32, t: f32, n: vec3<f32>, wo: vec3<f32>, wi: vec3<f32>, base: vec3<f32>, rough: f32, metal: f32) -> vec3<f32> {
    let li = light_at(j);
    if (li.on < 0.5 || wi.y <= 0.0) { return vec3<f32>(0.0); }
    let pb = mis_bsdf_pdf(n, wo, wi, base, rough, metal);
    let pl = light_hit_pdf(j, t, wi);
    return li.emission * (pb / max(pb + pl, 1e-12));
}

// direct lighting of a vertex (one lamp drawn according to its power), diffuse and specular parts
fn nee_split(p: vec3<f32>, n: vec3<f32>, wo: vec3<f32>, base: vec3<f32>, rough: f32, metal: f32) -> Brdf2 {
    var r: Brdf2;
    r.d = vec3<f32>(0.0);
    r.s = vec3<f32>(0.0);
    let origin = p + n * OFFSET;
    var total = 0.0;
    for (var j = 0u; j < n_lights(); j++) { total += light_power(j); }
    if (total <= 0.0) { return r; }
    var u = rnd() * total;
    var s = n_lights();
    var pmf = 0.0;
    for (var j = 0u; j < n_lights(); j++) {
        let pw = light_power(j);
        if (pw > 0.0 && (u < pw || j == n_lights() - 1u)) {
            s = j;
            pmf = pw / total;
            break;
        }
        u -= pw;
    }
    if (s >= n_lights()) { return r; }
    let li = light_at(s);
    let q = li.center + vec3<f32>((2.0 * rnd() - 1.0) * li.hx, 0.0, (2.0 * rnd() - 1.0) * li.hz);
    let w = q - origin;
    let dist2 = dot(w, w);
    let dist = sqrt(dist2);
    let wd = w / dist;
    let cos_l = wd.y;
    if (dot(n, wd) <= 0.0 || cos_l <= 0.0 || li.on < 0.5) { return r; }
    if (occluder(origin, wd, dist - 1e-4).found) { return r; }
    let f = brdf_cos(n, wo, wd, base, rough, metal);
    let k = li.emission * (cos_l / dist2 * (4.0 * li.hx * li.hz) / pmf);
    r.d = f.d * k;
    r.s = f.s * k;
    if (mis_mode != 0u) {
        let pl = pmf * dist2 / (cos_l * 4.0 * li.hx * li.hz);
        let wl = pl / max(pl + mis_bsdf_pdf(n, wo, wd, base, rough, metal), 1e-12);
        r.s *= wl;
        if (mis_mode == 2u) { r.d *= wl; }
    }
    return r;
}

fn cosine_dir(n: vec3<f32>) -> vec3<f32> {
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    let t = normalize(cross(n, a));
    let b = cross(n, t);
    let r = sqrt(rnd());
    let phi = 2.0 * PI * rnd();
    let z = sqrt(max(0.0, 1.0 - r * r));
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * z);
}

fn light_power(j: u32) -> f32 {
    let l = light_at(j);
    if (uni || l.on > 0.5) { return l.power; }
    return 0.0;
}

fn toggle_w(j: u32) -> f32 {
    if (!split) { return 0.0; }
    return bitcast<f32>(ev[146u + j]);
}

struct LL { l: vec3<f32>, dl: vec3<f32> };

// material and outgoing direction (toward the camera) of radiance()'s first vertex
fn set_primary(g: Gpx, diffuse_only: bool) {
    pm_wo = normalize(fr.eye - g.p);
    pm_rough = g_rough(g);
    pm_metal = g_metal(g);
    pm_diffuse_only = diffuse_only;
}

struct CacheHit { found: bool, mean: vec3<f32>, deps: vec2<u32> };

// index of the existing cell at point (p, n) (jitter drawn at random), or 0xffffffff
fn cell_at(p: vec3<f32>, n: vec3<f32>, dist: f32) -> u32 {
    let key = key_at(p, n, dist, vec2<f32>(rnd() - 0.5, rnd() - 0.5));
    var h = key & fr.table_mask;
    for (var probe = 0u; probe < 64u; probe++) {
        let kk = atomicLoad(&keys[h]);
        if (kk == key) { return atomicLoad(&vals[h]) - 1u; }
        if (kk == 0u) { break; }
        h = (h + 1u) & fr.table_mask;
    }
    return 0xffffffffu;
}

// existing cell at point (p, n), with enough samples
fn cache_at(p: vec3<f32>, n: vec3<f32>, dist: f32, albedo: vec3<f32>) -> CacheHit {
    var r: CacheHit;
    r.found = false;
    let c1 = cell_at(p, n, dist);
    {
        {
            let c = c1 + 1u;
            if (c == 0u) { return r; }
            // fields read one by one: the whole cell (144 bytes) is expensive in the tracing loop
            if (cells[c - 1u].n < RESAMPLE_N_MIN) { return r; }
            r.found = true;
            r.mean = cells[c - 1u].mean * demod(albedo);
            if (rec_on) {
                let b = (c - 1u) * ACC;
                r.deps = cells[c - 1u].deps_prev | vec2<u32>(atomicLoad(&acc[b + A_DEPS]), atomicLoad(&acc[b + A_DEPS + 1u]));
            }
            return r;
        }
    }
}

fn radiance(p0: vec3<f32>, n0: vec3<f32>, a0: vec3<f32>) -> LL {
    var p = p0;
    var n = n0;
    var albedo = a0;
    var beta = vec3<f32>(1.0);
    var res: LL;
    res.l = vec3<f32>(0.0);
    res.dl = vec3<f32>(0.0);
    rs_stop = false;
    nee_spec_only = false;
    d0_l = vec3<f32>(0.0);
    d0_dl = vec3<f32>(0.0);
    var wo = pm_wo;
    var rough = select(pm_rough, 1.0, pm_rough <= 0.0 && pm_metal <= 0.0);
    var metal = pm_metal;
    mark_vertex(p0);
    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        let origin = p + n * OFFSET;
        var total = 0.0;
        for (var j = 0u; j < n_lights(); j++) { total += light_power(j); }
        if (total > 0.0) {
            var u = rnd() * total;
            var s = n_lights();
            var pmf = 0.0;
            for (var j = 0u; j < n_lights(); j++) {
                let pw = light_power(j);
                if (pw > 0.0 && (u < pw || j == n_lights() - 1u)) {
                    s = j;
                    pmf = pw / total;
                    break;
                }
                u -= pw;
            }
            if (s < n_lights()) {
                if (rec_on) { m_lights = m_lights | (1u << s); }
                let li = light_at(s);
                let q = li.center + vec3<f32>((2.0 * rnd() - 1.0) * li.hx, 0.0, (2.0 * rnd() - 1.0) * li.hz);
                let w = q - origin;
                let dist2 = dot(w, w);
                let dist = sqrt(dist2);
                let wd = w / dist;
                let cos_x = dot(n, wd);
                let cos_l = wd.y;
                let on = li.on > 0.5;
                let tw = toggle_w(s);
                if (cos_x > 0.0 && cos_l > 0.0 && !on && tw == 0.0) {
                    mark_segment(origin, q);
                } else if (cos_x > 0.0 && cos_l > 0.0) {
                    let occ = occluder(origin, wd, dist - 1e-4);
                    if (occ.found) {
                        mark_segment(origin, origin + wd * occ.t);
                    } else {
                        mark_segment(origin, q);
                        var c: vec3<f32>;
                        if (is_diffuse(rough, metal)) {
                            let g = cos_x * cos_l / dist2 * (4.0 * li.hx * li.hz) / pmf / PI;
                            c = beta * albedo * li.emission * g;
                        } else {
                            let f = brdf_cos(n, wo, wd, albedo, rough, metal);
                            var fc = f.d + f.s;
                            if (depth == 0u && pm_diffuse_only) { fc = f.d; }
                            if (nee_spec_only) { fc = f.s; }
                            c = beta * fc * li.emission * (cos_l / dist2 * (4.0 * li.hx * li.hz) / pmf);
                        }
                        if (on) { res.l += c; }
                        res.dl += c * tw;
                    }
                }
            }
        }
        if (depth == 0u) {
            d0_l = res.l;
            d0_dl = res.dl;
        }
        if (rs_stop) { break; }
        var d: vec3<f32>;
        if (is_diffuse(rough, metal)) {
            d = cosine_dir(n);
            beta = beta * albedo;
        } else {
            let bs = bsdf_sample(n, wo, albedo, rough, metal, depth == 0u && pm_diffuse_only);
            if (!bs.ok) { break; }
            d = bs.wi;
            beta = beta * bs.w;
        }
        n_segs++;
        let h = intersect(origin, d);
        if (!h.found) {
            mark_segment(origin, origin + d * MISS_LEN);
            break;
        }
        mark_segment(origin, h.p);
        if (h.light >= 0) { break; }
        mark_vertex(h.p);
        p = h.p;
        n = h.n;
        albedo = h.albedo;
        rough = reg_rough(h.rough);
        metal = h.metal;
        wo = -d;
        // reuse: the visible cell of the 1st bounce (far enough not to be its own)
        if (reuse && depth == 0u) {
            let dist = length(p - fr.eye);
            if (h.t > cell_size(dist) * 1.7320508) {
                let c1 = cell_at(p, n, dist);
                if (c1 != 0xffffffffu && atomicLoad(&acc[c1 * ACC + A_FOOT]) > 0u) {
                    ru_cell = c1;
                    ru_prefix = res.l;
                    ru_beta = beta;
                }
            }
        }
        // resampling: at vertex rs_depth (1 or 2), if a cell exists far enough (the
        // segment goes beyond the cell, so as not to read itself back), its mean is taken; the
        // cell read passes on its dependencies (event detection stays conservative)
        if (resample && depth + 1u >= rs_depth && sp_cell == 0xffffffffu) {
            let dist = length(p - fr.eye);
            if (h.t > cell_size(dist) * 1.7320508) {
                let jit = vec2<f32>(rnd() - 0.5, rnd() - 0.5);
                if ((fr.flags & F_SUPPORT) != 0u) {
                    // support cell: found and mature → stop; otherwise created or fed
                    let key = key_at(p, n, dist, jit);
                    let sc = insert_key(key);
                    if (sc != 0xffffffffu && atomicLoad(&vals[(cells[sc].slot - 1u) & fr.table_mask]) != 0u
                        && cells[sc].n >= RESAMPLE_N_MIN) {
                        if (depth == 0u) { ru_cell = 0xffffffffu; }
                        rs_hit = true;
                        res.l += beta * cells[sc].mean * demod(albedo);
                        if (rec_on) {
                            let b = sc * ACC;
                            let dps = cells[sc].deps_prev | vec2<u32>(atomicLoad(&acc[b + A_DEPS]), atomicLoad(&acc[b + A_DEPS + 1u]));
                            m_trav = m_trav | dps;
                            m_vsup = m_vsup | dps;
                        }
                        break;
                    }
                    if (sc != 0xffffffffu) {
                        sp_cell = sc;
                        sp_prefix = res.l;
                        sp_beta = beta;
                        sp_albedo = albedo;
                    }
                }
                var ch: CacheHit;
                ch.found = false;
                if ((fr.flags & F_SUPPORT) == 0u) { ch = cache_at(p, n, dist, diff_albedo(albedo, metal)); }
                if (ch.found) {
                    // stopping on the cache at the 1st bounce: the rest is not a fresh sample of the
                    // cell hit (SHaRC does not count a sample in that case either)
                    if (depth == 0u) { ru_cell = 0xffffffffu; }
                    rs_hit = true;
                    res.l += beta * ch.mean;
                    if (rec_on) {
                        m_trav = m_trav | ch.deps;
                        m_vsup = m_vsup | ch.deps;
                    }
                    // "indirect only" cells: this vertex's direct lighting is missing,
                    // computed on the next round before stopping; glossy vertex: the cell carries only
                    // diffuse, the direct specular is missing
                    if ((fr.flags & F_INDIRECT) != 0u) {
                        rs_stop = true;
                    } else if (!is_diffuse(rough, metal)) {
                        rs_stop = true;
                        nee_spec_only = true;
                    } else {
                        break;
                    }
                }
            }
        }
        if (depth >= 2u && !rs_stop) {
            let q = clamp(max(beta.x, max(beta.y, beta.z)), 0.05, 0.95);
            if (rnd() > q) { break; }
            beta = beta / q;
        }
    }
    return res;
}

fn lum(c: vec3<f32>) -> f32 { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }

// records a path's dependencies in the cell
fn commit_deps(c: u32) {
    let b = c * ACC;
    if (m_trav.x != 0u) { atomicOr(&acc[b + A_DEPS], m_trav.x); }
    if (m_trav.y != 0u) { atomicOr(&acc[b + A_DEPS + 1u], m_trav.y); }
    atomicAdd(&acc[b + A_NS], 1u);
    // only the bits that are set (≈ 2 per path)
    for (var h = 0u; h < 2u; h++) {
        var m = m_vsup[h];
        while (m != 0u) {
            let s = h * 32u + countTrailingZeros(m);
            m &= m - 1u;
            atomicAdd(&acc[b + A_FREQ + (s >> 1u)], select(1u, 65536u, (s & 1u) == 1u));
        }
    }
}

// Task interleaving: the tasks of one cell are consecutive; a warp processing them
// together would fight over the same atomics. Thread g takes task
// (g mod 8)·S + g/8, with S = ⌈N/8⌉.
fn flat_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.y * nwg.x * 64u + gid.x;
}

fn stride_task(g: u32, n: u32) -> u32 {
    let s = (n + 7u) / 8u;
    return (g % 8u) * s + g / 8u;
}

fn reset_rec() {
    m_trav = vec2<u32>(0u);
    m_vsup = vec2<u32>(0u);
    m_lights = 0u;
    m_aff = false;
}

// ---------------------------------------------------------------- passes
@compute @workgroup_size(64)
fn gbuffer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= fr.w * fr.h) { return; }
    let i = k % fr.w;
    let j = k / fr.w;
    let sx = 2.0 * (f32(i) + 0.5) / f32(fr.w) - 1.0;
    let sy = 1.0 - 2.0 * (f32(j) + 0.5) / f32(fr.h);
    let d = normalize(fr.fwd + fr.right * (sx * fr.th) + fr.up * (sy * fr.tv));
    sel = 0u;
    let h = intersect(fr.eye, d);
    var g: Gpx;
    g.kind = 0u;
    if (h.found) {
        g.p = h.p;
        g.n = h.n;
        g.dist = h.t;
        if (h.light >= 0) {
            let l = lights[u32(h.light)];
            g.kind = 1u;
            g.albedo = select(vec3<f32>(0.0), l.emission, l.on > 0.5 && d.y > 0.0);
        } else {
            g.kind = 2u;
            g.albedo = h.albedo;
            g.p0 = pack2x16unorm(vec2<f32>(h.rough, h.metal));
            atomicAdd(&ctr[C_SURF], 1u);
        }
    }
    gbuf[k] = g;
}

fn jitter(k: u32, slot: u32) -> vec2<f32> {
    let h = pcg(k * 8u + slot + 0x5bd1e995u);
    return vec2<f32>(f32(h & 0xffffu) / 65536.0 - 0.5, f32((h >> 16u) & 0xffffu) / 65536.0 - 0.5);
}

fn cell_key(k: u32, slot: u32) -> u32 {
    let g = gbuf[k];
    return key_at(g.p, g.n, slot_dist(g.dist, slot), jitter(k, slot));
}

fn find_cell(key: u32) -> u32 {
    var h = key & fr.table_mask;
    for (var probe = 0u; probe < 64u; probe++) {
        let kk = atomicLoad(&keys[h]);
        if (kk == key) { return atomicLoad(&vals[h]) - 1u; }
        if (kk == 0u) { break; }
        h = (h + 1u) & fr.table_mask;
    }
    return 0xffffffffu;
}

// inserts a key (or finds it); returns the cell's index, or 0xffffffff if the table
// or the cell array is full
fn insert_key(key: u32) -> u32 {
    var h = key & fr.table_mask;
    for (var probe = 0u; probe < 64u; probe++) {
        let kk = atomicLoad(&keys[h]);
        if (kk == key) { return atomicLoad(&vals[h]) - 1u; }
        if (kk != 0u) {
            h = (h + 1u) & fr.table_mask;
            continue;
        }
        let r = atomicCompareExchangeWeak(&keys[h], 0u, key);
        if (r.exchanged) {
            let c = alloc_cell();
            if (c == 0xffffffffu) {
                atomicAdd(&ctr[C_OVERFLOW], 1u);
                return c;
            }
            cells[c].key = key;
            cells[c].slot = h + 1u;
            cells[c].seen = fr.clock;
            atomicStore(&vals[h], c + 1u);
            return c;
        }
        if (r.old_value == key) { return atomicLoad(&vals[h]) - 1u; }
        if (r.old_value == 0u) { probe--; continue; }
        h = (h + 1u) & fr.table_mask;
    }
    atomicAdd(&ctr[C_OVERFLOW], 1u);
    return 0xffffffffu;
}

// prior of a new cell: the cell of the coarser level, otherwise the finer one, at the same
// point (weight SEED_N at most), with its dependencies
fn seed_cell(c: u32, p: vec3<f32>, n: vec3<f32>, dist: f32, j: vec2<f32>) {
    for (var s = 0u; s < 2u; s++) {
        let pc = find_cell(key_at(p, n, select(dist * 2.0, dist * 0.5, s == 1u), j));
        if (pc == 0xffffffffu || pc == c) { continue; }
        let par = cells[pc];
        if (par.n < 1.0) { continue; }
        let ns = min(par.n, SEED_N);
        cells[c].mean = par.mean;
        cells[c].dmean = par.dmean;
        cells[c].lmean = par.lmean;
        cells[c].n = ns;
        cells[c].m2 = select(0.0, par.m2 / (par.n - 1.0) * (ns - 1.0), par.n > 1.0);
        let pb = pc * ACC;
        cells[c].deps_prev = par.deps_prev | vec2<u32>(atomicLoad(&acc[pb + A_DEPS]), atomicLoad(&acc[pb + A_DEPS + 1u]));
        cells[c].ns_prev = par.ns_prev;
        for (var w = 0u; w < 18u; w++) { cells[c].freq_prev[w] = par.freq_prev[w]; }
        return;
    }
}

fn cell_size(dist: f32) -> f32 { return exp2(floor(log2(fr.cell_px * dist * fr.px_angle))); }

// key of the cell of any point (size depending on the distance to the camera, jitter j)
fn key_at(p: vec3<f32>, n: vec3<f32>, dist: f32, j: vec2<f32>) -> u32 {
    let level = floor(log2(fr.cell_px * dist * fr.px_angle));
    let s = exp2(level);
    var axis = 2u;
    if (abs(n.x) > 0.5) { axis = 0u; } else if (abs(n.y) > 0.5) { axis = 1u; }
    var q: vec3<i32>;
    var ji = 0u;
    for (var a = 0u; a < 3u; a++) {
        let c = p[a] / s;
        if (a == axis) {
            q[a] = i32(round(c));
        } else {
            q[a] = i32(floor(c + j[ji]));
            ji++;
        }
    }
    let naxis = axis * 2u + select(0u, 1u, n[axis] > 0.0);
    var key = pcg(bitcast<u32>(i32(level)) ^ (naxis << 24u));
    key = pcg(key ^ bitcast<u32>(q.x));
    key = pcg(key ^ bitcast<u32>(q.y));
    key = pcg(key ^ bitcast<u32>(q.z));
    return clamp(key, 1u, TOMB - 1u);
}

// new cell: first the free list (filled by eviction, in another pass), otherwise
// at the end of the array
fn alloc_cell() -> u32 {
    // C_FREE does not change during insertion (it is only modified by eviction and by
    // prep_cells): a plain cursor is enough, without a CAS loop on a shared counter
    let n = atomicLoad(&ctr[C_FREE]);
    let i = atomicAdd(&ctr[C_POP], 1u);
    if (i < n) { return free_list[n - 1u - i]; }
    let c = atomicAdd(&ctr[C_CELLS], 1u);
    if (c >= fr.cell_cap) { return 0xffffffffu; }
    return c;
}

// eviction: tombstone in the table (linear probes go through it), cell reset to
// zero and given back to the free list
fn evict(c: u32) {
    let h = cells[c].slot - 1u;
    atomicStore(&keys[h], TOMB);
    atomicStore(&vals[h], 0u);
    clear_cell(c);
    cells[c].last = 0u;
    cells[c].seen = 0u;
    cells[c].key = 0u;
    cells[c].slot = 0u;
    let b = c * ACC;
    for (var w = 0u; w < ACC; w++) { atomicStore(&acc[b + w], 0u); }
    let i = atomicAdd(&ctr[C_FREE], 1u);
    free_list[i] = c;
    atomicAdd(&ctr[C_TOMB], 1u);
    atomicAdd(&ctr[C_EVICT], 1u);
}

// rebuild of the table when tombstones pile up: 1) clear, 2) reinsert
// the key of each live cell
@compute @workgroup_size(256)
fn rebuild_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h == 0u) { atomicStore(&ctr[C_TOMB], 0u); }
    if (h > fr.table_mask) { return; }
    atomicStore(&keys[h], 0u);
    atomicStore(&vals[h], 0u);
}

@compute @workgroup_size(64)
fn rebuild_insert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= min(atomicLoad(&ctr[C_CELLS]), fr.cell_cap) || cells[c].slot == 0u) { return; }
    let key = cells[c].key;
    var h = key & fr.table_mask;
    for (var probe = 0u; probe < 1024u; probe++) {
        let r = atomicCompareExchangeWeak(&keys[h], 0u, key);
        if (r.exchanged) {
            atomicStore(&vals[h], c + 1u);
            cells[c].slot = h + 1u;
            return;
        }
        if (r.old_value == 0u) { probe--; continue; }
        h = (h + 1u) & fr.table_mask;
    }
    atomicAdd(&ctr[C_OVERFLOW], 1u);
}

// 1st step: reserve the new keys (without spin-waiting)
@compute @workgroup_size(256)
fn insert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x / K;
    let slot = gid.x % K;
    if (k >= fr.w * fr.h || gbuf[k].kind != 2u) { return; }
    let key = cell_key(k, slot);
    var h = key & fr.table_mask;
    for (var probe = 0u; probe < 64u; probe++) {
        // read first: the key almost always exists, and a CAS on every probed slot
        // serialises the thousands of threads sharing a cell
        let kk = atomicLoad(&keys[h]);
        if (kk == key) { return; }
        if (kk != 0u) {
            h = (h + 1u) & fr.table_mask;
            continue;
        }
        let r = atomicCompareExchangeWeak(&keys[h], 0u, key);
        if (r.exchanged) {
            let c = alloc_cell();
            if (c == 0xffffffffu) {
                atomicAdd(&ctr[C_OVERFLOW], 1u);
                return;
            }
            cells[c].key = key;
            cells[c].slot = h + 1u;
            cells[c].seen = fr.clock;
            if ((fr.flags & F_SEED) != 0u) {
                let g = gbuf[k];
                seed_cell(c, g.p, g.n, slot_dist(g.dist, slot), jitter(k, slot));
            }
            atomicStore(&vals[h], c + 1u);
            return;
        }
        if (r.old_value == key) { return; }
        if (r.old_value == 0u) { probe--; continue; } // spurious failure of a weak CAS
        h = (h + 1u) & fr.table_mask;
    }
    atomicAdd(&ctr[C_OVERFLOW], 1u);
}

// 2nd step: find the cell again, footprint, reference pixel drawn uniformly
@compute @workgroup_size(256)
fn lookup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x / K;
    let slot = gid.x % K;
    if (k >= fr.w * fr.h) { return; }
    if (gbuf[k].kind != 2u) {
        pix_cells[gid.x] = 0xffffffffu;
        return;
    }
    let key = cell_key(k, slot);
    var h = key & fr.table_mask;
    var c = 0xffffffffu;
    for (var probe = 0u; probe < 64u; probe++) {
        let kk = atomicLoad(&keys[h]);
        if (kk == key) { c = atomicLoad(&vals[h]) - 1u; break; }
        if (kk == 0u) { break; }
        h = (h + 1u) & fr.table_mask;
    }
    pix_cells[gid.x] = c;
    if (c == 0xffffffffu) { return; }
    let b = c * ACC;
    if (atomicAdd(&acc[b + A_FOOT], 1u) == 0u) {
        vis_list[atomicAdd(&ctr[C_NVIS], 1u)] = c;
    }
    let r = pcg(gid.x ^ pcg(fr.frame + fr.seed)) >> 21u;
    atomicMax(&acc[b + A_REF], (r << 21u) | k);
}

fn ref_px(c: u32) -> u32 { return atomicLoad(&acc[c * ACC + A_REF]) & 0x1fffffu; }

fn freq_of(c: u32, s: u32) -> f32 {
    let b = c * ACC;
    let n = f32(atomicLoad(&acc[b + A_NS]) + cells[c].ns_prev);
    if (n == 0.0) { return 0.0; }
    let sh = select(0u, 16u, (s & 1u) == 1u);
    let cur = (atomicLoad(&acc[b + A_FREQ + (s >> 1u)]) >> sh) & 0xffffu;
    let prev = (cells[c].freq_prev[s >> 1u] >> sh) & 0xffffu;
    return min(f32(cur + prev) / n, 1.0);
}

fn clear_cell(c: u32) {
    let b = c * ACC;
    cells[c].n = 0.0;
    cells[c].lmean = 0.0;
    cells[c].m2 = 0.0;
    cells[c].mean = vec3<f32>(0.0);
    cells[c].rn = 0.0;
    cells[c].rl = 0.0;
    cells[c].rm2 = 0.0;
    cells[c].ns_prev = 0u;
    cells[c].deps_prev = vec2<u32>(0u);
    cells[c].dmean = vec3<f32>(0.0);
    for (var w = 0u; w < 18u; w++) {
        cells[c].freq_prev[w] = 0u;
        atomicStore(&acc[b + A_FREQ + w], 0u);
    }
    atomicStore(&acc[b + A_DEPS], 0u);
    atomicStore(&acc[b + A_DEPS + 1u], 0u);
    atomicStore(&acc[b + A_NS], 0u);
}

fn push_tasks(c: u32, m: u32, kind: u32) {
    let base = atomicAdd(&ctr[C_TASKS], m);
    let px = ref_px(c);
    for (var i = 0u; i < m; i++) {
        tasks[base + i] = vec4<u32>(c, px, pcg(base + i + fr.frame * 0x9e3779b9u), kind);
    }
}

// reset of the frame counters (the number of cells and the debt persist)
@compute @workgroup_size(64)
fn reset_ctr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= 256u || i == C_CELLS || i == C_DEBT || i == C_FREE || i == C_TOMB) { return; }
    atomicStore(&ctr[i], 0u);
}

// Amortisation: reserves m correlated paths under the frame's cap.
fn try_reserve(m: u32) -> bool {
    if ((fr.flags & F_AMORTIZE) == 0u) { return true; }
    let cap = u32(f32(atomicLoad(&ctr[C_SURF])) * fr.ppp * DELTA_CAP);
    let prev = atomicAdd(&ctr[C_DUSED], m);
    if (prev + m > cap) {
        atomicSub(&ctr[C_DUSED], m);
        return false;
    }
    return true;
}

// launches the correlated paths of a visible suspect cell, or puts it on hold
fn start_delta(c: u32, m: u32) {
    let b = c * ACC;
    if (try_reserve(m)) {
        atomicStore(&acc[b + A_SUSP], 1u);
        push_tasks(c, m, 1u);
    } else {
        atomicStore(&acc[b + A_SUSP], 2u);
        atomicAdd(&ctr[C_PENDING], 1u);
    }
}

// cells on hold (amortised lamp event): retried under this frame's cap
@compute @workgroup_size(64)
fn pending_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= min(atomicLoad(&ctr[C_CELLS]), fr.cell_cap)) { return; }
    let b = c * ACC;
    if (atomicLoad(&acc[b + A_SUSP]) != 2u) { return; }
    if (atomicLoad(&acc[b + A_FOOT]) == 0u) {
        clear_cell(c);
        atomicStore(&acc[b + A_SUSP], 0u);
        atomicAdd(&ctr[C_HARD], 1u);
        return;
    }
    start_delta(c, m_light());
}

fn m_light() -> u32 { return select(M_DELTA, fr.m_light, fr.m_light > 0u); }

// suspects: L2 dependencies AND (geometry, or expected energy effect > 1 %)
@compute @workgroup_size(64)
fn event_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= min(atomicLoad(&ctr[C_CELLS]), fr.cell_cap)) { return; }
    let b = c * ACC;
    // a cell still waiting for a previous event loses its history
    if (atomicLoad(&acc[b + A_SUSP]) == 2u) {
        clear_cell(c);
        atomicStore(&acc[b + A_SUSP], 0u);
        atomicAdd(&ctr[C_HARD], 1u);
        return;
    }
    if (cells[c].n == 0.0) { return; }
    let deps = vec2<u32>(atomicLoad(&acc[b + A_DEPS]), atomicLoad(&acc[b + A_DEPS + 1u])) | cells[c].deps_prev;
    let evl2 = vec2<u32>(ev[108], ev[109]);
    if (all((deps & evl2) == vec2<u32>(0u))) { return; }
    let geometry = (fr.flags & F_GEOMETRY) != 0u;
    if (!geometry) {
        var dl = 0.0;
        for (var s = 0u; s < 36u; s++) {
            let m = bitcast<f32>(ev[110u + s]);
            if (m > 0.0) { dl += freq_of(c, s) * m; }
        }
        if (dl <= fr.eps_energy * max(cells[c].lmean, 0.1)) { return; }
    }
    atomicAdd(&ctr[C_SUSPECTS], 1u);
    if (atomicLoad(&acc[b + A_FOOT]) == 0u) {
        clear_cell(c); // invisible: it cannot be measured, and that costs nothing
        atomicAdd(&ctr[C_HARD], 1u);
        return;
    }
    if (geometry) {
        atomicStore(&acc[b + A_SUSP], 1u);
        push_tasks(c, M_DELTA0, 1u);
    } else {
        start_delta(c, m_light());
    }
}

// arguments de dispatch indirect : round 0 = delta tour 1, 1 = delta tour 2, 2 = ordinaires
fn prep_indirect(round: u32) {
    let cur = atomicLoad(&ctr[C_TASKS]);
    var lo = 0u;
    if (round == 0u) {
        atomicStore(&ctr[C_D1], cur);
    } else if (round == 1u) {
        lo = atomicLoad(&ctr[C_D1]);
        atomicStore(&ctr[C_D2], cur);
    } else {
        lo = atomicLoad(&ctr[C_REG0]);
    }
    let n = cur - lo;
    // beyond 65,535 groups per dimension: 2D grid (index rebuilt by flat_id)
    let groups = (8u * ((n + 7u) / 8u) + 63u) / 64u;
    let gx = min(groups, 32768u);
    ind[round * 3u] = max(gx, 1u);
    ind[round * 3u + 1u] = (groups + gx - 1u) / max(gx, 1u);
    ind[round * 3u + 2u] = 1u;
}

// "per cell" passes: as many groups as real cells, not as capacity
@compute @workgroup_size(1)
fn prep_cells() {
    ind[12] = (atomicLoad(&ctr[C_NVIS]) + 63u) / 64u;
    ind[13] = 1u;
    ind[14] = 1u;
    let nf = atomicLoad(&ctr[C_FREE]);
    atomicStore(&ctr[C_FREE], nf - min(atomicLoad(&ctr[C_POP]), nf));
    atomicStore(&ctr[C_POP], 0u);
    let n = min(atomicLoad(&ctr[C_CELLS]), fr.cell_cap);
    ind[9] = (n + 63u) / 64u;
    ind[10] = 1u;
    ind[11] = 1u;
}

// after tracing: the list to resolve may have grown (support cells), and cells
// may have been taken from the free list
@compute @workgroup_size(1)
fn prep_vis2() {
    ind[12] = (atomicLoad(&ctr[C_NVIS]) + 63u) / 64u;
    let nf = atomicLoad(&ctr[C_FREE]);
    atomicStore(&ctr[C_FREE], nf - min(atomicLoad(&ctr[C_POP]), nf));
    atomicStore(&ctr[C_POP], 0u);
}

@compute @workgroup_size(1)
fn prep_d1() { prep_indirect(0u); }
@compute @workgroup_size(1)
fn prep_d2() { prep_indirect(1u); }
@compute @workgroup_size(1)
fn prep_reg() { prep_indirect(2u); }

// correlated paths: same seed in the current scene and the previous one
@compute @workgroup_size(64)
fn delta_trace1(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    delta_trace(flat_id(gid, nwg), 0u);
}
@compute @workgroup_size(64)
fn delta_trace2(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    delta_trace(flat_id(gid, nwg), 1u);
}

fn delta_trace(g: u32, round: u32) {
    var lo = 0u;
    var hi = atomicLoad(&ctr[C_D1]);
    if (round == 1u) { lo = hi; hi = atomicLoad(&ctr[C_D2]); }
    if (g >= 8u * ((hi - lo + 7u) / 8u)) { return; }
    let i = lo + stride_task(g, hi - lo);
    if (i >= hi) { return; }
    let t = tasks[i];
    let c = t.x;
    let gx = gbuf[t.y];
    let geometry = (fr.flags & F_GEOMETRY) != 0u;
    reset_rec();
    fine_dda = true;
    rec_on = true;
    uni = true;
    split = !geometry;
    sel = 0u;
    seed_rng(t.z, i, fr.frame);
    resample = (fr.flags & F_DRS) != 0u;
    rs_depth = select(1u, 2u, (pcg(t.z ^ 0x68e31da4u) & 1u) == 0u);
    let s0 = rng;
    set_primary(gx, true);
    let r = radiance(gx.p, gx.n, gx.albedo);
    var x = r.l;
    var y = r.l + r.dl;
    let x_d0 = d0_l;
    var y_d0 = d0_l + d0_dl;
    var affected = (m_lights & fr.ev_lights) != 0u;
    var retraced = 0u;
    if (geometry) {
        affected = affected || m_aff;
        y = x;
        if (affected) {
            let keep_trav = m_trav;
            let keep_vsup = m_vsup;
            rec_on = false;
            split = false;
            sel = 1u;
            rng = s0;
            y = radiance(gx.p, gx.n, gx.albedo).l;
            y_d0 = d0_l;
            sel = 0u;
            rec_on = true;
            m_trav = keep_trav;
            m_vsup = keep_vsup;
            retraced = 1u;
        }
    }
    commit_deps(c);
    if ((fr.flags & F_INDIRECT) != 0u) {
        x = x - x_d0;
        y = y - y_d0;
    }
    if ((fr.flags & F_DSPLIT) != 0u) {
        let pd = (x_d0 - y_d0) / demod(diff_albedo(gx.albedo, g_metal(gx)));
        if (any(pd != vec3<f32>(0.0))) {
            add_f32(c * ACC + A_PD, pd.x);
            add_f32(c * ACC + A_PD + 1u, pd.y);
            add_f32(c * ACC + A_PD + 2u, pd.z);
        }
    }
    x = x / demod(diff_albedo(gx.albedo, g_metal(gx)));
    y = y / demod(diff_albedo(gx.albedo, g_metal(gx)));
    let b = c * ACC;
    atomicAdd(&acc[b + A_PM], 1u);
    if (any(x != y)) { atomicStore(&acc[b + A_PDIFF], 1u); }
    if (affected) { atomicStore(&acc[b + A_PAFF], 1u); }
    add_f32(b + A_PX, x.x);
    add_f32(b + A_PX + 1u, x.y);
    add_f32(b + A_PX + 2u, x.z);
    add_f32(b + A_PY, y.x);
    add_f32(b + A_PY + 1u, y.y);
    add_f32(b + A_PY + 2u, y.z);
    let xl = lum(x);
    let yl = lum(y);
    add_f32(b + A_PXL, xl);
    add_f32(b + A_PYL, yl);
    add_f32(b + A_PXL2, xl * xl);
    add_f32(b + A_PYL2, yl * yl);
    add_f32(b + A_PXYL, xl * yl);
    atomicAdd(&ctr[C_DPATHS], 1u + retraced);
    atomicAdd(&ctr[C_DOLD], retraced);
}

// 2nd round (geometry): M_DELTA paths for cells one of whose paths was affected
@compute @workgroup_size(64)
fn delta_more(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= min(atomicLoad(&ctr[C_CELLS]), fr.cell_cap)) { return; }
    let b = c * ACC;
    if (atomicLoad(&acc[b + A_SUSP]) == 0u || atomicLoad(&acc[b + A_PAFF]) == 0u) { return; }
    push_tasks(c, M_DELTA - M_DELTA0, 1u);
}

// control variate: μ = x̄ − β (ȳ − μ_cache)
@compute @workgroup_size(64)
fn delta_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= min(atomicLoad(&ctr[C_CELLS]), fr.cell_cap)) { return; }
    let b = c * ACC;
    if (atomicLoad(&acc[b + A_SUSP]) != 1u) { return; }
    let m = f32(atomicLoad(&acc[b + A_PM]));
    let xr = vec3<f32>(ld_f32(b + A_PX), ld_f32(b + A_PX + 1u), ld_f32(b + A_PX + 2u)) / m;
    let yr = vec3<f32>(ld_f32(b + A_PY), ld_f32(b + A_PY + 1u), ld_f32(b + A_PY + 2u)) / m;
    let xl = ld_f32(b + A_PXL) / m;
    let yl = ld_f32(b + A_PYL) / m;
    if (atomicLoad(&acc[b + A_PDIFF]) == 0u) {
        // unchanged cell: the new samples are ordinary samples
        atomicAdd(&acc[b + A_FN], u32(m));
        add_fx(b + A_FR, xr.x * m);
        add_fx(b + A_FR + 1u, xr.y * m);
        add_fx(b + A_FR + 2u, xr.z * m);
        add_fx(b + A_FL, xl * m);
        add_f32(b + A_FL2, ld_f32(b + A_PXL2));
        atomicAdd(&ctr[C_DZERO], 1u);
        // None of the m paths saw the lamp: that is exact for those paths, not for the cell's
        // expectation (a small bright lamp may be seen only rarely). The estimate is
        // kept, but its weight capped, so that ordinary sampling brings it up to date.
        if ((fr.flags & (F_ZCAP | F_GEOMETRY)) == F_ZCAP) {
            let np = select(N_POST, fr.n_post, fr.n_post > 0.0);
            if (cells[c].n > np) {
                cells[c].m2 *= (np - 1.0) / (cells[c].n - 1.0);
                cells[c].n = np;
            }
        }
        return;
    }
    let sxx = max(ld_f32(b + A_PXL2) - m * xl * xl, 0.0);
    let syy = max(ld_f32(b + A_PYL2) - m * yl * yl, 0.0);
    let sxy = ld_f32(b + A_PXYL) - m * xl * yl;
    let mm = m * (m - 1.0);
    var cell = cells[c];
    var var_hist = 1e30;
    if (cell.n > 1.0) { var_hist = cell.m2 / (cell.n - 1.0) / cell.n; }
    var beta = 0.0;
    if (var_hist < 1e29) {
        let denom = syy / mm + var_hist;
        beta = select(1.0, clamp(sxy / mm / denom, 0.0, 1.0), denom > 0.0);
    }
    var var_mu = max(sxx - 2.0 * beta * sxy + beta * beta * syy, 0.0) / mm;
    if (beta > 0.0) { var_mu += beta * beta * var_hist; }
    let cell_old_l = cell.lmean;
    let pdm = vec3<f32>(ld_f32(b + A_PD), ld_f32(b + A_PD + 1u), ld_f32(b + A_PD + 2u)) / m;
    cell.dmean = max(cell.dmean + pdm, vec3<f32>(0.0));
    let mu = max(xr - (yr - cell.mean) * beta, vec3<f32>(0.0));
    let mu_l = max(xl - beta * (yl - cell.lmean), 0.0);
    var s2 = max(sxx / (m - 1.0), 1e-12);
    if (cell.n > 1.0) { s2 = max(s2, cell.m2 / (cell.n - 1.0) * select(1e-3, 1.0, (fr.flags & F_S2HIST) != 0u)); }
    var n_eff = cell.n + m;
    if (var_mu > 0.0) { n_eff = clamp(s2 / var_mu, min(m, 2.0), cell.n + m); }
    n_eff = min(n_eff, select(N_POST, fr.n_post, fr.n_post > 0.0));
    cell.n = n_eff;
    cell.mean = mu;
    cell.lmean = mu_l;
    cell.m2 = s2 * (n_eff - 1.0);
    // huge change: the history is useless, restart from the new samples alone
    // (the cell becomes a priority for allocation again)
    if (fr.reset_ratio > 0.0 && abs(mu_l - cell_old_l) > fr.reset_ratio * max(cell_old_l, 1e-6)) {
        cell.n = m;
        cell.mean = xr;
        cell.lmean = xl;
        cell.m2 = sxx;
    }
    cell.rn = 0.0;
    cell.rl = 0.0;
    cell.rm2 = 0.0;
    cell.last = fr.frame;
    cells[c] = cell;
}

fn rel_var(c: u32) -> f32 {
    let cell = cells[c];
    let v = select(0.0, cell.m2 / (cell.n - 1.0), cell.n > 1.0);
    return v / (cell.lmean * cell.lmean + REL_EPS);
}

@compute @workgroup_size(64)
fn alloc_hist(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= atomicLoad(&ctr[C_NVIS])) { return; }
    let c = vis_list[gid.x];
    // a cell on hold receives no ordinary samples (stale history)
    let foot = select(atomicLoad(&acc[c * ACC + A_FOOT]), 0u, atomicLoad(&acc[c * ACC + A_SUSP]) == 2u);
    let u = f32(pcg(c ^ pcg(fr.frame * 7u + fr.seed)) >> 8u) * (1.0 / 16777216.0);
    qbuf[c] = vec4<f32>(f32(foot) / f32(K), cells[c].n, rel_var(c), u);
    if (foot == 0u) { return; }
    atomicAdd(&ctr[C_VISIBLE], 1u);
    let ex = explore(c);
    if (ex > 0u) { atomicAdd(&ctr[C_EXPLORE], ex); }
    if (cells[c].n >= 8.0) {
        let bin = u32(clamp(floor((log2(max(rel_var(c), 1e-12)) + 20.0) * 2.0), 0.0, 63.0));
        atomicAdd(&ctr[C_HIST + bin], 1u);
        atomicAdd(&ctr[C_MATURE], 1u);
    }
}

fn post_var(n: f32, rel: f32, prior: f32) -> f32 {
    if (n < 1.0) { return prior; }
    return (KAPPA * prior + (n - 1.0) * rel) / (KAPPA + n - 1.0);
}

// number of samples for a threshold λ of marginal gain f·pv/((n+k)(n+k+1)) ≥ λ
fn quota(c: u32, lambda: f32, prior: f32) -> u32 {
    let q = qbuf[c];
    let foot = q.x;
    if (foot == 0.0) { return 0u; }
    let n = q.y;
    // Cap: with heavy-tailed radiance (small bright lamp, indirect), a cell that has just
    // drawn a rare value would see its variance explode and be immediately oversampled:
    // the rare value would be diluted in an n that grows because it was seen (bias toward dark).
    var rel = q.z;
    if (fr.rel_cap > 0.0) { rel = min(rel, fr.rel_cap * prior); }
    let pv = post_var(n, rel, prior);
    // target quality: a cell whose predicted relative error of the mean is under τ no
    // longer receives paths (only the exploration floor)
    if (fr.tau > 0.0 && n >= 1.0 && pv / n < fr.tau * fr.tau) { return 0u; }
    let t = (-1.0 + sqrt(1.0 + 4.0 * foot * pv / lambda)) * 0.5;
    // stochastic rounding (same draw at the sweep and at emission): Σ quotas follows the continuous solution
    var k = max(floor(t - n + q.w), 0.0);
    if (n < 1.0) { k = max(k, 1.0); }
    return min(u32(k), MAX_PER_CELL);
}

fn explore(c: u32) -> u32 {
    if (atomicLoad(&acc[c * ACC + A_FOOT]) == 0u || cells[c].n == 0.0) { return 0u; }
    if (atomicLoad(&acc[c * ACC + A_SUSP]) == 2u) { return 0u; }
    return select(0u, 1u, fr.frame - cells[c].last >= EXPLORE_PERIOD);
}

// Allocation: threshold λ of marginal gain such that Σ quota(λ) = budget. Three parallel sweeps
// of NCAND log-spaced candidates bracket λ (precision ≈ 2^(70/31³) ≈ 0.2 %): in the converged state,
// every cell is close to equilibrium and Σ quota varies by a factor 2 for 5 % on λ.
const NCAND: u32 = 32u;
var<workgroup> wsum: array<atomic<u32>, 32>;

@compute @workgroup_size(1)
fn alloc_prep() {
    var prior = 1.0;
    let mature = atomicLoad(&ctr[C_MATURE]);
    if (mature >= 16u) {
        var cum = 0u;
        for (var bin = 0u; bin < 64u; bin++) {
            cum += atomicLoad(&ctr[C_HIST + bin]);
            if (2u * cum >= mature) {
                prior = max(exp2((f32(bin) + 0.5) * 0.5 - 20.0), PRIOR_FLOOR);
                break;
            }
        }
    }
    let surf = atomicLoad(&ctr[C_SURF]);
    let debt = atomicLoad(&ctr[C_DEBT]) + atomicLoad(&ctr[C_DPATHS]);
    let budget = f32(surf) * fr.ppp;
    let rate = select(DEBT_RATE, fr.debt_rate, fr.debt_rate >= 0.0);
    let pay = min(debt, u32(budget * rate));
    atomicStore(&ctr[C_DEBT], debt - pay);
    scal[1] = prior;
    scal[2] = max(budget - f32(pay) - f32(atomicLoad(&ctr[C_EXPLORE])), 0.0);
    scal[3] = -40.0; // bornes de log2 λ
    scal[4] = 30.0;
}

fn cand(i: u32) -> f32 {
    return exp2(scal[3] + (scal[4] - scal[3]) * f32(i) / f32(NCAND - 1u));
}

@compute @workgroup_size(64)
fn alloc_scan1(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    alloc_scan(gid, lid, C_SCAN);
}
@compute @workgroup_size(64)
fn alloc_scan2(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    alloc_scan(gid, lid, C_SCAN + NCAND);
}

@compute @workgroup_size(64)
fn alloc_scan3(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    alloc_scan(gid, lid, C_SCAN + 2u * NCAND);
}

fn alloc_scan(gid: vec3<u32>, lid: vec3<u32>, base: u32) {
    if (lid.x < NCAND) { atomicStore(&wsum[lid.x], 0u); }
    workgroupBarrier();
    if (gid.x < atomicLoad(&ctr[C_NVIS])) {
        let c = vis_list[gid.x];
        let prior = scal[1];
        for (var i = 0u; i < NCAND; i++) {
            let q = quota(c, cand(i), prior);
            if (q > 0u) { atomicAdd(&wsum[i], q); }
        }
    }
    workgroupBarrier();
    if (lid.x < NCAND) {
        let v = atomicLoad(&wsum[lid.x]);
        if (v > 0u) { atomicAdd(&ctr[base + lid.x], v); }
    }
}

// keeps the smallest candidate whose sum fits in the budget, and tightens the bracket
@compute @workgroup_size(1)
fn alloc_pick1() { alloc_pick(C_SCAN); }
@compute @workgroup_size(1)
fn alloc_pick2() { alloc_pick(C_SCAN + NCAND); }
@compute @workgroup_size(1)
fn alloc_pick3() { alloc_pick(C_SCAN + 2u * NCAND); }

fn alloc_pick(base: u32) {
    let budget = scal[2];
    var i = NCAND - 1u;
    for (var j = 0u; j < NCAND; j++) {
        if (f32(atomicLoad(&ctr[base + j])) <= budget) { i = j; break; }
    }
    let lo = scal[3] + (scal[4] - scal[3]) * f32(max(i, 1u) - 1u) / f32(NCAND - 1u);
    let hi = scal[3] + (scal[4] - scal[3]) * f32(i) / f32(NCAND - 1u);
    scal[0] = exp2(hi);
    scal[3] = lo;
    scal[4] = hi;
    atomicStore(&ctr[C_REG0], atomicLoad(&ctr[C_TASKS]));
}

@compute @workgroup_size(64)
fn alloc_emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= atomicLoad(&ctr[C_NVIS])) { return; }
    let c = vis_list[gid.x];
    let k = quota(c, scal[0], scal[1]) + explore(c);
    if (k > 0u) { push_tasks(c, k, 0u); }
}

// SHaRC mode: one path per surface pixel, toward one of its K cells at random
@compute @workgroup_size(64)
fn sharc_emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= fr.w * fr.h || gbuf[k].kind != 2u) { return; }
    // ppp paths per pixel (stochastic rounding), each toward one of its K cells at random
    var h = pcg(k ^ pcg(fr.frame + fr.seed));
    let n = u32(floor(fr.ppp + f32(h >> 8u) * (1.0 / 16777216.0)));
    for (var j = 0u; j < n; j++) {
        h = pcg(h);
        let c = pix_cells[k * K + h % K];
        if (c == 0xffffffffu) { continue; }
        let i = atomicAdd(&ctr[C_TASKS], 1u);
        tasks[i] = vec4<u32>(c, k, pcg(i + fr.frame * 0x9e3779b9u), 0u);
    }
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let lo = atomicLoad(&ctr[C_REG0]);
    let hi = atomicLoad(&ctr[C_TASKS]);
    let g = flat_id(gid, nwg);
    if (g >= 8u * ((hi - lo + 7u) / 8u)) { return; }
    let i = lo + stride_task(g, hi - lo);
    if (i >= hi) { return; }
    let t = tasks[i];
    let c = t.x;
    let gx = gbuf[t.y];
    reset_rec();
    fine_dda = false;
    rec_on = (fr.flags & F_SHARC) == 0u && (pcg(t.z ^ 0x2545f491u) % max(fr.rec_every, 1u)) == 0u;
    uni = false;
    split = false;
    sel = 0u;
    seed_rng(t.z, i, fr.frame);
    resample = (fr.flags & F_RESAMPLE) != 0u;
    rs_depth = select(1u, 2u, rnd() < 0.5);
    reuse = (fr.flags & F_REUSE) != 0u;
    ru_cell = 0xffffffffu;
    n_segs = 0u;
    rs_hit = false;
    sp_cell = 0xffffffffu;
    set_primary(gx, true);
    let v_raw = radiance(gx.p, gx.n, gx.albedo).l;
    atomicAdd(&ctr[C_SEGS], n_segs);
    if (rs_hit) { atomicAdd(&ctr[C_RSHIT], 1u); }
    if (sp_cell != 0xffffffffu && all(sp_beta > vec3<f32>(0.0))) {
        // sample of the support cell: outgoing radiance at the vertex (rest of the path)
        let u = (v_raw - sp_prefix) / sp_beta / demod(sp_albedo);
        let bs = sp_cell * ACC;
        if (rec_on) { commit_deps(sp_cell); }
        if (atomicAdd(&acc[bs + A_FN], 1u) == 0u && atomicLoad(&acc[bs + A_FOOT]) == 0u) {
            vis_list[atomicAdd(&ctr[C_NVIS], 1u)] = sp_cell; // resolved with the visible cells
        }
        add_fx(bs + A_FR, u.x);
        add_fx(bs + A_FR + 1u, u.y);
        add_fx(bs + A_FR + 2u, u.z);
        let ls = lum(u);
        add_fx(bs + A_FL, ls);
        add_f32(bs + A_FL2, ls * ls);
    }
    let gda = diff_albedo(gx.albedo, g_metal(gx));
    var v = (v_raw - select(vec3<f32>(0.0), d0_l, (fr.flags & F_INDIRECT) != 0u)) / demod(gda);
    // game mode: firefly clamping (like games' caches) — a sample more than
    // FIREFLY_K times brighter than its cell's mean is brought back to that bound; otherwise
    // it makes a blotch the size of the cell, which the denoiser spreads
    var vscale = 1.0;
    if ((fr.flags & F_GAME) != 0u && (fr.off & 4u) == 0u && cells[c].n >= 8.0) {
        // young cell (after an event, a rare value weighs a lot): stricter bound
        let kf = select(FIREFLY_K, FIREFLY_K_YOUNG, cells[c].n < 64.0);
        let cap = kf * max(cells[c].lmean, 1e-4);
        let lv = lum(v);
        if (lv > cap) { vscale = cap / lv; }
    }
    v = v * vscale;
    if ((fr.flags & F_DSPLIT) != 0u) {
        let dd = d0_l * vscale / demod(gda);
        add_fx(c * ACC + A_FD, dd.x);
        add_fx(c * ACC + A_FD + 1u, dd.y);
        add_fx(c * ACC + A_FD + 2u, dd.z);
    }
    let b = c * ACC;
    if (rec_on) { commit_deps(c); }
    if (ru_cell != 0xffffffffu && all(ru_beta > vec3<f32>(0.0))) {
        // outgoing radiance at the 1st bounce: the rest of the path, divided by the throughput on arrival
        let u = (v_raw - ru_prefix) / ru_beta;
        let b1 = ru_cell * ACC;
        atomicAdd(&acc[b1 + A_FN], 1u);
        add_fx(b1 + A_FR, u.x);
        add_fx(b1 + A_FR + 1u, u.y);
        add_fx(b1 + A_FR + 2u, u.z);
        let l1 = lum(u);
        add_fx(b1 + A_FL, l1);
        add_f32(b1 + A_FL2, l1 * l1);
    }
    atomicAdd(&acc[b + A_FN], 1u);
    add_fx(b + A_FR, v.x);
    add_fx(b + A_FR + 1u, v.y);
    add_fx(b + A_FR + 2u, v.z);
    let l = lum(v);
    add_fx(b + A_FL, l);
    add_f32(b + A_FL2, l * l);
    if ((fr.flags & F_MULTI) != 0u) {
        // same sample for a cell of the other level of the starting pixel
        var lvl = 2u;
        for (var s = 0u; s < K; s++) {
            if (pix_cells[t.y * K + s] == c) { lvl = select(0u, 1u, s >= K / 2u); }
        }
        if (lvl < 2u) {
            let base = select(K / 2u, 0u, lvl == 1u);
            let pc = pix_cells[t.y * K + base + min(u32(rnd() * f32(K / 2u)), K / 2u - 1u)];
            if (pc != 0xffffffffu) {
                let bp = pc * ACC;
                if (rec_on) { commit_deps(pc); }
                atomicAdd(&acc[bp + A_FN], 1u);
                add_fx(bp + A_FR, v.x);
                add_fx(bp + A_FR + 1u, v.y);
                add_fx(bp + A_FR + 2u, v.z);
                add_fx(bp + A_FL, l);
                add_f32(bp + A_FL2, l * l);
            }
        }
    }
}

// merge of the frame's samples, safety net, ageing, reset
@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= atomicLoad(&ctr[C_NVIS])) { return; }
    let c = vis_list[gid.x];
    let b = c * ACC;
    var cell = cells[c];
    let fn_ = f32(atomicLoad(&acc[b + A_FN]));
    if (fn_ > 0.0) {
        let fm = vec3<f32>(ld_fx(b + A_FR), ld_fx(b + A_FR + 1u), ld_fx(b + A_FR + 2u)) / fn_;
        let fl = ld_fx(b + A_FL) / fn_;
        let fm2 = max(ld_f32(b + A_FL2) - fn_ * fl * fl, 0.0);
        if ((fr.flags & F_SHARC) != 0u) {
            let a = max(fn_ / (cell.n + fn_), 1.0 - pow(1.0 - EMA_MIN, fn_));
            cell.mean += (fm - cell.mean) * a;
            cell.lmean = lum(cell.mean);
            cell.n += fn_;
        } else {
            let n = cell.n + fn_;
            let dl = fl - cell.lmean;
            cell.mean += (fm - cell.mean) * (fn_ / n);
            if ((fr.flags & F_DSPLIT) != 0u) {
                let fdm = vec3<f32>(ld_fx(b + A_FD), ld_fx(b + A_FD + 1u), ld_fx(b + A_FD + 2u)) / fn_;
                cell.dmean += (fdm - cell.dmean) * (fn_ / n);
            }
            cell.m2 += fm2 + dl * dl * cell.n * fn_ / n;
            cell.lmean += dl * (fn_ / n);
            cell.n = n;
            // capped history: beyond it, running average (the small residual errors of the
            // corrections do not accumulate forever)
            if (cell.n > N_MAX) {
                cell.m2 *= (N_MAX - 1.0) / (cell.n - 1.0);
                cell.n = N_MAX;
            }
            // recent window of the safety net
            let rn = cell.rn + fn_;
            let rdl = fl - cell.rl;
            cell.rm2 += fm2 + rdl * rdl * cell.rn * fn_ / rn;
            cell.rl += rdl * (fn_ / rn);
            cell.rn = rn;
            if (cell.rn >= SAFETY_WINDOW) {
                if (cell.n > cell.rn) {
                    let v = max(max(cell.m2 / (cell.n - 1.0), cell.rm2 / (cell.rn - 1.0)), 1e-8);
                    let z = abs(cell.rl - cell.lmean) / sqrt(v / cell.rn + v / cell.n);
                    if (z > Z_SAFETY) {
                        // the recent window keeps only luminance: the colour is rescaled
                        cell.mean = cell.mean * (cell.rl / max(cell.lmean, 1e-12));
                        cell.n = cell.rn;
                        cell.lmean = cell.rl;
                        cell.m2 = cell.rm2;
                        atomicAdd(&ctr[C_SAFETY], 1u);
                    }
                }
                cell.rn = 0.0;
                cell.rl = 0.0;
                cell.rm2 = 0.0;
            }
        }
        cell.last = fr.frame;
    }
    let susp = atomicLoad(&acc[b + A_SUSP]);
    if (susp == 1u) { cell.last = fr.frame; }
    let foot = atomicLoad(&acc[b + A_FOOT]);
    if (foot > 0u || fn_ > 0.0) { cell.seen = fr.clock; }
    // ageing of dependencies (alternating windows of DEP_WINDOW samples)
    let ns = atomicLoad(&acc[b + A_NS]);
    if (ns >= DEP_WINDOW) {
        cell.deps_prev = vec2<u32>(atomicLoad(&acc[b + A_DEPS]), atomicLoad(&acc[b + A_DEPS + 1u]));
        cell.ns_prev = ns;
        for (var w = 0u; w < 18u; w++) {
            cell.freq_prev[w] = atomicLoad(&acc[b + A_FREQ + w]);
            atomicStore(&acc[b + A_FREQ + w], 0u);
        }
        atomicStore(&acc[b + A_DEPS], 0u);
        atomicStore(&acc[b + A_DEPS + 1u], 0u);
        atomicStore(&acc[b + A_NS], 0u);
    }
    cells[c] = cell;
    atomicMax(&ctr[C_MAXN], bitcast<u32>(cell.n));
    // reset of the frame accumulators
    atomicStore(&acc[b + A_FOOT], 0u);
    atomicStore(&acc[b + A_REF], 0u);
    atomicStore(&acc[b + A_FN], 0u);
    for (var w = A_FR; w <= A_FL2; w++) { atomicStore(&acc[b + w], 0u); }
    for (var w = A_PM; w < A_SUSP; w++) { atomicStore(&acc[b + w], 0u); }
    for (var w = A_FD; w < A_PD + 3u; w++) { atomicStore(&acc[b + w], 0u); }
    if (susp == 1u) { atomicStore(&acc[b + A_SUSP], 0u); } // 2 (en attente) persiste
}

// eviction: an eighth of the cells per frame (before the resolve, which resets the footprint to 0)
@compute @workgroup_size(64)
fn evict_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x * 8u + (fr.clock & 7u);
    if (c >= min(atomicLoad(&ctr[C_CELLS]), fr.cell_cap)) { return; }
    let b = c * ACC;
    if (cells[c].slot != 0u && atomicLoad(&acc[b + A_FOOT]) == 0u && atomicLoad(&acc[b + A_SUSP]) != 2u
        && fr.clock - cells[c].seen > STALE_FRAMES) {
        evict(c);
    }
}

@compute @workgroup_size(64)
fn display(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= fr.w * fr.h) { return; }
    let g = gbuf[k];
    var v = vec3<f32>(0.0);
    if (g.kind == 1u) {
        v = g.albedo;
    } else if (g.kind == 2u) {
        var w = 0.0;
        if ((fr.flags & F_MULTI) != 0u) {
            // fine and coarse levels: mean of each, then a gradual move to the fine one
            var vf = vec3<f32>(0.0);
            var vc = vec3<f32>(0.0);
            var wf = 0.0;
            var wc = 0.0;
            var nf = 0.0;
            for (var s = 0u; s < K; s++) {
                let c = pix_cells[k * K + s];
                if (c != 0xffffffffu && cells[c].n > 0.0) {
                    if (s < K / 2u) {
                        vf += cells[c].mean;
                        wf += 1.0;
                        nf += cells[c].n;
                    } else {
                        vc += cells[c].mean;
                        wc += 1.0;
                    }
                }
            }
            if (wf > 0.0 && wc > 0.0) {
                v = mix(vc / wc, vf / wf, clamp(nf / wf / MULTI_N, 0.0, 1.0));
                w = 1.0;
            } else if (wf > 0.0) {
                v = vf;
                w = wf;
            } else if (wc > 0.0) {
                v = vc;
                w = wc;
            }
        } else {
            for (var s = 0u; s < K; s++) {
                let c = pix_cells[k * K + s];
                if (c != 0xffffffffu && cells[c].n > 0.0) {
                    v += cells[c].mean;
                    w += 1.0;
                }
            }
        }
        if (w > 0.0) { v = v / w * demod(g.albedo); }
    }
    image[k] = vec4<f32>(v, f32(g.kind));
}

// lamp probes: rays provided by the CPU (probe[2i] = origin, probe[2i+1] = direction),
// first hit written back (position, found; albedo). fr.round: scene 0 current, 1 previous.
@group(0) @binding(21) var<storage, read_write> probe: array<vec4<f32>>;

@compute @workgroup_size(64)
fn probe_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&probe) / 2u) { return; }
    sel = fr.round;
    let h = scene_hit(probe[2u * i].xyz, probe[2u * i + 1u].xyz);
    var p = h.p;
    if (h.found) { p = probe[2u * i].xyz + probe[2u * i + 1u].xyz * h.t; }
    probe[2u * i] = vec4<f32>(p, select(0.0, 1.0, h.found));
    probe[2u * i + 1u] = vec4<f32>(h.albedo, 0.0);
}

// reference image: fr.pad0 paths per pixel added to image (seeds according to fr.frame)
@compute @workgroup_size(64)
fn reference_acc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= fr.w * fr.h) { return; }
    let g = gbuf[k];
    if (g.kind != 2u) { return; }
    reset_rec();
    rec_on = false;
    uni = false;
    split = false;
    sel = 0u;
    seed_rng(k, 0x51u, fr.frame);
    set_primary(g, false);
    var acc3 = vec3<f32>(0.0);
    for (var s = 0u; s < fr.pad0; s++) { acc3 += radiance(g.p, g.n, g.albedo).l; }
    image[k] += vec4<f32>(acc3, 0.0);
}
