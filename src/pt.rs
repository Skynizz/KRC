//! Diffuse path tracer with NEE, instrumented to record a path's dependencies.
use crate::deps::{mark_segment, Bits};
use crate::math::{Rng, V3};
use crate::scene::{cosine_dir, Scene, LIGHT_ID_BASE};

const MAX_DEPTH: u32 = 8;
const MISS_LEN: f64 = 30.0;

pub trait Rec {
    fn segment(&mut self, a: V3, b: V3);
    fn object(&mut self, id: usize);
    fn light_selected(&mut self, _j: usize) {}
    /// path vertex (point where it bounces): that is where a light change is picked up
    fn vertex(&mut self, _p: V3) {}
}

pub struct NoRec;
impl Rec for NoRec {
    fn segment(&mut self, _: V3, _: V3) {}
    fn object(&mut self, _: usize) {}
}

/// Dependencies of a path: regions crossed (H*), objects touched (H0), lamps chosen by NEE.
#[derive(Default)]
pub struct DepRec {
    pub regions: Bits,
    pub objs: u128,
    pub lights: u64,
    /// super-regions holding the path's vertices (bit s = super-region s)
    pub vsup: u64,
}
impl Rec for DepRec {
    fn segment(&mut self, a: V3, b: V3) {
        mark_segment(&mut self.regions, a, b);
    }
    fn object(&mut self, id: usize) {
        self.objs |= 1u128 << id;
    }
    fn light_selected(&mut self, j: usize) {
        self.lights |= 1 << j;
    }
    fn vertex(&mut self, p: V3) {
        self.vsup |= 1 << crate::deps::super_of(crate::deps::region_of(p));
    }
}

/// Lamp choice ∝ power. In `union` mode, the pmf ignores the on/off state: two scenes
/// that differ only by a lamp then consume the same random numbers (correlated paths).
fn sample_light(scene: &Scene, rng: &mut Rng, union: bool) -> Option<(usize, f64)> {
    let power = |j: usize| if union { scene.light_power_if_on(j) } else { scene.light_power(j) };
    let total: f64 = (0..scene.lights.len()).map(power).sum();
    if total <= 0.0 {
        return None;
    }
    let mut u = rng.f64() * total;
    for j in 0..scene.lights.len() {
        let p = power(j);
        if p > 0.0 && (u < p || j == scene.lights.len() - 1) {
            return Some((j, p / total));
        }
        u -= p;
    }
    None
}

/// One-sample estimator of the outgoing (diffuse) radiance at point p with normal n.
/// The consumption of random numbers per vertex is fixed, for correlated sampling.
/// Also returns Σ_j w_j · C_j, where C_j is the NEE contribution lamp j would
/// have if it were on (identical geometry). For a change touching only
/// lamps, with w_j = on_before − on_after, the old radiance is L + Σ w_j C_j:
/// the path of the old scene does not need to be retraced.
#[allow(clippy::too_many_arguments)]
pub fn radiance_split<R: Rec>(
    scene: &Scene,
    p0: V3,
    n0: V3,
    albedo0: V3,
    rng: &mut Rng,
    rec: &mut R,
    union: bool,
    toggle_w: &[f64],
) -> (V3, V3) {
    let (mut p, mut n, mut albedo) = (p0, n0, albedo0);
    let mut beta = V3::splat(1.0);
    let mut l = V3::default();
    let mut dl = V3::default();
    rec.vertex(p0);
    for depth in 0..MAX_DEPTH {
        let origin = p + n * 1e-6;
        // NEE
        if let Some((j, pmf)) = sample_light(scene, rng, union) {
            rec.light_selected(j);
            let li = &scene.lights[j];
            let q = li.center + crate::math::v3((2.0 * rng.f64() - 1.0) * li.hx, 0.0, (2.0 * rng.f64() - 1.0) * li.hz);
            let w = q - origin;
            let dist2 = w.dot(w);
            let dist = dist2.sqrt();
            let wd = w / dist;
            let (cos_x, cos_l) = (n.dot(wd), wd.y);
            let wj = toggle_w.get(j).copied().unwrap_or(0.0);
            if cos_x > 0.0 && cos_l > 0.0 && !li.on && wj == 0.0 {
                rec.segment(origin, q); // lamp off (union mode): conservative dependency
            } else if cos_x > 0.0 && cos_l > 0.0 {
                // a blocked shadow ray depends only on the space up to the occluder
                match scene.occluder(origin, wd, dist - 1e-6) {
                    Some((o, t)) => {
                        rec.segment(origin, origin + wd * t);
                        rec.object(o);
                    }
                    None => {
                        rec.segment(origin, q);
                        rec.object(LIGHT_ID_BASE + j);
                        let g = cos_x * cos_l / dist2 * li.area() / pmf / std::f64::consts::PI;
                        let c = beta.mul(albedo).mul(li.emission) * g;
                        if li.on {
                            l += c;
                        }
                        dl += c * wj;
                    }
                }
            }
        }
        // diffuse bounce
        let d = cosine_dir(n, rng);
        beta = beta.mul(albedo);
        let hit = scene.intersect(origin, d, f64::INFINITY);
        match hit {
            None => {
                rec.segment(origin, origin + d * MISS_LEN);
                break;
            }
            Some(h) => {
                rec.segment(origin, h.p);
                rec.object(h.id);
                if h.light.is_some() {
                    break; // emission already counted by NEE
                }
                rec.vertex(h.p);
                p = h.p;
                n = h.n;
                albedo = h.albedo;
            }
        }
        if depth >= 2 {
            let q = beta.max_c().clamp(0.05, 0.95);
            if rng.f64() > q {
                break;
            }
            beta = beta / q;
        }
    }
    (l, dl)
}
