//! Single path-tracing interface, on the CPU (pt.rs) or the GPU (trace.wgsl).
use crate::gpu::Gpu;
use crate::math::{Rng, V3};
use crate::par::par_map;
use crate::pt::{radiance_split, DepRec, NoRec};
use crate::scene::Scene;

/// A surface point to estimate, with its seed.
#[derive(Clone, Copy)]
pub struct Req {
    pub p: V3,
    pub n: V3,
    pub albedo: V3,
    pub seed: u64,
}

pub struct Opts<'a> {
    /// lamp pmf independent of their state (correlated paths)
    pub union: bool,
    /// record the path's dependencies
    pub record: bool,
    /// weight on_before − on_after per lamp (empty: none)
    pub toggle_w: &'a [f64],
    /// paths averaged per request
    pub spp: u32,
}

impl Opts<'_> {
    pub fn plain() -> Opts<'static> {
        Opts { union: false, record: false, toggle_w: &[], spp: 1 }
    }
}

/// Results of a batch: radiance, ΔL (changed lamps), and dependencies if `record`.
/// Dependencies are separate: a DepRec weighs ~450 bytes, and building them for nothing cost
/// more than the GPU trace itself.
#[derive(Default)]
pub struct Traced {
    pub l: Vec<V3>,
    pub dl: Vec<V3>,
    pub recs: Vec<DepRec>,
}

pub enum Tracer {
    Cpu,
    Gpu(Gpu),
}

impl Tracer {
    /// GPU if KRC_GPU is set (and an adapter is available), otherwise CPU.
    pub fn from_env() -> Tracer {
        if std::env::var("KRC_GPU").is_ok() {
            match Gpu::new() {
                Some(g) => {
                    eprintln!("traceur GPU : {}", g.name);
                    return Tracer::Gpu(g);
                }
                None => eprintln!("no usable GPU adapter, falling back to the CPU"),
            }
        }
        Tracer::Cpu
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Tracer::Gpu(_))
    }

    pub fn trace(&self, scene: &Scene, reqs: &[Req], o: &Opts) -> Traced {
        match self {
            Tracer::Gpu(g) => g.trace(scene, reqs, o),
            Tracer::Cpu => {
                let res = par_map(reqs.len(), |i| {
                    let r = &reqs[i];
                    let mut rng = Rng::new(r.seed);
                    let (mut l, mut dl) = (V3::default(), V3::default());
                    let mut rec = DepRec::default();
                    for _ in 0..o.spp {
                        let (a, b) = if o.record {
                            radiance_split(scene, r.p, r.n, r.albedo, &mut rng, &mut rec, o.union, o.toggle_w)
                        } else {
                            radiance_split(scene, r.p, r.n, r.albedo, &mut rng, &mut NoRec, o.union, o.toggle_w)
                        };
                        l += a;
                        dl += b;
                    }
                    let k = 1.0 / o.spp as f64;
                    (l * k, dl * k, if o.record { Some(rec) } else { None })
                });
                let mut t = Traced::default();
                for (l, dl, rec) in res {
                    t.l.push(l);
                    t.dl.push(dl);
                    t.recs.extend(rec);
                }
                t
            }
        }
    }
}
