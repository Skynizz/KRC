//! Cells of the spatial cache: statistics, verification state, dependencies.
use crate::deps::{Bits, Coarse, Freq};
use crate::math::{mix64, V3};
use std::collections::HashMap;

/// Welford accumulator (RGB mean, variance on luminance).
#[derive(Clone, Copy, Default)]
pub struct Stat {
    pub n: f64,
    pub mean: V3,
    pub lmean: f64,
    pub m2: f64,
}

impl Stat {
    pub fn add(&mut self, v: V3) {
        self.n += 1.0;
        self.mean += (v - self.mean) / self.n;
        let l = v.lum();
        let dl = l - self.lmean;
        self.lmean += dl / self.n;
        self.m2 += dl * (l - self.lmean);
    }
    /// Exponential average with a floor weight `amin` (SHaRC/DDGI-style hysteresis).
    pub fn add_ema(&mut self, v: V3, amin: f64) {
        self.n += 1.0;
        let a = (1.0 / self.n).max(amin);
        self.mean += (v - self.mean) * a;
        self.lmean = self.mean.lum();
    }
    pub fn var(&self) -> f64 {
        if self.n > 1.0 { self.m2 / (self.n - 1.0) } else { 0.0 }
    }
    pub fn merge(&mut self, o: &Stat) {
        if o.n == 0.0 {
            return;
        }
        let n = self.n + o.n;
        let dl = o.lmean - self.lmean;
        self.mean += (o.mean - self.mean) * (o.n / n);
        self.lmean += dl * (o.n / n);
        self.m2 += o.m2 + dl * dl * self.n * o.n / n;
        self.n = n;
    }
    /// Caps the history's weight at `nmax` samples.
    pub fn cap(&mut self, nmax: f64) {
        if self.n > nmax {
            self.m2 *= (nmax - 1.0) / (self.n - 1.0);
            self.n = nmax;
        }
    }
}

/// z-score of the difference of luminance means between two accumulators.
pub fn zscore(a: &Stat, b: &Stat) -> f64 {
    let v = a.var().max(b.var()).max(1e-8);
    (a.lmean - b.lmean).abs() / (v / a.n + v / b.n).sqrt()
}

#[derive(Clone, Default)]
pub struct Cell {
    pub main: Stat,
    /// Fresh samples of a suspect cell, waiting for verification.
    pub fresh: Option<Stat>,
    /// Recent window of the safety net.
    pub recent: Stat,
    /// Regions crossed by recent samples. With ageing, two alternating sets
    /// cover the last K to 2K samples: a region crossed with a
    /// frequency ≪ 1/K is forgotten (its influence is bounded by that frequency).
    pub deps: Bits,
    pub deps_prev: Bits,
    /// same dependencies at coarse resolution (levels 1 and 2), for the memory experiment
    pub cdeps: Coarse,
    pub cdeps_prev: Coarse,
    pub dep_count: u32,
    /// Frequencies of path vertices per super-region, same alternating windows as `deps`.
    pub freq: Freq,
    pub freq_prev: Freq,
    pub objs: u128,
    pub last: i64,
    pub cx: f64,
}

impl Cell {
    pub fn clear(&mut self) {
        self.main = Stat::default();
        self.fresh = None;
        self.recent = Stat::default();
        self.deps = Bits::default();
        self.deps_prev = Bits::default();
        self.cdeps = Coarse::default();
        self.cdeps_prev = Coarse::default();
        self.dep_count = 0;
        self.freq = Freq::default();
        self.freq_prev = Freq::default();
        self.objs = 0;
    }
    pub fn add_deps(&mut self, rec: &crate::pt::DepRec, window: Option<u32>) {
        let b = &rec.regions;
        self.deps.or(b);
        self.cdeps.or(&Coarse::from(b));
        self.freq.add(rec.vsup);
        if let Some(k) = window {
            self.dep_count += 1;
            if self.dep_count >= k {
                self.deps_prev = self.deps;
                self.deps = Bits::default();
                self.cdeps_prev = self.cdeps;
                self.cdeps = Coarse::default();
                self.freq_prev = self.freq;
                self.freq = Freq::default();
                self.dep_count = 0;
            }
        }
    }
    /// Fraction of recent samples with a vertex in the super-region.
    pub fn freq_of(&self, s: usize) -> f64 {
        let n = self.freq.n as f64 + self.freq_prev.n as f64;
        if n == 0.0 {
            return 0.0;
        }
        ((self.freq.cnt[s] as f64 + self.freq_prev.cnt[s] as f64) / n).min(1.0)
    }
    /// Did the cell cross these regions recently? `level` 0: fine bitset (0.6 m),
    /// 1: 1.2 m, 2: super-regions (2.4 m). Coarser = more suspects, never fewer.
    pub fn depends_on(&self, regions: &Bits, coarse: &Coarse, level: u8) -> bool {
        if level == 0 {
            self.deps.intersects(regions) || self.deps_prev.intersects(regions)
        } else {
            self.cdeps.intersects(coarse, level) || self.cdeps_prev.intersects(coarse, level)
        }
    }
    pub fn display(&self) -> Option<V3> {
        if self.main.n > 0.0 {
            Some(self.main.mean)
        } else {
            self.fresh.filter(|f| f.n > 0.0).map(|f| f.mean)
        }
    }
}

#[derive(Default)]
pub struct Cache {
    pub map: HashMap<u64, u32>,
    pub cells: Vec<Cell>,
}

impl Cache {
    pub fn get_or_create(&mut self, key: u64, cx: f64) -> u32 {
        let cells = &mut self.cells;
        *self.map.entry(key).or_insert_with(|| {
            cells.push(Cell { cx, last: -1_000, ..Default::default() });
            (cells.len() - 1) as u32
        })
    }
}

/// Target cell size in pixel footprints; quantised in powers of 2 (≈ 2–4 pixels).
pub const CELL_PX: f64 = 4.0;

/// Jittered spatial hash key (Binder et al.): the jitter is tangent to the surface.
/// `footprint`: size of a pixel projected at the point's distance.
pub fn cell_key(p: V3, n: V3, footprint: f64, px: usize, slot: usize) -> u64 {
    let level = (CELL_PX * footprint).log2().floor();
    let s = level.exp2();
    let h = mix64(((px as u64) << 3 | slot as u64) ^ 0x5bd1e995);
    let j = [(h & 0xffff) as f64 / 65536.0 - 0.5, ((h >> 16) & 0xffff) as f64 / 65536.0 - 0.5];
    let axis = if n.x.abs() > 0.5 { 0 } else if n.y.abs() > 0.5 { 1 } else { 2 };
    let mut q = [0i64; 3];
    let mut ji = 0;
    for a in 0..3 {
        let c = p.get(a) / s;
        q[a] = if a == axis {
            c.round() as i64
        } else {
            ji += 1;
            (c + j[ji - 1]).floor() as i64
        };
    }
    let naxis = (axis as u64) * 2 + (n.get(axis) > 0.0) as u64;
    let mut k = mix64(level as i64 as u64 ^ (naxis << 56));
    for c in q {
        k = mix64(k ^ c as u64);
    }
    k
}
