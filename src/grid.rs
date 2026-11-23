//! Region grid specific to a scene (dependencies), and its WGSL block.
//! The grid of rooms in a row (deps.rs) is a special case of it: Grid::boxes().
use crate::math::{v3, V3};
use crate::scene::Aabb;

/// Fine region bits on the GPU side (108 u32 words) and super-region counters (36).
pub const MAX_REGIONS: i64 = 3456;
pub const MAX_SUPERS: i64 = 36;

#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub origin: V3,
    pub region: f64,
    pub dims: [i64; 3],
    /// a super-region = sdiv regions per axis
    pub sdiv: [i64; 3],
}

impl Grid {
    pub fn boxes() -> Grid {
        Grid { origin: v3(-0.2, -0.2, -0.2), region: 0.6, dims: [72, 6, 8], sdiv: [4, 6, 4] }
    }

    /// Smallest regions (≥ 0.3 m) covering the box in ≤ MAX_REGIONS, then the smallest cubic
    /// super-regions that fit in ≤ MAX_SUPERS.
    pub fn fit(b: &Aabb) -> Grid {
        let size = b.max - b.min;
        let mut region = 0.3;
        let dims = loop {
            let d = [0, 1, 2].map(|a| (size.get(a) / region).ceil().max(1.0) as i64);
            if d[0] * d[1] * d[2] <= MAX_REGIONS {
                break d;
            }
            region *= 1.1;
        };
        let mut s = 1;
        let sdiv = loop {
            let sd = [s, s, s];
            let n: i64 = (0..3).map(|a| (dims[a] + sd[a] - 1) / sd[a]).product();
            if n <= MAX_SUPERS {
                break sd;
            }
            s += 1;
        };
        Grid { origin: b.min, region, dims, sdiv }
    }

    pub fn sdims(&self) -> [i64; 3] {
        [0, 1, 2].map(|a| (self.dims[a] + self.sdiv[a] - 1) / self.sdiv[a])
    }

    pub fn cell_of(&self, p: V3) -> [i64; 3] {
        [0, 1, 2].map(|a| (((p.get(a) - self.origin.get(a)) / self.region).floor() as i64).clamp(0, self.dims[a] - 1))
    }

    pub fn index(&self, c: [i64; 3]) -> usize {
        ((c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]) as usize
    }

    pub fn super_of(&self, c: [i64; 3]) -> usize {
        let sd = self.sdims();
        (((c[2] / self.sdiv[2]) * sd[1] + c[1] / self.sdiv[1]) * sd[0] + c[0] / self.sdiv[0]) as usize
    }

    /// Bits of the regions overlapping the box (GPU format: 108 u32 words).
    pub fn mark_aabb(&self, bits: &mut [u32], b: &Aabb) {
        let (lo, hi) = (self.cell_of(b.min), self.cell_of(b.max));
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let i = self.index([x, y, z]);
                    bits[i >> 5] |= 1 << (i & 31);
                }
            }
        }
    }

    /// Mask of the super-regions touched by a set of fine regions.
    pub fn supers_of_bits(&self, bits: &[u32]) -> u64 {
        let mut m = 0u64;
        for (w, &word) in bits.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let i = (w * 32 + word.trailing_zeros() as usize) as i64;
                word &= word - 1;
                let c = [i % self.dims[0], i / self.dims[0] % self.dims[1], i / (self.dims[0] * self.dims[1])];
                m |= 1 << self.super_of(c);
            }
        }
        m
    }

    /// Replaces the `// <grid>` … `// </grid>` block of the shader with this grid's constants.
    pub fn template(&self, src: &str) -> String {
        let (a, b) = (src.find("// <grid>").unwrap(), src.find("// </grid>").unwrap());
        let block = format!(
            "// <grid>\nconst REGION: f32 = {:.6};\nconst ORIGIN: vec3<f32> = vec3<f32>({:.6}, {:.6}, {:.6});\n\
const DIMS: vec3<i32> = vec3<i32>({}, {}, {});\nconst SDIV: vec3<i32> = vec3<i32>({}, {}, {});\n\
const SDIMS: vec3<i32> = vec3<i32>({}, {}, {});\n",
            self.region,
            self.origin.x,
            self.origin.y,
            self.origin.z,
            self.dims[0],
            self.dims[1],
            self.dims[2],
            self.sdiv[0],
            self.sdiv[1],
            self.sdiv[2],
            self.sdims()[0],
            self.sdims()[1],
            self.sdims()[2]
        );
        format!("{}{}{}", &src[..a], block, &src[b..])
    }
}
