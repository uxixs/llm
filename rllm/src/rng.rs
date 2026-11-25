//! SplitMix64 PRNG: tiny, seedable, deterministic.

#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (multiply-high; bias is negligible). `n` must be > 0.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0)");
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Uniform f32 in `[0, 1)` (24 random bits).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Standard normal sample (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        // u1 in (0, 1] so ln() is finite; u2 in [0, 1).
        let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = Rng::new(43);
        assert_ne!(Rng::new(42).next_u64(), c.next_u64());
        // SplitMix64 reference value for seed 0.
        assert_eq!(Rng::new(0).next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn below_in_range_and_covers() {
        let mut r = Rng::new(1);
        let mut seen = [false; 7];
        for _ in 0..1000 {
            let v = r.below(7);
            assert!(v < 7);
            seen[v] = true;
        }
        assert!(seen.iter().all(|&s| s));
        assert_eq!(r.below(1), 0);
    }

    #[test]
    fn normal_moments() {
        let mut r = Rng::new(7);
        let n = 20000;
        let xs: Vec<f32> = (0..n).map(|_| r.normal()).collect();
        let mean = xs.iter().sum::<f32>() / n as f32;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.08, "var {var}");
    }

    #[test]
    fn uniform_in_unit_interval() {
        let mut r = Rng::new(3);
        let xs: Vec<f32> = (0..10000).map(|_| r.uniform()).collect();
        assert!(xs.iter().all(|&x| (0.0..1.0).contains(&x)));
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
    }
}
