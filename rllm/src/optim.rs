//! Optimizers: AdamW (embedding + norm gains) and Muon (all weight matrices).
//!
//! All state and Newton-Schulz scratch is allocated in `new`; `step` never allocates
//! (beyond what the backend's matmul keeps in its own reusable scratch). Every NS5
//! product runs on `backend::matmul_fwd(be, ..)`.

use crate::backend::{matmul_fwd, Backend};
use crate::model::{Mat, Params};

const NS_COEF: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);
const NS_EPS: f32 = 1e-7;

/// Decoupled-weight-decay Adam with bias correction. One state slot per tensor.
pub struct AdamW {
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub wd: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: Vec<i32>,
}

impl AdamW {
    /// `sizes[i]` is the length of tensor slot `i`.
    pub fn new(sizes: &[usize]) -> AdamW {
        AdamW {
            lr: 0.008,
            b1: 0.9,
            b2: 0.95,
            eps: 1e-8,
            wd: 0.0,
            m: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            v: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            t: vec![0; sizes.len()],
        }
    }

    /// One update of tensor `slot` with effective lr `lr * lr_mult`.
    pub fn step(&mut self, slot: usize, params: &mut [f32], grads: &[f32], lr_mult: f32) {
        assert_eq!(params.len(), grads.len());
        let (m, v) = (&mut self.m[slot], &mut self.v[slot]);
        assert_eq!(m.len(), params.len());
        self.t[slot] += 1;
        let t = self.t[slot];
        let bc1 = 1.0 - self.b1.powi(t);
        let bc2 = 1.0 - self.b2.powi(t);
        let lr = self.lr * lr_mult;
        for i in 0..params.len() {
            let g = grads[i];
            m[i] = self.b1 * m[i] + (1.0 - self.b1) * g;
            v[i] = self.b2 * v[i] + (1.0 - self.b2) * g * g;
            let upd = (m[i] / bc1) / ((v[i] / bc2).sqrt() + self.eps);
            params[i] -= lr * (upd + self.wd * params[i]);
        }
    }
}

/// Preallocated buffers for one matrix shape; `r = min(rows, cols)`, `c = max`.
struct NsScratch {
    rows: usize,
    cols: usize,
    x: Vec<f32>,
    xt: Vec<f32>,
    a: Vec<f32>,
    aa: Vec<f32>,
    bx: Vec<f32>,
    out: Vec<f32>,
}

impl NsScratch {
    fn new(rows: usize, cols: usize) -> NsScratch {
        let (r, c) = (rows.min(cols), rows.max(cols));
        NsScratch {
            rows,
            cols,
            x: vec![0.0; r * c],
            xt: vec![0.0; r * c],
            a: vec![0.0; r * r],
            aa: vec![0.0; r * r],
            bx: vec![0.0; r * c],
            out: vec![0.0; rows * cols],
        }
    }

    /// Orthogonalizes `g` (rows x cols) into `self.out` (same orientation); all
    /// products on backend `be`.
    fn run(&mut self, g: &[f32], be: Backend) {
        let (rows, cols) = (self.rows, self.cols);
        assert_eq!(g.len(), rows * cols);
        let tall = rows > cols;
        let (r, c) = (rows.min(cols), rows.max(cols));
        let inv = 1.0 / (g.iter().map(|v| v * v).sum::<f32>().sqrt() + NS_EPS);
        if tall {
            for i in 0..rows {
                for j in 0..cols {
                    self.x[j * rows + i] = g[i * cols + j] * inv;
                }
            }
        } else {
            for (x, &v) in self.x.iter_mut().zip(g) {
                *x = v * inv;
            }
        }
        let (ca, cb, cc) = NS_COEF;
        for _ in 0..5 {
            // A = X Xᵀ (r x r); A² = A Aᵀ since A is symmetric.
            matmul_fwd(be, &mut self.a, &self.x, &self.x, r, c, r);
            matmul_fwd(be, &mut self.aa, &self.a, &self.a, r, r, r);
            // B = b·A + c·A², kept in `aa` (symmetric).
            for (q, &p) in self.aa.iter_mut().zip(&self.a) {
                *q = cb * p + cc * *q;
            }
            // B·X via matmul_fwd(x = B, w = Xᵀ) since it computes x·wᵀ.
            for i in 0..r {
                for j in 0..c {
                    self.xt[j * r + i] = self.x[i * c + j];
                }
            }
            matmul_fwd(be, &mut self.bx, &self.aa, &self.xt, r, r, c);
            for (x, &p) in self.x.iter_mut().zip(&self.bx) {
                *x = ca * *x + p;
            }
        }
        if tall {
            for i in 0..rows {
                for j in 0..cols {
                    self.out[i * cols + j] = self.x[j * rows + i];
                }
            }
        } else {
            self.out.copy_from_slice(&self.x);
        }
    }
}

/// Newton-Schulz quintic orthogonalization (5 iterations) with products on `be`.
/// Allocates; the optimizer uses preallocated scratch instead.
pub fn newton_schulz5(g: &Mat, be: Backend) -> Mat {
    let mut s = NsScratch::new(g.rows, g.cols);
    s.run(&g.data, be);
    Mat {
        rows: g.rows,
        cols: g.cols,
        data: s.out,
    }
}

/// Muon: Nesterov momentum, then orthogonalized update.
pub struct Muon {
    pub lr: f32,
    pub momentum: f32,
    buf: Vec<Vec<f32>>,
    u: Vec<f32>,
    scratch: Vec<NsScratch>,
}

impl Muon {
    /// `shapes[i]` is the (rows, cols) of matrix slot `i`.
    pub fn new(shapes: &[(usize, usize)]) -> Muon {
        let mut scratch: Vec<NsScratch> = Vec::new();
        for &(r, c) in shapes {
            if !scratch.iter().any(|s| s.rows == r && s.cols == c) {
                scratch.push(NsScratch::new(r, c));
            }
        }
        Muon {
            lr: 0.02,
            momentum: 0.95,
            buf: shapes.iter().map(|&(r, c)| vec![0.0; r * c]).collect(),
            u: vec![0.0; shapes.iter().map(|&(r, c)| r * c).max().unwrap_or(0)],
            scratch,
        }
    }

    /// One update of matrix `slot` with effective lr `lr * lr_mult`; NS5 on `be`.
    pub fn step(&mut self, slot: usize, w: &mut Mat, g: &Mat, lr_mult: f32, be: Backend) {
        assert_eq!((w.rows, w.cols), (g.rows, g.cols));
        let n = w.data.len();
        let (mu, buf, u) = (self.momentum, &mut self.buf[slot], &mut self.u[..n]);
        assert_eq!(buf.len(), n);
        for i in 0..n {
            buf[i] = mu * buf[i] + g.data[i];
            u[i] = g.data[i] + mu * buf[i];
        }
        let s = self
            .scratch
            .iter_mut()
            .find(|s| s.rows == w.rows && s.cols == w.cols)
            .expect("no scratch for shape");
        s.run(u, be);
        let scale = (w.rows as f32 / w.cols as f32).max(1.0).sqrt();
        let k = self.lr * lr_mult * scale;
        for (wv, &o) in w.data.iter_mut().zip(&s.out) {
            *wv -= k * o;
        }
    }
}

pub struct Optimizer {
    pub adam: AdamW,
    pub muon: Muon,
}

impl Optimizer {
    pub fn new(p: &Params) -> Optimizer {
        let mut adam_sizes = vec![p.wte.data.len()];
        let mut shapes = Vec::new();
        for l in &p.layers {
            adam_sizes.push(l.attn_norm.len());
            adam_sizes.push(l.mlp_norm.len());
            for m in [&l.wqkv, &l.wo, &l.w_up, &l.w_down] {
                shapes.push((m.rows, m.cols));
            }
        }
        adam_sizes.push(p.final_norm.len());
        shapes.extend(p.mtp.iter().map(|m| (m.rows, m.cols)));
        Optimizer {
            adam: AdamW::new(&adam_sizes),
            muon: Muon::new(&shapes),
        }
    }

    /// Muon: layer matrices + mtp mats (NS5 on `be`). AdamW: `wte` + norm gains.
    pub fn step(&mut self, p: &mut Params, g: &Params, lr_mult: f32, be: Backend) {
        let (mut a, mut m) = (0, 0);
        self.adam.step(a, &mut p.wte.data, &g.wte.data, lr_mult);
        a += 1;
        for (pl, gl) in p.layers.iter_mut().zip(&g.layers) {
            self.adam.step(a, &mut pl.attn_norm, &gl.attn_norm, lr_mult);
            self.adam
                .step(a + 1, &mut pl.mlp_norm, &gl.mlp_norm, lr_mult);
            a += 2;
            for (pw, gw) in [
                (&mut pl.wqkv, &gl.wqkv),
                (&mut pl.wo, &gl.wo),
                (&mut pl.w_up, &gl.w_up),
                (&mut pl.w_down, &gl.w_down),
            ] {
                self.muon.step(m, pw, gw, lr_mult, be);
                m += 1;
            }
        }
        self.adam.step(a, &mut p.final_norm, &g.final_norm, lr_mult);
        for (pw, gw) in p.mtp.iter_mut().zip(&g.mtp) {
            self.muon.step(m, pw, gw, lr_mult, be);
            m += 1;
        }
    }
}

pub fn zero_grads(g: &mut Params) {
    g.zero();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Batch;
    use crate::model::{backward, forward, Acts, Config};
    use crate::rng::Rng;

    fn randmat(rng: &mut Rng, rows: usize, cols: usize) -> Mat {
        Mat {
            rows,
            cols,
            data: (0..rows * cols).map(|_| rng.normal()).collect(),
        }
    }

    #[test]
    fn ns5_near_orthogonal() {
        let mut rng = Rng::new(1);
        let g = randmat(&mut rng, 32, 16);
        let x = newton_schulz5(&g, Backend::Scalar);
        assert_eq!((x.rows, x.cols), (32, 16));
        // rows > cols: columns are the short side, so XᵀX (16x16) should be ~identity.
        let c = 16;
        for i in 0..c {
            for j in 0..c {
                let dot: f32 = (0..32).map(|r| x.data[r * c + i] * x.data[r * c + j]).sum();
                if i == j {
                    assert!((dot - 1.0).abs() < 0.4, "diag {i}: {dot}");
                } else {
                    assert!(dot.abs() < 0.3, "off ({i},{j}): {dot}");
                }
            }
        }
        // wide case too: X·Xᵀ (8x8) for an 8x24 input.
        let w = newton_schulz5(&randmat(&mut rng, 8, 24), Backend::Scalar);
        for i in 0..8 {
            for j in 0..8 {
                let dot: f32 = (0..24)
                    .map(|k| w.data[i * 24 + k] * w.data[j * 24 + k])
                    .sum();
                if i == j {
                    assert!((dot - 1.0).abs() < 0.4, "wide diag {i}: {dot}");
                } else {
                    assert!(dot.abs() < 0.3, "wide off ({i},{j}): {dot}");
                }
            }
        }
    }

    /// The pre-backend NS5 (all products on `scalar::matmul_fwd`), kept as the reference.
    fn ns5_reference(g: &Mat) -> Vec<f32> {
        use crate::ops::scalar::matmul_fwd as mm;
        let (rows, cols) = (g.rows, g.cols);
        let tall = rows > cols;
        let (r, c) = (rows.min(cols), rows.max(cols));
        let inv = 1.0 / (g.data.iter().map(|v| v * v).sum::<f32>().sqrt() + NS_EPS);
        let mut x = vec![0.0f32; r * c];
        if tall {
            for i in 0..rows {
                for j in 0..cols {
                    x[j * rows + i] = g.data[i * cols + j] * inv;
                }
            }
        } else {
            for (xv, &v) in x.iter_mut().zip(&g.data) {
                *xv = v * inv;
            }
        }
        let (mut a, mut aa) = (vec![0.0f32; r * r], vec![0.0f32; r * r]);
        let (mut xt, mut bx) = (vec![0.0f32; r * c], vec![0.0f32; r * c]);
        let (ca, cb, cc) = NS_COEF;
        for _ in 0..5 {
            mm(&mut a, &x, &x, r, c, r);
            mm(&mut aa, &a, &a, r, r, r);
            for (q, &p) in aa.iter_mut().zip(&a) {
                *q = cb * p + cc * *q;
            }
            for i in 0..r {
                for j in 0..c {
                    xt[j * r + i] = x[i * c + j];
                }
            }
            mm(&mut bx, &aa, &xt, r, r, c);
            for (xv, &p) in x.iter_mut().zip(&bx) {
                *xv = ca * *xv + p;
            }
        }
        if tall {
            let mut out = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[i * cols + j] = x[j * rows + i];
                }
            }
            out
        } else {
            x
        }
    }

    const NS_SHAPES: [(usize, usize); 6] = [(1, 1), (7, 19), (19, 7), (32, 16), (128, 384), (384, 128)];

    #[test]
    fn ns5_scalar_backend_bit_exact_vs_reference() {
        let mut rng = Rng::new(3);
        for &(r, c) in &NS_SHAPES {
            let g = randmat(&mut rng, r, c);
            let want = ns5_reference(&g);
            let got = newton_schulz5(&g, Backend::Scalar);
            assert_eq!((got.rows, got.cols), (r, c));
            assert!(
                got.data.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{r}x{c}: scalar NS5 differs from reference"
            );
        }
    }

    #[test]
    fn ns5_avx512_close_to_scalar() {
        if !crate::backend::has_avx512() {
            eprintln!("note: avx512f absent, skipping ns5_avx512_close_to_scalar");
            return;
        }
        let mut rng = Rng::new(4);
        for &(r, c) in &NS_SHAPES {
            let g = randmat(&mut rng, r, c);
            let want = newton_schulz5(&g, Backend::Scalar);
            let mx = want.data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for be in [Backend::Avx512, Backend::Avx512Mt] {
                let got = newton_schulz5(&g, be);
                let err = got
                    .data
                    .iter()
                    .zip(&want.data)
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(err <= 1e-4 * mx, "{be:?} {r}x{c}: err {err} vs max {mx}");
            }
        }
    }

    #[test]
    fn adamw_minimizes_quadratic() {
        // f(w) = sum (w_i - target_i)^2
        let target = [3.0f32, -2.0, 0.5, 1.0];
        let mut w = vec![0.0f32; 4];
        let mut opt = AdamW::new(&[4]);
        opt.lr = 0.1;
        for _ in 0..500 {
            let g: Vec<f32> = w.iter().zip(&target).map(|(a, t)| 2.0 * (a - t)).collect();
            opt.step(0, &mut w, &g, 1.0);
        }
        for (a, t) in w.iter().zip(&target) {
            assert!((a - t).abs() < 0.05, "{a} vs {t}");
        }
    }

    #[test]
    fn adamw_weight_decay_shrinks() {
        let mut w = vec![1.0f32; 3];
        let mut opt = AdamW::new(&[3]);
        opt.wd = 0.1;
        opt.lr = 0.01;
        opt.step(0, &mut w, &[0.0; 3], 1.0);
        assert!((w[0] - (1.0 - 0.01 * 0.1)).abs() < 1e-7);
    }

    fn tiny_cfg() -> Config {
        Config {
            vocab: 19,
            d: 8,
            layers: 2,
            heads: 2,
            ctx: 6,
            softcap: 5.0,
            mtp: 1,
            bitnet: false,
        }
    }

    #[test]
    fn muon_reduces_loss_tiny_model() {
        let cfg = tiny_cfg();
        let mut rng = Rng::new(7);
        let mut p = Params::init(&cfg, &mut rng);
        let (b, t, extra) = (2, 5, 1);
        let (mut inputs, mut targets) = (Vec::new(), Vec::new());
        for _ in 0..b {
            let w: Vec<u16> = (0..t + 1 + extra)
                .map(|_| rng.below(cfg.vocab) as u16)
                .collect();
            inputs.extend_from_slice(&w[..t]);
            targets.extend_from_slice(&w[1..]);
        }
        let batch = Batch {
            b,
            t,
            extra,
            inputs,
            targets,
            mask: vec![1.0; b * t],
        };
        let mut a = Acts::new(&cfg, b, t);
        let mut g = p.zeros_like();
        let mut opt = Optimizer::new(&p);
        opt.muon.lr = 0.02;
        opt.adam.lr = 0.008;
        let first = forward(&p, &cfg, Backend::Scalar, &mut a, &batch);
        let mut last = first;
        for _ in 0..50 {
            zero_grads(&mut g);
            last = forward(&p, &cfg, Backend::Scalar, &mut a, &batch);
            backward(&p, &cfg, Backend::Scalar, &mut a, &batch, &mut g);
            opt.step(&mut p, &g, 1.0, Backend::Scalar);
        }
        let end = forward(&p, &cfg, Backend::Scalar, &mut a, &batch);
        assert!(
            end.is_finite() && end < first - 0.1,
            "loss {first} -> {end} (last train {last})"
        );
    }

    #[test]
    fn muon_update_has_scale_and_momentum() {
        // 1x1 matrix: NS(u) = sign-ish ~ +-1 magnitude; update direction opposes gradient.
        let mut w = Mat {
            rows: 1,
            cols: 1,
            data: vec![0.0],
        };
        let g = Mat {
            rows: 1,
            cols: 1,
            data: vec![2.0],
        };
        let mut m = Muon::new(&[(1, 1)]);
        m.lr = 0.1;
        m.step(0, &mut w, &g, 1.0, Backend::Scalar);
        assert!(w.data[0] < -0.05 && w.data[0] > -0.15, "{}", w.data[0]);
    }
}
