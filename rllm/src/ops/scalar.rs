//! Scalar reference ops (plain loops). Row-major; weights are `[out, in]`.
//! `*_bwd` accumulate (+=) into gradient outputs, except `xent_fwd_bwd`.

/// out[n x m] = x[n x k] . w[m x k]^T (overwrites `out`).
pub fn matmul_fwd(out: &mut [f32], x: &[f32], w: &[f32], n: usize, k: usize, m: usize) {
    assert_eq!(out.len(), n * m);
    assert_eq!(x.len(), n * k);
    assert_eq!(w.len(), m * k);
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += x[i * k + p] * w[j * k + p];
            }
            out[i * m + j] = acc;
        }
    }
}

/// dx += dout . w, dw += dout^T . x (accumulates).
#[allow(clippy::too_many_arguments)]
pub fn matmul_bwd(
    dx: &mut [f32],
    dw: &mut [f32],
    dout: &[f32],
    x: &[f32],
    w: &[f32],
    n: usize,
    k: usize,
    m: usize,
) {
    assert_eq!(dx.len(), n * k);
    assert_eq!(dw.len(), m * k);
    assert_eq!(dout.len(), n * m);
    assert_eq!(x.len(), n * k);
    assert_eq!(w.len(), m * k);
    for i in 0..n {
        for j in 0..m {
            let d = dout[i * m + j];
            for p in 0..k {
                dx[i * k + p] += d * w[j * k + p];
                dw[j * k + p] += d * x[i * k + p];
            }
        }
    }
}

/// y = x * rstd * g, rstd = 1/sqrt(mean(x^2) + 1e-6) per row; stores rstd.
pub fn rmsnorm_fwd(out: &mut [f32], rstd: &mut [f32], x: &[f32], g: &[f32], n: usize, c: usize) {
    assert_eq!(out.len(), n * c);
    assert_eq!(rstd.len(), n);
    assert_eq!(x.len(), n * c);
    assert_eq!(g.len(), c);
    for i in 0..n {
        let row = &x[i * c..(i + 1) * c];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / c as f32;
        let r = 1.0 / (ms + 1e-6).sqrt();
        rstd[i] = r;
        for j in 0..c {
            out[i * c + j] = row[j] * r * g[j];
        }
    }
}

/// Accumulates dx and dg. With r = rstd:
/// dx_j = r*g_j*dy_j - r^3 * x_j / c * sum_i(dy_i * g_i * x_i); dg_j = dy_j * x_j * r.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_bwd(
    dx: &mut [f32],
    dg: &mut [f32],
    dout: &[f32],
    x: &[f32],
    g: &[f32],
    rstd: &[f32],
    n: usize,
    c: usize,
) {
    assert_eq!(dx.len(), n * c);
    assert_eq!(dg.len(), c);
    assert_eq!(dout.len(), n * c);
    assert_eq!(x.len(), n * c);
    assert_eq!(g.len(), c);
    assert_eq!(rstd.len(), n);
    for i in 0..n {
        let r = rstd[i];
        let xr = &x[i * c..(i + 1) * c];
        let dy = &dout[i * c..(i + 1) * c];
        let dot: f32 = (0..c).map(|j| dy[j] * g[j] * xr[j]).sum();
        let coef = r * r * r * dot / c as f32;
        for j in 0..c {
            dx[i * c + j] += r * g[j] * dy[j] - coef * xr[j];
            dg[j] += dy[j] * xr[j] * r;
        }
    }
}

/// out = max(x, 0)^2.
pub fn relu2_fwd(out: &mut [f32], x: &[f32]) {
    assert_eq!(out.len(), x.len());
    for (o, &v) in out.iter_mut().zip(x) {
        let r = v.max(0.0);
        *o = r * r;
    }
}

/// dx += dout * 2 * max(x, 0).
pub fn relu2_bwd(dx: &mut [f32], dout: &[f32], x: &[f32]) {
    assert_eq!(dx.len(), x.len());
    assert_eq!(dout.len(), x.len());
    for i in 0..x.len() {
        dx[i] += dout[i] * 2.0 * x[i].max(0.0);
    }
}

/// out = cap * tanh(x / cap).
pub fn softcap_fwd(out: &mut [f32], x: &[f32], cap: f32) {
    assert_eq!(out.len(), x.len());
    for (o, &v) in out.iter_mut().zip(x) {
        *o = cap * (v / cap).tanh();
    }
}

/// dx += dout * (1 - (y/cap)^2), where `out_capped` = y is the forward output.
pub fn softcap_bwd(dx: &mut [f32], dout: &[f32], out_capped: &[f32], cap: f32) {
    assert_eq!(dx.len(), out_capped.len());
    assert_eq!(dout.len(), out_capped.len());
    for i in 0..dx.len() {
        let t = out_capped[i] / cap;
        dx[i] += dout[i] * (1.0 - t * t);
    }
}

/// Masked mean cross-entropy over `n` rows of `v` logits. Returns the loss;
/// overwrites `dlogits` = mask * (softmax - onehot) / sum(mask). If sum(mask)
/// is 0, returns 0 and zeroes `dlogits`. Stable: subtracts the row max.
pub fn xent_fwd_bwd(
    dlogits: &mut [f32],
    logits: &[f32],
    targets: &[u16],
    mask: &[f32],
    n: usize,
    v: usize,
) -> f32 {
    assert_eq!(dlogits.len(), n * v);
    assert_eq!(logits.len(), n * v);
    assert_eq!(targets.len(), n);
    assert_eq!(mask.len(), n);
    let msum: f32 = mask.iter().sum();
    if msum == 0.0 {
        dlogits.fill(0.0);
        return 0.0;
    }
    let mut loss = 0.0f64;
    for i in 0..n {
        let row = &logits[i * v..(i + 1) * v];
        let drow = &mut dlogits[i * v..(i + 1) * v];
        let t = targets[i] as usize;
        assert!(t < v, "target {t} out of range for vocab {v}");
        if mask[i] == 0.0 {
            drow.fill(0.0);
            continue;
        }
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..v {
            let e = (row[j] - max).exp();
            drow[j] = e;
            sum += e;
        }
        loss += mask[i] as f64 * ((sum.ln() + max - row[t]) as f64);
        let scale = mask[i] / msum;
        for d in drow.iter_mut() {
            *d = *d / sum * scale;
        }
        drow[t] -= scale;
    }
    (loss / msum as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::check;
    use crate::rng::Rng;

    const N: usize = 3;
    const K: usize = 5;
    const M: usize = 4;
    const C: usize = 8;
    const V: usize = 11;

    fn randn(rng: &mut Rng, len: usize) -> Vec<f32> {
        (0..len).map(|_| rng.normal()).collect()
    }

    /// Random-weighted sum of outputs, accumulated in f64.
    fn wsum(r: &[f32], y: &[f32]) -> f32 {
        r.iter().zip(y).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32
    }

    #[test]
    fn matmul_known_values() {
        // x is 2x3, w is 2x3 (out=2, in=3): out = x . w^T
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = [1.0, 0.0, -1.0, 2.0, 1.0, 0.5];
        let mut out = [99.0f32; 4];
        matmul_fwd(&mut out, &x, &w, 2, 3, 2);
        assert_eq!(out, [-2.0, 5.5, -2.0, 16.0]);
    }

    #[test]
    fn matmul_gradcheck() {
        let mut rng = Rng::new(1);
        let x = randn(&mut rng, N * K);
        let w = randn(&mut rng, M * K);
        let r = randn(&mut rng, N * M);
        let mut dx = vec![0.0; N * K];
        let mut dw = vec![0.0; M * K];
        matmul_bwd(&mut dx, &mut dw, &r, &x, &w, N, K, M);
        let fx = |x: &[f32]| {
            let mut o = vec![0.0; N * M];
            matmul_fwd(&mut o, x, &w, N, K, M);
            wsum(&r, &o)
        };
        check(fx, &x, &dx);
        let fw = |w: &[f32]| {
            let mut o = vec![0.0; N * M];
            matmul_fwd(&mut o, &x, w, N, K, M);
            wsum(&r, &o)
        };
        check(fw, &w, &dw);
    }

    #[test]
    fn matmul_bwd_accumulates() {
        let mut rng = Rng::new(1);
        let x = randn(&mut rng, N * K);
        let w = randn(&mut rng, M * K);
        let dout = randn(&mut rng, N * M);
        let mut dx = vec![0.0; N * K];
        let mut dw = vec![0.0; M * K];
        matmul_bwd(&mut dx, &mut dw, &dout, &x, &w, N, K, M);
        let (dx1, dw1) = (dx.clone(), dw.clone());
        matmul_bwd(&mut dx, &mut dw, &dout, &x, &w, N, K, M);
        for i in 0..dw.len() {
            assert!((dw[i] - 2.0 * dw1[i]).abs() < 1e-5);
        }
        for i in 0..dx.len() {
            assert!((dx[i] - 2.0 * dx1[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn rmsnorm_known_values() {
        let x = [3.0f32, 4.0];
        let g = [1.0f32, 2.0];
        let mut out = [0.0; 2];
        let mut rstd = [0.0; 1];
        rmsnorm_fwd(&mut out, &mut rstd, &x, &g, 1, 2);
        let expect_rstd = 1.0 / (12.5f32 + 1e-6).sqrt();
        assert!((rstd[0] - expect_rstd).abs() < 1e-6);
        assert!((out[0] - 3.0 * expect_rstd).abs() < 1e-6);
        assert!((out[1] - 8.0 * expect_rstd).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_gradcheck() {
        let mut rng = Rng::new(1);
        let x = randn(&mut rng, N * C);
        let g = randn(&mut rng, C);
        let r = randn(&mut rng, N * C);
        let mut out = vec![0.0; N * C];
        let mut rstd = vec![0.0; N];
        rmsnorm_fwd(&mut out, &mut rstd, &x, &g, N, C);
        let mut dx = vec![0.0; N * C];
        let mut dg = vec![0.0; C];
        rmsnorm_bwd(&mut dx, &mut dg, &r, &x, &g, &rstd, N, C);
        let run = |x: &[f32], g: &[f32]| {
            let mut o = vec![0.0; N * C];
            let mut rs = vec![0.0; N];
            rmsnorm_fwd(&mut o, &mut rs, x, g, N, C);
            wsum(&r, &o)
        };
        check(|x| run(x, &g), &x, &dx);
        check(|g| run(&x, g), &g, &dg);
    }

    #[test]
    fn relu2_known_values() {
        let x = [-1.0f32, 0.0, 2.0];
        let mut out = [0.0; 3];
        relu2_fwd(&mut out, &x);
        assert_eq!(out, [0.0, 0.0, 4.0]);
    }

    #[test]
    fn relu2_gradcheck() {
        let mut rng = Rng::new(1);
        let x = randn(&mut rng, N * C);
        let r = randn(&mut rng, N * C);
        let mut dx = vec![0.0; N * C];
        relu2_bwd(&mut dx, &r, &x);
        let f = |x: &[f32]| {
            let mut o = vec![0.0; x.len()];
            relu2_fwd(&mut o, x);
            wsum(&r, &o)
        };
        check(f, &x, &dx);
    }

    #[test]
    fn softcap_gradcheck() {
        let mut rng = Rng::new(1);
        let cap = 3.0f32;
        let x: Vec<f32> = randn(&mut rng, N * C).iter().map(|v| v * 2.0).collect();
        let r = randn(&mut rng, N * C);
        let mut y = vec![0.0; N * C];
        softcap_fwd(&mut y, &x, cap);
        assert!(y.iter().all(|v| v.abs() < cap));
        let mut dx = vec![0.0; N * C];
        softcap_bwd(&mut dx, &r, &y, cap);
        let f = |x: &[f32]| {
            let mut o = vec![0.0; x.len()];
            softcap_fwd(&mut o, x, cap);
            wsum(&r, &o)
        };
        check(f, &x, &dx);
    }

    fn xent_inputs() -> (Vec<f32>, Vec<u16>) {
        let mut rng = Rng::new(1);
        let logits = randn(&mut rng, N * V);
        let targets: Vec<u16> = (0..N).map(|_| rng.below(V) as u16).collect();
        (logits, targets)
    }

    #[test]
    fn xent_gradcheck() {
        let (logits, targets) = xent_inputs();
        let mask = [1.0f32, 0.5, 1.0];
        let mut d = vec![0.0; N * V];
        xent_fwd_bwd(&mut d, &logits, &targets, &mask, N, V);
        let f = |l: &[f32]| {
            let mut scratch = vec![0.0; N * V];
            xent_fwd_bwd(&mut scratch, l, &targets, &mask, N, V)
        };
        check(f, &logits, &d);
    }

    #[test]
    fn xent_uniform_logits_loss_is_ln_v() {
        let logits = vec![0.0f32; 2 * V];
        let mut d = vec![0.0; 2 * V];
        let loss = xent_fwd_bwd(&mut d, &logits, &[3, 7], &[1.0, 1.0], 2, V);
        assert!((loss - (V as f32).ln()).abs() < 1e-5);
    }

    #[test]
    fn xent_stable_for_huge_logits() {
        let mut logits = vec![0.0f32; V];
        logits[2] = 1000.0;
        let mut d = vec![0.0; V];
        let loss = xent_fwd_bwd(&mut d, &logits, &[2], &[1.0], 1, V);
        assert!(loss.is_finite() && loss < 1e-3);
        assert!(d.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn xent_masked_rows_ignored() {
        let (logits, targets) = xent_inputs();
        let mask = [1.0f32, 0.0, 1.0];
        let mut d = vec![7.0; N * V]; // must be overwritten
        let loss = xent_fwd_bwd(&mut d, &logits, &targets, &mask, N, V);
        assert!(d[V..2 * V].iter().all(|v| *v == 0.0));
        // Equals the loss of rows 0 and 2 alone.
        let l2: Vec<f32> = [&logits[..V], &logits[2 * V..]].concat();
        let t2 = [targets[0], targets[2]];
        let mut d2 = vec![0.0; 2 * V];
        let loss2 = xent_fwd_bwd(&mut d2, &l2, &t2, &[1.0, 1.0], 2, V);
        assert!((loss - loss2).abs() < 1e-6);
        assert_eq!(&d[..V], &d2[..V]);
        assert_eq!(&d[2 * V..], &d2[V..]);
    }

    #[test]
    fn xent_all_masked_is_zero() {
        let (logits, targets) = xent_inputs();
        let mut d = vec![7.0; N * V];
        let loss = xent_fwd_bwd(&mut d, &logits, &targets, &[0.0; N], N, V);
        assert_eq!(loss, 0.0);
        assert!(d.iter().all(|v| *v == 0.0));
    }
}
