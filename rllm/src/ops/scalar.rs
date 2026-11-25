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

/// (cos, sin) of `pos * 10000^(-2i/d)`.
fn rope_angle(pos: usize, i: usize, d: usize) -> (f32, f32) {
    let theta = 10000f32.powf(-2.0 * i as f32 / d as f32);
    let a = pos as f32 * theta;
    (a.cos(), a.sin())
}

/// In-place rotary embedding on `[b,t,h,d]`; pair (2i, 2i+1) rotates by `t * 10000^(-2i/d)`.
pub fn rope_fwd(x: &mut [f32], b: usize, t: usize, h: usize, d: usize) {
    rope_rotate(x, b, t, h, d, 1.0);
}

/// Inverse rotation in place (the transpose of `rope_fwd`'s rotation), for gradients.
pub fn rope_bwd(dx: &mut [f32], b: usize, t: usize, h: usize, d: usize) {
    rope_rotate(dx, b, t, h, d, -1.0);
}

fn rope_rotate(x: &mut [f32], b: usize, t: usize, h: usize, d: usize, sign: f32) {
    assert_eq!(x.len(), b * t * h * d);
    for bi in 0..b {
        for ti in 0..t {
            for hi in 0..h {
                let base = ((bi * t + ti) * h + hi) * d;
                for i in 0..d / 2 {
                    let (cos, sin) = rope_angle(ti, i, d);
                    let sin = sign * sin;
                    let (x0, x1) = (x[base + 2 * i], x[base + 2 * i + 1]);
                    x[base + 2 * i] = x0 * cos - x1 * sin;
                    x[base + 2 * i + 1] = x0 * sin + x1 * cos;
                }
            }
        }
    }
}

/// Causal multi-head attention. `qkv` `[b,t,3c]` (q|k|v, head h at cols h*d..), `out` `[b,t,c]`,
/// `att` `[b,h,t,t]` probabilities (future entries are 0). Scale 1/sqrt(c/h).
pub fn attn_fwd(
    out: &mut [f32],
    att: &mut [f32],
    qkv: &[f32],
    b: usize,
    t: usize,
    c: usize,
    h: usize,
) {
    let d = c / h;
    let scale = 1.0 / (d as f32).sqrt();
    let c3 = 3 * c;
    att.fill(0.0);
    out.fill(0.0);
    for bi in 0..b {
        for hi in 0..h {
            for tq in 0..t {
                let q = &qkv[(bi * t + tq) * c3 + hi * d..][..d];
                let row = &mut att[((bi * h + hi) * t + tq) * t..][..t];
                let mut max = f32::NEG_INFINITY;
                for s in 0..=tq {
                    let k = &qkv[(bi * t + s) * c3 + c + hi * d..][..d];
                    let dot: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
                    row[s] = dot * scale;
                    max = max.max(row[s]);
                }
                let mut sum = 0.0;
                for p in &mut row[..=tq] {
                    *p = (*p - max).exp();
                    sum += *p;
                }
                let o = &mut out[(bi * t + tq) * c + hi * d..][..d];
                for s in 0..=tq {
                    row[s] /= sum;
                    let v = &qkv[(bi * t + s) * c3 + 2 * c + hi * d..][..d];
                    for (oj, vj) in o.iter_mut().zip(v) {
                        *oj += row[s] * vj;
                    }
                }
            }
        }
    }
}

/// Backward of `attn_fwd`; ACCUMULATES into `dqkv` (`[b,t,3c]`). Uses the stored `att`.
pub fn attn_bwd(
    dqkv: &mut [f32],
    dout: &[f32],
    qkv: &[f32],
    att: &[f32],
    b: usize,
    t: usize,
    c: usize,
    h: usize,
) {
    let d = c / h;
    let scale = 1.0 / (d as f32).sqrt();
    let c3 = 3 * c;
    let mut dp = vec![0.0f32; t];
    for bi in 0..b {
        for hi in 0..h {
            for tq in 0..t {
                let q_off = (bi * t + tq) * c3 + hi * d;
                let do_ = &dout[(bi * t + tq) * c + hi * d..][..d];
                let p = &att[((bi * h + hi) * t + tq) * t..][..t];
                // dp[s] = dout . v_s ; dv_s += p[s] * dout
                let mut pdp = 0.0;
                for s in 0..=tq {
                    let v_off = (bi * t + s) * c3 + 2 * c + hi * d;
                    let mut acc = 0.0;
                    for j in 0..d {
                        acc += do_[j] * qkv[v_off + j];
                        dqkv[v_off + j] += p[s] * do_[j];
                    }
                    dp[s] = acc;
                    pdp += p[s] * acc;
                }
                // softmax Jacobian: ds = p * (dp - sum(p*dp)); scores = scale * q.k
                for s in 0..=tq {
                    let ds = p[s] * (dp[s] - pdp) * scale;
                    let k_off = (bi * t + s) * c3 + c + hi * d;
                    for j in 0..d {
                        dqkv[q_off + j] += ds * qkv[k_off + j];
                        dqkv[k_off + j] += ds * qkv[q_off + j];
                    }
                }
            }
        }
    }
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

    const AB: usize = 2;
    const AT: usize = 5;
    const AC: usize = 8;
    const AH: usize = 2;

    #[test]
    fn rope_known_values() {
        // b=1, t=2, h=1, d=4: position 0 is the identity; at position 1 pair 0
        // rotates by 1 rad and pair 1 by 10000^(-1/2) = 0.01 rad.
        let mut x = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        rope_fwd(&mut x, 1, 2, 1, 4);
        assert_eq!(&x[..4], &[1.0, 0.0, 1.0, 0.0]);
        assert!((x[4] - 1f32.cos()).abs() < 1e-6);
        assert!((x[5] - 1f32.sin()).abs() < 1e-6);
        assert!((x[6] - 0.01f32.cos()).abs() < 1e-6);
        assert!((x[7] - 0.01f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn rope_preserves_norm() {
        let (b, t, h, d) = (2, 5, 2, 8);
        let mut rng = Rng::new(10);
        let x = randn(&mut rng, b * t * h * d);
        let mut y = x.clone();
        rope_fwd(&mut y, b, t, h, d);
        for (xp, yp) in x.chunks(2).zip(y.chunks(2)) {
            let n0 = xp[0] * xp[0] + xp[1] * xp[1];
            let n1 = yp[0] * yp[0] + yp[1] * yp[1];
            assert!((n0 - n1).abs() < 1e-4);
        }
        assert_ne!(x, y);
    }

    #[test]
    fn rope_bwd_inverts() {
        let (b, t, h, d) = (2, 5, 2, 8);
        let mut rng = Rng::new(11);
        let x = randn(&mut rng, b * t * h * d);
        let mut y = x.clone();
        rope_fwd(&mut y, b, t, h, d);
        rope_bwd(&mut y, b, t, h, d);
        for (a, c) in x.iter().zip(&y) {
            assert!((a - c).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_gradcheck() {
        let (b, t, h, d) = (2, 5, 2, 8);
        let mut rng = Rng::new(12);
        let x = randn(&mut rng, b * t * h * d);
        let r = randn(&mut rng, b * t * h * d);
        let mut dx = r.clone();
        rope_bwd(&mut dx, b, t, h, d);
        let f = |x: &[f32]| {
            let mut y = x.to_vec();
            rope_fwd(&mut y, b, t, h, d);
            wsum(&r, &y)
        };
        check(f, &x, &dx);
    }

    #[test]
    fn attn_known_values_single_head() {
        // b=1, t=2, c=2, h=1 (d=2): q=k=0 gives uniform causal attention.
        // qkv rows: [q0 q1 | k0 k1 | v0 v1]
        let qkv = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0];
        let mut out = [9.0f32; 4];
        let mut att = [9.0f32; 4];
        attn_fwd(&mut out, &mut att, &qkv, 1, 2, 2, 1);
        assert_eq!(att, [1.0, 0.0, 0.5, 0.5]);
        assert_eq!(out, [1.0, 2.0, 2.0, 4.0]);
    }

    #[test]
    fn attn_probs_causal_and_normalized() {
        let mut rng = Rng::new(20);
        let qkv = randn(&mut rng, AB * AT * 3 * AC);
        let mut out = vec![0.0; AB * AT * AC];
        let mut att = vec![7.0; AB * AH * AT * AT];
        attn_fwd(&mut out, &mut att, &qkv, AB, AT, AC, AH);
        for row in att.chunks(AT).enumerate() {
            let (i, r) = row;
            let tq = i % AT;
            assert!((r.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(r[tq + 1..].iter().all(|p| *p == 0.0));
            assert!(r[..=tq].iter().all(|p| *p > 0.0));
        }
    }

    #[test]
    fn attn_causal() {
        let mut rng = Rng::new(21);
        let qkv = randn(&mut rng, AB * AT * 3 * AC);
        let mut out = vec![0.0; AB * AT * AC];
        let mut att = vec![0.0; AB * AH * AT * AT];
        attn_fwd(&mut out, &mut att, &qkv, AB, AT, AC, AH);
        // Perturb every q/k/v value of token 3 in batch 0.
        let mut qkv2 = qkv.clone();
        for v in &mut qkv2[3 * 3 * AC..4 * 3 * AC] {
            *v += 1.0;
        }
        let mut out2 = vec![0.0; AB * AT * AC];
        let mut att2 = vec![0.0; AB * AH * AT * AT];
        attn_fwd(&mut out2, &mut att2, &qkv2, AB, AT, AC, AH);
        assert_eq!(&out[..3 * AC], &out2[..3 * AC]);
        assert_ne!(&out[3 * AC..4 * AC], &out2[3 * AC..4 * AC]);
        // Batch 1 is untouched.
        assert_eq!(&out[AT * AC..], &out2[AT * AC..]);
    }

    #[test]
    fn attn_gradcheck() {
        let mut rng = Rng::new(22);
        let qkv = randn(&mut rng, AB * AT * 3 * AC);
        let r = randn(&mut rng, AB * AT * AC);
        let mut out = vec![0.0; AB * AT * AC];
        let mut att = vec![0.0; AB * AH * AT * AT];
        attn_fwd(&mut out, &mut att, &qkv, AB, AT, AC, AH);
        let mut dqkv = vec![0.0; qkv.len()];
        attn_bwd(&mut dqkv, &r, &qkv, &att, AB, AT, AC, AH);
        let f = |x: &[f32]| {
            let mut o = vec![0.0; AB * AT * AC];
            let mut a = vec![0.0; AB * AH * AT * AT];
            attn_fwd(&mut o, &mut a, x, AB, AT, AC, AH);
            wsum(&r, &o)
        };
        check(f, &qkv, &dqkv);
    }

    #[test]
    fn attn_bwd_accumulates() {
        let mut rng = Rng::new(23);
        let qkv = randn(&mut rng, AB * AT * 3 * AC);
        let r = randn(&mut rng, AB * AT * AC);
        let mut out = vec![0.0; AB * AT * AC];
        let mut att = vec![0.0; AB * AH * AT * AT];
        attn_fwd(&mut out, &mut att, &qkv, AB, AT, AC, AH);
        let mut d1 = vec![0.0; qkv.len()];
        attn_bwd(&mut d1, &r, &qkv, &att, AB, AT, AC, AH);
        let mut d2 = vec![1.0; qkv.len()];
        attn_bwd(&mut d2, &r, &qkv, &att, AB, AT, AC, AH);
        for (a, b) in d1.iter().zip(&d2) {
            assert!((a + 1.0 - b).abs() < 1e-5);
        }
    }
}
