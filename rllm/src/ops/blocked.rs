//! Cache-blocked matmul kernels (safe Rust, autovectorized, no packing).
//!
//! Layout: x[n*k], w[m*k] (row-major, both reduce over contiguous k), out[n*m].
//!
//! Tile sizes:
//! - `LANES = 16`: independent accumulator lanes for dot products (one AVX-512
//!   f32 register; the autovectorizer cannot reassociate a scalar reduction,
//!   so the lanes are explicit). Also the chunk width for axpy loops.
//! - `RJ = 4`: w rows (fwd) / dout columns (bwd dx) / x rows (bwd dw) processed
//!   per register tile; each loaded x/w chunk is reused 4 times.
//! - `KB = 256`: k tile. A 4-row w tile plus one x chunk is ~5 KB, well inside L1.
//! - `JB = 64`: w rows per L2 block. A JB*KB w block is 64 KB, reused across all
//!   n rows of x (fwd) or all n rows of dout/x (bwd) before moving on.
//! - `IB = 64`: n rows per block in bwd so the dx/x row tiles stay cache-hot
//!   while sweeping a JB block.

const LANES: usize = 16;
const RJ: usize = 4;
const KB: usize = 256;
const JB: usize = 64;
const IB: usize = 64;

/// Dot products of one x chunk against RJ w chunks, all of equal length.
#[inline(always)]
fn dot4(x: &[f32], w0: &[f32], w1: &[f32], w2: &[f32], w3: &[f32]) -> [f32; RJ] {
    let len = x.len();
    let (w0, w1, w2, w3) = (&w0[..len], &w1[..len], &w2[..len], &w3[..len]);
    let mut a = [[0.0f32; LANES]; RJ];
    let body = len - len % LANES;
    let mut p = 0;
    while p < body {
        let xc = &x[p..p + LANES];
        let c0 = &w0[p..p + LANES];
        let c1 = &w1[p..p + LANES];
        let c2 = &w2[p..p + LANES];
        let c3 = &w3[p..p + LANES];
        for l in 0..LANES {
            a[0][l] += xc[l] * c0[l];
            a[1][l] += xc[l] * c1[l];
            a[2][l] += xc[l] * c2[l];
            a[3][l] += xc[l] * c3[l];
        }
        p += LANES;
    }
    let mut r = [0.0f32; RJ];
    for (rj, acc) in r.iter_mut().zip(&a) {
        *rj = acc.iter().sum();
    }
    for q in body..len {
        r[0] += x[q] * w0[q];
        r[1] += x[q] * w1[q];
        r[2] += x[q] * w2[q];
        r[3] += x[q] * w3[q];
    }
    r
}

#[inline(always)]
fn dot1(x: &[f32], w: &[f32]) -> f32 {
    let len = x.len();
    let w = &w[..len];
    let mut a = [0.0f32; LANES];
    let body = len - len % LANES;
    let mut p = 0;
    while p < body {
        let xc = &x[p..p + LANES];
        let wc = &w[p..p + LANES];
        for l in 0..LANES {
            a[l] += xc[l] * wc[l];
        }
        p += LANES;
    }
    let mut r: f32 = a.iter().sum();
    for q in body..len {
        r += x[q] * w[q];
    }
    r
}

/// out[n*m] = x[n*k] . w[m*k]^T (overwrites).
pub fn matmul_fwd(out: &mut [f32], x: &[f32], w: &[f32], n: usize, k: usize, m: usize) {
    assert_eq!(out.len(), n * m);
    assert_eq!(x.len(), n * k);
    assert_eq!(w.len(), m * k);
    if k == 0 {
        out.fill(0.0);
        return;
    }
    for j0 in (0..m).step_by(JB) {
        let j1 = (j0 + JB).min(m);
        for k0 in (0..k).step_by(KB) {
            let k1 = (k0 + KB).min(k);
            let first = k0 == 0;
            for i in 0..n {
                let xr = &x[i * k + k0..i * k + k1];
                let orow = &mut out[i * m..(i + 1) * m];
                let mut j = j0;
                while j + RJ <= j1 {
                    let wr = |c: usize| &w[(j + c) * k + k0..(j + c) * k + k1];
                    let r = dot4(xr, wr(0), wr(1), wr(2), wr(3));
                    for c in 0..RJ {
                        if first {
                            orow[j + c] = r[c];
                        } else {
                            orow[j + c] += r[c];
                        }
                    }
                    j += RJ;
                }
                while j < j1 {
                    let r = dot1(xr, &w[j * k + k0..j * k + k1]);
                    if first {
                        orow[j] = r;
                    } else {
                        orow[j] += r;
                    }
                    j += 1;
                }
            }
        }
    }
}

/// dx[i,:] += sum_j dout[i,j] * w[j,:], four j at a time (contiguous over k).
#[inline(always)]
fn axpy4(dst: &mut [f32], d: [f32; RJ], s0: &[f32], s1: &[f32], s2: &[f32], s3: &[f32]) {
    let len = dst.len();
    let (s0, s1, s2, s3) = (&s0[..len], &s1[..len], &s2[..len], &s3[..len]);
    for p in 0..len {
        dst[p] += d[0] * s0[p] + d[1] * s1[p] + d[2] * s2[p] + d[3] * s3[p];
    }
}

#[inline(always)]
fn axpy1(dst: &mut [f32], d: f32, s: &[f32]) {
    let s = &s[..dst.len()];
    for p in 0..dst.len() {
        dst[p] += d * s[p];
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
    if k == 0 {
        return;
    }
    for j0 in (0..m).step_by(JB) {
        let j1 = (j0 + JB).min(m);
        for k0 in (0..k).step_by(KB) {
            let k1 = (k0 + KB).min(k);
            for i0 in (0..n).step_by(IB) {
                let i1 = (i0 + IB).min(n);
                // dx rows: 4 w rows per pass.
                for i in i0..i1 {
                    let drow = &dout[i * m..(i + 1) * m];
                    let dxr = &mut dx[i * k + k0..i * k + k1];
                    let mut j = j0;
                    while j + RJ <= j1 {
                        let wr = |c: usize| &w[(j + c) * k + k0..(j + c) * k + k1];
                        let d = [drow[j], drow[j + 1], drow[j + 2], drow[j + 3]];
                        axpy4(dxr, d, wr(0), wr(1), wr(2), wr(3));
                        j += RJ;
                    }
                    while j < j1 {
                        axpy1(dxr, drow[j], &w[j * k + k0..j * k + k1]);
                        j += 1;
                    }
                }
                // dw rows: 4 x rows per pass.
                for j in j0..j1 {
                    let dwr = &mut dw[j * k + k0..j * k + k1];
                    let mut i = i0;
                    while i + RJ <= i1 {
                        let xr = |c: usize| &x[(i + c) * k + k0..(i + c) * k + k1];
                        let d = [
                            dout[i * m + j],
                            dout[(i + 1) * m + j],
                            dout[(i + 2) * m + j],
                            dout[(i + 3) * m + j],
                        ];
                        axpy4(dwr, d, xr(0), xr(1), xr(2), xr(3));
                        i += RJ;
                    }
                    while i < i1 {
                        axpy1(dwr, dout[i * m + j], &x[i * k + k0..i * k + k1]);
                        i += 1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::scalar;
    use crate::rng::Rng;

    fn randn(rng: &mut Rng, len: usize) -> Vec<f32> {
        (0..len).map(|_| rng.normal()).collect()
    }

    fn assert_close(got: &[f32], want: &[f32], red: usize, what: &str) {
        let maxref = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let tol = 1e-4 * red as f32 * maxref;
        let maxd = got.iter().zip(want).fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
        assert!(maxd <= tol, "{what}: max|d|={maxd} > tol={tol}");
    }

    // Last extra shape crosses the KB/JB/IB tile edges with odd remainders.
    const SHAPES: [(usize, usize, usize); 5] =
        [(1, 1, 1), (7, 33, 19), (64, 128, 384), (300, 128, 2048), (67, 517, 71)];

    #[test]
    fn blocked_matches_scalar() {
        let mut rng = Rng::new(7);
        for &(n, k, m) in &SHAPES {
            let x = randn(&mut rng, n * k);
            let w = randn(&mut rng, m * k);
            let dout = randn(&mut rng, n * m);

            let mut a = vec![5.0; n * m];
            let mut b = vec![-3.0; n * m];
            scalar::matmul_fwd(&mut a, &x, &w, n, k, m);
            matmul_fwd(&mut b, &x, &w, n, k, m);
            assert_close(&b, &a, k, &format!("fwd {n}x{k}x{m}"));

            let dx0 = randn(&mut rng, n * k);
            let dw0 = randn(&mut rng, m * k);
            let (mut dxa, mut dwa) = (dx0.clone(), dw0.clone());
            let (mut dxb, mut dwb) = (dx0, dw0);
            scalar::matmul_bwd(&mut dxa, &mut dwa, &dout, &x, &w, n, k, m);
            matmul_bwd(&mut dxb, &mut dwb, &dout, &x, &w, n, k, m);
            assert_close(&dxb, &dxa, m, &format!("dx {n}x{k}x{m}"));
            assert_close(&dwb, &dwa, n, &format!("dw {n}x{k}x{m}"));
        }
    }

    #[test]
    #[ignore]
    fn blocked_timing_print() {
        let (n, k, m) = (4096, 128, 384);
        let mut rng = Rng::new(1);
        let x = randn(&mut rng, n * k);
        let w = randn(&mut rng, m * k);
        let mut out = vec![0.0; n * m];
        let t = std::time::Instant::now();
        for _ in 0..5 {
            matmul_fwd(&mut out, &x, &w, n, k, m);
        }
        println!("blocked fwd {n}x{k}x{m}: {:?}/iter", t.elapsed() / 5);
    }
}
