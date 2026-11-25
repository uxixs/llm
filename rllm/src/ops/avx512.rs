//! AVX-512 matmul and attention kernels. Same signatures and semantics as
//! `ops::scalar`; the `*_mt` variants (end of file) take a `&ThreadPool` first
//! and are bit-identical to the single-threaded ones. Requires avx512f only (masked loads/stores, FMA,
//! horizontal reduce); each public fn asserts it at runtime.
//!
//! # Matmul microkernel
//! All three products are one GEMM shape, C[rows x cols] (+)= A[rows x depth] .
//! B[depth x cols], where each B row is contiguous along `cols`:
//! - fwd: C=out, A=x, B=W^T packed into 64-column panels (`pack_wt`).
//! - dx:  C=dx,  A=dout, B=w as-is (w[j, p..] is contiguous in p).
//! - dw:  C=dw,  A=dout^T (strided scalar reads of dout), B=x as-is.
//!
//! The microkernel keeps an `MR x NV` = 6 x 4 tile of zmm accumulators (6 rows
//! x 64 cols, 24 regs) plus 4 B vectors and 1 broadcast A value: 29 of the 32
//! zmm registers. Per depth step: 4 loads, 6 broadcasts, 24 FMAs, so the single
//! 512-bit FMA port, not the load ports, is the limit. Column tails use masked
//! loads/stores (masked-off lanes are never accessed); row tails use the same
//! kernel instantiated with MR = 1..5.
//!
//! Blocking: depth in chunks of `DB` (a 256 x 64 B slice is 64 KB, reused from
//! L2 across a row block), rows in blocks of `IB`. C is loaded into the
//! accumulators for every depth chunk after the first (and for the first when
//! the op accumulates), so depth blocking needs no extra scratch.
//!
//! # Row ranges (for multithreading)
//! `matmul_fwd_rows`, `matmul_bwd_dx_rows` and `matmul_bwd_dw_rows` work on a
//! contiguous row range of `out` / `dx` / `dw`, so a caller can hand disjoint
//! `chunks_mut` to threads. `matmul_fwd` packs W into a thread-local buffer; a
//! parallel caller should `pack_wt` once and share the panels read-only.
//! Attention is computed per (batch, head) by `attn_fwd_head` /
//! `attn_bwd_head`; each pair owns a disjoint region of out / att / dqkv, so
//! `attn_*_mt` split over b*h (batch-only splits also work on sub-slices).
//!
//! # Scratch
//! `matmul_fwd` reuses a `thread_local!` pack buffer (grows to the largest
//! `packed_len` seen, never shrinks). `attn_bwd` reuses a `thread_local!`
//! length-t buffer for dp/ds. No per-call heap allocation after warm-up.
#![warn(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;
use std::cell::RefCell;

use crate::pool::ThreadPool;

/// f32 lanes per zmm.
const L: usize = 16;
/// zmm vectors per microkernel row.
const NV: usize = 4;
/// Columns per microkernel tile = packed panel width.
const NR: usize = NV * L;
/// Rows per microkernel tile.
const MR: usize = 6;
/// Depth chunk.
const DB: usize = 256;
/// Row block (a multiple of MR).
const IB: usize = 96;

fn require_avx512() {
    assert!(
        is_x86_feature_detected!("avx512f"),
        "ops::avx512 requires a CPU with AVX-512F (avx512f not detected)"
    );
}

/// Mask with the low `min(len, 16)` lanes set.
#[inline(always)]
fn tail_mask(len: usize) -> __mmask16 {
    if len >= L {
        0xFFFF
    } else {
        ((1u32 << len) - 1) as __mmask16
    }
}

thread_local! {
    static PACK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static DS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------- matmul

/// Length of the packed W^T buffer that `pack_wt` fills.
pub fn packed_len(k: usize, m: usize) -> usize {
    m.div_ceil(NR) * NR * k
}

/// Packs w[m x k] into panels of NR columns: panel `jb` holds, for each p in
/// 0..k, the NR values w[jb*NR + jj, p] (zero past m).
pub fn pack_wt(packed: &mut [f32], w: &[f32], k: usize, m: usize) {
    assert_eq!(w.len(), m * k);
    assert_eq!(packed.len(), packed_len(k, m));
    if k == 0 {
        return;
    }
    for (jb, panel) in packed.chunks_exact_mut(NR * k).enumerate() {
        for (p, dst) in panel.as_chunks_mut::<NR>().0.iter_mut().enumerate() {
            for (jj, d) in dst.iter_mut().enumerate() {
                let j = jb * NR + jj;
                *d = if j < m { w[j * k + p] } else { 0.0 };
            }
        }
    }
}

/// C[R x ncols] (+)= sum_p A[r, p] * B[p, 0..ncols], with ncols <= NR.
/// A[r, p] = *a.add(r*a_rs + p*a_cs); B row p starts at b.add(p*b_rs);
/// C row r starts at c.add(r*c_rs). If `acc_in`, C is read first, else overwritten.
///
/// # Safety
/// avx512f is available; for all r < R, p < depth, col < ncols the A, B and C
/// elements above lie inside their allocations; C does not overlap A or B.
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro<const R: usize>(
    a: *const f32,
    a_rs: usize,
    a_cs: usize,
    b: *const f32,
    b_rs: usize,
    c: *mut f32,
    c_rs: usize,
    depth: usize,
    ncols: usize,
    acc_in: bool,
) {
    // Vectors past ncols get mask 0 and offset 0, so their (in-bounds) pointer
    // touches no memory.
    let mut mask = [0 as __mmask16; NV];
    let mut voff = [0usize; NV];
    for v in 0..NV {
        let lanes = ncols.saturating_sub(v * L);
        if lanes > 0 {
            mask[v] = tail_mask(lanes);
            voff[v] = v * L;
        }
    }
    let mut acc = [[_mm512_setzero_ps(); NV]; R];
    if acc_in {
        for (r, accr) in acc.iter_mut().enumerate() {
            for v in 0..NV {
                // SAFETY: C row r is in bounds for cols < ncols; lanes at or
                // past ncols are masked off and not accessed.
                accr[v] = unsafe { _mm512_maskz_loadu_ps(mask[v], c.add(r * c_rs + voff[v])) };
            }
        }
    }
    for p in 0..depth {
        let mut bv = [_mm512_setzero_ps(); NV];
        for v in 0..NV {
            // SAFETY: B row p is in bounds for cols < ncols; other lanes masked.
            bv[v] = unsafe { _mm512_maskz_loadu_ps(mask[v], b.add(p * b_rs + voff[v])) };
        }
        for (r, accr) in acc.iter_mut().enumerate() {
            // SAFETY: A[r, p] is in bounds by the caller contract.
            let av = _mm512_set1_ps(unsafe { *a.add(r * a_rs + p * a_cs) });
            for v in 0..NV {
                accr[v] = _mm512_fmadd_ps(av, bv[v], accr[v]);
            }
        }
    }
    for (r, accr) in acc.iter().enumerate() {
        for v in 0..NV {
            // SAFETY: as for the C load; masked-off lanes are not written.
            unsafe { _mm512_mask_storeu_ps(c.add(r * c_rs + voff[v]), mask[v], accr[v]) };
        }
    }
}

/// Blocked driver: C[rows x cols] (+)= A[rows x depth] . B[depth x cols].
/// Column block `cb` (cols cb*NR..) of B starts at b.add(cb*b_blk); its row p
/// at a further p*b_rs.
///
/// # Safety
/// avx512f is available; every A[i, p], B(p, col) and C[i, col] for i < rows,
/// p < depth, col < cols is in bounds under the addressing of `micro` and the
/// block stride above; C does not overlap A or B. Requires depth > 0 when
/// `acc_in` is false (otherwise C would not be written).
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm(
    c: *mut f32,
    c_rs: usize,
    rows: usize,
    cols: usize,
    a: *const f32,
    a_rs: usize,
    a_cs: usize,
    b: *const f32,
    b_rs: usize,
    b_blk: usize,
    depth: usize,
    acc_in: bool,
) {
    let mut d0 = 0;
    while d0 < depth {
        let dl = DB.min(depth - d0);
        let acc = acc_in || d0 > 0;
        let mut i0 = 0;
        while i0 < rows {
            let i1 = (i0 + IB).min(rows);
            for cb in 0..cols.div_ceil(NR) {
                let j0 = cb * NR;
                let nc = NR.min(cols - j0);
                // SAFETY (this block): i + tile rows <= i1 <= rows, d0 + dl <=
                // depth and j0 + nc <= cols, so each offset stays inside its
                // operand and every element `micro` touches is one the caller
                // guarantees in bounds.
                unsafe {
                    let bp = b.add(cb * b_blk + d0 * b_rs);
                    let mut i = i0;
                    macro_rules! tile {
                        ($r:literal) => {{
                            let ap = a.add(i * a_rs + d0 * a_cs);
                            let cp = c.add(i * c_rs + j0);
                            micro::<$r>(ap, a_rs, a_cs, bp, b_rs, cp, c_rs, dl, nc, acc);
                        }};
                    }
                    while i + MR <= i1 {
                        tile!(6);
                        i += MR;
                    }
                    match i1 - i {
                        0 => {}
                        1 => tile!(1),
                        2 => tile!(2),
                        3 => tile!(3),
                        4 => tile!(4),
                        _ => tile!(5),
                    }
                }
            }
            i0 = i1;
        }
        d0 += dl;
    }
}

/// out[rows x m] = x[rows x k] . W^T, with W packed by `pack_wt` (overwrites).
/// Works on any contiguous row range of the full product: pass those x rows.
pub fn matmul_fwd_rows(
    out: &mut [f32],
    x: &[f32],
    packed: &[f32],
    rows: usize,
    k: usize,
    m: usize,
) {
    require_avx512();
    assert_eq!(out.len(), rows * m);
    assert_eq!(x.len(), rows * k);
    assert_eq!(packed.len(), packed_len(k, m));
    if k == 0 {
        out.fill(0.0);
        return;
    }
    // SAFETY: avx512f checked, depth k > 0. A=x: (i, p) -> i*k + p < rows*k.
    // B=packed: panel cb, row p, col jj -> cb*NR*k + p*NR + jj < packed_len.
    // C=out: i*m + j < rows*m; out is a distinct &mut, so no overlap.
    unsafe {
        gemm(
            out.as_mut_ptr(),
            m,
            rows,
            m,
            x.as_ptr(),
            k,
            1,
            packed.as_ptr(),
            NR,
            NR * k,
            k,
            false,
        )
    };
}

/// dx[rows x k] += dout[rows x m] . w[m x k], for a contiguous row range of dx
/// (pass the matching dout rows).
pub fn matmul_bwd_dx_rows(
    dx: &mut [f32],
    dout: &[f32],
    w: &[f32],
    rows: usize,
    k: usize,
    m: usize,
) {
    require_avx512();
    assert_eq!(dx.len(), rows * k);
    assert_eq!(dout.len(), rows * m);
    assert_eq!(w.len(), m * k);
    // SAFETY: avx512f checked. A=dout: (i, j) -> i*m + j < rows*m. B=w: block
    // cb starts at col cb*NR, (j, p) -> j*k + p < m*k. C=dx: i*k + p < rows*k.
    unsafe {
        gemm(
            dx.as_mut_ptr(),
            k,
            rows,
            k,
            dout.as_ptr(),
            m,
            1,
            w.as_ptr(),
            k,
            NR,
            m,
            true,
        )
    };
}

/// dw[j0..j0+rows, :] += dout[:, j0..j0+rows]^T . x. `dw` is only that row
/// range (rows*k values); `dout` [n x m] and `x` [n x k] are full.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bwd_dw_rows(
    dw: &mut [f32],
    j0: usize,
    rows: usize,
    dout: &[f32],
    x: &[f32],
    n: usize,
    k: usize,
    m: usize,
) {
    require_avx512();
    assert!(
        j0 + rows <= m,
        "dw rows {j0}..{} out of range for m={m}",
        j0 + rows
    );
    assert_eq!(dw.len(), rows * k);
    assert_eq!(dout.len(), n * m);
    assert_eq!(x.len(), n * k);
    if rows == 0 || n == 0 {
        return;
    }
    // SAFETY: avx512f checked; n > 0 and j0 < m, so dout.add(j0) is in bounds.
    // A[r, i] = dout[j0 + r + i*m] <= (m-1) + (n-1)*m < n*m. B=x: block cb
    // starts at col cb*NR, (i, p) -> i*k + p < n*k. C=dw: r*k + p < rows*k.
    unsafe {
        gemm(
            dw.as_mut_ptr(),
            k,
            rows,
            k,
            dout.as_ptr().add(j0),
            1,
            m,
            x.as_ptr(),
            k,
            NR,
            n,
            true,
        )
    };
}

/// out[n x m] = x[n x k] . w[m x k]^T (overwrites `out`).
pub fn matmul_fwd(out: &mut [f32], x: &[f32], w: &[f32], n: usize, k: usize, m: usize) {
    require_avx512();
    assert_eq!(out.len(), n * m);
    assert_eq!(x.len(), n * k);
    assert_eq!(w.len(), m * k);
    let need = packed_len(k, m);
    PACK.with(|buf| {
        let mut buf = buf.borrow_mut();
        if buf.len() < need {
            buf.resize(need, 0.0);
        }
        let packed = &mut buf[..need];
        pack_wt(packed, w, k, m);
        matmul_fwd_rows(out, x, packed, n, k, m);
    });
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
    matmul_bwd_dx_rows(dx, dout, w, n, k, m);
    matmul_bwd_dw_rows(dw, 0, m, dout, x, n, k, m);
}

// ------------------------------------------------------------- attention

/// sum_j a[j]*b[j] over j < d.
///
/// # Safety
/// avx512f is available; a[..d] and b[..d] are in bounds.
#[target_feature(enable = "avx512f")]
unsafe fn dot(a: *const f32, b: *const f32, d: usize) -> f32 {
    let mut acc = _mm512_setzero_ps();
    let mut j = 0;
    while j < d {
        let mk = tail_mask(d - j);
        // SAFETY: lanes j..min(j+16, d) are in bounds; the rest are masked off.
        let (va, vb) = unsafe {
            (
                _mm512_maskz_loadu_ps(mk, a.add(j)),
                _mm512_maskz_loadu_ps(mk, b.add(j)),
            )
        };
        acc = _mm512_fmadd_ps(va, vb, acc);
        j += L;
    }
    _mm512_reduce_add_ps(acc)
}

/// y[..d] += alpha * x[..d].
///
/// # Safety
/// avx512f is available; x[..d] and y[..d] are in bounds and do not overlap.
#[target_feature(enable = "avx512f")]
unsafe fn axpy(y: *mut f32, alpha: f32, x: *const f32, d: usize) {
    let va = _mm512_set1_ps(alpha);
    let mut j = 0;
    while j < d {
        let mk = tail_mask(d - j);
        // SAFETY: lanes j..min(j+16, d) are in bounds; the rest are masked off.
        unsafe {
            let vy = _mm512_maskz_loadu_ps(mk, y.add(j));
            let vx = _mm512_maskz_loadu_ps(mk, x.add(j));
            _mm512_mask_storeu_ps(y.add(j), mk, _mm512_fmadd_ps(va, vx, vy));
        }
        j += L;
    }
}

/// y[..d] (+)= sum_s coef[s] * x(s)[..d], where x(s) = x.add(s*x_stride).
/// Each 16-lane chunk of y stays in a register across s.
///
/// # Safety
/// avx512f is available; y[..d] and every x(s)[..d] for s < coef.len() are in
/// bounds; y overlaps no x(s).
#[target_feature(enable = "avx512f")]
unsafe fn weighted_sum(
    y: *mut f32,
    acc_in: bool,
    coef: &[f32],
    x: *const f32,
    x_stride: usize,
    d: usize,
) {
    let mut j = 0;
    while j < d {
        let mk = tail_mask(d - j);
        let mut acc = _mm512_setzero_ps();
        if acc_in {
            // SAFETY: y lanes < d are in bounds; the rest are masked off.
            acc = unsafe { _mm512_maskz_loadu_ps(mk, y.add(j)) };
        }
        for (s, &cs) in coef.iter().enumerate() {
            // SAFETY: x(s) lanes < d are in bounds; the rest are masked off.
            let vx = unsafe { _mm512_maskz_loadu_ps(mk, x.add(s * x_stride + j)) };
            acc = _mm512_fmadd_ps(_mm512_set1_ps(cs), vx, acc);
        }
        // SAFETY: as for the y load.
        unsafe { _mm512_mask_storeu_ps(y.add(j), mk, acc) };
        j += L;
    }
}

/// Checks features and shapes shared by attn_fwd/bwd; returns head_dim d.
/// `o_len` is the length of out (fwd) or dout (bwd).
fn attn_check(
    qkv: &[f32],
    att: &[f32],
    o_len: usize,
    b: usize,
    t: usize,
    c: usize,
    h: usize,
) -> usize {
    require_avx512();
    assert!(
        h > 0 && c > 0 && c.is_multiple_of(h),
        "c={c} must be a positive multiple of h={h}"
    );
    assert_eq!(qkv.len(), b * t * 3 * c);
    assert_eq!(att.len(), b * h * t * t);
    assert_eq!(o_len, b * t * c);
    c / h
}

/// Causal multi-head attention, same layouts as `scalar::attn_fwd`: qkv
/// `[b,t,3c]`, out `[b,t,c]`, att `[b,h,t,t]`. Overwrites out and att (future
/// entries 0). Scale 1/sqrt(c/h).
pub fn attn_fwd(
    out: &mut [f32],
    att: &mut [f32],
    qkv: &[f32],
    b: usize,
    t: usize,
    c: usize,
    h: usize,
) {
    let d = attn_check(qkv, att, out.len(), b, t, c, h);
    let op = out.as_mut_ptr();
    for bh in 0..b * h {
        let att_bh = &mut att[bh * t * t..][..t * t];
        // SAFETY: shapes checked by attn_check; `op` covers all of `out`, which
        // is exclusively borrowed here and distinct from qkv and att.
        unsafe { attn_fwd_head(op, att_bh, qkv, bh / h, bh % h, t, c, d) };
    }
}

/// One (batch bi, head hi) of `attn_fwd`: fills `att_bh` (that head's `[t,t]`
/// block of att) and out columns hi*d..hi*d+d of batch bi's t rows.
///
/// # Safety
/// avx512f is available; shapes were validated by `attn_check` (qkv is
/// `[b,t,3c]`, bi < b, hi < h, d = c/h); `out` points to a writable `[b,t,c]`
/// buffer that overlaps neither qkv nor att_bh, and no other thread accesses
/// the (bi, hi) out region during the call.
#[allow(clippy::too_many_arguments)]
unsafe fn attn_fwd_head(
    out: *mut f32,
    att_bh: &mut [f32],
    qkv: &[f32],
    bi: usize,
    hi: usize,
    t: usize,
    c: usize,
    d: usize,
) {
    let scale = 1.0 / (d as f32).sqrt();
    let c3 = 3 * c;
    let qp = qkv.as_ptr();
    for tq in 0..t {
        let q_off = (bi * t + tq) * c3 + hi * d;
        let row = &mut att_bh[tq * t..][..t];
        let mut max = f32::NEG_INFINITY;
        for (s, r) in row[..=tq].iter_mut().enumerate() {
            let k_off = (bi * t + s) * c3 + c + hi * d;
            // SAFETY: avx512f available; hi*d + d <= c and each token row
            // has 3c values, so q_off + d and k_off + d <= qkv.len().
            *r = unsafe { dot(qp.add(q_off), qp.add(k_off), d) } * scale;
            max = max.max(*r);
        }
        let mut sum = 0.0;
        for p in &mut row[..=tq] {
            *p = (*p - max).exp();
            sum += *p;
        }
        for p in &mut row[..=tq] {
            *p /= sum;
        }
        row[tq + 1..].fill(0.0);
        let o_off = (bi * t + tq) * c + hi * d;
        let v0 = bi * t * c3 + 2 * c + hi * d;
        // SAFETY: avx512f available; o_off + d <= b*t*c, inside out (caller
        // contract), and in the (bi, hi) region this call owns; v rows
        // s <= tq < t at v0 + s*c3 .. +d are inside qkv, distinct from out.
        unsafe { weighted_sum(out.add(o_off), false, &row[..=tq], qp.add(v0), c3, d) };
    }
}

/// Backward of `attn_fwd`; ACCUMULATES into `dqkv` (`[b,t,3c]`). Uses the stored `att`.
#[allow(clippy::too_many_arguments)]
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
    let d = attn_check(qkv, att, dout.len(), b, t, c, h);
    assert_eq!(dqkv.len(), qkv.len());
    let gp = dqkv.as_mut_ptr();
    DS.with(|buf| {
        let mut buf = buf.borrow_mut();
        if buf.len() < t {
            buf.resize(t, 0.0);
        }
        for bh in 0..b * h {
            // SAFETY: shapes checked; `gp` covers all of `dqkv`, which is
            // exclusively borrowed here and distinct from dout, qkv and att.
            unsafe {
                attn_bwd_head(
                    gp,
                    &mut buf[..t],
                    dout,
                    qkv,
                    att,
                    bh / h,
                    bh % h,
                    t,
                    c,
                    h,
                    d,
                )
            };
        }
    });
}

/// One (batch bi, head hi) of `attn_bwd`. Accumulates into the q, k and v
/// columns hi*d..hi*d+d of batch bi's t rows of dqkv. `ds` is length-t scratch.
///
/// # Safety
/// avx512f is available; shapes were validated by `attn_check` and
/// dqkv.len() == qkv.len() (bi < b, hi < h, d = c/h); `dqkv` points to a
/// writable `[b,t,3c]` buffer that overlaps none of ds, dout, qkv, att, and no
/// other thread accesses the (bi, hi) dqkv region during the call.
#[allow(clippy::too_many_arguments)]
unsafe fn attn_bwd_head(
    dqkv: *mut f32,
    ds_buf: &mut [f32],
    dout: &[f32],
    qkv: &[f32],
    att: &[f32],
    bi: usize,
    hi: usize,
    t: usize,
    c: usize,
    h: usize,
    d: usize,
) {
    let scale = 1.0 / (d as f32).sqrt();
    let c3 = 3 * c;
    let qp = qkv.as_ptr();
    let gp = dqkv;
    for tq in 0..t {
        let q_off = (bi * t + tq) * c3 + hi * d;
        let do_ = &dout[(bi * t + tq) * c + hi * d..][..d];
        let p = &att[((bi * h + hi) * t + tq) * t..][..=tq];
        let ds = &mut ds_buf[..=tq];
        // dp[s] = dout . v_s; dv_s += p[s] * dout.
        let mut pdp = 0.0;
        for (s, (dsv, &ps)) in ds.iter_mut().zip(p).enumerate() {
            let v_off = (bi * t + s) * c3 + 2 * c + hi * d;
            // SAFETY: avx512f available; v_off + d <= qkv.len() == dqkv
            // len, in the (bi, hi) region; do_ lives in dout, distinct.
            let dp = unsafe {
                axpy(gp.add(v_off), ps, do_.as_ptr(), d);
                dot(do_.as_ptr(), qp.add(v_off), d)
            };
            *dsv = dp;
            pdp += ps * dp;
        }
        // Softmax Jacobian: ds = p * (dp - sum(p*dp)) * scale.
        for (dsv, &ps) in ds.iter_mut().zip(p) {
            *dsv = ps * (*dsv - pdp) * scale;
        }
        let k0 = bi * t * c3 + c + hi * d;
        // SAFETY: avx512f available. dq at q_off..+d and dk rows at
        // k0 + s*c3 ..+d (s <= tq < t) are inside dqkv and the (bi, hi)
        // region; all reads are from qkv, a distinct slice, and dq (cols < c)
        // never overlaps a dk row (cols c..2c).
        unsafe {
            weighted_sum(gp.add(q_off), true, ds, qp.add(k0), c3, d);
            for (s, &dsv) in ds.iter().enumerate() {
                axpy(gp.add(k0 + s * c3), dsv, qp.add(q_off), d);
            }
        }
    }
}

// ------------------------------------------------------- multithreaded (12a)
//
// Each `_mt` kernel splits its output into disjoint ranges (rows of out / dx /
// dw, or (batch, head) pairs for attention) and runs the single-threaded inner
// fns on them, so per-element arithmetic, and therefore every result bit, is
// the same as the single-threaded kernel. About 4 tasks per pool thread for
// load balance; matmul row chunks are a multiple of MR so only the last chunk
// has a row tail.

/// Raw output pointer shared by tasks that write disjoint ranges of it.
#[derive(Clone, Copy)]
struct SyncPtr(*mut f32);
// SAFETY: a SyncPtr is only created from a `&mut [f32]` that the creating
// `_mt` fn holds exclusively for the whole `pool.run` call, and every task
// only touches a range of it that no other task touches (each use site states
// why its ranges are disjoint). So concurrent access never aliases.
unsafe impl Send for SyncPtr {}
// SAFETY: as above.
unsafe impl Sync for SyncPtr {}

impl SyncPtr {
    fn get(self) -> *mut f32 {
        self.0
    }

    /// `&mut buf[off..off + len]` of the buffer this pointer came from.
    ///
    /// # Safety
    /// off + len <= that buffer's length, the buffer outlives `'a`, and no
    /// other reference or task accesses that range during `'a`.
    unsafe fn slice<'a>(self, off: usize, len: usize) -> &'a mut [f32] {
        // SAFETY: caller contract.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

/// (chunk, n_tasks) splitting `total` items into ~4 tasks per pool thread,
/// with chunk a multiple of `align`.
fn split(total: usize, align: usize, pool: &ThreadPool) -> (usize, usize) {
    if total == 0 {
        return (align, 0);
    }
    let chunk = total.div_ceil(4 * pool.n_threads()).next_multiple_of(align);
    (chunk, total.div_ceil(chunk))
}

/// Multithreaded `matmul_fwd`: W is packed once on the calling thread (its
/// thread-local buffer) and shared read-only; tasks own row ranges of out.
pub fn matmul_fwd_mt(
    pool: &ThreadPool,
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    n: usize,
    k: usize,
    m: usize,
) {
    require_avx512();
    assert_eq!(out.len(), n * m);
    assert_eq!(x.len(), n * k);
    assert_eq!(w.len(), m * k);
    let need = packed_len(k, m);
    PACK.with(|buf| {
        let mut buf = buf.borrow_mut();
        if buf.len() < need {
            buf.resize(need, 0.0);
        }
        pack_wt(&mut buf[..need], w, k, m);
        let packed: &[f32] = &buf[..need];
        let (chunk, n_tasks) = split(n, MR, pool);
        let op = SyncPtr(out.as_mut_ptr());
        pool.run(n_tasks, &|i| {
            let r0 = i * chunk;
            let rows = chunk.min(n - r0);
            // SAFETY: task i owns out rows r0..r0+rows; rows ranges of distinct
            // i are disjoint and (r0 + rows) * m <= n * m = out.len(); out is
            // exclusively borrowed by this fn until run returns.
            let o = unsafe { op.slice(r0 * m, rows * m) };
            matmul_fwd_rows(o, &x[r0 * k..][..rows * k], packed, rows, k, m);
        });
    });
}

/// Multithreaded `matmul_bwd` (accumulates): one run whose first tasks own
/// row ranges of dx and whose remaining tasks own row ranges of dw. dx and dw
/// are distinct buffers, so there are no races and no reductions.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bwd_mt(
    pool: &ThreadPool,
    dx: &mut [f32],
    dw: &mut [f32],
    dout: &[f32],
    x: &[f32],
    w: &[f32],
    n: usize,
    k: usize,
    m: usize,
) {
    require_avx512();
    assert_eq!(dx.len(), n * k);
    assert_eq!(dw.len(), m * k);
    assert_eq!(dout.len(), n * m);
    assert_eq!(x.len(), n * k);
    assert_eq!(w.len(), m * k);
    let (cx, tx) = split(n, MR, pool);
    let (cw, tw) = split(m, MR, pool);
    let dxp = SyncPtr(dx.as_mut_ptr());
    let dwp = SyncPtr(dw.as_mut_ptr());
    pool.run(tx + tw, &|i| {
        if i < tx {
            let r0 = i * cx;
            let rows = cx.min(n - r0);
            // SAFETY: dx tasks own disjoint row ranges r0..r0+rows of dx,
            // (r0 + rows) * k <= dx.len(); dx is exclusively borrowed by this
            // fn and is distinct from dw.
            let d = unsafe { dxp.slice(r0 * k, rows * k) };
            matmul_bwd_dx_rows(d, &dout[r0 * m..][..rows * m], w, rows, k, m);
        } else {
            let j0 = (i - tx) * cw;
            let rows = cw.min(m - j0);
            // SAFETY: dw tasks own disjoint row ranges j0..j0+rows of dw,
            // (j0 + rows) * k <= dw.len(); dw is exclusively borrowed by this
            // fn and is distinct from dx.
            let d = unsafe { dwp.slice(j0 * k, rows * k) };
            matmul_bwd_dw_rows(d, j0, rows, dout, x, n, k, m);
        }
    });
}

/// Multithreaded `attn_fwd`: tasks own ranges of (batch, head) pairs.
#[allow(clippy::too_many_arguments)]
pub fn attn_fwd_mt(
    pool: &ThreadPool,
    out: &mut [f32],
    att: &mut [f32],
    qkv: &[f32],
    b: usize,
    t: usize,
    c: usize,
    h: usize,
) {
    let d = attn_check(qkv, att, out.len(), b, t, c, h);
    let (chunk, n_tasks) = split(b * h, 1, pool);
    let op = SyncPtr(out.as_mut_ptr());
    let ap = SyncPtr(att.as_mut_ptr());
    pool.run(n_tasks, &|i| {
        for bh in i * chunk..((i + 1) * chunk).min(b * h) {
            // SAFETY: each (bi, hi) pair belongs to exactly one task. Its att
            // block bh*t*t..+t*t (< b*h*t*t = att.len()) and its out region
            // (rows of batch bi, cols hi*d..+d) are touched by no other pair;
            // out and att are exclusively borrowed by this fn and distinct
            // from each other and from qkv; shapes checked by attn_check.
            unsafe {
                let att_bh = ap.slice(bh * t * t, t * t);
                attn_fwd_head(op.get(), att_bh, qkv, bh / h, bh % h, t, c, d);
            }
        }
    });
}

/// Multithreaded `attn_bwd` (accumulates into dqkv): tasks own ranges of
/// (batch, head) pairs; dp/ds scratch is each executing thread's `DS`.
#[allow(clippy::too_many_arguments)]
pub fn attn_bwd_mt(
    pool: &ThreadPool,
    dqkv: &mut [f32],
    dout: &[f32],
    qkv: &[f32],
    att: &[f32],
    b: usize,
    t: usize,
    c: usize,
    h: usize,
) {
    let d = attn_check(qkv, att, dout.len(), b, t, c, h);
    assert_eq!(dqkv.len(), qkv.len());
    let (chunk, n_tasks) = split(b * h, 1, pool);
    let gp = SyncPtr(dqkv.as_mut_ptr());
    pool.run(n_tasks, &|i| {
        DS.with(|buf| {
            let mut buf = buf.borrow_mut();
            if buf.len() < t {
                buf.resize(t, 0.0);
            }
            for bh in i * chunk..((i + 1) * chunk).min(b * h) {
                // SAFETY: each (bi, hi) pair belongs to exactly one task, and
                // its dqkv region (rows of batch bi; q, k, v cols of head hi)
                // is touched by no other pair; dqkv is exclusively borrowed by
                // this fn and distinct from dout, qkv, att and the thread's
                // own DS buffer; shapes checked above.
                unsafe {
                    attn_bwd_head(
                        gp.get(),
                        &mut buf[..t],
                        dout,
                        qkv,
                        att,
                        bh / h,
                        bh % h,
                        t,
                        c,
                        h,
                        d,
                    )
                };
            }
        });
    });
}

// ------------------------------------------------- softcap + cross-entropy
//
// `softcap_xent_fwd_bwd` fuses the model's head chain `softcap_fwd` ->
// `xent_fwd_bwd` -> `softcap_bwd` (into zeros) per row, in three passes over
// the row: (1) y = cap*tanh(x/cap) into dpre and the row max; (2) e =
// exp(y - max) into a thread-local row buffer and the row sum; (3) dpre =
// (e/sum - onehot) * scale * (1 - (y/cap)^2). exp is `exp_ps` (relative error
// <= 2e-7 on [-87, 0], see its test); tanh is built on the matching expm1 so it
// stays accurate near 0. Results are close to the scalar chain, not bit-equal
// (lane-order sums, own exp/tanh).

/// ln 2 split for Cody-Waite reduction: HI has few mantissa bits so n*HI is exact.
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
/// ln(2^-126): below it e^x would be subnormal; `exp_ps` returns 0 there.
const EXP_MIN: f32 = -87.336_54;
const EXP_MAX: f32 = 88.0;

/// Range reduction x = n*ln2 + r (|r| <= ln2/2) and q = e^r - 1 by a degree-7
/// Taylor polynomial (truncation < 6e-9 relative). Returns (2^n, q). Inputs
/// are clamped to [EXP_MIN, EXP_MAX].
///
/// # Safety
/// avx512f is available.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn exp_parts(x: __m512) -> (__m512, __m512) {
    let x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(EXP_MIN)), _mm512_set1_ps(EXP_MAX));
    let n = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm512_mul_ps(x, _mm512_set1_ps(std::f32::consts::LOG2_E)),
    );
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_HI), x);
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_LO), r);
    let mut p = _mm512_set1_ps(1.0 / 5040.0);
    for c in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0] {
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(c));
    }
    let q = _mm512_mul_ps(p, r);
    (_mm512_scalef_ps(_mm512_set1_ps(1.0), n), q)
}

/// e^x per lane as 2^n * (1 + q) with one final rounding. Max relative error
/// <= 2e-7 on [-87, 0] (tested); lanes below EXP_MIN return 0; NaN lanes stay
/// NaN (the clamp in `exp_parts` would otherwise make them finite).
///
/// # Safety
/// avx512f is available.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn exp_ps(x: __m512) -> __m512 {
    // SAFETY: avx512f is available (caller contract).
    let (s, q) = unsafe { exp_parts(x) };
    let under = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, _mm512_set1_ps(EXP_MIN));
    let e = _mm512_maskz_mov_ps(!under, _mm512_fmadd_ps(s, q, s));
    _mm512_mask_mov_ps(e, _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(x, x), x)
}

/// cap * tanh(x / cap) per lane, via tanh(a) = -m / (2 + m) with
/// m = expm1(-2|a|) = 2^n*q + (2^n - 1); 2^n - 1 is exact for the n that
/// matter, so this stays accurate near 0. The sign of x/cap is restored last.
/// NaN lanes stay NaN, like `f32::tanh` (a diverged run must not look finite).
///
/// # Safety
/// avx512f is available.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn softcap_ps(x: __m512, cap: __m512) -> __m512 {
    let z = _mm512_div_ps(x, cap);
    let sign_bit = _mm512_set1_epi32(i32::MIN);
    let zi = _mm512_castps_si512(z);
    let sign = _mm512_and_si512(zi, sign_bit);
    let a = _mm512_castsi512_ps(_mm512_andnot_si512(sign_bit, zi));
    // SAFETY: avx512f is available (caller contract).
    let (s, q) = unsafe { exp_parts(_mm512_mul_ps(a, _mm512_set1_ps(-2.0))) };
    let m = _mm512_fmadd_ps(s, q, _mm512_sub_ps(s, _mm512_set1_ps(1.0)));
    let t = _mm512_div_ps(
        _mm512_sub_ps(_mm512_setzero_ps(), m),
        _mm512_add_ps(_mm512_set1_ps(2.0), m),
    );
    let t = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(t), sign));
    // The clamp in `exp_parts` maps NaN to a finite value; put NaN back.
    let nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(z, z);
    _mm512_mask_mov_ps(_mm512_mul_ps(cap, t), nan, z)
}

thread_local! {
    /// One row of exp(y - max) for `softcap_xent_row` (per executing thread).
    static XE: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// Per-row loss terms of `softcap_xent_fwd_bwd_mt` (calling thread).
    static XLOSS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Checks features and shapes; returns sum(mask) (the same f32 sum as
/// `scalar::xent_fwd_bwd`).
fn softcap_xent_check(
    dpre: &[f32],
    pre: &[f32],
    targets: &[u16],
    mask: &[f32],
    n: usize,
    v: usize,
) -> f32 {
    require_avx512();
    assert_eq!(dpre.len(), n * v);
    assert_eq!(pre.len(), n * v);
    assert_eq!(targets.len(), n);
    assert_eq!(mask.len(), n);
    mask.iter().sum()
}

/// One row of the fused op: overwrites `drow` (len v) with d loss / d pre-cap
/// logits (already scaled by m/msum) and returns the row's loss term
/// `m * (ln sum + max - y_t)` in f64 (0 for a masked-out row). Uses the
/// executing thread's `XE` buffer. Callers have asserted avx512f.
fn softcap_xent_row(drow: &mut [f32], prow: &[f32], t: usize, m: f32, msum: f32, cap: f32) -> f64 {
    let v = prow.len();
    assert_eq!(drow.len(), v);
    assert!(t < v, "target {t} out of range for vocab {v}");
    if m == 0.0 {
        drow.fill(0.0);
        return 0.0;
    }
    let scale = m / msum;
    XE.with(|buf| {
        let mut buf = buf.borrow_mut();
        if buf.len() < v {
            buf.resize(v, 0.0);
        }
        // SAFETY: avx512f was asserted by the public entry points
        // (`softcap_xent_check`); drow, prow and the XE slice all have length
        // v > t (checked above) and are distinct buffers.
        let (sum, max, y_t) = unsafe { softcap_xent_row_simd(drow, prow, &mut buf[..v], t, scale, cap) };
        m as f64 * ((sum.ln() + max - y_t) as f64)
    })
}

/// SIMD body of `softcap_xent_row`; returns (sum, max, y_t).
///
/// # Safety
/// avx512f is available; drow, prow and e have the same length v > t and do
/// not overlap.
#[target_feature(enable = "avx512f")]
unsafe fn softcap_xent_row_simd(
    drow: &mut [f32],
    prow: &[f32],
    e: &mut [f32],
    t: usize,
    scale: f32,
    cap: f32,
) -> (f32, f32, f32) {
    let v = prow.len();
    let (dp, pp, ep) = (drow.as_mut_ptr(), prow.as_ptr(), e.as_mut_ptr());
    let capv = _mm512_set1_ps(cap);
    // Pass 1: y into drow; running max (masked-off lanes keep the old max).
    let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut j = 0;
    while j < v {
        let mk = tail_mask(v - j);
        // SAFETY: lanes j..min(j+16, v) are in bounds of prow and drow; the
        // rest are masked off.
        unsafe {
            let y = softcap_ps(_mm512_maskz_loadu_ps(mk, pp.add(j)), capv);
            _mm512_mask_storeu_ps(dp.add(j), mk, y);
            vmax = _mm512_mask_max_ps(vmax, mk, vmax, y);
        }
        j += L;
    }
    let max = _mm512_reduce_max_ps(vmax);
    // Pass 2: e = exp(y - max) into e; masked sum.
    let (maxv, mut vsum) = (_mm512_set1_ps(max), _mm512_setzero_ps());
    let mut j = 0;
    while j < v {
        let mk = tail_mask(v - j);
        // SAFETY: lanes j..min(j+16, v) are in bounds of drow and e; the rest
        // are masked off.
        unsafe {
            let ex = exp_ps(_mm512_sub_ps(_mm512_maskz_loadu_ps(mk, dp.add(j)), maxv));
            _mm512_mask_storeu_ps(ep.add(j), mk, ex);
            vsum = _mm512_mask_add_ps(vsum, mk, vsum, ex);
        }
        j += L;
    }
    let sum = _mm512_reduce_add_ps(vsum);
    let y_t = drow[t];
    // Pass 3: dpre = e * (scale/sum) * (1 - (y/cap)^2); the target is fixed below.
    let inv = scale / sum;
    let (invv, one) = (_mm512_set1_ps(inv), _mm512_set1_ps(1.0));
    let mut j = 0;
    while j < v {
        let mk = tail_mask(v - j);
        // SAFETY: lanes j..min(j+16, v) are in bounds of drow and e; the rest
        // are masked off.
        unsafe {
            let tt = _mm512_div_ps(_mm512_maskz_loadu_ps(mk, dp.add(j)), capv);
            let fac = _mm512_fnmadd_ps(tt, tt, one);
            let ex = _mm512_maskz_loadu_ps(mk, ep.add(j));
            _mm512_mask_storeu_ps(dp.add(j), mk, _mm512_mul_ps(_mm512_mul_ps(ex, invv), fac));
        }
        j += L;
    }
    let tt = y_t / cap;
    drow[t] = (e[t] * inv - scale) * (1.0 - tt * tt);
    (sum, max, y_t)
}

/// Fused `scalar::softcap_fwd` -> `scalar::xent_fwd_bwd` -> `scalar::softcap_bwd`
/// (into zeros): `pre` is `[n, v]` pre-cap logits; returns the masked mean xent
/// of the capped logits and OVERWRITES `dpre` with its gradient w.r.t. `pre`.
/// sum(mask) == 0 gives loss 0 and dpre 0. Close to, not bit-equal to, the chain.
pub fn softcap_xent_fwd_bwd(
    dpre: &mut [f32],
    pre: &[f32],
    targets: &[u16],
    mask: &[f32],
    n: usize,
    v: usize,
    cap: f32,
) -> f32 {
    let msum = softcap_xent_check(dpre, pre, targets, mask, n, v);
    if msum == 0.0 {
        dpre.fill(0.0);
        return 0.0;
    }
    let mut loss = 0.0f64;
    for i in 0..n {
        let r = i * v..(i + 1) * v;
        let (t, m) = (targets[i] as usize, mask[i]);
        loss += softcap_xent_row(&mut dpre[r.clone()], &pre[r], t, m, msum, cap);
    }
    (loss / msum as f64) as f32
}

/// Raw pointer to the per-row loss buffer, shared by tasks writing disjoint slots.
#[derive(Clone, Copy)]
struct SyncPtrF64(*mut f64);
// SAFETY: created only from the calling thread's XLOSS buffer, which
// `softcap_xent_fwd_bwd_mt` borrows mutably for the whole `pool.run`; each
// task writes only the slots of its own rows, so accesses never alias.
unsafe impl Send for SyncPtrF64 {}
// SAFETY: as above.
unsafe impl Sync for SyncPtrF64 {}

impl SyncPtrF64 {
    /// Method (not field) access, so closures capture the whole Sync wrapper.
    fn get(self) -> *mut f64 {
        self.0
    }
}

/// Multithreaded `softcap_xent_fwd_bwd`, bit-identical to it: tasks own row
/// ranges of dpre and store each row's loss term; the terms are then summed in
/// row order on the calling thread, exactly as the single-threaded loop does.
#[allow(clippy::too_many_arguments)]
pub fn softcap_xent_fwd_bwd_mt(
    pool: &ThreadPool,
    dpre: &mut [f32],
    pre: &[f32],
    targets: &[u16],
    mask: &[f32],
    n: usize,
    v: usize,
    cap: f32,
) -> f32 {
    let msum = softcap_xent_check(dpre, pre, targets, mask, n, v);
    if msum == 0.0 {
        dpre.fill(0.0);
        return 0.0;
    }
    XLOSS.with(|lb| {
        let mut lb = lb.borrow_mut();
        if lb.len() < n {
            lb.resize(n, 0.0);
        }
        let (chunk, n_tasks) = split(n, 1, pool);
        let (dp, lp) = (SyncPtr(dpre.as_mut_ptr()), SyncPtrF64(lb.as_mut_ptr()));
        pool.run(n_tasks, &|ti| {
            for i in ti * chunk..((ti + 1) * chunk).min(n) {
                // SAFETY: row i belongs to exactly one task; its dpre range
                // i*v..(i+1)*v (within n*v = dpre.len()) is touched by no other
                // task, and dpre is exclusively borrowed by this fn until run
                // returns.
                let drow = unsafe { dp.slice(i * v, v) };
                let r = i * v..(i + 1) * v;
                let term = softcap_xent_row(drow, &pre[r], targets[i] as usize, mask[i], msum, cap);
                // SAFETY: i < n <= lb.len(), and slot i is written only by the
                // task that owns row i; lb is not otherwise accessed during run.
                unsafe { *lp.get().add(i) = term };
            }
        });
        let loss = lb[..n].iter().fold(0.0f64, |acc, &x| acc + x);
        (loss / msum as f64) as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::scalar;
    use crate::rng::Rng;

    fn have_avx512() -> bool {
        let ok = is_x86_feature_detected!("avx512f");
        if !ok {
            println!("NOTE: avx512f not detected; skipping AVX-512 test");
        }
        ok
    }

    fn randn(rng: &mut Rng, len: usize) -> Vec<f32> {
        (0..len).map(|_| rng.normal()).collect()
    }

    fn max_abs(v: &[f32]) -> f32 {
        v.iter().fold(0.0f32, |a, x| a.max(x.abs()))
    }

    /// max|got - want| <= 1e-4 * scale * max|want|.
    fn assert_close(got: &[f32], want: &[f32], scale: usize, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let tol = 1e-4 * scale as f32 * max_abs(want);
        let maxd = got
            .iter()
            .zip(want)
            .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
        assert!(maxd <= tol, "{what}: max|d|={maxd} > tol={tol}");
    }

    const MM_SHAPES: [(usize, usize, usize); 5] = [
        (1, 1, 1),
        (7, 33, 19),
        (64, 128, 384),
        (300, 128, 2048),
        (67, 517, 71),
    ];

    #[test]
    fn avx512_matmul_matches_scalar() {
        if !have_avx512() {
            return;
        }
        let mut rng = Rng::new(11);
        for &(n, k, m) in &MM_SHAPES {
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

    /// Row-range entry points (for Task 12a) reproduce the full call when the
    /// rows are split at uneven points.
    #[test]
    fn avx512_matmul_row_ranges_match_full() {
        if !have_avx512() {
            return;
        }
        let (n, k, m) = (67, 45, 71);
        let mut rng = Rng::new(13);
        let x = randn(&mut rng, n * k);
        let w = randn(&mut rng, m * k);
        let dout = randn(&mut rng, n * m);
        let mut full = vec![0.0; n * m];
        matmul_fwd(&mut full, &x, &w, n, k, m);
        let mut packed = vec![0.0; packed_len(k, m)];
        pack_wt(&mut packed, &w, k, m);
        let mut split = vec![0.0; n * m];
        let (a, rest) = split.split_at_mut(13 * m);
        matmul_fwd_rows(a, &x[..13 * k], &packed, 13, k, m);
        matmul_fwd_rows(rest, &x[13 * k..], &packed, n - 13, k, m);
        assert_eq!(split, full);

        let dx0 = randn(&mut rng, n * k);
        let dw0 = randn(&mut rng, m * k);
        let (mut dxf, mut dwf) = (dx0.clone(), dw0.clone());
        matmul_bwd(&mut dxf, &mut dwf, &dout, &x, &w, n, k, m);
        let (mut dxs, mut dws) = (dx0, dw0);
        let (d1, d2) = dxs.split_at_mut(29 * k);
        matmul_bwd_dx_rows(d1, &dout[..29 * m], &w, 29, k, m);
        matmul_bwd_dx_rows(d2, &dout[29 * m..], &w, n - 29, k, m);
        let (w1, w2) = dws.split_at_mut(8 * k);
        matmul_bwd_dw_rows(w1, 0, 8, &dout, &x, n, k, m);
        matmul_bwd_dw_rows(w2, 8, m - 8, &dout, &x, n, k, m);
        assert_eq!(dxs, dxf);
        assert_eq!(dws, dwf);
    }

    #[test]
    #[ignore]
    fn avx512_timing_print() {
        if !have_avx512() {
            return;
        }
        let (n, k, m) = (4096, 128, 384);
        let mut rng = Rng::new(1);
        let x = randn(&mut rng, n * k);
        let w = randn(&mut rng, m * k);
        let dout = randn(&mut rng, n * m);
        let mut out = vec![0.0; n * m];
        let (mut dx, mut dw) = (vec![0.0; n * k], vec![0.0; m * k]);
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            matmul_fwd(&mut out, &x, &w, n, k, m);
        }
        println!("avx512 fwd {n}x{k}x{m}: {:?}/iter", t0.elapsed() / 5);
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            matmul_bwd(&mut dx, &mut dw, &dout, &x, &w, n, k, m);
        }
        println!("avx512 bwd {n}x{k}x{m}: {:?}/iter", t0.elapsed() / 5);
    }

    const ATTN_SHAPES: [(usize, usize, usize, usize); 4] =
        [(1, 1, 8, 1), (2, 7, 64, 2), (2, 256, 128, 4), (1, 9, 40, 2)];

    #[test]
    fn avx512_attn_matches_scalar() {
        if !have_avx512() {
            return;
        }
        let mut rng = Rng::new(12);
        for &(b, t, c, h) in &ATTN_SHAPES {
            let qkv = randn(&mut rng, b * t * 3 * c);
            let dout = randn(&mut rng, b * t * c);
            let what = format!("b{b} t{t} c{c} h{h}");

            let (mut oa, mut aa) = (vec![5.0; b * t * c], vec![5.0; b * h * t * t]);
            let (mut ob, mut ab) = (vec![-3.0; b * t * c], vec![-3.0; b * h * t * t]);
            scalar::attn_fwd(&mut oa, &mut aa, &qkv, b, t, c, h);
            attn_fwd(&mut ob, &mut ab, &qkv, b, t, c, h);
            assert_close(&ob, &oa, t, &format!("out {what}"));
            assert_close(&ab, &aa, t, &format!("att {what}"));

            let d0 = randn(&mut rng, qkv.len());
            let (mut da, mut db) = (d0.clone(), d0);
            scalar::attn_bwd(&mut da, &dout, &qkv, &aa, b, t, c, h);
            attn_bwd(&mut db, &dout, &qkv, &aa, b, t, c, h);
            assert_close(&db, &da, t, &format!("dqkv {what}"));
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    const MT_POOLS: [usize; 3] = [1, 3, 4];

    /// `_mt` matmuls are bit-identical to the single-threaded kernels.
    #[test]
    fn mt_matches_single_matmul() {
        if !have_avx512() {
            return;
        }
        let mut rng = Rng::new(21);
        for &size in &MT_POOLS {
            let pool = ThreadPool::new(size);
            for &(n, k, m) in &MM_SHAPES {
                let what = format!("pool {size} {n}x{k}x{m}");
                let x = randn(&mut rng, n * k);
                let w = randn(&mut rng, m * k);
                let dout = randn(&mut rng, n * m);

                let mut a = vec![5.0; n * m];
                let mut b = vec![-3.0; n * m];
                matmul_fwd(&mut a, &x, &w, n, k, m);
                matmul_fwd_mt(&pool, &mut b, &x, &w, n, k, m);
                assert_eq!(bits(&b), bits(&a), "fwd {what}");

                let dx0 = randn(&mut rng, n * k);
                let dw0 = randn(&mut rng, m * k);
                let (mut dxa, mut dwa) = (dx0.clone(), dw0.clone());
                let (mut dxb, mut dwb) = (dx0, dw0);
                matmul_bwd(&mut dxa, &mut dwa, &dout, &x, &w, n, k, m);
                matmul_bwd_mt(&pool, &mut dxb, &mut dwb, &dout, &x, &w, n, k, m);
                assert_eq!(bits(&dxb), bits(&dxa), "dx {what}");
                assert_eq!(bits(&dwb), bits(&dwa), "dw {what}");
            }
        }
    }

    /// `_mt` attention is bit-identical to the single-threaded kernels.
    #[test]
    fn mt_matches_single_attn() {
        if !have_avx512() {
            return;
        }
        let mut rng = Rng::new(22);
        for &size in &MT_POOLS {
            let pool = ThreadPool::new(size);
            for &(b, t, c, h) in &ATTN_SHAPES {
                let what = format!("pool {size} b{b} t{t} c{c} h{h}");
                let qkv = randn(&mut rng, b * t * 3 * c);
                let dout = randn(&mut rng, b * t * c);

                let (mut oa, mut aa) = (vec![5.0; b * t * c], vec![5.0; b * h * t * t]);
                let (mut ob, mut ab) = (vec![-3.0; b * t * c], vec![-3.0; b * h * t * t]);
                attn_fwd(&mut oa, &mut aa, &qkv, b, t, c, h);
                attn_fwd_mt(&pool, &mut ob, &mut ab, &qkv, b, t, c, h);
                assert_eq!(bits(&ob), bits(&oa), "out {what}");
                assert_eq!(bits(&ab), bits(&aa), "att {what}");

                let d0 = randn(&mut rng, qkv.len());
                let (mut da, mut db) = (d0.clone(), d0);
                attn_bwd(&mut da, &dout, &qkv, &aa, b, t, c, h);
                attn_bwd_mt(&pool, &mut db, &dout, &qkv, &aa, b, t, c, h);
                assert_eq!(bits(&db), bits(&da), "dqkv {what}");
            }
        }
    }

    /// Batch of 8 x 4 heads (the training shape) splits into several tasks.
    #[test]
    fn mt_attn_many_heads() {
        if !have_avx512() {
            return;
        }
        let (b, t, c, h) = (8, 33, 64, 4);
        let mut rng = Rng::new(23);
        let pool = ThreadPool::new(4);
        let qkv = randn(&mut rng, b * t * 3 * c);
        let dout = randn(&mut rng, b * t * c);
        let (mut oa, mut aa) = (vec![0.0; b * t * c], vec![0.0; b * h * t * t]);
        let (mut ob, mut ab) = (vec![1.0; b * t * c], vec![1.0; b * h * t * t]);
        attn_fwd(&mut oa, &mut aa, &qkv, b, t, c, h);
        attn_fwd_mt(&pool, &mut ob, &mut ab, &qkv, b, t, c, h);
        assert_eq!(bits(&ob), bits(&oa));
        assert_eq!(bits(&ab), bits(&aa));
        let d0 = randn(&mut rng, qkv.len());
        let (mut da, mut db) = (d0.clone(), d0);
        attn_bwd(&mut da, &dout, &qkv, &aa, b, t, c, h);
        attn_bwd_mt(&pool, &mut db, &dout, &qkv, &aa, b, t, c, h);
        assert_eq!(bits(&db), bits(&da));
    }

    /// Max relative error of `exp_ps` against f64 exp over [-87, 0].
    #[test]
    fn vec_exp_max_rel_error() {
        if !have_avx512() {
            return;
        }
        const N: usize = 1 << 22;
        let mut xs: Vec<f32> = (0..N).map(|i| -87.0 * i as f32 / (N - 1) as f32).collect();
        let ln2 = std::f32::consts::LN_2;
        xs.extend([0.0, -0.0, -1e-30, -1e-7, -0.34657, -0.34658, -ln2, -86.99999, -87.0]);
        let mut worst = (0.0f64, 0.0f32);
        for ch in xs.chunks(L) {
            let mk = tail_mask(ch.len());
            let mut out = [0.0f32; L];
            // SAFETY: avx512f checked; lanes < ch.len() are in bounds, rest masked.
            unsafe {
                let e = exp_ps(_mm512_maskz_loadu_ps(mk, ch.as_ptr()));
                _mm512_mask_storeu_ps(out.as_mut_ptr(), mk, e);
            }
            for (&x, &e) in ch.iter().zip(&out) {
                let r = (x as f64).exp();
                let rel = ((e as f64) - r).abs() / r;
                if rel > worst.0 {
                    worst = (rel, x);
                }
            }
        }
        println!("exp_ps max rel error {:.3e} at x = {}", worst.0, worst.1);
        assert!(worst.0 <= 2e-7, "exp_ps max rel error {} at x = {}", worst.0, worst.1);
    }

    /// The model's old chain: softcap_fwd -> xent_fwd_bwd -> softcap_bwd into zeros.
    fn softcap_xent_chain(
        pre: &[f32],
        tgt: &[u16],
        mask: &[f32],
        n: usize,
        v: usize,
        cap: f32,
    ) -> (f32, Vec<f32>) {
        let mut capped = vec![0.0; n * v];
        scalar::softcap_fwd(&mut capped, pre, cap);
        let mut dcap = vec![0.0; n * v];
        let loss = scalar::xent_fwd_bwd(&mut dcap, &capped, tgt, mask, n, v);
        let mut dpre = vec![0.0; n * v];
        scalar::softcap_bwd(&mut dpre, &dcap, &capped, cap);
        (loss, dpre)
    }

    /// (n, v, logit std, mask) cases; masks include zeros (and an all-zero mask).
    fn xent_cases() -> Vec<(usize, usize, f32, Vec<f32>)> {
        let m300: Vec<f32> = (0..300)
            .map(|i| if i % 7 == 3 { 0.0 } else if i % 5 == 0 { 0.5 } else { 1.0 })
            .collect();
        vec![
            (1, 7, 3.0, vec![1.0]),
            (1, 7, 3.0, vec![0.0]),
            (3, 2048, 10.0, vec![1.0, 0.0, 0.5]),
            (3, 2048, 50.0, vec![1.0, 1.0, 1.0]),
            (300, 2048, 5.0, m300.clone()),
            (300, 2048, 40.0, m300),
        ]
    }

    #[test]
    fn softcap_xent_matches_scalar_chain() {
        if !have_avx512() {
            return;
        }
        let mut rng = Rng::new(31);
        let cap = 30.0;
        for (n, v, std, mask) in xent_cases() {
            let what = format!("{n}x{v} std {std} mask0 {}", mask[0]);
            let pre: Vec<f32> = randn(&mut rng, n * v).iter().map(|x| x * std).collect();
            let tgt: Vec<u16> = (0..n).map(|_| rng.below(v) as u16).collect();
            let (l0, d0) = softcap_xent_chain(&pre, &tgt, &mask, n, v, cap);
            let mut d = vec![7.0; n * v];
            let l = softcap_xent_fwd_bwd(&mut d, &pre, &tgt, &mask, n, v, cap);
            assert!((l - l0).abs() <= 1e-5 * l0.abs(), "{what}: loss {l} vs {l0}");
            let mx = max_abs(&d0);
            let err = d.iter().zip(&d0).fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
            let rel = if l0 == 0.0 { 0.0 } else { (l - l0).abs() / l0.abs() };
            println!("{what}: loss rel err {rel:.2e}, dlogits err/max {:.2e}", err / mx.max(f32::MIN_POSITIVE));
            assert!(err <= 1e-5 * mx, "{what}: dlogits err {err} vs max {mx}");
            for (i, &m) in mask.iter().enumerate() {
                if m == 0.0 {
                    assert!(d[i * v..(i + 1) * v].iter().all(|&x| x == 0.0), "{what}: row {i}");
                }
            }
        }
    }

    #[test]
    fn softcap_xent_mt_bit_exact() {
        if !have_avx512() {
            return;
        }
        let mut rng = Rng::new(32);
        for &size in &MT_POOLS {
            let pool = ThreadPool::new(size);
            for (n, v, std, mask) in xent_cases() {
                let pre: Vec<f32> = randn(&mut rng, n * v).iter().map(|x| x * std).collect();
                let tgt: Vec<u16> = (0..n).map(|_| rng.below(v) as u16).collect();
                let (mut da, mut db) = (vec![1.0; n * v], vec![-2.0; n * v]);
                let la = softcap_xent_fwd_bwd(&mut da, &pre, &tgt, &mask, n, v, 30.0);
                let lb = softcap_xent_fwd_bwd_mt(&pool, &mut db, &pre, &tgt, &mask, n, v, 30.0);
                assert_eq!(lb.to_bits(), la.to_bits(), "pool {size} {n}x{v}: loss");
                assert_eq!(bits(&db), bits(&da), "pool {size} {n}x{v}: dlogits");
            }
        }
    }

    /// A NaN logit makes the loss NaN (and its row's gradient NaN), as in the
    /// scalar chain; a diverged run must not report a finite loss.
    #[test]
    fn softcap_xent_propagates_nan() {
        if !have_avx512() {
            return;
        }
        let (n, v, cap) = (3, 37, 30.0);
        let mut rng = Rng::new(33);
        let mut pre = randn(&mut rng, n * v);
        pre[v + 5] = f32::NAN;
        let tgt: Vec<u16> = vec![1, 2, 3];
        let mask = vec![1.0; n];
        let (l0, d0) = softcap_xent_chain(&pre, &tgt, &mask, n, v, cap);
        assert!(l0.is_nan(), "scalar chain: {l0}");
        let pool = ThreadPool::new(3);
        let mut d = vec![0.0; n * v];
        let l = softcap_xent_fwd_bwd(&mut d, &pre, &tgt, &mask, n, v, cap);
        let mut dm = vec![0.0; n * v];
        let lm = softcap_xent_fwd_bwd_mt(&pool, &mut dm, &pre, &tgt, &mask, n, v, cap);
        for (what, l, d) in [("st", l, &d), ("mt", lm, &dm)] {
            assert!(l.is_nan(), "{what}: loss {l} should be NaN");
            for (j, (&g, &r)) in d.iter().zip(&d0).enumerate() {
                assert_eq!(g.is_nan(), r.is_nan(), "{what}: lane {j}: {g} vs scalar {r}");
            }
        }
    }
}
