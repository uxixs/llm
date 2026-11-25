//! GPT model: parameters, preallocated activations, forward/backward, checkpoints.
//!
//! Block (pre-norm): `x += wo·attn(rope(qkv(rmsnorm(x))))`; `x += w_down·relu2(w_up·rmsnorm(x))`.
//! Final: `rmsnorm → x·wteᵀ → softcap`. Tied embedding. Optional MTP heads.

use crate::backend::{self, Backend};
use crate::data::Batch;
use crate::ops::scalar::{self, RopeTable};
use crate::rng::Rng;
use std::io;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub ctx: usize,
    pub softcap: f32,
    pub mtp: usize,
    pub bitnet: bool,
}

impl Config {
    pub fn baseline() -> Config {
        Config {
            vocab: 2048,
            d: 128,
            layers: 4,
            heads: 4,
            ctx: 256,
            softcap: 30.0,
            mtp: 0,
            bitnet: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub attn_norm: Vec<f32>,
    /// 3d × d
    pub wqkv: Mat,
    /// d × d
    pub wo: Mat,
    pub mlp_norm: Vec<f32>,
    /// 4d × d
    pub w_up: Mat,
    /// d × 4d
    pub w_down: Mat,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// V × d
    pub wte: Mat,
    pub layers: Vec<Layer>,
    pub final_norm: Vec<f32>,
    /// d × d each
    pub mtp: Vec<Mat>,
}

impl Mat {
    fn randn(rows: usize, cols: usize, std: f32, rng: &mut Rng) -> Mat {
        Mat {
            rows,
            cols,
            data: (0..rows * cols).map(|_| std * rng.normal()).collect(),
        }
    }
}

/// Number of scalars `cfg` implies (checked: a corrupt header must not overflow).
fn param_count(cfg: &Config) -> Option<usize> {
    let d = cfg.d;
    let d2 = d.checked_mul(d)?;
    let layer = d2.checked_mul(12)?.checked_add(2 * d)?;
    cfg.vocab
        .checked_mul(d)?
        .checked_add(layer.checked_mul(cfg.layers)?)?
        .checked_add(d)?
        .checked_add(d2.checked_mul(cfg.mtp)?)
}

const MAGIC: &[u8; 4] = b"RLLM";
const VERSION: u32 = 1;

impl Params {
    /// Norms = 1; matrices normal(0, 0.02) except `wo`/`w_down` at 0.02/√(2·layers).
    pub fn init(cfg: &Config, rng: &mut Rng) -> Params {
        let d = cfg.d;
        assert!(
            d > 0 && cfg.heads > 0 && d.is_multiple_of(cfg.heads),
            "d must be a multiple of heads"
        );
        assert!(
            (d / cfg.heads).is_multiple_of(2),
            "head_dim must be even for RoPE"
        );
        assert!(cfg.vocab <= 1 << 16, "token ids are u16");
        let (s, sr) = (0.02, 0.02 / ((2 * cfg.layers) as f32).sqrt());
        let wte = Mat::randn(cfg.vocab, d, s, rng);
        let layers = (0..cfg.layers)
            .map(|_| Layer {
                attn_norm: vec![1.0; d],
                wqkv: Mat::randn(3 * d, d, s, rng),
                wo: Mat::randn(d, d, sr, rng),
                mlp_norm: vec![1.0; d],
                w_up: Mat::randn(4 * d, d, s, rng),
                w_down: Mat::randn(d, 4 * d, sr, rng),
            })
            .collect();
        let mtp = (0..cfg.mtp).map(|_| Mat::randn(d, d, s, rng)).collect();
        Params {
            wte,
            layers,
            final_norm: vec![1.0; d],
            mtp,
        }
    }

    /// Same shapes, all zeros (for gradients).
    pub fn zeros_like(&self) -> Params {
        let mut g = self.clone();
        g.zero();
        g
    }

    pub fn zero(&mut self) {
        self.for_each_tensor_mut(|t| t.fill(0.0));
    }

    pub fn num_params(&self) -> usize {
        self.tensors().iter().map(|t| t.len()).sum()
    }

    /// All tensors in the canonical order shared by save/load and the optimizers:
    /// `wte`, then per layer `attn_norm, wqkv, wo, mlp_norm, w_up, w_down`, then
    /// `final_norm`, then `mtp[0..]`. Matrices are row-major `[rows, cols]`.
    pub fn tensors(&self) -> Vec<&[f32]> {
        let mut v: Vec<&[f32]> = vec![&self.wte.data];
        for l in &self.layers {
            v.extend([
                &l.attn_norm[..],
                &l.wqkv.data,
                &l.wo.data,
                &l.mlp_norm,
                &l.w_up.data,
                &l.w_down.data,
            ]);
        }
        v.push(&self.final_norm);
        v.extend(self.mtp.iter().map(|m| &m.data[..]));
        v
    }

    /// Same order as `tensors`.
    pub fn tensors_mut(&mut self) -> Vec<&mut [f32]> {
        let Params {
            wte,
            layers,
            final_norm,
            mtp,
        } = self;
        let mut v: Vec<&mut [f32]> = vec![&mut wte.data];
        for l in layers.iter_mut() {
            let Layer {
                attn_norm,
                wqkv,
                wo,
                mlp_norm,
                w_up,
                w_down,
            } = l;
            v.extend([
                &mut attn_norm[..],
                &mut wqkv.data,
                &mut wo.data,
                &mut mlp_norm[..],
                &mut w_up.data,
                &mut w_down.data,
            ]);
        }
        v.push(final_norm);
        v.extend(mtp.iter_mut().map(|m| &mut m.data[..]));
        v
    }

    /// Visits tensors in the `tensors` order.
    pub fn for_each_tensor_mut(&mut self, mut f: impl FnMut(&mut [f32])) {
        for t in self.tensors_mut() {
            f(t);
        }
    }

    /// Format: `b"RLLM"`, u32 version 1, Config as u32s (vocab, d, layers, heads, ctx,
    /// softcap f32 bits, mtp, bitnet 0/1), then every tensor as f32 LE in `tensors` order.
    pub fn save(&self, cfg: &Config, path: &Path) -> io::Result<()> {
        assert_eq!(
            Some(self.num_params()),
            param_count(cfg),
            "params do not match config"
        );
        let mut buf = Vec::with_capacity(40 + 4 * self.num_params());
        buf.extend_from_slice(MAGIC);
        let header = [
            VERSION,
            cfg.vocab as u32,
            cfg.d as u32,
            cfg.layers as u32,
            cfg.heads as u32,
            cfg.ctx as u32,
            cfg.softcap.to_bits(),
            cfg.mtp as u32,
            cfg.bitnet as u32,
        ];
        for w in header {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for t in self.tensors() {
            for v in t {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(path, buf)
    }

    pub fn load(path: &Path) -> io::Result<(Config, Params)> {
        let bytes = std::fs::read(path)?;
        let bad = |msg: &str| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {msg}", path.display()),
            )
        };
        if bytes.len() < 40 || &bytes[..4] != MAGIC {
            return Err(bad("not an RLLM checkpoint"));
        }
        let word = |i: usize| u32::from_le_bytes(bytes[4 + 4 * i..8 + 4 * i].try_into().unwrap());
        if word(0) != VERSION {
            return Err(bad("unsupported checkpoint version"));
        }
        let bitnet = match word(8) {
            0 => false,
            1 => true,
            _ => return Err(bad("bitnet flag must be 0 or 1")),
        };
        let cfg = Config {
            vocab: word(1) as usize,
            d: word(2) as usize,
            layers: word(3) as usize,
            heads: word(4) as usize,
            ctx: word(5) as usize,
            softcap: f32::from_bits(word(6)),
            mtp: word(7) as usize,
            bitnet,
        };
        let valid = cfg.d > 0
            && cfg.heads > 0
            && cfg.d.is_multiple_of(cfg.heads)
            && (cfg.d / cfg.heads).is_multiple_of(2);
        let body = &bytes[40..];
        if !valid
            || cfg.vocab > 1 << 16
            || param_count(&cfg).and_then(|n| n.checked_mul(4)) != Some(body.len())
        {
            return Err(bad("header and tensor data disagree"));
        }
        let mut p = Params::init(&cfg, &mut Rng::new(0));
        let mut vals = body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c));
        p.for_each_tensor_mut(|t| t.iter_mut().for_each(|v| *v = vals.next().unwrap()));
        Ok((cfg, p))
    }
}

/// Per-layer saved activations (all `[n, ·]` with n = b·t, row-major).
struct LayerActs {
    ln1: Vec<f32>,
    rstd1: Vec<f32>,
    /// `[n, 3d]`, q and k already rotated (what attention consumed).
    qkv: Vec<f32>,
    /// `[b, h, t, t]`
    att: Vec<f32>,
    attn_out: Vec<f32>,
    x_mid: Vec<f32>,
    ln2: Vec<f32>,
    rstd2: Vec<f32>,
    /// `[n, 4d]` pre-activation and relu² output.
    up: Vec<f32>,
    act: Vec<f32>,
}

/// Activations and scratch preallocated for a fixed (b, t). Backward allocates nothing.
pub struct Acts {
    pub b: usize,
    pub t: usize,
    /// MTP heads run by the last `forward`.
    n_mtp: usize,
    /// True after `forward`; `forward_logits` clears it (it may use a smaller shape).
    ready: bool,
    /// `layers + 1` residual streams; `xs[l]` is the input of layer l, `xs[layers]` the output.
    xs: Vec<Vec<f32>>,
    layers: Vec<LayerActs>,
    lnf: Vec<f32>,
    rstdf: Vec<f32>,
    /// `1 + mtp` heads `[n, V]`: `logits` holds softcapped logits after `forward_logits`
    /// (main head) and is the Scalar/Blocked scratch of `backend::softcap_xent` in
    /// `forward`; `dlogits` is d loss / d PRE-cap logits from the fused head op.
    logits: Vec<Vec<f32>>,
    dlogits: Vec<Vec<f32>>,
    /// MTP head hidden states `[n, d]`.
    hk: Vec<Vec<f32>>,
    tgt: Vec<u16>,
    ones: Vec<f32>,
    // Scratch.
    s_v: Vec<f32>,
    s_d: Vec<f32>,
    rope_buf: Vec<f32>,
    /// RoPE angles for positions `0..ctx`, head dim `d / heads`.
    rope: RopeTable,
    dx: Vec<f32>,
    dln: Vec<f32>,
    dlnf: Vec<f32>,
    dhk: Vec<f32>,
    dqkv: Vec<f32>,
    dup: Vec<f32>,
    dact: Vec<f32>,
}

impl Acts {
    pub fn new(cfg: &Config, b: usize, t: usize) -> Acts {
        assert!(t <= cfg.ctx, "t = {t} exceeds ctx = {}", cfg.ctx);
        let (n, d, v) = (b * t, cfg.d, cfg.vocab);
        let z = |len: usize| vec![0.0f32; len];
        let layers = (0..cfg.layers)
            .map(|_| LayerActs {
                ln1: z(n * d),
                rstd1: z(n),
                qkv: z(n * 3 * d),
                att: z(b * cfg.heads * t * t),
                attn_out: z(n * d),
                x_mid: z(n * d),
                ln2: z(n * d),
                rstd2: z(n),
                up: z(n * 4 * d),
                act: z(n * 4 * d),
            })
            .collect();
        Acts {
            b,
            t,
            n_mtp: 0,
            ready: false,
            xs: (0..=cfg.layers).map(|_| z(n * d)).collect(),
            layers,
            lnf: z(n * d),
            rstdf: z(n),
            logits: (0..=cfg.mtp).map(|_| z(n * v)).collect(),
            dlogits: (0..=cfg.mtp).map(|_| z(n * v)).collect(),
            hk: (0..cfg.mtp).map(|_| z(n * d)).collect(),
            tgt: vec![0; n],
            ones: vec![1.0; n],
            s_v: z(n * v),
            s_d: z(n * d),
            rope_buf: z(n * d),
            rope: RopeTable::new(cfg.ctx, d / cfg.heads),
            dx: z(n * d),
            dln: z(n * d),
            dlnf: z(n * d),
            dhk: z(n * d),
            dqkv: z(n * 3 * d),
            dup: z(n * 4 * d),
            dact: z(n * 4 * d),
        }
    }

    /// Main-head softcapped logits `[b·t, V]` after a Scalar/Blocked `forward` (where they
    /// are the fused op's scratch). Test-only: on AVX backends `forward` does not fill
    /// them; use the slice `forward_logits` returns instead.
    #[cfg(test)]
    pub(crate) fn logits(&self) -> &[f32] {
        &self.logits[0]
    }
}

/// RoPE (or its inverse) on the q and k thirds of a `[b, t, 3c]` buffer, via a contiguous scratch.
#[allow(clippy::too_many_arguments)]
fn rope_qk(
    qkv: &mut [f32],
    scratch: &mut [f32],
    tab: &RopeTable,
    b: usize,
    t: usize,
    c: usize,
    h: usize,
    inverse: bool,
) {
    let n = b * t;
    let s = &mut scratch[..n * c];
    for part in 0..2 {
        for i in 0..n {
            s[i * c..(i + 1) * c].copy_from_slice(&qkv[i * 3 * c + part * c..][..c]);
        }
        if inverse {
            scalar::rope_bwd_table(s, b, t, h, c / h, tab);
        } else {
            scalar::rope_fwd_table(s, b, t, h, c / h, tab);
        }
        for i in 0..n {
            qkv[i * 3 * c + part * c..][..c].copy_from_slice(&s[i * c..(i + 1) * c]);
        }
    }
}

/// Embedding → blocks → final norm → main-head PRE-cap logits in `a.s_v`, for `b ≤ a.b`, `t ≤ a.t`.
fn trunk(p: &Params, cfg: &Config, be: Backend, a: &mut Acts, inputs: &[u16], b: usize, t: usize) {
    assert!(
        b <= a.b && t <= a.t,
        "shape ({b},{t}) exceeds Acts ({},{})",
        a.b,
        a.t
    );
    assert_eq!(inputs.len(), b * t);
    assert_eq!(p.layers.len(), cfg.layers);
    assert_eq!(p.mtp.len(), cfg.mtp);
    let (n, d, h, v) = (b * t, cfg.d, cfg.heads, cfg.vocab);
    let (nd, n3, n4) = (n * d, n * 3 * d, n * 4 * d);
    for (i, &tok) in inputs.iter().enumerate() {
        a.xs[0][i * d..(i + 1) * d].copy_from_slice(&p.wte.data[tok as usize * d..][..d]);
    }
    for (l, (lp, la)) in p.layers.iter().zip(&mut a.layers).enumerate() {
        let (lo, hi) = a.xs.split_at_mut(l + 1);
        let (x_in, x_out) = (&lo[l][..nd], &mut hi[0][..nd]);
        scalar::rmsnorm_fwd(
            &mut la.ln1[..nd],
            &mut la.rstd1[..n],
            x_in,
            &lp.attn_norm,
            n,
            d,
        );
        backend::matmul_fwd(
            be,
            &mut la.qkv[..n3],
            &la.ln1[..nd],
            &lp.wqkv.data,
            n,
            d,
            3 * d,
        );
        rope_qk(&mut la.qkv[..n3], &mut a.rope_buf, &a.rope, b, t, d, h, false);
        backend::attn_fwd(
            be,
            &mut la.attn_out[..nd],
            &mut la.att[..b * h * t * t],
            &la.qkv[..n3],
            b,
            t,
            d,
            h,
        );
        backend::matmul_fwd(
            be,
            &mut a.s_d[..nd],
            &la.attn_out[..nd],
            &lp.wo.data,
            n,
            d,
            d,
        );
        for ((m, x), y) in la.x_mid[..nd].iter_mut().zip(x_in).zip(&a.s_d[..nd]) {
            *m = x + y;
        }
        scalar::rmsnorm_fwd(
            &mut la.ln2[..nd],
            &mut la.rstd2[..n],
            &la.x_mid[..nd],
            &lp.mlp_norm,
            n,
            d,
        );
        backend::matmul_fwd(
            be,
            &mut la.up[..n4],
            &la.ln2[..nd],
            &lp.w_up.data,
            n,
            d,
            4 * d,
        );
        scalar::relu2_fwd(&mut la.act[..n4], &la.up[..n4]);
        backend::matmul_fwd(
            be,
            &mut a.s_d[..nd],
            &la.act[..n4],
            &lp.w_down.data,
            n,
            4 * d,
            d,
        );
        for ((o, m), y) in x_out.iter_mut().zip(&la.x_mid[..nd]).zip(&a.s_d[..nd]) {
            *o = m + y;
        }
    }
    let x_last = &a.xs[cfg.layers][..nd];
    scalar::rmsnorm_fwd(
        &mut a.lnf[..nd],
        &mut a.rstdf[..n],
        x_last,
        &p.final_norm,
        n,
        d,
    );
    backend::matmul_fwd(be, &mut a.s_v[..n * v], &a.lnf[..nd], &p.wte.data, n, d, v);
}

/// Row r of `targets` (stride `t + extra`), columns `[k .. k + t]`, gathered into `out` `[b·t]`.
fn gather_targets(out: &mut [u16], batch: &Batch, k: usize) {
    let (t, te) = (batch.t, batch.t + batch.extra);
    for r in 0..batch.b {
        out[r * t..(r + 1) * t].copy_from_slice(&batch.targets[r * te + k..][..t]);
    }
}

/// Loss = main masked xent + (when MTP heads run) `0.5 · mean_k L_k` for
/// k = 1..=min(cfg.mtp, batch.extra). Head k: `final_norm(h)·M_kᵀ → tied logits → softcap`
/// against targets `[k .. k+t]`, all-ones mask. Panics if the batch shape differs from `a`.
pub fn forward(p: &Params, cfg: &Config, be: Backend, a: &mut Acts, batch: &Batch) -> f32 {
    assert_eq!(
        (batch.b, batch.t),
        (a.b, a.t),
        "batch shape differs from Acts"
    );
    let (n, d, v) = (batch.b * batch.t, cfg.d, cfg.vocab);
    assert_eq!(batch.targets.len(), batch.b * (batch.t + batch.extra));
    assert_eq!(batch.mask.len(), n);
    trunk(p, cfg, be, a, &batch.inputs, batch.b, batch.t);
    gather_targets(&mut a.tgt, batch, 0);
    let (nv, cap) = (n * v, cfg.softcap);
    let mut loss = backend::softcap_xent(
        be,
        &mut a.dlogits[0],
        &a.s_v[..nv],
        &mut a.logits[0],
        &a.tgt,
        &batch.mask,
        n,
        v,
        cap,
    );
    a.n_mtp = cfg.mtp.min(batch.extra);
    if a.n_mtp > 0 {
        let mut aux = 0.0;
        for k in 1..=a.n_mtp {
            backend::matmul_fwd(be, &mut a.hk[k - 1], &a.lnf, &p.mtp[k - 1].data, n, d, d);
            backend::matmul_fwd(be, &mut a.s_v, &a.hk[k - 1], &p.wte.data, n, d, v);
            gather_targets(&mut a.tgt, batch, k);
            aux += backend::softcap_xent(
                be,
                &mut a.dlogits[k],
                &a.s_v,
                &mut a.logits[k],
                &a.tgt,
                &a.ones,
                n,
                v,
                cap,
            );
        }
        loss += 0.5 * aux / a.n_mtp as f32;
    }
    a.ready = true;
    loss
}

/// Main-head softcapped logits `[b·t, V]` for `inputs` (`b ≤ a.b`, `t ≤ a.t`; no loss).
/// Invalidates `a` for `backward` until the next `forward`.
pub fn forward_logits<'a>(
    p: &Params,
    cfg: &Config,
    be: Backend,
    a: &'a mut Acts,
    inputs: &[u16],
    b: usize,
    t: usize,
) -> &'a [f32] {
    a.ready = false;
    trunk(p, cfg, be, a, inputs, b, t);
    let nv = b * t * cfg.vocab;
    scalar::softcap_fwd(&mut a.logits[0][..nv], &a.s_v[..nv], cfg.softcap);
    &a.logits[0][..nv]
}

/// Gradient of the last `forward`'s loss. ACCUMULATES into `g` (call `g.zero()` first).
/// Allocates nothing beyond what `a` holds.
pub fn backward(
    p: &Params,
    cfg: &Config,
    be: Backend,
    a: &mut Acts,
    batch: &Batch,
    g: &mut Params,
) {
    assert!(a.ready, "backward needs a preceding forward on this Acts");
    assert_eq!(
        (batch.b, batch.t),
        (a.b, a.t),
        "batch shape differs from Acts"
    );
    let Acts {
        b,
        t,
        n_mtp,
        xs,
        layers,
        lnf,
        rstdf,
        dlogits,
        hk,
        s_v,
        s_d,
        rope_buf,
        rope,
        dx,
        dln,
        dlnf,
        dhk,
        dqkv,
        dup,
        dact,
        ..
    } = a;
    let (b, t, n_mtp) = (*b, *t, *n_mtp);
    let (n, d, h, v) = (b * t, cfg.d, cfg.heads, cfg.vocab);

    // Heads → d(final-norm output). Tied embedding: logits matmul grads go into g.wte.
    // dlogits already hold d loss / d pre-cap logits (softcap backward fused in forward).
    dlnf.fill(0.0);
    backend::matmul_bwd(be, dlnf, &mut g.wte.data, &dlogits[0], lnf, &p.wte.data, n, d, v);
    for k in 1..=n_mtp {
        let w = 0.5 / n_mtp as f32;
        for (s, &dv) in s_v.iter_mut().zip(&dlogits[k]) {
            *s = dv * w;
        }
        dhk.fill(0.0);
        backend::matmul_bwd(
            be,
            dhk,
            &mut g.wte.data,
            s_v,
            &hk[k - 1],
            &p.wte.data,
            n,
            d,
            v,
        );
        backend::matmul_bwd(
            be,
            dlnf,
            &mut g.mtp[k - 1].data,
            dhk,
            lnf,
            &p.mtp[k - 1].data,
            n,
            d,
            d,
        );
    }
    dx.fill(0.0);
    scalar::rmsnorm_bwd(
        dx,
        &mut g.final_norm,
        dlnf,
        &xs[cfg.layers],
        &p.final_norm,
        rstdf,
        n,
        d,
    );

    // Blocks in reverse; dx carries the residual gradient (rmsnorm_bwd accumulates into it).
    for l in (0..cfg.layers).rev() {
        let (lp, la, lg) = (&p.layers[l], &layers[l], &mut g.layers[l]);
        dact.fill(0.0);
        backend::matmul_bwd(
            be,
            dact,
            &mut lg.w_down.data,
            dx,
            &la.act,
            &lp.w_down.data,
            n,
            4 * d,
            d,
        );
        dup.fill(0.0);
        scalar::relu2_bwd(dup, dact, &la.up);
        dln.fill(0.0);
        backend::matmul_bwd(
            be,
            dln,
            &mut lg.w_up.data,
            dup,
            &la.ln2,
            &lp.w_up.data,
            n,
            d,
            4 * d,
        );
        scalar::rmsnorm_bwd(
            dx,
            &mut lg.mlp_norm,
            dln,
            &la.x_mid,
            &lp.mlp_norm,
            &la.rstd2,
            n,
            d,
        );

        s_d.fill(0.0);
        backend::matmul_bwd(
            be,
            s_d,
            &mut lg.wo.data,
            dx,
            &la.attn_out,
            &lp.wo.data,
            n,
            d,
            d,
        );
        dqkv.fill(0.0);
        backend::attn_bwd(be, dqkv, s_d, &la.qkv, &la.att, b, t, d, h);
        rope_qk(dqkv, rope_buf, rope, b, t, d, h, true);
        dln.fill(0.0);
        backend::matmul_bwd(
            be,
            dln,
            &mut lg.wqkv.data,
            dqkv,
            &la.ln1,
            &lp.wqkv.data,
            n,
            d,
            3 * d,
        );
        scalar::rmsnorm_bwd(
            dx,
            &mut lg.attn_norm,
            dln,
            &xs[l],
            &lp.attn_norm,
            &la.rstd1,
            n,
            d,
        );
    }

    // Embedding lookup.
    for (i, &tok) in batch.inputs.iter().enumerate() {
        let row = &mut g.wte.data[tok as usize * d..][..d];
        for (gw, dv) in row.iter_mut().zip(&dx[i * d..(i + 1) * d]) {
            *gw += dv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Config {
        // softcap 5 (not 30) so the cap's curvature matters at these logit scales.
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

    /// Random windows of `t + 1 + extra` tokens per row, laid out like `data::Batch`.
    fn rand_batch(rng: &mut Rng, v: usize, b: usize, t: usize, extra: usize) -> Batch {
        let (mut inputs, mut targets) = (Vec::new(), Vec::new());
        for _ in 0..b {
            let w: Vec<u16> = (0..t + 1 + extra).map(|_| rng.below(v) as u16).collect();
            inputs.extend_from_slice(&w[..t]);
            targets.extend_from_slice(&w[1..]);
        }
        Batch {
            b,
            t,
            extra,
            inputs,
            targets,
            mask: vec![1.0; b * t],
        }
    }

    fn sgd(p: &mut Params, g: &Params, lr: f32) {
        for (pt, gt) in p.tensors_mut().into_iter().zip(g.tensors()) {
            for (w, d) in pt.iter_mut().zip(gt) {
                *w -= lr * d;
            }
        }
    }

    #[test]
    fn param_count_baseline() {
        let p = Params::init(&Config::baseline(), &mut Rng::new(0));
        assert_eq!(p.num_params(), 1_049_728);
        assert_eq!(
            1_049_728,
            2048 * 128 + 4 * (128 + 3 * 128 * 128 + 128 * 128 + 128 + 2 * 4 * 128 * 128) + 128
        );
        let mut cfg = Config::baseline();
        cfg.mtp = 2;
        assert_eq!(
            Params::init(&cfg, &mut Rng::new(0)).num_params(),
            1_049_728 + 2 * 128 * 128
        );
    }

    #[test]
    fn init_scales() {
        let cfg = Config::baseline();
        let p = Params::init(&cfg, &mut Rng::new(1));
        let std = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        assert!((std(&p.wte.data) - 0.02).abs() < 0.002);
        assert!((std(&p.layers[0].w_up.data) - 0.02).abs() < 0.002);
        let small = 0.02 / (2.0 * cfg.layers as f32).sqrt();
        assert!((std(&p.layers[1].wo.data) - small).abs() < 0.001);
        assert!((std(&p.layers[2].w_down.data) - small).abs() < 0.001);
        assert!(p
            .final_norm
            .iter()
            .chain(&p.layers[3].attn_norm)
            .all(|&g| g == 1.0));
    }

    #[test]
    fn model_gradcheck_tiny() {
        let cfg = tiny_cfg();
        let mut rng = Rng::new(3);
        let mut p = Params::init(&cfg, &mut rng);
        // Move away from the init so every term carries a sizeable gradient.
        p.for_each_tensor_mut(|t| t.iter_mut().for_each(|v| *v += 0.3 * rng.normal()));
        let (b, t) = (2, 5);
        let mut batch = rand_batch(&mut rng, cfg.vocab, b, t, 1);
        batch.mask[3] = 0.0;
        let mut a = Acts::new(&cfg, b, t);
        forward(&p, &cfg, Backend::Scalar, &mut a, &batch);
        let mut g = p.zeros_like();
        backward(&p, &cfg, Backend::Scalar, &mut a, &batch, &mut g);
        let grads: Vec<Vec<f32>> = g.tensors().iter().map(|t| t.to_vec()).collect();
        assert_eq!(grads.len(), 1 + 6 * cfg.layers + 1 + cfg.mtp);

        // 3 coords per tensor; wte additionally gets coords in rows that are inputs
        // (embedding-lookup term) and rows that are never inputs (unembedding only).
        let mut coords: Vec<(usize, usize)> = Vec::new();
        for (ti, gt) in grads.iter().enumerate() {
            for _ in 0..3 {
                coords.push((ti, rng.below(gt.len())));
            }
        }
        for &tok in &batch.inputs[..3] {
            coords.push((0, tok as usize * cfg.d + rng.below(cfg.d)));
        }
        let unused = (0..cfg.vocab as u16)
            .find(|v| !batch.inputs.contains(v))
            .unwrap();
        coords.push((0, unused as usize * cfg.d + rng.below(cfg.d)));
        assert!(coords.len() >= 30);

        let eps = 1e-2f32;
        let mut significant = 0;
        for &(ti, i) in &coords {
            let orig = p.tensors()[ti][i];
            p.tensors_mut()[ti][i] = orig + eps;
            let hi = forward(&p, &cfg, Backend::Scalar, &mut a, &batch);
            p.tensors_mut()[ti][i] = orig - eps;
            let lo = forward(&p, &cfg, Backend::Scalar, &mut a, &batch);
            p.tensors_mut()[ti][i] = orig;
            let n = (hi - lo) / (2.0 * eps);
            let an = grads[ti][i];
            let tol = 1e-3 + 2e-2 * an.abs().max(n.abs());
            assert!(
                (an - n).abs() <= tol,
                "tensor {ti} idx {i}: analytic {an}, numeric {n}"
            );
            if an.abs() > 1e-2 {
                significant += 1;
            }
        }
        assert!(
            significant >= coords.len() * 2 / 3,
            "only {significant}/{} gradients are non-trivial",
            coords.len()
        );
    }

    #[test]
    fn mtp_heads_follow_batch_extra() {
        // extra=0 → only the main loss; the MTP matrix gets no gradient.
        let cfg = tiny_cfg();
        let mut rng = Rng::new(5);
        let p = Params::init(&cfg, &mut rng);
        let batch = rand_batch(&mut rng, cfg.vocab, 2, 5, 0);
        let mut a = Acts::new(&cfg, 2, 5);
        let loss = forward(&p, &cfg, Backend::Scalar, &mut a, &batch);
        let mut g = p.zeros_like();
        backward(&p, &cfg, Backend::Scalar, &mut a, &batch, &mut g);
        assert!(g.mtp[0].data.iter().all(|&v| v == 0.0));
        let mut cfg0 = cfg;
        cfg0.mtp = 0;
        let mut p0 = p.clone();
        p0.mtp.clear();
        let mut a0 = Acts::new(&cfg0, 2, 5);
        assert_eq!(loss, forward(&p0, &cfg0, Backend::Scalar, &mut a0, &batch));
    }

    #[test]
    fn forward_logits_matches_forward() {
        let cfg = tiny_cfg();
        let mut rng = Rng::new(9);
        let p = Params::init(&cfg, &mut rng);
        let batch = rand_batch(&mut rng, cfg.vocab, 2, 5, 0);
        let mut a = Acts::new(&cfg, 2, 5);
        forward(&p, &cfg, Backend::Scalar, &mut a, &batch);
        let full = a.logits().to_vec();
        // Same rows, and a shorter prefix of row 0 reuses the same Acts.
        assert_eq!(
            forward_logits(&p, &cfg, Backend::Scalar, &mut a, &batch.inputs, 2, 5),
            &full[..]
        );
        let short = forward_logits(&p, &cfg, Backend::Scalar, &mut a, &batch.inputs[..3], 1, 3);
        for (x, y) in short.iter().zip(&full[..3 * cfg.vocab]) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    #[should_panic]
    fn forward_panics_on_shape_mismatch() {
        let cfg = tiny_cfg();
        let mut rng = Rng::new(1);
        let p = Params::init(&cfg, &mut rng);
        let batch = rand_batch(&mut rng, cfg.vocab, 2, 4, 0);
        forward(
            &p,
            &cfg,
            Backend::Scalar,
            &mut Acts::new(&cfg, 2, 5),
            &batch,
        );
    }

    #[test]
    fn initial_loss_near_ln_v() {
        let cfg = Config::baseline();
        let mut rng = Rng::new(11);
        let p = Params::init(&cfg, &mut rng);
        let batch = rand_batch(&mut rng, cfg.vocab, 1, 64, 0);
        let loss = forward(
            &p,
            &cfg,
            Backend::Scalar,
            &mut Acts::new(&cfg, 1, 64),
            &batch,
        );
        assert!((loss - (cfg.vocab as f32).ln()).abs() < 0.5, "loss {loss}");
    }

    #[test]
    fn save_load_roundtrip() {
        let cfg = Config {
            bitnet: true,
            mtp: 2,
            softcap: 12.5,
            ..tiny_cfg()
        };
        let p = Params::init(&cfg, &mut Rng::new(4));
        let path =
            std::env::temp_dir().join(format!("rllm_model_roundtrip_{}.bin", std::process::id()));
        p.save(&cfg, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..4], b"RLLM");
        assert_eq!(bytes.len(), 4 + 4 + 8 * 4 + 4 * p.num_params());
        let (cfg2, p2) = Params::load(&path).unwrap();
        assert_eq!(cfg2, cfg);
        assert_eq!(p2, p);
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(
            Params::load(&path).is_err(),
            "truncated file must be rejected"
        );
        let mut bad = bytes.clone();
        bad[0] = b'X';
        std::fs::write(&path, &bad).unwrap();
        assert!(Params::load(&path).is_err(), "bad magic must be rejected");
        std::fs::remove_file(&path).unwrap();
    }

    /// Ruling D: threshold 0.1 is binding. Values used: 300 steps, SGD lr 0.2 (lr 0.1 plateaued
    /// at ~0.20 after 300 steps; lr 0.2 reaches ~0.05).
    #[test]
    #[ignore = "slow; run with --release -- --include-ignored"]
    fn overfit_tiny_batch() {
        let cfg = Config {
            d: 32,
            ..Config::baseline()
        };
        let mut rng = Rng::new(7);
        let mut p = Params::init(&cfg, &mut rng);
        let batch = rand_batch(&mut rng, cfg.vocab, 2, 32, 0);
        let mut a = Acts::new(&cfg, 2, 32);
        let mut g = p.zeros_like();
        let mut loss = f32::INFINITY;
        for _ in 0..300 {
            loss = forward(&p, &cfg, Backend::Scalar, &mut a, &batch);
            g.zero();
            backward(&p, &cfg, Backend::Scalar, &mut a, &batch, &mut g);
            sgd(&mut p, &g, 0.2);
        }
        assert!(loss < 0.1, "loss after training {loss}");
    }

    #[test]
    #[should_panic(expected = "exceeds ctx")]
    fn acts_new_rejects_t_over_ctx() {
        let cfg = tiny_cfg();
        let _ = Acts::new(&cfg, 1, cfg.ctx + 1);
    }
}
