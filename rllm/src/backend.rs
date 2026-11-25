//! Kernel backend selection. Only matmul and attention are dispatched.
//! Scalar -> `ops::scalar`; Blocked -> `ops::blocked` matmul + scalar attention;
//! Avx512 -> `ops::avx512`. Avx512Mt and Asm use the single-threaded AVX-512
//! kernels until their own kernels land.

use crate::ops::{blocked, scalar};
#[cfg(target_arch = "x86_64")]
use crate::ops::avx512;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Scalar,
    Blocked,
    Avx512,
    Avx512Mt,
    Asm,
}

impl FromStr for Backend {
    type Err = String;
    fn from_str(s: &str) -> Result<Backend, String> {
        match s {
            "scalar" => Ok(Backend::Scalar),
            "blocked" => Ok(Backend::Blocked),
            "avx512" => Ok(Backend::Avx512),
            "avx512mt" => Ok(Backend::Avx512Mt),
            "asm" => Ok(Backend::Asm),
            _ => Err(format!(
                "unknown backend {s:?} (expected scalar|blocked|avx512|avx512mt|asm)"
            )),
        }
    }
}

/// True when the AVX-512 kernels can run on this CPU.
pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Err with a clear message when `be` cannot run on this machine.
pub fn check_supported(be: Backend) -> Result<(), String> {
    match be {
        Backend::Scalar | Backend::Blocked => Ok(()),
        Backend::Avx512 | Backend::Avx512Mt | Backend::Asm if has_avx512() => Ok(()),
        _ => Err(format!(
            "backend {be:?} needs an x86_64 CPU with AVX-512 (avx512f), which was not detected; use --backend scalar or blocked"
        )),
    }
}

/// See `scalar::matmul_fwd`.
pub fn matmul_fwd(
    be: Backend,
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    n: usize,
    k: usize,
    m: usize,
) {
    match be {
        Backend::Scalar => scalar::matmul_fwd(out, x, w, n, k, m),
        Backend::Blocked => blocked::matmul_fwd(out, x, w, n, k, m),
        // TODO(12b): Avx512Mt -> multithreaded kernels.
        // TODO(16): Asm -> hand-written assembly kernels.
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => avx512::matmul_fwd(out, x, w, n, k, m),
        // Unreachable in practice: `check_supported` rejects these off x86_64.
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => scalar::matmul_fwd(out, x, w, n, k, m),
    }
}

/// See `scalar::matmul_bwd` (accumulates).
#[allow(clippy::too_many_arguments)]
pub fn matmul_bwd(
    be: Backend,
    dx: &mut [f32],
    dw: &mut [f32],
    dout: &[f32],
    x: &[f32],
    w: &[f32],
    n: usize,
    k: usize,
    m: usize,
) {
    match be {
        Backend::Scalar => scalar::matmul_bwd(dx, dw, dout, x, w, n, k, m),
        Backend::Blocked => blocked::matmul_bwd(dx, dw, dout, x, w, n, k, m),
        // TODO(12b): Avx512Mt -> multithreaded kernels.
        // TODO(16): Asm -> hand-written assembly kernels.
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => avx512::matmul_bwd(dx, dw, dout, x, w, n, k, m),
        // Unreachable in practice: `check_supported` rejects these off x86_64.
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => scalar::matmul_bwd(dx, dw, dout, x, w, n, k, m),
    }
}

/// See `scalar::attn_fwd`.
#[allow(clippy::too_many_arguments)]
pub fn attn_fwd(
    be: Backend,
    out: &mut [f32],
    att: &mut [f32],
    qkv: &[f32],
    b: usize,
    t: usize,
    c: usize,
    h: usize,
) {
    match be {
        Backend::Scalar | Backend::Blocked => scalar::attn_fwd(out, att, qkv, b, t, c, h),
        // TODO(12b): Avx512Mt -> multithreaded kernels.
        // TODO(16): Asm -> hand-written assembly kernels.
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => avx512::attn_fwd(out, att, qkv, b, t, c, h),
        // Unreachable in practice: `check_supported` rejects these off x86_64.
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => scalar::attn_fwd(out, att, qkv, b, t, c, h),
    }
}

/// See `scalar::attn_bwd` (accumulates).
#[allow(clippy::too_many_arguments)]
pub fn attn_bwd(
    be: Backend,
    dqkv: &mut [f32],
    dout: &[f32],
    qkv: &[f32],
    att: &[f32],
    b: usize,
    t: usize,
    c: usize,
    h: usize,
) {
    match be {
        Backend::Scalar | Backend::Blocked => scalar::attn_bwd(dqkv, dout, qkv, att, b, t, c, h),
        // TODO(12b): Avx512Mt -> multithreaded kernels.
        // TODO(16): Asm -> hand-written assembly kernels.
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => avx512::attn_bwd(dqkv, dout, qkv, att, b, t, c, h),
        // Unreachable in practice: `check_supported` rejects these off x86_64.
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => scalar::attn_bwd(dqkv, dout, qkv, att, b, t, c, h),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_from_str() {
        let all = [
            ("scalar", Backend::Scalar),
            ("blocked", Backend::Blocked),
            ("avx512", Backend::Avx512),
            ("avx512mt", Backend::Avx512Mt),
            ("asm", Backend::Asm),
        ];
        for (s, be) in all {
            assert_eq!(s.parse::<Backend>(), Ok(be));
        }
        assert!("Scalar".parse::<Backend>().is_err());
        assert!("".parse::<Backend>().is_err());
    }

    #[test]
    fn check_supported_scalar_ok() {
        assert_eq!(check_supported(Backend::Scalar), Ok(()));
        assert_eq!(check_supported(Backend::Blocked), Ok(()));
        let has = has_avx512();
        for be in [Backend::Avx512, Backend::Avx512Mt, Backend::Asm] {
            match check_supported(be) {
                Ok(()) => assert!(has),
                Err(e) => {
                    assert!(!has);
                    assert!(e.contains("avx512"), "{e}");
                }
            }
        }
    }

    /// One forward+backward on a tiny model per backend, against Scalar.
    #[test]
    fn model_equivalence_vs_scalar() {
        use crate::data::Batch;
        use crate::model::{backward, forward, Acts, Config, Params};
        use crate::rng::Rng;
        let cfg = Config {
            vocab: 64,
            d: 32,
            layers: 2,
            heads: 2,
            ctx: 16,
            softcap: 30.0,
            mtp: 1,
            bitnet: false,
        };
        let (b, t, extra) = (2, 16, 1);
        let mut rng = Rng::new(11);
        let mut p = Params::init(&cfg, &mut rng);
        p.for_each_tensor_mut(|x| x.iter_mut().for_each(|v| *v += 0.1 * rng.normal()));
        let (mut inputs, mut targets) = (Vec::new(), Vec::new());
        for _ in 0..b {
            let w: Vec<u16> = (0..t + 1 + extra).map(|_| rng.below(cfg.vocab) as u16).collect();
            inputs.extend_from_slice(&w[..t]);
            targets.extend_from_slice(&w[1..]);
        }
        let batch = Batch { b, t, extra, inputs, targets, mask: vec![1.0; b * t] };
        let run = |be: Backend| {
            let mut a = Acts::new(&cfg, b, t);
            let loss = forward(&p, &cfg, be, &mut a, &batch);
            let mut g = p.zeros_like();
            backward(&p, &cfg, be, &mut a, &batch, &mut g);
            (loss, g)
        };
        let (l0, g0) = run(Backend::Scalar);
        let mut bes = vec![Backend::Blocked];
        if has_avx512() {
            bes.push(Backend::Avx512);
        } else {
            eprintln!("note: avx512f absent, skipping Avx512 equivalence");
        }
        for be in bes {
            let (l, g) = run(be);
            assert!(
                (l - l0).abs() <= 1e-4 * l0.abs(),
                "{be:?}: loss {l} vs scalar {l0}"
            );
            for (ti, (a, r)) in g.tensors().iter().zip(g0.tensors()).enumerate() {
                let mx = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                assert!(mx > 0.0, "tensor {ti} reference grad is all zero");
                let err = a.iter().zip(r).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
                assert!(err <= 1e-3 * mx, "{be:?} tensor {ti}: err {err} vs max {mx}");
            }
        }
    }
}
