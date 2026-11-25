//! Kernel backend selection. Only matmul and attention are dispatched; every
//! variant routes to `ops::scalar` until its own kernels land (Task 10+).

use crate::ops::scalar;
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
        Backend::Scalar | Backend::Blocked | Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => {
            scalar::matmul_fwd(out, x, w, n, k, m)
        }
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
        Backend::Scalar | Backend::Blocked | Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => {
            scalar::matmul_bwd(dx, dw, dout, x, w, n, k, m)
        }
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
        Backend::Scalar | Backend::Blocked | Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => {
            scalar::attn_fwd(out, att, qkv, b, t, c, h)
        }
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
        Backend::Scalar | Backend::Blocked | Backend::Avx512 | Backend::Avx512Mt | Backend::Asm => {
            scalar::attn_bwd(dqkv, dout, qkv, att, b, t, c, h)
        }
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
}
