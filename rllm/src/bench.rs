//! Benchmark harness (library side, so it is testable). The CLI in `bin/bench.rs`
//! prints [`report`] and appends it to `bench/results.md`.
//!
//! Measurement rules: a timing is the median of `reps` runs after `warmup` runs
//! (default 3 and 5). Numbers are only meaningful on a quiet machine.

use crate::backend::{self, has_avx512, Backend};
use crate::data::Batch;
use crate::model::{backward, forward, Acts, Config, Params};
use crate::ops::scalar;
use crate::optim::Optimizer;
use crate::rng::Rng;
use std::hint::black_box;
use std::time::Instant;

pub const WARMUP: usize = 3;
pub const REPS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Fwd,
    Bwd,
}

/// Median of `v` (mean of the middle two for an even count). `v` must be non-empty.
pub fn median(mut v: Vec<f64>) -> f64 {
    assert!(!v.is_empty());
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Runs `f` `warmup` times untimed, then `reps` times timed; returns the median seconds.
pub fn measure<F: FnMut()>(warmup: usize, reps: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let times = (0..reps.max(1))
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64()
        })
        .collect();
    median(times)
}

/// FLOPs of one matmul `[n,k] x [m,k]^T`: forward 2nkm, backward (dx and dw) 4nkm.
pub fn matmul_flops(which: Which, n: usize, k: usize, m: usize) -> f64 {
    let base = 2.0 * n as f64 * k as f64 * m as f64;
    match which {
        Which::Fwd => base,
        Which::Bwd => 2.0 * base,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn fma_chunk(iters: u64, seed: f32) -> f32 {
    use std::arch::x86_64::*;
    // c = c*a + b converges (|a| < 1), so values stay normal. 8 independent chains
    // cover the FMA latency on a single 512-bit FMA port.
    let (a, b) = (_mm512_set1_ps(0.999_999), _mm512_set1_ps(1e-3));
    let mut c = [_mm512_set1_ps(seed); 8];
    for _ in 0..iters {
        for acc in c.iter_mut() {
            *acc = _mm512_fmadd_ps(*acc, a, b);
        }
    }
    let mut s = c[0];
    for acc in &c[1..] {
        s = _mm512_add_ps(s, *acc);
    }
    _mm512_reduce_add_ps(s)
}

/// Measured AVX-512 FMA peak in GFLOPS summed over `threads` threads, each running
/// 8 independent zmm accumulator chains for `secs` seconds. 0 without avx512f.
pub fn fma_peak_gflops(threads: usize, secs: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if !has_avx512() {
            return 0.0;
        }
        const CHUNK: u64 = 1 << 14;
        const FLOPS_PER_ITER: f64 = 8.0 * 16.0 * 2.0;
        let rates: Vec<f64> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads.max(1))
                .map(|i| {
                    s.spawn(move || {
                        let t0 = Instant::now();
                        let mut iters = 0u64;
                        let mut sink = 0.0f32;
                        while t0.elapsed().as_secs_f64() < secs {
                            // SAFETY: avx512f checked above.
                            sink += unsafe { fma_chunk(CHUNK, 0.5 + i as f32) };
                            iters += CHUNK;
                        }
                        black_box(sink);
                        iters as f64 * FLOPS_PER_ITER / t0.elapsed().as_secs_f64() / 1e9
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        rates.iter().sum()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (threads, secs);
        0.0
    }
}

fn randn(rng: &mut Rng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.normal()).collect()
}

/// GFLOPS of one matmul (`n,k,m` as in `backend::matmul_fwd`) with explicit warmup/reps.
pub fn matmul_gflops_with(
    be: Backend,
    n: usize,
    k: usize,
    m: usize,
    which: Which,
    warmup: usize,
    reps: usize,
) -> f64 {
    let mut rng = Rng::new(7);
    let (x, w) = (randn(&mut rng, n * k), randn(&mut rng, m * k));
    let secs = match which {
        Which::Fwd => {
            let mut out = vec![0.0f32; n * m];
            measure(warmup, reps, || {
                backend::matmul_fwd(be, &mut out, &x, &w, n, k, m);
                black_box(&out);
            })
        }
        Which::Bwd => {
            let dout = randn(&mut rng, n * m);
            let (mut dx, mut dw) = (vec![0.0f32; n * k], vec![0.0f32; m * k]);
            measure(warmup, reps, || {
                backend::matmul_bwd(be, &mut dx, &mut dw, &dout, &x, &w, n, k, m);
                black_box((&dx, &dw));
            })
        }
    };
    matmul_flops(which, n, k, m) / secs / 1e9
}

pub fn matmul_gflops(be: Backend, n: usize, k: usize, m: usize, which: Which) -> f64 {
    matmul_gflops_with(be, n, k, m, which, WARMUP, REPS)
}

fn rand_batch(rng: &mut Rng, cfg: &Config, b: usize, t: usize) -> Batch {
    let extra = cfg.mtp;
    let (mut inputs, mut targets) = (Vec::new(), Vec::new());
    for _ in 0..b {
        let w: Vec<u16> = (0..t + 1 + extra).map(|_| rng.below(cfg.vocab) as u16).collect();
        inputs.extend_from_slice(&w[..t]);
        targets.extend_from_slice(&w[1..]);
    }
    Batch { b, t, extra, inputs, targets, mask: vec![1.0; b * t] }
}

/// Tokens per second of one forward + backward + optimizer step on random data.
pub fn train_step_tok_per_s_with(
    be: Backend,
    cfg: &Config,
    b: usize,
    t: usize,
    warmup: usize,
    reps: usize,
) -> f64 {
    let mut rng = Rng::new(1);
    let mut p = Params::init(cfg, &mut rng);
    let mut g = p.zeros_like();
    let mut acts = Acts::new(cfg, b, t);
    let mut opt = Optimizer::new(&p);
    let batch = rand_batch(&mut rng, cfg, b, t);
    let secs = measure(warmup, reps, || {
        g.zero();
        black_box(forward(&p, cfg, be, &mut acts, &batch));
        backward(&p, cfg, be, &mut acts, &batch, &mut g);
        opt.step(&mut p, &g, 1.0);
    });
    (b * t) as f64 / secs
}

pub fn train_step_tok_per_s(be: Backend, cfg: &Config, b: usize, t: usize) -> f64 {
    train_step_tok_per_s_with(be, cfg, b, t, WARMUP, REPS)
}

const OPS: [&str; 9] = [
    "matmul fwd",
    "matmul bwd",
    "attn fwd",
    "attn bwd",
    "rmsnorm",
    "relu2",
    "softcap+xent",
    "rope",
    "optimizer",
];
const MM_F: usize = 0;
const MM_B: usize = 1;
const AT_F: usize = 2;
const AT_B: usize = 3;
const RMS: usize = 4;
const RELU: usize = 5;
const CAP: usize = 6;
const ROPE: usize = 7;
const OPT: usize = 8;

fn timed(slot: &mut f64, f: impl FnOnce()) {
    let t0 = Instant::now();
    f();
    *slot += t0.elapsed().as_secs_f64();
}

/// Seconds per op type for one training step (median over `reps`), plus a `sum` row.
///
/// This mirrors the op sequence of `model::forward`/`model::backward` (embedding
/// gather, residual adds and the zero-fills are not timed; MTP heads are not run)
/// and calls the same dispatchers and scalar ops with the model's shapes, so it must
/// be kept in step with model.rs. Intermediate buffers are random, not chained: only
/// the timing is meaningful. Rows: the 8 op types plus `rope`, `optimizer`, `sum`.
pub fn profile_step_with(
    be: Backend,
    cfg: &Config,
    b: usize,
    t: usize,
    warmup: usize,
    reps: usize,
) -> Vec<(String, f64)> {
    let (n, d, h, v) = (b * t, cfg.d, cfg.heads, cfg.vocab);
    let mut rng = Rng::new(2);
    let mut p = Params::init(cfg, &mut rng);
    let mut g = p.zeros_like();
    let mut opt = Optimizer::new(&p);
    let x = randn(&mut rng, n * d);
    let rstd = vec![1.0f32; n];
    let (mut ln, mut sd) = (vec![0.0f32; n * d], vec![0.0f32; n * d]);
    let qkv = randn(&mut rng, n * 3 * d);
    let (mut att, mut ao) = (vec![0.0f32; b * h * t * t], vec![0.0f32; n * d]);
    let up = randn(&mut rng, n * 4 * d);
    let (mut act, mut up_buf) = (vec![0.0f32; n * 4 * d], vec![0.0f32; n * 4 * d]);
    let mut rope_buf = randn(&mut rng, n * d);
    let (mut raw, mut capped) = (vec![0.0f32; n * v], vec![0.0f32; n * v]);
    let mut dlogits = vec![0.0f32; n * v];
    let tgt: Vec<u16> = (0..n).map(|_| rng.below(v) as u16).collect();
    let mask = vec![1.0f32; n];
    let (mut dx, mut dln, mut dqkv) = (vec![0.0; n * d], vec![0.0; n * d], vec![0.0; n * 3 * d]);
    let (mut dup, mut dact) = (vec![0.0; n * 4 * d], vec![0.0; n * 4 * d]);
    let mut sv = vec![0.0f32; n * v];
    let mut qkv_work = qkv.clone();
    let mut dg = vec![0.0f32; d];
    let mut rs = rstd.clone();
    // Gradients are accumulated into freely (values are irrelevant to timing).
    for tn in g.tensors_mut() {
        tn.iter_mut().for_each(|v| *v = 1e-3 * rng.normal());
    }

    let mut step = || -> [f64; OPS.len()] {
        let mut tm = [0.0f64; OPS.len()];
        let tl = &mut tm;
        // Forward.
        for lp in &p.layers {
            timed(&mut tl[RMS], || scalar::rmsnorm_fwd(&mut ln, &mut rs, &x, &lp.attn_norm, n, d));
            timed(&mut tl[MM_F], || {
                backend::matmul_fwd(be, &mut qkv_work, &ln, &lp.wqkv.data, n, d, 3 * d)
            });
            timed(&mut tl[ROPE], || {
                scalar::rope_fwd(&mut rope_buf, b, t, h, d / h);
                scalar::rope_fwd(&mut rope_buf, b, t, h, d / h);
            });
            timed(&mut tl[AT_F], || backend::attn_fwd(be, &mut ao, &mut att, &qkv, b, t, d, h));
            timed(&mut tl[MM_F], || backend::matmul_fwd(be, &mut sd, &ao, &lp.wo.data, n, d, d));
            timed(&mut tl[RMS], || scalar::rmsnorm_fwd(&mut ln, &mut rs, &x, &lp.mlp_norm, n, d));
            timed(&mut tl[MM_F], || {
                backend::matmul_fwd(be, &mut up_buf, &ln, &lp.w_up.data, n, d, 4 * d)
            });
            timed(&mut tl[RELU], || scalar::relu2_fwd(&mut act, &up));
            timed(&mut tl[MM_F], || {
                backend::matmul_fwd(be, &mut sd, &act, &lp.w_down.data, n, 4 * d, d)
            });
        }
        timed(&mut tl[RMS], || scalar::rmsnorm_fwd(&mut ln, &mut rs, &x, &p.final_norm, n, d));
        timed(&mut tl[MM_F], || backend::matmul_fwd(be, &mut raw, &ln, &p.wte.data, n, d, v));
        timed(&mut tl[CAP], || {
            scalar::softcap_fwd(&mut capped, &raw, cfg.softcap);
            scalar::xent_fwd_bwd(&mut dlogits, &capped, &tgt, &mask, n, v);
        });

        // Backward.
        timed(&mut tl[CAP], || {
            sv.fill(0.0);
            scalar::softcap_bwd(&mut sv, &dlogits, &capped, cfg.softcap);
        });
        dln.fill(0.0);
        timed(&mut tl[MM_B], || {
            backend::matmul_bwd(be, &mut dln, &mut g.wte.data, &sv, &ln, &p.wte.data, n, d, v)
        });
        dx.fill(0.0);
        timed(&mut tl[RMS], || {
            scalar::rmsnorm_bwd(&mut dx, &mut dg, &dln, &x, &p.final_norm, &rstd, n, d)
        });
        for (lp, lg) in p.layers.iter().zip(g.layers.iter_mut()).rev() {
            dact.fill(0.0);
            timed(&mut tl[MM_B], || {
                backend::matmul_bwd(be, &mut dact, &mut lg.w_down.data, &dx, &act, &lp.w_down.data, n, 4 * d, d)
            });
            dup.fill(0.0);
            timed(&mut tl[RELU], || scalar::relu2_bwd(&mut dup, &dact, &up));
            dln.fill(0.0);
            timed(&mut tl[MM_B], || {
                backend::matmul_bwd(be, &mut dln, &mut lg.w_up.data, &dup, &ln, &lp.w_up.data, n, d, 4 * d)
            });
            timed(&mut tl[RMS], || {
                scalar::rmsnorm_bwd(&mut dx, &mut lg.mlp_norm, &dln, &x, &lp.mlp_norm, &rstd, n, d)
            });
            sd.fill(0.0);
            timed(&mut tl[MM_B], || {
                backend::matmul_bwd(be, &mut sd, &mut lg.wo.data, &dx, &ao, &lp.wo.data, n, d, d)
            });
            dqkv.fill(0.0);
            timed(&mut tl[AT_B], || backend::attn_bwd(be, &mut dqkv, &sd, &qkv, &att, b, t, d, h));
            timed(&mut tl[ROPE], || {
                scalar::rope_bwd(&mut rope_buf, b, t, h, d / h);
                scalar::rope_bwd(&mut rope_buf, b, t, h, d / h);
            });
            dln.fill(0.0);
            timed(&mut tl[MM_B], || {
                backend::matmul_bwd(be, &mut dln, &mut lg.wqkv.data, &dqkv, &ln, &lp.wqkv.data, n, d, 3 * d)
            });
            timed(&mut tl[RMS], || {
                scalar::rmsnorm_bwd(&mut dx, &mut lg.attn_norm, &dln, &x, &lp.attn_norm, &rstd, n, d)
            });
        }
        timed(&mut tl[OPT], || opt.step(&mut p, &g, 1.0));
        tm
    };

    for _ in 0..warmup {
        black_box(step());
    }
    let runs: Vec<[f64; OPS.len()]> = (0..reps.max(1)).map(|_| step()).collect();
    let mut out: Vec<(String, f64)> = OPS
        .iter()
        .enumerate()
        .map(|(i, name)| (name.to_string(), median(runs.iter().map(|r| r[i]).collect())))
        .collect();
    let sum = out.iter().map(|(_, s)| s).sum();
    out.push(("sum".to_string(), sum));
    out
}

pub fn profile_step(be: Backend, cfg: &Config, b: usize, t: usize) -> Vec<(String, f64)> {
    profile_step_with(be, cfg, b, t, WARMUP, REPS)
}

/// Tiny model for `--quick` and tests.
pub fn quick_cfg() -> Config {
    Config { vocab: 64, d: 32, layers: 2, heads: 2, ctx: 16, softcap: 30.0, mtp: 0, bitnet: false }
}

/// (year, month, day) in UTC for a Unix timestamp in seconds.
pub fn civil_date(unix_secs: u64) -> (i64, u32, u32) {
    let z = (unix_secs / 86400) as i64 + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + (m <= 2) as i64, m, d)
}

#[derive(Clone, Debug)]
pub struct BenchArgs {
    pub backends: Vec<Backend>,
    pub threads: usize,
    pub profile: bool,
    pub quick: bool,
}

pub const BENCH_USAGE: &str = "usage: bench [--backend scalar,blocked,avx512,avx512mt,asm|all] [--threads 4] [--profile-step] [--quick]\n  --quick: tiny shapes, 1 warmup, 1 rep; printed only, not appended to bench/results.md";

const ALL: [Backend; 5] =
    [Backend::Scalar, Backend::Blocked, Backend::Avx512, Backend::Avx512Mt, Backend::Asm];

impl BenchArgs {
    pub fn parse(mut it: impl Iterator<Item = String>) -> Result<BenchArgs, String> {
        let mut a = BenchArgs { backends: ALL.to_vec(), threads: 4, profile: false, quick: false };
        while let Some(f) = it.next() {
            match f.as_str() {
                "--backend" => {
                    let v = it.next().ok_or("--backend needs a value")?;
                    a.backends = if v == "all" {
                        ALL.to_vec()
                    } else {
                        v.split(',').map(str::parse).collect::<Result<_, _>>()?
                    };
                }
                "--threads" => {
                    let v = it.next().ok_or("--threads needs a value")?;
                    a.threads = v.parse().map_err(|_| format!("bad value {v:?} for --threads"))?;
                    if a.threads == 0 {
                        return Err("--threads must be >= 1".into());
                    }
                }
                "--profile-step" => a.profile = true,
                "--quick" => a.quick = true,
                _ => return Err(format!("unknown flag {f:?}")),
            }
        }
        Ok(a)
    }
}

const SHAPES: [(usize, usize, usize); 5] =
    [(4096, 128, 384), (4096, 128, 128), (4096, 128, 512), (4096, 512, 128), (4096, 128, 2048)];
const QUICK_SHAPES: [(usize, usize, usize); 2] = [(64, 32, 48), (33, 17, 19)];

/// Runs everything `args` asks for and returns a dated markdown section.
/// Backends the CPU cannot run are listed as skipped.
pub fn report(args: &BenchArgs) -> String {
    let (warm, reps) = if args.quick { (1, 1) } else { (WARMUP, REPS) };
    let peak_secs = if args.quick { 0.05 } else { 0.5 };
    let shapes: &[(usize, usize, usize)] = if args.quick { &QUICK_SHAPES } else { &SHAPES };
    let (cfg, b, t) = if args.quick { (quick_cfg(), 2, 16) } else { (Config::baseline(), 16, 256) };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, mo, da) = civil_date(now);

    let mut md = format!(
        "## bench {y:04}-{mo:02}-{da:02}{}\n\nthreads: {}; step: b={b}, t={t}, d={}, layers={}\n\n",
        if args.quick { " (quick, not a measurement)" } else { "" },
        args.threads,
        cfg.d,
        cfg.layers,
    );
    let (p1, pn) = (fma_peak_gflops(1, peak_secs), fma_peak_gflops(args.threads, peak_secs));
    md += &format!(
        "| measured FMA peak | GFLOPS |\n|---|---|\n| 1 thread | {p1:.1} |\n| {} threads | {pn:.1} |\n\n",
        args.threads
    );

    let mut run: Vec<Backend> = Vec::new();
    for &be in &args.backends {
        match backend::check_supported(be) {
            Ok(()) => run.push(be),
            Err(e) => md += &format!("skipped {be:?}: {e}\n\n"),
        }
    }
    let pct = |g: f64, be: Backend| {
        let peak = if be == Backend::Avx512Mt { pn } else { p1 };
        if peak > 0.0 {
            format!("{:.0}%", 100.0 * g / peak)
        } else {
            "n/a".to_string()
        }
    };

    md += "| matmul (n,k,m) | backend | fwd GFLOPS | fwd % of peak | bwd GFLOPS | bwd % of peak |\n|---|---|---|---|---|---|\n";
    for &(n, k, m) in shapes {
        for &be in &run {
            let f = matmul_gflops_with(be, n, k, m, Which::Fwd, warm, reps);
            let bw = matmul_gflops_with(be, n, k, m, Which::Bwd, warm, reps);
            md += &format!(
                "| ({n},{k},{m}) | {be:?} | {f:.1} | {} | {bw:.1} | {} |\n",
                pct(f, be),
                pct(bw, be)
            );
        }
    }
    md += "\n(% of peak: 1-thread FMA peak, except Avx512Mt vs the N-thread peak)\n\n";

    md += "| backend | train step tok/s | ms/step |\n|---|---|---|\n";
    for &be in &run {
        let tps = train_step_tok_per_s_with(be, &cfg, b, t, warm, reps);
        md += &format!("| {be:?} | {tps:.0} | {:.2} |\n", 1e3 * (b * t) as f64 / tps);
    }
    md += "\n";

    if args.profile {
        for &be in &run {
            let prof = profile_step_with(be, &cfg, b, t, warm, reps);
            let total = prof.last().map(|(_, s)| *s).unwrap_or(0.0).max(f64::MIN_POSITIVE);
            md += &format!("per-op profile, {be:?}\n\n| op | ms | % |\n|---|---|---|\n");
            for (name, s) in &prof {
                md += &format!("| {name} | {:.3} | {:.1}% |\n", 1e3 * s, 100.0 * s / total);
            }
            md += "\n";
        }
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{has_avx512, Backend};
    use std::time::Duration;

    #[test]
    fn measure_returns_median() {
        // Calls 0..2 are warmups (instant); the 5 timed calls sleep 3, 120, 1, 120, 1 ms,
        // so the median is the 3 ms call. Sleeps never undershoot, so only the upper bound
        // can be disturbed by scheduling, and 120 ms outliers must not leak into it.
        let sleeps_ms = [0u64, 0, 3, 120, 1, 120, 1];
        let mut i = 0;
        let med = measure(2, 5, || {
            std::thread::sleep(Duration::from_millis(sleeps_ms[i]));
            i += 1;
        });
        assert_eq!(i, 7);
        assert!(med >= 0.003, "median {med}");
        assert!(med < 0.100, "median {med}");
        assert_eq!(median(vec![5.0, 1.0, 3.0]), 3.0);
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), 2.5);
    }

    #[test]
    fn flops_formula() {
        assert_eq!(matmul_flops(Which::Fwd, 4096, 128, 384), 2.0 * 4096.0 * 128.0 * 384.0);
        assert_eq!(matmul_flops(Which::Bwd, 4096, 128, 384), 4.0 * 4096.0 * 128.0 * 384.0);
        assert_eq!(matmul_flops(Which::Fwd, 1, 1, 1), 2.0);
    }

    #[test]
    fn civil_date_known_values() {
        assert_eq!(civil_date(0), (1970, 1, 1));
        assert_eq!(civil_date(951_782_400), (2000, 2, 29));
        assert_eq!(civil_date(1_767_139_200), (2025, 12, 31));
    }

    #[test]
    fn parse_args() {
        let a = |v: &[&str]| BenchArgs::parse(v.iter().map(|s| s.to_string()));
        let d = a(&[]).unwrap();
        assert_eq!(d.threads, 4);
        assert!(!d.quick && !d.profile);
        assert_eq!(d.backends.len(), 5);
        let x = a(&["--backend", "scalar,blocked", "--threads", "2", "--quick", "--profile-step"]).unwrap();
        assert_eq!(x.backends, vec![Backend::Scalar, Backend::Blocked]);
        assert_eq!(x.threads, 2);
        assert!(x.quick && x.profile);
        assert!(a(&["--backend", "nope"]).is_err());
        assert!(a(&["--threads", "0"]).is_err());
        assert!(a(&["--wat"]).is_err());
    }

    #[test]
    fn fma_peak_absent_or_positive() {
        let g = fma_peak_gflops(1, 0.02);
        if has_avx512() {
            assert!(g.is_finite() && g > 0.0, "{g}");
        } else {
            assert_eq!(g, 0.0);
        }
    }

    #[test]
    fn bench_quick_smoke() {
        let mut bes = vec![Backend::Scalar, Backend::Blocked];
        if has_avx512() {
            bes.push(Backend::Avx512);
        }
        let cfg = quick_cfg();
        for be in bes {
            for which in [Which::Fwd, Which::Bwd] {
                // Odd shape on purpose.
                let g = matmul_gflops_with(be, 7, 33, 19, which, 1, 1);
                assert!(g.is_finite() && g > 0.0, "{be:?} {which:?}: {g}");
            }
            let tps = train_step_tok_per_s_with(be, &cfg, 2, 16, 1, 1);
            assert!(tps.is_finite() && tps > 0.0, "{be:?}: {tps}");
            let prof = profile_step_with(be, &cfg, 2, 16, 1, 1);
            let names: Vec<&str> = prof.iter().map(|(n, _)| n.as_str()).collect();
            for want in [
                "matmul fwd", "matmul bwd", "attn fwd", "attn bwd", "rmsnorm", "relu2",
                "softcap+xent", "optimizer",
            ] {
                assert!(names.contains(&want), "{be:?}: missing {want} in {names:?}");
            }
            assert!(prof.iter().all(|(_, s)| s.is_finite() && *s >= 0.0), "{prof:?}");
            assert!(prof.iter().any(|(_, s)| *s > 0.0));
        }
    }

    #[test]
    fn report_quick_has_tables() {
        let args = BenchArgs {
            backends: vec![Backend::Scalar, Backend::Blocked],
            threads: 2,
            profile: true,
            quick: true,
        };
        let md = report(&args);
        for want in ["peak", "matmul fwd", "tok/s", "Scalar", "Blocked", "% of peak"] {
            assert!(md.contains(want), "missing {want:?} in:\n{md}");
        }
    }
}
