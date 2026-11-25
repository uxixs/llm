//! 60-second budgeted training loop: pretrain phase, SFT phase, fixed validation,
//! checkpoint and results log. The budget clock starts at each phase's first step;
//! data loading, init, validation and saving are excluded.

use crate::backend::Backend;
use crate::bpe::Bpe;
use crate::data::{Batch, PretrainBatcher, SftBatcher};
use crate::model::{backward, forward, Acts, Config, Params};
use crate::optim::Optimizer;
use crate::rng::Rng;
use std::cell::Cell;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Pretrain validation: first this many windows of 257 tokens.
pub const VAL_PRETRAIN_WINDOWS: usize = 128;
/// SFT validation: first this many records.
pub const VAL_SFT_RECORDS: usize = 256;

pub const RESULTS_HEADER: &str = "timestamp,name,backend,bitnet,mtp,params,pretrain_steps,pretrain_tokens,sft_steps,sft_tokens,train_tok_per_s,wall_s,val_loss_pretrain,val_loss_sft";

pub trait Clock {
    /// Seconds since an arbitrary fixed origin.
    fn now(&self) -> f64;
}

pub struct RealClock(Instant);

impl RealClock {
    pub fn new() -> RealClock {
        RealClock(Instant::now())
    }
}

impl Default for RealClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for RealClock {
    fn now(&self) -> f64 {
        self.0.elapsed().as_secs_f64()
    }
}

/// Manually advanced clock for tests.
pub struct FakeClock(Cell<f64>);

impl FakeClock {
    pub fn new() -> FakeClock {
        FakeClock(Cell::new(0.0))
    }
    pub fn advance(&self, s: f64) {
        self.0.set(self.0.get() + s);
    }
}

impl Default for FakeClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for FakeClock {
    fn now(&self) -> f64 {
        self.0.get()
    }
}

/// Learning-rate multiplier at `frac = elapsed / phase_budget`: linear warmup from 0.1
/// (so step 0 is not a no-op) to 1.0 over 0..0.05, flat 1.0, linear decay to 0.1 over 0.7..1.0.
pub fn lr_mult(frac: f64) -> f32 {
    // NaN (e.g. 0/0 from a zero budget) must never reach the optimizer.
    let f = if frac.is_nan() { 0.0 } else { frac.clamp(0.0, 1.0) };
    let m = if f < 0.05 {
        0.1 + 0.9 * f / 0.05
    } else if f <= 0.7 {
        1.0
    } else {
        1.0 - 0.9 * (f - 0.7) / 0.3
    };
    m as f32
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseStats {
    pub steps: usize,
    pub last_loss: f32,
    /// Seconds from the first step's start to the last step's end.
    pub elapsed: f64,
}

/// Runs `step(lr_mult)` until the budget is used. Before each step the previous step's
/// duration is the estimate for the next; a step that would overrun is not started, but
/// at least one step always runs, except when `budget_s <= 0` (or NaN): a zero-length
/// phase is skipped entirely and runs zero steps.
pub fn run_phase(clock: &dyn Clock, budget_s: f64, mut step: impl FnMut(f32) -> f32) -> PhaseStats {
    if !(budget_s > 0.0) {
        return PhaseStats { steps: 0, last_loss: 0.0, elapsed: 0.0 };
    }
    let start = clock.now();
    let mut dur = f64::INFINITY;
    let (mut steps, mut last_loss) = (0, 0.0);
    loop {
        let elapsed = clock.now() - start;
        if steps >= 1 && elapsed + dur > budget_s {
            break;
        }
        let t0 = clock.now();
        last_loss = step(lr_mult(elapsed / budget_s));
        dur = clock.now() - t0;
        steps += 1;
    }
    PhaseStats { steps, last_loss, elapsed: clock.now() - start }
}

/// Appends one CSV row, writing the header first if the file is missing or empty.
pub fn append_results(path: &Path, row: &str) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let fresh = std::fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if fresh {
        writeln!(f, "{RESULTS_HEADER}")?;
    }
    writeln!(f, "{row}")
}

#[derive(Clone, Debug)]
pub struct Args {
    pub data: PathBuf,
    pub backend: Backend,
    pub budget: f64,
    pub pretrain_frac: f64,
    pub batch: usize,
    pub seed: u64,
    pub mtp: usize,
    pub bitnet: bool,
    pub name: String,
    pub muon_lr: Option<f32>,
    pub adam_lr: Option<f32>,
    /// Accepted for the multithreaded backend (Task 12); unused so far.
    pub threads: Option<usize>,
}

pub const USAGE: &str = "usage: train --name <run> [--data data] [--backend scalar|blocked|avx512|avx512mt|asm] [--budget 60] [--pretrain-frac 0.75] [--batch 16] [--seed 42] [--mtp N] [--bitnet] [--muon-lr X] [--adam-lr X] [--threads N]";

fn val<T: std::str::FromStr>(flag: &str, v: Option<String>) -> Result<T, String> {
    let v = v.ok_or(format!("{flag} needs a value"))?;
    v.parse().map_err(|_| format!("bad value {v:?} for {flag}"))
}

impl Args {
    pub fn parse(mut it: impl Iterator<Item = String>) -> Result<Args, String> {
        let mut a = Args {
            data: PathBuf::from("data"),
            backend: Backend::Scalar,
            budget: 60.0,
            pretrain_frac: 0.75,
            batch: 16,
            seed: 42,
            mtp: 0,
            bitnet: false,
            name: String::new(),
            muon_lr: None,
            adam_lr: None,
            threads: None,
        };
        while let Some(f) = it.next() {
            match f.as_str() {
                "--data" => a.data = PathBuf::from(val::<String>(&f, it.next())?),
                "--backend" => a.backend = val(&f, it.next())?,
                "--budget" => a.budget = val(&f, it.next())?,
                "--pretrain-frac" => a.pretrain_frac = val(&f, it.next())?,
                "--batch" => a.batch = val(&f, it.next())?,
                "--seed" => a.seed = val(&f, it.next())?,
                "--mtp" => a.mtp = val(&f, it.next())?,
                "--bitnet" => a.bitnet = true,
                "--name" => a.name = val(&f, it.next())?,
                "--muon-lr" => a.muon_lr = Some(val(&f, it.next())?),
                "--adam-lr" => a.adam_lr = Some(val(&f, it.next())?),
                "--threads" => a.threads = Some(val(&f, it.next())?),
                _ => return Err(format!("unknown argument {f}")),
            }
        }
        if a.name.is_empty() {
            return Err("--name is required".into());
        }
        if a.batch == 0 || !(a.budget > 0.0) || !(0.0..=1.0).contains(&a.pretrain_frac) {
            return Err("need --batch > 0, --budget > 0, 0 <= --pretrain-frac <= 1".into());
        }
        Ok(a)
    }
}

/// Mask-weighted mean loss over `n_batches` batches produced by `make`.
fn eval(
    p: &Params,
    cfg: &Config,
    be: Backend,
    acts: &mut Acts,
    n_batches: usize,
    make: impl Fn(usize) -> Batch,
) -> f64 {
    let (mut sum, mut wsum) = (0.0f64, 0.0f64);
    for i in 0..n_batches {
        let batch = make(i);
        let w: f64 = batch.mask.iter().map(|&m| m as f64).sum();
        if w == 0.0 {
            continue;
        }
        sum += forward(p, cfg, be, acts, &batch) as f64 * w;
        wsum += w;
    }
    if wsum == 0.0 {
        f64::NAN
    } else {
        sum / wsum
    }
}

struct Phase<'a> {
    name: &'a str,
    budget: f64,
    tokens_per_step: usize,
}

fn train_phase(
    ph: Phase,
    clock: &RealClock,
    p: &mut Params,
    cfg: &Config,
    be: Backend,
    acts: &mut Acts,
    opt: &mut Optimizer,
    mut next_batch: impl FnMut() -> Batch,
) -> PhaseStats {
    let mut g = p.zeros_like();
    let start = clock.now();
    let (mut n, mut prev) = (0usize, start);
    run_phase(clock, ph.budget, |m| {
        g.zero(); // backward accumulates
        let batch = next_batch();
        let loss = forward(p, cfg, be, acts, &batch);
        backward(p, cfg, be, acts, &batch, &mut g);
        opt.step(p, &g, m);
        let now = clock.now();
        println!(
            "{} {n} {loss:.4} {:.0} {:.2}",
            ph.name,
            ph.tokens_per_step as f64 / (now - prev),
            now - start
        );
        n += 1;
        prev = now;
        loss
    })
}

fn new_optimizer(p: &Params, args: &Args) -> Optimizer {
    let mut opt = Optimizer::new(p);
    if let Some(lr) = args.muon_lr {
        opt.muon.lr = lr;
    }
    if let Some(lr) = args.adam_lr {
        opt.adam.lr = lr;
    }
    opt
}

pub fn run(args: &Args) -> Result<(), String> {
    crate::backend::check_supported(args.backend)?;
    let ioe = |what: &str, e: io::Error| format!("{what}: {e}");
    let bpe = Bpe::load(&args.data.join("tokenizer.bin")).map_err(|e| ioe("tokenizer.bin", e))?;
    let mut cfg = Config::baseline();
    cfg.mtp = args.mtp;
    cfg.bitnet = args.bitnet;
    if bpe.vocab.len() != cfg.vocab {
        return Err(format!("tokenizer vocab {} != model vocab {}", bpe.vocab.len(), cfg.vocab));
    }
    let mut pre = PretrainBatcher::load(&args.data.join("pretrain_train.bin"))
        .map_err(|e| ioe("pretrain_train.bin", e))?;
    let pre_val = PretrainBatcher::load(&args.data.join("pretrain_val.bin"))
        .map_err(|e| ioe("pretrain_val.bin", e))?;
    let (au, en) = (bpe.assistant(), bpe.end());
    let mut sft = SftBatcher::load(&args.data.join("sft_train.bin"), au, en)
        .map_err(|e| ioe("sft_train.bin", e))?;
    let sft_val = SftBatcher::load(&args.data.join("sft_val.bin"), au, en)
        .map_err(|e| ioe("sft_val.bin", e))?;
    if sft.is_empty() {
        return Err("sft_train.bin has no records".into());
    }

    let mut p = Params::init(&cfg, &mut Rng::new(args.seed));
    let (b, t) = (args.batch, cfg.ctx);
    let mut acts = Acts::new(&cfg, b, t);
    let tokens_per_step = b * t;
    let clock = RealClock::new();

    let mut opt = new_optimizer(&p, args);
    let mut rng = Rng::new(args.seed.wrapping_add(1));
    let ph = Phase { name: "pretrain", budget: args.budget * args.pretrain_frac, tokens_per_step };
    let pre_stats = train_phase(ph, &clock, &mut p, &cfg, args.backend, &mut acts, &mut opt, || {
        pre.next(&mut rng, b, t, cfg.mtp)
    });

    // SFT continues from the pretrained weights with fresh optimizer moments.
    let mut opt = new_optimizer(&p, args);
    let mut rng = Rng::new(args.seed.wrapping_add(2));
    let ph = Phase { name: "sft", budget: args.budget * (1.0 - args.pretrain_frac), tokens_per_step };
    let sft_stats = train_phase(ph, &clock, &mut p, &cfg, args.backend, &mut acts, &mut opt, || {
        sft.next(&mut rng, b, t, 0)
    });

    // Validation (outside the budget). Rows past the requested count are masked out.
    let n_win = pre_val.num_windows(t);
    if n_win < VAL_PRETRAIN_WINDOWS {
        eprintln!("warning: pretrain_val.bin has only {n_win} windows (wanted {VAL_PRETRAIN_WINDOWS})");
    }
    let n_win = n_win.min(VAL_PRETRAIN_WINDOWS);
    let val_pre = eval(&p, &cfg, args.backend, &mut acts, n_win.div_ceil(b), |i| {
        let mut bt = pre_val.window_batch(i * b, b, t);
        bt.mask[(n_win - i * b).min(b) * t..].fill(0.0);
        bt
    });
    let n_rec = sft_val.len();
    if n_rec < VAL_SFT_RECORDS {
        eprintln!("warning: sft_val.bin has only {n_rec} records (wanted {VAL_SFT_RECORDS})");
    }
    let n_rec = n_rec.min(VAL_SFT_RECORDS);
    let val_sft = eval(&p, &cfg, args.backend, &mut acts, n_rec.div_ceil(b), |i| {
        let mut bt = sft_val.records_batch(i * b, b, t);
        bt.mask[(n_rec - i * b).min(b) * t..].fill(0.0);
        bt
    });
    println!("val_loss_pretrain {val_pre:.4} val_loss_sft {val_sft:.4}");

    let dir = Path::new("runs").join(&args.name);
    std::fs::create_dir_all(&dir).map_err(|e| ioe("runs dir", e))?;
    p.save(&cfg, &dir.join("model.bin")).map_err(|e| ioe("model.bin", e))?;

    // Throughput counts every position (masked SFT padding included): steps * batch * ctx.
    let pre_tok = pre_stats.steps * tokens_per_step;
    let sft_tok = sft_stats.steps * tokens_per_step;
    let wall = pre_stats.elapsed + sft_stats.elapsed;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let row = format!(
        "{ts},{},{},{},{},{},{},{pre_tok},{},{sft_tok},{:.1},{wall:.2},{val_pre:.4},{val_sft:.4}",
        args.name,
        format!("{:?}", args.backend).to_lowercase(),
        args.bitnet,
        args.mtp,
        p.num_params(),
        pre_stats.steps,
        sft_stats.steps,
        if wall > 0.0 { (pre_tok + sft_tok) as f64 / wall } else { 0.0 },
    );
    append_results(&Path::new("runs").join("results.csv"), &row).map_err(|e| ioe("results.csv", e))?;
    println!("train time {wall:.1}s, row appended to runs/results.csv");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lr_mult_shape() {
        assert!((lr_mult(0.0) - 0.1).abs() < 1e-6);
        assert!((lr_mult(0.025) - 0.55).abs() < 1e-6);
        assert!((lr_mult(0.05) - 1.0).abs() < 1e-6);
        assert_eq!(lr_mult(0.5), 1.0);
        assert!((lr_mult(0.7) - 1.0).abs() < 1e-6);
        assert!((lr_mult(0.85) - 0.55).abs() < 1e-6);
        assert!((lr_mult(1.0) - 0.1).abs() < 1e-6);
        assert!((lr_mult(1.5) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn budget_stops_before_overrun() {
        let clock = FakeClock::new();
        let st = run_phase(&clock, 20.0, |_| {
            clock.advance(7.0);
            1.5
        });
        assert_eq!(st.steps, 2);
        assert_eq!(st.last_loss, 1.5);
    }

    #[test]
    fn at_least_one_step_per_phase() {
        let clock = FakeClock::new();
        let st = run_phase(&clock, 20.0, |_| {
            clock.advance(30.0);
            2.0
        });
        assert_eq!(st.steps, 1);
    }

    #[test]
    fn zero_budget_phase_runs_no_steps() {
        let clock = FakeClock::new();
        for budget in [0.0, -1.0, f64::NAN] {
            let mut calls = 0;
            let st = run_phase(&clock, budget, |_| {
                calls += 1;
                1.0
            });
            assert_eq!((st.steps, calls, st.elapsed), (0, 0, 0.0));
        }
    }

    #[test]
    fn lr_mult_never_nan() {
        for f in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0 / 0.0, -3.0] {
            let m = lr_mult(f);
            assert!(m.is_finite() && (0.1..=1.0).contains(&m), "{f} -> {m}");
        }
    }

    #[test]
    fn lr_mult_passed_to_step() {
        let clock = FakeClock::new();
        let mut seen = vec![];
        run_phase(&clock, 10.0, |m| {
            seen.push(m);
            clock.advance(1.0);
            0.0
        });
        assert!((seen[0] - 0.1).abs() < 1e-6);
        assert!(seen.iter().all(|&m| (0.0..=1.0).contains(&m)));
        assert!(seen[5] == 1.0 && *seen.last().unwrap() < 0.5);
    }

    #[test]
    fn results_csv_header_once() {
        let dir = std::env::temp_dir().join(format!("rllm_csv_test_{}", std::process::id()));
        let p = dir.join("runs").join("results.csv");
        std::fs::remove_dir_all(&dir).ok();
        append_results(&p, "a,1").unwrap();
        append_results(&p, "b,2").unwrap();
        let s = std::fs::read_to_string(&p).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines, vec![RESULTS_HEADER, "a,1", "b,2"]);
        assert_eq!(s.matches("timestamp").count(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn args_parse() {
        let a = Args::parse(
            ["--data", "d", "--name", "x", "--batch", "4", "--bitnet", "--mtp", "2", "--muon-lr", "0.05"]
                .iter()
                .map(|s| s.to_string()),
        )
        .unwrap();
        assert_eq!((a.batch, a.mtp, a.bitnet, a.budget, a.pretrain_frac), (4, 2, true, 60.0, 0.75));
        assert_eq!(a.muon_lr, Some(0.05));
        assert_eq!(a.adam_lr, None);
        assert!(Args::parse(["--name".to_string()].into_iter()).is_err());
        assert!(Args::parse(["--bogus".to_string()].into_iter()).is_err());
        assert!(Args::parse(std::iter::empty()).is_err()); // --name required
    }
}
