//! Inference: KV-cached single-token forward, sampling, streaming UTF-8 decoding,
//! chat prompt building, and the chat/batch driver behind `bin/chat`.

use crate::backend::{self, Backend};
use crate::bpe::Bpe;
use crate::model::{Config, Params};
use crate::ops::scalar;
use crate::rng::Rng;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Per-layer K/V (post-RoPE K) `[ctx, d]`, plus scratch so a step allocates nothing
/// except the returned logits vector.
pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    x: Vec<f32>,
    ln: Vec<f32>,
    rstd: [f32; 1],
    qkv: Vec<f32>,
    scores: Vec<f32>,
    attn: Vec<f32>,
    tmp: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    logits: Vec<f32>,
    raw: Vec<f32>,
    /// RoPE angles for positions `0..ctx`, head dim `d / heads`.
    rope: scalar::RopeTable,
}

impl KvCache {
    pub fn new(cfg: &Config) -> KvCache {
        let (d, z) = (cfg.d, |n: usize| vec![0.0f32; n]);
        KvCache {
            k: (0..cfg.layers).map(|_| z(cfg.ctx * d)).collect(),
            v: (0..cfg.layers).map(|_| z(cfg.ctx * d)).collect(),
            x: z(d),
            ln: z(d),
            rstd: [0.0],
            qkv: z(3 * d),
            scores: z(cfg.ctx),
            attn: z(d),
            tmp: z(d),
            up: z(4 * d),
            act: z(4 * d),
            logits: z(cfg.vocab),
            raw: z(cfg.vocab),
            rope: scalar::RopeTable::new(cfg.ctx, d / cfg.heads),
        }
    }
}

/// Logits (after softcap) for `token` at position `pos`, attending to cached positions
/// `0..pos` plus itself; appends this position's K/V to the cache. Positions must be fed
/// in order from 0. Returns one `Vec` per token (the only allocation in a step).
/// Panics if `pos >= cfg.ctx`.
pub fn step(
    p: &Params,
    cfg: &Config,
    be: Backend,
    c: &mut KvCache,
    token: u16,
    pos: usize,
) -> Vec<f32> {
    assert!(
        pos < cfg.ctx,
        "position {pos} is outside the context window (ctx = {})",
        cfg.ctx
    );
    let (d, h) = (cfg.d, cfg.heads);
    let dh = d / h;
    let scale = 1.0 / (dh as f32).sqrt();
    c.x.copy_from_slice(&p.wte.data[token as usize * d..][..d]);
    for (l, lp) in p.layers.iter().enumerate() {
        scalar::rmsnorm_fwd(&mut c.ln, &mut c.rstd, &c.x, &lp.attn_norm, 1, d);
        backend::matmul_fwd(be, &mut c.qkv, &c.ln, &lp.wqkv.data, 1, d, 3 * d);
        scalar::rope_at_table(&mut c.qkv[..d], h, dh, pos, &c.rope);
        scalar::rope_at_table(&mut c.qkv[d..2 * d], h, dh, pos, &c.rope);
        c.k[l][pos * d..(pos + 1) * d].copy_from_slice(&c.qkv[d..2 * d]);
        c.v[l][pos * d..(pos + 1) * d].copy_from_slice(&c.qkv[2 * d..]);
        for hh in 0..h {
            let q = &c.qkv[hh * dh..(hh + 1) * dh];
            let mut max = f32::NEG_INFINITY;
            for j in 0..=pos {
                let kj = &c.k[l][j * d + hh * dh..][..dh];
                let s = q.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                c.scores[j] = s;
                max = max.max(s);
            }
            let mut sum = 0.0;
            for s in &mut c.scores[..=pos] {
                *s = (*s - max).exp();
                sum += *s;
            }
            let out = &mut c.attn[hh * dh..(hh + 1) * dh];
            out.fill(0.0);
            for j in 0..=pos {
                let w = c.scores[j] / sum;
                let vj = &c.v[l][j * d + hh * dh..][..dh];
                for (o, v) in out.iter_mut().zip(vj) {
                    *o += w * v;
                }
            }
        }
        backend::matmul_fwd(be, &mut c.tmp, &c.attn, &lp.wo.data, 1, d, d);
        for (x, y) in c.x.iter_mut().zip(&c.tmp) {
            *x += y;
        }
        scalar::rmsnorm_fwd(&mut c.ln, &mut c.rstd, &c.x, &lp.mlp_norm, 1, d);
        backend::matmul_fwd(be, &mut c.up, &c.ln, &lp.w_up.data, 1, d, 4 * d);
        scalar::relu2_fwd(&mut c.act, &c.up);
        backend::matmul_fwd(be, &mut c.tmp, &c.act, &lp.w_down.data, 1, 4 * d, d);
        for (x, y) in c.x.iter_mut().zip(&c.tmp) {
            *x += y;
        }
    }
    scalar::rmsnorm_fwd(&mut c.ln, &mut c.rstd, &c.x, &p.final_norm, 1, d);
    backend::matmul_fwd(be, &mut c.raw, &c.ln, &p.wte.data, 1, d, cfg.vocab);
    scalar::softcap_fwd(&mut c.logits, &c.raw, cfg.softcap);
    c.logits.clone()
}

/// Sample a token id. `temp <= 0` is argmax (first max on ties). Otherwise softmax at `temp`,
/// keep the smallest set of top tokens whose probability mass reaches `top_p`, and draw from it.
pub fn sample(logits: &[f32], temp: f32, top_p: f32, rng: &mut Rng) -> u16 {
    if temp <= 0.0 {
        let mut best = 0;
        for (i, &l) in logits.iter().enumerate() {
            if l > logits[best] {
                best = i;
            }
        }
        return best as u16;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(f32, u16)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (((l - max) / temp).exp(), i as u16))
        .collect();
    let total: f32 = probs.iter().map(|p| p.0).sum();
    probs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut kept, mut mass) = (0, 0.0);
    for &(pr, _) in &probs {
        mass += pr;
        kept += 1;
        if mass >= top_p * total {
            break;
        }
    }
    let mut r = rng.uniform() * mass;
    for &(pr, id) in &probs[..kept] {
        if r < pr {
            return id;
        }
        r -= pr;
    }
    probs[kept - 1].1
}

/// Turns a byte stream into text, emitting only complete UTF-8. A partial multibyte tail
/// (and an invalid sequence that ends the buffer) is held back; invalid sequences are
/// replaced with U+FFFD.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> StreamDecoder {
        StreamDecoder::default()
    }

    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut at = 0;
        loop {
            match std::str::from_utf8(&self.pending[at..]) {
                Ok(s) => {
                    out.push_str(s);
                    at = self.pending.len();
                    break;
                }
                Err(e) => {
                    let ok = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[at..at + ok]).unwrap());
                    at += ok;
                    match e.error_len() {
                        // Truncated sequence at the end: wait for more bytes.
                        None => break,
                        Some(n) if at + n == self.pending.len() => break,
                        Some(n) => {
                            out.push('\u{FFFD}');
                            at += n;
                        }
                    }
                }
            }
        }
        self.pending.drain(..at);
        out
    }

    /// Emit whatever is held back (lossily) and reset.
    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// `<|user|>u<|assistant|>a<|end|>` per past turn, then `<|user|>user<|assistant|>`.
/// Drops the oldest turns until `len <= ctx - reserve`; if the new message alone is still too
/// long, keeps its last `ctx - reserve - 2` tokens inside the user/assistant wrapper.
pub fn build_prompt(
    history: &[(Vec<u16>, Vec<u16>)],
    user: &[u16],
    bpe: &Bpe,
    ctx: usize,
    reserve: usize,
) -> Vec<u16> {
    let limit = ctx.saturating_sub(reserve);
    let mut start = 0;
    while start < history.len() {
        let len: usize = history[start..].iter().map(|(u, a)| 3 + u.len() + a.len()).sum();
        if len + 2 + user.len() <= limit {
            break;
        }
        start += 1;
    }
    let user = if history[start..].is_empty() && user.len() + 2 > limit {
        &user[user.len() - limit.saturating_sub(2).min(user.len())..]
    } else {
        user
    };
    let mut out = Vec::new();
    for (u, a) in &history[start..] {
        out.push(bpe.user());
        out.extend_from_slice(u);
        out.push(bpe.assistant());
        out.extend_from_slice(a);
        out.push(bpe.end());
    }
    out.push(bpe.user());
    out.extend_from_slice(user);
    out.push(bpe.assistant());
    out
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kernel {
    Fp32,
    Ternary,
}

pub struct ChatArgs {
    pub model: PathBuf,
    pub tokenizer: PathBuf,
    pub backend: Backend,
    pub temp: f32,
    pub top_p: f32,
    pub kernel: Kernel,
    pub batch: Option<PathBuf>,
    pub seed: u64,
    /// Pool size for `--backend avx512mt` (default 4).
    pub threads: usize,
}

pub const CHAT_USAGE: &str = "usage: chat --model runs/<run>/model.bin [--tokenizer data/tokenizer.bin] [--backend scalar|blocked|avx512|avx512mt|asm] [--temp 0.8] [--top-p 0.9] [--kernel fp32|ternary] [--batch questions.txt] [--seed 1] [--threads 4]";

impl ChatArgs {
    pub fn parse(mut it: impl Iterator<Item = String>) -> Result<ChatArgs, String> {
        let mut a = ChatArgs {
            model: PathBuf::new(),
            tokenizer: PathBuf::from("data/tokenizer.bin"),
            backend: Backend::Scalar,
            temp: 0.8,
            top_p: 0.9,
            kernel: Kernel::Fp32,
            batch: None,
            seed: 1,
            threads: 4,
        };
        let mut have_model = false;
        fn num<T: std::str::FromStr>(f: &str, v: Option<String>) -> Result<T, String> {
            let v = v.ok_or(format!("{f} needs a value"))?;
            v.parse().map_err(|_| format!("bad value {v:?} for {f}"))
        }
        while let Some(f) = it.next() {
            match f.as_str() {
                "--model" => {
                    a.model = it.next().ok_or("--model needs a value")?.into();
                    have_model = true;
                }
                "--tokenizer" => a.tokenizer = it.next().ok_or("--tokenizer needs a value")?.into(),
                "--backend" => a.backend = num(&f, it.next())?,
                "--temp" => a.temp = num(&f, it.next())?,
                "--top-p" => a.top_p = num(&f, it.next())?,
                "--seed" => a.seed = num(&f, it.next())?,
                "--threads" => a.threads = num(&f, it.next())?,
                "--batch" => a.batch = Some(it.next().ok_or("--batch needs a value")?.into()),
                "--kernel" => {
                    a.kernel = match it.next().as_deref() {
                        Some("fp32") => Kernel::Fp32,
                        Some("ternary") => Kernel::Ternary,
                        other => return Err(format!("bad --kernel {other:?} (expected fp32|ternary)")),
                    }
                }
                _ => return Err(format!("unknown flag {f:?}")),
            }
        }
        if !have_model {
            return Err("--model is required".into());
        }
        Ok(a)
    }
}

/// Generate a reply to `prompt` (positions restart at 0; the cache is overwritten), streaming
/// decoded text to `on_text`. Stops at any special token (`<|end|>`, and also `<|user|>` /
/// `<|assistant|>`, so the model cannot run on into a made-up next turn) or when the context
/// is full. Specials are neither printed nor returned. Returns the reply tokens and the
/// seconds spent generating (excluding prefill).
#[allow(clippy::too_many_arguments)]
fn generate(
    p: &Params,
    cfg: &Config,
    args: &ChatArgs,
    cache: &mut KvCache,
    bpe: &Bpe,
    prompt: &[u16],
    rng: &mut Rng,
    on_text: &mut dyn FnMut(&str) -> std::io::Result<()>,
) -> std::io::Result<(Vec<u16>, f64)> {
    let mut stepper = |tok: u16, pos: usize| step(p, cfg, args.backend, cache, tok, pos);
    generate_with(&mut stepper, cfg.ctx, args, bpe, prompt, rng, on_text)
}

/// `generate` over an arbitrary `(token, pos) -> logits` stepper (a stub in tests).
fn generate_with(
    stepper: &mut dyn FnMut(u16, usize) -> Vec<f32>,
    ctx: usize,
    args: &ChatArgs,
    bpe: &Bpe,
    prompt: &[u16],
    rng: &mut Rng,
    on_text: &mut dyn FnMut(&str) -> std::io::Result<()>,
) -> std::io::Result<(Vec<u16>, f64)> {
    let mut logits = Vec::new();
    for (pos, &t) in prompt.iter().enumerate() {
        logits = stepper(t, pos);
    }
    let t0 = Instant::now();
    let (mut pos, mut reply, mut dec) = (prompt.len(), Vec::new(), StreamDecoder::new());
    let specials = [bpe.user(), bpe.assistant(), bpe.end()];
    while pos < ctx {
        let tok = sample(&logits, args.temp, args.top_p, rng);
        if specials.contains(&tok) {
            break;
        }
        reply.push(tok);
        let text = dec.push(&bpe.decode(&[tok]));
        if !text.is_empty() {
            on_text(&text)?;
        }
        logits = stepper(tok, pos);
        pos += 1;
    }
    let tail = dec.flush();
    if !tail.is_empty() {
        on_text(&tail)?;
    }
    Ok((reply, t0.elapsed().as_secs_f64()))
}

fn stats(n: usize, secs: f64) -> String {
    format!("[{n} tok, {:.1} tok/s]", n as f64 / secs.max(1e-9))
}

/// Interactive REPL on `input`/`out` (stats go to `out`), or, with `args.batch`, one fresh
/// question per line of that file printed as `Q: ...` / `A: ...` (stats go to `err`).
/// `/reset` clears history; `/quit` or EOF ends the REPL.
pub fn run_chat(
    args: &ChatArgs,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| e.to_string();
    backend::check_supported(args.backend)?;
    backend::init_threads(args.threads)?;
    if args.kernel == Kernel::Ternary {
        return Err("--kernel ternary is not implemented yet (planned for the ternary-kernel task)".into());
    }
    // `Params::load` already names the file in its format errors; add it for the rest.
    let (cfg, params) = Params::load(&args.model).map_err(|e| {
        if e.kind() == std::io::ErrorKind::InvalidData {
            e.to_string()
        } else {
            format!("{}: {e}", args.model.display())
        }
    })?;
    let bpe = Bpe::load(&args.tokenizer)
        .map_err(|e| format!("{}: {e}", args.tokenizer.display()))?;
    if bpe.vocab.len() != cfg.vocab {
        return Err(format!(
            "tokenizer has {} tokens but the model expects {}",
            bpe.vocab.len(),
            cfg.vocab
        ));
    }
    let mut cache = KvCache::new(&cfg);
    let mut rng = Rng::new(args.seed);
    let mut history: Vec<(Vec<u16>, Vec<u16>)> = Vec::new();

    let questions = match &args.batch {
        Some(path) => Some(std::fs::read_to_string(path).map_err(io)?),
        None => None,
    };
    if let Some(text) = questions {
        for q in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            writeln!(out, "Q: {q}").map_err(io)?;
            write!(out, "A: ").map_err(io)?;
            let prompt = build_prompt(&[], &bpe.encode(q.as_bytes()), &bpe, cfg.ctx, 64);
            let mut emit = |s: &str| out.write_all(s.as_bytes());
            let (reply, secs) =
                generate(&params, &cfg, args, &mut cache, &bpe, &prompt, &mut rng, &mut emit).map_err(io)?;
            writeln!(out).map_err(io)?;
            writeln!(err, "{}", stats(reply.len(), secs)).map_err(io)?;
        }
        return out.flush().map_err(io);
    }

    let mut line = String::new();
    loop {
        write!(out, "> ").map_err(io)?;
        out.flush().map_err(io)?;
        line.clear();
        if input.read_line(&mut line).map_err(io)? == 0 {
            break;
        }
        let q = line.trim();
        match q {
            "" => continue,
            "/quit" => break,
            "/reset" => {
                history.clear();
                writeln!(out, "(history cleared)").map_err(io)?;
                continue;
            }
            _ => {}
        }
        let user = bpe.encode(q.as_bytes());
        let prompt = build_prompt(&history, &user, &bpe, cfg.ctx, 64);
        let mut emit = |s: &str| {
            out.write_all(s.as_bytes())?;
            out.flush()
        };
        let (reply, secs) =
            generate(&params, &cfg, args, &mut cache, &bpe, &prompt, &mut rng, &mut emit).map_err(io)?;
        writeln!(out).map_err(io)?;
        writeln!(out, "{}", stats(reply.len(), secs)).map_err(io)?;
        history.push((user, reply));
    }
    out.flush().map_err(io)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::train_bpe;
    use crate::model::{forward_logits, Acts};

    fn tiny_cfg(ctx: usize) -> Config {
        Config {
            vocab: 300,
            d: 16,
            layers: 2,
            heads: 2,
            ctx,
            softcap: 5.0,
            mtp: 1,
            bitnet: false,
        }
    }

    #[test]
    fn kv_step_matches_full_forward() {
        let cfg = Config {
            vocab: 19,
            d: 8,
            layers: 2,
            heads: 2,
            ctx: 9,
            softcap: 5.0,
            mtp: 1,
            bitnet: false,
        };
        let mut rng = Rng::new(11);
        let mut p = Params::init(&cfg, &mut rng);
        // Larger weights so attention and the cap are not trivially flat.
        p.for_each_tensor_mut(|t| t.iter_mut().for_each(|x| *x *= 10.0));
        let toks: Vec<u16> = (0..cfg.ctx).map(|_| rng.below(cfg.vocab) as u16).collect();
        let mut acts = Acts::new(&cfg, 1, cfg.ctx);
        let full = forward_logits(&p, &cfg, Backend::Scalar, &mut acts, &toks, 1, cfg.ctx).to_vec();
        let mut cache = KvCache::new(&cfg);
        for (pos, &t) in toks.iter().enumerate() {
            let got = step(&p, &cfg, Backend::Scalar, &mut cache, t, pos);
            assert_eq!(got.len(), cfg.vocab);
            let want = &full[pos * cfg.vocab..(pos + 1) * cfg.vocab];
            for (j, (a, b)) in got.iter().zip(want).enumerate() {
                assert!((a - b).abs() < 1e-4, "pos {pos} logit {j}: {a} vs {b}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "ctx")]
    fn step_past_ctx_panics() {
        let cfg = tiny_cfg(4);
        let p = Params::init(&cfg, &mut Rng::new(1));
        let mut cache = KvCache::new(&cfg);
        step(&p, &cfg, Backend::Scalar, &mut cache, 0, 4);
    }

    #[test]
    fn sample_argmax_temp0() {
        let mut rng = Rng::new(1);
        let logits = [0.1, 2.0, -1.0, 1.9];
        assert_eq!(sample(&logits, 0.0, 0.9, &mut rng), 1);
        assert_eq!(sample(&logits, -1.0, 1.0, &mut rng), 1);
    }

    #[test]
    fn top_p_excludes_tail() {
        let mut logits = vec![0.0f32; 50];
        logits[0] = 10.0;
        let mut rng = Rng::new(2);
        for _ in 0..100 {
            assert_eq!(sample(&logits, 1.0, 0.5, &mut rng), 0);
        }
    }

    #[test]
    fn top_p_keeps_threshold_crossing_token() {
        // probs [0.4, 0.4, 0.2], top_p 0.5: keep the first two, never the third.
        let logits = [0.4f32.ln(), 0.4f32.ln(), 0.2f32.ln()];
        let mut rng = Rng::new(4);
        let mut seen = [false; 3];
        for _ in 0..200 {
            seen[sample(&logits, 1.0, 0.5, &mut rng) as usize] = true;
        }
        assert_eq!(seen, [true, true, false]);
    }

    #[test]
    fn generation_stops_on_any_special() {
        let bpe = bpe300();
        let a = args(PathBuf::new(), PathBuf::new());
        for special in [bpe.user(), bpe.assistant(), bpe.end()] {
            let prompt = [bpe.user(), 104, bpe.assistant()];
            let mut calls = 0;
            // Two ordinary tokens ('A'), then the forced special.
            let mut stepper = |_t: u16, _pos: usize| {
                calls += 1;
                let mut l = vec![0.0f32; 300];
                l[if calls <= prompt.len() + 1 { 65 } else { special as usize }] = 10.0;
                l
            };
            let mut text = String::new();
            let mut emit = |s: &str| {
                text.push_str(s);
                Ok(())
            };
            let (reply, _) =
                generate_with(&mut stepper, 128, &a, &bpe, &prompt, &mut Rng::new(1), &mut emit).unwrap();
            assert_eq!(reply, [65, 65], "special {special}");
            assert_eq!(text, "AA", "special {special}");
        }
    }

    #[test]
    fn missing_files_name_their_path() {
        let a = args(PathBuf::from("no_such_dir/model.bin"), PathBuf::from("no_such_dir/tok.bin"));
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let e = run_chat(&a, &mut std::io::empty(), &mut out, &mut err).unwrap_err();
        assert!(e.contains("no_such_dir") && e.contains("model.bin"), "{e}");
        let (dir, model, _) = tiny_files("badtok", false);
        let a = args(model, dir.join("missing_tok.bin"));
        let e = run_chat(&a, &mut std::io::empty(), &mut out, &mut err).unwrap_err();
        assert!(e.contains("missing_tok.bin"), "{e}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn sample_temperature_spreads() {
        // Uniform logits, top_p 1: every token should show up.
        let logits = vec![0.0f32; 4];
        let mut rng = Rng::new(3);
        let mut seen = [false; 4];
        for _ in 0..200 {
            seen[sample(&logits, 1.0, 1.0, &mut rng) as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn stream_decoder_splits_multibyte() {
        let mut d = StreamDecoder::new();
        let bytes = "🙂".as_bytes();
        assert_eq!(bytes.len(), 4);
        assert_eq!(d.push(&bytes[..1]), "");
        assert_eq!(d.push(&bytes[1..2]), "");
        assert_eq!(d.push(&bytes[2..3]), "");
        assert_eq!(d.push(&bytes[3..]), "🙂");
        assert_eq!(d.flush(), "");

        let mut d = StreamDecoder::new();
        let mut s = d.push(&[0xff]);
        s.push_str(&d.flush());
        assert_eq!(s, "\u{FFFD}");

        // Invalid byte in the middle is replaced; surrounding text survives.
        let mut d = StreamDecoder::new();
        let mut s = d.push(b"a\xffb");
        s.push_str(&d.flush());
        assert_eq!(s, "a\u{FFFD}b");

        // Truncated multibyte at the end becomes a replacement on flush.
        let mut d = StreamDecoder::new();
        assert_eq!(d.push(&bytes[..2]), "");
        assert_eq!(d.flush(), "\u{FFFD}");
    }

    fn bpe300() -> Bpe {
        train_bpe(b"hello world, hello there. the quick brown fox", 300)
    }

    /// `<|user|>u<|assistant|>a<|end|>` length.
    fn turn(u: usize, a: usize) -> usize {
        3 + u + a
    }

    #[test]
    fn prompt_truncation_keeps_recent_turns() {
        let bpe = bpe300();
        let (ctx, reserve) = (100, 64);
        let mk = |tag: u16, n: usize| vec![tag; n];
        // Three turns of 23 tokens each. Budget is 36.
        let history = vec![
            (mk(1, 10), mk(2, 10)),
            (mk(3, 10), mk(4, 10)),
            (mk(5, 10), mk(6, 10)),
        ];
        let user = mk(7, 5);
        let got = build_prompt(&history, &user, &bpe, ctx, reserve);
        assert!(got.len() <= ctx - reserve, "len {}", got.len());
        // Only the newest turn fits: 23 + 2 + 5 = 30 <= 36; two turns would be 53.
        assert_eq!(got.len(), turn(10, 10) + 2 + 5);
        let mut want = vec![bpe.user()];
        want.extend(mk(5, 10));
        want.push(bpe.assistant());
        want.extend(mk(6, 10));
        want.push(bpe.end());
        want.push(bpe.user());
        want.extend(mk(7, 5));
        want.push(bpe.assistant());
        assert_eq!(got, want);
        // With a big window nothing is dropped.
        let all = build_prompt(&history, &user, &bpe, 1000, 64);
        assert_eq!(all.len(), 3 * turn(10, 10) + 2 + 5);
    }

    #[test]
    fn long_single_message_truncated() {
        let bpe = bpe300();
        let (ctx, reserve) = (256, 64);
        let user: Vec<u16> = (0..1000).map(|i| (i % 250) as u16).collect();
        let history = vec![(vec![1u16; 5], vec![2u16; 5])];
        let got = build_prompt(&history, &user, &bpe, ctx, reserve);
        assert!(got.len() <= ctx - reserve, "len {}", got.len());
        assert_eq!(*got.last().unwrap(), bpe.assistant());
        assert_eq!(got[0], bpe.user());
        assert_eq!(&got[got.len() - 11..got.len() - 1], &user[user.len() - 10..]);
        // History is gone: only the wrapped message remains.
        assert_eq!(got.len(), ctx - reserve);
    }

    /// Writes a tiny random model + tokenizer into a fresh temp dir.
    fn tiny_files(name: &str, bitnet: bool) -> (PathBuf, PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("rllm_infer_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = Config {
            bitnet,
            ..tiny_cfg(128)
        };
        let p = Params::init(&cfg, &mut Rng::new(9));
        let model = dir.join("model.bin");
        p.save(&cfg, &model).unwrap();
        let tok = dir.join("tok.bin");
        bpe300().save(&tok).unwrap();
        (dir, model, tok)
    }

    fn args(model: PathBuf, tok: PathBuf) -> ChatArgs {
        ChatArgs {
            model,
            tokenizer: tok,
            backend: Backend::Scalar,
            temp: 0.0,
            top_p: 0.9,
            kernel: Kernel::Fp32,
            batch: None,
            seed: 1,
            threads: 4,
        }
    }

    #[test]
    fn batch_mode_is_deterministic_and_independent() {
        let (dir, model, tok) = tiny_files("batch", false);
        let qs = dir.join("qs.txt");
        std::fs::write(&qs, "hello there\n\nwhy is the sky blue?\nhello there\n").unwrap();
        let mut a = args(model, tok);
        a.batch = Some(qs);
        let run = |a: &ChatArgs| {
            let (mut out, mut err) = (Vec::new(), Vec::new());
            run_chat(a, &mut std::io::empty(), &mut out, &mut err).unwrap();
            String::from_utf8_lossy(&out).into_owned()
        };
        let o1 = run(&a);
        assert_eq!(o1, run(&a), "temp 0 must be deterministic");
        let q_lines: Vec<&str> = o1.lines().filter(|l| l.starts_with("Q: ")).collect();
        assert_eq!(
            q_lines,
            ["Q: hello there", "Q: why is the sky blue?", "Q: hello there"]
        );
        // Fresh history per question: identical questions get identical answers.
        let answers: Vec<&str> = o1
            .split("Q: ")
            .skip(1)
            .map(|s| s.split_once("A: ").unwrap().1)
            .collect();
        assert_eq!(answers.len(), 3);
        assert_eq!(answers[0].split_once('\n').unwrap().0, answers[2].split_once('\n').unwrap().0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn repl_streams_resets_and_reports_speed() {
        let (dir, model, tok) = tiny_files("repl", false);
        let a = args(model, tok);
        let mut input = std::io::Cursor::new(b"hello\n/reset\nhi\n/quit\nignored\n".to_vec());
        let (mut out, mut err) = (Vec::new(), Vec::new());
        run_chat(&a, &mut input, &mut out, &mut err).unwrap();
        let out = String::from_utf8_lossy(&out).into_owned();
        assert_eq!(out.matches(" tok/s]").count(), 2, "{out}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ternary_kernel_rejected() {
        for bitnet in [false, true] {
            let (dir, model, tok) = tiny_files(if bitnet { "t1" } else { "t0" }, bitnet);
            let mut a = args(model, tok);
            a.kernel = Kernel::Ternary;
            let (mut out, mut err) = (Vec::new(), Vec::new());
            let e = run_chat(&a, &mut std::io::empty(), &mut out, &mut err).unwrap_err();
            assert!(e.contains("ternary"), "{e}");
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
