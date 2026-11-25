# Rust CPU LLM (60-second speedrun) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A from-scratch Rust GPT (hand-written forward/backward, Muon, BitNet b1.58, AVX-512/VNNI/asm kernels) that trains on Reddit Q&A data within a 60-second wall-clock budget on an i7-1165G7 and chats in a terminal.

**Architecture:** Cargo workspace with two crates. `rllm` holds tokenizer, model, ops, kernels, optimizers, training, and inference, with zero external dependencies. `rllm-prep` holds data extraction and tokenization tools, and may use `zstd` and `serde_json`. Every op has a scalar reference. Faster kernels sit behind a `Backend` enum, so every engine stage can be compared under the same 60-second budget. Milestone 1 (Tasks 1–9) gives an end-to-end scalar result first. Later tasks add speed and the experiments.

**Tech Stack:** Rust 1.99.1 stable, `std::arch` x86_64 intrinsics (AVX-512F/BW/VL/VNNI), `core::arch::asm!`, `std::thread`. In prep only: `zstd`, `serde`, `serde_json`.

**Spec:** `docs/superpowers/specs/2026-11-25-rust-cpu-llm-design.md`

## Global Constraints

- The `rllm` crate has **no dependencies** (`[dependencies]` empty; `[dev-dependencies]` empty too). Never use ndarray, candle, burn, matrixmultiply, BLAS, or rayon anywhere.
- Build for this laptop only. The workspace `.cargo/config.toml` sets `rustflags = ["-C", "target-cpu=native"]`. Scalar code serves as a correctness reference only. Do not write tuned AVX2 kernels.
- Every fast kernel needs a test that compares it to the scalar reference, including on odd shapes.
- Training budget: 60 s wall clock per run (pretrain 45 s + SFT 15 s by default). Data loading, model init, and final evaluation do not count toward the budget. The clock starts at the first train step.
- Baseline model: d_model 128, 4 layers, 4 heads (head_dim 32), context 256, vocab 2048 (≈1.05M params). Pre-norm RMSNorm, RoPE (base 10000), ReLU² MLP (hidden 4×d), no biases, tied embedding/unembedding, logit softcap 30.
- Special tokens are the last three ids: `<|user|>`=2045, `<|assistant|>`=2046, `<|end|>`=2047 (when vocab = 2048; in general V-3, V-2, V-1).
- SFT format: `<|user|>{q}<|assistant|>{a}<|end|>`. Loss applies only to tokens after `<|assistant|>`, up to and including `<|end|>`.
- Data comes from Pushshift subreddit dumps (r/explainlikeimfive, r/NoStupidQuestions, r/AskScience). SFT is mostly ELI5. AskScience is used for pretraining only. Do not publish models or processed data.
- Main metric: validation loss after 60 s. Each run appends one row to `runs/results.csv`.
- Windows 11. Commands below are shown for PowerShell.

## Review Focus

1. **Any byte sequence (emoji, Korean, invalid UTF-8)** through the tokenizer: encode→decode must round-trip exactly with no panic. → Task 1 `roundtrip_arbitrary_bytes`.
2. **Generation that cuts a multibyte character between tokens**: the streaming decoder holds incomplete bytes and prints invalid sequences with lossy replacement, never panics. → Task 9 `stream_decoder_splits_multibyte`.
3. **Chat history or a single message longer than the 256-token context**: drop the oldest turns first. If one message is still too long, keep its last tokens, and always leave ≥64 slots for generation. → Task 9 `prompt_truncation_keeps_recent_turns`, `long_single_message_truncated`.
4. **Slow engine where one step is near or over the remaining budget** (the scalar stage takes seconds per step): never start a step whose estimated duration would overrun the phase budget, but run at least one step per phase. → Task 8 `budget_stops_before_overrun`, `at_least_one_step_per_phase`.
5. **Matrix sizes that are not multiples of the SIMD width or block size**: tail handling must be correct. → Tasks 10/11/12/15/16 run odd shapes `(n,k,m) = (7,33,19)` and `(1,1,1)`.

---

## Milestone 1 — End-to-end scalar result

### Task 1: Workspace scaffold + byte-level BPE tokenizer

**Files:**
- Create: `Cargo.toml` (workspace: members `rllm`, `rllm-prep`), `.cargo/config.toml`, `.gitignore` (`target/`, `data/`, `runs/`)
- Create: `rllm/Cargo.toml`, `rllm/src/lib.rs`, `rllm/src/bpe.rs`
- Create: `rllm-prep/Cargo.toml`, `rllm-prep/src/lib.rs` (empty for now)
- Test: `rllm/src/bpe.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Produces:
  - `pub struct Bpe { pub merges: Vec<(u32,u32)>, pub vocab: Vec<Vec<u8>> }`, with `vocab.len() == vocab_size` (ids 0..256 = bytes, then merges, then 3 specials).
  - `pub fn train_bpe(text: &[u8], vocab_size: usize) -> Bpe`
  - `impl Bpe { pub fn encode(&self, text: &[u8]) -> Vec<u16>; pub fn decode(&self, ids: &[u16]) -> Vec<u8>; pub fn user(&self)->u16; pub fn assistant(&self)->u16; pub fn end(&self)->u16; pub fn save(&self, path:&Path)->io::Result<()>; pub fn load(path:&Path)->io::Result<Bpe> }`
  - The special tokens are never produced by `encode` (plain text such as `"<|end|>"` encodes as ordinary bytes) and decode to their literal strings.

- [ ] **Step 1:** Run `git init`. Create the workspace files and `.cargo/config.toml` with `[build] rustflags = ["-C","target-cpu=native"]`. Add a `[profile.release]` with `debug = 1` (for profiling) and `codegen-units = 1`.
- [ ] **Step 2: Write failing tests**

```rust
#[test] fn vocab_layout() {
    let b = train_bpe(b"hello hello hello world world", 260 + 3);
    assert_eq!(b.vocab.len(), 263);
    assert_eq!(b.end(), 262); assert_eq!(b.assistant(), 261); assert_eq!(b.user(), 260);
}
#[test] fn merges_shorten_text() {
    let t = b"the cat the cat the cat the cat".repeat(10);
    let b = train_bpe(&t, 300);
    assert!(b.encode(&t).len() < t.len() / 2);
}
#[test] fn roundtrip_arbitrary_bytes() {
    let b = train_bpe(b"abc abc abc", 270);
    for s in [&b"\xff\xfe\x00"[..], "안녕 🙂".as_bytes(), b"<|end|>", b""] {
        assert_eq!(b.decode(&b.encode(s)), s);
    }
    assert!(!b.encode(b"<|end|>").contains(&b.end()));
}
#[test] fn save_load_identical() { /* train, save to temp dir, load, assert encode output equal on a sample */ }
```

- [ ] **Step 3:** Run `cargo test -p rllm bpe`. Expect FAIL (not defined).
- [ ] **Step 4:** Implement standard byte-level BPE. Pre-split text on whitespace boundaries (a leading space stays attached to the word), count word frequencies, and repeatedly merge the most frequent pair. Break ties by smallest `(a,b)` so results are deterministic. Encode by applying merges in rank order within each pre-split chunk. Save format: magic `b"RBPE"`, u32 vocab_size, u32 n_merges, then pairs as u32 LE.
- [ ] **Step 5:** Run `cargo test -p rllm bpe`. Expect PASS.
- [ ] **Step 6:** Commit: `feat: workspace scaffold and byte-level BPE tokenizer`

### Task 2: Pushshift extraction (rllm-prep)

**Files:**
- Create: `rllm-prep/src/extract.rs`, `rllm-prep/src/bin/extract.rs`
- Test: `rllm-prep/tests/extract.rs` (fixtures are built inside the test by writing NDJSON lines and compressing them with `zstd::encode_all`)

**Interfaces:**
- Produces:
  - `pub fn open_zst_ndjson(path:&Path) -> io::Result<impl Iterator<Item=serde_json::Value>>`. The decoder must call `window_log_max(31)`, because Pushshift dumps use long-window zstd.
  - `pub fn clean_text(s:&str) -> Option<String>`. Returns `None` for `[deleted]`, `[removed]`, or empty text after cleaning. Converts `[text](url)` to `text`, removes bare `http(s)://…` tokens, and collapses runs of whitespace.
  - `pub struct SftPair { pub q: String, pub a: String }`
  - `pub fn split_is_val(submission_id:&str) -> bool`. A stable FNV-1a hash of the id, with val when `hash % 50 == 0` (2%).
  - CLI: `extract --subs <dir> --out data/ --max-pretrain-mb 40`. Reads `<sub>_submissions.zst` and `<sub>_comments.zst` for the three subreddits. Writes `data/pretrain_{train,val}.txt` (documents separated by `\n\x1e\n`) and `data/sft_{train,val}.jsonl` (`{"q":..,"a":..}`).
- SFT rules (pinned):
  - Question = `title` + `"\n"` + `selftext`, cleaned (selftext may be empty).
  - Answer = the highest-score top-level comment (`parent_id == link_id`) with score ≥ 10, author not `AutoModerator`, and cleaned length 20–800 chars.
  - r/explainlikeimfive provides all pairs that qualify. r/NoStupidQuestions provides only answers ≤ 300 chars and is capped at 20% of total SFT pairs. r/AskScience provides no SFT pairs.
- Pretrain: every cleaned submission (title+selftext) and comment body from all three subreddits, round-robin across subreddits, stopping at `--max-pretrain-mb`.

- [ ] **Step 1: Write failing tests**: `clean_text_rules` (each rule above), `deleted_answer_skipped`, `best_top_level_comment_chosen` (fixture with a reply comment of higher score than the top-level one: the top-level one is chosen), `askscience_not_in_sft`, `nsq_cap_20_percent`, `val_split_by_submission` (a question and its answer always land in the same split), `reads_long_window_zst` (compress the fixture with level 3 and window log 31 via `zstd::stream::Encoder::long_distance_matching` + `window_log(31)`).
- [ ] **Step 2:** Run `cargo test -p rllm-prep`. Expect FAIL.
- [ ] **Step 3:** Implement. Load the qualifying submissions of a subreddit into a `HashMap<id, question>`, then stream its comments and keep the best answer per `link_id` (strip the `t3_` prefix).
- [ ] **Step 4:** Run `cargo test -p rllm-prep`. Expect PASS.
- [ ] **Step 5 (manual, user):** The user downloads the six `.zst` files from the Pushshift subreddit dumps on Academic Torrents into a folder of their choice. Do not download on their behalf without explicit permission. Then run `cargo run --release -p rllm-prep --bin extract -- --subs <folder> --out data`. Expect pretrain_train.txt ≈ 40 MB and an SFT pair count printed (≥ 20k expected).
- [ ] **Step 6:** Commit: `feat: pushshift extraction with SFT pairing and filters`

### Task 3: Tokenized data files + batchers

**Files:**
- Create: `rllm-prep/src/bin/tokenize.rs`, `rllm/src/data.rs`, `rllm/src/rng.rs`
- Test: in-module tests in `data.rs`, `rng.rs`

**Interfaces:**
- Consumes: `Bpe` (Task 1), the prep outputs (Task 2).
- Produces:
  - `tokenize --data data/ --vocab 2048 --bpe-train-mb 10`. Trains BPE on the first 10 MB of pretrain_train.txt and writes `data/tokenizer.bin`.
  - It writes `data/pretrain_{train,val}.bin`: raw u16 LE tokens, with documents joined by `<|end|>`.
  - It writes `data/sft_{train,val}.bin`: a sequence of records `u16 len` + `len × u16` tokens, where each record is `user q assistant a end`. A record whose length > 257 (context+1) is **dropped**. The count of dropped records is printed.
  - `pub struct Rng(u64)` (SplitMix64): `new(seed)`, `next_u64`, `below(n:usize)->usize`, `normal()->f32` (Box–Muller)
  - `pub struct PretrainBatcher { tokens: Vec<u16>, .. }`: `load(path)`, `fn next(&mut self, rng:&mut Rng, b:usize, t:usize, extra:usize) -> Batch`. Draws random windows of length `t+1+extra`.
  - `pub struct SftBatcher`: `load(path)`, same `next`. Each row holds one sample, right-padded with `<|end|>`, and the mask is 0 on padding.
  - `pub struct Batch { pub b: usize, pub t: usize, pub inputs: Vec<u16> /*b*t*/, pub targets: Vec<u16> /*b*(t+extra), row r target j = token at r, j+1*/, pub mask: Vec<f32> /*b*t, for main targets*/ }`. For pretraining, mask is all 1.
  - `pub fn sft_mask(tokens:&[u16], assistant:u16, end:u16) -> Vec<f32>`. For target index j (predicting tokens[j+1]), mask = 1 iff some `assistant` occurs at index ≤ j and tokens[j+1] lies within that answer, up to and including its `<|end|>`.
- [ ] **Step 1: Write failing tests**: `rng_deterministic`, `pretrain_targets_shifted` (`targets[j] == inputs[j+1]` within a row), `sft_mask_only_answer` (for `[U,q1,q2,A,a1,a2,E]`, targets `[q1,q2,A,a1,a2,E]` get mask `[0,0,0,1,1,1]`), `sft_padding_masked`, `overlong_record_dropped` (a tokenize helper fn `encode_sft_record(...) -> Option<Vec<u16>>` returns None when len > 257), `empty_answer_dropped` (an answer string that is empty returns None).
- [ ] **Step 2:** Run `cargo test -p rllm data rng`. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expect PASS. Then run `cargo run --release -p rllm-prep --bin tokenize -- --data data`. Expect the .bin files to be written, a tokens count printed (pretrain ≈ 10M tokens), and the dropped count printed.
- [ ] **Step 5:** Commit: `feat: tokenized data files and seeded batchers`

### Task 4: Scalar ops — linear, RMSNorm, ReLU², softcap, cross-entropy

**Files:**
- Create: `rllm/src/ops/mod.rs`, `rllm/src/ops/scalar.rs`, `rllm/src/gradcheck.rs` (test helper, `#[cfg(test)]`)
- Test: in-module tests

**Interfaces:**
- Produces (all row-major, `n` rows; weights `w` are `[out, in]`, so y = x·Wᵀ):
  - `matmul_fwd(out:&mut[f32], x:&[f32], w:&[f32], n:usize, k:usize, m:usize)`: out[n×m] = x[n×k]·w[m×k]ᵀ (overwrites).
  - `matmul_bwd(dx:&mut[f32], dw:&mut[f32], dout:&[f32], x:&[f32], w:&[f32], n,k,m)`: dx += dout·w, dw += doutᵀ·x (**accumulates**).
  - `rmsnorm_fwd(out, rstd:&mut[f32], x, g:&[f32], n, c)`. Uses eps 1e-6 and stores rstd per row.
  - `rmsnorm_bwd(dx, dg, dout, x, g, rstd, n, c)` (accumulates).
  - `relu2_fwd(out, x)`, `relu2_bwd(dx, dout, x)` (accumulates).
  - `softcap_fwd(out, x, cap:f32)`, `softcap_bwd(dx, dout, out_capped, cap)`. The backward uses `1 - (y/cap)²`.
  - `xent_fwd_bwd(dlogits:&mut[f32], logits:&[f32], targets:&[u16], mask:&[f32], n, v) -> f32`. Returns the mean loss over Σmask. Writes dlogits = mask·(softmax − onehot)/Σmask (overwrites). Σmask=0 gives loss 0 and dlogits 0.
- `gradcheck::check(f: impl Fn(&[f32])->f32, x:&[f32], analytic:&[f32])`. Central difference with eps 1e-2. Asserts `|a−n| ≤ 1e-3 + 2e-2·max(|a|,|n|)` per element.

- [ ] **Step 1: Write failing tests**: `matmul_known_values` (2×3 · 3×2, hand-computed), and gradcheck tests for each op with a random input from `Rng` seed 1 (shapes n=3, k=5, m=4, c=8, v=11). Also `xent_masked_rows_ignored`, `xent_all_masked_is_zero`, `matmul_bwd_accumulates` (calling twice doubles dw).
- [ ] **Step 2:** Run `cargo test -p rllm ops`. Expect FAIL.
- [ ] **Step 3:** Implement the scalar versions. They are plain loops: no blocking, no unsafe.
- [ ] **Step 4:** Run the tests. Expect PASS.
- [ ] **Step 5:** Commit: `feat: scalar reference ops with gradient checks`

### Task 5: Scalar attention with RoPE

**Files:**
- Modify: `rllm/src/ops/scalar.rs`
- Test: in-module

**Interfaces:**
- Produces:
  - `rope_fwd(x:&mut[f32], b, t, h, d)` operates in place on `[b,t,h,d]`, rotating pairs `(2i, 2i+1)` with θ_i = 10000^(−2i/d), angle = pos·θ_i. `rope_bwd(dx:&mut[f32], ...)` applies the inverse rotation in place.
  - `attn_fwd(out:&mut[f32], att:&mut[f32], qkv:&[f32], b, t, c, h)`: qkv is `[b,t,3c]` (q|k|v), out is `[b,t,c]`, att stores the probabilities `[b,h,t,t]`. Causal, with scale 1/√(c/h).
  - `attn_bwd(dqkv:&mut[f32], dout, qkv, att, b, t, c, h)` (accumulates).
  - RoPE is applied to the q and k slices of qkv before attention. This is the model's job (Task 6); the ops above stay separate.
- [ ] **Step 1: Write failing tests**: `rope_preserves_norm`, `rope_bwd_inverts`, `attn_causal` (changing a token at position 3 leaves outputs at positions 0–2 unchanged), `attn_gradcheck` (b=2, t=5, c=8, h=2).
- [ ] **Step 2–4:** fail → implement → pass (`cargo test -p rllm attn rope`).
- [ ] **Step 5:** Commit: `feat: scalar causal attention and RoPE`

### Task 6: Model assembly — params, forward, backward, save/load

**Files:**
- Create: `rllm/src/model.rs`, `rllm/src/backend.rs`
- Test: in-module

**Interfaces:**
- Consumes: the ops (Tasks 4–5), `Rng`, `Batch`.
- Produces:
  - `pub enum Backend { Scalar, Blocked, Avx512, Avx512Mt, Asm }`. Each variant's ops live in their own module. Until Task 10, `backend.rs` routes every variant to scalar. Model code calls `backend::matmul_fwd(be, ...)` and the matching backward with identical signatures. Only matmul and attention are dispatched; elementwise ops stay scalar until profiling says otherwise.
  - `pub struct Config { pub vocab:usize, pub d:usize, pub layers:usize, pub heads:usize, pub ctx:usize, pub softcap:f32, pub mtp:usize, pub bitnet:bool }`. `Config::baseline()` = {2048,128,4,4,256,30.0,0,false}.
  - `pub struct Mat { pub rows:usize, pub cols:usize, pub data:Vec<f32> }`
  - `pub struct Layer { pub attn_norm:Vec<f32>, pub wqkv:Mat /*3d×d*/, pub wo:Mat /*d×d*/, pub mlp_norm:Vec<f32>, pub w_up:Mat /*4d×d*/, pub w_down:Mat /*d×4d*/ }`
  - `pub struct Params { pub wte:Mat /*V×d*/, pub layers:Vec<Layer>, pub final_norm:Vec<f32>, pub mtp:Vec<Mat> /*d×d each*/ }`. Grads have the same type (zeroable via `Params::zeros_like`).
  - `Params::init(cfg, rng)`: norms = 1. Matrices are normal(0, 0.02), except `wo` and `w_down`, which are 0.02/√(2·layers). `num_params()`.
  - `pub struct Acts` is preallocated for (b, t). It keeps everything backward needs.
  - `pub fn forward(p:&Params, cfg:&Config, be:Backend, a:&mut Acts, batch:&Batch) -> f32`. Returns the loss: main loss plus, when mtp>0, `0.5 * mean_k(L_k)`, where head k uses `final_norm(h)·M_kᵀ → tied logits → softcap` against targets offset by k (targets row slice `[k .. k+t]`).
  - `pub fn backward(p:&Params, cfg:&Config, be:Backend, a:&mut Acts, batch:&Batch, g:&mut Params)`
  - `pub fn forward_logits(...)`: used by inference in Task 9.
  - `Params::save(&self, cfg, path)` / `Params::load(path) -> (Config, Params)`. Format: magic `b"RLLM"`, u32 version=1, the Config fields as u32 (softcap as f32 bits, bitnet as 0/1), then all tensors as f32 LE in the declaration order above.
- Block (pre-norm): `x += wo·attn(rope(qkv(rmsnorm(x))))`; `x += w_down·relu2(w_up·rmsnorm(x))`. Final: `rmsnorm → x·wteᵀ → softcap(30)`.
- [ ] **Step 1: Write failing tests**:
  - `param_count_baseline` (assert 1_049_728 = 2048·128 + 4·(128+3·128²+128²+128+2·4·128²) + 128).
  - `model_gradcheck_tiny` (cfg {vocab 19, d 8, layers 2, heads 2, ctx 6, mtp 1}; check ≥ 30 random parameter coordinates spread across all tensors).
  - `overfit_tiny_batch` (baseline-shaped but d=32; one fixed batch b=2, t=32; 300 steps of plain SGD lr 0.1 bring the loss below 0.1).
  - `save_load_roundtrip`.
  - `initial_loss_near_ln_v` (baseline, |loss − ln 2048| < 0.5).
- [ ] **Step 2–4:** fail → implement → pass (`cargo test -p rllm model`). The overfit test must run with `--release` if it is slow: mark it `#[ignore]` and run `cargo test --release -p rllm model -- --include-ignored`.
- [ ] **Step 5:** Commit: `feat: GPT model forward/backward and checkpoint format`

### Task 7: Optimizers — AdamW and Muon

**Files:**
- Create: `rllm/src/optim.rs`
- Test: in-module

**Interfaces:**
- Consumes: `Params`, `Mat`.
- Produces:
  - `pub struct AdamW { pub lr:f32, b1:0.9, b2:0.95, eps:1e-8, wd:0.0, m, v, t }`, with `step(&mut self, params:&mut [f32], grads:&[f32])` working per tensor.
  - `pub struct Muon { pub lr:f32, momentum:0.95 /*nesterov*/, ns_steps:5, buf }`, with `step(&mut self, w:&mut Mat, g:&Mat)`.
  - `pub fn newton_schulz5(g:&Mat) -> Mat`. Normalize X = G/(‖G‖_F + 1e-7). If rows > cols, transpose. Repeat 5 times: `A = X·Xᵀ; B = b·A + c·A·A; X = a·X + B·X` with (a,b,c) = (3.4445, −4.7750, 2.0315). Transpose back. The update is scaled by `max(1, rows/cols)^0.5`.
  - `pub struct Optimizer { adam: AdamW, muon: Muon }`, with `step(&mut self, p:&mut Params, g:&Params, lr_mult:f32)`. **Muon** covers every `Layer` matrix (wqkv, wo, w_up, w_down) and the `mtp` mats. **AdamW** covers `wte` and all norm gains. Default base lrs: muon 0.02, adam 0.008 (to be tuned later).
  - `pub fn zero_grads(g:&mut Params)`.
- [ ] **Step 1: Write failing tests**: `ns5_near_orthogonal` (random 32×16: singular values of the output fall in [0.6, 1.3], checked via XᵀX diagonal ≈ 1 ± 0.4 and off-diagonal |·| < 0.3), `adamw_minimizes_quadratic`, `muon_reduces_loss_tiny_model` (the tiny cfg from Task 6 reaches a lower loss after 50 steps than at the start).
- [ ] **Step 2–4:** fail → implement (matmuls inside NS use the scalar `matmul_fwd`) → pass.
- [ ] **Step 5:** Commit: `feat: AdamW and Muon optimizers`

### Task 8: Training binary with 60-second budget

**Files:**
- Create: `rllm/src/train.rs`, `rllm/src/bin/train.rs`
- Test: in-module (`train.rs`)

**Interfaces:**
- Consumes: Tasks 3, 6, 7.
- Produces:
  - `pub trait Clock { fn now(&self) -> f64 /*seconds*/; }` with a real implementation (`std::time::Instant`) and a test `FakeClock`.
  - `pub fn lr_mult(frac:f64) -> f32`, where frac = elapsed/phase_budget. It is a linear warmup over 0–0.05, then 1.0, then linear decay to 0.1 over 0.7–1.0.
  - `pub fn run_phase(clock:&dyn Clock, budget_s:f64, mut step: impl FnMut(f32 /*lr_mult*/) -> f32 /*loss*/) -> PhaseStats { steps, last_loss }`. Before each step, estimate `dur` = duration of the previous step (for the first step, ∞ is allowed). Stop if `elapsed + dur > budget` **and** steps ≥ 1.
  - The CLI: `train --data data --backend scalar|blocked|avx512|avx512mt|asm --budget 60 --pretrain-frac 0.75 --batch 16 --seed 42 [--mtp N] [--bitnet] --name <run>`.
    - It loads tokenizer.bin and checks its vocab against `cfg.vocab`.
    - It runs the pretrain phase (`budget*frac`), then the SFT phase on the same weights with fresh optimizer moments.
    - It then evaluates fixed validation sets, excluded from the budget: the pretrain val uses the first 32 windows of 257 tokens of pretrain_val.bin, and the SFT val uses the first 256 records of sft_val.bin.
    - It saves `runs/<run>/model.bin` and appends to `runs/results.csv`. The header is written if the file is missing: `timestamp,name,backend,bitnet,mtp,params,pretrain_steps,pretrain_tokens,sft_steps,sft_tokens,train_tok_per_s,wall_s,val_loss_pretrain,val_loss_sft`.
    - Every step it prints a line: `phase step loss tok/s elapsed`.
- [ ] **Step 1: Write failing tests**:
  - `lr_mult_shape` (0→0, 0.05→1, 0.5→1, 1.0→0.1).
  - `budget_stops_before_overrun`: FakeClock where each step advances 7 s, budget 20 → exactly 2 steps (at elapsed 14, the next step estimate of 7 gives 21 > 20, so it stops).
  - `at_least_one_step_per_phase`: step takes 30 s, budget 20 → 1 step.
  - `results_csv_header_once`: two appends give 1 header and 2 rows.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5: First real run:** `cargo run --release -p rllm --bin train -- --data data --backend scalar --name scalar-baseline`. Expect it to finish in ≤ 62 s of train time, write a row to `runs/results.csv`, and show val losses below the initial ln(2048) ≈ 7.6. Record the numbers in the commit message.
- [ ] **Step 6:** Commit: `feat: 60s budgeted training loop with results log`

### Task 9: Inference, KV cache, chat CLI

**Files:**
- Create: `rllm/src/infer.rs`, `rllm/src/bin/chat.rs`
- Test: in-module

**Interfaces:**
- Consumes: `Params`, `Config`, `Bpe`, `Backend`.
- Produces:
  - `pub struct KvCache` (per layer K, V `[ctx, d]` after RoPE), with `pub fn step(p,cfg,be,cache:&mut KvCache, token:u16, pos:usize) -> Vec<f32> /*logits after softcap*/`.
  - `pub fn sample(logits:&[f32], temp:f32, top_p:f32, rng:&mut Rng) -> u16`. temp ≤ 0 means argmax.
  - `pub struct StreamDecoder`, with `push(&mut self, bytes:&[u8]) -> String` (it emits only complete UTF-8, keeps a partial tail of up to 3 bytes, and replaces invalid sequences lossily) and `flush() -> String`.
  - `pub fn build_prompt(history:&[(Vec<u16>,Vec<u16>)], user:&[u16], bpe:&Bpe, ctx:usize, reserve:usize /*64*/) -> Vec<u16>`. It drops the oldest turns until `len ≤ ctx − reserve`. If it is still too long, it keeps the last `ctx − reserve − 2` tokens of the user message, wrapped as `<|user|>…<|assistant|>`.
  - The CLI: `chat --model runs/<run>/model.bin --tokenizer data/tokenizer.bin [--backend ...] [--temp 0.8] [--top-p 0.9] [--kernel fp32|ternary]`. It runs a REPL; generation stops at `<|end|>` or when the context is full, then it prints `[N tok, X tok/s]`. `/reset` clears the history. `--kernel ternary` is rejected unless `cfg.bitnet` (Task 15 implements it).
- [ ] **Step 1: Write failing tests**:
  - `kv_step_matches_full_forward`: tiny cfg, logits at each position from `step` equal `forward_logits` within 1e-4.
  - `sample_argmax_temp0`.
  - `top_p_excludes_tail`: logits [10, 0, 0, …], top_p 0.5 → always 0 over 100 draws.
  - `stream_decoder_splits_multibyte`: "🙂" bytes pushed one byte at a time give "" "" "" "🙂"; `[0xff]` then flush gives "\u{FFFD}".
  - `prompt_truncation_keeps_recent_turns`.
  - `long_single_message_truncated` (the result length is ≤ ctx − 64, it ends with the assistant token, and it contains the last user tokens).
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Manual check: `cargo run --release -p rllm --bin chat -- --model runs/scalar-baseline/model.bin --tokenizer data/tokenizer.bin`. Ask "why is the sky blue?". Expect some text and a `[.. tok/s]` line with no panic.
- [ ] **Step 6:** Commit: `feat: KV-cache inference and chat CLI` — **Milestone 1 done: first 60 s result.**

---

## Milestone 2 — Speed ladder

### Task 10: Benchmark harness + cache-blocked kernels

**Files:**
- Create: `rllm/src/bin/bench.rs`, `rllm/src/ops/blocked.rs`
- Modify: `rllm/src/backend.rs` (route `Blocked`)
- Create: `bench/results.md` (appended by the bench binary)

**Interfaces:**
- Produces:
  - `blocked::matmul_fwd` / `matmul_bwd` with the same signatures as scalar.
  - `bench --backend all --threads 4` measures:
    - (a) measured peak: an AVX-512 FMA loop with 8 independent accumulators per thread, in GFLOPS, for 1 thread and N threads;
    - (b) matmul fwd and bwd GFLOPS for the model shapes `(n,k,m)`: (4096,128,384), (4096,128,128), (4096,128,512), (4096,512,128), (4096,128,2048);
    - (c) one full train step (b=16, t=256) as tok/s.
    It does 3 warm-up runs, then takes the median of 5. It appends a dated markdown table with a "% of measured peak" column.
- [ ] **Step 1: Write failing tests**: `blocked_matches_scalar` over shapes {(1,1,1), (7,33,19), (64,128,384), (300,128,2048)}. Tolerance: `max|Δ| ≤ 1e-4·k` relative to the largest |value|.
- [ ] **Step 2–4:** fail → implement (loop reordering + k/m tiling for L1/L2; tile sizes are the builder's call) → pass.
- [ ] **Step 5:** Run `cargo run --release -p rllm --bin bench -- --backend scalar,blocked`. Then run `train --backend blocked --name blocked`. Both rows are recorded.
- [ ] **Step 6:** Commit: `perf: benchmark harness and cache-blocked matmul`

### Task 11: AVX-512 kernels

**Files:**
- Create: `rllm/src/ops/avx512.rs`
- Modify: `rllm/src/backend.rs`

**Interfaces:**
- Produces:
  - `avx512::matmul_fwd` / `matmul_bwd` (same signatures). They use `#[target_feature(enable="avx512f")]` unsafe inner fns behind safe wrappers. They pack W into panels and use a register-blocked microkernel (shape is the builder's call; use the 32 zmm registers). Masked loads (`_mm512_maskz_loadu_ps`) handle tails.
  - `avx512::attn_fwd` / `attn_bwd` vectorize the dot products over head_dim 32 (one zmm = 16 lanes, 2 per head).
  - At startup, `backend.rs` asserts `is_x86_feature_detected!("avx512f")` when Avx512* or Asm is chosen, and prints a clear error otherwise.
- [ ] **Step 1: Write failing tests**: `avx512_matmul_matches_scalar` and `avx512_attn_matches_scalar` on the same shape set as Task 10, plus attention shapes (b=1, t=1), (b=2, t=7), (b=2, t=256).
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Bench and run a 60 s train (`--backend avx512 --name avx512`). Record the results.
- [ ] **Step 6:** Commit: `perf: AVX-512 matmul and attention kernels`

### Task 12: Thread pool + multithreaded kernels

**Files:**
- Create: `rllm/src/pool.rs`
- Modify: `rllm/src/ops/avx512.rs` (add `_mt` variants), `rllm/src/backend.rs`

**Interfaces:**
- Produces:
  - `pub struct ThreadPool`, with `new(n:usize)` (persistent workers parked on Condvar) and `fn run(&self, n_tasks:usize, f:&(dyn Fn(usize) + Sync))`, which blocks until all tasks are done. A panic in a task propagates to the caller.
  - A global pool is created lazily with `--threads` (default 4, since AVX-512 rarely benefits from hyperthreads; the bench compares 4 vs 8).
  - `Avx512Mt`: matmul splits fwd over the n rows, splits bwd dW over m rows (no reduction races), and splits attention over b·h.
- [ ] **Step 1: Write failing tests**: `pool_runs_all_tasks_once` (an atomic counter array where each index == 1), `pool_reusable_1000_runs`, `pool_propagates_panic` (`#[should_panic]`), `mt_matches_scalar` on the shape sets.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Run the bench with `--threads 4,8`. Run a 60 s train with `--backend avx512mt --name avx512mt`. Then use `bench`'s per-op timing (add `--profile-step` to print time per op type) to check whether elementwise ops now dominate. If any elementwise op takes > 10% of step time, vectorize it in `avx512.rs` within this task.
- [ ] **Step 6:** Commit: `perf: thread pool and multithreaded AVX-512 kernels`

---

## Milestone 3 — Experiments

### Task 13: Multi-token prediction runs

**Files:**
- Modify: none (MTP exists from Task 6). Create `runs/README.md` with the experiment log template.

- [ ] **Step 1:** Run three 60 s trains with `--backend avx512mt` and seed 42: `--mtp 0`, `--mtp 1`, `--mtp 2`. Then rerun each with seed 43 for noise.
- [ ] **Step 2:** Run a model-size sweep at mtp 0: d ∈ {96, 128, 192}, layers ∈ {3, 4, 6} (add `--d` and `--layers` CLI flags that override `Config::baseline()`). Find the best val_loss_sft within 60 s.
- [ ] **Step 2b:** Run a vocab sweep at the best size: vocab ∈ {1024, 4096}. Retokenize into `data-v1024/` and `data-v4096/` with `tokenize --vocab N --data <dir>`, after copying the prep outputs there, then add `--vocab` to train. Compare val loss in **bits per byte** rather than raw loss, since the vocabularies differ (bpb = loss·tokens/(ln2·bytes) on the same val text). Add a `val_bpb` column to results.csv in this step, and backfill it for the earlier rows by re-evaluating their saved models.
- [ ] **Step 3:** Write `runs/README.md`: a table per experiment and the chosen default config (update `Config::baseline()` if a different size wins by more than the seed noise).
- [ ] **Step 4:** Commit: `exp: MTP and model-size sweeps under 60s budget`

### Task 14: BitNet b1.58 training

**Files:**
- Create: `rllm/src/bitnet.rs`
- Modify: `rllm/src/model.rs` (when `cfg.bitnet`, the block matmuls wqkv/wo/w_up/w_down go through BitLinear; wte and mtp stay fp32)

**Interfaces:**
- Produces:
  - `pub fn quant_weights(w:&[f32], out:&mut[f32]) -> f32 /*gamma*/`. γ = mean|w| + 1e-5, and out = clamp(round(w/γ), −1, 1)·γ.
  - `pub fn quant_acts(x:&[f32], out:&mut[f32], n, k)`. Per row, s = 127/max(|x|, 1e-5), and out = clamp(round(x·s), −128, 127)/s.
  - BitLinear fwd: y = quant_acts(x)·quant_weights(W)ᵀ, using the same backend matmul. bwd (STE): dx = dy·W_q, dW = dyᵀ·x_q. Gradients pass through rounding unchanged.
- [ ] **Step 1: Write failing tests**: `quant_weights_ternary_values` (every out/γ ∈ {−1, 0, 1}), `quant_acts_int8_range`, `bitlinear_ste_grad_matches_formula` (compare against an explicit dy·W_q and dyᵀ·x_q), `bitnet_tiny_model_loss_decreases` (the tiny cfg with bitnet, 100 steps).
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Run 60 s trains `--bitnet` with seeds 42 and 43 (tune Muon/Adam lr ×{1, 2, 4} for BitNet; record all). Compare against the fp32 rows in `runs/README.md`: val losses and tokens trained within 60 s.
- [ ] **Step 6:** Commit: `feat: BitNet b1.58 BitLinear training with STE`

### Task 15: Ternary inference kernel (AVX-512 VNNI)

**Files:**
- Create: `rllm/src/ternary.rs`
- Modify: `rllm/src/infer.rs`, `rllm/src/bin/chat.rs` (`--kernel ternary`)

**Interfaces:**
- Produces:
  - `pub struct TernaryMat { rows, cols, gamma:f32, packed: Vec<u8> }`, with `from_f32(w:&Mat) -> TernaryMat`. Packing is the builder's call: either i8 {−1, 0, 1} laid out for `vpdpbusd`, or 2-bit packed and expanded in-register. Pick by bench, and note both sizes in results.
  - `pub fn ternary_matvec(out:&mut[f32], m:&TernaryMat, x_q:&[i8], x_scale:f32)`. It uses VNNI `_mm512_dpbusd_epi32`. Handle the u8×i8 operand rule by offsetting activations by +128 and subtracting 128·Σrow(w), with the row sums precomputed at pack time.
  - `pub fn quant_acts_i8(x:&[f32]) -> (Vec<i8>, f32)`, matching the training activation quantization.
  - When `--kernel ternary` is set, inference packs every BitLinear weight on load and runs the block matmuls through `ternary_matvec`.
- [ ] **Step 1: Write failing tests**: `ternary_matvec_matches_fakequant` (for random W and x with shapes (1,1), (19,33), (384,128), the result matches the fp32 `quant_acts`/`quant_weights` path within 1e-4 relative), `vnni_offset_correction_exact` (all-negative activations), `ternary_chat_logits_close` (the KV step logits for a tiny bitnet model match the fp32 fake-quant path within 1e-3).
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Bench chat generation tok/s: fp32 vs ternary on the trained BitNet model. Add a `bench --infer` mode that runs 200 tokens with argmax, median of 5. Record the tok/s and model file size in `runs/README.md`.
- [ ] **Step 6:** Commit: `perf: ternary VNNI inference kernel`

### Task 16: Inline-asm matmul microkernel experiment

**Files:**
- Create: `rllm/src/ops/asm.rs`
- Modify: `rllm/src/backend.rs` (`Asm` = avx512mt, but with the fwd matmul microkernel replaced by `asm!`)

**Interfaces:**
- Produces: `asm::matmul_fwd` (same signature). It reuses the Task 11 packing and replaces only the inner microkernel with `core::arch::asm!` (explicit zmm register allocation, unrolled FMA, prefetch). Tails fall back to the intrinsics microkernel.
- [ ] **Step 1: Write failing test**: `asm_matmul_matches_scalar` on the shape set.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Bench asm vs avx512 microkernel GFLOPS, then run a 60 s train with `--backend asm --name asm`. Write the conclusion (gain or no gain, and why) in `bench/results.md`.
- [ ] **Step 6:** Commit: `exp: inline asm matmul microkernel vs intrinsics`

### Task 17: Final comparison write-up

- [ ] **Step 1:** In `runs/README.md`, produce the final table: one row per engine stage (scalar → blocked → avx512 → avx512mt → asm) with tokens trained in 60 s and val_loss_sft. Add a fp32 vs BitNet row pair and the fixed 20-question outputs (create `runs/prompts.txt` with 20 ELI5-style questions; add `chat --batch runs/prompts.txt --temp 0` for non-interactive output).
- [ ] **Step 2:** Commit: `docs: final 60s speedrun results`
