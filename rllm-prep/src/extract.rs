//! Monthly all-Reddit Pushshift extraction: pretraining text and SFT pairs.

use memchr::memmem;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Subreddits to keep (matched case-insensitively).
pub const SUBS: [&str; 3] = ["explainlikeimfive", "NoStupidQuestions", "askscience"];
const NSQ_IDX: usize = 1;
const ASK_IDX: usize = 2;
const DOC_SEP: &str = "\n\x1e\n";
const PROGRESS_EVERY: u64 = 1_000_000;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SftPair {
    pub q: String,
    pub a: String,
}

// ---------------------------------------------------------------- reading

/// Streams every non-blank line of a zstd file to `f` (one pass, constant
/// memory). A read error mid-stream (truncated file) warns and stops this file.
pub fn for_each_line(path: &Path, mut f: impl FnMut(&[u8])) -> io::Result<()> {
    let file = File::open(path)
        .map_err(|e| io::Error::new(e.kind(), format!("cannot open {}: {e}", path.display())))?;
    let mut dec = zstd::stream::Decoder::new(file)?;
    dec.window_log_max(31)?; // Pushshift dumps use long-window zstd
    let mut reader = BufReader::with_capacity(1 << 20, dec);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => return Ok(()),
            Ok(_) => {}
            Err(e) => {
                eprintln!("warning: read error in {}, stopping this file early: {e}", path.display());
                return Ok(());
            }
        }
        if !buf.iter().all(|b| b.is_ascii_whitespace()) {
            f(&buf);
        }
    }
}

/// Cheap byte-level check run before JSON parsing: does the line contain
/// `"subreddit":"<name>"` for one of [`SUBS`] (case-insensitive)? It can give
/// false positives (e.g. a nested object), so callers must re-check the parsed field.
pub fn prefilter(line: &[u8]) -> bool {
    static KEY: OnceLock<memmem::Finder<'static>> = OnceLock::new();
    let key = KEY.get_or_init(|| memmem::Finder::new(b"\"subreddit\":\""));
    key.find_iter(line).any(|p| {
        let rest = &line[p + key.needle().len()..];
        SUBS.iter().any(|n| {
            rest.len() > n.len()
                && rest[..n.len()].eq_ignore_ascii_case(n.as_bytes())
                && rest[n.len()] == b'"'
        })
    })
}

// --------------------------------------------------------------- cleaning

fn is_deleted(s: &str) -> bool {
    let t = s.trim();
    t == "[deleted]" || t == "[removed]"
}

/// Replaces `[text](url)` with `text`.
fn strip_md_links(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('[') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        if let Some(j) = after.find("](") {
            let text = &after[..j];
            let tail = &after[j + 2..];
            if !text.contains('[') && !text.contains('\n') {
                if let Some(k) = tail.find(')') {
                    if !tail[..k].contains(char::is_whitespace) {
                        out.push_str(text);
                        rest = &tail[k + 1..];
                        continue;
                    }
                }
            }
        }
        out.push('[');
        rest = after;
    }
    out.push_str(rest);
    out
}

pub fn clean_text(s: &str) -> Option<String> {
    if is_deleted(s) {
        return None;
    }
    let no_links = strip_md_links(s);
    let cleaned = no_links
        .split_whitespace()
        .filter(|t| {
            let t = t.trim_start_matches('(');
            !(t.starts_with("http://") || t.starts_with("https://"))
        })
        .collect::<Vec<_>>()
        .join(" ");
    if cleaned.is_empty() || is_deleted(&cleaned) {
        None
    } else {
        Some(cleaned)
    }
}

/// Stable FNV-1a split: 2% of submissions go to validation.
pub fn split_is_val(submission_id: &str) -> bool {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in submission_id.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h % 50 == 0
}

// ------------------------------------------------------------ field access

fn sub_index(v: &Value) -> Option<usize> {
    let s = v.get("subreddit")?.as_str()?;
    SUBS.iter().position(|n| n.eq_ignore_ascii_case(s))
}

// ------------------------------------------------------------------- scan

struct Candidate {
    score: i64,
    answer: String,
}

/// State filled by the single pass over every file.
struct Scan {
    max_stream_bytes: u64,
    /// (is NSQ, cleaned "title\nselftext") per ELI5/NSQ submission id.
    questions: HashMap<String, (bool, String)>,
    /// Best qualifying top-level answer per submission id.
    best: HashMap<String, Candidate>,
    /// Pretrain buffers (split id, text): per subreddit, [submissions, comments].
    /// Each is capped at the pretrain byte limit, since the output can never
    /// use more than that from any single stream.
    streams: [Vec<(String, String)>; 6],
    stream_bytes: [u64; 6],
    kept_per_sub: [u64; 3],
    lines_read: u64,
    lines_kept: u64,
    skipped: u64,
}

impl Scan {
    fn new(max_stream_bytes: u64) -> Self {
        Scan {
            max_stream_bytes,
            questions: HashMap::new(),
            best: HashMap::new(),
            streams: Default::default(),
            stream_bytes: [0; 6],
            kept_per_sub: [0; 3],
            lines_read: 0,
            lines_kept: 0,
            skipped: 0,
        }
    }

    fn push_doc(&mut self, stream: usize, id: &str, text: String) {
        if self.stream_bytes[stream] < self.max_stream_bytes {
            self.stream_bytes[stream] += (text.len() + DOC_SEP.len()) as u64;
            self.streams[stream].push((id.to_string(), text));
        }
    }

    fn line(&mut self, line: &[u8], is_submission: bool) {
        self.lines_read += 1;
        if self.lines_read % PROGRESS_EVERY == 0 {
            eprintln!(
                "progress: {} lines read, {} kept, {:.1} MB pretrain text buffered",
                self.lines_read,
                self.lines_kept,
                self.stream_bytes.iter().sum::<u64>() as f64 / 1e6
            );
        }
        if !prefilter(line) {
            return;
        }
        let Ok(v) = serde_json::from_slice::<Value>(line) else {
            self.skipped += 1;
            return;
        };
        // the prefilter has false positives: trust only the parsed field
        let Some(sub) = sub_index(&v) else { return };
        self.lines_kept += 1;
        self.kept_per_sub[sub] += 1;
        let ok = if is_submission { self.submission(&v, sub) } else { self.comment(&v, sub) };
        if !ok {
            self.skipped += 1;
        }
    }

    /// Returns false when required fields are missing.
    fn submission(&mut self, v: &Value, sub: usize) -> bool {
        let (Some(id), Some(title)) =
            (v.get("id").and_then(Value::as_str), v.get("title").and_then(Value::as_str))
        else {
            return false;
        };
        let Some(title) = clean_text(title) else { return true };
        let q = match v.get("selftext").and_then(Value::as_str).and_then(clean_text) {
            Some(s) => format!("{title}\n{s}"),
            None => title,
        };
        if sub != ASK_IDX {
            self.questions.insert(id.to_string(), (sub == NSQ_IDX, q.clone()));
        }
        self.push_doc(sub * 2, id, q);
        true
    }

    /// Returns false when required fields are missing.
    fn comment(&mut self, v: &Value, sub: usize) -> bool {
        let (Some(link), Some(body)) =
            (v.get("link_id").and_then(Value::as_str), v.get("body").and_then(Value::as_str))
        else {
            return false;
        };
        let id = link.strip_prefix("t3_").unwrap_or(link);
        let Some(text) = clean_text(body) else { return true };
        // SFT: only top-level comments on a submission we have (RS may lack it)
        if let (Some(parent), Some(score), Some(author), Some((is_nsq, _))) = (
            v.get("parent_id").and_then(Value::as_str),
            v.get("score").and_then(Value::as_i64),
            v.get("author").and_then(Value::as_str),
            self.questions.get(id),
        ) {
            let max_chars = if *is_nsq { 300 } else { 800 };
            let n = text.chars().count();
            let beats = self.best.get(id).is_none_or(|b| b.score < score);
            if parent == link && score >= 10 && author != "AutoModerator" && beats && (20..=max_chars).contains(&n)
            {
                self.best.insert(id.to_string(), Candidate { score, answer: text.clone() });
            }
        }
        self.push_doc(sub * 2 + 1, id, text);
        true
    }
}

// -------------------------------------------------------------------- run

#[derive(Debug, Clone)]
pub struct ExtractConfig {
    /// Monthly submission files (`RS_YYYY-MM.zst`), read before the comments.
    pub rs_files: Vec<PathBuf>,
    /// Monthly comment files (`RC_YYYY-MM.zst`).
    pub rc_files: Vec<PathBuf>,
    pub out_dir: PathBuf,
    pub max_pretrain_bytes: u64,
}

#[derive(Debug, Default, Clone)]
pub struct ExtractStats {
    pub pretrain_train_bytes: u64,
    pub pretrain_val_bytes: u64,
    pub sft_train: usize,
    pub sft_val: usize,
    pub sft_total: usize,
    pub sft_eli5: usize,
    pub sft_nsq: usize,
    pub lines_read: u64,
    /// Lines whose parsed subreddit is one of the three kept.
    pub lines_kept: u64,
    /// Lines that passed the prefilter but were unparsable or lacked required fields.
    pub skipped: u64,
    /// Subreddits with zero matching lines (warned about, not an error).
    pub missing_subs: Vec<String>,
}

pub fn run(cfg: &ExtractConfig) -> io::Result<ExtractStats> {
    // fail fast on a bad path, before hours of scanning
    for p in cfg.rs_files.iter().chain(&cfg.rc_files) {
        File::open(p)
            .map_err(|e| io::Error::new(e.kind(), format!("cannot open {}: {e}", p.display())))?;
    }
    fs::create_dir_all(&cfg.out_dir)?;
    let mut stats = ExtractStats::default();

    // --- one pass per file; submissions first so comments can find their question
    let mut scan = Scan::new(cfg.max_pretrain_bytes);
    for (files, is_sub) in [(&cfg.rs_files, true), (&cfg.rc_files, false)] {
        for p in files {
            for_each_line(p, |l| scan.line(l, is_sub))?;
        }
    }
    stats.lines_read = scan.lines_read;
    stats.lines_kept = scan.lines_kept;
    stats.skipped = scan.skipped;
    for (name, n) in SUBS.iter().zip(scan.kept_per_sub) {
        if n == 0 {
            eprintln!("warning: no lines for r/{name}");
            stats.missing_subs.push(name.to_string());
        }
    }

    // --- SFT pairs
    let mut eli5: Vec<(String, i64, SftPair)> = vec![];
    let mut nsq: Vec<(String, i64, SftPair)> = vec![];
    for (id, c) in scan.best {
        let (is_nsq, q) = &scan.questions[&id];
        let item = (id, c.score, SftPair { q: q.clone(), a: c.answer });
        if *is_nsq { nsq.push(item) } else { eli5.push(item) }
    }
    eli5.sort_by(|a, b| a.0.cmp(&b.0));
    // NSQ must be <= 20% of the total: n <= eli5 / 4. Keep the highest scored.
    nsq.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    nsq.truncate(eli5.len() / 4);
    stats.sft_eli5 = eli5.len();
    stats.sft_nsq = nsq.len();
    let mut train = BufWriter::new(File::create(cfg.out_dir.join("sft_train.jsonl"))?);
    let mut val = BufWriter::new(File::create(cfg.out_dir.join("sft_val.jsonl"))?);
    for (id, _, pair) in eli5.iter().chain(nsq.iter()) {
        let line = serde_json::to_string(pair).map_err(io::Error::other)?;
        if split_is_val(id) {
            writeln!(val, "{line}")?;
            stats.sft_val += 1;
        } else {
            writeln!(train, "{line}")?;
            stats.sft_train += 1;
        }
    }
    train.flush()?;
    val.flush()?;
    stats.sft_total = stats.sft_train + stats.sft_val;

    // --- pretraining, round-robin across all 6 streams (submissions and
    // comments of each subreddit), so comment bodies appear even when the
    // byte limit is hit long before any submissions stream is exhausted
    let mut sources: Vec<_> = scan.streams.into_iter().map(Vec::into_iter).collect();
    let mut ptrain = BufWriter::new(File::create(cfg.out_dir.join("pretrain_train.txt"))?);
    let mut pval = BufWriter::new(File::create(cfg.out_dir.join("pretrain_val.txt"))?);
    let mut total: u64 = 0;
    'outer: loop {
        let mut any = false;
        for src in sources.iter_mut() {
            if total >= cfg.max_pretrain_bytes {
                break 'outer;
            }
            let Some((id, text)) = src.next() else { continue };
            any = true;
            let n = (text.len() + DOC_SEP.len()) as u64;
            let (w, counter) = if split_is_val(&id) {
                (&mut pval, &mut stats.pretrain_val_bytes)
            } else {
                (&mut ptrain, &mut stats.pretrain_train_bytes)
            };
            w.write_all(text.as_bytes())?;
            w.write_all(DOC_SEP.as_bytes())?;
            *counter += n;
            total += n;
        }
        if !any {
            break;
        }
    }
    ptrain.flush()?;
    pval.flush()?;
    Ok(stats)
}
