//! Pushshift dump extraction: pretraining text and SFT question/answer pairs.

use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Subreddit dump prefixes (exact case as used by Pushshift).
pub const SUBS: [&str; 3] = ["explainlikeimfive", "NoStupidQuestions", "askscience"];
const ELI5: &str = "explainlikeimfive";
const NSQ: &str = "NoStupidQuestions";
const DOC_SEP: &str = "\n\x1e\n";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SftPair {
    pub q: String,
    pub a: String,
}

// ---------------------------------------------------------------- reading

type Counter = Arc<AtomicU64>;

struct NdjsonIter<R: BufRead> {
    reader: R,
    buf: Vec<u8>,
    skipped: Counter,
}

impl<R: BufRead> Iterator for NdjsonIter<R> {
    type Item = Value;
    fn next(&mut self) -> Option<Value> {
        loop {
            self.buf.clear();
            match self.reader.read_until(b'\n', &mut self.buf) {
                Ok(0) => return None,
                Ok(_) => {}
                Err(e) => {
                    // truncated / corrupt stream: count once and stop
                    eprintln!("warning: read error, stopping this file early: {e}");
                    self.skipped.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            }
            if self.buf.iter().all(|b| b.is_ascii_whitespace()) {
                continue;
            }
            match serde_json::from_slice::<Value>(&self.buf) {
                Ok(v) => return Some(v),
                Err(_) => {
                    self.skipped.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Streams a zstd-compressed NDJSON file. Unparsable lines are counted in
/// `skipped` and skipped.
pub fn open_zst_ndjson_counted(
    path: &Path,
    skipped: Counter,
) -> io::Result<impl Iterator<Item = Value>> {
    let file = File::open(path)
        .map_err(|e| io::Error::new(e.kind(), format!("cannot open {}: {e}", path.display())))?;
    let mut dec = zstd::stream::Decoder::new(file)?;
    dec.window_log_max(31)?; // Pushshift dumps use long-window zstd
    Ok(NdjsonIter { reader: BufReader::with_capacity(1 << 20, dec), buf: Vec::new(), skipped })
}

pub fn open_zst_ndjson(path: &Path) -> io::Result<impl Iterator<Item = Value>> {
    open_zst_ndjson_counted(path, Arc::new(AtomicU64::new(0)))
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

/// (id, cleaned "title\nselftext") of a submission, or None when unusable.
fn parse_submission(v: &Value) -> Option<(String, String)> {
    let id = v.get("id")?.as_str()?;
    let title = clean_text(v.get("title")?.as_str()?)?;
    let selftext = v.get("selftext").and_then(Value::as_str).and_then(clean_text);
    let q = match selftext {
        Some(s) => format!("{title}\n{s}"),
        None => title,
    };
    Some((id.to_string(), q))
}

struct Comment<'a> {
    link_id: &'a str,
    top_level: bool,
    body: &'a str,
    score: i64,
    author: &'a str,
}

fn parse_comment(v: &Value) -> Option<Comment<'_>> {
    let link = v.get("link_id")?.as_str()?;
    let parent = v.get("parent_id")?.as_str()?;
    Some(Comment {
        link_id: link.strip_prefix("t3_").unwrap_or(link),
        top_level: parent == link,
        body: v.get("body")?.as_str()?,
        score: v.get("score")?.as_i64()?,
        author: v.get("author")?.as_str()?,
    })
}

// -------------------------------------------------------------- SFT pairs

struct Candidate {
    score: i64,
    answer: String,
}

/// Streams comments, keeping the best qualifying top-level answer per
/// submission that is in `questions`.
fn best_answers(
    comments: impl Iterator<Item = Value>,
    questions: &HashMap<String, String>,
    max_answer_chars: usize,
    skipped: &AtomicU64,
) -> HashMap<String, Candidate> {
    let mut best: HashMap<String, Candidate> = HashMap::new();
    for v in comments {
        let Some(c) = parse_comment(&v) else {
            skipped.fetch_add(1, Ordering::Relaxed);
            continue;
        };
        if !c.top_level || c.score < 10 || c.author == "AutoModerator" {
            continue;
        }
        if !questions.contains_key(c.link_id) {
            continue;
        }
        if best.get(c.link_id).is_some_and(|b| b.score >= c.score) {
            continue;
        }
        let Some(a) = clean_text(c.body) else { continue };
        let n = a.chars().count();
        if !(20..=max_answer_chars).contains(&n) {
            continue;
        }
        best.insert(c.link_id.to_string(), Candidate { score: c.score, answer: a });
    }
    best
}

/// Pairs (submission id, score, pair) for one subreddit.
fn sft_for_sub(
    subs_path: &Path,
    comments_path: &Path,
    max_answer_chars: usize,
    skipped: &Counter,
) -> io::Result<Vec<(String, i64, SftPair)>> {
    let mut questions = HashMap::new();
    for v in open_zst_ndjson_counted(subs_path, skipped.clone())? {
        match parse_submission(&v) {
            Some((id, q)) => {
                questions.insert(id, q);
            }
            None => {
                skipped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    let comments = open_zst_ndjson_counted(comments_path, skipped.clone())?;
    let best = best_answers(comments, &questions, max_answer_chars, skipped);
    let mut out: Vec<_> = best
        .into_iter()
        .map(|(id, c)| {
            let q = questions[&id].clone();
            (id, c.score, SftPair { q, a: c.answer })
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

// ---------------------------------------------------------------- pretrain

/// Lazily yields (submission id for split, text) for submissions then comments.
fn pretrain_docs(
    subs_path: &Path,
    comments_path: &Path,
    skipped: &Counter,
) -> io::Result<Box<dyn Iterator<Item = (String, String)>>> {
    let subs = open_zst_ndjson_counted(subs_path, skipped.clone())?.filter_map(|v| {
        let (id, q) = parse_submission(&v)?;
        Some((id, q))
    });
    let coms = open_zst_ndjson_counted(comments_path, skipped.clone())?.filter_map(|v| {
        let c = v.get("link_id")?.as_str()?;
        let id = c.strip_prefix("t3_").unwrap_or(c).to_string();
        let body = clean_text(v.get("body")?.as_str()?)?;
        Some((id, body))
    });
    Ok(Box::new(subs.chain(coms)))
}

// -------------------------------------------------------------------- run

#[derive(Debug, Clone)]
pub struct ExtractConfig {
    pub subs_dir: PathBuf,
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
    /// Malformed or incomplete lines skipped while building SFT pairs.
    pub skipped_sft: u64,
    /// Malformed or incomplete lines skipped while building pretraining text.
    pub skipped_pretrain: u64,
}

/// Finds `<sub>_<kind>.zst` in `dir`, matching the file name case-insensitively.
fn find_dump(dir: &Path, sub: &str, kind: &str) -> io::Result<PathBuf> {
    let want = format!("{sub}_{kind}.zst");
    let lower = want.to_lowercase();
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().to_lowercase() == lower {
                return Ok(e.path());
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("missing dump file {want} in {}", dir.display()),
    ))
}

pub fn run(cfg: &ExtractConfig) -> io::Result<ExtractStats> {
    let mut paths = Vec::new();
    for s in SUBS {
        paths.push((s, find_dump(&cfg.subs_dir, s, "submissions")?, find_dump(&cfg.subs_dir, s, "comments")?));
    }
    fs::create_dir_all(&cfg.out_dir)?;
    let mut stats = ExtractStats::default();

    // --- SFT
    let skipped_sft: Counter = Arc::new(AtomicU64::new(0));
    let find = |name: &str| paths.iter().find(|p| p.0 == name).unwrap();
    let (_, es, ec) = find(ELI5);
    let (_, ns, nc) = find(NSQ);
    let eli5 = sft_for_sub(es, ec, 800, &skipped_sft)?;
    let mut nsq = sft_for_sub(ns, nc, 300, &skipped_sft)?;
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
    stats.skipped_sft = skipped_sft.load(Ordering::Relaxed);

    // --- pretraining, round-robin across subreddits
    let skipped_pre: Counter = Arc::new(AtomicU64::new(0));
    let mut sources = Vec::new();
    for (_, s, c) in &paths {
        sources.push(pretrain_docs(s, c, &skipped_pre)?);
    }
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
    stats.skipped_pretrain = skipped_pre.load(Ordering::Relaxed);
    Ok(stats)
}
