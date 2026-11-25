use rllm::bpe::{train_bpe, Bpe};
use serde::Deserialize;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: tokenize --data <dir> [--vocab 2048] [--bpe-train-mb 10]";
const DOC_SEP: &[u8] = b"\n\x1e\n";
/// Context (256) + 1: longest SFT record kept.
const MAX_SFT_LEN: usize = 257;

#[derive(Deserialize)]
struct Pair {
    q: String,
    a: String,
}

/// `<|user|> q <|assistant|> a <|end|>`; None if the answer is empty or the record
/// is longer than `max_len` tokens.
fn encode_sft_record(bpe: &Bpe, q: &str, a: &str, max_len: usize) -> Option<Vec<u16>> {
    if a.trim().is_empty() {
        return None;
    }
    let mut rec = vec![bpe.user()];
    rec.extend(bpe.encode(q.as_bytes()));
    rec.push(bpe.assistant());
    rec.extend(bpe.encode(a.as_bytes()));
    rec.push(bpe.end());
    (rec.len() <= max_len).then_some(rec)
}

/// Documents (split on `\n\x1e\n`) encoded and each followed by `<|end|>`.
fn encode_pretrain(bpe: &Bpe, text: &[u8]) -> Vec<u16> {
    let end = bpe.end();
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let (doc, next) = match find(rest, DOC_SEP) {
            Some(i) => (&rest[..i], Some(&rest[i + DOC_SEP.len()..])),
            None => (rest, None),
        };
        if !doc.iter().all(|b| b.is_ascii_whitespace()) {
            out.extend(bpe.encode(doc));
            out.push(end);
        }
        match next {
            Some(n) => rest = n,
            None => break,
        }
    }
    out
}

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|w| w == n)
}

fn write_u16s(path: &Path, toks: &[u16]) -> io::Result<()> {
    std::fs::write(path, toks.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>())
}

/// Returns (file bytes, kept, dropped, malformed lines).
fn encode_sft_file(bpe: &Bpe, path: &Path) -> io::Result<(Vec<u8>, usize, usize, usize)> {
    let mut bytes = Vec::new();
    let (mut kept, mut dropped, mut bad) = (0, 0, 0);
    for line in BufReader::new(std::fs::File::open(path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(p) = serde_json::from_str::<Pair>(&line) else {
            bad += 1;
            continue;
        };
        match encode_sft_record(bpe, &p.q, &p.a, MAX_SFT_LEN) {
            Some(rec) => {
                bytes.extend_from_slice(&(rec.len() as u16).to_le_bytes());
                for t in rec {
                    bytes.extend_from_slice(&t.to_le_bytes());
                }
                kept += 1;
            }
            None => dropped += 1,
        }
    }
    Ok((bytes, kept, dropped, bad))
}

fn run(data: &Path, vocab: usize, bpe_mb: usize) -> io::Result<()> {
    let t0 = Instant::now();
    let train_txt = std::fs::read(data.join("pretrain_train.txt"))?;
    let mut cut = train_txt.len().min(bpe_mb * 1024 * 1024);
    // do not cut inside a UTF-8 sequence
    while cut > 0 && cut < train_txt.len() && (train_txt[cut] & 0xC0) == 0x80 {
        cut -= 1;
    }
    println!("training BPE (vocab {vocab}) on {:.1} MB ...", cut as f64 / 1e6);
    let bpe = train_bpe(&train_txt[..cut], vocab);
    bpe.save(&data.join("tokenizer.bin"))?;
    println!("tokenizer.bin written ({:.1}s)", t0.elapsed().as_secs_f64());

    for split in ["train", "val"] {
        let txt = if split == "train" {
            train_txt.clone()
        } else {
            std::fs::read(data.join("pretrain_val.txt"))?
        };
        let toks = encode_pretrain(&bpe, &txt);
        write_u16s(&data.join(format!("pretrain_{split}.bin")), &toks)?;
        println!(
            "pretrain_{split}.bin: {} tokens from {:.1} MB ({:.2} bytes/token)",
            toks.len(),
            txt.len() as f64 / 1e6,
            txt.len() as f64 / toks.len().max(1) as f64
        );
    }
    for split in ["train", "val"] {
        let (bytes, kept, dropped, bad) = encode_sft_file(&bpe, &data.join(format!("sft_{split}.jsonl")))?;
        std::fs::write(data.join(format!("sft_{split}.bin")), &bytes)?;
        println!("sft_{split}.bin: {kept} records ({} tokens), dropped {dropped} (empty answer or > {MAX_SFT_LEN} tokens), malformed {bad}", bytes.len() / 2 - kept);
    }
    println!("done in {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn main() -> ExitCode {
    let (mut data, mut vocab, mut mb): (Option<PathBuf>, usize, usize) = (None, 2048, 10);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args.next();
        match (a.as_str(), v) {
            ("--data", Some(v)) => data = Some(PathBuf::from(v)),
            ("--vocab", Some(v)) if v.parse::<usize>().is_ok() => vocab = v.parse().unwrap(),
            ("--bpe-train-mb", Some(v)) if v.parse::<usize>().is_ok() => mb = v.parse().unwrap(),
            _ => {
                eprintln!("bad argument {a}\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(data) = data else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match run(&data, vocab, mb) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bpe() -> Bpe {
        train_bpe(b"hello world hello there general kenobi hello world", 300)
    }

    #[test]
    fn record_layout() {
        let b = bpe();
        let r = encode_sft_record(&b, "hello", "world", 257).unwrap();
        assert_eq!(r[0], b.user());
        assert_eq!(*r.last().unwrap(), b.end());
        let ai = r.iter().position(|&t| t == b.assistant()).unwrap();
        assert_eq!(b.decode(&r[1..ai]), b"hello");
        assert_eq!(b.decode(&r[ai + 1..r.len() - 1]), b"world");
    }

    #[test]
    fn overlong_record_dropped() {
        let b = bpe();
                let ok = "x".repeat(257 - 3); // user, q="" , assistant, 254 a, end = 257
        assert_eq!(encode_sft_record(&b, "", &ok, 257).unwrap().len(), 257);
        let long = "x".repeat(257 - 2);
        assert!(encode_sft_record(&b, "", &long, 257).is_none());
        assert!(encode_sft_record(&b, &"y".repeat(1000), "a", 257).is_none());
    }

    #[test]
    fn empty_answer_dropped() {
        let b = bpe();
        assert!(encode_sft_record(&b, "hello", "", 257).is_none());
        assert!(encode_sft_record(&b, "hello", "  \n", 257).is_none());
    }

    #[test]
    fn pretrain_docs_joined_by_end() {
        let b = bpe();
        let toks = encode_pretrain(&b, "hello world\n\x1e\nhello there\n\x1e\n\n\x1e\nlast".as_bytes());
        let ends = toks.iter().filter(|&&t| t == b.end()).count();
        assert_eq!(ends, 3); // empty doc skipped
        assert_eq!(*toks.last().unwrap(), b.end());
        let docs: Vec<Vec<u8>> = toks.split(|&t| t == b.end()).filter(|d| !d.is_empty()).map(|d| b.decode(d)).collect();
        assert_eq!(docs, vec![b"hello world".to_vec(), b"hello there".to_vec(), b"last".to_vec()]);
    }
}
