//! Byte-level BPE tokenizer.
//!
//! Vocab layout: ids 0..256 are raw bytes, then learned merges, then (if training
//! stopped early) empty placeholder tokens, then the three specials
//! `<|user|>`, `<|assistant|>`, `<|end|>` as the last three ids.

use std::collections::HashMap;
use std::io::{self, Error, ErrorKind};
use std::path::Path;

const MAGIC: &[u8; 4] = b"RBPE";
const SPECIALS: [&[u8]; 3] = [b"<|user|>", b"<|assistant|>", b"<|end|>"];

#[derive(Clone, Debug)]
pub struct Bpe {
    pub merges: Vec<(u32, u32)>,
    pub vocab: Vec<Vec<u8>>,
}

/// Split into chunks; a whitespace run that follows a non-whitespace byte starts
/// a new chunk, so a leading space stays attached to the word after it.
fn chunks(text: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..text.len() {
        if text[i].is_ascii_whitespace() && !text[i - 1].is_ascii_whitespace() {
            out.push(&text[start..i]);
            start = i;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

fn build_vocab(merges: &[(u32, u32)], vocab_size: usize) -> Vec<Vec<u8>> {
    let mut vocab: Vec<Vec<u8>> = (0..256u32).map(|b| vec![b as u8]).collect();
    for &(a, b) in merges {
        let mut t = vocab[a as usize].clone();
        t.extend_from_slice(&vocab[b as usize]);
        vocab.push(t);
    }
    while vocab.len() < vocab_size - 3 {
        vocab.push(Vec::new());
    }
    vocab.extend(SPECIALS.iter().map(|s| s.to_vec()));
    vocab
}

pub fn train_bpe(text: &[u8], vocab_size: usize) -> Bpe {
    assert!(vocab_size >= 256 + 3, "vocab_size must be at least 259");
    assert!(vocab_size <= 65536, "token ids must fit in u16");
    let mut freq: HashMap<&[u8], u64> = HashMap::new();
    for c in chunks(text) {
        *freq.entry(c).or_insert(0) += 1;
    }
    let mut words: Vec<(Vec<u32>, u64)> = freq
        .into_iter()
        .map(|(w, n)| (w.iter().map(|&b| b as u32).collect(), n))
        .collect();
    // Deterministic order independent of HashMap iteration.
    words.sort();

    let mut merges: Vec<(u32, u32)> = Vec::new();
    while 256 + merges.len() < vocab_size - 3 {
        let mut counts: HashMap<(u32, u32), u64> = HashMap::new();
        for (w, n) in &words {
            for p in w.windows(2) {
                *counts.entry((p[0], p[1])).or_insert(0) += n;
            }
        }
        // Highest count, ties broken by smallest pair.
        let best = counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)));
        let Some((pair, count)) = best else { break };
        if count < 2 {
            break;
        }
        let new_id = 256 + merges.len() as u32;
        merges.push(pair);
        for (w, _) in words.iter_mut() {
            merge_pair(w, pair, new_id);
        }
    }
    let vocab = build_vocab(&merges, vocab_size);
    Bpe { merges, vocab }
}

fn merge_pair(w: &mut Vec<u32>, pair: (u32, u32), new_id: u32) {
    let mut out = 0;
    let mut i = 0;
    while i < w.len() {
        if i + 1 < w.len() && w[i] == pair.0 && w[i + 1] == pair.1 {
            w[out] = new_id;
            i += 2;
        } else {
            w[out] = w[i];
            i += 1;
        }
        out += 1;
    }
    w.truncate(out);
}

impl Bpe {
    fn n_special_base(&self) -> usize {
        self.vocab.len() - 3
    }
    pub fn user(&self) -> u16 {
        self.n_special_base() as u16
    }
    pub fn assistant(&self) -> u16 {
        self.n_special_base() as u16 + 1
    }
    pub fn end(&self) -> u16 {
        self.n_special_base() as u16 + 2
    }

    pub fn encode(&self, text: &[u8]) -> Vec<u16> {
        let rank: HashMap<(u32, u32), u32> = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i as u32))
            .collect();
        let mut out = Vec::new();
        for c in chunks(text) {
            let mut w: Vec<u32> = c.iter().map(|&b| b as u32).collect();
            loop {
                let best = w
                    .windows(2)
                    .filter_map(|p| rank.get(&(p[0], p[1])).map(|&r| (r, (p[0], p[1]))))
                    .min();
                let Some((r, pair)) = best else { break };
                merge_pair(&mut w, pair, 256 + r);
            }
            out.extend(w.into_iter().map(|t| t as u16));
        }
        out
    }

    pub fn decode(&self, ids: &[u16]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if let Some(t) = self.vocab.get(id as usize) {
                out.extend_from_slice(t);
            }
        }
        out
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut buf = Vec::with_capacity(12 + self.merges.len() * 8);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(self.vocab.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.merges.len() as u32).to_le_bytes());
        for &(a, b) in &self.merges {
            buf.extend_from_slice(&a.to_le_bytes());
            buf.extend_from_slice(&b.to_le_bytes());
        }
        std::fs::write(path, buf)
    }

    pub fn load(path: &Path) -> io::Result<Bpe> {
        let bad = |m: &str| Error::new(ErrorKind::InvalidData, m.to_string());
        let d = std::fs::read(path)?;
        if d.len() < 12 || &d[..4] != MAGIC {
            return Err(bad("bad BPE file header"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
        let vocab_size = u32_at(4) as usize;
        let n = u32_at(8) as usize;
        if d.len() != 12 + n * 8 || vocab_size < 259 || vocab_size > 65536 || 256 + n > vocab_size - 3 {
            return Err(bad("inconsistent BPE file"));
        }
        let mut merges = Vec::with_capacity(n);
        for i in 0..n {
            let (a, b) = (u32_at(12 + i * 8), u32_at(16 + i * 8));
            if a >= 256 + i as u32 || b >= 256 + i as u32 {
                return Err(bad("merge refers to undefined token"));
            }
            merges.push((a, b));
        }
        let vocab = build_vocab(&merges, vocab_size);
        Ok(Bpe { merges, vocab })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_layout() {
        let b = train_bpe(b"hello hello hello world world", 260 + 3);
        assert_eq!(b.vocab.len(), 263);
        assert_eq!(b.end(), 262);
        assert_eq!(b.assistant(), 261);
        assert_eq!(b.user(), 260);
    }

    #[test]
    fn merges_shorten_text() {
        let t = b"the cat the cat the cat the cat".repeat(10);
        let b = train_bpe(&t, 300);
        assert!(b.encode(&t).len() < t.len() / 2);
    }

    #[test]
    fn roundtrip_arbitrary_bytes() {
        let b = train_bpe(b"abc abc abc", 270);
        for s in [&b"\xff\xfe\x00"[..], "안녕 🙂".as_bytes(), b"<|end|>", b""] {
            assert_eq!(b.decode(&b.encode(s)), s);
        }
        assert!(!b.encode(b"<|end|>").contains(&b.end()));
    }

    #[test]
    fn specials_decode_to_literals() {
        let b = train_bpe(b"abc abc abc", 270);
        assert_eq!(b.decode(&[b.user()]), b"<|user|>");
        assert_eq!(b.decode(&[b.assistant()]), b"<|assistant|>");
        assert_eq!(b.decode(&[b.end()]), b"<|end|>");
    }

    #[test]
    fn placeholders_pad_when_no_pair_repeats() {
        let b = train_bpe(b"abcdefg", 300);
        assert_eq!(b.vocab.len(), 300);
        assert!(b.merges.is_empty());
        assert_eq!(b.end(), 299);
        assert_eq!(b.user(), 297);
        assert!(b.encode(b"abcdefg").iter().all(|&i| i < 256));
    }

    #[test]
    fn deterministic() {
        let t = b"low lower lowest newer newest wider".repeat(5);
        assert_eq!(train_bpe(&t, 290).merges, train_bpe(&t, 290).merges);
    }

    #[test]
    fn save_load_identical() {
        let t = b"the cat sat on the mat. the cat ate.\nthe end".repeat(8);
        let b = train_bpe(&t, 290);
        let dir = std::env::temp_dir().join(format!("rllm_bpe_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("bpe.bin");
        b.save(&p).unwrap();
        let l = Bpe::load(&p).unwrap();
        assert_eq!(l.merges, b.merges);
        assert_eq!(l.vocab, b.vocab);
        assert_eq!(l.encode(&t), b.encode(&t));
        assert_eq!(l.end(), b.end());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let dir = std::env::temp_dir().join(format!("rllm_bpe_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("bad.bin");
        std::fs::write(&p, b"XXXX\0\0\0\0\0\0\0\0").unwrap();
        assert!(Bpe::load(&p).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
