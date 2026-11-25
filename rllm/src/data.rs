//! Tokenized data files and seeded batchers.
//!
//! Pretrain file: raw u16 LE tokens. SFT file: records of `u16 len` + `len` u16 LE tokens.

use crate::rng::Rng;
use std::io::{self, Error, ErrorKind};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Batch {
    pub b: usize,
    pub t: usize,
    /// Extra lookahead targets per row (for multi-token prediction heads).
    pub extra: usize,
    /// `b*t`
    pub inputs: Vec<u16>,
    /// `b*(t+extra)`; row r target j = token at window position j+1.
    pub targets: Vec<u16>,
    /// `b*t`, for the main targets only.
    pub mask: Vec<f32>,
}

fn read_u16s(path: &Path) -> io::Result<Vec<u16>> {
    let d = std::fs::read(path)?;
    if d.len() % 2 != 0 {
        return Err(Error::new(ErrorKind::InvalidData, "odd byte count in u16 file"));
    }
    Ok(d.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

pub struct PretrainBatcher {
    tokens: Vec<u16>,
}

impl PretrainBatcher {
    pub fn load(path: &Path) -> io::Result<PretrainBatcher> {
        Ok(PretrainBatcher { tokens: read_u16s(path)? })
    }

    pub fn from_tokens(tokens: Vec<u16>) -> PretrainBatcher {
        PretrainBatcher { tokens }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// `b` random windows of `t+1+extra` tokens.
    pub fn next(&mut self, rng: &mut Rng, b: usize, t: usize, extra: usize) -> Batch {
        let w = t + 1 + extra;
        assert!(self.tokens.len() >= w, "pretrain data has {} tokens, need {w}", self.tokens.len());
        let n_starts = self.tokens.len() - w + 1;
        let mut inputs = Vec::with_capacity(b * t);
        let mut targets = Vec::with_capacity(b * (t + extra));
        for _ in 0..b {
            let s = rng.below(n_starts);
            let win = &self.tokens[s..s + w];
            inputs.extend_from_slice(&win[..t]);
            targets.extend_from_slice(&win[1..]);
        }
        Batch { b, t, extra, inputs, targets, mask: vec![1.0; b * t] }
    }
}

/// Mask over target indices `j` (predicting `tokens[j+1]`), length `tokens.len()-1`.
/// 1 iff an `assistant` token occurs at index <= j and `tokens[j+1]` lies in that
/// answer, up to and including its `end` token.
pub fn sft_mask(tokens: &[u16], assistant: u16, end: u16) -> Vec<f32> {
    let mut mask = Vec::with_capacity(tokens.len().saturating_sub(1));
    let mut in_answer = false;
    for j in 0..tokens.len().saturating_sub(1) {
        if tokens[j] == assistant {
            in_answer = true;
        }
        mask.push(if in_answer { 1.0 } else { 0.0 });
        if tokens[j + 1] == end {
            in_answer = false;
        }
    }
    mask
}

pub struct SftBatcher {
    records: Vec<Vec<u16>>,
    assistant: u16,
    end: u16,
}

impl SftBatcher {
    /// `assistant` and `end` are the special token ids (V-2, V-1).
    pub fn load(path: &Path, assistant: u16, end: u16) -> io::Result<SftBatcher> {
        let d = std::fs::read(path)?;
        let bad = || Error::new(ErrorKind::InvalidData, "truncated sft record");
        let mut records = Vec::new();
        let mut i = 0;
        while i < d.len() {
            if i + 2 > d.len() {
                return Err(bad());
            }
            let len = u16::from_le_bytes([d[i], d[i + 1]]) as usize;
            i += 2;
            if i + 2 * len > d.len() {
                return Err(bad());
            }
            records.push(
                d[i..i + 2 * len].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
            );
            i += 2 * len;
        }
        Ok(SftBatcher::from_records(records, assistant, end))
    }

    pub fn from_records(records: Vec<Vec<u16>>, assistant: u16, end: u16) -> SftBatcher {
        SftBatcher { records, assistant, end }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// One random sample per row, right-padded with `end`; mask is 0 on padding and
    /// covers only the first `t` targets (lookahead targets are never masked in).
    /// Intended use is `extra = 0`.
    pub fn next(&mut self, rng: &mut Rng, b: usize, t: usize, extra: usize) -> Batch {
        assert!(!self.records.is_empty(), "no sft records");
        let mut inputs = Vec::with_capacity(b * t);
        let mut targets = Vec::with_capacity(b * (t + extra));
        let mut mask = Vec::with_capacity(b * t);
        for _ in 0..b {
            let rec = &self.records[rng.below(self.records.len())];
            let m = sft_mask(rec, self.assistant, self.end);
            for j in 0..t {
                inputs.push(rec.get(j).copied().unwrap_or(self.end));
                mask.push(m.get(j).copied().unwrap_or(0.0));
            }
            for j in 0..t + extra {
                targets.push(rec.get(j + 1).copied().unwrap_or(self.end));
            }
        }
        Batch { b, t, extra, inputs, targets, mask }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u16 = 2046;
    const E: u16 = 2047;
    const U: u16 = 2045;

    #[test]
    fn pretrain_targets_shifted() {
        let toks: Vec<u16> = (0..500).collect();
        let mut pb = PretrainBatcher::from_tokens(toks);
        let mut rng = Rng::new(3);
        let b = pb.next(&mut rng, 4, 16, 2);
        assert_eq!((b.inputs.len(), b.targets.len(), b.mask.len()), (4 * 16, 4 * 18, 4 * 16));
        assert_eq!(b.extra, 2);
        for r in 0..4 {
            let inp = &b.inputs[r * 16..(r + 1) * 16];
            let tg = &b.targets[r * 18..(r + 1) * 18];
            for j in 0..15 {
                assert_eq!(tg[j], inp[j + 1]);
            }
            // consecutive-integer data: every target is previous + 1, lookahead included
            assert_eq!(tg[15], inp[15] + 1);
            assert_eq!(tg[17], inp[15] + 3);
            assert_eq!(tg[0], inp[0] + 1);
        }
        assert!(b.mask.iter().all(|&m| m == 1.0));
    }

    #[test]
    fn pretrain_window_can_reach_both_ends() {
        let mut pb = PretrainBatcher::from_tokens((0..10).collect());
        let mut rng = Rng::new(9);
        let (mut first, mut last) = (false, false);
        for _ in 0..200 {
            let b = pb.next(&mut rng, 1, 7, 2); // window 10 == whole data
            first |= b.inputs[0] == 0;
            last |= *b.targets.last().unwrap() == 9;
        }
        assert!(first && last);
    }

    #[test]
    fn sft_mask_only_answer() {
        let toks = [U, 1, 2, A, 10, 11, E];
        assert_eq!(sft_mask(&toks, A, E), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn sft_padding_masked() {
        let rec = vec![U, 1, 2, A, 10, 11, E];
        let mut sb = SftBatcher::from_records(vec![rec.clone()], A, E);
        let mut rng = Rng::new(1);
        let b = sb.next(&mut rng, 2, 10, 0);
        for r in 0..2 {
            let inp = &b.inputs[r * 10..(r + 1) * 10];
            let tg = &b.targets[r * 10..(r + 1) * 10];
            let m = &b.mask[r * 10..(r + 1) * 10];
            assert_eq!(&inp[..7], &rec[..]);
            assert!(inp[7..].iter().all(|&x| x == E));
            assert_eq!(&tg[..6], &rec[1..]);
            assert!(tg[6..].iter().all(|&x| x == E));
            assert_eq!(m, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn sft_extra_keeps_mask_on_main_targets_only() {
        let rec = vec![U, 1, A, 10, E];
        let mut sb = SftBatcher::from_records(vec![rec], A, E);
        let b = sb.next(&mut Rng::new(1), 1, 4, 2);
        assert_eq!((b.inputs.len(), b.targets.len(), b.mask.len()), (4, 6, 4));
        assert_eq!(b.targets, vec![1, A, 10, E, E, E]);
        assert_eq!(b.mask, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn sft_truncates_long_record() {
        let rec = vec![U, 1, A, 10, 11, 12, E];
        let mut sb = SftBatcher::from_records(vec![rec], A, E);
        let b = sb.next(&mut Rng::new(1), 1, 4, 0);
        assert_eq!(b.inputs, vec![U, 1, A, 10]);
        assert_eq!(b.targets, vec![1, A, 10, 11]);
        assert_eq!(b.mask, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn files_roundtrip() {
        let dir = std::env::temp_dir().join(format!("rllm_data_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("pre.bin");
        let toks: Vec<u16> = vec![1, 2, 300, 2047];
        std::fs::write(&p, toks.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        assert_eq!(PretrainBatcher::load(&p).unwrap().len(), 4);

        let s = dir.join("sft.bin");
        let mut bytes = Vec::new();
        for rec in [vec![U, 5, A, 6, E], vec![U, 7, A, 8, 9, E]] {
            bytes.extend_from_slice(&(rec.len() as u16).to_le_bytes());
            for t in rec {
                bytes.extend_from_slice(&t.to_le_bytes());
            }
        }
        std::fs::write(&s, &bytes).unwrap();
        assert_eq!(SftBatcher::load(&s, A, E).unwrap().len(), 2);
        bytes.pop();
        std::fs::write(&s, &bytes).unwrap();
        assert!(SftBatcher::load(&s, A, E).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
