//! Writes the same intermediate files as `extract`, from TinyStories (pretraining)
//! and DailyDialog (multi-turn SFT): pretrain_{train,val}.txt and sft_{train,val}.jsonl.

use rllm::rng::Rng;
use rllm_prep::synth::{distractor_dialogue, filler_pool, identity_dialogue, memory_dialogue, TRAIN};
use rllm_prep::tiny::{casualize_users, dialogue_views, parse_dialogue, split_stories};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: extract_tiny --stories <train.txt> --stories-val <valid.txt> \
--dd-train <dialogues_train.txt> --dd-val <dialogues_validation.txt> [--out data-tiny] [--max-pretrain-mb 100] \
[--synth-memory N] [--synth-identity N] [--synth-distractor N] [--casual-pct 0..100] [--seed 1]";

/// Counts of synthetic dialogues to append to the SFT training file.
struct SynthCounts {
    memory: usize,
    identity: usize,
    distractor: usize,
}

/// Appends synthetic memory, identity and distractor dialogues (train names and places
/// only); `casual_pct` percent of them get casual user turns.
fn append_synth(dst: &Path, dd: &Path, n: &SynthCounts, casual_pct: u64, seed: u64) -> io::Result<()> {
    let dialogues: Vec<Vec<String>> =
        std::fs::read_to_string(dd)?.lines().map(parse_dialogue).filter(|d| !d.is_empty()).collect();
    let pool = filler_pool(&dialogues);
    let mut f = BufWriter::new(std::fs::OpenOptions::new().append(true).open(dst)?);
    let mut rng = Rng::new(seed);
    for i in 0..n.memory + n.identity + n.distractor {
        let s = if i < n.memory {
            memory_dialogue(&mut rng, TRAIN, &pool)
        } else if i < n.memory + n.identity {
            identity_dialogue(&mut rng, &pool)
        } else {
            distractor_dialogue(&mut rng, TRAIN, &pool)
        };
        let mut turns = s.turns;
        if (rng.below(100) as u64) < casual_pct {
            casualize_users(&mut turns);
        }
        serde_json::to_writer(&mut f, &serde_json::json!({ "turns": turns }))?;
        f.write_all(b"\n")?;
    }
    f.flush()
}
const DOC_SEP: &str = "\n\x1e\n";

fn write_stories(src: &Path, dst: &Path, max_bytes: usize) -> io::Result<(usize, usize)> {
    let text = std::fs::read(src)?;
    let text = String::from_utf8_lossy(&text);
    let mut f = BufWriter::new(std::fs::File::create(dst)?);
    let (mut n, mut bytes) = (0, 0);
    for s in split_stories(&text, true) {
        if bytes + s.len() > max_bytes {
            break;
        }
        if n > 0 {
            f.write_all(DOC_SEP.as_bytes())?;
        }
        f.write_all(s.as_bytes())?;
        n += 1;
        bytes += s.len() + DOC_SEP.len();
    }
    f.flush()?;
    Ok((n, bytes))
}

/// `casual_pct` percent of the dialogue views get casual (lowercase, unpunctuated) user turns.
fn write_dialogues(src: &Path, dst: &Path, casual_pct: u64, rng: &mut Rng) -> io::Result<(usize, usize)> {
    let text = std::fs::read_to_string(src)?;
    let mut f = BufWriter::new(std::fs::File::create(dst)?);
    let (mut dialogues, mut records) = (0, 0);
    for line in text.lines() {
        let turns = parse_dialogue(line);
        if turns.is_empty() {
            continue;
        }
        dialogues += 1;
        for mut v in dialogue_views(&turns) {
            if (rng.below(100) as u64) < casual_pct {
                casualize_users(&mut v);
            }
            serde_json::to_writer(&mut f, &serde_json::json!({ "turns": v }))?;
            f.write_all(b"\n")?;
            records += 1;
        }
    }
    f.flush()?;
    Ok((dialogues, records))
}

fn main() -> ExitCode {
    let mut a: [Option<PathBuf>; 4] = Default::default();
    let mut out = PathBuf::from("data-tiny");
    let mut mb = 100usize;
    let (mut memory, mut identity, mut distractor, mut casual_pct, mut seed) = (0usize, 0usize, 0usize, 0u64, 1u64);
    let mut args = std::env::args().skip(1);
    while let Some(k) = args.next() {
        let Some(v) = args.next() else {
            eprintln!("{k} needs a value\n{USAGE}");
            return ExitCode::from(2);
        };
        match k.as_str() {
            "--stories" => a[0] = Some(v.into()),
            "--stories-val" => a[1] = Some(v.into()),
            "--dd-train" => a[2] = Some(v.into()),
            "--dd-val" => a[3] = Some(v.into()),
            "--out" => out = v.into(),
            "--max-pretrain-mb" | "--synth-memory" | "--synth-identity" | "--synth-distractor" | "--casual-pct"
            | "--seed" => match v.parse::<u64>() {
                Ok(n) => match k.as_str() {
                    "--max-pretrain-mb" => mb = n as usize,
                    "--synth-memory" => memory = n as usize,
                    "--synth-identity" => identity = n as usize,
                    "--synth-distractor" => distractor = n as usize,
                    "--casual-pct" => casual_pct = n.min(100),
                    _ => seed = n,
                },
                Err(_) => {
                    eprintln!("bad {k} {v}");
                    return ExitCode::from(2);
                }
            },
            _ => {
                eprintln!("unknown argument {k}\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let [Some(st), Some(sv), Some(dt), Some(dv)] = a else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let run = || -> io::Result<()> {
        std::fs::create_dir_all(&out)?;
        let (n, b) = write_stories(&st, &out.join("pretrain_train.txt"), mb * 1024 * 1024)?;
        println!("pretrain_train.txt: {n} stories, {:.1} MB", b as f64 / 1e6);
        let (n, b) = write_stories(&sv, &out.join("pretrain_val.txt"), usize::MAX)?;
        println!("pretrain_val.txt: {n} stories, {:.1} MB", b as f64 / 1e6);
        // validation stays as written, so its loss compares with earlier runs
        let mut rng = Rng::new(seed.wrapping_add(7));
        for (src, split, pct) in [(&dt, "train", casual_pct), (&dv, "val", 0)] {
            let (d, r) = write_dialogues(src, &out.join(format!("sft_{split}.jsonl")), pct, &mut rng)?;
            println!("sft_{split}.jsonl: {d} dialogues -> {r} records ({pct}% casual user turns)");
        }
        let counts = SynthCounts { memory, identity, distractor };
        if memory + identity + distractor > 0 {
            append_synth(&out.join("sft_train.jsonl"), &dt, &counts, casual_pct, seed)?;
            println!(
                "sft_train.jsonl: + {memory} memory, {identity} identity, {distractor} distractor dialogues"
            );
        }
        Ok(())
    };
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
