//! Writes the same intermediate files as `extract`, from TinyStories (pretraining)
//! and DailyDialog (multi-turn SFT): pretrain_{train,val}.txt and sft_{train,val}.jsonl.

use rllm::rng::Rng;
use rllm_prep::synth::{filler_pool, identity_dialogue, memory_dialogue, TRAIN};
use rllm_prep::tiny::{dialogue_views, parse_dialogue, split_stories};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: extract_tiny --stories <train.txt> --stories-val <valid.txt> \
--dd-train <dialogues_train.txt> --dd-val <dialogues_validation.txt> [--out data-tiny] [--max-pretrain-mb 100] \
[--synth-memory N] [--synth-identity N] [--seed 1]";

/// Appends synthetic memory and identity dialogues (train names and places only).
fn append_synth(dst: &Path, dd: &Path, memory: usize, identity: usize, seed: u64) -> io::Result<()> {
    let dialogues: Vec<Vec<String>> =
        std::fs::read_to_string(dd)?.lines().map(parse_dialogue).filter(|d| !d.is_empty()).collect();
    let pool = filler_pool(&dialogues);
    let mut f = BufWriter::new(std::fs::OpenOptions::new().append(true).open(dst)?);
    let mut rng = Rng::new(seed);
    for i in 0..memory + identity {
        let s = if i < memory { memory_dialogue(&mut rng, TRAIN, &pool) } else { identity_dialogue(&mut rng, &pool) };
        serde_json::to_writer(&mut f, &serde_json::json!({ "turns": s.turns }))?;
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

fn write_dialogues(src: &Path, dst: &Path) -> io::Result<(usize, usize)> {
    let text = std::fs::read_to_string(src)?;
    let mut f = BufWriter::new(std::fs::File::create(dst)?);
    let (mut dialogues, mut records) = (0, 0);
    for line in text.lines() {
        let turns = parse_dialogue(line);
        if turns.is_empty() {
            continue;
        }
        dialogues += 1;
        for v in dialogue_views(&turns) {
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
    let (mut memory, mut identity, mut seed) = (0usize, 0usize, 1u64);
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
            "--max-pretrain-mb" | "--synth-memory" | "--synth-identity" | "--seed" => match v.parse::<u64>() {
                Ok(n) => match k.as_str() {
                    "--max-pretrain-mb" => mb = n as usize,
                    "--synth-memory" => memory = n as usize,
                    "--synth-identity" => identity = n as usize,
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
        for (src, split) in [(&dt, "train"), (&dv, "val")] {
            let (d, r) = write_dialogues(src, &out.join(format!("sft_{split}.jsonl")))?;
            println!("sft_{split}.jsonl: {d} dialogues -> {r} records");
        }
        if memory + identity > 0 {
            append_synth(&out.join("sft_train.jsonl"), &dt, memory, identity, seed)?;
            println!("sft_train.jsonl: + {memory} memory and {identity} identity dialogues");
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
