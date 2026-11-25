//! Memory and identity test for a trained chat model. Plays fixed scripted conversations
//! (test-split names and places, never seen in training) turn by turn through the real chat
//! path, letting the model write its own replies, and checks whether the last reply contains
//! the fact the user gave earlier, or the bot's name.
//!
//! usage: memtest --model runs/<run>/model.bin --dd-val <dialogues_validation.txt>
//!        [--tokenizer path] [--n 100] [--seed 99] [--backend avx512mt] [--show 3]

use rllm::backend::Backend;
use rllm::infer::{ChatArgs, Kernel};
use rllm::rng::Rng;
use rllm::tui::{Chat, Style};
use rllm_prep::synth::{filler_pool, identity_dialogue, memory_dialogue, Synth, TEST};
use rllm_prep::tiny::parse_dialogue;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

/// Plays the user turns of `s`; returns the model's last reply and whether it is correct.
fn play(chat: &mut Chat, s: &Synth) -> Result<(String, bool), String> {
    chat.history.clear();
    let never = AtomicBool::new(false);
    let style = Style { color: false };
    let mut sink = Vec::new();
    for user in s.turns.iter().step_by(2) {
        chat.turn(&mut sink, &style, user, &never).map_err(|e| e.to_string())?;
    }
    let reply = chat.history.last().map(|(_, a)| String::from_utf8_lossy(&chat.bpe.decode(a)).into_owned());
    let reply = reply.unwrap_or_default();
    let ok = reply.to_lowercase().contains(&s.answer.to_lowercase());
    Ok((reply, ok))
}

fn main() -> ExitCode {
    let (mut model, mut tok, mut dd) = (None::<PathBuf>, None::<PathBuf>, None::<PathBuf>);
    let (mut n, mut seed, mut show, mut backend) = (100usize, 99u64, 3usize, "avx512mt".to_string());
    let mut args = std::env::args().skip(1);
    while let Some(k) = args.next() {
        let Some(v) = args.next() else {
            eprintln!("{k} needs a value");
            return ExitCode::from(2);
        };
        let num = |v: &str| v.parse::<u64>().map_err(|_| format!("bad {k} {v}"));
        let r = match k.as_str() {
            "--model" => Ok(model = Some(v.into())),
            "--tokenizer" => Ok(tok = Some(v.into())),
            "--dd-val" => Ok(dd = Some(v.into())),
            "--backend" => Ok(backend = v),
            "--n" => num(&v).map(|x| n = x as usize),
            "--seed" => num(&v).map(|x| seed = x),
            "--show" => num(&v).map(|x| show = x as usize),
            _ => Err(format!("unknown argument {k}")),
        };
        if let Err(e) = r {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    }
    let (Some(model), Some(dd)) = (model, dd) else {
        eprintln!("usage: memtest --model <model.bin> --dd-val <dialogues_validation.txt> [--tokenizer p] [--n 100]");
        return ExitCode::from(2);
    };
    let tok = tok.unwrap_or_else(|| model.with_file_name("tokenizer.bin"));
    let run = || -> Result<(), String> {
        let backend: Backend = backend.parse().map_err(|e| format!("{e:?}"))?;
        let args = ChatArgs {
            model: model.clone(),
            tokenizer: tok.clone(),
            backend,
            temp: 0.0,
            top_p: 1.0,
            kernel: Kernel::Fp32,
            batch: None,
            seed: 1,
            threads: 4,
        };
        let mut chat = Chat::load(args)?;
        let text = std::fs::read_to_string(&dd).map_err(|e| format!("{}: {e}", dd.display()))?;
        let dialogues: Vec<Vec<String>> = text.lines().map(parse_dialogue).filter(|d| !d.is_empty()).collect();
        let pool = filler_pool(&dialogues);
        let mut rng = Rng::new(seed);
        for (label, identity) in [("memory", false), ("identity", true)] {
            let mut correct = 0;
            for i in 0..n {
                let s = if identity { identity_dialogue(&mut rng, &pool) } else { memory_dialogue(&mut rng, TEST, &pool) };
                let (reply, ok) = play(&mut chat, &s)?;
                correct += ok as usize;
                if i < show {
                    println!("[{label} {i}] {} | want {:?} | got {:?}", if ok { "OK  " } else { "MISS" }, s.answer, reply);
                }
            }
            println!("{label}: {correct}/{n} = {:.0}%", 100.0 * correct as f64 / n as f64);
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
