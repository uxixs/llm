use rllm::infer::{run_chat, ChatArgs, CHAT_USAGE};
use rllm::tui::{find_latest_model, init_console, list_runs, pick_run, run_tui, Style, CANCEL};
use std::io::{stderr, stdin, stdout, IsTerminal};
use std::path::Path;
use std::process::ExitCode;

const NO_MODEL: &str = "모델이 없어요. 먼저 `train --name <run>`으로 학습하세요\n\
                        No model found under runs/. Train one first: train --name <run>";

fn main() -> ExitCode {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    let batch = argv.iter().any(|a| a == "--batch");
    // Interactive start without --model: let the user choose a run; otherwise the newest.
    let interactive = !batch && stdin().is_terminal() && stdout().is_terminal();
    let mut style: Option<Style> = None;
    if !argv.iter().any(|a| a == "--model") {
        let model = if interactive {
            let runs = list_runs(Path::new("runs"));
            match runs.len() {
                0 => None,
                1 => Some(runs[0].model.clone()),
                _ => {
                    let s = init_console();
                    style = Some(s);
                    match pick_run(&runs, &mut stdin().lock(), &mut stdout().lock(), &s) {
                        Ok(Some(i)) => Some(runs[i].model.clone()),
                        Ok(None) => return ExitCode::SUCCESS,
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
            }
        } else {
            find_latest_model(Path::new("runs"))
        };
        match model {
            Some(m) => {
                argv.push("--model".into());
                argv.push(m.to_string_lossy().into_owned());
            }
            None => {
                eprintln!("{NO_MODEL}");
                return ExitCode::FAILURE;
            }
        }
    }
    // A run folder keeps the tokenizer it was trained with; prefer it over data/.
    if !argv.iter().any(|a| a == "--tokenizer") {
        let model = argv.iter().position(|a| a == "--model").and_then(|i| argv.get(i + 1));
        if let Some(tok) = model.and_then(|m| Path::new(m).parent()).map(|d| d.join("tokenizer.bin")) {
            if tok.is_file() {
                argv.push("--tokenizer".into());
                argv.push(tok.to_string_lossy().into_owned());
            }
        }
    }
    let args = match ChatArgs::parse(argv.into_iter()) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n{CHAT_USAGE}");
            return ExitCode::from(2);
        }
    };
    let result = if args.batch.is_some() {
        run_chat(&args, &mut stdin().lock(), &mut stdout().lock(), &mut stderr().lock())
    } else {
        let style = style.unwrap_or_else(init_console);
        run_tui(args, &mut stdin().lock(), &mut stdout().lock(), style, &CANCEL)
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
