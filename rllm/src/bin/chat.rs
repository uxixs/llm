use rllm::infer::{run_chat, ChatArgs, CHAT_USAGE};
use rllm::tui::{find_latest_model, init_console, run_tui, CANCEL};
use std::io::{stderr, stdin, stdout};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    if !argv.iter().any(|a| a == "--model") {
        match find_latest_model(Path::new("runs")) {
            Some(m) => {
                argv.push("--model".into());
                argv.push(m.to_string_lossy().into_owned());
            }
            None => {
                eprintln!(
                    "모델이 없어요. 먼저 `train --name <run>`으로 학습하세요\n\
                     No model found under runs/. Train one first: train --name <run>"
                );
                return ExitCode::FAILURE;
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
        let style = init_console();
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
