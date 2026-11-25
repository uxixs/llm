use rllm::infer::{run_chat, ChatArgs, CHAT_USAGE};
use std::io::{stderr, stdin, stdout};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args = match ChatArgs::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n{CHAT_USAGE}");
            return ExitCode::from(2);
        }
    };
    match run_chat(&args, &mut stdin().lock(), &mut stdout().lock(), &mut stderr().lock()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
