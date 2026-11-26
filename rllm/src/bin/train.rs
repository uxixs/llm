use rllm::train::{run, Args, USAGE};
use std::process::ExitCode;

/// Keeps Windows from idle-sleeping while training runs; the request ends with the process.
#[cfg(windows)]
fn stay_awake() {
    extern "system" {
        fn SetThreadExecutionState(flags: u32) -> u32;
    }
    const ES_CONTINUOUS: u32 = 0x8000_0000;
    const ES_SYSTEM_REQUIRED: u32 = 0x0000_0001;
    // SAFETY: plain kernel32 call with integer flags; failure (0) only means no request.
    if unsafe { SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) } == 0 {
        eprintln!("note: could not ask Windows to stay awake during training");
    }
}

#[cfg(not(windows))]
fn stay_awake() {}

fn main() -> ExitCode {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    stay_awake();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
