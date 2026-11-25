use rllm::bench::{report, BenchArgs, BENCH_USAGE};
use std::io::Write;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args = match BenchArgs::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n{BENCH_USAGE}");
            return ExitCode::from(2);
        }
    };
    let md = report(&args);
    print!("{md}");
    if args.quick {
        return ExitCode::SUCCESS;
    }
    let appended = std::fs::create_dir_all("bench")
        .and_then(|()| std::fs::OpenOptions::new().create(true).append(true).open("bench/results.md"))
        .and_then(|mut f| f.write_all(format!("{md}\n").as_bytes()));
    match appended {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: bench/results.md: {e}");
            ExitCode::FAILURE
        }
    }
}
