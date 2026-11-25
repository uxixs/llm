use rllm_prep::extract::{run, ExtractConfig};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: extract --subs <dir with the six .zst dumps> [--out data] [--max-pretrain-mb 40]";

fn main() -> ExitCode {
    let mut subs: Option<PathBuf> = None;
    let mut out = PathBuf::from("data");
    let mut mb: u64 = 40;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--subs" => subs = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from).unwrap_or(out),
            "--max-pretrain-mb" => match args.next().and_then(|s| s.parse().ok()) {
                Some(n) => mb = n,
                None => {
                    eprintln!("--max-pretrain-mb needs a number\n{USAGE}");
                    return ExitCode::from(2);
                }
            },
            _ => {
                eprintln!("unknown argument {a}\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(subs_dir) = subs else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match run(&ExtractConfig { subs_dir, out_dir: out.clone(), max_pretrain_bytes: mb * 1024 * 1024 }) {
        Ok(s) => {
            println!(
                "pretrain_train.txt {:.1} MB, pretrain_val.txt {:.1} MB",
                s.pretrain_train_bytes as f64 / 1e6,
                s.pretrain_val_bytes as f64 / 1e6
            );
            println!(
                "SFT pairs: {} total ({} train, {} val); eli5 {}, nostupidquestions {}",
                s.sft_total, s.sft_train, s.sft_val, s.sft_eli5, s.sft_nsq
            );
            println!(
                "skipped malformed lines: {} (sft pass), {} (pretrain pass)",
                s.skipped_sft, s.skipped_pretrain
            );
            println!("wrote to {}", out.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
