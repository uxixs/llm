use rllm_prep::extract::{run, ExtractConfig};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: extract --rs <posts.zst|.jsonl>... --rc <comments.zst|.jsonl>... [--out data] [--max-pretrain-mb 40]\n\
  --rs/--rc accept .zst (Pushshift) and .jsonl (plain, e.g. Arctic Shift); one or more paths each (several values or repeated flags)";

fn main() -> ExitCode {
    let mut rs: Vec<PathBuf> = vec![];
    let mut rc: Vec<PathBuf> = vec![];
    let mut out = PathBuf::from("data");
    let mut mb: u64 = 40;
    let mut args = std::env::args().skip(1).peekable();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rs" | "--rc" => {
                let list = if a == "--rs" { &mut rs } else { &mut rc };
                while let Some(p) = args.next_if(|n| !n.starts_with("--")) {
                    list.push(PathBuf::from(p));
                }
            }
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
    if rs.is_empty() || rc.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    match run(&ExtractConfig { rs_files: rs, rc_files: rc, out_dir: out.clone(), max_pretrain_bytes: mb * 1024 * 1024 }) {
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
                "lines: {} read, {} kept, {} malformed skipped",
                s.lines_read, s.lines_kept, s.skipped
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
