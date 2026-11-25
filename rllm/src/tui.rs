//! Terminal chat app: ANSI rendering, console setup, Ctrl+C, slash commands, and the REPL loop.
//! Std only; the few Windows console calls are declared by hand in `sys`.

use crate::backend::{self, Backend};
use crate::bpe::Bpe;
use crate::infer::{build_prompt, generate_cancellable, step, ChatArgs, Kernel, KvCache};
use crate::model::{Config, Params};
use crate::rng::Rng;
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

/// Set by the Ctrl+C handler (Windows). The REPL treats it as "stop reply" while generating
/// and as "quit" at the prompt.
pub static CANCEL: AtomicBool = AtomicBool::new(false);

#[cfg(windows)]
mod sys {
    use super::CANCEL;
    use std::ffi::c_void;
    use std::sync::atomic::Ordering::Relaxed;

    type Handle = *mut c_void;
    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    const CP_UTF8: u32 = 65001;
    const CTRL_C_EVENT: u32 = 0;
    const CTRL_BREAK_EVENT: u32 = 1;

    extern "system" {
        fn GetStdHandle(n_std_handle: u32) -> Handle;
        fn GetConsoleMode(h: Handle, mode: *mut u32) -> i32;
        fn SetConsoleMode(h: Handle, mode: u32) -> i32;
        fn SetConsoleOutputCP(cp: u32) -> i32;
        fn SetConsoleCP(cp: u32) -> i32;
        fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
    }

    /// Only stores to a static and returns TRUE (handled) for Ctrl+C / Ctrl+Break.
    unsafe extern "system" fn on_ctrl(kind: u32) -> i32 {
        if kind == CTRL_C_EVENT || kind == CTRL_BREAK_EVENT {
            CANCEL.store(true, Relaxed);
            1
        } else {
            0
        }
    }

    pub fn utf8_code_pages() {
        // SAFETY: plain kernel32 calls with integer arguments; failure is ignored.
        unsafe {
            SetConsoleOutputCP(CP_UTF8);
            SetConsoleCP(CP_UTF8);
        }
    }

    /// Turns on ANSI escape processing for stdout; false if it is not a console or refuses.
    pub fn enable_vt() -> bool {
        // SAFETY: GetStdHandle takes an integer; the returned handle is only passed back to
        // console calls, and `mode` is a valid, writable u32 for GetConsoleMode.
        unsafe {
            let h = GetStdHandle(STD_OUTPUT_HANDLE);
            if h.is_null() || h as isize == -1 {
                return false;
            }
            let mut mode = 0u32;
            if GetConsoleMode(h, &mut mode) == 0 {
                return false;
            }
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
        }
    }

    pub fn install_ctrl_c() {
        // SAFETY: `on_ctrl` has the PHANDLER_ROUTINE signature, lives for the whole process,
        // and touches only an atomic static.
        unsafe {
            SetConsoleCtrlHandler(Some(on_ctrl), 1);
        }
    }
}

#[cfg(not(windows))]
mod sys {
    pub fn utf8_code_pages() {}
    pub fn enable_vt() -> bool {
        true
    }
    /// No-op: generation is not cancellable here; Ctrl+C ends the process as usual.
    pub fn install_ctrl_c() {}
}

/// Set up UTF-8 code pages and the Ctrl+C handler, and decide whether to use colour:
/// only on a real terminal that accepts ANSI codes, and never when `NO_COLOR` is set.
pub fn init_console() -> Style {
    sys::utf8_code_pages();
    sys::install_ctrl_c();
    let tty = std::io::stdout().is_terminal();
    let no_color = std::env::var_os("NO_COLOR").is_some();
    Style { color: tty && !no_color && sys::enable_vt() }
}

#[derive(Clone, Copy, Debug)]
pub struct Style {
    pub color: bool,
}

impl Style {
    fn wrap(&self, code: &str, s: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn bold(&self, s: &str) -> String {
        self.wrap("1", s)
    }
    pub fn bold_cyan(&self, s: &str) -> String {
        self.wrap("1;36", s)
    }
    pub fn bold_magenta(&self, s: &str) -> String {
        self.wrap("1;35", s)
    }
    pub fn dim(&self, s: &str) -> String {
        self.wrap("2", s)
    }
    pub fn red(&self, s: &str) -> String {
        self.wrap("31", s)
    }
}

#[derive(Debug, PartialEq)]
pub enum Command {
    Help,
    Reset,
    Quit,
    Temp(f32),
    TopP(f32),
    Model(String),
    /// Malformed arguments; carries the message to show.
    Invalid(String),
    Unknown(String),
    Text(String),
}

pub fn parse_command(line: &str) -> Command {
    let line = line.trim();
    if !line.starts_with('/') {
        return Command::Text(line.to_string());
    }
    let (cmd, rest) = match line.split_once(char::is_whitespace) {
        Some((c, r)) => (c, r.trim()),
        None => (line, ""),
    };
    let one_float = |name: &str, ok: fn(f32) -> bool, range: &str| -> Result<f32, String> {
        if rest.is_empty() || rest.split_whitespace().count() != 1 {
            return Err(format!("usage: {name} <number in {range}>"));
        }
        match rest.parse::<f32>() {
            Ok(v) if v.is_finite() && ok(v) => Ok(v),
            _ => Err(format!("{name}: {rest:?} is out of range, expected a number in {range}")),
        }
    };
    match cmd {
        "/help" => Command::Help,
        "/reset" => Command::Reset,
        "/quit" | "/exit" => Command::Quit,
        "/temp" => match one_float("/temp", |v| (0.0..=5.0).contains(&v), "0..5") {
            Ok(v) => Command::Temp(v),
            Err(e) => Command::Invalid(e),
        },
        "/top_p" => match one_float("/top_p", |v| v > 0.0 && v <= 1.0, "(0, 1]") {
            Ok(v) => Command::TopP(v),
            Err(e) => Command::Invalid(e),
        },
        "/model" if rest.is_empty() => Command::Invalid("usage: /model <path to model.bin>".into()),
        "/model" => Command::Model(rest.to_string()),
        _ => Command::Unknown(cmd.to_string()),
    }
}

/// `[N tok · X tok/s · ctx used/ctx]`
pub fn fmt_footer(n: usize, secs: f64, used: usize, ctx: usize) -> String {
    let rate = if secs > 0.0 { n as f64 / secs } else { 0.0 };
    format!("[{n} tok · {rate:.1} tok/s · ctx {used}/{ctx}]")
}

pub fn fmt_header(
    style: &Style,
    model: &Path,
    nparams: usize,
    cfg: &Config,
    backend: Backend,
    threads: usize,
) -> String {
    let params = if nparams >= 1_000_000 {
        format!("{:.2}M", nparams as f64 / 1e6)
    } else {
        format!("{nparams}")
    };
    format!(
        "{}\n{} · {} params\nd={} layers={} ctx={} vocab={} bitnet={}\nbackend={} threads={}\n{}\n",
        style.bold("rllm chat"),
        model.display(),
        params,
        cfg.d,
        cfg.layers,
        cfg.ctx,
        cfg.vocab,
        cfg.bitnet,
        format!("{backend:?}").to_lowercase(),
        threads,
        style.dim("/help for commands"),
    )
}

/// Newest `<dir>/*/model.bin` by modification time.
pub fn find_latest_model(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| {
            let p = e.ok()?.path().join("model.bin");
            let t = std::fs::metadata(&p).ok()?.modified().ok()?;
            Some((t, p))
        })
        .max_by_key(|(t, _)| *t)
        .map(|(_, p)| p)
}

const HELP: &str = "\
/help          show this help
/reset         clear the conversation
/temp X        sampling temperature, 0..5 (0 = greedy)
/top_p X       nucleus sampling, (0, 1]
/model PATH    load another model.bin (same tokenizer)
/quit, /exit   leave (also Ctrl+C at the prompt, or Ctrl+Z then Enter)
Ctrl+C while the assistant is replying stops that reply.";

pub struct Chat {
    pub params: Params,
    pub cfg: Config,
    pub args: ChatArgs,
    pub cache: KvCache,
    pub bpe: Bpe,
    pub rng: Rng,
    pub history: Vec<(Vec<u16>, Vec<u16>)>,
}

fn load_model(path: &Path) -> Result<(Config, Params), String> {
    Params::load(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::InvalidData {
            e.to_string()
        } else {
            format!("{}: {e}", path.display())
        }
    })
}

impl Chat {
    pub fn load(args: ChatArgs) -> Result<Chat, String> {
        backend::check_supported(args.backend)?;
        backend::init_threads(args.threads)?;
        if args.kernel == Kernel::Ternary {
            return Err("--kernel ternary is not implemented yet".into());
        }
        let (cfg, params) = load_model(&args.model)?;
        let bpe = Bpe::load(&args.tokenizer).map_err(|e| format!("{}: {e}", args.tokenizer.display()))?;
        if bpe.vocab.len() != cfg.vocab {
            return Err(format!(
                "tokenizer has {} tokens but the model expects {}",
                bpe.vocab.len(),
                cfg.vocab
            ));
        }
        Ok(Chat {
            cache: KvCache::new(&cfg),
            rng: Rng::new(args.seed),
            history: Vec::new(),
            params,
            cfg,
            args,
            bpe,
        })
    }

    /// Switch to another model; on any error the current one is untouched.
    pub fn swap_model(&mut self, path: &Path) -> Result<(), String> {
        let (cfg, params) = load_model(path)?;
        if cfg.vocab != self.bpe.vocab.len() {
            return Err(format!(
                "model vocab {} does not match the tokenizer ({})",
                cfg.vocab,
                self.bpe.vocab.len()
            ));
        }
        self.cache = KvCache::new(&cfg);
        self.cfg = cfg;
        self.params = params;
        self.args.model = path.to_path_buf();
        self.history.clear();
        Ok(())
    }

    /// One user turn: stream the reply to `out`, then a footer. A set `cancel` flag stops the
    /// reply (the partial text is kept in history and marked `[stopped]`); the flag is consumed.
    pub fn turn(
        &mut self,
        out: &mut dyn Write,
        style: &Style,
        text: &str,
        cancel: &AtomicBool,
    ) -> std::io::Result<()> {
        let user = self.bpe.encode(text.as_bytes());
        let prompt = build_prompt(&self.history, &user, &self.bpe, self.cfg.ctx, 64);
        writeln!(out, "{}", style.bold_magenta("Assistant"))?;
        out.flush()?;
        let (params, cfg, be) = (&self.params, &self.cfg, self.args.backend);
        let cache = &mut self.cache;
        let mut stepper = |tok: u16, pos: usize| step(params, cfg, be, cache, tok, pos);
        let mut emit = |s: &str| {
            out.write_all(s.as_bytes())?;
            out.flush()
        };
        let (reply, secs) = generate_cancellable(
            &mut stepper,
            cfg.ctx,
            &self.args,
            &self.bpe,
            &prompt,
            &mut self.rng,
            cancel,
            &mut emit,
        )?;
        let stopped = cancel.swap(false, Relaxed);
        writeln!(out)?;
        if stopped {
            writeln!(out, "{}", style.red("[stopped]"))?;
        }
        let used = prompt.len() + reply.len();
        writeln!(out, "{}\n", style.dim(&fmt_footer(reply.len(), secs, used, self.cfg.ctx)))?;
        self.history.push((user, reply));
        Ok(())
    }
}

/// Interactive chat loop. Reads lines lossily from `input`, writes everything to `out`.
/// Returns on `/quit`, EOF, or Ctrl+C at the prompt (`cancel` set while not generating).
pub fn run_tui(
    args: ChatArgs,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    style: Style,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let io = |e: std::io::Error| e.to_string();
    let mut chat = Chat::load(args)?;
    let header = fmt_header(
        &style,
        &chat.args.model,
        chat.params.num_params(),
        &chat.cfg,
        chat.args.backend,
        chat.args.threads,
    );
    write!(out, "{header}\n").map_err(io)?;
    let mut buf = Vec::new();
    loop {
        write!(out, "{} ", style.bold_cyan("You ›")).map_err(io)?;
        out.flush().map_err(io)?;
        buf.clear();
        let n = match input.read_until(b'\n', &mut buf) {
            Ok(n) => n,
            Err(_) if cancel.load(Relaxed) => 0,
            Err(e) => return Err(e.to_string()),
        };
        if n == 0 || cancel.swap(false, Relaxed) {
            writeln!(out).map_err(io)?;
            break;
        }
        let line = String::from_utf8_lossy(&buf).into_owned();
        if line.trim().is_empty() {
            continue;
        }
        match parse_command(&line) {
            Command::Quit => break,
            Command::Help => writeln!(out, "{}\n", style.dim(HELP)).map_err(io)?,
            Command::Reset => {
                chat.history.clear();
                writeln!(out, "{}\n", style.dim("— new conversation —")).map_err(io)?;
            }
            Command::Temp(v) => {
                chat.args.temp = v;
                writeln!(out, "{}\n", style.dim(&format!("temperature set to {v}"))).map_err(io)?;
            }
            Command::TopP(v) => {
                chat.args.top_p = v;
                writeln!(out, "{}\n", style.dim(&format!("top_p set to {v}"))).map_err(io)?;
            }
            Command::Model(path) => match chat.swap_model(Path::new(&path)) {
                Ok(()) => writeln!(
                    out,
                    "{}\n",
                    style.dim(&format!(
                        "loaded {path} ({} params); new conversation",
                        chat.params.num_params()
                    ))
                )
                .map_err(io)?,
                Err(e) => writeln!(
                    out,
                    "{}\n",
                    style.red(&format!("could not load {path}: {e} (keeping the current model)"))
                )
                .map_err(io)?,
            },
            Command::Invalid(msg) => writeln!(out, "{}\n", style.red(&msg)).map_err(io)?,
            Command::Unknown(c) => writeln!(
                out,
                "{}\n",
                style.red(&format!("Unknown command {c}. Type /help for commands."))
            )
            .map_err(io)?,
            Command::Text(text) => chat.turn(out, &style, &text, cancel).map_err(io)?,
        }
    }
    out.flush().map_err(io)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::train_bpe;
    use crate::infer::Kernel;
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::{Duration, SystemTime};

    #[test]
    fn parse_command_all() {
        assert_eq!(parse_command("/help"), Command::Help);
        assert_eq!(parse_command("  /reset  "), Command::Reset);
        assert_eq!(parse_command("/quit"), Command::Quit);
        assert_eq!(parse_command("/exit"), Command::Quit);
        assert_eq!(parse_command("/temp 0.5"), Command::Temp(0.5));
        assert_eq!(parse_command("/temp 0"), Command::Temp(0.0));
        assert_eq!(parse_command("/top_p 0.9"), Command::TopP(0.9));
        assert_eq!(parse_command("/top_p 1"), Command::TopP(1.0));
        assert_eq!(
            parse_command("/model runs/a b/model.bin"),
            Command::Model("runs/a b/model.bin".into())
        );
        for bad in [
            "/temp", "/temp x", "/temp -1", "/temp 99", "/temp nan", "/temp 1 2", "/top_p 0",
            "/top_p 1.5", "/top_p", "/top_p abc", "/model",
        ] {
            assert!(matches!(parse_command(bad), Command::Invalid(_)), "{bad}");
        }
        assert_eq!(parse_command("/frobnicate 1"), Command::Unknown("/frobnicate".into()));
        assert_eq!(parse_command("hello there"), Command::Text("hello there".into()));
        assert_eq!(parse_command("  안녕  "), Command::Text("안녕".into()));
    }

    #[test]
    fn find_latest_model_picks_newest() {
        let dir = std::env::temp_dir().join(format!("rllm_tui_latest_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(find_latest_model(&dir.join("nope")), None);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(find_latest_model(&dir), None);
        let now = SystemTime::now();
        for (name, age) in [("old", 1000), ("new", 10), ("older", 5000)] {
            let d = dir.join(name);
            std::fs::create_dir_all(&d).unwrap();
            let f = std::fs::File::create(d.join("model.bin")).unwrap();
            f.set_modified(now - Duration::from_secs(age)).unwrap();
        }
        // A newer dir without a model.bin must be ignored.
        std::fs::create_dir_all(dir.join("empty")).unwrap();
        assert_eq!(find_latest_model(&dir), Some(dir.join("new").join("model.bin")));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn no_color_emits_no_escapes() {
        let s = Style { color: false };
        let cfg = tiny_cfg();
        let all = [
            s.bold_cyan("a"),
            s.bold_magenta("b"),
            s.dim("c"),
            s.red("d"),
            fmt_header(&s, Path::new("runs/x/model.bin"), 1234, &cfg, Backend::Scalar, 4),
        ];
        for t in &all {
            assert!(!t.contains('\x1b'), "{t:?}");
        }
        assert_eq!(all[0], "a");
        let c = Style { color: true };
        assert!(c.bold_cyan("a").contains("\x1b[") && c.bold_cyan("a").contains('a'));
        assert!(c.dim("x").ends_with("\x1b[0m"));
    }

    #[test]
    fn footer_format() {
        assert_eq!(fmt_footer(12, 0.5, 40, 256), "[12 tok · 24.0 tok/s · ctx 40/256]");
        assert_eq!(fmt_footer(0, 0.0, 7, 256), "[0 tok · 0.0 tok/s · ctx 7/256]");
    }

    #[test]
    fn header_mentions_everything() {
        let h = fmt_header(
            &Style { color: false },
            Path::new("runs/x/model.bin"),
            1_049_728,
            &tiny_cfg(),
            Backend::Avx512Mt,
            4,
        );
        for want in [
            "rllm chat", "runs/x/model.bin", "1.05M", "d=16", "layers=2", "ctx=128", "vocab=300",
            "bitnet=false", "avx512mt", "threads=4", "/help for commands",
        ] {
            assert!(h.contains(want), "missing {want:?} in {h}");
        }
    }

    fn tiny_cfg() -> Config {
        Config { vocab: 300, d: 16, layers: 2, heads: 2, ctx: 128, softcap: 5.0, mtp: 1, bitnet: false }
    }

    fn tiny_files(name: &str, seed: u64) -> (PathBuf, PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("rllm_tui_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = tiny_cfg();
        let model = dir.join("model.bin");
        Params::init(&cfg, &mut Rng::new(seed)).save(&cfg, &model).unwrap();
        let tok = dir.join("tok.bin");
        train_bpe(b"hello world, hello there. the quick brown fox", 300).save(&tok).unwrap();
        (dir, model, tok)
    }

    fn args(model: PathBuf, tok: PathBuf) -> ChatArgs {
        ChatArgs {
            model,
            tokenizer: tok,
            backend: Backend::Scalar,
            temp: 0.8,
            top_p: 0.9,
            kernel: Kernel::Fp32,
            batch: None,
            seed: 1,
            threads: 4,
        }
    }

    #[test]
    fn run_tui_scripted() {
        let (dir, model, tok) = tiny_files("script", 9);
        let mut input = std::io::Cursor::new(b"hi\n/temp 0\n\nhello\n/reset\n/quit\nignored\n".to_vec());
        let mut out = Vec::new();
        let cancel = AtomicBool::new(false);
        run_tui(args(model, tok), &mut input, &mut out, Style { color: false }, &cancel).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(!out.contains('\x1b'), "{out}");
        assert_eq!(out.matches("tok/s").count(), 2, "{out}");
        assert_eq!(out.matches("Assistant").count(), 2, "{out}");
        assert!(out.contains("rllm chat") && out.contains("/help for commands"), "{out}");
        assert!(out.contains("temperature") && out.contains("new conversation"), "{out}");
        assert!(!out.contains("ignored"), "{out}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn run_tui_commands_and_eof() {
        let (dir, model, tok) = tiny_files("cmds", 9);
        let (dir2, model2, _) = tiny_files("cmds_other", 10);
        let script = format!(
            "/bogus\n/temp 99\n/help\n/model {}\n/model no/such.bin\nhi\n",
            model2.display()
        );
        let mut out = Vec::new();
        let cancel = AtomicBool::new(false);
        // No /quit: EOF must end cleanly. Invalid UTF-8 input must not abort.
        let mut bytes = script.into_bytes();
        bytes.extend_from_slice(b"caf\xff\n");
        let mut input = std::io::Cursor::new(bytes);
        run_tui(args(model, tok), &mut input, &mut out, Style { color: false }, &cancel).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("Unknown command /bogus"), "{out}");
        assert!(out.contains("/reset") && out.contains("/quit"), "{out}");
        assert!(out.contains("loaded"), "{out}");
        assert!(out.contains("no/such.bin"), "{out}");
        assert_eq!(out.matches("tok/s").count(), 2, "{out}");
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(dir2);
    }

    #[test]
    fn cancel_flag_stops_generation() {
        let (dir, model, tok) = tiny_files("cancel", 9);
        let mut chat = Chat::load(args(model, tok)).unwrap();
        let cancel = AtomicBool::new(true);
        let mut out = Vec::new();
        chat.turn(&mut out, &Style { color: false }, "hello", &cancel).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("[stopped]"), "{out}");
        assert_eq!(chat.history.len(), 1);
        assert!(chat.history[0].1.is_empty());
        assert!(!cancel.load(Relaxed), "flag is consumed");
        let _ = std::fs::remove_dir_all(dir);
    }
}
