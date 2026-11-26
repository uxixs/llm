//! Terminal chat app: ANSI rendering, console setup, Ctrl+C, slash commands, and the REPL loop.
//! Std only; the few Windows console calls are declared by hand in `sys`.
//!
//! Layout (colour terminals):
//!
//! ```text
//! ╭──────────────────────────────╮
//! │ ✻ Welcome to rllm chat!      │   welcome box
//! │   model / params / backend   │
//! ╰──────────────────────────────╯
//!
//!  > why is the sky blue?            your message: grey block
//!
//! ● Because the air scatters ...     reply: red bullet, wrapped, indented
//!   ⎿  16 tok · 494.5 tok/s · ctx 25/256
//!
//! ────────────────────────────────   input box: rule, red "> ", rule, hints
//! > _
//! ────────────────────────────────
//!   /help for commands · Ctrl+C stops a reply · temp 0.70 · top_p 0.90
//! ```
//!
//! Without colour (pipes, tests, `NO_COLOR`) the same text is printed with no escape codes
//! and no cursor movement.

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

/// Width used when the console width is unknown (pipes, tests).
pub const DEFAULT_WIDTH: usize = 80;

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

    #[repr(C)]
    #[derive(Default)]
    struct Coord {
        x: i16,
        y: i16,
    }
    #[repr(C)]
    #[derive(Default)]
    struct SmallRect {
        left: i16,
        top: i16,
        right: i16,
        bottom: i16,
    }
    #[repr(C)]
    #[derive(Default)]
    struct ScreenBufferInfo {
        size: Coord,
        cursor: Coord,
        attributes: u16,
        window: SmallRect,
        max_window: Coord,
    }

    extern "system" {
        fn GetStdHandle(n_std_handle: u32) -> Handle;
        fn GetConsoleMode(h: Handle, mode: *mut u32) -> i32;
        fn SetConsoleMode(h: Handle, mode: u32) -> i32;
        fn SetConsoleOutputCP(cp: u32) -> i32;
        fn SetConsoleCP(cp: u32) -> i32;
        fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
        fn GetConsoleScreenBufferInfo(h: Handle, info: *mut ScreenBufferInfo) -> i32;
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

    fn stdout_handle() -> Option<Handle> {
        // SAFETY: GetStdHandle takes an integer and returns a handle or null/INVALID.
        let h = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
        (!h.is_null() && h as isize != -1).then_some(h)
    }

    /// Turns on ANSI escape processing for stdout; false if it is not a console or refuses.
    pub fn enable_vt() -> bool {
        let Some(h) = stdout_handle() else { return false };
        let mut mode = 0u32;
        // SAFETY: `h` is a console handle from GetStdHandle and `mode` is a writable u32.
        unsafe {
            if GetConsoleMode(h, &mut mode) == 0 {
                return false;
            }
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
        }
    }

    /// Visible window width in columns, if stdout is a console.
    pub fn term_width() -> Option<usize> {
        let h = stdout_handle()?;
        let mut info = ScreenBufferInfo::default();
        // SAFETY: `info` is a writable, correctly laid out CONSOLE_SCREEN_BUFFER_INFO.
        if unsafe { GetConsoleScreenBufferInfo(h, &mut info) } == 0 {
            return None;
        }
        let w = info.window.right as i32 - info.window.left as i32 + 1;
        (w > 0).then_some(w as usize)
    }

    pub fn install_ctrl_c() {
        // SAFETY: `on_ctrl` has the PHANDLER_ROUTINE signature, lives for the whole process,
        // and touches only an atomic static.
        unsafe {
            SetConsoleCtrlHandler(Some(on_ctrl), 1);
        }
    }

    const STD_INPUT_HANDLE: u32 = -10i32 as u32;
    const ENABLE_PROCESSED_INPUT: u32 = 0x0001;
    const ENABLE_LINE_INPUT: u32 = 0x0002;
    const ENABLE_ECHO_INPUT: u32 = 0x0004;
    const ENABLE_VIRTUAL_TERMINAL_INPUT: u32 = 0x0200;

    /// Key-at-a-time console input: no line editing or echo, arrow keys arrive as VT
    /// sequences (`ESC [ A`), Ctrl+C arrives as byte 3. The old mode is restored on drop.
    pub struct RawInput {
        h: Handle,
        old: u32,
    }

    impl RawInput {
        pub fn enable() -> Option<RawInput> {
            // SAFETY: GetStdHandle takes an integer; the handle is only passed back to console
            // mode calls, and `old` is a writable u32.
            unsafe {
                let h = GetStdHandle(STD_INPUT_HANDLE);
                if h.is_null() || h as isize == -1 {
                    return None;
                }
                let mut old = 0u32;
                if GetConsoleMode(h, &mut old) == 0 {
                    return None;
                }
                let raw = (old & !(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT))
                    | ENABLE_VIRTUAL_TERMINAL_INPUT;
                (SetConsoleMode(h, raw) != 0).then_some(RawInput { h, old })
            }
        }
    }

    impl Drop for RawInput {
        fn drop(&mut self) {
            // SAFETY: restores the mode read from the same console handle in `enable`.
            unsafe {
                SetConsoleMode(self.h, self.old);
            }
        }
    }
}

#[cfg(not(windows))]
mod sys {
    pub fn utf8_code_pages() {}
    pub fn enable_vt() -> bool {
        true
    }
    pub fn term_width() -> Option<usize> {
        std::env::var("COLUMNS").ok()?.parse().ok()
    }
    /// No-op: generation is not cancellable here; Ctrl+C ends the process as usual.
    pub fn install_ctrl_c() {}

    /// Not implemented here: the picker falls back to typing a number.
    pub struct RawInput;
    impl RawInput {
        pub fn enable() -> Option<RawInput> {
            None
        }
    }
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

/// Usable line width: the console width minus one column (so a full line never triggers the
/// terminal's own wrap), or `DEFAULT_WIDTH` when unknown.
fn line_width(style: &Style) -> usize {
    let w = if style.color { sys::term_width() } else { None };
    w.unwrap_or(DEFAULT_WIDTH + 1).saturating_sub(1).max(20)
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
    /// Main accent: a bright, high-contrast red.
    pub fn accent(&self, s: &str) -> String {
        self.wrap("1;38;5;203", s)
    }
    /// Box borders and input rules.
    pub fn line(&self, s: &str) -> String {
        self.wrap("38;5;242", s)
    }
    /// The user's own message: light text on a grey band.
    pub fn user_block(&self, s: &str) -> String {
        self.wrap("38;5;255;48;5;238", s)
    }
    pub fn dim(&self, s: &str) -> String {
        self.wrap("2", s)
    }
    pub fn red(&self, s: &str) -> String {
        self.wrap("38;5;203", s)
    }
    /// The word still being generated: a soft red tint.
    pub fn tint(&self, s: &str) -> String {
        self.wrap("38;5;210", s)
    }
}

/// Terminal columns taken by `c`: 2 for Hangul, CJK, fullwidth forms and most emoji,
/// 0 for control and combining characters, 1 otherwise.
pub fn char_width(c: char) -> usize {
    let u = c as u32;
    if c.is_control() || (0x0300..=0x036F).contains(&u) || (0xFE00..=0xFE0F).contains(&u) || u == 0x200B {
        return 0;
    }
    let wide = matches!(u,
        0x1100..=0x115F | 0x2E80..=0x303E | 0x3041..=0x33FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF
        | 0xA000..=0xA4CF | 0xAC00..=0xD7A3 | 0xF900..=0xFAFF | 0xFE30..=0xFE4F | 0xFF00..=0xFF60
        | 0xFFE0..=0xFFE6 | 0x1F300..=0x1F64F | 0x1F900..=0x1F9FF | 0x20000..=0x3FFFD);
    if wide {
        2
    } else {
        1
    }
}

pub fn str_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// Rows a typed line occupies after `prompt_w` columns of prompt, at `width` columns per row.
pub fn input_rows(prompt_w: usize, text: &str, width: usize) -> usize {
    let w = width.max(1);
    (prompt_w + str_width(text)).div_ceil(w).max(1)
}

/// Streaming word wrap: text arrives in pieces, lines break between words at `width`
/// columns, and continuation lines start with `indent` spaces. Words longer than a line are
/// broken hard. Spaces at the start of a line are dropped.
pub struct Wrap {
    width: usize,
    indent: usize,
    col: usize,
    word: String,
    word_w: usize,
    /// A space seen after the last word, printed only if the next word stays on this line.
    space: bool,
}

impl Wrap {
    /// `col` is the column the cursor is at when the first piece is printed.
    pub fn new(width: usize, indent: usize, col: usize) -> Wrap {
        Wrap { width: width.max(indent + 1), indent, col, word: String::new(), word_w: 0, space: false }
    }

    /// Text to print for `s`. Part of the last word may be held back until `finish`.
    pub fn push(&mut self, s: &str) -> String {
        let mut out = String::new();
        for c in s.chars() {
            match c {
                '\n' => {
                    self.flush_word(&mut out);
                    self.newline(&mut out);
                }
                ' ' | '\t' => {
                    self.flush_word(&mut out);
                    self.space = self.col > self.indent;
                }
                _ => {
                    self.word.push(c);
                    self.word_w += char_width(c);
                }
            }
        }
        out
    }

    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        self.flush_word(&mut out);
        out
    }

    /// The word held back so far, for a live preview: (columns of leading space, word,
    /// word width). None when there is no partial word or it would not fit on this line
    /// (it will move to the next line once complete).
    pub fn preview(&self) -> Option<(usize, &str, usize)> {
        if self.word.is_empty() {
            return None;
        }
        let gap = usize::from(self.space);
        (self.col + gap + self.word_w <= self.width).then_some((gap, self.word.as_str(), self.word_w))
    }

    fn newline(&mut self, out: &mut String) {
        out.push('\n');
        out.push_str(&" ".repeat(self.indent));
        self.col = self.indent;
        self.space = false;
    }

    fn flush_word(&mut self, out: &mut String) {
        if self.word.is_empty() {
            return;
        }
        let gap = usize::from(self.space);
        if self.col + gap + self.word_w > self.width && self.col > self.indent {
            self.newline(out);
        } else if self.space {
            out.push(' ');
            self.col += 1;
            self.space = false;
        }
        for c in std::mem::take(&mut self.word).chars() {
            let w = char_width(c);
            if self.col + w > self.width {
                self.newline(out);
            }
            out.push(c);
            self.col += w;
        }
        self.word_w = 0;
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
    /// Reopen the model picker.
    Models,
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
        "/models" => Command::Models,
        "/model" if rest.is_empty() => Command::Invalid("usage: /model <path to model.bin>".into()),
        "/model" => Command::Model(rest.to_string()),
        _ => Command::Unknown(cmd.to_string()),
    }
}

/// `⎿  N tok · X tok/s · ctx used/ctx`
pub fn fmt_footer(n: usize, secs: f64, used: usize, ctx: usize) -> String {
    let rate = if secs > 0.0 { n as f64 / secs } else { 0.0 };
    format!("⎿  {n} tok · {rate:.1} tok/s · ctx {used}/{ctx}")
}

/// Welcome box with the model, its size and the backend.
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
    let title = "✻ Welcome to rllm chat!";
    // Box = "│ " + text + pad + "│": keep it inside the window.
    let max_text = line_width(style).saturating_sub(4).max(24);
    let model_line = {
        // city name + version, looked up among the runs next to this one
        let head = match run_title(model) {
            Some((c, v)) => format!("  model: {c} {v} · "),
            None => "  model: ".to_string(),
        };
        let head = head.as_str();
        let path = model.display().to_string();
        let room = max_text.saturating_sub(str_width(head));
        if str_width(&path) <= room {
            format!("{head}{path}")
        } else {
            // keep the end of the path (run name and file), drop the front
            let mut tail: Vec<char> = Vec::new();
            let mut w = 1; // the ellipsis
            for c in path.chars().rev() {
                if w + char_width(c) > room {
                    break;
                }
                w += char_width(c);
                tail.push(c);
            }
            format!("{head}…{}", tail.into_iter().rev().collect::<String>())
        }
    };
    let lines = [
        String::new(),
        "  /help for commands · Ctrl+C stops a reply".to_string(),
        String::new(),
        model_line,
        format!(
            "  {params} params · d={} layers={} ctx={} vocab={} bitnet={}",
            cfg.d, cfg.layers, cfg.ctx, cfg.vocab, cfg.bitnet
        ),
        format!("  backend: {} · threads={threads}", format!("{backend:?}").to_lowercase()),
    ];
    let inner = lines.iter().map(|l| str_width(l)).chain([str_width(title)]).max().unwrap_or(0) + 2;
    let row = |text: String, w: usize| {
        let pad = " ".repeat(inner.saturating_sub(w + 1));
        format!("{} {text}{pad}{}\n", style.line("│"), style.line("│"))
    };
    let mut out = style.line(&format!("╭{}╮", "─".repeat(inner)));
    out.push('\n');
    out.push_str(&row(style.accent(title), str_width(title)));
    for l in lines {
        let w = str_width(&l);
        out.push_str(&row(style.dim(&l), w));
    }
    out.push_str(&style.line(&format!("╰{}╯", "─".repeat(inner))));
    out.push('\n');
    out
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
/models        back to the model list to switch models
/model PATH    load another model.bin (its folder's tokenizer if it has one)
/quit, /exit   leave (also Ctrl+C at the prompt, or Ctrl+Z then Enter)
Ctrl+C while the assistant is replying stops that reply.";

/// Size tiers: the city says how much a model was trained, the way a model family's tier
/// names say how big its models are. (most training seconds, city); the last has no limit.
pub const TIERS: &[(f64, &str)] = &[
    (360.0, "Bruges"),    // up to ~5 minutes
    (1500.0, "Kyoto"),    // up to ~20 minutes
    (5400.0, "Seoul"),    // up to ~1 hour
    (21600.0, "New York"), // up to a night (6 h)
    (f64::INFINITY, "Atlantis"),
];

/// Tier city for a run trained for `wall_s` seconds (unknown counts as the smallest).
pub fn tier_for(wall_s: Option<f64>) -> &'static str {
    let w = wall_s.unwrap_or(0.0);
    TIERS.iter().find(|(max, _)| w <= *max).map_or("Atlantis", |t| t.1)
}

/// City name and version of the run that holds `model`, looked up among its sibling runs.
pub fn run_title(model: &Path) -> Option<(String, String)> {
    let dir = model.parent()?.parent()?;
    list_runs(dir).into_iter().find(|r| r.model == model).map(|r| (r.codename, r.version))
}

/// What `assign_versions` needs to know about one run.
#[derive(Clone, Copy, Debug)]
pub struct VersionInput {
    /// Runs with the same key share a tokenizer (one data generation).
    pub family: u64,
    pub age_s: u64,
}

/// `generation.release` per run. Generation numbers the families (same tokenizer) in order
/// of their oldest run, from 1; release numbers a family's runs by age, oldest `.0`. Both
/// depend only on when runs were made, so a name never changes once given.
pub fn assign_versions(runs: &[VersionInput]) -> Vec<String> {
    let mut families: Vec<(u64, u64)> = Vec::new(); // (family, oldest age)
    for r in runs {
        match families.iter_mut().find(|f| f.0 == r.family) {
            Some(f) => f.1 = f.1.max(r.age_s),
            None => families.push((r.family, r.age_s)),
        }
    }
    families.sort_by(|a, b| b.1.cmp(&a.1)); // oldest first
    let mut out = vec![String::new(); runs.len()];
    for (gi, (fam, _)) in families.iter().enumerate() {
        let mut members: Vec<usize> = (0..runs.len()).filter(|&i| runs[i].family == *fam).collect();
        members.sort_by(|&a, &b| runs[b].age_s.cmp(&runs[a].age_s)); // oldest first
        for (release, i) in members.into_iter().enumerate() {
            out[i] = format!("{}.{release}", gi + 1);
        }
    }
    out
}

/// Mean of the recorded validation losses (None if there are none).
fn mean_loss(r: &RunInfo) -> Option<f64> {
    let v: Vec<f64> = [r.val_pre, r.val_sft].into_iter().flatten().collect();
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

/// Marks, per generation, the run with the lowest mean validation loss (weak runs never).
pub fn mark_best(runs: &mut [RunInfo]) {
    let gens: Vec<String> = runs.iter().map(|r| r.version.split('.').next().unwrap_or("").to_string()).collect();
    for g in gens.iter().collect::<std::collections::HashSet<_>>() {
        let best = (0..runs.len())
            .filter(|&i| &gens[i] == g && !is_weak(&runs[i]))
            .min_by(|&a, &b| mean_loss(&runs[a]).unwrap_or(f64::INFINITY).total_cmp(&mean_loss(&runs[b]).unwrap_or(f64::INFINITY)));
        if let Some(i) = best {
            runs[i].best = true;
        }
    }
}

/// FNV-1a of a file's bytes (0 if unreadable).
fn file_key(path: &Path) -> u64 {
    std::fs::read(path)
        .map(|b| b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3)))
        .unwrap_or(0)
}

/// One trained run under `runs/`, for the startup picker.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunInfo {
    /// Tier city shown to the user (see `TIERS`).
    pub codename: String,
    /// `generation.release`, e.g. "2.4" (see `assign_versions`).
    pub version: String,
    pub name: String,
    pub model: PathBuf,
    /// Seconds since `model.bin` was written.
    pub age_s: u64,
    pub val_pre: Option<f64>,
    pub val_sft: Option<f64>,
    /// Training time from `results.csv`.
    pub wall_s: Option<f64>,
    /// Lowest mean validation loss in its generation.
    pub best: bool,
}

/// Runs with a `model.bin`, newest first, with validation losses and training time from
/// `results.csv` (the last row for a name wins), each with its tier city, version and
/// best-of-generation mark.
pub fn list_runs(dir: &Path) -> Vec<RunInfo> {
    let mut stats = std::collections::HashMap::new();
    if let Ok(csv) = std::fs::read_to_string(dir.join("results.csv")) {
        let mut lines = csv.lines();
        let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
        let col = |n: &str| header.iter().position(|h| *h == n);
        if let Some(ni) = col("name") {
            let (pi, si, wi) = (col("val_loss_pretrain"), col("val_loss_sft"), col("wall_s"));
            for l in lines {
                let f: Vec<&str> = l.split(',').collect();
                let num = |i: Option<usize>| {
                    i.and_then(|i| f.get(i)).and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite())
                };
                if let Some(n) = f.get(ni) {
                    stats.insert(n.to_string(), (num(pi), num(si), num(wi)));
                }
            }
        }
    }
    let now = std::time::SystemTime::now();
    let mut runs: Vec<(std::time::SystemTime, RunInfo)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| {
            let e = e.ok()?;
            let model = e.path().join("model.bin");
            let t = std::fs::metadata(&model).ok()?.modified().ok()?;
            let name = e.file_name().to_string_lossy().into_owned();
            let (val_pre, val_sft, wall_s) = stats.get(&name).copied().unwrap_or((None, None, None));
            let age_s = now.duration_since(t).map(|d| d.as_secs()).unwrap_or(0);
            let codename = tier_for(wall_s).to_string();
            Some((t, RunInfo { codename, name, model, age_s, val_pre, val_sft, wall_s, ..Default::default() }))
        })
        .collect();
    runs.sort_by(|a, b| b.0.cmp(&a.0));
    let mut runs: Vec<RunInfo> = runs.into_iter().map(|(_, r)| r).collect();
    let inputs: Vec<VersionInput> = runs
        .iter()
        .map(|r| VersionInput { family: file_key(&r.model.with_file_name("tokenizer.bin")), age_s: r.age_s })
        .collect();
    for (r, v) in runs.iter_mut().zip(assign_versions(&inputs)) {
        r.version = v;
    }
    mark_best(&mut runs);
    runs
}

/// `45s`, `12m`, `3h`, `2d`
pub fn fmt_age(s: u64) -> String {
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m", s / 60),
        3600..=86_399 => format!("{}h", s / 3600),
        _ => format!("{}d", s / 86_400),
    }
}

const PICK_TITLE: &str = "✻ Choose a model";
const ARROW_HINT: &str = "  ↑/↓ move · Enter start · 1-9 jump · q quit";

/// Runs whose mean validation loss is at least this are "weak" (they only print noise) and
/// hidden from the picker unless asked for.
pub const WEAK_LOSS: f64 = 6.0;

/// Mean validation loss of at least `WEAK_LOSS`, or no losses recorded at all.
pub fn is_weak(r: &RunInfo) -> bool {
    let v: Vec<f64> = [r.val_pre, r.val_sft].into_iter().flatten().collect();
    v.is_empty() || v.iter().sum::<f64>() / v.len() as f64 >= WEAK_LOSS
}

/// Indices of the runs to list: all of them, or only the non-weak ones (all, if every run
/// is weak).
pub fn visible_runs(runs: &[RunInfo], show_all: bool) -> Vec<usize> {
    let v: Vec<usize> = (0..runs.len()).filter(|&i| show_all || !is_weak(&runs[i])).collect();
    if v.is_empty() {
        (0..runs.len()).collect()
    } else {
        v
    }
}

fn subset(runs: &[RunInfo], idx: &[usize]) -> Vec<RunInfo> {
    idx.iter().map(|&i| runs[i].clone()).collect()
}

/// One line per run, the selected one marked with a red `❯` and its name in bold.
pub fn fmt_run_rows(style: &Style, runs: &[RunInfo], sel: usize) -> String {
    let title = |r: &RunInfo| {
        let star = if r.best { " ★" } else { "" };
        if r.version.is_empty() {
            format!("{}{star}", r.codename)
        } else {
            format!("{} {}{star}", r.codename, r.version)
        }
    };
    let wcode = runs.iter().map(|r| str_width(&title(r))).max().unwrap_or(0);
    let wname = runs.iter().map(|r| str_width(&r.name)).max().unwrap_or(0);
    let loss = |v: Option<f64>| v.map_or("  -  ".to_string(), |v| format!("{v:.2}"));
    let mut out = String::new();
    for (i, r) in runs.iter().enumerate() {
        let marker = if i == sel { style.accent("❯") } else { " ".to_string() };
        let t = title(r);
        let pad_c = " ".repeat(wcode - str_width(&t));
        let pad_n = " ".repeat(wname - str_width(&r.name));
        let info = style.dim(&format!(
            "{}{pad_n}  {:>4} ago · val pretrain {} · sft {}",
            r.name,
            fmt_age(r.age_s),
            loss(r.val_pre),
            loss(r.val_sft)
        ));
        let code = if i == sel { style.bold(&t) } else { t.clone() };
        out.push_str(&format!(" {marker} {:>2}. {code}{pad_c}  {info}\n", i + 1));
    }
    out
}

/// Numbered list of runs for the typed-number picker; the first (newest) is the default.
pub fn fmt_run_menu(style: &Style, runs: &[RunInfo]) -> String {
    let mut out = format!("{}\n\n", style.accent(PICK_TITLE));
    out.push_str(&fmt_run_rows(style, runs, 0));
    out.push_str(&format!("\n{}\n", style.dim("  Enter = 1 (newest) · number to pick · q to quit")));
    out
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Key {
    Up,
    Down,
    Home,
    End,
    Enter,
    Digit(u8),
    /// Show or hide weak models (`h`).
    Toggle,
    Quit,
    Other,
}

/// Keys in bytes read from a VT-mode console: `ESC [ A`/`B` (also `ESC O A`, and with
/// modifiers like `ESC [ 1 ; 5 A`), Home/End, Enter, digits 1-9, `j`/`k`, and quit on
/// `q`, a lone Esc or Ctrl+C (byte 3).
pub fn parse_keys(b: &[u8]) -> Vec<Key> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == 0x1b && matches!(b.get(i + 1), Some(b'[') | Some(b'O')) {
            let mut j = i + 2;
            while j < b.len() && !(0x40..=0x7e).contains(&b[j]) {
                j += 1;
            }
            out.push(match b.get(j) {
                Some(b'A') => Key::Up,
                Some(b'B') => Key::Down,
                Some(b'H') => Key::Home,
                Some(b'F') => Key::End,
                _ => Key::Other,
            });
            i = j + 1;
            continue;
        }
        out.push(match b[i] {
            0x1b | 3 | b'q' | b'Q' => Key::Quit,
            b'\r' | b'\n' => Key::Enter,
            b'k' => Key::Up,
            b'j' => Key::Down,
            b'h' | b'H' => Key::Toggle,
            c @ b'1'..=b'9' => Key::Digit(c - b'0'),
            _ => Key::Other,
        });
        i += 1;
    }
    out
}

/// Selection state of the arrow-key picker.
pub struct Picker {
    pub sel: usize,
    n: usize,
}

impl Picker {
    pub fn new(n: usize) -> Picker {
        Picker { sel: 0, n: n.max(1) }
    }

    /// Some(choice) when the picker is done: Some(Some(i)) picked, Some(None) quit.
    /// Up/Down wrap around; a digit jumps to that row (Enter confirms).
    pub fn apply(&mut self, k: Key) -> Option<Option<usize>> {
        match k {
            Key::Up => self.sel = (self.sel + self.n - 1) % self.n,
            Key::Down => self.sel = (self.sel + 1) % self.n,
            Key::Home => self.sel = 0,
            Key::End => self.sel = self.n - 1,
            Key::Digit(d) if (d as usize) <= self.n => self.sel = d as usize - 1,
            Key::Enter => return Some(Some(self.sel)),
            Key::Quit => return Some(None),
            _ => {}
        }
        None
    }
}

/// Arrow-key picker: draws the list, redraws it in place as the selection moves, and
/// returns the chosen index into `runs`. Weak models are hidden until `h` is pressed.
/// Ok(None) on quit or end of input. `input` must deliver keys one at a time (console in
/// raw mode).
pub fn pick_run_keys(
    runs: &[RunInfo],
    input: &mut dyn std::io::Read,
    out: &mut dyn Write,
    style: &Style,
) -> std::io::Result<Option<usize>> {
    let mut show_all = false;
    let mut vis = visible_runs(runs, show_all);
    let weak = runs.len() - visible_runs(runs, false).len();
    let mut p = Picker::new(vis.len());
    let draw = |out: &mut dyn Write, vis: &[usize], sel: usize, show_all: bool| -> std::io::Result<()> {
        let extra = match (weak, show_all) {
            (0, _) => String::new(),
            (_, true) => " · h hide weak".to_string(),
            (n, false) => format!(" · h show {n} weak"),
        };
        let hint = style.dim(&format!("{ARROW_HINT}{extra}"));
        write!(out, "{}\n{hint}\n", fmt_run_rows(style, &subset(runs, vis), sel))
    };
    write!(out, "{}\n\n\x1b[?25l", style.accent(PICK_TITLE))?; // hide the cursor
    draw(out, &vis, p.sel, show_all)?;
    let mut buf = [0u8; 64];
    let result = loop {
        out.flush()?;
        let n = input.read(&mut buf)?;
        if n == 0 {
            break None;
        }
        let (before, rows_before) = (p.sel, vis.len());
        let mut changed = false;
        let mut done = None;
        for k in parse_keys(&buf[..n]) {
            if k == Key::Toggle && weak > 0 {
                show_all = !show_all;
                vis = visible_runs(runs, show_all);
                p = Picker::new(vis.len());
                changed = true;
                continue;
            }
            if let Some(r) = p.apply(k) {
                done = Some(r.map(|i| vis[i]));
                break;
            }
        }
        if changed || p.sel != before {
            // back to the first row (rows + blank + hint), clear, redraw
            write!(out, "\x1b[{}A\r\x1b[J", rows_before + 2)?;
            draw(out, &vis, p.sel, show_all)?;
        }
        if let Some(r) = done {
            break r;
        }
    };
    write!(out, "\x1b[?25h\n")?; // show the cursor again
    out.flush()?;
    Ok(result)
}

/// Startup picker on the real console: arrow keys when the console allows key-at-a-time
/// input and colour is on, otherwise typing a number.
pub fn pick_run_console(runs: &[RunInfo], style: &Style) -> std::io::Result<Option<usize>> {
    let (stdin, stdout) = (std::io::stdin(), std::io::stdout());
    if style.color {
        if let Some(_raw) = sys::RawInput::enable() {
            return pick_run_keys(runs, &mut stdin.lock(), &mut stdout.lock(), style);
        }
    }
    pick_run(runs, &mut stdin.lock(), &mut stdout.lock(), style)
}

/// Reads the picker answer: empty = first run, `1..=n` = that run, `q` = quit (None).
pub fn parse_choice(line: &str, n: usize) -> Result<Option<usize>, String> {
    let t = line.trim();
    if t.is_empty() {
        return Ok(Some(0));
    }
    if t.eq_ignore_ascii_case("q") || t == "/quit" {
        return Ok(None);
    }
    match t.parse::<usize>() {
        Ok(k) if (1..=n).contains(&k) => Ok(Some(k - 1)),
        _ => Err(format!("pick a number from 1 to {n}, Enter for 1, or q")),
    }
}

/// Shows the picker on `out` and reads answers from `input` until one is valid; weak
/// models are left out. Returns an index into `runs`; Ok(None) when the user quits or
/// input ends.
pub fn pick_run(
    runs: &[RunInfo],
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    style: &Style,
) -> std::io::Result<Option<usize>> {
    let vis = visible_runs(runs, false);
    write!(out, "{}", fmt_run_menu(style, &subset(runs, &vis)))?;
    if vis.len() < runs.len() {
        writeln!(out, "{}", style.dim(&format!("  {} weak models hidden (use --model to pick one)", runs.len() - vis.len())))?;
    }
    let mut line = String::new();
    loop {
        write!(out, "{} ", style.accent(">"))?;
        out.flush()?;
        line.clear();
        if input.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        match parse_choice(&line, vis.len()) {
            Ok(c) => {
                writeln!(out)?;
                return Ok(c.map(|i| vis[i]));
            }
            Err(e) => writeln!(out, "{}", note(style, &e, true))?,
        }
    }
}

/// A system note under the last block: `  ⎿  text`, continuation lines aligned.
fn note(style: &Style, text: &str, error: bool) -> String {
    let body = text.replace('\n', "\n     ");
    let s = format!("  ⎿  {body}");
    if error {
        style.red(&s)
    } else {
        style.dim(&s)
    }
}

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

    /// Switch to another model, with the tokenizer saved next to it when there is one
    /// (otherwise the current tokenizer); on any error the current model is untouched.
    pub fn swap_model(&mut self, path: &Path) -> Result<(), String> {
        let (cfg, params) = load_model(path)?;
        let tok = path.parent().map(|d| d.join("tokenizer.bin")).filter(|p| p.is_file());
        let new_bpe = match &tok {
            Some(t) => Some(Bpe::load(t).map_err(|e| format!("{}: {e}", t.display()))?),
            None => None,
        };
        let vocab = new_bpe.as_ref().unwrap_or(&self.bpe).vocab.len();
        if cfg.vocab != vocab {
            return Err(format!("model vocab {} does not match the tokenizer ({vocab})", cfg.vocab));
        }
        if let (Some(b), Some(t)) = (new_bpe, tok) {
            self.bpe = b;
            self.args.tokenizer = t;
        }
        self.cache = KvCache::new(&cfg);
        self.cfg = cfg;
        self.params = params;
        self.args.model = path.to_path_buf();
        self.history.clear();
        Ok(())
    }

    /// One user turn: stream the reply to `out` as a wrapped `● ` block, then a footer.
    /// A set `cancel` flag stops the reply (the partial text is kept in history and marked
    /// interrupted); the flag is consumed.
    pub fn turn(
        &mut self,
        out: &mut dyn Write,
        style: &Style,
        text: &str,
        cancel: &AtomicBool,
    ) -> std::io::Result<()> {
        let user = self.bpe.encode(text.as_bytes());
        let prompt = build_prompt(&self.history, &user, &self.bpe, self.cfg.ctx, 64);
        write!(out, "{} ", style.accent("●"))?;
        out.flush()?;
        let mut wrap = Wrap::new(line_width(style), 2, 2);
        let (params, cfg, be) = (&self.params, &self.cfg, self.args.backend);
        let cache = &mut self.cache;
        let mut stepper = |tok: u16, pos: usize| step(params, cfg, be, cache, tok, pos);
        let (reply, secs) = {
            let mut emit = |s: &str| {
                if style.color {
                    // erase the last preview, print what is settled, then show the word still
                    // being generated in red and step back over it (it is redrawn next time)
                    out.write_all(b"\x1b[K")?;
                    out.write_all(wrap.push(s).as_bytes())?;
                    if let Some((gap, word, w)) = wrap.preview() {
                        write!(out, "{}{}\x1b[{}D", " ".repeat(gap), style.tint(word), gap + w)?;
                    }
                } else {
                    out.write_all(wrap.push(s).as_bytes())?;
                }
                out.flush()
            };
            generate_cancellable(
                &mut stepper,
                cfg.ctx,
                &self.args,
                &self.bpe,
                &prompt,
                &mut self.rng,
                cancel,
                &mut emit,
            )?
        };
        if style.color {
            out.write_all(b"\x1b[K")?; // drop the red preview; the last word prints normally
        }
        out.write_all(wrap.finish().as_bytes())?;
        let stopped = cancel.swap(false, Relaxed);
        writeln!(out)?;
        if stopped {
            writeln!(out, "{}", note(style, "Interrupted", true))?;
        }
        let used = prompt.len() + reply.len();
        writeln!(out, "  {}\n", style.dim(&fmt_footer(reply.len(), secs, used, self.cfg.ctx)))?;
        self.history.push((user, reply));
        Ok(())
    }
}

/// Draws the input box (rule, prompt row, rule, hint line) and leaves the cursor after the
/// red `> `. Without colour, prints only `> `.
fn draw_input_box(out: &mut dyn Write, style: &Style, chat: &Chat) -> std::io::Result<()> {
    if !style.color {
        return write!(out, "> ");
    }
    let w = line_width(style);
    let rule = style.line(&"─".repeat(w));
    let hint = style.dim(&format!(
        "  /help for commands · Ctrl+C stops a reply · temp {:.2} · top_p {:.2}",
        chat.args.temp, chat.args.top_p
    ));
    // rule / empty prompt row / rule / hint, then back up to the prompt row.
    write!(out, "{rule}\n\n{rule}\n{hint}\x1b[2A\r{} ", style.accent(">"))
}

/// After Enter: erase the input box and print the message as a grey block in its place.
fn replace_input_box(out: &mut dyn Write, style: &Style, text: &str) -> std::io::Result<()> {
    if !style.color {
        return Ok(());
    }
    let w = line_width(style);
    // The cursor is on the row after the typed text; go back to the top rule and clear down.
    let up = input_rows(2, text, w + 1) + 1;
    write!(out, "\x1b[{up}A\r\x1b[J")?;
    let body = format!(" > {text}");
    let pad = w.saturating_sub(str_width(&body));
    let pad = if str_width(&body) <= w { " ".repeat(pad) } else { String::new() };
    write!(out, "{}\n\n", style.user_block(&format!("{body}{pad}")))
}

/// Clears the input box when leaving at the prompt (Ctrl+C / EOF / `/quit`).
fn clear_input_box(out: &mut dyn Write, style: &Style, rows_up: usize) -> std::io::Result<()> {
    if style.color {
        write!(out, "\x1b[{rows_up}A\r\x1b[J")?;
    }
    Ok(())
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
        draw_input_box(out, &style, &chat).map_err(io)?;
        out.flush().map_err(io)?;
        buf.clear();
        let n = match input.read_until(b'\n', &mut buf) {
            Ok(n) => n,
            Err(_) if cancel.load(Relaxed) => 0,
            Err(e) => return Err(e.to_string()),
        };
        if n == 0 || cancel.swap(false, Relaxed) {
            // EOF / Ctrl+C: no newline was echoed, the cursor is still on the prompt row.
            clear_input_box(out, &style, 1).map_err(io)?;
            if !style.color {
                writeln!(out).map_err(io)?;
            }
            break;
        }
        let line = String::from_utf8_lossy(&buf).trim_end_matches(['\r', '\n']).to_string();
        if line.trim().is_empty() {
            clear_input_box(out, &style, 2).map_err(io)?;
            continue;
        }
        let cmd = parse_command(&line);
        if cmd == Command::Quit {
            clear_input_box(out, &style, input_rows(2, &line, line_width(&style) + 1) + 1).map_err(io)?;
            break;
        }
        replace_input_box(out, &style, line.trim()).map_err(io)?;
        let say = |out: &mut dyn Write, text: &str, error: bool| writeln!(out, "{}\n", note(&style, text, error));
        match cmd {
            Command::Quit => unreachable!(),
            Command::Help => say(out, HELP, false).map_err(io)?,
            Command::Reset => {
                chat.history.clear();
                say(out, "new conversation", false).map_err(io)?;
            }
            Command::Temp(v) => {
                chat.args.temp = v;
                say(out, &format!("temperature set to {v}"), false).map_err(io)?;
            }
            Command::TopP(v) => {
                chat.args.top_p = v;
                say(out, &format!("top_p set to {v}"), false).map_err(io)?;
            }
            Command::Model(path) => match chat.swap_model(Path::new(&path)) {
                Ok(()) => say(
                    out,
                    &format!("loaded {path} ({} params); new conversation", chat.params.num_params()),
                    false,
                )
                .map_err(io)?,
                Err(e) => say(out, &format!("could not load {path}: {e} (keeping the current model)"), true)
                    .map_err(io)?,
            },
            Command::Models => {
                let runs = list_runs(Path::new("runs"));
                if runs.is_empty() {
                    say(out, "no trained models under runs/", true).map_err(io)?;
                    continue;
                }
                // Arrow keys on a colour console, typed number otherwise (both read `input`).
                let raw = if style.color { sys::RawInput::enable() } else { None };
                let choice = if raw.is_some() {
                    pick_run_keys(&runs, input, out, &style)
                } else {
                    pick_run(&runs, input, out, &style)
                }
                .map_err(io)?;
                drop(raw);
                match choice {
                    Some(i) => match chat.swap_model(&runs[i].model) {
                        Ok(()) => {
                            let header = fmt_header(
                                &style,
                                &chat.args.model,
                                chat.params.num_params(),
                                &chat.cfg,
                                chat.args.backend,
                                chat.args.threads,
                            );
                            write!(out, "{header}\n").map_err(io)?;
                        }
                        Err(e) => say(out, &format!("could not load {}: {e}", runs[i].codename), true).map_err(io)?,
                    },
                    None => say(out, "kept the current model", false).map_err(io)?,
                }
            }
            Command::Invalid(msg) => say(out, &msg, true).map_err(io)?,
            Command::Unknown(c) => {
                say(out, &format!("Unknown command {c}. Type /help for commands."), true).map_err(io)?
            }
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
        assert_eq!(parse_command("/models"), Command::Models);
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
    fn list_runs_newest_first_with_losses() {
        let dir = std::env::temp_dir().join(format!("rllm_tui_runs_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let now = SystemTime::now();
        for (name, age) in [("old", 7200), ("new", 30), ("mid", 600)] {
            let d = dir.join(name);
            std::fs::create_dir_all(&d).unwrap();
            let f = std::fs::File::create(d.join("model.bin")).unwrap();
            f.set_modified(now - Duration::from_secs(age)).unwrap();
        }
        std::fs::create_dir_all(dir.join("no-model")).unwrap();
        std::fs::write(
            dir.join("results.csv"),
            "timestamp,name,val_loss_pretrain,val_loss_sft,wall_s\n1,new,9.0,9.0,1200\n2,new,3.13,3.78,1199.5\n3,old,NaN,4.5,3599\n",
        )
        .unwrap();
        let runs = list_runs(&dir);
        let names: Vec<&str> = runs.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["new", "mid", "old"]);
        assert_eq!((runs[0].val_pre, runs[0].val_sft), (Some(3.13), Some(3.78)), "last row wins");
        assert_eq!((runs[1].val_pre, runs[1].val_sft), (None, None));
        assert_eq!((runs[2].val_pre, runs[2].val_sft), (None, Some(4.5)), "NaN shown as missing");
        assert_eq!(runs[0].model, dir.join("new").join("model.bin"));
        assert!(list_runs(&dir.join("nope")).is_empty());
        // no tokenizer files: one generation; releases by age (old .0, mid .1, new .2);
        // tier by training time; "new" (mean 3.455) is the best non-weak run
        let titles: Vec<String> = runs
            .iter()
            .map(|r| format!("{} {}{}", r.codename, r.version, if r.best { " ★" } else { "" }))
            .collect();
        assert_eq!(titles, ["Kyoto 1.2 ★", "Bruges 1.1", "Seoul 1.0"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn versions_by_generation_and_release() {
        let v = |family, age_s| VersionInput { family, age_s };
        let runs = [v(7, 500), v(7, 400), v(9, 300), v(9, 200), v(9, 100), v(7, 50)];
        // generation by oldest run of each family, release by age within it
        assert_eq!(assign_versions(&runs), ["1.0", "1.1", "2.0", "2.1", "2.2", "1.2"]);
        // a newer run never changes existing versions
        let mut more = runs.to_vec();
        more.push(v(9, 10));
        assert_eq!(&assign_versions(&more)[..6], &assign_versions(&runs)[..]);
        assert_eq!(assign_versions(&more)[6], "2.3");
    }

    #[test]
    fn tiers_by_training_time() {
        assert_eq!(tier_for(Some(47.0)), "Bruges");
        assert_eq!(tier_for(Some(300.0)), "Bruges");
        assert_eq!(tier_for(Some(1199.8)), "Kyoto");
        assert_eq!(tier_for(Some(3600.0)), "Seoul");
        assert_eq!(tier_for(Some(8.0 * 3600.0)), "Atlantis");
        assert_eq!(tier_for(Some(5.0 * 3600.0)), "New York");
        assert_eq!(tier_for(None), "Bruges");
    }

    #[test]
    fn best_marked_per_generation() {
        let mk = |v: &str, pre: f64, sft: f64| RunInfo { version: v.into(), val_pre: Some(pre), val_sft: Some(sft), ..Default::default() };
        let mut runs = vec![mk("1.0", 7.5, 7.0), mk("1.1", 4.0, 3.7), mk("2.0", 2.5, 3.2), mk("2.1", 2.4, 3.1), mk("2.2", 6.3, 6.9)];
        mark_best(&mut runs);
        let best: Vec<bool> = runs.iter().map(|r| r.best).collect();
        assert_eq!(best, [false, true, false, true, false]);
    }

    #[test]
    fn swap_model_uses_sibling_tokenizer() {
        let (dir, model, tok) = tiny_files("swap_a", 9);
        let mut chat = Chat::load(args(model, tok.clone())).unwrap();
        // second run: its own folder holds model.bin + tokenizer.bin
        let dir2 = std::env::temp_dir().join(format!("rllm_tui_swap_b_{}", std::process::id()));
        std::fs::create_dir_all(&dir2).unwrap();
        let cfg = tiny_cfg();
        Params::init(&cfg, &mut Rng::new(3)).save(&cfg, &dir2.join("model.bin")).unwrap();
        train_bpe(b"other words entirely, other words", 300).save(&dir2.join("tokenizer.bin")).unwrap();
        chat.history.push((vec![1], vec![2]));
        chat.swap_model(&dir2.join("model.bin")).unwrap();
        assert_eq!(chat.args.tokenizer, dir2.join("tokenizer.bin"));
        assert!(chat.history.is_empty());
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(dir2);
    }

    #[test]
    fn keys_parsed() {
        use Key::*;
        assert_eq!(parse_keys(b"\x1b[A\x1b[B"), vec![Up, Down]);
        assert_eq!(parse_keys(b"\x1bOA\x1b[1;5B\x1b[H\x1b[F"), vec![Up, Down, Home, End]);
        assert_eq!(parse_keys(b"\r"), vec![Enter]);
        assert_eq!(parse_keys(b"kj3"), vec![Up, Down, Digit(3)]);
        assert_eq!(parse_keys(b"\x1b"), vec![Quit], "lone Esc");
        assert_eq!(parse_keys(b"\x03"), vec![Quit], "Ctrl+C");
        assert_eq!(parse_keys(b"q"), vec![Quit]);
        assert_eq!(parse_keys(b"x0\x1b[C"), vec![Other, Other, Other]);
    }

    #[test]
    fn picker_moves_and_wraps() {
        let mut p = Picker::new(3);
        assert_eq!(p.apply(Key::Up), None);
        assert_eq!(p.sel, 2, "wraps to the last row");
        p.apply(Key::Down);
        assert_eq!(p.sel, 0);
        p.apply(Key::Digit(2));
        assert_eq!(p.sel, 1);
        p.apply(Key::Digit(9));
        assert_eq!(p.sel, 1, "out-of-range digit ignored");
        p.apply(Key::End);
        assert_eq!(p.sel, 2);
        p.apply(Key::Home);
        assert_eq!(p.apply(Key::Enter), Some(Some(0)));
        assert_eq!(Picker::new(3).apply(Key::Quit), Some(None));
    }

    fn two_runs() -> Vec<RunInfo> {
        vec![
            RunInfo { codename: "Cairo".into(), version: "1.2".into(), name: "a".into(), model: "runs/a/model.bin".into(), age_s: 1, val_pre: None, val_sft: None, wall_s: None, best: false },
            RunInfo { codename: "Busan".into(), version: "1.1".into(), name: "b".into(), model: "runs/b/model.bin".into(), age_s: 2, val_pre: None, val_sft: None, wall_s: None, best: false },
            RunInfo { codename: "Amsterdam".into(), version: "1.0".into(), name: "c".into(), model: "runs/c/model.bin".into(), age_s: 3, val_pre: None, val_sft: None, wall_s: None, best: false },
        ]
    }

    fn run(code: &str, pre: Option<f64>, sft: Option<f64>) -> RunInfo {
        RunInfo {
            codename: code.into(),
            version: "1.0".into(),
            name: code.to_lowercase(),
            model: format!("runs/{code}/model.bin").into(),
            age_s: 1,
            val_pre: pre,
            val_sft: sft,
            ..Default::default()
        }
    }

    #[test]
    fn weak_models_hidden_until_toggled() {
        let runs = vec![
            run("Good", Some(2.5), Some(3.2)),
            run("Noise", Some(6.3), Some(6.9)),
            run("Okay", Some(4.0), Some(3.7)),
            run("Empty", None, None),
        ];
        assert!(!is_weak(&runs[0]) && is_weak(&runs[1]) && !is_weak(&runs[2]) && is_weak(&runs[3]));
        assert_eq!(visible_runs(&runs, false), [0, 2]);
        assert_eq!(visible_runs(&runs, true), [0, 1, 2, 3]);
        let s = Style { color: false };
        // down, Enter: second visible row = "Okay" (index 2 in the full list)
        let mut out = Vec::new();
        let mut input = std::io::Cursor::new(b"\x1b[B\r".to_vec());
        assert_eq!(pick_run_keys(&runs, &mut input, &mut out, &s).unwrap(), Some(2));
        let shown = String::from_utf8(out).unwrap();
        assert!(!shown.contains("Noise") && shown.contains("h show 2 weak"), "{shown}");
        // h shows all four, then down, Enter picks "Noise" (index 1)
        let mut out = Vec::new();
        let mut input = std::io::Cursor::new(b"h\x1b[B\r".to_vec());
        assert_eq!(pick_run_keys(&runs, &mut input, &mut out, &s).unwrap(), Some(1));
        let shown = String::from_utf8(out).unwrap();
        assert!(shown.contains("Noise") && shown.contains("h hide weak"), "{shown}");
        assert!(shown.contains("\x1b[4A"), "toggle redraw moves up the old 2 rows + 2: {shown:?}");
        // number picker: only visible runs, numbers map back to the full list
        let mut out = Vec::new();
        let mut input = std::io::Cursor::new(b"2\n".to_vec());
        assert_eq!(pick_run(&runs, &mut input, &mut out, &s).unwrap(), Some(2));
        assert!(String::from_utf8(out).unwrap().contains("2 weak models hidden"));
    }

    #[test]
    fn arrow_picker_scripted() {
        let runs = two_runs();
        let s = Style { color: false };
        let rows = fmt_run_rows(&s, &runs, 1);
        assert!(rows.lines().nth(1).unwrap().contains("❯  2. Busan 1.1      b"), "{rows}");
        assert!(!rows.lines().next().unwrap().contains('❯'));
        // down, down, up, Enter -> row 2 (index 1); keys may arrive in one read
        let mut out = Vec::new();
        let mut input = std::io::Cursor::new(b"\x1b[B\x1b[B\x1b[A\r".to_vec());
        assert_eq!(pick_run_keys(&runs, &mut input, &mut out, &s).unwrap(), Some(1));
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("\x1b[5A"), "redraw moves up rows + 2: {out:?}");
        assert!(out.contains("\x1b[?25l") && out.ends_with("\x1b[?25h\n"), "cursor hidden then shown");
        // Esc quits, end of input quits
        let mut esc = std::io::Cursor::new(b"\x1b".to_vec());
        assert_eq!(pick_run_keys(&runs, &mut esc, &mut Vec::new(), &s).unwrap(), None);
        let mut eof = std::io::Cursor::new(Vec::new());
        assert_eq!(pick_run_keys(&runs, &mut eof, &mut Vec::new(), &s).unwrap(), None);
    }

    #[test]
    fn choice_parsing() {
        assert_eq!(parse_choice("\n", 3), Ok(Some(0)));
        assert_eq!(parse_choice(" 2 ", 3), Ok(Some(1)));
        assert_eq!(parse_choice("3", 3), Ok(Some(2)));
        assert_eq!(parse_choice("q", 3), Ok(None));
        assert_eq!(parse_choice("Q", 3), Ok(None));
        for bad in ["0", "4", "-1", "abc", "1.5"] {
            assert!(parse_choice(bad, 3).is_err(), "{bad}");
        }
    }

    #[test]
    fn picker_menu_and_retry() {
        let runs = vec![
            RunInfo { codename: "Istanbul".into(), version: "2.4".into(), name: "tiny-mem".into(), model: "runs/tiny-mem/model.bin".into(), age_s: 90, val_pre: Some(2.56), val_sft: Some(3.2), wall_s: None, best: false },
            RunInfo { codename: "Dublin".into(), version: "1.3".into(), name: "reddit".into(), model: "runs/reddit/model.bin".into(), age_s: 7200, val_pre: Some(4.03), val_sft: Some(3.75), wall_s: None, best: false },
        ];
        let s = Style { color: false };
        let menu = fmt_run_menu(&s, &runs);
        assert!(!menu.contains('\x1b'));
        for want in ["1. Istanbul 2.4", "2. Dublin 1.3", "tiny-mem", "reddit", "1m ago", "2h ago", "2.56", "3.20", "Enter = 1"] {
            assert!(menu.contains(want), "missing {want:?} in {menu}");
        }
        let mut out = Vec::new();
        let mut input = std::io::Cursor::new(b"7\nzz\n2\n".to_vec());
        assert_eq!(pick_run(&runs, &mut input, &mut out, &s).unwrap(), Some(1));
        let out = String::from_utf8(out).unwrap();
        assert_eq!(out.matches("pick a number").count(), 2, "{out}");
        let mut eof = std::io::Cursor::new(Vec::new());
        assert_eq!(pick_run(&runs, &mut eof, &mut Vec::new(), &s).unwrap(), None);
        assert_eq!(fmt_age(59), "59s");
        assert_eq!(fmt_age(86_400 * 3), "3d");
    }

    #[test]
    fn no_color_emits_no_escapes() {
        let s = Style { color: false };
        let cfg = tiny_cfg();
        let all = [
            s.accent("a"),
            s.line("b"),
            s.user_block("u"),
            s.dim("c"),
            s.red("d"),
            note(&s, "x\ny", true),
            fmt_header(&s, Path::new("runs/x/model.bin"), 1234, &cfg, Backend::Scalar, 4),
        ];
        for t in &all {
            assert!(!t.contains('\x1b'), "{t:?}");
        }
        assert_eq!(all[0], "a");
        let c = Style { color: true };
        assert!(c.accent("a").contains("\x1b[") && c.accent("a").contains('a'));
        assert!(c.dim("x").ends_with("\x1b[0m"));
    }

    #[test]
    fn widths() {
        assert_eq!(str_width("abc"), 3);
        assert_eq!(str_width("안녕"), 4);
        assert_eq!(str_width("日本"), 4);
        assert_eq!(str_width("🙂"), 2);
        assert_eq!(str_width("é"), 1);
        assert_eq!(str_width("e\u{301}"), 1);
        assert_eq!(input_rows(2, "", 80), 1);
        assert_eq!(input_rows(2, &"x".repeat(78), 80), 1);
        assert_eq!(input_rows(2, &"x".repeat(79), 80), 2);
        assert_eq!(input_rows(2, &"가".repeat(40), 80), 2); // 2 + 80 columns
    }

    fn wrap_all(width: usize, pieces: &[&str]) -> String {
        let mut w = Wrap::new(width, 2, 2);
        let mut s: String = pieces.iter().map(|p| w.push(p)).collect();
        s.push_str(&w.finish());
        s
    }

    #[test]
    fn wrap_breaks_between_words_with_indent() {
        let s = wrap_all(12, &["the quick ", "brown fox jumps"]);
        assert_eq!(s, "the quick\n  brown fox\n  jumps");
        for line in s.lines() {
            assert!(str_width(line) <= 12, "{line:?}");
        }
    }

    #[test]
    fn wrap_streaming_pieces_split_words() {
        assert_eq!(wrap_all(40, &["hel", "lo wor", "ld"]), "hello world");
        // leading space of the first token is dropped, newlines keep the indent
        assert_eq!(wrap_all(40, &[" Hi.\nBye"]), "Hi.\n  Bye");
    }

    #[test]
    fn wrap_hard_breaks_long_words_and_wide_chars() {
        let s = wrap_all(8, &["abcdefghijkl"]);
        assert_eq!(s, "abcdef\n  ghijkl");
        let s = wrap_all(8, &["가나다라마"]);
        for line in s.lines() {
            assert!(str_width(line) <= 8, "{line:?}");
        }
        assert_eq!(s.replace(['\n', ' '], ""), "가나다라마");
    }

    #[test]
    fn wrap_preview_shows_partial_word() {
        let mut w = Wrap::new(20, 2, 2);
        assert_eq!(w.push("Hel"), "");
        assert_eq!(w.preview(), Some((0, "Hel", 3)));
        assert_eq!(w.push("lo wor"), "Hello");
        assert_eq!(w.preview(), Some((1, "wor", 3)), "space before the partial word");
        assert_eq!(w.push("ld "), " world");
        assert_eq!(w.preview(), None, "nothing pending after a space");
        // a word that will not fit on this line gets no preview
        let mut w = Wrap::new(10, 2, 2);
        w.push("abcde ");
        w.push("fgh");
        assert_eq!(w.preview(), None);
    }

    #[test]
    fn colour_turn_tints_partial_word() {
        let (dir, model, tok) = tiny_files("tint", 9);
        let mut chat = Chat::load(args(model, tok)).unwrap();
        let mut out = Vec::new();
        chat.turn(&mut out, &Style { color: true }, "hello", &AtomicBool::new(false)).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("\x1b[K"), "previews are erased: {out:?}");
        if out.contains("\x1b[38;5;210m") {
            assert!(out.contains("D"), "cursor steps back over the preview");
        }
        // plain mode: no escapes at all
        let mut plain = Vec::new();
        chat.turn(&mut plain, &Style { color: false }, "hello", &AtomicBool::new(false)).unwrap();
        assert!(!String::from_utf8(plain).unwrap().contains('\x1b'));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn footer_format() {
        assert_eq!(fmt_footer(12, 0.5, 40, 256), "⎿  12 tok · 24.0 tok/s · ctx 40/256");
        assert_eq!(fmt_footer(0, 0.0, 7, 256), "⎿  0 tok · 0.0 tok/s · ctx 7/256");
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

    #[test]
    fn header_box_rows_line_up() {
        let h = fmt_header(
            &Style { color: false },
            Path::new("runs/모델/model.bin"),
            1_049_728,
            &tiny_cfg(),
            Backend::Avx512Mt,
            4,
        );
        let rows: Vec<&str> = h.lines().collect();
        assert!(rows[0].starts_with('╭') && rows.last().unwrap().starts_with('╰'));
        let w = str_width(rows[0]);
        for r in &rows {
            assert_eq!(str_width(r), w, "{r:?}");
        }
    }

    #[test]
    fn header_box_fits_width_with_long_path() {
        let long = format!("C:\\{}\\runs\\tiny-dd-1200s\\model.bin", "very-long-folder\\".repeat(20));
        let h = fmt_header(&Style { color: false }, Path::new(&long), 1_049_728, &tiny_cfg(), Backend::Avx512Mt, 4);
        for r in h.lines() {
            assert!(str_width(r) <= DEFAULT_WIDTH, "{} > {DEFAULT_WIDTH}: {r:?}", str_width(r));
        }
        assert!(h.contains("…") && h.contains("tiny-dd-1200s\\model.bin"), "{h}");
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
        assert_eq!(out.matches("● ").count(), 2, "{out}");
        assert!(out.contains("rllm chat") && out.contains("/help for commands"), "{out}");
        assert!(out.contains("temperature") && out.contains("new conversation"), "{out}");
        assert!(!out.contains("ignored"), "{out}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn run_tui_colour_draws_box_and_user_block() {
        let (dir, model, tok) = tiny_files("colour", 9);
        let mut input = std::io::Cursor::new("안녕 there\n/quit\n".as_bytes().to_vec());
        let mut out = Vec::new();
        let cancel = AtomicBool::new(false);
        run_tui(args(model, tok), &mut input, &mut out, Style { color: true }, &cancel).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains('╭') && out.contains("─"), "input rules / welcome box");
        assert!(out.contains(" > 안녕 there"), "user block");
        assert!(out.contains("\x1b[2A"), "cursor moved up into the input box");
        assert!(out.contains("\x1b[J"), "input box cleared after Enter");
        assert_eq!(out.matches("tok/s").count(), 1, "{out}");
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
        assert!(out.contains("Interrupted"), "{out}");
        assert_eq!(chat.history.len(), 1);
        assert!(chat.history[0].1.is_empty());
        assert!(!cancel.load(Relaxed), "flag is consumed");
        let _ = std::fs::remove_dir_all(dir);
    }
}
