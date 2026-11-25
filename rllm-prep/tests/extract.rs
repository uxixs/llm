use rllm_prep::extract::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rllm_prep_{}_{}", name, std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn write_zst(path: &Path, lines: &[String]) {
    let raw = lines.join("\n");
    fs::write(path, zstd::encode_all(raw.as_bytes(), 3).unwrap()).unwrap();
}

fn sub(id: &str, title: &str, selftext: &str) -> String {
    json!({"id": id, "subreddit": "x", "title": title, "selftext": selftext, "score": 5, "author": "u"}).to_string()
}

fn com(id: &str, link: &str, parent: &str, body: &str, score: i64, author: &str) -> String {
    json!({"id": id, "link_id": format!("t3_{link}"), "parent_id": parent, "body": body, "score": score, "author": author}).to_string()
}

fn ans(n: usize) -> String {
    format!("This is a reasonably long answer number {n} for testing.")
}

type SubData = (&'static str, Vec<String>, Vec<String>);

/// Writes all six dumps (empty unless given) into a fresh dir.
fn make_dumps(name: &str, data: &[SubData]) -> PathBuf {
    let d = tmp(name);
    for s in ["explainlikeimfive", "NoStupidQuestions", "askscience"] {
        let (subs, coms) = data
            .iter()
            .find(|(n, _, _)| *n == s)
            .map(|(_, a, b)| (a.clone(), b.clone()))
            .unwrap_or_default();
        write_zst(&d.join(format!("{s}_submissions.zst")), &subs);
        write_zst(&d.join(format!("{s}_comments.zst")), &coms);
    }
    d
}

fn run_extract(dumps: &Path, name: &str, max_bytes: u64) -> (PathBuf, ExtractStats) {
    let out = tmp(&format!("{name}_out"));
    let stats = run(&ExtractConfig {
        subs_dir: dumps.to_path_buf(),
        out_dir: out.clone(),
        max_pretrain_bytes: max_bytes,
    })
    .unwrap();
    (out, stats)
}

fn read_sft(out: &Path, file: &str) -> Vec<Value> {
    fs::read_to_string(out.join(file))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn all_sft(out: &Path) -> Vec<Value> {
    let mut v = read_sft(out, "sft_train.jsonl");
    v.extend(read_sft(out, "sft_val.jsonl"));
    v
}

const MB: u64 = 1024 * 1024;

#[test]
fn clean_text_rules() {
    assert_eq!(clean_text("[deleted]"), None);
    assert_eq!(clean_text("[removed]"), None);
    assert_eq!(clean_text("   \n\t "), None);
    assert_eq!(clean_text(""), None);
    assert_eq!(clean_text("see [this page](https://a.com/x) now").unwrap(), "see this page now");
    assert_eq!(
        clean_text("go to https://a.com/x?y=1 or http://b.org today").unwrap(),
        "go to or today"
    );
    assert_eq!(clean_text("a  b\n\n c\t\td").unwrap(), "a b c d");
    assert_eq!(clean_text("https://only.link/here"), None);
    assert_eq!(clean_text("[ok]").unwrap(), "[ok]");
}

#[test]
fn split_is_stable_and_about_two_percent() {
    assert_eq!(split_is_val("abc123"), split_is_val("abc123"));
    let n = 50_000;
    let v = (0..n).filter(|i| split_is_val(&format!("id{i}"))).count();
    assert!(v > n / 100 && v < n * 3 / 100, "val fraction {v}/{n}");
}

#[test]
fn deleted_answer_skipped() {
    let dumps = make_dumps(
        "deleted",
        &[(
            "explainlikeimfive",
            vec![sub("s1", "Why sky blue?", "")],
            vec![
                com("c1", "s1", "t3_s1", "[removed]", 500, "bob"),
                com("c2", "s1", "t3_s1", &ans(2), 30, "bob"),
            ],
        )],
    );
    let (out, _) = run_extract(&dumps, "deleted", MB);
    let pairs = all_sft(&out);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0]["a"], ans(2));
    assert_eq!(pairs[0]["q"], "Why sky blue?");
}

#[test]
fn best_top_level_comment_chosen() {
    let dumps = make_dumps(
        "best",
        &[(
            "explainlikeimfive",
            vec![sub("s1", "Title", "Body text")],
            vec![
                com("c1", "s1", "t3_s1", &ans(1), 20, "a"),
                com("c2", "s1", "t1_c1", &ans(2), 900, "b"), // reply, higher score
                com("c3", "s1", "t3_s1", &ans(3), 15, "c"),
                com("c4", "s1", "t3_s1", &ans(4), 999, "AutoModerator"),
                com("c5", "s1", "t3_s1", "too short", 800, "d"),
                com("c6", "s1", "t3_s1", &ans(6), 9, "e"), // below score 10
            ],
        )],
    );
    let (out, _) = run_extract(&dumps, "best", MB);
    let pairs = all_sft(&out);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0]["a"], ans(1));
    assert_eq!(pairs[0]["q"], "Title\nBody text");
}

#[test]
fn askscience_not_in_sft() {
    let dumps = make_dumps(
        "ask",
        &[(
            "askscience",
            vec![sub("s1", "Why?", "")],
            vec![com("c1", "s1", "t3_s1", &ans(1), 100, "a")],
        )],
    );
    let (out, stats) = run_extract(&dumps, "ask", MB);
    assert!(all_sft(&out).is_empty());
    assert_eq!(stats.sft_total, 0);
    // but its text is in pretraining
    let pre = fs::read_to_string(out.join("pretrain_train.txt")).unwrap()
        + &fs::read_to_string(out.join("pretrain_val.txt")).unwrap();
    assert!(pre.contains(&ans(1)));
}

#[test]
fn nsq_cap_20_percent() {
    let mk = |prefix: &str, n: usize| {
        let mut s = vec![];
        let mut c = vec![];
        for i in 0..n {
            let id = format!("{prefix}{i}");
            s.push(sub(&id, &format!("Question {id}"), ""));
            c.push(com(&format!("c{id}"), &id, &format!("t3_{id}"), &ans(i), 50 + i as i64, "a"));
        }
        (s, c)
    };
    let (es, ec) = mk("e", 4);
    let (ns, mut nc) = mk("n", 4);
    // an NSQ answer over 300 chars is never used
    nc.push(com("clong", "n0", "t3_n0", &"x".repeat(400), 9999, "a"));
    let dumps = make_dumps("cap", &[("explainlikeimfive", es, ec), ("NoStupidQuestions", ns, nc)]);
    let (out, stats) = run_extract(&dumps, "cap", MB);
    let pairs = all_sft(&out);
    let nsq = pairs.iter().filter(|p| p["q"].as_str().unwrap().contains("Question n")).count();
    assert_eq!(pairs.len(), 5);
    assert_eq!(nsq, 1);
    assert_eq!(stats.sft_total, 5);
    assert!(pairs.iter().all(|p| p["a"].as_str().unwrap().len() <= 300));
}

#[test]
fn val_split_by_submission() {
    let n = 600;
    let mut s = vec![];
    let mut c = vec![];
    for i in 0..n {
        let id = format!("s{i}");
        s.push(sub(&id, &format!("Q{i}"), ""));
        c.push(com(&format!("c{i}"), &id, &format!("t3_{id}"), &ans(i), 50, "a"));
    }
    let dumps = make_dumps("split", &[("explainlikeimfive", s, c)]);
    let (out, _) = run_extract(&dumps, "split", MB);
    let train = read_sft(&out, "sft_train.jsonl");
    let val = read_sft(&out, "sft_val.jsonl");
    assert_eq!(train.len() + val.len(), n);
    assert!(!val.is_empty(), "expected some val pairs out of {n}");
    let idx = |p: &Value| -> usize { p["q"].as_str().unwrap()[1..].parse().unwrap() };
    for p in &val {
        assert!(split_is_val(&format!("s{}", idx(p))));
        assert_eq!(p["a"], ans(idx(p)));
    }
    for p in &train {
        assert!(!split_is_val(&format!("s{}", idx(p))));
        assert_eq!(p["a"], ans(idx(p)));
    }
}

#[test]
fn reads_long_window_zst() {
    let dir = tmp("longwin");
    let path = dir.join("f.zst");
    let lines: Vec<String> = (0..50).map(|i| json!({"id": i, "t": "hello"}).to_string()).collect();
    let mut enc = zstd::stream::Encoder::new(Vec::new(), 3).unwrap();
    enc.long_distance_matching(true).unwrap();
    enc.window_log(31).unwrap();
    enc.write_all(lines.join("\n").as_bytes()).unwrap();
    fs::write(&path, enc.finish().unwrap()).unwrap();
    let vals: Vec<Value> = open_zst_ndjson(&path).unwrap().collect();
    assert_eq!(vals.len(), 50);
    assert_eq!(vals[49]["id"], 49);
}

#[test]
fn malformed_lines_are_skipped_and_counted() {
    let dumps = make_dumps(
        "bad",
        &[(
            "explainlikeimfive",
            vec![
                "{not json".to_string(),
                sub("s1", "Good question", ""),
                json!({"id": "s2"}).to_string(), // missing fields
                "42".to_string(),
            ],
            vec![
                "garbage".to_string(),
                com("c1", "s1", "t3_s1", &ans(1), 40, "a"),
                json!({"id": "c2", "link_id": "t3_s1"}).to_string(),
            ],
        )],
    );
    let (out, stats) = run_extract(&dumps, "bad", MB);
    assert_eq!(all_sft(&out).len(), 1);
    assert_eq!(stats.skipped_sft, 5);
}

#[test]
fn missing_file_error_names_it() {
    let d = tmp("missing");
    let err = run(&ExtractConfig {
        subs_dir: d,
        out_dir: tmp("missing_out"),
        max_pretrain_bytes: 1000,
    })
    .unwrap_err();
    assert!(err.to_string().contains("explainlikeimfive_submissions.zst"), "{err}");
}

#[test]
fn subs_matched_case_insensitively() {
    let d = tmp("case");
    for c in ["explainlikeimfive", "nostupidquestions", "AskScience"] {
        write_zst(&d.join(format!("{c}_submissions.zst")), &[sub("a", "T", "")]);
        write_zst(&d.join(format!("{c}_comments.zst")), &[]);
    }
    let (out, _) = run_extract(&d, "case", MB);
    assert!(out.join("pretrain_train.txt").exists());
}

#[test]
fn pretrain_round_robin_and_limit() {
    let mk = |tag: &str| -> Vec<String> {
        (0..200)
            .map(|i| sub(&format!("{tag}{i}"), &format!("{tag} title {i} {}", "w".repeat(200)), ""))
            .collect()
    };
    let dumps = make_dumps(
        "rr",
        &[
            ("explainlikeimfive", mk("e"), vec![]),
            ("NoStupidQuestions", mk("n"), vec![]),
            ("askscience", mk("a"), vec![]),
        ],
    );
    let (out, _) = run_extract(&dumps, "rr", 20_000);
    let text = fs::read_to_string(out.join("pretrain_train.txt")).unwrap()
        + &fs::read_to_string(out.join("pretrain_val.txt")).unwrap();
    assert!(text.len() >= 20_000 && text.len() < 21_000, "len {}", text.len());
    let mut counts: HashMap<char, usize> = HashMap::new();
    for d in text.split("\n\x1e\n").filter(|d| !d.is_empty()) {
        *counts.entry(d.chars().next().unwrap()).or_default() += 1;
    }
    let (e, n, a) = (counts[&'e'], counts[&'n'], counts[&'a']);
    assert!(e.abs_diff(n) <= 1 && n.abs_diff(a) <= 1, "{e} {n} {a}");
}

#[test]
fn pretrain_includes_comments_when_submissions_exceed_limit() {
    let s: Vec<String> = (0..500)
        .map(|i| sub(&format!("s{i}"), &format!("title {i} {}", "w".repeat(200)), ""))
        .collect();
    let c: Vec<String> = (0..500)
        .map(|i| com(&format!("c{i}"), &format!("s{i}"), &format!("t3_s{i}"), &format!("COMMENTBODY {i}"), 1, "a"))
        .collect();
    let dumps = make_dumps("mix", &[("askscience", s, c)]);
    // submissions alone (~100 KB) are far over the 10 KB limit
    let (out, _) = run_extract(&dumps, "mix", 10_000);
    let text = fs::read_to_string(out.join("pretrain_train.txt")).unwrap()
        + &fs::read_to_string(out.join("pretrain_val.txt")).unwrap();
    assert!(text.contains("COMMENTBODY"), "no comment bodies in pretrain output");
    assert!(text.contains("title "));
}
