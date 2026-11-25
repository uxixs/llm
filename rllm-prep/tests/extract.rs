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

const ELI5: &str = "explainlikeimfive";
const NSQ: &str = "NoStupidQuestions";
const ASK: &str = "askscience";

fn sub(id: &str, subreddit: &str, title: &str, selftext: &str) -> String {
    json!({"id": id, "subreddit": subreddit, "title": title, "selftext": selftext, "score": 5, "author": "u"}).to_string()
}

fn com(id: &str, subreddit: &str, link: &str, parent: &str, body: &str, score: i64, author: &str) -> String {
    json!({"id": id, "subreddit": subreddit, "link_id": format!("t3_{link}"), "parent_id": parent, "body": body, "score": score, "author": author}).to_string()
}

fn ans(n: usize) -> String {
    format!("This is a reasonably long answer number {n} for testing.")
}

/// Writes one RS and one RC monthly fixture; returns (rs, rc).
fn make_month(name: &str, rs: &[String], rc: &[String]) -> (PathBuf, PathBuf) {
    let d = tmp(name);
    let (s, c) = (d.join("RS_2014-01.zst"), d.join("RC_2014-01.zst"));
    write_zst(&s, rs);
    write_zst(&c, rc);
    (s, c)
}

fn run_files(rs: Vec<PathBuf>, rc: Vec<PathBuf>, name: &str, max_bytes: u64) -> (PathBuf, ExtractStats) {
    let out = tmp(&format!("{name}_out"));
    let stats = run(&ExtractConfig {
        rs_files: rs,
        rc_files: rc,
        out_dir: out.clone(),
        max_pretrain_bytes: max_bytes,
    })
    .unwrap();
    (out, stats)
}

fn run_extract(name: &str, rs: &[String], rc: &[String], max_bytes: u64) -> (PathBuf, ExtractStats) {
    let (s, c) = make_month(name, rs, rc);
    run_files(vec![s], vec![c], name, max_bytes)
}

fn pretrain_text(out: &Path) -> String {
    fs::read_to_string(out.join("pretrain_train.txt")).unwrap()
        + &fs::read_to_string(out.join("pretrain_val.txt")).unwrap()
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
    let (out, _) = run_extract(
        "deleted",
        &[sub("s1", ELI5, "Why sky blue?", "")],
        &[
            com("c1", ELI5, "s1", "t3_s1", "[removed]", 500, "bob"),
            com("c2", ELI5, "s1", "t3_s1", &ans(2), 30, "bob"),
        ],
        MB,
    );
    let pairs = all_sft(&out);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0]["a"], ans(2));
    assert_eq!(pairs[0]["q"], "Why sky blue?");
}

#[test]
fn best_top_level_comment_chosen() {
    let (out, _) = run_extract(
        "best",
        &[sub("s1", ELI5, "Title", "Body text")],
        &[
            com("c1", ELI5, "s1", "t3_s1", &ans(1), 20, "a"),
            com("c2", ELI5, "s1", "t1_c1", &ans(2), 900, "b"), // reply, higher score
            com("c3", ELI5, "s1", "t3_s1", &ans(3), 15, "c"),
            com("c4", ELI5, "s1", "t3_s1", &ans(4), 999, "AutoModerator"),
            com("c5", ELI5, "s1", "t3_s1", "too short", 800, "d"),
            com("c6", ELI5, "s1", "t3_s1", &ans(6), 9, "e"), // below score 10
        ],
        MB,
    );
    let pairs = all_sft(&out);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0]["a"], ans(1));
    assert_eq!(pairs[0]["q"], "Title\nBody text");
}

#[test]
fn askscience_not_in_sft() {
    let (out, stats) = run_extract(
        "ask",
        &[sub("s1", ASK, "Why?", "")],
        &[com("c1", ASK, "s1", "t3_s1", &ans(1), 100, "a")],
        MB,
    );
    assert!(all_sft(&out).is_empty());
    assert_eq!(stats.sft_total, 0);
    // but its text is in pretraining
    assert!(pretrain_text(&out).contains(&ans(1)));
}

#[test]
fn nsq_cap_20_percent() {
    let mk = |sr: &str, prefix: &str, n: usize| {
        let mut s = vec![];
        let mut c = vec![];
        for i in 0..n {
            let id = format!("{prefix}{i}");
            s.push(sub(&id, sr, &format!("Question {id}"), ""));
            c.push(com(&format!("c{id}"), sr, &id, &format!("t3_{id}"), &ans(i), 50 + i as i64, "a"));
        }
        (s, c)
    };
    let (mut rs, mut rc) = mk(ELI5, "e", 4);
    let (ns, mut nc) = mk(NSQ, "n", 4);
    // an NSQ answer over 300 chars is never used
    nc.push(com("clong", NSQ, "n0", "t3_n0", &"x".repeat(400), 9999, "a"));
    rs.extend(ns);
    rc.extend(nc);
    let (out, stats) = run_extract("cap", &rs, &rc, MB);
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
        s.push(sub(&id, ELI5, &format!("Q{i}"), ""));
        c.push(com(&format!("c{i}"), ELI5, &id, &format!("t3_{id}"), &ans(i), 50, "a"));
    }
    let (out, _) = run_extract("split", &s, &c, MB);
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
    let mut got: Vec<Value> = vec![];
    for_each_line(&path, |l| got.push(serde_json::from_slice(l).unwrap())).unwrap();
    assert_eq!(got.len(), 50);
    assert_eq!(got[49]["id"], 49);
}

#[test]
fn monthly_filters_by_subreddit() {
    let (out, stats) = run_extract(
        "filter",
        &[
            sub("s1", "ExplainLikeImFive", "KEEPSUB1", ""), // case-insensitive match
            sub("s2", "nostupidquestions", "KEEPSUB2", ""),
            sub("s3", "ASKSCIENCE", "KEEPSUB3", ""),
            sub("s4", "politics", "DROPSUB", ""),
            sub("s5", "explainlikeimfive_extra", "DROPSUB2", ""),
        ],
        &[
            com("c1", "ExplainLikeImFive", "s1", "t3_s1", "KEEPCOM1 answer text", 30, "a"),
            com("c2", "askscience", "s3", "t3_s3", "KEEPCOM3 answer text", 30, "a"),
            com("c3", "gaming", "s4", "t3_s4", "DROPCOM answer text", 30, "a"),
        ],
        MB,
    );
    let text = pretrain_text(&out);
    for k in ["KEEPSUB1", "KEEPSUB2", "KEEPSUB3", "KEEPCOM1", "KEEPCOM3"] {
        assert!(text.contains(k), "missing {k}");
    }
    assert!(!text.contains("DROP"), "non-matching subreddit leaked");
    assert_eq!(stats.lines_read, 8);
    assert_eq!(stats.lines_kept, 5);
}

#[test]
fn prefilter_false_positive_rejected() {
    // Passes the byte prefilter (nested object has an ELI5 subreddit) but the
    // real top-level subreddit is "politics".
    let line = json!({
        "id": "s9", "subreddit": "politics", "title": "FALSEPOSITIVE", "selftext": "",
        "crosspost_parent_list": [{"subreddit": "explainlikeimfive"}]
    })
    .to_string();
    assert!(prefilter(line.as_bytes()), "fixture must pass the prefilter");
    assert!(prefilter(br#"{"subreddit":"AskScience","id":"x"}"#));
    assert!(!prefilter(br#"{"subreddit":"politics","id":"x"}"#));
    assert!(!prefilter(br#"{"subreddit":"askscience2","id":"x"}"#));
    let (out, stats) = run_extract("fp", &[line, sub("s1", ELI5, "REALSUB", "")], &[], MB);
    let text = pretrain_text(&out);
    assert!(text.contains("REALSUB"));
    assert!(!text.contains("FALSEPOSITIVE"));
    assert_eq!(stats.lines_kept, 1);
}

#[test]
fn missing_subreddit_warns_not_errors() {
    let (out, stats) = run_extract(
        "missing_sub",
        &[sub("s1", ELI5, "Only eli5", "")],
        &[com("c1", ELI5, "s1", "t3_s1", &ans(1), 30, "a")],
        MB,
    );
    assert_eq!(stats.missing_subs, vec![NSQ.to_string(), ASK.to_string()]);
    assert_eq!(all_sft(&out).len(), 1);
}

#[test]
fn comment_without_submission_pretrain_only() {
    // submission s1 is in RS, but the comment on s2 has no submission there
    let (out, stats) = run_extract(
        "orphan",
        &[sub("s1", ELI5, "Has submission", "")],
        &[
            com("c1", ELI5, "s1", "t3_s1", &ans(1), 30, "a"),
            com("c2", ELI5, "s2", "t3_s2", "ORPHANCOMMENT long enough to qualify", 500, "a"),
        ],
        MB,
    );
    let pairs = all_sft(&out);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0]["a"], ans(1));
    assert_eq!(stats.sft_total, 1);
    assert!(pretrain_text(&out).contains("ORPHANCOMMENT"));
}

#[test]
fn multiple_files_per_kind() {
    let d = tmp("multi");
    let (s1, s2) = (d.join("RS_2014-01.zst"), d.join("RS_2014-02.zst"));
    let (c1, c2) = (d.join("RC_2014-01.zst"), d.join("RC_2014-02.zst"));
    write_zst(&s1, &[sub("s1", ELI5, "Jan question", "")]);
    write_zst(&s2, &[sub("s2", ELI5, "Feb question", "")]);
    write_zst(&c1, &[com("c1", ELI5, "s1", "t3_s1", &ans(1), 30, "a")]);
    write_zst(&c2, &[com("c2", ELI5, "s2", "t3_s2", &ans(2), 30, "a")]);
    let (out, _) = run_files(vec![s1, s2], vec![c1, c2], "multi", MB);
    assert_eq!(all_sft(&out).len(), 2);
}

#[test]
fn malformed_lines_are_skipped_and_counted() {
    let (out, stats) = run_extract(
        "bad",
        &[
            "{not json \"subreddit\":\"explainlikeimfive\"".to_string(),
            sub("s1", ELI5, "Good question", ""),
            json!({"id": "s2", "subreddit": ELI5}).to_string(), // missing title
            "42".to_string(),                                   // fails prefilter: not counted
        ],
        &[
            "garbage \"subreddit\":\"askscience\"".to_string(),
            com("c1", ELI5, "s1", "t3_s1", &ans(1), 40, "a"),
            json!({"id": "c2", "subreddit": ELI5, "link_id": "t3_s1"}).to_string(), // no body
        ],
        MB,
    );
    assert_eq!(all_sft(&out).len(), 1);
    assert_eq!(stats.skipped, 4);
}

#[test]
fn missing_file_error_names_it() {
    let d = tmp("missing");
    let (s, c) = make_month("missing_ok", &[], &[]);
    let gone = d.join("RC_2099-01.zst");
    let err = run(&ExtractConfig {
        rs_files: vec![s],
        rc_files: vec![c, gone],
        out_dir: tmp("missing_out"),
        max_pretrain_bytes: 1000,
    })
    .unwrap_err();
    assert!(err.to_string().contains("RC_2099-01.zst"), "{err}");
}

#[test]
fn pretrain_round_robin_and_limit() {
    // one RS stream mixed across the three subreddits
    let mut rs = vec![];
    for i in 0..200 {
        for (tag, sr) in [("e", ELI5), ("n", NSQ), ("a", ASK)] {
            rs.push(sub(&format!("{tag}{i}"), sr, &format!("{tag} title {i} {}", "w".repeat(200)), ""));
        }
    }
    let (out, _) = run_extract("rr", &rs, &[], 20_000);
    let text = pretrain_text(&out);
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
    // one mixed RS and one mixed RC; submissions alone (~100 KB) exceed the limit
    let mut s: Vec<String> = vec![];
    let mut c: Vec<String> = vec![];
    for i in 0..500 {
        let sr = [ELI5, NSQ, ASK][i % 3];
        s.push(sub(&format!("s{i}"), sr, &format!("title {i} {}", "w".repeat(200)), ""));
        c.push(com(&format!("c{i}"), sr, &format!("s{i}"), &format!("t3_s{i}"), &format!("COMMENTBODY {i}"), 1, "a"));
    }
    let (out, _) = run_extract("mix", &s, &c, 10_000);
    let text = pretrain_text(&out);
    assert!(text.contains("COMMENTBODY"), "no comment bodies in pretrain output");
    assert!(text.contains("title "));
}

fn read_all(out: &Path) -> Vec<String> {
    ["sft_train.jsonl", "sft_val.jsonl", "pretrain_train.txt", "pretrain_val.txt"]
        .iter()
        .map(|f| fs::read_to_string(out.join(f)).unwrap())
        .collect()
}

#[test]
fn plain_jsonl_input_matches_zst() {
    let rs = vec![sub("a1", ELI5, "Why is the sky blue", "please explain"), sub("a2", NSQ, "Why yawn", "")];
    let rc = vec![
        com("c1", ELI5, "a1", "t3_a1", &ans(1), 50, "bob"),
        com("c2", NSQ, "a2", "t3_a2", &ans(2), 40, "amy"),
    ];
    let d = tmp("plain_vs_zst");
    let (zs, zc) = (d.join("RS.zst"), d.join("RC.zst"));
    write_zst(&zs, &rs);
    write_zst(&zc, &rc);
    let (js, jc) = (d.join("posts.jsonl"), d.join("comments.jsonl"));
    fs::write(&js, rs.join("\n") + "\n").unwrap();
    fs::write(&jc, rc.join("\n") + "\n").unwrap();
    let (oz, sz) = run_files(vec![zs], vec![zc], "plain_zst", 40 * MB);
    let (oj, sj) = run_files(vec![js], vec![jc], "plain_jsonl", 40 * MB);
    assert_eq!(sz.lines_read, 4);
    assert_eq!(sj.lines_read, sz.lines_read);
    assert_eq!(sj.lines_kept, sz.lines_kept);
    assert!(sj.sft_total >= 1);
    assert_eq!(read_all(&oj), read_all(&oz));
}

#[test]
fn meta_key_line_parsed() {
    let post = format!(
        r#"{{"_meta":{{"source":"arctic_shift","retrieved_on":1700000000}},"id":"m1","subreddit":"{ELI5}","title":"Why meta","selftext":"body here","score":5,"author":"u"}}"#
    );
    let comment = format!(
        r#"{{"_meta":{{"source":"arctic_shift"}},"id":"mc1","subreddit":"{ELI5}","link_id":"t3_m1","parent_id":"t3_m1","body":"{}","score":30,"author":"bob"}}"#,
        ans(9)
    );
    let d = tmp("meta_key");
    let (s, c) = (d.join("p.jsonl"), d.join("c.jsonl"));
    fs::write(&s, post + "\n").unwrap();
    fs::write(&c, comment + "\n").unwrap();
    let (out, st) = run_files(vec![s], vec![c], "meta_key", 40 * MB);
    assert_eq!(st.lines_kept, 2);
    assert_eq!(st.skipped, 0);
    assert_eq!(all_sft(&out).len(), 1);
}
