//! TinyStories (pretraining) and DailyDialog (multi-turn SFT) preparation.

/// Story separator used by the TinyStories text files.
pub const STORY_SEP: &str = "<|endoftext|>";

/// Stories in `text`, trimmed, empty ones skipped. When `complete_only` is set and
/// `text` does not end with a separator (a file cut by a ranged download), the last,
/// partial story is dropped.
pub fn split_stories(text: &str, complete_only: bool) -> Vec<&str> {
    let mut parts: Vec<&str> = text.split(STORY_SEP).collect();
    if complete_only && !text.trim_end().ends_with(STORY_SEP) {
        parts.pop();
    }
    parts.into_iter().map(str::trim).filter(|s| !s.is_empty()).collect()
}

/// Undoes DailyDialog's tokenized spacing: `"Say , Jim , it ’ s late ?"` becomes
/// `"Say, Jim, it's late?"`. Curly apostrophes become ASCII; whitespace collapses.
pub fn clean_dd_turn(s: &str) -> String {
    let words: Vec<String> = s.split_whitespace().map(|w| w.replace(['’', '‘'], "'")).collect();
    let mut out = String::new();
    let mut glue_next = false;
    for w in words {
        let attach = w.chars().all(|c| ",.?!;:)".contains(c)) || w == "'" || w.starts_with('\'');
        if !out.is_empty() && !attach && !glue_next {
            out.push(' ');
        }
        glue_next = w == "'" || w == "(";
        out.push_str(&w);
    }
    out
}

/// One DailyDialog line (`turn __eou__ turn __eou__ ...`) as cleaned, non-empty turns.
pub fn parse_dialogue(line: &str) -> Vec<String> {
    line.split("__eou__").map(clean_dd_turn).filter(|t| !t.is_empty()).collect()
}

/// Training views of one dialogue: speaker A as user (all turns) and speaker B as user
/// (from turn 1). Each view keeps an even number of turns (user, assistant pairs) and
/// is dropped if it has fewer than 2.
pub fn dialogue_views(turns: &[String]) -> Vec<Vec<String>> {
    [0, 1]
        .into_iter()
        .filter_map(|start| {
            let v = turns.get(start..)?;
            let v = &v[..v.len() / 2 * 2];
            (v.len() >= 2).then(|| v.to_vec())
        })
        .collect()
}

/// Documents from `a` and `b`, interleaved so that about `b_pct` percent of the bytes come
/// from `b`, each source in its own order, up to `max_bytes` in total. The mix ends when
/// the source it needs next runs out, so the ratio holds throughout.
pub fn interleave<'a>(a: &[&'a str], b: &[&'a str], b_pct: u64, max_bytes: usize) -> Vec<&'a str> {
    let (mut ia, mut ib, mut total, mut from_b) = (0usize, 0usize, 0usize, 0usize);
    let mut out = Vec::new();
    loop {
        let take_b = b_pct > 0 && (from_b as u64) * 100 < b_pct * total as u64;
        let doc = if take_b { b.get(ib) } else { a.get(ia) };
        let Some(&doc) = doc else { break };
        if total + doc.len() > max_bytes {
            break;
        }
        if take_b {
            ib += 1;
            from_b += doc.len();
        } else {
            ia += 1;
        }
        total += doc.len();
        out.push(doc);
    }
    out
}

/// A turn as people type it in a chat box: lowercase, no final period or exclamation mark
/// ("How are you?" stays a question: "how are you?").
pub fn casual(s: &str) -> String {
    let t = s.trim().trim_end_matches(['.', '!']).trim_end();
    t.to_lowercase()
}

/// Makes every user turn (even index) of a dialogue view casual.
pub fn casualize_users(turns: &mut [String]) {
    for t in turns.iter_mut().step_by(2) {
        *t = casual(t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn interleave_hits_ratio_and_limit() {
        let a: Vec<String> = (0..5000).map(|i| format!("story {i} ").repeat(10)).collect();
        let b: Vec<String> = (0..1000).map(|i| format!("wiki {i} ").repeat(20)).collect();
        let (a, b): (Vec<&str>, Vec<&str>) = (a.iter().map(|s| s.as_str()).collect(), b.iter().map(|s| s.as_str()).collect());
        let mixed = interleave(&a, &b, 20, 200_000);
        let total: usize = mixed.iter().map(|d| d.len()).sum();
        let wiki: usize = mixed.iter().filter(|d| d.starts_with("wiki")).map(|d| d.len()).sum();
        assert!(total <= 200_000 && total > 190_000, "{total}");
        let share = wiki as f64 / total as f64;
        assert!((0.17..0.23).contains(&share), "wiki share {share}");
        // documents stay in their source order
        let firsts: Vec<&str> = mixed.iter().filter(|d| d.starts_with("story")).take(3).copied().collect();
        assert_eq!(firsts, [a[0], a[1], a[2]]);
        // 0% means only `a`; an exhausted source ends the mix early
        assert!(interleave(&a, &b, 0, 50_000).iter().all(|d| d.starts_with("story")));
        assert_eq!(interleave(&a[..2], &b, 50, usize::MAX).iter().filter(|d| d.starts_with("story")).count(), 2);
    }

    #[test]
    fn casual_turns() {
        assert_eq!(casual("Hi, Tom."), "hi, tom");
        assert_eq!(casual("Wow!!"), "wow");
        assert_eq!(casual("How are you?"), "how are you?");
        assert_eq!(casual("  I'm FINE. "), "i'm fine");
        let mut v = s(&["Hello.", "Hi there.", "What's up?", "Not much."]);
        casualize_users(&mut v);
        assert_eq!(v, s(&["hello", "Hi there.", "what's up?", "Not much."]), "assistant turns untouched");
    }

    #[test]
    fn stories_split_and_trimmed() {
        let t = "Once a cat.\n<|endoftext|>\n\n<|endoftext|>\nA dog ran.\n<|endoftext|>\n";
        assert_eq!(split_stories(t, true), vec!["Once a cat.", "A dog ran."]);
    }

    #[test]
    fn partial_last_story_dropped() {
        let t = "One.\n<|endoftext|>\nTwo is cut of";
        assert_eq!(split_stories(t, true), vec!["One."]);
        assert_eq!(split_stories(t, false), vec!["One.", "Two is cut of"]);
    }

    #[test]
    fn dd_spacing_undone() {
        assert_eq!(
            clean_dd_turn(" Say , Jim , how about going for a few beers after dinner ? "),
            "Say, Jim, how about going for a few beers after dinner?"
        );
        assert_eq!(clean_dd_turn("I ’ m fine . It ' s ok !"), "I'm fine. It's ok!");
        assert_eq!(clean_dd_turn("Wait ... what ; really : yes"), "Wait... what; really: yes");
        assert_eq!(clean_dd_turn("a  \t b"), "a b");
    }

    #[test]
    fn dialogue_parsed() {
        let l = "Hi , Tom . __eou__ Hello ! __eou__  __eou__ Bye . __eou__\n";
        assert_eq!(parse_dialogue(l), s(&["Hi, Tom.", "Hello!", "Bye."]));
    }

    #[test]
    fn views_both_speakers_even_turns() {
        let d = s(&["a1", "b1", "a2", "b2", "a3"]);
        assert_eq!(
            dialogue_views(&d),
            vec![s(&["a1", "b1", "a2", "b2"]), s(&["b1", "a2", "b2", "a3"])]
        );
        assert_eq!(dialogue_views(&s(&["a1", "b1"])), vec![s(&["a1", "b1"])]);
        assert!(dialogue_views(&s(&["a1"])).is_empty());
    }
}
