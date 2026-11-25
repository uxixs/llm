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

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
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
