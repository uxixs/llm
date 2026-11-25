//! Synthetic dialogues that teach a small model who it is and to recall facts the user
//! gave earlier in the conversation. Train and test use disjoint names and places, so the
//! memory test measures copying from context, not memorised pairs.

use rllm::rng::Rng;

pub const BOT_NAME: &str = "Rllm";

pub const TRAIN_NAMES: &[&str] = &[
    "Tom", "Anna", "Mike", "Sara", "John", "Emma", "David", "Lily", "Chris", "Mia", "Alex", "Olivia",
    "Ben", "Grace", "Sam", "Chloe", "Jack", "Ella", "Ryan", "Zoe", "Kevin", "Amy", "Daniel", "Lucy",
    "Mark", "Kate", "Paul", "Ruby", "Eric", "Nora", "Jason", "Ivy", "Leo", "Hannah", "Noah", "Maya",
    "Luke", "Ava", "Adam", "Isla", "Peter", "Clara", "Henry", "Alice", "Oscar", "Rose", "Victor",
    "Julia", "Max", "Sophie", "Ethan", "Laura", "Owen", "Eva", "Tyler", "Megan", "Dylan", "Jenny",
    "Minho", "Jisoo", "Hana", "Yuna", "Joon", "Seo", "Hyun", "Min", "Kenji", "Yuki", "Haruto", "Aiko",
    "Wei", "Li", "Mei", "Chen", "Ravi", "Priya", "Arjun", "Anika", "Omar", "Layla", "Ali", "Sana",
    "Carlos", "Maria", "Diego", "Lucia", "Pedro", "Sofia", "Luca", "Giulia", "Hans", "Greta", "Ivan",
    "Olga", "Pierre", "Camille", "Lars", "Freya", "Kofi", "Amara", "Tariq", "Nadia", "Felix", "Iris",
];

pub const TEST_NAMES: &[&str] = &[
    "Jesse", "Morgan", "Riley", "Casey", "Jordan", "Taylor", "Quinn", "Avery", "Harper", "Rowan",
    "Sunwoo", "Daeho", "Akira", "Bao", "Nikhil", "Farah", "Mateo", "Elena", "Bruno", "Ingrid",
];

pub const TRAIN_PLACES: &[&str] = &[
    "London", "Paris", "Tokyo", "Seoul", "New York", "Berlin", "Madrid", "Rome", "Sydney", "Toronto",
    "Chicago", "Boston", "Osaka", "Beijing", "Shanghai", "Bangkok", "Hanoi", "Manila", "Jakarta",
    "Delhi", "Mumbai", "Cairo", "Lagos", "Nairobi", "Lima", "Bogota", "Mexico City", "Vancouver",
    "Dublin", "Oslo", "Stockholm", "Vienna", "Prague", "Warsaw", "Lisbon", "Athens", "Istanbul",
    "Dubai", "Singapore", "Taipei", "Daegu", "Incheon", "Kyoto", "Melbourne", "Seattle", "Miami",
    "Texas", "California", "Canada", "Japan", "Korea", "China", "France", "Spain", "Italy", "Brazil",
];

pub const TEST_PLACES: &[&str] = &[
    "Busan", "Gwangju", "Sapporo", "Denver", "Atlanta", "Munich", "Milan", "Porto", "Helsinki",
    "Auckland", "Perth", "Quebec", "Havana", "Santiago", "Hong Kong",
];

pub const TRAIN_FOODS: &[&str] = &[
    "pizza", "sushi", "noodles", "rice", "pasta", "tacos", "burgers", "kimchi", "curry", "salad",
    "ice cream", "chocolate", "soup", "steak", "dumplings", "fried chicken",
];
pub const TEST_FOODS: &[&str] = &["bibimbap", "ramen", "pancakes", "falafel", "lasagna"];

pub const TRAIN_COLORS: &[&str] = &["red", "blue", "green", "yellow", "purple", "orange", "pink", "black"];
pub const TEST_COLORS: &[&str] = &["white", "brown", "gray", "gold"];

/// Value lists for one split.
#[derive(Clone, Copy)]
pub struct Pools<'a> {
    pub names: &'a [&'a str],
    pub places: &'a [&'a str],
    pub foods: &'a [&'a str],
    pub colors: &'a [&'a str],
}

pub const TRAIN: Pools<'static> = Pools { names: TRAIN_NAMES, places: TRAIN_PLACES, foods: TRAIN_FOODS, colors: TRAIN_COLORS };
pub const TEST: Pools<'static> = Pools { names: TEST_NAMES, places: TEST_PLACES, foods: TEST_FOODS, colors: TEST_COLORS };

/// A dialogue (alternating user, assistant turns) and the word the last assistant turn
/// must contain to count as correct.
#[derive(Clone, Debug, PartialEq)]
pub struct Synth {
    pub turns: Vec<String>,
    pub answer: String,
}

/// Longest dialogue in characters; about 180 tokens, below the 257-token record limit.
pub const MAX_CHARS: usize = 700;

fn pick<'a>(rng: &mut Rng, xs: &[&'a str]) -> &'a str {
    xs[rng.below(xs.len())]
}

fn fill(t: &str, v: &str) -> String {
    t.replace("{}", v)
}

struct FactKind {
    tell: &'static [&'static str],
    ack: &'static [&'static str],
    ask: &'static [&'static str],
    answer: &'static [&'static str],
}

const NAME: FactKind = FactKind {
    tell: &["Hi, I'm {}.", "My name is {}.", "hello! my name is {}", "I'm {}, nice to meet you.", "Call me {}.", "hey, {} here."],
    ack: &["Nice to meet you, {}!", "Hi {}! How are you?", "Hello, {}. I'm Rllm.", "Good to meet you, {}."],
    ask: &["What's my name?", "Do you remember my name?", "what is my name again?", "Who am I?", "whats my name", "Can you tell me my name?"],
    answer: &["Your name is {}.", "You're {}.", "You told me your name is {}.", "Of course, you're {}."],
};
const PLACE: FactKind = FactKind {
    tell: &["I live in {}.", "I'm from {}.", "i'm in {} right now.", "I just moved to {}."],
    ack: &["Oh, {}! How do you like it there?", "{} sounds like a nice place.", "I see, you're in {}.", "Cool, I've heard {} is nice."],
    ask: &["Where do I live?", "Do you remember where I'm from?", "where am i?", "Which place did I tell you about?"],
    answer: &["You're in {}.", "You told me {}.", "You said {}."],
};
const FOOD: FactKind = FactKind {
    tell: &["My favorite food is {}.", "I really love {}.", "i could eat {} every day"],
    ack: &["{} is delicious!", "Yum, {} sounds good.", "I see, you like {}."],
    ask: &["What's my favorite food?", "What food do I love?", "do you remember what I like to eat?"],
    answer: &["Your favorite food is {}.", "You love {}.", "You said {}."],
};
const COLOR: FactKind = FactKind {
    tell: &["I like the color {}.", "My favorite color is {}.", "{} is my favorite color."],
    ack: &["{} is a nice color.", "Oh, I like {} too.", "Got it, {}."],
    ask: &["What color do I like?", "What's my favorite color?", "do you remember my favorite color?"],
    answer: &["You like {}.", "Your favorite color is {}.", "You said {}."],
};

const IDENTITY: &[(&str, &[&str])] = &[
    ("What's your name?", &["My name is Rllm.", "I'm Rllm.", "You can call me Rllm."]),
    ("Who are you?", &["I'm Rllm, a small chat program.", "I'm Rllm."]),
    ("your name is?", &["Rllm.", "It's Rllm."]),
    ("What should I call you?", &["You can call me Rllm.", "Call me Rllm."]),
    ("what's your name again?", &["It's Rllm.", "I'm Rllm."]),
    ("Do you have a name?", &["Yes, my name is Rllm."]),
    ("tell me your name", &["My name is Rllm."]),
    ("Are you a human?", &["No, I'm Rllm, a small chat program."]),
    ("Are you a person?", &["No, I'm Rllm, a computer program."]),
    ("How old are you?", &["I'm Rllm, a program, so I don't have an age."]),
];

/// Up to `max` short (user, assistant) filler exchanges; total stays within `budget` chars.
fn fillers(rng: &mut Rng, pool: &[(String, String)], max: usize, budget: usize) -> Vec<String> {
    let mut out = Vec::new();
    if pool.is_empty() {
        return out;
    }
    let mut used = 0;
    for _ in 0..rng.below(max + 1) {
        let (u, a) = &pool[rng.below(pool.len())];
        if used + u.len() + a.len() > budget {
            break;
        }
        used += u.len() + a.len();
        out.push(u.clone());
        out.push(a.clone());
    }
    out
}

/// The user tells one or two facts, some filler exchanges follow, then the user asks for
/// one of the facts and the assistant states it.
pub fn memory_dialogue(rng: &mut Rng, pools: Pools, filler_pool: &[(String, String)]) -> Synth {
    let kinds: [(&FactKind, &[&str]); 4] =
        [(&NAME, pools.names), (&PLACE, pools.places), (&FOOD, pools.foods), (&COLOR, pools.colors)];
    let first = rng.below(4);
    let mut chosen = vec![first];
    if rng.below(2) == 0 {
        chosen.push((first + 1 + rng.below(3)) % 4);
    }
    let mut turns = Vec::new();
    let mut values = Vec::new();
    for &k in &chosen {
        let (kind, vals) = kinds[k];
        let v = pick(rng, vals);
        turns.push(fill(pick(rng, kind.tell), v));
        turns.push(fill(pick(rng, kind.ack), v));
        values.push((k, v));
    }
    let so_far: usize = turns.iter().map(String::len).sum();
    turns.extend(fillers(rng, filler_pool, 3, MAX_CHARS.saturating_sub(so_far + 120)));
    let (k, v) = values[rng.below(values.len())];
    let kind = kinds[k].0;
    turns.push(pick(rng, kind.ask).to_string());
    turns.push(fill(pick(rng, kind.answer), v));
    Synth { turns, answer: v.to_string() }
}

/// Some filler exchanges, then a question about the assistant's identity.
pub fn identity_dialogue(rng: &mut Rng, filler_pool: &[(String, String)]) -> Synth {
    let mut turns = fillers(rng, filler_pool, 2, MAX_CHARS - 120);
    let (q, answers) = IDENTITY[rng.below(IDENTITY.len())];
    turns.push(q.to_string());
    turns.push(pick(rng, answers).to_string());
    Synth { turns, answer: BOT_NAME.to_string() }
}

/// First exchanges of dialogues whose two turns are each at most 80 characters.
pub fn filler_pool(dialogues: &[Vec<String>]) -> Vec<(String, String)> {
    dialogues
        .iter()
        .filter(|d| d.len() >= 2 && d[0].len() <= 80 && d[1].len() <= 80)
        .map(|d| (d[0].clone(), d[1].clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> Vec<(String, String)> {
        vec![
            ("How's the weather?".into(), "It's sunny today.".into()),
            ("Want some coffee?".into(), "Sure, thanks.".into()),
        ]
    }

    #[test]
    fn splits_are_disjoint() {
        for (a, b) in [
            (TRAIN_NAMES, TEST_NAMES),
            (TRAIN_PLACES, TEST_PLACES),
            (TRAIN_FOODS, TEST_FOODS),
            (TRAIN_COLORS, TEST_COLORS),
        ] {
            assert!(a.iter().all(|x| !b.contains(x)), "overlap between splits");
        }
    }

    #[test]
    fn memory_answer_was_told_earlier() {
        let mut rng = Rng::new(1);
        for _ in 0..500 {
            let s = memory_dialogue(&mut rng, TRAIN, &pool());
            assert!(s.turns.len() >= 4 && s.turns.len() % 2 == 0, "{:?}", s.turns);
            let last = s.turns.last().unwrap();
            assert!(last.contains(&s.answer), "{:?}", s.turns);
            let told = s.turns[..s.turns.len() - 2].iter().step_by(2).any(|u| u.contains(&s.answer));
            assert!(told, "fact never told by the user: {:?}", s.turns);
            assert!(s.turns.iter().map(String::len).sum::<usize>() <= MAX_CHARS);
        }
    }

    #[test]
    fn test_split_uses_only_test_values() {
        let mut rng = Rng::new(2);
        let all_test: Vec<&str> =
            [TEST_NAMES, TEST_PLACES, TEST_FOODS, TEST_COLORS].iter().flat_map(|x| x.iter().copied()).collect();
        for _ in 0..200 {
            let s = memory_dialogue(&mut rng, TEST, &pool());
            assert!(all_test.contains(&s.answer.as_str()), "{}", s.answer);
        }
    }

    #[test]
    fn identity_ends_with_bot_name() {
        let mut rng = Rng::new(3);
        for _ in 0..100 {
            let s = identity_dialogue(&mut rng, &pool());
            assert!(s.turns.len() % 2 == 0 && s.turns.last().unwrap().contains(BOT_NAME));
        }
    }

    #[test]
    fn deterministic_for_seed() {
        let a = memory_dialogue(&mut Rng::new(7), TRAIN, &pool());
        let b = memory_dialogue(&mut Rng::new(7), TRAIN, &pool());
        assert_eq!(a, b);
    }

    #[test]
    fn filler_pool_keeps_short_openers() {
        let d = vec![
            vec!["Hi.".to_string(), "Hello.".to_string(), "x".to_string()],
            vec!["a".repeat(100), "b".to_string()],
            vec!["only one".to_string()],
        ];
        assert_eq!(filler_pool(&d), vec![("Hi.".to_string(), "Hello.".to_string())]);
    }
}
