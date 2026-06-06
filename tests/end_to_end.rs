//! End-to-end generation tests against the real Q4_K_M weights.
//!
//! These load the full ~5.2 GB GGUF and run real generation, so they are gated behind
//! `#[ignore]`: the default `cargo test` stays fast and needs no model file. Run them
//! explicitly once the weights are present:
//!
//! ```sh
//! cargo test --release -- --ignored
//! ```
//!
//! The weights path comes from `$BEBELM_WEIGHTS_FILE`, defaulting to the GGUF in the repo
//! root (same resolution as the CLI). Assertions check robust signals (a known substring, the
//! presence of source data, structural characters) rather than exact token ids, which can
//! drift across builds and architectures.

use bebelm::agent::Agent;
use bebelm::model::Model;

/// Default weights path when `$BEBELM_WEIGHTS_FILE` is unset (relative to the cwd).
const DEFAULT_WEIGHTS_FILE: &str = "./LFM2.5-8B-A1B-Q4_K_M.gguf";

/// Load the weights from `$BEBELM_WEIGHTS_FILE` (or the default path), panicking with context.
fn load_model() -> Model {
    let path = std::env::var("BEBELM_WEIGHTS_FILE").unwrap_or_else(|_| DEFAULT_WEIGHTS_FILE.to_string());
    Model::load(&path).unwrap_or_else(|e| panic!("failed to load weights from {path:?}: {e}"))
}

/// Greedy completion of a factual prompt should name Paris. This exercises the whole stack:
/// GGUF load, tokenizer, prefill, cached decode, and detokenization.
#[test]
#[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
fn capital_of_france_is_paris() {
    let model = load_model();

    let mut agent = Agent::new(&model).expect("build agent").greedy().max_gen(8);
    agent.append("The capital of France is");
    let turn = agent.generate(|_id, _piece| {});

    assert!(
        turn.text.contains("Paris"),
        "expected the completion to mention Paris, got {:?}",
        turn.text
    );
}

/// A ChatML instruction turn that asks the model to render a small CSV as a Markdown table.
/// This exercises instruction following over inlined file content and multi-line structured
/// output. We assert robust signals: every source row's name is reproduced, and the reply
/// contains enough table pipes to be an actual Markdown table.
#[test]
#[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
fn csv_to_markdown_table() {
    // Compiled in from `tests/users.csv` so the test is independent of the working directory.
    let csv = include_str!("users.csv");
    let model = load_model();

    // `--no-think` (max_think 0): answer directly instead of opening a reasoning block, so the
    // table is the whole reply and a modest token budget suffices.
    let mut agent = Agent::new(&model).expect("build agent").greedy().max_think(0).max_gen(200);
    agent.append_user(&format!(
        "Convert the following CSV into a Markdown table. Output only the table.\n\n```csv\n{csv}```"
    ));
    let turn = agent.assistant_turn(|_id, _piece| {});
    let out = &turn.text;

    // Every row from the CSV should appear in the rendered table.
    for name in ["Alice", "Bob", "Carol", "David", "Eve", "Frank"] {
        assert!(out.contains(name), "table is missing {name:?}, got:\n{out}");
    }
    // A 3-column table over a header + 6 rows has far more than this; 12 is a safe floor that
    // still proves the output is a pipe-delimited table rather than prose.
    let pipes = out.matches('|').count();
    assert!(pipes >= 12, "expected a Markdown table (>=12 '|'), found {pipes} in:\n{out}");
}

/// Greedy decoding (temperature 0, argmax) must be reproducible: the same prompt run twice
/// against the same weights yields identical token ids. A build-/architecture-portable
/// determinism guard over the whole numeric pipeline, with no hardcoded ids.
#[test]
#[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
fn greedy_decoding_is_deterministic() {
    let model = load_model();
    let run = || {
        let mut agent = Agent::new(&model).expect("build agent").greedy().max_think(0).max_gen(32);
        agent.append_user("List three primary colors, one per line.");
        agent.assistant_turn(|_id, _piece| {}).ids
    };

    let first = run();
    let second = run();
    assert!(!first.is_empty(), "expected the model to generate at least one token");
    assert_eq!(first, second, "greedy decoding should be bit-identical run-to-run");
}

/// A two-turn conversation: the model is told a name, then asked to recall it in a later turn.
/// This exercises cache persistence across turns — the second turn prefills only its own new
/// tokens on top of the retained KV/conv caches rather than reprocessing the whole transcript.
#[test]
#[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
fn multi_turn_remembers_context() {
    let model = load_model();
    let mut agent = Agent::new(&model).expect("build agent").greedy().max_think(0).max_gen(64);

    agent.append_user("My name is Quentin. Please remember it.");
    agent.assistant_turn(|_id, _piece| {});

    agent.append_user("What is my name? Answer with just the name.");
    let turn = agent.assistant_turn(|_id, _piece| {});

    assert!(
        turn.text.contains("Quentin"),
        "the model should recall the name from the earlier turn, got:\n{}",
        turn.text
    );
}
