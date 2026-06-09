//! End-to-end generation tests against the real Q4_K_M weights.
//!
//! These load the full ~5.2 GB GGUF and run real generation, so they are gated behind
//! `#[ignore]`: the default `cargo test` stays fast and needs no model file. Run them
//! explicitly once the weights are present:
//!
//! ```sh
//! cargo test --release -- --ignored --test-threads=1
//! ```
//!
//! The weights path comes from `$BEBELM_WEIGHTS_FILE`, defaulting to the GGUF in the repo
//! root (same resolution as the CLI). Assertions check robust signals (a known substring, the
//! presence of source data, structural characters) rather than exact token ids, which can
//! drift across builds and architectures.

use bebelm::agent::Agent;
use bebelm::model::Model;
use bebelm::tool::{Schema, Tool, Type};

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

    let mut agent = Agent::new(&model).greedy().max_gen(8);
    agent.append("The capital of France is");
    let turn = agent.generate(|_id, _piece| {});

    assert!(
        turn.text.contains("Paris"),
        "expected the completion to mention Paris, got {:?}",
        turn.text
    );
}

/// Greedy instruction following to correct a misspelled sentence. This exercises
/// the model's ability to handle noisy input and perform basic text editing.
#[test]
#[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
fn spellchecking_correction() {
    let model = load_model();

    let mut agent = Agent::new(&model).greedy().max_think(200).max_gen(64);
    agent.append_user("Please correct the spelling in this sentence: 'I am goign to the park tomorow.' Output only the corrected sentence.");
    let turn = agent.assistant_turn(|_id, _piece| {});

    assert!(
        turn.text.contains("going") && turn.text.contains("tomorrow"),
        "expected the corrected sentence to contain 'going' and 'tomorrow', got {:?}",
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
    let mut agent = Agent::new(&model).greedy().max_think(200).max_gen(200);
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

/// A ChatML instruction turn that asks the model to convert a small Markdown table into an HTML
/// table. Companion to `csv_to_markdown_table`: it exercises the same instruction-following over
/// inlined structured input, but the target format is markup rather than more Markdown. We assert
/// robust signals: every source cell is reproduced, and the reply contains the structural HTML
/// table tags rather than prose or a passed-through Markdown table.
#[test]
#[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
fn markdown_table_to_html() {
    let model = load_model();

    // A minimal 2-column Markdown table: a header row, the separator, and two data rows.
    let markdown = "\
| Name | Role |
| ---- | ---- |
| Alice | Engineer |
| Bob | Designer |";

    // `--no-think` (max_think 0): answer directly instead of opening a reasoning block, so the
    // table is the whole reply and a modest token budget suffices.
    let mut agent = Agent::new(&model).greedy().max_think(200).max_gen(200);
    agent.append_user(&format!(
        "Convert the following Markdown table into an HTML table. Output only the HTML.\n\n{markdown}"
    ));
    let turn = agent.assistant_turn(|_id, _piece| {});
    let out = &turn.text;

    // Every cell from the source table should survive the conversion.
    for cell in ["Name", "Role", "Alice", "Engineer", "Bob", "Designer"] {
        assert!(out.contains(cell), "HTML table is missing {cell:?}, got:\n{out}");
    }
    // Structural tags that prove the output is an actual HTML table and not prose or a
    // passed-through Markdown table. `<td` or `<th` (cell open) must appear for the data cells.
    assert!(out.contains("<table"), "expected a <table> element, got:\n{out}");
    assert!(out.contains("<tr"), "expected table rows (<tr>), got:\n{out}");
    assert!(
        out.contains("<td") || out.contains("<th"),
        "expected table cells (<td>/<th>), got:\n{out}"
    );
}

/// Greedy decoding (temperature 0, argmax) must be reproducible: the same prompt run twice
/// against the same weights yields identical token ids. A build-/architecture-portable
/// determinism guard over the whole numeric pipeline, with no hardcoded ids.
#[test]
#[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
fn greedy_decoding_is_deterministic() {
    let model = load_model();
    let run = || {
        let mut agent = Agent::new(&model).greedy().max_think(200).max_gen(32);
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
    let mut agent = Agent::new(&model).greedy().max_think(200).max_gen(64);

    agent.append_user("My name is Quentin. Please remember it.");
    agent.assistant_turn(|_id, _piece| {});

    agent.append_user("What is my name? Answer with just the name I just stated and no other output.");
    let turn = agent.assistant_turn(|_id, _piece| {});

    assert!(
        turn.text.contains("Quentin"),
        "the model should recall the name from the earlier turn, got:\n{}",
        turn.text
    );
}

/// Cloning forks a prefilled conversation: each clone continues independently from the shared
/// prefix without re-running its prefill, and generating on one clone leaves the other's
/// transcript untouched.
#[test]
#[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
fn clone_forks_independent_continuations() {
    let model = load_model();
    let mut base = Agent::new(&model).greedy().max_think(200).max_gen(64);

    base.append_user("My name is Quentin. Please remember it.");
    base.assistant_turn(|_id, _piece| {});
    let base_len_before = base.history().len();

    let mut fork = base.clone();
    fork.append_user("What is my name? Answer with just the name I just stated and no other output.");
    let turn = fork.assistant_turn(|_id, _piece| {});

    assert!(
        turn.text.contains("Quentin"),
        "the fork should recall the name from the shared prefix, got:\n{}",
        turn.text
    );
    assert_eq!(
        base.history().len(),
        base_len_before,
        "cloning should fork the transcript — generating on the clone must not mutate the original"
    );
}

/// Function calling, end to end: register an `add` tool, ask a question that needs it, and run
/// the agentic loop. This exercises the whole tool path — schema emission into the system block,
/// the model emitting a call between `<|tool_call_start|>`/`<|tool_call_end|>`, parsing and
/// dispatching it, feeding the result back as a `tool`-role message, and the model using it in a
/// final answer. We assert robust signals: the tool was invoked returning 42, and the final
/// reply states 42.
#[test]
#[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
fn tool_call_add_round_trip() {
    let model = load_model();
    let mut agent = Agent::new(&model).greedy().max_think(200).max_gen(256).add_tool(Tool::new(
        "add",
        "Add two integers and return their sum.",
        Schema::new().req("a", Type::Int, "First addend").req("b", Type::Int, "Second addend"),
        |c| {
            // Args arrive as raw text; `parse_arg` parses one into the receiver's type.
            let a: i64 = c.parse_arg("a").unwrap_or(0);
            let b: i64 = c.parse_arg("b").unwrap_or(0);
            (a + b).to_string()
        },
    ));
    agent.append_system("You are a helpful assistant. Use the provided tools when they apply.");
    agent.append_user("What is 21 + 21? Use the add tool, then state the result.");

    // Record each dispatched call so we can assert the tool actually ran (not just that the
    // final text happens to contain 42).
    let mut calls: Vec<(String, String)> = Vec::new();
    let turn = agent.assistant_turn_with_tools(
        4,
        |_id, _text| {},
        |call, result| calls.push((call.name.clone(), result.to_string())),
    );

    assert!(
        calls.iter().any(|(name, result)| name == "add" && result == "42"),
        "expected the model to call add(...) returning 42; calls: {calls:?}\nreply:\n{}",
        turn.text
    );
    assert!(
        turn.text.contains("42"),
        "expected the final answer to state 42, got:\n{}",
        turn.text
    );
}

/// Multiple tool calls, end to end: register a `get_age` tool, ask a question that needs it
/// for several people, and run the agentic loop. We assert that the model calls the tool
/// multiple times and produces a Markdown table with the results.
#[test]
#[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
fn tool_call_multi_get_age_table() {
    let model = load_model();
    let mut agent = Agent::new(&model).greedy().max_think(512).max_gen(1024).add_tool(Tool::new(
        "get_age",
        "Retrieve the age of a person given their name.",
        Schema::new().req("name", Type::Str, "The name of the person to look up"),
        |c| {
            let name = c.arg("name").unwrap_or("");
            match name {
                "Alice" => "25".to_string(),
                "Bob" => "30".to_string(),
                "Charlie" => "35".to_string(),
                _ => "unknown".to_string(),
            }
        },
    ));
    agent.append_system("You are a helpful assistant with access to tools. Use the get_age tool to find ages for multiple people when requested.");
    agent.append_user("I need the ages of Alice, Bob, and Charlie. Use the get_age tool for each, then present the results in a Markdown table.");

    let mut calls: Vec<(String, String)> = Vec::new();

    let turn = agent.assistant_turn_with_tools(
        4,
        |_id, _text| {},
        |call, result| calls.push((call.name.clone(), result.to_string())),
    );

    dbg!(&turn.text);

    // We expect 3 calls to get_age.
    let get_age_calls: Vec<_> = calls.iter().filter(|(name, _)| name == "get_age").collect();
    assert_eq!(get_age_calls.len(), 3, "expected 3 calls to get_age, got: {calls:?}");

    // Check that results are in the table.
    let out = &turn.text;
    for (name, age) in [("Alice", "25"), ("Bob", "30"), ("Charlie", "35")] {
        assert!(out.contains(name), "table is missing name {name:?}, got:\n{out}");
        assert!(out.contains(age), "table is missing age {age:?} for {name:?}, got:\n{out}");
    }

    // It should be a table.
    let pipes = out.matches('|').count();
    assert!(pipes >= 6, "expected a Markdown table, found {pipes} pipes in:\n{out}");
}
