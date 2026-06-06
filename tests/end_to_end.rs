//! End-to-end generation test against the real Q4_K_M weights.
//!
//! This loads the full ~5.2 GB GGUF and runs a short greedy completion, so it is gated
//! behind `#[ignore]`: the default `cargo test` stays fast and needs no model file. Run it
//! explicitly once the weights are present:
//!
//! ```sh
//! cargo test --release -- --ignored
//! ```
//!
//! The weights path comes from `$BEBELM_WEIGHTS_FILE`, defaulting to the GGUF in the repo
//! root (same resolution as the CLI).

use bebelm::agent::Agent;
use bebelm::model::Model;

/// Default weights path when `$BEBELM_WEIGHTS_FILE` is unset (relative to the cwd).
const DEFAULT_WEIGHTS_FILE: &str = "./LFM2.5-8B-A1B-Q4_K_M.gguf";

fn weights_path() -> String {
    std::env::var("BEBELM_WEIGHTS_FILE").unwrap_or_else(|_| DEFAULT_WEIGHTS_FILE.to_string())
}

/// Greedy completion of a factual prompt should name Paris. This exercises the whole stack:
/// GGUF load, tokenizer, prefill, cached decode, and detokenization.
#[test]
#[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
fn capital_of_france_is_paris() {
    let path = weights_path();
    let model = Model::load(&path)
        .unwrap_or_else(|e| panic!("failed to load weights from {path:?}: {e}"));

    let mut agent = Agent::new(&model).expect("build agent").greedy().max_gen(8);
    agent.append("The capital of France is");
    let turn = agent.generate(|_id, _piece| {});

    assert!(
        turn.text.contains("Paris"),
        "expected the completion to mention Paris, got {:?}",
        turn.text
    );
}
