# BebeLM

Pure-Rust, CPU-only implementation of [LFM2.5-8B-A1B Q4_K_M](https://www.liquid.ai/blog/lfm2-5-8b-a1b).
This model is very capable and has only 1B active parameters, making it possible for the
model to run at interactive speeds without a GPU.

This package intentionally has very few dependencies and requires no extra system
packages to run, making it easy to build and run.
This is a library crate which can be imported into your Rust projects, and it's now available
via [crates.io](https://crates.io/crates/bebelm). There is also a basic command-line
interface that you can use.

BebeLM was tested on an M5 CPU as well as Ryzen 7x and Threadripper CPUs. It should work
on Intel and on Raspberry Pi 4/5 as well, but this is untested.

## Setup instructions

Install cargo or update your rust toolchain:
```sh
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update Rust toolchain
rustup update
```

Running also requires downloading the ~5.2 GB Q4_K_M model weights:

```sh
curl -L -o LFM2.5-8B-A1B-Q4_K_M.gguf \
  "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/main/LFM2.5-8B-A1B-Q4_K_M.gguf"
```

The CLI reads the weights path from `BEBELM_WEIGHTS_FILE`, defaulting to
`./LFM2.5-8B-A1B-Q4_K_M.gguf` (the current directory). Point it elsewhere with:

```sh
export BEBELM_WEIGHTS_FILE=/path/to/LFM2.5-8B-A1B-Q4_K_M.gguf
```

### Installing via cargo

Install the CLI from crates.io — this puts a `bebelm` binary on your `PATH`:

```sh
cargo install bebelm
```

### Development setup

Clone the repo and build from source:

```sh
git clone https://github.com/maximecb/bebelm
cd bebelm
cargo build --release
```

## Command-line interface

Build with `cargo build --release`, then run a subcommand on `./target/release/bebelm` (the
examples below use `cargo run --release --` for convenience). Every subcommand loads the
weights from `BEBELM_WEIGHTS_FILE` (see above).

- **`generate [options] <prompt>…`** — one-shot text completion of a prompt; streams tokens as
  they are produced and reports prefill/decode throughput.
- **`chat [options]`** — interactive multi-turn chat. Streams the model's full output, showing
  the `<think>...</think>` reasoning and the final answer in different colors. The KV / conv
  caches persist across turns, so each message only prefills its own new tokens. `Ctrl-D` or
  `/exit` to quit.

Both commands take the same options (sampling defaults to the model's recommended settings):

- `--greedy` — deterministic greedy decoding instead of sampling.
- `--max-gen N` — cap tokens generated per turn (default 2048).
- `--max-think N` — cap the `<think>` reasoning block to N tokens (forces `</think>`).
- `--no-think` — disable reasoning (equivalent to `--max-think 0`).
- `--num-threads N` — cap the rayon worker pool (default: one per available core).

```sh
# Interactive chat
cargo run --release -- chat

# One-shot completion
cargo run --release -- generate --max-gen 64 "The capital of France is"
```

## Public crate API

`bebelm` is a library first; the CLI is a thin wrapper over it. The high-level entry point is
`bebelm::agent::Agent` — a conversation bound to a loaded model that owns the token transcript
and the decode-time caches.

Load the model once, then back one or more agents with it:

```rust
use bebelm::agent::Agent;
use bebelm::model::Model;

// mmaps + validates the GGUF.
let model = Model::load("LFM2.5-8B-A1B-Q4_K_M.gguf")?;

// An agent borrows the model — the ~5.2 GB of weights are shared, so several agents are cheap.
let mut agent = Agent::new(&model)?;

agent.append_user("What is the capital of France?");
let turn = agent.assistant_turn(|_, _| {});   // generate the whole reply at once
println!("{}", turn.text);

// Keep chatting — the KV/conv caches persist, so only the new tokens are prefilled.
agent.append_user("And of Italy?");
let turn = agent.assistant_turn(|_, _| {});
println!("{}", turn.text);
```

Here `|_, _| {}` is a do-nothing token callback, so the whole reply is just collected into
`turn.text`. To instead stream tokens as they are generated, pass a real callback — see
**Generating** below.

**Configuration** — builder methods chained after `Agent::new(..)?` (sampling defaults to the
model's recommended temperature 0.2 / top-k 80 / repeat-penalty 1.05):

- `.greedy()` — deterministic argmax decoding.
- `.temperature(f32)` / `.top_k(usize)` / `.repeat_penalty(f32)` — individual sampler knobs.
- `.max_gen(usize)` — tokens generated per turn (default 2048).
- `.max_context(usize)` — KV attention-window cap in tokens (default 32768); older context
  slides out rather than stopping generation.
- `.max_think(usize)` — cap the `<think>` reasoning block (`0` ⇒ no reasoning block at all).

**Building the prompt** — these only grow the transcript; nothing runs until you generate:

- `append_user(&str)` — wrap a ChatML user turn (`<|im_start|>user\n…<|im_end|>\n`).
- `append(&str)` — append raw text (BOS is added automatically on the first append).
- `append_tokens(&[u32])` — append already-tokenized ids (e.g. a tool result).

**Generating** — `assistant_turn` and `generate` both return a `Turn` and take an `on_token`
callback:

- `assistant_turn(on_token)` — open an assistant turn (ChatML), stream the reply, and close the
  turn; pair it with `append_user` (as above).
- `generate(on_token)` — the lower-level primitive: prefill pending tokens, then decode a raw
  continuation (no ChatML framing) until EOS or `max_gen`; pair it with `append` for plain text
  completion:

```rust
let mut agent = Agent::new(&model)?;
agent.append("The capital of France is");
let turn = agent.generate(|_, _| {});      // raw continuation; turn.text = " the city of Paris…"
println!("The capital of France is{}", turn.text);
```

The returned `Turn`:

```rust
pub struct Turn {
    pub ids: Vec<u32>,    // generated ids (excludes the prompt and the terminating EOS)
    pub text: String,     // the decoded reply
    pub stats: GenStats,  // prompt_tokens, generated_tokens, prefill/decode Durations + *_tps()
    pub stop: StopReason, // Eos or MaxNew
}
```

The `on_token` callback is `impl FnMut(u32, &str)`, called once per visible token as it is
decoded — its arguments are `(id, text)`:

- `id: u32` — the token id; compare it against the `bebelm::tokenizer` constants below for
  control-token logic (e.g. spotting `<think>` / `</think>` to colour the reasoning).
- `text: &str` — that same token decoded to a string, ready to print.

The terminating EOS is not passed to the callback, and the full reply is in `turn.text` either
way. To stream tokens as they are produced:

```rust
use bebelm::tokenizer;

agent.append_user("Explain RoPE briefly.");
agent.assistant_turn(|id, text| {
    if id == tokenizer::TOKEN_THINK_END {
        println!();  // the <think> reasoning block just ended
    }
    print!("{text}");
});
```

`agent.clear()` resets the conversation (keeping the weights); `agent.history()` returns the
full token transcript.

**Special tokens** live in `bebelm::tokenizer` as `u32` constants. The agent handles BOS, EOS,
and the ChatML / `<think>` framing for you — these are mostly for interpreting the `id` your
`on_token` callback receives:

- `TOKEN_BOS` — `<|startoftext|>`, start-of-sequence (auto-prepended on the first `append`).
- `TOKEN_IM_START` / `TOKEN_IM_END` — `<|im_start|>` / `<|im_end|>`, ChatML turn delimiters.
- `TOKEN_EOS` — alias of `TOKEN_IM_END`; ends a turn.
- `TOKEN_THINK` / `TOKEN_THINK_END` — `<think>` / `</think>`, reasoning-block delimiters.
- `TOKEN_ENDOFTEXT` / `TOKEN_PAD` — `<|endoftext|>` / `<|pad|>`, document/pad markers.
- `TOKEN_TOOL_LIST_START` / `TOKEN_TOOL_LIST_END` / `TOKEN_TOOL_CALL_START` / `TOKEN_TOOL_CALL_END`
  — `<|tool_*|>` delimiters.
- `TOKEN_FIM_PRE` / `TOKEN_FIM_MID` / `TOKEN_FIM_SUF` — `<|fim_*|>` fill-in-the-middle markers.

For lower-level use, `Model::forward_step(token, &mut Cache)` runs the cached forward pass
directly, and `bebelm::tokenizer::Tokenizer` (`encode` / `decode`) and `bebelm::sampler::Sampler`
are public if you want to drive decoding yourself.

## CPU / SIMD build

The x86 SIMD kernels are tuned for the machine you build on: `.cargo/config.toml` sets
`target-cpu=native`, so a build automatically uses **AVX2 + FMA** when the CPU has them
and falls back to whatever it supports otherwise.

Because `native` targets the build host, a binary built on an AVX2 machine may fault on an
older CPU. To build a portable binary, override the CPU target via `RUSTFLAGS` (it takes
precedence over `.cargo/config.toml`):

```sh
# AVX2 baseline — runs on any Haswell (2013) or newer x86:
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release

# Universal baseline — runs on any x86_64 (SSE2 only, slowest):
RUSTFLAGS="-C target-cpu=x86-64" cargo build --release
```

The instruction set is chosen at build time; there is no single binary that switches at
runtime.

## Running the tests

The test suite has two layers:

- **Fast unit tests** run with plain `cargo test` — they need no model file and finish in
  seconds, so they are the default and what CI runs first.
- **End-to-end tests** (`tests/end_to_end.rs`) load the full ~5.2 GB Q4_K_M GGUF and run real
  generation against it. They are gated behind `#[ignore]` so `cargo test` stays model-free, and
  they read the weights path from `BEBELM_WEIGHTS_FILE` (defaulting to the repo-root GGUF, same
  resolution as the CLI — see **Setup instructions** for downloading it).

Run the **full** end-to-end suite — every `#[ignore]`d test — with `--ignored`:

```sh
cargo test --release -- --ignored
```

Each test loads the model independently and runs real decoding, so the full suite is slow. For
a quick **partial** run, append a test-name filter (a substring match) — e.g. the single
Paris-completion smoke test, the fastest one:

```sh
# one end-to-end test (fast smoke check)
cargo test --release -- --ignored capital_of_france_is_paris
```

A broader substring targets a group, e.g. `cargo test --release -- --ignored multi_turn`. List
the available end-to-end tests without running them with
`cargo test --release -- --ignored --list`. Always use `--release`: a debug build runs the
numeric kernels far slower.
