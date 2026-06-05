BebeLM
------

Pure-Rust, CPU-only implementation of [LFM2.5-8B-A1B](https://www.liquid.ai/blog/lfm2-5-8b-a1b).
This model is very capable and has only 1B active parameters, making it possible for the
model to run at interactive speeds without a GPU.

This package intentionally has very few dependencies and requires no extra system
packages to run, making it easy to build and run.
This is a library crate so the model can be imported. There is also a basic command-line
interface that you can use.

### Setup instructions

Install cargo or update your rust toolchain:
```sh
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update Rust toolchain
rustup update
```

Running requires the ~5.2 GB Q4_K_M GGUF. Download it into the repo root:

```sh
curl -L -o LFM2.5-8B-A1B-Q4_K_M.gguf \
  "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/main/LFM2.5-8B-A1B-Q4_K_M.gguf"
```

The CLI reads the weights path from the `BEBELM_WEIGHTS_FILE` environment variable. This defaults
to `./LFM2.5-8B-A1B-Q4_K_M.gguf` (repo root). You can optionally point it elsewhere with:

```sh
export BEBELM_WEIGHTS_FILE=/path/to/LFM2.5-8B-A1B-Q4_K_M.gguf
```

### Command-line interface

Build with `cargo build --release`, then run a subcommand on `./target/release/bebelm` (the
examples below use `cargo run --release --` for convenience). Every subcommand loads the
weights from `BEBELM_WEIGHTS_FILE` (see above).

- **`chat [max-new]`** — interactive multi-turn chat. Streams the model's full output, showing
  the `<think>...</think>` reasoning and the final answer in different colors. The KV / conv
  caches persist across turns, so each message only prefills its own new tokens. Sampling uses
  the model's recommended defaults. `max-new` caps the tokens generated per turn (default 2048).
  `Ctrl-D` or `/exit` to quit.
- **`complete <max-gen> <text>…`** — greedy text completion of a prompt; streams tokens as they
  are produced and reports prefill/decode throughput.
- **`tokenize <text>…`** — encode text to token ids and decode it back (a vocab round-trip check).
- **`generate <max-gen> <token-id>…`** — greedy-generate from raw prompt token ids.
- **`logits <token-id>…`** — run one forward pass on raw token ids and print a summary of the
  next-token logits (argmax + top-5).

```sh
# Interactive chat
cargo run --release -- chat

# One-shot completion
cargo run --release -- complete 64 "The capital of France is"
```

### CPU / SIMD build

The x86 SIMD kernels are tuned for the machine you build on: `.cargo/config.toml` sets
`target-cpu=native`, so a build automatically uses **AVX2 + FMA** when the CPU has them
and falls back to whatever it supports otherwise. (Without this the default
x86_64 target is SSE2-only and runs the vector dot products at half width.) arm64 (Apple
Silicon / NEON) is unaffected and needs no flags.

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
