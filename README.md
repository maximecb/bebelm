BebeLM
------

Pure-Rust, CPU-only implementation of LFM2.5-8B-A1B.
Intentionally has very few dependencies and requires no extra system packages to run.

This is a library crate so the model can be imported.

### Download the weights

Running requires the ~5.2 GB Q4_K_M GGUF. Download it into the repo root:

```sh
curl -L -o LFM2.5-8B-A1B-Q4_K_M.gguf \
  "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/main/LFM2.5-8B-A1B-Q4_K_M.gguf"
```

`benchmark.sh` and `profile.sh` default to this filename in the repo root; you can also pass a
different path as their first argument.

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
