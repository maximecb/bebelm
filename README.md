BebeLM
------

Pure-Rust, CPU-only implementation of LFM2.5-8B-A1B.
Intentionally has very few dependencies and requires no extra system packages to run.

This is a library crate so the model can be imported.

In order to run, the weights GGUF file needs to be downloaded first (5.2GB).

### CPU requirements

The SIMD kernels are built for AVX2 + FMA on x86 (the `x86-64-v3` level, i.e. Intel Haswell /
2013 and newer) via `.cargo/config.toml` — without it the default x86_64 target is SSE2-only
and runs the vector dot products at half width. A pre-AVX2 x86 CPU will fault; either build
for the baseline (`target-cpu=x86-64`) or, to tune for the exact machine you build on, use
`target-cpu=native`. arm64 (Apple Silicon / NEON) is unaffected and needs no flags.

