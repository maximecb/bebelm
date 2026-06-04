#!/usr/bin/env bash
#
# CPU profiler for bebelm's decode loop, using macOS's built-in `sample(1)`.
#
# `sample` is a statistical sampling profiler shipped with macOS: it pauses the target
# every INTERVAL ms and records each thread's call stack, then aggregates. No install, no
# sudo, no SIP changes (it only attaches to your own processes), and it sees all threads —
# so rayon's per-row matmul work shows up too. Instruments/xctrace need full Xcode (we only
# have Command Line Tools), and dtrace/cargo-flamegraph are SIP-restricted, so `sample` is
# the path of least resistance here.
#
# Usage: ./profile.sh [path/to/model.gguf]
#
# Env knobs:
#   DURATION  seconds to sample            (default 20)
#   WARMUP    seconds to skip before sampling — model load + prefill + page-in (default 5)
#   MAX_NEW   tokens to generate           (default 2000; just needs to outlast the window)
#   INTERVAL  sampling interval in ms      (default 1)
#
# This profiles steady-state *decode* (matvec, memory-bound), which is where the benchmark
# spends ~all its time (one short prefill vs many decode steps). Prefill (batched, compute-
# bound matmul) would need a long prompt and a separate run.

set -euo pipefail

MODEL="${1:-models/LFM2.5-8B-A1B-Q4_K_M.gguf}"
PROMPT="Tell me about the capital of France"

DURATION="${DURATION:-20}"
WARMUP="${WARMUP:-5}"
MAX_NEW="${MAX_NEW:-2000}"
INTERVAL="${INTERVAL:-1}"

RAW=profile.txt
DEMANGLED=profile.demangled.txt

command -v sample >/dev/null || { echo "error: sample(1) not found (macOS only)" >&2; exit 1; }

if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    echo "       pass the path as an argument, or download it (see design.md)." >&2
    exit 1
fi

echo "building (release)..."
cargo build --release --quiet
BIN=./target/release/bebelm

# Run the workload in the background, with its token stream tucked into a log so it doesn't
# interleave with the profiler's output. MAX_NEW is kept large so the process stays alive
# for the whole window; we kill it afterward in the trap.
LOG="$(mktemp)"
"$BIN" complete "$MODEL" "$MAX_NEW" "$PROMPT" >"$LOG" 2>&1 &
PID=$!

cleanup() {
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
    rm -f "$LOG"
}
trap cleanup EXIT

# Skip model load + prefill + the initial lazy page-faulting of weights, so we sample
# steady-state decode. The GGUF is mmap'd, so on a cold page cache the first run also pays
# disk I/O to fault ~5 GB of weights in (and MoE touches a different expert subset each
# token) — run profile.sh twice, or raise WARMUP, for a pure-compute picture.
echo "warming up ${WARMUP}s (model load + prefill)..."
sleep "$WARMUP"

if ! kill -0 "$PID" 2>/dev/null; then
    echo "error: workload exited during warmup — output below:" >&2
    cat "$LOG" >&2
    exit 1
fi

echo "sampling pid $PID for ${DURATION}s every ${INTERVAL}ms..."
# -mayDie: read symbol info up front, so we still symbolicate if the process exits mid-run.
sample "$PID" "$DURATION" "$INTERVAL" -mayDie -file "$RAW"

# `sample` prints Rust's _ZN..-mangled (Itanium ABI) symbols; c++filt makes them readable.
c++filt < "$RAW" > "$DEMANGLED"

echo
echo "=== hottest leaf functions (sample's 'Sort by top of stack') ==="
# The leaf-attribution table sample appends near the end: where samples actually landed,
# i.e. self time. Read the 'Call graph:' section of the report for inclusive (caller) costs.
awk '/^Sort by top of stack/{p=1} /^Binary Images:/{p=0} p' "$DEMANGLED" | head -40

echo
echo "wrote $DEMANGLED (readable) and $RAW (raw)."
echo "tip: read the 'Call graph:' section of $DEMANGLED top-down for inclusive time per call path."
