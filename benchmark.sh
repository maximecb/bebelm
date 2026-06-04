#!/usr/bin/env bash
#
# Simple generation benchmark + correctness smoke test for bebelm.
#
# Greedily generates a fixed number of tokens from a fixed prompt, prints the prefill /
# decode throughput (tokens/sec), and checks the continuation against an expected string.
#
# Usage: ./benchmark.sh [path/to/model.gguf]
#
# Determinism: greedy decoding (temperature 0) is bit-identical run-to-run on the same
# binary/machine (pure f32, no RNG). Parallelizing matvec over output rows preserves this
# (each row's accumulation order is unchanged); a later move to SIMD / FMA contraction
# could shift the exact tokens — if the matmul math changes, update EXPECTED below.

set -euo pipefail

MODEL="${1:-models/LFM2.5-8B-A1B-Q4_K_M.gguf}"
PROMPT="Tell me about the capital of France"
MAX_NEW=64
# Expected greedy token ids (deterministic on a given build: f32 + fixed-order SIMD/FMA
# reductions, no RNG). Keep in sync with PROMPT/MAX_NEW. The first ~20 decode to ", Paris.
# Provide a detailed description, description of its architecture, and a list of notable
# landmarks." then the small model repeats; this is a determinism/regression gate, not a
# quality bar (the Paris fallback below is the softer correctness signal).
EXPECTED_IDS="[20, 4741, 22, 43972, 267, 9688, 8818, 20, 8818, 302, 851, 10957, 20, 309, 267, 1815, 302, 16389, 59796, 22, 64019, 124901, 207, 597, 4695, 10966, 267, 9688, 8818, 302, 278, 5205, 302, 3980, 20, 4741, 20, 1951, 10957, 309, 267, 1815, 302, 16389, 59796, 22, 1978, 589, 794, 117377, 3639, 22, 1672, 522, 510, 33255, 22, 43972, 267, 14286, 8818, 20, 10957, 342]"

if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    echo "       pass the path as an argument, or download it (see design.md)." >&2
    exit 1
fi

echo "building (release)..."
cargo build --release --quiet

echo "running: complete $MAX_NEW \"$PROMPT\""
echo
# `tee` to a temp file so the generation streams to the terminal live (the per-token
# flushes in `complete` make it appear token-by-token) while we still capture it to parse.
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
./target/release/bebelm complete "$MODEL" "$MAX_NEW" "$PROMPT" | tee "$TMP"
OUT="$(cat "$TMP")"
echo

CONT="$(printf '%s\n' "$OUT" | sed -n 's/^continuation : //p')"
GEN_IDS="$(printf '%s\n' "$OUT" | sed -n 's/^gen ids *: //p')"
PREFILL_TPS="$(printf '%s\n' "$OUT" | sed -n 's/^prefill .*(\(.*\) tok\/s)$/\1/p')"
DECODE_TPS="$(printf '%s\n' "$OUT" | sed -n 's/^decode .*(\(.*\) tok\/s)$/\1/p')"

echo "prefill throughput: ${PREFILL_TPS:-?} tok/s"
echo "decode throughput:  ${DECODE_TPS:-?} tok/s"

if [ "$GEN_IDS" = "$EXPECTED_IDS" ]; then
    echo "PASS: generated ids match expected"
    exit 0
elif printf '%s' "$CONT" | grep -q "Paris"; then
    echo "WARN: ids mismatch, but output still mentions Paris (FP/impl drift?)"
    echo "  expected ids: $EXPECTED_IDS"
    echo "  actual ids  : $GEN_IDS"
    exit 1
else
    echo "FAIL: unexpected output"
    echo "  expected ids: $EXPECTED_IDS"
    echo "  actual ids  : $GEN_IDS"
    exit 1
fi
