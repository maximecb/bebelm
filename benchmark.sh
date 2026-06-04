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
# binary/machine (pure f32, single-thread scalar, no RNG). `rayon` row-parallelism (opt
# 9c) preserves this, but SIMD (9d) / FMA contraction can shift the exact tokens — if the
# matmul math changes, update EXPECTED below.

set -euo pipefail

MODEL="${1:-models/LFM2.5-8B-A1B-Q4_K_M.gguf}"
PROMPT="The capital of France is"
MAX_NEW=8
EXPECTED=" the city of Paris. city of Paris"   # greedy, 8 tokens (keep in sync with PROMPT/MAX_NEW)

if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    echo "       pass the path as an argument, or download it (see design.md)." >&2
    exit 1
fi

echo "building (release)..."
cargo build --release --quiet

echo "running: complete $MAX_NEW \"$PROMPT\""
OUT="$(./target/release/bebelm complete "$MODEL" "$MAX_NEW" "$PROMPT")"
echo "$OUT"
echo

CONT="$(printf '%s\n' "$OUT" | sed -n 's/^continuation : "\(.*\)"$/\1/p')"
DECODE_TPS="$(printf '%s\n' "$OUT" | sed -n 's/^decode .*(\(.*\) tok\/s)$/\1/p')"

echo "decode throughput: ${DECODE_TPS:-?} tok/s"

if [ "$CONT" = "$EXPECTED" ]; then
    echo "PASS: output matches expected exactly"
    exit 0
elif printf '%s' "$CONT" | grep -q "Paris"; then
    echo "WARN: exact mismatch, but output still mentions Paris (FP/impl drift?)"
    echo "  expected: \"$EXPECTED\""
    echo "  actual  : \"$CONT\""
    exit 1
else
    echo "FAIL: unexpected output"
    echo "  expected: \"$EXPECTED\""
    echo "  actual  : \"$CONT\""
    exit 1
fi
