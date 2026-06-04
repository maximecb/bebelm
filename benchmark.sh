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
# Expected greedy token ids (deterministic on single-core scalar f32). Robust to text
# formatting; keep in sync with PROMPT/MAX_NEW. Decodes to " the city of Paris. city of Paris".
EXPECTED_IDS="[278, 3270, 302, 4741, 22, 3270, 302, 4741]"

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
