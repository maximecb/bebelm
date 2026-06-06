#!/usr/bin/env bash
#
# Simple generation benchmark + correctness smoke test for bebelm.
#
# Greedily generates a fixed number of tokens from a fixed chat prompt, prints the prefill /
# decode throughput (tokens/sec), and checks the continuation against an expected string.
#
# Usage: ./benchmark.sh [path/to/model.gguf] [num-threads]
#
# num-threads caps the rayon worker pool (passed through as `--num-threads`); omit it to use
# all available cores. Set it to benchmark thread scaling, e.g. `./benchmark.sh model.gguf 4`.
#
# Determinism: greedy decoding (temperature 0) is bit-identical run-to-run on the same
# binary/machine (pure f32, no RNG). Parallelizing matvec over output rows preserves this
# (each row's accumulation order is unchanged, regardless of worker count, so num-threads does
# not move the tokens); a later move to SIMD / FMA contraction could shift the exact tokens —
# if the matmul math changes, update EXPECTED below.

set -euo pipefail

MODEL="${1:-LFM2.5-8B-A1B-Q4_K_M.gguf}"
export BEBELM_WEIGHTS_FILE="$MODEL"
MAX_NEW=64

# Optional rayon worker count (positional arg 2). Empty means "let bebelm use all cores".
# Built as an array so the flag is omitted entirely when unset; the `[@]+` guard keeps the
# empty-array expansion safe under `set -u` on older bash (e.g. macOS's 3.2).
NUM_THREADS="${2:-}"
THREAD_OPT=()
if [ -n "$NUM_THREADS" ]; then
    THREAD_OPT=(--num-threads "$NUM_THREADS")
fi

# A single user turn in the model's ChatML chat format. `generate` prepends BOS
# (<|startoftext|>) and stops at <|im_end|>, so we open at <|im_start|>user and end with the
# assistant-turn opener; the tokenizer encodes <|im_start|>/<|im_end|> as atomic token ids.
USER_MSG="Tell me about the capital of France"
PROMPT=$'<|im_start|>user\n'"$USER_MSG"$'<|im_end|>\n<|im_start|>assistant\n'

# Expected greedy token ids (deterministic on a given build: f32 + fixed-order SIMD/FMA
# reductions, no RNG). Keep in sync with PROMPT/MAX_NEW. LFM2.5 is a reasoning model, so the
# reply opens with a <think> block — `<think>\n The user asks: "..." ... information about
# Paris ...` — a reasoning preamble, not a finished answer; this is a determinism/regression
# gate, not a quality bar (the Paris fallback below is the softer correctness signal).
EXPECTED_IDS="[124901, 207, 597, 4695, 20589, 34, 496, 51985, 622, 836, 278, 5205, 302, 3980, 2784, 3584, 589, 267, 90139, 5439, 374, 1702, 836, 4741, 22, 1978, 589, 794, 117377, 3639, 22, 43972, 267, 55911, 702, 14286, 8818, 34, 5748, 20, 2628, 20, 59796, 20, 4508, 20, 7552, 20, 2942, 20, 4222, 22, 40806, 13173, 2456, 928, 7063, 10944, 22, 440, 4695, 4992, 1400, 23843]"

if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    echo "       pass the path as an argument, or download it (see design.md)." >&2
    exit 1
fi

echo "building (release)..."
cargo build --release --quiet

echo "running: chat completion ($MAX_NEW tokens${NUM_THREADS:+, $NUM_THREADS threads}) of: \"$USER_MSG\""
echo
# `tee` to a temp file so the generation streams to the terminal live (the per-token
# flushes in `generate` make it appear token-by-token) while we still capture it to parse.
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
./target/release/bebelm generate --greedy --max-gen "$MAX_NEW" ${THREAD_OPT[@]+"${THREAD_OPT[@]}"} "$PROMPT" | tee "$TMP"
OUT="$(cat "$TMP")"
echo

# The reply spans multiple lines (it opens with a multi-line <think> block), so capture from
# the "continuation : " line up to (but not including) the "prompt ids" line.
CONT="$(printf '%s\n' "$OUT" | awk '/^continuation : /{f=1} /^prompt ids/{f=0} f')"
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
