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
# Correctness: greedy decoding (temperature 0) is deterministic run-to-run on a given
# binary/machine (pure f32, no RNG), but the exact tokens can drift across builds and
# architectures (FP reduction order, SIMD/FMA contraction, activation quantization). So this
# checks the softer, robust signal — that the reply actually talks about Paris — rather than
# pinning an exact token-id sequence.

set -euo pipefail

MODEL="${1:-LFM2.5-8B-A1B-Q4_K_M.gguf}"
export BEBELM_WEIGHTS_FILE="$MODEL"
MAX_NEW=256

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
# A longer (~200-token) prompt so the benchmark exercises batched prefill, not just decode. It asks
# about the capital of France *without naming it*, so the model reliably says "Paris" itself (in its
# reasoning and/or answer) — the correctness signal checked below.
USER_MSG="I am planning a week-long trip to the capital of France this spring and I would like your help preparing for it. Could you tell me about this city in detail: its most famous landmarks and museums, the historic neighborhoods that are worth exploring slowly on foot, the role its main river plays in the layout and daily life of the city, and a little about its history from the medieval period through the grand nineteenth-century redevelopment of its boulevards. I am also genuinely curious about the local food and cafe culture, the best times of year to visit if I want to avoid the largest crowds, and any practical tips for getting around efficiently using its metro and the regional trains. Please organize your answer clearly into sections, keep everything factual and reasonably concise, and focus on the things that a thoughtful first-time visitor would most want to understand before arriving."
PROMPT=$'<|im_start|>user\n'"$USER_MSG"$'<|im_end|>\n<|im_start|>assistant\n'

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
