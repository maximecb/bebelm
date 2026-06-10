# bebelm — a CPU-only, pure-Rust LFM2.5-8B-A1B

## Goal

Produce a **pure-Rust**, **CPU-only** inference implementation of
**Liquid AI LFM2.5-8B-A1B**, running the **Q4_K_M** GGUF quantization.

### Constraints

- **Pure Rust.** No bindings to llama.cpp / ggml / candle / PyTorch. We write the
  GGUF parser, all kernels, and the model forward pass ourselves.
- **No system-package dependencies.** No `*-sys` crates, nothing that invokes a C
  compiler or links a system library (no OpenBLAS, no `onig`, etc.). Pure-Rust crates
  that make FFI calls into the *already-present* system libc (e.g. `memmap2` → `libc`)
  are acceptable — there is nothing to install and no C toolchain involved.
- **CPU only.** Compute in `f32`. Start single-core; add SIMD + threads later.

### Non-goals (initially)

- GPU / Metal / CUDA.
- Training or fine-tuning.
- Formats other than Q4_K_M (Q4_K + Q6_K + F32/F16). Other quants can be added later
  behind the same dequant interface.

---

## Target weights

Download the official GGUF from Hugging Face:

- Repo: `LiquidAI/LFM2.5-8B-A1B-GGUF`
- File: **`LFM2.5-8B-A1B-Q4_K_M.gguf`** (~5.16 GB)
- Direct URL:
  `https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/main/LFM2.5-8B-A1B-Q4_K_M.gguf`

```sh
curl -L -o LFM2.5-8B-A1B-Q4_K_M.gguf \
  "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/main/LFM2.5-8B-A1B-Q4_K_M.gguf"
```

(Verified live: the `resolve/main` URL returns HTTP 200 and redirects to the HF Xet CDN.)

`Q4_K_M` is a *mixed* quantization: most 2D weight matrices are **Q4_K**, while a few
quality-sensitive tensors (typically the token-embedding/output matrix, the attention
`v` projections, and some `ffn_down`) are stored as **Q6_K**. 1-D tensors (RMSNorm
gains, the MoE router, expert biases) are stored as **F32**. Our loader must therefore
dispatch on the per-tensor dtype recorded in the GGUF header.

> ✅ Done (milestone 1): inspecting the file confirmed it has 256 tensors with the
> Q4_K/Q6_K/F32 mix and the tensor names recorded in the verified mapping below.

---

## Model architecture

`Lfm2MoeForCausalLM` — a hybrid backbone of **gated short-convolution** blocks and a
small number of **grouped-query attention** blocks, with **sparse-MoE** SwiGLU FFNs on
all but the first two layers. Pre-norm (RMSNorm) throughout.

### Hyperparameters (from official `config.json`)

| Field | Value |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 24 |
| `vocab_size` | 128000 |
| `tie_word_embeddings` | **true** (output projection == token-embedding matrix) |
| `norm_eps` (RMSNorm) | 1e-5 |
| **Attention** | |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA, group size 4) |
| `head_dim` | 64 (= 2048 / 32) |
| q/k layernorm | RMSNorm over `head_dim`, applied before RoPE |
| `rope_theta` | 5,000,000 (default RoPE) |
| `max_position_embeddings` | 128000 |
| **Short conv** | |
| `conv_L_cache` (kernel size) | 3 |
| `conv_bias` | false |
| **MoE** | |
| `num_experts` | 32 |
| `num_experts_per_tok` (top-k) | 4 |
| `moe_intermediate_size` | 1792 |
| `num_dense_layers` | 2 (layers 0–1 are dense, not MoE) |
| `use_expert_bias` | true |
| `norm_topk_prob` | true |
| `routed_scaling_factor` | 1.0 |
| **Dense FFN** (layers 0–1) | |
| `intermediate_size` | 7168 |

### Layer schedule (`layer_types`, 0-indexed)

```
0  conv     6  attn    12 conv    18 attn
1  conv     7  conv    13 conv    19 conv
2  attn     8  conv    14 attn    20 conv
3  conv     9  conv    15 conv    21 attn
4  conv    10  attn    16 conv    22 conv
5  conv    11  conv    17 conv    23 conv
```

- **Attention layers:** 2, 6, 10, 14, 18, 21  (6 total)
- **Conv layers:** all others (18 total)
- **Dense FFN layers:** 0, 1.  **MoE FFN layers:** 2–23.

### Decoder layer (pre-norm, residual-first)

```
h = h + operator(operator_norm(h))    # operator = short-conv OR attention
h = h + feed_forward(ffn_norm(h))      # feed_forward = dense MLP OR MoE
```

### Block: gated short convolution (the novel part)

`in_proj`: 2048 → 3·2048, split into B, C, x (each 2048 wide).

```
B, C, x  = split(in_proj(h), 3)
Bx       = B * x                                   # elementwise gate
conv_out = causal_depthwise_conv1d(Bx, W_conv)     # kernel size 3, groups=2048, no bias, no activation
y        = C * conv_out                            # second elementwise gate
out      = out_proj(y)                             # 2048 → 2048
```

- Depthwise: each of the 2048 channels has its own length-3 causal filter.
- "Causal" = output at position t depends on positions t-2, t-1, t only.
- **Conv state cache** (decode): keep the last `L_cache-1 = 2` columns of `Bx` per
  conv layer so single-token steps don't recompute history. (Replaces a KV cache for
  these layers; far smaller.)

### Block: grouped-query attention

```
q = q_layernorm(reshape(q_proj(h)))   # 32 heads × 64
k = k_layernorm(reshape(k_proj(h)))   #  8 heads × 64
v =            reshape(v_proj(h))     #  8 heads × 64
q, k = rope(q, k, theta=5e6, pos)
k, v = repeat_kv(k, v, groups=4)      # broadcast 8 → 32 heads
attn = softmax(q·kᵀ / sqrt(64) + causal_mask) · v
out  = o_proj(merge_heads(attn))
```

- **KV cache:** store k, v per attention layer for all past positions. 6 layers ×
  8 kv-heads × 64 = 3072 floats each for k and v per token. f32 ≈ 24 KB/token;
  optionally store as f16 to halve it.

### Block: dense FFN (layers 0–1) — SwiGLU

```
out = down_proj( silu(gate_proj(h)) * up_proj(h) )   # 2048→7168→2048
```

### Block: sparse MoE FFN (layers 2–23)

```
logits = router(h)                        # 2048 → 32 (router weight, F32)
score  = sigmoid(logits)                  # NOT softmax
sel    = topk_indices(score + expert_bias, k=4)   # bias used for SELECTION only
w      = gather(score, sel)               # weights are the sigmoid scores (no bias)
w      = w / (sum(w) + 1e-6)              # norm_topk_prob
w      = w * routed_scaling_factor        # = 1.0
out    = Σ_e  w[e] · down_e( silu(gate_e(h)) * up_e(h) )   # each expert SwiGLU, 2048→1792→2048
```

> To verify against the reference during implementation: that the gathered weights come
> from `score` (sigmoid) and not from `score + bias`, and the exact `1e-6` epsilon.

---

## Quantization formats (Q4_K_M)

All quantized matmuls dequantize weights to `f32` on the fly; activations stay `f32`.

### Q4_K — super-block of 256 weights, 144 bytes (~4.5 bits/weight)

```
d:      f16                  # super-scale for the 8 sub-block scales
dmin:   f16                  # super-scale for the 8 sub-block mins
scales: u8[12]               # 8×6-bit scales + 8×6-bit mins, bit-packed
qs:     u8[128]              # 256 × 4-bit quants (nibbles)
```
256 weights split into 8 sub-blocks of 32. For sub-block j with 6-bit `sc_j`, `m_j`:
`w = d·sc_j·q − dmin·m_j`. (Scale/min unpacking = ggml `get_scale_min_k4`.)

### Q6_K — super-block of 256 weights, 210 bytes (~6.5625 bits/weight)

```
ql:     u8[128]              # low 4 bits of each quant
qh:     u8[64]               # high 2 bits of each quant
scales: i8[16]               # 16 sub-block scales (one per 16 weights)
d:      f16                  # super-scale
```
6-bit quant `q ∈ [0,63]`, recentered: `w = d · scale_subblock · (q − 32)`.

### F16 / F32

1-D tensors (RMSNorm gains, router, expert bias, conv filters) are stored **F32** and
read directly. The file has **no F16 or BF16 whole tensors** — f16 only appears as the
per-block scales *inside* Q4_K/Q6_K, so we need a correct IEEE **f16→f32** for those (no
bf16 path required).

---

## GGUF file format (what we parse)

- Header: magic `GGUF`, version, tensor count, metadata-KV count.
- Metadata KV pairs: typed values (incl. arrays) — we read hyperparameters,
  tensor-name list, and (later) the tokenizer vocab/merges/scores from here.
- Tensor info: name, n_dims, shape, dtype (ggml type enum), byte offset.
- Tensor data blob: aligned (default 32-byte alignment), mmapped read-only.

We map the file once and expose each tensor as a zero-copy `&[u8]` slice + dtype +
shape. Dequant kernels read straight from these slices.

### Tensor-name mapping (VERIFIED against the Q4_K_M file — 256 tensors)

GGUF architecture string: **`lfm2moe`**. dtype mix observed: **F32 ×123** (6.3 MiB),
**Q4_K ×118** (3.67 GiB), **Q6_K ×15** (1.12 GiB); total 4.79 GiB. No F16/Q8_0 *tensors*
(f16 appears only as in-block scales inside Q4_K/Q6_K — so no bf16 path is needed).
GGUF lists dims fastest-first; for a weight `y = W·x`, dims are `[in_features, out_features]`.

```
# --- global ---
token_embd.weight         Q6_K  [2048, 128000]   # tied: also the output projection
token_embd_norm.weight    F32   [2048]           # FINAL RMSNorm before logits
                                                  #   (LFM2's "embedding_norm"; there is
                                                  #    NO output_norm / output.weight tensor)

# --- per block {i}, 0..23 ---
blk.{i}.attn_norm.weight  F32   [2048]           # pre-OPERATOR norm (conv AND attn layers)
blk.{i}.ffn_norm.weight   F32   [2048]           # pre-FFN norm

# operator — attention layers {2,6,10,14,18,21}:
blk.{i}.attn_q.weight     Q4_K  [2048, 2048]     # 32 heads × 64
blk.{i}.attn_k.weight     Q4_K  [2048, 512]      #  8 kv-heads × 64
blk.{i}.attn_v.weight     Q4_K  [2048, 512]
blk.{i}.attn_output.weight Q4_K [2048, 2048]
blk.{i}.attn_q_norm.weight F32  [64]             # RMSNorm over head_dim, before RoPE
blk.{i}.attn_k_norm.weight F32  [64]

# operator — conv layers (all others):
blk.{i}.shortconv.in_proj.weight  Q4_K [2048, 6144]  # 2048 → 3·2048 (B,C,x)
blk.{i}.shortconv.conv.weight     F32  [3, 2048]     # depthwise filter [L_cache, channels]
blk.{i}.shortconv.out_proj.weight Q4_K [2048, 2048]

# FFN — dense layers {0,1}:
blk.{i}.ffn_gate.weight   Q4_K  [2048, 7168]
blk.{i}.ffn_up.weight     Q4_K  [2048, 7168]
blk.{i}.ffn_down.weight   Q6_K  [7168, 2048]

# FFN — MoE layers (all others, 2..23):
blk.{i}.ffn_gate_inp.weight  F32  [2048, 32]         # router (sigmoid; expert_gating_func=2)
blk.{i}.exp_probs_b.bias     F32  [32]               # expert bias (selection only)
blk.{i}.ffn_gate_exps.weight Q4_K [2048, 1792, 32]   # stacked: [in, out, n_experts]
blk.{i}.ffn_up_exps.weight   Q4_K [2048, 1792, 32]
blk.{i}.ffn_down_exps.weight Q6_K [1792, 2048, 32]
```

Notes:
- A subset of `ffn_down` tensors are Q6_K and the rest Q4_K (the Q4_K_M recipe upgrades
  some); the loader dispatches per-tensor from the GGUF dtype, so the exact split is
  irrelevant to us.
- The conv/attn schedule is *also* encoded in metadata as the per-layer array
  `lfm2moe.attention.head_count_kv = {0,0,8,0,0,0,8,…}` (0 = conv, 8 = attention). We can
  derive the operator type from this array or from tensor presence.
- Tokenizer metadata (phase 2): `tokenizer.ggml.model = "gpt2"` (byte-level BPE),
  `tokenizer.ggml.pre = "lfm2"`, 293,320 merges, 128,000 tokens; bos=124894, eos=124900,
  pad=124893. A ChatML-like `tokenizer.chat_template` is also embedded.

---

## Tokenizer (deferred)

**Phase 1: defer.** Drive the model on raw token IDs and validate the math against a
reference (e.g. compare logits/next-token to llama.cpp on a fixed prompt) before
worrying about text I/O.

**Phase 2: hand-rolled, pure Rust, no `regex` crate.** Build a byte-level BPE tokenizer
from the vocab + merges embedded in the GGUF metadata, operating **directly on UTF-8
strings** (hand-written pre-tokenization splitting rather than a regex engine). Apply
the model's ChatML-style chat template for the instruct model.

---

## Inference pipeline

1. **Load:** mmap GGUF → parse header → build `Config` + tensor table.
2. **Prefill:** run the full prompt through all layers; populate KV cache (attention
   layers) and conv-state cache (conv layers).
3. **Decode:** one token at a time, reusing both caches; sample; repeat to EOS / limit.
4. **Sampling — one sampler only (KISS):** temperature + top-k, with `temperature == 0`
   short-circuiting to argmax (greedy). Defaults follow Liquid's recommendation for this
   model: **temperature 0.2, top-k 80, repetition_penalty 1.05**. `temperature == 0`
   gives the deterministic path we need to validate against a reference. Top-k via a
   size-k min-heap (no full sort). Repetition penalty = divide each already-generated
   token's logit by 1.05. Small hand-rolled PRNG (no `rand` crate). Logits =
   `token_embd · h_final` (tied weights). No top-p / min-p / beam search.

### Caches

- **KV cache** — per attention layer, grows with sequence length.
- **Conv-state cache** — per conv layer, fixed `(2048 × 2)`; tiny.

---

## Kernels

Single-core scalar `f32` first; correctness before speed. Each kernel in its own file
under `src/kernels/`.

| File | Kernel | Notes |
|---|---|---|
| `dequant.rs` | Q4_K, Q6_K, F16→f32, bf16→f32 block decoders | foundation for all matmuls |
| `matmul.rs` | quantized matmul: `y = W·x` (GEMV for decode, GEMM for prefill) | dequant-on-the-fly; the hot path |
| `rmsnorm.rs` | RMSNorm (+ weight gain) | used for operator/ffn/q/k/final norms |
| `rope.rs` | rotary position embedding | theta 5e6, applied to q,k |
| `softmax.rs` | numerically-stable softmax | attention scores |
| `attention.rs` | GQA scaled-dot-product attention w/ KV cache | q·kᵀ, mask, softmax, ·v, GQA broadcast |
| `conv.rs` | causal depthwise conv1d (kernel 3) + conv-state update | per-channel filters |
| `activation.rs` | SiLU, sigmoid, SwiGLU glue | FFN + MoE routing |
| `elementwise.rs` | residual add, elementwise mul (gates), scale | small helpers |

MoE routing logic (sigmoid → bias → top-4 → normalize → weighted sum) lives in the model
layer (`src/model.rs`); the expert MLPs reuse `matmul.rs` + `activation.rs`.

---

## Crate dependencies

Pure Rust, no system packages. Each justified; minimal tree.

| Crate | Phase | Why | System deps? |
|---|---|---|---|
| `memmap2` | core | mmap the ~5 GB GGUF: lazy paging, shared page cache, no 5 GB upfront read+alloc | None — pure-Rust FFI to system libc; nothing to install. |
| `rayon` | phase 2 (opt 9c) | data-parallel matmul rows / per-expert / per-head work across cores | None — built on std threads. |
| `wide` | phase 2 (opt 9d) | cross-platform portable SIMD (`f32x8`) for dot/dequant; compiles to SSE/AVX2/NEON | None — pure Rust, no nightly. |
| `bytemuck` | phase 2 (opt 9h/9k) | zero-copy reinterpret between same-size POD SIMD vectors (split/bit-cast in the quant kernels); already transitive via `wide`. *Declared but not yet referenced* — kept for the int-dot / unpack work. | None — pure Rust. |

> `half` was **dropped**: the file has no F16/BF16 whole tensors, so we only need f16→f32
> for in-block scales — hand-rolled as a tested ~25-line `f16_to_f32` in
> `kernels/dequant.rs`. Current dependency tree: `memmap2`, `rayon`, `wide`, `bytemuck`
> (the last already pulled in transitively by `wide`).

**Deliberately NOT taking crates for:**

- **CLI args** → `std::env::args` by hand (no `clap`).
- **Sampling RNG** → ~10-line PCG/xorshift (no `rand`).
- **Errors** → `Box<dyn Error>` / small enums (no `anyhow`/`thiserror`).
- **SIMD** (phase 2) → favor **cross-platform** portable SIMD via the `wide` crate
  (pure Rust, stable, no system deps) over `std::arch` platform intrinsics. `std::simd`
  is equivalent but nightly-only.
- **Tokenizer** → hand-rolled byte-level BPE on UTF-8 (no `regex`, no `tokenizers`; the
  latter's default `onig` feature is a C library = forbidden system dep).

---

## Project layout

A library crate (`lib.rs`) holds everything; `main.rs` is a thin CLI over it (this keeps
`pub` kernels off the dead-code lint as the model is built bottom-up). `✓` = implemented.

```
src/
  lib.rs             # ✓ library surface (pub mod gguf/tensor/kernels/…)
  main.rs            # ✓ CLI: dump + dequant (later: load model, prompt, generate)
  gguf.rs            # ✓ GGUF parser + mmap-backed tensor table
  tensor.rs          # ✓ dtype enum + block sizing
  config.rs          # ✓ hardcoded architecture consts + validate(&gguf)
  model.rs           # ✓ weight loading + static forward pass (embed→layers→norm→logits)
  cache.rs           # ✓ KV cache (attn) + conv-state cache (conv)
  sampler.rs         # ✓ temperature + top-k (+ rep penalty); temp 0 = greedy; hand-rolled PRNG
  tokenizer.rs       # ✓ byte-level BPE from GGUF (hand-rolled lfm2 pretokenizer, no regex)
  agent.rs           # ✓ conversation session: transcript + caches, generation, tool loop
  tool.rs            # ✓ function calling: Tool/Schema/ToolCall + Pythonic call parser (no serde)
  chat.rs            # ✓ interactive chat REPL (bin module under main.rs)
  kernels/
    mod.rs           # ✓
    dequant.rs ✓  matmul.rs ✓  rmsnorm.rs ✓  rope.rs ✓
    softmax.rs ✓  attention.rs ✓  conv.rs ✓  activation.rs ✓  elementwise.rs ✓
```

---

## Implementation milestones

1. ✅ **GGUF loader + tensor dump.** Parse header, list tensors (name/dtype/shape);
   confirmed the Q4_K/Q6_K/F32 mix and the verified tensor-name mapping above.
2. ✅ **Dequant kernels** (`kernels/dequant.rs`): hand-rolled `f16_to_f32`, Q4_K + Q6_K
   block decoders (exact ggml reference) + an F32/F16/Q4_K/Q6_K dispatcher. Unit-tested on
   synthetic blocks and validated on real tensors from the GGUF.
3. ✅ **Quantized GEMV + RMSNorm** (`kernels/matmul.rs`, `kernels/rmsnorm.rs`):
   `matvec(dtype, W, n_in, n_out, x, y)` (dequantize-row-then-dot, reused buffer) and
   `rmsnorm(x, gain, eps, out)`. Unit-tested on hand-computed cases (incl. row-major
   layout check). Also restructured into a lib crate + thin bin (clean dead-code story).
4. ✅ **Config + model loading** (`config.rs`, `model.rs`): architecture as hardcoded
   `const`s + `validate(&gguf)`; `Model::load` resolves all 256 tensors by name and
   checks shapes, confirmed against the real file. (We chose a **static
   forward pass** — see note below — rather than runtime config interpretation.)
5. ✅ **Remaining kernels + single forward pass** (no cache): `rope`, `softmax`,
   `activation` (SiLU/SwiGLU), `elementwise`, `conv`, `attention`; wired the static layer
   loop (embed → 24 layers → final norm → logits) incl. MoE routing. (Correctness
   confirmed end-to-end by the milestone-8 continuation test, not a logit-reference script.)
6. ✅ **KV + conv-state caches** (`cache.rs`): single-token cached forward path
   (`forward_step`), prefill-then-decode generation. Output bit-identical to the
   uncached path; generation now O(n) (4 tok 34s→13s, 16 tok ~29s vs ~230s).
7. ✅ **Sampling + generation** (`sampler.rs`, `Model::generate`): one sampler (temp +
   top-k, temp 0 = greedy, rep penalty), hand-rolled PRNG; autoregressive loop (no cache
   yet). `bebelm generate` works on raw token ids.
8. ✅ **Tokenizer** (`tokenizer.rs`): byte-level BPE from the GGUF, hand-rolled lfm2
   pre-tokenizer (no `regex`), GPT-2 byte↔char table, merges. Round-trips on the real
   vocab. **Correctness gate PASSED:** `bebelm complete … "The capital of France is"` →
   " the city of Paris" (fluent + factually correct), validating the whole pipeline.
9. **Optimizations** (sub-items 9a–9l in the *Optimizations* section above; ordered
   easiest-first with impact estimates). Baseline already has MoE sparsity + caches;
   **9a–9e + 9i (MoE expert batching) + 9j (AVX2 build) done** — now **~16 tok/s decode /
   ~17–19 prefill** on the M5 MacBook Air (~18× over the 0.87 single-core baseline).
   Profiling (`profile.sh`) shows decode is **compute-bound**: ~82% of samples sit in the
   quantized dot kernels (Q4_K `nibble_dot32`/`dot_q4k_block`, Q6_K `dot_q6k_block`), ~18%
   in fork/join idle, and only ~11% of the M5's 153 GB/s is used (~1.1 GB active weights/
   token × ~16 tok/s ≈ 17 GB/s). So decode is compute-bound, but *not* on the unpack: the
   **9k** experiment (vectorizing the K-quant unpack) was bit-identical yet ~neutral on M5,
   so the dot kernels are bound by the fixed per-8-weight f32 FMA + load throughput, not the
   sub-byte unpack. The real compute lever is therefore **9h** (integer dot). Remaining:
   9h, plus 9f (long prompts) / 9g (long contexts); 9k/9l deprioritized on M5.
  - llama.cpp has a custom GEMM (matrix multiply) and MoE routing kernels tuned for
    hybrid (convolution + attention) architecture. We could potentially take inspiration
    from this, but we should probably start by profiling our current kernels.
10. **Cleanup:** remove unused code and validation code that is no longer needed.

## Optimizations

Baseline after milestone 6 (Apple silicon, single-core scalar): **~0.87 tok/s decode**
(`./benchmark.sh`), dominated by `matvec` (dequantize-on-the-fly + dot, re-done per token).
Already in place: **MoE sparsity** (only the top-4 experts run per token) and the **KV +
conv-state caches** (decode is O(n)). `benchmark.sh` greedily generates 8 tokens, reports
prefill/decode tok/s, and checks the (deterministic) continuation against an expected
string — use it to measure each optimization below.

Ordered easiest → hardest, with rough impact. Effects are roughly **multiplicative**
(threads × SIMD-lanes × cache), so the combined ceiling is large (~10–30× over baseline).

| # | Optimization | Effort | Rough impact |
|---|---|---|---|
| **9a ✅** | **Skip prefill logits** — only the last prompt token needs the `2048×128000` logit matmul (`run_layers` vs `forward_step`); skip it for the rest. | trivial | **measured ~13% faster prefill** (0.8→0.9 tok/s, 6-tok prompt); no decode effect |
| **9b ✅** | **Precompute small F32 tensors at load** — dequantize the F32 norm/conv/bias tensors once into `Model` (~0.85 MB; router excluded) and read via `f32()`; removed `dequant_vec` + ~101 allocs/token. See notes. | easy | **measured: within noise** — F32 work is dwarfed by the matmuls; cleaner code, no speed change |
| **9c ✅** | **Multithread `matvec` over output rows (`rayon`)** — rows are independent (dequant row + dot). The single hot path (all projections, experts, logits). Serial fallback below 64 rows (router, k/v proj). | easy | **measured ~5.3× decode** (0.87→4.6 tok/s) + ~5–6× prefill on 10 cores (4P+6E); ids bit-identical |
| **9d ✅** | **Cross-platform SIMD for dot + dequant MAC** — **`wide`** `f32x8` (= one 256-bit AVX2 reg; 2× 128-bit NEON on arm64). `wide` has no `f32x16` (that's AVX-512-only, absent here); for more throughput use multiple `f32x8` accumulators (ILP), not wider lanes. Favor `wide` over `std::arch`; `std::simd` is nightly. | moderate | **measured ~1.75× decode** from the SIMD `dot` alone (4.75→8.31 tok/s); **~2.8× decode / 4.0× prefill** total once fused with 9e (→13.4 / 15.2 tok/s on 10 cores); ids still bit-identical |
| **9e ✅** | **Fuse dequant-and-dot in `matvec`** — accumulate per block instead of materializing a full dequantized row buffer (better cache locality). | moderate | **done as part of 9d**: Q4_K rows dequantize each 256-weight block straight into the `f32x8` dot via `Σ(d·q−min)·x = d·Σ(q·x)−min·Σx` (scale/min applied once per sub-block). **Q6_K rows now fused too** (`Σ w·x = d·Σ_sub sc·Σ(q−32)·x`): **+~12% decode / +8% prefill** on top (→14.98 / 16.4 tok/s) — smaller than its MAC share since the Q6_K fallback already used the SIMD `dot`, so only its dequant arithmetic + scratch traffic improved. Only the tiny F32 router still uses the dequant-then-dot path |
| **9f ✅** | **Batched GEMM prefill** — `matmul` (token-major `Y=W·X`) applies each weight row to all prompt tokens at once; conv/attention reuse the per-token kernels over the cache prefix; MoE is **token-grouped** (each expert runs once over the tokens that picked it). Per dtype: Q4_K = blocked Q8 int dot (9l), Q6_K = unpack-once f32, F32 = dequant-once. Driven from `Agent::generate` in 512-token chunks, with a per-token fallback when the KV window would slide mid-prefill. **Bit-for-bit identical** to per-token prefill (`matmul_matches_per_token_matvec` + e2e `batched_prefill_matches_per_token`). | moderate | **measured ~2.1× prefill on M5** (43→90 tok/s, 1153-tok prompt). The big lever was **batching Q6_K**: profiling showed the per-token f32 `fused_row_dot` (the MoE `ffn_down_exps`) was ~75% of prefill because it re-ran the scalar 6-bit unpack per token; unpacking once per block → ~1.5× alone. |
| **9g** | **f16 KV cache** — store K/V as f16 to halve attention memory bandwidth. | easy–moderate | small now; grows with context length |
| **9h** | **Q8 activation + integer block dot** (llama.cpp `vec_dot`) — *per-matmul*, quantize the **input vector** to Q8_K (activations stay f32 everywhere else) and dot in the integer domain directly against the Q4_K/Q6_K quants; no f32 weight dequant. Supersedes 9d/9e on the hot path. See notes. | hard | **HIGH ~2–4×**, most complex |
| **9i ✅** | **Batch MoE expert matvecs** — run all selected experts' gate+up rows (then all down rows) as one parallel region (`matvec_fused_batch`) instead of 12 separate fork/joins per MoE layer. | easy | **measured ~+18% decode** (13.7→~16 tok/s); fewer fork/join barriers → better core saturation |
| **9j ✅** | **AVX2 + FMA on x86** — `.cargo/config.toml` sets `target-cpu=native`; the default x86_64 baseline is SSE2-only, so `wide`'s `f32x8` ran at half width. Scoped to x86_64 (arm64/NEON untouched). | trivial | x86 target only (not yet measured on the i5); arm64 unchanged |
| **9k ⚠️** | **Vectorize the K-quant unpack** — replace the per-lane scalar gather in `nibble_dot32` with in-vector `wide` widening (`u8x16 → i16x16 → i32x8 → round_float → f32x8`, split via `bytemuck::cast`). **Tried on `nibble_dot32`: bit-identical, but measured ~neutral on M5 (≤4%, within noise) → reverted.** The unpack isn't the dot's bottleneck (the fixed f32 FMA/load throughput is). May still pay off on x86/AVX2 (untested), where the scalar gather likely costs more. See notes. | moderate | **~0 on M5** (NEON); possibly positive on AVX2 |
| **9l ✅ (prefill)** | **Multi-row register blocking** — `matmul`'s Q4_K path computes `Q4K_ROW_TILE = 4` output rows per pass (`dot_q4k_rowtile_q8_batch`), giving 4 independent dot chains for ILP and reusing each token's Q8 block across the tile; the per-row dot also defers the horizontal reduction (`wsd_q4k`: one `vaddvq`/block via a `vmlaq` accumulator, vs one per sub-block). Bit-identical. **Applied to batched prefill only** (decode `matvec` is still single-row). | moderate | **~4% on M5** on top of 9f's Q4_K path (the int8 dot, not the reduce, is the floor); decode unchanged |
| **9m ✅** | **`#[inline(always)]` on hot kernel leaves** — force-inline the small private functions in the `matvec`/dequant inner loops so they fold into the per-block dot (`load8`, `nibble_dot32`, `nibble_idot32`, `q6_dot16`, `get_scale_min_k4`, and the single-call-site per-block dots `dot_q4k_block`/`dot_q6k_block`/`dot_q4k_block_q8`/`dot_q4k_row_q8`). See notes. | trivial | within noise (≈neutral–slightly positive on M5); output bit-identical |

Suggested order: **9a–9f + 9i + 9j + 9l(prefill) + 9m done**. Next: **9h** (int8 Q6_K dot — now
the top prefill lever, see note), **9g** for long contexts. Further fork/join trimming beyond 9i
was tried (batching q/k/v + dense gate/up) and measured **neutral**, so it's deprioritized.

**Prefill perf vs llama.cpp (the gap).** Reference llama.cpp does ~500 tok/s prefill on this
model/machine; bebelm batched prefill is now **~90 tok/s** (1153-tok prompt, all cores), up from
~43 per-token. Thread scaling is ~2.6× at 4 P-cores, ~3.3× at 10 (the 6 E-cores add ~27% — weak
for int8 GEMM), so per-core kernel efficiency still dominates the remaining gap. Done so far: 9f
(batch) + 9l (Q4_K block/ILP) + Q6_K unpack-once. Remaining levers, biggest first:
- **Int8 Q6_K dot (9h)** — Q6_K (the MoE `ffn_down_exps`) is now the largest slice: it batches the
  unpack but still dots in **f32** (~¼ the density of the Q4_K `sdot`). Recentering `q−32` to i8
  and using `sdot` (as llama.cpp's `vec_dot_q6_K_q8_K` does) would lift it. **Caveat:** this
  quantizes the Q6_K matmul's *activations* to Q8 — a numerics change — so it must also replace the
  decode Q6_K path to keep prefill≡decode, and re-validate the golden prefix.
- **Heterogeneous-core scheduling** — the 6 E-cores barely help; P-core-only affinity or
  size-weighted work splitting could recover some of the plateau.

### Notes on selected sub-items

**9b — exact F32 tensors to precompute.** These were re-dequantized via `dequant_vec` on
every `forward_step` (~101 small heap allocations + copies per token). Pre-dequantized once
at load into `Model.f32_cache` (~0.85 MB). The F32 **router** (`ffn_gate_inp`, ~5.8 MB —
the bulk of the F32 footprint) is *excluded*: it goes through `matvec` on raw bytes, not
`f32()`. The precomputed set:

| Tensor | shape | count |
|---|---|---|
| `blk.{i}.attn_norm.weight` (operator pre-norm gain) | F32 [2048] | 24 |
| `blk.{i}.ffn_norm.weight` (FFN pre-norm gain) | F32 [2048] | 24 |
| `blk.{i}.attn_q_norm.weight` / `attn_k_norm.weight` | F32 [64] | 6 + 6 |
| `blk.{i}.shortconv.conv.weight` (depthwise filter) | F32 [3×2048] | 18 |
| `blk.{i}.exp_probs_b.bias` (expert bias) | F32 [32] | 22 |
| `token_embd_norm.weight` (final norm gain) | F32 [2048] | 1 |

The F32 *router* matrices `ffn_gate_inp.weight` `[2048×32]` go through `matvec` (not
`dequant_vec`), so they're separate and lower priority.

**9d — SIMD width.** AVX2 is 256-bit = **8×f32**, so `f32x8` is the correct width (one
YMM register); on arm64 NEON (128-bit = 4×f32) it lowers to 2× `f32x4`. `f32x16` would be
AVX-512 (512-bit) — not present on this machine and not provided by `wide`. To exceed
8-wide throughput, use **multiple independent `f32x8` accumulators** to hide FMA latency,
not wider lanes.

**9k — portable SIMD unpack (tried, ~neutral on M5).** `wide` 0.7.33 *does* provide the
widening conversions (`From<u8x16> for i16x16`, `From<i16x8> for i32x8`, `i32x8::round_float()
-> f32x8`), so the sub-byte → f32 expansion can stay in-vector: per 16 weights, `u8x16 →
i16x16`, mask/shift the nibble (`& 0x0f`, `>> 4 & 0x0f`), split into two `i16x8` (`bytemuck::
cast`, no memory round-trip), `→ i32x8 → round_float → f32x8`, then the existing `mul_add`.
Keeping lane order identical makes it **bit-identical** (no `EXPECTED_IDS` change). **But** an
implementation on `nibble_dot32` measured **~neutral on M5** (≤4%, within run-to-run noise),
with both the `to_array` and `bytemuck` splits — so the split wasn't the issue; the unpack
simply isn't the bottleneck inside the dot (the per-8-weight f32 FMA + weight/`x` loads are,
and those are unchanged). Reverted. The same kernel on x86/AVX2 (the i5 target) is untested and
*might* benefit — the scalar→vector gather (`f32x8::from([f32;8])`) and `vpmovzxbw` widening
have a different cost balance there. If revisited, measure on the i5 specifically.

**9h — what it implies.** *Not* a global switch to Q8 activations; the residual stream,
norms, attention, etc. stay f32. Per matmul: (1) quantize just the input vector `x` to
**Q8_K** (256-value blocks, int8 + scale + per-16 sums) — a cheap one-pass step; (2)
compute the dot in the integer domain against the weight's packed quants (Q4_K nibbles /
Q6_K 6-bit), accumulating int32, then scale by `weight_scale × activation_scale` per
block (plus Q4_K's `min` term). I.e. `vec_dot_q4_K_q8_K` / `vec_dot_q6_K_q8_K`. Benefits:
no f32 weight materialization, compact weight traffic, fast int8 dot. Caveats: it
**replaces** (not stacks with) 9d/9e on the hot path; peak int8-dot throughput relies on
**VNNI** (x86) / **dotprod** (NEON) instructions that `wide`/`std::simd` don't expose, so
a fully portable version gets the algorithmic win but maybe not the absolute peak without
`std::arch` — a slight tension with the cross-platform-SIMD preference.

**9m — inline directive for kernels.** Under the release profile (`codegen-units = 1` +
thin LTO) the compiler already inlines small intra-crate functions, so this mostly (a)
encodes intent, (b) guarantees inlining in the dev profile (`opt-level = 1`) and if the
CGU count is ever split, and (c) forces *cross-crate* inlining of the `pub` leaves. Rule of
thumb for kernel code:

- `#[inline(always)]` → **private** hot-path functions that are either tiny SIMD leaves
  *or* have a single call site (so there's no code-bloat risk — they expand in one place
  regardless): the micro-ops and per-block dots listed in 9m.
- `#[inline]` → **`pub`, multi-call-site** functions (`dot`, `fused_row_dot`,
  `f16_to_f32`): let the optimizer weigh bloat across call sites, and enable cross-crate
  inlining without forcing it.
- Leave the per-call kernel **entry points** (`matvec`, `matvec_fused_batch`, `rmsnorm`,
  `softmax`, `attention_decode`, `conv_step`) and the `model.rs` orchestration un-annotated
  — they run only O(layers)/token, so call overhead is negligible and force-inlining would
  only bloat code with no hot-loop payoff.

---

## Tool use

> ✅ **Implemented** (`src/tool.rs` + `Agent` driver in `src/agent.rs`): tool declarations,
> the Pythonic call parser, and the agentic call/result loop, validated end-to-end against the
> real weights (`tests/end_to_end.rs::tool_call_add_round_trip`).

LFM2.5 supports function calling. The model is *told* about tools in the system prompt,
emits calls as a Pythonic list wrapped in special tokens, and reads results back from a
`tool`-role message. Ground truth for the wire format is the `tokenizer.chat_template`
embedded in the GGUF (cross-checked against Liquid's
[tool-use docs](https://docs.liquid.ai/lfm/key-concepts/tool-use)).

### Wire format (verified against the embedded chat template)

**1. Tool declarations** — appended to the *system* block, on a line after any system text.
The template builds `system_prompt + "\nList of tools: [" + (tools | join(", ")) + "]"`:

```
<|im_start|>system
{optional system text}
List of tools: [{"name": "get_weather", "description": "...", "parameters": {"type":"object","properties":{...},"required":[...]}}, {...}]<|im_end|>
```

Each tool is a JSON object with `name`, `description`, and `parameters` (a JSON-Schema
object).

**2. Tool-call emission** — the assistant turn produces a *Python-style* call list between
`<|tool_call_start|>` and `<|tool_call_end|>`, before the turn's closing `<|im_end|>`:

```
<|tool_call_start|>[get_weather(location='Paris', units='celsius')]<|tool_call_end|>
```

Multiple calls are allowed (comma-separated). String args are quoted; per the template,
dict/list args are JSON-encoded and numbers/bools stringified. (Adding *"Output function
calls as JSON"* to the system prompt switches the model to JSON-object calls, but Pythonic
is the default and what the template renders.)

**3. Tool-result feedback** — results go back as a `tool`-role message; then the assistant
turn is reopened to let the model use them:

```
<|im_start|>tool
{result text, typically JSON}<|im_end|>
<|im_start|>assistant
```

**⚠️ Token caveat (LFM2 vs LFM2.5).** The vocab carries `<|tool_list_start|>` /
`<|tool_list_end|>` (124903/4) *and* `<|tool_call_start|>` / `<|tool_call_end|>` (124905/6).
Per Liquid's docs the `<|tool_list_*|>` (and `<|tool_response_*|>`) tokens are the *older
LFM2* convention; **LFM2.5 does not use them** — it uses the plaintext `List of tools:`
preamble plus only the `<|tool_call_start/end|>` pair. So we ignore the `tool_list` tokens
even though they exist in the vocab.

### Rust interface

Two constraints shaped this: add **no new dependency** (so tool schemas and parsed call
args are strings, not `serde_json::Value`), and keep `Agent: Clone` (so the tool set is
shared behind `Arc`).

A `Tool` value carries the JSON declaration plus a callback; `ToolCall` is one parsed call:

```rust
// src/tool.rs
#[derive(Clone)]
pub struct Tool {
    name: String,                          // must match the name inside `schema`
    schema: String,                        // {"name":...,"description":...,"parameters":{...}}
    call: Arc<dyn Fn(&ToolCall) -> String>, // Arc keeps Agent Clone; 'static (tools own state)
}
impl Tool {
    // Structured params via the `Schema` builder; the full tool JSON is assembled for you
    // (name/description JSON-escaped).
    pub fn new(name: impl Into<String>, description: &str, params: Schema,
               call: impl Fn(&ToolCall) -> String + 'static) -> Self { ... }
    // Escape hatch: supply the entire tool JSON (incl. `parameters`) verbatim.
    pub fn raw(name: impl Into<String>, schema: impl Into<String>,
               call: impl Fn(&ToolCall) -> String + 'static) -> Self { ... }
}

/// JSON-Schema parameter type → emits `"type":"string"|"integer"|"number"|"boolean"`.
pub enum Type { Str, Int, Num, Bool }   // extend later: Enum(&[&str]), Array(Box<Type>)

/// Builder for a tool's `parameters` object; renders the standard JSON-Schema string
/// `{"type":"object","properties":{…},"required":[…]}` via a small hand-rolled emitter
/// (no serde). `type` is kept because it's part of the format the model was trained on.
pub struct Schema { /* Vec<{ name, ty: Type, desc, required }> */ }
impl Schema {
    pub fn new() -> Self { ... }
    pub fn req(self, name: &str, ty: Type, desc: &str) -> Self { ... } // required field
    pub fn opt(self, name: &str, ty: Type, desc: &str) -> Self { ... } // optional field
}

pub struct ToolCall {
    pub name: String,
    args: Vec<(String, String)>,           // arg name -> raw value text (quotes stripped for
    pub raw: String,                       //   simple literals; {..}/[..] passed verbatim)
}
impl ToolCall { pub fn arg(&self, name: &str) -> Option<&str> { ... } }

// Parse the content between the tool-call tokens into calls. Hand-rolled (consistent with
// the no-regex tokenizer): quoted strings, numbers, bools, brace/bracket-balanced values.
pub(crate) fn parse_tool_calls(s: &str) -> Vec<ToolCall> { ... }
```

`Agent` gains a tool list, a system-block emitter, and an agentic driver:

```rust
pub struct Agent<'m> { /* ...existing... */ tools: Vec<Tool> } // Tool: Clone, so derive holds

impl<'m> Agent<'m> {
    pub fn add_tool(mut self, tool: Tool) -> Self { ... } // register (before the system block)

    // Emit `<|im_start|>system\n{text}\nList of tools: [...]<|im_end|>\n`. Must be appended
    // first (the system block follows BOS, before any user turn).
    pub fn append_system(&mut self, text: &str) { ... }

    // Generate; on each emitted tool call, dispatch to the callback, append the tool-role
    // result, and continue — until the model answers with no tool call (or `max_rounds`).
    pub fn assistant_turn_with_tools(
        &mut self,
        max_rounds: usize,
        on_token: impl FnMut(u32, &str),
        on_tool: impl FnMut(&ToolCall, &str),   // observe call + result (UI / logging)
    ) -> Turn { ... }
}
```

Driver loop:
1. `append(ASSISTANT_OPEN)`, then `generate(...)`, **also stopping at `<|tool_call_end|>`**
   (add it to the stop set when tools are registered) so we act without decoding the rest.
2. Scan `turn.ids` for `<|tool_call_start|> … <|tool_call_end|>`. None ⇒ final answer: close
   with `<|im_end|>\n` and return.
3. Otherwise close the assistant turn, `parse_tool_calls`, invoke each `Tool::call`, append
   `<|im_start|>tool\n{joined results}<|im_end|>\n`, and loop (bounded by `max_rounds`).

`max_rounds` caps the number of assistant turns; the final one is reserved to *use* the last
results, so at most `max_rounds - 1` rounds of tools run. An unknown tool name dispatches to an
`"Error: no tool named …"` string fed back to the model rather than aborting. The returned
`Turn` aggregates every round: `ids`/`text` concatenate, `stats` sum, and `stop` is the last
round's reason (`Eos` for a normal answer, the new `StopReason::ToolCall` if the budget ran out
mid-loop). When tools are registered, `generate` also stops at `<|tool_call_end|>` (kept in the
transcript) so the driver acts without decoding the rest of the turn.

### Example

```rust
use bebelm::agent::Agent;
use bebelm::model::Model;
use bebelm::tool::{Schema, Tool, Type::Int, Type::Str};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::load("LFM2.5-8B-A1B-Q4_K_M.gguf")?;

    // Register tools at build time (before the system block is emitted).
    let mut agent = Agent::new(&model)
        .add_tool(Tool::new(
            "get_weather",
            "Current weather for a city.",
            Schema::new().req("city", Str, "City to look up"),
            |c| {
                let city = c.arg("city").unwrap_or("?");
                format!(r#"{{"city":"{city}","temp_c":18,"sky":"clear"}}"#)
            },
        ))
        .add_tool(Tool::new(
            "add",
            "Add two integers.",
            Schema::new().req("a", Int, "First addend").req("b", Int, "Second addend"),
            |c| {
                // Args arrive as raw text; the callback parses what it needs.
                let a: i64 = c.arg("a").and_then(|s| s.parse().ok()).unwrap_or(0);
                let b: i64 = c.arg("b").and_then(|s| s.parse().ok()).unwrap_or(0);
                (a + b).to_string()
            },
        ));

    agent.append_system("You are a helpful assistant.");
    agent.append_user("What's the weather in Paris, and what is 21 + 21?");

    // Run the agentic loop (≤ 8 tool rounds): stream answer tokens, and log each tool call.
    let turn = agent.assistant_turn_with_tools(
        8,
        |_id, text| print!("{text}"),
        |call, result| eprintln!("[tool] {} -> {result}", call.name),
    );
    println!("\n({} tokens)", turn.stats.generated_tokens);
    Ok(())
}
```

Behind the scenes this lays down one system block carrying both tool schemas
(`List of tools: [{…get_weather…}, {…add…}]`), the user turn, and then — as the model
calls `get_weather(city='Paris')` and `add(a=21, b=21)` — alternating `tool`-role results
and reopened assistant turns, until the model produces a plain-text final answer.

### Design decisions (resolved)

- **String args, no serde** — chosen, matching the dependency ethos: tool schemas are built by
  a hand-rolled JSON emitter and parsed call args are raw strings (the callback parses what it
  needs). `serde_json` was *not* pulled in.
- **Automated loop on `Agent`** — chosen: `assistant_turn_with_tools` runs the tools itself and
  reports each call via the `on_tool` observer, rather than returning control to the caller per
  call. Callers wanting manual dispatch can still drive `generate` + `parse_tool_calls` directly.
- **System/tools appended first** — chosen: `append_system` emits the system block (with the
  `List of tools:` preamble) explicitly, called before any user turn, fitting the append-only
  transcript. Tools must be registered via `add_tool` before `append_system`.
