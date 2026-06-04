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
mkdir -p models
curl -L -o models/LFM2.5-8B-A1B-Q4_K_M.gguf \
  "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/main/LFM2.5-8B-A1B-Q4_K_M.gguf"
```

(Verified live: the `resolve/main` URL returns HTTP 200 and redirects to the HF Xet CDN.)

`Q4_K_M` is a *mixed* quantization: most 2D weight matrices are **Q4_K**, while a few
quality-sensitive tensors (typically the token-embedding/output matrix, the attention
`v` projections, and some `ffn_down`) are stored as **Q6_K**. 1-D tensors (RMSNorm
gains, the MoE router, expert biases) are stored as **F32**. Our loader must therefore
dispatch on the per-tensor dtype recorded in the GGUF header.

> ✅ Done (milestone 1): `bebelm dump` confirmed the file has 256 tensors with the
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
| `rayon` | phase 2 (opt) | data-parallel matmul rows / per-expert / per-head work across cores | None — built on std threads. |

> `half` was **dropped**: the file has no F16/BF16 whole tensors, so we only need f16→f32
> for in-block scales — hand-rolled as a tested ~25-line `f16_to_f32` in
> `kernels/dequant.rs`. Current dependency tree is just `memmap2` (+ `rayon` later).

**Deliberately NOT taking crates for:**

- **CLI args** → `std::env::args` by hand (no `clap`).
- **Sampling RNG** → ~10-line PCG/xorshift (no `rand`).
- **Errors** → `Box<dyn Error>` / small enums (no `anyhow`/`thiserror`).
- **SIMD** (phase 2) → `std::arch` intrinsics with `is_*_feature_detected!` (no crate);
  optionally `wide` for portable pure-Rust SIMD without nightly.
- **Tokenizer** → hand-rolled byte-level BPE on UTF-8 (no `regex`, no `tokenizers`; the
  latter's default `onig` feature is a C library = forbidden system dep).

---

## Optimizations (phased)

1. **Correctness, single-core, scalar.** Match a reference on next-token logits.
2. **KV cache + conv-state cache.** (Listed as a core feature; needed for usable decode.)
3. **Quantized GEMV without full materialization.** Dequantize a weight block, use it
   immediately against the activation, accumulate — keeps the weight in cache, avoids a
   second pass over a dequantized buffer.
4. **Multithreading (`rayon`).** Parallelize over output rows (matmul), experts (MoE),
   and heads (attention).
5. **SIMD (`std::arch`).** Vectorize dequant + dot-products (AVX2/FMA on x86, NEON on
   Apple silicon). Block/tile GEMM for prefill.
6. **MoE sparsity.** Only run the 4 selected experts per token (the whole point of A1B).
7. **f16 KV cache** to cut decode-time memory bandwidth.

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
  model.rs           # ◐ weight loading by name + shape check (forward pass next)
  cache.rs           #   KV cache + conv-state cache
  sampler.rs         #   temperature + top-k (+ rep penalty); temp 0 = greedy; hand-rolled PRNG
  tokenizer.rs       #   (phase 2) byte-level BPE from GGUF metadata
  kernels/
    mod.rs           # ✓
    dequant.rs ✓  matmul.rs ✓  rmsnorm.rs ✓  rope.rs ✓
    softmax.rs ✓  attention.rs  conv.rs ✓  activation.rs ✓  elementwise.rs ✓
```

---

## Implementation milestones

1. ✅ **GGUF loader + tensor dump.** Parse header, list tensors (name/dtype/shape);
   confirmed the Q4_K/Q6_K/F32 mix and the verified tensor-name mapping above.
2. ✅ **Dequant kernels** (`kernels/dequant.rs`): hand-rolled `f16_to_f32`, Q4_K + Q6_K
   block decoders (exact ggml reference) + an F32/F16/Q4_K/Q6_K dispatcher. Unit-tested on
   synthetic blocks and validated on real tensors via `bebelm dequant <file> <tensor>`.
3. ✅ **Quantized GEMV + RMSNorm** (`kernels/matmul.rs`, `kernels/rmsnorm.rs`):
   `matvec(dtype, W, n_in, n_out, x, y)` (dequantize-row-then-dot, reused buffer) and
   `rmsnorm(x, gain, eps, out)`. Unit-tested on hand-computed cases (incl. row-major
   layout check). Also restructured into a lib crate + thin bin (clean dead-code story).
4. ✅ **Config + model loading** (`config.rs`, `model.rs`): architecture as hardcoded
   `const`s + `validate(&gguf)`; `Model::load` resolves all 256 tensors by name and
   checks shapes. `bebelm load` confirmed it against the real file. (We chose a **static
   forward pass** — see note below — rather than runtime config interpretation.)
5. **Remaining kernels + single forward pass** (no cache): `rope`, `softmax`,
   `activation` (SiLU/SwiGLU), `elementwise`, `conv`, `attention`; wire the static layer
   loop (embed → 24 layers → final norm → logits) incl. MoE routing. Validate logits
   against a reference (throwaway HF `transformers` script) on a fixed token sequence.
6. **KV + conv-state caches**; multi-token prefill then incremental decode.
7. **Sampling (temp + top-k, temp 0 = greedy) + generation** on raw token IDs; verify a
   known continuation.
8. **Tokenizer** (byte-level BPE from GGUF) + chat template → end-to-end text.
9. **Optimizations:** sparse-expert execution, rayon, SIMD, f16 KV cache.
  - TODO: break this down into sub-steps
10. **Cleanup:** remove unused code and validation code that is no longer needed.

### Design note: static forward pass

The architecture is fixed (one target model), so it lives in `config.rs` as compile-time
`const`s, not runtime-parsed config. The forward pass is written as plain static code
using those consts (e.g. `const HIDDEN = 2048`, a `const` conv/attn schedule) — **not** a
code generator and **not** a runtime config interpreter. We still load the ~4.8 GB of
weights at runtime (they can't live in the binary) and look them up **by name** (offsets
are file-specific and fragile). `config::validate(&gguf)` asserts the file matches the
constants at startup, so a wrong/updated file fails loudly. Benefits: const dimensions
let the compiler elide bounds checks / use fixed-size buffers, and the forward pass stays
readable and debuggable.
