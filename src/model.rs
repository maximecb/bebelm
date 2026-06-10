//! Model loading + the static forward pass. Opens the GGUF, validates it against the
//! hardcoded [`crate::config`], resolves tensors by name, and runs embed → 24 layers
//! (conv/attn operator + dense/MoE FFN) → final norm → logits.

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

use crate::config::{
    self, CONV_L_CACHE, DENSE_FF, HEAD_DIM, HIDDEN, KV_DIM, MOE_FF, N_EXPERTS, N_EXPERTS_USED,
    N_HEADS, N_KV_HEADS, N_LAYERS, RMS_EPS, ROPE_THETA, VOCAB,
};
use crate::cache::Cache;
use crate::gguf::{GgufFile, TensorInfo};
use crate::kernels::activation::{sigmoid_slice, swiglu};
use crate::kernels::attention::attention_decode;
use crate::kernels::conv::{conv_prefill, conv_step};
use crate::kernels::elementwise::{add_assign, add_scaled};
use crate::kernels::matmul::{matmul, matvec, matvec_fused_batch, quantize_q8, FusedJob, Q8Vec};
use crate::kernels::rmsnorm::rmsnorm;
use crate::kernels::rope::rope_neox;
use crate::tensor::GgmlType;
use crate::tokenizer::Tokenizer;

/// A loaded, validated model: the mmapped GGUF plus a name → tensor index, plus the small
/// F32 tensors (norm gains, conv filters, expert biases) pre-dequantized once (9b).
pub struct Model {
    gguf: GgufFile,
    by_name: HashMap<String, usize>,
    f32_cache: HashMap<String, Vec<f32>>,
    tokenizer: Tokenizer,
}

impl Model {
    /// Open, validate config, and check that all expected tensors are present and shaped.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Model, Box<dyn Error>> {
        let gguf = GgufFile::open(path)?;
        config::validate(&gguf)?;
        let by_name = gguf
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let mut model = Model { gguf, by_name, f32_cache: HashMap::new(), tokenizer };
        model.check_tensors()?;
        model.precompute_f32();
        Ok(model)
    }

    /// Pre-dequantize the small F32 tensors used per-token (norms, q/k norms, conv filters,
    /// expert biases, final norm). The F32 router (`ffn_gate_inp`) is excluded — it goes
    /// through `matvec` on its raw bytes, not [`f32`](Self::f32).
    fn precompute_f32(&mut self) {
        let mut cache = HashMap::new();
        for t in &self.gguf.tensors {
            if t.ggml_type == GgmlType::F32 && !t.name.ends_with("ffn_gate_inp.weight") {
                let v = crate::kernels::dequant::dequantize(
                    t.ggml_type,
                    self.gguf.tensor_data(t),
                    t.n_elements() as usize,
                );
                cache.insert(t.name.clone(), v);
            }
        }
        self.f32_cache = cache;
    }

    /// A pre-dequantized F32 tensor by name (see [`precompute_f32`](Self::precompute_f32)).
    fn f32(&self, name: &str) -> &[f32] {
        self.f32_cache
            .get(name)
            .unwrap_or_else(|| panic!("f32 tensor not precomputed: {name}"))
            .as_slice()
    }

    /// Look up a tensor's metadata by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.gguf.tensors[i])
    }

    /// Raw (still-quantized) bytes for a tensor.
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        self.gguf.tensor_data(t)
    }

    /// The model's tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    // --- forward pass (single-token, cached) ---

    /// Process one token at position `cache.pos`, update the caches, and return its
    /// next-token logits. One pass over the weights, plus O(context) attention.
    pub fn forward_step(&self, token: u32, cache: &mut Cache) -> Vec<f32> {
        let h = self.run_layers(token, cache);
        self.logits_from_hidden(&h)
    }

    /// Embed + run the 24 decoder layers for one token (updating the KV/conv caches and
    /// `cache.pos`), returning the final hidden state — *without* the logits projection.
    /// Used for prefill tokens whose logits aren't needed (9a).
    pub(crate) fn run_layers(&self, token: u32, cache: &mut Cache) -> Vec<f32> {
        let pos = cache.pos;
        let mut h = vec![0.0f32; HIDDEN];
        self.embed_token(token, &mut h);

        for layer in 0..N_LAYERS {
            let normed = self.norm_one(&h, &name(layer, "attn_norm.weight"));
            let op = if config::is_attention(layer) {
                self.attn_step(layer, &normed, pos, cache)
            } else {
                self.conv_step_op(layer, &normed, cache)
            };
            add_assign(&mut h, &op);

            let normed = self.norm_one(&h, &name(layer, "ffn_norm.weight"));
            let ffn = if config::is_dense_ffn(layer) {
                self.dense_ffn(layer, &normed)
            } else {
                self.moe_ffn(layer, &normed)
            };
            add_assign(&mut h, &ffn);
        }

        cache.pos += 1;
        h
    }

    // --- forward pass (batched prefill) ---

    /// Embed + run the 24 decoder layers for a **batch** of `tokens` (positions
    /// `cache.pos .. cache.pos + tokens.len()`), updating the KV/conv caches and `cache.pos`,
    /// and returning the batch's final hidden states (token-major, `tokens.len() × HIDDEN`).
    ///
    /// This is the batched analogue of [`run_layers`](Self::run_layers): the weight projections
    /// run once over all tokens via [`matmul`] (reading each weight row once instead of once per
    /// token — opt 9f), while the cheap per-token ops (norm, RoPE, SDPA, conv, SwiGLU, MoE
    /// routing) loop over the batch reusing the same kernels as decode. Every output is computed
    /// by the identical arithmetic the per-token path uses, so the caches it leaves — and hence
    /// the subsequent decode — are **bit-for-bit identical** to prefilling token by token.
    ///
    /// The caller must ensure the batch does not overflow the KV window (no mid-batch sliding);
    /// [`crate::agent::Agent`] only batches when the whole prompt fits, falling back otherwise.
    pub(crate) fn run_layers_batch(&self, tokens: &[u32], cache: &mut Cache) -> Vec<f32> {
        let t = tokens.len();
        let pos0 = cache.pos;
        let mut h = vec![0.0f32; t * HIDDEN];
        for (i, &tok) in tokens.iter().enumerate() {
            self.embed_token(tok, &mut h[i * HIDDEN..(i + 1) * HIDDEN]);
        }

        for layer in 0..N_LAYERS {
            let normed = self.norm_batch(&h, &name(layer, "attn_norm.weight"), t);
            let op = if config::is_attention(layer) {
                self.attn_batch(layer, &normed, pos0, t, cache)
            } else {
                self.conv_batch(layer, &normed, t, cache)
            };
            add_assign(&mut h, &op);

            let normed = self.norm_batch(&h, &name(layer, "ffn_norm.weight"), t);
            let ffn = if config::is_dense_ffn(layer) {
                self.dense_ffn_batch(layer, &normed, t)
            } else {
                self.moe_ffn_batch(layer, &normed, t)
            };
            add_assign(&mut h, &ffn);
        }

        cache.pos += t;
        h
    }

    /// RMSNorm each of the `t` token rows of `h` (token-major `t × HIDDEN`) with the named gain.
    fn norm_batch(&self, h: &[f32], gain_name: &str, t: usize) -> Vec<f32> {
        let gain = self.f32(gain_name);
        let mut out = vec![0.0f32; t * HIDDEN];
        for (hi, oi) in h.chunks_exact(HIDDEN).zip(out.chunks_exact_mut(HIDDEN)) {
            rmsnorm(hi, gain, RMS_EPS, oi);
        }
        out
    }

    /// Batched gated short-conv operator: in/out projections run over all `t` tokens via
    /// [`matmul`]; the depthwise conv runs over the batch's positions via [`conv_prefill`],
    /// consuming and updating the conv-state cache.
    fn conv_batch(&self, layer: usize, x: &[f32], t: usize, cache: &mut Cache) -> Vec<f32> {
        let mut bcx = vec![0.0f32; t * 3 * HIDDEN];
        self.matmul1(&name(layer, "shortconv.in_proj.weight"), x, HIDDEN, 3 * HIDDEN, t, &mut bcx);

        // Bx = B · x_gate (chunks 0 and 2 of in_proj) per token.
        let mut bx = vec![0.0f32; t * HIDDEN];
        for i in 0..t {
            let base = i * 3 * HIDDEN;
            for (c, bxc) in bx[i * HIDDEN..(i + 1) * HIDDEN].iter_mut().enumerate() {
                *bxc = bcx[base + c] * bcx[base + 2 * HIDDEN + c];
            }
        }

        let conv_w = self.f32(&name(layer, "shortconv.conv.weight"));
        let mut conv_out = vec![0.0f32; t * HIDDEN];
        conv_prefill(&mut cache.conv[layer], &bx, conv_w, HIDDEN, CONV_L_CACHE, t, &mut conv_out);

        // y = C · conv_out (chunk 1) per token, then out_proj.
        let mut y = vec![0.0f32; t * HIDDEN];
        for i in 0..t {
            let base = i * 3 * HIDDEN;
            for (c, yc) in y[i * HIDDEN..(i + 1) * HIDDEN].iter_mut().enumerate() {
                *yc = bcx[base + HIDDEN + c] * conv_out[i * HIDDEN + c];
            }
        }
        let mut out = vec![0.0f32; t * HIDDEN];
        self.matmul1(&name(layer, "shortconv.out_proj.weight"), &y, HIDDEN, HIDDEN, t, &mut out);
        out
    }

    /// Batched GQA attention operator: q/k/v/o projections run over all `t` tokens via [`matmul`];
    /// each token is then q/k-normed + RoPE'd at its own position, appended to the KV cache, and
    /// attended over the cache prefix it can see (causal by construction — token `i` attends to
    /// the `pos0 + i + 1` keys present once its own k/v are appended).
    fn attn_batch(&self, layer: usize, x: &[f32], pos0: usize, t: usize, cache: &mut Cache) -> Vec<f32> {
        let mut q = vec![0.0f32; t * HIDDEN];
        let mut k = vec![0.0f32; t * KV_DIM];
        let mut v = vec![0.0f32; t * KV_DIM];
        self.matmul1(&name(layer, "attn_q.weight"), x, HIDDEN, HIDDEN, t, &mut q);
        self.matmul1(&name(layer, "attn_k.weight"), x, HIDDEN, KV_DIM, t, &mut k);
        self.matmul1(&name(layer, "attn_v.weight"), x, HIDDEN, KV_DIM, t, &mut v);

        let q_gain = self.f32(&name(layer, "attn_q_norm.weight"));
        let k_gain = self.f32(&name(layer, "attn_k_norm.weight"));
        for i in 0..t {
            norm_rope_heads(&mut q[i * HIDDEN..(i + 1) * HIDDEN], N_HEADS, q_gain, pos0 + i);
            norm_rope_heads(&mut k[i * KV_DIM..(i + 1) * KV_DIM], N_KV_HEADS, k_gain, pos0 + i);
        }

        let mut attn = vec![0.0f32; t * HIDDEN];
        for i in 0..t {
            cache.k[layer].extend_from_slice(&k[i * KV_DIM..(i + 1) * KV_DIM]);
            cache.v[layer].extend_from_slice(&v[i * KV_DIM..(i + 1) * KV_DIM]);
            let n_ctx = cache.k[layer].len() / KV_DIM;
            attention_decode(
                &q[i * HIDDEN..(i + 1) * HIDDEN],
                &cache.k[layer],
                &cache.v[layer],
                n_ctx,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                &mut attn[i * HIDDEN..(i + 1) * HIDDEN],
            );
        }

        let mut out = vec![0.0f32; t * HIDDEN];
        self.matmul1(&name(layer, "attn_output.weight"), &attn, HIDDEN, HIDDEN, t, &mut out);
        out
    }

    /// Batched dense SwiGLU MLP (layers 0,1): gate/up/down projections run over all `t` tokens.
    fn dense_ffn_batch(&self, layer: usize, x: &[f32], t: usize) -> Vec<f32> {
        let mut gate = vec![0.0f32; t * DENSE_FF];
        let mut up = vec![0.0f32; t * DENSE_FF];
        self.matmul1(&name(layer, "ffn_gate.weight"), x, HIDDEN, DENSE_FF, t, &mut gate);
        self.matmul1(&name(layer, "ffn_up.weight"), x, HIDDEN, DENSE_FF, t, &mut up);
        let mut act = vec![0.0f32; t * DENSE_FF];
        for ((g, u), a) in gate
            .chunks_exact(DENSE_FF)
            .zip(up.chunks_exact(DENSE_FF))
            .zip(act.chunks_exact_mut(DENSE_FF))
        {
            swiglu(g, u, a);
        }
        let mut out = vec![0.0f32; t * HIDDEN];
        self.matmul1(&name(layer, "ffn_down.weight"), &act, DENSE_FF, HIDDEN, t, &mut out);
        out
    }

    /// Batched MoE FFN (token-grouped, opt 9f). Routing is per token, but each expert's
    /// gate/up/down then run **once over all the tokens that selected it** via [`matmul`] —
    /// reading the expert's weights once per batch instead of once per (token, expert). This is
    /// **bit-for-bit identical** to running [`moe_ffn`](Self::moe_ffn) per token: every expert
    /// output equals the per-token matvec (proven for [`matmul`]), and each token's experts are
    /// summed back in the same selection order with the same weights.
    fn moe_ffn_batch(&self, layer: usize, x: &[f32], t: usize) -> Vec<f32> {
        let router = self.tensor(&name(layer, "ffn_gate_inp.weight")).expect("router");
        let bias = self.f32(&name(layer, "exp_probs_b.bias"));
        let gate_exps = self.tensor(&name(layer, "ffn_gate_exps.weight")).expect("gate_exps");
        let up_exps = self.tensor(&name(layer, "ffn_up_exps.weight")).expect("up_exps");
        let down_exps = self.tensor(&name(layer, "ffn_down_exps.weight")).expect("down_exps");
        let gate_stride = expert_bytes(gate_exps.ggml_type, HIDDEN, MOE_FF);
        let up_stride = expert_bytes(up_exps.ggml_type, HIDDEN, MOE_FF);
        let down_stride = expert_bytes(down_exps.ggml_type, MOE_FF, HIDDEN);

        // Route every token (batched router), then per token: sigmoid → top-k by (score+bias) →
        // normalize the selected sigmoid scores. `sel`/`wts` are token-major `t × N_EXPERTS_USED`,
        // in score-descending (selection) order — exactly as `moe_ffn` computes them.
        let mut scores = vec![0.0f32; t * N_EXPERTS];
        matmul(router.ggml_type, self.data(router), HIDDEN, N_EXPERTS, x, t, &mut scores);
        let mut sel = vec![0usize; t * N_EXPERTS_USED];
        let mut wts = vec![0.0f32; t * N_EXPERTS_USED];
        for i in 0..t {
            let sc = &mut scores[i * N_EXPERTS..(i + 1) * N_EXPERTS];
            sigmoid_slice(sc);
            let mut order: Vec<usize> = (0..N_EXPERTS).collect();
            order.sort_unstable_by(|&a, &b| (sc[b] + bias[b]).total_cmp(&(sc[a] + bias[a])));
            let chosen = &order[..N_EXPERTS_USED];
            let denom: f32 = chosen.iter().map(|&e| sc[e]).sum::<f32>() + 1e-6;
            for (k, &e) in chosen.iter().enumerate() {
                sel[i * N_EXPERTS_USED + k] = e;
                wts[i * N_EXPERTS_USED + k] = sc[e] / denom;
            }
        }

        // Group each (token, slot) by the expert it selected.
        let mut groups: Vec<Vec<(usize, usize)>> = vec![Vec::new(); N_EXPERTS];
        for i in 0..t {
            for k in 0..N_EXPERTS_USED {
                groups[sel[i * N_EXPERTS_USED + k]].push((i, k));
            }
        }

        // Per expert with members: gather its tokens' inputs, run gate+up → SwiGLU → down over the
        // whole group at once, and scatter each result into its (token, slot) row.
        let gate_data = self.data(gate_exps);
        let up_data = self.data(up_exps);
        let down_data = self.data(down_exps);
        let mut slots = vec![0.0f32; t * N_EXPERTS_USED * HIDDEN]; // (token, slot) → down output
        for (e, members) in groups.iter().enumerate() {
            if members.is_empty() {
                continue;
            }
            let g = members.len();
            let mut xe = vec![0.0f32; g * HIDDEN];
            for (j, &(i, _)) in members.iter().enumerate() {
                xe[j * HIDDEN..(j + 1) * HIDDEN].copy_from_slice(&x[i * HIDDEN..(i + 1) * HIDDEN]);
            }

            let mut gate = vec![0.0f32; g * MOE_FF];
            let mut up = vec![0.0f32; g * MOE_FF];
            matmul(gate_exps.ggml_type, &gate_data[e * gate_stride..(e + 1) * gate_stride], HIDDEN, MOE_FF, &xe, g, &mut gate);
            matmul(up_exps.ggml_type, &up_data[e * up_stride..(e + 1) * up_stride], HIDDEN, MOE_FF, &xe, g, &mut up);
            let mut act = vec![0.0f32; g * MOE_FF];
            for ((gx, ux), ax) in gate
                .chunks_exact(MOE_FF)
                .zip(up.chunks_exact(MOE_FF))
                .zip(act.chunks_exact_mut(MOE_FF))
            {
                swiglu(gx, ux, ax);
            }
            let mut down = vec![0.0f32; g * HIDDEN];
            matmul(down_exps.ggml_type, &down_data[e * down_stride..(e + 1) * down_stride], MOE_FF, HIDDEN, &act, g, &mut down);

            for (j, &(i, k)) in members.iter().enumerate() {
                let slot = i * N_EXPERTS_USED + k;
                slots[slot * HIDDEN..(slot + 1) * HIDDEN].copy_from_slice(&down[j * HIDDEN..(j + 1) * HIDDEN]);
            }
        }

        // Weighted sum per token over its selected experts, in selection order (as `moe_ffn`).
        let mut out = vec![0.0f32; t * HIDDEN];
        for i in 0..t {
            let oi = &mut out[i * HIDDEN..(i + 1) * HIDDEN];
            for k in 0..N_EXPERTS_USED {
                let slot = i * N_EXPERTS_USED + k;
                add_scaled(oi, &slots[slot * HIDDEN..(slot + 1) * HIDDEN], wts[i * N_EXPERTS_USED + k]);
            }
        }
        out
    }

    /// Batched [`matvec1`](Self::matvec1): `Y = W·X` over `n_tokens` token-major columns.
    fn matmul1(&self, tensor: &str, x: &[f32], n_in: usize, n_out: usize, n_tokens: usize, out: &mut [f32]) {
        let tt = self.tensor(tensor).expect("matmul1: tensor");
        matmul(tt.ggml_type, self.data(tt), n_in, n_out, x, n_tokens, out);
    }

    /// Final RMSNorm + tied logits (`token_embd · h`) for a hidden state.
    fn logits_from_hidden(&self, h: &[f32]) -> Vec<f32> {
        let gain = self.f32("token_embd_norm.weight");
        let mut normed = vec![0.0f32; HIDDEN];
        rmsnorm(h, gain, RMS_EPS, &mut normed);
        let tok_embd = self.tensor("token_embd.weight").expect("token_embd");
        let mut logits = vec![0.0f32; VOCAB];
        matvec(tok_embd.ggml_type, self.data(tok_embd), HIDDEN, VOCAB, &normed, &mut logits);
        logits
    }

    /// Embedding lookup: dequantize row `token` of token_embd into `out` (`HIDDEN`).
    fn embed_token(&self, token: u32, out: &mut [f32]) {
        let te = self.tensor("token_embd.weight").expect("token_embd");
        let (blk_elems, blk_bytes) = te.ggml_type.block().expect("embd block");
        let row_bytes = (HIDDEN / blk_elems as usize) * blk_bytes as usize;
        let off = token as usize * row_bytes;
        crate::kernels::dequant::dequantize_into(te.ggml_type, &self.data(te)[off..off + row_bytes], out);
    }

    /// Gated short-conv operator for one token, using + updating the conv-state cache.
    fn conv_step_op(&self, layer: usize, x: &[f32], cache: &mut Cache) -> Vec<f32> {
        let mut bcx = vec![0.0f32; 3 * HIDDEN];
        self.matvec1(&name(layer, "shortconv.in_proj.weight"), x, HIDDEN, 3 * HIDDEN, &mut bcx);

        // Bx = B · x_gate (chunks 0 and 2 of the in_proj output).
        let mut bx = vec![0.0f32; HIDDEN];
        for (c, bxc) in bx.iter_mut().enumerate() {
            *bxc = bcx[c] * bcx[2 * HIDDEN + c];
        }

        let conv_w = self.f32(&name(layer, "shortconv.conv.weight"));
        let mut conv_out = vec![0.0f32; HIDDEN];
        conv_step(&cache.conv[layer], &bx, conv_w, HIDDEN, CONV_L_CACHE, &mut conv_out);

        // Shift the conv state left one column and append the current Bx.
        let state = &mut cache.conv[layer];
        state.copy_within(HIDDEN.., 0);
        state[(CONV_L_CACHE - 2) * HIDDEN..].copy_from_slice(&bx);

        // y = C · conv_out (chunk 1), then out_proj.
        let mut y = vec![0.0f32; HIDDEN];
        for (c, yc) in y.iter_mut().enumerate() {
            *yc = bcx[HIDDEN + c] * conv_out[c];
        }
        let mut out = vec![0.0f32; HIDDEN];
        self.matvec1(&name(layer, "shortconv.out_proj.weight"), &y, HIDDEN, HIDDEN, &mut out);
        out
    }

    /// GQA attention operator for one token, appending to + reading the KV cache.
    fn attn_step(&self, layer: usize, x: &[f32], pos: usize, cache: &mut Cache) -> Vec<f32> {
        let mut q = vec![0.0f32; HIDDEN];
        let mut k = vec![0.0f32; KV_DIM];
        let mut v = vec![0.0f32; KV_DIM];
        self.matvec1(&name(layer, "attn_q.weight"), x, HIDDEN, HIDDEN, &mut q);
        self.matvec1(&name(layer, "attn_k.weight"), x, HIDDEN, KV_DIM, &mut k);
        self.matvec1(&name(layer, "attn_v.weight"), x, HIDDEN, KV_DIM, &mut v);

        let q_gain = self.f32(&name(layer, "attn_q_norm.weight"));
        let k_gain = self.f32(&name(layer, "attn_k_norm.weight"));
        norm_rope_heads(&mut q, N_HEADS, q_gain, pos);
        norm_rope_heads(&mut k, N_KV_HEADS, k_gain, pos);

        cache.k[layer].extend_from_slice(&k);
        cache.v[layer].extend_from_slice(&v);
        // Attend over whatever is in the KV window — normally `pos + 1`, fewer once the window
        // has been slid (old positions evicted). RoPE keys keep their original-position rotation,
        // so the query↔key offsets stay correct.
        let n_ctx = cache.k[layer].len() / KV_DIM;

        let mut attn = vec![0.0f32; HIDDEN];
        attention_decode(&q, &cache.k[layer], &cache.v[layer], n_ctx, N_HEADS, N_KV_HEADS, HEAD_DIM, &mut attn);

        let mut out = vec![0.0f32; HIDDEN];
        self.matvec1(&name(layer, "attn_output.weight"), &attn, HIDDEN, HIDDEN, &mut out);
        out
    }

    /// Dense SwiGLU MLP (layers 0,1) for one token.
    fn dense_ffn(&self, layer: usize, x: &[f32]) -> Vec<f32> {
        let mut gate = vec![0.0f32; DENSE_FF];
        let mut up = vec![0.0f32; DENSE_FF];
        self.matvec1(&name(layer, "ffn_gate.weight"), x, HIDDEN, DENSE_FF, &mut gate);
        self.matvec1(&name(layer, "ffn_up.weight"), x, HIDDEN, DENSE_FF, &mut up);
        let mut act = vec![0.0f32; DENSE_FF];
        swiglu(&gate, &up, &mut act);
        let mut out = vec![0.0f32; HIDDEN];
        self.matvec1(&name(layer, "ffn_down.weight"), &act, DENSE_FF, HIDDEN, &mut out);
        out
    }

    /// Sparse MoE FFN for one token: sigmoid router, top-4 by (score+bias), normalize the
    /// selected **sigmoid** scores, weighted sum of the 4 experts' SwiGLU MLPs.
    fn moe_ffn(&self, layer: usize, x: &[f32]) -> Vec<f32> {
        let router = self.tensor(&name(layer, "ffn_gate_inp.weight")).expect("router");
        let bias = self.f32(&name(layer, "exp_probs_b.bias"));
        let gate_exps = self.tensor(&name(layer, "ffn_gate_exps.weight")).expect("gate_exps");
        let up_exps = self.tensor(&name(layer, "ffn_up_exps.weight")).expect("up_exps");
        let down_exps = self.tensor(&name(layer, "ffn_down_exps.weight")).expect("down_exps");
        let gate_stride = expert_bytes(gate_exps.ggml_type, HIDDEN, MOE_FF);
        let up_stride = expert_bytes(up_exps.ggml_type, HIDDEN, MOE_FF);
        let down_stride = expert_bytes(down_exps.ggml_type, MOE_FF, HIDDEN);

        let mut scores = vec![0.0f32; N_EXPERTS];
        matvec(router.ggml_type, self.data(router), HIDDEN, N_EXPERTS, x, &mut scores);
        sigmoid_slice(&mut scores);
        let mut order: Vec<usize> = (0..N_EXPERTS).collect();
        order.sort_unstable_by(|&a, &b| (scores[b] + bias[b]).total_cmp(&(scores[a] + bias[a])));
        let sel = &order[..N_EXPERTS_USED];

        // Weights are the (bias-free) sigmoid scores of the selected experts, normalized.
        let mut w: Vec<f32> = sel.iter().map(|&e| scores[e]).collect();
        let denom: f32 = w.iter().sum::<f32>() + 1e-6;
        for wi in w.iter_mut() {
            *wi /= denom;
        }

        // Gate + up for all selected experts in one parallel region (2·N_EXPERTS_USED fused
        // matvecs sharing input `x`): gate rows first, then up rows. Each expert's matrix is
        // small, so pooling their rows under a single fork/join keeps every core busy. All
        // gate/up experts share `x`, so quantize it to Q8 once for their integer dot.
        let x_q8 = quantize_q8(x);
        let gate_data = self.data(gate_exps);
        let up_data = self.data(up_exps);
        let mut jobs = Vec::with_capacity(2 * N_EXPERTS_USED);
        for &e in sel {
            let w = &gate_data[e * gate_stride..(e + 1) * gate_stride];
            let qx = is_q8_int(gate_exps.ggml_type).then_some(&x_q8);
            jobs.push(FusedJob { dtype: gate_exps.ggml_type, w, n_in: HIDDEN, n_out: MOE_FF, x, qx });
        }
        for &e in sel {
            let w = &up_data[e * up_stride..(e + 1) * up_stride];
            let qx = is_q8_int(up_exps.ggml_type).then_some(&x_q8);
            jobs.push(FusedJob { dtype: up_exps.ggml_type, w, n_in: HIDDEN, n_out: MOE_FF, x, qx });
        }
        let mut gate_up = vec![0.0f32; 2 * N_EXPERTS_USED * MOE_FF];
        matvec_fused_batch(&jobs, &mut gate_up);
        let (gate_all, up_all) = gate_up.split_at(N_EXPERTS_USED * MOE_FF);

        // SwiGLU each expert's gate/up into its activation row.
        let mut act_all = vec![0.0f32; N_EXPERTS_USED * MOE_FF];
        for i in 0..N_EXPERTS_USED {
            let r = i * MOE_FF..(i + 1) * MOE_FF;
            swiglu(&gate_all[r.clone()], &up_all[r.clone()], &mut act_all[r]);
        }

        // Down projection for all experts in one parallel region (each consumes its own
        // activation row), then weighted-sum the results in selection order (as before). Each
        // expert has a distinct activation, so quantize them one-per-expert (when down is Q4_K).
        let down_data = self.data(down_exps);
        let down_q8: Vec<Q8Vec> = if is_q8_int(down_exps.ggml_type) {
            (0..N_EXPERTS_USED).map(|i| quantize_q8(&act_all[i * MOE_FF..(i + 1) * MOE_FF])).collect()
        } else {
            Vec::new()
        };
        let down_jobs: Vec<FusedJob> = sel
            .iter()
            .enumerate()
            .map(|(i, &e)| FusedJob {
                dtype: down_exps.ggml_type,
                w: &down_data[e * down_stride..(e + 1) * down_stride],
                n_in: MOE_FF,
                n_out: HIDDEN,
                x: &act_all[i * MOE_FF..(i + 1) * MOE_FF],
                qx: down_q8.get(i),
            })
            .collect();
        let mut down_all = vec![0.0f32; N_EXPERTS_USED * HIDDEN];
        matvec_fused_batch(&down_jobs, &mut down_all);

        let mut out = vec![0.0f32; HIDDEN];
        for i in 0..N_EXPERTS_USED {
            add_scaled(&mut out, &down_all[i * HIDDEN..(i + 1) * HIDDEN], w[i]);
        }
        out
    }

    /// Single-vector `matvec` against a named weight.
    fn matvec1(&self, tensor: &str, x: &[f32], n_in: usize, n_out: usize, out: &mut [f32]) {
        let t = self.tensor(tensor).expect("matvec1: tensor");
        matvec(t.ggml_type, self.data(t), n_in, n_out, x, out);
    }

    /// RMSNorm one `HIDDEN` vector with the named F32 gain.
    fn norm_one(&self, h: &[f32], gain_name: &str) -> Vec<f32> {
        let gain = self.f32(gain_name);
        let mut out = vec![0.0f32; HIDDEN];
        rmsnorm(h, gain, RMS_EPS, &mut out);
        out
    }

    /// Verify every tensor the forward pass will need exists with the expected shape.
    fn check_tensors(&self) -> Result<(), Box<dyn Error>> {
        for (name, shape) in expected_tensors() {
            let t = self
                .tensor(&name)
                .ok_or_else(|| format!("missing tensor {name}"))?;
            if t.dims != shape {
                return Err(
                    format!("tensor {name}: shape {:?} != expected {shape:?}", t.dims).into(),
                );
            }
        }
        Ok(())
    }
}

/// `"blk.{layer}.{suffix}"` — a per-layer tensor name.
fn name(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}

/// Whether a weight dtype takes the Q8-activation integer dot (so a shared `x` should be
/// pre-quantized once and passed as [`FusedJob::qx`]). The K-quants do; F32/F16 don't.
fn is_q8_int(dtype: GgmlType) -> bool {
    matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K)
}

/// Per-head RMSNorm (over head_dim) then NEOX RoPE, in place over a packed `n_heads ×
/// head_dim` buffer for one position.
fn norm_rope_heads(buf: &mut [f32], n_heads: usize, gain: &[f32], pos: usize) {
    let mut tmp = [0.0f32; HEAD_DIM];
    for hh in 0..n_heads {
        let head = &mut buf[hh * HEAD_DIM..(hh + 1) * HEAD_DIM];
        rmsnorm(head, gain, RMS_EPS, &mut tmp);
        head.copy_from_slice(&tmp);
        rope_neox(head, pos, ROPE_THETA);
    }
}

/// Byte size of one expert's `[n_in, n_out]` weight matrix within a stacked
/// `[n_in, n_out, n_experts]` tensor of the given dtype.
fn expert_bytes(dtype: GgmlType, n_in: usize, n_out: usize) -> usize {
    let (blk_elems, blk_bytes) = dtype.block().expect("expert dtype has a block size");
    n_out * (n_in / blk_elems as usize) * blk_bytes as usize
}

/// The full list of `(name, shape)` the forward pass depends on, derived from the
/// hardcoded schedule. GGUF dims are `[in, out]` for a `y = W·x` weight.
pub fn expected_tensors() -> Vec<(String, Vec<u64>)> {
    use config::*;
    let h = HIDDEN as u64;
    let mut v: Vec<(String, Vec<u64>)> = vec![
        ("token_embd.weight".into(), vec![h, VOCAB as u64]),
        ("token_embd_norm.weight".into(), vec![h]),
    ];
    for i in 0..N_LAYERS {
        let p = format!("blk.{i}");
        v.push((format!("{p}.attn_norm.weight"), vec![h]));
        v.push((format!("{p}.ffn_norm.weight"), vec![h]));

        if is_attention(i) {
            let kv = KV_DIM as u64;
            v.push((format!("{p}.attn_q.weight"), vec![h, h]));
            v.push((format!("{p}.attn_k.weight"), vec![h, kv]));
            v.push((format!("{p}.attn_v.weight"), vec![h, kv]));
            v.push((format!("{p}.attn_output.weight"), vec![h, h]));
            v.push((format!("{p}.attn_q_norm.weight"), vec![HEAD_DIM as u64]));
            v.push((format!("{p}.attn_k_norm.weight"), vec![HEAD_DIM as u64]));
        } else {
            v.push((format!("{p}.shortconv.in_proj.weight"), vec![h, 3 * h]));
            v.push((format!("{p}.shortconv.conv.weight"), vec![CONV_L_CACHE as u64, h]));
            v.push((format!("{p}.shortconv.out_proj.weight"), vec![h, h]));
        }

        if is_dense_ffn(i) {
            v.push((format!("{p}.ffn_gate.weight"), vec![h, DENSE_FF as u64]));
            v.push((format!("{p}.ffn_up.weight"), vec![h, DENSE_FF as u64]));
            v.push((format!("{p}.ffn_down.weight"), vec![DENSE_FF as u64, h]));
        } else {
            let ff = MOE_FF as u64;
            let e = N_EXPERTS as u64;
            v.push((format!("{p}.ffn_gate_inp.weight"), vec![h, e]));
            v.push((format!("{p}.exp_probs_b.bias"), vec![e]));
            v.push((format!("{p}.ffn_gate_exps.weight"), vec![h, ff, e]));
            v.push((format!("{p}.ffn_up_exps.weight"), vec![h, ff, e]));
            v.push((format!("{p}.ffn_down_exps.weight"), vec![ff, h, e]));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::Cache;

    /// Batched prefill must be **bit-for-bit identical** to prefilling token by token: same KV /
    /// conv caches and same final logits. This is the core correctness guarantee of opt 9f, so
    /// the comparison is exact (`assert_eq` on f32), not tolerance-based. Also checks that
    /// chunking the batch (the [`crate::agent`] does this) changes nothing. Loads the full GGUF,
    /// so it is `#[ignore]`d like the other end-to-end tests.
    #[test]
    #[ignore = "loads the full ~5.2 GB GGUF; run with `cargo test --release -- --ignored`"]
    fn batched_prefill_matches_per_token() {
        let path = std::env::var("BEBELM_WEIGHTS_FILE")
            .unwrap_or_else(|_| "./LFM2.5-8B-A1B-Q4_K_M.gguf".to_string());
        let model = Model::load(&path).expect("load weights");
        // A multi-token prompt that exercises conv + both attention and several MoE layers.
        let ids = model
            .tokenizer()
            .encode("The capital of France is Paris, a historic city of light and art.", true);
        assert!(ids.len() >= 6, "need a few tokens to prefill, got {}", ids.len());
        let (&last, rest) = ids.split_last().unwrap();

        // Reference: per-token prefill.
        let mut ca = Cache::new();
        for &tok in rest {
            model.run_layers(tok, &mut ca);
        }
        let logits_a = model.forward_step(last, &mut ca);

        // One-shot batched, and chunked batched (chunk 3 < rest.len()) — both must match exactly.
        for chunk in [rest.len(), 3] {
            let mut cb = Cache::new();
            for part in rest.chunks(chunk) {
                model.run_layers_batch(part, &mut cb);
            }
            let logits_b = model.forward_step(last, &mut cb);

            assert_eq!(ca.pos, cb.pos, "pos (chunk {chunk})");
            for l in 0..N_LAYERS {
                assert_eq!(ca.k[l], cb.k[l], "k cache layer {l} (chunk {chunk})");
                assert_eq!(ca.v[l], cb.v[l], "v cache layer {l} (chunk {chunk})");
                assert_eq!(ca.conv[l], cb.conv[l], "conv cache layer {l} (chunk {chunk})");
            }
            assert_eq!(logits_a, logits_b, "final logits (chunk {chunk})");
        }
    }

    #[test]
    fn expected_tensor_count_matches_file() {
        // The real Q4_K_M file has exactly 256 tensors; our derived list must match.
        assert_eq!(expected_tensors().len(), 256);
    }

    #[test]
    fn expected_tensors_have_unique_names() {
        let mut names: Vec<&String> = Vec::new();
        let list = expected_tensors();
        for (n, _) in &list {
            names.push(n);
        }
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate tensor names generated");
    }
}
