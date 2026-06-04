//! Model loading + the static forward pass. Opens the GGUF, validates it against the
//! hardcoded [`crate::config`], resolves tensors by name, and runs embed → 24 layers
//! (conv/attn operator + dense/MoE FFN) → final norm → logits.

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::config::{
    self, CONV_L_CACHE, DENSE_FF, HEAD_DIM, HIDDEN, KV_DIM, MOE_FF, N_EXPERTS, N_EXPERTS_USED,
    N_HEADS, N_KV_HEADS, N_LAYERS, RMS_EPS, ROPE_THETA, VOCAB,
};
use crate::cache::Cache;
use crate::gguf::{GgufFile, TensorInfo};
use crate::kernels::activation::{sigmoid_slice, swiglu};
use crate::kernels::attention::attention_decode;
use crate::kernels::conv::conv_step;
use crate::kernels::elementwise::{add_assign, add_scaled};
use crate::kernels::matmul::matvec;
use crate::kernels::rmsnorm::rmsnorm;
use crate::kernels::rope::rope_neox;
use crate::sampler::Sampler;
use crate::tensor::GgmlType;

/// Timing + counts from a generation run.
pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill: Duration,
    pub decode: Duration,
}

impl GenStats {
    /// Prefill throughput (prompt tokens per second).
    pub fn prefill_tps(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill.as_secs_f64().max(f64::MIN_POSITIVE)
    }

    /// Decode throughput (generated tokens per second).
    pub fn decode_tps(&self) -> f64 {
        self.generated_tokens as f64 / self.decode.as_secs_f64().max(f64::MIN_POSITIVE)
    }
}

/// A loaded, validated model: the mmapped GGUF plus a name → tensor index.
pub struct Model {
    gguf: GgufFile,
    by_name: HashMap<String, usize>,
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
        let model = Model { gguf, by_name };
        model.check_tensors()?;
        Ok(model)
    }

    /// Look up a tensor's metadata by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.gguf.tensors[i])
    }

    /// Raw (still-quantized) bytes for a tensor.
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        self.gguf.tensor_data(t)
    }

    /// The underlying GGUF (e.g. to build a [`crate::tokenizer::Tokenizer`] from the same mmap).
    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    // --- forward pass (single-token, cached) ---

    /// Run the prompt through the model and return the logits for the **last** position
    /// (length [`VOCAB`]), using a fresh cache internally. Intermediate tokens only update
    /// the cache (their logits are skipped — 9a).
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let (last, rest) = tokens.split_last().expect("forward: empty token sequence");
        let mut cache = Cache::new();
        for &tok in rest {
            self.run_layers(tok, &mut cache);
        }
        self.forward_step(*last, &mut cache)
    }

    /// Process one token at position `cache.pos`, update the caches, and return its
    /// next-token logits. One pass over the weights, plus O(context) attention.
    pub fn forward_step(&self, token: u32, cache: &mut Cache) -> Vec<f32> {
        let h = self.run_layers(token, cache);
        self.logits_from_hidden(&h)
    }

    /// Embed + run the 24 decoder layers for one token (updating the KV/conv caches and
    /// `cache.pos`), returning the final hidden state — *without* the logits projection.
    /// Used for prefill tokens whose logits aren't needed (9a).
    fn run_layers(&self, token: u32, cache: &mut Cache) -> Vec<f32> {
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

    /// Final RMSNorm + tied logits (`token_embd · h`) for a hidden state.
    fn logits_from_hidden(&self, h: &[f32]) -> Vec<f32> {
        let gain = self.dequant_vec("token_embd_norm.weight");
        let mut normed = vec![0.0f32; HIDDEN];
        rmsnorm(h, &gain, RMS_EPS, &mut normed);
        let tok_embd = self.tensor("token_embd.weight").expect("token_embd");
        let mut logits = vec![0.0f32; VOCAB];
        matvec(tok_embd.ggml_type, self.data(tok_embd), HIDDEN, VOCAB, &normed, &mut logits);
        logits
    }

    /// Autoregressive generation with the KV + conv caches. Stops at `eos` or after
    /// `max_new` tokens. Returns the newly generated token ids (excluding the prompt).
    pub fn generate(&self, prompt: &[u32], sampler: &mut Sampler, max_new: usize, eos: u32) -> Vec<u32> {
        self.generate_with_stats(prompt, sampler, max_new, eos, |_| {}).0
    }

    /// Like [`generate`](Self::generate) but also returns prefill/decode timing and calls
    /// `on_token` with each token as it is produced (for streaming output).
    pub fn generate_with_stats(
        &self,
        prompt: &[u32],
        sampler: &mut Sampler,
        max_new: usize,
        eos: u32,
        mut on_token: impl FnMut(u32),
    ) -> (Vec<u32>, GenStats) {
        let mut cache = Cache::new();
        let mut history = prompt.to_vec();

        // Prefill: only the last prompt token needs logits (9a); the rest just fill caches.
        let t_prefill = Instant::now();
        let (last, rest) = prompt.split_last().expect("generate: empty prompt");
        for &tok in rest {
            self.run_layers(tok, &mut cache);
        }
        let mut logits = self.forward_step(*last, &mut cache);
        let prefill = t_prefill.elapsed();

        // Decode: sample, then compute the next logits (skip that on the final token).
        let t_decode = Instant::now();
        let mut generated = Vec::with_capacity(max_new);
        for i in 0..max_new {
            let next = sampler.sample(&mut logits, &history);
            generated.push(next);
            history.push(next);
            on_token(next);
            if next == eos {
                break;
            }
            if i + 1 < max_new {
                logits = self.forward_step(next, &mut cache);
            }
        }
        let decode = t_decode.elapsed();

        let stats = GenStats {
            prompt_tokens: prompt.len(),
            generated_tokens: generated.len(),
            prefill,
            decode,
        };
        (generated, stats)
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

        let conv_w = self.dequant_vec(&name(layer, "shortconv.conv.weight"));
        let mut conv_out = vec![0.0f32; HIDDEN];
        conv_step(&cache.conv[layer], &bx, &conv_w, HIDDEN, CONV_L_CACHE, &mut conv_out);

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

        let q_gain = self.dequant_vec(&name(layer, "attn_q_norm.weight"));
        let k_gain = self.dequant_vec(&name(layer, "attn_k_norm.weight"));
        norm_rope_heads(&mut q, N_HEADS, &q_gain, pos);
        norm_rope_heads(&mut k, N_KV_HEADS, &k_gain, pos);

        cache.k[layer].extend_from_slice(&k);
        cache.v[layer].extend_from_slice(&v);
        let n_ctx = pos + 1;

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
        let bias = self.dequant_vec(&name(layer, "exp_probs_b.bias"));
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

        let mut out = vec![0.0f32; HIDDEN];
        for (i, &e) in sel.iter().enumerate() {
            let gate_w = &self.data(gate_exps)[e * gate_stride..(e + 1) * gate_stride];
            let up_w = &self.data(up_exps)[e * up_stride..(e + 1) * up_stride];
            let down_w = &self.data(down_exps)[e * down_stride..(e + 1) * down_stride];
            let mut g = vec![0.0f32; MOE_FF];
            let mut u = vec![0.0f32; MOE_FF];
            matvec(gate_exps.ggml_type, gate_w, HIDDEN, MOE_FF, x, &mut g);
            matvec(up_exps.ggml_type, up_w, HIDDEN, MOE_FF, x, &mut u);
            let mut act = vec![0.0f32; MOE_FF];
            swiglu(&g, &u, &mut act);
            let mut down = vec![0.0f32; HIDDEN];
            matvec(down_exps.ggml_type, down_w, MOE_FF, HIDDEN, &act, &mut down);
            add_scaled(&mut out, &down, w[i]);
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
        let gain = self.dequant_vec(gain_name);
        let mut out = vec![0.0f32; HIDDEN];
        rmsnorm(h, &gain, RMS_EPS, &mut out);
        out
    }

    /// Fully dequantize a (usually small, F32) tensor by name into a `Vec<f32>`.
    fn dequant_vec(&self, tensor: &str) -> Vec<f32> {
        let t = self.tensor(tensor).expect("dequant_vec: tensor");
        crate::kernels::dequant::dequantize(t.ggml_type, self.data(t), t.n_elements() as usize)
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
