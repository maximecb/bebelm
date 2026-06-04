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
use crate::gguf::{GgufFile, TensorInfo};
use crate::kernels::activation::{sigmoid_slice, swiglu};
use crate::kernels::attention::attention;
use crate::kernels::conv::causal_depthwise_conv;
use crate::kernels::elementwise::{add_assign, add_scaled};
use crate::kernels::matmul::matvec;
use crate::kernels::rmsnorm::rmsnorm;
use crate::kernels::rope::rope_neox;
use crate::sampler::Sampler;
use crate::tensor::GgmlType;

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

    // --- forward pass ---

    /// Run the prompt (token ids) through the model and return the logits for the **last**
    /// position (length [`VOCAB`]). No KV/conv cache yet — recomputes the whole sequence.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let seq = tokens.len();
        assert!(seq > 0, "forward: empty token sequence");

        // Embedding: row `tok` of token_embd (Q6_K) is the embedding of token `tok`.
        let tok_embd = self.tensor("token_embd.weight").expect("token_embd");
        let embd_bytes = self.data(tok_embd);
        let (blk_elems, blk_bytes) = tok_embd.ggml_type.block().expect("embd block");
        let row_bytes = (HIDDEN / blk_elems as usize) * blk_bytes as usize;
        let mut h = vec![0.0f32; seq * HIDDEN];
        for (t, &tok) in tokens.iter().enumerate() {
            let off = tok as usize * row_bytes;
            crate::kernels::dequant::dequantize_into(
                tok_embd.ggml_type,
                &embd_bytes[off..off + row_bytes],
                &mut h[t * HIDDEN..(t + 1) * HIDDEN],
            );
        }

        // Decoder layers: pre-norm, residual-first.
        for layer in 0..N_LAYERS {
            let normed = self.norm_seq(&h, &name(layer, "attn_norm.weight"), seq);
            let op = if config::is_attention(layer) {
                self.attention_op(layer, &normed, seq)
            } else {
                self.conv_op(layer, &normed, seq)
            };
            add_assign(&mut h, &op);

            let normed = self.norm_seq(&h, &name(layer, "ffn_norm.weight"), seq);
            let ffn = if config::is_dense_ffn(layer) {
                self.dense_ffn(layer, &normed, seq)
            } else {
                self.moe_ffn(layer, &normed, seq)
            };
            add_assign(&mut h, &ffn);
        }

        // Final norm on the last position, then tied logits = token_embd · h.
        let last = &h[(seq - 1) * HIDDEN..seq * HIDDEN];
        let gain = self.dequant_vec("token_embd_norm.weight");
        let mut normed_last = vec![0.0f32; HIDDEN];
        rmsnorm(last, &gain, RMS_EPS, &mut normed_last);
        let mut logits = vec![0.0f32; VOCAB];
        matvec(tok_embd.ggml_type, embd_bytes, HIDDEN, VOCAB, &normed_last, &mut logits);
        logits
    }

    /// Autoregressive generation. Recomputes the whole sequence each step (no cache yet),
    /// so cost grows with length. Stops at `eos` or after `max_new` tokens. Returns the
    /// newly generated token ids (not including the prompt).
    pub fn generate(&self, prompt: &[u32], sampler: &mut Sampler, max_new: usize, eos: u32) -> Vec<u32> {
        let mut tokens = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let mut logits = self.forward(&tokens);
            let next = sampler.sample(&mut logits, &tokens);
            tokens.push(next);
            generated.push(next);
            if next == eos {
                break;
            }
        }
        generated
    }

    /// Gated short-conv operator: in_proj → (B·x) → causal conv → (C·) → out_proj.
    fn conv_op(&self, layer: usize, x: &[f32], seq: usize) -> Vec<f32> {
        // in_proj: HIDDEN -> 3*HIDDEN, split per position into [B | C | x].
        let mut bcx = vec![0.0f32; seq * 3 * HIDDEN];
        self.matvec_seq(&name(layer, "shortconv.in_proj.weight"), x, seq, HIDDEN, 3 * HIDDEN, &mut bcx);

        let mut bx = vec![0.0f32; seq * HIDDEN];
        for t in 0..seq {
            let base = t * 3 * HIDDEN;
            let b = &bcx[base..base + HIDDEN];
            let xg = &bcx[base + 2 * HIDDEN..base + 3 * HIDDEN];
            let dst = &mut bx[t * HIDDEN..(t + 1) * HIDDEN];
            for ((d, &bb), &xx) in dst.iter_mut().zip(b).zip(xg) {
                *d = bb * xx;
            }
        }

        let conv_w = self.dequant_vec(&name(layer, "shortconv.conv.weight"));
        let mut conv_out = vec![0.0f32; seq * HIDDEN];
        causal_depthwise_conv(&bx, &conv_w, seq, HIDDEN, CONV_L_CACHE, &mut conv_out);

        let mut y = vec![0.0f32; seq * HIDDEN];
        for t in 0..seq {
            let c = &bcx[t * 3 * HIDDEN + HIDDEN..t * 3 * HIDDEN + 2 * HIDDEN];
            let co = &conv_out[t * HIDDEN..(t + 1) * HIDDEN];
            let dst = &mut y[t * HIDDEN..(t + 1) * HIDDEN];
            for ((d, &cc), &c2) in dst.iter_mut().zip(c).zip(co) {
                *d = cc * c2;
            }
        }

        let mut out = vec![0.0f32; seq * HIDDEN];
        self.matvec_seq(&name(layer, "shortconv.out_proj.weight"), &y, seq, HIDDEN, HIDDEN, &mut out);
        out
    }

    /// GQA attention operator: q/k/v proj → per-head q/k norm → RoPE → SDPA → o_proj.
    fn attention_op(&self, layer: usize, x: &[f32], seq: usize) -> Vec<f32> {
        let mut q = vec![0.0f32; seq * HIDDEN];
        let mut k = vec![0.0f32; seq * KV_DIM];
        let mut v = vec![0.0f32; seq * KV_DIM];
        self.matvec_seq(&name(layer, "attn_q.weight"), x, seq, HIDDEN, HIDDEN, &mut q);
        self.matvec_seq(&name(layer, "attn_k.weight"), x, seq, HIDDEN, KV_DIM, &mut k);
        self.matvec_seq(&name(layer, "attn_v.weight"), x, seq, HIDDEN, KV_DIM, &mut v);

        let q_gain = self.dequant_vec(&name(layer, "attn_q_norm.weight"));
        let k_gain = self.dequant_vec(&name(layer, "attn_k_norm.weight"));
        for t in 0..seq {
            norm_rope_heads(&mut q[t * HIDDEN..(t + 1) * HIDDEN], N_HEADS, &q_gain, t);
            norm_rope_heads(&mut k[t * KV_DIM..(t + 1) * KV_DIM], N_KV_HEADS, &k_gain, t);
        }

        let mut attn_out = vec![0.0f32; seq * HIDDEN];
        attention(&q, &k, &v, seq, N_HEADS, N_KV_HEADS, HEAD_DIM, &mut attn_out);

        let mut out = vec![0.0f32; seq * HIDDEN];
        self.matvec_seq(&name(layer, "attn_output.weight"), &attn_out, seq, HIDDEN, HIDDEN, &mut out);
        out
    }

    /// Dense SwiGLU MLP (layers 0,1): down(silu(gate(x)) · up(x)).
    fn dense_ffn(&self, layer: usize, x: &[f32], seq: usize) -> Vec<f32> {
        let mut gate = vec![0.0f32; seq * DENSE_FF];
        let mut up = vec![0.0f32; seq * DENSE_FF];
        self.matvec_seq(&name(layer, "ffn_gate.weight"), x, seq, HIDDEN, DENSE_FF, &mut gate);
        self.matvec_seq(&name(layer, "ffn_up.weight"), x, seq, HIDDEN, DENSE_FF, &mut up);
        let mut act = vec![0.0f32; seq * DENSE_FF];
        swiglu(&gate, &up, &mut act);
        let mut out = vec![0.0f32; seq * HIDDEN];
        self.matvec_seq(&name(layer, "ffn_down.weight"), &act, seq, DENSE_FF, HIDDEN, &mut out);
        out
    }

    /// Sparse MoE FFN: sigmoid router, top-4 by (score+bias), normalize the selected
    /// **sigmoid** scores, weighted sum of the 4 experts' SwiGLU MLPs. Routed per token.
    fn moe_ffn(&self, layer: usize, x: &[f32], seq: usize) -> Vec<f32> {
        let router = self.tensor(&name(layer, "ffn_gate_inp.weight")).expect("router");
        let bias = self.dequant_vec(&name(layer, "exp_probs_b.bias"));
        let gate_exps = self.tensor(&name(layer, "ffn_gate_exps.weight")).expect("gate_exps");
        let up_exps = self.tensor(&name(layer, "ffn_up_exps.weight")).expect("up_exps");
        let down_exps = self.tensor(&name(layer, "ffn_down_exps.weight")).expect("down_exps");
        let gate_stride = expert_bytes(gate_exps.ggml_type, HIDDEN, MOE_FF);
        let up_stride = expert_bytes(up_exps.ggml_type, HIDDEN, MOE_FF);
        let down_stride = expert_bytes(down_exps.ggml_type, MOE_FF, HIDDEN);

        let mut out = vec![0.0f32; seq * HIDDEN];
        for t in 0..seq {
            let xt = &x[t * HIDDEN..(t + 1) * HIDDEN];

            // Router -> sigmoid scores; select top-k by (score + bias).
            let mut scores = vec![0.0f32; N_EXPERTS];
            matvec(router.ggml_type, self.data(router), HIDDEN, N_EXPERTS, xt, &mut scores);
            sigmoid_slice(&mut scores);
            let mut order: Vec<usize> = (0..N_EXPERTS).collect();
            order.sort_unstable_by(|&a, &b| {
                (scores[b] + bias[b]).total_cmp(&(scores[a] + bias[a]))
            });
            let sel = &order[..N_EXPERTS_USED];

            // Weights are the (bias-free) sigmoid scores of the selected experts, normalized.
            let mut w: Vec<f32> = sel.iter().map(|&e| scores[e]).collect();
            let denom: f32 = w.iter().sum::<f32>() + 1e-6;
            for wi in w.iter_mut() {
                *wi /= denom;
            }

            let out_t = &mut out[t * HIDDEN..(t + 1) * HIDDEN];
            for (i, &e) in sel.iter().enumerate() {
                let gate_w = &self.data(gate_exps)[e * gate_stride..(e + 1) * gate_stride];
                let up_w = &self.data(up_exps)[e * up_stride..(e + 1) * up_stride];
                let down_w = &self.data(down_exps)[e * down_stride..(e + 1) * down_stride];
                let mut g = vec![0.0f32; MOE_FF];
                let mut u = vec![0.0f32; MOE_FF];
                matvec(gate_exps.ggml_type, gate_w, HIDDEN, MOE_FF, xt, &mut g);
                matvec(up_exps.ggml_type, up_w, HIDDEN, MOE_FF, xt, &mut u);
                let mut act = vec![0.0f32; MOE_FF];
                swiglu(&g, &u, &mut act);
                let mut down = vec![0.0f32; HIDDEN];
                matvec(down_exps.ggml_type, down_w, MOE_FF, HIDDEN, &act, &mut down);
                add_scaled(out_t, &down, w[i]);
            }
        }
        out
    }

    /// Apply `matvec` to every position of a `seq × n_in` input, writing `seq × n_out`.
    fn matvec_seq(&self, tensor: &str, x: &[f32], seq: usize, n_in: usize, n_out: usize, out: &mut [f32]) {
        let t = self.tensor(tensor).expect("matvec_seq: tensor");
        let data = self.data(t);
        for p in 0..seq {
            matvec(t.ggml_type, data, n_in, n_out, &x[p * n_in..(p + 1) * n_in], &mut out[p * n_out..(p + 1) * n_out]);
        }
    }

    /// RMSNorm every position of `h` (`seq × HIDDEN`) with the named F32 gain.
    fn norm_seq(&self, h: &[f32], gain_name: &str, seq: usize) -> Vec<f32> {
        let gain = self.dequant_vec(gain_name);
        let mut out = vec![0.0f32; seq * HIDDEN];
        for t in 0..seq {
            rmsnorm(&h[t * HIDDEN..(t + 1) * HIDDEN], &gain, RMS_EPS, &mut out[t * HIDDEN..(t + 1) * HIDDEN]);
        }
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
