//! Model loading: open the GGUF, validate it against the hardcoded [`crate::config`], and
//! resolve every tensor the forward pass needs by name (with a shape check). The forward
//! pass itself is added in the next milestone.

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

use crate::config;
use crate::gguf::{GgufFile, TensorInfo};

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
