//! Hardcoded architecture for LFM2.5-8B-A1B (Q4_K_M).
//!
//! These were extracted from the real GGUF in milestone 1, so the forward pass can treat
//! them as compile-time constants instead of parsing config at runtime. [`validate`] is a
//! cheap startup check that the loaded file actually matches — a wrong or updated file
//! fails loudly instead of silently producing garbage.

use std::error::Error;

use crate::gguf::GgufFile;

pub const ARCH: &str = "lfm2moe";

pub const HIDDEN: usize = 2048; // embedding_length
pub const N_LAYERS: usize = 24; // block_count
pub const VOCAB: usize = 128_000;

pub const N_HEADS: usize = 32; // attention.head_count
pub const N_KV_HEADS: usize = 8; // attention.head_count_kv (on attention layers)
pub const HEAD_DIM: usize = HIDDEN / N_HEADS; // 64
pub const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 512

pub const DENSE_FF: usize = 7168; // feed_forward_length (layers 0,1)
pub const MOE_FF: usize = 1792; // expert_feed_forward_length
pub const N_EXPERTS: usize = 32; // expert_count
pub const N_EXPERTS_USED: usize = 4; // expert_used_count (top-k)
pub const N_DENSE_LAYERS: usize = 2; // leading_dense_block_count

pub const CONV_L_CACHE: usize = 3; // shortconv kernel size

pub const ROPE_THETA: f32 = 5_000_000.0;
pub const RMS_EPS: f32 = 1e-5;

pub const BOS_TOKEN: u32 = 124_894;
pub const EOS_TOKEN: u32 = 124_900;
pub const PAD_TOKEN: u32 = 124_893;

/// 0-indexed layers using grouped-query attention; all others use the gated short conv.
pub const ATTENTION_LAYERS: [usize; 6] = [2, 6, 10, 14, 18, 21];

/// Whether layer `i`'s operator is attention (vs. the gated short convolution).
pub fn is_attention(layer: usize) -> bool {
    ATTENTION_LAYERS.contains(&layer)
}

/// Whether layer `i`'s FFN is a dense SwiGLU MLP (vs. the sparse MoE).
pub fn is_dense_ffn(layer: usize) -> bool {
    layer < N_DENSE_LAYERS
}

/// Assert the GGUF metadata matches the hardcoded constants above.
pub fn validate(g: &GgufFile) -> Result<(), Box<dyn Error>> {
    let arch = g.architecture().ok_or("missing general.architecture")?;
    if arch != ARCH {
        return Err(format!("architecture: expected {ARCH:?}, got {arch:?}").into());
    }

    expect_u32(g, "lfm2moe.block_count", N_LAYERS as u32)?;
    expect_u32(g, "lfm2moe.embedding_length", HIDDEN as u32)?;
    expect_u32(g, "lfm2moe.vocab_size", VOCAB as u32)?;
    expect_u32(g, "lfm2moe.attention.head_count", N_HEADS as u32)?;
    expect_u32(g, "lfm2moe.expert_count", N_EXPERTS as u32)?;
    expect_u32(g, "lfm2moe.expert_used_count", N_EXPERTS_USED as u32)?;
    expect_u32(g, "lfm2moe.feed_forward_length", DENSE_FF as u32)?;
    expect_u32(g, "lfm2moe.expert_feed_forward_length", MOE_FF as u32)?;
    expect_u32(g, "lfm2moe.leading_dense_block_count", N_DENSE_LAYERS as u32)?;
    expect_u32(g, "lfm2moe.shortconv.l_cache", CONV_L_CACHE as u32)?;

    expect_f32(g, "lfm2moe.rope.freq_base", ROPE_THETA, 1.0)?;
    expect_f32(g, "lfm2moe.attention.layer_norm_rms_epsilon", RMS_EPS, 1e-9)?;

    // The per-layer kv-head array encodes the conv/attn schedule: 8 on attention layers,
    // 0 on conv layers. Cross-check it against ATTENTION_LAYERS.
    let kv = g
        .get_u32_array("lfm2moe.attention.head_count_kv")
        .ok_or("attention.head_count_kv missing or not an array")?;
    if kv.len() != N_LAYERS {
        return Err(format!("head_count_kv has {} entries, expected {N_LAYERS}", kv.len()).into());
    }
    for (i, &h) in kv.iter().enumerate() {
        let want = if is_attention(i) { N_KV_HEADS as u32 } else { 0 };
        if h != want {
            return Err(
                format!("layer {i}: head_count_kv={h}, expected {want} (schedule mismatch)").into(),
            );
        }
    }

    Ok(())
}

fn expect_u32(g: &GgufFile, key: &str, want: u32) -> Result<(), Box<dyn Error>> {
    let got = g.get_u32(key).ok_or_else(|| format!("missing metadata {key}"))?;
    if got != want {
        return Err(format!("{key}: expected {want}, got {got}").into());
    }
    Ok(())
}

fn expect_f32(g: &GgufFile, key: &str, want: f32, tol: f32) -> Result<(), Box<dyn Error>> {
    let got = g.get_f32(key).ok_or_else(|| format!("missing metadata {key}"))?;
    if (got - want).abs() > tol {
        return Err(format!("{key}: expected {want}, got {got}").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_predicates() {
        assert!(is_attention(2) && is_attention(21));
        assert!(!is_attention(0) && !is_attention(3));
        assert_eq!(ATTENTION_LAYERS.iter().filter(|&&i| i < N_LAYERS).count(), 6);
        assert!(is_dense_ffn(0) && is_dense_ffn(1));
        assert!(!is_dense_ffn(2));
    }

    #[test]
    fn dimension_consistency() {
        assert_eq!(HEAD_DIM * N_HEADS, HIDDEN);
        assert_eq!(KV_DIM, 512);
    }
}
