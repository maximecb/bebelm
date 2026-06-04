//! Single-query causal grouped-query attention (the decode-step SDPA core).
//!
//! One query (the latest position) attends to a cached history of `n_ctx` key/value
//! positions. Inputs are already projected, q/k-normed, and RoPE'd. With GQA, query head
//! `h` uses kv head `kv = h / (n_heads / n_kv_heads)`:
//! `out[h] = Σ_{j<n_ctx} softmax_j( q[h]·k[j,kv]/√head_dim ) · v[j,kv]`.
//!
//! Layout: `q`/`out` are `n_heads × head_dim`; `k`/`v` are `n_ctx × n_kv_heads × head_dim`.
//! `out` is contiguous `[n_heads·head_dim] = [hidden]`, ready for o_proj.

use crate::kernels::matmul::dot;
use crate::kernels::softmax::softmax;

/// See module docs. The query is the latest position, so it attends to all `n_ctx` keys
/// (no extra causal mask needed — the cache only contains positions `≤` the query).
#[allow(clippy::too_many_arguments)]
pub fn attention_decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_ctx: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(k.len(), n_ctx * n_kv_heads * head_dim);
    debug_assert_eq!(v.len(), n_ctx * n_kv_heads * head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);
    debug_assert_eq!(n_heads % n_kv_heads, 0);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let group = n_heads / n_kv_heads;
    let mut scores = vec![0.0f32; n_ctx];

    for h in 0..n_heads {
        let kv = h / group;
        let q_vec = &q[h * head_dim..(h + 1) * head_dim];
        for (j, s) in scores.iter_mut().enumerate() {
            let k_vec = &k[(j * n_kv_heads + kv) * head_dim..][..head_dim];
            *s = dot(q_vec, k_vec) * scale;
        }
        softmax(&mut scores);

        let out_vec = &mut out[h * head_dim..(h + 1) * head_dim];
        out_vec.fill(0.0);
        for (j, &w) in scores.iter().enumerate() {
            let v_vec = &v[(j * n_kv_heads + kv) * head_dim..][..head_dim];
            for (o, &vv) in out_vec.iter_mut().zip(v_vec) {
                *o += w * vv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_context_returns_value() {
        // n_ctx 1, 1 head: softmax over one score = 1, so out == v.
        let q = [1.0f32, 0.0];
        let k = [1.0f32, 0.0];
        let v = [5.0f32, 7.0];
        let mut out = [0.0f32; 2];
        attention_decode(&q, &k, &v, 1, 1, 1, 2, &mut out);
        assert_eq!(out, [5.0, 7.0]);
    }

    #[test]
    fn averages_equal_scored_context() {
        // 1 head, head_dim 2; query attends to 2 keys with equal q·k -> 50/50 average.
        let q = [1.0f32, 0.0];
        let k = [1.0f32, 0.0, 1.0, 0.0]; // 2 positions, identical
        let v = [2.0f32, 0.0, 4.0, 0.0]; // v0=[2,0], v1=[4,0]
        let mut out = [0.0f32; 2];
        attention_decode(&q, &k, &v, 2, 1, 1, 2, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6 && out[1] == 0.0);
    }

    #[test]
    fn gqa_shares_kv_head() {
        // 2 query heads, 1 kv head, n_ctx 1: both heads use the same k/v -> both output v.
        let q = [1.0f32, 0.0, 0.0, 1.0]; // head0, head1
        let k = [1.0f32, 1.0]; // single kv head
        let v = [5.0f32, 7.0];
        let mut out = [0.0f32; 4];
        attention_decode(&q, &k, &v, 1, 2, 1, 2, &mut out);
        assert_eq!(out, [5.0, 7.0, 5.0, 7.0]);
    }
}
