//! Causal grouped-query attention (SDPA core).
//!
//! Inputs are already projected, q/k-normed, and RoPE'd. For each query position `t` and
//! head `h`, attends over key positions `0..=t` (causal):
//! `out[t,h] = Σ_{j≤t} softmax_j( q[t,h]·k[j,kv]/√head_dim ) · v[j,kv]`,
//! where the kv head is `kv = h / (n_heads / n_kv_heads)` (GQA).
//!
//! Layout (all position-major, then head-major, then dim):
//! - `q`, `out`: `seq_len × n_heads × head_dim`
//! - `k`, `v`:   `seq_len × n_kv_heads × head_dim`
//!
//! `out` is contiguous `[seq_len, n_heads·head_dim] = [seq_len, hidden]`, ready for o_proj.

use crate::kernels::matmul::dot;
use crate::kernels::softmax::softmax;

#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(q.len(), seq_len * n_heads * head_dim);
    debug_assert_eq!(k.len(), seq_len * n_kv_heads * head_dim);
    debug_assert_eq!(out.len(), seq_len * n_heads * head_dim);

    // Each query position t attends to keys 0..=t — i.e. a decode step against the first
    // t+1 cached positions.
    let qrow = n_heads * head_dim;
    let kvrow = n_kv_heads * head_dim;
    for t in 0..seq_len {
        let n_ctx = t + 1;
        attention_decode(
            &q[t * qrow..(t + 1) * qrow],
            &k[..n_ctx * kvrow],
            &v[..n_ctx * kvrow],
            n_ctx,
            n_heads,
            n_kv_heads,
            head_dim,
            &mut out[t * qrow..(t + 1) * qrow],
        );
    }
}

/// Single-query attention: one query (`q`, `n_heads × head_dim`) attends to a cached
/// history of `n_ctx` key/value positions (`k`/`v`, `n_ctx × n_kv_heads × head_dim`).
/// Writes `out` (`n_heads × head_dim`). This is the decode-step core; the query is the
/// latest position, so it attends to all `n_ctx` keys (no extra mask needed).
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
    fn single_token_returns_value() {
        // seq 1, 1 head: softmax over one score = 1, so out == v.
        let q = [1.0f32, 0.0];
        let k = [1.0f32, 0.0];
        let v = [5.0f32, 7.0];
        let mut out = [0.0f32; 2];
        attention(&q, &k, &v, 1, 1, 1, 2, &mut out);
        assert_eq!(out, [5.0, 7.0]);
    }

    #[test]
    fn causal_masking_and_averaging() {
        // 2 tokens, 1 head, head_dim 2. Equal q,k so position 1 attends 50/50.
        let q = [1.0f32, 0.0, 1.0, 0.0]; // t0, t1
        let k = [1.0f32, 0.0, 1.0, 0.0];
        let v = [2.0f32, 0.0, 4.0, 0.0]; // v0=[2,0], v1=[4,0]
        let mut out = [0.0f32; 4];
        attention(&q, &k, &v, 2, 1, 1, 2, &mut out);
        // pos0 sees only v0; pos1 averages v0,v1.
        assert!((out[0] - 2.0).abs() < 1e-6 && out[1] == 0.0);
        assert!((out[2] - 3.0).abs() < 1e-6 && out[3] == 0.0);
    }

    #[test]
    fn gqa_shares_kv_head() {
        // 2 query heads, 1 kv head: both heads use the same k/v.
        // seq 1 -> softmax = 1 -> each head outputs v.
        let q = [1.0f32, 0.0, 0.0, 1.0]; // head0, head1
        let k = [1.0f32, 1.0]; // single kv head
        let v = [5.0f32, 7.0];
        let mut out = [0.0f32; 4];
        attention(&q, &k, &v, 1, 2, 1, 2, &mut out);
        assert_eq!(out, [5.0, 7.0, 5.0, 7.0]);
    }
}
