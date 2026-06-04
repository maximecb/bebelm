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
    debug_assert_eq!(v.len(), seq_len * n_kv_heads * head_dim);
    debug_assert_eq!(out.len(), seq_len * n_heads * head_dim);
    debug_assert_eq!(n_heads % n_kv_heads, 0);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let group = n_heads / n_kv_heads; // query heads per kv head
    let mut scores = vec![0.0f32; seq_len];

    // index of head `hh` (of `n` heads) at position `t`, dim 0
    let head_off = |t: usize, hh: usize, n: usize| (t * n + hh) * head_dim;

    for t in 0..seq_len {
        for h in 0..n_heads {
            let kv = h / group;
            let q_vec = &q[head_off(t, h, n_heads)..][..head_dim];

            // causal scores over keys 0..=t
            let scores = &mut scores[..=t];
            for (j, s) in scores.iter_mut().enumerate() {
                let k_vec = &k[head_off(j, kv, n_kv_heads)..][..head_dim];
                *s = dot(q_vec, k_vec) * scale;
            }
            softmax(scores);

            // weighted sum of values
            let out_vec = &mut out[head_off(t, h, n_heads)..][..head_dim];
            out_vec.fill(0.0);
            for (j, &w) in scores.iter().enumerate() {
                let v_vec = &v[head_off(j, kv, n_kv_heads)..][..head_dim];
                for (o, &vv) in out_vec.iter_mut().zip(v_vec) {
                    *o += w * vv;
                }
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
