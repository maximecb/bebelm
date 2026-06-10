//! Causal depthwise 1-D convolution — the LFM2 "short conv" (ggml `ssm_conv` equivalent),
//! single-token / decode form.
//!
//! Each channel has its own length-`L` filter. Matching ggml/HF, the filter's **last** tap
//! multiplies the current token and earlier taps the cached history (oldest first):
//!
//! ```text
//! out[c] = Σ_{k<L-1} weight[c, k] · state[k, c]  +  weight[c, L-1] · bx[c]
//! ```
//!
//! `state` is the previous `L-1` columns of Bx; at the start of a sequence it is zeros
//! (so only the current tap contributes), matching a fresh conv state.

/// Single-token causal depthwise conv. `state` holds the previous `l_cache-1` columns of
/// Bx (oldest first, `channels` each); `bx` is the current column. `weight` is the
/// per-channel taps, `channels × l_cache`, tap-contiguous (`weight[c*l_cache + k]`), as the
/// GGUF `shortconv.conv.weight` is laid out.
pub fn conv_step(
    state: &[f32],
    bx: &[f32],
    weight: &[f32],
    channels: usize,
    l_cache: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(state.len(), channels * (l_cache - 1));
    debug_assert_eq!(bx.len(), channels);
    debug_assert_eq!(out.len(), channels);
    debug_assert_eq!(weight.len(), channels * l_cache);

    for (c, o) in out.iter_mut().enumerate() {
        let w = &weight[c * l_cache..c * l_cache + l_cache];
        let mut sum = w[l_cache - 1] * bx[c];
        for (k, &wk) in w.iter().take(l_cache - 1).enumerate() {
            sum += wk * state[k * channels + c];
        }
        *o = sum;
    }
}

/// Batched (prefill) causal depthwise conv over `n_tokens` consecutive positions.
///
/// `bx` and `out` are token-major (`bx[t*channels + c]`, oldest position first). `state` holds
/// the `l_cache-1` columns *preceding* this batch (oldest first, `channels` each), exactly as
/// the decode [`conv_step`] cache does; on return it holds the batch's trailing `l_cache-1`
/// columns, ready for the next step. The result is **bit-for-bit identical** to running
/// `conv_step` once per position with the rolling state update — it *is* that loop, hoisted here
/// so the conv operator processes the whole prompt in one call (its in/out projections batch via
/// `matmul`).
pub fn conv_prefill(
    state: &mut [f32],
    bx: &[f32],
    weight: &[f32],
    channels: usize,
    l_cache: usize,
    n_tokens: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(state.len(), channels * (l_cache - 1));
    debug_assert_eq!(bx.len(), channels * n_tokens);
    debug_assert_eq!(out.len(), channels * n_tokens);

    for t in 0..n_tokens {
        let bx_t = &bx[t * channels..(t + 1) * channels];
        conv_step(state, bx_t, weight, channels, l_cache, &mut out[t * channels..(t + 1) * channels]);
        // Slide the state left one column and append this position's Bx (matches conv_step_op).
        state.copy_within(channels.., 0);
        state[(l_cache - 2) * channels..].copy_from_slice(bx_t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_tap_is_pointwise() {
        // l_cache = 1 -> no state, out = bx * weight.
        let mut out = [0.0f32; 1];
        conv_step(&[], &[3.0], &[5.0], 1, 1, &mut out);
        assert_eq!(out, [15.0]);
    }

    #[test]
    fn causal_alignment_one_channel() {
        // weight = [w0=1 (oldest), w1=10, w2=100 (current)]; state = [2, 3], current = 4.
        let weight = [1.0f32, 10.0, 100.0];
        let state = [2.0f32, 3.0];
        let mut out = [0.0f32; 1];
        conv_step(&state, &[4.0], &weight, 1, 3, &mut out);
        // 1*2 (oldest) + 10*3 + 100*4 (current) = 432
        assert_eq!(out, [432.0]);
    }

    #[test]
    fn zero_state_uses_only_current_tap() {
        // Start of a sequence: state is zeros -> only the last (current) tap contributes.
        let weight = [1.0f32, 10.0, 100.0];
        let mut out = [0.0f32; 1];
        conv_step(&[0.0, 0.0], &[4.0], &weight, 1, 3, &mut out);
        assert_eq!(out, [400.0]);
    }

    #[test]
    fn prefill_matches_rolling_conv_step() {
        // conv_prefill must be bit-identical to conv_step rolled per position with the same
        // state update. 3 channels, l_cache 3, 5 positions, a non-trivial initial state.
        let channels = 3;
        let l_cache = 3;
        let n = 5;
        let weight: Vec<f32> = (0..channels * l_cache).map(|i| (i as f32 - 4.0) * 0.5).collect();
        let bx: Vec<f32> = (0..channels * n).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();
        let init_state: Vec<f32> = (0..channels * (l_cache - 1)).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Reference: the per-token loop (what conv_step_op does).
        let mut ref_state = init_state.clone();
        let mut ref_out = vec![0.0f32; channels * n];
        for t in 0..n {
            let bx_t = &bx[t * channels..(t + 1) * channels];
            conv_step(&ref_state, bx_t, &weight, channels, l_cache, &mut ref_out[t * channels..(t + 1) * channels]);
            ref_state.copy_within(channels.., 0);
            ref_state[(l_cache - 2) * channels..].copy_from_slice(bx_t);
        }

        let mut state = init_state.clone();
        let mut out = vec![0.0f32; channels * n];
        conv_prefill(&mut state, &bx, &weight, channels, l_cache, n, &mut out);

        assert_eq!(out, ref_out, "outputs differ");
        assert_eq!(state, ref_state, "final state differs");
    }

    #[test]
    fn per_channel_independent_layout() {
        // 2 channels, l_cache 2. weight: ch0=[1,2], ch1=[3,4] -> [1,2,3,4].
        // state (1 col): ch0=10, ch1=20. current bx = [1, 1].
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let state = [10.0f32, 20.0];
        let mut out = [0.0f32; 2];
        conv_step(&state, &[1.0, 1.0], &weight, 2, 2, &mut out);
        // out[0] = w[0][1]*1 + w[0][0]*10 = 2 + 10 = 12
        // out[1] = w[1][1]*1 + w[1][0]*20 = 4 + 60 = 64
        assert_eq!(out, [12.0, 64.0]);
    }
}
