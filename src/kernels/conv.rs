//! Causal depthwise 1-D convolution — the LFM2 "short conv" (ggml `ssm_conv` equivalent).
//!
//! Each channel has its own length-`L` filter; the output at position `t` depends only on
//! positions `≤ t` (causal). Matching ggml/HF, the filter's **last** tap multiplies the
//! current token and the **first** tap the oldest:
//!
//! ```text
//! out[t, c] = Σ_{k=0}^{L-1} weight[c, k] · bx[t - (L-1) + k, c]      (bx[<0] = 0)
//! ```
//!
//! This is the from-scratch (prefill) form: positions before 0 are zero — correct for a
//! fresh sequence whose conv state starts at zero. The decode form (prepend the cached
//! last `L-1` columns instead of zeros) comes with the conv-state cache later.

/// Causal depthwise conv over a full sequence.
///
/// - `bx`: input, `seq_len × channels`, position-major (`bx[t*channels + c]`).
/// - `weight`: per-channel taps, `channels × l_cache`, tap-contiguous (`weight[c*l_cache + k]`),
///   exactly as the GGUF `shortconv.conv.weight` is laid out (dims `[l_cache, channels]`).
/// - `out`: same shape/layout as `bx`.
pub fn causal_depthwise_conv(
    bx: &[f32],
    weight: &[f32],
    seq_len: usize,
    channels: usize,
    l_cache: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(bx.len(), seq_len * channels);
    debug_assert_eq!(out.len(), seq_len * channels);
    debug_assert_eq!(weight.len(), channels * l_cache);

    let pad = l_cache - 1;
    for t in 0..seq_len {
        let out_row = &mut out[t * channels..(t + 1) * channels];
        for (c, o) in out_row.iter_mut().enumerate() {
            let w = &weight[c * l_cache..c * l_cache + l_cache];
            let mut sum = 0.0f32;
            for (k, &wk) in w.iter().enumerate() {
                // padded position of this tap relative to the current chunk
                let pos = t as isize - pad as isize + k as isize;
                if pos >= 0 {
                    sum += wk * bx[pos as usize * channels + c];
                }
            }
            *o = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_tap_is_pointwise() {
        // l_cache = 1 -> out = bx * weight (per channel).
        let bx = [1.0f32, 2.0, 3.0, 4.0]; // seq 4, ch 1
        let weight = [5.0f32]; // one channel, one tap
        let mut out = [0.0f32; 4];
        causal_depthwise_conv(&bx, &weight, 4, 1, 1, &mut out);
        assert_eq!(out, [5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn causal_alignment_one_channel() {
        // weight = [w0=1 (oldest), w1=10, w2=100 (current)]; bx = [2, 3, 4].
        let bx = [2.0f32, 3.0, 4.0];
        let weight = [1.0f32, 10.0, 100.0];
        let mut out = [0.0f32; 3];
        causal_depthwise_conv(&bx, &weight, 3, 1, 3, &mut out);
        // out[0] = 100*2 (only current tap in range)
        // out[1] = 10*2 + 100*3
        // out[2] = 1*2 + 10*3 + 100*4
        assert_eq!(out, [200.0, 320.0, 432.0]);
    }

    #[test]
    fn per_channel_independent_layout() {
        // 2 channels, l_cache 2. weight: ch0=[1,2], ch1=[3,4] -> [1,2,3,4].
        // bx position-major: t0=[a0,a1]=[1,1], t1=[b0,b1]=[1,1] -> [1,1,1,1].
        let bx = [1.0f32, 1.0, 1.0, 1.0];
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        causal_depthwise_conv(&bx, &weight, 2, 2, 2, &mut out);
        // out[0,c] = w[c][1]*bx[0,c] (only current tap): [2, 4]
        // out[1,c] = w[c][0]*bx[0,c] + w[c][1]*bx[1,c]: [1+2, 3+4] = [3, 7]
        assert_eq!(out, [2.0, 4.0, 3.0, 7.0]);
    }
}
