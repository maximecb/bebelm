//! RMSNorm kernel.

/// Root-mean-square layer norm: `out[i] = x[i] / sqrt(mean(x²) + eps) · gain[i]`.
///
/// `x`, `gain`, and `out` must share the same length. Used for the operator/ffn/final
/// norms (length = hidden_size) and the per-head q/k norms (length = head_dim). The sum of
/// squares is accumulated in `f32`, matching ggml's reference.
pub fn rmsnorm(x: &[f32], gain: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(gain.len(), n, "rmsnorm: gain length");
    debug_assert_eq!(out.len(), n, "rmsnorm: out length");

    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for ((o, &xi), &g) in out.iter_mut().zip(x).zip(gain) {
        *o = xi * scale * g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_to_unit_rms() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let gain = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        rmsnorm(&x, &gain, 0.0, &mut out);

        // With unit gain, the result should have mean-square ≈ 1.
        let ms: f32 = out.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((ms - 1.0).abs() < 1e-5, "ms = {ms}");

        // scale = 1/sqrt(mean(x²)) = 1/sqrt(7.5)
        let s = 1.0 / 7.5f32.sqrt();
        assert!((out[0] - s).abs() < 1e-6);
        assert!((out[3] - 4.0 * s).abs() < 1e-6);
    }

    #[test]
    fn applies_gain() {
        // x all ones -> rms = 1, scale = 1, so out == gain.
        let x = [1.0f32; 4];
        let gain = [2.0f32, 0.5, 1.0, 3.0];
        let mut out = [0.0f32; 4];
        rmsnorm(&x, &gain, 0.0, &mut out);
        assert_eq!(out, [2.0, 0.5, 1.0, 3.0]);
    }

    #[test]
    fn eps_guards_zero_input() {
        let x = [0.0f32; 4];
        let gain = [1.0f32; 4];
        let mut out = [9.0f32; 4];
        rmsnorm(&x, &gain, 1e-5, &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]); // 0 * finite scale = 0, no NaN
    }
}
