//! RMSNorm kernel.

/// Root-mean-square layer norm: `out[i] = x[i] / sqrt(mean(x²) + eps) · gain[i]`.
///
/// `x`, `gain`, and `out` must share the same length. Used for the operator/ffn/final
/// norms (length = hidden_size) and the per-head q/k norms (length = head_dim). The sum of
/// squares is accumulated in `f64` (each f32 square widened into the sum), matching ggml's
/// reference (`ggml_float` = `double`); an f32 accumulator loses low bits over a long row.
/// The `mean` is then narrowed back to f32 for the scale, as ggml does.
pub fn rmsnorm(x: &[f32], gain: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(gain.len(), n, "rmsnorm: gain length");
    debug_assert_eq!(out.len(), n, "rmsnorm: out length");

    let ss: f64 = x.iter().map(|&v| (v * v) as f64).sum();
    let mean = (ss / n as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();
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

    /// Deterministic vector whose f32 sequential Σx² loses precision: a few large entries up
    /// front make the running sum big, so the many small squares that follow get absorbed
    /// below the f32 ulp at that magnitude (an f64 accumulator keeps them). Mirrors a real
    /// activation row with a handful of outliers.
    fn precision_stress_vector(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| if i < 8 { 50.0 } else { 0.05 * (((i % 7) as f32) - 3.0) })
            .collect()
    }

    #[test]
    fn rmsnorm_sum_of_squares_uses_wide_accumulator() {
        // Guards the f64 accumulation of Σx². Compare the kernel to a full-f64 ground truth on
        // a vector built to expose f32 absorption error. With the f64 accumulator the gap is
        // just f32 tail rounding (measured ~8e-8); an f32 accumulator drops the small squares
        // and the scale drifts ~4.5e-6 — so the tight bound below catches that regression.
        let n = 4096;
        let x = precision_stress_vector(n);
        let gain = vec![1.0f32; n];
        let eps = 1e-5f32;

        // Ground truth: accumulate and scale entirely in f64, narrow only the final output.
        let ss64: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let scale64 = 1.0 / (ss64 / n as f64 + eps as f64).sqrt();
        let reference: Vec<f32> = x.iter().map(|&xi| (xi as f64 * scale64) as f32).collect();

        let max_rel = |scale: f32| {
            x.iter()
                .zip(&reference)
                .map(|(&xi, &r)| (xi * scale - r).abs() / r.abs().max(1e-6))
                .fold(0.0f32, f32::max)
        };

        let mut out = vec![0.0f32; n];
        rmsnorm(&x, &gain, eps, &mut out);
        let got = out
            .iter()
            .zip(&reference)
            .map(|(&o, &r)| (o - r).abs() / r.abs().max(1e-6))
            .fold(0.0f32, f32::max);

        // The relative error of an f32 accumulator on this vector — kept so the bound is set
        // with knowledge of what a regression looks like and the test is provably discriminating.
        let ss32: f32 = x.iter().map(|&v| v * v).sum();
        let regressed = max_rel(1.0 / (ss32 / n as f32 + eps).sqrt());

        // Tight over the measured f64-path error (~8e-8) to catch precision regressions early,
        // while sitting well below the f32-accumulator drift (~4.5e-6).
        const TOL: f32 = 3e-7;
        assert!(got < TOL, "rmsnorm vs f64 ground truth: max_rel={got:.3e}");
        assert!(
            regressed > TOL,
            "test no longer discriminates: f32-accumulator rel={regressed:.3e} ≤ TOL={TOL:.1e}"
        );
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
