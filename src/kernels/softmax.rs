//! Numerically-stable softmax.

/// In-place softmax over a slice: subtract the max before exp to avoid overflow.
/// An empty slice is a no-op.
///
/// The normalizing sum of exponentials is accumulated in `f64` (each f32 `exp` widened into
/// the sum), matching ggml's reference (`ggml_float` = `double`); a narrower f32 accumulator
/// drops small tail terms under a large running sum, perturbing every output uniformly.
pub fn softmax(x: &mut [f32]) {
    let Some(&max) = x.iter().reduce(|a, b| if a >= b { a } else { b }) else {
        return; // empty
    };
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v as f64;
    }
    let inv = (1.0 / sum) as f32;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < tol, "{x} vs {y}");
        }
    }

    #[test]
    fn uniform_input_uniform_output() {
        let mut x = [1.0f32, 1.0, 1.0, 1.0];
        softmax(&mut x);
        approx(&x, &[0.25, 0.25, 0.25, 0.25], 1e-7);
    }

    #[test]
    fn known_distribution_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0];
        softmax(&mut x);
        approx(&x, &[0.090_030_57, 0.244_728_47, 0.665_240_96], 1e-6);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stable_for_large_inputs() {
        let mut x = [1000.0f32, 1000.0];
        softmax(&mut x);
        approx(&x, &[0.5, 0.5], 1e-7); // no overflow to NaN/inf
    }

    /// Logits whose softmax sum loses precision in f32: a block of max-valued entries up front
    /// drives the running Σexp near `n_peak`, so the many far-below entries that follow have
    /// `exp` under the f32 ulp at that magnitude and an f32 accumulator drops them (an f64 one
    /// keeps them). Models a sharply peaked attention row over a long context.
    fn peaked_logits(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| if i < 16 { 0.0 } else { -15.0 + (i % 3) as f32 * 0.2 })
            .collect()
    }

    #[test]
    fn softmax_sum_uses_wide_accumulator() {
        // Guards the f64 accumulation of Σexp. Softmax divides every output by that sum, so a
        // relative error in the sum shifts all outputs by the same relative amount. Compare to a
        // ground truth that reuses the kernel's f32 exps but sums them in f64: with the f64
        // accumulator the gap is at most a couple f32 ulps (here exactly 0); an f32 accumulator
        // drops the tail and the sum drifts ~1e-4 — far past the tight bound below.
        let n = 4096;
        let logits = peaked_logits(n);

        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
        let sum64: f64 = exps.iter().map(|&e| e as f64).sum();
        let reference: Vec<f32> = exps.iter().map(|&e| (e as f64 / sum64) as f32).collect();

        let max_rel = |inv: f32| {
            exps.iter()
                .zip(&reference)
                .map(|(&e, &r)| (e * inv - r).abs() / r.abs().max(1e-9))
                .fold(0.0f32, f32::max)
        };

        let mut out = logits.clone();
        softmax(&mut out);
        let got = out
            .iter()
            .zip(&reference)
            .map(|(&o, &r)| (o - r).abs() / r.abs().max(1e-9))
            .fold(0.0f32, f32::max);

        // What an f32 accumulator would produce — kept so the bound is set knowing what a
        // regression looks like and the test is provably discriminating.
        let sum32: f32 = exps.iter().sum();
        let regressed = max_rel(1.0 / sum32);

        const TOL: f32 = 1e-6;
        assert!(got < TOL, "softmax vs f64 ground truth: max_rel={got:.3e}");
        assert!(
            regressed > TOL,
            "test no longer discriminates: f32-accumulator rel={regressed:.3e} ≤ TOL={TOL:.1e}"
        );
    }

    #[test]
    fn empty_is_noop() {
        let mut x: [f32; 0] = [];
        softmax(&mut x); // must not panic
    }
}
