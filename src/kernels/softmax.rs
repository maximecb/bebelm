//! Numerically-stable softmax.

/// In-place softmax over a slice: subtract the max before exp to avoid overflow.
/// An empty slice is a no-op.
pub fn softmax(x: &mut [f32]) {
    let Some(&max) = x.iter().reduce(|a, b| if a >= b { a } else { b }) else {
        return; // empty
    };
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
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

    #[test]
    fn empty_is_noop() {
        let mut x: [f32; 0] = [];
        softmax(&mut x); // must not panic
    }
}
