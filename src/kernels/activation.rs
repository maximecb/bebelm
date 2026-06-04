//! Activation functions: SiLU, sigmoid, and the SwiGLU glue.

/// Logistic sigmoid: `1 / (1 + e^-x)`.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU / swish: `x · sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU glue: `out[i] = silu(gate[i]) · up[i]` (FFN and experts).
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = silu(g) * u;
    }
}

/// In-place sigmoid over a slice (MoE router scores).
pub fn sigmoid_slice(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = sigmoid(*v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_silu_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!(sigmoid(-100.0) < 1e-6);
        assert!(sigmoid(100.0) > 1.0 - 1e-6);

        assert_eq!(silu(0.0), 0.0);
        // silu(x) = x*sigmoid(x); at x=1: 1*0.7310586 = 0.7310586
        assert!((silu(1.0) - 0.731_058_6).abs() < 1e-6);
        // for large positive x, silu(x) ~= x
        assert!((silu(20.0) - 20.0).abs() < 1e-3);
    }

    #[test]
    fn swiglu_glue() {
        let gate = [0.0f32, 1.0];
        let up = [5.0f32, 2.0];
        let mut out = [0.0f32; 2];
        swiglu(&gate, &up, &mut out);
        assert_eq!(out[0], 0.0); // silu(0)*5 = 0
        assert!((out[1] - 0.731_058_6 * 2.0).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_over_slice() {
        let mut x = [0.0f32, 0.0];
        sigmoid_slice(&mut x);
        assert_eq!(x, [0.5, 0.5]);
    }
}
