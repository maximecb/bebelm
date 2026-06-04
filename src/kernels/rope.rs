//! Rotary position embedding (RoPE), NEOX / "split-half" convention.
//!
//! This matches HF `Lfm2`'s `rotate_half` and ggml's `GGML_ROPE_TYPE_NEOX`: dimension `i`
//! is paired with `i + head_dim/2` (two contiguous halves), *not* interleaved pairs. We
//! reproduce ggml's iterative angle computation (`theta *= theta_scale`) so the rounding
//! matches the reference. LFM2 uses full rotation (n_rot = head_dim) and `freq_base` 5e6.
//!
//! For pair `i` (`0 ≤ i < head_dim/2`) at position `pos`, with
//! `theta_scale = freq_base^(-2/head_dim)` and `θ = pos · theta_scale^i`:
//! ```text
//! x[i]            = x[i]·cos θ − x[i+half]·sin θ
//! x[i + half]     = x[i]·sin θ + x[i+half]·cos θ
//! ```

/// Apply NEOX RoPE in place to one head's vector `x` (length = head_dim) at `pos`.
pub fn rope_neox(x: &mut [f32], pos: usize, freq_base: f32) {
    let d = x.len();
    let half = d / 2;
    let theta_scale = freq_base.powf(-2.0 / d as f32);
    let mut theta = pos as f32;
    for i in 0..half {
        let (sin, cos) = theta.sin_cos();
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos - b * sin;
        x[i + half] = a * sin + b * cos;
        theta *= theta_scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THETA: f32 = 5_000_000.0;

    #[test]
    fn position_zero_is_identity() {
        let orig = [0.3f32, -1.2, 0.7, 2.5, -0.1, 0.9];
        let mut x = orig;
        rope_neox(&mut x, 0, THETA);
        assert_eq!(x, orig);
    }

    #[test]
    fn rotates_first_pair_by_pos_radians() {
        // head_dim = 2 -> single pair, freq = theta^0 = 1, angle = pos.
        let mut x = [1.0f32, 0.0];
        rope_neox(&mut x, 1, THETA);
        assert!((x[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((x[1] - 1.0f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn preserves_vector_norm() {
        // RoPE is a rotation, so the L2 norm is unchanged.
        let orig = [0.5f32, -1.5, 2.0, 0.25, -0.75, 1.1, 0.9, -0.4];
        let mut x = orig;
        rope_neox(&mut x, 7, THETA);
        let n0: f32 = orig.iter().map(|v| v * v).sum();
        let n1: f32 = x.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-4, "{n0} vs {n1}");
    }
}
