//! Elementwise vector helpers: residual add and scaled accumulation.

/// In-place add: `a[i] += b[i]` (residual connections).
pub fn add_assign(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, &y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// Scaled accumulate: `out[i] += s * b[i]` (weighted expert sums in MoE).
pub fn add_scaled(out: &mut [f32], b: &[f32], s: f32) {
    debug_assert_eq!(out.len(), b.len());
    for (o, &x) in out.iter_mut().zip(b) {
        *o += s * x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let mut a = [1.0f32, 2.0, 3.0];
        add_assign(&mut a, &[10.0, 20.0, 30.0]);
        assert_eq!(a, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn scaled_accumulate() {
        let mut out = [1.0f32, 1.0, 1.0];
        add_scaled(&mut out, &[2.0, 4.0, 6.0], 0.5);
        assert_eq!(out, [2.0, 3.0, 4.0]);
    }
}
