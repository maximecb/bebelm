//! Elementwise vector helpers: residual adds, gate multiplies, scaled accumulation.

/// In-place add: `a[i] += b[i]` (residual connections).
pub fn add_assign(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, &y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// In-place multiply: `a[i] *= b[i]` (elementwise gates).
pub fn mul_assign(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, &y) in a.iter_mut().zip(b) {
        *x *= y;
    }
}

/// Out-of-place multiply: `out[i] = a[i] * b[i]`.
pub fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x * y;
    }
}

/// In-place scale: `a[i] *= s`.
pub fn scale(a: &mut [f32], s: f32) {
    for x in a.iter_mut() {
        *x *= s;
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
    fn add_and_mul() {
        let mut a = [1.0f32, 2.0, 3.0];
        add_assign(&mut a, &[10.0, 20.0, 30.0]);
        assert_eq!(a, [11.0, 22.0, 33.0]);

        let mut b = [2.0f32, 3.0, 4.0];
        mul_assign(&mut b, &[5.0, 6.0, 7.0]);
        assert_eq!(b, [10.0, 18.0, 28.0]);

        let mut out = [0.0f32; 3];
        mul(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &mut out);
        assert_eq!(out, [4.0, 10.0, 18.0]);
    }

    #[test]
    fn scale_and_accumulate() {
        let mut a = [1.0f32, -2.0, 3.0];
        scale(&mut a, 2.0);
        assert_eq!(a, [2.0, -4.0, 6.0]);

        let mut out = [1.0f32, 1.0, 1.0];
        add_scaled(&mut out, &[2.0, 4.0, 6.0], 0.5);
        assert_eq!(out, [2.0, 3.0, 4.0]);
    }
}
