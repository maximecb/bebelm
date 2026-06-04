//! Matrix-vector product against a (possibly quantized) weight matrix.
//!
//! GGUF stores a weight for `y = W·x` with dims `[in_features, out_features]`, laid out so
//! that each output's `in_features` weights are **contiguous** — i.e. output `o` is the dot
//! product of contiguous weight row `o` with `x`. Quantization runs along `in_features`, so
//! each row is a whole number of 256-weight K-quant super-blocks.
//!
//! Each output row is independent — dequantize its weight row into a scratch buffer, then
//! dot with `x` — so the row loop runs across CPU cores via rayon. Partitioning by row
//! leaves every dot's accumulation order unchanged, so the result is bit-for-bit identical
//! to the serial path regardless of thread count.

use crate::kernels::dequant;
use crate::tensor::GgmlType;
use rayon::prelude::*;
use wide::f32x8;

/// Below this many output rows, dispatching work to the thread pool costs more than the
/// rows save, so `matvec` runs serially (the router and k/v projections fall here).
const PAR_MIN_ROWS: usize = 64;

/// Read the first 8 elements of `s` as an `f32x8` (one 256-bit / 2× NEON vector).
#[inline]
fn load8(s: &[f32]) -> f32x8 {
    f32x8::from(<[f32; 8]>::try_from(&s[..8]).unwrap())
}

/// Dot product of two equal-length `f32` slices.
///
/// Vectorized with `f32x8` over four independent accumulators (ILP, to hide FMA latency),
/// with a scalar tail for any remainder. Because this sums lane-wise partial products with
/// fused multiply-add, the result is **not** bit-identical to a left-to-right scalar dot —
/// the rounding differs. (Inputs in `matvec` are always a multiple of the 256-wide block,
/// so the tail there is empty; the tail only serves small/odd callers and tests.)
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    const W: usize = 8; // f32x8 lane count
    const STEP: usize = W * 4; // four accumulators per iteration
    let n = a.len();

    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);

    let mut i = 0;
    while i + STEP <= n {
        acc0 = load8(&a[i..]).mul_add(load8(&b[i..]), acc0);
        acc1 = load8(&a[i + W..]).mul_add(load8(&b[i + W..]), acc1);
        acc2 = load8(&a[i + 2 * W..]).mul_add(load8(&b[i + 2 * W..]), acc2);
        acc3 = load8(&a[i + 3 * W..]).mul_add(load8(&b[i + 3 * W..]), acc3);
        i += STEP;
    }
    while i + W <= n {
        acc0 = load8(&a[i..]).mul_add(load8(&b[i..]), acc0);
        i += W;
    }

    let mut sum = ((acc0 + acc1) + (acc2 + acc3)).reduce_add();
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Compute `y = W·x`, where `W` is `[n_in, n_out]` quantized as `dtype` in `w`.
///
/// `x.len() == n_in`, `y.len() == n_out`. Panics if `dtype` is unsupported or `n_in` is
/// not a multiple of the dtype's block size.
pub fn matvec(dtype: GgmlType, w: &[u8], n_in: usize, n_out: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), n_in, "matvec: x length must equal n_in");
    assert_eq!(y.len(), n_out, "matvec: y length must equal n_out");
    assert!(dequant::supports(dtype), "matvec: unsupported weight dtype {dtype}");

    let (blk_elems, blk_bytes) = dtype.block().expect("supported dtype has a block size");
    let blk_elems = blk_elems as usize;
    assert_eq!(n_in % blk_elems, 0, "matvec: n_in ({n_in}) not a multiple of block ({blk_elems})");
    let row_bytes = (n_in / blk_elems) * blk_bytes as usize;

    // One output row: dequantize weight row `o` into `scratch`, then dot with `x`.
    let compute_row = |o: usize, scratch: &mut [f32]| -> f32 {
        let off = o * row_bytes;
        dequant::dequantize_into(dtype, &w[off..off + row_bytes], scratch);
        dot(scratch, x)
    };

    if n_out < PAR_MIN_ROWS {
        let mut scratch = vec![0.0f32; n_in];
        for (o, yo) in y.iter_mut().enumerate() {
            *yo = compute_row(o, &mut scratch);
        }
    } else {
        // Rows are independent; each worker keeps its own scratch buffer (allocated once
        // per bout of work and reused across that bout's rows).
        y.par_iter_mut().enumerate().for_each_init(
            || vec![0.0f32; n_in],
            |scratch, (o, yo)| *yo = compute_row(o, scratch),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn dot_basic() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    fn matvec_f32_small() {
        // W = [in=2, out=3], rows (by output) [1,2], [3,4], [5,6]; x = [1,1].
        let w = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = [1.0f32, 1.0];
        let mut y = [0.0f32; 3];
        matvec(GgmlType::F32, &w, 2, 3, &x, &mut y);
        assert_eq!(y, [3.0, 7.0, 11.0]);
    }

    #[test]
    fn matvec_f32_selects_with_basis_vector() {
        // x = e1 picks out column 1 of each row: [2, 4, 6].
        let w = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = [0.0f32, 1.0];
        let mut y = [0.0f32; 3];
        matvec(GgmlType::F32, &w, 2, 3, &x, &mut y);
        assert_eq!(y, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn matvec_q4k_single_row() {
        // One Q4_K row (n_in=256, n_out=1): d=1, sub-block 0 sc=1, qs[0] low nibble=7
        // -> dequantized row = [7, 0, 0, ...]; x = e0 -> y[0] = 7.
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[4] = 1; // sc for sub-block 0
        block[16] = 0x07; // qs[0]
        let mut x = vec![0.0f32; 256];
        x[0] = 1.0;
        let mut y = [0.0f32; 1];
        matvec(GgmlType::Q4_K, &block, 256, 1, &x, &mut y);
        assert_eq!(y[0], 7.0);
    }
}
