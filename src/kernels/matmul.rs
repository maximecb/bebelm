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

/// Over 32 weights from one nibble half of `q` (low if `!high`, else the high nibble) and
/// the matching 32 activations, return `(Σ nibble_i · x_i, Σ x_i)` — the two sums the Q4_K
/// factoring below needs. The dot/sum accumulate in `f32x8`; the per-byte nibble mask/shift
/// stays scalar (portable SIMD can't widen `u8`→`f32` lanes without a scalar gather).
#[inline]
fn nibble_dot32(q: &[u8], x: &[f32], high: bool) -> (f32, f32) {
    let mut qx = f32x8::splat(0.0);
    let mut xs = f32x8::splat(0.0);
    for k in 0..4 {
        let b = &q[k * 8..k * 8 + 8];
        let nib = if high {
            f32x8::from([
                (b[0] >> 4) as f32, (b[1] >> 4) as f32, (b[2] >> 4) as f32, (b[3] >> 4) as f32,
                (b[4] >> 4) as f32, (b[5] >> 4) as f32, (b[6] >> 4) as f32, (b[7] >> 4) as f32,
            ])
        } else {
            f32x8::from([
                (b[0] & 0xf) as f32, (b[1] & 0xf) as f32, (b[2] & 0xf) as f32, (b[3] & 0xf) as f32,
                (b[4] & 0xf) as f32, (b[5] & 0xf) as f32, (b[6] & 0xf) as f32, (b[7] & 0xf) as f32,
            ])
        };
        let xv = load8(&x[k * 8..]);
        qx = nib.mul_add(xv, qx);
        xs += xv;
    }
    (qx.reduce_add(), xs.reduce_add())
}

/// Fused dequantize-and-dot of one 144-byte Q4_K super-block against the matching 256
/// activations `x`: returns `Σ_i w_i · x[i]` without materializing the dequantized weights.
///
/// Uses `Σ (d·q − min)·x = d·Σ(q·x) − min·Σx`, so each sub-block's scale/min apply once
/// (not per weight). Block layout (see `dequant`'s module doc): `d:f16  dmin:f16
/// scales:u8[12]  qs:u8[128]`; the 4 chunks of 32 packed bytes each yield a low-nibble then
/// a high-nibble sub-block, matching `dequant::dequantize_q4_k_block`'s output ordering.
fn dot_q4k_block(block: &[u8], x: &[f32]) -> f32 {
    let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut sum = 0.0f32;
    for chunk in 0..4 {
        let (sc1, m1) = dequant::get_scale_min_k4(2 * chunk, scales);
        let (sc2, m2) = dequant::get_scale_min_k4(2 * chunk + 1, scales);
        let q = &qs[chunk * 32..chunk * 32 + 32];

        let (qx_lo, xsum_lo) = nibble_dot32(q, &x[chunk * 64..], false);
        let (qx_hi, xsum_hi) = nibble_dot32(q, &x[chunk * 64 + 32..], true);
        sum += (d * sc1 as f32) * qx_lo - (dmin * m1 as f32) * xsum_lo;
        sum += (d * sc2 as f32) * qx_hi - (dmin * m2 as f32) * xsum_hi;
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

    // One output row. Q4_K (the bulk of the weights) is fused: each 256-weight block
    // dequantizes straight into the dot, so it needs no scratch. Other dtypes dequantize
    // the whole row into `scratch`, then dot.
    let blk_bytes = blk_bytes as usize;
    let compute_row = |o: usize, scratch: &mut [f32]| -> f32 {
        let row = &w[o * row_bytes..(o + 1) * row_bytes];
        if dtype == GgmlType::Q4_K {
            row.chunks_exact(blk_bytes)
                .zip(x.chunks_exact(blk_elems))
                .map(|(blk, xb)| dot_q4k_block(blk, xb))
                .sum()
        } else {
            dequant::dequantize_into(dtype, row, scratch);
            dot(scratch, x)
        }
    };

    // Fused Q4_K rows ignore the scratch buffer, so don't allocate one for them.
    let scratch_len = if dtype == GgmlType::Q4_K { 0 } else { n_in };
    if n_out < PAR_MIN_ROWS {
        let mut scratch = vec![0.0f32; scratch_len];
        for (o, yo) in y.iter_mut().enumerate() {
            *yo = compute_row(o, &mut scratch);
        }
    } else {
        // Rows are independent; each worker keeps its own scratch buffer (allocated once
        // per bout of work and reused across that bout's rows).
        y.par_iter_mut().enumerate().for_each_init(
            || vec![0.0f32; scratch_len],
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
    fn matvec_q4k_fused_matches_dequant() {
        // A non-trivial Q4_K block (varied scales/mins/nibbles) dotted against varied x:
        // the fused path must match dequantize-then-dot to f32 tolerance.
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        block[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // dmin = 0.25
        for (j, b) in block[4..16].iter_mut().enumerate() {
            *b = (j * 17 + 5) as u8;
        }
        for (i, b) in block[16..144].iter_mut().enumerate() {
            *b = (i * 37 + 11) as u8;
        }
        let x: Vec<f32> = (0..256).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        let weights = dequant::dequantize(GgmlType::Q4_K, &block, 256);
        let reference: f32 = weights.iter().zip(&x).map(|(&w, &xi)| w * xi).sum();

        let mut y = [0.0f32];
        matvec(GgmlType::Q4_K, &block, 256, 1, &x, &mut y);
        assert!((y[0] - reference).abs() <= 1e-3 * reference.abs().max(1.0), "{} vs {}", y[0], reference);
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
