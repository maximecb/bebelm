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

/// Over one 16-weight Q6_K sub-block, return `Σ (q_i − 32)·x_i` — the `−32` recentering
/// folded into each lane. `ql`/`qh` are the current half's slices; `(ql_off, high, shift)`
/// pick this group's `ql` nibble and `qh` 2-bit field (see [`dot_q6k_block`]); `l_start` is
/// the sub-block's offset (0 or 16) into the half's 32-wide index `l`.
#[inline]
fn q6_dot16(ql: &[u8], qh: &[u8], ql_off: usize, high: bool, shift: u32, l_start: usize, x: &[f32]) -> f32 {
    let mut qx = f32x8::splat(0.0);
    for c in 0..2 {
        let l = l_start + c * 8;
        let mut q = [0.0f32; 8];
        for (j, qj) in q.iter_mut().enumerate() {
            let li = l + j;
            let low = if high { (ql[ql_off + li] >> 4) as i32 } else { (ql[ql_off + li] & 0x0f) as i32 };
            let hi = ((qh[li] >> shift) & 3) as i32;
            *qj = ((low | (hi << 4)) - 32) as f32;
        }
        qx = f32x8::from(q).mul_add(load8(&x[c * 8..]), qx);
    }
    qx.reduce_add()
}

/// Fused dequantize-and-dot of one 210-byte Q6_K super-block against the matching 256
/// activations `x`: returns `Σ_i w_i · x[i]` without materializing the weights.
///
/// `w = d · sc_sub · (q − 32)`, with one i8 `sc` per 16 weights and one f16 `d` per block,
/// so `Σ w·x = d · Σ_sub sc_sub · Σ(q−32)·x` (block `d` factored out, applied once). Layout
/// (see `dequant`'s module doc): `ql:u8[128]  qh:u8[64]  scales:i8[16]  d:f16`. Each
/// 128-weight half splits into 4 groups of 32 (a low/high `ql` nibble + a 2-bit `qh` field),
/// each group into two 16-weight sub-blocks — matching `dequant::dequantize_q6_k_block`.
fn dot_q6k_block(block: &[u8], x: &[f32]) -> f32 {
    // (ql byte offset within the half, take ql's high nibble?, qh bit shift) per group.
    const GROUPS: [(usize, bool, u32); 4] = [(0, false, 0), (32, false, 2), (0, true, 4), (32, true, 6)];

    let d = dequant::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql_all = &block[0..128];
    let qh_all = &block[128..192];
    let sc_all = &block[192..208]; // i8 scales as raw bytes

    let mut acc = 0.0f32; // Σ_sub sc·Σ(q−32)·x; scaled by the common block d once at the end
    for n in 0..2 {
        let ql = &ql_all[n * 64..n * 64 + 64];
        let qh = &qh_all[n * 32..n * 32 + 32];
        let sc = &sc_all[n * 8..n * 8 + 8];
        let xh = &x[n * 128..]; // this half's 128 activations
        for (g, &(ql_off, high, shift)) in GROUPS.iter().enumerate() {
            let xg = &xh[g * 32..]; // this group's 32 activations
            for sub in 0..2 {
                let sc_s = sc[2 * g + sub] as i8 as f32;
                acc += sc_s * q6_dot16(ql, qh, ql_off, high, shift, sub * 16, &xg[sub * 16..]);
            }
        }
    }
    d * acc
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

    // One output row. The K-quants (Q4_K/Q6_K — the bulk of the weights) are fused: each
    // 256-weight block dequantizes straight into the dot, so they need no scratch. Other
    // dtypes (F32/F16 — e.g. the MoE router) dequantize the whole row into `scratch`, then dot.
    let blk_bytes = blk_bytes as usize;
    let fused = matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K);
    let compute_row = |o: usize, scratch: &mut [f32]| -> f32 {
        let row = &w[o * row_bytes..(o + 1) * row_bytes];
        let blocks = || row.chunks_exact(blk_bytes).zip(x.chunks_exact(blk_elems));
        match dtype {
            GgmlType::Q4_K => blocks().map(|(blk, xb)| dot_q4k_block(blk, xb)).sum(),
            GgmlType::Q6_K => blocks().map(|(blk, xb)| dot_q6k_block(blk, xb)).sum(),
            _ => {
                dequant::dequantize_into(dtype, row, scratch);
                dot(scratch, x)
            }
        }
    };

    // Fused rows ignore the scratch buffer, so don't allocate one for them.
    let scratch_len = if fused { 0 } else { n_in };
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
    fn matvec_q6k_fused_matches_dequant() {
        // A non-trivial Q6_K block (varied scales incl. negatives, varied quants) dotted
        // against varied x: the fused path must match dequantize-then-dot to f32 tolerance.
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        for (i, b) in block[0..128].iter_mut().enumerate() {
            *b = (i * 53 + 17) as u8; // ql
        }
        for (i, b) in block[128..192].iter_mut().enumerate() {
            *b = (i * 97 + 5) as u8; // qh
        }
        for (j, b) in block[192..208].iter_mut().enumerate() {
            *b = (j.wrapping_mul(29).wrapping_add(3)) as u8; // i8 scales (some negative)
        }
        let x: Vec<f32> = (0..256).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();

        let weights = dequant::dequantize(GgmlType::Q6_K, &block, 256);
        let reference: f32 = weights.iter().zip(&x).map(|(&w, &xi)| w * xi).sum();

        let mut y = [0.0f32];
        matvec(GgmlType::Q6_K, &block, 256, 1, &x, &mut y);
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
