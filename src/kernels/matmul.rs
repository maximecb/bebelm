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
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
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

// --- Q8-activation integer dot (opt 9h) ---
//
// The quantized matvec's bottleneck is the f32 dot, not the unpack. Quantizing the *input*
// vector to int8 once per matmul lets each weight row dot in the integer domain — replacing
// the f32 FMA with a fused int8 dot-product. Activations stay f32 everywhere else.

/// One activation vector quantized to Q8: int8 quants, one f32 `scale` per 256-block, and the
/// sum of quants per 32-wide sub-block (for Q4_K's `min` term, which needs `Σ q_x`). Built by
/// [`quantize_q8`] and consumed via [`FusedJob::qx`]; its internals are an implementation detail.
pub struct Q8Vec {
    q: Vec<i8>,
    scales: Vec<f32>,
    sums: Vec<i32>,
}

/// Quantize activations to Q8: per 256-block `scale = max|x|/127`, `q = round(x/scale)` clamped
/// to ±127. `x.len()` must be a multiple of 256 (true for K-quant `n_in`).
pub fn quantize_q8(x: &[f32]) -> Q8Vec {
    let nblocks = x.len() / 256;
    let mut q = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; nblocks];
    let mut sums = vec![0i32; x.len() / 32];
    for b in 0..nblocks {
        let xb = &x[b * 256..b * 256 + 256];
        let amax = xb.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        scales[b] = amax / 127.0;
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        let qb = &mut q[b * 256..b * 256 + 256];
        for (qi, &v) in qb.iter_mut().zip(xb) {
            *qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        for (s, sum) in sums[b * 8..b * 8 + 8].iter_mut().enumerate() {
            *sum = qb[s * 32..s * 32 + 32].iter().map(|&v| v as i32).sum();
        }
    }
    Q8Vec { q, scales, sums }
}

/// `Σ_{i=0..32} nib_i · qx_i`, where `nib` are the low (or high) nibbles of the 32 bytes `q`
/// and `qx` are 32 int8 activations. The only arch-specific kernel: aarch64 uses the `sdot`
/// int8 dot-product instruction; elsewhere `wide` widens to i16 and uses `i16x16::dot`
/// (`pmaddwd` on x86/AVX2). `#[inline(always)]` so it folds into the per-block loop.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn nibble_idot32(q: &[u8], qx: &[i8], high: bool) -> i32 {
    use core::arch::aarch64::*;
    use core::arch::asm;
    // SAFETY: aarch64 implies NEON; `dotprod` is in this target's features (default on Apple
    // Silicon) so `sdot` is valid. Callers always pass ≥ 32 bytes of `q` and `qx`. The `sdot`
    // *intrinsic* (`vdotq_s32`) is still nightly-gated (`stdarch_neon_dotprod`), so we emit the
    // instruction with stable inline asm; the load/mask use stable NEON intrinsics.
    unsafe {
        let mut acc = vdupq_n_s32(0);
        for c in 0..2 {
            let bytes = vld1q_u8(q.as_ptr().add(c * 16));
            let nib = if high { vshrq_n_u8::<4>(bytes) } else { vandq_u8(bytes, vdupq_n_u8(0x0f)) };
            let w = vreinterpretq_s8_u8(nib);
            let xv = vld1q_s8(qx.as_ptr().add(c * 16));
            asm!(
                "sdot {acc:v}.4s, {w:v}.16b, {xv:v}.16b",
                acc = inout(vreg) acc,
                w = in(vreg) w,
                xv = in(vreg) xv,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        vaddvq_s32(acc)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn nibble_idot32(q: &[u8], qx: &[i8], high: bool) -> i32 {
    use wide::{i16x16, i32x8, i8x16, u8x16};
    const LOW: i16x16 = i16x16::new([0x0f; 16]);
    let mut acc = i32x8::new([0; 8]);
    for c in 0..2 {
        let bytes: [u8; 16] = q[c * 16..c * 16 + 16].try_into().unwrap();
        let w = i16x16::from(u8x16::new(bytes));
        let nib = if high { (w >> 4) & LOW } else { w & LOW };
        let xb: [i8; 16] = qx[c * 16..c * 16 + 16].try_into().unwrap();
        acc = acc + nib.dot(i16x16::from(i8x16::new(xb)));
    }
    acc.reduce_add()
}

/// Q8-activation integer dot of one 144-byte Q4_K block against pre-quantized activations
/// (`qx` = 256 int8, `sx` = block scale, `sums` = the 8 per-32 sub-block sums of `qx`):
/// `Σ w·x = sx·(d·Σ_j sc_j·⟨q_w,q_x⟩_j − dmin·Σ_j m_j·Σq_x_j)`. The `⟨·,·⟩` are exact integer
/// dots; only the activations carry Q8 rounding error.
///
/// `sc_j·⟨q_w,q_x⟩_j` and `m_j·Σq_x_j` are integer products, so both sums accumulate in `i32`
/// (matching ggml) and convert to f32 only once at the end. `sd`'s worst case (sc≤63, |⟨⟩|≤32·15·127
/// over 8 sub-blocks ≈ 31M) far exceeds f32's 2²⁴ exact-integer ceiling, so f32 accumulation would
/// drop low bits; i32 (±2.1B) holds it exactly.
#[inline(always)]
fn dot_q4k_block_q8(block: &[u8], qx: &[i8], sx: f32, sums: &[i32]) -> f32 {
    let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut sd = 0i32; // Σ_j sc_j · ⟨q_w, q_x⟩_j  (exact)
    let mut sm = 0i32; // Σ_j m_j · Σ q_x_j         (exact)
    for c in 0..4 {
        let q = &qs[c * 32..c * 32 + 32];
        let (sc1, m1) = dequant::get_scale_min_k4(2 * c, scales);
        let (sc2, m2) = dequant::get_scale_min_k4(2 * c + 1, scales);
        let lo = nibble_idot32(q, &qx[(2 * c) * 32..], false);
        let hi = nibble_idot32(q, &qx[(2 * c + 1) * 32..], true);
        sd += sc1 as i32 * lo + sc2 as i32 * hi;
        sm += m1 as i32 * sums[2 * c] + m2 as i32 * sums[2 * c + 1];
    }
    sx * (d * sd as f32 - dmin * sm as f32)
}

/// Q8-activation integer dot of a whole Q4_K weight row against pre-quantized activations.
#[inline(always)]
fn dot_q4k_row_q8(row: &[u8], a: &Q8Vec) -> f32 {
    row.chunks_exact(144)
        .enumerate()
        .map(|(b, blk)| dot_q4k_block_q8(blk, &a.q[b * 256..b * 256 + 256], a.scales[b], &a.sums[b * 8..b * 8 + 8]))
        .sum()
}

/// Weighted sub-block integer dot of one unpacked Q4_K block: `Σ_{s=0..8} sc[s]·⟨nib_s, qx_s⟩`,
/// where `nib`/`qx` are 256 int8 weights/activations in 8 contiguous 32-wide sub-blocks and `sc`
/// the 8 sub-block scales. The per-sub-block dot accumulates into a **vector** accumulator scaled
/// by `sc[s]` (one `vmlaq`/lane-multiply), and the horizontal reduction runs **once** at the end —
/// not once per sub-block as a naive `Σ sc·idot` would — which is what the [tiled
/// kernel](dot_q4k_rowtile_q8_batch) needs to expose ILP. The total is an exact integer, so the
/// result equals the per-sub-block form bit-for-bit. aarch64 emits `sdot`; elsewhere `wide` widens
/// to i16 and uses `i16x16::dot`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn wsd_q4k(nib: &[i8], qx: &[i8], sc: &[i32]) -> i32 {
    use core::arch::aarch64::*;
    use core::arch::asm;
    // SAFETY: aarch64 implies NEON + `dotprod` (Apple Silicon default), so `sdot` is valid; see
    // `nibble_idot32`. The intrinsic is still nightly-gated, so emit `sdot` via stable inline asm.
    unsafe {
        let mut vacc = vdupq_n_s32(0);
        // `s` drives both the slice index and the load pointer offsets, so enumerate doesn't fit.
        #[allow(clippy::needless_range_loop)]
        for s in 0..8 {
            let n0 = vld1q_s8(nib.as_ptr().add(s * 32));
            let n1 = vld1q_s8(nib.as_ptr().add(s * 32 + 16));
            let x0 = vld1q_s8(qx.as_ptr().add(s * 32));
            let x1 = vld1q_s8(qx.as_ptr().add(s * 32 + 16));
            let mut t = vdupq_n_s32(0);
            asm!(
                "sdot {t:v}.4s, {n:v}.16b, {x:v}.16b",
                t = inout(vreg) t, n = in(vreg) n0, x = in(vreg) x0,
                options(pure, nomem, nostack, preserves_flags),
            );
            asm!(
                "sdot {t:v}.4s, {n:v}.16b, {x:v}.16b",
                t = inout(vreg) t, n = in(vreg) n1, x = in(vreg) x1,
                options(pure, nomem, nostack, preserves_flags),
            );
            vacc = vmlaq_s32(vacc, t, vdupq_n_s32(sc[s]));
        }
        vaddvq_s32(vacc)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn wsd_q4k(nib: &[i8], qx: &[i8], sc: &[i32]) -> i32 {
    use wide::{i16x16, i32x8, i8x16};
    let mut acc = i32x8::new([0; 8]);
    #[allow(clippy::needless_range_loop)]
    for s in 0..8 {
        let mut sub = i32x8::new([0; 8]);
        for c in 0..2 {
            let av: [i8; 16] = nib[s * 32 + c * 16..s * 32 + c * 16 + 16].try_into().unwrap();
            let bv: [i8; 16] = qx[s * 32 + c * 16..s * 32 + c * 16 + 16].try_into().unwrap();
            sub = sub + i16x16::from(i8x16::new(av)).dot(i16x16::from(i8x16::new(bv)));
        }
        acc = acc + sub * i32x8::splat(sc[s]);
    }
    acc.reduce_add()
}

/// Batched Q8-activation integer dot of one Q4_K weight row against **all** `q8s` token columns,
/// accumulating each token's result into `out[t]`. Each 256-weight block's 4-bit quants are
/// unpacked to int8 **once** (into a 256-byte stack buffer, sub-block contiguous to match
/// [`wsd_q4k`]'s pairing) and then dotted against every token — amortizing the K-quant unpack
/// over the batch (the prefill GEMM's compute win). The per-(row, token) arithmetic is identical
/// to [`dot_q4k_row_q8`] (exact integer dots, same i32 accumulation, same f32 scaling), so the
/// output is **bit-for-bit** equal to calling it per token. `out` must be pre-zeroed, length
/// `q8s.len()`.
fn dot_q4k_row_q8_batch(row: &[u8], q8s: &[Q8Vec], out: &mut [f32]) {
    for (b, block) in row.chunks_exact(144).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];

        // Unpack the block's 256 nibbles once. Sub-block 2c = low nibbles of qs chunk c, 2c+1 =
        // high nibbles, each paired index-for-index with the matching 32 activations.
        let mut nib = [0i8; 256];
        for c in 0..4 {
            let q = &qs[c * 32..c * 32 + 32];
            for (i, &byte) in q.iter().enumerate() {
                nib[2 * c * 32 + i] = (byte & 0x0f) as i8;
                nib[(2 * c + 1) * 32 + i] = (byte >> 4) as i8;
            }
        }
        let mut sc = [0i32; 8];
        let mut m = [0i32; 8];
        for s in 0..8 {
            let (a, b) = dequant::get_scale_min_k4(s, scales);
            sc[s] = a as i32;
            m[s] = b as i32;
        }

        for (q8, o) in q8s.iter().zip(out.iter_mut()) {
            let qx = &q8.q[b * 256..b * 256 + 256];
            let sums = &q8.sums[b * 8..b * 8 + 8];
            let sd = wsd_q4k(&nib, qx, &sc);
            let sm: i32 = (0..8).map(|s| m[s] * sums[s]).sum();
            *o += q8.scales[b] * (d * sd as f32 - dmin * sm as f32);
        }
    }
}

/// Over one 16-weight Q6_K sub-block, return `Σ (q_i − 32)·x_i` — the `−32` recentering
/// folded into each lane. `ql`/`qh` are the current half's slices; `(ql_off, high, shift)`
/// pick this group's `ql` nibble and `qh` 2-bit field (see [`dot_q6k_block`]); `l_start` is
/// the sub-block's offset (0 or 16) into the half's 32-wide index `l`.
#[inline(always)]
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
#[inline(always)]
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

/// Batched Q6_K matmul of one weight row against **all** `n_tokens` token-major columns of `x`
/// (token `t`'s block `b` is `x[t*n_in + b*256..]`), accumulating into `out[t]`. Each 256-weight
/// block is unpacked **once** to its `(q−32)` f32 values (in [`dot_q6k_block`]'s sub-block order,
/// so each 16-wide sub-block lines up with a contiguous 16-wide `x` slice) and then dotted against
/// every token — hoisting Q6_K's costly per-element 6-bit unpack out of the per-token loop, which
/// is the bulk of the f32 fused path's cost. Reuses [`dot`] for the 16-wide sub-block dot, whose
/// reduction matches [`q6_dot16`] exactly, so the result is **bit-for-bit** equal to
/// [`fused_row_dot`] per token. `out` must be pre-zeroed; its length is the token count.
fn dot_q6k_row_batch(row: &[u8], x: &[f32], n_in: usize, out: &mut [f32]) {
    // Same group layout as `dot_q6k_block`: (ql byte offset in the half, high nibble?, qh shift).
    const GROUPS: [(usize, bool, u32); 4] = [(0, false, 0), (32, false, 2), (0, true, 4), (32, true, 6)];
    for (b, block) in row.chunks_exact(210).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let ql_all = &block[0..128];
        let qh_all = &block[128..192];
        let sc_all = &block[192..208];

        // Unpack the block's 16 sub-blocks (16 weights each) once: qf = (q−32) as f32, sc = scale.
        let mut qf = [0.0f32; 256];
        let mut sc = [0.0f32; 16];
        let mut k = 0;
        for n in 0..2 {
            let ql = &ql_all[n * 64..n * 64 + 64];
            let qh = &qh_all[n * 32..n * 32 + 32];
            let scn = &sc_all[n * 8..n * 8 + 8];
            for (g, &(ql_off, high, shift)) in GROUPS.iter().enumerate() {
                for sub in 0..2 {
                    sc[k] = scn[2 * g + sub] as i8 as f32;
                    let l_start = sub * 16;
                    for (i, qfi) in qf[k * 16..k * 16 + 16].iter_mut().enumerate() {
                        let li = l_start + i;
                        let low = if high { (ql[ql_off + li] >> 4) as i32 } else { (ql[ql_off + li] & 0x0f) as i32 };
                        let hi = ((qh[li] >> shift) & 3) as i32;
                        *qfi = ((low | (hi << 4)) - 32) as f32;
                    }
                    k += 1;
                }
            }
        }

        for (t, o) in out.iter_mut().enumerate() {
            let xt = &x[t * n_in + b * 256..t * n_in + b * 256 + 256];
            let mut acc = 0.0f32;
            for kk in 0..16 {
                acc += sc[kk] * dot(&qf[kk * 16..kk * 16 + 16], &xt[kk * 16..kk * 16 + 16]);
            }
            *o += d * acc;
        }
    }
}

/// Dot one **fused** K-quant weight row (`row` = that output row's packed Q4_K/Q6_K bytes)
/// with `x`, dequantizing each 256-weight block straight into the dot (no scratch). Panics
/// for dtypes without a fused path. `row` and `x` must span the same whole number of blocks.
#[inline]
pub fn fused_row_dot(dtype: GgmlType, row: &[u8], x: &[f32]) -> f32 {
    let (blk_elems, blk_bytes) = dtype.block().expect("fused dtype has a block size");
    let blocks = || row.chunks_exact(blk_bytes as usize).zip(x.chunks_exact(blk_elems as usize));
    match dtype {
        GgmlType::Q4_K => blocks().map(|(blk, xb)| dot_q4k_block(blk, xb)).sum(),
        GgmlType::Q6_K => blocks().map(|(blk, xb)| dot_q6k_block(blk, xb)).sum(),
        other => panic!("fused_row_dot: {other} has no fused path"),
    }
}

/// Weight rows computed together by [`dot_q4k_rowtile_q8_batch`] — the register/ILP tile width.
/// Four independent dot chains hide the int8-dot + horizontal-reduce latency that bottlenecks a
/// single row; all the K-quant matmul shapes (`n_out` ∈ {512, 2048, 1792, 6144, 7168}) are
/// multiples of 4, so the remainder path is rarely taken.
const Q4K_ROW_TILE: usize = 4;

/// Blocked Q4_K matmul micro-kernel: dot `Q4K_ROW_TILE` consecutive weight rows (`rows` =
/// `TILE × row_bytes`) against **all** `q8s` token columns, writing row `r`, token `c` to
/// `out[r*n_tokens + c]`. Processing the tile's rows together loads each token's Q8 block once and
/// reuses it across the tile, and the `TILE` independent dot chains expose instruction-level
/// parallelism to overlap the int8-dot latency. Each `(row, token)` result is computed by the
/// identical arithmetic of [`dot_q4k_row_q8_batch`], so the output is **bit-for-bit** equal.
/// `out` must be pre-zeroed, length `TILE * n_tokens`.
fn dot_q4k_rowtile_q8_batch(rows: &[u8], row_bytes: usize, n_tokens: usize, q8s: &[Q8Vec], out: &mut [f32]) {
    const TILE: usize = Q4K_ROW_TILE;
    let n_blocks = row_bytes / 144;
    for b in 0..n_blocks {
        // Decode each tile row's block b once: unpack its 256 nibbles (sub-block contiguous, to
        // match `idot32`) and its 8 (scale, min) pairs + the two super-scales.
        let mut nib = [[0i8; 256]; TILE];
        let mut sc = [[0i32; 8]; TILE];
        let mut m = [[0i32; 8]; TILE];
        let mut d = [0.0f32; TILE];
        let mut dmin = [0.0f32; TILE];
        for r in 0..TILE {
            let block = &rows[r * row_bytes + b * 144..r * row_bytes + b * 144 + 144];
            d[r] = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            dmin[r] = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales = &block[4..16];
            let qs = &block[16..144];
            for c in 0..4 {
                let q = &qs[c * 32..c * 32 + 32];
                for (i, &byte) in q.iter().enumerate() {
                    nib[r][2 * c * 32 + i] = (byte & 0x0f) as i8;
                    nib[r][(2 * c + 1) * 32 + i] = (byte >> 4) as i8;
                }
            }
            for s in 0..8 {
                let (a, bb) = dequant::get_scale_min_k4(s, scales);
                sc[r][s] = a as i32;
                m[r][s] = bb as i32;
            }
        }

        for (c, q8) in q8s.iter().enumerate() {
            let qx = &q8.q[b * 256..b * 256 + 256];
            let sums = &q8.sums[b * 8..b * 8 + 8];
            let scale = q8.scales[b];
            // TILE independent (row) dot chains over this token's shared Q8 block — each with its
            // own deferred-reduction accumulator, so their `sdot`/`vmlaq` chains overlap.
            for r in 0..TILE {
                let sd = wsd_q4k(&nib[r], qx, &sc[r]);
                let sm: i32 = (0..8).map(|s| m[r][s] * sums[s]).sum();
                out[r * n_tokens + c] += scale * (d[r] * sd as f32 - dmin[r] * sm as f32);
            }
        }
    }
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
    let fused = matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K);
    // Q4_K uses the Q8-activation integer dot: quantize x once, then dot it (read-only,
    // shared across rows) against every weight row. Q6_K keeps the f32 fused dot for now;
    // other dtypes dequantize a row into `scratch` then dot.
    let q8 = (dtype == GgmlType::Q4_K).then(|| quantize_q8(x));
    let compute_row = |o: usize, scratch: &mut [f32]| -> f32 {
        let row = &w[o * row_bytes..(o + 1) * row_bytes];
        match dtype {
            GgmlType::Q4_K => dot_q4k_row_q8(row, q8.as_ref().unwrap()),
            GgmlType::Q6_K => fused_row_dot(dtype, row, x),
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

/// Batched matmul `Y = W·X`: apply one weight matrix to `n_tokens` activation columns at once.
///
/// `W` is `[n_in, n_out]` quantized as `dtype`; `x` and `y` are **token-major** — token `t`'s
/// input is `x[t*n_in..][..n_in]` and its output `y[t*n_out..][..n_out]`. The result is
/// **bit-for-bit identical** to calling [`matvec`] once per token: each output is the same
/// per-(row, token) dot, in the same accumulation order. The win is structural — each weight
/// row is read (and, for F32/F16, dequantized) **once** and reused across all `n_tokens`
/// columns, instead of re-reading the whole matrix per token. This amortizes weight memory
/// traffic and the fork/join over the batch (the prefill GEMM, opt 9f).
///
/// Parallelism is over output rows (each row independent → order-preserving), so the result is
/// thread-count-independent just like [`matvec`]. `n_tokens == 1` delegates to [`matvec`] so the
/// decode path is untouched.
pub fn matmul(dtype: GgmlType, w: &[u8], n_in: usize, n_out: usize, x: &[f32], n_tokens: usize, y: &mut [f32]) {
    assert_eq!(x.len(), n_in * n_tokens, "matmul: x length must equal n_in * n_tokens");
    assert_eq!(y.len(), n_out * n_tokens, "matmul: y length must equal n_out * n_tokens");
    if n_tokens == 1 {
        matvec(dtype, w, n_in, n_out, x, y);
        return;
    }
    assert!(dequant::supports(dtype), "matmul: unsupported weight dtype {dtype}");
    let (blk_elems, blk_bytes) = dtype.block().expect("supported dtype has a block size");
    let blk_elems = blk_elems as usize;
    assert_eq!(n_in % blk_elems, 0, "matmul: n_in ({n_in}) not a multiple of block ({blk_elems})");
    let row_bytes = (n_in / blk_elems) * blk_bytes as usize;

    let col = |t: usize| &x[t * n_in..(t + 1) * n_in];

    // Compute into a feature-major scratch (`out_fm[o*n_tokens + t]`) so each row (or row tile)
    // owns a contiguous chunk — the only layout that lets rayon split rows without aliasing — then
    // transpose into the token-major `y`. `out_fm` is pre-zeroed; the Q4_K kernels accumulate.
    let mut out_fm = vec![0.0f32; n_out * n_tokens];
    let parallel = n_out >= PAR_MIN_ROWS;

    if dtype == GgmlType::Q4_K {
        // Q4_K takes the Q8-activation integer dot: quantize each token column once (shared,
        // read-only, across every weight row), then run the blocked micro-kernel over
        // `Q4K_ROW_TILE` rows at a time for ILP. The rare leftover rows use the single-row kernel.
        let q8: Vec<Q8Vec> = (0..n_tokens).map(|t| quantize_q8(col(t))).collect();
        let fill_tile = |ti: usize, chunk: &mut [f32]| {
            let r0 = ti * Q4K_ROW_TILE;
            let nrows = chunk.len() / n_tokens;
            if nrows == Q4K_ROW_TILE {
                dot_q4k_rowtile_q8_batch(&w[r0 * row_bytes..(r0 + nrows) * row_bytes], row_bytes, n_tokens, &q8, chunk);
            } else {
                for r in 0..nrows {
                    let row = &w[(r0 + r) * row_bytes..(r0 + r + 1) * row_bytes];
                    dot_q4k_row_q8_batch(row, &q8, &mut chunk[r * n_tokens..(r + 1) * n_tokens]);
                }
            }
        };
        if parallel {
            out_fm
                .par_chunks_mut(Q4K_ROW_TILE * n_tokens)
                .enumerate()
                .for_each(|(ti, chunk)| fill_tile(ti, chunk));
        } else {
            out_fm.chunks_mut(Q4K_ROW_TILE * n_tokens).enumerate().for_each(|(ti, chunk)| fill_tile(ti, chunk));
        }
    } else {
        // One weight row dotted against all token columns. Q6_K unpacks each block to f32 once and
        // reuses it across tokens; F32/F16 dequantize the row into `scratch` once, then dot.
        let scratch_len = if dtype == GgmlType::Q6_K { 0 } else { n_in };
        let compute_row = |o: usize, dst: &mut [f32], scratch: &mut [f32]| {
            let row = &w[o * row_bytes..(o + 1) * row_bytes];
            if dtype == GgmlType::Q6_K {
                dot_q6k_row_batch(row, x, n_in, dst);
            } else {
                dequant::dequantize_into(dtype, row, scratch);
                for (t, d) in dst.iter_mut().enumerate() {
                    *d = dot(scratch, col(t));
                }
            }
        };
        if parallel {
            out_fm.par_chunks_mut(n_tokens).enumerate().for_each_init(
                || vec![0.0f32; scratch_len],
                |scratch, (o, dst)| compute_row(o, dst, scratch),
            );
        } else {
            let mut scratch = vec![0.0f32; scratch_len];
            for (o, dst) in out_fm.chunks_mut(n_tokens).enumerate() {
                compute_row(o, dst, &mut scratch);
            }
        }
    }

    for o in 0..n_out {
        for t in 0..n_tokens {
            y[t * n_out + o] = out_fm[o * n_tokens + t];
        }
    }
}

/// One fused (Q4_K/Q6_K) matvec `y = W·x` for [`matvec_fused_batch`]: `w` is `n_out`
/// contiguous weight rows of `n_in` weights each, dotted against `x` (`x.len() == n_in`).
pub struct FusedJob<'a> {
    pub dtype: GgmlType,
    pub w: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
    pub x: &'a [f32],
    /// `x` pre-quantized via [`quantize_q8`] (so several jobs sharing one `x` quantize it once).
    /// When `Some` — only valid for Q4_K — rows take the Q8-activation integer dot; `None` rows
    /// take the f32 fused dot over `x`. Must be the quantization of *this* job's `x`.
    pub qx: Option<&'a Q8Vec>,
}

/// Run several fused matvecs as a **single** parallel region over all their output rows
/// combined, writing job `j`'s `n_out` results into `out` at the running offset (so `out`
/// must be `Σ n_out` long, jobs in order). Pooling the rows lets a few small matrices — e.g.
/// the handful of selected MoE experts — saturate every core under one fork/join, instead of
/// underutilizing it (and paying a join) one small matvec at a time. Each row uses the same
/// per-row kernel a standalone call would: Q4_K jobs carrying a pre-quantized [`qx`](FusedJob::qx)
/// take the Q8-activation integer dot, all others the f32 fused dot — so the batch is identical
/// to running the jobs separately.
pub fn matvec_fused_batch(jobs: &[FusedJob], out: &mut [f32]) {
    // Per job: its row stride in bytes, and the first `out` row it owns (a prefix sum of
    // n_out). `starts` is ascending, so a row's owning job is a binary search away.
    let mut starts = Vec::with_capacity(jobs.len());
    let mut row_bytes = Vec::with_capacity(jobs.len());
    let mut total = 0usize;
    for j in jobs {
        let (blk_elems, blk_bytes) = j.dtype.block().expect("fused dtype has a block size");
        assert_eq!(j.x.len(), j.n_in, "matvec_fused_batch: x length must equal n_in");
        assert!(j.qx.is_none() || j.dtype == GgmlType::Q4_K, "matvec_fused_batch: qx only valid for Q4_K");
        assert_eq!(j.n_in % blk_elems as usize, 0, "matvec_fused_batch: n_in not a block multiple");
        starts.push(total);
        row_bytes.push((j.n_in / blk_elems as usize) * blk_bytes as usize);
        total += j.n_out;
    }
    assert_eq!(out.len(), total, "matvec_fused_batch: out length must equal Σ n_out");

    let row = |gr: usize, o: &mut f32| {
        // The owning job is the last `start` at or before this global row index.
        let j = starts.partition_point(|&s| s <= gr) - 1;
        let local = gr - starts[j];
        let job = &jobs[j];
        let rb = &job.w[local * row_bytes[j]..(local + 1) * row_bytes[j]];
        *o = match job.qx {
            Some(q8) => dot_q4k_row_q8(rb, q8),
            None => fused_row_dot(job.dtype, rb, job.x),
        };
    };
    if total < PAR_MIN_ROWS {
        out.iter_mut().enumerate().for_each(|(gr, o)| row(gr, o));
    } else {
        out.par_iter_mut().enumerate().for_each(|(gr, o)| row(gr, o));
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
    fn dot_covers_simd_and_tail_lengths() {
        // `dot_basic` only reaches the scalar tail (len 3). These lengths exercise the
        // 32-wide 4-accumulator loop, the 8-wide loop, and their seams with the tail.
        // The SIMD reduction reorders the sum (FMA rounding), so compare to a plain
        // left-to-right dot with tolerance rather than bit-for-bit.
        fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(&x, &y)| x * y).sum()
        }
        for n in [8usize, 11, 16, 19, 32, 35, 40, 64, 100] {
            let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 1.0).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
            let got = dot(&a, &b);
            let want = scalar_dot(&a, &b);
            assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "n={n}: {got} vs {want}");
        }
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

    /// A non-trivial Q4_K block + varied x, reused by the f32-path and Q8-path tests below.
    fn q4k_test_block() -> (Vec<u8>, Vec<f32>) {
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
        (block, x)
    }

    #[test]
    fn fused_row_dot_q4k_matches_dequant() {
        // The f32 fused path (still used by `matvec_fused_batch`) must match dequant-then-dot.
        let (block, x) = q4k_test_block();
        let weights = dequant::dequantize(GgmlType::Q4_K, &block, 256);
        let reference: f32 = weights.iter().zip(&x).map(|(&w, &xi)| w * xi).sum();
        let got = fused_row_dot(GgmlType::Q4_K, &block, &x);
        assert!((got - reference).abs() <= 1e-3 * reference.abs().max(1.0), "{got} vs {reference}");
    }

    #[test]
    fn matvec_q4k_q8_matches_quantized_reference() {
        // matvec's Q4_K path is the Q8-activation integer dot. It should equal the exact dot of
        // the dequantized weights with the Q8-rounded activations (`sx · q_x`) — i.e. the only
        // error vs true is the int8 quantization of x, not the integer dot itself.
        let (block, x) = q4k_test_block();
        let weights = dequant::dequantize(GgmlType::Q4_K, &block, 256);
        let a = quantize_q8(&x);
        let reference: f32 = weights.iter().zip(&a.q).map(|(&w, &q)| w * (a.scales[0] * q as f32)).sum();

        let mut y = [0.0f32];
        matvec(GgmlType::Q4_K, &block, 256, 1, &x, &mut y);
        assert!((y[0] - reference).abs() <= 1e-3 * reference.abs().max(1.0), "{} vs {reference}", y[0]);
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

    #[test]
    fn matvec_f16_small() {
        // Same shape as `matvec_f32_small`, but the weights are F16 (the dequant-into-scratch
        // path). f16 bit patterns for rows [1,2], [3,4], [5,6]; x = [1,1] -> [3, 7, 11].
        let halves = [0x3c00u16, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600]; // 1,2,3,4,5,6
        let w: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let x = [1.0f32, 1.0];
        let mut y = [0.0f32; 3];
        matvec(GgmlType::F16, &w, 2, 3, &x, &mut y);
        assert_eq!(y, [3.0, 7.0, 11.0]);
    }

    #[test]
    fn matvec_parallel_path_matches_per_row_dot() {
        // n_out >= PAR_MIN_ROWS forces the rayon path. Each output row is an independent
        // dot, so the result must be bit-for-bit equal to dotting each weight row with x
        // directly — exactly what the serial path computes (the module's threading claim).
        let n_in = 40usize; // 32 + 8: drives the SIMD dot too
        let n_out = 128usize; // > PAR_MIN_ROWS (64)
        let w: Vec<f32> = (0..n_in * n_out).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let mut y = vec![0.0f32; n_out];
        matvec(GgmlType::F32, &f32_bytes(&w), n_in, n_out, &x, &mut y);
        for (o, &yo) in y.iter().enumerate() {
            assert_eq!(yo, dot(&w[o * n_in..(o + 1) * n_in], &x), "row {o}");
        }
    }

    /// A Q4_K weight matrix of `n_out` rows (each one 256-weight block), bytes varied by row.
    fn q4k_matrix(n_out: usize, seed: usize) -> Vec<u8> {
        let mut w = Vec::with_capacity(n_out * 144);
        for r in 0..n_out {
            let mut block = vec![0u8; 144];
            block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
            block[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // dmin = 0.25
            for (j, b) in block[4..16].iter_mut().enumerate() {
                *b = ((r + seed) * 7 + j * 17 + 5) as u8;
            }
            for (i, b) in block[16..144].iter_mut().enumerate() {
                *b = ((r + seed) * 3 + i * 37 + 11) as u8;
            }
            w.extend_from_slice(&block);
        }
        w
    }

    /// A Q6_K weight matrix of `n_out` rows (each one 256-weight block), bytes varied by row.
    fn q6k_matrix(n_out: usize, seed: usize) -> Vec<u8> {
        let mut w = Vec::with_capacity(n_out * 210);
        for r in 0..n_out {
            let mut block = vec![0u8; 210];
            block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
            for (i, b) in block[0..128].iter_mut().enumerate() {
                *b = ((r + seed) * 5 + i * 53 + 17) as u8; // ql
            }
            for (i, b) in block[128..192].iter_mut().enumerate() {
                *b = ((r + seed) * 11 + i * 97 + 5) as u8; // qh
            }
            for (j, b) in block[192..208].iter_mut().enumerate() {
                *b = ((r + seed).wrapping_mul(29).wrapping_add(j * 13 + 3)) as u8; // i8 scales
            }
            w.extend_from_slice(&block);
        }
        w
    }

    #[test]
    fn matvec_fused_batch_matches_separate_jobs() {
        // A batch must write each job's rows at its running offset, identical to running the
        // jobs one at a time with the same per-row kernel (bit-for-bit). The Q4_K job carries a
        // pre-quantized `qx`, so it must match the Q8 dot; the Q6_K job must match the f32 fused
        // dot. Two dtypes span differing row strides (144 vs 210 B), exercising the
        // partition_point row attribution; two sizes hit both the serial and parallel branches.
        let x1: Vec<f32> = (0..256).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let x2: Vec<f32> = (0..256).map(|i| ((i % 5) as f32 - 2.0) * 0.07).collect();
        let q8 = quantize_q8(&x1);

        // (3,5): total 8 < PAR_MIN_ROWS -> serial. (40,50): total 90 -> parallel.
        for &(n_a, n_b) in &[(3usize, 5usize), (40usize, 50usize)] {
            let wa = q4k_matrix(n_a, 1);
            let wb = q6k_matrix(n_b, 99);
            let jobs = vec![
                FusedJob { dtype: GgmlType::Q4_K, w: &wa, n_in: 256, n_out: n_a, x: &x1, qx: Some(&q8) },
                FusedJob { dtype: GgmlType::Q6_K, w: &wb, n_in: 256, n_out: n_b, x: &x2, qx: None },
            ];
            let mut out = vec![0.0f32; n_a + n_b];
            matvec_fused_batch(&jobs, &mut out);

            for o in 0..n_a {
                let row = &wa[o * 144..(o + 1) * 144];
                assert_eq!(out[o], dot_q4k_row_q8(row, &q8), "job A row {o}");
            }
            for o in 0..n_b {
                let row = &wb[o * 210..(o + 1) * 210];
                assert_eq!(out[n_a + o], fused_row_dot(GgmlType::Q6_K, row, &x2), "job B row {o}");
            }
        }
    }

    /// Scalar fp32 reference for one Q4_K block: fully dequantize then compute the true dot.
    /// Only exists in test builds — never pulled into release.
    fn ref_dot_q4k_block_fp32(block: &[u8], x: &[f32]) -> f32 {
        let mut weights = [0.0f32; 256];
        dequant::dequantize_q4_k_block(block, &mut weights);
        weights.iter().zip(x).map(|(&w, &xi)| w * xi).sum()
    }

    /// A deterministic, realistically-scaled Q4_K block + activations for the precision test.
    /// Unlike `q4k_test_block` (which maxes out the 6-bit sub-block scales, inflating the
    /// relative error), this uses a small super-block `d`/`dmin` and keeps every packed scale
    /// byte < 64, so the dequantized weights land in a transformer-like range (~±0.25) and the
    /// activations are O(1). A fixed-seed LCG fills the quants/scales/x so it's stable run-to-run.
    fn realistic_q4k_block() -> (Vec<u8>, Vec<f32>) {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };

        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&0x1000u16.to_le_bytes()); // d    ≈ 0.000488 (f16)
        block[2..4].copy_from_slice(&0x1c00u16.to_le_bytes()); // dmin ≈ 0.00391  (f16)
        // 6-bit packed scales/mins: every byte < 64 keeps the extracted sc/m modest (no
        // maxed-out scales) so d·sc·q stays in a realistic weight range.
        for b in block[4..16].iter_mut() {
            *b = (16 + next() % 32) as u8; // 16..47
        }
        // 4-bit quants: full nibble range, deterministic.
        for b in block[16..144].iter_mut() {
            *b = next() as u8;
        }
        // Activations: O(1), centered near zero — like a residual-stream vector.
        let x: Vec<f32> = (0..256).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
        (block, x)
    }

    #[test]
    fn dot_q4k_block_q8_precision_vs_fp32() {
        // The Q8-activation integer dot vs the true fp32 dot (fully dequantized weights · true
        // x). With realistic weight scales and O(1) activations, the only error is Q8
        // quantization of x; the integer dot itself is exact. Inputs are deterministic, so the
        // measured error is stable. Run single-threaded for determinism (single block anyway).
        let (block, x) = realistic_q4k_block();
        let reference = ref_dot_q4k_block_fp32(&block, &x);

        let a = quantize_q8(&x);
        let got = dot_q4k_block_q8(&block, &a.q[..256], a.scales[0], &a.sums[..8]);

        // Measured relative error here is 0.0041545 (≈0.42%), in line with ggml's Q4_K×Q8_K
        // accuracy. The whole test is deterministic and the result is bit-identical across
        // platforms (exact integer dot; deterministic IEEE scaling/reduction), so the bound is
        // set tight — ~8% over the measured value — to catch any precision regression early.
        let rel = (got - reference).abs() / reference.abs();
        assert!(
            rel < 0.0045,
            "dot_q4k_block_q8 vs fp32 ref: got={got:.6}, ref={reference:.6}, rel={rel:.7}"
        );
    }

    /// `matmul` over `n_tokens` columns must be bit-for-bit identical to calling `matvec` once
    /// per token — the module's core claim (loop reordering, not a numeric change). Checks every
    /// dtype path (Q4_K Q8-int, Q6_K f32-fused, F32 dequant) and both the serial and parallel
    /// row branches.
    fn assert_matmul_eq_per_token(dtype: GgmlType, w: &[u8], n_in: usize, n_out: usize, n_tokens: usize) {
        let x: Vec<f32> = (0..n_in * n_tokens).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let mut batched = vec![0.0f32; n_out * n_tokens];
        matmul(dtype, w, n_in, n_out, &x, n_tokens, &mut batched);
        for t in 0..n_tokens {
            let mut single = vec![0.0f32; n_out];
            matvec(dtype, w, n_in, n_out, &x[t * n_in..(t + 1) * n_in], &mut single);
            assert_eq!(&batched[t * n_out..(t + 1) * n_out], &single[..], "dtype {dtype}, token {t}");
        }
    }

    #[test]
    fn matmul_matches_per_token_matvec() {
        // n_out 50 < PAR_MIN_ROWS (serial), 100 > it (parallel); 4 tokens exercises the batch.
        for &n_out in &[50usize, 100] {
            assert_matmul_eq_per_token(GgmlType::Q4_K, &q4k_matrix(n_out, 3), 256, n_out, 4);
            assert_matmul_eq_per_token(GgmlType::Q6_K, &q6k_matrix(n_out, 7), 256, n_out, 4);
            let f32w = f32_bytes(&(0..2 * n_out).map(|i| (i as f32 - 3.0) * 0.1).collect::<Vec<_>>());
            assert_matmul_eq_per_token(GgmlType::F32, &f32w, 2, n_out, 4);
        }
    }

    #[test]
    fn matmul_single_token_matches_matvec() {
        // The n_tokens == 1 fast path must equal matvec exactly (it delegates).
        assert_matmul_eq_per_token(GgmlType::Q4_K, &q4k_matrix(80, 1), 256, 80, 1);
    }

    #[test]
    fn matvec_fused_batch_q4k_without_qx_falls_back_to_f32() {
        // A Q4_K job with no pre-quantized activations takes the f32 fused dot (the `None` arm),
        // identical to a standalone fused_row_dot.
        let x: Vec<f32> = (0..256).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let w = q4k_matrix(4, 7);
        let jobs = vec![FusedJob { dtype: GgmlType::Q4_K, w: &w, n_in: 256, n_out: 4, x: &x, qx: None }];
        let mut out = vec![0.0f32; 4];
        matvec_fused_batch(&jobs, &mut out);
        for o in 0..4 {
            assert_eq!(out[o], fused_row_dot(GgmlType::Q4_K, &w[o * 144..(o + 1) * 144], &x), "row {o}");
        }
    }
}
