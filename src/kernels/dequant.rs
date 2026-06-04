//! Dequantization kernels: GGML block formats -> `f32`.
//!
//! Q4_K and Q6_K follow ggml's reference `dequantize_row_*` exactly (see
//! `ggml/src/ggml-quants.c`). Block byte layouts (little-endian), field order as on disk:
//!
//! ```text
//! block_q4_K (144 B): d:f16  dmin:f16  scales:u8[12]  qs:u8[128]
//! block_q6_K (210 B): ql:u8[128]  qh:u8[64]  scales:i8[16]  d:f16
//! ```

use crate::tensor::GgmlType;

const QK_K: usize = 256; // weights per K-quant super-block
const Q4_K_BYTES: usize = 144;
const Q6_K_BYTES: usize = 210;

/// Convert an IEEE-754 half-precision bit pattern to `f32`.
///
/// Hand-rolled (exact, incl. subnormals/inf/nan) so we don't pull in a crate for the one
/// conversion we need — f16 only appears as the per-block scales inside Q4_K/Q6_K.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x03ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign // signed zero
        } else {
            // Subnormal: normalize into a f32 normal.
            let mut e: i32 = -1;
            let mut m = mant;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x0400 != 0 {
                    break;
                }
            }
            let mant32 = (m & 0x03ff) << 13;
            let exp32 = ((127 - 15 - e) as u32) << 23;
            sign | exp32 | mant32
        }
    } else if exp == 0x1f {
        // Inf / NaN: propagate the mantissa.
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        // Normal: rebias exponent (15 -> 127), shift mantissa (10 -> 23 bits).
        let exp32 = ((exp as i32 - 15 + 127) as u32) << 23;
        sign | exp32 | (mant << 13)
    };
    f32::from_bits(bits)
}

#[inline]
fn read_f16(bytes: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// Unpack the 6-bit sub-block scale `sc` and min `m` for sub-block `j` from the packed
/// 12-byte `scales` array of a Q4_K block (ggml's `get_scale_min_k4`).
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Dequantize one 144-byte Q4_K super-block into 256 `f32`s.
pub fn dequantize_q4_k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q4_K_BYTES);
    debug_assert_eq!(out.len(), QK_K);

    let d = read_f16(block, 0);
    let dmin = read_f16(block, 2);
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut oi = 0; // output position
    let mut is = 0; // sub-block scale index
    // 4 iterations, each producing 64 weights from 32 packed bytes.
    for chunk in 0..4 {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;

        let q = &qs[chunk * 32..chunk * 32 + 32];
        for &b in q {
            out[oi] = d1 * (b & 0x0f) as f32 - min1;
            oi += 1;
        }
        for &b in q {
            out[oi] = d2 * (b >> 4) as f32 - min2;
            oi += 1;
        }
        is += 2;
    }
}

/// Dequantize one 210-byte Q6_K super-block into 256 `f32`s.
// `qh[l] >> 0` is kept (vs. clippy's identity_op) so q1..q4 read as the same shifted
// pattern (>>0, >>2, >>4, >>6) as the ggml reference, which aids verification.
#[allow(clippy::identity_op)]
pub fn dequantize_q6_k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q6_K_BYTES);
    debug_assert_eq!(out.len(), QK_K);

    let d = read_f16(block, 208);
    let ql_all = &block[0..128];
    let qh_all = &block[128..192];
    let sc_all = &block[192..208]; // i8 scales stored as raw bytes

    // 2 iterations, each producing 128 weights.
    for n in 0..2 {
        let out_base = n * 128;
        let ql = &ql_all[n * 64..n * 64 + 64];
        let qh = &qh_all[n * 32..n * 32 + 32];
        let sc = &sc_all[n * 8..n * 8 + 8];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0f) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) - 32;
            let q2 = ((ql[l + 32] & 0x0f) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
            let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
            let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;
            out[out_base + l] = d * (sc[is] as i8) as f32 * q1 as f32;
            out[out_base + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            out[out_base + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            out[out_base + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

/// Whether [`dequantize`] can handle this dtype.
pub fn supports(dtype: GgmlType) -> bool {
    matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::Q4_K | GgmlType::Q6_K)
}

/// Dequantize a whole tensor of weights into the caller-provided `out` buffer.
///
/// Panics if `dtype` is not [`supports`]ed — callers should check first. `data` must be
/// the exact byte slice for the tensor (as sized by [`GgmlType::byte_size`]), and
/// `out.len()` the element count.
pub fn dequantize_into(dtype: GgmlType, data: &[u8], out: &mut [f32]) {
    match dtype {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(4)) {
                *o = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()));
            }
        }
        GgmlType::Q4_K => {
            for (block, dst) in data.chunks_exact(Q4_K_BYTES).zip(out.chunks_mut(QK_K)) {
                dequantize_q4_k_block(block, dst);
            }
        }
        GgmlType::Q6_K => {
            for (block, dst) in data.chunks_exact(Q6_K_BYTES).zip(out.chunks_mut(QK_K)) {
                dequantize_q6_k_block(block, dst);
            }
        }
        other => panic!("dequantize: unsupported type {other}"),
    }
}

/// Dequantize a whole tensor of `n_elements` weights into a fresh `Vec<f32>`.
pub fn dequantize(dtype: GgmlType, data: &[u8], n_elements: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_elements];
    dequantize_into(dtype, data, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        // smallest positive subnormal = 2^-24
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        // 0x3555 ~= 0.333251953125
        assert!((f16_to_f32(0x3555) - 0.333_251_95).abs() < 1e-6);
    }

    #[test]
    fn q4_k_block_known() {
        // d = 1.0, dmin = 0; sub-block 0: sc=3,m=0; sub-block 1: sc=5,m=0.
        let mut block = vec![0u8; Q4_K_BYTES];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        block[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0
        // scales: [sc0, sc1, .., m0, m1, ..] for j<4 -> scales[j], scales[j+4]
        block[4] = 3; // sc for sub-block 0
        block[5] = 5; // sc for sub-block 1
        block[8] = 0; // m for sub-block 0
        block[9] = 0; // m for sub-block 1
        block[16] = 0x21; // qs[0]: low nibble 1, high nibble 2

        let mut out = vec![0.0f32; QK_K];
        dequantize_q4_k_block(&block, &mut out);
        // out[0] = d*sc0*(low nibble) - dmin*m0 = 1*3*1 - 0 = 3
        assert_eq!(out[0], 3.0);
        // out[32] = d*sc1*(high nibble) - dmin*m1 = 1*5*2 - 0 = 10
        assert_eq!(out[32], 10.0);
        // qs[1..] = 0 -> out[1] = 0
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn q6_k_block_known() {
        // d = 1.0; scales[0] = 2; ql[0]=0x05, qh[0]=0x01.
        let mut block = vec![0u8; Q6_K_BYTES];
        block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        block[192] = 2; // scales[0] = 2 (i8)
        block[0] = 0x05; // ql[0]
        block[128] = 0x01; // qh[0]

        let mut out = vec![0.0f32; QK_K];
        dequantize_q6_k_block(&block, &mut out);
        // q1 = (5 | (1<<4)) - 32 = 21 - 32 = -11; out[0] = d*sc[0]*q1 = 1*2*-11 = -22
        assert_eq!(out[0], -22.0);
        // all-zero quant elsewhere in sub-block 0 -> q = -32, scale[0]=2 -> -64
        assert_eq!(out[1], -64.0);
    }

    #[test]
    fn dispatch_f32_passthrough() {
        let data: Vec<u8> = [1.0f32, -2.5, 3.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = dequantize(GgmlType::F32, &data, 3);
        assert_eq!(out, vec![1.0, -2.5, 3.25]);
    }

    #[test]
    fn dispatch_two_q4k_blocks() {
        // Two identical blocks -> 512 outputs, second mirrors the first.
        let mut block = vec![0u8; Q4_K_BYTES];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[4] = 1; // sc0 = 1
        block[16] = 0x07; // qs[0] low nibble 7
        let mut data = block.clone();
        data.extend_from_slice(&block);

        let out = dequantize(GgmlType::Q4_K, &data, 512);
        assert_eq!(out[0], 7.0);
        assert_eq!(out[256], 7.0);
    }
}
