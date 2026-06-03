//! GGML dtype enum and per-type block sizing.
//!
//! GGML stores quantized tensors in fixed-size "blocks": a group of `block_elems`
//! weights packed into `block_bytes` of storage. Non-quantized types are treated as
//! 1-element blocks. Byte size of a tensor is therefore
//! `(n_elements / block_elems) * block_bytes`.

use std::fmt;

/// A GGML tensor data type. We model the types we expect plus a catch-all that carries
/// the raw enum value, so the loader can report (rather than crash on) anything new.
#[allow(non_camel_case_types)] // these are the canonical ggml type names
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
    /// Any ggml type we don't model yet; carries the raw enum discriminant.
    Other(u32),
}

impl GgmlType {
    /// Map the on-disk ggml type enum value to a [`GgmlType`].
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            other => Self::Other(other),
        }
    }

    /// `(elements per block, bytes per block)`, or `None` for types we don't model.
    pub fn block(self) -> Option<(u64, u64)> {
        let b = match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::BF16 => (1, 2),
            Self::F64 => (1, 8),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 => (1, 8),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2_K => (256, 84),
            Self::Q3_K => (256, 110),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::Q8_K => (256, 292),
            Self::Other(_) => return None,
        };
        Some(b)
    }

    /// Byte size of a tensor holding `n_elements` of this type.
    ///
    /// Returns `None` if the type is unmodeled, or if `n_elements` is not a whole number
    /// of blocks (which would indicate a malformed tensor for a quantized type).
    pub fn byte_size(self, n_elements: u64) -> Option<u64> {
        let (blk_elems, blk_bytes) = self.block()?;
        if !n_elements.is_multiple_of(blk_elems) {
            return None;
        }
        Some((n_elements / blk_elems) * blk_bytes)
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::BF16 => "BF16",
            Self::Other(v) => return write!(f, "Other({v})"),
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_block_sizes() {
        assert_eq!(GgmlType::Q4_K.block(), Some((256, 144)));
        assert_eq!(GgmlType::Q6_K.block(), Some((256, 210)));
        assert_eq!(GgmlType::Q8_0.block(), Some((32, 34)));
        assert_eq!(GgmlType::F32.block(), Some((1, 4)));
        assert_eq!(GgmlType::Other(99).block(), None);
    }

    #[test]
    fn byte_size_math() {
        // 2048x2048 Q4_K matrix: 4_194_304 elems / 256 * 144
        assert_eq!(GgmlType::Q4_K.byte_size(2048 * 2048), Some(2_359_296));
        assert_eq!(GgmlType::F32.byte_size(10), Some(40));
        // not a whole number of blocks -> None
        assert_eq!(GgmlType::Q4_K.byte_size(100), None);
    }

    #[test]
    fn type_roundtrip_names() {
        assert_eq!(GgmlType::from_u32(12), GgmlType::Q4_K);
        assert_eq!(GgmlType::from_u32(14), GgmlType::Q6_K);
        assert_eq!(GgmlType::from_u32(30), GgmlType::BF16);
        assert_eq!(format!("{}", GgmlType::Q4_K), "Q4_K");
        assert_eq!(format!("{}", GgmlType::Other(123)), "Other(123)");
    }
}
