//! Minimal, pure-Rust GGUF parser.
//!
//! Parses the GGUF container (header, metadata key/value pairs, tensor table) and
//! exposes each tensor's data as a zero-copy slice over the mmapped file. See
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md> for the format.
//!
//! All multi-byte values are little-endian (the standard GGUF byte order).

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::tensor::GgmlType;

const DEFAULT_ALIGNMENT: u64 = 32;

pub type Result<T> = std::result::Result<T, GgufError>;

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    UnexpectedEof { needed: usize, pos: usize, len: usize },
    UnknownValueType(u32),
    InvalidUtf8,
    UnknownTensorType { name: String, ty: u32 },
    TensorOutOfBounds { name: String },
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::BadMagic(m) => write!(f, "not a GGUF file (magic = {m:?})"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            Self::UnexpectedEof { needed, pos, len } => {
                write!(f, "unexpected end of file: need {needed} bytes at offset {pos}, file is {len} bytes")
            }
            Self::UnknownValueType(v) => write!(f, "unknown metadata value type {v}"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 in a GGUF string"),
            Self::UnknownTensorType { name, ty } => {
                write!(f, "tensor '{name}' has unmodeled ggml type {ty}")
            }
            Self::TensorOutOfBounds { name } => {
                write!(f, "tensor '{name}' data range falls outside the file")
            }
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// GGUF metadata value type enum (the on-disk discriminants).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }
}

/// A parsed GGUF metadata value.
#[derive(Clone, Debug)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    /// A homogeneous array. `elem_type` is the raw value-type discriminant of items.
    Array { elem_type: u32, items: Vec<MetaValue> },
}

/// Description of one tensor in the file.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGUF order (fastest-varying first).
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset relative to the start of the tensor-data section.
    pub offset: u64,
    /// Absolute byte offset of this tensor's data within the file.
    pub data_start: u64,
    /// Length of this tensor's data in bytes.
    pub data_len: u64,
}

impl TensorInfo {
    #[allow(dead_code)] // used for buffer sizing in the matmul/dequant kernels (next milestone)
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// Backing storage for the file bytes: an mmap in production, an owned buffer in tests.
enum Backing {
    Mmap(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Backing {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => &m[..],
            Backing::Owned(v) => &v[..],
        }
    }
}

/// A parsed GGUF file. Holds the mmap alive and indexes into it for tensor data.
pub struct GgufFile {
    // Read via `tensor_data`; that accessor is consumed by the dequant kernels in the
    // next milestone, so it reads as dead in the current `dump`-only binary.
    #[allow(dead_code)]
    backing: Backing,
    pub version: u32,
    pub alignment: u64,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset where the aligned tensor-data section begins.
    pub data_offset: u64,
}

impl GgufFile {
    /// Open and parse a GGUF file from disk (mmapped read-only).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        // Safety: we only read the mapping; the file is not modified while mapped.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_backing(Backing::Mmap(mmap))
    }

    /// Parse a GGUF file already resident in memory (used by tests and small files).
    #[allow(dead_code)] // entry point for tests; will be used for in-memory parsing later
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::from_backing(Backing::Owned(bytes))
    }

    fn from_backing(backing: Backing) -> Result<Self> {
        let Parsed { version, alignment, metadata, tensors, data_offset } =
            parse(&backing)?;
        Ok(Self { backing, version, alignment, metadata, tensors, data_offset })
    }

    /// Zero-copy view of a tensor's raw (still-quantized) bytes.
    #[allow(dead_code)] // consumed by the dequant kernels in the next milestone
    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        &self.backing[t.data_start as usize..(t.data_start + t.data_len) as usize]
    }

    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    /// Read an integer-valued metadata entry, widening any unsigned/signed integer type.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.get(key)? {
            MetaValue::U8(v) => Some(*v as u32),
            MetaValue::U16(v) => Some(*v as u32),
            MetaValue::U32(v) => Some(*v),
            MetaValue::U64(v) => Some(*v as u32),
            MetaValue::I8(v) => Some(*v as u32),
            MetaValue::I16(v) => Some(*v as u32),
            MetaValue::I32(v) => Some(*v as u32),
            MetaValue::I64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.get(key)? {
            MetaValue::F32(v) => Some(*v),
            MetaValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            MetaValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// The model architecture string, e.g. `"lfm2_moe"`.
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }
}

struct Parsed {
    version: u32,
    alignment: u64,
    metadata: HashMap<String, MetaValue>,
    tensors: Vec<TensorInfo>,
    data_offset: u64,
}

fn parse(buf: &[u8]) -> Result<Parsed> {
    let mut r = Reader::new(buf);

    let magic: [u8; 4] = r.take(4)?.try_into().unwrap();
    if &magic != b"GGUF" {
        return Err(GgufError::BadMagic(magic));
    }
    let version = r.u32()?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;

    let mut metadata = HashMap::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let key = r.gguf_string()?;
        let vt = ValueType::from_u32(r.u32()?)?;
        let val = read_value(&mut r, vt)?;
        metadata.insert(key, val);
    }

    let alignment = match metadata.get("general.alignment") {
        Some(MetaValue::U32(v)) => *v as u64,
        Some(MetaValue::U64(v)) => *v,
        _ => DEFAULT_ALIGNMENT,
    };

    // Read the tensor table (offsets are relative to the data section, not the file).
    let mut raw = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.gguf_string()?;
        let n_dims = r.u32()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let raw_type = r.u32()?;
        let offset = r.u64()?;
        raw.push((name, dims, raw_type, offset));
    }

    // The tensor-data section starts at the next `alignment` boundary.
    let data_offset = align_up(r.pos() as u64, alignment);

    let file_len = buf.len() as u64;
    let mut tensors = Vec::with_capacity(raw.len());
    for (name, dims, raw_type, offset) in raw {
        let ggml_type = GgmlType::from_u32(raw_type);
        let n_elem: u64 = dims.iter().product();
        let data_len = ggml_type
            .byte_size(n_elem)
            .ok_or_else(|| GgufError::UnknownTensorType { name: name.clone(), ty: raw_type })?;
        let data_start = data_offset + offset;
        if data_start + data_len > file_len {
            return Err(GgufError::TensorOutOfBounds { name });
        }
        tensors.push(TensorInfo { name, dims, ggml_type, offset, data_start, data_len });
    }

    Ok(Parsed { version, alignment, metadata, tensors, data_offset })
}

fn read_value(r: &mut Reader, vt: ValueType) -> Result<MetaValue> {
    Ok(match vt {
        ValueType::U8 => MetaValue::U8(r.u8()?),
        ValueType::I8 => MetaValue::I8(r.u8()? as i8),
        ValueType::U16 => MetaValue::U16(r.u16()?),
        ValueType::I16 => MetaValue::I16(r.u16()? as i16),
        ValueType::U32 => MetaValue::U32(r.u32()?),
        ValueType::I32 => MetaValue::I32(r.u32()? as i32),
        ValueType::F32 => MetaValue::F32(f32::from_bits(r.u32()?)),
        ValueType::Bool => MetaValue::Bool(r.u8()? != 0),
        ValueType::String => MetaValue::String(r.gguf_string()?),
        ValueType::U64 => MetaValue::U64(r.u64()?),
        ValueType::I64 => MetaValue::I64(r.u64()? as i64),
        ValueType::F64 => MetaValue::F64(f64::from_bits(r.u64()?)),
        ValueType::Array => {
            let elem_type = r.u32()?;
            let elem_vt = ValueType::from_u32(elem_type)?;
            let len = r.u64()? as usize;
            // Cap pre-allocation so a malformed huge length can't OOM us; the loop still
            // bounds-checks every element against the real buffer.
            let mut items = Vec::with_capacity(len.min(4096));
            for _ in 0..len {
                items.push(read_value(r, elem_vt)?);
            }
            MetaValue::Array { elem_type, items }
        }
    })
}

fn align_up(x: u64, a: u64) -> u64 {
    if a == 0 {
        x
    } else {
        x.div_ceil(a) * a
    }
}

/// A little-endian, bounds-checked cursor over the file bytes.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(GgufError::UnexpectedEof {
            needed: n,
            pos: self.pos,
            len: self.buf.len(),
        })?;
        if end > self.buf.len() {
            return Err(GgufError::UnexpectedEof { needed: n, pos: self.pos, len: self.buf.len() });
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn gguf_string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Build a small but structurally complete GGUF in memory and parse it back.
    fn build_sample() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&3u64.to_le_bytes()); // kv_count

        // kv: general.architecture = "lfm2_moe"
        push_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&(ValueType::String as u32).to_le_bytes());
        push_str(&mut buf, "lfm2_moe");

        // kv: lfm2_moe.block_count = u32 24
        push_str(&mut buf, "lfm2_moe.block_count");
        buf.extend_from_slice(&(ValueType::U32 as u32).to_le_bytes());
        buf.extend_from_slice(&24u32.to_le_bytes());

        // kv: tokenizer.ggml.tokens = array of 2 strings
        push_str(&mut buf, "tokenizer.ggml.tokens");
        buf.extend_from_slice(&(ValueType::Array as u32).to_le_bytes());
        buf.extend_from_slice(&(ValueType::String as u32).to_le_bytes()); // elem type
        buf.extend_from_slice(&2u64.to_le_bytes()); // len
        push_str(&mut buf, "<bos>");
        push_str(&mut buf, "hello");

        // tensor 0: "a" F32 shape [4], offset 0
        push_str(&mut buf, "a");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim0
        buf.extend_from_slice(&0u32.to_le_bytes()); // type F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // tensor 1: "b" F32 shape [2,3], offset 16 (after tensor a's 16 bytes)
        push_str(&mut buf, "b");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&2u64.to_le_bytes()); // dim0
        buf.extend_from_slice(&3u64.to_le_bytes()); // dim1
        buf.extend_from_slice(&0u32.to_le_bytes()); // type F32
        buf.extend_from_slice(&16u64.to_le_bytes()); // offset

        // pad to alignment 32
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        // tensor data: a = [1,2,3,4], b = [10..16)
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn parses_header_and_metadata() {
        let g = GgufFile::from_bytes(build_sample()).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.alignment, 32);
        assert_eq!(g.architecture(), Some("lfm2_moe"));
        assert_eq!(g.get_u32("lfm2_moe.block_count"), Some(24));

        match g.get("tokenizer.ggml.tokens") {
            Some(MetaValue::Array { items, .. }) => assert_eq!(items.len(), 2),
            other => panic!("expected token array, got {other:?}"),
        }
    }

    #[test]
    fn parses_tensors_and_data() {
        let g = GgufFile::from_bytes(build_sample()).unwrap();
        assert_eq!(g.tensors.len(), 2);

        let a = &g.tensors[0];
        assert_eq!(a.name, "a");
        assert_eq!(a.dims, vec![4]);
        assert_eq!(a.ggml_type, GgmlType::F32);
        assert_eq!(a.n_elements(), 4);
        assert_eq!(a.data_len, 16);

        let b = &g.tensors[1];
        assert_eq!(b.name, "b");
        assert_eq!(b.dims, vec![2, 3]);
        assert_eq!(b.n_elements(), 6);

        // Read tensor `a` back as f32 and check values.
        let bytes = g.tensor_data(a);
        let vals: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);

        // And the first element of `b`.
        let bb = g.tensor_data(b);
        let first = f32::from_le_bytes(bb[0..4].try_into().unwrap());
        assert_eq!(first, 10.0);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = build_sample();
        buf[0] = b'X';
        assert!(matches!(GgufFile::from_bytes(buf), Err(GgufError::BadMagic(_))));
    }

    #[test]
    fn rejects_truncated_file() {
        let mut buf = build_sample();
        buf.truncate(10);
        assert!(GgufFile::from_bytes(buf).is_err());
    }
}
