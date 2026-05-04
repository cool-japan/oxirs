//! Pure-Rust GGUF v2/v3 metadata parser.
//!
//! Reads model header information (key-value metadata and tensor layout) WITHOUT
//! loading the full tensor weight data into RAM.  This enables lazy model
//! discovery and registration at negligible memory cost.
//!
//! # GGUF binary layout (little-endian)
//!
//! ```text
//! [4 bytes]  magic:    0x47 0x47 0x55 0x46 ("GGUF")
//! [4 bytes]  version:  u32  (2 or 3)
//! [8 bytes]  n_tensors: u64
//! [8 bytes]  n_kv:      u64
//!
//! Then n_kv key-value entries:
//!   key:        {u64 len, [u8] bytes}
//!   value_type: u32
//!   value:      depends on value_type
//!
//! Then n_tensors tensor info records:
//!   name:       {u64 len, [u8] bytes}
//!   n_dims:     u32
//!   dims:       [u64; n_dims]
//!   data_type:  u32
//!   offset:     u64
//! ```
//!
//! # Version notes
//!
//! GGUF v1 used narrower integer types; this parser supports only v2 and v3.
//! Parsing v1 files returns [`GgufParseError::UnsupportedVersion`].
//!
//! Nested arrays (array-of-array values) are not common in practice; the parser
//! returns [`GgufParseError::NestedArrayUnsupported`] rather than recursing
//! without bound.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

// ─── Magic constant ──────────────────────────────────────────────────────────

const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"

// ─── GGML value type constants ───────────────────────────────────────────────

const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

// ─── GgufValue ───────────────────────────────────────────────────────────────

/// A typed GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Return the inner value as `u64` if this is a numeric integer variant.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I16(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Return the inner string slice if this is a `Str` variant.
    pub fn as_str(&self) -> Option<&str> {
        if let GgufValue::Str(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    /// Return the inner value as `f32` if this is a floating-point variant.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

// ─── GgufModelArch ───────────────────────────────────────────────────────────

/// Architecture metadata extracted from GGUF key-value pairs.
///
/// Common GGUF architecture keys follow the pattern `<arch>.<field>`, e.g.
/// `llama.context_length`.  This struct collects the most useful fields.
#[derive(Debug, Clone, Default)]
pub struct GgufModelArch {
    /// Model architecture identifier (e.g. `"llama"`, `"mistral"`, `"phi3"`).
    pub architecture: Option<String>,
    /// Maximum context length in tokens.
    pub context_length: Option<u64>,
    /// Hidden state dimension (also called d_model or embedding_length).
    pub embedding_length: Option<u64>,
    /// Feed-forward / intermediate dimension.
    pub feed_forward_length: Option<u64>,
    /// Number of attention heads.
    pub head_count: Option<u64>,
    /// Number of key-value attention heads (GQA).
    pub head_count_kv: Option<u64>,
    /// Number of hidden layers.
    pub layer_count: Option<u64>,
    /// RoPE dimension count.
    pub rope_dimension_count: Option<u64>,
    /// Vocabulary size.
    pub vocab_size: Option<u64>,
}

// ─── GgufTensorInfo ──────────────────────────────────────────────────────────

/// Metadata for one tensor stored in the GGUF file.
///
/// The actual weight data is **not** loaded; `offset` gives the byte position
/// within the tensor-data region at the end of the file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g. `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Shape (number of elements per dimension; first dim is rows in gguf convention).
    pub dims: Vec<u64>,
    /// GGML type code (`GGML_TYPE_F32 = 0`, `GGML_TYPE_F16 = 1`, etc.).
    pub data_type: u32,
    /// Byte offset from the start of the tensor-data region.
    pub offset: u64,
    /// Total number of elements (product of all dims).
    pub param_count: u64,
}

// ─── GgufMetadata ────────────────────────────────────────────────────────────

/// Complete GGUF file metadata parsed from the header.
///
/// No tensor weight data is loaded; this struct is cheap to keep in memory.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    /// GGUF format version (2 or 3).
    pub version: u32,
    /// Number of tensors declared in the header.
    pub n_tensors: u64,
    /// All key-value metadata entries.
    pub kv: HashMap<String, GgufValue>,
    /// Tensor info records (shape and location, no data).
    pub tensors: Vec<GgufTensorInfo>,
    /// Extracted architecture metadata for convenient access.
    pub arch: GgufModelArch,
    /// Total file size in bytes, if known.
    pub file_size_bytes: Option<u64>,
}

impl GgufMetadata {
    /// Estimate the total parameter count by summing `param_count` across all tensors.
    pub fn total_params(&self) -> u64 {
        self.tensors.iter().map(|t| t.param_count).sum()
    }

    /// Rough estimate of tensor data bytes.
    ///
    /// For F32 tensors this is exact; for quantised types (Q4, Q5, Q8, etc.)
    /// the result is an approximation because block-quantised formats have
    /// non-trivial bytes-per-element.  Use only for sizing heuristics.
    pub fn estimated_size_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .map(|t| {
                let bpe: u64 = ggml_bytes_per_element(t.data_type);
                t.param_count.saturating_mul(bpe)
            })
            .sum()
    }

    /// Return all tensors whose name starts with `prefix`.
    pub fn tensors_with_prefix(&self, prefix: &str) -> Vec<&GgufTensorInfo> {
        self.tensors
            .iter()
            .filter(|t| t.name.starts_with(prefix))
            .collect()
    }
}

/// Return an approximate bytes-per-element for a GGML type code.
///
/// Block-quantised types (Q4_0, Q5_K, …) return an approximation.
fn ggml_bytes_per_element(data_type: u32) -> u64 {
    match data_type {
        0 => 4,  // GGML_TYPE_F32
        1 => 2,  // GGML_TYPE_F16
        2 => 1,  // GGML_TYPE_Q4_0  (approx: 4.5 bits/elem → round to 1)
        3 => 1,  // GGML_TYPE_Q4_1
        6 => 1,  // GGML_TYPE_Q5_0
        7 => 1,  // GGML_TYPE_Q5_1
        8 => 1,  // GGML_TYPE_Q8_0
        9 => 1,  // GGML_TYPE_Q8_1
        10 => 1, // GGML_TYPE_Q2_K
        11 => 1, // GGML_TYPE_Q3_K
        12 => 1, // GGML_TYPE_Q4_K
        13 => 1, // GGML_TYPE_Q5_K
        14 => 1, // GGML_TYPE_Q6_K
        15 => 1, // GGML_TYPE_Q8_K
        16 => 2, // GGML_TYPE_IQ2_XXS
        17 => 2, // GGML_TYPE_IQ2_XS
        18 => 4, // GGML_TYPE_I8  — 1 byte each
        19 => 2, // GGML_TYPE_I16
        20 => 4, // GGML_TYPE_I32
        _ => 4,  // unknown → assume F32
    }
}

// ─── GgufParseError ──────────────────────────────────────────────────────────

/// Errors that can occur when parsing a GGUF file.
#[derive(Debug, thiserror::Error)]
pub enum GgufParseError {
    /// IO error (file not found, read failure, etc.).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The first 4 bytes are not the GGUF magic `GGUF`.
    #[error("invalid GGUF magic bytes")]
    InvalidMagic,

    /// The file declares a GGUF version other than 2 or 3.
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    /// A key or tensor name contains invalid UTF-8.
    #[error("invalid UTF-8 in key/name: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    /// The value-type field has an unrecognised tag.
    #[error("unknown value type: {0}")]
    UnknownValueType(u32),

    /// The file ended before all expected bytes were read.
    #[error("truncated file")]
    Truncated,

    /// Nested arrays (array-of-array) are not supported.
    #[error("nested GGUF arrays are not supported")]
    NestedArrayUnsupported,
}

// ─── Internal read helpers ────────────────────────────────────────────────────

fn read_exact_or_truncated<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<(), GgufParseError> {
    r.read_exact(buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufParseError::Truncated
        } else {
            GgufParseError::Io(e)
        }
    })
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8, GgufParseError> {
    let mut buf = [0u8; 1];
    read_exact_or_truncated(r, &mut buf)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8, GgufParseError> {
    read_u8(r).map(|v| v as i8)
}

fn read_u16_le<R: Read>(r: &mut R) -> Result<u16, GgufParseError> {
    let mut buf = [0u8; 2];
    read_exact_or_truncated(r, &mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16_le<R: Read>(r: &mut R) -> Result<i16, GgufParseError> {
    read_u16_le(r).map(|v| v as i16)
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, GgufParseError> {
    let mut buf = [0u8; 4];
    read_exact_or_truncated(r, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32_le<R: Read>(r: &mut R) -> Result<i32, GgufParseError> {
    read_u32_le(r).map(|v| v as i32)
}

fn read_f32_le<R: Read>(r: &mut R) -> Result<f32, GgufParseError> {
    let bits = read_u32_le(r)?;
    Ok(f32::from_bits(bits))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, GgufParseError> {
    let mut buf = [0u8; 8];
    read_exact_or_truncated(r, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64_le<R: Read>(r: &mut R) -> Result<i64, GgufParseError> {
    read_u64_le(r).map(|v| v as i64)
}

fn read_f64_le<R: Read>(r: &mut R) -> Result<f64, GgufParseError> {
    let bits = read_u64_le(r)?;
    Ok(f64::from_bits(bits))
}

/// Read a GGUF string: `{u64 length, [u8] bytes}`.
fn read_gguf_string<R: Read>(r: &mut R) -> Result<String, GgufParseError> {
    let len = read_u64_le(r)? as usize;
    let mut buf = vec![0u8; len];
    read_exact_or_truncated(r, &mut buf)?;
    Ok(String::from_utf8(buf)?)
}

// ─── GgufParser ──────────────────────────────────────────────────────────────

/// Parser for GGUF model files.
///
/// Reads only the file header — magic, version, key-value metadata, and tensor
/// info records — without loading any tensor data.
pub struct GgufParser;

impl GgufParser {
    /// Parse GGUF metadata from any [`Read`] + [`Seek`] source.
    ///
    /// The reader is positioned just past the header when this method returns.
    /// Tensor weight data (which follows the header) is not read.
    pub fn parse<R: Read + Seek>(reader: &mut R) -> Result<GgufMetadata, GgufParseError> {
        // ── File size (best-effort) ───────────────────────────────────────────
        let file_size_bytes = reader.seek(SeekFrom::End(0)).ok().map(|sz| {
            // Rewind to start before we begin parsing.
            let _ = reader.seek(SeekFrom::Start(0));
            sz
        });
        // Ensure we are at position 0 (seek may have failed silently above).
        reader
            .seek(SeekFrom::Start(0))
            .map_err(GgufParseError::Io)?;

        // ── Magic ─────────────────────────────────────────────────────────────
        let mut magic = [0u8; 4];
        read_exact_or_truncated(reader, &mut magic)?;
        if magic != GGUF_MAGIC {
            return Err(GgufParseError::InvalidMagic);
        }

        // ── Version ───────────────────────────────────────────────────────────
        let version = read_u32_le(reader)?;
        if version != 2 && version != 3 {
            return Err(GgufParseError::UnsupportedVersion(version));
        }

        // ── Header counts ─────────────────────────────────────────────────────
        let n_tensors = read_u64_le(reader)?;
        let n_kv = read_u64_le(reader)?;

        // ── Key-value metadata ────────────────────────────────────────────────
        let mut kv: HashMap<String, GgufValue> = HashMap::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = read_gguf_string(reader)?;
            let value_type = read_u32_le(reader)?;
            let value = read_value(reader, value_type)?;
            kv.insert(key, value);
        }

        // ── Tensor info ───────────────────────────────────────────────────────
        let mut tensors: Vec<GgufTensorInfo> = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_gguf_string(reader)?;
            let n_dims = read_u32_le(reader)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64_le(reader)?);
            }
            let data_type = read_u32_le(reader)?;
            let offset = read_u64_le(reader)?;
            let param_count = dims.iter().product::<u64>().max(1);
            tensors.push(GgufTensorInfo {
                name,
                dims,
                data_type,
                offset,
                param_count,
            });
        }

        // ── Extract architecture metadata from kv ─────────────────────────────
        let arch = extract_arch(&kv);

        Ok(GgufMetadata {
            version,
            n_tensors,
            kv,
            tensors,
            arch,
            file_size_bytes,
        })
    }

    /// Parse GGUF metadata from a file path.
    pub fn parse_file(path: &std::path::Path) -> Result<GgufMetadata, GgufParseError> {
        let mut file = std::fs::File::open(path).map_err(GgufParseError::Io)?;
        Self::parse(&mut file)
    }

    /// Parse GGUF metadata from an in-memory byte slice.
    ///
    /// Useful for tests that construct minimal synthetic GGUF buffers.
    pub fn parse_bytes(bytes: &[u8]) -> Result<GgufMetadata, GgufParseError> {
        let mut cursor = std::io::Cursor::new(bytes);
        Self::parse(&mut cursor)
    }
}

// ─── Value reading ────────────────────────────────────────────────────────────

/// Read one typed value from the reader given its type tag.
fn read_value<R: Read>(reader: &mut R, value_type: u32) -> Result<GgufValue, GgufParseError> {
    match value_type {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(read_u8(reader)?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(read_i8(reader)?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(read_u16_le(reader)?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(read_i16_le(reader)?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(read_u32_le(reader)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(read_i32_le(reader)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32_le(reader)?)),
        GGUF_TYPE_BOOL => {
            let b = read_u8(reader)?;
            Ok(GgufValue::Bool(b != 0))
        }
        GGUF_TYPE_STRING => Ok(GgufValue::Str(read_gguf_string(reader)?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32_le(reader)?;
            if elem_type == GGUF_TYPE_ARRAY {
                return Err(GgufParseError::NestedArrayUnsupported);
            }
            let count = read_u64_le(reader)?;
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(read_value(reader, elem_type)?);
            }
            Ok(GgufValue::Array(items))
        }
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(read_u64_le(reader)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(read_i64_le(reader)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64_le(reader)?)),
        unknown => Err(GgufParseError::UnknownValueType(unknown)),
    }
}

// ─── Architecture extraction ──────────────────────────────────────────────────

/// Extract known architecture fields from the metadata KV map.
fn extract_arch(kv: &HashMap<String, GgufValue>) -> GgufModelArch {
    // Determine architecture name from `general.architecture`.
    let architecture: Option<String> = kv
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    // Clone the prefix string so that `architecture` can be moved into the struct.
    let arch_prefix: String = architecture.clone().unwrap_or_else(|| "llama".to_owned());

    // Helper: look up `<arch>.<suffix>` or `<suffix>` (fallback without prefix).
    let get_u64 = |suffix: &str| -> Option<u64> {
        kv.get(&format!("{arch_prefix}.{suffix}"))
            .or_else(|| kv.get(suffix))
            .and_then(|v| v.as_u64())
    };

    GgufModelArch {
        architecture,
        context_length: get_u64("context_length"),
        embedding_length: get_u64("embedding_length"),
        feed_forward_length: get_u64("feed_forward_length"),
        head_count: get_u64("attention.head_count"),
        head_count_kv: get_u64("attention.head_count_kv"),
        layer_count: get_u64("block_count"),
        rope_dimension_count: get_u64("rope.dimension_count"),
        vocab_size: kv
            .get("tokenizer.ggml.token_type")
            .and_then(|v| {
                if let GgufValue::Array(arr) = v {
                    Some(arr.len() as u64)
                } else {
                    None
                }
            })
            .or_else(|| get_u64("vocab_size")),
    }
}
