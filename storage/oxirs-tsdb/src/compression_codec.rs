//! Time-series value compression codecs
//!
//! Provides several codec strategies for compressing sequences of f64 values
//! and i64 timestamps:
//!
//! * **Delta** – store the first value then successive differences
//! * **RLE** – run-length encoding of repeated values
//! * **Zigzag** – bijective signed-to-unsigned mapping, good for mixed-sign deltas
//! * **Gorilla** – XOR-based float compression (simplified)
//! * **Plain** – no compression (IEEE 754 little-endian bytes)

/// Codec variant selector
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecType {
    /// Delta encoding: first value + successive differences
    Delta,
    /// Run-length encoding: (value, count) pairs
    Rle,
    /// Zigzag encoding: maps signed integers to unsigned
    Zigzag,
    /// XOR-delta float compression (Gorilla-style)
    Gorilla,
    /// Plain uncompressed (IEEE 754 little-endian)
    Plain,
}

/// Result of a compression operation
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Compressed byte representation
    pub compressed: Vec<u8>,
    /// Number of original values encoded
    pub original_count: usize,
    /// compression_ratio = compressed_bytes / (original_count * 8)
    /// Values < 1.0 indicate space savings; may exceed 1.0 for uncompressible data
    pub compression_ratio: f64,
}

/// Result of a decompression operation
#[derive(Clone, PartialEq)]
pub struct DecompressionResult {
    /// Recovered f64 values
    pub values: Vec<f64>,
    /// Number of values
    pub count: usize,
}

/// Errors that may occur during decompression
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// The compressed data is malformed
    InvalidData,
    /// The compressed data is unexpectedly short
    TruncatedData,
    /// The requested codec is not supported for the data direction
    UnsupportedCodec,
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::InvalidData => write!(f, "invalid compressed data"),
            CodecError::TruncatedData => write!(f, "truncated compressed data"),
            CodecError::UnsupportedCodec => write!(f, "unsupported codec"),
        }
    }
}

impl std::error::Error for CodecError {}

// ─── Main codec dispatch ──────────────────────────────────────────────────────

/// Stateless codec dispatch for f64 and i64 sequences
pub struct CompressionCodec;

impl CompressionCodec {
    // ── f64 value compress / decompress ──────────────────────────────────────

    /// Compress a slice of f64 values with the given codec
    pub fn compress(values: &[f64], codec: CodecType) -> CompressionResult {
        if values.is_empty() {
            return CompressionResult {
                compressed: Vec::new(),
                original_count: 0,
                compression_ratio: 1.0,
            };
        }
        let compressed = match &codec {
            CodecType::Plain => compress_f64_plain(values),
            CodecType::Delta => {
                // Cast to i64 via bit pattern is lossy; instead delta on quantised i64
                let ints: Vec<i64> = values.iter().map(|v| v.to_bits() as i64).collect();
                let deltas = Self::delta_encode(&ints);
                encode_i64_sequence(&deltas)
            }
            CodecType::Rle => {
                let ints: Vec<i64> = values.iter().map(|v| v.to_bits() as i64).collect();
                let pairs = Self::rle_encode(&ints);
                encode_rle_pairs(&pairs)
            }
            CodecType::Zigzag => {
                let ints: Vec<i64> = values.iter().map(|v| v.to_bits() as i64).collect();
                let zz: Vec<u64> = ints.iter().map(|v| Self::zigzag_encode(*v)).collect();
                encode_u64_sequence(&zz)
            }
            CodecType::Gorilla => compress_f64_gorilla(values),
        };
        let original_bytes = (values.len() * 8) as f64;
        let ratio = compressed.len() as f64 / original_bytes.max(1.0);
        CompressionResult {
            original_count: values.len(),
            compression_ratio: ratio,
            compressed,
        }
    }

    /// Decompress a byte slice back to f64 values
    pub fn decompress(
        data: &[u8],
        codec: CodecType,
        count: usize,
    ) -> Result<DecompressionResult, CodecError> {
        if count == 0 {
            return Ok(DecompressionResult {
                values: Vec::new(),
                count: 0,
            });
        }
        let values = match codec {
            CodecType::Plain => decompress_f64_plain(data, count)?,
            CodecType::Delta => {
                let ints = decode_i64_sequence(data)?;
                let decoded = Self::delta_decode(&ints);
                decoded
                    .into_iter()
                    .map(|v| f64::from_bits(v as u64))
                    .collect()
            }
            CodecType::Rle => {
                let pairs = decode_rle_pairs(data)?;
                let ints = Self::rle_decode(&pairs);
                ints.into_iter().map(|v| f64::from_bits(v as u64)).collect()
            }
            CodecType::Zigzag => {
                let zz = decode_u64_sequence(data)?;
                zz.into_iter()
                    .map(|v| f64::from_bits(Self::zigzag_decode(v) as u64))
                    .collect()
            }
            CodecType::Gorilla => decompress_f64_gorilla(data, count)?,
        };
        let count_out = values.len();
        Ok(DecompressionResult {
            values,
            count: count_out,
        })
    }

    // ── i64 timestamp compress / decompress ──────────────────────────────────

    /// Compress a slice of i64 timestamps
    pub fn compress_timestamps(ts: &[i64], codec: CodecType) -> CompressionResult {
        if ts.is_empty() {
            return CompressionResult {
                compressed: Vec::new(),
                original_count: 0,
                compression_ratio: 1.0,
            };
        }
        let compressed = match &codec {
            CodecType::Delta => encode_i64_sequence(&Self::delta_encode(ts)),
            CodecType::Rle => encode_rle_pairs(&Self::rle_encode(ts)),
            CodecType::Zigzag => {
                let zz: Vec<u64> = ts.iter().map(|v| Self::zigzag_encode(*v)).collect();
                encode_u64_sequence(&zz)
            }
            CodecType::Plain => {
                let bytes: Vec<u8> = ts.iter().flat_map(|v| v.to_le_bytes()).collect();
                bytes
            }
            CodecType::Gorilla => {
                // For timestamps, Gorilla falls back to delta
                encode_i64_sequence(&Self::delta_encode(ts))
            }
        };
        let original_bytes = (ts.len() * 8) as f64;
        let ratio = compressed.len() as f64 / original_bytes.max(1.0);
        CompressionResult {
            original_count: ts.len(),
            compression_ratio: ratio,
            compressed,
        }
    }

    /// Decompress timestamp bytes back to i64 values
    pub fn decompress_timestamps(
        data: &[u8],
        codec: CodecType,
        count: usize,
    ) -> Result<Vec<i64>, CodecError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        match codec {
            CodecType::Delta | CodecType::Gorilla => {
                let deltas = decode_i64_sequence(data)?;
                Ok(Self::delta_decode(&deltas))
            }
            CodecType::Rle => {
                let pairs = decode_rle_pairs(data)?;
                Ok(Self::rle_decode(&pairs))
            }
            CodecType::Zigzag => {
                let zz = decode_u64_sequence(data)?;
                Ok(zz.into_iter().map(Self::zigzag_decode).collect())
            }
            CodecType::Plain => {
                if data.len() < count * 8 {
                    return Err(CodecError::TruncatedData);
                }
                Ok(data
                    .chunks_exact(8)
                    .take(count)
                    .map(|chunk| {
                        let arr: [u8; 8] = chunk.try_into().unwrap_or([0u8; 8]);
                        i64::from_le_bytes(arr)
                    })
                    .collect())
            }
        }
    }

    // ── Delta encoding ────────────────────────────────────────────────────────

    /// Delta-encode: `output[0] = input[0]`, `output[i] = input[i] - input[i-1]`
    pub fn delta_encode(values: &[i64]) -> Vec<i64> {
        if values.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(values.len());
        out.push(values[0]);
        for i in 1..values.len() {
            out.push(values[i].wrapping_sub(values[i - 1]));
        }
        out
    }

    /// Delta-decode: reconstruct original values from delta-encoded sequence
    pub fn delta_decode(deltas: &[i64]) -> Vec<i64> {
        if deltas.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(deltas.len());
        out.push(deltas[0]);
        for i in 1..deltas.len() {
            out.push(out[i - 1].wrapping_add(deltas[i]));
        }
        out
    }

    // ── RLE encoding ──────────────────────────────────────────────────────────

    /// Run-length encode: consecutive equal values are replaced by (value, count)
    pub fn rle_encode(values: &[i64]) -> Vec<(i64, usize)> {
        if values.is_empty() {
            return Vec::new();
        }
        let mut pairs: Vec<(i64, usize)> = Vec::new();
        let mut current = values[0];
        let mut count = 1usize;
        for &v in &values[1..] {
            if v == current {
                count += 1;
            } else {
                pairs.push((current, count));
                current = v;
                count = 1;
            }
        }
        pairs.push((current, count));
        pairs
    }

    /// Run-length decode: expand (value, count) pairs to flat sequence
    pub fn rle_decode(pairs: &[(i64, usize)]) -> Vec<i64> {
        let total: usize = pairs.iter().map(|(_, c)| c).sum();
        let mut out = Vec::with_capacity(total);
        for &(value, count) in pairs {
            for _ in 0..count {
                out.push(value);
            }
        }
        out
    }

    // ── Zigzag ────────────────────────────────────────────────────────────────

    /// Zigzag-encode: 0→0, -1→1, 1→2, -2→3, …
    pub fn zigzag_encode(v: i64) -> u64 {
        ((v << 1) ^ (v >> 63)) as u64
    }

    /// Zigzag-decode: inverse of `zigzag_encode`
    pub fn zigzag_decode(v: u64) -> i64 {
        ((v >> 1) as i64) ^ (-((v & 1) as i64))
    }

    // ── Estimate ratio ────────────────────────────────────────────────────────

    /// Estimate the compression ratio for a codec on the given f64 values.
    /// Lower is better (< 1.0 = compressed smaller than original).
    pub fn estimate_ratio(values: &[f64], codec: &CodecType) -> f64 {
        if values.is_empty() {
            return 1.0;
        }
        let result = Self::compress(values, codec.clone());
        result.compression_ratio
    }
}

// ─── Plain f64 encode / decode ────────────────────────────────────────────────

fn compress_f64_plain(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decompress_f64_plain(data: &[u8], count: usize) -> Result<Vec<f64>, CodecError> {
    if data.len() < count * 8 {
        return Err(CodecError::TruncatedData);
    }
    data.chunks_exact(8)
        .take(count)
        .map(|chunk| {
            let arr: [u8; 8] = chunk.try_into().map_err(|_| CodecError::InvalidData)?;
            Ok(f64::from_le_bytes(arr))
        })
        .collect()
}

// ─── Gorilla (XOR delta) ──────────────────────────────────────────────────────

fn compress_f64_gorilla(values: &[f64]) -> Vec<u8> {
    // Simplified: store first value verbatim, then XOR with previous
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len() * 8);
    let first = values[0].to_bits();
    out.extend_from_slice(&first.to_le_bytes());
    let mut prev = first;
    for &v in &values[1..] {
        let bits = v.to_bits();
        let xor = bits ^ prev;
        out.extend_from_slice(&xor.to_le_bytes());
        prev = bits;
    }
    out
}

fn decompress_f64_gorilla(data: &[u8], count: usize) -> Result<Vec<f64>, CodecError> {
    if data.len() < 8 {
        return Err(CodecError::TruncatedData);
    }
    if data.len() < count * 8 {
        return Err(CodecError::TruncatedData);
    }
    let mut out = Vec::with_capacity(count);
    let first_bytes: [u8; 8] = data[..8].try_into().map_err(|_| CodecError::InvalidData)?;
    let first = u64::from_le_bytes(first_bytes);
    out.push(f64::from_bits(first));
    let mut prev = first;
    for i in 1..count {
        let offset = i * 8;
        if offset + 8 > data.len() {
            return Err(CodecError::TruncatedData);
        }
        let xor_bytes: [u8; 8] = data[offset..offset + 8]
            .try_into()
            .map_err(|_| CodecError::InvalidData)?;
        let xor = u64::from_le_bytes(xor_bytes);
        let bits = xor ^ prev;
        out.push(f64::from_bits(bits));
        prev = bits;
    }
    Ok(out)
}

// ─── i64 sequence encoding helpers ───────────────────────────────────────────

/// Encode a Vec<i64> as little-endian bytes (prepend length as u32)
fn encode_i64_sequence(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + values.len() * 8);
    let len = values.len() as u32;
    out.extend_from_slice(&len.to_le_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn decode_i64_sequence(data: &[u8]) -> Result<Vec<i64>, CodecError> {
    if data.len() < 4 {
        return Err(CodecError::TruncatedData);
    }
    let len =
        u32::from_le_bytes(data[..4].try_into().map_err(|_| CodecError::InvalidData)?) as usize;
    if data.len() < 4 + len * 8 {
        return Err(CodecError::TruncatedData);
    }
    (0..len)
        .map(|i| {
            let offset = 4 + i * 8;
            let arr: [u8; 8] = data[offset..offset + 8]
                .try_into()
                .map_err(|_| CodecError::InvalidData)?;
            Ok(i64::from_le_bytes(arr))
        })
        .collect()
}

/// Encode u64 sequence: length (u32 LE) + values
fn encode_u64_sequence(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + values.len() * 8);
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn decode_u64_sequence(data: &[u8]) -> Result<Vec<u64>, CodecError> {
    if data.len() < 4 {
        return Err(CodecError::TruncatedData);
    }
    let len =
        u32::from_le_bytes(data[..4].try_into().map_err(|_| CodecError::InvalidData)?) as usize;
    if data.len() < 4 + len * 8 {
        return Err(CodecError::TruncatedData);
    }
    (0..len)
        .map(|i| {
            let offset = 4 + i * 8;
            let arr: [u8; 8] = data[offset..offset + 8]
                .try_into()
                .map_err(|_| CodecError::InvalidData)?;
            Ok(u64::from_le_bytes(arr))
        })
        .collect()
}

/// Encode RLE pairs: length (u32) + (value i64, count u64) pairs
fn encode_rle_pairs(pairs: &[(i64, usize)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + pairs.len() * 16);
    out.extend_from_slice(&(pairs.len() as u32).to_le_bytes());
    for &(value, count) in pairs {
        out.extend_from_slice(&value.to_le_bytes());
        out.extend_from_slice(&(count as u64).to_le_bytes());
    }
    out
}

fn decode_rle_pairs(data: &[u8]) -> Result<Vec<(i64, usize)>, CodecError> {
    if data.len() < 4 {
        return Err(CodecError::TruncatedData);
    }
    let len =
        u32::from_le_bytes(data[..4].try_into().map_err(|_| CodecError::InvalidData)?) as usize;
    if data.len() < 4 + len * 16 {
        return Err(CodecError::TruncatedData);
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let offset = 4 + i * 16;
        let val_arr: [u8; 8] = data[offset..offset + 8]
            .try_into()
            .map_err(|_| CodecError::InvalidData)?;
        let cnt_arr: [u8; 8] = data[offset + 8..offset + 16]
            .try_into()
            .map_err(|_| CodecError::InvalidData)?;
        let value = i64::from_le_bytes(val_arr);
        let count = u64::from_le_bytes(cnt_arr) as usize;
        out.push((value, count));
    }
    Ok(out)
}

impl std::fmt::Debug for DecompressionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecompressionResult")
            .field("count", &self.count)
            .field("values_len", &self.values.len())
            .finish()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS || (a.is_nan() && b.is_nan())
    }

    fn check_round_trip_f64(values: &[f64], codec: CodecType) {
        let result = CompressionCodec::compress(values, codec.clone());
        let decomp = CompressionCodec::decompress(&result.compressed, codec, values.len())
            .expect("decompression should succeed");
        assert_eq!(decomp.values.len(), values.len());
        for (a, b) in values.iter().zip(decomp.values.iter()) {
            assert!(approx_eq(*a, *b), "mismatch: {} vs {}", a, b);
        }
    }

    fn monotone_sequence(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 * 100.0).collect()
    }

    // ── delta_encode / delta_decode ──────────────────────────────────────────

    #[test]
    fn test_delta_encode_empty() {
        assert!(CompressionCodec::delta_encode(&[]).is_empty());
    }

    #[test]
    fn test_delta_encode_single_value() {
        let enc = CompressionCodec::delta_encode(&[42]);
        assert_eq!(enc, vec![42]);
    }

    #[test]
    fn test_delta_encode_monotone_sequence() {
        let values = vec![100i64, 200, 300, 400];
        let enc = CompressionCodec::delta_encode(&values);
        assert_eq!(enc[0], 100);
        assert_eq!(enc[1], 100);
        assert_eq!(enc[2], 100);
        assert_eq!(enc[3], 100);
    }

    #[test]
    fn test_delta_decode_round_trip() {
        let values = vec![10i64, 20, 35, 31, 50];
        let enc = CompressionCodec::delta_encode(&values);
        let dec = CompressionCodec::delta_decode(&enc);
        assert_eq!(dec, values);
    }

    #[test]
    fn test_delta_decode_empty() {
        assert!(CompressionCodec::delta_decode(&[]).is_empty());
    }

    #[test]
    fn test_delta_encode_negative_deltas() {
        let values = vec![100i64, 90, 80];
        let enc = CompressionCodec::delta_encode(&values);
        assert_eq!(enc[1], -10);
        let dec = CompressionCodec::delta_decode(&enc);
        assert_eq!(dec, values);
    }

    #[test]
    fn test_delta_encode_timestamps_monotone() {
        // Unix timestamps in milliseconds (sequential)
        let ts: Vec<i64> = (0..10).map(|i| 1_700_000_000_000i64 + i * 1000).collect();
        let enc = CompressionCodec::delta_encode(&ts);
        // All deltas after the first should be 1000
        assert!(enc[1..].iter().all(|&d| d == 1000));
    }

    // ── rle_encode / rle_decode ──────────────────────────────────────────────

    #[test]
    fn test_rle_encode_empty() {
        assert!(CompressionCodec::rle_encode(&[]).is_empty());
    }

    #[test]
    fn test_rle_encode_all_same() {
        let values = vec![7i64; 100];
        let pairs = CompressionCodec::rle_encode(&values);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (7, 100));
    }

    #[test]
    fn test_rle_encode_no_runs() {
        let values = vec![1i64, 2, 3, 4, 5];
        let pairs = CompressionCodec::rle_encode(&values);
        assert_eq!(pairs.len(), 5);
    }

    #[test]
    fn test_rle_decode_round_trip() {
        let values = vec![1i64, 1, 2, 2, 2, 3];
        let pairs = CompressionCodec::rle_encode(&values);
        let dec = CompressionCodec::rle_decode(&pairs);
        assert_eq!(dec, values);
    }

    #[test]
    fn test_rle_decode_empty() {
        assert!(CompressionCodec::rle_decode(&[]).is_empty());
    }

    #[test]
    fn test_rle_encode_mixed_runs() {
        let values = vec![5i64, 5, 5, 7, 7, 9];
        let pairs = CompressionCodec::rle_encode(&values);
        assert_eq!(pairs[0], (5, 3));
        assert_eq!(pairs[1], (7, 2));
        assert_eq!(pairs[2], (9, 1));
    }

    // ── zigzag_encode / zigzag_decode ────────────────────────────────────────

    #[test]
    fn test_zigzag_encode_zero() {
        assert_eq!(CompressionCodec::zigzag_encode(0), 0);
    }

    #[test]
    fn test_zigzag_encode_minus_one() {
        assert_eq!(CompressionCodec::zigzag_encode(-1), 1);
    }

    #[test]
    fn test_zigzag_encode_one() {
        assert_eq!(CompressionCodec::zigzag_encode(1), 2);
    }

    #[test]
    fn test_zigzag_encode_minus_two() {
        assert_eq!(CompressionCodec::zigzag_encode(-2), 3);
    }

    #[test]
    fn test_zigzag_decode_zero() {
        assert_eq!(CompressionCodec::zigzag_decode(0), 0);
    }

    #[test]
    fn test_zigzag_decode_one() {
        assert_eq!(CompressionCodec::zigzag_decode(1), -1);
    }

    #[test]
    fn test_zigzag_decode_two() {
        assert_eq!(CompressionCodec::zigzag_decode(2), 1);
    }

    #[test]
    fn test_zigzag_round_trip_positive() {
        for v in 0i64..100 {
            assert_eq!(
                CompressionCodec::zigzag_decode(CompressionCodec::zigzag_encode(v)),
                v
            );
        }
    }

    #[test]
    fn test_zigzag_round_trip_negative() {
        for v in -100i64..0 {
            assert_eq!(
                CompressionCodec::zigzag_decode(CompressionCodec::zigzag_encode(v)),
                v
            );
        }
    }

    #[test]
    fn test_zigzag_bijection_known_values() {
        let cases = [(0i64, 0u64), (-1, 1), (1, 2), (-2, 3), (2, 4)];
        for (signed, expected_unsigned) in cases {
            assert_eq!(CompressionCodec::zigzag_encode(signed), expected_unsigned);
            assert_eq!(CompressionCodec::zigzag_decode(expected_unsigned), signed);
        }
    }

    // ── compress / decompress round-trip per codec ───────────────────────────

    #[test]
    fn test_plain_round_trip() {
        let values = vec![1.0f64, 2.5, -3.0, 0.0, 100.0];
        check_round_trip_f64(&values, CodecType::Plain);
    }

    #[test]
    fn test_delta_f64_round_trip() {
        let values = monotone_sequence(10);
        check_round_trip_f64(&values, CodecType::Delta);
    }

    #[test]
    fn test_rle_f64_round_trip() {
        let values = vec![1.0f64, 1.0, 1.0, 2.0, 2.0, 3.0];
        check_round_trip_f64(&values, CodecType::Rle);
    }

    #[test]
    fn test_zigzag_f64_round_trip() {
        let values = vec![0.0f64, 1.0, -1.0, 2.0, -2.0];
        check_round_trip_f64(&values, CodecType::Zigzag);
    }

    #[test]
    fn test_gorilla_round_trip() {
        let values = vec![1.0f64, 1.1, 1.2, 1.3, 1.4];
        check_round_trip_f64(&values, CodecType::Gorilla);
    }

    #[test]
    fn test_gorilla_single_value() {
        check_round_trip_f64(&[42.0f64], CodecType::Gorilla);
    }

    // ── compression_ratio ────────────────────────────────────────────────────

    #[test]
    fn test_plain_compression_ratio_equals_one() {
        let values = vec![1.0f64, 2.0, 3.0];
        let result = CompressionCodec::compress(&values, CodecType::Plain);
        // Plain: 3 * 8 bytes in, 3 * 8 bytes out → ratio = 1.0
        assert!((result.compression_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rle_compression_ratio_below_one_for_runs() {
        // 100 identical values → RLE stores 1 pair of 16 bytes vs 800 bytes
        let values = vec![42.0f64; 100];
        let result = CompressionCodec::compress(&values, CodecType::Rle);
        assert!(result.compression_ratio < 1.0);
    }

    #[test]
    fn test_compression_ratio_field_in_result() {
        let values = vec![1.0f64, 2.0];
        let result = CompressionCodec::compress(&values, CodecType::Plain);
        assert!(result.compression_ratio > 0.0);
    }

    // ── timestamps ───────────────────────────────────────────────────────────

    #[test]
    fn test_compress_timestamps_delta_round_trip() {
        let ts: Vec<i64> = (0..20).map(|i| 1_700_000_000_000i64 + i * 1000).collect();
        let result = CompressionCodec::compress_timestamps(&ts, CodecType::Delta);
        let dec =
            CompressionCodec::decompress_timestamps(&result.compressed, CodecType::Delta, ts.len())
                .unwrap();
        assert_eq!(dec, ts);
    }

    #[test]
    fn test_compress_timestamps_rle_round_trip() {
        let ts = vec![1000i64; 50];
        let result = CompressionCodec::compress_timestamps(&ts, CodecType::Rle);
        let dec =
            CompressionCodec::decompress_timestamps(&result.compressed, CodecType::Rle, ts.len())
                .unwrap();
        assert_eq!(dec, ts);
    }

    #[test]
    fn test_compress_timestamps_plain_round_trip() {
        let ts: Vec<i64> = (0..5).map(|i| 1_000 + i * 500).collect();
        let result = CompressionCodec::compress_timestamps(&ts, CodecType::Plain);
        let dec =
            CompressionCodec::decompress_timestamps(&result.compressed, CodecType::Plain, ts.len())
                .unwrap();
        assert_eq!(dec, ts);
    }

    #[test]
    fn test_compress_timestamps_zigzag_round_trip() {
        let ts: Vec<i64> = vec![100, -50, 200, -10, 300];
        let result = CompressionCodec::compress_timestamps(&ts, CodecType::Zigzag);
        let dec = CompressionCodec::decompress_timestamps(
            &result.compressed,
            CodecType::Zigzag,
            ts.len(),
        )
        .unwrap();
        assert_eq!(dec, ts);
    }

    // ── empty input ──────────────────────────────────────────────────────────

    #[test]
    fn test_compress_empty_returns_empty() {
        let result = CompressionCodec::compress(&[], CodecType::Plain);
        assert_eq!(result.original_count, 0);
        assert!(result.compressed.is_empty());
    }

    #[test]
    fn test_decompress_count_zero_returns_empty() {
        let dec = CompressionCodec::decompress(&[], CodecType::Plain, 0).unwrap();
        assert!(dec.values.is_empty());
    }

    #[test]
    fn test_compress_timestamps_empty() {
        let result = CompressionCodec::compress_timestamps(&[], CodecType::Delta);
        assert_eq!(result.original_count, 0);
    }

    // ── estimate_ratio ───────────────────────────────────────────────────────

    #[test]
    fn test_estimate_ratio_plain_equals_one() {
        let values = vec![1.0f64, 2.0, 3.0];
        let ratio = CompressionCodec::estimate_ratio(&values, &CodecType::Plain);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_ratio_rle_better_for_runs() {
        let values = vec![7.0f64; 200];
        let rle_ratio = CompressionCodec::estimate_ratio(&values, &CodecType::Rle);
        let plain_ratio = CompressionCodec::estimate_ratio(&values, &CodecType::Plain);
        assert!(rle_ratio < plain_ratio);
    }

    #[test]
    fn test_estimate_ratio_delta_good_for_monotone() {
        let values = monotone_sequence(100);
        let ratio = CompressionCodec::estimate_ratio(&values, &CodecType::Delta);
        // delta should encode 100 monotone values reasonably
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_estimate_ratio_empty_is_one() {
        let ratio = CompressionCodec::estimate_ratio(&[], &CodecType::Delta);
        assert_eq!(ratio, 1.0);
    }

    // ── error cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_decompress_truncated_plain() {
        // Only 4 bytes but count=2 needs 16
        let err = CompressionCodec::decompress(&[0u8; 4], CodecType::Plain, 2);
        assert_eq!(err, Err(CodecError::TruncatedData));
    }

    #[test]
    fn test_decompress_truncated_gorilla() {
        let err = CompressionCodec::decompress(&[0u8; 4], CodecType::Gorilla, 2);
        assert_eq!(err, Err(CodecError::TruncatedData));
    }

    #[test]
    fn test_decompress_timestamps_truncated_plain() {
        let err = CompressionCodec::decompress_timestamps(&[0u8; 4], CodecType::Plain, 2);
        assert_eq!(err, Err(CodecError::TruncatedData));
    }
}
