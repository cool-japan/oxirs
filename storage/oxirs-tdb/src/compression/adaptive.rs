//! Adaptive compression implementation that selects the best algorithm

use crate::compression::{
    bitmap::{BitmapRoaringEncoder, BitmapWAHEncoder},
    column_store::ColumnStoreCompressor,
    delta::DeltaEncoder,
    dictionary::AdaptiveDictionary,
    frame_of_reference::FrameOfReferenceEncoder,
    run_length::RunLengthEncoder,
    AdvancedCompressionType, CompressedData, CompressionAlgorithm,
};
use anyhow::{anyhow, Result};
use std::time::Instant;

/// Adaptive compressor that selects the best algorithm for given data
pub struct AdaptiveCompressor {
    /// Sample size for analysis
    sample_size: usize,
    /// Compression threshold
    threshold: f64,
}

impl AdaptiveCompressor {
    /// Create new adaptive compressor
    pub fn new(sample_size: usize, threshold: f64) -> Self {
        Self {
            sample_size,
            threshold,
        }
    }

    /// Select best compression algorithm for data
    pub fn select_best_algorithm(&self, data: &[u8]) -> AdvancedCompressionType {
        if data.len() < 10 {
            return AdvancedCompressionType::RunLength; // Default for small data
        }

        let sample_data = if data.len() > self.sample_size {
            &data[..self.sample_size]
        } else {
            data
        };

        // Analyze data characteristics
        let stats = self.analyze_data(sample_data);

        // Check if compression is beneficial based on threshold
        if stats.repetition_ratio < self.threshold && stats.sparsity < self.threshold {
            // Data doesn't meet compression threshold, use simple algorithm
            return AdvancedCompressionType::RunLength;
        }

        // Decision tree based on data characteristics
        if stats.sparsity > 0.9 {
            // Very sparse data (lots of zeros) - use bitmap compression
            if stats.bit_runs > stats.byte_runs {
                AdvancedCompressionType::BitmapRoaring
            } else {
                AdvancedCompressionType::BitmapWAH
            }
        } else if stats.repetition_ratio > 0.5 {
            // High repetition - use run-length encoding
            AdvancedCompressionType::RunLength
        } else if stats.is_sorted && stats.delta_efficiency > 0.7 {
            // Sorted numeric data - use delta or FOR encoding
            if stats.range_ratio < 0.1 {
                AdvancedCompressionType::FrameOfReference
            } else {
                AdvancedCompressionType::Delta
            }
        } else if stats.dictionary_efficiency > 0.6 {
            // Text-like data with repeated patterns
            AdvancedCompressionType::AdaptiveDictionary
        } else if data.len() % 8 == 0 && stats.structured_score > 0.5 {
            // Structured data that might benefit from column store
            AdvancedCompressionType::ColumnStore
        } else {
            // Default fallback
            AdvancedCompressionType::RunLength
        }
    }

    /// Analyze data characteristics
    fn analyze_data(&self, data: &[u8]) -> DataStats {
        let mut stats = DataStats::default();

        if data.is_empty() {
            return stats;
        }

        // Basic statistics
        stats.length = data.len();
        stats.unique_bytes = data.iter().collect::<std::collections::HashSet<_>>().len();

        // Sparsity analysis (count zeros)
        let zero_count = data.iter().filter(|&&b| b == 0).count();
        stats.sparsity = zero_count as f64 / data.len() as f64;

        // Repetition analysis
        let mut repetitions = 0;
        let mut current_byte = data[0];
        let mut run_length = 1;

        for &byte in &data[1..] {
            if byte == current_byte {
                run_length += 1;
            } else {
                if run_length > 1 {
                    repetitions += run_length;
                }
                stats.byte_runs += 1;
                current_byte = byte;
                run_length = 1;
            }
        }

        if run_length > 1 {
            repetitions += run_length;
        }
        stats.byte_runs += 1;

        stats.repetition_ratio = repetitions as f64 / data.len() as f64;

        // Bit-level runs (for bitmap analysis)
        let mut bit_runs = 0;
        let mut current_bit = data[0] & 1;

        for &byte in data {
            for i in 0..8 {
                let bit = (byte >> i) & 1;
                if bit != current_bit {
                    bit_runs += 1;
                    current_bit = bit;
                }
            }
        }
        stats.bit_runs = bit_runs;

        // Sortedness check (for numeric data)
        if data.len() >= 8 {
            let u64_values: Vec<u64> = data
                .chunks_exact(8)
                .map(|chunk| {
                    u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();

            if u64_values.len() > 1 {
                let mut sorted_count = 0;
                for i in 1..u64_values.len() {
                    if u64_values[i] >= u64_values[i - 1] {
                        sorted_count += 1;
                    }
                }
                stats.is_sorted = sorted_count as f64 / (u64_values.len() - 1) as f64 > 0.8;

                // Range analysis for FOR encoding
                if let (Some(&min_val), Some(&max_val)) =
                    (u64_values.iter().min(), u64_values.iter().max())
                {
                    stats.range_ratio = if max_val > 0 {
                        (max_val - min_val) as f64 / max_val as f64
                    } else {
                        0.0
                    };
                }

                // Delta efficiency
                let mut small_deltas = 0;
                for i in 1..u64_values.len() {
                    if u64_values[i].abs_diff(u64_values[i - 1]) < 65536 {
                        small_deltas += 1;
                    }
                }
                stats.delta_efficiency = small_deltas as f64 / (u64_values.len() - 1) as f64;
            }
        }

        // Dictionary efficiency (text-like patterns)
        let mut word_count = 0;
        let mut word_bytes = 0;
        let mut in_word = false;

        for &byte in data {
            if byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' {
                if !in_word {
                    word_count += 1;
                    in_word = true;
                }
                word_bytes += 1;
            } else {
                in_word = false;
            }
        }

        if word_count > 0 {
            stats.dictionary_efficiency = word_bytes as f64 / data.len() as f64;
        }

        // Structured data score (regularity in byte patterns)
        if data.len() >= 16 {
            let mut pattern_score = 0.0;
            let chunk_size = 8;
            let chunks: Vec<_> = data.chunks_exact(chunk_size).collect();

            if chunks.len() > 1 {
                let mut similar_chunks = 0;
                for i in 1..chunks.len() {
                    let mut differences = 0;
                    // Need index j to compare elements at the same position in different chunks
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..chunk_size {
                        if chunks[i][j] != chunks[0][j] {
                            differences += 1;
                        }
                    }
                    if differences <= chunk_size / 2 {
                        similar_chunks += 1;
                    }
                }
                pattern_score = similar_chunks as f64 / (chunks.len() - 1) as f64;
            }
            stats.structured_score = pattern_score;
        }

        stats
    }

    /// Compress data using selected algorithm
    pub fn compress_adaptive(&self, data: &[u8]) -> Result<CompressedData> {
        let algorithm = self.select_best_algorithm(data);

        let start = Instant::now();
        let result = match algorithm {
            AdvancedCompressionType::RunLength => {
                let encoder = RunLengthEncoder;
                encoder.compress(data)?
            }
            AdvancedCompressionType::Delta => {
                let encoder = DeltaEncoder;
                encoder.compress(data)?
            }
            AdvancedCompressionType::FrameOfReference => {
                let encoder = FrameOfReferenceEncoder::default();
                encoder.compress(data)?
            }
            AdvancedCompressionType::AdaptiveDictionary => {
                let encoder = AdaptiveDictionary::new();
                encoder.compress(data)?
            }
            AdvancedCompressionType::BitmapWAH => {
                let encoder = BitmapWAHEncoder;
                encoder.compress(data)?
            }
            AdvancedCompressionType::BitmapRoaring => {
                let encoder = BitmapRoaringEncoder;
                encoder.compress(data)?
            }
            AdvancedCompressionType::ColumnStore => {
                let encoder = ColumnStoreCompressor::new();
                encoder.compress(data)?
            }
            AdvancedCompressionType::Lz4 => {
                use crate::compression::Lz4Compressor;
                let encoder = Lz4Compressor::balanced();
                encoder.compress(data)?
            }
            AdvancedCompressionType::Zstd => {
                use crate::compression::ZstdCompressor;
                let encoder = ZstdCompressor::balanced();
                encoder.compress(data)?
            }
            AdvancedCompressionType::Snappy => {
                use crate::compression::SnappyCompressor;
                let encoder = SnappyCompressor::new();
                encoder.compress(data)?
            }
            AdvancedCompressionType::Brotli => {
                use crate::compression::BrotliCompressor;
                let encoder = BrotliCompressor::balanced();
                encoder.compress(data)?
            }
            AdvancedCompressionType::Adaptive => {
                return Err(anyhow!("Recursive adaptive compression not allowed"));
            }
        };
        let selection_time = start.elapsed();

        // Override metadata to indicate this was adaptively selected
        let mut metadata = result.metadata;
        metadata.algorithm = AdvancedCompressionType::Adaptive;
        metadata.compression_time_us += selection_time.as_micros() as u64;
        metadata
            .metadata
            .insert("selected_algorithm".to_string(), algorithm.to_string());

        Ok(CompressedData {
            data: result.data,
            metadata,
        })
    }

    /// Decompress adaptively compressed data
    pub fn decompress_adaptive(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        if compressed.metadata.algorithm != AdvancedCompressionType::Adaptive {
            return Err(anyhow!(
                "Invalid compression algorithm: expected Adaptive, got {}",
                compressed.metadata.algorithm
            ));
        }

        // Get the actual algorithm that was used
        let selected_algorithm = compressed
            .metadata
            .metadata
            .get("selected_algorithm")
            .ok_or_else(|| {
                anyhow!("Missing selected algorithm in adaptive compression metadata")
            })?;

        let algorithm = match selected_algorithm.as_str() {
            "RunLength" => AdvancedCompressionType::RunLength,
            "Delta" => AdvancedCompressionType::Delta,
            "FrameOfReference" => AdvancedCompressionType::FrameOfReference,
            "AdaptiveDictionary" => AdvancedCompressionType::AdaptiveDictionary,
            "BitmapWAH" => AdvancedCompressionType::BitmapWAH,
            "BitmapRoaring" => AdvancedCompressionType::BitmapRoaring,
            "ColumnStore" => AdvancedCompressionType::ColumnStore,
            "LZ4" => AdvancedCompressionType::Lz4,
            "Zstd" => AdvancedCompressionType::Zstd,
            "Snappy" => AdvancedCompressionType::Snappy,
            "Brotli" => AdvancedCompressionType::Brotli,
            _ => {
                return Err(anyhow!(
                    "Unknown selected algorithm: {}",
                    selected_algorithm
                ))
            }
        };

        // Create temporary compressed data with the original algorithm
        let mut temp_compressed = compressed.clone();
        temp_compressed.metadata.algorithm = algorithm;

        // Decompress using the original algorithm
        match algorithm {
            AdvancedCompressionType::RunLength => {
                let encoder = RunLengthEncoder;
                encoder.decompress(&temp_compressed)
            }
            AdvancedCompressionType::Delta => {
                let encoder = DeltaEncoder;
                encoder.decompress(&temp_compressed)
            }
            AdvancedCompressionType::FrameOfReference => {
                let encoder = FrameOfReferenceEncoder::default();
                encoder.decompress(&temp_compressed)
            }
            AdvancedCompressionType::AdaptiveDictionary => {
                let encoder = AdaptiveDictionary::new();
                encoder.decompress(&temp_compressed)
            }
            AdvancedCompressionType::BitmapWAH => {
                let encoder = BitmapWAHEncoder;
                encoder.decompress(&temp_compressed)
            }
            AdvancedCompressionType::BitmapRoaring => {
                let encoder = BitmapRoaringEncoder;
                encoder.decompress(&temp_compressed)
            }
            AdvancedCompressionType::ColumnStore => {
                let encoder = ColumnStoreCompressor::new();
                encoder.decompress(&temp_compressed)
            }
            AdvancedCompressionType::Lz4 => {
                use crate::compression::Lz4Compressor;
                let encoder = Lz4Compressor::balanced();
                encoder.decompress(&temp_compressed)
            }
            AdvancedCompressionType::Zstd => {
                use crate::compression::ZstdCompressor;
                let encoder = ZstdCompressor::balanced();
                encoder.decompress(&temp_compressed)
            }
            AdvancedCompressionType::Snappy => {
                use crate::compression::SnappyCompressor;
                let encoder = SnappyCompressor::new();
                encoder.decompress(&temp_compressed)
            }
            AdvancedCompressionType::Brotli => {
                use crate::compression::BrotliCompressor;
                let encoder = BrotliCompressor::balanced();
                encoder.decompress(&temp_compressed)
            }
            AdvancedCompressionType::Adaptive => {
                Err(anyhow!("Recursive adaptive compression not allowed"))
            }
        }
    }
}

impl Default for AdaptiveCompressor {
    fn default() -> Self {
        Self::new(1024, 0.8) // Default sample size and threshold
    }
}

impl CompressionAlgorithm for AdaptiveCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        self.compress_adaptive(data)
    }

    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        self.decompress_adaptive(compressed)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::Adaptive
    }
}

/// Data statistics for algorithm selection
#[derive(Debug, Default)]
struct DataStats {
    length: usize,
    unique_bytes: usize,
    sparsity: f64,
    repetition_ratio: f64,
    byte_runs: usize,
    bit_runs: usize,
    is_sorted: bool,
    range_ratio: f64,
    delta_efficiency: f64,
    dictionary_efficiency: f64,
    structured_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_compressor_basic() {
        let compressor = AdaptiveCompressor::default();

        // Test with repetitive data (should select RunLength)
        let repetitive_data = vec![1u8; 100];
        let algorithm = compressor.select_best_algorithm(&repetitive_data);
        assert_eq!(algorithm, AdvancedCompressionType::RunLength);

        let compressed = compressor.compress(&repetitive_data).unwrap();
        assert_eq!(
            compressed.metadata.algorithm,
            AdvancedCompressionType::Adaptive
        );
        assert!(compressed
            .metadata
            .metadata
            .contains_key("selected_algorithm"));

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(repetitive_data, decompressed);
    }

    #[test]
    fn test_sparse_data_selection() {
        let compressor = AdaptiveCompressor::default();

        // Create sparse data (mostly zeros)
        let mut sparse_data = vec![0u8; 1000];
        sparse_data[10] = 1;
        sparse_data[100] = 1;
        sparse_data[500] = 1;

        let algorithm = compressor.select_best_algorithm(&sparse_data);
        // Should select a bitmap algorithm for sparse data
        assert!(matches!(
            algorithm,
            AdvancedCompressionType::BitmapWAH | AdvancedCompressionType::BitmapRoaring
        ));
    }

    #[test]
    fn test_empty_data() {
        let compressor = AdaptiveCompressor::default();
        let data = vec![];

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_small_data() {
        let compressor = AdaptiveCompressor::default();
        let data = vec![1, 2, 3];

        let algorithm = compressor.select_best_algorithm(&data);
        assert_eq!(algorithm, AdvancedCompressionType::RunLength); // Default for small data
    }

    #[test]
    fn test_data_analysis() {
        let compressor = AdaptiveCompressor::default();

        // Test with sorted numeric data
        let sorted_data: Vec<u8> = (0..64u64).flat_map(|i| i.to_le_bytes()).collect();

        let stats = compressor.analyze_data(&sorted_data);
        assert!(stats.is_sorted);
        assert!(stats.delta_efficiency > 0.5);
    }
}

// ============================================================================
// AdaptiveCodec - v0.3.0 lightweight adaptive codec with CodecType selection
// ============================================================================

use crate::error::{Result as TdbResult, TdbError};

/// Codec selection result for [`AdaptiveCodec`].
///
/// Each variant represents a distinct compression strategy chosen adaptively
/// based on data characteristics (size, entropy, structure).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecType {
    /// Zstd without a pre-trained dictionary (best for data > threshold)
    ZstdBasic,
    /// Zstd with a pre-trained Zstd dictionary (best for small repetitive blocks)
    ZstdDict,
    /// Delta encoding for sorted or near-sorted numerical streams
    Delta,
    /// No compression (data is incompressible or below size threshold)
    None,
}

impl std::fmt::Display for CodecType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecType::ZstdBasic => write!(f, "ZstdBasic"),
            CodecType::ZstdDict => write!(f, "ZstdDict"),
            CodecType::Delta => write!(f, "Delta"),
            CodecType::None => write!(f, "None"),
        }
    }
}

/// Magic byte prepended to adaptive-compressed output to identify the [`CodecType`].
const CODEC_MAGIC_ZSTD_BASIC: u8 = 0x01;
const CODEC_MAGIC_ZSTD_DICT: u8 = 0x02;
const CODEC_MAGIC_DELTA: u8 = 0x03;
const CODEC_MAGIC_NONE: u8 = 0x00;

/// A lightweight adaptive codec that selects the best compression strategy based
/// on the data being compressed.
///
/// `AdaptiveCodec` serves as the primary entry point for v0.3.0 compression.
/// It examines data characteristics (size, byte entropy, byte-delta statistics)
/// and dispatches to the most appropriate codec:
///
/// | Condition | Selected Codec |
/// |-----------|---------------|
/// | `data.len() < threshold_bytes` | `CodecType::None` |
/// | Has Zstd dictionary **and** data appears text-like | `CodecType::ZstdDict` |
/// | Data is sorted/near-sorted u8 stream | `CodecType::Delta` |
/// | Otherwise | `CodecType::ZstdBasic` |
///
/// The compressed output is self-describing: a single-byte header encodes the
/// codec type so that [`decompress`](AdaptiveCodec::decompress) can always recover
/// the original data without out-of-band metadata.
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_tdb::compression::adaptive::{AdaptiveCodec, CodecType};
///
/// let codec = AdaptiveCodec { threshold_bytes: 64, zstd_dict: None, zstd_level: 3 };
/// let data = b"<http://example.org/s> <p> <o> .".repeat(10);
/// let (compressed, chosen) = codec.compress(&data).unwrap();
/// let decompressed = codec.decompress(&compressed, chosen).unwrap();
/// assert_eq!(decompressed, data);
/// ```
pub struct AdaptiveCodec {
    /// Minimum data size (bytes) required before compression is attempted.
    /// Data smaller than this threshold is stored as-is (`CodecType::None`).
    pub threshold_bytes: usize,
    /// Optional pre-trained Zstd dictionary to use for `CodecType::ZstdDict`.
    pub zstd_dict: Option<Vec<u8>>,
    /// Zstd compression level used when dispatching to Zstd codecs (1-22).
    pub zstd_level: i32,
}

impl AdaptiveCodec {
    /// Create a new `AdaptiveCodec` with the given threshold and no dictionary.
    ///
    /// # Arguments
    ///
    /// * `threshold_bytes` - Minimum data size to apply compression
    pub fn new(threshold_bytes: usize) -> Self {
        Self {
            threshold_bytes,
            zstd_dict: None,
            zstd_level: 3,
        }
    }

    /// Create a new `AdaptiveCodec` with a pre-trained Zstd dictionary.
    pub fn with_dict(threshold_bytes: usize, dict: Vec<u8>, zstd_level: i32) -> Self {
        Self {
            threshold_bytes,
            zstd_dict: Some(dict),
            zstd_level: zstd_level.clamp(1, 22),
        }
    }

    /// Select the most appropriate codec for the given data.
    ///
    /// The decision is based on:
    /// 1. Data size vs `threshold_bytes`
    /// 2. Availability of a Zstd dictionary
    /// 3. Byte entropy / sortedness heuristics
    pub fn select_codec(&self, data: &[u8]) -> CodecType {
        if data.len() < self.threshold_bytes {
            return CodecType::None;
        }

        // If we have a dictionary and the data looks text-like (high ASCII printable ratio)
        // prefer dictionary compression for smaller blocks
        if self.zstd_dict.is_some() {
            let printable_ratio = Self::compute_printable_ratio(data);
            // Text/URI data benefits most from dictionary compression
            if printable_ratio > 0.7 {
                return CodecType::ZstdDict;
            }
        }

        // Check if data is sorted/near-sorted: delta encoding excels here
        if Self::is_mostly_sorted(data) {
            return CodecType::Delta;
        }

        // Fall back to basic Zstd
        CodecType::ZstdBasic
    }

    /// Compress `data` using adaptively selected codec.
    ///
    /// Returns `(compressed_bytes, chosen_codec)`. The compressed bytes include a
    /// single-byte codec marker as the first byte so [`decompress`](AdaptiveCodec::decompress)
    /// can determine the algorithm without external metadata.
    pub fn compress(&self, data: &[u8]) -> TdbResult<(Vec<u8>, CodecType)> {
        let codec = self.select_codec(data);
        let compressed = match codec {
            CodecType::None => {
                let mut out = Vec::with_capacity(1 + data.len());
                out.push(CODEC_MAGIC_NONE);
                out.extend_from_slice(data);
                out
            }
            CodecType::ZstdBasic => {
                let payload = Self::zstd_compress_raw(data, self.zstd_level, None)?;
                let mut out = Vec::with_capacity(1 + payload.len());
                out.push(CODEC_MAGIC_ZSTD_BASIC);
                out.extend_from_slice(&payload);
                out
            }
            CodecType::ZstdDict => {
                let dict = self.zstd_dict.as_deref().ok_or_else(|| {
                    TdbError::Other("ZstdDict selected but no dictionary loaded".to_string())
                })?;
                let payload = Self::zstd_compress_raw(data, self.zstd_level, Some(dict))?;
                let mut out = Vec::with_capacity(1 + payload.len());
                out.push(CODEC_MAGIC_ZSTD_DICT);
                out.extend_from_slice(&payload);
                out
            }
            CodecType::Delta => {
                let payload = Self::delta_compress(data)?;
                let mut out = Vec::with_capacity(1 + payload.len());
                out.push(CODEC_MAGIC_DELTA);
                out.extend_from_slice(&payload);
                out
            }
        };
        Ok((compressed, codec))
    }

    /// Decompress data that was compressed by [`AdaptiveCodec::compress`].
    ///
    /// The `codec` argument must match the codec used during compression (the same value
    /// returned from `compress`). Alternatively, use [`decompress_auto`](AdaptiveCodec::decompress_auto)
    /// to auto-detect from the embedded header byte.
    pub fn decompress(&self, data: &[u8], codec: CodecType) -> TdbResult<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        // data.len() >= 1 guaranteed by is_empty check above
        let header = data[0];
        let payload = &data[1..];

        // Verify header matches expected codec
        let expected_header = match codec {
            CodecType::None => CODEC_MAGIC_NONE,
            CodecType::ZstdBasic => CODEC_MAGIC_ZSTD_BASIC,
            CodecType::ZstdDict => CODEC_MAGIC_ZSTD_DICT,
            CodecType::Delta => CODEC_MAGIC_DELTA,
        };
        if header != expected_header {
            return Err(TdbError::Other(format!(
                "Codec mismatch: header byte 0x{:02x} does not match expected codec {}",
                header, codec
            )));
        }

        self.decompress_payload(payload, codec)
    }

    /// Decompress data, auto-detecting the codec from the embedded header byte.
    ///
    /// This is more convenient than [`decompress`](AdaptiveCodec::decompress) when the
    /// original `CodecType` is not stored separately.
    pub fn decompress_auto(&self, data: &[u8]) -> TdbResult<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        // data.len() >= 1 guaranteed by is_empty check above
        let header = data[0];
        let payload = &data[1..];

        let codec = match header {
            CODEC_MAGIC_NONE => CodecType::None,
            CODEC_MAGIC_ZSTD_BASIC => CodecType::ZstdBasic,
            CODEC_MAGIC_ZSTD_DICT => CodecType::ZstdDict,
            CODEC_MAGIC_DELTA => CodecType::Delta,
            _ => {
                return Err(TdbError::Other(format!(
                    "Unknown codec header byte: 0x{:02x}",
                    header
                )))
            }
        };

        self.decompress_payload(payload, codec)
    }

    // -- Private helpers --

    fn decompress_payload(&self, payload: &[u8], codec: CodecType) -> TdbResult<Vec<u8>> {
        match codec {
            CodecType::None => Ok(payload.to_vec()),
            CodecType::ZstdBasic => Self::zstd_decompress_raw(payload, None),
            CodecType::ZstdDict => {
                let dict = self.zstd_dict.as_deref().ok_or_else(|| {
                    TdbError::Other(
                        "ZstdDict decompression requested but no dictionary loaded".to_string(),
                    )
                })?;
                Self::zstd_decompress_raw(payload, Some(dict))
            }
            CodecType::Delta => Self::delta_decompress(payload),
        }
    }

    fn zstd_compress_raw(data: &[u8], level: i32, dict: Option<&[u8]>) -> TdbResult<Vec<u8>> {
        use std::io::Write;
        let compressed = match dict {
            None => {
                let mut enc = zstd::Encoder::new(Vec::new(), level)
                    .map_err(|e| TdbError::Other(format!("zstd init: {e}")))?;
                enc.write_all(data)
                    .map_err(|e| TdbError::Other(format!("zstd write: {e}")))?;
                enc.finish()
                    .map_err(|e| TdbError::Other(format!("zstd finish: {e}")))?
            }
            Some(d) => {
                let mut enc = zstd::Encoder::with_dictionary(Vec::new(), level, d)
                    .map_err(|e| TdbError::Other(format!("zstd dict init: {e}")))?;
                enc.write_all(data)
                    .map_err(|e| TdbError::Other(format!("zstd dict write: {e}")))?;
                enc.finish()
                    .map_err(|e| TdbError::Other(format!("zstd dict finish: {e}")))?
            }
        };
        Ok(compressed)
    }

    fn zstd_decompress_raw(data: &[u8], dict: Option<&[u8]>) -> TdbResult<Vec<u8>> {
        use std::io::Read;
        match dict {
            None => {
                let mut dec = zstd::Decoder::new(data)
                    .map_err(|e| TdbError::Other(format!("zstd dec init: {e}")))?;
                let mut out = Vec::new();
                dec.read_to_end(&mut out)
                    .map_err(|e| TdbError::Other(format!("zstd decompress: {e}")))?;
                Ok(out)
            }
            Some(d) => {
                let mut dec = zstd::Decoder::with_dictionary(data, d)
                    .map_err(|e| TdbError::Other(format!("zstd dict dec init: {e}")))?;
                let mut out = Vec::new();
                dec.read_to_end(&mut out)
                    .map_err(|e| TdbError::Other(format!("zstd dict decompress: {e}")))?;
                Ok(out)
            }
        }
    }

    /// Simple byte delta encoding: store first byte, then signed i16 deltas.
    fn delta_compress(data: &[u8]) -> TdbResult<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(1 + (data.len() - 1) * 2);
        out.push(data[0]);
        for i in 1..data.len() {
            let delta = data[i] as i16 - data[i - 1] as i16;
            out.extend_from_slice(&delta.to_le_bytes());
        }
        Ok(out)
    }

    fn delta_decompress(encoded: &[u8]) -> TdbResult<Vec<u8>> {
        if encoded.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        out.push(encoded[0]);
        let mut current = encoded[0] as i16;
        for chunk in encoded[1..].chunks_exact(2) {
            let delta = i16::from_le_bytes([chunk[0], chunk[1]]);
            current = (current + delta).clamp(0, 255);
            out.push(current as u8);
        }
        Ok(out)
    }

    /// Fraction of bytes in `data` that are printable ASCII (0x20-0x7E).
    fn compute_printable_ratio(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let printable = data.iter().filter(|&&b| (0x20..=0x7e).contains(&b)).count();
        printable as f64 / data.len() as f64
    }

    /// Heuristic: returns `true` if >= 80% of consecutive byte pairs are non-decreasing.
    fn is_mostly_sorted(data: &[u8]) -> bool {
        if data.len() < 2 {
            return false;
        }
        let non_decreasing = data.windows(2).filter(|w| w[1] >= w[0]).count();
        non_decreasing as f64 / (data.len() - 1) as f64 >= 0.80
    }
}

impl Default for AdaptiveCodec {
    fn default() -> Self {
        Self::new(64)
    }
}

#[cfg(test)]
mod adaptive_codec_tests {
    use super::*;

    #[test]
    fn test_adaptive_codec_none_below_threshold() {
        let codec = AdaptiveCodec::new(1000);
        let data = b"small data".to_vec();
        let selected = codec.select_codec(&data);
        assert_eq!(
            selected,
            CodecType::None,
            "small data below threshold should select None"
        );
    }

    #[test]
    fn test_adaptive_codec_zstd_basic_roundtrip() {
        let codec = AdaptiveCodec::new(10);
        // Not sorted, not text-like: should pick ZstdBasic
        let data: Vec<u8> = (0..200u8).rev().collect(); // descending bytes
        let (compressed, chosen) = codec.compress(&data).expect("compress");
        // Accept either ZstdBasic or Delta (both are valid adaptive choices)
        assert!(
            chosen == CodecType::ZstdBasic || chosen == CodecType::Delta,
            "expected ZstdBasic or Delta, got {:?}",
            chosen
        );
        let decompressed = codec.decompress(&compressed, chosen).expect("decompress");
        assert_eq!(decompressed, data, "ZstdBasic roundtrip must be identical");
    }

    #[test]
    fn test_adaptive_codec_zstd_dict_roundtrip() {
        use crate::compression::zstd_codec::ZstdCodec;

        // Train a dictionary
        let samples: Vec<Vec<u8>> = (0..300)
            .map(|i| {
                format!(
                    "<http://example.org/r/{}> <http://schema.org/name> \"Name {}\" .",
                    i, i
                )
                .into_bytes()
            })
            .collect();
        let dict = ZstdCodec::train_dict(&samples, 4096).expect("train dict");

        let codec = AdaptiveCodec::with_dict(10, dict, 3);
        // Text data with a dictionary should prefer ZstdDict
        let data = b"<http://example.org/r/42> <http://schema.org/name> \"Name 42\" .".repeat(5);
        let (compressed, chosen) = codec.compress(&data).expect("compress with dict");
        assert_eq!(
            chosen,
            CodecType::ZstdDict,
            "text-like data with dict should select ZstdDict"
        );

        let decompressed = codec
            .decompress(&compressed, chosen)
            .expect("decompress with dict");
        assert_eq!(decompressed, data, "ZstdDict roundtrip must be identical");
    }

    #[test]
    fn test_adaptive_codec_delta_roundtrip() {
        let codec = AdaptiveCodec::new(10);
        // Sorted bytes: delta encoding should be selected
        let data: Vec<u8> = (0..200u8).collect();
        let (compressed, chosen) = codec.compress(&data).expect("compress sorted");
        assert_eq!(
            chosen,
            CodecType::Delta,
            "sorted data should select Delta codec"
        );

        let decompressed = codec
            .decompress(&compressed, chosen)
            .expect("decompress delta");
        assert_eq!(decompressed, data, "Delta roundtrip must be identical");
    }

    #[test]
    fn test_adaptive_codec_none_roundtrip() {
        let codec = AdaptiveCodec::new(1000);
        let data = b"small".to_vec();
        let (compressed, chosen) = codec.compress(&data).expect("compress none");
        assert_eq!(chosen, CodecType::None);

        let decompressed = codec
            .decompress(&compressed, chosen)
            .expect("decompress none");
        assert_eq!(decompressed, data, "None codec roundtrip must be identical");
    }

    #[test]
    fn test_adaptive_codec_decompress_auto() {
        let codec = AdaptiveCodec::new(10);
        let data = b"Auto-detect decompression test data for the adaptive codec.".repeat(10);
        let (compressed, _chosen) = codec.compress(&data).expect("compress");
        let decompressed = codec.decompress_auto(&compressed).expect("decompress auto");
        assert_eq!(
            decompressed, data,
            "decompress_auto roundtrip must be identical"
        );
    }

    #[test]
    fn test_adaptive_codec_empty_data() {
        let codec = AdaptiveCodec::new(64);
        let (compressed, chosen) = codec.compress(&[]).expect("compress empty");
        // Empty data is below threshold: CodecType::None
        assert_eq!(chosen, CodecType::None);
        let decompressed = codec
            .decompress(&compressed, chosen)
            .expect("decompress empty");
        assert_eq!(decompressed, b"");
    }

    #[test]
    fn test_adaptive_codec_codec_type_display() {
        assert_eq!(CodecType::ZstdBasic.to_string(), "ZstdBasic");
        assert_eq!(CodecType::ZstdDict.to_string(), "ZstdDict");
        assert_eq!(CodecType::Delta.to_string(), "Delta");
        assert_eq!(CodecType::None.to_string(), "None");
    }

    #[test]
    fn test_adaptive_codec_wrong_codec_type_error() {
        let codec = AdaptiveCodec::new(10);
        let data: Vec<u8> = (0..200u8).collect(); // sorted -> Delta
        let (compressed, _chosen) = codec.compress(&data).expect("compress");
        // Deliberately pass wrong codec type
        let result = codec.decompress(&compressed, CodecType::ZstdBasic);
        assert!(result.is_err(), "wrong CodecType should return error");
    }

    #[test]
    fn test_adaptive_codec_large_rdf_data() {
        let codec = AdaptiveCodec::new(64);
        let rdf_triple = b"<http://dbpedia.org/resource/Thing> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Thing> .\n";
        let mut data = Vec::new();
        for _ in 0..500 {
            data.extend_from_slice(rdf_triple);
        }

        let (compressed, chosen) = codec.compress(&data).expect("compress large RDF");
        println!(
            "Large RDF: {} -> {} bytes ({:?})",
            data.len(),
            compressed.len(),
            chosen
        );

        let decompressed = codec
            .decompress(&compressed, chosen)
            .expect("decompress large RDF");
        assert_eq!(decompressed, data, "large RDF roundtrip must be identical");
        // Should achieve compression
        assert!(
            compressed.len() < data.len(),
            "large repetitive RDF should compress, ratio: {:.2}",
            compressed.len() as f64 / data.len() as f64
        );
    }
}
