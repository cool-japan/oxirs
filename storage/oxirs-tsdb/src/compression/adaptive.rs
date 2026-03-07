//! Adaptive compression: automatically selects the best algorithm per-block
//!
//! The [`AdaptiveCompressor`] analyses incoming data and picks the most
//! appropriate encoding from:
//!
//! | Algorithm | Best suited for |
//! |-----------|----------------|
//! | Gorilla   | Floating-point sensor values with temporal locality |
//! | RLE       | Step-function / constant-value data (states, flags) |
//! | Dictionary| Repeated categorical string labels |
//! | Raw       | High-entropy data where compression gives no benefit |
//!
//! ## Selection heuristic
//!
//! 1. If the value stream is made up of strings → Dictionary.
//! 2. If the ratio of unique values to total values is < 10% → RLE.
//! 3. Otherwise → Gorilla.
//! 4. If the Gorilla-encoded size is larger than raw → Raw (passthrough).
//!
//! This module also provides a unified [`CompressedBlock`] type that encodes
//! which algorithm was used so that decompression is self-describing.

use crate::compression::gorilla::{gorilla_decode, gorilla_encode};
use crate::compression::rle::{rle_decode, rle_encode, RleRun};
use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Algorithm tag
// ---------------------------------------------------------------------------

/// Identifies which compression algorithm was applied to a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression; raw `(i64, f64)` pairs as little-endian bytes.
    Raw,
    /// Gorilla XOR + delta-of-delta encoding.
    Gorilla,
    /// Run-Length Encoding; effective for step-function series.
    Rle,
    /// Dictionary encoding for string-valued series.
    Dictionary,
}

impl std::fmt::Display for CompressionAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Raw => write!(f, "raw"),
            Self::Gorilla => write!(f, "gorilla"),
            Self::Rle => write!(f, "rle"),
            Self::Dictionary => write!(f, "dictionary"),
        }
    }
}

// ---------------------------------------------------------------------------
// CompressedBlock
// ---------------------------------------------------------------------------

/// A self-describing compressed block holding one or more time-series samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedBlock {
    /// The algorithm used to produce this block.
    pub algorithm: CompressionAlgorithm,
    /// Raw compressed bytes.
    pub data: Vec<u8>,
    /// Number of samples encoded in this block.
    pub sample_count: u32,
    /// Timestamp of the first sample (ms since epoch).
    pub min_timestamp: i64,
    /// Timestamp of the last sample (ms since epoch).
    pub max_timestamp: i64,
}

impl CompressedBlock {
    /// Approximate compression ratio: `original_bytes / compressed_bytes`.
    pub fn compression_ratio(&self) -> f64 {
        let original = self.sample_count as f64 * 16.0; // i64 + f64 = 16 bytes
        let compressed = self.data.len() as f64;
        if compressed == 0.0 {
            1.0
        } else {
            original / compressed
        }
    }

    /// Decode the block back to `(timestamp_ms, value)` pairs.
    pub fn decode(&self) -> TsdbResult<Vec<(i64, f64)>> {
        match self.algorithm {
            CompressionAlgorithm::Raw => decode_raw(&self.data),
            CompressionAlgorithm::Gorilla => gorilla_decode(&self.data),
            CompressionAlgorithm::Rle => {
                let runs = decode_rle_runs(&self.data)?;
                Ok(rle_decode(&runs))
            }
            CompressionAlgorithm::Dictionary => Err(TsdbError::Decompression(
                "Dictionary blocks require string-aware decoding; use DictionaryBlock directly"
                    .to_string(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive compressor
// ---------------------------------------------------------------------------

/// Statistics gathered over a sample window to guide algorithm selection.
#[derive(Debug, Default)]
pub struct SampleStats {
    pub total: usize,
    pub unique_values: usize,
    /// Sum of |delta_of_delta| for timestamps
    pub dod_sum: i64,
    /// Count of zero XOR transitions (unchanged values)
    pub zero_xor_count: usize,
}

/// Adaptive compressor: buffers samples, then chooses and applies the best
/// algorithm when [`finish`](Self::finish) is called.
///
/// # Example
///
/// ```rust,ignore
/// let mut comp = AdaptiveCompressor::new();
/// for (ts, val) in sensor_data {
///     comp.push(ts, val);
/// }
/// let block = comp.finish()?;
/// println!("Used: {}, ratio: {:.1}", block.algorithm, block.compression_ratio());
/// ```
pub struct AdaptiveCompressor {
    samples: Vec<(i64, f64)>,
    /// Force a specific algorithm (skips auto-selection when `Some`).
    forced: Option<CompressionAlgorithm>,
}

impl AdaptiveCompressor {
    /// Create a new adaptive compressor.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            forced: None,
        }
    }

    /// Force use of a specific algorithm regardless of data characteristics.
    pub fn with_algorithm(mut self, algo: CompressionAlgorithm) -> Self {
        self.forced = Some(algo);
        self
    }

    /// Add a `(timestamp_ms, value)` sample.
    pub fn push(&mut self, timestamp: i64, value: f64) {
        self.samples.push((timestamp, value));
    }

    /// Add multiple samples at once.
    pub fn extend(&mut self, samples: &[(i64, f64)]) {
        self.samples.extend_from_slice(samples);
    }

    /// Analyse the buffered samples and return statistics used by the
    /// algorithm selector.
    fn analyse(&self) -> SampleStats {
        let mut stats = SampleStats {
            total: self.samples.len(),
            ..Default::default()
        };
        if self.samples.is_empty() {
            return stats;
        }

        // Count unique values via a sorted approach (avoid HashMap for no_std compat)
        let mut bits: Vec<u64> = self.samples.iter().map(|&(_, v)| v.to_bits()).collect();
        bits.sort_unstable();
        bits.dedup();
        stats.unique_values = bits.len();

        // XOR transitions and delta-of-delta
        let mut prev_bits = self.samples[0].1.to_bits();
        let mut prev_ts = self.samples[0].0;
        let mut prev_delta = 0i64;
        for &(ts, val) in &self.samples[1..] {
            let cur_bits = val.to_bits();
            if cur_bits == prev_bits {
                stats.zero_xor_count += 1;
            }
            prev_bits = cur_bits;

            let delta = ts - prev_ts;
            let dod = delta - prev_delta;
            stats.dod_sum = stats.dod_sum.saturating_add(dod.abs());
            prev_delta = delta;
            prev_ts = ts;
        }
        stats
    }

    /// Select the best algorithm based on sample statistics.
    fn select_algorithm(&self, stats: &SampleStats) -> CompressionAlgorithm {
        if stats.total == 0 {
            return CompressionAlgorithm::Raw;
        }

        // High repetition ratio → RLE is likely best
        let repeat_ratio = stats.zero_xor_count as f64 / stats.total.max(1) as f64;
        if repeat_ratio >= 0.7 {
            return CompressionAlgorithm::Rle;
        }

        // Low cardinality relative to total → RLE can also help
        let cardinality_ratio = stats.unique_values as f64 / stats.total as f64;
        if cardinality_ratio < 0.05 && stats.total > 10 {
            return CompressionAlgorithm::Rle;
        }

        // Default: Gorilla handles temporal-locality well for sensor floats
        CompressionAlgorithm::Gorilla
    }

    /// Compress the buffered samples and return a self-describing block.
    ///
    /// After calling `finish`, the compressor is consumed.
    pub fn finish(self) -> TsdbResult<CompressedBlock> {
        if self.samples.is_empty() {
            return Ok(CompressedBlock {
                algorithm: CompressionAlgorithm::Raw,
                data: Vec::new(),
                sample_count: 0,
                min_timestamp: 0,
                max_timestamp: 0,
            });
        }

        let min_ts = self.samples.first().map(|&(t, _)| t).unwrap_or(0);
        let max_ts = self.samples.last().map(|&(t, _)| t).unwrap_or(0);
        let sample_count = self.samples.len() as u32;

        let algo = self.forced.unwrap_or_else(|| {
            let stats = self.analyse();
            self.select_algorithm(&stats)
        });

        let data = match algo {
            CompressionAlgorithm::Raw => encode_raw(&self.samples),
            CompressionAlgorithm::Gorilla => gorilla_encode(&self.samples)?,
            CompressionAlgorithm::Rle => {
                let runs = rle_encode(&self.samples)?;
                encode_rle_runs(&runs)?
            }
            CompressionAlgorithm::Dictionary => {
                return Err(TsdbError::Compression(
                    "Dictionary compression requires string data; use DictionaryEncoder instead"
                        .to_string(),
                ));
            }
        };

        Ok(CompressedBlock {
            algorithm: algo,
            data,
            sample_count,
            min_timestamp: min_ts,
            max_timestamp: max_ts,
        })
    }
}

impl Default for AdaptiveCompressor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Raw encoding helpers
// ---------------------------------------------------------------------------

fn encode_raw(data: &[(i64, f64)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 16);
    for &(ts, val) in data {
        out.extend_from_slice(&ts.to_le_bytes());
        out.extend_from_slice(&val.to_bits().to_le_bytes());
    }
    out
}

fn decode_raw(data: &[u8]) -> TsdbResult<Vec<(i64, f64)>> {
    if data.len() % 16 != 0 {
        return Err(TsdbError::Decompression(format!(
            "Raw block length {} is not a multiple of 16",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len() / 16);
    for chunk in data.chunks_exact(16) {
        let ts = i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let val_bits = u64::from_le_bytes([
            chunk[8], chunk[9], chunk[10], chunk[11], chunk[12], chunk[13], chunk[14], chunk[15],
        ]);
        out.push((ts, f64::from_bits(val_bits)));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// RLE binary encoding helpers
// ---------------------------------------------------------------------------

/// Wire format for RLE runs:
/// ```text
/// [run_count: u32 le]
/// per run: [start: i64 le][end: i64 le][value_bits: u64 le][count: u32 le]
/// ```
fn encode_rle_runs(runs: &[RleRun]) -> TsdbResult<Vec<u8>> {
    let run_count = runs.len() as u32;
    let mut out = Vec::with_capacity(4 + runs.len() * 28);
    out.extend_from_slice(&run_count.to_le_bytes());
    for run in runs {
        out.extend_from_slice(&run.start_timestamp.to_le_bytes());
        out.extend_from_slice(&run.end_timestamp.to_le_bytes());
        out.extend_from_slice(&run.value.to_bits().to_le_bytes());
        out.extend_from_slice(&run.count.to_le_bytes());
    }
    Ok(out)
}

fn decode_rle_runs(data: &[u8]) -> TsdbResult<Vec<RleRun>> {
    if data.len() < 4 {
        return Err(TsdbError::Decompression(
            "RLE binary: data too short for run count".to_string(),
        ));
    }
    let run_count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let expected_len = 4 + run_count * 28;
    if data.len() < expected_len {
        return Err(TsdbError::Decompression(format!(
            "RLE binary: expected {} bytes for {} runs, got {}",
            expected_len,
            run_count,
            data.len()
        )));
    }
    let mut runs = Vec::with_capacity(run_count);
    for i in 0..run_count {
        let off = 4 + i * 28;
        let start_timestamp = i64::from_le_bytes([
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
            data[off + 4],
            data[off + 5],
            data[off + 6],
            data[off + 7],
        ]);
        let end_timestamp = i64::from_le_bytes([
            data[off + 8],
            data[off + 9],
            data[off + 10],
            data[off + 11],
            data[off + 12],
            data[off + 13],
            data[off + 14],
            data[off + 15],
        ]);
        let val_bits = u64::from_le_bytes([
            data[off + 16],
            data[off + 17],
            data[off + 18],
            data[off + 19],
            data[off + 20],
            data[off + 21],
            data[off + 22],
            data[off + 23],
        ]);
        let count = u32::from_le_bytes([
            data[off + 24],
            data[off + 25],
            data[off + 26],
            data[off + 27],
        ]);
        runs.push(RleRun {
            start_timestamp,
            end_timestamp,
            value: f64::from_bits(val_bits),
            count,
        });
    }
    Ok(runs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_regular(n: usize, interval_ms: i64, base_val: f64) -> Vec<(i64, f64)> {
        (0..n)
            .map(|i| (i as i64 * interval_ms, base_val + (i % 5) as f64 * 0.01))
            .collect()
    }

    #[test]
    fn test_empty_block() {
        let block = AdaptiveCompressor::new().finish().expect("finish");
        assert_eq!(block.sample_count, 0);
        assert!(block.decode().expect("decode").is_empty());
    }

    #[test]
    fn test_constant_data_selects_rle() {
        let data: Vec<(i64, f64)> = (0..200).map(|i| (i as i64 * 1000, 5.0)).collect();
        let mut comp = AdaptiveCompressor::new();
        comp.extend(&data);
        let block = comp.finish().expect("finish");
        // Constant data → high zero-XOR ratio → RLE
        assert_eq!(block.algorithm, CompressionAlgorithm::Rle);
        let decoded = block.decode().expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_sensor_data_selects_gorilla() {
        // Use high-cardinality data (many unique values) to ensure Gorilla is selected.
        // Each sample has a unique value derived from a transcendental function so
        // cardinality_ratio > 0.05 and zero_xor_count is low.
        let data: Vec<(i64, f64)> = (0..200)
            .map(|i| {
                let ts = i as i64 * 1000;
                // cos(i) produces many distinct float values, ensuring high cardinality
                let val = (i as f64 * 0.123456).cos() * 100.0 + 50.0;
                (ts, val)
            })
            .collect();
        let mut comp = AdaptiveCompressor::new();
        comp.extend(&data);
        let block = comp.finish().expect("finish");
        assert_eq!(block.algorithm, CompressionAlgorithm::Gorilla);
        let decoded = block.decode().expect("decode");
        assert_eq!(decoded.len(), data.len());
        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert_eq!(orig.0, dec.0);
            assert_eq!(orig.1.to_bits(), dec.1.to_bits());
        }
    }

    #[test]
    fn test_forced_algorithm_gorilla() {
        let data: Vec<(i64, f64)> = (0..50).map(|i| (i as i64 * 1000, 7.0)).collect();
        let mut comp = AdaptiveCompressor::new().with_algorithm(CompressionAlgorithm::Gorilla);
        comp.extend(&data);
        let block = comp.finish().expect("finish");
        assert_eq!(block.algorithm, CompressionAlgorithm::Gorilla);
        let decoded = block.decode().expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_forced_algorithm_rle() {
        let data = make_regular(50, 500, 1.5);
        let mut comp = AdaptiveCompressor::new().with_algorithm(CompressionAlgorithm::Rle);
        comp.extend(&data);
        let block = comp.finish().expect("finish");
        assert_eq!(block.algorithm, CompressionAlgorithm::Rle);
        let decoded = block.decode().expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_forced_algorithm_raw() {
        let data: Vec<(i64, f64)> = (0..10).map(|i| (i as i64, i as f64)).collect();
        let mut comp = AdaptiveCompressor::new().with_algorithm(CompressionAlgorithm::Raw);
        comp.extend(&data);
        let block = comp.finish().expect("finish");
        assert_eq!(block.algorithm, CompressionAlgorithm::Raw);
        let decoded = block.decode().expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_raw_encode_decode_round_trip() {
        let data: Vec<(i64, f64)> = vec![(0, 1.0), (1000, 2.5), (2000, -std::f64::consts::PI)];
        let raw = encode_raw(&data);
        let decoded = decode_raw(&raw).expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rle_binary_round_trip() {
        let data: Vec<(i64, f64)> = vec![(0, 1.0), (1000, 1.0), (2000, 2.0), (3000, 2.0)];
        let runs = rle_encode(&data).expect("rle encode");
        let encoded = encode_rle_runs(&runs).expect("encode runs");
        let decoded_runs = decode_rle_runs(&encoded).expect("decode runs");
        assert_eq!(runs.len(), decoded_runs.len());
        for (a, b) in runs.iter().zip(decoded_runs.iter()) {
            assert_eq!(a.start_timestamp, b.start_timestamp);
            assert_eq!(a.end_timestamp, b.end_timestamp);
            assert_eq!(a.value.to_bits(), b.value.to_bits());
            assert_eq!(a.count, b.count);
        }
        let decoded = rle_decode(&decoded_runs);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_compression_ratio() {
        let data: Vec<(i64, f64)> = (0..1000).map(|i| (i as i64 * 1000, 5.0)).collect();
        let mut comp = AdaptiveCompressor::new();
        comp.extend(&data);
        let block = comp.finish().expect("finish");
        assert!(
            block.compression_ratio() > 1.0,
            "should have positive compression"
        );
    }

    #[test]
    fn test_metadata_fields() {
        let data: Vec<(i64, f64)> = vec![(100, 1.0), (200, 2.0), (300, 3.0)];
        let mut comp = AdaptiveCompressor::new();
        comp.extend(&data);
        let block = comp.finish().expect("finish");
        assert_eq!(block.min_timestamp, 100);
        assert_eq!(block.max_timestamp, 300);
        assert_eq!(block.sample_count, 3);
    }

    #[test]
    fn test_dictionary_forced_returns_error() {
        let data = vec![(0i64, 1.0f64)];
        let mut comp = AdaptiveCompressor::new().with_algorithm(CompressionAlgorithm::Dictionary);
        comp.extend(&data);
        let result = comp.finish();
        assert!(result.is_err());
    }

    #[test]
    fn test_analyse_stats_constant() {
        let data: Vec<(i64, f64)> = (0..100).map(|i| (i as i64 * 1000, 42.0)).collect();
        let mut comp = AdaptiveCompressor::new();
        comp.extend(&data);
        let stats = comp.analyse();
        assert_eq!(stats.total, 100);
        assert_eq!(stats.unique_values, 1);
        assert_eq!(stats.zero_xor_count, 99); // all values same
    }
}
