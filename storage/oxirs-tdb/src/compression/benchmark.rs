//! # Compression Benchmarking
//!
//! Measures throughput and compression ratio for different compression algorithms
//! on TSDB-style data (typically sorted integer sequences).
//!
//! ## Usage
//!
//! ```rust
//! use oxirs_tdb::compression::benchmark::{CompressionAlgo, CompressionBenchmark};
//!
//! let data = b"hello world hello world hello world";
//! let result = CompressionBenchmark::benchmark_algorithm(CompressionAlgo::Lz4, data);
//! println!("Ratio: {:.2}, Speed: {:.1} MB/s", result.ratio, result.compress_mb_s);
//! ```

use std::time::{Duration, Instant};

use crate::compression::{
    delta::DeltaEncoder, lz4::Lz4Compressor, run_length::RunLengthEncoder,
    snappy::SnappyCompressor, zstd_compression::ZstdCompressor, CompressionAlgorithm,
};

// ─────────────────────────────────────────────────────────────────────────────
// CompressionAlgo
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies a compression algorithm for benchmarking purposes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompressionAlgo {
    /// Variable-length delta encoding (optimised for sorted integer IDs).
    Delta,
    /// Run-length encoding (optimised for repeated values).
    RunLength,
    /// LZ4 general-purpose compression (very fast).
    Lz4,
    /// Zstandard compression (excellent ratio).
    Zstd,
    /// Snappy compression (fast with good ratio).
    Snappy,
}

impl std::fmt::Display for CompressionAlgo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            CompressionAlgo::Delta => "Delta",
            CompressionAlgo::RunLength => "RunLength",
            CompressionAlgo::Lz4 => "LZ4",
            CompressionAlgo::Zstd => "Zstd",
            CompressionAlgo::Snappy => "Snappy",
        };
        write!(f, "{}", name)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BenchmarkResult
// ─────────────────────────────────────────────────────────────────────────────

/// Benchmark measurements for a single compression algorithm on a data sample.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// The algorithm that was benchmarked.
    pub algo: CompressionAlgo,
    /// Compression ratio: `compressed_size / original_size` (lower is better).
    /// A ratio > 1.0 means the data expanded (compression was counter-productive).
    pub ratio: f64,
    /// Compression throughput in MB/s.
    pub compress_mb_s: f64,
    /// Decompression throughput in MB/s.
    pub decompress_mb_s: f64,
    /// Original data size in bytes.
    pub original_size: usize,
    /// Compressed data size in bytes.
    pub compressed_size: usize,
}

impl BenchmarkResult {
    /// Return `true` if compression actually shrinks the data.
    pub fn is_beneficial(&self) -> bool {
        self.ratio < 1.0
    }

    /// Space savings as a percentage (0–100).  Negative means data expanded.
    pub fn space_savings_pct(&self) -> f64 {
        (1.0 - self.ratio) * 100.0
    }
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: ratio={:.3} compress={:.1}MB/s decompress={:.1}MB/s",
            self.algo, self.ratio, self.compress_mb_s, self.decompress_mb_s
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum nanoseconds to report (avoids division by zero on very fast ops).
const MIN_NS: u128 = 1;

/// Compute MB/s given `data_bytes` and `elapsed` time.
fn mb_per_second(data_bytes: usize, elapsed: Duration) -> f64 {
    let ns = elapsed.as_nanos().max(MIN_NS);
    (data_bytes as f64 / 1_048_576.0) / (ns as f64 / 1_000_000_000.0)
}

/// Produce a [`BenchmarkResult`] where compression was not available.
///
/// Uses original data as compressed data, giving ratio = 1.0 and max speed.
fn unavailable_result(algo: CompressionAlgo, data: &[u8]) -> BenchmarkResult {
    BenchmarkResult {
        algo,
        ratio: 1.0,
        compress_mb_s: 0.0,
        decompress_mb_s: 0.0,
        original_size: data.len(),
        compressed_size: data.len(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompressionBenchmark
// ─────────────────────────────────────────────────────────────────────────────

/// Measures compression throughput and ratio for different algorithms.
///
/// All benchmarks are single-shot (not averaged); for statistical precision
/// use the `benchmarking` feature with multiple iterations.
pub struct CompressionBenchmark;

impl CompressionBenchmark {
    /// Create a new `CompressionBenchmark` instance (stateless).
    pub fn new() -> Self {
        Self
    }

    /// Benchmark a single algorithm on the given data.
    ///
    /// Returns a [`BenchmarkResult`] regardless of whether compression succeeds:
    /// on error, ratio is set to `1.0` and throughputs to `0.0`.
    pub fn benchmark_algorithm(algo: CompressionAlgo, data: &[u8]) -> BenchmarkResult {
        if data.is_empty() {
            return BenchmarkResult {
                algo,
                ratio: 1.0,
                compress_mb_s: 0.0,
                decompress_mb_s: 0.0,
                original_size: 0,
                compressed_size: 0,
            };
        }

        match &algo {
            CompressionAlgo::Delta => Self::bench_delta(data),
            CompressionAlgo::RunLength => Self::bench_run_length(data),
            CompressionAlgo::Lz4 => Self::bench_lz4(data),
            CompressionAlgo::Zstd => Self::bench_zstd(data),
            CompressionAlgo::Snappy => Self::bench_snappy(data),
        }
    }

    /// Benchmark all supported algorithms on the given data.
    ///
    /// Results are returned in algorithm order (Delta, RunLength, LZ4, Zstd, Snappy).
    pub fn benchmark_all(data: &[u8]) -> Vec<BenchmarkResult> {
        vec![
            Self::benchmark_algorithm(CompressionAlgo::Delta, data),
            Self::benchmark_algorithm(CompressionAlgo::RunLength, data),
            Self::benchmark_algorithm(CompressionAlgo::Lz4, data),
            Self::benchmark_algorithm(CompressionAlgo::Zstd, data),
            Self::benchmark_algorithm(CompressionAlgo::Snappy, data),
        ]
    }

    /// Return the result with the best compression ratio (lowest ratio value).
    pub fn best_ratio(results: &[BenchmarkResult]) -> Option<&BenchmarkResult> {
        results.iter().min_by(|a, b| {
            a.ratio
                .partial_cmp(&b.ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return the result with the highest compression throughput (MB/s).
    pub fn best_compress_speed(results: &[BenchmarkResult]) -> Option<&BenchmarkResult> {
        results.iter().max_by(|a, b| {
            a.compress_mb_s
                .partial_cmp(&b.compress_mb_s)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return the result with the highest decompression throughput (MB/s).
    pub fn best_decompress_speed(results: &[BenchmarkResult]) -> Option<&BenchmarkResult> {
        results.iter().max_by(|a, b| {
            a.decompress_mb_s
                .partial_cmp(&b.decompress_mb_s)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    // ── Private per-algorithm benchmarks ─────────────────────────────────────

    fn bench_delta(data: &[u8]) -> BenchmarkResult {
        // Treat raw bytes as u64 sequence (chunked by 8 bytes).
        let values: Vec<u64> = data
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();

        if values.is_empty() {
            // Data too small for u64 chunking; fall back to byte-level encoding
            let t0 = Instant::now();
            let enc = DeltaEncoder::encode_byte_sequence(data);
            let compress_elapsed = t0.elapsed();

            let compressed = match enc {
                Ok(c) => c,
                Err(_) => return unavailable_result(CompressionAlgo::Delta, data),
            };

            let t1 = Instant::now();
            let _ = DeltaEncoder::decode_byte_sequence(&compressed);
            let decompress_elapsed = t1.elapsed();

            let ratio = compressed.len() as f64 / data.len() as f64;
            return BenchmarkResult {
                algo: CompressionAlgo::Delta,
                ratio,
                compress_mb_s: mb_per_second(data.len(), compress_elapsed),
                decompress_mb_s: mb_per_second(data.len(), decompress_elapsed),
                original_size: data.len(),
                compressed_size: compressed.len(),
            };
        }

        let t0 = Instant::now();
        let encoded = DeltaEncoder::encode(&values);
        let compress_elapsed = t0.elapsed();

        let t1 = Instant::now();
        let _ = DeltaEncoder::decode(&encoded);
        let decompress_elapsed = t1.elapsed();

        let ratio = encoded.len() as f64 / data.len() as f64;
        BenchmarkResult {
            algo: CompressionAlgo::Delta,
            ratio,
            compress_mb_s: mb_per_second(data.len(), compress_elapsed),
            decompress_mb_s: mb_per_second(data.len(), decompress_elapsed),
            original_size: data.len(),
            compressed_size: encoded.len(),
        }
    }

    fn bench_run_length(data: &[u8]) -> BenchmarkResult {
        let t0 = Instant::now();
        let enc = RunLengthEncoder::encode(data);
        let compress_elapsed = t0.elapsed();

        let compressed = match enc {
            Ok(c) => c,
            Err(_) => return unavailable_result(CompressionAlgo::RunLength, data),
        };

        let t1 = Instant::now();
        let _ = RunLengthEncoder::decode(&compressed);
        let decompress_elapsed = t1.elapsed();

        let ratio = if data.is_empty() {
            1.0
        } else {
            compressed.len() as f64 / data.len() as f64
        };

        BenchmarkResult {
            algo: CompressionAlgo::RunLength,
            ratio,
            compress_mb_s: mb_per_second(data.len(), compress_elapsed),
            decompress_mb_s: mb_per_second(data.len(), decompress_elapsed),
            original_size: data.len(),
            compressed_size: compressed.len(),
        }
    }

    fn bench_lz4(data: &[u8]) -> BenchmarkResult {
        let compressor = Lz4Compressor::new();

        let t0 = Instant::now();
        let compressed_data = compressor.compress(data);
        let compress_elapsed = t0.elapsed();

        let compressed_data = match compressed_data {
            Ok(c) => c,
            Err(_) => return unavailable_result(CompressionAlgo::Lz4, data),
        };

        let t1 = Instant::now();
        let _ = compressor.decompress(&compressed_data);
        let decompress_elapsed = t1.elapsed();

        let ratio = compressed_data.data.len() as f64 / data.len() as f64;
        BenchmarkResult {
            algo: CompressionAlgo::Lz4,
            ratio,
            compress_mb_s: mb_per_second(data.len(), compress_elapsed),
            decompress_mb_s: mb_per_second(data.len(), decompress_elapsed),
            original_size: data.len(),
            compressed_size: compressed_data.data.len(),
        }
    }

    fn bench_zstd(data: &[u8]) -> BenchmarkResult {
        let compressor = ZstdCompressor::new();

        let t0 = Instant::now();
        let compressed_data = compressor.compress(data);
        let compress_elapsed = t0.elapsed();

        let compressed_data = match compressed_data {
            Ok(c) => c,
            Err(_) => return unavailable_result(CompressionAlgo::Zstd, data),
        };

        let t1 = Instant::now();
        let _ = compressor.decompress(&compressed_data);
        let decompress_elapsed = t1.elapsed();

        let ratio = compressed_data.data.len() as f64 / data.len() as f64;
        BenchmarkResult {
            algo: CompressionAlgo::Zstd,
            ratio,
            compress_mb_s: mb_per_second(data.len(), compress_elapsed),
            decompress_mb_s: mb_per_second(data.len(), decompress_elapsed),
            original_size: data.len(),
            compressed_size: compressed_data.data.len(),
        }
    }

    fn bench_snappy(data: &[u8]) -> BenchmarkResult {
        let compressor = SnappyCompressor::new();

        let t0 = Instant::now();
        let compressed_data = compressor.compress(data);
        let compress_elapsed = t0.elapsed();

        let compressed_data = match compressed_data {
            Ok(c) => c,
            Err(_) => return unavailable_result(CompressionAlgo::Snappy, data),
        };

        let t1 = Instant::now();
        let _ = compressor.decompress(&compressed_data);
        let decompress_elapsed = t1.elapsed();

        let ratio = compressed_data.data.len() as f64 / data.len() as f64;
        BenchmarkResult {
            algo: CompressionAlgo::Snappy,
            ratio,
            compress_mb_s: mb_per_second(data.len(), compress_elapsed),
            decompress_mb_s: mb_per_second(data.len(), decompress_elapsed),
            original_size: data.len(),
            compressed_size: compressed_data.data.len(),
        }
    }
}

impl Default for CompressionBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a repetitive u64 sequence and flatten to bytes for benchmarking.
    fn repetitive_u64_data(count: usize) -> Vec<u8> {
        let values: Vec<u64> = (0..count as u64).map(|i| i / 10).collect();
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Build a sorted ascending u64 sequence as bytes.
    fn sorted_u64_data(count: usize) -> Vec<u8> {
        let values: Vec<u64> = (0..count as u64).collect();
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // ── Individual algorithm benchmarks ──────────────────────────────────────

    #[test]
    fn test_benchmark_delta_valid_result() {
        let data = sorted_u64_data(100);
        let result = CompressionBenchmark::benchmark_algorithm(CompressionAlgo::Delta, &data);
        assert_eq!(result.algo, CompressionAlgo::Delta);
        assert!(result.original_size > 0);
        assert!(result.ratio >= 0.0);
        assert!(result.compress_mb_s >= 0.0);
        assert!(result.decompress_mb_s >= 0.0);
    }

    #[test]
    fn test_benchmark_run_length_valid_result() {
        let data = repetitive_u64_data(100);
        let result = CompressionBenchmark::benchmark_algorithm(CompressionAlgo::RunLength, &data);
        assert_eq!(result.algo, CompressionAlgo::RunLength);
        assert!(result.original_size > 0);
        assert!(result.ratio >= 0.0);
    }

    #[test]
    fn test_benchmark_lz4_valid_result() {
        let data = b"hello world hello world hello world hello world".to_vec();
        let result = CompressionBenchmark::benchmark_algorithm(CompressionAlgo::Lz4, &data);
        assert_eq!(result.algo, CompressionAlgo::Lz4);
        assert!(result.original_size == data.len());
        assert!(result.ratio >= 0.0);
    }

    #[test]
    fn test_benchmark_zstd_valid_result() {
        let data = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec();
        let result = CompressionBenchmark::benchmark_algorithm(CompressionAlgo::Zstd, &data);
        assert_eq!(result.algo, CompressionAlgo::Zstd);
        assert!(result.ratio >= 0.0);
    }

    #[test]
    fn test_benchmark_snappy_valid_result() {
        let data = b"snappy snappy snappy snappy snappy".to_vec();
        let result = CompressionBenchmark::benchmark_algorithm(CompressionAlgo::Snappy, &data);
        assert_eq!(result.algo, CompressionAlgo::Snappy);
        assert!(result.ratio >= 0.0);
    }

    // ── benchmark_all ─────────────────────────────────────────────────────────

    #[test]
    fn test_benchmark_all_returns_five_results() {
        let data = b"test data for benchmarking all algorithms".to_vec();
        let results = CompressionBenchmark::benchmark_all(&data);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_benchmark_all_all_algos_present() {
        let data = sorted_u64_data(50);
        let results = CompressionBenchmark::benchmark_all(&data);
        let algos: Vec<&CompressionAlgo> = results.iter().map(|r| &r.algo).collect();
        assert!(algos.contains(&&CompressionAlgo::Delta));
        assert!(algos.contains(&&CompressionAlgo::RunLength));
        assert!(algos.contains(&&CompressionAlgo::Lz4));
        assert!(algos.contains(&&CompressionAlgo::Zstd));
        assert!(algos.contains(&&CompressionAlgo::Snappy));
    }

    // ── best_ratio ────────────────────────────────────────────────────────────

    #[test]
    fn test_best_ratio_returns_minimum() {
        let data = repetitive_u64_data(100);
        let results = CompressionBenchmark::benchmark_all(&data);
        let best = CompressionBenchmark::best_ratio(&results);
        assert!(best.is_some());
        let best = best.unwrap();
        // Verify it really is the minimum
        for r in &results {
            assert!(best.ratio <= r.ratio + f64::EPSILON);
        }
    }

    #[test]
    fn test_best_ratio_empty_returns_none() {
        let result = CompressionBenchmark::best_ratio(&[]);
        assert!(result.is_none());
    }

    // ── best_compress_speed ───────────────────────────────────────────────────

    #[test]
    fn test_best_compress_speed_returns_maximum() {
        let data = sorted_u64_data(100);
        let results = CompressionBenchmark::benchmark_all(&data);
        let best = CompressionBenchmark::best_compress_speed(&results);
        assert!(best.is_some());
        let best = best.unwrap();
        for r in &results {
            assert!(best.compress_mb_s >= r.compress_mb_s - f64::EPSILON);
        }
    }

    #[test]
    fn test_best_compress_speed_empty_returns_none() {
        let result = CompressionBenchmark::best_compress_speed(&[]);
        assert!(result.is_none());
    }

    // ── best_decompress_speed ─────────────────────────────────────────────────

    #[test]
    fn test_best_decompress_speed_returns_maximum() {
        let data = sorted_u64_data(100);
        let results = CompressionBenchmark::benchmark_all(&data);
        let best = CompressionBenchmark::best_decompress_speed(&results);
        assert!(best.is_some());
    }

    // ── BenchmarkResult helpers ───────────────────────────────────────────────

    #[test]
    fn test_benchmark_result_ratio_beneficial() {
        let r = BenchmarkResult {
            algo: CompressionAlgo::Lz4,
            ratio: 0.5,
            compress_mb_s: 100.0,
            decompress_mb_s: 200.0,
            original_size: 1000,
            compressed_size: 500,
        };
        assert!(r.is_beneficial());
        assert!((r.space_savings_pct() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_benchmark_result_ratio_not_beneficial() {
        let r = BenchmarkResult {
            algo: CompressionAlgo::RunLength,
            ratio: 1.5,
            compress_mb_s: 50.0,
            decompress_mb_s: 80.0,
            original_size: 100,
            compressed_size: 150,
        };
        assert!(!r.is_beneficial());
        assert!(r.space_savings_pct() < 0.0);
    }

    #[test]
    fn test_benchmark_empty_data() {
        let result = CompressionBenchmark::benchmark_algorithm(CompressionAlgo::Lz4, &[]);
        assert_eq!(result.original_size, 0);
        assert_eq!(result.compressed_size, 0);
    }

    #[test]
    fn test_benchmark_result_display() {
        let r = BenchmarkResult {
            algo: CompressionAlgo::Zstd,
            ratio: 0.3,
            compress_mb_s: 120.0,
            decompress_mb_s: 400.0,
            original_size: 1000,
            compressed_size: 300,
        };
        let s = r.to_string();
        assert!(s.contains("Zstd"));
        assert!(s.contains("0.300"));
    }

    #[test]
    fn test_compression_algo_display() {
        assert_eq!(CompressionAlgo::Delta.to_string(), "Delta");
        assert_eq!(CompressionAlgo::RunLength.to_string(), "RunLength");
        assert_eq!(CompressionAlgo::Lz4.to_string(), "LZ4");
        assert_eq!(CompressionAlgo::Zstd.to_string(), "Zstd");
        assert_eq!(CompressionAlgo::Snappy.to_string(), "Snappy");
    }

    #[test]
    fn test_delta_compresses_sorted_ids_efficiently() {
        // Sorted triple IDs with small deltas should compress well
        let data = sorted_u64_data(200);
        let result = CompressionBenchmark::benchmark_algorithm(CompressionAlgo::Delta, &data);
        // Expect better than 50% ratio for sequential IDs (delta=1 → 1 byte via LEB128)
        assert!(
            result.ratio < 0.5,
            "Expected ratio < 0.5, got {}",
            result.ratio
        );
    }

    #[test]
    fn test_rle_compresses_repetitive_ids_efficiently() {
        // Highly repetitive data should compress well with RLE
        let data = vec![42u8; 1000];
        let result = CompressionBenchmark::benchmark_algorithm(CompressionAlgo::RunLength, &data);
        assert!(
            result.ratio < 0.1,
            "Expected ratio < 0.1 for repetitive data, got {}",
            result.ratio
        );
    }
}
