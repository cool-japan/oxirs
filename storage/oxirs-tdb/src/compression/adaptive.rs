//! Adaptive compression implementation that selects the best algorithm

use crate::compression::{
    bitmap::{BitmapRoaringEncoder, BitmapWAHEncoder},
    column_store::ColumnStoreCompressor,
    delta::DeltaEncoder,
    dictionary::AdaptiveDictionary,
    frame_of_reference::FrameOfReferenceEncoder,
    run_length::RunLengthEncoder,
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
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

        // Decision tree based on data characteristics
        if stats.sparsity < 0.1 {
            // Very sparse data - use bitmap compression
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
                let encoder = FrameOfReferenceEncoder;
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
                let encoder = FrameOfReferenceEncoder;
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
            AdvancedCompressionType::Adaptive => {
                return Err(anyhow!("Recursive adaptive compression not allowed"));
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
