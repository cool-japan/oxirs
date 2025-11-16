//! Snappy compression algorithm implementation
//!
//! Snappy is a compression algorithm developed by Google that prioritizes
//! speed over compression ratio. It's designed for very fast compression
//! and decompression with reasonable compression ratios.
//!
//! # Performance Characteristics
//!
//! - Compression speed: ~250-500 MB/s
//! - Decompression speed: ~500-1500 MB/s
//! - Compression ratio: 1.5-2x on typical data
//!
//! # Use Cases
//!
//! - Real-time data processing
//! - Network protocols requiring low latency
//! - Cache compression where speed is paramount
//! - Google BigTable-compatible storage

use super::{
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
    MAX_COMPRESSION_INPUT_SIZE,
};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::time::Instant;

/// Snappy compression algorithm
///
/// Uses the snap crate for pure Rust Snappy compression.
/// Optimized for speed with reasonable compression ratios.
#[derive(Debug, Clone)]
pub struct SnappyCompressor {
    /// Use framing format (compatible with streaming)
    use_framing: bool,
}

impl SnappyCompressor {
    /// Create a new Snappy compressor with default settings
    pub fn new() -> Self {
        Self {
            use_framing: false, // Raw format by default
        }
    }

    /// Create a Snappy compressor with framing format
    ///
    /// The framing format is compatible with the Snappy streaming format
    /// and includes checksums for data integrity.
    pub fn with_framing() -> Self {
        Self { use_framing: true }
    }

    /// Create a Snappy compressor with raw format (no framing)
    pub fn raw() -> Self {
        Self { use_framing: false }
    }
}

impl Default for SnappyCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionAlgorithm for SnappyCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        if data.is_empty() {
            return Ok(CompressedData {
                data: Vec::new(),
                metadata: CompressionMetadata {
                    algorithm: AdvancedCompressionType::Adaptive, // Will add Snappy type
                    original_size: 0,
                    compressed_size: 0,
                    compression_time_us: 0,
                    metadata: HashMap::new(),
                },
            });
        }

        if data.len() > MAX_COMPRESSION_INPUT_SIZE {
            bail!(
                "Input data too large: {} bytes (max: {})",
                data.len(),
                MAX_COMPRESSION_INPUT_SIZE
            );
        }

        let start = Instant::now();

        // Compress using snap
        let compressed = if self.use_framing {
            // Use framing format with checksums
            let mut encoder = snap::write::FrameEncoder::new(Vec::new());
            use std::io::Write;
            encoder.write_all(data)?;
            encoder.into_inner()?
        } else {
            // Use raw format (faster, no checksums)
            snap::raw::Encoder::new().compress_vec(data)?
        };

        let compression_time = start.elapsed();

        let mut metadata_map = HashMap::new();
        metadata_map.insert("framing".to_string(), self.use_framing.to_string());
        metadata_map.insert(
            "compression_ratio".to_string(),
            format!("{:.2}", compressed.len() as f64 / data.len() as f64),
        );

        let compressed_size = compressed.len() as u64;

        Ok(CompressedData {
            data: compressed,
            metadata: CompressionMetadata {
                algorithm: AdvancedCompressionType::Adaptive, // Will add Snappy type
                original_size: data.len() as u64,
                compressed_size,
                compression_time_us: compression_time.as_micros() as u64,
                metadata: metadata_map,
            },
        })
    }

    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        if compressed.data.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();

        // Determine if data was compressed with framing
        let use_framing = compressed
            .metadata
            .metadata
            .get("framing")
            .and_then(|v| v.parse().ok())
            .unwrap_or(self.use_framing);

        // Decompress using snap
        let decompressed = if use_framing {
            // Use framing format with checksums
            use std::io::Read;
            let mut decoder = snap::read::FrameDecoder::new(&compressed.data[..]);
            let mut result = Vec::new();
            decoder.read_to_end(&mut result)?;
            result
        } else {
            // Use raw format
            snap::raw::Decoder::new().decompress_vec(&compressed.data)?
        };

        let decompression_time = start.elapsed();

        // Verify decompressed size matches metadata
        if decompressed.len() != compressed.metadata.original_size as usize {
            bail!(
                "Decompressed size mismatch: expected {}, got {}",
                compressed.metadata.original_size,
                decompressed.len()
            );
        }

        log::trace!(
            "Snappy decompression: {} bytes -> {} bytes in {:?}",
            compressed.data.len(),
            decompressed.len(),
            decompression_time
        );

        Ok(decompressed)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::Adaptive // Will add Snappy type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snappy_compress_empty() {
        let compressor = SnappyCompressor::new();
        let result = compressor.compress(&[]).unwrap();
        assert_eq!(result.data.len(), 0);
        assert_eq!(result.metadata.original_size, 0);
        assert_eq!(result.metadata.compressed_size, 0);
    }

    #[test]
    fn test_snappy_compress_decompress_raw() {
        let compressor = SnappyCompressor::raw();
        let original = b"Hello, World! This is a test of Snappy compression. ".repeat(100);

        let compressed = compressor.compress(&original).unwrap();
        assert!(compressed.data.len() < original.len());
        assert_eq!(compressed.metadata.original_size, original.len() as u64);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_snappy_compress_decompress_framed() {
        let compressor = SnappyCompressor::with_framing();
        let original = b"Hello, World! This is a test of Snappy framed compression. ".repeat(100);

        let compressed = compressor.compress(&original).unwrap();
        assert!(compressed.data.len() < original.len());
        assert_eq!(compressed.metadata.original_size, original.len() as u64);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_snappy_format_comparison() {
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(50);

        let raw = SnappyCompressor::raw();
        let framed = SnappyCompressor::with_framing();

        let raw_result = raw.compress(&data).unwrap();
        let framed_result = framed.compress(&data).unwrap();

        // Both should decompress to the same data
        assert_eq!(raw.decompress(&raw_result).unwrap(), data);
        assert_eq!(framed.decompress(&framed_result).unwrap(), data);

        // Raw format should be slightly smaller (no framing overhead)
        println!("Raw: {} bytes", raw_result.data.len());
        println!("Framed: {} bytes", framed_result.data.len());
        assert!(raw_result.data.len() <= framed_result.data.len());
    }

    #[test]
    fn test_snappy_highly_compressible_data() {
        let compressor = SnappyCompressor::new();
        let data = vec![b'A'; 10000]; // Highly repetitive data

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        // Should achieve good compression on repetitive data
        assert!(compressed.data.len() < data.len() / 5);
        println!(
            "Snappy highly compressible: {} -> {} bytes ({:.2}x)",
            data.len(),
            compressed.data.len(),
            data.len() as f64 / compressed.data.len() as f64
        );
    }

    #[test]
    fn test_snappy_incompressible_data() {
        let compressor = SnappyCompressor::new();
        // Use scirs2_core::random for random data generation
        use scirs2_core::random::{rng, Random, Rng};

        let mut rng = rng();
        let data: Vec<u8> = (0..1000).map(|_| rng.random_range(0..256) as u8).collect();

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        // Random data should not compress well
        println!(
            "Snappy incompressible: {} -> {} bytes",
            data.len(),
            compressed.data.len()
        );
    }

    #[test]
    fn test_snappy_too_large_input() {
        let compressor = SnappyCompressor::new();
        let data = vec![0u8; MAX_COMPRESSION_INPUT_SIZE + 1];

        let result = compressor.compress(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_snappy_metadata() {
        let compressor = SnappyCompressor::with_framing();
        let data = b"Testing metadata collection with Snappy".repeat(10);

        let compressed = compressor.compress(&data).unwrap();

        assert_eq!(compressed.metadata.metadata.get("framing").unwrap(), "true");
        assert!(compressed
            .metadata
            .metadata
            .contains_key("compression_ratio"));
        assert!(compressed.metadata.compression_time_us > 0);
    }

    #[test]
    fn test_snappy_small_data() {
        let compressor = SnappyCompressor::new();
        let data = b"Small";

        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        // Very small data might not compress
        println!(
            "Snappy small data: {} -> {} bytes",
            data.len(),
            compressed.data.len()
        );
    }

    #[test]
    fn test_snappy_rdf_data_compression() {
        let compressor = SnappyCompressor::new();
        // Simulate RDF triple data with typical patterns
        let rdf_data = b"<http://example.org/resource/1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Class> .\n"
            .repeat(100);

        let compressed = compressor.compress(&rdf_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, rdf_data);

        // RDF data with repetitive URIs should compress well
        let ratio = rdf_data.len() as f64 / compressed.data.len() as f64;
        println!("Snappy RDF compression ratio: {:.2}x", ratio);
        assert!(ratio > 1.5); // Should achieve >1.5x compression
    }

    #[test]
    fn test_snappy_roundtrip_consistency() {
        let compressor = SnappyCompressor::new();
        let data = b"Consistency test data".repeat(50);

        // Compress and decompress multiple times
        let mut current = data.to_vec();
        for _ in 0..5 {
            let compressed = compressor.compress(&current).unwrap();
            current = compressor.decompress(&compressed).unwrap();
        }

        assert_eq!(current, data);
    }
}
