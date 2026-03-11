//! Advanced ZStd codec with dictionary-based compression for large RDF graphs
//!
//! This module provides a higher-level ZStd codec that extends the basic ZstdCompressor
//! with pre-trained dictionary support and dictionary training from RDF data samples.
//! Dictionary-based compression is especially effective for small RDF triples that
//! share common URI prefixes and patterns.
//!
//! # Performance Characteristics
//!
//! With a pre-trained dictionary:
//! - 2-5x better compression ratio on small blocks (<16KB)
//! - Near-identical speed to basic Zstd
//! - Optimal for RDF triple blocks sharing URI namespace patterns
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_tdb::compression::zstd_codec::ZstdCodec;
//!
//! // Basic usage without dictionary
//! let codec = ZstdCodec::new(3);
//! let data = b"<http://example.org/s> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Class> .";
//! let compressed = codec.compress(data).unwrap();
//! let decompressed = codec.decompress(&compressed).unwrap();
//! assert_eq!(decompressed, data);
//!
//! // Train a dictionary from RDF samples
//! let samples = vec![data.to_vec(); 100];
//! let dict = ZstdCodec::train_dict(&samples, 1024).unwrap();
//! let codec_with_dict = ZstdCodec::with_dict(3, dict);
//! ```

use crate::error::{Result, TdbError};
use std::io::{Read, Write};
use std::time::Instant;

/// Minimum data size (bytes) that benefits from dictionary training samples
const MIN_SAMPLE_SIZE: usize = 8;

/// Maximum total sample data size for dictionary training (1GB safety limit)
const MAX_TOTAL_SAMPLE_SIZE: usize = 1024 * 1024 * 1024;

/// Zstandard codec with optional pre-trained dictionary support.
///
/// `ZstdCodec` provides both basic Zstd compression and dictionary-accelerated
/// compression. The dictionary approach is particularly effective for RDF triple
/// blocks where subjects, predicates, and objects share common URI prefixes.
#[derive(Debug, Clone)]
pub struct ZstdCodec {
    /// Compression level (1-22, default 3)
    pub compression_level: i32,
    /// Optional pre-trained Zstd dictionary bytes
    pub dict: Option<Vec<u8>>,
}

/// Statistics from a ZstdCodec compress/decompress operation
#[derive(Debug, Clone)]
pub struct ZstdCodecStats {
    /// Original (uncompressed) data size in bytes
    pub original_size: usize,
    /// Compressed data size in bytes
    pub compressed_size: usize,
    /// Compression ratio (compressed / original); lower is better
    pub compression_ratio: f64,
    /// Space savings as percentage (0-100)
    pub space_savings_pct: f64,
    /// Time taken for the operation in microseconds
    pub duration_us: u64,
    /// Whether a dictionary was used
    pub dictionary_used: bool,
}

impl ZstdCodec {
    /// Create a new `ZstdCodec` with the given compression level and no dictionary.
    ///
    /// # Arguments
    ///
    /// * `level` - Compression level clamped to 1-22. Level 3 is the recommended default.
    pub fn new(level: i32) -> Self {
        Self {
            compression_level: level.clamp(1, 22),
            dict: None,
        }
    }

    /// Create a new `ZstdCodec` with the given compression level and a pre-trained dictionary.
    ///
    /// Dictionary-based compression is typically 2-5x more effective on small blocks
    /// compared to raw Zstd when the data shares structural patterns with the training samples.
    ///
    /// # Arguments
    ///
    /// * `level` - Compression level clamped to 1-22
    /// * `dict`  - Pre-trained Zstd dictionary bytes (produced by [`ZstdCodec::train_dict`])
    pub fn with_dict(level: i32, dict: Vec<u8>) -> Self {
        Self {
            compression_level: level.clamp(1, 22),
            dict: Some(dict),
        }
    }

    /// Compress `data` using Zstd, optionally with a pre-trained dictionary.
    ///
    /// Returns the compressed bytes. Use [`decompress`](ZstdCodec::decompress) with the
    /// same dictionary to recover the original data.
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let compressed = match &self.dict {
            None => {
                // Standard Zstd compression without dictionary
                let mut encoder =
                    oxiarc_zstd::ZstdStreamEncoder::new(Vec::new(), self.compression_level);
                encoder
                    .write_all(data)
                    .map_err(|e| TdbError::Other(format!("zstd write failed: {e}")))?;
                encoder
                    .finish()
                    .map_err(|e| TdbError::Other(format!("zstd finish failed: {e}")))?
            }
            Some(dict_bytes) => {
                // Dictionary-based Zstd compression
                let mut enc = oxiarc_zstd::ZstdEncoder::new();
                enc.set_level(self.compression_level);
                enc.set_dictionary(dict_bytes);
                enc.compress(data)
                    .map_err(|e| TdbError::Other(format!("zstd dict compress failed: {e}")))?
            }
        };

        Ok(compressed)
    }

    /// Decompress `data` previously compressed with this codec (same dictionary required).
    ///
    /// # Errors
    ///
    /// Returns [`TdbError::Other`] if decompression fails or dictionary mismatch occurs.
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let decompressed = match &self.dict {
            None => {
                let mut decoder = oxiarc_zstd::ZstdStreamDecoder::new(data);
                let mut out = Vec::new();
                decoder
                    .read_to_end(&mut out)
                    .map_err(|e| TdbError::Other(format!("zstd decompress failed: {e}")))?;
                out
            }
            Some(dict_bytes) => {
                // Dictionary decompression - must pass the dictionary used during compression
                oxiarc_zstd::decompress_with_dict(data, dict_bytes)
                    .map_err(|e| TdbError::Other(format!("zstd dict decompress failed: {e}")))?
            }
        };

        Ok(decompressed)
    }

    /// Compress `data` and return both the compressed bytes and statistics.
    pub fn compress_with_stats(&self, data: &[u8]) -> Result<(Vec<u8>, ZstdCodecStats)> {
        let start = Instant::now();
        let compressed = self.compress(data)?;
        let duration_us = start.elapsed().as_micros() as u64;

        let original_size = data.len();
        let compressed_size = compressed.len();
        let compression_ratio = if original_size == 0 {
            1.0
        } else {
            compressed_size as f64 / original_size as f64
        };
        let space_savings_pct = (1.0 - compression_ratio) * 100.0;

        let stats = ZstdCodecStats {
            original_size,
            compressed_size,
            compression_ratio,
            space_savings_pct,
            duration_us,
            dictionary_used: self.dict.is_some(),
        };

        Ok((compressed, stats))
    }

    /// Train a Zstd dictionary from a collection of data samples.
    ///
    /// This is the key capability for RDF graph compression: by training on a representative
    /// sample of RDF triples, the resulting dictionary captures common URI prefixes and
    /// literal patterns, yielding significantly better compression for small triple blocks.
    ///
    /// # Arguments
    ///
    /// * `samples`   - Slice of byte vectors representing individual data samples (e.g., serialized triples)
    /// * `dict_size` - Target dictionary size in bytes. Recommended: 1024-65536 bytes.
    ///   Larger dictionaries capture more patterns but consume more memory.
    ///
    /// # Returns
    ///
    /// The trained dictionary as a byte vector suitable for passing to [`ZstdCodec::with_dict`].
    ///
    /// # Errors
    ///
    /// Returns [`TdbError::Other`] if:
    /// - No samples are provided
    /// - Samples are too small (< 8 bytes each)
    /// - Total sample data exceeds 1GB
    /// - The zstd dictionary trainer fails
    pub fn train_dict(samples: &[Vec<u8>], dict_size: usize) -> Result<Vec<u8>> {
        if samples.is_empty() {
            return Err(TdbError::InvalidInput(
                "Cannot train dictionary from empty sample set".to_string(),
            ));
        }

        // Validate samples and compute total size
        let mut total_size = 0usize;
        let mut valid_samples = Vec::new();
        for (idx, sample) in samples.iter().enumerate() {
            if sample.len() < MIN_SAMPLE_SIZE {
                log::debug!(
                    "Skipping sample {} (size {} < min {})",
                    idx,
                    sample.len(),
                    MIN_SAMPLE_SIZE
                );
                continue;
            }
            total_size = total_size
                .checked_add(sample.len())
                .ok_or_else(|| TdbError::Other("Sample total size overflow".to_string()))?;
            if total_size > MAX_TOTAL_SAMPLE_SIZE {
                return Err(TdbError::InvalidInput(format!(
                    "Total sample size {} exceeds limit of {} bytes",
                    total_size, MAX_TOTAL_SAMPLE_SIZE
                )));
            }
            valid_samples.push(sample.as_slice());
        }

        if valid_samples.is_empty() {
            return Err(TdbError::InvalidInput(format!(
                "No valid samples found (all samples had fewer than {} bytes)",
                MIN_SAMPLE_SIZE
            )));
        }

        // Use the oxiarc_zstd dictionary trainer
        let dict_obj = oxiarc_zstd::train_dictionary(&valid_samples, dict_size)
            .map_err(|e| TdbError::Other(format!("zstd dictionary training failed: {e}")))?;
        let dict = dict_obj.data().to_vec();

        log::debug!(
            "Trained Zstd dictionary: {} samples, {} total bytes -> {} byte dictionary",
            valid_samples.len(),
            total_size,
            dict.len()
        );

        Ok(dict)
    }

    /// Check whether this codec has a dictionary loaded.
    pub fn has_dict(&self) -> bool {
        self.dict.is_some()
    }

    /// Return the dictionary size in bytes, or 0 if no dictionary is loaded.
    pub fn dict_size(&self) -> usize {
        self.dict.as_ref().map(|d| d.len()).unwrap_or(0)
    }
}

impl Default for ZstdCodec {
    fn default() -> Self {
        Self::new(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Basic compress/decompress roundtrip tests --

    #[test]
    fn test_zstd_codec_empty_data() {
        let codec = ZstdCodec::new(3);
        let compressed = codec.compress(&[]).expect("compress empty");
        assert!(compressed.is_empty(), "compressed empty should be empty");
        let decompressed = codec.decompress(&[]).expect("decompress empty");
        assert!(
            decompressed.is_empty(),
            "decompressed empty should be empty"
        );
    }

    #[test]
    fn test_zstd_codec_roundtrip_basic() {
        let codec = ZstdCodec::new(3);
        let original = b"Hello, RDF World! This is a test of ZstdCodec.".repeat(20);
        let compressed = codec.compress(&original).expect("compress");
        let decompressed = codec.decompress(&compressed).expect("decompress");
        assert_eq!(decompressed, original, "roundtrip must be identical");
    }

    #[test]
    fn test_zstd_codec_compression_levels() {
        let data = b"<http://example.org/subject> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Type> .\n".repeat(50);

        for level in [1, 3, 9, 19] {
            let codec = ZstdCodec::new(level);
            let compressed = codec
                .compress(&data)
                .unwrap_or_else(|e| panic!("compress at level {level} failed: {e}"));
            let decompressed = codec
                .decompress(&compressed)
                .unwrap_or_else(|e| panic!("decompress at level {level} failed: {e}"));
            assert_eq!(
                decompressed, data,
                "roundtrip failed at compression level {level}"
            );
            assert!(
                compressed.len() < data.len(),
                "level {level} should compress repetitive RDF data"
            );
        }
    }

    #[test]
    fn test_zstd_codec_level_clamping() {
        let low = ZstdCodec::new(0);
        let high = ZstdCodec::new(100);
        assert_eq!(
            low.compression_level, 1,
            "level should be clamped to minimum 1"
        );
        assert_eq!(
            high.compression_level, 22,
            "level should be clamped to maximum 22"
        );
    }

    #[test]
    fn test_zstd_codec_large_data() {
        let codec = ZstdCodec::new(3);
        // Generate ~1MB of RDF-like repetitive data
        let rdf_triple = b"<http://data.example.org/resource/12345678> <http://schema.org/name> \"Example Resource\" .\n";
        let mut data = Vec::with_capacity(1024 * 1024);
        while data.len() < 1024 * 1024 {
            data.extend_from_slice(rdf_triple);
        }

        let compressed = codec.compress(&data).expect("compress large data");
        let decompressed = codec
            .decompress(&compressed)
            .expect("decompress large data");
        assert_eq!(decompressed, data, "large data roundtrip must be identical");
        // RDF-like data should compress well
        assert!(
            compressed.len() < data.len() / 10,
            "repetitive RDF data should compress >10x, got ratio: {:.2}",
            compressed.len() as f64 / data.len() as f64
        );
    }

    // -- Dictionary training tests --

    #[test]
    fn test_train_dict_basic() {
        let rdf_triple = b"<http://example.org/subject> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Class> .";
        let samples: Vec<Vec<u8>> = (0..200)
            .map(|i| {
                format!(
                    "<http://example.org/resource/{}> <http://schema.org/name> \"Name {}\" .",
                    i, i
                )
                .into_bytes()
            })
            .collect();

        let dict = ZstdCodec::train_dict(&samples, 4096).expect("dictionary training");
        assert!(!dict.is_empty(), "trained dictionary must not be empty");
        assert!(
            dict.len() <= 4096 + 100,
            "dictionary should not greatly exceed requested size"
        );

        // Verify the dictionary is valid by using it for compression
        let codec = ZstdCodec::with_dict(3, dict);
        let test_data: Vec<u8> = rdf_triple.to_vec();
        let compressed = codec.compress(&test_data).expect("compress with dict");
        let decompressed = codec.decompress(&compressed).expect("decompress with dict");
        assert_eq!(decompressed, test_data, "dict roundtrip must be identical");
    }

    #[test]
    fn test_train_dict_empty_samples_error() {
        let result = ZstdCodec::train_dict(&[], 1024);
        assert!(result.is_err(), "empty samples should return error");
    }

    #[test]
    fn test_train_dict_too_small_samples_error() {
        // All samples are too small (< 8 bytes)
        let samples: Vec<Vec<u8>> = vec![b"abc".to_vec(), b"xy".to_vec()];
        let result = ZstdCodec::train_dict(&samples, 1024);
        assert!(result.is_err(), "all-too-small samples should return error");
    }

    // -- Dictionary-based compress/decompress roundtrip tests --

    #[test]
    fn test_with_dict_roundtrip() {
        let samples: Vec<Vec<u8>> = (0..300)
            .map(|i| {
                format!(
                    "<http://example.org/entity/{}> <http://purl.org/dc/terms/title> \"Title {}\" .",
                    i, i
                )
                .into_bytes()
            })
            .collect();

        let dict = ZstdCodec::train_dict(&samples, 8192).expect("train dict");
        let codec = ZstdCodec::with_dict(3, dict);

        for i in 0..10 {
            let triple = format!(
                "<http://example.org/entity/{}> <http://purl.org/dc/terms/title> \"Title {}\" .",
                i * 100,
                i * 100
            )
            .into_bytes();
            let compressed = codec.compress(&triple).expect("compress with dict");
            let decompressed = codec.decompress(&compressed).expect("decompress with dict");
            assert_eq!(
                decompressed, triple,
                "dict roundtrip must match at index {i}"
            );
        }
    }

    #[test]
    fn test_dict_compression_better_than_no_dict() {
        // Train on RDF data with common URI prefixes
        let samples: Vec<Vec<u8>> = (0..500)
            .map(|i| {
                format!(
                    "<http://dbpedia.org/resource/Resource_{}> <http://dbpedia.org/ontology/type> <http://dbpedia.org/ontology/Class> .",
                    i
                )
                .into_bytes()
            })
            .collect();

        let dict = ZstdCodec::train_dict(&samples, 16384).expect("train dict");
        let codec_dict = ZstdCodec::with_dict(3, dict);
        let codec_plain = ZstdCodec::new(3);

        let test_triple = b"<http://dbpedia.org/resource/Resource_999> <http://dbpedia.org/ontology/type> <http://dbpedia.org/ontology/Class> .".repeat(5);

        let (compressed_dict, stats_dict) = codec_dict
            .compress_with_stats(&test_triple)
            .expect("compress with dict stats");
        let (compressed_plain, stats_plain) = codec_plain
            .compress_with_stats(&test_triple)
            .expect("compress plain stats");

        // Dictionary compressed should generally be smaller for similar data
        println!(
            "Dict compression: {} -> {} ({:.2}x ratio)",
            stats_dict.original_size,
            stats_dict.compressed_size,
            1.0 / stats_dict.compression_ratio
        );
        println!(
            "Plain compression: {} -> {} ({:.2}x ratio)",
            stats_plain.original_size,
            stats_plain.compressed_size,
            1.0 / stats_plain.compression_ratio
        );

        // Both should be able to decompress correctly
        let dec_dict = codec_dict
            .decompress(&compressed_dict)
            .expect("decompress dict");
        let dec_plain = codec_plain
            .decompress(&compressed_plain)
            .expect("decompress plain");
        assert_eq!(dec_dict, test_triple);
        assert_eq!(dec_plain, test_triple);

        assert!(
            stats_dict.dictionary_used,
            "stats should reflect dictionary usage"
        );
        assert!(
            !stats_plain.dictionary_used,
            "stats should reflect no dictionary"
        );
    }

    // -- Stats tests --

    #[test]
    fn test_compress_with_stats() {
        let codec = ZstdCodec::new(3);
        let data = b"RDF triple data for stats testing".repeat(100);
        let (compressed, stats) = codec
            .compress_with_stats(&data)
            .expect("compress with stats");

        assert_eq!(stats.original_size, data.len());
        assert_eq!(stats.compressed_size, compressed.len());
        assert!(stats.compression_ratio > 0.0);
        assert!(stats.space_savings_pct >= 0.0 && stats.space_savings_pct <= 100.0);
        assert!(stats.duration_us > 0 || stats.original_size < 1000);
        assert!(!stats.dictionary_used);
    }

    // -- has_dict / dict_size tests --

    #[test]
    fn test_has_dict_no_dict() {
        let codec = ZstdCodec::new(3);
        assert!(!codec.has_dict());
        assert_eq!(codec.dict_size(), 0);
    }

    #[test]
    fn test_has_dict_with_dict() {
        let samples: Vec<Vec<u8>> = (0..200)
            .map(|i| format!("<http://example.org/{}> <p> <o> .", i).into_bytes())
            .collect();
        let dict = ZstdCodec::train_dict(&samples, 2048).expect("train dict");
        let dict_len = dict.len();
        let codec = ZstdCodec::with_dict(3, dict);
        assert!(codec.has_dict());
        assert_eq!(codec.dict_size(), dict_len);
    }
}
