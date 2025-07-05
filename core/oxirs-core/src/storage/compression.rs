//! Advanced compression for RDF data
//!
//! This module provides compression algorithms optimized for RDF data,
//! including custom RDF-specific compression techniques.

use crate::OxirsError;
use lz4::{Decoder, EncoderBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zstd;

/// Compression algorithm
#[derive(Debug, Clone)]
pub enum Algorithm {
    /// No compression
    None,
    /// LZ4 compression (fast)
    Lz4 { level: u32 },
    /// Zstandard compression (high ratio)
    Zstd { level: i32 },
    /// Custom RDF compression
    RdfCustom { options: RdfCompressionOptions },
}

/// RDF-specific compression options
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RdfCompressionOptions {
    /// Use dictionary compression for URIs
    pub uri_dictionary: bool,
    /// Use prefix compression
    pub prefix_compression: bool,
    /// Dictionary size limit
    pub dictionary_size: usize,
    /// Use datatype-specific compression
    pub datatype_compression: bool,
}

impl Default for RdfCompressionOptions {
    fn default() -> Self {
        RdfCompressionOptions {
            uri_dictionary: true,
            prefix_compression: true,
            dictionary_size: 16384,
            datatype_compression: true,
        }
    }
}

/// Compression result
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Compressed data
    pub data: Vec<u8>,
    /// Original size
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Compression ratio
    pub ratio: f64,
    /// Algorithm used
    pub algorithm: String,
    /// Compression time in microseconds
    pub compression_time_us: u64,
}

/// RDF compressor
pub struct Compressor {
    algorithm: Algorithm,
    uri_dictionary: Option<UriDictionary>,
    stats: CompressionStats,
}

/// URI dictionary for compression
struct UriDictionary {
    /// URI to ID mapping
    uri_to_id: HashMap<String, u32>,
    /// ID to URI mapping
    id_to_uri: HashMap<u32, String>,
    /// Next available ID
    next_id: u32,
    /// Common prefixes
    prefixes: Vec<(String, String)>,
}

/// Compression statistics
#[derive(Debug, Default)]
struct CompressionStats {
    total_compressed: u64,
    total_original: u64,
    compression_count: u64,
    total_time_us: u64,
}

impl Compressor {
    /// Create a new compressor
    pub fn new(algorithm: Algorithm) -> Self {
        let uri_dictionary = match &algorithm {
            Algorithm::RdfCustom { options } if options.uri_dictionary => {
                Some(UriDictionary::new())
            }
            _ => None,
        };

        Compressor {
            algorithm,
            uri_dictionary,
            stats: CompressionStats::default(),
        }
    }

    /// Compress data
    pub fn compress(&mut self, data: &[u8]) -> Result<CompressionResult, OxirsError> {
        let start = std::time::Instant::now();
        let original_size = data.len();

        let (compressed, algorithm_name) = match self.algorithm.clone() {
            Algorithm::None => (data.to_vec(), "none"),
            Algorithm::Lz4 { level } => {
                let compressed = self.compress_lz4(data, level)?;
                (compressed, "lz4")
            }
            Algorithm::Zstd { level } => {
                let compressed = self.compress_zstd(data, level)?;
                (compressed, "zstd")
            }
            Algorithm::RdfCustom { options } => {
                let compressed = self.compress_rdf_custom(data, &options)?;
                (compressed, "rdf_custom")
            }
        };

        let compressed_size = compressed.len();
        let ratio = original_size as f64 / compressed_size as f64;
        let compression_time_us = start.elapsed().as_micros() as u64;

        // Update stats
        self.stats.total_original += original_size as u64;
        self.stats.total_compressed += compressed_size as u64;
        self.stats.compression_count += 1;
        self.stats.total_time_us += compression_time_us;

        Ok(CompressionResult {
            data: compressed,
            original_size,
            compressed_size,
            ratio,
            algorithm: algorithm_name.to_string(),
            compression_time_us,
        })
    }

    /// Decompress data
    pub fn decompress(&mut self, data: &[u8]) -> Result<Vec<u8>, OxirsError> {
        match &self.algorithm {
            Algorithm::None => Ok(data.to_vec()),
            Algorithm::Lz4 { .. } => self.decompress_lz4(data),
            Algorithm::Zstd { .. } => self.decompress_zstd(data),
            Algorithm::RdfCustom { .. } => self.decompress_rdf_custom(data),
        }
    }

    /// Compress using LZ4
    fn compress_lz4(&self, data: &[u8], level: u32) -> Result<Vec<u8>, OxirsError> {
        let mut encoder = EncoderBuilder::new().level(level).build(Vec::new())?;
        std::io::copy(&mut &data[..], &mut encoder)?;
        let (compressed, _) = encoder.finish();
        Ok(compressed)
    }

    /// Decompress LZ4
    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>, OxirsError> {
        let mut decoder = Decoder::new(&data[..])?;
        let mut decompressed = Vec::new();
        std::io::copy(&mut decoder, &mut decompressed)?;
        Ok(decompressed)
    }

    /// Compress using Zstandard
    fn compress_zstd(&self, data: &[u8], level: i32) -> Result<Vec<u8>, OxirsError> {
        Ok(zstd::encode_all(&data[..], level)?)
    }

    /// Decompress Zstandard
    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>, OxirsError> {
        Ok(zstd::decode_all(&data[..])?)
    }

    /// Custom RDF compression
    fn compress_rdf_custom(
        &mut self,
        data: &[u8],
        options: &RdfCompressionOptions,
    ) -> Result<Vec<u8>, OxirsError> {
        // Parse RDF data
        let rdf_data = String::from_utf8_lossy(data);

        // Build compressed representation
        let mut compressed = RdfCompressedData {
            version: 1,
            options: options.clone(),
            dictionary: Vec::new(),
            triples: Vec::new(),
        };

        // If using URI dictionary, build it
        if options.uri_dictionary {
            if let Some(dict) = &mut self.uri_dictionary {
                // Extract URIs and build dictionary
                // This is a simplified implementation
                for line in rdf_data.lines() {
                    if let Some(uri) = extract_uri(line) {
                        dict.add_uri(uri);
                    }
                }

                // Store dictionary in compressed data
                compressed.dictionary = dict.export();
            }
        }

        // Compress triples using dictionary
        // This is a placeholder - real implementation would parse and compress properly
        compressed.triples = data.to_vec();

        // Apply secondary compression
        let serialized = bincode::serialize(&compressed)?;
        self.compress_zstd(&serialized, 3)
    }

    /// Decompress custom RDF format
    fn decompress_rdf_custom(&mut self, data: &[u8]) -> Result<Vec<u8>, OxirsError> {
        // Decompress outer layer
        let decompressed = self.decompress_zstd(data)?;

        // Deserialize
        let compressed: RdfCompressedData = bincode::deserialize(&decompressed)?;

        // Restore dictionary
        if compressed.options.uri_dictionary && !compressed.dictionary.is_empty() {
            if let Some(dict) = &mut self.uri_dictionary {
                dict.import(compressed.dictionary);
            }
        }

        // Decompress triples
        // This is a placeholder - real implementation would properly reconstruct
        Ok(compressed.triples)
    }

    /// Get compression statistics
    pub fn stats(&self) -> CompressionStatsSummary {
        CompressionStatsSummary {
            total_compressed_mb: self.stats.total_compressed as f64 / 1_048_576.0,
            total_original_mb: self.stats.total_original as f64 / 1_048_576.0,
            average_ratio: if self.stats.total_compressed > 0 {
                self.stats.total_original as f64 / self.stats.total_compressed as f64
            } else {
                1.0
            },
            compression_count: self.stats.compression_count,
            avg_time_us: if self.stats.compression_count > 0 {
                self.stats.total_time_us / self.stats.compression_count
            } else {
                0
            },
        }
    }
}

/// Compression statistics summary
#[derive(Debug, Clone)]
pub struct CompressionStatsSummary {
    pub total_compressed_mb: f64,
    pub total_original_mb: f64,
    pub average_ratio: f64,
    pub compression_count: u64,
    pub avg_time_us: u64,
}

/// Compressed RDF data format
#[derive(Debug, Serialize, Deserialize)]
struct RdfCompressedData {
    version: u32,
    options: RdfCompressionOptions,
    dictionary: Vec<(String, u32)>,
    triples: Vec<u8>,
}

impl UriDictionary {
    fn new() -> Self {
        let mut dict = UriDictionary {
            uri_to_id: HashMap::new(),
            id_to_uri: HashMap::new(),
            next_id: 0,
            prefixes: Vec::new(),
        };

        // Add common RDF prefixes
        dict.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        dict.add_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#");
        dict.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");
        dict.add_prefix("owl", "http://www.w3.org/2002/07/owl#");

        dict
    }

    fn add_prefix(&mut self, prefix: &str, uri: &str) {
        self.prefixes.push((prefix.to_string(), uri.to_string()));
    }

    fn add_uri(&mut self, uri: &str) -> u32 {
        if let Some(&id) = self.uri_to_id.get(uri) {
            return id;
        }

        let id = self.next_id;
        self.uri_to_id.insert(uri.to_string(), id);
        self.id_to_uri.insert(id, uri.to_string());
        self.next_id += 1;
        id
    }

    #[allow(dead_code)]
    fn get_uri(&self, id: u32) -> Option<&str> {
        self.id_to_uri.get(&id).map(|s| s.as_str())
    }

    fn export(&self) -> Vec<(String, u32)> {
        self.uri_to_id
            .iter()
            .map(|(uri, id)| (uri.clone(), *id))
            .collect()
    }

    fn import(&mut self, data: Vec<(String, u32)>) {
        self.uri_to_id.clear();
        self.id_to_uri.clear();

        for (uri, id) in data {
            self.uri_to_id.insert(uri.clone(), id);
            self.id_to_uri.insert(id, uri);
            self.next_id = self.next_id.max(id + 1);
        }
    }
}

/// Extract URI from a line (simplified)
fn extract_uri(line: &str) -> Option<&str> {
    if let Some(start) = line.find('<') {
        if let Some(end) = line[start..].find('>') {
            return Some(&line[start + 1..start + end]);
        }
    }
    None
}

/// Compress RDF data using the specified algorithm
pub async fn compress_rdf(
    data: &[u8],
    algorithm: Algorithm,
) -> Result<CompressionResult, OxirsError> {
    let mut compressor = Compressor::new(algorithm);
    compressor.compress(data)
}

/// Decompress RDF data
pub async fn decompress_rdf(data: &[u8], algorithm: Algorithm) -> Result<Vec<u8>, OxirsError> {
    let mut compressor = Compressor::new(algorithm);
    compressor.decompress(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_compression() {
        let data = b"<http://example.org/s> <http://example.org/p> <http://example.org/o> .";
        let mut compressor = Compressor::new(Algorithm::Lz4 { level: 1 });

        let result = compressor.compress(data).unwrap();
        assert!(result.compressed_size < result.original_size);

        let decompressed = compressor.decompress(&result.data).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_zstd_compression() {
        let data = b"<http://example.org/s> <http://example.org/p> \"literal value\" .";
        let mut compressor = Compressor::new(Algorithm::Zstd { level: 3 });

        let result = compressor.compress(data).unwrap();
        assert!(result.compressed_size < result.original_size);

        let decompressed = compressor.decompress(&result.data).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_uri_dictionary() {
        let mut dict = UriDictionary::new();

        let id1 = dict.add_uri("http://example.org/test");
        let id2 = dict.add_uri("http://example.org/test");
        assert_eq!(id1, id2);

        let id3 = dict.add_uri("http://example.org/other");
        assert_ne!(id1, id3);

        assert_eq!(dict.get_uri(id1), Some("http://example.org/test"));
    }
}
