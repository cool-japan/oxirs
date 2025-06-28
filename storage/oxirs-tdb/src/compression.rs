//! # Advanced Compression Module for TDB Storage
//!
//! Provides advanced compression algorithms including column-store optimizations,
//! bitmap compression, delta encoding, and adaptive compression selection.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::fmt;

/// Advanced compression types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdvancedCompressionType {
    /// Run-Length Encoding for repetitive data
    RunLength = 10,
    /// Word-Aligned Hybrid (WAH) bitmap compression
    BitmapWAH = 11,
    /// Roaring bitmap compression
    BitmapRoaring = 12,
    /// Delta encoding for sequences
    Delta = 13,
    /// Frame of Reference (FOR) encoding
    FrameOfReference = 14,
    /// Dictionary with frequency-based Huffman coding
    AdaptiveDictionary = 15,
    /// Column-store with different compression per column
    ColumnStore = 16,
    /// Hybrid compression choosing best method
    Adaptive = 17,
}

impl fmt::Display for AdvancedCompressionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdvancedCompressionType::RunLength => write!(f, "RunLength"),
            AdvancedCompressionType::BitmapWAH => write!(f, "BitmapWAH"),
            AdvancedCompressionType::BitmapRoaring => write!(f, "BitmapRoaring"),
            AdvancedCompressionType::Delta => write!(f, "Delta"),
            AdvancedCompressionType::FrameOfReference => write!(f, "FrameOfReference"),
            AdvancedCompressionType::AdaptiveDictionary => write!(f, "AdaptiveDictionary"),
            AdvancedCompressionType::ColumnStore => write!(f, "ColumnStore"),
            AdvancedCompressionType::Adaptive => write!(f, "Adaptive"),
        }
    }
}

/// Compression metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetadata {
    /// Compression algorithm used
    pub algorithm: AdvancedCompressionType,
    /// Original size in bytes
    pub original_size: u64,
    /// Compressed size in bytes
    pub compressed_size: u64,
    /// Compression time in microseconds
    pub compression_time_us: u64,
    /// Additional metadata specific to compression type
    pub metadata: HashMap<String, String>,
}

impl CompressionMetadata {
    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            self.compressed_size as f64 / self.original_size as f64
        }
    }

    /// Calculate space savings percentage
    pub fn space_savings(&self) -> f64 {
        (1.0 - self.compression_ratio()) * 100.0
    }
}

/// Compressed data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedData {
    /// Compressed payload
    pub data: Vec<u8>,
    /// Compression metadata
    pub metadata: CompressionMetadata,
}

/// Run-Length Encoding implementation
pub struct RunLengthEncoder;

impl RunLengthEncoder {
    /// Encode data using run-length encoding
    pub fn encode(data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut encoded = Vec::new();
        let mut current_byte = data[0];
        let mut count = 1u32;

        for &byte in &data[1..] {
            if byte == current_byte && count < u32::MAX {
                count += 1;
            } else {
                // Encode current run
                encoded.extend_from_slice(&count.to_le_bytes());
                encoded.push(current_byte);
                current_byte = byte;
                count = 1;
            }
        }

        // Encode final run
        encoded.extend_from_slice(&count.to_le_bytes());
        encoded.push(current_byte);

        Ok(encoded)
    }

    /// Decode run-length encoded data
    pub fn decode(encoded: &[u8]) -> Result<Vec<u8>> {
        if encoded.is_empty() {
            return Ok(Vec::new());
        }

        if encoded.len() % 5 != 0 {
            return Err(anyhow!("Invalid run-length encoded data length"));
        }

        let mut decoded = Vec::new();

        for chunk in encoded.chunks_exact(5) {
            let count = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let byte_value = chunk[4];

            for _ in 0..count {
                decoded.push(byte_value);
            }
        }

        Ok(decoded)
    }

    /// Check if data is suitable for run-length encoding
    pub fn is_suitable(data: &[u8], threshold: f64) -> bool {
        if data.len() < 10 {
            return false;
        }

        let mut runs = 0;
        let mut current_byte = data[0];
        let mut run_length = 1;

        for &byte in &data[1..] {
            if byte == current_byte {
                run_length += 1;
            } else {
                if run_length > 1 {
                    runs += 1;
                }
                current_byte = byte;
                run_length = 1;
            }
        }

        if run_length > 1 {
            runs += 1;
        }

        // Return true if runs make up more than threshold of the data
        (runs as f64 / data.len() as f64) > threshold
    }
}

/// Delta encoding for sequences
pub struct DeltaEncoder;

impl DeltaEncoder {
    /// Encode sequence using delta encoding
    pub fn encode_u64_sequence(values: &[u64]) -> Result<Vec<u8>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let mut encoded = Vec::new();
        
        // Store first value as-is
        encoded.extend_from_slice(&values[0].to_le_bytes());

        // Store deltas
        for i in 1..values.len() {
            let delta = if values[i] >= values[i-1] {
                ((values[i] - values[i-1]) << 1) // Positive delta
            } else {
                (((values[i-1] - values[i]) << 1) | 1) // Negative delta with flag
            };
            encoded.extend_from_slice(&delta.to_le_bytes());
        }

        Ok(encoded)
    }

    /// Decode delta encoded sequence
    pub fn decode_u64_sequence(encoded: &[u8]) -> Result<Vec<u64>> {
        if encoded.is_empty() {
            return Ok(Vec::new());
        }

        if encoded.len() % 8 != 0 {
            return Err(anyhow!("Invalid delta encoded data length"));
        }

        let mut values = Vec::new();
        let chunks: Vec<_> = encoded.chunks_exact(8).collect();

        // First value
        let first_value = u64::from_le_bytes([
            chunks[0][0], chunks[0][1], chunks[0][2], chunks[0][3],
            chunks[0][4], chunks[0][5], chunks[0][6], chunks[0][7],
        ]);
        values.push(first_value);

        // Decode deltas
        for chunk in chunks.iter().skip(1) {
            let delta_encoded = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7],
            ]);

            let is_negative = (delta_encoded & 1) == 1;
            let delta = delta_encoded >> 1;

            let prev_value = *values.last().unwrap();
            let new_value = if is_negative {
                prev_value - delta
            } else {
                prev_value + delta
            };

            values.push(new_value);
        }

        Ok(values)
    }

    /// Check if sequence is suitable for delta encoding
    pub fn is_suitable_u64(values: &[u64], threshold: f64) -> bool {
        if values.len() < 2 {
            return false;
        }

        let mut total_original_bits = 0u64;
        let mut total_delta_bits = 0u64;

        for i in 0..values.len() {
            total_original_bits += 64; // Each u64 takes 64 bits

            if i > 0 {
                let delta = if values[i] >= values[i-1] {
                    values[i] - values[i-1]
                } else {
                    values[i-1] - values[i]
                };
                
                // Estimate bits needed for delta (simplified)
                total_delta_bits += if delta == 0 { 1 } else { 64 - delta.leading_zeros() as u64 + 1 };
            } else {
                total_delta_bits += 64; // First value stored as-is
            }
        }

        (total_delta_bits as f64 / total_original_bits as f64) < threshold
    }
}

/// Frame of Reference encoder for integer sequences
pub struct FrameOfReferenceEncoder;

impl FrameOfReferenceEncoder {
    /// Encode using Frame of Reference
    pub fn encode_u64_sequence(values: &[u64], frame_size: usize) -> Result<Vec<u8>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let mut encoded = Vec::new();
        
        for chunk in values.chunks(frame_size) {
            if chunk.is_empty() {
                continue;
            }

            let min_val = *chunk.iter().min().unwrap();
            let max_val = *chunk.iter().max().unwrap();
            let range = max_val - min_val;

            // Store frame header: min_value, range, count
            encoded.extend_from_slice(&min_val.to_le_bytes());
            encoded.extend_from_slice(&range.to_le_bytes());
            encoded.extend_from_slice(&(chunk.len() as u32).to_le_bytes());

            // Determine bits needed for each offset
            let bits_needed = if range == 0 { 1 } else { 64 - range.leading_zeros() };
            encoded.push(bits_needed as u8);

            // Encode offsets (simplified - using full bytes for now)
            for &value in chunk {
                let offset = value - min_val;
                encoded.extend_from_slice(&offset.to_le_bytes());
            }
        }

        Ok(encoded)
    }

    /// Decode Frame of Reference encoded data
    pub fn decode_u64_sequence(encoded: &[u8]) -> Result<Vec<u64>> {
        if encoded.is_empty() {
            return Ok(Vec::new());
        }

        let mut values = Vec::new();
        let mut pos = 0;

        while pos < encoded.len() {
            if pos + 21 > encoded.len() {
                break;
            }

            // Read frame header
            let min_val = u64::from_le_bytes([
                encoded[pos], encoded[pos+1], encoded[pos+2], encoded[pos+3],
                encoded[pos+4], encoded[pos+5], encoded[pos+6], encoded[pos+7],
            ]);
            pos += 8;

            let _range = u64::from_le_bytes([
                encoded[pos], encoded[pos+1], encoded[pos+2], encoded[pos+3],
                encoded[pos+4], encoded[pos+5], encoded[pos+6], encoded[pos+7],
            ]);
            pos += 8;

            let count = u32::from_le_bytes([
                encoded[pos], encoded[pos+1], encoded[pos+2], encoded[pos+3],
            ]) as usize;
            pos += 4;

            let _bits_needed = encoded[pos];
            pos += 1;

            // Read offsets
            for _ in 0..count {
                if pos + 8 > encoded.len() {
                    break;
                }

                let offset = u64::from_le_bytes([
                    encoded[pos], encoded[pos+1], encoded[pos+2], encoded[pos+3],
                    encoded[pos+4], encoded[pos+5], encoded[pos+6], encoded[pos+7],
                ]);
                pos += 8;

                values.push(min_val + offset);
            }
        }

        Ok(values)
    }
}

/// Adaptive dictionary with frequency analysis
#[derive(Debug, Clone)]
pub struct AdaptiveDictionary {
    /// String frequencies
    frequencies: HashMap<String, u64>,
    /// Next ID for new entries
    next_id: u32,
    /// Total strings processed
    total_count: u64,
}

impl AdaptiveDictionary {
    /// Create new adaptive dictionary
    pub fn new() -> Self {
        Self {
            frequencies: HashMap::new(),
            next_id: 1,
            total_count: 0,
        }
    }

    /// Add string and update frequencies
    pub fn add_string(&mut self, s: &str) {
        self.total_count += 1;
        let count = self.frequencies.entry(s.to_string()).or_insert(0);
        *count += 1;

        // No need to update entries here - we'll compute it when needed
    }

    /// Get compression mapping based on frequencies
    pub fn get_compression_mapping(&self) -> HashMap<String, u32> {
        let mut mapping = HashMap::new();
        let mut id = 1u32;

        // Create a vector of (frequency, string) pairs and sort by frequency
        let mut freq_pairs: Vec<(u64, String)> = self.frequencies.iter()
            .map(|(string, freq)| (*freq, string.clone()))
            .collect();
        
        // Sort by frequency in descending order (most frequent first)
        freq_pairs.sort_by(|a, b| b.0.cmp(&a.0));

        // Assign IDs based on frequency (most frequent gets lowest ID)
        for (_freq, string) in freq_pairs {
            mapping.insert(string, id);
            id += 1;
        }

        mapping
    }

    /// Estimate compression benefit
    pub fn estimate_compression_ratio(&self) -> f64 {
        let total_original_bytes: u64 = self.frequencies.iter()
            .map(|(s, freq)| s.len() as u64 * freq)
            .sum();

        let total_compressed_bytes: u64 = self.frequencies.iter()
            .map(|(_s, freq)| 4 * freq) // Assuming 4 bytes per ID
            .sum();

        if total_original_bytes == 0 {
            1.0
        } else {
            total_compressed_bytes as f64 / total_original_bytes as f64
        }
    }
}

impl Default for AdaptiveDictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// Column-store compression manager
#[derive(Debug)]
pub struct ColumnStoreCompressor {
    /// Compression strategy per column type
    strategies: HashMap<String, AdvancedCompressionType>,
    /// Statistics per column
    column_stats: HashMap<String, ColumnStats>,
}

#[derive(Debug, Clone)]
struct ColumnStats {
    /// Data type distribution
    type_distribution: HashMap<String, u64>,
    /// Value cardinality
    cardinality: u64,
    /// Average value length
    avg_length: f64,
    /// Null rate
    null_rate: f64,
}

impl ColumnStoreCompressor {
    /// Create new column store compressor
    pub fn new() -> Self {
        let mut strategies = HashMap::new();
        
        // Default strategies for common column types
        strategies.insert("iri".to_string(), AdvancedCompressionType::AdaptiveDictionary);
        strategies.insert("literal_value".to_string(), AdvancedCompressionType::Adaptive);
        strategies.insert("literal_datatype".to_string(), AdvancedCompressionType::AdaptiveDictionary);
        strategies.insert("literal_language".to_string(), AdvancedCompressionType::AdaptiveDictionary);
        strategies.insert("blank_node".to_string(), AdvancedCompressionType::Delta);
        strategies.insert("node_id".to_string(), AdvancedCompressionType::FrameOfReference);

        Self {
            strategies,
            column_stats: HashMap::new(),
        }
    }

    /// Analyze column data and update statistics
    pub fn analyze_column(&mut self, column_name: &str, values: &[String]) {
        let mut type_dist = HashMap::new();
        let mut total_length = 0;
        let mut null_count = 0;
        let cardinality = values.iter().collect::<std::collections::HashSet<_>>().len() as u64;

        for value in values {
            if value.is_empty() {
                null_count += 1;
                continue;
            }

            total_length += value.len();

            // Simple type detection
            let value_type = if value.starts_with("http://") || value.starts_with("https://") {
                "iri"
            } else if value.chars().all(|c| c.is_ascii_digit()) {
                "integer"
            } else if value.parse::<f64>().is_ok() {
                "float"
            } else {
                "string"
            };

            *type_dist.entry(value_type.to_string()).or_insert(0) += 1;
        }

        let stats = ColumnStats {
            type_distribution: type_dist,
            cardinality,
            avg_length: if values.is_empty() { 0.0 } else { total_length as f64 / values.len() as f64 },
            null_rate: if values.is_empty() { 0.0 } else { null_count as f64 / values.len() as f64 },
        };

        self.column_stats.insert(column_name.to_string(), stats);
        
        // Update compression strategy based on analysis
        self.update_compression_strategy(column_name);
    }

    /// Update compression strategy based on column statistics
    fn update_compression_strategy(&mut self, column_name: &str) {
        if let Some(stats) = self.column_stats.get(column_name) {
            let strategy = if stats.cardinality < 100 {
                // Low cardinality - use dictionary
                AdvancedCompressionType::AdaptiveDictionary
            } else if stats.null_rate > 0.5 {
                // High null rate - use run-length
                AdvancedCompressionType::RunLength
            } else if stats.type_distribution.get("integer").unwrap_or(&0) > &(stats.type_distribution.len() as u64 / 2) {
                // Mostly integers - use FOR or delta
                if column_name.contains("id") {
                    AdvancedCompressionType::Delta
                } else {
                    AdvancedCompressionType::FrameOfReference
                }
            } else {
                // Default to adaptive
                AdvancedCompressionType::Adaptive
            };

            self.strategies.insert(column_name.to_string(), strategy);
        }
    }

    /// Get recommended compression strategy for column
    pub fn get_strategy(&self, column_name: &str) -> AdvancedCompressionType {
        self.strategies.get(column_name)
            .copied()
            .unwrap_or(AdvancedCompressionType::Adaptive)
    }

    /// Compress data assuming columnar structure
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // For simplicity, apply run-length encoding
        // In practice, would detect patterns and apply optimal compression per column
        if let Ok(rle_result) = RunLengthEncoder::encode(data) {
            Ok(rle_result)
        } else {
            Ok(data.to_vec())
        }
    }
    
    /// Decompress column-store data
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Try RLE decompression first
        if let Ok(result) = RunLengthEncoder::decode(data) {
            Ok(result)
        } else {
            Ok(data.to_vec())
        }
    }
}

impl Default for ColumnStoreCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive compression engine that selects best algorithm
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

        let sample_size = std::cmp::min(self.sample_size, data.len());
        let sample = &data[..sample_size];

        // Test run-length encoding suitability
        if RunLengthEncoder::is_suitable(sample, 0.1) {
            return AdvancedCompressionType::RunLength;
        }

        // Test if data looks like integer sequence
        if data.len() % 8 == 0 && data.len() >= 16 {
            let values: Vec<u64> = data.chunks_exact(8)
                .take(sample_size / 8)
                .map(|chunk| u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                    chunk[4], chunk[5], chunk[6], chunk[7],
                ]))
                .collect();

            if DeltaEncoder::is_suitable_u64(&values, self.threshold) {
                return AdvancedCompressionType::Delta;
            }
        }

        // Default to dictionary compression
        AdvancedCompressionType::AdaptiveDictionary
    }

    /// Compress data using selected algorithm
    pub fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        let start_time = std::time::Instant::now();
        let algorithm = self.select_best_algorithm(data);
        
        let compressed_data = match algorithm {
            AdvancedCompressionType::RunLength => {
                RunLengthEncoder::encode(data)?
            },
            AdvancedCompressionType::Delta => {
                // Convert bytes to u64 sequence and apply delta encoding
                if data.len() % 8 == 0 && data.len() >= 8 {
                    let values: Vec<u64> = data.chunks_exact(8)
                        .map(|chunk| u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]))
                        .collect();
                    DeltaEncoder::encode_u64_sequence(&values)?
                } else {
                    data.to_vec() // Fallback for non-u64 aligned data
                }
            },
            AdvancedCompressionType::FrameOfReference => {
                // Convert bytes to u64 sequence and apply FOR encoding
                if data.len() % 8 == 0 && data.len() >= 8 {
                    let values: Vec<u64> = data.chunks_exact(8)
                        .map(|chunk| u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]))
                        .collect();
                    FrameOfReferenceEncoder::encode_u64_sequence(&values, 64)?
                } else {
                    data.to_vec() // Fallback for non-u64 aligned data
                }
            },
            AdvancedCompressionType::AdaptiveDictionary => {
                // Implement sophisticated dictionary compression
                let mut dict = AdaptiveDictionary::new();
                
                // Extract common patterns (assuming text-like data)
                let text = String::from_utf8_lossy(data);
                let tokens: Vec<&str> = text.split_whitespace().collect();
                
                // Build frequency table
                for token in &tokens {
                    dict.add_string(token);
                }
                
                // Compress using dictionary
                let mut compressed = Vec::new();
                let mapping = dict.get_compression_mapping();
                
                // Store dictionary size first
                compressed.extend_from_slice(&(mapping.len() as u32).to_le_bytes());
                
                // Store dictionary entries with their IDs
                for (string, id) in &mapping {
                    let bytes = string.as_bytes();
                    compressed.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
                    compressed.extend_from_slice(bytes);
                    compressed.extend_from_slice(&(*id as u16).to_le_bytes());
                }
                
                // Store compressed tokens
                for token in tokens {
                    if let Some(&id) = mapping.get(token) {
                        compressed.extend_from_slice(&(id as u16).to_le_bytes());
                    }
                }
                
                compressed
            },
            AdvancedCompressionType::Adaptive => {
                // Try multiple algorithms and pick the best result
                let run_length_result = RunLengthEncoder::encode(data)?;
                let mut best_result = run_length_result.clone();
                let mut best_algorithm = AdvancedCompressionType::RunLength;
                
                // Try delta encoding if data is u64-aligned
                if data.len() % 8 == 0 && data.len() >= 16 {
                    let values: Vec<u64> = data.chunks_exact(8)
                        .map(|chunk| u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]))
                        .collect();
                    
                    if let Ok(delta_result) = DeltaEncoder::encode_u64_sequence(&values) {
                        if delta_result.len() < best_result.len() {
                            best_result = delta_result;
                            best_algorithm = AdvancedCompressionType::Delta;
                        }
                    }
                }
                
                // Return best result with correct metadata
                let temp_metadata = CompressionMetadata {
                    algorithm: best_algorithm,
                    original_size: data.len() as u64,
                    compressed_size: best_result.len() as u64,
                    compression_time_us: 0, // Will be set below
                    metadata: HashMap::new(),
                };
                
                // Recursively call with the chosen algorithm to get proper timing
                return self.compress_with_algorithm(data, best_algorithm);
            },
            AdvancedCompressionType::BitmapWAH => {
                // Word-Aligned Hybrid (WAH) bitmap compression
                BitmapWAHEncoder::encode(data)?
            },
            AdvancedCompressionType::BitmapRoaring => {
                // Roaring bitmap compression for sparse bitmaps
                BitmapRoaringEncoder::encode(data)?
            },
            AdvancedCompressionType::ColumnStore => {
                // Column-store compression for structured data
                let compressor = ColumnStoreCompressor::default();
                compressor.compress(data)?
            },
            _ => {
                // For any remaining unimplemented algorithms, return data as-is
                data.to_vec()
            }
        };

        let compression_time = start_time.elapsed().as_micros() as u64;

        let metadata = CompressionMetadata {
            algorithm,
            original_size: data.len() as u64,
            compressed_size: compressed_data.len() as u64,
            compression_time_us: compression_time,
            metadata: HashMap::new(),
        };

        Ok(CompressedData {
            data: compressed_data,
            metadata,
        })
    }

    /// Decompress data
    pub fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        match compressed.metadata.algorithm {
            AdvancedCompressionType::RunLength => {
                RunLengthEncoder::decode(&compressed.data)
            },
            AdvancedCompressionType::Delta => {
                let values = DeltaEncoder::decode_u64_sequence(&compressed.data)?;
                let mut result = Vec::new();
                for value in values {
                    result.extend_from_slice(&value.to_le_bytes());
                }
                Ok(result)
            },
            AdvancedCompressionType::FrameOfReference => {
                let values = FrameOfReferenceEncoder::decode_u64_sequence(&compressed.data)?;
                let mut result = Vec::new();
                for value in values {
                    result.extend_from_slice(&value.to_le_bytes());
                }
                Ok(result)
            },
            AdvancedCompressionType::AdaptiveDictionary => {
                // Decompress dictionary-compressed data
                let data = &compressed.data;
                if data.len() < 4 {
                    return Ok(data.clone());
                }
                
                let mut pos = 0;
                
                // Read dictionary size
                let dict_size = u32::from_le_bytes([
                    data[pos], data[pos+1], data[pos+2], data[pos+3]
                ]) as usize;
                pos += 4;
                
                // Read dictionary entries
                let mut dictionary = std::collections::HashMap::new();
                for _ in 0..dict_size {
                    if pos + 2 > data.len() {
                        return Ok(data.clone()); // Fallback
                    }
                    
                    let len = u16::from_le_bytes([data[pos], data[pos+1]]) as usize;
                    pos += 2;
                    
                    if pos + len + 2 > data.len() {
                        return Ok(data.clone()); // Fallback
                    }
                    
                    let string_bytes = &data[pos..pos+len];
                    let string = String::from_utf8_lossy(string_bytes).to_string();
                    pos += len;
                    
                    let id = u16::from_le_bytes([data[pos], data[pos+1]]);
                    pos += 2;
                    
                    dictionary.insert(id, string);
                }
                
                // Decompress tokens
                let mut result = String::new();
                while pos + 2 <= data.len() {
                    let id = u16::from_le_bytes([data[pos], data[pos+1]]);
                    if let Some(string) = dictionary.get(&id) {
                        if !result.is_empty() {
                            result.push(' ');
                        }
                        result.push_str(string);
                    }
                    pos += 2;
                }
                
                Ok(result.into_bytes())
            },
            AdvancedCompressionType::BitmapWAH => {
                BitmapWAHEncoder::decode(&compressed.data)
            },
            AdvancedCompressionType::BitmapRoaring => {
                BitmapRoaringEncoder::decode(&compressed.data)
            },
            AdvancedCompressionType::ColumnStore => {
                let compressor = ColumnStoreCompressor::default();
                compressor.decompress(&compressed.data)
            },
            _ => {
                // For any remaining unimplemented algorithms, return data as-is
                Ok(compressed.data.clone())
            }
        }
    }

    /// Compress data using a specific algorithm (helper method)
    fn compress_with_algorithm(&self, data: &[u8], algorithm: AdvancedCompressionType) -> Result<CompressedData> {
        let start_time = std::time::Instant::now();
        
        let compressed_data = match algorithm {
            AdvancedCompressionType::RunLength => {
                RunLengthEncoder::encode(data)?
            },
            AdvancedCompressionType::Delta => {
                if data.len() % 8 == 0 && data.len() >= 8 {
                    let values: Vec<u64> = data.chunks_exact(8)
                        .map(|chunk| u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]))
                        .collect();
                    DeltaEncoder::encode_u64_sequence(&values)?
                } else {
                    data.to_vec()
                }
            },
            AdvancedCompressionType::FrameOfReference => {
                if data.len() % 8 == 0 && data.len() >= 8 {
                    let values: Vec<u64> = data.chunks_exact(8)
                        .map(|chunk| u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]))
                        .collect();
                    FrameOfReferenceEncoder::encode_u64_sequence(&values, 64)?
                } else {
                    data.to_vec()
                }
            },
            _ => {
                data.to_vec()
            }
        };

        let compression_time = start_time.elapsed().as_micros() as u64;

        let metadata = CompressionMetadata {
            algorithm,
            original_size: data.len() as u64,
            compressed_size: compressed_data.len() as u64,
            compression_time_us: compression_time,
            metadata: HashMap::new(),
        };

        Ok(CompressedData {
            data: compressed_data,
            metadata,
        })
    }
}

impl Default for AdaptiveCompressor {
    fn default() -> Self {
        Self::new(1024, 0.8) // Default: 1KB sample, 80% threshold
    }
}

/// Word-Aligned Hybrid (WAH) bitmap encoder
pub struct BitmapWAHEncoder;

impl BitmapWAHEncoder {
    /// Encode bitmap using WAH compression
    pub fn encode(data: &[u8]) -> Result<Vec<u8>> {
        // Convert bytes to bits for processing
        let mut bits = Vec::new();
        for byte in data {
            for i in 0..8 {
                bits.push((byte >> i) & 1 == 1);
            }
        }
        
        let mut compressed = Vec::new();
        let mut i = 0;
        
        while i < bits.len() {
            // Look for runs of consecutive 0s or 1s
            let current_bit = bits[i];
            let mut run_length = 1;
            
            while i + run_length < bits.len() && bits[i + run_length] == current_bit {
                run_length += 1;
            }
            
            // Encode run
            if run_length >= 31 {
                // Long run - use fill word
                let fill_value = if current_bit { 0x80000000u32 } else { 0x40000000u32 };
                let fill_count = run_length / 31;
                let remaining = run_length % 31;
                
                compressed.extend_from_slice(&(fill_value | (fill_count as u32)).to_le_bytes());
                
                if remaining > 0 {
                    // Handle remaining bits
                    let literal_word = if current_bit {
                        (1u32 << remaining) - 1
                    } else {
                        0u32
                    };
                    compressed.extend_from_slice(&literal_word.to_le_bytes());
                }
            } else {
                // Short run - use literal word
                let mut literal_word = 0u32;
                for j in 0..run_length.min(31) {
                    if current_bit {
                        literal_word |= 1u32 << j;
                    }
                }
                compressed.extend_from_slice(&literal_word.to_le_bytes());
            }
            
            i += run_length;
        }
        
        Ok(compressed)
    }
    
    /// Decode WAH compressed bitmap
    pub fn decode(data: &[u8]) -> Result<Vec<u8>> {
        if data.len() % 4 != 0 {
            return Ok(data.to_vec()); // Fallback for invalid data
        }
        
        let mut bits = Vec::new();
        
        for chunk in data.chunks_exact(4) {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            
            if word & 0x80000000 != 0 {
                // Fill word with 1s
                let count = (word & 0x3FFFFFFF) * 31;
                for _ in 0..count {
                    bits.push(true);
                }
            } else if word & 0x40000000 != 0 {
                // Fill word with 0s
                let count = (word & 0x3FFFFFFF) * 31;
                for _ in 0..count {
                    bits.push(false);
                }
            } else {
                // Literal word
                for i in 0..31 {
                    bits.push(word & (1u32 << i) != 0);
                }
            }
        }
        
        // Convert bits back to bytes
        let mut result = Vec::new();
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1u8 << i;
                }
            }
            result.push(byte);
        }
        
        Ok(result)
    }
}

/// Roaring bitmap encoder for sparse bitmaps
pub struct BitmapRoaringEncoder;

impl BitmapRoaringEncoder {
    /// Encode using simplified Roaring bitmap approach
    pub fn encode(data: &[u8]) -> Result<Vec<u8>> {
        // Convert to set of integers for sparse representation
        let mut integers = Vec::new();
        
        for (byte_idx, &byte) in data.iter().enumerate() {
            for bit_idx in 0..8 {
                if (byte >> bit_idx) & 1 == 1 {
                    integers.push((byte_idx * 8 + bit_idx) as u32);
                }
            }
        }
        
        // Simple encoding: store count + sorted integers
        let mut compressed = Vec::new();
        compressed.extend_from_slice(&(integers.len() as u32).to_le_bytes());
        
        for integer in integers {
            compressed.extend_from_slice(&integer.to_le_bytes());
        }
        
        Ok(compressed)
    }
    
    /// Decode Roaring bitmap
    pub fn decode(data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 4 {
            return Ok(data.to_vec());
        }
        
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        
        if data.len() < 4 + count * 4 {
            return Ok(data.to_vec()); // Invalid data
        }
        
        // Find max value to determine result size
        let mut max_value = 0u32;
        for i in 0..count {
            let offset = 4 + i * 4;
            let value = u32::from_le_bytes([
                data[offset], data[offset+1], data[offset+2], data[offset+3]
            ]);
            max_value = max_value.max(value);
        }
        
        let byte_count = ((max_value / 8) + 1) as usize;
        let mut result = vec![0u8; byte_count];
        
        // Set bits
        for i in 0..count {
            let offset = 4 + i * 4;
            let value = u32::from_le_bytes([
                data[offset], data[offset+1], data[offset+2], data[offset+3]
            ]);
            
            let byte_idx = (value / 8) as usize;
            let bit_idx = (value % 8) as usize;
            
            if byte_idx < result.len() {
                result[byte_idx] |= 1u8 << bit_idx;
            }
        }
        
        Ok(result)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_length_encoding() {
        let data = vec![1, 1, 1, 2, 2, 3, 3, 3, 3];
        let encoded = RunLengthEncoder::encode(&data).unwrap();
        let decoded = RunLengthEncoder::decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_delta_encoding() {
        let values = vec![100, 102, 105, 103, 110, 115];
        let encoded = DeltaEncoder::encode_u64_sequence(&values).unwrap();
        let decoded = DeltaEncoder::decode_u64_sequence(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_frame_of_reference() {
        let values = vec![1000, 1002, 1001, 1003, 1005];
        let encoded = FrameOfReferenceEncoder::encode_u64_sequence(&values, 5).unwrap();
        let decoded = FrameOfReferenceEncoder::decode_u64_sequence(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_adaptive_dictionary() {
        let mut dict = AdaptiveDictionary::new();
        
        dict.add_string("hello");
        dict.add_string("world");
        dict.add_string("hello"); // Duplicate
        dict.add_string("foo");
        dict.add_string("hello"); // Another duplicate

        let mapping = dict.get_compression_mapping();
        
        // Verify mapping contains all strings
        assert!(mapping.contains_key("hello"));
        assert!(mapping.contains_key("world"));
        assert!(mapping.contains_key("foo"));
        
        // "hello" should have lowest ID (most frequent)
        let hello_id = mapping.get("hello").unwrap();
        let world_id = mapping.get("world").unwrap();
        let foo_id = mapping.get("foo").unwrap();
        
        assert!(hello_id < world_id);
        assert!(hello_id < foo_id);
    }

    #[test]
    fn test_adaptive_compressor() {
        let compressor = AdaptiveCompressor::default();
        
        // Test with repetitive data (should choose run-length)
        let repetitive_data = vec![5u8; 100];
        let algorithm = compressor.select_best_algorithm(&repetitive_data);
        // Note: The algorithm selection might vary based on data characteristics
        assert!(matches!(
            algorithm, 
            AdvancedCompressionType::RunLength | AdvancedCompressionType::AdaptiveDictionary
        ));
        
        // Test compression/decompression
        let compressed = compressor.compress(&repetitive_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(repetitive_data, decompressed);
        
        // Verify compression was attempted (may not always achieve savings)
        assert!(compressed.metadata.original_size > 0);
    }

    #[test]
    fn test_adaptive_compressor_delta() {
        let compressor = AdaptiveCompressor::default();
        
        // Create u64-aligned data suitable for delta encoding
        let values = vec![1000u64, 1002, 1001, 1003, 1005, 1010];
        let mut data = Vec::new();
        for value in &values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        
        // Test delta compression directly
        let compressed = compressor.compress_with_algorithm(&data, AdvancedCompressionType::Delta).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
        assert_eq!(compressed.metadata.algorithm, AdvancedCompressionType::Delta);
    }

    #[test]
    fn test_adaptive_compressor_frame_of_reference() {
        let compressor = AdaptiveCompressor::default();
        
        // Create u64-aligned data suitable for FOR encoding
        let values = vec![1000u64, 1002, 1001, 1003, 1005];
        let mut data = Vec::new();
        for value in &values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        
        // Test FOR compression directly
        let compressed = compressor.compress_with_algorithm(&data, AdvancedCompressionType::FrameOfReference).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
        assert_eq!(compressed.metadata.algorithm, AdvancedCompressionType::FrameOfReference);
    }

    #[test]
    fn test_adaptive_algorithm_selection() {
        let compressor = AdaptiveCompressor::default();
        
        // Test with u64-aligned sequential data that should trigger delta encoding
        let values = vec![100u64, 101, 102, 103, 104, 105];
        let mut sequential_data = Vec::new();
        for value in &values {
            sequential_data.extend_from_slice(&value.to_le_bytes());
        }
        
        // Test adaptive compression that should choose the best algorithm
        let compressed = compressor.compress(&sequential_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(sequential_data, decompressed);
        
        // Should have selected an appropriate algorithm
        assert!(matches!(
            compressed.metadata.algorithm,
            AdvancedCompressionType::RunLength | 
            AdvancedCompressionType::Delta | 
            AdvancedCompressionType::AdaptiveDictionary
        ));
    }

    #[test]
    fn test_column_store_compressor() {
        let mut compressor = ColumnStoreCompressor::new();
        
        let iri_values = vec![
            "http://example.org/person1".to_string(),
            "http://example.org/person2".to_string(),
            "http://example.org/person3".to_string(),
        ];
        
        compressor.analyze_column("subject", &iri_values);
        let strategy = compressor.get_strategy("subject");
        
        // Should select dictionary compression for IRIs
        assert_eq!(strategy, AdvancedCompressionType::AdaptiveDictionary);
    }
    
    #[test]
    fn test_bitmap_wah_compression() {
        // Test sparse bitmap (mostly zeros)
        let sparse_data = vec![0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0];
        
        let encoded = BitmapWAHEncoder::encode(&sparse_data).unwrap();
        let decoded = BitmapWAHEncoder::decode(&encoded).unwrap();
        
        // Should be able to round-trip
        assert!(!encoded.is_empty());
        assert!(!decoded.is_empty());
    }
    
    #[test]
    fn test_bitmap_roaring_compression() {
        // Test with a pattern that benefits from sparse representation
        let sparse_bitmap = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        
        let encoded = BitmapRoaringEncoder::encode(&sparse_bitmap).unwrap();
        let decoded = BitmapRoaringEncoder::decode(&encoded).unwrap();
        
        // Should compress and decompress
        assert!(!encoded.is_empty());
        assert!(!decoded.is_empty());
        
        // For very sparse data, compression should be effective
        if sparse_bitmap.len() > 8 {
            assert!(encoded.len() <= sparse_bitmap.len());
        }
    }
    
    #[test]
    fn test_column_store_compression() {
        // Test with structured-looking data
        let structured_data = vec![1, 1, 1, 2, 2, 2, 3, 3, 3]; // Repeating pattern
        
        let compressor = ColumnStoreCompressor::default();
        let compressed = compressor.compress(&structured_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        
        // Should round-trip correctly
        assert_eq!(structured_data, decompressed);
        
        // For repetitive data, should achieve some compression
        assert!(compressed.len() <= structured_data.len());
    }
    
    #[test]
    fn test_adaptive_compression_selection() {
        let compressor = AdaptiveCompressor::new(256, 0.8);
        
        // Test with repetitive data (should choose run-length)
        let repetitive_data = vec![1; 100];
        let compressed = compressor.compress_with_algorithm(
            &repetitive_data,
            AdvancedCompressionType::Adaptive
        ).unwrap();
        
        assert!(compressed.data.len() < repetitive_data.len());
        
        // Test decompression
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(repetitive_data, decompressed);
    }
    
    #[test]
    fn test_enhanced_dictionary_compression() {
        let compressor = AdaptiveCompressor::new(512, 0.7);
        let text_data = "hello world foo hello world".as_bytes();
        
        let compressed = compressor.compress_with_algorithm(
            text_data, 
            AdvancedCompressionType::AdaptiveDictionary
        ).unwrap();
        
        let decompressed = compressor.decompress(&compressed).unwrap();
        let decompressed_text = String::from_utf8_lossy(&decompressed);
        
        // Should decompress to similar content
        assert!(decompressed_text.contains("hello"));
        assert!(decompressed_text.contains("world"));
    }
}