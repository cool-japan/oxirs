//! # Advanced Compression Module for TDB Storage
//!
//! Provides advanced compression algorithms including column-store optimizations,
//! bitmap compression, delta encoding, and adaptive compression selection.

pub mod run_length;
pub mod delta;
pub mod frame_of_reference;
pub mod dictionary;
pub mod column_store;
pub mod bitmap;
pub mod adaptive;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// Re-export main compression implementations
pub use run_length::RunLengthEncoder;
pub use delta::DeltaEncoder;
pub use frame_of_reference::FrameOfReferenceEncoder;
pub use dictionary::AdaptiveDictionary;
pub use column_store::ColumnStoreCompressor;
pub use bitmap::{BitmapWAHEncoder, BitmapRoaringEncoder};
pub use adaptive::AdaptiveCompressor;

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

impl Default for CompressionMetadata {
    fn default() -> Self {
        Self {
            algorithm: AdvancedCompressionType::RunLength,
            original_size: 0,
            compressed_size: 0,
            compression_time_us: 0,
            metadata: HashMap::new(),
        }
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

/// Maximum allowed data size for compression (100MB) to prevent resource exhaustion
pub const MAX_COMPRESSION_INPUT_SIZE: usize = 100 * 1024 * 1024;

/// Maximum allowed block count to prevent memory exhaustion
pub const MAX_BLOCK_COUNT: usize = 10000;

/// Maximum allowed dictionary size to prevent memory exhaustion  
pub const MAX_DICTIONARY_SIZE: usize = 100000;

/// Trait for compression algorithms
pub trait CompressionAlgorithm {
    /// Compress data and return compressed bytes with metadata
    fn compress(&self, data: &[u8]) -> Result<CompressedData>;
    
    /// Decompress data
    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>>;
    
    /// Get algorithm type
    fn algorithm_type(&self) -> AdvancedCompressionType;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_metadata() {
        let metadata = CompressionMetadata {
            algorithm: AdvancedCompressionType::RunLength,
            original_size: 1000,
            compressed_size: 500,
            compression_time_us: 100,
            metadata: HashMap::new(),
        };

        assert_eq!(metadata.compression_ratio(), 0.5);
        assert_eq!(metadata.space_savings(), 50.0);
    }

    #[test]
    fn test_compression_type_display() {
        assert_eq!(
            AdvancedCompressionType::RunLength.to_string(),
            "RunLength"
        );
        assert_eq!(
            AdvancedCompressionType::BitmapWAH.to_string(),
            "BitmapWAH"
        );
    }
}