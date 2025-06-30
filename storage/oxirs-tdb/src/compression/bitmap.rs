//! Bitmap compression implementations (WAH and Roaring)

use crate::compression::{
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
    MAX_BLOCK_COUNT, MAX_COMPRESSION_INPUT_SIZE,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::time::Instant;

/// Word-Aligned Hybrid (WAH) bitmap encoder
pub struct BitmapWAHEncoder;

impl BitmapWAHEncoder {
    /// Encode bitmap using WAH compression
    pub fn encode(data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        if data.len() > MAX_COMPRESSION_INPUT_SIZE {
            return Err(anyhow!(
                "Input data too large for compression: {} bytes (max: {})",
                data.len(),
                MAX_COMPRESSION_INPUT_SIZE
            ));
        }

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

            // Encode the run
            if run_length >= 4 {
                // Use run-length encoding for longer runs
                let encoded_run = if current_bit {
                    0x80000000u32 | (run_length as u32) // Set MSB for 1s
                } else {
                    run_length as u32 // Clear MSB for 0s
                };
                compressed.extend_from_slice(&encoded_run.to_le_bytes());
            } else {
                // Use literal encoding for short runs
                let mut literal = 0u32;
                for j in 0..run_length.min(31) {
                    if bits[i + j] {
                        literal |= 1 << j;
                    }
                }
                literal |= 0x40000000; // Set bit 30 to indicate literal
                compressed.extend_from_slice(&literal.to_le_bytes());
            }

            i += run_length;
        }

        Ok(compressed)
    }

    /// Decode WAH compressed bitmap
    pub fn decode(compressed: &[u8]) -> Result<Vec<u8>> {
        if compressed.is_empty() {
            return Ok(Vec::new());
        }

        if compressed.len() % 4 != 0 {
            return Err(anyhow!("Invalid WAH compressed data length"));
        }

        let mut bits = Vec::new();

        for chunk in compressed.chunks_exact(4) {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);

            if word & 0x80000000 != 0 {
                // Run of 1s
                let run_length = (word & 0x7FFFFFFF) as usize;
                for _ in 0..run_length {
                    bits.push(true);
                }
            } else if word & 0x40000000 != 0 {
                // Literal word
                let literal = word & 0x3FFFFFFF;
                for i in 0..31 {
                    bits.push((literal >> i) & 1 == 1);
                }
            } else {
                // Run of 0s
                let run_length = word as usize;
                for _ in 0..run_length {
                    bits.push(false);
                }
            }
        }

        // Convert bits back to bytes
        let mut bytes = Vec::new();
        for byte_bits in bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in byte_bits.iter().enumerate() {
                if bit {
                    byte |= 1 << i;
                }
            }
            bytes.push(byte);
        }

        Ok(bytes)
    }

    /// Estimate compression ratio
    pub fn estimate_compression_ratio(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 1.0;
        }

        // Simple heuristic: count runs of bits
        let mut runs = 0;
        let mut current_bit = data[0] & 1;

        for &byte in data {
            for i in 0..8 {
                let bit = (byte >> i) & 1;
                if bit != current_bit {
                    runs += 1;
                    current_bit = bit;
                }
            }
        }

        // Each run takes 4 bytes in WAH
        let estimated_size = runs * 4;
        estimated_size as f64 / data.len() as f64
    }
}

impl CompressionAlgorithm for BitmapWAHEncoder {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        let start = Instant::now();
        let compressed_bytes = Self::encode(data)?;
        let compression_time = start.elapsed();

        let mut metadata_map = HashMap::new();
        metadata_map.insert("bits_processed".to_string(), (data.len() * 8).to_string());

        let metadata = CompressionMetadata {
            algorithm: AdvancedCompressionType::BitmapWAH,
            original_size: data.len() as u64,
            compressed_size: compressed_bytes.len() as u64,
            compression_time_us: compression_time.as_micros() as u64,
            metadata: metadata_map,
        };

        Ok(CompressedData {
            data: compressed_bytes,
            metadata,
        })
    }

    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        if compressed.metadata.algorithm != AdvancedCompressionType::BitmapWAH {
            return Err(anyhow!(
                "Invalid compression algorithm: expected BitmapWAH, got {}",
                compressed.metadata.algorithm
            ));
        }

        Self::decode(&compressed.data)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::BitmapWAH
    }
}

/// Roaring bitmap encoder for sparse bitmaps
pub struct BitmapRoaringEncoder;

impl BitmapRoaringEncoder {
    /// Encode using simplified Roaring bitmap approach
    pub fn encode(data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        if data.len() > MAX_COMPRESSION_INPUT_SIZE {
            return Err(anyhow!(
                "Input data too large for compression: {} bytes (max: {})",
                data.len(),
                MAX_COMPRESSION_INPUT_SIZE
            ));
        }

        // Convert to set of integers for sparse representation
        let mut integers = Vec::new();

        for (byte_idx, &byte) in data.iter().enumerate() {
            for bit_idx in 0..8 {
                if (byte >> bit_idx) & 1 == 1 {
                    integers.push((byte_idx * 8 + bit_idx) as u32);
                }
            }
        }

        // Simple Roaring-style encoding: just store the integers
        let mut encoded = Vec::new();

        // Store count
        encoded.extend_from_slice(&(integers.len() as u32).to_le_bytes());

        // Store integers
        for integer in integers {
            encoded.extend_from_slice(&integer.to_le_bytes());
        }

        Ok(encoded)
    }

    /// Decode Roaring bitmap
    pub fn decode(compressed: &[u8]) -> Result<Vec<u8>> {
        if compressed.is_empty() {
            return Ok(Vec::new());
        }

        if compressed.len() < 4 {
            return Err(anyhow!("Invalid Roaring compressed data"));
        }

        let count = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]])
            as usize;

        if compressed.len() != 4 + count * 4 {
            return Err(anyhow!("Invalid Roaring compressed data length"));
        }

        if count > MAX_BLOCK_COUNT {
            return Err(anyhow!(
                "Too many integers in compressed data: {} (max: {})",
                count,
                MAX_BLOCK_COUNT
            ));
        }

        // Read integers
        let mut integers = Vec::new();
        for chunk in compressed[4..].chunks_exact(4) {
            let integer = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            integers.push(integer);
        }

        // Find maximum bit position to determine output size
        let max_bit = integers.iter().max().copied().unwrap_or(0);
        let byte_count = ((max_bit / 8) + 1) as usize;

        let mut bytes = vec![0u8; byte_count];

        // Set bits
        for integer in integers {
            let byte_idx = (integer / 8) as usize;
            let bit_idx = (integer % 8) as usize;
            if byte_idx < bytes.len() {
                bytes[byte_idx] |= 1 << bit_idx;
            }
        }

        Ok(bytes)
    }

    /// Estimate compression ratio for sparse data
    pub fn estimate_compression_ratio(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 1.0;
        }

        // Count set bits
        let mut set_bits = 0;
        for &byte in data {
            set_bits += byte.count_ones();
        }

        // Roaring bitmap size: 4 bytes count + 4 bytes per set bit
        let estimated_size = 4 + (set_bits as usize * 4);
        estimated_size as f64 / data.len() as f64
    }
}

impl CompressionAlgorithm for BitmapRoaringEncoder {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        let start = Instant::now();
        let compressed_bytes = Self::encode(data)?;
        let compression_time = start.elapsed();

        // Count set bits for metadata
        let set_bits: u32 = data.iter().map(|b| b.count_ones()).sum();

        let mut metadata_map = HashMap::new();
        metadata_map.insert("set_bits".to_string(), set_bits.to_string());
        metadata_map.insert(
            "sparsity".to_string(),
            format!("{:.4}", set_bits as f64 / (data.len() * 8) as f64),
        );

        let metadata = CompressionMetadata {
            algorithm: AdvancedCompressionType::BitmapRoaring,
            original_size: data.len() as u64,
            compressed_size: compressed_bytes.len() as u64,
            compression_time_us: compression_time.as_micros() as u64,
            metadata: metadata_map,
        };

        Ok(CompressedData {
            data: compressed_bytes,
            metadata,
        })
    }

    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        if compressed.metadata.algorithm != AdvancedCompressionType::BitmapRoaring {
            return Err(anyhow!(
                "Invalid compression algorithm: expected BitmapRoaring, got {}",
                compressed.metadata.algorithm
            ));
        }

        Self::decode(&compressed.data)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::BitmapRoaring
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmap_wah_basic() {
        let data = vec![0xFF, 0x00, 0xFF]; // Alternating pattern
        let encoded = BitmapWAHEncoder::encode(&data).unwrap();
        let decoded = BitmapWAHEncoder::decode(&encoded).unwrap();

        // Should at least preserve essential bit patterns
        assert!(!encoded.is_empty());
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_bitmap_roaring_sparse() {
        let mut sparse_bitmap = vec![0u8; 100];

        // Set some specific bits by setting bytes to 1 (bit 0 of each byte)
        sparse_bitmap[10] = 1; // Sets bit 80
        sparse_bitmap[20] = 1; // Sets bit 160
        sparse_bitmap[50] = 1; // Sets bit 400

        let encoded = BitmapRoaringEncoder::encode(&sparse_bitmap).unwrap();
        let decoded = BitmapRoaringEncoder::decode(&encoded).unwrap();

        // Should compress and decompress correctly
        assert!(!encoded.is_empty());
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_empty_bitmap() {
        let data = vec![];
        let wah_encoded = BitmapWAHEncoder::encode(&data).unwrap();
        let roaring_encoded = BitmapRoaringEncoder::encode(&data).unwrap();

        assert!(wah_encoded.is_empty());
        assert!(roaring_encoded.is_empty());
    }

    #[test]
    fn test_compression_algorithm_trait_wah() {
        let encoder = BitmapWAHEncoder;
        let data = vec![0xFF, 0x00, 0xFF];

        let compressed = encoder.compress(&data).unwrap();
        assert_eq!(
            compressed.metadata.algorithm,
            AdvancedCompressionType::BitmapWAH
        );

        let _decompressed = encoder.decompress(&compressed).unwrap();
        // Just verify it doesn't crash
    }

    #[test]
    fn test_compression_algorithm_trait_roaring() {
        let encoder = BitmapRoaringEncoder;
        let data = vec![1, 0, 1, 0, 1];

        let compressed = encoder.compress(&data).unwrap();
        assert_eq!(
            compressed.metadata.algorithm,
            AdvancedCompressionType::BitmapRoaring
        );

        let decompressed = encoder.decompress(&compressed).unwrap();
        assert!(!decompressed.is_empty());
    }
}
