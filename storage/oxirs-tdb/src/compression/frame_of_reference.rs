//! Frame of Reference (FOR) compression for integer sequences
//!
//! Frame of Reference compression is designed for sequences of integers with small deltas.
//! It stores a base value (reference frame) and encodes deltas from that base.

use super::{
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
    MAX_COMPRESSION_INPUT_SIZE,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::time::Instant;

/// Frame of Reference encoder for integer sequences
#[derive(Debug, Clone)]
pub struct FrameOfReferenceEncoder {
    /// Number of bits per delta value
    pub bits_per_delta: u8,
}

impl Default for FrameOfReferenceEncoder {
    fn default() -> Self {
        Self {
            bits_per_delta: 8, // Default to 8 bits per delta
        }
    }
}

impl FrameOfReferenceEncoder {
    /// Create new FOR encoder with specified bits per delta
    pub fn new(bits_per_delta: u8) -> Self {
        Self {
            bits_per_delta: bits_per_delta.min(32), // Cap at 32 bits
        }
    }

    /// Compress sequence of u32 integers
    pub fn compress_u32(&self, values: &[u32]) -> Result<CompressedData> {
        if values.is_empty() {
            return Ok(CompressedData {
                data: vec![],
                metadata: CompressionMetadata {
                    algorithm: AdvancedCompressionType::FrameOfReference,
                    original_size: 0,
                    compressed_size: 0,
                    compression_time_us: 0,
                    metadata: HashMap::new(),
                },
            });
        }

        let start_time = Instant::now();

        // Find base value (minimum)
        let base = *values.iter().min().unwrap();

        // Calculate deltas
        let deltas: Vec<u32> = values.iter().map(|&v| v - base).collect();

        // Find maximum delta to determine required bits
        let max_delta = *deltas.iter().max().unwrap();
        let required_bits = if max_delta == 0 {
            1
        } else {
            32 - max_delta.leading_zeros() as u8
        };

        // Use smaller of required bits or configured bits
        let bits_per_delta = required_bits.min(self.bits_per_delta);

        // Encode: [base (4 bytes)] [bits_per_delta (1 byte)] [count (4 bytes)] [packed deltas]
        let mut result = Vec::new();
        result.extend_from_slice(&base.to_le_bytes());
        result.push(bits_per_delta);
        result.extend_from_slice(&(values.len() as u32).to_le_bytes());

        // Pack deltas
        if bits_per_delta > 0 {
            let packed_deltas = self.pack_deltas(&deltas, bits_per_delta)?;
            result.extend_from_slice(&packed_deltas);
        }

        let compression_time = start_time.elapsed().as_micros() as u64;

        let mut metadata_map = HashMap::new();
        metadata_map.insert("base".to_string(), base.to_string());
        metadata_map.insert("bits_per_delta".to_string(), bits_per_delta.to_string());
        metadata_map.insert("count".to_string(), values.len().to_string());

        let compressed_size = result.len() as u64;
        Ok(CompressedData {
            data: result,
            metadata: CompressionMetadata {
                algorithm: AdvancedCompressionType::FrameOfReference,
                original_size: (values.len() * 4) as u64,
                compressed_size,
                compression_time_us: compression_time,
                metadata: metadata_map,
            },
        })
    }

    /// Pack deltas using specified bits per value
    fn pack_deltas(&self, deltas: &[u32], bits_per_delta: u8) -> Result<Vec<u8>> {
        if bits_per_delta == 0 {
            return Ok(vec![]);
        }

        let mut result = Vec::new();
        let mut current_byte = 0u8;
        let mut bits_in_current_byte = 0u8;

        for &delta in deltas {
            // Ensure delta fits in specified bits
            let masked_delta = delta & ((1u32 << bits_per_delta) - 1);

            let mut remaining_bits = bits_per_delta;
            let mut value = masked_delta;

            while remaining_bits > 0 {
                let bits_to_write = remaining_bits.min(8 - bits_in_current_byte);
                let shift = remaining_bits - bits_to_write;
                let bits_value = ((value >> shift) & ((1u32 << bits_to_write) - 1)) as u8;

                current_byte |= bits_value << (8 - bits_in_current_byte - bits_to_write);
                bits_in_current_byte += bits_to_write;
                remaining_bits -= bits_to_write;
                value &= (1u32 << shift) - 1;

                if bits_in_current_byte == 8 {
                    result.push(current_byte);
                    current_byte = 0;
                    bits_in_current_byte = 0;
                }
            }
        }

        // Add remaining partial byte
        if bits_in_current_byte > 0 {
            result.push(current_byte);
        }

        Ok(result)
    }

    /// Decompress sequence of u32 integers
    pub fn decompress_u32(&self, data: &[u8]) -> Result<Vec<u32>> {
        if data.is_empty() {
            return Ok(vec![]);
        }

        if data.len() < 9 {
            return Err(anyhow!("Invalid FOR compressed data: too short"));
        }

        // Read header
        let base = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let bits_per_delta = data[4];
        let count = u32::from_le_bytes([data[5], data[6], data[7], data[8]]) as usize;

        if count == 0 {
            return Ok(vec![]);
        }

        // Unpack deltas
        let deltas = if bits_per_delta == 0 {
            vec![0; count]
        } else {
            self.unpack_deltas(&data[9..], bits_per_delta, count)?
        };

        // Reconstruct original values
        let values: Vec<u32> = deltas.iter().map(|&delta| base + delta).collect();

        Ok(values)
    }

    /// Unpack deltas from bit-packed data
    fn unpack_deltas(&self, data: &[u8], bits_per_delta: u8, count: usize) -> Result<Vec<u32>> {
        let mut result = Vec::with_capacity(count);
        let mut byte_index = 0;
        let mut bit_index = 0u8;

        for _ in 0..count {
            let mut value = 0u32;
            let mut remaining_bits = bits_per_delta;

            while remaining_bits > 0 && byte_index < data.len() {
                let bits_available = 8 - bit_index;
                let bits_to_read = remaining_bits.min(bits_available);

                let byte_value = data[byte_index];
                let shift = bits_available - bits_to_read;
                let mask = if bits_to_read >= 8 {
                    0xFFu8
                } else {
                    ((1u8 << bits_to_read) - 1) << shift
                };
                let extracted_bits = (byte_value & mask) >> shift;

                value = (value << bits_to_read) | (extracted_bits as u32);
                remaining_bits -= bits_to_read;
                bit_index += bits_to_read;

                if bit_index == 8 {
                    byte_index += 1;
                    bit_index = 0;
                }
            }

            result.push(value);
        }

        Ok(result)
    }
}

impl CompressionAlgorithm for FrameOfReferenceEncoder {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        if data.len() > MAX_COMPRESSION_INPUT_SIZE {
            return Err(anyhow!("Input too large for compression"));
        }

        // Convert bytes to u32 sequence (assuming 4-byte alignment)
        if data.len() % 4 != 0 {
            return Err(anyhow!(
                "Input data must be 4-byte aligned for FOR compression"
            ));
        }

        let values: Vec<u32> = data
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        self.compress_u32(&values)
    }

    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        let values = self.decompress_u32(&compressed.data)?;

        let mut result = Vec::with_capacity(values.len() * 4);
        for value in values {
            result.extend_from_slice(&value.to_le_bytes());
        }

        Ok(result)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::FrameOfReference
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_algorithm_trait() {
        let encoder = FrameOfReferenceEncoder::default();

        // Test with sequence: [100, 101, 102, 103] as bytes
        let values = vec![100u32, 101u32, 102u32, 103u32];
        let mut data = Vec::new();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }

        let compressed = encoder.compress(&data).unwrap();
        assert_eq!(
            compressed.metadata.algorithm,
            AdvancedCompressionType::FrameOfReference
        );

        let decompressed = encoder.decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_constant_sequence() {
        let encoder = FrameOfReferenceEncoder::default();
        let values = vec![42u32; 100];

        let compressed = encoder.compress_u32(&values).unwrap();
        let decompressed = encoder.decompress_u32(&compressed.data).unwrap();

        assert_eq!(values, decompressed);
        // Should achieve good compression for constant sequence
        assert!(compressed.metadata.compressed_size < compressed.metadata.original_size);
    }

    #[test]
    fn test_empty_sequence() {
        let encoder = FrameOfReferenceEncoder::default();
        let values = vec![];

        let compressed = encoder.compress_u32(&values).unwrap();
        let decompressed = encoder.decompress_u32(&compressed.data).unwrap();

        assert_eq!(values, decompressed);
        assert_eq!(compressed.metadata.compressed_size, 0);
    }

    #[test]
    fn test_small_delta_sequence() {
        let encoder = FrameOfReferenceEncoder::new(4); // 4 bits per delta
        let values = vec![1000, 1001, 1002, 1003, 1004, 1005];

        let compressed = encoder.compress_u32(&values).unwrap();
        let decompressed = encoder.decompress_u32(&compressed.data).unwrap();

        assert_eq!(values, decompressed);

        // Verify metadata
        assert_eq!(compressed.metadata.metadata.get("base").unwrap(), "1000");
        assert_eq!(compressed.metadata.metadata.get("count").unwrap(), "6");
    }

    #[test]
    fn test_large_delta_sequence() {
        let encoder = FrameOfReferenceEncoder::new(16); // 16 bits per delta
        let values = vec![1000, 5000, 10000, 15000];

        let compressed = encoder.compress_u32(&values).unwrap();
        let decompressed = encoder.decompress_u32(&compressed.data).unwrap();

        assert_eq!(values, decompressed);
    }
}
