//! Frame of Reference (FOR) encoding implementation

use crate::compression::{
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::time::Instant;

/// Frame of Reference encoding implementation
pub struct FrameOfReferenceEncoder;

impl FrameOfReferenceEncoder {
    /// Encode sequence using Frame of Reference
    pub fn encode_u64_sequence(values: &[u64]) -> Result<Vec<u8>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let mut encoded = Vec::new();

        // Find minimum value as reference
        let reference = *values.iter().min().unwrap();
        encoded.extend_from_slice(&reference.to_le_bytes());

        // Find maximum delta to determine bit width
        let max_delta = values.iter().map(|&v| v - reference).max().unwrap();
        let bit_width = if max_delta == 0 {
            1
        } else {
            64 - max_delta.leading_zeros()
        };

        // Store bit width
        encoded.push(bit_width as u8);

        // Encode deltas with minimal bit width
        let mut bit_buffer = 0u64;
        let mut bits_used = 0;

        for &value in values {
            let delta = value - reference;

            // Add delta to bit buffer
            bit_buffer |= delta << bits_used;
            bits_used += bit_width;

            // Flush complete bytes
            while bits_used >= 8 {
                encoded.push(bit_buffer as u8);
                bit_buffer >>= 8;
                bits_used -= 8;
            }
        }

        // Flush remaining bits
        if bits_used > 0 {
            encoded.push(bit_buffer as u8);
        }

        Ok(encoded)
    }

    /// Decode Frame of Reference encoded sequence
    pub fn decode_u64_sequence(encoded: &[u8]) -> Result<Vec<u64>> {
        if encoded.is_empty() {
            return Ok(Vec::new());
        }

        if encoded.len() < 9 {
            return Err(anyhow!("Invalid FOR encoded data: too short"));
        }

        // Read reference value
        let reference = u64::from_le_bytes([
            encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6],
            encoded[7],
        ]);

        // Read bit width
        let bit_width = encoded[8] as u32;
        if bit_width == 0 || bit_width > 64 {
            return Err(anyhow!("Invalid bit width: {}", bit_width));
        }

        let mut values = Vec::new();
        let data = &encoded[9..];

        let mut bit_buffer = 0u64;
        let mut bits_available = 0;
        let mut byte_index = 0;

        // Calculate mask for extracting values
        let mask = (1u64 << bit_width) - 1;

        while byte_index < data.len() {
            // Load more bytes into buffer as needed
            while bits_available < bit_width && byte_index < data.len() {
                bit_buffer |= (data[byte_index] as u64) << bits_available;
                bits_available += 8;
                byte_index += 1;
            }

            if bits_available >= bit_width {
                // Extract value
                let delta = bit_buffer & mask;
                values.push(reference + delta);

                // Remove used bits
                bit_buffer >>= bit_width;
                bits_available -= bit_width;
            } else {
                break;
            }
        }

        Ok(values)
    }

    /// Estimate compression ratio
    pub fn estimate_compression_ratio(values: &[u64]) -> f64 {
        if values.len() < 2 {
            return 1.0;
        }

        let min_val = *values.iter().min().unwrap();
        let max_val = *values.iter().max().unwrap();
        let range = max_val - min_val;

        let bits_needed = if range == 0 {
            1
        } else {
            64 - range.leading_zeros()
        };

        // Estimate: reference (8 bytes) + bit_width (1 byte) + packed data
        let estimated_size = 9 + (values.len() * bits_needed as usize + 7) / 8;
        estimated_size as f64 / (values.len() * 8) as f64
    }
}

impl CompressionAlgorithm for FrameOfReferenceEncoder {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        // Convert bytes to u64 sequence (simplified approach)
        if data.len() % 8 != 0 {
            return Err(anyhow!(
                "Data length must be multiple of 8 for FOR encoding"
            ));
        }

        let values: Vec<u64> = data
            .chunks_exact(8)
            .map(|chunk| {
                u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();

        let start = Instant::now();
        let compressed_bytes = Self::encode_u64_sequence(&values)?;
        let compression_time = start.elapsed();

        let mut metadata_map = HashMap::new();
        if !values.is_empty() {
            let reference = *values.iter().min().unwrap();
            let max_delta = values.iter().map(|&v| v - reference).max().unwrap();
            let bit_width = if max_delta == 0 {
                1
            } else {
                64 - max_delta.leading_zeros()
            };

            metadata_map.insert("reference".to_string(), reference.to_string());
            metadata_map.insert("bit_width".to_string(), bit_width.to_string());
            metadata_map.insert("value_count".to_string(), values.len().to_string());
        }

        let metadata = CompressionMetadata {
            algorithm: AdvancedCompressionType::FrameOfReference,
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
        if compressed.metadata.algorithm != AdvancedCompressionType::FrameOfReference {
            return Err(anyhow!(
                "Invalid compression algorithm: expected FrameOfReference, got {}",
                compressed.metadata.algorithm
            ));
        }

        let values = Self::decode_u64_sequence(&compressed.data)?;

        // Convert u64 sequence back to bytes
        let mut data = Vec::new();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }

        Ok(data)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::FrameOfReference
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_of_reference_encoding() {
        let values = vec![1000, 1002, 1005, 1003, 1010];
        let encoded = FrameOfReferenceEncoder::encode_u64_sequence(&values).unwrap();
        let decoded = FrameOfReferenceEncoder::decode_u64_sequence(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_constant_sequence() {
        let values = vec![42; 10];
        let encoded = FrameOfReferenceEncoder::encode_u64_sequence(&values).unwrap();
        let decoded = FrameOfReferenceEncoder::decode_u64_sequence(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_empty_sequence() {
        let values = vec![];
        let encoded = FrameOfReferenceEncoder::encode_u64_sequence(&values).unwrap();
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_single_value() {
        let values = vec![12345];
        let encoded = FrameOfReferenceEncoder::encode_u64_sequence(&values).unwrap();
        let decoded = FrameOfReferenceEncoder::decode_u64_sequence(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_large_range() {
        let values = vec![0, 1000000, 2000000];
        let encoded = FrameOfReferenceEncoder::encode_u64_sequence(&values).unwrap();
        let decoded = FrameOfReferenceEncoder::decode_u64_sequence(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_compression_algorithm_trait() {
        let encoder = FrameOfReferenceEncoder;
        // Create test data as bytes (multiple of 8)
        let data = vec![0u8; 24]; // 3 u64 values

        let compressed = encoder.compress(&data).unwrap();
        assert_eq!(
            compressed.metadata.algorithm,
            AdvancedCompressionType::FrameOfReference
        );

        let decompressed = encoder.decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }
}
