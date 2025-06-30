//! Delta encoding for sequences

use anyhow::{anyhow, Result};
use crate::compression::{
    AdvancedCompressionType, CompressionAlgorithm, CompressionMetadata, CompressedData
};
use std::collections::HashMap;
use std::time::Instant;

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
            let delta = if values[i] >= values[i - 1] {
                ((values[i] - values[i - 1]) << 1) // Positive delta
            } else {
                (((values[i - 1] - values[i]) << 1) | 1) // Negative delta with flag
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

        let mut current_value = first_value;

        // Process deltas
        for chunk in &chunks[1..] {
            let encoded_delta = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7],
            ]);

            let is_negative = (encoded_delta & 1) == 1;
            let delta = encoded_delta >> 1;

            current_value = if is_negative {
                current_value.saturating_sub(delta)
            } else {
                current_value.saturating_add(delta)
            };

            values.push(current_value);
        }

        Ok(values)
    }

    /// Encode byte sequence using delta encoding
    pub fn encode_byte_sequence(data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut encoded = Vec::new();
        encoded.push(data[0]);

        for i in 1..data.len() {
            let delta = data[i] as i16 - data[i - 1] as i16;
            encoded.extend_from_slice(&delta.to_le_bytes());
        }

        Ok(encoded)
    }

    /// Decode byte sequence using delta encoding
    pub fn decode_byte_sequence(encoded: &[u8]) -> Result<Vec<u8>> {
        if encoded.is_empty() {
            return Ok(Vec::new());
        }

        let mut decoded = Vec::new();
        decoded.push(encoded[0]);

        let mut current_value = encoded[0] as i16;

        for chunk in encoded[1..].chunks_exact(2) {
            let delta = i16::from_le_bytes([chunk[0], chunk[1]]);
            current_value = (current_value + delta).clamp(0, 255);
            decoded.push(current_value as u8);
        }

        Ok(decoded)
    }

    /// Estimate compression ratio for byte sequences
    pub fn estimate_byte_compression_ratio(data: &[u8]) -> f64 {
        if data.len() < 2 {
            return 1.0;
        }

        let compressed_size = 1 + (data.len() - 1) * 2; // First byte + deltas
        compressed_size as f64 / data.len() as f64
    }

    /// Estimate compression ratio for u64 sequences
    pub fn estimate_u64_compression_ratio(values: &[u64]) -> f64 {
        if values.len() < 2 {
            return 1.0;
        }

        // Estimate based on typical delta sizes
        let mut small_deltas = 0;
        for i in 1..values.len() {
            let delta = values[i].abs_diff(values[i - 1]);
            if delta < 65536 {
                small_deltas += 1;
            }
        }

        // Conservative estimate - in practice delta encoding can be much better
        // for sorted sequences
        if small_deltas > values.len() / 2 {
            0.5 // Good compression expected
        } else {
            1.0 // No compression expected
        }
    }
}

impl CompressionAlgorithm for DeltaEncoder {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        let start = Instant::now();
        let compressed_bytes = Self::encode_byte_sequence(data)?;
        let compression_time = start.elapsed();

        let mut metadata_map = HashMap::new();
        if !data.is_empty() {
            metadata_map.insert("first_value".to_string(), data[0].to_string());
            metadata_map.insert("delta_count".to_string(), (data.len().saturating_sub(1)).to_string());
        }

        let metadata = CompressionMetadata {
            algorithm: AdvancedCompressionType::Delta,
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
        if compressed.metadata.algorithm != AdvancedCompressionType::Delta {
            return Err(anyhow!(
                "Invalid compression algorithm: expected Delta, got {}",
                compressed.metadata.algorithm
            ));
        }

        Self::decode_byte_sequence(&compressed.data)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::Delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64_delta_encoding() {
        let values = vec![100, 102, 105, 103, 110];
        let encoded = DeltaEncoder::encode_u64_sequence(&values).unwrap();
        let decoded = DeltaEncoder::decode_u64_sequence(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_byte_delta_encoding() {
        let data = vec![50, 52, 55, 53, 60];
        let encoded = DeltaEncoder::encode_byte_sequence(&data).unwrap();
        let decoded = DeltaEncoder::decode_byte_sequence(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_empty_sequence() {
        let data = vec![];
        let encoded = DeltaEncoder::encode_byte_sequence(&data).unwrap();
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_single_value() {
        let data = vec![42];
        let encoded = DeltaEncoder::encode_byte_sequence(&data).unwrap();
        let decoded = DeltaEncoder::decode_byte_sequence(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_compression_algorithm_trait() {
        let encoder = DeltaEncoder;
        let data = vec![10, 12, 15, 13, 20];
        
        let compressed = encoder.compress(&data).unwrap();
        assert_eq!(compressed.metadata.algorithm, AdvancedCompressionType::Delta);
        
        let decompressed = encoder.decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_sorted_sequence() {
        let values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let encoded = DeltaEncoder::encode_u64_sequence(&values).unwrap();
        let decoded = DeltaEncoder::decode_u64_sequence(&encoded).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_decreasing_sequence() {
        let values = vec![100, 90, 80, 70, 60];
        let encoded = DeltaEncoder::encode_u64_sequence(&values).unwrap();
        let decoded = DeltaEncoder::decode_u64_sequence(&encoded).unwrap();
        assert_eq!(values, decoded);
    }
}