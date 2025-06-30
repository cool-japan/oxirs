//! Run-Length Encoding implementation for TDB storage

use anyhow::{anyhow, Result};
use crate::compression::{
    AdvancedCompressionType, CompressionAlgorithm, CompressionMetadata, CompressedData,
    MAX_COMPRESSION_INPUT_SIZE, MAX_BLOCK_COUNT
};
use std::collections::HashMap;
use std::time::Instant;

/// Run-Length Encoding implementation
pub struct RunLengthEncoder;

impl RunLengthEncoder {
    /// Encode data using run-length encoding
    pub fn encode(data: &[u8]) -> Result<Vec<u8>> {
        // Production hardening: Input validation
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
        // Production hardening: Input validation
        if encoded.is_empty() {
            return Ok(Vec::new());
        }

        if encoded.len() % 5 != 0 {
            return Err(anyhow!(
                "Invalid run-length encoded data length: {} bytes",
                encoded.len()
            ));
        }

        if encoded.len() > MAX_COMPRESSION_INPUT_SIZE {
            return Err(anyhow!(
                "Encoded data too large for decompression: {} bytes (max: {})",
                encoded.len(),
                MAX_COMPRESSION_INPUT_SIZE
            ));
        }

        let run_count = encoded.len() / 5;
        if run_count > MAX_BLOCK_COUNT {
            return Err(anyhow!(
                "Too many runs in encoded data: {} (max: {})",
                run_count,
                MAX_BLOCK_COUNT
            ));
        }

        let mut decoded = Vec::new();
        let mut total_output_size = 0u64;

        for chunk in encoded.chunks_exact(5) {
            let count = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let byte_value = chunk[4];

            // Production hardening: Prevent decompression bombs
            total_output_size += count as u64;
            if total_output_size > MAX_COMPRESSION_INPUT_SIZE as u64 {
                return Err(anyhow!(
                    "Decompressed data would exceed size limit: {} bytes (max: {})",
                    total_output_size,
                    MAX_COMPRESSION_INPUT_SIZE
                ));
            }

            // Production hardening: Reasonable run length check
            if count > 10_000_000 {
                return Err(anyhow!(
                    "Suspicious run length detected: {} (possible decompression bomb)",
                    count
                ));
            }

            for _ in 0..count {
                decoded.push(byte_value);
            }
        }

        Ok(decoded)
    }

    /// Estimate compression ratio for given data
    pub fn estimate_compression_ratio(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 1.0;
        }

        let mut runs = 0;
        let mut current_byte = data[0];
        for &byte in &data[1..] {
            if byte != current_byte {
                runs += 1;
                current_byte = byte;
            }
        }
        runs += 1; // Count final run

        let compressed_size = runs * 5; // 4 bytes count + 1 byte value
        compressed_size as f64 / data.len() as f64
    }
}

impl CompressionAlgorithm for RunLengthEncoder {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        let start = Instant::now();
        let compressed_bytes = Self::encode(data)?;
        let compression_time = start.elapsed();

        let mut metadata_map = HashMap::new();
        metadata_map.insert("runs".to_string(), (compressed_bytes.len() / 5).to_string());

        let metadata = CompressionMetadata {
            algorithm: AdvancedCompressionType::RunLength,
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
        if compressed.metadata.algorithm != AdvancedCompressionType::RunLength {
            return Err(anyhow!(
                "Invalid compression algorithm: expected RunLength, got {}",
                compressed.metadata.algorithm
            ));
        }

        Self::decode(&compressed.data)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::RunLength
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
    fn test_empty_data() {
        let data = vec![];
        let encoded = RunLengthEncoder::encode(&data).unwrap();
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_single_byte() {
        let data = vec![42];
        let encoded = RunLengthEncoder::encode(&data).unwrap();
        let decoded = RunLengthEncoder::decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_compression_algorithm_trait() {
        let encoder = RunLengthEncoder;
        let data = vec![1, 1, 1, 2, 2, 3, 3, 3, 3];
        
        let compressed = encoder.compress(&data).unwrap();
        assert_eq!(compressed.metadata.algorithm, AdvancedCompressionType::RunLength);
        assert_eq!(compressed.metadata.original_size, data.len() as u64);
        
        let decompressed = encoder.decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_estimate_compression_ratio() {
        let highly_repetitive = vec![1; 1000];
        let ratio = RunLengthEncoder::estimate_compression_ratio(&highly_repetitive);
        assert!(ratio < 0.1); // Should compress very well

        let random_data = (0..100).collect::<Vec<u8>>();
        let ratio = RunLengthEncoder::estimate_compression_ratio(&random_data);
        assert!(ratio > 1.0); // Should expand
    }
}