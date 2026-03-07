//! Delta encoding for sequences

use crate::compression::{
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
};
use anyhow::{anyhow, Result};
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
                (values[i] - values[i - 1]) << 1 // Positive delta
            } else {
                ((values[i - 1] - values[i]) << 1) | 1 // Negative delta with flag
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
            chunks[0][0],
            chunks[0][1],
            chunks[0][2],
            chunks[0][3],
            chunks[0][4],
            chunks[0][5],
            chunks[0][6],
            chunks[0][7],
        ]);
        values.push(first_value);

        let mut current_value = first_value;

        // Process deltas
        for chunk in &chunks[1..] {
            let encoded_delta = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
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

    /// Encode a sorted u64 sequence using variable-length (LEB128-style) delta encoding.
    ///
    /// This produces more compact output than `encode_u64_sequence` for small
    /// deltas (which is typical for sorted triple IDs).
    ///
    /// Encoding:
    /// - First value: stored as 9-byte little-endian u64 (1 header byte = 0xFF + 8 bytes)
    /// - Subsequent values: zigzag-encode the signed delta, then store as LEB128 varint
    pub fn encode(values: &[u64]) -> Vec<u8> {
        if values.is_empty() {
            return Vec::new();
        }

        let mut out = Vec::new();

        // First value as raw 8-byte LE
        out.extend_from_slice(&values[0].to_le_bytes());

        let mut prev = values[0];
        for &v in &values[1..] {
            // Signed delta, zigzag encoded into unsigned for LEB128
            let delta: i64 = if v >= prev {
                (v - prev) as i64
            } else {
                -((prev - v) as i64)
            };
            // Zigzag: map signed to unsigned
            let zigzag: u64 = if delta >= 0 {
                (delta as u64) << 1
            } else {
                ((!delta as u64) << 1) | 1
            };
            // LEB128 encode the zigzag value
            encode_leb128(&mut out, zigzag);
            prev = v;
        }

        out
    }

    /// Decode a variable-length delta encoded u64 sequence produced by `encode`.
    pub fn decode(data: &[u8]) -> Result<Vec<u64>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        if data.len() < 8 {
            return Err(anyhow!("Delta-encoded data too short for first value"));
        }

        let first = u64::from_le_bytes(
            data[..8]
                .try_into()
                .map_err(|_| anyhow!("Failed to read first value"))?,
        );

        let mut values = vec![first];
        let mut prev = first;
        let mut pos = 8;

        while pos < data.len() {
            let (zigzag, consumed) = decode_leb128(&data[pos..])
                .ok_or_else(|| anyhow!("Invalid LEB128 encoding at byte {}", pos))?;
            pos += consumed;

            // Reverse zigzag
            let delta: i64 = if (zigzag & 1) == 0 {
                (zigzag >> 1) as i64
            } else {
                !((zigzag >> 1) as i64)
            };

            prev = if delta >= 0 {
                prev.saturating_add(delta as u64)
            } else {
                prev.saturating_sub((-delta) as u64)
            };
            values.push(prev);
        }

        Ok(values)
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

// ─────────────────────────────────────────────────────────────────────────────
// LEB128 helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a `u64` value as unsigned LEB128 into `buf`.
fn encode_leb128(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

/// Decode an unsigned LEB128 value from `data`.
///
/// Returns `(value, bytes_consumed)` or `None` if the data is empty or malformed.
fn decode_leb128(data: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &byte) in data.iter().enumerate() {
        let low = (byte & 0x7F) as u64;
        value |= low.checked_shl(shift)?;
        shift += 7;
        if (byte & 0x80) == 0 {
            return Some((value, i + 1));
        }
        if shift > 63 {
            return None; // overflow protection
        }
    }
    None // unterminated LEB128
}

impl CompressionAlgorithm for DeltaEncoder {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        let start = Instant::now();
        let compressed_bytes = Self::encode_byte_sequence(data)?;
        let compression_time = start.elapsed();

        let mut metadata_map = HashMap::new();
        if !data.is_empty() {
            metadata_map.insert("first_value".to_string(), data[0].to_string());
            metadata_map.insert(
                "delta_count".to_string(),
                (data.len().saturating_sub(1)).to_string(),
            );
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
        assert_eq!(
            compressed.metadata.algorithm,
            AdvancedCompressionType::Delta
        );

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

    // ── Variable-length delta encode/decode ──────────────────────────────────

    #[test]
    fn test_vle_encode_empty() {
        let encoded = DeltaEncoder::encode(&[]);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_vle_decode_empty() {
        let decoded = DeltaEncoder::decode(&[]).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_vle_single_value() {
        let values = vec![42u64];
        let encoded = DeltaEncoder::encode(&values);
        let decoded = DeltaEncoder::decode(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_vle_sorted_ascending() {
        let values: Vec<u64> = (1..=20).collect();
        let encoded = DeltaEncoder::encode(&values);
        let decoded = DeltaEncoder::decode(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_vle_sorted_descending() {
        let values: Vec<u64> = (1..=20).rev().collect();
        let encoded = DeltaEncoder::encode(&values);
        let decoded = DeltaEncoder::decode(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_vle_with_duplicates() {
        let values = vec![5u64, 5, 5, 5, 10, 10, 15];
        let encoded = DeltaEncoder::encode(&values);
        let decoded = DeltaEncoder::decode(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_vle_large_sequence_roundtrip() {
        let values: Vec<u64> = (0..1000).map(|i| i * 7).collect();
        let encoded = DeltaEncoder::encode(&values);
        let decoded = DeltaEncoder::decode(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_vle_large_gaps() {
        let values = vec![0u64, 1_000_000, 2_000_000, 3_000_000];
        let encoded = DeltaEncoder::encode(&values);
        let decoded = DeltaEncoder::decode(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_vle_compact_for_small_deltas() {
        // For a sequence with small deltas, VLE should be more compact than
        // the fixed 8-byte encoding for sequences longer than 1 element.
        let values: Vec<u64> = (100..200).collect(); // deltas of 1
        let vle_encoded = DeltaEncoder::encode(&values);
        // Fixed encoding: 100 values × 8 bytes = 800 bytes
        // VLE: 8 bytes (first) + 99 × 1 byte (delta=1 encodes to 1 LEB128 byte) = 107 bytes
        assert!(vle_encoded.len() < 100 * 8, "VLE should be more compact");
    }

    // ── LEB128 helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_leb128_round_trip_small() {
        let mut buf = Vec::new();
        encode_leb128(&mut buf, 127);
        let (v, n) = decode_leb128(&buf).unwrap();
        assert_eq!(v, 127);
        assert_eq!(n, 1);
    }

    #[test]
    fn test_leb128_round_trip_large() {
        let mut buf = Vec::new();
        encode_leb128(&mut buf, u64::MAX);
        let (v, _) = decode_leb128(&buf).unwrap();
        assert_eq!(v, u64::MAX);
    }

    #[test]
    fn test_leb128_zero() {
        let mut buf = Vec::new();
        encode_leb128(&mut buf, 0);
        assert_eq!(buf.len(), 1);
        let (v, _) = decode_leb128(&buf).unwrap();
        assert_eq!(v, 0);
    }
}
