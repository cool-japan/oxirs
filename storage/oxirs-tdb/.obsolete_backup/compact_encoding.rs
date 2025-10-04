//! # Compact Binary Encoding for OxiRS TDB
//!
//! Advanced compact binary encoding schemes for Node IDs, triples, and quads
//! with variable-length encoding, delta compression, and bit packing optimization.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::nodes::NodeId;
use crate::triple_store::{Quad, Triple};

/// Compact encoding schemes for different data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactEncodingScheme {
    /// Variable-length encoding (LEB128)
    VarInt,
    /// Delta encoding for sequences
    Delta,
    /// Bit packing for small values
    BitPacked,
    /// Frame of reference for clustered values
    FrameOfReference,
    /// Hybrid approach combining multiple schemes
    Hybrid,
}

/// Compact encoder for Node IDs and related data structures
pub struct CompactEncoder {
    scheme: CompactEncodingScheme,
    /// Statistics for adaptive encoding decisions
    stats: EncodingStats,
}

/// Statistics for encoding efficiency tracking
#[derive(Debug, Clone, Default)]
pub struct EncodingStats {
    pub total_encoded: u64,
    pub total_bytes_before: u64,
    pub total_bytes_after: u64,
    pub varint_usage: u64,
    pub delta_usage: u64,
    pub bitpack_usage: u64,
    pub frame_ref_usage: u64,
}

impl EncodingStats {
    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.total_bytes_before == 0 {
            1.0
        } else {
            self.total_bytes_after as f64 / self.total_bytes_before as f64
        }
    }

    /// Get the most effective encoding scheme based on statistics
    pub fn best_scheme(&self) -> CompactEncodingScheme {
        // Simple heuristic: use the scheme that's been most successful
        let schemes = [
            (self.varint_usage, CompactEncodingScheme::VarInt),
            (self.delta_usage, CompactEncodingScheme::Delta),
            (self.bitpack_usage, CompactEncodingScheme::BitPacked),
            (
                self.frame_ref_usage,
                CompactEncodingScheme::FrameOfReference,
            ),
        ];

        schemes
            .iter()
            .max_by_key(|(usage, _)| *usage)
            .map(|(_, scheme)| *scheme)
            .unwrap_or(CompactEncodingScheme::VarInt)
    }
}

impl CompactEncoder {
    /// Create a new compact encoder with the specified scheme
    pub fn new(scheme: CompactEncodingScheme) -> Self {
        Self {
            scheme,
            stats: EncodingStats::default(),
        }
    }

    /// Create an adaptive encoder that selects the best scheme automatically
    pub fn adaptive() -> Self {
        Self::new(CompactEncodingScheme::Hybrid)
    }

    /// Encode a single Node ID
    pub fn encode_node_id(&mut self, node_id: NodeId) -> Vec<u8> {
        match self.scheme {
            CompactEncodingScheme::VarInt => {
                self.stats.varint_usage += 1;
                self.encode_varint(node_id)
            }
            CompactEncodingScheme::BitPacked => {
                self.stats.bitpack_usage += 1;
                // For single values, bit packing isn't beneficial
                self.encode_varint(node_id)
            }
            _ => {
                // Default to varint for single values
                self.stats.varint_usage += 1;
                self.encode_varint(node_id)
            }
        }
    }

    /// Encode a sequence of Node IDs with delta compression
    pub fn encode_node_id_sequence(&mut self, node_ids: &[NodeId]) -> Vec<u8> {
        if node_ids.is_empty() {
            return Vec::new();
        }

        let original_size = node_ids.len() * 8; // 8 bytes per u64

        let encoded = match self.scheme {
            CompactEncodingScheme::VarInt => {
                self.stats.varint_usage += 1;
                self.encode_node_ids_varint(node_ids)
            }
            CompactEncodingScheme::Delta => {
                self.stats.delta_usage += 1;
                self.encode_node_ids_delta(node_ids)
            }
            CompactEncodingScheme::BitPacked => {
                self.stats.bitpack_usage += 1;
                self.encode_node_ids_bitpacked(node_ids)
            }
            CompactEncodingScheme::FrameOfReference => {
                self.stats.frame_ref_usage += 1;
                self.encode_node_ids_frame_of_reference(node_ids)
            }
            CompactEncodingScheme::Hybrid => {
                // Choose the best encoding based on data characteristics
                self.encode_node_ids_adaptive(node_ids)
            }
        };

        // Update statistics
        self.stats.total_encoded += node_ids.len() as u64;
        self.stats.total_bytes_before += original_size as u64;
        self.stats.total_bytes_after += encoded.len() as u64;

        encoded
    }

    /// Encode a triple with compact representation
    pub fn encode_triple(&mut self, triple: &Triple) -> Vec<u8> {
        let mut result = Vec::new();

        // Encode each component
        let subject_bytes = self.encode_node_id(triple.subject);
        let predicate_bytes = self.encode_node_id(triple.predicate);
        let object_bytes = self.encode_node_id(triple.object);

        // Pack into result with length prefixes for each component
        self.encode_varint_to_vec(&mut result, subject_bytes.len() as u64);
        result.extend_from_slice(&subject_bytes);

        self.encode_varint_to_vec(&mut result, predicate_bytes.len() as u64);
        result.extend_from_slice(&predicate_bytes);

        self.encode_varint_to_vec(&mut result, object_bytes.len() as u64);
        result.extend_from_slice(&object_bytes);

        result
    }

    /// Encode a quad with compact representation
    pub fn encode_quad(&mut self, quad: &Quad) -> Vec<u8> {
        let mut result = Vec::new();

        // Encode each component
        let subject_bytes = self.encode_node_id(quad.subject);
        let predicate_bytes = self.encode_node_id(quad.predicate);
        let object_bytes = self.encode_node_id(quad.object);

        // Pack into result with length prefixes
        self.encode_varint_to_vec(&mut result, subject_bytes.len() as u64);
        result.extend_from_slice(&subject_bytes);

        self.encode_varint_to_vec(&mut result, predicate_bytes.len() as u64);
        result.extend_from_slice(&predicate_bytes);

        self.encode_varint_to_vec(&mut result, object_bytes.len() as u64);
        result.extend_from_slice(&object_bytes);

        // Encode optional graph (None is encoded as 0, Some as non-zero)
        match quad.graph {
            Some(graph_id) => {
                let graph_bytes = self.encode_node_id(graph_id);
                self.encode_varint_to_vec(&mut result, graph_bytes.len() as u64);
                result.extend_from_slice(&graph_bytes);
            }
            None => {
                self.encode_varint_to_vec(&mut result, 0);
            }
        }

        result
    }

    /// Variable-length integer encoding (LEB128)
    fn encode_varint(&self, value: u64) -> Vec<u8> {
        let mut result = Vec::new();
        self.encode_varint_to_vec(&mut result, value);
        result
    }

    fn encode_varint_to_vec(&self, result: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            result.push((value & 0x7F) as u8 | 0x80);
            value >>= 7;
        }
        result.push(value as u8);
    }

    /// Encode Node IDs using variable-length encoding
    fn encode_node_ids_varint(&self, node_ids: &[NodeId]) -> Vec<u8> {
        let mut result = Vec::new();

        // Encode count first
        self.encode_varint_to_vec(&mut result, node_ids.len() as u64);

        // Encode each ID
        for &id in node_ids {
            self.encode_varint_to_vec(&mut result, id);
        }

        result
    }

    /// Encode Node IDs using delta compression
    fn encode_node_ids_delta(&self, node_ids: &[NodeId]) -> Vec<u8> {
        let mut result = Vec::new();

        // Encode count and encoding scheme
        result.push(1u8); // Delta encoding marker
        self.encode_varint_to_vec(&mut result, node_ids.len() as u64);

        if node_ids.is_empty() {
            return result;
        }

        // Encode first value as-is
        self.encode_varint_to_vec(&mut result, node_ids[0]);

        // Encode deltas
        for i in 1..node_ids.len() {
            let delta = if node_ids[i] >= node_ids[i - 1] {
                node_ids[i] - node_ids[i - 1]
            } else {
                // Handle decreasing sequences (shouldn't be common for Node IDs)
                node_ids[i]
            };
            self.encode_varint_to_vec(&mut result, delta);
        }

        result
    }

    /// Encode Node IDs using bit packing for small values
    fn encode_node_ids_bitpacked(&self, node_ids: &[NodeId]) -> Vec<u8> {
        let mut result = Vec::new();

        // Analyze the data to determine optimal bit width
        let max_value = node_ids.iter().copied().max().unwrap_or(0);
        let bits_per_value = if max_value == 0 {
            1
        } else {
            64 - max_value.leading_zeros()
        };

        result.push(2u8); // Bit-packed encoding marker
        self.encode_varint_to_vec(&mut result, node_ids.len() as u64);
        result.push(bits_per_value as u8);

        // Pack values into bits
        let mut bit_buffer = 0u64;
        let mut bits_in_buffer = 0u8;

        for &value in node_ids {
            bit_buffer |= (value & ((1u64 << bits_per_value) - 1)) << bits_in_buffer;
            bits_in_buffer += bits_per_value as u8;

            while bits_in_buffer >= 8 {
                result.push(bit_buffer as u8);
                bit_buffer >>= 8;
                bits_in_buffer -= 8;
            }
        }

        // Handle remaining bits
        if bits_in_buffer > 0 {
            result.push(bit_buffer as u8);
        }

        result
    }

    /// Encode Node IDs using frame of reference compression
    fn encode_node_ids_frame_of_reference(&self, node_ids: &[NodeId]) -> Vec<u8> {
        let mut result = Vec::new();

        result.push(3u8); // Frame of reference encoding marker
        self.encode_varint_to_vec(&mut result, node_ids.len() as u64);

        if node_ids.is_empty() {
            return result;
        }

        // Use the minimum value as the frame of reference
        let min_value = node_ids.iter().copied().min().unwrap();
        self.encode_varint_to_vec(&mut result, min_value);

        // Encode deltas from the frame of reference
        for &value in node_ids {
            let delta = value - min_value;
            self.encode_varint_to_vec(&mut result, delta);
        }

        result
    }

    /// Adaptive encoding that chooses the best scheme for the data
    fn encode_node_ids_adaptive(&self, node_ids: &[NodeId]) -> Vec<u8> {
        if node_ids.len() < 4 {
            // For small sequences, use simple varint
            return self.encode_node_ids_varint(node_ids);
        }

        // Try different encodings and pick the smallest
        let varint_encoded = self.encode_node_ids_varint(node_ids);
        let delta_encoded = self.encode_node_ids_delta(node_ids);
        let bitpacked_encoded = self.encode_node_ids_bitpacked(node_ids);
        let frame_ref_encoded = self.encode_node_ids_frame_of_reference(node_ids);

        // Return the smallest encoding
        vec![
            (varint_encoded.len(), varint_encoded),
            (delta_encoded.len(), delta_encoded),
            (bitpacked_encoded.len(), bitpacked_encoded),
            (frame_ref_encoded.len(), frame_ref_encoded),
        ]
        .into_iter()
        .min_by_key(|(size, _)| *size)
        .map(|(_, data)| data)
        .unwrap_or_else(|| self.encode_node_ids_varint(node_ids))
    }

    /// Get encoding statistics
    pub fn get_stats(&self) -> &EncodingStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = EncodingStats::default();
    }
}

/// Compact decoder for the encoded data
pub struct CompactDecoder;

impl CompactDecoder {
    /// Decode a single Node ID
    pub fn decode_node_id(&self, data: &[u8]) -> Result<NodeId> {
        let mut offset = 0;
        self.decode_varint(data, &mut offset)
    }

    /// Decode a sequence of Node IDs
    pub fn decode_node_id_sequence(&self, data: &[u8]) -> Result<Vec<NodeId>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut offset = 0;

        // Check if this has an encoding scheme marker
        if data[0] <= 3 {
            let scheme_marker = data[0];
            offset = 1;

            match scheme_marker {
                1 => self.decode_node_ids_delta(data, &mut offset),
                2 => self.decode_node_ids_bitpacked(data, &mut offset),
                3 => self.decode_node_ids_frame_of_reference(data, &mut offset),
                _ => self.decode_node_ids_varint(data, &mut offset),
            }
        } else {
            // Legacy varint encoding without scheme marker
            self.decode_node_ids_varint(data, &mut offset)
        }
    }

    /// Decode a triple
    pub fn decode_triple(&self, data: &[u8]) -> Result<Triple> {
        let mut offset = 0;

        // Decode subject
        let subject_len = self.decode_varint(data, &mut offset)? as usize;
        let subject_bytes = &data[offset..offset + subject_len];
        let subject = self.decode_node_id(subject_bytes)?;
        offset += subject_len;

        // Decode predicate
        let predicate_len = self.decode_varint(data, &mut offset)? as usize;
        let predicate_bytes = &data[offset..offset + predicate_len];
        let predicate = self.decode_node_id(predicate_bytes)?;
        offset += predicate_len;

        // Decode object
        let object_len = self.decode_varint(data, &mut offset)? as usize;
        let object_bytes = &data[offset..offset + object_len];
        let object = self.decode_node_id(object_bytes)?;

        Ok(Triple::new(subject, predicate, object))
    }

    /// Decode a quad
    pub fn decode_quad(&self, data: &[u8]) -> Result<Quad> {
        let mut offset = 0;

        // Decode subject
        let subject_len = self.decode_varint(data, &mut offset)? as usize;
        let subject_bytes = &data[offset..offset + subject_len];
        let subject = self.decode_node_id(subject_bytes)?;
        offset += subject_len;

        // Decode predicate
        let predicate_len = self.decode_varint(data, &mut offset)? as usize;
        let predicate_bytes = &data[offset..offset + predicate_len];
        let predicate = self.decode_node_id(predicate_bytes)?;
        offset += predicate_len;

        // Decode object
        let object_len = self.decode_varint(data, &mut offset)? as usize;
        let object_bytes = &data[offset..offset + object_len];
        let object = self.decode_node_id(object_bytes)?;
        offset += object_len;

        // Decode optional graph
        let graph_len = self.decode_varint(data, &mut offset)? as usize;
        let graph = if graph_len == 0 {
            None
        } else {
            let graph_bytes = &data[offset..offset + graph_len];
            Some(self.decode_node_id(graph_bytes)?)
        };

        Ok(Quad::new(subject, predicate, object, graph))
    }

    /// Decode variable-length integer
    fn decode_varint(&self, data: &[u8], offset: &mut usize) -> Result<u64> {
        let mut result = 0u64;
        let mut shift = 0;

        while *offset < data.len() {
            let byte = data[*offset];
            *offset += 1;

            result |= ((byte & 0x7F) as u64) << shift;

            if (byte & 0x80) == 0 {
                return Ok(result);
            }

            shift += 7;
            if shift >= 64 {
                return Err(anyhow!("Variable integer too large"));
            }
        }

        Err(anyhow!("Incomplete variable integer"))
    }

    /// Decode varint-encoded Node IDs
    fn decode_node_ids_varint(&self, data: &[u8], offset: &mut usize) -> Result<Vec<NodeId>> {
        let count = self.decode_varint(data, offset)? as usize;
        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            result.push(self.decode_varint(data, offset)?);
        }

        Ok(result)
    }

    /// Decode delta-encoded Node IDs
    fn decode_node_ids_delta(&self, data: &[u8], offset: &mut usize) -> Result<Vec<NodeId>> {
        let count = self.decode_varint(data, offset)? as usize;

        if count == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(count);

        // Decode first value
        let first_value = self.decode_varint(data, offset)?;
        result.push(first_value);

        // Decode deltas and reconstruct values
        let mut current_value = first_value;
        for _ in 1..count {
            let delta = self.decode_varint(data, offset)?;
            current_value += delta;
            result.push(current_value);
        }

        Ok(result)
    }

    /// Decode bit-packed Node IDs
    fn decode_node_ids_bitpacked(&self, data: &[u8], offset: &mut usize) -> Result<Vec<NodeId>> {
        let count = self.decode_varint(data, offset)? as usize;

        if count == 0 {
            return Ok(Vec::new());
        }

        if *offset >= data.len() {
            return Err(anyhow!("Missing bits per value"));
        }

        let bits_per_value = data[*offset] as u32;
        *offset += 1;

        let mut result = Vec::with_capacity(count);
        let mask = (1u64 << bits_per_value) - 1;

        let mut bit_buffer = 0u64;
        let mut bits_in_buffer = 0u8;
        let mut byte_offset = *offset;

        for _ in 0..count {
            // Ensure we have enough bits in the buffer
            while bits_in_buffer < bits_per_value as u8 && byte_offset < data.len() {
                bit_buffer |= (data[byte_offset] as u64) << bits_in_buffer;
                bits_in_buffer += 8;
                byte_offset += 1;
            }

            if bits_in_buffer < bits_per_value as u8 {
                return Err(anyhow!("Insufficient data for bit-packed values"));
            }

            // Extract the value
            let value = bit_buffer & mask;
            result.push(value);

            // Remove the extracted bits
            bit_buffer >>= bits_per_value;
            bits_in_buffer -= bits_per_value as u8;
        }

        *offset = byte_offset;
        Ok(result)
    }

    /// Decode frame-of-reference encoded Node IDs
    fn decode_node_ids_frame_of_reference(
        &self,
        data: &[u8],
        offset: &mut usize,
    ) -> Result<Vec<NodeId>> {
        let count = self.decode_varint(data, offset)? as usize;

        if count == 0 {
            return Ok(Vec::new());
        }

        let frame_of_reference = self.decode_varint(data, offset)?;
        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            let delta = self.decode_varint(data, offset)?;
            result.push(frame_of_reference + delta);
        }

        Ok(result)
    }
}

#[cfg(test)]
#[allow(clippy::uninlined_format_args)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encoding_decoding() {
        let mut encoder = CompactEncoder::new(CompactEncodingScheme::VarInt);
        let decoder = CompactDecoder;

        let test_values = vec![0, 1, 127, 128, 255, 256, 16383, 16384, u64::MAX];

        for value in test_values {
            let encoded = encoder.encode_node_id(value);
            let decoded = decoder.decode_node_id(&encoded).unwrap();
            assert_eq!(value, decoded, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_sequence_encoding_schemes() {
        let mut encoder = CompactEncoder::new(CompactEncodingScheme::Delta);
        let decoder = CompactDecoder;

        let node_ids = vec![1, 3, 5, 7, 9, 11]; // Sequential with delta of 2

        let encoded = encoder.encode_node_id_sequence(&node_ids);
        let decoded = decoder.decode_node_id_sequence(&encoded).unwrap();

        assert_eq!(node_ids, decoded);
    }

    #[test]
    fn test_triple_encoding_decoding() {
        let mut encoder = CompactEncoder::new(CompactEncodingScheme::VarInt);
        let decoder = CompactDecoder;

        let triple = Triple::new(123, 456, 789);

        let encoded = encoder.encode_triple(&triple);
        let decoded = decoder.decode_triple(&encoded).unwrap();

        assert_eq!(triple.subject, decoded.subject);
        assert_eq!(triple.predicate, decoded.predicate);
        assert_eq!(triple.object, decoded.object);
    }

    #[test]
    fn test_quad_encoding_decoding() {
        let mut encoder = CompactEncoder::new(CompactEncodingScheme::VarInt);
        let decoder = CompactDecoder;

        let quad_with_graph = Quad::new(123, 456, 789, Some(999));
        let quad_without_graph = Quad::new(123, 456, 789, None);

        // Test quad with graph
        let encoded = encoder.encode_quad(&quad_with_graph);
        let decoded = decoder.decode_quad(&encoded).unwrap();
        assert_eq!(quad_with_graph.subject, decoded.subject);
        assert_eq!(quad_with_graph.graph, decoded.graph);

        // Test quad without graph
        let encoded = encoder.encode_quad(&quad_without_graph);
        let decoded = decoder.decode_quad(&encoded).unwrap();
        assert_eq!(quad_without_graph.subject, decoded.subject);
        assert_eq!(quad_without_graph.graph, decoded.graph);
    }

    #[test]
    fn test_adaptive_encoding() {
        let mut encoder = CompactEncoder::new(CompactEncodingScheme::Hybrid);
        let decoder = CompactDecoder;

        // Test with different patterns to see adaptive behavior
        let sequential = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sparse = vec![1, 1000, 2000, 3000, 4000];
        let clustered = vec![100, 101, 102, 103, 104, 200, 201, 202, 203, 204];

        for test_data in [sequential, sparse, clustered] {
            let encoded = encoder.encode_node_id_sequence(&test_data);
            let decoded = decoder.decode_node_id_sequence(&encoded).unwrap();
            assert_eq!(test_data, decoded);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let mut encoder = CompactEncoder::new(CompactEncodingScheme::VarInt);

        // Encode some data to generate statistics
        let node_ids = (1..1000).collect::<Vec<_>>();
        let _encoded = encoder.encode_node_id_sequence(&node_ids);

        let stats = encoder.get_stats();

        // Should have some compression
        assert!(stats.compression_ratio() < 1.0);
        assert!(stats.total_encoded > 0);
    }
}
