//! Adaptive dictionary compression implementation

use anyhow::{anyhow, Result};
use crate::compression::{
    AdvancedCompressionType, CompressionAlgorithm, CompressionMetadata, CompressedData,
    MAX_DICTIONARY_SIZE
};
use std::collections::HashMap;
use std::time::Instant;

/// Adaptive dictionary compression
pub struct AdaptiveDictionary {
    /// Dictionary mapping strings to IDs
    string_to_id: HashMap<Vec<u8>, u32>,
    /// Dictionary mapping IDs to strings  
    id_to_string: HashMap<u32, Vec<u8>>,
    /// Next available ID
    next_id: u32,
}

impl AdaptiveDictionary {
    /// Create new adaptive dictionary
    pub fn new() -> Self {
        Self {
            string_to_id: HashMap::new(),
            id_to_string: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add string to dictionary and return ID
    pub fn add_string(&mut self, data: &[u8]) -> Result<u32> {
        if let Some(&id) = self.string_to_id.get(data) {
            return Ok(id);
        }

        if self.string_to_id.len() >= MAX_DICTIONARY_SIZE {
            return Err(anyhow!(
                "Dictionary size limit exceeded: {} entries (max: {})",
                self.string_to_id.len(),
                MAX_DICTIONARY_SIZE
            ));
        }

        let id = self.next_id;
        self.next_id += 1;

        self.string_to_id.insert(data.to_vec(), id);
        self.id_to_string.insert(id, data.to_vec());

        Ok(id)
    }

    /// Get string by ID
    pub fn get_string(&self, id: u32) -> Option<&[u8]> {
        self.id_to_string.get(&id).map(|v| v.as_slice())
    }

    /// Get ID by string
    pub fn get_id(&self, data: &[u8]) -> Option<u32> {
        self.string_to_id.get(data).copied()
    }

    /// Compress data using dictionary
    pub fn compress_data(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut compressed = Vec::new();
        
        // Simple word-based compression (split on whitespace)
        let mut current_word = Vec::new();
        
        for &byte in data {
            if byte.is_ascii_whitespace() {
                if !current_word.is_empty() {
                    let id = self.add_string(&current_word)?;
                    compressed.extend_from_slice(&id.to_le_bytes());
                    current_word.clear();
                }
                // Store whitespace as-is (negative ID)
                compressed.extend_from_slice(&(u32::MAX - byte as u32).to_le_bytes());
            } else {
                current_word.push(byte);
            }
        }

        // Handle final word
        if !current_word.is_empty() {
            let id = self.add_string(&current_word)?;
            compressed.extend_from_slice(&id.to_le_bytes());
        }

        Ok(compressed)
    }

    /// Decompress data using dictionary
    pub fn decompress_data(&self, compressed: &[u8]) -> Result<Vec<u8>> {
        if compressed.is_empty() {
            return Ok(Vec::new());
        }

        if compressed.len() % 4 != 0 {
            return Err(anyhow!("Invalid compressed data length"));
        }

        let mut decompressed = Vec::new();

        for chunk in compressed.chunks_exact(4) {
            let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            
            if id > u32::MAX - 256 {
                // It's a whitespace character
                let byte = (u32::MAX - id) as u8;
                decompressed.push(byte);
            } else {
                // It's a dictionary ID
                if let Some(data) = self.get_string(id) {
                    decompressed.extend_from_slice(data);
                } else {
                    return Err(anyhow!("Invalid dictionary ID: {}", id));
                }
            }
        }

        Ok(decompressed)
    }

    /// Serialize dictionary for storage
    pub fn serialize_dictionary(&self) -> Vec<u8> {
        let mut serialized = Vec::new();
        
        // Store dictionary size
        let size = self.id_to_string.len() as u32;
        serialized.extend_from_slice(&size.to_le_bytes());

        // Store each entry
        for (&id, data) in &self.id_to_string {
            serialized.extend_from_slice(&id.to_le_bytes());
            let len = data.len() as u32;
            serialized.extend_from_slice(&len.to_le_bytes());
            serialized.extend_from_slice(data);
        }

        serialized
    }

    /// Deserialize dictionary from storage
    pub fn deserialize_dictionary(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 4 {
            return Err(anyhow!("Invalid dictionary data"));
        }

        let size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut offset = 4;

        self.string_to_id.clear();
        self.id_to_string.clear();
        self.next_id = 0;

        for _ in 0..size {
            if offset + 8 > data.len() {
                return Err(anyhow!("Truncated dictionary data"));
            }

            let id = u32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
            ]);
            offset += 4;

            let len = u32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
            ]) as usize;
            offset += 4;

            if offset + len > data.len() {
                return Err(anyhow!("Truncated dictionary entry"));
            }

            let string_data = data[offset..offset + len].to_vec();
            offset += len;

            self.string_to_id.insert(string_data.clone(), id);
            self.id_to_string.insert(id, string_data);
            self.next_id = self.next_id.max(id + 1);
        }

        Ok(())
    }

    /// Get dictionary statistics
    pub fn statistics(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("entries".to_string(), self.string_to_id.len().to_string());
        stats.insert("next_id".to_string(), self.next_id.to_string());
        
        if !self.id_to_string.is_empty() {
            let avg_length: f64 = self.id_to_string.values()
                .map(|v| v.len())
                .sum::<usize>() as f64 / self.id_to_string.len() as f64;
            stats.insert("avg_entry_length".to_string(), format!("{:.2}", avg_length));
        }

        stats
    }
}

impl Default for AdaptiveDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionAlgorithm for AdaptiveDictionary {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        let mut dict = self.clone();
        let start = Instant::now();
        let compressed_data = dict.compress_data(data)?;
        let compression_time = start.elapsed();

        // Combine dictionary and compressed data
        let dict_data = dict.serialize_dictionary();
        let mut final_data = Vec::new();
        final_data.extend_from_slice(&(dict_data.len() as u32).to_le_bytes());
        final_data.extend_from_slice(&dict_data);
        final_data.extend_from_slice(&compressed_data);

        let metadata = CompressionMetadata {
            algorithm: AdvancedCompressionType::AdaptiveDictionary,
            original_size: data.len() as u64,
            compressed_size: final_data.len() as u64,
            compression_time_us: compression_time.as_micros() as u64,
            metadata: dict.statistics(),
        };

        Ok(CompressedData {
            data: final_data,
            metadata,
        })
    }

    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        if compressed.metadata.algorithm != AdvancedCompressionType::AdaptiveDictionary {
            return Err(anyhow!(
                "Invalid compression algorithm: expected AdaptiveDictionary, got {}",
                compressed.metadata.algorithm
            ));
        }

        let data = &compressed.data;
        if data.len() < 4 {
            return Err(anyhow!("Invalid compressed data"));
        }

        // Extract dictionary
        let dict_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + dict_len {
            return Err(anyhow!("Invalid compressed data"));
        }

        let mut dict = AdaptiveDictionary::new();
        dict.deserialize_dictionary(&data[4..4 + dict_len])?;

        // Decompress data
        let compressed_data = &data[4 + dict_len..];
        dict.decompress_data(compressed_data)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::AdaptiveDictionary
    }
}

// Implement Clone for AdaptiveDictionary
impl Clone for AdaptiveDictionary {
    fn clone(&self) -> Self {
        Self {
            string_to_id: self.string_to_id.clone(),
            id_to_string: self.id_to_string.clone(),
            next_id: self.next_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_basic() {
        let mut dict = AdaptiveDictionary::new();
        
        let id1 = dict.add_string(b"hello").unwrap();
        let id2 = dict.add_string(b"world").unwrap();
        let id3 = dict.add_string(b"hello").unwrap(); // Should reuse
        
        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(dict.get_string(id1), Some(&b"hello"[..]));
        assert_eq!(dict.get_string(id2), Some(&b"world"[..]));
    }

    #[test]
    fn test_compression_algorithm_trait() {
        let dict = AdaptiveDictionary::new();
        let data = b"hello world hello world";
        
        let compressed = dict.compress(data).unwrap();
        assert_eq!(compressed.metadata.algorithm, AdvancedCompressionType::AdaptiveDictionary);
        
        let decompressed = dict.decompress(&compressed).unwrap();
        assert_eq!(data, decompressed.as_slice());
    }

    #[test]
    fn test_dictionary_serialization() {
        let mut dict = AdaptiveDictionary::new();
        dict.add_string(b"test").unwrap();
        dict.add_string(b"data").unwrap();
        
        let serialized = dict.serialize_dictionary();
        
        let mut new_dict = AdaptiveDictionary::new();
        new_dict.deserialize_dictionary(&serialized).unwrap();
        
        assert_eq!(dict.get_id(b"test"), new_dict.get_id(b"test"));
        assert_eq!(dict.get_id(b"data"), new_dict.get_id(b"data"));
    }
}