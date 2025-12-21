use crate::error::{Result, TdbError};
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Prefix compression for dictionary strings
///
/// Stores common prefixes separately to reduce storage overhead
/// for IRIs sharing the same namespace.
#[derive(Debug, Clone)]
pub struct PrefixCompressor {
    /// Prefix dictionary: prefix_id -> prefix string
    prefixes: HashMap<u32, String>,
    /// Reverse mapping: prefix -> prefix_id
    prefix_to_id: HashMap<String, u32>,
    /// Next prefix ID
    next_prefix_id: u32,
    /// Minimum prefix length to compress
    min_prefix_len: usize,
}

/// Compressed string representation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub struct CompressedString {
    /// Prefix ID (0 = no prefix)
    pub prefix_id: u32,
    /// Suffix string
    pub suffix: String,
}

impl PrefixCompressor {
    /// Create a new prefix compressor
    pub fn new(min_prefix_len: usize) -> Self {
        Self {
            prefixes: HashMap::new(),
            prefix_to_id: HashMap::new(),
            next_prefix_id: 1,
            min_prefix_len,
        }
    }

    /// Compress a string by extracting common prefix
    pub fn compress(&mut self, s: &str) -> Result<CompressedString> {
        // Try to find longest matching prefix
        let mut best_prefix_id = 0;
        let mut best_prefix_len = 0;

        for (prefix, &prefix_id) in &self.prefix_to_id {
            if s.starts_with(prefix) && prefix.len() > best_prefix_len {
                best_prefix_id = prefix_id;
                best_prefix_len = prefix.len();
            }
        }

        if best_prefix_len > 0 {
            // Use existing prefix
            Ok(CompressedString {
                prefix_id: best_prefix_id,
                suffix: s[best_prefix_len..].to_string(),
            })
        } else if s.len() >= self.min_prefix_len {
            // Try to create new prefix from common namespace patterns
            if let Some(prefix_end) = self.find_prefix_boundary(s) {
                let prefix = &s[..prefix_end];
                let prefix_id = self.add_prefix(prefix);

                Ok(CompressedString {
                    prefix_id,
                    suffix: s[prefix_end..].to_string(),
                })
            } else {
                // No prefix found
                Ok(CompressedString {
                    prefix_id: 0,
                    suffix: s.to_string(),
                })
            }
        } else {
            // String too short to compress
            Ok(CompressedString {
                prefix_id: 0,
                suffix: s.to_string(),
            })
        }
    }

    /// Decompress a compressed string
    pub fn decompress(&self, compressed: &CompressedString) -> Result<String> {
        if compressed.prefix_id == 0 {
            Ok(compressed.suffix.clone())
        } else {
            let prefix = self.prefixes.get(&compressed.prefix_id).ok_or_else(|| {
                TdbError::Other(format!("Prefix {} not found", compressed.prefix_id))
            })?;

            Ok(format!("{}{}", prefix, compressed.suffix))
        }
    }

    /// Add a new prefix
    fn add_prefix(&mut self, prefix: &str) -> u32 {
        if let Some(&id) = self.prefix_to_id.get(prefix) {
            return id;
        }

        let id = self.next_prefix_id;
        self.next_prefix_id += 1;

        self.prefixes.insert(id, prefix.to_string());
        self.prefix_to_id.insert(prefix.to_string(), id);

        id
    }

    /// Find prefix boundary (e.g., after namespace separator)
    fn find_prefix_boundary(&self, s: &str) -> Option<usize> {
        // For IRIs, try to split at '#' or last '/'
        if let Some(pos) = s.rfind('#') {
            return Some(pos + 1);
        }

        if let Some(pos) = s.rfind('/') {
            if pos + 1 < s.len() {
                return Some(pos + 1);
            }
        }

        None
    }

    /// Get compression statistics
    pub fn stats(&self) -> CompressionStats {
        CompressionStats {
            num_prefixes: self.prefixes.len(),
            total_prefix_bytes: self.prefixes.values().map(|p| p.len()).sum(),
        }
    }

    /// Get number of prefixes
    pub fn num_prefixes(&self) -> usize {
        self.prefixes.len()
    }

    /// Get a prefix by ID
    pub fn get_prefix(&self, prefix_id: u32) -> Option<&String> {
        self.prefixes.get(&prefix_id)
    }
}

impl Default for PrefixCompressor {
    fn default() -> Self {
        Self::new(10) // Default minimum prefix length
    }
}

/// Compression statistics
#[derive(Debug, Clone, Copy)]
pub struct CompressionStats {
    /// Number of unique prefixes
    pub num_prefixes: usize,
    /// Total bytes used by prefixes
    pub total_prefix_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_compression_basic() {
        let mut compressor = PrefixCompressor::new(10);

        let iri1 = "http://example.org/Person";
        let iri2 = "http://example.org/name";

        let c1 = compressor.compress(iri1).unwrap();
        let c2 = compressor.compress(iri2).unwrap();

        // Both should share the same prefix
        assert_eq!(c1.prefix_id, c2.prefix_id);
        assert!(c1.prefix_id > 0);

        assert_eq!(c1.suffix, "Person");
        assert_eq!(c2.suffix, "name");

        let d1 = compressor.decompress(&c1).unwrap();
        let d2 = compressor.decompress(&c2).unwrap();

        assert_eq!(d1, iri1);
        assert_eq!(d2, iri2);
    }

    #[test]
    fn test_prefix_compression_no_prefix() {
        let mut compressor = PrefixCompressor::new(10);

        let short_str = "abc";
        let c = compressor.compress(short_str).unwrap();

        assert_eq!(c.prefix_id, 0);
        assert_eq!(c.suffix, "abc");

        let d = compressor.decompress(&c).unwrap();
        assert_eq!(d, short_str);
    }

    #[test]
    fn test_prefix_compression_hash_iri() {
        let mut compressor = PrefixCompressor::new(10);

        let iri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        let c = compressor.compress(iri).unwrap();

        assert!(c.prefix_id > 0);
        assert_eq!(c.suffix, "type");

        let d = compressor.decompress(&c).unwrap();
        assert_eq!(d, iri);
    }

    #[test]
    fn test_prefix_compression_multiple_prefixes() {
        let mut compressor = PrefixCompressor::new(10);

        let iri1 = "http://example.org/Person";
        let iri2 = "http://xmlns.com/foaf/0.1/name";

        let c1 = compressor.compress(iri1).unwrap();
        let c2 = compressor.compress(iri2).unwrap();

        // Different prefixes
        assert_ne!(c1.prefix_id, c2.prefix_id);
        assert!(c1.prefix_id > 0);
        assert!(c2.prefix_id > 0);

        assert_eq!(compressor.num_prefixes(), 2);
    }

    #[test]
    fn test_prefix_compression_stats() {
        let mut compressor = PrefixCompressor::new(10);

        compressor.compress("http://example.org/Person").unwrap();
        compressor.compress("http://example.org/name").unwrap();

        let stats = compressor.stats();
        assert_eq!(stats.num_prefixes, 1);
        assert!(stats.total_prefix_bytes > 0);
    }

    #[test]
    fn test_prefix_reuse() {
        let mut compressor = PrefixCompressor::new(10);

        let iri1 = "http://example.org/Person";
        let iri2 = "http://example.org/name";
        let iri3 = "http://example.org/age";

        let c1 = compressor.compress(iri1).unwrap();
        let c2 = compressor.compress(iri2).unwrap();
        let c3 = compressor.compress(iri3).unwrap();

        assert_eq!(c1.prefix_id, c2.prefix_id);
        assert_eq!(c2.prefix_id, c3.prefix_id);
        assert_eq!(compressor.num_prefixes(), 1);
    }

    #[test]
    fn test_prefix_get() {
        let mut compressor = PrefixCompressor::new(10);

        let iri = "http://example.org/Person";
        let c = compressor.compress(iri).unwrap();

        let prefix = compressor.get_prefix(c.prefix_id).unwrap();
        assert_eq!(prefix, "http://example.org/");
    }

    #[test]
    fn test_compression_serialization() {
        let compressed = CompressedString {
            prefix_id: 1,
            suffix: "Person".to_string(),
        };

        let serialized = bincode::encode_to_vec(&compressed, bincode::config::standard()).unwrap();
        let deserialized: CompressedString =
            bincode::decode_from_slice(&serialized, bincode::config::standard())
                .unwrap()
                .0;

        assert_eq!(compressed, deserialized);
    }
}
