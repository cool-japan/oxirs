//! Adaptive dictionary compression implementation

use crate::compression::{
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
    MAX_DICTIONARY_SIZE,
};
use anyhow::{anyhow, Result};
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
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;

            let len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
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
            let avg_length: f64 = self.id_to_string.values().map(|v| v.len()).sum::<usize>() as f64
                / self.id_to_string.len() as f64;
            stats.insert("avg_entry_length".to_string(), format!("{avg_length:.2}"));
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

/// Dictionary encoding specifically for RDF triple components (subjects, predicates, objects).
///
/// `TripleDictionary` maps RDF URI strings and literals to compact 64-bit integer IDs,
/// dramatically reducing storage size for large RDF graphs where the same URIs appear
/// across millions of triples.  Subjects, predicates and objects maintain separate
/// namespaces so that ID `42` for a subject is unambiguous relative to predicate ID `42`.
///
/// # Compression Characteristics
///
/// For a typical SPARQL dataset:
/// - A 40-byte URI becomes an 8-byte u64 NodeID (~5x size reduction per occurrence)
/// - With 1M triples sharing 10K unique predicates the savings are dramatic
/// - `compression_ratio()` reflects actual savings in the live dictionary
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_tdb::compression::dictionary::TripleDictionary;
///
/// let mut dict = TripleDictionary::new();
/// let (s_id, p_id, o_id) = dict.encode_triple(
///     "http://example.org/Alice",
///     "http://schema.org/knows",
///     "http://example.org/Bob",
/// );
/// let (s, p, o) = dict.decode_triple(s_id, p_id, o_id).unwrap();
/// assert_eq!(s, "http://example.org/Alice");
/// ```
#[derive(Debug)]
pub struct TripleDictionary {
    /// Forward mapping: subject string -> subject ID
    subject_dict: HashMap<String, u64>,
    /// Forward mapping: predicate string -> predicate ID
    predicate_dict: HashMap<String, u64>,
    /// Forward mapping: object string -> object ID
    object_dict: HashMap<String, u64>,
    /// Reverse mapping: ID -> string (shared across all three components;
    /// IDs are globally unique across the entire dictionary)
    reverse: HashMap<u64, String>,
    /// Counter for next ID to assign (monotonically increasing)
    next_id: u64,
}

impl TripleDictionary {
    /// Create a new empty `TripleDictionary`.
    pub fn new() -> Self {
        Self {
            subject_dict: HashMap::new(),
            predicate_dict: HashMap::new(),
            object_dict: HashMap::new(),
            reverse: HashMap::new(),
            next_id: 0,
        }
    }

    /// Encode a triple (subject, predicate, object) as three 64-bit integer IDs.
    ///
    /// If a component has been seen before its existing ID is returned;
    /// otherwise a new monotonically-increasing ID is assigned.
    ///
    /// Returns `(subject_id, predicate_id, object_id)`.
    pub fn encode_triple(&mut self, s: &str, p: &str, o: &str) -> (u64, u64, u64) {
        let s_id = self.intern_subject(s);
        let p_id = self.intern_predicate(p);
        let o_id = self.intern_object(o);
        (s_id, p_id, o_id)
    }

    /// Decode a triple from three 64-bit integer IDs back to strings.
    ///
    /// Returns `Some((subject, predicate, object))` or `None` if any ID is unknown.
    pub fn decode_triple(
        &self,
        s_id: u64,
        p_id: u64,
        o_id: u64,
    ) -> Option<(String, String, String)> {
        let s = self.reverse.get(&s_id)?.clone();
        let p = self.reverse.get(&p_id)?.clone();
        let o = self.reverse.get(&o_id)?.clone();
        Some((s, p, o))
    }

    /// Return the total number of unique strings (subjects + predicates + objects combined)
    /// currently interned in the dictionary.
    pub fn size(&self) -> usize {
        self.reverse.len()
    }

    /// Estimate the compression ratio achieved by this dictionary.
    ///
    /// The ratio is computed as:
    ///   `(total_string_bytes) / (entries * 8)`
    ///
    /// where `entries * 8` is the storage cost of the IDs alone (8 bytes per u64).
    /// A ratio > 1.0 means the dictionary saves space; typical values are 4-10x for
    /// RDF URI-heavy workloads.
    pub fn compression_ratio(&self) -> f64 {
        if self.reverse.is_empty() {
            return 1.0;
        }
        let total_string_bytes: usize = self.reverse.values().map(|s| s.len()).sum();
        let id_bytes = self.reverse.len() * 8; // 8 bytes per u64 ID
        if id_bytes == 0 {
            return 1.0;
        }
        total_string_bytes as f64 / id_bytes as f64
    }

    /// Look up the ID for a subject string, returning `None` if not interned.
    pub fn get_subject_id(&self, s: &str) -> Option<u64> {
        self.subject_dict.get(s).copied()
    }

    /// Look up the ID for a predicate string, returning `None` if not interned.
    pub fn get_predicate_id(&self, p: &str) -> Option<u64> {
        self.predicate_dict.get(p).copied()
    }

    /// Look up the ID for an object string, returning `None` if not interned.
    pub fn get_object_id(&self, o: &str) -> Option<u64> {
        self.object_dict.get(o).copied()
    }

    /// Decode any ID back to the original string, regardless of component type.
    ///
    /// Returns `None` if the ID is not known.
    pub fn decode_id(&self, id: u64) -> Option<&str> {
        self.reverse.get(&id).map(|s| s.as_str())
    }

    /// Return the number of unique subjects.
    pub fn subject_count(&self) -> usize {
        self.subject_dict.len()
    }

    /// Return the number of unique predicates.
    pub fn predicate_count(&self) -> usize {
        self.predicate_dict.len()
    }

    /// Return the number of unique objects.
    pub fn object_count(&self) -> usize {
        self.object_dict.len()
    }

    // -- Internal helpers --

    fn intern_subject(&mut self, s: &str) -> u64 {
        if let Some(&id) = self.subject_dict.get(s) {
            return id;
        }
        let id = self.alloc_id();
        self.subject_dict.insert(s.to_string(), id);
        self.reverse.insert(id, s.to_string());
        id
    }

    fn intern_predicate(&mut self, p: &str) -> u64 {
        if let Some(&id) = self.predicate_dict.get(p) {
            return id;
        }
        let id = self.alloc_id();
        self.predicate_dict.insert(p.to_string(), id);
        self.reverse.insert(id, p.to_string());
        id
    }

    fn intern_object(&mut self, o: &str) -> u64 {
        if let Some(&id) = self.object_dict.get(o) {
            return id;
        }
        let id = self.alloc_id();
        self.object_dict.insert(o.to_string(), id);
        self.reverse.insert(id, o.to_string());
        id
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }
}

impl Default for TripleDictionary {
    fn default() -> Self {
        Self::new()
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
        assert_eq!(
            compressed.metadata.algorithm,
            AdvancedCompressionType::AdaptiveDictionary
        );

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

    // -- TripleDictionary tests --

    #[test]
    fn test_triple_dictionary_encode_decode_roundtrip() {
        let mut dict = TripleDictionary::new();

        let s = "http://example.org/Alice";
        let p = "http://schema.org/knows";
        let o = "http://example.org/Bob";

        let (s_id, p_id, o_id) = dict.encode_triple(s, p, o);
        let decoded = dict
            .decode_triple(s_id, p_id, o_id)
            .expect("decode must succeed");
        assert_eq!(decoded.0, s);
        assert_eq!(decoded.1, p);
        assert_eq!(decoded.2, o);
    }

    #[test]
    fn test_triple_dictionary_same_uri_same_id() {
        let mut dict = TripleDictionary::new();

        let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        let class_a = "http://example.org/ClassA";
        let class_b = "http://example.org/ClassB";

        let (_, p1, _) = dict.encode_triple("http://example.org/s1", rdf_type, class_a);
        let (_, p2, _) = dict.encode_triple("http://example.org/s2", rdf_type, class_b);
        // Same predicate URI must get same ID
        assert_eq!(p1, p2, "same predicate URI must map to same ID");
    }

    #[test]
    fn test_triple_dictionary_size() {
        let mut dict = TripleDictionary::new();
        assert_eq!(dict.size(), 0);

        // Three distinct URIs
        dict.encode_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        );
        assert_eq!(dict.size(), 3, "three unique URIs should give size 3");

        // Repeat the same triple - no new entries
        dict.encode_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        );
        assert_eq!(
            dict.size(),
            3,
            "duplicate triple must not grow the dictionary"
        );
    }

    #[test]
    fn test_triple_dictionary_compression_ratio() {
        let mut dict = TripleDictionary::new();
        // Long URIs give a good compression ratio
        for i in 0..100 {
            dict.encode_triple(
                &format!("http://very-long-namespace.example.org/subject/{}", i),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://very-long-namespace.example.org/SomeClass",
            );
        }
        let ratio = dict.compression_ratio();
        assert!(
            ratio > 1.0,
            "long URIs should have compression_ratio > 1.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_triple_dictionary_decode_unknown_id_returns_none() {
        let dict = TripleDictionary::new();
        let result = dict.decode_triple(9999, 9998, 9997);
        assert!(result.is_none(), "unknown IDs should return None");
    }

    #[test]
    fn test_triple_dictionary_component_counts() {
        let mut dict = TripleDictionary::new();

        dict.encode_triple(
            "http://example.org/s1",
            "http://schema.org/name",
            "\"Alice\"",
        );
        dict.encode_triple("http://example.org/s2", "http://schema.org/name", "\"Bob\"");
        dict.encode_triple("http://example.org/s1", "http://schema.org/age", "\"30\"");

        assert_eq!(dict.subject_count(), 2, "two distinct subjects");
        assert_eq!(dict.predicate_count(), 2, "two distinct predicates");
        assert_eq!(dict.object_count(), 3, "three distinct objects");
    }

    #[test]
    fn test_triple_dictionary_large_graph() {
        let mut dict = TripleDictionary::new();

        // Simulate a medium-sized RDF graph
        let predicates = [
            "http://schema.org/name",
            "http://schema.org/knows",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        ];

        for i in 0..1000 {
            let s = format!("http://example.org/person/{}", i);
            let p = predicates[i % predicates.len()];
            let o = format!("http://example.org/person/{}", (i + 1) % 1000);
            dict.encode_triple(&s, p, &o);
        }

        // 1000 unique subjects = 1000 s IDs; 3 unique predicates; subjects also used as objects
        assert_eq!(dict.subject_count(), 1000);
        assert_eq!(dict.predicate_count(), 3);
        // objects are same URIs as subjects (partially reused via the same global ID)
        // after the full loop all 1000 person URIs appear as objects too, but they get
        // separate entries in the object_dict (they're tracked independently per role)
        assert!(dict.object_count() <= 1000);

        let ratio = dict.compression_ratio();
        // With ~35-byte URIs each mapped to 8 bytes: ratio ~ 35/8 ~ 4.4x
        assert!(
            ratio > 2.0,
            "large graph should achieve > 2x compression ratio, got {}",
            ratio
        );
    }

    #[test]
    fn test_triple_dictionary_get_id_helpers() {
        let mut dict = TripleDictionary::new();
        let s = "http://example.org/s";
        let p = "http://schema.org/p";
        let o = "http://example.org/o";

        let (s_id, p_id, o_id) = dict.encode_triple(s, p, o);

        assert_eq!(dict.get_subject_id(s), Some(s_id));
        assert_eq!(dict.get_predicate_id(p), Some(p_id));
        assert_eq!(dict.get_object_id(o), Some(o_id));
        assert_eq!(dict.get_subject_id("unknown"), None);
    }

    #[test]
    fn test_triple_dictionary_decode_id() {
        let mut dict = TripleDictionary::new();
        let (s_id, p_id, o_id) = dict.encode_triple(
            "http://example.org/s",
            "http://schema.org/p",
            "http://example.org/o",
        );
        assert_eq!(dict.decode_id(s_id), Some("http://example.org/s"));
        assert_eq!(dict.decode_id(p_id), Some("http://schema.org/p"));
        assert_eq!(dict.decode_id(o_id), Some("http://example.org/o"));
        assert_eq!(dict.decode_id(99999), None);
    }
}
