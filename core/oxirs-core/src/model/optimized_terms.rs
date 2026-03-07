// Optimized RDF term representations based on Oxigraph's oxrdf optimizations
// This module provides memory-efficient, hash-based term storage and encoding

use crate::model::{BlankNode, Literal, NamedNode};
use siphasher::sip128::{Hasher128, SipHasher24};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

/// A 16-byte hash for efficient string deduplication (Oxigraph-inspired optimization)
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub struct OxiStrHash {
    hash: [u8; 16],
}

impl OxiStrHash {
    pub fn new(value: &str) -> Self {
        let mut hasher = SipHasher24::new();
        hasher.write(value.as_bytes());
        Self {
            hash: u128::from(hasher.finish128()).to_be_bytes(),
        }
    }

    #[inline]
    pub fn from_be_bytes(hash: [u8; 16]) -> Self {
        Self { hash }
    }

    #[inline]
    pub fn to_be_bytes(self) -> [u8; 16] {
        self.hash
    }
}

impl Hash for OxiStrHash {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u128(u128::from_ne_bytes(self.hash))
    }
}

/// Compact encoded representation of RDF terms (Oxigraph-inspired optimization)
#[derive(Debug, Clone, PartialEq)]
pub enum OxiEncodedTerm {
    DefaultGraph,
    NamedNode {
        iri: OxiStrHash,
    },
    BlankNode {
        id: OxiStrHash,
    },
    Literal {
        value: OxiStrHash,
        datatype: Option<OxiStrHash>,
        language: Option<String>,
    },
    // Optimized encodings for common literal types
    BooleanLiteral(bool),
    IntegerLiteral(i64),
    FloatLiteral(f32),
    DoubleLiteral(f64),
    StringLiteral(OxiStrHash),
}

impl Hash for OxiEncodedTerm {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            OxiEncodedTerm::DefaultGraph => {
                0u8.hash(state);
            }
            OxiEncodedTerm::NamedNode { iri } => {
                1u8.hash(state);
                iri.hash(state);
            }
            OxiEncodedTerm::BlankNode { id } => {
                2u8.hash(state);
                id.hash(state);
            }
            OxiEncodedTerm::Literal {
                value,
                datatype,
                language,
            } => {
                3u8.hash(state);
                value.hash(state);
                datatype.hash(state);
                language.hash(state);
            }
            OxiEncodedTerm::BooleanLiteral(value) => {
                4u8.hash(state);
                value.hash(state);
            }
            OxiEncodedTerm::IntegerLiteral(value) => {
                5u8.hash(state);
                value.hash(state);
            }
            OxiEncodedTerm::FloatLiteral(value) => {
                6u8.hash(state);
                // For floating point values, we use the bit representation for hashing
                value.to_bits().hash(state);
            }
            OxiEncodedTerm::DoubleLiteral(value) => {
                7u8.hash(state);
                // For floating point values, we use the bit representation for hashing
                value.to_bits().hash(state);
            }
            OxiEncodedTerm::StringLiteral(value) => {
                8u8.hash(state);
                value.hash(state);
            }
        }
    }
}

impl Eq for OxiEncodedTerm {}

/// High-performance string interner using hash-based deduplication
#[derive(Debug, Default)]
pub struct StringInterner {
    /// Maps hashes to actual strings
    string_storage: HashMap<OxiStrHash, String>,
    /// Statistics
    total_strings: usize,
    total_deduplication_saves: usize,
}

impl StringInterner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string and return its hash
    pub fn intern(&mut self, value: &str) -> OxiStrHash {
        let hash = OxiStrHash::new(value);

        if let std::collections::hash_map::Entry::Vacant(e) = self.string_storage.entry(hash) {
            e.insert(value.to_string());
            self.total_strings += 1;
        } else {
            self.total_deduplication_saves += 1;
        }

        hash
    }

    /// Resolve a hash back to its string
    pub fn resolve(&self, hash: &OxiStrHash) -> Option<&str> {
        self.string_storage.get(hash).map(|s| s.as_str())
    }

    /// Get interning statistics
    pub fn stats(&self) -> InternerStats {
        InternerStats {
            total_strings: self.total_strings,
            deduplication_saves: self.total_deduplication_saves,
            memory_usage: self.string_storage.values().map(|s| s.len()).sum(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InternerStats {
    pub total_strings: usize,
    pub deduplication_saves: usize,
    pub memory_usage: usize,
}

/// Thread-safe term encoder with optimized storage
pub struct OptimizedTermEncoder {
    interner: Arc<RwLock<StringInterner>>,
}

impl OptimizedTermEncoder {
    pub fn new() -> Self {
        Self {
            interner: Arc::new(RwLock::new(StringInterner::new())),
        }
    }

    /// Encode a named node efficiently
    pub fn encode_named_node(&self, node: &NamedNode) -> OxiEncodedTerm {
        let mut interner = self
            .interner
            .write()
            .expect("interner lock should not be poisoned");
        let iri_hash = interner.intern(node.as_str());
        OxiEncodedTerm::NamedNode { iri: iri_hash }
    }

    /// Encode a blank node efficiently
    pub fn encode_blank_node(&self, node: &BlankNode) -> OxiEncodedTerm {
        let mut interner = self
            .interner
            .write()
            .expect("interner lock should not be poisoned");
        let id_hash = interner.intern(node.as_str());
        OxiEncodedTerm::BlankNode { id: id_hash }
    }

    /// Encode a literal with type-specific optimizations
    pub fn encode_literal(&self, literal: &Literal) -> OxiEncodedTerm {
        let literal_str = literal.value();

        // Try type-specific optimizations first
        let datatype = literal.datatype();
        match datatype.as_str() {
            "http://www.w3.org/2001/XMLSchema#boolean" => {
                if let Ok(value) = literal_str.parse::<bool>() {
                    return OxiEncodedTerm::BooleanLiteral(value);
                }
            }
            "http://www.w3.org/2001/XMLSchema#integer"
            | "http://www.w3.org/2001/XMLSchema#int"
            | "http://www.w3.org/2001/XMLSchema#long" => {
                if let Ok(value) = literal_str.parse::<i64>() {
                    return OxiEncodedTerm::IntegerLiteral(value);
                }
            }
            "http://www.w3.org/2001/XMLSchema#float" => {
                if let Ok(value) = literal_str.parse::<f32>() {
                    return OxiEncodedTerm::FloatLiteral(value);
                }
            }
            "http://www.w3.org/2001/XMLSchema#double" => {
                if let Ok(value) = literal_str.parse::<f64>() {
                    return OxiEncodedTerm::DoubleLiteral(value);
                }
            }
            "http://www.w3.org/2001/XMLSchema#string" => {
                let mut interner = self
                    .interner
                    .write()
                    .expect("interner lock should not be poisoned");
                let value_hash = interner.intern(literal_str);
                return OxiEncodedTerm::StringLiteral(value_hash);
            }
            _ => {
                // Fall through to general encoding
            }
        }

        // General literal encoding
        let mut interner = self
            .interner
            .write()
            .expect("interner lock should not be poisoned");
        let value_hash = interner.intern(literal_str);

        let datatype_hash = Some(interner.intern(datatype.as_str()));
        let language = literal.language().map(|lang| lang.to_string());

        OxiEncodedTerm::Literal {
            value: value_hash,
            datatype: datatype_hash,
            language,
        }
    }

    /// Decode an encoded term back to its original form
    pub fn decode_term(&self, encoded: &OxiEncodedTerm) -> Result<DecodedTerm, String> {
        let interner = self
            .interner
            .read()
            .expect("interner lock should not be poisoned");

        match encoded {
            OxiEncodedTerm::DefaultGraph => Ok(DecodedTerm::DefaultGraph),

            OxiEncodedTerm::NamedNode { iri } => {
                let iri_str = interner
                    .resolve(iri)
                    .ok_or("IRI hash not found in interner")?;
                Ok(DecodedTerm::NamedNode(iri_str.to_string()))
            }

            OxiEncodedTerm::BlankNode { id } => {
                let id_str = interner
                    .resolve(id)
                    .ok_or("Blank node ID hash not found in interner")?;
                Ok(DecodedTerm::BlankNode(id_str.to_string()))
            }

            OxiEncodedTerm::BooleanLiteral(value) => Ok(DecodedTerm::Literal {
                value: value.to_string(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#boolean".to_string()),
                language: None,
            }),

            OxiEncodedTerm::IntegerLiteral(value) => Ok(DecodedTerm::Literal {
                value: value.to_string(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                language: None,
            }),

            OxiEncodedTerm::FloatLiteral(value) => Ok(DecodedTerm::Literal {
                value: value.to_string(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#float".to_string()),
                language: None,
            }),

            OxiEncodedTerm::DoubleLiteral(value) => Ok(DecodedTerm::Literal {
                value: value.to_string(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
                language: None,
            }),

            OxiEncodedTerm::StringLiteral(value_hash) => {
                let value_str = interner
                    .resolve(value_hash)
                    .ok_or("String literal hash not found in interner")?;
                Ok(DecodedTerm::Literal {
                    value: value_str.to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                    language: None,
                })
            }

            OxiEncodedTerm::Literal {
                value,
                datatype,
                language,
            } => {
                let value_str = interner
                    .resolve(value)
                    .ok_or("Literal value hash not found in interner")?;

                let datatype_str = if let Some(dt_hash) = datatype {
                    Some(
                        interner
                            .resolve(dt_hash)
                            .ok_or("Datatype hash not found in interner")?
                            .to_string(),
                    )
                } else {
                    None
                };

                Ok(DecodedTerm::Literal {
                    value: value_str.to_string(),
                    datatype: datatype_str,
                    language: language.clone(),
                })
            }
        }
    }

    /// Get interner statistics
    pub fn stats(&self) -> InternerStats {
        self.interner
            .read()
            .expect("interner lock should not be poisoned")
            .stats()
    }
}

impl Default for OptimizedTermEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Decoded term representation for reconstruction
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedTerm {
    DefaultGraph,
    NamedNode(String),
    BlankNode(String),
    Literal {
        value: String,
        datatype: Option<String>,
        language: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[test]
    fn test_string_interner() {
        let mut interner = StringInterner::new();

        let hash1 = interner.intern("http://example.org/test");
        let hash2 = interner.intern("http://example.org/test"); // Same string
        let hash3 = interner.intern("http://example.org/other");

        assert_eq!(hash1, hash2); // Same hash for same string
        assert_ne!(hash1, hash3); // Different hash for different string

        assert_eq!(interner.resolve(&hash1), Some("http://example.org/test"));
        assert_eq!(interner.resolve(&hash3), Some("http://example.org/other"));

        let stats = interner.stats();
        assert_eq!(stats.total_strings, 2); // Only 2 unique strings
        assert_eq!(stats.deduplication_saves, 1); // 1 deduplication
    }

    #[test]
    fn test_optimized_encoding() -> Result<(), Box<dyn std::error::Error>> {
        let encoder = OptimizedTermEncoder::new();

        // Test named node encoding
        let named_node = NamedNode::new("http://example.org/test")?;
        let encoded = encoder.encode_named_node(&named_node);

        match encoder.decode_term(&encoded)? {
            DecodedTerm::NamedNode(iri) => {
                assert_eq!(iri, "http://example.org/test");
            }
            _ => panic!("Expected named node"),
        }

        // Test optimized integer literal
        let int_literal = Literal::new_typed_literal(
            "42",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?,
        );
        let encoded = encoder.encode_literal(&int_literal);

        assert!(matches!(encoded, OxiEncodedTerm::IntegerLiteral(42)));

        // Test optimized boolean literal
        let bool_literal = Literal::new_typed_literal(
            "true",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")?,
        );
        let encoded = encoder.encode_literal(&bool_literal);

        assert!(matches!(encoded, OxiEncodedTerm::BooleanLiteral(true)));

        Ok(())
    }

    #[test]
    fn test_hash_consistency() {
        let hash1 = OxiStrHash::new("test string");
        let hash2 = OxiStrHash::new("test string");
        let hash3 = OxiStrHash::new("different string");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);

        // Test byte conversion
        let bytes = hash1.to_be_bytes();
        let reconstructed = OxiStrHash::from_be_bytes(bytes);
        assert_eq!(hash1, reconstructed);
    }

    #[test]
    fn test_edge_cases_empty_string() {
        let mut interner = StringInterner::new();

        // Test empty string handling
        let empty_hash = interner.intern("");
        assert_eq!(interner.resolve(&empty_hash), Some(""));

        // Test multiple empty strings (should deduplicate)
        let empty_hash2 = interner.intern("");
        assert_eq!(empty_hash, empty_hash2);

        let stats = interner.stats();
        assert_eq!(stats.total_strings, 1);
        assert_eq!(stats.deduplication_saves, 1);
    }

    #[test]
    fn test_edge_cases_unicode_strings() {
        let mut interner = StringInterner::new();

        // Test Unicode strings
        let unicode_test_cases = [
            "Hello, 世界!",
            "Ħello, мир!",
            "🌍🚀✨",
            "नमस्ते",
            "مرحبا",
            "\u{1F4A9}\u{200D}\u{1F4BB}", // Complex emoji sequence
        ];

        for test_case in &unicode_test_cases {
            let hash = interner.intern(test_case);
            assert_eq!(interner.resolve(&hash), Some(*test_case));
        }
    }

    #[test]
    fn test_edge_cases_large_strings() {
        let mut interner = StringInterner::new();

        // Test very large strings
        let large_string = "x".repeat(1_000_000); // 1MB string
        let hash = interner.intern(&large_string);
        assert_eq!(interner.resolve(&hash), Some(large_string.as_str()));

        // Test deduplication of large strings
        let hash2 = interner.intern(&large_string);
        assert_eq!(hash, hash2);

        let stats = interner.stats();
        assert_eq!(stats.deduplication_saves, 1);
    }

    #[test]
    fn test_error_conditions_invalid_hashes() {
        let interner = StringInterner::new();

        // Test resolving non-existent hash
        let fake_hash = OxiStrHash::from_be_bytes([0xFF; 16]);
        assert_eq!(interner.resolve(&fake_hash), None);
    }

    #[test]
    fn test_error_conditions_decode_failures() -> Result<(), Box<dyn std::error::Error>> {
        let encoder = OptimizedTermEncoder::new();

        // Create an encoded term with a hash that doesn't exist in the interner
        let fake_hash = OxiStrHash::from_be_bytes([0xFF; 16]);
        let encoded = OxiEncodedTerm::NamedNode { iri: fake_hash };

        // This should fail when trying to decode
        assert!(encoder.decode_term(&encoded).is_err());

        Ok(())
    }

    #[test]
    fn test_numeric_literal_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        let encoder = OptimizedTermEncoder::new();

        // Test integer boundary values
        let max_int = Literal::new_typed_literal(
            i64::MAX.to_string(),
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?,
        );
        let encoded = encoder.encode_literal(&max_int);
        assert!(matches!(encoded, OxiEncodedTerm::IntegerLiteral(i64::MAX)));

        let min_int = Literal::new_typed_literal(
            i64::MIN.to_string(),
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?,
        );
        let encoded = encoder.encode_literal(&min_int);
        assert!(matches!(encoded, OxiEncodedTerm::IntegerLiteral(i64::MIN)));

        // Test float special values
        let nan_float = Literal::new_typed_literal(
            "NaN",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#float")?,
        );
        let encoded = encoder.encode_literal(&nan_float);
        if let OxiEncodedTerm::FloatLiteral(val) = encoded {
            assert!(val.is_nan());
        } else {
            panic!("Expected FloatLiteral");
        }

        let inf_float = Literal::new_typed_literal(
            "INF",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#float")?,
        );
        let encoded = encoder.encode_literal(&inf_float);
        if let OxiEncodedTerm::FloatLiteral(val) = encoded {
            assert!(val.is_infinite() && val.is_sign_positive());
        } else {
            panic!("Expected FloatLiteral");
        }

        Ok(())
    }

    #[test]
    fn test_invalid_numeric_literals() -> Result<(), Box<dyn std::error::Error>> {
        let encoder = OptimizedTermEncoder::new();

        // Test invalid integer (should fall back to general literal encoding)
        let invalid_int = Literal::new_typed_literal(
            "not_a_number",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?,
        );
        let encoded = encoder.encode_literal(&invalid_int);
        assert!(matches!(encoded, OxiEncodedTerm::Literal { .. }));

        // Test invalid float (should fall back to general literal encoding)
        let invalid_float = Literal::new_typed_literal(
            "not_a_float",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#float")?,
        );
        let encoded = encoder.encode_literal(&invalid_float);
        assert!(matches!(encoded, OxiEncodedTerm::Literal { .. }));

        Ok(())
    }

    #[test]
    fn test_memory_efficiency() {
        let mut interner = StringInterner::new();

        // Intern many duplicate strings to test memory efficiency
        let test_string = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        let num_duplicates = 10000;

        for _ in 0..num_duplicates {
            interner.intern(test_string);
        }

        let stats = interner.stats();
        assert_eq!(stats.total_strings, 1); // Only one unique string
        assert_eq!(stats.deduplication_saves, num_duplicates - 1);

        // Memory usage should be just the size of one string
        assert_eq!(stats.memory_usage, test_string.len());
    }

    #[test]
    fn test_concurrent_safety_simulation() {
        use std::sync::Arc;
        use std::thread;

        let encoder = Arc::new(OptimizedTermEncoder::new());
        let test_strings = vec![
            "http://example.org/test1",
            "http://example.org/test2",
            "http://example.org/test3",
        ];

        let handles: Vec<_> = test_strings
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                let encoder = Arc::clone(&encoder);
                let s = s.to_string();
                thread::spawn(move || {
                    // Simulate concurrent access
                    let named_node = NamedNode::new(&s).unwrap();
                    let encoded = encoder.encode_named_node(&named_node);
                    (i, encoded)
                })
            })
            .collect();

        // Wait for all threads and collect results
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results.len(), 3);

        // Verify all encodings are valid
        for (_, encoded) in results {
            assert!(matches!(encoded, OxiEncodedTerm::NamedNode { .. }));
        }
    }
}
