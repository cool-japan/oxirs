//! Efficient encoding for RDF terms
//!
//! This implementation is extracted and adapted from Oxigraph's numeric_encoder.rs
//! to provide zero-dependency term encoding with hash-based optimization.

use crate::model::{
    BlankNode, BlankNodeRef, Literal, LiteralRef, NamedNode, NamedNodeRef, Term, TermRef,
};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};

/// A hash of a string for efficient storage and comparison
#[derive(Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StrHash {
    hash: [u8; 16],
}

impl StrHash {
    /// Creates a new string hash using a fast hashing algorithm
    pub fn new(value: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        hasher.write(value.as_bytes());
        let hash_value = hasher.finish();

        // Create a 16-byte hash by using the 8-byte hash twice with different salts
        let mut full_hash = [0u8; 16];
        full_hash[0..8].copy_from_slice(&hash_value.to_be_bytes());

        // Create second hash with salt for better distribution
        let mut hasher2 = DefaultHasher::new();
        hasher2.write(&[0xDE, 0xAD, 0xBE, 0xEF]); // Salt
        hasher2.write(value.as_bytes());
        let hash_value2 = hasher2.finish();
        full_hash[8..16].copy_from_slice(&hash_value2.to_be_bytes());

        Self { hash: full_hash }
    }

    /// Creates a StrHash from raw bytes
    #[inline]
    pub fn from_be_bytes(hash: [u8; 16]) -> Self {
        Self { hash }
    }

    /// Returns the hash as raw bytes
    #[inline]
    pub fn to_be_bytes(self) -> [u8; 16] {
        self.hash
    }
}

impl Hash for StrHash {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use the first 8 bytes as the hash value
        let hash_val = u64::from_be_bytes([
            self.hash[0],
            self.hash[1],
            self.hash[2],
            self.hash[3],
            self.hash[4],
            self.hash[5],
            self.hash[6],
            self.hash[7],
        ]);
        state.write_u64(hash_val);
    }
}

impl Display for StrHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StrHash({})", hex::encode(self.hash))
    }
}

/// Small string optimization for commonly used strings
/// Stores strings up to 15 bytes inline without allocation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SmallString {
    data: [u8; 16],
    len: u8,
}

impl SmallString {
    const MAX_INLINE_LEN: usize = 15;

    /// Creates a new SmallString from a string slice
    pub fn new(s: &str) -> Option<Self> {
        if s.len() > Self::MAX_INLINE_LEN {
            return None;
        }

        let mut data = [0u8; 16];
        data[..s.len()].copy_from_slice(s.as_bytes());

        Some(SmallString {
            data,
            len: s.len() as u8,
        })
    }

    /// Returns the string slice
    pub fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.data[..self.len as usize]) }
    }

    /// Returns the length of the string
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns true if the string is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Display for SmallString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for SmallString {
    fn from(s: &str) -> Self {
        Self::new(s).expect("String too long for SmallString")
    }
}

/// Encoded term representation for efficient storage
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EncodedTerm {
    /// Default graph (for quad-based stores)
    DefaultGraph,

    /// Named node with hashed IRI
    NamedNode { iri_id: StrHash },

    /// Blank node with numerical ID (16 bytes for uniqueness)
    NumericalBlankNode { id: [u8; 16] },

    /// Small blank node with inline string
    SmallBlankNode(SmallString),

    /// Large blank node with hashed ID
    BigBlankNode { id_id: StrHash },

    /// Small string literal (inline)
    SmallStringLiteral(SmallString),

    /// Large string literal (hashed)
    BigStringLiteral { value_id: StrHash },

    /// Small value, small language tag
    SmallSmallLangStringLiteral {
        value: SmallString,
        language: SmallString,
    },

    /// Small value, large language tag
    SmallBigLangStringLiteral {
        value: SmallString,
        language_id: StrHash,
    },

    /// Large value, small language tag
    BigSmallLangStringLiteral {
        value_id: StrHash,
        language: SmallString,
    },

    /// Large value, large language tag
    BigBigLangStringLiteral {
        value_id: StrHash,
        language_id: StrHash,
    },

    /// Typed literal with small value and small datatype
    SmallSmallTypedLiteral {
        value: SmallString,
        datatype: SmallString,
    },

    /// Typed literal with small value and hashed datatype
    SmallBigTypedLiteral {
        value: SmallString,
        datatype_id: StrHash,
    },

    /// Typed literal with hashed value and small datatype
    BigSmallTypedLiteral {
        value_id: StrHash,
        datatype: SmallString,
    },

    /// Typed literal with hashed value and hashed datatype
    BigBigTypedLiteral {
        value_id: StrHash,
        datatype_id: StrHash,
    },
}

impl EncodedTerm {
    /// Encodes a named node
    pub fn encode_named_node(node: &NamedNode) -> Self {
        EncodedTerm::NamedNode {
            iri_id: StrHash::new(node.as_str()),
        }
    }

    /// Encodes a named node reference
    pub fn encode_named_node_ref(node: NamedNodeRef<'_>) -> Self {
        EncodedTerm::NamedNode {
            iri_id: StrHash::new(node.as_str()),
        }
    }

    /// Encodes a blank node
    pub fn encode_blank_node(node: &BlankNode) -> Self {
        let id_str = node.as_str();

        // Try to use numerical representation if it's a hex ID
        if let Ok(bytes) = hex::decode(id_str) {
            if bytes.len() == 16 {
                let mut id = [0u8; 16];
                id.copy_from_slice(&bytes);
                return EncodedTerm::NumericalBlankNode { id };
            }
        }

        // Use string representation
        if let Some(small_string) = SmallString::new(id_str) {
            EncodedTerm::SmallBlankNode(small_string)
        } else {
            EncodedTerm::BigBlankNode {
                id_id: StrHash::new(id_str),
            }
        }
    }

    /// Encodes a blank node reference
    pub fn encode_blank_node_ref(node: BlankNodeRef<'_>) -> Self {
        let id_str = node.as_str();

        // Try to use numerical representation if it's a hex ID
        if let Ok(bytes) = hex::decode(id_str) {
            if bytes.len() == 16 {
                let mut id = [0u8; 16];
                id.copy_from_slice(&bytes);
                return EncodedTerm::NumericalBlankNode { id };
            }
        }

        // Use string representation
        if let Some(small_string) = SmallString::new(id_str) {
            EncodedTerm::SmallBlankNode(small_string)
        } else {
            EncodedTerm::BigBlankNode {
                id_id: StrHash::new(id_str),
            }
        }
    }

    /// Encodes a literal
    pub fn encode_literal(literal: &Literal) -> Self {
        let value = literal.value();

        if let Some(language) = literal.language() {
            // Language-tagged string literal
            match (SmallString::new(value), SmallString::new(language)) {
                (Some(small_value), Some(small_lang)) => EncodedTerm::SmallSmallLangStringLiteral {
                    value: small_value,
                    language: small_lang,
                },
                (Some(small_value), None) => EncodedTerm::SmallBigLangStringLiteral {
                    value: small_value,
                    language_id: StrHash::new(language),
                },
                (None, Some(small_lang)) => EncodedTerm::BigSmallLangStringLiteral {
                    value_id: StrHash::new(value),
                    language: small_lang,
                },
                (None, None) => EncodedTerm::BigBigLangStringLiteral {
                    value_id: StrHash::new(value),
                    language_id: StrHash::new(language),
                },
            }
        } else {
            let datatype = literal.datatype();
            let datatype_str = datatype.as_str();

            // Check if it's a simple string literal (xsd:string)
            if datatype_str == "http://www.w3.org/2001/XMLSchema#string" {
                if let Some(small_value) = SmallString::new(value) {
                    EncodedTerm::SmallStringLiteral(small_value)
                } else {
                    EncodedTerm::BigStringLiteral {
                        value_id: StrHash::new(value),
                    }
                }
            } else {
                // Typed literal
                match (SmallString::new(value), SmallString::new(datatype_str)) {
                    (Some(small_value), Some(small_datatype)) => {
                        EncodedTerm::SmallSmallTypedLiteral {
                            value: small_value,
                            datatype: small_datatype,
                        }
                    }
                    (Some(small_value), None) => EncodedTerm::SmallBigTypedLiteral {
                        value: small_value,
                        datatype_id: StrHash::new(datatype_str),
                    },
                    (None, Some(small_datatype)) => EncodedTerm::BigSmallTypedLiteral {
                        value_id: StrHash::new(value),
                        datatype: small_datatype,
                    },
                    (None, None) => EncodedTerm::BigBigTypedLiteral {
                        value_id: StrHash::new(value),
                        datatype_id: StrHash::new(datatype_str),
                    },
                }
            }
        }
    }

    /// Encodes a literal reference
    pub fn encode_literal_ref(literal: LiteralRef<'_>) -> Self {
        let value = literal.value();

        if let Some(language) = literal.language() {
            // Language-tagged string literal
            match (SmallString::new(value), SmallString::new(language)) {
                (Some(small_value), Some(small_lang)) => EncodedTerm::SmallSmallLangStringLiteral {
                    value: small_value,
                    language: small_lang,
                },
                (Some(small_value), None) => EncodedTerm::SmallBigLangStringLiteral {
                    value: small_value,
                    language_id: StrHash::new(language),
                },
                (None, Some(small_lang)) => EncodedTerm::BigSmallLangStringLiteral {
                    value_id: StrHash::new(value),
                    language: small_lang,
                },
                (None, None) => EncodedTerm::BigBigLangStringLiteral {
                    value_id: StrHash::new(value),
                    language_id: StrHash::new(language),
                },
            }
        } else {
            let datatype = literal.datatype();
            let datatype_str = datatype.as_str();

            // Check if it's a simple string literal (xsd:string)
            if datatype_str == "http://www.w3.org/2001/XMLSchema#string" {
                if let Some(small_value) = SmallString::new(value) {
                    EncodedTerm::SmallStringLiteral(small_value)
                } else {
                    EncodedTerm::BigStringLiteral {
                        value_id: StrHash::new(value),
                    }
                }
            } else {
                // Typed literal
                match (SmallString::new(value), SmallString::new(datatype_str)) {
                    (Some(small_value), Some(small_datatype)) => {
                        EncodedTerm::SmallSmallTypedLiteral {
                            value: small_value,
                            datatype: small_datatype,
                        }
                    }
                    (Some(small_value), None) => EncodedTerm::SmallBigTypedLiteral {
                        value: small_value,
                        datatype_id: StrHash::new(datatype_str),
                    },
                    (None, Some(small_datatype)) => EncodedTerm::BigSmallTypedLiteral {
                        value_id: StrHash::new(value),
                        datatype: small_datatype,
                    },
                    (None, None) => EncodedTerm::BigBigTypedLiteral {
                        value_id: StrHash::new(value),
                        datatype_id: StrHash::new(datatype_str),
                    },
                }
            }
        }
    }

    /// Encodes a variable (currently not supported in storage context)
    pub fn encode_variable(_variable: &crate::model::Variable) -> Self {
        panic!("Variables cannot be encoded for storage - they are only used in queries")
    }

    /// Encodes any term
    pub fn encode_term(term: &Term) -> Self {
        match term {
            Term::NamedNode(n) => Self::encode_named_node(n),
            Term::BlankNode(b) => Self::encode_blank_node(b),
            Term::Literal(l) => Self::encode_literal(l),
            Term::Variable(_) => panic!("Cannot encode variable in this context"),
            Term::QuotedTriple(_) => todo!("RDF-star encoding not yet implemented"),
        }
    }

    /// Encodes any term reference
    pub fn encode_term_ref(term: TermRef<'_>) -> Self {
        match term {
            TermRef::NamedNode(n) => Self::encode_named_node_ref(n),
            TermRef::BlankNode(b) => Self::encode_blank_node_ref(b),
            TermRef::Literal(l) => Self::encode_literal_ref(l),
            TermRef::Variable(v) => Self::encode_variable(v),
            #[cfg(feature = "rdf-star")]
            TermRef::Triple(_) => todo!("RDF-star encoding not yet implemented"),
        }
    }

    /// Returns the type discriminant for sorting and indexing
    pub fn type_discriminant(&self) -> u8 {
        match self {
            EncodedTerm::DefaultGraph => 0,
            EncodedTerm::NamedNode { .. } => 1,
            EncodedTerm::NumericalBlankNode { .. } => 2,
            EncodedTerm::SmallBlankNode(_) => 3,
            EncodedTerm::BigBlankNode { .. } => 4,
            EncodedTerm::SmallStringLiteral(_) => 5,
            EncodedTerm::BigStringLiteral { .. } => 6,
            EncodedTerm::SmallSmallLangStringLiteral { .. } => 7,
            EncodedTerm::SmallBigLangStringLiteral { .. } => 8,
            EncodedTerm::BigSmallLangStringLiteral { .. } => 9,
            EncodedTerm::BigBigLangStringLiteral { .. } => 10,
            EncodedTerm::SmallSmallTypedLiteral { .. } => 11,
            EncodedTerm::SmallBigTypedLiteral { .. } => 12,
            EncodedTerm::BigSmallTypedLiteral { .. } => 13,
            EncodedTerm::BigBigTypedLiteral { .. } => 14,
        }
    }

    /// Returns true if this is a named node
    pub fn is_named_node(&self) -> bool {
        matches!(self, EncodedTerm::NamedNode { .. })
    }

    /// Returns true if this is a blank node
    pub fn is_blank_node(&self) -> bool {
        matches!(
            self,
            EncodedTerm::NumericalBlankNode { .. }
                | EncodedTerm::SmallBlankNode(_)
                | EncodedTerm::BigBlankNode { .. }
        )
    }

    /// Returns true if this is a literal
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            EncodedTerm::SmallStringLiteral(_)
                | EncodedTerm::BigStringLiteral { .. }
                | EncodedTerm::SmallSmallLangStringLiteral { .. }
                | EncodedTerm::SmallBigLangStringLiteral { .. }
                | EncodedTerm::BigSmallLangStringLiteral { .. }
                | EncodedTerm::BigBigLangStringLiteral { .. }
                | EncodedTerm::SmallSmallTypedLiteral { .. }
                | EncodedTerm::SmallBigTypedLiteral { .. }
                | EncodedTerm::BigSmallTypedLiteral { .. }
                | EncodedTerm::BigBigTypedLiteral { .. }
        )
    }

    /// Returns the size in bytes of this encoded term
    pub fn size_hint(&self) -> usize {
        match self {
            EncodedTerm::DefaultGraph => 1,
            EncodedTerm::NamedNode { .. } => 1 + 16,
            EncodedTerm::NumericalBlankNode { .. } => 1 + 16,
            EncodedTerm::SmallBlankNode(_) => 1 + 16 + 1,
            EncodedTerm::BigBlankNode { .. } => 1 + 16,
            EncodedTerm::SmallStringLiteral(_) => 1 + 16 + 1,
            EncodedTerm::BigStringLiteral { .. } => 1 + 16,
            EncodedTerm::SmallSmallLangStringLiteral { .. } => 1 + 16 + 1 + 16 + 1,
            EncodedTerm::SmallBigLangStringLiteral { .. } => 1 + 16 + 1 + 16,
            EncodedTerm::BigSmallLangStringLiteral { .. } => 1 + 16 + 16 + 1,
            EncodedTerm::BigBigLangStringLiteral { .. } => 1 + 16 + 16,
            EncodedTerm::SmallSmallTypedLiteral { .. } => 1 + 16 + 1 + 16 + 1,
            EncodedTerm::SmallBigTypedLiteral { .. } => 1 + 16 + 1 + 16,
            EncodedTerm::BigSmallTypedLiteral { .. } => 1 + 16 + 16 + 1,
            EncodedTerm::BigBigTypedLiteral { .. } => 1 + 16 + 16,
        }
    }
}

/// Encoded triple for efficient storage and indexing
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EncodedTriple {
    pub subject: EncodedTerm,
    pub predicate: EncodedTerm,
    pub object: EncodedTerm,
}

impl EncodedTriple {
    /// Creates a new encoded triple
    pub fn new(subject: EncodedTerm, predicate: EncodedTerm, object: EncodedTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Returns the size hint for this triple
    pub fn size_hint(&self) -> usize {
        self.subject.size_hint() + self.predicate.size_hint() + self.object.size_hint()
    }
}

/// Encoded quad for efficient storage and indexing
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EncodedQuad {
    pub subject: EncodedTerm,
    pub predicate: EncodedTerm,
    pub object: EncodedTerm,
    pub graph_name: EncodedTerm,
}

impl EncodedQuad {
    /// Creates a new encoded quad
    pub fn new(
        subject: EncodedTerm,
        predicate: EncodedTerm,
        object: EncodedTerm,
        graph_name: EncodedTerm,
    ) -> Self {
        Self {
            subject,
            predicate,
            object,
            graph_name,
        }
    }

    /// Returns the size hint for this quad
    pub fn size_hint(&self) -> usize {
        self.subject.size_hint()
            + self.predicate.size_hint()
            + self.object.size_hint()
            + self.graph_name.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;
    use crate::vocab::xsd;

    #[test]
    fn test_str_hash() {
        let hash1 = StrHash::new("http://example.org/test");
        let hash2 = StrHash::new("http://example.org/test");
        let hash3 = StrHash::new("http://example.org/different");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_small_string() {
        let small = SmallString::new("test").unwrap();
        assert_eq!(small.as_str(), "test");
        assert_eq!(small.len(), 4);
        assert!(!small.is_empty());

        let empty = SmallString::new("").unwrap();
        assert!(empty.is_empty());

        // Test too long string
        let long_str = "this is a very long string that exceeds the maximum inline length";
        assert!(SmallString::new(long_str).is_none());
    }

    #[test]
    fn test_encode_named_node() {
        let node = NamedNode::new("http://example.org/test").unwrap();
        let encoded = EncodedTerm::encode_named_node(&node);

        assert!(encoded.is_named_node());
        assert!(!encoded.is_blank_node());
        assert!(!encoded.is_literal());
    }

    #[test]
    fn test_encode_blank_node() {
        let node = BlankNode::new("test").unwrap();
        let encoded = EncodedTerm::encode_blank_node(&node);

        assert!(!encoded.is_named_node());
        assert!(encoded.is_blank_node());
        assert!(!encoded.is_literal());
    }

    #[test]
    fn test_encode_literal() {
        // Simple string literal
        let literal = Literal::new("test");
        let encoded = EncodedTerm::encode_literal(&literal);
        assert!(encoded.is_literal());
        assert!(matches!(encoded, EncodedTerm::SmallStringLiteral(_)));

        // Language-tagged literal
        let literal = Literal::new_lang("test", "en").unwrap();
        let encoded = EncodedTerm::encode_literal(&literal);
        assert!(encoded.is_literal());
        assert!(matches!(
            encoded,
            EncodedTerm::SmallSmallLangStringLiteral { .. }
        ));

        // Typed literal
        let literal = Literal::new_typed("42", xsd::INTEGER.clone());
        let encoded = EncodedTerm::encode_literal(&literal);
        assert!(encoded.is_literal());
        assert!(matches!(encoded, EncodedTerm::SmallBigTypedLiteral { .. }));
    }

    #[test]
    fn test_encoded_triple() {
        let subject =
            EncodedTerm::encode_named_node(&NamedNode::new("http://example.org/s").unwrap());
        let predicate =
            EncodedTerm::encode_named_node(&NamedNode::new("http://example.org/p").unwrap());
        let object = EncodedTerm::encode_literal(&Literal::new("test"));

        let triple = EncodedTriple::new(subject, predicate, object);
        assert!(triple.size_hint() > 0);
    }

    #[test]
    fn test_type_discriminant() {
        let named_node = EncodedTerm::NamedNode {
            iri_id: StrHash::new("http://example.org/test"),
        };
        let blank_node = EncodedTerm::SmallBlankNode(SmallString::new("test").unwrap());
        let literal = EncodedTerm::SmallStringLiteral(SmallString::new("test").unwrap());

        assert_eq!(named_node.type_discriminant(), 1);
        assert_eq!(blank_node.type_discriminant(), 3);
        assert_eq!(literal.type_discriminant(), 5);
    }
}
