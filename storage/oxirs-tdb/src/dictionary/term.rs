//! RDF Term representation for dictionary encoding

use oxicode::Decode;
use serde::{Deserialize, Serialize};
use std::fmt;

/// RDF Term (IRI, Literal, or Blank Node)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Term {
    /// IRI (Internationalized Resource Identifier)
    Iri(String),

    /// Literal value with optional language tag or datatype
    Literal {
        /// The literal value
        value: String,
        /// Optional language tag (e.g., "en", "fr")
        language: Option<String>,
        /// Optional datatype IRI
        datatype: Option<String>,
    },

    /// Blank node with identifier
    BlankNode(String),
}

impl Term {
    /// Create an IRI term
    pub fn iri<S: Into<String>>(iri: S) -> Self {
        Term::Iri(iri.into())
    }

    /// Create a simple literal (string without language or datatype)
    pub fn literal<S: Into<String>>(value: S) -> Self {
        Term::Literal {
            value: value.into(),
            language: None,
            datatype: None,
        }
    }

    /// Create a literal with language tag
    pub fn literal_with_lang<S: Into<String>, L: Into<String>>(value: S, lang: L) -> Self {
        Term::Literal {
            value: value.into(),
            language: Some(lang.into()),
            datatype: None,
        }
    }

    /// Create a literal with datatype
    pub fn literal_with_datatype<S: Into<String>, D: Into<String>>(value: S, datatype: D) -> Self {
        Term::Literal {
            value: value.into(),
            language: None,
            datatype: Some(datatype.into()),
        }
    }

    /// Create a blank node
    pub fn blank_node<S: Into<String>>(id: S) -> Self {
        Term::BlankNode(id.into())
    }

    /// Check if this is an IRI
    pub fn is_iri(&self) -> bool {
        matches!(self, Term::Iri(_))
    }

    /// Check if this is a literal
    pub fn is_literal(&self) -> bool {
        matches!(self, Term::Literal { .. })
    }

    /// Check if this is a blank node
    pub fn is_blank_node(&self) -> bool {
        matches!(self, Term::BlankNode(_))
    }

    /// Get the IRI value if this is an IRI
    pub fn as_iri(&self) -> Option<&str> {
        match self {
            Term::Iri(iri) => Some(iri.as_str()),
            _ => None,
        }
    }

    /// Get the literal value if this is a literal
    pub fn as_literal(&self) -> Option<&str> {
        match self {
            Term::Literal { value, .. } => Some(value.as_str()),
            _ => None,
        }
    }

    /// Estimate serialized size in bytes
    pub fn estimated_size(&self) -> usize {
        match self {
            Term::Iri(iri) => 1 + iri.len(),
            Term::Literal {
                value,
                language,
                datatype,
            } => {
                1 + value.len()
                    + language.as_ref().map_or(0, |l| l.len())
                    + datatype.as_ref().map_or(0, |d| d.len())
            }
            Term::BlankNode(id) => 1 + id.len(),
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Iri(iri) => write!(f, "<{}>", iri),
            Term::Literal {
                value,
                language: Some(lang),
                ..
            } => {
                write!(f, "\"{}\"@{}", value, lang)
            }
            Term::Literal {
                value,
                datatype: Some(dt),
                ..
            } => {
                write!(f, "\"{}\"^^<{}>", value, dt)
            }
            Term::Literal { value, .. } => write!(f, "\"{}\"", value),
            Term::BlankNode(id) => write!(f, "_:{}", id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_iri() {
        let term = Term::iri("http://example.org/resource");
        assert!(term.is_iri());
        assert!(!term.is_literal());
        assert!(!term.is_blank_node());
        assert_eq!(term.as_iri(), Some("http://example.org/resource"));
    }

    #[test]
    fn test_term_literal() {
        let term = Term::literal("Hello World");
        assert!(term.is_literal());
        assert!(!term.is_iri());
        assert_eq!(term.as_literal(), Some("Hello World"));
    }

    #[test]
    fn test_term_literal_with_lang() {
        let term = Term::literal_with_lang("Hello", "en");
        assert!(term.is_literal());

        if let Term::Literal { language, .. } = &term {
            assert_eq!(language.as_deref(), Some("en"));
        } else {
            panic!("Expected literal");
        }
    }

    #[test]
    fn test_term_literal_with_datatype() {
        let term = Term::literal_with_datatype("42", "http://www.w3.org/2001/XMLSchema#integer");

        if let Term::Literal { datatype, .. } = &term {
            assert_eq!(
                datatype.as_deref(),
                Some("http://www.w3.org/2001/XMLSchema#integer")
            );
        } else {
            panic!("Expected literal");
        }
    }

    #[test]
    fn test_term_blank_node() {
        let term = Term::blank_node("b0");
        assert!(term.is_blank_node());
        assert!(!term.is_iri());
        assert!(!term.is_literal());
    }

    #[test]
    fn test_term_ordering() {
        let term1 = Term::iri("http://a.com");
        let term2 = Term::iri("http://b.com");
        assert!(term1 < term2);
    }

    #[test]
    fn test_term_serialization() {
        let term = Term::literal_with_lang("Bonjour", "fr");
        let serialized = oxicode::serde::encode_to_vec(&term, oxicode::config::standard()).unwrap();
        let deserialized: Term =
            oxicode::serde::decode_from_slice(&serialized, oxicode::config::standard())
                .unwrap()
                .0;
        assert_eq!(term, deserialized);
    }

    #[test]
    fn test_term_display() {
        assert_eq!(
            Term::iri("http://example.org").to_string(),
            "<http://example.org>"
        );
        assert_eq!(Term::literal("test").to_string(), "\"test\"");
        assert_eq!(
            Term::literal_with_lang("test", "en").to_string(),
            "\"test\"@en"
        );
        assert_eq!(Term::blank_node("b1").to_string(), "_:b1");
    }
}
