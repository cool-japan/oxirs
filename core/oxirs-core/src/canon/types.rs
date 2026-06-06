//! Core types for URDNA2015 RDF Dataset Normalization.
//!
//! These types form the canonical data representation used throughout the
//! canonicalization pipeline. They are intentionally self-contained and do not
//! depend on the `oxirs_core::model` types so that the canon module can be used
//! independently or fed from external sources.

/// A term in an RDF quad: either an IRI, a blank node, or an RDF literal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QuadTerm {
    /// An IRI reference (the angle brackets are NOT stored; serialization adds them).
    Iri(String),
    /// A blank node identifier (the `_:` prefix is NOT stored).
    BlankNode(String),
    /// An RDF literal with an explicit datatype IRI.
    ///
    /// Language-tagged strings MUST set `language` to `Some(tag)` and set
    /// `datatype` to `rdf:langString`
    /// (`http://www.w3.org/1999/02/22-rdf-syntax-ns#langString`).
    Literal {
        /// The lexical form (unescaped).
        value: String,
        /// The datatype IRI (angle brackets NOT included).
        datatype: String,
        /// Optional BCP-47 language tag (lowercase).
        language: Option<String>,
    },
}

impl QuadTerm {
    /// Returns `true` if this term is a blank node.
    #[inline]
    pub fn is_blank_node(&self) -> bool {
        matches!(self, QuadTerm::BlankNode(_))
    }

    /// Returns the blank node identifier if this term is a blank node.
    #[inline]
    pub fn blank_node_id(&self) -> Option<&str> {
        match self {
            QuadTerm::BlankNode(id) => Some(id.as_str()),
            _ => None,
        }
    }

    /// Convenience constructor: create an IRI term.
    #[inline]
    pub fn iri(iri: impl Into<String>) -> Self {
        QuadTerm::Iri(iri.into())
    }

    /// Convenience constructor: create a blank node term.
    #[inline]
    pub fn blank(id: impl Into<String>) -> Self {
        QuadTerm::BlankNode(id.into())
    }

    /// Convenience constructor: create a plain literal (XSD string datatype).
    #[inline]
    pub fn string_literal(value: impl Into<String>) -> Self {
        QuadTerm::Literal {
            value: value.into(),
            datatype: "http://www.w3.org/2001/XMLSchema#string".to_string(),
            language: None,
        }
    }

    /// Convenience constructor: create a typed literal.
    #[inline]
    pub fn typed_literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        QuadTerm::Literal {
            value: value.into(),
            datatype: datatype.into(),
            language: None,
        }
    }

    /// Convenience constructor: create a language-tagged string literal.
    #[inline]
    pub fn lang_literal(value: impl Into<String>, lang: impl Into<String>) -> Self {
        QuadTerm::Literal {
            value: value.into(),
            datatype: "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".to_string(),
            language: Some(lang.into()),
        }
    }
}

/// An RDF quad (subject, predicate, object, optional named graph).
///
/// When `graph` is `None` the quad belongs to the default graph.
///
/// Per the N-Quads specification:
/// - `subject` MUST be an IRI or a blank node
/// - `predicate` MUST be an IRI
/// - `object` MAY be an IRI, blank node, or literal
/// - `graph` (if present) MUST be an IRI or blank node
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdfQuad {
    pub subject: QuadTerm,
    pub predicate: QuadTerm,
    pub object: QuadTerm,
    /// `None` means the default graph; `Some(_)` is a named graph.
    pub graph: Option<QuadTerm>,
}

impl RdfQuad {
    /// Construct a new quad in the default graph.
    pub fn new(subject: QuadTerm, predicate: QuadTerm, object: QuadTerm) -> Self {
        RdfQuad {
            subject,
            predicate,
            object,
            graph: None,
        }
    }

    /// Construct a new quad with a named graph.
    pub fn new_in_graph(
        subject: QuadTerm,
        predicate: QuadTerm,
        object: QuadTerm,
        graph: QuadTerm,
    ) -> Self {
        RdfQuad {
            subject,
            predicate,
            object,
            graph: Some(graph),
        }
    }

    /// Return all blank node identifiers appearing in this quad.
    pub fn blank_nodes(&self) -> Vec<&str> {
        let mut out = Vec::new();
        if let QuadTerm::BlankNode(id) = &self.subject {
            out.push(id.as_str());
        }
        // Predicate can be blank in RDF-star but standard RDF forbids it;
        // we check anyway for correctness.
        if let QuadTerm::BlankNode(id) = &self.predicate {
            out.push(id.as_str());
        }
        if let QuadTerm::BlankNode(id) = &self.object {
            out.push(id.as_str());
        }
        if let Some(QuadTerm::BlankNode(id)) = &self.graph {
            out.push(id.as_str());
        }
        out
    }
}
