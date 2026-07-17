//! Shared types for Turtle parsing and serialization
//!
//! This module contains common types used by both parser and serializer.

use crate::error::TextPosition;
use crate::toolkit::iri_normalizer::resolve_reference;
use crate::toolkit::StringInterner;
use oxirs_core::model::{NamedNode, Triple};
use std::collections::{HashMap, HashSet};

/// Parsing context for Turtle documents
#[derive(Debug, Clone)]
pub struct TurtleParsingContext {
    /// Prefix declarations
    pub prefixes: HashMap<String, String>,
    /// Base IRI for resolving relative IRIs
    pub base_iri: Option<String>,
    /// Blank node counter used to mint fresh auto-generated blank node IDs
    pub blank_node_counter: usize,
    /// Blank node labels explicitly written by the user (without the `_:` prefix).
    ///
    /// Auto-generated blank node IDs are checked against this set so that a generated
    /// identifier can never collide with a label the document author wrote by hand
    /// (e.g. a document mixing `_:genid-b0` and an explicit `[...]` property list).
    pub explicit_blank_labels: HashSet<String>,
    /// Pending triples from blank node property lists
    pub pending_triples: Vec<Triple>,
    /// String interner for deduplicating IRIs, prefixes, and language tags
    pub string_interner: StringInterner,
}

impl TurtleParsingContext {
    /// Create a new parsing context
    pub(crate) fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base_iri: None,
            blank_node_counter: 0,
            explicit_blank_labels: HashSet::new(),
            pending_triples: Vec::new(),
            string_interner: StringInterner::with_common_namespaces(),
        }
    }

    /// Register a blank node label the user wrote explicitly (e.g. `_:foo`), so that
    /// subsequently auto-generated blank node IDs never collide with it.
    pub(crate) fn register_blank_label(&mut self, label: &str) {
        let clean = label.strip_prefix("_:").unwrap_or(label);
        self.explicit_blank_labels.insert(clean.to_string());
    }

    /// Generate a unique blank node ID.
    ///
    /// Generated IDs live in a `genid-b{N}` namespace that is very unlikely to be used by
    /// hand-written labels; on top of that, every candidate is checked against
    /// [`explicit_blank_labels`](Self::explicit_blank_labels) and skipped on collision, so the
    /// generated ID is guaranteed disjoint from every explicit label seen so far in the
    /// document.
    pub(crate) fn generate_blank_node_id(&mut self) -> String {
        loop {
            let candidate = format!("genid-b{}", self.blank_node_counter);
            self.blank_node_counter += 1;
            if !self.explicit_blank_labels.contains(&candidate) {
                return format!("_:{candidate}");
            }
        }
    }

    /// Resolve an IRI reference against the base IRI, per RFC 3986 §5.3.
    pub(crate) fn resolve_iri(&self, iri: &str) -> String {
        match &self.base_iri {
            Some(base) => resolve_reference(base, iri),
            None => iri.to_string(),
        }
    }

    /// Create a NamedNode with string interning for efficient memory usage
    pub(crate) fn create_named_node(
        &mut self,
        iri: &str,
    ) -> Result<NamedNode, oxirs_core::error::OxirsError> {
        // Intern the IRI to deduplicate common strings
        let interned = self.string_interner.intern(iri);
        // Create NamedNode with the interned string (dereference Arc to get &str)
        // Note: This still clones, but the interner reduces allocations overall
        NamedNode::new(interned.as_str())
    }
}

/// Statements in a Turtle document
#[derive(Debug, Clone)]
pub enum TurtleStatement {
    /// Single RDF triple statement
    Triple(Triple),
    /// Multiple RDF triples (from semicolon/comma syntax)
    Triples(Vec<Triple>),
    /// Prefix declaration statement
    PrefixDecl(String, String),
    /// Base URI declaration statement
    BaseDecl(String),
}

/// Token types for Turtle lexing
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    /// `@prefix` keyword (Turtle 1.1 form, terminated by `.`)
    PrefixKeyword,
    /// `@base` keyword (Turtle 1.1 form, terminated by `.`)
    BaseKeyword,
    /// `PREFIX` keyword (SPARQL-style, no terminating `.`)
    SparqlPrefixKeyword,
    /// `BASE` keyword (SPARQL-style, no terminating `.`)
    SparqlBaseKeyword,
    /// 'a' shorthand for rdf:type
    A,

    // Punctuation
    /// Dot punctuation (.)
    Dot,
    /// Semicolon punctuation (;)
    Semicolon,
    /// Comma punctuation (,)
    Comma,
    /// Left bracket ([)
    LeftBracket,
    /// Right bracket (])
    RightBracket,
    /// Left parenthesis (()
    LeftParen,
    /// Right parenthesis ())
    RightParen,
    /// Colon punctuation (:)
    Colon,
    /// Data type annotation (^^)
    DataTypeAnnotation,
    /// Double less-than for quoted triples (<<) - RDF 1.2
    DoubleLessThan,
    /// Double greater-than for quoted triples (>>) - RDF 1.2
    DoubleGreaterThan,

    // Literals and identifiers
    /// IRI reference enclosed in angle brackets
    IriRef(String),
    /// Prefixed name with prefix and local parts
    PrefixedName(String, String),
    /// Prefix name used in @prefix declarations
    PrefixName(String),
    /// Blank node label
    BlankNodeLabel(String),
    /// String literal value
    StringLiteral(String),
    /// Language tag for literals, with optional direction for RDF 1.2
    LanguageTag(String, Option<String>),
    /// Boolean literal (true or false)
    Boolean(bool),
    /// Integer literal
    Integer(String),
    /// Decimal literal
    Decimal(String),
    /// Double literal (scientific notation)
    Double(String),

    // Whitespace and comments
    /// Whitespace characters
    Whitespace,
    /// Comment text
    Comment(String),

    // End of input
    /// End of input marker
    Eof,
}

/// A token with position information
#[derive(Debug, Clone)]
pub struct Token {
    /// The type of token
    pub kind: TokenKind,
    /// Position in the input text
    pub position: TextPosition,
}
