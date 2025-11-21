//! Shared types for Turtle parsing and serialization
//!
//! This module contains common types used by both parser and serializer.

use crate::error::TextPosition;
use crate::toolkit::StringInterner;
use oxirs_core::model::{NamedNode, Triple};
use std::collections::HashMap;

/// Parsing context for Turtle documents
#[derive(Debug, Clone)]
pub struct TurtleParsingContext {
    /// Prefix declarations
    pub prefixes: HashMap<String, String>,
    /// Base IRI for resolving relative IRIs
    pub base_iri: Option<String>,
    /// Blank node counter
    pub blank_node_counter: usize,
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
            pending_triples: Vec::new(),
            string_interner: StringInterner::with_common_namespaces(),
        }
    }

    /// Generate a unique blank node ID
    pub(crate) fn generate_blank_node_id(&mut self) -> String {
        let id = format!("_:b{}", self.blank_node_counter);
        self.blank_node_counter += 1;
        id
    }

    /// Resolve an IRI against the base IRI
    pub(crate) fn resolve_iri(&self, iri: &str) -> String {
        if let Some(ref base) = self.base_iri {
            // Simplified IRI resolution
            if iri.contains(':') {
                iri.to_string() // Absolute IRI
            } else {
                format!("{base}{iri}") // Relative IRI
            }
        } else {
            iri.to_string()
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
    /// @prefix keyword
    PrefixKeyword,
    /// @base keyword
    BaseKeyword,
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
