//! Error types for Turtle-family format parsing
//!
//! This module provides comprehensive error handling for parsing and serialization
//! operations, including position tracking and error recovery capabilities.

use std::fmt;
use thiserror::Error;

/// Position in a text document
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextPosition {
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based) 
    pub column: usize,
    /// Byte offset from start of document
    pub offset: usize,
}

impl Default for TextPosition {
    fn default() -> Self {
        Self::start()
    }
}

impl TextPosition {
    /// Create a new text position
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self { line, column, offset }
    }
    
    /// Initial position at start of document
    pub fn start() -> Self {
        Self::new(1, 1, 0)
    }
    
    /// Advance position by one character
    pub fn advance_char(&mut self, ch: char) {
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        self.offset += ch.len_utf8();
    }
    
    /// Advance position by multiple bytes
    pub fn advance_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            if byte == b'\n' {
                self.line += 1;
                self.column = 1;
            } else if byte >= 0x20 || byte == b'\t' {
                self.column += 1;
            }
            self.offset += 1;
        }
    }
}

impl fmt::Display for TextPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

/// Syntax error in Turtle-family format
#[derive(Debug, Clone, Error)]
pub enum TurtleSyntaxError {
    /// Unexpected character
    #[error("Unexpected character '{character}' at {position}")]
    UnexpectedCharacter {
        /// The unexpected character
        character: char,
        /// Position where error occurred
        position: TextPosition,
    },
    
    /// Unexpected end of input
    #[error("Unexpected end of input at {position}")]
    UnexpectedEof {
        /// Position where EOF was encountered
        position: TextPosition,
    },
    
    /// Invalid IRI
    #[error("Invalid IRI '{iri}' at {position}: {reason}")]
    InvalidIri {
        /// The invalid IRI
        iri: String,
        /// Reason for invalidity
        reason: String,
        /// Position of the IRI
        position: TextPosition,
    },
    
    /// Invalid language tag
    #[error("Invalid language tag '{tag}' at {position}: {reason}")]
    InvalidLanguageTag {
        /// The invalid language tag
        tag: String,
        /// Reason for invalidity
        reason: String,
        /// Position of the tag
        position: TextPosition,
    },
    
    /// Invalid literal
    #[error("Invalid literal '{literal}' at {position}: {reason}")]
    InvalidLiteral {
        /// The invalid literal
        literal: String,
        /// Reason for invalidity
        reason: String,
        /// Position of the literal
        position: TextPosition,
    },
    
    /// Invalid escape sequence
    #[error("Invalid escape sequence '\\{sequence}' at {position}")]
    InvalidEscape {
        /// The invalid escape sequence (without backslash)
        sequence: String,
        /// Position of the escape
        position: TextPosition,
    },
    
    /// Invalid Unicode code point
    #[error("Invalid Unicode code point U+{codepoint:04X} at {position}")]
    InvalidUnicode {
        /// The invalid code point
        codepoint: u32,
        /// Position of the code point
        position: TextPosition,
    },
    
    /// Invalid blank node identifier
    #[error("Invalid blank node identifier '{id}' at {position}")]
    InvalidBlankNode {
        /// The invalid identifier
        id: String,
        /// Position of the identifier
        position: TextPosition,
    },
    
    /// Undefined prefix
    #[error("Undefined prefix '{prefix}' at {position}")]
    UndefinedPrefix {
        /// The undefined prefix
        prefix: String,
        /// Position where prefix was used
        position: TextPosition,
    },
    
    /// Invalid prefix declaration
    #[error("Invalid prefix declaration for '{prefix}' at {position}: {reason}")]
    InvalidPrefix {
        /// The prefix being declared
        prefix: String,
        /// Reason for invalidity
        reason: String,
        /// Position of the declaration
        position: TextPosition,
    },
    
    /// Invalid base IRI declaration
    #[error("Invalid base IRI '{iri}' at {position}: {reason}")]
    InvalidBase {
        /// The invalid base IRI
        iri: String,
        /// Reason for invalidity
        reason: String,
        /// Position of the declaration
        position: TextPosition,
    },
    
    /// Generic syntax error
    #[error("Syntax error at {position}: {message}")]
    Generic {
        /// Error message
        message: String,
        /// Position where error occurred
        position: TextPosition,
    },
}

/// High-level parsing error
#[derive(Debug, Error)]
pub enum TurtleParseError {
    /// Syntax error in the input
    #[error("Syntax error: {0}")]
    Syntax(#[from] TurtleSyntaxError),
    
    /// I/O error while reading
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// RDF model error (invalid terms, etc.)
    #[error("RDF model error: {0}")]
    Model(#[from] oxirs_core::OxirsError),
    
    /// Multiple errors (for batch processing)
    #[error("Multiple errors occurred ({} errors)", .errors.len())]
    Multiple {
        /// Collection of errors
        errors: Vec<TurtleParseError>,
    },
}

/// Result type for parsing operations
pub type TurtleResult<T> = Result<T, TurtleParseError>;

/// Error that can occur during tokenization
#[derive(Debug, Clone, Error)]
pub enum TokenRecognizerError {
    /// Unexpected character
    #[error("Unexpected character: {0}")]
    UnexpectedCharacter(char),
    
    /// Unexpected end of input
    #[error("Unexpected end of input")]
    UnexpectedEof,
    
    /// Invalid token
    #[error("Invalid token: {0}")]
    Invalid(String),
}

/// Error that can occur during rule recognition
#[derive(Debug, Clone, Error)]
pub enum RuleRecognizerError {
    /// Unexpected token
    #[error("Unexpected token: {0}")]
    UnexpectedToken(String),
    
    /// Missing required token
    #[error("Missing required token: {0}")]
    MissingToken(String),
    
    /// Invalid rule application
    #[error("Invalid rule: {0}")]
    InvalidRule(String),
}

impl TurtleParseError {
    /// Create a new syntax error
    pub fn syntax(error: TurtleSyntaxError) -> Self {
        Self::Syntax(error)
    }
    
    /// Create a new I/O error
    pub fn io(error: std::io::Error) -> Self {
        Self::Io(error)
    }
    
    /// Create a new model error
    pub fn model(error: oxirs_core::OxirsError) -> Self {
        Self::Model(error)
    }
    
    /// Combine multiple errors
    pub fn multiple(errors: Vec<TurtleParseError>) -> Self {
        Self::Multiple { errors }
    }
    
    /// Get the position of this error, if available
    pub fn position(&self) -> Option<TextPosition> {
        match self {
            Self::Syntax(syntax_error) => Some(syntax_error.position()),
            _ => None,
        }
    }
}

impl TurtleSyntaxError {
    /// Get the position where this error occurred
    pub fn position(&self) -> TextPosition {
        match self {
            Self::UnexpectedCharacter { position, .. } => *position,
            Self::UnexpectedEof { position } => *position,
            Self::InvalidIri { position, .. } => *position,
            Self::InvalidLanguageTag { position, .. } => *position,
            Self::InvalidLiteral { position, .. } => *position,
            Self::InvalidEscape { position, .. } => *position,
            Self::InvalidUnicode { position, .. } => *position,
            Self::InvalidBlankNode { position, .. } => *position,
            Self::UndefinedPrefix { position, .. } => *position,
            Self::InvalidPrefix { position, .. } => *position,
            Self::InvalidBase { position, .. } => *position,
            Self::Generic { position, .. } => *position,
        }
    }
}