//! Error types for SAMM operations

use std::fmt;

/// Result type for SAMM operations
pub type Result<T> = std::result::Result<T, SammError>;

/// Source location information for error reporting
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub column: usize,
    /// File path or URI
    pub source: Option<String>,
}

impl SourceLocation {
    /// Create a new source location
    pub fn new(line: usize, column: usize) -> Self {
        Self {
            line,
            column,
            source: None,
        }
    }

    /// Create a source location with file path
    pub fn with_source(line: usize, column: usize, source: String) -> Self {
        Self {
            line,
            column,
            source: Some(source),
        }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(source) = &self.source {
            write!(f, "{}:{}:{}", source, self.line, self.column)
        } else {
            write!(f, "line {}:{}", self.line, self.column)
        }
    }
}

/// Error types for SAMM operations
#[derive(Debug, thiserror::Error)]
pub enum SammError {
    /// Error parsing a SAMM model
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Parse error with source location
    #[error("Parse error at {location}: {message}")]
    ParseErrorWithLocation {
        /// Error message
        message: String,
        /// Source location
        location: SourceLocation,
    },

    /// Error validating a SAMM model
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Validation error with source location
    #[error("Validation error at {location}: {message}")]
    ValidationErrorWithLocation {
        /// Error message
        message: String,
        /// Source location
        location: SourceLocation,
    },

    /// Error resolving a model element
    #[error("Resolution error: {0}")]
    ResolutionError(String),

    /// Invalid URN format
    #[error("Invalid URN: {0}")]
    InvalidUrn(String),

    /// Missing required element
    #[error("Missing required element: {0}")]
    MissingElement(String),

    /// Invalid model structure
    #[error("Invalid model structure: {0}")]
    InvalidStructure(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// RDF error
    #[error("RDF error: {0}")]
    Rdf(String),

    /// SHACL validation error
    #[error("SHACL validation failed: {0}")]
    ShaclValidation(String),

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    /// Code generation error
    #[error("Code generation error: {0}")]
    Generation(String),

    /// Network error (for HTTP/HTTPS resolution)
    #[error("Network error: {0}")]
    Network(String),

    /// Generic error
    #[error("SAMM error: {0}")]
    Other(String),
}

impl From<anyhow::Error> for SammError {
    fn from(err: anyhow::Error) -> Self {
        SammError::Other(err.to_string())
    }
}

impl From<String> for SammError {
    fn from(msg: String) -> Self {
        SammError::Other(msg)
    }
}

impl From<&str> for SammError {
    fn from(msg: &str) -> Self {
        SammError::Other(msg.to_string())
    }
}
