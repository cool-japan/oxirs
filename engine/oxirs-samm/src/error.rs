//! Error types for SAMM operations

use std::fmt;

/// Result type for SAMM operations
pub type Result<T> = std::result::Result<T, SammError>;

/// Error types for SAMM operations
#[derive(Debug, thiserror::Error)]
pub enum SammError {
    /// Error parsing a SAMM model
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Error validating a SAMM model
    #[error("Validation error: {0}")]
    ValidationError(String),

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
