//! Error handling for RDF format operations
//!
//! Extracted and adapted from OxiGraph error handling with OxiRS enhancements.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::io;

/// Position in a text document
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextPosition {
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based)
    pub column: usize,
    /// Byte offset from start of document
    pub offset: usize,
}

impl TextPosition {
    /// Create a new text position
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }

    /// Position at start of document
    pub fn start() -> Self {
        Self::new(1, 1, 0)
    }
}

impl fmt::Display for TextPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

/// Syntax error during RDF parsing
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RdfSyntaxError {
    /// Error message
    pub message: String,
    /// Position where error occurred
    pub position: Option<TextPosition>,
    /// Context around the error (line content)
    pub context: Option<String>,
}

impl RdfSyntaxError {
    /// Create a new syntax error
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            position: None,
            context: None,
        }
    }

    /// Create a syntax error with position
    pub fn with_position(message: impl Into<String>, position: TextPosition) -> Self {
        Self {
            message: message.into(),
            position: Some(position),
            context: None,
        }
    }

    /// Create a syntax error with position and context
    pub fn with_context(
        message: impl Into<String>,
        position: TextPosition,
        context: impl Into<String>,
    ) -> Self {
        Self {
            message: message.into(),
            position: Some(position),
            context: Some(context.into()),
        }
    }

    /// Add position information to the error
    pub fn at_position(mut self, position: TextPosition) -> Self {
        self.position = Some(position);
        self
    }

    /// Add context information to the error
    pub fn with_context_str(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

impl fmt::Display for RdfSyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Syntax error: {}", self.message)?;

        if let Some(position) = &self.position {
            write!(f, " at {}", position)?;
        }

        if let Some(context) = &self.context {
            write!(f, "\nContext: {}", context)?;
        }

        Ok(())
    }
}

impl Error for RdfSyntaxError {}

/// Parse error during RDF processing  
#[derive(Debug)]
pub enum RdfParseError {
    /// Syntax error in the RDF document
    Syntax(RdfSyntaxError),
    /// I/O error during reading
    Io(io::Error),
    /// Invalid IRI format
    InvalidIri(String),
    /// Invalid literal format
    InvalidLiteral(String),
    /// Invalid blank node format
    InvalidBlankNode(String),
    /// Invalid datatype
    InvalidDatatype(String),
    /// Invalid language tag
    InvalidLanguageTag(String),
    /// Unsupported feature
    UnsupportedFeature(String),
    /// Internal processing error
    Internal(String),
}

impl RdfParseError {
    /// Create a syntax error
    pub fn syntax(message: impl Into<String>) -> Self {
        Self::Syntax(RdfSyntaxError::new(message))
    }

    /// Create a syntax error with position
    pub fn syntax_at(message: impl Into<String>, position: TextPosition) -> Self {
        Self::Syntax(RdfSyntaxError::with_position(message, position))
    }

    /// Create an invalid IRI error
    pub fn invalid_iri(iri: impl Into<String>) -> Self {
        Self::InvalidIri(iri.into())
    }

    /// Create an invalid literal error
    pub fn invalid_literal(literal: impl Into<String>) -> Self {
        Self::InvalidLiteral(literal.into())
    }

    /// Create an unsupported feature error
    pub fn unsupported(feature: impl Into<String>) -> Self {
        Self::UnsupportedFeature(feature.into())
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

impl fmt::Display for RdfParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(err) => write!(f, "{}", err),
            Self::Io(err) => write!(f, "I/O error: {}", err),
            Self::InvalidIri(iri) => write!(f, "Invalid IRI: {}", iri),
            Self::InvalidLiteral(literal) => write!(f, "Invalid literal: {}", literal),
            Self::InvalidBlankNode(bnode) => write!(f, "Invalid blank node: {}", bnode),
            Self::InvalidDatatype(datatype) => write!(f, "Invalid datatype: {}", datatype),
            Self::InvalidLanguageTag(tag) => write!(f, "Invalid language tag: {}", tag),
            Self::UnsupportedFeature(feature) => write!(f, "Unsupported feature: {}", feature),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl Error for RdfParseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Syntax(err) => Some(err),
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for RdfParseError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<RdfSyntaxError> for RdfParseError {
    fn from(err: RdfSyntaxError) -> Self {
        Self::Syntax(err)
    }
}

impl From<crate::OxirsError> for RdfParseError {
    fn from(err: crate::OxirsError) -> Self {
        match err {
            crate::OxirsError::Parse(msg) => Self::syntax(msg),
            crate::OxirsError::Io(msg) => Self::syntax(format!("IO error: {}", msg)),
            crate::OxirsError::Store(msg) => Self::internal(format!("Store error: {}", msg)),
            crate::OxirsError::Query(msg) => Self::internal(format!("Query error: {}", msg)),
            crate::OxirsError::Serialize(msg) => {
                Self::internal(format!("Serialization error: {}", msg))
            }
        }
    }
}

// Add direct conversion from LanguageTagParseError to RdfParseError
impl From<crate::model::literal::LanguageTagParseError> for RdfParseError {
    fn from(err: crate::model::literal::LanguageTagParseError) -> Self {
        Self::InvalidLanguageTag(err.to_string())
    }
}

/// General format error for high-level operations
#[derive(Debug)]
pub enum FormatError {
    /// Parse error
    Parse(RdfParseError),
    /// Serialization error
    Serialize(io::Error),
    /// Unsupported format
    UnsupportedFormat(String),
    /// Invalid data
    InvalidData(String),
    /// Missing required component
    MissingComponent(String),
    /// Configuration error
    Configuration(String),
}

impl FormatError {
    /// Create an unsupported format error
    pub fn unsupported_format(format: impl Into<String>) -> Self {
        Self::UnsupportedFormat(format.into())
    }

    /// Create an invalid data error
    pub fn invalid_data(message: impl Into<String>) -> Self {
        Self::InvalidData(message.into())
    }

    /// Create a missing component error
    pub fn missing_component(component: impl Into<String>) -> Self {
        Self::MissingComponent(component.into())
    }

    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration(message.into())
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(err) => write!(f, "Parse error: {}", err),
            Self::Serialize(err) => write!(f, "Serialization error: {}", err),
            Self::UnsupportedFormat(format) => write!(f, "Unsupported format: {}", format),
            Self::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            Self::MissingComponent(component) => write!(f, "Missing component: {}", component),
            Self::Configuration(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl Error for FormatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parse(err) => Some(err),
            Self::Serialize(err) => Some(err),
            _ => None,
        }
    }
}

impl From<RdfParseError> for FormatError {
    fn from(err: RdfParseError) -> Self {
        Self::Parse(err)
    }
}

impl From<io::Error> for FormatError {
    fn from(err: io::Error) -> Self {
        Self::Serialize(err)
    }
}

impl From<RdfSyntaxError> for FormatError {
    fn from(err: RdfSyntaxError) -> Self {
        Self::Parse(err.into())
    }
}

/// Result type for parsing operations
pub type ParseResult<T> = Result<T, RdfParseError>;

/// Result type for serialization operations  
pub type SerializeResult<T> = Result<T, io::Error>;

/// Result type for general format operations
pub type FormatResult<T> = Result<T, FormatError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_position() {
        let pos = TextPosition::new(10, 5, 100);
        assert_eq!(pos.line, 10);
        assert_eq!(pos.column, 5);
        assert_eq!(pos.offset, 100);

        let start = TextPosition::start();
        assert_eq!(start.line, 1);
        assert_eq!(start.column, 1);
        assert_eq!(start.offset, 0);
    }

    #[test]
    fn test_syntax_error() {
        let err = RdfSyntaxError::new("Invalid syntax");
        assert_eq!(err.message, "Invalid syntax");
        assert!(err.position.is_none());

        let pos = TextPosition::new(5, 10, 50);
        let err_with_pos = RdfSyntaxError::with_position("Bad token", pos);
        assert_eq!(err_with_pos.position, Some(pos));
    }

    #[test]
    fn test_parse_error() {
        let syntax_err = RdfParseError::syntax("Bad syntax");
        assert!(matches!(syntax_err, RdfParseError::Syntax(_)));

        let iri_err = RdfParseError::invalid_iri("not-an-iri");
        assert!(matches!(iri_err, RdfParseError::InvalidIri(_)));

        let unsupported_err = RdfParseError::unsupported("Some feature");
        assert!(matches!(
            unsupported_err,
            RdfParseError::UnsupportedFeature(_)
        ));
    }

    #[test]
    fn test_format_error() {
        let format_err = FormatError::unsupported_format("unknown/format");
        assert!(matches!(format_err, FormatError::UnsupportedFormat(_)));

        let data_err = FormatError::invalid_data("Bad data");
        assert!(matches!(data_err, FormatError::InvalidData(_)));
    }

    #[test]
    fn test_error_conversion() {
        let syntax_err = RdfSyntaxError::new("Bad syntax");
        let parse_err: RdfParseError = syntax_err.into();
        let format_err: FormatError = parse_err.into();

        assert!(matches!(format_err, FormatError::Parse(_)));
    }
}
