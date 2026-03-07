//! Error types for SAMM operations

use std::fmt;

/// Result type for SAMM operations
pub type Result<T> = std::result::Result<T, SammError>;

/// Source location information for error reporting
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    /// Line number (1-indexed), if available
    pub line: Option<usize>,
    /// Column number (1-indexed), if available
    pub column: Option<usize>,
    /// File path or URI
    pub source: Option<String>,
}

impl SourceLocation {
    /// Create a new source location with line and column
    pub fn new(line: usize, column: usize) -> Self {
        Self {
            line: Some(line),
            column: Some(column),
            source: None,
        }
    }

    /// Create a source location with file path
    pub fn with_source(line: usize, column: usize, source: String) -> Self {
        Self {
            line: Some(line),
            column: Some(column),
            source: Some(source),
        }
    }

    /// Create a source location with only file path (no line/column)
    pub fn from_source(source: String) -> Self {
        Self {
            line: None,
            column: None,
            source: Some(source),
        }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.source, self.line, self.column) {
            (Some(source), Some(line), Some(column)) => {
                write!(f, "{}:{}:{}", source, line, column)
            }
            (Some(source), Some(line), None) => {
                write!(f, "{}:{}", source, line)
            }
            (Some(source), None, None) => {
                write!(f, "{}", source)
            }
            (None, Some(line), Some(column)) => {
                write!(f, "line {}:{}", line, column)
            }
            (None, Some(line), None) => {
                write!(f, "line {}", line)
            }
            _ => {
                write!(f, "unknown location")
            }
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

    /// Graph operation error
    #[error("Graph error: {0}")]
    GraphError(String),

    /// Cloud storage error
    #[error("Cloud storage error: {0}")]
    CloudError(String),

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

impl SammError {
    /// Get an actionable suggestion for how to fix this error
    ///
    /// Returns a helpful message that guides users towards resolving the issue.
    pub fn suggestion(&self) -> Option<String> {
        match self {
            SammError::InvalidUrn(msg) => {
                if msg.contains("must start with 'urn:samm:'") {
                    Some("URN format: urn:samm:<namespace>:<version>#<element>\nExample: urn:samm:org.eclipse.examples:1.0.0#Movement".to_string())
                } else if msg.contains("must contain exactly one '#'") {
                    Some("URNs must have exactly one '#' separating the model from the element name.\nExample: urn:samm:org.eclipse.examples:1.0.0#Movement".to_string())
                } else if msg.contains("namespace and version") {
                    Some("URN structure: urn:samm:<namespace>:<version>#<element>\nNamespace and version are separated by ':'".to_string())
                } else {
                    Some("Valid URN format: urn:samm:<namespace>:<version>#<element>".to_string())
                }
            }
            SammError::ParseError(msg) => {
                if msg.contains("Failed to open") || msg.contains("Failed to read") {
                    Some("Check that:\n  1. The file path exists\n  2. You have read permissions\n  3. The file is not locked by another process".to_string())
                } else if msg.contains("Invalid ZIP") || msg.contains("Failed to read ZIP") {
                    Some("Ensure the file is a valid ZIP/AASX archive:\n  1. Try opening it with a ZIP tool\n  2. Re-export the package if corrupted\n  3. Check if the file was completely downloaded".to_string())
                } else if msg.contains("No AAS content found") {
                    Some("AASX packages must contain either:\n  - aasx/xml/content.xml, or\n  - aasx/json/content.json".to_string())
                } else if msg.contains("Namespace directory not found") {
                    Some("Verify:\n  1. The namespace path is correct\n  2. The directory structure follows: <models_root>/<namespace>/<version>/\n  3. The namespace matches your URN".to_string())
                } else if msg.contains("No .ttl files found") {
                    Some("The namespace directory should contain .ttl (Turtle) files.\nCheck if:\n  1. Files have .ttl extension\n  2. You're in the correct namespace directory".to_string())
                } else {
                    Some("Review the error message above and check your input data format.".to_string())
                }
            }
            SammError::ValidationError(msg) => {
                if msg.contains("missing") {
                    Some("Required elements must be present in the model.\nRefer to SAMM specification 2.3.0 for mandatory fields.".to_string())
                } else if msg.contains("invalid") {
                    Some("Check that all model elements conform to SAMM 2.3.0 specification.".to_string())
                } else {
                    Some("Run SHACL validation to get detailed constraint violations:\n  validator.validate(&aspect)".to_string())
                }
            }
            SammError::MissingElement(element) => {
                Some(format!(
                    "Element '{}' not found. Ensure:\n  1. The element is defined in the model\n  2. URN references are correct\n  3. External models are loaded with ModelResolver",
                    element
                ))
            }
            SammError::ResolutionError(msg) => {
                if msg.contains("not found in cache") {
                    Some("The element might be in an external file.\nUse ModelResolver to load external dependencies:\n  resolver.add_file_base(\"path/to/models/\")".to_string())
                } else {
                    Some("Check:\n  1. URN format is correct\n  2. External files are accessible\n  3. Base paths are configured in ModelResolver".to_string())
                }
            }
            SammError::Network(msg) => {
                if msg.contains("timeout") {
                    Some("Network request timed out. Try:\n  1. Increasing timeout: resolver.set_http_timeout(Duration::from_secs(60))\n  2. Checking your internet connection\n  3. Verifying the URL is accessible".to_string())
                } else if msg.contains("404") || msg.contains("Not Found") {
                    Some("The remote resource was not found.\nVerify:\n  1. The URL is correct\n  2. The resource exists on the remote server\n  3. You have access permissions".to_string())
                } else {
                    Some("Check:\n  1. Internet connection is available\n  2. Remote server is accessible\n  3. Firewall/proxy settings allow the request".to_string())
                }
            }
            SammError::ShaclValidation(_) => {
                Some("SHACL validation failed. Review the constraint violations above.\nCommon issues:\n  1. Missing required properties\n  2. Invalid data types\n  3. Constraint violations (min/max values)\nRefer to: https://eclipse-esmf.github.io/samm-specification/".to_string())
            }
            SammError::Unsupported(feature) => {
                Some(format!(
                    "Feature '{}' is not yet implemented.\nConsider:\n  1. Using an alternative approach\n  2. Filing a feature request\n  3. Checking for updates",
                    feature
                ))
            }
            SammError::Generation(msg) => {
                if msg.contains("template") {
                    Some("Template errors can occur due to:\n  1. Invalid template syntax\n  2. Missing required fields in the model\n  3. Incompatible data types\nCheck the template file for syntax errors.".to_string())
                } else {
                    Some("Code generation failed. Ensure:\n  1. The model is valid\n  2. All required characteristics are defined\n  3. Data types are supported".to_string())
                }
            }
            SammError::ParseErrorWithLocation { message, location } => {
                Some(format!(
                    "Error at {}\nCheck the Turtle syntax around this location.\nCommon issues:\n  1. Missing semicolon or period\n  2. Invalid prefix usage\n  3. Unclosed brackets or quotes\nMessage: {}",
                    location, message
                ))
            }
            SammError::ValidationErrorWithLocation { message, location } => {
                Some(format!(
                    "Validation failed at {}\nReview the SAMM constraints for this element.\nMessage: {}",
                    location, message
                ))
            }
            _ => None,
        }
    }

    /// Get the error category for programmatic error handling
    ///
    /// This allows applications to handle different error types appropriately.
    pub fn category(&self) -> ErrorCategory {
        match self {
            SammError::ParseError(_) | SammError::ParseErrorWithLocation { .. } => {
                ErrorCategory::Parsing
            }
            SammError::ValidationError(_)
            | SammError::ValidationErrorWithLocation { .. }
            | SammError::ShaclValidation(_) => ErrorCategory::Validation,
            SammError::ResolutionError(_) | SammError::MissingElement(_) => {
                ErrorCategory::Resolution
            }
            SammError::InvalidUrn(_) | SammError::InvalidStructure(_) => ErrorCategory::Structure,
            SammError::Io(_) => ErrorCategory::Io,
            SammError::Network(_) => ErrorCategory::Network,
            SammError::Rdf(_) => ErrorCategory::Rdf,
            SammError::Unsupported(_) => ErrorCategory::Unsupported,
            SammError::Generation(_) => ErrorCategory::Generation,
            SammError::GraphError(_) => ErrorCategory::Other,
            SammError::CloudError(_) => ErrorCategory::Network,
            SammError::Other(_) => ErrorCategory::Other,
        }
    }

    /// Check if this error is recoverable (can be retried or has fallback options)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            SammError::Network(_)
                | SammError::ResolutionError(_)
                | SammError::Io(_)
                | SammError::CloudError(_)
        )
    }

    /// Get a user-friendly error message suitable for displaying to end users
    pub fn user_message(&self) -> String {
        match self {
            SammError::ParseError(msg) | SammError::ValidationError(msg) => {
                format!("Processing failed: {}", simplify_technical_message(msg))
            }
            SammError::ParseErrorWithLocation { message, location } => {
                format!(
                    "Error at {}: {}",
                    location,
                    simplify_technical_message(message)
                )
            }
            SammError::ValidationErrorWithLocation { message, location } => {
                format!(
                    "Validation failed at {}: {}",
                    location,
                    simplify_technical_message(message)
                )
            }
            SammError::InvalidUrn(_) => {
                "The URN format is invalid. Please check the reference format.".to_string()
            }
            SammError::MissingElement(elem) => {
                format!("Required element '{}' could not be found.", elem)
            }
            SammError::Network(_) => {
                "Network error occurred. Please check your connection and try again.".to_string()
            }
            SammError::Io(_) => {
                "File access error. Please check the file path and permissions.".to_string()
            }
            SammError::CloudError(_) => {
                "Cloud storage error. Please check your credentials and network connection."
                    .to_string()
            }
            _ => self.to_string(),
        }
    }

    /// Create a cloud storage error
    pub fn cloud_error(msg: impl Into<String>) -> Self {
        SammError::CloudError(msg.into())
    }
}

/// Category of error for programmatic handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Parsing errors (syntax, format)
    Parsing,
    /// Validation errors (constraints, rules)
    Validation,
    /// Resolution errors (missing elements, URN lookup)
    Resolution,
    /// Structure errors (invalid model structure)
    Structure,
    /// I/O errors (file access)
    Io,
    /// Network errors (HTTP requests)
    Network,
    /// RDF processing errors
    Rdf,
    /// Unsupported features
    Unsupported,
    /// Code generation errors
    Generation,
    /// Other errors
    Other,
}

/// Simplify technical error messages for end users
fn simplify_technical_message(msg: &str) -> &str {
    // Remove common technical prefixes
    msg.strip_prefix("Failed to ")
        .or_else(|| msg.strip_prefix("Unable to "))
        .or_else(|| msg.strip_prefix("Error: "))
        .unwrap_or(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_urn_suggestion() {
        let err = SammError::InvalidUrn("URN must start with 'urn:samm:'".to_string());
        let suggestion = err.suggestion().expect("suggestion should be available");
        assert!(suggestion.contains("urn:samm:"));
        assert!(suggestion.contains("Example:"));
    }

    #[test]
    fn test_parse_error_file_not_found_suggestion() {
        let err = SammError::ParseError("Failed to open file".to_string());
        let suggestion = err.suggestion().expect("suggestion should be available");
        assert!(suggestion.contains("file path exists"));
        assert!(suggestion.contains("read permissions"));
    }

    #[test]
    fn test_network_timeout_suggestion() {
        let err = SammError::Network("Request timeout after 30s".to_string());
        let suggestion = err.suggestion().expect("suggestion should be available");
        assert!(suggestion.contains("timeout"));
        assert!(suggestion.contains("set_http_timeout"));
    }

    #[test]
    fn test_error_category_parsing() {
        let err = SammError::ParseError("test".to_string());
        assert_eq!(err.category(), ErrorCategory::Parsing);
    }

    #[test]
    fn test_error_category_validation() {
        let err = SammError::ValidationError("test".to_string());
        assert_eq!(err.category(), ErrorCategory::Validation);
    }

    #[test]
    fn test_error_category_network() {
        let err = SammError::Network("test".to_string());
        assert_eq!(err.category(), ErrorCategory::Network);
    }

    #[test]
    fn test_is_recoverable_network() {
        let err = SammError::Network("timeout".to_string());
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_is_recoverable_validation() {
        let err = SammError::ValidationError("invalid".to_string());
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_user_message_parse_error() {
        let err = SammError::ParseError("Failed to parse TTL".to_string());
        let msg = err.user_message();
        assert!(msg.contains("Processing failed"));
        assert!(!msg.contains("Failed to")); // Should be simplified
    }

    #[test]
    fn test_user_message_invalid_urn() {
        let err = SammError::InvalidUrn("bad urn".to_string());
        let msg = err.user_message();
        assert!(msg.contains("URN format is invalid"));
    }

    #[test]
    fn test_user_message_with_location() {
        let location = SourceLocation::with_source(10, 5, "test.ttl".to_string());
        let err = SammError::ParseErrorWithLocation {
            message: "Failed to parse entity".to_string(),
            location,
        };
        let msg = err.user_message();
        assert!(msg.contains("test.ttl:10:5"));
        assert!(!msg.contains("Failed to")); // Should be simplified
    }

    #[test]
    fn test_shacl_validation_suggestion() {
        let err = SammError::ShaclValidation("Missing required property".to_string());
        let suggestion = err.suggestion().expect("suggestion should be available");
        assert!(suggestion.contains("SHACL validation failed"));
        assert!(suggestion.contains("https://eclipse-esmf.github.io"));
    }

    #[test]
    fn test_missing_element_suggestion() {
        let err = SammError::MissingElement("Movement".to_string());
        let suggestion = err.suggestion().expect("suggestion should be available");
        assert!(suggestion.contains("Movement"));
        assert!(suggestion.contains("ModelResolver"));
    }

    #[test]
    fn test_generation_error_suggestion() {
        let err = SammError::Generation("template rendering failed".to_string());
        let suggestion = err.suggestion().expect("suggestion should be available");
        assert!(suggestion.contains("template"));
        assert!(suggestion.contains("syntax"));
    }

    #[test]
    fn test_simplify_technical_message() {
        assert_eq!(simplify_technical_message("Failed to parse"), "parse");
        assert_eq!(simplify_technical_message("Unable to load"), "load");
        assert_eq!(simplify_technical_message("Error: invalid"), "invalid");
        assert_eq!(
            simplify_technical_message("something else"),
            "something else"
        );
    }

    #[test]
    fn test_error_category_equality() {
        assert_eq!(ErrorCategory::Parsing, ErrorCategory::Parsing);
        assert_ne!(ErrorCategory::Parsing, ErrorCategory::Validation);
    }

    #[test]
    fn test_resolution_error_suggestion() {
        let err = SammError::ResolutionError("Element not found in cache".to_string());
        let suggestion = err.suggestion().expect("suggestion should be available");
        assert!(suggestion.contains("external file"));
        assert!(suggestion.contains("add_file_base"));
    }

    #[test]
    fn test_validation_error_with_location() {
        let location = SourceLocation::new(15, 10);
        let err = SammError::ValidationErrorWithLocation {
            message: "Property name must be camelCase".to_string(),
            location,
        };
        let suggestion = err.suggestion().expect("suggestion should be available");
        assert!(suggestion.contains("line 15:10"));
        assert!(suggestion.contains("SAMM constraints"));
    }

    #[test]
    fn test_unsupported_feature_suggestion() {
        let err = SammError::Unsupported("Advanced query language".to_string());
        let suggestion = err.suggestion().expect("suggestion should be available");
        assert!(suggestion.contains("Advanced query language"));
        assert!(suggestion.contains("feature request"));
    }
}
