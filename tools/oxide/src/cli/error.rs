//! Enhanced error handling for CLI
//!
//! Provides contextual error messages, suggestions, and recovery hints.

use std::fmt;
use std::io;

/// CLI-specific result type
pub type CliResult<T> = Result<T, CliError>;

/// Enhanced CLI error with context and suggestions
#[derive(Debug)]
pub struct CliError {
    /// The underlying error
    pub kind: CliErrorKind,
    /// Context about where the error occurred
    pub context: Option<String>,
    /// Suggestions for fixing the error
    pub suggestions: Vec<String>,
    /// Error code for documentation reference
    pub code: Option<String>,
}

#[derive(Debug)]
pub enum CliErrorKind {
    /// Invalid arguments provided
    InvalidArguments(String),
    /// File or resource not found
    NotFound(String),
    /// Permission denied
    PermissionDenied(String),
    /// Invalid format or syntax
    InvalidFormat(String),
    /// Network or connection error
    NetworkError(String),
    /// Configuration error
    ConfigError(String),
    /// Validation error
    ValidationError(String),
    /// IO error
    IoError(io::Error),
    /// Other error
    Other(String), // Changed from Box<dyn Error> to avoid Send+Sync issues
}

impl CliError {
    /// Create a new CLI error
    pub fn new(kind: CliErrorKind) -> Self {
        Self {
            kind,
            context: None,
            suggestions: Vec::new(),
            code: None,
        }
    }

    /// Add context to the error
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Add a suggestion for fixing the error
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Add multiple suggestions
    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions.extend(suggestions);
        self
    }

    /// Set error code for documentation reference
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Create an invalid arguments error
    pub fn invalid_arguments(message: impl Into<String>) -> Self {
        Self::new(CliErrorKind::InvalidArguments(message.into()))
    }

    /// Create a not found error
    pub fn not_found(resource: impl Into<String>) -> Self {
        Self::new(CliErrorKind::NotFound(resource.into()))
    }

    /// Create a permission denied error
    pub fn permission_denied(resource: impl Into<String>) -> Self {
        Self::new(CliErrorKind::PermissionDenied(resource.into()))
    }

    /// Create an invalid format error
    pub fn invalid_format(message: impl Into<String>) -> Self {
        Self::new(CliErrorKind::InvalidFormat(message.into()))
    }

    /// Create a validation error
    pub fn validation_error(message: impl Into<String>) -> Self {
        Self::new(CliErrorKind::ValidationError(message.into()))
    }

    /// Create a configuration error
    pub fn config_error(message: impl Into<String>) -> Self {
        Self::new(CliErrorKind::ConfigError(message.into()))
    }

    /// Get user-friendly error message
    pub fn user_message(&self) -> String {
        match &self.kind {
            CliErrorKind::InvalidArguments(msg) => {
                format!("Invalid arguments: {}", msg)
            }
            CliErrorKind::NotFound(resource) => {
                format!("Not found: {}", resource)
            }
            CliErrorKind::PermissionDenied(resource) => {
                format!("Permission denied: {}", resource)
            }
            CliErrorKind::InvalidFormat(msg) => {
                format!("Invalid format: {}", msg)
            }
            CliErrorKind::NetworkError(msg) => {
                format!("Network error: {}", msg)
            }
            CliErrorKind::ConfigError(msg) => {
                format!("Configuration error: {}", msg)
            }
            CliErrorKind::ValidationError(msg) => {
                format!("Validation error: {}", msg)
            }
            CliErrorKind::IoError(err) => {
                format!("IO error: {}", err)
            }
            CliErrorKind::Other(msg) => {
                format!("Error: {}", msg)
            }
        }
    }

    /// Format the error with all context and suggestions
    pub fn format_detailed(&self) -> String {
        let mut output = String::new();
        
        // Main error message
        output.push_str(&format!("Error: {}\n", self.user_message()));
        
        // Context if available
        if let Some(ref context) = self.context {
            output.push_str(&format!("\nContext: {}\n", context));
        }
        
        // Suggestions if available
        if !self.suggestions.is_empty() {
            output.push_str("\nSuggestions:\n");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, suggestion));
            }
        }
        
        // Error code for documentation
        if let Some(ref code) = self.code {
            output.push_str(&format!(
                "\nFor more information, see: https://oxirs.io/docs/errors/{}\n",
                code
            ));
        }
        
        output
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.user_message())
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            CliErrorKind::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for CliError {
    fn from(err: io::Error) -> Self {
        Self::new(CliErrorKind::IoError(err))
    }
}

impl From<Box<dyn std::error::Error>> for CliError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        Self::new(CliErrorKind::Other(err.to_string()))
    }
}

/// Helper functions for common error scenarios
pub mod helpers {
    use super::*;
    use std::path::Path;

    /// Create an error for file not found with suggestions
    pub fn file_not_found_error(path: &Path) -> CliError {
        CliError::not_found(path.display().to_string())
            .with_context(format!("Failed to access file: {}", path.display()))
            .with_suggestions(vec![
                "Check if the file exists".to_string(),
                "Verify the file path is correct".to_string(),
                "Ensure you have read permissions".to_string(),
                format!("Try using an absolute path instead of relative"),
            ])
            .with_code("E001")
    }

    /// Create an error for invalid RDF format
    pub fn invalid_rdf_format_error(format: &str, supported: &[&str]) -> CliError {
        CliError::invalid_format(format)
            .with_context("The specified RDF format is not supported")
            .with_suggestions(vec![
                format!("Supported formats: {}", supported.join(", ")),
                "Use 'auto' to automatically detect the format".to_string(),
                "Check the file extension matches the format".to_string(),
            ])
            .with_code("E002")
    }

    /// Create an error for invalid SPARQL query
    pub fn invalid_sparql_error(error: &str, line: Option<usize>) -> CliError {
        let mut err = CliError::validation_error(error)
            .with_context("Failed to parse SPARQL query");
        
        if let Some(line_num) = line {
            err = err.with_context(format!("Error at line {}", line_num));
        }
        
        err.with_suggestions(vec![
            "Check the SPARQL syntax".to_string(),
            "Verify all prefixes are defined".to_string(),
            "Ensure brackets and quotes are balanced".to_string(),
            "Use 'oxide qparse' to validate the query".to_string(),
        ])
        .with_code("E003")
    }

    /// Create an error for connection issues
    pub fn connection_error(endpoint: &str, error: &str) -> CliError {
        CliError::new(CliErrorKind::NetworkError(error.to_string()))
            .with_context(format!("Failed to connect to: {}", endpoint))
            .with_suggestions(vec![
                "Check if the endpoint is reachable".to_string(),
                "Verify the URL is correct".to_string(),
                "Check your network connection".to_string(),
                "Try increasing the timeout with --timeout".to_string(),
            ])
            .with_code("E004")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = CliError::invalid_arguments("Missing required argument")
            .with_context("Processing command-line arguments")
            .with_suggestion("Use --help to see all available options");

        assert!(err.user_message().contains("Invalid arguments"));
        assert!(err.context.is_some());
        assert_eq!(err.suggestions.len(), 1);
    }

    #[test]
    fn test_error_formatting() {
        let err = CliError::not_found("config.toml")
            .with_context("Loading configuration")
            .with_suggestions(vec![
                "Create a config file with 'oxide config init'".to_string(),
                "Specify a config file with --config".to_string(),
            ])
            .with_code("E001");

        let formatted = err.format_detailed();
        assert!(formatted.contains("Not found: config.toml"));
        assert!(formatted.contains("Context:"));
        assert!(formatted.contains("Suggestions:"));
        assert!(formatted.contains("E001"));
    }
}