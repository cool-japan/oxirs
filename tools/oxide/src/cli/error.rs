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

    /// Create an IO error
    pub fn io_error(error: io::Error) -> Self {
        Self::new(CliErrorKind::IoError(error))
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
    
    /// Create an error for dataset not found
    pub fn dataset_not_found_error(name: &str) -> CliError {
        CliError::not_found(format!("Dataset '{}'", name))
            .with_context("The specified dataset does not exist")
            .with_suggestions(vec![
                format!("Create the dataset with: oxide init {}", name),
                "List available datasets with: oxide dataset list".to_string(),
                "Check if the dataset name is spelled correctly".to_string(),
            ])
            .with_code("E005")
    }
    
    /// Create an error for permission issues
    pub fn permission_error(path: &Path, operation: &str) -> CliError {
        CliError::permission_denied(path.display().to_string())
            .with_context(format!("Cannot {} file/directory", operation))
            .with_suggestions(vec![
                format!("Check permissions: ls -la {}", path.display()),
                "Run with appropriate permissions (sudo if necessary)".to_string(),
                "Ensure the parent directory is writable".to_string(),
            ])
            .with_code("E006")
    }
    
    /// Create an error for timeout
    pub fn timeout_error(operation: &str, timeout_secs: u64) -> CliError {
        CliError::new(CliErrorKind::Other(format!("Operation timed out after {}s", timeout_secs)))
            .with_context(format!("Timeout while: {}", operation))
            .with_suggestions(vec![
                format!("Increase timeout with: --timeout {}", timeout_secs * 2),
                "Check if the server is responding".to_string(),
                "Try with a smaller dataset or query".to_string(),
                "Check server logs for issues".to_string(),
            ])
            .with_code("E007")
    }
    
    /// Create an error for incompatible versions
    pub fn version_mismatch_error(expected: &str, found: &str) -> CliError {
        CliError::new(CliErrorKind::Other(format!("Version mismatch: expected {}, found {}", expected, found)))
            .with_context("Incompatible dataset or file version")
            .with_suggestions(vec![
                format!("Migrate the dataset with: oxide migrate --from {} --to {}", found, expected),
                "Use a compatible version of oxide".to_string(),
                "Check the migration guide in the documentation".to_string(),
            ])
            .with_code("E008")
    }
    
    /// Create an error for missing dependencies
    pub fn missing_dependency_error(dependency: &str, feature: &str) -> CliError {
        CliError::new(CliErrorKind::Other(format!("Missing dependency: {}", dependency)))
            .with_context(format!("Required for: {}", feature))
            .with_suggestions(vec![
                format!("Install {} to enable this feature", dependency),
                "Check the documentation for installation instructions".to_string(),
                format!("This feature requires the '{}' feature flag", feature),
            ])
            .with_code("E009")
    }
    
    /// Create an error for configuration issues with specific field
    pub fn config_field_error(field: &str, value: &str, expected: &str) -> CliError {
        CliError::config_error(format!("Invalid value '{}' for field '{}'", value, field))
            .with_context("Configuration validation failed")
            .with_suggestions(vec![
                format!("Expected: {}", expected),
                format!("Check your config file for the '{}' field", field),
                "Run 'oxide config validate' to check all settings".to_string(),
                "Use 'oxide config init' to generate a valid config".to_string(),
            ])
            .with_code("E010")
    }
}

/// Error recovery suggestions based on error type
pub mod recovery {
    use super::*;
    
    /// Suggest recovery actions for IO errors
    pub fn suggest_io_recovery(error: &io::Error) -> Vec<String> {
        use io::ErrorKind::*;
        
        match error.kind() {
            NotFound => vec![
                "Check if the file or directory exists".to_string(),
                "Verify the path is correct".to_string(),
                "Use an absolute path instead of relative".to_string(),
            ],
            PermissionDenied => vec![
                "Check file permissions".to_string(),
                "Run with appropriate privileges".to_string(),
                "Ensure parent directory is accessible".to_string(),
            ],
            AlreadyExists => vec![
                "Use a different name".to_string(),
                "Delete the existing file/directory first".to_string(),
                "Use --force to overwrite".to_string(),
            ],
            WouldBlock => vec![
                "The resource is temporarily unavailable".to_string(),
                "Try again in a moment".to_string(),
                "Check if another process is using the resource".to_string(),
            ],
            InvalidInput => vec![
                "Check the input format".to_string(),
                "Verify the file encoding (UTF-8 expected)".to_string(),
                "Remove any invalid characters".to_string(),
            ],
            _ => vec![
                "Check system resources (disk space, memory)".to_string(),
                "Verify file system integrity".to_string(),
                "Check system logs for more details".to_string(),
            ],
        }
    }
    
    /// Suggest fixes for common SPARQL errors
    pub fn suggest_sparql_fixes(error_msg: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if error_msg.contains("Undefined prefix") {
            suggestions.push("Define the prefix at the beginning of your query".to_string());
            suggestions.push("Example: PREFIX foaf: <http://xmlns.com/foaf/0.1/>".to_string());
        }
        
        if error_msg.contains("Expected") && error_msg.contains("found") {
            suggestions.push("Check for syntax errors around the indicated position".to_string());
            suggestions.push("Verify brackets, quotes, and punctuation".to_string());
        }
        
        if error_msg.contains("Variable") && error_msg.contains("not in scope") {
            suggestions.push("Ensure all variables in SELECT are defined in WHERE".to_string());
            suggestions.push("Check for typos in variable names".to_string());
        }
        
        suggestions
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