//! Enhanced error messages with better context and suggestions
//!
//! This module provides:
//! - Rich error messages with context
//! - Source location information
//! - Suggestions for common errors
//! - Error categorization
//! - Stack traces for debugging

use serde::{Deserialize, Serialize};
use std::fmt;

/// Error category for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Syntax error in GraphQL query
    Syntax,
    /// Validation error
    Validation,
    /// Type error
    Type,
    /// Field not found
    FieldNotFound,
    /// Argument error
    Argument,
    /// Authentication/authorization error
    Auth,
    /// Rate limiting error
    RateLimit,
    /// Internal server error
    Internal,
    /// SPARQL translation error
    SparqlTranslation,
    /// Data fetching error
    DataFetch,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::Syntax => write!(f, "Syntax Error"),
            ErrorCategory::Validation => write!(f, "Validation Error"),
            ErrorCategory::Type => write!(f, "Type Error"),
            ErrorCategory::FieldNotFound => write!(f, "Field Not Found"),
            ErrorCategory::Argument => write!(f, "Argument Error"),
            ErrorCategory::Auth => write!(f, "Authorization Error"),
            ErrorCategory::RateLimit => write!(f, "Rate Limit Exceeded"),
            ErrorCategory::Internal => write!(f, "Internal Server Error"),
            ErrorCategory::SparqlTranslation => write!(f, "SPARQL Translation Error"),
            ErrorCategory::DataFetch => write!(f, "Data Fetching Error"),
        }
    }
}

/// Source location in query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    pub line: usize,
    pub column: usize,
    pub offset: usize,
}

impl SourceLocation {
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

/// Code snippet for error context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSnippet {
    /// The source code
    pub source: String,
    /// Start location
    pub start: SourceLocation,
    /// End location (optional)
    pub end: Option<SourceLocation>,
    /// Highlighted sections
    pub highlights: Vec<(usize, usize)>,
}

impl CodeSnippet {
    pub fn new(source: String, start: SourceLocation) -> Self {
        Self {
            source,
            start,
            end: None,
            highlights: Vec::new(),
        }
    }

    pub fn with_end(mut self, end: SourceLocation) -> Self {
        self.end = Some(end);
        self
    }

    pub fn with_highlight(mut self, start: usize, end: usize) -> Self {
        self.highlights.push((start, end));
        self
    }

    /// Format the code snippet with error indicators
    pub fn format_with_indicator(&self) -> String {
        let lines: Vec<&str> = self.source.lines().collect();
        let mut output = String::new();

        let start_line = self.start.line.saturating_sub(1);
        let end_line = self.end.as_ref().map(|e| e.line).unwrap_or(self.start.line);

        // Show context (2 lines before and after)
        let context_start = start_line.saturating_sub(2);
        let context_end = (end_line + 2).min(lines.len());

        for (idx, line) in lines
            .iter()
            .enumerate()
            .skip(context_start)
            .take(context_end - context_start)
        {
            let line_num = idx + 1;
            output.push_str(&format!("{:4} | {}\n", line_num, line));

            // Add indicator for error line
            if line_num == self.start.line {
                output.push_str("     | ");
                for _ in 0..self.start.column.saturating_sub(1) {
                    output.push(' ');
                }
                output.push('^');
                if let Some(end) = &self.end {
                    if end.line == self.start.line {
                        for _ in self.start.column..end.column.saturating_sub(1) {
                            output.push('~');
                        }
                    }
                }
                output.push('\n');
            }
        }

        output
    }
}

/// Error suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSuggestion {
    /// Suggestion message
    pub message: String,
    /// Optional fix (replacement text)
    pub fix: Option<String>,
    /// Priority (higher = more likely to be correct)
    pub priority: u32,
}

impl ErrorSuggestion {
    pub fn new(message: String) -> Self {
        Self {
            message,
            fix: None,
            priority: 50,
        }
    }

    pub fn with_fix(mut self, fix: String) -> Self {
        self.fix = Some(fix);
        self
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

/// Enhanced GraphQL error with rich context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedError {
    /// Error category
    pub category: ErrorCategory,
    /// Primary error message
    pub message: String,
    /// Detailed explanation
    pub details: Option<String>,
    /// Source location
    pub location: Option<SourceLocation>,
    /// Code snippet
    pub snippet: Option<CodeSnippet>,
    /// Suggestions for fixing
    pub suggestions: Vec<ErrorSuggestion>,
    /// Path to the field that caused error
    pub path: Vec<String>,
    /// Extensions (additional metadata)
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

impl EnhancedError {
    pub fn new(category: ErrorCategory, message: String) -> Self {
        Self {
            category,
            message,
            details: None,
            location: None,
            snippet: None,
            suggestions: Vec::new(),
            path: Vec::new(),
            extensions: serde_json::Map::new(),
        }
    }

    pub fn with_details(mut self, details: String) -> Self {
        self.details = Some(details);
        self
    }

    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.location = Some(location);
        self
    }

    pub fn with_snippet(mut self, snippet: CodeSnippet) -> Self {
        self.snippet = Some(snippet);
        self
    }

    pub fn with_suggestion(mut self, suggestion: ErrorSuggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    pub fn with_path(mut self, path: Vec<String>) -> Self {
        self.path = path;
        self
    }

    pub fn add_extension(&mut self, key: String, value: serde_json::Value) {
        self.extensions.insert(key, value);
    }

    /// Format error as a user-friendly message
    pub fn format_user_message(&self) -> String {
        let mut output = String::new();

        // Category and message
        output.push_str(&format!("{}: {}\n", self.category, self.message));

        // Location
        if let Some(location) = &self.location {
            output.push_str(&format!("  at {}\n", location));
        }

        // Path
        if !self.path.is_empty() {
            output.push_str(&format!("  in path: {}\n", self.path.join(".")));
        }

        // Details
        if let Some(details) = &self.details {
            output.push_str(&format!("\n{}\n", details));
        }

        // Code snippet
        if let Some(snippet) = &self.snippet {
            output.push('\n');
            output.push_str(&snippet.format_with_indicator());
        }

        // Suggestions
        if !self.suggestions.is_empty() {
            output.push_str("\nSuggestions:\n");
            let mut sorted_suggestions = self.suggestions.clone();
            sorted_suggestions.sort_by(|a, b| b.priority.cmp(&a.priority));

            for (idx, suggestion) in sorted_suggestions.iter().enumerate().take(3) {
                output.push_str(&format!("  {}. {}\n", idx + 1, suggestion.message));
                if let Some(fix) = &suggestion.fix {
                    output.push_str(&format!("     Try: {}\n", fix));
                }
            }
        }

        output
    }

    /// Convert to GraphQL error format
    pub fn to_graphql_error(&self) -> serde_json::Value {
        let mut error = serde_json::json!({
            "message": self.message,
        });

        if let Some(location) = &self.location {
            error["locations"] = serde_json::json!([{
                "line": location.line,
                "column": location.column,
            }]);
        }

        if !self.path.is_empty() {
            error["path"] = serde_json::json!(self.path);
        }

        let mut extensions = self.extensions.clone();
        extensions.insert(
            "category".to_string(),
            serde_json::json!(format!("{:?}", self.category)),
        );

        if let Some(details) = &self.details {
            extensions.insert("details".to_string(), serde_json::json!(details));
        }

        if !self.suggestions.is_empty() {
            let suggestions: Vec<String> =
                self.suggestions.iter().map(|s| s.message.clone()).collect();
            extensions.insert("suggestions".to_string(), serde_json::json!(suggestions));
        }

        error["extensions"] = serde_json::json!(extensions);

        error
    }
}

impl fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_user_message())
    }
}

impl std::error::Error for EnhancedError {}

/// Common error builders
pub struct ErrorBuilder;

impl ErrorBuilder {
    /// Field not found error
    pub fn field_not_found(type_name: &str, field_name: &str) -> EnhancedError {
        EnhancedError::new(
            ErrorCategory::FieldNotFound,
            format!("Field '{}' not found on type '{}'", field_name, type_name),
        )
        .with_details(format!(
            "The type '{}' does not have a field named '{}'.",
            type_name, field_name
        ))
        .with_suggestion(
            ErrorSuggestion::new(format!(
                "Check the schema documentation for available fields on type '{}'",
                type_name
            ))
            .with_priority(90),
        )
    }

    /// Type mismatch error
    pub fn type_mismatch(expected: &str, actual: &str, field: &str) -> EnhancedError {
        EnhancedError::new(
            ErrorCategory::Type,
            format!(
                "Type mismatch for field '{}': expected '{}', got '{}'",
                field, expected, actual
            ),
        )
        .with_details(format!(
            "The field '{}' expects values of type '{}', but received '{}'.",
            field, expected, actual
        ))
        .with_suggestion(
            ErrorSuggestion::new(format!("Ensure the value is of type '{}'", expected))
                .with_priority(95),
        )
    }

    /// Missing required argument
    pub fn missing_argument(field: &str, argument: &str) -> EnhancedError {
        EnhancedError::new(
            ErrorCategory::Argument,
            format!(
                "Missing required argument '{}' for field '{}'",
                argument, field
            ),
        )
        .with_details(format!(
            "The field '{}' requires an argument '{}' to be provided.",
            field, argument
        ))
        .with_suggestion(
            ErrorSuggestion::new(format!(
                "Add the '{}' argument to the field '{}'",
                argument, field
            ))
            .with_fix(format!("{}({}: <value>)", field, argument))
            .with_priority(100),
        )
    }

    /// Syntax error
    pub fn syntax_error(message: String, location: SourceLocation) -> EnhancedError {
        EnhancedError::new(ErrorCategory::Syntax, message)
            .with_location(location)
            .with_suggestion(
                ErrorSuggestion::new("Check the GraphQL query syntax".to_string())
                    .with_priority(80),
            )
    }

    /// Rate limit exceeded
    pub fn rate_limit_exceeded(limit: u32, window: u64) -> EnhancedError {
        EnhancedError::new(
            ErrorCategory::RateLimit,
            format!(
                "Rate limit of {} requests per {} seconds exceeded",
                limit, window
            ),
        )
        .with_details(
            "You have made too many requests. Please wait before trying again.".to_string(),
        )
        .with_suggestion(
            ErrorSuggestion::new(format!(
                "Wait {} seconds before making another request",
                window
            ))
            .with_priority(100),
        )
    }

    /// Internal server error
    pub fn internal_error(message: String) -> EnhancedError {
        EnhancedError::new(ErrorCategory::Internal, "Internal server error".to_string())
            .with_details(message)
            .with_suggestion(
                ErrorSuggestion::new(
                    "This is an internal error. Please contact support if it persists.".to_string(),
                )
                .with_priority(50),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_category_display() {
        assert_eq!(ErrorCategory::Syntax.to_string(), "Syntax Error");
        assert_eq!(ErrorCategory::Validation.to_string(), "Validation Error");
        assert_eq!(ErrorCategory::Type.to_string(), "Type Error");
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new(5, 10, 42);
        assert_eq!(loc.line, 5);
        assert_eq!(loc.column, 10);
        assert_eq!(loc.offset, 42);
        assert_eq!(loc.to_string(), "line 5, column 10");
    }

    #[test]
    fn test_error_suggestion() {
        let suggestion = ErrorSuggestion::new("Check the field name".to_string())
            .with_fix("Use 'name' instead of 'names'".to_string())
            .with_priority(90);

        assert_eq!(suggestion.message, "Check the field name");
        assert!(suggestion.fix.is_some());
        assert_eq!(suggestion.priority, 90);
    }

    #[test]
    fn test_enhanced_error_creation() {
        let error = EnhancedError::new(ErrorCategory::FieldNotFound, "Field not found".to_string())
            .with_details("The field 'foo' does not exist".to_string())
            .with_path(vec!["query".to_string(), "user".to_string()]);

        assert_eq!(error.category, ErrorCategory::FieldNotFound);
        assert_eq!(error.message, "Field not found");
        assert_eq!(error.path.len(), 2);
    }

    #[test]
    fn test_error_builder_field_not_found() {
        let error = ErrorBuilder::field_not_found("User", "email");

        assert_eq!(error.category, ErrorCategory::FieldNotFound);
        assert!(error.message.contains("email"));
        assert!(error.message.contains("User"));
        assert_eq!(error.suggestions.len(), 1);
    }

    #[test]
    fn test_error_builder_type_mismatch() {
        let error = ErrorBuilder::type_mismatch("String", "Int", "name");

        assert_eq!(error.category, ErrorCategory::Type);
        assert!(error.message.contains("String"));
        assert!(error.message.contains("Int"));
    }

    #[test]
    fn test_error_builder_missing_argument() {
        let error = ErrorBuilder::missing_argument("getUser", "id");

        assert_eq!(error.category, ErrorCategory::Argument);
        assert!(error.message.contains("id"));
        assert!(error.message.contains("getUser"));
        assert!(error.suggestions[0].fix.is_some());
    }

    #[test]
    fn test_error_to_graphql() {
        let error = EnhancedError::new(ErrorCategory::Syntax, "Parse error".to_string())
            .with_location(SourceLocation::new(1, 5, 4))
            .with_path(vec!["query".to_string()]);

        let graphql_error = error.to_graphql_error();

        assert_eq!(graphql_error["message"], "Parse error");
        assert!(graphql_error["locations"].is_array());
        assert!(graphql_error["path"].is_array());
        assert!(graphql_error["extensions"].is_object());
    }

    #[test]
    fn test_code_snippet_formatting() {
        let source = "query {\n  user {\n    naame\n  }\n}".to_string();
        let snippet = CodeSnippet::new(source, SourceLocation::new(3, 5, 0));

        let formatted = snippet.format_with_indicator();
        assert!(formatted.contains("naame"));
        assert!(formatted.contains("^"));
    }

    #[test]
    fn test_error_format_user_message() {
        let error = EnhancedError::new(
            ErrorCategory::FieldNotFound,
            "Field 'email' not found".to_string(),
        )
        .with_suggestion(ErrorSuggestion::new("Check spelling".to_string()));

        let message = error.format_user_message();
        assert!(message.contains("Field Not Found"));
        assert!(message.contains("email"));
        assert!(message.contains("Suggestions"));
    }

    #[test]
    fn test_error_builder_rate_limit() {
        let error = ErrorBuilder::rate_limit_exceeded(100, 60);

        assert_eq!(error.category, ErrorCategory::RateLimit);
        assert!(error.message.contains("100"));
        assert!(error.message.contains("60"));
    }

    #[test]
    fn test_error_builder_internal() {
        let error = ErrorBuilder::internal_error("Database connection failed".to_string());

        assert_eq!(error.category, ErrorCategory::Internal);
        assert!(error.details.is_some());
    }
}
