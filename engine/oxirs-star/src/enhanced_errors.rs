//! Enhanced Error Handling for RDF-star
//!
//! This module provides improved error message quality, structured error reporting,
//! and enhanced context preservation for RDF-star operations.

use crate::{StarError, StarResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Enhanced error context with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Source file or input identifier
    pub source: Option<String>,
    /// Line number where error occurred
    pub line: Option<usize>,
    /// Column number where error occurred
    pub column: Option<usize>,
    /// Surrounding text for context
    pub snippet: Option<String>,
    /// Operation being performed
    pub operation: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new() -> Self {
        Self {
            source: None,
            line: None,
            column: None,
            snippet: None,
            operation: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the source file or identifier
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set the line and column position
    pub fn with_position(mut self, line: usize, column: usize) -> Self {
        self.line = Some(line);
        self.column = Some(column);
        self
    }

    /// Set a code snippet for context
    pub fn with_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.snippet = Some(snippet.into());
        self
    }

    /// Set the operation being performed
    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = Some(operation.into());
        self
    }

    /// Add metadata key-value pair
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced error with rich context and formatting
#[derive(Debug)]
pub struct EnhancedError {
    /// The underlying StarError
    pub error: StarError,
    /// Enhanced context information
    pub context: Box<ErrorContext>,
    /// Severity level
    pub severity: ErrorSeverity,
    /// Error category for grouping
    pub category: ErrorCategory,
    /// Recovery suggestions
    pub suggestions: Box<Vec<String>>,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Critical errors that prevent operation
    Critical,
    /// Errors that cause operation failure
    Error,
    /// Warnings that indicate potential issues
    Warning,
    /// Informational messages
    Info,
}

/// Error categories for grouping and handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Syntax and parsing errors
    Syntax,
    /// Semantic validation errors
    Semantic,
    /// Configuration and setup errors
    Configuration,
    /// Runtime and execution errors
    Runtime,
    /// I/O and resource errors
    IO,
    /// Network and connectivity errors
    Network,
}

impl EnhancedError {
    /// Create a new enhanced error
    pub fn new(error: StarError) -> Self {
        let (severity, category) = Self::classify_error(&error);
        Self {
            error,
            context: Box::new(ErrorContext::new()),
            severity,
            category,
            suggestions: Box::new(Vec::new()),
        }
    }

    /// Add context to the error
    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = Box::new(context);
        self
    }

    /// Set the severity level
    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Add a recovery suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Add multiple recovery suggestions
    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions.extend(suggestions);
        self
    }

    /// Classify error to determine severity and category
    fn classify_error(error: &StarError) -> (ErrorSeverity, ErrorCategory) {
        match error {
            StarError::InvalidQuotedTriple { .. } => {
                (ErrorSeverity::Error, ErrorCategory::Semantic)
            }
            StarError::ParseError(_) => (ErrorSeverity::Error, ErrorCategory::Syntax),
            StarError::SerializationError { .. } => (ErrorSeverity::Error, ErrorCategory::Runtime),
            StarError::QueryError { .. } => (ErrorSeverity::Error, ErrorCategory::Syntax),
            StarError::CoreError(_) => (ErrorSeverity::Error, ErrorCategory::Runtime),
            StarError::ReificationError { .. } => (ErrorSeverity::Error, ErrorCategory::Semantic),
            StarError::InvalidTermType { .. } => (ErrorSeverity::Error, ErrorCategory::Semantic),
            StarError::NestingDepthExceeded { .. } => {
                (ErrorSeverity::Error, ErrorCategory::Configuration)
            }
            StarError::UnsupportedFormat { .. } => {
                (ErrorSeverity::Error, ErrorCategory::Configuration)
            }
            StarError::ConfigurationError { .. } => {
                (ErrorSeverity::Error, ErrorCategory::Configuration)
            }
        }
    }

    /// Generate enhanced error message with context
    pub fn formatted_message(&self) -> String {
        let mut message = format!("[{}] {}", self.severity_label(), self.error);

        // Add context information
        if let Some(source) = &self.context.source {
            message.push_str(&format!("\n  Source: {source}"));
        }

        if let (Some(line), Some(column)) = (self.context.line, self.context.column) {
            message.push_str(&format!("\n  Location: line {line}, column {column}"));
        }

        if let Some(operation) = &self.context.operation {
            message.push_str(&format!("\n  Operation: {operation}"));
        }

        if let Some(snippet) = &self.context.snippet {
            message.push_str(&format!(
                "\n  Context:\n    {}",
                snippet.replace('\n', "\n    ")
            ));
        }

        // Add metadata
        if !self.context.metadata.is_empty() {
            message.push_str("\n  Details:");
            for (key, value) in &self.context.metadata {
                message.push_str(&format!("\n    {key}: {value}"));
            }
        }

        // Add suggestions
        if !self.suggestions.is_empty() {
            message.push_str("\n  Suggestions:");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                message.push_str(&format!("\n    {}. {suggestion}", i + 1));
            }
        }

        message
    }

    /// Get severity label
    fn severity_label(&self) -> &'static str {
        match self.severity {
            ErrorSeverity::Critical => "CRITICAL",
            ErrorSeverity::Error => "ERROR",
            ErrorSeverity::Warning => "WARNING",
            ErrorSeverity::Info => "INFO",
        }
    }

    /// Convert to JSON for structured reporting
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "severity": self.severity,
            "category": self.category,
            "message": self.error.to_string(),
            "context": {
                "source": self.context.source,
                "line": self.context.line,
                "column": self.context.column,
                "snippet": self.context.snippet,
                "operation": self.context.operation,
                "metadata": self.context.metadata
            },
            "suggestions": self.suggestions
        })
    }
}

impl fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.formatted_message())
    }
}

/// Error aggregator for collecting and reporting multiple errors
#[derive(Debug, Default)]
pub struct ErrorAggregator {
    errors: Vec<EnhancedError>,
    warnings: Vec<EnhancedError>,
    max_errors: Option<usize>,
    max_warnings: Option<usize>,
}

impl ErrorAggregator {
    /// Create a new error aggregator
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            max_errors: None,
            max_warnings: None,
        }
    }

    /// Set maximum number of errors to collect
    pub fn with_max_errors(mut self, max: usize) -> Self {
        self.max_errors = Some(max);
        self
    }

    /// Set maximum number of warnings to collect
    pub fn with_max_warnings(mut self, max: usize) -> Self {
        self.max_warnings = Some(max);
        self
    }

    /// Add an enhanced error
    pub fn add_error(&mut self, error: EnhancedError) {
        match error.severity {
            ErrorSeverity::Critical | ErrorSeverity::Error => {
                if let Some(max) = self.max_errors {
                    if self.errors.len() >= max {
                        return;
                    }
                }
                self.errors.push(error);
            }
            ErrorSeverity::Warning | ErrorSeverity::Info => {
                if let Some(max) = self.max_warnings {
                    if self.warnings.len() >= max {
                        return;
                    }
                }
                self.warnings.push(error);
            }
        }
    }

    /// Add a StarError with automatic enhancement
    pub fn add_star_error(&mut self, error: StarError, context: Option<ErrorContext>) {
        let enhanced = if let Some(ctx) = context {
            EnhancedError::new(error).with_context(ctx)
        } else {
            EnhancedError::new(error)
        };
        self.add_error(enhanced);
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Check if there are any warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }

    /// Get all errors
    pub fn errors(&self) -> &[EnhancedError] {
        &self.errors
    }

    /// Get all warnings
    pub fn warnings(&self) -> &[EnhancedError] {
        &self.warnings
    }

    /// Generate a comprehensive error report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        // Summary
        report.push_str(&format!(
            "Error Report: {} error(s), {} warning(s)\n",
            self.errors.len(),
            self.warnings.len()
        ));
        report.push_str("═".repeat(50).as_str());
        report.push('\n');

        // Errors
        if !self.errors.is_empty() {
            report.push_str("\nERRORS:\n");
            for (i, error) in self.errors.iter().enumerate() {
                report.push_str(&format!("\n{}. {}\n", i + 1, error.formatted_message()));
            }
        }

        // Warnings
        if !self.warnings.is_empty() {
            report.push_str("\nWARNINGS:\n");
            for (i, warning) in self.warnings.iter().enumerate() {
                report.push_str(&format!("\n{}. {}\n", i + 1, warning.formatted_message()));
            }
        }

        report
    }

    /// Generate structured JSON report
    pub fn generate_json_report(&self) -> serde_json::Value {
        serde_json::json!({
            "summary": {
                "error_count": self.errors.len(),
                "warning_count": self.warnings.len(),
                "has_errors": self.has_errors()
            },
            "errors": self.errors.iter().map(|e| e.to_json()).collect::<Vec<_>>(),
            "warnings": self.warnings.iter().map(|w| w.to_json()).collect::<Vec<_>>()
        })
    }

    /// Clear all collected errors and warnings
    pub fn clear(&mut self) {
        self.errors.clear();
        self.warnings.clear();
    }
}

/// Enhanced result type with error context
pub type EnhancedResult<T> = Result<T, EnhancedError>;

/// Trait for converting StarError to EnhancedError with context
pub trait WithErrorContext<T> {
    /// Add context to an error result
    fn with_context(self, context: ErrorContext) -> EnhancedResult<T>;

    /// Add context and suggestions to an error result
    fn with_context_and_suggestions(
        self,
        context: ErrorContext,
        suggestions: Vec<String>,
    ) -> EnhancedResult<T>;
}

impl<T> WithErrorContext<T> for StarResult<T> {
    fn with_context(self, context: ErrorContext) -> EnhancedResult<T> {
        self.map_err(|e| EnhancedError::new(e).with_context(context))
    }

    fn with_context_and_suggestions(
        self,
        context: ErrorContext,
        suggestions: Vec<String>,
    ) -> EnhancedResult<T> {
        self.map_err(|e| {
            EnhancedError::new(e)
                .with_context(context)
                .with_suggestions(suggestions)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_builder() {
        let context = ErrorContext::new()
            .with_source("test.ttls")
            .with_position(10, 5)
            .with_snippet("<< ex:alice ex:knows ex:bob >> ex:certainty 0.9 .")
            .with_operation("parsing")
            .with_metadata("format", "turtle-star");

        assert_eq!(context.source, Some("test.ttls".to_string()));
        assert_eq!(context.line, Some(10));
        assert_eq!(context.column, Some(5));
        assert!(context.snippet.is_some());
        assert_eq!(context.operation, Some("parsing".to_string()));
        assert_eq!(
            context.metadata.get("format"),
            Some(&"turtle-star".to_string())
        );
    }

    #[test]
    fn test_enhanced_error_classification() {
        let parse_error = StarError::parse_error("Invalid syntax");
        let enhanced = EnhancedError::new(parse_error);

        assert_eq!(enhanced.severity, ErrorSeverity::Error);
        assert_eq!(enhanced.category, ErrorCategory::Syntax);
    }

    #[test]
    fn test_error_aggregator() {
        let mut aggregator = ErrorAggregator::new().with_max_errors(2);

        aggregator.add_star_error(StarError::parse_error("Error 1"), None);
        aggregator.add_star_error(StarError::parse_error("Error 2"), None);
        aggregator.add_star_error(StarError::parse_error("Error 3"), None); // Should be ignored

        assert_eq!(aggregator.error_count(), 2);
        assert!(aggregator.has_errors());
    }

    #[test]
    fn test_enhanced_error_formatting() {
        let context = ErrorContext::new()
            .with_source("test.ttls")
            .with_position(5, 10);

        let enhanced = EnhancedError::new(StarError::parse_error("Invalid token"))
            .with_context(context)
            .with_suggestion("Check syntax around line 5");

        let message = enhanced.formatted_message();
        assert!(message.contains("ERROR"));
        assert!(message.contains("test.ttls"));
        assert!(message.contains("line 5, column 10"));
        assert!(message.contains("Check syntax"));
    }

    #[test]
    fn test_json_report_generation() {
        let mut aggregator = ErrorAggregator::new();
        aggregator.add_star_error(StarError::parse_error("Test error"), None);

        let report = aggregator.generate_json_report();
        assert_eq!(report["summary"]["error_count"], 1);
        assert!(report["errors"].is_array());
    }
}
