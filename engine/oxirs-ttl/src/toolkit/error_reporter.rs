//! Enhanced error reporting with code snippets and suggestions
//!
//! This module provides utilities for creating user-friendly error messages
//! with context, code snippets, and helpful suggestions for fixing common issues.
//!
//! # Examples
//!
//! ```rust
//! use oxirs_ttl::toolkit::error_reporter::{ErrorReporter, ErrorSuggestion};
//! use oxirs_ttl::error::TextPosition;
//!
//! let source = r#"
//! @prefix ex: <http://example.org/> .
//! ex:subject ex:predicate "unclosed string
//! "#;
//!
//! let reporter = ErrorReporter::new(source);
//! let pos = TextPosition::new(3, 30, 60);
//!
//! let report = reporter.format_error(
//!     "Unterminated string literal",
//!     pos,
//!     Some("String literals must be closed with a matching quote"),
//! );
//!
//! println!("{}", report);
//! ```

use crate::error::TextPosition;
use std::fmt;

/// Suggested fix for an error
#[derive(Debug, Clone)]
pub struct ErrorSuggestion {
    /// Description of the suggestion
    pub message: String,
    /// Optional replacement text
    pub replacement: Option<String>,
    /// Whether this is the primary suggestion
    pub primary: bool,
}

impl ErrorSuggestion {
    /// Create a new suggestion
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            replacement: None,
            primary: true,
        }
    }

    /// Create a suggestion with replacement text
    pub fn with_replacement(message: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            replacement: Some(replacement.into()),
            primary: true,
        }
    }

    /// Mark this as a secondary suggestion
    pub fn secondary(mut self) -> Self {
        self.primary = false;
        self
    }
}

/// Enhanced error reporter with code snippets
pub struct ErrorReporter<'a> {
    #[allow(dead_code)]
    source: &'a str,
    lines: Vec<&'a str>,
    context_lines: usize,
}

impl<'a> ErrorReporter<'a> {
    /// Create a new error reporter for the given source code
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            lines: source.lines().collect(),
            context_lines: 2,
        }
    }

    /// Set the number of context lines to show before and after the error
    pub fn with_context_lines(mut self, lines: usize) -> Self {
        self.context_lines = lines;
        self
    }

    /// Format an error with code snippet and context
    pub fn format_error(
        &self,
        message: &str,
        position: TextPosition,
        help: Option<&str>,
    ) -> String {
        let mut output = String::new();

        // Error header
        output.push_str(&format!("\n┌─ Error at {}\n", position));
        output.push_str("│\n");
        output.push_str(&format!("│  {}\n", message));

        // Code snippet with context
        if let Some(snippet) = self.get_code_snippet(position) {
            output.push_str("│\n");
            output.push_str(&snippet);
        }

        // Help message
        if let Some(help_text) = help {
            output.push_str("│\n");
            output.push_str(&format!("│  Help: {}\n", help_text));
        }

        output.push_str("└─\n");
        output
    }

    /// Format an error with suggestions
    pub fn format_error_with_suggestions(
        &self,
        message: &str,
        position: TextPosition,
        suggestions: Vec<ErrorSuggestion>,
    ) -> String {
        let mut output = self.format_error(message, position, None);

        if !suggestions.is_empty() {
            output.push_str("│\n");
            output.push_str("│  Suggestions:\n");
            for (i, suggestion) in suggestions.iter().enumerate() {
                let prefix = if suggestion.primary { "→" } else { " " };
                output.push_str(&format!(
                    "│  {} {}. {}\n",
                    prefix,
                    i + 1,
                    suggestion.message
                ));
                if let Some(ref replacement) = suggestion.replacement {
                    output.push_str(&format!("│     Try: {}\n", replacement));
                }
            }
            output.push_str("└─\n");
        }

        output
    }

    /// Get a code snippet with context around the error position
    fn get_code_snippet(&self, position: TextPosition) -> Option<String> {
        if self.lines.is_empty() || position.line == 0 || position.line > self.lines.len() {
            return None;
        }

        let mut snippet = String::new();
        let error_line = position.line - 1; // Convert to 0-based

        // Calculate range of lines to show
        let start_line = error_line.saturating_sub(self.context_lines);
        let end_line = (error_line + self.context_lines + 1).min(self.lines.len());

        // Show line numbers with padding
        let max_line_num = end_line;
        let line_num_width = max_line_num.to_string().len();

        for line_idx in start_line..end_line {
            let line_num = line_idx + 1;
            let line_content = self.lines[line_idx];

            // Line number and content
            if line_idx == error_line {
                snippet.push_str(&format!(
                    "│ {:>width$} │ {}\n",
                    line_num,
                    line_content,
                    width = line_num_width
                ));

                // Error indicator
                let indicator =
                    "^".repeat(5.min(line_content.len().saturating_sub(position.column - 1) + 1));
                snippet.push_str(&format!(
                    "│ {:>width$} │ {}{}",
                    "",
                    " ".repeat(position.column.saturating_sub(1)),
                    indicator,
                    width = line_num_width
                ));
                snippet.push_str(" here\n");
            } else {
                snippet.push_str(&format!(
                    "│ {:>width$} │ {}\n",
                    line_num,
                    line_content,
                    width = line_num_width
                ));
            }
        }

        Some(snippet)
    }

    /// Get common suggestions for syntax errors
    pub fn get_common_suggestions(error_type: &str) -> Vec<ErrorSuggestion> {
        match error_type {
            "unterminated_string" => vec![
                ErrorSuggestion::new("Add a closing quote (\") to terminate the string literal"),
                ErrorSuggestion::with_replacement(
                    "Use triple quotes for multi-line strings",
                    "\"\"\"...\"\"\"",
                )
                .secondary(),
            ],
            "missing_dot" => vec![ErrorSuggestion::new(
                "Add a period (.) at the end of the statement",
            )],
            "invalid_iri" => vec![
                ErrorSuggestion::new("Ensure the IRI is enclosed in angle brackets: <http://...>"),
                ErrorSuggestion::with_replacement(
                    "Use a prefixed name instead",
                    "prefix:localName",
                )
                .secondary(),
            ],
            "undefined_prefix" => vec![
                ErrorSuggestion::new("Define the prefix with @prefix directive before using it"),
                ErrorSuggestion::with_replacement(
                    "Example prefix definition",
                    "@prefix ex: <http://example.org/> .",
                )
                .secondary(),
            ],
            "missing_predicate" => vec![
                ErrorSuggestion::new("Add a predicate between subject and object"),
                ErrorSuggestion::with_replacement(
                    "Use 'a' as shorthand for rdf:type",
                    "subject a ClassName .",
                )
                .secondary(),
            ],
            _ => vec![],
        }
    }
}

/// Format a simple error message
pub fn format_simple_error(message: &str, position: TextPosition) -> String {
    format!("Error at {}: {}", position, message)
}

/// Create a detailed error report from source and position
pub fn create_error_report(
    source: &str,
    message: &str,
    position: TextPosition,
    error_type: Option<&str>,
) -> String {
    let reporter = ErrorReporter::new(source);

    if let Some(err_type) = error_type {
        let suggestions = ErrorReporter::get_common_suggestions(err_type);
        if !suggestions.is_empty() {
            return reporter.format_error_with_suggestions(message, position, suggestions);
        }
    }

    reporter.format_error(message, position, None)
}

impl fmt::Display for ErrorSuggestion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(ref replacement) = self.replacement {
            write!(f, " (try: {})", replacement)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_reporter_basic() {
        let source = "ex:subject ex:predicate \"value\" .";
        let reporter = ErrorReporter::new(source);
        let pos = TextPosition::new(1, 10, 9);

        let report = reporter.format_error("Test error", pos, Some("This is help text"));
        assert!(report.contains("Test error"));
        assert!(report.contains("This is help text"));
        assert!(report.contains("ex:subject"));
    }

    #[test]
    fn test_error_reporter_with_context() {
        let source = "@prefix ex: <http://example.org/> .\nex:subject ex:predicate \"value\" .\nex:another ex:prop ex:obj .";
        let reporter = ErrorReporter::new(source).with_context_lines(1);
        let pos = TextPosition::new(2, 15, 50);

        let report = reporter.format_error("Syntax error", pos, None);
        assert!(report.contains("Syntax error"));
        assert!(report.contains("ex:subject"));
    }

    #[test]
    fn test_suggestions() {
        let suggestions = ErrorReporter::get_common_suggestions("unterminated_string");
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].primary);
    }

    #[test]
    fn test_error_with_suggestions() {
        let source = "ex:subject ex:predicate \"unclosed";
        let reporter = ErrorReporter::new(source);
        let pos = TextPosition::new(1, 25, 24);
        let suggestions = ErrorReporter::get_common_suggestions("unterminated_string");

        let report =
            reporter.format_error_with_suggestions("Unterminated string", pos, suggestions);
        assert!(report.contains("Unterminated string"));
        assert!(report.contains("Suggestions"));
    }

    #[test]
    fn test_simple_error_format() {
        let pos = TextPosition::new(5, 10, 100);
        let msg = format_simple_error("Test error", pos);
        assert!(msg.contains("line 5, column 10"));
        assert!(msg.contains("Test error"));
    }
}
