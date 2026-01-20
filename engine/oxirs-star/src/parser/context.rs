//! Parser context and state management for RDF-star parsing
//!
//! This module provides the core data structures for managing parsing state,
//! including namespace prefixes, blank node generation, error tracking, and
//! TriG-star graph context management.

use std::collections::HashMap;
use std::fmt;

use crate::model::StarTerm;
use crate::{StarError, StarResult};

/// Enhanced state tracking for TriG-star parsing
///
/// Manages the complex state machine required for parsing TriG-star format,
/// including graph block nesting, graph name parsing, and context transitions.
#[derive(Debug, Default)]
pub(super) struct TrigParserState {
    /// Current graph context
    pub(super) current_graph: Option<StarTerm>,
    /// Nesting level (for tracking braces)
    pub(super) brace_depth: usize,
    /// Whether we're inside a graph block
    pub(super) in_graph_block: bool,
    /// Buffer for accumulating multi-line graph names
    pub(super) graph_name_buffer: String,
    /// Whether we're parsing a graph name declaration
    pub(super) parsing_graph_name: bool,
}

impl TrigParserState {
    /// Create a new TriG parser state
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Reset graph context to default state
    pub(super) fn reset_graph_context(&mut self) {
        self.current_graph = None;
        self.in_graph_block = false;
        self.brace_depth = 0;
        self.graph_name_buffer.clear();
        self.parsing_graph_name = false;
    }

    /// Enter a graph block with the given graph name
    pub(super) fn enter_graph_block(&mut self, graph: Option<StarTerm>) {
        self.current_graph = graph;
        self.in_graph_block = true;
        self.brace_depth += 1;
        self.parsing_graph_name = false;
        self.graph_name_buffer.clear();
    }

    /// Exit the current graph block
    ///
    /// Returns true if we've completely exited all nested graph blocks
    pub(super) fn exit_graph_block(&mut self) -> bool {
        if self.brace_depth > 0 {
            self.brace_depth -= 1;
            if self.brace_depth == 0 {
                self.reset_graph_context();
                return true;
            }
        }
        false
    }
}

/// Error severity levels for parser errors
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSeverity {
    /// Warning - parsing can continue
    Warning,
    /// Error - current statement failed but parsing can continue
    Error,
    /// Fatal - parsing must stop
    Fatal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Warning => write!(f, "Warning"),
            ErrorSeverity::Error => write!(f, "Error"),
            ErrorSeverity::Fatal => write!(f, "Fatal"),
        }
    }
}

/// Enhanced error information for parser errors
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Error message
    pub message: String,
    /// Line number where error occurred
    pub line: usize,
    /// Column position where error occurred
    pub column: usize,
    /// Context around the error (nearby text)
    pub context: String,
    /// Severity level
    pub severity: ErrorSeverity,
}

/// Parser context for maintaining state during parsing
///
/// This structure tracks all mutable state during parsing, including:
/// - Namespace prefix mappings
/// - Base IRI
/// - Blank node generation
/// - Position tracking for error reporting
/// - Accumulated parsing errors
#[derive(Debug, Default)]
pub(super) struct ParseContext {
    /// Namespace prefixes
    pub(super) prefixes: HashMap<String, String>,
    /// Current base IRI
    pub(super) base_iri: Option<String>,
    /// Current graph name (for TriG/N-Quads)
    #[allow(dead_code)]
    pub(super) current_graph: Option<StarTerm>,
    /// Blank node counter
    pub(super) blank_node_counter: usize,
    /// Current line number for error reporting
    pub(super) line_number: usize,
    /// Current column position for error reporting
    pub(super) column_position: usize,
    /// Strict parsing mode (reject any malformed input)
    pub(super) strict_mode: bool,
    /// Error recovery mode (try to continue parsing after errors)
    pub(super) error_recovery: bool,
    /// Accumulated parsing errors for batch reporting
    pub(super) parsing_errors: Vec<ParseError>,
}

impl ParseContext {
    /// Create a new default parsing context
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Create context with configuration
    pub(super) fn with_config(strict_mode: bool, error_recovery: bool) -> Self {
        Self {
            strict_mode,
            error_recovery,
            ..Default::default()
        }
    }

    /// Update position tracking
    pub(super) fn update_position(&mut self, line: usize, column: usize) {
        self.line_number = line;
        self.column_position = column;
    }

    /// Add a parsing error with context
    pub(super) fn add_error(&mut self, message: String, context: String, severity: ErrorSeverity) {
        let error = ParseError {
            message,
            line: self.line_number,
            column: self.column_position,
            context,
            severity,
        };
        self.parsing_errors.push(error);
    }

    /// Check if fatal errors occurred
    pub(super) fn has_fatal_errors(&self) -> bool {
        self.parsing_errors
            .iter()
            .any(|e| e.severity == ErrorSeverity::Fatal)
    }

    /// Get all errors
    pub(super) fn get_errors(&self) -> &[ParseError] {
        &self.parsing_errors
    }

    /// Clear errors
    #[allow(dead_code)]
    pub(super) fn clear_errors(&mut self) {
        self.parsing_errors.clear();
    }

    /// Generate a new blank node identifier
    pub(super) fn next_blank_node(&mut self) -> String {
        self.blank_node_counter += 1;
        let counter = self.blank_node_counter;
        format!("_:b{counter}")
    }

    /// Resolve a prefixed name to full IRI with enhanced error reporting
    pub(super) fn resolve_prefix(&mut self, prefixed: &str) -> StarResult<String> {
        if let Some(colon_pos) = prefixed.find(':') {
            let prefix = &prefixed[..colon_pos];
            let local = &prefixed[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                Ok(format!("{namespace}{local}"))
            } else {
                let error_msg = format!("Unknown prefix: '{prefix}'");
                let context = format!("in prefixed name '{prefixed}'");

                if self.strict_mode {
                    self.add_error(error_msg.clone(), context, ErrorSeverity::Fatal);
                    Err(StarError::parse_error(error_msg))
                } else {
                    self.add_error(error_msg.clone(), context, ErrorSeverity::Warning);
                    // In non-strict mode, return the prefixed name as-is
                    Ok(prefixed.to_string())
                }
            }
        } else {
            let error_msg = format!("Invalid prefixed name: '{prefixed}'");
            self.add_error(
                error_msg.clone(),
                prefixed.to_string(),
                ErrorSeverity::Error,
            );
            Err(StarError::parse_error(error_msg))
        }
    }

    /// Resolve relative IRI against base
    pub(super) fn resolve_relative(&self, iri: &str) -> String {
        if let Some(ref base) = self.base_iri {
            // Simple relative IRI resolution (not fully RFC compliant)
            if iri.starts_with('#') {
                format!("{base}{iri}")
            } else {
                iri.to_string()
            }
        } else {
            iri.to_string()
        }
    }

    /// Try to recover from parsing error
    pub(super) fn try_recover_from_error(&mut self, error_context: &str) -> bool {
        if !self.error_recovery {
            return false;
        }

        // Add recovery attempt to error log
        let recovery_msg = format!("Attempting error recovery from: {error_context}");
        self.add_error(
            recovery_msg,
            error_context.to_string(),
            ErrorSeverity::Warning,
        );

        // Simple recovery strategies could be implemented here
        // For now, just indicate that recovery is possible
        true
    }
}
