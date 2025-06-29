//! # OxiRS RDF-Star
//!
//! RDF-star and SPARQL-star implementation providing comprehensive support for quoted triples.
//!
//! This crate extends the standard RDF model with RDF-star capabilities, allowing triples
//! to be used as subjects or objects in other triples (quoted triples). It provides:
//!
//! - Complete RDF-star data model with proper type safety
//! - Parsing support for Turtle-star, N-Triples-star, TriG-star, and N-Quads-star
//! - SPARQL-star query execution with quoted triple patterns
//! - Serialization to all major RDF-star formats
//! - Storage backend integration with oxirs-core
//! - Performance-optimized handling of nested quoted triples
//!
//! ## Examples
//!
//! ```rust
//! use oxirs_star::{StarStore, StarTriple, StarTerm};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut store = StarStore::new();
//!
//! // Create a quoted triple
//! let quoted = StarTriple::new(
//!     StarTerm::iri("http://example.org/person1")?,
//!     StarTerm::iri("http://example.org/age")?,
//!     StarTerm::literal("25")?,
//! );
//!
//! // Use the quoted triple as a subject
//! let meta_triple = StarTriple::new(
//!     StarTerm::quoted_triple(quoted),
//!     StarTerm::iri("http://example.org/certainty")?,
//!     StarTerm::literal("0.9")?,
//! );
//!
//! store.insert(&meta_triple)?;
//! # Ok(())
//! # }
//! ```

use oxirs_core::OxirsError;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, span, Level};

pub mod cli;
pub mod docs;
pub mod functions;
pub mod model;
pub mod parser;
pub mod profiling;
pub mod query;
pub mod reification;
pub mod serializer;
pub mod store;

// Re-export main types
pub use model::*;
pub use store::StarStore;

/// RDF-star specific error types
#[derive(Debug, Error)]
pub enum StarError {
    #[error("Invalid quoted triple: {message}")]
    InvalidQuotedTriple {
        message: String,
        context: Option<String>,
        suggestion: Option<String>,
    },
    #[error("Parse error in RDF-star format: {message}")]
    ParseError {
        message: String,
        line: Option<usize>,
        column: Option<usize>,
        input_fragment: Option<String>,
        expected: Option<String>,
        suggestion: Option<String>,
    },
    #[error("Serialization error: {message}")]
    SerializationError {
        message: String,
        format: Option<String>,
        context: Option<String>,
    },
    #[error("SPARQL-star query error: {message}")]
    QueryError {
        message: String,
        query_fragment: Option<String>,
        position: Option<usize>,
        suggestion: Option<String>,
    },
    #[error("Core RDF error: {0}")]
    CoreError(#[from] OxirsError),
    #[error("Reification error: {message}")]
    ReificationError {
        message: String,
        reification_strategy: Option<String>,
        context: Option<String>,
    },
    #[error("Invalid term type for RDF-star context: {message}")]
    InvalidTermType {
        message: String,
        term_type: Option<String>,
        expected_types: Option<Vec<String>>,
        suggestion: Option<String>,
    },
    #[error("Nesting depth exceeded: maximum depth {max_depth} reached")]
    NestingDepthExceeded {
        max_depth: usize,
        current_depth: usize,
        context: Option<String>,
    },
    #[error("Format not supported: {format}")]
    UnsupportedFormat {
        format: String,
        available_formats: Vec<String>,
    },
    #[error("Configuration error: {message}")]
    ConfigurationError {
        message: String,
        parameter: Option<String>,
        valid_range: Option<String>,
    },
}

/// Result type for RDF-star operations
pub type StarResult<T> = std::result::Result<T, StarError>;

impl StarError {
    /// Create a simple invalid quoted triple error (backward compatibility)
    pub fn invalid_quoted_triple(message: impl Into<String>) -> Self {
        Self::InvalidQuotedTriple {
            message: message.into(),
            context: None,
            suggestion: None,
        }
    }

    /// Create a simple parse error (backward compatibility)
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::ParseError {
            message: message.into(),
            line: None,
            column: None,
            input_fragment: None,
            expected: None,
            suggestion: None,
        }
    }

    /// Create a simple serialization error (backward compatibility)
    pub fn serialization_error(message: impl Into<String>) -> Self {
        Self::SerializationError {
            message: message.into(),
            format: None,
            context: None,
        }
    }

    /// Create a simple query error (backward compatibility)
    pub fn query_error(message: impl Into<String>) -> Self {
        Self::QueryError {
            message: message.into(),
            query_fragment: None,
            position: None,
            suggestion: None,
        }
    }

    /// Create a simple reification error (backward compatibility)
    pub fn reification_error(message: impl Into<String>) -> Self {
        Self::ReificationError {
            message: message.into(),
            reification_strategy: None,
            context: None,
        }
    }

    /// Create a simple invalid term type error (backward compatibility)
    pub fn invalid_term_type(message: impl Into<String>) -> Self {
        Self::InvalidTermType {
            message: message.into(),
            term_type: None,
            expected_types: None,
            suggestion: None,
        }
    }

    /// Create a nesting depth error
    pub fn nesting_depth_exceeded(
        max_depth: usize,
        current_depth: usize,
        context: Option<String>,
    ) -> Self {
        Self::NestingDepthExceeded {
            max_depth,
            current_depth,
            context,
        }
    }

    /// Create an unsupported format error with available alternatives
    pub fn unsupported_format(format: impl Into<String>, available: Vec<String>) -> Self {
        Self::UnsupportedFormat {
            format: format.into(),
            available_formats: available,
        }
    }

    /// Get available recovery suggestions for the error
    pub fn recovery_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        match self {
            Self::NestingDepthExceeded { max_depth, .. } => {
                suggestions.push(format!(
                    "Consider increasing max_nesting_depth beyond {}",
                    max_depth
                ));
                suggestions.push("Check for circular references in quoted triples".to_string());
            }
            Self::UnsupportedFormat {
                available_formats, ..
            } => {
                suggestions.push(format!(
                    "Supported formats: {}",
                    available_formats.join(", ")
                ));
            }
            Self::ConfigurationError { valid_range, .. } => {
                if let Some(range) = valid_range {
                    suggestions.push(format!("Valid range: {}", range));
                }
            }
            _ => {}
        }

        suggestions
    }
}

/// Configuration for RDF-star processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarConfig {
    /// Maximum nesting depth for quoted triples (default: 10)
    pub max_nesting_depth: usize,
    /// Enable automatic reification fallback
    pub enable_reification_fallback: bool,
    /// Strict mode for parsing (reject invalid constructs)
    pub strict_mode: bool,
    /// Enable SPARQL-star extensions
    pub enable_sparql_star: bool,
    /// Buffer size for streaming operations
    pub buffer_size: usize,
    /// Maximum parse errors before aborting (None for unlimited)
    pub max_parse_errors: Option<usize>,
}

impl Default for StarConfig {
    fn default() -> Self {
        Self {
            max_nesting_depth: 10,
            enable_reification_fallback: true,
            strict_mode: false,
            enable_sparql_star: true,
            buffer_size: 8192,
            max_parse_errors: Some(100),
        }
    }
}

/// Statistics for RDF-star processing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StarStatistics {
    /// Total number of quoted triples processed
    pub quoted_triples_count: usize,
    /// Maximum nesting depth encountered
    pub max_nesting_encountered: usize,
    /// Number of reified triples
    pub reified_triples_count: usize,
    /// Number of SPARQL-star queries executed
    pub sparql_star_queries_count: usize,
    /// Processing time statistics (in microseconds)
    pub processing_time_us: u64,
}

/// Initialize the RDF-star system with configuration
pub fn init_star_system(config: StarConfig) -> StarResult<()> {
    let span = span!(Level::INFO, "init_star_system");
    let _enter = span.enter();

    info!("Initializing OxiRS RDF-star system");
    debug!("Configuration: {:?}", config);

    // Validate configuration
    if config.max_nesting_depth == 0 {
        return Err(StarError::ConfigurationError {
            message: "Max nesting depth must be greater than 0".to_string(),
            parameter: Some("max_nesting_depth".to_string()),
            valid_range: Some("1..=1000".to_string()),
        });
    }

    if config.buffer_size == 0 {
        return Err(StarError::ConfigurationError {
            message: "Buffer size must be greater than 0".to_string(),
            parameter: Some("buffer_size".to_string()),
            valid_range: Some("1..=1048576".to_string()),
        });
    }

    // Additional validation for reasonable limits
    if config.max_nesting_depth > 1000 {
        return Err(StarError::ConfigurationError {
            message: "Max nesting depth is too large and may cause performance issues".to_string(),
            parameter: Some("max_nesting_depth".to_string()),
            valid_range: Some("1..=1000".to_string()),
        });
    }

    info!("RDF-star system initialized successfully");
    Ok(())
}

/// Utility function to validate quoted triple nesting depth
pub fn validate_nesting_depth(term: &StarTerm, max_depth: usize) -> StarResult<()> {
    fn check_depth(term: &StarTerm, current_depth: usize, max_depth: usize) -> StarResult<usize> {
        match term {
            StarTerm::QuotedTriple(triple) => {
                if current_depth >= max_depth {
                    return Err(StarError::InvalidQuotedTriple {
                        message: format!(
                            "Nesting depth {} exceeds maximum {}",
                            current_depth, max_depth
                        ),
                        context: None,
                        suggestion: None,
                    });
                }

                let subj_depth = check_depth(&triple.subject, current_depth + 1, max_depth)?;
                let pred_depth = check_depth(&triple.predicate, current_depth + 1, max_depth)?;
                let obj_depth = check_depth(&triple.object, current_depth + 1, max_depth)?;

                Ok(subj_depth.max(pred_depth).max(obj_depth))
            }
            _ => Ok(current_depth),
        }
    }

    check_depth(term, 0, max_depth)?;
    Ok(())
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Developer tooling and debugging utilities
pub mod dev_tools {
    use super::*;
    use std::collections::HashMap;

    /// RDF-star format detection result
    #[derive(Debug, Clone, PartialEq)]
    pub enum DetectedFormat {
        TurtleStar,
        NTriplesStar,
        TrigStar,
        NQuadsStar,
        Unknown,
    }

    /// Detect RDF-star format from input content
    pub fn detect_format(content: &str) -> DetectedFormat {
        let content = content.trim();

        // Check for TriG-star indicators
        if content.contains("GRAPH") || content.contains("{") && content.contains("}") {
            return DetectedFormat::TrigStar;
        }

        // Check for N-Quads-star (4 terms per line)
        let lines: Vec<&str> = content
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
            .collect();
        if !lines.is_empty() {
            let first_line = lines[0].trim();
            let terms: Vec<&str> = first_line.split_whitespace().collect();
            if terms.len() >= 4 && first_line.ends_with('.') {
                return DetectedFormat::NQuadsStar;
            }
        }

        // Check for quoted triples (RDF-star indicator)
        if content.contains("<<") && content.contains(">>") {
            // If has quotes and prefixes, likely Turtle-star
            if content.contains("@prefix") || content.contains("PREFIX") {
                return DetectedFormat::TurtleStar;
            }
            // Otherwise, likely N-Triples-star
            return DetectedFormat::NTriplesStar;
        }

        // Check for Turtle-star prefixes
        if content.contains("@prefix") || content.contains("@base") {
            return DetectedFormat::TurtleStar;
        }

        DetectedFormat::Unknown
    }

    /// Validate RDF-star content and return detailed diagnostic information
    pub fn validate_content(content: &str, config: &StarConfig) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Basic format detection
        result.detected_format = detect_format(content);

        // Count quoted triples
        let quoted_count = content.matches("<<").count();
        result.quoted_triple_count = quoted_count;

        // Check for potential issues
        if quoted_count > 10000 && !config.enable_reification_fallback {
            result.warnings.push("Large number of quoted triples detected. Consider enabling reification fallback for better performance.".to_string());
        }

        // Check nesting depth by counting nested <<
        let max_nesting = find_max_nesting_depth(content);
        result.max_nesting_depth = max_nesting;

        if max_nesting > config.max_nesting_depth {
            result.errors.push(format!(
                "Nesting depth {} exceeds configured maximum {}",
                max_nesting, config.max_nesting_depth
            ));
        }

        // Check for common syntax issues
        check_syntax_issues(content, &mut result);

        result
    }

    /// Validation result with detailed diagnostics
    #[derive(Debug, Clone)]
    pub struct ValidationResult {
        pub detected_format: DetectedFormat,
        pub quoted_triple_count: usize,
        pub max_nesting_depth: usize,
        pub errors: Vec<String>,
        pub warnings: Vec<String>,
        pub suggestions: Vec<String>,
        pub line_errors: HashMap<u32, String>,
    }

    impl ValidationResult {
        fn new() -> Self {
            Self {
                detected_format: DetectedFormat::Unknown,
                quoted_triple_count: 0,
                max_nesting_depth: 0,
                errors: Vec::new(),
                warnings: Vec::new(),
                suggestions: Vec::new(),
                line_errors: HashMap::new(),
            }
        }

        /// Check if validation passed without errors
        pub fn is_valid(&self) -> bool {
            self.errors.is_empty()
        }

        /// Get a summary report of the validation
        pub fn summary(&self) -> String {
            let mut summary = String::new();
            summary.push_str(&format!("Format: {:?}\n", self.detected_format));
            summary.push_str(&format!("Quoted triples: {}\n", self.quoted_triple_count));
            summary.push_str(&format!("Max nesting depth: {}\n", self.max_nesting_depth));

            if !self.errors.is_empty() {
                summary.push_str(&format!("Errors: {}\n", self.errors.len()));
            }

            if !self.warnings.is_empty() {
                summary.push_str(&format!("Warnings: {}\n", self.warnings.len()));
            }

            summary
        }
    }

    fn find_max_nesting_depth(content: &str) -> usize {
        let mut max_depth = 0;
        let mut current_depth = 0;

        for ch in content.chars() {
            match ch {
                '<' => {
                    // Look ahead for another '<' to detect quoted triple start
                    current_depth += 1;
                }
                '>' => {
                    if current_depth > 0 {
                        current_depth -= 1;
                    }
                }
                _ => {}
            }
            max_depth = max_depth.max(current_depth / 2); // Divide by 2 since we count both < and >
        }

        max_depth
    }

    fn check_syntax_issues(content: &str, result: &mut ValidationResult) {
        let lines: Vec<&str> = content.lines().collect();

        for (line_num, line) in lines.iter().enumerate() {
            let line_num = line_num as u32 + 1;
            let trimmed = line.trim();

            // Skip comments and empty lines
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Check for unmatched quoted triple brackets
            let open_count = trimmed.matches("<<").count();
            let close_count = trimmed.matches(">>").count();

            if open_count != close_count {
                result.line_errors.insert(
                    line_num,
                    format!(
                        "Unmatched quoted triple brackets: {} << vs {} >>",
                        open_count, close_count
                    ),
                );
            }

            // Check for missing periods in N-Triples/N-Quads style
            if result.detected_format == DetectedFormat::NTriplesStar
                || result.detected_format == DetectedFormat::NQuadsStar
            {
                if !trimmed.ends_with('.')
                    && !trimmed.starts_with('@')
                    && !trimmed.starts_with("PREFIX")
                {
                    result.warnings.push(format!(
                        "Line {}: Missing period at end of statement",
                        line_num
                    ));
                }
            }
        }
    }

    /// Performance profiler for RDF-star operations
    pub struct StarProfiler {
        start_time: std::time::Instant,
        operation_times: HashMap<String, u64>,
    }

    impl StarProfiler {
        pub fn new() -> Self {
            Self {
                start_time: std::time::Instant::now(),
                operation_times: HashMap::new(),
            }
        }

        pub fn time_operation<F, R>(&mut self, name: &str, operation: F) -> R
        where
            F: FnOnce() -> R,
        {
            let start = std::time::Instant::now();
            let result = operation();
            let duration = start.elapsed().as_micros() as u64;
            self.operation_times.insert(name.to_string(), duration);
            result
        }

        pub fn get_stats(&self) -> HashMap<String, u64> {
            self.operation_times.clone()
        }

        pub fn total_time(&self) -> u64 {
            self.start_time.elapsed().as_micros() as u64
        }
    }

    /// Generate a diagnostic report for RDF-star content
    pub fn generate_diagnostic_report(content: &str, config: &StarConfig) -> String {
        let validation = validate_content(content, config);
        let mut report = String::new();

        report.push_str("=== RDF-star Diagnostic Report ===\n\n");
        report.push_str(&validation.summary());

        if !validation.errors.is_empty() {
            report.push_str("\nErrors:\n");
            for (i, error) in validation.errors.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, error));
            }
        }

        if !validation.warnings.is_empty() {
            report.push_str("\nWarnings:\n");
            for (i, warning) in validation.warnings.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, warning));
            }
        }

        if !validation.suggestions.is_empty() {
            report.push_str("\nSuggestions:\n");
            for (i, suggestion) in validation.suggestions.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, suggestion));
            }
        }

        if !validation.line_errors.is_empty() {
            report.push_str("\nLine-specific issues:\n");
            let mut sorted_lines: Vec<_> = validation.line_errors.iter().collect();
            sorted_lines.sort_by_key(|(line, _)| *line);

            for (line, error) in sorted_lines {
                report.push_str(&format!("  Line {}: {}\n", line, error));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = StarConfig::default();
        assert_eq!(config.max_nesting_depth, 10);
        assert!(config.enable_reification_fallback);
        assert!(!config.strict_mode);
        assert!(config.enable_sparql_star);
    }

    #[test]
    fn test_nesting_depth_validation() {
        let simple_term = StarTerm::iri("http://example.org/test").unwrap();
        assert!(validate_nesting_depth(&simple_term, 5).is_ok());

        // Test nested quoted triple
        let inner_triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );
        let nested_term = StarTerm::quoted_triple(inner_triple);
        assert!(validate_nesting_depth(&nested_term, 5).is_ok());
        assert!(validate_nesting_depth(&nested_term, 0).is_err());
    }
}
