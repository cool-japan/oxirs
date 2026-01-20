//! Error recovery strategies for TTL parser
//!
//! This module provides strategies for recovering from parse errors,
//! allowing the parser to continue processing even when encountering
//! malformed input. This is particularly useful for:
//!
//! - IDE integrations (incremental parsing with errors)
//! - Linting tools (show all errors, not just the first one)
//! - Partial model loading (load what's valid, skip what's not)
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_samm::parser::error_recovery::{ErrorRecoveryStrategy, RecoveryAction};
//! use oxirs_samm::error::SammError;
//!
//! let strategy = ErrorRecoveryStrategy::default();
//!
//! // When encountering a parse error
//! let error = SammError::ParseError("Invalid triple".to_string());
//! let action = strategy.recover(&error, "ttl content");
//!
//! match action {
//!     RecoveryAction::Skip => {
//!         // Skip this statement and continue
//!     }
//!     RecoveryAction::Insert(text) => {
//!         // Insert missing text and retry
//!     }
//!     RecoveryAction::Abort => {
//!         // Error is fatal, stop parsing
//!     }
//!     RecoveryAction::UseDefault(value) => {
//!         // Use a default value and continue
//!     }
//!     RecoveryAction::Replace(text) => {
//!         // Replace with corrected text
//!     }
//! }
//! ```

use crate::error::{SammError, SourceLocation};
use std::collections::HashMap;

/// Action to take when recovering from an error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Skip the current statement and continue parsing
    Skip,
    /// Insert missing text and retry parsing
    Insert(String),
    /// Abort parsing (error is fatal)
    Abort,
    /// Use a default value and continue
    UseDefault(String),
    /// Replace with corrected text
    Replace(String),
}

/// Error recovery strategy configuration
#[derive(Debug, Clone)]
pub struct ErrorRecoveryStrategy {
    /// Maximum number of errors to tolerate before aborting
    pub max_errors: usize,
    /// Enable automatic correction of common typos
    pub auto_correct_typos: bool,
    /// Enable insertion of missing punctuation
    pub auto_insert_punctuation: bool,
    /// Enable skipping of malformed statements
    pub skip_malformed: bool,
    /// Enable use of default values for missing elements
    pub use_defaults: bool,
    /// Custom recovery rules (pattern -> action)
    pub custom_rules: HashMap<String, RecoveryAction>,
}

impl Default for ErrorRecoveryStrategy {
    fn default() -> Self {
        Self {
            max_errors: 100,
            auto_correct_typos: true,
            auto_insert_punctuation: true,
            skip_malformed: true,
            use_defaults: false,
            custom_rules: HashMap::new(),
        }
    }
}

impl ErrorRecoveryStrategy {
    /// Create a strict recovery strategy (aborts on first error)
    pub fn strict() -> Self {
        Self {
            max_errors: 1,
            auto_correct_typos: false,
            auto_insert_punctuation: false,
            skip_malformed: false,
            use_defaults: false,
            custom_rules: HashMap::new(),
        }
    }

    /// Create a lenient recovery strategy (tries hard to recover)
    pub fn lenient() -> Self {
        Self {
            max_errors: 1000,
            auto_correct_typos: true,
            auto_insert_punctuation: true,
            skip_malformed: true,
            use_defaults: true,
            custom_rules: HashMap::new(),
        }
    }

    /// Attempt to recover from a parse error
    pub fn recover(&self, error: &SammError, context: &str) -> RecoveryAction {
        // Check custom rules first
        if let Some(action) = self.check_custom_rules(error) {
            return action;
        }

        // Apply recovery strategies based on error type
        match error {
            SammError::ParseError(msg) | SammError::ParseErrorWithLocation { message: msg, .. } => {
                self.recover_from_parse_error(msg, context)
            }
            SammError::ValidationError(msg)
            | SammError::ValidationErrorWithLocation { message: msg, .. } => {
                self.recover_from_validation_error(msg)
            }
            SammError::MissingElement(elem) => {
                if self.use_defaults {
                    RecoveryAction::UseDefault(format!("urn:samm:default:1.0.0#{}", elem))
                } else {
                    RecoveryAction::Abort
                }
            }
            SammError::InvalidUrn(msg) => {
                if self.auto_correct_typos {
                    self.try_correct_urn(msg)
                } else {
                    RecoveryAction::Skip
                }
            }
            _ => RecoveryAction::Abort,
        }
    }

    fn check_custom_rules(&self, error: &SammError) -> Option<RecoveryAction> {
        let error_msg = match error {
            SammError::ParseError(msg) => msg,
            SammError::ParseErrorWithLocation { message, .. } => message,
            SammError::ValidationError(msg) => msg,
            SammError::InvalidUrn(msg) => msg,
            _ => return None,
        };

        for (pattern, action) in &self.custom_rules {
            if error_msg.contains(pattern) {
                return Some(action.clone());
            }
        }

        None
    }

    fn recover_from_parse_error(&self, msg: &str, context: &str) -> RecoveryAction {
        // Missing semicolon
        if (msg.contains("expected ';'") || msg.contains("missing semicolon"))
            && self.auto_insert_punctuation
        {
            return RecoveryAction::Insert(";".to_string());
        }

        // Missing period
        if (msg.contains("expected '.'") || msg.contains("missing period"))
            && self.auto_insert_punctuation
        {
            return RecoveryAction::Insert(".".to_string());
        }

        // Unclosed bracket
        if (msg.contains("unclosed bracket") || msg.contains("expected ']'"))
            && self.auto_insert_punctuation
        {
            return RecoveryAction::Insert("]".to_string());
        }

        // Unclosed parenthesis
        if (msg.contains("unclosed parenthesis") || msg.contains("expected ')"))
            && self.auto_insert_punctuation
        {
            return RecoveryAction::Insert(")".to_string());
        }

        // Invalid prefix
        if msg.contains("undefined prefix") && self.auto_correct_typos {
            return self.try_correct_prefix(msg, context);
        }

        // Malformed triple - skip if allowed
        if (msg.contains("malformed triple") || msg.contains("invalid syntax"))
            && self.skip_malformed
        {
            return RecoveryAction::Skip;
        }

        RecoveryAction::Abort
    }

    fn recover_from_validation_error(&self, msg: &str) -> RecoveryAction {
        // Missing required property
        if msg.contains("missing required") && self.use_defaults {
            return RecoveryAction::UseDefault("default_value".to_string());
        }

        // Invalid data type
        if msg.contains("invalid type") && self.auto_correct_typos {
            return self.try_correct_datatype(msg);
        }

        RecoveryAction::Skip
    }

    fn try_correct_urn(&self, msg: &str) -> RecoveryAction {
        // Common URN typos
        if msg.contains("urn:bamm:") {
            // Old BAMM namespace, should be SAMM
            return RecoveryAction::Replace("urn:samm:".to_string());
        }

        if msg.contains("missing '#'") {
            return RecoveryAction::Insert("#".to_string());
        }

        if msg.contains("missing version") {
            return RecoveryAction::Insert("1.0.0".to_string());
        }

        RecoveryAction::Skip
    }

    fn try_correct_prefix(&self, msg: &str, context: &str) -> RecoveryAction {
        // Common prefix typos
        let common_prefixes = vec![
            (
                "samm",
                "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .",
            ),
            (
                "samm-c",
                "@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#> .",
            ),
            (
                "samm-e",
                "@prefix samm-e: <urn:samm:org.eclipse.esmf.samm:entity:2.3.0#> .",
            ),
            ("xsd", "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> ."),
        ];

        for (prefix, definition) in common_prefixes {
            if msg.contains(prefix) {
                return RecoveryAction::Insert(definition.to_string());
            }
        }

        RecoveryAction::Skip
    }

    fn try_correct_datatype(&self, msg: &str) -> RecoveryAction {
        // Common datatype typos
        let corrections = vec![
            ("String", "xsd:string"),
            ("Integer", "xsd:integer"),
            ("Boolean", "xsd:boolean"),
            ("Float", "xsd:float"),
            ("Double", "xsd:double"),
            ("Date", "xsd:date"),
            ("DateTime", "xsd:dateTime"),
        ];

        for (typo, correct) in corrections {
            if msg.contains(typo) {
                return RecoveryAction::Replace(correct.to_string());
            }
        }

        RecoveryAction::Skip
    }

    /// Add a custom recovery rule
    pub fn add_custom_rule(&mut self, pattern: String, action: RecoveryAction) {
        self.custom_rules.insert(pattern, action);
    }

    /// Check if an error is recoverable with current strategy
    pub fn is_recoverable(&self, error: &SammError) -> bool {
        !matches!(self.recover(error, ""), RecoveryAction::Abort)
    }
}

/// Error recovery context for tracking recovery state
#[derive(Debug)]
pub struct RecoveryContext {
    /// Number of errors encountered
    pub error_count: usize,
    /// Errors that were recovered from
    pub recovered_errors: Vec<(SammError, RecoveryAction)>,
    /// Errors that could not be recovered
    pub fatal_errors: Vec<SammError>,
    /// Recovery strategy
    pub strategy: ErrorRecoveryStrategy,
}

impl RecoveryContext {
    /// Create a new recovery context
    pub fn new(strategy: ErrorRecoveryStrategy) -> Self {
        Self {
            error_count: 0,
            recovered_errors: Vec::new(),
            fatal_errors: Vec::new(),
            strategy,
        }
    }

    /// Record a recovered error
    pub fn record_recovery(&mut self, error: SammError, action: RecoveryAction) {
        self.error_count += 1;
        self.recovered_errors.push((error, action));
    }

    /// Record a fatal error
    pub fn record_fatal(&mut self, error: SammError) {
        self.error_count += 1;
        self.fatal_errors.push(error);
    }

    /// Check if max errors exceeded
    pub fn is_max_errors_exceeded(&self) -> bool {
        self.error_count >= self.strategy.max_errors
    }

    /// Get total error count
    pub fn total_errors(&self) -> usize {
        self.error_count
    }

    /// Get recovery success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.error_count == 0 {
            1.0
        } else {
            self.recovered_errors.len() as f64 / self.error_count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_strategy() {
        let strategy = ErrorRecoveryStrategy::default();
        assert_eq!(strategy.max_errors, 100);
        assert!(strategy.auto_correct_typos);
        assert!(strategy.skip_malformed);
    }

    #[test]
    fn test_strict_strategy() {
        let strategy = ErrorRecoveryStrategy::strict();
        assert_eq!(strategy.max_errors, 1);
        assert!(!strategy.auto_correct_typos);
        assert!(!strategy.skip_malformed);
    }

    #[test]
    fn test_lenient_strategy() {
        let strategy = ErrorRecoveryStrategy::lenient();
        assert_eq!(strategy.max_errors, 1000);
        assert!(strategy.auto_correct_typos);
        assert!(strategy.use_defaults);
    }

    #[test]
    fn test_recover_missing_semicolon() {
        let strategy = ErrorRecoveryStrategy::default();
        let error = SammError::ParseError("expected ';'".to_string());
        let action = strategy.recover(&error, "");

        assert_eq!(action, RecoveryAction::Insert(";".to_string()));
    }

    #[test]
    fn test_recover_malformed_triple() {
        let strategy = ErrorRecoveryStrategy::default();
        let error = SammError::ParseError("malformed triple".to_string());
        let action = strategy.recover(&error, "");

        assert_eq!(action, RecoveryAction::Skip);
    }

    #[test]
    fn test_recover_old_bamm_namespace() {
        let strategy = ErrorRecoveryStrategy::default();
        let error = SammError::InvalidUrn("urn:bamm: is deprecated".to_string());
        let action = strategy.recover(&error, "");

        assert_eq!(action, RecoveryAction::Replace("urn:samm:".to_string()));
    }

    #[test]
    fn test_custom_recovery_rule() {
        let mut strategy = ErrorRecoveryStrategy::default();
        strategy.add_custom_rule(
            "my custom error".to_string(),
            RecoveryAction::UseDefault("custom_value".to_string()),
        );

        let error = SammError::ParseError("my custom error occurred".to_string());
        let action = strategy.recover(&error, "");

        assert_eq!(
            action,
            RecoveryAction::UseDefault("custom_value".to_string())
        );
    }

    #[test]
    fn test_is_recoverable() {
        let strategy = ErrorRecoveryStrategy::default();
        let recoverable = SammError::ParseError("expected ';'".to_string());
        let fatal = SammError::Other("unknown error".to_string());

        assert!(strategy.is_recoverable(&recoverable));
        assert!(!strategy.is_recoverable(&fatal));
    }

    #[test]
    fn test_recovery_context() {
        let strategy = ErrorRecoveryStrategy::default();
        let mut context = RecoveryContext::new(strategy);

        assert_eq!(context.total_errors(), 0);
        assert_eq!(context.success_rate(), 1.0);

        context.record_recovery(
            SammError::ParseError("test".to_string()),
            RecoveryAction::Skip,
        );
        assert_eq!(context.total_errors(), 1);
        assert_eq!(context.success_rate(), 1.0);

        context.record_fatal(SammError::Other("fatal".to_string()));
        assert_eq!(context.total_errors(), 2);
        assert_eq!(context.success_rate(), 0.5);
    }

    #[test]
    fn test_max_errors_exceeded() {
        let strategy = ErrorRecoveryStrategy {
            max_errors: 2,
            ..Default::default()
        };
        let mut context = RecoveryContext::new(strategy);

        assert!(!context.is_max_errors_exceeded());

        context.record_recovery(
            SammError::ParseError("test1".to_string()),
            RecoveryAction::Skip,
        );
        assert!(!context.is_max_errors_exceeded());

        context.record_recovery(
            SammError::ParseError("test2".to_string()),
            RecoveryAction::Skip,
        );
        assert!(context.is_max_errors_exceeded());
    }

    #[test]
    fn test_datatype_correction() {
        let strategy = ErrorRecoveryStrategy::default();
        let error = SammError::ValidationError("invalid type String".to_string());
        let action = strategy.recover(&error, "");

        assert_eq!(action, RecoveryAction::Replace("xsd:string".to_string()));
    }
}
