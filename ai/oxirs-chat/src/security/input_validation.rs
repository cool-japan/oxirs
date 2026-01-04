//! Input validation and sanitization for security

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub sanitized_input: String,
    pub violations: Vec<ValidationViolation>,
    pub risk_score: f64,
}

impl ValidationResult {
    pub fn valid(sanitized: String) -> Self {
        Self {
            is_valid: true,
            sanitized_input: sanitized,
            violations: vec![],
            risk_score: 0.0,
        }
    }

    pub fn invalid(violations: Vec<ValidationViolation>, risk_score: f64) -> Self {
        Self {
            is_valid: false,
            sanitized_input: String::new(),
            violations,
            risk_score,
        }
    }
}

/// Validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    pub violation_type: ViolationType,
    pub description: String,
    pub severity: Severity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    SqlInjection,
    XssAttempt,
    CommandInjection,
    PathTraversal,
    ExcessiveLength,
    MalformedInput,
    SuspiciousPattern,
    ProhibitedContent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Input validator
pub struct InputValidator {
    max_input_size: usize,
    max_tokens: usize,
    sql_injection_patterns: Vec<Regex>,
    xss_patterns: Vec<Regex>,
    command_injection_patterns: Vec<Regex>,
    path_traversal_patterns: Vec<Regex>,
}

impl InputValidator {
    pub fn new(max_input_size: usize, max_tokens: usize) -> Self {
        Self {
            max_input_size,
            max_tokens,
            sql_injection_patterns: Self::build_sql_patterns(),
            xss_patterns: Self::build_xss_patterns(),
            command_injection_patterns: Self::build_command_patterns(),
            path_traversal_patterns: Self::build_path_patterns(),
        }
    }

    /// Validate input
    pub fn validate(&self, input: &str) -> Result<ValidationResult> {
        let mut violations = Vec::new();
        let mut risk_score = 0.0;

        // Check length
        if input.len() > self.max_input_size {
            violations.push(ValidationViolation {
                violation_type: ViolationType::ExcessiveLength,
                description: format!("Input exceeds maximum size of {} bytes", self.max_input_size),
                severity: Severity::High,
            });
            risk_score += 0.5;
        }

        // Check for SQL injection
        if let Some(violation) = self.check_sql_injection(input) {
            violations.push(violation);
            risk_score += 0.8;
        }

        // Check for XSS
        if let Some(violation) = self.check_xss(input) {
            violations.push(violation);
            risk_score += 0.7;
        }

        // Check for command injection
        if let Some(violation) = self.check_command_injection(input) {
            violations.push(violation);
            risk_score += 0.9;
        }

        // Check for path traversal
        if let Some(violation) = self.check_path_traversal(input) {
            violations.push(violation);
            risk_score += 0.6;
        }

        // If any critical violations, reject
        if violations.iter().any(|v| v.severity == Severity::Critical) {
            return Ok(ValidationResult::invalid(violations, risk_score));
        }

        // Sanitize input
        let sanitized = self.sanitize(input);

        if violations.is_empty() {
            Ok(ValidationResult::valid(sanitized))
        } else {
            Ok(ValidationResult {
                is_valid: risk_score < 0.5,
                sanitized_input: sanitized,
                violations,
                risk_score,
            })
        }
    }

    fn check_sql_injection(&self, input: &str) -> Option<ValidationViolation> {
        for pattern in &self.sql_injection_patterns {
            if pattern.is_match(input) {
                return Some(ValidationViolation {
                    violation_type: ViolationType::SqlInjection,
                    description: "Potential SQL injection detected".to_string(),
                    severity: Severity::Critical,
                });
            }
        }
        None
    }

    fn check_xss(&self, input: &str) -> Option<ValidationViolation> {
        for pattern in &self.xss_patterns {
            if pattern.is_match(input) {
                return Some(ValidationViolation {
                    violation_type: ViolationType::XssAttempt,
                    description: "Potential XSS attack detected".to_string(),
                    severity: Severity::High,
                });
            }
        }
        None
    }

    fn check_command_injection(&self, input: &str) -> Option<ValidationViolation> {
        for pattern in &self.command_injection_patterns {
            if pattern.is_match(input) {
                return Some(ValidationViolation {
                    violation_type: ViolationType::CommandInjection,
                    description: "Potential command injection detected".to_string(),
                    severity: Severity::Critical,
                });
            }
        }
        None
    }

    fn check_path_traversal(&self, input: &str) -> Option<ValidationViolation> {
        for pattern in &self.path_traversal_patterns {
            if pattern.is_match(input) {
                return Some(ValidationViolation {
                    violation_type: ViolationType::PathTraversal,
                    description: "Potential path traversal detected".to_string(),
                    severity: Severity::High,
                });
            }
        }
        None
    }

    fn sanitize(&self, input: &str) -> String {
        // Basic sanitization: remove null bytes, control characters
        input
            .chars()
            .filter(|c| !c.is_control() || c.is_whitespace())
            .collect()
    }

    fn build_sql_patterns() -> Vec<Regex> {
        vec![
            Regex::new(r"(?i)(union\s+select)").expect("valid regex pattern"),
            Regex::new(r"(?i)(drop\s+table)").expect("valid regex pattern"),
            Regex::new(r"(?i)(delete\s+from)").expect("valid regex pattern"),
            Regex::new(r"(?i)(insert\s+into)").expect("valid regex pattern"),
            Regex::new(r"(?i)(--\s*$)").expect("valid regex pattern"),
            Regex::new(r"(?i)(;\s*drop)").expect("valid regex pattern"),
        ]
    }

    fn build_xss_patterns() -> Vec<Regex> {
        vec![
            Regex::new(r"(?i)(<script.*?>)").expect("valid regex pattern"),
            Regex::new(r"(?i)(javascript:)").expect("valid regex pattern"),
            Regex::new(r"(?i)(onerror\s*=)").expect("valid regex pattern"),
            Regex::new(r"(?i)(onload\s*=)").expect("valid regex pattern"),
        ]
    }

    fn build_command_patterns() -> Vec<Regex> {
        vec![
            Regex::new(r"[;&|`$]").expect("valid regex pattern"),
            Regex::new(r"(?i)(rm\s+-rf)").expect("valid regex pattern"),
            Regex::new(r"(?i)(wget|curl)\s+http").expect("valid regex pattern"),
        ]
    }

    fn build_path_patterns() -> Vec<Regex> {
        vec![
            Regex::new(r"\.\.\/").expect("valid regex pattern"),
            Regex::new(r"\.\.\\").expect("valid regex pattern"),
            Regex::new(r"(?i)(\/etc\/passwd)").expect("valid regex pattern"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_input() {
        let validator = InputValidator::new(1024 * 1024, 4096);
        let result = validator.validate("SELECT ?s ?p ?o WHERE { ?s ?p ?o }").unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_sql_injection_detection() {
        let validator = InputValidator::new(1024 * 1024, 4096);
        let result = validator.validate("'; DROP TABLE users; --").unwrap();
        assert!(!result.is_valid);
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_xss_detection() {
        let validator = InputValidator::new(1024 * 1024, 4096);
        let result = validator.validate("<script>alert('xss')</script>").unwrap();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_command_injection_detection() {
        let validator = InputValidator::new(1024 * 1024, 4096);
        let result = validator.validate("test; rm -rf /").unwrap();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_excessive_length() {
        let validator = InputValidator::new(100, 4096);
        let long_input = "a".repeat(200);
        let result = validator.validate(&long_input).unwrap();
        assert!(!result.violations.is_empty());
    }
}
