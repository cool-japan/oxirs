//! GraphQL field-level validation rules engine.
//!
//! Provides a declarative rule system for validating GraphQL field values
//! (strings and numbers) with configurable constraints and structured
//! violation reporting.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FieldRule
// ---------------------------------------------------------------------------

/// A validation rule that can be attached to a named field.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldRule {
    /// Field value must be present (non-empty for strings).
    Required,
    /// String value must be at least `n` characters long.
    MinLength(usize),
    /// String value must be at most `n` characters long.
    MaxLength(usize),
    /// String value must match the given regular expression pattern.
    /// (Stored as a plain string; matching uses a simple prefix/contains check
    /// for the no-external-dep constraint — real regex support would need the
    /// `regex` crate.)
    Pattern(String),
    /// Numeric value must fall within `[min, max]` (inclusive).
    Range { min: f64, max: f64 },
    /// A custom rule identified by a string key (passes through with a
    /// user-defined message).
    Custom(String),
}

impl FieldRule {
    /// A short label used in violation reports.
    pub fn label(&self) -> String {
        match self {
            FieldRule::Required => "required".to_string(),
            FieldRule::MinLength(n) => format!("min_length({n})"),
            FieldRule::MaxLength(n) => format!("max_length({n})"),
            FieldRule::Pattern(p) => format!("pattern({p})"),
            FieldRule::Range { min, max } => format!("range({min},{max})"),
            FieldRule::Custom(k) => format!("custom({k})"),
        }
    }
}

// ---------------------------------------------------------------------------
// FieldViolation
// ---------------------------------------------------------------------------

/// A single validation violation for a field.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldViolation {
    /// Name of the field that failed validation.
    pub field: String,
    /// Label of the rule that was violated.
    pub rule: String,
    /// Human-readable description of the violation.
    pub message: String,
}

impl FieldViolation {
    /// Create a new violation.
    pub fn new(
        field: impl Into<String>,
        rule: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            field: field.into(),
            rule: rule.into(),
            message: message.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// Collects `FieldViolation` instances produced during a validation pass.
#[derive(Debug, Default, Clone)]
pub struct ValidationReport {
    violations: Vec<FieldViolation>,
}

impl ValidationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a violation into the report.
    pub fn add(&mut self, violation: FieldViolation) {
        self.violations.push(violation);
    }

    /// Merge another report into this one.
    pub fn merge(&mut self, other: ValidationReport) {
        self.violations.extend(other.violations);
    }

    /// Return `true` when there are no violations.
    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }

    /// All recorded violations.
    pub fn violations(&self) -> &[FieldViolation] {
        &self.violations
    }

    /// Total number of violations.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Filter violations by field name.
    pub fn violations_for(&self, field: &str) -> Vec<&FieldViolation> {
        self.violations
            .iter()
            .filter(|v| v.field == field)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// FieldValidator
// ---------------------------------------------------------------------------

/// Associates `FieldRule`s with named fields and validates values against them.
#[derive(Debug, Default)]
pub struct FieldValidator {
    /// Rules keyed by field name; a field may have multiple rules.
    rules: HashMap<String, Vec<FieldRule>>,
}

impl FieldValidator {
    /// Create a new, empty `FieldValidator`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `rule` for `field`.
    pub fn add_rule(&mut self, field: &str, rule: FieldRule) {
        self.rules.entry(field.to_string()).or_default().push(rule);
    }

    /// Total number of rules across all fields.
    pub fn rule_count(&self) -> usize {
        self.rules.values().map(|v| v.len()).sum()
    }

    /// Names of all fields that have at least one rule.
    pub fn fields(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.rules.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    // -----------------------------------------------------------------------
    // String validation
    // -----------------------------------------------------------------------

    /// Validate a string `value` against every rule registered for `field`.
    pub fn validate_string(&self, field: &str, value: &str) -> Vec<FieldViolation> {
        let mut violations = Vec::new();

        let rules = match self.rules.get(field) {
            Some(r) => r,
            None => return violations,
        };

        for rule in rules {
            match rule {
                FieldRule::Required => {
                    if value.trim().is_empty() {
                        violations.push(FieldViolation::new(
                            field,
                            rule.label(),
                            format!("field '{field}' is required and must not be empty"),
                        ));
                    }
                }
                FieldRule::MinLength(n) => {
                    if value.len() < *n {
                        violations.push(FieldViolation::new(
                            field,
                            rule.label(),
                            format!(
                                "field '{field}' must be at least {n} characters \
                                 (got {})",
                                value.len()
                            ),
                        ));
                    }
                }
                FieldRule::MaxLength(n) => {
                    if value.len() > *n {
                        violations.push(FieldViolation::new(
                            field,
                            rule.label(),
                            format!(
                                "field '{field}' must be at most {n} characters \
                                 (got {})",
                                value.len()
                            ),
                        ));
                    }
                }
                FieldRule::Pattern(pat) => {
                    // Simple pattern check: treat `pat` as a required substring.
                    // This avoids a `regex` dependency while still exercising the rule.
                    if !value.contains(pat.as_str()) {
                        violations.push(FieldViolation::new(
                            field,
                            rule.label(),
                            format!("field '{field}' does not match the required pattern '{pat}'"),
                        ));
                    }
                }
                FieldRule::Range { min, max } => {
                    // For strings, interpret the value as a number if possible.
                    match value.trim().parse::<f64>() {
                        Ok(n) if n < *min || n > *max => {
                            violations.push(FieldViolation::new(
                                field,
                                rule.label(),
                                format!(
                                    "field '{field}' value {n} is outside the range \
                                     [{min}, {max}]"
                                ),
                            ));
                        }
                        Err(_) => {
                            violations.push(FieldViolation::new(
                                field,
                                rule.label(),
                                format!(
                                    "field '{field}' value '{value}' is not a number; \
                                     cannot check range [{min}, {max}]"
                                ),
                            ));
                        }
                        Ok(_) => {} // within range, no violation
                    }
                }
                FieldRule::Custom(key) => {
                    // Custom rules always pass at the engine level; the caller
                    // implements the semantics.  We record a violation only when
                    // the value is empty to serve as a sensible default demo.
                    if value.is_empty() {
                        violations.push(FieldViolation::new(
                            field,
                            rule.label(),
                            format!("custom rule '{key}' failed for field '{field}'"),
                        ));
                    }
                }
            }
        }

        violations
    }

    // -----------------------------------------------------------------------
    // Numeric validation
    // -----------------------------------------------------------------------

    /// Validate a numeric `value` against every rule registered for `field`.
    pub fn validate_number(&self, field: &str, value: f64) -> Vec<FieldViolation> {
        let mut violations = Vec::new();

        let rules = match self.rules.get(field) {
            Some(r) => r,
            None => return violations,
        };

        for rule in rules {
            match rule {
                FieldRule::Required => {
                    // NaN is considered "not present".
                    if value.is_nan() {
                        violations.push(FieldViolation::new(
                            field,
                            rule.label(),
                            format!("field '{field}' is required (got NaN)"),
                        ));
                    }
                }
                FieldRule::MinLength(n) => {
                    // For numbers, treat MinLength as a minimum integer value.
                    if (value as i64) < (*n as i64) {
                        violations.push(FieldViolation::new(
                            field,
                            rule.label(),
                            format!(
                                "field '{field}' numeric value {value} \
                                 is below the minimum {n}"
                            ),
                        ));
                    }
                }
                FieldRule::MaxLength(n) => {
                    // For numbers, treat MaxLength as a maximum integer value.
                    if (value as i64) > (*n as i64) {
                        violations.push(FieldViolation::new(
                            field,
                            rule.label(),
                            format!(
                                "field '{field}' numeric value {value} \
                                 exceeds the maximum {n}"
                            ),
                        ));
                    }
                }
                FieldRule::Pattern(_) => {
                    // Pattern does not apply to numeric values — skip silently.
                }
                FieldRule::Range { min, max } => {
                    if value < *min || value > *max {
                        violations.push(FieldViolation::new(
                            field,
                            rule.label(),
                            format!(
                                "field '{field}' value {value} is outside the range \
                                 [{min}, {max}]"
                            ),
                        ));
                    }
                }
                FieldRule::Custom(key) => {
                    // Same "demo" semantics: NaN fails.
                    if value.is_nan() {
                        violations.push(FieldViolation::new(
                            field,
                            rule.label(),
                            format!("custom rule '{key}' failed for field '{field}'"),
                        ));
                    }
                }
            }
        }

        violations
    }

    // -----------------------------------------------------------------------
    // Map validation
    // -----------------------------------------------------------------------

    /// Validate a map of field → string value.
    ///
    /// For each field in the map that has registered rules, `validate_string`
    /// is called.  Fields without rules are ignored.  Fields with rules but
    /// absent from `values` are treated as empty strings.
    pub fn validate_map(&self, values: &HashMap<String, String>) -> Vec<FieldViolation> {
        let mut violations = Vec::new();

        for field in self.rules.keys() {
            let value = values.get(field).map(|s| s.as_str()).unwrap_or("");
            violations.extend(self.validate_string(field, value));
        }

        // Sort for determinism.
        violations.sort_by(|a, b| a.field.cmp(&b.field).then(a.rule.cmp(&b.rule)));
        violations
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // FieldRule
    // -----------------------------------------------------------------------

    #[test]
    fn test_rule_label_required() {
        assert_eq!(FieldRule::Required.label(), "required");
    }

    #[test]
    fn test_rule_label_min_length() {
        assert_eq!(FieldRule::MinLength(3).label(), "min_length(3)");
    }

    #[test]
    fn test_rule_label_max_length() {
        assert_eq!(FieldRule::MaxLength(100).label(), "max_length(100)");
    }

    #[test]
    fn test_rule_label_pattern() {
        assert_eq!(
            FieldRule::Pattern("@example.com".to_string()).label(),
            "pattern(@example.com)"
        );
    }

    #[test]
    fn test_rule_label_range() {
        assert_eq!(
            FieldRule::Range { min: 0.0, max: 1.0 }.label(),
            "range(0,1)"
        );
    }

    #[test]
    fn test_rule_label_custom() {
        assert_eq!(
            FieldRule::Custom("email_dns".to_string()).label(),
            "custom(email_dns)"
        );
    }

    // -----------------------------------------------------------------------
    // FieldViolation
    // -----------------------------------------------------------------------

    #[test]
    fn test_violation_new() {
        let v = FieldViolation::new("email", "required", "email is required");
        assert_eq!(v.field, "email");
        assert_eq!(v.rule, "required");
        assert_eq!(v.message, "email is required");
    }

    // -----------------------------------------------------------------------
    // ValidationReport
    // -----------------------------------------------------------------------

    #[test]
    fn test_report_empty_is_valid() {
        let report = ValidationReport::new();
        assert!(report.is_valid());
        assert_eq!(report.violation_count(), 0);
    }

    #[test]
    fn test_report_add_violation() {
        let mut report = ValidationReport::new();
        report.add(FieldViolation::new("x", "required", "msg"));
        assert!(!report.is_valid());
        assert_eq!(report.violation_count(), 1);
    }

    #[test]
    fn test_report_merge() {
        let mut a = ValidationReport::new();
        a.add(FieldViolation::new("x", "required", "msg"));
        let mut b = ValidationReport::new();
        b.add(FieldViolation::new("y", "required", "msg"));
        a.merge(b);
        assert_eq!(a.violation_count(), 2);
    }

    #[test]
    fn test_report_violations_for() {
        let mut report = ValidationReport::new();
        report.add(FieldViolation::new("email", "required", "m1"));
        report.add(FieldViolation::new("name", "required", "m2"));
        let email_violations = report.violations_for("email");
        assert_eq!(email_violations.len(), 1);
    }

    // -----------------------------------------------------------------------
    // FieldValidator::new / add_rule / rule_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_validator_new_is_empty() {
        let v = FieldValidator::new();
        assert_eq!(v.rule_count(), 0);
    }

    #[test]
    fn test_validator_add_rule_increments_count() {
        let mut v = FieldValidator::new();
        v.add_rule("name", FieldRule::Required);
        assert_eq!(v.rule_count(), 1);
    }

    #[test]
    fn test_validator_multiple_rules_for_field() {
        let mut v = FieldValidator::new();
        v.add_rule("name", FieldRule::Required);
        v.add_rule("name", FieldRule::MinLength(2));
        v.add_rule("name", FieldRule::MaxLength(50));
        assert_eq!(v.rule_count(), 3);
    }

    #[test]
    fn test_validator_fields_returns_sorted_names() {
        let mut v = FieldValidator::new();
        v.add_rule("zebra", FieldRule::Required);
        v.add_rule("alpha", FieldRule::Required);
        assert_eq!(v.fields(), vec!["alpha", "zebra"]);
    }

    // -----------------------------------------------------------------------
    // validate_string — Required
    // -----------------------------------------------------------------------

    #[test]
    fn test_string_required_passes_non_empty() {
        let mut v = FieldValidator::new();
        v.add_rule("email", FieldRule::Required);
        let viols = v.validate_string("email", "user@example.com");
        assert!(viols.is_empty());
    }

    #[test]
    fn test_string_required_fails_empty() {
        let mut v = FieldValidator::new();
        v.add_rule("email", FieldRule::Required);
        let viols = v.validate_string("email", "");
        assert_eq!(viols.len(), 1);
        assert_eq!(viols[0].rule, "required");
    }

    #[test]
    fn test_string_required_fails_whitespace_only() {
        let mut v = FieldValidator::new();
        v.add_rule("name", FieldRule::Required);
        let viols = v.validate_string("name", "   ");
        assert_eq!(viols.len(), 1);
    }

    // -----------------------------------------------------------------------
    // validate_string — MinLength / MaxLength
    // -----------------------------------------------------------------------

    #[test]
    fn test_string_min_length_passes() {
        let mut v = FieldValidator::new();
        v.add_rule("username", FieldRule::MinLength(3));
        let viols = v.validate_string("username", "alice");
        assert!(viols.is_empty());
    }

    #[test]
    fn test_string_min_length_fails() {
        let mut v = FieldValidator::new();
        v.add_rule("username", FieldRule::MinLength(5));
        let viols = v.validate_string("username", "bob");
        assert_eq!(viols.len(), 1);
        assert!(viols[0].rule.contains("min_length"));
    }

    #[test]
    fn test_string_max_length_passes() {
        let mut v = FieldValidator::new();
        v.add_rule("bio", FieldRule::MaxLength(10));
        let viols = v.validate_string("bio", "short");
        assert!(viols.is_empty());
    }

    #[test]
    fn test_string_max_length_fails() {
        let mut v = FieldValidator::new();
        v.add_rule("bio", FieldRule::MaxLength(5));
        let viols = v.validate_string("bio", "this is too long");
        assert_eq!(viols.len(), 1);
        assert!(viols[0].rule.contains("max_length"));
    }

    // -----------------------------------------------------------------------
    // validate_string — Pattern
    // -----------------------------------------------------------------------

    #[test]
    fn test_string_pattern_passes() {
        let mut v = FieldValidator::new();
        v.add_rule("email", FieldRule::Pattern("@".to_string()));
        let viols = v.validate_string("email", "user@example.com");
        assert!(viols.is_empty());
    }

    #[test]
    fn test_string_pattern_fails() {
        let mut v = FieldValidator::new();
        v.add_rule("email", FieldRule::Pattern("@".to_string()));
        let viols = v.validate_string("email", "notanemail");
        assert_eq!(viols.len(), 1);
        assert!(viols[0].rule.contains("pattern"));
    }

    // -----------------------------------------------------------------------
    // validate_string — Range
    // -----------------------------------------------------------------------

    #[test]
    fn test_string_range_passes_numeric() {
        let mut v = FieldValidator::new();
        v.add_rule(
            "age",
            FieldRule::Range {
                min: 0.0,
                max: 150.0,
            },
        );
        let viols = v.validate_string("age", "25");
        assert!(viols.is_empty());
    }

    #[test]
    fn test_string_range_fails_out_of_range() {
        let mut v = FieldValidator::new();
        v.add_rule(
            "age",
            FieldRule::Range {
                min: 0.0,
                max: 150.0,
            },
        );
        let viols = v.validate_string("age", "200");
        assert_eq!(viols.len(), 1);
        assert!(viols[0].rule.contains("range"));
    }

    #[test]
    fn test_string_range_fails_non_numeric() {
        let mut v = FieldValidator::new();
        v.add_rule(
            "age",
            FieldRule::Range {
                min: 0.0,
                max: 150.0,
            },
        );
        let viols = v.validate_string("age", "old");
        assert_eq!(viols.len(), 1);
    }

    // -----------------------------------------------------------------------
    // validate_number — Range
    // -----------------------------------------------------------------------

    #[test]
    fn test_number_range_passes() {
        let mut v = FieldValidator::new();
        v.add_rule(
            "score",
            FieldRule::Range {
                min: 0.0,
                max: 100.0,
            },
        );
        let viols = v.validate_number("score", 75.0);
        assert!(viols.is_empty());
    }

    #[test]
    fn test_number_range_fails_below() {
        let mut v = FieldValidator::new();
        v.add_rule(
            "score",
            FieldRule::Range {
                min: 0.0,
                max: 100.0,
            },
        );
        let viols = v.validate_number("score", -1.0);
        assert_eq!(viols.len(), 1);
    }

    #[test]
    fn test_number_range_fails_above() {
        let mut v = FieldValidator::new();
        v.add_rule(
            "score",
            FieldRule::Range {
                min: 0.0,
                max: 100.0,
            },
        );
        let viols = v.validate_number("score", 101.0);
        assert_eq!(viols.len(), 1);
    }

    #[test]
    fn test_number_required_fails_nan() {
        let mut v = FieldValidator::new();
        v.add_rule("value", FieldRule::Required);
        let viols = v.validate_number("value", f64::NAN);
        assert_eq!(viols.len(), 1);
    }

    #[test]
    fn test_number_required_passes_finite() {
        let mut v = FieldValidator::new();
        v.add_rule("value", FieldRule::Required);
        let viols = v.validate_number("value", 42.0);
        assert!(viols.is_empty());
    }

    #[test]
    fn test_number_pattern_ignored() {
        let mut v = FieldValidator::new();
        v.add_rule("n", FieldRule::Pattern("abc".to_string()));
        // Pattern rules are silently skipped for numbers.
        let viols = v.validate_number("n", 999.0);
        assert!(viols.is_empty());
    }

    // -----------------------------------------------------------------------
    // validate_map
    // -----------------------------------------------------------------------

    #[test]
    fn test_map_all_valid() {
        let mut v = FieldValidator::new();
        v.add_rule("name", FieldRule::Required);
        v.add_rule("email", FieldRule::Pattern("@".to_string()));
        let mut values = HashMap::new();
        values.insert("name".to_string(), "Alice".to_string());
        values.insert("email".to_string(), "alice@example.com".to_string());
        let viols = v.validate_map(&values);
        assert!(viols.is_empty());
    }

    #[test]
    fn test_map_missing_field_treated_as_empty() {
        let mut v = FieldValidator::new();
        v.add_rule("name", FieldRule::Required);
        let values: HashMap<String, String> = HashMap::new();
        let viols = v.validate_map(&values);
        assert_eq!(viols.len(), 1);
    }

    #[test]
    fn test_map_multiple_violations() {
        let mut v = FieldValidator::new();
        v.add_rule("name", FieldRule::Required);
        v.add_rule("email", FieldRule::Pattern("@".to_string()));
        let mut values = HashMap::new();
        values.insert("email".to_string(), "notanemail".to_string());
        // name is absent → empty → Required fails
        let viols = v.validate_map(&values);
        assert_eq!(viols.len(), 2);
    }

    #[test]
    fn test_map_extra_fields_ignored() {
        let mut v = FieldValidator::new();
        v.add_rule("name", FieldRule::Required);
        let mut values = HashMap::new();
        values.insert("name".to_string(), "Alice".to_string());
        values.insert("unknown_field".to_string(), "x".to_string());
        let viols = v.validate_map(&values);
        assert!(viols.is_empty());
    }

    // -----------------------------------------------------------------------
    // validate_string — no rules for field
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_rules_returns_empty() {
        let v = FieldValidator::new();
        let viols = v.validate_string("anything", "value");
        assert!(viols.is_empty());
    }

    // -----------------------------------------------------------------------
    // Combined rules
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_violations_on_one_field() {
        let mut v = FieldValidator::new();
        v.add_rule("pass", FieldRule::MinLength(8));
        v.add_rule("pass", FieldRule::MaxLength(4)); // impossible range → both fire
        let viols = v.validate_string("pass", "abc"); // length 3: < 8 AND ≤ 4 (ok for max)
                                                      // Only MinLength fires (3 < 8); MaxLength passes (3 ≤ 4).
        assert_eq!(viols.len(), 1);
    }

    #[test]
    fn test_both_min_and_max_fail() {
        let mut v = FieldValidator::new();
        v.add_rule("f", FieldRule::MinLength(10));
        v.add_rule("f", FieldRule::MaxLength(2));
        // "hello" length 5: < 10 (min fails) and > 2 (max fails)
        let viols = v.validate_string("f", "hello");
        assert_eq!(viols.len(), 2);
    }

    #[test]
    fn test_custom_rule_fails_empty_string() {
        let mut v = FieldValidator::new();
        v.add_rule("token", FieldRule::Custom("jwt_validate".to_string()));
        let viols = v.validate_string("token", "");
        assert_eq!(viols.len(), 1);
        assert!(viols[0].rule.contains("custom"));
    }

    #[test]
    fn test_custom_rule_passes_non_empty_string() {
        let mut v = FieldValidator::new();
        v.add_rule("token", FieldRule::Custom("jwt_validate".to_string()));
        let viols = v.validate_string("token", "eyJhbGci...");
        assert!(viols.is_empty());
    }

    #[test]
    fn test_range_boundary_values_pass() {
        let mut v = FieldValidator::new();
        v.add_rule("pct", FieldRule::Range { min: 0.0, max: 1.0 });
        assert!(v.validate_number("pct", 0.0).is_empty());
        assert!(v.validate_number("pct", 1.0).is_empty());
    }

    #[test]
    fn test_range_boundary_exclusive_fails() {
        let mut v = FieldValidator::new();
        v.add_rule("pct", FieldRule::Range { min: 0.0, max: 1.0 });
        // Slightly outside boundary
        assert_eq!(v.validate_number("pct", -0.001).len(), 1);
        assert_eq!(v.validate_number("pct", 1.001).len(), 1);
    }
}
