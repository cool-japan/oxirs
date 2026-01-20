//! Comprehensive data validation for vector operations
//!
//! This module provides validation utilities for vectors, indices, and search operations
//! to ensure data integrity and catch errors early.

use crate::{Vector, VectorPrecision};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Validation severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Informational message
    Info,
    /// Warning that should be addressed
    Warning,
    /// Error that must be fixed
    Error,
    /// Critical error that prevents operation
    Critical,
}

/// Validation rule violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// Severity level
    pub severity: ValidationSeverity,
    /// Rule that was violated
    pub rule: String,
    /// Detailed message
    pub message: String,
    /// Optional context
    pub context: Option<String>,
}

impl ValidationViolation {
    pub fn new(
        severity: ValidationSeverity,
        rule: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            rule: rule.into(),
            message: message.into(),
            context: None,
        }
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed (no errors or critical issues)
    pub passed: bool,
    /// List of violations
    pub violations: Vec<ValidationViolation>,
    /// Validation timestamp
    pub timestamp: u64,
}

impl ValidationResult {
    pub fn success() -> Self {
        Self {
            passed: true,
            violations: Vec::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    pub fn with_violations(violations: Vec<ValidationViolation>) -> Self {
        let passed = !violations.iter().any(|v| {
            matches!(
                v.severity,
                ValidationSeverity::Error | ValidationSeverity::Critical
            )
        });

        Self {
            passed,
            violations,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    pub fn has_errors(&self) -> bool {
        self.violations.iter().any(|v| {
            matches!(
                v.severity,
                ValidationSeverity::Error | ValidationSeverity::Critical
            )
        })
    }

    pub fn has_warnings(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity == ValidationSeverity::Warning)
    }

    pub fn error_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| {
                matches!(
                    v.severity,
                    ValidationSeverity::Error | ValidationSeverity::Critical
                )
            })
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == ValidationSeverity::Warning)
            .count()
    }
}

/// Vector validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorValidationRules {
    /// Minimum allowed dimensions
    pub min_dimensions: Option<usize>,
    /// Maximum allowed dimensions
    pub max_dimensions: Option<usize>,
    /// Require normalized vectors (L2 norm = 1)
    pub require_normalized: bool,
    /// Tolerance for normalization check
    pub normalization_tolerance: f32,
    /// Check for NaN or infinite values
    pub check_for_invalid_values: bool,
    /// Check for zero vectors
    pub disallow_zero_vectors: bool,
    /// Expected precision (if any)
    pub expected_precision: Option<VectorPrecision>,
    /// Minimum non-zero values (for sparse vectors)
    pub min_non_zero: Option<usize>,
    /// Maximum magnitude
    pub max_magnitude: Option<f32>,
}

impl Default for VectorValidationRules {
    fn default() -> Self {
        Self {
            min_dimensions: Some(1),
            max_dimensions: None,
            require_normalized: false,
            normalization_tolerance: 1e-6,
            check_for_invalid_values: true,
            disallow_zero_vectors: false,
            expected_precision: None,
            min_non_zero: None,
            max_magnitude: None,
        }
    }
}

/// Vector validator
pub struct VectorValidator {
    rules: VectorValidationRules,
}

impl VectorValidator {
    pub fn new(rules: VectorValidationRules) -> Self {
        Self { rules }
    }

    pub fn with_default_rules() -> Self {
        Self::new(VectorValidationRules::default())
    }

    /// Validate a single vector
    pub fn validate(&self, vector: &Vector) -> ValidationResult {
        let mut violations = Vec::new();

        // Check dimensions
        if let Some(min_dim) = self.rules.min_dimensions {
            if vector.dimensions < min_dim {
                violations.push(ValidationViolation::new(
                    ValidationSeverity::Error,
                    "min_dimensions",
                    format!(
                        "Vector has {} dimensions, minimum is {}",
                        vector.dimensions, min_dim
                    ),
                ));
            }
        }

        if let Some(max_dim) = self.rules.max_dimensions {
            if vector.dimensions > max_dim {
                violations.push(ValidationViolation::new(
                    ValidationSeverity::Error,
                    "max_dimensions",
                    format!(
                        "Vector has {} dimensions, maximum is {}",
                        vector.dimensions, max_dim
                    ),
                ));
            }
        }

        // Check for invalid values
        if self.rules.check_for_invalid_values {
            let values = vector.as_f32();
            let has_nan = values.iter().any(|v| v.is_nan());
            let has_inf = values.iter().any(|v| v.is_infinite());

            if has_nan {
                violations.push(ValidationViolation::new(
                    ValidationSeverity::Critical,
                    "invalid_values",
                    "Vector contains NaN values",
                ));
            }

            if has_inf {
                violations.push(ValidationViolation::new(
                    ValidationSeverity::Critical,
                    "invalid_values",
                    "Vector contains infinite values",
                ));
            }
        }

        // Check for zero vector
        if self.rules.disallow_zero_vectors {
            let magnitude = vector.magnitude();
            if magnitude < 1e-10 {
                violations.push(ValidationViolation::new(
                    ValidationSeverity::Error,
                    "zero_vector",
                    "Vector is approximately zero",
                ));
            }
        }

        // Check normalization
        if self.rules.require_normalized {
            let magnitude = vector.magnitude();
            if (magnitude - 1.0).abs() > self.rules.normalization_tolerance {
                violations.push(ValidationViolation::new(
                    ValidationSeverity::Warning,
                    "normalization",
                    format!("Vector is not normalized (magnitude: {:.6})", magnitude),
                ));
            }
        }

        // Check precision
        if let Some(expected_precision) = self.rules.expected_precision {
            if vector.precision != expected_precision {
                violations.push(ValidationViolation::new(
                    ValidationSeverity::Warning,
                    "precision",
                    format!(
                        "Vector precision {:?} does not match expected {:?}",
                        vector.precision, expected_precision
                    ),
                ));
            }
        }

        // Check sparsity
        if let Some(min_non_zero) = self.rules.min_non_zero {
            let values = vector.as_f32();
            let non_zero_count = values.iter().filter(|&&v| v.abs() > 1e-10).count();

            if non_zero_count < min_non_zero {
                violations.push(ValidationViolation::new(
                    ValidationSeverity::Warning,
                    "sparsity",
                    format!(
                        "Vector has {} non-zero values, minimum is {}",
                        non_zero_count, min_non_zero
                    ),
                ));
            }
        }

        // Check maximum magnitude
        if let Some(max_mag) = self.rules.max_magnitude {
            let magnitude = vector.magnitude();
            if magnitude > max_mag {
                violations.push(ValidationViolation::new(
                    ValidationSeverity::Error,
                    "magnitude",
                    format!(
                        "Vector magnitude {:.6} exceeds maximum {:.6}",
                        magnitude, max_mag
                    ),
                ));
            }
        }

        ValidationResult::with_violations(violations)
    }

    /// Validate multiple vectors
    pub fn validate_batch(
        &self,
        vectors: &[(String, Vector)],
    ) -> HashMap<String, ValidationResult> {
        vectors
            .iter()
            .map(|(id, vector)| (id.clone(), self.validate(vector)))
            .collect()
    }

    /// Validate and return only invalid vectors
    pub fn find_invalid(&self, vectors: &[(String, Vector)]) -> Vec<(String, ValidationResult)> {
        vectors
            .iter()
            .map(|(id, vector)| (id.clone(), self.validate(vector)))
            .filter(|(_, result)| !result.passed)
            .collect()
    }
}

/// Dimension consistency validator
pub struct DimensionValidator {
    expected_dimension: Option<usize>,
}

impl DimensionValidator {
    pub fn new() -> Self {
        Self {
            expected_dimension: None,
        }
    }

    pub fn with_expected_dimension(dimension: usize) -> Self {
        Self {
            expected_dimension: Some(dimension),
        }
    }

    /// Validate dimension consistency across multiple vectors
    pub fn validate_consistency(&mut self, vectors: &[(String, Vector)]) -> ValidationResult {
        let mut violations = Vec::new();

        if vectors.is_empty() {
            return ValidationResult::success();
        }

        // Check if expected dimension is set, if not, use first vector's dimension
        let expected = if let Some(dim) = self.expected_dimension {
            dim
        } else {
            let first_dim = vectors[0].1.dimensions;
            self.expected_dimension = Some(first_dim);
            first_dim
        };

        // Check all vectors have consistent dimensions
        for (id, vector) in vectors {
            if vector.dimensions != expected {
                violations.push(
                    ValidationViolation::new(
                        ValidationSeverity::Error,
                        "dimension_mismatch",
                        format!(
                            "Vector '{}' has {} dimensions, expected {}",
                            id, vector.dimensions, expected
                        ),
                    )
                    .with_context(format!(
                        "expected={}, actual={}",
                        expected, vector.dimensions
                    )),
                );
            }
        }

        ValidationResult::with_violations(violations)
    }

    /// Get the established dimension
    pub fn established_dimension(&self) -> Option<usize> {
        self.expected_dimension
    }
}

/// Metadata validator
pub struct MetadataValidator {
    required_fields: Vec<String>,
    field_patterns: HashMap<String, regex::Regex>,
}

impl MetadataValidator {
    pub fn new() -> Self {
        Self {
            required_fields: Vec::new(),
            field_patterns: HashMap::new(),
        }
    }

    pub fn require_field(&mut self, field: impl Into<String>) -> &mut Self {
        self.required_fields.push(field.into());
        self
    }

    pub fn require_pattern(
        &mut self,
        field: impl Into<String>,
        pattern: &str,
    ) -> Result<&mut Self> {
        let regex = regex::Regex::new(pattern)?;
        self.field_patterns.insert(field.into(), regex);
        Ok(self)
    }

    /// Validate metadata
    pub fn validate(&self, metadata: &HashMap<String, String>) -> ValidationResult {
        let mut violations = Vec::new();

        // Check required fields
        for field in &self.required_fields {
            if !metadata.contains_key(field) {
                violations.push(ValidationViolation::new(
                    ValidationSeverity::Error,
                    "missing_field",
                    format!("Required field '{}' is missing", field),
                ));
            }
        }

        // Check patterns
        for (field, pattern) in &self.field_patterns {
            if let Some(value) = metadata.get(field) {
                if !pattern.is_match(value) {
                    violations.push(ValidationViolation::new(
                        ValidationSeverity::Error,
                        "pattern_mismatch",
                        format!(
                            "Field '{}' value '{}' does not match required pattern",
                            field, value
                        ),
                    ));
                }
            }
        }

        ValidationResult::with_violations(violations)
    }
}

impl Default for MetadataValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DimensionValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive validator for all operations
pub struct ComprehensiveValidator {
    vector_validator: VectorValidator,
    dimension_validator: DimensionValidator,
    metadata_validator: Option<MetadataValidator>,
}

impl ComprehensiveValidator {
    pub fn new(vector_rules: VectorValidationRules, expected_dimension: Option<usize>) -> Self {
        Self {
            vector_validator: VectorValidator::new(vector_rules),
            dimension_validator: if let Some(dim) = expected_dimension {
                DimensionValidator::with_expected_dimension(dim)
            } else {
                DimensionValidator::new()
            },
            metadata_validator: None,
        }
    }

    pub fn with_metadata_validator(mut self, validator: MetadataValidator) -> Self {
        self.metadata_validator = Some(validator);
        self
    }

    /// Validate vector with all rules
    pub fn validate_vector(
        &self,
        id: &str,
        vector: &Vector,
        metadata: Option<&HashMap<String, String>>,
    ) -> ValidationResult {
        let mut all_violations = Vec::new();

        // Vector validation
        let vector_result = self.vector_validator.validate(vector);
        all_violations.extend(vector_result.violations);

        // Dimension validation (single vector check)
        if let Some(expected_dim) = self.dimension_validator.established_dimension() {
            if vector.dimensions != expected_dim {
                all_violations.push(ValidationViolation::new(
                    ValidationSeverity::Error,
                    "dimension_mismatch",
                    format!(
                        "Vector '{}' has {} dimensions, expected {}",
                        id, vector.dimensions, expected_dim
                    ),
                ));
            }
        }

        // Metadata validation
        if let (Some(validator), Some(meta)) = (&self.metadata_validator, metadata) {
            let meta_result = validator.validate(meta);
            all_violations.extend(meta_result.violations);
        }

        ValidationResult::with_violations(all_violations)
    }

    /// Validate batch of vectors
    #[allow(clippy::type_complexity)]
    pub fn validate_batch(
        &mut self,
        vectors: &[(String, Vector, Option<HashMap<String, String>>)],
    ) -> HashMap<String, ValidationResult> {
        let mut results = HashMap::new();

        // First pass: dimension consistency
        let vectors_only: Vec<(String, Vector)> = vectors
            .iter()
            .map(|(id, vec, _)| (id.clone(), vec.clone()))
            .collect();

        let dim_result = self.dimension_validator.validate_consistency(&vectors_only);
        if dim_result.has_errors() {
            // If dimension consistency fails, report it for all vectors
            for (id, _, _) in vectors {
                results.insert(id.clone(), dim_result.clone());
            }
            return results;
        }

        // Second pass: individual validation
        for (id, vector, metadata) in vectors {
            let result = self.validate_vector(id, vector, metadata.as_ref());
            results.insert(id.clone(), result);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_vector() {
        let rules = VectorValidationRules::default();
        let validator = VectorValidator::new(rules);

        let vector = Vector::new(vec![1.0, 2.0, 3.0]);
        let result = validator.validate(&vector);

        assert!(result.passed);
        assert_eq!(result.violations.len(), 0);
    }

    #[test]
    fn test_invalid_dimensions() {
        let rules = VectorValidationRules {
            min_dimensions: Some(5),
            ..Default::default()
        };
        let validator = VectorValidator::new(rules);

        let vector = Vector::new(vec![1.0, 2.0]);
        let result = validator.validate(&vector);

        assert!(!result.passed);
        assert!(result.has_errors());
    }

    #[test]
    fn test_normalized_vector() {
        let rules = VectorValidationRules {
            require_normalized: true,
            ..Default::default()
        };
        let validator = VectorValidator::new(rules);

        // Not normalized
        let vector1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let result1 = validator.validate(&vector1);
        assert!(result1.has_warnings());

        // Normalized
        let vector2 = Vector::new(vec![1.0, 0.0, 0.0]);
        let result2 = validator.validate(&vector2);
        assert!(result2.passed);
    }

    #[test]
    fn test_invalid_values() {
        let rules = VectorValidationRules {
            check_for_invalid_values: true,
            ..Default::default()
        };
        let validator = VectorValidator::new(rules);

        let vector = Vector::new(vec![1.0, f32::NAN, 3.0]);
        let result = validator.validate(&vector);

        assert!(!result.passed);
        assert_eq!(result.error_count(), 1);
    }

    #[test]
    fn test_dimension_consistency() {
        let mut validator = DimensionValidator::new();

        let vectors = vec![
            ("vec1".to_string(), Vector::new(vec![1.0, 2.0, 3.0])),
            ("vec2".to_string(), Vector::new(vec![4.0, 5.0, 6.0])),
            ("vec3".to_string(), Vector::new(vec![7.0, 8.0])), // Wrong dimension
        ];

        let result = validator.validate_consistency(&vectors);

        assert!(!result.passed);
        assert_eq!(result.error_count(), 1);
    }

    #[test]
    fn test_metadata_validation() {
        let mut validator = MetadataValidator::new();
        validator.require_field("category");
        validator
            .require_pattern("status", r"^(active|inactive)$")
            .unwrap();

        let mut valid_metadata = HashMap::new();
        valid_metadata.insert("category".to_string(), "news".to_string());
        valid_metadata.insert("status".to_string(), "active".to_string());

        let result1 = validator.validate(&valid_metadata);
        assert!(result1.passed);

        let mut invalid_metadata = HashMap::new();
        invalid_metadata.insert("status".to_string(), "pending".to_string()); // Wrong pattern, missing category

        let result2 = validator.validate(&invalid_metadata);
        assert!(!result2.passed);
        assert_eq!(result2.error_count(), 2);
    }

    #[test]
    fn test_comprehensive_validator() {
        let rules = VectorValidationRules::default();
        let mut validator = ComprehensiveValidator::new(rules, None); // Don't set expected dimension upfront

        let vectors = vec![
            ("vec1".to_string(), Vector::new(vec![1.0, 2.0, 3.0]), None),
            ("vec2".to_string(), Vector::new(vec![4.0, 5.0]), None), // Wrong dimension
        ];

        let results = validator.validate_batch(&vectors);

        // First vector should fail because dimension consistency check fails
        // (vec1 has 3 dims, vec2 has 2 dims - they're inconsistent)
        assert!(!results["vec1"].passed); // Dimension inconsistency reported for all
        assert!(!results["vec2"].passed);
    }
}
