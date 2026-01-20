//! Constraint validation and conflict detection

use serde::{Deserialize, Serialize};

use super::types::{
    ConflictResolution, ConflictType, Constraint, ConstraintConflict, GeneratedConstraint,
};

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether constraint is valid
    pub is_valid: bool,
    /// Validation issues
    pub issues: Vec<ValidationIssue>,
    /// Conflicts with other constraints
    pub conflicts: Vec<ConstraintConflict>,
    /// Warnings
    pub warnings: Vec<String>,
}

/// Validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Issue severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

/// Constraint validator
pub struct ConstraintValidator {
    strict_mode: bool,
}

impl ConstraintValidator {
    pub fn new(strict_mode: bool) -> Self {
        Self { strict_mode }
    }

    pub fn non_strict() -> Self {
        Self { strict_mode: false }
    }

    /// Validate a single constraint
    pub fn validate_constraint(&self, constraint: &GeneratedConstraint) -> ValidationResult {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        // Check confidence threshold
        if constraint.metadata.confidence < 0.5 {
            issues.push(ValidationIssue {
                severity: if self.strict_mode {
                    IssueSeverity::Error
                } else {
                    IssueSeverity::Warning
                },
                description: format!(
                    "Low confidence: {:.1}%",
                    constraint.metadata.confidence * 100.0
                ),
                suggested_fix: Some(
                    "Collect more samples or adjust detection parameters".to_string(),
                ),
            });
        }

        // Check support threshold
        if constraint.metadata.support < 0.7 {
            warnings.push(format!(
                "Low support: {:.1}% of data matches constraint",
                constraint.metadata.support * 100.0
            ));
        }

        // Check counter-examples
        if constraint.metadata.counter_examples > 0 {
            let violation_rate = constraint.metadata.counter_examples as f64
                / constraint.metadata.sample_count.max(1) as f64;

            if violation_rate > 0.1 {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    description: format!(
                        "{} violations found ({:.1}% of data)",
                        constraint.metadata.counter_examples,
                        violation_rate * 100.0
                    ),
                    suggested_fix: Some("Review violations before applying constraint".to_string()),
                });
            }
        }

        // Validate constraint-specific rules
        match &constraint.constraint {
            Constraint::Cardinality { min, max } => {
                if let (Some(min_val), Some(max_val)) = (min, max) {
                    if min_val > max_val {
                        issues.push(ValidationIssue {
                            severity: IssueSeverity::Error,
                            description: "Invalid cardinality: min > max".to_string(),
                            suggested_fix: Some("Ensure min <= max".to_string()),
                        });
                    }
                }
            }
            Constraint::ValueRange {
                min_inclusive,
                max_inclusive,
                ..
            } => {
                if let (Some(min_val), Some(max_val)) = (min_inclusive, max_inclusive) {
                    if min_val > max_val {
                        issues.push(ValidationIssue {
                            severity: IssueSeverity::Error,
                            description: "Invalid range: min > max".to_string(),
                            suggested_fix: Some("Ensure min <= max".to_string()),
                        });
                    }
                }
            }
            Constraint::Pattern { pattern, .. } => {
                if let Err(e) = regex::Regex::new(pattern) {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        description: format!("Invalid regex pattern: {}", e),
                        suggested_fix: Some("Fix regex syntax".to_string()),
                    });
                }
            }
            _ => {}
        }

        let is_valid = issues.iter().all(|i| i.severity != IssueSeverity::Error);

        ValidationResult {
            is_valid,
            issues,
            conflicts: Vec::new(), // Populated by conflict detection
            warnings,
        }
    }

    /// Detect conflicts between constraints
    pub fn detect_conflicts(&self, constraints: &[GeneratedConstraint]) -> Vec<ConstraintConflict> {
        let mut conflicts = Vec::new();

        for i in 0..constraints.len() {
            for j in (i + 1)..constraints.len() {
                if let Some(conflict) = self.check_conflict(&constraints[i], &constraints[j]) {
                    conflicts.push(conflict);
                }
            }
        }

        conflicts
    }

    fn check_conflict(
        &self,
        c1: &GeneratedConstraint,
        c2: &GeneratedConstraint,
    ) -> Option<ConstraintConflict> {
        // Only check constraints on the same target
        if c1.target != c2.target {
            return None;
        }

        // Check for cardinality conflicts
        if let (
            Constraint::Cardinality {
                min: min1,
                max: max1,
            },
            Constraint::Cardinality {
                min: min2,
                max: max2,
            },
        ) = (&c1.constraint, &c2.constraint)
        {
            // Check for contradiction
            if let (Some(max1_val), Some(min2_val)) = (max1, min2) {
                if max1_val < min2_val {
                    return Some(ConstraintConflict {
                        constraint1_id: c1.id.clone(),
                        constraint2_id: c2.id.clone(),
                        conflict_type: ConflictType::Contradiction,
                        description: format!(
                            "Cardinality contradiction: max {} < min {}",
                            max1_val, min2_val
                        ),
                        severity: 1.0,
                        resolution: ConflictResolution::KeepHigherConfidence,
                    });
                }
            }

            // Check for subsumption
            if min1 <= min2 && max1 >= max2 {
                return Some(ConstraintConflict {
                    constraint1_id: c1.id.clone(),
                    constraint2_id: c2.id.clone(),
                    conflict_type: ConflictType::Subsumption,
                    description: "One cardinality constraint subsumes the other".to_string(),
                    severity: 0.5,
                    resolution: ConflictResolution::KeepMoreSpecific,
                });
            }
        }

        // Check for datatype conflicts
        if let (Constraint::Datatype { datatype: dt1 }, Constraint::Datatype { datatype: dt2 }) =
            (&c1.constraint, &c2.constraint)
        {
            if dt1 != dt2 {
                return Some(ConstraintConflict {
                    constraint1_id: c1.id.clone(),
                    constraint2_id: c2.id.clone(),
                    conflict_type: ConflictType::Contradiction,
                    description: format!("Conflicting datatypes: {} vs {}", dt1, dt2),
                    severity: 1.0,
                    resolution: ConflictResolution::KeepHigherConfidence,
                });
            }
        }

        None
    }

    /// Resolve conflicts automatically
    pub fn resolve_conflicts(
        &self,
        constraints: Vec<GeneratedConstraint>,
        conflicts: Vec<ConstraintConflict>,
    ) -> Vec<GeneratedConstraint> {
        let mut resolved = constraints;
        let mut to_remove = std::collections::HashSet::new();

        for conflict in conflicts {
            match conflict.resolution {
                ConflictResolution::KeepHigherConfidence => {
                    let c1 = resolved
                        .iter()
                        .find(|c| c.id == conflict.constraint1_id)
                        .unwrap();
                    let c2 = resolved
                        .iter()
                        .find(|c| c.id == conflict.constraint2_id)
                        .unwrap();

                    if c1.metadata.confidence < c2.metadata.confidence {
                        to_remove.insert(conflict.constraint1_id.clone());
                    } else {
                        to_remove.insert(conflict.constraint2_id.clone());
                    }
                }
                ConflictResolution::KeepMoreSpecific => {
                    // Keep the constraint with tighter bounds
                    to_remove.insert(conflict.constraint1_id.clone());
                }
                _ => {}
            }
        }

        resolved.retain(|c| !to_remove.contains(&c.id));
        resolved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint_generation::types::*;
    use oxirs_core::model::NamedNode;

    fn create_test_constraint(confidence: f64, constraint: Constraint) -> GeneratedConstraint {
        GeneratedConstraint {
            id: format!("test_{}", uuid::Uuid::new_v4()),
            constraint_type: ConstraintType::Cardinality,
            target: NamedNode::new_unchecked("http://example.org/prop"),
            constraint,
            metadata: ConstraintMetadata {
                confidence,
                support: 0.8,
                sample_count: 100,
                generation_method: "test".to_string(),
                generated_at: chrono::Utc::now(),
                evidence: vec![],
                counter_examples: 0,
            },
            quality: ConstraintQuality::calculate(confidence, 0.8),
        }
    }

    #[test]
    fn test_validator_creation() {
        let validator = ConstraintValidator::non_strict();
        assert!(!validator.strict_mode);
    }

    #[test]
    fn test_validate_valid_constraint() {
        let validator = ConstraintValidator::non_strict();
        let constraint = create_test_constraint(
            0.9,
            Constraint::Cardinality {
                min: Some(1),
                max: Some(5),
            },
        );

        let result = validator.validate_constraint(&constraint);
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_invalid_cardinality() {
        let validator = ConstraintValidator::non_strict();
        let constraint = create_test_constraint(
            0.9,
            Constraint::Cardinality {
                min: Some(5),
                max: Some(1),
            },
        );

        let result = validator.validate_constraint(&constraint);
        assert!(!result.is_valid);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_validate_invalid_pattern() {
        let validator = ConstraintValidator::non_strict();
        let constraint = create_test_constraint(
            0.9,
            Constraint::Pattern {
                pattern: "[invalid(".to_string(),
                flags: None,
            },
        );

        let result = validator.validate_constraint(&constraint);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_detect_cardinality_conflict() {
        let validator = ConstraintValidator::non_strict();
        let constraints = vec![
            create_test_constraint(
                0.9,
                Constraint::Cardinality {
                    min: Some(1),
                    max: Some(3),
                },
            ),
            create_test_constraint(
                0.8,
                Constraint::Cardinality {
                    min: Some(5),
                    max: Some(10),
                },
            ),
        ];

        let conflicts = validator.detect_conflicts(&constraints);
        assert!(!conflicts.is_empty());
        assert_eq!(conflicts[0].conflict_type, ConflictType::Contradiction);
    }

    #[test]
    fn test_resolve_conflicts() {
        let validator = ConstraintValidator::non_strict();
        let c1 = create_test_constraint(
            0.9,
            Constraint::Cardinality {
                min: Some(1),
                max: Some(3),
            },
        );
        let c2 = create_test_constraint(
            0.8,
            Constraint::Cardinality {
                min: Some(5),
                max: Some(10),
            },
        );

        let conflict = ConstraintConflict {
            constraint1_id: c1.id.clone(),
            constraint2_id: c2.id.clone(),
            conflict_type: ConflictType::Contradiction,
            description: "Test conflict".to_string(),
            severity: 1.0,
            resolution: ConflictResolution::KeepHigherConfidence,
        };

        let resolved = validator.resolve_conflicts(vec![c1, c2], vec![conflict]);
        // Should keep only the one with higher confidence
        assert_eq!(resolved.len(), 1);
    }
}
