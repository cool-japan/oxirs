//! AI-Assisted Constraint Validation
//!
//! This module provides heuristic and statistical validation of SHACL-like
//! constraints derived from the shape mining and constraint learning modules.
//!
//! Rather than full W3C SHACL validation (which is handled by `oxirs-shacl`),
//! this validator applies lightweight, ML-informed scoring to check how well
//! a given set of `PropertyConstraint`s applies to a triple set.  It produces
//! a structured `AiValidationReport` with per-constraint scores and
//! human-readable findings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::shape_miner::PropertyConstraint;

/// The kind of finding produced by AI-assisted validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationFindingKind {
    /// Constraint is well-supported and consistent with the data.
    Conformant,
    /// Cardinality bounds are violated by some subjects.
    CardinalityViolation,
    /// Datatype mismatch detected in object values.
    DatatypeMismatch,
    /// The predicate is absent from most subjects.
    MissingProperty,
    /// Too many values found for a property that should be functional.
    FunctionalityViolation,
    /// Constraint is redundant or overlaps with another.
    Redundant,
    /// Low confidence; insufficient data to validate reliably.
    InsufficientData,
}

impl ValidationFindingKind {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            ValidationFindingKind::Conformant => "conformant",
            ValidationFindingKind::CardinalityViolation => "cardinality-violation",
            ValidationFindingKind::DatatypeMismatch => "datatype-mismatch",
            ValidationFindingKind::MissingProperty => "missing-property",
            ValidationFindingKind::FunctionalityViolation => "functionality-violation",
            ValidationFindingKind::Redundant => "redundant",
            ValidationFindingKind::InsufficientData => "insufficient-data",
        }
    }

    /// Returns `true` if this finding indicates a genuine problem.
    pub fn is_violation(&self) -> bool {
        !matches!(
            self,
            ValidationFindingKind::Conformant
                | ValidationFindingKind::Redundant
                | ValidationFindingKind::InsufficientData
        )
    }
}

/// A single finding produced during constraint validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFinding {
    /// Predicate URI the finding relates to.
    pub predicate: String,
    /// Kind of finding.
    pub kind: ValidationFindingKind,
    /// Human-readable description.
    pub message: String,
    /// Severity score (0.0 = informational, 1.0 = critical).
    pub severity: f64,
    /// Number of subjects exhibiting this issue.
    pub affected_count: usize,
}

/// Validation score for a single `PropertyConstraint`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintValidationScore {
    /// Predicate URI being validated.
    pub predicate: String,
    /// Fraction of subjects conforming to this constraint (0.0 – 1.0).
    pub conformance_rate: f64,
    /// Weighted penalty for violations (0.0 = no violations, 1.0 = all violated).
    pub violation_penalty: f64,
    /// Overall AI score (higher = better) computed as `conformance_rate * (1 - violation_penalty)`.
    pub ai_score: f64,
    /// Detailed findings for this constraint.
    pub findings: Vec<ValidationFinding>,
}

/// Configuration for `ShapeValidatorAi`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeValidatorAiConfig {
    /// Minimum number of subjects required to produce a reliable validation score.
    pub min_subjects_for_reliability: usize,
    /// Penalty multiplier applied for cardinality violations.
    pub cardinality_penalty_weight: f64,
    /// Penalty multiplier applied for datatype mismatches.
    pub datatype_penalty_weight: f64,
    /// Penalty multiplier applied when a property is missing from most subjects.
    pub missing_property_penalty_weight: f64,
    /// Minimum conformance rate below which a constraint is flagged as MissingProperty.
    pub missing_property_threshold: f64,
}

impl Default for ShapeValidatorAiConfig {
    fn default() -> Self {
        Self {
            min_subjects_for_reliability: 3,
            cardinality_penalty_weight: 0.8,
            datatype_penalty_weight: 0.6,
            missing_property_penalty_weight: 1.0,
            missing_property_threshold: 0.3,
        }
    }
}

/// Full validation report produced by `ShapeValidatorAi`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiValidationReport {
    /// Per-constraint scores.
    pub scores: Vec<ConstraintValidationScore>,
    /// Aggregate conformance rate (average across all constraints).
    pub aggregate_conformance: f64,
    /// Number of constraints with at least one violation finding.
    pub constraints_with_violations: usize,
    /// Total subjects in the validated triple set.
    pub total_subjects: usize,
    /// Suggested actions summarised from findings.
    pub suggestions: Vec<String>,
}

/// AI-assisted constraint validator.
///
/// # Example
/// ```rust
/// use oxirs_shacl_ai::shape_learning::{ShapeValidatorAi, ShapeValidatorAiConfig, PropertyConstraint, NodeKind};
///
/// let constraints = vec![
///     PropertyConstraint {
///         predicate: "http://ex.org/name".into(),
///         min_count: Some(1),
///         max_count: Some(1),
///         datatype: None,
///         node_kind: Some(NodeKind::Literal),
///         support: 1.0,
///         confidence: 1.0,
///     },
/// ];
/// let triples = vec![
///     ("http://ex.org/alice".into(), "http://ex.org/name".into(), "\"Alice\"".into()),
///     ("http://ex.org/bob".into(),   "http://ex.org/name".into(), "\"Bob\"".into()),
/// ];
/// let validator = ShapeValidatorAi::new(ShapeValidatorAiConfig::default());
/// let report = validator.validate(&constraints, &triples);
/// assert!(report.aggregate_conformance > 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct ShapeValidatorAi {
    config: ShapeValidatorAiConfig,
}

impl ShapeValidatorAi {
    /// Create a validator with the given configuration.
    pub fn new(config: ShapeValidatorAiConfig) -> Self {
        Self { config }
    }

    /// Create a validator with the default configuration.
    pub fn default_config() -> Self {
        Self::new(ShapeValidatorAiConfig::default())
    }

    /// Access the current configuration.
    pub fn config(&self) -> &ShapeValidatorAiConfig {
        &self.config
    }

    /// Validate a slice of `PropertyConstraint`s against a triple set.
    ///
    /// Returns an `AiValidationReport` with per-constraint scores and aggregate metrics.
    pub fn validate(
        &self,
        constraints: &[PropertyConstraint],
        triples: &[(String, String, String)],
    ) -> AiValidationReport {
        // Build index: subject → predicate → vec<object>
        let mut index: HashMap<&str, HashMap<&str, Vec<&str>>> = HashMap::new();
        for (subj, pred, obj) in triples {
            index
                .entry(subj.as_str())
                .or_default()
                .entry(pred.as_str())
                .or_default()
                .push(obj.as_str());
        }

        let total_subjects = index.len();

        if total_subjects == 0 || constraints.is_empty() {
            return AiValidationReport {
                scores: vec![],
                aggregate_conformance: 1.0,
                constraints_with_violations: 0,
                total_subjects,
                suggestions: vec![],
            };
        }

        let mut scores: Vec<ConstraintValidationScore> = Vec::new();
        let mut all_suggestions: Vec<String> = Vec::new();

        for constraint in constraints {
            let score = self.score_constraint(constraint, &index, total_subjects);

            // Collect suggestions from findings
            for finding in &score.findings {
                if finding.kind.is_violation() {
                    all_suggestions.push(format!(
                        "[{}] {} — {} subjects affected",
                        finding.kind.label(),
                        finding.message,
                        finding.affected_count
                    ));
                }
            }

            scores.push(score);
        }

        let constraints_with_violations = scores
            .iter()
            .filter(|s| s.findings.iter().any(|f| f.kind.is_violation()))
            .count();

        let aggregate_conformance = if scores.is_empty() {
            1.0
        } else {
            scores.iter().map(|s| s.conformance_rate).sum::<f64>() / scores.len() as f64
        };

        // Deduplicate suggestions
        all_suggestions.sort();
        all_suggestions.dedup();

        AiValidationReport {
            scores,
            aggregate_conformance,
            constraints_with_violations,
            total_subjects,
            suggestions: all_suggestions,
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn score_constraint(
        &self,
        constraint: &PropertyConstraint,
        index: &HashMap<&str, HashMap<&str, Vec<&str>>>,
        total_subjects: usize,
    ) -> ConstraintValidationScore {
        let pred = constraint.predicate.as_str();
        let mut findings: Vec<ValidationFinding> = Vec::new();
        let mut total_penalty: f64 = 0.0;

        // Subjects that have the predicate
        let subjects_with_pred: Vec<(&str, Vec<&str>)> = index
            .iter()
            .filter_map(|(&subj, pm)| pm.get(pred).map(|objs| (subj, objs.clone())))
            .collect();

        let subjects_missing_pred = total_subjects - subjects_with_pred.len();

        // Check if data is sufficient
        if total_subjects < self.config.min_subjects_for_reliability {
            findings.push(ValidationFinding {
                predicate: pred.to_string(),
                kind: ValidationFindingKind::InsufficientData,
                message: format!(
                    "Only {} subject(s) available; at least {} required for reliable validation",
                    total_subjects, self.config.min_subjects_for_reliability
                ),
                severity: 0.2,
                affected_count: total_subjects,
            });
        }

        // Missing-property check
        let presence_rate = subjects_with_pred.len() as f64 / total_subjects as f64;
        if let Some(min) = constraint.min_count {
            if min >= 1 && presence_rate < self.config.missing_property_threshold {
                let penalty = self.config.missing_property_penalty_weight * (1.0 - presence_rate);
                total_penalty += penalty;
                findings.push(ValidationFinding {
                    predicate: pred.to_string(),
                    kind: ValidationFindingKind::MissingProperty,
                    message: format!(
                        "Property <{}> required (min_count={}) but absent from {:.0}% of subjects",
                        pred,
                        min,
                        (1.0 - presence_rate) * 100.0
                    ),
                    severity: penalty.min(1.0),
                    affected_count: subjects_missing_pred,
                });
            }
        }

        // Cardinality check
        let mut cardinality_violations = 0usize;
        for (_, objs) in &subjects_with_pred {
            let count = objs.len() as u32;
            let min_ok = constraint.min_count.map_or(true, |m| count >= m);
            let max_ok = constraint.max_count.map_or(true, |m| count <= m);
            if !min_ok || !max_ok {
                cardinality_violations += 1;
            }
            // Functionality check: max_count == 1 but > 1 values
            if constraint.max_count == Some(1) && count > 1 {
                findings.push(ValidationFinding {
                    predicate: pred.to_string(),
                    kind: ValidationFindingKind::FunctionalityViolation,
                    message: format!(
                        "Property <{}> should be functional (max 1) but has {} values",
                        pred, count
                    ),
                    severity: 0.7,
                    affected_count: 1,
                });
            }
        }

        if cardinality_violations > 0 {
            let viol_rate = cardinality_violations as f64 / total_subjects as f64;
            let penalty = self.config.cardinality_penalty_weight * viol_rate;
            total_penalty += penalty;
            findings.push(ValidationFinding {
                predicate: pred.to_string(),
                kind: ValidationFindingKind::CardinalityViolation,
                message: format!(
                    "Property <{}> cardinality constraint violated by {} subject(s)",
                    pred, cardinality_violations
                ),
                severity: penalty.min(1.0),
                affected_count: cardinality_violations,
            });
        }

        // Datatype check
        if let Some(expected_dt) = &constraint.datatype {
            let dt_mismatches: usize = subjects_with_pred
                .iter()
                .flat_map(|(_, objs)| objs.iter())
                .filter(|&&obj| {
                    // Typed literal must contain `^^<expected_dt>` or `^^expected_dt`
                    if obj.contains("^^") {
                        let actual = obj
                            .find("^^")
                            .map(|i| &obj[i + 2..])
                            .unwrap_or("")
                            .trim_matches(|c| c == '<' || c == '>');
                        actual != expected_dt.trim_matches(|c: char| c == '<' || c == '>')
                    } else {
                        // Not a typed literal at all — mismatch if datatype is expected
                        !expected_dt.is_empty()
                    }
                })
                .count();

            if dt_mismatches > 0 {
                let penalty = self.config.datatype_penalty_weight
                    * (dt_mismatches as f64 / total_subjects as f64);
                total_penalty += penalty;
                findings.push(ValidationFinding {
                    predicate: pred.to_string(),
                    kind: ValidationFindingKind::DatatypeMismatch,
                    message: format!(
                        "Property <{}> expected datatype <{}> but {} value(s) differ",
                        pred, expected_dt, dt_mismatches
                    ),
                    severity: penalty.min(1.0),
                    affected_count: dt_mismatches,
                });
            }
        }

        // If no violation findings, mark as conformant
        let has_violations = findings.iter().any(|f| f.kind.is_violation());
        if !has_violations {
            findings.push(ValidationFinding {
                predicate: pred.to_string(),
                kind: ValidationFindingKind::Conformant,
                message: format!("Property <{}> satisfies all constraints", pred),
                severity: 0.0,
                affected_count: 0,
            });
        }

        let violation_penalty = total_penalty.min(1.0);
        let conformance_rate = presence_rate * (1.0 - violation_penalty.min(1.0));
        let ai_score = conformance_rate * (1.0 - violation_penalty);

        ConstraintValidationScore {
            predicate: pred.to_string(),
            conformance_rate,
            violation_penalty,
            ai_score,
            findings,
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape_learning::shape_miner::NodeKind;

    fn name_constraint() -> PropertyConstraint {
        PropertyConstraint {
            predicate: "http://ex.org/name".into(),
            min_count: Some(1),
            max_count: Some(1),
            datatype: None,
            node_kind: Some(NodeKind::Literal),
            support: 1.0,
            confidence: 1.0,
        }
    }

    fn two_person_triples() -> Vec<(String, String, String)> {
        vec![
            (
                "http://ex.org/alice".into(),
                "http://ex.org/name".into(),
                "\"Alice\"".into(),
            ),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/name".into(),
                "\"Bob\"".into(),
            ),
            (
                "http://ex.org/carol".into(),
                "http://ex.org/name".into(),
                "\"Carol\"".into(),
            ),
            (
                "http://ex.org/dave".into(),
                "http://ex.org/name".into(),
                "\"Dave\"".into(),
            ),
        ]
    }

    #[test]
    fn test_validate_fully_conformant() {
        let validator = ShapeValidatorAi::default_config();
        let report = validator.validate(&[name_constraint()], &two_person_triples());
        assert!(
            report.aggregate_conformance > 0.5,
            "expected conformant, got {}",
            report.aggregate_conformance
        );
        assert_eq!(report.constraints_with_violations, 0);
    }

    #[test]
    fn test_validate_missing_property() {
        // Only 1 of 4 subjects has the name property
        let triples = vec![
            (
                "http://ex.org/alice".into(),
                "http://ex.org/name".into(),
                "\"Alice\"".into(),
            ),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/other".into(),
                "\"X\"".into(),
            ),
            (
                "http://ex.org/carol".into(),
                "http://ex.org/other".into(),
                "\"Y\"".into(),
            ),
            (
                "http://ex.org/dave".into(),
                "http://ex.org/other".into(),
                "\"Z\"".into(),
            ),
        ];
        let validator = ShapeValidatorAi::default_config();
        let report = validator.validate(&[name_constraint()], &triples);
        assert!(
            report.constraints_with_violations >= 1,
            "expected missing property violation"
        );
    }

    #[test]
    fn test_validate_cardinality_violation() {
        // Alice has two names → violates max_count=1
        let triples = vec![
            (
                "http://ex.org/alice".into(),
                "http://ex.org/name".into(),
                "\"Alice\"".into(),
            ),
            (
                "http://ex.org/alice".into(),
                "http://ex.org/name".into(),
                "\"Alicia\"".into(),
            ),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/name".into(),
                "\"Bob\"".into(),
            ),
            (
                "http://ex.org/carol".into(),
                "http://ex.org/name".into(),
                "\"Carol\"".into(),
            ),
        ];
        let validator = ShapeValidatorAi::default_config();
        let report = validator.validate(&[name_constraint()], &triples);
        let score = &report.scores[0];
        assert!(
            score
                .findings
                .iter()
                .any(|f| f.kind == ValidationFindingKind::CardinalityViolation
                    || f.kind == ValidationFindingKind::FunctionalityViolation),
            "expected cardinality or functionality violation"
        );
    }

    #[test]
    fn test_validate_datatype_mismatch() {
        let dt_constraint = PropertyConstraint {
            predicate: "http://ex.org/age".into(),
            min_count: Some(1),
            max_count: None,
            datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
            node_kind: Some(NodeKind::Literal),
            support: 1.0,
            confidence: 0.9,
        };
        // Bob's age is a string, not integer
        let triples = vec![
            (
                "http://ex.org/alice".into(),
                "http://ex.org/age".into(),
                "\"30\"^^<http://www.w3.org/2001/XMLSchema#integer>".into(),
            ),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/age".into(),
                "\"thirty\"^^<http://www.w3.org/2001/XMLSchema#string>".into(),
            ),
            (
                "http://ex.org/carol".into(),
                "http://ex.org/age".into(),
                "\"25\"^^<http://www.w3.org/2001/XMLSchema#integer>".into(),
            ),
            (
                "http://ex.org/dave".into(),
                "http://ex.org/age".into(),
                "\"28\"^^<http://www.w3.org/2001/XMLSchema#integer>".into(),
            ),
        ];
        let validator = ShapeValidatorAi::default_config();
        let report = validator.validate(&[dt_constraint], &triples);
        assert!(
            report.scores[0]
                .findings
                .iter()
                .any(|f| f.kind == ValidationFindingKind::DatatypeMismatch),
            "expected datatype mismatch finding"
        );
    }

    #[test]
    fn test_validate_empty_triples_returns_perfect_score() {
        let validator = ShapeValidatorAi::default_config();
        let report = validator.validate(&[name_constraint()], &[]);
        assert_eq!(report.aggregate_conformance, 1.0);
        assert_eq!(report.total_subjects, 0);
    }

    #[test]
    fn test_validate_empty_constraints_returns_report() {
        let validator = ShapeValidatorAi::default_config();
        let report = validator.validate(&[], &two_person_triples());
        assert!(report.scores.is_empty());
        assert_eq!(report.aggregate_conformance, 1.0);
    }

    #[test]
    fn test_insufficient_data_finding() {
        // Only 1 subject – below default threshold of 3
        let triples = vec![(
            "http://ex.org/solo".into(),
            "http://ex.org/name".into(),
            "\"Solo\"".into(),
        )];
        let validator = ShapeValidatorAi::default_config();
        let report = validator.validate(&[name_constraint()], &triples);
        assert!(
            report.scores[0]
                .findings
                .iter()
                .any(|f| f.kind == ValidationFindingKind::InsufficientData),
            "expected InsufficientData finding for 1 subject"
        );
    }

    #[test]
    fn test_ai_score_between_zero_and_one() {
        let validator = ShapeValidatorAi::default_config();
        let report = validator.validate(&[name_constraint()], &two_person_triples());
        for score in &report.scores {
            assert!(
                score.ai_score >= 0.0 && score.ai_score <= 1.0,
                "ai_score {} out of [0,1]",
                score.ai_score
            );
        }
    }

    #[test]
    fn test_multiple_constraints() {
        let constraints = vec![
            name_constraint(),
            PropertyConstraint {
                predicate: "http://ex.org/age".into(),
                min_count: Some(1),
                max_count: None,
                datatype: None,
                node_kind: None,
                support: 0.8,
                confidence: 0.8,
            },
        ];
        let mut triples = two_person_triples();
        triples.extend(vec![
            (
                "http://ex.org/alice".into(),
                "http://ex.org/age".into(),
                "\"30\"".into(),
            ),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/age".into(),
                "\"25\"".into(),
            ),
            (
                "http://ex.org/carol".into(),
                "http://ex.org/age".into(),
                "\"20\"".into(),
            ),
            (
                "http://ex.org/dave".into(),
                "http://ex.org/age".into(),
                "\"35\"".into(),
            ),
        ]);
        let validator = ShapeValidatorAi::default_config();
        let report = validator.validate(&constraints, &triples);
        assert_eq!(report.scores.len(), 2);
        assert!(report.aggregate_conformance > 0.0);
    }
}
