//! Certification runner: executes a suite of test cases and computes metrics.

use std::collections::HashMap;

use super::metrics::{ClassificationMetrics, ConstraintTypeMetrics};
use super::report::{CertificationReport, CertificationStatus};

/// A single certification test case.
///
/// Pairs a ground-truth SHACL engine result with the ML model's prediction
/// for a specific node/constraint combination.
#[derive(Debug, Clone)]
pub struct CertificationCase {
    /// Unique identifier for this case (e.g. dataset + node URI).
    pub id: String,
    /// Constraint type being certified (e.g. `"sh:minCount"`, `"sh:pattern"`).
    pub constraint_type: String,
    /// What the deterministic SHACL engine determined (`true` = violation).
    pub ground_truth_violation: bool,
    /// What the ML model predicted (`true` = violation predicted).
    pub model_predicted_violation: bool,
    /// Optional model confidence score in `[0, 1]`.
    pub confidence: Option<f64>,
}

/// A named collection of `CertificationCase` values.
#[derive(Debug, Default)]
pub struct CertificationSuite {
    /// Human-readable name for this suite.
    pub name: String,
    /// All test cases belonging to this suite.
    pub cases: Vec<CertificationCase>,
}

impl CertificationSuite {
    /// Create an empty suite with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            cases: Vec::new(),
        }
    }

    /// Append a single case to the suite.
    pub fn add_case(&mut self, case: CertificationCase) {
        self.cases.push(case);
    }

    /// Build a suite from a pre-existing vector of cases.
    pub fn from_cases(name: &str, cases: Vec<CertificationCase>) -> Self {
        Self {
            name: name.to_string(),
            cases,
        }
    }
}

/// Runs a `CertificationSuite` and evaluates metrics against configurable
/// acceptance thresholds.
pub struct CertificationRunner {
    /// Minimum overall F1 score required to pass.  Default: 0.80.
    pub min_f1_threshold: f64,
    /// Minimum overall precision required (false-positive ceiling).  Default: 0.75.
    pub min_precision_threshold: f64,
    /// Minimum overall recall required (missed-violation ceiling).  Default: 0.70.
    pub min_recall_threshold: f64,
}

impl Default for CertificationRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl CertificationRunner {
    /// Create a runner with sensible production defaults.
    ///
    /// Defaults:
    /// - min_f1 = 0.80
    /// - min_precision = 0.75
    /// - min_recall = 0.70
    pub fn new() -> Self {
        Self {
            min_f1_threshold: 0.80,
            min_precision_threshold: 0.75,
            min_recall_threshold: 0.70,
        }
    }

    /// Create a runner with explicit threshold values.
    pub fn with_thresholds(min_f1: f64, min_precision: f64, min_recall: f64) -> Self {
        Self {
            min_f1_threshold: min_f1,
            min_precision_threshold: min_precision,
            min_recall_threshold: min_recall,
        }
    }

    /// Execute the suite and produce a `CertificationReport`.
    ///
    /// Returns `CertificationStatus::Insufficient` when fewer than 10 cases
    /// are present — not enough data to certify a model.
    pub fn run(&self, suite: &CertificationSuite) -> CertificationReport {
        let total_cases = suite.cases.len() as u64;

        // Guard: require a minimum number of cases.
        if suite.cases.len() < 10 {
            return CertificationReport::new(
                &suite.name,
                CertificationStatus::Insufficient {
                    reason: format!(
                        "Only {} case(s) provided; at least 10 are required to certify a model.",
                        suite.cases.len()
                    ),
                },
                ClassificationMetrics::default(),
                Vec::new(),
                total_cases,
            );
        }

        // Accumulate overall metrics and per-constraint-type metrics.
        let mut overall = ClassificationMetrics::default();
        let mut per_type: HashMap<String, (ClassificationMetrics, u64)> = HashMap::new();

        for case in &suite.cases {
            let cell = update_metrics_for_case(case);
            overall.add(&cell);

            let entry = per_type
                .entry(case.constraint_type.clone())
                .or_insert_with(|| (ClassificationMetrics::default(), 0));
            entry.0.add(&cell);
            entry.1 += 1;
        }

        // Build per-constraint list (sorted for deterministic output).
        let mut per_constraint_metrics: Vec<ConstraintTypeMetrics> = per_type
            .into_iter()
            .map(
                |(constraint_type, (metrics, sample_count))| ConstraintTypeMetrics {
                    constraint_type,
                    metrics,
                    sample_count,
                },
            )
            .collect();
        per_constraint_metrics.sort_by(|a, b| a.constraint_type.cmp(&b.constraint_type));

        // Evaluate thresholds.
        let status = self.evaluate_status(&overall);

        CertificationReport::new(
            &suite.name,
            status,
            overall,
            per_constraint_metrics,
            total_cases,
        )
    }

    fn evaluate_status(&self, metrics: &ClassificationMetrics) -> CertificationStatus {
        let f1 = metrics.f1_score();
        let precision = metrics.precision();
        let recall = metrics.recall();

        let mut reasons = Vec::new();

        if f1 < self.min_f1_threshold {
            reasons.push(format!(
                "F1 score {:.4} is below minimum threshold {:.4}",
                f1, self.min_f1_threshold
            ));
        }
        if precision < self.min_precision_threshold {
            reasons.push(format!(
                "Precision {:.4} is below minimum threshold {:.4}",
                precision, self.min_precision_threshold
            ));
        }
        if recall < self.min_recall_threshold {
            reasons.push(format!(
                "Recall {:.4} is below minimum threshold {:.4}",
                recall, self.min_recall_threshold
            ));
        }

        if reasons.is_empty() {
            CertificationStatus::Passed
        } else {
            CertificationStatus::Failed { reasons }
        }
    }
}

/// Compute the classification cell (TP/FP/TN/FN) for a single case.
fn update_metrics_for_case(case: &CertificationCase) -> ClassificationMetrics {
    match (case.ground_truth_violation, case.model_predicted_violation) {
        (true, true) => ClassificationMetrics {
            true_positives: 1,
            ..Default::default()
        },
        (false, true) => ClassificationMetrics {
            false_positives: 1,
            ..Default::default()
        },
        (false, false) => ClassificationMetrics {
            true_negatives: 1,
            ..Default::default()
        },
        (true, false) => ClassificationMetrics {
            false_negatives: 1,
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_case(id: &str, ct: &str, truth: bool, predicted: bool) -> CertificationCase {
        CertificationCase {
            id: id.to_string(),
            constraint_type: ct.to_string(),
            ground_truth_violation: truth,
            model_predicted_violation: predicted,
            confidence: None,
        }
    }

    #[test]
    fn test_insufficient_cases_fewer_than_10() {
        let suite = CertificationSuite::from_cases(
            "tiny",
            vec![make_case("c1", "sh:minCount", true, true)],
        );
        let runner = CertificationRunner::new();
        let report = runner.run(&suite);
        assert!(matches!(
            report.status,
            CertificationStatus::Insufficient { .. }
        ));
    }

    #[test]
    fn test_defaults_are_sane() {
        let runner = CertificationRunner::new();
        assert!((runner.min_f1_threshold - 0.80).abs() < 1e-10);
        assert!((runner.min_precision_threshold - 0.75).abs() < 1e-10);
        assert!((runner.min_recall_threshold - 0.70).abs() < 1e-10);
    }
}
