//! Certification report: the output of a completed certification run.

use chrono::Utc;

use super::metrics::{ClassificationMetrics, ConstraintTypeMetrics};

/// Outcome of a certification run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertificationStatus {
    /// All metric thresholds were satisfied.
    Passed,
    /// One or more thresholds were violated.  `reasons` lists each failure.
    Failed {
        /// Human-readable descriptions of each failed threshold.
        reasons: Vec<String>,
    },
    /// Not enough test cases to make a certification decision.
    Insufficient {
        /// Explains why certification could not proceed.
        reason: String,
    },
}

/// The complete result of running a `CertificationSuite`.
#[derive(Debug, Clone)]
pub struct CertificationReport {
    /// Name of the suite that was executed.
    pub suite_name: String,
    /// Whether the model passed, failed, or had insufficient data.
    pub status: CertificationStatus,
    /// Aggregate metrics across all cases.
    pub overall_metrics: ClassificationMetrics,
    /// Metrics broken down by constraint type.
    pub per_constraint_metrics: Vec<ConstraintTypeMetrics>,
    /// Total number of cases evaluated.
    pub total_cases: u64,
    /// ISO 8601 UTC timestamp of report generation.
    pub generated_at: String,
}

impl CertificationReport {
    /// Construct a new report.  Timestamp is set to the current UTC time.
    pub(crate) fn new(
        suite_name: &str,
        status: CertificationStatus,
        overall_metrics: ClassificationMetrics,
        per_constraint_metrics: Vec<ConstraintTypeMetrics>,
        total_cases: u64,
    ) -> Self {
        Self {
            suite_name: suite_name.to_string(),
            status,
            overall_metrics,
            per_constraint_metrics,
            total_cases,
            generated_at: Utc::now().to_rfc3339(),
        }
    }

    /// Returns `true` when status is [`CertificationStatus::Passed`].
    pub fn passed(&self) -> bool {
        matches!(self.status, CertificationStatus::Passed)
    }

    /// Render a human-readable Markdown report.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("# Certification Report — {}\n\n", self.suite_name));
        out.push_str(&format!("**Generated at:** {}\n\n", self.generated_at));
        out.push_str(&format!("**Total cases:** {}\n\n", self.total_cases));

        // Status block
        match &self.status {
            CertificationStatus::Passed => {
                out.push_str("**Status:** PASSED ✓\n\n");
            }
            CertificationStatus::Failed { reasons } => {
                out.push_str("**Status:** FAILED ✗\n\n");
                out.push_str("**Failure reasons:**\n\n");
                for reason in reasons {
                    out.push_str(&format!("- {reason}\n"));
                }
                out.push('\n');
            }
            CertificationStatus::Insufficient { reason } => {
                out.push_str("**Status:** INSUFFICIENT DATA ⚠\n\n");
                out.push_str(&format!("**Reason:** {reason}\n\n"));
            }
        }

        // Overall metrics table
        out.push_str("## Overall Metrics\n\n");
        out.push_str("| Metric | Value |\n");
        out.push_str("|--------|-------|\n");
        out.push_str(&format!(
            "| True Positives | {} |\n",
            self.overall_metrics.true_positives
        ));
        out.push_str(&format!(
            "| False Positives | {} |\n",
            self.overall_metrics.false_positives
        ));
        out.push_str(&format!(
            "| True Negatives | {} |\n",
            self.overall_metrics.true_negatives
        ));
        out.push_str(&format!(
            "| False Negatives | {} |\n",
            self.overall_metrics.false_negatives
        ));
        out.push_str(&format!(
            "| Precision | {:.4} |\n",
            self.overall_metrics.precision()
        ));
        out.push_str(&format!(
            "| Recall | {:.4} |\n",
            self.overall_metrics.recall()
        ));
        out.push_str(&format!(
            "| F1 Score | {:.4} |\n",
            self.overall_metrics.f1_score()
        ));
        out.push_str(&format!(
            "| Accuracy | {:.4} |\n",
            self.overall_metrics.accuracy()
        ));
        out.push_str(&format!(
            "| False Positive Rate | {:.4} |\n",
            self.overall_metrics.false_positive_rate()
        ));
        out.push_str(&format!(
            "| False Negative Rate | {:.4} |\n",
            self.overall_metrics.false_negative_rate()
        ));
        out.push_str(&format!(
            "| MCC | {:.4} |\n",
            self.overall_metrics.matthew_correlation_coefficient()
        ));
        out.push('\n');

        // Per-constraint breakdown
        if !self.per_constraint_metrics.is_empty() {
            out.push_str("## Per-Constraint Type Metrics\n\n");
            out.push_str("| Constraint Type | Samples | Precision | Recall | F1 | MCC |\n");
            out.push_str("|-----------------|---------|-----------|--------|----|-----|\n");
            for ct in &self.per_constraint_metrics {
                out.push_str(&format!(
                    "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                    ct.constraint_type,
                    ct.sample_count,
                    ct.metrics.precision(),
                    ct.metrics.recall(),
                    ct.metrics.f1_score(),
                    ct.metrics.matthew_correlation_coefficient(),
                ));
            }
            out.push('\n');
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passed_report() {
        let report = CertificationReport::new(
            "smoke",
            CertificationStatus::Passed,
            ClassificationMetrics {
                true_positives: 10,
                false_positives: 0,
                true_negatives: 10,
                false_negatives: 0,
            },
            vec![],
            20,
        );
        assert!(report.passed());
    }

    #[test]
    fn test_failed_report_not_passed() {
        let report = CertificationReport::new(
            "low_f1",
            CertificationStatus::Failed {
                reasons: vec!["F1 below threshold".to_string()],
            },
            ClassificationMetrics::default(),
            vec![],
            10,
        );
        assert!(!report.passed());
    }

    #[test]
    fn test_markdown_contains_key_sections() {
        let report = CertificationReport::new(
            "markdown-test",
            CertificationStatus::Passed,
            ClassificationMetrics {
                true_positives: 8,
                false_positives: 2,
                true_negatives: 8,
                false_negatives: 2,
            },
            vec![ConstraintTypeMetrics {
                constraint_type: "sh:minCount".to_string(),
                metrics: ClassificationMetrics {
                    true_positives: 8,
                    false_positives: 2,
                    true_negatives: 8,
                    false_negatives: 2,
                },
                sample_count: 20,
            }],
            20,
        );
        let md = report.to_markdown();
        assert!(md.contains("markdown-test"), "should contain suite name");
        assert!(
            md.contains("Overall Metrics"),
            "should contain overall metrics section"
        );
        assert!(
            md.contains("Per-Constraint Type Metrics"),
            "should contain per-constraint section"
        );
        assert!(md.contains("sh:minCount"), "should contain constraint type");
        assert!(md.contains("Precision"), "should contain precision label");
        assert!(md.contains("Recall"), "should contain recall label");
    }
}
