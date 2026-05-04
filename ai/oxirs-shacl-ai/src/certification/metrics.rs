//! Classification metrics for ML model certification.
//!
//! Provides binary classification metrics (precision, recall, F1, MCC, etc.)
//! and per-constraint-type breakdowns used by the certification runner.

/// Binary classification metrics for SHACL violation prediction.
///
/// Counts the four cells of a confusion matrix for a binary classifier
/// where "positive" means "violation predicted / present".
#[derive(Debug, Clone, Default)]
pub struct ClassificationMetrics {
    /// Model predicted violation AND ground truth is violation.
    pub true_positives: u64,
    /// Model predicted violation BUT ground truth is no violation.
    pub false_positives: u64,
    /// Model predicted no violation AND ground truth is no violation.
    pub true_negatives: u64,
    /// Model predicted no violation BUT ground truth is violation.
    pub false_negatives: u64,
}

impl ClassificationMetrics {
    /// Precision = TP / (TP + FP).  Returns 0.0 when denominator is 0.
    pub fn precision(&self) -> f64 {
        let denom = (self.true_positives + self.false_positives) as f64;
        if denom == 0.0 {
            0.0
        } else {
            self.true_positives as f64 / denom
        }
    }

    /// Recall = TP / (TP + FN).  Returns 0.0 when denominator is 0.
    pub fn recall(&self) -> f64 {
        let denom = (self.true_positives + self.false_negatives) as f64;
        if denom == 0.0 {
            0.0
        } else {
            self.true_positives as f64 / denom
        }
    }

    /// F1 = 2 * P * R / (P + R).  Returns 0.0 when both P and R are 0.
    pub fn f1_score(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        let denom = p + r;
        if denom == 0.0 {
            0.0
        } else {
            2.0 * p * r / denom
        }
    }

    /// Accuracy = (TP + TN) / (TP + TN + FP + FN).  Returns 0.0 when total is 0.
    pub fn accuracy(&self) -> f64 {
        let total = (self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives) as f64;
        if total == 0.0 {
            0.0
        } else {
            (self.true_positives + self.true_negatives) as f64 / total
        }
    }

    /// False-positive rate = FP / (FP + TN).  Returns 0.0 when denominator is 0.
    pub fn false_positive_rate(&self) -> f64 {
        let denom = (self.false_positives + self.true_negatives) as f64;
        if denom == 0.0 {
            0.0
        } else {
            self.false_positives as f64 / denom
        }
    }

    /// False-negative rate = FN / (FN + TP).  Returns 0.0 when denominator is 0.
    pub fn false_negative_rate(&self) -> f64 {
        let denom = (self.false_negatives + self.true_positives) as f64;
        if denom == 0.0 {
            0.0
        } else {
            self.false_negatives as f64 / denom
        }
    }

    /// Matthews Correlation Coefficient.
    ///
    /// MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)).
    ///
    /// Returns 0.0 when ANY of the four factors inside the square root is 0.
    /// Returns exactly −1.0 when every positive is missed and every negative
    /// is flagged (perfectly wrong classifier).
    pub fn matthew_correlation_coefficient(&self) -> f64 {
        let tp = self.true_positives as f64;
        let tn = self.true_negatives as f64;
        let fp = self.false_positives as f64;
        let fn_ = self.false_negatives as f64;

        let f1 = tp + fp;
        let f2 = tp + fn_;
        let f3 = tn + fp;
        let f4 = tn + fn_;

        if f1 == 0.0 || f2 == 0.0 || f3 == 0.0 || f4 == 0.0 {
            return 0.0;
        }

        let denom = (f1 * f2 * f3 * f4).sqrt();
        (tp * tn - fp * fn_) / denom
    }

    /// Merge another set of metrics into this one (element-wise addition).
    pub fn add(&mut self, other: &ClassificationMetrics) {
        self.true_positives += other.true_positives;
        self.false_positives += other.false_positives;
        self.true_negatives += other.true_negatives;
        self.false_negatives += other.false_negatives;
    }

    /// Total number of samples represented by these metrics.
    pub fn total(&self) -> u64 {
        self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
    }
}

/// Per-constraint-type metrics breakdown.
#[derive(Debug, Clone, Default)]
pub struct ConstraintTypeMetrics {
    /// The constraint type identifier (e.g. `"sh:minCount"`, `"sh:pattern"`).
    pub constraint_type: String,
    /// Accumulated binary classification metrics for this constraint type.
    pub metrics: ClassificationMetrics,
    /// Number of certification cases included in these metrics.
    pub sample_count: u64,
}

/// Confusion matrix for multi-class certification.
///
/// `matrix[true_idx][predicted_idx]` holds the count of cases where the
/// ground-truth class is `labels[true_idx]` and the model predicted
/// `labels[predicted_idx]`.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Ordered label list; both rows and columns follow this order.
    pub labels: Vec<String>,
    /// Row = true label index, column = predicted label index.
    pub matrix: Vec<Vec<u64>>,
}

impl ConfusionMatrix {
    /// Create a new, zeroed confusion matrix for the given labels.
    ///
    /// Labels must be non-empty and unique; duplicate labels are silently
    /// deduplicated (first occurrence wins).
    pub fn new(labels: Vec<String>) -> Self {
        // Deduplicate while preserving order.
        let mut seen = std::collections::HashSet::new();
        let labels: Vec<String> = labels
            .into_iter()
            .filter(|l| seen.insert(l.clone()))
            .collect();

        let n = labels.len();
        let matrix = vec![vec![0u64; n]; n];
        Self { labels, matrix }
    }

    /// Record a single observation.
    ///
    /// If either label is not found in `self.labels` the call is a no-op
    /// (avoids panics from unknown labels).
    pub fn record(&mut self, true_label: &str, predicted_label: &str) {
        let true_idx = self.labels.iter().position(|l| l == true_label);
        let pred_idx = self.labels.iter().position(|l| l == predicted_label);

        if let (Some(ti), Some(pi)) = (true_idx, pred_idx) {
            self.matrix[ti][pi] += 1;
        }
    }

    /// Derive per-class binary `ConstraintTypeMetrics` from this matrix.
    ///
    /// For each label *k*, the one-vs-rest breakdown is:
    /// - TP  = matrix\[k\]\[k\]
    /// - FP  = sum of column *k* excluding row *k*
    /// - FN  = sum of row *k* excluding column *k*
    /// - TN  = everything else
    pub fn per_class_metrics(&self) -> Vec<ConstraintTypeMetrics> {
        let n = self.labels.len();
        let total: u64 = self.matrix.iter().flat_map(|row| row.iter()).sum();

        self.labels
            .iter()
            .enumerate()
            .map(|(k, label)| {
                let tp = self.matrix[k][k];

                let fp: u64 = (0..n).filter(|&r| r != k).map(|r| self.matrix[r][k]).sum();

                let fn_: u64 = (0..n).filter(|&c| c != k).map(|c| self.matrix[k][c]).sum();

                let tn = total.saturating_sub(tp + fp + fn_);

                let sample_count: u64 = self.matrix[k].iter().sum();

                ConstraintTypeMetrics {
                    constraint_type: label.clone(),
                    metrics: ClassificationMetrics {
                        true_positives: tp,
                        false_positives: fp,
                        true_negatives: tn,
                        false_negatives: fn_,
                    },
                    sample_count,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_recall_f1_perfect() {
        let m = ClassificationMetrics {
            true_positives: 10,
            false_positives: 0,
            true_negatives: 10,
            false_negatives: 0,
        };
        assert!(
            (m.precision() - 1.0).abs() < 1e-10,
            "precision should be 1.0"
        );
        assert!((m.recall() - 1.0).abs() < 1e-10, "recall should be 1.0");
        assert!((m.f1_score() - 1.0).abs() < 1e-10, "f1 should be 1.0");
    }

    #[test]
    fn test_zero_denominator_returns_zero() {
        let m = ClassificationMetrics::default();
        assert_eq!(m.precision(), 0.0);
        assert_eq!(m.recall(), 0.0);
        assert_eq!(m.f1_score(), 0.0);
        assert_eq!(m.accuracy(), 0.0);
        assert_eq!(m.false_positive_rate(), 0.0);
        assert_eq!(m.false_negative_rate(), 0.0);
        assert_eq!(m.matthew_correlation_coefficient(), 0.0);
    }

    #[test]
    fn test_mcc_perfectly_wrong() {
        // All TP=0, TN=0, FP=5, FN=5 → MCC should be -1.0
        let m = ClassificationMetrics {
            true_positives: 0,
            false_positives: 5,
            true_negatives: 0,
            false_negatives: 5,
        };
        let mcc = m.matthew_correlation_coefficient();
        assert!(
            (mcc - (-1.0)).abs() < 1e-10,
            "MCC for perfectly wrong classifier should be -1.0, got {mcc}"
        );
    }

    #[test]
    fn test_add_metrics() {
        let mut a = ClassificationMetrics {
            true_positives: 3,
            false_positives: 1,
            true_negatives: 4,
            false_negatives: 2,
        };
        let b = ClassificationMetrics {
            true_positives: 7,
            false_positives: 2,
            true_negatives: 6,
            false_negatives: 1,
        };
        a.add(&b);
        assert_eq!(a.true_positives, 10);
        assert_eq!(a.false_positives, 3);
        assert_eq!(a.true_negatives, 10);
        assert_eq!(a.false_negatives, 3);
    }

    #[test]
    fn test_confusion_matrix_record_and_per_class() {
        let mut cm = ConfusionMatrix::new(vec!["A".to_string(), "B".to_string()]);
        cm.record("A", "A");
        cm.record("A", "A");
        cm.record("A", "B"); // FN for A
        cm.record("B", "A"); // FP for A
        cm.record("B", "B");

        let metrics = cm.per_class_metrics();
        let a_metrics = metrics
            .iter()
            .find(|m| m.constraint_type == "A")
            .expect("A metrics");
        assert_eq!(a_metrics.metrics.true_positives, 2);
        assert_eq!(a_metrics.metrics.false_negatives, 1);
        assert_eq!(a_metrics.metrics.false_positives, 1);
    }

    #[test]
    fn test_confusion_matrix_unknown_label_noop() {
        let mut cm = ConfusionMatrix::new(vec!["A".to_string()]);
        // Should not panic
        cm.record("UNKNOWN", "A");
        cm.record("A", "UNKNOWN");
        assert_eq!(cm.matrix[0][0], 0);
    }
}
