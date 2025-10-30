//! Types for anomaly detection

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Anomaly types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Point anomaly (single outlier)
    Outlier,
    /// Collective anomaly (group of related anomalies)
    CollectiveAnomaly,
    /// Contextual anomaly (depends on context)
    ContextualAnomaly,
    /// Novel pattern not seen before
    NovelPattern,
    /// Data distribution drift
    DataDistributionDrift,
    /// Constraint violation pattern
    ConstraintViolationPattern,
}

impl AnomalyType {
    /// Get severity score (0.0 = low, 1.0 = high)
    pub fn severity(&self) -> f64 {
        match self {
            Self::Outlier => 0.3,
            Self::ContextualAnomaly => 0.5,
            Self::NovelPattern => 0.6,
            Self::CollectiveAnomaly => 0.7,
            Self::ConstraintViolationPattern => 0.8,
            Self::DataDistributionDrift => 0.9,
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Outlier => "Single data point anomaly",
            Self::CollectiveAnomaly => "Group of related anomalies",
            Self::ContextualAnomaly => "Context-dependent anomaly",
            Self::NovelPattern => "Previously unseen pattern",
            Self::ConstraintViolationPattern => "Pattern of constraint violations",
            Self::DataDistributionDrift => "Change in data distribution",
        }
    }
}

/// Anomaly score with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyScore {
    /// Anomaly score (0.0 = normal, 1.0 = highly anomalous)
    pub score: f64,
    /// Confidence in the score
    pub confidence: f64,
    /// Contributing factors
    pub factors: HashMap<String, f64>,
    /// Threshold used for detection
    pub threshold: f64,
}

impl AnomalyScore {
    pub fn new(score: f64, confidence: f64, threshold: f64) -> Self {
        Self {
            score,
            confidence,
            factors: HashMap::new(),
            threshold,
        }
    }

    pub fn is_anomaly(&self) -> bool {
        self.score > self.threshold
    }

    pub fn with_factor(mut self, name: String, value: f64) -> Self {
        self.factors.insert(name, value);
        self
    }
}

/// Detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Unique identifier
    pub id: String,
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    /// Anomaly score
    pub score: AnomalyScore,
    /// Description
    pub description: String,
    /// Affected entities
    pub affected_entities: Vec<String>,
    /// Detection timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Contextual information
    pub context: HashMap<String, String>,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// RDF-specific anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfAnomaly {
    /// Base anomaly information
    pub anomaly: Anomaly,
    /// Affected triples
    pub affected_triples: Vec<String>,
    /// Affected shapes
    pub affected_shapes: Vec<String>,
    /// Property path
    pub property_path: Option<String>,
    /// Expected pattern
    pub expected_pattern: Option<String>,
    /// Actual pattern
    pub actual_pattern: Option<String>,
}

/// Data distribution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataDistribution {
    /// Mean
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Median
    pub median: f64,
    /// Quartiles (Q1, Q2, Q3)
    pub quartiles: [f64; 3],
    /// Interquartile range
    pub iqr: f64,
    /// Min value
    pub min: f64,
    /// Max value
    pub max: f64,
    /// Sample count
    pub count: usize,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
}

impl DataDistribution {
    /// Calculate outlier bounds using IQR method
    pub fn outlier_bounds(&self, multiplier: f64) -> (f64, f64) {
        let lower = self.quartiles[0] - multiplier * self.iqr;
        let upper = self.quartiles[2] + multiplier * self.iqr;
        (lower, upper)
    }

    /// Calculate z-score threshold
    pub fn z_score_threshold(&self, sigma: f64) -> (f64, f64) {
        let lower = self.mean - sigma * self.std_dev;
        let upper = self.mean + sigma * self.std_dev;
        (lower, upper)
    }

    /// Check if distribution is normal
    pub fn is_normal(&self, tolerance: f64) -> bool {
        // Check skewness and kurtosis
        self.skewness.abs() < tolerance && (self.kurtosis - 3.0).abs() < tolerance
    }
}

/// Detection metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionMetrics {
    /// Total samples analyzed
    pub total_samples: usize,
    /// Number of anomalies detected
    pub anomalies_detected: usize,
    /// True positives (if ground truth available)
    pub true_positives: Option<usize>,
    /// False positives (if ground truth available)
    pub false_positives: Option<usize>,
    /// False negatives (if ground truth available)
    pub false_negatives: Option<usize>,
    /// Precision
    pub precision: Option<f64>,
    /// Recall
    pub recall: Option<f64>,
    /// F1 score
    pub f1_score: Option<f64>,
    /// Detection time
    pub detection_time_ms: f64,
    /// Average anomaly score
    pub avg_anomaly_score: f64,
}

impl DetectionMetrics {
    pub fn new(total_samples: usize, anomalies_detected: usize, detection_time_ms: f64) -> Self {
        Self {
            total_samples,
            anomalies_detected,
            true_positives: None,
            false_positives: None,
            false_negatives: None,
            precision: None,
            recall: None,
            f1_score: None,
            detection_time_ms,
            avg_anomaly_score: 0.0,
        }
    }

    pub fn anomaly_rate(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.anomalies_detected as f64 / self.total_samples as f64
        }
    }

    pub fn calculate_metrics(&mut self) {
        if let (Some(tp), Some(fp), Some(fn_)) = (
            self.true_positives,
            self.false_positives,
            self.false_negatives,
        ) {
            // Precision = TP / (TP + FP)
            self.precision = if tp + fp > 0 {
                Some(tp as f64 / (tp + fp) as f64)
            } else {
                None
            };

            // Recall = TP / (TP + FN)
            self.recall = if tp + fn_ > 0 {
                Some(tp as f64 / (tp + fn_) as f64)
            } else {
                None
            };

            // F1 = 2 * (Precision * Recall) / (Precision + Recall)
            if let (Some(p), Some(r)) = (self.precision, self.recall) {
                self.f1_score = if p + r > 0.0 {
                    Some(2.0 * p * r / (p + r))
                } else {
                    None
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_type_severity() {
        assert!(AnomalyType::Outlier.severity() < AnomalyType::CollectiveAnomaly.severity());
    }

    #[test]
    fn test_anomaly_score_is_anomaly() {
        let score = AnomalyScore::new(0.8, 0.9, 0.5);
        assert!(score.is_anomaly());

        let normal = AnomalyScore::new(0.3, 0.9, 0.5);
        assert!(!normal.is_anomaly());
    }

    #[test]
    fn test_data_distribution_outlier_bounds() {
        let dist = DataDistribution {
            mean: 50.0,
            std_dev: 10.0,
            median: 50.0,
            quartiles: [40.0, 50.0, 60.0],
            iqr: 20.0,
            min: 20.0,
            max: 80.0,
            count: 100,
            skewness: 0.0,
            kurtosis: 3.0,
        };

        let (lower, upper) = dist.outlier_bounds(1.5);
        assert_eq!(lower, 10.0); // Q1 - 1.5 * IQR = 40 - 30 = 10
        assert_eq!(upper, 90.0); // Q3 + 1.5 * IQR = 60 + 30 = 90
    }

    #[test]
    fn test_detection_metrics_calculation() {
        let mut metrics = DetectionMetrics::new(100, 10, 50.0);
        metrics.true_positives = Some(8);
        metrics.false_positives = Some(2);
        metrics.false_negatives = Some(5);

        metrics.calculate_metrics();

        assert_eq!(metrics.precision, Some(0.8)); // 8 / (8 + 2)
        assert_eq!(metrics.recall, Some(8.0 / 13.0)); // 8 / (8 + 5)
        assert!(metrics.f1_score.is_some());
    }
}
