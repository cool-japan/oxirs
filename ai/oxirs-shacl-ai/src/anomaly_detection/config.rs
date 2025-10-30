//! Configuration for anomaly detection

use serde::{Deserialize, Serialize};
use std::fmt;

/// Detector types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectorType {
    /// Isolation Forest
    IsolationForest,
    /// Local Outlier Factor
    LocalOutlierFactor,
    /// One-Class SVM
    OneClassSVM,
    /// Statistical outlier detection (Z-score, IQR)
    StatisticalOutlier,
    /// DBSCAN-based anomaly detection
    DBSCANAnomaly,
    /// Autoencoder-based detection
    Autoencoder,
    /// Ensemble of detectors
    Ensemble,
}

impl fmt::Display for DetectorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IsolationForest => write!(f, "IsolationForest"),
            Self::LocalOutlierFactor => write!(f, "LOF"),
            Self::OneClassSVM => write!(f, "OneClassSVM"),
            Self::StatisticalOutlier => write!(f, "Statistical"),
            Self::DBSCANAnomaly => write!(f, "DBSCAN"),
            Self::Autoencoder => write!(f, "Autoencoder"),
            Self::Ensemble => write!(f, "Ensemble"),
        }
    }
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyConfig {
    /// Enable anomaly detection
    pub enabled: bool,
    /// Detector type
    pub detector_type: DetectorType,
    /// Anomaly threshold (0.0 - 1.0)
    pub threshold: f64,
    /// Minimum confidence for reporting
    pub min_confidence: f64,
    /// Enable real-time detection
    pub enable_realtime: bool,
    /// Enable drift detection
    pub enable_drift_detection: bool,
    /// Enable novelty detection
    pub enable_novelty_detection: bool,
    /// Enable explainability
    pub enable_explainability: bool,
    /// Outlier detection method
    pub outlier_method: OutlierMethod,
    /// Ensemble configuration
    pub ensemble_config: Option<EnsembleConfig>,
    /// Adaptive threshold tuning
    pub adaptive_threshold: bool,
    /// Maximum anomalies to report
    pub max_anomalies: usize,
    /// Contamination rate (expected % of anomalies)
    pub contamination: f64,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detector_type: DetectorType::Ensemble,
            threshold: 0.7,
            min_confidence: 0.6,
            enable_realtime: true,
            enable_drift_detection: true,
            enable_novelty_detection: true,
            enable_explainability: true,
            outlier_method: OutlierMethod::IQR,
            ensemble_config: Some(EnsembleConfig::default()),
            adaptive_threshold: true,
            max_anomalies: 1000,
            contamination: 0.1,
        }
    }
}

/// Outlier detection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutlierMethod {
    /// Interquartile Range method
    IQR,
    /// Z-score method
    ZScore,
    /// Modified Z-score (using median)
    ModifiedZScore,
    /// Isolation Forest
    IsolationForest,
    /// Local Outlier Factor
    LOF,
}

impl OutlierMethod {
    pub fn name(&self) -> &'static str {
        match self {
            Self::IQR => "IQR",
            Self::ZScore => "Z-Score",
            Self::ModifiedZScore => "Modified Z-Score",
            Self::IsolationForest => "Isolation Forest",
            Self::LOF => "Local Outlier Factor",
        }
    }
}

/// Ensemble detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Detectors to use in ensemble
    pub detectors: Vec<DetectorType>,
    /// Voting strategy
    pub voting_strategy: VotingStrategy,
    /// Minimum votes required for anomaly detection
    pub min_votes: usize,
    /// Weight for each detector
    pub detector_weights: Vec<f64>,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            detectors: vec![
                DetectorType::IsolationForest,
                DetectorType::LocalOutlierFactor,
                DetectorType::StatisticalOutlier,
            ],
            voting_strategy: VotingStrategy::WeightedMajority,
            min_votes: 2,
            detector_weights: vec![1.0, 1.0, 0.8],
        }
    }
}

/// Voting strategies for ensemble detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VotingStrategy {
    /// Simple majority voting
    Majority,
    /// Weighted majority voting
    WeightedMajority,
    /// Unanimous voting
    Unanimous,
    /// Average score
    AverageScore,
    /// Maximum score
    MaxScore,
}

impl VotingStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Majority => "Majority",
            Self::WeightedMajority => "Weighted Majority",
            Self::Unanimous => "Unanimous",
            Self::AverageScore => "Average Score",
            Self::MaxScore => "Maximum Score",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_config_default() {
        let config = AnomalyConfig::default();
        assert!(config.enabled);
        assert_eq!(config.detector_type, DetectorType::Ensemble);
        assert_eq!(config.threshold, 0.7);
        assert!(config.enable_explainability);
    }

    #[test]
    fn test_detector_type_display() {
        assert_eq!(DetectorType::IsolationForest.to_string(), "IsolationForest");
        assert_eq!(DetectorType::LocalOutlierFactor.to_string(), "LOF");
    }

    #[test]
    fn test_ensemble_config() {
        let config = EnsembleConfig::default();
        assert_eq!(config.detectors.len(), 3);
        assert_eq!(config.min_votes, 2);
        assert_eq!(config.voting_strategy, VotingStrategy::WeightedMajority);
    }

    #[test]
    fn test_outlier_method_names() {
        assert_eq!(OutlierMethod::IQR.name(), "IQR");
        assert_eq!(OutlierMethod::ZScore.name(), "Z-Score");
        assert_eq!(OutlierMethod::LOF.name(), "Local Outlier Factor");
    }
}
