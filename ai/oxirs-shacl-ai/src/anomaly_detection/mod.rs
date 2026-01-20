//! Anomaly Detection Module for SHACL Validation
//!
//! This module provides comprehensive anomaly detection capabilities for RDF data,
//! including outlier detection, novelty detection, drift detection, and collective
//! anomaly identification with explainability.

pub mod advanced_explainer;
pub mod config;
pub mod detectors;
pub mod drift;
pub mod ensemble;
pub mod explainer;
pub mod novelty;
pub mod outlier;
pub mod types;

// Re-export key types
pub use advanced_explainer::{
    AdvancedAnomalyExplainer, AdvancedExplanationReport, ConfidenceBreakdown, DetailedExplanation,
    ExplainerConfig, ExplanationDetailLevel, ExplanationTechnique, Priority as ExplainerPriority,
    RemediationSuggestion, VisualizationData,
};
pub use config::{AnomalyConfig, DetectorType, EnsembleConfig};
pub use detectors::{AnomalyDetector, DetectorResult};
pub use drift::{DriftDetector, DriftResult, DriftType};
pub use ensemble::{EnsembleDetector, EnsembleResult};
pub use explainer::{AnomalyExplainer, ExplanationReport};
pub use novelty::{NoveltyDetector, NoveltyResult};
pub use outlier::{OutlierDetector, OutlierMethod, OutlierResult};
pub use types::{
    Anomaly, AnomalyScore, AnomalyType, DataDistribution, DetectionMetrics, RdfAnomaly,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detector_creation() {
        let config = AnomalyConfig::default();
        let detector = AnomalyDetector::new(config);
        assert!(detector.config().enabled);
    }

    #[test]
    fn test_detector_types() {
        assert_eq!(DetectorType::IsolationForest.to_string(), "IsolationForest");
        assert_eq!(DetectorType::LocalOutlierFactor.to_string(), "LOF");
        assert_eq!(DetectorType::StatisticalOutlier.to_string(), "Statistical");
    }

    #[test]
    fn test_anomaly_type_severity() {
        assert!(AnomalyType::Outlier.severity() < AnomalyType::CollectiveAnomaly.severity());
        assert!(
            AnomalyType::NovelPattern.severity() < AnomalyType::DataDistributionDrift.severity()
        );
    }
}
