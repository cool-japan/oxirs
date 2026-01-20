//! Main anomaly detector integrating all detection methods

use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};

use super::config::{AnomalyConfig, DetectorType};
use super::drift::{DriftDetector, DriftResult};
use super::ensemble::{EnsembleDetector, EnsembleResult};
use super::explainer::{AnomalyExplainer, ExplanationReport};
use super::novelty::{NoveltyDetector, NoveltyResult};
use super::outlier::{OutlierDetector, OutlierMethod, OutlierResult};
use super::types::{Anomaly, DetectionMetrics};
use crate::{Result, ShaclAiError};

/// Complete detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorResult {
    /// Detected anomalies
    pub anomalies: Vec<Anomaly>,
    /// Detection metrics
    pub metrics: DetectionMetrics,
    /// Outlier detection result (if applicable)
    pub outlier_result: Option<OutlierResult>,
    /// Drift detection result (if applicable)
    pub drift_result: Option<DriftResult>,
    /// Novelty detection result (if applicable)
    pub novelty_result: Option<NoveltyResult>,
    /// Ensemble result (if applicable)
    pub ensemble_result: Option<EnsembleResult>,
    /// Explanations (if enabled)
    pub explanations: Vec<ExplanationReport>,
}

/// Main anomaly detector
pub struct AnomalyDetector {
    config: AnomalyConfig,
    drift_detector: Option<DriftDetector>,
    novelty_detector: Option<NoveltyDetector>,
    explainer: AnomalyExplainer,
}

impl AnomalyDetector {
    pub fn new(config: AnomalyConfig) -> Self {
        let drift_detector = if config.enable_drift_detection {
            Some(DriftDetector::new(100))
        } else {
            None
        };

        let novelty_detector = if config.enable_novelty_detection {
            Some(NoveltyDetector::new())
        } else {
            None
        };

        let explainer = AnomalyExplainer::new();

        Self {
            config,
            drift_detector,
            novelty_detector,
            explainer,
        }
    }

    pub fn config(&self) -> &AnomalyConfig {
        &self.config
    }

    /// Detect anomalies in 1D data
    pub fn detect(&self, data: &Array1<f64>) -> Result<DetectorResult> {
        if !self.config.enabled {
            return Ok(DetectorResult {
                anomalies: Vec::new(),
                metrics: DetectionMetrics::new(data.len(), 0, 0.0),
                outlier_result: None,
                drift_result: None,
                novelty_result: None,
                ensemble_result: None,
                explanations: Vec::new(),
            });
        }

        let start_time = std::time::Instant::now();
        let mut all_anomalies = Vec::new();

        // Run primary detector
        let (outlier_result, drift_result, novelty_result, ensemble_result) =
            match self.config.detector_type {
                DetectorType::Ensemble => {
                    let ensemble_config = self.config.ensemble_config.clone().unwrap_or_default();
                    let ensemble = EnsembleDetector::new(ensemble_config);
                    let result = ensemble.detect_ensemble(data)?;
                    all_anomalies.extend(result.anomalies.clone());
                    (None, None, None, Some(result))
                }
                DetectorType::IsolationForest
                | DetectorType::LocalOutlierFactor
                | DetectorType::StatisticalOutlier => {
                    let method = match self.config.detector_type {
                        DetectorType::IsolationForest => OutlierMethod::IsolationForest,
                        DetectorType::LocalOutlierFactor => OutlierMethod::LOF,
                        DetectorType::StatisticalOutlier => self.config.outlier_method,
                        _ => OutlierMethod::IQR,
                    };

                    let detector = OutlierDetector::new(method);
                    let result = detector.detect_outliers_1d(data)?;
                    all_anomalies.extend(result.outliers.clone());
                    (
                        Some(result),
                        None::<DriftResult>,
                        None::<NoveltyResult>,
                        None::<EnsembleResult>,
                    )
                }
                _ => (
                    None::<OutlierResult>,
                    None::<DriftResult>,
                    None::<NoveltyResult>,
                    None::<EnsembleResult>,
                ),
            };

        // Run drift detection if enabled
        let drift_result = if self.config.enable_drift_detection {
            if let Some(ref drift_detector) = self.drift_detector {
                match drift_detector.detect_drift(data) {
                    Ok(result) => {
                        // Add drift anomalies if drift detected
                        if result.drift_detected {
                            // Create a collective anomaly for drift
                            let drift_anomaly = Anomaly {
                                id: "drift_detected".to_string(),
                                anomaly_type: super::types::AnomalyType::DataDistributionDrift,
                                score: super::types::AnomalyScore::new(
                                    result.drift_score,
                                    result.confidence,
                                    0.7,
                                ),
                                description: format!(
                                    "Data distribution drift detected (score: {:.3})",
                                    result.drift_score
                                ),
                                affected_entities: vec!["entire_dataset".to_string()],
                                timestamp: chrono::Utc::now(),
                                context: std::collections::HashMap::new(),
                                recommendations: result.recommendations.clone(),
                            };
                            all_anomalies.push(drift_anomaly);
                        }
                        Some(result)
                    }
                    Err(_) => None,
                }
            } else {
                None
            }
        } else {
            None
        };

        // Run novelty detection if enabled and detector is trained
        let novelty_result = if self.config.enable_novelty_detection {
            if let Some(ref novelty_detector) = self.novelty_detector {
                match novelty_detector.detect_1d(data) {
                    Ok(result) => {
                        all_anomalies.extend(result.novel_patterns.clone());
                        Some(result)
                    }
                    Err(_) => None, // Detector not trained or error
                }
            } else {
                None
            }
        } else {
            None
        };

        // Filter by confidence threshold
        all_anomalies.retain(|a| a.score.confidence >= self.config.min_confidence);

        // Limit number of anomalies reported
        if all_anomalies.len() > self.config.max_anomalies {
            // Sort by score and keep top N
            all_anomalies.sort_by(|a, b| {
                b.score
                    .score
                    .partial_cmp(&a.score.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all_anomalies.truncate(self.config.max_anomalies);
        }

        // Generate explanations if enabled
        let explanations = if self.config.enable_explainability {
            all_anomalies
                .iter()
                .map(|anomaly| self.explainer.explain(anomaly))
                .collect()
        } else {
            Vec::new()
        };

        let detection_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let mut metrics = DetectionMetrics::new(data.len(), all_anomalies.len(), detection_time);

        // Calculate average anomaly score
        if !all_anomalies.is_empty() {
            metrics.avg_anomaly_score = all_anomalies.iter().map(|a| a.score.score).sum::<f64>()
                / all_anomalies.len() as f64;
        }

        Ok(DetectorResult {
            anomalies: all_anomalies,
            metrics,
            outlier_result,
            drift_result,
            novelty_result,
            ensemble_result,
            explanations,
        })
    }

    /// Train novelty detector on normal data
    pub fn train_novelty(&mut self, normal_data: &[Vec<f64>]) -> Result<()> {
        if let Some(ref mut novelty_detector) = self.novelty_detector {
            novelty_detector.train(normal_data)?;
        } else {
            return Err(ShaclAiError::Analytics(
                "Novelty detection not enabled".to_string(),
            ));
        }
        Ok(())
    }

    /// Update drift detector reference data
    pub fn update_drift_reference(&mut self, data: &[f64]) {
        if let Some(ref mut drift_detector) = self.drift_detector {
            drift_detector.update_reference(data);
        }
    }

    /// Get detection statistics
    pub fn get_statistics(&self) -> DetectorStatistics {
        DetectorStatistics {
            detector_type: self.config.detector_type,
            drift_detection_enabled: self.config.enable_drift_detection,
            novelty_detection_enabled: self.config.enable_novelty_detection,
            explainability_enabled: self.config.enable_explainability,
            threshold: self.config.threshold,
        }
    }
}

/// Detector statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorStatistics {
    pub detector_type: DetectorType,
    pub drift_detection_enabled: bool,
    pub novelty_detection_enabled: bool,
    pub explainability_enabled: bool,
    pub threshold: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let config = AnomalyConfig::default();
        let detector = AnomalyDetector::new(config);
        assert!(detector.config().enabled);
    }

    #[test]
    fn test_detection_disabled() {
        let config = AnomalyConfig {
            enabled: false,
            ..Default::default()
        };

        let detector = AnomalyDetector::new(config);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 100.0]);

        let result = detector.detect(&data).unwrap();
        assert!(result.anomalies.is_empty());
    }

    #[test]
    fn test_ensemble_detection() {
        let config = AnomalyConfig::default();
        let detector = AnomalyDetector::new(config);

        let data = Array1::from_vec(vec![
            1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 100.0,
        ]);

        let result = detector.detect(&data).unwrap();

        assert!(!result.anomalies.is_empty());
        assert!(result.ensemble_result.is_some());
        assert!(result.metrics.detection_time_ms > 0.0);
    }

    #[test]
    fn test_outlier_detection() {
        let config = AnomalyConfig {
            detector_type: DetectorType::StatisticalOutlier,
            ..Default::default()
        };

        let detector = AnomalyDetector::new(config);
        let data = Array1::from_vec(vec![
            1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 100.0,
        ]);

        let result = detector.detect(&data).unwrap();

        assert!(!result.anomalies.is_empty());
        assert!(result.outlier_result.is_some());
    }

    #[test]
    fn test_explainability() {
        let config = AnomalyConfig {
            enable_explainability: true,
            ..Default::default()
        };

        let detector = AnomalyDetector::new(config);
        let data = Array1::from_vec(vec![
            1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 100.0,
        ]);

        let result = detector.detect(&data).unwrap();

        if !result.anomalies.is_empty() {
            assert!(!result.explanations.is_empty());
        }
    }

    #[test]
    fn test_statistics() {
        let config = AnomalyConfig::default();
        let detector = AnomalyDetector::new(config);

        let stats = detector.get_statistics();
        assert_eq!(stats.detector_type, DetectorType::Ensemble);
        assert!(stats.drift_detection_enabled);
        assert!(stats.explainability_enabled);
    }
}
