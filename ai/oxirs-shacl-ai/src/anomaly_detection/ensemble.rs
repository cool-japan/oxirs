//! Ensemble anomaly detection combining multiple detectors

use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};

use super::config::{DetectorType, EnsembleConfig, VotingStrategy};
use super::outlier::{OutlierDetector, OutlierMethod};
use super::types::{Anomaly, AnomalyScore, AnomalyType};
use crate::{Result, ShaclAiError};

/// Ensemble detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResult {
    /// Detected anomalies
    pub anomalies: Vec<Anomaly>,
    /// Individual detector results
    pub detector_results: Vec<DetectorResult>,
    /// Voting summary
    pub voting_summary: VotingSummary,
}

/// Individual detector result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorResult {
    /// Detector type
    pub detector_type: DetectorType,
    /// Number of anomalies detected
    pub anomalies_count: usize,
    /// Detection time
    pub detection_time_ms: f64,
    /// Detector weight
    pub weight: f64,
}

/// Voting summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingSummary {
    /// Total votes per sample
    pub total_votes: Vec<usize>,
    /// Weighted votes per sample
    pub weighted_votes: Vec<f64>,
    /// Samples identified as anomalies
    pub anomaly_indices: Vec<usize>,
}

/// Ensemble detector
pub struct EnsembleDetector {
    config: EnsembleConfig,
}

impl EnsembleDetector {
    pub fn new(config: EnsembleConfig) -> Self {
        Self { config }
    }

    /// Detect anomalies using ensemble of detectors
    pub fn detect_ensemble(&self, data: &Array1<f64>) -> Result<EnsembleResult> {
        let mut detector_results = Vec::new();
        let mut all_detections: Vec<Vec<bool>> = Vec::new();

        // Run each detector
        for (idx, detector_type) in self.config.detectors.iter().enumerate() {
            let start_time = std::time::Instant::now();
            let weight = self
                .config
                .detector_weights
                .get(idx)
                .copied()
                .unwrap_or(1.0);

            let detections = match detector_type {
                DetectorType::IsolationForest => self.run_isolation_forest(data)?,
                DetectorType::LocalOutlierFactor => self.run_lof(data)?,
                DetectorType::StatisticalOutlier => self.run_statistical(data)?,
                DetectorType::DBSCANAnomaly => self.run_dbscan(data)?,
                _ => {
                    // Fallback to statistical for unsupported types
                    self.run_statistical(data)?
                }
            };

            let detection_time = start_time.elapsed().as_secs_f64() * 1000.0;

            detector_results.push(DetectorResult {
                detector_type: *detector_type,
                anomalies_count: detections.iter().filter(|&&x| x).count(),
                detection_time_ms: detection_time,
                weight,
            });

            all_detections.push(detections);
        }

        // Aggregate results using voting strategy
        let (anomaly_indices, voting_summary) =
            self.aggregate_results(&all_detections, data.len())?;

        // Create anomaly objects for detected anomalies
        let mut anomalies = Vec::new();
        for &idx in &anomaly_indices {
            let value = data[idx];
            let weighted_vote = voting_summary.weighted_votes[idx];

            let anomaly_score = AnomalyScore::new(weighted_vote, 0.85, 0.5).with_factor(
                "ensemble_vote".to_string(),
                voting_summary.total_votes[idx] as f64,
            );

            anomalies.push(Anomaly {
                id: format!("ensemble_anomaly_{}", idx),
                anomaly_type: AnomalyType::Outlier,
                score: anomaly_score,
                description: format!(
                    "Ensemble detected anomaly at value {} with {} votes",
                    value, voting_summary.total_votes[idx]
                ),
                affected_entities: vec![format!("entity_{}", idx)],
                timestamp: chrono::Utc::now(),
                context: std::collections::HashMap::from([
                    ("value".to_string(), value.to_string()),
                    (
                        "total_votes".to_string(),
                        voting_summary.total_votes[idx].to_string(),
                    ),
                    ("weighted_vote".to_string(), weighted_vote.to_string()),
                ]),
                recommendations: vec![
                    "Multiple detectors agree on anomaly".to_string(),
                    "High confidence detection".to_string(),
                ],
            });
        }

        Ok(EnsembleResult {
            anomalies,
            detector_results,
            voting_summary,
        })
    }

    fn run_isolation_forest(&self, data: &Array1<f64>) -> Result<Vec<bool>> {
        let detector = OutlierDetector::new(OutlierMethod::IsolationForest);
        let result = detector.detect_outliers_1d(data)?;

        let mut detections = vec![false; data.len()];
        for anomaly in result.outliers {
            if let Some(idx_str) = anomaly.id.strip_prefix("outlier_iforest_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if idx < detections.len() {
                        detections[idx] = true;
                    }
                }
            }
        }

        Ok(detections)
    }

    fn run_lof(&self, data: &Array1<f64>) -> Result<Vec<bool>> {
        let detector = OutlierDetector::new(OutlierMethod::LOF);
        let result = detector.detect_outliers_1d(data)?;

        let mut detections = vec![false; data.len()];
        for anomaly in result.outliers {
            if let Some(idx_str) = anomaly.id.strip_prefix("outlier_lof_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if idx < detections.len() {
                        detections[idx] = true;
                    }
                }
            }
        }

        Ok(detections)
    }

    fn run_statistical(&self, data: &Array1<f64>) -> Result<Vec<bool>> {
        let detector = OutlierDetector::new(OutlierMethod::IQR);
        let result = detector.detect_outliers_1d(data)?;

        let mut detections = vec![false; data.len()];
        for anomaly in result.outliers {
            if let Some(idx_str) = anomaly.id.strip_prefix("outlier_iqr_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if idx < detections.len() {
                        detections[idx] = true;
                    }
                }
            }
        }

        Ok(detections)
    }

    fn run_dbscan(&self, data: &Array1<f64>) -> Result<Vec<bool>> {
        // Simplified DBSCAN: points far from cluster are outliers
        let eps = self.calculate_eps(data);
        let min_pts = 3;

        let mut detections = vec![false; data.len()];

        for (i, detection) in detections.iter_mut().enumerate() {
            let neighbors = self.find_neighbors(i, data, eps);
            if neighbors.len() < min_pts {
                *detection = true;
            }
        }

        Ok(detections)
    }

    fn calculate_eps(&self, data: &Array1<f64>) -> f64 {
        // Use standard deviation as epsilon
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }

    fn find_neighbors(&self, idx: usize, data: &Array1<f64>, eps: f64) -> Vec<usize> {
        let value = data[idx];
        (0..data.len())
            .filter(|&i| (data[i] - value).abs() <= eps)
            .collect()
    }

    fn aggregate_results(
        &self,
        all_detections: &[Vec<bool>],
        data_len: usize,
    ) -> Result<(Vec<usize>, VotingSummary)> {
        let mut total_votes = vec![0; data_len];
        let mut weighted_votes = vec![0.0; data_len];

        // Count votes
        for (detector_idx, detections) in all_detections.iter().enumerate() {
            let weight = self
                .config
                .detector_weights
                .get(detector_idx)
                .copied()
                .unwrap_or(1.0);

            for (sample_idx, &is_anomaly) in detections.iter().enumerate() {
                if is_anomaly {
                    total_votes[sample_idx] += 1;
                    weighted_votes[sample_idx] += weight;
                }
            }
        }

        // Normalize weighted votes
        let max_weight: f64 = self.config.detector_weights.iter().sum();
        if max_weight > 0.0 {
            for vote in &mut weighted_votes {
                *vote /= max_weight;
            }
        }

        // Apply voting strategy
        let anomaly_indices = match self.config.voting_strategy {
            VotingStrategy::Majority => {
                let threshold = (all_detections.len() + 1) / 2;
                total_votes
                    .iter()
                    .enumerate()
                    .filter(|(_, &votes)| votes >= threshold)
                    .map(|(idx, _)| idx)
                    .collect::<Vec<usize>>()
            }
            VotingStrategy::WeightedMajority => {
                let threshold = 0.5;
                weighted_votes
                    .iter()
                    .enumerate()
                    .filter(|(_, &votes)| votes >= threshold)
                    .map(|(idx, _)| idx)
                    .collect()
            }
            VotingStrategy::Unanimous => total_votes
                .iter()
                .enumerate()
                .filter(|(_, &votes)| votes == all_detections.len())
                .map(|(idx, _)| idx)
                .collect(),
            VotingStrategy::AverageScore => {
                let threshold = 0.5;
                weighted_votes
                    .iter()
                    .enumerate()
                    .filter(|(_, &votes)| votes >= threshold)
                    .map(|(idx, _)| idx)
                    .collect()
            }
            VotingStrategy::MaxScore => weighted_votes
                .iter()
                .enumerate()
                .filter(|(_, &votes)| votes > 0.0)
                .map(|(idx, _)| idx)
                .collect(),
        };

        Ok((
            anomaly_indices.clone(),
            VotingSummary {
                total_votes,
                weighted_votes,
                anomaly_indices,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_detector_creation() {
        let config = EnsembleConfig::default();
        let detector = EnsembleDetector::new(config);
        assert_eq!(detector.config.detectors.len(), 3);
    }

    #[test]
    fn test_ensemble_detection() {
        let config = EnsembleConfig::default();
        let detector = EnsembleDetector::new(config);

        let data = Array1::from_vec(vec![
            1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 100.0,
        ]);

        let result = detector.detect_ensemble(&data).unwrap();

        assert!(!result.anomalies.is_empty());
        assert_eq!(result.detector_results.len(), 3);
    }

    #[test]
    fn test_voting_strategies() {
        let config = EnsembleConfig {
            voting_strategy: VotingStrategy::Unanimous,
            ..Default::default()
        };

        let detector = EnsembleDetector::new(config);
        let data = Array1::from_vec(vec![
            1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 100.0,
        ]);

        let result = detector.detect_ensemble(&data).unwrap();
        // Unanimous voting is strict, so may have fewer detections
        assert!(result.voting_summary.anomaly_indices.len() <= 1);
    }
}
