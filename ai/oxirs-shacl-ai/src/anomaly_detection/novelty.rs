//! Novelty detection for identifying previously unseen patterns

use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::{Anomaly, AnomalyScore, AnomalyType};
use crate::{Result, ShaclAiError};

/// Novelty detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyResult {
    /// Detected novel patterns
    pub novel_patterns: Vec<Anomaly>,
    /// Novelty scores for all samples
    pub novelty_scores: Vec<f64>,
    /// Detection threshold used
    pub threshold: f64,
    /// Model confidence
    pub model_confidence: f64,
}

/// Novelty detector using One-Class SVM-like approach
pub struct NoveltyDetector {
    training_data: Vec<Vec<f64>>,
    threshold: f64,
    nu: f64, // Upper bound on fraction of outliers
}

impl NoveltyDetector {
    pub fn new() -> Self {
        Self {
            training_data: Vec::new(),
            threshold: 0.0,
            nu: 0.1,
        }
    }

    pub fn with_nu(mut self, nu: f64) -> Self {
        self.nu = nu.max(0.0).min(1.0);
        self
    }

    /// Train the novelty detector on normal data
    pub fn train(&mut self, normal_data: &[Vec<f64>]) -> Result<()> {
        if normal_data.is_empty() {
            return Err(ShaclAiError::Analytics(
                "Cannot train novelty detector on empty data".to_string(),
            ));
        }

        self.training_data = normal_data.to_vec();

        // Calculate threshold based on distances in training data
        let mut distances = Vec::new();
        for (i, sample_i) in normal_data.iter().enumerate() {
            for (j, sample_j) in normal_data.iter().enumerate() {
                if i < j {
                    let dist = self.euclidean_distance(sample_i, sample_j);
                    distances.push(dist);
                }
            }
        }

        if distances.is_empty() {
            self.threshold = 1.0;
        } else {
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // Set threshold at (1-nu) quantile
            let quantile_idx = ((1.0 - self.nu) * distances.len() as f64) as usize;
            self.threshold = distances[quantile_idx.min(distances.len() - 1)];
        }

        Ok(())
    }

    /// Detect novelty in new data
    pub fn detect(&self, test_data: &[Vec<f64>]) -> Result<NoveltyResult> {
        if self.training_data.is_empty() {
            return Err(ShaclAiError::Analytics(
                "Novelty detector not trained".to_string(),
            ));
        }

        let mut novel_patterns = Vec::new();
        let mut novelty_scores = Vec::new();

        for (idx, test_sample) in test_data.iter().enumerate() {
            // Calculate minimum distance to training samples
            let min_distance = self
                .training_data
                .iter()
                .map(|train_sample| self.euclidean_distance(test_sample, train_sample))
                .fold(f64::INFINITY, f64::min);

            // Normalize score
            let novelty_score = if self.threshold > 0.0 {
                (min_distance / self.threshold).min(1.0)
            } else {
                0.0
            };

            novelty_scores.push(novelty_score);

            // Check if novel
            if min_distance > self.threshold {
                let confidence = (novelty_score - 0.5).max(0.0) * 2.0; // Scale to 0-1

                let anomaly_score = AnomalyScore::new(novelty_score, confidence, 0.5)
                    .with_factor("min_distance".to_string(), min_distance)
                    .with_factor("threshold".to_string(), self.threshold);

                novel_patterns.push(Anomaly {
                    id: format!("novelty_{}", idx),
                    anomaly_type: AnomalyType::NovelPattern,
                    score: anomaly_score,
                    description: format!(
                        "Novel pattern detected with distance {:.3} (threshold: {:.3})",
                        min_distance, self.threshold
                    ),
                    affected_entities: vec![format!("entity_{}", idx)],
                    timestamp: chrono::Utc::now(),
                    context: HashMap::from([
                        ("min_distance".to_string(), min_distance.to_string()),
                        ("threshold".to_string(), self.threshold.to_string()),
                        ("novelty_score".to_string(), novelty_score.to_string()),
                    ]),
                    recommendations: vec![
                        "Review pattern for legitimacy".to_string(),
                        "Consider adding to training set if valid".to_string(),
                        "Investigate data source".to_string(),
                    ],
                });
            }
        }

        Ok(NoveltyResult {
            novel_patterns,
            novelty_scores,
            threshold: self.threshold,
            model_confidence: 0.8,
        })
    }

    /// Detect novelty in 1D data (convenience method)
    pub fn detect_1d(&self, test_data: &Array1<f64>) -> Result<NoveltyResult> {
        let test_vectors: Vec<Vec<f64>> = test_data.iter().map(|&x| vec![x]).collect();
        self.detect(&test_vectors)
    }

    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let min_len = a.len().min(b.len());
        let sum_sq: f64 = a[..min_len]
            .iter()
            .zip(b[..min_len].iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum();
        sum_sq.sqrt()
    }
}

impl Default for NoveltyDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_novelty_detector_creation() {
        let detector = NoveltyDetector::new();
        assert_eq!(detector.nu, 0.1);
        assert!(detector.training_data.is_empty());
    }

    #[test]
    fn test_train_detector() {
        let mut detector = NoveltyDetector::new();
        let training_data = vec![
            vec![1.0, 2.0],
            vec![1.5, 2.5],
            vec![2.0, 3.0],
            vec![2.5, 3.5],
        ];

        let result = detector.train(&training_data);
        assert!(result.is_ok());
        assert!(!detector.training_data.is_empty());
        assert!(detector.threshold > 0.0);
    }

    #[test]
    fn test_detect_novelty() {
        let mut detector = NoveltyDetector::new().with_nu(0.2);

        // Train on clustered data
        let training_data: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64 * 0.1, i as f64 * 0.1])
            .collect();
        detector.train(&training_data).unwrap();

        // Test with novel point
        let test_data = vec![vec![100.0, 100.0]];
        let result = detector.detect(&test_data).unwrap();

        assert!(!result.novel_patterns.is_empty());
    }

    #[test]
    fn test_detect_1d() {
        let mut detector = NoveltyDetector::new();

        // Train on normal range
        let training: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64]).collect();
        detector.train(&training).unwrap();

        // Test with novel value
        let test_data = Array1::from_vec(vec![1000.0]);
        let result = detector.detect_1d(&test_data).unwrap();

        assert!(!result.novel_patterns.is_empty());
    }

    #[test]
    fn test_untrained_detector() {
        let detector = NoveltyDetector::new();
        let test_data = vec![vec![1.0]];

        let result = detector.detect(&test_data);
        assert!(result.is_err());
    }
}
