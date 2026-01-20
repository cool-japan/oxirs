//! Outlier detection algorithms

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1};
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use super::config::OutlierMethod;
use super::types::{Anomaly, AnomalyScore, AnomalyType, DataDistribution};
use crate::{Result, ShaclAiError};

/// Outlier detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierResult {
    /// Detected outliers
    pub outliers: Vec<Anomaly>,
    /// Data distribution
    pub distribution: DataDistribution,
    /// Method used
    pub method: OutlierMethod,
    /// Detection time
    pub detection_time_ms: f64,
}

/// Outlier detector
pub struct OutlierDetector {
    method: OutlierMethod,
    threshold_multiplier: f64,
    min_samples: usize,
}

impl OutlierDetector {
    pub fn new(method: OutlierMethod) -> Self {
        Self {
            method,
            threshold_multiplier: 1.5,
            min_samples: 10,
        }
    }

    pub fn with_threshold(mut self, multiplier: f64) -> Self {
        self.threshold_multiplier = multiplier;
        self
    }

    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Detect outliers in 1D data
    pub fn detect_outliers_1d(&self, data: &Array1<f64>) -> Result<OutlierResult> {
        let start_time = std::time::Instant::now();

        if data.len() < self.min_samples {
            return Err(ShaclAiError::Analytics(format!(
                "Insufficient samples for outlier detection: {} < {}",
                data.len(),
                self.min_samples
            )));
        }

        let distribution = self.calculate_distribution(data)?;
        let outliers = match self.method {
            OutlierMethod::IQR => self.detect_iqr_outliers(data, &distribution)?,
            OutlierMethod::ZScore => self.detect_zscore_outliers(data, &distribution)?,
            OutlierMethod::ModifiedZScore => {
                self.detect_modified_zscore_outliers(data, &distribution)?
            }
            OutlierMethod::IsolationForest => {
                self.detect_isolation_forest_outliers(data, &distribution)?
            }
            OutlierMethod::LOF => self.detect_lof_outliers(data, &distribution)?,
        };

        Ok(OutlierResult {
            outliers,
            distribution,
            method: self.method,
            detection_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Calculate data distribution statistics
    fn calculate_distribution(&self, data: &Array1<f64>) -> Result<DataDistribution> {
        let n = data.len();
        if n == 0 {
            return Err(ShaclAiError::Analytics(
                "Cannot calculate distribution for empty data".to_string(),
            ));
        }

        // Calculate basic statistics
        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // Calculate sorted statistics
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        let q1 = sorted[n / 4];
        let q3 = sorted[3 * n / 4];
        let iqr = q3 - q1;

        // Calculate skewness and kurtosis
        let skewness = if std_dev > 0.0 {
            data.iter()
                .map(|&x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>()
                / n as f64
        } else {
            0.0
        };

        let kurtosis = if std_dev > 0.0 {
            data.iter()
                .map(|&x| ((x - mean) / std_dev).powi(4))
                .sum::<f64>()
                / n as f64
        } else {
            3.0
        };

        Ok(DataDistribution {
            mean,
            std_dev,
            median,
            quartiles: [q1, median, q3],
            iqr,
            min: sorted[0],
            max: sorted[n - 1],
            count: n,
            skewness,
            kurtosis,
        })
    }

    /// IQR-based outlier detection
    fn detect_iqr_outliers(
        &self,
        data: &Array1<f64>,
        dist: &DataDistribution,
    ) -> Result<Vec<Anomaly>> {
        let (lower, upper) = dist.outlier_bounds(self.threshold_multiplier);
        let mut outliers = Vec::new();

        for (idx, &value) in data.iter().enumerate() {
            if value < lower || value > upper {
                let score = if value < lower {
                    (lower - value) / dist.iqr
                } else {
                    (value - upper) / dist.iqr
                };

                let anomaly_score = AnomalyScore::new(score.min(1.0), 0.85, 0.5)
                    .with_factor("iqr_deviation".to_string(), score);

                outliers.push(Anomaly {
                    id: format!("outlier_iqr_{}", idx),
                    anomaly_type: AnomalyType::Outlier,
                    score: anomaly_score,
                    description: format!(
                        "Value {} is outside IQR bounds [{}, {}]",
                        value, lower, upper
                    ),
                    affected_entities: vec![format!("entity_{}", idx)],
                    timestamp: chrono::Utc::now(),
                    context: HashMap::from([
                        ("value".to_string(), value.to_string()),
                        ("lower_bound".to_string(), lower.to_string()),
                        ("upper_bound".to_string(), upper.to_string()),
                    ]),
                    recommendations: vec![
                        "Investigate data quality".to_string(),
                        "Check for data entry errors".to_string(),
                    ],
                });
            }
        }

        Ok(outliers)
    }

    /// Z-score based outlier detection
    fn detect_zscore_outliers(
        &self,
        data: &Array1<f64>,
        dist: &DataDistribution,
    ) -> Result<Vec<Anomaly>> {
        if dist.std_dev == 0.0 {
            return Ok(Vec::new()); // No variance, no outliers
        }

        let threshold = self.threshold_multiplier * 2.0; // Typically 2-3 sigma
        let mut outliers = Vec::new();

        for (idx, &value) in data.iter().enumerate() {
            let z_score = (value - dist.mean).abs() / dist.std_dev;

            if z_score > threshold {
                let anomaly_score = AnomalyScore::new((z_score / 5.0).min(1.0), 0.9, 0.5)
                    .with_factor("z_score".to_string(), z_score);

                outliers.push(Anomaly {
                    id: format!("outlier_zscore_{}", idx),
                    anomaly_type: AnomalyType::Outlier,
                    score: anomaly_score,
                    description: format!(
                        "Value {} has Z-score {} (threshold: {})",
                        value, z_score, threshold
                    ),
                    affected_entities: vec![format!("entity_{}", idx)],
                    timestamp: chrono::Utc::now(),
                    context: HashMap::from([
                        ("value".to_string(), value.to_string()),
                        ("z_score".to_string(), z_score.to_string()),
                        ("mean".to_string(), dist.mean.to_string()),
                        ("std_dev".to_string(), dist.std_dev.to_string()),
                    ]),
                    recommendations: vec![
                        "Review data point validity".to_string(),
                        "Consider data transformation".to_string(),
                    ],
                });
            }
        }

        Ok(outliers)
    }

    /// Modified Z-score (using median)
    fn detect_modified_zscore_outliers(
        &self,
        data: &Array1<f64>,
        dist: &DataDistribution,
    ) -> Result<Vec<Anomaly>> {
        // Calculate MAD (Median Absolute Deviation)
        let mad = {
            let mut deviations: Vec<f64> = data.iter().map(|&x| (x - dist.median).abs()).collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = deviations.len();
            if n % 2 == 0 {
                (deviations[n / 2 - 1] + deviations[n / 2]) / 2.0
            } else {
                deviations[n / 2]
            }
        };

        if mad == 0.0 {
            return Ok(Vec::new());
        }

        let threshold = self.threshold_multiplier * 2.0;
        let mut outliers = Vec::new();

        for (idx, &value) in data.iter().enumerate() {
            let modified_z = 0.6745 * (value - dist.median).abs() / mad;

            if modified_z > threshold {
                let anomaly_score = AnomalyScore::new((modified_z / 5.0).min(1.0), 0.88, 0.5)
                    .with_factor("modified_z_score".to_string(), modified_z);

                outliers.push(Anomaly {
                    id: format!("outlier_modified_z_{}", idx),
                    anomaly_type: AnomalyType::Outlier,
                    score: anomaly_score,
                    description: format!(
                        "Value {} has modified Z-score {} (threshold: {})",
                        value, modified_z, threshold
                    ),
                    affected_entities: vec![format!("entity_{}", idx)],
                    timestamp: chrono::Utc::now(),
                    context: HashMap::from([
                        ("value".to_string(), value.to_string()),
                        ("modified_z_score".to_string(), modified_z.to_string()),
                        ("median".to_string(), dist.median.to_string()),
                        ("mad".to_string(), mad.to_string()),
                    ]),
                    recommendations: vec!["Investigate extreme value".to_string()],
                });
            }
        }

        Ok(outliers)
    }

    /// Simplified Isolation Forest implementation
    fn detect_isolation_forest_outliers(
        &self,
        data: &Array1<f64>,
        _dist: &DataDistribution,
    ) -> Result<Vec<Anomaly>> {
        // Simplified implementation: use random trees to isolate points
        let n_trees = 100;
        let subsample_size = (data.len() / 4).max(8);
        let mut rng = Random::seed(42);

        let mut isolation_scores = vec![0.0; data.len()];

        for _ in 0..n_trees {
            // Create a random subsample
            let indices: Vec<usize> = (0..data.len())
                .filter(|_| rng.random::<f64>() < (subsample_size as f64 / data.len() as f64))
                .take(subsample_size)
                .collect();

            if indices.len() < 2 {
                continue;
            }

            // Build a simple tree with random splits
            let mut tree_depth = HashMap::new();
            for &idx in &indices {
                let depth = self.calculate_isolation_depth(data[idx], data, &indices, &mut rng);
                tree_depth.insert(idx, depth);
            }

            // Update scores
            for (&idx, &depth) in &tree_depth {
                isolation_scores[idx] += 1.0 / (depth as f64 + 1.0);
            }
        }

        // Normalize scores
        let max_score = isolation_scores.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_score > 0.0 {
            for score in &mut isolation_scores {
                *score /= max_score;
            }
        }

        // Identify outliers
        let threshold = 0.7; // High isolation score = anomaly
        let mut outliers = Vec::new();

        for (idx, &score) in isolation_scores.iter().enumerate() {
            if score > threshold {
                let anomaly_score = AnomalyScore::new(score, 0.8, threshold)
                    .with_factor("isolation_score".to_string(), score);

                outliers.push(Anomaly {
                    id: format!("outlier_iforest_{}", idx),
                    anomaly_type: AnomalyType::Outlier,
                    score: anomaly_score,
                    description: format!(
                        "Value {} is easily isolated (score: {:.3})",
                        data[idx], score
                    ),
                    affected_entities: vec![format!("entity_{}", idx)],
                    timestamp: chrono::Utc::now(),
                    context: HashMap::from([
                        ("value".to_string(), data[idx].to_string()),
                        ("isolation_score".to_string(), score.to_string()),
                    ]),
                    recommendations: vec!["Verify data authenticity".to_string()],
                });
            }
        }

        Ok(outliers)
    }

    fn calculate_isolation_depth<R: Rng>(
        &self,
        value: f64,
        data: &Array1<f64>,
        indices: &[usize],
        rng: &mut Random<R>,
    ) -> usize {
        let mut depth = 0;
        let mut current_indices = indices.to_vec();
        let max_depth = 10;

        while current_indices.len() > 1 && depth < max_depth {
            // Random split
            let min_val = current_indices
                .iter()
                .map(|&i| data[i])
                .fold(f64::INFINITY, f64::min);
            let max_val = current_indices
                .iter()
                .map(|&i| data[i])
                .fold(f64::NEG_INFINITY, f64::max);

            if (max_val - min_val).abs() < 1e-10 {
                break;
            }

            let split_point = min_val + rng.random::<f64>() * (max_val - min_val);

            current_indices.retain(|&i| {
                let retain = if value <= split_point {
                    data[i] <= split_point
                } else {
                    data[i] > split_point
                };
                retain
            });

            depth += 1;
        }

        depth
    }

    /// Simplified LOF implementation
    fn detect_lof_outliers(
        &self,
        data: &Array1<f64>,
        _dist: &DataDistribution,
    ) -> Result<Vec<Anomaly>> {
        let k = 5.min(data.len() / 2); // Number of neighbors
        if k < 2 {
            return Ok(Vec::new());
        }

        let mut lof_scores = vec![0.0; data.len()];

        for i in 0..data.len() {
            // Find k nearest neighbors
            let mut distances: Vec<(usize, f64)> = (0..data.len())
                .filter(|&j| i != j)
                .map(|j| (j, (data[i] - data[j]).abs()))
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let neighbors: Vec<usize> = distances.iter().take(k).map(|&(idx, _)| idx).collect();

            // Calculate local reachability density
            let lrd_i = self.calculate_lrd(i, &neighbors, data, k);

            // Calculate LOF
            let lof = if lrd_i > 0.0 {
                let sum_lrd: f64 = neighbors
                    .iter()
                    .map(|&j| {
                        let j_neighbors = self.find_k_neighbors(j, data, k);
                        self.calculate_lrd(j, &j_neighbors, data, k)
                    })
                    .sum();
                (sum_lrd / (k as f64 * lrd_i)).max(0.0)
            } else {
                1.0
            };

            lof_scores[i] = lof;
        }

        // Identify outliers (LOF > 1.5 typically indicates outlier)
        let threshold = 1.5;
        let mut outliers = Vec::new();

        for (idx, &lof) in lof_scores.iter().enumerate() {
            if lof > threshold {
                let normalized_score = ((lof - 1.0) / 2.0).min(1.0);
                let anomaly_score = AnomalyScore::new(normalized_score, 0.82, 0.5)
                    .with_factor("lof_score".to_string(), lof);

                outliers.push(Anomaly {
                    id: format!("outlier_lof_{}", idx),
                    anomaly_type: AnomalyType::Outlier,
                    score: anomaly_score,
                    description: format!("Value {} has LOF score {:.3}", data[idx], lof),
                    affected_entities: vec![format!("entity_{}", idx)],
                    timestamp: chrono::Utc::now(),
                    context: HashMap::from([
                        ("value".to_string(), data[idx].to_string()),
                        ("lof_score".to_string(), lof.to_string()),
                    ]),
                    recommendations: vec!["Check for local density anomaly".to_string()],
                });
            }
        }

        Ok(outliers)
    }

    fn find_k_neighbors(&self, idx: usize, data: &Array1<f64>, k: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f64)> = (0..data.len())
            .filter(|&j| idx != j)
            .map(|j| (j, (data[idx] - data[j]).abs()))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.iter().take(k).map(|&(i, _)| i).collect()
    }

    fn calculate_lrd(&self, idx: usize, neighbors: &[usize], data: &Array1<f64>, k: usize) -> f64 {
        if neighbors.is_empty() {
            return 1.0;
        }

        let sum_reach_dist: f64 = neighbors
            .iter()
            .map(|&j| {
                let dist = (data[idx] - data[j]).abs();
                // Reachability distance is max of actual distance and k-distance of j
                let k_dist_j = self.calculate_k_distance(j, data, k);
                dist.max(k_dist_j)
            })
            .sum();

        if sum_reach_dist > 0.0 {
            neighbors.len() as f64 / sum_reach_dist
        } else {
            1.0
        }
    }

    fn calculate_k_distance(&self, idx: usize, data: &Array1<f64>, k: usize) -> f64 {
        let mut distances: Vec<f64> = (0..data.len())
            .filter(|&j| idx != j)
            .map(|j| (data[idx] - data[j]).abs())
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        distances
            .get(k.min(distances.len()) - 1)
            .copied()
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outlier_detector_creation() {
        let detector = OutlierDetector::new(OutlierMethod::IQR);
        assert_eq!(detector.method, OutlierMethod::IQR);
        assert_eq!(detector.threshold_multiplier, 1.5);
    }

    #[test]
    fn test_iqr_outlier_detection() {
        let detector = OutlierDetector::new(OutlierMethod::IQR).with_min_samples(5);
        let data = Array1::from_vec(vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 100.0]);

        let result = detector.detect_outliers_1d(&data).unwrap();
        assert!(!result.outliers.is_empty());
        assert_eq!(result.method, OutlierMethod::IQR);
    }

    #[test]
    fn test_zscore_outlier_detection() {
        let detector = OutlierDetector::new(OutlierMethod::ZScore);
        let data = Array1::from_vec(vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 50.0]);

        let result = detector.detect_outliers_1d(&data).unwrap();
        assert!(!result.outliers.is_empty());
    }

    #[test]
    fn test_distribution_calculation() {
        let detector = OutlierDetector::new(OutlierMethod::IQR);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let dist = detector.calculate_distribution(&data).unwrap();
        assert_eq!(dist.mean, 5.5);
        assert_eq!(dist.median, 5.5);
        assert_eq!(dist.count, 10);
    }

    #[test]
    fn test_insufficient_samples() {
        let detector = OutlierDetector::new(OutlierMethod::IQR).with_min_samples(10);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = detector.detect_outliers_1d(&data);
        assert!(result.is_err());
    }
}
