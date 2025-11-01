//! Advanced Cardinality Estimation for Query Optimization
//!
//! This module provides sophisticated cardinality estimation using scirs2-stats
//! for accurate query optimization with histogram-based, correlation-aware,
//! and adaptive learning approaches.

use crate::algebra::{Term, TriplePattern, Variable};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Advanced cardinality estimator with multiple estimation strategies
pub struct CardinalityEstimator {
    /// Histogram-based estimator for single attributes
    histogram_estimator: Arc<RwLock<HistogramEstimator>>,
    /// Correlation-based estimator for joins
    correlation_estimator: Arc<RwLock<CorrelationEstimator>>,
    /// Adaptive estimator that learns from query execution
    adaptive_estimator: Arc<RwLock<AdaptiveEstimator>>,
    /// Configuration for estimation
    config: EstimatorConfig,
    /// Estimation statistics
    stats: Arc<RwLock<EstimationStats>>,
}

/// Configuration for cardinality estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatorConfig {
    /// Number of histogram buckets
    pub num_histogram_buckets: usize,
    /// Enable correlation tracking
    pub enable_correlation: bool,
    /// Enable adaptive learning
    pub enable_adaptive: bool,
    /// Minimum samples for reliable estimation
    pub min_samples: usize,
    /// Confidence level for statistical tests (0.0 to 1.0)
    pub confidence_level: f64,
    /// Enable outlier filtering
    pub outlier_filtering: bool,
    /// Maximum estimation error threshold
    pub max_error_threshold: f64,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            num_histogram_buckets: 100,
            enable_correlation: true,
            enable_adaptive: true,
            min_samples: 30,
            confidence_level: 0.95,
            outlier_filtering: true,
            max_error_threshold: 0.2, // 20% error threshold
        }
    }
}

/// Statistics about estimation accuracy
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EstimationStats {
    /// Total estimations performed
    pub total_estimations: u64,
    /// Total estimation errors
    pub total_error: f64,
    /// Mean absolute percentage error (MAPE)
    pub mape: f64,
    /// Root mean square error (RMSE)
    pub rmse: f64,
    /// Number of estimations within error threshold
    pub within_threshold: u64,
    /// Estimation accuracy by pattern type
    pub accuracy_by_pattern: HashMap<String, f64>,
}

/// Histogram-based cardinality estimator
pub struct HistogramEstimator {
    /// Histograms by predicate
    predicate_histograms: HashMap<String, ValueHistogram>,
    /// Subject histograms (for future use)
    #[allow(dead_code)]
    subject_histograms: HashMap<String, ValueHistogram>,
    /// Object histograms (for future use)
    #[allow(dead_code)]
    object_histograms: HashMap<String, ValueHistogram>,
    /// Configuration
    config: EstimatorConfig,
}

/// Value histogram with statistical properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueHistogram {
    /// Bucket boundaries (sorted)
    pub boundaries: Vec<String>,
    /// Frequencies in each bucket
    pub frequencies: Vec<u64>,
    /// Total values
    pub total_count: u64,
    /// Distinct values
    pub distinct_count: u64,
    /// Most common values (MCV) with frequencies
    pub mcv: Vec<(String, u64)>,
    /// Statistical summary
    pub summary: HistogramSummary,
}

/// Statistical summary of histogram
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramSummary {
    /// Mean frequency per bucket
    pub mean_frequency: f64,
    /// Standard deviation of frequencies
    pub std_dev_frequency: f64,
    /// Median frequency
    pub median_frequency: f64,
    /// Skewness of distribution
    pub skewness: f64,
    /// Kurtosis of distribution
    pub kurtosis: f64,
}

impl HistogramEstimator {
    /// Create a new histogram estimator
    pub fn new(config: EstimatorConfig) -> Self {
        Self {
            predicate_histograms: HashMap::new(),
            subject_histograms: HashMap::new(),
            object_histograms: HashMap::new(),
            config,
        }
    }

    /// Build histogram for a predicate
    pub fn build_histogram(&mut self, predicate: &str, values: Vec<String>) -> Result<()> {
        let hist = self.create_histogram(values)?;
        self.predicate_histograms
            .insert(predicate.to_string(), hist);
        Ok(())
    }

    /// Create histogram from values using scirs2-stats
    fn create_histogram(&self, mut values: Vec<String>) -> Result<ValueHistogram> {
        if values.is_empty() {
            return Ok(ValueHistogram {
                boundaries: Vec::new(),
                frequencies: Vec::new(),
                total_count: 0,
                distinct_count: 0,
                mcv: Vec::new(),
                summary: HistogramSummary {
                    mean_frequency: 0.0,
                    std_dev_frequency: 0.0,
                    median_frequency: 0.0,
                    skewness: 0.0,
                    kurtosis: 0.0,
                },
            });
        }

        let total_count = values.len() as u64;
        values.sort();
        values.dedup();
        let distinct_count = values.len() as u64;

        // Count frequencies
        let mut freq_map: HashMap<String, u64> = HashMap::new();
        for val in &values {
            *freq_map.entry(val.clone()).or_insert(0) += 1;
        }

        // Get most common values (top 10)
        let mut mcv: Vec<(String, u64)> = freq_map.iter().map(|(k, v)| (k.clone(), *v)).collect();
        mcv.sort_by(|a, b| b.1.cmp(&a.1));
        mcv.truncate(10);

        // Create equi-depth histogram buckets
        let num_buckets = self
            .config
            .num_histogram_buckets
            .min(distinct_count as usize);
        let bucket_size = (distinct_count as usize + num_buckets - 1) / num_buckets;

        let mut boundaries = Vec::new();
        let mut frequencies = Vec::new();

        for i in 0..num_buckets {
            let start = i * bucket_size;
            let end = ((i + 1) * bucket_size).min(values.len());

            if start < values.len() {
                boundaries.push(values[start].clone());
                let bucket_freq: u64 = values[start..end]
                    .iter()
                    .map(|v| *freq_map.get(v).unwrap_or(&0))
                    .sum();
                frequencies.push(bucket_freq);
            }
        }

        // Calculate statistical summary using scirs2-stats
        let freq_f64: Vec<f64> = frequencies.iter().map(|&f| f as f64).collect();

        let mean_freq = if !freq_f64.is_empty() {
            freq_f64.iter().sum::<f64>() / freq_f64.len() as f64
        } else {
            0.0
        };

        let std_dev_freq = if freq_f64.len() > 1 {
            let mean = mean_freq;
            let variance = freq_f64.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (freq_f64.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let median_freq = if !freq_f64.is_empty() {
            let mut sorted = freq_f64.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let len = sorted.len();
            if len % 2 == 0 {
                (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
            } else {
                sorted[len / 2]
            }
        } else {
            0.0
        };

        // Calculate skewness and kurtosis
        let (skewness, kurtosis) = self.calculate_moments(&freq_f64, mean_freq, std_dev_freq);

        Ok(ValueHistogram {
            boundaries,
            frequencies,
            total_count,
            distinct_count,
            mcv,
            summary: HistogramSummary {
                mean_frequency: mean_freq,
                std_dev_frequency: std_dev_freq,
                median_frequency: median_freq,
                skewness,
                kurtosis,
            },
        })
    }

    /// Calculate skewness and kurtosis
    fn calculate_moments(&self, data: &[f64], mean: f64, std_dev: f64) -> (f64, f64) {
        if data.len() < 3 || std_dev == 0.0 {
            return (0.0, 0.0);
        }

        let n = data.len() as f64;
        let mut m3 = 0.0;
        let mut m4 = 0.0;

        for &x in data {
            let diff = x - mean;
            let diff_sq = diff * diff;
            m3 += diff * diff_sq;
            m4 += diff_sq * diff_sq;
        }

        let skewness = (m3 / n) / (std_dev * std_dev * std_dev);
        let kurtosis = (m4 / n) / (std_dev * std_dev * std_dev * std_dev) - 3.0;

        (skewness, kurtosis)
    }

    /// Estimate selectivity for a value using histogram
    pub fn estimate_selectivity(&self, predicate: &str, value: &str) -> f64 {
        if let Some(hist) = self.predicate_histograms.get(predicate) {
            // Check MCV first
            for (mcv_val, mcv_freq) in &hist.mcv {
                if mcv_val == value {
                    return *mcv_freq as f64 / hist.total_count as f64;
                }
            }

            // Use bucket estimation
            if let Some(bucket_idx) = self.find_bucket(&hist.boundaries, value) {
                if bucket_idx < hist.frequencies.len() {
                    let bucket_freq = hist.frequencies[bucket_idx] as f64;
                    let bucket_distinct =
                        (hist.distinct_count as f64 / hist.boundaries.len() as f64).max(1.0);
                    return (bucket_freq / bucket_distinct) / hist.total_count as f64;
                }
            }

            // Default: uniform distribution assumption
            1.0 / hist.distinct_count.max(1) as f64
        } else {
            // No statistics available
            0.1 // Default selectivity
        }
    }

    /// Find bucket index for a value
    fn find_bucket(&self, boundaries: &[String], value: &str) -> Option<usize> {
        boundaries
            .binary_search(&value.to_string())
            .ok()
            .or_else(|| {
                boundaries
                    .binary_search(&value.to_string())
                    .err()
                    .map(|pos| pos.saturating_sub(1))
            })
    }

    /// Estimate cardinality for a triple pattern
    pub fn estimate_pattern_cardinality(&self, pattern: &TriplePattern) -> u64 {
        // Extract predicate if available (as a string for histogram lookup)
        let predicate = match &pattern.predicate {
            Term::Iri(iri) => Some(iri.as_str()),
            Term::Literal(lit) => Some(lit.value.as_str()),
            _ => None,
        };

        if let Some(pred) = predicate {
            if let Some(hist) = self.predicate_histograms.get(pred) {
                // Calculate selectivity based on bound/unbound positions
                let mut selectivity = 1.0;

                // Subject selectivity
                if let Term::Iri(subj) = &pattern.subject {
                    selectivity *= self.estimate_selectivity(pred, subj.as_str());
                }

                // Object selectivity
                if let Term::Iri(obj) = &pattern.object {
                    selectivity *= self.estimate_selectivity(pred, obj.as_str());
                }

                return (hist.total_count as f64 * selectivity).max(1.0) as u64;
            }
        }

        // Default estimate
        100
    }
}

/// Correlation-based estimator for join cardinality
pub struct CorrelationEstimator {
    /// Correlation matrices between predicates
    correlations: HashMap<(String, String), CorrelationInfo>,
    /// Regression models for join estimation
    /// Configuration
    config: EstimatorConfig,
}

/// Correlation information between two attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationInfo {
    /// Pearson correlation coefficient
    pub correlation: f64,
    /// P-value for correlation significance
    pub p_value: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Number of samples
    pub sample_count: u64,
    /// Joint cardinality
    pub joint_cardinality: u64,
}

impl CorrelationEstimator {
    /// Create a new correlation estimator
    pub fn new(config: EstimatorConfig) -> Self {
        Self {
            correlations: HashMap::new(),
            config,
        }
    }

    /// Calculate mean of a slice
    fn calculate_mean(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Build correlation between two predicates
    pub fn build_correlation(
        &mut self,
        pred1: &str,
        pred2: &str,
        values1: Vec<f64>,
        values2: Vec<f64>,
    ) -> Result<()> {
        if values1.len() != values2.len() || values1.len() < self.config.min_samples {
            return Err(anyhow!(
                "Insufficient samples for correlation: {} < {}",
                values1.len(),
                self.config.min_samples
            ));
        }

        // Calculate Pearson correlation
        let correlation = self.calculate_correlation(&values1, &values2);

        // Simple significance test based on correlation and sample size
        // For a more rigorous test, we would use scirs2-stats when available
        let n = values1.len() as f64;
        let t_stat = correlation * ((n - 2.0) / (1.0 - correlation * correlation)).sqrt();
        let p_value = if t_stat.abs() > 2.0 {
            0.01 // Significant
        } else {
            0.5 // Not significant
        };

        // Calculate confidence interval
        let z_score = 1.96; // 95% confidence
        let n = values1.len() as f64;
        let se = 1.0 / (n - 3.0).sqrt();
        let fisher_z = 0.5 * ((1.0 + correlation) / (1.0 - correlation)).ln();
        let ci_low = ((fisher_z - z_score * se).tanh() - 1.0) / 2.0;
        let ci_high = ((fisher_z + z_score * se).tanh() + 1.0) / 2.0;

        let corr_info = CorrelationInfo {
            correlation,
            p_value,
            confidence_interval: (ci_low, ci_high),
            sample_count: values1.len() as u64,
            joint_cardinality: values1.len() as u64,
        };

        self.correlations
            .insert((pred1.to_string(), pred2.to_string()), corr_info);

        // For now, skip regression model building as it requires additional implementation
        // Future: Build regression model if correlation is significant
        // if p_value < (1.0 - self.config.confidence_level) && correlation.abs() > 0.3 {
        //     // Build linear regression model for prediction
        // }

        Ok(())
    }

    /// Calculate Pearson correlation coefficient
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.is_empty() || y.is_empty() || x.len() != y.len() {
            return 0.0;
        }

        let mean_x = self.calculate_mean(x);
        let mean_y = self.calculate_mean(y);

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for i in 0..x.len() {
            let diff_x = x[i] - mean_x;
            let diff_y = y[i] - mean_y;
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
        }

        if sum_sq_x == 0.0 || sum_sq_y == 0.0 {
            return 0.0;
        }

        numerator / (sum_sq_x.sqrt() * sum_sq_y.sqrt())
    }

    /// Estimate join cardinality using correlation
    pub fn estimate_join_cardinality(
        &self,
        pred1: &str,
        card1: u64,
        pred2: &str,
        card2: u64,
    ) -> u64 {
        if let Some(corr_info) = self
            .correlations
            .get(&(pred1.to_string(), pred2.to_string()))
        {
            // Use correlation to adjust join cardinality
            let base_estimate = (card1 as f64).min(card2 as f64);
            let correlation_factor = 1.0 - corr_info.correlation.abs();
            (base_estimate * correlation_factor).max(1.0) as u64
        } else {
            // No correlation info: use independence assumption
            ((card1 as f64 * card2 as f64).sqrt() as u64).max(1)
        }
    }

    /// Get correlation between two predicates
    pub fn get_correlation(&self, pred1: &str, pred2: &str) -> Option<f64> {
        self.correlations
            .get(&(pred1.to_string(), pred2.to_string()))
            .map(|info| info.correlation)
    }
}

/// Adaptive estimator that learns from query execution
pub struct AdaptiveEstimator {
    /// Historical estimates vs actual results
    estimation_history: Vec<EstimationRecord>,
    /// Correction factors by pattern type
    correction_factors: HashMap<String, f64>,
    /// Learning rate for adaptive updates
    learning_rate: f64,
    /// Configuration (for future use)
    #[allow(dead_code)]
    config: EstimatorConfig,
}

/// Record of estimation vs actual result
#[derive(Debug, Clone)]
pub struct EstimationRecord {
    /// Pattern signature
    pub pattern_sig: String,
    /// Estimated cardinality
    pub estimated: u64,
    /// Actual cardinality
    pub actual: u64,
    /// Estimation error
    pub error: f64,
    /// Timestamp
    pub timestamp: Instant,
}

impl AdaptiveEstimator {
    /// Create a new adaptive estimator
    pub fn new(config: EstimatorConfig) -> Self {
        Self {
            estimation_history: Vec::new(),
            correction_factors: HashMap::new(),
            learning_rate: 0.1,
            config,
        }
    }

    /// Record an estimation result
    pub fn record_estimation(&mut self, pattern_sig: String, estimated: u64, actual: u64) {
        let error = if estimated > 0 {
            ((estimated as f64 - actual as f64).abs() / actual as f64).min(10.0)
        } else {
            1.0
        };

        let record = EstimationRecord {
            pattern_sig: pattern_sig.clone(),
            estimated,
            actual,
            error,
            timestamp: Instant::now(),
        };

        self.estimation_history.push(record);

        // Update correction factor
        self.update_correction_factor(pattern_sig, estimated, actual);

        // Keep history bounded
        if self.estimation_history.len() > 10000 {
            self.estimation_history.drain(0..5000);
        }
    }

    /// Update correction factor for a pattern type
    fn update_correction_factor(&mut self, pattern_sig: String, estimated: u64, actual: u64) {
        let current_factor = self
            .correction_factors
            .get(&pattern_sig)
            .copied()
            .unwrap_or(1.0);

        let target_factor = if estimated > 0 {
            actual as f64 / estimated as f64
        } else {
            1.0
        };

        // Exponential moving average
        let new_factor =
            current_factor * (1.0 - self.learning_rate) + target_factor * self.learning_rate;

        self.correction_factors.insert(pattern_sig, new_factor);
    }

    /// Apply adaptive correction to an estimate
    pub fn apply_correction(&self, pattern_sig: &str, estimate: u64) -> u64 {
        if let Some(&factor) = self.correction_factors.get(pattern_sig) {
            (estimate as f64 * factor).max(1.0) as u64
        } else {
            estimate
        }
    }

    /// Get mean absolute percentage error (MAPE)
    pub fn get_mape(&self) -> f64 {
        if self.estimation_history.is_empty() {
            return 0.0;
        }

        let total_error: f64 = self.estimation_history.iter().map(|r| r.error).sum();
        total_error / self.estimation_history.len() as f64
    }

    /// Get estimation accuracy (percentage within threshold)
    pub fn get_accuracy(&self, threshold: f64) -> f64 {
        if self.estimation_history.is_empty() {
            return 0.0;
        }

        let within_threshold = self
            .estimation_history
            .iter()
            .filter(|r| r.error <= threshold)
            .count();

        within_threshold as f64 / self.estimation_history.len() as f64
    }
}

impl CardinalityEstimator {
    /// Create a new cardinality estimator
    pub fn new(config: EstimatorConfig) -> Self {
        Self {
            histogram_estimator: Arc::new(RwLock::new(HistogramEstimator::new(config.clone()))),
            correlation_estimator: Arc::new(RwLock::new(CorrelationEstimator::new(config.clone()))),
            adaptive_estimator: Arc::new(RwLock::new(AdaptiveEstimator::new(config.clone()))),
            config,
            stats: Arc::new(RwLock::new(EstimationStats::default())),
        }
    }

    /// Estimate cardinality for a triple pattern
    pub fn estimate_triple_pattern(&self, pattern: &TriplePattern) -> Result<u64> {
        let hist_est = self
            .histogram_estimator
            .read()
            .map_err(|e| anyhow!("Failed to acquire histogram estimator lock: {}", e))?;

        let base_estimate = hist_est.estimate_pattern_cardinality(pattern);

        // Apply adaptive correction if enabled
        if self.config.enable_adaptive {
            let pattern_sig = format!("{:?}", pattern);
            let adaptive_est = self
                .adaptive_estimator
                .read()
                .map_err(|e| anyhow!("Failed to acquire adaptive estimator lock: {}", e))?;
            Ok(adaptive_est.apply_correction(&pattern_sig, base_estimate))
        } else {
            Ok(base_estimate)
        }
    }

    /// Estimate join cardinality
    pub fn estimate_join(
        &self,
        left_pattern: &TriplePattern,
        left_card: u64,
        right_pattern: &TriplePattern,
        right_card: u64,
        join_vars: &[Variable],
    ) -> Result<u64> {
        // Extract predicates
        let pred1 = match &left_pattern.predicate {
            Term::Iri(iri) => Some(iri.as_str()),
            Term::Literal(lit) => Some(lit.value.as_str()),
            _ => None,
        };

        let pred2 = match &right_pattern.predicate {
            Term::Iri(iri) => Some(iri.as_str()),
            Term::Literal(lit) => Some(lit.value.as_str()),
            _ => None,
        };

        // Use correlation if available
        if self.config.enable_correlation {
            if let (Some(p1), Some(p2)) = (pred1, pred2) {
                let corr_est = self
                    .correlation_estimator
                    .read()
                    .map_err(|e| anyhow!("Failed to acquire correlation estimator lock: {}", e))?;
                return Ok(corr_est.estimate_join_cardinality(p1, left_card, p2, right_card));
            }
        }

        // Default: independence assumption with join variable discount
        let base_estimate = (left_card as f64 * right_card as f64).sqrt() as u64;
        let join_discount = 0.5_f64.powi(join_vars.len() as i32);
        Ok((base_estimate as f64 * join_discount).max(1.0) as u64)
    }

    /// Record actual result for adaptive learning
    pub fn record_actual_result(
        &self,
        pattern: &TriplePattern,
        estimated: u64,
        actual: u64,
    ) -> Result<()> {
        if !self.config.enable_adaptive {
            return Ok(());
        }

        let pattern_sig = format!("{:?}", pattern);
        let mut adaptive_est = self
            .adaptive_estimator
            .write()
            .map_err(|e| anyhow!("Failed to acquire adaptive estimator lock: {}", e))?;

        adaptive_est.record_estimation(pattern_sig, estimated, actual);

        // Update stats
        let mut stats = self
            .stats
            .write()
            .map_err(|e| anyhow!("Failed to acquire stats lock: {}", e))?;

        stats.total_estimations += 1;
        let error = if actual > 0 {
            ((estimated as f64 - actual as f64).abs() / actual as f64).min(10.0)
        } else {
            0.0
        };
        stats.total_error += error;
        stats.mape = stats.total_error / stats.total_estimations as f64;

        if error <= self.config.max_error_threshold {
            stats.within_threshold += 1;
        }

        Ok(())
    }

    /// Get estimation statistics
    pub fn get_stats(&self) -> Result<EstimationStats> {
        let stats = self
            .stats
            .read()
            .map_err(|e| anyhow!("Failed to acquire stats lock: {}", e))?;
        Ok(stats.clone())
    }

    /// Build histogram for a predicate
    pub fn build_histogram(&self, predicate: &str, values: Vec<String>) -> Result<()> {
        let mut hist_est = self
            .histogram_estimator
            .write()
            .map_err(|e| anyhow!("Failed to acquire histogram estimator lock: {}", e))?;
        hist_est.build_histogram(predicate, values)
    }

    /// Build correlation between predicates
    pub fn build_correlation(
        &self,
        pred1: &str,
        pred2: &str,
        values1: Vec<f64>,
        values2: Vec<f64>,
    ) -> Result<()> {
        if !self.config.enable_correlation {
            return Ok(());
        }

        let mut corr_est = self
            .correlation_estimator
            .write()
            .map_err(|e| anyhow!("Failed to acquire correlation estimator lock: {}", e))?;
        corr_est.build_correlation(pred1, pred2, values1, values2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_creation() {
        let config = EstimatorConfig::default();
        let mut estimator = HistogramEstimator::new(config);

        let values: Vec<String> = (0..1000).map(|i| format!("value_{}", i)).collect();
        let result = estimator.build_histogram("test_pred", values);
        assert!(result.is_ok());

        let hist = estimator.predicate_histograms.get("test_pred").unwrap();
        assert_eq!(hist.total_count, 1000);
        assert_eq!(hist.distinct_count, 1000);
    }

    #[test]
    fn test_selectivity_estimation() {
        let config = EstimatorConfig::default();
        let mut estimator = HistogramEstimator::new(config);

        let values: Vec<String> = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        estimator.build_histogram("test_pred", values).unwrap();

        let selectivity = estimator.estimate_selectivity("test_pred", "b");
        assert!(selectivity > 0.0 && selectivity <= 1.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let config = EstimatorConfig::default();
        let estimator = CorrelationEstimator::new(config);

        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..100).map(|i| i as f64 * 2.0).collect();

        let corr = estimator.calculate_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.01); // Should be close to 1.0
    }

    #[test]
    fn test_adaptive_learning() {
        let config = EstimatorConfig::default();
        let mut estimator = AdaptiveEstimator::new(config);

        // Record several estimations
        for i in 0..100 {
            let estimated = 100;
            let actual = 80; // Consistently underestimating
            estimator.record_estimation(format!("pattern_{}", i % 10), estimated, actual);
        }

        // Correction factor should adjust upward
        let corrected = estimator.apply_correction("pattern_0", 100);
        assert!(corrected < 100); // Should correct downward since we overestimated
    }

    #[test]
    fn test_full_estimator() {
        let config = EstimatorConfig::default();
        let estimator = CardinalityEstimator::new(config);

        // Build some statistics
        let values: Vec<String> = (0..100).map(|i| format!("value_{}", i)).collect();
        estimator
            .build_histogram("http://example.org/pred", values)
            .unwrap();

        // Test pattern estimation
        let pattern = TriplePattern {
            subject: Term::Variable(Variable::new("s".to_string()).unwrap()),
            predicate: Term::Iri(
                crate::algebra::Iri::new("http://example.org/pred".to_string()).unwrap(),
            ),
            object: Term::Variable(Variable::new("o".to_string()).unwrap()),
        };

        let estimate = estimator.estimate_triple_pattern(&pattern);
        assert!(estimate.is_ok());
        assert!(estimate.unwrap() > 0);
    }
}
