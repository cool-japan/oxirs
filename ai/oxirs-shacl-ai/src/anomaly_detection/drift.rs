//! Data drift detection for monitoring distribution changes over time

use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::types::{Anomaly, AnomalyScore, AnomalyType, DataDistribution};
use crate::{Result, ShaclAiError};

/// Drift detection types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftType {
    /// Sudden drift (abrupt change)
    Sudden,
    /// Gradual drift (slow change over time)
    Gradual,
    /// Incremental drift (step-by-step changes)
    Incremental,
    /// Recurring drift (seasonal patterns)
    Recurring,
}

/// Drift detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftResult {
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Type of drift
    pub drift_type: Option<DriftType>,
    /// Drift score (0.0 = no drift, 1.0 = significant drift)
    pub drift_score: f64,
    /// Confidence in drift detection
    pub confidence: f64,
    /// Reference distribution
    pub reference_distribution: DataDistribution,
    /// Current distribution
    pub current_distribution: DataDistribution,
    /// Statistical tests results
    pub test_results: Vec<StatisticalTestResult>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResult {
    /// Test name
    pub test_name: String,
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Significance level
    pub alpha: f64,
    /// Whether test is significant
    pub is_significant: bool,
}

/// Drift detector with sliding window
pub struct DriftDetector {
    window_size: usize,
    reference_data: VecDeque<f64>,
    significance_level: f64,
    drift_threshold: f64,
}

impl DriftDetector {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            reference_data: VecDeque::with_capacity(window_size),
            significance_level: 0.05,
            drift_threshold: 0.7,
        }
    }

    pub fn with_significance_level(mut self, alpha: f64) -> Self {
        self.significance_level = alpha;
        self
    }

    pub fn with_drift_threshold(mut self, threshold: f64) -> Self {
        self.drift_threshold = threshold;
        self
    }

    /// Update reference window with new data
    pub fn update_reference(&mut self, data: &[f64]) {
        for &value in data {
            if self.reference_data.len() >= self.window_size {
                self.reference_data.pop_front();
            }
            self.reference_data.push_back(value);
        }
    }

    /// Detect drift between reference and current data
    pub fn detect_drift(&self, current_data: &Array1<f64>) -> Result<DriftResult> {
        if self.reference_data.is_empty() {
            return Err(ShaclAiError::Analytics(
                "No reference data available for drift detection".to_string(),
            ));
        }

        if current_data.len() < 10 {
            return Err(ShaclAiError::Analytics(
                "Insufficient current data for drift detection".to_string(),
            ));
        }

        // Convert reference data to Array1
        let reference_array = Array1::from_vec(self.reference_data.iter().copied().collect());

        // Calculate distributions
        let ref_dist = self.calculate_distribution(&reference_array)?;
        let curr_dist = self.calculate_distribution(current_data)?;

        // Perform statistical tests
        let mut test_results = Vec::new();

        // Kolmogorov-Smirnov test
        let ks_test = self.kolmogorov_smirnov_test(&reference_array, current_data)?;
        test_results.push(ks_test);

        // Population Stability Index (PSI)
        let psi_test = self.calculate_psi(&reference_array, current_data)?;
        test_results.push(psi_test);

        // Jensen-Shannon divergence
        let js_test = self.jensen_shannon_divergence(&reference_array, current_data)?;
        test_results.push(js_test);

        // Aggregate results
        let significant_tests = test_results.iter().filter(|t| t.is_significant).count();
        let drift_score = significant_tests as f64 / test_results.len() as f64;
        let drift_detected = drift_score >= self.drift_threshold;

        // Determine drift type
        let drift_type = if drift_detected {
            Some(self.classify_drift_type(&ref_dist, &curr_dist))
        } else {
            None
        };

        // Generate recommendations
        let recommendations = if drift_detected {
            vec![
                "Consider retraining models on current data".to_string(),
                "Investigate root cause of distribution change".to_string(),
                "Update data validation rules".to_string(),
                "Monitor for continued drift".to_string(),
            ]
        } else {
            vec!["Continue monitoring for drift".to_string()]
        };

        Ok(DriftResult {
            drift_detected,
            drift_type,
            drift_score,
            confidence: drift_score,
            reference_distribution: ref_dist,
            current_distribution: curr_dist,
            test_results,
            recommendations,
        })
    }

    fn calculate_distribution(&self, data: &Array1<f64>) -> Result<DataDistribution> {
        let n = data.len();
        if n == 0 {
            return Err(ShaclAiError::Analytics(
                "Cannot calculate distribution for empty data".to_string(),
            ));
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

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

    /// Kolmogorov-Smirnov test
    fn kolmogorov_smirnov_test(
        &self,
        ref_data: &Array1<f64>,
        curr_data: &Array1<f64>,
    ) -> Result<StatisticalTestResult> {
        // Sort both datasets
        let mut ref_sorted = ref_data.to_vec();
        let mut curr_sorted = curr_data.to_vec();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        curr_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate empirical CDFs and max difference
        let mut max_diff = 0.0_f64;
        let mut i = 0;
        let mut j = 0;

        while i < ref_sorted.len() && j < curr_sorted.len() {
            let ref_cdf = (i + 1) as f64 / ref_sorted.len() as f64;
            let curr_cdf = (j + 1) as f64 / curr_sorted.len() as f64;

            max_diff = max_diff.max((ref_cdf - curr_cdf).abs());

            if ref_sorted[i] < curr_sorted[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        // Calculate critical value
        let n1 = ref_sorted.len() as f64;
        let n2 = curr_sorted.len() as f64;
        let critical_value =
            (-(2.0 * self.significance_level).ln() / 2.0).sqrt() * ((n1 + n2) / (n1 * n2)).sqrt();

        Ok(StatisticalTestResult {
            test_name: "Kolmogorov-Smirnov".to_string(),
            statistic: max_diff,
            p_value: 0.0, // Simplified - proper calculation requires complex formula
            alpha: self.significance_level,
            is_significant: max_diff > critical_value,
        })
    }

    /// Population Stability Index
    fn calculate_psi(
        &self,
        ref_data: &Array1<f64>,
        curr_data: &Array1<f64>,
    ) -> Result<StatisticalTestResult> {
        let n_bins = 10;

        // Calculate bin edges
        let all_data: Vec<f64> = ref_data.iter().chain(curr_data.iter()).copied().collect();
        let mut sorted_all = all_data.clone();
        sorted_all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = sorted_all[0];
        let max_val = sorted_all[sorted_all.len() - 1];
        let bin_width = (max_val - min_val) / n_bins as f64;

        // Calculate bin counts
        let mut ref_counts = vec![0.0; n_bins];
        let mut curr_counts = vec![0.0; n_bins];

        for &val in ref_data.iter() {
            let bin = ((val - min_val) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            ref_counts[bin] += 1.0;
        }

        for &val in curr_data.iter() {
            let bin = ((val - min_val) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            curr_counts[bin] += 1.0;
        }

        // Convert to proportions
        let ref_total = ref_counts.iter().sum::<f64>();
        let curr_total = curr_counts.iter().sum::<f64>();

        for count in &mut ref_counts {
            *count /= ref_total;
        }
        for count in &mut curr_counts {
            *count /= curr_total;
        }

        // Calculate PSI
        let psi: f64 = ref_counts
            .iter()
            .zip(curr_counts.iter())
            .map(|(&ref_prop, &curr_prop)| {
                let ref_prop = ref_prop.max(1e-10);
                let curr_prop = curr_prop.max(1e-10);
                (curr_prop - ref_prop) * (curr_prop / ref_prop).ln()
            })
            .sum();

        // PSI thresholds: <0.1 = stable, 0.1-0.25 = moderate drift, >0.25 = significant drift
        Ok(StatisticalTestResult {
            test_name: "Population Stability Index".to_string(),
            statistic: psi,
            p_value: 0.0,
            alpha: 0.25, // Threshold for significant drift
            is_significant: psi > 0.25,
        })
    }

    /// Jensen-Shannon divergence
    fn jensen_shannon_divergence(
        &self,
        ref_data: &Array1<f64>,
        curr_data: &Array1<f64>,
    ) -> Result<StatisticalTestResult> {
        let n_bins = 10;

        // Create histograms
        let all_data: Vec<f64> = ref_data.iter().chain(curr_data.iter()).copied().collect();
        let mut sorted_all = all_data.clone();
        sorted_all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = sorted_all[0];
        let max_val = sorted_all[sorted_all.len() - 1];
        let bin_width = (max_val - min_val) / n_bins as f64;

        let mut ref_hist = vec![0.0; n_bins];
        let mut curr_hist = vec![0.0; n_bins];

        for &val in ref_data.iter() {
            let bin = ((val - min_val) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            ref_hist[bin] += 1.0;
        }

        for &val in curr_data.iter() {
            let bin = ((val - min_val) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            curr_hist[bin] += 1.0;
        }

        // Normalize
        let ref_sum: f64 = ref_hist.iter().sum();
        let curr_sum: f64 = curr_hist.iter().sum();

        for val in &mut ref_hist {
            *val /= ref_sum;
        }
        for val in &mut curr_hist {
            *val /= curr_sum;
        }

        // Calculate JSD
        let mut m = vec![0.0; n_bins];
        for i in 0..n_bins {
            m[i] = (ref_hist[i] + curr_hist[i]) / 2.0;
        }

        let kl_pm = self.kl_divergence(&ref_hist, &m);
        let kl_qm = self.kl_divergence(&curr_hist, &m);
        let jsd = (kl_pm + kl_qm) / 2.0;

        Ok(StatisticalTestResult {
            test_name: "Jensen-Shannon Divergence".to_string(),
            statistic: jsd,
            p_value: 0.0,
            alpha: 0.1, // Threshold for significant divergence
            is_significant: jsd > 0.1,
        })
    }

    fn kl_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| {
                if pi > 1e-10 && qi > 1e-10 {
                    pi * (pi / qi).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    fn classify_drift_type(
        &self,
        ref_dist: &DataDistribution,
        curr_dist: &DataDistribution,
    ) -> DriftType {
        // Simplified drift classification based on distribution changes
        let mean_diff = (curr_dist.mean - ref_dist.mean).abs() / ref_dist.std_dev.max(1e-10);
        let std_ratio = curr_dist.std_dev / ref_dist.std_dev.max(1e-10);

        if mean_diff > 2.0 && (std_ratio - 1.0).abs() > 0.5 {
            DriftType::Sudden
        } else if mean_diff > 1.0 {
            DriftType::Incremental
        } else if (std_ratio - 1.0).abs() > 0.5 {
            DriftType::Gradual
        } else {
            DriftType::Recurring
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_detector_creation() {
        let detector = DriftDetector::new(100);
        assert_eq!(detector.window_size, 100);
        assert_eq!(detector.significance_level, 0.05);
    }

    #[test]
    fn test_update_reference() {
        let mut detector = DriftDetector::new(5);
        detector.update_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(detector.reference_data.len(), 5);
    }

    #[test]
    fn test_drift_detection_no_drift() {
        let mut detector = DriftDetector::new(50);
        let reference: Vec<f64> = (0..50).map(|i| i as f64).collect();
        detector.update_reference(&reference);

        let current = Array1::from_vec((0..50).map(|i| i as f64 + 0.1).collect());
        let result = detector.detect_drift(&current).unwrap();

        // Small change should not trigger drift
        assert!(result.drift_score < 1.0);
    }

    #[test]
    fn test_drift_detection_with_drift() {
        let mut detector = DriftDetector::new(50);
        let reference: Vec<f64> = (0..50).map(|i| i as f64).collect();
        detector.update_reference(&reference);

        let current = Array1::from_vec((0..50).map(|i| i as f64 + 50.0).collect());
        let result = detector.detect_drift(&current).unwrap();

        // Large shift should trigger drift
        assert!(result.drift_detected);
        assert!(result.drift_score > 0.0);
    }
}
