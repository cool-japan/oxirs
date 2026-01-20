//! # ModelAnalytics - compute_statistical_metrics_group Methods
//!
//! This module contains method implementations for `ModelAnalytics`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::modelanalytics_type::ModelAnalytics;
use crate::analytics::{QualityTest, Severity, StatisticalAnomaly, StatisticalMetrics};
use scirs2_core::ndarray_ext::Array1;
use scirs2_stats::{iqr, kurtosis, mean_abs_deviation, median_abs_deviation, skew};
use std::collections::{HashMap, HashSet};

impl ModelAnalytics {
    /// Compute advanced statistical metrics for model properties
    ///
    /// Uses scirs2-stats to provide comprehensive statistical analysis including
    /// dispersion measures, shape statistics, and robustness metrics.
    ///
    /// # Returns
    ///
    /// StatisticalMetrics containing advanced statistics about the model
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use oxirs_samm::analytics::ModelAnalytics;
    /// use oxirs_samm::metamodel::Aspect;
    ///
    /// let aspect = Aspect::new("urn:samm:org.example:1.0.0#MyAspect".to_string());
    /// let analytics = ModelAnalytics::analyze(&aspect);
    /// let stats = analytics.compute_statistical_metrics();
    ///
    /// println!("Coefficient of Variation: {:.2}%", stats.coefficient_variation * 100.0);
    /// println!("Median Absolute Deviation: {:.2}", stats.median_abs_deviation);
    /// ```
    pub fn compute_statistical_metrics(&self) -> StatisticalMetrics {
        let prop_count = self.distributions.property_distribution.mean;
        let data = Array1::from_vec(vec![
            prop_count,
            self.complexity_assessment.structural,
            self.complexity_assessment.cognitive,
            self.complexity_assessment.coupling * 100.0,
        ]);
        let n = data.len() as f64;
        let sum: f64 = data.iter().sum();
        let mean_value = sum / n;
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_value = if sorted_data.is_empty() {
            0.0
        } else {
            sorted_data[sorted_data.len() / 2]
        };
        let sq_diff_sum: f64 = data.iter().map(|x| (x - mean_value).powi(2)).sum();
        let var_value = sq_diff_sum / (n - 1.0);
        let std_value = var_value.sqrt();
        let data_view = data.view();
        let mad = mean_abs_deviation(&data_view, None).unwrap_or(0.0);
        let median_ad = median_abs_deviation(&data_view, None, Some(1.4826)).unwrap_or(0.0);
        let iqr_value = iqr(&data_view, None).unwrap_or(0.0);
        let cv = if mean_value.abs() > 0.001 {
            std_value / mean_value.abs()
        } else {
            0.0
        };
        let skewness = skew(&data_view, false, None).unwrap_or(0.0);
        let kurt = kurtosis(&data_view, true, false, None).unwrap_or(0.0);
        StatisticalMetrics {
            mean: mean_value,
            median: median_value,
            std_dev: std_value,
            variance: var_value,
            mean_abs_deviation: mad,
            median_abs_deviation: median_ad,
            interquartile_range: iqr_value,
            coefficient_variation: cv,
            skewness,
            kurtosis: kurt,
        }
    }
    /// Detect statistical anomalies using robust methods
    ///
    /// Uses median absolute deviation (MAD) for robust outlier detection,
    /// which is less sensitive to outliers than standard deviation.
    ///
    /// # Returns
    ///
    /// Vector of StatisticalAnomaly indicating unusual patterns
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use oxirs_samm::analytics::ModelAnalytics;
    ///
    /// let analytics = ModelAnalytics::analyze(&aspect);
    /// let anomalies = analytics.detect_statistical_anomalies();
    ///
    /// for anomaly in anomalies {
    ///     println!("⚠ {}: {} (score: {:.2})",
    ///              anomaly.metric_name, anomaly.description, anomaly.deviation_score);
    /// }
    /// ```
    pub fn detect_statistical_anomalies(&self) -> Vec<StatisticalAnomaly> {
        let mut anomalies = Vec::new();
        let stats = self.compute_statistical_metrics();
        if stats.coefficient_variation > 1.0 {
            anomalies.push(StatisticalAnomaly {
                metric_name: "Coefficient of Variation".to_string(),
                description: format!(
                    "High variability detected: {:.1}% (threshold: 100%)",
                    stats.coefficient_variation * 100.0
                ),
                deviation_score: stats.coefficient_variation,
                severity: if stats.coefficient_variation > 2.0 {
                    Severity::Error
                } else {
                    Severity::Warning
                },
            });
        }
        if stats.skewness.abs() > 2.0 {
            anomalies.push(StatisticalAnomaly {
                metric_name: "Skewness".to_string(),
                description: format!(
                    "Highly skewed distribution: {:.2} (threshold: ±2.0)",
                    stats.skewness
                ),
                deviation_score: stats.skewness.abs(),
                severity: Severity::Info,
            });
        }
        if stats.kurtosis.abs() > 3.0 {
            anomalies.push(StatisticalAnomaly {
                metric_name: "Kurtosis".to_string(),
                description: format!(
                    "Heavy-tailed distribution: {:.2} (threshold: ±3.0)",
                    stats.kurtosis
                ),
                deviation_score: stats.kurtosis.abs(),
                severity: Severity::Info,
            });
        }
        if stats.median_abs_deviation > stats.median * 0.5 {
            anomalies.push(StatisticalAnomaly {
                metric_name: "Median Absolute Deviation".to_string(),
                description: format!(
                    "High spread around median: {:.2} (>50% of median)",
                    stats.median_abs_deviation
                ),
                deviation_score: stats.median_abs_deviation / stats.median.max(1.0),
                severity: Severity::Warning,
            });
        }
        anomalies
    }
    /// Assess model quality using statistical hypothesis testing
    ///
    /// Applies statistical tests to determine if the model meets quality thresholds.
    /// Uses robust statistical methods from scirs2-stats.
    ///
    /// # Returns
    ///
    /// QualityTest results with statistical confidence levels
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use oxirs_samm::analytics::ModelAnalytics;
    ///
    /// let analytics = ModelAnalytics::analyze(&aspect);
    /// let test = analytics.statistical_quality_test();
    ///
    /// if test.passes_threshold {
    ///     println!("✓ Model meets quality standards (confidence: {:.1}%)",
    ///              test.confidence_level * 100.0);
    /// }
    /// ```
    pub fn statistical_quality_test(&self) -> QualityTest {
        let stats = self.compute_statistical_metrics();
        let cv_ok = stats.coefficient_variation < 0.8;
        let skew_ok = stats.skewness.abs() < 1.5;
        let score_ok = self.quality_score > stats.median;
        let tests_passed = [cv_ok, skew_ok, score_ok].iter().filter(|&&x| x).count();
        let confidence = tests_passed as f64 / 3.0;
        QualityTest {
            passes_threshold: tests_passed >= 2,
            confidence_level: confidence,
            cv_check: cv_ok,
            skewness_check: skew_ok,
            score_check: score_ok,
            details: format!(
                "Passed {}/3 tests: CV={:.1}%, Skewness={:.2}, Score={:.1}",
                tests_passed,
                stats.coefficient_variation * 100.0,
                stats.skewness,
                self.quality_score
            ),
        }
    }
}
