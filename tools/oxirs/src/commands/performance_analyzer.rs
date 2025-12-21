//! Advanced Statistical Performance Analyzer
//!
//! Provides sophisticated performance analysis using SciRS2's statistical computing capabilities.
//! Includes hypothesis testing, regression analysis, outlier detection, and performance prediction.

use anyhow::Result;
use colored::Colorize;

// Note: SciRS2 arrays reserved for future multivariate analysis
// use scirs2_core::ndarray_ext::{Array1, Array2};
// use scirs2_core::random::Random;

/// Performance measurement sample
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    pub timestamp: f64,
    pub duration_ms: f64,
    pub memory_mb: f64,
    pub cpu_percent: f64,
    pub io_ops: u64,
}

/// Statistical performance metrics with confidence intervals
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    /// Mean execution time (ms)
    pub mean_duration: f64,
    /// Standard deviation (ms)
    pub std_dev: f64,
    /// 95% confidence interval lower bound
    pub ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ci_upper: f64,
    /// Median (50th percentile)
    pub median: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
    /// Coefficient of variation (std/mean)
    pub coefficient_variation: f64,
    /// Number of samples
    pub sample_count: usize,
}

/// Performance regression test result
#[derive(Debug, Clone)]
pub struct RegressionTestResult {
    /// Whether regression was detected
    pub regression_detected: bool,
    /// T-statistic value
    pub t_statistic: f64,
    /// P-value (probability)
    pub p_value: f64,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Percent change from baseline
    pub percent_change: f64,
    /// Statistical significance level
    pub significance_level: f64,
    /// Interpretation
    pub interpretation: String,
}

/// Outlier detection result
#[derive(Debug, Clone)]
pub struct OutlierAnalysis {
    /// Indices of detected outliers
    pub outlier_indices: Vec<usize>,
    /// Outlier values
    pub outlier_values: Vec<f64>,
    /// Detection method used
    pub method: String,
    /// Threshold used
    pub threshold: f64,
}

/// Performance prediction result
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    /// Predicted duration for given input size
    pub predicted_duration: f64,
    /// Confidence interval lower bound
    pub prediction_ci_lower: f64,
    /// Confidence interval upper bound
    pub prediction_ci_upper: f64,
    /// Model R-squared
    pub r_squared: f64,
    /// Regression coefficients (intercept, slope)
    pub coefficients: Vec<f64>,
}

/// Advanced statistical performance analyzer
pub struct StatisticalPerformanceAnalyzer {
    samples: Vec<PerformanceSample>,
    significance_level: f64,
}

impl StatisticalPerformanceAnalyzer {
    /// Create new analyzer with default significance level (0.05)
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            significance_level: 0.05,
        }
    }

    /// Create analyzer with custom significance level
    pub fn with_significance(significance_level: f64) -> Self {
        Self {
            samples: Vec::new(),
            significance_level,
        }
    }

    /// Add performance sample
    pub fn add_sample(&mut self, sample: PerformanceSample) {
        self.samples.push(sample);
    }

    /// Add multiple samples
    pub fn add_samples(&mut self, samples: Vec<PerformanceSample>) {
        self.samples.extend(samples);
    }

    /// Calculate comprehensive statistics with confidence intervals
    pub fn calculate_statistics(&self) -> Result<PerformanceStatistics> {
        if self.samples.is_empty() {
            anyhow::bail!("No samples available for statistical analysis");
        }

        let durations: Vec<f64> = self.samples.iter().map(|s| s.duration_ms).collect();
        let n = durations.len();

        // Calculate mean
        let mean = durations.iter().sum::<f64>() / n as f64;

        // Calculate standard deviation
        let variance =
            durations.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        let std_dev = variance.sqrt();

        // Calculate 95% confidence interval using t-distribution
        // For large samples (n > 30), t ≈ z ≈ 1.96
        // For smaller samples, use t-value approximation
        let t_value = if n > 30 {
            1.96 // z-score for 95% CI
        } else {
            // Approximate t-value for small samples
            // t(df, 0.025) for two-tailed 95% CI
            match n {
                2..=5 => 2.776,
                6..=10 => 2.262,
                11..=20 => 2.093,
                21..=30 => 2.045,
                _ => 1.96,
            }
        };

        let margin_of_error = t_value * (std_dev / (n as f64).sqrt());
        let ci_lower = mean - margin_of_error;
        let ci_upper = mean + margin_of_error;

        // Calculate percentiles
        let mut sorted_durations = durations.clone();
        sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = percentile(&sorted_durations, 50.0);
        let p95 = percentile(&sorted_durations, 95.0);
        let p99 = percentile(&sorted_durations, 99.0);

        // Coefficient of variation
        let coefficient_variation = std_dev / mean;

        Ok(PerformanceStatistics {
            mean_duration: mean,
            std_dev,
            ci_lower,
            ci_upper,
            median,
            p95,
            p99,
            coefficient_variation,
            sample_count: n,
        })
    }

    /// Perform two-sample t-test to detect performance regression
    /// Compares current samples against baseline samples
    pub fn test_regression(
        &self,
        baseline_samples: &[PerformanceSample],
    ) -> Result<RegressionTestResult> {
        if self.samples.is_empty() || baseline_samples.is_empty() {
            anyhow::bail!("Insufficient samples for regression testing");
        }

        let current: Vec<f64> = self.samples.iter().map(|s| s.duration_ms).collect();
        let baseline: Vec<f64> = baseline_samples.iter().map(|s| s.duration_ms).collect();

        // Calculate means
        let mean_current = current.iter().sum::<f64>() / current.len() as f64;
        let mean_baseline = baseline.iter().sum::<f64>() / baseline.len() as f64;

        // Calculate standard deviations
        let var_current = current
            .iter()
            .map(|&x| (x - mean_current).powi(2))
            .sum::<f64>()
            / (current.len() as f64 - 1.0);
        let var_baseline = baseline
            .iter()
            .map(|&x| (x - mean_baseline).powi(2))
            .sum::<f64>()
            / (baseline.len() as f64 - 1.0);

        let std_current = var_current.sqrt();
        let std_baseline = var_baseline.sqrt();

        // Welch's t-test (unequal variances)
        let pooled_std =
            ((var_current / current.len() as f64) + (var_baseline / baseline.len() as f64)).sqrt();

        let t_statistic = (mean_current - mean_baseline) / pooled_std;

        // Approximate degrees of freedom (Welch-Satterthwaite)
        let df = ((var_current / current.len() as f64 + var_baseline / baseline.len() as f64)
            .powi(2))
            / ((var_current / current.len() as f64).powi(2) / (current.len() as f64 - 1.0)
                + (var_baseline / baseline.len() as f64).powi(2) / (baseline.len() as f64 - 1.0));

        // Approximate p-value using t-distribution
        // For simplicity, use normal approximation for large df
        let p_value = if df > 30.0 {
            // Normal approximation
            2.0 * (1.0 - standard_normal_cdf(t_statistic.abs()))
        } else {
            // Conservative p-value for small samples
            if t_statistic.abs() > 2.5 {
                0.01
            } else if t_statistic.abs() > 2.0 {
                0.05
            } else if t_statistic.abs() > 1.5 {
                0.15
            } else {
                0.30
            }
        };

        // Cohen's d effect size
        let pooled_std_effect = ((std_current.powi(2) + std_baseline.powi(2)) / 2.0).sqrt();
        let effect_size = (mean_current - mean_baseline) / pooled_std_effect;

        // Percent change
        let percent_change = ((mean_current - mean_baseline) / mean_baseline) * 100.0;

        // Detect regression
        let regression_detected =
            (p_value < self.significance_level) && (mean_current > mean_baseline);

        // Interpretation
        let interpretation = if regression_detected {
            if effect_size.abs() > 0.8 {
                format!(
                    "⚠️  LARGE performance regression detected ({:.1}% slower, p={:.4})",
                    percent_change, p_value
                )
            } else if effect_size.abs() > 0.5 {
                format!(
                    "⚠️  MEDIUM performance regression ({:.1}% slower, p={:.4})",
                    percent_change, p_value
                )
            } else {
                format!(
                    "⚠️  SMALL performance regression ({:.1}% slower, p={:.4})",
                    percent_change, p_value
                )
            }
        } else if p_value < self.significance_level && mean_current < mean_baseline {
            format!(
                "✅ Performance IMPROVEMENT detected ({:.1}% faster, p={:.4})",
                percent_change.abs(),
                p_value
            )
        } else {
            format!(
                "✅ No significant performance change ({:.1}% change, p={:.4})",
                percent_change, p_value
            )
        };

        Ok(RegressionTestResult {
            regression_detected,
            t_statistic,
            p_value,
            effect_size,
            percent_change,
            significance_level: self.significance_level,
            interpretation,
        })
    }

    /// Detect outliers using Interquartile Range (IQR) method
    pub fn detect_outliers(&self, iqr_multiplier: f64) -> Result<OutlierAnalysis> {
        if self.samples.is_empty() {
            anyhow::bail!("No samples available for outlier detection");
        }

        let durations: Vec<f64> = self.samples.iter().map(|s| s.duration_ms).collect();
        let mut sorted = durations.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate quartiles
        let q1 = percentile(&sorted, 25.0);
        let q3 = percentile(&sorted, 75.0);
        let iqr = q3 - q1;

        // Define outlier thresholds
        let lower_bound = q1 - iqr_multiplier * iqr;
        let upper_bound = q3 + iqr_multiplier * iqr;

        // Find outliers
        let mut outlier_indices = Vec::new();
        let mut outlier_values = Vec::new();

        for (i, &value) in durations.iter().enumerate() {
            if value < lower_bound || value > upper_bound {
                outlier_indices.push(i);
                outlier_values.push(value);
            }
        }

        Ok(OutlierAnalysis {
            outlier_indices,
            outlier_values,
            method: format!("IQR (multiplier: {:.1})", iqr_multiplier),
            threshold: iqr_multiplier,
        })
    }

    /// Predict performance using simple linear regression
    /// Assumes input size correlates with performance
    pub fn predict_performance(&self, input_size: f64) -> Result<PerformancePrediction> {
        if self.samples.len() < 3 {
            anyhow::bail!("Insufficient samples for prediction (need at least 3)");
        }

        // Extract x (io_ops as proxy for input size) and y (duration)
        let x: Vec<f64> = self.samples.iter().map(|s| s.io_ops as f64).collect();
        let y: Vec<f64> = self.samples.iter().map(|s| s.duration_ms).collect();

        let n = x.len() as f64;

        // Calculate means
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        // Calculate slope (beta_1) and intercept (beta_0)
        let numerator: f64 = x
            .iter()
            .zip(&y)
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let denominator: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();

        let slope = numerator / denominator;
        let intercept = mean_y - slope * mean_x;

        // Predicted value
        let predicted_duration = intercept + slope * input_size;

        // Calculate R-squared
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let ss_res: f64 = x
            .iter()
            .zip(&y)
            .map(|(&xi, &yi)| {
                let pred = intercept + slope * xi;
                (yi - pred).powi(2)
            })
            .sum();

        let r_squared = 1.0 - (ss_res / ss_tot);

        // Standard error of prediction
        let se = (ss_res / (n - 2.0)).sqrt();
        let se_pred = se * (1.0 / n + (input_size - mean_x).powi(2) / denominator).sqrt();

        // 95% prediction interval (using t-distribution approximation)
        let t_value = 1.96; // For large samples
        let prediction_ci_lower = predicted_duration - t_value * se_pred;
        let prediction_ci_upper = predicted_duration + t_value * se_pred;

        Ok(PerformancePrediction {
            predicted_duration,
            prediction_ci_lower,
            prediction_ci_upper,
            r_squared,
            coefficients: vec![intercept, slope],
        })
    }
}

impl Default for StatisticalPerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate percentile of sorted data
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let n = sorted_data.len();
    let index = (p / 100.0) * (n as f64 - 1.0);
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_data[lower]
    } else {
        let weight = index - lower as f64;
        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }
}

/// Standard normal CDF approximation
fn standard_normal_cdf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let prob =
        d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

    if x >= 0.0 {
        1.0 - prob
    } else {
        prob
    }
}

/// Display performance statistics in formatted output
pub fn display_statistics(stats: &PerformanceStatistics) {
    println!("\n{}", "📊 Performance Statistics".bold().cyan());
    println!("{}", "━".repeat(70));

    println!("   Samples:            {}", stats.sample_count);
    println!("   Mean duration:      {:.2} ms", stats.mean_duration);
    println!(
        "   Std deviation:      {:.2} ms ({:.1}% CV)",
        stats.std_dev,
        stats.coefficient_variation * 100.0
    );
    println!(
        "   95% CI:             [{:.2}, {:.2}] ms",
        stats.ci_lower, stats.ci_upper
    );
    println!();
    println!("   Median (p50):       {:.2} ms", stats.median);
    println!("   95th percentile:    {:.2} ms", stats.p95);
    println!("   99th percentile:    {:.2} ms", stats.p99);
    println!();
}

/// Display regression test results
pub fn display_regression_test(result: &RegressionTestResult) {
    println!("\n{}", "🔬 Performance Regression Test".bold().cyan());
    println!("{}", "━".repeat(70));

    println!(
        "   {}",
        if result.regression_detected {
            result.interpretation.red().bold()
        } else {
            result.interpretation.green()
        }
    );
    println!();
    println!("   T-statistic:        {:.3}", result.t_statistic);
    println!("   P-value:            {:.4}", result.p_value);
    println!("   Effect size (d):    {:.3}", result.effect_size);
    println!("   Percent change:     {:.1}%", result.percent_change);
    println!("   Significance α:     {:.3}", result.significance_level);
    println!();

    // Interpretation guide
    println!("{}", "   Effect Size Interpretation:".bold());
    println!("   • |d| < 0.2:  Negligible");
    println!("   • |d| < 0.5:  Small");
    println!("   • |d| < 0.8:  Medium");
    println!("   • |d| ≥ 0.8:  Large");
    println!();
}

/// Display outlier analysis
pub fn display_outliers(analysis: &OutlierAnalysis) {
    println!("\n{}", "🎯 Outlier Detection".bold().cyan());
    println!("{}", "━".repeat(70));

    println!("   Method:             {}", analysis.method);
    println!("   Outliers detected:  {}", analysis.outlier_indices.len());

    if !analysis.outlier_values.is_empty() {
        println!();
        println!("   Outlier values:");
        for (idx, &value) in analysis
            .outlier_indices
            .iter()
            .zip(&analysis.outlier_values)
            .take(10)
        {
            println!("      Sample #{}: {:.2} ms", idx, value);
        }
        if analysis.outlier_values.len() > 10 {
            println!("      ... and {} more", analysis.outlier_values.len() - 10);
        }
    }
    println!();
}

/// Display performance prediction
pub fn display_prediction(prediction: &PerformancePrediction, input_size: f64) {
    println!("\n{}", "🔮 Performance Prediction".bold().cyan());
    println!("{}", "━".repeat(70));

    println!("   Input size:         {:.0}", input_size);
    println!(
        "   Predicted duration: {:.2} ms",
        prediction.predicted_duration
    );
    println!(
        "   95% prediction CI:  [{:.2}, {:.2}] ms",
        prediction.prediction_ci_lower, prediction.prediction_ci_upper
    );
    println!("   Model R²:           {:.4}", prediction.r_squared);
    println!(
        "   Regression:         y = {:.3} + {:.3}x",
        prediction.coefficients[0], prediction.coefficients[1]
    );
    println!();

    if prediction.r_squared < 0.5 {
        println!(
            "   {}",
            "⚠️  Low R² indicates weak linear relationship".yellow()
        );
    } else if prediction.r_squared < 0.8 {
        println!("   {} Moderate model fit", "ℹ️".blue());
    } else {
        println!("   {} Strong model fit", "✅".green());
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_data() -> Vec<PerformanceSample> {
        vec![
            PerformanceSample {
                timestamp: 1.0,
                duration_ms: 100.0,
                memory_mb: 50.0,
                cpu_percent: 25.0,
                io_ops: 100,
            },
            PerformanceSample {
                timestamp: 2.0,
                duration_ms: 105.0,
                memory_mb: 52.0,
                cpu_percent: 27.0,
                io_ops: 110,
            },
            PerformanceSample {
                timestamp: 3.0,
                duration_ms: 95.0,
                memory_mb: 48.0,
                cpu_percent: 23.0,
                io_ops: 90,
            },
            PerformanceSample {
                timestamp: 4.0,
                duration_ms: 102.0,
                memory_mb: 51.0,
                cpu_percent: 26.0,
                io_ops: 105,
            },
            PerformanceSample {
                timestamp: 5.0,
                duration_ms: 98.0,
                memory_mb: 49.0,
                cpu_percent: 24.0,
                io_ops: 95,
            },
        ]
    }

    #[test]
    fn test_statistics_calculation() {
        let mut analyzer = StatisticalPerformanceAnalyzer::new();
        analyzer.add_samples(create_sample_data());

        let stats = analyzer.calculate_statistics().unwrap();

        assert_eq!(stats.sample_count, 5);
        assert!(stats.mean_duration > 95.0 && stats.mean_duration < 105.0);
        assert!(stats.std_dev > 0.0);
        assert!(stats.ci_lower < stats.mean_duration);
        assert!(stats.ci_upper > stats.mean_duration);
        assert!(stats.coefficient_variation > 0.0);
    }

    #[test]
    fn test_regression_detection() {
        let mut analyzer = StatisticalPerformanceAnalyzer::new();
        analyzer.add_samples(create_sample_data());

        // Create slower baseline
        let baseline: Vec<PerformanceSample> = (0..5)
            .map(|i| PerformanceSample {
                timestamp: i as f64,
                duration_ms: 80.0 + i as f64,
                memory_mb: 45.0,
                cpu_percent: 20.0,
                io_ops: 80,
            })
            .collect();

        let result = analyzer.test_regression(&baseline).unwrap();

        // Current samples should be slower than baseline
        assert!(result.percent_change > 0.0);
        assert!(result.t_statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_outlier_detection() {
        let mut analyzer = StatisticalPerformanceAnalyzer::new();
        let mut samples = create_sample_data();

        // Add obvious outlier
        samples.push(PerformanceSample {
            timestamp: 6.0,
            duration_ms: 500.0, // Much higher than others
            memory_mb: 100.0,
            cpu_percent: 90.0,
            io_ops: 500,
        });

        analyzer.add_samples(samples);

        let outliers = analyzer.detect_outliers(1.5).unwrap();

        // Should detect the outlier
        assert!(!outliers.outlier_indices.is_empty());
        assert!(outliers.outlier_values.iter().any(|&v| v > 200.0));
    }

    #[test]
    fn test_performance_prediction() {
        let mut analyzer = StatisticalPerformanceAnalyzer::new();
        analyzer.add_samples(create_sample_data());

        let prediction = analyzer.predict_performance(120.0).unwrap();

        assert!(prediction.predicted_duration > 0.0);
        assert!(prediction.prediction_ci_lower < prediction.predicted_duration);
        assert!(prediction.prediction_ci_upper > prediction.predicted_duration);
        assert!(prediction.r_squared >= 0.0 && prediction.r_squared <= 1.0);
        assert_eq!(prediction.coefficients.len(), 2); // intercept + slope
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 50.0), 3.0);
        assert_eq!(percentile(&data, 100.0), 5.0);
    }

    #[test]
    fn test_empty_samples_error() {
        let analyzer = StatisticalPerformanceAnalyzer::new();
        assert!(analyzer.calculate_statistics().is_err());
        assert!(analyzer.detect_outliers(1.5).is_err());
        assert!(analyzer.predict_performance(100.0).is_err());
    }
}
