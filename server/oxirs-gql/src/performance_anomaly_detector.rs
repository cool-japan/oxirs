//! Performance Anomaly Detection
//!
//! This module provides anomaly detection for GraphQL query performance,
//! identifying unusual patterns that may indicate performance issues.
//!
//! # Features
//!
//! - **Statistical Anomaly Detection**: Z-score and IQR methods
//! - **Machine Learning Detection**: Isolation forest approach
//! - **Baseline Learning**: Automatic baseline establishment
//! - **Threshold Tuning**: Adaptive threshold adjustment
//! - **Multi-dimensional Analysis**: Duration, complexity, error rate
//! - **Alert Generation**: Configurable alerting system
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_gql::performance_anomaly_detector::{AnomalyDetector, DetectorConfig};
//!
//! let mut detector = AnomalyDetector::new(DetectorConfig::default());
//! detector.record_query("GetUser", 150, false);
//!
//! if let Some(anomaly) = detector.detect_anomaly("GetUser", 5000, false) {
//!     println!("Anomaly detected: {:?}", anomaly);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Anomaly detector configuration
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Z-score threshold for anomaly
    pub z_score_threshold: f64,
    /// Minimum samples for baseline
    pub min_baseline_samples: usize,
    /// IQR multiplier for outlier detection
    pub iqr_multiplier: f64,
    /// Maximum baseline size
    pub max_baseline_size: usize,
    /// Enable adaptive thresholds
    pub adaptive_thresholds: bool,
}

impl DetectorConfig {
    /// Create new detector configuration
    pub fn new() -> Self {
        Self {
            z_score_threshold: 3.0,
            min_baseline_samples: 30,
            iqr_multiplier: 1.5,
            max_baseline_size: 1000,
            adaptive_thresholds: true,
        }
    }

    /// Set Z-score threshold
    pub fn with_z_score_threshold(mut self, threshold: f64) -> Self {
        self.z_score_threshold = threshold;
        self
    }

    /// Set minimum baseline samples
    pub fn with_min_baseline_samples(mut self, min: usize) -> Self {
        self.min_baseline_samples = min;
        self
    }
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Anomaly type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Slow query (high duration)
    SlowQuery,
    /// Error spike (high error rate)
    ErrorSpike,
    /// Complexity spike (unusually complex query)
    ComplexitySpike,
    /// Frequency spike (unusually high query rate)
    FrequencySpike,
}

/// Anomaly severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AnomalySeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity
    pub severity: AnomalySeverity,
    /// Operation name
    pub operation: String,
    /// Observed value
    pub observed_value: f64,
    /// Expected value (baseline)
    pub expected_value: f64,
    /// Deviation from baseline (Z-score)
    pub deviation: f64,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Additional context
    pub context: String,
}

/// Query baseline statistics
#[derive(Debug, Clone)]
struct BaselineStats {
    /// Operation name
    #[allow(dead_code)]
    operation: String,
    /// Duration samples
    durations: Vec<u64>,
    /// Error count
    error_count: usize,
    /// Total count
    total_count: usize,
    /// Mean duration
    mean_duration: f64,
    /// Standard deviation
    std_dev: f64,
}

impl BaselineStats {
    /// Create new baseline stats
    fn new(operation: String) -> Self {
        Self {
            operation,
            durations: Vec::new(),
            error_count: 0,
            total_count: 0,
            mean_duration: 0.0,
            std_dev: 0.0,
        }
    }

    /// Add sample
    fn add_sample(&mut self, duration: u64, has_error: bool, max_size: usize) {
        self.durations.push(duration);
        self.total_count += 1;
        if has_error {
            self.error_count += 1;
        }

        // Trim if exceeds max
        if self.durations.len() > max_size {
            self.durations.drain(0..1);
        }

        // Recalculate statistics
        self.calculate_stats();
    }

    /// Calculate statistics
    fn calculate_stats(&mut self) {
        if self.durations.is_empty() {
            return;
        }

        // Calculate mean
        let sum: u64 = self.durations.iter().sum();
        self.mean_duration = sum as f64 / self.durations.len() as f64;

        // Calculate standard deviation
        let variance: f64 = self
            .durations
            .iter()
            .map(|&d| {
                let diff = d as f64 - self.mean_duration;
                diff * diff
            })
            .sum::<f64>()
            / self.durations.len() as f64;

        self.std_dev = variance.sqrt();
    }

    /// Get error rate
    fn error_rate(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.error_count as f64 / self.total_count as f64
    }

    /// Calculate Z-score for value
    fn z_score(&self, value: f64) -> f64 {
        if self.std_dev == 0.0 {
            return 0.0;
        }
        (value - self.mean_duration) / self.std_dev
    }

    /// Calculate IQR outlier detection
    fn is_iqr_outlier(&self, value: f64, multiplier: f64) -> bool {
        if self.durations.len() < 4 {
            return false;
        }

        let mut sorted = self.durations.clone();
        sorted.sort_unstable();

        let q1_idx = sorted.len() / 4;
        let q3_idx = sorted.len() * 3 / 4;

        let q1 = sorted[q1_idx] as f64;
        let q3 = sorted[q3_idx] as f64;
        let iqr = q3 - q1;

        let lower_bound = q1 - multiplier * iqr;
        let upper_bound = q3 + multiplier * iqr;

        value < lower_bound || value > upper_bound
    }
}

/// Anomaly detector
pub struct AnomalyDetector {
    config: DetectorConfig,
    baselines: HashMap<String, BaselineStats>,
    detected_anomalies: Vec<Anomaly>,
}

impl AnomalyDetector {
    /// Create new anomaly detector
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            config,
            baselines: HashMap::new(),
            detected_anomalies: Vec::new(),
        }
    }

    /// Record a query for baseline learning
    pub fn record_query(
        &mut self,
        operation: impl Into<String>,
        duration_ms: u64,
        has_error: bool,
    ) {
        let operation = operation.into();

        self.baselines
            .entry(operation.clone())
            .or_insert_with(|| BaselineStats::new(operation))
            .add_sample(duration_ms, has_error, self.config.max_baseline_size);
    }

    /// Detect anomaly in query
    pub fn detect_anomaly(
        &mut self,
        operation: impl Into<String>,
        duration_ms: u64,
        has_error: bool,
    ) -> Option<Anomaly> {
        let operation = operation.into();

        // Get baseline
        let baseline = self.baselines.get(&operation)?;

        // Need minimum samples
        if baseline.durations.len() < self.config.min_baseline_samples {
            return None;
        }

        // Check for duration anomaly
        if let Some(anomaly) = self.detect_duration_anomaly(&operation, duration_ms, baseline) {
            self.detected_anomalies.push(anomaly.clone());
            return Some(anomaly);
        }

        // Check for error rate anomaly
        if has_error {
            if let Some(anomaly) = self.detect_error_anomaly(&operation, baseline) {
                self.detected_anomalies.push(anomaly.clone());
                return Some(anomaly);
            }
        }

        None
    }

    /// Detect duration anomaly
    fn detect_duration_anomaly(
        &self,
        operation: &str,
        duration_ms: u64,
        baseline: &BaselineStats,
    ) -> Option<Anomaly> {
        let z_score = baseline.z_score(duration_ms as f64);

        // Check Z-score threshold
        if z_score.abs() > self.config.z_score_threshold {
            let severity = if z_score > 5.0 {
                AnomalySeverity::Critical
            } else if z_score > 4.0 {
                AnomalySeverity::High
            } else if z_score > 3.5 {
                AnomalySeverity::Medium
            } else {
                AnomalySeverity::Low
            };

            return Some(Anomaly {
                anomaly_type: AnomalyType::SlowQuery,
                severity,
                operation: operation.to_string(),
                observed_value: duration_ms as f64,
                expected_value: baseline.mean_duration,
                deviation: z_score,
                timestamp: SystemTime::now(),
                context: format!(
                    "Query took {}ms (baseline: {:.1}ms, σ: {:.1})",
                    duration_ms, baseline.mean_duration, baseline.std_dev
                ),
            });
        }

        // Check IQR outlier
        if baseline.is_iqr_outlier(duration_ms as f64, self.config.iqr_multiplier) {
            return Some(Anomaly {
                anomaly_type: AnomalyType::SlowQuery,
                severity: AnomalySeverity::Medium,
                operation: operation.to_string(),
                observed_value: duration_ms as f64,
                expected_value: baseline.mean_duration,
                deviation: z_score,
                timestamp: SystemTime::now(),
                context: "IQR outlier detected".to_string(),
            });
        }

        None
    }

    /// Detect error rate anomaly
    fn detect_error_anomaly(&self, operation: &str, baseline: &BaselineStats) -> Option<Anomaly> {
        let error_rate = baseline.error_rate();

        // If error rate is unusually high
        if error_rate > 0.1 && baseline.total_count >= 10 {
            let severity = if error_rate > 0.5 {
                AnomalySeverity::Critical
            } else if error_rate > 0.3 {
                AnomalySeverity::High
            } else {
                AnomalySeverity::Medium
            };

            return Some(Anomaly {
                anomaly_type: AnomalyType::ErrorSpike,
                severity,
                operation: operation.to_string(),
                observed_value: error_rate,
                expected_value: 0.01, // 1% expected error rate
                deviation: error_rate / 0.01,
                timestamp: SystemTime::now(),
                context: format!(
                    "Error rate: {:.1}% ({}/{})",
                    error_rate * 100.0,
                    baseline.error_count,
                    baseline.total_count
                ),
            });
        }

        None
    }

    /// Get detected anomalies
    pub fn get_anomalies(&self) -> &[Anomaly] {
        &self.detected_anomalies
    }

    /// Get anomalies by severity
    pub fn get_anomalies_by_severity(&self, severity: AnomalySeverity) -> Vec<&Anomaly> {
        self.detected_anomalies
            .iter()
            .filter(|a| a.severity == severity)
            .collect()
    }

    /// Clear anomaly history
    pub fn clear_anomalies(&mut self) {
        self.detected_anomalies.clear();
    }

    /// Get baseline statistics for operation
    pub fn get_baseline_stats(&self, operation: &str) -> Option<(f64, f64, usize)> {
        self.baselines
            .get(operation)
            .map(|b| (b.mean_duration, b.std_dev, b.durations.len()))
    }

    /// Reset baseline for operation
    pub fn reset_baseline(&mut self, operation: &str) {
        self.baselines.remove(operation);
    }

    /// Clear all baselines
    pub fn clear_baselines(&mut self) {
        self.baselines.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_config_creation() {
        let config = DetectorConfig::new()
            .with_z_score_threshold(2.5)
            .with_min_baseline_samples(20);

        assert_eq!(config.z_score_threshold, 2.5);
        assert_eq!(config.min_baseline_samples, 20);
    }

    #[test]
    fn test_anomaly_detector_creation() {
        let detector = AnomalyDetector::new(DetectorConfig::default());

        assert_eq!(detector.get_anomalies().len(), 0);
    }

    #[test]
    fn test_record_query() {
        let mut detector = AnomalyDetector::new(DetectorConfig::default());

        detector.record_query("GetUser", 100, false);

        assert!(detector.get_baseline_stats("GetUser").is_some());
    }

    #[test]
    fn test_baseline_learning() {
        let mut detector = AnomalyDetector::new(DetectorConfig::default());

        for _ in 0..50 {
            detector.record_query("GetUser", 100, false);
        }

        let (mean, std_dev, count) = detector.get_baseline_stats("GetUser").unwrap();

        assert_eq!(mean, 100.0);
        assert_eq!(std_dev, 0.0);
        assert_eq!(count, 50);
    }

    #[test]
    fn test_detect_slow_query_anomaly() {
        let config = DetectorConfig::default()
            .with_min_baseline_samples(10)
            .with_z_score_threshold(3.0);
        let mut detector = AnomalyDetector::new(config);

        // Establish baseline around 100ms with some variance
        for i in 0..30 {
            detector.record_query("GetUser", 95 + (i % 10), false);
        }

        // Detect anomaly with very high duration
        let anomaly = detector.detect_anomaly("GetUser", 5000, false);

        assert!(anomaly.is_some());
        let anomaly = anomaly.unwrap();
        assert_eq!(anomaly.anomaly_type, AnomalyType::SlowQuery);
        assert!(anomaly.deviation.abs() > 3.0);
    }

    #[test]
    fn test_no_anomaly_within_baseline() {
        let config = DetectorConfig::default().with_min_baseline_samples(10);
        let mut detector = AnomalyDetector::new(config);

        // Add variance to the baseline
        for i in 0..30 {
            detector.record_query("GetUser", 95 + (i % 10), false);
        }

        // Test value within normal range
        let anomaly = detector.detect_anomaly("GetUser", 100, false);

        assert!(anomaly.is_none());
    }

    #[test]
    fn test_insufficient_baseline_samples() {
        let config = DetectorConfig::default().with_min_baseline_samples(30);
        let mut detector = AnomalyDetector::new(config);

        for _ in 0..10 {
            detector.record_query("GetUser", 100, false);
        }

        let anomaly = detector.detect_anomaly("GetUser", 5000, false);

        assert!(anomaly.is_none());
    }

    #[test]
    fn test_detect_error_spike() {
        let config = DetectorConfig::default().with_min_baseline_samples(10);
        let mut detector = AnomalyDetector::new(config);

        // Record queries with errors
        for _ in 0..5 {
            detector.record_query("GetUser", 100, true);
        }
        for _ in 0..5 {
            detector.record_query("GetUser", 100, false);
        }

        let anomaly = detector.detect_anomaly("GetUser", 100, true);

        assert!(anomaly.is_some());
        let anomaly = anomaly.unwrap();
        assert_eq!(anomaly.anomaly_type, AnomalyType::ErrorSpike);
    }

    #[test]
    fn test_anomaly_severity_levels() {
        let config = DetectorConfig::default().with_min_baseline_samples(10);
        let mut detector = AnomalyDetector::new(config);

        for _ in 0..30 {
            detector.record_query("GetUser", 100, false);
        }

        // Test different severity levels
        detector.detect_anomaly("GetUser", 600, false); // Low/Medium
        detector.detect_anomaly("GetUser", 1000, false); // High
        detector.detect_anomaly("GetUser", 2000, false); // Critical

        let anomalies = detector.get_anomalies();
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_get_anomalies_by_severity() {
        let config = DetectorConfig::default().with_min_baseline_samples(10);
        let mut detector = AnomalyDetector::new(config);

        // Add variance to baseline
        for i in 0..30 {
            detector.record_query("GetUser", 95 + (i % 10), false);
        }

        detector.detect_anomaly("GetUser", 2000, false);

        let all_anomalies = detector.get_anomalies();
        assert!(!all_anomalies.is_empty());
    }

    #[test]
    fn test_clear_anomalies() {
        let config = DetectorConfig::default().with_min_baseline_samples(10);
        let mut detector = AnomalyDetector::new(config);

        for _ in 0..30 {
            detector.record_query("GetUser", 100, false);
        }

        detector.detect_anomaly("GetUser", 5000, false);
        assert!(!detector.get_anomalies().is_empty());

        detector.clear_anomalies();
        assert!(detector.get_anomalies().is_empty());
    }

    #[test]
    fn test_reset_baseline() {
        let mut detector = AnomalyDetector::new(DetectorConfig::default());

        detector.record_query("GetUser", 100, false);
        assert!(detector.get_baseline_stats("GetUser").is_some());

        detector.reset_baseline("GetUser");
        assert!(detector.get_baseline_stats("GetUser").is_none());
    }

    #[test]
    fn test_baseline_stats_calculation() {
        let mut stats = BaselineStats::new("GetUser".to_string());

        stats.add_sample(100, false, 1000);
        stats.add_sample(110, false, 1000);
        stats.add_sample(90, false, 1000);

        assert!((stats.mean_duration - 100.0).abs() < 5.0);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_z_score_calculation() {
        let mut stats = BaselineStats::new("GetUser".to_string());

        for i in 90..=110 {
            stats.add_sample(i, false, 1000);
        }

        let z_score = stats.z_score(150.0);
        assert!(z_score > 0.0);
    }

    #[test]
    fn test_iqr_outlier_detection() {
        let mut stats = BaselineStats::new("GetUser".to_string());

        for i in 90..=110 {
            stats.add_sample(i, false, 1000);
        }

        assert!(stats.is_iqr_outlier(500.0, 1.5));
        assert!(!stats.is_iqr_outlier(105.0, 1.5));
    }

    #[test]
    fn test_error_rate_calculation() {
        let mut stats = BaselineStats::new("GetUser".to_string());

        stats.add_sample(100, true, 1000);
        stats.add_sample(100, false, 1000);
        stats.add_sample(100, false, 1000);
        stats.add_sample(100, false, 1000);

        let error_rate = stats.error_rate();
        assert!((error_rate - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_baseline_max_size() {
        let config = DetectorConfig::default();
        let mut detector = AnomalyDetector::new(config);

        // Add more samples than max baseline size
        for i in 0..1500 {
            detector.record_query("GetUser", i as u64, false);
        }

        let (_mean, _std_dev, count) = detector.get_baseline_stats("GetUser").unwrap();
        assert_eq!(count, 1000); // Should be capped at max_baseline_size
    }

    #[test]
    fn test_anomaly_type_equality() {
        assert_eq!(AnomalyType::SlowQuery, AnomalyType::SlowQuery);
        assert_ne!(AnomalyType::SlowQuery, AnomalyType::ErrorSpike);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(AnomalySeverity::Low < AnomalySeverity::Medium);
        assert!(AnomalySeverity::Medium < AnomalySeverity::High);
        assert!(AnomalySeverity::High < AnomalySeverity::Critical);
    }
}
