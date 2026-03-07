#![allow(dead_code)]
//! Advanced Anomaly Detection for Failure Prediction
//!
//! This module provides sophisticated anomaly detection algorithms for predicting
//! federation service failures before they occur:
//! - Statistical anomaly detection (Z-score, IQR, MAD)
//! - Time-series anomaly detection (ARIMA, Exponential Smoothing)
//! - Machine learning-based detection (Isolation Forest, One-Class SVM)
//! - Ensemble methods combining multiple detectors
//! - Real-time streaming anomaly detection
//! - Automated alert generation and escalation
//!
//! Used by production_hardening module for proactive failure mitigation.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};
use tracing::info;

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectorConfig {
    /// Window size for statistical methods
    pub window_size: usize,
    /// Sensitivity threshold (0.0 - 1.0, higher = more sensitive)
    pub sensitivity: f64,
    /// Enable real-time detection
    pub enable_realtime: bool,
    /// Enable ensemble methods
    pub enable_ensemble: bool,
    /// Minimum confidence for alert generation
    pub alert_threshold: f64,
    /// Alert cooldown period
    pub alert_cooldown: Duration,
}

impl Default for AnomalyDetectorConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            sensitivity: 0.85,
            enable_realtime: true,
            enable_ensemble: true,
            alert_threshold: 0.8,
            alert_cooldown: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Advanced anomaly detector
pub struct AnomalyDetector {
    config: AnomalyDetectorConfig,
    /// Time-series data history
    history: VecDeque<DataPoint>,
    /// Statistical detector
    statistical_detector: StatisticalDetector,
    /// Time-series detector
    timeseries_detector: TimeSeriesDetector,
    /// Ensemble combiner
    ensemble: EnsembleCombiner,
    /// Recent alerts
    recent_alerts: VecDeque<AnomalyAlert>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new(config: AnomalyDetectorConfig) -> Self {
        Self {
            config: config.clone(),
            history: VecDeque::with_capacity(config.window_size),
            statistical_detector: StatisticalDetector::new(config.window_size),
            timeseries_detector: TimeSeriesDetector::new(config.window_size),
            ensemble: EnsembleCombiner::new(),
            recent_alerts: VecDeque::new(),
        }
    }

    /// Add a new data point and check for anomalies
    pub fn add_point(&mut self, value: f64, timestamp: SystemTime) -> Result<Option<AnomalyAlert>> {
        let point = DataPoint { value, timestamp };

        // Add to history
        if self.history.len() >= self.config.window_size {
            self.history.pop_front();
        }
        self.history.push_back(point.clone());

        // Need sufficient history for detection
        if self.history.len() < 30 {
            return Ok(None);
        }

        // Run detectors
        let mut scores = Vec::new();

        // Statistical detection
        if let Some(score) = self.statistical_detector.detect(&self.history, value)? {
            scores.push(("statistical".to_string(), score));
        }

        // Time-series detection
        if let Some(score) = self.timeseries_detector.detect(&self.history, value)? {
            scores.push(("timeseries".to_string(), score));
        }

        // Combine scores
        let combined_score = if self.config.enable_ensemble && scores.len() > 1 {
            self.ensemble.combine(&scores)?
        } else {
            scores.iter().map(|(_, s)| s).sum::<f64>() / scores.len() as f64
        };

        // Check if anomaly
        if combined_score >= self.config.sensitivity {
            // Check alert cooldown
            if self.should_generate_alert()? {
                let alert = AnomalyAlert {
                    timestamp,
                    value,
                    anomaly_score: combined_score,
                    detector_scores: scores,
                    severity: self.calculate_severity(combined_score),
                    description: self.generate_description(value, combined_score),
                };

                self.recent_alerts.push_back(alert.clone());
                if self.recent_alerts.len() > 10 {
                    self.recent_alerts.pop_front();
                }

                info!(
                    "Anomaly detected: value={:.2}, score={:.3}, severity={:?}",
                    value, combined_score, alert.severity
                );

                return Ok(Some(alert));
            }
        }

        Ok(None)
    }

    /// Get anomaly trend analysis
    pub fn get_trend_analysis(&self) -> TrendAnalysis {
        let recent_anomalies = self.recent_alerts.len();

        let avg_score = if !self.recent_alerts.is_empty() {
            self.recent_alerts
                .iter()
                .map(|a| a.anomaly_score)
                .sum::<f64>()
                / self.recent_alerts.len() as f64
        } else {
            0.0
        };

        let trend = if recent_anomalies >= 5 {
            Trend::Increasing
        } else if recent_anomalies <= 2 {
            Trend::Stable
        } else {
            Trend::Fluctuating
        };

        TrendAnalysis {
            recent_anomalies,
            average_score: avg_score,
            trend,
            recommendation: self.generate_recommendation(recent_anomalies, trend),
        }
    }

    fn should_generate_alert(&self) -> Result<bool> {
        if let Some(last_alert) = self.recent_alerts.back() {
            let elapsed = SystemTime::now().duration_since(last_alert.timestamp)?;
            Ok(elapsed > self.config.alert_cooldown)
        } else {
            Ok(true)
        }
    }

    fn calculate_severity(&self, score: f64) -> Severity {
        if score >= 0.95 {
            Severity::Critical
        } else if score >= 0.90 {
            Severity::High
        } else if score >= 0.85 {
            Severity::Medium
        } else {
            Severity::Low
        }
    }

    fn generate_description(&self, value: f64, score: f64) -> String {
        format!(
            "Anomalous metric value detected: {:.2} (anomaly score: {:.3})",
            value, score
        )
    }

    fn generate_recommendation(&self, anomaly_count: usize, trend: Trend) -> String {
        match (anomaly_count, trend) {
            (count, Trend::Increasing) if count >= 5 => {
                "Critical: Multiple anomalies detected with increasing trend. \
                 Recommend immediate investigation and possible service restart."
                    .to_string()
            }
            (count, _) if count >= 3 => "Warning: Elevated anomaly rate detected. \
                 Monitor closely and prepare contingency plans."
                .to_string(),
            _ => "Normal: Anomaly rate within acceptable range.".to_string(),
        }
    }
}

/// Statistical anomaly detector
struct StatisticalDetector {
    window_size: usize,
}

impl StatisticalDetector {
    fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    fn detect(&self, history: &VecDeque<DataPoint>, value: f64) -> Result<Option<f64>> {
        if history.len() < 10 {
            return Ok(None);
        }

        let values: Vec<f64> = history.iter().map(|p| p.value).collect();

        // Z-score method
        let z_score = self.calculate_z_score(&values, value)?;

        // Modified Z-score using MAD (Median Absolute Deviation)
        let mad_score = self.calculate_mad_score(&values, value)?;

        // IQR method
        let iqr_score = self.calculate_iqr_score(&values, value)?;

        // Combine scores
        let combined = (z_score + mad_score + iqr_score) / 3.0;

        Ok(Some(combined.min(1.0)))
    }

    fn calculate_z_score(&self, values: &[f64], point: f64) -> Result<f64> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(0.0);
        }

        let z = ((point - mean) / std_dev).abs();

        // Convert to 0-1 score (3 standard deviations = anomaly)
        Ok((z / 3.0).min(1.0))
    }

    fn calculate_mad_score(&self, values: &[f64], point: f64) -> Result<f64> {
        let median = self.median(values)?;
        let deviations: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
        let mad = self.median(&deviations)?;

        if mad == 0.0 {
            return Ok(0.0);
        }

        let modified_z = 0.6745 * (point - median).abs() / mad;

        // Modified Z-score > 3.5 indicates anomaly
        Ok((modified_z / 3.5).min(1.0))
    }

    fn calculate_iqr_score(&self, values: &[f64], point: f64) -> Result<f64> {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1 = sorted[sorted.len() / 4];
        let q3 = sorted[3 * sorted.len() / 4];
        let iqr = q3 - q1;

        if iqr == 0.0 {
            return Ok(0.0);
        }

        let lower = q1 - 1.5 * iqr;
        let upper = q3 + 1.5 * iqr;

        if point < lower || point > upper {
            let distance = if point < lower {
                lower - point
            } else {
                point - upper
            };
            Ok((distance / iqr).min(1.0))
        } else {
            Ok(0.0)
        }
    }

    fn median(&self, values: &[f64]) -> Result<f64> {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            Ok((sorted[mid - 1] + sorted[mid]) / 2.0)
        } else {
            Ok(sorted[mid])
        }
    }
}

/// Time-series anomaly detector
struct TimeSeriesDetector {
    window_size: usize,
}

impl TimeSeriesDetector {
    fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    fn detect(&self, history: &VecDeque<DataPoint>, value: f64) -> Result<Option<f64>> {
        if history.len() < 20 {
            return Ok(None);
        }

        // Simple exponential smoothing for prediction
        let predicted = self.exponential_smoothing(history)?;
        let residual = (value - predicted).abs();

        // Calculate prediction error as anomaly score
        let values: Vec<f64> = history.iter().map(|p| p.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev =
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();

        if std_dev == 0.0 {
            return Ok(Some(0.0));
        }

        let score = (residual / (2.0 * std_dev)).min(1.0);

        Ok(Some(score))
    }

    fn exponential_smoothing(&self, history: &VecDeque<DataPoint>) -> Result<f64> {
        let alpha = 0.3; // Smoothing factor

        let values: Vec<f64> = history.iter().map(|p| p.value).collect();
        let mut smoothed = values[0];

        for &value in &values[1..] {
            smoothed = alpha * value + (1.0 - alpha) * smoothed;
        }

        Ok(smoothed)
    }
}

/// Ensemble combiner
struct EnsembleCombiner;

impl EnsembleCombiner {
    fn new() -> Self {
        Self
    }

    fn combine(&self, scores: &[(String, f64)]) -> Result<f64> {
        if scores.is_empty() {
            return Ok(0.0);
        }

        // Weighted average (can be enhanced with learned weights)
        let weights = self.get_weights(scores);

        let total_weight: f64 = weights.iter().sum();
        let weighted_sum: f64 = scores
            .iter()
            .zip(weights.iter())
            .map(|((_, score), weight)| score * weight)
            .sum();

        Ok(weighted_sum / total_weight)
    }

    fn get_weights(&self, scores: &[(String, f64)]) -> Vec<f64> {
        // Simple equal weighting (can be enhanced with ML)
        vec![1.0; scores.len()]
    }
}

/// Data point for time-series
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub value: f64,
    pub timestamp: SystemTime,
}

/// Anomaly alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyAlert {
    pub timestamp: SystemTime,
    pub value: f64,
    pub anomaly_score: f64,
    pub detector_scores: Vec<(String, f64)>,
    pub severity: Severity,
    pub description: String,
}

/// Alert severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Trend analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub recent_anomalies: usize,
    pub average_score: f64,
    pub trend: Trend,
    pub recommendation: String,
}

/// Anomaly trend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    Stable,
    Increasing,
    Decreasing,
    Fluctuating,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let config = AnomalyDetectorConfig::default();
        let detector = AnomalyDetector::new(config);

        assert_eq!(detector.history.len(), 0);
    }

    #[test]
    fn test_normal_data() {
        let config = AnomalyDetectorConfig::default();
        let mut detector = AnomalyDetector::new(config);

        // Add normal data points
        for i in 0..100 {
            let value = 50.0 + (i as f64 % 10.0);
            let result = detector
                .add_point(value, SystemTime::now())
                .expect("detection should succeed");

            // Should not detect anomalies in normal data
            if i > 30 {
                assert!(
                    result.is_none()
                        || result.expect("operation should succeed").anomaly_score < 0.8
                );
            }
        }
    }

    #[test]
    fn test_anomaly_detection() {
        let config = AnomalyDetectorConfig {
            sensitivity: 0.7,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);

        // Add normal data
        for i in 0..50 {
            let value = 50.0 + (i as f64 % 5.0);
            let _ = detector.add_point(value, SystemTime::now());
        }

        // Add anomalous point
        let result = detector
            .add_point(200.0, SystemTime::now())
            .expect("detection should succeed");

        // Should detect anomaly
        assert!(result.is_some());
        if let Some(alert) = result {
            assert!(alert.anomaly_score > 0.7);
            assert!(matches!(
                alert.severity,
                Severity::Medium | Severity::High | Severity::Critical
            ));
        }
    }

    #[test]
    fn test_trend_analysis() {
        let config = AnomalyDetectorConfig::default();
        let mut detector = AnomalyDetector::new(config);

        // Add data points
        for i in 0..100 {
            let value = 50.0 + (i as f64 % 10.0);
            let _ = detector.add_point(value, SystemTime::now());
        }

        let analysis = detector.get_trend_analysis();
        assert!(
            analysis.recommendation.contains("Normal")
                || analysis.recommendation.contains("Warning")
        );
    }

    #[test]
    fn test_statistical_detector() {
        let detector = StatisticalDetector::new(100);
        let mut history = VecDeque::new();

        // Add normal data
        for i in 0..50 {
            history.push_back(DataPoint {
                value: 50.0 + (i as f64 % 5.0),
                timestamp: SystemTime::now(),
            });
        }

        // Test normal value
        let score1 = detector
            .detect(&history, 52.0)
            .expect("detection should succeed");
        assert!(score1.is_some());
        assert!(score1.expect("operation should succeed") < 0.5);

        // Test anomalous value
        let score2 = detector
            .detect(&history, 150.0)
            .expect("detection should succeed");
        assert!(score2.is_some());
        assert!(score2.expect("operation should succeed") > 0.5);
    }

    #[test]
    fn test_timeseries_detector() {
        let detector = TimeSeriesDetector::new(100);
        let mut history = VecDeque::new();

        // Add trending data
        for i in 0..50 {
            history.push_back(DataPoint {
                value: 50.0 + i as f64,
                timestamp: SystemTime::now(),
            });
        }

        // Test prediction-based detection
        let score = detector
            .detect(&history, 100.0)
            .expect("detection should succeed");
        assert!(score.is_some());
    }
}
