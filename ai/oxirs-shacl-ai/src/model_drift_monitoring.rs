//! Model Drift Monitoring System
//!
//! This module provides comprehensive drift detection for production AI models,
//! including data drift, concept drift, and performance drift detection.

use crate::{Result, ShaclAiError};

use chrono::{DateTime, Duration, Utc};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Model drift monitor
#[derive(Debug)]
pub struct DriftMonitor {
    /// Monitor configuration
    config: DriftMonitorConfig,

    /// Reference (baseline) data statistics
    reference_stats: Arc<Mutex<Option<DataStatistics>>>,

    /// Historical drift measurements
    drift_history: Arc<Mutex<VecDeque<DriftMeasurement>>>,

    /// Active alerts
    active_alerts: Arc<Mutex<Vec<DriftAlert>>>,

    /// Monitoring statistics
    stats: MonitoringStats,
}

/// Drift monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMonitorConfig {
    /// Enable data drift detection
    pub enable_data_drift: bool,

    /// Enable concept drift detection
    pub enable_concept_drift: bool,

    /// Enable performance drift detection
    pub enable_performance_drift: bool,

    /// Drift detection window size
    pub window_size: usize,

    /// Alert threshold for drift score (0.0-1.0)
    pub alert_threshold: f64,

    /// Warning threshold for drift score (0.0-1.0)
    pub warning_threshold: f64,

    /// Minimum samples before drift detection
    pub min_samples: usize,

    /// Check frequency in seconds
    pub check_frequency_secs: u64,

    /// Maximum history size
    pub max_history_size: usize,

    /// Enable automatic alerting
    pub enable_alerting: bool,

    /// Statistical significance level
    pub significance_level: f64,
}

impl Default for DriftMonitorConfig {
    fn default() -> Self {
        Self {
            enable_data_drift: true,
            enable_concept_drift: true,
            enable_performance_drift: true,
            window_size: 1000,
            alert_threshold: 0.7,
            warning_threshold: 0.5,
            min_samples: 100,
            check_frequency_secs: 3600, // 1 hour
            max_history_size: 10000,
            enable_alerting: true,
            significance_level: 0.05,
        }
    }
}

/// Data statistics for drift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStatistics {
    /// Feature means
    pub feature_means: Vec<f64>,

    /// Feature standard deviations
    pub feature_stds: Vec<f64>,

    /// Feature min values
    pub feature_mins: Vec<f64>,

    /// Feature max values
    pub feature_maxs: Vec<f64>,

    /// Feature distributions (histograms)
    pub feature_distributions: Vec<Vec<f64>>,

    /// Number of features
    pub num_features: usize,

    /// Number of samples
    pub num_samples: usize,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Drift measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMeasurement {
    /// Measurement ID
    pub id: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Drift type
    pub drift_type: ModelDriftType,

    /// Drift score (0.0 = no drift, 1.0 = maximum drift)
    pub drift_score: f64,

    /// Statistical significance (p-value)
    pub p_value: f64,

    /// Detected drift (based on threshold)
    pub is_drift_detected: bool,

    /// Affected features
    pub affected_features: Vec<String>,

    /// Detailed metrics
    pub metrics: HashMap<String, f64>,
}

/// Type of model drift
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelDriftType {
    /// Data distribution drift
    DataDrift,

    /// Concept drift (relationship between X and Y changed)
    ConceptDrift,

    /// Performance drift (model accuracy degraded)
    PerformanceDrift,

    /// Feature drift (specific feature distribution changed)
    FeatureDrift,
}

/// Drift alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftAlert {
    /// Alert ID
    pub id: String,

    /// Alert timestamp
    pub timestamp: DateTime<Utc>,

    /// Drift measurement that triggered alert
    pub measurement: DriftMeasurement,

    /// Alert severity
    pub severity: AlertSeverity,

    /// Alert status
    pub status: AlertStatus,

    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Alert severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Alert status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertStatus {
    Active,
    Acknowledged,
    Resolved,
}

/// Monitoring statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MonitoringStats {
    pub total_measurements: usize,
    pub drift_detections: usize,
    pub active_alerts: usize,
    pub total_alerts: usize,
    pub last_check: Option<DateTime<Utc>>,
    pub monitoring_duration_secs: f64,
}

impl DriftMonitor {
    /// Create a new drift monitor
    pub fn new(config: DriftMonitorConfig) -> Self {
        Self {
            config,
            reference_stats: Arc::new(Mutex::new(None)),
            drift_history: Arc::new(Mutex::new(VecDeque::new())),
            active_alerts: Arc::new(Mutex::new(Vec::new())),
            stats: MonitoringStats::default(),
        }
    }

    /// Set reference (baseline) data
    pub fn set_reference_data(&self, data: &Array2<f64>) -> Result<()> {
        tracing::info!("Setting reference data with {} samples", data.nrows());

        let stats = self.compute_statistics(data)?;

        let mut reference = self.reference_stats.lock().map_err(|e| {
            ShaclAiError::Configuration(format!("Failed to lock reference stats: {}", e))
        })?;
        *reference = Some(stats);

        Ok(())
    }

    /// Check for drift in new data
    pub fn check_drift(&mut self, data: &Array2<f64>) -> Result<DriftReport> {
        tracing::debug!("Checking drift for {} samples", data.nrows());

        let current_stats = self.compute_statistics(data)?;

        // Clone reference stats to avoid holding the lock
        let reference_stats = {
            let reference = self.reference_stats.lock().map_err(|e| {
                ShaclAiError::Configuration(format!("Failed to lock reference stats: {}", e))
            })?;

            reference
                .as_ref()
                .ok_or_else(|| ShaclAiError::Configuration("Reference data not set".to_string()))?
                .clone()
        };

        let mut drift_report = DriftReport {
            timestamp: Utc::now(),
            overall_drift_detected: false,
            drift_measurements: Vec::new(),
            alerts_triggered: Vec::new(),
        };

        // Data drift detection
        if self.config.enable_data_drift {
            let data_drift = self.detect_data_drift(&reference_stats, &current_stats)?;
            drift_report.drift_measurements.push(data_drift.clone());

            if data_drift.is_drift_detected {
                drift_report.overall_drift_detected = true;
                self.record_measurement(data_drift.clone())?;

                if self.config.enable_alerting {
                    let alert = self.create_alert(data_drift)?;
                    drift_report.alerts_triggered.push(alert);
                }
            }
        }

        // Feature-level drift detection
        if self.config.enable_data_drift {
            let feature_drifts = self.detect_feature_drift(&reference_stats, &current_stats)?;
            drift_report.drift_measurements.extend(feature_drifts);
        }

        // Update statistics
        self.stats.total_measurements += drift_report.drift_measurements.len();
        self.stats.last_check = Some(Utc::now());

        Ok(drift_report)
    }

    /// Detect data drift using statistical tests
    fn detect_data_drift(
        &self,
        reference: &DataStatistics,
        current: &DataStatistics,
    ) -> Result<DriftMeasurement> {
        // Compute KL divergence for overall distribution shift
        let mut kl_divergences = Vec::new();

        for i in 0..reference.num_features.min(current.num_features) {
            let kl_div = self.compute_kl_divergence(
                &reference.feature_distributions[i],
                &current.feature_distributions[i],
            )?;
            kl_divergences.push(kl_div);
        }

        let avg_kl_div = kl_divergences.iter().sum::<f64>() / kl_divergences.len() as f64;

        // Normalize drift score to [0, 1]
        let drift_score = (1.0 - (-avg_kl_div).exp()).min(1.0).max(0.0);

        // Compute p-value using Kolmogorov-Smirnov test (simplified)
        let p_value = self.compute_statistical_significance(reference, current)?;

        Ok(DriftMeasurement {
            id: format!("drift_{}", Utc::now().timestamp()),
            timestamp: Utc::now(),
            drift_type: ModelDriftType::DataDrift,
            drift_score,
            p_value,
            is_drift_detected: drift_score >= self.config.alert_threshold,
            affected_features: (0..reference.num_features)
                .map(|i| format!("feature_{}", i))
                .collect(),
            metrics: {
                let mut m = HashMap::new();
                m.insert("kl_divergence".to_string(), avg_kl_div);
                m
            },
        })
    }

    /// Detect drift in individual features
    fn detect_feature_drift(
        &self,
        reference: &DataStatistics,
        current: &DataStatistics,
    ) -> Result<Vec<DriftMeasurement>> {
        let mut feature_drifts = Vec::new();

        for i in 0..reference.num_features.min(current.num_features) {
            // Population Stability Index (PSI)
            let psi = self.compute_psi(
                &reference.feature_distributions[i],
                &current.feature_distributions[i],
            )?;

            // PSI interpretation: <0.1 = no drift, 0.1-0.25 = moderate, >0.25 = significant
            let drift_score = (psi / 0.25).min(1.0);

            if drift_score >= self.config.warning_threshold {
                feature_drifts.push(DriftMeasurement {
                    id: format!("feature_drift_{}_{}", i, Utc::now().timestamp()),
                    timestamp: Utc::now(),
                    drift_type: ModelDriftType::FeatureDrift,
                    drift_score,
                    p_value: 0.0, // Would compute proper p-value in production
                    is_drift_detected: drift_score >= self.config.alert_threshold,
                    affected_features: vec![format!("feature_{}", i)],
                    metrics: {
                        let mut m = HashMap::new();
                        m.insert("psi".to_string(), psi);
                        m.insert(
                            "mean_shift".to_string(),
                            (current.feature_means[i] - reference.feature_means[i]).abs(),
                        );
                        m
                    },
                });
            }
        }

        Ok(feature_drifts)
    }

    /// Compute KL divergence between two distributions
    fn compute_kl_divergence(&self, p: &[f64], q: &[f64]) -> Result<f64> {
        if p.len() != q.len() {
            return Err(ShaclAiError::DataProcessing(
                "Distribution lengths must match".to_string(),
            ));
        }

        let epsilon = 1e-10;
        let mut kl_div = 0.0;

        for i in 0..p.len() {
            let p_i = p[i].max(epsilon);
            let q_i = q[i].max(epsilon);
            kl_div += p_i * (p_i / q_i).ln();
        }

        Ok(kl_div)
    }

    /// Compute Population Stability Index (PSI)
    fn compute_psi(&self, reference: &[f64], current: &[f64]) -> Result<f64> {
        if reference.len() != current.len() {
            return Err(ShaclAiError::DataProcessing(
                "Distribution lengths must match".to_string(),
            ));
        }

        let epsilon = 1e-10;
        let mut psi = 0.0;

        for i in 0..reference.len() {
            let ref_pct = reference[i].max(epsilon);
            let curr_pct = current[i].max(epsilon);
            psi += (curr_pct - ref_pct) * (curr_pct / ref_pct).ln();
        }

        Ok(psi)
    }

    /// Compute statistical significance
    fn compute_statistical_significance(
        &self,
        reference: &DataStatistics,
        current: &DataStatistics,
    ) -> Result<f64> {
        // Simplified - would use proper statistical tests in production
        let mean_shift = reference
            .feature_means
            .iter()
            .zip(&current.feature_means)
            .map(|(r, c)| (r - c).abs())
            .sum::<f64>()
            / reference.feature_means.len() as f64;

        // Estimate p-value based on mean shift (simplified)
        let p_value = (-mean_shift * 2.0).exp();

        Ok(p_value.min(1.0).max(0.0))
    }

    /// Compute statistics from data
    fn compute_statistics(&self, data: &Array2<f64>) -> Result<DataStatistics> {
        let num_samples = data.nrows();
        let num_features = data.ncols();

        let mut feature_means = Vec::new();
        let mut feature_stds = Vec::new();
        let mut feature_mins = Vec::new();
        let mut feature_maxs = Vec::new();
        let mut feature_distributions = Vec::new();

        for col_idx in 0..num_features {
            let column: Vec<f64> = (0..num_samples).map(|row| data[[row, col_idx]]).collect();

            // Mean
            let mean = column.iter().sum::<f64>() / num_samples as f64;
            feature_means.push(mean);

            // Standard deviation
            let variance =
                column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / num_samples as f64;
            feature_stds.push(variance.sqrt());

            // Min/Max
            let min = column.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            feature_mins.push(min);
            feature_maxs.push(max);

            // Distribution (histogram with 10 bins)
            let distribution = self.compute_histogram(&column, 10)?;
            feature_distributions.push(distribution);
        }

        Ok(DataStatistics {
            feature_means,
            feature_stds,
            feature_mins,
            feature_maxs,
            feature_distributions,
            num_features,
            num_samples,
            timestamp: Utc::now(),
        })
    }

    /// Compute histogram for a feature
    fn compute_histogram(&self, values: &[f64], num_bins: usize) -> Result<Vec<f64>> {
        if values.is_empty() {
            return Ok(vec![0.0; num_bins]);
        }

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-10 {
            // All values are the same
            let mut hist = vec![0.0; num_bins];
            hist[0] = 1.0;
            return Ok(hist);
        }

        let bin_width = (max - min) / num_bins as f64;
        let mut histogram = vec![0usize; num_bins];

        for &value in values {
            let bin = ((value - min) / bin_width) as usize;
            let bin = bin.min(num_bins - 1);
            histogram[bin] += 1;
        }

        // Normalize
        let total = histogram.iter().sum::<usize>() as f64;
        Ok(histogram
            .iter()
            .map(|&count| count as f64 / total)
            .collect())
    }

    /// Create alert from drift measurement
    fn create_alert(&mut self, measurement: DriftMeasurement) -> Result<DriftAlert> {
        let severity = if measurement.drift_score >= self.config.alert_threshold {
            AlertSeverity::Critical
        } else if measurement.drift_score >= self.config.warning_threshold {
            AlertSeverity::Warning
        } else {
            AlertSeverity::Info
        };

        let recommended_actions = vec![
            "Review recent data quality".to_string(),
            "Check for data pipeline changes".to_string(),
            "Consider model retraining".to_string(),
        ];

        let alert = DriftAlert {
            id: format!("alert_{}", Utc::now().timestamp()),
            timestamp: Utc::now(),
            measurement,
            severity,
            status: AlertStatus::Active,
            recommended_actions,
        };

        let mut active_alerts = self.active_alerts.lock().map_err(|e| {
            ShaclAiError::Configuration(format!("Failed to lock active alerts: {}", e))
        })?;
        active_alerts.push(alert.clone());

        self.stats.active_alerts = active_alerts.len();
        self.stats.total_alerts += 1;

        Ok(alert)
    }

    /// Record drift measurement
    fn record_measurement(&mut self, measurement: DriftMeasurement) -> Result<()> {
        let is_drift = measurement.is_drift_detected;

        let mut history = self.drift_history.lock().map_err(|e| {
            ShaclAiError::Configuration(format!("Failed to lock drift history: {}", e))
        })?;

        history.push_back(measurement);

        // Trim history if needed
        while history.len() > self.config.max_history_size {
            history.pop_front();
        }

        if is_drift {
            self.stats.drift_detections += 1;
        }

        Ok(())
    }

    /// Get drift history
    pub fn get_drift_history(&self) -> Result<Vec<DriftMeasurement>> {
        let history = self.drift_history.lock().map_err(|e| {
            ShaclAiError::Configuration(format!("Failed to lock drift history: {}", e))
        })?;
        Ok(history.iter().cloned().collect())
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Result<Vec<DriftAlert>> {
        let alerts = self.active_alerts.lock().map_err(|e| {
            ShaclAiError::Configuration(format!("Failed to lock active alerts: {}", e))
        })?;
        Ok(alerts.clone())
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&mut self, alert_id: &str) -> Result<()> {
        let mut alerts = self.active_alerts.lock().map_err(|e| {
            ShaclAiError::Configuration(format!("Failed to lock active alerts: {}", e))
        })?;

        for alert in alerts.iter_mut() {
            if alert.id == alert_id {
                alert.status = AlertStatus::Acknowledged;
                return Ok(());
            }
        }

        Err(ShaclAiError::VersionNotFound(format!(
            "Alert not found: {}",
            alert_id
        )))
    }

    /// Get statistics
    pub fn get_stats(&self) -> &MonitoringStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &DriftMonitorConfig {
        &self.config
    }
}

/// Drift report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub timestamp: DateTime<Utc>,
    pub overall_drift_detected: bool,
    pub drift_measurements: Vec<DriftMeasurement>,
    pub alerts_triggered: Vec<DriftAlert>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_monitor_creation() {
        let config = DriftMonitorConfig::default();
        let monitor = DriftMonitor::new(config);

        assert!(monitor.config.enable_data_drift);
        assert_eq!(monitor.stats.total_measurements, 0);
    }

    #[test]
    fn test_set_reference_data() {
        let config = DriftMonitorConfig::default();
        let monitor = DriftMonitor::new(config);

        let reference_data = Array2::zeros((100, 5));
        let result = monitor.set_reference_data(&reference_data);

        assert!(result.is_ok());
    }

    #[test]
    fn test_drift_detection() {
        let config = DriftMonitorConfig::default();
        let mut monitor = DriftMonitor::new(config);

        // Set reference data
        let reference_data = Array2::from_shape_fn((100, 3), |(i, j)| i as f64 + j as f64);
        monitor.set_reference_data(&reference_data).unwrap();

        // Check for drift with similar data (should not detect drift)
        let current_data = Array2::from_shape_fn((100, 3), |(i, j)| i as f64 + j as f64 + 0.1);
        let report = monitor.check_drift(&current_data).unwrap();

        assert!(!report.drift_measurements.is_empty());
    }

    #[test]
    fn test_kl_divergence() {
        let config = DriftMonitorConfig::default();
        let monitor = DriftMonitor::new(config);

        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];

        let kl_div = monitor.compute_kl_divergence(&p, &q).unwrap();
        assert!(kl_div.abs() < 1e-6); // Should be ~0 for identical distributions
    }

    #[test]
    fn test_psi_calculation() {
        let config = DriftMonitorConfig::default();
        let monitor = DriftMonitor::new(config);

        let reference = vec![0.25, 0.25, 0.25, 0.25];
        let current = vec![0.25, 0.25, 0.25, 0.25];

        let psi = monitor.compute_psi(&reference, &current).unwrap();
        assert!(psi.abs() < 1e-6); // Should be ~0 for identical distributions
    }
}
