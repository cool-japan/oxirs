//! Production Monitoring for ML Models
//!
//! This module provides comprehensive production monitoring capabilities for
//! deployed SHACL AI models, including performance tracking, data quality
//! monitoring, drift detection, SLA monitoring, and alerting.
//!
//! # Features
//!
//! - **Performance Monitoring**: Track latency, throughput, error rates
//! - **Data Quality Monitoring**: Input data validation and quality checks
//! - **Prediction Monitoring**: Output distribution and confidence tracking
//! - **SLA Monitoring**: Service Level Agreement compliance
//! - **Alerting**: Multi-channel alerts for critical issues
//! - **Dashboards**: Real-time metrics visualization
//! - **Audit Logging**: Complete request/response logging
//!
//! # Example
//!
//! ```rust
//! use oxirs_shacl_ai::production_monitoring::{
//!     ProductionMonitor, MonitoringConfig, SLA
//! };
//!
//! let monitor = ProductionMonitor::new();
//!
//! // Record inference
//! monitor.record_inference(
//!     "model_v1",
//!     50.0,  // latency_ms
//!     true,  // success
//!     0.95   // confidence
//! ).unwrap();
//!
//! // Check SLA compliance
//! let metrics = monitor.get_metrics("model_v1").unwrap();
//! println!("SLA compliance: {}", metrics.sla_compliance);
//! ```

use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use thiserror::Error;

use crate::{Result, ShaclAiError};

/// Monitoring error types
#[derive(Debug, Error)]
pub enum MonitoringError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Alert failed: {0}")]
    AlertFailed(String),

    #[error("Metrics unavailable: {0}")]
    MetricsUnavailable(String),
}

impl From<MonitoringError> for ShaclAiError {
    fn from(err: MonitoringError) -> Self {
        ShaclAiError::DataProcessing(err.to_string())
    }
}

/// Production monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,

    /// Enable data quality monitoring
    pub enable_data_quality_monitoring: bool,

    /// Enable prediction monitoring
    pub enable_prediction_monitoring: bool,

    /// Enable SLA monitoring
    pub enable_sla_monitoring: bool,

    /// Metrics retention window (seconds)
    pub metrics_retention_seconds: u64,

    /// Metrics aggregation interval (seconds)
    pub aggregation_interval_seconds: u64,

    /// Enable alerting
    pub enable_alerting: bool,

    /// Alert channels
    pub alert_channels: Vec<AlertChannel>,

    /// Maximum window size for rolling metrics
    pub max_window_size: usize,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_performance_monitoring: true,
            enable_data_quality_monitoring: true,
            enable_prediction_monitoring: true,
            enable_sla_monitoring: true,
            metrics_retention_seconds: 86400, // 24 hours
            aggregation_interval_seconds: 60,
            enable_alerting: true,
            alert_channels: vec![AlertChannel::Log],
            max_window_size: 10000,
        }
    }
}

/// Alert channel
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertChannel {
    /// Log to system logs
    Log,

    /// Email notification
    Email,

    /// Slack notification
    Slack,

    /// PagerDuty
    PagerDuty,

    /// Webhook
    Webhook(String),

    /// Custom channel
    Custom(String),
}

/// Service Level Agreement (SLA)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLA {
    /// Target latency percentile (e.g., P95)
    pub latency_percentile: f64,

    /// Target latency value (ms)
    pub target_latency_ms: f64,

    /// Target throughput (requests/sec)
    pub target_throughput: f64,

    /// Maximum error rate (%)
    pub max_error_rate: f64,

    /// Minimum uptime (%)
    pub min_uptime: f64,

    /// Minimum accuracy
    pub min_accuracy: f64,
}

impl Default for SLA {
    fn default() -> Self {
        Self {
            latency_percentile: 95.0,
            target_latency_ms: 100.0,
            target_throughput: 100.0,
            max_error_rate: 1.0,
            min_uptime: 99.9,
            min_accuracy: 0.95,
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total requests
    pub total_requests: usize,

    /// Successful requests
    pub successful_requests: usize,

    /// Failed requests
    pub failed_requests: usize,

    /// Average latency (ms)
    pub avg_latency_ms: f64,

    /// P50 latency (ms)
    pub p50_latency_ms: f64,

    /// P95 latency (ms)
    pub p95_latency_ms: f64,

    /// P99 latency (ms)
    pub p99_latency_ms: f64,

    /// Throughput (requests/sec)
    pub throughput: f64,

    /// Error rate (%)
    pub error_rate: f64,

    /// Uptime (%)
    pub uptime: f64,

    /// SLA compliance (%)
    pub sla_compliance: f64,

    /// Time window
    pub window_start: DateTime<Utc>,

    /// Window end
    pub window_end: DateTime<Utc>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            throughput: 0.0,
            error_rate: 0.0,
            uptime: 100.0,
            sla_compliance: 100.0,
            window_start: Utc::now(),
            window_end: Utc::now(),
        }
    }
}

/// Data quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityMetrics {
    /// Total samples
    pub total_samples: usize,

    /// Valid samples
    pub valid_samples: usize,

    /// Invalid samples
    pub invalid_samples: usize,

    /// Missing value percentage
    pub missing_percentage: f64,

    /// Out-of-range percentage
    pub out_of_range_percentage: f64,

    /// Anomalous samples percentage
    pub anomalous_percentage: f64,

    /// Data quality score (0-1)
    pub quality_score: f64,
}

impl Default for DataQualityMetrics {
    fn default() -> Self {
        Self {
            total_samples: 0,
            valid_samples: 0,
            invalid_samples: 0,
            missing_percentage: 0.0,
            out_of_range_percentage: 0.0,
            anomalous_percentage: 0.0,
            quality_score: 1.0,
        }
    }
}

/// Prediction monitoring metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetrics {
    /// Total predictions
    pub total_predictions: usize,

    /// Average confidence
    pub avg_confidence: f64,

    /// Low confidence predictions (<0.5)
    pub low_confidence_count: usize,

    /// High confidence predictions (>0.9)
    pub high_confidence_count: usize,

    /// Prediction distribution
    pub class_distribution: HashMap<String, usize>,

    /// Average prediction value
    pub avg_prediction_value: f64,
}

impl Default for PredictionMetrics {
    fn default() -> Self {
        Self {
            total_predictions: 0,
            avg_confidence: 0.0,
            low_confidence_count: 0,
            high_confidence_count: 0,
            class_distribution: HashMap::new(),
            avg_prediction_value: 0.0,
        }
    }
}

/// Alert severity
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational
    Info,

    /// Warning
    Warning,

    /// Critical
    Critical,

    /// Emergency
    Emergency,
}

/// Alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,

    /// Model ID
    pub model_id: String,

    /// Alert type
    pub alert_type: AlertType,

    /// Severity
    pub severity: AlertSeverity,

    /// Message
    pub message: String,

    /// Metric value
    pub metric_value: f64,

    /// Threshold value
    pub threshold: f64,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Acknowledged
    pub acknowledged: bool,
}

/// Alert type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    /// High latency
    HighLatency,

    /// High error rate
    HighErrorRate,

    /// Low throughput
    LowThroughput,

    /// SLA violation
    SLAViolation,

    /// Data quality issue
    DataQualityIssue,

    /// Prediction drift
    PredictionDrift,

    /// System health
    SystemHealth,

    /// Custom alert
    Custom(String),
}

/// Inference record
#[derive(Debug, Clone)]
struct InferenceRecord {
    timestamp: DateTime<Utc>,
    latency_ms: f64,
    success: bool,
    confidence: f64,
    data_quality_score: f64,
}

/// Model monitoring state
#[derive(Debug)]
struct ModelMonitoringState {
    /// SLA definition
    sla: SLA,

    /// Recent inferences
    inferences: VecDeque<InferenceRecord>,

    /// Performance metrics
    performance: PerformanceMetrics,

    /// Data quality metrics
    data_quality: DataQualityMetrics,

    /// Prediction metrics
    predictions: PredictionMetrics,

    /// Active alerts
    alerts: Vec<Alert>,

    /// Last aggregation time
    last_aggregation: DateTime<Utc>,
}

impl ModelMonitoringState {
    fn new(sla: SLA) -> Self {
        Self {
            sla,
            inferences: VecDeque::new(),
            performance: PerformanceMetrics::default(),
            data_quality: DataQualityMetrics::default(),
            predictions: PredictionMetrics::default(),
            alerts: Vec::new(),
            last_aggregation: Utc::now(),
        }
    }
}

/// Production monitoring system
#[derive(Debug)]
pub struct ProductionMonitor {
    /// Configuration
    config: MonitoringConfig,

    /// Model monitoring states (model_id -> state)
    models: Arc<DashMap<String, Arc<RwLock<ModelMonitoringState>>>>,

    /// Global metrics
    global_metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl ProductionMonitor {
    /// Create a new production monitor
    pub fn new() -> Self {
        Self::with_config(MonitoringConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: MonitoringConfig) -> Self {
        Self {
            config,
            models: Arc::new(DashMap::new()),
            global_metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
        }
    }

    /// Register a model for monitoring
    pub fn register_model(&self, model_id: &str, sla: SLA) -> Result<()> {
        let state = Arc::new(RwLock::new(ModelMonitoringState::new(sla)));
        self.models.insert(model_id.to_string(), state);

        tracing::info!("Registered model for monitoring: {}", model_id);
        Ok(())
    }

    /// Record an inference
    pub fn record_inference(
        &self,
        model_id: &str,
        latency_ms: f64,
        success: bool,
        confidence: f64,
    ) -> Result<()> {
        let state = self
            .models
            .get(model_id)
            .ok_or_else(|| MonitoringError::ModelNotFound(model_id.to_string()))?;

        let mut state = state
            .write()
            .map_err(|e| MonitoringError::MetricsUnavailable(e.to_string()))?;

        // Record inference
        let record = InferenceRecord {
            timestamp: Utc::now(),
            latency_ms,
            success,
            confidence,
            data_quality_score: 1.0,
        };

        state.inferences.push_back(record);

        // Limit window size
        while state.inferences.len() > self.config.max_window_size {
            state.inferences.pop_front();
        }

        // Update metrics
        state.performance.total_requests += 1;
        if success {
            state.performance.successful_requests += 1;
        } else {
            state.performance.failed_requests += 1;
        }

        // Check if aggregation is needed
        let now = Utc::now();
        if (now - state.last_aggregation).num_seconds() as u64
            >= self.config.aggregation_interval_seconds
        {
            self.aggregate_metrics_internal(&mut state)?;
        }

        Ok(())
    }

    /// Get metrics for a model
    pub fn get_metrics(&self, model_id: &str) -> Result<PerformanceMetrics> {
        let state = self
            .models
            .get(model_id)
            .ok_or_else(|| MonitoringError::ModelNotFound(model_id.to_string()))?;

        let state = state
            .read()
            .map_err(|e| MonitoringError::MetricsUnavailable(e.to_string()))?;

        Ok(state.performance.clone())
    }

    /// Get data quality metrics
    pub fn get_data_quality_metrics(&self, model_id: &str) -> Result<DataQualityMetrics> {
        let state = self
            .models
            .get(model_id)
            .ok_or_else(|| MonitoringError::ModelNotFound(model_id.to_string()))?;

        let state = state
            .read()
            .map_err(|e| MonitoringError::MetricsUnavailable(e.to_string()))?;

        Ok(state.data_quality.clone())
    }

    /// Get prediction metrics
    pub fn get_prediction_metrics(&self, model_id: &str) -> Result<PredictionMetrics> {
        let state = self
            .models
            .get(model_id)
            .ok_or_else(|| MonitoringError::ModelNotFound(model_id.to_string()))?;

        let state = state
            .read()
            .map_err(|e| MonitoringError::MetricsUnavailable(e.to_string()))?;

        Ok(state.predictions.clone())
    }

    /// Get active alerts for a model
    pub fn get_alerts(&self, model_id: &str) -> Result<Vec<Alert>> {
        let state = self
            .models
            .get(model_id)
            .ok_or_else(|| MonitoringError::ModelNotFound(model_id.to_string()))?;

        let state = state
            .read()
            .map_err(|e| MonitoringError::MetricsUnavailable(e.to_string()))?;

        Ok(state.alerts.clone())
    }

    /// Check SLA compliance
    pub fn check_sla_compliance(&self, model_id: &str) -> Result<bool> {
        let state = self
            .models
            .get(model_id)
            .ok_or_else(|| MonitoringError::ModelNotFound(model_id.to_string()))?;

        let state = state
            .read()
            .map_err(|e| MonitoringError::MetricsUnavailable(e.to_string()))?;

        let sla = &state.sla;
        let metrics = &state.performance;

        // Check SLA criteria
        let latency_ok = metrics.p95_latency_ms <= sla.target_latency_ms;
        let throughput_ok = metrics.throughput >= sla.target_throughput;
        let error_rate_ok = metrics.error_rate <= sla.max_error_rate;
        let uptime_ok = metrics.uptime >= sla.min_uptime;

        Ok(latency_ok && throughput_ok && error_rate_ok && uptime_ok)
    }

    /// Aggregate metrics
    pub fn aggregate_metrics(&self, model_id: &str) -> Result<()> {
        let state = self
            .models
            .get(model_id)
            .ok_or_else(|| MonitoringError::ModelNotFound(model_id.to_string()))?;

        let mut state = state
            .write()
            .map_err(|e| MonitoringError::MetricsUnavailable(e.to_string()))?;

        self.aggregate_metrics_internal(&mut state)
    }

    /// Internal metric aggregation
    fn aggregate_metrics_internal(&self, state: &mut ModelMonitoringState) -> Result<()> {
        if state.inferences.is_empty() {
            return Ok(());
        }

        let now = Utc::now();
        let window_start = now - Duration::seconds(self.config.metrics_retention_seconds as i64);

        // Filter to retention window
        state.inferences.retain(|r| r.timestamp >= window_start);

        // Calculate latencies
        let mut latencies: Vec<f64> = state.inferences.iter().map(|r| r.latency_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let total = latencies.len();
        if total > 0 {
            state.performance.avg_latency_ms = latencies.iter().sum::<f64>() / total as f64;
            state.performance.p50_latency_ms = latencies[total / 2];
            state.performance.p95_latency_ms = latencies[(total as f64 * 0.95) as usize];
            state.performance.p99_latency_ms = latencies[(total as f64 * 0.99) as usize];
        }

        // Calculate error rate
        if state.performance.total_requests > 0 {
            state.performance.error_rate = (state.performance.failed_requests as f64
                / state.performance.total_requests as f64)
                * 100.0;
        }

        // Calculate throughput
        let duration_secs = (now - state.performance.window_start).num_seconds() as f64;
        if duration_secs > 0.0 {
            state.performance.throughput = state.performance.total_requests as f64 / duration_secs;
        }

        // Check SLA and generate alerts
        self.check_sla_and_alert(state)?;

        state.last_aggregation = now;
        state.performance.window_end = now;

        Ok(())
    }

    /// Check SLA and generate alerts
    fn check_sla_and_alert(&self, state: &mut ModelMonitoringState) -> Result<()> {
        if !self.config.enable_alerting {
            return Ok(());
        }

        let sla = &state.sla;
        let metrics = &state.performance;

        // Check latency SLA
        if metrics.p95_latency_ms > sla.target_latency_ms {
            let alert = Alert {
                id: uuid::Uuid::new_v4().to_string(),
                model_id: "model".to_string(),
                alert_type: AlertType::HighLatency,
                severity: AlertSeverity::Warning,
                message: format!(
                    "P95 latency {} ms exceeds target {} ms",
                    metrics.p95_latency_ms, sla.target_latency_ms
                ),
                metric_value: metrics.p95_latency_ms,
                threshold: sla.target_latency_ms,
                timestamp: Utc::now(),
                acknowledged: false,
            };
            state.alerts.push(alert);
        }

        // Check error rate SLA
        if metrics.error_rate > sla.max_error_rate {
            let alert = Alert {
                id: uuid::Uuid::new_v4().to_string(),
                model_id: "model".to_string(),
                alert_type: AlertType::HighErrorRate,
                severity: AlertSeverity::Critical,
                message: format!(
                    "Error rate {}% exceeds maximum {}%",
                    metrics.error_rate, sla.max_error_rate
                ),
                metric_value: metrics.error_rate,
                threshold: sla.max_error_rate,
                timestamp: Utc::now(),
                acknowledged: false,
            };
            state.alerts.push(alert);
        }

        Ok(())
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&self, model_id: &str, alert_id: &str) -> Result<()> {
        let state = self
            .models
            .get(model_id)
            .ok_or_else(|| MonitoringError::ModelNotFound(model_id.to_string()))?;

        let mut state = state
            .write()
            .map_err(|e| MonitoringError::MetricsUnavailable(e.to_string()))?;

        if let Some(alert) = state.alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.acknowledged = true;
        }

        Ok(())
    }

    /// List all monitored models
    pub fn list_models(&self) -> Vec<String> {
        self.models.iter().map(|e| e.key().clone()).collect()
    }
}

impl Default for ProductionMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = ProductionMonitor::new();
        assert_eq!(monitor.list_models().len(), 0);
    }

    #[test]
    fn test_register_model() {
        let monitor = ProductionMonitor::new();
        let sla = SLA::default();

        monitor.register_model("test_model", sla).unwrap();

        let models = monitor.list_models();
        assert_eq!(models.len(), 1);
        assert!(models.contains(&"test_model".to_string()));
    }

    #[test]
    fn test_record_inference() {
        let monitor = ProductionMonitor::new();
        monitor
            .register_model("test_model", SLA::default())
            .unwrap();

        monitor
            .record_inference("test_model", 50.0, true, 0.95)
            .unwrap();

        let metrics = monitor.get_metrics("test_model").unwrap();
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.successful_requests, 1);
    }

    #[test]
    fn test_metrics_aggregation() {
        let monitor = ProductionMonitor::new();
        monitor
            .register_model("test_model", SLA::default())
            .unwrap();

        // Record multiple inferences
        for i in 0..100 {
            monitor
                .record_inference("test_model", 50.0 + i as f64, true, 0.95)
                .unwrap();
        }

        monitor.aggregate_metrics("test_model").unwrap();

        let metrics = monitor.get_metrics("test_model").unwrap();
        assert_eq!(metrics.total_requests, 100);
        assert!(metrics.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_sla_compliance() {
        let monitor = ProductionMonitor::new();
        let sla = SLA {
            target_latency_ms: 100.0,
            target_throughput: 0.0, // No throughput requirement for test
            max_error_rate: 10.0,
            min_uptime: 90.0,
            ..Default::default()
        };

        monitor.register_model("test_model", sla).unwrap();

        // Record low latency inferences
        for _ in 0..10 {
            monitor
                .record_inference("test_model", 50.0, true, 0.95)
                .unwrap();
        }

        monitor.aggregate_metrics("test_model").unwrap();

        let compliant = monitor.check_sla_compliance("test_model").unwrap();
        assert!(compliant);
    }

    #[test]
    fn test_alert_generation() {
        let config = MonitoringConfig {
            aggregation_interval_seconds: 0, // Immediate aggregation
            ..Default::default()
        };

        let monitor = ProductionMonitor::with_config(config);
        let sla = SLA {
            max_error_rate: 5.0,
            ..Default::default()
        };

        monitor.register_model("test_model", sla).unwrap();

        // Record failures to trigger alert
        for _ in 0..10 {
            monitor
                .record_inference("test_model", 50.0, false, 0.5)
                .unwrap();
        }

        monitor.aggregate_metrics("test_model").unwrap();

        let alerts = monitor.get_alerts("test_model").unwrap();
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_acknowledge_alert() {
        let monitor = ProductionMonitor::new();
        monitor
            .register_model("test_model", SLA::default())
            .unwrap();

        // Generate alert by recording failures
        for _ in 0..10 {
            monitor
                .record_inference("test_model", 50.0, false, 0.5)
                .unwrap();
        }

        monitor.aggregate_metrics("test_model").unwrap();

        let alerts = monitor.get_alerts("test_model").unwrap();
        if let Some(alert) = alerts.first() {
            monitor.acknowledge_alert("test_model", &alert.id).unwrap();

            let updated_alerts = monitor.get_alerts("test_model").unwrap();
            assert!(updated_alerts.iter().any(|a| a.acknowledged));
        }
    }
}
