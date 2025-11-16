//! # Enterprise Monitoring and SLA Tracking
//!
//! Comprehensive monitoring system with SLA (Service Level Agreement) tracking,
//! alerting, dashboards, and performance analytics for production operations.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Enterprise monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnterpriseMonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// SLA configuration
    pub sla: SlaConfig,
    /// Alerting configuration
    pub alerting: AlertingConfig,
    /// Metrics collection configuration
    pub metrics: MetricsConfig,
    /// Health check configuration
    pub health_checks: HealthCheckConfig,
    /// Performance profiling
    pub profiling: ProfilingConfig,
}

impl Default for EnterpriseMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sla: SlaConfig::default(),
            alerting: AlertingConfig::default(),
            metrics: MetricsConfig::default(),
            health_checks: HealthCheckConfig::default(),
            profiling: ProfilingConfig::default(),
        }
    }
}

/// SLA (Service Level Agreement) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaConfig {
    /// Enable SLA tracking
    pub enabled: bool,
    /// SLA objectives
    pub objectives: Vec<SlaObjective>,
    /// Reporting interval in seconds
    pub reporting_interval_secs: u64,
    /// Breach notification
    pub breach_notification: BreachNotificationConfig,
}

impl Default for SlaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            objectives: vec![
                SlaObjective {
                    name: "Availability".to_string(),
                    metric_type: SlaMetricType::Availability,
                    target_value: 99.99,
                    measurement_window: MeasurementWindow::Rolling30Days,
                    severity: SlaSeverity::Critical,
                },
                SlaObjective {
                    name: "Latency P99".to_string(),
                    metric_type: SlaMetricType::LatencyP99,
                    target_value: 10.0, // 10ms
                    measurement_window: MeasurementWindow::Rolling24Hours,
                    severity: SlaSeverity::High,
                },
                SlaObjective {
                    name: "Error Rate".to_string(),
                    metric_type: SlaMetricType::ErrorRate,
                    target_value: 0.01, // 1% max
                    measurement_window: MeasurementWindow::Rolling1Hour,
                    severity: SlaSeverity::High,
                },
            ],
            reporting_interval_secs: 300, // 5 minutes
            breach_notification: BreachNotificationConfig::default(),
        }
    }
}

/// SLA objective definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaObjective {
    /// Objective name
    pub name: String,
    /// Metric type
    pub metric_type: SlaMetricType,
    /// Target value (percentage or milliseconds)
    pub target_value: f64,
    /// Measurement window
    pub measurement_window: MeasurementWindow,
    /// Breach severity
    pub severity: SlaSeverity,
}

/// SLA metric types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SlaMetricType {
    /// Service availability (percentage)
    Availability,
    /// Latency P50 (milliseconds)
    LatencyP50,
    /// Latency P95 (milliseconds)
    LatencyP95,
    /// Latency P99 (milliseconds)
    LatencyP99,
    /// Throughput (events per second)
    Throughput,
    /// Error rate (percentage)
    ErrorRate,
    /// Response time (milliseconds)
    ResponseTime,
}

impl fmt::Display for SlaMetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlaMetricType::Availability => write!(f, "Availability"),
            SlaMetricType::LatencyP50 => write!(f, "Latency P50"),
            SlaMetricType::LatencyP95 => write!(f, "Latency P95"),
            SlaMetricType::LatencyP99 => write!(f, "Latency P99"),
            SlaMetricType::Throughput => write!(f, "Throughput"),
            SlaMetricType::ErrorRate => write!(f, "Error Rate"),
            SlaMetricType::ResponseTime => write!(f, "Response Time"),
        }
    }
}

/// Measurement window for SLA metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MeasurementWindow {
    /// Real-time (last minute)
    RealTime,
    /// Rolling 1 hour
    Rolling1Hour,
    /// Rolling 24 hours
    Rolling24Hours,
    /// Rolling 7 days
    Rolling7Days,
    /// Rolling 30 days
    Rolling30Days,
    /// Custom duration in seconds
    Custom(u64),
}

impl fmt::Display for MeasurementWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeasurementWindow::RealTime => write!(f, "Real-time"),
            MeasurementWindow::Rolling1Hour => write!(f, "1 hour"),
            MeasurementWindow::Rolling24Hours => write!(f, "24 hours"),
            MeasurementWindow::Rolling7Days => write!(f, "7 days"),
            MeasurementWindow::Rolling30Days => write!(f, "30 days"),
            MeasurementWindow::Custom(secs) => write!(f, "{} seconds", secs),
        }
    }
}

/// SLA severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum SlaSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for SlaSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlaSeverity::Low => write!(f, "LOW"),
            SlaSeverity::Medium => write!(f, "MEDIUM"),
            SlaSeverity::High => write!(f, "HIGH"),
            SlaSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Breach notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachNotificationConfig {
    /// Enable notifications
    pub enabled: bool,
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
    /// Escalation policy
    pub escalation: EscalationPolicy,
}

impl Default for BreachNotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            channels: vec![NotificationChannel::Email {
                recipients: vec!["ops@example.com".to_string()],
            }],
            escalation: EscalationPolicy::default(),
        }
    }
}

/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email { recipients: Vec<String> },
    Slack { webhook_url: String },
    PagerDuty { service_key: String },
    Webhook { url: String },
    SMS { phone_numbers: Vec<String> },
}

/// Escalation policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            levels: vec![
                EscalationLevel {
                    level: 1,
                    wait_minutes: 5,
                    channels: vec![NotificationChannel::Email {
                        recipients: vec!["ops@example.com".to_string()],
                    }],
                },
                EscalationLevel {
                    level: 2,
                    wait_minutes: 15,
                    channels: vec![NotificationChannel::Email {
                        recipients: vec!["manager@example.com".to_string()],
                    }],
                },
            ],
        }
    }
}

/// Escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Escalation level number
    pub level: u32,
    /// Wait time before escalating (minutes)
    pub wait_minutes: u32,
    /// Notification channels for this level
    pub channels: Vec<NotificationChannel>,
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Alert aggregation window (seconds)
    pub aggregation_window_secs: u64,
    /// Deduplication enabled
    pub deduplication_enabled: bool,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: vec![],
            aggregation_window_secs: 60,
            deduplication_enabled: true,
        }
    }
}

/// Alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Condition
    pub condition: AlertCondition,
    /// Severity
    pub severity: AlertSeverity,
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
    /// Enabled
    pub enabled: bool,
}

/// Alert condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Threshold condition
    Threshold {
        metric: String,
        operator: ComparisonOperator,
        value: f64,
        duration_secs: u64,
    },
    /// Anomaly detection
    Anomaly { metric: String, sensitivity: f64 },
    /// Rate of change
    RateOfChange {
        metric: String,
        threshold_percent: f64,
        window_secs: u64,
    },
}

/// Comparison operators
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Alert severity
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Error => write!(f, "ERROR"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    /// Collection interval in seconds
    pub collection_interval_secs: u64,
    /// Metrics to collect
    pub metrics: Vec<MetricDefinition>,
    /// Export configuration
    pub export: MetricsExportConfig,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_secs: 10,
            metrics: vec![],
            export: MetricsExportConfig::default(),
        }
    }
}

/// Metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Description
    pub description: String,
    /// Labels
    pub labels: Vec<String>,
}

/// Metric types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Metrics export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExportConfig {
    /// Export format
    pub format: MetricsFormat,
    /// Export endpoints
    pub endpoints: Vec<MetricsEndpoint>,
}

impl Default for MetricsExportConfig {
    fn default() -> Self {
        Self {
            format: MetricsFormat::Prometheus,
            endpoints: vec![],
        }
    }
}

/// Metrics formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MetricsFormat {
    Prometheus,
    OpenMetrics,
    JSON,
    StatsD,
}

/// Metrics endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsEndpoint {
    /// Endpoint type
    pub endpoint_type: MetricsEndpointType,
    /// Endpoint URL
    pub url: String,
    /// Push interval in seconds (for push-based endpoints)
    pub push_interval_secs: Option<u64>,
}

/// Metrics endpoint types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MetricsEndpointType {
    /// Pull-based (HTTP server)
    Pull,
    /// Push-based (Pushgateway)
    Push,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Enable health checks
    pub enabled: bool,
    /// Health check interval in seconds
    pub interval_secs: u64,
    /// Health check timeout in seconds
    pub timeout_secs: u64,
    /// Health check endpoints
    pub endpoints: Vec<HealthCheckEndpoint>,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_secs: 30,
            timeout_secs: 5,
            endpoints: vec![],
        }
    }
}

/// Health check endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckEndpoint {
    /// Endpoint name
    pub name: String,
    /// Check type
    pub check_type: HealthCheckType,
    /// Critical (affects overall health)
    pub critical: bool,
}

/// Health check types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheckType {
    /// TCP connection check
    TcpConnect { host: String, port: u16 },
    /// HTTP endpoint check
    Http { url: String, expected_status: u16 },
    /// Database connection check
    Database { connection_string: String },
    /// Custom check
    Custom { command: String },
}

/// Performance profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enable profiling
    pub enabled: bool,
    /// CPU profiling
    pub cpu_profiling: bool,
    /// Memory profiling
    pub memory_profiling: bool,
    /// Sampling rate (samples per second)
    pub sampling_rate: u32,
    /// Profile duration in seconds
    pub duration_secs: u64,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cpu_profiling: true,
            memory_profiling: true,
            sampling_rate: 100,
            duration_secs: 30,
        }
    }
}

/// Enterprise monitoring system
pub struct EnterpriseMonitoringSystem {
    config: EnterpriseMonitoringConfig,
    sla_tracker: Arc<RwLock<SlaTracker>>,
    alert_manager: Arc<RwLock<AlertManager>>,
    metrics_collector: Arc<RwLock<MetricsCollector>>,
}

/// SLA tracker
pub struct SlaTracker {
    objectives: Vec<SlaObjective>,
    measurements: HashMap<String, Vec<SlaMeasurement>>,
    breaches: Vec<SlaBreach>,
}

/// SLA measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaMeasurement {
    /// Measurement timestamp
    pub timestamp: DateTime<Utc>,
    /// Metric type
    pub metric_type: SlaMetricType,
    /// Measured value
    pub value: f64,
    /// Meets objective
    pub meets_objective: bool,
}

/// SLA breach record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaBreach {
    /// Breach ID
    pub breach_id: String,
    /// Objective name
    pub objective_name: String,
    /// Metric type
    pub metric_type: SlaMetricType,
    /// Target value
    pub target_value: f64,
    /// Actual value
    pub actual_value: f64,
    /// Breach timestamp
    pub timestamp: DateTime<Utc>,
    /// Severity
    pub severity: SlaSeverity,
    /// Resolved
    pub resolved: bool,
    /// Resolution timestamp
    pub resolved_at: Option<DateTime<Utc>>,
}

impl SlaTracker {
    pub fn new(objectives: Vec<SlaObjective>) -> Self {
        Self {
            objectives,
            measurements: HashMap::new(),
            breaches: Vec::new(),
        }
    }

    pub fn record_measurement(&mut self, measurement: SlaMeasurement) {
        let key = measurement.metric_type.to_string();
        self.measurements.entry(key).or_default().push(measurement);
    }

    pub fn check_objectives(&mut self) -> Vec<SlaBreach> {
        let mut new_breaches = Vec::new();

        for objective in &self.objectives {
            let key = objective.metric_type.to_string();
            if let Some(measurements) = self.measurements.get(&key) {
                if let Some(latest) = measurements.last() {
                    if !latest.meets_objective {
                        new_breaches.push(SlaBreach {
                            breach_id: uuid::Uuid::new_v4().to_string(),
                            objective_name: objective.name.clone(),
                            metric_type: objective.metric_type,
                            target_value: objective.target_value,
                            actual_value: latest.value,
                            timestamp: latest.timestamp,
                            severity: objective.severity,
                            resolved: false,
                            resolved_at: None,
                        });
                    }
                }
            }
        }

        self.breaches.extend(new_breaches.clone());
        new_breaches
    }
}

/// Alert manager
pub struct AlertManager {
    rules: Vec<AlertRule>,
    active_alerts: Vec<Alert>,
}

/// Active alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub alert_id: String,
    /// Rule ID
    pub rule_id: String,
    /// Alert name
    pub name: String,
    /// Severity
    pub severity: AlertSeverity,
    /// Triggered timestamp
    pub triggered_at: DateTime<Utc>,
    /// Resolved
    pub resolved: bool,
    /// Resolved timestamp
    pub resolved_at: Option<DateTime<Utc>>,
    /// Details
    pub details: HashMap<String, String>,
}

impl AlertManager {
    pub fn new(rules: Vec<AlertRule>) -> Self {
        Self {
            rules,
            active_alerts: Vec::new(),
        }
    }

    pub fn evaluate_rules(&mut self, metrics: &HashMap<String, f64>) -> Vec<Alert> {
        let mut new_alerts = Vec::new();

        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }

            if self.should_trigger_alert(rule, metrics) {
                let alert = Alert {
                    alert_id: uuid::Uuid::new_v4().to_string(),
                    rule_id: rule.id.clone(),
                    name: rule.name.clone(),
                    severity: rule.severity,
                    triggered_at: Utc::now(),
                    resolved: false,
                    resolved_at: None,
                    details: HashMap::new(),
                };
                new_alerts.push(alert.clone());
                self.active_alerts.push(alert);
            }
        }

        new_alerts
    }

    fn should_trigger_alert(&self, _rule: &AlertRule, _metrics: &HashMap<String, f64>) -> bool {
        // Placeholder implementation
        false
    }
}

/// Metrics collector
pub struct MetricsCollector {
    metrics: HashMap<String, Vec<MetricValue>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub labels: HashMap<String, String>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    pub fn record_metric(&mut self, name: String, value: MetricValue) {
        self.metrics.entry(name).or_default().push(value);
    }

    pub fn get_latest_values(&self) -> HashMap<String, f64> {
        self.metrics
            .iter()
            .filter_map(|(name, values)| values.last().map(|v| (name.clone(), v.value)))
            .collect()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl EnterpriseMonitoringSystem {
    pub fn new(config: EnterpriseMonitoringConfig) -> Self {
        Self {
            sla_tracker: Arc::new(RwLock::new(SlaTracker::new(config.sla.objectives.clone()))),
            alert_manager: Arc::new(RwLock::new(AlertManager::new(
                config.alerting.rules.clone(),
            ))),
            metrics_collector: Arc::new(RwLock::new(MetricsCollector::new())),
            config,
        }
    }

    pub async fn initialize(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Enterprise monitoring is disabled");
            return Ok(());
        }

        info!("Initializing enterprise monitoring system");
        Ok(())
    }

    pub async fn record_sla_measurement(&self, measurement: SlaMeasurement) -> Result<()> {
        let mut tracker = self.sla_tracker.write().await;
        tracker.record_measurement(measurement);

        let breaches = tracker.check_objectives();
        if !breaches.is_empty() {
            warn!("SLA breaches detected: {}", breaches.len());
            for breach in &breaches {
                error!(
                    "SLA breach: {} - {} (target: {}, actual: {})",
                    breach.objective_name,
                    breach.metric_type,
                    breach.target_value,
                    breach.actual_value
                );
            }
        }

        Ok(())
    }

    pub async fn get_sla_status(&self) -> Result<SlaStatus> {
        let tracker = self.sla_tracker.read().await;
        Ok(SlaStatus {
            total_objectives: tracker.objectives.len() as u64,
            objectives_met: 0, // Placeholder
            objectives_breached: tracker.breaches.len() as u64,
            active_breaches: tracker.breaches.iter().filter(|b| !b.resolved).count() as u64,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaStatus {
    pub total_objectives: u64,
    pub objectives_met: u64,
    pub objectives_breached: u64,
    pub active_breaches: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitoring_config_default() {
        let config = EnterpriseMonitoringConfig::default();
        assert!(config.enabled);
        assert!(config.sla.enabled);
    }

    #[tokio::test]
    async fn test_sla_severity_ordering() {
        assert!(SlaSeverity::Critical > SlaSeverity::High);
        assert!(SlaSeverity::High > SlaSeverity::Medium);
        assert!(SlaSeverity::Medium > SlaSeverity::Low);
    }

    #[tokio::test]
    async fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Critical > AlertSeverity::Error);
        assert!(AlertSeverity::Error > AlertSeverity::Warning);
        assert!(AlertSeverity::Warning > AlertSeverity::Info);
    }
}
