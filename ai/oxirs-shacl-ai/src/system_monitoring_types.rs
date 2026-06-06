//! Shared types for the system monitoring subsystem.
//!
//! Configuration, metrics, alerts, dashboard, health checks and notification
//! primitives used by [`crate::system_monitoring`] and its sibling modules.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::forecasting_models::{TimeSeries, TimeSeriesDataPoint};
use crate::Result;

/// Configuration for system monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_real_time: bool,
    pub collection_interval_secs: u64,
    pub retention_days: u32,
    pub enable_alerting: bool,
    pub health_check_interval_secs: u64,
    pub enable_performance_tracking: bool,
    pub enable_quality_tracking: bool,
    pub enable_error_tracking: bool,
    pub dashboard_refresh_secs: u64,
    pub max_buffer_size: usize,
    pub enable_predictive_alerts: bool,
    pub alert_thresholds: AlertThresholds,
    pub notification_channels: Vec<NotificationChannel>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_real_time: true,
            collection_interval_secs: 60,
            retention_days: 30,
            enable_alerting: true,
            health_check_interval_secs: 30,
            enable_performance_tracking: true,
            enable_quality_tracking: true,
            enable_error_tracking: true,
            dashboard_refresh_secs: 30,
            max_buffer_size: 10000,
            enable_predictive_alerts: true,
            alert_thresholds: AlertThresholds::default(),
            notification_channels: vec![NotificationChannel::Console],
        }
    }
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub max_response_time_ms: f64,
    pub min_throughput: f64,
    pub max_error_rate: f64,
    pub max_memory_usage_percent: f64,
    pub max_cpu_usage_percent: f64,
    pub min_quality_score: f64,
    pub max_quality_degradation_percent: f64,
    pub max_consecutive_failures: u32,
    pub min_uptime_percent: f64,
    pub prediction_confidence_threshold: f64,
    pub trend_significance_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_response_time_ms: 5000.0,
            min_throughput: 100.0,
            max_error_rate: 0.05,
            max_memory_usage_percent: 85.0,
            max_cpu_usage_percent: 80.0,
            min_quality_score: 0.8,
            max_quality_degradation_percent: 10.0,
            max_consecutive_failures: 5,
            min_uptime_percent: 99.0,
            prediction_confidence_threshold: 0.7,
            trend_significance_threshold: 0.8,
        }
    }
}

/// Notification channels for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Console,
    Email(String),
    Slack(String),
    Webhook(String),
    SMS(String),
    PagerDuty(String),
}

/// Metrics collection engine
#[derive(Debug)]
pub struct MetricsCollector {
    pub(crate) performance_metrics: VecDeque<PerformanceMetric>,
    pub(crate) quality_metrics: VecDeque<QualityMetric>,
    pub(crate) error_metrics: VecDeque<ErrorMetric>,
    pub(crate) system_metrics: VecDeque<SystemMetric>,
    pub(crate) custom_metrics: HashMap<String, VecDeque<CustomMetric>>,
    pub(crate) collection_start_time: Instant,
    pub(crate) last_collection_time: Option<Instant>,
}

/// Performance metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub timestamp: DateTime<Utc>,
    pub response_time_ms: f64,
    pub throughput: f64,
    pub concurrent_requests: u32,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub disk_io_mb_s: f64,
    pub network_io_mb_s: f64,
    pub gc_time_ms: Option<f64>,
    pub cache_hit_rate: Option<f64>,
    pub tags: HashMap<String, String>,
}

/// Quality metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetric {
    pub timestamp: DateTime<Utc>,
    pub overall_quality_score: f64,
    pub completeness_score: f64,
    pub consistency_score: f64,
    pub accuracy_score: f64,
    pub validity_score: f64,
    pub timeliness_score: f64,
    pub data_volume: u64,
    pub shapes_validated: u32,
    pub constraints_checked: u32,
    pub quality_trend: QualityTrend,
    pub degradation_indicators: Vec<String>,
}

/// Quality trend indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityTrend {
    Improving,
    Stable,
    Degrading,
    Critical,
    Unknown,
}

/// Error metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetric {
    pub timestamp: DateTime<Utc>,
    pub error_type: ErrorType,
    pub error_count: u32,
    pub error_rate: f64,
    pub severity: ErrorSeverity,
    pub source_component: String,
    pub error_message: String,
    pub stack_trace: Option<String>,
    pub resolution_time_ms: Option<f64>,
    pub impact_scope: ErrorImpactScope,
    pub recovery_action: Option<String>,
}

/// Types of errors tracked
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    ValidationError,
    PerformanceTimeout,
    SystemFailure,
    DataQualityIssue,
    ConfigurationError,
    SecurityViolation,
    ResourceExhaustion,
    NetworkError,
    DatabaseError,
    Unknown,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Fatal,
}

/// Error impact scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorImpactScope {
    Single,
    Component,
    System,
    Cluster,
    Global,
}

/// System resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetric {
    pub timestamp: DateTime<Utc>,
    pub cpu_cores: u32,
    pub total_memory_mb: f64,
    pub available_memory_mb: f64,
    pub disk_total_gb: f64,
    pub disk_available_gb: f64,
    pub network_connections: u32,
    pub uptime_seconds: u64,
    pub load_average: Vec<f64>,
    pub swap_usage_mb: f64,
    pub process_count: u32,
    pub thread_count: u32,
}

/// Custom metric for application-specific tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub labels: HashMap<String, String>,
    pub description: String,
}

/// Alert management system
#[derive(Debug)]
pub struct AlertManager {
    pub(crate) active_alerts: HashMap<String, Alert>,
    pub(crate) alert_history: VecDeque<AlertHistoryEntry>,
    pub(crate) alert_rules: Vec<AlertRule>,
    pub(crate) notification_queue: VecDeque<AlertNotification>,
    pub(crate) suppression_rules: Vec<SuppressionRule>,
}

/// Alert definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub title: String,
    pub description: String,
    pub triggered_at: DateTime<Utc>,
    pub metric_value: f64,
    pub threshold_value: f64,
    pub source_metric: String,
    pub related_components: Vec<String>,
    pub suggested_actions: Vec<String>,
    pub auto_resolve: bool,
    pub escalation_level: u32,
    pub tags: HashMap<String, String>,
}

/// Types of alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    Performance,
    Quality,
    Error,
    System,
    Security,
    Predictive,
    Custom,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub duration_secs: u64,
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub notification_channels: Vec<NotificationChannel>,
    pub auto_resolve: bool,
    pub escalation_rules: Vec<EscalationRule>,
}

/// Alert condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    PercentageChange,
    Trend,
    Anomaly,
    Threshold,
}

/// Escalation rule for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRule {
    pub delay_secs: u64,
    pub severity: AlertSeverity,
    pub notification_channels: Vec<NotificationChannel>,
    pub actions: Vec<String>,
}

/// Alert history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertHistoryEntry {
    pub alert: Alert,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolution_method: Option<ResolutionMethod>,
    pub notifications_sent: u32,
    pub escalations: u32,
    pub mean_time_to_resolution: Option<Duration>,
}

/// How an alert was resolved
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionMethod {
    AutoResolved,
    ManualResolved,
    SuppressedResolved,
    TimeoutResolved,
}

/// Alert notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertNotification {
    pub id: String,
    pub alert_id: String,
    pub channel: NotificationChannel,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub status: NotificationStatus,
    pub retry_count: u32,
    pub delivery_time: Option<DateTime<Utc>>,
}

/// Notification delivery status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationStatus {
    Pending,
    Sent,
    Delivered,
    Failed,
    Retrying,
}

/// Alert suppression rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionRule {
    pub id: String,
    pub name: String,
    pub condition: SuppressionCondition,
    pub duration_secs: u64,
    pub enabled: bool,
}

/// Suppression conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuppressionCondition {
    MaintenanceWindow,
    ComponentMaintenance,
    AlertFlood,
    DependencyDown,
    Custom(String),
}

/// Real-time monitoring dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringDashboard {
    pub last_updated: DateTime<Utc>,
    pub overall_health: SystemHealth,
    pub performance_summary: PerformanceSummary,
    pub quality_summary: QualitySummary,
    pub error_summary: ErrorSummary,
    pub active_alerts: Vec<Alert>,
    pub trends: TrendAnalysis,
    pub recommendations: Vec<MonitoringRecommendation>,
    pub uptime_stats: UptimeStatistics,
    pub resource_utilization: ResourceUtilization,
}

/// Overall system health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemHealth {
    Healthy,
    Warning,
    Critical,
    Degraded,
    Unknown,
}

/// Performance summary for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub avg_response_time_ms: f64,
    pub current_throughput: f64,
    pub peak_throughput_24h: f64,
    pub error_rate_percent: f64,
    pub availability_percent: f64,
    pub response_time_trend: TrendDirection,
    pub throughput_trend: TrendDirection,
    pub performance_score: f64,
}

/// Quality summary for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySummary {
    pub overall_quality_score: f64,
    pub quality_trend: QualityTrend,
    pub issues_detected: u32,
    pub shapes_validated_24h: u32,
    pub quality_improvement_percent: f64,
    pub data_completeness_percent: f64,
    pub consistency_score: f64,
}

/// Error summary for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSummary {
    pub total_errors_24h: u32,
    pub critical_errors_24h: u32,
    pub error_rate_trend: TrendDirection,
    pub mean_time_to_resolution_min: f64,
    pub most_frequent_error_type: ErrorType,
    pub resolved_errors_24h: u32,
    pub unresolved_errors: u32,
}

/// Trend direction indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
    Unknown,
}

/// Trend analysis for various metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub performance_trends: HashMap<String, TrendData>,
    pub quality_trends: HashMap<String, TrendData>,
    pub error_trends: HashMap<String, TrendData>,
    pub usage_trends: HashMap<String, TrendData>,
    pub seasonal_patterns: Vec<SeasonalPattern>,
    pub anomalies_detected: Vec<AnomalyDetection>,
}

/// Trend data for a specific metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendData {
    pub direction: TrendDirection,
    pub change_percent: f64,
    pub confidence: f64,
    pub prediction_7d: f64,
    pub prediction_30d: f64,
    pub significance: f64,
    pub correlation_factors: Vec<String>,
}

/// Seasonal pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    pub metric_name: String,
    pub pattern_type: PatternType,
    pub period_hours: f64,
    pub amplitude: f64,
    pub confidence: f64,
    pub next_peak: DateTime<Utc>,
    pub next_trough: DateTime<Utc>,
}

/// Types of patterns detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Daily,
    Weekly,
    Monthly,
    Custom(f64),
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    pub metric_name: String,
    pub detected_at: DateTime<Utc>,
    pub anomaly_type: AnomalyType,
    pub severity: f64,
    pub expected_value: f64,
    pub actual_value: f64,
    pub confidence: f64,
    pub context: HashMap<String, String>,
}

/// Types of anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    PositiveSpike,
    NegativeDip,
    TrendChange,
    PatternDisruption,
    CorrelationBreakdown,
    SeasonalDeviation,
}

/// Monitoring recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringRecommendation {
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: RecommendationCategory,
    pub priority: u8,
    pub impact: String,
    pub actions: Vec<String>,
    pub estimated_effort: String,
    pub created_at: DateTime<Utc>,
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Performance,
    Reliability,
    Security,
    Cost,
    Maintenance,
    Scaling,
}

/// Uptime statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UptimeStatistics {
    pub current_uptime_secs: u64,
    pub uptime_percent_24h: f64,
    pub uptime_percent_7d: f64,
    pub uptime_percent_30d: f64,
    pub longest_uptime_secs: u64,
    pub total_downtime_24h_secs: u64,
    pub outage_count_24h: u32,
    pub mean_time_between_failures_hours: f64,
}

/// Resource utilization summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_usage_mb_s: f64,
    pub connection_count: u32,
    pub thread_count: u32,
    pub file_descriptor_count: u32,
    pub resource_trends: HashMap<String, TrendDirection>,
}

/// Health checker for system components
#[derive(Debug)]
pub struct HealthChecker {
    pub(crate) component_status: HashMap<String, ComponentHealth>,
    pub(crate) health_checks: Vec<HealthCheck>,
    pub(crate) last_check_time: Option<Instant>,
    pub(crate) check_history: VecDeque<HealthCheckResult>,
    pub(crate) total_checks_performed: Arc<AtomicU32>,
}

/// Component health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub last_check: DateTime<Utc>,
    pub response_time_ms: f64,
    pub uptime_percent: f64,
    pub error_count: u32,
    pub dependencies: Vec<String>,
    pub health_score: f64,
}

/// Health status levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
    Maintenance,
}

/// Health check definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub component: String,
    pub check_type: HealthCheckType,
    pub interval_secs: u64,
    pub timeout_secs: u64,
    pub enabled: bool,
    pub critical: bool,
}

/// Types of health checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheckType {
    Ping,
    HttpEndpoint(String),
    DatabaseConnection,
    FileSystem,
    Memory,
    CPU,
    Custom(String),
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub check_name: String,
    pub component: String,
    pub timestamp: DateTime<Utc>,
    pub status: HealthStatus,
    pub response_time_ms: f64,
    pub message: String,
    pub error: Option<String>,
}

/// Monitoring data storage
#[derive(Debug)]
pub struct MonitoringStorage {
    pub(crate) metrics_buffer: HashMap<String, VecDeque<TimeSeriesDataPoint>>,
    pub(crate) aggregated_data: HashMap<String, TimeSeries>,
    pub(crate) retention_policy: RetentionPolicy,
    pub(crate) compression_enabled: bool,
    pub(crate) backup_enabled: bool,
}

/// Data retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub raw_data_days: u32,
    pub hourly_aggregation_days: u32,
    pub daily_aggregation_days: u32,
    pub monthly_aggregation_days: u32,
    pub compression_threshold_days: u32,
}

/// Notification engine for alerts
#[derive(Debug)]
pub struct NotificationEngine {
    pub(crate) channels: HashMap<String, Box<dyn NotificationSender>>,
    pub(crate) templates: HashMap<String, NotificationTemplate>,
    pub(crate) delivery_status: HashMap<String, NotificationStatus>,
    pub(crate) rate_limits: HashMap<String, RateLimit>,
    pub(crate) total_notifications_sent: Arc<AtomicU32>,
}

/// Notification template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationTemplate {
    pub name: String,
    pub subject_template: String,
    pub body_template: String,
    pub format: NotificationFormat,
}

/// Notification formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationFormat {
    Text,
    Html,
    Markdown,
    Json,
}

/// Rate limiting for notifications
#[derive(Debug, Clone)]
pub struct RateLimit {
    pub max_per_hour: u32,
    pub max_per_day: u32,
    pub current_hour_count: u32,
    pub current_day_count: u32,
    pub reset_time: Instant,
}

/// Trait for notification senders
pub trait NotificationSender: Send + Sync + std::fmt::Debug {
    fn send(&self, notification: &AlertNotification) -> Result<()>;
    fn channel_type(&self) -> &str;
    fn is_available(&self) -> bool;
}

/// Monitoring statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStatistics {
    pub metrics_collected: u64,
    pub alerts_triggered: u32,
    pub uptime_seconds: u64,
    pub data_points_stored: u64,
    pub health_checks_performed: u32,
    pub notifications_sent: u32,
}
