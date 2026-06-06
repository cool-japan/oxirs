//! Types for performance monitoring: metrics, thresholds, alerts, time windows.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable real-time monitoring
    pub enable_realtime_monitoring: bool,

    /// Enable performance optimization recommendations
    pub enable_optimization_recommendations: bool,

    /// Enable alerting system
    pub enable_alerting: bool,

    /// Monitoring interval in milliseconds
    pub monitoring_interval_ms: u64,

    /// Metrics retention period in seconds
    pub metrics_retention_seconds: u64,

    /// Performance threshold for alerting
    pub performance_alert_threshold: f64,

    /// Memory usage alert threshold (in MB)
    pub memory_alert_threshold_mb: f64,

    /// CPU usage alert threshold (percentage)
    pub cpu_alert_threshold_percent: f64,

    /// Validation latency alert threshold (in ms)
    pub latency_alert_threshold_ms: f64,

    /// Enable predictive performance analytics
    pub enable_predictive_analytics: bool,

    /// Enable performance profiling
    pub enable_profiling: bool,

    /// Dashboard update frequency in seconds
    pub dashboard_update_frequency_seconds: u64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_realtime_monitoring: true,
            enable_optimization_recommendations: true,
            enable_alerting: true,
            monitoring_interval_ms: 1000,      // 1 second
            metrics_retention_seconds: 86400,  // 24 hours
            performance_alert_threshold: 0.8,  // 80% threshold
            memory_alert_threshold_mb: 1024.0, // 1GB
            cpu_alert_threshold_percent: 80.0,
            latency_alert_threshold_ms: 2000.0, // 2 seconds
            enable_predictive_analytics: true,
            enable_profiling: true,
            dashboard_update_frequency_seconds: 5,
        }
    }
}

/// Individual performance metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub validation_latency_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub throughput_validations_per_second: f64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub concurrent_validations: usize,
    pub queue_depth: usize,
    pub gc_activity: GcActivity,
}

/// Garbage collection activity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcActivity {
    pub gc_count: usize,
    pub gc_time_ms: f64,
    pub heap_usage_mb: f64,
    pub heap_growth_rate: f64,
}

/// Validation-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetric {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub shape_id: String,
    pub validation_duration_ms: f64,
    pub constraint_evaluation_count: usize,
    pub data_size_triples: usize,
    pub success: bool,
    pub error_type: Option<String>,
    pub optimization_applied: bool,
    pub cache_utilized: bool,
    pub parallel_execution: bool,
    pub resource_usage: ResourceUsage,
}

/// Resource usage for individual validations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_time_ms: f64,
    pub memory_peak_mb: f64,
    pub io_operations: usize,
    pub network_calls: usize,
    pub index_lookups: usize,
}

/// System-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetric {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub total_memory_mb: f64,
    pub available_memory_mb: f64,
    pub cpu_cores_available: usize,
    pub cpu_load_average: f64,
    pub disk_io_rate_mbps: f64,
    pub network_io_rate_mbps: f64,
    pub active_connections: usize,
    pub system_health_score: f64,
    pub temperature_celsius: Option<f64>,
}

/// Shape-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapePerformanceMetric {
    pub shape_id: String,
    pub average_validation_time_ms: f64,
    pub peak_validation_time_ms: f64,
    pub validation_count: usize,
    pub success_rate: f64,
    pub optimization_effectiveness: f64,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub performance_trend: PerformanceTrend,
    pub resource_consumption: ResourceConsumption,
}

/// Bottleneck analysis for shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: String,
    pub bottleneck_impact_percent: f64,
    pub constraint_hotspots: Vec<ConstraintHotspot>,
    pub io_bottlenecks: Vec<IoBottleneck>,
    pub concurrency_issues: Vec<ConcurrencyIssue>,
}

/// Constraint performance hotspot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintHotspot {
    pub constraint_path: String,
    pub average_execution_time_ms: f64,
    pub execution_count: usize,
    pub failure_rate: f64,
    pub optimization_potential: f64,
}

/// I/O performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoBottleneck {
    pub operation_type: String,
    pub average_latency_ms: f64,
    pub operation_count: usize,
    pub cache_effectiveness: f64,
    pub suggested_optimization: String,
}

/// Concurrency performance issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyIssue {
    pub issue_type: String,
    pub contention_level: f64,
    pub affected_operations: Vec<String>,
    pub suggested_resolution: String,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub performance_change_percent: f64,
    pub trend_analysis_period_hours: f64,
    pub predicted_future_performance: f64,
    pub confidence_level: f64,
}

/// Trend direction types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
    Unknown,
}

/// Resource consumption analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConsumption {
    pub memory_efficiency: f64,
    pub cpu_efficiency: f64,
    pub io_efficiency: f64,
    pub cache_efficiency: f64,
    pub parallelization_effectiveness: f64,
    pub resource_waste_percent: f64,
}

/// Anomaly detection algorithm
#[derive(Debug, Clone)]
pub struct AnomalyDetectionAlgorithm {
    pub algorithm_type: AnomalyAlgorithmType,
    pub sensitivity: f64,
    pub confidence_threshold: f64,
    pub enabled: bool,
}

/// Types of anomaly detection algorithms
#[derive(Debug, Clone)]
pub enum AnomalyAlgorithmType {
    Statistical,
    MachineLearning,
    ThresholdBased,
    Seasonal,
    Multivariate,
}

/// Baseline performance model
#[derive(Debug, Clone)]
pub struct BaselineModel {
    pub metric_name: String,
    pub baseline_value: f64,
    pub variance: f64,
    pub confidence_interval: (f64, f64),
    pub model_created_at: chrono::DateTime<chrono::Utc>,
    pub model_validity_hours: f64,
}

/// Anomaly event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub event_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub anomaly_type: PerformanceAnomalyType,
    pub severity: AnomalySeverity,
    pub affected_metrics: Vec<String>,
    pub description: String,
    pub impact_assessment: ImpactAssessment,
    pub recommended_actions: Vec<String>,
    pub resolution_status: ResolutionStatus,
}

/// Types of performance anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceAnomalyType {
    PerformanceDegradation,
    MemoryLeak,
    CpuSpike,
    LatencyIncrease,
    ThroughputDrop,
    ErrorRateIncrease,
    ResourceExhaustion,
    UnusualPattern,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Impact assessment for anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    pub user_impact_level: f64,
    pub system_impact_level: f64,
    pub business_impact_level: f64,
    pub affected_operations: Vec<String>,
    pub estimated_resolution_time: Duration,
    pub potential_data_loss_risk: f64,
}

/// Anomaly resolution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStatus {
    Open,
    InProgress,
    Resolved,
    Ignored,
    Escalated,
}

/// Trend model for metrics
#[derive(Debug, Clone)]
pub struct TrendModel {
    pub metric_name: String,
    pub model_type: TrendModelType,
    pub parameters: HashMap<String, f64>,
    pub accuracy: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Types of trend models
#[derive(Debug, Clone)]
pub enum TrendModelType {
    LinearRegression,
    ExponentialSmoothing,
    SeasonalDecomposition,
    ARIMA,
    NeuralNetwork,
}

/// Types of optimization rules
#[derive(Debug, Clone)]
pub enum OptimizationRuleType {
    CacheOptimization,
    ParallelizationImprovement,
    ResourceAllocation,
    ConstraintOrdering,
    IndexOptimization,
    MemoryManagement,
    ConcurrencyOptimization,
}

/// Comparison operators for conditions
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Trigger condition for optimization
#[derive(Debug, Clone)]
pub struct TriggerCondition {
    pub metric_name: String,
    pub operator: ComparisonOperator,
    pub threshold_value: f64,
    pub duration_seconds: f64,
}

/// Optimization action
#[derive(Debug, Clone)]
pub struct OptimizationAction {
    pub action_type: String,
    pub parameters: HashMap<String, String>,
    pub implementation_complexity: f64,
    pub resource_requirements: f64,
    pub rollback_strategy: String,
}

/// Performance optimization rule
#[derive(Debug, Clone)]
pub struct OptimizationRule {
    pub rule_id: String,
    pub rule_type: OptimizationRuleType,
    pub trigger_conditions: Vec<TriggerCondition>,
    pub optimization_action: OptimizationAction,
    pub expected_improvement: f64,
    pub confidence: f64,
    pub enabled: bool,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub optimization_type: String,
    pub description: String,
    pub current_performance: f64,
    pub expected_performance: f64,
    pub improvement_percentage: f64,
    pub implementation_effort: ImplementationEffort,
    pub risk_level: RiskLevel,
    pub prerequisites: Vec<String>,
    pub implementation_steps: Vec<String>,
    pub success_criteria: Vec<String>,
    pub monitoring_recommendations: Vec<String>,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
}

/// Risk levels for optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub rule_id: String,
    pub rule_name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub notification_channels: Vec<String>,
    pub escalation_policy: Option<String>,
    pub suppression_duration: Option<Duration>,
}

/// Alert condition
#[derive(Debug, Clone)]
pub struct AlertCondition {
    pub metric_name: String,
    pub comparison: ComparisonOperator,
    pub threshold: f64,
    pub evaluation_window: Duration,
    pub consecutive_breaches: usize,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Active alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    pub alert_id: String,
    pub rule_id: String,
    pub triggered_at: chrono::DateTime<chrono::Utc>,
    pub severity: AlertSeverity,
    pub current_value: f64,
    pub threshold_value: f64,
    pub description: String,
    pub impact_assessment: String,
    pub recommended_actions: Vec<String>,
    pub acknowledgment_status: AcknowledgmentStatus,
    pub escalation_level: usize,
}

/// Alert acknowledgment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcknowledgmentStatus {
    Unacknowledged,
    Acknowledged,
    Resolved,
    Suppressed,
}

/// Notification channel for alerts
#[derive(Debug, Clone)]
pub struct NotificationChannel {
    pub channel_id: String,
    pub channel_type: NotificationChannelType,
    pub configuration: HashMap<String, String>,
    pub enabled: bool,
}

/// Types of notification channels
#[derive(Debug, Clone)]
pub enum NotificationChannelType {
    Email,
    Slack,
    WebHook,
    SMS,
    PagerDuty,
    Discord,
}

/// Escalation policy for critical alerts
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    pub policy_id: String,
    pub escalation_steps: Vec<EscalationStep>,
    pub max_escalation_time: Duration,
}

/// Individual escalation step
#[derive(Debug, Clone)]
pub struct EscalationStep {
    pub step_number: usize,
    pub delay: Duration,
    pub notification_channels: Vec<String>,
    pub escalation_contacts: Vec<String>,
}

/// Types of dashboard widgets
#[derive(Debug, Clone)]
pub enum WidgetType {
    LineChart,
    BarChart,
    Gauge,
    Counter,
    Table,
    Heatmap,
    StatusIndicator,
}

/// Time range for metrics display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    #[serde(skip, default)]
    pub resolution: Duration,
}

/// Visualization configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    pub color_scheme: String,
    pub axis_labels: HashMap<String, String>,
    pub thresholds: Vec<VisualizationThreshold>,
    pub aggregation_method: AggregationMethod,
}

/// Visualization threshold
#[derive(Debug, Clone)]
pub struct VisualizationThreshold {
    pub value: f64,
    pub color: String,
    pub label: String,
}

/// Aggregation methods for metrics
#[derive(Debug, Clone)]
pub enum AggregationMethod {
    Average,
    Sum,
    Count,
    Min,
    Max,
    Percentile(f64),
}

/// Custom chart definition
#[derive(Debug, Clone)]
pub struct CustomChart {
    pub chart_id: String,
    pub chart_type: String,
    pub query: String,
    pub visualization_options: HashMap<String, String>,
}

/// Dashboard widget for metrics display
#[derive(Debug, Clone)]
pub struct DashboardWidget {
    pub widget_id: String,
    pub widget_type: WidgetType,
    pub title: String,
    pub metrics: Vec<String>,
    pub time_range: TimeRange,
    pub refresh_interval: Duration,
    pub visualization_config: VisualizationConfig,
}

/// Current metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentMetrics {
    pub average_latency_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub throughput_per_second: f64,
    pub error_rate_percent: f64,
    pub cache_hit_rate_percent: f64,
    pub active_validations: usize,
}

/// Trend analysis summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendSummary {
    pub performance_trend: TrendDirection,
    pub trend_confidence: f64,
    pub predicted_issues: Vec<String>,
    pub capacity_forecast: CapacityForecast,
}

/// Capacity forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityForecast {
    pub time_to_capacity_limit: Option<Duration>,
    pub growth_rate_percent: f64,
    pub recommended_scaling_actions: Vec<String>,
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub opportunity_type: String,
    pub description: String,
    pub potential_improvement_percent: f64,
    pub implementation_complexity: ImplementationEffort,
    pub risk_assessment: RiskLevel,
}

/// Performance snapshot for current state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub overall_health_score: f64,
    pub current_metrics: CurrentMetrics,
    pub active_alerts: Vec<ActiveAlert>,
    pub recent_anomalies: Vec<AnomalyEvent>,
    pub trend_analysis: TrendSummary,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Export formats for performance data
#[derive(Debug, Clone)]
pub enum ExportFormat {
    JSON,
    CSV,
    XML,
}

/// Performance data export structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDataExport {
    pub time_range: TimeRange,
    pub metrics: Vec<PerformanceMetric>,
    pub validation_metrics: Vec<ValidationMetric>,
    pub system_metrics: Vec<SystemMetric>,
    pub export_metadata: ExportMetadata,
}

/// Export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub exported_at: chrono::DateTime<chrono::Utc>,
    pub total_records: usize,
    pub data_quality_score: f64,
    pub export_format: String,
}

/// Monitoring statistics
#[derive(Debug, Clone, Default)]
pub struct MonitoringStatistics {
    pub total_metrics_collected: usize,
    pub total_alerts_triggered: usize,
    pub total_anomalies_detected: usize,
    pub total_optimizations_recommended: usize,
    pub average_response_time_ms: f64,
    pub monitoring_uptime_percent: f64,
    pub data_quality_score: f64,
    pub alert_false_positive_rate: f64,
    pub optimization_success_rate: f64,
    pub system_health_score: f64,
}
