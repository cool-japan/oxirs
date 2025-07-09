//! Public API types and data structures for federation monitoring

use serde::Serialize;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Metrics for specific query types
#[derive(Debug, Clone, Serialize, Default)]
pub struct QueryTypeMetrics {
    pub total_count: u64,
    pub success_count: u64,
    pub error_count: u64,
    pub total_duration: Duration,
    pub avg_duration: Duration,
}

impl QueryTypeMetrics {
    pub fn new() -> Self {
        Self {
            total_count: 0,
            success_count: 0,
            error_count: 0,
            total_duration: Duration::from_secs(0),
            avg_duration: Duration::from_secs(0),
        }
    }
}

/// Metrics for individual services
#[derive(Debug, Clone, Serialize, Default)]
pub struct ServiceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub total_response_size: u64,
    pub avg_response_size: u64,
    pub last_seen: u64,
}

impl ServiceMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_duration: Duration::from_secs(0),
            avg_duration: Duration::from_secs(0),
            total_response_size: 0,
            avg_response_size: 0,
            last_seen: 0,
        }
    }
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Default)]
pub struct CacheMetrics {
    pub total_requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

impl CacheMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
        }
    }
}

/// Types of federation events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum FederationEventType {
    QueryStart,
    QueryComplete,
    ServiceRegistered,
    ServiceUnregistered,
    ServiceFailure,
    SchemaUpdate,
    CacheInvalidation,
    Error,
    Warning,
    EntityUpdate,
    SchemaChange,
    ServiceAvailability,
}

/// Health status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Monitoring statistics
#[derive(Debug, Clone, Serialize)]
pub struct MonitorStats {
    pub uptime: Duration,
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub success_rate: f64,
    pub query_type_metrics: HashMap<String, QueryTypeMetrics>,
    pub service_metrics: HashMap<String, ServiceMetrics>,
    pub cache_metrics: HashMap<String, CacheMetrics>,
    pub response_time_histogram: HashMap<String, u64>,
    pub recent_events_count: usize,
    pub event_type_counts: HashMap<FederationEventType, u64>,
    pub avg_queries_per_second: f64,
}

/// Health monitoring metrics
#[derive(Debug, Clone, Serialize)]
pub struct HealthMetrics {
    pub overall_health: HealthStatus,
    pub service_health: HashMap<String, HealthStatus>,
    pub error_rate: f64,
    pub avg_response_time: Duration,
    pub active_services: usize,
    pub recent_error_count: usize,
    pub cache_hit_rate: f64,
    pub timestamp: u64,
}

/// Performance report
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceReport {
    pub report_timestamp: u64,
    pub uptime: Duration,
    pub total_queries: u64,
    pub overall_success_rate: f64,
    pub query_trends: HashMap<String, QueryTrend>,
    pub top_errors: Vec<ErrorSummary>,
    pub performance_summary: PerformanceSummary,
    pub bottlenecks: Vec<BottleneckReport>,
    pub performance_regressions: Vec<RegressionReport>,
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
}

/// Query performance trend
#[derive(Debug, Clone, Serialize)]
pub struct QueryTrend {
    pub query_type: String,
    pub total_queries: u64,
    pub avg_response_time: Duration,
    pub error_rate: f64,
    pub queries_per_second: f64,
}

/// Error summary for reporting
#[derive(Debug, Clone, Serialize)]
pub struct ErrorSummary {
    pub message: String,
    pub count: u64,
}

/// Performance summary
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceSummary {
    pub total_services: usize,
    pub healthy_services: usize,
    pub avg_query_time: Duration,
    pub cache_efficiency: f64,
}

/// Bottleneck analysis report
#[derive(Debug, Clone, Serialize)]
pub struct BottleneckReport {
    pub bottleneck_type: BottleneckType,
    pub component: String,
    pub severity: BottleneckSeverity,
    pub description: String,
    pub metric_value: f64,
    pub threshold: f64,
    pub impact_score: f64,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, Serialize)]
pub enum BottleneckType {
    SlowService,
    HighErrorRate,
    PoorCachePerformance,
    NetworkLatency,
    ResourceContention,
    QueryComplexity,
}

/// Severity levels for bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum BottleneckSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Performance regression report
#[derive(Debug, Clone, Serialize)]
pub struct RegressionReport {
    pub component: String,
    pub regression_type: RegressionType,
    pub severity: RegressionSeverity,
    pub description: String,
    pub historical_value: f64,
    pub current_value: f64,
    pub detected_at: u64,
    pub confidence: f64,
}

/// Types of performance regressions
#[derive(Debug, Clone, Copy, Serialize)]
pub enum RegressionType {
    ResponseTimeIncrease,
    ErrorRateIncrease,
    ThroughputDecrease,
    CacheHitRateDecrease,
}

/// Severity levels for regressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum RegressionSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize)]
pub struct OptimizationRecommendation {
    pub category: OptimizationCategory,
    pub priority: OptimizationPriority,
    pub title: String,
    pub description: String,
    pub estimated_impact: String,
    pub implementation_effort: ImplementationEffort,
    pub metrics_to_monitor: Vec<String>,
}

/// Categories of optimizations
#[derive(Debug, Clone, Copy, Serialize)]
pub enum OptimizationCategory {
    Caching,
    Performance,
    Scaling,
    QueryOptimization,
    NetworkOptimization,
    ResourceUtilization,
}

/// Priority levels for optimization recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum OptimizationPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Implementation effort estimation
#[derive(Debug, Clone, Copy, Serialize)]
pub enum ImplementationEffort {
    Low,    // Hours to 1 day
    Medium, // 1-3 days
    High,   // 1+ weeks
}

/// Distributed tracing span
#[derive(Debug, Clone)]
pub struct TraceSpan {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub start_time: SystemTime,
    pub duration: Duration,
    pub tags: HashMap<String, String>,
    pub service_id: Option<String>,
}

/// Trace statistics for distributed tracing analysis
#[derive(Debug, Clone)]
pub struct TraceStatistics {
    pub total_spans: u64,
    pub total_duration: Duration,
    pub avg_span_duration: Duration,
    pub span_duration_histogram: HashMap<String, u64>,
}

impl Default for TraceStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceStatistics {
    pub fn new() -> Self {
        Self {
            total_spans: 0,
            total_duration: Duration::from_secs(0),
            avg_span_duration: Duration::from_secs(0),
            span_duration_histogram: HashMap::new(),
        }
    }
}

/// Anomaly detection report
#[derive(Debug, Clone, Serialize)]
pub struct AnomalyReport {
    pub anomaly_type: AnomalyType,
    pub detected_at: u64,
    pub details: String,
    pub severity: AnomalySeverity,
    pub confidence: f64,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Copy, Serialize)]
pub enum AnomalyType {
    ErrorSpike,
    PerformanceDegradation,
    UnusualTrafficPattern,
    ServiceUnavailability,
    MemoryLeak,
    ResourceExhaustion,
}

/// Severity levels for anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum AnomalySeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Performance prediction
#[derive(Debug, Clone, Serialize)]
pub struct PerformancePrediction {
    pub prediction_type: PredictionType,
    pub component: String,
    pub predicted_at: u64,
    pub confidence: f64,
    pub description: String,
    pub recommended_actions: Vec<String>,
}

/// Types of performance predictions
#[derive(Debug, Clone, Copy, Serialize)]
pub enum PredictionType {
    PerformanceDegradation,
    CapacityIssue,
    ServiceFailure,
    ResourceBottleneck,
}

/// Cross-service latency analysis
#[derive(Debug, Clone, Serialize)]
pub struct CrossServiceLatencyAnalysis {
    pub interactions: Vec<ServiceInteractionLatency>,
    pub total_traces_analyzed: usize,
    pub analysis_timestamp: u64,
}

/// Latency metrics for service interactions
#[derive(Debug, Clone, Serialize)]
pub struct ServiceInteractionLatency {
    pub from_service: String,
    pub to_service: String,
    pub avg_latency: Duration,
    pub min_latency: Duration,
    pub max_latency: Duration,
    pub sample_count: usize,
}

/// Circuit breaker states for resilience
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Recovery recommendation
#[derive(Debug, Clone, Serialize)]
pub struct RecoveryRecommendation {
    pub recommendation_type: RecoveryType,
    pub component: String,
    pub priority: RecoveryPriority,
    pub description: String,
    pub estimated_recovery_time: Duration,
    pub success_probability: f64,
    pub required_actions: Vec<String>,
}

/// Types of recovery recommendations
#[derive(Debug, Clone, Copy, Serialize)]
pub enum RecoveryType {
    ServiceRestart,
    CacheClearance,
    LoadRebalancing,
    ResourceReallocation,
    ConfigurationUpdate,
}

/// Priority levels for recovery actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum RecoveryPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Emergency = 4,
}

/// Failure prediction report
#[derive(Debug, Clone, Serialize)]
pub struct FailurePrediction {
    pub prediction_type: FailureType,
    pub component: String,
    pub predicted_failure_time: u64,
    pub confidence: f64,
    pub warning_threshold_reached: bool,
    pub preventive_actions: Vec<String>,
}

/// Types of predicted failures
#[derive(Debug, Clone, Copy, Serialize)]
pub enum FailureType {
    ServiceOverload,
    ResourceExhaustion,
    NetworkPartition,
    DataCorruption,
    ConfigurationError,
}

/// Auto-healing action report
#[derive(Debug, Clone, Serialize)]
pub struct AutoHealingAction {
    pub action_type: HealingActionType,
    pub component: String,
    pub executed_at: u64,
    pub success: bool,
    pub description: String,
    pub impact_assessment: String,
}

/// Types of auto-healing actions
#[derive(Debug, Clone, Copy, Serialize)]
pub enum HealingActionType {
    ServiceRestart,
    CacheInvalidation,
    LoadRedistribution,
    CircuitBreakerReset,
    ResourceOptimization,
}