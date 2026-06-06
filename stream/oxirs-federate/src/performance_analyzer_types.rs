//! Performance analyzer type definitions.
//!
//! Contains configuration, metric, bottleneck, recommendation, trend, alert,
//! and ML query-plan optimizer data types used across the performance analyzer
//! subsystem.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Analyzer configuration
// ---------------------------------------------------------------------------

/// Configuration for the performance analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    /// Enable real-time performance monitoring
    pub enable_real_time_monitoring: bool,
    /// History retention period
    pub history_retention_hours: u64,
    /// Analysis interval
    pub analysis_interval: Duration,
    /// Enable predictive analysis
    pub enable_predictive_analysis: bool,
    /// Minimum data points for reliable analysis
    pub min_data_points: usize,
    /// Performance baseline update frequency
    pub baseline_update_frequency: Duration,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            enable_real_time_monitoring: true,
            history_retention_hours: 24,
            analysis_interval: Duration::from_secs(60),
            enable_predictive_analysis: true,
            min_data_points: 10,
            baseline_update_frequency: Duration::from_secs(300), // 5 minutes
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// System-wide performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPerformanceMetrics {
    pub timestamp: SystemTime,
    pub overall_latency_p50: Duration,
    pub overall_latency_p95: Duration,
    pub overall_latency_p99: Duration,
    pub throughput_qps: f64,
    pub error_rate: f64,
    pub timeout_rate: f64,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub network_bandwidth_mbps: f64,
    pub active_connections: usize,
    pub queue_depth: usize,
}

/// Service-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePerformanceMetrics {
    pub service_id: String,
    pub timestamp: SystemTime,
    pub response_time_p50: Duration,
    pub response_time_p95: Duration,
    pub response_time_p99: Duration,
    pub requests_per_second: f64,
    pub error_rate: f64,
    pub timeout_rate: f64,
    pub availability: f64,
    pub data_transfer_kb: f64,
    pub connection_pool_utilization: f64,
}

/// Query execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExecutionMetrics {
    pub query_id: String,
    pub timestamp: SystemTime,
    pub total_execution_time: Duration,
    pub planning_time: Duration,
    pub execution_time: Duration,
    pub result_serialization_time: Duration,
    pub services_involved: Vec<String>,
    pub result_size_bytes: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub parallel_steps: usize,
    pub sequential_steps: usize,
}

/// Historical metrics storage
#[derive(Debug)]
pub struct MetricsHistory {
    pub system_metrics: VecDeque<SystemPerformanceMetrics>,
    pub service_metrics: HashMap<String, VecDeque<ServicePerformanceMetrics>>,
    pub query_metrics: VecDeque<QueryExecutionMetrics>,
    pub max_entries: usize,
}

// ---------------------------------------------------------------------------
// Bottleneck analysis
// ---------------------------------------------------------------------------

/// Bottleneck identification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: BottleneckType,
    pub contributing_factors: Vec<BottleneckFactor>,
    pub severity_score: f64,       // 0.0 - 1.0
    pub confidence_level: f64,     // 0.0 - 1.0
    pub impact_on_throughput: f64, // percentage
    pub recommended_actions: Vec<String>,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BottleneckType {
    NetworkLatency,
    ServiceResponseTime,
    MemoryPressure,
    CPUUtilization,
    ConnectionPoolExhaustion,
    QueryComplexity,
    DataTransferVolume,
    CacheInefficiency,
    ParallelizationLimits,
    Unknown,
}

/// Factors contributing to bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckFactor {
    pub factor_type: FactorType,
    pub description: String,
    pub weight: f64, // contribution weight
    pub metric_value: f64,
    pub threshold: f64,
}

/// Types of bottleneck factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorType {
    Latency,
    Throughput,
    ErrorRate,
    ResourceUtilization,
    QueueDepth,
    CachePerformance,
}

/// Bottleneck detection engine
#[derive(Debug)]
pub struct BottleneckDetector {
    pub(crate) baseline_metrics: Option<SystemPerformanceMetrics>,
    pub(crate) detection_thresholds: DetectionThresholds,
}

/// Detection thresholds for bottlenecks
#[derive(Debug, Clone)]
pub struct DetectionThresholds {
    pub latency_degradation_threshold: f64, // percentage increase
    pub throughput_degradation_threshold: f64, // percentage decrease
    pub error_rate_threshold: f64,          // error rate threshold
    pub memory_usage_threshold: f64,        // percentage of total memory
    pub cpu_usage_threshold: f64,           // percentage
    pub cache_hit_rate_threshold: f64,      // minimum cache hit rate
}

// ---------------------------------------------------------------------------
// Recommendations
// ---------------------------------------------------------------------------

/// Performance optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendations {
    pub high_priority: Vec<Recommendation>,
    pub medium_priority: Vec<Recommendation>,
    pub low_priority: Vec<Recommendation>,
    pub long_term: Vec<Recommendation>,
}

/// Individual optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub title: String,
    pub description: String,
    pub category: RecommendationCategory,
    pub expected_improvement: String,
    pub implementation_effort: ImplementationEffort,
    pub estimated_impact_score: f64, // 0.0 - 1.0
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Configuration,
    Architecture,
    QueryOptimization,
    ResourceScaling,
    CachingStrategy,
    NetworkOptimization,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,    // Configuration changes
    Medium, // Code changes
    High,   // Architecture changes
}

/// Recommendation engine
#[derive(Debug)]
pub struct RecommendationEngine {
    pub(crate) rule_base: Vec<OptimizationRule>,
}

/// Optimization rule
#[derive(Debug, Clone)]
pub struct OptimizationRule {
    pub condition: RuleCondition,
    pub recommendation: Recommendation,
    pub priority: f64,
}

/// Rule condition
#[derive(Debug, Clone)]
pub struct RuleCondition {
    pub metric_type: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
}

/// Comparison operators for rules
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
}

// ---------------------------------------------------------------------------
// Trends and alerts
// ---------------------------------------------------------------------------

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub rate_of_change: f64,      // percentage per hour
    pub confidence: f64,          // 0.0 - 1.0
    pub prediction_accuracy: f64, // 0.0 - 1.0
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

/// Alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub critical_latency_ms: u128,
    pub critical_error_rate: f64,
    pub critical_memory_usage: f64,
    pub critical_cpu_usage: f64,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub severity: AlertSeverity,
    pub title: String,
    pub description: String,
    pub timestamp: SystemTime,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

// ---------------------------------------------------------------------------
// Query plan optimizer types
// ---------------------------------------------------------------------------

/// Configuration for query plan optimization
#[derive(Debug, Clone)]
pub struct QueryOptimizerConfig {
    pub enable_ml_optimization: bool,
    pub max_historical_plans: usize,
    pub retraining_interval: usize,
    pub plan_cache_size: usize,
    pub enable_adaptive_timeout: bool,
    pub enable_cost_estimation: bool,
    pub enable_parallel_execution: bool,
    pub confidence_threshold: f64,
}

impl Default for QueryOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_ml_optimization: true,
            max_historical_plans: 10000,
            retraining_interval: 100,
            plan_cache_size: 1000,
            enable_adaptive_timeout: true,
            enable_cost_estimation: true,
            enable_parallel_execution: true,
            confidence_threshold: 0.75,
        }
    }
}

/// Historical execution data for a query plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlanExecution {
    pub query_hash: String,
    pub plan_type: PlanType,
    pub services_involved: Vec<String>,
    pub execution_time: Duration,
    pub result_count: usize,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub network_io_mb: f64,
    pub cache_hit_rate: f64,
    pub timestamp: SystemTime,
    pub query_complexity: QueryComplexity,
    pub success: bool,
}

/// Types of query execution plans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanType {
    Sequential,
    Parallel,
    Hybrid,
    CacheFirst,
    BindJoin,
    HashJoin,
    NestedLoop,
}

/// Query complexity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryComplexity {
    pub triple_patterns: usize,
    pub join_count: usize,
    pub optional_patterns: usize,
    pub filter_count: usize,
    pub union_count: usize,
    pub service_count: usize,
    pub complexity_score: f64,
}

/// ML model for query plan prediction
#[derive(Debug, Clone)]
pub struct QueryPlanModel {
    /// Feature weights for execution time prediction
    pub(crate) execution_time_weights: Vec<f64>,
    /// Feature weights for memory usage prediction
    pub(crate) memory_usage_weights: Vec<f64>,
    /// Feature weights for success rate prediction
    pub(crate) success_rate_weights: Vec<f64>,
    /// Model accuracy metrics
    pub model_accuracy: ModelAccuracy,
    /// Number of training samples
    pub(crate) training_samples: usize,
}

/// Model accuracy tracking
#[derive(Debug, Clone)]
pub struct ModelAccuracy {
    pub execution_time_r_squared: f64,
    pub memory_usage_r_squared: f64,
    pub success_rate_accuracy: f64,
    pub last_training: SystemTime,
}

impl Default for QueryPlanModel {
    fn default() -> Self {
        Self {
            execution_time_weights: vec![0.0; 15], // 15 features
            memory_usage_weights: vec![0.0; 15],
            success_rate_weights: vec![0.0; 15],
            model_accuracy: ModelAccuracy {
                execution_time_r_squared: 0.0,
                memory_usage_r_squared: 0.0,
                success_rate_accuracy: 0.0,
                last_training: SystemTime::now(),
            },
            training_samples: 0,
        }
    }
}

/// Optimal plan recommendation
#[derive(Debug, Clone)]
pub struct OptimalPlan {
    pub plan_type: PlanType,
    pub predicted_execution_time: Duration,
    pub predicted_memory_usage: f64,
    pub predicted_success_rate: f64,
    pub confidence: f64,
    pub service_execution_order: Vec<String>,
    pub parallel_groups: Vec<Vec<String>>,
    pub timeout_recommendation: Duration,
    pub cache_strategy: CacheStrategy,
}

/// Cache strategy recommendation
#[derive(Debug, Clone)]
pub enum CacheStrategy {
    NoCache,
    ResultCache,
    IntermediateCache,
    AggressiveCache,
}

/// Statistics for query optimization
#[derive(Debug, Default)]
pub struct QueryOptimizationStats {
    pub total_optimizations: AtomicU64,
    pub successful_predictions: AtomicU64,
    pub model_retrainings: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub average_improvement: Arc<RwLock<f64>>,
    pub total_time_saved: Arc<RwLock<Duration>>,
}
