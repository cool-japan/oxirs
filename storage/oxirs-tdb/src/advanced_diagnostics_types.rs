//! Data types for advanced TDB diagnostics.
//!
//! This module defines the report and analysis structures produced by the
//! [`AdvancedDiagnosticEngine`], along with the engine's internal tracking
//! structures:
//! - Public report types ([`AdvancedDiagnosticReport`] and its component analyses)
//! - Trend/forecast types ([`PredictiveHealthIndicators`], [`CapacityForecast`], etc.)
//! - The [`AdvancedDiagnosticEngine`] struct and its internal trackers
//!
//! [`AdvancedDiagnosticEngine`]: crate::advanced_diagnostics_engine::AdvancedDiagnosticEngine

use crate::diagnostics::Severity;
use crate::storage::BufferPoolStats;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Advanced diagnostic report with trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedDiagnosticReport {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Query performance analysis
    pub query_analysis: QueryPerformanceAnalysis,
    /// Transaction pattern analysis
    pub transaction_analysis: TransactionPatternAnalysis,
    /// Fragmentation analysis
    pub fragmentation_analysis: FragmentationAnalysis,
    /// Index usage statistics
    pub index_usage: IndexUsageStatistics,
    /// Predictive health indicators
    pub predictive_health: PredictiveHealthIndicators,
    /// Auto-tuning recommendations
    pub tuning_recommendations: Vec<TuningRecommendation>,
    /// Anomaly detections
    pub anomalies: Vec<AnomalyDetection>,
    /// Capacity planning forecast
    pub capacity_forecast: CapacityForecast,
}

/// Query performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceAnalysis {
    /// Total queries analyzed
    pub total_queries: u64,
    /// Average query execution time
    pub avg_execution_time: Duration,
    /// Median query execution time
    pub median_execution_time: Duration,
    /// P95 query execution time
    pub p95_execution_time: Duration,
    /// P99 query execution time
    pub p99_execution_time: Duration,
    /// Slow query count (above threshold)
    pub slow_query_count: u64,
    /// Query patterns detected
    pub query_patterns: Vec<QueryPattern>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<String>,
    /// Query cache hit rate
    pub cache_hit_rate: f64,
}

/// Query pattern detected in workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    /// Pattern signature (simplified query structure)
    pub signature: String,
    /// Frequency of this pattern
    pub frequency: u64,
    /// Average execution time for this pattern
    pub avg_time: Duration,
    /// Index usage for this pattern
    pub indexes_used: Vec<String>,
    /// Optimization potential (0.0 = none, 1.0 = high)
    pub optimization_potential: f64,
}

/// Transaction pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionPatternAnalysis {
    /// Total transactions analyzed
    pub total_transactions: u64,
    /// Average transaction duration
    pub avg_duration: Duration,
    /// Median transaction duration
    pub median_duration: Duration,
    /// Transaction commit rate
    pub commit_rate: f64,
    /// Transaction abort rate
    pub abort_rate: f64,
    /// Conflict rate (conflicts per transaction)
    pub conflict_rate: f64,
    /// Deadlock rate (deadlocks per 1000 transactions)
    pub deadlock_rate: f64,
    /// Lock contention points
    pub contention_points: Vec<ContentionPoint>,
    /// Transaction size distribution
    pub size_distribution: TransactionSizeDistribution,
}

/// Lock contention point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionPoint {
    /// Resource identifier (e.g., "SPO_INDEX", "DICTIONARY")
    pub resource: String,
    /// Number of contentions detected
    pub contention_count: u64,
    /// Average wait time
    pub avg_wait_time: Duration,
    /// Severity (0.0 = low, 1.0 = high)
    pub severity: f64,
}

/// Transaction size distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionSizeDistribution {
    /// Small transactions (< 10 operations)
    pub small_pct: f64,
    /// Medium transactions (10-100 operations)
    pub medium_pct: f64,
    /// Large transactions (> 100 operations)
    pub large_pct: f64,
}

/// Storage fragmentation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationAnalysis {
    /// Overall fragmentation percentage
    pub overall_fragmentation_pct: f64,
    /// Dictionary fragmentation
    pub dictionary_fragmentation_pct: f64,
    /// Index fragmentation by type
    pub index_fragmentation: HashMap<String, f64>,
    /// Free space distribution
    pub free_space_distribution: Vec<FreeSpaceRegion>,
    /// Compaction benefit estimate (space recoverable)
    pub compaction_benefit_bytes: u64,
    /// Recommended compaction priority (0.0 = low, 1.0 = critical)
    pub compaction_priority: f64,
}

/// Free space region in storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeSpaceRegion {
    /// Starting offset
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
}

/// Index usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexUsageStatistics {
    /// Total index scans performed
    pub total_scans: u64,
    /// Usage by index type
    pub usage_by_index: HashMap<String, IndexUsageStats>,
    /// Unused indexes
    pub unused_indexes: Vec<String>,
    /// Overused indexes (potential bottleneck)
    pub overused_indexes: Vec<String>,
    /// Missing index opportunities
    pub missing_index_opportunities: Vec<MissingIndexOpportunity>,
}

/// Statistics for a specific index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexUsageStats {
    /// Number of scans
    pub scan_count: u64,
    /// Average selectivity (rows returned / rows scanned)
    pub avg_selectivity: f64,
    /// Average scan time
    pub avg_scan_time: Duration,
    /// Index efficiency score (0.0 = inefficient, 1.0 = highly efficient)
    pub efficiency_score: f64,
}

/// Opportunity to create a missing index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingIndexOpportunity {
    /// Suggested index name
    pub suggested_index: String,
    /// Query patterns that would benefit
    pub benefiting_patterns: Vec<String>,
    /// Estimated performance improvement (percentage)
    pub estimated_improvement_pct: f64,
}

/// Predictive health indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveHealthIndicators {
    /// Time series of historical metrics
    pub historical_metrics: HistoricalMetrics,
    /// Predicted issues in next 24 hours
    pub predicted_issues_24h: Vec<PredictedIssue>,
    /// Predicted issues in next 7 days
    pub predicted_issues_7d: Vec<PredictedIssue>,
    /// Health trend (improving, stable, degrading)
    pub health_trend: HealthTrend,
    /// Resource exhaustion predictions
    pub resource_predictions: ResourcePredictions,
}

/// Historical metrics for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalMetrics {
    /// Number of data points
    pub data_points: usize,
    /// Time range covered
    pub time_range: Duration,
    /// Query latency trend (per hour)
    pub query_latency_trend: Vec<f64>,
    /// Transaction throughput trend (per hour)
    pub txn_throughput_trend: Vec<f64>,
    /// Storage growth trend (bytes per hour)
    pub storage_growth_trend: Vec<f64>,
    /// Error rate trend (errors per hour)
    pub error_rate_trend: Vec<f64>,
}

/// Predicted issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedIssue {
    /// Issue category
    pub category: String,
    /// Issue description
    pub description: String,
    /// Confidence level (0.0 = low, 1.0 = certain)
    pub confidence: f64,
    /// Estimated time until issue occurs
    pub eta: Duration,
    /// Severity if issue occurs
    pub severity: Severity,
    /// Preventive action
    pub preventive_action: String,
}

/// Health trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthTrend {
    /// System health is improving
    Improving,
    /// System health is stable
    Stable,
    /// System health is degrading slowly
    DegradingSlowly,
    /// System health is degrading rapidly
    DegradingRapidly,
}

/// Resource exhaustion predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePredictions {
    /// Predicted storage full date
    pub storage_full_eta: Option<SystemTime>,
    /// Predicted memory exhaustion
    pub memory_exhaustion_eta: Option<SystemTime>,
    /// Predicted connection pool saturation
    pub connection_saturation_eta: Option<SystemTime>,
}

/// Auto-tuning recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningRecommendation {
    /// Configuration parameter
    pub parameter: String,
    /// Current value
    pub current_value: String,
    /// Recommended value
    pub recommended_value: String,
    /// Rationale
    pub rationale: String,
    /// Expected impact
    pub expected_impact: String,
    /// Priority (0.0 = low, 1.0 = critical)
    pub priority: f64,
}

/// Anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Metric name
    pub metric: String,
    /// Anomaly description
    pub description: String,
    /// Detected value
    pub detected_value: f64,
    /// Expected value range
    pub expected_range: (f64, f64),
    /// Deviation from normal (standard deviations)
    pub deviation: f64,
    /// Detection timestamp
    pub detected_at: SystemTime,
    /// Severity
    pub severity: Severity,
}

/// Capacity planning forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityForecast {
    /// Current storage usage (bytes)
    pub current_storage_bytes: u64,
    /// Predicted storage in 30 days
    pub predicted_30d_bytes: u64,
    /// Predicted storage in 90 days
    pub predicted_90d_bytes: u64,
    /// Growth rate (bytes per day)
    pub growth_rate_per_day: f64,
    /// Days until 80% capacity
    pub days_until_80pct: Option<u32>,
    /// Days until 90% capacity
    pub days_until_90pct: Option<u32>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Advanced diagnostic engine
pub struct AdvancedDiagnosticEngine {
    /// Historical metrics buffer (circular buffer)
    pub(crate) historical_buffer: VecDeque<MetricSnapshot>,
    /// Maximum history size (24 hours of hourly snapshots)
    pub(crate) max_history_size: usize,
    /// Query performance tracker
    pub(crate) query_tracker: QueryPerformanceTracker,
    /// Transaction pattern tracker
    pub(crate) transaction_tracker: TransactionPatternTracker,
    /// Anomaly detector
    pub(crate) anomaly_detector: AnomalyDetector,
}

/// Snapshot of metrics at a point in time
#[derive(Debug, Clone)]
pub(crate) struct MetricSnapshot {
    pub(crate) timestamp: SystemTime,
    pub(crate) query_latency: f64,
    pub(crate) txn_throughput: f64,
    pub(crate) storage_bytes: u64,
    pub(crate) error_count: u64,
    #[allow(dead_code)]
    pub(crate) buffer_pool_stats: BufferPoolStats,
}

/// Query performance tracker
pub(crate) struct QueryPerformanceTracker {
    /// Query execution times (most recent 1000)
    pub(crate) recent_queries: VecDeque<Duration>,
    /// Query patterns
    pub(crate) patterns: HashMap<String, QueryPatternStats>,
    /// Cache hits vs misses
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct QueryPatternStats {
    pub(crate) frequency: u64,
    pub(crate) total_time: Duration,
    pub(crate) indexes_used: Vec<String>,
}

/// Transaction pattern tracker
pub(crate) struct TransactionPatternTracker {
    /// Recent transaction durations
    pub(crate) recent_durations: VecDeque<Duration>,
    /// Commit/abort counts
    pub(crate) commits: u64,
    pub(crate) aborts: u64,
    /// Conflict tracking
    pub(crate) conflicts: u64,
    pub(crate) deadlocks: u64,
    /// Contention points
    pub(crate) contention_map: HashMap<String, ContentionStats>,
}

#[derive(Debug, Clone)]
pub(crate) struct ContentionStats {
    pub(crate) count: u64,
    pub(crate) total_wait_time: Duration,
}

/// Anomaly detector using statistical methods
pub(crate) struct AnomalyDetector {
    /// Detection threshold (standard deviations)
    pub(crate) threshold: f64,
}
