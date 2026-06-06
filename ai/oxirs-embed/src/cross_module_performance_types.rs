//! Cross-Module Performance — Types
//!
//! Benchmark structs, module dependency types, and performance metric types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::time::Duration;

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for cross-module performance coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    pub enable_predictive_scaling: bool,
    pub enable_intelligent_prefetching: bool,
    pub enable_dynamic_allocation: bool,
    pub enable_cross_module_caching: bool,
    pub monitoring_interval_ms: u64,
    pub reallocation_threshold: f64,
    pub prefetch_window_seconds: u64,
    pub max_concurrent_optimizations: usize,
    pub enable_performance_learning: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            enable_predictive_scaling: true,
            enable_intelligent_prefetching: true,
            enable_dynamic_allocation: true,
            enable_cross_module_caching: true,
            monitoring_interval_ms: 1000,
            reallocation_threshold: 0.8,
            prefetch_window_seconds: 30,
            max_concurrent_optimizations: 4,
            enable_performance_learning: true,
        }
    }
}

// ── Metrics ───────────────────────────────────────────────────────────────────

/// Module performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMetrics {
    pub cpu_usage: f64,
    pub memory_usage: u64,
    pub gpu_memory_usage: Option<u64>,
    pub network_io_bps: u64,
    pub disk_io_bps: u64,
    pub request_rate: f64,
    pub avg_response_time: Duration,
    pub error_rate: f64,
    pub cache_hit_rate: f64,
    pub active_connections: usize,
    pub queue_depth: usize,
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub metrics: ModuleMetrics,
    pub timestamp: DateTime<Utc>,
}

// ── Resources ─────────────────────────────────────────────────────────────────

/// Resource allocation for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub cpu_cores: usize,
    pub memory_bytes: u64,
    pub gpu_memory_bytes: Option<u64>,
    pub priority: u8,
    pub allocated_at: DateTime<Utc>,
    pub expected_duration: Option<Duration>,
}

/// Resource allocation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    pub module_name: String,
    pub event_type: AllocationType,
    pub allocation: ResourceAllocation,
    pub performance_impact: Option<PerformanceImpact>,
    pub timestamp: DateTime<Utc>,
}

/// Allocation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationType {
    Initial,
    Increase,
    Decrease,
    Rebalance,
    Emergency,
}

/// Allocation strategy
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    Proportional,
    PriorityBased,
    PerformanceBased,
    Predictive,
}

/// Performance impact of allocation changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    pub latency_change_pct: f64,
    pub throughput_change_pct: f64,
    pub efficiency_change_pct: f64,
    pub overall_score: f64,
}

// ── Models & predictions ──────────────────────────────────────────────────────

/// Model types for performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    LinearRegression,
    TimeSeriesARIMA,
    NeuralNetwork,
    EnsembleModel,
    AdaptiveFilter,
}

/// Performance prediction model
#[derive(Debug, Clone)]
pub struct PerformanceModel {
    pub model_type: ModelType,
    pub parameters: std::collections::HashMap<String, f64>,
    pub training_window: Duration,
    pub accuracy: f64,
    pub last_trained: DateTime<Utc>,
}

/// Performance baseline
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub metrics: ModuleMetrics,
    pub established_at: DateTime<Utc>,
    pub confidence: f64,
    pub update_count: u64,
}

// ── Anomalies ─────────────────────────────────────────────────────────────────

/// Anomaly detection algorithm
#[derive(Debug, Clone)]
pub enum AnomalyAlgorithm {
    StatisticalOutlier { z_threshold: f64 },
    IsolationForest { contamination: f64 },
    OneClassSVM { nu: f64 },
    LocalOutlierFactor { n_neighbors: usize },
    EllipticEnvelope { contamination: f64 },
}

/// Anomaly event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub module_name: String,
    pub anomaly_type: AnomalyType,
    pub severity: SeverityLevel,
    pub score: f64,
    pub affected_metrics: Vec<String>,
    pub recommended_actions: Vec<String>,
    pub detected_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
}

/// Anomaly types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyType {
    PerformanceDegradation,
    ResourceSpike,
    MemoryLeak,
    ThroughputDrop,
    LatencyIncrease,
    ErrorRateSpike,
    CacheEfficiencyDrop,
    ConnectionPoolExhaustion,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeverityLevel {
    Low,
    Medium,
    High,
    Critical,
}

// ── Optimizations ─────────────────────────────────────────────────────────────

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub module_name: String,
    pub optimization_type: OptimizationType,
    pub priority: Priority,
    pub description: String,
    pub estimated_impact: PerformanceImpact,
    pub implementation_steps: Vec<String>,
}

/// Optimization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    ResourceReallocation,
    MemoryOptimization,
    CacheOptimization,
    PerformanceTuning,
    GeneralOptimization,
}

/// Priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResults {
    pub anomalies_detected: usize,
    pub optimizations_applied: usize,
    pub optimization_failures: usize,
    pub total_performance_gain: f64,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub execution_time: Duration,
}

impl OptimizationResults {
    pub fn new() -> Self {
        Self {
            anomalies_detected: 0,
            optimizations_applied: 0,
            optimization_failures: 0,
            total_performance_gain: 0.0,
            recommendations: Vec::new(),
            execution_time: Duration::from_secs(0),
        }
    }
}

impl Default for OptimizationResults {
    fn default() -> Self {
        Self::new()
    }
}

// ── Caching & stats ───────────────────────────────────────────────────────────

/// Cached optimization
#[derive(Debug, Clone)]
pub struct CachedOptimization {
    pub recommendation: OptimizationRecommendation,
    pub actual_impact: PerformanceImpact,
    pub cached_at: DateTime<Utc>,
    pub hit_count: u64,
}

/// Cache statistics
#[derive(Debug)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub size: AtomicUsize,
}

/// Training sample for learning
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub features: Vec<f64>,
    pub target: f64,
    pub context: std::collections::HashMap<String, String>,
    pub weight: f64,
    pub timestamp: DateTime<Utc>,
}

/// Cached prediction
#[derive(Debug, Clone)]
pub struct CachedPrediction {
    pub value: f64,
    pub confidence_interval: (f64, f64),
    pub predicted_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub hit_count: u64,
}
