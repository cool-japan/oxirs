//! Core types for validation performance optimization
//!
//! This module contains the core data structures used throughout the validation performance system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

pub use super::config::{
    OptimizationStrategy, PerformanceImprovement, PerformanceStats, TaskPriority, ViolationSeverity,
};

/// Constraint performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintPerformanceStats {
    pub constraint_id: String,
    pub average_execution_time_ms: f64,
    pub success_rate: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub selectivity: f64, // How many items this constraint filters out
    pub execution_count: u64,
    pub last_updated: DateTime<Utc>,
}

/// Constraint dependency graph for ordering optimization
#[derive(Debug, Clone)]
pub struct ConstraintDependencyGraph {
    pub dependencies: HashMap<String, Vec<String>>,
    pub execution_costs: HashMap<String, f64>,
    pub selectivity_scores: HashMap<String, f64>,
}

/// Validation task for parallel execution
#[derive(Debug, Clone)]
pub struct ValidationTask {
    pub task_id: Uuid,
    pub constraint_id: String,
    pub data_subset: Vec<String>, // Simplified representation
    pub priority: TaskPriority,
    pub estimated_duration: Duration,
}

/// Cached validation result
#[derive(Debug, Clone)]
pub struct CachedValidationResult {
    pub key: String,
    pub result: ValidationResult,
    pub created_at: Instant,
    pub ttl: Duration,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_rate: f64,
    pub memory_usage_mb: f64,
    pub eviction_count: u64,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub violations: Vec<ValidationViolation>,
    pub execution_time: Duration,
    pub memory_usage_mb: f64,
    pub constraint_results: HashMap<String, bool>,
}

/// Validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    pub violation_id: Uuid,
    pub constraint_id: String,
    pub severity: ViolationSeverity,
    pub message: String,
    pub focus_node: String,
    pub result_path: Option<String>,
    pub value: Option<String>,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub disk_io_mb_per_sec: f64,
    pub network_io_mb_per_sec: f64,
    pub thread_count: usize,
    pub timestamp: DateTime<Utc>,
}

/// Resource threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    pub max_cpu_usage_percent: f64,
    pub max_memory_usage_mb: f64,
    pub max_thread_count: usize,
    pub alert_threshold_percent: f64,
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            max_cpu_usage_percent: 90.0,
            max_memory_usage_mb: 4096.0,
            max_thread_count: 16,
            alert_threshold_percent: 80.0,
        }
    }
}

/// Index usage patterns for optimization
#[derive(Debug, Clone)]
pub struct IndexUsagePatterns {
    pub index_name: String,
    pub access_count: u64,
    pub last_accessed: DateTime<Utc>,
    pub selectivity: f64,
    pub cost_estimate: f64,
}

/// Query performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceMetrics {
    pub query_id: String,
    pub execution_time_ms: f64,
    pub rows_examined: u64,
    pub rows_returned: u64,
    pub index_usage: Vec<String>,
    pub cost_estimate: f64,
    pub memory_usage_mb: f64,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_id: Uuid,
    pub optimization_type: String,
    pub description: String,
    pub expected_improvement: PerformanceImprovement,
    pub implementation_complexity: String,
    pub priority: TaskPriority,
    pub created_at: DateTime<Utc>,
}

/// Parallel execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecutionStats {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub average_task_time_ms: f64,
    pub parallelization_efficiency: f64,
    pub resource_utilization: f64,
}

/// Quantum-enhanced performance metrics (for advanced optimization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPerformanceMetrics {
    pub entanglement_efficiency: f64,
    pub superposition_states: usize,
    pub quantum_speedup_factor: f64,
    pub decoherence_rate: f64,
    pub consciousness_synchronization: f64,
}

/// Consciousness validation state (for advanced optimization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    pub awareness_level: f64,
    pub processing_capacity: f64,
    pub learning_efficiency: f64,
    pub intuitive_insights: Vec<String>,
    pub emotional_validation_weight: f64,
}

/// Neural pattern recognition results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPatternResult {
    pub pattern_id: String,
    pub confidence: f64,
    pub optimization_suggestion: String,
    pub expected_performance_gain: f64,
}

/// Advanced optimization context
#[derive(Debug, Clone)]
pub struct OptimizationContext {
    pub constraints: Vec<String>,
    pub data_size: usize,
    pub performance_requirements: PerformanceRequirements,
    pub resource_constraints: ResourceThresholds,
    pub historical_performance: HashMap<String, PerformanceStats>,
}

/// Performance requirements specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    pub max_execution_time_ms: u64,
    pub max_memory_usage_mb: f64,
    pub min_throughput_ops_per_sec: f64,
    pub max_error_rate_percent: f64,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_execution_time_ms: 5000,
            max_memory_usage_mb: 1024.0,
            min_throughput_ops_per_sec: 100.0,
            max_error_rate_percent: 1.0,
        }
    }
}
