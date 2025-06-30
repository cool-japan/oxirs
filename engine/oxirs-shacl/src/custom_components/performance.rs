//! Performance monitoring and statistics for custom constraint components
//!
//! This module provides comprehensive performance tracking, metrics collection,
//! and analysis capabilities for custom constraint components.

use crate::ConstraintComponentId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::security::SecurityPolicy;

/// Performance statistics for components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPerformanceStats {
    /// Number of executions
    pub execution_count: u64,
    /// Total execution time
    #[serde(skip)]
    pub total_execution_time: Duration,
    /// Average execution time
    #[serde(skip)]
    pub average_execution_time: Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
    /// Error statistics
    pub error_stats: ErrorStats,
    /// Last execution timestamp
    #[serde(skip)]
    pub last_execution: Option<Instant>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    /// Average memory usage in bytes
    pub average_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Memory efficiency score
    pub efficiency_score: f64,
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    /// Total error count
    pub total_errors: u64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Common error types
    pub error_types: HashMap<String, u64>,
    /// Error trend (increasing, decreasing, stable)
    pub error_trend: ErrorTrend,
}

/// Error trend enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorTrend {
    Increasing,
    Decreasing,
    Stable,
    Unknown,
}

/// Component execution context
#[derive(Debug, Clone)]
pub struct ComponentExecutionContext {
    /// Execution start time
    pub start_time: Instant,
    /// Memory usage at start
    pub initial_memory: usize,
    /// Current memory usage
    pub current_memory: usize,
    /// Number of SPARQL queries executed
    pub sparql_query_count: u32,
    /// Execution depth (for recursion tracking)
    pub depth: u32,
    /// Security policy
    pub security_policy: SecurityPolicy,
}

/// Execution metrics
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Execution time
    pub execution_time: Duration,
    /// Memory used
    pub memory_used: usize,
    /// Number of SPARQL queries
    pub sparql_queries: u32,
    /// Success flag
    pub success: bool,
    /// Error details (if any)
    pub error: Option<String>,
}

impl Default for ComponentPerformanceStats {
    fn default() -> Self {
        Self {
            execution_count: 0,
            total_execution_time: Duration::from_secs(0),
            average_execution_time: Duration::from_secs(0),
            success_rate: 1.0,
            memory_usage: MemoryUsageStats {
                average_usage: 0,
                peak_usage: 0,
                efficiency_score: 1.0,
            },
            error_stats: ErrorStats {
                total_errors: 0,
                error_rate: 0.0,
                error_types: HashMap::new(),
                error_trend: ErrorTrend::Unknown,
            },
            last_execution: None,
        }
    }
}

impl ComponentPerformanceStats {
    /// Update performance statistics after execution
    pub fn update_execution(&mut self, execution_time: Duration, success: bool) {
        self.execution_count += 1;
        self.total_execution_time += execution_time;
        self.average_execution_time = self.total_execution_time / self.execution_count as u32;

        // Update success rate using exponential moving average
        let alpha = 0.1;
        let new_success_rate = if success { 1.0 } else { 0.0 };
        self.success_rate = alpha * new_success_rate + (1.0 - alpha) * self.success_rate;

        self.last_execution = Some(Instant::now());
    }

    /// Record an error in statistics
    pub fn record_error(&mut self, error_type: &str) {
        self.error_stats.total_errors += 1;

        // Update error rate using exponential moving average
        let alpha = 0.1;
        let new_error_rate = 1.0;
        self.error_stats.error_rate =
            alpha * new_error_rate + (1.0 - alpha) * self.error_stats.error_rate;

        // Track error type
        *self
            .error_stats
            .error_types
            .entry(error_type.to_string())
            .or_insert(0) += 1;

        // Update error trend
        self.error_stats.error_trend = if self.error_stats.error_rate > 0.1 {
            ErrorTrend::Increasing
        } else if self.error_stats.error_rate < 0.01 {
            ErrorTrend::Decreasing
        } else {
            ErrorTrend::Stable
        };
    }

    /// Get overall performance score (0.0 to 1.0)
    pub fn performance_score(&self) -> f64 {
        let success_weight = 0.4;
        let speed_weight = 0.3;
        let memory_weight = 0.3;

        let success_score = self.success_rate;
        let speed_score = if self.average_execution_time <= Duration::from_millis(100) {
            1.0
        } else if self.average_execution_time <= Duration::from_secs(1) {
            0.8
        } else if self.average_execution_time <= Duration::from_secs(5) {
            0.6
        } else {
            0.3
        };
        let memory_score = self.memory_usage.efficiency_score;

        success_weight * success_score + speed_weight * speed_score + memory_weight * memory_score
    }
}