//! Execution Statistics
//!
//! This module provides statistics collection for query execution.

use std::time::Duration;

/// Join algorithm selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JoinAlgorithm {
    NestedLoop,
    Hash,
    SortMerge,
    IndexNestedLoop,
}

/// Query execution statistics
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Execution time
    pub execution_time: Duration,
    /// Number of intermediate results
    pub intermediate_results: usize,
    /// Number of final results
    pub final_results: usize,
    /// Memory used (estimated)
    pub memory_used: usize,
    /// Number of operations performed
    pub operations: usize,
    /// Number of property path evaluations
    pub property_path_evaluations: usize,
    /// Time spent on property path evaluations
    pub time_spent_on_paths: Duration,
    /// Number of service calls
    pub service_calls: usize,
    /// Time spent on service calls
    pub time_spent_on_services: Duration,
    /// Warnings during execution
    pub warnings: Vec<String>,
}

impl ExecutionStats {
    /// Create a new empty ExecutionStats instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge statistics from another ExecutionStats instance
    pub fn merge_from(&mut self, other: &ExecutionStats) {
        self.execution_time += other.execution_time;
        self.intermediate_results += other.intermediate_results;
        self.final_results += other.final_results;
        self.memory_used += other.memory_used;
        self.operations += other.operations;
        self.property_path_evaluations += other.property_path_evaluations;
        self.time_spent_on_paths += other.time_spent_on_paths;
        self.service_calls += other.service_calls;
        self.time_spent_on_services += other.time_spent_on_services;
        self.warnings.extend(other.warnings.clone());
    }

    /// Add a warning message
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Increment operations counter
    pub fn increment_operations(&mut self) {
        self.operations += 1;
    }

    /// Add memory usage
    pub fn add_memory_usage(&mut self, bytes: usize) {
        self.memory_used += bytes;
    }
}
