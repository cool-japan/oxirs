//! Performance Profiling for Embedding Operations
//!
//! This module provides comprehensive performance profiling capabilities for
//! knowledge graph embedding operations, including training, inference, and
//! similarity computations.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Operation types for profiling
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    Training,
    Inference,
    SimilarityComputation,
    VectorSearch,
    ModelSaving,
    ModelLoading,
    BatchProcessing,
    EntityEmbedding,
    RelationEmbedding,
    TripleScoring,
    Prediction,
    Custom(String),
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Training => write!(f, "Training"),
            Self::Inference => write!(f, "Inference"),
            Self::SimilarityComputation => write!(f, "Similarity"),
            Self::VectorSearch => write!(f, "VectorSearch"),
            Self::ModelSaving => write!(f, "ModelSave"),
            Self::ModelLoading => write!(f, "ModelLoad"),
            Self::BatchProcessing => write!(f, "BatchProcessing"),
            Self::EntityEmbedding => write!(f, "EntityEmbedding"),
            Self::RelationEmbedding => write!(f, "RelationEmbedding"),
            Self::TripleScoring => write!(f, "TripleScoring"),
            Self::Prediction => write!(f, "Prediction"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Statistics for a specific operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    pub operation_type: OperationType,
    pub total_count: u64,
    pub total_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub average_duration: Duration,
    pub percentile_95: Duration,
    pub percentile_99: Duration,
    pub error_count: u64,
}

impl OperationStats {
    fn new(operation_type: OperationType) -> Self {
        Self {
            operation_type,
            total_count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            average_duration: Duration::ZERO,
            percentile_95: Duration::ZERO,
            percentile_99: Duration::ZERO,
            error_count: 0,
        }
    }

    fn update(&mut self, duration: Duration, is_error: bool) {
        self.total_count += 1;
        self.total_duration += duration;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.average_duration = self.total_duration / self.total_count as u32;

        if is_error {
            self.error_count += 1;
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            ((self.total_count - self.error_count) as f64 / self.total_count as f64) * 100.0
        }
    }

    /// Calculate throughput (operations per second)
    pub fn throughput(&self) -> f64 {
        if self.total_duration.as_secs_f64() > 0.0 {
            self.total_count as f64 / self.total_duration.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Performance profiler for embedding operations
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    stats: Arc<RwLock<HashMap<OperationType, OperationStats>>>,
    durations_buffer: Arc<RwLock<HashMap<OperationType, Vec<Duration>>>>,
    enabled: bool,
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(HashMap::new())),
            durations_buffer: Arc::new(RwLock::new(HashMap::new())),
            enabled: true,
        }
    }

    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start timing an operation
    pub fn start_operation(&self, operation_type: OperationType) -> OperationTimer {
        OperationTimer::new(operation_type, self.clone())
    }

    /// Record an operation duration
    pub fn record_operation(
        &self,
        operation_type: OperationType,
        duration: Duration,
        is_error: bool,
    ) {
        if !self.enabled {
            return;
        }

        // Update stats
        let mut stats = self.stats.write().expect("lock should not be poisoned");
        stats
            .entry(operation_type.clone())
            .or_insert_with(|| OperationStats::new(operation_type.clone()))
            .update(duration, is_error);

        // Store duration for percentile calculation
        let mut durations = self
            .durations_buffer
            .write()
            .expect("lock should not be poisoned");
        durations
            .entry(operation_type.clone())
            .or_default()
            .push(duration);

        // Keep buffer size manageable (last 1000 operations)
        if let Some(buffer) = durations.get_mut(&operation_type) {
            if buffer.len() > 1000 {
                buffer.remove(0);
            }
        }
    }

    /// Get statistics for a specific operation type
    pub fn get_stats(&self, operation_type: OperationType) -> Option<OperationStats> {
        let stats = self.stats.read().expect("read lock should not be poisoned");
        stats.get(&operation_type).cloned()
    }

    /// Get all statistics
    pub fn get_all_stats(&self) -> HashMap<OperationType, OperationStats> {
        let stats = self.stats.read().expect("read lock should not be poisoned");
        stats.clone()
    }

    /// Calculate percentiles for an operation type
    pub fn calculate_percentiles(&self, operation_type: OperationType) -> Option<OperationStats> {
        let durations = self
            .durations_buffer
            .read()
            .expect("read lock should not be poisoned");
        let mut stats = self.stats.write().expect("lock should not be poisoned");

        if let Some(durations_vec) = durations.get(&operation_type) {
            if let Some(op_stats) = stats.get_mut(&operation_type) {
                let mut sorted_durations = durations_vec.clone();
                sorted_durations.sort();

                if !sorted_durations.is_empty() {
                    let p95_index = (sorted_durations.len() as f64 * 0.95) as usize;
                    let p99_index = (sorted_durations.len() as f64 * 0.99) as usize;

                    op_stats.percentile_95 =
                        sorted_durations[p95_index.min(sorted_durations.len() - 1)];
                    op_stats.percentile_99 =
                        sorted_durations[p99_index.min(sorted_durations.len() - 1)];
                }

                return Some(op_stats.clone());
            }
        }

        None
    }

    /// Reset all statistics
    pub fn reset(&self) {
        let mut stats = self.stats.write().expect("lock should not be poisoned");
        let mut durations = self
            .durations_buffer
            .write()
            .expect("lock should not be poisoned");
        stats.clear();
        durations.clear();
    }

    /// Generate a performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let stats = self.get_all_stats();

        let total_operations: u64 = stats.values().map(|s| s.total_count).sum();
        let total_errors: u64 = stats.values().map(|s| s.error_count).sum();
        let total_duration: Duration = stats.values().map(|s| s.total_duration).sum();

        PerformanceReport {
            total_operations,
            total_errors,
            total_duration,
            overall_success_rate: if total_operations > 0 {
                ((total_operations - total_errors) as f64 / total_operations as f64) * 100.0
            } else {
                0.0
            },
            operation_stats: stats,
        }
    }

    /// Export statistics to JSON
    pub fn export_json(&self) -> Result<String> {
        let report = self.generate_report();
        serde_json::to_string_pretty(&report)
            .map_err(|e| anyhow::anyhow!("Failed to serialize report: {}", e))
    }
}

/// Timer for tracking operation duration
pub struct OperationTimer {
    operation_type: OperationType,
    start_time: Instant,
    profiler: PerformanceProfiler,
    recorded: bool,
}

impl OperationTimer {
    fn new(operation_type: OperationType, profiler: PerformanceProfiler) -> Self {
        Self {
            operation_type,
            start_time: Instant::now(),
            profiler,
            recorded: false,
        }
    }

    /// Stop the timer and record the duration
    pub fn stop(mut self) {
        self.record(false);
    }

    /// Stop the timer and record as an error
    pub fn stop_with_error(mut self) {
        self.record(true);
    }

    fn record(&mut self, is_error: bool) {
        if !self.recorded {
            let duration = self.start_time.elapsed();
            self.profiler
                .record_operation(self.operation_type.clone(), duration, is_error);
            self.recorded = true;
        }
    }
}

impl Drop for OperationTimer {
    fn drop(&mut self) {
        // Auto-record if not explicitly stopped
        if !self.recorded {
            self.record(false);
        }
    }
}

/// Comprehensive performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub total_operations: u64,
    pub total_errors: u64,
    pub total_duration: Duration,
    pub overall_success_rate: f64,
    pub operation_stats: HashMap<OperationType, OperationStats>,
}

impl PerformanceReport {
    /// Generate a human-readable summary
    pub fn summary(&self) -> String {
        let mut output = String::new();
        output.push_str("╔════════════════════════════════════════════════════════════════════╗\n");
        output.push_str("║           Embedding Performance Profiling Report                  ║\n");
        output
            .push_str("╚════════════════════════════════════════════════════════════════════╝\n\n");

        output.push_str(&format!("Total Operations: {}\n", self.total_operations));
        output.push_str(&format!("Total Errors: {}\n", self.total_errors));
        output.push_str(&format!(
            "Overall Success Rate: {:.2}%\n",
            self.overall_success_rate
        ));
        output.push_str(&format!(
            "Total Duration: {:.2}s\n\n",
            self.total_duration.as_secs_f64()
        ));

        output.push_str("Operation Statistics:\n");
        output.push_str("─────────────────────────────────────────────────────────────────────\n");

        let mut sorted_ops: Vec<_> = self.operation_stats.iter().collect();
        sorted_ops.sort_by_key(|(_, stats)| std::cmp::Reverse(stats.total_count));

        for (_, stats) in sorted_ops {
            output.push_str(&format!("\n{} Operations:\n", stats.operation_type));
            output.push_str(&format!("  Count: {}\n", stats.total_count));
            output.push_str(&format!("  Success Rate: {:.2}%\n", stats.success_rate()));
            output.push_str(&format!(
                "  Average Duration: {:.2}ms\n",
                stats.average_duration.as_secs_f64() * 1000.0
            ));
            output.push_str(&format!(
                "  Min Duration: {:.2}ms\n",
                stats.min_duration.as_secs_f64() * 1000.0
            ));
            output.push_str(&format!(
                "  Max Duration: {:.2}ms\n",
                stats.max_duration.as_secs_f64() * 1000.0
            ));
            output.push_str(&format!(
                "  P95 Duration: {:.2}ms\n",
                stats.percentile_95.as_secs_f64() * 1000.0
            ));
            output.push_str(&format!(
                "  P99 Duration: {:.2}ms\n",
                stats.percentile_99.as_secs_f64() * 1000.0
            ));
            output.push_str(&format!(
                "  Throughput: {:.2} ops/sec\n",
                stats.throughput()
            ));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_creation() {
        let profiler = PerformanceProfiler::new();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_operation_recording() {
        let profiler = PerformanceProfiler::new();

        profiler.record_operation(OperationType::Training, Duration::from_millis(100), false);
        profiler.record_operation(OperationType::Training, Duration::from_millis(150), false);
        profiler.record_operation(OperationType::Training, Duration::from_millis(120), true);

        let stats = profiler.get_stats(OperationType::Training).unwrap();
        assert_eq!(stats.total_count, 3);
        assert_eq!(stats.error_count, 1);
        assert!((stats.success_rate() - 66.67).abs() < 0.1);
    }

    #[test]
    fn test_operation_timer() {
        let profiler = PerformanceProfiler::new();

        {
            let _timer = profiler.start_operation(OperationType::Inference);
            thread::sleep(Duration::from_millis(50));
        }

        let stats = profiler.get_stats(OperationType::Inference).unwrap();
        assert_eq!(stats.total_count, 1);
        assert!(stats.total_duration >= Duration::from_millis(50));
    }

    #[test]
    fn test_multiple_operation_types() {
        let profiler = PerformanceProfiler::new();

        profiler.record_operation(OperationType::Training, Duration::from_millis(100), false);
        profiler.record_operation(OperationType::Inference, Duration::from_millis(50), false);
        profiler.record_operation(
            OperationType::SimilarityComputation,
            Duration::from_millis(25),
            false,
        );

        let all_stats = profiler.get_all_stats();
        assert_eq!(all_stats.len(), 3);
    }

    #[test]
    fn test_profiler_reset() {
        let profiler = PerformanceProfiler::new();

        profiler.record_operation(OperationType::Training, Duration::from_millis(100), false);
        assert_eq!(profiler.get_all_stats().len(), 1);

        profiler.reset();
        assert_eq!(profiler.get_all_stats().len(), 0);
    }

    #[test]
    fn test_performance_report_generation() {
        let profiler = PerformanceProfiler::new();

        profiler.record_operation(OperationType::Training, Duration::from_millis(100), false);
        profiler.record_operation(OperationType::Inference, Duration::from_millis(50), false);

        let report = profiler.generate_report();
        assert_eq!(report.total_operations, 2);
        assert_eq!(report.total_errors, 0);
        assert_eq!(report.overall_success_rate, 100.0);

        let summary = report.summary();
        assert!(summary.contains("Total Operations: 2"));
    }

    #[test]
    fn test_percentile_calculation() {
        let profiler = PerformanceProfiler::new();

        // Record 100 operations with varying durations
        for i in 1..=100 {
            profiler.record_operation(OperationType::Inference, Duration::from_millis(i), false);
        }

        let stats = profiler
            .calculate_percentiles(OperationType::Inference)
            .unwrap();
        assert!(stats.percentile_95 >= Duration::from_millis(90));
        assert!(stats.percentile_99 >= Duration::from_millis(95));
    }

    #[test]
    fn test_profiler_disable() {
        let mut profiler = PerformanceProfiler::new();
        profiler.disable();

        profiler.record_operation(OperationType::Training, Duration::from_millis(100), false);

        assert_eq!(profiler.get_all_stats().len(), 0);
    }

    #[test]
    fn test_json_export() {
        let profiler = PerformanceProfiler::new();

        profiler.record_operation(OperationType::Training, Duration::from_millis(100), false);

        let json = profiler.export_json().unwrap();
        assert!(json.contains("total_operations"));
        assert!(json.contains("Training"));
    }
}
