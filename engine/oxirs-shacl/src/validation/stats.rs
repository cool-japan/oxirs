//! Validation statistics and performance tracking
//!
//! This module provides comprehensive statistics and metrics for SHACL validation
//! including shape conformance rates, data quality metrics, and performance analytics.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

use crate::{Severity, ShapeId, ConstraintComponentId};

/// Comprehensive statistics for validation operations
#[derive(Debug, Clone, Default)]
pub struct ValidationStats {
    pub total_validations: usize,
    pub total_node_validations: usize,
    pub total_constraint_evaluations: usize,
    pub total_validation_time: Duration,
    pub last_validation_time: Option<Duration>,
    pub avg_validation_time: Duration,
    pub constraint_evaluation_times: HashMap<String, Duration>,
    pub cache_hits: usize,
    pub cache_misses: usize,
    
    // Enhanced metrics for shape conformance and data quality
    pub shape_conformance_stats: ShapeConformanceStats,
    pub data_quality_metrics: DataQualityMetrics,
    pub performance_analytics: PerformanceAnalytics,
}

impl ValidationStats {
    /// Record a constraint evaluation with its duration
    pub fn record_constraint_evaluation(&mut self, constraint_type: String, duration: Duration) {
        self.total_constraint_evaluations += 1;
        *self
            .constraint_evaluation_times
            .entry(constraint_type)
            .or_insert(Duration::ZERO) += duration;

        if self.total_validations > 0 {
            self.avg_validation_time = self.total_validation_time / self.total_validations as u32;
        }
    }

    /// Get average constraint evaluation time for a specific constraint type
    pub fn get_avg_constraint_time(&self, constraint_type: &str) -> Option<Duration> {
        self.constraint_evaluation_times
            .get(constraint_type)
            .map(|total| *total / self.total_constraint_evaluations as u32)
    }

    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total_accesses = self.cache_hits + self.cache_misses;
        if total_accesses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_accesses as f64
        }
    }

    /// Record a cache hit
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Record a cache miss
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    /// Record shape validation result
    pub fn record_shape_validation(&mut self, shape_id: ShapeId, conforms: bool, violations: usize, evaluation_time: Duration) {
        self.shape_conformance_stats.record_shape_validation(shape_id, conforms, violations, evaluation_time);
        self.data_quality_metrics.update_from_shape_validation(conforms, violations);
        self.performance_analytics.record_evaluation_time(evaluation_time);
    }

    /// Record constraint violation with severity
    pub fn record_violation(&mut self, shape_id: ShapeId, constraint_id: ConstraintComponentId, severity: Severity) {
        self.shape_conformance_stats.record_violation(shape_id, constraint_id, severity);
        self.data_quality_metrics.record_violation(severity);
    }

    /// Get overall data quality score (0.0 to 1.0)
    pub fn data_quality_score(&self) -> f64 {
        self.data_quality_metrics.calculate_quality_score()
    }

    /// Get shape conformance rate for a specific shape
    pub fn shape_conformance_rate(&self, shape_id: &ShapeId) -> f64 {
        self.shape_conformance_stats.conformance_rate(shape_id)
    }

    /// Get overall conformance rate across all shapes
    pub fn overall_conformance_rate(&self) -> f64 {
        self.shape_conformance_stats.overall_conformance_rate()
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.total_validations = 0;
        self.total_node_validations = 0;
        self.total_constraint_evaluations = 0;
        self.total_validation_time = Duration::ZERO;
        self.last_validation_time = None;
        self.avg_validation_time = Duration::ZERO;
        self.constraint_evaluation_times.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.shape_conformance_stats.reset();
        self.data_quality_metrics.reset();
        self.performance_analytics.reset();
    }

    /// Generate comprehensive statistics report
    pub fn generate_report(&self) -> ValidationStatsReport {
        ValidationStatsReport {
            basic_stats: BasicValidationStats {
                total_validations: self.total_validations,
                total_node_validations: self.total_node_validations,
                total_constraint_evaluations: self.total_constraint_evaluations,
                cache_hit_rate: self.cache_hit_rate(),
                avg_validation_time_ms: self.avg_validation_time.as_millis() as f64,
            },
            shape_conformance: self.shape_conformance_stats.clone(),
            data_quality: self.data_quality_metrics.clone(),
            performance: self.performance_analytics.clone(),
        }
    }
}

/// Shape conformance statistics tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShapeConformanceStats {
    /// Per-shape conformance tracking
    shape_stats: HashMap<ShapeId, ShapeMetrics>,
    /// Overall conformance metrics
    overall_conforming_nodes: usize,
    overall_non_conforming_nodes: usize,
    total_shapes_evaluated: usize,
}

/// Metrics for individual shapes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShapeMetrics {
    /// Number of nodes that conform to this shape
    conforming_nodes: usize,
    /// Number of nodes that don't conform to this shape
    non_conforming_nodes: usize,
    /// Total violations found for this shape
    total_violations: usize,
    /// Violations by constraint type
    violations_by_constraint: HashMap<ConstraintComponentId, ViolationMetrics>,
    /// Average validation time for this shape
    total_evaluation_time: Duration,
    evaluations: usize,
}

/// Violation metrics for specific constraints
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ViolationMetrics {
    count: usize,
    by_severity: HashMap<Severity, usize>,
}

impl ShapeConformanceStats {
    /// Record a shape validation result
    pub fn record_shape_validation(&mut self, shape_id: ShapeId, conforms: bool, violations: usize, evaluation_time: Duration) {
        let shape_metrics = self.shape_stats.entry(shape_id).or_default();
        
        if conforms {
            shape_metrics.conforming_nodes += 1;
            self.overall_conforming_nodes += 1;
        } else {
            shape_metrics.non_conforming_nodes += 1;
            self.overall_non_conforming_nodes += 1;
        }
        
        shape_metrics.total_violations += violations;
        shape_metrics.total_evaluation_time += evaluation_time;
        shape_metrics.evaluations += 1;
        self.total_shapes_evaluated += 1;
    }

    /// Record a constraint violation
    pub fn record_violation(&mut self, shape_id: ShapeId, constraint_id: ConstraintComponentId, severity: Severity) {
        let shape_metrics = self.shape_stats.entry(shape_id).or_default();
        let violation_metrics = shape_metrics.violations_by_constraint.entry(constraint_id).or_default();
        
        violation_metrics.count += 1;
        *violation_metrics.by_severity.entry(severity).or_insert(0) += 1;
    }

    /// Get conformance rate for a specific shape (0.0 to 1.0)
    pub fn conformance_rate(&self, shape_id: &ShapeId) -> f64 {
        if let Some(metrics) = self.shape_stats.get(shape_id) {
            let total = metrics.conforming_nodes + metrics.non_conforming_nodes;
            if total > 0 {
                metrics.conforming_nodes as f64 / total as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Get overall conformance rate across all shapes
    pub fn overall_conformance_rate(&self) -> f64 {
        let total = self.overall_conforming_nodes + self.overall_non_conforming_nodes;
        if total > 0 {
            self.overall_conforming_nodes as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get the most problematic shapes (lowest conformance rates)
    pub fn most_problematic_shapes(&self, limit: usize) -> Vec<(ShapeId, f64)> {
        let mut shape_rates: Vec<(ShapeId, f64)> = self.shape_stats
            .iter()
            .map(|(shape_id, metrics)| {
                let total = metrics.conforming_nodes + metrics.non_conforming_nodes;
                let rate = if total > 0 {
                    metrics.conforming_nodes as f64 / total as f64
                } else {
                    0.0
                };
                (shape_id.clone(), rate)
            })
            .collect();

        shape_rates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        shape_rates.truncate(limit);
        shape_rates
    }

    /// Get average evaluation time for a shape
    pub fn average_evaluation_time(&self, shape_id: &ShapeId) -> Duration {
        if let Some(metrics) = self.shape_stats.get(shape_id) {
            if metrics.evaluations > 0 {
                metrics.total_evaluation_time / metrics.evaluations as u32
            } else {
                Duration::ZERO
            }
        } else {
            Duration::ZERO
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.shape_stats.clear();
        self.overall_conforming_nodes = 0;
        self.overall_non_conforming_nodes = 0;
        self.total_shapes_evaluated = 0;
    }

    /// Get total number of shapes evaluated
    pub fn total_shapes(&self) -> usize {
        self.shape_stats.len()
    }

    /// Get shape with highest violation count
    pub fn shape_with_most_violations(&self) -> Option<(ShapeId, usize)> {
        self.shape_stats
            .iter()
            .max_by_key(|(_, metrics)| metrics.total_violations)
            .map(|(shape_id, metrics)| (shape_id.clone(), metrics.total_violations))
    }
}

/// Data quality metrics and scoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataQualityMetrics {
    /// Total violations by severity
    violations_by_severity: HashMap<Severity, usize>,
    /// Overall data quality indicators
    total_nodes_evaluated: usize,
    nodes_with_violations: usize,
    nodes_without_violations: usize,
    /// Violation density (violations per node)
    total_violations: usize,
    /// Quality trends over time
    quality_measurements: Vec<QualityMeasurement>,
}

/// Point-in-time quality measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMeasurement {
    timestamp: Instant,
    quality_score: f64,
    violation_count: usize,
    nodes_evaluated: usize,
}

impl DataQualityMetrics {
    /// Update metrics from a shape validation
    pub fn update_from_shape_validation(&mut self, conforms: bool, violations: usize) {
        self.total_nodes_evaluated += 1;
        self.total_violations += violations;
        
        if violations > 0 {
            self.nodes_with_violations += 1;
        } else {
            self.nodes_without_violations += 1;
        }
    }

    /// Record a violation with severity
    pub fn record_violation(&mut self, severity: Severity) {
        *self.violations_by_severity.entry(severity).or_insert(0) += 1;
    }

    /// Calculate overall data quality score (0.0 to 1.0, higher is better)
    pub fn calculate_quality_score(&self) -> f64 {
        if self.total_nodes_evaluated == 0 {
            return 1.0; // Perfect score for no data
        }

        // Base score on conforming nodes
        let conformance_score = self.nodes_without_violations as f64 / self.total_nodes_evaluated as f64;
        
        // Apply severity penalties
        let severity_penalty = self.calculate_severity_penalty();
        
        // Apply violation density penalty
        let density_penalty = self.calculate_density_penalty();
        
        // Combine scores (weighted average)
        let quality_score = (conformance_score * 0.5) + 
                           ((1.0 - severity_penalty) * 0.3) + 
                           ((1.0 - density_penalty) * 0.2);
        
        quality_score.max(0.0).min(1.0)
    }

    /// Calculate penalty based on violation severity distribution
    fn calculate_severity_penalty(&self) -> f64 {
        if self.total_violations == 0 {
            return 0.0;
        }

        let total_weighted_violations = self.violations_by_severity.iter()
            .map(|(severity, count)| {
                let weight = match severity {
                    Severity::Violation => 1.0,
                    Severity::Warning => 0.5,
                    Severity::Info => 0.1,
                };
                *count as f64 * weight
            })
            .sum::<f64>();

        // Normalize to penalty score (0.0 = no penalty, 1.0 = maximum penalty)
        (total_weighted_violations / self.total_violations as f64).min(1.0)
    }

    /// Calculate penalty based on violation density
    fn calculate_density_penalty(&self) -> f64 {
        if self.total_nodes_evaluated == 0 {
            return 0.0;
        }

        let density = self.total_violations as f64 / self.total_nodes_evaluated as f64;
        
        // Apply logarithmic scaling to density penalty
        if density > 0.0 {
            (density.ln() + 5.0).max(0.0).min(1.0) / 5.0
        } else {
            0.0
        }
    }

    /// Get violation rate by severity
    pub fn violation_rate_by_severity(&self, severity: Severity) -> f64 {
        if self.total_violations == 0 {
            0.0
        } else {
            let count = self.violations_by_severity.get(&severity).unwrap_or(&0);
            *count as f64 / self.total_violations as f64
        }
    }

    /// Get violation density (violations per node)
    pub fn violation_density(&self) -> f64 {
        if self.total_nodes_evaluated == 0 {
            0.0
        } else {
            self.total_violations as f64 / self.total_nodes_evaluated as f64
        }
    }

    /// Record a quality measurement
    pub fn record_measurement(&mut self) {
        let measurement = QualityMeasurement {
            timestamp: Instant::now(),
            quality_score: self.calculate_quality_score(),
            violation_count: self.total_violations,
            nodes_evaluated: self.total_nodes_evaluated,
        };
        
        self.quality_measurements.push(measurement);
        
        // Keep only last 100 measurements to prevent unbounded growth
        if self.quality_measurements.len() > 100 {
            self.quality_measurements.remove(0);
        }
    }

    /// Get quality trend (positive = improving, negative = degrading)
    pub fn quality_trend(&self) -> f64 {
        if self.quality_measurements.len() < 2 {
            return 0.0;
        }

        let recent = &self.quality_measurements[self.quality_measurements.len() - 1];
        let previous = &self.quality_measurements[self.quality_measurements.len() - 2];
        
        recent.quality_score - previous.quality_score
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.violations_by_severity.clear();
        self.total_nodes_evaluated = 0;
        self.nodes_with_violations = 0;
        self.nodes_without_violations = 0;
        self.total_violations = 0;
        self.quality_measurements.clear();
    }
}

/// Performance analytics and profiling
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceAnalytics {
    /// Evaluation time statistics
    evaluation_times: Vec<Duration>,
    total_evaluation_time: Duration,
    min_evaluation_time: Option<Duration>,
    max_evaluation_time: Option<Duration>,
    
    /// Throughput metrics
    throughput_measurements: Vec<ThroughputMeasurement>,
    
    /// Memory usage tracking
    memory_usage_history: Vec<MemoryMeasurement>,
    
    /// Bottleneck identification
    slow_operations: Vec<SlowOperation>,
}

/// Throughput measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMeasurement {
    timestamp: Instant,
    nodes_per_second: f64,
    constraints_per_second: f64,
}

/// Memory usage measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMeasurement {
    timestamp: Instant,
    memory_usage_bytes: usize,
    cache_size_bytes: usize,
}

/// Slow operation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowOperation {
    operation_type: String,
    duration: Duration,
    timestamp: Instant,
    context: String,
}

impl PerformanceAnalytics {
    /// Record an evaluation time
    pub fn record_evaluation_time(&mut self, duration: Duration) {
        self.evaluation_times.push(duration);
        self.total_evaluation_time += duration;
        
        if self.min_evaluation_time.is_none() || duration < self.min_evaluation_time.unwrap() {
            self.min_evaluation_time = Some(duration);
        }
        
        if self.max_evaluation_time.is_none() || duration > self.max_evaluation_time.unwrap() {
            self.max_evaluation_time = Some(duration);
        }
        
        // Track slow operations (>1 second)
        if duration > Duration::from_secs(1) {
            self.slow_operations.push(SlowOperation {
                operation_type: "validation".to_string(),
                duration,
                timestamp: Instant::now(),
                context: "shape evaluation".to_string(),
            });
        }
        
        // Keep only recent measurements to prevent unbounded growth
        if self.evaluation_times.len() > 10000 {
            self.evaluation_times.remove(0);
        }
    }

    /// Record throughput measurement
    pub fn record_throughput(&mut self, nodes_per_second: f64, constraints_per_second: f64) {
        let measurement = ThroughputMeasurement {
            timestamp: Instant::now(),
            nodes_per_second,
            constraints_per_second,
        };
        
        self.throughput_measurements.push(measurement);
        
        // Keep only last 100 measurements
        if self.throughput_measurements.len() > 100 {
            self.throughput_measurements.remove(0);
        }
    }

    /// Record memory usage
    pub fn record_memory_usage(&mut self, memory_usage_bytes: usize, cache_size_bytes: usize) {
        let measurement = MemoryMeasurement {
            timestamp: Instant::now(),
            memory_usage_bytes,
            cache_size_bytes,
        };
        
        self.memory_usage_history.push(measurement);
        
        // Keep only last 100 measurements
        if self.memory_usage_history.len() > 100 {
            self.memory_usage_history.remove(0);
        }
    }

    /// Get average evaluation time
    pub fn average_evaluation_time(&self) -> Duration {
        if self.evaluation_times.is_empty() {
            Duration::ZERO
        } else {
            self.total_evaluation_time / self.evaluation_times.len() as u32
        }
    }

    /// Get evaluation time percentile
    pub fn evaluation_time_percentile(&self, percentile: f64) -> Duration {
        if self.evaluation_times.is_empty() {
            return Duration::ZERO;
        }

        let mut times = self.evaluation_times.clone();
        times.sort();
        
        let index = ((times.len() - 1) as f64 * percentile / 100.0).round() as usize;
        times[index]
    }

    /// Get current throughput (nodes per second)
    pub fn current_throughput(&self) -> f64 {
        self.throughput_measurements
            .last()
            .map(|m| m.nodes_per_second)
            .unwrap_or(0.0)
    }

    /// Get average throughput
    pub fn average_throughput(&self) -> f64 {
        if self.throughput_measurements.is_empty() {
            0.0
        } else {
            let sum: f64 = self.throughput_measurements.iter().map(|m| m.nodes_per_second).sum();
            sum / self.throughput_measurements.len() as f64
        }
    }

    /// Get current memory usage
    pub fn current_memory_usage(&self) -> usize {
        self.memory_usage_history
            .last()
            .map(|m| m.memory_usage_bytes)
            .unwrap_or(0)
    }

    /// Get peak memory usage
    pub fn peak_memory_usage(&self) -> usize {
        self.memory_usage_history
            .iter()
            .map(|m| m.memory_usage_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Get number of slow operations
    pub fn slow_operations_count(&self) -> usize {
        self.slow_operations.len()
    }

    /// Reset all analytics
    pub fn reset(&mut self) {
        self.evaluation_times.clear();
        self.total_evaluation_time = Duration::ZERO;
        self.min_evaluation_time = None;
        self.max_evaluation_time = None;
        self.throughput_measurements.clear();
        self.memory_usage_history.clear();
        self.slow_operations.clear();
    }
}

/// Basic validation statistics for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicValidationStats {
    pub total_validations: usize,
    pub total_node_validations: usize,
    pub total_constraint_evaluations: usize,
    pub cache_hit_rate: f64,
    pub avg_validation_time_ms: f64,
}

/// Comprehensive validation statistics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatsReport {
    pub basic_stats: BasicValidationStats,
    pub shape_conformance: ShapeConformanceStats,
    pub data_quality: DataQualityMetrics,
    pub performance: PerformanceAnalytics,
}

impl ValidationStatsReport {
    /// Generate a human-readable summary
    pub fn generate_summary(&self) -> String {
        format!(
            "SHACL Validation Statistics Report\n\
             =====================================\n\
             \n\
             Basic Statistics:\n\
             - Total Validations: {}\n\
             - Total Node Validations: {}\n\
             - Total Constraint Evaluations: {}\n\
             - Cache Hit Rate: {:.2}%\n\
             - Average Validation Time: {:.2}ms\n\
             \n\
             Shape Conformance:\n\
             - Overall Conformance Rate: {:.2}%\n\
             - Total Shapes Evaluated: {}\n\
             - Conforming Nodes: {}\n\
             - Non-conforming Nodes: {}\n\
             \n\
             Data Quality:\n\
             - Quality Score: {:.2}/1.0\n\
             - Violation Density: {:.2} violations/node\n\
             - Quality Trend: {:.3}\n\
             \n\
             Performance:\n\
             - Average Evaluation Time: {:.2}ms\n\
             - Current Throughput: {:.1} nodes/sec\n\
             - Average Throughput: {:.1} nodes/sec\n\
             - Peak Memory Usage: {} bytes\n\
             - Slow Operations: {}",
            self.basic_stats.total_validations,
            self.basic_stats.total_node_validations,
            self.basic_stats.total_constraint_evaluations,
            self.basic_stats.cache_hit_rate * 100.0,
            self.basic_stats.avg_validation_time_ms,
            self.shape_conformance.overall_conformance_rate() * 100.0,
            self.shape_conformance.total_shapes(),
            self.shape_conformance.overall_conforming_nodes,
            self.shape_conformance.overall_non_conforming_nodes,
            self.data_quality.calculate_quality_score(),
            self.data_quality.violation_density(),
            self.data_quality.quality_trend(),
            self.performance.average_evaluation_time().as_millis(),
            self.performance.current_throughput(),
            self.performance.average_throughput(),
            self.performance.peak_memory_usage(),
            self.performance.slow_operations_count()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Severity, ShapeId, ConstraintComponentId};
    use std::time::Duration;

    #[test]
    fn test_shape_conformance_stats() {
        let mut stats = ShapeConformanceStats::default();
        let shape1 = ShapeId::new("Shape1");
        let shape2 = ShapeId::new("Shape2");

        // Record some validation results
        stats.record_shape_validation(shape1.clone(), true, 0, Duration::from_millis(10));
        stats.record_shape_validation(shape1.clone(), false, 2, Duration::from_millis(15));
        stats.record_shape_validation(shape2.clone(), true, 0, Duration::from_millis(5));

        // Test conformance rates
        assert_eq!(stats.conformance_rate(&shape1), 0.5); // 1 out of 2 conform
        assert_eq!(stats.conformance_rate(&shape2), 1.0); // 1 out of 1 conform
        assert_eq!(stats.overall_conformance_rate(), 2.0 / 3.0); // 2 out of 3 overall

        // Test most problematic shapes
        let problematic = stats.most_problematic_shapes(2);
        assert_eq!(problematic.len(), 2);
        assert_eq!(problematic[0].0, shape1); // shape1 has lower conformance rate
        assert_eq!(problematic[0].1, 0.5);
        assert_eq!(problematic[1].0, shape2);
        assert_eq!(problematic[1].1, 1.0);
    }

    #[test]
    fn test_data_quality_metrics() {
        let mut metrics = DataQualityMetrics::default();

        // Record some validations
        metrics.update_from_shape_validation(true, 0);  // Perfect node
        metrics.update_from_shape_validation(false, 2); // Node with 2 violations
        metrics.update_from_shape_validation(false, 1); // Node with 1 violation

        // Record violations by severity
        metrics.record_violation(Severity::Violation);
        metrics.record_violation(Severity::Warning);
        metrics.record_violation(Severity::Info);

        assert_eq!(metrics.violation_density(), 1.0); // 3 violations / 3 nodes = 1.0
        assert_eq!(metrics.violation_rate_by_severity(Severity::Violation), 1.0 / 3.0);
        
        let quality_score = metrics.calculate_quality_score();
        assert!(quality_score >= 0.0 && quality_score <= 1.0);
        assert!(quality_score < 1.0); // Should be less than perfect due to violations
    }

    #[test]
    fn test_performance_analytics() {
        let mut analytics = PerformanceAnalytics::default();

        // Record some evaluation times
        analytics.record_evaluation_time(Duration::from_millis(100));
        analytics.record_evaluation_time(Duration::from_millis(200));
        analytics.record_evaluation_time(Duration::from_millis(150));

        assert_eq!(analytics.average_evaluation_time(), Duration::from_millis(150));
        
        // Test percentiles
        let p50 = analytics.evaluation_time_percentile(50.0);
        assert_eq!(p50, Duration::from_millis(150)); // Median

        let p95 = analytics.evaluation_time_percentile(95.0);
        assert_eq!(p95, Duration::from_millis(200)); // Near max

        // Record throughput
        analytics.record_throughput(100.0, 500.0);
        analytics.record_throughput(120.0, 600.0);

        assert_eq!(analytics.current_throughput(), 120.0);
        assert_eq!(analytics.average_throughput(), 110.0);

        // Record memory usage
        analytics.record_memory_usage(1000, 200);
        analytics.record_memory_usage(1200, 250);

        assert_eq!(analytics.current_memory_usage(), 1200);
        assert_eq!(analytics.peak_memory_usage(), 1200);
    }

    #[test]
    fn test_validation_stats_integration() {
        let mut stats = ValidationStats::default();
        let shape_id = ShapeId::new("TestShape");
        let constraint_id = ConstraintComponentId::new("sh:minCount");

        // Record a shape validation
        stats.record_shape_validation(shape_id.clone(), false, 1, Duration::from_millis(50));
        
        // Record a violation
        stats.record_violation(shape_id.clone(), constraint_id, Severity::Violation);

        // Test integrated metrics
        assert_eq!(stats.shape_conformance_rate(&shape_id), 0.0); // 0 out of 1 conform
        assert_eq!(stats.overall_conformance_rate(), 0.0);
        assert!(stats.data_quality_score() < 1.0);

        // Generate report
        let report = stats.generate_report();
        assert_eq!(report.basic_stats.total_validations, 0); // Only shape validation recorded
        assert_eq!(report.shape_conformance.total_shapes(), 1);
        assert!(report.data_quality.calculate_quality_score() < 1.0);
        
        let summary = report.generate_summary();
        assert!(summary.contains("SHACL Validation Statistics Report"));
        assert!(summary.contains("Shape Conformance"));
        assert!(summary.contains("Data Quality"));
        assert!(summary.contains("Performance"));
    }

    #[test]
    fn test_validation_stats_reset() {
        let mut stats = ValidationStats::default();
        let shape_id = ShapeId::new("TestShape");
        let constraint_id = ConstraintComponentId::new("sh:minCount");

        // Add some data
        stats.record_shape_validation(shape_id.clone(), true, 0, Duration::from_millis(25));
        stats.record_violation(shape_id, constraint_id, Severity::Warning);
        stats.record_constraint_evaluation("test".to_string(), Duration::from_millis(10));
        stats.record_cache_hit();

        // Verify data exists
        assert!(stats.overall_conformance_rate() > 0.0);
        assert!(stats.cache_hit_rate() > 0.0);

        // Reset and verify everything is cleared
        stats.reset();
        assert_eq!(stats.overall_conformance_rate(), 0.0);
        assert_eq!(stats.cache_hit_rate(), 0.0);
        assert_eq!(stats.total_validations, 0);
        assert_eq!(stats.shape_conformance_stats.total_shapes(), 0);
    }

    #[test]
    fn test_quality_measurement_trends() {
        let mut metrics = DataQualityMetrics::default();

        // Initial good quality
        metrics.update_from_shape_validation(true, 0);
        metrics.update_from_shape_validation(true, 0);
        metrics.record_measurement();

        // Degrading quality
        metrics.update_from_shape_validation(false, 2);
        metrics.update_from_shape_validation(false, 3);
        metrics.record_measurement();

        let trend = metrics.quality_trend();
        assert!(trend < 0.0); // Quality should be degrading
    }

    #[test]
    fn test_slow_operation_tracking() {
        let mut analytics = PerformanceAnalytics::default();

        // Record normal operations
        analytics.record_evaluation_time(Duration::from_millis(100));
        analytics.record_evaluation_time(Duration::from_millis(200));

        // Record slow operation (>1 second)
        analytics.record_evaluation_time(Duration::from_secs(2));

        assert_eq!(analytics.slow_operations_count(), 1);
    }

    #[test]
    fn test_constraint_violation_metrics() {
        let mut stats = ShapeConformanceStats::default();
        let shape_id = ShapeId::new("TestShape");
        let constraint1 = ConstraintComponentId::new("sh:minCount");
        let constraint2 = ConstraintComponentId::new("sh:datatype");

        // Record violations for different constraints
        stats.record_violation(shape_id.clone(), constraint1, Severity::Violation);
        stats.record_violation(shape_id.clone(), constraint1, Severity::Warning);
        stats.record_violation(shape_id.clone(), constraint2, Severity::Violation);

        // Test shape with most violations
        let most_violations = stats.shape_with_most_violations();
        assert!(most_violations.is_some());
        assert_eq!(most_violations.unwrap().0, shape_id);
        assert_eq!(most_violations.unwrap().1, 3); // Total violations recorded
    }
}

/// Performance statistics for qualified value shape constraint validation
#[derive(Debug, Clone, Default)]
pub struct QualifiedValidationStats {
    /// Total number of value validations performed
    total_validations: usize,

    /// Total time spent on validations
    total_validation_time: Duration,

    /// Number of conforming validations
    conforming_validations: usize,

    /// Number of non-conforming validations
    non_conforming_validations: usize,

    /// Individual validation times for performance analysis
    validation_times: Vec<Duration>,
}

impl QualifiedValidationStats {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a validation result and its timing
    pub fn record_validation(&mut self, validation_time: Duration, conforms: bool) {
        self.total_validations += 1;
        self.total_validation_time += validation_time;
        self.validation_times.push(validation_time);

        if conforms {
            self.conforming_validations += 1;
        } else {
            self.non_conforming_validations += 1;
        }
    }

    /// Get average validation time in milliseconds
    pub fn average_validation_time_ms(&self) -> f64 {
        if self.total_validations == 0 {
            0.0
        } else {
            self.total_validation_time.as_secs_f64() * 1000.0 / self.total_validations as f64
        }
    }

    /// Get conformance rate (0.0 to 1.0)
    pub fn conformance_rate(&self) -> f64 {
        if self.total_validations == 0 {
            0.0
        } else {
            self.conforming_validations as f64 / self.total_validations as f64
        }
    }

    /// Get total validation count
    pub fn total_validations(&self) -> usize {
        self.total_validations
    }

    /// Get total validation time
    pub fn total_validation_time(&self) -> Duration {
        self.total_validation_time
    }

    /// Get median validation time (requires sorting, so can be expensive)
    pub fn median_validation_time_ms(&self) -> f64 {
        if self.validation_times.is_empty() {
            return 0.0;
        }

        let mut times = self.validation_times.clone();
        times.sort();

        let len = times.len();
        if len % 2 == 0 {
            let mid1 = times[len / 2 - 1].as_secs_f64() * 1000.0;
            let mid2 = times[len / 2].as_secs_f64() * 1000.0;
            (mid1 + mid2) / 2.0
        } else {
            times[len / 2].as_secs_f64() * 1000.0
        }
    }

    /// Get percentile validation time (p should be between 0.0 and 1.0)
    pub fn percentile_validation_time_ms(&self, p: f64) -> f64 {
        if self.validation_times.is_empty() || p < 0.0 || p > 1.0 {
            return 0.0;
        }

        let mut times = self.validation_times.clone();
        times.sort();

        let index = ((times.len() - 1) as f64 * p).round() as usize;
        times[index].as_secs_f64() * 1000.0
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.total_validations = 0;
        self.total_validation_time = Duration::ZERO;
        self.conforming_validations = 0;
        self.non_conforming_validations = 0;
        self.validation_times.clear();
    }

    /// Get performance summary as a formatted string
    pub fn summary(&self) -> String {
        format!(
            "QualifiedValidationStats {{ total: {}, conformance_rate: {:.2}%, avg_time: {:.2}ms, median_time: {:.2}ms }}",
            self.total_validations,
            self.conformance_rate() * 100.0,
            self.average_validation_time_ms(),
            self.median_validation_time_ms()
        )
    }
}
