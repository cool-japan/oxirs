//! Performance Monitoring and Metrics Collection
//!
//! This module handles query performance tracking, metrics aggregation,
//! and trend analysis for the adaptive query optimizer.

use std::collections::{HashMap, VecDeque};
use std::time::SystemTime;

use oxirs_core::query::{
    algebra::{AlgebraTriplePattern, TermPattern as AlgebraTermPattern},
    pattern_optimizer::IndexType,
};
use serde::{Deserialize, Serialize};

use super::config::AdaptiveOptimizerConfig;

/// Performance monitoring for queries
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Recent performance records
    performance_history: VecDeque<QueryPerformanceRecord>,

    /// Performance metrics aggregation
    aggregated_metrics: PerformanceMetrics,

    /// Pattern performance tracking
    pattern_performance: HashMap<String, PatternPerformanceStats>,

    /// Configuration
    config: AdaptiveOptimizerConfig,
}

/// Individual query performance record
#[derive(Debug, Clone)]
pub struct QueryPerformanceRecord {
    pub query_id: String,
    pub patterns: Vec<AlgebraTriplePattern>,
    pub plan_type: OptimizationPlanType,
    pub execution_time_ms: f64,
    pub memory_usage_mb: f64,
    pub result_count: usize,
    pub index_usage: HashMap<IndexType, usize>,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub timestamp: SystemTime,
    pub success: bool,
    pub error_type: Option<String>,
    pub plan_id: String,
}

/// Type of optimization plan used
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationPlanType {
    Classical,
    Quantum,
    NeuralTransformer,
    Hybrid,
    Adaptive,
}

/// Aggregated performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub avg_execution_time_ms: f64,
    pub p95_execution_time_ms: f64,
    pub p99_execution_time_ms: f64,
    pub success_rate: f64,
    pub avg_memory_usage_mb: f64,
    pub queries_per_second: f64,
    pub cache_hit_rate: f64,
    pub plan_type_distribution: HashMap<OptimizationPlanType, f64>,
    pub trend_direction: TrendDirection,
    pub confidence_score: f64,
}

/// Performance trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

/// Performance statistics for individual patterns
#[derive(Debug, Clone)]
pub struct PatternPerformanceStats {
    pub pattern_signature: String,
    pub execution_count: usize,
    pub avg_execution_time_ms: f64,
    pub success_rate: f64,
    pub best_plan_type: OptimizationPlanType,
    pub optimal_index: IndexType,
    pub selectivity_estimate: f64,
    pub last_updated: SystemTime,
}

impl PerformanceMonitor {
    pub fn new(config: AdaptiveOptimizerConfig) -> Self {
        Self {
            performance_history: VecDeque::new(),
            aggregated_metrics: PerformanceMetrics::default(),
            pattern_performance: HashMap::new(),
            config,
        }
    }

    /// Record query performance
    pub fn record_performance(&mut self, record: QueryPerformanceRecord) {
        // Add to history
        self.performance_history.push_back(record.clone());

        // Maintain window size
        while self.performance_history.len() > self.config.performance_window_size {
            self.performance_history.pop_front();
        }

        // Update pattern-specific performance
        self.update_pattern_performance(&record);

        // Update aggregated metrics
        self.update_aggregated_metrics();
    }

    /// Update pattern-specific performance statistics
    fn update_pattern_performance(&mut self, record: &QueryPerformanceRecord) {
        for pattern in &record.patterns {
            let pattern_signature = self.compute_pattern_signature(pattern);

            let stats = self
                .pattern_performance
                .entry(pattern_signature.clone())
                .or_insert_with(|| PatternPerformanceStats {
                    pattern_signature: pattern_signature.clone(),
                    execution_count: 0,
                    avg_execution_time_ms: 0.0,
                    success_rate: 0.0,
                    best_plan_type: OptimizationPlanType::Classical,
                    optimal_index: IndexType::SPO,
                    selectivity_estimate: 0.1,
                    last_updated: SystemTime::now(),
                });

            // Update running averages
            let prev_count = stats.execution_count as f64;
            let new_count = prev_count + 1.0;

            stats.avg_execution_time_ms =
                (stats.avg_execution_time_ms * prev_count + record.execution_time_ms) / new_count;

            stats.success_rate = (stats.success_rate * prev_count
                + if record.success { 1.0 } else { 0.0 })
                / new_count;

            stats.execution_count += 1;
            stats.last_updated = SystemTime::now();

            // Update best plan type if this performed better
            if record.success && record.execution_time_ms < stats.avg_execution_time_ms {
                stats.best_plan_type = record.plan_type.clone();
            }
        }
    }

    /// Compute pattern signature for tracking
    fn compute_pattern_signature(&self, pattern: &AlgebraTriplePattern) -> String {
        let s_type = match &pattern.subject {
            AlgebraTermPattern::Variable(_) => "VAR",
            AlgebraTermPattern::NamedNode(_) => "NODE",
            AlgebraTermPattern::BlankNode(_) => "BLANK",
            AlgebraTermPattern::Literal(_) => "LIT",
            AlgebraTermPattern::QuotedTriple(_) => "QUOTED",
        };

        let p_type = match &pattern.predicate {
            AlgebraTermPattern::Variable(_) => "VAR",
            AlgebraTermPattern::NamedNode(_) => "NODE",
            AlgebraTermPattern::BlankNode(_) => "BLANK",
            AlgebraTermPattern::Literal(_) => "LIT",
            AlgebraTermPattern::QuotedTriple(_) => "QUOTED",
        };

        let o_type = match &pattern.object {
            AlgebraTermPattern::Variable(_) => "VAR",
            AlgebraTermPattern::NamedNode(_) => "NODE",
            AlgebraTermPattern::BlankNode(_) => "BLANK",
            AlgebraTermPattern::Literal(_) => "LIT",
            AlgebraTermPattern::QuotedTriple(_) => "QUOTED",
        };

        format!("{s_type}:{p_type}:{o_type}")
    }

    /// Update aggregated performance metrics
    fn update_aggregated_metrics(&mut self) {
        if self.performance_history.is_empty() {
            return;
        }

        let records: Vec<&QueryPerformanceRecord> = self.performance_history.iter().collect();

        // Calculate execution time statistics
        let mut execution_times: Vec<f64> = records.iter().map(|r| r.execution_time_ms).collect();
        execution_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        self.aggregated_metrics.avg_execution_time_ms =
            execution_times.iter().sum::<f64>() / execution_times.len() as f64;

        if !execution_times.is_empty() {
            let p95_idx = (execution_times.len() as f64 * 0.95) as usize;
            let p99_idx = (execution_times.len() as f64 * 0.99) as usize;

            self.aggregated_metrics.p95_execution_time_ms = *execution_times
                .get(p95_idx.min(execution_times.len() - 1))
                .unwrap_or(&0.0);

            self.aggregated_metrics.p99_execution_time_ms = *execution_times
                .get(p99_idx.min(execution_times.len() - 1))
                .unwrap_or(&0.0);
        }

        // Calculate success rate
        let successful_queries = records.iter().filter(|r| r.success).count();
        self.aggregated_metrics.success_rate = successful_queries as f64 / records.len() as f64;

        // Calculate memory usage
        self.aggregated_metrics.avg_memory_usage_mb =
            records.iter().map(|r| r.memory_usage_mb).sum::<f64>() / records.len() as f64;

        // Calculate cache hit rate
        let total_hits: usize = records.iter().map(|r| r.cache_hits).sum();
        let total_requests: usize = records.iter().map(|r| r.cache_hits + r.cache_misses).sum();

        self.aggregated_metrics.cache_hit_rate = if total_requests > 0 {
            total_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        // Calculate plan type distribution
        let mut plan_counts: HashMap<OptimizationPlanType, usize> = HashMap::new();
        for record in &records {
            *plan_counts.entry(record.plan_type.clone()).or_insert(0) += 1;
        }

        for (plan_type, count) in plan_counts {
            let proportion = count as f64 / records.len() as f64;
            self.aggregated_metrics
                .plan_type_distribution
                .insert(plan_type, proportion);
        }

        // Determine trend direction
        self.aggregated_metrics.trend_direction = self.calculate_trend_direction(&execution_times);

        // Calculate confidence score
        self.aggregated_metrics.confidence_score = self.calculate_confidence_score();
    }

    /// Calculate performance trend direction
    fn calculate_trend_direction(&self, execution_times: &[f64]) -> TrendDirection {
        if execution_times.len() < 10 {
            return TrendDirection::Stable;
        }

        let mid_point = execution_times.len() / 2;
        let first_half_avg: f64 =
            execution_times[..mid_point].iter().sum::<f64>() / mid_point as f64;
        let second_half_avg: f64 = execution_times[mid_point..].iter().sum::<f64>()
            / (execution_times.len() - mid_point) as f64;

        let change_ratio = (second_half_avg - first_half_avg) / first_half_avg;

        if change_ratio > 0.1 {
            TrendDirection::Degrading
        } else if change_ratio < -0.1 {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        }
    }

    /// Calculate confidence score for metrics
    fn calculate_confidence_score(&self) -> f64 {
        let sample_size = self.performance_history.len() as f64;
        let max_sample_size = self.config.performance_window_size as f64;

        // Base confidence on sample size
        let size_confidence = (sample_size / max_sample_size).min(1.0);

        // Adjust for success rate
        let success_confidence = self.aggregated_metrics.success_rate;

        // Combine confidences
        (size_confidence + success_confidence) / 2.0
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.aggregated_metrics.clone()
    }

    /// Check if adaptation is needed
    pub fn needs_adaptation(&self) -> bool {
        if self.performance_history.len() < self.config.min_queries_for_adaptation {
            return false;
        }

        match self.aggregated_metrics.trend_direction {
            TrendDirection::Degrading => self.aggregated_metrics.confidence_score > 0.7,
            _ => false,
        }
    }

    /// Get pattern performance for specific signature
    pub fn get_pattern_performance(
        &self,
        pattern_signature: &str,
    ) -> Option<&PatternPerformanceStats> {
        self.pattern_performance.get(pattern_signature)
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_execution_time_ms: 0.0,
            p95_execution_time_ms: 0.0,
            p99_execution_time_ms: 0.0,
            success_rate: 1.0,
            avg_memory_usage_mb: 0.0,
            queries_per_second: 0.0,
            cache_hit_rate: 0.0,
            plan_type_distribution: HashMap::new(),
            trend_direction: TrendDirection::Stable,
            confidence_score: 0.0,
        }
    }
}
