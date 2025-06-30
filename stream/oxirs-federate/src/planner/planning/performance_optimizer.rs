//! Performance Optimization and Query Reoptimization
//!
//! This module handles performance analysis, query optimization strategies,
//! and adaptive reoptimization based on execution metrics.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, info, warn};

use super::query_analysis::{QueryComplexity, QueryInfo};
use super::types::*;

/// Performance optimizer for federated queries
#[derive(Debug)]
pub struct PerformanceOptimizer {
    historical_performance: HistoricalPerformance,
    optimization_config: OptimizationConfig,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    pub fn new() -> Self {
        Self {
            historical_performance: HistoricalPerformance {
                query_patterns: HashMap::new(),
                service_performance: HashMap::new(),
                join_performance: HashMap::new(),
                avg_response_times: HashMap::new(),
            },
            optimization_config: OptimizationConfig::default(),
        }
    }

    /// Create a new performance optimizer with custom configuration
    pub fn with_config(config: OptimizationConfig) -> Self {
        Self {
            historical_performance: HistoricalPerformance {
                query_patterns: HashMap::new(),
                service_performance: HashMap::new(),
                join_performance: HashMap::new(),
                avg_response_times: HashMap::new(),
            },
            optimization_config: config,
        }
    }

    /// Analyze performance and suggest reoptimization
    pub fn analyze_performance(
        &self,
        execution_metrics: &ExecutionMetrics,
        context: &ExecutionContext,
    ) -> Result<ReoptimizationAnalysis> {
        debug!("Analyzing performance for query: {}", context.query_id);

        let mut analysis = ReoptimizationAnalysis {
            should_reoptimize: false,
            performance_degradation: 0.0,
            suggested_changes: Vec::new(),
            confidence_score: 0.0,
        };

        // Check if execution time has degraded
        let performance_degradation =
            self.calculate_performance_degradation(execution_metrics, context);
        analysis.performance_degradation = performance_degradation;

        // Suggest reoptimization if degradation exceeds threshold
        if performance_degradation > self.optimization_config.reoptimization_threshold {
            analysis.should_reoptimize = true;
            analysis.suggested_changes =
                self.generate_optimization_suggestions(execution_metrics, context);
            analysis.confidence_score = self.calculate_confidence_score(execution_metrics);
        }

        // Check for specific performance issues
        self.analyze_specific_issues(&mut analysis, execution_metrics, context);

        Ok(analysis)
    }

    /// Calculate performance degradation compared to historical averages
    fn calculate_performance_degradation(
        &self,
        metrics: &ExecutionMetrics,
        context: &ExecutionContext,
    ) -> f64 {
        let query_pattern = self.extract_query_pattern(context);

        if let Some(&historical_avg) = self
            .historical_performance
            .query_patterns
            .get(&query_pattern)
        {
            let current_time = metrics.total_execution_time.as_millis() as f64;
            let degradation = (current_time - historical_avg) / historical_avg;
            degradation.max(0.0) // Only positive degradation
        } else {
            0.0 // No historical data, assume no degradation
        }
    }

    /// Extract query pattern for performance tracking
    fn extract_query_pattern(&self, context: &ExecutionContext) -> String {
        // Simplified pattern extraction - in reality, would normalize query structure
        format!("query_{}", context.query_id.len() % 10)
    }

    /// Generate optimization suggestions based on metrics
    fn generate_optimization_suggestions(
        &self,
        metrics: &ExecutionMetrics,
        context: &ExecutionContext,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Analyze service execution times
        let slowest_service = metrics
            .service_times
            .iter()
            .max_by_key(|(_, &time)| time.as_millis());

        if let Some((service_id, time)) = slowest_service {
            if time.as_millis() > self.optimization_config.slow_service_threshold_ms {
                suggestions.push(format!(
                    "Consider optimizing service '{}' ({}ms execution time)",
                    service_id,
                    time.as_millis()
                ));
            }
        }

        // Check for excessive parallel execution
        if metrics.parallel_steps_count > self.optimization_config.max_parallel_steps {
            suggestions.push(format!(
                "Reduce parallelism: {} parallel steps may cause resource contention",
                metrics.parallel_steps_count
            ));
        }

        // Check for large result sets
        if metrics.total_result_size > self.optimization_config.large_result_threshold_bytes {
            suggestions.push(format!(
                "Large result set detected ({}MB) - consider pagination or field selection",
                metrics.total_result_size / (1024 * 1024)
            ));
        }

        // Check for network overhead
        let network_time: Duration = metrics.service_times.values().sum();
        let total_time = metrics.total_execution_time;
        let network_ratio = network_time.as_millis() as f64 / total_time.as_millis() as f64;

        if network_ratio > self.optimization_config.high_network_ratio_threshold {
            suggestions.push(format!(
                "High network overhead ({:.1}%) - consider query batching or caching",
                network_ratio * 100.0
            ));
        }

        suggestions
    }

    /// Calculate confidence score for reoptimization suggestions
    fn calculate_confidence_score(&self, metrics: &ExecutionMetrics) -> f64 {
        let mut confidence = 0.0;

        // Higher confidence with more service executions (more data points)
        confidence += (metrics.service_times.len() as f64) * 0.1;

        // Higher confidence with longer execution times (more significant)
        if metrics.total_execution_time.as_millis() > 1000 {
            confidence += 0.3;
        }

        // Higher confidence with larger result sets (more impact)
        if metrics.total_result_size > 1024 * 1024 {
            confidence += 0.2;
        }

        confidence.min(1.0) // Cap at 1.0
    }

    /// Analyze specific performance issues
    fn analyze_specific_issues(
        &self,
        analysis: &mut ReoptimizationAnalysis,
        metrics: &ExecutionMetrics,
        _context: &ExecutionContext,
    ) {
        // Check for service timeout issues
        if metrics.timeout_count > 0 {
            analysis.should_reoptimize = true;
            analysis.suggested_changes.push(format!(
                "Service timeouts detected ({}) - increase timeout or optimize queries",
                metrics.timeout_count
            ));
        }

        // Check for error rate
        if metrics.error_count > 0 {
            let error_rate = metrics.error_count as f64 / (metrics.service_times.len() as f64);
            if error_rate > self.optimization_config.high_error_rate_threshold {
                analysis.should_reoptimize = true;
                analysis.suggested_changes.push(format!(
                    "High error rate ({:.1}%) detected - review service health and query complexity",
                    error_rate * 100.0
                ));
            }
        }

        // Check for memory pressure
        if metrics.peak_memory_usage
            > self.optimization_config.high_memory_threshold_mb * 1024 * 1024
        {
            analysis.should_reoptimize = true;
            analysis.suggested_changes.push(format!(
                "High memory usage ({}MB) - consider streaming or reducing batch sizes",
                metrics.peak_memory_usage / (1024 * 1024)
            ));
        }
    }

    /// Update historical performance data
    pub fn update_performance_history(
        &mut self,
        metrics: &ExecutionMetrics,
        context: &ExecutionContext,
    ) {
        let query_pattern = self.extract_query_pattern(context);
        let execution_time = metrics.total_execution_time.as_millis() as f64;

        // Update query pattern performance
        let current_avg = self
            .historical_performance
            .query_patterns
            .get(&query_pattern)
            .copied()
            .unwrap_or(execution_time);

        // Exponential moving average
        let alpha = 0.1;
        let new_avg = alpha * execution_time + (1.0 - alpha) * current_avg;
        self.historical_performance
            .query_patterns
            .insert(query_pattern, new_avg);

        // Update service performance
        for (service_id, &service_time) in &metrics.service_times {
            let service_time_ms = service_time.as_millis() as f64;
            let current_avg = self
                .historical_performance
                .service_performance
                .get(service_id)
                .copied()
                .unwrap_or(service_time_ms);

            let new_avg = alpha * service_time_ms + (1.0 - alpha) * current_avg;
            self.historical_performance
                .service_performance
                .insert(service_id.clone(), new_avg);
        }

        // Update average response times
        self.historical_performance
            .avg_response_times
            .insert(context.query_id.clone(), metrics.total_execution_time);
    }

    /// Get performance recommendations for query planning
    pub fn get_planning_recommendations(&self, query_info: &QueryInfo) -> PlanningRecommendations {
        let mut recommendations = PlanningRecommendations {
            preferred_execution_strategy: ExecutionStrategy::Sequential,
            suggested_timeout: Duration::from_secs(30),
            enable_caching: false,
            batch_size_limit: None,
            memory_limit_mb: None,
        };

        // Recommend parallel execution for complex queries
        if query_info.complexity.field_count > 5 {
            recommendations.preferred_execution_strategy = ExecutionStrategy::Parallel;
            recommendations.suggested_timeout = Duration::from_secs(60);
        }

        // Recommend caching for frequently accessed queries
        if self.is_frequently_accessed_query(query_info) {
            recommendations.enable_caching = true;
        }

        // Set memory limits for large queries
        if query_info.complexity.estimated_cost > 100.0 {
            recommendations.memory_limit_mb = Some(512);
            recommendations.batch_size_limit = Some(1000);
        }

        recommendations
    }

    /// Check if query is frequently accessed
    fn is_frequently_accessed_query(&self, _query_info: &QueryInfo) -> bool {
        // Simplified check - in reality, would track query frequency
        false
    }

    /// Analyze join performance
    pub fn analyze_join_performance(&self, join_metrics: &JoinMetrics) -> JoinOptimizationAdvice {
        let mut advice = JoinOptimizationAdvice {
            recommended_join_strategy: JoinStrategy::HashJoin,
            estimated_cost: join_metrics.execution_time.as_millis() as f64,
            memory_efficient: true,
            suggestions: Vec::new(),
        };

        // Recommend join strategy based on result set sizes
        if join_metrics.left_result_size > 10000 && join_metrics.right_result_size > 10000 {
            advice.recommended_join_strategy = JoinStrategy::SortMergeJoin;
            advice
                .suggestions
                .push("Large result sets detected - sort-merge join recommended".to_string());
        } else if join_metrics.left_result_size < 100 || join_metrics.right_result_size < 100 {
            advice.recommended_join_strategy = JoinStrategy::NestedLoopJoin;
            advice
                .suggestions
                .push("Small result sets - nested loop join is efficient".to_string());
        }

        // Check memory efficiency
        let total_memory = join_metrics.left_result_size + join_metrics.right_result_size;
        if total_memory > (self.optimization_config.high_memory_threshold_mb * 1024 * 1024) as usize {
            advice.memory_efficient = false;
            advice
                .suggestions
                .push("High memory usage - consider streaming joins".to_string());
        }

        advice
    }

    /// Generate optimization report
    pub fn generate_optimization_report(
        &self,
        metrics: &ExecutionMetrics,
        context: &ExecutionContext,
    ) -> Result<OptimizationReport> {
        let analysis = self.analyze_performance(metrics, context)?;

        let mut report = OptimizationReport {
            query_id: context.query_id.clone(),
            execution_time: metrics.total_execution_time,
            performance_score: self.calculate_performance_score(metrics),
            bottlenecks: self.identify_bottlenecks(metrics),
            recommendations: analysis.suggested_changes,
            historical_comparison: self.compare_with_history(metrics, context),
        };

        // Add service-specific analysis
        for (service_id, &service_time) in &metrics.service_times {
            if service_time.as_millis() > self.optimization_config.slow_service_threshold_ms {
                report.bottlenecks.push(format!(
                    "Slow service: {} ({}ms)",
                    service_id,
                    service_time.as_millis()
                ));
            }
        }

        Ok(report)
    }

    /// Calculate overall performance score
    fn calculate_performance_score(&self, metrics: &ExecutionMetrics) -> f64 {
        let mut score = 100.0;

        // Deduct points for slow execution
        if metrics.total_execution_time.as_millis() > 1000 {
            score -= 20.0;
        }

        // Deduct points for errors
        score -= (metrics.error_count as f64) * 10.0;

        // Deduct points for timeouts
        score -= (metrics.timeout_count as f64) * 15.0;

        // Deduct points for high memory usage
        if metrics.peak_memory_usage > 100 * 1024 * 1024 {
            score -= 10.0;
        }

        score.max(0.0).min(100.0)
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(&self, metrics: &ExecutionMetrics) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        // Network bottlenecks
        let total_network_time: Duration = metrics.service_times.values().sum();
        let network_ratio =
            total_network_time.as_millis() as f64 / metrics.total_execution_time.as_millis() as f64;
        if network_ratio > 0.7 {
            bottlenecks.push("Network latency is the primary bottleneck".to_string());
        }

        // Memory bottlenecks
        if metrics.peak_memory_usage
            > self.optimization_config.high_memory_threshold_mb * 1024 * 1024
        {
            bottlenecks.push("High memory usage may be limiting performance".to_string());
        }

        // Parallelization bottlenecks
        if metrics.parallel_steps_count == 1 && metrics.service_times.len() > 3 {
            bottlenecks.push("Sequential execution limiting parallelization benefits".to_string());
        }

        bottlenecks
    }

    /// Compare current performance with historical data
    fn compare_with_history(
        &self,
        metrics: &ExecutionMetrics,
        context: &ExecutionContext,
    ) -> String {
        let query_pattern = self.extract_query_pattern(context);

        if let Some(&historical_avg) = self
            .historical_performance
            .query_patterns
            .get(&query_pattern)
        {
            let current_time = metrics.total_execution_time.as_millis() as f64;
            let diff_percent = ((current_time - historical_avg) / historical_avg) * 100.0;

            if diff_percent > 10.0 {
                format!(
                    "Performance degraded by {:.1}% compared to historical average",
                    diff_percent
                )
            } else if diff_percent < -10.0 {
                format!(
                    "Performance improved by {:.1}% compared to historical average",
                    -diff_percent
                )
            } else {
                "Performance consistent with historical average".to_string()
            }
        } else {
            "No historical data available for comparison".to_string()
        }
    }
}

impl Default for PerformanceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub reoptimization_threshold: f64,
    pub slow_service_threshold_ms: u128,
    pub max_parallel_steps: usize,
    pub large_result_threshold_bytes: u64,
    pub high_network_ratio_threshold: f64,
    pub high_error_rate_threshold: f64,
    pub high_memory_threshold_mb: u64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            reoptimization_threshold: 0.5, // 50% degradation
            slow_service_threshold_ms: 1000,
            max_parallel_steps: 10,
            large_result_threshold_bytes: 10 * 1024 * 1024, // 10MB
            high_network_ratio_threshold: 0.8,              // 80%
            high_error_rate_threshold: 0.1,                 // 10%
            high_memory_threshold_mb: 256,
        }
    }
}

/// Execution metrics for performance analysis
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    pub total_execution_time: Duration,
    pub service_times: HashMap<String, Duration>,
    pub parallel_steps_count: usize,
    pub total_result_size: u64,
    pub peak_memory_usage: u64,
    pub error_count: usize,
    pub timeout_count: usize,
}

/// Planning recommendations based on performance analysis
#[derive(Debug, Clone)]
pub struct PlanningRecommendations {
    pub preferred_execution_strategy: ExecutionStrategy,
    pub suggested_timeout: Duration,
    pub enable_caching: bool,
    pub batch_size_limit: Option<usize>,
    pub memory_limit_mb: Option<u64>,
}

/// Execution strategy options
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStrategy {
    Sequential,
    Parallel,
    Adaptive,
}

/// Join performance metrics
#[derive(Debug, Clone)]
pub struct JoinMetrics {
    pub execution_time: Duration,
    pub left_result_size: usize,
    pub right_result_size: usize,
    pub output_size: usize,
    pub memory_usage: u64,
}

/// Join optimization advice
#[derive(Debug, Clone)]
pub struct JoinOptimizationAdvice {
    pub recommended_join_strategy: JoinStrategy,
    pub estimated_cost: f64,
    pub memory_efficient: bool,
    pub suggestions: Vec<String>,
}

/// Join strategy options
#[derive(Debug, Clone, PartialEq)]
pub enum JoinStrategy {
    HashJoin,
    SortMergeJoin,
    NestedLoopJoin,
    StreamingJoin,
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub query_id: String,
    pub execution_time: Duration,
    pub performance_score: f64,
    pub bottlenecks: Vec<String>,
    pub recommendations: Vec<String>,
    pub historical_comparison: String,
}
