//! Performance Optimization and Query Reoptimization
//!
//! This module handles performance analysis, query optimization strategies,
//! and adaptive reoptimization based on execution metrics.

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::debug;

use super::query_analysis::QueryInfo;
use super::types::*;

/// Performance optimizer for federated queries
#[derive(Debug)]
pub struct PerformanceOptimizer {
    historical_performance: HistoricalPerformance,
    optimization_config: OptimizationConfig,
    query_frequency_tracker: Arc<RwLock<QueryFrequencyTracker>>,
    predictive_analytics: PredictiveAnalytics,
    pattern_extractor: QueryPatternExtractor,
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
            query_frequency_tracker: Arc::new(RwLock::new(QueryFrequencyTracker::new())),
            predictive_analytics: PredictiveAnalytics::new(),
            pattern_extractor: QueryPatternExtractor::new(),
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
            query_frequency_tracker: Arc::new(RwLock::new(QueryFrequencyTracker::new())),
            predictive_analytics: PredictiveAnalytics::new(),
            pattern_extractor: QueryPatternExtractor::new(),
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
        self.pattern_extractor.extract_pattern(&context.query_id)
    }

    /// Generate optimization suggestions based on metrics
    fn generate_optimization_suggestions(
        &self,
        metrics: &ExecutionMetrics,
        _context: &ExecutionContext,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Analyze service execution times
        let slowest_service = metrics
            .service_times
            .iter()
            .max_by_key(|&(_, &time)| time.as_millis());

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
    pub async fn update_performance_history(
        &mut self,
        metrics: &ExecutionMetrics,
        context: &ExecutionContext,
    ) {
        let query_pattern = self.extract_query_pattern(context);
        let execution_time = metrics.total_execution_time.as_millis() as f64;

        // Update query frequency tracker
        if let Ok(mut tracker) = self.query_frequency_tracker.try_write() {
            tracker.record_query(&query_pattern);
        }

        // Update predictive analytics
        self.predictive_analytics
            .add_data_point(execution_time, metrics);

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

        // Use predictive analytics to get performance predictions
        let predicted_execution_time = self.predictive_analytics.predict_execution_time(
            query_info.complexity.estimated_cost,
            query_info.complexity.field_count as f64,
        );

        // Recommend parallel execution for complex queries
        if query_info.complexity.field_count > 5 || predicted_execution_time > 5000.0 {
            recommendations.preferred_execution_strategy = ExecutionStrategy::Parallel;
            recommendations.suggested_timeout =
                Duration::from_millis(predicted_execution_time as u64 + 10000);
        }

        // Use adaptive strategy for medium complexity queries
        if query_info.complexity.field_count > 2 && query_info.complexity.estimated_cost > 50.0 {
            recommendations.preferred_execution_strategy = ExecutionStrategy::Adaptive;
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

        // Adjust recommendations based on predicted performance
        if predicted_execution_time > 10000.0 {
            recommendations.memory_limit_mb = Some(1024);
            recommendations.batch_size_limit = Some(500);
        }

        recommendations
    }

    /// Check if query is frequently accessed
    fn is_frequently_accessed_query(&self, query_info: &QueryInfo) -> bool {
        // Use operation type as a proxy for query pattern since query_text is not available
        let pattern = format!("{:?}_operation", query_info.operation_type);

        // Check if this pattern appears frequently in our tracker
        match self.query_frequency_tracker.try_read() {
            Ok(tracker) => tracker.is_frequent_pattern(&pattern),
            _ => false,
        }
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
        if total_memory > (self.optimization_config.high_memory_threshold_mb * 1024 * 1024) as usize
        {
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

        score.clamp(0.0, 100.0)
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
                format!("Performance degraded by {diff_percent:.1}% compared to historical average")
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

impl Default for QueryFrequencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PredictiveAnalytics {
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

/// Query frequency tracker for caching decisions
#[derive(Debug)]
pub struct QueryFrequencyTracker {
    pattern_counts: HashMap<String, usize>,
    recent_queries: VecDeque<String>,
    max_recent_queries: usize,
    frequency_threshold: usize,
}

impl QueryFrequencyTracker {
    pub fn new() -> Self {
        Self {
            pattern_counts: HashMap::new(),
            recent_queries: VecDeque::new(),
            max_recent_queries: 1000,
            frequency_threshold: 5,
        }
    }

    pub fn record_query(&mut self, pattern: &str) {
        // Update pattern count
        *self.pattern_counts.entry(pattern.to_string()).or_insert(0) += 1;

        // Add to recent queries
        self.recent_queries.push_back(pattern.to_string());
        if self.recent_queries.len() > self.max_recent_queries {
            if let Some(old_pattern) = self.recent_queries.pop_front() {
                if let Some(count) = self.pattern_counts.get_mut(&old_pattern) {
                    *count -= 1;
                    if *count == 0 {
                        self.pattern_counts.remove(&old_pattern);
                    }
                }
            }
        }
    }

    pub fn is_frequent_pattern(&self, pattern: &str) -> bool {
        self.pattern_counts
            .get(pattern)
            .is_some_and(|&count| count >= self.frequency_threshold)
    }

    pub fn get_pattern_frequency(&self, pattern: &str) -> usize {
        self.pattern_counts.get(pattern).copied().unwrap_or(0)
    }
}

/// Predictive analytics for performance forecasting
#[derive(Debug)]
pub struct PredictiveAnalytics {
    data_points: Vec<PerformanceDataPoint>,
    max_data_points: usize,
    linear_model: Option<LinearRegressionModel>,
}

impl PredictiveAnalytics {
    pub fn new() -> Self {
        Self {
            data_points: Vec::new(),
            max_data_points: 1000,
            linear_model: None,
        }
    }

    pub fn add_data_point(&mut self, execution_time: f64, metrics: &ExecutionMetrics) {
        let data_point = PerformanceDataPoint {
            execution_time,
            service_count: metrics.service_times.len() as f64,
            parallel_steps: metrics.parallel_steps_count as f64,
            result_size: metrics.total_result_size as f64,
            memory_usage: metrics.peak_memory_usage as f64,
        };

        self.data_points.push(data_point);
        if self.data_points.len() > self.max_data_points {
            self.data_points.remove(0);
        }

        // Update linear model if we have enough data
        if self.data_points.len() >= 10 {
            self.update_linear_model();
        }
    }

    pub fn predict_execution_time(&self, estimated_cost: f64, field_count: f64) -> f64 {
        if let Some(ref model) = self.linear_model {
            model.predict(estimated_cost, field_count)
        } else {
            // Fallback to simple heuristic
            estimated_cost * 10.0 + field_count * 100.0
        }
    }

    fn update_linear_model(&mut self) {
        // Simple linear regression: execution_time = a * service_count + b * parallel_steps + c
        let mut model = LinearRegressionModel::new();

        for point in &self.data_points {
            model.add_training_data(
                vec![point.service_count, point.parallel_steps],
                point.execution_time,
            );
        }

        model.train();
        self.linear_model = Some(model);
    }
}

/// Performance data point for analytics
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    pub execution_time: f64,
    pub service_count: f64,
    pub parallel_steps: f64,
    pub result_size: f64,
    pub memory_usage: f64,
}

/// Simple linear regression model for performance prediction
#[derive(Debug, Clone)]
pub struct LinearRegressionModel {
    coefficients: Vec<f64>,
    intercept: f64,
    training_data: Vec<(Vec<f64>, f64)>,
    is_trained: bool,
}

impl Default for LinearRegressionModel {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegressionModel {
    pub fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            intercept: 0.0,
            training_data: Vec::new(),
            is_trained: false,
        }
    }

    pub fn add_training_data(&mut self, features: Vec<f64>, target: f64) {
        self.training_data.push((features, target));
    }

    pub fn train(&mut self) {
        if self.training_data.is_empty() {
            return;
        }

        let n = self.training_data.len();
        let feature_count = self.training_data[0].0.len();

        // Simple least squares regression
        let mut sum_y = 0.0;
        let mut sum_x = vec![0.0; feature_count];
        let mut sum_xx = vec![0.0; feature_count];
        let mut sum_xy = vec![0.0; feature_count];

        for (features, target) in &self.training_data {
            sum_y += target;
            for (i, &feature) in features.iter().enumerate() {
                sum_x[i] += feature;
                sum_xx[i] += feature * feature;
                sum_xy[i] += feature * target;
            }
        }

        let n_f = n as f64;
        self.coefficients = Vec::with_capacity(feature_count);

        for i in 0..feature_count {
            let coefficient = if sum_xx[i] * n_f - sum_x[i] * sum_x[i] != 0.0 {
                (sum_xy[i] * n_f - sum_x[i] * sum_y) / (sum_xx[i] * n_f - sum_x[i] * sum_x[i])
            } else {
                0.0
            };
            self.coefficients.push(coefficient);
        }

        self.intercept = (sum_y
            - self
                .coefficients
                .iter()
                .zip(sum_x.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>())
            / n_f;
        self.is_trained = true;
    }

    pub fn predict(&self, estimated_cost: f64, field_count: f64) -> f64 {
        if !self.is_trained || self.coefficients.len() < 2 {
            return estimated_cost * 10.0 + field_count * 100.0;
        }

        let features = [estimated_cost, field_count];
        let mut prediction = self.intercept;

        for (i, &coef) in self.coefficients.iter().enumerate() {
            if i < features.len() {
                prediction += coef * features[i];
            }
        }

        prediction.max(0.0)
    }
}

/// Advanced query pattern extractor
#[derive(Debug)]
pub struct QueryPatternExtractor {
    #[allow(dead_code)]
    select_regex: Regex,
    join_regex: Regex,
    filter_regex: Regex,
    function_regex: Regex,
}

impl Default for QueryPatternExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryPatternExtractor {
    pub fn new() -> Self {
        Self {
            select_regex: Regex::new(r"SELECT\s+([^\s]+(?:\s+[^\s]+)*?)\s+WHERE")
                .expect("valid regex pattern"),
            join_regex: Regex::new(r"\{[^}]*\}\s*\{[^}]*\}").expect("valid regex pattern"),
            filter_regex: Regex::new(r"FILTER\s*\([^)]+\)").expect("valid regex pattern"),
            function_regex: Regex::new(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\(")
                .expect("construction should succeed"),
        }
    }

    pub fn extract_pattern(&self, query: &str) -> String {
        let mut pattern_parts = Vec::new();

        // Normalize query (remove extra whitespace, convert to uppercase)
        let normalized = query.trim().to_uppercase();

        // Extract query type
        if normalized.contains("SELECT") {
            pattern_parts.push("SELECT");
        } else if normalized.contains("INSERT") {
            pattern_parts.push("INSERT");
        } else if normalized.contains("DELETE") {
            pattern_parts.push("DELETE");
        } else if normalized.contains("UPDATE") {
            pattern_parts.push("UPDATE");
        } else {
            pattern_parts.push("UNKNOWN");
        }

        // Check for aggregation functions
        if self.function_regex.is_match(&normalized) {
            pattern_parts.push("AGGREGATE");
        }

        // Check for filters
        if self.filter_regex.is_match(&normalized) {
            pattern_parts.push("FILTER");
        }

        // Check for joins (multiple graph patterns)
        if self.join_regex.is_match(&normalized) {
            pattern_parts.push("JOIN");
        }

        // Check for optional patterns
        if normalized.contains("OPTIONAL") {
            pattern_parts.push("OPTIONAL");
        }

        // Check for union
        if normalized.contains("UNION") {
            pattern_parts.push("UNION");
        }

        // Check for subqueries
        if normalized.matches("SELECT").count() > 1 {
            pattern_parts.push("SUBQUERY");
        }

        // Check for specific predicates that indicate query complexity
        if normalized.contains("GEO:") || normalized.contains("WGS84:") {
            pattern_parts.push("GEOSPATIAL");
        }

        if normalized.contains("TEXT:") || normalized.contains("LUCENE:") {
            pattern_parts.push("FULLTEXT");
        }

        // Estimate complexity based on query length and patterns
        let complexity = if normalized.len() < 100 {
            "SIMPLE"
        } else if normalized.len() < 500 {
            "MEDIUM"
        } else {
            "COMPLEX"
        };

        pattern_parts.push(complexity);

        pattern_parts.join("_")
    }
}
