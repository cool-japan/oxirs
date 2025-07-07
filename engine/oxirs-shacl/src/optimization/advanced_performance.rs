//! Advanced Performance Analytics for SHACL Validation
//!
//! This module provides comprehensive performance analytics with adaptive threshold
//! adjustment, predictive optimization, and real-time performance monitoring.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{
    constraints::{Constraint, ConstraintEvaluationResult},
    ConstraintComponentId, ShapeId,
};

/// Advanced performance analytics engine with adaptive optimization
#[derive(Debug)]
pub struct AdvancedPerformanceAnalytics {
    /// Constraint performance metrics
    constraint_metrics: HashMap<ConstraintComponentId, ConstraintPerformanceMetrics>,
    /// Shape performance metrics
    shape_metrics: HashMap<ShapeId, ShapePerformanceMetrics>,
    /// Global validation performance
    global_metrics: GlobalPerformanceMetrics,
    /// Adaptive configuration
    config: AnalyticsConfig,
    /// Performance trends analyzer
    trends_analyzer: PerformanceTrendsAnalyzer,
    /// Real-time monitoring
    real_time_monitor: RealTimePerformanceMonitor,
}

/// Configuration for performance analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable adaptive threshold adjustment
    pub enable_adaptive_thresholds: bool,
    /// Threshold adjustment sensitivity (0.0-1.0)
    pub adjustment_sensitivity: f64,
    /// Performance history window size
    pub history_window_size: usize,
    /// Enable predictive optimization
    pub enable_predictive_optimization: bool,
    /// Real-time monitoring interval
    pub monitoring_interval: Duration,
    /// Performance baseline percentile (e.g., 95th percentile)
    pub baseline_percentile: f64,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_thresholds: true,
            adjustment_sensitivity: 0.1,
            history_window_size: 1000,
            enable_predictive_optimization: true,
            monitoring_interval: Duration::from_secs(10),
            baseline_percentile: 0.95,
        }
    }
}

/// Performance metrics for individual constraints
#[derive(Debug, Clone)]
pub struct ConstraintPerformanceMetrics {
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Standard deviation of execution times
    pub execution_time_std_dev: Duration,
    /// Minimum execution time
    pub min_execution_time: Duration,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Success rate (successful evaluations / total evaluations)
    pub success_rate: f64,
    /// Selectivity (violations / total evaluations)
    pub selectivity: f64,
    /// Total evaluations
    pub total_evaluations: usize,
    /// Adaptive performance threshold
    pub adaptive_threshold: Duration,
    /// Performance trend (improving/degrading)
    pub performance_trend: PerformanceTrend,
    /// Last evaluation timestamp
    pub last_evaluation: Instant,
}

/// Performance metrics for shapes
#[derive(Debug, Clone)]
pub struct ShapePerformanceMetrics {
    /// Total validation time for this shape
    pub total_validation_time: Duration,
    /// Average validation time per focus node
    pub avg_validation_time: Duration,
    /// Number of focus nodes validated
    pub focus_nodes_validated: usize,
    /// Total violations found
    pub total_violations: usize,
    /// Constraint evaluation breakdown
    pub constraint_breakdown: HashMap<ConstraintComponentId, Duration>,
    /// Performance efficiency score (0.0-1.0)
    pub efficiency_score: f64,
    /// Memory usage during validation
    pub memory_usage_mb: f64,
}

/// Global validation performance metrics
#[derive(Debug, Clone)]
pub struct GlobalPerformanceMetrics {
    /// Total validation time across all operations
    pub total_validation_time: Duration,
    /// Total shapes evaluated
    pub total_shapes_evaluated: usize,
    /// Total constraints evaluated
    pub total_constraints_evaluated: usize,
    /// Overall success rate
    pub overall_success_rate: f64,
    /// Average validation throughput (nodes/second)
    pub avg_throughput: f64,
    /// Peak memory usage
    pub peak_memory_usage_mb: f64,
    /// Performance optimization score
    pub optimization_score: f64,
}

/// Performance trend indicators
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceTrend {
    /// Performance is improving
    Improving(f64),
    /// Performance is stable
    Stable,
    /// Performance is degrading
    Degrading(f64),
    /// Insufficient data for trend analysis
    Unknown,
}

/// Performance trends analyzer
#[derive(Debug)]
pub struct PerformanceTrendsAnalyzer {
    /// Historical performance data
    performance_history: Vec<PerformanceDataPoint>,
    /// Trend analysis configuration
    config: TrendAnalysisConfig,
}

/// Configuration for trend analysis
#[derive(Debug, Clone)]
pub struct TrendAnalysisConfig {
    /// Window size for trend calculation
    pub trend_window_size: usize,
    /// Minimum data points needed for trend analysis
    pub min_data_points: usize,
    /// Trend sensitivity threshold
    pub trend_sensitivity: f64,
}

/// Historical performance data point
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    /// Timestamp
    pub timestamp: Instant,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Throughput
    pub throughput: f64,
    /// Memory usage
    pub memory_usage_mb: f64,
}

/// Real-time performance monitor
#[derive(Debug)]
pub struct RealTimePerformanceMonitor {
    /// Current monitoring window
    current_window: PerformanceWindow,
    /// Alert thresholds
    alert_thresholds: AlertThresholds,
    /// Active alerts
    active_alerts: Vec<PerformanceAlert>,
}

/// Performance monitoring window
#[derive(Debug)]
pub struct PerformanceWindow {
    /// Window start time
    pub start_time: Instant,
    /// Window duration
    pub duration: Duration,
    /// Operations in this window
    pub operations: Vec<PerformanceOperation>,
}

/// Performance operation record
#[derive(Debug, Clone)]
pub struct PerformanceOperation {
    /// Operation timestamp
    pub timestamp: Instant,
    /// Operation type
    pub operation_type: OperationType,
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage
    pub memory_usage: usize,
    /// Success indicator
    pub success: bool,
}

/// Types of performance operations
#[derive(Debug, Clone)]
pub enum OperationType {
    /// Constraint evaluation
    ConstraintEvaluation(ConstraintComponentId),
    /// Shape validation
    ShapeValidation(ShapeId),
    /// Target selection
    TargetSelection,
    /// Path evaluation
    PathEvaluation,
}

/// Performance alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Maximum acceptable execution time
    pub max_execution_time: Duration,
    /// Minimum acceptable success rate
    pub min_success_rate: f64,
    /// Maximum acceptable memory usage
    pub max_memory_usage_mb: f64,
    /// Minimum acceptable throughput
    pub min_throughput: f64,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    /// Alert type
    pub alert_type: AlertType,
    /// Alert message
    pub message: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert timestamp
    pub timestamp: Instant,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Types of performance alerts
#[derive(Debug, Clone)]
pub enum AlertType {
    /// High execution time
    HighExecutionTime,
    /// Low success rate
    LowSuccessRate,
    /// High memory usage
    HighMemoryUsage,
    /// Low throughput
    LowThroughput,
    /// Performance degradation
    PerformanceDegradation,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
}

impl AdvancedPerformanceAnalytics {
    /// Create a new advanced performance analytics engine
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            constraint_metrics: HashMap::new(),
            shape_metrics: HashMap::new(),
            global_metrics: GlobalPerformanceMetrics::default(),
            config,
            trends_analyzer: PerformanceTrendsAnalyzer::new(),
            real_time_monitor: RealTimePerformanceMonitor::new(),
        }
    }

    /// Record constraint evaluation performance
    pub fn record_constraint_evaluation(
        &mut self,
        constraint_id: &ConstraintComponentId,
        _constraint: &Constraint,
        result: &ConstraintEvaluationResult,
        execution_time: Duration,
        memory_usage: usize,
    ) {
        // Update constraint metrics
        let metrics = self
            .constraint_metrics
            .entry(constraint_id.clone())
            .or_insert_with(ConstraintPerformanceMetrics::new);

        metrics.update(execution_time, result.is_satisfied(), memory_usage);

        // Update global metrics
        self.global_metrics.total_constraints_evaluated += 1;
        self.global_metrics.total_validation_time += execution_time;

        // Record for real-time monitoring
        self.real_time_monitor
            .record_operation(PerformanceOperation {
                timestamp: Instant::now(),
                operation_type: OperationType::ConstraintEvaluation(constraint_id.clone()),
                execution_time,
                memory_usage,
                success: result.is_satisfied(),
            });

        // Apply adaptive threshold adjustment
        if self.config.enable_adaptive_thresholds {
            self.adjust_adaptive_thresholds(constraint_id, execution_time);
        }

        // Update performance trends
        self.update_performance_trends(execution_time, memory_usage);
    }

    /// Record shape validation performance
    pub fn record_shape_validation(
        &mut self,
        shape_id: &ShapeId,
        execution_time: Duration,
        focus_nodes_count: usize,
        violations_count: usize,
        memory_usage: f64,
    ) {
        let metrics = self
            .shape_metrics
            .entry(shape_id.clone())
            .or_insert_with(ShapePerformanceMetrics::new);

        metrics.update(
            execution_time,
            focus_nodes_count,
            violations_count,
            memory_usage,
        );

        // Update global metrics
        self.global_metrics.total_shapes_evaluated += 1;
        self.global_metrics.peak_memory_usage_mb =
            self.global_metrics.peak_memory_usage_mb.max(memory_usage);

        // Record for real-time monitoring
        self.real_time_monitor
            .record_operation(PerformanceOperation {
                timestamp: Instant::now(),
                operation_type: OperationType::ShapeValidation(shape_id.clone()),
                execution_time,
                memory_usage: memory_usage as usize,
                success: violations_count == 0,
            });
    }

    /// Get performance predictions for optimization
    pub fn get_performance_predictions(
        &self,
        constraint_id: &ConstraintComponentId,
    ) -> Option<PerformancePrediction> {
        if !self.config.enable_predictive_optimization {
            return None;
        }

        self.constraint_metrics
            .get(constraint_id)
            .map(|metrics| PerformancePrediction {
                expected_execution_time: metrics.avg_execution_time,
                confidence_interval: (metrics.min_execution_time, metrics.max_execution_time),
                predicted_success_rate: metrics.success_rate,
                predicted_selectivity: metrics.selectivity,
                trend: metrics.performance_trend.clone(),
            })
    }

    /// Adjust adaptive thresholds based on recent performance
    fn adjust_adaptive_thresholds(
        &mut self,
        constraint_id: &ConstraintComponentId,
        execution_time: Duration,
    ) {
        if let Some(metrics) = self.constraint_metrics.get_mut(constraint_id) {
            // Calculate new threshold based on recent performance
            let current_threshold = metrics.adaptive_threshold;
            let adjustment_factor = self.config.adjustment_sensitivity;

            // If execution time is consistently above threshold, adjust upward
            if execution_time > current_threshold {
                let adjustment = execution_time.saturating_sub(current_threshold);
                let new_threshold = current_threshold
                    + Duration::from_nanos(
                        (adjustment.as_nanos() as f64 * adjustment_factor) as u64,
                    );
                metrics.adaptive_threshold = new_threshold;
            }
            // If execution time is consistently below threshold, adjust downward gradually
            else if execution_time < current_threshold {
                let adjustment = current_threshold.saturating_sub(execution_time);
                let new_threshold = current_threshold
                    - Duration::from_nanos(
                        (adjustment.as_nanos() as f64 * adjustment_factor * 0.1) as u64,
                    );
                metrics.adaptive_threshold = new_threshold;
            }
        }
    }

    /// Update performance trends
    fn update_performance_trends(&mut self, execution_time: Duration, memory_usage: usize) {
        let data_point = PerformanceDataPoint {
            timestamp: Instant::now(),
            avg_execution_time: execution_time,
            success_rate: self.global_metrics.overall_success_rate,
            throughput: self.global_metrics.avg_throughput,
            memory_usage_mb: memory_usage as f64 / (1024.0 * 1024.0),
        };

        self.trends_analyzer.add_data_point(data_point);
    }

    /// Get comprehensive performance report
    pub fn get_performance_report(&self) -> PerformanceReport {
        PerformanceReport {
            global_metrics: self.global_metrics.clone(),
            constraint_metrics: self.constraint_metrics.clone(),
            shape_metrics: self.shape_metrics.clone(),
            active_alerts: self.real_time_monitor.active_alerts.clone(),
            optimization_recommendations: self.get_optimization_recommendations(),
            performance_trends: self.trends_analyzer.get_current_trends(),
        }
    }

    /// Get optimization recommendations based on performance analysis
    fn get_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze constraint performance for recommendations
        for (constraint_id, metrics) in &self.constraint_metrics {
            if metrics.avg_execution_time > Duration::from_millis(100) {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::OptimizeConstraint,
                    target: constraint_id.as_str().to_string(),
                    description: format!(
                        "Constraint {} has high average execution time ({:?})",
                        constraint_id, metrics.avg_execution_time
                    ),
                    priority: if metrics.avg_execution_time > Duration::from_millis(1000) {
                        RecommendationPriority::High
                    } else {
                        RecommendationPriority::Medium
                    },
                    estimated_improvement: 20.0, // Percentage
                });
            }

            if metrics.selectivity > 0.8 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::EarlyTermination,
                    target: constraint_id.as_str().to_string(),
                    description: format!(
                        "Constraint {} has high selectivity ({:.2}), consider early termination",
                        constraint_id, metrics.selectivity
                    ),
                    priority: RecommendationPriority::Medium,
                    estimated_improvement: 15.0,
                });
            }
        }

        recommendations
    }

    /// Get current performance status
    pub fn get_performance_status(&self) -> PerformanceStatus {
        let total_operations = self.global_metrics.total_constraints_evaluated
            + self.global_metrics.total_shapes_evaluated;

        PerformanceStatus {
            overall_health: if self.global_metrics.overall_success_rate > 0.95 {
                PerformanceHealth::Good
            } else if self.global_metrics.overall_success_rate > 0.8 {
                PerformanceHealth::Fair
            } else {
                PerformanceHealth::Poor
            },
            total_operations,
            avg_response_time: if total_operations > 0 {
                self.global_metrics.total_validation_time / total_operations as u32
            } else {
                Duration::ZERO
            },
            current_throughput: self.global_metrics.avg_throughput,
            memory_usage: self.global_metrics.peak_memory_usage_mb,
            active_alerts_count: self.real_time_monitor.active_alerts.len(),
        }
    }
}

/// Performance prediction result
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    /// Expected execution time
    pub expected_execution_time: Duration,
    /// Confidence interval (min, max)
    pub confidence_interval: (Duration, Duration),
    /// Predicted success rate
    pub predicted_success_rate: f64,
    /// Predicted selectivity
    pub predicted_selectivity: f64,
    /// Performance trend
    pub trend: PerformanceTrend,
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Global performance metrics
    pub global_metrics: GlobalPerformanceMetrics,
    /// Per-constraint metrics
    pub constraint_metrics: HashMap<ConstraintComponentId, ConstraintPerformanceMetrics>,
    /// Per-shape metrics
    pub shape_metrics: HashMap<ShapeId, ShapePerformanceMetrics>,
    /// Active performance alerts
    pub active_alerts: Vec<PerformanceAlert>,
    /// Optimization recommendations
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    /// Performance trends
    pub performance_trends: HashMap<String, PerformanceTrend>,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    /// Target of recommendation (constraint ID, shape ID, etc.)
    pub target: String,
    /// Description of the recommendation
    pub description: String,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Estimated performance improvement (percentage)
    pub estimated_improvement: f64,
}

/// Types of optimization recommendations
#[derive(Debug, Clone)]
pub enum RecommendationType {
    /// Optimize specific constraint
    OptimizeConstraint,
    /// Enable early termination
    EarlyTermination,
    /// Improve caching
    ImproveCache,
    /// Parallel execution
    ParallelExecution,
    /// Memory optimization
    MemoryOptimization,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Current performance status
#[derive(Debug, Clone)]
pub struct PerformanceStatus {
    /// Overall performance health
    pub overall_health: PerformanceHealth,
    /// Total operations performed
    pub total_operations: usize,
    /// Average response time
    pub avg_response_time: Duration,
    /// Current throughput
    pub current_throughput: f64,
    /// Current memory usage
    pub memory_usage: f64,
    /// Number of active alerts
    pub active_alerts_count: usize,
}

/// Performance health indicators
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformanceHealth {
    /// Performance is good
    Good,
    /// Performance is fair
    Fair,
    /// Performance is poor
    Poor,
    /// Performance status unknown
    Unknown,
}

// Implementation of helper methods
impl ConstraintPerformanceMetrics {
    fn new() -> Self {
        Self {
            avg_execution_time: Duration::ZERO,
            execution_time_std_dev: Duration::ZERO,
            min_execution_time: Duration::MAX,
            max_execution_time: Duration::ZERO,
            success_rate: 1.0,
            selectivity: 0.0,
            total_evaluations: 0,
            adaptive_threshold: Duration::from_millis(10),
            performance_trend: PerformanceTrend::Unknown,
            last_evaluation: Instant::now(),
        }
    }

    fn update(&mut self, execution_time: Duration, success: bool, _memory_usage: usize) {
        self.total_evaluations += 1;

        // Update execution time statistics
        let total_time = self.avg_execution_time * self.total_evaluations.saturating_sub(1) as u32
            + execution_time;
        self.avg_execution_time = total_time / self.total_evaluations as u32;

        self.min_execution_time = self.min_execution_time.min(execution_time);
        self.max_execution_time = self.max_execution_time.max(execution_time);

        // Update success rate
        let success_count = (self.success_rate * (self.total_evaluations - 1) as f64)
            + if success { 1.0 } else { 0.0 };
        self.success_rate = success_count / self.total_evaluations as f64;

        // Update selectivity (assuming success means no violation for simplicity)
        let violation_count = ((1.0 - self.selectivity) * (self.total_evaluations - 1) as f64)
            + if !success { 1.0 } else { 0.0 };
        self.selectivity = violation_count / self.total_evaluations as f64;

        self.last_evaluation = Instant::now();
    }
}

impl ShapePerformanceMetrics {
    fn new() -> Self {
        Self {
            total_validation_time: Duration::ZERO,
            avg_validation_time: Duration::ZERO,
            focus_nodes_validated: 0,
            total_violations: 0,
            constraint_breakdown: HashMap::new(),
            efficiency_score: 1.0,
            memory_usage_mb: 0.0,
        }
    }

    fn update(
        &mut self,
        execution_time: Duration,
        focus_nodes: usize,
        violations: usize,
        memory_usage: f64,
    ) {
        self.total_validation_time += execution_time;
        self.focus_nodes_validated += focus_nodes;
        self.total_violations += violations;
        self.memory_usage_mb = memory_usage;

        if self.focus_nodes_validated > 0 {
            self.avg_validation_time =
                self.total_validation_time / self.focus_nodes_validated as u32;
        }

        // Calculate efficiency score (lower violation rate and faster validation = higher score)
        let violation_rate = if self.focus_nodes_validated > 0 {
            self.total_violations as f64 / self.focus_nodes_validated as f64
        } else {
            0.0
        };

        self.efficiency_score = (1.0 - violation_rate).max(0.0);
    }
}

impl Default for GlobalPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_validation_time: Duration::ZERO,
            total_shapes_evaluated: 0,
            total_constraints_evaluated: 0,
            overall_success_rate: 1.0,
            avg_throughput: 0.0,
            peak_memory_usage_mb: 0.0,
            optimization_score: 1.0,
        }
    }
}

impl PerformanceTrendsAnalyzer {
    fn new() -> Self {
        Self {
            performance_history: Vec::new(),
            config: TrendAnalysisConfig {
                trend_window_size: 100,
                min_data_points: 10,
                trend_sensitivity: 0.05,
            },
        }
    }

    fn add_data_point(&mut self, data_point: PerformanceDataPoint) {
        self.performance_history.push(data_point);

        // Keep only recent history within window size
        if self.performance_history.len() > self.config.trend_window_size {
            self.performance_history.remove(0);
        }
    }

    fn get_current_trends(&self) -> HashMap<String, PerformanceTrend> {
        let mut trends = HashMap::new();

        if self.performance_history.len() >= self.config.min_data_points {
            trends.insert(
                "execution_time".to_string(),
                self.analyze_execution_time_trend(),
            );
            trends.insert(
                "success_rate".to_string(),
                self.analyze_success_rate_trend(),
            );
            trends.insert("throughput".to_string(), self.analyze_throughput_trend());
        }

        trends
    }

    fn analyze_execution_time_trend(&self) -> PerformanceTrend {
        // Simple linear regression on execution times
        let times: Vec<f64> = self
            .performance_history
            .iter()
            .map(|dp| dp.avg_execution_time.as_secs_f64())
            .collect();

        self.calculate_trend(&times)
    }

    fn analyze_success_rate_trend(&self) -> PerformanceTrend {
        let rates: Vec<f64> = self
            .performance_history
            .iter()
            .map(|dp| dp.success_rate)
            .collect();

        self.calculate_trend(&rates)
    }

    fn analyze_throughput_trend(&self) -> PerformanceTrend {
        let throughputs: Vec<f64> = self
            .performance_history
            .iter()
            .map(|dp| dp.throughput)
            .collect();

        self.calculate_trend(&throughputs)
    }

    fn calculate_trend(&self, values: &[f64]) -> PerformanceTrend {
        if values.len() < 2 {
            return PerformanceTrend::Unknown;
        }

        // Simple slope calculation
        let n = values.len() as f64;
        let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));

        if slope.abs() < self.config.trend_sensitivity {
            PerformanceTrend::Stable
        } else if slope > 0.0 {
            PerformanceTrend::Improving(slope)
        } else {
            PerformanceTrend::Degrading(-slope)
        }
    }
}

impl RealTimePerformanceMonitor {
    fn new() -> Self {
        Self {
            current_window: PerformanceWindow {
                start_time: Instant::now(),
                duration: Duration::from_secs(60),
                operations: Vec::new(),
            },
            alert_thresholds: AlertThresholds {
                max_execution_time: Duration::from_millis(1000),
                min_success_rate: 0.8,
                max_memory_usage_mb: 1024.0,
                min_throughput: 10.0,
            },
            active_alerts: Vec::new(),
        }
    }

    fn record_operation(&mut self, operation: PerformanceOperation) {
        // Check if we need to start a new window
        if operation
            .timestamp
            .duration_since(self.current_window.start_time)
            > self.current_window.duration
        {
            self.analyze_window_and_generate_alerts();
            self.start_new_window(operation.timestamp);
        }

        self.current_window.operations.push(operation);
    }

    fn analyze_window_and_generate_alerts(&mut self) {
        // Analyze current window for performance issues
        let operations = &self.current_window.operations;
        if operations.is_empty() {
            return;
        }

        // Check average execution time
        let avg_execution_time: Duration = operations
            .iter()
            .map(|op| op.execution_time)
            .sum::<Duration>()
            / operations.len() as u32;

        if avg_execution_time > self.alert_thresholds.max_execution_time {
            self.active_alerts.push(PerformanceAlert {
                alert_type: AlertType::HighExecutionTime,
                message: format!(
                    "Average execution time ({:?}) exceeds threshold ({:?})",
                    avg_execution_time, self.alert_thresholds.max_execution_time
                ),
                severity: AlertSeverity::Warning,
                timestamp: Instant::now(),
                suggested_actions: vec![
                    "Consider enabling constraint caching".to_string(),
                    "Review constraint ordering optimization".to_string(),
                    "Check for expensive SPARQL constraints".to_string(),
                ],
            });
        }

        // Check success rate
        let success_rate =
            operations.iter().filter(|op| op.success).count() as f64 / operations.len() as f64;

        if success_rate < self.alert_thresholds.min_success_rate {
            self.active_alerts.push(PerformanceAlert {
                alert_type: AlertType::LowSuccessRate,
                message: format!(
                    "Success rate ({:.2}) below threshold ({:.2})",
                    success_rate, self.alert_thresholds.min_success_rate
                ),
                severity: AlertSeverity::Critical,
                timestamp: Instant::now(),
                suggested_actions: vec![
                    "Review validation logic for errors".to_string(),
                    "Check data quality".to_string(),
                    "Verify shape definitions".to_string(),
                ],
            });
        }
    }

    fn start_new_window(&mut self, start_time: Instant) {
        self.current_window = PerformanceWindow {
            start_time,
            duration: Duration::from_secs(60),
            operations: Vec::new(),
        };
    }
}
