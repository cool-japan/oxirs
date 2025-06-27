//! Type definitions for analytics engine

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use oxirs_shacl::{ShapeId, ValidationReport};

use crate::insights::{PerformanceInsight, QualityInsight, ValidationInsight};

/// Validation insights container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationInsights {
    /// Validation pattern insights
    pub validation_insights: Vec<ValidationInsight>,

    /// Performance insights
    pub performance_insights: Vec<PerformanceInsight>,

    /// Quality insights
    pub quality_insights: Vec<QualityInsight>,

    /// Trend analysis
    pub trend_analysis: Option<TrendAnalysis>,

    /// Actionable recommendations
    pub recommendations: Vec<ActionableRecommendation>,

    /// Summary of insights
    pub summary: InsightsSummary,

    /// When insights were generated
    pub generation_timestamp: chrono::DateTime<chrono::Utc>,
}

impl ValidationInsights {
    pub fn new() -> Self {
        Self {
            validation_insights: Vec::new(),
            performance_insights: Vec::new(),
            quality_insights: Vec::new(),
            trend_analysis: None,
            recommendations: Vec::new(),
            summary: InsightsSummary::default(),
            generation_timestamp: chrono::Utc::now(),
        }
    }
}

impl Default for ValidationInsights {
    fn default() -> Self {
        Self::new()
    }
}

/// Trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Individual trends
    pub trends: Vec<Trend>,

    /// Overall trend direction
    pub overall_trend: TrendDirection,

    /// Confidence in trend analysis
    pub trend_confidence: f64,

    /// Analysis period
    pub analysis_period: AnalysisPeriod,
}

/// Individual trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trend {
    /// Metric name
    pub metric_name: String,

    /// Trend direction
    pub trend_direction: TrendDirection,

    /// Magnitude of change
    pub magnitude: f64,

    /// Confidence in trend
    pub confidence: f64,

    /// Time period
    pub time_period: String,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Analysis period
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisPeriod {
    Short,
    Medium,
    Long,
}

/// Actionable recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableRecommendation {
    /// Recommendation category
    pub category: RecommendationCategory,

    /// Priority level
    pub priority: RecommendationPriority,

    /// Recommendation title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Specific actions to take
    pub actions: Vec<String>,

    /// Estimated impact (0.0 - 1.0)
    pub estimated_impact: f64,

    /// Estimated effort required
    pub estimated_effort: EstimatedEffort,

    /// Confidence in recommendation
    pub confidence: f64,
}

/// Recommendation categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Validation,
    Performance,
    Quality,
    TrendReversal,
    Optimization,
}

/// Recommendation priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Estimated effort
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EstimatedEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Insights summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InsightsSummary {
    /// Total number of insights
    pub total_insights: usize,

    /// Number of critical issues
    pub critical_issues: usize,

    /// Number of high-priority issues
    pub high_priority_issues: usize,

    /// Overall health assessment
    pub overall_health: OverallHealth,

    /// Key findings
    pub key_findings: Vec<String>,

    /// Priority actions
    pub priority_actions: Vec<String>,
}

/// Overall health assessment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverallHealth {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

impl Default for OverallHealth {
    fn default() -> Self {
        Self::Good
    }
}

/// Insight severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum InsightSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    /// Execution time analysis
    pub execution_time_analysis: ExecutionTimeAnalysis,

    /// Memory usage analysis
    pub memory_analysis: MemoryUsageAnalysis,

    /// Throughput analysis
    pub throughput_analysis: ThroughputAnalysis,

    /// Performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneckInfo>,

    /// Performance insights
    pub insights: Vec<PerformanceInsightInfo>,

    /// Analysis period
    pub analysis_period: AnalysisPeriodInfo,

    /// Analysis execution time
    pub analysis_time: Duration,
}

/// Dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    /// Overview metrics
    pub overview_metrics: OverviewMetrics,

    /// Performance charts
    pub performance_charts: Vec<ChartData>,

    /// Quality metrics
    pub quality_metrics: QualityMetricsInfo,

    /// Trend indicators
    pub trend_indicators: Vec<TrendIndicator>,

    /// Active alerts
    pub alerts: Vec<Alert>,

    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl DashboardData {
    pub fn new() -> Self {
        Self {
            overview_metrics: OverviewMetrics::default(),
            performance_charts: Vec::new(),
            quality_metrics: QualityMetricsInfo::default(),
            trend_indicators: Vec::new(),
            alerts: Vec::new(),
            last_updated: chrono::Utc::now(),
        }
    }
}

impl Default for DashboardData {
    fn default() -> Self {
        Self::new()
    }
}

/// Overview metrics for dashboard
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OverviewMetrics {
    pub total_validations: usize,
    pub success_rate: f64,
    pub avg_execution_time: Duration,
    pub total_violations: usize,
}

/// Chart data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    pub chart_type: String,
    pub title: String,
    pub data_points: Vec<ChartDataPoint>,
}

/// Chart data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartDataPoint {
    pub x: chrono::DateTime<chrono::Utc>,
    pub y: f64,
}

/// Quality metrics info
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityMetricsInfo {
    pub completeness_score: f64,
    pub consistency_score: f64,
    pub accuracy_score: f64,
    pub conformance_score: f64,
}

/// Trend indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendIndicator {
    pub metric_name: String,
    pub current_value: f64,
    pub trend_direction: TrendDirection,
    pub change_percentage: f64,
}

/// Alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub alert_type: String,
    pub severity: String,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Performance data point
#[derive(Debug, Clone)]
pub(super) struct PerformanceDataPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub execution_time: Duration,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: u8,
    pub violation_count: u32,
    pub success: bool,
}

/// Execution time analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTimeAnalysis {
    pub average_time: Duration,
    pub trend_direction: TrendDirection,
    pub variability: f64,
}

/// Memory usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageAnalysis {
    pub average_usage_mb: u64,
    pub peak_usage_mb: u64,
    pub trend_direction: TrendDirection,
}

/// Throughput analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputAnalysis {
    pub validations_per_hour: f64,
    pub trend_direction: TrendDirection,
}

/// Performance bottleneck info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneckInfo {
    pub bottleneck_type: String,
    pub description: String,
    pub severity: String,
    pub affected_percentage: f64,
}

/// Performance insight info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInsightInfo {
    pub insight_type: String,
    pub description: String,
    pub metric_value: f64,
    pub trend_direction: TrendDirection,
    pub confidence: f64,
}

/// Analysis period info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisPeriodInfo {
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub total_validations: usize,
    pub period_type: String,
}

/// Violation pattern
#[derive(Debug, Clone)]
pub(super) struct ViolationPattern {
    pub constraint_type: String,
    pub failure_rate: f64,
    pub confidence: f64,
    pub affected_shapes: Vec<ShapeId>,
    pub recommendations: Vec<String>,
    pub supporting_data: HashMap<String, String>,
}

/// Metrics collector
#[derive(Debug)]
pub(super) struct MetricsCollector {
    pub collection_interval: Duration,
    pub metrics_buffer: Vec<MetricData>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            collection_interval: Duration::from_secs(60),
            metrics_buffer: Vec::new(),
        }
    }
}

/// Metric data
#[derive(Debug, Clone)]
pub(super) struct MetricData {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metric_name: String,
    pub value: f64,
    pub tags: HashMap<String, String>,
}

/// Analytics model state
#[derive(Debug)]
pub(super) struct AnalyticsModelState {
    pub version: String,
    pub accuracy: f64,
    pub loss: f64,
    pub training_epochs: usize,
    pub last_training: Option<chrono::DateTime<chrono::Utc>>,
}

impl AnalyticsModelState {
    pub fn new() -> Self {
        Self {
            version: "1.0.0".to_string(),
            accuracy: 0.75,
            loss: 0.25,
            training_epochs: 0,
            last_training: None,
        }
    }
}

/// Cached analytics result
#[derive(Debug, Clone)]
pub(super) struct CachedAnalytics {
    pub result: CachedAnalyticsResult,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub ttl: Duration,
}

impl CachedAnalytics {
    pub fn is_expired(&self) -> bool {
        let now = chrono::Utc::now();
        let expiry = self.timestamp + chrono::Duration::from_std(self.ttl).unwrap_or_default();
        now > expiry
    }
}

/// Cached analytics result types
#[derive(Debug, Clone)]
pub(super) enum CachedAnalyticsResult {
    ValidationInsights(ValidationInsights),
    PerformanceAnalysis(PerformanceAnalysis),
    DashboardData(DashboardData),
}

/// Analytics statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalyticsStatistics {
    pub total_insights_generated: usize,
    pub total_analysis_time: Duration,
    pub performance_analyses: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub model_trained: bool,
}

/// Training data for analytics models
#[derive(Debug, Clone)]
pub struct AnalyticsTrainingData {
    pub examples: Vec<AnalyticsExample>,
    pub validation_examples: Vec<AnalyticsExample>,
}

/// Training example for analytics
#[derive(Debug, Clone)]
pub struct AnalyticsExample {
    pub validation_data: Vec<ValidationReport>,
    pub expected_insights: ValidationInsights,
    pub context_metadata: HashMap<String, String>,
}
