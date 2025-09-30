//! Analytics and insights engine for SHACL validation
//!
//! This module implements comprehensive analytics for SHACL validation operations,
//! performance monitoring, and data quality insights.

pub mod config;
pub mod engine;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export all public types and structs
pub use config::*;
pub use engine::*;
pub use types::{
    ActionableRecommendation, Alert, AnalysisPeriod, AnalysisPeriodInfo, AnalyticsStatistics,
    AnalyticsTrainingData, ChartData, ChartDataPoint, DashboardData, EstimatedEffort,
    ExecutionTimeAnalysis, InsightSeverity, InsightsSummary, MemoryUsageAnalysis, OverallHealth,
    OverviewMetrics, PerformanceAnalysis, PerformanceBottleneckInfo, PerformanceInsightInfo,
    QualityMetricsInfo, RecommendationCategory, RecommendationPriority, ThroughputAnalysis, Trend,
    TrendAnalysis, TrendDirection, TrendIndicator, ValidationInsights,
};
