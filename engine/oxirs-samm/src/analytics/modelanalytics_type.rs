//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::analytics::{
    Anomaly, BenchmarkComparison, BestPracticeReport, ComplexityAssessment, DependencyMetrics,
    DistributionAnalysis, Recommendation,
};
use serde::{Deserialize, Serialize};

/// Comprehensive model analytics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAnalytics {
    /// Overall quality score (0-100)
    pub quality_score: f64,
    /// Complexity assessment across multiple dimensions
    pub complexity_assessment: ComplexityAssessment,
    /// Best practice compliance
    pub best_practices: BestPracticeReport,
    /// Statistical distributions
    pub distributions: DistributionAnalysis,
    /// Dependency and coupling metrics
    pub dependency_metrics: DependencyMetrics,
    /// Detected anomalies
    pub anomalies: Vec<Anomaly>,
    /// Actionable recommendations
    pub recommendations: Vec<Recommendation>,
    /// Benchmarking against industry standards
    pub benchmark: BenchmarkComparison,
}
