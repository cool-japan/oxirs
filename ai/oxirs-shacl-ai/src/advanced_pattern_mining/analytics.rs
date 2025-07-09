//! Analytics and performance monitoring for pattern mining

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

use super::types::*;

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Type of detected anomaly
    pub anomaly_type: AnomalyType,

    /// Severity level
    pub severity: AnomalySeverity,

    /// Confidence in detection
    pub confidence: f64,

    /// Description of the anomaly
    pub description: String,

    /// Suggested actions
    pub suggested_actions: Vec<String>,

    /// Affected patterns or components
    pub affected_items: Vec<String>,
}

/// Distribution analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    /// Pattern type distribution
    pub pattern_type_distribution: HashMap<String, usize>,

    /// Quality score distribution
    pub quality_distribution: QualityDistribution,

    /// Coverage analysis
    pub coverage_analysis: CoverageAnalysis,

    /// Temporal distribution
    pub temporal_distribution: TemporalDistribution,

    /// Hierarchical distribution
    pub hierarchical_distribution: HierarchicalDistribution,
}

/// Quality score distribution buckets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDistribution {
    /// Excellent patterns (>0.9)
    pub excellent: usize,

    /// Good patterns (0.7-0.9)
    pub good: usize,

    /// Fair patterns (0.5-0.7)
    pub fair: usize,

    /// Poor patterns (<0.5)
    pub poor: usize,

    /// Average quality score
    pub average_quality: f64,

    /// Quality standard deviation
    pub quality_std_dev: f64,
}

/// Coverage analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageAnalysis {
    /// Overall coverage ratio
    pub overall_coverage: f64,

    /// Class coverage distribution
    pub class_coverage: HashMap<String, f64>,

    /// Property coverage distribution
    pub property_coverage: HashMap<String, f64>,

    /// Efficiency score
    pub efficiency_score: f64,
}

/// Temporal distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDistribution {
    /// Patterns by time granularity
    pub granularity_distribution: HashMap<TimeGranularity, usize>,

    /// Seasonal patterns count
    pub seasonal_patterns: usize,

    /// Trending patterns count
    pub trending_patterns: usize,

    /// Stable patterns count
    pub stable_patterns: usize,
}

/// Hierarchical distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalDistribution {
    /// Patterns by hierarchy level
    pub level_distribution: HashMap<usize, usize>,

    /// Average hierarchy depth
    pub average_depth: f64,

    /// Maximum hierarchy depth
    pub max_depth: usize,

    /// Hierarchical complexity score
    pub complexity_score: f64,
}

/// Comprehensive analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    /// Report generation timestamp
    pub generated_at: SystemTime,

    /// Pattern mining statistics
    pub stats: PatternMiningStats,

    /// Distribution analysis
    pub distribution: DistributionAnalysis,

    /// Detected anomalies
    pub anomalies: Vec<AnomalyResult>,

    /// Performance metrics
    pub performance: PerformanceReport,

    /// Recommendations
    pub recommendations: Vec<AnalysisRecommendation>,

    /// Executive summary
    pub executive_summary: ExecutiveSummary,
}

/// Performance analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Processing performance
    pub processing_performance: ProcessingMetrics,

    /// Memory utilization
    pub memory_utilization: MemoryMetrics,

    /// Cache performance
    pub cache_performance: CachePerformanceMetrics,

    /// Scalability analysis
    pub scalability: ScalabilityMetrics,
}

/// Processing performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Total processing time
    pub total_time_ms: u64,

    /// Patterns per second
    pub patterns_per_second: f64,

    /// Memory per pattern (MB)
    pub memory_per_pattern_mb: f64,

    /// CPU utilization percentage
    pub cpu_utilization: f64,

    /// I/O performance metrics
    pub io_metrics: IoMetrics,
}

/// Memory utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Peak memory usage
    pub peak_memory_mb: f64,

    /// Average memory usage
    pub avg_memory_mb: f64,

    /// Memory efficiency score
    pub efficiency_score: f64,

    /// Garbage collection impact
    pub gc_impact: f64,
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePerformanceMetrics {
    /// Overall hit rate
    pub hit_rate: f64,

    /// Average access time
    pub avg_access_time_ms: f64,

    /// Cache efficiency score
    pub efficiency_score: f64,

    /// Memory utilization
    pub memory_utilization: f64,

    /// Eviction rate
    pub eviction_rate: f64,
}

/// Scalability analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityMetrics {
    /// Data size scaling factor
    pub data_scaling_factor: f64,

    /// Performance degradation rate
    pub degradation_rate: f64,

    /// Recommended maximum data size
    pub max_recommended_size: usize,

    /// Bottleneck identification
    pub bottlenecks: Vec<String>,
}

/// I/O performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoMetrics {
    /// Read operations per second
    pub reads_per_second: f64,

    /// Write operations per second
    pub writes_per_second: f64,

    /// Average read latency
    pub avg_read_latency_ms: f64,

    /// Average write latency
    pub avg_write_latency_ms: f64,
}

/// Analysis recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisRecommendation {
    /// Recommendation category
    pub category: RecommendationCategory,

    /// Priority level
    pub priority: RecommendationPriority,

    /// Recommendation description
    pub description: String,

    /// Expected impact
    pub expected_impact: String,

    /// Implementation complexity
    pub complexity: ImplementationComplexity,

    /// Estimated effort in hours
    pub estimated_effort_hours: f64,
}

/// Recommendation category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Performance,
    Memory,
    Quality,
    Scalability,
    Maintenance,
    Configuration,
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Implementation complexity level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationComplexity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Executive summary of analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    /// Overall health score (0-100)
    pub overall_health_score: f64,

    /// Key findings
    pub key_findings: Vec<String>,

    /// Critical issues
    pub critical_issues: Vec<String>,

    /// Performance summary
    pub performance_summary: String,

    /// Recommended actions
    pub recommended_actions: Vec<String>,

    /// Business impact assessment
    pub business_impact: String,
}