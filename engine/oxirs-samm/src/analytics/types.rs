//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Individual best practice check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPracticeCheck {
    /// Check name
    pub name: String,
    /// Whether check passed
    pub passed: bool,
    /// Check category
    pub category: CheckCategory,
    /// Details/explanation
    pub details: String,
}
/// Statistical quality test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTest {
    /// Whether the model passes quality thresholds
    pub passes_threshold: bool,
    /// Confidence level (0-1)
    pub confidence_level: f64,
    /// Coefficient of variation check passed
    pub cv_check: bool,
    /// Skewness check passed
    pub skewness_check: bool,
    /// Score check passed
    pub score_check: bool,
    /// Detailed test results
    pub details: String,
}
/// Correlation insight between two features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationInsight {
    /// First feature name
    pub feature1: String,
    /// Second feature name
    pub feature2: String,
    /// Correlation coefficient (-1 to 1)
    pub coefficient: f64,
    /// Strength of correlation
    pub strength: CorrelationStrength,
    /// Direction of correlation
    pub direction: CorrelationDirection,
    /// Human-readable interpretation
    pub interpretation: String,
}
/// Benchmark comparison against industry standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    /// How model compares to typical models
    pub comparison: BenchmarkLevel,
    /// Properties count percentile (0-100)
    pub property_count_percentile: f64,
    /// Complexity percentile (0-100)
    pub complexity_percentile: f64,
    /// Documentation percentile (0-100)
    pub documentation_percentile: f64,
}
/// Strength of correlation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrelationStrength {
    /// Weak correlation (0.3 < |r| <= 0.5)
    Weak,
    /// Moderate correlation (0.5 < |r| <= 0.7)
    Moderate,
    /// Strong correlation (|r| > 0.7)
    Strong,
}
/// Statistical anomaly detected using robust methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnomaly {
    /// Name of the metric that triggered the anomaly
    pub metric_name: String,
    /// Description of the anomaly
    pub description: String,
    /// Deviation score (how far from normal)
    pub deviation_score: f64,
    /// Severity level
    pub severity: Severity,
}
/// Property correlation matrix with insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyCorrelationMatrix {
    /// Feature names (row/column labels)
    pub feature_names: Vec<String>,
    /// Correlation matrix (symmetric)
    pub correlation_matrix: Vec<Vec<f64>>,
    /// Significant correlation insights
    pub insights: Vec<CorrelationInsight>,
    /// Correlation method used (e.g., "Pearson", "Spearman")
    pub method: String,
}
/// Severity level for anomalies and recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Error (should be fixed)
    Error,
    /// Critical (must be fixed)
    Critical,
}
/// Detected anomaly in model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: Severity,
    /// Location (URN or description)
    pub location: String,
    /// Description of the anomaly
    pub description: String,
}
/// Direction of correlation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrelationDirection {
    /// Positive correlation (r > 0)
    Positive,
    /// Negative correlation (r < 0)
    Negative,
}
/// Statistical distribution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionStats {
    /// Mean value
    pub mean: f64,
    /// Variance
    pub variance: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
}
/// Actionable recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation type
    pub rec_type: RecommendationType,
    /// Severity level
    pub severity: Severity,
    /// Affected element URN
    pub target: String,
    /// Recommendation message
    pub message: String,
    /// Suggested action
    pub suggested_action: String,
}
/// Dependency and coupling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyMetrics {
    /// Total number of dependencies
    pub total_dependencies: usize,
    /// Average dependencies per property
    pub avg_dependencies_per_property: f64,
    /// Maximum dependency depth
    pub max_dependency_depth: usize,
    /// Coupling factor (0-1)
    pub coupling_factor: f64,
    /// Cohesion score (0-1)
    pub cohesion_score: f64,
    /// Circular dependency count
    pub circular_dependencies: usize,
}
/// Statistical distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    /// Property count distribution
    pub property_distribution: DistributionStats,
    /// Type usage frequency
    pub type_distribution: HashMap<String, usize>,
    /// Characteristic kind distribution
    pub characteristic_distribution: HashMap<String, usize>,
    /// Optional vs required ratio
    pub optionality_ratio: f64,
    /// Collection usage percentage
    pub collection_percentage: f64,
}
/// Best practice compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPracticeReport {
    /// Number of checks passed
    pub passed_checks: usize,
    /// Total number of checks
    pub total_checks: usize,
    /// Compliance percentage (0-100)
    pub compliance_percentage: f64,
    /// Detailed check results
    pub checks: Vec<BestPracticeCheck>,
}
/// Benchmark level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkLevel {
    /// Below average
    BelowAverage,
    /// Average
    Average,
    /// Above average
    AboveAverage,
    /// Excellent
    Excellent,
}
/// Type of detected anomaly
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Unusually high property count
    HighPropertyCount,
    /// Missing documentation
    MissingDocumentation,
    /// Inconsistent naming
    InconsistentNaming,
    /// Deep nesting
    DeepNesting,
    /// High coupling
    HighCoupling,
    /// Unused entity
    UnusedEntity,
    /// Duplicate patterns
    DuplicatePatterns,
}
/// Advanced statistical metrics computed using scirs2-stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalMetrics {
    /// Mean value
    pub mean: f64,
    /// Median value (robust to outliers)
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Variance
    pub variance: f64,
    /// Mean absolute deviation
    pub mean_abs_deviation: f64,
    /// Median absolute deviation (robust dispersion)
    pub median_abs_deviation: f64,
    /// Interquartile range (Q3 - Q1)
    pub interquartile_range: f64,
    /// Coefficient of variation (std/mean)
    pub coefficient_variation: f64,
    /// Skewness (distribution asymmetry)
    pub skewness: f64,
    /// Kurtosis (distribution tail weight)
    pub kurtosis: f64,
}
/// Type of recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Refactoring suggestion
    Refactoring,
    /// Documentation improvement
    Documentation,
    /// Naming convention fix
    Naming,
    /// Complexity reduction
    ComplexityReduction,
    /// Performance optimization
    Performance,
    /// Best practice alignment
    BestPractice,
}
/// Multi-dimensional complexity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAssessment {
    /// Structural complexity (0-100)
    pub structural: f64,
    /// Cognitive complexity (0-100) - how hard to understand
    pub cognitive: f64,
    /// Cyclomatic complexity
    pub cyclomatic: f64,
    /// Coupling complexity
    pub coupling: f64,
    /// Overall complexity level
    pub overall_level: ComplexityLevel,
}
/// Complexity level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityLevel {
    /// Low complexity (<= 20)
    Low,
    /// Medium complexity (21-50)
    Medium,
    /// High complexity (51-80)
    High,
    /// Very high complexity (> 80)
    VeryHigh,
}
/// Best practice check category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckCategory {
    /// Naming conventions
    Naming,
    /// Documentation completeness
    Documentation,
    /// Structural patterns
    Structure,
    /// Type usage
    Types,
    /// Metadata quality
    Metadata,
}
