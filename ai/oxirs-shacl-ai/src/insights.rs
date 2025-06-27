//! AI-powered insights for SHACL validation and data quality
//!
//! This module defines various types of insights that can be generated
//! from SHACL validation data, quality assessments, and performance metrics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use oxirs_core::model::{NamedNode, Term};
use oxirs_shacl::ShapeId;

use crate::analytics::{InsightSeverity, TrendDirection};

/// Validation insight - insights derived from validation patterns and results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationInsight {
    /// Type of validation insight
    pub insight_type: ValidationInsightType,

    /// Human-readable title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Severity level
    pub severity: InsightSeverity,

    /// Confidence in this insight (0.0 - 1.0)
    pub confidence: f64,

    /// Shapes affected by this insight
    pub affected_shapes: Vec<ShapeId>,

    /// Actionable recommendations
    pub recommendations: Vec<String>,

    /// Supporting data and evidence
    pub supporting_data: HashMap<String, String>,
}

impl ValidationInsight {
    /// Get the severity of this insight
    pub fn severity(&self) -> &InsightSeverity {
        &self.severity
    }

    /// Check if this is a high-priority insight
    pub fn is_high_priority(&self) -> bool {
        matches!(
            self.severity,
            InsightSeverity::Critical | InsightSeverity::High
        )
    }

    /// Get the primary recommendation
    pub fn primary_recommendation(&self) -> Option<&str> {
        self.recommendations.first().map(|s| s.as_str())
    }
}

/// Types of validation insights
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationInsightType {
    /// Low success rate detected
    LowSuccessRate,

    /// Recurring violation pattern
    ViolationPattern,

    /// Performance degradation
    PerformanceDegradation,

    /// Quality issue pattern
    QualityIssue,

    /// Temporal validation pattern
    TemporalPattern,

    /// Shape complexity issue
    ShapeComplexity,

    /// Target selection inefficiency
    TargetInefficiency,

    /// Constraint ordering issue
    ConstraintOrdering,

    /// Recursive validation issue
    RecursiveValidation,

    /// Custom validation insight
    Custom(String),
}

/// Quality insight - insights about data quality patterns and issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityInsight {
    /// Type of quality insight
    pub insight_type: QualityInsightType,

    /// Human-readable title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Severity level
    pub severity: InsightSeverity,

    /// Confidence in this insight (0.0 - 1.0)
    pub confidence: f64,

    /// Quality dimension affected
    pub quality_dimension: String,

    /// Current quality score for this dimension
    pub current_score: f64,

    /// Trend direction for this quality dimension
    pub trend_direction: TrendDirection,

    /// Actionable recommendations
    pub recommendations: Vec<String>,

    /// Supporting data and evidence
    pub supporting_data: HashMap<String, String>,
}

impl QualityInsight {
    /// Get the severity of this insight
    pub fn severity(&self) -> &InsightSeverity {
        &self.severity
    }

    /// Check if quality is improving
    pub fn is_improving(&self) -> bool {
        matches!(self.trend_direction, TrendDirection::Increasing) && self.current_score > 0.5
    }

    /// Check if quality is degrading
    pub fn is_degrading(&self) -> bool {
        matches!(self.trend_direction, TrendDirection::Decreasing)
    }

    /// Get quality status based on score and trend
    pub fn quality_status(&self) -> QualityStatus {
        match (self.current_score, &self.trend_direction) {
            (score, TrendDirection::Increasing) if score > 0.8 => QualityStatus::Excellent,
            (score, TrendDirection::Stable) if score > 0.8 => QualityStatus::Good,
            (score, _) if score > 0.6 => QualityStatus::Acceptable,
            (score, TrendDirection::Decreasing) if score < 0.4 => QualityStatus::Critical,
            _ => QualityStatus::Poor,
        }
    }
}

/// Types of quality insights
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityInsightType {
    /// Completeness analysis
    Completeness,

    /// Consistency analysis
    Consistency,

    /// Accuracy analysis
    Accuracy,

    /// Validity analysis
    Validity,

    /// Timeliness analysis
    Timeliness,

    /// Uniqueness analysis
    Uniqueness,

    /// Conformity analysis
    Conformity,

    /// Integrity analysis
    Integrity,

    /// Trend analysis
    TrendAnalysis,

    /// Anomaly detection
    AnomalyDetection,

    /// Pattern recognition
    PatternRecognition,

    /// Data profiling
    DataProfiling,

    /// Custom quality insight
    Custom(String),
}

/// Quality status levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityStatus {
    Excellent,
    Good,
    Acceptable,
    Poor,
    Critical,
}

/// Performance insight - insights about validation performance and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInsight {
    /// Type of performance insight
    pub insight_type: PerformanceInsightType,

    /// Human-readable title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Severity level
    pub severity: InsightSeverity,

    /// Confidence in this insight (0.0 - 1.0)
    pub confidence: f64,

    /// Performance metric name
    pub metric_name: String,

    /// Current metric value
    pub current_value: f64,

    /// Trend direction for this metric
    pub trend_direction: TrendDirection,

    /// Actionable recommendations
    pub recommendations: Vec<String>,

    /// Supporting data and evidence
    pub supporting_data: HashMap<String, String>,
}

impl PerformanceInsight {
    /// Get the severity of this insight
    pub fn severity(&self) -> &InsightSeverity {
        &self.severity
    }

    /// Check if performance is improving
    pub fn is_improving(&self) -> bool {
        // For most metrics, decreasing is better (less time, less memory)
        // For throughput, increasing is better
        match self.metric_name.as_str() {
            "throughput" | "success_rate" => {
                matches!(self.trend_direction, TrendDirection::Increasing)
            }
            _ => matches!(self.trend_direction, TrendDirection::Decreasing),
        }
    }

    /// Check if performance is degrading
    pub fn is_degrading(&self) -> bool {
        !self.is_improving() && !matches!(self.trend_direction, TrendDirection::Stable)
    }

    /// Get performance status
    pub fn performance_status(&self) -> PerformanceStatus {
        match (&self.trend_direction, &self.severity) {
            (TrendDirection::Increasing, _) if self.metric_name == "throughput" => {
                PerformanceStatus::Improving
            }
            (TrendDirection::Decreasing, _) if self.metric_name != "throughput" => {
                PerformanceStatus::Improving
            }
            (TrendDirection::Stable, &InsightSeverity::Low | &InsightSeverity::Info) => {
                PerformanceStatus::Stable
            }
            (_, &InsightSeverity::Critical | &InsightSeverity::High) => PerformanceStatus::Critical,
            _ => PerformanceStatus::Degrading,
        }
    }
}

/// Types of performance insights
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceInsightType {
    /// Performance degradation detected
    DegradingPerformance,

    /// Memory usage issue
    MemoryIssue,

    /// Throughput issue
    ThroughputIssue,

    /// Bottleneck detected
    BottleneckDetected,

    /// Optimization opportunity
    OptimizationOpportunity,

    /// Resource exhaustion
    ResourceExhaustion,

    /// Latency issue
    LatencyIssue,

    /// Scalability issue
    ScalabilityIssue,

    /// Efficiency improvement
    EfficiencyImprovement,

    /// Cache optimization
    CacheOptimization,

    /// Parallel processing opportunity
    ParallelProcessing,

    /// Custom performance insight
    Custom(String),
}

/// Performance status levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceStatus {
    Excellent,
    Good,
    Stable,
    Degrading,
    Critical,
    Improving,
}

/// Shape insight - insights about SHACL shape design and effectiveness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeInsight {
    /// Type of shape insight
    pub insight_type: ShapeInsightType,

    /// Human-readable title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Severity level
    pub severity: InsightSeverity,

    /// Confidence in this insight (0.0 - 1.0)
    pub confidence: f64,

    /// Shape ID this insight relates to
    pub shape_id: ShapeId,

    /// Shape effectiveness score (0.0 - 1.0)
    pub effectiveness_score: f64,

    /// Complexity metrics
    pub complexity_metrics: ShapeComplexityMetrics,

    /// Actionable recommendations
    pub recommendations: Vec<String>,

    /// Supporting data and evidence
    pub supporting_data: HashMap<String, String>,
}

impl ShapeInsight {
    /// Get the severity of this insight
    pub fn severity(&self) -> &InsightSeverity {
        &self.severity
    }

    /// Check if shape is well-designed
    pub fn is_well_designed(&self) -> bool {
        self.effectiveness_score > 0.8
            && self.complexity_metrics.overall_complexity < ComplexityLevel::High
    }

    /// Check if shape needs optimization
    pub fn needs_optimization(&self) -> bool {
        self.effectiveness_score < 0.6
            || self.complexity_metrics.overall_complexity >= ComplexityLevel::High
    }
}

/// Types of shape insights
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeInsightType {
    /// Shape is overly complex
    OverlyComplex,

    /// Shape is too permissive
    TooPermissive,

    /// Shape is too restrictive
    TooRestrictive,

    /// Shape has conflicting constraints
    ConflictingConstraints,

    /// Shape has redundant constraints
    RedundantConstraints,

    /// Shape has inefficient target selection
    InefficientTargets,

    /// Shape has poor constraint ordering
    PoorConstraintOrdering,

    /// Shape has unused constraints
    UnusedConstraints,

    /// Shape has missing constraints
    MissingConstraints,

    /// Shape design best practice violation
    DesignViolation,

    /// Shape effectiveness analysis
    EffectivenessAnalysis,

    /// Custom shape insight
    Custom(String),
}

/// Shape complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeComplexityMetrics {
    /// Overall complexity level
    pub overall_complexity: ComplexityLevel,

    /// Number of constraints
    pub constraint_count: u32,

    /// Number of targets
    pub target_count: u32,

    /// Property path complexity
    pub path_complexity: u32,

    /// Nesting depth
    pub nesting_depth: u32,

    /// Cyclic complexity (for recursive shapes)
    pub cyclic_complexity: u32,

    /// Estimated performance impact
    pub performance_impact: PerformanceImpact,
}

/// Complexity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ComplexityLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Performance impact levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceImpact {
    Negligible,
    Low,
    Medium,
    High,
    Severe,
}

/// Data insight - insights about the RDF data being validated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataInsight {
    /// Type of data insight
    pub insight_type: DataInsightType,

    /// Human-readable title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Severity level
    pub severity: InsightSeverity,

    /// Confidence in this insight (0.0 - 1.0)
    pub confidence: f64,

    /// Data elements affected
    pub affected_elements: Vec<Term>,

    /// Data quality impact
    pub quality_impact: QualityImpact,

    /// Suggested data improvements
    pub data_improvements: Vec<String>,

    /// Supporting statistics
    pub statistics: HashMap<String, f64>,
}

/// Types of data insights
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataInsightType {
    /// Missing required data
    MissingData,

    /// Inconsistent data patterns
    InconsistentPatterns,

    /// Data type inconsistencies
    TypeInconsistencies,

    /// Duplicate data detected
    DuplicateData,

    /// Outlier data detected
    OutlierData,

    /// Data distribution anomalies
    DistributionAnomalies,

    /// Schema violations
    SchemaViolations,

    /// Data relationship issues
    RelationshipIssues,

    /// Data freshness issues
    FreshnessIssues,

    /// Data completeness patterns
    CompletenessPatterns,

    /// Custom data insight
    Custom(String),
}

/// Quality impact levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityImpact {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Insight aggregation and collection structures

/// Comprehensive insight collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightCollection {
    /// Validation insights
    pub validation_insights: Vec<ValidationInsight>,

    /// Quality insights
    pub quality_insights: Vec<QualityInsight>,

    /// Performance insights
    pub performance_insights: Vec<PerformanceInsight>,

    /// Shape insights
    pub shape_insights: Vec<ShapeInsight>,

    /// Data insights
    pub data_insights: Vec<DataInsight>,

    /// Collection metadata
    pub metadata: InsightMetadata,
}

impl InsightCollection {
    /// Create a new empty insight collection
    pub fn new() -> Self {
        Self {
            validation_insights: Vec::new(),
            quality_insights: Vec::new(),
            performance_insights: Vec::new(),
            shape_insights: Vec::new(),
            data_insights: Vec::new(),
            metadata: InsightMetadata::new(),
        }
    }

    /// Get total number of insights
    pub fn total_insights(&self) -> usize {
        self.validation_insights.len()
            + self.quality_insights.len()
            + self.performance_insights.len()
            + self.shape_insights.len()
            + self.data_insights.len()
    }

    /// Get high-priority insights count
    pub fn high_priority_count(&self) -> usize {
        let mut count = 0;

        count += self
            .validation_insights
            .iter()
            .filter(|i| i.is_high_priority())
            .count();
        count += self
            .quality_insights
            .iter()
            .filter(|i| {
                matches!(
                    i.severity,
                    InsightSeverity::Critical | InsightSeverity::High
                )
            })
            .count();
        count += self
            .performance_insights
            .iter()
            .filter(|i| {
                matches!(
                    i.severity,
                    InsightSeverity::Critical | InsightSeverity::High
                )
            })
            .count();
        count += self
            .shape_insights
            .iter()
            .filter(|i| {
                matches!(
                    i.severity,
                    InsightSeverity::Critical | InsightSeverity::High
                )
            })
            .count();
        count += self
            .data_insights
            .iter()
            .filter(|i| {
                matches!(
                    i.severity,
                    InsightSeverity::Critical | InsightSeverity::High
                )
            })
            .count();

        count
    }

    /// Get all recommendations from all insights
    pub fn all_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        for insight in &self.validation_insights {
            recommendations.extend(insight.recommendations.clone());
        }
        for insight in &self.quality_insights {
            recommendations.extend(insight.recommendations.clone());
        }
        for insight in &self.performance_insights {
            recommendations.extend(insight.recommendations.clone());
        }
        for insight in &self.shape_insights {
            recommendations.extend(insight.recommendations.clone());
        }
        for insight in &self.data_insights {
            recommendations.extend(insight.data_improvements.clone());
        }

        // Remove duplicates
        recommendations.sort();
        recommendations.dedup();

        recommendations
    }

    /// Filter insights by severity
    pub fn filter_by_severity(&self, min_severity: InsightSeverity) -> InsightCollection {
        let mut filtered = InsightCollection::new();

        filtered.validation_insights = self
            .validation_insights
            .iter()
            .filter(|i| i.severity >= min_severity)
            .cloned()
            .collect();

        filtered.quality_insights = self
            .quality_insights
            .iter()
            .filter(|i| i.severity >= min_severity)
            .cloned()
            .collect();

        filtered.performance_insights = self
            .performance_insights
            .iter()
            .filter(|i| i.severity >= min_severity)
            .cloned()
            .collect();

        filtered.shape_insights = self
            .shape_insights
            .iter()
            .filter(|i| i.severity >= min_severity)
            .cloned()
            .collect();

        filtered.data_insights = self
            .data_insights
            .iter()
            .filter(|i| i.severity >= min_severity)
            .cloned()
            .collect();

        filtered.metadata = self.metadata.clone();
        filtered
    }
}

impl Default for InsightCollection {
    fn default() -> Self {
        Self::new()
    }
}

/// Insight metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightMetadata {
    /// When insights were generated
    pub generation_timestamp: chrono::DateTime<chrono::Utc>,

    /// Version of insight generation engine
    pub engine_version: String,

    /// Analysis parameters used
    pub analysis_parameters: HashMap<String, String>,

    /// Time taken to generate insights
    pub generation_duration: Duration,

    /// Confidence threshold used
    pub confidence_threshold: f64,

    /// Data sources analyzed
    pub data_sources: Vec<String>,
}

impl InsightMetadata {
    pub fn new() -> Self {
        Self {
            generation_timestamp: chrono::Utc::now(),
            engine_version: "1.0.0".to_string(),
            analysis_parameters: HashMap::new(),
            generation_duration: Duration::from_secs(0),
            confidence_threshold: 0.7,
            data_sources: Vec::new(),
        }
    }
}

impl Default for InsightMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Insight summary for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightSummary {
    /// Total insights generated
    pub total_insights: usize,

    /// Breakdown by type
    pub insights_by_type: HashMap<String, usize>,

    /// Breakdown by severity
    pub insights_by_severity: HashMap<InsightSeverity, usize>,

    /// Most common recommendations
    pub top_recommendations: Vec<(String, usize)>,

    /// Overall insight score (0.0 - 1.0)
    pub overall_score: f64,

    /// Summary of key findings
    pub key_findings: Vec<String>,
}

impl InsightSummary {
    /// Create summary from insight collection
    pub fn from_collection(collection: &InsightCollection) -> Self {
        let total_insights = collection.total_insights();

        let mut insights_by_type = HashMap::new();
        insights_by_type.insert(
            "validation".to_string(),
            collection.validation_insights.len(),
        );
        insights_by_type.insert("quality".to_string(), collection.quality_insights.len());
        insights_by_type.insert(
            "performance".to_string(),
            collection.performance_insights.len(),
        );
        insights_by_type.insert("shape".to_string(), collection.shape_insights.len());
        insights_by_type.insert("data".to_string(), collection.data_insights.len());

        let mut insights_by_severity = HashMap::new();
        for insight in &collection.validation_insights {
            *insights_by_severity
                .entry(insight.severity.clone())
                .or_insert(0) += 1;
        }
        for insight in &collection.quality_insights {
            *insights_by_severity
                .entry(insight.severity.clone())
                .or_insert(0) += 1;
        }
        for insight in &collection.performance_insights {
            *insights_by_severity
                .entry(insight.severity.clone())
                .or_insert(0) += 1;
        }
        for insight in &collection.shape_insights {
            *insights_by_severity
                .entry(insight.severity.clone())
                .or_insert(0) += 1;
        }
        for insight in &collection.data_insights {
            *insights_by_severity
                .entry(insight.severity.clone())
                .or_insert(0) += 1;
        }

        // Calculate overall score based on severity distribution
        let high_priority = collection.high_priority_count();
        let overall_score = if total_insights == 0 {
            1.0
        } else {
            1.0 - (high_priority as f64 / total_insights as f64 * 0.5)
        };

        Self {
            total_insights,
            insights_by_type,
            insights_by_severity,
            top_recommendations: Vec::new(), // Would calculate from collection
            overall_score,
            key_findings: Vec::new(), // Would extract from collection
        }
    }
}

/// Helper traits for insight operations

/// Trait for objects that can provide insights
pub trait InsightProvider {
    /// Generate insights from this object
    fn generate_insights(&self) -> Vec<Box<dyn InsightTrait>>;
}

/// Base trait for all insight types
pub trait InsightTrait {
    /// Get the insight title
    fn title(&self) -> &str;

    /// Get the insight description
    fn description(&self) -> &str;

    /// Get the insight severity
    fn severity(&self) -> &InsightSeverity;

    /// Get the insight confidence
    fn confidence(&self) -> f64;

    /// Get recommendations
    fn recommendations(&self) -> &[String];

    /// Check if this is actionable
    fn is_actionable(&self) -> bool {
        !self.recommendations().is_empty()
    }

    /// Get insight category
    fn category(&self) -> InsightCategory;
}

/// Insight categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsightCategory {
    Validation,
    Quality,
    Performance,
    Shape,
    Data,
    Security,
    Maintenance,
    Optimization,
}

/// Implement InsightTrait for ValidationInsight
impl InsightTrait for ValidationInsight {
    fn title(&self) -> &str {
        &self.title
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn severity(&self) -> &InsightSeverity {
        &self.severity
    }
    fn confidence(&self) -> f64 {
        self.confidence
    }
    fn recommendations(&self) -> &[String] {
        &self.recommendations
    }
    fn category(&self) -> InsightCategory {
        InsightCategory::Validation
    }
}

/// Implement InsightTrait for QualityInsight
impl InsightTrait for QualityInsight {
    fn title(&self) -> &str {
        &self.title
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn severity(&self) -> &InsightSeverity {
        &self.severity
    }
    fn confidence(&self) -> f64 {
        self.confidence
    }
    fn recommendations(&self) -> &[String] {
        &self.recommendations
    }
    fn category(&self) -> InsightCategory {
        InsightCategory::Quality
    }
}

/// Implement InsightTrait for PerformanceInsight
impl InsightTrait for PerformanceInsight {
    fn title(&self) -> &str {
        &self.title
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn severity(&self) -> &InsightSeverity {
        &self.severity
    }
    fn confidence(&self) -> f64 {
        self.confidence
    }
    fn recommendations(&self) -> &[String] {
        &self.recommendations
    }
    fn category(&self) -> InsightCategory {
        InsightCategory::Performance
    }
}

/// Implement InsightTrait for ShapeInsight
impl InsightTrait for ShapeInsight {
    fn title(&self) -> &str {
        &self.title
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn severity(&self) -> &InsightSeverity {
        &self.severity
    }
    fn confidence(&self) -> f64 {
        self.confidence
    }
    fn recommendations(&self) -> &[String] {
        &self.recommendations
    }
    fn category(&self) -> InsightCategory {
        InsightCategory::Shape
    }
}

/// Implement InsightTrait for DataInsight
impl InsightTrait for DataInsight {
    fn title(&self) -> &str {
        &self.title
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn severity(&self) -> &InsightSeverity {
        &self.severity
    }
    fn confidence(&self) -> f64 {
        self.confidence
    }
    fn recommendations(&self) -> &[String] {
        &self.data_improvements
    }
    fn category(&self) -> InsightCategory {
        InsightCategory::Data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_insight_creation() {
        let insight = ValidationInsight {
            insight_type: ValidationInsightType::LowSuccessRate,
            title: "Low Success Rate".to_string(),
            description: "Validation success rate is below threshold".to_string(),
            severity: InsightSeverity::High,
            confidence: 0.9,
            affected_shapes: vec![ShapeId::new("test_shape")],
            recommendations: vec!["Review shape definitions".to_string()],
            supporting_data: HashMap::new(),
        };

        assert!(insight.is_high_priority());
        assert_eq!(
            insight.primary_recommendation(),
            Some("Review shape definitions")
        );
    }

    #[test]
    fn test_quality_insight_status() {
        let insight = QualityInsight {
            insight_type: QualityInsightType::Completeness,
            title: "Data Completeness".to_string(),
            description: "Data completeness analysis".to_string(),
            severity: InsightSeverity::Medium,
            confidence: 0.8,
            quality_dimension: "completeness".to_string(),
            current_score: 0.85,
            trend_direction: TrendDirection::Increasing,
            recommendations: Vec::new(),
            supporting_data: HashMap::new(),
        };

        assert!(insight.is_improving());
        assert!(!insight.is_degrading());
        assert_eq!(insight.quality_status(), QualityStatus::Excellent);
    }

    #[test]
    fn test_performance_insight_status() {
        let insight = PerformanceInsight {
            insight_type: PerformanceInsightType::DegradingPerformance,
            title: "Performance Degradation".to_string(),
            description: "Execution time is increasing".to_string(),
            severity: InsightSeverity::High,
            confidence: 0.8,
            metric_name: "execution_time".to_string(),
            current_value: 10.0,
            trend_direction: TrendDirection::Increasing,
            recommendations: Vec::new(),
            supporting_data: HashMap::new(),
        };

        assert!(insight.is_degrading());
        assert!(!insight.is_improving());
        assert_eq!(insight.performance_status(), PerformanceStatus::Critical);
    }

    #[test]
    fn test_shape_insight_analysis() {
        let complexity_metrics = ShapeComplexityMetrics {
            overall_complexity: ComplexityLevel::Low,
            constraint_count: 5,
            target_count: 2,
            path_complexity: 1,
            nesting_depth: 1,
            cyclic_complexity: 0,
            performance_impact: PerformanceImpact::Low,
        };

        let insight = ShapeInsight {
            insight_type: ShapeInsightType::EffectivenessAnalysis,
            title: "Shape Effectiveness".to_string(),
            description: "Shape design analysis".to_string(),
            severity: InsightSeverity::Low,
            confidence: 0.85,
            shape_id: ShapeId::new("test_shape"),
            effectiveness_score: 0.9,
            complexity_metrics,
            recommendations: Vec::new(),
            supporting_data: HashMap::new(),
        };

        assert!(insight.is_well_designed());
        assert!(!insight.needs_optimization());
    }

    #[test]
    fn test_insight_collection() {
        let mut collection = InsightCollection::new();

        collection.validation_insights.push(ValidationInsight {
            insight_type: ValidationInsightType::LowSuccessRate,
            title: "Test".to_string(),
            description: "Test".to_string(),
            severity: InsightSeverity::High,
            confidence: 0.9,
            affected_shapes: Vec::new(),
            recommendations: vec!["Fix issue".to_string()],
            supporting_data: HashMap::new(),
        });

        collection.quality_insights.push(QualityInsight {
            insight_type: QualityInsightType::Completeness,
            title: "Test".to_string(),
            description: "Test".to_string(),
            severity: InsightSeverity::Medium,
            confidence: 0.8,
            quality_dimension: "test".to_string(),
            current_score: 0.8,
            trend_direction: TrendDirection::Stable,
            recommendations: Vec::new(),
            supporting_data: HashMap::new(),
        });

        assert_eq!(collection.total_insights(), 2);
        assert_eq!(collection.high_priority_count(), 1);

        let filtered = collection.filter_by_severity(InsightSeverity::High);
        assert_eq!(filtered.total_insights(), 1);
    }

    #[test]
    fn test_insight_summary() {
        let collection = InsightCollection::new();
        let summary = InsightSummary::from_collection(&collection);

        assert_eq!(summary.total_insights, 0);
        assert_eq!(summary.overall_score, 1.0);
    }

    #[test]
    fn test_insight_trait() {
        let insight = ValidationInsight {
            insight_type: ValidationInsightType::LowSuccessRate,
            title: "Test Insight".to_string(),
            description: "Test description".to_string(),
            severity: InsightSeverity::High,
            confidence: 0.9,
            affected_shapes: Vec::new(),
            recommendations: vec!["Do something".to_string()],
            supporting_data: HashMap::new(),
        };

        assert_eq!(insight.title(), "Test Insight");
        assert_eq!(insight.description(), "Test description");
        assert_eq!(insight.severity(), &InsightSeverity::High);
        assert_eq!(insight.confidence(), 0.9);
        assert!(insight.is_actionable());
        assert_eq!(insight.category(), InsightCategory::Validation);
    }
}
