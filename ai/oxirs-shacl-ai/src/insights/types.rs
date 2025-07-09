//! Type definitions for AI-powered insights

use crate::analytics::{InsightSeverity, TrendDirection};
use oxirs_shacl::ShapeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    /// Ineffective constraints
    IneffectiveConstraints,
    /// Missing constraints
    MissingConstraints,
    /// Redundant constraints
    RedundantConstraints,
    /// Poor target selection
    PoorTargetSelection,
    /// Shape reusability opportunity
    ReusabilityOpportunity,
    /// Shape composition issue
    CompositionIssue,
    /// Custom shape insight
    Custom(String),
}

/// Data insight - insights about data patterns and characteristics
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

    /// Data characteristics identified
    pub data_characteristics: Vec<String>,

    /// Data quality recommendations
    pub data_improvements: Vec<String>,

    /// Supporting data and evidence
    pub supporting_data: HashMap<String, String>,
}

/// Types of data insights
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataInsightType {
    /// Data distribution analysis
    Distribution,
    /// Data completeness analysis
    Completeness,
    /// Data consistency analysis
    Consistency,
    /// Data freshness analysis
    Freshness,
    /// Data volume analysis
    Volume,
    /// Data velocity analysis
    Velocity,
    /// Data variety analysis
    Variety,
    /// Data veracity analysis
    Veracity,
    /// Custom data insight
    Custom(String),
}

/// Complexity level enumeration
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Shape complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeComplexityMetrics {
    /// Overall complexity level
    pub overall_complexity: ComplexityLevel,

    /// Number of constraints
    pub constraint_count: usize,

    /// Maximum constraint depth
    pub max_constraint_depth: usize,

    /// Number of targets
    pub target_count: usize,

    /// Path complexity score
    pub path_complexity: f64,

    /// Cyclic dependencies detected
    pub has_cycles: bool,
}

/// Insight category enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsightCategory {
    Validation,
    Quality,
    Performance,
    Shape,
    Data,
}

/// Common trait for all insight types
pub trait InsightTrait {
    /// Get the insight title
    fn title(&self) -> &str;

    /// Get the insight description
    fn description(&self) -> &str;

    /// Get the insight severity
    fn severity(&self) -> &InsightSeverity;

    /// Get the confidence score
    fn confidence(&self) -> f64;

    /// Get recommendations
    fn recommendations(&self) -> &[String];

    /// Get insight category
    fn category(&self) -> InsightCategory;
}
