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

    /// Filter insights by minimum severity (inclusive)
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

/// Insight generation engine
#[derive(Debug)]
pub struct InsightGenerator {
    config: InsightGenerationConfig,
    validation_analyzer: ValidationInsightAnalyzer,
    quality_analyzer: QualityInsightAnalyzer,
    performance_analyzer: PerformanceInsightAnalyzer,
    shape_analyzer: ShapeInsightAnalyzer,
    data_analyzer: DataInsightAnalyzer,
}

/// Configuration for insight generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightGenerationConfig {
    /// Minimum confidence threshold for insights
    pub min_confidence_threshold: f64,
    
    /// Enable validation insights
    pub enable_validation_insights: bool,
    
    /// Enable quality insights
    pub enable_quality_insights: bool,
    
    /// Enable performance insights
    pub enable_performance_insights: bool,
    
    /// Enable shape insights
    pub enable_shape_insights: bool,
    
    /// Enable data insights
    pub enable_data_insights: bool,
    
    /// Maximum insights per category
    pub max_insights_per_category: usize,
    
    /// Time window for trend analysis (in seconds)
    pub trend_analysis_window: u64,
}

impl Default for InsightGenerationConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.7,
            enable_validation_insights: true,
            enable_quality_insights: true,
            enable_performance_insights: true,
            enable_shape_insights: true,
            enable_data_insights: true,
            max_insights_per_category: 50,
            trend_analysis_window: 3600, // 1 hour
        }
    }
}

impl InsightGenerator {
    /// Create a new insight generator
    pub fn new(config: InsightGenerationConfig) -> Self {
        Self {
            config: config.clone(),
            validation_analyzer: ValidationInsightAnalyzer::new(config.clone()),
            quality_analyzer: QualityInsightAnalyzer::new(config.clone()),
            performance_analyzer: PerformanceInsightAnalyzer::new(config.clone()),
            shape_analyzer: ShapeInsightAnalyzer::new(config.clone()),
            data_analyzer: DataInsightAnalyzer::new(config),
        }
    }

    /// Generate comprehensive insights from all available data
    pub fn generate_insights(
        &mut self,
        validation_data: &crate::ValidationData,
        quality_data: &crate::quality::QualityAssessmentData,
        performance_data: &crate::PerformanceData,
        shape_data: &crate::ShapeData,
        rdf_data: &crate::RdfData,
    ) -> crate::Result<InsightCollection> {
        let start_time = std::time::Instant::now();
        let mut collection = InsightCollection::new();

        tracing::info!("Starting comprehensive insight generation");

        // Generate validation insights
        if self.config.enable_validation_insights {
            match self.validation_analyzer.analyze(validation_data) {
                Ok(mut insights) => {
                    insights.truncate(self.config.max_insights_per_category);
                    collection.validation_insights = insights;
                    tracing::debug!("Generated {} validation insights", collection.validation_insights.len());
                }
                Err(e) => tracing::warn!("Failed to generate validation insights: {}", e),
            }
        }

        // Generate quality insights
        if self.config.enable_quality_insights {
            match self.quality_analyzer.analyze(quality_data) {
                Ok(mut insights) => {
                    insights.truncate(self.config.max_insights_per_category);
                    collection.quality_insights = insights;
                    tracing::debug!("Generated {} quality insights", collection.quality_insights.len());
                }
                Err(e) => tracing::warn!("Failed to generate quality insights: {}", e),
            }
        }

        // Generate performance insights
        if self.config.enable_performance_insights {
            match self.performance_analyzer.analyze(performance_data) {
                Ok(mut insights) => {
                    insights.truncate(self.config.max_insights_per_category);
                    collection.performance_insights = insights;
                    tracing::debug!("Generated {} performance insights", collection.performance_insights.len());
                }
                Err(e) => tracing::warn!("Failed to generate performance insights: {}", e),
            }
        }

        // Generate shape insights
        if self.config.enable_shape_insights {
            match self.shape_analyzer.analyze(shape_data) {
                Ok(mut insights) => {
                    insights.truncate(self.config.max_insights_per_category);
                    collection.shape_insights = insights;
                    tracing::debug!("Generated {} shape insights", collection.shape_insights.len());
                }
                Err(e) => tracing::warn!("Failed to generate shape insights: {}", e),
            }
        }

        // Generate data insights
        if self.config.enable_data_insights {
            match self.data_analyzer.analyze(rdf_data) {
                Ok(mut insights) => {
                    insights.truncate(self.config.max_insights_per_category);
                    collection.data_insights = insights;
                    tracing::debug!("Generated {} data insights", collection.data_insights.len());
                }
                Err(e) => tracing::warn!("Failed to generate data insights: {}", e),
            }
        }

        // Update metadata
        collection.metadata.generation_duration = start_time.elapsed();
        collection.metadata.engine_version = env!("CARGO_PKG_VERSION").to_string();
        collection.metadata.confidence_threshold = self.config.min_confidence_threshold;

        tracing::info!(
            "Generated {} total insights in {:?}",
            collection.total_insights(),
            collection.metadata.generation_duration
        );

        Ok(collection)
    }

    /// Generate insights from validation results only
    pub fn generate_validation_insights(
        &mut self,
        validation_data: &crate::ValidationData,
    ) -> crate::Result<Vec<ValidationInsight>> {
        self.validation_analyzer.analyze(validation_data)
    }

    /// Generate insights from quality assessment only
    pub fn generate_quality_insights(
        &mut self,
        quality_data: &crate::quality::QualityAssessmentData,
    ) -> crate::Result<Vec<QualityInsight>> {
        self.quality_analyzer.analyze(quality_data)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: InsightGenerationConfig) {
        self.config = config.clone();
        self.validation_analyzer.update_config(config.clone());
        self.quality_analyzer.update_config(config.clone());
        self.performance_analyzer.update_config(config.clone());
        self.shape_analyzer.update_config(config.clone());
        self.data_analyzer.update_config(config);
    }
}

/// Validation insight analyzer
#[derive(Debug)]
struct ValidationInsightAnalyzer {
    config: InsightGenerationConfig,
}

impl ValidationInsightAnalyzer {
    fn new(config: InsightGenerationConfig) -> Self {
        Self { config }
    }

    fn update_config(&mut self, config: InsightGenerationConfig) {
        self.config = config;
    }

    fn analyze(&self, data: &crate::ValidationData) -> crate::Result<Vec<ValidationInsight>> {
        let mut insights = Vec::new();

        // Analyze success rates
        if let Some(insight) = self.analyze_success_rates(data)? {
            insights.push(insight);
        }

        // Analyze violation patterns
        insights.extend(self.analyze_violation_patterns(data)?);

        // Analyze performance trends
        if let Some(insight) = self.analyze_performance_trends(data)? {
            insights.push(insight);
        }

        // Filter by confidence
        insights.retain(|i| i.confidence >= self.config.min_confidence_threshold);

        Ok(insights)
    }

    fn analyze_success_rates(&self, data: &crate::ValidationData) -> crate::Result<Option<ValidationInsight>> {
        let success_rate = data.calculate_success_rate();
        
        if success_rate < 0.7 {
            let severity = if success_rate < 0.3 {
                InsightSeverity::Critical
            } else if success_rate < 0.5 {
                InsightSeverity::High
            } else {
                InsightSeverity::Medium
            };

            let insight = ValidationInsight {
                insight_type: ValidationInsightType::LowSuccessRate,
                title: "Low Validation Success Rate Detected".to_string(),
                description: format!(
                    "Validation success rate is {:.1}%, which is below the recommended threshold of 70%.",
                    success_rate * 100.0
                ),
                severity,
                confidence: 0.95,
                affected_shapes: data.get_failing_shapes(),
                recommendations: vec![
                    "Review failing shape constraints for overly restrictive rules".to_string(),
                    "Analyze common violation patterns to identify data quality issues".to_string(),
                    "Consider refining target selectors to improve precision".to_string(),
                ],
                supporting_data: {
                    let mut data_map = HashMap::new();
                    data_map.insert("success_rate".to_string(), format!("{:.3}", success_rate));
                    data_map.insert("total_validations".to_string(), data.total_validations().to_string());
                    data_map.insert("failed_validations".to_string(), data.failed_validations().to_string());
                    data_map
                },
            };

            Ok(Some(insight))
        } else {
            Ok(None)
        }
    }

    fn analyze_violation_patterns(&self, data: &crate::ValidationData) -> crate::Result<Vec<ValidationInsight>> {
        let mut insights = Vec::new();
        let patterns = data.extract_violation_patterns();

        for pattern in patterns {
            if pattern.frequency >= 0.1 && pattern.confidence >= self.config.min_confidence_threshold {
                let insight = ValidationInsight {
                    insight_type: ValidationInsightType::ViolationPattern,
                    title: format!("Recurring Violation Pattern: {}", pattern.pattern_type),
                    description: format!(
                        "Pattern '{}' occurs in {:.1}% of validation failures, indicating a systematic issue.",
                        pattern.description, pattern.frequency * 100.0
                    ),
                    severity: if pattern.frequency > 0.5 {
                        InsightSeverity::High
                    } else if pattern.frequency > 0.2 {
                        InsightSeverity::Medium
                    } else {
                        InsightSeverity::Low
                    },
                    confidence: pattern.confidence,
                    affected_shapes: pattern.affected_shapes,
                    recommendations: pattern.recommendations,
                    supporting_data: pattern.evidence,
                };
                insights.push(insight);
            }
        }

        Ok(insights)
    }

    fn analyze_performance_trends(&self, data: &crate::ValidationData) -> crate::Result<Option<ValidationInsight>> {
        let trend = data.calculate_performance_trend();
        
        if trend.is_degrading() && trend.significance > 0.8 {
            let insight = ValidationInsight {
                insight_type: ValidationInsightType::PerformanceDegradation,
                title: "Validation Performance Degradation".to_string(),
                description: format!(
                    "Validation performance has degraded by {:.1}% over the last {} validations.",
                    trend.degradation_percentage, trend.sample_size
                ),
                severity: InsightSeverity::Medium,
                confidence: trend.significance,
                affected_shapes: Vec::new(),
                recommendations: vec![
                    "Review recent changes to shape definitions".to_string(),
                    "Analyze query complexity and optimization opportunities".to_string(),
                    "Consider implementing validation result caching".to_string(),
                ],
                supporting_data: trend.to_map(),
            };

            Ok(Some(insight))
        } else {
            Ok(None)
        }
    }
}

/// Quality insight analyzer
#[derive(Debug)]
struct QualityInsightAnalyzer {
    config: InsightGenerationConfig,
}

impl QualityInsightAnalyzer {
    fn new(config: InsightGenerationConfig) -> Self {
        Self { config }
    }

    fn update_config(&mut self, config: InsightGenerationConfig) {
        self.config = config;
    }

    fn analyze(&self, data: &crate::quality::QualityAssessmentData) -> crate::Result<Vec<QualityInsight>> {
        let mut insights = Vec::new();

        // Analyze each quality dimension
        for dimension in &data.quality_dimensions {
            if let Some(insight) = self.analyze_quality_dimension(dimension)? {
                insights.push(insight);
            }
        }

        // Analyze overall trends
        insights.extend(self.analyze_quality_trends(data)?);

        // Filter by confidence
        insights.retain(|i| i.confidence >= self.config.min_confidence_threshold);

        Ok(insights)
    }

    fn analyze_quality_dimension(&self, dimension: &crate::quality::QualityDimension) -> crate::Result<Option<QualityInsight>> {
        if dimension.score < 0.6 {
            let insight_type = match dimension.dimension_type.as_str() {
                "completeness" => QualityInsightType::Completeness,
                "consistency" => QualityInsightType::Consistency,
                "accuracy" => QualityInsightType::Accuracy,
                "validity" => QualityInsightType::Validity,
                _ => QualityInsightType::Custom(dimension.dimension_type.clone()),
            };

            let severity = if dimension.score < 0.3 {
                InsightSeverity::Critical
            } else if dimension.score < 0.45 {
                InsightSeverity::High
            } else {
                InsightSeverity::Medium
            };

            let insight = QualityInsight {
                insight_type,
                title: format!("{} Quality Issue", dimension.dimension_type.to_uppercase()),
                description: format!(
                    "{} quality score is {:.1}%, indicating significant quality issues in this dimension.",
                    dimension.dimension_type, dimension.score * 100.0
                ),
                severity,
                confidence: dimension.confidence,
                quality_dimension: dimension.dimension_type.clone(),
                current_score: dimension.score,
                trend_direction: dimension.trend_direction.clone(),
                recommendations: dimension.improvement_recommendations.clone(),
                supporting_data: dimension.evidence.clone(),
            };

            Ok(Some(insight))
        } else {
            Ok(None)
        }
    }

    fn analyze_quality_trends(&self, data: &crate::quality::QualityAssessmentData) -> crate::Result<Vec<QualityInsight>> {
        let mut insights = Vec::new();

        // Analyze overall quality trend
        let overall_trend = data.calculate_overall_trend();
        if overall_trend.is_significant_decline() {
            let insight = QualityInsight {
                insight_type: QualityInsightType::TrendAnalysis,
                title: "Overall Data Quality Declining".to_string(),
                description: format!(
                    "Overall data quality has declined by {:.1}% over the analysis period.",
                    overall_trend.decline_percentage
                ),
                severity: InsightSeverity::High,
                confidence: overall_trend.confidence,
                quality_dimension: "overall".to_string(),
                current_score: data.overall_score,
                trend_direction: TrendDirection::Decreasing,
                recommendations: vec![
                    "Implement data quality monitoring alerts".to_string(),
                    "Review data ingestion processes for quality controls".to_string(),
                    "Establish data quality improvement initiatives".to_string(),
                ],
                supporting_data: overall_trend.to_map(),
            };
            insights.push(insight);
        }

        Ok(insights)
    }
}

/// Performance insight analyzer
#[derive(Debug)]
struct PerformanceInsightAnalyzer {
    config: InsightGenerationConfig,
}

impl PerformanceInsightAnalyzer {
    fn new(config: InsightGenerationConfig) -> Self {
        Self { config }
    }

    fn update_config(&mut self, config: InsightGenerationConfig) {
        self.config = config;
    }

    fn analyze(&self, data: &crate::PerformanceData) -> crate::Result<Vec<PerformanceInsight>> {
        let mut insights = Vec::new();

        // Analyze execution time trends
        if let Some(insight) = self.analyze_execution_times(data)? {
            insights.push(insight);
        }

        // Analyze memory usage
        if let Some(insight) = self.analyze_memory_usage(data)? {
            insights.push(insight);
        }

        // Analyze throughput
        if let Some(insight) = self.analyze_throughput(data)? {
            insights.push(insight);
        }

        // Filter by confidence
        insights.retain(|i| i.confidence >= self.config.min_confidence_threshold);

        Ok(insights)
    }

    fn analyze_execution_times(&self, data: &crate::PerformanceData) -> crate::Result<Option<PerformanceInsight>> {
        let execution_trend = data.calculate_execution_time_trend();
        
        if execution_trend.is_increasing() && execution_trend.significance > 0.75 {
            let insight = PerformanceInsight {
                insight_type: PerformanceInsightType::DegradingPerformance,
                title: "Execution Time Increasing".to_string(),
                description: format!(
                    "Average execution time has increased by {:.1}% over recent validations.",
                    execution_trend.increase_percentage
                ),
                severity: if execution_trend.increase_percentage > 50.0 {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: execution_trend.significance,
                metric_name: "execution_time".to_string(),
                current_value: data.current_avg_execution_time,
                trend_direction: TrendDirection::Increasing,
                recommendations: vec![
                    "Profile validation queries for optimization opportunities".to_string(),
                    "Consider implementing query result caching".to_string(),
                    "Review shape complexity and constraint ordering".to_string(),
                ],
                supporting_data: execution_trend.to_map(),
            };

            Ok(Some(insight))
        } else {
            Ok(None)
        }
    }

    fn analyze_memory_usage(&self, data: &crate::PerformanceData) -> crate::Result<Option<PerformanceInsight>> {
        if data.peak_memory_usage > data.memory_threshold {
            let insight = PerformanceInsight {
                insight_type: PerformanceInsightType::MemoryIssue,
                title: "High Memory Usage Detected".to_string(),
                description: format!(
                    "Peak memory usage ({:.1} MB) exceeds threshold ({:.1} MB).",
                    data.peak_memory_usage / 1024.0 / 1024.0,
                    data.memory_threshold / 1024.0 / 1024.0
                ),
                severity: InsightSeverity::Medium,
                confidence: 0.9,
                metric_name: "memory_usage".to_string(),
                current_value: data.peak_memory_usage,
                trend_direction: TrendDirection::Stable,
                recommendations: vec![
                    "Implement memory-efficient validation strategies".to_string(),
                    "Consider processing data in smaller batches".to_string(),
                    "Review memory usage patterns for optimization".to_string(),
                ],
                supporting_data: {
                    let mut map = HashMap::new();
                    map.insert("peak_memory".to_string(), data.peak_memory_usage.to_string());
                    map.insert("threshold".to_string(), data.memory_threshold.to_string());
                    map
                },
            };

            Ok(Some(insight))
        } else {
            Ok(None)
        }
    }

    fn analyze_throughput(&self, data: &crate::PerformanceData) -> crate::Result<Option<PerformanceInsight>> {
        let throughput_trend = data.calculate_throughput_trend();
        
        if throughput_trend.is_declining() && throughput_trend.significance > 0.8 {
            let insight = PerformanceInsight {
                insight_type: PerformanceInsightType::ThroughputIssue,
                title: "Validation Throughput Declining".to_string(),
                description: format!(
                    "Validation throughput has decreased by {:.1}% over recent validations.",
                    throughput_trend.decline_percentage
                ),
                severity: InsightSeverity::Medium,
                confidence: throughput_trend.significance,
                metric_name: "throughput".to_string(),
                current_value: data.current_throughput,
                trend_direction: TrendDirection::Decreasing,
                recommendations: vec![
                    "Investigate performance bottlenecks".to_string(),
                    "Consider parallel validation processing".to_string(),
                    "Optimize shape definitions for better performance".to_string(),
                ],
                supporting_data: throughput_trend.to_map(),
            };

            Ok(Some(insight))
        } else {
            Ok(None)
        }
    }
}

/// Shape insight analyzer
#[derive(Debug)]
struct ShapeInsightAnalyzer {
    config: InsightGenerationConfig,
}

impl ShapeInsightAnalyzer {
    fn new(config: InsightGenerationConfig) -> Self {
        Self { config }
    }

    fn update_config(&mut self, config: InsightGenerationConfig) {
        self.config = config;
    }

    fn analyze(&self, data: &crate::ShapeData) -> crate::Result<Vec<ShapeInsight>> {
        let mut insights = Vec::new();

        for shape_analysis in &data.shape_analyses {
            if let Some(insight) = self.analyze_shape_complexity(shape_analysis)? {
                insights.push(insight);
            }

            if let Some(insight) = self.analyze_shape_effectiveness(shape_analysis)? {
                insights.push(insight);
            }
        }

        // Filter by confidence
        insights.retain(|i| i.confidence >= self.config.min_confidence_threshold);

        Ok(insights)
    }

    fn analyze_shape_complexity(&self, analysis: &crate::ShapeAnalysis) -> crate::Result<Option<ShapeInsight>> {
        if analysis.complexity_metrics.overall_complexity >= ComplexityLevel::High {
            let insight = ShapeInsight {
                insight_type: ShapeInsightType::OverlyComplex,
                title: format!("Shape {} is Overly Complex", analysis.shape_id.as_str()),
                description: format!(
                    "Shape has high complexity with {} constraints and nesting depth of {}.",
                    analysis.complexity_metrics.constraint_count,
                    analysis.complexity_metrics.nesting_depth
                ),
                severity: InsightSeverity::Medium,
                confidence: 0.85,
                shape_id: analysis.shape_id.clone(),
                effectiveness_score: analysis.effectiveness_score,
                complexity_metrics: analysis.complexity_metrics.clone(),
                recommendations: vec![
                    "Consider breaking down complex constraints into simpler ones".to_string(),
                    "Review necessity of all constraints".to_string(),
                    "Optimize constraint ordering for better performance".to_string(),
                ],
                supporting_data: {
                    let mut map = HashMap::new();
                    map.insert("constraint_count".to_string(), analysis.complexity_metrics.constraint_count.to_string());
                    map.insert("nesting_depth".to_string(), analysis.complexity_metrics.nesting_depth.to_string());
                    map
                },
            };

            Ok(Some(insight))
        } else {
            Ok(None)
        }
    }

    fn analyze_shape_effectiveness(&self, analysis: &crate::ShapeAnalysis) -> crate::Result<Option<ShapeInsight>> {
        if analysis.effectiveness_score < 0.6 {
            let insight = ShapeInsight {
                insight_type: ShapeInsightType::EffectivenessAnalysis,
                title: format!("Shape {} Has Low Effectiveness", analysis.shape_id.as_str()),
                description: format!(
                    "Shape effectiveness score is {:.1}%, indicating potential design issues.",
                    analysis.effectiveness_score * 100.0
                ),
                severity: InsightSeverity::Medium,
                confidence: 0.8,
                shape_id: analysis.shape_id.clone(),
                effectiveness_score: analysis.effectiveness_score,
                complexity_metrics: analysis.complexity_metrics.clone(),
                recommendations: vec![
                    "Review shape targets for precision".to_string(),
                    "Analyze constraint relevance and accuracy".to_string(),
                    "Consider user feedback on shape usefulness".to_string(),
                ],
                supporting_data: {
                    let mut map = HashMap::new();
                    map.insert("effectiveness_score".to_string(), analysis.effectiveness_score.to_string());
                    map
                },
            };

            Ok(Some(insight))
        } else {
            Ok(None)
        }
    }
}

/// Data insight analyzer
#[derive(Debug)]
struct DataInsightAnalyzer {
    config: InsightGenerationConfig,
}

impl DataInsightAnalyzer {
    fn new(config: InsightGenerationConfig) -> Self {
        Self { config }
    }

    fn update_config(&mut self, config: InsightGenerationConfig) {
        self.config = config;
    }

    fn analyze(&self, data: &crate::RdfData) -> crate::Result<Vec<DataInsight>> {
        let mut insights = Vec::new();

        // Analyze missing data patterns
        if let Some(insight) = self.analyze_missing_data(data)? {
            insights.push(insight);
        }

        // Analyze data inconsistencies
        insights.extend(self.analyze_data_inconsistencies(data)?);

        // Analyze data distribution
        if let Some(insight) = self.analyze_data_distribution(data)? {
            insights.push(insight);
        }

        // Filter by confidence
        insights.retain(|i| i.confidence >= self.config.min_confidence_threshold);

        Ok(insights)
    }

    fn analyze_missing_data(&self, data: &crate::RdfData) -> crate::Result<Option<DataInsight>> {
        let missing_data_percentage = data.calculate_missing_data_percentage();
        
        if missing_data_percentage > 0.15 {
            let insight = DataInsight {
                insight_type: DataInsightType::MissingData,
                title: "Significant Missing Data Detected".to_string(),
                description: format!(
                    "{:.1}% of expected data properties are missing across the dataset.",
                    missing_data_percentage * 100.0
                ),
                severity: if missing_data_percentage > 0.3 {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: 0.9,
                affected_elements: data.get_missing_data_elements(),
                quality_impact: if missing_data_percentage > 0.3 {
                    QualityImpact::High
                } else {
                    QualityImpact::Medium
                },
                data_improvements: vec![
                    "Implement data completeness validation".to_string(),
                    "Review data collection processes".to_string(),
                    "Add default values where appropriate".to_string(),
                ],
                statistics: {
                    let mut map = HashMap::new();
                    map.insert("missing_percentage".to_string(), missing_data_percentage);
                    map.insert("total_elements".to_string(), data.total_elements() as f64);
                    map
                },
            };

            Ok(Some(insight))
        } else {
            Ok(None)
        }
    }

    fn analyze_data_inconsistencies(&self, data: &crate::RdfData) -> crate::Result<Vec<DataInsight>> {
        let mut insights = Vec::new();
        let inconsistencies = data.detect_inconsistencies();

        for inconsistency in inconsistencies {
            if inconsistency.significance > self.config.min_confidence_threshold {
                let insight = DataInsight {
                    insight_type: DataInsightType::InconsistentPatterns,
                    title: format!("Data Inconsistency: {}", inconsistency.pattern_type),
                    description: inconsistency.description,
                    severity: match inconsistency.impact_level {
                        crate::InconsistencyImpact::High => InsightSeverity::High,
                        crate::InconsistencyImpact::Medium => InsightSeverity::Medium,
                        crate::InconsistencyImpact::Low => InsightSeverity::Low,
                    },
                    confidence: inconsistency.significance,
                    affected_elements: inconsistency.affected_elements,
                    quality_impact: match inconsistency.impact_level {
                        crate::InconsistencyImpact::High => QualityImpact::High,
                        crate::InconsistencyImpact::Medium => QualityImpact::Medium,
                        crate::InconsistencyImpact::Low => QualityImpact::Low,
                    },
                    data_improvements: inconsistency.suggested_fixes,
                    statistics: inconsistency.evidence,
                };
                insights.push(insight);
            }
        }

        Ok(insights)
    }

    fn analyze_data_distribution(&self, data: &crate::RdfData) -> crate::Result<Option<DataInsight>> {
        let distribution_analysis = data.analyze_distribution();
        
        if distribution_analysis.has_significant_anomalies() {
            let insight = DataInsight {
                insight_type: DataInsightType::DistributionAnomalies,
                title: "Data Distribution Anomalies Detected".to_string(),
                description: "Unusual patterns in data distribution detected that may indicate quality issues.".to_string(),
                severity: InsightSeverity::Medium,
                confidence: distribution_analysis.confidence,
                affected_elements: distribution_analysis.anomalous_elements,
                quality_impact: QualityImpact::Medium,
                data_improvements: vec![
                    "Investigate outlier values".to_string(),
                    "Review data collection and processing procedures".to_string(),
                    "Consider data normalization techniques".to_string(),
                ],
                statistics: distribution_analysis.statistics,
            };

            Ok(Some(insight))
        } else {
            Ok(None)
        }
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
        assert_eq!(filtered.total_insights(), 1); // Should only include High severity insights
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
