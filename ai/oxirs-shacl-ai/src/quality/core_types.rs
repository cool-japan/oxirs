//! Quality Assessment Core — Types
//!
//! Structs, enums, quality dimension types, and scoring types for the
//! quality assessment subsystem.

use oxirs_core::Term;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Training data for quality assessment models
#[derive(Debug, Clone)]
pub struct QualityTrainingData {
    pub quality_examples: Vec<QualityExample>,
    pub metadata: QualityTrainingMetadata,
}

/// Individual quality assessment example
#[derive(Debug, Clone)]
pub struct QualityExample {
    pub graph_features: Vec<f64>,
    pub quality_metrics: QualityScores,
    pub quality_score: f64,
}

/// Training metadata for quality models
#[derive(Debug, Clone)]
pub struct QualityTrainingMetadata {
    pub dataset_name: String,
    pub collection_date: chrono::DateTime<chrono::Utc>,
    pub total_examples: usize,
}

/// Quality assessment data for insight analysis
#[derive(Debug, Clone)]
pub struct QualityAssessmentData {
    pub quality_dimensions: Vec<QualityDimension>,
    pub overall_score: f64,
    pub assessment_timestamp: chrono::DateTime<chrono::Utc>,
}

impl QualityAssessmentData {
    pub fn calculate_overall_trend(&self) -> QualityTrend {
        QualityTrend {
            decline_percentage: 10.0,
            confidence: 0.8,
        }
    }
}

/// Quality dimension for analysis
#[derive(Debug, Clone)]
pub struct QualityDimension {
    pub dimension_type: String,
    pub score: f64,
    pub confidence: f64,
    pub trend_direction: crate::analytics::TrendDirection,
    pub improvement_recommendations: Vec<String>,
    pub evidence: HashMap<String, String>,
}

/// Quality trend analysis
#[derive(Debug, Clone)]
pub struct QualityTrend {
    pub decline_percentage: f64,
    pub confidence: f64,
}

impl QualityTrend {
    pub fn is_significant_decline(&self) -> bool {
        self.decline_percentage > 5.0 && self.confidence > 0.7
    }

    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "decline_percentage".to_string(),
            self.decline_percentage.to_string(),
        );
        map.insert("confidence".to_string(), self.confidence.to_string());
        map
    }
}

/// Configuration for quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Enable automatic quality assessment
    pub enable_assessment: bool,

    /// Quality scoring thresholds
    pub quality_thresholds: QualityThresholds,

    /// Assessment algorithms to use
    pub algorithms: QualityAlgorithms,

    /// Enable quality reporting
    pub enable_reporting: bool,

    /// Enable training on quality data
    pub enable_training: bool,

    /// Maximum number of issues to report per category
    pub max_issues_per_category: usize,

    /// Minimum confidence for quality recommendations
    pub min_recommendation_confidence: f64,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            enable_assessment: true,
            quality_thresholds: QualityThresholds::default(),
            algorithms: QualityAlgorithms::default(),
            enable_reporting: true,
            enable_training: true,
            max_issues_per_category: 50,
            min_recommendation_confidence: 0.7,
        }
    }
}

/// Quality scoring thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum completeness score (0.0 - 1.0)
    pub min_completeness: f64,

    /// Minimum consistency score (0.0 - 1.0)
    pub min_consistency: f64,

    /// Minimum accuracy score (0.0 - 1.0)
    pub min_accuracy: f64,

    /// Minimum conformance score (0.0 - 1.0)
    pub min_conformance: f64,

    /// Maximum duplicate ratio (0.0 - 1.0)
    pub max_duplicate_ratio: f64,

    /// Minimum schema adherence (0.0 - 1.0)
    pub min_schema_adherence: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_completeness: 0.8,
            min_consistency: 0.9,
            min_accuracy: 0.85,
            min_conformance: 0.95,
            max_duplicate_ratio: 0.05,
            min_schema_adherence: 0.9,
        }
    }
}

/// Quality assessment algorithms configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAlgorithms {
    /// Enable completeness assessment
    pub enable_completeness: bool,

    /// Enable consistency checking
    pub enable_consistency: bool,

    /// Enable accuracy analysis
    pub enable_accuracy: bool,

    /// Enable duplicate detection
    pub enable_duplicate_detection: bool,

    /// Enable schema adherence checking
    pub enable_schema_adherence: bool,

    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,

    /// Enable pattern analysis
    pub enable_pattern_analysis: bool,
}

impl Default for QualityAlgorithms {
    fn default() -> Self {
        Self {
            enable_completeness: true,
            enable_consistency: true,
            enable_accuracy: true,
            enable_duplicate_detection: true,
            enable_schema_adherence: true,
            enable_anomaly_detection: true,
            enable_pattern_analysis: true,
        }
    }
}

/// Quality assessment report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub overall_score: f64,
    pub completeness_score: f64,
    pub consistency_score: f64,
    pub accuracy_score: f64,
    pub conformance_score: f64,
    pub duplicate_ratio: f64,
    pub schema_adherence_score: f64,
    pub issues: Vec<QualityIssue>,
    pub recommendations: Vec<QualityRecommendation>,
    pub ai_insights: Option<Vec<crate::insights::QualityInsight>>,
    pub assessment_timestamp: chrono::DateTime<chrono::Utc>,
}

impl QualityReport {
    pub fn new() -> Self {
        Self {
            overall_score: 0.0,
            completeness_score: 0.0,
            consistency_score: 0.0,
            accuracy_score: 0.0,
            conformance_score: 0.0,
            duplicate_ratio: 0.0,
            schema_adherence_score: 0.0,
            issues: Vec::new(),
            recommendations: Vec::new(),
            ai_insights: None,
            assessment_timestamp: chrono::Utc::now(),
        }
    }

    pub fn set_overall_score(&mut self, score: f64) {
        self.overall_score = score;
    }
    pub fn set_completeness_score(&mut self, score: f64) {
        self.completeness_score = score;
    }
    pub fn set_consistency_score(&mut self, score: f64) {
        self.consistency_score = score;
    }
    pub fn set_accuracy_score(&mut self, score: f64) {
        self.accuracy_score = score;
    }
    pub fn set_conformance_score(&mut self, score: f64) {
        self.conformance_score = score;
    }
    pub fn set_duplicate_ratio(&mut self, ratio: f64) {
        self.duplicate_ratio = ratio;
    }
    pub fn set_schema_adherence_score(&mut self, score: f64) {
        self.schema_adherence_score = score;
    }
    pub fn set_recommendations(&mut self, recommendations: Vec<QualityRecommendation>) {
        self.recommendations = recommendations;
    }

    pub fn add_validation_issues(&mut self, issues: Vec<QualityIssue>) {
        self.issues.extend(issues);
    }

    pub fn add_duplicate_issues(&mut self, issues: Vec<QualityIssue>) {
        self.issues.extend(issues);
    }

    pub fn add_anomaly_issues(&mut self, issues: Vec<QualityIssue>) {
        self.issues.extend(issues);
    }
}

impl Default for QualityReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality issue found during assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub category: QualityIssueCategory,
    pub severity: QualityIssueSeverity,
    pub description: String,
    pub affected_nodes: Vec<Term>,
    pub recommendation: String,
    pub confidence: f64,
}

/// Categories of quality issues
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityIssueCategory {
    Completeness,
    Consistency,
    Accuracy,
    ShapeViolation,
    Duplicate,
    SchemaAdherence,
    Anomaly,
}

/// Severity levels for quality issues
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityIssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

impl QualityIssueSeverity {
    pub fn from_shacl_severity(severity: &oxirs_shacl::Severity) -> Self {
        match severity {
            oxirs_shacl::Severity::Violation => Self::High,
            oxirs_shacl::Severity::Warning => Self::Medium,
            oxirs_shacl::Severity::Info => Self::Info,
        }
    }
}

/// Quality improvement recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRecommendation {
    pub category: QualityRecommendationCategory,
    pub priority: RecommendationPriority,
    pub description: String,
    pub estimated_impact: f64,
    pub estimated_effort: ImplementationEffort,
    pub confidence: f64,
}

/// Categories of quality recommendations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityRecommendationCategory {
    Completeness,
    Consistency,
    Accuracy,
    Deduplication,
    SchemaAdherence,
    Performance,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Implementation effort estimates
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Quality assessment statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityStatistics {
    pub total_assessments: usize,
    pub total_assessment_time: std::time::Duration,
    pub last_overall_score: f64,
    pub model_trained: bool,
    pub average_completeness: f64,
    pub average_consistency: f64,
    pub average_accuracy: f64,
}

/// Quality score breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScores {
    pub completeness: f64,
    pub consistency: f64,
    pub accuracy: f64,
    pub conformance: f64,
    pub overall: f64,
}

/// Quality assessment weights for overall score calculation (internal)
#[derive(Debug, Clone)]
pub(crate) struct QualityWeights {
    pub completeness: f64,
    pub consistency: f64,
    pub accuracy: f64,
    pub conformance: f64,
    pub schema_adherence: f64,
}

/// Internal helper: conformance computation result
#[derive(Debug)]
pub(crate) struct ConformanceResult {
    pub score: f64,
    pub issues: Vec<QualityIssue>,
}

/// Internal helper: duplicate detection result
#[derive(Debug)]
pub(crate) struct DuplicateResult {
    pub ratio: f64,
    pub issues: Vec<QualityIssue>,
}

/// Internal helper: a group of duplicate entities
#[derive(Debug)]
pub(crate) struct DuplicateGroup {
    pub signature: String,
    pub entities: Vec<Term>,
    pub confidence: f64,
}

/// Internal helper: consistency counts
#[derive(Debug)]
pub(crate) struct ConsistencyCheck {
    pub total: usize,
    pub consistent: usize,
}

/// Internal helper: accuracy counts
#[derive(Debug)]
pub(crate) struct AccuracyCheck {
    pub total: usize,
    pub accurate: usize,
}

/// Internal helper: schema adherence counts
#[derive(Debug)]
pub(crate) struct AdherenceCheck {
    pub total: usize,
    pub adherent: usize,
}
