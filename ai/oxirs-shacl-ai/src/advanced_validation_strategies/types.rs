//! Supporting types for advanced validation strategies

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::config::*;

/// Validation context for strategy selection
#[derive(Debug, Clone)]
pub struct ValidationContext {
    pub data_characteristics: DataCharacteristics,
    pub shape_characteristics: ShapeCharacteristics,
    pub domain_context: DomainContext,
    pub performance_requirements: PerformanceRequirements,
    pub quality_requirements: QualityRequirements,
    pub temporal_context: TemporalContext,
}

/// Data characteristics for context analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    pub total_triples: usize,
    pub unique_subjects: usize,
    pub unique_predicates: usize,
    pub unique_objects: usize,
    pub average_degree: f64,
    pub graph_density: f64,
    pub has_temporal_data: bool,
    pub has_spatial_data: bool,
    pub data_quality_score: f64,
    pub schema_complexity: f64,
}

/// Shape characteristics for context analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeCharacteristics {
    pub total_shapes: usize,
    pub average_constraints_per_shape: f64,
    pub max_constraint_depth: usize,
    pub has_recursive_shapes: bool,
    pub complexity_distribution: HashMap<String, usize>,
    pub dependency_graph_complexity: f64,
}

/// Domain context information
#[derive(Debug, Clone)]
pub struct DomainContext {
    pub domain_type: DomainType,
    pub domain_specific_rules: Vec<DomainRule>,
    pub semantic_relationships: HashMap<String, Vec<String>>,
    pub business_rules: Vec<BusinessRule>,
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    pub max_validation_time: Duration,
    pub max_memory_usage_mb: f64,
    pub min_throughput_per_second: f64,
    pub priority_level: PriorityLevel,
}

/// Quality requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    pub min_precision: f64,
    pub min_recall: f64,
    pub min_f1_score: f64,
    pub max_false_positive_rate: f64,
    pub max_false_negative_rate: f64,
    pub require_explainability: bool,
}

/// Temporal context for validation
#[derive(Debug, Clone)]
pub struct TemporalContext {
    pub validation_timestamp: SystemTime,
    pub data_freshness: Duration,
    pub temporal_validation_window: Option<Duration>,
    pub historical_performance: Vec<HistoricalPerformanceRecord>,
}

/// Performance record for strategy monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    pub strategy_name: String,
    pub timestamp: SystemTime,
    pub execution_time: Duration,
    pub memory_usage_mb: f64,
    pub validation_accuracy: f64,
    pub context_hash: u64,
    pub quality_metrics: super::core::QualityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionScore {
    pub score: f64,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyFactor {
    pub factor_name: String,
    pub importance: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceFactor {
    pub factor_name: String,
    pub confidence_impact: f64,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRecommendation {
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub priority: f64,
    pub estimated_improvement: f64,
}

#[derive(Debug, Clone)]
pub struct DomainRule {
    pub rule_id: String,
    pub description: String,
    pub conditions: Vec<String>,
    pub actions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BusinessRule {
    pub rule_id: String,
    pub description: String,
    pub priority: f64,
    pub enforcement_level: EnforcementLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalPerformanceRecord {
    pub timestamp: SystemTime,
    pub strategy_name: String,
    pub performance_score: f64,
    pub context_similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceFeedback {
    pub strategy_name: String,
    pub performance_delta: f64,
    pub quality_delta: f64,
    pub parameter_suggestions: HashMap<String, f64>,
}

// Placeholder implementations for supporting types
#[derive(Debug)]
pub struct StrategySelectionModel;

#[derive(Debug)]
pub struct MultiArmedBanditState;

#[derive(Debug)]
pub struct ContextPattern;

#[derive(Debug)]
pub struct SemanticAnalyzer;

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
pub struct DomainKnowledgeBase;

impl Default for DomainKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainKnowledgeBase {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
pub struct ExplanationModel;

#[derive(Debug)]
pub struct NaturalLanguageGenerator;

impl Default for NaturalLanguageGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl NaturalLanguageGenerator {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
pub struct UncertaintyModel;

#[derive(Debug)]
pub struct CalibrationData;

impl Default for CalibrationData {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationData {
    pub fn new() -> Self {
        Self
    }
}