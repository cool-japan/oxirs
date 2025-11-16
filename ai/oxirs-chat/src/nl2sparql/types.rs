//! Natural Language to SPARQL Conversion Module
//!
//! Implements template-based and LLM-powered SPARQL query generation from natural language,
//! with validation, optimization, and explanation features.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// NL2SPARQL system configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NL2SPARQLConfig {
    pub generation: GenerationConfig,
    pub templates: TemplateConfig,
    pub validation: ValidationConfig,
    pub optimization: OptimizationConfig,
    pub explanation: ExplanationConfig,
}

/// Generation strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub strategy: GenerationStrategy,
    pub fallback_strategy: GenerationStrategy,
    pub combine_strategies: bool,
    pub confidence_threshold: f32,
    pub max_iterations: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            strategy: GenerationStrategy::Hybrid,
            fallback_strategy: GenerationStrategy::Template,
            combine_strategies: true,
            confidence_threshold: 0.7,
            max_iterations: 3,
        }
    }
}

/// Generation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerationStrategy {
    Template,
    LLM,
    Hybrid,
    RuleBased,
}

/// Template system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConfig {
    pub template_dir: Option<String>,
    pub enable_custom_templates: bool,
    pub template_cache_size: usize,
    pub parameter_extraction: ParameterExtractionConfig,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            template_dir: None,
            enable_custom_templates: true,
            template_cache_size: 100,
            parameter_extraction: ParameterExtractionConfig::default(),
        }
    }
}

/// Parameter extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterExtractionConfig {
    pub entity_linking_threshold: f32,
    pub property_matching_threshold: f32,
    pub type_inference_enabled: bool,
    pub value_normalization: bool,
}

impl Default for ParameterExtractionConfig {
    fn default() -> Self {
        Self {
            entity_linking_threshold: 0.8,
            property_matching_threshold: 0.7,
            type_inference_enabled: true,
            value_normalization: true,
        }
    }
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub syntax_validation: bool,
    pub semantic_validation: bool,
    pub schema_validation: bool,
    pub dry_run_execution: bool,
    pub error_correction: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            syntax_validation: true,
            semantic_validation: true,
            schema_validation: true,
            dry_run_execution: true,
            error_correction: true,
        }
    }
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub enable_optimization: bool,
    pub query_rewriting: bool,
    pub index_hints: bool,
    pub performance_estimation: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_optimization: true,
            query_rewriting: true,
            index_hints: true,
            performance_estimation: true,
        }
    }
}

/// Explanation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationConfig {
    pub generate_explanations: bool,
    pub explanation_level: ExplanationLevel,
    pub include_reasoning: bool,
    pub include_alternatives: bool,
}

impl Default for ExplanationConfig {
    fn default() -> Self {
        Self {
            generate_explanations: true,
            explanation_level: ExplanationLevel::Detailed,
            include_reasoning: true,
            include_alternatives: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationLevel {
    Basic,
    Detailed,
    Expert,
}

/// Query intent classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryIntent {
    Select,
    Count,
    Ask,
    List,
    Describe,
    Compare,
    Filter,
    Aggregate,
    Unknown,
}

/// SPARQL query generation result
#[derive(Debug, Clone)]
pub struct SPARQLGenerationResult {
    pub query: String,
    pub confidence: f32,
    pub generation_method: GenerationMethod,
    pub parameters: HashMap<String, String>,
    pub explanation: Option<QueryExplanation>,
    pub validation_result: ValidationResult,
    pub optimization_hints: Vec<OptimizationHint>,
    pub metadata: GenerationMetadata,
}

/// Generation method used
#[derive(Debug, Clone)]
pub enum GenerationMethod {
    Template(String),
    LLM(String),
    Hybrid,
    RuleBased,
}

/// Query explanation
#[derive(Debug, Clone)]
pub struct QueryExplanation {
    pub natural_language: String,
    pub reasoning_steps: Vec<ReasoningStep>,
    pub parameter_mapping: HashMap<String, String>,
    pub alternatives: Vec<String>,
}

/// Reasoning step in query generation
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    pub step_type: ReasoningStepType,
    pub description: String,
    pub input: String,
    pub output: String,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub enum ReasoningStepType {
    EntityExtraction,
    PropertyMapping,
    TemplateSelection,
    ParameterFilling,
    QueryConstruction,
    Validation,
    Optimization,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub syntax_errors: Vec<SyntaxError>,
    pub semantic_warnings: Vec<SemanticWarning>,
    pub schema_issues: Vec<SchemaIssue>,
    pub suggestions: Vec<String>,
}

/// Syntax error in SPARQL query
#[derive(Debug, Clone)]
pub struct SyntaxError {
    pub message: String,
    pub position: Option<usize>,
    pub error_type: SyntaxErrorType,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone)]
pub enum SyntaxErrorType {
    InvalidSyntax,
    UnknownPrefix,
    InvalidIRI,
    TypeMismatch,
    MissingVariable,
}

/// Semantic warning
#[derive(Debug, Clone)]
pub struct SemanticWarning {
    pub message: String,
    pub warning_type: SemanticWarningType,
    pub severity: WarningSeverity,
}

#[derive(Debug, Clone)]
pub enum SemanticWarningType {
    UnboundVariable,
    PossibleCartesianProduct,
    ComplexQuery,
    PerformanceIssue,
}

#[derive(Debug, Clone)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
}

/// Schema validation issue
#[derive(Debug, Clone)]
pub struct SchemaIssue {
    pub message: String,
    pub issue_type: SchemaIssueType,
    pub affected_element: String,
}

#[derive(Debug, Clone)]
pub enum SchemaIssueType {
    UnknownClass,
    UnknownProperty,
    DomainRangeViolation,
    CardinalityViolation,
}

/// Optimization hint
#[derive(Debug, Clone)]
pub struct OptimizationHint {
    pub hint_type: OptimizationHintType,
    pub description: String,
    pub estimated_improvement: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum OptimizationHintType {
    AddIndex,
    ReorderTriples,
    UseFilter,
    SimplifyExpression,
}

/// Generation metadata
#[derive(Debug, Clone)]
pub struct GenerationMetadata {
    pub generation_time_ms: u64,
    pub template_used: Option<String>,
    pub llm_model_used: Option<String>,
    pub iterations: usize,
    pub fallback_used: bool,
}

/// SPARQL execution result
#[derive(Debug, Clone)]
pub struct SPARQLExecutionResult {
    pub bindings: Vec<HashMap<String, String>>,
    pub result_count: usize,
    pub execution_time_ms: u64,
    pub query_type: SPARQLQueryType,
    pub errors: Vec<String>,
}

/// SPARQL query type
#[derive(Debug, Clone)]
pub enum SPARQLQueryType {
    Select,
    Construct,
    Ask,
    Describe,
    Update,
    Unknown,
}

/// SPARQL query template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SPARQLTemplate {
    pub name: String,
    pub description: String,
    pub intent_patterns: Vec<String>,
    pub template: String,
    pub parameters: Vec<TemplateParameter>,
    pub examples: Vec<TemplateExample>,
    pub complexity: QueryComplexity,
}

/// Template parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateParameter {
    pub name: String,
    pub parameter_type: ParameterType,
    pub required: bool,
    pub default_value: Option<String>,
    pub extraction_pattern: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Entity,
    Property,
    Literal,
    Class,
    Variable,
}

/// Template example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateExample {
    pub natural_language: String,
    pub parameters: HashMap<String, String>,
    pub expected_sparql: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryComplexity {
    Simple,
    Medium,
    Complex,
    Expert,
}
