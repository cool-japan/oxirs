//! Natural Language to SPARQL Conversion Module
//!
//! Implements template-based and LLM-powered SPARQL query generation from natural language,
//! with validation, optimization, and explanation features.

use anyhow::{anyhow, Result};
use handlebars::{Handlebars, Template};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tracing::{debug, error, info, warn};

use crate::{
    llm::{ChatMessage, ChatRole, LLMManager, LLMRequest, Priority, UseCase},
    rag::{ExtractedEntity, ExtractedRelationship, QueryContext, QueryIntent},
};
use oxirs_core::{query::QueryResult, Store};

/// NL2SPARQL system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NL2SPARQLConfig {
    pub generation: GenerationConfig,
    pub templates: TemplateConfig,
    pub validation: ValidationConfig,
    pub optimization: OptimizationConfig,
    pub explanation: ExplanationConfig,
}

impl Default for NL2SPARQLConfig {
    fn default() -> Self {
        Self {
            generation: GenerationConfig::default(),
            templates: TemplateConfig::default(),
            validation: ValidationConfig::default(),
            optimization: OptimizationConfig::default(),
            explanation: ExplanationConfig::default(),
        }
    }
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct TemplateParameter {
    pub name: String,
    pub parameter_type: ParameterType,
    pub required: bool,
    pub default_value: Option<String>,
    pub extraction_pattern: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ParameterType {
    Entity,
    Property,
    Literal,
    Class,
    Variable,
}

/// Template example
#[derive(Debug, Clone)]
pub struct TemplateExample {
    pub natural_language: String,
    pub parameters: HashMap<String, String>,
    pub expected_sparql: String,
}

#[derive(Debug, Clone)]
pub enum QueryComplexity {
    Simple,
    Medium,
    Complex,
    Expert,
}

/// Main NL2SPARQL system
pub struct NL2SPARQLSystem {
    config: NL2SPARQLConfig,
    llm_manager: Option<Arc<LLMManager>>,
    template_engine: Handlebars<'static>,
    templates: HashMap<String, SPARQLTemplate>,
    validator: SPARQLValidator,
    optimizer: SPARQLOptimizer,
    store: Option<Arc<Store>>,
}

impl NL2SPARQLSystem {
    pub fn new(config: NL2SPARQLConfig, llm_manager: Option<Arc<LLMManager>>) -> Result<Self> {
        let mut system = Self {
            config,
            llm_manager,
            template_engine: Handlebars::new(),
            templates: HashMap::new(),
            validator: SPARQLValidator::new(),
            optimizer: SPARQLOptimizer::new(),
            store: None,
        };

        system.initialize_templates()?;
        Ok(system)
    }

    pub fn with_store(
        config: NL2SPARQLConfig,
        llm_manager: Option<Arc<LLMManager>>,
        store: Arc<Store>,
    ) -> Result<Self> {
        let mut system = Self {
            config,
            llm_manager,
            template_engine: Handlebars::new(),
            templates: HashMap::new(),
            validator: SPARQLValidator::new(),
            optimizer: SPARQLOptimizer::new(),
            store: Some(store),
        };

        system.initialize_templates()?;
        Ok(system)
    }

    /// Generate SPARQL query from natural language
    pub async fn generate_sparql(
        &mut self,
        query_context: &QueryContext,
    ) -> Result<SPARQLGenerationResult> {
        let start_time = std::time::Instant::now();

        info!("Starting SPARQL generation for: {}", query_context.query);

        let mut result = match self.config.generation.strategy {
            GenerationStrategy::Template => self.generate_with_templates(query_context).await?,
            GenerationStrategy::LLM => self.generate_with_llm(query_context).await?,
            GenerationStrategy::Hybrid => self.generate_hybrid(query_context).await?,
            GenerationStrategy::RuleBased => self.generate_rule_based(query_context).await?,
        };

        // Validate the generated query
        result.validation_result = self.validator.validate(&result.query)?;

        // Apply optimizations if enabled
        if self.config.optimization.enable_optimization {
            let (optimized_query, hints) = self.optimizer.optimize(&result.query)?;
            result.query = optimized_query;
            result.optimization_hints = hints;
        }

        // Generate explanation if enabled
        if self.config.explanation.generate_explanations {
            result.explanation = Some(self.generate_explanation(&result, query_context).await?);
        }

        result.metadata.generation_time_ms = start_time.elapsed().as_millis() as u64;

        info!(
            "SPARQL generation completed in {}ms",
            result.metadata.generation_time_ms
        );
        Ok(result)
    }

    /// Execute a SPARQL query against the store and return results
    pub async fn execute_sparql_query(&self, query: &str) -> Result<SPARQLExecutionResult> {
        if let Some(ref store) = self.store {
            let start_time = std::time::Instant::now();

            info!("Executing SPARQL query: {}", query);

            match store.query(query) {
                Ok(results) => {
                    let execution_time = start_time.elapsed();

                    let mut bindings = Vec::new();
                    let mut result_count = 0;

                    // Process query results
                    // Note: OxirsQueryResults is currently a placeholder
                    // TODO: Implement proper SPARQL result processing when Store.query() returns actual results
                    let _results = results; // Currently just a placeholder
                    result_count = 0; // No real results from placeholder implementation

                    info!(
                        "Query executed successfully: {} results in {}ms",
                        result_count,
                        execution_time.as_millis()
                    );

                    Ok(SPARQLExecutionResult {
                        bindings,
                        result_count,
                        execution_time_ms: execution_time.as_millis() as u64,
                        query_type: determine_query_type(query),
                        errors: Vec::new(),
                    })
                }
                Err(e) => {
                    error!("SPARQL query execution failed: {}", e);
                    Ok(SPARQLExecutionResult {
                        bindings: Vec::new(),
                        result_count: 0,
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                        query_type: determine_query_type(query),
                        errors: vec![format!("Query execution error: {}", e)],
                    })
                }
            }
        } else {
            Err(anyhow!("No store available for query execution"))
        }
    }

    fn initialize_templates(&mut self) -> Result<()> {
        // Define built-in SPARQL templates
        self.add_built_in_templates()?;

        // Load custom templates if configured
        if let Some(template_dir) = self.config.templates.template_dir.clone() {
            self.load_templates_from_directory(&template_dir)?;
        }

        Ok(())
    }

    fn add_built_in_templates(&mut self) -> Result<()> {
        // Simple factual lookup template
        let factual_template = SPARQLTemplate {
            name: "factual_lookup".to_string(),
            description: "Simple factual lookup queries".to_string(),
            intent_patterns: vec![
                "what is".to_string(),
                "who is".to_string(),
                "where is".to_string(),
                "when was".to_string(),
            ],
            template: r#"
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?answer WHERE {
    {{entity}} {{property}} ?answer .
    OPTIONAL { ?answer rdfs:label ?label }
}
"#.to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "entity".to_string(),
                    parameter_type: ParameterType::Entity,
                    required: true,
                    default_value: None,
                    extraction_pattern: Some(r"(?:what|who|where|when) (?:is|was) (.+?)(?:\?|$)".to_string()),
                },
                TemplateParameter {
                    name: "property".to_string(),
                    parameter_type: ParameterType::Property,
                    required: true,
                    default_value: Some("rdfs:label".to_string()),
                    extraction_pattern: None,
                },
            ],
            examples: vec![
                TemplateExample {
                    natural_language: "What is the capital of France?".to_string(),
                    parameters: [
                        ("entity".to_string(), "<http://example.org/France>".to_string()),
                        ("property".to_string(), "<http://example.org/capital>".to_string()),
                    ].iter().cloned().collect(),
                    expected_sparql: "SELECT ?answer WHERE { <http://example.org/France> <http://example.org/capital> ?answer }".to_string(),
                },
            ],
            complexity: QueryComplexity::Simple,
        };

        self.templates
            .insert("factual_lookup".to_string(), factual_template);

        // Relationship query template
        let relationship_template = SPARQLTemplate {
            name: "relationship_query".to_string(),
            description: "Queries about relationships between entities".to_string(),
            intent_patterns: vec![
                "how is".to_string(),
                "what is the relationship".to_string(),
                "how are.*related".to_string(),
            ],
            template: r#"
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?path ?relation WHERE {
    {{entity1}} ?relation {{entity2}} .
    OPTIONAL { ?relation rdfs:label ?path }
}
UNION {
    {{entity1}} ?p1 ?intermediate .
    ?intermediate ?p2 {{entity2}} .
    BIND(CONCAT(STR(?p1), " -> ", STR(?p2)) AS ?path)
    BIND(?p1 AS ?relation)
}
"#
            .to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "entity1".to_string(),
                    parameter_type: ParameterType::Entity,
                    required: true,
                    default_value: None,
                    extraction_pattern: Some(
                        r"(?:how|what) (?:is|are) (.+?) (?:related to|connected to) (.+?)"
                            .to_string(),
                    ),
                },
                TemplateParameter {
                    name: "entity2".to_string(),
                    parameter_type: ParameterType::Entity,
                    required: true,
                    default_value: None,
                    extraction_pattern: None,
                },
            ],
            examples: vec![],
            complexity: QueryComplexity::Medium,
        };

        self.templates
            .insert("relationship_query".to_string(), relationship_template);

        // List query template
        let list_template = SPARQLTemplate {
            name: "list_query".to_string(),
            description: "Queries that return lists of items".to_string(),
            intent_patterns: vec![
                "list all".to_string(),
                "show me".to_string(),
                "what are".to_string(),
                "which".to_string(),
            ],
            template: r#"
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT DISTINCT ?item ?label WHERE {
    ?item rdf:type {{type}} .
    {{#if filter}}
    ?item {{filter_property}} {{filter_value}} .
    {{/if}}
    OPTIONAL { ?item rdfs:label ?label }
}
ORDER BY ?label
LIMIT {{limit}}
"#
            .to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "type".to_string(),
                    parameter_type: ParameterType::Class,
                    required: true,
                    default_value: None,
                    extraction_pattern: Some(
                        r"(?:list all|show me|what are) (.+?)(?:\?|$)".to_string(),
                    ),
                },
                TemplateParameter {
                    name: "limit".to_string(),
                    parameter_type: ParameterType::Literal,
                    required: false,
                    default_value: Some("100".to_string()),
                    extraction_pattern: None,
                },
            ],
            examples: vec![],
            complexity: QueryComplexity::Simple,
        };

        self.templates
            .insert("list_query".to_string(), list_template);

        Ok(())
    }

    fn load_templates_from_directory(&mut self, _dir: &str) -> Result<()> {
        // TODO: Implement loading templates from directory
        Ok(())
    }

    async fn generate_with_templates(
        &self,
        query_context: &QueryContext,
    ) -> Result<SPARQLGenerationResult> {
        // Select best matching template
        let template = self.select_template(query_context)?;

        // Extract parameters from the query
        let parameters = self.extract_parameters(&template, query_context)?;

        // Fill template with parameters
        let sparql_query = self.fill_template(&template, &parameters)?;

        Ok(SPARQLGenerationResult {
            query: sparql_query,
            confidence: 0.8, // TODO: Calculate actual confidence
            generation_method: GenerationMethod::Template(template.name.clone()),
            parameters,
            explanation: None,
            validation_result: ValidationResult {
                is_valid: true,
                syntax_errors: Vec::new(),
                semantic_warnings: Vec::new(),
                schema_issues: Vec::new(),
                suggestions: Vec::new(),
            },
            optimization_hints: Vec::new(),
            metadata: GenerationMetadata {
                generation_time_ms: 0,
                template_used: Some(template.name.clone()),
                llm_model_used: None,
                iterations: 1,
                fallback_used: false,
            },
        })
    }

    async fn generate_with_llm(
        &mut self,
        query_context: &QueryContext,
    ) -> Result<SPARQLGenerationResult> {
        let system_prompt = self.create_sparql_generation_prompt();

        if let Some(ref llm_manager) = self.llm_manager {
            let user_message = format!(
                "Convert this natural language query to SPARQL: {}",
                query_context.query
            );

            let llm_request = LLMRequest {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: user_message,
                    metadata: None,
                }],
                system_prompt: Some(system_prompt),
                temperature: 0.3, // Lower temperature for more deterministic code generation
                max_tokens: Some(1000),
                use_case: UseCase::SparqlGeneration,
                priority: Priority::Normal,
                timeout: None,
            };

            match llm_manager.generate_response(llm_request).await {
                Ok(response) => {
                    let sparql_query = self.extract_sparql_from_response(&response.content)?;

                    Ok(SPARQLGenerationResult {
                        query: sparql_query,
                        confidence: 0.7, // TODO: Calculate based on LLM confidence
                        generation_method: GenerationMethod::LLM(response.model_used.clone()),
                        parameters: HashMap::new(),
                        explanation: None,
                        validation_result: ValidationResult {
                            is_valid: true,
                            syntax_errors: Vec::new(),
                            semantic_warnings: Vec::new(),
                            schema_issues: Vec::new(),
                            suggestions: Vec::new(),
                        },
                        optimization_hints: Vec::new(),
                        metadata: GenerationMetadata {
                            generation_time_ms: response.latency.as_millis() as u64,
                            template_used: None,
                            llm_model_used: Some(response.model_used),
                            iterations: 1,
                            fallback_used: false,
                        },
                    })
                }
                Err(e) => {
                    warn!("LLM generation failed: {}", e);
                    // Fallback to template-based generation
                    let mut result = self.generate_with_templates(query_context).await?;
                    result.metadata.fallback_used = true;
                    Ok(result)
                }
            }
        } else {
            // No LLM available, fallback to templates
            self.generate_with_templates(query_context).await
        }
    }

    async fn generate_hybrid(
        &mut self,
        query_context: &QueryContext,
    ) -> Result<SPARQLGenerationResult> {
        // Try template-based first, then enhance with LLM if needed
        let template_result = self.generate_with_templates(query_context).await?;

        if template_result.confidence < self.config.generation.confidence_threshold {
            // Template confidence is low, try LLM enhancement
            if let Ok(llm_result) = self.generate_with_llm(query_context).await {
                if llm_result.confidence > template_result.confidence {
                    return Ok(SPARQLGenerationResult {
                        generation_method: GenerationMethod::Hybrid,
                        ..llm_result
                    });
                }
            }
        }

        Ok(SPARQLGenerationResult {
            generation_method: GenerationMethod::Hybrid,
            ..template_result
        })
    }

    async fn generate_rule_based(
        &self,
        _query_context: &QueryContext,
    ) -> Result<SPARQLGenerationResult> {
        // TODO: Implement rule-based generation
        Err(anyhow!("Rule-based generation not yet implemented"))
    }

    fn select_template(&self, query_context: &QueryContext) -> Result<&SPARQLTemplate> {
        let query_lower = query_context.query.to_lowercase();

        // Match based on intent and patterns
        for template in self.templates.values() {
            for pattern in &template.intent_patterns {
                if query_lower.contains(pattern) {
                    return Ok(template);
                }
            }
        }

        // Default to factual lookup if no specific pattern matches
        self.templates
            .get("factual_lookup")
            .ok_or_else(|| anyhow!("No suitable template found"))
    }

    fn extract_parameters(
        &self,
        template: &SPARQLTemplate,
        query_context: &QueryContext,
    ) -> Result<HashMap<String, String>> {
        let mut parameters = HashMap::new();

        for param in &template.parameters {
            if let Some(ref pattern) = param.extraction_pattern {
                if let Ok(regex) = Regex::new(pattern) {
                    if let Some(captures) = regex.captures(&query_context.query) {
                        if let Some(captured) = captures.get(1) {
                            parameters.insert(param.name.clone(), captured.as_str().to_string());
                        }
                    }
                }
            }

            // Use default value if parameter not extracted and has default
            if !parameters.contains_key(&param.name) {
                if let Some(ref default) = param.default_value {
                    parameters.insert(param.name.clone(), default.clone());
                } else if param.required {
                    return Err(anyhow!("Required parameter '{}' not found", param.name));
                }
            }
        }

        Ok(parameters)
    }

    fn fill_template(
        &self,
        template: &SPARQLTemplate,
        parameters: &HashMap<String, String>,
    ) -> Result<String> {
        let template_obj = self
            .template_engine
            .render_template(&template.template, &parameters)?;
        Ok(template_obj)
    }

    fn create_sparql_generation_prompt(&self) -> String {
        r#"You are an expert at converting natural language queries to SPARQL queries.
        
Guidelines:
1. Generate valid SPARQL 1.1 syntax
2. Use appropriate prefixes (rdf, rdfs, owl, etc.)
3. Include OPTIONAL clauses for optional data
4. Use FILTER clauses for constraints
5. Add LIMIT clauses for list queries
6. Use proper variable naming
7. Include comments explaining complex parts

Always respond with just the SPARQL query, no additional explanation unless requested."#
            .to_string()
    }

    fn extract_sparql_from_response(&self, response: &str) -> Result<String> {
        // Extract SPARQL query from LLM response
        // Look for patterns like ```sparql ... ``` or just return the whole response if it looks like SPARQL

        if let Some(start) = response.find("```sparql") {
            if let Some(end) = response[start..].find("```") {
                let query = &response[start + 9..start + end];
                return Ok(query.trim().to_string());
            }
        }

        if let Some(start) = response.find("```") {
            if let Some(end) = response[start + 3..].find("```") {
                let query = &response[start + 3..start + 3 + end];
                return Ok(query.trim().to_string());
            }
        }

        // If no code blocks found, check if the response looks like SPARQL
        let trimmed = response.trim();
        if trimmed.to_uppercase().contains("SELECT")
            || trimmed.to_uppercase().contains("CONSTRUCT")
            || trimmed.to_uppercase().contains("ASK")
            || trimmed.to_uppercase().contains("DESCRIBE")
        {
            return Ok(trimmed.to_string());
        }

        Err(anyhow!("Could not extract SPARQL query from response"))
    }

    async fn generate_explanation(
        &self,
        result: &SPARQLGenerationResult,
        query_context: &QueryContext,
    ) -> Result<QueryExplanation> {
        // TODO: Implement explanation generation
        Ok(QueryExplanation {
            natural_language: format!("Generated SPARQL query for: {}", query_context.query),
            reasoning_steps: Vec::new(),
            parameter_mapping: result.parameters.clone(),
            alternatives: Vec::new(),
        })
    }
}

/// SPARQL validation component with comprehensive checks
pub struct SPARQLValidator {
    syntax_patterns: HashMap<String, Regex>,
    common_prefixes: HashMap<String, String>,
}

impl SPARQLValidator {
    pub fn new() -> Self {
        let mut syntax_patterns = HashMap::new();

        // Basic SPARQL syntax patterns
        syntax_patterns.insert(
            "select_pattern".to_string(),
            Regex::new(r"(?i)^\s*SELECT\s+(?:DISTINCT\s+)?(?:\*|\?\w+(?:\s+\?\w+)*)\s+WHERE\s*\{")
                .unwrap(),
        );
        syntax_patterns.insert(
            "construct_pattern".to_string(),
            Regex::new(r"(?i)^\s*CONSTRUCT\s*\{").unwrap(),
        );
        syntax_patterns.insert(
            "ask_pattern".to_string(),
            Regex::new(r"(?i)^\s*ASK\s*\{").unwrap(),
        );
        syntax_patterns.insert(
            "describe_pattern".to_string(),
            Regex::new(r"(?i)^\s*DESCRIBE\s+").unwrap(),
        );
        syntax_patterns.insert(
            "variable_pattern".to_string(),
            Regex::new(r"\?[a-zA-Z][a-zA-Z0-9_]*").unwrap(),
        );
        syntax_patterns.insert(
            "iri_pattern".to_string(),
            Regex::new(r"<[^<>\s]+>").unwrap(),
        );

        // Common SPARQL prefixes
        let mut common_prefixes = HashMap::new();
        common_prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        common_prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        common_prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        common_prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        common_prefixes.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        common_prefixes.insert(
            "skos".to_string(),
            "http://www.w3.org/2004/02/skos/core#".to_string(),
        );

        Self {
            syntax_patterns,
            common_prefixes,
        }
    }

    pub fn validate(&self, query: &str) -> Result<ValidationResult> {
        let mut syntax_errors = Vec::new();
        let mut semantic_warnings = Vec::new();
        let mut schema_issues = Vec::new();
        let mut suggestions = Vec::new();

        // Basic syntax validation
        if !self.validate_basic_syntax(query) {
            syntax_errors.push(SyntaxError {
                message: "Query does not match any valid SPARQL query pattern".to_string(),
                position: Some(0),
                error_type: SyntaxErrorType::InvalidSyntax,
                suggestion: Some(
                    "Ensure query starts with SELECT, CONSTRUCT, ASK, or DESCRIBE".to_string(),
                ),
            });
        }

        // Validate query structure
        self.validate_query_structure(query, &mut syntax_errors, &mut semantic_warnings)?;

        // Check for common issues
        self.check_common_issues(query, &mut semantic_warnings, &mut suggestions)?;

        // Validate prefixes
        self.validate_prefixes(query, &mut syntax_errors, &mut suggestions)?;

        // Check for performance issues
        self.check_performance_issues(query, &mut semantic_warnings)?;

        let is_valid = syntax_errors.is_empty();

        Ok(ValidationResult {
            is_valid,
            syntax_errors,
            semantic_warnings,
            schema_issues,
            suggestions,
        })
    }

    fn validate_basic_syntax(&self, query: &str) -> bool {
        let query_trimmed = query.trim();

        // Check if query starts with valid SPARQL keyword
        for pattern in self.syntax_patterns.values() {
            if pattern.is_match(query_trimmed) {
                return true;
            }
        }

        false
    }

    fn validate_query_structure(
        &self,
        query: &str,
        syntax_errors: &mut Vec<SyntaxError>,
        semantic_warnings: &mut Vec<SemanticWarning>,
    ) -> Result<()> {
        // Check for balanced braces
        let open_braces = query.matches('{').count();
        let close_braces = query.matches('}').count();

        if open_braces != close_braces {
            syntax_errors.push(SyntaxError {
                message: format!(
                    "Unbalanced braces: {} open, {} close",
                    open_braces, close_braces
                ),
                position: None,
                error_type: SyntaxErrorType::InvalidSyntax,
                suggestion: Some(
                    "Check that all opening braces have matching closing braces".to_string(),
                ),
            });
        }

        // Check for variables in SELECT that aren't used in WHERE
        if let Some(var_pattern) = self.syntax_patterns.get("variable_pattern") {
            let variables: HashSet<&str> =
                var_pattern.find_iter(query).map(|m| m.as_str()).collect();

            if variables.is_empty()
                && query.to_uppercase().contains("SELECT")
                && !query.contains("*")
            {
                semantic_warnings.push(SemanticWarning {
                    message: "No variables found in SELECT query".to_string(),
                    warning_type: SemanticWarningType::UnboundVariable,
                    severity: WarningSeverity::Medium,
                });
            }
        }

        Ok(())
    }

    fn check_common_issues(
        &self,
        query: &str,
        semantic_warnings: &mut Vec<SemanticWarning>,
        suggestions: &mut Vec<String>,
    ) -> Result<()> {
        let query_upper = query.to_uppercase();

        // Check for potential Cartesian products
        if !query_upper.contains("FILTER") && query_upper.matches(".").count() > 3 {
            semantic_warnings.push(SemanticWarning {
                message: "Query may produce Cartesian product - consider adding FILTER clauses"
                    .to_string(),
                warning_type: SemanticWarningType::PossibleCartesianProduct,
                severity: WarningSeverity::Medium,
            });
            suggestions.push(
                "Add FILTER clauses to constrain results and improve performance".to_string(),
            );
        }

        // Check for missing LIMIT
        if !query_upper.contains("LIMIT") && query_upper.contains("SELECT") {
            suggestions
                .push("Consider adding a LIMIT clause to prevent large result sets".to_string());
        }

        // Check for complex queries without ORDER BY
        if query.len() > 200 && !query_upper.contains("ORDER BY") && query_upper.contains("SELECT")
        {
            suggestions.push("Consider adding ORDER BY for consistent result ordering".to_string());
        }

        Ok(())
    }

    fn validate_prefixes(
        &self,
        query: &str,
        syntax_errors: &mut Vec<SyntaxError>,
        suggestions: &mut Vec<String>,
    ) -> Result<()> {
        // Extract used prefixes
        let prefix_usage_pattern = Regex::new(r"(\w+):")?;
        let used_prefixes: HashSet<&str> = prefix_usage_pattern
            .find_iter(query)
            .map(|m| m.as_str().trim_end_matches(':'))
            .collect();

        // Extract declared prefixes
        let prefix_declaration_pattern =
            Regex::new(r"(?i)PREFIX\s+(\w+):").unwrap_or_else(|_| Regex::new(r"PREFIX").unwrap());
        let declared_prefixes: HashSet<&str> = prefix_declaration_pattern
            .captures_iter(query)
            .filter_map(|cap| cap.get(1))
            .map(|m| m.as_str())
            .collect();

        // Check for undeclared prefixes
        for prefix in &used_prefixes {
            if !declared_prefixes.contains(prefix) && self.common_prefixes.contains_key(*prefix) {
                suggestions.push(format!(
                    "Add prefix declaration: PREFIX {}: <{}>",
                    prefix,
                    self.common_prefixes.get(*prefix).unwrap()
                ));
            } else if !declared_prefixes.contains(prefix) {
                syntax_errors.push(SyntaxError {
                    message: format!("Undeclared prefix: {}", prefix),
                    position: None,
                    error_type: SyntaxErrorType::UnknownPrefix,
                    suggestion: Some(format!("Declare prefix {} or use full IRI", prefix)),
                });
            }
        }

        Ok(())
    }

    fn check_performance_issues(
        &self,
        query: &str,
        semantic_warnings: &mut Vec<SemanticWarning>,
    ) -> Result<()> {
        let query_upper = query.to_uppercase();

        // Check for expensive operations
        if query_upper.contains("REGEX") {
            semantic_warnings.push(SemanticWarning {
                message: "REGEX operations can be expensive - consider alternatives if possible"
                    .to_string(),
                warning_type: SemanticWarningType::PerformanceIssue,
                severity: WarningSeverity::Low,
            });
        }

        if query_upper.contains("UNION") && query_upper.matches("UNION").count() > 2 {
            semantic_warnings.push(SemanticWarning {
                message: "Multiple UNION clauses may impact performance".to_string(),
                warning_type: SemanticWarningType::PerformanceIssue,
                severity: WarningSeverity::Medium,
            });
        }

        if query.len() > 1000 {
            semantic_warnings.push(SemanticWarning {
                message: "Very long query may be difficult to optimize".to_string(),
                warning_type: SemanticWarningType::ComplexQuery,
                severity: WarningSeverity::Low,
            });
        }

        Ok(())
    }
}

/// SPARQL optimization component with query rewriting capabilities
pub struct SPARQLOptimizer {
    optimization_rules: Vec<OptimizationRule>,
}

struct OptimizationRule {
    name: String,
    pattern: Regex,
    replacement: String,
    description: String,
    estimated_improvement: f32,
}

impl SPARQLOptimizer {
    pub fn new() -> Self {
        let mut optimization_rules = Vec::new();

        // Add DISTINCT optimization
        optimization_rules.push(OptimizationRule {
            name: "redundant_distinct".to_string(),
            pattern: Regex::new(r"(?i)SELECT\s+DISTINCT\s+DISTINCT").unwrap(),
            replacement: "SELECT DISTINCT".to_string(),
            description: "Remove redundant DISTINCT clauses".to_string(),
            estimated_improvement: 0.1,
        });

        // Add LIMIT pushdown optimization
        optimization_rules.push(OptimizationRule {
            name: "limit_optimization".to_string(),
            pattern: Regex::new(r"(?i)ORDER\s+BY\s+[^}]+}\s*$").unwrap(),
            replacement: "$0 LIMIT 1000".to_string(),
            description: "Add default LIMIT for safety".to_string(),
            estimated_improvement: 0.3,
        });

        Self { optimization_rules }
    }

    pub fn optimize(&self, query: &str) -> Result<(String, Vec<OptimizationHint>)> {
        let mut optimized_query = query.to_string();
        let mut hints = Vec::new();

        // Apply optimization rules
        for rule in &self.optimization_rules {
            if rule.pattern.is_match(&optimized_query) {
                optimized_query = rule
                    .pattern
                    .replace_all(&optimized_query, &rule.replacement)
                    .to_string();
                hints.push(OptimizationHint {
                    hint_type: OptimizationHintType::SimplifyExpression,
                    description: rule.description.clone(),
                    estimated_improvement: Some(rule.estimated_improvement),
                });
            }
        }

        // Additional optimizations
        let additional_hints = self.analyze_query_structure(&optimized_query)?;
        hints.extend(additional_hints);

        // Query rewriting optimizations
        optimized_query = self.rewrite_query_patterns(optimized_query)?;

        Ok((optimized_query, hints))
    }

    fn analyze_query_structure(&self, query: &str) -> Result<Vec<OptimizationHint>> {
        let mut hints = Vec::new();
        let query_upper = query.to_uppercase();

        // Suggest index hints for large result sets
        if !query_upper.contains("LIMIT") && query_upper.contains("SELECT") {
            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::AddIndex,
                description: "Consider adding LIMIT clause to prevent large result sets"
                    .to_string(),
                estimated_improvement: Some(0.5),
            });
        }

        // Suggest reordering for better join performance
        if query_upper.contains("OPTIONAL") && query_upper.contains("FILTER") {
            let optional_pos = query_upper.find("OPTIONAL").unwrap_or(0);
            let filter_pos = query_upper.find("FILTER").unwrap_or(0);

            if filter_pos > optional_pos {
                hints.push(OptimizationHint {
                    hint_type: OptimizationHintType::ReorderTriples,
                    description:
                        "Consider moving FILTER clauses before OPTIONAL for better performance"
                            .to_string(),
                    estimated_improvement: Some(0.3),
                });
            }
        }

        // Check for Cartesian products
        let triple_count = query_upper.matches(" . ").count();
        if triple_count > 5 && !query_upper.contains("FILTER") {
            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::UseFilter,
                description:
                    "Multiple triple patterns without filters may create Cartesian products"
                        .to_string(),
                estimated_improvement: Some(0.7),
            });
        }

        Ok(hints)
    }

    fn rewrite_query_patterns(&self, query: String) -> Result<String> {
        let mut rewritten = query;

        // Rewrite inefficient UNION patterns
        let union_pattern = Regex::new(r"(?i)\{\s*(.+?)\s*\}\s*UNION\s*\{\s*(.+?)\s*\}")?;
        if union_pattern.is_match(&rewritten) {
            // This is a simplified rewrite - in practice, we'd need more sophisticated analysis
            info!("Detected UNION pattern that could potentially be optimized");
        }

        // Rewrite FILTER patterns for better performance
        let filter_pattern =
            Regex::new(r#"(?i)FILTER\s*\(\s*regex\s*\(\s*\?(\w+)\s*,\s*"([^"]+)"\s*\)\s*\)"#)?;
        rewritten = filter_pattern
            .replace_all(&rewritten, |caps: &regex::Captures| {
                format!(
                    "FILTER(CONTAINS(LCASE(?{}), LCASE(\"{}\")))",
                    &caps[1], &caps[2]
                )
            })
            .to_string();

        Ok(rewritten)
    }

    /// Get optimization recommendations for a query
    pub fn get_recommendations(&self, query: &str) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        let query_upper = query.to_uppercase();

        // Basic recommendations
        if !query_upper.contains("PREFIX") && query.contains(":") {
            recommendations.push("Add PREFIX declarations for better readability".to_string());
        }

        if query.len() > 500 && !query_upper.contains("LIMIT") {
            recommendations.push("Add LIMIT clause for large queries".to_string());
        }

        if query_upper.contains("SELECT *") {
            recommendations
                .push("Select specific variables instead of * for better performance".to_string());
        }

        if query_upper.matches("OPTIONAL").count() > 3 {
            recommendations.push("Consider restructuring multiple OPTIONAL clauses".to_string());
        }

        Ok(recommendations)
    }
}

/// Determine the type of SPARQL query
fn determine_query_type(query: &str) -> SPARQLQueryType {
    let query_upper = query.trim().to_uppercase();

    if query_upper.starts_with("SELECT") {
        SPARQLQueryType::Select
    } else if query_upper.starts_with("CONSTRUCT") {
        SPARQLQueryType::Construct
    } else if query_upper.starts_with("ASK") {
        SPARQLQueryType::Ask
    } else if query_upper.starts_with("DESCRIBE") {
        SPARQLQueryType::Describe
    } else if query_upper.starts_with("INSERT")
        || query_upper.starts_with("DELETE")
        || query_upper.starts_with("LOAD")
        || query_upper.starts_with("CLEAR")
    {
        SPARQLQueryType::Update
    } else {
        SPARQLQueryType::Unknown
    }
}
