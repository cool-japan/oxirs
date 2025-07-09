//! Natural Language to SPARQL Conversion Module
//!
//! Implements template-based and LLM-powered SPARQL query generation from natural language,
//! with validation, optimization, and explanation features.

use anyhow::{anyhow, Result};
use handlebars::Handlebars;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
    sync::Arc,
};
use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, error, info, warn};

use crate::{
    llm::{ChatMessage, ChatRole, LLMManager, LLMRequest, Priority, UseCase},
    rag::QueryContext,
};
use oxirs_core::Store;

/// NL2SPARQL system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
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

/// Main NL2SPARQL system
pub struct NL2SPARQLSystem {
    config: NL2SPARQLConfig,
    llm_manager: Option<Arc<TokioMutex<LLMManager>>>,
    template_engine: Handlebars<'static>,
    templates: HashMap<String, SPARQLTemplate>,
    validator: SPARQLValidator,
    optimizer: SPARQLOptimizer,
    store: Option<Arc<dyn Store>>,
}

impl NL2SPARQLSystem {
    pub fn new(
        config: NL2SPARQLConfig,
        llm_manager: Option<Arc<TokioMutex<LLMManager>>>,
    ) -> Result<Self> {
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
        llm_manager: Option<Arc<TokioMutex<LLMManager>>>,
        store: Arc<dyn Store>,
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

        let query_text = query_context
            .conversation_history
            .iter()
            .rev()
            .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
            .map(|msg| msg.content.as_str())
            .unwrap_or("Unknown query");
        info!("Starting SPARQL generation for: {}", query_text);

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

                    // Process query results based on query type
                    match results.results() {
                        oxirs_core::rdf_store::QueryResults::Bindings(result_bindings) => {
                            // Convert Variable bindings to String-based bindings for API compatibility
                            for binding in result_bindings {
                                let mut string_binding = HashMap::new();
                                for var in binding.variables() {
                                    if let Some(term) = binding.get(var) {
                                        string_binding.insert(var.clone(), format!("{term}"));
                                    }
                                }
                                bindings.push(string_binding);
                            }
                            result_count = bindings.len();
                        }
                        oxirs_core::rdf_store::QueryResults::Boolean(answer) => {
                            // For ASK queries, create a single binding with the boolean result
                            let mut ask_binding = HashMap::new();
                            ask_binding.insert("result".to_string(), answer.to_string());
                            bindings.push(ask_binding);
                            result_count = 1;
                        }
                        oxirs_core::rdf_store::QueryResults::Graph(quads) => {
                            // For CONSTRUCT queries, convert quads to bindings
                            for (index, quad) in quads.iter().enumerate() {
                                let mut quad_binding = HashMap::new();
                                quad_binding
                                    .insert("subject".to_string(), format!("{}", quad.subject()));
                                quad_binding.insert(
                                    "predicate".to_string(),
                                    format!("{}", quad.predicate()),
                                );
                                quad_binding
                                    .insert("object".to_string(), format!("{}", quad.object()));
                                let graph = quad.graph_name();
                                quad_binding.insert("graph".to_string(), format!("{graph}"));
                                quad_binding.insert("quad_index".to_string(), index.to_string());
                                bindings.push(quad_binding);
                            }
                            result_count = quads.len();
                        }
                    }

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

    fn load_templates_from_directory(&mut self, dir: &str) -> Result<()> {
        let dir_path = Path::new(dir);

        if !dir_path.exists() {
            warn!("Template directory does not exist: {}", dir);
            return Ok(());
        }

        info!("Loading templates from directory: {}", dir);

        let entries = fs::read_dir(dir_path)?;
        let mut loaded_count = 0;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let extension = path.extension().and_then(|s| s.to_str());

                match extension {
                    Some("json") => {
                        if let Err(e) = self.load_json_template(&path) {
                            error!("Failed to load JSON template from {:?}: {}", path, e);
                        } else {
                            loaded_count += 1;
                            debug!("Loaded template from {:?}", path);
                        }
                    }
                    Some("yaml") | Some("yml") => {
                        if let Err(e) = self.load_yaml_template(&path) {
                            error!("Failed to load YAML template from {:?}: {}", path, e);
                        } else {
                            loaded_count += 1;
                            debug!("Loaded template from {:?}", path);
                        }
                    }
                    _ => {
                        debug!("Skipping non-template file: {:?}", path);
                    }
                }
            }
        }

        info!("Loaded {} templates from directory: {}", loaded_count, dir);
        Ok(())
    }

    /// Load a single JSON template file
    fn load_json_template(&mut self, path: &Path) -> Result<()> {
        let content = fs::read_to_string(path)?;
        let template: SPARQLTemplate = serde_json::from_str(&content)?;

        // Validate the template
        if template.name.is_empty() {
            return Err(anyhow!("Template name cannot be empty"));
        }

        if template.template.is_empty() {
            return Err(anyhow!("Template SPARQL cannot be empty"));
        }

        // Register the template in the Handlebars engine
        self.template_engine
            .register_template_string(&template.name, &template.template)?;

        // Store the template
        self.templates.insert(template.name.clone(), template);

        Ok(())
    }

    /// Load a single YAML template file
    fn load_yaml_template(&mut self, path: &Path) -> Result<()> {
        let content = fs::read_to_string(path)?;
        let template: SPARQLTemplate = serde_yaml::from_str(&content)?;

        // Validate the template
        if template.name.is_empty() {
            return Err(anyhow!("Template name cannot be empty"));
        }

        if template.template.is_empty() {
            return Err(anyhow!("Template SPARQL cannot be empty"));
        }

        // Register the template in the Handlebars engine
        self.template_engine
            .register_template_string(&template.name, &template.template)?;

        // Store the template
        self.templates.insert(template.name.clone(), template);

        Ok(())
    }

    async fn generate_with_templates(
        &self,
        query_context: &QueryContext,
    ) -> Result<SPARQLGenerationResult> {
        // Select best matching template
        let template = self.select_template(query_context)?;

        // Extract parameters from the query
        let parameters = self.extract_parameters(template, query_context)?;

        // Fill template with parameters
        let sparql_query = self.fill_template(template, &parameters)?;

        // Calculate confidence based on template quality and parameter extraction
        let confidence =
            self.calculate_template_confidence(template, &parameters, query_context)?;

        Ok(SPARQLGenerationResult {
            query: sparql_query,
            confidence,
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

        if let Some(llm_manager) = self.llm_manager.clone() {
            let query_text = query_context
                .conversation_history
                .iter()
                .rev()
                .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
                .map(|msg| msg.content.as_str())
                .unwrap_or("Unknown query");
            let user_message = format!(
                "Convert this natural language query to SPARQL: {query_text}"
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

            let mut manager = llm_manager.lock().await;
            match manager.generate_response(llm_request).await {
                Ok(response) => {
                    let sparql_query = self.extract_sparql_from_response(&response.content)?;

                    // Calculate confidence based on LLM response quality and SPARQL validity
                    let confidence =
                        self.calculate_llm_confidence(&response, &sparql_query, query_context)?;

                    Ok(SPARQLGenerationResult {
                        query: sparql_query,
                        confidence,
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
        query_context: &QueryContext,
    ) -> Result<SPARQLGenerationResult> {
        // Extract the user query from conversation history
        let query_text = query_context
            .conversation_history
            .iter()
            .rev()
            .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
            .map(|msg| msg.content.as_str())
            .unwrap_or("")
            .to_lowercase();

        if query_text.is_empty() {
            return Err(anyhow!("No user query found in context"));
        }

        // Initialize rule-based analysis
        let mut parameters = HashMap::new();
        let mut confidence: f32 = 0.0;
        let mut hints = Vec::new();

        // Rule 1: Basic SELECT pattern detection
        let select_query = if query_text.contains("find") || query_text.contains("show") || 
                              query_text.contains("list") || query_text.contains("get") ||
                              query_text.contains("what") || query_text.contains("which") {
            confidence += 0.3;
            true
        } else {
            false
        };

        // Rule 2: COUNT pattern detection
        let count_query = if query_text.contains("how many") || query_text.contains("count") ||
                            query_text.contains("number of") {
            confidence += 0.3;
            true
        } else {
            false
        };

        // Rule 3: ASK pattern detection  
        let ask_query = if query_text.starts_with("is") || query_text.starts_with("does") ||
                          query_text.starts_with("has") || query_text.contains("whether") {
            confidence += 0.3;
            true
        } else {
            false
        };

        // Rule 4: Entity extraction using extracted entities from context
        let mut subjects = Vec::new();
        let mut predicates = Vec::new();
        let mut objects = Vec::new();

        // Use entities from query context if available (entities is Vec<Vec<String>>)
        for entity_group in &query_context.entities {
            for entity in entity_group {
                // Treat all entities as potential subjects/objects
                if entity.starts_with("http") || entity.starts_with("urn:") {
                    subjects.push(format!("<{entity}>"));
                } else {
                    objects.push(format!("\"{entity}\""));
                }
                parameters.insert(format!("entity_{entity}"), entity.clone());
            }
        }

        // Rule 5: Common property patterns
        if query_text.contains("name") || query_text.contains("label") {
            predicates.push("rdfs:label".to_string());
            confidence += 0.1;
        }
        if query_text.contains("type") || query_text.contains("kind") {
            predicates.push("rdf:type".to_string());
            confidence += 0.1;
        }
        if query_text.contains("born") || query_text.contains("birth") {
            predicates.push("dbo:birthDate".to_string());
            confidence += 0.1;
        }
        if query_text.contains("location") || query_text.contains("place") {
            predicates.push("dbo:location".to_string());
            confidence += 0.1;
        }

        // Rule 6: Generate SPARQL based on detected patterns
        let sparql_query = if count_query {
            // Generate COUNT query
            let subject_var = if subjects.is_empty() { "?entity" } else { "?s" };
            let predicate = if predicates.is_empty() { "?p" } else { &predicates[0] };
            let object_var = if objects.is_empty() { "?o" } else { &objects[0] };

            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::SimplifyExpression,
                description: "COUNT queries can be optimized with LIMIT".to_string(),
                estimated_improvement: Some(0.5),
            });

            format!(
                "SELECT (COUNT(*) AS ?count) WHERE {{\n  {subject_var} {predicate} {object_var} .\n}}"
            )
        } else if ask_query {
            // Generate ASK query
            let subject = if subjects.is_empty() { "?s" } else { &subjects[0] };
            let predicate = if predicates.is_empty() { "?p" } else { &predicates[0] };
            let object = if objects.is_empty() { "?o" } else { &objects[0] };

            format!(
                "ASK {{\n  {subject} {predicate} {object} .\n}}"
            )
        } else {
            // Generate basic SELECT query
            let mut select_vars = Vec::new();
            let mut where_patterns = Vec::new();

            if subjects.is_empty() {
                select_vars.push("?subject");
                where_patterns.push(format!("?subject {} ?object", 
                    if predicates.is_empty() { "?predicate" } else { &predicates[0] }));
            } else {
                select_vars.push("?object");
                where_patterns.push(format!("{} {} ?object", 
                    &subjects[0], 
                    if predicates.is_empty() { "?predicate" } else { &predicates[0] }));
            }

            if predicates.is_empty() {
                select_vars.push("?predicate");
            }

            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::UseFilter,
                description: "Consider adding LIMIT clause for large result sets".to_string(),
                estimated_improvement: Some(0.8),
            });

            format!(
                "SELECT {} WHERE {{\n  {}\n}} LIMIT 100",
                select_vars.join(" "),
                where_patterns.join(" .\n  ")
            )
        };

        // Rule 7: Confidence adjustment based on query complexity and rule coverage
        if !subjects.is_empty() { confidence += 0.2; }
        if !predicates.is_empty() { confidence += 0.2; }
        if !objects.is_empty() { confidence += 0.1; }

        // Ensure confidence is within bounds
        confidence = confidence.min(1.0).max(0.0);

        // If confidence is too low, provide a generic fallback
        let final_query = if confidence < 0.3 {
            // Generic fallback query
            confidence = 0.3;
            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::SimplifyExpression,
                description: "Low confidence - using generic query pattern".to_string(),
                estimated_improvement: Some(0.2),
            });
            "SELECT ?subject ?predicate ?object WHERE {\n  ?subject ?predicate ?object .\n} LIMIT 10".to_string()
        } else {
            sparql_query
        };

        // Create generation metadata
        let metadata = GenerationMetadata {
            generation_time_ms: 50, // Rule-based is fast
            template_used: None,     // No template used
            llm_model_used: None,    // No LLM used
            iterations: 1,           // Single iteration for rule-based
            fallback_used: confidence < 0.3, // Fallback used if confidence too low
        };

        // Create explanation
        let explanation = QueryExplanation {
            natural_language: format!(
                "This query was generated using rule-based analysis. Detected patterns: {}, confidence: {:.1}%",
                if select_query { "selection" } 
                else if count_query { "counting" } 
                else if ask_query { "yes/no question" } 
                else { "general query" },
                confidence * 100.0
            ),
            reasoning_steps: vec![
                ReasoningStep {
                    step_type: ReasoningStepType::EntityExtraction,
                    description: "Analyzed natural language for query patterns".to_string(),
                    input: query_text.clone(),
                    output: format!("Detected: select={select_query}, count={count_query}, ask={ask_query}"),
                    confidence: 0.8,
                },
                ReasoningStep {
                    step_type: ReasoningStepType::EntityExtraction,
                    description: "Extracted entities from conversation context".to_string(),
                    input: "Entity groups from context".to_string(),
                    output: format!("Found {} subjects, {} objects", subjects.len(), objects.len()),
                    confidence: 0.7,
                },
                ReasoningStep {
                    step_type: ReasoningStepType::PropertyMapping,
                    description: "Inferred predicates from common vocabulary patterns".to_string(),
                    input: "Common property keywords".to_string(),
                    output: format!("Detected {} predicates", predicates.len()),
                    confidence: 0.6,
                },
                ReasoningStep {
                    step_type: ReasoningStepType::QueryConstruction,
                    description: "Generated SPARQL based on linguistic rules".to_string(),
                    input: "Combined patterns and entities".to_string(),
                    output: "SPARQL query structure".to_string(),
                    confidence,
                },
            ],
            parameter_mapping: parameters.clone(),
            alternatives: vec![
                "Template-based generation for better structure".to_string(),
                "LLM-based generation for complex queries".to_string(),
            ],
        };

        Ok(SPARQLGenerationResult {
            query: final_query,
            confidence,
            generation_method: GenerationMethod::RuleBased,
            parameters,
            explanation: Some(explanation),
            validation_result: ValidationResult {
                is_valid: true, // Assume generated queries are syntactically valid
                syntax_errors: Vec::new(),
                semantic_warnings: if confidence < 0.5 { 
                    vec![SemanticWarning {
                        message: "Low confidence rule-based generation".to_string(),
                        warning_type: SemanticWarningType::PerformanceIssue,
                        severity: WarningSeverity::Medium,
                    }] 
                } else { 
                    Vec::new() 
                },
                schema_issues: Vec::new(),
                suggestions: vec![
                    "Consider using template-based or LLM-based generation for better accuracy".to_string(),
                ],
            },
            optimization_hints: hints,
            metadata,
        })
    }

    fn select_template(&self, query_context: &QueryContext) -> Result<&SPARQLTemplate> {
        let query_text = query_context
            .conversation_history
            .iter()
            .rev()
            .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
            .map(|msg| msg.content.as_str())
            .unwrap_or("Unknown query");
        let query_lower = query_text.to_lowercase();

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
        let query_text = query_context
            .conversation_history
            .iter()
            .rev()
            .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
            .map(|msg| msg.content.as_str())
            .unwrap_or("Unknown query");

        for param in &template.parameters {
            if let Some(ref pattern) = param.extraction_pattern {
                if let Ok(regex) = Regex::new(pattern) {
                    if let Some(captures) = regex.captures(query_text) {
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
        let query_text = query_context
            .conversation_history
            .iter()
            .rev()
            .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
            .map(|msg| msg.content.as_str())
            .unwrap_or("Unknown query");

        let mut reasoning_steps = Vec::new();
        let mut alternatives = Vec::new();

        // Step 1: Input Analysis
        reasoning_steps.push(ReasoningStep {
            step_type: ReasoningStepType::EntityExtraction,
            description: "Analyzed natural language query".to_string(),
            input: query_text.to_string(),
            output: format!("Identified intent: {:?}", query_context.query_intent),
            confidence: 0.9,
        });

        // Step 2: Method Selection
        let method_description = match result.generation_method {
            GenerationMethod::Template(ref template_name) => {
                format!(
                    "Selected template-based generation using template: {template_name}"
                )
            }
            GenerationMethod::LLM(ref model_name) => {
                format!("Selected LLM-based generation using model: {model_name}")
            }
            GenerationMethod::Hybrid => {
                "Selected hybrid approach combining template and LLM generation".to_string()
            }
            GenerationMethod::RuleBased => "Selected rule-based generation approach".to_string(),
        };

        reasoning_steps.push(ReasoningStep {
            step_type: ReasoningStepType::TemplateSelection,
            description: method_description,
            input: "Query analysis results".to_string(),
            output: format!("Generation method: {:?}", result.generation_method),
            confidence: result.confidence,
        });

        // Step 3: Parameter Extraction
        if !result.parameters.is_empty() {
            let parameters_description = result
                .parameters
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join(", ");

            reasoning_steps.push(ReasoningStep {
                step_type: ReasoningStepType::ParameterFilling,
                description: "Extracted parameters from natural language".to_string(),
                input: query_text.to_string(),
                output: parameters_description,
                confidence: 0.8,
            });
        }

        // Step 4: Query Construction
        reasoning_steps.push(ReasoningStep {
            step_type: ReasoningStepType::QueryConstruction,
            description: "Constructed SPARQL query from parameters".to_string(),
            input: "Template and extracted parameters".to_string(),
            output: result.query.clone(),
            confidence: result.confidence,
        });

        // Step 5: Validation
        if result.validation_result.is_valid {
            reasoning_steps.push(ReasoningStep {
                step_type: ReasoningStepType::Validation,
                description: "Query validated successfully".to_string(),
                input: result.query.clone(),
                output: "Query is syntactically and semantically valid".to_string(),
                confidence: 1.0,
            });
        } else {
            reasoning_steps.push(ReasoningStep {
                step_type: ReasoningStepType::Validation,
                description: "Query validation found issues".to_string(),
                input: result.query.clone(),
                output: format!(
                    "Found {} errors and {} warnings",
                    result.validation_result.syntax_errors.len(),
                    result.validation_result.semantic_warnings.len()
                ),
                confidence: 0.5,
            });
        }

        // Step 6: Optimization
        if !result.optimization_hints.is_empty() {
            reasoning_steps.push(ReasoningStep {
                step_type: ReasoningStepType::Optimization,
                description: "Applied query optimizations".to_string(),
                input: "Original query".to_string(),
                output: format!(
                    "Applied {} optimization hints",
                    result.optimization_hints.len()
                ),
                confidence: 0.9,
            });
        }

        // Generate alternative explanations
        alternatives.push("Could have used different parameter extraction patterns".to_string());
        alternatives
            .push("Could have selected a different template or generation method".to_string());

        if result.confidence < 0.8 {
            alternatives.push("Consider rephrasing the query for better accuracy".to_string());
        }

        // Generate natural language explanation
        let natural_language = self
            .generate_natural_language_explanation(query_text, result, &reasoning_steps)
            .await?;

        Ok(QueryExplanation {
            natural_language,
            reasoning_steps,
            parameter_mapping: result.parameters.clone(),
            alternatives,
        })
    }

    /// Generate natural language explanation
    async fn generate_natural_language_explanation(
        &self,
        query_text: &str,
        result: &SPARQLGenerationResult,
        reasoning_steps: &[ReasoningStep],
    ) -> Result<String> {
        let mut explanation = String::new();

        explanation.push_str(&format!("For your query '{query_text}', I:\n\n"));

        for (i, step) in reasoning_steps.iter().enumerate() {
            explanation.push_str(&format!("{}. {}\n", i + 1, step.description));

            if step.confidence < 0.7 {
                explanation.push_str(&format!(
                    "   (Note: This step has lower confidence: {:.1}%)\n",
                    step.confidence * 100.0
                ));
            }
        }

        explanation.push_str(&format!(
            "\nThe resulting SPARQL query has a confidence score of {:.1}%.\n",
            result.confidence * 100.0
        ));

        if result.confidence < 0.7 {
            explanation.push_str("You may want to rephrase your query for better results.\n");
        }

        if !result.validation_result.is_valid {
            explanation.push_str(
                "Note: The generated query has some validation issues that may affect execution.\n",
            );
        }

        Ok(explanation)
    }

    /// Calculate confidence for template-based generation
    fn calculate_template_confidence(
        &self,
        template: &SPARQLTemplate,
        parameters: &HashMap<String, String>,
        query_context: &QueryContext,
    ) -> Result<f32> {
        let mut confidence_factors = Vec::new();

        // Factor 1: Template specificity (0.6-1.0)
        let specificity_score = if !template.parameters.is_empty() {
            0.6 + (template.parameters.len() as f32 * 0.1).min(0.4)
        } else {
            0.6
        };
        confidence_factors.push(specificity_score);

        // Factor 2: Parameter extraction quality (0.5-1.0)
        let mut param_quality: f32 = 1.0;
        for param in &template.parameters {
            if let Some(value) = parameters.get(&param.name) {
                if value.is_empty() {
                    param_quality *= 0.7;
                } else if value.len() < 2 {
                    param_quality *= 0.8;
                } else if value.contains("unknown") || value.contains("undefined") {
                    param_quality *= 0.6;
                }
            } else {
                param_quality *= 0.5;
            }
        }
        confidence_factors.push(param_quality.max(0.5));

        // Factor 3: Intent pattern match quality (0.4-1.0)
        let intent_match_score = if template.intent_patterns.is_empty() {
            0.7
        } else {
            let query_text = query_context
                .conversation_history
                .iter()
                .rev()
                .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
                .map(|msg| msg.content.as_str())
                .unwrap_or("");
            let mut best_match: f32 = 0.0;

            for pattern in &template.intent_patterns {
                let pattern_lower = pattern.to_lowercase();
                let pattern_words: std::collections::HashSet<_> =
                    pattern_lower.split_whitespace().collect();
                let query_lower = query_text.to_lowercase();
                let query_words: std::collections::HashSet<_> =
                    query_lower.split_whitespace().collect();

                let intersection_size = pattern_words.intersection(&query_words).count();
                let union_size = pattern_words.union(&query_words).count();

                if union_size > 0 {
                    let jaccard_similarity = intersection_size as f32 / union_size as f32;
                    best_match = best_match.max(jaccard_similarity);
                }
            }

            0.4 + best_match * 0.6
        };
        confidence_factors.push(intent_match_score);

        let base_confidence =
            confidence_factors.iter().sum::<f32>() / confidence_factors.len() as f32;
        let template_bonus = if !template.examples.is_empty() {
            0.05
        } else {
            0.0
        };
        let final_confidence = (base_confidence + template_bonus).min(1.0).max(0.1);

        debug!(
            "Template confidence for '{}': {:.3}",
            template.name, final_confidence
        );

        Ok(final_confidence)
    }

    /// Calculate confidence for LLM-based generation
    fn calculate_llm_confidence(
        &mut self,
        response: &crate::llm::LLMResponse,
        sparql_query: &str,
        _query_context: &QueryContext,
    ) -> Result<f32> {
        let mut confidence_factors = Vec::new();

        // Factor 1: LLM response confidence if available (0.3-1.0)
        let llm_confidence: f32 = 0.7; // Default confidence since field is not available
        confidence_factors.push(llm_confidence.max(0.3));

        // Factor 2: SPARQL syntax validity (0.2-1.0)
        let syntax_score = if self
            .validator
            .validate(sparql_query)
            .map(|r| r.is_valid)
            .unwrap_or(false)
        {
            1.0
        } else {
            0.2
        };
        confidence_factors.push(syntax_score);

        // Factor 3: Query completeness (0.4-1.0)
        let completeness_score = if sparql_query.trim().is_empty() {
            0.4
        } else if sparql_query.to_uppercase().contains("SELECT")
            && sparql_query.to_uppercase().contains("WHERE")
        {
            0.9
        } else if sparql_query.to_uppercase().contains("CONSTRUCT")
            && sparql_query.to_uppercase().contains("WHERE")
        {
            0.9
        } else if sparql_query.to_uppercase().contains("ASK") {
            0.8
        } else {
            0.6
        };
        confidence_factors.push(completeness_score);

        let weights = [0.25, 0.35, 0.40];
        let weighted_sum: f32 = confidence_factors
            .iter()
            .zip(weights.iter())
            .map(|(factor, weight)| factor * weight)
            .sum();

        let final_confidence = weighted_sum.min(1.0).max(0.1);

        debug!("LLM confidence for query: {:.3}", final_confidence);

        Ok(final_confidence)
    }
}

/// SPARQL validation component with comprehensive checks
pub struct SPARQLValidator {
    syntax_patterns: HashMap<String, Regex>,
    common_prefixes: HashMap<String, String>,
}

impl Default for SPARQLValidator {
    fn default() -> Self {
        Self::new()
    }
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
        let schema_issues = Vec::new();
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
                    "Unbalanced braces: {open_braces} open, {close_braces} close"
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
                    message: format!("Undeclared prefix: {prefix}"),
                    position: None,
                    error_type: SyntaxErrorType::UnknownPrefix,
                    suggestion: Some(format!("Declare prefix {prefix} or use full IRI")),
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

impl Default for SPARQLOptimizer {
    fn default() -> Self {
        Self::new()
    }
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
