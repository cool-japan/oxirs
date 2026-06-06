//! Natural Language to SPARQL Translation System
//!
//! Provides advanced NL2SPARQL capabilities with template-based generation,
//! semantic parsing, query optimization, and comprehensive validation.

pub mod builder;
pub mod context_aware;
pub mod optimizer;
pub mod semantic_understanding;
pub mod types;
pub mod validator;

pub use optimizer::SPARQLOptimizer;
pub use semantic_understanding::*;
pub use types::*;
pub use validator::SPARQLValidator;

use anyhow::{anyhow, Result};
use handlebars::Handlebars;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use tracing::info;

use crate::llm::LLMManager;
use crate::schema_introspection::{DiscoveredSchema, SchemaIntrospector};
use crate::QueryContext;
use oxirs_core::Store;

pub struct NL2SPARQLSystem {
    pub(crate) config: NL2SPARQLConfig,
    pub(crate) llm_manager: Option<Arc<TokioMutex<LLMManager>>>,
    pub(crate) template_engine: Handlebars<'static>,
    pub(crate) templates: HashMap<String, SPARQLTemplate>,
    pub(crate) validator: SPARQLValidator,
    pub(crate) optimizer: SPARQLOptimizer,
    pub(crate) store: Option<Arc<dyn Store>>,
    /// Cached discovered schema for schema-aware query generation
    pub(crate) schema: Option<DiscoveredSchema>,
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
            schema: None,
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
            schema: None,
        };

        system.initialize_templates()?;
        Ok(system)
    }

    /// Discover and cache schema from the store for schema-aware query generation
    pub async fn discover_schema(&mut self) -> Result<()> {
        if let Some(store) = &self.store {
            info!("Discovering schema for NL2SPARQL enhancement");
            let introspector = SchemaIntrospector::new(store.clone());
            let schema = introspector.discover_schema().await?;
            info!("{}", schema.summary());
            self.schema = Some(schema);
            Ok(())
        } else {
            Err(anyhow!("Store required for schema discovery"))
        }
    }

    /// Get the discovered schema if available
    pub fn get_schema(&self) -> Option<&DiscoveredSchema> {
        self.schema.as_ref()
    }

    /// Set schema manually (for testing or external schema sources)
    pub fn set_schema(&mut self, schema: DiscoveredSchema) {
        self.schema = Some(schema);
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

        result.validation_result = self.validator.validate(&result.query)?;

        if self.config.optimization.enable_optimization {
            let (optimized_query, hints) = self.optimizer.optimize(&result.query)?;
            result.query = optimized_query;
            result.optimization_hints = hints;
        }

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

                    let result_count = match results.results() {
                        oxirs_core::rdf_store::QueryResults::Bindings(result_bindings) => {
                            for binding in result_bindings {
                                let mut string_binding = HashMap::new();
                                for var in binding.variables() {
                                    if let Some(term) = binding.get(var) {
                                        string_binding.insert(var.clone(), format!("{term}"));
                                    }
                                }
                                bindings.push(string_binding);
                            }
                            bindings.len()
                        }
                        oxirs_core::rdf_store::QueryResults::Boolean(answer) => {
                            let mut ask_binding = HashMap::new();
                            ask_binding.insert("result".to_string(), answer.to_string());
                            bindings.push(ask_binding);
                            1
                        }
                        oxirs_core::rdf_store::QueryResults::Graph(quads) => {
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
                            quads.len()
                        }
                    };

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
                    tracing::error!("SPARQL query execution failed: {}", e);
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
