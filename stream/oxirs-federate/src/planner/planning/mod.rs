//! Federated Query Planner Module
//!
//! This module provides comprehensive query planning and optimization for federated
//! GraphQL queries across multiple services. It includes:
//!
//! - Type definitions and data structures
//! - Entity resolution and dependency tracking
//! - Schema composition and validation
//! - Query analysis and parsing
//! - Schema introspection and capability discovery
//! - Performance optimization and reoptimization
//! - Core planner implementation

pub mod entity_resolution;
pub mod performance_optimizer;
pub mod query_analysis;
pub mod schema_composition;
pub mod schema_introspection;
pub mod types;

// Re-export commonly used types and functions
pub use entity_resolution::EntityResolver;
pub use performance_optimizer::PerformanceOptimizer;
pub use query_analysis::{QueryAnalyzer, QueryComplexity, QueryInfo};
pub use schema_composition::SchemaComposer;
pub use schema_introspection::SchemaIntrospector;
pub use types::*;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{
    metadata::ExtendedServiceMetadata,
    query_decomposition::{DecomposerConfig, DecompositionResult, QueryDecomposer},
    service::ServiceMetadata,
    service_optimizer::{ServiceOptimizer, ServiceOptimizerConfig},
    FederatedService, ServiceCapability, ServiceRegistry, StepResult,
};

/// Main federated query planner
#[derive(Debug)]
pub struct FederatedQueryPlanner {
    /// Query analyzer for parsing and decomposition
    query_analyzer: QueryAnalyzer,
    /// Schema composer for federation
    schema_composer: SchemaComposer,
    /// Entity resolver for federation
    entity_resolver: EntityResolver,
    /// Schema introspector for capability discovery
    schema_introspector: SchemaIntrospector,
    /// Performance optimizer
    performance_optimizer: PerformanceOptimizer,
    /// Configuration
    config: PlannerConfig,
    /// Historical performance data
    performance_history: Arc<RwLock<HistoricalPerformance>>,
}

impl FederatedQueryPlanner {
    /// Create a new federated query planner
    pub fn new() -> Self {
        Self {
            query_analyzer: QueryAnalyzer,
            schema_composer: SchemaComposer,
            entity_resolver: EntityResolver,
            schema_introspector: SchemaIntrospector,
            performance_optimizer: PerformanceOptimizer::new(),
            config: PlannerConfig::default(),
            performance_history: Arc::new(RwLock::new(HistoricalPerformance {
                query_patterns: HashMap::new(),
                service_performance: HashMap::new(),
                join_performance: HashMap::new(),
                avg_response_times: HashMap::new(),
            })),
        }
    }

    /// Create a new federated query planner with custom configuration
    pub fn with_config(config: PlannerConfig) -> Self {
        Self {
            query_analyzer: QueryAnalyzer,
            schema_composer: SchemaComposer,
            entity_resolver: EntityResolver,
            schema_introspector: SchemaIntrospector,
            performance_optimizer: PerformanceOptimizer::with_config(
                config.optimization_config.clone(),
            ),
            config,
            performance_history: Arc::new(RwLock::new(HistoricalPerformance {
                query_patterns: HashMap::new(),
                service_performance: HashMap::new(),
                join_performance: HashMap::new(),
                avg_response_times: HashMap::new(),
            })),
        }
    }

    /// Plan a federated query execution
    pub async fn plan_federated_query(
        &self,
        query: &str,
        variables: Option<serde_json::Value>,
        context: &ExecutionContext,
        service_registry: &ServiceRegistry,
    ) -> Result<ExecutionPlan> {
        let start_time = Instant::now();
        info!("Planning federated query: {}", context.query_id);

        // Parse and analyze the query
        let parsed_query = QueryAnalyzer::parse_graphql_query(query)?;
        let query_info = QueryAnalyzer::extract_query_info(&parsed_query);

        // Get planning recommendations based on performance history
        let recommendations = self
            .performance_optimizer
            .get_planning_recommendations(&query_info);

        // Create unified schema from all services
        let unified_schema = self
            .create_unified_schema_from_registry(service_registry)
            .await?;

        // Validate query against schema
        let validation_errors =
            QueryAnalyzer::validate_query_against_schema(&parsed_query, &unified_schema)?;
        if !validation_errors.is_empty() {
            return Err(anyhow!(
                "Query validation failed: {}",
                validation_errors.join(", ")
            ));
        }

        // Check if federation is required
        if !QueryAnalyzer::requires_federation(&parsed_query, &unified_schema) {
            // Single service query - create simple plan
            return self
                .create_single_service_plan(query, variables, context, service_registry)
                .await;
        }

        // Decompose query for federation
        let service_queries = QueryAnalyzer::decompose_query(query, &unified_schema).await?;

        // Extract entity references for resolution
        let entity_refs = EntityResolver::extract_entity_references(query)?;

        // Build entity resolution plan if needed
        let entity_plan = if !entity_refs.is_empty() {
            Some(EntityResolver::build_entity_resolution_plan(&entity_refs).await?)
        } else {
            None
        };

        // Create execution steps
        let mut steps = Vec::new();

        // Add service query steps
        for (idx, service_query) in service_queries.iter().enumerate() {
            let step = ExecutionStep {
                step_id: format!("service_query_{}", idx),
                step_type: StepType::ServiceQuery,
                service_id: Some(service_query.service_id.clone()),
                query_fragment: service_query.query.clone(),
                dependencies: Vec::new(),
                estimated_cost: self
                    .estimate_step_cost(&service_query.query, &service_query.service_id),
                timeout: recommendations.suggested_timeout,
                retry_config: self.config.default_retry_config.as_ref().map(|rc| {
                    crate::planner::planning::types::RetryConfig {
                        max_attempts: rc.max_attempts as usize,
                        initial_delay: std::time::Duration::from_millis(rc.initial_delay_ms),
                        max_delay: std::time::Duration::from_millis(rc.max_delay_ms),
                        backoff_multiplier: rc.backoff_multiplier,
                    }
                }),
            };
            steps.push(step);
        }

        // Add entity resolution steps if needed
        if let Some(entity_plan) = entity_plan {
            for (idx, entity_step) in entity_plan.steps.iter().enumerate() {
                let step = ExecutionStep {
                    step_id: format!("entity_resolution_{}", idx),
                    step_type: StepType::EntityResolution,
                    service_id: Some(entity_step.service_name.clone()),
                    query_fragment: entity_step.query.clone(),
                    dependencies: entity_step.depends_on.clone(),
                    estimated_cost: 10.0, // Base cost for entity resolution
                    timeout: recommendations.suggested_timeout,
                    retry_config: self.config.default_retry_config.as_ref().map(|rc| {
                        crate::planner::planning::types::RetryConfig {
                            max_attempts: rc.max_attempts as usize,
                            initial_delay: std::time::Duration::from_millis(rc.initial_delay_ms),
                            max_delay: std::time::Duration::from_millis(rc.max_delay_ms),
                            backoff_multiplier: rc.backoff_multiplier,
                        }
                    }),
                };
                steps.push(step);
            }
        }

        // Add result stitching step
        if steps.len() > 1 {
            let stitch_step = ExecutionStep {
                step_id: "result_stitching".to_string(),
                step_type: StepType::ResultStitching,
                service_id: None,
                query_fragment: "".to_string(),
                dependencies: steps.iter().map(|s| s.step_id.clone()).collect(),
                estimated_cost: 5.0,
                timeout: std::time::Duration::from_secs(10),
                retry_config: None,
            };
            steps.push(stitch_step);
        }

        let planning_time = start_time.elapsed();

        let estimated_total_cost: f64 = steps.iter().map(|s| s.estimated_cost).sum();

        Ok(ExecutionPlan {
            query_id: context.query_id.clone(),
            steps,
            estimated_total_cost,
            max_parallelism: if recommendations.preferred_execution_strategy
                == performance_optimizer::ExecutionStrategy::Parallel
            {
                self.config.max_parallel_steps
            } else {
                1
            },
            planning_time,
            cache_key: self.generate_cache_key(query, &variables),
            metadata: self.extract_planning_metadata(&parsed_query, &recommendations),
            parallelizable_steps: vec![], // TODO: Implement parallelization analysis
        })
    }

    /// Create a simple plan for single-service queries
    async fn create_single_service_plan(
        &self,
        query: &str,
        variables: Option<serde_json::Value>,
        context: &ExecutionContext,
        service_registry: &ServiceRegistry,
    ) -> Result<ExecutionPlan> {
        // Find the appropriate service (simplified - would use more sophisticated logic)
        let service_id = service_registry
            .get_all_services()
            .next()
            .map(|s| s.id.clone())
            .unwrap_or_else(|| "default".to_string());

        let step = ExecutionStep {
            step_id: "single_service_query".to_string(),
            step_type: StepType::ServiceQuery,
            service_id: Some(service_id.clone()),
            query_fragment: query.to_string(),
            dependencies: Vec::new(),
            estimated_cost: self.estimate_step_cost(query, &service_id),
            timeout: std::time::Duration::from_secs(30),
            retry_config: self.config.default_retry_config.as_ref().map(|rc| {
                crate::planner::planning::types::RetryConfig {
                    max_attempts: rc.max_attempts as usize,
                    initial_delay: std::time::Duration::from_millis(rc.initial_delay_ms),
                    max_delay: std::time::Duration::from_millis(rc.max_delay_ms),
                    backoff_multiplier: rc.backoff_multiplier,
                }
            }),
        };

        Ok(ExecutionPlan {
            query_id: context.query_id.clone(),
            steps: vec![step],
            estimated_total_cost: self.estimate_step_cost(query, &service_id),
            max_parallelism: 1,
            planning_time: std::time::Duration::from_millis(1),
            cache_key: self.generate_cache_key(query, &variables),
            metadata: HashMap::new(),
            parallelizable_steps: Vec::new(),
        })
    }

    /// Create unified schema from service registry
    async fn create_unified_schema_from_registry(
        &self,
        service_registry: &ServiceRegistry,
    ) -> Result<UnifiedSchema> {
        let mut unified = UnifiedSchema {
            types: HashMap::new(),
            queries: HashMap::new(),
            mutations: HashMap::new(),
            subscriptions: HashMap::new(),
            directives: HashMap::new(),
            schema_mapping: HashMap::new(),
        };

        for service in service_registry.get_all_services() {
            // In a real implementation, would fetch actual schema from service
            let mock_schema = self.create_mock_schema_for_service(&service.id);
            SchemaComposer::merge_schema_into_unified(
                &mut unified,
                &service.id,
                &mock_schema,
                &GraphQLFederationConfig::default(),
            )?;
        }

        SchemaComposer::validate_unified_schema(&unified)?;
        Ok(unified)
    }

    /// Create a mock schema for a service (placeholder)
    fn create_mock_schema_for_service(&self, service_id: &str) -> FederatedSchema {
        FederatedSchema {
            service_id: service_id.to_string(),
            types: HashMap::new(),
            queries: HashMap::new(),
            mutations: HashMap::new(),
            subscriptions: HashMap::new(),
            directives: HashMap::new(),
        }
    }

    /// Estimate cost for an execution step
    fn estimate_step_cost(&self, query: &str, service_id: &str) -> f64 {
        // Simplified cost estimation based on query length and service
        let base_cost = query.len() as f64 * 0.1;
        let service_multiplier = if service_id.contains("slow") {
            2.0
        } else {
            1.0
        };
        base_cost * service_multiplier
    }

    /// Generate cache key for the query
    fn generate_cache_key(
        &self,
        query: &str,
        variables: &Option<serde_json::Value>,
    ) -> Option<String> {
        if self.config.enable_caching {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(query, &mut hasher);
            if let Some(vars) = variables {
                std::hash::Hash::hash(&vars.to_string(), &mut hasher);
            }
            Some(format!("query_{:x}", std::hash::Hasher::finish(&hasher)))
        } else {
            None
        }
    }

    /// Extract planning metadata
    fn extract_planning_metadata(
        &self,
        parsed_query: &ParsedQuery,
        recommendations: &performance_optimizer::PlanningRecommendations,
    ) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert(
            "operation_type".to_string(),
            format!("{:?}", parsed_query.operation_type),
        );
        metadata.insert(
            "field_count".to_string(),
            parsed_query.selection_set.len().to_string(),
        );
        metadata.insert(
            "has_variables".to_string(),
            (!parsed_query.variables.is_empty()).to_string(),
        );
        metadata.insert(
            "execution_strategy".to_string(),
            format!("{:?}", recommendations.preferred_execution_strategy),
        );
        metadata.insert(
            "caching_enabled".to_string(),
            recommendations.enable_caching.to_string(),
        );
        metadata
    }

    /// Analyze query performance and suggest reoptimization
    pub async fn analyze_performance(
        &self,
        execution_metrics: &performance_optimizer::ExecutionMetrics,
        context: &ExecutionContext,
    ) -> Result<ReoptimizationAnalysis> {
        self.performance_optimizer
            .analyze_performance(execution_metrics, context)
    }

    /// Update performance history
    pub async fn update_performance_history(
        &mut self,
        metrics: &performance_optimizer::ExecutionMetrics,
        context: &ExecutionContext,
    ) {
        self.performance_optimizer
            .update_performance_history(metrics, context);
    }

    /// Get historical performance data
    pub async fn get_performance_history(&self) -> HistoricalPerformance {
        self.performance_history.read().await.clone()
    }
}

impl Default for FederatedQueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the federated query planner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerConfig {
    pub max_parallel_steps: usize,
    pub enable_caching: bool,
    pub cache_ttl_seconds: u64,
    pub default_timeout_seconds: u64,
    pub max_query_complexity: f64,
    pub enable_performance_analysis: bool,
    pub optimization_config: performance_optimizer::OptimizationConfig,
    pub default_retry_config: Option<RetryConfig>,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_parallel_steps: 10,
            enable_caching: true,
            cache_ttl_seconds: 300, // 5 minutes
            default_timeout_seconds: 30,
            max_query_complexity: 1000.0,
            enable_performance_analysis: true,
            optimization_config: performance_optimizer::OptimizationConfig::default(),
            default_retry_config: Some(RetryConfig {
                max_attempts: 3,
                initial_delay_ms: 100,
                max_delay_ms: 5000,
                backoff_multiplier: 2.0,
            }),
        }
    }
}

/// Retry configuration for execution steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}
