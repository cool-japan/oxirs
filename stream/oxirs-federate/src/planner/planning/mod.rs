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
pub use query_analysis::{QueryAnalyzer, QueryComplexity};
pub use schema_composition::SchemaComposer;
pub use schema_introspection::SchemaIntrospector;
pub use types::*;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::info;

use crate::service_registry::ServiceRegistry;

/// Main federated query planner
#[derive(Debug)]
pub struct FederatedQueryPlanner {
    /// Query analyzer for parsing and decomposition
    #[allow(dead_code)]
    query_analyzer: QueryAnalyzer,
    /// Schema composer for federation
    #[allow(dead_code)]
    schema_composer: SchemaComposer,
    /// Entity resolver for federation
    #[allow(dead_code)]
    entity_resolver: EntityResolver,
    /// Schema introspector for capability discovery
    #[allow(dead_code)]
    schema_introspector: SchemaIntrospector,
    /// Performance optimizer
    performance_optimizer: PerformanceOptimizer,
    /// Configuration
    config: PlannerConfig,
    /// Historical performance data
    #[allow(dead_code)]
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
            // Retrieve service details directly from registry endpoints
            let (service_url, auth_config) = service_registry
                .get_sparql_endpoints()
                .into_iter()
                .find(|ep| ep.id == service_query.service_id)
                .map(|ep| (Some(ep.url.to_string()), ep.auth.clone()))
                .or_else(|| {
                    // Check GraphQL services if not found in SPARQL endpoints
                    service_registry
                        .get_graphql_services()
                        .into_iter()
                        .find(|svc| svc.id == service_query.service_id)
                        .map(|svc| (Some(svc.url.to_string()), svc.auth.clone()))
                })
                .unwrap_or((None, None));

            let step = ExecutionStep {
                step_id: format!("service_query_{idx}"),
                step_type: StepType::ServiceQuery,
                service_id: Some(service_query.service_id.clone()),
                service_url,
                auth_config,
                query_fragment: service_query.query.clone(),
                dependencies: Vec::new(),
                estimated_cost: self
                    .estimate_step_cost(&service_query.query, &service_query.service_id),
                timeout: recommendations.suggested_timeout,
                retry_config: self.config.default_retry_config.as_ref().map(|rc| {
                    crate::planner::planning::types::RetryConfig {
                        max_attempts: rc.max_attempts,
                        initial_delay: rc.initial_delay,
                        max_delay: rc.max_delay,
                        backoff_multiplier: rc.backoff_multiplier,
                    }
                }),
            };
            steps.push(step);
        }

        // Add entity resolution steps if needed
        if let Some(entity_plan) = entity_plan {
            for (idx, entity_step) in entity_plan.steps.iter().enumerate() {
                // Retrieve service details directly from registry endpoints
                let (service_url, auth_config) = service_registry
                    .get_sparql_endpoints()
                    .into_iter()
                    .find(|ep| ep.id == entity_step.service_name)
                    .map(|ep| (Some(ep.url.to_string()), ep.auth.clone()))
                    .or_else(|| {
                        service_registry
                            .get_graphql_services()
                            .into_iter()
                            .find(|svc| svc.id == entity_step.service_name)
                            .map(|svc| (Some(svc.url.to_string()), svc.auth.clone()))
                    })
                    .unwrap_or((None, None));

                let step = ExecutionStep {
                    step_id: format!("entity_resolution_{idx}"),
                    step_type: StepType::EntityResolution,
                    service_id: Some(entity_step.service_name.clone()),
                    service_url,
                    auth_config,
                    query_fragment: entity_step.query.clone(),
                    dependencies: entity_step.depends_on.clone(),
                    estimated_cost: 10.0, // Base cost for entity resolution
                    timeout: recommendations.suggested_timeout,
                    retry_config: self.config.default_retry_config.as_ref().map(|rc| {
                        crate::planner::planning::types::RetryConfig {
                            max_attempts: rc.max_attempts,
                            initial_delay: rc.initial_delay,
                            max_delay: rc.max_delay,
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
                service_url: None,
                auth_config: None,
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

        // Analyze parallelization opportunities before moving steps
        let parallelizable_steps = self.analyze_parallelizable_steps(&steps);

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
            parallelizable_steps,
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
            .into_iter()
            .next()
            .map(|s| s.id.clone())
            .unwrap_or_else(|| "default".to_string());

        let step = ExecutionStep {
            step_id: "single_service_query".to_string(),
            step_type: StepType::ServiceQuery,
            service_id: Some(service_id.clone()),
            service_url: None,
            auth_config: None,
            query_fragment: query.to_string(),
            dependencies: Vec::new(),
            estimated_cost: self.estimate_step_cost(query, &service_id),
            timeout: std::time::Duration::from_secs(30),
            retry_config: self.config.default_retry_config.as_ref().map(|rc| {
                crate::planner::planning::types::RetryConfig {
                    max_attempts: rc.max_attempts,
                    initial_delay: rc.initial_delay,
                    max_delay: rc.max_delay,
                    backoff_multiplier: rc.backoff_multiplier,
                }
            }),
        };

        let steps = vec![step];
        Ok(ExecutionPlan {
            query_id: context.query_id.clone(),
            steps: steps.clone(),
            estimated_total_cost: self.estimate_step_cost(query, &service_id),
            max_parallelism: 1,
            planning_time: std::time::Duration::from_millis(1),
            cache_key: self.generate_cache_key(query, &variables),
            metadata: HashMap::new(),
            parallelizable_steps: self.analyze_parallelizable_steps(&steps),
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

    /// Analyze step dependencies and identify parallelizable steps
    /// Returns a Vec<Vec<String>> where each inner Vec contains step IDs
    /// that can be executed in parallel (same level in dependency graph)
    fn analyze_parallelizable_steps(&self, steps: &[ExecutionStep]) -> Vec<Vec<String>> {
        if steps.is_empty() {
            return Vec::new();
        }

        // Build dependency graph
        let mut dependency_graph: HashMap<String, Vec<String>> = HashMap::new();
        let mut reverse_deps: HashMap<String, Vec<String>> = HashMap::new();
        let mut all_step_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

        for step in steps {
            all_step_ids.insert(step.step_id.clone());
            dependency_graph.insert(step.step_id.clone(), step.dependencies.clone());

            // Build reverse dependency map (which steps depend on this step)
            for dep in &step.dependencies {
                reverse_deps
                    .entry(dep.clone())
                    .or_default()
                    .push(step.step_id.clone());
            }
        }

        // Topological sort with level-wise grouping for parallelization
        let mut result: Vec<Vec<String>> = Vec::new();
        let mut processed: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut in_degree: HashMap<String, usize> = HashMap::new();

        // Calculate in-degree for each step
        for step in steps {
            in_degree.insert(step.step_id.clone(), step.dependencies.len());
        }

        // Process steps level by level
        while processed.len() < steps.len() {
            // Find all steps with no remaining dependencies (in-degree = 0)
            let mut current_level: Vec<String> = Vec::new();

            for step_id in &all_step_ids {
                if !processed.contains(step_id) && in_degree.get(step_id).copied().unwrap_or(0) == 0
                {
                    current_level.push(step_id.clone());
                }
            }

            if current_level.is_empty() {
                // Circular dependency or error - break to avoid infinite loop
                break;
            }

            // Sort for deterministic output
            current_level.sort();

            // Add this level to results
            result.push(current_level.clone());

            // Mark these steps as processed
            for step_id in &current_level {
                processed.insert(step_id.clone());

                // Decrease in-degree of dependent steps
                if let Some(dependents) = reverse_deps.get(step_id) {
                    for dependent in dependents {
                        if let Some(degree) = in_degree.get_mut(dependent) {
                            *degree = degree.saturating_sub(1);
                        }
                    }
                }
            }
        }

        result
    }

    /// Update performance history
    pub async fn update_performance_history(
        &mut self,
        metrics: &performance_optimizer::ExecutionMetrics,
        context: &ExecutionContext,
    ) {
        self.performance_optimizer
            .update_performance_history(metrics, context)
            .await;
    }

    /// Get historical performance data
    pub async fn get_performance_history(&self) -> HistoricalPerformance {
        self.performance_history.read().await.clone()
    }

    /// Analyze a SPARQL query and extract query information
    pub async fn analyze_sparql(&self, query: &str) -> Result<QueryInfo> {
        // Parse SPARQL query and extract information
        let query_upper = query.to_uppercase();

        // Determine query type based on the first keyword
        let query_type = if query_upper.trim_start().starts_with("SELECT") {
            QueryType::Select
        } else if query_upper.trim_start().starts_with("CONSTRUCT") {
            QueryType::Construct
        } else if query_upper.trim_start().starts_with("ASK") {
            QueryType::Ask
        } else if query_upper.trim_start().starts_with("DESCRIBE") {
            QueryType::Describe
        } else if query_upper.trim_start().starts_with("INSERT")
            || query_upper.trim_start().starts_with("DELETE")
        {
            QueryType::Update
        } else {
            QueryType::Sparql // fallback for other cases
        };

        // Extract variables (simplified - look for ?variable patterns)
        let mut variables = std::collections::HashSet::new();
        let var_regex = regex::Regex::new(r"\?[a-zA-Z_][a-zA-Z0-9_]*")?;
        for mat in var_regex.find_iter(query) {
            variables.insert(mat.as_str().to_string());
        }

        // Extract basic triple patterns (simplified - look for WHERE clause content)
        let mut patterns = Vec::new();
        if let Some(where_start) = query_upper.find("WHERE") {
            let where_clause = &query[where_start + 5..];
            if let Some(brace_start) = where_clause.find('{') {
                if let Some(brace_end) = where_clause.rfind('}') {
                    let where_content = &where_clause[brace_start + 1..brace_end];

                    // Extract patterns from both regular WHERE clause and SERVICE blocks
                    self.extract_patterns_recursive(where_content, &mut patterns);
                }
            }
        }

        // Extract filters (look for FILTER expressions)
        let mut filters = Vec::new();
        let filter_regex = regex::Regex::new(r"FILTER\s*\(\s*([^)]+)\s*\)")?;
        let var_regex = regex::Regex::new(r"\?[a-zA-Z_][a-zA-Z0-9_]*")?;
        for cap in filter_regex.captures_iter(query) {
            let filter_expr = cap[1].to_string();

            // Extract variables from the filter expression
            let mut filter_variables = Vec::new();
            for mat in var_regex.find_iter(&filter_expr) {
                filter_variables.push(mat.as_str().to_string());
            }

            filters.push(FilterExpression {
                expression: filter_expr,
                variables: filter_variables,
            });
        }

        Ok(QueryInfo {
            query_type,
            original_query: query.to_string(),
            patterns,
            variables,
            filters,
            complexity: query.len() as u64 / 10, // Simple complexity estimate
            estimated_cost: query.len() as u64,
        })
    }

    /// Recursively extract patterns from WHERE clause content, including SERVICE blocks
    fn extract_patterns_recursive(&self, content: &str, patterns: &mut Vec<TriplePattern>) {
        // Handle SERVICE blocks
        let service_regex =
            regex::Regex::new(r"SERVICE\s+(?:SILENT\s+)?<([^>]+)>\s*\{([^}]+)\}").unwrap();
        let mut remaining_content = content.to_string();

        // Extract patterns from SERVICE blocks
        for captures in service_regex.captures_iter(content) {
            let service_content = &captures[2];
            // Recursively extract patterns from within the SERVICE block
            self.extract_simple_patterns(service_content, patterns);

            // Remove the SERVICE block from remaining content to avoid double-processing
            remaining_content = remaining_content.replace(&captures[0], "");
        }

        // Extract patterns from remaining content (non-SERVICE patterns)
        self.extract_simple_patterns(&remaining_content, patterns);
    }

    /// Extract simple triple patterns from a content string
    fn extract_simple_patterns(&self, content: &str, patterns: &mut Vec<TriplePattern>) {
        // Extract triple patterns by splitting on '.' and parsing each triple
        let statements: Vec<&str> = content.split('.').collect();
        for statement in statements {
            let trimmed = statement.trim();
            if !trimmed.is_empty()
                && !trimmed.starts_with('}')
                && !trimmed.starts_with('{')
                && !trimmed.to_uppercase().starts_with("SERVICE")
                && trimmed.contains('?')
            {
                // Parse the triple pattern (subject predicate object)
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 3 {
                    let subject = parts[0].to_string();
                    let predicate = parts[1].to_string();
                    let object = parts[2..].join(" ").trim_end_matches('.').to_string();

                    patterns.push(TriplePattern {
                        subject: Some(subject),
                        predicate: Some(predicate),
                        object: Some(object),
                        pattern_string: trimmed.to_string(),
                    });
                }
            }
        }
    }

    /// Plan a SPARQL query execution across federated services
    pub async fn plan_sparql(
        &self,
        query_info: &QueryInfo,
        service_registry: &ServiceRegistry,
    ) -> Result<ExecutionPlan> {
        let start_time = Instant::now();
        info!("Planning SPARQL federated query");

        // For SPARQL queries, we skip GraphQL parsing and validation
        // and go directly to execution planning based on the QueryInfo

        let context = ExecutionContext {
            query_id: "sparql_query".to_string(),
            execution_id: "exec_1".to_string(),
            start_time: std::time::Instant::now(),
            timeout: None,
            variables: std::collections::HashMap::new(),
            metadata: std::collections::HashMap::new(),
        };

        // Analyze query patterns to determine required capabilities
        let required_capabilities = self.analyze_query_capabilities(query_info);

        // Select the most appropriate service based on capabilities
        let selected_service_id = self
            .select_service_for_capabilities(&required_capabilities, service_registry)
            .await?;

        // Create execution plan directly from SPARQL query info
        let plan = ExecutionPlan {
            query_id: context.query_id.clone(),
            steps: vec![ExecutionStep {
                step_id: "sparql_step_1".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some(selected_service_id),
                service_url: None,
                auth_config: None,
                query_fragment: query_info.original_query.clone(),
                dependencies: vec![],
                estimated_cost: query_info.estimated_cost as f64,
                timeout: std::time::Duration::from_secs(30),
                retry_config: Some(types::RetryConfig {
                    max_attempts: 3,
                    initial_delay: std::time::Duration::from_millis(100),
                    max_delay: std::time::Duration::from_secs(5),
                    backoff_multiplier: 2.0,
                }),
            }],
            estimated_total_cost: query_info.estimated_cost as f64,
            max_parallelism: 4,
            planning_time: start_time.elapsed(),
            cache_key: None,
            metadata: std::collections::HashMap::new(),
            parallelizable_steps: vec![vec!["sparql_step_1".to_string()]],
        };

        Ok(plan)
    }

    /// Analyze a GraphQL query and extract query information
    pub async fn analyze_graphql(
        &self,
        query: &str,
        _variables: Option<&serde_json::Value>,
    ) -> Result<QueryInfo> {
        // Parse GraphQL query and extract information
        let patterns = Vec::new(); // Simplified - would parse actual GraphQL
        let variables_set = std::collections::HashSet::new(); // Would extract from query

        Ok(QueryInfo {
            query_type: QueryType::GraphQL,
            original_query: query.to_string(),
            patterns,
            variables: variables_set,
            filters: Vec::new(),
            complexity: query.len() as u64 / 10, // Simple complexity estimate
            estimated_cost: query.len() as u64,
        })
    }

    /// Plan a GraphQL query execution across federated services
    pub async fn plan_graphql(
        &self,
        query_info: &QueryInfo,
        service_registry: &ServiceRegistry,
    ) -> Result<ExecutionPlan> {
        let context = ExecutionContext {
            query_id: "graphql_query".to_string(),
            execution_id: "exec_1".to_string(),
            start_time: std::time::Instant::now(),
            timeout: None,
            variables: std::collections::HashMap::new(),
            metadata: std::collections::HashMap::new(),
        };

        self.plan_federated_query(&query_info.original_query, None, &context, service_registry)
            .await
    }

    /// Analyze query patterns to determine required service capabilities
    fn analyze_query_capabilities(&self, query_info: &QueryInfo) -> Vec<crate::ServiceCapability> {
        let mut capabilities = Vec::new();

        // Basic SPARQL support is always required
        capabilities.push(crate::ServiceCapability::SparqlQuery);

        // Analyze patterns to detect specific capability requirements
        for pattern in &query_info.patterns {
            // Check for geospatial patterns
            if pattern
                .predicate
                .as_ref()
                .is_some_and(|p| p.contains("geo:") || p.contains("wgs84") || p.contains("geof:"))
                || pattern.pattern_string.contains("geo:")
            {
                capabilities.push(crate::ServiceCapability::Geospatial);
            }

            // Check for full-text search patterns
            if pattern
                .predicate
                .as_ref()
                .is_some_and(|p| p.contains("pf:") || p.contains("text:") || p.contains("lucene:"))
            {
                capabilities.push(crate::ServiceCapability::FullTextSearch);
            }
        }

        // Check original query for additional capabilities
        let query_lower = query_info.original_query.to_lowercase();
        if query_lower.contains("insert")
            || query_lower.contains("delete")
            || query_lower.contains("update")
        {
            capabilities.push(crate::ServiceCapability::SparqlUpdate);
        }

        capabilities
    }

    /// Select the most appropriate service based on required capabilities
    async fn select_service_for_capabilities(
        &self,
        required_capabilities: &[crate::ServiceCapability],
        service_registry: &ServiceRegistry,
    ) -> Result<String> {
        let services = service_registry.get_all_services();

        // Find services that have all required capabilities
        let mut suitable_services = Vec::new();

        for service in services {
            let has_all_capabilities = required_capabilities
                .iter()
                .all(|cap| service.capabilities.contains(cap));

            if has_all_capabilities {
                suitable_services.push(service);
            }
        }

        // If no suitable services found, fall back to any available service
        if suitable_services.is_empty() {
            let all_services = service_registry.get_all_services();
            if let Some(service) = all_services.into_iter().next() {
                return Ok(service.id.clone());
            } else {
                return Err(anyhow!("No services available in registry"));
            }
        }

        // Select the first suitable service (could be enhanced with more sophisticated selection)
        Ok(suitable_services[0].id.clone())
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
    pub default_retry_config: Option<types::RetryConfig>,
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
            default_retry_config: Some(types::RetryConfig {
                max_attempts: 3,
                initial_delay: std::time::Duration::from_millis(100),
                max_delay: std::time::Duration::from_secs(5),
                backoff_multiplier: 2.0,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallelization_analysis_no_dependencies() {
        let planner = FederatedQueryPlanner::new();

        // Create steps with no dependencies
        let steps = vec![
            ExecutionStep {
                step_id: "step1".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service1".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query1".to_string(),
                dependencies: Vec::new(),
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
            ExecutionStep {
                step_id: "step2".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service2".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query2".to_string(),
                dependencies: Vec::new(),
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
            ExecutionStep {
                step_id: "step3".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service3".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query3".to_string(),
                dependencies: Vec::new(),
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
        ];

        let parallelizable = planner.analyze_parallelizable_steps(&steps);

        // All steps should be in one parallel group
        assert_eq!(parallelizable.len(), 1);
        assert_eq!(parallelizable[0].len(), 3);
        assert!(parallelizable[0].contains(&"step1".to_string()));
        assert!(parallelizable[0].contains(&"step2".to_string()));
        assert!(parallelizable[0].contains(&"step3".to_string()));
    }

    #[test]
    fn test_parallelization_analysis_linear_dependencies() {
        let planner = FederatedQueryPlanner::new();

        // Create steps with linear dependencies: step1 -> step2 -> step3
        let steps = vec![
            ExecutionStep {
                step_id: "step1".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service1".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query1".to_string(),
                dependencies: Vec::new(),
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
            ExecutionStep {
                step_id: "step2".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service2".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query2".to_string(),
                dependencies: vec!["step1".to_string()],
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
            ExecutionStep {
                step_id: "step3".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service3".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query3".to_string(),
                dependencies: vec!["step2".to_string()],
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
        ];

        let parallelizable = planner.analyze_parallelizable_steps(&steps);

        // Each step should be in its own group (sequential execution)
        assert_eq!(parallelizable.len(), 3);
        assert_eq!(parallelizable[0], vec!["step1"]);
        assert_eq!(parallelizable[1], vec!["step2"]);
        assert_eq!(parallelizable[2], vec!["step3"]);
    }

    #[test]
    fn test_parallelization_analysis_diamond_pattern() {
        let planner = FederatedQueryPlanner::new();

        // Create diamond pattern: step1 -> (step2, step3) -> step4
        let steps = vec![
            ExecutionStep {
                step_id: "step1".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service1".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query1".to_string(),
                dependencies: Vec::new(),
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
            ExecutionStep {
                step_id: "step2".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service2".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query2".to_string(),
                dependencies: vec!["step1".to_string()],
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
            ExecutionStep {
                step_id: "step3".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service3".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query3".to_string(),
                dependencies: vec!["step1".to_string()],
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
            ExecutionStep {
                step_id: "step4".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service4".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query4".to_string(),
                dependencies: vec!["step2".to_string(), "step3".to_string()],
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
        ];

        let parallelizable = planner.analyze_parallelizable_steps(&steps);

        // Should have 3 levels: [step1], [step2, step3], [step4]
        assert_eq!(parallelizable.len(), 3);
        assert_eq!(parallelizable[0], vec!["step1"]);
        assert_eq!(parallelizable[1].len(), 2);
        assert!(parallelizable[1].contains(&"step2".to_string()));
        assert!(parallelizable[1].contains(&"step3".to_string()));
        assert_eq!(parallelizable[2], vec!["step4"]);
    }

    #[test]
    fn test_parallelization_analysis_complex_pattern() {
        let planner = FederatedQueryPlanner::new();

        // Create more complex pattern with multiple parallel branches
        let steps = vec![
            ExecutionStep {
                step_id: "step1".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service1".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query1".to_string(),
                dependencies: Vec::new(),
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
            ExecutionStep {
                step_id: "step2".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service2".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query2".to_string(),
                dependencies: Vec::new(),
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
            ExecutionStep {
                step_id: "step3".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service3".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query3".to_string(),
                dependencies: vec!["step1".to_string()],
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
            ExecutionStep {
                step_id: "step4".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service4".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query4".to_string(),
                dependencies: vec!["step1".to_string()],
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
            ExecutionStep {
                step_id: "step5".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service5".to_string()),
                service_url: None,
                auth_config: None,
                query_fragment: "query5".to_string(),
                dependencies: vec!["step2".to_string()],
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            },
        ];

        let parallelizable = planner.analyze_parallelizable_steps(&steps);

        // Level 0: step1, step2 (no dependencies)
        // Level 1: step3, step4, step5 (depend on step1 or step2)
        assert_eq!(parallelizable.len(), 2);
        assert_eq!(parallelizable[0].len(), 2);
        assert!(parallelizable[0].contains(&"step1".to_string()));
        assert!(parallelizable[0].contains(&"step2".to_string()));
        assert_eq!(parallelizable[1].len(), 3);
        assert!(parallelizable[1].contains(&"step3".to_string()));
        assert!(parallelizable[1].contains(&"step4".to_string()));
        assert!(parallelizable[1].contains(&"step5".to_string()));
    }

    #[test]
    fn test_parallelization_analysis_empty_steps() {
        let planner = FederatedQueryPlanner::new();
        let steps: Vec<ExecutionStep> = Vec::new();

        let parallelizable = planner.analyze_parallelizable_steps(&steps);

        assert!(parallelizable.is_empty());
    }
}
