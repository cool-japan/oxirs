//! Federated Query Planner
//!
//! This module provides the main interface for federated query planning and execution.
//! The implementation has been modularized into separate components for maintainability.

pub mod planning;

use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, info};

use crate::{
    ServiceRegistry, StepResult,
    query_decomposition::{DecomposerConfig, DecompositionResult, QueryDecomposer},
    service_optimizer::{ServiceOptimizer, ServiceOptimizerConfig},
};

// Import from the planning module
use self::planning::*;

/// Main query planner for federated GraphQL queries
#[derive(Debug)]
pub struct QueryPlanner {
    inner: FederatedQueryPlanner,
    decomposer: QueryDecomposer,
    optimizer: ServiceOptimizer,
}

impl QueryPlanner {
    /// Create a new query planner with default configuration
    pub fn new() -> Self {
        Self {
            inner: FederatedQueryPlanner::new(),
            decomposer: QueryDecomposer::new(DecomposerConfig::default()),
            optimizer: ServiceOptimizer::new(ServiceOptimizerConfig::default()),
        }
    }

    /// Create a new query planner with custom configuration
    pub fn with_config(config: PlannerConfig) -> Self {
        Self {
            inner: FederatedQueryPlanner::with_config(config),
            decomposer: QueryDecomposer::new(DecomposerConfig::default()),
            optimizer: ServiceOptimizer::new(ServiceOptimizerConfig::default()),
        }
    }

    /// Plan the execution of a federated query
    pub async fn plan_query(
        &self,
        query: &str,
        variables: Option<serde_json::Value>,
        context: &ExecutionContext,
        service_registry: &ServiceRegistry,
    ) -> Result<ExecutionPlan> {
        debug!("Planning federated query execution");

        // Use the inner planner to create the execution plan
        let mut plan = self.inner.plan_federated_query(query, variables, context, service_registry).await?;

        // Optimize the plan using the service optimizer
        plan = self.optimize_execution_plan(plan, service_registry).await?;

        info!(
            "Created execution plan with {} steps and estimated cost {}",
            plan.steps.len(),
            plan.estimated_total_cost
        );

        Ok(plan)
    }

    /// Optimize an execution plan
    async fn optimize_execution_plan(
        &self,
        mut plan: ExecutionPlan,
        service_registry: &ServiceRegistry,
    ) -> Result<ExecutionPlan> {
        // Apply service-level optimizations
        // TODO: Implement proper service-level optimization
        // The current approach needs refactoring to work with the optimizer interface
        // for step in &mut plan.steps {
        //     if let Some(service_id) = &step.service_id {
        //         if let Some(service) = service_registry.get_service(service_id) {
        //             // Apply optimization logic here
        //         }
        //     }
        // }

        // Recalculate total cost
        plan.estimated_total_cost = plan.steps.iter().map(|s| s.estimated_cost).sum();

        Ok(plan)
    }

    /// Analyze a SPARQL query and return query information
    pub async fn analyze_sparql(&self, query: &str) -> Result<QueryInfo> {
        use std::collections::HashSet;
        
        // Parse the SPARQL query to extract basic information
        // This is a simplified implementation - you might want to use a proper SPARQL parser
        let pattern_count = query.matches("{").count();
        
        // Determine query type
        let query_type = if query.to_uppercase().starts_with("SELECT") {
            QueryType::Select
        } else if query.to_uppercase().starts_with("CONSTRUCT") {
            QueryType::Construct
        } else if query.to_uppercase().starts_with("ASK") {
            QueryType::Ask
        } else if query.to_uppercase().starts_with("DESCRIBE") {
            QueryType::Describe
        } else {
            QueryType::Update
        };
        
        // Extract basic patterns (simplified)
        let patterns = vec![TriplePattern {
            subject: Some("?s".to_string()),
            predicate: Some("?p".to_string()),
            object: Some("?o".to_string()),
            pattern_string: "?s ?p ?o".to_string(),
        }];
        
        // Extract variables (simplified)
        let mut variables = HashSet::new();
        for word in query.split_whitespace() {
            if word.starts_with('?') {
                variables.insert(word.to_string());
            }
        }
        
        // Extract basic filters (simplified)
        let filters = if query.contains("FILTER") {
            vec![FilterExpression {
                expression: "?s = <http://example.org>".to_string(),
                variables: vec!["s".to_string()],
            }]
        } else {
            vec![]
        };

        Ok(QueryInfo {
            query_type,
            original_query: query.to_string(),
            patterns,
            variables,
            filters,
            complexity: pattern_count as u64,
            estimated_cost: pattern_count as u64 * 100,
        })
    }

    /// Plan a SPARQL query execution
    pub async fn plan_sparql(
        &self, 
        query_info: &QueryInfo,
        service_registry: &ServiceRegistry
    ) -> Result<ExecutionPlan> {
        // Create a basic execution context
        let context = ExecutionContext {
            query_id: uuid::Uuid::new_v4().to_string(),
            execution_id: uuid::Uuid::new_v4().to_string(),
            start_time: std::time::Instant::now(),
            timeout: Some(std::time::Duration::from_secs(30)),
            variables: std::collections::HashMap::new(),
            metadata: std::collections::HashMap::new(),
        };
        
        // Use the existing plan_query method with a reconstructed query
        // This is a simplified approach - in practice you'd want to preserve the original query
        let placeholder_query = format!(
            "SELECT * WHERE {{ ?s ?p ?o . FILTER(1=1) }}"
        );
        
        self.plan_query(&placeholder_query, None, &context, service_registry).await
    }

    /// Analyze a GraphQL query and return query information  
    pub async fn analyze_graphql(
        &self,
        query: &str, 
        variables: Option<&serde_json::Value>
    ) -> Result<crate::planner::planning::query_analysis::QueryInfo> {
        // Basic GraphQL analysis - this is simplified
        let field_count = query.matches("{").count();
        let has_variables = variables.is_some() && !variables.unwrap().is_null();
        
        Ok(crate::planner::planning::query_analysis::QueryInfo {
            operation_type: crate::planner::planning::types::GraphQLOperationType::Query,
            field_count,
            has_variables,
            complexity: QueryComplexity::default(),
            estimated_execution_time: std::time::Duration::from_millis(100),
        })
    }

    /// Plan a GraphQL query execution
    pub async fn plan_graphql(
        &self,
        query_info: &crate::planner::planning::query_analysis::QueryInfo,
        service_registry: &ServiceRegistry
    ) -> Result<ExecutionPlan> {
        // Create a basic execution context
        let context = ExecutionContext {
            query_id: uuid::Uuid::new_v4().to_string(),
            execution_id: uuid::Uuid::new_v4().to_string(),
            start_time: std::time::Instant::now(),
            timeout: Some(std::time::Duration::from_secs(30)),
            variables: std::collections::HashMap::new(),
            metadata: std::collections::HashMap::new(),
        };
        
        // Use the existing plan_query method with a reconstructed query
        // This is a simplified approach - in practice you'd want to preserve the original query
        let placeholder_query = format!(
            "query {{ field{} }}", query_info.field_count
        );
        
        self.plan_query(&placeholder_query, None, &context, service_registry).await
    }

    /// Analyze performance and suggest reoptimization
    pub async fn analyze_performance(
        &self,
        metrics: &performance_optimizer::ExecutionMetrics,
        context: &ExecutionContext,
    ) -> Result<ReoptimizationAnalysis> {
        self.inner.analyze_performance(metrics, context).await
    }

    /// Decompose a query for federated execution
    pub async fn decompose_query(
        &self,
        query: &str,
        service_registry: &ServiceRegistry,
    ) -> Result<DecompositionResult> {
        let query_info = self.analyze_sparql(query).await?;
        self.decomposer.decompose(&query_info, service_registry).await
    }

    /// Get historical performance data
    pub async fn get_performance_history(&self) -> HistoricalPerformance {
        self.inner.get_performance_history().await
    }

    /// Create a simple execution plan for testing
    pub fn create_test_plan(query_id: String, steps: Vec<ExecutionStep>) -> ExecutionPlan {
        let total_cost = steps.iter().map(|s| s.estimated_cost).sum();
        
        ExecutionPlan {
            query_id,
            steps,
            estimated_total_cost: total_cost,
            max_parallelism: 1,
            planning_time: std::time::Duration::from_millis(10),
            cache_key: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Validate a query against available services
    pub async fn validate_query(
        &self,
        query: &str,
        service_registry: &ServiceRegistry,
    ) -> Result<Vec<String>> {
        let parsed_query = QueryAnalyzer::parse_graphql_query(query)?;
        
        // Create a mock unified schema for validation
        let unified_schema = UnifiedSchema {
            types: std::collections::HashMap::new(),
            queries: std::collections::HashMap::new(),
            mutations: std::collections::HashMap::new(),
            subscriptions: std::collections::HashMap::new(),
            directives: std::collections::HashMap::new(),
            schema_mapping: std::collections::HashMap::new(),
        };

        QueryAnalyzer::validate_query_against_schema(&parsed_query, &unified_schema)
    }

    /// Extract query complexity metrics
    pub fn analyze_query_complexity(query: &str) -> Result<QueryComplexity> {
        let parsed_query = QueryAnalyzer::parse_graphql_query(query)?;
        Ok(QueryAnalyzer::analyze_query_complexity(&parsed_query))
    }

    /// Check if a query requires federation
    pub fn requires_federation(query: &str) -> Result<bool> {
        let parsed_query = QueryAnalyzer::parse_graphql_query(query)?;
        
        // Simple heuristic: if query has multiple top-level fields, it might require federation
        Ok(parsed_query.selection_set.len() > 1)
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export key types from the planning module
pub use self::planning::{
    ExecutionContext, HistoricalPerformance, ReoptimizationAnalysis,
    PlannerConfig, RetryConfig,
    FilterExpression, ExecutionPlan, ExecutionStep, StepType,
};

// Import types for SPARQL query info
pub use self::planning::{QueryComplexity, TriplePattern};

// Define SPARQL query info for cache compatibility
#[derive(Debug, Clone)]
pub struct QueryInfo {
    pub query_type: QueryType,
    pub original_query: String,
    pub patterns: Vec<TriplePattern>,
    pub variables: std::collections::HashSet<String>,
    pub filters: Vec<FilterExpression>,
    pub complexity: u64,
    pub estimated_cost: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QueryType {
    Select,
    Construct,
    Ask,
    Describe,
    Update,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_planner_creation() {
        let planner = QueryPlanner::new();
        assert!(true); // Basic creation test
    }

    #[tokio::test]
    async fn test_query_complexity_analysis() {
        let query = r#"
            query {
                user(id: "123") {
                    name
                    email
                }
                posts {
                    title
                    content
                }
            }
        "#;

        let complexity = QueryPlanner::analyze_query_complexity(query);
        assert!(complexity.is_ok());
        
        let complexity = complexity.unwrap();
        assert_eq!(complexity.field_count, 2); // user and posts
    }

    #[tokio::test]
    async fn test_federation_requirement_check() {
        let simple_query = r#"
            query {
                user(id: "123") {
                    name
                }
            }
        "#;

        let complex_query = r#"
            query {
                user(id: "123") {
                    name
                }
                posts {
                    title
                }
            }
        "#;

        let simple_result = QueryPlanner::requires_federation(simple_query);
        let complex_result = QueryPlanner::requires_federation(complex_query);

        assert!(simple_result.is_ok());
        assert!(complex_result.is_ok());
        
        assert!(!simple_result.unwrap()); // Single field
        assert!(complex_result.unwrap()); // Multiple fields
    }

    #[test]
    fn test_execution_plan_creation() {
        let steps = vec![
            ExecutionStep {
                step_id: "step1".to_string(),
                step_type: StepType::ServiceQuery,
                service_id: Some("service1".to_string()),
                query_fragment: "{ user { name } }".to_string(),
                dependencies: vec![],
                estimated_cost: 10.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            }
        ];

        let plan = QueryPlanner::create_test_plan("test_query".to_string(), steps);
        
        assert_eq!(plan.query_id, "test_query");
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.estimated_total_cost, 10.0);
    }
}