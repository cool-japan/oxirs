//! Federated Query Planner
//!
//! This module provides the main interface for federated query planning and execution.
//! The implementation has been modularized into separate components for maintainability.

pub mod planning;

use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, info};

use crate::{
    query_decomposition::{DecomposerConfig, DecompositionResult, QueryDecomposer},
    service_optimizer::{ServiceOptimizer, ServiceOptimizerConfig},
    ServiceRegistry, StepResult,
};

// Import specific items from the planning module

// Re-export key types from the planning module - be specific to avoid conflicts
pub use planning::types::{
    ExecutionContext, ExecutionPlan, ExecutionStep, StepType, HistoricalPerformance as TypesHistoricalPerformance,
    QueryType, QueryInfo, TriplePattern, FilterExpression, UnifiedSchema, FederatedSchema, 
    GraphQLFederationConfig, ParsedQuery, GraphQLOperationType, ServiceQuery, EntityResolutionPlan,
    EntityResolutionStep, ReoptimizationAnalysis, RetryConfig
};
// Re-export main planner types
pub use self::planning::{PlannerConfig, FederatedQueryPlanner, HistoricalPerformance};

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
            inner: FederatedQueryPlanner::with_config(config.clone()),
            decomposer: QueryDecomposer::new(DecomposerConfig::default()),
            optimizer: ServiceOptimizer::new(ServiceOptimizerConfig::default()),
        }
    }

    /// Analyze a SPARQL query and extract query information
    pub async fn analyze_sparql(&self, query: &str) -> Result<QueryInfo> {
        self.inner.analyze_sparql(query).await
    }

    /// Plan a SPARQL query execution across federated services
    pub async fn plan_sparql(
        &self,
        query_info: &QueryInfo,
        service_registry: &crate::ServiceRegistry,
    ) -> Result<ExecutionPlan> {
        self.inner.plan_sparql(query_info, service_registry).await
    }

    /// Analyze a GraphQL query and extract query information
    pub async fn analyze_graphql(
        &self,
        query: &str,
        variables: Option<&serde_json::Value>,
    ) -> Result<QueryInfo> {
        self.inner.analyze_graphql(query, variables).await
    }

    /// Plan a GraphQL query execution across federated services
    pub async fn plan_graphql(
        &self,
        query_info: &QueryInfo,
        service_registry: &crate::ServiceRegistry,
    ) -> Result<ExecutionPlan> {
        self.inner.plan_graphql(query_info, service_registry).await
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}