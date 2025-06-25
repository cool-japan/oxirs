//! OxiRS Federation Engine
//!
//! This module provides federated query processing capabilities for SPARQL and GraphQL,
//! including service discovery, query decomposition, result integration, and fault tolerance.
//!
//! # Features
//!
//! - SPARQL SERVICE planner and executor for federated SPARQL queries
//! - GraphQL schema stitching and federation
//! - Service discovery and capability detection
//! - Query decomposition and optimization across multiple sources
//! - Result integration with fault tolerance and partial result handling
//! - Load balancing and performance monitoring
//!
//! # Architecture
//!
//! The federation engine consists of several key components:
//!
//! - `ServiceRegistry`: Manages available federated services and their capabilities
//! - `QueryPlanner`: Decomposes queries across multiple services
//! - `Executor`: Executes federated queries with parallel processing
//! - `ResultIntegrator`: Combines results from multiple sources
//! - `FaultHandler`: Manages service failures and retries

use anyhow::{anyhow, Result};
use oxirs_core::{Graph, Triple, Quad, Term};
use oxirs_gql::types::Schema as GraphQLSchema;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

pub mod service;
pub mod planner;
pub mod executor;
pub mod integration;
pub mod graphql;
pub mod discovery;
pub mod monitoring;
pub mod cache;

pub use service::*;
pub use planner::*;
pub use executor::*;
pub use integration::*;
pub use graphql::*;
pub use discovery::*;
pub use monitoring::*;
pub use cache::*;

/// Main federation engine that coordinates all federated query processing
#[derive(Debug, Clone)]
pub struct FederationEngine {
    /// Registry of available services
    service_registry: Arc<RwLock<ServiceRegistry>>,
    /// Query planner for service selection and decomposition
    query_planner: Arc<QueryPlanner>,
    /// Execution engine for federated queries
    executor: Arc<FederatedExecutor>,
    /// Result integration engine
    integrator: Arc<ResultIntegrator>,
    /// GraphQL federation manager
    graphql_federation: Arc<GraphQLFederation>,
    /// Performance monitoring
    monitor: Arc<FederationMonitor>,
    /// Advanced caching system
    cache: Arc<FederationCache>,
}

impl FederationEngine {
    /// Create a new federation engine with default configuration
    pub fn new() -> Self {
        let service_registry = Arc::new(RwLock::new(ServiceRegistry::new()));
        let query_planner = Arc::new(QueryPlanner::new());
        let executor = Arc::new(FederatedExecutor::new());
        let integrator = Arc::new(ResultIntegrator::new());
        let graphql_federation = Arc::new(GraphQLFederation::new());
        let monitor = Arc::new(FederationMonitor::new());
        let cache = Arc::new(FederationCache::new());

        Self {
            service_registry,
            query_planner,
            executor,
            integrator,
            graphql_federation,
            monitor,
            cache,
        }
    }

    /// Create a new federation engine with custom configuration
    pub fn with_config(config: FederationConfig) -> Self {
        let service_registry = Arc::new(RwLock::new(ServiceRegistry::with_config(config.registry_config)));
        let query_planner = Arc::new(QueryPlanner::with_config(config.planner_config));
        let executor = Arc::new(FederatedExecutor::with_config(config.executor_config));
        let integrator = Arc::new(ResultIntegrator::with_config(config.integrator_config));
        let graphql_federation = Arc::new(GraphQLFederation::with_config(config.graphql_config));
        let monitor = Arc::new(FederationMonitor::with_config(config.monitor_config));
        let cache = Arc::new(FederationCache::with_config(config.cache_config));

        Self {
            service_registry,
            query_planner,
            executor,
            integrator,
            graphql_federation,
            monitor,
            cache,
        }
    }

    /// Register a new federated service
    pub async fn register_service(&self, service: FederatedService) -> Result<()> {
        let mut registry = self.service_registry.write().await;
        registry.register(service).await
    }

    /// Unregister a federated service
    pub async fn unregister_service(&self, service_id: &str) -> Result<()> {
        let mut registry = self.service_registry.write().await;
        registry.unregister(service_id).await
    }

    /// Execute a federated SPARQL query
    pub async fn execute_sparql(&self, query: &str) -> Result<FederatedResult> {
        let start_time = Instant::now();
        
        // Parse and analyze the query
        let query_info = self.query_planner.analyze_sparql(query).await?;
        
        // Generate cache key
        let cache_key = self.cache.generate_query_key(&query_info);
        
        // Check cache first
        if let Some(cached_result) = self.cache.get_query_result(&cache_key).await {
            let execution_time = start_time.elapsed();
            
            // Record cache hit
            self.monitor.record_cache_hit("query_cache", true).await;
            self.monitor.record_query_execution("sparql", execution_time, true).await;
            
            return match cached_result {
                QueryResultCache::Sparql(sparql_results) => {
                    // Convert back to FederatedResult
                    let result_bindings: Vec<HashMap<String, oxirs_core::Term>> = sparql_results
                        .results
                        .bindings
                        .into_iter()
                        .map(|binding| {
                            // Convert SparqlBinding to HashMap<String, Term>
                            binding.into_iter()
                                .filter_map(|(var, sparql_value)| {
                                    match sparql_value.value_type.as_str() {
                                        "uri" => {
                                            if let Ok(iri) = oxirs_core::NamedNode::new(&sparql_value.value) {
                                                Some((var, oxirs_core::Term::NamedNode(iri)))
                                            } else {
                                                None
                                            }
                                        }
                                        "literal" => {
                                            if let Some(datatype_str) = sparql_value.datatype {
                                                if let Ok(datatype) = oxirs_core::NamedNode::new(&datatype_str) {
                                                    Some((var, oxirs_core::Term::Literal(
                                                        oxirs_core::Literal::new_typed(&sparql_value.value, datatype)
                                                    )))
                                                } else {
                                                    Some((var, oxirs_core::Term::Literal(
                                                        oxirs_core::Literal::new(&sparql_value.value)
                                                    )))
                                                }
                                            } else if let Some(lang) = sparql_value.lang {
                                                if let Ok(literal) = oxirs_core::Literal::new_lang(&sparql_value.value, &lang) {
                                                    Some((var, oxirs_core::Term::Literal(literal)))
                                                } else {
                                                    Some((var, oxirs_core::Term::Literal(
                                                        oxirs_core::Literal::new(&sparql_value.value)
                                                    )))
                                                }
                                            } else {
                                                Some((var, oxirs_core::Term::Literal(
                                                    oxirs_core::Literal::new(&sparql_value.value)
                                                )))
                                            }
                                        }
                                        "bnode" => {
                                            if let Ok(bnode) = oxirs_core::BlankNode::new(&sparql_value.value) {
                                                Some((var, oxirs_core::Term::BlankNode(bnode)))
                                            } else {
                                                None
                                            }
                                        }
                                        _ => None,
                                    }
                                })
                                .collect()
                        })
                        .collect();

                    Ok(FederatedResult {
                        data: QueryResult::Sparql(result_bindings),
                        metadata: ExecutionMetadata {
                            execution_time,
                            services_used: 0, // From cache
                            subqueries_executed: 0,
                            cache_hit: true,
                            plan_summary: "Cached result".to_string(),
                        },
                        errors: vec![],
                    })
                }
                _ => {
                    // Invalid cache entry type
                    self.cache.remove(&cache_key).await;
                    return Err(anyhow!("Invalid cached result type for SPARQL query"));
                }
            };
        }
        
        // Cache miss - execute normally
        self.monitor.record_cache_hit("query_cache", false).await;
        
        // Plan the federated execution
        let registry = self.service_registry.read().await;
        let execution_plan = self.query_planner.plan_sparql(&query_info, &*registry).await?;
        drop(registry);
        
        // Execute the plan
        let partial_results = self.executor.execute_plan(&execution_plan).await?;
        
        // Integrate results
        let final_result = self.integrator.integrate_sparql_results(partial_results).await?;
        
        // Cache the result if successful
        if final_result.is_success() {
            if let QueryResult::Sparql(ref result_bindings) = final_result.data {
                // Convert to cacheable format
                let sparql_bindings: Vec<crate::executor::SparqlBinding> = result_bindings
                    .iter()
                    .map(|binding| {
                        binding.iter()
                            .map(|(var, term)| {
                                let sparql_value = match term {
                                    oxirs_core::Term::NamedNode(node) => crate::executor::SparqlValue {
                                        value_type: "uri".to_string(),
                                        value: node.to_string(),
                                        datatype: None,
                                        lang: None,
                                    },
                                    oxirs_core::Term::Literal(literal) => {
                                        if let Some(lang) = literal.language() {
                                            crate::executor::SparqlValue {
                                                value_type: "literal".to_string(),
                                                value: literal.value().to_string(),
                                                datatype: None,
                                                lang: Some(lang.to_string()),
                                            }
                                        } else {
                                            crate::executor::SparqlValue {
                                                value_type: "literal".to_string(),
                                                value: literal.value().to_string(),
                                                datatype: literal.datatype().map(|dt| dt.to_string()),
                                                lang: None,
                                            }
                                        }
                                    }
                                    oxirs_core::Term::BlankNode(bnode) => crate::executor::SparqlValue {
                                        value_type: "bnode".to_string(),
                                        value: bnode.to_string(),
                                        datatype: None,
                                        lang: None,
                                    },
                                    oxirs_core::Term::Variable(var) => crate::executor::SparqlValue {
                                        value_type: "variable".to_string(),
                                        value: var.to_string(),
                                        datatype: None,
                                        lang: None,
                                    },
                                };
                                (var.clone(), sparql_value)
                            })
                            .collect()
                    })
                    .collect();

                let cached_result = crate::executor::SparqlResults {
                    head: crate::executor::SparqlHead {
                        vars: result_bindings.first()
                            .map(|binding| binding.keys().cloned().collect())
                            .unwrap_or_default(),
                    },
                    results: crate::executor::SparqlResultSet {
                        bindings: sparql_bindings,
                    },
                };

                // Cache with default TTL
                self.cache.put_query_result(&cache_key, QueryResultCache::Sparql(cached_result), None).await;
            }
        }
        
        // Record metrics
        let execution_time = start_time.elapsed();
        self.monitor.record_query_execution("sparql", execution_time, final_result.is_success()).await;
        
        Ok(final_result)
    }

    /// Execute a federated GraphQL query
    pub async fn execute_graphql(&self, query: &str, variables: Option<serde_json::Value>) -> Result<FederatedResult> {
        let start_time = Instant::now();
        
        // Parse and analyze the GraphQL query
        let query_info = self.query_planner.analyze_graphql(query, variables.as_ref()).await?;
        
        // Plan the federated execution
        let registry = self.service_registry.read().await;
        let execution_plan = self.query_planner.plan_graphql(&query_info, &*registry).await?;
        drop(registry);
        
        // Execute the plan using GraphQL federation
        let partial_results = self.graphql_federation.execute_federated(&execution_plan).await?;
        
        // Integrate results
        let final_result = self.integrator.integrate_graphql_results(partial_results).await?;
        
        // Record metrics
        let execution_time = start_time.elapsed();
        self.monitor.record_query_execution("graphql", execution_time, final_result.is_success()).await;
        
        Ok(final_result)
    }

    /// Get federation statistics and health information
    pub async fn get_stats(&self) -> FederationStats {
        let registry_stats = {
            let registry = self.service_registry.read().await;
            registry.get_stats().await
        };
        
        let monitor_stats = self.monitor.get_stats().await;
        
        FederationStats {
            registry: registry_stats,
            monitor: monitor_stats,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Perform health check on all registered services
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let registry = self.service_registry.read().await;
        registry.health_check().await
    }

    /// Update service capabilities through discovery
    pub async fn discover_services(&self) -> Result<()> {
        let mut registry = self.service_registry.write().await;
        let discovery = ServiceDiscovery::new();
        discovery.update_service_capabilities(&mut *registry).await
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> CacheStats {
        self.cache.get_stats().await
    }

    /// Invalidate cache for a specific service
    pub async fn invalidate_service_cache(&self, service_id: &str) {
        self.cache.invalidate_service(service_id).await;
    }

    /// Invalidate all query caches
    pub async fn invalidate_query_cache(&self) {
        self.cache.invalidate_queries().await;
    }

    /// Warm up the cache with commonly used data
    pub async fn warmup_cache(&self) -> Result<()> {
        self.cache.warmup().await
    }

    /// Clean up expired cache entries
    pub async fn cleanup_cache(&self) {
        self.cache.cleanup_expired().await;
    }
}

impl Default for FederationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the federation engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    pub registry_config: ServiceRegistryConfig,
    pub planner_config: QueryPlannerConfig,
    pub executor_config: FederatedExecutorConfig,
    pub integrator_config: ResultIntegratorConfig,
    pub graphql_config: GraphQLFederationConfig,
    pub monitor_config: FederationMonitorConfig,
    pub cache_config: CacheConfig,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            registry_config: ServiceRegistryConfig::default(),
            planner_config: QueryPlannerConfig::default(),
            executor_config: FederatedExecutorConfig::default(),
            integrator_config: ResultIntegratorConfig::default(),
            graphql_config: GraphQLFederationConfig::default(),
            monitor_config: FederationMonitorConfig::default(),
            cache_config: CacheConfig::default(),
        }
    }
}

/// Result of a federated query execution
#[derive(Debug, Clone)]
pub struct FederatedResult {
    /// The integrated query results
    pub data: QueryResult,
    /// Metadata about the execution
    pub metadata: ExecutionMetadata,
    /// Any errors or warnings from the execution
    pub errors: Vec<FederationError>,
}

impl FederatedResult {
    /// Check if the execution was successful (no critical errors)
    pub fn is_success(&self) -> bool {
        !self.errors.iter().any(|e| e.is_critical())
    }
    
    /// Get the number of results
    pub fn result_count(&self) -> usize {
        match &self.data {
            QueryResult::Sparql(results) => results.len(),
            QueryResult::GraphQL(result) => {
                if result.is_object() {
                    1
                } else if result.is_array() {
                    result.as_array().map(|a| a.len()).unwrap_or(0)
                } else {
                    0
                }
            }
        }
    }
}

/// Enumeration of different query result types
#[derive(Debug, Clone)]
pub enum QueryResult {
    Sparql(Vec<HashMap<String, Term>>),
    GraphQL(serde_json::Value),
}

/// Metadata about query execution
#[derive(Debug, Clone)]
pub struct ExecutionMetadata {
    /// Total execution time
    pub execution_time: Duration,
    /// Number of services involved
    pub services_used: usize,
    /// Number of subqueries executed
    pub subqueries_executed: usize,
    /// Whether results were cached
    pub cache_hit: bool,
    /// Execution plan used
    pub plan_summary: String,
}

/// Federation-specific error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum FederationError {
    #[error("Service unavailable: {service_id}")]
    ServiceUnavailable { service_id: String },
    
    #[error("Query planning failed: {reason}")]
    PlanningFailed { reason: String },
    
    #[error("Execution timeout after {timeout:?}")]
    ExecutionTimeout { timeout: Duration },
    
    #[error("Result integration failed: {reason}")]
    IntegrationFailed { reason: String },
    
    #[error("Partial results: {successful_services}/{total_services} services responded")]
    PartialResults { successful_services: usize, total_services: usize },
    
    #[error("Schema conflict: {conflict}")]
    SchemaConflict { conflict: String },
    
    #[error("Authentication failed for service: {service_id}")]
    AuthenticationFailed { service_id: String },
    
    #[error("Rate limit exceeded for service: {service_id}")]
    RateLimitExceeded { service_id: String },
}

impl FederationError {
    /// Check if this error should cause the entire query to fail
    pub fn is_critical(&self) -> bool {
        match self {
            FederationError::ServiceUnavailable { .. } => false, // Can continue with other services
            FederationError::PlanningFailed { .. } => true,
            FederationError::ExecutionTimeout { .. } => true,
            FederationError::IntegrationFailed { .. } => true,
            FederationError::PartialResults { .. } => false, // Warning, not critical
            FederationError::SchemaConflict { .. } => true,
            FederationError::AuthenticationFailed { .. } => false,
            FederationError::RateLimitExceeded { .. } => false,
        }
    }
}

/// Statistics about federation performance
#[derive(Debug, Clone, Serialize)]
pub struct FederationStats {
    pub registry: ServiceRegistryStats,
    pub monitor: MonitorStats,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Health status of the federation system
#[derive(Debug, Clone, Serialize)]
pub struct HealthStatus {
    pub overall_status: ServiceStatus,
    pub service_statuses: HashMap<String, ServiceStatus>,
    pub total_services: usize,
    pub healthy_services: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Service status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unavailable,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_federation_engine_creation() {
        let engine = FederationEngine::new();
        let stats = engine.get_stats().await;
        
        assert_eq!(stats.registry.total_services, 0);
    }

    #[tokio::test]
    async fn test_federation_engine_with_config() {
        let config = FederationConfig::default();
        let engine = FederationEngine::with_config(config);
        let health = engine.health_check().await.unwrap();
        
        assert_eq!(health.overall_status, ServiceStatus::Healthy);
        assert_eq!(health.total_services, 0);
    }

    #[tokio::test]
    async fn test_federation_error_criticality() {
        let critical_error = FederationError::PlanningFailed {
            reason: "Test error".to_string(),
        };
        assert!(critical_error.is_critical());
        
        let non_critical_error = FederationError::ServiceUnavailable {
            service_id: "test-service".to_string(),
        };
        assert!(!non_critical_error.is_critical());
    }

    #[tokio::test]
    async fn test_federated_result_success() {
        let result = FederatedResult {
            data: QueryResult::Sparql(vec![]),
            metadata: ExecutionMetadata {
                execution_time: Duration::from_millis(100),
                services_used: 1,
                subqueries_executed: 1,
                cache_hit: false,
                plan_summary: "Test plan".to_string(),
            },
            errors: vec![],
        };
        
        assert!(result.is_success());
        assert_eq!(result.result_count(), 0);
    }
}
