//! SPARQL integration for vector search and hybrid symbolic-vector queries
//! 
//! This module provides comprehensive SPARQL integration capabilities for vector operations,
//! including cross-language search, federated queries, and custom function support.

use crate::{
    embeddings::{EmbeddingManager, EmbeddingStrategy},
    graph_aware_search::{GraphAwareConfig, GraphAwareSearch},
    VectorStore,
};
use anyhow::Result;
use std::collections::HashMap;

// Re-export main types and modules
pub mod config;
pub mod cross_language;
pub mod federation;
pub mod monitoring;
pub mod query_executor;
pub mod sparql_functions;

pub use config::{
    VectorQuery, VectorQueryOptimizer, VectorQueryResult, VectorServiceArg, VectorServiceConfig,
    VectorServiceFunction, VectorServiceParameter, VectorServiceResult,
};
pub use cross_language::CrossLanguageProcessor;
pub use federation::{FederatedQueryResult, FederationManager};
pub use monitoring::{PerformanceMonitor, PerformanceReport};
pub use query_executor::QueryExecutor;
pub use sparql_functions::{CustomVectorFunction, SparqlVectorFunctions};

/// Main SPARQL vector service implementation
pub struct SparqlVectorService {
    config: VectorServiceConfig,
    query_executor: QueryExecutor,
    sparql_functions: SparqlVectorFunctions,
    federation_manager: Option<FederationManager>,
    performance_monitor: Option<PerformanceMonitor>,
}

impl SparqlVectorService {
    /// Create a new SPARQL vector service
    pub fn new(config: VectorServiceConfig, embedding_strategy: EmbeddingStrategy) -> Result<Self> {
        let vector_store = VectorStore::new();
        let embedding_manager = EmbeddingManager::new(embedding_strategy, 1000)?;

        let performance_monitor = if config.enable_monitoring {
            Some(PerformanceMonitor::new())
        } else {
            None
        };

        let graph_aware_search = if config.enable_monitoring {
            Some(GraphAwareSearch::new(GraphAwareConfig::default()))
        } else {
            None
        };

        let optimizer = VectorQueryOptimizer::default();
        let query_executor = QueryExecutor::new(
            vector_store,
            embedding_manager,
            optimizer,
            performance_monitor.clone(),
            graph_aware_search,
        );

        let sparql_functions = SparqlVectorFunctions::new();

        Ok(Self {
            config,
            query_executor,
            sparql_functions,
            federation_manager: None,
            performance_monitor,
        })
    }

    /// Execute a SPARQL vector function
    pub fn execute_function(
        &mut self,
        function_name: &str,
        args: &[VectorServiceArg],
    ) -> Result<VectorServiceResult> {
        let start_time = std::time::Instant::now();

        let result = self.sparql_functions.execute_function(
            function_name,
            args,
            &mut self.query_executor,
        );

        // Record performance metrics
        if let Some(ref monitor) = self.performance_monitor {
            let duration = start_time.elapsed();
            monitor.record_query(duration, result.is_ok());
            monitor.record_operation(&format!("function_{}", function_name), duration);
        }

        result
    }

    /// Execute an optimized vector query
    pub fn execute_query(&mut self, query: &VectorQuery) -> Result<VectorQueryResult> {
        self.query_executor.execute_optimized_query(query)
    }

    /// Register a custom SPARQL function
    pub fn register_function(&mut self, function: VectorServiceFunction) {
        self.sparql_functions.register_function(function);
    }

    /// Register a custom function implementation
    pub fn register_custom_function(
        &mut self,
        name: String,
        function: Box<dyn CustomVectorFunction>,
    ) {
        self.sparql_functions
            .register_custom_function(name, function);
    }

    /// Enable federation with specified endpoints
    pub fn enable_federation(&mut self, endpoint_urls: Vec<String>) {
        self.federation_manager = Some(FederationManager::new(endpoint_urls));
    }

    /// Execute federated query
    pub async fn execute_federated_query(
        &mut self,
        endpoints: &[String],
        query: &VectorQuery,
    ) -> Result<FederatedQueryResult> {
        if let Some(ref mut manager) = self.federation_manager {
            manager.execute_federated_query(endpoints, query).await
        } else {
            Err(anyhow::anyhow!("Federation not enabled"))
        }
    }

    /// Get performance report
    pub fn get_performance_report(&self) -> Option<PerformanceReport> {
        self.performance_monitor.as_ref().map(|m| m.generate_report())
    }

    /// Get function documentation
    pub fn get_function_documentation(&self, name: &str) -> Option<String> {
        self.sparql_functions.get_function_documentation(name)
    }

    /// Generate SPARQL function definitions
    pub fn generate_sparql_definitions(&self) -> String {
        self.sparql_functions.generate_sparql_definitions()
    }

    /// Check if a function is registered
    pub fn is_function_registered(&self, name: &str) -> bool {
        self.sparql_functions.is_function_registered(name)
    }

    /// Get all registered functions
    pub fn get_all_functions(&self) -> &HashMap<String, VectorServiceFunction> {
        self.sparql_functions.get_all_functions()
    }

    /// Clear query cache
    pub fn clear_cache(&mut self) {
        self.query_executor.clear_cache();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        self.query_executor.cache_stats()
    }

    /// Update configuration
    pub fn update_config(&mut self, config: VectorServiceConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &VectorServiceConfig {
        &self.config
    }
}

/// Builder for creating SPARQL vector service with custom configuration
pub struct SparqlVectorServiceBuilder {
    config: VectorServiceConfig,
    embedding_strategy: Option<EmbeddingStrategy>,
    federation_endpoints: Vec<String>,
    custom_functions: Vec<(String, Box<dyn CustomVectorFunction>)>,
}

impl SparqlVectorServiceBuilder {
    pub fn new() -> Self {
        Self {
            config: VectorServiceConfig::default(),
            embedding_strategy: None,
            federation_endpoints: Vec::new(),
            custom_functions: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: VectorServiceConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_embedding_strategy(mut self, strategy: EmbeddingStrategy) -> Self {
        self.embedding_strategy = Some(strategy);
        self
    }

    pub fn with_federation_endpoints(mut self, endpoints: Vec<String>) -> Self {
        self.federation_endpoints = endpoints;
        self
    }

    pub fn with_custom_function(
        mut self,
        name: String,
        function: Box<dyn CustomVectorFunction>,
    ) -> Self {
        self.custom_functions.push((name, function));
        self
    }

    pub fn build(self) -> Result<SparqlVectorService> {
        let embedding_strategy = self
            .embedding_strategy
            .unwrap_or(EmbeddingStrategy::SentenceTransformers);

        let mut service = SparqlVectorService::new(self.config, embedding_strategy)?;

        // Enable federation if endpoints provided
        if !self.federation_endpoints.is_empty() {
            service.enable_federation(self.federation_endpoints);
        }

        // Register custom functions
        for (name, function) in self.custom_functions {
            service.register_custom_function(name, function);
        }

        Ok(service)
    }
}

impl Default for SparqlVectorServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common operations
pub mod convenience {
    use super::*;

    /// Create a basic SPARQL vector service with default configuration
    pub fn create_basic_service() -> Result<SparqlVectorService> {
        SparqlVectorService::new(
            VectorServiceConfig::default(),
            EmbeddingStrategy::SentenceTransformers,
        )
    }

    /// Create a high-performance SPARQL vector service
    pub fn create_high_performance_service() -> Result<SparqlVectorService> {
        let config = VectorServiceConfig {
            enable_caching: true,
            cache_size: 10000,
            enable_optimization: true,
            enable_monitoring: true,
            ..Default::default()
        };

        SparqlVectorService::new(config, EmbeddingStrategy::SentenceTransformers)
    }

    /// Create a federated SPARQL vector service
    pub fn create_federated_service(endpoints: Vec<String>) -> Result<SparqlVectorService> {
        let mut service = create_basic_service()?;
        service.enable_federation(endpoints);
        Ok(service)
    }

    /// Execute a simple similarity query
    pub fn execute_similarity_query(
        service: &mut SparqlVectorService,
        resource1: &str,
        resource2: &str,
    ) -> Result<f32> {
        let args = vec![
            VectorServiceArg::IRI(resource1.to_string()),
            VectorServiceArg::IRI(resource2.to_string()),
        ];

        match service.execute_function("similarity", &args)? {
            VectorServiceResult::Number(score) => Ok(score),
            VectorServiceResult::SimilarityList(results) => {
                Ok(results.get(0).map(|(_, score)| *score).unwrap_or(0.0))
            }
            _ => Err(anyhow::anyhow!("Unexpected result type for similarity query")),
        }
    }

    /// Execute a simple search query
    pub fn execute_search_query(
        service: &mut SparqlVectorService,
        query_text: &str,
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<(String, f32)>> {
        let args = vec![
            VectorServiceArg::String(query_text.to_string()),
            VectorServiceArg::Number(limit as f32),
            VectorServiceArg::Number(threshold),
        ];

        match service.execute_function("search", &args)? {
            VectorServiceResult::SimilarityList(results) => Ok(results),
            _ => Err(anyhow::anyhow!("Unexpected result type for search query")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingStrategy;

    #[test]
    fn test_service_creation() {
        let config = VectorServiceConfig::default();
        let service = SparqlVectorService::new(config, EmbeddingStrategy::TfIdf);
        assert!(service.is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let service = SparqlVectorServiceBuilder::new()
            .with_embedding_strategy(EmbeddingStrategy::SentenceTransformers)
            .with_federation_endpoints(vec!["http://endpoint1.com".to_string()])
            .build();

        assert!(service.is_ok());
    }

    #[test]
    fn test_function_registration() {
        let mut service = convenience::create_basic_service().unwrap();

        assert!(service.is_function_registered("similarity"));
        assert!(service.is_function_registered("search"));
        assert!(!service.is_function_registered("nonexistent"));
    }

    #[test]
    fn test_convenience_functions() {
        let basic_service = convenience::create_basic_service();
        assert!(basic_service.is_ok());

        let hp_service = convenience::create_high_performance_service();
        assert!(hp_service.is_ok());

        let federated_service = convenience::create_federated_service(vec![
            "http://endpoint1.com".to_string(),
        ]);
        assert!(federated_service.is_ok());
    }

    #[test]
    fn test_configuration_update() {
        let mut service = convenience::create_basic_service().unwrap();

        let new_config = VectorServiceConfig {
            default_threshold: 0.8,
            default_limit: 20,
            ..Default::default()
        };

        service.update_config(new_config.clone());
        assert_eq!(service.get_config().default_threshold, 0.8);
        assert_eq!(service.get_config().default_limit, 20);
    }

    #[tokio::test]
    async fn test_function_documentation() {
        let service = convenience::create_basic_service().unwrap();

        let doc = service.get_function_documentation("similarity");
        assert!(doc.is_some());
        assert!(doc.unwrap().contains("similarity"));

        let sparql_defs = service.generate_sparql_definitions();
        assert!(sparql_defs.contains("vec:similarity"));
        assert!(sparql_defs.contains("SELECT"));
    }
}