//! SPARQL integration for vector search and hybrid symbolic-vector queries

use crate::{
    embeddings::{EmbeddableContent, EmbeddingManager, EmbeddingStrategy},
    clustering::{ClusteringEngine, ClusteringConfig, ClusteringAlgorithm},
    graph_aware_search::{GraphAwareSearch, GraphAwareConfig, GraphContext, GraphSearchScope},
    Vector, VectorStore,
};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;

/// SPARQL vector service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorServiceConfig {
    /// Service namespace URI
    pub service_uri: String,
    /// Default similarity threshold
    pub default_threshold: f32,
    /// Default number of results to return
    pub default_limit: usize,
    /// Default similarity metric
    pub default_metric: crate::similarity::SimilarityMetric,
    /// Enable caching of vector search results
    pub enable_caching: bool,
    /// Cache size for search results
    pub cache_size: usize,
    /// Enable query optimization
    pub enable_optimization: bool,
    /// Enable result explanations
    pub enable_explanations: bool,
    /// Performance monitoring
    pub enable_monitoring: bool,
}

impl Default for VectorServiceConfig {
    fn default() -> Self {
        Self {
            service_uri: "http://oxirs.org/vec/".to_string(),
            default_threshold: 0.7,
            default_limit: 10,
            default_metric: crate::similarity::SimilarityMetric::Cosine,
            enable_caching: true,
            cache_size: 1000,
            enable_optimization: true,
            enable_explanations: false,
            enable_monitoring: false,
        }
    }
}

/// Vector service function registry
#[derive(Debug, Clone)]
pub struct VectorServiceFunction {
    pub name: String,
    pub arity: usize,
    pub description: String,
    pub parameters: Vec<VectorServiceParameter>,
}

#[derive(Debug, Clone)]
pub struct VectorServiceParameter {
    pub name: String,
    pub param_type: VectorParameterType,
    pub required: bool,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum VectorParameterType {
    IRI,
    Literal,
    Vector,
    Number,
    String,
}

/// Performance monitoring for vector operations
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    pub query_stats: Arc<RwLock<QueryStats>>,
    pub operation_timings: Arc<RwLock<HashMap<String, Vec<Duration>>>>,
    pub cache_stats: Arc<RwLock<CacheStats>>,
}

#[derive(Debug, Clone, Default)]
pub struct QueryStats {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub avg_response_time: Duration,
    pub max_response_time: Duration,
    pub min_response_time: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_size: usize,
    pub cache_capacity: usize,
}

/// Custom vector function trait for user-defined functions
pub trait CustomVectorFunction: Send + Sync {
    fn execute(&self, args: &[VectorServiceArg]) -> Result<VectorServiceResult>;
    fn arity(&self) -> usize;
    fn description(&self) -> String;
}

/// Vector query optimizer for performance enhancement
#[derive(Debug, Clone)]
pub struct VectorQueryOptimizer {
    pub enable_caching: bool,
    pub enable_parallel_execution: bool,
    pub enable_index_selection: bool,
    pub cost_model: CostModel,
}

#[derive(Debug, Clone)]
pub struct CostModel {
    pub linear_search_cost: f32,
    pub index_search_cost: f32,
    pub embedding_generation_cost: f32,
    pub cache_lookup_cost: f32,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            linear_search_cost: 1.0,
            index_search_cost: 0.1,
            embedding_generation_cost: 10.0,
            cache_lookup_cost: 0.01,
        }
    }
}

impl Default for VectorQueryOptimizer {
    fn default() -> Self {
        Self {
            enable_caching: true,
            enable_parallel_execution: true,
            enable_index_selection: true,
            cost_model: CostModel::default(),
        }
    }
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            query_stats: Arc::new(RwLock::new(QueryStats::default())),
            operation_timings: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    pub fn record_query(&self, duration: Duration, success: bool) {
        let mut stats = self.query_stats.write();
        stats.total_queries += 1;
        
        if success {
            stats.successful_queries += 1;
        } else {
            stats.failed_queries += 1;
        }

        if stats.total_queries == 1 {
            stats.avg_response_time = duration;
            stats.max_response_time = duration;
            stats.min_response_time = duration;
        } else {
            // Update running average
            let total_time = stats.avg_response_time.mul_f64(stats.total_queries as f64 - 1.0) + duration;
            stats.avg_response_time = total_time.div_f64(stats.total_queries as f64);
            
            if duration > stats.max_response_time {
                stats.max_response_time = duration;
            }
            if duration < stats.min_response_time {
                stats.min_response_time = duration;
            }
        }
    }

    pub fn record_operation(&self, operation: &str, duration: Duration) {
        let mut timings = self.operation_timings.write();
        timings.entry(operation.to_string()).or_default().push(duration);
    }

    pub fn record_cache_hit(&self) {
        let mut stats = self.cache_stats.write();
        stats.cache_hits += 1;
    }

    pub fn record_cache_miss(&self) {
        let mut stats = self.cache_stats.write();
        stats.cache_misses += 1;
    }

    pub fn get_stats(&self) -> (QueryStats, CacheStats) {
        let query_stats = self.query_stats.read().clone();
        let cache_stats = self.cache_stats.read().clone();
        (query_stats, cache_stats)
    }
}

/// SPARQL vector service implementation
pub struct SparqlVectorService {
    config: VectorServiceConfig,
    vector_store: VectorStore,
    embedding_manager: EmbeddingManager,
    function_registry: HashMap<String, VectorServiceFunction>,
    query_cache: HashMap<String, Vec<(String, f32)>>,
    performance_monitor: Option<PerformanceMonitor>,
    custom_functions: HashMap<String, Box<dyn CustomVectorFunction>>,
    query_optimizer: VectorQueryOptimizer,
    graph_aware_search: Option<GraphAwareSearch>,
}

impl SparqlVectorService {
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

        let mut service = Self {
            config,
            vector_store,
            embedding_manager,
            function_registry: HashMap::new(),
            query_cache: HashMap::new(),
            performance_monitor,
            custom_functions: HashMap::new(),
            query_optimizer: VectorQueryOptimizer::default(),
            graph_aware_search,
        };

        service.register_builtin_functions();
        Ok(service)
    }

    /// Register built-in vector service functions
    fn register_builtin_functions(&mut self) {
        // vec:similar(resource, limit, threshold) -> results
        self.register_function(VectorServiceFunction {
            name: "similar".to_string(),
            arity: 3,
            description: "Find resources similar to the given resource".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "resource".to_string(),
                    param_type: VectorParameterType::IRI,
                    required: true,
                    description: "Resource URI to find similar items for".to_string(),
                },
                VectorServiceParameter {
                    name: "limit".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Maximum number of results to return".to_string(),
                },
                VectorServiceParameter {
                    name: "threshold".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Minimum similarity threshold".to_string(),
                },
            ],
        });

        // vec:similarity(resource1, resource2) -> similarity_score
        self.register_function(VectorServiceFunction {
            name: "similarity".to_string(),
            arity: 2,
            description: "Calculate similarity between two resources".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "resource1".to_string(),
                    param_type: VectorParameterType::IRI,
                    required: true,
                    description: "First resource URI".to_string(),
                },
                VectorServiceParameter {
                    name: "resource2".to_string(),
                    param_type: VectorParameterType::IRI,
                    required: true,
                    description: "Second resource URI".to_string(),
                },
            ],
        });

        // vec:embed_text(text) -> vector
        self.register_function(VectorServiceFunction {
            name: "embed_text".to_string(),
            arity: 1,
            description: "Generate embedding vector for text content".to_string(),
            parameters: vec![VectorServiceParameter {
                name: "text".to_string(),
                param_type: VectorParameterType::String,
                required: true,
                description: "Text content to embed".to_string(),
            }],
        });

        // vec:search_text(query_text, limit, threshold) -> results
        self.register_function(VectorServiceFunction {
            name: "search_text".to_string(),
            arity: 3,
            description: "Search for resources similar to query text".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "query_text".to_string(),
                    param_type: VectorParameterType::String,
                    required: true,
                    description: "Query text to search for".to_string(),
                },
                VectorServiceParameter {
                    name: "limit".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Maximum number of results to return".to_string(),
                },
                VectorServiceParameter {
                    name: "threshold".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Minimum similarity threshold".to_string(),
                },
            ],
        });

        // vec:cluster(resources, threshold) -> clusters
        self.register_function(VectorServiceFunction {
            name: "cluster".to_string(),
            arity: 2,
            description: "Cluster resources by similarity".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "resources".to_string(),
                    param_type: VectorParameterType::IRI,
                    required: true,
                    description: "List of resource URIs to cluster".to_string(),
                },
                VectorServiceParameter {
                    name: "threshold".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Similarity threshold for clustering".to_string(),
                },
            ],
        });

        // vec:vector_similarity(vector1, vector2) -> similarity_score
        self.register_function(VectorServiceFunction {
            name: "vector_similarity".to_string(),
            arity: 2,
            description: "Calculate similarity between two vectors".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "vector1".to_string(),
                    param_type: VectorParameterType::Vector,
                    required: true,
                    description: "First vector".to_string(),
                },
                VectorServiceParameter {
                    name: "vector2".to_string(),
                    param_type: VectorParameterType::Vector,
                    required: true,
                    description: "Second vector".to_string(),
                },
            ],
        });

        // vec:search_in_graph(query_text, graph_uri, limit) -> results
        self.register_function(VectorServiceFunction {
            name: "search_in_graph".to_string(),
            arity: 3,
            description: "Search for resources in a specific named graph".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "query_text".to_string(),
                    param_type: VectorParameterType::String,
                    required: true,
                    description: "Query text to search for".to_string(),
                },
                VectorServiceParameter {
                    name: "graph_uri".to_string(),
                    param_type: VectorParameterType::IRI,
                    required: true,
                    description: "Named graph URI to search in".to_string(),
                },
                VectorServiceParameter {
                    name: "limit".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Maximum number of results to return".to_string(),
                },
            ],
        });

        // vec:search(query_text, limit) -> results (as per TODO.md Phase 4.1.1)
        self.register_function(VectorServiceFunction {
            name: "search".to_string(),
            arity: 2,
            description: "Search for resources similar to query text".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "query_text".to_string(),
                    param_type: VectorParameterType::String,
                    required: true,
                    description: "Text query to search for".to_string(),
                },
                VectorServiceParameter {
                    name: "limit".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Maximum number of results to return".to_string(),
                },
            ],
        });

        // vec:searchIn(query, graph, limit) -> results (as per TODO.md Phase 4.1.1)
        self.register_function(VectorServiceFunction {
            name: "searchIn".to_string(),
            arity: 3,
            description: "Search for resources within a specific graph".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "query".to_string(),
                    param_type: VectorParameterType::String,
                    required: true,
                    description: "Query text to search for".to_string(),
                },
                VectorServiceParameter {
                    name: "graph".to_string(),
                    param_type: VectorParameterType::IRI,
                    required: true,
                    description: "Graph URI to search within".to_string(),
                },
                VectorServiceParameter {
                    name: "limit".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Maximum number of results to return".to_string(),
                },
            ],
        });
    }

    /// Register a custom vector service function
    pub fn register_function(&mut self, function: VectorServiceFunction) {
        self.function_registry
            .insert(function.name.clone(), function);
    }

    /// Register a custom vector function implementation
    pub fn register_custom_function(&mut self, name: String, function: Box<dyn CustomVectorFunction>) {
        self.custom_functions.insert(name, function);
    }

    /// Execute query with performance monitoring and optimization
    pub fn execute_optimized_query(&mut self, query: &VectorQuery) -> Result<VectorQueryResult> {
        let start_time = Instant::now();
        
        // Apply query optimization if enabled
        let optimized_query = if self.query_optimizer.enable_index_selection {
            self.optimize_query(query)?
        } else {
            query.clone()
        };

        // Execute the query
        let result = self.execute_query_internal(&optimized_query);
        
        // Record performance metrics
        let duration = start_time.elapsed();
        if let Some(ref monitor) = self.performance_monitor {
            monitor.record_query(duration, result.is_ok());
            monitor.record_operation(&format!("query_{}", query.operation_type), duration);
        }
        
        result
    }

    /// Optimize query for better performance
    fn optimize_query(&self, query: &VectorQuery) -> Result<VectorQuery> {
        let mut optimized = query.clone();
        
        // Index selection optimization
        if self.query_optimizer.enable_index_selection {
            optimized.preferred_index = self.select_optimal_index(&query)?;
        }
        
        // Caching optimization
        if self.query_optimizer.enable_caching {
            optimized.use_cache = true;
        }
        
        // Parallel execution optimization
        if self.query_optimizer.enable_parallel_execution && query.can_parallelize() {
            optimized.parallel_execution = true;
        }
        
        Ok(optimized)
    }

    /// Select optimal index for query execution
    fn select_optimal_index(&self, query: &VectorQuery) -> Result<Option<String>> {
        let cost_model = &self.query_optimizer.cost_model;
        
        match query.operation_type.as_str() {
            "similarity_search" => {
                // For similarity search, index is usually better for large datasets
                if query.estimated_result_size.unwrap_or(0) > 1000 {
                    Ok(Some("hnsw".to_string()))
                } else {
                    Ok(Some("memory".to_string()))
                }
            }
            "threshold_search" => {
                // Threshold search benefits from approximate indices
                Ok(Some("lsh".to_string()))
            }
            _ => Ok(None),
        }
    }

    /// Execute query with internal optimizations
    fn execute_query_internal(&mut self, query: &VectorQuery) -> Result<VectorQueryResult> {
        // Check cache first if enabled
        if query.use_cache && self.config.enable_caching {
            if let Some(cached_result) = self.get_cached_result(&query.cache_key()) {
                if let Some(ref monitor) = self.performance_monitor {
                    monitor.record_cache_hit();
                }
                return Ok(cached_result);
            } else if let Some(ref monitor) = self.performance_monitor {
                monitor.record_cache_miss();
            }
        }
        
        // Execute based on operation type
        let result = match query.operation_type.as_str() {
            "similarity_search" => self.execute_similarity_search_query(query),
            "threshold_search" => self.execute_threshold_search_query(query),
            "text_search" => self.execute_text_search_query(query),
            _ => Err(anyhow!("Unknown query operation type: {}", query.operation_type)),
        }?;
        
        // Cache result if enabled
        if query.use_cache && self.config.enable_caching {
            self.cache_result(&query.cache_key(), &result);
        }
        
        Ok(result)
    }

    /// Execute similarity search query
    fn execute_similarity_search_query(&mut self, query: &VectorQuery) -> Result<VectorQueryResult> {
        let resource_uri = query.parameters.get("resource_uri")
            .ok_or_else(|| anyhow!("Missing resource_uri parameter"))?;
        let limit = query.parameters.get("limit")
            .and_then(|v| v.parse().ok())
            .unwrap_or(self.config.default_limit);
        
        let results = self.vector_store.similarity_search(resource_uri, limit)?;
        
        Ok(VectorQueryResult {
            results,
            metadata: query.metadata.clone(),
            execution_time: None,
            cache_hit: false,
        })
    }

    /// Execute threshold search query
    fn execute_threshold_search_query(&mut self, query: &VectorQuery) -> Result<VectorQueryResult> {
        let query_text = query.parameters.get("query_text")
            .ok_or_else(|| anyhow!("Missing query_text parameter"))?;
        let threshold = query.parameters.get("threshold")
            .and_then(|v| v.parse().ok())
            .unwrap_or(self.config.default_threshold);
        
        let results = self.vector_store.threshold_search(query_text, threshold)?;
        
        Ok(VectorQueryResult {
            results,
            metadata: query.metadata.clone(),
            execution_time: None,
            cache_hit: false,
        })
    }

    /// Execute text search query
    fn execute_text_search_query(&mut self, query: &VectorQuery) -> Result<VectorQueryResult> {
        let query_text = query.parameters.get("query_text")
            .ok_or_else(|| anyhow!("Missing query_text parameter"))?;
        let limit = query.parameters.get("limit")
            .and_then(|v| v.parse().ok())
            .unwrap_or(self.config.default_limit);
        
        let results = self.vector_store.similarity_search(query_text, limit)?;
        
        Ok(VectorQueryResult {
            results,
            metadata: query.metadata.clone(),
            execution_time: None,
            cache_hit: false,
        })
    }

    /// Get cached result
    fn get_cached_result(&self, cache_key: &str) -> Option<VectorQueryResult> {
        self.query_cache.get(cache_key).map(|results| {
            VectorQueryResult {
                results: results.clone(),
                metadata: HashMap::new(),
                execution_time: None,
                cache_hit: true,
            }
        })
    }

    /// Cache query result
    fn cache_result(&mut self, cache_key: &str, result: &VectorQueryResult) {
        if self.query_cache.len() >= self.config.cache_size {
            // Simple LRU: remove a random entry
            if let Some(key) = self.query_cache.keys().next().cloned() {
                self.query_cache.remove(&key);
            }
        }
        self.query_cache.insert(cache_key.to_string(), result.results.clone());
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> Option<(QueryStats, CacheStats)> {
        self.performance_monitor.as_ref().map(|m| m.get_stats())
    }

    /// Execute a vector service function call
    pub fn execute_function(
        &mut self,
        function_name: &str,
        args: &[VectorServiceArg],
    ) -> Result<VectorServiceResult> {
        let start_time = Instant::now();
        
        // Check if it's a custom function first
        if let Some(custom_function) = self.custom_functions.get(function_name) {
            let result = custom_function.execute(args);
            
            // Record performance
            let duration = start_time.elapsed();
            if let Some(ref monitor) = self.performance_monitor {
                monitor.record_operation(&format!("custom_{}", function_name), duration);
            }
            
            return result;
        }
        
        // Execute built-in functions
        let result = match function_name {
            "similar" => self.execute_similar(args),
            "similarity" => self.execute_similarity(args),
            "embed_text" => self.execute_embed_text(args),
            "search_text" => self.execute_search_text(args),
            "search" => self.execute_search(args),  // New vec:search function
            "searchIn" => self.execute_search_in(args),  // New vec:searchIn function
            "cluster" => self.execute_cluster(args),
            "vector_similarity" => self.execute_vector_similarity(args),
            "search_in_graph" => self.execute_search_in_graph(args),
            _ => Err(anyhow!(
                "Unknown vector service function: {}",
                function_name
            )),
        };
        
        // Record performance
        let duration = start_time.elapsed();
        if let Some(ref monitor) = self.performance_monitor {
            monitor.record_operation(&format!("builtin_{}", function_name), duration);
        }
        
        result
    }

    /// Add resource embedding to the vector store
    pub fn add_resource_embedding(&mut self, uri: &str, content: &EmbeddableContent) -> Result<()> {
        let vector = self.embedding_manager.get_embedding(content)?;
        self.vector_store
            .index_resource(uri.to_string(), &content.to_text())?;
        Ok(())
    }

    /// Add resource with graph membership information
    pub fn add_resource_with_graphs(
        &mut self,
        uri: &str,
        content: &EmbeddableContent,
        graphs: Vec<String>,
    ) -> Result<()> {
        // Add to vector store
        self.add_resource_embedding(uri, content)?;
        
        // Register graph membership if graph-aware search is enabled
        if let Some(ref mut graph_search) = self.graph_aware_search {
            graph_search.register_resource_graph(uri.to_string(), graphs);
        }
        
        Ok(())
    }

    /// Set up graph hierarchy for hierarchical search
    pub fn configure_graph_hierarchy(&mut self, parent_child: HashMap<String, Vec<String>>) {
        if let Some(ref mut graph_search) = self.graph_aware_search {
            graph_search.config.graph_hierarchy.parent_child = parent_child;
        }
    }

    /// Set graph weights for ranking
    pub fn set_graph_weights(&mut self, weights: HashMap<String, f32>) {
        if let Some(ref mut graph_search) = self.graph_aware_search {
            graph_search.config.graph_hierarchy.graph_weights = weights;
        }
    }

    /// Generate SPARQL SERVICE query for vector operations
    pub fn generate_service_query(&self, operation: &VectorOperation) -> String {
        match operation {
            VectorOperation::FindSimilar {
                resource,
                limit,
                threshold,
            } => {
                format!(
                    r#"
                    SERVICE <{}> {{
                        ?result vec:similar <{}> .
                        ?result vec:similarity ?score .
                        FILTER(?score >= {})
                    }}
                    ORDER BY DESC(?score)
                    LIMIT {}
                    "#,
                    self.config.service_uri,
                    resource,
                    threshold.unwrap_or(self.config.default_threshold),
                    limit.unwrap_or(self.config.default_limit)
                )
            }
            VectorOperation::CalculateSimilarity {
                resource1,
                resource2,
            } => {
                format!(
                    r#"
                    SERVICE <{}> {{
                        BIND(vec:similarity(<{}>, <{}>) AS ?similarity_score)
                    }}
                    "#,
                    self.config.service_uri, resource1, resource2
                )
            }
            VectorOperation::SearchText {
                query_text,
                limit,
                threshold,
            } => {
                format!(
                    r#"
                    SERVICE <{}> {{
                        ?result vec:search_text "{}" .
                        ?result vec:similarity ?score .
                        FILTER(?score >= {})
                    }}
                    ORDER BY DESC(?score)
                    LIMIT {}
                    "#,
                    self.config.service_uri,
                    query_text,
                    threshold.unwrap_or(self.config.default_threshold),
                    limit.unwrap_or(self.config.default_limit)
                )
            }
        }
    }

    /// Execute hybrid query that combines symbolic and vector operations
    pub fn execute_hybrid_query(&mut self, query: &HybridQuery) -> Result<Vec<HybridQueryResult>> {
        let mut results = Vec::new();

        // First, execute the symbolic part of the query
        let symbolic_results = self.execute_symbolic_query(&query.symbolic_part)?;

        // Then, for each symbolic result, apply vector operations
        for symbolic_result in symbolic_results {
            let vector_results =
                self.execute_vector_operations(&query.vector_operations, &symbolic_result)?;

            results.push(HybridQueryResult {
                symbolic_bindings: symbolic_result,
                vector_results,
                combined_score: self.calculate_combined_score(&query.scoring),
            });
        }

        // Sort by combined score if requested
        if query.sort_by_score {
            results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        }

        Ok(results)
    }

    // Implementation of individual service functions

    fn execute_similar(&mut self, args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        if args.is_empty() {
            return Err(anyhow!("similar function requires at least one argument"));
        }

        let resource_uri = match &args[0] {
            VectorServiceArg::IRI(uri) => uri,
            _ => return Err(anyhow!("First argument must be a resource URI")),
        };

        let limit = if args.len() > 1 {
            match &args[1] {
                VectorServiceArg::Number(n) => *n as usize,
                _ => self.config.default_limit,
            }
        } else {
            self.config.default_limit
        };

        let threshold = if args.len() > 2 {
            match &args[2] {
                VectorServiceArg::Number(t) => *t,
                _ => self.config.default_threshold,
            }
        } else {
            self.config.default_threshold
        };

        // Check for metric parameter
        let metric = if args.len() > 3 {
            match &args[3] {
                VectorServiceArg::Metric(m) => *m,
                _ => self.config.default_metric,
            }
        } else {
            self.config.default_metric
        };

        // Check cache first
        let cache_key = format!("similar:{}:{}:{}:{:?}", resource_uri, limit, threshold, metric);
        
        // Execute similarity search with specified metric
        let results = self.vector_store.similarity_search(resource_uri, limit)?;
        
        if self.config.enable_explanations {
            // Return detailed results with explanations
            let mut detailed_results = Vec::new();
            let mut rank = 1;
            
            for (resource, score) in results {
                if score >= threshold {
                    let explanation = Some(format!(
                        "Resource '{}' has a {:?} similarity score of {:.3} with '{}'",
                        resource, metric, score, resource_uri
                    ));
                    
                    detailed_results.push(SimilarityResult {
                        resource: resource.clone(),
                        score,
                        rank,
                        metric,
                        explanation,
                    });
                    rank += 1;
                }
            }
            
            Ok(VectorServiceResult::DetailedSimilarityList(detailed_results))
        } else {
            // Return simple results
            let filtered_results: Vec<(String, f32)> = results
                .into_iter()
                .filter(|(_, score)| *score >= threshold)
                .collect();

            // Cache results
            if self.config.enable_caching {
                self.query_cache.insert(cache_key, filtered_results.clone());
            }

            Ok(VectorServiceResult::SimilarityList(filtered_results))
        }
    }

    fn execute_similarity(&mut self, args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        if args.len() != 2 {
            return Err(anyhow!(
                "similarity function requires exactly two arguments"
            ));
        }

        let uri1 = match &args[0] {
            VectorServiceArg::IRI(uri) => uri,
            _ => return Err(anyhow!("First argument must be a resource URI")),
        };

        let uri2 = match &args[1] {
            VectorServiceArg::IRI(uri) => uri,
            _ => return Err(anyhow!("Second argument must be a resource URI")),
        };

        // Calculate actual similarity between two resources
        let similarity = self.vector_store.calculate_similarity(uri1, uri2)?;

        Ok(VectorServiceResult::Number(similarity))
    }

    fn execute_embed_text(&mut self, args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        if args.len() != 1 {
            return Err(anyhow!("embed_text function requires exactly one argument"));
        }

        let text = match &args[0] {
            VectorServiceArg::String(text) => text,
            _ => return Err(anyhow!("Argument must be a string")),
        };

        let content = EmbeddableContent::Text(text.clone());
        let vector = self.embedding_manager.get_embedding(&content)?;

        Ok(VectorServiceResult::Vector(vector))
    }

    fn execute_search_text(&mut self, args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        if args.is_empty() {
            return Err(anyhow!(
                "search_text function requires at least one argument"
            ));
        }

        let query_text = match &args[0] {
            VectorServiceArg::String(text) => text,
            _ => return Err(anyhow!("First argument must be a string")),
        };

        let limit = if args.len() > 1 {
            match &args[1] {
                VectorServiceArg::Number(n) => *n as usize,
                _ => self.config.default_limit,
            }
        } else {
            self.config.default_limit
        };

        let results = self.vector_store.similarity_search(query_text, limit)?;
        Ok(VectorServiceResult::SimilarityList(results))
    }

    fn execute_cluster(&mut self, args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        if args.is_empty() {
            return Err(anyhow!("cluster function requires at least one argument"));
        }

        let resources_arg = match &args[0] {
            VectorServiceArg::String(resources_list) => resources_list,
            _ => return Err(anyhow!("First argument must be a list of resource URIs")),
        };

        let threshold = if args.len() > 1 {
            match &args[1] {
                VectorServiceArg::Number(t) => *t,
                _ => self.config.default_threshold,
            }
        } else {
            self.config.default_threshold
        };

        // Parse the resources list (simplified - in practice would be more sophisticated)
        let resource_uris: Vec<String> = resources_arg
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        // Get embeddings for all resources
        let mut resources_with_embeddings = Vec::new();
        for uri in &resource_uris {
            if let Ok(results) = self.vector_store.similarity_search(uri, 1) {
                if let Some((found_uri, _)) = results.first() {
                    // Try to get the actual embedding vector
                    // For now, we'll use a placeholder - in practice this would come from the vector store
                    let placeholder_vector = Vector::new(vec![1.0; 128]); // Placeholder
                    resources_with_embeddings.push((found_uri.clone(), placeholder_vector));
                }
            }
        }

        if resources_with_embeddings.is_empty() {
            return Ok(VectorServiceResult::Clusters(Vec::new()));
        }

        // Configure clustering
        let clustering_config = ClusteringConfig {
            algorithm: ClusteringAlgorithm::Similarity,
            similarity_threshold: threshold,
            min_cluster_size: 2,
            ..Default::default()
        };

        let clustering_engine = ClusteringEngine::new(clustering_config);
        let clustering_result = clustering_engine.cluster(&resources_with_embeddings)?;

        // Convert clustering result to SPARQL service result format
        let clusters: Vec<Vec<String>> = clustering_result
            .clusters
            .into_iter()
            .map(|cluster| cluster.members)
            .collect();

        Ok(VectorServiceResult::Clusters(clusters))
    }

    fn execute_vector_similarity(&mut self, args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        if args.len() != 2 {
            return Err(anyhow!(
                "vector_similarity function requires exactly two arguments"
            ));
        }

        let vector1 = match &args[0] {
            VectorServiceArg::Vector(v) => v,
            _ => return Err(anyhow!("First argument must be a vector")),
        };

        let vector2 = match &args[1] {
            VectorServiceArg::Vector(v) => v,
            _ => return Err(anyhow!("Second argument must be a vector")),
        };

        let similarity = vector1.cosine_similarity(vector2)?;
        Ok(VectorServiceResult::Number(similarity))
    }

    fn execute_search(&mut self, args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        if args.is_empty() {
            return Err(anyhow!("search function requires at least one argument"));
        }

        let query_text = match &args[0] {
            VectorServiceArg::String(text) => text,
            _ => return Err(anyhow!("First argument must be a string")),
        };

        let limit = if args.len() > 1 {
            match &args[1] {
                VectorServiceArg::Number(n) => *n as usize,
                _ => self.config.default_limit,
            }
        } else {
            self.config.default_limit
        };

        // Check for threshold parameter
        let threshold = if args.len() > 2 {
            match &args[2] {
                VectorServiceArg::Number(t) => *t,
                _ => self.config.default_threshold,
            }
        } else {
            self.config.default_threshold
        };

        // Check for metric parameter
        let metric = if args.len() > 3 {
            match &args[3] {
                VectorServiceArg::Metric(m) => *m,
                _ => self.config.default_metric,
            }
        } else {
            self.config.default_metric
        };

        // Execute text-based search
        let results = self.vector_store.similarity_search(query_text, limit)?;
        
        if self.config.enable_explanations {
            // Return detailed results with explanations
            let mut detailed_results = Vec::new();
            let mut rank = 1;
            
            for (resource, score) in results {
                if score >= threshold {
                    let explanation = Some(format!(
                        "Text query '{}' matched resource '{}' with {:?} similarity score of {:.3}",
                        query_text, resource, metric, score
                    ));
                    
                    detailed_results.push(SimilarityResult {
                        resource: resource.clone(),
                        score,
                        rank,
                        metric,
                        explanation,
                    });
                    rank += 1;
                }
            }
            
            Ok(VectorServiceResult::DetailedSimilarityList(detailed_results))
        } else {
            let filtered_results: Vec<(String, f32)> = results
                .into_iter()
                .filter(|(_, score)| *score >= threshold)
                .collect();
            Ok(VectorServiceResult::SimilarityList(filtered_results))
        }
    }

    fn execute_search_in(&mut self, args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        if args.len() < 2 {
            return Err(anyhow!("searchIn function requires at least two arguments"));
        }

        let query_text = match &args[0] {
            VectorServiceArg::String(text) => text,
            _ => return Err(anyhow!("First argument must be a string")),
        };

        let graph_uri = match &args[1] {
            VectorServiceArg::IRI(uri) => uri,
            _ => return Err(anyhow!("Second argument must be a graph URI")),
        };

        let limit = if args.len() > 2 {
            match &args[2] {
                VectorServiceArg::Number(n) => *n as usize,
                _ => self.config.default_limit,
            }
        } else {
            self.config.default_limit
        };

        // Use graph-aware search if available
        let results = if let Some(ref graph_search) = self.graph_aware_search {
            // Create graph context for search
            let graph_context = GraphContext {
                primary_graph: graph_uri.clone(),
                additional_graphs: Vec::new(),
                scope: GraphSearchScope::Exact,
                context_weights: HashMap::new(),
            };

            // Perform graph-aware search
            let graph_results = graph_search.search_in_graph(
                &self.vector_store,
                query_text,
                &graph_context,
                limit,
            )?;

            // Convert to simple (String, f32) tuples
            graph_results
                .into_iter()
                .map(|result| (result.resource, result.final_score))
                .collect()
        } else {
            // Fallback to regular search with simple graph filtering
            let all_results = self.vector_store.similarity_search(query_text, limit * 2)?;
            
            // Filter results to only include items from the specified graph
            // This is a simplified approach - in practice would use proper graph membership
            all_results
                .into_iter()
                .filter(|(uri, _)| {
                    // Simple heuristic: check if URI contains graph URI or is prefixed by it
                    uri.starts_with(graph_uri) || uri.contains(&graph_uri.replace("http://", ""))
                })
                .take(limit)
                .collect()
        };
        
        if self.config.enable_explanations {
            let mut detailed_results = Vec::new();
            let mut rank = 1;
            
            for (resource, score) in results {
                let explanation = Some(format!(
                    "Text query '{}' in graph '{}' matched resource '{}' with score {:.3}",
                    query_text, graph_uri, resource, score
                ));
                
                detailed_results.push(SimilarityResult {
                    resource: resource.clone(),
                    score,
                    rank,
                    metric: self.config.default_metric,
                    explanation,
                });
                rank += 1;
            }
            
            Ok(VectorServiceResult::DetailedSimilarityList(detailed_results))
        } else {
            Ok(VectorServiceResult::SimilarityList(results))
        }
    }

    fn execute_search_in_graph(&mut self, args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        if args.len() < 2 {
            return Err(anyhow!(
                "search_in_graph function requires at least two arguments"
            ));
        }

        let query_text = match &args[0] {
            VectorServiceArg::String(text) => text,
            _ => return Err(anyhow!("First argument must be a string")),
        };

        let graph_uri = match &args[1] {
            VectorServiceArg::IRI(uri) => uri,
            _ => return Err(anyhow!("Second argument must be a graph URI")),
        };

        let limit = if args.len() > 2 {
            match &args[2] {
                VectorServiceArg::Number(n) => *n as usize,
                _ => self.config.default_limit,
            }
        } else {
            self.config.default_limit
        };

        // Check for hierarchical search scope parameter
        let scope = if args.len() > 3 {
            match &args[3] {
                VectorServiceArg::String(scope_str) => {
                    match scope_str.to_lowercase().as_str() {
                        "exact" => GraphSearchScope::Exact,
                        "children" | "include_children" => GraphSearchScope::IncludeChildren,
                        "parents" | "include_parents" => GraphSearchScope::IncludeParents,
                        "hierarchy" | "full_hierarchy" => GraphSearchScope::FullHierarchy,
                        "related" => GraphSearchScope::Related,
                        _ => GraphSearchScope::Exact,
                    }
                },
                _ => GraphSearchScope::Exact,
            }
        } else {
            GraphSearchScope::Exact
        };

        // Use graph-aware search if available
        let results = if let Some(ref graph_search) = self.graph_aware_search {
            let graph_context = GraphContext {
                primary_graph: graph_uri.clone(),
                additional_graphs: Vec::new(),
                scope,
                context_weights: HashMap::new(),
            };

            let graph_results = graph_search.search_in_graph(
                &self.vector_store,
                query_text,
                &graph_context,
                limit,
            )?;

            graph_results
                .into_iter()
                .map(|result| (result.resource, result.final_score))
                .collect()
        } else {
            // Fallback implementation
            let mut results = self.vector_store.similarity_search(query_text, limit)?;
            
            // Simple graph filtering
            results.retain(|(uri, _)| uri.starts_with(graph_uri));
            results
        };

        Ok(VectorServiceResult::SimilarityList(results))
    }

    // Helper methods for hybrid queries

    fn execute_symbolic_query(&self, _query: &str) -> Result<Vec<HashMap<String, String>>> {
        // Placeholder for SPARQL execution
        // In a real implementation, this would execute the SPARQL query
        Ok(vec![HashMap::new()])
    }

    fn execute_vector_operations(
        &mut self,
        operations: &[VectorOperation],
        _context: &HashMap<String, String>,
    ) -> Result<Vec<(String, f32)>> {
        let mut results = Vec::new();

        for operation in operations {
            match operation {
                VectorOperation::FindSimilar {
                    resource,
                    limit,
                    threshold: _,
                } => {
                    let search_results = self
                        .vector_store
                        .similarity_search(resource, limit.unwrap_or(self.config.default_limit))?;
                    results.extend(search_results);
                }
                _ => {
                    // Handle other operations
                }
            }
        }

        Ok(results)
    }

    fn calculate_combined_score(&self, _scoring: &ScoringStrategy) -> f32 {
        // Placeholder for combined scoring
        0.8
    }

    /// Get available service functions
    pub fn get_available_functions(&self) -> Vec<&VectorServiceFunction> {
        self.function_registry.values().collect()
    }

    /// Clear query cache
    pub fn clear_cache(&mut self) {
        self.query_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.query_cache.len(), self.config.cache_size)
    }

    /// Calculate similarity between two URIs (simplified implementation)
    fn calculate_uri_similarity(&self, uri1: &str, uri2: &str) -> f32 {
        if uri1 == uri2 {
            return 1.0;
        }

        // Simple Jaccard similarity based on character n-grams
        let ngrams1 = self.generate_ngrams(uri1, 3);
        let ngrams2 = self.generate_ngrams(uri2, 3);

        let intersection: usize = ngrams1.iter().filter(|g| ngrams2.contains(*g)).count();
        let union_size = ngrams1.len() + ngrams2.len() - intersection;

        if union_size == 0 {
            0.0
        } else {
            intersection as f32 / union_size as f32
        }
    }

    /// Generate character n-grams for similarity calculation
    fn generate_ngrams(&self, text: &str, n: usize) -> std::collections::HashSet<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut ngrams = std::collections::HashSet::new();

        if chars.len() >= n {
            for i in 0..=chars.len() - n {
                let ngram: String = chars[i..i + n].iter().collect();
                ngrams.insert(ngram);
            }
        }

        ngrams
    }
}

/// Vector service function arguments
#[derive(Debug, Clone)]
pub enum VectorServiceArg {
    IRI(String),
    String(String),
    Number(f32),
    Vector(Vector),
    Metric(crate::similarity::SimilarityMetric),
    Boolean(bool),
}

/// Enhanced similarity result with explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub resource: String,
    pub score: f32,
    pub rank: usize,
    pub metric: crate::similarity::SimilarityMetric,
    pub explanation: Option<String>,
}

/// Vector service function results
#[derive(Debug, Clone)]
pub enum VectorServiceResult {
    SimilarityList(Vec<(String, f32)>),
    DetailedSimilarityList(Vec<SimilarityResult>),
    Number(f32),
    String(String),
    Vector(Vector),
    Clusters(Vec<Vec<String>>),
    Boolean(bool),
}

/// Enhanced vector query structure for optimization
#[derive(Debug, Clone)]
pub struct VectorQuery {
    pub operation_type: String,
    pub parameters: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
    pub preferred_index: Option<String>,
    pub use_cache: bool,
    pub parallel_execution: bool,
    pub estimated_result_size: Option<usize>,
}

impl VectorQuery {
    pub fn new(operation_type: String) -> Self {
        Self {
            operation_type,
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            preferred_index: None,
            use_cache: false,
            parallel_execution: false,
            estimated_result_size: None,
        }
    }

    pub fn with_parameter(mut self, key: String, value: String) -> Self {
        self.parameters.insert(key, value);
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn cache_key(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        self.operation_type.hash(&mut hasher);
        
        // Create a sorted parameter string for consistent hashing
        let mut params: Vec<(&String, &String)> = self.parameters.iter().collect();
        params.sort_by_key(|(k, _)| *k);
        for (k, v) in params {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        
        format!("query_{:x}", hasher.finish())
    }

    pub fn can_parallelize(&self) -> bool {
        matches!(self.operation_type.as_str(), "similarity_search" | "text_search" | "threshold_search")
    }
}

/// Enhanced vector query result with metadata
#[derive(Debug, Clone)]
pub struct VectorQueryResult {
    pub results: Vec<(String, f32)>,
    pub metadata: HashMap<String, String>,
    pub execution_time: Option<Duration>,
    pub cache_hit: bool,
}

/// Vector operations for hybrid queries
#[derive(Debug, Clone)]
pub enum VectorOperation {
    FindSimilar {
        resource: String,
        limit: Option<usize>,
        threshold: Option<f32>,
    },
    CalculateSimilarity {
        resource1: String,
        resource2: String,
    },
    SearchText {
        query_text: String,
        limit: Option<usize>,
        threshold: Option<f32>,
    },
}

/// Hybrid query combining symbolic and vector operations
#[derive(Debug, Clone)]
pub struct HybridQuery {
    pub symbolic_part: String,
    pub vector_operations: Vec<VectorOperation>,
    pub scoring: ScoringStrategy,
    pub sort_by_score: bool,
}

/// Scoring strategies for hybrid queries
#[derive(Debug, Clone)]
pub enum ScoringStrategy {
    VectorOnly,
    SymbolicOnly,
    Weighted {
        vector_weight: f32,
        symbolic_weight: f32,
    },
    Multiplicative,
    Maximum,
    Minimum,
}

/// Result of a hybrid query
#[derive(Debug, Clone)]
pub struct HybridQueryResult {
    pub symbolic_bindings: HashMap<String, String>,
    pub vector_results: Vec<(String, f32)>,
    pub combined_score: f32,
}

/// Query optimizer for hybrid vector-symbolic queries
pub struct HybridQueryOptimizer {
    config: VectorServiceConfig,
}

impl HybridQueryOptimizer {
    pub fn new(config: VectorServiceConfig) -> Self {
        Self { config }
    }

    /// Optimize a hybrid query for better performance
    pub fn optimize_query(&self, query: &HybridQuery) -> Result<HybridQuery> {
        let mut optimized = query.clone();

        if self.config.enable_optimization {
            // Move vector operations that can be executed early
            optimized = self.reorder_operations(optimized)?;

            // Combine similar vector operations
            optimized = self.combine_operations(optimized)?;

            // Add caching hints
            optimized = self.add_caching_hints(optimized)?;
        }

        Ok(optimized)
    }

    fn reorder_operations(&self, query: HybridQuery) -> Result<HybridQuery> {
        // Placeholder for operation reordering logic
        Ok(query)
    }

    fn combine_operations(&self, query: HybridQuery) -> Result<HybridQuery> {
        // Placeholder for operation combination logic
        Ok(query)
    }

    fn add_caching_hints(&self, query: HybridQuery) -> Result<HybridQuery> {
        // Placeholder for caching hint logic
        Ok(query)
    }
}

/// Vector query builder for constructing complex vector searches
pub struct VectorQueryBuilder {
    operations: Vec<VectorOperation>,
    scoring: ScoringStrategy,
    sort_by_score: bool,
}

impl VectorQueryBuilder {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            scoring: ScoringStrategy::VectorOnly,
            sort_by_score: true,
        }
    }

    pub fn find_similar(
        mut self,
        resource: String,
        limit: Option<usize>,
        threshold: Option<f32>,
    ) -> Self {
        self.operations.push(VectorOperation::FindSimilar {
            resource,
            limit,
            threshold,
        });
        self
    }

    pub fn calculate_similarity(mut self, resource1: String, resource2: String) -> Self {
        self.operations.push(VectorOperation::CalculateSimilarity {
            resource1,
            resource2,
        });
        self
    }

    pub fn search_text(
        mut self,
        query_text: String,
        limit: Option<usize>,
        threshold: Option<f32>,
    ) -> Self {
        self.operations.push(VectorOperation::SearchText {
            query_text,
            limit,
            threshold,
        });
        self
    }

    pub fn with_scoring(mut self, scoring: ScoringStrategy) -> Self {
        self.scoring = scoring;
        self
    }

    pub fn sort_by_score(mut self, sort: bool) -> Self {
        self.sort_by_score = sort;
        self
    }

    pub fn build_hybrid_query(self, symbolic_part: String) -> HybridQuery {
        HybridQuery {
            symbolic_part,
            vector_operations: self.operations,
            scoring: self.scoring,
            sort_by_score: self.sort_by_score,
        }
    }
}

impl Default for VectorQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Vector service registry for managing multiple vector services
pub struct VectorServiceRegistry {
    services: HashMap<String, SparqlVectorService>,
    default_service: Option<String>,
}

impl VectorServiceRegistry {
    pub fn new() -> Self {
        Self {
            services: HashMap::new(),
            default_service: None,
        }
    }

    pub fn register_service(&mut self, name: String, service: SparqlVectorService) {
        if self.default_service.is_none() {
            self.default_service = Some(name.clone());
        }
        self.services.insert(name, service);
    }

    pub fn get_service(&mut self, name: &str) -> Option<&mut SparqlVectorService> {
        self.services.get_mut(name)
    }

    pub fn get_default_service(&mut self) -> Option<&mut SparqlVectorService> {
        if let Some(ref default_name) = self.default_service.clone() {
            self.services.get_mut(default_name)
        } else {
            None
        }
    }

    pub fn execute_service_function(
        &mut self,
        service_name: Option<&str>,
        function_name: &str,
        args: &[VectorServiceArg],
    ) -> Result<VectorServiceResult> {
        let service = if let Some(name) = service_name {
            self.get_service(name)
                .ok_or_else(|| anyhow!("Service '{}' not found", name))?
        } else {
            self.get_default_service()
                .ok_or_else(|| anyhow!("No default service available"))?
        };

        service.execute_function(function_name, args)
    }
}

impl Default for VectorServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}
