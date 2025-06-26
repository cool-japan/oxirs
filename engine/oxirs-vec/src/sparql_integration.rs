//! SPARQL integration for vector search and hybrid symbolic-vector queries

use crate::{
    embeddings::{EmbeddableContent, EmbeddingManager, EmbeddingStrategy},
    Vector, VectorStore,
};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SPARQL vector service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorServiceConfig {
    /// Service namespace URI
    pub service_uri: String,
    /// Default similarity threshold
    pub default_threshold: f32,
    /// Default number of results to return
    pub default_limit: usize,
    /// Enable caching of vector search results
    pub enable_caching: bool,
    /// Cache size for search results
    pub cache_size: usize,
    /// Enable query optimization
    pub enable_optimization: bool,
}

impl Default for VectorServiceConfig {
    fn default() -> Self {
        Self {
            service_uri: "http://oxirs.org/vec/".to_string(),
            default_threshold: 0.7,
            default_limit: 10,
            enable_caching: true,
            cache_size: 1000,
            enable_optimization: true,
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

/// SPARQL vector service implementation
pub struct SparqlVectorService {
    config: VectorServiceConfig,
    vector_store: VectorStore,
    embedding_manager: EmbeddingManager,
    function_registry: HashMap<String, VectorServiceFunction>,
    query_cache: HashMap<String, Vec<(String, f32)>>,
}

impl SparqlVectorService {
    pub fn new(config: VectorServiceConfig, embedding_strategy: EmbeddingStrategy) -> Result<Self> {
        let vector_store = VectorStore::new();
        let embedding_manager = EmbeddingManager::new(embedding_strategy, 1000)?;

        let mut service = Self {
            config,
            vector_store,
            embedding_manager,
            function_registry: HashMap::new(),
            query_cache: HashMap::new(),
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
    }

    /// Register a custom vector service function
    pub fn register_function(&mut self, function: VectorServiceFunction) {
        self.function_registry
            .insert(function.name.clone(), function);
    }

    /// Execute a vector service function call
    pub fn execute_function(
        &mut self,
        function_name: &str,
        args: &[VectorServiceArg],
    ) -> Result<VectorServiceResult> {
        match function_name {
            "similar" => self.execute_similar(args),
            "similarity" => self.execute_similarity(args),
            "embed_text" => self.execute_embed_text(args),
            "search_text" => self.execute_search_text(args),
            "cluster" => self.execute_cluster(args),
            "vector_similarity" => self.execute_vector_similarity(args),
            "search_in_graph" => self.execute_search_in_graph(args),
            _ => Err(anyhow!(
                "Unknown vector service function: {}",
                function_name
            )),
        }
    }

    /// Add resource embedding to the vector store
    pub fn add_resource_embedding(&mut self, uri: &str, content: &EmbeddableContent) -> Result<()> {
        let vector = self.embedding_manager.get_embedding(content)?;
        self.vector_store
            .index_resource(uri.to_string(), &content.to_text())?;
        Ok(())
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

        // Check cache first
        let cache_key = format!("similar:{}:{}:{}", resource_uri, limit, threshold);
        if self.config.enable_caching {
            if let Some(cached_results) = self.query_cache.get(&cache_key) {
                return Ok(VectorServiceResult::SimilarityList(cached_results.clone()));
            }
        }

        // Execute similarity search
        let results = self.vector_store.similarity_search(resource_uri, limit)?;
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

    fn execute_cluster(&mut self, _args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        // Placeholder implementation for clustering
        let clusters = vec![
            vec!["cluster1_item1".to_string(), "cluster1_item2".to_string()],
            vec!["cluster2_item1".to_string(), "cluster2_item2".to_string()],
        ];
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

        // For now, just perform regular text search and add graph context
        let mut results = self.vector_store.similarity_search(query_text, limit)?;
        
        // Filter results to only include items from the specified graph
        // In a real implementation, this would check graph membership
        results.retain(|(uri, _)| uri.starts_with(graph_uri));

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
}

/// Vector service function results
#[derive(Debug, Clone)]
pub enum VectorServiceResult {
    SimilarityList(Vec<(String, f32)>),
    Number(f32),
    String(String),
    Vector(Vector),
    Clusters(Vec<Vec<String>>),
    Boolean(bool),
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
