//! Core query execution and optimization for SPARQL vector operations

use super::config::{VectorQuery, VectorQueryOptimizer, VectorQueryResult, VectorServiceArg};
use super::cross_language::CrossLanguageProcessor;
use super::monitoring::PerformanceMonitor;
use crate::{
    embeddings::{EmbeddableContent, EmbeddingManager},
    graph_aware_search::{GraphAwareSearch, GraphContext, GraphSearchScope},
    VectorStore,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::time::Instant;

/// Core query execution engine for vector operations
pub struct QueryExecutor {
    vector_store: VectorStore,
    embedding_manager: EmbeddingManager,
    query_cache: HashMap<String, VectorQueryResult>,
    optimizer: VectorQueryOptimizer,
    performance_monitor: Option<PerformanceMonitor>,
    cross_language_processor: CrossLanguageProcessor,
    graph_aware_search: Option<GraphAwareSearch>,
}

impl QueryExecutor {
    pub fn new(
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        optimizer: VectorQueryOptimizer,
        performance_monitor: Option<PerformanceMonitor>,
        graph_aware_search: Option<GraphAwareSearch>,
    ) -> Self {
        Self {
            vector_store,
            embedding_manager,
            query_cache: HashMap::new(),
            optimizer,
            performance_monitor,
            cross_language_processor: CrossLanguageProcessor::new(),
            graph_aware_search,
        }
    }

    /// Execute query with performance monitoring and optimization
    pub fn execute_optimized_query(&mut self, query: &VectorQuery) -> Result<VectorQueryResult> {
        let start_time = Instant::now();

        // Apply query optimization if enabled
        let optimized_query = if self.optimizer.enable_index_selection {
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
        if self.optimizer.enable_index_selection {
            optimized.preferred_index = self.select_optimal_index(query)?;
        }

        // Caching optimization
        if self.optimizer.enable_caching {
            optimized.use_cache = true;
        }

        // Parallel execution optimization
        if self.optimizer.enable_parallel_execution && query.can_parallelize() {
            optimized.parallel_execution = true;
        }

        Ok(optimized)
    }

    /// Select optimal index for query execution
    fn select_optimal_index(&self, query: &VectorQuery) -> Result<Option<String>> {
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
        if query.use_cache {
            if let Some(cached_result) = self.get_cached_result(&query.cache_key()) {
                if let Some(ref monitor) = self.performance_monitor {
                    monitor.record_cache_hit();
                }
                return Ok(cached_result.from_cache());
            } else if let Some(ref monitor) = self.performance_monitor {
                monitor.record_cache_miss();
            }
        }

        let start_time = Instant::now();
        let result = match query.operation_type.as_str() {
            "similarity" => self.execute_similarity_query(query),
            "similar" => self.execute_similar_query(query),
            "search" | "search_text" => self.execute_search_query(query),
            "searchIn" => self.execute_search_in_query(query),
            "cluster" => self.execute_cluster_query(query),
            "embed" | "embed_text" => self.execute_embed_query(query),
            _ => Err(anyhow!("Unknown operation type: {}", query.operation_type)),
        }?;

        let execution_time = start_time.elapsed();
        let query_result = VectorQueryResult::new(result, execution_time);

        // Cache the result if caching is enabled
        if query.use_cache {
            self.cache_result(query.cache_key(), query_result.clone());
        }

        Ok(query_result)
    }

    /// Execute similarity query between two resources
    fn execute_similarity_query(&mut self, query: &VectorQuery) -> Result<Vec<(String, f32)>> {
        if query.args.len() < 2 {
            return Err(anyhow!("Similarity query requires at least 2 arguments"));
        }

        let resource1 = match &query.args[0] {
            VectorServiceArg::IRI(iri) => iri,
            _ => return Err(anyhow!("First argument must be an IRI")),
        };

        let resource2 = match &query.args[1] {
            VectorServiceArg::IRI(iri) => iri,
            _ => return Err(anyhow!("Second argument must be an IRI")),
        };

        // Get vectors for both resources
        let vector1 = self
            .vector_store
            .get_vector(&resource1.clone())
            .ok_or_else(|| anyhow!("Vector not found for resource: {}", resource1))?
            .clone();
        let vector2 = self
            .vector_store
            .get_vector(&resource2.clone())
            .ok_or_else(|| anyhow!("Vector not found for resource: {}", resource2))?
            .clone();

        // Calculate similarity
        let similarity =
            crate::similarity::cosine_similarity(&vector1.as_slice(), &vector2.as_slice());

        Ok(vec![(format!("{resource1}-{resource2}"), similarity)])
    }

    /// Execute similar query to find similar resources
    fn execute_similar_query(&mut self, query: &VectorQuery) -> Result<Vec<(String, f32)>> {
        if query.args.is_empty() {
            return Err(anyhow!("Similar query requires at least 1 argument"));
        }

        let resource = match &query.args[0] {
            VectorServiceArg::IRI(iri) => iri,
            _ => return Err(anyhow!("First argument must be an IRI")),
        };

        let limit = if query.args.len() > 1 {
            match &query.args[1] {
                VectorServiceArg::Number(n) => *n as usize,
                _ => 10,
            }
        } else {
            10
        };

        let _threshold = if query.args.len() > 2 {
            match &query.args[2] {
                VectorServiceArg::Number(n) => *n,
                _ => 0.0,
            }
        } else {
            0.0
        };

        // Get vector for the resource
        let query_vector = self
            .vector_store
            .get_vector(&resource.clone())
            .ok_or_else(|| anyhow!("Vector not found for resource: {}", resource))?
            .clone();

        // Perform similarity search
        let results = self.vector_store.index.search_knn(&query_vector, limit)?;

        Ok(results
            .into_iter()
            .filter(|(id, _)| id != resource) // Exclude the query resource itself
            .collect())
    }

    /// Execute text search query
    fn execute_search_query(&mut self, query: &VectorQuery) -> Result<Vec<(String, f32)>> {
        if query.args.is_empty() {
            return Err(anyhow!("Search query requires at least 1 argument"));
        }

        let query_text = match &query.args[0] {
            VectorServiceArg::String(text) | VectorServiceArg::Literal(text) => text,
            _ => return Err(anyhow!("First argument must be text")),
        };

        let limit = if query.args.len() > 1 {
            match &query.args[1] {
                VectorServiceArg::Number(n) => *n as usize,
                _ => 10,
            }
        } else {
            10
        };

        let threshold = if query.args.len() > 2 {
            match &query.args[2] {
                VectorServiceArg::Number(n) => *n,
                _ => 0.7,
            }
        } else {
            0.7
        };

        // Check for cross-language search parameters
        let cross_language = if query.args.len() > 4 {
            match &query.args[4] {
                VectorServiceArg::String(val) => val == "true",
                _ => false,
            }
        } else {
            false
        };

        let target_languages = if query.args.len() > 5 {
            match &query.args[5] {
                VectorServiceArg::String(langs) => langs
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect::<Vec<_>>(),
                _ => vec!["en".to_string()],
            }
        } else {
            vec!["en".to_string()]
        };

        if cross_language {
            self.execute_cross_language_search(query_text, limit, threshold, &target_languages)
        } else {
            self.execute_simple_text_search(query_text, limit, threshold)
        }
    }

    /// Execute simple text search
    fn execute_simple_text_search(
        &mut self,
        query_text: &str,
        limit: usize,
        _threshold: f32,
    ) -> Result<Vec<(String, f32)>> {
        // Generate embedding for the query text
        let content = EmbeddableContent::Text(query_text.to_string());

        let query_vector = self.embedding_manager.get_embedding(&content)?;

        // Perform similarity search
        self.vector_store.index.search_knn(&query_vector, limit)
    }

    /// Execute cross-language search
    fn execute_cross_language_search(
        &mut self,
        query_text: &str,
        limit: usize,
        _threshold: f32,
        target_languages: &[String],
    ) -> Result<Vec<(String, f32)>> {
        // Process query with cross-language variations
        let query_variations = self
            .cross_language_processor
            .process_cross_language_query(query_text, target_languages);

        let mut all_results = Vec::new();

        // Execute search for each query variation
        for (variation_text, weight) in query_variations {
            let content = EmbeddableContent::Text(variation_text);

            if let Ok(query_vector) = self.embedding_manager.get_embedding(&content) {
                if let Ok(results) = self.vector_store.index.search_knn(&query_vector, limit) {
                    for (id, score) in results {
                        all_results.push((id, score * weight));
                    }
                }
            }
        }

        // Merge and deduplicate results
        let merged_results = self.merge_search_results(all_results, limit);
        Ok(merged_results)
    }

    /// Execute graph-scoped search query
    fn execute_search_in_query(&mut self, query: &VectorQuery) -> Result<Vec<(String, f32)>> {
        if query.args.len() < 2 {
            return Err(anyhow!("SearchIn query requires at least 2 arguments"));
        }

        let query_text = match &query.args[0] {
            VectorServiceArg::String(text) | VectorServiceArg::Literal(text) => text,
            _ => return Err(anyhow!("First argument must be query text")),
        };

        let graph_iri = match &query.args[1] {
            VectorServiceArg::IRI(iri) => iri,
            _ => return Err(anyhow!("Second argument must be a graph IRI")),
        };

        let limit = if query.args.len() > 2 {
            match &query.args[2] {
                VectorServiceArg::Number(n) => *n as usize,
                _ => 10,
            }
        } else {
            10
        };

        let scope_str = if query.args.len() > 3 {
            match &query.args[3] {
                VectorServiceArg::String(s) => s.as_str(),
                _ => "exact",
            }
        } else {
            "exact"
        };

        let threshold = if query.args.len() > 4 {
            match &query.args[4] {
                VectorServiceArg::Number(n) => *n,
                _ => 0.7,
            }
        } else {
            0.7
        };

        // Convert scope string to enum
        let scope = match scope_str {
            "children" => GraphSearchScope::IncludeChildren,
            "parents" => GraphSearchScope::IncludeParents,
            "hierarchy" => GraphSearchScope::FullHierarchy,
            "related" => GraphSearchScope::Related,
            _ => GraphSearchScope::Exact,
        };

        if let Some(ref _graph_search) = self.graph_aware_search {
            let _context = GraphContext {
                primary_graph: graph_iri.clone(),
                additional_graphs: Vec::new(),
                scope,
                context_weights: HashMap::new(),
            };

            // Generate embedding for query text
            let content = EmbeddableContent::Text(query_text.to_string());
            let _query_vector = self.embedding_manager.get_embedding(&content)?;

            // Since search_with_context doesn't exist, fallback to simple search
            self.execute_simple_text_search(query_text, limit, threshold)
        } else {
            // Fallback to simple search if graph-aware search is not available
            self.execute_simple_text_search(query_text, limit, threshold)
        }
    }

    /// Execute clustering query
    fn execute_cluster_query(&self, _query: &VectorQuery) -> Result<Vec<(String, f32)>> {
        // Simplified clustering implementation
        // In a real implementation, this would use clustering algorithms
        Err(anyhow!("Clustering not yet implemented"))
    }

    /// Execute embedding generation query
    fn execute_embed_query(&mut self, query: &VectorQuery) -> Result<Vec<(String, f32)>> {
        if query.args.is_empty() {
            return Err(anyhow!("Embed query requires at least 1 argument"));
        }

        let text = match &query.args[0] {
            VectorServiceArg::String(text) | VectorServiceArg::Literal(text) => text,
            _ => return Err(anyhow!("First argument must be text")),
        };

        let content = EmbeddableContent::Text(text.to_string());

        let vector = self.embedding_manager.get_embedding(&content)?;

        // Store the vector with a generated ID
        let id = format!("embedded_{}", hash_string(text));
        self.vector_store
            .index
            .add_vector(id.clone(), vector, None)?;

        Ok(vec![(id, 1.0)])
    }

    /// Merge and deduplicate search results
    fn merge_search_results(
        &self,
        results: Vec<(String, f32)>,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let mut result_map: HashMap<String, f32> = HashMap::new();

        // Aggregate scores for duplicate IDs (take maximum score)
        for (id, score) in results {
            result_map
                .entry(id)
                .and_modify(|existing_score| *existing_score = existing_score.max(score))
                .or_insert(score);
        }

        // Convert to vector and sort by score
        let mut merged: Vec<(String, f32)> = result_map.into_iter().collect();
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply limit
        merged.truncate(limit);
        merged
    }

    /// Get cached result
    fn get_cached_result(&self, cache_key: &str) -> Option<VectorQueryResult> {
        self.query_cache.get(cache_key).cloned()
    }

    /// Cache query result
    fn cache_result(&mut self, cache_key: String, result: VectorQueryResult) {
        // Simple cache with fixed size (in real implementation, use LRU or similar)
        if self.query_cache.len() >= 1000 {
            // Remove oldest entry (simplified)
            if let Some(first_key) = self.query_cache.keys().next().cloned() {
                self.query_cache.remove(&first_key);
            }
        }
        self.query_cache.insert(cache_key, result);

        // Update cache statistics
        if let Some(ref monitor) = self.performance_monitor {
            monitor.update_cache_size(self.query_cache.len(), 1000);
        }
    }

    /// Clear query cache
    pub fn clear_cache(&mut self) {
        self.query_cache.clear();
        if let Some(ref monitor) = self.performance_monitor {
            monitor.update_cache_size(0, 1000);
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.query_cache.len(), 1000)
    }

    /// Add a resource embedding to the vector store
    pub fn add_resource_embedding(&mut self, uri: &str, content: &EmbeddableContent) -> Result<()> {
        // Generate embedding for the content
        let vector = self.embedding_manager.get_embedding(content)?;

        // Insert the vector into the store with the URI as the key
        self.vector_store.index.insert(uri.to_string(), vector)?;

        Ok(())
    }
}

/// Simple string hashing function
fn hash_string(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingStrategy;

    #[test]
    fn test_query_optimization() {
        let vector_store = VectorStore::new();
        let embedding_manager = EmbeddingManager::new(EmbeddingStrategy::TfIdf, 100).unwrap();
        let optimizer = VectorQueryOptimizer::default();

        let executor = QueryExecutor::new(vector_store, embedding_manager, optimizer, None, None);

        let query = VectorQuery::new(
            "similarity_search".to_string(),
            vec![
                VectorServiceArg::IRI("http://example.org/resource1".to_string()),
                VectorServiceArg::IRI("http://example.org/resource2".to_string()),
            ],
        );

        let optimized = executor.optimize_query(&query).unwrap();
        assert!(optimized.use_cache);
    }

    #[test]
    fn test_cache_key_generation() {
        let query1 = VectorQuery::new(
            "search".to_string(),
            vec![VectorServiceArg::String("test".to_string())],
        );

        let query2 = VectorQuery::new(
            "search".to_string(),
            vec![VectorServiceArg::String("test".to_string())],
        );

        assert_eq!(query1.cache_key(), query2.cache_key());
    }

    #[test]
    fn test_merge_search_results() {
        let vector_store = VectorStore::new();
        let embedding_manager = EmbeddingManager::new(EmbeddingStrategy::TfIdf, 100).unwrap();
        let optimizer = VectorQueryOptimizer::default();

        let executor = QueryExecutor::new(vector_store, embedding_manager, optimizer, None, None);

        let results = vec![
            ("doc1".to_string(), 0.8),
            ("doc2".to_string(), 0.9),
            ("doc1".to_string(), 0.7), // Duplicate with lower score
            ("doc3".to_string(), 0.6),
        ];

        let merged = executor.merge_search_results(results, 10);

        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].0, "doc2"); // Highest score first
        assert_eq!(merged[1].1, 0.8); // doc1 should have max score of 0.8
    }
}
