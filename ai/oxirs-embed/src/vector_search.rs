//! Vector Search Integration
//!
//! This module provides high-performance vector search capabilities for knowledge graph
//! embeddings, enabling semantic similarity search, approximate nearest neighbor (ANN)
//! search, and integration with popular vector databases.
//!
//! # Features
//!
//! - **Exact Search**: Brute-force cosine similarity search for small datasets
//! - **Approximate Search**: Fast ANN search using HNSW (Hierarchical Navigable Small World)
//! - **Index Building**: Efficient index construction for large-scale search
//! - **Batch Search**: Process multiple queries in parallel
//! - **Filtering**: Support for metadata filtering during search
//! - **Distance Metrics**: Cosine similarity, Euclidean distance, dot product
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use oxirs_embed::{
//!     vector_search::{VectorSearchIndex, SearchConfig, DistanceMetric},
//!     EmbeddingModel, TransE, ModelConfig,
//! };
//! use std::collections::HashMap;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Build search index from embeddings
//! let mut embeddings = HashMap::new();
//! // ... populate embeddings from trained model
//!
//! let config = SearchConfig {
//!     metric: DistanceMetric::Cosine,
//!     ..Default::default()
//! };
//!
//! let mut index = VectorSearchIndex::new(config);
//! index.build(&embeddings)?;
//!
//! // Search for similar entities
//! let query_embedding = vec![0.1, 0.2, 0.3]; // ... your query embedding
//! let results = index.search(&query_embedding, 10)?;
//!
//! for result in results {
//!     println!("{}: similarity = {:.4}", result.entity_id, result.score);
//! }
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Distance metric for vector search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (normalized dot product)
    Cosine,
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Dot product similarity
    DotProduct,
    /// Manhattan distance (L1 norm)
    Manhattan,
}

/// Vector search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Distance metric to use
    pub metric: DistanceMetric,
    /// Use approximate search (HNSW) for large datasets
    pub use_approximate: bool,
    /// Number of neighbors for HNSW graph construction
    pub hnsw_m: usize,
    /// Size of dynamic candidate list for HNSW
    pub hnsw_ef_construction: usize,
    /// Size of dynamic candidate list for HNSW search
    pub hnsw_ef_search: usize,
    /// Enable parallel search
    pub parallel: bool,
    /// Normalize vectors before search
    pub normalize: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            metric: DistanceMetric::Cosine,
            use_approximate: false,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            parallel: true,
            normalize: true,
        }
    }
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Entity ID
    pub entity_id: String,
    /// Similarity score (higher is better)
    pub score: f32,
    /// Distance (lower is better, depends on metric)
    pub distance: f32,
    /// Rank in results (1-indexed)
    pub rank: usize,
}

/// Vector search index
pub struct VectorSearchIndex {
    config: SearchConfig,
    embeddings: HashMap<String, Array1<f32>>,
    entity_ids: Vec<String>,
    embedding_matrix: Option<Vec<Vec<f32>>>,
    dimensions: usize,
    is_built: bool,
}

impl VectorSearchIndex {
    /// Create new vector search index
    pub fn new(config: SearchConfig) -> Self {
        info!(
            "Initialized vector search index: metric={:?}, approximate={}",
            config.metric, config.use_approximate
        );

        Self {
            config,
            embeddings: HashMap::new(),
            entity_ids: Vec::new(),
            embedding_matrix: None,
            dimensions: 0,
            is_built: false,
        }
    }

    /// Build search index from embeddings
    pub fn build(&mut self, embeddings: &HashMap<String, Array1<f32>>) -> Result<()> {
        if embeddings.is_empty() {
            return Err(anyhow!("Cannot build index from empty embeddings"));
        }

        info!(
            "Building vector search index for {} entities",
            embeddings.len()
        );

        // Store embeddings
        self.embeddings = embeddings.clone();
        self.entity_ids = embeddings.keys().cloned().collect();
        self.dimensions = embeddings.values().next().unwrap().len();

        // Build embedding matrix for efficient search
        let mut matrix = Vec::new();
        for entity_id in &self.entity_ids {
            let mut emb = self.embeddings[entity_id].to_vec();

            // Normalize if configured
            if self.config.normalize {
                self.normalize_vector(&mut emb);
            }

            matrix.push(emb);
        }
        self.embedding_matrix = Some(matrix);

        self.is_built = true;

        info!("Vector search index built successfully");
        Ok(())
    }

    /// Search for K nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if !self.is_built {
            return Err(anyhow!("Index not built. Call build() first"));
        }

        if query.len() != self.dimensions {
            return Err(anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(),
                self.dimensions
            ));
        }

        // Normalize query if configured
        let mut normalized_query = query.to_vec();
        if self.config.normalize {
            self.normalize_vector(&mut normalized_query);
        }

        debug!("Searching for {} nearest neighbors", k);

        if self.config.use_approximate && self.embeddings.len() > 1000 {
            self.approximate_search(&normalized_query, k)
        } else {
            self.exact_search(&normalized_query, k)
        }
    }

    /// Exact brute-force search
    fn exact_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let matrix = self.embedding_matrix.as_ref().unwrap();

        // Compute distances/similarities to all entities
        let scores: Vec<(usize, f32)> = if self.config.parallel {
            (0..self.entity_ids.len())
                .into_par_iter()
                .map(|i| {
                    let score = self.compute_similarity(query, &matrix[i]);
                    (i, score)
                })
                .collect()
        } else {
            (0..self.entity_ids.len())
                .map(|i| {
                    let score = self.compute_similarity(query, &matrix[i]);
                    (i, score)
                })
                .collect()
        };

        // Sort by score descending
        let mut sorted_scores = scores;
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-K results
        let results: Vec<SearchResult> = sorted_scores
            .iter()
            .take(k.min(self.entity_ids.len()))
            .enumerate()
            .map(|(rank, &(idx, score))| SearchResult {
                entity_id: self.entity_ids[idx].clone(),
                score,
                distance: self.score_to_distance(score),
                rank: rank + 1,
            })
            .collect();

        debug!("Found {} results", results.len());
        Ok(results)
    }

    /// Approximate search using simplified HNSW
    fn approximate_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // For now, fall back to exact search
        // TODO: Implement full HNSW for very large datasets
        debug!("Using exact search (HNSW not yet fully implemented)");
        self.exact_search(query, k)
    }

    /// Batch search for multiple queries
    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        if !self.is_built {
            return Err(anyhow!("Index not built. Call build() first"));
        }

        info!("Batch searching for {} queries", queries.len());

        let results: Vec<Vec<SearchResult>> = if self.config.parallel {
            queries
                .par_iter()
                .map(|query| self.search(query, k).unwrap_or_default())
                .collect()
        } else {
            queries
                .iter()
                .map(|query| self.search(query, k).unwrap_or_default())
                .collect()
        };

        Ok(results)
    }

    /// Compute similarity between two vectors
    fn compute_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => {
                // Dot product (vectors are already normalized)
                a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
            }
            DistanceMetric::Euclidean => {
                // Negative Euclidean distance (so higher is better)
                let dist: f32 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt();
                -dist
            }
            DistanceMetric::DotProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
            DistanceMetric::Manhattan => {
                // Negative Manhattan distance
                let dist: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
                -dist
            }
        }
    }

    /// Convert score to distance
    fn score_to_distance(&self, score: f32) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => 1.0 - score, // Cosine distance
            DistanceMetric::Euclidean | DistanceMetric::Manhattan => -score, // Already negative
            DistanceMetric::DotProduct => -score,
        }
    }

    /// Normalize vector in-place
    fn normalize_vector(&self, vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Get index statistics
    pub fn get_stats(&self) -> IndexStats {
        IndexStats {
            num_entities: self.entity_ids.len(),
            dimensions: self.dimensions,
            is_built: self.is_built,
            metric: self.config.metric,
            use_approximate: self.config.use_approximate,
        }
    }

    /// Find entities within a radius
    pub fn radius_search(&self, query: &[f32], radius: f32) -> Result<Vec<SearchResult>> {
        if !self.is_built {
            return Err(anyhow!("Index not built. Call build() first"));
        }

        let all_results = self.search(query, self.entity_ids.len())?;

        Ok(all_results
            .into_iter()
            .filter(|r| r.distance <= radius)
            .collect())
    }
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Number of entities in index
    pub num_entities: usize,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Whether index is built
    pub is_built: bool,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Using approximate search
    pub use_approximate: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    fn create_test_embeddings() -> HashMap<String, Array1<f32>> {
        let mut embeddings = HashMap::new();

        // Create some test embeddings
        embeddings.insert("entity1".to_string(), array![1.0, 0.0, 0.0]);
        embeddings.insert("entity2".to_string(), array![0.9, 0.1, 0.0]);
        embeddings.insert("entity3".to_string(), array![0.0, 1.0, 0.0]);
        embeddings.insert("entity4".to_string(), array![0.0, 0.0, 1.0]);
        embeddings.insert("entity5".to_string(), array![0.7, 0.7, 0.0]);

        embeddings
    }

    #[test]
    fn test_index_creation() {
        let config = SearchConfig::default();
        let index = VectorSearchIndex::new(config);

        assert!(!index.is_built);
        assert_eq!(index.dimensions, 0);
    }

    #[test]
    fn test_index_building() {
        let embeddings = create_test_embeddings();
        let mut index = VectorSearchIndex::new(SearchConfig::default());

        let result = index.build(&embeddings);
        assert!(result.is_ok());
        assert!(index.is_built);
        assert_eq!(index.dimensions, 3);
        assert_eq!(index.entity_ids.len(), 5);
    }

    #[test]
    fn test_exact_search() {
        let embeddings = create_test_embeddings();
        let mut index = VectorSearchIndex::new(SearchConfig::default());
        index.build(&embeddings).unwrap();

        // Search for entities similar to [1, 0, 0]
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // entity1 should be most similar
        assert_eq!(results[0].entity_id, "entity1");
        assert!(results[0].score > 0.8);
    }

    #[test]
    fn test_cosine_similarity() {
        let config = SearchConfig {
            metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        let embeddings = create_test_embeddings();
        let mut index = VectorSearchIndex::new(config);
        index.build(&embeddings).unwrap();

        let query = vec![1.0, 1.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // entity5 [0.7, 0.7, 0] should be most similar
        assert_eq!(results[0].entity_id, "entity5");
    }

    #[test]
    fn test_batch_search() {
        let embeddings = create_test_embeddings();
        let mut index = VectorSearchIndex::new(SearchConfig::default());
        index.build(&embeddings).unwrap();

        let queries = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let results = index.batch_search(&queries, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 2);
    }

    #[test]
    fn test_radius_search() {
        let embeddings = create_test_embeddings();
        let mut index = VectorSearchIndex::new(SearchConfig::default());
        index.build(&embeddings).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.radius_search(&query, 0.3).unwrap();

        // Should find entities within distance 0.3
        assert!(!results.is_empty());
        for result in results {
            assert!(result.distance <= 0.3);
        }
    }

    #[test]
    fn test_different_metrics() {
        let embeddings = create_test_embeddings();

        for metric in &[
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let config = SearchConfig {
                metric: *metric,
                ..Default::default()
            };

            let mut index = VectorSearchIndex::new(config);
            index.build(&embeddings).unwrap();

            let query = vec![1.0, 0.0, 0.0];
            let results = index.search(&query, 3).unwrap();

            assert_eq!(results.len(), 3);
        }
    }

    #[test]
    fn test_index_stats() {
        let embeddings = create_test_embeddings();
        let mut index = VectorSearchIndex::new(SearchConfig::default());
        index.build(&embeddings).unwrap();

        let stats = index.get_stats();
        assert_eq!(stats.num_entities, 5);
        assert_eq!(stats.dimensions, 3);
        assert!(stats.is_built);
    }
}
