//! Vector Store for Efficient Similarity Search
//!
//! This module provides high-performance vector storage and similarity search
//! capabilities optimized for knowledge graph embeddings and AI applications.

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Vector store trait for similarity search
#[async_trait::async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert a vector with metadata
    async fn insert(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()>;

    /// Insert multiple vectors
    async fn insert_batch(&self, vectors: Vec<VectorData>) -> Result<()>;

    /// Search for similar vectors
    async fn search(&self, query: &VectorQuery) -> Result<Vec<(String, f32)>>;

    /// Get vector by ID
    async fn get(&self, id: &str) -> Result<Option<VectorData>>;

    /// Delete vector by ID
    async fn delete(&self, id: &str) -> Result<bool>;

    /// Update vector
    async fn update(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()>;

    /// Get store size
    fn size(&self) -> usize;

    /// Build index for faster search
    async fn build_index(&self) -> Result<()>;

    /// Get store statistics
    async fn get_statistics(&self) -> Result<VectorStoreStats>;
}

/// Vector data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorData {
    /// Unique identifier
    pub id: String,

    /// Vector values
    pub vector: Vec<f32>,

    /// Optional metadata
    pub metadata: Option<HashMap<String, String>>,

    /// Timestamp when inserted
    pub timestamp: std::time::SystemTime,
}

/// Vector query parameters
#[derive(Debug, Clone)]
pub struct VectorQuery {
    /// Query vector
    pub vector: Vec<f32>,

    /// Number of results to return
    pub k: usize,

    /// Distance metric
    pub metric: Option<SimilarityMetric>,

    /// Include metadata in results
    pub include_metadata: bool,

    /// Filter conditions
    pub filters: Option<Vec<Filter>>,

    /// Minimum similarity threshold
    pub min_similarity: Option<f32>,
}

/// Filter conditions for vector search
#[derive(Debug, Clone)]
pub struct Filter {
    /// Field name
    pub field: String,

    /// Filter operation
    pub operation: FilterOperation,

    /// Filter value
    pub value: String,
}

/// Filter operations
#[derive(Debug, Clone)]
pub enum FilterOperation {
    Equals,
    NotEquals,
    Contains,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    In(Vec<String>),
    NotIn(Vec<String>),
}

/// Similarity metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Cosine similarity
    Cosine,

    /// Euclidean distance
    Euclidean,

    /// Manhattan distance (L1)
    Manhattan,

    /// Dot product
    DotProduct,

    /// Jaccard similarity
    Jaccard,

    /// Hamming distance
    Hamming,
}

impl std::fmt::Display for SimilarityMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimilarityMetric::Cosine => write!(f, "cosine"),
            SimilarityMetric::Euclidean => write!(f, "euclidean"),
            SimilarityMetric::Manhattan => write!(f, "manhattan"),
            SimilarityMetric::DotProduct => write!(f, "dot_product"),
            SimilarityMetric::Jaccard => write!(f, "jaccard"),
            SimilarityMetric::Hamming => write!(f, "hamming"),
        }
    }
}

/// Vector store statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreStats {
    /// Total number of vectors
    pub total_vectors: usize,

    /// Vector dimension
    pub dimension: usize,

    /// Index type
    pub index_type: String,

    /// Index build time
    pub index_build_time: std::time::Duration,

    /// Memory usage in bytes
    pub memory_usage: usize,

    /// Average query time
    pub avg_query_time: std::time::Duration,

    /// Cache hit rate
    pub cache_hit_rate: f32,
}

/// In-memory vector store with various indexing strategies
pub struct InMemoryVectorStore {
    /// Vector storage
    vectors: Arc<DashMap<String, VectorData>>,

    /// Vector index for fast similarity search
    index: Arc<RwLock<Option<Box<dyn VectorIndex>>>>,

    /// Configuration
    config: VectorStoreConfig,

    /// Query cache
    query_cache: Arc<DashMap<String, Vec<(String, f32)>>>,

    /// Statistics
    stats: Arc<RwLock<VectorStoreStats>>,
}

/// Vector store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Vector dimension
    pub dimension: usize,

    /// Default similarity metric
    pub default_metric: SimilarityMetric,

    /// Index type
    pub index_type: IndexType,

    /// Enable query caching
    pub enable_cache: bool,

    /// Cache size limit
    pub cache_size: usize,

    /// Cache TTL in seconds
    pub cache_ttl: u64,

    /// Batch insert size
    pub batch_size: usize,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            default_metric: SimilarityMetric::Cosine,
            index_type: IndexType::Flat,
            enable_cache: true,
            cache_size: 10000,
            cache_ttl: 3600,
            batch_size: 1000,
        }
    }
}

/// Vector index types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    /// Flat index (brute force search)
    Flat,

    /// HNSW (Hierarchical Navigable Small World)
    HNSW {
        max_connections: usize,
        ef_construction: usize,
        ef_search: usize,
    },

    /// IVF (Inverted File)
    IVF {
        num_clusters: usize,
        num_probes: usize,
    },

    /// LSH (Locality Sensitive Hashing)
    LSH {
        num_tables: usize,
        hash_length: usize,
    },

    /// Product Quantization
    PQ {
        num_subquantizers: usize,
        bits_per_subquantizer: usize,
    },
}

/// Vector index trait
#[async_trait::async_trait]
pub trait VectorIndex: Send + Sync {
    /// Build index from vectors
    async fn build(&mut self, vectors: &DashMap<String, VectorData>) -> Result<()>;

    /// Search for similar vectors
    async fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>>;

    /// Add vector to index
    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()>;

    /// Remove vector from index
    async fn remove(&mut self, id: &str) -> Result<()>;

    /// Get index statistics
    fn get_stats(&self) -> IndexStats;
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub index_type: String,
    pub num_vectors: usize,
    pub build_time: std::time::Duration,
    pub memory_usage: usize,
}

/// Flat index implementation (brute force)
pub struct FlatIndex {
    /// Vector storage reference
    vectors: HashMap<String, Vec<f32>>,

    /// Statistics
    stats: IndexStats,
}

impl FlatIndex {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            stats: IndexStats {
                index_type: "Flat".to_string(),
                num_vectors: 0,
                build_time: std::time::Duration::from_secs(0),
                memory_usage: 0,
            },
        }
    }
}

#[async_trait::async_trait]
impl VectorIndex for FlatIndex {
    async fn build(&mut self, vectors: &DashMap<String, VectorData>) -> Result<()> {
        let start = std::time::Instant::now();

        self.vectors.clear();
        for entry in vectors.iter() {
            self.vectors
                .insert(entry.key().clone(), entry.value().vector.clone());
        }

        self.stats.num_vectors = self.vectors.len();
        self.stats.build_time = start.elapsed();
        self.stats.memory_usage = self.vectors.len()
            * self
                .vectors
                .values()
                .next()
                .map(|v| v.len() * 4)
                .unwrap_or(0);

        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>> {
        let mut similarities = Vec::new();

        for (id, vector) in &self.vectors {
            let similarity = compute_similarity(query, vector, metric)?;
            similarities.push((id.clone(), similarity));
        }

        // Sort by similarity (higher is better for most metrics)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Return top k results
        similarities.truncate(k);
        Ok(similarities)
    }

    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        self.vectors.insert(id, vector);
        self.stats.num_vectors = self.vectors.len();
        Ok(())
    }

    async fn remove(&mut self, id: &str) -> Result<()> {
        self.vectors.remove(id);
        self.stats.num_vectors = self.vectors.len();
        Ok(())
    }

    fn get_stats(&self) -> IndexStats {
        self.stats.clone()
    }
}

/// HNSW index implementation
pub struct HNSWIndex {
    /// Configuration
    max_connections: usize,
    ef_construction: usize,
    ef_search: usize,

    /// Graph layers
    layers: Vec<HashMap<String, Vec<String>>>,

    /// Vector storage
    vectors: HashMap<String, Vec<f32>>,

    /// Entry point
    entry_point: Option<String>,

    /// Statistics
    stats: IndexStats,
}

impl HNSWIndex {
    pub fn new(max_connections: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            max_connections,
            ef_construction,
            ef_search,
            layers: Vec::new(),
            vectors: HashMap::new(),
            entry_point: None,
            stats: IndexStats {
                index_type: "HNSW".to_string(),
                num_vectors: 0,
                build_time: std::time::Duration::from_secs(0),
                memory_usage: 0,
            },
        }
    }
}

#[async_trait::async_trait]
impl VectorIndex for HNSWIndex {
    async fn build(&mut self, vectors: &DashMap<String, VectorData>) -> Result<()> {
        let start = std::time::Instant::now();

        // Initialize layers
        self.layers.clear();
        self.vectors.clear();

        // Add vectors one by one (simplified HNSW construction)
        for entry in vectors.iter() {
            let id = entry.key().clone();
            let vector = entry.value().vector.clone();

            self.vectors.insert(id.clone(), vector);

            // Determine layer for this vector
            let layer = self.get_random_layer();

            // Ensure we have enough layers
            while self.layers.len() <= layer {
                self.layers.push(HashMap::new());
            }

            // Add to layers
            for l in 0..=layer {
                if l >= self.layers.len() {
                    self.layers.push(HashMap::new());
                }
                self.layers[l].insert(id.clone(), Vec::new());
            }

            // Set entry point if this is the first vector or higher layer
            if self.entry_point.is_none() || layer >= self.layers.len() - 1 {
                self.entry_point = Some(id.clone());
            }
        }

        // Build connections (simplified)
        self.build_connections().await?;

        self.stats.num_vectors = self.vectors.len();
        self.stats.build_time = start.elapsed();

        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Simplified HNSW search
        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::new();

        // Start from entry point
        if let Some(entry) = &self.entry_point {
            if let Some(entry_vector) = self.vectors.get(entry) {
                let similarity = compute_similarity(query, entry_vector, metric)?;
                candidates.push(SimilarityItem {
                    id: entry.clone(),
                    similarity,
                });
                visited.insert(entry.clone());
            }
        }

        // Greedy search (simplified)
        let mut results = Vec::new();
        while let Some(item) = candidates.pop() {
            results.push((item.id.clone(), item.similarity));

            if results.len() >= k {
                break;
            }

            // Explore neighbors (simplified)
            if let Some(neighbors) = self.layers.get(0).and_then(|layer| layer.get(&item.id)) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        if let Some(neighbor_vector) = self.vectors.get(neighbor) {
                            let similarity = compute_similarity(query, neighbor_vector, metric)?;
                            candidates.push(SimilarityItem {
                                id: neighbor.clone(),
                                similarity,
                            });
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        self.vectors.insert(id.clone(), vector);

        // Add to bottom layer
        if self.layers.is_empty() {
            self.layers.push(HashMap::new());
        }
        self.layers[0].insert(id.clone(), Vec::new());

        if self.entry_point.is_none() {
            self.entry_point = Some(id);
        }

        self.stats.num_vectors = self.vectors.len();
        Ok(())
    }

    async fn remove(&mut self, id: &str) -> Result<()> {
        self.vectors.remove(id);

        // Remove from all layers
        for layer in &mut self.layers {
            layer.remove(id);
        }

        self.stats.num_vectors = self.vectors.len();
        Ok(())
    }

    fn get_stats(&self) -> IndexStats {
        self.stats.clone()
    }
}

impl HNSWIndex {
    fn get_random_layer(&self) -> usize {
        // Simplified layer assignment
        let mut layer = 0;
        while rand::thread_rng().gen::<f32>() < 0.5 && layer < 16 {
            layer += 1;
        }
        layer
    }

    async fn build_connections(&mut self) -> Result<()> {
        // Simplified connection building
        // In a real implementation, this would create proper HNSW connections
        Ok(())
    }
}

/// Helper struct for priority queue
#[derive(Debug, Clone)]
struct SimilarityItem {
    id: String,
    similarity: f32,
}

impl PartialEq for SimilarityItem {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for SimilarityItem {}

impl PartialOrd for SimilarityItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SimilarityItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for max heap behavior
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

impl InMemoryVectorStore {
    /// Create new in-memory vector store
    pub fn new(config: VectorStoreConfig) -> Self {
        let stats = VectorStoreStats {
            total_vectors: 0,
            dimension: config.dimension,
            index_type: format!("{:?}", config.index_type),
            index_build_time: std::time::Duration::from_secs(0),
            memory_usage: 0,
            avg_query_time: std::time::Duration::from_millis(0),
            cache_hit_rate: 0.0,
        };

        Self {
            vectors: Arc::new(DashMap::new()),
            index: Arc::new(RwLock::new(None)),
            config,
            query_cache: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(stats)),
        }
    }

    /// Apply filters to search results
    fn apply_filters(&self, data: &VectorData, filters: &[Filter]) -> bool {
        if let Some(metadata) = &data.metadata {
            for filter in filters {
                if let Some(value) = metadata.get(&filter.field) {
                    match &filter.operation {
                        FilterOperation::Equals => {
                            if value != &filter.value {
                                return false;
                            }
                        }
                        FilterOperation::NotEquals => {
                            if value == &filter.value {
                                return false;
                            }
                        }
                        FilterOperation::Contains => {
                            if !value.contains(&filter.value) {
                                return false;
                            }
                        }
                        FilterOperation::StartsWith => {
                            if !value.starts_with(&filter.value) {
                                return false;
                            }
                        }
                        FilterOperation::EndsWith => {
                            if !value.ends_with(&filter.value) {
                                return false;
                            }
                        }
                        FilterOperation::In(values) => {
                            if !values.contains(value) {
                                return false;
                            }
                        }
                        FilterOperation::NotIn(values) => {
                            if values.contains(value) {
                                return false;
                            }
                        }
                        _ => {
                            // TODO: Implement numeric comparisons
                        }
                    }
                } else {
                    return false; // Filter field not found
                }
            }
        } else if !filters.is_empty() {
            return false; // No metadata but filters provided
        }

        true
    }
}

#[async_trait::async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn insert(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()> {
        if vector.len() != self.config.dimension {
            return Err(anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                vector.len()
            ));
        }

        let data = VectorData {
            id: id.clone(),
            vector,
            metadata,
            timestamp: std::time::SystemTime::now(),
        };

        let id_for_lookup = id.clone();
        self.vectors.insert(id.clone(), data);

        // Add to index if it exists
        if let Some(index) = self.index.write().await.as_mut() {
            index
                .add(id, self.vectors.get(&id_for_lookup).unwrap().vector.clone())
                .await?;
        }

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_vectors = self.vectors.len();

        Ok(())
    }

    async fn insert_batch(&self, vectors: Vec<VectorData>) -> Result<()> {
        for data in vectors {
            if data.vector.len() != self.config.dimension {
                return Err(anyhow!("Vector dimension mismatch"));
            }
            self.vectors.insert(data.id.clone(), data);
        }

        // Rebuild index if it exists
        if self.index.read().await.is_some() {
            self.build_index().await?;
        }

        Ok(())
    }

    async fn search(&self, query: &VectorQuery) -> Result<Vec<(String, f32)>> {
        let metric = query.metric.unwrap_or(self.config.default_metric);

        // Check cache first
        if self.config.enable_cache {
            let cache_key = format!(
                "{:?}_{}_{}_{:?}",
                query.vector, query.k, metric, query.filters
            );
            if let Some(cached) = self.query_cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        let start = std::time::Instant::now();

        let results = if let Some(index) = self.index.read().await.as_ref() {
            // Use index for search
            index.search(&query.vector, query.k, metric).await?
        } else {
            // Brute force search
            let mut similarities = Vec::new();

            for entry in self.vectors.iter() {
                let data = entry.value();

                // Apply filters
                if let Some(filters) = &query.filters {
                    if !self.apply_filters(data, filters) {
                        continue;
                    }
                }

                let similarity = compute_similarity(&query.vector, &data.vector, metric)?;

                // Apply minimum similarity threshold
                if let Some(min_sim) = query.min_similarity {
                    if similarity < min_sim {
                        continue;
                    }
                }

                similarities.push((entry.key().clone(), similarity));
            }

            // Sort by similarity
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            similarities.truncate(query.k);
            similarities
        };

        // Update statistics
        let query_time = start.elapsed();
        let mut stats = self.stats.write().await;
        stats.avg_query_time = (stats.avg_query_time + query_time) / 2;

        // Cache result
        if self.config.enable_cache {
            let cache_key = format!(
                "{:?}_{}_{}_{:?}",
                query.vector, query.k, metric, query.filters
            );
            self.query_cache.insert(cache_key, results.clone());
        }

        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<VectorData>> {
        Ok(self.vectors.get(id).map(|entry| entry.value().clone()))
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let removed = self.vectors.remove(id).is_some();

        if removed {
            // Remove from index
            if let Some(index) = self.index.write().await.as_mut() {
                index.remove(id).await?;
            }

            // Update statistics
            let mut stats = self.stats.write().await;
            stats.total_vectors = self.vectors.len();
        }

        Ok(removed)
    }

    async fn update(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()> {
        if vector.len() != self.config.dimension {
            return Err(anyhow!("Vector dimension mismatch"));
        }

        let data = VectorData {
            id: id.clone(),
            vector: vector.clone(),
            metadata,
            timestamp: std::time::SystemTime::now(),
        };

        self.vectors.insert(id.clone(), data);

        // Update index
        if let Some(index) = self.index.write().await.as_mut() {
            index.remove(&id).await?;
            index.add(id, vector).await?;
        }

        Ok(())
    }

    fn size(&self) -> usize {
        self.vectors.len()
    }

    async fn build_index(&self) -> Result<()> {
        let start = std::time::Instant::now();

        let mut new_index: Box<dyn VectorIndex> = match &self.config.index_type {
            IndexType::Flat => Box::new(FlatIndex::new()),
            IndexType::HNSW {
                max_connections,
                ef_construction,
                ef_search,
            } => Box::new(HNSWIndex::new(
                *max_connections,
                *ef_construction,
                *ef_search,
            )),
            _ => return Err(anyhow!("Index type not yet implemented")),
        };

        new_index.build(&self.vectors).await?;

        *self.index.write().await = Some(new_index);

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.index_build_time = start.elapsed();

        Ok(())
    }

    async fn get_statistics(&self) -> Result<VectorStoreStats> {
        Ok(self.stats.read().await.clone())
    }
}

/// Compute similarity between two vectors
pub fn compute_similarity(a: &[f32], b: &[f32], metric: SimilarityMetric) -> Result<f32> {
    if a.len() != b.len() {
        return Err(anyhow!("Vector dimension mismatch"));
    }

    match metric {
        SimilarityMetric::Cosine => {
            let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

            if norm_a == 0.0 || norm_b == 0.0 {
                Ok(0.0)
            } else {
                Ok(dot_product / (norm_a * norm_b))
            }
        }

        SimilarityMetric::Euclidean => {
            let distance: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt();
            Ok(1.0 / (1.0 + distance)) // Convert distance to similarity
        }

        SimilarityMetric::Manhattan => {
            let distance: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
            Ok(1.0 / (1.0 + distance))
        }

        SimilarityMetric::DotProduct => Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()),

        SimilarityMetric::Jaccard => {
            // Treat vectors as binary (>0 = 1, <=0 = 0)
            let a_binary: Vec<bool> = a.iter().map(|&x| x > 0.0).collect();
            let b_binary: Vec<bool> = b.iter().map(|&x| x > 0.0).collect();

            let intersection: usize = a_binary
                .iter()
                .zip(b_binary.iter())
                .map(|(x, y)| if *x && *y { 1 } else { 0 })
                .sum();

            let union: usize = a_binary
                .iter()
                .zip(b_binary.iter())
                .map(|(x, y)| if *x || *y { 1 } else { 0 })
                .sum();

            if union == 0 {
                Ok(0.0)
            } else {
                Ok(intersection as f32 / union as f32)
            }
        }

        SimilarityMetric::Hamming => {
            // Convert to binary and count differences
            let a_binary: Vec<bool> = a.iter().map(|&x| x > 0.0).collect();
            let b_binary: Vec<bool> = b.iter().map(|&x| x > 0.0).collect();

            let differences: usize = a_binary
                .iter()
                .zip(b_binary.iter())
                .map(|(x, y)| if x != y { 1 } else { 0 })
                .sum();

            Ok(1.0 - (differences as f32 / a.len() as f32))
        }
    }
}

/// Create vector store based on configuration
pub fn create_vector_store(config: &VectorStoreConfig) -> Result<Arc<dyn VectorStore>> {
    Ok(Arc::new(InMemoryVectorStore::new(config.clone())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vector_store_creation() {
        let config = VectorStoreConfig::default();
        let store = InMemoryVectorStore::new(config);
        assert_eq!(store.size(), 0);
    }

    #[tokio::test]
    async fn test_vector_insertion_and_retrieval() {
        let config = VectorStoreConfig {
            dimension: 3,
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        let vector = vec![1.0, 2.0, 3.0];
        let metadata = Some(
            [("type".to_string(), "test".to_string())]
                .iter()
                .cloned()
                .collect(),
        );

        store
            .insert("test1".to_string(), vector.clone(), metadata.clone())
            .await
            .unwrap();

        let retrieved = store.get("test1").await.unwrap().unwrap();
        assert_eq!(retrieved.vector, vector);
        assert_eq!(retrieved.metadata, metadata);
    }

    #[tokio::test]
    async fn test_similarity_search() {
        let config = VectorStoreConfig {
            dimension: 3,
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        // Insert test vectors
        store
            .insert("vec1".to_string(), vec![1.0, 0.0, 0.0], None)
            .await
            .unwrap();
        store
            .insert("vec2".to_string(), vec![0.9, 0.1, 0.0], None)
            .await
            .unwrap();
        store
            .insert("vec3".to_string(), vec![0.0, 1.0, 0.0], None)
            .await
            .unwrap();

        let query = VectorQuery {
            vector: vec![1.0, 0.0, 0.0],
            k: 2,
            metric: Some(SimilarityMetric::Cosine),
            include_metadata: false,
            filters: None,
            min_similarity: None,
        };

        let results = store.search(&query).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "vec1"); // Should be most similar
    }

    #[test]
    fn test_similarity_metrics() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];

        let cosine = compute_similarity(&a, &b, SimilarityMetric::Cosine).unwrap();
        assert!((cosine - 1.0).abs() < 1e-6); // Should be 1.0 (parallel vectors)

        let dot_product = compute_similarity(&a, &b, SimilarityMetric::DotProduct).unwrap();
        assert_eq!(dot_product, 28.0); // 1*2 + 2*4 + 3*6 = 28
    }

    #[tokio::test]
    async fn test_index_building() {
        let config = VectorStoreConfig {
            dimension: 3,
            index_type: IndexType::Flat,
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        store
            .insert("vec1".to_string(), vec![1.0, 0.0, 0.0], None)
            .await
            .unwrap();
        store
            .insert("vec2".to_string(), vec![0.0, 1.0, 0.0], None)
            .await
            .unwrap();

        store.build_index().await.unwrap();

        let stats = store.get_statistics().await.unwrap();
        assert_eq!(stats.total_vectors, 2);
    }
}
