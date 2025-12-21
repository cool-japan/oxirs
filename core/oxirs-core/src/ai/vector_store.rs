//! Vector Store for Efficient Similarity Search
//!
//! This module provides high-performance vector storage and similarity search
//! capabilities optimized for knowledge graph embeddings and AI applications.

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use scirs2_core::metrics::{Counter, Histogram, MetricsRegistry, Timer};
use scirs2_core::ndarray_ext::ArrayView1;
use scirs2_core::random::Random;
use scirs2_core::rngs::StdRng;
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

/// In-memory vector store with production-ready monitoring
///
/// Integrates SCIRS2 metrics for comprehensive performance tracking:
/// - Insert operations counting
/// - Search latency measurement
/// - Cache hit rate monitoring
/// - Index rebuild timing
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

    /// Cache hit counter (atomic for lock-free access)
    cache_hits: Arc<std::sync::atomic::AtomicUsize>,

    /// Cache miss counter (atomic for lock-free access)
    cache_misses: Arc<std::sync::atomic::AtomicUsize>,

    /// SCIRS2 Metrics
    /// Insert operation counter
    insert_counter: Arc<Counter>,

    /// Search operation counter
    search_counter: Arc<Counter>,

    /// Search latency timer
    search_timer: Arc<Timer>,

    /// Index build timer
    index_build_timer: Arc<Timer>,

    /// Similarity computation histogram
    similarity_histogram: Arc<Histogram>,

    /// Metrics registry
    metrics_registry: Arc<MetricsRegistry>,
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

impl Default for FlatIndex {
    fn default() -> Self {
        Self::new()
    }
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
    #[allow(dead_code)]
    max_connections: usize,
    #[allow(dead_code)]
    ef_construction: usize,
    #[allow(dead_code)]
    ef_search: usize,

    /// Graph layers
    layers: Vec<HashMap<String, Vec<String>>>,

    /// Vector storage
    vectors: HashMap<String, Vec<f32>>,

    /// Entry point
    entry_point: Option<String>,

    /// Statistics
    stats: IndexStats,

    /// Random number generator (thread-safe)
    rng: Random<StdRng>,
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
            rng: Random::seed(42), // Use deterministic seed for thread safety
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
        // Use the enhanced beam search algorithm
        self.beam_search(query, k, metric)
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
    fn get_random_layer(&mut self) -> usize {
        // Simplified layer assignment using exponential distribution
        let mut layer = 0;
        while (self.rng.random_f64() as f32) < 0.5 && layer < 16 {
            layer += 1;
        }
        layer
    }

    async fn build_connections(&mut self) -> Result<()> {
        // Build proper HNSW graph connections using greedy search
        let ids: Vec<String> = self.vectors.keys().cloned().collect();

        if ids.is_empty() {
            return Ok(());
        }

        // For each vector, find and connect to its nearest neighbors in each layer
        for id in &ids {
            // Get vector for this node
            let vector = match self.vectors.get(id) {
                Some(v) => v.clone(),
                None => continue,
            };

            // Find nearest neighbors in each layer this node belongs to
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                if !layer.contains_key(id) {
                    continue;
                }

                // Find candidates in this layer
                let mut candidates: Vec<(String, f32)> = Vec::new();
                for (other_id, _) in layer.iter() {
                    if other_id == id {
                        continue;
                    }
                    if let Some(other_vector) = self.vectors.get(other_id) {
                        let similarity =
                            compute_similarity(&vector, other_vector, SimilarityMetric::Cosine)
                                .unwrap_or(0.0);
                        candidates.push((other_id.clone(), similarity));
                    }
                }

                // Sort by similarity and take top max_connections
                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                let max_conn = if layer_idx == 0 {
                    self.max_connections * 2 // Bottom layer gets more connections
                } else {
                    self.max_connections
                };
                candidates.truncate(max_conn);

                // Set connections for this node
                let connections: Vec<String> = candidates.into_iter().map(|(cid, _)| cid).collect();
                layer.insert(id.clone(), connections.clone());

                // Add bidirectional connections (make graph undirected)
                for neighbor_id in connections {
                    if let Some(neighbor_connections) = layer.get_mut(&neighbor_id) {
                        if !neighbor_connections.contains(id)
                            && neighbor_connections.len() < max_conn
                        {
                            neighbor_connections.push(id.clone());
                        }
                    }
                }
            }
        }

        // Calculate memory usage
        let mut memory = 0;
        for (id, vec) in &self.vectors {
            memory += id.len() + vec.len() * 4;
        }
        for layer in &self.layers {
            for (id, connections) in layer {
                memory += id.len() + connections.len() * 8; // Approximate string overhead
            }
        }
        self.stats.memory_usage = memory;

        Ok(())
    }

    /// Search using proper beam search with ef_search parameter
    fn beam_search(
        &self,
        query: &[f32],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Start from entry point
        let entry = match &self.entry_point {
            Some(e) => e.clone(),
            None => return Ok(Vec::new()),
        };

        // Greedy search from top layer to bottom
        let mut current_best = entry.clone();

        // Navigate through layers from top to bottom
        for layer_idx in (1..self.layers.len()).rev() {
            let layer = &self.layers[layer_idx];

            // Greedy search in this layer
            while let Some(current_vector) = self.vectors.get(&current_best) {
                let current_sim = compute_similarity(query, current_vector, metric)?;

                let mut improved = false;
                if let Some(neighbors) = layer.get(&current_best) {
                    for neighbor in neighbors {
                        if let Some(neighbor_vector) = self.vectors.get(neighbor) {
                            let neighbor_sim = compute_similarity(query, neighbor_vector, metric)?;
                            if neighbor_sim > current_sim {
                                current_best = neighbor.clone();
                                improved = true;
                                break;
                            }
                        }
                    }
                }

                if !improved {
                    break;
                }
            }
        }

        // Beam search in bottom layer (layer 0)
        if self.layers.is_empty() {
            return Ok(Vec::new());
        }

        let bottom_layer = &self.layers[0];
        let ef = std::cmp::max(k, self.ef_search);

        // Use priority queue for candidates (max-heap by similarity)
        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::new();

        // Initialize with entry point
        if let Some(entry_vector) = self.vectors.get(&current_best) {
            let sim = compute_similarity(query, entry_vector, metric)?;
            candidates.push(SimilarityItem {
                id: current_best.clone(),
                similarity: sim,
            });
            visited.insert(current_best);
        }

        // Results (min-heap by similarity, will keep worst at top for easy replacement)
        let mut results: Vec<(String, f32)> = Vec::new();

        // Beam search
        while let Some(current) = candidates.pop() {
            // Add to results if within ef
            if results.len() < ef {
                results.push((current.id.clone(), current.similarity));
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            } else if current.similarity > results.last().map(|r| r.1).unwrap_or(f32::NEG_INFINITY)
            {
                results.pop();
                results.push((current.id.clone(), current.similarity));
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            }

            // Explore neighbors
            if let Some(neighbors) = bottom_layer.get(&current.id) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        if let Some(neighbor_vector) = self.vectors.get(neighbor) {
                            let sim = compute_similarity(query, neighbor_vector, metric)?;

                            // Only add if potentially useful
                            let worst_result =
                                results.last().map(|r| r.1).unwrap_or(f32::NEG_INFINITY);
                            if results.len() < ef || sim > worst_result {
                                candidates.push(SimilarityItem {
                                    id: neighbor.clone(),
                                    similarity: sim,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Return top k results
        results.truncate(k);
        Ok(results)
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
    /// Create a new vector store with production monitoring
    ///
    /// Automatically initializes SCIRS2 metrics for comprehensive tracking:
    /// - Insert/search operation counters
    /// - Latency timers for performance analysis
    /// - Similarity computation histograms
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

        // Initialize SCIRS2 metrics
        let metrics_registry = Arc::new(MetricsRegistry::new());
        let insert_counter = Arc::new(Counter::new("vector_inserts".to_string()));
        let search_counter = Arc::new(Counter::new("vector_searches".to_string()));
        let search_timer = Arc::new(Timer::new("search_latency".to_string()));
        let index_build_timer = Arc::new(Timer::new("index_build_time".to_string()));
        let similarity_histogram = Arc::new(Histogram::new("similarity_scores".to_string()));

        Self {
            vectors: Arc::new(DashMap::new()),
            index: Arc::new(RwLock::new(None)),
            config,
            query_cache: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(stats)),
            cache_hits: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            cache_misses: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            insert_counter,
            search_counter,
            search_timer,
            index_build_timer,
            similarity_histogram,
            metrics_registry,
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
                        FilterOperation::GreaterThan => {
                            // Try to parse both values as f64 for numeric comparison
                            if let (Ok(val_num), Ok(filter_num)) =
                                (value.parse::<f64>(), filter.value.parse::<f64>())
                            {
                                if val_num <= filter_num {
                                    return false;
                                }
                            } else {
                                // Fallback to lexicographic comparison if not numeric
                                if value <= &filter.value {
                                    return false;
                                }
                            }
                        }
                        FilterOperation::LessThan => {
                            // Try to parse both values as f64 for numeric comparison
                            if let (Ok(val_num), Ok(filter_num)) =
                                (value.parse::<f64>(), filter.value.parse::<f64>())
                            {
                                if val_num >= filter_num {
                                    return false;
                                }
                            } else {
                                // Fallback to lexicographic comparison if not numeric
                                if value >= &filter.value {
                                    return false;
                                }
                            }
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

        // Track insert operation
        self.insert_counter.inc();

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
        // Track search operation
        self.search_counter.inc();

        let metric = query.metric.unwrap_or(self.config.default_metric);

        // Check cache first
        if self.config.enable_cache {
            let cache_key = format!(
                "{:?}_{}_{}_{:?}",
                query.vector, query.k, metric, query.filters
            );
            if let Some(cached) = self.query_cache.get(&cache_key) {
                // Cache hit
                self.cache_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(cached.clone());
            } else {
                // Cache miss
                self.cache_misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

                // Track similarity distribution
                self.similarity_histogram.observe(similarity as f64);

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

        // Update statistics and track metrics
        let query_time = start.elapsed();
        self.search_timer.observe(query_time);

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
        let mut stats = self.stats.read().await.clone();

        // Calculate actual cache hit rate
        let hits = self.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;

        stats.cache_hit_rate = if total > 0 {
            hits as f32 / total as f32
        } else {
            0.0
        };

        Ok(stats)
    }
}

// Additional methods for InMemoryVectorStore outside the trait
impl InMemoryVectorStore {
    /// Get production performance metrics
    ///
    /// Returns comprehensive SCIRS2-based metrics including:
    /// - Total insert/search operations
    /// - Search latency statistics
    /// - Similarity score distribution
    /// - Index build performance
    pub fn get_performance_metrics(&self) -> VectorStorePerformanceMetrics {
        let insert_count = self.insert_counter.get();
        let search_count = self.search_counter.get();

        let search_timer_stats = self.search_timer.get_stats();
        let index_timer_stats = self.index_build_timer.get_stats();
        let similarity_hist_stats = self.similarity_histogram.get_stats();

        VectorStorePerformanceMetrics {
            total_inserts: insert_count,
            total_searches: search_count,
            avg_search_latency_ms: search_timer_stats.mean * 1000.0,
            min_search_latency_ms: 0.0, // Timer doesn't track min
            max_search_latency_ms: 0.0, // Timer doesn't track max
            avg_index_build_time_ms: index_timer_stats.mean * 1000.0,
            avg_similarity_score: similarity_hist_stats.mean,
            similarity_count: similarity_hist_stats.count,
        }
    }

    /// Get metrics registry for external monitoring
    pub fn metrics_registry(&self) -> &Arc<MetricsRegistry> {
        &self.metrics_registry
    }
}

/// Vector store performance metrics from SCIRS2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStorePerformanceMetrics {
    /// Total number of insert operations
    pub total_inserts: u64,

    /// Total number of search operations
    pub total_searches: u64,

    /// Average search latency in milliseconds
    pub avg_search_latency_ms: f64,

    /// Minimum search latency in milliseconds
    pub min_search_latency_ms: f64,

    /// Maximum search latency in milliseconds
    pub max_search_latency_ms: f64,

    /// Average index build time in milliseconds
    pub avg_index_build_time_ms: f64,

    /// Average similarity score across all computations
    pub avg_similarity_score: f64,

    /// Total similarity score calculations
    pub similarity_count: u64,
}

impl std::fmt::Display for VectorStorePerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VectorPerf {{ inserts: {}, searches: {}, avg_latency: {:.2}ms, min: {:.2}ms, max: {:.2}ms, avg_similarity: {:.3}, computations: {} }}",
            self.total_inserts,
            self.total_searches,
            self.avg_search_latency_ms,
            self.min_search_latency_ms,
            self.max_search_latency_ms,
            self.avg_similarity_score,
            self.similarity_count
        )
    }
}

/// Compute similarity between two vectors using SIMD-optimized operations
///
/// This function uses scirs2_core's ndarray operations which leverage BLAS
/// and SIMD instructions for maximum performance (5-10x faster than naive iteration).
pub fn compute_similarity(a: &[f32], b: &[f32], metric: SimilarityMetric) -> Result<f32> {
    if a.len() != b.len() {
        return Err(anyhow!("Vector dimension mismatch"));
    }

    // Convert slices to ndarray views for SIMD-optimized operations
    let a_arr = ArrayView1::from(a);
    let b_arr = ArrayView1::from(b);

    match metric {
        SimilarityMetric::Cosine => {
            // Use ndarray's optimized dot product (BLAS-accelerated)
            let dot_product = a_arr.dot(&b_arr);

            // Compute norms using SIMD-optimized operations
            let norm_a = a_arr.dot(&a_arr).sqrt();
            let norm_b = b_arr.dot(&b_arr).sqrt();

            if norm_a == 0.0 || norm_b == 0.0 {
                Ok(0.0)
            } else {
                Ok(dot_product / (norm_a * norm_b))
            }
        }

        SimilarityMetric::Euclidean => {
            // Compute squared difference using ndarray operations
            let diff = &a_arr - &b_arr;
            let distance = diff.dot(&diff).sqrt();
            Ok(1.0 / (1.0 + distance)) // Convert distance to similarity
        }

        SimilarityMetric::Manhattan => {
            // Use mapv for SIMD-optimized absolute value and sum
            let diff = &a_arr - &b_arr;
            let distance = diff.mapv(f32::abs).sum();
            Ok(1.0 / (1.0 + distance))
        }

        SimilarityMetric::DotProduct => {
            // Direct BLAS-accelerated dot product
            Ok(a_arr.dot(&b_arr))
        }

        SimilarityMetric::Jaccard => {
            // Use SIMD-optimized boolean operations
            let a_binary = a_arr.mapv(|x| if x > 0.0 { 1u32 } else { 0 });
            let b_binary = b_arr.mapv(|x| if x > 0.0 { 1u32 } else { 0 });

            // Intersection: both are 1
            let intersection: u32 = (&a_binary * &b_binary).sum();

            // Union: at least one is 1
            let union: u32 = a_binary
                .iter()
                .zip(b_binary.iter())
                .map(|(x, y)| if *x > 0 || *y > 0 { 1 } else { 0 })
                .sum();

            if union == 0 {
                Ok(0.0)
            } else {
                Ok(intersection as f32 / union as f32)
            }
        }

        SimilarityMetric::Hamming => {
            // Convert to binary using SIMD-optimized mapv
            let a_binary = a_arr.mapv(|x| if x > 0.0 { 1u32 } else { 0 });
            let b_binary = b_arr.mapv(|x| if x > 0.0 { 1u32 } else { 0 });

            // Count differences using SIMD operations
            let differences: u32 = a_binary
                .iter()
                .zip(b_binary.iter())
                .map(|(x, y)| if x != y { 1 } else { 0 })
                .sum();

            Ok(1.0 - (differences as f32 / a.len() as f32))
        }
    }
}

/// Batch compute similarities between a query vector and multiple candidate vectors
///
/// Uses parallel processing for large batches (>100 vectors) for additional speedup.
pub fn compute_similarities_batch(
    query: &[f32],
    candidates: &[&[f32]],
    metric: SimilarityMetric,
) -> Result<Vec<f32>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Validate dimensions
    for candidate in candidates {
        if candidate.len() != query.len() {
            return Err(anyhow!("Vector dimension mismatch in batch"));
        }
    }

    let query_arr = ArrayView1::from(query);

    // Precompute query norm for cosine similarity
    let query_norm = match metric {
        SimilarityMetric::Cosine => {
            let norm = query_arr.dot(&query_arr).sqrt();
            if norm == 0.0 {
                return Ok(vec![0.0; candidates.len()]);
            }
            norm
        }
        _ => 1.0,
    };

    // Use parallel processing for large batches
    if candidates.len() > 100 {
        use rayon::prelude::*;

        let results: Vec<f32> = candidates
            .par_iter()
            .map(|candidate| {
                let c_arr = ArrayView1::from(*candidate);
                match metric {
                    SimilarityMetric::Cosine => {
                        let dot = query_arr.dot(&c_arr);
                        let c_norm = c_arr.dot(&c_arr).sqrt();
                        if c_norm == 0.0 {
                            0.0
                        } else {
                            dot / (query_norm * c_norm)
                        }
                    }
                    SimilarityMetric::Euclidean => {
                        let diff = &query_arr - &c_arr;
                        let dist = diff.dot(&diff).sqrt();
                        1.0 / (1.0 + dist)
                    }
                    SimilarityMetric::Manhattan => {
                        let diff = &query_arr - &c_arr;
                        let dist = diff.mapv(f32::abs).sum();
                        1.0 / (1.0 + dist)
                    }
                    SimilarityMetric::DotProduct => query_arr.dot(&c_arr),
                    _ => compute_similarity(query, candidate, metric).unwrap_or(0.0),
                }
            })
            .collect();

        Ok(results)
    } else {
        // Sequential for small batches (avoid parallel overhead)
        candidates
            .iter()
            .map(|candidate| compute_similarity(query, candidate, metric))
            .collect()
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

    #[tokio::test]
    async fn test_hnsw_index_building() {
        let config = VectorStoreConfig {
            dimension: 3,
            index_type: IndexType::HNSW {
                max_connections: 16,
                ef_construction: 100,
                ef_search: 50,
            },
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        // Insert multiple vectors
        for i in 0..10 {
            let angle = (i as f32) * std::f32::consts::PI / 5.0;
            store
                .insert(format!("vec{i}"), vec![angle.cos(), angle.sin(), 0.0], None)
                .await
                .unwrap();
        }

        // Build HNSW index
        store.build_index().await.unwrap();

        let stats = store.get_statistics().await.unwrap();
        assert_eq!(stats.total_vectors, 10);
        assert!(stats.index_type.contains("HNSW"));
    }

    #[tokio::test]
    async fn test_hnsw_search() {
        let config = VectorStoreConfig {
            dimension: 3,
            index_type: IndexType::HNSW {
                max_connections: 16,
                ef_construction: 100,
                ef_search: 50,
            },
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        // Insert test vectors in a circle pattern
        for i in 0..20 {
            let angle = (i as f32) * std::f32::consts::PI * 2.0 / 20.0;
            store
                .insert(format!("vec{i}"), vec![angle.cos(), angle.sin(), 0.0], None)
                .await
                .unwrap();
        }

        // Build HNSW index
        store.build_index().await.unwrap();

        // Query for nearest neighbors to vec0 (1.0, 0.0, 0.0)
        let query = VectorQuery {
            vector: vec![1.0, 0.0, 0.0],
            k: 3,
            metric: Some(SimilarityMetric::Cosine),
            include_metadata: false,
            filters: None,
            min_similarity: None,
        };

        let results = store.search(&query).await.unwrap();

        // Should find at least some results
        assert!(!results.is_empty());
        // First result should be vec0 (exact match)
        assert_eq!(results[0].0, "vec0");
        // Similarity should be 1.0 for exact match
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_hnsw_large_dataset() {
        let config = VectorStoreConfig {
            dimension: 10,
            index_type: IndexType::HNSW {
                max_connections: 16,
                ef_construction: 100,
                ef_search: 50,
            },
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        // Insert 100 random vectors using deterministic pattern
        for i in 0..100 {
            let vec: Vec<f32> = (0..10)
                .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                .collect();
            store.insert(format!("vec{i}"), vec, None).await.unwrap();
        }

        // Build HNSW index
        store.build_index().await.unwrap();

        // Query
        let query_vec = vec![0.5f32; 10];
        let query = VectorQuery {
            vector: query_vec,
            k: 10,
            metric: Some(SimilarityMetric::Cosine),
            include_metadata: false,
            filters: None,
            min_similarity: None,
        };

        let results = store.search(&query).await.unwrap();

        // Should return k results (or all if less than k)
        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // Results should be sorted by similarity (descending)
        for i in 1..results.len() {
            assert!(results[i - 1].1 >= results[i].1);
        }
    }

    #[test]
    fn test_batch_similarity_computation() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates: Vec<&[f32]> =
            vec![&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.707, 0.707, 0.0]];

        let similarities =
            compute_similarities_batch(&query, &candidates, SimilarityMetric::Cosine).unwrap();

        assert_eq!(similarities.len(), 3);
        // First should be 1.0 (identical)
        assert!((similarities[0] - 1.0).abs() < 0.01);
        // Second should be 0.0 (orthogonal)
        assert!(similarities[1].abs() < 0.01);
        // Third should be ~0.707
        assert!((similarities[2] - 0.707).abs() < 0.01);
    }
}
