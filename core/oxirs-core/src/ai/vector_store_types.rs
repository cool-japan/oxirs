//! Type definitions for the OxiRS vector store system.
//!
//! Contains all traits, structs, and enums shared across the vector store
//! sub-modules:
//!
//! - [`VectorStore`]  — async insert/search/delete trait
//! - [`VectorData`]   — stored vector with metadata
//! - [`VectorQuery`]  — search parameters
//! - [`Filter`] / [`FilterOperation`] — metadata filter predicates
//! - [`SimilarityMetric`] — distance metric enumeration
//! - [`VectorStoreStats`] / [`VectorStoreConfig`] — config and statistics
//! - [`IndexType`]    — index backend selector
//! - [`VectorIndex`]  — async index trait
//! - [`IndexStats`]   — index build/memory statistics
//! - [`SimilarityItem`] — heap element for priority queue search

use anyhow::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core store trait
// ---------------------------------------------------------------------------

/// Async vector store trait for similarity search.
#[async_trait::async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert a vector with optional metadata.
    async fn insert(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()>;

    /// Insert multiple vectors in a single batch.
    async fn insert_batch(&self, vectors: Vec<VectorData>) -> Result<()>;

    /// Search for the *k* most similar vectors.
    async fn search(&self, query: &VectorQuery) -> Result<Vec<(String, f32)>>;

    /// Retrieve a vector by its ID.
    async fn get(&self, id: &str) -> Result<Option<VectorData>>;

    /// Delete a vector by its ID.  Returns `true` if the vector existed.
    async fn delete(&self, id: &str) -> Result<bool>;

    /// Update an existing vector's data and metadata.
    async fn update(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()>;

    /// Return the number of vectors currently stored.
    fn size(&self) -> usize;

    /// Build or rebuild the internal index for fast ANN search.
    async fn build_index(&self) -> Result<()>;

    /// Return aggregate statistics for this store.
    async fn get_statistics(&self) -> Result<VectorStoreStats>;
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single stored vector with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorData {
    /// Unique identifier.
    pub id: String,
    /// Raw float32 values.
    pub vector: Vec<f32>,
    /// Optional key-value metadata.
    pub metadata: Option<HashMap<String, String>>,
    /// Wall-clock insertion time.
    pub timestamp: std::time::SystemTime,
}

/// Parameters for a similarity search query.
#[derive(Debug, Clone)]
pub struct VectorQuery {
    /// Query vector.
    pub vector: Vec<f32>,
    /// Maximum number of results to return.
    pub k: usize,
    /// Distance metric.  Uses the store default when `None`.
    pub metric: Option<SimilarityMetric>,
    /// If `true`, include metadata fields in results.
    pub include_metadata: bool,
    /// Optional metadata filter predicates (ANDed together).
    pub filters: Option<Vec<Filter>>,
    /// Minimum similarity threshold; results below this are discarded.
    pub min_similarity: Option<f32>,
}

/// A metadata filter predicate.
#[derive(Debug, Clone)]
pub struct Filter {
    /// Metadata field name.
    pub field: String,
    /// Comparison operation.
    pub operation: FilterOperation,
    /// Reference value (string-encoded).
    pub value: String,
}

/// Supported filter comparison operations.
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

// ---------------------------------------------------------------------------
// Similarity metrics
// ---------------------------------------------------------------------------

/// Distance/similarity metric selector.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Cosine similarity (range [−1, 1]; higher = more similar).
    Cosine,
    /// Euclidean distance converted to similarity via `1/(1+d)`.
    Euclidean,
    /// Manhattan (L1) distance converted to similarity via `1/(1+d)`.
    Manhattan,
    /// Raw dot product.
    DotProduct,
    /// Binary Jaccard similarity.
    Jaccard,
    /// Binary Hamming similarity.
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

// ---------------------------------------------------------------------------
// Statistics and configuration
// ---------------------------------------------------------------------------

/// Aggregate statistics for a vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreStats {
    /// Total number of stored vectors.
    pub total_vectors: usize,
    /// Dimensionality of stored vectors.
    pub dimension: usize,
    /// Human-readable index type identifier.
    pub index_type: String,
    /// Time spent building the last index.
    pub index_build_time: std::time::Duration,
    /// Estimated memory usage in bytes.
    pub memory_usage: usize,
    /// Exponentially-averaged query latency.
    pub avg_query_time: std::time::Duration,
    /// Fraction of searches served from cache `[0, 1]`.
    pub cache_hit_rate: f32,
}

/// Configuration parameters for a vector store instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Expected dimensionality of all vectors in this store.
    pub dimension: usize,
    /// Default similarity metric used when a query specifies `None`.
    pub default_metric: SimilarityMetric,
    /// Index backend to use.
    pub index_type: IndexType,
    /// Whether query result caching is active.
    pub enable_cache: bool,
    /// Maximum number of entries in the query cache.
    pub cache_size: usize,
    /// Cache TTL in seconds.
    pub cache_ttl: u64,
    /// Preferred batch-insert chunk size.
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

// ---------------------------------------------------------------------------
// Index type selector
// ---------------------------------------------------------------------------

/// Available ANN (approximate nearest-neighbour) index backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    /// Flat brute-force index (exact search, no approximation).
    Flat,

    /// HNSW — Hierarchical Navigable Small World graph.
    HNSW {
        max_connections: usize,
        ef_construction: usize,
        ef_search: usize,
    },

    /// IVF — Inverted-File (k-means quantisation).
    IVF {
        num_clusters: usize,
        num_probes: usize,
    },

    /// LSH — Locality Sensitive Hashing.
    LSH {
        num_tables: usize,
        hash_length: usize,
    },

    /// PQ — Product Quantisation.
    PQ {
        num_subquantizers: usize,
        bits_per_subquantizer: usize,
    },
}

// ---------------------------------------------------------------------------
// Index trait
// ---------------------------------------------------------------------------

/// Async index backend trait.
#[async_trait::async_trait]
pub trait VectorIndex: Send + Sync {
    /// Build the index from an existing vector map.
    async fn build(&mut self, vectors: &DashMap<String, VectorData>) -> Result<()>;

    /// Search for the `k` most similar vectors.
    async fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>>;

    /// Incrementally add one vector to the index.
    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()>;

    /// Remove one vector from the index.
    async fn remove(&mut self, id: &str) -> Result<()>;

    /// Return index-level statistics.
    fn get_stats(&self) -> IndexStats;
}

/// Statistics reported by an index backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Human-readable index type identifier.
    pub index_type: String,
    /// Number of vectors currently indexed.
    pub num_vectors: usize,
    /// Time spent on the last full build.
    pub build_time: std::time::Duration,
    /// Estimated memory usage in bytes.
    pub memory_usage: usize,
}

// ---------------------------------------------------------------------------
// Priority-queue helper
// ---------------------------------------------------------------------------

/// Priority-queue element used during beam / greedy ANN search.
///
/// Implements `Ord` in *reverse* order so that a `BinaryHeap<SimilarityItem>`
/// becomes a **max-heap** on `similarity`.
#[derive(Debug, Clone)]
pub struct SimilarityItem {
    /// Vector identifier.
    pub id: String,
    /// Computed similarity score.
    pub similarity: f32,
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
        // Reversed so BinaryHeap is a max-heap (highest similarity at top).
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}
