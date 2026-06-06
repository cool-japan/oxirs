//! Search algorithms and the primary `InMemoryVectorStore` implementation.
//!
//! Exports:
//! - [`compute_similarity`]        — single-pair SIMD-accelerated metric computation
//! - [`compute_similarities_batch`] — batch computation (parallel for >100 candidates)
//! - [`create_vector_store`]        — factory function
//! - [`InMemoryVectorStore`]        — production store with SCIRS2 metrics

use super::ivf_index::IVFIndex;
use super::lsh_index::LSHIndex;
use super::pq_index::PQIndexLocal;
use crate::ai::vector_store_index::{FlatIndex, HNSWIndex};
use crate::ai::vector_store_types::{
    Filter, FilterOperation, IndexType, SimilarityMetric, VectorData, VectorIndex, VectorQuery,
    VectorStore, VectorStoreConfig, VectorStoreStats,
};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use scirs2_core::metrics::{Counter, Histogram, MetricsRegistry, Timer};
use scirs2_core::ndarray_ext::ArrayView1;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Similarity computation kernels
// ---------------------------------------------------------------------------

/// Compute the similarity between two float32 vectors.
///
/// Uses `scirs2_core` ndarray views which delegate to BLAS / SIMD operations
/// for maximum throughput.
pub fn compute_similarity(a: &[f32], b: &[f32], metric: SimilarityMetric) -> Result<f32> {
    if a.len() != b.len() {
        return Err(anyhow!("Vector dimension mismatch"));
    }

    let a_arr = ArrayView1::from(a);
    let b_arr = ArrayView1::from(b);

    match metric {
        SimilarityMetric::Cosine => {
            let dot_product = a_arr.dot(&b_arr);
            let norm_a = a_arr.dot(&a_arr).sqrt();
            let norm_b = b_arr.dot(&b_arr).sqrt();

            if norm_a == 0.0 || norm_b == 0.0 {
                Ok(0.0)
            } else {
                Ok(dot_product / (norm_a * norm_b))
            }
        }

        SimilarityMetric::Euclidean => {
            let diff = &a_arr - &b_arr;
            let distance = diff.dot(&diff).sqrt();
            Ok(1.0 / (1.0 + distance))
        }

        SimilarityMetric::Manhattan => {
            let diff = &a_arr - &b_arr;
            let distance = diff.mapv(f32::abs).sum();
            Ok(1.0 / (1.0 + distance))
        }

        SimilarityMetric::DotProduct => Ok(a_arr.dot(&b_arr)),

        SimilarityMetric::Jaccard => {
            let a_binary = a_arr.mapv(|x| if x > 0.0 { 1u32 } else { 0 });
            let b_binary = b_arr.mapv(|x| if x > 0.0 { 1u32 } else { 0 });

            let intersection: u32 = (&a_binary * &b_binary).sum();
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
            let a_binary = a_arr.mapv(|x| if x > 0.0 { 1u32 } else { 0 });
            let b_binary = b_arr.mapv(|x| if x > 0.0 { 1u32 } else { 0 });

            let differences: u32 = a_binary
                .iter()
                .zip(b_binary.iter())
                .map(|(x, y)| if x != y { 1 } else { 0 })
                .sum();

            Ok(1.0 - (differences as f32 / a.len() as f32))
        }
    }
}

/// Batch-compute similarities between one query vector and many candidates.
///
/// Uses Rayon parallel iterators for batches larger than 100 entries to
/// amortise parallel-dispatch overhead.
pub fn compute_similarities_batch(
    query: &[f32],
    candidates: &[&[f32]],
    metric: SimilarityMetric,
) -> Result<Vec<f32>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    for candidate in candidates {
        if candidate.len() != query.len() {
            return Err(anyhow!("Vector dimension mismatch in batch"));
        }
    }

    let query_arr = ArrayView1::from(query);

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
        candidates
            .iter()
            .map(|candidate| compute_similarity(query, candidate, metric))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/// Create a new [`VectorStore`] backed by an [`InMemoryVectorStore`].
pub fn create_vector_store(config: &VectorStoreConfig) -> Result<Arc<dyn VectorStore>> {
    Ok(Arc::new(InMemoryVectorStore::new(config.clone())))
}

// ---------------------------------------------------------------------------
// InMemoryVectorStore
// ---------------------------------------------------------------------------

/// In-memory vector store with SCIRS2-backed production monitoring.
///
/// Tracks the following metrics:
/// - `vector_inserts`    — total insert operations
/// - `vector_searches`   — total search operations
/// - `search_latency`    — search latency distribution
/// - `index_build_time`  — time to build/rebuild the index
/// - `similarity_scores` — histogram of computed similarity values
pub struct InMemoryVectorStore {
    vectors: Arc<DashMap<String, VectorData>>,
    index: Arc<RwLock<Option<Box<dyn VectorIndex>>>>,
    config: VectorStoreConfig,
    query_cache: Arc<DashMap<String, Vec<(String, f32)>>>,
    stats: Arc<RwLock<VectorStoreStats>>,
    cache_hits: Arc<std::sync::atomic::AtomicUsize>,
    cache_misses: Arc<std::sync::atomic::AtomicUsize>,
    insert_counter: Arc<Counter>,
    search_counter: Arc<Counter>,
    search_timer: Arc<Timer>,
    index_build_timer: Arc<Timer>,
    similarity_histogram: Arc<Histogram>,
    metrics_registry: Arc<MetricsRegistry>,
}

impl InMemoryVectorStore {
    /// Create a new store with the given configuration.
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

    /// Test whether a vector entry passes all metadata filters.
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
                            if let (Ok(val_num), Ok(filter_num)) =
                                (value.parse::<f64>(), filter.value.parse::<f64>())
                            {
                                if val_num <= filter_num {
                                    return false;
                                }
                            } else if value <= &filter.value {
                                return false;
                            }
                        }
                        FilterOperation::LessThan => {
                            if let (Ok(val_num), Ok(filter_num)) =
                                (value.parse::<f64>(), filter.value.parse::<f64>())
                            {
                                if val_num >= filter_num {
                                    return false;
                                }
                            } else if value >= &filter.value {
                                return false;
                            }
                        }
                    }
                } else {
                    return false;
                }
            }
        } else if !filters.is_empty() {
            return false;
        }

        true
    }

    /// Return SCIRS2-based performance metrics for this store instance.
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
            min_search_latency_ms: 0.0,
            max_search_latency_ms: 0.0,
            avg_index_build_time_ms: index_timer_stats.mean * 1000.0,
            avg_similarity_score: similarity_hist_stats.mean,
            similarity_count: similarity_hist_stats.count,
        }
    }

    /// Expose the underlying SCIRS2 [`MetricsRegistry`] for external monitoring.
    pub fn metrics_registry(&self) -> &Arc<MetricsRegistry> {
        &self.metrics_registry
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
        self.insert_counter.inc();

        if let Some(index) = self.index.write().await.as_mut() {
            index
                .add(
                    id,
                    self.vectors
                        .get(&id_for_lookup)
                        .expect("vector should exist after insert")
                        .vector
                        .clone(),
                )
                .await?;
        }

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

        if self.index.read().await.is_some() {
            self.build_index().await?;
        }

        Ok(())
    }

    async fn search(&self, query: &VectorQuery) -> Result<Vec<(String, f32)>> {
        self.search_counter.inc();

        let metric = query.metric.unwrap_or(self.config.default_metric);

        if self.config.enable_cache {
            let cache_key = format!(
                "{:?}_{}_{}_{:?}",
                query.vector, query.k, metric, query.filters
            );
            if let Some(cached) = self.query_cache.get(&cache_key) {
                self.cache_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(cached.clone());
            } else {
                self.cache_misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        let start = std::time::Instant::now();

        let results = if let Some(index) = self.index.read().await.as_ref() {
            index.search(&query.vector, query.k, metric).await?
        } else {
            let mut similarities = Vec::new();

            for entry in self.vectors.iter() {
                let data = entry.value();

                if let Some(filters) = &query.filters {
                    if !self.apply_filters(data, filters) {
                        continue;
                    }
                }

                let similarity = compute_similarity(&query.vector, &data.vector, metric)?;
                self.similarity_histogram.observe(similarity as f64);

                if let Some(min_sim) = query.min_similarity {
                    if similarity < min_sim {
                        continue;
                    }
                }

                similarities.push((entry.key().clone(), similarity));
            }

            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            similarities.truncate(query.k);
            similarities
        };

        let query_time = start.elapsed();
        self.search_timer.observe(query_time);

        {
            let mut stats = self.stats.write().await;
            stats.avg_query_time = (stats.avg_query_time + query_time) / 2;
        }

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
            if let Some(index) = self.index.write().await.as_mut() {
                index.remove(id).await?;
            }

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
            IndexType::IVF {
                num_clusters,
                num_probes,
            } => Box::new(IVFIndex::new(*num_clusters, *num_probes)),
            IndexType::LSH {
                num_tables,
                hash_length,
            } => Box::new(LSHIndex::new(*num_tables, *hash_length)),
            IndexType::PQ {
                num_subquantizers,
                bits_per_subquantizer,
            } => Box::new(PQIndexLocal::new(
                *num_subquantizers,
                *bits_per_subquantizer,
            )),
        };

        new_index.build(&self.vectors).await?;
        *self.index.write().await = Some(new_index);

        let mut stats = self.stats.write().await;
        stats.index_build_time = start.elapsed();

        Ok(())
    }

    async fn get_statistics(&self) -> Result<VectorStoreStats> {
        let mut stats = self.stats.read().await.clone();

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

// ---------------------------------------------------------------------------
// Performance metrics DTO
// ---------------------------------------------------------------------------

/// SCIRS2-sourced performance metrics for an [`InMemoryVectorStore`] instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStorePerformanceMetrics {
    pub total_inserts: u64,
    pub total_searches: u64,
    pub avg_search_latency_ms: f64,
    pub min_search_latency_ms: f64,
    pub max_search_latency_ms: f64,
    pub avg_index_build_time_ms: f64,
    pub avg_similarity_score: f64,
    pub similarity_count: u64,
}

impl std::fmt::Display for VectorStorePerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VectorPerf {{ inserts: {}, searches: {}, avg_latency: {:.2}ms, \
             min: {:.2}ms, max: {:.2}ms, avg_similarity: {:.3}, computations: {} }}",
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
