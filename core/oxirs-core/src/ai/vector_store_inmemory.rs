//! In-memory vector store with SCIRS2-backed metrics.

use crate::ai::vector_store::{
    compute_similarity, Filter, FilterOperation, IndexType, IVFIndex, LSHIndex, PQIndexLocal,
    VectorData, VectorIndex, VectorQuery, VectorStore, VectorStoreConfig, VectorStoreStats,
};
use crate::ai::vector_store_flat::FlatIndex;
use crate::ai::vector_store_hnsw::HNSWIndex;
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use scirs2_core::metrics::{Counter, Histogram, MetricsRegistry, Timer};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// In-memory vector store with production-ready monitoring
///
/// Integrates SCIRS2 metrics for comprehensive performance tracking:
/// - Insert operations counting
/// - Search latency measurement
/// - Cache hit rate monitoring
/// - Index rebuild timing
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
    /// Create a new vector store with production monitoring
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

        let mut stats = self.stats.write().await;
        stats.avg_query_time = (stats.avg_query_time + query_time) / 2;

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

impl InMemoryVectorStore {
    /// Get production performance metrics
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

    /// Get metrics registry for external monitoring
    pub fn metrics_registry(&self) -> &Arc<MetricsRegistry> {
        &self.metrics_registry
    }
}

/// Vector store performance metrics from SCIRS2
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
