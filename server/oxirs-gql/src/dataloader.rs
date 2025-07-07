//! DataLoader implementation for efficient batching and caching
//!
//! This module provides the DataLoader pattern implementation to prevent N+1 queries
//! in GraphQL resolvers by batching requests and caching results.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::time::sleep;

/// Trait for batch loading functions
#[async_trait]
pub trait BatchLoadFn<K, V>: Send + Sync
where
    K: Send + Sync + Clone + Hash + Eq,
    V: Send + Sync + Clone,
{
    async fn load(&self, keys: Vec<K>) -> Result<HashMap<K, V>>;
}

/// Configuration for DataLoader behavior
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Maximum number of keys to batch together
    pub max_batch_size: usize,
    /// Maximum time to wait before dispatching a batch
    pub batch_delay: Duration,
    /// Cache TTL for loaded values
    pub cache_ttl: Duration,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Enable caching
    pub enable_cache: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            batch_delay: Duration::from_millis(10),
            cache_ttl: Duration::from_secs(300), // 5 minutes
            max_cache_size: 1000,
            enable_cache: true,
        }
    }
}

/// Cached entry with TTL
#[derive(Debug, Clone)]
struct CachedEntry<V> {
    value: V,
    created_at: Instant,
    ttl: Duration,
}

impl<V> CachedEntry<V> {
    fn new(value: V, ttl: Duration) -> Self {
        Self {
            value,
            created_at: Instant::now(),
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Pending batch of requests
#[derive(Debug)]
struct PendingBatch<K, V>
where
    K: Send + Sync + Clone + Hash + Eq,
    V: Send + Sync + Clone,
{
    keys: Vec<K>,
    resolvers: Vec<tokio::sync::oneshot::Sender<Result<Option<V>>>>,
    created_at: Instant,
}

impl<K, V> PendingBatch<K, V>
where
    K: Send + Sync + Clone + Hash + Eq,
    V: Send + Sync + Clone,
{
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            resolvers: Vec::new(),
            created_at: Instant::now(),
        }
    }

    fn add_request(&mut self, key: K, resolver: tokio::sync::oneshot::Sender<Result<Option<V>>>) {
        self.keys.push(key);
        self.resolvers.push(resolver);
    }

    fn should_dispatch(&self, config: &DataLoaderConfig) -> bool {
        self.keys.len() >= config.max_batch_size || self.created_at.elapsed() >= config.batch_delay
    }
}

/// DataLoader for efficient batching and caching
pub struct DataLoader<K, V>
where
    K: Send + Sync + Clone + Hash + Eq + 'static,
    V: Send + Sync + Clone + 'static,
{
    batch_fn: Arc<dyn BatchLoadFn<K, V>>,
    config: DataLoaderConfig,
    cache: Arc<RwLock<HashMap<K, CachedEntry<V>>>>,
    pending_batch: Arc<Mutex<Option<PendingBatch<K, V>>>>,
    stats: Arc<RwLock<DataLoaderStats>>,
}

/// DataLoader performance statistics
#[derive(Debug, Default, Clone)]
pub struct DataLoaderStats {
    pub requests_total: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub batches_dispatched: u64,
    pub avg_batch_size: f64,
    pub total_load_time: Duration,
}

impl DataLoaderStats {
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.requests_total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.requests_total as f64
        }
    }

    pub fn avg_load_time(&self) -> Duration {
        if self.batches_dispatched == 0 {
            Duration::from_secs(0)
        } else {
            self.total_load_time / self.batches_dispatched as u32
        }
    }
}

impl<K, V> DataLoader<K, V>
where
    K: Send + Sync + Clone + Hash + Eq + 'static,
    V: Send + Sync + Clone + 'static,
{
    /// Create a new DataLoader with the given batch function
    pub fn new(batch_fn: Arc<dyn BatchLoadFn<K, V>>) -> Self {
        Self::with_config(batch_fn, DataLoaderConfig::default())
    }

    /// Create a new DataLoader with custom configuration
    pub fn with_config(batch_fn: Arc<dyn BatchLoadFn<K, V>>, config: DataLoaderConfig) -> Self {
        let loader = Self {
            batch_fn,
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            pending_batch: Arc::new(Mutex::new(None)),
            stats: Arc::new(RwLock::new(DataLoaderStats::default())),
        };

        // Start the batch dispatcher
        loader.start_batch_dispatcher();
        loader
    }

    /// Load a single value by key
    pub async fn load(&self, key: K) -> Result<Option<V>> {
        self.update_stats_request().await;

        // Check cache first
        if self.config.enable_cache {
            if let Some(cached_value) = self.get_from_cache(&key).await {
                self.update_stats_cache_hit().await;
                return Ok(Some(cached_value));
            }
        }

        self.update_stats_cache_miss().await;

        // Create a oneshot channel for the result
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Add to pending batch
        self.add_to_batch(key, tx).await;

        // Wait for result
        rx.await.map_err(|_| anyhow!("DataLoader batch failed"))?
    }

    /// Load multiple values by keys
    pub async fn load_many(&self, keys: Vec<K>) -> Result<HashMap<K, V>> {
        if keys.is_empty() {
            return Ok(HashMap::new());
        }

        // Check cache for all keys
        let mut results = HashMap::new();
        let mut missing_keys = Vec::new();

        if self.config.enable_cache {
            for key in keys {
                match self.get_from_cache(&key).await {
                    Some(cached_value) => {
                        results.insert(key, cached_value);
                        self.update_stats_cache_hit().await;
                    }
                    _ => {
                        missing_keys.push(key);
                        self.update_stats_cache_miss().await;
                    }
                }
                self.update_stats_request().await;
            }
        } else {
            missing_keys = keys;
            for _ in &missing_keys {
                self.update_stats_request().await;
                self.update_stats_cache_miss().await;
            }
        }

        if missing_keys.is_empty() {
            return Ok(results);
        }

        // Load missing keys in batches
        let batch_results = self.load_batch_direct(missing_keys).await?;
        results.extend(batch_results);

        Ok(results)
    }

    /// Prime the cache with a value
    pub async fn prime(&self, key: K, value: V) {
        if self.config.enable_cache {
            self.set_in_cache(key, value).await;
        }
    }

    /// Clear the cache
    pub async fn clear_cache(&self) {
        if self.config.enable_cache {
            let mut cache = self.cache.write().await;
            cache.clear();
        }
    }

    /// Get performance statistics
    pub async fn get_stats(&self) -> DataLoaderStats {
        self.stats.read().await.clone()
    }

    /// Clear statistics
    pub async fn clear_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = DataLoaderStats::default();
    }

    async fn get_from_cache(&self, key: &K) -> Option<V> {
        let cache = self.cache.read().await;
        if let Some(entry) = cache.get(key) {
            if !entry.is_expired() {
                return Some(entry.value.clone());
            }
        }
        None
    }

    async fn set_in_cache(&self, key: K, value: V) {
        let mut cache = self.cache.write().await;

        // Evict expired entries
        self.evict_expired_entries(&mut cache);

        // Enforce size limit
        if cache.len() >= self.config.max_cache_size {
            // Simple LRU: remove one entry
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }

        cache.insert(key, CachedEntry::new(value, self.config.cache_ttl));
    }

    fn evict_expired_entries(&self, cache: &mut HashMap<K, CachedEntry<V>>) {
        let expired_keys: Vec<K> = cache
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            cache.remove(&key);
        }
    }

    async fn add_to_batch(
        &self,
        key: K,
        resolver: tokio::sync::oneshot::Sender<Result<Option<V>>>,
    ) {
        let mut pending = self.pending_batch.lock().await;
        if pending.is_none() {
            *pending = Some(PendingBatch::new());
        }

        if let Some(batch) = pending.as_mut() {
            batch.add_request(key, resolver);
        }
    }

    async fn load_batch_direct(&self, keys: Vec<K>) -> Result<HashMap<K, V>> {
        let start_time = Instant::now();
        let result = self.batch_fn.load(keys).await;
        let load_time = start_time.elapsed();

        // Update stats
        self.update_stats_batch_dispatched(load_time).await;

        match result {
            Ok(loaded_values) => {
                // Cache the results
                if self.config.enable_cache {
                    for (key, value) in &loaded_values {
                        self.set_in_cache(key.clone(), value.clone()).await;
                    }
                }
                Ok(loaded_values)
            }
            Err(e) => Err(e),
        }
    }

    fn start_batch_dispatcher(&self) {
        let pending_batch = Arc::clone(&self.pending_batch);
        let batch_fn = Arc::clone(&self.batch_fn);
        let config = self.config.clone();
        let cache = Arc::clone(&self.cache);
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            loop {
                sleep(config.batch_delay).await;

                let batch_to_dispatch = {
                    let mut pending = pending_batch.lock().await;
                    if let Some(batch) = pending.as_ref() {
                        if batch.should_dispatch(&config) {
                            pending.take()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                if let Some(batch) = batch_to_dispatch {
                    Self::dispatch_batch(batch, &batch_fn, &config, &cache, &stats).await;
                }
            }
        });
    }

    async fn dispatch_batch(
        batch: PendingBatch<K, V>,
        batch_fn: &Arc<dyn BatchLoadFn<K, V>>,
        config: &DataLoaderConfig,
        cache: &Arc<RwLock<HashMap<K, CachedEntry<V>>>>,
        stats: &Arc<RwLock<DataLoaderStats>>,
    ) {
        let start_time = Instant::now();
        let batch_size = batch.keys.len();

        // Update stats
        {
            let mut stats = stats.write().await;
            stats.batches_dispatched += 1;
            stats.avg_batch_size =
                (stats.avg_batch_size + batch_size as f64) / stats.batches_dispatched as f64;
        }

        match batch_fn.load(batch.keys.clone()).await {
            Ok(loaded_values) => {
                let load_time = start_time.elapsed();

                // Update load time stats
                {
                    let mut stats = stats.write().await;
                    stats.total_load_time += load_time;
                }

                // Cache results if enabled
                if config.enable_cache {
                    let mut cache = cache.write().await;
                    for (key, value) in &loaded_values {
                        cache.insert(
                            key.clone(),
                            CachedEntry::new(value.clone(), config.cache_ttl),
                        );
                    }
                }

                // Send results to resolvers
                for (key, resolver) in batch.keys.into_iter().zip(batch.resolvers.into_iter()) {
                    let result = loaded_values.get(&key).cloned();
                    let _ = resolver.send(Ok(result));
                }
            }
            Err(e) => {
                // Send error to all resolvers
                for resolver in batch.resolvers {
                    let _ = resolver.send(Err(anyhow!("Batch load failed: {}", e)));
                }
            }
        }
    }

    async fn update_stats_request(&self) {
        let mut stats = self.stats.write().await;
        stats.requests_total += 1;
    }

    async fn update_stats_cache_hit(&self) {
        let mut stats = self.stats.write().await;
        stats.cache_hits += 1;
    }

    async fn update_stats_cache_miss(&self) {
        let mut stats = self.stats.write().await;
        stats.cache_misses += 1;
    }

    async fn update_stats_batch_dispatched(&self, load_time: Duration) {
        let mut stats = self.stats.write().await;
        stats.batches_dispatched += 1;
        stats.total_load_time += load_time;
    }
}

/// Factory for creating common DataLoaders
pub struct DataLoaderFactory {
    default_config: DataLoaderConfig,
}

impl DataLoaderFactory {
    pub fn new() -> Self {
        Self {
            default_config: DataLoaderConfig::default(),
        }
    }

    pub fn with_config(config: DataLoaderConfig) -> Self {
        Self {
            default_config: config,
        }
    }

    /// Create a DataLoader for loading RDF subjects
    pub fn create_subject_loader(
        &self,
        store: Arc<crate::RdfStore>,
    ) -> DataLoader<String, serde_json::Value> {
        let batch_fn = Arc::new(SubjectBatchLoader { store });
        DataLoader::with_config(batch_fn, self.default_config.clone())
    }

    /// Create a DataLoader for loading RDF predicates
    pub fn create_predicate_loader(
        &self,
        store: Arc<crate::RdfStore>,
    ) -> DataLoader<String, Vec<String>> {
        let batch_fn = Arc::new(PredicateBatchLoader { store });
        DataLoader::with_config(batch_fn, self.default_config.clone())
    }
}

impl Default for DataLoaderFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch loader for RDF subjects
struct SubjectBatchLoader {
    store: Arc<crate::RdfStore>,
}

#[async_trait]
impl BatchLoadFn<String, serde_json::Value> for SubjectBatchLoader {
    async fn load(&self, keys: Vec<String>) -> Result<HashMap<String, serde_json::Value>> {
        let mut results = HashMap::new();

        for key in keys {
            // Build SPARQL query for subject
            let query = format!("SELECT ?p ?o WHERE {{ <{}> ?p ?o }}", key);

            match self.store.query(&query) {
                Ok(crate::QueryResults::Solutions(solutions)) => {
                    let mut triples = Vec::new();
                    for solution in solutions {
                        if let (Some(p), Some(o)) = (
                            solution.get(&oxirs_core::model::Variable::new("p").unwrap()),
                            solution.get(&oxirs_core::model::Variable::new("o").unwrap()),
                        ) {
                            triples.push(serde_json::json!({
                                "predicate": p.to_string(),
                                "object": o.to_string()
                            }));
                        }
                    }
                    results.insert(key, serde_json::json!(triples));
                }
                _ => {
                    // Insert empty result for subjects with no data
                    results.insert(key, serde_json::json!([]));
                }
            }
        }

        Ok(results)
    }
}

/// Batch loader for RDF predicates
struct PredicateBatchLoader {
    store: Arc<crate::RdfStore>,
}

#[async_trait]
impl BatchLoadFn<String, Vec<String>> for PredicateBatchLoader {
    async fn load(&self, keys: Vec<String>) -> Result<HashMap<String, Vec<String>>> {
        let mut results = HashMap::new();

        for key in keys {
            // Build SPARQL query for predicate values
            let query = format!("SELECT DISTINCT ?s WHERE {{ ?s <{}> ?o }}", key);

            match self.store.query(&query) {
                Ok(crate::QueryResults::Solutions(solutions)) => {
                    let mut subjects = Vec::new();
                    for solution in solutions {
                        if let Some(s) =
                            solution.get(&oxirs_core::model::Variable::new("s").unwrap())
                        {
                            subjects.push(s.to_string());
                        }
                    }
                    results.insert(key, subjects);
                }
                _ => {
                    results.insert(key, Vec::new());
                }
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestBatchLoader;

    #[async_trait]
    impl BatchLoadFn<i32, String> for TestBatchLoader {
        async fn load(&self, keys: Vec<i32>) -> Result<HashMap<i32, String>> {
            // Simulate some work
            sleep(Duration::from_millis(10)).await;

            let mut results = HashMap::new();
            for key in keys {
                results.insert(key, format!("value_{key}"));
            }
            Ok(results)
        }
    }

    #[tokio::test]
    async fn test_dataloader_basic() {
        let batch_fn = Arc::new(TestBatchLoader);
        let loader = DataLoader::new(batch_fn);

        let result = loader.load(1).await.unwrap();
        assert_eq!(result, Some("value_1".to_string()));

        let stats = loader.get_stats().await;
        assert_eq!(stats.requests_total, 1);
    }

    #[tokio::test]
    async fn test_dataloader_caching() {
        let batch_fn = Arc::new(TestBatchLoader);
        let loader = DataLoader::new(batch_fn);

        // First load - cache miss
        let result1 = loader.load(1).await.unwrap();
        assert_eq!(result1, Some("value_1".to_string()));

        // Second load - cache hit
        let result2 = loader.load(1).await.unwrap();
        assert_eq!(result2, Some("value_1".to_string()));

        let stats = loader.get_stats().await;
        assert_eq!(stats.requests_total, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[tokio::test]
    async fn test_dataloader_load_many() {
        let batch_fn = Arc::new(TestBatchLoader);
        let loader = DataLoader::new(batch_fn);

        let keys = vec![1, 2, 3];
        let results = loader.load_many(keys).await.unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results.get(&1), Some(&"value_1".to_string()));
        assert_eq!(results.get(&2), Some(&"value_2".to_string()));
        assert_eq!(results.get(&3), Some(&"value_3".to_string()));
    }
}
