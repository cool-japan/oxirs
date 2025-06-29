//! Advanced GraphQL Caching System
//!
//! This module provides sophisticated caching strategies including adaptive TTL,
//! cache warming, intelligent invalidation, and multi-tier caching.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex as AsyncMutex, RwLock as AsyncRwLock};
use tokio::time::{interval, sleep};

/// Cache configuration with adaptive settings
#[derive(Debug, Clone)]
pub struct AdvancedCacheConfig {
    pub l1_max_size: usize,             // In-memory cache size
    pub l2_max_size: usize,             // Persistent cache size
    pub default_ttl: Duration,          // Default cache TTL
    pub adaptive_ttl: bool,             // Enable adaptive TTL
    pub max_ttl: Duration,              // Maximum TTL
    pub min_ttl: Duration,              // Minimum TTL
    pub cache_warming: bool,            // Enable cache warming
    pub intelligent_invalidation: bool, // Enable smart invalidation
    pub compression: bool,              // Enable cache compression
    pub persistence: bool,              // Enable persistent cache
    pub metrics_collection: bool,       // Enable detailed metrics
}

impl Default for AdvancedCacheConfig {
    fn default() -> Self {
        Self {
            l1_max_size: 1000,
            l2_max_size: 10000,
            default_ttl: Duration::from_secs(300),
            adaptive_ttl: true,
            max_ttl: Duration::from_secs(3600),
            min_ttl: Duration::from_secs(30),
            cache_warming: true,
            intelligent_invalidation: true,
            compression: true,
            persistence: false,
            metrics_collection: true,
        }
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub data: serde_json::Value,
    pub created_at: SystemTime,
    pub ttl: Duration,
    pub access_count: u32,
    pub last_accessed: SystemTime,
    pub size_bytes: usize,
    pub query_complexity: usize,
    pub dependencies: HashSet<String>,
    pub tags: HashSet<String>,
}

impl CacheEntry {
    pub fn new(data: serde_json::Value, ttl: Duration) -> Self {
        let size_bytes = data.to_string().len();
        Self {
            data,
            created_at: SystemTime::now(),
            ttl,
            access_count: 0,
            last_accessed: SystemTime::now(),
            size_bytes,
            query_complexity: 0,
            dependencies: HashSet::new(),
            tags: HashSet::new(),
        }
    }

    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed().unwrap_or(Duration::from_secs(0)) > self.ttl
    }

    pub fn access(&mut self) {
        self.access_count += 1;
        self.last_accessed = SystemTime::now();
    }

    pub fn with_complexity(mut self, complexity: usize) -> Self {
        self.query_complexity = complexity;
        self
    }

    pub fn with_dependencies(mut self, deps: HashSet<String>) -> Self {
        self.dependencies = deps;
        self
    }

    pub fn with_tags(mut self, tags: HashSet<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Cache statistics and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub evictions: u64,
    pub invalidations: u64,
    pub warming_operations: u64,
    pub total_size_bytes: usize,
    pub average_ttl: Duration,
    pub hit_ratio: f64,
    pub popular_queries: Vec<PopularQuery>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopularQuery {
    pub cache_key: String,
    pub access_count: u32,
    pub avg_execution_time: Duration,
    pub last_accessed: SystemTime,
}

/// Cache warming strategy
#[derive(Debug, Clone)]
pub enum WarmingStrategy {
    Popular,    // Warm most popular queries
    Predictive, // Warm based on usage patterns
    Schedule,   // Warm on schedule
    Dependency, // Warm based on dependencies
}

/// Cache invalidation strategy
#[derive(Debug, Clone)]
pub enum InvalidationStrategy {
    TTL,        // Time-based invalidation
    Manual,     // Manual invalidation
    Dependency, // Dependency-based invalidation
    Smart,      // Intelligent invalidation
}

/// Advanced multi-tier cache system
pub struct AdvancedCache {
    config: AdvancedCacheConfig,
    l1_cache: Arc<AsyncRwLock<HashMap<String, CacheEntry>>>,
    l2_cache: Arc<AsyncRwLock<HashMap<String, CacheEntry>>>,
    metrics: Arc<AsyncRwLock<CacheMetrics>>,
    access_patterns: Arc<AsyncRwLock<HashMap<String, AccessPattern>>>,
    dependency_graph: Arc<AsyncRwLock<DependencyGraph>>,
    warming_queue: Arc<AsyncMutex<Vec<WarmingTask>>>,
}

/// Access pattern tracking
#[derive(Debug, Clone)]
struct AccessPattern {
    access_times: VecDeque<Instant>,
    frequency: f64,
    last_prediction: Option<Instant>,
}

use std::collections::VecDeque;

impl AccessPattern {
    fn new() -> Self {
        Self {
            access_times: VecDeque::new(),
            frequency: 0.0,
            last_prediction: None,
        }
    }

    fn record_access(&mut self, now: Instant) {
        self.access_times.push_back(now);

        // Keep only recent accesses (last hour)
        let cutoff = now - Duration::from_secs(3600);
        while let Some(&front) = self.access_times.front() {
            if front < cutoff {
                self.access_times.pop_front();
            } else {
                break;
            }
        }

        // Calculate frequency (accesses per hour)
        self.frequency = self.access_times.len() as f64;
    }

    fn predict_next_access(&self) -> Option<Instant> {
        if self.access_times.len() < 2 {
            return None;
        }

        // Simple prediction based on average interval
        let intervals: Vec<Duration> = self
            .access_times
            .iter()
            .zip(self.access_times.iter().skip(1))
            .map(|(first, second)| *second - *first)
            .collect();

        if intervals.is_empty() {
            return None;
        }

        let total_nanos =
            intervals.iter().map(|d| d.as_nanos()).sum::<u128>() / intervals.len() as u128;
        let avg_interval = Duration::from_nanos(total_nanos.min(u64::MAX as u128) as u64);

        self.access_times.back().map(|&last| last + avg_interval)
    }
}

/// Dependency tracking graph
#[derive(Debug, Default)]
struct DependencyGraph {
    dependencies: HashMap<String, HashSet<String>>,
    dependents: HashMap<String, HashSet<String>>,
}

impl DependencyGraph {
    fn add_dependency(&mut self, cache_key: &str, dependency: &str) {
        self.dependencies
            .entry(cache_key.to_string())
            .or_insert_with(HashSet::new)
            .insert(dependency.to_string());

        self.dependents
            .entry(dependency.to_string())
            .or_insert_with(HashSet::new)
            .insert(cache_key.to_string());
    }

    fn get_affected_keys(&self, changed_key: &str) -> HashSet<String> {
        let mut affected = HashSet::new();
        let mut to_check = vec![changed_key.to_string()];

        while let Some(key) = to_check.pop() {
            if let Some(dependents) = self.dependents.get(&key) {
                for dependent in dependents {
                    if affected.insert(dependent.clone()) {
                        to_check.push(dependent.clone());
                    }
                }
            }
        }

        affected
    }
}

/// Cache warming task
#[derive(Debug, Clone)]
struct WarmingTask {
    cache_key: String,
    query: String,
    variables: HashMap<String, serde_json::Value>,
    priority: u8,
    scheduled_time: Instant,
}

impl AdvancedCache {
    /// Create a new advanced cache system
    pub fn new(config: AdvancedCacheConfig) -> Self {
        let cache = Self {
            config: config.clone(),
            l1_cache: Arc::new(AsyncRwLock::new(HashMap::new())),
            l2_cache: Arc::new(AsyncRwLock::new(HashMap::new())),
            metrics: Arc::new(AsyncRwLock::new(CacheMetrics {
                l1_hits: 0,
                l1_misses: 0,
                l2_hits: 0,
                l2_misses: 0,
                evictions: 0,
                invalidations: 0,
                warming_operations: 0,
                total_size_bytes: 0,
                average_ttl: config.default_ttl,
                hit_ratio: 0.0,
                popular_queries: Vec::new(),
            })),
            access_patterns: Arc::new(AsyncRwLock::new(HashMap::new())),
            dependency_graph: Arc::new(AsyncRwLock::new(DependencyGraph::default())),
            warming_queue: Arc::new(AsyncMutex::new(Vec::new())),
        };

        // Start background tasks
        cache.start_cleanup_task();
        cache.start_warming_task();
        cache.start_metrics_task();

        cache
    }

    /// Get a value from cache with intelligent tier selection
    pub async fn get(&self, key: &str) -> Option<serde_json::Value> {
        // Try L1 cache first
        {
            let mut l1 = self.l1_cache.write().await;
            if let Some(entry) = l1.get_mut(key) {
                if !entry.is_expired() {
                    entry.access();
                    self.update_access_pattern(key).await;
                    self.record_l1_hit().await;
                    return Some(entry.data.clone());
                } else {
                    l1.remove(key);
                }
            }
        }

        self.record_l1_miss().await;

        // Try L2 cache
        {
            let mut l2 = self.l2_cache.write().await;
            if let Some(entry) = l2.get_mut(key) {
                if !entry.is_expired() {
                    entry.access();

                    // Promote to L1 if frequently accessed
                    if entry.access_count > 5 {
                        self.promote_to_l1(key.to_string(), entry.clone()).await;
                    }

                    self.update_access_pattern(key).await;
                    self.record_l2_hit().await;
                    return Some(entry.data.clone());
                } else {
                    l2.remove(key);
                }
            }
        }

        self.record_l2_miss().await;
        None
    }

    /// Store a value in cache with intelligent tier placement
    pub async fn set(
        &self,
        key: String,
        value: serde_json::Value,
        ttl: Option<Duration>,
        complexity: Option<usize>,
        dependencies: Option<HashSet<String>>,
        tags: Option<HashSet<String>>,
    ) {
        let effective_ttl = if self.config.adaptive_ttl {
            self.calculate_adaptive_ttl(&key, complexity.unwrap_or(0))
                .await
        } else {
            ttl.unwrap_or(self.config.default_ttl)
        };

        let mut entry = CacheEntry::new(value, effective_ttl);

        if let Some(complexity) = complexity {
            entry = entry.with_complexity(complexity);
        }

        if let Some(deps) = dependencies {
            entry = entry.with_dependencies(deps.clone());
            self.add_dependencies(&key, &deps).await;
        }

        if let Some(tags) = tags {
            entry = entry.with_tags(tags);
        }

        // Decide tier placement based on expected access pattern
        let should_use_l1 = self.should_place_in_l1(&key, &entry).await;

        if should_use_l1 {
            self.set_l1(key, entry).await;
        } else {
            self.set_l2(key, entry).await;
        }
    }

    /// Intelligent cache invalidation
    pub async fn invalidate(&self, key: &str, strategy: InvalidationStrategy) {
        match strategy {
            InvalidationStrategy::TTL => {
                // Natural TTL expiration - no action needed
            }
            InvalidationStrategy::Manual => {
                self.remove_from_all_tiers(key).await;
            }
            InvalidationStrategy::Dependency => {
                let affected_keys = self.get_affected_keys(key).await;
                for affected_key in affected_keys {
                    self.remove_from_all_tiers(&affected_key).await;
                }
            }
            InvalidationStrategy::Smart => {
                self.smart_invalidation(key).await;
            }
        }

        self.record_invalidation().await;
    }

    /// Warm cache with predicted queries
    pub async fn warm_cache(&self, strategy: WarmingStrategy) {
        match strategy {
            WarmingStrategy::Popular => {
                self.warm_popular_queries().await;
            }
            WarmingStrategy::Predictive => {
                self.warm_predicted_queries().await;
            }
            WarmingStrategy::Schedule => {
                self.warm_scheduled_queries().await;
            }
            WarmingStrategy::Dependency => {
                self.warm_dependency_queries().await;
            }
        }

        self.record_warming_operation().await;
    }

    /// Get comprehensive cache metrics
    pub async fn get_metrics(&self) -> CacheMetrics {
        let mut metrics = self.metrics.read().await.clone();

        // Calculate hit ratio
        let total_requests =
            metrics.l1_hits + metrics.l1_misses + metrics.l2_hits + metrics.l2_misses;
        if total_requests > 0 {
            metrics.hit_ratio = (metrics.l1_hits + metrics.l2_hits) as f64 / total_requests as f64;
        }

        // Update popular queries
        metrics.popular_queries = self.get_popular_queries().await;

        metrics
    }

    /// Clear all caches
    pub async fn clear_all(&self) {
        self.l1_cache.write().await.clear();
        self.l2_cache.write().await.clear();
        self.access_patterns.write().await.clear();
        self.dependency_graph.write().await.dependencies.clear();
        self.dependency_graph.write().await.dependents.clear();
    }

    // Private helper methods

    async fn promote_to_l1(&self, key: String, entry: CacheEntry) {
        self.set_l1(key, entry).await;
    }

    async fn set_l1(&self, key: String, entry: CacheEntry) {
        let mut l1 = self.l1_cache.write().await;

        // Evict if necessary
        while l1.len() >= self.config.l1_max_size {
            self.evict_l1_entry(&mut l1).await;
        }

        l1.insert(key, entry);
    }

    async fn set_l2(&self, key: String, entry: CacheEntry) {
        let mut l2 = self.l2_cache.write().await;

        // Evict if necessary
        while l2.len() >= self.config.l2_max_size {
            self.evict_l2_entry(&mut l2).await;
        }

        l2.insert(key, entry);
    }

    async fn evict_l1_entry(&self, l1: &mut HashMap<String, CacheEntry>) {
        // LRU eviction
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = SystemTime::now();

        for (key, entry) in l1.iter() {
            if entry.last_accessed < oldest_time {
                oldest_time = entry.last_accessed;
                oldest_key = Some(key.clone());
            }
        }

        if let Some(key) = oldest_key {
            if let Some(entry) = l1.remove(&key) {
                // Demote to L2 if still valuable
                if entry.access_count > 1 && !entry.is_expired() {
                    self.set_l2(key, entry).await;
                }
            }
        }

        self.record_eviction().await;
    }

    async fn evict_l2_entry(&self, l2: &mut HashMap<String, CacheEntry>) {
        // LRU eviction for L2
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = SystemTime::now();

        for (key, entry) in l2.iter() {
            if entry.last_accessed < oldest_time {
                oldest_time = entry.last_accessed;
                oldest_key = Some(key.clone());
            }
        }

        if let Some(key) = oldest_key {
            l2.remove(&key);
        }

        self.record_eviction().await;
    }

    async fn should_place_in_l1(&self, key: &str, _entry: &CacheEntry) -> bool {
        // Check access patterns to decide tier placement
        let patterns = self.access_patterns.read().await;
        if let Some(pattern) = patterns.get(key) {
            pattern.frequency > 10.0 // More than 10 accesses per hour
        } else {
            false // New entries go to L2 first
        }
    }

    async fn calculate_adaptive_ttl(&self, key: &str, complexity: usize) -> Duration {
        let patterns = self.access_patterns.read().await;

        let base_ttl = if let Some(pattern) = patterns.get(key) {
            // Higher frequency = longer TTL
            let frequency_factor = (pattern.frequency / 10.0).min(3.0).max(0.5);
            Duration::from_secs(
                (self.config.default_ttl.as_secs() as f64 * frequency_factor) as u64,
            )
        } else {
            self.config.default_ttl
        };

        // Adjust based on complexity (more complex = longer TTL)
        let complexity_factor = 1.0 + (complexity as f64 / 100.0).min(2.0);
        let adjusted_ttl =
            Duration::from_secs((base_ttl.as_secs() as f64 * complexity_factor) as u64);

        // Clamp to min/max TTL
        adjusted_ttl
            .max(self.config.min_ttl)
            .min(self.config.max_ttl)
    }

    async fn update_access_pattern(&self, key: &str) {
        let mut patterns = self.access_patterns.write().await;
        let pattern = patterns
            .entry(key.to_string())
            .or_insert_with(AccessPattern::new);
        pattern.record_access(Instant::now());
    }

    async fn add_dependencies(&self, key: &str, dependencies: &HashSet<String>) {
        let mut graph = self.dependency_graph.write().await;
        for dep in dependencies {
            graph.add_dependency(key, dep);
        }
    }

    async fn get_affected_keys(&self, key: &str) -> HashSet<String> {
        let graph = self.dependency_graph.read().await;
        graph.get_affected_keys(key)
    }

    async fn remove_from_all_tiers(&self, key: &str) {
        self.l1_cache.write().await.remove(key);
        self.l2_cache.write().await.remove(key);
    }

    async fn smart_invalidation(&self, key: &str) {
        // Implement intelligent invalidation logic
        // For now, just remove from all tiers
        self.remove_from_all_tiers(key).await;
    }

    async fn warm_popular_queries(&self) {
        // Implementation for warming popular queries
    }

    async fn warm_predicted_queries(&self) {
        // Implementation for warming predicted queries based on patterns
    }

    async fn warm_scheduled_queries(&self) {
        // Implementation for warming scheduled queries
    }

    async fn warm_dependency_queries(&self) {
        // Implementation for warming based on dependencies
    }

    async fn get_popular_queries(&self) -> Vec<PopularQuery> {
        let mut popular = Vec::new();

        // Collect from both tiers
        let l1 = self.l1_cache.read().await;
        let l2 = self.l2_cache.read().await;

        for (key, entry) in l1.iter().chain(l2.iter()) {
            if entry.access_count > 5 {
                popular.push(PopularQuery {
                    cache_key: key.clone(),
                    access_count: entry.access_count,
                    avg_execution_time: Duration::from_millis(100), // Placeholder
                    last_accessed: entry.last_accessed,
                });
            }
        }

        popular.sort_by(|a, b| b.access_count.cmp(&a.access_count));
        popular.truncate(10);
        popular
    }

    // Metrics recording methods
    async fn record_l1_hit(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.l1_hits += 1;
    }

    async fn record_l1_miss(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.l1_misses += 1;
    }

    async fn record_l2_hit(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.l2_hits += 1;
    }

    async fn record_l2_miss(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.l2_misses += 1;
    }

    async fn record_eviction(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.evictions += 1;
    }

    async fn record_invalidation(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.invalidations += 1;
    }

    async fn record_warming_operation(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.warming_operations += 1;
    }

    // Background task methods
    fn start_cleanup_task(&self) {
        let l1_cache = Arc::clone(&self.l1_cache);
        let l2_cache = Arc::clone(&self.l2_cache);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                // Clean expired entries from L1
                {
                    let mut l1 = l1_cache.write().await;
                    l1.retain(|_, entry| !entry.is_expired());
                }

                // Clean expired entries from L2
                {
                    let mut l2 = l2_cache.write().await;
                    l2.retain(|_, entry| !entry.is_expired());
                }
            }
        });
    }

    fn start_warming_task(&self) {
        if !self.config.cache_warming {
            return;
        }

        let warming_queue = Arc::clone(&self.warming_queue);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(300)); // Every 5 minutes

            loop {
                interval.tick().await;

                // Process warming queue
                let mut queue = warming_queue.lock().await;
                let now = Instant::now();

                queue.retain(|task| {
                    if task.scheduled_time <= now {
                        // Process warming task
                        // Implementation would execute the query and cache result
                        false // Remove from queue
                    } else {
                        true // Keep in queue
                    }
                });
            }
        });
    }

    fn start_metrics_task(&self) {
        if !self.config.metrics_collection {
            return;
        }

        let metrics = Arc::clone(&self.metrics);
        let l1_cache = Arc::clone(&self.l1_cache);
        let l2_cache = Arc::clone(&self.l2_cache);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                // Update size metrics
                let mut metrics = metrics.write().await;
                let l1 = l1_cache.read().await;
                let l2 = l2_cache.read().await;

                metrics.total_size_bytes = l1
                    .values()
                    .chain(l2.values())
                    .map(|entry| entry.size_bytes)
                    .sum();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_advanced_cache_basic_operations() {
        let config = AdvancedCacheConfig::default();
        let cache = AdvancedCache::new(config);

        let key = "test_key".to_string();
        let value = serde_json::json!({"test": "value"});

        // Test set and get
        cache
            .set(key.clone(), value.clone(), None, None, None, None)
            .await;
        let result = cache.get(&key).await;

        assert_eq!(result, Some(value));
    }

    #[tokio::test]
    async fn test_cache_tiers() {
        let mut config = AdvancedCacheConfig::default();
        config.l1_max_size = 1; // Force L2 usage

        let cache = AdvancedCache::new(config);

        // Fill L1
        cache
            .set(
                "key1".to_string(),
                serde_json::json!(1),
                None,
                None,
                None,
                None,
            )
            .await;
        cache
            .set(
                "key2".to_string(),
                serde_json::json!(2),
                None,
                None,
                None,
                None,
            )
            .await;

        // key1 should be evicted to L2
        let result1 = cache.get("key1").await;
        let result2 = cache.get("key2").await;

        assert_eq!(result1, Some(serde_json::json!(1)));
        assert_eq!(result2, Some(serde_json::json!(2)));
    }

    #[tokio::test]
    async fn test_dependency_invalidation() {
        let cache = AdvancedCache::new(AdvancedCacheConfig::default());

        let mut deps = HashSet::new();
        deps.insert("dependency1".to_string());

        cache
            .set(
                "main_key".to_string(),
                serde_json::json!({"data": "test"}),
                None,
                None,
                Some(deps),
                None,
            )
            .await;

        // Invalidate dependency
        cache
            .invalidate("dependency1", InvalidationStrategy::Dependency)
            .await;

        // Main key should be invalidated
        let result = cache.get("main_key").await;
        assert_eq!(result, None);
    }
}
