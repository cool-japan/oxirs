//! Query Plan Caching System
//!
//! This module provides intelligent caching of optimized SPARQL query plans
//! to avoid redundant optimization work for frequently executed queries.
//!
//! ## Features
//! - LRU-based cache eviction
//! - Cache invalidation based on statistics changes
//! - Parameterized query support
//! - Cache hit/miss tracking
//! - TTL-based expiration

use crate::algebra::Algebra;
use crate::cache::CacheCoordinator;
use crate::optimizer::Statistics;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Query plan cache for avoiding redundant optimization
pub struct QueryPlanCache {
    /// Cache storage (query signature -> cached plan)
    cache: Arc<DashMap<QuerySignature, CachedPlan>>,
    /// Configuration
    config: CachingConfig,
    /// Cache statistics
    stats: Arc<CacheStatistics>,
    /// LRU tracking (access order)
    access_counter: Arc<AtomicU64>,
    /// Invalidation coordinator (optional for backward compatibility)
    invalidation_coordinator: Option<Arc<CacheCoordinator>>,
    /// Invalidation flags (tracks which entries have been invalidated)
    invalidated_entries: Arc<dashmap::DashSet<QuerySignature>>,
}

/// Configuration for query plan caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable caching
    pub enabled: bool,
    /// Maximum number of cached plans
    pub max_cache_size: usize,
    /// Time-to-live for cached plans (in seconds)
    pub ttl_seconds: u64,
    /// Enable parameterized query support
    pub parameterized_queries: bool,
    /// Invalidate cache when statistics change significantly
    pub invalidate_on_stats_change: bool,
    /// Threshold for statistics change (0.0 to 1.0)
    pub stats_change_threshold: f64,
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_cache_size: 10000,
            ttl_seconds: 3600, // 1 hour
            parameterized_queries: true,
            invalidate_on_stats_change: true,
            stats_change_threshold: 0.2, // 20% change
        }
    }
}

/// Signature for uniquely identifying a query
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct QuerySignature {
    /// Normalized query string (with parameters replaced)
    normalized_query: String,
    /// Parameter types for parameterized queries
    parameter_types: Vec<String>,
    /// Statistics hash (to detect when stats have changed)
    stats_hash: u64,
}

impl QuerySignature {
    /// Create a new query signature
    pub fn new(query: &str, params: Vec<String>, stats: &Statistics) -> Self {
        Self {
            normalized_query: Self::normalize_query(query),
            parameter_types: params,
            stats_hash: Self::hash_statistics(stats),
        }
    }

    /// Normalize query by replacing literals with placeholders
    fn normalize_query(query: &str) -> String {
        // Simplified normalization - replace numeric literals and strings with placeholders
        let mut normalized = query.to_string();

        // Replace string literals: "..." -> "?"
        let re_string = regex::Regex::new(r#""[^"]*""#).expect("regex pattern should be valid");
        normalized = re_string.replace_all(&normalized, "\"?\"").to_string();

        // Replace numeric literals: 123 -> ?
        let re_number =
            regex::Regex::new(r"\b\d+(\.\d+)?\b").expect("regex pattern should be valid");
        normalized = re_number.replace_all(&normalized, "?").to_string();

        // Collapse whitespace
        let re_whitespace = regex::Regex::new(r"\s+").expect("regex pattern should be valid");
        re_whitespace.replace_all(&normalized, " ").to_string()
    }

    /// Hash statistics to detect changes
    fn hash_statistics(stats: &Statistics) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();

        // Hash cardinalities
        for (pattern, card) in &stats.cardinalities {
            pattern.hash(&mut hasher);
            card.hash(&mut hasher);
        }

        // Hash predicate frequencies
        for (pred, freq) in &stats.predicate_frequency {
            pred.hash(&mut hasher);
            freq.hash(&mut hasher);
        }

        hasher.finish()
    }
}

/// Cached query plan with metadata
#[derive(Debug, Clone)]
pub struct CachedPlan {
    /// Optimized query plan
    pub plan: Algebra,
    /// When the plan was cached
    pub cached_at: Instant,
    /// How many times this plan was used
    pub hit_count: Arc<AtomicUsize>,
    /// Last access timestamp (for LRU)
    pub last_accessed: Arc<AtomicU64>,
    /// Estimated cost of the plan
    pub estimated_cost: f64,
    /// Statistics snapshot at cache time
    pub stats_snapshot: StatisticsSnapshot,
}

/// Snapshot of statistics for cache invalidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsSnapshot {
    /// Cardinalities at cache time
    pub cardinalities: BTreeMap<String, usize>,
    /// Predicate frequencies at cache time
    pub predicate_frequency: BTreeMap<String, usize>,
    /// Timestamp when snapshot was taken
    pub snapshot_time: u64,
}

impl StatisticsSnapshot {
    /// Create from Statistics
    pub fn from_statistics(stats: &Statistics) -> Self {
        Self {
            cardinalities: stats
                .cardinalities
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            predicate_frequency: stats
                .predicate_frequency
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            snapshot_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
        }
    }

    /// Check if statistics have changed significantly
    pub fn has_changed_significantly(&self, current_stats: &Statistics, threshold: f64) -> bool {
        // Check cardinality changes
        for (pattern, old_card) in &self.cardinalities {
            let current_card = current_stats
                .cardinalities
                .get(pattern)
                .copied()
                .unwrap_or(0);

            if *old_card == 0 && current_card > 0 {
                return true; // New pattern appeared
            }

            if *old_card > 0 {
                let change_ratio =
                    (current_card as f64 - *old_card as f64).abs() / *old_card as f64;
                if change_ratio > threshold {
                    return true;
                }
            }
        }

        // Check predicate frequency changes
        for (pred, old_freq) in &self.predicate_frequency {
            let current_freq = current_stats
                .predicate_frequency
                .get(pred)
                .copied()
                .unwrap_or(0);

            if *old_freq > 0 {
                let change_ratio =
                    (current_freq as f64 - *old_freq as f64).abs() / *old_freq as f64;
                if change_ratio > threshold {
                    return true;
                }
            }
        }

        false
    }
}

/// Cache hit/miss statistics
#[derive(Debug, Default)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: AtomicU64,
    /// Total cache misses
    pub misses: AtomicU64,
    /// Total evictions
    pub evictions: AtomicU64,
    /// Total invalidations
    pub invalidations: AtomicU64,
    /// Total size in bytes (approximate)
    pub size_bytes: AtomicU64,
}

impl CacheStatistics {
    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Get total requests
    pub fn total_requests(&self) -> u64 {
        self.hits.load(Ordering::Relaxed) + self.misses.load(Ordering::Relaxed)
    }
}

impl QueryPlanCache {
    /// Create a new query plan cache
    pub fn new() -> Self {
        Self::with_config(CachingConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: CachingConfig) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            config,
            stats: Arc::new(CacheStatistics::default()),
            access_counter: Arc::new(AtomicU64::new(0)),
            invalidation_coordinator: None,
            invalidated_entries: Arc::new(dashmap::DashSet::new()),
        }
    }

    /// Create with invalidation coordinator
    pub fn with_invalidation_coordinator(
        config: CachingConfig,
        coordinator: Arc<CacheCoordinator>,
    ) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            config,
            stats: Arc::new(CacheStatistics::default()),
            access_counter: Arc::new(AtomicU64::new(0)),
            invalidation_coordinator: Some(coordinator),
            invalidated_entries: Arc::new(dashmap::DashSet::new()),
        }
    }

    /// Attach invalidation coordinator
    pub fn attach_coordinator(&mut self, coordinator: Arc<CacheCoordinator>) {
        self.invalidation_coordinator = Some(coordinator);
    }

    /// Get a cached plan if available
    pub fn get(
        &self,
        query: &str,
        params: Vec<String>,
        current_stats: &Statistics,
    ) -> Option<Algebra> {
        if !self.config.enabled {
            return None;
        }

        let signature = QuerySignature::new(query, params, current_stats);

        // Check if entry has been invalidated
        if self.invalidated_entries.contains(&signature) {
            self.stats.invalidations.fetch_add(1, Ordering::Relaxed);
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        if let Some(entry) = self.cache.get_mut(&signature) {
            // Check TTL
            let elapsed = entry.cached_at.elapsed();
            if elapsed.as_secs() > self.config.ttl_seconds {
                drop(entry); // Release lock before removing
                self.cache.remove(&signature);
                self.stats.invalidations.fetch_add(1, Ordering::Relaxed);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Check if statistics have changed significantly
            if self.config.invalidate_on_stats_change
                && entry
                    .stats_snapshot
                    .has_changed_significantly(current_stats, self.config.stats_change_threshold)
            {
                drop(entry); // Release lock before removing
                self.cache.remove(&signature);
                self.stats.invalidations.fetch_add(1, Ordering::Relaxed);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Update access tracking
            entry.hit_count.fetch_add(1, Ordering::Relaxed);
            let access_time = self.access_counter.fetch_add(1, Ordering::Relaxed);
            entry.last_accessed.store(access_time, Ordering::Relaxed);

            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Some(entry.plan.clone());
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Cache a query plan
    pub fn insert(
        &self,
        query: &str,
        params: Vec<String>,
        plan: Algebra,
        estimated_cost: f64,
        current_stats: &Statistics,
    ) {
        if !self.config.enabled {
            return;
        }

        // Evict entries if cache is full
        if self.cache.len() >= self.config.max_cache_size {
            self.evict_lru();
        }

        let signature = QuerySignature::new(query, params, current_stats);

        let cached_plan = CachedPlan {
            plan,
            cached_at: Instant::now(),
            hit_count: Arc::new(AtomicUsize::new(0)),
            last_accessed: Arc::new(AtomicU64::new(self.access_counter.load(Ordering::Relaxed))),
            estimated_cost,
            stats_snapshot: StatisticsSnapshot::from_statistics(current_stats),
        };

        self.cache.insert(signature, cached_plan);
    }

    /// Evict least recently used entry
    fn evict_lru(&self) {
        // Find entry with oldest access time
        let mut oldest_key = None;
        let mut oldest_access = u64::MAX;

        for entry in self.cache.iter() {
            let access_time = entry.last_accessed.load(Ordering::Relaxed);
            if access_time < oldest_access {
                oldest_access = access_time;
                oldest_key = Some(entry.key().clone());
            }
        }

        if let Some(key) = oldest_key {
            self.cache.remove(&key);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Clear the entire cache
    pub fn clear(&self) {
        let count = self.cache.len();
        self.cache.clear();
        self.stats
            .invalidations
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    /// Invalidate entries that reference a specific pattern
    pub fn invalidate_pattern(&self, pattern: &str) {
        let keys_to_remove: Vec<_> = self
            .cache
            .iter()
            .filter(|entry| entry.stats_snapshot.cardinalities.contains_key(pattern))
            .map(|entry| entry.key().clone())
            .collect();

        for key in keys_to_remove {
            self.invalidated_entries.insert(key.clone());
            self.cache.remove(&key);
            self.stats.invalidations.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Mark entry as invalidated without removing (for batched invalidation)
    pub fn mark_invalidated(&self, signature: QuerySignature) {
        self.invalidated_entries.insert(signature);
        self.stats.invalidations.fetch_add(1, Ordering::Relaxed);
    }

    /// Invalidate by signature (for coordinator integration)
    pub fn invalidate_signature(&self, signature: &QuerySignature) {
        self.invalidated_entries.insert(signature.clone());
        self.cache.remove(signature);
        self.stats.invalidations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStats {
        CacheStats {
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            invalidations: self.stats.invalidations.load(Ordering::Relaxed),
            size: self.cache.len(),
            capacity: self.config.max_cache_size,
            hit_rate: self.stats.hit_rate(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CachingConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: CachingConfig) {
        self.config = config;
    }
}

impl Default for QueryPlanCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Total invalidations
    pub invalidations: u64,
    /// Current cache size
    pub size: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_query_plan_cache_basic() {
        let cache = QueryPlanCache::new();
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10";
        let stats = Statistics::new();

        // Cache miss on first access
        assert!(cache.get(query, vec![], &stats).is_none());

        // Insert a plan
        let plan = Algebra::Bgp(vec![]);
        cache.insert(query, vec![], plan.clone(), 100.0, &stats);

        // Cache hit on second access
        let cached = cache.get(query, vec![], &stats);
        assert!(cached.is_some());

        // Verify statistics
        let stats = cache.statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_normalization() {
        let stats = Statistics::new();

        let query1 = "SELECT ?s WHERE { ?s <http://example.org/p> \"Alice\" }";
        let query2 = "SELECT ?s WHERE { ?s <http://example.org/p> \"Bob\" }";

        // Both queries should normalize to the same signature
        let sig1 = QuerySignature::new(query1, vec![], &stats);
        let sig2 = QuerySignature::new(query2, vec![], &stats);

        // The normalized versions should be the same
        assert_eq!(sig1.normalized_query, sig2.normalized_query);
    }

    #[test]
    fn test_cache_ttl() {
        let config = CachingConfig {
            ttl_seconds: 1, // 1 second TTL
            ..Default::default()
        };
        let cache = QueryPlanCache::with_config(config);
        let query = "SELECT ?s WHERE { ?s ?p ?o }";
        let stats = Statistics::new();

        // Insert plan
        cache.insert(query, vec![], Algebra::Bgp(vec![]), 100.0, &stats);

        // Should be cached
        assert!(cache.get(query, vec![], &stats).is_some());

        // Wait for TTL to expire
        std::thread::sleep(Duration::from_secs(2));

        // Should be invalidated
        assert!(cache.get(query, vec![], &stats).is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let config = CachingConfig {
            max_cache_size: 2,
            ..Default::default()
        };
        let cache = QueryPlanCache::with_config(config);
        let stats = Statistics::new();

        // Insert 3 plans (should evict oldest)
        cache.insert("query1", vec![], Algebra::Bgp(vec![]), 100.0, &stats);
        cache.insert("query2", vec![], Algebra::Bgp(vec![]), 100.0, &stats);
        cache.insert("query3", vec![], Algebra::Bgp(vec![]), 100.0, &stats);

        // Should have evicted query1
        assert_eq!(cache.cache.len(), 2);

        let stats = cache.statistics();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_cache_clear() {
        let cache = QueryPlanCache::new();
        let stats = Statistics::new();

        // Insert multiple plans with different queries (not just different numbers)
        for i in 0..10 {
            let query = format!("SELECT ?s ?var{} WHERE {{ ?s ?p{} ?o{} }}", i, i, i);
            cache.insert(&query, vec![], Algebra::Bgp(vec![]), 100.0, &stats);
        }

        let initial_len = cache.cache.len();
        assert!(initial_len > 0, "Cache should have entries");

        // Clear cache
        cache.clear();
        assert_eq!(cache.cache.len(), 0);

        let cache_stats = cache.statistics();
        assert_eq!(cache_stats.invalidations, initial_len as u64);
    }

    #[test]
    fn test_statistics_snapshot() {
        let stats = Statistics::new();
        let snapshot = StatisticsSnapshot::from_statistics(&stats);

        // Snapshot should not detect change when stats are the same
        assert!(!snapshot.has_changed_significantly(&stats, 0.2));
    }

    #[test]
    fn test_cache_disabled() {
        let config = CachingConfig {
            enabled: false,
            ..Default::default()
        };
        let cache = QueryPlanCache::with_config(config);
        let stats = Statistics::new();

        // Insert should do nothing when disabled
        cache.insert("query", vec![], Algebra::Bgp(vec![]), 100.0, &stats);

        // Get should always return None when disabled
        assert!(cache.get("query", vec![], &stats).is_none());
    }

    #[test]
    fn test_hit_rate_calculation() {
        let cache = QueryPlanCache::new();
        let stats = Statistics::new();

        // Start with no requests
        assert_eq!(cache.statistics().hit_rate, 0.0);

        // Insert and access
        cache.insert(
            "SELECT ?s WHERE { ?s ?p ?o }",
            vec![],
            Algebra::Bgp(vec![]),
            100.0,
            &stats,
        );
        cache.get("SELECT ?s WHERE { ?s ?p ?o }", vec![], &stats); // Hit
        cache.get("SELECT ?x WHERE { ?x ?y ?z }", vec![], &stats); // Miss

        let cache_stats = cache.statistics();
        assert_eq!(cache_stats.hits, 1);
        assert_eq!(cache_stats.misses, 1); // One from q2
        assert!((cache_stats.hit_rate - 0.5).abs() < 0.01); // 1 hit out of 2 requests = 0.5
    }
}
