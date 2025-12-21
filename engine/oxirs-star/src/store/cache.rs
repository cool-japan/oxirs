//! Cache management for frequently accessed RDF-star data.
//!
//! This module provides LRU-based caching for triple lookups and pattern queries,
//! with configurable eviction policies and performance tracking.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::model::StarTriple;

/// Cache for frequently accessed data
#[derive(Debug)]
pub struct StarCache {
    /// LRU cache for triple lookups
    triple_cache: Arc<RwLock<HashMap<String, Vec<StarTriple>>>>,
    /// Cache for pattern queries
    pattern_cache: Arc<RwLock<HashMap<String, Vec<StarTriple>>>>,
    /// Cache configuration
    config: CacheConfig,
    /// Access frequency tracking
    access_frequency: Arc<RwLock<HashMap<String, usize>>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStatistics>>,
}

/// Configuration for the cache system
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in triple cache
    pub max_triple_entries: usize,
    /// Maximum number of entries in pattern cache
    pub max_pattern_entries: usize,
    /// Time to live for cache entries (seconds)
    pub ttl_seconds: u64,
    /// Enable LRU eviction
    pub enable_lru: bool,
    /// Cache hit rate threshold for optimization
    pub optimization_threshold: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_triple_entries: 10000,
            max_pattern_entries: 5000,
            ttl_seconds: 300, // 5 minutes
            enable_lru: true,
            optimization_threshold: 0.8,
        }
    }
}

/// Cache performance statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_lookups: u64,
}

impl CacheStatistics {
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_lookups as f64
        }
    }
}

impl StarCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            triple_cache: Arc::new(RwLock::new(HashMap::new())),
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            access_frequency: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStatistics::default())),
        }
    }

    /// Get cached results for a query
    pub fn get(&self, key: &str) -> Option<Vec<StarTriple>> {
        let mut stats = self.stats.write().unwrap();
        stats.total_lookups += 1;

        // Check triple cache first
        if let Some(results) = self.triple_cache.read().unwrap().get(key) {
            stats.hits += 1;

            // Update access frequency
            let mut freq = self.access_frequency.write().unwrap();
            *freq.entry(key.to_string()).or_insert(0) += 1;

            return Some(results.clone());
        }

        // Check pattern cache
        if let Some(results) = self.pattern_cache.read().unwrap().get(key) {
            stats.hits += 1;

            let mut freq = self.access_frequency.write().unwrap();
            *freq.entry(key.to_string()).or_insert(0) += 1;

            return Some(results.clone());
        }

        stats.misses += 1;
        None
    }

    /// Store results in cache
    pub fn put(&self, key: String, results: Vec<StarTriple>) {
        // Simple LRU eviction if cache is full
        if self.config.enable_lru {
            let mut cache = self.triple_cache.write().unwrap();
            if cache.len() >= self.config.max_triple_entries {
                // Remove least frequently used entry
                if let Some(lfu_key) = self.find_least_frequent_key() {
                    cache.remove(&lfu_key);
                    let mut stats = self.stats.write().unwrap();
                    stats.evictions += 1;
                }
            }
            cache.insert(key, results);
        }
    }

    fn find_least_frequent_key(&self) -> Option<String> {
        let freq = self.access_frequency.read().unwrap();
        freq.iter()
            .min_by_key(|&(_, &count)| count)
            .map(|(key, _)| key.clone())
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> CacheStatistics {
        self.stats.read().unwrap().clone()
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        self.triple_cache.write().unwrap().clear();
        self.pattern_cache.write().unwrap().clear();
        self.access_frequency.write().unwrap().clear();
    }
}
