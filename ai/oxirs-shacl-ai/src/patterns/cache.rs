//! Caching functionality for pattern analysis

use super::config::PatternCacheSettings;
use super::types::{CachedPatternResult, Pattern};
use crate::{Result, ShaclAiError};
use oxirs_core::Store;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

/// Pattern cache manager
#[derive(Debug)]
pub struct PatternCache {
    cache: HashMap<String, CachedPatternResult>,
    settings: PatternCacheSettings,
}

impl PatternCache {
    /// Create a new pattern cache with settings
    pub fn new(settings: PatternCacheSettings) -> Self {
        Self {
            cache: HashMap::new(),
            settings,
        }
    }

    /// Create a new pattern cache with default settings
    pub fn default() -> Self {
        Self::new(PatternCacheSettings::default())
    }

    /// Get cached patterns if available and not expired
    pub fn get(&self, key: &str) -> Option<&CachedPatternResult> {
        if !self.settings.enable_caching {
            return None;
        }

        if let Some(cached) = self.cache.get(key) {
            if !cached.is_expired() {
                return Some(cached);
            }
        }
        None
    }

    /// Cache patterns with the given key
    pub fn put(&mut self, key: String, patterns: Vec<Pattern>) {
        if !self.settings.enable_caching {
            return;
        }

        // Check cache size limit and remove oldest if necessary
        if self.cache.len() >= self.settings.max_cache_size {
            if let Some(oldest_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&oldest_key);
            }
        }

        let cached = CachedPatternResult {
            patterns,
            timestamp: chrono::Utc::now(),
            ttl: std::time::Duration::from_secs(self.settings.cache_ttl_seconds),
        };

        self.cache.insert(key, cached);
    }

    /// Clear all cached patterns
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Create a cache key for the given store and graph name
    pub fn create_key(&self, _store: &dyn Store, graph_name: Option<&str>) -> String {
        let mut hasher = DefaultHasher::new();
        graph_name.hash(&mut hasher);
        format!("patterns_{}", hasher.finish())
    }

    /// Create a cache key for shape analysis
    pub fn create_shape_key(&self, shape_count: usize, algorithm_config: &str) -> String {
        let mut hasher = DefaultHasher::new();
        shape_count.hash(&mut hasher);
        algorithm_config.hash(&mut hasher);
        format!("shapes_{}_{}", shape_count, hasher.finish())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_entries: self.cache.len(),
            max_size: self.settings.max_cache_size,
            enabled: self.settings.enable_caching,
            expired_entries: self.cache.values().filter(|c| c.is_expired()).count(),
        }
    }

    /// Remove expired entries from cache
    pub fn cleanup_expired(&mut self) {
        let expired_keys: Vec<String> = self
            .cache
            .iter()
            .filter(|(_, cached)| cached.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            self.cache.remove(&key);
        }
    }

    /// Check if caching is enabled
    pub fn is_enabled(&self) -> bool {
        self.settings.enable_caching
    }

    /// Update cache settings
    pub fn update_settings(&mut self, settings: PatternCacheSettings) {
        self.settings = settings;

        // Clear cache if caching is disabled
        if !self.settings.enable_caching {
            self.clear();
        }

        // Trim cache if new max size is smaller
        while self.cache.len() > self.settings.max_cache_size {
            if let Some(oldest_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&oldest_key);
            } else {
                break;
            }
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub max_size: usize,
    pub enabled: bool,
    pub expired_entries: usize,
}
