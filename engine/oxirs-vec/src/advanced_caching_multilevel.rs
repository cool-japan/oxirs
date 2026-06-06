//! Multi-level cache and invalidation utilities for the advanced caching system.
//!
//! Contains:
//! - `MultiLevelCache` — combines memory + persistent caches
//! - `MultiLevelCacheStats` — aggregated hit/miss statistics
//! - `CacheInvalidator` — tag-indexed and namespace-indexed invalidation
//! - `InvalidationStats` — invalidation tracking statistics

use crate::advanced_caching::{CacheConfig, CacheEntry, CacheKey, CacheStats};
use crate::advanced_caching_eviction::{MemoryCache, PersistentCache};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Type alias for complex tag index structure
type TagIndex = Arc<RwLock<HashMap<String, HashMap<String, Vec<CacheKey>>>>>;

/// Multi-level cache combining memory and persistent storage
pub struct MultiLevelCache {
    pub(super) memory_cache: Arc<RwLock<MemoryCache>>,
    pub(super) persistent_cache: Option<Arc<PersistentCache>>,
    #[allow(dead_code)]
    pub(super) config: CacheConfig,
    pub(super) stats: Arc<RwLock<MultiLevelCacheStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct MultiLevelCacheStats {
    pub memory_hits: u64,
    pub memory_misses: u64,
    pub persistent_hits: u64,
    pub persistent_misses: u64,
    pub total_requests: u64,
}

impl MultiLevelCache {
    pub fn new(config: CacheConfig) -> Result<Self> {
        let memory_cache = Arc::new(RwLock::new(MemoryCache::new(config.clone())));

        let persistent_cache = if config.enable_persistent {
            Some(Arc::new(PersistentCache::new(config.clone())?))
        } else {
            None
        };

        Ok(Self {
            memory_cache,
            persistent_cache,
            config,
            stats: Arc::new(RwLock::new(MultiLevelCacheStats::default())),
        })
    }

    /// Insert entry into cache
    pub fn insert(&self, key: CacheKey, data: crate::Vector) -> Result<()> {
        let entry = CacheEntry::new(data);

        // Insert into memory cache
        {
            let mut memory = self.memory_cache.write().expect("lock poisoned");
            memory.insert(key.clone(), entry.clone())?;
        }

        // Insert into persistent cache
        if let Some(ref persistent) = self.persistent_cache {
            persistent.store(&key, &entry)?;
        }

        Ok(())
    }

    /// Get entry from cache
    pub fn get(&self, key: &CacheKey) -> Option<crate::Vector> {
        self.update_stats_total();

        // Try memory cache first
        {
            let mut memory = self.memory_cache.write().expect("lock poisoned");
            if let Some(data) = memory.get(key) {
                self.update_stats_memory_hit();
                return Some(data.clone());
            }
        }

        self.update_stats_memory_miss();

        // Try persistent cache
        if let Some(ref persistent) = self.persistent_cache {
            if let Ok(Some(mut entry)) = persistent.load(key) {
                self.update_stats_persistent_hit();

                // Promote to memory cache
                let data = entry.data.clone();
                entry.touch();
                if let Ok(mut memory) = self.memory_cache.write() {
                    let _ = memory.insert(key.clone(), entry);
                }

                return Some(data);
            }
        }

        self.update_stats_persistent_miss();
        None
    }

    /// Remove entry from cache
    pub fn remove(&self, key: &CacheKey) -> Result<()> {
        // Remove from memory cache
        {
            let mut memory = self.memory_cache.write().expect("lock poisoned");
            memory.remove(key);
        }

        // Remove from persistent cache
        if let Some(ref persistent) = self.persistent_cache {
            persistent.remove(key)?;
        }

        Ok(())
    }

    /// Clear all caches
    pub fn clear(&self) -> Result<()> {
        // Clear memory cache
        {
            let mut memory = self.memory_cache.write().expect("lock poisoned");
            memory.clear();
        }

        // Clear persistent cache
        if let Some(ref persistent) = self.persistent_cache {
            persistent.clear()?;
        }

        // Reset stats
        {
            let mut stats = self.stats.write().expect("lock poisoned");
            *stats = MultiLevelCacheStats::default();
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> MultiLevelCacheStats {
        self.stats.read().expect("lock poisoned").clone()
    }

    /// Get memory cache statistics
    pub fn get_memory_stats(&self) -> CacheStats {
        let memory = self.memory_cache.read().expect("lock poisoned");
        memory.stats()
    }

    // Stats update methods
    fn update_stats_total(&self) {
        let mut stats = self.stats.write().expect("lock poisoned");
        stats.total_requests += 1;
    }

    fn update_stats_memory_hit(&self) {
        let mut stats = self.stats.write().expect("lock poisoned");
        stats.memory_hits += 1;
    }

    fn update_stats_memory_miss(&self) {
        let mut stats = self.stats.write().expect("lock poisoned");
        stats.memory_misses += 1;
    }

    fn update_stats_persistent_hit(&self) {
        let mut stats = self.stats.write().expect("lock poisoned");
        stats.persistent_hits += 1;
    }

    fn update_stats_persistent_miss(&self) {
        let mut stats = self.stats.write().expect("lock poisoned");
        stats.persistent_misses += 1;
    }
}

// ---------------------------------------------------------------------------
// CacheInvalidator
// ---------------------------------------------------------------------------

/// Cache invalidation utilities with indexing support
pub struct CacheInvalidator {
    cache: Arc<MultiLevelCache>,
    tag_index: TagIndex, // tag_key -> tag_value -> keys
    namespace_index: Arc<RwLock<HashMap<String, Vec<CacheKey>>>>, // namespace -> keys
}

impl CacheInvalidator {
    pub fn new(cache: Arc<MultiLevelCache>) -> Self {
        Self {
            cache,
            tag_index: Arc::new(RwLock::new(HashMap::new())),
            namespace_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a cache entry for invalidation tracking
    pub fn register_entry(&self, key: &CacheKey, tags: &HashMap<String, String>) {
        // Index by namespace
        {
            let mut ns_index = self.namespace_index.write().expect("lock poisoned");
            ns_index
                .entry(key.namespace.clone())
                .or_default()
                .push(key.clone());
        }

        // Index by tags
        {
            let mut tag_idx = self.tag_index.write().expect("lock poisoned");
            for (tag_key, tag_value) in tags {
                tag_idx
                    .entry(tag_key.clone())
                    .or_default()
                    .entry(tag_value.clone())
                    .or_default()
                    .push(key.clone());
            }
        }
    }

    /// Unregister a cache entry from invalidation tracking
    pub fn unregister_entry(&self, key: &CacheKey) {
        // Remove from namespace index
        {
            let mut ns_index = self.namespace_index.write().expect("lock poisoned");
            if let Some(keys) = ns_index.get_mut(&key.namespace) {
                keys.retain(|k| k != key);
                if keys.is_empty() {
                    ns_index.remove(&key.namespace);
                }
            }
        }

        // Remove from tag index
        {
            let mut tag_idx = self.tag_index.write().expect("lock poisoned");
            let mut tags_to_remove = Vec::new();

            for (tag_key, tag_values) in tag_idx.iter_mut() {
                let mut values_to_remove = Vec::new();

                for (tag_value, keys) in tag_values.iter_mut() {
                    keys.retain(|k| k != key);
                    if keys.is_empty() {
                        values_to_remove.push(tag_value.clone());
                    }
                }

                for value in values_to_remove {
                    tag_values.remove(&value);
                }

                if tag_values.is_empty() {
                    tags_to_remove.push(tag_key.clone());
                }
            }

            for tag in tags_to_remove {
                tag_idx.remove(&tag);
            }
        }
    }

    /// Invalidate entries by tag
    pub fn invalidate_by_tag(&self, tag_key: &str, tag_value: &str) -> Result<usize> {
        let keys_to_invalidate = {
            let tag_idx = self.tag_index.read().expect("lock poisoned");
            tag_idx
                .get(tag_key)
                .and_then(|values| values.get(tag_value))
                .cloned()
                .unwrap_or_default()
        };

        let mut invalidated_count = 0;
        for key in &keys_to_invalidate {
            if self.cache.remove(key).is_ok() {
                invalidated_count += 1;
            }
            self.unregister_entry(key);
        }

        Ok(invalidated_count)
    }

    /// Invalidate entries by namespace
    pub fn invalidate_namespace(&self, namespace: &str) -> Result<usize> {
        let keys_to_invalidate = {
            let ns_index = self.namespace_index.read().expect("lock poisoned");
            ns_index.get(namespace).cloned().unwrap_or_default()
        };

        let mut invalidated_count = 0;
        for key in &keys_to_invalidate {
            if self.cache.remove(key).is_ok() {
                invalidated_count += 1;
            }
            self.unregister_entry(key);
        }

        Ok(invalidated_count)
    }

    /// Invalidate all expired entries
    pub fn invalidate_expired(&self) -> Result<usize> {
        // Memory cache cleans expired entries automatically during operations.
        // For persistent cache, scan and remove expired files.
        if let Some(ref persistent) = self.cache.persistent_cache {
            return self.scan_and_remove_expired_files(persistent);
        }
        Ok(0)
    }

    /// Scan persistent cache directory and remove expired files
    fn scan_and_remove_expired_files(&self, persistent_cache: &PersistentCache) -> Result<usize> {
        let cache_dir = &persistent_cache.cache_dir;
        let mut removed_count = 0;

        if !cache_dir.exists() {
            return Ok(0);
        }

        // Walk through all cache files
        for entry in std::fs::read_dir(cache_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                // Recursively scan subdirectories
                for sub_entry in std::fs::read_dir(entry.path())? {
                    let sub_entry = sub_entry?;
                    if sub_entry.file_type()?.is_file() {
                        if let Some(file_name) = sub_entry.file_name().to_str() {
                            if file_name.ends_with(".cache") {
                                // Decode cache key from filename
                                if let Some(cache_key) =
                                    persistent_cache.decode_cache_key_from_filename(file_name)
                                {
                                    // Load the actual cache entry to check expiration
                                    if let Ok(Some(loaded)) = persistent_cache.load(&cache_key) {
                                        if loaded.is_expired() {
                                            let _ = std::fs::remove_file(sub_entry.path());
                                            removed_count += 1;
                                        }
                                    } else {
                                        // Corrupted entry — remove it
                                        let _ = std::fs::remove_file(sub_entry.path());
                                        removed_count += 1;
                                    }
                                } else {
                                    // Old format — fall back to file-age heuristic
                                    if let Ok(metadata) = std::fs::metadata(sub_entry.path()) {
                                        if let Ok(modified) = metadata.modified() {
                                            let age = modified
                                                .elapsed()
                                                .unwrap_or(Duration::from_secs(0));
                                            // Remove files older than 24 hours
                                            if age > Duration::from_secs(24 * 3600) {
                                                let _ = std::fs::remove_file(sub_entry.path());
                                                removed_count += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(removed_count)
    }

    /// Get invalidation statistics
    pub fn get_stats(&self) -> InvalidationStats {
        let tag_idx = self.tag_index.read().expect("lock poisoned");
        let ns_index = self.namespace_index.read().expect("lock poisoned");

        let total_tag_entries = tag_idx
            .values()
            .flat_map(|values| values.values())
            .map(|keys| keys.len())
            .sum();

        let total_namespace_entries = ns_index.values().map(|keys| keys.len()).sum();

        InvalidationStats {
            tracked_tags: tag_idx.len(),
            tracked_namespaces: ns_index.len(),
            total_tag_entries,
            total_namespace_entries,
        }
    }
}

/// Statistics for cache invalidation tracking
#[derive(Debug, Clone)]
pub struct InvalidationStats {
    pub tracked_tags: usize,
    pub tracked_namespaces: usize,
    pub total_tag_entries: usize,
    pub total_namespace_entries: usize,
}
