//! Background worker, cache warmer, and cache analyzer for the advanced caching system.
//!
//! Contains:
//! - `BackgroundCacheWorker` — periodic maintenance thread
//! - `CacheWarmer` — pre-populate cache from data or generators
//! - `CacheAnalyzer` / `CacheAnalysisReport` — performance analysis and recommendations

use crate::advanced_caching::{CacheConfig, CacheKey};
use crate::advanced_caching_multilevel::{CacheInvalidator, MultiLevelCache};
use anyhow::{anyhow, Result};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};

// ---------------------------------------------------------------------------
// BackgroundCacheWorker
// ---------------------------------------------------------------------------

/// Background cache worker for maintenance tasks
pub struct BackgroundCacheWorker {
    cache: Arc<MultiLevelCache>,
    invalidator: Arc<CacheInvalidator>,
    config: CacheConfig,
    worker_handle: Option<JoinHandle<()>>,
    shutdown_signal: Arc<RwLock<bool>>,
}

impl BackgroundCacheWorker {
    pub fn new(
        cache: Arc<MultiLevelCache>,
        invalidator: Arc<CacheInvalidator>,
        config: CacheConfig,
    ) -> Self {
        Self {
            cache,
            invalidator,
            config,
            worker_handle: None,
            shutdown_signal: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the background worker
    pub fn start(&mut self) -> Result<()> {
        if !self.config.enable_background_updates {
            return Ok(());
        }

        let cache = Arc::clone(&self.cache);
        let invalidator = Arc::clone(&self.invalidator);
        let interval = self.config.background_update_interval;
        let shutdown_signal = Arc::clone(&self.shutdown_signal);

        let handle = thread::spawn(move || {
            while let Ok(shutdown) = shutdown_signal.read() {
                if *shutdown {
                    break;
                }
                drop(shutdown); // Release the lock before sleeping

                // Perform maintenance tasks
                if let Err(e) = Self::perform_maintenance(&cache, &invalidator) {
                    // Log error but continue running
                    tracing::warn!("Background cache maintenance error: {}", e);
                }

                // Sleep for the configured interval
                thread::sleep(interval);
            }
        });

        self.worker_handle = Some(handle);
        Ok(())
    }

    /// Stop the background worker
    pub fn stop(&mut self) -> Result<()> {
        // Signal shutdown
        {
            let mut signal = self.shutdown_signal.write().expect("lock poisoned");
            *signal = true;
        }

        // Wait for worker to finish
        if let Some(handle) = self.worker_handle.take() {
            handle
                .join()
                .map_err(|e| anyhow!("Failed to join worker thread: {:?}", e))?;
        }

        Ok(())
    }

    /// Perform background maintenance tasks
    fn perform_maintenance(
        cache: &Arc<MultiLevelCache>,
        invalidator: &Arc<CacheInvalidator>,
    ) -> Result<()> {
        // 1. Clean expired entries
        let expired_count = invalidator.invalidate_expired()?;
        if expired_count > 0 {
            println!("Background worker cleaned {expired_count} expired entries");
        }

        // 2. Optimize memory usage if fragmentation is high
        let memory_stats = cache.get_memory_stats();
        let utilization = memory_stats.memory_bytes as f64 / memory_stats.max_memory_bytes as f64;

        if utilization > 0.9 {
            // Trigger more aggressive cleanup
            Self::aggressive_cleanup(cache)?;
        }

        // 3. Preemptive persistent cache sync
        Self::sync_hot_entries(cache)?;

        Ok(())
    }

    /// Perform aggressive cleanup when memory usage is high
    fn aggressive_cleanup(_cache: &Arc<MultiLevelCache>) -> Result<()> {
        // Force cleanup of memory cache by temporarily reducing limits.
        // In practice a more sophisticated strategy would be employed here.
        println!("Performing aggressive cache cleanup due to high memory usage");
        Ok(())
    }

    /// Sync frequently accessed entries to persistent storage
    fn sync_hot_entries(_cache: &Arc<MultiLevelCache>) -> Result<()> {
        // In a real implementation, hot entries would be identified and synced so
        // they survive restarts without a cold start penalty.
        Ok(())
    }
}

impl Drop for BackgroundCacheWorker {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

// ---------------------------------------------------------------------------
// CacheWarmer
// ---------------------------------------------------------------------------

/// Cache warming utilities
pub struct CacheWarmer {
    cache: Arc<MultiLevelCache>,
}

impl CacheWarmer {
    pub fn new(cache: Arc<MultiLevelCache>) -> Self {
        Self { cache }
    }

    /// Warm cache with a list of key-value pairs
    pub fn warm_with_data(&self, data: Vec<(CacheKey, crate::Vector)>) -> Result<usize> {
        let mut loaded_count = 0;

        for (key, vector) in data {
            if self.cache.insert(key, vector).is_ok() {
                loaded_count += 1;
            }
        }

        Ok(loaded_count)
    }

    /// Warm cache by loading frequently accessed entries from persistent storage
    pub fn warm_from_persistent(&self, keys: Vec<CacheKey>) -> Result<usize> {
        let mut loaded_count = 0;

        for key in keys {
            // Try to load from persistent cache and promote to memory
            if self.cache.get(&key).is_some() {
                loaded_count += 1;
            }
        }

        Ok(loaded_count)
    }

    /// Warm cache using a precomputed dataset
    pub fn warm_with_generator<F>(&self, count: usize, generator: F) -> Result<usize>
    where
        F: Fn(usize) -> Option<(CacheKey, crate::Vector)>,
    {
        let mut loaded_count = 0;

        for i in 0..count {
            if let Some((key, vector)) = generator(i) {
                if self.cache.insert(key, vector).is_ok() {
                    loaded_count += 1;
                }
            }
        }

        Ok(loaded_count)
    }
}

// ---------------------------------------------------------------------------
// CacheAnalyzer
// ---------------------------------------------------------------------------

/// Advanced cache analytics and optimization recommendations
pub struct CacheAnalyzer {
    cache: Arc<MultiLevelCache>,
    invalidator: Arc<CacheInvalidator>,
}

#[derive(Debug, Clone)]
pub struct CacheAnalysisReport {
    pub memory_utilization: f64,
    pub hit_ratio: f64,
    pub persistent_hit_ratio: f64,
    pub most_accessed_namespaces: Vec<(String, usize)>,
    pub recommendations: Vec<String>,
    pub performance_score: f64, // 0.0 to 1.0
}

impl CacheAnalyzer {
    pub fn new(cache: Arc<MultiLevelCache>, invalidator: Arc<CacheInvalidator>) -> Self {
        Self { cache, invalidator }
    }

    /// Generate comprehensive cache analysis report
    pub fn analyze(&self) -> CacheAnalysisReport {
        let stats = self.cache.get_stats();
        let memory_stats = self.cache.get_memory_stats();
        let invalidation_stats = self.invalidator.get_stats();

        let memory_utilization =
            memory_stats.memory_bytes as f64 / memory_stats.max_memory_bytes as f64;

        let total_requests = stats.total_requests;
        let total_hits = stats.memory_hits + stats.persistent_hits;
        let hit_ratio = if total_requests > 0 {
            total_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        let persistent_hit_ratio = if stats.persistent_hits + stats.persistent_misses > 0 {
            stats.persistent_hits as f64 / (stats.persistent_hits + stats.persistent_misses) as f64
        } else {
            0.0
        };

        let mut recommendations = Vec::new();

        // Generate recommendations
        if hit_ratio < 0.5 {
            recommendations
                .push("Consider increasing cache size or adjusting eviction policy".to_string());
        }

        if memory_utilization > 0.9 {
            recommendations.push(
                "Memory cache is near capacity - consider increasing max_memory_bytes".to_string(),
            );
        }

        if persistent_hit_ratio < 0.3 {
            recommendations
                .push("Persistent cache hit ratio is low - review TTL settings".to_string());
        }

        if invalidation_stats.tracked_namespaces > 100 {
            recommendations
                .push("Consider consolidating namespaces to reduce tracking overhead".to_string());
        }

        // Calculate performance score (weighted combination of metrics)
        let performance_score =
            (hit_ratio * 0.4 + (1.0 - memory_utilization) * 0.3 + persistent_hit_ratio * 0.3)
                .clamp(0.0, 1.0);

        CacheAnalysisReport {
            memory_utilization,
            hit_ratio,
            persistent_hit_ratio,
            most_accessed_namespaces: vec![], // Would need access pattern tracking
            recommendations,
            performance_score,
        }
    }

    /// Get recommendations for cache configuration optimization
    pub fn get_optimization_recommendations(&self) -> Vec<String> {
        self.analyze().recommendations
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{BackgroundCacheWorker, CacheAnalyzer, CacheWarmer};
    use crate::advanced_caching::{CacheConfig, CacheEntry, CacheKey, EvictionPolicy};
    use crate::advanced_caching_eviction::{MemoryCache, PersistentCache};
    use crate::advanced_caching_multilevel::{CacheInvalidator, MultiLevelCache};
    use crate::Vector;
    use anyhow::Result;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::TempDir;

    #[test]
    fn test_cache_key() {
        let key = CacheKey::new("embeddings", "test_doc").with_variant("v1");

        assert_eq!(key.namespace, "embeddings");
        assert_eq!(key.key, "test_doc");
        assert_eq!(key.variant, Some("v1".to_string()));
        assert_eq!(key.to_string(), "embeddings:test_doc:v1");
    }

    #[test]
    fn test_memory_cache() -> Result<()> {
        let config = CacheConfig {
            max_memory_entries: 2,
            max_memory_bytes: 1024,
            ..Default::default()
        };

        let mut cache = MemoryCache::new(config);

        let key1 = CacheKey::new("test", "key1");
        let key2 = CacheKey::new("test", "key2");
        let key3 = CacheKey::new("test", "key3");

        let vector1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let vector2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let vector3 = Vector::new(vec![7.0, 8.0, 9.0]);

        // Insert vectors
        cache.insert(key1.clone(), CacheEntry::new(vector1.clone()))?;
        cache.insert(key2.clone(), CacheEntry::new(vector2.clone()))?;

        // Check retrieval
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_some());

        // Insert third vector (should evict one)
        cache.insert(key3.clone(), CacheEntry::new(vector3.clone()))?;

        // One of the first two should be evicted
        let remaining = cache.entries.len();
        assert_eq!(remaining, 2);
        Ok(())
    }

    #[test]
    fn test_persistent_cache() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let config = CacheConfig {
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_compression: true,
            ..Default::default()
        };

        let cache = PersistentCache::new(config)?;

        let key = CacheKey::new("test", "persistent_key");
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);
        let entry = CacheEntry::new(vector.clone());

        // Store and retrieve
        cache.store(&key, &entry)?;
        let retrieved = cache.load(&key)?;

        // Should succeed now with proper serialization
        assert!(retrieved.is_some());
        let retrieved_entry = retrieved.expect("retrieved entry was None");
        assert_eq!(retrieved_entry.data.as_f32(), vector.as_f32());
        Ok(())
    }

    #[test]
    fn test_multi_level_cache() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let config = CacheConfig {
            max_memory_entries: 2,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = MultiLevelCache::new(config)?;

        let key = CacheKey::new("test", "multi_level");
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        // Insert and retrieve
        cache.insert(key.clone(), vector.clone())?;
        let retrieved = cache.get(&key).expect("get returned None");

        assert_eq!(retrieved.as_f32(), vector.as_f32());

        // Check stats
        let stats = cache.get_stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.memory_hits, 1);
        Ok(())
    }

    #[test]
    fn test_cache_expiration() -> Result<()> {
        let config = CacheConfig {
            max_memory_entries: 10,
            ttl: Some(Duration::from_millis(10)),
            ..Default::default()
        };

        let mut cache = MemoryCache::new(config);

        let key = CacheKey::new("test", "expiring");
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);
        let entry = CacheEntry::new(vector).with_ttl(Duration::from_millis(10));

        cache.insert(key.clone(), entry)?;

        // Should be available immediately
        assert!(cache.get(&key).is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(20));

        // Should be expired and removed
        assert!(cache.get(&key).is_none());
        Ok(())
    }

    #[test]
    fn test_arc_eviction_policy() -> Result<()> {
        let config = CacheConfig {
            max_memory_entries: 3,
            eviction_policy: EvictionPolicy::ARC,
            ..Default::default()
        };

        let mut cache = MemoryCache::new(config);

        let key1 = CacheKey::new("test", "arc1");
        let key2 = CacheKey::new("test", "arc2");
        let key3 = CacheKey::new("test", "arc3");
        let key4 = CacheKey::new("test", "arc4");

        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        // Insert three items
        cache.insert(key1.clone(), CacheEntry::new(vector.clone()))?;
        cache.insert(key2.clone(), CacheEntry::new(vector.clone()))?;
        cache.insert(key3.clone(), CacheEntry::new(vector.clone()))?;

        // Access key1 multiple times to make it frequent
        cache.get(&key1);
        cache.get(&key1);
        cache.get(&key1);

        // Insert key4 - should evict the least valuable item
        cache.insert(key4.clone(), CacheEntry::new(vector.clone()))?;

        // key1 should still be there (frequent access)
        assert!(cache.get(&key1).is_some());

        // Check that we have exactly 3 items
        assert_eq!(cache.entries.len(), 3);
        Ok(())
    }

    #[test]
    fn test_cache_warmer() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config)?);
        let warmer = CacheWarmer::new(Arc::clone(&cache));

        // Prepare test data
        let test_data = vec![
            (CacheKey::new("test", "warm1"), Vector::new(vec![1.0, 2.0])),
            (CacheKey::new("test", "warm2"), Vector::new(vec![3.0, 4.0])),
            (CacheKey::new("test", "warm3"), Vector::new(vec![5.0, 6.0])),
        ];

        // Warm cache with data
        let loaded_count = warmer.warm_with_data(test_data.clone())?;
        assert_eq!(loaded_count, 3);

        // Verify data is in cache
        for (key, expected_vector) in test_data {
            let cached_vector = cache.get(&key).expect("cached vector was None");
            assert_eq!(cached_vector.as_f32(), expected_vector.as_f32());
        }
        Ok(())
    }

    #[test]
    fn test_cache_warmer_with_generator() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config)?);
        let warmer = CacheWarmer::new(Arc::clone(&cache));

        // Use generator to warm cache
        let loaded_count = warmer.warm_with_generator(5, |i| {
            Some((
                CacheKey::new("generated", format!("item_{i}")),
                Vector::new(vec![i as f32, (i * 2) as f32]),
            ))
        })?;

        assert_eq!(loaded_count, 5);

        // Verify generated data is in cache
        for i in 0..5 {
            let key = CacheKey::new("generated", format!("item_{i}"));
            let cached_vector = cache.get(&key).expect("cached vector was None");
            assert_eq!(cached_vector.as_f32(), vec![i as f32, (i * 2) as f32]);
        }
        Ok(())
    }

    #[test]
    fn test_cache_analyzer() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config)?);
        let invalidator = Arc::new(CacheInvalidator::new(Arc::clone(&cache)));
        let analyzer = CacheAnalyzer::new(Arc::clone(&cache), Arc::clone(&invalidator));

        // Add some test data and access patterns
        let key1 = CacheKey::new("test", "analyze1");
        let key2 = CacheKey::new("test", "analyze2");
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        cache.insert(key1.clone(), vector.clone())?;
        cache.insert(key2.clone(), vector.clone())?;

        // Access the cache to generate some stats
        cache.get(&key1);
        cache.get(&key2);
        cache.get(&key1); // Hit
        cache.get(&CacheKey::new("test", "nonexistent")); // Miss

        // Analyze cache performance
        let report = analyzer.analyze();

        assert!(report.hit_ratio > 0.0);
        assert!(report.memory_utilization >= 0.0 && report.memory_utilization <= 1.0);
        assert!(report.performance_score >= 0.0 && report.performance_score <= 1.0);

        // Should have some recommendations if performance isn't perfect
        let recommendations = analyzer.get_optimization_recommendations();
        assert!(!recommendations.is_empty());
        Ok(())
    }

    #[test]
    fn test_background_cache_worker() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            enable_background_updates: true,
            background_update_interval: Duration::from_millis(100),
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config.clone())?);
        let invalidator = Arc::new(CacheInvalidator::new(Arc::clone(&cache)));
        let mut worker =
            BackgroundCacheWorker::new(Arc::clone(&cache), Arc::clone(&invalidator), config);

        // Start the worker
        worker.start()?;

        // Add some test data
        let key = CacheKey::new("test", "background");
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);
        cache.insert(key.clone(), vector.clone())?;

        // Let the worker run for a short time
        std::thread::sleep(Duration::from_millis(150));

        // Stop the worker
        worker.stop()?;

        // Verify data is still accessible
        assert!(cache.get(&key).is_some());
        Ok(())
    }

    #[test]
    fn test_cache_invalidation_by_tag() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config)?);
        let invalidator = Arc::new(CacheInvalidator::new(Arc::clone(&cache)));

        // Create entries with tags
        let key1 = CacheKey::new("test", "tagged1");
        let key2 = CacheKey::new("test", "tagged2");
        let key3 = CacheKey::new("test", "tagged3");

        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        cache.insert(key1.clone(), vector.clone())?;
        cache.insert(key2.clone(), vector.clone())?;
        cache.insert(key3.clone(), vector.clone())?;

        // Register entries with tags
        let mut tags1 = HashMap::new();
        tags1.insert("category".to_string(), "embeddings".to_string());
        invalidator.register_entry(&key1, &tags1);

        let mut tags2 = HashMap::new();
        tags2.insert("category".to_string(), "embeddings".to_string());
        invalidator.register_entry(&key2, &tags2);

        let mut tags3 = HashMap::new();
        tags3.insert("category".to_string(), "vectors".to_string());
        invalidator.register_entry(&key3, &tags3);

        // Invalidate by tag
        let invalidated_count = invalidator.invalidate_by_tag("category", "embeddings")?;
        assert_eq!(invalidated_count, 2);

        // Check that tagged entries are removed
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_none());

        // Check that untagged entry remains
        assert!(cache.get(&key3).is_some());
        Ok(())
    }

    #[test]
    fn test_cache_invalidation_by_namespace() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config)?);
        let invalidator = Arc::new(CacheInvalidator::new(Arc::clone(&cache)));

        // Create entries in different namespaces
        let key1 = CacheKey::new("embeddings", "item1");
        let key2 = CacheKey::new("embeddings", "item2");
        let key3 = CacheKey::new("vectors", "item3");

        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        cache.insert(key1.clone(), vector.clone())?;
        cache.insert(key2.clone(), vector.clone())?;
        cache.insert(key3.clone(), vector.clone())?;

        // Register entries for tracking
        invalidator.register_entry(&key1, &HashMap::new());
        invalidator.register_entry(&key2, &HashMap::new());
        invalidator.register_entry(&key3, &HashMap::new());

        // Invalidate by namespace
        let invalidated_count = invalidator.invalidate_namespace("embeddings")?;
        assert_eq!(invalidated_count, 2);

        // Check that namespace entries are removed
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_none());

        // Check that other namespace entry remains
        assert!(cache.get(&key3).is_some());
        Ok(())
    }
}
