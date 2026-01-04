//! Shared Advanced Caching Framework for OxiRS Engine Modules
//!
//! This module provides a unified, high-performance caching system that can be shared
//! across all OxiRS engine modules (ARQ, SHACL, Vec, Star) for optimal performance
//! and memory efficiency.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, RwLock,
};
use std::time::{Duration, Instant};

/// Multi-level cache configuration
#[derive(Debug, Clone)]
pub struct AdvancedCacheConfig {
    /// L1 cache size (hot data, fastest access)
    pub l1_cache_size: usize,
    /// L2 cache size (warm data, fast access)
    pub l2_cache_size: usize,
    /// L3 cache size (cold data, compressed storage)
    pub l3_cache_size: usize,
    /// Default TTL for cache entries
    pub default_ttl: Duration,
    /// Enable adaptive TTL based on access patterns
    pub adaptive_ttl: bool,
    /// Enable compression for L3 cache
    pub enable_compression: bool,
    /// Memory pressure threshold (MB)
    pub memory_pressure_threshold: usize,
    /// Cache warming strategy
    pub warming_strategy: CacheWarmingStrategy,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable cross-module cache sharing
    pub enable_cross_module_sharing: bool,
}

impl Default for AdvancedCacheConfig {
    fn default() -> Self {
        Self {
            l1_cache_size: 10000,    // 10K entries
            l2_cache_size: 50000,    // 50K entries
            l3_cache_size: 200000,   // 200K entries
            default_ttl: Duration::from_secs(3600), // 1 hour
            adaptive_ttl: true,
            enable_compression: true,
            memory_pressure_threshold: 1000, // 1GB
            warming_strategy: CacheWarmingStrategy::Predictive,
            eviction_policy: EvictionPolicy::AdaptiveLRU,
            enable_cross_module_sharing: true,
        }
    }
}

/// Cache warming strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheWarmingStrategy {
    /// No proactive warming
    None,
    /// Warm based on access patterns
    PatternBased,
    /// Predictive warming using ML
    Predictive,
    /// Aggressive warming for performance
    Aggressive,
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Adaptive Replacement Cache
    ARC,
    /// Adaptive LRU with cost consideration
    AdaptiveLRU,
    /// Time-based expiration
    TimeToLive,
}

/// Multi-level cache with advanced optimization features
#[derive(Debug)]
pub struct AdvancedCache<K, V>
where
    K: Clone + Hash + Eq + Debug + Send + Sync,
    V: Clone + Debug + Send + Sync,
{
    /// Configuration
    config: AdvancedCacheConfig,
    /// L1 cache (hot data)
    l1_cache: Arc<RwLock<LeveledCache<K, V>>>,
    /// L2 cache (warm data)
    l2_cache: Arc<RwLock<LeveledCache<K, V>>>,
    /// L3 cache (cold data, compressed)
    l3_cache: Arc<RwLock<CompressedCache<K, V>>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStatistics>>,
    /// Access pattern tracker
    access_tracker: Arc<RwLock<AccessPatternTracker<K>>>,
    /// Memory usage monitor
    memory_monitor: Arc<RwLock<MemoryMonitor>>,
}

/// Single level cache implementation
#[derive(Debug)]
struct LeveledCache<K, V>
where
    K: Clone + Hash + Eq + Debug,
    V: Clone + Debug,
{
    /// Cache entries
    entries: HashMap<K, CacheEntry<V>>,
    /// Access order queue for LRU
    access_order: VecDeque<K>,
    /// Frequency tracker for LFU
    frequency_tracker: HashMap<K, usize>,
    /// Maximum capacity
    max_capacity: usize,
    /// Eviction policy
    eviction_policy: EvictionPolicy,
}

/// Compressed cache for L3 storage
#[derive(Debug)]
struct CompressedCache<K, V>
where
    K: Clone + Hash + Eq + Debug,
    V: Clone + Debug,
{
    /// Compressed entries
    entries: HashMap<K, CompressedEntry>,
    /// Access order
    access_order: VecDeque<K>,
    /// Maximum capacity
    max_capacity: usize,
    /// Compression ratio achieved
    compression_ratio: f64,
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    /// The cached value
    value: V,
    /// Time when cached
    cached_at: Instant,
    /// Last access time
    last_accessed: Instant,
    /// Access count
    access_count: usize,
    /// TTL for this entry
    ttl: Duration,
    /// Size estimate in bytes
    size_bytes: usize,
    /// Cost of recreating this entry
    recreation_cost: f64,
}

/// Compressed cache entry
#[derive(Debug, Clone)]
struct CompressedEntry {
    /// Compressed data
    compressed_data: Vec<u8>,
    /// Original size
    original_size: usize,
    /// Compression metadata
    metadata: CompressionMetadata,
    /// Time when cached
    cached_at: Instant,
    /// Last access time
    last_accessed: Instant,
    /// Access count
    access_count: usize,
}

/// Compression metadata
#[derive(Debug, Clone)]
struct CompressionMetadata {
    /// Compression algorithm used
    algorithm: CompressionAlgorithm,
    /// Compression ratio achieved
    ratio: f64,
    /// Time taken to compress
    compression_time: Duration,
}

/// Compression algorithms
#[derive(Debug, Clone, Copy)]
enum CompressionAlgorithm {
    None,
    LZ4,
    Zstd,
    Snappy,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    /// L1 cache hits
    pub l1_hits: usize,
    /// L2 cache hits
    pub l2_hits: usize,
    /// L3 cache hits
    pub l3_hits: usize,
    /// Total cache misses
    pub misses: usize,
    /// Total entries cached
    pub entries_cached: usize,
    /// Total evictions
    pub evictions: usize,
    /// Average access time (microseconds)
    pub avg_access_time_us: f64,
    /// Cache efficiency score (0-1)
    pub efficiency_score: f64,
    /// Memory usage by level
    pub memory_usage_by_level: HashMap<usize, usize>,
    /// Compression statistics
    pub compression_stats: CompressionStatistics,
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStatistics {
    /// Total bytes before compression
    pub total_uncompressed_bytes: usize,
    /// Total bytes after compression
    pub total_compressed_bytes: usize,
    /// Average compression ratio
    pub avg_compression_ratio: f64,
    /// Time spent compressing
    pub total_compression_time: Duration,
    /// Time spent decompressing
    pub total_decompression_time: Duration,
}

/// Access pattern tracker for predictive caching
#[derive(Debug)]
struct AccessPatternTracker<K>
where
    K: Clone + Hash + Eq + Debug,
{
    /// Recent access patterns
    access_history: VecDeque<AccessEvent<K>>,
    /// Pattern frequencies
    pattern_frequencies: HashMap<AccessPattern<K>, usize>,
    /// Prediction model
    prediction_model: AccessPredictionModel<K>,
    /// Maximum history size
    max_history_size: usize,
}

/// Access event
#[derive(Debug, Clone)]
struct AccessEvent<K> {
    /// Key accessed
    key: K,
    /// Timestamp
    timestamp: Instant,
    /// Cache level where found (or None if miss)
    cache_level: Option<usize>,
    /// Access type
    access_type: AccessType,
}

/// Access pattern
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct AccessPattern<K> {
    /// Sequence of keys
    key_sequence: Vec<K>,
    /// Time window
    time_window: Duration,
}

/// Access prediction model
#[derive(Debug)]
struct AccessPredictionModel<K>
where
    K: Clone + Hash + Eq + Debug,
{
    /// Predicted next accesses
    predictions: HashMap<K, PredictionMetadata>,
    /// Model accuracy
    accuracy_score: f64,
    /// Last model update
    last_update: Instant,
}

/// Prediction metadata
#[derive(Debug, Clone)]
struct PredictionMetadata {
    /// Predicted access probability (0-1)
    probability: f64,
    /// Predicted access time
    predicted_time: Option<Instant>,
    /// Confidence score (0-1)
    confidence: f64,
}

/// Access types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessType {
    Read,
    Write,
    Eviction,
    Warming,
}

/// Memory monitor
#[derive(Debug, Clone, Default)]
struct MemoryMonitor {
    /// Current memory usage by level
    current_usage: HashMap<usize, usize>,
    /// Peak memory usage
    peak_usage: usize,
    /// Memory pressure alerts
    pressure_alerts: usize,
    /// Last cleanup time
    last_cleanup: Option<Instant>,
}

impl<K, V> AdvancedCache<K, V>
where
    K: Clone + Hash + Eq + Debug + Send + Sync + 'static,
    V: Clone + Debug + Send + Sync + 'static,
{
    /// Create a new advanced cache
    pub fn new(config: AdvancedCacheConfig) -> Self {
        Self {
            l1_cache: Arc::new(RwLock::new(LeveledCache::new(
                config.l1_cache_size,
                config.eviction_policy,
            ))),
            l2_cache: Arc::new(RwLock::new(LeveledCache::new(
                config.l2_cache_size,
                config.eviction_policy,
            ))),
            l3_cache: Arc::new(RwLock::new(CompressedCache::new(config.l3_cache_size))),
            stats: Arc::new(RwLock::new(CacheStatistics::default())),
            access_tracker: Arc::new(RwLock::new(AccessPatternTracker::new())),
            memory_monitor: Arc::new(RwLock::new(MemoryMonitor::default())),
            config,
        }
    }

    /// Get value from cache with multi-level lookup
    pub fn get(&self, key: &K) -> Option<V> {
        let start_time = Instant::now();

        // Track access event
        self.track_access(key, AccessType::Read);

        // Try L1 cache first (hot data)
        if let Some(value) = self.get_from_l1(key) {
            self.record_cache_hit(1, start_time.elapsed());
            return Some(value);
        }

        // Try L2 cache (warm data)
        if let Some(value) = self.get_from_l2(key) {
            // Promote to L1 for future fast access
            self.promote_to_l1(key, &value);
            self.record_cache_hit(2, start_time.elapsed());
            return Some(value);
        }

        // Try L3 cache (cold data, compressed)
        if let Some(value) = self.get_from_l3(key) {
            // Promote to L2 and potentially L1
            self.promote_from_l3(key, &value);
            self.record_cache_hit(3, start_time.elapsed());
            return Some(value);
        }

        // Cache miss
        self.record_cache_miss(start_time.elapsed());
        None
    }

    /// Put value into cache with intelligent level placement
    pub fn put(&self, key: K, value: V) -> Result<()> {
        let start_time = Instant::now();

        // Track access event
        self.track_access(&key, AccessType::Write);

        // Determine optimal cache level based on access patterns
        let target_level = self.determine_optimal_level(&key, &value);

        match target_level {
            1 => self.put_in_l1(key, value)?,
            2 => self.put_in_l2(key, value)?,
            3 => self.put_in_l3(key, value)?,
            _ => self.put_in_l1(key, value)?, // Default to L1
        }

        // Update statistics
        self.record_cache_put(start_time.elapsed());

        // Check for memory pressure and cleanup if needed
        if self.check_memory_pressure() {
            self.perform_cleanup()?;
        }

        // Update prediction model
        self.update_prediction_model(&key);

        Ok(())
    }

    /// Remove value from all cache levels
    pub fn remove(&self, key: &K) -> Option<V> {
        self.track_access(key, AccessType::Eviction);

        // Try removing from each level, return first found
        if let Some(value) = self.remove_from_l1(key) {
            return Some(value);
        }
        if let Some(value) = self.remove_from_l2(key) {
            return Some(value);
        }
        self.remove_from_l3(key)
    }

    /// Clear all cache levels
    pub fn clear(&self) {
        {
            let mut l1 = self.l1_cache.write().expect("lock poisoned");
            l1.clear();
        }
        {
            let mut l2 = self.l2_cache.write().expect("lock poisoned");
            l2.clear();
        }
        {
            let mut l3 = self.l3_cache.write().expect("lock poisoned");
            l3.clear();
        }

        // Reset statistics
        {
            let mut stats = self.stats.write().expect("lock poisoned");
            *stats = CacheStatistics::default();
        }
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        let stats = self.stats.read().expect("lock poisoned");
        stats.clone()
    }

    /// Perform cache warming based on access patterns
    pub fn warm_cache(&self) -> Result<()> {
        match self.config.warming_strategy {
            CacheWarmingStrategy::None => Ok(()),
            CacheWarmingStrategy::PatternBased => self.warm_from_patterns(),
            CacheWarmingStrategy::Predictive => self.warm_predictively(),
            CacheWarmingStrategy::Aggressive => self.warm_aggressively(),
        }
    }

    /// Private implementation methods
    fn get_from_l1(&self, key: &K) -> Option<V> {
        let mut l1 = self.l1_cache.write().expect("lock poisoned");
        l1.get(key)
    }

    fn get_from_l2(&self, key: &K) -> Option<V> {
        let mut l2 = self.l2_cache.write().expect("lock poisoned");
        l2.get(key)
    }

    fn get_from_l3(&self, key: &K) -> Option<V> {
        let mut l3 = self.l3_cache.write().expect("lock poisoned");
        l3.get(key)
    }

    fn promote_to_l1(&self, key: &K, value: &V) {
        let mut l1 = self.l1_cache.write().expect("lock poisoned");
        l1.put(key.clone(), value.clone());
    }

    fn promote_from_l3(&self, key: &K, value: &V) {
        // Move to L2, and potentially L1 based on access frequency
        let mut l2 = self.l2_cache.write().expect("lock poisoned");
        l2.put(key.clone(), value.clone());
    }

    fn put_in_l1(&self, key: K, value: V) -> Result<()> {
        let mut l1 = self.l1_cache.write().expect("lock poisoned");
        l1.put(key, value);
        Ok(())
    }

    fn put_in_l2(&self, key: K, value: V) -> Result<()> {
        let mut l2 = self.l2_cache.write().expect("lock poisoned");
        l2.put(key, value);
        Ok(())
    }

    fn put_in_l3(&self, key: K, value: V) -> Result<()> {
        let mut l3 = self.l3_cache.write().expect("lock poisoned");
        l3.put(key, value)?;
        Ok(())
    }

    fn remove_from_l1(&self, key: &K) -> Option<V> {
        let mut l1 = self.l1_cache.write().expect("lock poisoned");
        l1.remove(key)
    }

    fn remove_from_l2(&self, key: &K) -> Option<V> {
        let mut l2 = self.l2_cache.write().expect("lock poisoned");
        l2.remove(key)
    }

    fn remove_from_l3(&self, key: &K) -> Option<V> {
        let mut l3 = self.l3_cache.write().expect("lock poisoned");
        l3.remove(key)
    }

    fn determine_optimal_level(&self, _key: &K, _value: &V) -> usize {
        // Placeholder implementation - would analyze access patterns and value characteristics
        1 // Default to L1 for now
    }

    fn track_access(&self, key: &K, access_type: AccessType) {
        let mut tracker = self.access_tracker.write().expect("lock poisoned");
        tracker.track_access(key.clone(), access_type);
    }

    fn record_cache_hit(&self, level: usize, access_time: Duration) {
        let mut stats = self.stats.write().expect("lock poisoned");
        match level {
            1 => stats.l1_hits += 1,
            2 => stats.l2_hits += 1,
            3 => stats.l3_hits += 1,
            _ => {}
        }
        self.update_avg_access_time(&mut stats, access_time);
    }

    fn record_cache_miss(&self, access_time: Duration) {
        let mut stats = self.stats.write().expect("lock poisoned");
        stats.misses += 1;
        self.update_avg_access_time(&mut stats, access_time);
    }

    fn record_cache_put(&self, _put_time: Duration) {
        let mut stats = self.stats.write().expect("lock poisoned");
        stats.entries_cached += 1;
    }

    fn update_avg_access_time(&self, stats: &mut CacheStatistics, access_time: Duration) {
        let total_operations = stats.l1_hits + stats.l2_hits + stats.l3_hits + stats.misses;
        let access_time_us = access_time.as_micros() as f64;

        if total_operations == 1 {
            stats.avg_access_time_us = access_time_us;
        } else {
            stats.avg_access_time_us =
                (stats.avg_access_time_us * (total_operations - 1) as f64 + access_time_us)
                / total_operations as f64;
        }
    }

    fn check_memory_pressure(&self) -> bool {
        let monitor = self.memory_monitor.read().expect("lock poisoned");
        let total_usage: usize = monitor.current_usage.values().sum();
        total_usage > self.config.memory_pressure_threshold * 1024 * 1024 // Convert MB to bytes
    }

    fn perform_cleanup(&self) -> Result<()> {
        // Implement cleanup logic based on eviction policy
        // This is a simplified implementation
        {
            let mut l3 = self.l3_cache.write().expect("lock poisoned");
            l3.evict_least_recently_used(l3.entries.len() / 10); // Evict 10%
        }

        let mut monitor = self.memory_monitor.write().expect("lock poisoned");
        monitor.last_cleanup = Some(Instant::now());
        monitor.pressure_alerts += 1;

        Ok(())
    }

    fn update_prediction_model(&self, _key: &K) {
        // Placeholder implementation - would update ML prediction model
    }

    fn warm_from_patterns(&self) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }

    fn warm_predictively(&self) -> Result<()> {
        // Placeholder implementation using ML predictions
        Ok(())
    }

    fn warm_aggressively(&self) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }
}

// Implementation for LeveledCache
impl<K, V> LeveledCache<K, V>
where
    K: Clone + Hash + Eq + Debug,
    V: Clone + Debug,
{
    fn new(max_capacity: usize, eviction_policy: EvictionPolicy) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            frequency_tracker: HashMap::new(),
            max_capacity,
            eviction_policy,
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;

            // Update access tracking
            self.update_access_tracking(key);

            Some(entry.value.clone())
        } else {
            None
        }
    }

    fn put(&mut self, key: K, value: V) {
        // Check if we need to evict
        if self.entries.len() >= self.max_capacity && !self.entries.contains_key(&key) {
            self.evict_one();
        }

        let now = Instant::now();
        let entry = CacheEntry {
            value,
            cached_at: now,
            last_accessed: now,
            access_count: 1,
            ttl: Duration::from_secs(3600), // Default TTL
            size_bytes: 1024, // Placeholder
            recreation_cost: 1.0, // Placeholder
        };

        self.entries.insert(key.clone(), entry);
        self.update_access_tracking(&key);
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.remove(key) {
            self.remove_from_tracking(key);
            Some(entry.value)
        } else {
            None
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.frequency_tracker.clear();
    }

    fn update_access_tracking(&mut self, key: &K) {
        // Update LRU tracking
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        self.access_order.push_back(key.clone());

        // Update LFU tracking
        *self.frequency_tracker.entry(key.clone()).or_insert(0) += 1;
    }

    fn remove_from_tracking(&mut self, key: &K) {
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        self.frequency_tracker.remove(key);
    }

    fn evict_one(&mut self) {
        let key_to_evict = match self.eviction_policy {
            EvictionPolicy::LRU => self.access_order.front().cloned(),
            EvictionPolicy::LFU => self.find_least_frequent(),
            EvictionPolicy::AdaptiveLRU => self.find_adaptive_lru_candidate(),
            _ => self.access_order.front().cloned(),
        };

        if let Some(key) = key_to_evict {
            self.remove(&key);
        }
    }

    fn find_least_frequent(&self) -> Option<K> {
        self.frequency_tracker
            .iter()
            .min_by_key(|(_, &freq)| freq)
            .map(|(key, _)| key.clone())
    }

    fn find_adaptive_lru_candidate(&self) -> Option<K> {
        // Simplified adaptive LRU - considers both recency and frequency
        self.access_order.front().cloned()
    }
}

// Implementation for CompressedCache
impl<K, V> CompressedCache<K, V>
where
    K: Clone + Hash + Eq + Debug,
    V: Clone + Debug,
{
    fn new(max_capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            max_capacity,
            compression_ratio: 0.0,
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;

            // Update access order
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push_back(key.clone());

            // Decompress and return value
            self.decompress_entry(entry)
        } else {
            None
        }
    }

    fn put(&mut self, key: K, value: V) -> Result<()> {
        // Check if we need to evict
        if self.entries.len() >= self.max_capacity && !self.entries.contains_key(&key) {
            self.evict_least_recently_used(1);
        }

        // Compress value
        let compressed_entry = self.compress_value(value)?;

        self.entries.insert(key.clone(), compressed_entry);
        self.access_order.push_back(key);

        Ok(())
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.remove(key) {
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            self.decompress_entry(&entry)
        } else {
            None
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }

    fn evict_least_recently_used(&mut self, count: usize) {
        for _ in 0..count.min(self.access_order.len()) {
            if let Some(key) = self.access_order.pop_front() {
                self.entries.remove(&key);
            }
        }
    }

    fn compress_value(&self, _value: V) -> Result<CompressedEntry> {
        // Placeholder compression implementation
        let now = Instant::now();
        Ok(CompressedEntry {
            compressed_data: vec![0u8; 100], // Placeholder
            original_size: 1024, // Placeholder
            metadata: CompressionMetadata {
                algorithm: CompressionAlgorithm::LZ4,
                ratio: 0.5,
                compression_time: Duration::from_micros(100),
            },
            cached_at: now,
            last_accessed: now,
            access_count: 0,
        })
    }

    fn decompress_entry(&self, _entry: &CompressedEntry) -> Option<V> {
        // Placeholder decompression implementation
        None
    }
}

// Implementation for AccessPatternTracker
impl<K> AccessPatternTracker<K>
where
    K: Clone + Hash + Eq + Debug,
{
    fn new() -> Self {
        Self {
            access_history: VecDeque::with_capacity(10000),
            pattern_frequencies: HashMap::new(),
            prediction_model: AccessPredictionModel::new(),
            max_history_size: 10000,
        }
    }

    fn track_access(&mut self, key: K, access_type: AccessType) {
        let event = AccessEvent {
            key,
            timestamp: Instant::now(),
            cache_level: None, // Would be populated based on where the access occurred
            access_type,
        };

        self.access_history.push_back(event);

        // Maintain history size
        if self.access_history.len() > self.max_history_size {
            self.access_history.pop_front();
        }

        // Update pattern analysis
        self.analyze_patterns();
    }

    fn analyze_patterns(&mut self) {
        // Placeholder pattern analysis implementation
        // Would analyze access sequences and update pattern frequencies
    }
}

impl<K> AccessPredictionModel<K>
where
    K: Clone + Hash + Eq + Debug,
{
    fn new() -> Self {
        Self {
            predictions: HashMap::new(),
            accuracy_score: 0.0,
            last_update: Instant::now(),
        }
    }
}

/// Helper traits for cross-module cache integration
pub trait CacheKey: Clone + Hash + Eq + Debug + Send + Sync {}
pub trait CacheValue: Clone + Debug + Send + Sync {}

// Implement for common types
impl<T> CacheKey for T where T: Clone + Hash + Eq + Debug + Send + Sync {}
impl<T> CacheValue for T where T: Clone + Debug + Send + Sync {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let config = AdvancedCacheConfig::default();
        let cache: AdvancedCache<String, String> = AdvancedCache::new(config);

        // Test basic functionality
        assert!(cache.get(&"test".to_string()).is_none());
    }

    #[test]
    fn test_cache_put_get() {
        let config = AdvancedCacheConfig::default();
        let cache: AdvancedCache<String, String> = AdvancedCache::new(config);

        let key = "test_key".to_string();
        let value = "test_value".to_string();

        cache.put(key.clone(), value.clone()).unwrap();
        assert_eq!(cache.get(&key), Some(value));
    }

    #[test]
    fn test_cache_statistics() {
        let config = AdvancedCacheConfig::default();
        let cache: AdvancedCache<String, String> = AdvancedCache::new(config);

        let stats = cache.statistics();
        assert_eq!(stats.l1_hits, 0);
        assert_eq!(stats.misses, 0);
    }
}