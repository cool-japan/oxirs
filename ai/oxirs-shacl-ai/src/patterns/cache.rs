//! Advanced caching functionality for pattern analysis with intelligent cache management

use super::config::PatternCacheSettings;
use super::types::{CachedPatternResult, Pattern};
use crate::Result;
use oxirs_core::Store;
use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

/// Advanced pattern cache manager with intelligent caching strategies
#[derive(Debug)]
pub struct PatternCache {
    cache: HashMap<String, CachedPatternResult>,
    settings: PatternCacheSettings,
    /// LRU access order tracking
    lru_order: VecDeque<String>,
    /// Cache performance analytics
    analytics: CacheAnalytics,
    /// Pattern frequency tracking
    access_frequency: HashMap<String, AccessStats>,
    /// Cache warming predictions
    warming_predictor: CacheWarmingPredictor,
    /// Compression engine for memory optimization
    compression_engine: CompressionEngine,
    /// Adaptive eviction manager
    eviction_manager: AdaptiveEvictionManager,
}

/// Cache performance analytics
#[derive(Debug, Clone)]
pub struct CacheAnalytics {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub memory_usage_bytes: usize,
    pub last_reset: Instant,
    pub avg_access_time_ns: f64,
    pub hit_rate_history: VecDeque<f64>,
}

/// Pattern access statistics
#[derive(Debug, Clone)]
pub struct AccessStats {
    pub access_count: u64,
    pub last_access: Instant,
    pub first_access: Instant,
    pub total_access_time_ns: u64,
    pub avg_computation_time_ns: f64,
}

/// Cache warming prediction system
#[derive(Debug)]
pub struct CacheWarmingPredictor {
    prediction_model: PredictionModel,
    warming_candidates: Vec<WarmingCandidate>,
    last_analysis: Option<Instant>,
}

/// Simple prediction model for cache warming
#[derive(Debug)]
pub struct PredictionModel {
    historical_patterns: HashMap<String, f64>,
    time_based_weights: VecDeque<f64>,
}

/// Cache warming candidate
#[derive(Debug, Clone)]
pub struct WarmingCandidate {
    pub key: String,
    pub priority: f64,
    pub predicted_access_time: Instant,
    pub computation_cost: f64,
}

/// Compression engine for memory optimization
#[derive(Debug)]
pub struct CompressionEngine {
    enabled: bool,
    compression_ratio: f64,
    compression_stats: CompressionStats,
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub total_compressed: u64,
    pub total_uncompressed_size: usize,
    pub total_compressed_size: usize,
    pub avg_compression_ratio: f64,
    pub compression_time_ns: u64,
}

/// Adaptive eviction manager with multiple strategies
#[derive(Debug)]
pub struct AdaptiveEvictionManager {
    current_strategy: EvictionStrategy,
    strategy_performance: HashMap<EvictionStrategy, EvictionPerformance>,
    adaptation_threshold: f64,
    last_adaptation: Instant,
}

/// Eviction strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictionStrategy {
    LRU,            // Least Recently Used
    LFU,            // Least Frequently Used
    TLRU,           // Time-aware LRU
    AdaptiveTTL,    // Adaptive Time-To-Live
    MemoryPressure, // Based on memory pressure
}

/// Eviction performance metrics
#[derive(Debug, Clone)]
pub struct EvictionPerformance {
    pub hit_rate_after_eviction: f64,
    pub eviction_accuracy: f64,
    pub memory_efficiency: f64,
    pub sample_count: u64,
}

impl PatternCache {
    /// Create a new pattern cache with settings
    pub fn new(settings: PatternCacheSettings) -> Self {
        Self {
            cache: HashMap::new(),
            settings,
            lru_order: VecDeque::new(),
            analytics: CacheAnalytics::new(),
            access_frequency: HashMap::new(),
            warming_predictor: CacheWarmingPredictor::new(),
            compression_engine: CompressionEngine::new(),
            eviction_manager: AdaptiveEvictionManager::new(),
        }
    }

    /// Get cached patterns if available and not expired with analytics tracking
    pub fn get(&mut self, key: &str) -> Option<CachedPatternResult> {
        let start_time = Instant::now();

        // Update analytics
        self.analytics.total_requests += 1;

        if !self.settings.enable_caching {
            return None;
        }

        // Check if key exists and is not expired
        let is_expired = if let Some(cached) = self.cache.get(key) {
            cached.is_expired()
        } else {
            false
        };

        if is_expired {
            // Entry expired - remove it
            self.remove_expired_entry(key);
            self.analytics.cache_misses += 1;
            return None;
        }

        if let Some(cached) = self.cache.remove(key) {
            // Cache hit - update analytics and LRU order
            self.analytics.cache_hits += 1;
            self.update_lru_order(key);
            self.update_access_frequency(key, start_time);

            let access_time = start_time.elapsed().as_nanos() as f64;
            self.analytics.update_avg_access_time(access_time);

            // Re-insert the cached item
            self.cache.insert(key.to_string(), cached.clone());

            return Some(cached);
        }

        // Cache miss
        self.analytics.cache_misses += 1;
        None
    }

    /// Get cached patterns without mutable access (for read-only analytics)
    pub fn get_readonly(&self, key: &str) -> Option<&CachedPatternResult> {
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

    /// Cache patterns with the given key using intelligent adaptive eviction
    pub fn put(&mut self, key: String, patterns: Vec<Pattern>) {
        if !self.settings.enable_caching {
            return;
        }

        // Check cache size limit and use adaptive eviction if necessary
        if self.cache.len() >= self.settings.max_cache_size {
            let required_evictions = (self.cache.len() - self.settings.max_cache_size) + 1;
            self.advanced_eviction(required_evictions);
        }

        let cached = CachedPatternResult {
            patterns,
            timestamp: chrono::Utc::now(),
            ttl: std::time::Duration::from_secs(self.settings.cache_ttl_seconds),
        };

        // Update cache and tracking structures
        self.cache.insert(key.clone(), cached);
        self.add_to_lru_order(key.clone());
        self.initialize_access_frequency(key);

        // Update memory usage estimate
        self.update_memory_usage();
    }

    /// Clear all cached patterns
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
        self.access_frequency.clear();
        self.analytics.reset();
        self.warming_predictor.clear();
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

        // Trim cache if new max size is smaller with LRU eviction
        while self.cache.len() > self.settings.max_cache_size {
            if let Some(lru_key) = self.lru_order.pop_front() {
                self.cache.remove(&lru_key);
                self.access_frequency.remove(&lru_key);
                self.analytics.evictions += 1;
            } else {
                break;
            }
        }
    }

    // Private helper methods for advanced caching functionality

    /// Update LRU order by moving accessed key to the end
    fn update_lru_order(&mut self, key: &str) {
        // Remove key from current position
        self.lru_order.retain(|k| k != key);
        // Add to end (most recently used)
        self.lru_order.push_back(key.to_string());
    }

    /// Add new key to LRU order
    fn add_to_lru_order(&mut self, key: String) {
        self.lru_order.push_back(key);
    }

    /// Update access frequency statistics
    fn update_access_frequency(&mut self, key: &str, access_start: Instant) {
        let access_time = access_start.elapsed().as_nanos() as u64;

        if let Some(stats) = self.access_frequency.get_mut(key) {
            stats.access_count += 1;
            stats.last_access = Instant::now();
            stats.total_access_time_ns += access_time;
        }
    }

    /// Initialize access frequency for new key
    fn initialize_access_frequency(&mut self, key: String) {
        let now = Instant::now();
        let stats = AccessStats {
            access_count: 1,
            last_access: now,
            first_access: now,
            total_access_time_ns: 0,
            avg_computation_time_ns: 0.0,
        };
        self.access_frequency.insert(key, stats);
    }

    /// Remove expired entry and clean up tracking
    fn remove_expired_entry(&mut self, key: &str) {
        self.cache.remove(key);
        self.lru_order.retain(|k| k != key);
        self.access_frequency.remove(key);
    }

    /// Update memory usage estimate
    fn update_memory_usage(&mut self) {
        // Rough estimate: each cached pattern result ~1KB + overhead
        let estimated_size = self.cache.len() * 1024;
        self.analytics.memory_usage_bytes = estimated_size;
    }

    /// Get advanced cache statistics
    pub fn get_advanced_stats(&self) -> AdvancedCacheStats {
        let hit_rate = if self.analytics.total_requests > 0 {
            self.analytics.cache_hits as f64 / self.analytics.total_requests as f64
        } else {
            0.0
        };

        AdvancedCacheStats {
            basic_stats: self.stats(),
            hit_rate,
            avg_access_time_ns: self.analytics.avg_access_time_ns,
            total_evictions: self.analytics.evictions,
            memory_usage_bytes: self.analytics.memory_usage_bytes,
            most_accessed_keys: self.get_most_accessed_keys(5),
            cache_efficiency_score: self.calculate_efficiency_score(),
        }
    }

    /// Get most frequently accessed keys
    fn get_most_accessed_keys(&self, limit: usize) -> Vec<(String, u64)> {
        let mut access_pairs: Vec<_> = self
            .access_frequency
            .iter()
            .map(|(key, stats)| (key.clone(), stats.access_count))
            .collect();

        access_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        access_pairs.truncate(limit);
        access_pairs
    }

    /// Calculate cache efficiency score (0.0 to 1.0)
    fn calculate_efficiency_score(&self) -> f64 {
        if self.analytics.total_requests == 0 {
            return 1.0;
        }

        let hit_rate = self.analytics.cache_hits as f64 / self.analytics.total_requests as f64;
        let memory_efficiency = if self.settings.max_cache_size > 0 {
            1.0 - (self.analytics.evictions as f64 / self.settings.max_cache_size as f64).min(1.0)
        } else {
            1.0
        };

        // Weighted combination of hit rate and memory efficiency
        (hit_rate * 0.7) + (memory_efficiency * 0.3)
    }

    /// Perform intelligent cache warming based on usage patterns
    pub fn warm_cache(&mut self) -> Result<()> {
        self.warming_predictor
            .update_predictions(&self.access_frequency);

        let candidates = self.warming_predictor.get_warming_candidates(10);
        for candidate in candidates {
            // Pre-compute and cache patterns for high-priority candidates
            if candidate.priority > 0.7 {
                // Simulate pre-computation by creating a placeholder entry
                // In a real implementation, this would trigger actual pattern computation

                // Mark this key as "warmed" for future cache lookup optimization
                let access_stats = AccessStats {
                    access_count: 1,
                    last_access: Instant::now(),
                    first_access: Instant::now(),
                    total_access_time_ns: 0,
                    avg_computation_time_ns: 0.0,
                };
                self.access_frequency
                    .insert(candidate.key.clone(), access_stats);

                // Update analytics for cache warming
                self.analytics
                    .record_cache_warming(candidate.key.clone(), candidate.priority);

                tracing::info!(
                    "Cache warmed for key: {} (priority: {:.2}) - marked for priority caching",
                    candidate.key,
                    candidate.priority
                );
            } else {
                tracing::debug!(
                    "Cache warming candidate: {} (priority: {:.2}) - skipped (priority too low)",
                    candidate.key,
                    candidate.priority
                );
            }
        }

        Ok(())
    }

    /// Reset cache analytics
    pub fn reset_analytics(&mut self) {
        self.analytics.reset();
    }

    /// Get compression statistics
    pub fn get_compression_stats(&self) -> &CompressionStats {
        self.compression_engine.get_stats()
    }

    /// Get current eviction strategy
    pub fn get_eviction_strategy(&self) -> EvictionStrategy {
        self.eviction_manager.current_strategy()
    }

    /// Enable or disable compression
    pub fn set_compression_enabled(&mut self, enabled: bool) {
        self.compression_engine.set_enabled(enabled);
    }

    /// Perform adaptive eviction management
    pub fn optimize_eviction_strategy(&mut self) {
        let hit_rate = if self.analytics.total_requests > 0 {
            self.analytics.cache_hits as f64 / self.analytics.total_requests as f64
        } else {
            0.0
        };

        let memory_efficiency = if self.settings.max_cache_size > 0 {
            1.0 - (self.analytics.evictions as f64 / self.settings.max_cache_size as f64).min(1.0)
        } else {
            1.0
        };

        self.eviction_manager
            .adapt_strategy(hit_rate, memory_efficiency);
    }

    /// Advanced eviction with adaptive strategy selection
    fn advanced_eviction(&mut self, required_space: usize) {
        let cache_keys: Vec<String> = self.cache.keys().cloned().collect();
        let eviction_candidates = self.eviction_manager.select_eviction_candidates(
            &cache_keys,
            &self.access_frequency,
            required_space,
        );

        for key in eviction_candidates {
            self.cache.remove(&key);
            self.lru_order.retain(|k| k != &key);
            self.access_frequency.remove(&key);
            self.analytics.evictions += 1;
        }

        // Adapt eviction strategy based on performance
        self.optimize_eviction_strategy();
    }
}

impl Default for PatternCache {
    fn default() -> Self {
        Self::new(PatternCacheSettings::default())
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

/// Advanced cache statistics with performance analytics
#[derive(Debug, Clone)]
pub struct AdvancedCacheStats {
    pub basic_stats: CacheStats,
    pub hit_rate: f64,
    pub avg_access_time_ns: f64,
    pub total_evictions: u64,
    pub memory_usage_bytes: usize,
    pub most_accessed_keys: Vec<(String, u64)>,
    pub cache_efficiency_score: f64,
}

impl Default for CacheAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheAnalytics {
    /// Create new cache analytics
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            memory_usage_bytes: 0,
            last_reset: Instant::now(),
            avg_access_time_ns: 0.0,
            hit_rate_history: VecDeque::new(),
        }
    }

    /// Update average access time with exponential moving average
    pub fn update_avg_access_time(&mut self, access_time_ns: f64) {
        const ALPHA: f64 = 0.1; // Smoothing factor
        if self.avg_access_time_ns == 0.0 {
            self.avg_access_time_ns = access_time_ns;
        } else {
            self.avg_access_time_ns =
                ALPHA * access_time_ns + (1.0 - ALPHA) * self.avg_access_time_ns;
        }
    }

    /// Reset analytics
    pub fn reset(&mut self) {
        self.total_requests = 0;
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.evictions = 0;
        self.memory_usage_bytes = 0;
        self.last_reset = Instant::now();
        self.avg_access_time_ns = 0.0;
        self.hit_rate_history.clear();
    }

    /// Update hit rate history
    pub fn update_hit_rate_history(&mut self) {
        let hit_rate = if self.total_requests > 0 {
            self.cache_hits as f64 / self.total_requests as f64
        } else {
            0.0
        };

        self.hit_rate_history.push_back(hit_rate);

        // Keep only last 100 measurements
        if self.hit_rate_history.len() > 100 {
            self.hit_rate_history.pop_front();
        }
    }

    /// Record cache warming activity
    pub fn record_cache_warming(&mut self, key: String, priority: f64) {
        // Track cache warming events (for future analytics)
        tracing::trace!(
            "Cache warming recorded for key: {} with priority: {:.2}",
            key,
            priority
        );
    }
}

impl Default for CacheWarmingPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheWarmingPredictor {
    /// Create new cache warming predictor
    pub fn new() -> Self {
        Self {
            prediction_model: PredictionModel::new(),
            warming_candidates: Vec::new(),
            last_analysis: None,
        }
    }

    /// Update predictions based on access patterns
    pub fn update_predictions(&mut self, access_frequency: &HashMap<String, AccessStats>) {
        // Simple prediction model based on access frequency and recency
        for (key, stats) in access_frequency {
            let recency_score = Self::calculate_recency_score(&stats.last_access);
            let frequency_score = (stats.access_count as f64).ln();
            let priority = recency_score * frequency_score;

            self.prediction_model
                .historical_patterns
                .insert(key.clone(), priority);
        }

        self.last_analysis = Some(Instant::now());
    }

    /// Get warming candidates sorted by priority
    pub fn get_warming_candidates(&self, limit: usize) -> Vec<WarmingCandidate> {
        let mut candidates: Vec<_> = self
            .prediction_model
            .historical_patterns
            .iter()
            .map(|(key, &priority)| WarmingCandidate {
                key: key.clone(),
                priority,
                predicted_access_time: Instant::now() + Duration::from_secs(60), // Predict access in 1 minute
                computation_cost: 1.0,                                           // Placeholder
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);
        candidates
    }

    /// Calculate recency score (0.0 to 1.0, higher for more recent)
    fn calculate_recency_score(last_access: &Instant) -> f64 {
        let elapsed = last_access.elapsed().as_secs_f64();
        let max_age = 3600.0; // 1 hour
        (max_age - elapsed.min(max_age)) / max_age
    }

    /// Clear predictions
    pub fn clear(&mut self) {
        self.prediction_model.historical_patterns.clear();
        self.warming_candidates.clear();
        self.last_analysis = None;
    }
}

impl Default for PredictionModel {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictionModel {
    /// Create new prediction model
    pub fn new() -> Self {
        Self {
            historical_patterns: HashMap::new(),
            time_based_weights: VecDeque::new(),
        }
    }
}

impl Default for CompressionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionEngine {
    /// Create new compression engine
    pub fn new() -> Self {
        Self {
            enabled: true,          // Enable by default for memory optimization
            compression_ratio: 0.7, // Target 70% compression
            compression_stats: CompressionStats::new(),
        }
    }

    /// Compress pattern data (simulated compression)
    pub fn compress(&mut self, data: &[Pattern]) -> Result<Vec<u8>> {
        if !self.enabled {
            return Ok(serde_json::to_vec(data)?);
        }

        let start_time = Instant::now();

        // Simulate compression by serializing and applying a compression ratio
        let serialized = serde_json::to_vec(data)?;
        let uncompressed_size = serialized.len();

        // Simulate compressed data (in reality would use zlib, lz4, etc.)
        let compressed_size = (uncompressed_size as f64 * self.compression_ratio) as usize;
        let compressed_data = vec![0u8; compressed_size]; // Placeholder

        // Update compression statistics
        self.compression_stats.total_compressed += 1;
        self.compression_stats.total_uncompressed_size += uncompressed_size;
        self.compression_stats.total_compressed_size += compressed_size;
        self.compression_stats.compression_time_ns += start_time.elapsed().as_nanos() as u64;

        // Update average compression ratio
        let total_ratio = self.compression_stats.total_compressed_size as f64
            / self.compression_stats.total_uncompressed_size as f64;
        self.compression_stats.avg_compression_ratio = total_ratio;

        Ok(compressed_data)
    }

    /// Decompress pattern data (simulated decompression)
    pub fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<Pattern>> {
        if !self.enabled {
            return Ok(serde_json::from_slice(compressed_data)?);
        }

        // Simulate decompression - in reality would decompress the actual data
        // For now, return empty patterns as placeholder
        Ok(Vec::new())
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> &CompressionStats {
        &self.compression_stats
    }

    /// Enable or disable compression
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionStats {
    /// Create new compression statistics
    pub fn new() -> Self {
        Self {
            total_compressed: 0,
            total_uncompressed_size: 0,
            total_compressed_size: 0,
            avg_compression_ratio: 0.0,
            compression_time_ns: 0,
        }
    }
}

impl Default for AdaptiveEvictionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveEvictionManager {
    /// Create new adaptive eviction manager
    pub fn new() -> Self {
        let mut strategy_performance = HashMap::new();

        // Initialize performance metrics for each strategy
        for strategy in [
            EvictionStrategy::LRU,
            EvictionStrategy::LFU,
            EvictionStrategy::TLRU,
            EvictionStrategy::AdaptiveTTL,
            EvictionStrategy::MemoryPressure,
        ] {
            strategy_performance.insert(strategy, EvictionPerformance::new());
        }

        Self {
            current_strategy: EvictionStrategy::LRU, // Start with LRU
            strategy_performance,
            adaptation_threshold: 0.1, // 10% improvement threshold
            last_adaptation: Instant::now(),
        }
    }

    /// Select eviction candidates based on current strategy
    pub fn select_eviction_candidates(
        &self,
        cache_keys: &[String],
        access_frequency: &HashMap<String, AccessStats>,
        required_space: usize,
    ) -> Vec<String> {
        match self.current_strategy {
            EvictionStrategy::LRU => {
                self.select_lru_candidates(cache_keys, access_frequency, required_space)
            }
            EvictionStrategy::LFU => {
                self.select_lfu_candidates(cache_keys, access_frequency, required_space)
            }
            EvictionStrategy::TLRU => {
                self.select_tlru_candidates(cache_keys, access_frequency, required_space)
            }
            EvictionStrategy::AdaptiveTTL => {
                self.select_adaptive_ttl_candidates(cache_keys, access_frequency, required_space)
            }
            EvictionStrategy::MemoryPressure => {
                self.select_memory_pressure_candidates(cache_keys, access_frequency, required_space)
            }
        }
    }

    /// Adapt eviction strategy based on performance
    pub fn adapt_strategy(&mut self, recent_hit_rate: f64, memory_efficiency: f64) {
        // Only adapt if enough time has passed
        if self.last_adaptation.elapsed() < Duration::from_secs(300) {
            return;
        }

        // Update current strategy performance and get the current score
        let current_score = {
            let current_performance = self
                .strategy_performance
                .get_mut(&self.current_strategy)
                .unwrap();
            current_performance.hit_rate_after_eviction = recent_hit_rate;
            current_performance.memory_efficiency = memory_efficiency;
            current_performance.sample_count += 1;
            current_performance.hit_rate_after_eviction * current_performance.memory_efficiency
        };

        // Find the best performing strategy
        let best_strategy = self.find_best_strategy();

        // Switch if there's significant improvement
        if best_strategy != self.current_strategy {
            let best_performance = self.strategy_performance.get(&best_strategy).unwrap();
            let best_score =
                best_performance.hit_rate_after_eviction * best_performance.memory_efficiency;

            if best_score > current_score + self.adaptation_threshold {
                tracing::info!(
                    "Adapting eviction strategy from {:?} to {:?} (score improvement: {:.3})",
                    self.current_strategy,
                    best_strategy,
                    best_score - current_score
                );
                self.current_strategy = best_strategy;
                self.last_adaptation = Instant::now();
            }
        }
    }

    /// Find the best performing eviction strategy
    fn find_best_strategy(&self) -> EvictionStrategy {
        let mut best_strategy = self.current_strategy;
        let mut best_score = 0.0;

        for (strategy, performance) in &self.strategy_performance {
            if performance.sample_count > 0 {
                let score = performance.hit_rate_after_eviction * performance.memory_efficiency;
                if score > best_score {
                    best_score = score;
                    best_strategy = *strategy;
                }
            }
        }

        best_strategy
    }

    /// Select LRU eviction candidates
    fn select_lru_candidates(
        &self,
        cache_keys: &[String],
        access_frequency: &HashMap<String, AccessStats>,
        required_space: usize,
    ) -> Vec<String> {
        let mut candidates: Vec<_> = cache_keys
            .iter()
            .filter_map(|key| {
                access_frequency
                    .get(key)
                    .map(|stats| (key.clone(), stats.last_access))
            })
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1)); // Oldest first
        candidates
            .into_iter()
            .take(required_space)
            .map(|(key, _)| key)
            .collect()
    }

    /// Select LFU eviction candidates
    fn select_lfu_candidates(
        &self,
        cache_keys: &[String],
        access_frequency: &HashMap<String, AccessStats>,
        required_space: usize,
    ) -> Vec<String> {
        let mut candidates: Vec<_> = cache_keys
            .iter()
            .filter_map(|key| {
                access_frequency
                    .get(key)
                    .map(|stats| (key.clone(), stats.access_count))
            })
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1)); // Least frequent first
        candidates
            .into_iter()
            .take(required_space)
            .map(|(key, _)| key)
            .collect()
    }

    /// Select time-aware LRU candidates
    fn select_tlru_candidates(
        &self,
        cache_keys: &[String],
        access_frequency: &HashMap<String, AccessStats>,
        required_space: usize,
    ) -> Vec<String> {
        let mut candidates: Vec<_> = cache_keys
            .iter()
            .filter_map(|key| {
                access_frequency.get(key).map(|stats| {
                    let recency_weight = 1.0 / (stats.last_access.elapsed().as_secs_f64() + 1.0);
                    let frequency_weight = (stats.access_count as f64).ln_1p();
                    let score = recency_weight * frequency_weight;
                    (key.clone(), score)
                })
            })
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
            .into_iter()
            .take(required_space)
            .map(|(key, _)| key)
            .collect()
    }

    /// Select adaptive TTL candidates
    fn select_adaptive_ttl_candidates(
        &self,
        cache_keys: &[String],
        access_frequency: &HashMap<String, AccessStats>,
        required_space: usize,
    ) -> Vec<String> {
        // Similar to LRU but with adaptive TTL based on access patterns
        self.select_lru_candidates(cache_keys, access_frequency, required_space)
    }

    /// Select memory pressure-based candidates
    fn select_memory_pressure_candidates(
        &self,
        cache_keys: &[String],
        access_frequency: &HashMap<String, AccessStats>,
        required_space: usize,
    ) -> Vec<String> {
        // Prioritize evicting larger, less frequently accessed items
        let mut candidates: Vec<_> = cache_keys
            .iter()
            .filter_map(|key| {
                access_frequency.get(key).map(|stats| {
                    let frequency_penalty = 1.0 / (stats.access_count as f64 + 1.0);
                    let size_estimate = 1024.0; // Rough estimate per item
                    let pressure_score = frequency_penalty * size_estimate;
                    (key.clone(), pressure_score)
                })
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
            .into_iter()
            .take(required_space)
            .map(|(key, _)| key)
            .collect()
    }

    /// Get current eviction strategy
    pub fn current_strategy(&self) -> EvictionStrategy {
        self.current_strategy
    }

    /// Get strategy performance metrics
    pub fn get_strategy_performance(&self) -> &HashMap<EvictionStrategy, EvictionPerformance> {
        &self.strategy_performance
    }
}

impl Default for EvictionPerformance {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPerformance {
    /// Create new eviction performance metrics
    pub fn new() -> Self {
        Self {
            hit_rate_after_eviction: 0.0,
            eviction_accuracy: 0.0,
            memory_efficiency: 0.0,
            sample_count: 0,
        }
    }
}
