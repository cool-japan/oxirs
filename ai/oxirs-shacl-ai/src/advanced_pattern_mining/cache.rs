//! Intelligent pattern cache system with advanced management features

use std::collections::HashMap;
use std::time::SystemTime;

use super::patterns::AdvancedPattern;
use super::types::*;

/// Intelligent pattern cache with advanced management features
#[derive(Debug)]
pub struct IntelligentPatternCache {
    /// Cache entries with access tracking
    entries: HashMap<String, CacheEntry>,

    /// Cache analytics and performance metrics
    analytics: CacheAnalytics,

    /// Cache configuration
    config: CacheConfig,

    /// Access frequency tracking for LRU eviction
    access_tracker: AccessTracker,

    /// Cache warming predictor for proactive caching
    warming_predictor: CacheWarmingPredictor,

    /// Performance tuner for adaptive optimization
    performance_tuner: CachePerformanceTuner,

    /// Advanced eviction strategy selector
    eviction_strategy: EvictionStrategy,
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Cached patterns
    patterns: Vec<AdvancedPattern>,

    /// Last access timestamp
    last_accessed: SystemTime,

    /// Access count
    access_count: usize,

    /// Cache insertion time
    created_at: SystemTime,

    /// Entry size in bytes (estimated)
    size_bytes: usize,

    /// Quality score of cached patterns
    quality_score: f64,
}

/// Cache analytics for performance monitoring
#[derive(Debug, Default, Clone)]
pub struct CacheAnalytics {
    /// Total cache hits
    pub hits: usize,

    /// Total cache misses
    pub misses: usize,

    /// Cache hit rate (calculated)
    pub hit_rate: f64,

    /// Total memory used by cache
    pub memory_usage_bytes: usize,

    /// Average access time in milliseconds
    pub avg_access_time_ms: f64,

    /// Cache efficiency score
    pub efficiency_score: f64,

    /// Most frequently accessed keys
    pub hot_keys: Vec<String>,

    /// Access pattern analytics
    pub access_patterns: HashMap<String, AccessPattern>,
}

/// Access tracking for intelligent eviction
#[derive(Debug, Default)]
pub struct AccessTracker {
    /// Access frequency for each key
    access_frequencies: HashMap<String, usize>,

    /// Last access times for LRU tracking
    last_access_times: HashMap<String, SystemTime>,

    /// Access time series for pattern analysis
    access_history: HashMap<String, Vec<SystemTime>>,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of cache entries
    pub max_entries: usize,

    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,

    /// TTL for cache entries in seconds
    pub ttl_seconds: u64,

    /// Enable intelligent eviction (LRU + quality-based)
    pub intelligent_eviction: bool,

    /// Enable cache analytics
    pub enable_analytics: bool,

    /// Minimum quality score for caching
    pub min_quality_for_cache: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_memory_bytes: 100 * 1024 * 1024, // 100MB
            ttl_seconds: 3600, // 1 hour
            intelligent_eviction: true,
            enable_analytics: true,
            min_quality_for_cache: 0.7,
        }
    }
}

/// Access pattern for analytics
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Frequency distribution over time
    pub frequency_distribution: Vec<(SystemTime, usize)>,

    /// Peak access times
    pub peak_times: Vec<SystemTime>,

    /// Average interval between accesses
    pub avg_interval_seconds: f64,

    /// Access trend (increasing, decreasing, stable)
    pub trend: AccessTrend,
}

/// Cache warming predictor for proactive caching
#[derive(Debug, Default)]
pub struct CacheWarmingPredictor {
    /// Prediction models for each key
    prediction_models: HashMap<String, PredictionModel>,

    /// Historical access patterns
    historical_patterns: HashMap<String, Vec<AccessEvent>>,

    /// Prediction configuration
    config: PredictionConfig,
}

/// Cache performance tuner for adaptive optimization
#[derive(Debug, Default)]
pub struct CachePerformanceTuner {
    /// Current tuning state
    state: TuningState,

    /// Performance metrics history
    metrics_history: Vec<PerformanceSnapshot>,

    /// Tuning configuration
    config: TuningConfig,

    /// Strategy switching configuration
    strategy_config: StrategySwitchingConfig,
}

/// Advanced eviction strategy selector
#[derive(Debug)]
pub struct EvictionStrategy {
    /// Current algorithm in use
    current_algorithm: EvictionAlgorithm,

    /// Algorithm performance history
    algorithm_performance: HashMap<EvictionAlgorithm, PerformanceMetrics>,

    /// Strategy switching configuration
    switching_config: StrategySwitchingConfig,
}

/// Prediction model for cache warming
#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Model type (linear, exponential, etc.)
    model_type: String,

    /// Model parameters
    parameters: HashMap<String, f64>,

    /// Prediction accuracy
    accuracy: f64,

    /// Last update time
    last_updated: SystemTime,
}

/// Access event for prediction
#[derive(Debug, Clone)]
pub struct AccessEvent {
    /// Event timestamp
    timestamp: SystemTime,

    /// Access type
    access_type: AccessResult,

    /// Context information
    context: HashMap<String, String>,
}

/// Prediction configuration
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    /// Enable predictive warming
    pub enable_prediction: bool,

    /// Prediction window size
    pub window_size: usize,

    /// Minimum accuracy threshold
    pub min_accuracy: f64,

    /// Update frequency
    pub update_frequency_seconds: u64,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            enable_prediction: true,
            window_size: 100,
            min_accuracy: 0.6,
            update_frequency_seconds: 300,
        }
    }
}

/// Performance snapshot for tuning
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Snapshot timestamp
    timestamp: SystemTime,

    /// Hit rate at snapshot time
    hit_rate: f64,

    /// Average response time
    avg_response_time_ms: f64,

    /// Memory usage
    memory_usage_bytes: usize,

    /// Cache efficiency score
    efficiency_score: f64,
}

/// Tuning configuration
#[derive(Debug, Clone)]
pub struct TuningConfig {
    /// Enable adaptive tuning
    pub enable_tuning: bool,

    /// Tuning interval
    pub tuning_interval_seconds: u64,

    /// Performance degradation threshold
    pub degradation_threshold: f64,

    /// Minimum improvement for changes
    pub min_improvement: f64,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            enable_tuning: true,
            tuning_interval_seconds: 600, // 10 minutes
            degradation_threshold: 0.1,
            min_improvement: 0.05,
        }
    }
}

/// Strategy switching configuration
#[derive(Debug, Clone)]
pub struct StrategySwitchingConfig {
    /// Enable automatic strategy switching
    pub enable_switching: bool,

    /// Evaluation window size
    pub evaluation_window: usize,

    /// Performance threshold for switching
    pub switch_threshold: f64,

    /// Cooldown period between switches
    pub cooldown_seconds: u64,
}

impl Default for StrategySwitchingConfig {
    fn default() -> Self {
        Self {
            enable_switching: true,
            evaluation_window: 100,
            switch_threshold: 0.15,
            cooldown_seconds: 1800, // 30 minutes
        }
    }
}

/// Performance metrics for eviction algorithms
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Hit rate achieved
    pub hit_rate: f64,

    /// Average access time
    pub avg_access_time_ms: f64,

    /// Memory efficiency
    pub memory_efficiency: f64,

    /// Quality score retention
    pub quality_retention: f64,

    /// Overall performance score
    pub overall_score: f64,
}

impl Default for EvictionStrategy {
    fn default() -> Self {
        Self {
            current_algorithm: EvictionAlgorithm::Intelligent,
            algorithm_performance: HashMap::new(),
            switching_config: StrategySwitchingConfig::default(),
        }
    }
}

impl Default for IntelligentPatternCache {
    fn default() -> Self {
        Self::new()
    }
}

impl IntelligentPatternCache {
    /// Create new intelligent pattern cache
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            analytics: CacheAnalytics::default(),
            config: CacheConfig::default(),
            access_tracker: AccessTracker::default(),
            warming_predictor: CacheWarmingPredictor::default(),
            performance_tuner: CachePerformanceTuner::default(),
            eviction_strategy: EvictionStrategy::default(),
        }
    }

    /// Create cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            analytics: CacheAnalytics::default(),
            config,
            access_tracker: AccessTracker::default(),
            warming_predictor: CacheWarmingPredictor::default(),
            performance_tuner: CachePerformanceTuner::default(),
            eviction_strategy: EvictionStrategy::default(),
        }
    }

    /// Get patterns from cache
    pub fn get(&mut self, key: &str) -> Option<Vec<AdvancedPattern>> {
        let now = SystemTime::now();
        
        // First check if entry exists and get expiration info
        let (is_expired, patterns) = if let Some(entry) = self.entries.get(key) {
            let expired = self.is_expired(entry, now);
            let patterns = if !expired { Some(entry.patterns.clone()) } else { None };
            (expired, patterns)
        } else {
            (false, None)
        };

        if is_expired {
            // Remove expired entry
            self.entries.remove(key);
            
            // Record expired access for warming predictor
            self.warming_predictor.record_access(key, AccessResult::Expired, 0.0);
            
            self.analytics.misses += 1;
            self.update_analytics();
            return None;
        }

        if let Some(patterns) = patterns {
            // Update access tracking for existing entry
            let response_time = 1.0; // In real implementation, measure actual response time
            if let Some(entry) = self.entries.get_mut(key) {
                entry.last_accessed = now;
                entry.access_count += 1;
            }
            
            // Update access tracker
            self.access_tracker.record_access(key, now);
            
            // Record access event for warming predictor
            self.warming_predictor.record_access(key, AccessResult::Hit, response_time);
            
            // Update analytics
            self.analytics.hits += 1;
            self.update_analytics();
            
            Some(patterns)
        } else {
            // Record miss for warming predictor
            self.warming_predictor.record_access(key, AccessResult::Miss, 0.0);
            
            self.analytics.misses += 1;
            self.update_analytics();
            None
        }
    }

    /// Store patterns in cache
    pub fn put(&mut self, key: String, patterns: Vec<AdvancedPattern>) {
        let now = SystemTime::now();
        
        // Calculate quality score for patterns
        let quality_score = self.calculate_patterns_quality(&patterns);
        
        // Check if quality meets minimum threshold
        if quality_score < self.config.min_quality_for_cache {
            return;
        }

        // Estimate entry size
        let size_bytes = self.estimate_entry_size(&patterns);
        
        // Check memory limits and evict if necessary
        if self.needs_eviction(size_bytes) {
            self.evict_entries(size_bytes);
        }

        // Create new cache entry
        let entry = CacheEntry {
            patterns,
            last_accessed: now,
            access_count: 1,
            created_at: now,
            size_bytes,
            quality_score,
        };

        // Insert entry
        self.entries.insert(key.clone(), entry);
        
        // Update access tracker
        self.access_tracker.record_access(&key, now);
        
        // Update analytics
        self.update_memory_usage();
        self.update_analytics();
    }

    /// Clear all cache entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_tracker = AccessTracker::default();
        self.analytics = CacheAnalytics::default();
    }

    /// Get cache analytics
    pub fn get_analytics(&self) -> &CacheAnalytics {
        &self.analytics
    }

    /// Get cache statistics as JSON
    pub fn get_stats(&self) -> crate::Result<serde_json::Value> {
        Ok(serde_json::json!({
            "entries": self.entries.len(),
            "hit_rate": self.analytics.hit_rate,
            "memory_usage_bytes": self.analytics.memory_usage_bytes,
            "efficiency_score": self.analytics.efficiency_score
        }))
    }

    /// Check if cache entry is expired
    fn is_expired(&self, entry: &CacheEntry, now: SystemTime) -> bool {
        if self.config.ttl_seconds == 0 {
            return false; // No expiration
        }
        
        match now.duration_since(entry.created_at) {
            Ok(duration) => duration.as_secs() > self.config.ttl_seconds,
            Err(_) => true, // Clock went backwards, consider expired
        }
    }

    /// Calculate quality score for patterns
    fn calculate_patterns_quality(&self, patterns: &[AdvancedPattern]) -> f64 {
        if patterns.is_empty() {
            return 0.0;
        }

        patterns.iter().map(|p| p.quality_score).sum::<f64>() / patterns.len() as f64
    }

    /// Estimate memory usage for patterns
    fn estimate_entry_size(&self, patterns: &[AdvancedPattern]) -> usize {
        // Rough estimation - 1KB per pattern
        patterns.len() * 1024
    }

    /// Check if eviction is needed
    fn needs_eviction(&self, new_entry_size: usize) -> bool {
        if self.entries.len() >= self.config.max_entries {
            return true;
        }

        let current_memory = self.analytics.memory_usage_bytes;
        current_memory + new_entry_size > self.config.max_memory_bytes
    }

    /// Evict entries to make space
    fn evict_entries(&mut self, space_needed: usize) {
        // Simplified LRU eviction
        let mut entries_to_remove = Vec::new();
        
        // Sort entries by last access time
        let mut sorted_entries: Vec<_> = self.entries.iter().collect();
        sorted_entries.sort_by_key(|(_, entry)| entry.last_accessed);

        let mut freed_space = 0;
        for (key, entry) in sorted_entries {
            if freed_space >= space_needed {
                break;
            }
            entries_to_remove.push(key.clone());
            freed_space += entry.size_bytes;
        }

        for key in entries_to_remove {
            self.entries.remove(&key);
        }
    }

    /// Update memory usage analytics
    fn update_memory_usage(&mut self) {
        self.analytics.memory_usage_bytes = self.entries
            .values()
            .map(|entry| entry.size_bytes)
            .sum();
    }

    /// Update cache analytics
    fn update_analytics(&mut self) {
        let total_accesses = self.analytics.hits + self.analytics.misses;
        if total_accesses > 0 {
            self.analytics.hit_rate = self.analytics.hits as f64 / total_accesses as f64;
        }

        // Calculate efficiency score
        self.analytics.efficiency_score = self.analytics.hit_rate * 0.7 + 
            (1.0 - (self.analytics.memory_usage_bytes as f64 / self.config.max_memory_bytes as f64)) * 0.3;
    }
}

impl AccessTracker {
    /// Record access for a key
    pub fn record_access(&mut self, key: &str, timestamp: SystemTime) {
        // Update frequency
        *self.access_frequencies.entry(key.to_string()).or_insert(0) += 1;
        
        // Update last access time
        self.last_access_times.insert(key.to_string(), timestamp);
        
        // Add to history
        self.access_history
            .entry(key.to_string())
            .or_default()
            .push(timestamp);
    }
}

impl CacheWarmingPredictor {
    /// Record access event
    pub fn record_access(&mut self, key: &str, result: AccessResult, response_time: f64) {
        let event = AccessEvent {
            timestamp: SystemTime::now(),
            access_type: result,
            context: HashMap::new(),
        };

        self.historical_patterns
            .entry(key.to_string())
            .or_default()
            .push(event);
    }
}

impl CachePerformanceTuner {
    /// Record performance snapshot
    pub fn record_snapshot(&mut self, snapshot: PerformanceSnapshot) {
        self.metrics_history.push(snapshot);
        
        // Limit history size
        if self.metrics_history.len() > 100 {
            self.metrics_history.remove(0);
        }
    }
}