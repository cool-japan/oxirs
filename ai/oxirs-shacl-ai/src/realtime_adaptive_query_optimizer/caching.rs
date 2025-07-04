//! Adaptive Plan Caching with ML-driven Eviction
//!
//! This module provides intelligent caching of query execution plans
//! with machine learning-based eviction policies and performance tracking.

use std::collections::HashMap;
use std::time::SystemTime;

use oxirs_core::query::pattern_optimizer::OptimizedPatternPlan;

use super::config::AdaptiveOptimizerConfig;
use super::performance::{OptimizationPlanType, TrendDirection};

/// Adaptive plan cache with ML-driven eviction
#[derive(Debug)]
pub struct AdaptivePlanCache {
    /// Cached plans
    cache: HashMap<String, CachedPlan>,

    /// Plan access patterns
    access_patterns: HashMap<String, AccessPattern>,

    /// Configuration
    config: AdaptiveOptimizerConfig,

    /// Cache statistics
    stats: CacheStatistics,
}

/// Cached query plan with metadata
#[derive(Debug, Clone)]
pub struct CachedPlan {
    pub plan: OptimizedPatternPlan,
    pub plan_type: OptimizationPlanType,
    pub creation_time: SystemTime,
    pub last_access_time: SystemTime,
    pub access_count: usize,
    pub average_performance_ms: f64,
    pub success_rate: f64,
    pub cache_key: String,
}

/// Access pattern for cache entries
#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub frequency: f64,
    pub recency: f64,
    pub performance_score: f64,
    pub trend: TrendDirection,
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub total_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub evictions: usize,
    pub hit_rate: f64,
    pub avg_lookup_time_ms: f64,
}

impl AdaptivePlanCache {
    pub fn new(config: AdaptiveOptimizerConfig) -> Self {
        Self {
            cache: HashMap::new(),
            access_patterns: HashMap::new(),
            config,
            stats: CacheStatistics::default(),
        }
    }

    /// Get cached plan if available
    pub fn get(&mut self, cache_key: &str) -> Option<CachedPlan> {
        self.stats.total_requests += 1;

        let result = if let Some(plan) = self.cache.get_mut(cache_key) {
            plan.last_access_time = SystemTime::now();
            plan.access_count += 1;
            Some(plan.clone())
        } else {
            None
        };

        if result.is_some() {
            // Update access pattern after releasing the mutable borrow
            self.update_access_pattern(cache_key);
            self.stats.cache_hits += 1;
            self.stats.hit_rate = self.stats.cache_hits as f64 / self.stats.total_requests as f64;
        } else {
            self.stats.cache_misses += 1;
            self.stats.hit_rate = self.stats.cache_hits as f64 / self.stats.total_requests as f64;
        }

        result
    }

    /// Store plan in cache
    pub fn put(
        &mut self,
        cache_key: String,
        plan: OptimizedPatternPlan,
        plan_type: OptimizationPlanType,
    ) {
        let cached_plan = CachedPlan {
            plan,
            plan_type,
            creation_time: SystemTime::now(),
            last_access_time: SystemTime::now(),
            access_count: 1,
            average_performance_ms: 0.0,
            success_rate: 1.0,
            cache_key: cache_key.clone(),
        };

        // Check if cache is full
        if self.cache.len() >= self.config.plan_cache_size {
            self.evict_least_valuable();
        }

        self.cache.insert(cache_key.clone(), cached_plan);
        self.init_access_pattern(&cache_key);
    }

    /// Update access pattern for cache entry
    fn update_access_pattern(&mut self, cache_key: &str) {
        let pattern = self
            .access_patterns
            .entry(cache_key.to_string())
            .or_insert_with(AccessPattern::default);

        pattern.frequency += 1.0;
        pattern.recency = 1.0; // Reset recency on access

        // Apply decay to frequency over time
        pattern.frequency *= 0.99;
    }

    /// Initialize access pattern for new entry
    fn init_access_pattern(&mut self, cache_key: &str) {
        self.access_patterns
            .insert(cache_key.to_string(), AccessPattern::default());
    }

    /// Evict least valuable cache entry
    fn evict_least_valuable(&mut self) {
        let mut least_valuable_key: Option<String> = None;
        let mut least_value = f64::INFINITY;

        for (key, pattern) in &self.access_patterns {
            let value = self.calculate_cache_value(pattern);
            if value < least_value {
                least_value = value;
                least_valuable_key = Some(key.clone());
            }
        }

        if let Some(key) = least_valuable_key {
            self.cache.remove(&key);
            self.access_patterns.remove(&key);
            self.stats.evictions += 1;
        }
    }

    /// Calculate cache value for eviction policy
    fn calculate_cache_value(&self, pattern: &AccessPattern) -> f64 {
        // Combine frequency, recency, and performance
        let frequency_weight = 0.4;
        let recency_weight = 0.3;
        let performance_weight = 0.3;

        frequency_weight * pattern.frequency
            + recency_weight * pattern.recency
            + performance_weight * pattern.performance_score
    }

    /// Update performance for cached plan
    pub fn update_performance(&mut self, cache_key: &str, execution_time_ms: f64, success: bool) {
        if let Some(cached_plan) = self.cache.get_mut(cache_key) {
            let prev_avg = cached_plan.average_performance_ms;
            let count = cached_plan.access_count as f64;

            cached_plan.average_performance_ms =
                (prev_avg * (count - 1.0) + execution_time_ms) / count;

            let prev_success_rate = cached_plan.success_rate;
            cached_plan.success_rate =
                (prev_success_rate * (count - 1.0) + if success { 1.0 } else { 0.0 }) / count;
        }

        // Update access pattern performance score
        if let Some(pattern) = self.access_patterns.get_mut(cache_key) {
            pattern.performance_score = if execution_time_ms > 0.0 {
                1000.0 / execution_time_ms // Higher score for faster execution
            } else {
                1.0
            };
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStatistics {
        self.stats.clone()
    }
}

impl Default for AccessPattern {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            recency: 1.0,
            performance_score: 1.0,
            trend: TrendDirection::Stable,
        }
    }
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            hit_rate: 0.0,
            avg_lookup_time_ms: 0.0,
        }
    }
}
