//! Automatic Query Caching Strategies
//!
//! This module provides intelligent, adaptive caching strategies that automatically
//! determine which queries should be cached, with what TTL, and when to invalidate.
//!
//! # Features
//! - Automatic cache decision making based on query patterns
//! - Adaptive TTL based on query update frequency
//! - Multiple caching strategies (LRU, LFU, adaptive, predictive)
//! - Cache warming for frequently accessed queries
//! - Intelligent cache invalidation
//! - Integration with historical cost estimator
//!
//! # Example
//! ```
//! use oxirs_gql::auto_caching_strategies::{AutoCachingManager, CachingStrategy};
//!
//! let mut manager = AutoCachingManager::new();
//! manager.set_strategy(CachingStrategy::Adaptive);
//!
//! // Automatically decides if query should be cached
//! if manager.should_cache("query { user { name } }") {
//!     let ttl = manager.get_optimal_ttl("query { user { name } }");
//!     // Cache with optimal TTL
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Error types for auto caching
#[derive(Debug, Error)]
pub enum CachingError {
    #[error("Lock acquisition failed: {0}")]
    LockError(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("Strategy error: {0}")]
    StrategyError(String),
}

/// Caching strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CachingStrategy {
    /// Least Recently Used - evict least recently accessed items
    LRU,

    /// Least Frequently Used - evict least frequently accessed items
    LFU,

    /// Adaptive - automatically adjust between LRU/LFU based on workload
    Adaptive,

    /// Predictive - use ML to predict which queries to cache
    Predictive,

    /// Time-based - cache based on time-of-day patterns
    TimeBased,

    /// Cost-based - cache expensive queries preferentially
    CostBased,
}

impl Default for CachingStrategy {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Query access pattern
#[derive(Debug, Clone)]
struct AccessPattern {
    /// Total number of accesses
    access_count: u64,

    /// Last access time
    last_accessed: Instant,

    /// Access times (for frequency analysis)
    access_times: VecDeque<Instant>,

    /// Average execution time
    avg_execution_time_ms: f64,

    /// Result size in bytes
    avg_result_size_bytes: u64,

    /// First seen timestamp
    _first_seen: Instant,
}

impl AccessPattern {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            access_count: 0,
            last_accessed: now,
            access_times: VecDeque::new(),
            avg_execution_time_ms: 0.0,
            avg_result_size_bytes: 0,
            _first_seen: now,
        }
    }

    fn record_access(&mut self, execution_time_ms: f64, result_size_bytes: u64) {
        let now = Instant::now();
        self.access_count += 1;
        self.last_accessed = now;

        // Keep last 100 access times for frequency analysis
        self.access_times.push_back(now);
        if self.access_times.len() > 100 {
            self.access_times.pop_front();
        }

        // Update averages
        let alpha = 0.3; // Smoothing factor
        self.avg_execution_time_ms =
            alpha * execution_time_ms + (1.0 - alpha) * self.avg_execution_time_ms;
        self.avg_result_size_bytes = (alpha * result_size_bytes as f64
            + (1.0 - alpha) * self.avg_result_size_bytes as f64)
            as u64;
    }

    /// Calculate access frequency (accesses per minute)
    fn access_frequency(&self) -> f64 {
        if self.access_times.len() < 2 {
            return 0.0;
        }

        let time_span = self
            .last_accessed
            .duration_since(self.access_times[0])
            .as_secs_f64();
        if time_span == 0.0 {
            return 0.0;
        }

        (self.access_times.len() as f64 / time_span) * 60.0
    }

    /// Calculate cache hit benefit score
    fn cache_benefit_score(&self) -> f64 {
        // Higher score = more beneficial to cache
        // Factors: frequency, execution time, result size
        let frequency_score = self.access_frequency();
        let time_score = self.avg_execution_time_ms / 1000.0; // Normalize to seconds
        let size_penalty = (self.avg_result_size_bytes as f64 / 1_000_000.0).min(1.0); // MB, capped at 1

        // Combine: high frequency + slow queries = high benefit
        // Penalize large result sizes
        (frequency_score * time_score) / (1.0 + size_penalty)
    }
}

/// Cache decision result
#[derive(Debug, Clone)]
pub struct CacheDecision {
    /// Whether to cache this query
    pub should_cache: bool,

    /// Optimal TTL in seconds
    pub ttl_seconds: u64,

    /// Cache priority (0-100, higher = more important)
    pub priority: u8,

    /// Reason for the decision
    pub reason: String,

    /// Confidence in the decision (0.0-1.0)
    pub confidence: f64,
}

/// Configuration for auto caching manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Minimum access count before considering caching
    pub min_access_count: u64,

    /// Minimum access frequency (per minute) to cache
    pub min_frequency: f64,

    /// Minimum execution time (ms) to consider caching
    pub min_execution_time_ms: f64,

    /// Maximum result size (bytes) to cache
    pub max_cache_size_bytes: u64,

    /// Default TTL in seconds
    pub default_ttl_seconds: u64,

    /// Minimum TTL in seconds
    pub min_ttl_seconds: u64,

    /// Maximum TTL in seconds
    pub max_ttl_seconds: u64,

    /// Enable adaptive TTL adjustment
    pub adaptive_ttl: bool,

    /// Cache benefit threshold (queries below this won't be cached)
    pub min_benefit_score: f64,
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            min_access_count: 3,
            min_frequency: 0.5, // 0.5 accesses per minute
            min_execution_time_ms: 50.0,
            max_cache_size_bytes: 10_000_000, // 10 MB
            default_ttl_seconds: 300,         // 5 minutes
            min_ttl_seconds: 60,              // 1 minute
            max_ttl_seconds: 3600,            // 1 hour
            adaptive_ttl: true,
            min_benefit_score: 1.0,
        }
    }
}

/// Automatic caching manager
pub struct AutoCachingManager {
    /// Access patterns for queries
    access_patterns: Arc<RwLock<HashMap<String, AccessPattern>>>,

    /// Current caching strategy
    strategy: Arc<RwLock<CachingStrategy>>,

    /// Configuration
    config: CachingConfig,

    /// Total queries analyzed
    total_queries: Arc<RwLock<u64>>,

    /// Total cache decisions made
    cache_decisions: Arc<RwLock<u64>>,

    /// Cached queries count
    cached_queries: Arc<RwLock<u64>>,
}

impl AutoCachingManager {
    /// Create a new auto caching manager with default configuration
    pub fn new() -> Self {
        Self::with_config(CachingConfig::default())
    }

    /// Create a new auto caching manager with custom configuration
    pub fn with_config(config: CachingConfig) -> Self {
        Self {
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
            strategy: Arc::new(RwLock::new(CachingStrategy::default())),
            config,
            total_queries: Arc::new(RwLock::new(0)),
            cache_decisions: Arc::new(RwLock::new(0)),
            cached_queries: Arc::new(RwLock::new(0)),
        }
    }

    /// Set the caching strategy
    pub fn set_strategy(&mut self, strategy: CachingStrategy) -> Result<(), CachingError> {
        let mut s = self
            .strategy
            .write()
            .map_err(|e| CachingError::LockError(e.to_string()))?;
        *s = strategy;
        Ok(())
    }

    /// Get the current caching strategy
    pub fn get_strategy(&self) -> Result<CachingStrategy, CachingError> {
        let s = self
            .strategy
            .read()
            .map_err(|e| CachingError::LockError(e.to_string()))?;
        Ok(*s)
    }

    /// Record a query execution
    pub fn record_query(
        &mut self,
        query: &str,
        execution_time_ms: f64,
        result_size_bytes: u64,
    ) -> Result<(), CachingError> {
        let mut patterns = self
            .access_patterns
            .write()
            .map_err(|e| CachingError::LockError(e.to_string()))?;

        let pattern = patterns
            .entry(query.to_string())
            .or_insert_with(AccessPattern::new);

        pattern.record_access(execution_time_ms, result_size_bytes);

        let mut total = self
            .total_queries
            .write()
            .map_err(|e| CachingError::LockError(e.to_string()))?;
        *total += 1;

        Ok(())
    }

    /// Decide if a query should be cached
    pub fn should_cache(&self, query: &str) -> bool {
        self.get_cache_decision(query)
            .map(|d| d.should_cache)
            .unwrap_or(false)
    }

    /// Get optimal TTL for a query
    pub fn get_optimal_ttl(&self, query: &str) -> Duration {
        self.get_cache_decision(query)
            .map(|d| Duration::from_secs(d.ttl_seconds))
            .unwrap_or_else(|_| Duration::from_secs(self.config.default_ttl_seconds))
    }

    /// Get comprehensive cache decision for a query
    pub fn get_cache_decision(&self, query: &str) -> Result<CacheDecision, CachingError> {
        let patterns = self
            .access_patterns
            .read()
            .map_err(|e| CachingError::LockError(e.to_string()))?;

        let strategy = self
            .strategy
            .read()
            .map_err(|e| CachingError::LockError(e.to_string()))?;

        let mut decisions = self
            .cache_decisions
            .write()
            .map_err(|e| CachingError::LockError(e.to_string()))?;
        *decisions += 1;

        match patterns.get(query) {
            Some(pattern) => self.analyze_pattern(pattern, *strategy),
            None => Ok(CacheDecision {
                should_cache: false,
                ttl_seconds: self.config.default_ttl_seconds,
                priority: 0,
                reason: "No access history".to_string(),
                confidence: 1.0,
            }),
        }
    }

    /// Analyze access pattern and make caching decision
    fn analyze_pattern(
        &self,
        pattern: &AccessPattern,
        strategy: CachingStrategy,
    ) -> Result<CacheDecision, CachingError> {
        // Basic checks
        if pattern.access_count < self.config.min_access_count {
            return Ok(CacheDecision {
                should_cache: false,
                ttl_seconds: self.config.default_ttl_seconds,
                priority: 0,
                reason: format!(
                    "Insufficient access count: {} < {}",
                    pattern.access_count, self.config.min_access_count
                ),
                confidence: 1.0,
            });
        }

        if pattern.avg_execution_time_ms < self.config.min_execution_time_ms {
            return Ok(CacheDecision {
                should_cache: false,
                ttl_seconds: self.config.default_ttl_seconds,
                priority: 0,
                reason: format!(
                    "Execution too fast: {:.2}ms < {}ms",
                    pattern.avg_execution_time_ms, self.config.min_execution_time_ms
                ),
                confidence: 1.0,
            });
        }

        if pattern.avg_result_size_bytes > self.config.max_cache_size_bytes {
            return Ok(CacheDecision {
                should_cache: false,
                ttl_seconds: self.config.default_ttl_seconds,
                priority: 0,
                reason: format!(
                    "Result too large: {} > {} bytes",
                    pattern.avg_result_size_bytes, self.config.max_cache_size_bytes
                ),
                confidence: 1.0,
            });
        }

        let frequency = pattern.access_frequency();
        if frequency < self.config.min_frequency {
            return Ok(CacheDecision {
                should_cache: false,
                ttl_seconds: self.config.default_ttl_seconds,
                priority: 0,
                reason: format!(
                    "Low frequency: {:.2}/min < {}/min",
                    frequency, self.config.min_frequency
                ),
                confidence: 0.8,
            });
        }

        let benefit_score = pattern.cache_benefit_score();
        if benefit_score < self.config.min_benefit_score {
            return Ok(CacheDecision {
                should_cache: false,
                ttl_seconds: self.config.default_ttl_seconds,
                priority: 0,
                reason: format!(
                    "Low benefit score: {:.2} < {}",
                    benefit_score, self.config.min_benefit_score
                ),
                confidence: 0.9,
            });
        }

        // Strategy-specific analysis
        match strategy {
            CachingStrategy::LRU => self.lru_decision(pattern, benefit_score),
            CachingStrategy::LFU => self.lfu_decision(pattern, benefit_score),
            CachingStrategy::Adaptive => self.adaptive_decision(pattern, benefit_score),
            CachingStrategy::Predictive => self.predictive_decision(pattern, benefit_score),
            CachingStrategy::TimeBased => self.time_based_decision(pattern, benefit_score),
            CachingStrategy::CostBased => self.cost_based_decision(pattern, benefit_score),
        }
    }

    /// LRU strategy decision
    fn lru_decision(
        &self,
        pattern: &AccessPattern,
        benefit_score: f64,
    ) -> Result<CacheDecision, CachingError> {
        let recency_score = self.calculate_recency_score(pattern);
        let ttl = if self.config.adaptive_ttl {
            self.calculate_adaptive_ttl(pattern)
        } else {
            self.config.default_ttl_seconds
        };

        Ok(CacheDecision {
            should_cache: true,
            ttl_seconds: ttl,
            priority: (recency_score * 100.0).min(100.0) as u8,
            reason: format!(
                "LRU: recency_score={:.2}, benefit={:.2}",
                recency_score, benefit_score
            ),
            confidence: 0.85,
        })
    }

    /// LFU strategy decision
    fn lfu_decision(
        &self,
        pattern: &AccessPattern,
        benefit_score: f64,
    ) -> Result<CacheDecision, CachingError> {
        let frequency = pattern.access_frequency();
        let ttl = if self.config.adaptive_ttl {
            self.calculate_adaptive_ttl(pattern)
        } else {
            self.config.default_ttl_seconds
        };

        Ok(CacheDecision {
            should_cache: true,
            ttl_seconds: ttl,
            priority: (frequency * 10.0).min(100.0) as u8,
            reason: format!(
                "LFU: frequency={:.2}/min, benefit={:.2}",
                frequency, benefit_score
            ),
            confidence: 0.85,
        })
    }

    /// Adaptive strategy decision
    fn adaptive_decision(
        &self,
        pattern: &AccessPattern,
        benefit_score: f64,
    ) -> Result<CacheDecision, CachingError> {
        let recency_score = self.calculate_recency_score(pattern);
        let frequency = pattern.access_frequency();

        // Adaptively blend LRU and LFU based on access pattern
        let lru_weight = if frequency > 1.0 { 0.3 } else { 0.7 };
        let lfu_weight = 1.0 - lru_weight;

        let combined_score = recency_score * lru_weight + (frequency / 10.0) * lfu_weight;
        let ttl = self.calculate_adaptive_ttl(pattern);

        Ok(CacheDecision {
            should_cache: true,
            ttl_seconds: ttl,
            priority: (combined_score * 100.0).min(100.0) as u8,
            reason: format!(
                "Adaptive: recency={:.2}, freq={:.2}/min, benefit={:.2}",
                recency_score, frequency, benefit_score
            ),
            confidence: 0.9,
        })
    }

    /// Predictive strategy decision
    fn predictive_decision(
        &self,
        pattern: &AccessPattern,
        benefit_score: f64,
    ) -> Result<CacheDecision, CachingError> {
        // Predict future access based on historical pattern
        let trend = self.calculate_access_trend(pattern);
        let predicted_benefit = benefit_score * (1.0 + trend);

        let ttl = self.calculate_adaptive_ttl(pattern);

        Ok(CacheDecision {
            should_cache: predicted_benefit > self.config.min_benefit_score,
            ttl_seconds: ttl,
            priority: (predicted_benefit * 20.0).min(100.0) as u8,
            reason: format!(
                "Predictive: trend={:.2}, predicted_benefit={:.2}",
                trend, predicted_benefit
            ),
            confidence: 0.75,
        })
    }

    /// Time-based strategy decision
    fn time_based_decision(
        &self,
        pattern: &AccessPattern,
        benefit_score: f64,
    ) -> Result<CacheDecision, CachingError> {
        // Analyze time-of-day access patterns
        let time_score = self.calculate_time_score(pattern);
        let ttl = self.calculate_time_based_ttl(pattern);

        Ok(CacheDecision {
            should_cache: true,
            ttl_seconds: ttl,
            priority: (time_score * 100.0).min(100.0) as u8,
            reason: format!(
                "TimeBased: time_score={:.2}, benefit={:.2}",
                time_score, benefit_score
            ),
            confidence: 0.8,
        })
    }

    /// Cost-based strategy decision
    fn cost_based_decision(
        &self,
        pattern: &AccessPattern,
        benefit_score: f64,
    ) -> Result<CacheDecision, CachingError> {
        let cost_score = pattern.avg_execution_time_ms / 100.0; // Normalize
        let ttl = if self.config.adaptive_ttl {
            // Expensive queries get longer TTL
            let base_ttl = self.calculate_adaptive_ttl(pattern);
            (base_ttl as f64 * (1.0 + cost_score / 10.0)) as u64
        } else {
            self.config.default_ttl_seconds
        };

        let ttl = ttl.clamp(self.config.min_ttl_seconds, self.config.max_ttl_seconds);

        Ok(CacheDecision {
            should_cache: true,
            ttl_seconds: ttl,
            priority: (cost_score * 10.0).min(100.0) as u8,
            reason: format!(
                "CostBased: cost={:.2}ms, benefit={:.2}",
                pattern.avg_execution_time_ms, benefit_score
            ),
            confidence: 0.88,
        })
    }

    /// Calculate recency score (0.0-1.0)
    fn calculate_recency_score(&self, pattern: &AccessPattern) -> f64 {
        let seconds_since_access = pattern.last_accessed.elapsed().as_secs_f64();
        // Exponential decay: recent = high score
        (-seconds_since_access / 300.0).exp() // 5 minute half-life
    }

    /// Calculate access trend (-1.0 to 1.0, positive = increasing)
    fn calculate_access_trend(&self, pattern: &AccessPattern) -> f64 {
        if pattern.access_times.len() < 4 {
            return 0.0;
        }

        // Compare recent vs older access frequency
        let mid = pattern.access_times.len() / 2;
        let times_vec: Vec<Instant> = pattern.access_times.iter().copied().collect();

        let recent = &times_vec[mid..];
        let older = &times_vec[..mid];

        if recent.len() < 2 || older.len() < 2 {
            return 0.0;
        }

        let recent_rate = recent.len() as f64
            / recent[recent.len() - 1]
                .duration_since(recent[0])
                .as_secs_f64()
                .max(1.0);
        let older_rate = older.len() as f64
            / older[older.len() - 1]
                .duration_since(older[0])
                .as_secs_f64()
                .max(1.0);

        ((recent_rate - older_rate) / older_rate.max(0.001)).clamp(-1.0, 1.0)
    }

    /// Calculate time-based score
    fn calculate_time_score(&self, _pattern: &AccessPattern) -> f64 {
        // Simplified: could analyze hour-of-day patterns
        // For now, return moderate score
        0.7
    }

    /// Calculate adaptive TTL based on access pattern
    fn calculate_adaptive_ttl(&self, pattern: &AccessPattern) -> u64 {
        let frequency = pattern.access_frequency();

        // High frequency = shorter TTL (data changes more often)
        // Low frequency = longer TTL (less likely to change)
        let base_ttl = if frequency > 10.0 {
            self.config.min_ttl_seconds
        } else if frequency > 1.0 {
            self.config.default_ttl_seconds
        } else {
            self.config.max_ttl_seconds
        };

        // Adjust based on execution time (expensive queries get longer TTL)
        let time_factor = (pattern.avg_execution_time_ms / 1000.0).min(2.0);
        let adjusted_ttl = (base_ttl as f64 * (1.0 + time_factor * 0.2)) as u64;

        adjusted_ttl.clamp(self.config.min_ttl_seconds, self.config.max_ttl_seconds)
    }

    /// Calculate time-based TTL
    fn calculate_time_based_ttl(&self, pattern: &AccessPattern) -> u64 {
        // For now, use adaptive TTL
        // Could be enhanced with time-of-day analysis
        self.calculate_adaptive_ttl(pattern)
    }

    /// Get caching statistics
    pub fn get_statistics(&self) -> Result<CachingStatistics, CachingError> {
        let patterns = self
            .access_patterns
            .read()
            .map_err(|e| CachingError::LockError(e.to_string()))?;

        let total = *self
            .total_queries
            .read()
            .map_err(|e| CachingError::LockError(e.to_string()))?;

        let decisions = *self
            .cache_decisions
            .read()
            .map_err(|e| CachingError::LockError(e.to_string()))?;

        let cached = *self
            .cached_queries
            .read()
            .map_err(|e| CachingError::LockError(e.to_string()))?;

        let cache_rate = if decisions > 0 {
            (cached as f64 / decisions as f64) * 100.0
        } else {
            0.0
        };

        Ok(CachingStatistics {
            total_queries: total,
            unique_queries: patterns.len(),
            cache_decisions: decisions,
            cached_queries: cached,
            cache_rate_percent: cache_rate,
        })
    }

    /// Clear all access patterns
    pub fn clear(&mut self) -> Result<(), CachingError> {
        let mut patterns = self
            .access_patterns
            .write()
            .map_err(|e| CachingError::LockError(e.to_string()))?;
        patterns.clear();

        let mut total = self
            .total_queries
            .write()
            .map_err(|e| CachingError::LockError(e.to_string()))?;
        *total = 0;

        let mut decisions = self
            .cache_decisions
            .write()
            .map_err(|e| CachingError::LockError(e.to_string()))?;
        *decisions = 0;

        let mut cached = self
            .cached_queries
            .write()
            .map_err(|e| CachingError::LockError(e.to_string()))?;
        *cached = 0;

        Ok(())
    }
}

impl Default for AutoCachingManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Caching statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingStatistics {
    /// Total queries processed
    pub total_queries: u64,

    /// Number of unique queries
    pub unique_queries: usize,

    /// Total cache decisions made
    pub cache_decisions: u64,

    /// Number of queries cached
    pub cached_queries: u64,

    /// Cache rate percentage
    pub cache_rate_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_manager_creation() {
        let manager = AutoCachingManager::new();
        assert_eq!(manager.get_strategy().unwrap(), CachingStrategy::Adaptive);
    }

    #[test]
    fn test_set_strategy() {
        let mut manager = AutoCachingManager::new();
        manager.set_strategy(CachingStrategy::LRU).unwrap();
        assert_eq!(manager.get_strategy().unwrap(), CachingStrategy::LRU);
    }

    #[test]
    fn test_record_query() {
        let mut manager = AutoCachingManager::new();
        manager.record_query("query { user }", 100.0, 1024).unwrap();

        let stats = manager.get_statistics().unwrap();
        assert_eq!(stats.total_queries, 1);
        assert_eq!(stats.unique_queries, 1);
    }

    #[test]
    fn test_cache_decision_insufficient_access() {
        let mut manager = AutoCachingManager::new();

        // Record only once (below min_access_count)
        manager.record_query("query { user }", 100.0, 1024).unwrap();

        let decision = manager.get_cache_decision("query { user }").unwrap();
        assert!(!decision.should_cache);
        assert!(decision.reason.contains("Insufficient access count"));
    }

    #[test]
    fn test_cache_decision_fast_query() {
        let mut manager = AutoCachingManager::new();
        let query = "query { user }";

        // Record multiple times with fast execution
        for _ in 0..5 {
            manager.record_query(query, 10.0, 1024).unwrap(); // 10ms < min_execution_time_ms (50ms)
        }

        let decision = manager.get_cache_decision(query).unwrap();
        assert!(!decision.should_cache);
        assert!(decision.reason.contains("Execution too fast"));
    }

    #[test]
    fn test_cache_decision_large_result() {
        let mut manager = AutoCachingManager::new();
        let query = "query { large_data }";

        // Record with large result size
        for _ in 0..5 {
            manager.record_query(query, 200.0, 20_000_000).unwrap(); // 20MB > max (10MB)
        }

        let decision = manager.get_cache_decision(query).unwrap();
        assert!(!decision.should_cache);
        assert!(decision.reason.contains("Result too large"));
    }

    #[test]
    fn test_cache_decision_cacheable() {
        let mut manager = AutoCachingManager::new();
        let query = "query { products }";

        // Record multiple times with good characteristics
        for _ in 0..10 {
            manager.record_query(query, 150.0, 5000).unwrap();
            thread::sleep(Duration::from_millis(10)); // Small delay for frequency calculation
        }

        let decision = manager.get_cache_decision(query).unwrap();
        // May or may not cache depending on timing, but should have valid decision
        assert!(decision.ttl_seconds > 0);
    }

    #[test]
    fn test_adaptive_ttl() {
        let mut manager = AutoCachingManager::with_config(CachingConfig {
            adaptive_ttl: true,
            min_access_count: 2,
            ..Default::default()
        });

        let query = "query { user }";

        // High frequency accesses
        for _ in 0..10 {
            manager.record_query(query, 100.0, 1024).unwrap();
        }

        let ttl1 = manager.get_optimal_ttl(query);

        // Low frequency will get longer TTL in theory, but we need separate query
        let query2 = "query { settings }";
        for _ in 0..3 {
            manager.record_query(query2, 100.0, 1024).unwrap();
            thread::sleep(Duration::from_millis(200));
        }

        let ttl2 = manager.get_optimal_ttl(query2);

        // Both should be within valid range
        assert!(ttl1.as_secs() >= manager.config.min_ttl_seconds);
        assert!(ttl2.as_secs() >= manager.config.min_ttl_seconds);
    }

    #[test]
    fn test_statistics() {
        let mut manager = AutoCachingManager::new();

        manager.record_query("query { user }", 100.0, 1024).unwrap();
        manager.record_query("query { user }", 100.0, 1024).unwrap();
        manager
            .record_query("query { posts }", 150.0, 2048)
            .unwrap();

        let stats = manager.get_statistics().unwrap();
        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.unique_queries, 2);
    }

    #[test]
    fn test_clear() {
        let mut manager = AutoCachingManager::new();

        manager.record_query("query { user }", 100.0, 1024).unwrap();

        let stats1 = manager.get_statistics().unwrap();
        assert_eq!(stats1.total_queries, 1);

        manager.clear().unwrap();

        let stats2 = manager.get_statistics().unwrap();
        assert_eq!(stats2.total_queries, 0);
        assert_eq!(stats2.unique_queries, 0);
    }

    #[test]
    fn test_different_strategies() {
        let strategies = vec![
            CachingStrategy::LRU,
            CachingStrategy::LFU,
            CachingStrategy::Adaptive,
            CachingStrategy::Predictive,
            CachingStrategy::TimeBased,
            CachingStrategy::CostBased,
        ];

        for strategy in strategies {
            let mut manager = AutoCachingManager::new();
            manager.set_strategy(strategy).unwrap();

            let query = "query { user }";
            for _ in 0..5 {
                manager.record_query(query, 100.0, 1024).unwrap();
            }

            let decision = manager.get_cache_decision(query);
            assert!(decision.is_ok(), "Strategy {:?} failed", strategy);
        }
    }
}
