//! # Buffer Pool Tuner
//!
//! Provides adaptive tuning of buffer pool parameters based on access patterns
//! and performance metrics. Uses scirs2-core metrics API for production-grade monitoring.

use crate::error::TdbError;
use crate::storage::{BufferPool, BufferPoolStats};
use scirs2_core::metrics::{Counter, Gauge, Histogram, Timer};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for buffer pool tuner
#[derive(Debug, Clone)]
pub struct BufferPoolTunerConfig {
    /// Minimum buffer pool size
    pub min_size: usize,
    /// Maximum buffer pool size
    pub max_size: usize,
    /// Target hit rate (0.0 - 1.0)
    pub target_hit_rate: f64,
    /// Tuning interval in seconds
    pub tuning_interval_secs: u64,
    /// Adjustment step size (percentage of current size)
    pub adjustment_step: f64,
    /// Enable aggressive tuning
    pub aggressive_tuning: bool,
}

impl Default for BufferPoolTunerConfig {
    fn default() -> Self {
        Self {
            min_size: 64,         // 64 pages minimum (256 KB)
            max_size: 65536,      // 65536 pages maximum (256 MB)
            target_hit_rate: 0.9, // 90% target hit rate
            tuning_interval_secs: 60,
            adjustment_step: 0.1, // 10% adjustment steps
            aggressive_tuning: false,
        }
    }
}

/// Access pattern classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential access pattern (high locality)
    Sequential,
    /// Random access pattern (low locality)
    Random,
    /// Mixed access pattern
    Mixed,
    /// Scan-heavy workload
    ScanHeavy,
    /// Unknown pattern
    Unknown,
}

/// Eviction policy for buffer pool
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// Least Recently Used
    #[default]
    LRU,
    /// Least Frequently Used
    LFU,
    /// Adaptive Replacement Cache
    ARC,
    /// Clock/Second Chance
    Clock,
}

/// Tuning recommendation
#[derive(Debug, Clone)]
pub struct TuningRecommendation {
    /// Recommended buffer pool size
    pub recommended_size: usize,
    /// Recommended eviction policy
    pub recommended_policy: EvictionPolicy,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Reasoning behind the recommendation
    pub reason: String,
}

/// Performance report from tuner
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Current buffer pool size
    pub current_size: usize,
    /// Current hit rate
    pub hit_rate: f64,
    /// Average access latency
    pub avg_latency_us: f64,
    /// Detected access pattern
    pub access_pattern: AccessPattern,
    /// Number of tuning iterations
    pub tuning_iterations: u64,
    /// Total bytes saved through optimization
    pub bytes_saved: u64,
}

/// Buffer pool tuner with adaptive optimization
pub struct BufferPoolTuner {
    /// Configuration
    config: BufferPoolTunerConfig,
    /// Last tuning timestamp
    last_tuning: Instant,
    /// Access pattern history
    access_history: Vec<AccessPattern>,
    /// Tuning iterations counter
    tuning_iterations: u64,

    // Metrics (scirs2-core v0.1.0+ API)
    /// Hit rate gauge
    hit_rate_gauge: Gauge,
    /// Miss rate gauge
    miss_rate_gauge: Gauge,
    /// Pool size gauge
    pool_size_gauge: Gauge,
    /// Eviction counter
    eviction_counter: Counter,
    /// Access counter
    access_counter: Counter,
    /// Access latency histogram
    access_latency_histogram: Histogram,
    /// Tuning adjustments counter
    tuning_adjustments: Counter,
}

impl BufferPoolTuner {
    /// Create a new buffer pool tuner
    pub fn new(config: BufferPoolTunerConfig) -> Self {
        Self {
            config,
            last_tuning: Instant::now(),
            access_history: Vec::new(),
            tuning_iterations: 0,

            // Initialize metrics
            hit_rate_gauge: Gauge::new("buffer_pool_hit_rate".to_string()),
            miss_rate_gauge: Gauge::new("buffer_pool_miss_rate".to_string()),
            pool_size_gauge: Gauge::new("buffer_pool_size".to_string()),
            eviction_counter: Counter::new("buffer_pool_evictions".to_string()),
            access_counter: Counter::new("buffer_pool_accesses".to_string()),
            access_latency_histogram: Histogram::new("buffer_pool_access_latency".to_string()),
            tuning_adjustments: Counter::new("buffer_pool_tuning_adjustments".to_string()),
        }
    }

    /// Update metrics from buffer pool stats
    pub fn update_metrics(
        &mut self,
        stats: &BufferPoolStats,
        pool_size: usize,
        cached_pages: usize,
    ) {
        use std::sync::atomic::Ordering;

        let total_fetches = stats.total_fetches.load(Ordering::Relaxed);
        let cache_hits = stats.cache_hits.load(Ordering::Relaxed);
        let cache_misses = stats.cache_misses.load(Ordering::Relaxed);
        let evictions = stats.evictions.load(Ordering::Relaxed);

        if total_fetches > 0 {
            let hit_rate = cache_hits as f64 / total_fetches as f64;
            let miss_rate = cache_misses as f64 / total_fetches as f64;

            self.hit_rate_gauge.set(hit_rate);
            self.miss_rate_gauge.set(miss_rate);
        }

        self.pool_size_gauge.set(pool_size as f64);
        self.eviction_counter.add(evictions);
        self.access_counter.add(total_fetches);
    }

    /// Record an access latency
    pub fn record_access_latency(&self, latency: Duration) {
        self.access_latency_histogram
            .observe(latency.as_secs_f64() * 1_000_000.0); // Convert to microseconds
    }

    /// Detect access pattern from buffer pool statistics
    pub fn detect_access_pattern(&mut self, stats: &BufferPoolStats) -> AccessPattern {
        use std::sync::atomic::Ordering;

        let total_fetches = stats.total_fetches.load(Ordering::Relaxed);
        let cache_hits = stats.cache_hits.load(Ordering::Relaxed);
        let evictions = stats.evictions.load(Ordering::Relaxed);

        if total_fetches == 0 {
            return AccessPattern::Unknown;
        }

        let hit_rate = cache_hits as f64 / total_fetches as f64;

        // Simple heuristic-based pattern detection
        let pattern = if hit_rate > 0.95 {
            AccessPattern::Sequential // Very high hit rate suggests sequential access
        } else if hit_rate < 0.5 {
            AccessPattern::Random // Low hit rate suggests random access
        } else if evictions > total_fetches / 2 {
            AccessPattern::ScanHeavy // Many evictions suggest scan operations
        } else {
            AccessPattern::Mixed
        };

        // Update access history (keep last 10 patterns)
        self.access_history.push(pattern);
        if self.access_history.len() > 10 {
            self.access_history.remove(0);
        }

        pattern
    }

    /// Generate tuning recommendation
    pub fn recommend_tuning(
        &mut self,
        stats: &BufferPoolStats,
        current_pool_size: usize,
    ) -> Option<TuningRecommendation> {
        use std::sync::atomic::Ordering;

        let total_fetches = stats.total_fetches.load(Ordering::Relaxed);
        let cache_hits = stats.cache_hits.load(Ordering::Relaxed);

        if total_fetches == 0 {
            return None;
        }

        let current_hit_rate = cache_hits as f64 / total_fetches as f64;
        let pattern = self.detect_access_pattern(stats);

        // Determine if tuning is needed
        if current_hit_rate >= self.config.target_hit_rate {
            // Hit rate is good, no tuning needed
            return None;
        }

        // Calculate recommended size adjustment
        let hit_rate_deficit = self.config.target_hit_rate - current_hit_rate;
        let adjustment_multiplier = if self.config.aggressive_tuning {
            1.5
        } else {
            1.0
        };

        let size_increase_factor =
            1.0 + (hit_rate_deficit * self.config.adjustment_step * adjustment_multiplier).min(0.5);

        let recommended_size = ((current_pool_size as f64 * size_increase_factor) as usize)
            .clamp(self.config.min_size, self.config.max_size);

        // Recommend eviction policy based on access pattern
        let recommended_policy = match pattern {
            AccessPattern::Sequential => EvictionPolicy::LRU,
            AccessPattern::Random => EvictionPolicy::ARC,
            AccessPattern::ScanHeavy => EvictionPolicy::Clock,
            AccessPattern::Mixed => EvictionPolicy::LFU,
            AccessPattern::Unknown => EvictionPolicy::LRU,
        };

        let confidence = if self.access_history.len() >= 5 {
            0.8
        } else {
            0.5
        };

        Some(TuningRecommendation {
            recommended_size,
            recommended_policy,
            confidence,
            reason: format!(
                "Hit rate {:.2}% is below target {:.2}%. Detected {:?} access pattern.",
                current_hit_rate * 100.0,
                self.config.target_hit_rate * 100.0,
                pattern
            ),
        })
    }

    /// Apply tuning recommendation (returns true if applied)
    pub fn apply_tuning(&mut self, recommendation: &TuningRecommendation) -> bool {
        // Check if enough time has elapsed since last tuning
        let elapsed = self.last_tuning.elapsed();
        if elapsed < Duration::from_secs(self.config.tuning_interval_secs) {
            return false;
        }

        // Only apply high-confidence recommendations
        if recommendation.confidence < 0.6 {
            return false;
        }

        self.last_tuning = Instant::now();
        self.tuning_iterations += 1;
        self.tuning_adjustments.inc();

        true
    }

    /// Generate performance report
    pub fn generate_report(
        &self,
        stats: &BufferPoolStats,
        current_pool_size: usize,
    ) -> PerformanceReport {
        use std::sync::atomic::Ordering;

        let total_fetches = stats.total_fetches.load(Ordering::Relaxed);
        let cache_hits = stats.cache_hits.load(Ordering::Relaxed);

        let hit_rate = if total_fetches > 0 {
            cache_hits as f64 / total_fetches as f64
        } else {
            0.0
        };

        let latency_stats = self.access_latency_histogram.get_stats();
        let avg_latency_us = latency_stats.mean;

        let access_pattern = if let Some(&last_pattern) = self.access_history.last() {
            last_pattern
        } else {
            AccessPattern::Unknown
        };

        PerformanceReport {
            current_size: current_pool_size,
            hit_rate,
            avg_latency_us,
            access_pattern,
            tuning_iterations: self.tuning_iterations,
            bytes_saved: 0, // Could be calculated based on tuning history
        }
    }

    /// Get current hit rate from metrics
    pub fn get_hit_rate(&self) -> f64 {
        self.hit_rate_gauge.get()
    }

    /// Get total accesses from metrics
    pub fn get_total_accesses(&self) -> u64 {
        self.access_counter.get()
    }

    /// Get total evictions from metrics
    pub fn get_total_evictions(&self) -> u64 {
        self.eviction_counter.get()
    }

    /// Get tuning iterations count
    pub fn get_tuning_iterations(&self) -> u64 {
        self.tuning_iterations
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        // Note: scirs2-core metrics don't have a reset method, so we recreate them
        self.hit_rate_gauge = Gauge::new("buffer_pool_hit_rate".to_string());
        self.miss_rate_gauge = Gauge::new("buffer_pool_miss_rate".to_string());
        self.pool_size_gauge = Gauge::new("buffer_pool_size".to_string());
        self.eviction_counter = Counter::new("buffer_pool_evictions".to_string());
        self.access_counter = Counter::new("buffer_pool_accesses".to_string());
        self.access_latency_histogram = Histogram::new("buffer_pool_access_latency".to_string());
        self.tuning_adjustments = Counter::new("buffer_pool_tuning_adjustments".to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_tuner_creation() {
        let config = BufferPoolTunerConfig::default();
        let tuner = BufferPoolTuner::new(config.clone());

        assert_eq!(tuner.get_tuning_iterations(), 0);
        assert_eq!(tuner.get_total_accesses(), 0);
        assert_eq!(tuner.get_total_evictions(), 0);
    }

    #[test]
    fn test_update_metrics() {
        use std::sync::atomic::{AtomicU64, Ordering};

        let config = BufferPoolTunerConfig::default();
        let mut tuner = BufferPoolTuner::new(config);

        let stats = BufferPoolStats {
            total_fetches: AtomicU64::new(1000),
            cache_hits: AtomicU64::new(900),
            cache_misses: AtomicU64::new(100),
            evictions: AtomicU64::new(50),
            writes: AtomicU64::new(10),
        };

        tuner.update_metrics(&stats, 1024, 500);

        let hit_rate = tuner.get_hit_rate();
        assert!((hit_rate - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_access_pattern_detection() {
        use std::sync::atomic::AtomicU64;

        let config = BufferPoolTunerConfig::default();
        let mut tuner = BufferPoolTuner::new(config);

        // High hit rate -> Sequential
        let sequential_stats = BufferPoolStats {
            total_fetches: AtomicU64::new(1000),
            cache_hits: AtomicU64::new(980),
            cache_misses: AtomicU64::new(20),
            evictions: AtomicU64::new(5),
            writes: AtomicU64::new(10),
        };
        let pattern = tuner.detect_access_pattern(&sequential_stats);
        assert_eq!(pattern, AccessPattern::Sequential);

        // Low hit rate -> Random
        let random_stats = BufferPoolStats {
            total_fetches: AtomicU64::new(1000),
            cache_hits: AtomicU64::new(300),
            cache_misses: AtomicU64::new(700),
            evictions: AtomicU64::new(200),
            writes: AtomicU64::new(50),
        };
        let pattern = tuner.detect_access_pattern(&random_stats);
        assert_eq!(pattern, AccessPattern::Random);
    }

    #[test]
    fn test_tuning_recommendation() {
        use std::sync::atomic::AtomicU64;

        let config = BufferPoolTunerConfig {
            target_hit_rate: 0.9,
            ..Default::default()
        };
        let mut tuner = BufferPoolTuner::new(config);

        // Low hit rate should trigger recommendation
        let stats = BufferPoolStats {
            total_fetches: AtomicU64::new(1000),
            cache_hits: AtomicU64::new(600),
            cache_misses: AtomicU64::new(400),
            evictions: AtomicU64::new(100),
            writes: AtomicU64::new(50),
        };

        let current_pool_size = 1024;
        let recommendation = tuner.recommend_tuning(&stats, current_pool_size);
        assert!(recommendation.is_some());

        let rec = recommendation.unwrap();
        assert!(rec.recommended_size > current_pool_size);
        assert!(rec.confidence > 0.0);
    }

    #[test]
    fn test_no_tuning_when_hit_rate_good() {
        use std::sync::atomic::AtomicU64;

        let config = BufferPoolTunerConfig {
            target_hit_rate: 0.9,
            ..Default::default()
        };
        let mut tuner = BufferPoolTuner::new(config);

        // Good hit rate should not trigger recommendation
        let stats = BufferPoolStats {
            total_fetches: AtomicU64::new(1000),
            cache_hits: AtomicU64::new(950),
            cache_misses: AtomicU64::new(50),
            evictions: AtomicU64::new(10),
            writes: AtomicU64::new(5),
        };

        let recommendation = tuner.recommend_tuning(&stats, 1024);
        assert!(recommendation.is_none());
    }

    #[test]
    fn test_performance_report() {
        use std::sync::atomic::AtomicU64;

        let config = BufferPoolTunerConfig::default();
        let mut tuner = BufferPoolTuner::new(config);

        let stats = BufferPoolStats {
            total_fetches: AtomicU64::new(1000),
            cache_hits: AtomicU64::new(800),
            cache_misses: AtomicU64::new(200),
            evictions: AtomicU64::new(50),
            writes: AtomicU64::new(25),
        };

        tuner.update_metrics(&stats, 1024, 500);
        let report = tuner.generate_report(&stats, 1024);

        assert_eq!(report.current_size, 1024);
        assert!((report.hit_rate - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_access_latency_recording() {
        let config = BufferPoolTunerConfig::default();
        let tuner = BufferPoolTuner::new(config);

        tuner.record_access_latency(Duration::from_micros(100));
        tuner.record_access_latency(Duration::from_micros(200));
        tuner.record_access_latency(Duration::from_micros(150));

        let stats = tuner.access_latency_histogram.get_stats();
        assert_eq!(stats.count, 3);
        assert!(stats.mean > 0.0);
    }

    #[test]
    fn test_aggressive_tuning() {
        use std::sync::atomic::AtomicU64;

        let config = BufferPoolTunerConfig {
            target_hit_rate: 0.9,
            aggressive_tuning: true,
            ..Default::default()
        };
        let mut tuner_aggressive = BufferPoolTuner::new(config.clone());

        let config_normal = BufferPoolTunerConfig {
            target_hit_rate: 0.9,
            aggressive_tuning: false,
            ..Default::default()
        };
        let mut tuner_normal = BufferPoolTuner::new(config_normal);

        let stats = BufferPoolStats {
            total_fetches: AtomicU64::new(1000),
            cache_hits: AtomicU64::new(600),
            cache_misses: AtomicU64::new(400),
            evictions: AtomicU64::new(100),
            writes: AtomicU64::new(50),
        };

        let current_pool_size = 1000;
        let rec_aggressive = tuner_aggressive
            .recommend_tuning(&stats, current_pool_size)
            .unwrap();
        let rec_normal = tuner_normal
            .recommend_tuning(&stats, current_pool_size)
            .unwrap();

        // Aggressive should recommend larger increase
        assert!(rec_aggressive.recommended_size >= rec_normal.recommended_size);
    }

    #[test]
    fn test_eviction_policy_recommendation() {
        use std::sync::atomic::AtomicU64;

        let config = BufferPoolTunerConfig::default();
        let mut tuner = BufferPoolTuner::new(config);

        // Sequential pattern should recommend LRU
        let sequential_stats = BufferPoolStats {
            total_fetches: AtomicU64::new(1000),
            cache_hits: AtomicU64::new(980),
            cache_misses: AtomicU64::new(20),
            evictions: AtomicU64::new(5),
            writes: AtomicU64::new(3),
        };
        tuner.detect_access_pattern(&sequential_stats);
        if let Some(rec) = tuner.recommend_tuning(&sequential_stats, 500) {
            assert_eq!(rec.recommended_policy, EvictionPolicy::LRU);
        }

        // Random pattern should recommend ARC
        let random_stats = BufferPoolStats {
            total_fetches: AtomicU64::new(1000),
            cache_hits: AtomicU64::new(300),
            cache_misses: AtomicU64::new(700),
            evictions: AtomicU64::new(200),
            writes: AtomicU64::new(100),
        };
        tuner.detect_access_pattern(&random_stats);
        if let Some(rec) = tuner.recommend_tuning(&random_stats, 500) {
            assert_eq!(rec.recommended_policy, EvictionPolicy::ARC);
        }
    }

    #[test]
    fn test_reset_metrics() {
        use std::sync::atomic::AtomicU64;

        let config = BufferPoolTunerConfig::default();
        let mut tuner = BufferPoolTuner::new(config);

        let stats = BufferPoolStats {
            total_fetches: AtomicU64::new(1000),
            cache_hits: AtomicU64::new(900),
            cache_misses: AtomicU64::new(100),
            evictions: AtomicU64::new(50),
            writes: AtomicU64::new(25),
        };

        tuner.update_metrics(&stats, 1024, 500);
        assert!(tuner.get_hit_rate() > 0.0);

        tuner.reset_metrics();
        assert_eq!(tuner.get_hit_rate(), 0.0);
        assert_eq!(tuner.get_total_accesses(), 0);
    }
}
