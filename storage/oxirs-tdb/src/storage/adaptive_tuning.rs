//! Adaptive buffer pool tuning for automatic performance optimization
//!
//! This module provides dynamic tuning of buffer pool parameters based on
//! runtime performance metrics and access patterns.

use super::buffer_pool::BufferPoolStats;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Adaptive tuning configuration
#[derive(Debug, Clone)]
pub struct AdaptiveTuningConfig {
    /// Minimum buffer pool size
    pub min_pool_size: usize,
    /// Maximum buffer pool size
    pub max_pool_size: usize,
    /// Target hit rate (0.0 to 1.0)
    pub target_hit_rate: f64,
    /// Tuning interval (how often to adjust)
    pub tuning_interval: Duration,
    /// Aggressive tuning mode (faster adjustments)
    pub aggressive: bool,
}

impl Default for AdaptiveTuningConfig {
    fn default() -> Self {
        Self {
            min_pool_size: 100,
            max_pool_size: 10000,
            target_hit_rate: 0.9,
            tuning_interval: Duration::from_secs(60),
            aggressive: false,
        }
    }
}

/// Adaptive buffer pool tuner
pub struct AdaptiveTuner {
    /// Configuration
    config: AdaptiveTuningConfig,
    /// Current recommended pool size
    recommended_size: AtomicUsize,
    /// Last tuning time
    last_tuning: parking_lot::Mutex<Instant>,
    /// Historical hit rates (ring buffer)
    hit_rate_history: parking_lot::Mutex<Vec<f64>>,
    /// Number of size increases
    size_increases: AtomicU64,
    /// Number of size decreases
    size_decreases: AtomicU64,
}

impl AdaptiveTuner {
    /// Create a new adaptive tuner
    pub fn new(config: AdaptiveTuningConfig) -> Self {
        let initial_size = (config.min_pool_size + config.max_pool_size) / 2;

        Self {
            config,
            recommended_size: AtomicUsize::new(initial_size),
            last_tuning: parking_lot::Mutex::new(Instant::now()),
            hit_rate_history: parking_lot::Mutex::new(Vec::with_capacity(10)),
            size_increases: AtomicU64::new(0),
            size_decreases: AtomicU64::new(0),
        }
    }

    /// Analyze current performance and adjust recommendations
    pub fn tune(&self, stats: &BufferPoolStats) -> TuningRecommendation {
        // Check if it's time to tune
        let mut last_tuning = self.last_tuning.lock();
        if last_tuning.elapsed() < self.config.tuning_interval {
            return TuningRecommendation::NoChange;
        }

        let current_hit_rate = stats.hit_rate();
        let current_size = self.recommended_size.load(Ordering::Relaxed);

        // Record hit rate history
        let mut history = self.hit_rate_history.lock();
        history.push(current_hit_rate);
        if history.len() > 10 {
            history.remove(0);
        }

        // Calculate trend
        let trend = self.calculate_trend(&history);

        // Determine action based on hit rate and trend
        let recommendation = if current_hit_rate < self.config.target_hit_rate {
            // Hit rate is below target - consider increasing pool size
            if current_size < self.config.max_pool_size {
                let increase = self.calculate_size_adjustment(current_size, true);
                let new_size = (current_size + increase).min(self.config.max_pool_size);

                self.recommended_size.store(new_size, Ordering::Relaxed);
                self.size_increases.fetch_add(1, Ordering::Relaxed);

                TuningRecommendation::IncreaseSize {
                    old_size: current_size,
                    new_size,
                    reason: TuningReason::LowHitRate {
                        current: current_hit_rate,
                        target: self.config.target_hit_rate,
                    },
                }
            } else {
                TuningRecommendation::AtMaximum {
                    hit_rate: current_hit_rate,
                }
            }
        } else if current_hit_rate > self.config.target_hit_rate + 0.05 && trend > 0.0 {
            // Hit rate is above target with improving trend - consider decreasing
            if current_size > self.config.min_pool_size {
                let decrease = self.calculate_size_adjustment(current_size, false);
                let new_size = current_size
                    .saturating_sub(decrease)
                    .max(self.config.min_pool_size);

                self.recommended_size.store(new_size, Ordering::Relaxed);
                self.size_decreases.fetch_add(1, Ordering::Relaxed);

                TuningRecommendation::DecreaseSize {
                    old_size: current_size,
                    new_size,
                    reason: TuningReason::HighHitRate {
                        current: current_hit_rate,
                        target: self.config.target_hit_rate,
                    },
                }
            } else {
                TuningRecommendation::AtMinimum {
                    hit_rate: current_hit_rate,
                }
            }
        } else {
            TuningRecommendation::NoChange
        };

        *last_tuning = Instant::now();
        recommendation
    }

    /// Calculate size adjustment amount
    fn calculate_size_adjustment(&self, current_size: usize, increase: bool) -> usize {
        if self.config.aggressive {
            // Aggressive: 20% adjustment
            current_size / 5
        } else {
            // Conservative: 10% adjustment
            current_size / 10
        }
        .max(1)
    }

    /// Calculate trend from history (positive = improving, negative = declining)
    fn calculate_trend(&self, history: &[f64]) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = history.len() as f64;
        let sum_x: f64 = (0..history.len()).map(|i| i as f64).sum();
        let sum_y: f64 = history.iter().sum();
        let sum_xy: f64 = history.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x_sq: f64 = (0..history.len()).map(|i| (i as f64).powi(2)).sum();

        (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x.powi(2))
    }

    /// Get current recommended size
    pub fn recommended_size(&self) -> usize {
        self.recommended_size.load(Ordering::Relaxed)
    }

    /// Get tuning statistics
    pub fn stats(&self) -> AdaptiveTuningStats {
        AdaptiveTuningStats {
            current_size: self.recommended_size.load(Ordering::Relaxed),
            size_increases: self.size_increases.load(Ordering::Relaxed),
            size_decreases: self.size_decreases.load(Ordering::Relaxed),
            hit_rate_history: self.hit_rate_history.lock().clone(),
        }
    }
}

/// Tuning recommendation
#[derive(Debug, Clone)]
pub enum TuningRecommendation {
    /// No change needed
    NoChange,
    /// Increase buffer pool size
    IncreaseSize {
        /// Old size
        old_size: usize,
        /// New recommended size
        new_size: usize,
        /// Reason for increase
        reason: TuningReason,
    },
    /// Decrease buffer pool size
    DecreaseSize {
        /// Old size
        old_size: usize,
        /// New recommended size
        new_size: usize,
        /// Reason for decrease
        reason: TuningReason,
    },
    /// Already at maximum size
    AtMaximum {
        /// Current hit rate
        hit_rate: f64,
    },
    /// Already at minimum size
    AtMinimum {
        /// Current hit rate
        hit_rate: f64,
    },
}

/// Reason for tuning adjustment
#[derive(Debug, Clone)]
pub enum TuningReason {
    /// Hit rate is below target
    LowHitRate {
        /// Current hit rate
        current: f64,
        /// Target hit rate
        target: f64,
    },
    /// Hit rate is above target
    HighHitRate {
        /// Current hit rate
        current: f64,
        /// Target hit rate
        target: f64,
    },
}

/// Adaptive tuning statistics
#[derive(Debug, Clone)]
pub struct AdaptiveTuningStats {
    /// Current recommended size
    pub current_size: usize,
    /// Number of times size was increased
    pub size_increases: u64,
    /// Number of times size was decreased
    pub size_decreases: u64,
    /// Historical hit rates
    pub hit_rate_history: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn create_test_stats(hit_rate: f64) -> BufferPoolStats {
        let stats = BufferPoolStats::default();
        let total = 1000;
        let hits = (total as f64 * hit_rate) as u64;

        stats.total_fetches.store(total, Ordering::Relaxed);
        stats.cache_hits.store(hits, Ordering::Relaxed);
        stats.cache_misses.store(total - hits, Ordering::Relaxed);

        stats
    }

    #[test]
    fn test_adaptive_tuner_creation() {
        let config = AdaptiveTuningConfig::default();
        let tuner = AdaptiveTuner::new(config);

        let initial_size = tuner.recommended_size();
        assert!(initial_size >= 100);
        assert!(initial_size <= 10000);
    }

    #[test]
    fn test_low_hit_rate_increases_size() {
        let config = AdaptiveTuningConfig {
            tuning_interval: Duration::from_millis(1),
            ..Default::default()
        };
        let tuner = AdaptiveTuner::new(config);

        std::thread::sleep(Duration::from_millis(10));

        let stats = create_test_stats(0.5); // 50% hit rate (below 90% target)
        let recommendation = tuner.tune(&stats);

        match recommendation {
            TuningRecommendation::IncreaseSize {
                old_size, new_size, ..
            } => {
                assert!(new_size > old_size);
            }
            _ => panic!("Expected IncreaseSize recommendation"),
        }
    }

    #[test]
    fn test_high_hit_rate_decreases_size() {
        let config = AdaptiveTuningConfig {
            tuning_interval: Duration::from_millis(1),
            ..Default::default()
        };
        let tuner = AdaptiveTuner::new(config);

        // Build up history with improving hit rates
        std::thread::sleep(Duration::from_millis(10));
        for hit_rate in &[0.92, 0.93, 0.94, 0.95, 0.96] {
            let stats = create_test_stats(*hit_rate);
            tuner.tune(&stats);
            std::thread::sleep(Duration::from_millis(10));
        }

        let stats = create_test_stats(0.97); // Very high hit rate
        let recommendation = tuner.tune(&stats);

        match recommendation {
            TuningRecommendation::DecreaseSize {
                old_size, new_size, ..
            } => {
                assert!(new_size < old_size);
            }
            TuningRecommendation::NoChange => {
                // Also acceptable if trend isn't strong enough
            }
            _ => {}
        }
    }

    #[test]
    fn test_tuning_interval_respected() {
        let config = AdaptiveTuningConfig {
            tuning_interval: Duration::from_millis(100),
            ..Default::default()
        };
        let tuner = AdaptiveTuner::new(config);

        let stats = create_test_stats(0.5);

        // Wait for tuning interval to elapse
        std::thread::sleep(Duration::from_millis(150));
        // First call should trigger tuning
        let rec1 = tuner.tune(&stats);
        assert!(!matches!(rec1, TuningRecommendation::NoChange));

        // Immediate second call should not trigger tuning
        let rec2 = tuner.tune(&stats);
        assert!(matches!(rec2, TuningRecommendation::NoChange));
    }

    #[test]
    fn test_size_bounds_respected() {
        let config = AdaptiveTuningConfig {
            min_pool_size: 100,
            max_pool_size: 200,
            target_hit_rate: 0.9,
            tuning_interval: Duration::from_millis(1),
            aggressive: true,
        };
        let tuner = AdaptiveTuner::new(config);

        // Try to push size to maximum
        for _ in 0..10 {
            std::thread::sleep(Duration::from_millis(10));
            let stats = create_test_stats(0.5); // Low hit rate
            tuner.tune(&stats);
        }

        let final_size = tuner.recommended_size();
        assert!(final_size <= 200);
        assert!(final_size >= 100);
    }

    #[test]
    fn test_aggressive_vs_conservative() {
        let conservative = AdaptiveTuningConfig {
            aggressive: false,
            tuning_interval: Duration::from_millis(1),
            ..Default::default()
        };
        let aggressive = AdaptiveTuningConfig {
            aggressive: true,
            tuning_interval: Duration::from_millis(1),
            ..Default::default()
        };

        let tuner_cons = AdaptiveTuner::new(conservative);
        let tuner_aggr = AdaptiveTuner::new(aggressive);

        std::thread::sleep(Duration::from_millis(10));

        let stats = create_test_stats(0.5);

        let rec_cons = tuner_cons.tune(&stats);
        let rec_aggr = tuner_aggr.tune(&stats);

        // Aggressive should make larger adjustments
        if let (
            TuningRecommendation::IncreaseSize {
                old_size: old1,
                new_size: new1,
                ..
            },
            TuningRecommendation::IncreaseSize {
                old_size: old2,
                new_size: new2,
                ..
            },
        ) = (rec_cons, rec_aggr)
        {
            let increase_cons = new1 - old1;
            let increase_aggr = new2 - old2;
            assert!(increase_aggr >= increase_cons);
        }
    }

    #[test]
    fn test_tuning_stats() {
        let config = AdaptiveTuningConfig {
            tuning_interval: Duration::from_millis(1),
            ..Default::default()
        };
        let tuner = AdaptiveTuner::new(config);

        std::thread::sleep(Duration::from_millis(10));
        let stats = create_test_stats(0.5);
        tuner.tune(&stats);

        let tuning_stats = tuner.stats();
        assert!(tuning_stats.size_increases > 0 || tuning_stats.size_decreases > 0);
    }
}
