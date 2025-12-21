//! Compaction strategies

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Compaction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionStrategy {
    /// Periodic compaction at fixed intervals
    Periodic,
    /// Threshold-based (compact when fragmentation exceeds threshold)
    ThresholdBased,
    /// Size-based (compact when wasted space exceeds threshold)
    SizeBased,
    /// Adaptive (automatically adjust based on workload)
    Adaptive,
    /// Manual (only compact when explicitly triggered)
    Manual,
}

impl Default for CompactionStrategy {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Strategy evaluator
pub struct StrategyEvaluator {
    strategy: CompactionStrategy,
    last_compaction: Option<SystemTime>,
}

impl StrategyEvaluator {
    /// Create a new strategy evaluator
    pub fn new(strategy: CompactionStrategy) -> Self {
        Self {
            strategy,
            last_compaction: None,
        }
    }

    /// Check if compaction should be triggered
    pub fn should_compact(
        &self,
        fragmentation: f64,
        wasted_bytes: u64,
        time_since_last: Option<Duration>,
        interval: Duration,
        fragmentation_threshold: f64,
        size_threshold_bytes: u64,
    ) -> bool {
        match self.strategy {
            CompactionStrategy::Periodic => {
                // Compact if interval has elapsed
                time_since_last.map(|d| d >= interval).unwrap_or(true)
            }
            CompactionStrategy::ThresholdBased => {
                // Compact if fragmentation exceeds threshold
                fragmentation >= fragmentation_threshold
            }
            CompactionStrategy::SizeBased => {
                // Compact if wasted space exceeds threshold
                wasted_bytes >= size_threshold_bytes
            }
            CompactionStrategy::Adaptive => {
                // Combine multiple factors
                self.evaluate_adaptive(
                    fragmentation,
                    wasted_bytes,
                    time_since_last,
                    interval,
                    fragmentation_threshold,
                    size_threshold_bytes,
                )
            }
            CompactionStrategy::Manual => {
                // Never automatically compact
                false
            }
        }
    }

    /// Adaptive evaluation combining multiple factors
    fn evaluate_adaptive(
        &self,
        fragmentation: f64,
        wasted_bytes: u64,
        time_since_last: Option<Duration>,
        interval: Duration,
        fragmentation_threshold: f64,
        size_threshold_bytes: u64,
    ) -> bool {
        let mut score = 0.0;

        // Factor 1: Fragmentation (40% weight)
        if fragmentation > 0.0 {
            let frag_ratio = fragmentation / fragmentation_threshold;
            score += frag_ratio.min(1.0) * 0.4;
        }

        // Factor 2: Wasted space (30% weight)
        if wasted_bytes > 0 {
            let size_ratio = wasted_bytes as f64 / size_threshold_bytes as f64;
            score += size_ratio.min(1.0) * 0.3;
        }

        // Factor 3: Time since last compaction (30% weight)
        if let Some(time_elapsed) = time_since_last {
            let time_ratio = time_elapsed.as_secs_f64() / interval.as_secs_f64();
            score += time_ratio.min(1.0) * 0.3;
        } else {
            // Never compacted, give full weight
            score += 0.3;
        }

        // Trigger if score exceeds threshold
        score >= 0.7
    }

    /// Update last compaction time
    pub fn record_compaction(&mut self) {
        self.last_compaction = Some(SystemTime::now());
    }

    /// Get time since last compaction
    pub fn time_since_last_compaction(&self) -> Option<Duration> {
        self.last_compaction
            .and_then(|t| SystemTime::now().duration_since(t).ok())
    }

    /// Calculate priority for compaction (0.0 - 1.0, higher = more urgent)
    pub fn calculate_priority(
        &self,
        fragmentation: f64,
        wasted_bytes: u64,
        time_since_last: Option<Duration>,
    ) -> f64 {
        match self.strategy {
            CompactionStrategy::Periodic => {
                // Priority based on time
                time_since_last
                    .map(|d| (d.as_secs() as f64 / 3600.0).min(1.0))
                    .unwrap_or(1.0)
            }
            CompactionStrategy::ThresholdBased => {
                // Priority based on fragmentation
                fragmentation
            }
            CompactionStrategy::SizeBased => {
                // Priority based on wasted space
                let size_gb = wasted_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                (size_gb / 10.0).min(1.0) // Normalize to 10GB max
            }
            CompactionStrategy::Adaptive => {
                // Weighted combination
                let frag_priority = fragmentation * 0.4;
                let size_priority =
                    (wasted_bytes as f64 / (1024.0 * 1024.0 * 1024.0) / 10.0).min(1.0) * 0.3;
                let time_priority = time_since_last
                    .map(|d| (d.as_secs() as f64 / 3600.0).min(1.0) * 0.3)
                    .unwrap_or(0.3);
                (frag_priority + size_priority + time_priority).min(1.0)
            }
            CompactionStrategy::Manual => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_periodic_strategy() {
        let evaluator = StrategyEvaluator::new(CompactionStrategy::Periodic);

        // Should compact if enough time has passed
        assert!(evaluator.should_compact(
            0.1,
            0,
            Some(Duration::from_secs(7200)),
            Duration::from_secs(3600),
            0.3,
            100_000_000,
        ));

        // Should not compact if not enough time
        assert!(!evaluator.should_compact(
            0.5,
            1_000_000_000,
            Some(Duration::from_secs(1800)),
            Duration::from_secs(3600),
            0.3,
            100_000_000,
        ));
    }

    #[test]
    fn test_threshold_based_strategy() {
        let evaluator = StrategyEvaluator::new(CompactionStrategy::ThresholdBased);

        // Should compact if fragmentation exceeds threshold
        assert!(evaluator.should_compact(
            0.5,
            0,
            None,
            Duration::from_secs(3600),
            0.3,
            100_000_000,
        ));

        // Should not compact if below threshold
        assert!(!evaluator.should_compact(
            0.2,
            1_000_000_000,
            None,
            Duration::from_secs(3600),
            0.3,
            100_000_000,
        ));
    }

    #[test]
    fn test_size_based_strategy() {
        let evaluator = StrategyEvaluator::new(CompactionStrategy::SizeBased);

        // Should compact if wasted space exceeds threshold
        assert!(evaluator.should_compact(
            0.1,
            200_000_000,
            None,
            Duration::from_secs(3600),
            0.3,
            100_000_000,
        ));

        // Should not compact if below threshold
        assert!(!evaluator.should_compact(
            0.5,
            50_000_000,
            None,
            Duration::from_secs(3600),
            0.3,
            100_000_000,
        ));
    }

    #[test]
    fn test_adaptive_strategy() {
        let evaluator = StrategyEvaluator::new(CompactionStrategy::Adaptive);

        // Should compact with high fragmentation and time
        assert!(evaluator.should_compact(
            0.4,
            150_000_000,
            Some(Duration::from_secs(7200)),
            Duration::from_secs(3600),
            0.3,
            100_000_000,
        ));

        // Should not compact with low values
        assert!(!evaluator.should_compact(
            0.05,
            10_000_000,
            Some(Duration::from_secs(300)),
            Duration::from_secs(3600),
            0.3,
            100_000_000,
        ));
    }

    #[test]
    fn test_manual_strategy() {
        let evaluator = StrategyEvaluator::new(CompactionStrategy::Manual);

        // Should never auto-compact
        assert!(!evaluator.should_compact(
            0.9,
            10_000_000_000,
            Some(Duration::from_secs(100000)),
            Duration::from_secs(3600),
            0.3,
            100_000_000,
        ));
    }

    #[test]
    fn test_priority_calculation() {
        let evaluator = StrategyEvaluator::new(CompactionStrategy::Adaptive);

        let priority =
            evaluator.calculate_priority(0.5, 500_000_000, Some(Duration::from_secs(7200)));

        assert!(priority > 0.0 && priority <= 1.0);
    }
}
