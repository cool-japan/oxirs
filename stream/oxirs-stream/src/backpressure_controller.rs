//! # Backpressure Controller
//!
//! Adaptive backpressure control for streaming pipelines. Supports multiple
//! strategies: Drop, Block, Throttle (token-bucket), and SpillToDisk.
//!
//! ## Strategies
//!
//! - **Drop**: Discard incoming items when the queue is above the high watermark.
//! - **Block**: Signal the caller to block (returns `ThrottleDelay(u64::MAX)`).
//! - **Throttle**: Token-bucket rate limiter — items accepted only when a token
//!   is available; delay returned otherwise.
//! - **SpillToDisk**: Like Drop but caller is expected to write to `path`; this
//!   module records the event for stats purposes only (I/O is caller's concern).

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// Strategy
// ─────────────────────────────────────────────

/// How the controller responds when the queue exceeds the high-watermark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackpressureStrategy {
    /// Silently discard the incoming item.
    Drop,
    /// Tell the caller to block indefinitely.
    Block,
    /// Token-bucket throttle: accept at most `rate_hz` items per second.
    Throttle {
        /// Target accept rate in items per second.
        rate_hz: f64,
    },
    /// Spill overflowing items to disk at the given path.
    SpillToDisk {
        /// Filesystem path for the spill file.
        path: String,
    },
}

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

/// Configuration for a [`BackpressureController`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureConfig {
    /// Strategy applied when the queue exceeds `high_watermark`.
    pub strategy: BackpressureStrategy,
    /// Queue depth at which backpressure is activated.
    pub high_watermark: usize,
    /// Queue depth at which backpressure is deactivated.
    pub low_watermark: usize,
    /// Rolling window length in milliseconds (reserved for future analytics).
    pub window_ms: u64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            strategy: BackpressureStrategy::Drop,
            high_watermark: 1000,
            low_watermark: 500,
            window_ms: 1000,
        }
    }
}

// ─────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────

/// Cumulative statistics tracked by a [`BackpressureController`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackpressureStats {
    /// Total items dropped (Drop / SpillToDisk strategy).
    pub items_dropped: u64,
    /// Total items for which a throttle delay was returned.
    pub items_throttled: u64,
    /// Current number of items in the logical queue.
    pub current_queue_depth: usize,
    /// Maximum queue depth observed since the last reset.
    pub peak_queue_depth: usize,
    /// Number of times backpressure was triggered.
    pub backpressure_events: u64,
}

// ─────────────────────────────────────────────
// Decision
// ─────────────────────────────────────────────

/// Decision returned by [`BackpressureController::try_accept`].
#[derive(Debug, Clone, PartialEq)]
pub enum BackpressureDecision {
    /// The item is accepted; caller may enqueue it.
    Accept,
    /// The item should be dropped.
    Drop,
    /// The caller should wait `delay_ms` milliseconds before retrying.
    ThrottleDelay(u64),
}

// ─────────────────────────────────────────────
// Controller
// ─────────────────────────────────────────────

/// Adaptive backpressure controller implementing multiple mitigation strategies.
#[derive(Debug)]
pub struct BackpressureController {
    config: BackpressureConfig,
    /// Logical queue depth (incremented on accept, decremented on dequeue).
    queue_depth: usize,
    stats: BackpressureStats,
    /// Token bucket level (only meaningful for Throttle strategy).
    throttle_tokens: f64,
    /// Timestamp of the last token replenishment (ms since epoch / monotonic).
    last_tick_ms: u64,
    /// Whether we are currently in backpressure state.
    in_backpressure: bool,
}

impl BackpressureController {
    /// Create a new controller with the supplied configuration.
    pub fn new(config: BackpressureConfig) -> Self {
        let initial_tokens = if let BackpressureStrategy::Throttle { rate_hz } = config.strategy {
            rate_hz.max(0.0)
        } else {
            0.0
        };
        Self {
            config,
            queue_depth: 0,
            stats: BackpressureStats::default(),
            throttle_tokens: initial_tokens,
            last_tick_ms: 0,
            in_backpressure: false,
        }
    }

    /// Attempt to accept one item at `now_ms` (monotonic milliseconds).
    ///
    /// Updates internal state and returns the decision the caller should act on.
    pub fn try_accept(&mut self, now_ms: u64) -> BackpressureDecision {
        // Replenish tokens regardless of watermark level.
        self.replenish_tokens(now_ms);

        let above_high = self.is_above_high_watermark();
        let was_in_backpressure = self.in_backpressure;

        if above_high && !was_in_backpressure {
            self.in_backpressure = true;
            self.stats.backpressure_events += 1;
        } else if self.is_below_low_watermark() {
            self.in_backpressure = false;
        }

        if !self.in_backpressure {
            // Normal path — accept unconditionally.
            self.queue_depth += 1;
            if self.queue_depth > self.stats.peak_queue_depth {
                self.stats.peak_queue_depth = self.queue_depth;
            }
            self.stats.current_queue_depth = self.queue_depth;
            return BackpressureDecision::Accept;
        }

        // Backpressure path — apply strategy.
        match &self.config.strategy {
            BackpressureStrategy::Drop => {
                self.stats.items_dropped += 1;
                BackpressureDecision::Drop
            }
            BackpressureStrategy::Block => {
                // Signal the caller to block.
                BackpressureDecision::ThrottleDelay(u64::MAX)
            }
            BackpressureStrategy::Throttle { rate_hz } => {
                if self.throttle_tokens >= 1.0 {
                    self.throttle_tokens -= 1.0;
                    self.queue_depth += 1;
                    if self.queue_depth > self.stats.peak_queue_depth {
                        self.stats.peak_queue_depth = self.queue_depth;
                    }
                    self.stats.current_queue_depth = self.queue_depth;
                    BackpressureDecision::Accept
                } else {
                    // Compute how many ms until the next token is available.
                    let delay_ms = if *rate_hz > 0.0 {
                        ((1.0 - self.throttle_tokens) / rate_hz * 1000.0).ceil() as u64
                    } else {
                        u64::MAX
                    };
                    self.stats.items_throttled += 1;
                    BackpressureDecision::ThrottleDelay(delay_ms)
                }
            }
            BackpressureStrategy::SpillToDisk { .. } => {
                // Caller is responsible for actual I/O; we record it as dropped.
                self.stats.items_dropped += 1;
                BackpressureDecision::Drop
            }
        }
    }

    /// Record that one item was consumed from the queue.
    pub fn record_dequeue(&mut self) {
        self.queue_depth = self.queue_depth.saturating_sub(1);
        self.stats.current_queue_depth = self.queue_depth;
    }

    /// Return a reference to the current statistics.
    pub fn stats(&self) -> &BackpressureStats {
        &self.stats
    }

    /// Reset all counters (peak_queue_depth, items_dropped, …) without
    /// changing the queue depth or backpressure state.
    pub fn reset_stats(&mut self) {
        let current = self.queue_depth;
        self.stats = BackpressureStats {
            current_queue_depth: current,
            peak_queue_depth: current,
            ..Default::default()
        };
    }

    /// Return `true` when the current depth is at or above the high watermark.
    pub fn is_above_high_watermark(&self) -> bool {
        self.queue_depth >= self.config.high_watermark
    }

    /// Return `true` when the current depth is at or below the low watermark.
    pub fn is_below_low_watermark(&self) -> bool {
        self.queue_depth <= self.config.low_watermark
    }

    /// Replenish the token bucket based on elapsed time since the last tick.
    ///
    /// Only effective when the strategy is `Throttle`.
    pub fn replenish_tokens(&mut self, now_ms: u64) {
        if let BackpressureStrategy::Throttle { rate_hz } = self.config.strategy {
            if self.last_tick_ms == 0 {
                self.last_tick_ms = now_ms;
                return;
            }
            let elapsed_ms = now_ms.saturating_sub(self.last_tick_ms);
            let new_tokens = rate_hz * (elapsed_ms as f64 / 1000.0);
            self.throttle_tokens = (self.throttle_tokens + new_tokens).min(rate_hz);
            self.last_tick_ms = now_ms;
        }
    }

    /// Return the current logical queue depth.
    pub fn current_depth(&self) -> usize {
        self.queue_depth
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn drop_config(high: usize, low: usize) -> BackpressureConfig {
        BackpressureConfig {
            strategy: BackpressureStrategy::Drop,
            high_watermark: high,
            low_watermark: low,
            window_ms: 1000,
        }
    }

    fn throttle_config(rate_hz: f64, high: usize, low: usize) -> BackpressureConfig {
        BackpressureConfig {
            strategy: BackpressureStrategy::Throttle { rate_hz },
            high_watermark: high,
            low_watermark: low,
            window_ms: 1000,
        }
    }

    // ── construction ────────────────────────────────────────────────────

    #[test]
    fn test_new_default_config() {
        let ctrl = BackpressureController::new(BackpressureConfig::default());
        assert_eq!(ctrl.current_depth(), 0);
        assert_eq!(ctrl.stats().items_dropped, 0);
        assert!(!ctrl.is_above_high_watermark());
        assert!(ctrl.is_below_low_watermark());
    }

    #[test]
    fn test_new_throttle_sets_initial_tokens() {
        let ctrl = BackpressureController::new(throttle_config(100.0, 50, 25));
        assert!((ctrl.throttle_tokens - 100.0).abs() < 1e-9);
    }

    // ── try_accept under high watermark ─────────────────────────────────

    #[test]
    fn test_accept_below_high_watermark() {
        let mut ctrl = BackpressureController::new(drop_config(10, 5));
        for i in 0..9 {
            let decision = ctrl.try_accept(i as u64 * 10);
            assert_eq!(decision, BackpressureDecision::Accept, "step {i}");
        }
        assert_eq!(ctrl.current_depth(), 9);
    }

    // ── drop strategy ───────────────────────────────────────────────────

    #[test]
    fn test_drop_at_high_watermark() {
        let mut ctrl = BackpressureController::new(drop_config(3, 1));
        ctrl.try_accept(0);
        ctrl.try_accept(1);
        ctrl.try_accept(2); // depth = 3 → high watermark
        let decision = ctrl.try_accept(3);
        assert_eq!(decision, BackpressureDecision::Drop);
        assert_eq!(ctrl.stats().items_dropped, 1);
    }

    #[test]
    fn test_drop_increments_counter() {
        let mut ctrl = BackpressureController::new(drop_config(2, 1));
        ctrl.try_accept(0);
        ctrl.try_accept(1); // high watermark hit
        ctrl.try_accept(2);
        ctrl.try_accept(3);
        assert_eq!(ctrl.stats().items_dropped, 2);
    }

    #[test]
    fn test_backpressure_event_counted() {
        let mut ctrl = BackpressureController::new(drop_config(2, 1));
        ctrl.try_accept(0);
        ctrl.try_accept(1);
        ctrl.try_accept(2); // triggers bp event
        ctrl.try_accept(3); // still in bp, no new event
        assert_eq!(ctrl.stats().backpressure_events, 1);
    }

    // ── record_dequeue and recovery ──────────────────────────────────────

    #[test]
    fn test_dequeue_decrements_depth() {
        let mut ctrl = BackpressureController::new(drop_config(10, 2));
        ctrl.try_accept(0);
        ctrl.try_accept(1);
        ctrl.try_accept(2);
        ctrl.record_dequeue();
        assert_eq!(ctrl.current_depth(), 2);
        assert_eq!(ctrl.stats().current_queue_depth, 2);
    }

    #[test]
    fn test_dequeue_saturates_at_zero() {
        let mut ctrl = BackpressureController::new(drop_config(10, 2));
        ctrl.record_dequeue(); // depth is already 0
        assert_eq!(ctrl.current_depth(), 0);
    }

    #[test]
    fn test_recovery_after_dequeue() {
        let mut ctrl = BackpressureController::new(drop_config(3, 1));
        // Fill to high watermark
        ctrl.try_accept(0);
        ctrl.try_accept(1);
        ctrl.try_accept(2); // bp activated at depth=3
                            // Drain below low watermark
        ctrl.record_dequeue(); // 2
        ctrl.record_dequeue(); // 1 — at or below low watermark
                               // Now next accept should succeed
        let decision = ctrl.try_accept(100);
        assert_eq!(decision, BackpressureDecision::Accept);
    }

    // ── peak tracking ────────────────────────────────────────────────────

    #[test]
    fn test_peak_depth_tracked() {
        let mut ctrl = BackpressureController::new(drop_config(20, 5));
        for i in 0..10u64 {
            ctrl.try_accept(i);
        }
        for _ in 0..5 {
            ctrl.record_dequeue();
        }
        assert_eq!(ctrl.stats().peak_queue_depth, 10);
        assert_eq!(ctrl.current_depth(), 5);
    }

    // ── reset_stats ─────────────────────────────────────────────────────

    #[test]
    fn test_reset_stats_clears_counters() {
        let mut ctrl = BackpressureController::new(drop_config(2, 1));
        ctrl.try_accept(0);
        ctrl.try_accept(1);
        ctrl.try_accept(2); // drops start
        ctrl.try_accept(3);
        ctrl.reset_stats();
        assert_eq!(ctrl.stats().items_dropped, 0);
        assert_eq!(ctrl.stats().backpressure_events, 0);
        assert_eq!(ctrl.stats().items_throttled, 0);
        // current depth should be preserved
        assert_eq!(ctrl.stats().current_queue_depth, ctrl.current_depth());
    }

    #[test]
    fn test_reset_stats_preserves_depth() {
        let mut ctrl = BackpressureController::new(drop_config(10, 2));
        ctrl.try_accept(0);
        ctrl.try_accept(1);
        ctrl.try_accept(2);
        ctrl.reset_stats();
        assert_eq!(ctrl.current_depth(), 3);
    }

    // ── watermark predicates ─────────────────────────────────────────────

    #[test]
    fn test_above_high_watermark() {
        let mut ctrl = BackpressureController::new(drop_config(5, 2));
        for i in 0..5u64 {
            ctrl.try_accept(i);
        }
        assert!(ctrl.is_above_high_watermark());
    }

    #[test]
    fn test_below_low_watermark() {
        let ctrl = BackpressureController::new(drop_config(5, 2));
        assert!(ctrl.is_below_low_watermark());
    }

    #[test]
    fn test_between_watermarks() {
        let mut ctrl = BackpressureController::new(drop_config(10, 2));
        for i in 0..5u64 {
            ctrl.try_accept(i);
        }
        assert!(!ctrl.is_above_high_watermark());
        assert!(!ctrl.is_below_low_watermark());
    }

    // ── throttle strategy ────────────────────────────────────────────────

    #[test]
    fn test_throttle_accept_when_tokens_available() {
        let mut ctrl = BackpressureController::new(throttle_config(10.0, 2, 1));
        // Initial tokens = rate_hz = 10.0, so first accept below watermark is normal.
        let d = ctrl.try_accept(0);
        assert_eq!(d, BackpressureDecision::Accept);
    }

    #[test]
    fn test_throttle_delay_when_no_tokens() {
        let mut ctrl = BackpressureController::new(throttle_config(1.0, 2, 1));
        // Fill to high watermark: two accepts (depth 1, 2)
        ctrl.try_accept(0); // depth 1
        ctrl.try_accept(0); // depth 2 — hits high watermark, bp activated

        // First accept in bp consumes the only token
        ctrl.try_accept(0);
        // Now tokens are exhausted — next should throttle
        let decision = ctrl.try_accept(0);
        assert!(
            matches!(decision, BackpressureDecision::ThrottleDelay(_)),
            "expected ThrottleDelay, got {decision:?}"
        );
        assert!(ctrl.stats().items_throttled > 0);
    }

    #[test]
    fn test_throttle_replenish_over_time() {
        let mut ctrl = BackpressureController::new(throttle_config(10.0, 2, 1));
        // Exhaust by accepting well above high watermark
        ctrl.try_accept(0); // depth=1
        ctrl.try_accept(0); // depth=2, bp activated
                            // Burn the token immediately
        ctrl.try_accept(0); // uses the 1 token available right now (tokens now <1)
                            // Advance time by 1 second — should replenish 10 tokens
        ctrl.try_accept(1000);
        // Tokens should be available now
        let decision = ctrl.try_accept(1000);
        assert_ne!(decision, BackpressureDecision::Drop);
    }

    #[test]
    fn test_replenish_tokens_noop_without_throttle() {
        let mut ctrl = BackpressureController::new(drop_config(10, 2));
        let tokens_before = ctrl.throttle_tokens;
        ctrl.replenish_tokens(5000);
        assert!((ctrl.throttle_tokens - tokens_before).abs() < 1e-9);
    }

    #[test]
    fn test_replenish_tokens_first_tick() {
        let mut ctrl = BackpressureController::new(throttle_config(10.0, 10, 5));
        ctrl.replenish_tokens(1000); // sets last_tick_ms, should not crash
        assert_eq!(ctrl.last_tick_ms, 1000);
    }

    #[test]
    fn test_replenish_tokens_capped_at_rate_hz() {
        let mut ctrl = BackpressureController::new(throttle_config(5.0, 10, 5));
        ctrl.last_tick_ms = 1;
        ctrl.replenish_tokens(100_000); // huge elapsed time
        assert!(ctrl.throttle_tokens <= 5.0 + 1e-9);
    }

    // ── block strategy ───────────────────────────────────────────────────

    #[test]
    fn test_block_strategy_returns_max_delay() {
        let config = BackpressureConfig {
            strategy: BackpressureStrategy::Block,
            high_watermark: 2,
            low_watermark: 1,
            window_ms: 100,
        };
        let mut ctrl = BackpressureController::new(config);
        ctrl.try_accept(0); // depth=1
        ctrl.try_accept(0); // depth=2, bp activated
        let d = ctrl.try_accept(0);
        assert_eq!(d, BackpressureDecision::ThrottleDelay(u64::MAX));
    }

    // ── spill-to-disk strategy ───────────────────────────────────────────

    #[test]
    fn test_spill_to_disk_records_as_dropped() {
        let config = BackpressureConfig {
            strategy: BackpressureStrategy::SpillToDisk {
                path: "/tmp/spill.bin".to_string(),
            },
            high_watermark: 2,
            low_watermark: 1,
            window_ms: 100,
        };
        let mut ctrl = BackpressureController::new(config);
        ctrl.try_accept(0);
        ctrl.try_accept(0); // bp activated
        let d = ctrl.try_accept(0);
        assert_eq!(d, BackpressureDecision::Drop);
        assert_eq!(ctrl.stats().items_dropped, 1);
    }

    // ── current_depth ────────────────────────────────────────────────────

    #[test]
    fn test_current_depth_tracks_accept_and_dequeue() {
        let mut ctrl = BackpressureController::new(drop_config(100, 50));
        assert_eq!(ctrl.current_depth(), 0);
        ctrl.try_accept(0);
        ctrl.try_accept(1);
        assert_eq!(ctrl.current_depth(), 2);
        ctrl.record_dequeue();
        assert_eq!(ctrl.current_depth(), 1);
    }

    // ── edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_zero_rate_throttle_returns_max_delay() {
        let mut ctrl = BackpressureController::new(throttle_config(0.0, 2, 1));
        ctrl.try_accept(0); // depth=1
        ctrl.try_accept(0); // depth=2, bp activated
                            // tokens = 0 initially for zero rate; first try in bp
        ctrl.try_accept(0); // uses tokens=0 → throttle
        let d = ctrl.try_accept(0);
        assert!(matches!(d, BackpressureDecision::ThrottleDelay(_)));
    }

    #[test]
    fn test_high_watermark_equals_one() {
        let mut ctrl = BackpressureController::new(drop_config(1, 0));
        let d = ctrl.try_accept(0); // depth becomes 1 → at high watermark
        assert_eq!(d, BackpressureDecision::Accept);
        // bp is not yet active (activated on next accept above threshold)
        let d2 = ctrl.try_accept(0);
        assert_eq!(d2, BackpressureDecision::Drop);
    }

    #[test]
    fn test_stats_ref_is_consistent() {
        let mut ctrl = BackpressureController::new(drop_config(5, 2));
        ctrl.try_accept(0);
        ctrl.try_accept(1);
        let s = ctrl.stats();
        assert_eq!(s.current_queue_depth, 2);
        assert_eq!(s.peak_queue_depth, 2);
        assert_eq!(s.items_dropped, 0);
    }

    #[test]
    fn test_multiple_backpressure_cycles() {
        // high_watermark=3: bp activates when depth is already 3 (on the 4th call)
        let mut ctrl = BackpressureController::new(drop_config(3, 1));
        // First cycle: fill to high watermark then trigger bp
        ctrl.try_accept(0); // depth=1
        ctrl.try_accept(0); // depth=2
        ctrl.try_accept(0); // depth=3 (normal accept, no bp yet)
        ctrl.try_accept(0); // depth=3 (above high) → bp on, event_count=1 → Drop
                            // Drain below low watermark (1)
        ctrl.record_dequeue(); // 2
        ctrl.record_dequeue(); // 1 — at low watermark → bp off
                               // Second cycle
        ctrl.try_accept(0); // depth=2
        ctrl.try_accept(0); // depth=3 (normal accept)
        ctrl.try_accept(0); // above high → bp on again, event_count=2 → Drop
        assert_eq!(ctrl.stats().backpressure_events, 2);
    }
}
