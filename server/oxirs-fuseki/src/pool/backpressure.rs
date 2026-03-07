//! Backpressure Controller for Connection Pools
//!
//! Monitors pool utilization and queue depth to decide whether to accept,
//! queue, or reject incoming requests.
//!
//! Decision policy
//! ───────────────
//! - utilization < queue_threshold                  → Accept
//! - queue_threshold ≤ utilization < reject_threshold  → Queue (increments queue depth)
//! - utilization ≥ reject_threshold
//!   OR queue_depth ≥ max_queue_depth              → Reject (with exponential back-off hint)
//!
//! Thread-safe: uses `Arc<AtomicU32>` / `Arc<AtomicU64>` for all mutable state.

use serde::Serialize;
use std::sync::{
    atomic::{AtomicU32, AtomicU64, Ordering},
    Arc,
};

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Tuning knobs for `BackpressureController`.
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Pool utilization (0.0–1.0) above which requests begin to be queued.
    /// Default: 0.70
    pub queue_threshold: f64,
    /// Pool utilization (0.0–1.0) above which requests are rejected outright.
    /// Default: 0.90
    pub reject_threshold: f64,
    /// Maximum queue depth.  Once exceeded, new requests are rejected even if
    /// utilization is below `reject_threshold`.
    /// Default: 100
    pub max_queue_depth: u32,
    /// Base `retry_after_ms` returned in a `Reject` decision.
    /// The actual value is scaled by `(1 + queue_depth / 10)` to provide
    /// adaptive back-off.
    /// Default: 100 ms
    pub base_retry_after_ms: u64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        BackpressureConfig {
            queue_threshold: 0.70,
            reject_threshold: 0.90,
            max_queue_depth: 100,
            base_retry_after_ms: 100,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Decision
// ──────────────────────────────────────────────────────────────────────────────

/// The action the server should take for an incoming request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackpressureDecision {
    /// Accept the request immediately — pool has capacity.
    Accept,
    /// Queue the request — pool is busy but not saturated.
    Queue,
    /// Reject the request — pool is overloaded.
    ///
    /// The `retry_after_ms` hint tells the caller when it may retry.
    Reject {
        /// Suggested delay before retrying, in milliseconds.
        retry_after_ms: u64,
    },
}

// ──────────────────────────────────────────────────────────────────────────────
// Statistics
// ──────────────────────────────────────────────────────────────────────────────

/// Snapshot statistics for a `BackpressureController`.
#[derive(Debug, Clone, Serialize)]
pub struct BackpressureStats {
    /// Number of requests currently queued
    pub queue_depth: u32,
    /// Total requests accepted
    pub total_accepted: u64,
    /// Total requests queued
    pub total_queued: u64,
    /// Total requests rejected
    pub total_rejected: u64,
    /// Fraction of requests rejected: rejected / total
    pub rejection_rate: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// BackpressureController
// ──────────────────────────────────────────────────────────────────────────────

/// Thread-safe backpressure controller.
///
/// All internal counters are atomic so that multiple threads can call
/// `should_accept_request`, `record_dequeue`, and `record_completion`
/// concurrently without any explicit locking.
#[derive(Clone)]
pub struct BackpressureController {
    config: BackpressureConfig,
    queue_depth: Arc<AtomicU32>,
    total_accepted: Arc<AtomicU64>,
    total_queued: Arc<AtomicU64>,
    total_rejected: Arc<AtomicU64>,
}

impl BackpressureController {
    /// Create a new controller with the given configuration.
    pub fn new(config: BackpressureConfig) -> Self {
        BackpressureController {
            config,
            queue_depth: Arc::new(AtomicU32::new(0)),
            total_accepted: Arc::new(AtomicU64::new(0)),
            total_queued: Arc::new(AtomicU64::new(0)),
            total_rejected: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Evaluate whether an incoming request should be accepted, queued, or rejected.
    ///
    /// `utilization` is the fraction of pool connections currently in use (0.0–1.0).
    /// Values outside [0.0, 1.0] are clamped.
    ///
    /// Side effects:
    /// - **Accept** → increments `total_accepted`
    /// - **Queue**  → increments `queue_depth` and `total_queued`
    /// - **Reject** → increments `total_rejected`
    pub fn should_accept_request(&self, utilization: f64) -> BackpressureDecision {
        let utilization = utilization.clamp(0.0, 1.0);
        let depth = self.queue_depth.load(Ordering::Acquire);

        // Reject if utilization is above reject_threshold OR queue is full
        if utilization >= self.config.reject_threshold || depth >= self.config.max_queue_depth {
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            // Adaptive retry delay: base * (1 + depth/10)
            let backoff_multiplier = 1u64 + (depth as u64 / 10);
            let retry_after_ms = self
                .config
                .base_retry_after_ms
                .saturating_mul(backoff_multiplier);
            return BackpressureDecision::Reject { retry_after_ms };
        }

        // Queue if utilization is between the two thresholds
        if utilization >= self.config.queue_threshold {
            self.queue_depth.fetch_add(1, Ordering::AcqRel);
            self.total_queued.fetch_add(1, Ordering::Relaxed);
            return BackpressureDecision::Queue;
        }

        // Accept
        self.total_accepted.fetch_add(1, Ordering::Relaxed);
        BackpressureDecision::Accept
    }

    /// Signal that a queued request has been dequeued and is now being processed.
    ///
    /// Decrements the queue depth (saturating at 0).
    pub fn record_dequeue(&self) {
        // Saturating decrement: loop with CAS to avoid underflow
        loop {
            let cur = self.queue_depth.load(Ordering::Acquire);
            if cur == 0 {
                break;
            }
            if self
                .queue_depth
                .compare_exchange(cur, cur - 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Signal that a previously accepted (or dequeued) request has completed.
    ///
    /// Currently a no-op reserved for future rate tracking (e.g. throughput metrics).
    pub fn record_completion(&self) {
        // Reserved for future throughput-tracking logic.
    }

    /// Current queue depth.
    pub fn current_queue_depth(&self) -> u32 {
        self.queue_depth.load(Ordering::Relaxed)
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> BackpressureStats {
        let accepted = self.total_accepted.load(Ordering::Relaxed);
        let queued = self.total_queued.load(Ordering::Relaxed);
        let rejected = self.total_rejected.load(Ordering::Relaxed);
        let total = accepted + queued + rejected;

        BackpressureStats {
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            total_accepted: accepted,
            total_queued: queued,
            total_rejected: rejected,
            rejection_rate: if total == 0 {
                0.0
            } else {
                rejected as f64 / total as f64
            },
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_controller() -> BackpressureController {
        BackpressureController::new(BackpressureConfig::default())
    }

    // ── BackpressureConfig defaults ───────────────────────────────────────────

    #[test]
    fn test_default_config_values() {
        let cfg = BackpressureConfig::default();
        assert!((cfg.queue_threshold - 0.70).abs() < f64::EPSILON);
        assert!((cfg.reject_threshold - 0.90).abs() < f64::EPSILON);
        assert_eq!(cfg.max_queue_depth, 100);
        assert_eq!(cfg.base_retry_after_ms, 100);
    }

    // ── Accept decisions ──────────────────────────────────────────────────────

    #[test]
    fn test_accept_when_zero_utilization() {
        let ctrl = make_controller();
        assert_eq!(
            ctrl.should_accept_request(0.0),
            BackpressureDecision::Accept
        );
    }

    #[test]
    fn test_accept_below_queue_threshold() {
        let ctrl = make_controller();
        assert_eq!(
            ctrl.should_accept_request(0.5),
            BackpressureDecision::Accept
        );
    }

    #[test]
    fn test_accept_just_below_queue_threshold() {
        let ctrl = make_controller();
        // 0.699 < 0.70 → Accept
        assert_eq!(
            ctrl.should_accept_request(0.699),
            BackpressureDecision::Accept
        );
    }

    // ── Queue decisions ───────────────────────────────────────────────────────

    #[test]
    fn test_queue_at_queue_threshold() {
        let ctrl = make_controller();
        // exactly 0.70 → Queue
        assert_eq!(
            ctrl.should_accept_request(0.70),
            BackpressureDecision::Queue
        );
    }

    #[test]
    fn test_queue_between_thresholds() {
        let ctrl = make_controller();
        assert_eq!(
            ctrl.should_accept_request(0.80),
            BackpressureDecision::Queue
        );
    }

    #[test]
    fn test_queue_just_below_reject_threshold() {
        let ctrl = make_controller();
        // 0.899 < 0.90 → Queue
        assert_eq!(
            ctrl.should_accept_request(0.899),
            BackpressureDecision::Queue
        );
    }

    // ── Reject decisions ──────────────────────────────────────────────────────

    #[test]
    fn test_reject_at_reject_threshold() {
        let ctrl = make_controller();
        match ctrl.should_accept_request(0.90) {
            BackpressureDecision::Reject { .. } => {}
            other => panic!("Expected Reject, got {:?}", other),
        }
    }

    #[test]
    fn test_reject_above_reject_threshold() {
        let ctrl = make_controller();
        match ctrl.should_accept_request(1.0) {
            BackpressureDecision::Reject { .. } => {}
            other => panic!("Expected Reject, got {:?}", other),
        }
    }

    #[test]
    fn test_reject_when_queue_full() {
        let ctrl = BackpressureController::new(BackpressureConfig {
            queue_threshold: 0.70,
            reject_threshold: 0.90,
            max_queue_depth: 3,
            base_retry_after_ms: 100,
        });

        // Fill the queue: utilization 0.80 (Queue) × 3
        ctrl.should_accept_request(0.80);
        ctrl.should_accept_request(0.80);
        ctrl.should_accept_request(0.80);

        // Now queue depth == 3 == max_queue_depth → Reject
        match ctrl.should_accept_request(0.80) {
            BackpressureDecision::Reject { .. } => {}
            other => panic!("Expected Reject when queue full, got {:?}", other),
        }
    }

    // ── retry_after_ms backoff ────────────────────────────────────────────────

    #[test]
    fn test_retry_after_increases_with_queue_depth() {
        let ctrl = BackpressureController::new(BackpressureConfig {
            queue_threshold: 0.70,
            reject_threshold: 0.90,
            max_queue_depth: 1000,
            base_retry_after_ms: 100,
        });

        // Queue up 10 requests to set depth = 10
        for _ in 0..10 {
            ctrl.should_accept_request(0.80); // Queue
        }

        // First reject at depth 10: multiplier = 1 + 10/10 = 2 → 200 ms
        let first_reject = ctrl.should_accept_request(0.95);
        // Queue up 10 more: depth = 20
        for _ in 0..10 {
            ctrl.should_accept_request(0.80);
        }
        let second_reject = ctrl.should_accept_request(0.95);

        let first_ms = match first_reject {
            BackpressureDecision::Reject { retry_after_ms } => retry_after_ms,
            other => panic!("Expected Reject, got {:?}", other),
        };
        let second_ms = match second_reject {
            BackpressureDecision::Reject { retry_after_ms } => retry_after_ms,
            other => panic!("Expected Reject, got {:?}", other),
        };

        assert!(
            second_ms >= first_ms,
            "retry_after_ms should be >= at higher queue depth: {} vs {}",
            second_ms,
            first_ms
        );
    }

    #[test]
    fn test_retry_after_base_when_zero_depth() {
        let ctrl = BackpressureController::new(BackpressureConfig {
            base_retry_after_ms: 50,
            reject_threshold: 0.0, // reject immediately
            ..Default::default()
        });
        match ctrl.should_accept_request(0.0) {
            BackpressureDecision::Reject { retry_after_ms } => {
                // depth=0 → multiplier=1 → 50*1 = 50
                assert_eq!(retry_after_ms, 50);
            }
            other => panic!("Expected Reject, got {:?}", other),
        }
    }

    // ── record_dequeue ────────────────────────────────────────────────────────

    #[test]
    fn test_record_dequeue_decrements() {
        let ctrl = make_controller();
        ctrl.should_accept_request(0.80); // Queue → depth=1
        ctrl.should_accept_request(0.80); // Queue → depth=2
        assert_eq!(ctrl.current_queue_depth(), 2);

        ctrl.record_dequeue();
        assert_eq!(ctrl.current_queue_depth(), 1);

        ctrl.record_dequeue();
        assert_eq!(ctrl.current_queue_depth(), 0);
    }

    #[test]
    fn test_record_dequeue_saturates_at_zero() {
        let ctrl = make_controller();
        assert_eq!(ctrl.current_queue_depth(), 0);
        ctrl.record_dequeue(); // Should not underflow
        assert_eq!(ctrl.current_queue_depth(), 0);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_state() {
        let ctrl = make_controller();
        let stats = ctrl.stats();
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.total_accepted, 0);
        assert_eq!(stats.total_queued, 0);
        assert_eq!(stats.total_rejected, 0);
        assert!((stats.rejection_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_counts_accumulate() {
        let ctrl = make_controller();
        ctrl.should_accept_request(0.0); // Accept
        ctrl.should_accept_request(0.5); // Accept
        ctrl.should_accept_request(0.80); // Queue
        ctrl.should_accept_request(0.95); // Reject

        let stats = ctrl.stats();
        assert_eq!(stats.total_accepted, 2);
        assert_eq!(stats.total_queued, 1);
        assert_eq!(stats.total_rejected, 1);
    }

    #[test]
    fn test_stats_rejection_rate() {
        let ctrl = make_controller();
        // 1 accept, 1 reject → rate = 0.5
        ctrl.should_accept_request(0.0); // Accept
        ctrl.should_accept_request(0.95); // Reject

        let stats = ctrl.stats();
        assert!(
            (stats.rejection_rate - 0.5).abs() < f64::EPSILON,
            "Expected 0.5, got {}",
            stats.rejection_rate
        );
    }

    #[test]
    fn test_stats_rejection_rate_zero_when_no_rejects() {
        let ctrl = make_controller();
        ctrl.should_accept_request(0.0);
        ctrl.should_accept_request(0.5);
        let stats = ctrl.stats();
        assert!((stats.rejection_rate - 0.0).abs() < f64::EPSILON);
    }

    // ── thread safety ─────────────────────────────────────────────────────────

    #[test]
    fn test_concurrent_decisions_do_not_corrupt_state() {
        use std::thread;

        let ctrl = Arc::new(BackpressureController::new(BackpressureConfig {
            queue_threshold: 0.70,
            reject_threshold: 0.95,
            max_queue_depth: 1000,
            base_retry_after_ms: 10,
        }));

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let c = Arc::clone(&ctrl);
                thread::spawn(move || {
                    let util = (i as f64) / 10.0;
                    for _ in 0..100 {
                        let decision = c.should_accept_request(util);
                        if decision == BackpressureDecision::Queue {
                            c.record_dequeue();
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("Thread panicked");
        }

        // No panic = thread safety maintained
        let stats = ctrl.stats();
        assert_eq!(
            stats.total_accepted + stats.total_queued + stats.total_rejected,
            800,
            "Total decisions should equal 8 threads × 100 requests"
        );
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_utilization_above_one_clamped_to_reject() {
        let ctrl = make_controller();
        match ctrl.should_accept_request(2.0) {
            BackpressureDecision::Reject { .. } => {}
            other => panic!("Expected Reject for utilization > 1.0, got {:?}", other),
        }
    }

    #[test]
    fn test_utilization_below_zero_clamped_to_accept() {
        let ctrl = make_controller();
        assert_eq!(
            ctrl.should_accept_request(-0.5),
            BackpressureDecision::Accept
        );
    }
}
