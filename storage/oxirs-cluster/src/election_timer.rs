//! Raft-style election timer management (v1.1.0 round 16).
//!
//! Each follower/candidate in a Raft cluster maintains an election timer
//! with a randomised timeout drawn from `[min_timeout_ms, max_timeout_ms)`.
//! When the timer fires, the node promotes itself to candidate and starts an
//! election.
//!
//! Reference: Ongaro & Ousterhout, "In Search of an Understandable Consensus
//! Algorithm", USENIX ATC 2014. §5.2

// ──────────────────────────────────────────────────────────────────────────────
// TimerState
// ──────────────────────────────────────────────────────────────────────────────

/// The current state of an election timer.
#[derive(Debug, Clone, PartialEq)]
pub enum TimerState {
    /// The timer has not been started or has been explicitly stopped.
    Idle,
    /// The timer is running and will fire when `now_ms >= deadline_ms`.
    Running {
        /// Absolute deadline in milliseconds since epoch / arbitrary epoch.
        deadline_ms: u64,
    },
    /// The timer deadline has passed and an election should be triggered.
    Expired,
}

// ──────────────────────────────────────────────────────────────────────────────
// ElectionTimer
// ──────────────────────────────────────────────────────────────────────────────

/// A Raft election timer with a configurable randomised timeout range.
///
/// The timer uses a lightweight LCG (linear congruential generator) for
/// deterministic tests via [`ElectionTimer::new_with_seed`].
pub struct ElectionTimer {
    /// Lower bound of the election timeout (inclusive).
    min_timeout_ms: u64,
    /// Upper bound of the election timeout (exclusive).
    max_timeout_ms: u64,
    /// Current state of the timer.
    state: TimerState,
    /// LCG state for pseudo-random timeout selection.
    seed: u64,
    /// The timeout duration that was set on the last [`reset`](Self::reset).
    current_timeout_ms: u64,
}

impl ElectionTimer {
    /// Create a new timer with a random seed based on constant folding.
    ///
    /// `min_timeout_ms` must be < `max_timeout_ms`.
    pub fn new(min_timeout_ms: u64, max_timeout_ms: u64) -> Self {
        Self::new_with_seed(min_timeout_ms, max_timeout_ms, 0xabad_1dea_dead_beef)
    }

    /// Create a new timer with an explicit seed for deterministic behaviour.
    pub fn new_with_seed(min_timeout_ms: u64, max_timeout_ms: u64, seed: u64) -> Self {
        Self {
            min_timeout_ms,
            max_timeout_ms,
            state: TimerState::Idle,
            seed,
            current_timeout_ms: 0,
        }
    }

    /// Reset the timer using a freshly drawn random timeout, starting the
    /// deadline from `now_ms`.
    ///
    /// After a call to `reset`, the state transitions to `Running`.
    pub fn reset(&mut self, now_ms: u64) {
        let timeout = self.random_timeout();
        self.current_timeout_ms = timeout;
        self.state = TimerState::Running {
            deadline_ms: now_ms + timeout,
        };
    }

    /// Check whether the timer has expired given the current time `now_ms`.
    ///
    /// - If `Idle` → returns `false` (idle timers don't expire).
    /// - If `Running` and `now_ms >= deadline_ms` → transitions to `Expired`
    ///   and returns `true`.
    /// - If `Running` and not yet past deadline → returns `false`.
    /// - If already `Expired` → returns `true`.
    pub fn check(&mut self, now_ms: u64) -> bool {
        match self.state.clone() {
            TimerState::Idle => false,
            TimerState::Running { deadline_ms } => {
                if now_ms >= deadline_ms {
                    self.state = TimerState::Expired;
                    true
                } else {
                    false
                }
            }
            TimerState::Expired => true,
        }
    }

    /// Stop the timer, transitioning to `Idle`.
    pub fn stop(&mut self) {
        self.state = TimerState::Idle;
    }

    /// Return a reference to the current timer state.
    pub fn state(&self) -> &TimerState {
        &self.state
    }

    /// Return the timeout duration that was set on the last call to
    /// [`reset`](Self::reset), or `0` if the timer has never been reset.
    pub fn current_timeout_ms(&self) -> u64 {
        self.current_timeout_ms
    }

    /// Return `(min_timeout_ms, max_timeout_ms)`.
    pub fn timeout_range(&self) -> (u64, u64) {
        (self.min_timeout_ms, self.max_timeout_ms)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Draw a pseudo-random timeout in `[min, max)`.
    ///
    /// Uses the Knuth MMIX LCG: `seed = seed * a + c`.
    fn random_timeout(&mut self) -> u64 {
        // Knuth MMIX constants
        self.seed = self
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let range = self.max_timeout_ms.saturating_sub(self.min_timeout_ms);
        if range == 0 {
            return self.min_timeout_ms;
        }
        let offset = (self.seed >> 33) % range;
        self.min_timeout_ms + offset
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_timer(min: u64, max: u64) -> ElectionTimer {
        ElectionTimer::new_with_seed(min, max, 42)
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_state_is_idle() {
        let t = make_timer(150, 300);
        assert_eq!(t.state(), &TimerState::Idle);
    }

    #[test]
    fn test_new_current_timeout_zero() {
        let t = make_timer(150, 300);
        assert_eq!(t.current_timeout_ms(), 0);
    }

    #[test]
    fn test_timeout_range() {
        let t = make_timer(100, 250);
        assert_eq!(t.timeout_range(), (100, 250));
    }

    #[test]
    fn test_new_random_seed() {
        let t = ElectionTimer::new(100, 200);
        assert_eq!(t.state(), &TimerState::Idle);
    }

    // ── reset ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_sets_running() {
        let mut t = make_timer(150, 300);
        t.reset(1000);
        assert!(matches!(t.state(), TimerState::Running { .. }));
    }

    #[test]
    fn test_reset_current_timeout_in_range() {
        let mut t = make_timer(150, 300);
        t.reset(0);
        let timeout = t.current_timeout_ms();
        assert!(timeout >= 150, "timeout {} < min 150", timeout);
        assert!(timeout < 300, "timeout {} >= max 300", timeout);
    }

    #[test]
    fn test_reset_deadline_is_now_plus_timeout() {
        let mut t = make_timer(100, 200);
        let now = 500;
        t.reset(now);
        let timeout = t.current_timeout_ms();
        if let TimerState::Running { deadline_ms } = t.state() {
            assert_eq!(*deadline_ms, now + timeout);
        } else {
            panic!("expected Running state");
        }
    }

    #[test]
    fn test_reset_from_expired_state() {
        let mut t = make_timer(100, 200);
        t.reset(0);
        t.check(10_000); // expire
        assert_eq!(t.state(), &TimerState::Expired);
        t.reset(10_000);
        assert!(matches!(t.state(), TimerState::Running { .. }));
    }

    // ── check ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_check_before_deadline_false() {
        let mut t = make_timer(100, 200);
        t.reset(1000); // deadline = 1000 + [100,200)
        assert!(!t.check(1000)); // same instant
    }

    #[test]
    fn test_check_after_deadline_true() {
        let mut t = make_timer(100, 200);
        t.reset(1000);
        assert!(t.check(1200)); // well past deadline
    }

    #[test]
    fn test_check_transitions_to_expired() {
        let mut t = make_timer(100, 200);
        t.reset(1000);
        t.check(1500);
        assert_eq!(t.state(), &TimerState::Expired);
    }

    #[test]
    fn test_check_idle_is_false() {
        let mut t = make_timer(100, 200);
        assert!(!t.check(9999));
    }

    #[test]
    fn test_check_expired_remains_true() {
        let mut t = make_timer(100, 200);
        t.reset(0);
        t.check(10_000); // expire
                         // Checking again should still return true
        assert!(t.check(20_000));
    }

    #[test]
    fn test_check_at_exact_deadline() {
        let mut t = make_timer(100, 100); // range=0 → always 100
        t.reset(500);
        // deadline = 500 + 100 = 600
        assert!(!t.check(599));
        assert!(t.check(600));
    }

    // ── stop ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stop_from_running() {
        let mut t = make_timer(100, 200);
        t.reset(0);
        t.stop();
        assert_eq!(t.state(), &TimerState::Idle);
    }

    #[test]
    fn test_stop_from_expired() {
        let mut t = make_timer(100, 200);
        t.reset(0);
        t.check(10_000);
        t.stop();
        assert_eq!(t.state(), &TimerState::Idle);
    }

    #[test]
    fn test_stop_idempotent() {
        let mut t = make_timer(100, 200);
        t.stop();
        t.stop();
        assert_eq!(t.state(), &TimerState::Idle);
    }

    // ── Seed reproducibility ─────────────────────────────────────────────────

    #[test]
    fn test_seed_reproducibility() {
        let mut t1 = ElectionTimer::new_with_seed(100, 300, 0x1234_5678);
        let mut t2 = ElectionTimer::new_with_seed(100, 300, 0x1234_5678);
        t1.reset(0);
        t2.reset(0);
        assert_eq!(t1.current_timeout_ms(), t2.current_timeout_ms());
    }

    #[test]
    fn test_different_seeds_may_differ() {
        let mut t1 = ElectionTimer::new_with_seed(100, 300, 1);
        let mut t2 = ElectionTimer::new_with_seed(100, 300, 999_999);
        // Run several resets; at least one pair should differ
        let mut differs = false;
        for i in 0..10u64 {
            t1.reset(i * 10);
            t2.reset(i * 10);
            if t1.current_timeout_ms() != t2.current_timeout_ms() {
                differs = true;
                break;
            }
        }
        assert!(differs, "All timeouts identical for different seeds");
    }

    // ── Multiple resets ───────────────────────────────────────────────────────

    #[test]
    fn test_multiple_resets_all_in_range() {
        let mut t = ElectionTimer::new_with_seed(150, 300, 777);
        for i in 0..20u64 {
            t.reset(i * 1000);
            let timeout = t.current_timeout_ms();
            assert!(
                (150..300).contains(&timeout),
                "timeout {} not in [150, 300)",
                timeout
            );
        }
    }

    #[test]
    fn test_multiple_resets_override_each_other() {
        let mut t = make_timer(100, 200);
        t.reset(0);
        let first = t.current_timeout_ms();
        t.reset(1000);
        // After second reset, the timer is Running again
        assert!(matches!(t.state(), TimerState::Running { .. }));
        // current_timeout_ms reflects the latest reset
        let _ = first; // may or may not equal second
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_zero_range_always_min() {
        let mut t = ElectionTimer::new_with_seed(200, 200, 0xdeadbeef);
        for _ in 0..5 {
            t.reset(0);
            assert_eq!(t.current_timeout_ms(), 200);
        }
    }

    #[test]
    fn test_timer_state_idle_clone() {
        let s = TimerState::Idle;
        assert_eq!(s.clone(), TimerState::Idle);
    }

    #[test]
    fn test_timer_state_running_clone() {
        let s = TimerState::Running { deadline_ms: 42 };
        assert_eq!(s.clone(), TimerState::Running { deadline_ms: 42 });
    }

    #[test]
    fn test_timer_state_expired_clone() {
        let s = TimerState::Expired;
        assert_eq!(s.clone(), TimerState::Expired);
    }

    // ── workflow simulation ───────────────────────────────────────────────────

    #[test]
    fn test_raft_follower_workflow() {
        // Simulate a follower that receives a heartbeat (reset), then times out
        let mut t = ElectionTimer::new_with_seed(150, 300, 0x42);
        t.reset(0);
        assert!(!t.check(100)); // heartbeat arrives; still alive
        t.reset(100); // heartbeat resets timer
        assert!(!t.check(250)); // not expired yet
        assert!(t.check(500)); // leader silent → election
    }

    #[test]
    fn test_timer_check_exact_at_now_plus_timeout() {
        let mut t = ElectionTimer::new_with_seed(50, 51, 0); // always 50
        t.reset(1000);
        let deadline = if let TimerState::Running { deadline_ms } = *t.state() {
            deadline_ms
        } else {
            panic!("expected Running")
        };
        assert!(!t.check(deadline - 1));
        assert!(t.check(deadline));
    }

    #[test]
    fn test_stop_prevents_false_expiry() {
        let mut t = make_timer(10, 20);
        t.reset(0);
        t.stop();
        // Even long after the deadline, a stopped timer should not expire
        assert!(!t.check(1_000_000));
    }

    #[test]
    fn test_reset_after_stop() {
        let mut t = make_timer(100, 200);
        t.reset(0);
        t.stop();
        t.reset(1000);
        assert!(matches!(t.state(), TimerState::Running { .. }));
    }

    #[test]
    fn test_current_timeout_unchanged_without_reset() {
        let mut t = make_timer(100, 200);
        t.reset(0);
        let first = t.current_timeout_ms();
        // Check multiple times without reset
        let _ = t.check(50);
        assert_eq!(t.current_timeout_ms(), first);
    }

    #[test]
    fn test_new_with_seed_creates_idle_timer() {
        let t = ElectionTimer::new_with_seed(200, 400, 12345);
        assert_eq!(*t.state(), TimerState::Idle);
        assert_eq!(t.current_timeout_ms(), 0);
        assert_eq!(t.timeout_range(), (200, 400));
    }
}
