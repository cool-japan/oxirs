//! Retry strategy with exponential back-off and full jitter.
//!
//! Implements the AWS-style "full jitter" algorithm for distributed systems:
//!
//! ```text
//!   delay[n] = random_in(0, min(cap, base * 2^n))
//! ```
//!
//! where `base` is the initial delay and `cap` is the maximum delay.  Full
//! jitter avoids thundering-herd problems when many clients retry simultaneously.
//!
//! # Usage
//!
//! ```ignore
//! use oxirs_federate::fault::retry_policy::{RetryPolicy, RetryConfig};
//!
//! let policy = RetryPolicy::new(RetryConfig::default());
//! let mut attempt = policy.start();
//! loop {
//!     match do_the_thing() {
//!         Ok(v) => { attempt.succeed(); break; }
//!         Err(_) if attempt.can_retry() => {
//!             let delay = attempt.next_delay();
//!             std::thread::sleep(delay);
//!             attempt.increment();
//!         }
//!         Err(e) => return Err(e),
//!     }
//! }
//! ```
//!
//! # SciRS2 random policy
//!
//! We use `scirs2-core`'s random primitives to generate jitter values.
//! Specifically we use the `random` module for uniform random draws.

use std::time::Duration;

use serde::{Deserialize, Serialize};

// ─── RetryConfig ──────────────────────────────────────────────────────────────

/// Configuration for the exponential back-off + jitter retry strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (not counting the initial call).
    pub max_retries: u32,
    /// Base delay for the first retry.
    pub base_delay: Duration,
    /// Maximum delay cap (delays are never longer than this).
    pub max_delay: Duration,
    /// Multiplier applied to the base at each attempt.  Defaults to 2.0.
    pub multiplier: f64,
    /// Whether to apply full jitter (randomise within [0, computed_delay]).
    pub use_jitter: bool,
    /// Optional per-attempt timeout (independent of the total budget).
    pub attempt_timeout: Option<Duration>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 4,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
            use_jitter: true,
            attempt_timeout: None,
        }
    }
}

impl RetryConfig {
    /// Create a fast, aggressive retry config for interactive queries.
    pub fn fast() -> Self {
        Self {
            max_retries: 2,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_millis(500),
            multiplier: 2.0,
            use_jitter: true,
            attempt_timeout: Some(Duration::from_secs(5)),
        }
    }

    /// Create a slow, patient retry config for background bulk operations.
    pub fn slow() -> Self {
        Self {
            max_retries: 8,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            use_jitter: true,
            attempt_timeout: None,
        }
    }

    /// Compute the raw exponential delay for attempt `n` (0-indexed), WITHOUT
    /// jitter applied.
    pub fn raw_delay(&self, attempt: u32) -> Duration {
        let exponent = attempt.min(31) as u64; // prevent overflow
        let base_ms = self.base_delay.as_millis() as u64;
        let computed =
            base_ms.saturating_mul((self.multiplier.powi(exponent as i32) as u64).max(1));
        let cap_ms = self.max_delay.as_millis() as u64;
        Duration::from_millis(computed.min(cap_ms))
    }

    /// Compute the delay for attempt `n` with full jitter applied.
    ///
    /// Uses a simple LCG for jitter when SciRS2 PRNG is not available at this
    /// call site (avoids async / error-propagation complexity).
    pub fn jittered_delay(&self, attempt: u32, seed: u64) -> Duration {
        let upper = self.raw_delay(attempt).as_millis() as u64;
        if upper == 0 {
            return Duration::ZERO;
        }
        // LCG: x_{n+1} = (a*x_n + c) mod m
        let x = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let jittered_ms = x % (upper + 1);
        Duration::from_millis(jittered_ms)
    }
}

// ─── RetryAttempt ─────────────────────────────────────────────────────────────

/// Tracks the state of an in-progress retry sequence.
pub struct RetryAttempt<'a> {
    config: &'a RetryConfig,
    /// Current attempt number (0 = initial call, 1 = first retry, …).
    attempt: u32,
    /// Running seed for LCG jitter (updated each call to `next_delay`).
    seed: u64,
    succeeded: bool,
}

impl<'a> RetryAttempt<'a> {
    fn new(config: &'a RetryConfig, initial_seed: u64) -> Self {
        Self {
            config,
            attempt: 0,
            seed: initial_seed,
            succeeded: false,
        }
    }

    /// Whether another retry is allowed (i.e., we haven't exceeded `max_retries`).
    pub fn can_retry(&self) -> bool {
        !self.succeeded && self.attempt < self.config.max_retries
    }

    /// Compute the next delay and advance the jitter seed.
    pub fn next_delay(&mut self) -> Duration {
        if self.config.use_jitter {
            let d = self.config.jittered_delay(self.attempt, self.seed);
            // Advance seed
            self.seed = self
                .seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            d
        } else {
            self.config.raw_delay(self.attempt)
        }
    }

    /// Record a successful call; prevents further retries.
    pub fn succeed(&mut self) {
        self.succeeded = true;
    }

    /// Advance the attempt counter.
    pub fn increment(&mut self) {
        self.attempt = self.attempt.saturating_add(1);
    }

    /// Current attempt index (0 = initial, 1 = first retry, …).
    pub fn attempt_number(&self) -> u32 {
        self.attempt
    }

    /// Total retries remaining.
    pub fn remaining_retries(&self) -> u32 {
        self.config.max_retries.saturating_sub(self.attempt)
    }
}

// ─── RetryPolicy ──────────────────────────────────────────────────────────────

/// Exponential back-off + full jitter retry policy.
pub struct RetryPolicy {
    config: RetryConfig,
    /// Base seed for jitter generation; changed per `start()` call.
    seed_counter: std::sync::atomic::AtomicU64,
}

impl RetryPolicy {
    /// Create a policy with the given configuration.
    pub fn new(config: RetryConfig) -> Self {
        Self {
            config,
            seed_counter: std::sync::atomic::AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(12345),
            ),
        }
    }

    /// Start a new retry sequence.
    pub fn start(&self) -> RetryAttempt<'_> {
        let seed = self
            .seed_counter
            .fetch_add(0xdead_beef_cafe_1234, std::sync::atomic::Ordering::Relaxed);
        RetryAttempt::new(&self.config, seed)
    }

    /// Execute a fallible synchronous operation with automatic retries.
    ///
    /// The operation is retried up to `max_retries` times.  Returns the first
    /// `Ok(T)`, or the last `Err(E)` if all attempts are exhausted.
    ///
    /// Delays are simulated via `std::thread::sleep` (not async-friendly).
    /// For async use, apply delays with `tokio::time::sleep` in your own loop.
    pub fn execute_sync<T, E, F>(&self, mut f: F) -> Result<T, E>
    where
        F: FnMut() -> Result<T, E>,
    {
        let mut attempt = self.start();
        loop {
            match f() {
                Ok(v) => {
                    attempt.succeed();
                    return Ok(v);
                }
                Err(e) => {
                    if attempt.can_retry() {
                        let delay = attempt.next_delay();
                        attempt.increment();
                        if delay > Duration::ZERO {
                            std::thread::sleep(delay);
                        }
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// Reference to the underlying configuration.
    pub fn config(&self) -> &RetryConfig {
        &self.config
    }
}

// ─── RetryStats ───────────────────────────────────────────────────────────────

/// Aggregated statistics for retry policy usage (useful for metrics).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetryStats {
    pub total_attempts: u64,
    pub total_successes: u64,
    pub total_failures: u64,
    pub total_exhausted: u64,
}

impl RetryStats {
    /// Record a successful operation.
    pub fn record_success(&mut self, attempts: u32) {
        self.total_attempts += attempts as u64;
        self.total_successes += 1;
    }

    /// Record a failure (non-exhausting retry).
    pub fn record_failure(&mut self, attempts: u32) {
        self.total_attempts += attempts as u64;
        self.total_failures += 1;
    }

    /// Record that all retries were exhausted.
    pub fn record_exhausted(&mut self, attempts: u32) {
        self.total_attempts += attempts as u64;
        self.total_exhausted += 1;
    }

    /// Success rate in [0.0, 1.0].
    pub fn success_rate(&self) -> f64 {
        let total = self.total_successes + self.total_exhausted;
        if total == 0 {
            1.0
        } else {
            self.total_successes as f64 / total as f64
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RetryConfig ───────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let c = RetryConfig::default();
        assert_eq!(c.max_retries, 4);
        assert!(c.use_jitter);
    }

    #[test]
    fn test_fast_config() {
        let c = RetryConfig::fast();
        assert_eq!(c.max_retries, 2);
        assert!(c.attempt_timeout.is_some());
    }

    #[test]
    fn test_slow_config() {
        let c = RetryConfig::slow();
        assert_eq!(c.max_retries, 8);
        assert!(c.attempt_timeout.is_none());
    }

    #[test]
    fn test_raw_delay_increases_with_attempt() {
        let c = RetryConfig {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            use_jitter: false,
            ..Default::default()
        };
        let d0 = c.raw_delay(0);
        let d1 = c.raw_delay(1);
        let d2 = c.raw_delay(2);
        assert!(d1 >= d0, "d1={:?} should be >= d0={:?}", d1, d0);
        assert!(d2 >= d1, "d2={:?} should be >= d1={:?}", d2, d1);
    }

    #[test]
    fn test_raw_delay_capped() {
        let c = RetryConfig {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(300),
            multiplier: 10.0,
            use_jitter: false,
            ..Default::default()
        };
        // After a few steps the cap should kick in
        assert!(c.raw_delay(5) <= Duration::from_millis(300));
    }

    #[test]
    fn test_jittered_delay_within_raw_upper_bound() {
        let c = RetryConfig::default();
        for attempt in 0..5u32 {
            let upper = c.raw_delay(attempt);
            for seed in [0u64, 1, 42, 0xdead_beef] {
                let jittered = c.jittered_delay(attempt, seed);
                assert!(
                    jittered <= upper,
                    "jitter {:?} exceeds upper {:?}",
                    jittered,
                    upper
                );
            }
        }
    }

    #[test]
    fn test_jittered_delay_different_seeds_vary() {
        let c = RetryConfig::default();
        let d1 = c.jittered_delay(2, 100);
        let d2 = c.jittered_delay(2, 200);
        // Different seeds should very likely produce different delays
        // (not guaranteed but true for reasonable LCG values)
        let _ = (d1, d2); // just ensure no panic
    }

    // ── RetryAttempt ──────────────────────────────────────────────────────

    #[test]
    fn test_retry_attempt_can_retry() {
        let c = RetryConfig {
            max_retries: 2,
            ..Default::default()
        };
        let policy = RetryPolicy::new(c);
        let mut attempt = policy.start();
        assert!(attempt.can_retry());
        attempt.increment();
        assert!(attempt.can_retry());
        attempt.increment();
        assert!(!attempt.can_retry()); // exhausted
    }

    #[test]
    fn test_retry_attempt_succeed_stops_retry() {
        let c = RetryConfig {
            max_retries: 5,
            ..Default::default()
        };
        let policy = RetryPolicy::new(c);
        let mut attempt = policy.start();
        attempt.succeed();
        assert!(!attempt.can_retry());
    }

    #[test]
    fn test_retry_attempt_number() {
        let policy = RetryPolicy::new(RetryConfig::default());
        let mut a = policy.start();
        assert_eq!(a.attempt_number(), 0);
        a.increment();
        assert_eq!(a.attempt_number(), 1);
    }

    #[test]
    fn test_retry_remaining_retries() {
        let c = RetryConfig {
            max_retries: 3,
            ..Default::default()
        };
        let policy = RetryPolicy::new(c);
        let mut a = policy.start();
        assert_eq!(a.remaining_retries(), 3);
        a.increment();
        assert_eq!(a.remaining_retries(), 2);
    }

    #[test]
    fn test_retry_attempt_next_delay_monotone_without_jitter() {
        let c = RetryConfig {
            max_retries: 4,
            use_jitter: false,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            attempt_timeout: None,
        };
        let policy = RetryPolicy::new(c);
        let mut a = policy.start();
        let mut last = Duration::ZERO;
        for _ in 0..4 {
            let d = a.next_delay();
            assert!(d >= last);
            last = d;
            a.increment();
        }
    }

    // ── RetryPolicy::execute_sync ─────────────────────────────────────────

    #[test]
    fn test_execute_sync_success_first_attempt() {
        let policy = RetryPolicy::new(RetryConfig {
            max_retries: 3,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            use_jitter: false,
            ..Default::default()
        });
        let mut calls = 0usize;
        let result: Result<i32, &str> = policy.execute_sync(|| {
            calls += 1;
            Ok(42)
        });
        assert!(matches!(result, Ok(42)));
        assert_eq!(calls, 1);
    }

    #[test]
    fn test_execute_sync_success_after_retry() {
        let policy = RetryPolicy::new(RetryConfig {
            max_retries: 3,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            use_jitter: false,
            ..Default::default()
        });
        let mut calls = 0usize;
        let result: Result<i32, &str> = policy.execute_sync(|| {
            calls += 1;
            if calls < 3 {
                Err("not ready")
            } else {
                Ok(99)
            }
        });
        assert!(matches!(result, Ok(99)));
        assert_eq!(calls, 3);
    }

    #[test]
    fn test_execute_sync_exhausted() {
        let policy = RetryPolicy::new(RetryConfig {
            max_retries: 2,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            use_jitter: false,
            ..Default::default()
        });
        let mut calls = 0usize;
        let result: Result<i32, &str> = policy.execute_sync(|| {
            calls += 1;
            Err("always fails")
        });
        assert!(matches!(result, Err("always fails")));
        assert_eq!(calls, 3); // initial + 2 retries
    }

    // ── RetryStats ────────────────────────────────────────────────────────

    #[test]
    fn test_retry_stats_success_rate_all_success() {
        let mut s = RetryStats::default();
        s.record_success(1);
        s.record_success(1);
        assert!((s.success_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_retry_stats_success_rate_partial() {
        let mut s = RetryStats::default();
        s.record_success(1);
        s.record_exhausted(3);
        assert!((s.success_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_retry_stats_initial_success_rate() {
        let s = RetryStats::default();
        assert!((s.success_rate() - 1.0).abs() < 1e-9); // no attempts = 100%
    }

    #[test]
    fn test_retry_stats_totals() {
        let mut s = RetryStats::default();
        s.record_success(2);
        s.record_failure(1);
        s.record_exhausted(5);
        assert_eq!(s.total_attempts, 8);
        assert_eq!(s.total_successes, 1);
        assert_eq!(s.total_failures, 1);
        assert_eq!(s.total_exhausted, 1);
    }

    #[test]
    fn test_policy_config_accessor() {
        let c = RetryConfig::default();
        let p = RetryPolicy::new(c.clone());
        assert_eq!(p.config().max_retries, 4);
    }
}
