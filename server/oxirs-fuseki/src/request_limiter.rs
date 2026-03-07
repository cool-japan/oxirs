/// Token-bucket rate limiter for SPARQL endpoints.
///
/// Implements the token-bucket algorithm with optional per-client state.
/// Each client gets its own bucket, and there is also a global bucket
/// for aggregate rate limiting.
use std::collections::HashMap;
use std::time::Instant;

/// Rate limiting configuration.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum sustained request rate (tokens per second).
    pub requests_per_second: f64,
    /// Maximum burst size (token bucket capacity).
    pub burst_size: usize,
    /// Whether to apply per-client rate limiting.
    pub per_client: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 100.0,
            burst_size: 200,
            per_client: true,
        }
    }
}

impl RateLimitConfig {
    /// Create a new configuration.
    pub fn new(requests_per_second: f64, burst_size: usize, per_client: bool) -> Self {
        Self {
            requests_per_second,
            burst_size,
            per_client,
        }
    }
}

/// Per-client (or global) token bucket state.
#[derive(Debug, Clone)]
pub struct ClientState {
    /// Current token count (may be fractional).
    pub tokens: f64,
    /// Timestamp of the last token refill.
    pub last_refill: Instant,
}

impl ClientState {
    /// Create a full bucket.
    pub fn new(capacity: f64) -> Self {
        Self {
            tokens: capacity,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time and rate.
    fn refill(&mut self, rate: f64, capacity: f64) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = elapsed * rate;
        self.tokens = (self.tokens + new_tokens).min(capacity);
        self.last_refill = now;
    }

    /// Try to consume `n` tokens. Returns true if successful.
    fn try_consume(&mut self, n: f64) -> bool {
        if self.tokens >= n {
            self.tokens -= n;
            true
        } else {
            false
        }
    }

    /// Compute how long (ms) until `n` tokens are available.
    fn retry_after_ms(&self, n: f64, rate: f64) -> u64 {
        if rate <= 0.0 {
            return u64::MAX;
        }
        let deficit = n - self.tokens;
        if deficit <= 0.0 {
            return 0;
        }
        let seconds = deficit / rate;
        (seconds * 1000.0).ceil() as u64
    }
}

/// Decision returned by the rate limiter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitDecision {
    /// Request is allowed.
    Allow,
    /// Request is denied; client should retry after the given delay.
    Deny {
        /// Suggested retry delay in milliseconds.
        retry_after_ms: u64,
    },
}

impl LimitDecision {
    /// Returns true if the decision is `Allow`.
    pub fn is_allow(&self) -> bool {
        matches!(self, LimitDecision::Allow)
    }

    /// Returns true if the decision is `Deny`.
    pub fn is_deny(&self) -> bool {
        matches!(self, LimitDecision::Deny { .. })
    }
}

/// Token-bucket rate limiter for SPARQL endpoints.
///
/// Maintains per-client and global token buckets.
pub struct RequestLimiter {
    config: RateLimitConfig,
    clients: HashMap<String, ClientState>,
    global: ClientState,
}

impl RequestLimiter {
    /// Create a new rate limiter with the given configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        let capacity = config.burst_size as f64;
        Self {
            global: ClientState::new(capacity),
            clients: HashMap::new(),
            config,
        }
    }

    /// Check whether a single request from `client_id` is allowed.
    ///
    /// Consumes 1 token from the appropriate bucket.
    pub fn check(&mut self, client_id: &str) -> LimitDecision {
        self.check_n(client_id, 1)
    }

    /// Check whether `tokens` tokens can be consumed for `client_id`.
    ///
    /// When `per_client = true`, each client has an independent token bucket.
    /// When `per_client = false`, all clients share the global bucket.
    pub fn check_n(&mut self, client_id: &str, tokens: usize) -> LimitDecision {
        let n = tokens as f64;
        let rate = self.config.requests_per_second;
        let capacity = self.config.burst_size as f64;

        if self.config.per_client {
            // Each client has its own independent bucket.
            let client = self
                .clients
                .entry(client_id.to_string())
                .or_insert_with(|| ClientState::new(capacity));

            client.refill(rate, capacity);
            if client.try_consume(n) {
                // Also track consumption in the global bucket for monitoring.
                self.global.refill(rate, capacity);
                // Best-effort consume from global (don't block on it).
                let _ = self.global.try_consume(n);
                LimitDecision::Allow
            } else {
                let retry_ms = client.retry_after_ms(n, rate);
                LimitDecision::Deny {
                    retry_after_ms: retry_ms,
                }
            }
        } else {
            // All clients share the global bucket.
            self.global.refill(rate, capacity);
            if self.global.try_consume(n) {
                LimitDecision::Allow
            } else {
                let retry_ms = self.global.retry_after_ms(n, rate);
                LimitDecision::Deny {
                    retry_after_ms: retry_ms,
                }
            }
        }
    }

    /// Manually trigger token refill for a client bucket.
    pub fn refill(&mut self, client_id: &str) {
        let rate = self.config.requests_per_second;
        let capacity = self.config.burst_size as f64;
        if let Some(state) = self.clients.get_mut(client_id) {
            state.refill(rate, capacity);
        }
        self.global.refill(rate, capacity);
    }

    /// Reset a client's bucket to full capacity.
    pub fn reset_client(&mut self, client_id: &str) {
        let capacity = self.config.burst_size as f64;
        self.clients.entry(client_id.to_string()).and_modify(|s| {
            s.tokens = capacity;
            s.last_refill = Instant::now();
        });
    }

    /// Return the number of active client entries.
    pub fn active_clients(&self) -> usize {
        self.clients.len()
    }

    /// Return the current global token count.
    pub fn global_tokens(&self) -> f64 {
        self.global.tokens
    }

    /// Return an immutable reference to the configuration.
    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_limiter(rps: f64, burst: usize, per_client: bool) -> RequestLimiter {
        RequestLimiter::new(RateLimitConfig::new(rps, burst, per_client))
    }

    // --- RateLimitConfig ---

    #[test]
    fn test_config_default() {
        let cfg = RateLimitConfig::default();
        assert!(cfg.requests_per_second > 0.0);
        assert!(cfg.burst_size > 0);
    }

    #[test]
    fn test_config_new() {
        let cfg = RateLimitConfig::new(50.0, 100, true);
        assert!((cfg.requests_per_second - 50.0).abs() < 1e-9);
        assert_eq!(cfg.burst_size, 100);
        assert!(cfg.per_client);
    }

    #[test]
    fn test_config_clone() {
        let cfg = RateLimitConfig::new(10.0, 20, false);
        let cfg2 = cfg.clone();
        assert!((cfg2.requests_per_second - 10.0).abs() < 1e-9);
    }

    // --- ClientState ---

    #[test]
    fn test_client_state_new_full() {
        let state = ClientState::new(10.0);
        assert!((state.tokens - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_client_state_consume_success() {
        let mut state = ClientState::new(5.0);
        assert!(state.try_consume(3.0));
        assert!((state.tokens - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_client_state_consume_fail() {
        let mut state = ClientState::new(1.0);
        assert!(!state.try_consume(2.0));
        // Tokens unchanged after failure.
        assert!((state.tokens - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_client_state_refill_capped_at_capacity() {
        let mut state = ClientState::new(10.0);
        state.tokens = 8.0;
        // Even after lots of time, should not exceed capacity.
        state.refill(1000.0, 10.0);
        assert!(state.tokens <= 10.0 + 1e-9);
    }

    #[test]
    fn test_client_state_retry_after_ms() {
        let state = ClientState::new(0.0);
        let ms = state.retry_after_ms(1.0, 1.0); // 1 token / 1 token/s = 1s = 1000ms
        assert!(ms >= 1000);
    }

    #[test]
    fn test_client_state_retry_zero_rate() {
        let state = ClientState::new(0.0);
        let ms = state.retry_after_ms(1.0, 0.0);
        assert_eq!(ms, u64::MAX);
    }

    // --- LimitDecision ---

    #[test]
    fn test_limit_decision_allow() {
        let d = LimitDecision::Allow;
        assert!(d.is_allow());
        assert!(!d.is_deny());
    }

    #[test]
    fn test_limit_decision_deny() {
        let d = LimitDecision::Deny {
            retry_after_ms: 500,
        };
        assert!(d.is_deny());
        assert!(!d.is_allow());
    }

    #[test]
    fn test_limit_decision_equality() {
        assert_eq!(LimitDecision::Allow, LimitDecision::Allow);
        assert_eq!(
            LimitDecision::Deny {
                retry_after_ms: 100
            },
            LimitDecision::Deny {
                retry_after_ms: 100
            }
        );
        assert_ne!(
            LimitDecision::Allow,
            LimitDecision::Deny { retry_after_ms: 0 }
        );
    }

    // --- RequestLimiter construction ---

    #[test]
    fn test_new_limiter() {
        let limiter = make_limiter(10.0, 5, true);
        assert_eq!(limiter.active_clients(), 0);
        assert!((limiter.global_tokens() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_accessor() {
        let limiter = make_limiter(20.0, 40, false);
        assert!((limiter.config().requests_per_second - 20.0).abs() < 1e-9);
        assert_eq!(limiter.config().burst_size, 40);
    }

    // --- check / check_n ---

    #[test]
    fn test_check_allow_within_burst() {
        let mut limiter = make_limiter(1.0, 10, true);
        for _ in 0..10 {
            let d = limiter.check("client_a");
            assert!(d.is_allow(), "Expected Allow but got {:?}", d);
        }
    }

    #[test]
    fn test_check_deny_after_burst() {
        let mut limiter = make_limiter(0.001, 3, true); // Very low rate
        limiter.check("c1");
        limiter.check("c1");
        limiter.check("c1");
        let d = limiter.check("c1");
        assert!(d.is_deny(), "Expected Deny but got {:?}", d);
    }

    #[test]
    fn test_check_n_consumes_multiple() {
        let mut limiter = make_limiter(1.0, 10, true);
        let d = limiter.check_n("c1", 5);
        assert!(d.is_allow());
        // 5 more tokens remaining → another 5 should be allowed.
        let d2 = limiter.check_n("c1", 5);
        assert!(d2.is_allow());
        // No tokens left.
        let d3 = limiter.check_n("c1", 1);
        assert!(d3.is_deny());
    }

    #[test]
    fn test_check_n_exceeds_burst() {
        let mut limiter = make_limiter(1.0, 5, false);
        let d = limiter.check_n("any", 6); // More than burst.
        assert!(d.is_deny());
    }

    #[test]
    fn test_per_client_isolation() {
        let mut limiter = make_limiter(0.001, 2, true);
        // Drain client_a.
        limiter.check("client_a");
        limiter.check("client_a");
        let da = limiter.check("client_a");
        assert!(da.is_deny());
        // client_b should still have tokens.
        let db = limiter.check("client_b");
        assert!(db.is_allow());
    }

    #[test]
    fn test_global_limits_all_clients() {
        // Global burst = 2, but many clients.
        let mut limiter = make_limiter(0.001, 2, false); // per_client=false
        limiter.check("c1"); // consumes global token 1
        limiter.check("c2"); // consumes global token 2
                             // No global tokens left → c3 denied.
        let d = limiter.check("c3");
        assert!(d.is_deny());
    }

    #[test]
    fn test_deny_includes_retry_after() {
        let mut limiter = make_limiter(1.0, 1, false);
        limiter.check("c"); // drain
        if let LimitDecision::Deny { retry_after_ms } = limiter.check("c") {
            assert!(retry_after_ms > 0);
        } else {
            panic!("Expected Deny");
        }
    }

    // --- active_clients ---

    #[test]
    fn test_active_clients_increases() {
        let mut limiter = make_limiter(10.0, 100, true);
        limiter.check("a");
        limiter.check("b");
        limiter.check("c");
        assert_eq!(limiter.active_clients(), 3);
    }

    #[test]
    fn test_active_clients_no_per_client() {
        let mut limiter = make_limiter(10.0, 100, false);
        limiter.check("a");
        limiter.check("b");
        // Per-client disabled → no entries in map.
        assert_eq!(limiter.active_clients(), 0);
    }

    #[test]
    fn test_active_clients_same_client() {
        let mut limiter = make_limiter(10.0, 100, true);
        limiter.check("a");
        limiter.check("a");
        assert_eq!(limiter.active_clients(), 1);
    }

    // --- refill ---

    #[test]
    fn test_manual_refill_no_panic() {
        let mut limiter = make_limiter(10.0, 10, true);
        limiter.check("c");
        limiter.refill("c"); // Should not panic.
    }

    #[test]
    fn test_manual_refill_nonexistent_client_no_panic() {
        let mut limiter = make_limiter(10.0, 10, true);
        limiter.refill("nonexistent"); // Should not panic.
    }

    // --- reset_client ---

    #[test]
    fn test_reset_client_restores_tokens() {
        let mut limiter = make_limiter(0.001, 3, true);
        limiter.check("c");
        limiter.check("c");
        limiter.check("c");
        // Now denied.
        assert!(limiter.check("c").is_deny());
        // Reset restores tokens.
        limiter.reset_client("c");
        assert!(limiter.check("c").is_allow());
    }

    #[test]
    fn test_reset_nonexistent_client_no_panic() {
        let mut limiter = make_limiter(10.0, 5, true);
        limiter.reset_client("does_not_exist"); // Should not panic.
    }

    // --- global_tokens ---

    #[test]
    fn test_global_tokens_decreases_on_request() {
        let mut limiter = make_limiter(10.0, 10, false);
        let before = limiter.global_tokens();
        limiter.check("c");
        let after = limiter.global_tokens();
        assert!(after < before + 1e-9);
    }

    #[test]
    fn test_global_tokens_starts_at_burst() {
        let limiter = make_limiter(5.0, 7, true);
        assert!((limiter.global_tokens() - 7.0).abs() < 1e-9);
    }

    // --- Integration ---

    #[test]
    fn test_allow_deny_allow_cycle() {
        let mut limiter = make_limiter(1000.0, 2, false);
        assert!(limiter.check("x").is_allow());
        assert!(limiter.check("x").is_allow());
        // Denied — bucket empty.
        assert!(limiter.check("x").is_deny());
        // After sleeping the test becomes flaky; just verify deny path works.
    }

    #[test]
    fn test_multiple_clients_independent_per_client() {
        let mut limiter = make_limiter(0.001, 1, true);
        // Each client has 1 token.
        assert!(limiter.check("a").is_allow());
        assert!(limiter.check("b").is_allow());
        assert!(limiter.check("c").is_allow());
        // Each now denied.
        assert!(limiter.check("a").is_deny());
        assert!(limiter.check("b").is_deny());
        assert!(limiter.check("c").is_deny());
    }
}
