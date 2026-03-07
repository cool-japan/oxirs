//! Token-bucket rate limiting for SPARQL endpoints.
//!
//! Each client gets its own token bucket refilled at a constant rate.
//! An optional global bucket limits total throughput across all clients.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for a token-bucket rate limiter.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Sustained throughput in requests per second.
    pub requests_per_second: f64,
    /// Maximum burst size (initial / maximum token count).
    pub burst_size: usize,
    /// Extra wait in ms applied when a client is explicitly penalised.
    pub penalty_ms: u64,
}

impl RateLimitConfig {
    pub fn new(requests_per_second: f64, burst_size: usize) -> Self {
        Self {
            requests_per_second,
            burst_size,
            penalty_ms: 5_000,
        }
    }

    pub fn with_penalty(mut self, ms: u64) -> Self {
        self.penalty_ms = ms;
        self
    }
}

/// Sentinel value meaning the bucket has never been ticked.
const NEVER_REFILLED: u64 = u64::MAX;

/// Per-client bucket state.
#[derive(Debug, Clone)]
pub struct ClientRateState {
    pub client_id: String,
    /// Current token count (fractional).
    pub tokens: f64,
    /// Timestamp (ms) of the last refill, or `NEVER_REFILLED` if not yet initialised.
    pub last_refill: u64,
    /// Total requests that were allowed.
    pub total_requests: u64,
    /// Total requests that were rejected (throttled or penalised).
    pub rejected_requests: u64,
    /// If `Some(until_ms)`, the client is penalised until that timestamp.
    penalty_until_ms: Option<u64>,
    /// Reason for the current penalty.
    penalty_reason: Option<String>,
}

impl ClientRateState {
    fn new(client_id: &str, initial_tokens: f64) -> Self {
        Self {
            client_id: client_id.to_string(),
            tokens: initial_tokens,
            last_refill: NEVER_REFILLED,
            total_requests: 0,
            rejected_requests: 0,
            penalty_until_ms: None,
            penalty_reason: None,
        }
    }
}

/// Outcome of a rate-limit check.
#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitResult {
    /// Request is allowed; a token has been consumed.
    Allowed,
    /// Request is throttled; the bucket is empty.
    Throttled {
        /// Approximate ms until the next token becomes available.
        retry_after_ms: u64,
    },
    /// Client is currently in the penalty box.
    Penalized {
        reason: String,
        /// Remaining ms of the penalty.
        wait_ms: u64,
    },
}

/// Token-bucket rate limiter for SPARQL endpoints.
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    clients: HashMap<String, ClientRateState>,
    /// Global token bucket (if a global limit is configured).
    global_tokens: f64,
    global_last_refill: u64,
    global_limit: Option<RateLimitConfig>,
    /// Running totals across *all* clients.
    total_allowed: u64,
    total_rejected: u64,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            clients: HashMap::new(),
            global_tokens: 0.0,
            global_last_refill: NEVER_REFILLED,
            global_limit: None,
            total_allowed: 0,
            total_rejected: 0,
        }
    }

    /// Attach an optional global (server-wide) rate limit.
    pub fn with_global_limit(mut self, global: RateLimitConfig) -> Self {
        self.global_tokens = global.burst_size as f64;
        self.global_limit = Some(global);
        self
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn ensure_client(&mut self, client_id: &str) {
        self.clients.entry(client_id.to_string()).or_insert_with(|| {
            ClientRateState::new(client_id, self.config.burst_size as f64)
        });
    }

    /// Add tokens accrued since `last_refill` up to `burst_size`.
    ///
    /// On first call (`last_refill == NEVER_REFILLED`) the timestamp is seeded
    /// without adding tokens — the bucket was already pre-filled by the constructor.
    fn refill_bucket(
        tokens: &mut f64,
        last_refill: &mut u64,
        current_time_ms: u64,
        rps: f64,
        burst_size: usize,
    ) {
        if *last_refill == NEVER_REFILLED {
            // First call: just seed the timestamp; tokens were pre-initialised.
            *last_refill = current_time_ms;
            return;
        }
        let elapsed_s = (current_time_ms.saturating_sub(*last_refill)) as f64 / 1_000.0;
        let new_tokens = elapsed_s * rps;
        *tokens = (*tokens + new_tokens).min(burst_size as f64);
        *last_refill = current_time_ms;
    }

    // ── public API ───────────────────────────────────────────────────────────

    /// Check whether `client_id` may perform a request at `current_time_ms`.
    ///
    /// If allowed, one token is consumed from the per-client bucket (and the
    /// optional global bucket).  Counters are updated accordingly.
    pub fn check(&mut self, client_id: &str, current_time_ms: u64) -> RateLimitResult {
        self.ensure_client(client_id);
        self.refill(client_id, current_time_ms);

        // ── penalty check ────────────────────────────────────────────────────
        let penalty_result = {
            let state = self.clients.get(client_id).expect("just ensured");
            if let Some(until) = state.penalty_until_ms {
                if current_time_ms < until {
                    let wait = until - current_time_ms;
                    let reason = state
                        .penalty_reason
                        .clone()
                        .unwrap_or_else(|| "penalised".to_string());
                    Some(RateLimitResult::Penalized { reason, wait_ms: wait })
                } else {
                    None // penalty expired
                }
            } else {
                None
            }
        };

        if let Some(penalized) = penalty_result {
            let state = self.clients.get_mut(client_id).expect("just ensured");
            state.rejected_requests += 1;
            self.total_rejected += 1;
            return penalized;
        }

        // Clear expired penalty
        {
            let state = self.clients.get_mut(client_id).expect("just ensured");
            if state.penalty_until_ms.is_some() {
                state.penalty_until_ms = None;
                state.penalty_reason = None;
            }
        }

        // ── global limit check ───────────────────────────────────────────────
        if let Some(ref global_cfg) = self.global_limit.clone() {
            let rps = global_cfg.requests_per_second;
            let burst = global_cfg.burst_size;
            Self::refill_bucket(
                &mut self.global_tokens,
                &mut self.global_last_refill,
                current_time_ms,
                rps,
                burst,
            );
            if self.global_tokens < 1.0 {
                let retry_after_ms = ((1.0 - self.global_tokens) / rps * 1_000.0).ceil() as u64;
                let state = self.clients.get_mut(client_id).expect("just ensured");
                state.rejected_requests += 1;
                self.total_rejected += 1;
                return RateLimitResult::Throttled { retry_after_ms };
            }
        }

        // ── per-client bucket ────────────────────────────────────────────────
        let rps = self.config.requests_per_second;
        let (result, tokens_after) = {
            let state = self.clients.get(client_id).expect("just ensured");
            if state.tokens >= 1.0 {
                (RateLimitResult::Allowed, state.tokens - 1.0)
            } else {
                let retry = ((1.0 - state.tokens) / rps * 1_000.0).ceil() as u64;
                (RateLimitResult::Throttled { retry_after_ms: retry }, state.tokens)
            }
        };

        let state = self.clients.get_mut(client_id).expect("just ensured");
        state.tokens = tokens_after;

        if matches!(result, RateLimitResult::Allowed) {
            state.total_requests += 1;
            self.total_allowed += 1;
            if let Some(ref global_cfg) = self.global_limit.clone() {
                let _ = global_cfg; // borrow already ended
                self.global_tokens -= 1.0;
            }
        } else {
            state.rejected_requests += 1;
            self.total_rejected += 1;
        }
        result
    }

    /// Manually trigger a token refill for `client_id` at `current_time_ms`.
    pub fn refill(&mut self, client_id: &str, current_time_ms: u64) {
        self.ensure_client(client_id);
        let rps = self.config.requests_per_second;
        let burst = self.config.burst_size;
        let state = self.clients.get_mut(client_id).expect("just ensured");
        Self::refill_bucket(
            &mut state.tokens,
            &mut state.last_refill,
            current_time_ms,
            rps,
            burst,
        );
    }

    /// Reset a client's token bucket to `burst_size` and clear any penalty.
    pub fn reset_client(&mut self, client_id: &str) {
        if let Some(state) = self.clients.get_mut(client_id) {
            state.tokens = self.config.burst_size as f64;
            state.last_refill = NEVER_REFILLED;
            state.penalty_until_ms = None;
            state.penalty_reason = None;
        }
    }

    /// Remove a client from the limiter.  Returns `true` if the client existed.
    pub fn remove_client(&mut self, client_id: &str) -> bool {
        self.clients.remove(client_id).is_some()
    }

    /// Read-only view of a client's state.
    pub fn client_state(&self, client_id: &str) -> Option<&ClientRateState> {
        self.clients.get(client_id)
    }

    /// Number of tracked clients.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Place a client in the penalty box for `config.penalty_ms` milliseconds.
    pub fn penalize(&mut self, client_id: &str, reason: impl Into<String>) {
        self.ensure_client(client_id);
        let penalty_ms = self.config.penalty_ms;
        let state = self.clients.get_mut(client_id).expect("just ensured");
        // Penalty starts from the last known check time; if never checked, use 0 as epoch.
        let now = if state.last_refill == NEVER_REFILLED { 0 } else { state.last_refill };
        state.penalty_until_ms = Some(now + penalty_ms);
        state.penalty_reason = Some(reason.into());
    }

    /// Total requests allowed across all clients (since creation).
    pub fn total_allowed(&self) -> u64 {
        self.total_allowed
    }

    /// Total requests rejected across all clients (since creation).
    pub fn total_rejected(&self) -> u64 {
        self.total_rejected
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn limiter(rps: f64, burst: usize) -> RateLimiter {
        RateLimiter::new(RateLimitConfig::new(rps, burst))
    }

    // ── basic burst allowance ─────────────────────────────────────────────────

    #[test]
    fn test_burst_allows_burst_requests() {
        let mut rl = limiter(1.0, 5);
        // First 5 requests at t=0 should all be allowed (burst)
        for i in 0..5u64 {
            let res = rl.check("client1", i); // each call advances state marginally
            assert_eq!(res, RateLimitResult::Allowed, "request {} should be allowed", i);
        }
    }

    #[test]
    fn test_burst_exhaustion_throttles() {
        let mut rl = limiter(1.0, 3);
        for _ in 0..3 {
            let res = rl.check("c", 0);
            assert_eq!(res, RateLimitResult::Allowed);
        }
        // 4th request at same timestamp → throttled
        match rl.check("c", 0) {
            RateLimitResult::Throttled { .. } => {}
            other => panic!("expected Throttled, got {:?}", other),
        }
    }

    // ── token refill ──────────────────────────────────────────────────────────

    #[test]
    fn test_refill_over_time() {
        let mut rl = limiter(10.0, 10); // 10 rps, burst=10
        // Drain completely
        for _ in 0..10 {
            rl.check("c", 0);
        }
        assert!(matches!(rl.check("c", 0), RateLimitResult::Throttled { .. }));
        // After 1 second (1000 ms) we should have 10 new tokens
        for i in 0..10u64 {
            assert_eq!(rl.check("c", 1_000 + i), RateLimitResult::Allowed);
        }
    }

    #[test]
    fn test_refill_partial() {
        let mut rl = limiter(2.0, 4); // 2 rps
        // Drain completely at t=0
        for _ in 0..4 {
            rl.check("c", 0);
        }
        // At t=500ms (0.5s) we should have ~1 new token
        let res = rl.check("c", 500);
        assert_eq!(res, RateLimitResult::Allowed);
    }

    #[test]
    fn test_tokens_capped_at_burst() {
        let mut rl = limiter(10.0, 5); // burst=5
        // Wait a long time
        rl.check("c", 0); // seed last_refill
        rl.check("c", 1_000_000); // huge time elapsed → tokens capped at 5
        let state = rl.client_state("c").expect("state exists");
        assert!(state.tokens <= 5.0 + f64::EPSILON);
    }

    // ── retry_after_ms ────────────────────────────────────────────────────────

    #[test]
    fn test_retry_after_ms_positive_when_throttled() {
        let mut rl = limiter(1.0, 1);
        rl.check("c", 0); // consume the single token
        match rl.check("c", 0) {
            RateLimitResult::Throttled { retry_after_ms } => {
                assert!(retry_after_ms > 0, "retry_after_ms should be > 0");
                assert!(retry_after_ms <= 1_100, "should be at most ~1000ms");
            }
            other => panic!("expected Throttled, got {:?}", other),
        }
    }

    #[test]
    fn test_retry_after_ms_zero_not_returned() {
        let mut rl = limiter(1.0, 2);
        // Drain both tokens
        rl.check("c", 0);
        rl.check("c", 0);
        // Immediately after, no tokens → retry must be > 0
        match rl.check("c", 0) {
            RateLimitResult::Throttled { retry_after_ms } => {
                assert!(retry_after_ms > 0);
            }
            other => panic!("{:?}", other),
        }
    }

    // ── penalize ──────────────────────────────────────────────────────────────

    #[test]
    fn test_penalize_blocks_requests() {
        let mut rl = RateLimiter::new(RateLimitConfig::new(100.0, 100).with_penalty(5_000));
        rl.check("c", 1_000); // initialise
        rl.penalize("c", "abuse detected");
        // Penalty starts from last_refill (1000ms), lasts 5000ms → until 6000ms
        match rl.check("c", 2_000) {
            RateLimitResult::Penalized { reason, wait_ms } => {
                assert!(reason.contains("abuse"));
                assert!(wait_ms > 0);
            }
            other => panic!("expected Penalized, got {:?}", other),
        }
    }

    #[test]
    fn test_penalty_expires() {
        let mut rl = RateLimiter::new(RateLimitConfig::new(100.0, 100).with_penalty(1_000));
        rl.check("c", 0); // seed
        rl.penalize("c", "test");
        // After penalty window ends (> 1000ms from seed)
        let res = rl.check("c", 2_000);
        assert_eq!(res, RateLimitResult::Allowed);
    }

    #[test]
    fn test_penalize_nonexistent_client_creates_entry() {
        let mut rl = limiter(10.0, 10);
        rl.penalize("new_client", "preemptive");
        assert!(rl.client_state("new_client").is_some());
    }

    // ── reset_client ──────────────────────────────────────────────────────────

    #[test]
    fn test_reset_client_refills_tokens() {
        let mut rl = limiter(1.0, 3);
        // Drain
        for _ in 0..3 {
            rl.check("c", 0);
        }
        assert!(matches!(rl.check("c", 0), RateLimitResult::Throttled { .. }));
        rl.reset_client("c");
        assert_eq!(rl.check("c", 0), RateLimitResult::Allowed);
    }

    #[test]
    fn test_reset_client_clears_penalty() {
        let mut rl = RateLimiter::new(RateLimitConfig::new(100.0, 100).with_penalty(60_000));
        rl.check("c", 0);
        rl.penalize("c", "bad actor");
        rl.reset_client("c");
        assert_eq!(rl.check("c", 0), RateLimitResult::Allowed);
    }

    // ── remove_client ────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_client() {
        let mut rl = limiter(1.0, 1);
        rl.check("c", 0);
        assert!(rl.remove_client("c"));
        assert_eq!(rl.client_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_client_returns_false() {
        let mut rl = limiter(1.0, 1);
        assert!(!rl.remove_client("ghost"));
    }

    // ── client_count ─────────────────────────────────────────────────────────

    #[test]
    fn test_client_count_tracks_clients() {
        let mut rl = limiter(10.0, 10);
        rl.check("a", 0);
        rl.check("b", 0);
        rl.check("c", 0);
        assert_eq!(rl.client_count(), 3);
    }

    // ── global limit ─────────────────────────────────────────────────────────

    #[test]
    fn test_global_limit_blocks_when_exceeded() {
        let mut rl = RateLimiter::new(RateLimitConfig::new(1_000.0, 1_000))
            .with_global_limit(RateLimitConfig::new(2.0, 2));
        // Global burst = 2 → first two allowed, third throttled
        assert_eq!(rl.check("a", 0), RateLimitResult::Allowed);
        assert_eq!(rl.check("b", 0), RateLimitResult::Allowed);
        match rl.check("c", 0) {
            RateLimitResult::Throttled { .. } => {}
            other => panic!("expected global throttle, got {:?}", other),
        }
    }

    #[test]
    fn test_global_limit_refills_over_time() {
        let mut rl = RateLimiter::new(RateLimitConfig::new(1_000.0, 1_000))
            .with_global_limit(RateLimitConfig::new(1.0, 1));
        rl.check("a", 0);
        // After 1 second global token available again
        assert_eq!(rl.check("b", 1_000), RateLimitResult::Allowed);
    }

    // ── total_allowed / total_rejected ───────────────────────────────────────

    #[test]
    fn test_total_allowed_counter() {
        let mut rl = limiter(100.0, 10);
        for _ in 0..5 {
            rl.check("c", 0);
        }
        assert_eq!(rl.total_allowed(), 5);
    }

    #[test]
    fn test_total_rejected_counter() {
        let mut rl = limiter(1.0, 2);
        rl.check("c", 0);
        rl.check("c", 0);
        rl.check("c", 0); // rejected
        rl.check("c", 0); // rejected
        assert_eq!(rl.total_allowed(), 2);
        assert_eq!(rl.total_rejected(), 2);
    }

    #[test]
    fn test_total_counters_across_clients() {
        let mut rl = limiter(100.0, 3);
        rl.check("a", 0);
        rl.check("a", 0);
        rl.check("b", 0);
        assert_eq!(rl.total_allowed(), 3);
    }

    // ── client_state ─────────────────────────────────────────────────────────

    #[test]
    fn test_client_state_tracks_totals() {
        let mut rl = limiter(1.0, 2);
        rl.check("c", 0);
        rl.check("c", 0);
        rl.check("c", 0); // rejected
        let state = rl.client_state("c").expect("should exist");
        assert_eq!(state.total_requests, 2);
        assert_eq!(state.rejected_requests, 1);
    }

    #[test]
    fn test_client_state_nonexistent_returns_none() {
        let rl = limiter(1.0, 5);
        assert!(rl.client_state("ghost").is_none());
    }

    // ── explicit refill ───────────────────────────────────────────────────────

    #[test]
    fn test_explicit_refill_increases_tokens() {
        let mut rl = limiter(2.0, 4);
        for _ in 0..4 {
            rl.check("c", 0);
        }
        rl.refill("c", 1_000); // 2 tokens accrued
        let state = rl.client_state("c").expect("exists");
        assert!(state.tokens >= 1.9);
    }

    // ── independence of clients ───────────────────────────────────────────────

    #[test]
    fn test_clients_independent_buckets() {
        let mut rl = limiter(1.0, 2);
        rl.check("a", 0);
        rl.check("a", 0);
        // a is exhausted; b should still be full
        assert_eq!(rl.check("b", 0), RateLimitResult::Allowed);
        assert_eq!(rl.check("b", 0), RateLimitResult::Allowed);
    }

    #[test]
    fn test_many_clients_tracked() {
        let mut rl = limiter(10.0, 10);
        for i in 0..20u32 {
            rl.check(&format!("client_{}", i), 0);
        }
        assert_eq!(rl.client_count(), 20);
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_zero_burst_always_throttles() {
        let mut rl = RateLimiter::new(RateLimitConfig {
            requests_per_second: 1.0,
            burst_size: 0,
            penalty_ms: 1_000,
        });
        match rl.check("c", 0) {
            RateLimitResult::Throttled { .. } => {}
            other => panic!("expected Throttled for burst=0, got {:?}", other),
        }
    }

    #[test]
    fn test_high_rps_allows_many_requests() {
        let mut rl = limiter(1_000.0, 1_000);
        for i in 0..1_000u64 {
            assert_eq!(rl.check("c", i), RateLimitResult::Allowed);
        }
    }

    #[test]
    fn test_penalize_rejected_count_incremented() {
        let mut rl = RateLimiter::new(RateLimitConfig::new(100.0, 100).with_penalty(10_000));
        rl.check("c", 0);
        rl.penalize("c", "violation");
        rl.check("c", 1_000);
        let state = rl.client_state("c").expect("exists");
        assert!(state.rejected_requests >= 1);
    }

    #[test]
    fn test_total_rejected_counts_penalty_rejections() {
        let mut rl = RateLimiter::new(RateLimitConfig::new(100.0, 100).with_penalty(10_000));
        rl.check("c", 0);
        rl.penalize("c", "reason");
        rl.check("c", 500); // penalized → rejected
        assert!(rl.total_rejected() >= 1);
    }
}
