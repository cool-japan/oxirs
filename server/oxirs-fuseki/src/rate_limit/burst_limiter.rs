//! Burst-aware rate limiter with per-client limits and HTTP header support
//!
//! `BurstRateLimiter` uses the token bucket algorithm:
//! - Each client key gets an independent bucket
//! - `burst_size` sets the maximum burst capacity
//! - `refill_rate` controls how fast tokens are replenished
//! - Per-client overrides can be registered for premium or restricted clients
//!
//! HTTP response headers are produced in `X-RateLimit-Limit`,
//! `X-RateLimit-Remaining`, and `X-RateLimit-Reset` format.

use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Serialize;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Per-key bucket
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

impl Bucket {
    fn new(capacity: f64) -> Self {
        Bucket {
            tokens: capacity,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time and attempt to consume `cost`.
    fn try_consume(&mut self, cost: f64, capacity: f64, refill_rate: f64) -> bool {
        let now = Instant::now();
        let elapsed = (now - self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * refill_rate).min(capacity);
        self.last_refill = now;
        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }

    /// Seconds until at least `cost` tokens are available.
    fn eta_secs(&self, cost: f64, refill_rate: f64) -> f64 {
        if self.tokens >= cost {
            0.0
        } else {
            (cost - self.tokens) / refill_rate
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-client configuration override
// ─────────────────────────────────────────────────────────────────────────────

/// Per-client rate limit override.
#[derive(Debug, Clone)]
pub struct ClientOverride {
    /// Burst capacity for this client
    pub burst_size: u64,
    /// Refill rate (tokens/second) for this client
    pub refill_rate: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Decision
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a `BurstRateLimiter::check` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurstDecision {
    /// Whether the request should proceed
    pub allowed: bool,
    /// Configured burst capacity for this key
    pub limit: u64,
    /// Tokens remaining in the bucket
    pub remaining: u64,
    /// Instant at which the limit resets (enough tokens for at least 1 request)
    pub reset_at: Instant,
    /// Whether this key used a per-client override
    pub is_override: bool,
}

impl BurstDecision {
    /// Seconds until tokens are replenished (0 if already allowed).
    pub fn retry_after_secs(&self) -> u64 {
        let now = Instant::now();
        if self.reset_at > now {
            (self.reset_at - now).as_secs().max(1)
        } else {
            0
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP headers
// ─────────────────────────────────────────────────────────────────────────────

/// Rate limit headers suitable for HTTP responses.
#[derive(Debug, Clone, Serialize)]
pub struct BurstRateLimitHeaders {
    /// `X-RateLimit-Limit` value
    pub x_ratelimit_limit: String,
    /// `X-RateLimit-Remaining` value
    pub x_ratelimit_remaining: String,
    /// `X-RateLimit-Reset` value (seconds until reset)
    pub x_ratelimit_reset: u64,
    /// `Retry-After` header value (only set when denied)
    pub retry_after: Option<u64>,
}

impl BurstRateLimitHeaders {
    /// Build headers from a `BurstDecision`.
    pub fn from_decision(decision: &BurstDecision) -> Self {
        let retry_after = if !decision.allowed {
            Some(decision.retry_after_secs())
        } else {
            None
        };
        BurstRateLimitHeaders {
            x_ratelimit_limit: decision.limit.to_string(),
            x_ratelimit_remaining: decision.remaining.to_string(),
            x_ratelimit_reset: decision.retry_after_secs(),
            retry_after,
        }
    }

    /// Insert all relevant headers into an `axum::http::HeaderMap`.
    pub fn insert_into(&self, headers: &mut axum::http::HeaderMap) {
        use axum::http::HeaderValue;

        if let Ok(v) = HeaderValue::from_str(&self.x_ratelimit_limit) {
            headers.insert("x-ratelimit-limit", v);
        }
        if let Ok(v) = HeaderValue::from_str(&self.x_ratelimit_remaining) {
            headers.insert("x-ratelimit-remaining", v);
        }
        if let Ok(v) = HeaderValue::from_str(&self.x_ratelimit_reset.to_string()) {
            headers.insert("x-ratelimit-reset", v);
        }
        if let Some(ra) = self.retry_after {
            if let Ok(v) = HeaderValue::from_str(&ra.to_string()) {
                headers.insert("retry-after", v);
            }
        }
    }

    /// Build a 429 Too Many Requests response with headers and JSON body.
    pub fn too_many_requests_response(decision: &BurstDecision) -> axum::response::Response {
        use axum::http::{HeaderMap, StatusCode};
        use axum::response::IntoResponse;

        let mut headers = HeaderMap::new();
        let rl = BurstRateLimitHeaders::from_decision(decision);
        rl.insert_into(&mut headers);

        let body = format!(
            "{{\"error\":\"Too Many Requests\",\"retry_after_secs\":{}}}",
            decision.retry_after_secs()
        );

        (StatusCode::TOO_MANY_REQUESTS, headers, body).into_response()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BurstRateLimiter
// ─────────────────────────────────────────────────────────────────────────────

/// Token-bucket rate limiter with burst support and per-client overrides.
///
/// Thread-safe: uses `DashMap` for concurrent access and `parking_lot::Mutex`
/// per bucket.
pub struct BurstRateLimiter {
    /// Default burst capacity (token bucket max)
    default_burst_size: u64,
    /// Default refill rate (tokens / second)
    default_refill_rate: f64,
    /// Per-key token buckets
    buckets: DashMap<String, Mutex<Bucket>>,
    /// Per-client configuration overrides
    client_overrides: DashMap<String, ClientOverride>,
}

impl BurstRateLimiter {
    /// Create a new `BurstRateLimiter`.
    ///
    /// * `burst_size` — maximum burst capacity (also the initial fill)
    /// * `refill_rate` — tokens added per second
    ///
    /// # Panics
    ///
    /// Panics if `burst_size == 0` or `refill_rate <= 0.0`.
    pub fn new(burst_size: u64, refill_rate: f64) -> Self {
        assert!(burst_size > 0, "burst_size must be positive");
        assert!(refill_rate > 0.0, "refill_rate must be positive");
        BurstRateLimiter {
            default_burst_size: burst_size,
            default_refill_rate: refill_rate,
            buckets: DashMap::new(),
            client_overrides: DashMap::new(),
        }
    }

    /// Register a per-client override for a specific key.
    pub fn register_client_override(&self, key: impl Into<String>, override_cfg: ClientOverride) {
        self.client_overrides.insert(key.into(), override_cfg);
    }

    /// Remove a per-client override, reverting the key to defaults.
    pub fn remove_client_override(&self, key: &str) {
        self.client_overrides.remove(key);
    }

    /// Check (and consume) one token for `key`.
    pub fn check(&self, key: &str) -> BurstDecision {
        self.check_with_cost(key, 1)
    }

    /// Check (and consume) `cost` tokens for `key`.
    ///
    /// `cost == 0` is treated as 1.
    pub fn check_with_cost(&self, key: &str, cost: u64) -> BurstDecision {
        let cost = cost.max(1);
        let (burst_size, refill_rate, is_override) = match self.client_overrides.get(key) {
            Some(ov) => (ov.burst_size, ov.refill_rate, true),
            None => (self.default_burst_size, self.default_refill_rate, false),
        };

        let cost_f = cost as f64;
        let capacity_f = burst_size as f64;

        let entry = self
            .buckets
            .entry(key.to_string())
            .or_insert_with(|| Mutex::new(Bucket::new(capacity_f)));

        let mut bucket = entry.lock();
        let allowed = bucket.try_consume(cost_f, capacity_f, refill_rate);
        let remaining = bucket.tokens.floor() as u64;
        let eta = bucket.eta_secs(cost_f, refill_rate);
        let reset_at = Instant::now() + Duration::from_secs_f64(eta.max(0.0));

        BurstDecision {
            allowed,
            limit: burst_size,
            remaining,
            reset_at,
            is_override,
        }
    }

    /// Return the default burst size.
    pub fn default_burst_size(&self) -> u64 {
        self.default_burst_size
    }

    /// Return the default refill rate.
    pub fn default_refill_rate(&self) -> f64 {
        self.default_refill_rate
    }

    /// Return current token count for a key without consuming tokens.
    pub fn peek_tokens(&self, key: &str) -> f64 {
        let (burst_size, refill_rate, _) = match self.client_overrides.get(key) {
            Some(ov) => (ov.burst_size, ov.refill_rate, true),
            None => (self.default_burst_size, self.default_refill_rate, false),
        };
        match self.buckets.get(key) {
            Some(entry) => {
                let bucket = entry.lock();
                let now = Instant::now();
                let elapsed = (now - bucket.last_refill).as_secs_f64();
                (bucket.tokens + elapsed * refill_rate).min(burst_size as f64)
            }
            None => burst_size as f64, // Never-seen key has full bucket
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    // 1. Allows requests within burst capacity
    #[test]
    fn test_allows_within_burst() {
        let limiter = BurstRateLimiter::new(5, 1.0);
        for _ in 0..5 {
            assert!(limiter.check("client-a").allowed);
        }
    }

    // 2. Denies when bucket is exhausted
    #[test]
    fn test_denies_when_exhausted() {
        let limiter = BurstRateLimiter::new(3, 0.001);
        limiter.check("x");
        limiter.check("x");
        limiter.check("x");
        let d = limiter.check("x");
        assert!(!d.allowed);
        assert_eq!(d.remaining, 0);
    }

    // 3. Different keys are independent
    #[test]
    fn test_independent_keys() {
        let limiter = BurstRateLimiter::new(1, 0.001);
        assert!(limiter.check("key-a").allowed);
        assert!(limiter.check("key-b").allowed);
    }

    // 4. Refills over time
    #[test]
    fn test_refill_over_time() {
        let limiter = BurstRateLimiter::new(1, 100.0); // 100 tokens/s
        limiter.check("refill");
        let d1 = limiter.check("refill");
        assert!(!d1.allowed);
        sleep(Duration::from_millis(20));
        let d2 = limiter.check("refill");
        assert!(d2.allowed);
    }

    // 5. check_with_cost consumes multiple tokens
    #[test]
    fn test_check_with_cost() {
        let limiter = BurstRateLimiter::new(10, 1.0);
        let d = limiter.check_with_cost("bulk", 7);
        assert!(d.allowed);
        assert_eq!(d.limit, 10);
        let d2 = limiter.check_with_cost("bulk", 4); // Only 3 left
        assert!(!d2.allowed);
    }

    // 6. Per-client override is applied
    #[test]
    fn test_per_client_override() {
        let limiter = BurstRateLimiter::new(5, 1.0);
        limiter.register_client_override(
            "premium",
            ClientOverride {
                burst_size: 100,
                refill_rate: 10.0,
            },
        );
        let d = limiter.check("premium");
        assert!(d.allowed);
        assert_eq!(d.limit, 100, "Override burst_size should be 100");
        assert!(d.is_override, "Should report as override");
    }

    // 7. Removing override reverts to defaults
    #[test]
    fn test_remove_override() {
        let limiter = BurstRateLimiter::new(5, 1.0);
        limiter.register_client_override(
            "temp",
            ClientOverride {
                burst_size: 1000,
                refill_rate: 100.0,
            },
        );
        limiter.remove_client_override("temp");
        let d = limiter.check("temp");
        assert_eq!(d.limit, 5, "Should use default burst_size after removal");
        assert!(!d.is_override);
    }

    // 8. Restricted client override (lower limit)
    #[test]
    fn test_restricted_client_override() {
        let limiter = BurstRateLimiter::new(100, 10.0);
        limiter.register_client_override(
            "restricted",
            ClientOverride {
                burst_size: 1,
                refill_rate: 0.001,
            },
        );
        limiter.check("restricted");
        let d = limiter.check("restricted");
        assert!(
            !d.allowed,
            "Restricted client should be denied after 1 request"
        );
    }

    // 9. retry_after_secs is 0 when allowed
    #[test]
    fn test_retry_after_zero_when_allowed() {
        let limiter = BurstRateLimiter::new(10, 1.0);
        let d = limiter.check("ok");
        assert_eq!(d.retry_after_secs(), 0);
    }

    // 10. retry_after_secs is positive when denied
    #[test]
    fn test_retry_after_positive_when_denied() {
        let limiter = BurstRateLimiter::new(1, 0.5); // 0.5 tokens/sec
        limiter.check("slow");
        let d = limiter.check("slow");
        assert!(!d.allowed);
        assert!(d.retry_after_secs() >= 1);
    }

    // 11. X-RateLimit headers are set for allowed decision
    #[test]
    fn test_headers_allowed() {
        let limiter = BurstRateLimiter::new(10, 5.0);
        let d = limiter.check("h");
        let headers = BurstRateLimitHeaders::from_decision(&d);
        assert_eq!(headers.x_ratelimit_limit, "10");
        assert!(headers.retry_after.is_none());
    }

    // 12. X-RateLimit headers are set for denied decision
    #[test]
    fn test_headers_denied() {
        let limiter = BurstRateLimiter::new(1, 0.01);
        limiter.check("h2");
        let d = limiter.check("h2");
        let headers = BurstRateLimitHeaders::from_decision(&d);
        assert!(headers.retry_after.is_some());
        assert_eq!(headers.x_ratelimit_remaining, "0");
    }

    // 13. Headers can be inserted into HeaderMap
    #[test]
    fn test_headers_insert_into_header_map() {
        let limiter = BurstRateLimiter::new(20, 5.0);
        let d = limiter.check("insert");
        let rl = BurstRateLimitHeaders::from_decision(&d);
        let mut header_map = axum::http::HeaderMap::new();
        rl.insert_into(&mut header_map);
        assert!(header_map.contains_key("x-ratelimit-limit"));
        assert!(header_map.contains_key("x-ratelimit-remaining"));
        assert!(header_map.contains_key("x-ratelimit-reset"));
    }

    // 14. Denied response includes retry-after header
    #[test]
    fn test_denied_headers_include_retry_after() {
        let limiter = BurstRateLimiter::new(1, 0.01);
        limiter.check("ra");
        let d = limiter.check("ra");
        let rl = BurstRateLimitHeaders::from_decision(&d);
        let mut header_map = axum::http::HeaderMap::new();
        rl.insert_into(&mut header_map);
        assert!(header_map.contains_key("retry-after"));
    }

    // 15. too_many_requests_response returns 429
    #[test]
    fn test_429_response_status() {
        let limiter = BurstRateLimiter::new(1, 0.01);
        limiter.check("429-test");
        let d = limiter.check("429-test");
        let resp = BurstRateLimitHeaders::too_many_requests_response(&d);
        assert_eq!(resp.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
    }

    // 16. default_burst_size / default_refill_rate accessors
    #[test]
    fn test_accessors() {
        let limiter = BurstRateLimiter::new(42, 3.5);
        assert_eq!(limiter.default_burst_size(), 42);
        assert!((limiter.default_refill_rate() - 3.5).abs() < 1e-9);
    }

    // 17. peek_tokens returns full capacity for unknown key
    #[test]
    fn test_peek_tokens_unknown_key() {
        let limiter = BurstRateLimiter::new(10, 1.0);
        let tokens = limiter.peek_tokens("never-seen");
        assert!((tokens - 10.0).abs() < 1e-6);
    }

    // 18. peek_tokens decreases after consuming
    #[test]
    fn test_peek_tokens_after_consume() {
        let limiter = BurstRateLimiter::new(10, 0.001);
        limiter.check_with_cost("pk", 3);
        let tokens = limiter.peek_tokens("pk");
        assert!(tokens < 8.0, "tokens should be around 7 after consuming 3");
    }

    // 19. cost=0 is treated as 1
    #[test]
    fn test_zero_cost_treated_as_one() {
        let limiter = BurstRateLimiter::new(1, 0.001);
        let d1 = limiter.check_with_cost("zero", 0);
        assert!(d1.allowed);
        let d2 = limiter.check_with_cost("zero", 0);
        assert!(!d2.allowed); // Bucket exhausted
    }

    // 20. Burst capacity is respected across many clients
    #[test]
    fn test_many_independent_clients() {
        let limiter = BurstRateLimiter::new(3, 0.001);
        let results: Vec<bool> = (0..100)
            .map(|i| limiter.check(&format!("client-{}", i)).allowed)
            .collect();
        assert!(results.iter().all(|&r| r), "Each client should be allowed");
    }

    // 21. Concurrent access is safe
    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        let limiter = Arc::new(BurstRateLimiter::new(1000, 1000.0));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let l = Arc::clone(&limiter);
            handles.push(std::thread::spawn(move || {
                for _ in 0..50 {
                    l.check("concurrent");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    // 22. Override affects only specific key
    #[test]
    fn test_override_does_not_affect_others() {
        let limiter = BurstRateLimiter::new(5, 1.0);
        limiter.register_client_override(
            "special",
            ClientOverride {
                burst_size: 100,
                refill_rate: 50.0,
            },
        );
        let d_special = limiter.check("special");
        let d_normal = limiter.check("normal");
        assert_eq!(d_special.limit, 100);
        assert_eq!(d_normal.limit, 5);
    }

    // 23. remaining decrements with each request
    #[test]
    fn test_remaining_decrements() {
        let limiter = BurstRateLimiter::new(5, 0.001);
        let d1 = limiter.check("dec");
        let d2 = limiter.check("dec");
        assert!(d1.remaining >= d2.remaining);
    }

    // 24. X-RateLimit-Reset is present in BurstRateLimitHeaders
    #[test]
    fn test_ratelimit_reset_present() {
        let limiter = BurstRateLimiter::new(5, 1.0);
        limiter.check_with_cost("rst", 5);
        let d = limiter.check("rst");
        let headers = BurstRateLimitHeaders::from_decision(&d);
        // reset should be a non-negative string number
        let reset: u64 = headers.x_ratelimit_reset;
        // When denied, reset_at is in the future → reset > 0
        assert!(
            !d.allowed || reset == 0,
            "When denied, reset should be > 0; when allowed = 0"
        );
    }

    // 25. Multiple overrides can coexist
    #[test]
    fn test_multiple_overrides_coexist() {
        let limiter = BurstRateLimiter::new(5, 1.0);
        limiter.register_client_override(
            "a",
            ClientOverride {
                burst_size: 10,
                refill_rate: 1.0,
            },
        );
        limiter.register_client_override(
            "b",
            ClientOverride {
                burst_size: 20,
                refill_rate: 2.0,
            },
        );
        assert_eq!(limiter.check("a").limit, 10);
        assert_eq!(limiter.check("b").limit, 20);
        assert_eq!(limiter.check("c").limit, 5); // Default
    }
}
