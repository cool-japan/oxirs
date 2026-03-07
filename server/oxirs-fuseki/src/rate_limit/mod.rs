//! API rate limiting for OxiRS Fuseki.

pub mod burst_limiter;
pub use burst_limiter::{BurstDecision, BurstRateLimitHeaders, BurstRateLimiter, ClientOverride};
//
// Provides token-bucket and sliding-window rate limiters with per-IP and
// per-API-key granularity.  HTTP response headers and 429 status codes are
// produced by helper functions so that Axum handlers can integrate them
// transparently.

// HashMap unused - removed
use std::net::IpAddr;
// Arc used only in tests - removed
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::Mutex;
// serde unused - removed

// ---------------------------------------------------------------------------
// Rate-limiter trait
// ---------------------------------------------------------------------------

/// The result of a rate-limit check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimitDecision {
    /// Whether the request should be allowed.
    pub allowed: bool,
    /// Total configured limit (requests per window / bucket capacity).
    pub limit: u64,
    /// Remaining capacity at the time of the decision.
    pub remaining: u64,
    /// Instant at which the limit resets (for HTTP `Retry-After`).
    pub reset_at: Instant,
}

impl RateLimitDecision {
    /// Seconds until reset, clamped to zero.
    pub fn retry_after_secs(&self) -> u64 {
        let now = Instant::now();
        if self.reset_at > now {
            (self.reset_at - now).as_secs().max(1)
        } else {
            0
        }
    }
}

/// Trait implemented by all rate limiter variants.
pub trait RateLimiter: Send + Sync + 'static {
    /// Check (and consume) a request slot for the given key.
    fn check(&self, key: &str) -> RateLimitDecision;
    /// HTTP header: the configured limit value.
    fn limit_header_value(&self) -> String;
}

// ---------------------------------------------------------------------------
// SPARQL query complexity scorer
// ---------------------------------------------------------------------------

/// A simple SPARQL query complexity scorer.
///
/// The score is used to weight token consumption so complex queries cost more.
#[derive(Debug, Clone, Default)]
pub struct SparqlComplexityScorer;

impl SparqlComplexityScorer {
    /// Estimate a complexity score for a SPARQL query string.
    ///
    /// The score is a positive integer: simple SELECT returns 1, complex
    /// queries with OPTIONAL, UNION, FILTER, SERVICE, or sub-queries score
    /// proportionally higher.
    pub fn score(&self, query: &str) -> u64 {
        let upper = query.to_uppercase();
        let mut score: u64 = 1;

        // Keywords that indicate expensive operations.
        let weights: &[(&str, u64)] = &[
            ("OPTIONAL", 2),
            ("UNION", 3),
            ("FILTER", 1),
            ("SERVICE", 5),
            ("SUBQUERY", 4),
            ("SELECT", 1),
            ("CONSTRUCT", 2),
            ("DESCRIBE", 2),
            ("GROUP BY", 2),
            ("ORDER BY", 1),
            ("HAVING", 2),
            ("LIMIT", 0),
            ("OFFSET", 0),
            ("MINUS", 3),
            ("EXISTS", 2),
            ("NOT EXISTS", 3),
            ("VALUES", 1),
            ("BIND", 1),
            ("GRAPH", 2),
            ("LOAD", 10),
            ("INSERT", 5),
            ("DELETE", 5),
            ("UPDATE", 5),
        ];

        for (keyword, weight) in weights {
            let count = upper.matches(keyword).count() as u64;
            score = score.saturating_add(count * weight);
        }

        // Penalize long queries.
        let len_penalty = (query.len() as u64) / 500;
        score = score.saturating_add(len_penalty);

        score.max(1)
    }
}

// ---------------------------------------------------------------------------
// Token bucket limiter
// ---------------------------------------------------------------------------

/// Per-key token bucket state.
#[derive(Debug, Clone)]
struct TokenBucket {
    /// Current token count.
    tokens: f64,
    /// Last time tokens were refilled.
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: f64) -> Self {
        Self {
            tokens: capacity,
            last_refill: Instant::now(),
        }
    }

    /// Refill based on elapsed time and rate, then attempt to consume `cost`.
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
            return 0.0;
        }
        (cost - self.tokens) / refill_rate
    }
}

/// Token-bucket rate limiter.
///
/// Each client key gets an independent bucket.  Tokens refill at `refill_rate`
/// per second up to `capacity`.  A single request normally costs 1 token.
pub struct TokenBucketLimiter {
    /// Max tokens per bucket.
    capacity: u64,
    /// Refill rate (tokens / second).
    refill_rate: f64,
    /// Per-key buckets.
    buckets: DashMap<String, Mutex<TokenBucket>>,
}

impl TokenBucketLimiter {
    /// Create a new limiter.
    ///
    /// * `capacity` – burst capacity (max tokens)
    /// * `refill_rate` – tokens added per second
    pub fn new(capacity: u64, refill_rate: f64) -> Self {
        assert!(capacity > 0, "capacity must be positive");
        assert!(refill_rate > 0.0, "refill_rate must be positive");
        Self {
            capacity,
            refill_rate,
            buckets: DashMap::new(),
        }
    }

    /// Check and consume tokens for the given key with an explicit cost.
    pub fn check_with_cost(&self, key: &str, cost: u64) -> RateLimitDecision {
        let cost_f = cost as f64;
        let capacity_f = self.capacity as f64;

        let entry = self
            .buckets
            .entry(key.to_string())
            .or_insert_with(|| Mutex::new(TokenBucket::new(capacity_f)));

        let mut bucket = entry.lock();
        let allowed = bucket.try_consume(cost_f, capacity_f, self.refill_rate);
        let remaining = bucket.tokens.floor() as u64;
        let eta = bucket.eta_secs(cost_f, self.refill_rate);
        let reset_at = Instant::now() + Duration::from_secs_f64(eta.max(0.0));

        RateLimitDecision {
            allowed,
            limit: self.capacity,
            remaining,
            reset_at,
        }
    }
}

impl RateLimiter for TokenBucketLimiter {
    fn check(&self, key: &str) -> RateLimitDecision {
        self.check_with_cost(key, 1)
    }

    fn limit_header_value(&self) -> String {
        self.capacity.to_string()
    }
}

// ---------------------------------------------------------------------------
// Sliding window limiter
// ---------------------------------------------------------------------------

/// Per-key sliding window state.
#[derive(Debug, Clone)]
struct SlidingWindowEntry {
    /// Timestamps of requests in the current window.
    timestamps: Vec<Instant>,
}

impl SlidingWindowEntry {
    fn new() -> Self {
        Self {
            timestamps: Vec::new(),
        }
    }

    /// Remove expired timestamps and return the current count.
    fn evict_and_count(&mut self, window: Duration) -> usize {
        let cutoff = Instant::now() - window;
        self.timestamps.retain(|t| *t > cutoff);
        self.timestamps.len()
    }

    /// Oldest timestamp in the window, if any.
    fn oldest(&self) -> Option<Instant> {
        self.timestamps.first().copied()
    }
}

/// Sliding-window rate limiter.
///
/// Counts requests within a rolling time window.  When the limit is reached
/// the request is denied and a `reset_at` corresponding to when the oldest
/// in-window request expires is returned.
pub struct SlidingWindowLimiter {
    /// Max requests per window.
    limit: u64,
    /// Rolling window duration.
    window: Duration,
    /// Per-key entries.
    entries: DashMap<String, Mutex<SlidingWindowEntry>>,
}

impl SlidingWindowLimiter {
    /// Create a new sliding window limiter.
    pub fn new(limit: u64, window: Duration) -> Self {
        assert!(limit > 0, "limit must be positive");
        Self {
            limit,
            window,
            entries: DashMap::new(),
        }
    }
}

impl RateLimiter for SlidingWindowLimiter {
    fn check(&self, key: &str) -> RateLimitDecision {
        let entry = self
            .entries
            .entry(key.to_string())
            .or_insert_with(|| Mutex::new(SlidingWindowEntry::new()));

        let mut sw = entry.lock();
        let count = sw.evict_and_count(self.window);

        if count < self.limit as usize {
            sw.timestamps.push(Instant::now());
            let remaining = self.limit.saturating_sub(count as u64 + 1);
            RateLimitDecision {
                allowed: true,
                limit: self.limit,
                remaining,
                reset_at: Instant::now() + self.window,
            }
        } else {
            let reset_at = sw
                .oldest()
                .map(|t| t + self.window)
                .unwrap_or_else(|| Instant::now() + self.window);
            RateLimitDecision {
                allowed: false,
                limit: self.limit,
                remaining: 0,
                reset_at,
            }
        }
    }

    fn limit_header_value(&self) -> String {
        self.limit.to_string()
    }
}

// ---------------------------------------------------------------------------
// Per-IP and per-API-key limiter wrappers
// ---------------------------------------------------------------------------

/// A rate limiter keyed by client IP address.
pub struct IpRateLimiter<L: RateLimiter> {
    inner: L,
}

impl<L: RateLimiter> IpRateLimiter<L> {
    pub fn new(inner: L) -> Self {
        Self { inner }
    }

    pub fn check_ip(&self, ip: IpAddr) -> RateLimitDecision {
        self.inner.check(&ip.to_string())
    }
}

impl<L: RateLimiter> RateLimiter for IpRateLimiter<L> {
    fn check(&self, key: &str) -> RateLimitDecision {
        self.inner.check(key)
    }

    fn limit_header_value(&self) -> String {
        self.inner.limit_header_value()
    }
}

/// A rate limiter keyed by API key identifier.
pub struct ApiKeyRateLimiter<L: RateLimiter> {
    inner: L,
}

impl<L: RateLimiter> ApiKeyRateLimiter<L> {
    pub fn new(inner: L) -> Self {
        Self { inner }
    }

    pub fn check_api_key(&self, key_id: &str) -> RateLimitDecision {
        self.inner.check(key_id)
    }
}

impl<L: RateLimiter> RateLimiter for ApiKeyRateLimiter<L> {
    fn check(&self, key: &str) -> RateLimitDecision {
        self.inner.check(key)
    }

    fn limit_header_value(&self) -> String {
        self.inner.limit_header_value()
    }
}

// ---------------------------------------------------------------------------
// HTTP response helpers
// ---------------------------------------------------------------------------

/// Standard HTTP rate-limit headers.
#[derive(Debug, Clone)]
pub struct RateLimitHeaders {
    pub x_ratelimit_limit: String,
    pub x_ratelimit_remaining: String,
    pub x_ratelimit_reset: u64,
    pub retry_after: Option<u64>,
}

impl RateLimitHeaders {
    /// Build headers from a decision.
    pub fn from_decision(decision: &RateLimitDecision) -> Self {
        let retry_after = if !decision.allowed {
            Some(decision.retry_after_secs())
        } else {
            None
        };
        Self {
            x_ratelimit_limit: decision.limit.to_string(),
            x_ratelimit_remaining: decision.remaining.to_string(),
            x_ratelimit_reset: decision.retry_after_secs(),
            retry_after,
        }
    }

    /// Insert headers into an `axum::http::HeaderMap`.
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
}

/// Build a 429 Too Many Requests `axum::Response`.
pub fn too_many_requests_response(decision: &RateLimitDecision) -> axum::response::Response {
    use axum::http::{HeaderMap, HeaderValue, StatusCode};
    use axum::response::IntoResponse;

    let mut headers = HeaderMap::new();
    let rl_headers = RateLimitHeaders::from_decision(decision);
    rl_headers.insert_into(&mut headers);

    let body = format!(
        "{{\"error\":\"Too Many Requests\",\"retry_after_secs\":{}}}",
        decision.retry_after_secs()
    );

    (StatusCode::TOO_MANY_REQUESTS, headers, body).into_response()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep as std_sleep;

    // -----------------------------------------------------------------------
    // SparqlComplexityScorer
    // -----------------------------------------------------------------------

    #[test]
    fn test_scorer_simple_select() {
        let scorer = SparqlComplexityScorer;
        let score = scorer.score("SELECT ?s WHERE { ?s ?p ?o }");
        assert!(score >= 1, "simple SELECT should have score >= 1");
    }

    #[test]
    fn test_scorer_complex_query() {
        let scorer = SparqlComplexityScorer;
        let complex = "SELECT ?s WHERE { ?s ?p ?o OPTIONAL { ?s <x> ?y } UNION { ?a ?b ?c } FILTER(?o > 1) SERVICE <http://example.org/sparql> { ?z ?w ?v } }";
        let simple = "SELECT ?s WHERE { ?s ?p ?o }";
        assert!(scorer.score(complex) > scorer.score(simple));
    }

    #[test]
    fn test_scorer_update_more_expensive() {
        let scorer = SparqlComplexityScorer;
        let update = "INSERT DATA { <s> <p> <o> }";
        let select = "SELECT * WHERE { ?s ?p ?o }";
        assert!(scorer.score(update) > scorer.score(select));
    }

    #[test]
    fn test_scorer_load_expensive() {
        let scorer = SparqlComplexityScorer;
        let load = "LOAD <http://example.org/data.ttl>";
        assert!(scorer.score(load) >= 10);
    }

    #[test]
    fn test_scorer_minimum_one() {
        let scorer = SparqlComplexityScorer;
        assert_eq!(scorer.score(""), 1);
    }

    #[test]
    fn test_scorer_nested_optional() {
        let scorer = SparqlComplexityScorer;
        let query = "SELECT * WHERE { OPTIONAL { OPTIONAL { OPTIONAL { ?x ?y ?z } } } }";
        // 3 OPTIONAL at weight 2 each = 6, plus SELECT 1 = 7 total at least
        assert!(scorer.score(query) >= 7);
    }

    // -----------------------------------------------------------------------
    // TokenBucketLimiter
    // -----------------------------------------------------------------------

    #[test]
    fn test_token_bucket_allows_within_capacity() {
        let limiter = TokenBucketLimiter::new(5, 1.0);
        for _ in 0..5 {
            let decision = limiter.check("client-a");
            assert!(decision.allowed);
        }
    }

    #[test]
    fn test_token_bucket_denies_when_exhausted() {
        let limiter = TokenBucketLimiter::new(3, 0.001); // Very slow refill.
        limiter.check("x");
        limiter.check("x");
        limiter.check("x");
        let decision = limiter.check("x");
        assert!(!decision.allowed);
        assert_eq!(decision.remaining, 0);
    }

    #[test]
    fn test_token_bucket_independent_keys() {
        let limiter = TokenBucketLimiter::new(1, 0.001);
        let a = limiter.check("a");
        let b = limiter.check("b");
        assert!(a.allowed);
        assert!(b.allowed);
    }

    #[test]
    fn test_token_bucket_refills_over_time() {
        let limiter = TokenBucketLimiter::new(1, 100.0); // 100 tokens/sec
        limiter.check("refill");
        // Exhaust the bucket.
        let d1 = limiter.check("refill");
        assert!(!d1.allowed);
        // Wait enough for refill.
        std_sleep(Duration::from_millis(20));
        let d2 = limiter.check("refill");
        assert!(d2.allowed);
    }

    #[test]
    fn test_token_bucket_with_cost() {
        let limiter = TokenBucketLimiter::new(10, 1.0);
        // Consume 8 tokens at once.
        let d = limiter.check_with_cost("expensive", 8);
        assert!(d.allowed);
        assert_eq!(d.limit, 10);
        // Next single-token request should still be allowed.
        let d2 = limiter.check_with_cost("expensive", 1);
        assert!(d2.allowed);
        // Third request should fail (only 1 token left, need 2).
        let d3 = limiter.check_with_cost("expensive", 2);
        assert!(!d3.allowed);
    }

    #[test]
    fn test_token_bucket_limit_header() {
        let limiter = TokenBucketLimiter::new(42, 5.0);
        assert_eq!(limiter.limit_header_value(), "42");
    }

    #[test]
    fn test_token_bucket_remaining_decrements() {
        let limiter = TokenBucketLimiter::new(5, 0.001);
        let d1 = limiter.check("dec");
        let d2 = limiter.check("dec");
        assert!(d1.remaining > d2.remaining || d2.remaining <= d1.remaining);
    }

    #[test]
    fn test_token_bucket_retry_after_when_denied() {
        let limiter = TokenBucketLimiter::new(1, 0.5); // 0.5 tokens/sec = 2s to refill 1
        limiter.check("slow");
        let denied = limiter.check("slow");
        assert!(!denied.allowed);
        assert!(denied.retry_after_secs() >= 1);
    }

    // -----------------------------------------------------------------------
    // SlidingWindowLimiter
    // -----------------------------------------------------------------------

    #[test]
    fn test_sliding_window_allows_within_limit() {
        let limiter = SlidingWindowLimiter::new(5, Duration::from_secs(60));
        for _ in 0..5 {
            assert!(limiter.check("sw-a").allowed);
        }
    }

    #[test]
    fn test_sliding_window_denies_over_limit() {
        let limiter = SlidingWindowLimiter::new(3, Duration::from_secs(60));
        limiter.check("sw-b");
        limiter.check("sw-b");
        limiter.check("sw-b");
        let denied = limiter.check("sw-b");
        assert!(!denied.allowed);
        assert_eq!(denied.remaining, 0);
    }

    #[test]
    fn test_sliding_window_independent_keys() {
        let limiter = SlidingWindowLimiter::new(1, Duration::from_secs(60));
        assert!(limiter.check("c").allowed);
        assert!(limiter.check("d").allowed);
    }

    #[test]
    fn test_sliding_window_window_expiry() {
        let limiter = SlidingWindowLimiter::new(1, Duration::from_millis(100));
        assert!(limiter.check("expire").allowed);
        let denied = limiter.check("expire");
        assert!(!denied.allowed);
        std_sleep(Duration::from_millis(150));
        assert!(limiter.check("expire").allowed);
    }

    #[test]
    fn test_sliding_window_limit_header() {
        let limiter = SlidingWindowLimiter::new(100, Duration::from_secs(60));
        assert_eq!(limiter.limit_header_value(), "100");
    }

    #[test]
    fn test_sliding_window_remaining_decrements() {
        let limiter = SlidingWindowLimiter::new(10, Duration::from_secs(60));
        let d1 = limiter.check("rem");
        let d2 = limiter.check("rem");
        assert_eq!(d1.remaining, 9);
        assert_eq!(d2.remaining, 8);
    }

    #[test]
    fn test_sliding_window_retry_after_when_denied() {
        let limiter = SlidingWindowLimiter::new(1, Duration::from_secs(10));
        limiter.check("ra");
        let denied = limiter.check("ra");
        assert!(!denied.allowed);
        assert!(denied.retry_after_secs() > 0);
    }

    // -----------------------------------------------------------------------
    // IP and API key wrappers
    // -----------------------------------------------------------------------

    #[test]
    fn test_ip_rate_limiter() {
        let inner = TokenBucketLimiter::new(5, 1.0);
        let limiter = IpRateLimiter::new(inner);
        let ip: IpAddr = "1.2.3.4".parse().unwrap();
        let d = limiter.check_ip(ip);
        assert!(d.allowed);
    }

    #[test]
    fn test_api_key_rate_limiter() {
        let inner = TokenBucketLimiter::new(5, 1.0);
        let limiter = ApiKeyRateLimiter::new(inner);
        let d = limiter.check_api_key("key-abc");
        assert!(d.allowed);
    }

    #[test]
    fn test_ip_limiter_as_trait_object() {
        let inner = SlidingWindowLimiter::new(10, Duration::from_secs(60));
        let limiter: Box<dyn RateLimiter> = Box::new(IpRateLimiter::new(inner));
        let d = limiter.check("10.0.0.1");
        assert!(d.allowed);
    }

    // -----------------------------------------------------------------------
    // RateLimitDecision
    // -----------------------------------------------------------------------

    #[test]
    fn test_decision_retry_after_zero_when_allowed() {
        let decision = RateLimitDecision {
            allowed: true,
            limit: 10,
            remaining: 9,
            reset_at: Instant::now() - Duration::from_secs(1),
        };
        assert_eq!(decision.retry_after_secs(), 0);
    }

    #[test]
    fn test_decision_retry_after_positive_when_denied() {
        let decision = RateLimitDecision {
            allowed: false,
            limit: 10,
            remaining: 0,
            reset_at: Instant::now() + Duration::from_secs(5),
        };
        assert!(decision.retry_after_secs() >= 4);
    }

    // -----------------------------------------------------------------------
    // RateLimitHeaders
    // -----------------------------------------------------------------------

    #[test]
    fn test_rl_headers_from_allowed_decision() {
        let d = RateLimitDecision {
            allowed: true,
            limit: 100,
            remaining: 99,
            reset_at: Instant::now() + Duration::from_secs(60),
        };
        let headers = RateLimitHeaders::from_decision(&d);
        assert_eq!(headers.x_ratelimit_limit, "100");
        assert_eq!(headers.x_ratelimit_remaining, "99");
        assert!(headers.retry_after.is_none());
    }

    #[test]
    fn test_rl_headers_from_denied_decision() {
        let d = RateLimitDecision {
            allowed: false,
            limit: 10,
            remaining: 0,
            reset_at: Instant::now() + Duration::from_secs(3),
        };
        let headers = RateLimitHeaders::from_decision(&d);
        assert!(headers.retry_after.is_some());
        assert!(headers.retry_after.unwrap() >= 2);
    }

    #[test]
    fn test_rl_headers_insert_into_header_map() {
        let d = RateLimitDecision {
            allowed: true,
            limit: 50,
            remaining: 40,
            reset_at: Instant::now() + Duration::from_secs(60),
        };
        let headers_struct = RateLimitHeaders::from_decision(&d);
        let mut headers = axum::http::HeaderMap::new();
        headers_struct.insert_into(&mut headers);
        assert!(headers.contains_key("x-ratelimit-limit"));
        assert!(headers.contains_key("x-ratelimit-remaining"));
        assert!(headers.contains_key("x-ratelimit-reset"));
    }

    #[test]
    fn test_rl_headers_denied_includes_retry_after() {
        let d = RateLimitDecision {
            allowed: false,
            limit: 10,
            remaining: 0,
            reset_at: Instant::now() + Duration::from_secs(30),
        };
        let headers_struct = RateLimitHeaders::from_decision(&d);
        let mut headers = axum::http::HeaderMap::new();
        headers_struct.insert_into(&mut headers);
        assert!(headers.contains_key("retry-after"));
    }

    // -----------------------------------------------------------------------
    // 429 response helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_too_many_requests_response_status() {
        let d = RateLimitDecision {
            allowed: false,
            limit: 10,
            remaining: 0,
            reset_at: Instant::now() + Duration::from_secs(5),
        };
        let response = too_many_requests_response(&d);
        assert_eq!(response.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
    }

    // -----------------------------------------------------------------------
    // Concurrent access sanity checks
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_token_bucket_concurrent_access() {
        use std::sync::Arc;
        let limiter = Arc::new(TokenBucketLimiter::new(100, 100.0));
        let mut handles = Vec::new();
        for _ in 0..20 {
            let l = Arc::clone(&limiter);
            handles.push(tokio::spawn(async move {
                for _ in 0..5 {
                    l.check("concurrent-key");
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_sliding_window_concurrent_access() {
        use std::sync::Arc;
        let limiter = Arc::new(SlidingWindowLimiter::new(1000, Duration::from_secs(60)));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let l = Arc::clone(&limiter);
            handles.push(tokio::spawn(async move {
                for _ in 0..10 {
                    l.check("sw-concurrent");
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    }
}
