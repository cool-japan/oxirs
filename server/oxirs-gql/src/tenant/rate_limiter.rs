//! Per-tenant rate limiting for GraphQL operations.
//!
//! `TenantRateLimiter` tracks request counts per tenant per operation type
//! using a sliding 60-second window backed by an in-process `Mutex`.  Limits
//! are fully configurable per tenant via `TenantLimits`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Per-tenant operation rate limits.
#[derive(Debug, Clone)]
pub struct TenantLimits {
    /// Maximum number of `query` operations per minute.
    pub queries_per_minute: u32,
    /// Maximum number of `mutation` operations per minute.
    pub mutations_per_minute: u32,
    /// Maximum query complexity budget consumed per minute.
    pub complexity_budget: u32,
}

impl Default for TenantLimits {
    fn default() -> Self {
        Self {
            queries_per_minute: 60,
            mutations_per_minute: 20,
            complexity_budget: 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Outcome of a rate-limit check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimitResult {
    /// Whether the request is permitted.
    pub allowed: bool,
    /// Remaining requests in the current window (saturating at 0).
    pub remaining: u32,
    /// Unix-epoch milliseconds at which the current window resets.
    pub reset_at_ms: u64,
}

// ---------------------------------------------------------------------------
// Internal window tracker
// ---------------------------------------------------------------------------

/// Sliding-window counter for a single operation type.
struct WindowCounter {
    /// Timestamps of past operations within the window.
    timestamps: Vec<Instant>,
    window: Duration,
}

impl WindowCounter {
    fn new(window: Duration) -> Self {
        Self {
            timestamps: Vec::new(),
            window,
        }
    }

    /// Returns the number of requests in the current window after evicting
    /// stale entries.
    fn count(&mut self) -> usize {
        let now = Instant::now();
        self.timestamps
            .retain(|t| now.duration_since(*t) < self.window);
        self.timestamps.len()
    }

    /// Record a new request and return whether it is within the limit.
    fn check_and_record(&mut self, limit: u32) -> (bool, u32) {
        let now = Instant::now();
        self.timestamps
            .retain(|t| now.duration_since(*t) < self.window);
        let current = self.timestamps.len() as u32;
        if current < limit {
            self.timestamps.push(now);
            let remaining = limit.saturating_sub(current + 1);
            (true, remaining)
        } else {
            (false, 0)
        }
    }

    /// Earliest timestamp in the window — used to compute `reset_at`.
    #[allow(dead_code)]
    fn earliest(&self) -> Option<Instant> {
        self.timestamps
            .iter()
            .copied()
            .reduce(|a, b| if a < b { a } else { b })
    }
}

// ---------------------------------------------------------------------------
// Complexity tracker
// ---------------------------------------------------------------------------

struct ComplexityTracker {
    buckets: Vec<(Instant, u32)>,
    window: Duration,
}

impl ComplexityTracker {
    fn new(window: Duration) -> Self {
        Self {
            buckets: Vec::new(),
            window,
        }
    }

    fn total(&mut self) -> u32 {
        let now = Instant::now();
        self.buckets
            .retain(|(t, _)| now.duration_since(*t) < self.window);
        self.buckets.iter().map(|(_, c)| c).sum()
    }

    fn add(&mut self, cost: u32, budget: u32) -> (bool, u32) {
        let current = self.total();
        if current.saturating_add(cost) <= budget {
            self.buckets.push((Instant::now(), cost));
            let remaining = budget.saturating_sub(current + cost);
            (true, remaining)
        } else {
            (false, budget.saturating_sub(current))
        }
    }
}

// ---------------------------------------------------------------------------
// Per-tenant state
// ---------------------------------------------------------------------------

struct TenantState {
    limits: TenantLimits,
    query_counter: WindowCounter,
    mutation_counter: WindowCounter,
    complexity: ComplexityTracker,
}

impl TenantState {
    fn new(limits: TenantLimits) -> Self {
        let window = Duration::from_secs(60);
        Self {
            limits,
            query_counter: WindowCounter::new(window),
            mutation_counter: WindowCounter::new(window),
            complexity: ComplexityTracker::new(window),
        }
    }
}

// ---------------------------------------------------------------------------
// Public rate limiter
// ---------------------------------------------------------------------------

/// Thread-safe per-tenant rate limiter.
///
/// Tracks requests per tenant per operation type within a 60-second sliding
/// window.  Tenants without explicit configuration use `TenantLimits::default()`.
pub struct TenantRateLimiter {
    state: Arc<Mutex<HashMap<String, TenantState>>>,
    default_limits: TenantLimits,
}

impl std::fmt::Debug for TenantRateLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantRateLimiter").finish()
    }
}

impl TenantRateLimiter {
    /// Create a new rate limiter with the given default limits.
    pub fn new(default_limits: TenantLimits) -> Self {
        Self {
            state: Arc::new(Mutex::new(HashMap::new())),
            default_limits,
        }
    }

    /// Set per-tenant limits, replacing any existing configuration.
    pub fn configure_tenant(&self, tenant_id: &str, limits: TenantLimits) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.insert(tenant_id.to_string(), TenantState::new(limits));
    }

    /// Check whether `tenant_id` may perform `operation` and record the
    /// request if allowed.
    ///
    /// `operation` should be one of `"query"`, `"mutation"`, or `"subscription"`.
    /// Unknown operations fall back to the query limit.
    pub fn check_rate_limit(&self, tenant_id: &str, operation: &str) -> RateLimitResult {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let tenant = state
            .entry(tenant_id.to_string())
            .or_insert_with(|| TenantState::new(self.default_limits.clone()));

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Compute reset_at as now + 60s
        let reset_at_ms = now_ms + 60_000;

        let (allowed, remaining) = match operation.to_lowercase().as_str() {
            "mutation" => tenant
                .mutation_counter
                .check_and_record(tenant.limits.mutations_per_minute),
            _ => tenant
                .query_counter
                .check_and_record(tenant.limits.queries_per_minute),
        };

        RateLimitResult {
            allowed,
            remaining,
            reset_at_ms,
        }
    }

    /// Check and record query complexity consumption for a tenant.
    ///
    /// `cost` is the estimated complexity of the operation. Returns an error
    /// result (`.allowed = false`) if the complexity budget for the current
    /// window is exhausted.
    pub fn check_complexity(&self, tenant_id: &str, cost: u32) -> RateLimitResult {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let tenant = state
            .entry(tenant_id.to_string())
            .or_insert_with(|| TenantState::new(self.default_limits.clone()));

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let reset_at_ms = now_ms + 60_000;
        let budget = tenant.limits.complexity_budget;
        let (allowed, remaining) = tenant.complexity.add(cost, budget);
        RateLimitResult {
            allowed,
            remaining,
            reset_at_ms,
        }
    }

    /// Return the current request count for a tenant and operation within the
    /// window (without recording a new request).
    pub fn current_count(&self, tenant_id: &str, operation: &str) -> u32 {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let tenant = state
            .entry(tenant_id.to_string())
            .or_insert_with(|| TenantState::new(self.default_limits.clone()));
        match operation.to_lowercase().as_str() {
            "mutation" => tenant.mutation_counter.count() as u32,
            _ => tenant.query_counter.count() as u32,
        }
    }
}

impl Default for TenantRateLimiter {
    fn default() -> Self {
        Self::new(TenantLimits::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn strict_limiter() -> TenantRateLimiter {
        TenantRateLimiter::new(TenantLimits {
            queries_per_minute: 3,
            mutations_per_minute: 2,
            complexity_budget: 100,
        })
    }

    #[test]
    fn test_first_request_allowed() {
        let rl = strict_limiter();
        let result = rl.check_rate_limit("t1", "query");
        assert!(result.allowed);
    }

    #[test]
    fn test_remaining_decrements() {
        let rl = strict_limiter();
        let r1 = rl.check_rate_limit("t1", "query");
        let r2 = rl.check_rate_limit("t1", "query");
        assert!(r1.remaining > r2.remaining || r2.remaining == 0);
    }

    #[test]
    fn test_limit_exceeded_returns_denied() {
        let rl = strict_limiter();
        rl.check_rate_limit("t2", "query");
        rl.check_rate_limit("t2", "query");
        rl.check_rate_limit("t2", "query");
        let result = rl.check_rate_limit("t2", "query");
        assert!(!result.allowed);
        assert_eq!(result.remaining, 0);
    }

    #[test]
    fn test_mutation_limit_separate_from_query() {
        let rl = strict_limiter();
        rl.check_rate_limit("t3", "query");
        rl.check_rate_limit("t3", "query");
        rl.check_rate_limit("t3", "query");
        // query exhausted, mutation should still be allowed
        let m = rl.check_rate_limit("t3", "mutation");
        assert!(m.allowed);
    }

    #[test]
    fn test_configure_tenant_overrides_default() {
        let rl = strict_limiter();
        rl.configure_tenant(
            "premium",
            TenantLimits {
                queries_per_minute: 1_000,
                mutations_per_minute: 500,
                complexity_budget: 1_000_000,
            },
        );
        for _ in 0..100 {
            let result = rl.check_rate_limit("premium", "query");
            assert!(result.allowed, "premium tenant should not be rate-limited");
        }
    }

    #[test]
    fn test_tenants_are_isolated() {
        let rl = strict_limiter();
        rl.check_rate_limit("ta", "query");
        rl.check_rate_limit("ta", "query");
        rl.check_rate_limit("ta", "query");
        // ta exhausted; tb is fresh
        let tb_result = rl.check_rate_limit("tb", "query");
        assert!(tb_result.allowed);
    }

    #[test]
    fn test_reset_at_in_future() {
        let rl = strict_limiter();
        let result = rl.check_rate_limit("t4", "query");
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        assert!(result.reset_at_ms > now_ms);
    }

    #[test]
    fn test_complexity_within_budget_allowed() {
        let rl = strict_limiter();
        let r = rl.check_complexity("t5", 50);
        assert!(r.allowed);
        assert_eq!(r.remaining, 50);
    }

    #[test]
    fn test_complexity_exceeds_budget_denied() {
        let rl = strict_limiter();
        rl.check_complexity("t6", 80); // 80 of 100
        let r = rl.check_complexity("t6", 30); // 80 + 30 = 110 > 100
        assert!(!r.allowed);
    }

    #[test]
    fn test_current_count_tracks_requests() {
        let rl = strict_limiter();
        rl.check_rate_limit("t7", "query");
        rl.check_rate_limit("t7", "query");
        assert_eq!(rl.current_count("t7", "query"), 2);
    }

    #[test]
    fn test_unknown_operation_uses_query_limit() {
        let rl = strict_limiter();
        let r = rl.check_rate_limit("t8", "subscription");
        assert!(r.allowed); // defaults to query limit (3 per min)
    }

    #[test]
    fn test_mutation_limit_enforced_separately() {
        let rl = strict_limiter();
        rl.check_rate_limit("t9", "mutation");
        rl.check_rate_limit("t9", "mutation");
        let r = rl.check_rate_limit("t9", "mutation");
        assert!(!r.allowed); // limit is 2
    }
}
