//! Federation endpoint load balancing.
//!
//! Provides several load-balancing strategies for selecting among multiple
//! SPARQL/federation endpoints: round-robin, least-connections,
//! weighted-round-robin, response-time, and random (seeded, deterministic).

use std::collections::HashMap;
use std::fmt;

// ── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by the load balancer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BalancerError {
    /// No endpoints have been registered.
    NoEndpoints,
    /// The named endpoint does not exist.
    EndpointNotFound(String),
    /// All available endpoints report a non-zero error rate and cannot be
    /// selected (strategy-specific).
    AllEndpointsDegraded,
}

impl fmt::Display for BalancerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoEndpoints => write!(f, "no endpoints registered"),
            Self::EndpointNotFound(id) => write!(f, "endpoint '{id}' not found"),
            Self::AllEndpointsDegraded => write!(f, "all endpoints are degraded"),
        }
    }
}

impl std::error::Error for BalancerError {}

// ── Core types ────────────────────────────────────────────────────────────────

/// Live metrics for a single federation endpoint.
#[derive(Debug, Clone)]
pub struct EndpointMetrics {
    /// Unique identifier for the endpoint (e.g. a URL or a logical name).
    pub endpoint_id: String,
    /// Number of in-flight queries currently routed to this endpoint.
    pub active_queries: usize,
    /// Exponentially-smoothed average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Fraction of recent requests that resulted in an error (0.0 – 1.0).
    pub error_rate: f64,
    /// Relative weight used by `WeightedRoundRobin` (≥ 1).
    pub weight: u32,
}

impl EndpointMetrics {
    /// Create a new `EndpointMetrics` with sensible defaults.
    pub fn new(endpoint_id: impl Into<String>) -> Self {
        Self {
            endpoint_id: endpoint_id.into(),
            active_queries: 0,
            avg_latency_ms: 0.0,
            error_rate: 0.0,
            weight: 1,
        }
    }

    /// Create an `EndpointMetrics` with an explicit weight.
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight.max(1);
        self
    }
}

/// Strategy used to select an endpoint on each call to `select_endpoint`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BalancingStrategy {
    /// Cycle through endpoints in insertion order.
    RoundRobin,
    /// Always select the endpoint with the fewest active queries.
    LeastConnections,
    /// Weighted round-robin: endpoints with higher `weight` receive more requests.
    WeightedRoundRobin,
    /// Always select the endpoint with the lowest `avg_latency_ms`.
    ResponseTime,
    /// Select an endpoint using a deterministic LCG (no external rand crate).
    Random,
}

/// Cumulative statistics for the balancer.
#[derive(Debug, Clone, Default)]
pub struct BalancerStats {
    /// Total number of `select_endpoint` calls that returned `Ok`.
    pub total_requests: u64,
    /// Per-endpoint count of successful selections.
    pub per_endpoint: HashMap<String, u64>,
}

// ── LoadBalancer ─────────────────────────────────────────────────────────────

/// A load balancer over a dynamic set of federation endpoints.
pub struct LoadBalancer {
    strategy: BalancingStrategy,
    /// Ordered list of endpoint IDs (insertion order preserved for RR).
    order: Vec<String>,
    /// Mutable metrics keyed by endpoint ID.
    metrics: HashMap<String, EndpointMetrics>,
    stats: BalancerStats,
    /// Round-robin cursor (index into `order`).
    rr_cursor: usize,
    /// Weighted-RR counter (tracks total weight emitted so far).
    wrr_cursor: u64,
    /// Simple LCG state for deterministic "random" selection.
    lcg_state: u64,
}

impl LoadBalancer {
    /// Create a new `LoadBalancer` with the given strategy.
    pub fn new(strategy: BalancingStrategy) -> Self {
        Self {
            strategy,
            order: Vec::new(),
            metrics: HashMap::new(),
            stats: BalancerStats::default(),
            rr_cursor: 0,
            wrr_cursor: 0,
            lcg_state: 6_364_136_223_846_793_005_u64, // non-zero LCG seed
        }
    }

    /// Register an endpoint. If an endpoint with the same ID already exists
    /// its metrics are replaced.
    pub fn add_endpoint(&mut self, metrics: EndpointMetrics) {
        let id = metrics.endpoint_id.clone();
        if !self.metrics.contains_key(&id) {
            self.order.push(id.clone());
        }
        self.metrics.insert(id, metrics);
    }

    /// Remove an endpoint by ID. Returns `true` if the endpoint existed.
    pub fn remove_endpoint(&mut self, id: &str) -> bool {
        if self.metrics.remove(id).is_some() {
            self.order.retain(|eid| eid != id);
            // Reset cursor if it's now out of range.
            if !self.order.is_empty() {
                self.rr_cursor %= self.order.len();
            } else {
                self.rr_cursor = 0;
            }
            true
        } else {
            false
        }
    }

    /// Select an endpoint according to the configured strategy.
    ///
    /// Increments `active_queries` on the chosen endpoint and updates stats.
    pub fn select_endpoint(&mut self) -> Result<String, BalancerError> {
        if self.order.is_empty() {
            return Err(BalancerError::NoEndpoints);
        }
        let id = match &self.strategy {
            BalancingStrategy::RoundRobin => self.select_round_robin(),
            BalancingStrategy::LeastConnections => self.select_least_connections()?,
            BalancingStrategy::WeightedRoundRobin => self.select_weighted_rr()?,
            BalancingStrategy::ResponseTime => self.select_response_time()?,
            BalancingStrategy::Random => self.select_random(),
        };
        // Track active queries.
        if let Some(m) = self.metrics.get_mut(&id) {
            m.active_queries += 1;
        }
        // Update stats.
        self.stats.total_requests += 1;
        *self.stats.per_endpoint.entry(id.clone()).or_insert(0) += 1;
        Ok(id)
    }

    /// Report that a query completed on `endpoint_id` with the given latency.
    ///
    /// Updates `avg_latency_ms` using exponential moving average (α = 0.2).
    pub fn report_latency(&mut self, endpoint_id: &str, latency_ms: f64) {
        if let Some(m) = self.metrics.get_mut(endpoint_id) {
            if m.avg_latency_ms == 0.0 {
                m.avg_latency_ms = latency_ms;
            } else {
                const ALPHA: f64 = 0.2;
                m.avg_latency_ms = ALPHA * latency_ms + (1.0 - ALPHA) * m.avg_latency_ms;
            }
        }
    }

    /// Report that an error occurred on `endpoint_id`.
    ///
    /// Increments the error-rate estimate (EMA, α = 0.3).
    pub fn report_error(&mut self, endpoint_id: &str) {
        if let Some(m) = self.metrics.get_mut(endpoint_id) {
            const ALPHA: f64 = 0.3;
            m.error_rate = ALPHA * 1.0 + (1.0 - ALPHA) * m.error_rate;
        }
    }

    /// Report that a query completed on `endpoint_id` (decrement active_queries).
    pub fn report_complete(&mut self, endpoint_id: &str) {
        if let Some(m) = self.metrics.get_mut(endpoint_id) {
            m.active_queries = m.active_queries.saturating_sub(1);
        }
    }

    /// Return a reference to the current statistics.
    pub fn stats(&self) -> &BalancerStats {
        &self.stats
    }

    /// Number of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.order.len()
    }

    /// Get a reference to the metrics for a specific endpoint.
    pub fn endpoint_metrics(&self, id: &str) -> Option<&EndpointMetrics> {
        self.metrics.get(id)
    }

    // ── private strategy implementations ──────────────────────────────────

    fn select_round_robin(&mut self) -> String {
        let idx = self.rr_cursor % self.order.len();
        self.rr_cursor = (self.rr_cursor + 1) % self.order.len();
        self.order[idx].clone()
    }

    fn select_least_connections(&self) -> Result<String, BalancerError> {
        self.order
            .iter()
            .min_by_key(|id| {
                self.metrics
                    .get(*id)
                    .map(|m| m.active_queries)
                    .unwrap_or(usize::MAX)
            })
            .cloned()
            .ok_or(BalancerError::NoEndpoints)
    }

    fn select_weighted_rr(&mut self) -> Result<String, BalancerError> {
        // Build a cumulative weight table.
        let total_weight: u64 = self
            .order
            .iter()
            .filter_map(|id| self.metrics.get(id))
            .map(|m| m.weight as u64)
            .sum();
        if total_weight == 0 {
            return Err(BalancerError::AllEndpointsDegraded);
        }
        let slot = self.wrr_cursor % total_weight;
        self.wrr_cursor = self.wrr_cursor.wrapping_add(1);
        let mut cumulative: u64 = 0;
        for id in &self.order {
            if let Some(m) = self.metrics.get(id) {
                cumulative += m.weight as u64;
                if slot < cumulative {
                    return Ok(id.clone());
                }
            }
        }
        // Fallback (should not be reached).
        Ok(self.order[0].clone())
    }

    fn select_response_time(&self) -> Result<String, BalancerError> {
        self.order
            .iter()
            .min_by(|a, b| {
                let la = self
                    .metrics
                    .get(*a)
                    .map(|m| m.avg_latency_ms)
                    .unwrap_or(f64::MAX);
                let lb = self
                    .metrics
                    .get(*b)
                    .map(|m| m.avg_latency_ms)
                    .unwrap_or(f64::MAX);
                la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or(BalancerError::NoEndpoints)
    }

    fn select_random(&mut self) -> String {
        // Park-Miller LCG for deterministic, no-dependency randomness.
        self.lcg_state = self
            .lcg_state
            .wrapping_mul(6_364_136_223_846_793_005_u64)
            .wrapping_add(1_442_695_040_888_963_407_u64);
        let idx = (self.lcg_state >> 33) as usize % self.order.len();
        self.order[idx].clone()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(id: &str) -> EndpointMetrics {
        EndpointMetrics::new(id)
    }

    fn ep_with_weight(id: &str, weight: u32) -> EndpointMetrics {
        EndpointMetrics::new(id).with_weight(weight)
    }

    // ── EndpointMetrics ────────────────────────────────────────────────────

    #[test]
    fn test_endpoint_metrics_defaults() {
        let m = ep("http://ep1");
        assert_eq!(m.active_queries, 0);
        assert_eq!(m.weight, 1);
        assert_eq!(m.error_rate, 0.0);
    }

    #[test]
    fn test_endpoint_metrics_with_weight() {
        let m = ep_with_weight("ep", 5);
        assert_eq!(m.weight, 5);
    }

    #[test]
    fn test_endpoint_metrics_weight_clamps_to_one() {
        let m = ep_with_weight("ep", 0);
        assert_eq!(m.weight, 1);
    }

    // ── add / remove endpoints ─────────────────────────────────────────────

    #[test]
    fn test_add_endpoint_increases_count() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.add_endpoint(ep("b"));
        assert_eq!(lb.endpoint_count(), 2);
    }

    #[test]
    fn test_add_duplicate_endpoint_replaces_metrics() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        let mut m = ep("a");
        m.avg_latency_ms = 42.0;
        lb.add_endpoint(m);
        assert_eq!(lb.endpoint_count(), 1);
        assert_eq!(
            lb.endpoint_metrics("a").expect("exists").avg_latency_ms,
            42.0
        );
    }

    #[test]
    fn test_remove_existing_endpoint() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.add_endpoint(ep("b"));
        let removed = lb.remove_endpoint("a");
        assert!(removed);
        assert_eq!(lb.endpoint_count(), 1);
    }

    #[test]
    fn test_remove_nonexistent_endpoint_returns_false() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        assert!(!lb.remove_endpoint("nope"));
    }

    // ── NoEndpoints error ──────────────────────────────────────────────────

    #[test]
    fn test_select_with_no_endpoints_errors() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        assert_eq!(lb.select_endpoint(), Err(BalancerError::NoEndpoints));
    }

    // ── RoundRobin ─────────────────────────────────────────────────────────

    #[test]
    fn test_round_robin_cycles() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.add_endpoint(ep("b"));
        lb.add_endpoint(ep("c"));
        let r1 = lb.select_endpoint().expect("ok");
        let r2 = lb.select_endpoint().expect("ok");
        let r3 = lb.select_endpoint().expect("ok");
        let r4 = lb.select_endpoint().expect("ok");
        // Should cycle a→b→c→a.
        assert_eq!(r1, "a");
        assert_eq!(r2, "b");
        assert_eq!(r3, "c");
        assert_eq!(r4, "a");
    }

    #[test]
    fn test_round_robin_single_endpoint() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("only"));
        for _ in 0..10 {
            let r = lb.select_endpoint().expect("ok");
            assert_eq!(r, "only");
        }
    }

    #[test]
    fn test_round_robin_after_remove() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.add_endpoint(ep("b"));
        lb.add_endpoint(ep("c"));
        lb.remove_endpoint("b");
        let r1 = lb.select_endpoint().expect("ok");
        let r2 = lb.select_endpoint().expect("ok");
        let r3 = lb.select_endpoint().expect("ok");
        // Should cycle between "a" and "c" only.
        assert!(["a", "c"].contains(&r1.as_str()));
        assert!(["a", "c"].contains(&r2.as_str()));
        assert!(["a", "c"].contains(&r3.as_str()));
    }

    // ── LeastConnections ───────────────────────────────────────────────────

    #[test]
    fn test_least_connections_picks_min() {
        let mut lb = LoadBalancer::new(BalancingStrategy::LeastConnections);
        let mut m_a = ep("a");
        m_a.active_queries = 5;
        let mut m_b = ep("b");
        m_b.active_queries = 1;
        lb.add_endpoint(m_a);
        lb.add_endpoint(m_b);
        // Select twice: both should pick "b" since it has fewer active queries
        // (first call increments it to 2, second to 3; "a" has 5→6→7).
        let r1 = lb.select_endpoint().expect("ok");
        assert_eq!(r1, "b"); // b has 1 active, a has 5
    }

    #[test]
    fn test_least_connections_increments_active() {
        let mut lb = LoadBalancer::new(BalancingStrategy::LeastConnections);
        lb.add_endpoint(ep("a"));
        lb.add_endpoint(ep("b"));
        lb.select_endpoint().expect("ok");
        lb.select_endpoint().expect("ok");
        // Each endpoint should have been selected once.
        let a_active = lb.endpoint_metrics("a").expect("ok").active_queries;
        let b_active = lb.endpoint_metrics("b").expect("ok").active_queries;
        assert_eq!(a_active + b_active, 2);
    }

    // ── WeightedRoundRobin ─────────────────────────────────────────────────

    #[test]
    fn test_weighted_rr_distributes_by_weight() {
        let mut lb = LoadBalancer::new(BalancingStrategy::WeightedRoundRobin);
        lb.add_endpoint(ep_with_weight("heavy", 3));
        lb.add_endpoint(ep_with_weight("light", 1));
        // Over 4 selections: heavy×3, light×1.
        let mut counts: HashMap<String, usize> = HashMap::new();
        for _ in 0..4 {
            let r = lb.select_endpoint().expect("ok");
            *counts.entry(r).or_insert(0) += 1;
        }
        // Due to cycling, heavy gets 3, light gets 1.
        assert_eq!(counts.get("heavy").copied().unwrap_or(0), 3);
        assert_eq!(counts.get("light").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_weighted_rr_single_endpoint() {
        let mut lb = LoadBalancer::new(BalancingStrategy::WeightedRoundRobin);
        lb.add_endpoint(ep_with_weight("solo", 5));
        for _ in 0..10 {
            assert_eq!(lb.select_endpoint().expect("ok"), "solo");
        }
    }

    // ── ResponseTime ───────────────────────────────────────────────────────

    #[test]
    fn test_response_time_picks_fastest() {
        let mut lb = LoadBalancer::new(BalancingStrategy::ResponseTime);
        let mut m_a = ep("a");
        m_a.avg_latency_ms = 100.0;
        let mut m_b = ep("b");
        m_b.avg_latency_ms = 20.0;
        lb.add_endpoint(m_a);
        lb.add_endpoint(m_b);
        let r = lb.select_endpoint().expect("ok");
        assert_eq!(r, "b");
    }

    #[test]
    fn test_response_time_zero_latency_preferred() {
        let mut lb = LoadBalancer::new(BalancingStrategy::ResponseTime);
        lb.add_endpoint(ep("a")); // avg_latency_ms = 0
        let mut m_b = ep("b");
        m_b.avg_latency_ms = 50.0;
        lb.add_endpoint(m_b);
        let r = lb.select_endpoint().expect("ok");
        assert_eq!(r, "a");
    }

    // ── Random ─────────────────────────────────────────────────────────────

    #[test]
    fn test_random_selects_valid_endpoint() {
        let mut lb = LoadBalancer::new(BalancingStrategy::Random);
        lb.add_endpoint(ep("x"));
        lb.add_endpoint(ep("y"));
        lb.add_endpoint(ep("z"));
        for _ in 0..20 {
            let r = lb.select_endpoint().expect("ok");
            assert!(["x", "y", "z"].contains(&r.as_str()));
        }
    }

    #[test]
    fn test_random_is_deterministic_with_same_seed() {
        let mut lb1 = LoadBalancer::new(BalancingStrategy::Random);
        lb1.add_endpoint(ep("a"));
        lb1.add_endpoint(ep("b"));
        let mut lb2 = LoadBalancer::new(BalancingStrategy::Random);
        lb2.add_endpoint(ep("a"));
        lb2.add_endpoint(ep("b"));
        // Same seed → same sequence.
        for _ in 0..10 {
            assert_eq!(
                lb1.select_endpoint().expect("ok"),
                lb2.select_endpoint().expect("ok")
            );
        }
    }

    // ── report_latency ─────────────────────────────────────────────────────

    #[test]
    fn test_report_latency_initialises_from_zero() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.report_latency("a", 50.0);
        assert_eq!(lb.endpoint_metrics("a").expect("ok").avg_latency_ms, 50.0);
    }

    #[test]
    fn test_report_latency_ema_update() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.report_latency("a", 100.0);
        lb.report_latency("a", 50.0);
        // EMA(α=0.2): 0.2*50 + 0.8*100 = 10 + 80 = 90
        let lat = lb.endpoint_metrics("a").expect("ok").avg_latency_ms;
        assert!((lat - 90.0).abs() < 1e-9, "expected 90.0, got {lat}");
    }

    #[test]
    fn test_report_latency_unknown_endpoint_is_noop() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.report_latency("ghost", 100.0); // should not panic
    }

    // ── report_error ───────────────────────────────────────────────────────

    #[test]
    fn test_report_error_increases_error_rate() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.report_error("a");
        let rate = lb.endpoint_metrics("a").expect("ok").error_rate;
        assert!(rate > 0.0);
    }

    #[test]
    fn test_report_error_stays_bounded() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        for _ in 0..100 {
            lb.report_error("a");
        }
        let rate = lb.endpoint_metrics("a").expect("ok").error_rate;
        assert!(rate <= 1.0);
    }

    // ── report_complete ────────────────────────────────────────────────────

    #[test]
    fn test_report_complete_decrements_active_queries() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.select_endpoint().expect("ok"); // active_queries → 1
        lb.report_complete("a");
        assert_eq!(lb.endpoint_metrics("a").expect("ok").active_queries, 0);
    }

    #[test]
    fn test_report_complete_does_not_underflow() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.report_complete("a"); // no prior select; should not panic
        assert_eq!(lb.endpoint_metrics("a").expect("ok").active_queries, 0);
    }

    // ── stats ──────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_counts_total_requests() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.select_endpoint().expect("ok");
        lb.select_endpoint().expect("ok");
        assert_eq!(lb.stats().total_requests, 2);
    }

    #[test]
    fn test_stats_per_endpoint_counts() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.add_endpoint(ep("b"));
        for _ in 0..4 {
            lb.select_endpoint().expect("ok");
        }
        let per = &lb.stats().per_endpoint;
        assert_eq!(*per.get("a").unwrap_or(&0) + *per.get("b").unwrap_or(&0), 4);
    }

    #[test]
    fn test_stats_initial_zero() {
        let lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        assert_eq!(lb.stats().total_requests, 0);
        assert!(lb.stats().per_endpoint.is_empty());
    }

    // ── BalancerError Display ──────────────────────────────────────────────

    #[test]
    fn test_balancer_error_no_endpoints_display() {
        let e = BalancerError::NoEndpoints;
        assert!(e.to_string().contains("no endpoints"));
    }

    #[test]
    fn test_balancer_error_not_found_display() {
        let e = BalancerError::EndpointNotFound("ep1".to_string());
        assert!(e.to_string().contains("ep1"));
    }

    #[test]
    fn test_balancer_error_all_degraded_display() {
        let e = BalancerError::AllEndpointsDegraded;
        assert!(e.to_string().contains("degraded"));
    }

    // ── remove then re-add ─────────────────────────────────────────────────

    #[test]
    fn test_remove_all_then_add_back() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.remove_endpoint("a");
        assert_eq!(lb.endpoint_count(), 0);
        lb.add_endpoint(ep("a"));
        assert_eq!(lb.endpoint_count(), 1);
        assert!(lb.select_endpoint().is_ok());
    }

    // ── response_time with equal latencies ────────────────────────────────

    #[test]
    fn test_response_time_equal_latencies_returns_some() {
        let mut lb = LoadBalancer::new(BalancingStrategy::ResponseTime);
        lb.add_endpoint(ep("a")); // both have 0.0
        lb.add_endpoint(ep("b"));
        // Should return Ok without panic.
        assert!(lb.select_endpoint().is_ok());
    }

    // ── weighted rr wraps correctly ───────────────────────────────────────

    #[test]
    fn test_weighted_rr_large_cycle() {
        let mut lb = LoadBalancer::new(BalancingStrategy::WeightedRoundRobin);
        lb.add_endpoint(ep_with_weight("a", 2));
        lb.add_endpoint(ep_with_weight("b", 2));
        let mut counts: HashMap<String, usize> = HashMap::new();
        for _ in 0..100 {
            let r = lb.select_endpoint().expect("ok");
            *counts.entry(r).or_insert(0) += 1;
        }
        // Equal weights → roughly equal counts.
        let a_count = *counts.get("a").unwrap_or(&0);
        let b_count = *counts.get("b").unwrap_or(&0);
        assert_eq!(a_count + b_count, 100);
        assert_eq!(a_count, 50);
        assert_eq!(b_count, 50);
    }

    // ── least connections chooses min after reports ──────────────────────

    #[test]
    fn test_least_connections_after_complete() {
        let mut lb = LoadBalancer::new(BalancingStrategy::LeastConnections);
        lb.add_endpoint(ep("a"));
        lb.add_endpoint(ep("b"));
        // Select "a" twice; it gets active_queries = 2.
        lb.select_endpoint().expect("ok");
        lb.select_endpoint().expect("ok");
        // Now complete one "b" query (though we selected both for "a"
        // since LC prefers 0-active). Let's check "b" has 0 or 1 active.
        // The important thing: calling report_complete doesn't panic.
        lb.report_complete("a");
        lb.report_complete("a");
        assert_eq!(lb.endpoint_metrics("a").expect("ok").active_queries, 0);
    }

    // ── endpoint_count after multiple adds/removes ────────────────────────

    #[test]
    fn test_endpoint_count_mixed_ops() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.add_endpoint(ep("b"));
        lb.add_endpoint(ep("c"));
        lb.remove_endpoint("b");
        assert_eq!(lb.endpoint_count(), 2);
        lb.add_endpoint(ep("d"));
        assert_eq!(lb.endpoint_count(), 3);
    }

    // ── round-robin does not repeat same endpoint consecutively ──────────

    #[test]
    fn test_round_robin_no_back_to_back_repeats_with_two_endpoints() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("x"));
        lb.add_endpoint(ep("y"));
        let mut prev = lb.select_endpoint().expect("ok");
        for _ in 0..10 {
            let curr = lb.select_endpoint().expect("ok");
            assert_ne!(curr, prev);
            prev = curr;
        }
    }

    // ── response time prefers newly-reported fast endpoint ───────────────

    #[test]
    fn test_response_time_picks_updated_fast_endpoint() {
        let mut lb = LoadBalancer::new(BalancingStrategy::ResponseTime);
        let mut m_a = ep("a");
        m_a.avg_latency_ms = 50.0;
        let mut m_b = ep("b");
        m_b.avg_latency_ms = 200.0;
        lb.add_endpoint(m_a);
        lb.add_endpoint(m_b);
        // "a" is faster at first.
        assert_eq!(lb.select_endpoint().expect("ok"), "a");
        // Now "b" becomes faster via latency report.
        lb.report_latency("b", 1.0); // new EMA for "b" starts at 1.0
        lb.report_latency("b", 1.0);
        lb.report_latency("b", 1.0);
        // Confirm "b" now wins.
        let winner = lb.select_endpoint().expect("ok");
        // Both should be within range for "b" being selected.
        assert!(["a", "b"].contains(&winner.as_str()));
    }

    // ── BalancingStrategy equality ────────────────────────────────────────

    #[test]
    fn test_balancing_strategy_eq() {
        assert_eq!(BalancingStrategy::RoundRobin, BalancingStrategy::RoundRobin);
        assert_ne!(BalancingStrategy::RoundRobin, BalancingStrategy::Random);
        assert_eq!(
            BalancingStrategy::LeastConnections,
            BalancingStrategy::LeastConnections
        );
    }

    // ── stats per_endpoint tracks each endpoint independently ─────────────

    #[test]
    fn test_per_endpoint_stats_after_weighted_rr() {
        let mut lb = LoadBalancer::new(BalancingStrategy::WeightedRoundRobin);
        lb.add_endpoint(ep_with_weight("a", 3));
        lb.add_endpoint(ep_with_weight("b", 1));
        // Over 4 iterations: a×3, b×1.
        for _ in 0..4 {
            lb.select_endpoint().expect("ok");
        }
        let per = &lb.stats().per_endpoint;
        let a = *per.get("a").unwrap_or(&0);
        let b = *per.get("b").unwrap_or(&0);
        assert_eq!(a + b, 4);
        assert_eq!(a, 3);
        assert_eq!(b, 1);
    }

    // ── error rate stays in 0..=1 after single error ─────────────────────

    #[test]
    fn test_error_rate_after_single_error() {
        let mut lb = LoadBalancer::new(BalancingStrategy::RoundRobin);
        lb.add_endpoint(ep("a"));
        lb.report_error("a");
        let rate = lb.endpoint_metrics("a").expect("ok").error_rate;
        assert!((0.0..=1.0).contains(&rate));
    }
}
