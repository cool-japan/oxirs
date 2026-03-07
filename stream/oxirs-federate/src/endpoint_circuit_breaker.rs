//! # Per-Endpoint Circuit Breaker
//!
//! Provides independent circuit breakers for each federated SPARQL endpoint.
//! Prevents cascading failures by isolating unhealthy endpoints and
//! allowing automatic recovery via half-open probe requests.
//!
//! ## Features
//!
//! - **Per-endpoint state**: Each endpoint has its own circuit breaker
//! - **Three states**: Closed (normal) → Open (failing) → Half-Open (probing)
//! - **Configurable thresholds**: Failure count, success count, timeout
//! - **Sliding window**: Rolling failure rate calculation
//! - **Half-open probing**: Limited requests to test recovery
//! - **Endpoint health scoring**: Weighted health score per endpoint
//! - **Statistics and metrics**: Detailed per-endpoint and aggregate stats

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::{info, warn};

// ─────────────────────────────────────────────
// Circuit Breaker State
// ─────────────────────────────────────────────

/// State of a circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Normal operation — requests flow through.
    Closed,
    /// Endpoint is failing — requests are rejected.
    Open,
    /// Probing recovery — limited requests allowed.
    HalfOpen,
}

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for per-endpoint circuit breakers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointCircuitBreakerConfig {
    /// Number of failures in the sliding window to trip the breaker (default: 5).
    pub failure_threshold: u32,
    /// Number of successes in half-open to close the breaker (default: 3).
    pub success_threshold: u32,
    /// Duration the breaker stays open before transitioning to half-open (default: 30s).
    pub open_duration: Duration,
    /// Sliding window size in seconds for failure rate calculation (default: 60s).
    pub sliding_window_secs: u64,
    /// Maximum concurrent requests in half-open state (default: 1).
    pub half_open_max_requests: u32,
    /// Failure rate threshold (0.0–1.0) to trip the breaker (default: 0.5).
    pub failure_rate_threshold: f64,
    /// Minimum requests in window before rate-based tripping applies (default: 10).
    pub min_requests_for_rate: u32,
}

impl Default for EndpointCircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            open_duration: Duration::from_secs(30),
            sliding_window_secs: 60,
            half_open_max_requests: 1,
            failure_rate_threshold: 0.5,
            min_requests_for_rate: 10,
        }
    }
}

// ─────────────────────────────────────────────
// Request Outcome
// ─────────────────────────────────────────────

/// Outcome of a request to track in the sliding window.
#[derive(Debug, Clone)]
struct RequestOutcome {
    success: bool,
    timestamp: Instant,
}

// ─────────────────────────────────────────────
// Per-Endpoint Breaker
// ─────────────────────────────────────────────

/// Circuit breaker state for a single endpoint.
struct EndpointBreaker {
    /// Current state.
    state: CircuitState,
    /// When the state last changed.
    state_changed_at: Instant,
    /// Sliding window of recent request outcomes.
    outcomes: VecDeque<RequestOutcome>,
    /// Consecutive failure count.
    consecutive_failures: u32,
    /// Consecutive success count (for half-open recovery).
    consecutive_successes: u32,
    /// Total requests.
    total_requests: u64,
    /// Total failures.
    total_failures: u64,
    /// Total state transitions.
    transitions: u64,
    /// Requests rejected while open.
    rejected_count: u64,
    /// Active half-open requests.
    half_open_active: u32,
    /// Last successful request time.
    last_success: Option<Instant>,
    /// Last failure time.
    last_failure: Option<Instant>,
}

impl EndpointBreaker {
    fn new() -> Self {
        Self {
            state: CircuitState::Closed,
            state_changed_at: Instant::now(),
            outcomes: VecDeque::new(),
            consecutive_failures: 0,
            consecutive_successes: 0,
            total_requests: 0,
            total_failures: 0,
            transitions: 0,
            rejected_count: 0,
            half_open_active: 0,
            last_success: None,
            last_failure: None,
        }
    }
}

// ─────────────────────────────────────────────
// Per-Endpoint Stats
// ─────────────────────────────────────────────

/// Statistics for a single endpoint's circuit breaker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointBreakerStats {
    /// Endpoint URL/identifier.
    pub endpoint: String,
    /// Current circuit state.
    pub state: CircuitState,
    /// Current failure rate in sliding window.
    pub failure_rate: f64,
    /// Total requests.
    pub total_requests: u64,
    /// Total failures.
    pub total_failures: u64,
    /// Requests rejected while open.
    pub rejected_count: u64,
    /// State transitions.
    pub transitions: u64,
    /// Health score (0.0–1.0, 1.0 = healthy).
    pub health_score: f64,
    /// Time in current state (ms).
    pub state_duration_ms: u64,
}

/// Aggregate statistics across all endpoints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateCircuitBreakerStats {
    /// Total endpoints tracked.
    pub total_endpoints: usize,
    /// Endpoints in Closed state.
    pub closed_count: usize,
    /// Endpoints in Open state.
    pub open_count: usize,
    /// Endpoints in Half-Open state.
    pub half_open_count: usize,
    /// Average health score across all endpoints.
    pub avg_health_score: f64,
    /// Total requests across all endpoints.
    pub total_requests: u64,
    /// Total failures across all endpoints.
    pub total_failures: u64,
}

// ─────────────────────────────────────────────
// Endpoint Circuit Breaker Manager
// ─────────────────────────────────────────────

/// Manages per-endpoint circuit breakers for federated SPARQL endpoints.
pub struct EndpointCircuitBreakerManager {
    config: EndpointCircuitBreakerConfig,
    breakers: HashMap<String, EndpointBreaker>,
}

impl EndpointCircuitBreakerManager {
    /// Create a new manager.
    pub fn new(config: EndpointCircuitBreakerConfig) -> Self {
        Self {
            config,
            breakers: HashMap::new(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(EndpointCircuitBreakerConfig::default())
    }

    /// Check if a request to an endpoint is allowed.
    ///
    /// Returns `true` if the request should proceed, `false` if rejected.
    pub fn allow_request(&mut self, endpoint: &str) -> bool {
        let breaker = self
            .breakers
            .entry(endpoint.to_string())
            .or_insert_with(EndpointBreaker::new);

        match breaker.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if we should transition to half-open
                if breaker.state_changed_at.elapsed() >= self.config.open_duration {
                    breaker.state = CircuitState::HalfOpen;
                    breaker.state_changed_at = Instant::now();
                    breaker.consecutive_successes = 0;
                    breaker.half_open_active = 0;
                    breaker.transitions += 1;
                    info!(endpoint, "Circuit breaker transitioning to Half-Open");
                    // Allow the first probe request
                    breaker.half_open_active += 1;
                    true
                } else {
                    breaker.rejected_count += 1;
                    false
                }
            }
            CircuitState::HalfOpen => {
                if breaker.half_open_active < self.config.half_open_max_requests {
                    breaker.half_open_active += 1;
                    true
                } else {
                    breaker.rejected_count += 1;
                    false
                }
            }
        }
    }

    /// Record a successful request to an endpoint.
    pub fn record_success(&mut self, endpoint: &str) {
        let window_secs = self.config.sliding_window_secs;
        let success_threshold = self.config.success_threshold;

        let breaker = self
            .breakers
            .entry(endpoint.to_string())
            .or_insert_with(EndpointBreaker::new);

        breaker.total_requests += 1;
        breaker.consecutive_failures = 0;
        breaker.consecutive_successes += 1;
        breaker.last_success = Some(Instant::now());

        breaker.outcomes.push_back(RequestOutcome {
            success: true,
            timestamp: Instant::now(),
        });
        Self::trim_window(&mut breaker.outcomes, window_secs);

        match breaker.state {
            CircuitState::HalfOpen => {
                if breaker.half_open_active > 0 {
                    breaker.half_open_active -= 1;
                }
                if breaker.consecutive_successes >= success_threshold {
                    breaker.state = CircuitState::Closed;
                    breaker.state_changed_at = Instant::now();
                    breaker.transitions += 1;
                    info!(endpoint, "Circuit breaker closed (recovered)");
                }
            }
            CircuitState::Closed => {}
            CircuitState::Open => {}
        }
    }

    /// Record a failed request to an endpoint.
    pub fn record_failure(&mut self, endpoint: &str) {
        let window_secs = self.config.sliding_window_secs;
        let failure_threshold = self.config.failure_threshold;
        let failure_rate_threshold = self.config.failure_rate_threshold;
        let min_requests = self.config.min_requests_for_rate;

        let breaker = self
            .breakers
            .entry(endpoint.to_string())
            .or_insert_with(EndpointBreaker::new);

        breaker.total_requests += 1;
        breaker.total_failures += 1;
        breaker.consecutive_failures += 1;
        breaker.consecutive_successes = 0;
        breaker.last_failure = Some(Instant::now());

        breaker.outcomes.push_back(RequestOutcome {
            success: false,
            timestamp: Instant::now(),
        });
        Self::trim_window(&mut breaker.outcomes, window_secs);

        match breaker.state {
            CircuitState::Closed => {
                // Check if we should trip to open
                let should_trip = breaker.consecutive_failures >= failure_threshold
                    || Self::check_failure_rate(
                        &breaker.outcomes,
                        failure_rate_threshold,
                        min_requests,
                    );

                if should_trip {
                    breaker.state = CircuitState::Open;
                    breaker.state_changed_at = Instant::now();
                    breaker.transitions += 1;
                    warn!(
                        endpoint,
                        consecutive_failures = breaker.consecutive_failures,
                        "Circuit breaker tripped to Open"
                    );
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open trips back to open
                breaker.state = CircuitState::Open;
                breaker.state_changed_at = Instant::now();
                breaker.transitions += 1;
                if breaker.half_open_active > 0 {
                    breaker.half_open_active -= 1;
                }
                warn!(endpoint, "Circuit breaker re-opened from Half-Open");
            }
            CircuitState::Open => {}
        }
    }

    /// Get the current state of an endpoint's circuit breaker.
    pub fn get_state(&self, endpoint: &str) -> CircuitState {
        self.breakers
            .get(endpoint)
            .map_or(CircuitState::Closed, |b| b.state)
    }

    /// Get statistics for a specific endpoint.
    pub fn endpoint_stats(&self, endpoint: &str) -> Option<EndpointBreakerStats> {
        self.breakers.get(endpoint).map(|b| {
            let failure_rate = Self::compute_failure_rate(&b.outcomes);
            EndpointBreakerStats {
                endpoint: endpoint.to_string(),
                state: b.state,
                failure_rate,
                total_requests: b.total_requests,
                total_failures: b.total_failures,
                rejected_count: b.rejected_count,
                transitions: b.transitions,
                health_score: Self::compute_health_score(b),
                state_duration_ms: b.state_changed_at.elapsed().as_millis() as u64,
            }
        })
    }

    /// Get aggregate statistics.
    pub fn aggregate_stats(&self) -> AggregateCircuitBreakerStats {
        let mut stats = AggregateCircuitBreakerStats::default();
        let mut total_health = 0.0;

        stats.total_endpoints = self.breakers.len();
        for b in self.breakers.values() {
            match b.state {
                CircuitState::Closed => stats.closed_count += 1,
                CircuitState::Open => stats.open_count += 1,
                CircuitState::HalfOpen => stats.half_open_count += 1,
            }
            stats.total_requests += b.total_requests;
            stats.total_failures += b.total_failures;
            total_health += Self::compute_health_score(b);
        }

        if stats.total_endpoints > 0 {
            stats.avg_health_score = total_health / stats.total_endpoints as f64;
        }

        stats
    }

    /// Get all endpoint identifiers.
    pub fn endpoints(&self) -> Vec<String> {
        self.breakers.keys().cloned().collect()
    }

    /// Reset a specific endpoint's circuit breaker.
    pub fn reset_endpoint(&mut self, endpoint: &str) -> bool {
        if let Some(breaker) = self.breakers.get_mut(endpoint) {
            breaker.state = CircuitState::Closed;
            breaker.state_changed_at = Instant::now();
            breaker.consecutive_failures = 0;
            breaker.consecutive_successes = 0;
            breaker.outcomes.clear();
            info!(endpoint, "Circuit breaker manually reset");
            true
        } else {
            false
        }
    }

    /// Remove an endpoint from tracking.
    pub fn remove_endpoint(&mut self, endpoint: &str) -> bool {
        self.breakers.remove(endpoint).is_some()
    }

    /// Get the configuration.
    pub fn config(&self) -> &EndpointCircuitBreakerConfig {
        &self.config
    }

    /// Number of tracked endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.breakers.len()
    }

    /// Get all healthy endpoints (Closed state).
    pub fn healthy_endpoints(&self) -> Vec<String> {
        self.breakers
            .iter()
            .filter(|(_, b)| b.state == CircuitState::Closed)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Get all unhealthy endpoints (Open state).
    pub fn unhealthy_endpoints(&self) -> Vec<String> {
        self.breakers
            .iter()
            .filter(|(_, b)| b.state == CircuitState::Open)
            .map(|(k, _)| k.clone())
            .collect()
    }

    // ─── Internal helpers ───

    fn trim_window(outcomes: &mut VecDeque<RequestOutcome>, window_secs: u64) {
        let cutoff = Duration::from_secs(window_secs);
        while let Some(front) = outcomes.front() {
            if front.timestamp.elapsed() > cutoff {
                outcomes.pop_front();
            } else {
                break;
            }
        }
    }

    fn compute_failure_rate(outcomes: &VecDeque<RequestOutcome>) -> f64 {
        if outcomes.is_empty() {
            return 0.0;
        }
        let failures = outcomes.iter().filter(|o| !o.success).count();
        failures as f64 / outcomes.len() as f64
    }

    fn check_failure_rate(
        outcomes: &VecDeque<RequestOutcome>,
        threshold: f64,
        min_requests: u32,
    ) -> bool {
        if outcomes.len() < min_requests as usize {
            return false;
        }
        Self::compute_failure_rate(outcomes) >= threshold
    }

    fn compute_health_score(breaker: &EndpointBreaker) -> f64 {
        match breaker.state {
            CircuitState::Closed => {
                let rate = Self::compute_failure_rate(&breaker.outcomes);
                1.0 - rate
            }
            CircuitState::HalfOpen => 0.3,
            CircuitState::Open => 0.0,
        }
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_mgr() -> EndpointCircuitBreakerManager {
        EndpointCircuitBreakerManager::with_defaults()
    }

    fn fast_mgr() -> EndpointCircuitBreakerManager {
        EndpointCircuitBreakerManager::new(EndpointCircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            open_duration: Duration::from_millis(50),
            ..Default::default()
        })
    }

    #[test]
    fn test_initial_state_is_closed() {
        let mgr = default_mgr();
        assert_eq!(
            mgr.get_state("http://example.com/sparql"),
            CircuitState::Closed
        );
    }

    #[test]
    fn test_allow_request_closed() {
        let mut mgr = default_mgr();
        assert!(mgr.allow_request("http://example.com/sparql"));
    }

    #[test]
    fn test_trip_on_failures() {
        let mut mgr = fast_mgr();
        let ep = "http://ep1.example.com/sparql";
        for _ in 0..3 {
            mgr.allow_request(ep);
            mgr.record_failure(ep);
        }
        assert_eq!(mgr.get_state(ep), CircuitState::Open);
    }

    #[test]
    fn test_reject_when_open() {
        let mut mgr = fast_mgr();
        let ep = "http://ep1.example.com/sparql";
        for _ in 0..3 {
            mgr.allow_request(ep);
            mgr.record_failure(ep);
        }
        assert!(!mgr.allow_request(ep));
    }

    #[test]
    fn test_half_open_after_timeout() {
        let mut mgr = fast_mgr();
        let ep = "http://ep1.example.com/sparql";
        for _ in 0..3 {
            mgr.allow_request(ep);
            mgr.record_failure(ep);
        }
        std::thread::sleep(Duration::from_millis(100));
        assert!(mgr.allow_request(ep));
        assert_eq!(mgr.get_state(ep), CircuitState::HalfOpen);
    }

    #[test]
    fn test_close_from_half_open() {
        let mut mgr = fast_mgr();
        let ep = "http://ep1.example.com/sparql";
        for _ in 0..3 {
            mgr.allow_request(ep);
            mgr.record_failure(ep);
        }
        std::thread::sleep(Duration::from_millis(100));
        mgr.allow_request(ep);
        mgr.record_success(ep);
        mgr.record_success(ep);
        assert_eq!(mgr.get_state(ep), CircuitState::Closed);
    }

    #[test]
    fn test_reopen_from_half_open() {
        let mut mgr = fast_mgr();
        let ep = "http://ep1.example.com/sparql";
        for _ in 0..3 {
            mgr.allow_request(ep);
            mgr.record_failure(ep);
        }
        std::thread::sleep(Duration::from_millis(100));
        mgr.allow_request(ep);
        mgr.record_failure(ep);
        assert_eq!(mgr.get_state(ep), CircuitState::Open);
    }

    #[test]
    fn test_success_resets_consecutive_failures() {
        let mut mgr = fast_mgr();
        let ep = "http://ep1.example.com/sparql";
        mgr.allow_request(ep);
        mgr.record_failure(ep);
        mgr.allow_request(ep);
        mgr.record_failure(ep);
        mgr.allow_request(ep);
        mgr.record_success(ep); // Reset consecutive
        mgr.allow_request(ep);
        mgr.record_failure(ep);
        // Only 1 consecutive failure, threshold is 3
        assert_eq!(mgr.get_state(ep), CircuitState::Closed);
    }

    #[test]
    fn test_independent_endpoints() {
        let mut mgr = fast_mgr();
        let ep1 = "http://ep1.example.com/sparql";
        let ep2 = "http://ep2.example.com/sparql";

        for _ in 0..3 {
            mgr.allow_request(ep1);
            mgr.record_failure(ep1);
        }
        assert_eq!(mgr.get_state(ep1), CircuitState::Open);
        assert_eq!(mgr.get_state(ep2), CircuitState::Closed);
    }

    #[test]
    fn test_endpoint_stats() {
        let mut mgr = default_mgr();
        let ep = "http://ep.example.com/sparql";
        mgr.allow_request(ep);
        mgr.record_success(ep);
        mgr.allow_request(ep);
        mgr.record_failure(ep);

        let stats = mgr.endpoint_stats(ep).expect("stats should exist");
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.total_failures, 1);
        assert_eq!(stats.state, CircuitState::Closed);
    }

    #[test]
    fn test_aggregate_stats() {
        let mut mgr = fast_mgr();
        mgr.allow_request("ep1");
        mgr.record_success("ep1");
        mgr.allow_request("ep2");
        mgr.record_success("ep2");

        let stats = mgr.aggregate_stats();
        assert_eq!(stats.total_endpoints, 2);
        assert_eq!(stats.closed_count, 2);
        assert_eq!(stats.total_requests, 2);
    }

    #[test]
    fn test_healthy_endpoints() {
        let mut mgr = fast_mgr();
        mgr.allow_request("ep1");
        mgr.record_success("ep1");
        for _ in 0..3 {
            mgr.allow_request("ep2");
            mgr.record_failure("ep2");
        }

        let healthy = mgr.healthy_endpoints();
        assert!(healthy.contains(&"ep1".to_string()));
        assert!(!healthy.contains(&"ep2".to_string()));
    }

    #[test]
    fn test_unhealthy_endpoints() {
        let mut mgr = fast_mgr();
        for _ in 0..3 {
            mgr.allow_request("ep1");
            mgr.record_failure("ep1");
        }
        let unhealthy = mgr.unhealthy_endpoints();
        assert!(unhealthy.contains(&"ep1".to_string()));
    }

    #[test]
    fn test_reset_endpoint() {
        let mut mgr = fast_mgr();
        for _ in 0..3 {
            mgr.allow_request("ep1");
            mgr.record_failure("ep1");
        }
        assert_eq!(mgr.get_state("ep1"), CircuitState::Open);

        mgr.reset_endpoint("ep1");
        assert_eq!(mgr.get_state("ep1"), CircuitState::Closed);
    }

    #[test]
    fn test_reset_nonexistent() {
        let mut mgr = default_mgr();
        assert!(!mgr.reset_endpoint("nonexistent"));
    }

    #[test]
    fn test_remove_endpoint() {
        let mut mgr = default_mgr();
        mgr.allow_request("ep1");
        mgr.record_success("ep1");
        assert!(mgr.remove_endpoint("ep1"));
        assert_eq!(mgr.endpoint_count(), 0);
    }

    #[test]
    fn test_endpoint_count() {
        let mut mgr = default_mgr();
        mgr.allow_request("ep1");
        mgr.allow_request("ep2");
        mgr.allow_request("ep3");
        assert_eq!(mgr.endpoint_count(), 3);
    }

    #[test]
    fn test_config_defaults() {
        let config = EndpointCircuitBreakerConfig::default();
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.success_threshold, 3);
        assert_eq!(config.open_duration, Duration::from_secs(30));
        assert_eq!(config.half_open_max_requests, 1);
    }

    #[test]
    fn test_config_serialization() {
        let config = EndpointCircuitBreakerConfig::default();
        let json = serde_json::to_string(&config).expect("serialize failed");
        assert!(json.contains("failure_threshold"));
    }

    #[test]
    fn test_stats_serialization() {
        let stats = EndpointBreakerStats {
            endpoint: "ep".into(),
            state: CircuitState::Closed,
            failure_rate: 0.1,
            total_requests: 100,
            total_failures: 10,
            rejected_count: 5,
            transitions: 2,
            health_score: 0.9,
            state_duration_ms: 5000,
        };
        let json = serde_json::to_string(&stats).expect("serialize failed");
        assert!(json.contains("health_score"));
    }

    #[test]
    fn test_health_score_closed() {
        let mut mgr = default_mgr();
        for _ in 0..10 {
            mgr.allow_request("ep");
            mgr.record_success("ep");
        }
        let stats = mgr.endpoint_stats("ep").expect("stats");
        assert!(stats.health_score > 0.9);
    }

    #[test]
    fn test_health_score_open() {
        let mut mgr = fast_mgr();
        for _ in 0..3 {
            mgr.allow_request("ep");
            mgr.record_failure("ep");
        }
        let stats = mgr.endpoint_stats("ep").expect("stats");
        assert!(stats.health_score < 0.01);
    }

    #[test]
    fn test_failure_rate_threshold() {
        let mut mgr = EndpointCircuitBreakerManager::new(EndpointCircuitBreakerConfig {
            failure_threshold: 100, // High count threshold
            failure_rate_threshold: 0.5,
            min_requests_for_rate: 4,
            ..Default::default()
        });

        // 3 successes, then 3 failures = 50% failure rate
        for _ in 0..3 {
            mgr.allow_request("ep");
            mgr.record_success("ep");
        }
        for _ in 0..3 {
            mgr.allow_request("ep");
            mgr.record_failure("ep");
        }
        // With 6 requests and 50% rate >= threshold, should trip
        assert_eq!(mgr.get_state("ep"), CircuitState::Open);
    }

    #[test]
    fn test_rejected_count() {
        let mut mgr = fast_mgr();
        for _ in 0..3 {
            mgr.allow_request("ep");
            mgr.record_failure("ep");
        }
        // 3 rejections
        for _ in 0..5 {
            mgr.allow_request("ep");
        }
        let stats = mgr.endpoint_stats("ep").expect("stats");
        assert!(stats.rejected_count >= 3);
    }

    #[test]
    fn test_endpoints_list() {
        let mut mgr = default_mgr();
        mgr.allow_request("ep1");
        mgr.allow_request("ep2");
        let eps = mgr.endpoints();
        assert_eq!(eps.len(), 2);
    }

    #[test]
    fn test_half_open_max_requests() {
        let mut mgr = EndpointCircuitBreakerManager::new(EndpointCircuitBreakerConfig {
            failure_threshold: 2,
            half_open_max_requests: 1,
            open_duration: Duration::from_millis(50),
            ..Default::default()
        });

        for _ in 0..2 {
            mgr.allow_request("ep");
            mgr.record_failure("ep");
        }
        std::thread::sleep(Duration::from_millis(100));

        // First request in half-open should be allowed
        assert!(mgr.allow_request("ep"));
        // Second should be rejected (max = 1)
        assert!(!mgr.allow_request("ep"));
    }

    #[test]
    fn test_endpoint_stats_none() {
        let mgr = default_mgr();
        assert!(mgr.endpoint_stats("nonexistent").is_none());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut mgr = default_mgr();
        assert!(!mgr.remove_endpoint("nonexistent"));
    }

    #[test]
    fn test_aggregate_health_score() {
        let mut mgr = default_mgr();
        for _ in 0..10 {
            mgr.allow_request("ep1");
            mgr.record_success("ep1");
            mgr.allow_request("ep2");
            mgr.record_success("ep2");
        }
        let stats = mgr.aggregate_stats();
        assert!(stats.avg_health_score > 0.9);
    }
}
