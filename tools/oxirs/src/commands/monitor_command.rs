//! Real-time SPARQL endpoint monitoring command.
//!
//! Records latency and availability metrics for a SPARQL endpoint, computes
//! P95 latency and uptime percentages, and exposes health-check utilities —
//! all without performing actual network I/O (those are integration concerns).

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the endpoint monitor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitorConfig {
    /// URL of the SPARQL endpoint, e.g. `"http://localhost:3030/sparql"`.
    pub endpoint_url: String,
    /// How often to probe the endpoint (seconds).
    pub interval_secs: u64,
    /// Latency threshold in milliseconds above which an alert is raised.
    pub alert_threshold_ms: u64,
    /// Number of consecutive errors before the endpoint is considered down.
    pub max_errors: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            endpoint_url: "http://localhost:3030/sparql".to_string(),
            interval_secs: 30,
            alert_threshold_ms: 5_000,
            max_errors: 3,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metric types
// ─────────────────────────────────────────────────────────────────────────────

/// Status of a single endpoint probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryStatus {
    /// The probe succeeded within the latency threshold.
    Ok,
    /// The probe timed out.
    Timeout,
    /// The probe returned an error.
    Error(String),
}

/// One recorded endpoint check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointMetric {
    /// Unix epoch (seconds) when the check was performed.
    pub timestamp: u64,
    /// Round-trip latency in milliseconds.
    pub latency_ms: u64,
    /// Outcome of the probe.
    pub status: QueryStatus,
    /// Triple count returned by a lightweight ASK/SELECT, if available.
    pub triple_count: Option<usize>,
}

impl EndpointMetric {
    /// Create a successful metric.
    pub fn ok(timestamp: u64, latency_ms: u64, triple_count: Option<usize>) -> Self {
        Self {
            timestamp,
            latency_ms,
            status: QueryStatus::Ok,
            triple_count,
        }
    }

    /// Create a timeout metric.
    pub fn timeout(timestamp: u64, latency_ms: u64) -> Self {
        Self {
            timestamp,
            latency_ms,
            status: QueryStatus::Timeout,
            triple_count: None,
        }
    }

    /// Create an error metric.
    pub fn error(timestamp: u64, latency_ms: u64, msg: impl Into<String>) -> Self {
        Self {
            timestamp,
            latency_ms,
            status: QueryStatus::Error(msg.into()),
            triple_count: None,
        }
    }

    /// Return `true` if the status is `Ok`.
    pub fn is_ok(&self) -> bool {
        self.status == QueryStatus::Ok
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated monitoring statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct MonitorStats {
    /// Total number of checks performed.
    pub total_checks: usize,
    /// Number of successful checks (status `Ok`).
    pub ok_count: usize,
    /// Number of failed checks (status `Timeout` or `Error`).
    pub error_count: usize,
    /// Mean latency in milliseconds over all checks.
    pub avg_latency_ms: f64,
    /// 95th-percentile latency in milliseconds.
    pub p95_latency_ms: f64,
    /// Uptime percentage: `ok_count / total_checks * 100`.
    pub uptime_pct: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// MonitorState
// ─────────────────────────────────────────────────────────────────────────────

/// Holds the running state for a monitoring session.
#[derive(Debug)]
pub struct MonitorState {
    config: MonitorConfig,
    history: VecDeque<EndpointMetric>,
    max_history: usize,
}

impl MonitorState {
    /// Create a new monitor state.
    ///
    /// `max_history` caps the number of metrics retained in memory.
    pub fn new(config: MonitorConfig, max_history: usize) -> Self {
        Self {
            config,
            history: VecDeque::with_capacity(max_history.min(4096)),
            max_history: max_history.max(1),
        }
    }

    /// Record a new metric.  Evicts the oldest entry if the history is full.
    pub fn record(&mut self, metric: EndpointMetric) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(metric);
    }

    /// Compute aggregate statistics over all recorded metrics.
    pub fn stats(&self) -> MonitorStats {
        let total_checks = self.history.len();
        if total_checks == 0 {
            return MonitorStats {
                total_checks: 0,
                ok_count: 0,
                error_count: 0,
                avg_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                uptime_pct: 100.0,
            };
        }

        let ok_count = self.history.iter().filter(|m| m.is_ok()).count();
        let error_count = total_checks - ok_count;

        let latencies: Vec<u64> = self.history.iter().map(|m| m.latency_ms).collect();
        let avg_latency_ms = latencies.iter().sum::<u64>() as f64 / total_checks as f64;
        let p95_latency_ms = Self::compute_p95(&latencies) as f64;

        let uptime_pct = if total_checks == 0 {
            100.0
        } else {
            ok_count as f64 / total_checks as f64 * 100.0
        };

        MonitorStats {
            total_checks,
            ok_count,
            error_count,
            avg_latency_ms,
            p95_latency_ms,
            uptime_pct,
        }
    }

    /// Return the `n` most-recent metrics (fewer if the history is shorter).
    pub fn recent(&self, n: usize) -> Vec<&EndpointMetric> {
        // Collect from the tail of the deque, preserving chronological order.
        // Using iter().rev().take(n).collect() then reversing is correct for
        // all VecDeque internal layouts (wrapping or contiguous).
        let mut result: Vec<&EndpointMetric> = self.history.iter().rev().take(n).collect();
        result.reverse();
        result
    }

    /// Return `true` if the last `max_errors` consecutive checks all succeeded.
    ///
    /// Returns `true` if there are no checks yet.
    pub fn is_healthy(&self) -> bool {
        let n = self.config.max_errors;
        if self.history.is_empty() {
            return true;
        }
        let tail_len = self.history.len().min(n);
        self.history.iter().rev().take(tail_len).all(|m| m.is_ok())
    }

    /// Count the number of consecutive errors at the tail of the history.
    pub fn consecutive_errors(&self) -> usize {
        self.history.iter().rev().take_while(|m| !m.is_ok()).count()
    }

    /// Compute the 95th-percentile latency from a slice of values.
    ///
    /// Returns 0 for an empty slice.
    pub fn compute_p95(latencies: &[u64]) -> u64 {
        if latencies.is_empty() {
            return 0;
        }
        let mut sorted = latencies.to_vec();
        sorted.sort_unstable();
        // Index for P95: ceil(0.95 * n) − 1 clamped to [0, n−1]
        let idx = ((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1);
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Access the monitor configuration.
    pub fn config(&self) -> &MonitorConfig {
        &self.config
    }

    /// Access the full history (read-only).
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_state() -> MonitorState {
        MonitorState::new(MonitorConfig::default(), 100)
    }

    // ── MonitorConfig::default ────────────────────────────────────────────────

    #[test]
    fn test_default_config_url() {
        let c = MonitorConfig::default();
        assert!(c.endpoint_url.contains("localhost"));
    }

    #[test]
    fn test_default_config_interval() {
        let c = MonitorConfig::default();
        assert_eq!(c.interval_secs, 30);
    }

    #[test]
    fn test_default_config_alert_threshold() {
        let c = MonitorConfig::default();
        assert_eq!(c.alert_threshold_ms, 5_000);
    }

    #[test]
    fn test_default_config_max_errors() {
        let c = MonitorConfig::default();
        assert_eq!(c.max_errors, 3);
    }

    // ── MonitorConfig clone / eq ──────────────────────────────────────────────

    #[test]
    fn test_monitor_config_clone() {
        let c = MonitorConfig::default();
        let c2 = c.clone();
        assert_eq!(c, c2);
    }

    // ── EndpointMetric constructors ────────────────────────────────────────────

    #[test]
    fn test_metric_ok_is_ok() {
        let m = EndpointMetric::ok(1000, 50, None);
        assert!(m.is_ok());
        assert_eq!(m.status, QueryStatus::Ok);
    }

    #[test]
    fn test_metric_timeout_not_ok() {
        let m = EndpointMetric::timeout(1000, 5001);
        assert!(!m.is_ok());
    }

    #[test]
    fn test_metric_error_not_ok() {
        let m = EndpointMetric::error(1000, 100, "connection refused");
        assert!(!m.is_ok());
        match &m.status {
            QueryStatus::Error(msg) => assert!(msg.contains("connection")),
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn test_metric_ok_triple_count() {
        let m = EndpointMetric::ok(1000, 20, Some(1_000_000));
        assert_eq!(m.triple_count, Some(1_000_000));
    }

    #[test]
    fn test_metric_timeout_no_triple_count() {
        let m = EndpointMetric::timeout(1000, 6000);
        assert!(m.triple_count.is_none());
    }

    // ── MonitorState::new ─────────────────────────────────────────────────────

    #[test]
    fn test_new_empty_history() {
        let state = default_state();
        assert_eq!(state.history_len(), 0);
    }

    #[test]
    fn test_new_max_history_at_least_one() {
        let state = MonitorState::new(MonitorConfig::default(), 0);
        assert_eq!(state.max_history, 1);
    }

    // ── record ────────────────────────────────────────────────────────────────

    #[test]
    fn test_record_single() {
        let mut state = default_state();
        state.record(EndpointMetric::ok(1, 10, None));
        assert_eq!(state.history_len(), 1);
    }

    #[test]
    fn test_record_evicts_oldest() {
        let mut state = MonitorState::new(MonitorConfig::default(), 3);
        for i in 0..5u64 {
            state.record(EndpointMetric::ok(i, i * 10, None));
        }
        assert_eq!(state.history_len(), 3);
    }

    #[test]
    fn test_record_preserves_newest() {
        let mut state = MonitorState::new(MonitorConfig::default(), 2);
        state.record(EndpointMetric::ok(1, 10, None));
        state.record(EndpointMetric::ok(2, 20, None));
        state.record(EndpointMetric::ok(3, 30, None));
        let recent = state.recent(2);
        assert!(recent.iter().any(|m| m.timestamp == 2));
        assert!(recent.iter().any(|m| m.timestamp == 3));
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let state = default_state();
        let s = state.stats();
        assert_eq!(s.total_checks, 0);
        assert_eq!(s.uptime_pct, 100.0);
    }

    #[test]
    fn test_stats_all_ok() {
        let mut state = default_state();
        for i in 0..5u64 {
            state.record(EndpointMetric::ok(i, 100, None));
        }
        let s = state.stats();
        assert_eq!(s.ok_count, 5);
        assert_eq!(s.error_count, 0);
        assert!((s.uptime_pct - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_mixed() {
        let mut state = default_state();
        state.record(EndpointMetric::ok(1, 100, None));
        state.record(EndpointMetric::error(2, 50, "err"));
        let s = state.stats();
        assert_eq!(s.total_checks, 2);
        assert_eq!(s.ok_count, 1);
        assert_eq!(s.error_count, 1);
        assert!((s.uptime_pct - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_avg_latency() {
        let mut state = default_state();
        state.record(EndpointMetric::ok(1, 100, None));
        state.record(EndpointMetric::ok(2, 200, None));
        let s = state.stats();
        assert!((s.avg_latency_ms - 150.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_p95_single() {
        let mut state = default_state();
        state.record(EndpointMetric::ok(1, 42, None));
        let s = state.stats();
        assert_eq!(s.p95_latency_ms as u64, 42);
    }

    // ── recent ────────────────────────────────────────────────────────────────

    #[test]
    fn test_recent_empty() {
        let state = default_state();
        assert!(state.recent(5).is_empty());
    }

    #[test]
    fn test_recent_fewer_than_requested() {
        let mut state = default_state();
        state.record(EndpointMetric::ok(1, 10, None));
        assert_eq!(state.recent(10).len(), 1);
    }

    #[test]
    fn test_recent_exact() {
        let mut state = default_state();
        for i in 0..5u64 {
            state.record(EndpointMetric::ok(i, 10, None));
        }
        assert_eq!(state.recent(5).len(), 5);
    }

    // ── is_healthy ────────────────────────────────────────────────────────────

    #[test]
    fn test_is_healthy_empty() {
        let state = default_state();
        assert!(state.is_healthy());
    }

    #[test]
    fn test_is_healthy_all_ok() {
        let mut state = default_state();
        for i in 0..5u64 {
            state.record(EndpointMetric::ok(i, 10, None));
        }
        assert!(state.is_healthy());
    }

    #[test]
    fn test_is_healthy_recent_error() {
        let mut state = default_state();
        state.record(EndpointMetric::ok(1, 10, None));
        state.record(EndpointMetric::error(2, 20, "err1"));
        state.record(EndpointMetric::error(3, 20, "err2"));
        state.record(EndpointMetric::error(4, 20, "err3"));
        assert!(!state.is_healthy());
    }

    #[test]
    fn test_is_healthy_recovers() {
        let mut state = default_state();
        state.record(EndpointMetric::error(1, 20, "e"));
        state.record(EndpointMetric::ok(2, 10, None));
        state.record(EndpointMetric::ok(3, 10, None));
        state.record(EndpointMetric::ok(4, 10, None));
        assert!(state.is_healthy());
    }

    // ── consecutive_errors ────────────────────────────────────────────────────

    #[test]
    fn test_consecutive_errors_zero_on_ok() {
        let mut state = default_state();
        state.record(EndpointMetric::ok(1, 10, None));
        assert_eq!(state.consecutive_errors(), 0);
    }

    #[test]
    fn test_consecutive_errors_count() {
        let mut state = default_state();
        state.record(EndpointMetric::ok(1, 10, None));
        state.record(EndpointMetric::error(2, 20, "e1"));
        state.record(EndpointMetric::error(3, 20, "e2"));
        assert_eq!(state.consecutive_errors(), 2);
    }

    #[test]
    fn test_consecutive_errors_reset_by_ok() {
        let mut state = default_state();
        state.record(EndpointMetric::error(1, 20, "e"));
        state.record(EndpointMetric::error(2, 20, "e"));
        state.record(EndpointMetric::ok(3, 10, None));
        assert_eq!(state.consecutive_errors(), 0);
    }

    // ── compute_p95 ───────────────────────────────────────────────────────────

    #[test]
    fn test_compute_p95_empty() {
        assert_eq!(MonitorState::compute_p95(&[]), 0);
    }

    #[test]
    fn test_compute_p95_single() {
        assert_eq!(MonitorState::compute_p95(&[77]), 77);
    }

    #[test]
    fn test_compute_p95_two() {
        // n=2, ceil(0.95*2)=2, idx=1 → second element when sorted
        let v = vec![100u64, 50];
        assert_eq!(MonitorState::compute_p95(&v), 100);
    }

    #[test]
    fn test_compute_p95_twenty() {
        // 20 values [1..20]; P95 = ceil(19) - 1 = idx 18 → value 19
        let v: Vec<u64> = (1..=20).collect();
        let p95 = MonitorState::compute_p95(&v);
        assert!((18..=20).contains(&p95), "p95={p95}");
    }

    #[test]
    fn test_compute_p95_uniform() {
        let v = vec![100u64; 100];
        assert_eq!(MonitorState::compute_p95(&v), 100);
    }

    // ── config accessor ───────────────────────────────────────────────────────

    #[test]
    fn test_config_accessor() {
        let c = MonitorConfig {
            endpoint_url: "http://test.example".to_string(),
            interval_secs: 10,
            alert_threshold_ms: 1000,
            max_errors: 5,
        };
        let state = MonitorState::new(c.clone(), 50);
        assert_eq!(state.config().endpoint_url, "http://test.example");
    }

    // ── QueryStatus ───────────────────────────────────────────────────────────

    #[test]
    fn test_query_status_ok_eq() {
        assert_eq!(QueryStatus::Ok, QueryStatus::Ok);
    }

    #[test]
    fn test_query_status_timeout_eq() {
        assert_eq!(QueryStatus::Timeout, QueryStatus::Timeout);
    }

    #[test]
    fn test_query_status_error_eq() {
        assert_eq!(
            QueryStatus::Error("x".to_string()),
            QueryStatus::Error("x".to_string())
        );
    }

    #[test]
    fn test_query_status_error_ne_ok() {
        assert_ne!(QueryStatus::Error("x".to_string()), QueryStatus::Ok);
    }

    // ── MonitorStats fields ───────────────────────────────────────────────────

    #[test]
    fn test_monitor_stats_uptime_all_errors() {
        let mut state = default_state();
        for i in 0..4u64 {
            state.record(EndpointMetric::error(i, 100, "e"));
        }
        let s = state.stats();
        assert_eq!(s.uptime_pct, 0.0);
        assert_eq!(s.ok_count, 0);
        assert_eq!(s.error_count, 4);
    }

    // ── history_len ───────────────────────────────────────────────────────────

    #[test]
    fn test_history_len() {
        let mut state = default_state();
        state.record(EndpointMetric::ok(1, 10, None));
        state.record(EndpointMetric::ok(2, 20, None));
        assert_eq!(state.history_len(), 2);
    }

    // ── EndpointMetric clone ──────────────────────────────────────────────────

    #[test]
    fn test_endpoint_metric_clone() {
        let m = EndpointMetric::ok(1000, 50, Some(42));
        let m2 = m.clone();
        assert_eq!(m, m2);
    }

    // ── compute_p95 sorted order ──────────────────────────────────────────────

    #[test]
    fn test_compute_p95_handles_unsorted() {
        let v = vec![50u64, 10, 90, 30, 70];
        let p95 = MonitorState::compute_p95(&v);
        // sorted: [10, 30, 50, 70, 90], P95 idx = ceil(4.75)-1 = 4 → 90
        assert_eq!(p95, 90);
    }

    // ── consecutive_errors with timeout ──────────────────────────────────────

    #[test]
    fn test_consecutive_errors_counts_timeout() {
        let mut state = default_state();
        state.record(EndpointMetric::ok(1, 10, None));
        state.record(EndpointMetric::timeout(2, 6000));
        state.record(EndpointMetric::timeout(3, 6000));
        assert_eq!(state.consecutive_errors(), 2);
    }

    // ── is_healthy with exactly max_errors failures ───────────────────────────

    #[test]
    fn test_is_healthy_exactly_max_errors() {
        let config = MonitorConfig {
            max_errors: 2,
            ..MonitorConfig::default()
        };
        let mut state = MonitorState::new(config, 50);
        state.record(EndpointMetric::error(1, 10, "e"));
        state.record(EndpointMetric::error(2, 10, "e"));
        assert!(!state.is_healthy());
    }

    // ── stats p95 with all same latency ──────────────────────────────────────

    #[test]
    fn test_stats_p95_all_same() {
        let mut state = default_state();
        for i in 0..10u64 {
            state.record(EndpointMetric::ok(i, 250, None));
        }
        let s = state.stats();
        assert_eq!(s.p95_latency_ms as u64, 250);
    }
}
