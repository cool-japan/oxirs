//! Endpoint health monitoring for federated SPARQL.
//!
//! Tracks availability, latency, error rates, and reliability of remote
//! SPARQL endpoints using a rolling-window model.  Supports:
//!
//! - Per-endpoint probe recording
//! - Rolling-window eviction (configurable window duration)
//! - Success rate, average latency, p95/p99 percentile calculation
//! - Ranking endpoints by observed latency

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ─── ProbeResult ──────────────────────────────────────────────────────────────

/// A single probe result for an endpoint.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// The endpoint that was probed
    pub endpoint_id: String,
    /// When this probe was conducted
    pub timestamp: Instant,
    /// Whether the probe succeeded
    pub success: bool,
    /// Round-trip latency observed
    pub latency: Duration,
    /// Error message if the probe failed
    pub error: Option<String>,
}

impl ProbeResult {
    /// Create a successful probe result.
    pub fn success(endpoint_id: impl Into<String>, latency: Duration) -> Self {
        Self {
            endpoint_id: endpoint_id.into(),
            timestamp: Instant::now(),
            success: true,
            latency,
            error: None,
        }
    }

    /// Create a failed probe result.
    pub fn failure(
        endpoint_id: impl Into<String>,
        latency: Duration,
        error: impl Into<String>,
    ) -> Self {
        Self {
            endpoint_id: endpoint_id.into(),
            timestamp: Instant::now(),
            success: false,
            latency,
            error: Some(error.into()),
        }
    }
}

// ─── EndpointHealthWindow ────────────────────────────────────────────────────

/// Rolling window of probe results for a single endpoint.
///
/// Probes older than `window_duration` are automatically evicted when
/// statistics are requested or new probes are added.
pub struct EndpointHealthWindow {
    window_duration: Duration,
    probes: VecDeque<ProbeResult>,
}

impl EndpointHealthWindow {
    /// Create a new health window with the given duration.
    pub fn new(window: Duration) -> Self {
        Self {
            window_duration: window,
            probes: VecDeque::new(),
        }
    }

    /// Add a probe result to the window and evict stale entries.
    pub fn add_probe(&mut self, result: ProbeResult) {
        self.probes.push_back(result);
        self.evict_old();
    }

    /// Remove probes that fall outside the rolling window.
    pub fn evict_old(&mut self) {
        let now = Instant::now();
        while let Some(front) = self.probes.front() {
            if now.duration_since(front.timestamp) > self.window_duration {
                self.probes.pop_front();
            } else {
                break;
            }
        }
    }

    /// Fraction of probes in this window that succeeded, in `[0.0, 1.0]`.
    ///
    /// Returns `1.0` when the window is empty (no evidence of failure).
    pub fn success_rate(&self) -> f64 {
        if self.probes.is_empty() {
            return 1.0;
        }
        let successes = self.probes.iter().filter(|p| p.success).count();
        successes as f64 / self.probes.len() as f64
    }

    /// Arithmetic mean latency across all probes in the window.
    ///
    /// Returns `None` when the window is empty.
    pub fn avg_latency(&self) -> Option<Duration> {
        if self.probes.is_empty() {
            return None;
        }
        let total: Duration = self.probes.iter().map(|p| p.latency).sum();
        Some(total / self.probes.len() as u32)
    }

    /// 95th-percentile latency across probes in the window.
    ///
    /// Returns `None` when the window is empty.
    pub fn p95_latency(&self) -> Option<Duration> {
        self.percentile_latency(95)
    }

    /// 99th-percentile latency across probes in the window.
    ///
    /// Returns `None` when the window is empty.
    pub fn p99_latency(&self) -> Option<Duration> {
        self.percentile_latency(99)
    }

    /// Total number of probes currently in the window.
    pub fn total_probes(&self) -> usize {
        self.probes.len()
    }

    /// Returns `true` when the success rate exceeds 95%.
    pub fn is_healthy(&self) -> bool {
        self.success_rate() > 0.95
    }

    // ── helpers ──

    fn percentile_latency(&self, percentile: usize) -> Option<Duration> {
        if self.probes.is_empty() {
            return None;
        }
        let mut latencies: Vec<Duration> = self.probes.iter().map(|p| p.latency).collect();
        latencies.sort();
        // Index calculation: (percentile / 100) * n, clamped to valid range
        let idx = ((percentile * latencies.len()) / 100)
            .saturating_sub(1)
            .min(latencies.len() - 1);
        Some(latencies[idx])
    }
}

// ─── EndpointStatus ──────────────────────────────────────────────────────────

/// Derived health status of a single endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EndpointStatus {
    /// Success rate > 95%, endpoint is operating normally
    Healthy,
    /// Success rate between 50% and 95%
    Degraded { success_rate: f64 },
    /// Success rate below 50% or a definitive failure has been recorded
    Unhealthy { reason: String },
    /// Fewer than `min_probes_for_status` probes in the current window
    Unknown,
}

impl EndpointStatus {
    /// Return `true` when the endpoint is healthy or its status is unknown.
    pub fn is_usable(&self) -> bool {
        matches!(self, EndpointStatus::Healthy | EndpointStatus::Unknown)
    }
}

// ─── EndpointHealthMonitor ───────────────────────────────────────────────────

/// Central health monitor for all registered endpoints.
///
/// Aggregates [`ProbeResult`]s per endpoint into [`EndpointHealthWindow`]s
/// and exposes summary statistics and ranked endpoint lists.
pub struct EndpointHealthMonitor {
    windows: HashMap<String, EndpointHealthWindow>,
    window_size: Duration,
    min_probes_for_status: usize,
}

impl EndpointHealthMonitor {
    /// Create a new monitor with the specified rolling-window size.
    ///
    /// `min_probes_for_status` controls how many probes must be observed
    /// before a conclusive status (other than `Unknown`) is reported.
    pub fn new(window_size: Duration) -> Self {
        Self {
            windows: HashMap::new(),
            window_size,
            min_probes_for_status: 3,
        }
    }

    /// Create a monitor with a custom minimum probe threshold.
    pub fn with_min_probes(mut self, min_probes: usize) -> Self {
        self.min_probes_for_status = min_probes;
        self
    }

    /// Register an endpoint so that its window is pre-allocated.
    pub fn register(&mut self, endpoint_id: &str) {
        self.windows
            .entry(endpoint_id.to_string())
            .or_insert_with(|| EndpointHealthWindow::new(self.window_size));
    }

    /// Record a probe result for an endpoint, registering it if needed.
    pub fn record_probe(&mut self, result: ProbeResult) {
        let window_size = self.window_size;
        self.windows
            .entry(result.endpoint_id.clone())
            .or_insert_with(|| EndpointHealthWindow::new(window_size))
            .add_probe(result);
    }

    /// Derive the current [`EndpointStatus`] for an endpoint.
    pub fn status(&self, endpoint_id: &str) -> EndpointStatus {
        let window = match self.windows.get(endpoint_id) {
            Some(w) => w,
            None => return EndpointStatus::Unknown,
        };

        if window.total_probes() < self.min_probes_for_status {
            return EndpointStatus::Unknown;
        }

        let rate = window.success_rate();

        if rate > 0.95 {
            EndpointStatus::Healthy
        } else if rate >= 0.50 {
            EndpointStatus::Degraded { success_rate: rate }
        } else {
            EndpointStatus::Unhealthy {
                reason: format!("Success rate {:.1} % is below 50%", rate * 100.0),
            }
        }
    }

    /// Return a snapshot of statuses for all registered endpoints.
    pub fn all_statuses(&self) -> HashMap<String, EndpointStatus> {
        self.windows
            .keys()
            .map(|id| (id.clone(), self.status(id)))
            .collect()
    }

    /// Return the IDs of all endpoints currently classified as `Healthy`.
    pub fn healthy_endpoints(&self) -> Vec<String> {
        self.windows
            .keys()
            .filter(|id| matches!(self.status(id), EndpointStatus::Healthy))
            .cloned()
            .collect()
    }

    /// Sort a slice of endpoint IDs by their observed average latency
    /// (ascending, lowest latency first).
    ///
    /// Endpoints without latency data are placed at the end.
    pub fn sort_by_latency(&self, endpoint_ids: &[String]) -> Vec<String> {
        let mut with_latency: Vec<(String, Duration)> = endpoint_ids
            .iter()
            .map(|id| {
                let lat = self
                    .windows
                    .get(id)
                    .and_then(|w| w.avg_latency())
                    .unwrap_or(Duration::MAX);
                (id.clone(), lat)
            })
            .collect();
        with_latency.sort_by_key(|(_, lat)| *lat);
        with_latency.into_iter().map(|(id, _)| id).collect()
    }

    /// Return the average latency for an endpoint, if available.
    pub fn avg_latency(&self, endpoint_id: &str) -> Option<Duration> {
        self.windows.get(endpoint_id).and_then(|w| w.avg_latency())
    }

    /// Return the p95 latency for an endpoint, if available.
    pub fn p95_latency(&self, endpoint_id: &str) -> Option<Duration> {
        self.windows.get(endpoint_id).and_then(|w| w.p95_latency())
    }

    /// Return the p99 latency for an endpoint, if available.
    pub fn p99_latency(&self, endpoint_id: &str) -> Option<Duration> {
        self.windows.get(endpoint_id).and_then(|w| w.p99_latency())
    }

    /// Return the number of probes currently tracked for an endpoint.
    pub fn probe_count(&self, endpoint_id: &str) -> usize {
        self.windows
            .get(endpoint_id)
            .map(|w| w.total_probes())
            .unwrap_or(0)
    }

    /// Return the success rate for an endpoint in `[0.0, 1.0]`.
    pub fn success_rate(&self, endpoint_id: &str) -> f64 {
        self.windows
            .get(endpoint_id)
            .map(|w| w.success_rate())
            .unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_probe(id: &str, success: bool, latency_ms: u64) -> ProbeResult {
        ProbeResult {
            endpoint_id: id.to_string(),
            timestamp: Instant::now(),
            success,
            latency: Duration::from_millis(latency_ms),
            error: if success {
                None
            } else {
                Some("timeout".to_string())
            },
        }
    }

    // ── EndpointHealthWindow ─────────────────────────────────────────────────

    #[test]
    fn test_empty_window_success_rate() {
        let window = EndpointHealthWindow::new(Duration::from_secs(60));
        assert_eq!(window.success_rate(), 1.0);
        assert!(window.avg_latency().is_none());
        assert!(window.p95_latency().is_none());
        assert!(window.is_healthy());
    }

    #[test]
    fn test_window_success_rate_all_success() {
        let mut window = EndpointHealthWindow::new(Duration::from_secs(60));
        for _ in 0..10 {
            window.add_probe(make_probe("ep1", true, 50));
        }
        assert!((window.success_rate() - 1.0).abs() < f64::EPSILON);
        assert!(window.is_healthy());
    }

    #[test]
    fn test_window_success_rate_mixed() {
        let mut window = EndpointHealthWindow::new(Duration::from_secs(60));
        for _ in 0..8 {
            window.add_probe(make_probe("ep1", true, 50));
        }
        for _ in 0..2 {
            window.add_probe(make_probe("ep1", false, 5000));
        }
        let rate = window.success_rate();
        assert!((rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_window_avg_latency() {
        let mut window = EndpointHealthWindow::new(Duration::from_secs(60));
        window.add_probe(make_probe("ep1", true, 100));
        window.add_probe(make_probe("ep1", true, 200));
        window.add_probe(make_probe("ep1", true, 300));

        let avg = window.avg_latency().expect("should have latency");
        assert_eq!(avg, Duration::from_millis(200));
    }

    #[test]
    fn test_window_p95_latency() {
        let mut window = EndpointHealthWindow::new(Duration::from_secs(60));
        for i in 1..=100u64 {
            window.add_probe(make_probe("ep1", true, i * 10));
        }
        let p95 = window.p95_latency().expect("should have p95");
        // 95th index (0-based) of sorted [10, 20, ..., 1000] is element at index 94 = 950ms
        assert_eq!(p95, Duration::from_millis(950));
    }

    #[test]
    fn test_window_evicts_old_probes() {
        // Use a very short window
        let mut window = EndpointHealthWindow::new(Duration::from_millis(1));
        let mut probe = make_probe("ep1", true, 50);
        // Back-date the timestamp so it falls outside the window
        probe.timestamp = Instant::now() - Duration::from_secs(10);
        window.probes.push_back(probe);

        // Evict when adding a new probe
        window.add_probe(make_probe("ep1", true, 50));
        // Only the fresh probe should remain
        assert_eq!(window.total_probes(), 1);
    }

    #[test]
    fn test_is_healthy_threshold() {
        let mut window = EndpointHealthWindow::new(Duration::from_secs(60));
        // 95/100 = 0.95 — exactly on the boundary (not > 0.95)
        for _ in 0..95 {
            window.add_probe(make_probe("ep1", true, 50));
        }
        for _ in 0..5 {
            window.add_probe(make_probe("ep1", false, 5000));
        }
        assert!(!window.is_healthy()); // 0.95 is NOT > 0.95
    }

    // ── EndpointHealthMonitor ────────────────────────────────────────────────

    #[test]
    fn test_unknown_status_for_unregistered() {
        let monitor = EndpointHealthMonitor::new(Duration::from_secs(60));
        assert_eq!(monitor.status("missing"), EndpointStatus::Unknown);
    }

    #[test]
    fn test_unknown_status_with_too_few_probes() {
        let mut monitor = EndpointHealthMonitor::new(Duration::from_secs(60));
        monitor.register("ep1");
        monitor.record_probe(make_probe("ep1", true, 50));
        monitor.record_probe(make_probe("ep1", true, 50));
        // min_probes_for_status defaults to 3, so 2 probes → Unknown
        assert_eq!(monitor.status("ep1"), EndpointStatus::Unknown);
    }

    #[test]
    fn test_healthy_status() {
        let mut monitor = EndpointHealthMonitor::new(Duration::from_secs(60));
        for _ in 0..10 {
            monitor.record_probe(make_probe("ep1", true, 50));
        }
        assert_eq!(monitor.status("ep1"), EndpointStatus::Healthy);
    }

    #[test]
    fn test_degraded_status() {
        let mut monitor = EndpointHealthMonitor::new(Duration::from_secs(60));
        for _ in 0..7 {
            monitor.record_probe(make_probe("ep1", true, 50));
        }
        for _ in 0..3 {
            monitor.record_probe(make_probe("ep1", false, 5000));
        }
        let status = monitor.status("ep1");
        assert!(matches!(status, EndpointStatus::Degraded { .. }));
    }

    #[test]
    fn test_unhealthy_status() {
        let mut monitor = EndpointHealthMonitor::new(Duration::from_secs(60));
        for _ in 0..3 {
            monitor.record_probe(make_probe("ep1", false, 5000));
        }
        let status = monitor.status("ep1");
        assert!(matches!(status, EndpointStatus::Unhealthy { .. }));
    }

    #[test]
    fn test_healthy_endpoints_list() {
        let mut monitor = EndpointHealthMonitor::new(Duration::from_secs(60));
        for _ in 0..5 {
            monitor.record_probe(make_probe("ep1", true, 50));
        }
        for _ in 0..5 {
            monitor.record_probe(make_probe("ep2", false, 5000));
        }
        let healthy = monitor.healthy_endpoints();
        assert!(healthy.contains(&"ep1".to_string()));
        assert!(!healthy.contains(&"ep2".to_string()));
    }

    #[test]
    fn test_sort_by_latency() {
        let mut monitor = EndpointHealthMonitor::new(Duration::from_secs(60));
        for _ in 0..5 {
            monitor.record_probe(make_probe("fast", true, 10));
        }
        for _ in 0..5 {
            monitor.record_probe(make_probe("slow", true, 500));
        }
        let ids = vec!["slow".to_string(), "fast".to_string()];
        let sorted = monitor.sort_by_latency(&ids);
        assert_eq!(sorted[0], "fast");
        assert_eq!(sorted[1], "slow");
    }

    #[test]
    fn test_all_statuses_snapshot() {
        let mut monitor = EndpointHealthMonitor::new(Duration::from_secs(60));
        monitor.register("ep1");
        monitor.register("ep2");
        let statuses = monitor.all_statuses();
        assert_eq!(statuses.len(), 2);
        // Both have zero probes → Unknown
        assert_eq!(statuses["ep1"], EndpointStatus::Unknown);
        assert_eq!(statuses["ep2"], EndpointStatus::Unknown);
    }

    #[test]
    fn test_probe_count() {
        let mut monitor = EndpointHealthMonitor::new(Duration::from_secs(60));
        assert_eq!(monitor.probe_count("ep1"), 0);
        monitor.record_probe(make_probe("ep1", true, 50));
        assert_eq!(monitor.probe_count("ep1"), 1);
    }

    #[test]
    fn test_success_rate_for_unknown_endpoint() {
        let monitor = EndpointHealthMonitor::new(Duration::from_secs(60));
        // Unknown endpoint → optimistic 1.0
        assert_eq!(monitor.success_rate("nonexistent"), 1.0);
    }

    #[test]
    fn test_p95_p99_latency() {
        let mut monitor = EndpointHealthMonitor::new(Duration::from_secs(60));
        for i in 1..=100u64 {
            monitor.record_probe(make_probe("ep1", true, i * 10));
        }
        assert!(monitor.p95_latency("ep1").is_some());
        assert!(monitor.p99_latency("ep1").is_some());
    }

    #[test]
    fn test_endpoint_status_is_usable() {
        assert!(EndpointStatus::Healthy.is_usable());
        assert!(EndpointStatus::Unknown.is_usable());
        assert!(!EndpointStatus::Degraded { success_rate: 0.7 }.is_usable());
        assert!(!EndpointStatus::Unhealthy {
            reason: "down".into()
        }
        .is_usable());
    }
}
