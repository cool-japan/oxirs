//! # Endpoint Health Monitor
//!
//! Provides continuous health monitoring for federated SPARQL endpoints with
//! latency tracking, availability scoring, degradation detection, and
//! configurable health check intervals.
//!
//! ## Features
//!
//! - **Periodic health checks**: Configurable interval per endpoint
//! - **Latency tracking**: Rolling window of latency samples with percentile computation
//! - **Availability scoring**: Exponentially weighted availability score
//! - **Degradation detection**: Automatic detection of latency spikes and error rate increases
//! - **Health history**: Bounded history of health snapshots for trend analysis

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the endpoint health monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitorConfig {
    /// How often to perform health checks (default: 30s).
    pub check_interval: Duration,
    /// Timeout for a single health check probe (default: 5s).
    pub check_timeout: Duration,
    /// Number of latency samples to keep (default: 100).
    pub latency_window_size: usize,
    /// Number of health snapshots to keep in history (default: 1000).
    pub history_size: usize,
    /// Latency threshold (ms) above which the endpoint is considered degraded.
    pub degradation_latency_ms: u64,
    /// Error rate threshold (0.0 - 1.0) above which the endpoint is considered degraded.
    pub degradation_error_rate: f64,
    /// Exponential decay factor for availability scoring (0.0 - 1.0).
    pub availability_decay: f64,
    /// Number of consecutive failures before marking endpoint as down.
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking a down endpoint as recovered.
    pub recovery_threshold: u32,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            check_timeout: Duration::from_secs(5),
            latency_window_size: 100,
            history_size: 1000,
            degradation_latency_ms: 2000,
            degradation_error_rate: 0.1,
            availability_decay: 0.95,
            failure_threshold: 3,
            recovery_threshold: 2,
        }
    }
}

// ─────────────────────────────────────────────
// Health status types
// ─────────────────────────────────────────────

/// Overall health status of an endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointStatus {
    /// Endpoint is healthy and responsive.
    Healthy,
    /// Endpoint is responding but with degraded performance.
    Degraded,
    /// Endpoint is not responding (down).
    Down,
    /// Status is unknown (not yet checked).
    Unknown,
}

/// Result of a single health check probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Whether the check succeeded.
    pub success: bool,
    /// Latency of the check in milliseconds.
    pub latency_ms: u64,
    /// When the check was performed.
    pub checked_at: DateTime<Utc>,
    /// Optional error message on failure.
    pub error: Option<String>,
    /// HTTP status code if applicable.
    pub status_code: Option<u16>,
}

impl HealthCheckResult {
    /// Create a successful health check result.
    pub fn success(latency_ms: u64) -> Self {
        Self {
            success: true,
            latency_ms,
            checked_at: Utc::now(),
            error: None,
            status_code: Some(200),
        }
    }

    /// Create a failed health check result.
    pub fn failure(error: impl Into<String>, latency_ms: u64) -> Self {
        Self {
            success: false,
            latency_ms,
            checked_at: Utc::now(),
            error: Some(error.into()),
            status_code: None,
        }
    }

    /// Create a failed health check with status code.
    pub fn failure_with_status(status_code: u16, latency_ms: u64) -> Self {
        Self {
            success: false,
            latency_ms,
            checked_at: Utc::now(),
            error: Some(format!("HTTP {status_code}")),
            status_code: Some(status_code),
        }
    }
}

/// A snapshot of an endpoint's health at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSnapshot {
    /// The endpoint ID.
    pub endpoint_id: String,
    /// Overall status.
    pub status: EndpointStatus,
    /// Availability score (0.0 - 1.0).
    pub availability_score: f64,
    /// Average latency (ms) over the window.
    pub avg_latency_ms: f64,
    /// P50 latency.
    pub p50_latency_ms: u64,
    /// P95 latency.
    pub p95_latency_ms: u64,
    /// P99 latency.
    pub p99_latency_ms: u64,
    /// Error rate (0.0 - 1.0) over the window.
    pub error_rate: f64,
    /// Total checks performed.
    pub total_checks: u64,
    /// Total successful checks.
    pub successful_checks: u64,
    /// When this snapshot was taken.
    pub snapshot_at: DateTime<Utc>,
    /// Whether degradation is detected.
    pub degraded: bool,
}

/// Degradation event emitted when an endpoint's health changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationEvent {
    /// Endpoint that changed.
    pub endpoint_id: String,
    /// Previous status.
    pub previous_status: EndpointStatus,
    /// New status.
    pub new_status: EndpointStatus,
    /// Reason for the change.
    pub reason: String,
    /// When the change was detected.
    pub detected_at: DateTime<Utc>,
}

// ─────────────────────────────────────────────
// Per-endpoint tracker
// ─────────────────────────────────────────────

/// Internal state for tracking a single endpoint's health.
#[derive(Debug, Clone)]
struct EndpointTracker {
    endpoint_id: String,
    endpoint_url: String,
    status: EndpointStatus,
    latency_samples: VecDeque<u64>,
    check_results: VecDeque<HealthCheckResult>,
    availability_score: f64,
    total_checks: u64,
    successful_checks: u64,
    consecutive_failures: u32,
    consecutive_successes: u32,
    history: VecDeque<HealthSnapshot>,
}

impl EndpointTracker {
    fn new(endpoint_id: impl Into<String>, endpoint_url: impl Into<String>) -> Self {
        Self {
            endpoint_id: endpoint_id.into(),
            endpoint_url: endpoint_url.into(),
            status: EndpointStatus::Unknown,
            latency_samples: VecDeque::new(),
            check_results: VecDeque::new(),
            availability_score: 1.0,
            total_checks: 0,
            successful_checks: 0,
            consecutive_failures: 0,
            consecutive_successes: 0,
            history: VecDeque::new(),
        }
    }
}

// ─────────────────────────────────────────────
// EndpointHealthMonitor
// ─────────────────────────────────────────────

/// Monitors the health of federated SPARQL endpoints.
pub struct EndpointHealthMonitor {
    config: HealthMonitorConfig,
    trackers: HashMap<String, EndpointTracker>,
    degradation_events: VecDeque<DegradationEvent>,
}

impl EndpointHealthMonitor {
    /// Create a new health monitor with default configuration.
    pub fn new() -> Self {
        Self::with_config(HealthMonitorConfig::default())
    }

    /// Create a new health monitor with the given configuration.
    pub fn with_config(config: HealthMonitorConfig) -> Self {
        Self {
            config,
            trackers: HashMap::new(),
            degradation_events: VecDeque::new(),
        }
    }

    /// Register an endpoint for monitoring.
    pub fn register_endpoint(&mut self, endpoint_id: impl Into<String>, url: impl Into<String>) {
        let id = endpoint_id.into();
        let tracker = EndpointTracker::new(id.clone(), url);
        self.trackers.insert(id, tracker);
    }

    /// Remove an endpoint from monitoring.
    pub fn unregister_endpoint(&mut self, endpoint_id: &str) -> bool {
        self.trackers.remove(endpoint_id).is_some()
    }

    /// Get the number of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.trackers.len()
    }

    /// Get the current status of an endpoint.
    pub fn get_status(&self, endpoint_id: &str) -> Option<EndpointStatus> {
        self.trackers.get(endpoint_id).map(|t| t.status)
    }

    /// Get the URL of a registered endpoint.
    pub fn get_url(&self, endpoint_id: &str) -> Option<&str> {
        self.trackers
            .get(endpoint_id)
            .map(|t| t.endpoint_url.as_str())
    }

    /// Get the availability score of an endpoint.
    pub fn get_availability_score(&self, endpoint_id: &str) -> Option<f64> {
        self.trackers.get(endpoint_id).map(|t| t.availability_score)
    }

    /// Record a health check result for an endpoint.
    pub fn record_check(
        &mut self,
        endpoint_id: &str,
        result: HealthCheckResult,
    ) -> Option<DegradationEvent> {
        let config = self.config.clone();

        // First phase: update tracker state (scoped mutable borrow)
        let previous_status = {
            let tracker = self.trackers.get_mut(endpoint_id)?;

            tracker.total_checks += 1;
            let prev = tracker.status;

            // Update latency samples
            if tracker.latency_samples.len() >= config.latency_window_size {
                tracker.latency_samples.pop_front();
            }
            tracker.latency_samples.push_back(result.latency_ms);

            // Update check results (bounded)
            if tracker.check_results.len() >= config.latency_window_size {
                tracker.check_results.pop_front();
            }
            tracker.check_results.push_back(result.clone());

            // Update availability score with exponential decay
            let success_val = if result.success { 1.0 } else { 0.0 };
            tracker.availability_score = config.availability_decay * tracker.availability_score
                + (1.0 - config.availability_decay) * success_val;

            if result.success {
                tracker.successful_checks += 1;
                tracker.consecutive_failures = 0;
                tracker.consecutive_successes += 1;
            } else {
                tracker.consecutive_successes = 0;
                tracker.consecutive_failures += 1;
            }

            prev
        };
        // Mutable borrow from first phase is now dropped

        // Second phase: compute status (needs immutable self)
        let new_status = self.compute_status(endpoint_id);

        // Third phase: apply status and snapshot (scoped mutable borrow)
        if let Some(tracker) = self.trackers.get_mut(endpoint_id) {
            tracker.status = new_status;

            // Store snapshot
            let snapshot = Self::take_snapshot_for(tracker);
            if tracker.history.len() >= config.history_size {
                tracker.history.pop_front();
            }
            tracker.history.push_back(snapshot);
        }

        // Emit degradation event if status changed
        if previous_status != new_status {
            let reason = self.degradation_reason(endpoint_id, previous_status, new_status);
            let event = DegradationEvent {
                endpoint_id: endpoint_id.to_string(),
                previous_status,
                new_status,
                reason,
                detected_at: Utc::now(),
            };
            self.degradation_events.push_back(event.clone());
            Some(event)
        } else {
            None
        }
    }

    /// Get a health snapshot for an endpoint.
    pub fn snapshot(&self, endpoint_id: &str) -> Option<HealthSnapshot> {
        self.trackers.get(endpoint_id).map(Self::take_snapshot_for)
    }

    /// Get all endpoint IDs.
    pub fn endpoint_ids(&self) -> Vec<String> {
        self.trackers.keys().cloned().collect()
    }

    /// Get all endpoints currently marked as healthy.
    pub fn healthy_endpoints(&self) -> Vec<String> {
        self.trackers
            .iter()
            .filter(|(_, t)| t.status == EndpointStatus::Healthy)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get all endpoints currently marked as down.
    pub fn down_endpoints(&self) -> Vec<String> {
        self.trackers
            .iter()
            .filter(|(_, t)| t.status == EndpointStatus::Down)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get all degradation events.
    pub fn drain_degradation_events(&mut self) -> Vec<DegradationEvent> {
        self.degradation_events.drain(..).collect()
    }

    /// Get the history of snapshots for an endpoint.
    pub fn history(&self, endpoint_id: &str) -> Option<&VecDeque<HealthSnapshot>> {
        self.trackers.get(endpoint_id).map(|t| &t.history)
    }

    /// Get the average latency for an endpoint.
    pub fn avg_latency_ms(&self, endpoint_id: &str) -> Option<f64> {
        self.trackers.get(endpoint_id).and_then(|t| {
            if t.latency_samples.is_empty() {
                None
            } else {
                let sum: u64 = t.latency_samples.iter().sum();
                Some(sum as f64 / t.latency_samples.len() as f64)
            }
        })
    }

    /// Get a percentile latency for an endpoint.
    pub fn percentile_latency(&self, endpoint_id: &str, percentile: f64) -> Option<u64> {
        self.trackers.get(endpoint_id).and_then(|t| {
            if t.latency_samples.is_empty() {
                return None;
            }
            let mut sorted: Vec<u64> = t.latency_samples.iter().copied().collect();
            sorted.sort_unstable();
            let idx = ((percentile / 100.0) * (sorted.len() as f64 - 1.0))
                .round()
                .max(0.0) as usize;
            let idx = idx.min(sorted.len() - 1);
            Some(sorted[idx])
        })
    }

    /// Get the error rate for an endpoint.
    pub fn error_rate(&self, endpoint_id: &str) -> Option<f64> {
        self.trackers.get(endpoint_id).and_then(|t| {
            if t.check_results.is_empty() {
                return None;
            }
            let failures = t.check_results.iter().filter(|r| !r.success).count();
            Some(failures as f64 / t.check_results.len() as f64)
        })
    }

    /// Rank endpoints by availability score (highest first).
    pub fn ranked_endpoints(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<_> = self
            .trackers
            .iter()
            .map(|(id, t)| (id.clone(), t.availability_score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    // ─── Internal helpers ────────────────────────────────

    fn compute_status(&self, endpoint_id: &str) -> EndpointStatus {
        let tracker = match self.trackers.get(endpoint_id) {
            Some(t) => t,
            None => return EndpointStatus::Unknown,
        };

        if tracker.total_checks == 0 {
            return EndpointStatus::Unknown;
        }

        // Check for down
        if tracker.consecutive_failures >= self.config.failure_threshold {
            return EndpointStatus::Down;
        }

        // If previously down, require recovery_threshold successes
        if tracker.status == EndpointStatus::Down
            && tracker.consecutive_successes < self.config.recovery_threshold
        {
            return EndpointStatus::Down;
        }

        // If just recovered from Down (met recovery_threshold), go straight to
        // Healthy.  The sliding window still contains the old failures so the
        // error-rate / latency checks below would incorrectly classify the
        // endpoint as Degraded.
        if tracker.status == EndpointStatus::Down
            && tracker.consecutive_successes >= self.config.recovery_threshold
        {
            return EndpointStatus::Healthy;
        }

        // Check for degradation
        let avg = if !tracker.latency_samples.is_empty() {
            let sum: u64 = tracker.latency_samples.iter().sum();
            sum / tracker.latency_samples.len() as u64
        } else {
            0
        };

        let error_rate = if !tracker.check_results.is_empty() {
            let failures = tracker.check_results.iter().filter(|r| !r.success).count();
            failures as f64 / tracker.check_results.len() as f64
        } else {
            0.0
        };

        if avg > self.config.degradation_latency_ms
            || error_rate > self.config.degradation_error_rate
        {
            return EndpointStatus::Degraded;
        }

        EndpointStatus::Healthy
    }

    fn take_snapshot_for(tracker: &EndpointTracker) -> HealthSnapshot {
        let avg_latency_ms = if tracker.latency_samples.is_empty() {
            0.0
        } else {
            let sum: u64 = tracker.latency_samples.iter().sum();
            sum as f64 / tracker.latency_samples.len() as f64
        };

        let (p50, p95, p99) = if tracker.latency_samples.is_empty() {
            (0, 0, 0)
        } else {
            let mut sorted: Vec<u64> = tracker.latency_samples.iter().copied().collect();
            sorted.sort_unstable();
            let len = sorted.len();
            let p50_idx = ((0.5 * (len as f64 - 1.0)).round().max(0.0) as usize).min(len - 1);
            let p95_idx = ((0.95 * (len as f64 - 1.0)).round().max(0.0) as usize).min(len - 1);
            let p99_idx = ((0.99 * (len as f64 - 1.0)).round().max(0.0) as usize).min(len - 1);
            (sorted[p50_idx], sorted[p95_idx], sorted[p99_idx])
        };

        let error_rate = if tracker.check_results.is_empty() {
            0.0
        } else {
            let failures = tracker.check_results.iter().filter(|r| !r.success).count();
            failures as f64 / tracker.check_results.len() as f64
        };

        HealthSnapshot {
            endpoint_id: tracker.endpoint_id.clone(),
            status: tracker.status,
            availability_score: tracker.availability_score,
            avg_latency_ms,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            error_rate,
            total_checks: tracker.total_checks,
            successful_checks: tracker.successful_checks,
            snapshot_at: Utc::now(),
            degraded: tracker.status == EndpointStatus::Degraded,
        }
    }

    fn degradation_reason(
        &self,
        endpoint_id: &str,
        previous: EndpointStatus,
        new: EndpointStatus,
    ) -> String {
        let tracker = match self.trackers.get(endpoint_id) {
            Some(t) => t,
            None => return "endpoint not found".to_string(),
        };

        match (previous, new) {
            (_, EndpointStatus::Down) => {
                format!(
                    "Endpoint went down after {} consecutive failures",
                    tracker.consecutive_failures
                )
            }
            (EndpointStatus::Down, EndpointStatus::Healthy) => {
                format!(
                    "Endpoint recovered after {} consecutive successes",
                    tracker.consecutive_successes
                )
            }
            (_, EndpointStatus::Degraded) => {
                let avg = if !tracker.latency_samples.is_empty() {
                    let sum: u64 = tracker.latency_samples.iter().sum();
                    sum / tracker.latency_samples.len() as u64
                } else {
                    0
                };
                format!(
                    "Degradation detected: avg latency {}ms, availability {:.2}%",
                    avg,
                    tracker.availability_score * 100.0
                )
            }
            (EndpointStatus::Degraded, EndpointStatus::Healthy) => {
                "Degradation resolved, endpoint healthy".to_string()
            }
            _ => format!("Status changed from {previous:?} to {new:?}"),
        }
    }
}

impl Default for EndpointHealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor() -> EndpointHealthMonitor {
        let mut monitor = EndpointHealthMonitor::new();
        monitor.register_endpoint("ep1", "http://example.org/sparql");
        monitor.register_endpoint("ep2", "http://example.com/sparql");
        monitor
    }

    // ═══ Registration tests ══════════════════════════════

    #[test]
    fn test_register_endpoint() {
        let mut monitor = EndpointHealthMonitor::new();
        monitor.register_endpoint("ep1", "http://example.org/sparql");
        assert_eq!(monitor.endpoint_count(), 1);
    }

    #[test]
    fn test_unregister_endpoint() {
        let mut monitor = make_monitor();
        assert!(monitor.unregister_endpoint("ep1"));
        assert_eq!(monitor.endpoint_count(), 1);
        assert!(!monitor.unregister_endpoint("nonexistent"));
    }

    #[test]
    fn test_initial_status_unknown() {
        let monitor = make_monitor();
        assert_eq!(monitor.get_status("ep1"), Some(EndpointStatus::Unknown));
    }

    #[test]
    fn test_get_url() {
        let monitor = make_monitor();
        assert_eq!(monitor.get_url("ep1"), Some("http://example.org/sparql"));
        assert!(monitor.get_url("nonexistent").is_none());
    }

    #[test]
    fn test_endpoint_ids() {
        let monitor = make_monitor();
        let ids = monitor.endpoint_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"ep1".to_string()));
        assert!(ids.contains(&"ep2".to_string()));
    }

    // ═══ Health check recording tests ════════════════════

    #[test]
    fn test_record_success() {
        let mut monitor = make_monitor();
        let event = monitor.record_check("ep1", HealthCheckResult::success(50));
        // Should transition from Unknown to Healthy
        assert!(event.is_some());
        assert_eq!(monitor.get_status("ep1"), Some(EndpointStatus::Healthy));
    }

    #[test]
    fn test_record_failure() {
        let mut monitor = make_monitor();
        // Record enough failures to trigger Down
        for _ in 0..3 {
            monitor.record_check("ep1", HealthCheckResult::failure("timeout", 5000));
        }
        assert_eq!(monitor.get_status("ep1"), Some(EndpointStatus::Down));
    }

    #[test]
    fn test_record_nonexistent_endpoint() {
        let mut monitor = make_monitor();
        let event = monitor.record_check("nonexistent", HealthCheckResult::success(50));
        assert!(event.is_none());
    }

    #[test]
    fn test_availability_score_decay() {
        let mut monitor = make_monitor();
        // All successes
        for _ in 0..10 {
            monitor.record_check("ep1", HealthCheckResult::success(50));
        }
        let score1 = monitor.get_availability_score("ep1").unwrap_or(0.0);
        assert!(score1 > 0.9);

        // Record some failures
        for _ in 0..5 {
            monitor.record_check("ep1", HealthCheckResult::failure("err", 100));
        }
        let score2 = monitor.get_availability_score("ep1").unwrap_or(0.0);
        assert!(score2 < score1);
    }

    // ═══ Latency tracking tests ══════════════════════════

    #[test]
    fn test_avg_latency() {
        let mut monitor = make_monitor();
        monitor.record_check("ep1", HealthCheckResult::success(100));
        monitor.record_check("ep1", HealthCheckResult::success(200));
        monitor.record_check("ep1", HealthCheckResult::success(300));
        let avg = monitor.avg_latency_ms("ep1").unwrap_or(0.0);
        assert!((avg - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_avg_latency_empty() {
        let monitor = make_monitor();
        assert!(monitor.avg_latency_ms("ep1").is_none());
    }

    #[test]
    fn test_percentile_latency() {
        let mut monitor = make_monitor();
        for i in 1..=100 {
            monitor.record_check("ep1", HealthCheckResult::success(i));
        }
        let p50 = monitor.percentile_latency("ep1", 50.0).unwrap_or(0);
        assert!((49..=51).contains(&p50));
        let p99 = monitor.percentile_latency("ep1", 99.0).unwrap_or(0);
        assert!(p99 >= 98);
    }

    #[test]
    fn test_percentile_latency_empty() {
        let monitor = make_monitor();
        assert!(monitor.percentile_latency("ep1", 50.0).is_none());
    }

    // ═══ Error rate tests ════════════════════════════════

    #[test]
    fn test_error_rate_all_success() {
        let mut monitor = make_monitor();
        for _ in 0..10 {
            monitor.record_check("ep1", HealthCheckResult::success(50));
        }
        let rate = monitor.error_rate("ep1").unwrap_or(1.0);
        assert!(rate < 0.01);
    }

    #[test]
    fn test_error_rate_mixed() {
        let mut monitor = make_monitor();
        for _ in 0..8 {
            monitor.record_check("ep1", HealthCheckResult::success(50));
        }
        for _ in 0..2 {
            monitor.record_check("ep1", HealthCheckResult::failure("err", 100));
        }
        let rate = monitor.error_rate("ep1").unwrap_or(0.0);
        assert!((rate - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_error_rate_empty() {
        let monitor = make_monitor();
        assert!(monitor.error_rate("ep1").is_none());
    }

    // ═══ Degradation detection tests ═════════════════════

    #[test]
    fn test_degradation_high_latency() {
        let config = HealthMonitorConfig {
            degradation_latency_ms: 500,
            failure_threshold: 10,
            ..Default::default()
        };
        let mut monitor = EndpointHealthMonitor::with_config(config);
        monitor.register_endpoint("ep1", "http://example.org/sparql");

        for _ in 0..5 {
            monitor.record_check("ep1", HealthCheckResult::success(1000));
        }
        assert_eq!(monitor.get_status("ep1"), Some(EndpointStatus::Degraded));
    }

    #[test]
    fn test_degradation_event_emitted() {
        let mut monitor = make_monitor();
        // Healthy first
        for _ in 0..5 {
            monitor.record_check("ep1", HealthCheckResult::success(50));
        }
        // Fail enough to go down
        for _ in 0..3 {
            monitor.record_check("ep1", HealthCheckResult::failure("err", 100));
        }
        let events = monitor.drain_degradation_events();
        // Should have at least 2 events: Unknown->Healthy, then eventually Healthy->Down
        assert!(!events.is_empty());
    }

    #[test]
    fn test_recovery_from_down() {
        let config = HealthMonitorConfig {
            failure_threshold: 2,
            recovery_threshold: 2,
            ..Default::default()
        };
        let mut monitor = EndpointHealthMonitor::with_config(config);
        monitor.register_endpoint("ep1", "http://example.org/sparql");

        // Go down
        monitor.record_check("ep1", HealthCheckResult::failure("err", 100));
        monitor.record_check("ep1", HealthCheckResult::failure("err", 100));
        assert_eq!(monitor.get_status("ep1"), Some(EndpointStatus::Down));

        // One success isn't enough
        monitor.record_check("ep1", HealthCheckResult::success(50));
        assert_eq!(monitor.get_status("ep1"), Some(EndpointStatus::Down));

        // Two successes should recover
        monitor.record_check("ep1", HealthCheckResult::success(50));
        assert_eq!(monitor.get_status("ep1"), Some(EndpointStatus::Healthy));
    }

    // ═══ Snapshot tests ══════════════════════════════════

    #[test]
    fn test_snapshot() {
        let mut monitor = make_monitor();
        for _ in 0..5 {
            monitor.record_check("ep1", HealthCheckResult::success(100));
        }
        let snap = monitor.snapshot("ep1");
        assert!(snap.is_some());
        let snap = snap.expect("snapshot should exist");
        assert_eq!(snap.endpoint_id, "ep1");
        assert_eq!(snap.total_checks, 5);
        assert_eq!(snap.successful_checks, 5);
    }

    #[test]
    fn test_snapshot_nonexistent() {
        let monitor = make_monitor();
        assert!(monitor.snapshot("nonexistent").is_none());
    }

    // ═══ History tests ═══════════════════════════════════

    #[test]
    fn test_history_bounded() {
        let config = HealthMonitorConfig {
            history_size: 5,
            ..Default::default()
        };
        let mut monitor = EndpointHealthMonitor::with_config(config);
        monitor.register_endpoint("ep1", "http://example.org/sparql");

        for _ in 0..10 {
            monitor.record_check("ep1", HealthCheckResult::success(50));
        }
        let history = monitor.history("ep1");
        assert!(history.is_some());
        assert!(history.expect("history should exist").len() <= 5);
    }

    // ═══ Ranking tests ═══════════════════════════════════

    #[test]
    fn test_ranked_endpoints() {
        let mut monitor = make_monitor();
        // ep1 all success, ep2 mixed
        for _ in 0..10 {
            monitor.record_check("ep1", HealthCheckResult::success(50));
        }
        for _ in 0..5 {
            monitor.record_check("ep2", HealthCheckResult::success(50));
        }
        for _ in 0..5 {
            monitor.record_check("ep2", HealthCheckResult::failure("err", 100));
        }
        let ranked = monitor.ranked_endpoints();
        assert_eq!(ranked.len(), 2);
        // ep1 should rank higher
        assert_eq!(ranked[0].0, "ep1");
    }

    // ═══ Healthy/Down endpoint queries ═══════════════════

    #[test]
    fn test_healthy_endpoints() {
        let mut monitor = make_monitor();
        for _ in 0..5 {
            monitor.record_check("ep1", HealthCheckResult::success(50));
        }
        let healthy = monitor.healthy_endpoints();
        assert!(healthy.contains(&"ep1".to_string()));
    }

    #[test]
    fn test_down_endpoints() {
        let mut monitor = make_monitor();
        for _ in 0..3 {
            monitor.record_check("ep1", HealthCheckResult::failure("err", 100));
        }
        let down = monitor.down_endpoints();
        assert!(down.contains(&"ep1".to_string()));
    }

    // ═══ Config tests ════════════════════════════════════

    #[test]
    fn test_default_config() {
        let config = HealthMonitorConfig::default();
        assert_eq!(config.check_interval, Duration::from_secs(30));
        assert_eq!(config.check_timeout, Duration::from_secs(5));
        assert_eq!(config.latency_window_size, 100);
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.recovery_threshold, 2);
    }

    #[test]
    fn test_custom_config() {
        let config = HealthMonitorConfig {
            check_interval: Duration::from_secs(10),
            failure_threshold: 5,
            ..Default::default()
        };
        assert_eq!(config.check_interval, Duration::from_secs(10));
        assert_eq!(config.failure_threshold, 5);
    }

    // ═══ HealthCheckResult tests ═════════════════════════

    #[test]
    fn test_health_check_success() {
        let result = HealthCheckResult::success(100);
        assert!(result.success);
        assert_eq!(result.latency_ms, 100);
        assert!(result.error.is_none());
        assert_eq!(result.status_code, Some(200));
    }

    #[test]
    fn test_health_check_failure() {
        let result = HealthCheckResult::failure("connection refused", 0);
        assert!(!result.success);
        assert_eq!(result.error, Some("connection refused".to_string()));
    }

    #[test]
    fn test_health_check_failure_with_status() {
        let result = HealthCheckResult::failure_with_status(503, 200);
        assert!(!result.success);
        assert_eq!(result.status_code, Some(503));
        assert_eq!(result.error, Some("HTTP 503".to_string()));
    }

    // ═══ Latency window bounded tests ════════════════════

    #[test]
    fn test_latency_window_bounded() {
        let config = HealthMonitorConfig {
            latency_window_size: 5,
            ..Default::default()
        };
        let mut monitor = EndpointHealthMonitor::with_config(config);
        monitor.register_endpoint("ep1", "http://example.org/sparql");

        for i in 0..20 {
            monitor.record_check("ep1", HealthCheckResult::success(i * 10));
        }
        // Avg should be based on last 5 samples only
        let avg = monitor.avg_latency_ms("ep1").unwrap_or(0.0);
        // Last 5: 150, 160, 170, 180, 190 => avg = 170
        assert!((avg - 170.0).abs() < 1.0);
    }

    // ═══ Default impl test ═══════════════════════════════

    #[test]
    fn test_default_impl() {
        let monitor = EndpointHealthMonitor::default();
        assert_eq!(monitor.endpoint_count(), 0);
    }
}
