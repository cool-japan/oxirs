//! SLA compliance reporting and per-node metric checks
//!
//! Provides:
//! - `SlaViolationRecord` — a structured record of a single detected violation
//! - `SlaReport` — a full compliance report over an observation window
//! - `SlaReporter` — generates reports from accumulated metric samples
//! - `NodeSlaChecker` — checks per-node availability, latency, and throughput

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// NodeId alias
// ---------------------------------------------------------------------------

/// Opaque node identifier type for SLA checks
pub type NodeId = String;

// ---------------------------------------------------------------------------
// Metric check result types
// ---------------------------------------------------------------------------

/// Result of an availability check for a node over a window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AvailabilityResult {
    /// Availability is within the required threshold
    Ok { availability_pct: f64 },
    /// Availability is below the required threshold
    Violation {
        actual: f64,
        threshold: f64,
        message: String,
    },
}

impl AvailabilityResult {
    /// Return `true` if the check passed (no violation)
    pub fn is_ok(&self) -> bool {
        matches!(self, AvailabilityResult::Ok { .. })
    }
}

/// Result of a P99 latency check for a node over a window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyResult {
    /// P99 latency is within the required threshold
    Ok { p99_ms: u64 },
    /// P99 latency exceeds the required threshold
    Violation {
        actual: u64,
        threshold: u64,
        message: String,
    },
}

impl LatencyResult {
    /// Return `true` if the check passed (no violation)
    pub fn is_ok(&self) -> bool {
        matches!(self, LatencyResult::Ok { .. })
    }
}

/// Result of a throughput check for a node over a window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThroughputResult {
    /// Throughput meets or exceeds the minimum target
    Ok { ops_per_sec: f64 },
    /// Throughput is below the minimum target
    Violation {
        actual: f64,
        threshold: f64,
        message: String,
    },
}

impl ThroughputResult {
    /// Return `true` if the check passed (no violation)
    pub fn is_ok(&self) -> bool {
        matches!(self, ThroughputResult::Ok { .. })
    }
}

// ---------------------------------------------------------------------------
// Sample types for node metric recording
// ---------------------------------------------------------------------------

/// A single operation sample recorded for a node
#[derive(Debug, Clone)]
pub struct NodeSample {
    pub timestamp: Instant,
    /// Latency in milliseconds
    pub latency_ms: u64,
    pub success: bool,
}

// ---------------------------------------------------------------------------
// NodeSlaChecker
// ---------------------------------------------------------------------------

/// Proactively checks per-node SLA metrics: availability, P99 latency, throughput.
///
/// Stores a sliding window of `NodeSample`s for each registered node and
/// evaluates metrics on demand.
pub struct NodeSlaChecker {
    /// Per-node sample windows: node_id → deque of samples
    node_samples: HashMap<NodeId, VecDeque<NodeSample>>,
    /// Maximum samples retained per node
    max_samples_per_node: usize,
    /// Minimum availability threshold (0.0–1.0)
    min_availability: f64,
    /// Maximum P99 latency threshold in milliseconds
    max_p99_latency_ms: u64,
    /// Minimum throughput threshold in ops/sec
    min_throughput_ops: f64,
}

impl NodeSlaChecker {
    /// Create a new checker with default thresholds:
    /// - availability ≥ 99.9%
    /// - P99 latency ≤ 500 ms
    /// - throughput ≥ 100 ops/sec
    pub fn new() -> Self {
        Self::with_thresholds(0.999, 500, 100.0)
    }

    /// Create a checker with custom thresholds
    pub fn with_thresholds(
        min_availability: f64,
        max_p99_latency_ms: u64,
        min_throughput_ops: f64,
    ) -> Self {
        Self {
            node_samples: HashMap::new(),
            max_samples_per_node: 10_000,
            min_availability,
            max_p99_latency_ms,
            min_throughput_ops,
        }
    }

    /// Register a node for tracking (idempotent)
    pub fn register_node(&mut self, node_id: impl Into<NodeId>) {
        self.node_samples.entry(node_id.into()).or_default();
    }

    /// Record an operation sample for a node
    pub fn record_sample(&mut self, node_id: &str, latency_ms: u64, success: bool) {
        let samples = self.node_samples.entry(node_id.to_string()).or_default();
        if samples.len() >= self.max_samples_per_node {
            samples.pop_front();
        }
        samples.push_back(NodeSample {
            timestamp: Instant::now(),
            latency_ms,
            success,
        });
    }

    /// Check availability for a node over the last `window_s` seconds.
    ///
    /// Availability = (successful ops / total ops) in the window.
    pub fn check_availability(&self, node_id: &NodeId, window_s: u64) -> AvailabilityResult {
        let samples = match self.node_samples.get(node_id) {
            Some(s) => s,
            None => {
                return AvailabilityResult::Ok {
                    availability_pct: 100.0,
                };
            }
        };

        let cutoff = Instant::now()
            .checked_sub(Duration::from_secs(window_s))
            .unwrap_or_else(Instant::now);

        let window: Vec<&NodeSample> = samples.iter().filter(|s| s.timestamp >= cutoff).collect();

        if window.is_empty() {
            return AvailabilityResult::Ok {
                availability_pct: 100.0,
            };
        }

        let total = window.len() as f64;
        let successes = window.iter().filter(|s| s.success).count() as f64;
        let availability = successes / total;

        if availability >= self.min_availability {
            AvailabilityResult::Ok {
                availability_pct: availability * 100.0,
            }
        } else {
            AvailabilityResult::Violation {
                actual: availability * 100.0,
                threshold: self.min_availability * 100.0,
                message: format!(
                    "Node '{}' availability {:.2}% is below threshold {:.2}%",
                    node_id,
                    availability * 100.0,
                    self.min_availability * 100.0
                ),
            }
        }
    }

    /// Check P99 latency for a node over the last `window_s` seconds.
    pub fn check_latency_p99(&self, node_id: &NodeId, window_s: u64) -> LatencyResult {
        let samples = match self.node_samples.get(node_id) {
            Some(s) => s,
            None => {
                return LatencyResult::Ok { p99_ms: 0 };
            }
        };

        let cutoff = Instant::now()
            .checked_sub(Duration::from_secs(window_s))
            .unwrap_or_else(Instant::now);

        let mut latencies: Vec<u64> = samples
            .iter()
            .filter(|s| s.timestamp >= cutoff)
            .map(|s| s.latency_ms)
            .collect();

        if latencies.is_empty() {
            return LatencyResult::Ok { p99_ms: 0 };
        }

        latencies.sort_unstable();
        let p99_idx = (99 * latencies.len()).saturating_sub(1) / 100;
        let p99_ms = latencies[p99_idx.min(latencies.len() - 1)];

        if p99_ms <= self.max_p99_latency_ms {
            LatencyResult::Ok { p99_ms }
        } else {
            LatencyResult::Violation {
                actual: p99_ms,
                threshold: self.max_p99_latency_ms,
                message: format!(
                    "Node '{}' P99 latency {}ms exceeds threshold {}ms",
                    node_id, p99_ms, self.max_p99_latency_ms
                ),
            }
        }
    }

    /// Check throughput for a node over the last `window_s` seconds.
    ///
    /// Throughput = total operations / window duration in seconds.
    pub fn check_throughput(&self, node_id: &NodeId, window_s: u64) -> ThroughputResult {
        let samples = match self.node_samples.get(node_id) {
            Some(s) => s,
            None => {
                return ThroughputResult::Ok { ops_per_sec: 0.0 };
            }
        };

        let cutoff = Instant::now()
            .checked_sub(Duration::from_secs(window_s))
            .unwrap_or_else(Instant::now);

        let window: Vec<&NodeSample> = samples.iter().filter(|s| s.timestamp >= cutoff).collect();

        if window.is_empty() {
            return ThroughputResult::Ok { ops_per_sec: 0.0 };
        }

        let ops_per_sec = if window_s > 0 {
            window.len() as f64 / window_s as f64
        } else {
            0.0
        };

        if ops_per_sec >= self.min_throughput_ops {
            ThroughputResult::Ok { ops_per_sec }
        } else {
            ThroughputResult::Violation {
                actual: ops_per_sec,
                threshold: self.min_throughput_ops,
                message: format!(
                    "Node '{}' throughput {:.1} ops/sec is below threshold {:.1} ops/sec",
                    node_id, ops_per_sec, self.min_throughput_ops
                ),
            }
        }
    }

    /// Run all checks for a node and return a summary
    pub fn check_all(&self, node_id: &NodeId, window_s: u64) -> NodeHealthSummary {
        let availability = self.check_availability(node_id, window_s);
        let latency = self.check_latency_p99(node_id, window_s);
        let throughput = self.check_throughput(node_id, window_s);
        let healthy = availability.is_ok() && latency.is_ok() && throughput.is_ok();
        NodeHealthSummary {
            node_id: node_id.clone(),
            availability,
            latency,
            throughput,
            is_healthy: healthy,
        }
    }

    /// Number of registered nodes
    pub fn node_count(&self) -> usize {
        self.node_samples.len()
    }

    /// Number of samples for a specific node
    pub fn sample_count(&self, node_id: &str) -> usize {
        self.node_samples.get(node_id).map(|s| s.len()).unwrap_or(0)
    }

    /// Current thresholds
    pub fn thresholds(&self) -> (f64, u64, f64) {
        (
            self.min_availability,
            self.max_p99_latency_ms,
            self.min_throughput_ops,
        )
    }
}

impl Default for NodeSlaChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of all SLA checks for a single node
#[derive(Debug, Clone)]
pub struct NodeHealthSummary {
    pub node_id: NodeId,
    pub availability: AvailabilityResult,
    pub latency: LatencyResult,
    pub throughput: ThroughputResult,
    pub is_healthy: bool,
}

// ---------------------------------------------------------------------------
// SlaViolationRecord
// ---------------------------------------------------------------------------

/// A structured record of a single SLA violation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaViolationRecord {
    /// The node or policy that was violated
    pub target_id: String,
    /// Type of violation
    pub violation_type: ViolationType,
    /// Actual observed metric value
    pub actual_value: f64,
    /// Required threshold
    pub threshold_value: f64,
    /// Human-readable description
    pub message: String,
}

/// Category of SLA violation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    Availability,
    LatencyP99,
    Throughput,
    ErrorRate,
}

impl std::fmt::Display for ViolationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViolationType::Availability => write!(f, "Availability"),
            ViolationType::LatencyP99 => write!(f, "LatencyP99"),
            ViolationType::Throughput => write!(f, "Throughput"),
            ViolationType::ErrorRate => write!(f, "ErrorRate"),
        }
    }
}

// ---------------------------------------------------------------------------
// SlaReport
// ---------------------------------------------------------------------------

/// A complete SLA compliance report over an observation window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaReport {
    /// When the report window started
    #[serde(with = "instant_as_duration")]
    pub period_start: Instant,
    /// Length of the report window
    pub window: Duration,
    /// Overall availability percentage (0.0–100.0)
    pub availability_pct: f64,
    /// P99 latency in milliseconds over the window
    pub p99_latency_ms: u64,
    /// Estimated operations per second
    pub ops_per_sec: f64,
    /// All violations detected during the window
    pub violations: Vec<SlaViolationRecord>,
    /// Number of nodes checked
    pub nodes_checked: usize,
    /// Number of nodes currently healthy
    pub healthy_nodes: usize,
}

// Serde helper for Instant (stored as duration from an epoch)
mod instant_as_duration {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, Instant};

    pub fn serialize<S: Serializer>(instant: &Instant, s: S) -> Result<S::Ok, S::Error> {
        // Serialize as seconds elapsed since the creation point (best effort)
        let d = instant.elapsed();
        d.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Instant, D::Error> {
        let dur = Duration::deserialize(d)?;
        Ok(Instant::now().checked_sub(dur).unwrap_or_else(Instant::now))
    }
}

impl SlaReport {
    /// Return true if there are no violations in the report
    pub fn is_compliant(&self) -> bool {
        self.violations.is_empty()
    }

    /// Count violations of a specific type
    pub fn violation_count(&self, kind: ViolationType) -> usize {
        self.violations
            .iter()
            .filter(|v| v.violation_type == kind)
            .count()
    }
}

// ---------------------------------------------------------------------------
// SlaReporter
// ---------------------------------------------------------------------------

/// Generates SLA compliance reports from a `NodeSlaChecker`.
///
/// Collects metrics for all registered nodes and assembles an `SlaReport`
/// for a given window.
pub struct SlaReporter {
    checker: NodeSlaChecker,
}

impl SlaReporter {
    /// Create a new reporter using an existing `NodeSlaChecker`
    pub fn new(checker: NodeSlaChecker) -> Self {
        Self { checker }
    }

    /// Create a reporter with default thresholds
    pub fn default_reporter() -> Self {
        Self::new(NodeSlaChecker::new())
    }

    /// Access the inner checker (for recording samples)
    pub fn checker_mut(&mut self) -> &mut NodeSlaChecker {
        &mut self.checker
    }

    /// Access the inner checker immutably
    pub fn checker(&self) -> &NodeSlaChecker {
        &self.checker
    }

    /// Generate a compliance report for all registered nodes over `window`.
    ///
    /// The window duration is converted to whole seconds for sample filtering.
    pub fn generate_report(&self, window: Duration) -> SlaReport {
        let window_s = window.as_secs().max(1);
        let period_start = Instant::now()
            .checked_sub(window)
            .unwrap_or_else(Instant::now);

        let mut violations: Vec<SlaViolationRecord> = Vec::new();
        let mut total_availability = 0.0_f64;
        let mut all_latencies: Vec<u64> = Vec::new();
        let mut total_ops: u64 = 0;
        let mut healthy_nodes: usize = 0;
        let nodes_checked = self.checker.node_count();

        for (node_id, samples) in &self.checker.node_samples {
            let cutoff = Instant::now()
                .checked_sub(window)
                .unwrap_or_else(Instant::now);

            let window_samples: Vec<&NodeSample> =
                samples.iter().filter(|s| s.timestamp >= cutoff).collect();

            if window_samples.is_empty() {
                total_availability += 100.0;
                healthy_nodes += 1;
                continue;
            }

            // Collect latencies
            for s in &window_samples {
                all_latencies.push(s.latency_ms);
            }
            total_ops += window_samples.len() as u64;

            // Availability check
            let avail_result = self.checker.check_availability(node_id, window_s);
            match &avail_result {
                AvailabilityResult::Ok { availability_pct } => {
                    total_availability += availability_pct;
                }
                AvailabilityResult::Violation {
                    actual,
                    threshold,
                    message,
                } => {
                    total_availability += actual;
                    violations.push(SlaViolationRecord {
                        target_id: node_id.clone(),
                        violation_type: ViolationType::Availability,
                        actual_value: *actual,
                        threshold_value: *threshold,
                        message: message.clone(),
                    });
                }
            }

            // Latency check
            let lat_result = self.checker.check_latency_p99(node_id, window_s);
            if let LatencyResult::Violation {
                actual,
                threshold,
                message,
            } = &lat_result
            {
                violations.push(SlaViolationRecord {
                    target_id: node_id.clone(),
                    violation_type: ViolationType::LatencyP99,
                    actual_value: *actual as f64,
                    threshold_value: *threshold as f64,
                    message: message.clone(),
                });
            }

            // Throughput check
            let tp_result = self.checker.check_throughput(node_id, window_s);
            if let ThroughputResult::Violation {
                actual,
                threshold,
                message,
            } = &tp_result
            {
                violations.push(SlaViolationRecord {
                    target_id: node_id.clone(),
                    violation_type: ViolationType::Throughput,
                    actual_value: *actual,
                    threshold_value: *threshold,
                    message: message.clone(),
                });
            }

            // Track healthy nodes
            let is_healthy = avail_result.is_ok() && lat_result.is_ok() && tp_result.is_ok();
            if is_healthy {
                healthy_nodes += 1;
            }
        }

        // Aggregate metrics
        let availability_pct = if nodes_checked > 0 {
            total_availability / nodes_checked as f64
        } else {
            100.0
        };

        let p99_latency_ms = if !all_latencies.is_empty() {
            let mut sorted = all_latencies.clone();
            sorted.sort_unstable();
            let idx = (99 * sorted.len()).saturating_sub(1) / 100;
            sorted[idx.min(sorted.len() - 1)]
        } else {
            0
        };

        let ops_per_sec = if window_s > 0 {
            total_ops as f64 / window_s as f64
        } else {
            0.0
        };

        SlaReport {
            period_start,
            window,
            availability_pct,
            p99_latency_ms,
            ops_per_sec,
            violations,
            nodes_checked,
            healthy_nodes,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── NodeSlaChecker — availability ────────────────────────────────────────

    #[test]
    fn test_check_availability_all_success() {
        let mut checker = NodeSlaChecker::with_thresholds(0.99, 500, 10.0);
        checker.register_node("node-1");
        for _ in 0..100 {
            checker.record_sample("node-1", 10, true);
        }
        let result = checker.check_availability(&"node-1".to_string(), 300);
        assert!(result.is_ok(), "100% success should pass 99% threshold");
    }

    #[test]
    fn test_check_availability_violation() {
        let mut checker = NodeSlaChecker::with_thresholds(0.999, 500, 10.0);
        checker.register_node("node-1");
        // 10% error rate → 90% availability
        for i in 0..100 {
            checker.record_sample("node-1", 10, i % 10 != 0);
        }
        let result = checker.check_availability(&"node-1".to_string(), 300);
        assert!(
            !result.is_ok(),
            "90% availability should fail 99.9% threshold"
        );
        if let AvailabilityResult::Violation {
            actual, threshold, ..
        } = &result
        {
            assert!(*actual < *threshold);
        } else {
            panic!("Expected Violation variant");
        }
    }

    #[test]
    fn test_check_availability_no_samples() {
        let checker = NodeSlaChecker::new();
        let result = checker.check_availability(&"nonexistent".to_string(), 60);
        assert!(result.is_ok(), "No samples => assume 100% availability");
    }

    #[test]
    fn test_check_availability_unknown_node() {
        let checker = NodeSlaChecker::new();
        let result = checker.check_availability(&"unknown".to_string(), 60);
        assert!(result.is_ok());
    }

    // ── NodeSlaChecker — latency ─────────────────────────────────────────────

    #[test]
    fn test_check_latency_p99_ok() {
        let mut checker = NodeSlaChecker::with_thresholds(0.99, 200, 10.0);
        checker.register_node("node-1");
        for _ in 0..100 {
            checker.record_sample("node-1", 50, true);
        }
        let result = checker.check_latency_p99(&"node-1".to_string(), 300);
        assert!(result.is_ok());
        if let LatencyResult::Ok { p99_ms } = result {
            assert!(p99_ms <= 50);
        }
    }

    #[test]
    fn test_check_latency_p99_violation() {
        let mut checker = NodeSlaChecker::with_thresholds(0.99, 100, 10.0);
        checker.register_node("node-1");
        // 50 ops at 50ms and 50 ops at 500ms: P99 will be 500ms (> 100ms threshold)
        for _ in 0..50 {
            checker.record_sample("node-1", 50, true);
        }
        for _ in 0..50 {
            checker.record_sample("node-1", 500, true);
        }
        let result = checker.check_latency_p99(&"node-1".to_string(), 300);
        // sorted: [50ms×50, 500ms×50]. P99 idx = (99*100-1)/100 = 98 → 500ms
        assert!(!result.is_ok(), "P99=500ms should violate 100ms threshold");
    }

    #[test]
    fn test_check_latency_no_samples() {
        let checker = NodeSlaChecker::new();
        let result = checker.check_latency_p99(&"nonexistent".to_string(), 60);
        assert!(result.is_ok());
        if let LatencyResult::Ok { p99_ms } = result {
            assert_eq!(p99_ms, 0);
        }
    }

    // ── NodeSlaChecker — throughput ──────────────────────────────────────────

    #[test]
    fn test_check_throughput_ok() {
        let mut checker = NodeSlaChecker::with_thresholds(0.99, 500, 100.0);
        checker.register_node("node-1");
        // Record 600 samples over a 60-second window → 10 ops/sec
        for _ in 0..600 {
            checker.record_sample("node-1", 10, true);
        }
        let result = checker.check_throughput(&"node-1".to_string(), 60);
        // 600 / 60 = 10 ops/sec. Threshold is 100 → violation
        // Let's use a lower threshold to test the ok path
        let mut checker2 = NodeSlaChecker::with_thresholds(0.99, 500, 5.0);
        checker2.register_node("node-2");
        for _ in 0..600 {
            checker2.record_sample("node-2", 10, true);
        }
        let result2 = checker2.check_throughput(&"node-2".to_string(), 60);
        assert!(
            result2.is_ok(),
            "600 samples / 60s = 10 ops/sec >= 5 threshold"
        );
        // Original should be violation
        assert!(!result.is_ok(), "10 ops/sec < 100 threshold");
    }

    #[test]
    fn test_check_throughput_violation() {
        let mut checker = NodeSlaChecker::with_thresholds(0.99, 500, 1000.0);
        checker.register_node("node-1");
        for _ in 0..50 {
            checker.record_sample("node-1", 20, true);
        }
        let result = checker.check_throughput(&"node-1".to_string(), 300);
        assert!(!result.is_ok(), "50/300 = 0.17 ops/sec < 1000 threshold");
    }

    #[test]
    fn test_check_throughput_no_samples() {
        let checker = NodeSlaChecker::new();
        let result = checker.check_throughput(&"nonexistent".to_string(), 60);
        assert!(result.is_ok());
    }

    // ── NodeSlaChecker — check_all ───────────────────────────────────────────

    #[test]
    fn test_check_all_healthy() {
        let mut checker = NodeSlaChecker::with_thresholds(0.99, 500, 5.0);
        checker.register_node("node-1");
        for _ in 0..300 {
            checker.record_sample("node-1", 50, true);
        }
        let summary = checker.check_all(&"node-1".to_string(), 60);
        assert!(summary.is_healthy);
    }

    #[test]
    fn test_check_all_latency_violation() {
        let mut checker = NodeSlaChecker::with_thresholds(0.99, 50, 5.0);
        checker.register_node("n1");
        for _ in 0..100 {
            checker.record_sample("n1", 200, true); // 200ms > 50ms threshold
        }
        let summary = checker.check_all(&"n1".to_string(), 300);
        assert!(!summary.is_healthy);
        assert!(!summary.latency.is_ok());
    }

    #[test]
    fn test_check_all_availability_violation() {
        let mut checker = NodeSlaChecker::with_thresholds(0.999, 500, 5.0);
        checker.register_node("n1");
        for i in 0..100 {
            checker.record_sample("n1", 10, i % 5 != 0); // 80% success
        }
        let summary = checker.check_all(&"n1".to_string(), 300);
        assert!(!summary.is_healthy);
        assert!(!summary.availability.is_ok());
    }

    #[test]
    fn test_checker_sample_count() {
        let mut checker = NodeSlaChecker::new();
        checker.register_node("n1");
        for _ in 0..50 {
            checker.record_sample("n1", 10, true);
        }
        assert_eq!(checker.sample_count("n1"), 50);
        assert_eq!(checker.sample_count("nonexistent"), 0);
    }

    #[test]
    fn test_checker_node_count() {
        let mut checker = NodeSlaChecker::new();
        for i in 0..10 {
            checker.register_node(format!("node-{}", i));
        }
        assert_eq!(checker.node_count(), 10);
    }

    // ── SlaReporter ──────────────────────────────────────────────────────────

    #[test]
    fn test_reporter_empty_report() {
        let reporter = SlaReporter::default_reporter();
        let report = reporter.generate_report(Duration::from_secs(60));
        assert!(report.is_compliant());
        assert_eq!(report.nodes_checked, 0);
        assert_eq!(report.violations.len(), 0);
    }

    #[test]
    fn test_reporter_healthy_nodes() {
        let mut reporter = SlaReporter::new(NodeSlaChecker::with_thresholds(0.99, 500, 5.0));
        for i in 0..5 {
            let node_id = format!("node-{}", i);
            reporter.checker_mut().register_node(&node_id);
            for _ in 0..300 {
                reporter.checker_mut().record_sample(&node_id, 50, true);
            }
        }
        let report = reporter.generate_report(Duration::from_secs(60));
        assert_eq!(report.nodes_checked, 5);
        assert_eq!(report.healthy_nodes, 5);
        assert!(report.is_compliant());
    }

    #[test]
    fn test_reporter_detects_violations() {
        let mut reporter = SlaReporter::new(NodeSlaChecker::with_thresholds(0.999, 100, 5.0));
        reporter.checker_mut().register_node("bad-node");
        for i in 0..100 {
            // High latency + some failures
            let success = i % 5 != 0;
            reporter
                .checker_mut()
                .record_sample("bad-node", 500, success);
        }
        let report = reporter.generate_report(Duration::from_secs(300));
        assert!(!report.is_compliant());
        assert!(!report.violations.is_empty());
    }

    #[test]
    fn test_reporter_availability_pct_aggregation() {
        let mut reporter = SlaReporter::new(NodeSlaChecker::with_thresholds(0.99, 500, 1.0));
        for i in 0..3 {
            let node_id = format!("node-{}", i);
            reporter.checker_mut().register_node(&node_id);
            for _ in 0..100 {
                reporter.checker_mut().record_sample(&node_id, 20, true);
            }
        }
        let report = reporter.generate_report(Duration::from_secs(300));
        assert!(
            report.availability_pct >= 99.0,
            "All healthy nodes => high availability"
        );
    }

    #[test]
    fn test_reporter_p99_latency_in_report() {
        let mut reporter = SlaReporter::new(NodeSlaChecker::with_thresholds(0.99, 1000, 1.0));
        reporter.checker_mut().register_node("node-a");
        for _ in 0..99 {
            reporter.checker_mut().record_sample("node-a", 10, true);
        }
        reporter.checker_mut().record_sample("node-a", 800, true);
        let report = reporter.generate_report(Duration::from_secs(300));
        assert!(report.p99_latency_ms > 0);
    }

    #[test]
    fn test_reporter_ops_per_sec() {
        let mut reporter = SlaReporter::new(NodeSlaChecker::with_thresholds(0.99, 500, 1.0));
        reporter.checker_mut().register_node("node-a");
        for _ in 0..300 {
            reporter.checker_mut().record_sample("node-a", 20, true);
        }
        let report = reporter.generate_report(Duration::from_secs(60));
        assert!(report.ops_per_sec > 0.0);
    }

    #[test]
    fn test_reporter_violation_count_by_type() {
        let mut reporter = SlaReporter::new(NodeSlaChecker::with_thresholds(0.999, 50, 5.0));
        for i in 0..3 {
            let node_id = format!("violating-node-{}", i);
            reporter.checker_mut().register_node(&node_id);
            for j in 0..100 {
                let success = j % 5 != 0; // 80% success → availability violation
                reporter.checker_mut().record_sample(&node_id, 200, success); // 200ms > 50ms threshold
            }
        }
        let report = reporter.generate_report(Duration::from_secs(300));
        assert!(!report.is_compliant());
        let avail_violations = report.violation_count(ViolationType::Availability);
        let latency_violations = report.violation_count(ViolationType::LatencyP99);
        assert!(avail_violations > 0, "Should have availability violations");
        assert!(latency_violations > 0, "Should have latency violations");
    }

    #[test]
    fn test_reporter_mixed_healthy_unhealthy() {
        let mut reporter = SlaReporter::new(NodeSlaChecker::with_thresholds(0.99, 200, 1.0));
        // 2 healthy nodes
        for i in 0..2 {
            let node_id = format!("healthy-{}", i);
            reporter.checker_mut().register_node(&node_id);
            for _ in 0..100 {
                reporter.checker_mut().record_sample(&node_id, 50, true);
            }
        }
        // 1 unhealthy node
        reporter.checker_mut().register_node("sick");
        for _ in 0..100 {
            reporter.checker_mut().record_sample("sick", 500, false); // all failures + high latency
        }
        let report = reporter.generate_report(Duration::from_secs(300));
        assert_eq!(report.nodes_checked, 3);
        assert!(report.healthy_nodes < report.nodes_checked);
        assert!(!report.is_compliant());
    }
}
