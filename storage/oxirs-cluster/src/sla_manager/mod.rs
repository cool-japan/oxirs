//! SLA-based resource management for OxiRS cluster.
//!
//! This module provides Service Level Agreement (SLA) enforcement with:
//! - SLO target definitions (latency P99, throughput, availability)
//! - Per-tenant/per-operation SLA compliance tracking
//! - Real-time SLA violation detection and alerting
//! - Priority-aware resource budget allocation
//! - Per-node availability/latency/throughput checks (`reporter` sub-module)
//! - SLA compliance reports (`SlaReporter`, `SlaReport`)
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_cluster::sla_manager::{SlaPolicy, SlaTracker, SlaViolationDetector, ResourceBudgetManager};
//! use std::time::Duration;
//!
//! let policy = SlaPolicy::new("tenant-a", SlaClass::Gold)
//!     .with_p99_latency(Duration::from_millis(100))
//!     .with_throughput_rps(5000.0)
//!     .with_availability(0.9999);
//! ```

pub mod reporter;
pub use reporter::{
    AvailabilityResult, LatencyResult, NodeHealthSummary, NodeSlaChecker, SlaReport, SlaReporter,
    SlaViolationRecord, ThroughputResult, ViolationType,
};

use crate::error::{ClusterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

// ─────────────────────────────────────────────
//  SLA classification
// ─────────────────────────────────────────────

/// Service-level classification for tenants and workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SlaClass {
    /// Best-effort tier: no hard guarantees.
    Bronze = 0,
    /// Standard tier: moderate latency and availability targets.
    Silver = 1,
    /// Premium tier: tight latency and high availability targets.
    Gold = 2,
    /// Mission-critical tier: strictest targets with pre-emptive resources.
    Platinum = 3,
}

impl SlaClass {
    /// Returns the numeric priority weight (higher = more resources).
    pub fn priority_weight(self) -> u32 {
        match self {
            SlaClass::Bronze => 1,
            SlaClass::Silver => 2,
            SlaClass::Gold => 4,
            SlaClass::Platinum => 8,
        }
    }

    /// Default SLO targets for this class.
    pub fn default_targets(self) -> SloTargets {
        match self {
            SlaClass::Bronze => SloTargets {
                p99_latency: Duration::from_secs(5),
                p95_latency: Duration::from_secs(3),
                p50_latency: Duration::from_millis(500),
                min_throughput_rps: 100.0,
                min_availability: 0.99,
                max_error_rate: 0.05,
            },
            SlaClass::Silver => SloTargets {
                p99_latency: Duration::from_secs(1),
                p95_latency: Duration::from_millis(500),
                p50_latency: Duration::from_millis(100),
                min_throughput_rps: 1_000.0,
                min_availability: 0.999,
                max_error_rate: 0.01,
            },
            SlaClass::Gold => SloTargets {
                p99_latency: Duration::from_millis(200),
                p95_latency: Duration::from_millis(100),
                p50_latency: Duration::from_millis(20),
                min_throughput_rps: 5_000.0,
                min_availability: 0.9999,
                max_error_rate: 0.001,
            },
            SlaClass::Platinum => SloTargets {
                p99_latency: Duration::from_millis(50),
                p95_latency: Duration::from_millis(25),
                p50_latency: Duration::from_millis(5),
                min_throughput_rps: 20_000.0,
                min_availability: 0.99999,
                max_error_rate: 0.0001,
            },
        }
    }
}

impl std::fmt::Display for SlaClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            SlaClass::Bronze => "Bronze",
            SlaClass::Silver => "Silver",
            SlaClass::Gold => "Gold",
            SlaClass::Platinum => "Platinum",
        };
        write!(f, "{}", s)
    }
}

// ─────────────────────────────────────────────
//  SLO Targets
// ─────────────────────────────────────────────

/// Quantitative SLO targets that form the basis of an SLA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloTargets {
    /// P99 end-to-end latency target.
    pub p99_latency: Duration,
    /// P95 end-to-end latency target.
    pub p95_latency: Duration,
    /// P50 (median) end-to-end latency target.
    pub p50_latency: Duration,
    /// Minimum sustained throughput in requests per second.
    pub min_throughput_rps: f64,
    /// Minimum availability (fraction of time the service is operational, 0.0–1.0).
    pub min_availability: f64,
    /// Maximum acceptable error rate (fraction of requests that may fail, 0.0–1.0).
    pub max_error_rate: f64,
}

impl SloTargets {
    /// Validates that all targets are internally consistent and within sane ranges.
    pub fn validate(&self) -> Result<()> {
        if self.p50_latency > self.p95_latency {
            return Err(ClusterError::Config(
                "p50_latency must be <= p95_latency".into(),
            ));
        }
        if self.p95_latency > self.p99_latency {
            return Err(ClusterError::Config(
                "p95_latency must be <= p99_latency".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.min_availability) {
            return Err(ClusterError::Config(
                "min_availability must be in [0.0, 1.0]".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.max_error_rate) {
            return Err(ClusterError::Config(
                "max_error_rate must be in [0.0, 1.0]".into(),
            ));
        }
        if self.min_throughput_rps < 0.0 {
            return Err(ClusterError::Config(
                "min_throughput_rps must be >= 0.0".into(),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────
//  SlaPolicy
// ─────────────────────────────────────────────

/// A complete SLA policy binding a tenant/operation to SLO targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaPolicy {
    /// Logical name identifying the tenant or workload.
    pub name: String,
    /// Classification tier driving default targets and resource weighting.
    pub class: SlaClass,
    /// The SLO targets for this policy.
    pub targets: SloTargets,
    /// Observation window for compliance calculations.
    pub window: Duration,
    /// Whether violations should trigger external alerts.
    pub alert_on_violation: bool,
}

impl SlaPolicy {
    /// Creates a new SLA policy with default targets for the given class.
    pub fn new(name: impl Into<String>, class: SlaClass) -> Self {
        Self {
            name: name.into(),
            targets: class.default_targets(),
            class,
            window: Duration::from_secs(60),
            alert_on_violation: true,
        }
    }

    /// Overrides the P99 latency target.
    pub fn with_p99_latency(mut self, latency: Duration) -> Self {
        self.targets.p99_latency = latency;
        self
    }

    /// Overrides the minimum throughput target.
    pub fn with_throughput_rps(mut self, rps: f64) -> Self {
        self.targets.min_throughput_rps = rps;
        self
    }

    /// Overrides the minimum availability target.
    pub fn with_availability(mut self, availability: f64) -> Self {
        self.targets.min_availability = availability;
        self
    }

    /// Sets the compliance observation window.
    pub fn with_window(mut self, window: Duration) -> Self {
        self.window = window;
        self
    }

    /// Validates all SLO targets.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(ClusterError::Config(
                "SLA policy name cannot be empty".into(),
            ));
        }
        self.targets.validate()
    }
}

// ─────────────────────────────────────────────
//  Observation record
// ─────────────────────────────────────────────

/// A single request observation used for compliance tracking.
#[derive(Debug, Clone)]
pub struct RequestObservation {
    /// Wall-clock time the observation was recorded.
    pub timestamp: Instant,
    /// End-to-end latency of the request.
    pub latency: Duration,
    /// Whether the request succeeded.
    pub success: bool,
}

impl RequestObservation {
    /// Creates a new observation for a successful request.
    pub fn success(latency: Duration) -> Self {
        Self {
            timestamp: Instant::now(),
            latency,
            success: true,
        }
    }

    /// Creates a new observation for a failed request.
    pub fn failure(latency: Duration) -> Self {
        Self {
            timestamp: Instant::now(),
            latency,
            success: false,
        }
    }
}

// ─────────────────────────────────────────────
//  SlaTracker
// ─────────────────────────────────────────────

/// Tracks SLA compliance metrics for a single tenant/operation pair.
///
/// Observations are stored in a sliding time window; metrics are calculated
/// on-demand from the retained observations.
pub struct SlaTracker {
    policy: SlaPolicy,
    observations: RwLock<VecDeque<RequestObservation>>,
    total_requests: AtomicU64,
    total_errors: AtomicU64,
}

impl SlaTracker {
    /// Creates a new tracker bound to the given policy.
    pub fn new(policy: SlaPolicy) -> Result<Self> {
        policy.validate()?;
        Ok(Self {
            policy,
            observations: RwLock::new(VecDeque::new()),
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
        })
    }

    /// Records a new request observation, evicting stale entries outside the window.
    pub async fn record(&self, obs: RequestObservation) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        if !obs.success {
            self.total_errors.fetch_add(1, Ordering::Relaxed);
        }

        let cutoff = Instant::now()
            .checked_sub(self.policy.window)
            .unwrap_or(Instant::now());

        let mut deque = self.observations.write().await;
        // Evict observations outside the sliding window.
        while let Some(front) = deque.front() {
            if front.timestamp < cutoff {
                deque.pop_front();
            } else {
                break;
            }
        }
        deque.push_back(obs);
    }

    /// Computes the current compliance snapshot for the policy's SLO targets.
    pub async fn compliance_snapshot(&self) -> ComplianceSnapshot {
        let observations = self.observations.read().await;
        let window_len = observations.len();

        if window_len == 0 {
            return ComplianceSnapshot {
                policy_name: self.policy.name.clone(),
                class: self.policy.class,
                window_requests: 0,
                p50_latency: Duration::ZERO,
                p95_latency: Duration::ZERO,
                p99_latency: Duration::ZERO,
                actual_error_rate: 0.0,
                estimated_rps: 0.0,
                meets_p99_latency: true,
                meets_p95_latency: true,
                meets_throughput: true,
                meets_error_rate: true,
            };
        }

        // Collect latencies in nanoseconds for percentile calculation.
        let mut latencies_ns: Vec<u64> = observations
            .iter()
            .map(|o| o.latency.as_nanos() as u64)
            .collect();
        latencies_ns.sort_unstable();

        let error_count = observations.iter().filter(|o| !o.success).count();
        let error_rate = error_count as f64 / window_len as f64;

        let p50 = percentile_ns(&latencies_ns, 50);
        let p95 = percentile_ns(&latencies_ns, 95);
        let p99 = percentile_ns(&latencies_ns, 99);

        // Estimate RPS from the oldest/newest timestamps within the window.
        let rps = if window_len >= 2 {
            let oldest = observations
                .front()
                .map(|o| o.timestamp)
                .unwrap_or_else(Instant::now);
            let elapsed = oldest.elapsed();
            if elapsed.as_secs_f64() > 0.0 {
                window_len as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            }
        } else {
            0.0
        };

        let targets = &self.policy.targets;
        ComplianceSnapshot {
            policy_name: self.policy.name.clone(),
            class: self.policy.class,
            window_requests: window_len as u64,
            p50_latency: Duration::from_nanos(p50),
            p95_latency: Duration::from_nanos(p95),
            p99_latency: Duration::from_nanos(p99),
            actual_error_rate: error_rate,
            estimated_rps: rps,
            meets_p99_latency: Duration::from_nanos(p99) <= targets.p99_latency,
            meets_p95_latency: Duration::from_nanos(p95) <= targets.p95_latency,
            meets_throughput: rps >= targets.min_throughput_rps || window_len < 10,
            meets_error_rate: error_rate <= targets.max_error_rate,
        }
    }

    /// Returns the underlying SLA policy.
    pub fn policy(&self) -> &SlaPolicy {
        &self.policy
    }

    /// Total requests observed since tracker creation.
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Total error requests observed since tracker creation.
    pub fn total_errors(&self) -> u64 {
        self.total_errors.load(Ordering::Relaxed)
    }
}

/// Calculates the Nth percentile value from a sorted slice of nanosecond durations.
fn percentile_ns(sorted: &[u64], pct: u8) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((pct as usize * sorted.len()).saturating_sub(1)) / 100;
    sorted[idx.min(sorted.len() - 1)]
}

/// A point-in-time compliance snapshot for a single SLA policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceSnapshot {
    /// The policy name.
    pub policy_name: String,
    /// SLA class.
    pub class: SlaClass,
    /// Number of requests in the current window.
    pub window_requests: u64,
    /// Observed P50 latency.
    pub p50_latency: Duration,
    /// Observed P95 latency.
    pub p95_latency: Duration,
    /// Observed P99 latency.
    pub p99_latency: Duration,
    /// Observed error rate (0.0–1.0).
    pub actual_error_rate: f64,
    /// Estimated throughput (requests per second).
    pub estimated_rps: f64,
    /// Whether the P99 latency SLO is currently being met.
    pub meets_p99_latency: bool,
    /// Whether the P95 latency SLO is currently being met.
    pub meets_p95_latency: bool,
    /// Whether the throughput SLO is currently being met.
    pub meets_throughput: bool,
    /// Whether the error rate SLO is currently being met.
    pub meets_error_rate: bool,
}

impl ComplianceSnapshot {
    /// Returns true if all SLO targets are currently being met.
    pub fn is_compliant(&self) -> bool {
        self.meets_p99_latency
            && self.meets_p95_latency
            && self.meets_throughput
            && self.meets_error_rate
    }

    /// Returns a list of all active violations.
    pub fn violations(&self) -> Vec<SlaViolation> {
        let mut v = Vec::new();
        if !self.meets_p99_latency {
            v.push(SlaViolation::LatencyExceeded {
                percentile: 99,
                actual: self.p99_latency,
            });
        }
        if !self.meets_p95_latency {
            v.push(SlaViolation::LatencyExceeded {
                percentile: 95,
                actual: self.p95_latency,
            });
        }
        if !self.meets_throughput {
            v.push(SlaViolation::ThroughputBelowMinimum {
                actual_rps: self.estimated_rps,
            });
        }
        if !self.meets_error_rate {
            v.push(SlaViolation::ErrorRateExceeded {
                actual_rate: self.actual_error_rate,
            });
        }
        v
    }
}

// ─────────────────────────────────────────────
//  SLA Violations
// ─────────────────────────────────────────────

/// Describes a specific SLA violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlaViolation {
    /// A latency percentile exceeded its target.
    LatencyExceeded {
        /// Which percentile (50, 95, or 99).
        percentile: u8,
        /// The actual observed latency.
        actual: Duration,
    },
    /// Throughput dropped below the minimum target.
    ThroughputBelowMinimum {
        /// The actual estimated RPS.
        actual_rps: f64,
    },
    /// The error rate exceeded the maximum target.
    ErrorRateExceeded {
        /// The actual error rate (0.0–1.0).
        actual_rate: f64,
    },
    /// Availability dropped below the minimum target.
    AvailabilityBelowMinimum {
        /// The actual availability fraction (0.0–1.0).
        actual: f64,
    },
}

impl std::fmt::Display for SlaViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlaViolation::LatencyExceeded { percentile, actual } => {
                write!(
                    f,
                    "P{} latency {:.2}ms exceeded SLO",
                    percentile,
                    actual.as_secs_f64() * 1000.0
                )
            }
            SlaViolation::ThroughputBelowMinimum { actual_rps } => {
                write!(f, "Throughput {:.1} RPS below minimum SLO", actual_rps)
            }
            SlaViolation::ErrorRateExceeded { actual_rate } => {
                write!(f, "Error rate {:.4} exceeded maximum SLO", actual_rate)
            }
            SlaViolation::AvailabilityBelowMinimum { actual } => {
                write!(f, "Availability {:.6} below minimum SLO", actual)
            }
        }
    }
}

/// A complete violation event with context.
#[derive(Debug, Clone)]
pub struct ViolationEvent {
    /// Policy that was violated.
    pub policy_name: String,
    /// SLA class.
    pub class: SlaClass,
    /// The specific violation.
    pub violation: SlaViolation,
    /// When the violation was detected.
    pub detected_at: std::time::SystemTime,
    /// Severity (derived from class).
    pub severity: ViolationSeverity,
}

/// Severity level for SLA violation events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Informational – soft target missed.
    Info,
    /// Warning – SLO breach that should be addressed.
    Warning,
    /// Critical – hard SLO breach affecting Platinum/Gold tenants.
    Critical,
}

impl From<SlaClass> for ViolationSeverity {
    fn from(class: SlaClass) -> Self {
        match class {
            SlaClass::Bronze | SlaClass::Silver => ViolationSeverity::Warning,
            SlaClass::Gold => ViolationSeverity::Warning,
            SlaClass::Platinum => ViolationSeverity::Critical,
        }
    }
}

// ─────────────────────────────────────────────
//  SlaViolationDetector
// ─────────────────────────────────────────────

/// Detects and publishes SLA violation events across all registered trackers.
pub struct SlaViolationDetector {
    trackers: Arc<RwLock<HashMap<String, Arc<SlaTracker>>>>,
    violation_log: Arc<RwLock<VecDeque<ViolationEvent>>>,
    max_log_size: usize,
}

impl SlaViolationDetector {
    /// Creates a new detector with the specified violation log capacity.
    pub fn new(max_log_size: usize) -> Self {
        Self {
            trackers: Arc::new(RwLock::new(HashMap::new())),
            violation_log: Arc::new(RwLock::new(VecDeque::new())),
            max_log_size,
        }
    }

    /// Registers an SLA tracker with the detector.
    pub async fn register(&self, tracker: Arc<SlaTracker>) {
        let mut map = self.trackers.write().await;
        map.insert(tracker.policy().name.clone(), tracker);
    }

    /// Removes a tracker by policy name.
    pub async fn deregister(&self, policy_name: &str) {
        let mut map = self.trackers.write().await;
        map.remove(policy_name);
    }

    /// Scans all trackers for violations and appends events to the violation log.
    ///
    /// Returns the number of new violation events detected.
    pub async fn scan(&self) -> Result<usize> {
        let trackers = self.trackers.read().await;
        let mut new_violations = 0_usize;

        for tracker in trackers.values() {
            let snapshot = tracker.compliance_snapshot().await;
            if snapshot.is_compliant() {
                continue;
            }

            let violations = snapshot.violations();
            for v in violations {
                let severity = ViolationSeverity::from(snapshot.class);
                match severity {
                    ViolationSeverity::Critical => {
                        error!(
                            policy = %snapshot.policy_name,
                            violation = %v,
                            "Critical SLA violation detected"
                        );
                    }
                    ViolationSeverity::Warning => {
                        warn!(
                            policy = %snapshot.policy_name,
                            violation = %v,
                            "SLA violation detected"
                        );
                    }
                    ViolationSeverity::Info => {
                        info!(
                            policy = %snapshot.policy_name,
                            violation = %v,
                            "SLA soft target missed"
                        );
                    }
                }

                let event = ViolationEvent {
                    policy_name: snapshot.policy_name.clone(),
                    class: snapshot.class,
                    violation: v,
                    detected_at: std::time::SystemTime::now(),
                    severity,
                };

                let mut log = self.violation_log.write().await;
                if log.len() >= self.max_log_size {
                    log.pop_front();
                }
                log.push_back(event);
                new_violations += 1;
            }
        }

        Ok(new_violations)
    }

    /// Returns recent violation events (newest first), up to `limit`.
    pub async fn recent_violations(&self, limit: usize) -> Vec<ViolationEvent> {
        let log = self.violation_log.read().await;
        log.iter().rev().take(limit).cloned().collect()
    }

    /// Returns the total number of violations logged.
    pub async fn total_violations(&self) -> usize {
        self.violation_log.read().await.len()
    }
}

// ─────────────────────────────────────────────
//  ResourceBudgetManager
// ─────────────────────────────────────────────

/// Allocation slot for a single resource dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// CPU share (fraction of total cluster CPUs, 0.0–1.0).
    pub cpu_share: f64,
    /// Memory share in MiB.
    pub memory_mib: u64,
    /// Network bandwidth share in Mbps.
    pub network_mbps: f64,
    /// Maximum concurrent I/O operations.
    pub max_iops: u32,
}

/// Manages resource budget allocation across tenants based on SLA priority.
///
/// Uses weighted fair queuing: each tenant's share is proportional to their
/// `SlaClass::priority_weight()`.  The total across all tenants is normalised
/// against the cluster-wide resource pool.
pub struct ResourceBudgetManager {
    /// Total available CPU cores in the cluster.
    total_cpus: f64,
    /// Total available memory in MiB.
    total_memory_mib: u64,
    /// Total available network bandwidth in Mbps.
    total_network_mbps: f64,
    /// Total available IOPS across the cluster.
    total_iops: u32,
    /// Registered policies, keyed by tenant/policy name.
    policies: Arc<RwLock<HashMap<String, SlaPolicy>>>,
    /// Computed allocations, refreshed on demand.
    allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
}

impl ResourceBudgetManager {
    /// Creates a new budget manager with the specified cluster resource pool.
    pub fn new(
        total_cpus: f64,
        total_memory_mib: u64,
        total_network_mbps: f64,
        total_iops: u32,
    ) -> Self {
        Self {
            total_cpus,
            total_memory_mib,
            total_network_mbps,
            total_iops,
            policies: Arc::new(RwLock::new(HashMap::new())),
            allocations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Registers (or updates) an SLA policy and recomputes allocations.
    pub async fn register_policy(&self, policy: SlaPolicy) -> Result<()> {
        policy.validate()?;
        {
            let mut map = self.policies.write().await;
            map.insert(policy.name.clone(), policy);
        }
        self.recompute_allocations().await
    }

    /// Removes a policy by name and recomputes allocations.
    pub async fn remove_policy(&self, name: &str) -> Result<()> {
        {
            let mut map = self.policies.write().await;
            map.remove(name);
        }
        self.recompute_allocations().await
    }

    /// Returns the current resource allocation for the named tenant/policy.
    pub async fn allocation_for(&self, name: &str) -> Option<ResourceAllocation> {
        let allocs = self.allocations.read().await;
        allocs.get(name).cloned()
    }

    /// Recomputes allocations using weighted fair queuing over registered policies.
    async fn recompute_allocations(&self) -> Result<()> {
        let policies = self.policies.read().await;
        if policies.is_empty() {
            let mut allocs = self.allocations.write().await;
            allocs.clear();
            return Ok(());
        }

        let total_weight: u32 = policies.values().map(|p| p.class.priority_weight()).sum();

        if total_weight == 0 {
            return Err(ClusterError::Config(
                "Total SLA priority weight must be > 0".into(),
            ));
        }

        let mut new_allocs = HashMap::with_capacity(policies.len());
        for (name, policy) in policies.iter() {
            let w = policy.class.priority_weight() as f64 / total_weight as f64;
            new_allocs.insert(
                name.clone(),
                ResourceAllocation {
                    cpu_share: self.total_cpus * w,
                    memory_mib: (self.total_memory_mib as f64 * w) as u64,
                    network_mbps: self.total_network_mbps * w,
                    max_iops: (self.total_iops as f64 * w) as u32,
                },
            );
        }

        let mut allocs = self.allocations.write().await;
        *allocs = new_allocs;
        Ok(())
    }

    /// Returns a snapshot of all current allocations.
    pub async fn all_allocations(&self) -> HashMap<String, ResourceAllocation> {
        self.allocations.read().await.clone()
    }

    /// Returns the number of registered policies.
    pub async fn policy_count(&self) -> usize {
        self.policies.read().await.len()
    }
}

// ─────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn gold_policy() -> SlaPolicy {
        SlaPolicy::new("tenant-gold", SlaClass::Gold)
    }

    fn silver_policy() -> SlaPolicy {
        SlaPolicy::new("tenant-silver", SlaClass::Silver)
    }

    // ── SlaPolicy ────────────────────────────

    #[test]
    fn test_sla_policy_default_targets_sensible() {
        let p = gold_policy();
        assert!(p.targets.p50_latency <= p.targets.p95_latency);
        assert!(p.targets.p95_latency <= p.targets.p99_latency);
        assert!(p.targets.min_availability > 0.0);
    }

    #[test]
    fn test_sla_policy_validate_ok() {
        let p = gold_policy();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_sla_policy_validate_empty_name() {
        let p = SlaPolicy::new("", SlaClass::Gold);
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_sla_policy_builder_overrides() {
        let p = gold_policy()
            .with_p99_latency(Duration::from_millis(300))
            .with_throughput_rps(10_000.0)
            .with_availability(0.9995);
        assert_eq!(p.targets.p99_latency, Duration::from_millis(300));
        assert!((p.targets.min_throughput_rps - 10_000.0).abs() < f64::EPSILON);
        assert!((p.targets.min_availability - 0.9995).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sla_class_priority_order() {
        assert!(SlaClass::Bronze.priority_weight() < SlaClass::Silver.priority_weight());
        assert!(SlaClass::Silver.priority_weight() < SlaClass::Gold.priority_weight());
        assert!(SlaClass::Gold.priority_weight() < SlaClass::Platinum.priority_weight());
    }

    // ── SlaTracker ───────────────────────────

    #[tokio::test]
    async fn test_sla_tracker_empty_compliance() {
        let tracker = SlaTracker::new(gold_policy()).expect("tracker creation");
        let snap = tracker.compliance_snapshot().await;
        assert!(snap.is_compliant(), "empty tracker must be compliant");
    }

    #[tokio::test]
    async fn test_sla_tracker_records_success() {
        let tracker = SlaTracker::new(gold_policy()).expect("tracker creation");
        for _ in 0..10 {
            tracker
                .record(RequestObservation::success(Duration::from_millis(10)))
                .await;
        }
        assert_eq!(tracker.total_requests(), 10);
        assert_eq!(tracker.total_errors(), 0);
    }

    #[tokio::test]
    async fn test_sla_tracker_records_errors() {
        let tracker = SlaTracker::new(silver_policy()).expect("tracker creation");
        for _ in 0..5 {
            tracker
                .record(RequestObservation::failure(Duration::from_millis(50)))
                .await;
        }
        assert_eq!(tracker.total_errors(), 5);
    }

    #[tokio::test]
    async fn test_sla_tracker_latency_violation() {
        // Gold P99 target is 200ms; inject all observations at 500ms.
        let policy = gold_policy().with_window(Duration::from_secs(300));
        let tracker = SlaTracker::new(policy).expect("tracker creation");
        for _ in 0..20 {
            tracker
                .record(RequestObservation::success(Duration::from_millis(500)))
                .await;
        }
        let snap = tracker.compliance_snapshot().await;
        assert!(!snap.meets_p99_latency, "P99 latency should be violated");
        assert!(!snap.violations().is_empty());
    }

    #[tokio::test]
    async fn test_sla_tracker_error_rate_violation() {
        let policy = gold_policy().with_window(Duration::from_secs(300));
        let tracker = SlaTracker::new(policy).expect("tracker");
        // Gold max error rate = 0.001; inject 10% error rate.
        for i in 0..100 {
            if i % 10 == 0 {
                tracker
                    .record(RequestObservation::failure(Duration::from_millis(5)))
                    .await;
            } else {
                tracker
                    .record(RequestObservation::success(Duration::from_millis(5)))
                    .await;
            }
        }
        let snap = tracker.compliance_snapshot().await;
        assert!(!snap.meets_error_rate, "Error rate should be violated");
    }

    // ── SlaViolationDetector ─────────────────

    #[tokio::test]
    async fn test_violation_detector_no_violations() {
        let detector = SlaViolationDetector::new(100);
        let tracker = Arc::new(SlaTracker::new(gold_policy()).expect("tracker"));
        // Record good observations.
        for _ in 0..10 {
            tracker
                .record(RequestObservation::success(Duration::from_millis(10)))
                .await;
        }
        detector.register(tracker).await;
        let count = detector.scan().await.expect("scan");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_violation_detector_detects_violation() {
        let detector = SlaViolationDetector::new(100);
        let policy = gold_policy().with_window(Duration::from_secs(300));
        let tracker = Arc::new(SlaTracker::new(policy).expect("tracker"));
        // Inject latency violations.
        for _ in 0..20 {
            tracker
                .record(RequestObservation::success(Duration::from_millis(1000)))
                .await;
        }
        detector.register(tracker).await;
        let count = detector.scan().await.expect("scan");
        assert!(count > 0, "should detect at least one violation");
    }

    #[tokio::test]
    async fn test_violation_detector_log_recent() {
        let detector = SlaViolationDetector::new(50);
        let policy = gold_policy().with_window(Duration::from_secs(300));
        let tracker = Arc::new(SlaTracker::new(policy).expect("tracker"));
        for _ in 0..20 {
            tracker
                .record(RequestObservation::success(Duration::from_millis(1000)))
                .await;
        }
        detector.register(tracker).await;
        detector.scan().await.expect("scan");
        let recent = detector.recent_violations(10).await;
        assert!(!recent.is_empty());
    }

    // ── ResourceBudgetManager ────────────────

    #[tokio::test]
    async fn test_budget_manager_single_policy() {
        let mgr = ResourceBudgetManager::new(16.0, 65_536, 10_000.0, 100_000);
        mgr.register_policy(gold_policy()).await.expect("register");
        let alloc = mgr.allocation_for("tenant-gold").await.expect("alloc");
        // Only one policy; should receive all resources.
        assert!((alloc.cpu_share - 16.0).abs() < 0.001);
        assert_eq!(alloc.memory_mib, 65_536);
    }

    #[tokio::test]
    async fn test_budget_manager_two_policies_weighted() {
        let mgr = ResourceBudgetManager::new(100.0, 100_000, 10_000.0, 200_000);
        mgr.register_policy(SlaPolicy::new("bronze", SlaClass::Bronze))
            .await
            .expect("reg");
        mgr.register_policy(SlaPolicy::new("platinum", SlaClass::Platinum))
            .await
            .expect("reg");

        let bronze_alloc = mgr.allocation_for("bronze").await.expect("bronze alloc");
        let plat_alloc = mgr.allocation_for("platinum").await.expect("plat alloc");

        // Platinum (weight 8) should have more resources than Bronze (weight 1).
        assert!(plat_alloc.cpu_share > bronze_alloc.cpu_share);
        assert!(plat_alloc.memory_mib > bronze_alloc.memory_mib);
    }

    #[tokio::test]
    async fn test_budget_manager_remove_policy() {
        let mgr = ResourceBudgetManager::new(16.0, 32_768, 1_000.0, 50_000);
        mgr.register_policy(gold_policy()).await.expect("reg gold");
        mgr.register_policy(silver_policy())
            .await
            .expect("reg silver");
        assert_eq!(mgr.policy_count().await, 2);

        mgr.remove_policy("tenant-silver").await.expect("remove");
        assert_eq!(mgr.policy_count().await, 1);
        assert!(mgr.allocation_for("tenant-silver").await.is_none());
    }

    #[tokio::test]
    async fn test_budget_manager_all_allocations() {
        let mgr = ResourceBudgetManager::new(32.0, 65_536, 5_000.0, 100_000);
        mgr.register_policy(gold_policy()).await.expect("reg");
        mgr.register_policy(silver_policy()).await.expect("reg");
        let allocs = mgr.all_allocations().await;
        assert_eq!(allocs.len(), 2);
    }

    #[test]
    fn test_compliance_snapshot_violations_list() {
        let snap = ComplianceSnapshot {
            policy_name: "test".into(),
            class: SlaClass::Gold,
            window_requests: 100,
            p50_latency: Duration::from_millis(10),
            p95_latency: Duration::from_millis(150),
            p99_latency: Duration::from_millis(500),
            actual_error_rate: 0.02,
            estimated_rps: 100.0,
            meets_p99_latency: false,
            meets_p95_latency: false,
            meets_throughput: true,
            meets_error_rate: false,
        };
        let vs = snap.violations();
        // Expects P99, P95, and error rate violations.
        assert_eq!(vs.len(), 3);
        assert!(!snap.is_compliant());
    }

    #[test]
    fn test_slo_targets_validate_inconsistent_latency() {
        let mut t = SlaClass::Gold.default_targets();
        // Force p50 > p95 to trigger validation error.
        t.p50_latency = Duration::from_secs(10);
        t.p95_latency = Duration::from_millis(100);
        assert!(t.validate().is_err());
    }
}
