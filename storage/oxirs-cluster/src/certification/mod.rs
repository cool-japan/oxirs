//! Comprehensive certification module for `oxirs-cluster`.
//!
//! This module provides a [`CertificationSuite`] that validates the operational
//! correctness guarantees of the cluster crate through a series of in-memory
//! simulations.  No real network sockets are used — all simulation is
//! deterministic and synchronous (std::sync::mpsc channels; no tokio runtime
//! required).
//!
//! # Sub-modules
//!
//! | Module | Checks |
//! |--------|--------|
//! | [`consistency`] | Read-your-writes, linearizability probes, convergence |
//! | [`partition`] | Island formation, quorum loss/recovery, split-brain prevention |
//! | [`raft`] | Leader uniqueness, log monotonicity, safety invariants |
//! | [`sla`] | Read/write p99 latency bounds, throughput floor |
//!
//! # Example
//!
//! ```rust
//! use oxirs_cluster::certification::{CertificationSuite, CertificationConfig};
//!
//! let suite = CertificationSuite::new(CertificationConfig::default());
//! let report = suite.run();
//! assert!(report.passed, "Certification failed: {}", report.summary);
//! ```

pub mod consistency;
pub mod partition;
pub mod raft;
pub mod sla;

// ── Public re-exports ────────────────────────────────────────────────────────

pub use consistency::certify as run_consistency;
pub use partition::certify as run_partition;
pub use raft::certify as run_raft;
pub use sla::certify as run_sla;

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for all certification sub-suites.
#[derive(Debug, Clone)]
pub struct CertificationConfig {
    /// Number of simulated cluster nodes (default 5).
    pub node_count: usize,
    /// Fraction of nodes isolated in partition tests (default 0.4 = 40%).
    ///
    /// With `node_count=5` this isolates 2 nodes, leaving a quorum-capable
    /// majority of 3.
    pub partition_fraction: f64,
    /// Read p99 latency threshold in microseconds (default 1 000 µs = 1 ms).
    pub sla_read_latency_us: u64,
    /// Write p99 latency threshold in microseconds (default 5 000 µs = 5 ms).
    pub sla_write_latency_us: u64,
    /// Number of epochs / iterations for each check (default 50).
    pub epochs: usize,
}

impl Default for CertificationConfig {
    fn default() -> Self {
        CertificationConfig {
            node_count: 5,
            partition_fraction: 0.4,
            sla_read_latency_us: 1_000,
            sla_write_latency_us: 5_000,
            epochs: 50,
        }
    }
}

// ── Result types ─────────────────────────────────────────────────────────────

/// A single named check within a certification sub-suite.
#[derive(Debug, Clone)]
pub struct CheckResult {
    /// Short, machine-readable identifier (snake_case).
    pub name: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Human-readable detail string (one line).
    pub detail: String,
}

/// Aggregate result of the consistency certification.
#[derive(Debug, Clone)]
pub struct ConsistencyResult {
    /// Overall pass / fail for this sub-suite.
    pub passed: bool,
    /// Individual check results.
    pub checks: Vec<CheckResult>,
    /// Prose summary of the consistency run.
    pub notes: String,
}

/// Aggregate result of the network-partition resilience certification.
#[derive(Debug, Clone)]
pub struct PartitionResult {
    /// Overall pass / fail for this sub-suite.
    pub passed: bool,
    /// Individual check results.
    pub checks: Vec<CheckResult>,
    /// Prose summary of the partition run.
    pub notes: String,
}

/// Aggregate result of the Raft correctness certification.
#[derive(Debug, Clone)]
pub struct RaftResult {
    /// Overall pass / fail for this sub-suite.
    pub passed: bool,
    /// Individual check results.
    pub checks: Vec<CheckResult>,
    /// Prose summary of the Raft run.
    pub notes: String,
}

/// Aggregate result of the SLA / latency certification.
#[derive(Debug, Clone)]
pub struct SlaResult {
    /// Overall pass / fail for this sub-suite.
    pub passed: bool,
    /// Individual check results.
    pub checks: Vec<CheckResult>,
    /// Prose summary of the SLA run.
    pub notes: String,
}

/// Full certification report produced by [`CertificationSuite::run`].
#[derive(Debug, Clone)]
pub struct CertificationReport {
    /// `true` if and only if all four sub-suites passed.
    pub passed: bool,
    /// Consistency guarantee results.
    pub consistency_result: ConsistencyResult,
    /// Network partition resilience results.
    pub partition_result: PartitionResult,
    /// Raft correctness property results.
    pub raft_result: RaftResult,
    /// SLA / latency bound results.
    pub sla_result: SlaResult,
    /// One-line human-readable summary.
    pub summary: String,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Entry point for the comprehensive certification suite.
///
/// Construct with [`CertificationSuite::new`] and call [`CertificationSuite::run`]
/// to obtain a [`CertificationReport`].
pub struct CertificationSuite {
    config: CertificationConfig,
}

impl CertificationSuite {
    /// Create a new certification suite with the given configuration.
    pub fn new(config: CertificationConfig) -> Self {
        CertificationSuite { config }
    }

    /// Run all four certification sub-suites synchronously and return a report.
    ///
    /// This method is intentionally synchronous — all sub-suites use in-memory
    /// simulation with `std::sync::mpsc`; no tokio runtime is required.
    pub fn run(&self) -> CertificationReport {
        let consistency_result = consistency::certify(&self.config);
        let partition_result = partition::certify(&self.config);
        let raft_result = raft::certify(&self.config);
        let sla_result = sla::certify(&self.config);

        let passed = consistency_result.passed
            && partition_result.passed
            && raft_result.passed
            && sla_result.passed;

        let summary = build_summary(
            passed,
            &consistency_result,
            &partition_result,
            &raft_result,
            &sla_result,
        );

        CertificationReport {
            passed,
            consistency_result,
            partition_result,
            raft_result,
            sla_result,
            summary,
        }
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn build_summary(
    passed: bool,
    c: &ConsistencyResult,
    p: &PartitionResult,
    r: &RaftResult,
    s: &SlaResult,
) -> String {
    let status = if passed { "PASS" } else { "FAIL" };
    let mut parts: Vec<String> = Vec::new();

    for (label, result_passed) in [
        ("consistency", c.passed),
        ("partition", p.passed),
        ("raft", r.passed),
        ("sla", s.passed),
    ] {
        let marker = if result_passed { "[+]" } else { "[-]" };
        parts.push(format!("{marker} {label}"));
    }

    format!("[{status}] Certification: {}", parts.join("  "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_fields() {
        let cfg = CertificationConfig::default();
        assert_eq!(cfg.node_count, 5);
        assert!((cfg.partition_fraction - 0.4).abs() < 1e-9);
        assert_eq!(cfg.sla_read_latency_us, 1_000);
        assert_eq!(cfg.sla_write_latency_us, 5_000);
        assert_eq!(cfg.epochs, 50);
    }

    #[test]
    fn test_suite_new_does_not_panic() {
        let _suite = CertificationSuite::new(CertificationConfig::default());
    }

    #[test]
    fn test_run_returns_report() {
        let suite = CertificationSuite::new(CertificationConfig::default());
        let report = suite.run();
        // Just ensure we get a report back.
        assert!(!report.summary.is_empty());
    }

    #[test]
    fn test_report_summary_contains_status() {
        let suite = CertificationSuite::new(CertificationConfig::default());
        let report = suite.run();
        assert!(
            report.summary.contains("PASS") || report.summary.contains("FAIL"),
            "Summary should contain PASS or FAIL: {}",
            report.summary
        );
    }

    #[test]
    fn test_report_passed_flag() {
        let suite = CertificationSuite::new(CertificationConfig::default());
        let report = suite.run();
        let all_pass = report.consistency_result.passed
            && report.partition_result.passed
            && report.raft_result.passed
            && report.sla_result.passed;
        assert_eq!(report.passed, all_pass);
    }

    #[test]
    fn test_build_summary_all_pass() {
        let c = ConsistencyResult {
            passed: true,
            checks: vec![],
            notes: String::new(),
        };
        let p = PartitionResult {
            passed: true,
            checks: vec![],
            notes: String::new(),
        };
        let r = RaftResult {
            passed: true,
            checks: vec![],
            notes: String::new(),
        };
        let s = SlaResult {
            passed: true,
            checks: vec![],
            notes: String::new(),
        };
        let summary = build_summary(true, &c, &p, &r, &s);
        assert!(summary.contains("PASS"), "Expected PASS in: {summary}");
    }

    #[test]
    fn test_build_summary_with_failure() {
        let c = ConsistencyResult {
            passed: false,
            checks: vec![],
            notes: String::new(),
        };
        let p = PartitionResult {
            passed: true,
            checks: vec![],
            notes: String::new(),
        };
        let r = RaftResult {
            passed: true,
            checks: vec![],
            notes: String::new(),
        };
        let s = SlaResult {
            passed: true,
            checks: vec![],
            notes: String::new(),
        };
        let summary = build_summary(false, &c, &p, &r, &s);
        assert!(summary.contains("FAIL"), "Expected FAIL in: {summary}");
    }
}
