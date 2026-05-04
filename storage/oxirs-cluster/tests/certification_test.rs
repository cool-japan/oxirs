//! Integration tests for the comprehensive certification module.
//!
//! Each test exercises the public API of `oxirs_cluster::certification` from
//! outside the crate boundary, verifying that the suite compiles, runs, and
//! produces well-formed reports.

use oxirs_cluster::certification::{
    CertificationConfig, CertificationSuite, CheckResult, ConsistencyResult, PartitionResult,
    RaftResult, SlaResult,
};

// ── Helper ────────────────────────────────────────────────────────────────────

fn default_suite() -> CertificationSuite {
    CertificationSuite::new(CertificationConfig::default())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// The default config runs without panicking and produces a report.
#[test]
fn test_default_config_runs() {
    let suite = default_suite();
    let report = suite.run();
    // Just assert it came back — further assertions in subsequent tests.
    assert!(!report.summary.is_empty());
}

/// Running the suite produces a structurally complete CertificationReport.
#[test]
fn test_certification_produces_report() {
    let suite = default_suite();
    let report = suite.run();

    assert!(
        report.passed,
        "Certification should pass with default config.\nSummary: {}",
        report.summary
    );
    assert!(!report.summary.is_empty(), "Summary must not be empty");
}

/// The consistency sub-result has the expected structure.
#[test]
fn test_consistency_result_structure() {
    let suite = default_suite();
    let report = suite.run();
    let cr: &ConsistencyResult = &report.consistency_result;

    assert!(cr.passed, "Consistency checks should pass");
    assert!(
        !cr.checks.is_empty(),
        "At least one consistency check expected"
    );
    assert!(!cr.notes.is_empty(), "Consistency notes must not be empty");

    for check in &cr.checks {
        assert!(!check.name.is_empty(), "Check name must not be empty");
        assert!(!check.detail.is_empty(), "Check detail must not be empty");
    }
}

/// The partition sub-result has the expected structure.
#[test]
fn test_partition_result_structure() {
    let suite = default_suite();
    let report = suite.run();
    let pr: &PartitionResult = &report.partition_result;

    assert!(pr.passed, "Partition checks should pass");
    assert!(
        !pr.checks.is_empty(),
        "At least one partition check expected"
    );
    assert!(!pr.notes.is_empty(), "Partition notes must not be empty");

    for check in &pr.checks {
        assert!(!check.name.is_empty());
        assert!(!check.detail.is_empty());
    }
}

/// Leader uniqueness is verified by the Raft sub-suite.
#[test]
fn test_raft_leader_uniqueness() {
    let suite = default_suite();
    let report = suite.run();
    let rr: &RaftResult = &report.raft_result;

    let lu_check: Option<&CheckResult> = rr.checks.iter().find(|c| c.name == "leader_uniqueness");

    let check = lu_check.expect("leader_uniqueness check must be present");
    assert!(
        check.passed,
        "Leader uniqueness check failed: {}",
        check.detail
    );
}

/// Log monotonicity is verified by the Raft sub-suite.
#[test]
fn test_raft_log_monotonicity() {
    let suite = default_suite();
    let report = suite.run();
    let rr: &RaftResult = &report.raft_result;

    let lm_check: Option<&CheckResult> = rr.checks.iter().find(|c| c.name == "log_monotonicity");

    let check = lm_check.expect("log_monotonicity check must be present");
    assert!(
        check.passed,
        "Log monotonicity check failed: {}",
        check.detail
    );
}

/// The SLA sub-result has the expected structure and passes.
#[test]
fn test_sla_result_structure() {
    let suite = default_suite();
    let report = suite.run();
    let sr: &SlaResult = &report.sla_result;

    assert!(sr.passed, "SLA checks should pass with default config");
    assert!(sr.checks.len() >= 2, "Expected at least 2 SLA checks");
    assert!(!sr.notes.is_empty(), "SLA notes must not be empty");

    for check in &sr.checks {
        assert!(!check.name.is_empty());
        assert!(!check.detail.is_empty());
    }
}

/// The report summary is non-empty and contains a status marker.
#[test]
fn test_report_summary_nonempty() {
    let suite = default_suite();
    let report = suite.run();

    assert!(!report.summary.is_empty());
    assert!(
        report.summary.contains("PASS") || report.summary.contains("FAIL"),
        "Summary should contain PASS or FAIL: '{}'",
        report.summary
    );
}

/// A single-node cluster still produces a valid certification report.
#[test]
fn test_single_node_certification() {
    let config = CertificationConfig {
        node_count: 1,
        partition_fraction: 0.0,
        ..CertificationConfig::default()
    };
    let suite = CertificationSuite::new(config);
    let report = suite.run();

    // For n=1, quorum=1, so all properties are trivially satisfiable.
    assert!(
        report.passed,
        "Single-node certification should pass.\nSummary: {}",
        report.summary
    );
}

/// A custom config with different parameters still produces a valid report.
#[test]
fn test_custom_config() {
    let config = CertificationConfig {
        node_count: 7,
        partition_fraction: 0.3,
        sla_read_latency_us: 2_000,
        sla_write_latency_us: 10_000,
        epochs: 20,
    };
    let suite = CertificationSuite::new(config);
    let report = suite.run();

    assert!(
        report.passed,
        "Custom config certification should pass.\nSummary: {}",
        report.summary
    );
    assert!(!report.summary.is_empty());
}

/// The Raft sub-suite contains the expected checks.
#[test]
fn test_raft_result_structure() {
    let suite = default_suite();
    let report = suite.run();
    let rr: &RaftResult = &report.raft_result;

    assert!(rr.passed, "Raft certification should pass");
    assert!(rr.checks.len() >= 3, "Expected at least 3 Raft checks");

    let names: Vec<&str> = rr.checks.iter().map(|c| c.name.as_str()).collect();
    assert!(
        names.contains(&"leader_uniqueness"),
        "Missing leader_uniqueness check"
    );
    assert!(
        names.contains(&"log_monotonicity"),
        "Missing log_monotonicity check"
    );
    assert!(
        names.contains(&"safety_invariant"),
        "Missing safety_invariant check"
    );
}

/// The report `passed` flag is the logical AND of all four sub-results.
#[test]
fn test_report_passed_is_conjunction_of_sub_results() {
    let suite = default_suite();
    let report = suite.run();

    let expected = report.consistency_result.passed
        && report.partition_result.passed
        && report.raft_result.passed
        && report.sla_result.passed;

    assert_eq!(
        report.passed, expected,
        "report.passed should be AND of all sub-results"
    );
}

/// Read-your-writes is specifically checked in consistency.
#[test]
fn test_consistency_read_your_writes_check() {
    let suite = default_suite();
    let report = suite.run();
    let cr: &ConsistencyResult = &report.consistency_result;

    let ryw: Option<&CheckResult> = cr.checks.iter().find(|c| c.name == "read_your_writes");

    let check = ryw.expect("read_your_writes check must be present");
    assert!(check.passed, "read_your_writes failed: {}", check.detail);
}
