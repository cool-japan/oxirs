//! Integration tests for the ARQ runtime query resource governor.
//!
//! These tests exercise [`oxirs_arq::query_governor`] in isolation — no
//! dataset or algebra required.

use oxirs_arq::query_governor::{BudgetExceeded, ExecutionBudget, ResourceBudget};
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

fn unlimited() -> std::sync::Arc<ExecutionBudget> {
    ExecutionBudget::new(ResourceBudget::unlimited())
}

// ─────────────────────────────────────────────────────────────────────────────
// Core limit tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_unlimited_budget_never_triggers() {
    let budget = unlimited();
    // Extreme values that would normally blow any limit
    budget
        .record_triple_scan(u64::MAX)
        .expect("unlimited triple scan should never error");
    for _ in 0..1_000 {
        budget
            .record_result_row()
            .expect("unlimited result rows should never error");
    }
    budget
        .check_time()
        .expect("unlimited time should never error");
}

#[test]
fn test_triple_scan_limit_enforced() {
    let budget = ExecutionBudget::new(ResourceBudget {
        max_wall_time: None,
        max_result_rows: None,
        max_triples_scanned: Some(100),
    });

    // Scanning exactly the limit is OK
    budget
        .record_triple_scan(100)
        .expect("scanning exactly 100 should be within limit");

    // One more scan pushes total to 101 — must fail
    let err = budget
        .record_triple_scan(1)
        .expect_err("101 triples scanned should exceed limit of 100");
    assert!(
        matches!(err, BudgetExceeded::TriplesScannedExceeded { scanned, limit } if scanned == 101 && limit == 100),
        "unexpected error variant: {err:?}"
    );
}

#[test]
fn test_result_row_limit_enforced() {
    let budget = ExecutionBudget::new(ResourceBudget {
        max_wall_time: None,
        max_result_rows: Some(5),
        max_triples_scanned: None,
    });

    for _ in 0..5 {
        budget
            .record_result_row()
            .expect("rows 1-5 should be within limit");
    }

    let err = budget
        .record_result_row()
        .expect_err("6th row should exceed limit of 5");
    assert!(
        matches!(err, BudgetExceeded::ResultRowsExceeded { produced, limit } if produced == 6 && limit == 5),
        "unexpected error variant: {err:?}"
    );
}

#[test]
fn test_timeout_enforced() {
    // Set a very tight wall-time limit (10 ms), then sleep 60 ms past it.
    let budget = ExecutionBudget::new(ResourceBudget {
        max_wall_time: Some(Duration::from_millis(10)),
        max_result_rows: None,
        max_triples_scanned: None,
    });

    // Spin briefly to ensure we cross the 10 ms boundary without relying on
    // std::thread::sleep accuracy
    let deadline = std::time::Instant::now() + Duration::from_millis(80);
    while std::time::Instant::now() < deadline {
        std::hint::spin_loop();
    }

    let err = budget
        .check_time()
        .expect_err("should detect timeout after > 10 ms");
    assert!(
        matches!(err, BudgetExceeded::TimeoutExceeded { elapsed_ms, limit_ms } if elapsed_ms >= 10 && limit_ms == 10),
        "unexpected error variant: {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// SLA tier tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_bronze_tier_budget() {
    let b = ResourceBudget::for_sla_tier("bronze");
    assert_eq!(b.max_wall_time, Some(Duration::from_secs(30)));
    assert_eq!(b.max_result_rows, Some(1_000));
    assert_eq!(b.max_triples_scanned, Some(1_000_000));
}

#[test]
fn test_silver_tier_budget() {
    let b = ResourceBudget::for_sla_tier("silver");
    assert_eq!(b.max_wall_time, Some(Duration::from_secs(60)));
    assert_eq!(b.max_result_rows, Some(10_000));
    assert_eq!(b.max_triples_scanned, Some(5_000_000));
}

#[test]
fn test_gold_tier_budget() {
    let b = ResourceBudget::for_sla_tier("gold");
    assert_eq!(b.max_wall_time, Some(Duration::from_secs(300)));
    assert_eq!(b.max_result_rows, Some(100_000));
    assert_eq!(b.max_triples_scanned, Some(50_000_000));
}

#[test]
fn test_platinum_unlimited() {
    let b = ResourceBudget::for_sla_tier("platinum");
    assert!(
        b.max_wall_time.is_none(),
        "platinum must have no time limit"
    );
    assert!(
        b.max_result_rows.is_none(),
        "platinum must have no row limit"
    );
    assert!(
        b.max_triples_scanned.is_none(),
        "platinum must have no scan limit"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Counter / state tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_counters_accumulate() {
    let budget = unlimited();

    budget
        .record_triple_scan(42)
        .expect("within unlimited budget");
    budget
        .record_triple_scan(58)
        .expect("within unlimited budget");
    assert_eq!(
        budget.triples_scanned(),
        100,
        "triple scan counter should accumulate to 100"
    );

    for _ in 0..7 {
        budget.record_result_row().expect("within unlimited budget");
    }
    assert_eq!(
        budget.result_rows(),
        7,
        "result row counter should accumulate to 7"
    );
}

#[test]
fn test_budget_exceeded_display() {
    let timeout = BudgetExceeded::TimeoutExceeded {
        elapsed_ms: 500,
        limit_ms: 100,
    };
    let msg = timeout.to_string();
    assert!(
        msg.contains("500") && msg.contains("100"),
        "display should include elapsed and limit: {msg}"
    );

    let rows = BudgetExceeded::ResultRowsExceeded {
        produced: 11,
        limit: 10,
    };
    let msg = rows.to_string();
    assert!(
        msg.contains("11") && msg.contains("10"),
        "display should include produced and limit: {msg}"
    );

    let scans = BudgetExceeded::TriplesScannedExceeded {
        scanned: 999_999,
        limit: 500_000,
    };
    let msg = scans.to_string();
    assert!(
        msg.contains("999999") && msg.contains("500000"),
        "display should include scanned and limit: {msg}"
    );
}

#[test]
fn test_two_independent_budgets() {
    // Two separate ExecutionBudget instances must not share counters.
    let budget_a = ExecutionBudget::new(ResourceBudget {
        max_wall_time: None,
        max_result_rows: None,
        max_triples_scanned: Some(50),
    });
    let budget_b = ExecutionBudget::new(ResourceBudget {
        max_wall_time: None,
        max_result_rows: None,
        max_triples_scanned: Some(50),
    });

    budget_a
        .record_triple_scan(50)
        .expect("A at limit should succeed");
    // B should still be at zero — scanning on A must not affect B
    assert_eq!(
        budget_b.triples_scanned(),
        0,
        "B counter must be independent of A"
    );

    budget_b
        .record_triple_scan(50)
        .expect("B at limit should succeed");
    // A should fail now (already at 50, one more pushes to 51)
    let err = budget_a
        .record_triple_scan(1)
        .expect_err("A at 51/50 should fail");
    assert!(matches!(err, BudgetExceeded::TriplesScannedExceeded { .. }));
}

#[test]
fn test_check_time_on_fresh_budget_passes() {
    // A brand-new budget with a generous time limit must always pass immediately.
    let budget = ExecutionBudget::new(ResourceBudget {
        max_wall_time: Some(Duration::from_secs(3600)),
        max_result_rows: None,
        max_triples_scanned: None,
    });
    budget
        .check_time()
        .expect("brand-new budget with 1-hour limit must not timeout immediately");
}

// ─────────────────────────────────────────────────────────────────────────────
// Additional robustness tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_budget_exceeded_is_error() {
    // BudgetExceeded must satisfy std::error::Error
    let e: Box<dyn std::error::Error> = Box::new(BudgetExceeded::TimeoutExceeded {
        elapsed_ms: 1,
        limit_ms: 0,
    });
    assert!(!e.to_string().is_empty());
}

#[test]
fn test_scan_batch_at_exact_limit_ok() {
    let budget = ExecutionBudget::new(ResourceBudget {
        max_wall_time: None,
        max_result_rows: None,
        max_triples_scanned: Some(1_000_000),
    });
    // One big batch scan exactly at the limit — must succeed
    budget
        .record_triple_scan(1_000_000)
        .expect("batch scan at exact limit should be OK");
}

#[test]
fn test_case_insensitive_tier_names() {
    let b1 = ResourceBudget::for_sla_tier("GOLD");
    let b2 = ResourceBudget::for_sla_tier("Gold");
    let b3 = ResourceBudget::for_sla_tier("gold");
    assert_eq!(b1.max_result_rows, b2.max_result_rows);
    assert_eq!(b2.max_result_rows, b3.max_result_rows);
}

#[test]
fn test_unknown_tier_is_unlimited() {
    let b = ResourceBudget::for_sla_tier("diamond");
    assert!(b.max_wall_time.is_none());
    assert!(b.max_result_rows.is_none());
    assert!(b.max_triples_scanned.is_none());
}
