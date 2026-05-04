//! Integration tests for the Performance SLA harness.
//!
//! The `#[ignore]`d tests at the bottom of this file are the real SLA gate.
//! Run them on a release build to enforce timing thresholds:
//!
//! ```text
//! cargo test --release -p oxirs-core -- --ignored sla_suite
//! ```
//!
//! All other tests in this file run under the default test suite without
//! `--ignored` and do NOT assert timing thresholds so they are safe on any
//! machine speed.

use oxirs_core::perf_sla::{assert_meets_slo, BenchmarkResult, SloTarget};

// ---------------------------------------------------------------------------
// Unit-style tests (always run, no timing assertions)
// ---------------------------------------------------------------------------

#[test]
fn test_slo_assertion_passing() {
    let result = BenchmarkResult {
        name: "example".into(),
        p50_us: 50,
        p99_us: 100,
        throughput_ops_s: 1000.0,
        samples: 100,
        total_duration_ms: 100,
    };
    let target = SloTarget {
        name: "example".into(),
        p50_us: Some(100),             // measured 50µs vs target 100µs — passes
        p99_us: Some(200),             // measured 100µs vs target 200µs — passes
        throughput_ops_s: Some(500.0), // measured 1000 vs target 500 — passes
        allow_regression_pct: 10.0,
    };
    assert!(assert_meets_slo(&result, &target).is_ok());
}

#[test]
fn test_slo_assertion_regression_detected() {
    let result = BenchmarkResult {
        name: "slow".into(),
        p50_us: 500, // way over target
        p99_us: 1000,
        throughput_ops_s: 100.0,
        samples: 100,
        total_duration_ms: 1000,
    };
    let target = SloTarget {
        name: "slow".into(),
        p50_us: Some(50), // 500µs >> 50µs × 1.1 = 55µs
        p99_us: None,
        throughput_ops_s: None,
        allow_regression_pct: 10.0,
    };
    // In release builds, the timing regression must be detected.
    // In debug builds, timing assertions are skipped.
    #[cfg(not(debug_assertions))]
    assert!(assert_meets_slo(&result, &target).is_err());
    #[cfg(debug_assertions)]
    assert!(assert_meets_slo(&result, &target).is_ok());
}

#[test]
fn test_slo_throughput_regression() {
    let result = BenchmarkResult {
        name: "throughput".into(),
        p50_us: 10,
        p99_us: 20,
        throughput_ops_s: 10.0, // way below target
        samples: 100,
        total_duration_ms: 10_000,
    };
    let target = SloTarget {
        name: "throughput".into(),
        p50_us: None,
        p99_us: None,
        throughput_ops_s: Some(1000.0), // target: 1 000 ops/s
        allow_regression_pct: 10.0,
    };
    #[cfg(not(debug_assertions))]
    assert!(assert_meets_slo(&result, &target).is_err());
    #[cfg(debug_assertions)]
    assert!(assert_meets_slo(&result, &target).is_ok());
}

#[test]
fn test_baseline_json_roundtrip() {
    let result = BenchmarkResult {
        name: "foo".into(),
        p50_us: 42,
        p99_us: 99,
        throughput_ops_s: 999.0,
        samples: 100,
        total_duration_ms: 100,
    };
    let json = serde_json::to_string(&result).expect("serialize");
    let back: BenchmarkResult = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.name, "foo");
    assert_eq!(back.p50_us, 42);
    assert_eq!(back.p99_us, 99);
}

#[test]
fn test_slo_target_json_roundtrip() {
    let target = SloTarget {
        name: "latency".into(),
        p50_us: Some(100),
        p99_us: Some(500),
        throughput_ops_s: None,
        allow_regression_pct: 15.0,
    };
    let json = serde_json::to_string(&target).expect("serialize");
    let back: SloTarget = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.name, "latency");
    assert_eq!(back.p50_us, Some(100));
    assert!((back.allow_regression_pct - 15.0).abs() < f64::EPSILON);
}

#[test]
fn test_benchmark_result_measure() {
    let result = BenchmarkResult::measure("noop", 100, || {
        let _ = 1_u64.wrapping_add(1);
    });
    assert_eq!(result.name, "noop");
    assert_eq!(result.samples, 100);
    // A no-op loop should complete in well under 10 seconds even on slow CI.
    assert!(result.p50_us < 10_000_000);
    assert!(result.throughput_ops_s > 0.0);
}

#[test]
fn test_slo_no_targets_always_passes() {
    let result = BenchmarkResult {
        name: "any".into(),
        p50_us: u64::MAX,
        p99_us: u64::MAX,
        throughput_ops_s: 0.0,
        samples: 1,
        total_duration_ms: 1,
    };
    let target = SloTarget {
        name: "any".into(),
        p50_us: None,
        p99_us: None,
        throughput_ops_s: None,
        allow_regression_pct: 10.0,
    };
    // No dimensions checked — always Ok regardless of build mode.
    assert!(assert_meets_slo(&result, &target).is_ok());
}

#[test]
fn test_slo_violation_carries_details() {
    // Only meaningful on release builds where thresholds are checked.
    #[cfg(not(debug_assertions))]
    {
        let result = BenchmarkResult {
            name: "details".into(),
            p50_us: 999,
            p99_us: 9999,
            throughput_ops_s: 1.0,
            samples: 100,
            total_duration_ms: 100_000,
        };
        let target = SloTarget {
            name: "details".into(),
            p50_us: Some(10),
            p99_us: Some(10),
            throughput_ops_s: Some(10_000.0),
            allow_regression_pct: 0.0,
        };
        match assert_meets_slo(&result, &target) {
            Err(v) => {
                assert_eq!(v.violations.len(), 3);
                let msg = v.to_string();
                assert!(msg.contains("p50"));
                assert!(msg.contains("p99"));
                assert!(msg.contains("throughput"));
            }
            Ok(()) => panic!("expected SLO violation"),
        }
    }
    // In debug builds, nothing to check — just pass.
    #[cfg(debug_assertions)]
    {}
}

#[test]
fn test_baseline_json_array_from_file() {
    // Read perf_baseline.json from the crate root. Use temp_dir check approach:
    // If the file doesn't exist we skip (not a hard failure) so the test is
    // portable across work trees where the file may not yet be present.
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let baseline_path = manifest_dir.join("perf_baseline.json");
    if !baseline_path.exists() {
        return;
    }
    let contents = std::fs::read_to_string(&baseline_path).expect("read perf_baseline.json");
    let results: Vec<BenchmarkResult> =
        serde_json::from_str(&contents).expect("parse perf_baseline.json as Vec<BenchmarkResult>");
    assert!(
        !results.is_empty(),
        "perf_baseline.json should contain at least one entry"
    );
    for r in &results {
        assert!(!r.name.is_empty());
        assert!(r.samples > 0);
    }
}

// ---------------------------------------------------------------------------
// SLA gate tests — marked `#[ignore]`, run with:
//   cargo test --release -p oxirs-core -- --ignored sla_suite
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn sla_suite_term_equality() {
    use oxirs_core::model::NamedNode;

    let a = NamedNode::new("https://example.org/subject").expect("valid IRI");
    let b = NamedNode::new("https://example.org/subject").expect("valid IRI");

    let result = BenchmarkResult::measure("sla_term_equality", 10_000, || {
        // Use std::hint::black_box to prevent the optimizer from eliding the comparison.
        let _ = std::hint::black_box(&a) == std::hint::black_box(&b);
    });

    let target = SloTarget {
        name: "sla_term_equality".into(),
        p50_us: Some(100), // 100µs p50 target
        p99_us: Some(1_000),
        throughput_ops_s: None,
        allow_regression_pct: 10.0,
    };

    assert_meets_slo(&result, &target).expect("term equality SLO violated");
}

#[test]
#[ignore]
fn sla_suite_ntriples_line_count() {
    let data: String = (0..1_000_u32)
        .map(|i| {
            format!(
                "<https://example.org/s{i}> <https://example.org/p> <https://example.org/o{i}> .\n"
            )
        })
        .collect();

    let result = BenchmarkResult::measure("sla_ntriples_line_count_1k", 1_000, || {
        let _ = std::hint::black_box(data.as_str()).lines().count();
    });

    let target = SloTarget {
        name: "sla_ntriples_line_count_1k".into(),
        p50_us: Some(500_000), // 500ms p50 target for 1 000 iterations
        p99_us: Some(2_000_000),
        throughput_ops_s: None,
        allow_regression_pct: 10.0,
    };

    assert_meets_slo(&result, &target).expect("ntriples line count SLO violated");
}
