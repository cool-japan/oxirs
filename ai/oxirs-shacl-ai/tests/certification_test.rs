//! Integration tests for the ML model certification suite.
//!
//! Covers 15 distinct scenarios spanning metrics math, runner threshold logic,
//! confusion matrix bookkeeping, and report rendering.

use oxirs_shacl_ai::certification::{
    CertificationCase, CertificationReport, CertificationRunner, CertificationStatus,
    CertificationSuite, ClassificationMetrics, ConfusionMatrix, ConstraintTypeMetrics,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `CertificationCase` with no confidence annotation.
fn case(id: &str, constraint_type: &str, truth: bool, predicted: bool) -> CertificationCase {
    CertificationCase {
        id: id.to_string(),
        constraint_type: constraint_type.to_string(),
        ground_truth_violation: truth,
        model_predicted_violation: predicted,
        confidence: None,
    }
}

/// Create exactly `n` perfect cases of the given constraint type.
fn perfect_cases(n: usize, ct: &str) -> Vec<CertificationCase> {
    (0..n)
        .map(|i| {
            let violation = i % 2 == 0;
            case(&format!("c{i}"), ct, violation, violation)
        })
        .collect()
}

/// Create a suite where every prediction is exactly wrong.
fn all_wrong_cases(n: usize) -> Vec<CertificationCase> {
    (0..n)
        .map(|i| {
            let truth = i % 2 == 0;
            case(&format!("w{i}"), "sh:minCount", truth, !truth)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test 1: Perfect model → F1 = 1.0, status = Passed
// ---------------------------------------------------------------------------
#[test]
fn test_perfect_model_passes() {
    let suite = CertificationSuite::from_cases("perfect", perfect_cases(20, "sh:minCount"));
    let runner = CertificationRunner::new();
    let report = runner.run(&suite);

    assert!(report.passed(), "perfect model should pass");
    assert!(
        (report.overall_metrics.f1_score() - 1.0).abs() < 1e-10,
        "perfect model F1 should be 1.0"
    );
    assert_eq!(report.total_cases, 20);
}

// ---------------------------------------------------------------------------
// Test 2: All predictions wrong → F1 = 0.0, status = Failed
// ---------------------------------------------------------------------------
#[test]
fn test_all_wrong_predictions_fails() {
    // 5 true→false and 5 false→true: TP=0, TN=0, FP=5, FN=5
    let cases: Vec<CertificationCase> = (0..10)
        .map(|i| {
            let truth = i < 5; // first 5 are violations
            case(&format!("w{i}"), "sh:minCount", truth, !truth)
        })
        .collect();
    let suite = CertificationSuite::from_cases("all-wrong", cases);
    let runner = CertificationRunner::new();
    let report = runner.run(&suite);

    assert!(!report.passed(), "all-wrong model should fail");
    assert_eq!(
        report.overall_metrics.f1_score(),
        0.0,
        "F1 should be 0.0 for all-wrong model"
    );
    assert!(matches!(report.status, CertificationStatus::Failed { .. }));
}

// ---------------------------------------------------------------------------
// Test 3: High precision, low recall → threshold failure for recall
// ---------------------------------------------------------------------------
#[test]
fn test_high_precision_low_recall_fails_recall_threshold() {
    // 10 true violations, model only catches 2 (TP=2, FN=8).
    // 10 non-violations, model correctly says "ok" (TN=10, FP=0).
    // Precision = 2/(2+0) = 1.0,  Recall = 2/(2+8) = 0.2
    let mut cases: Vec<CertificationCase> = (0..10)
        .map(|i| case(&format!("viol{i}"), "sh:minCount", true, i < 2))
        .collect();
    cases.extend((0..10).map(|i| case(&format!("ok{i}"), "sh:minCount", false, false)));

    let suite = CertificationSuite::from_cases("high-prec-low-recall", cases);
    let runner = CertificationRunner::new(); // min_recall=0.70
    let report = runner.run(&suite);

    assert!(!report.passed());
    let CertificationStatus::Failed { reasons } = &report.status else {
        panic!("Expected Failed status");
    };
    assert!(
        reasons.iter().any(|r| r.to_lowercase().contains("recall")),
        "failure reasons should mention recall: {reasons:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Insufficient cases (< 10) → Insufficient status
// ---------------------------------------------------------------------------
#[test]
fn test_insufficient_cases_fewer_than_10() {
    let suite = CertificationSuite::from_cases(
        "tiny",
        vec![
            case("c1", "sh:minCount", true, true),
            case("c2", "sh:minCount", false, false),
            case("c3", "sh:minCount", true, true),
        ],
    );
    let runner = CertificationRunner::new();
    let report = runner.run(&suite);

    assert!(
        matches!(report.status, CertificationStatus::Insufficient { .. }),
        "should be Insufficient when fewer than 10 cases"
    );
    assert!(!report.passed());
}

// ---------------------------------------------------------------------------
// Test 5: Mixed constraint types → per_constraint_metrics has multiple entries
// ---------------------------------------------------------------------------
#[test]
fn test_mixed_constraint_types_per_constraint_metrics() {
    let mut cases = perfect_cases(10, "sh:minCount");
    cases.extend(perfect_cases(10, "sh:pattern"));

    let suite = CertificationSuite::from_cases("mixed", cases);
    let runner = CertificationRunner::new();
    let report = runner.run(&suite);

    assert_eq!(
        report.per_constraint_metrics.len(),
        2,
        "should have metrics for two distinct constraint types"
    );
    let types: Vec<&str> = report
        .per_constraint_metrics
        .iter()
        .map(|m| m.constraint_type.as_str())
        .collect();
    assert!(types.contains(&"sh:minCount"), "should contain sh:minCount");
    assert!(types.contains(&"sh:pattern"), "should contain sh:pattern");
}

// ---------------------------------------------------------------------------
// Test 6: MCC with balanced set
// ---------------------------------------------------------------------------
#[test]
fn test_mcc_balanced_perfect() {
    // TP=5, TN=5, FP=0, FN=0 → MCC = 1.0
    let m = ClassificationMetrics {
        true_positives: 5,
        false_positives: 0,
        true_negatives: 5,
        false_negatives: 0,
    };
    let mcc = m.matthew_correlation_coefficient();
    assert!(
        (mcc - 1.0).abs() < 1e-10,
        "perfect balanced MCC should be 1.0, got {mcc}"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Confusion matrix records correctly and per_class_metrics match
// ---------------------------------------------------------------------------
#[test]
fn test_confusion_matrix_records_correctly() {
    let mut cm = ConfusionMatrix::new(vec!["sh:minCount".to_string(), "sh:pattern".to_string()]);

    // 3 minCount correct, 1 minCount→pattern (FN for minCount)
    cm.record("sh:minCount", "sh:minCount");
    cm.record("sh:minCount", "sh:minCount");
    cm.record("sh:minCount", "sh:minCount");
    cm.record("sh:minCount", "sh:pattern"); // misclassification

    // 4 pattern correct
    cm.record("sh:pattern", "sh:pattern");
    cm.record("sh:pattern", "sh:pattern");
    cm.record("sh:pattern", "sh:pattern");
    cm.record("sh:pattern", "sh:pattern");

    let per_class = cm.per_class_metrics();

    let min_count_m = per_class
        .iter()
        .find(|m| m.constraint_type == "sh:minCount")
        .expect("minCount metrics should be present");

    assert_eq!(min_count_m.metrics.true_positives, 3, "TP for minCount");
    assert_eq!(min_count_m.metrics.false_negatives, 1, "FN for minCount");
    assert_eq!(
        min_count_m.metrics.false_positives, 0,
        "FP for minCount (no pattern→minCount)"
    );
    assert_eq!(min_count_m.sample_count, 4, "4 minCount ground-truth cases");
}

// ---------------------------------------------------------------------------
// Test 8: Markdown report contains key sections and constraint type
// ---------------------------------------------------------------------------
#[test]
fn test_markdown_report_contains_key_sections() {
    let mut cases = perfect_cases(12, "sh:minCount");
    cases.extend(perfect_cases(8, "sh:pattern"));

    let suite = CertificationSuite::from_cases("md-suite", cases);
    let runner = CertificationRunner::new();
    let report = runner.run(&suite);
    let md = report.to_markdown();

    assert!(
        md.contains("md-suite"),
        "markdown should contain suite name"
    );
    assert!(
        md.contains("Overall Metrics"),
        "markdown should contain overall metrics section"
    );
    assert!(
        md.contains("Per-Constraint Type Metrics"),
        "markdown should contain per-constraint section"
    );
    assert!(
        md.contains("sh:minCount"),
        "markdown should list sh:minCount constraint type"
    );
    assert!(
        md.contains("Precision"),
        "markdown should list Precision metric"
    );
    assert!(md.contains("Recall"), "markdown should list Recall metric");
    assert!(md.contains("F1"), "markdown should list F1 metric");
    assert!(md.contains("MCC"), "markdown should list MCC metric");
}

// ---------------------------------------------------------------------------
// Test 9: CertificationRunner::new() defaults are sane
// ---------------------------------------------------------------------------
#[test]
fn test_runner_defaults_are_sane() {
    let runner = CertificationRunner::new();
    assert!(
        runner.min_f1_threshold > 0.0 && runner.min_f1_threshold <= 1.0,
        "F1 threshold should be in (0, 1]"
    );
    assert!(
        runner.min_precision_threshold > 0.0 && runner.min_precision_threshold <= 1.0,
        "precision threshold should be in (0, 1]"
    );
    assert!(
        runner.min_recall_threshold > 0.0 && runner.min_recall_threshold <= 1.0,
        "recall threshold should be in (0, 1]"
    );
    assert!(
        (runner.min_f1_threshold - 0.80).abs() < 1e-10,
        "default F1 threshold should be 0.80"
    );
}

// ---------------------------------------------------------------------------
// Test 10: CertificationReport::passed() matches status
// ---------------------------------------------------------------------------
#[test]
fn test_passed_reflects_status() {
    // Passed
    let passed_report = {
        let suite = CertificationSuite::from_cases("p", perfect_cases(20, "sh:minCount"));
        CertificationRunner::new().run(&suite)
    };
    assert!(
        passed_report.passed(),
        "passed_report.passed() should be true"
    );
    assert_eq!(passed_report.status, CertificationStatus::Passed);

    // Failed
    let failed_report = {
        let suite = CertificationSuite::from_cases("f", all_wrong_cases(10));
        CertificationRunner::new().run(&suite)
    };
    assert!(
        !failed_report.passed(),
        "failed_report.passed() should be false"
    );
}

// ---------------------------------------------------------------------------
// Test 11: Precision = 0 when all predictions are false positives
// ---------------------------------------------------------------------------
#[test]
fn test_precision_zero_when_all_false_positives() {
    // All ground truths are "no violation", model always predicts "violation"
    // → FP=10, TN=0, TP=0, FN=0
    let cases: Vec<CertificationCase> = (0..10)
        .map(|i| case(&format!("fp{i}"), "sh:minCount", false, true))
        .collect();
    let suite = CertificationSuite::from_cases("all-fp", cases);
    let report = CertificationRunner::new().run(&suite);

    assert_eq!(
        report.overall_metrics.precision(),
        0.0,
        "precision should be 0.0 when all predictions are false positives"
    );
    assert_eq!(report.overall_metrics.true_positives, 0);
    assert_eq!(report.overall_metrics.false_positives, 10);
}

// ---------------------------------------------------------------------------
// Test 12: Recall = 0 when all ground-truth violations are missed
// ---------------------------------------------------------------------------
#[test]
fn test_recall_zero_when_all_violations_missed() {
    // All cases are violations, model always predicts "no violation"
    // → TP=0, FN=10, FP=0, TN=0
    let cases: Vec<CertificationCase> = (0..10)
        .map(|i| case(&format!("fn{i}"), "sh:minCount", true, false))
        .collect();
    let suite = CertificationSuite::from_cases("all-fn", cases);
    let report = CertificationRunner::new().run(&suite);

    assert_eq!(
        report.overall_metrics.recall(),
        0.0,
        "recall should be 0.0 when all violations are missed"
    );
    assert_eq!(report.overall_metrics.true_positives, 0);
    assert_eq!(report.overall_metrics.false_negatives, 10);
}

// ---------------------------------------------------------------------------
// Test 13: F1 symmetric: P=0.8, R=0.8 → F1 ≈ 0.8
// ---------------------------------------------------------------------------
#[test]
fn test_f1_symmetric_precision_recall_equal() {
    // TP=8, FP=2, FN=2, TN=8 → P=0.8, R=0.8, F1=0.8
    let m = ClassificationMetrics {
        true_positives: 8,
        false_positives: 2,
        true_negatives: 8,
        false_negatives: 2,
    };
    let p = m.precision();
    let r = m.recall();
    let f1 = m.f1_score();

    assert!((p - 0.8).abs() < 1e-10, "precision should be 0.8, got {p}");
    assert!((r - 0.8).abs() < 1e-10, "recall should be 0.8, got {r}");
    assert!(
        (f1 - 0.8).abs() < 1e-10,
        "F1 should be 0.8 when P=R=0.8, got {f1}"
    );
}

// ---------------------------------------------------------------------------
// Test 14: MCC = -1 for perfectly wrong predictions
// ---------------------------------------------------------------------------
#[test]
fn test_mcc_minus_one_for_perfectly_wrong() {
    // 5 true violations predicted as "no violation" and
    // 5 non-violations predicted as "violation":
    // TP=0, TN=0, FP=5, FN=5
    let m = ClassificationMetrics {
        true_positives: 0,
        false_positives: 5,
        true_negatives: 0,
        false_negatives: 5,
    };
    let mcc = m.matthew_correlation_coefficient();
    assert!(
        (mcc - (-1.0)).abs() < 1e-10,
        "MCC for perfectly wrong classifier should be -1.0, got {mcc}"
    );
}

// ---------------------------------------------------------------------------
// Test 15: Per-constraint filtering — sh:minCount separate from sh:pattern
// ---------------------------------------------------------------------------
#[test]
fn test_per_constraint_filtering_separates_types() {
    // sh:minCount: all perfect (TP=5, TN=5)
    let mut cases: Vec<CertificationCase> = (0..10)
        .map(|i| {
            let v = i % 2 == 0;
            case(&format!("mc{i}"), "sh:minCount", v, v)
        })
        .collect();

    // sh:pattern: all wrong (FP=5, FN=5)
    cases.extend((0..10).map(|i| {
        let v = i % 2 == 0;
        case(&format!("pat{i}"), "sh:pattern", v, !v)
    }));

    let suite = CertificationSuite::from_cases("filter-test", cases);
    let runner = CertificationRunner::with_thresholds(0.0, 0.0, 0.0); // disable thresholds
    let report = runner.run(&suite);

    let mc = report
        .per_constraint_metrics
        .iter()
        .find(|m| m.constraint_type == "sh:minCount")
        .expect("minCount metrics should exist");
    let pat = report
        .per_constraint_metrics
        .iter()
        .find(|m| m.constraint_type == "sh:pattern")
        .expect("pattern metrics should exist");

    assert!(
        (mc.metrics.f1_score() - 1.0).abs() < 1e-10,
        "minCount (perfect) F1 should be 1.0, got {}",
        mc.metrics.f1_score()
    );
    assert_eq!(
        pat.metrics.f1_score(),
        0.0,
        "pattern (all-wrong) F1 should be 0.0, got {}",
        pat.metrics.f1_score()
    );
}

// ---------------------------------------------------------------------------
// Additional: with_thresholds constructor works correctly
// ---------------------------------------------------------------------------
#[test]
fn test_with_thresholds_constructor() {
    let runner = CertificationRunner::with_thresholds(0.90, 0.85, 0.80);
    assert!((runner.min_f1_threshold - 0.90).abs() < 1e-10);
    assert!((runner.min_precision_threshold - 0.85).abs() < 1e-10);
    assert!((runner.min_recall_threshold - 0.80).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Additional: ConfusionMatrix with zero observations
// ---------------------------------------------------------------------------
#[test]
fn test_confusion_matrix_empty_returns_zero_metrics() {
    let cm = ConfusionMatrix::new(vec!["sh:minCount".to_string(), "sh:pattern".to_string()]);
    let metrics = cm.per_class_metrics();
    for m in &metrics {
        assert_eq!(m.metrics.true_positives, 0);
        assert_eq!(m.metrics.false_positives, 0);
        assert_eq!(m.metrics.true_negatives, 0);
        assert_eq!(m.metrics.false_negatives, 0);
    }
}

// ---------------------------------------------------------------------------
// Additional: add() accumulates correctly
// ---------------------------------------------------------------------------
#[test]
fn test_classification_metrics_add_accumulates() {
    let mut a = ClassificationMetrics {
        true_positives: 3,
        false_positives: 1,
        true_negatives: 4,
        false_negatives: 2,
    };
    let b = ClassificationMetrics {
        true_positives: 7,
        false_positives: 2,
        true_negatives: 6,
        false_negatives: 1,
    };
    a.add(&b);
    assert_eq!(a.true_positives, 10);
    assert_eq!(a.false_positives, 3);
    assert_eq!(a.true_negatives, 10);
    assert_eq!(a.false_negatives, 3);
}

// ---------------------------------------------------------------------------
// Additional: CertificationSuite::add_case works
// ---------------------------------------------------------------------------
#[test]
fn test_certification_suite_add_case() {
    let mut suite = CertificationSuite::new("incremental");
    assert_eq!(suite.cases.len(), 0);

    suite.add_case(case("c1", "sh:minCount", true, true));
    suite.add_case(case("c2", "sh:minCount", false, false));

    assert_eq!(suite.cases.len(), 2);
    assert_eq!(suite.name, "incremental");
}

// ---------------------------------------------------------------------------
// Additional: insufficient check happens before threshold evaluation
// ---------------------------------------------------------------------------
#[test]
fn test_insufficient_returns_before_threshold_check() {
    // 5 perfectly-correct cases — would pass if thresholds were applied,
    // but should be Insufficient because count < 10.
    let cases = perfect_cases(5, "sh:minCount");
    let suite = CertificationSuite::from_cases("tiny-perfect", cases);
    let runner = CertificationRunner::with_thresholds(0.0, 0.0, 0.0);
    let report = runner.run(&suite);

    assert!(
        matches!(report.status, CertificationStatus::Insufficient { .. }),
        "should be Insufficient even with thresholds disabled when < 10 cases"
    );
}
