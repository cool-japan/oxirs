//! Regression tests for quality issue detection (kept in a sibling module
//! to keep `issue_detection.rs` under the 2000-line limit).

use super::*;

use oxirs_core::model::{Literal, NamedNode, Quad};
use oxirs_core::ConcreteStore;

fn stable_snapshot(score: f64) -> QualitySnapshot {
    let mut dims = HashMap::new();
    dims.insert("completeness".to_string(), score);
    dims.insert("consistency".to_string(), score);
    dims.insert("accuracy".to_string(), score);
    dims.insert("conformance".to_string(), score);
    QualitySnapshot {
        timestamp: chrono::Utc::now(),
        overall_quality_score: score,
        quality_dimensions: dims,
        performance_metrics: PerformanceSnapshot {
            validation_time_ms: 0.0,
            memory_usage_mb: 0.0,
            throughput_ops_per_sec: 0.0,
            error_rate: 0.05,
            resource_utilization: 0.0,
        },
        data_characteristics: DataCharacteristics {
            total_triples: 100,
            unique_subjects: 20,
            unique_predicates: 5,
            unique_objects: 50,
            schema_complexity: 0.2,
            data_density: 0.5,
        },
        validation_results: ValidationSnapshot {
            total_validations: 5,
            successful_validations: 5,
            violation_count: 0,
            average_violation_severity: 0.0,
            validation_coverage: 1.0,
        },
    }
}

fn report(overall: f64) -> QualityReport {
    let mut r = QualityReport::new();
    r.overall_score = overall;
    r.completeness_score = overall;
    r.consistency_score = overall;
    r.accuracy_score = overall;
    r.conformance_score = overall;
    r
}

/// Regression: the quality snapshot must derive `data_characteristics` from
/// the real store, not from hardcoded constants (10000 triples, etc.).
#[test]
fn regression_snapshot_derives_data_characteristics_from_store() {
    let store = ConcreteStore::new().expect("store");
    let pred = NamedNode::new("http://example.org/name").expect("iri");
    for i in 0..4 {
        let subj = NamedNode::new(format!("http://example.org/s{i}")).expect("iri");
        store
            .insert_quad(Quad::new_default_graph(
                subj,
                pred.clone(),
                Literal::new(format!("value{i}")),
            ))
            .expect("insert");
    }

    let mut detector = QualityIssueDetector::new();
    detector
        .detect_quality_issues(&store, &[], &report(0.9), None)
        .expect("detection");

    let snap = detector.historical_data.last().expect("snapshot");
    assert_eq!(snap.data_characteristics.total_triples, 4);
    assert_eq!(snap.data_characteristics.unique_subjects, 4);
    assert_eq!(snap.data_characteristics.unique_predicates, 1);
}

/// Regression: after a stable history, a sharp quality drop must be flagged
/// as an anomaly and drive proactive alerts (the old code always reported
/// "stable, no anomalies").
#[test]
fn regression_anomaly_detection_flags_quality_drop() {
    let store = ConcreteStore::new().expect("store");
    let mut detector = QualityIssueDetector::new();
    for _ in 0..14 {
        detector.add_historical_data(stable_snapshot(0.9));
    }

    let result = detector
        .detect_quality_issues(&store, &[], &report(0.2), None)
        .expect("detection");

    assert!(
        result.anomaly_detection.overall_anomaly_score > 0.0,
        "a large quality drop must produce a non-zero anomaly score"
    );
    assert!(
        !result.anomaly_detection.statistical_anomalies.is_empty(),
        "the overall-score drop must be flagged as a statistical anomaly"
    );
    assert!(
        !result.proactive_alerts.is_empty(),
        "a real regression must raise at least one proactive alert"
    );
}
