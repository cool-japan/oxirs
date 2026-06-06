//! Unit tests for the advanced diagnostics engine.

#![cfg(test)]

use crate::advanced_diagnostics_types::{AdvancedDiagnosticEngine, HealthTrend};
use crate::storage::BufferPoolStats;
use std::time::Duration;

#[test]
fn test_advanced_diagnostic_engine_creation() {
    let engine = AdvancedDiagnosticEngine::new();
    assert_eq!(engine.historical_buffer.len(), 0);
    assert_eq!(engine.query_tracker.recent_queries.len(), 0);
    assert_eq!(engine.transaction_tracker.commits, 0);
}

#[test]
fn test_record_snapshot() {
    let mut engine = AdvancedDiagnosticEngine::new();
    let stats = BufferPoolStats::default();

    engine.record_snapshot(0.1, 100.0, 1_000_000, 0, &stats);
    assert_eq!(engine.historical_buffer.len(), 1);

    // Add more than max_history_size snapshots
    for _ in 0..30 {
        engine.record_snapshot(0.1, 100.0, 1_000_000, 0, &stats);
    }

    assert!(engine.historical_buffer.len() <= engine.max_history_size);
}

#[test]
fn test_record_query() {
    let mut engine = AdvancedDiagnosticEngine::new();

    engine.record_query(
        Duration::from_millis(100),
        "SELECT_PATTERN".to_string(),
        vec!["SPO".to_string()],
    );

    assert_eq!(engine.query_tracker.recent_queries.len(), 1);
    assert_eq!(engine.query_tracker.patterns.len(), 1);
    assert_eq!(
        engine
            .query_tracker
            .patterns
            .get("SELECT_PATTERN")
            .unwrap()
            .frequency,
        1
    );
}

#[test]
fn test_record_cache_hit_miss() {
    let mut engine = AdvancedDiagnosticEngine::new();

    engine.record_cache_hit(true);
    engine.record_cache_hit(true);
    engine.record_cache_hit(false);

    assert_eq!(engine.query_tracker.cache_hits, 2);
    assert_eq!(engine.query_tracker.cache_misses, 1);
}

#[test]
fn test_record_transaction() {
    let mut engine = AdvancedDiagnosticEngine::new();

    engine.record_transaction(Duration::from_millis(50), true);
    engine.record_transaction(Duration::from_millis(75), false);

    assert_eq!(engine.transaction_tracker.commits, 1);
    assert_eq!(engine.transaction_tracker.aborts, 1);
    assert_eq!(engine.transaction_tracker.recent_durations.len(), 2);
}

#[test]
fn test_record_conflict_and_deadlock() {
    let mut engine = AdvancedDiagnosticEngine::new();

    engine.record_conflict();
    engine.record_conflict();
    engine.record_deadlock();

    assert_eq!(engine.transaction_tracker.conflicts, 2);
    assert_eq!(engine.transaction_tracker.deadlocks, 1);
}

#[test]
fn test_record_contention() {
    let mut engine = AdvancedDiagnosticEngine::new();

    engine.record_contention("SPO_INDEX".to_string(), Duration::from_millis(10));
    engine.record_contention("SPO_INDEX".to_string(), Duration::from_millis(20));

    let stats = engine
        .transaction_tracker
        .contention_map
        .get("SPO_INDEX")
        .unwrap();
    assert_eq!(stats.count, 2);
    assert_eq!(stats.total_wait_time, Duration::from_millis(30));
}

#[test]
fn test_analyze_query_performance_empty() {
    let engine = AdvancedDiagnosticEngine::new();
    let analysis = engine.analyze_query_performance().unwrap();

    assert_eq!(analysis.total_queries, 0);
    assert_eq!(analysis.avg_execution_time, Duration::ZERO);
    assert_eq!(analysis.cache_hit_rate, 0.0);
}

#[test]
fn test_analyze_query_performance_with_data() {
    let mut engine = AdvancedDiagnosticEngine::new();

    // Record some queries
    for i in 0..10 {
        engine.record_query(
            Duration::from_millis(100 + i * 10),
            "PATTERN_A".to_string(),
            vec!["SPO".to_string()],
        );
    }

    engine.record_cache_hit(true);
    engine.record_cache_hit(true);
    engine.record_cache_hit(false);

    let analysis = engine.analyze_query_performance().unwrap();

    assert_eq!(analysis.total_queries, 10);
    assert!(analysis.avg_execution_time.as_millis() > 0);
    assert!((analysis.cache_hit_rate - 0.666).abs() < 0.01);
    assert_eq!(analysis.query_patterns.len(), 1);
}

#[test]
fn test_analyze_transaction_patterns_empty() {
    let engine = AdvancedDiagnosticEngine::new();
    let analysis = engine.analyze_transaction_patterns().unwrap();

    assert_eq!(analysis.total_transactions, 0);
    assert_eq!(analysis.commit_rate, 0.0);
    assert_eq!(analysis.abort_rate, 0.0);
}

#[test]
fn test_analyze_transaction_patterns_with_data() {
    let mut engine = AdvancedDiagnosticEngine::new();

    // Record transactions
    engine.record_transaction(Duration::from_millis(50), true);
    engine.record_transaction(Duration::from_millis(75), true);
    engine.record_transaction(Duration::from_millis(100), false);

    engine.record_conflict();
    engine.record_contention("SPO_INDEX".to_string(), Duration::from_millis(25));

    let analysis = engine.analyze_transaction_patterns().unwrap();

    assert_eq!(analysis.total_transactions, 3);
    assert!((analysis.commit_rate - 0.666).abs() < 0.01);
    assert!((analysis.abort_rate - 0.333).abs() < 0.01);
    assert!((analysis.conflict_rate - 0.333).abs() < 0.01);
    assert_eq!(analysis.contention_points.len(), 1);
}

#[test]
fn test_health_trend_determination() {
    let engine = AdvancedDiagnosticEngine::new();

    // Stable trend
    let latency = vec![0.1, 0.11, 0.1, 0.12, 0.1];
    let errors = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    assert_eq!(
        engine.determine_health_trend(&latency, &errors),
        HealthTrend::Stable
    );

    // Improving trend
    let latency = vec![0.5, 0.4, 0.3, 0.2, 0.1];
    let errors = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    assert_eq!(
        engine.determine_health_trend(&latency, &errors),
        HealthTrend::Improving
    );

    // Degrading rapidly
    let latency = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let errors = vec![1.0, 2.0, 4.0, 8.0, 10.0];
    assert_eq!(
        engine.determine_health_trend(&latency, &errors),
        HealthTrend::DegradingRapidly
    );
}

#[test]
fn test_generate_report_empty() {
    let engine = AdvancedDiagnosticEngine::new();
    let report = engine.generate_report(1_000_000, 10_000_000).unwrap();

    assert_eq!(report.query_analysis.total_queries, 0);
    assert_eq!(report.transaction_analysis.total_transactions, 0);
    assert_eq!(report.predictive_health.historical_metrics.data_points, 0);
}

#[test]
fn test_generate_report_with_data() {
    let mut engine = AdvancedDiagnosticEngine::new();
    let stats = BufferPoolStats::default();

    // Add some data
    engine.record_snapshot(0.1, 100.0, 1_000_000, 0, &stats);
    engine.record_query(
        Duration::from_millis(100),
        "PATTERN_A".to_string(),
        vec!["SPO".to_string()],
    );
    engine.record_transaction(Duration::from_millis(50), true);

    let report = engine.generate_report(1_000_000, 10_000_000).unwrap();

    assert_eq!(report.query_analysis.total_queries, 1);
    assert_eq!(report.transaction_analysis.total_transactions, 1);
    assert_eq!(report.predictive_health.historical_metrics.data_points, 1);
}

#[test]
fn test_tuning_recommendations() {
    let mut engine = AdvancedDiagnosticEngine::new();

    // Low cache hit rate
    engine.record_cache_hit(false);
    engine.record_cache_hit(false);
    engine.record_cache_hit(false);
    engine.record_cache_hit(true);

    let recommendations = engine
        .generate_tuning_recommendations(1_000_000, 10_000_000)
        .unwrap();

    assert!(!recommendations.is_empty());
    assert!(recommendations
        .iter()
        .any(|r| r.parameter == "query_cache_size"));
}

#[test]
fn test_anomaly_detection() {
    let mut engine = AdvancedDiagnosticEngine::new();
    let stats = BufferPoolStats::default();

    // Normal values with tight range
    for _ in 0..10 {
        engine.record_snapshot(0.1, 100.0, 1_000_000, 0, &stats);
    }

    // Anomalous value - much higher to ensure detection
    engine.record_snapshot(50.0, 100.0, 1_000_000, 0, &stats); // Very high latency (50x normal)

    let anomalies = engine.detect_anomalies().unwrap();

    assert!(!anomalies.is_empty());
    assert!(anomalies.iter().any(|a| a.metric == "query_latency"));
}

#[test]
fn test_capacity_forecast() {
    let mut engine = AdvancedDiagnosticEngine::new();
    let stats = BufferPoolStats::default();

    // Simulate growth - need measurable time difference for rate calculation
    engine.record_snapshot(0.1, 100.0, 1_000_000, 0, &stats);
    std::thread::sleep(std::time::Duration::from_millis(10));
    engine.record_snapshot(0.1, 100.0, 1_500_000, 0, &stats);

    let forecast = engine.forecast_capacity(1_500_000, 10_000_000).unwrap();

    assert_eq!(forecast.current_storage_bytes, 1_500_000);
    assert!(forecast.growth_rate_per_day > 0.0);
}

#[test]
fn test_fragmentation_analysis() {
    let engine = AdvancedDiagnosticEngine::new();
    let analysis = engine.analyze_fragmentation(10_000_000).unwrap();

    assert!(analysis.overall_fragmentation_pct >= 0.0);
    assert!(analysis.compaction_priority >= 0.0 && analysis.compaction_priority <= 1.0);
    assert!(!analysis.index_fragmentation.is_empty());
}

#[test]
fn test_index_usage_analysis() {
    let engine = AdvancedDiagnosticEngine::new();
    let stats = engine.analyze_index_usage().unwrap();

    assert!(stats.total_scans > 0);
    assert!(!stats.usage_by_index.is_empty());
    assert!(stats.usage_by_index.contains_key("SPO"));
    assert!(stats.usage_by_index.contains_key("POS"));
    assert!(stats.usage_by_index.contains_key("OSP"));
}

#[test]
fn test_predictive_health_empty() {
    let engine = AdvancedDiagnosticEngine::new();
    let health = engine.predict_health_issues().unwrap();

    assert_eq!(health.historical_metrics.data_points, 0);
    assert_eq!(health.predicted_issues_24h.len(), 0);
    assert_eq!(health.health_trend, HealthTrend::Stable);
}

#[test]
fn test_predictive_health_with_trends() {
    let mut engine = AdvancedDiagnosticEngine::new();
    let stats = BufferPoolStats::default();

    // Add snapshots with increasing latency (degrading)
    for i in 0..10 {
        engine.record_snapshot(
            0.1 * (i as f64 + 1.0),
            100.0,
            1_000_000 + i * 100_000,
            i,
            &stats,
        );
    }

    let health = engine.predict_health_issues().unwrap();

    assert_eq!(health.historical_metrics.data_points, 10);
    assert!(!health.predicted_issues_24h.is_empty() || health.health_trend != HealthTrend::Stable);
}
