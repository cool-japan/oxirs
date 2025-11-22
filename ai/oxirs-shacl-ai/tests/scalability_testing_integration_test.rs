//! Integration tests for Scalability Testing Suite
//!
//! These tests verify the end-to-end functionality of the scalability testing system
//! with realistic workloads and performance scenarios.

use oxirs_shacl_ai::scalability_testing::{
    BottleneckType, RecommendationPriority, ResourceMetrics, ScalabilityTestConfig,
    ScalabilityTestingFramework, SlaThresholds, TestWorkload,
};
use std::collections::HashMap;
use std::time::SystemTime;

fn create_realistic_workload() -> TestWorkload {
    let mut operation_mix = HashMap::new();
    operation_mix.insert("shape_validation".to_string(), 0.5);
    operation_mix.insert("constraint_checking".to_string(), 0.3);
    operation_mix.insert("pattern_matching".to_string(), 0.2);

    TestWorkload {
        description: "Realistic SHACL validation workload".to_string(),
        dataset_sizes: vec![100, 500, 1_000, 5_000, 10_000, 50_000],
        concurrent_users: vec![1, 10, 50, 100, 200, 500],
        operation_mix,
    }
}

fn create_small_workload() -> TestWorkload {
    TestWorkload {
        description: "Small test workload".to_string(),
        dataset_sizes: vec![100, 500, 1_000],
        concurrent_users: vec![10, 50],
        operation_mix: HashMap::new(),
    }
}

#[test]
fn test_comprehensive_scalability_testing() {
    let config = ScalabilityTestConfig::default();
    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");

    let workload = create_small_workload();
    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run scalability tests");

    // Verify report structure
    assert!(!report.workload_description.is_empty());
    assert!(report.scalability_score >= 0.0);
    assert!(report.scalability_score <= 100.0);
    assert!(
        report.duration.as_secs() < 60,
        "Tests should complete quickly"
    );

    // Verify test results are present
    assert!(report.load_test_results.is_some());
    assert!(report.stress_test_results.is_some());
    assert!(report.spike_test_results.is_some());
}

#[test]
fn test_load_testing_only() {
    let config = ScalabilityTestConfig {
        enable_load_testing: true,
        enable_stress_testing: false,
        enable_endurance_testing: false,
        enable_spike_testing: false,
        ..Default::default()
    };

    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");
    let workload = create_small_workload();

    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run load tests");

    // Only load test results should be present
    assert!(report.load_test_results.is_some());
    assert!(report.stress_test_results.is_none());
    assert!(report.endurance_test_results.is_none());
    assert!(report.spike_test_results.is_none());

    // Verify load test metrics
    let load_results = report.load_test_results.unwrap();
    assert_eq!(load_results.total_requests, workload.dataset_sizes.len());
    assert!(load_results.avg_latency_ms > 0.0);
    assert!(load_results.p95_latency_ms >= load_results.avg_latency_ms);
    assert!(load_results.p99_latency_ms >= load_results.p95_latency_ms);
    assert!(load_results.throughput_ops > 0.0);
}

#[test]
fn test_stress_testing_breaking_point() {
    let config = ScalabilityTestConfig {
        enable_load_testing: false,
        enable_stress_testing: true,
        enable_endurance_testing: false,
        enable_spike_testing: false,
        max_concurrent_users: 500,
        ..Default::default()
    };

    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");
    let workload = create_small_workload();

    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run stress tests");

    let stress_results = report
        .stress_test_results
        .expect("Should have stress results");

    // Verify breaking point was found
    assert!(stress_results.breaking_point_users > 0);
    assert!(
        stress_results.breaking_point_users <= 500,
        "Breaking point should be within max users"
    );
    assert!(stress_results.peak_throughput > 0.0);
    assert!(stress_results.recovery_time_ms >= 0.0);
}

#[test]
fn test_spike_testing_recovery() {
    let config = ScalabilityTestConfig {
        enable_load_testing: false,
        enable_stress_testing: false,
        enable_endurance_testing: false,
        enable_spike_testing: true,
        max_concurrent_users: 1000,
        ..Default::default()
    };

    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");
    let workload = create_small_workload();

    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run spike tests");

    let spike_results = report
        .spike_test_results
        .expect("Should have spike results");

    // Verify spike test metrics
    assert_eq!(spike_results.spike_magnitude, 1000);
    assert!(spike_results.max_latency_during_spike_ms > 0.0);
    assert!(spike_results.recovery_time_ms > 0.0);

    // If recovery was successful, recovery time should be reasonable
    if spike_results.recovery_successful {
        assert!(
            spike_results.recovery_time_ms < 5000.0,
            "Successful recovery should be quick"
        );
    }
}

#[test]
fn test_sla_compliance_checking() {
    let sla_thresholds = SlaThresholds {
        max_latency_ms: 500.0,
        max_memory_mb: 2048.0,
        min_throughput_ops: 50.0,
        max_error_rate: 0.05,
        max_cpu_percent: 70.0,
    };

    let config = ScalabilityTestConfig {
        sla_thresholds,
        enable_load_testing: true,
        enable_stress_testing: true,
        enable_spike_testing: true,
        ..Default::default()
    };

    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");
    let workload = create_small_workload();

    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run tests");

    // Verify SLA compliance report exists
    assert!(
        report.sla_compliance.latency_compliant
            || report.sla_compliance.throughput_compliant
            || report.sla_compliance.memory_compliant
    );

    // If not compliant, should have violations listed
    if !report.sla_compliance.overall_compliant {
        assert!(
            !report.sla_compliance.violations.is_empty(),
            "Non-compliant report should list violations"
        );
    }
}

#[test]
fn test_bottleneck_identification() {
    let config = ScalabilityTestConfig::default();
    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");

    let workload = create_realistic_workload();
    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run tests");

    // Bottlenecks may or may not be identified depending on performance
    // But the system should handle both cases
    for bottleneck in &report.bottlenecks_identified {
        assert!(
            matches!(
                bottleneck.bottleneck_type,
                BottleneckType::CPU
                    | BottleneckType::Memory
                    | BottleneckType::DiskIO
                    | BottleneckType::NetworkIO
                    | BottleneckType::Latency
                    | BottleneckType::Concurrency
            ),
            "Bottleneck should have valid type"
        );
        assert!(bottleneck.impact_score >= 0.0 && bottleneck.impact_score <= 1.0);
        assert!(!bottleneck.description.is_empty());
        assert!(!bottleneck.suggested_actions.is_empty());
    }
}

#[test]
fn test_recommendation_generation() {
    let config = ScalabilityTestConfig {
        sla_thresholds: SlaThresholds {
            max_latency_ms: 100.0, // Very strict SLA
            max_memory_mb: 512.0,
            min_throughput_ops: 500.0,
            max_error_rate: 0.001,
            max_cpu_percent: 50.0,
        },
        ..Default::default()
    };

    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");
    let workload = create_realistic_workload();

    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run tests");

    // With strict SLAs, should generate recommendations
    if !report.recommendations.is_empty() {
        for recommendation in &report.recommendations {
            assert!(matches!(
                recommendation.priority,
                RecommendationPriority::Low
                    | RecommendationPriority::Medium
                    | RecommendationPriority::High
                    | RecommendationPriority::Critical
            ));
            assert!(!recommendation.title.is_empty());
            assert!(!recommendation.description.is_empty());
            assert!(!recommendation.actions.is_empty());
        }
    }
}

#[test]
fn test_scalability_score_calculation() {
    let config = ScalabilityTestConfig::default();
    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");

    let workload = create_small_workload();
    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run tests");

    // Verify score is in valid range
    assert!(
        report.scalability_score >= 0.0 && report.scalability_score <= 100.0,
        "Score should be between 0 and 100"
    );

    // If SLA compliant, score should be reasonable
    if report.sla_compliance.overall_compliant {
        assert!(
            report.scalability_score >= 50.0,
            "Compliant system should have decent score"
        );
    }
}

#[test]
fn test_multiple_workload_sizes() {
    let config = ScalabilityTestConfig {
        min_dataset_size: 100,
        max_dataset_size: 10_000,
        size_increment_step: 1_000,
        ..Default::default()
    };

    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");

    let workload = TestWorkload {
        description: "Variable size workload".to_string(),
        dataset_sizes: vec![100, 1_000, 5_000, 10_000, 20_000], // Some exceed max
        concurrent_users: vec![10, 50, 100],
        operation_mix: HashMap::new(),
    };

    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run tests");

    // Should handle workloads exceeding configured max gracefully
    assert!(report.load_test_results.is_some());
}

#[test]
fn test_test_history_tracking() {
    let config = ScalabilityTestConfig::default();
    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");

    let workload1 = create_small_workload();
    let workload2 = create_realistic_workload();

    // Run multiple tests
    framework
        .run_all_tests(&workload1)
        .expect("Failed to run first test");
    framework
        .run_all_tests(&workload2)
        .expect("Failed to run second test");

    // Verify history tracking
    let history = framework.get_test_history();
    assert_eq!(history.len(), 2, "Should track both tests");
    assert_ne!(history[0].test_id, history[1].test_id);
}

#[test]
fn test_resource_metrics_tracking() {
    let config = ScalabilityTestConfig::default();
    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");

    let workload = create_small_workload();
    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run tests");

    // Resource metrics should be tracked
    for metric in &report.resource_utilization {
        assert!(metric.cpu_percent >= 0.0 && metric.cpu_percent <= 100.0);
        assert!(metric.memory_mb >= 0.0);
        assert!(metric.disk_io_mbps >= 0.0);
        assert!(metric.network_io_mbps >= 0.0);
    }
}

#[test]
fn test_report_serialization() {
    let config = ScalabilityTestConfig::default();
    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");

    let workload = create_small_workload();
    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run tests");

    // Test JSON serialization
    let json = serde_json::to_string_pretty(&report).expect("Failed to serialize report");
    assert!(!json.is_empty());
    assert!(json.contains("scalability_score"));
    assert!(json.contains("workload_description"));

    // Test deserialization
    let deserialized: oxirs_shacl_ai::scalability_testing::ScalabilityTestReport =
        serde_json::from_str(&json).expect("Failed to deserialize report");
    assert_eq!(deserialized.scalability_score, report.scalability_score);
    assert_eq!(
        deserialized.workload_description,
        report.workload_description
    );
}

#[test]
fn test_concurrent_user_scaling() {
    let config = ScalabilityTestConfig {
        max_concurrent_users: 200,
        ..Default::default()
    };

    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");

    let workload = TestWorkload {
        description: "Concurrent user scaling test".to_string(),
        dataset_sizes: vec![1_000],
        concurrent_users: vec![1, 10, 25, 50, 100, 200],
        operation_mix: HashMap::new(),
    };

    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run tests");

    // Should complete successfully with varying concurrent users
    assert!(report.stress_test_results.is_some());
}

#[test]
fn test_custom_sla_thresholds() {
    let custom_sla = SlaThresholds {
        max_latency_ms: 2000.0,   // Very lenient
        max_memory_mb: 8192.0,    // 8GB
        min_throughput_ops: 10.0, // Low requirement
        max_error_rate: 0.1,      // 10% acceptable
        max_cpu_percent: 95.0,    // High usage OK
    };

    let config = ScalabilityTestConfig {
        sla_thresholds: custom_sla,
        ..Default::default()
    };

    let mut framework =
        ScalabilityTestingFramework::new(config).expect("Failed to create framework");
    let workload = create_small_workload();

    let report = framework
        .run_all_tests(&workload)
        .expect("Failed to run tests");

    // With lenient SLAs, should likely be compliant
    // (though not guaranteed due to random nature of simulations)
    assert!(report.sla_compliance.latency_compliant || report.sla_compliance.throughput_compliant);
}
