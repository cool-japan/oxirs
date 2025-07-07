//! Integration tests for W3C SHACL test suite functionality

use oxirs_shacl::w3c_test_suite::{TestCategory, W3cTestConfig, W3cTestSuiteRunner};
use std::collections::HashSet;

#[tokio::test]
async fn test_w3c_test_suite_runner_basic_functionality() {
    // Create test configuration
    let mut enabled_categories = HashSet::new();
    enabled_categories.insert(TestCategory::Core);

    let config = W3cTestConfig {
        test_suite_location: "https://example.org/test-data/w3c-shacl-test-suite/".to_string(),
        enabled_categories,
        test_timeout_seconds: 10,
        max_parallel_tests: 1,
        verbose_logging: true,
        output_directory: None,
        test_filters: Vec::new(),
    };

    // Create test runner
    let mut runner = W3cTestSuiteRunner::new(config).unwrap();

    // Load test manifests
    runner.load_manifests().await.unwrap();

    // Verify manifests were loaded
    assert!(!runner.manifests.is_empty());

    // Check that we have core constraint tests
    let core_manifest = runner
        .manifests
        .iter()
        .find(|m| m.category == TestCategory::Core);
    assert!(core_manifest.is_some());

    let core_manifest = core_manifest.unwrap();
    assert_eq!(core_manifest.id, "core-constraints");
    assert!(!core_manifest.entries.is_empty());
}

#[tokio::test]
async fn test_w3c_test_suite_execution() {
    let config = W3cTestConfig::default();
    let mut runner = W3cTestSuiteRunner::new(config).unwrap();

    // Load manifests
    runner.load_manifests().await.unwrap();

    // Execute all tests
    let stats = runner.execute_all_tests().await.unwrap();

    // Verify statistics
    assert!(stats.total_tests > 0);
    // total_execution_time_ms is always >= 0 by type invariant

    // At least some tests should be executed or skipped
    assert!(stats.tests_passed + stats.tests_failed + stats.tests_skipped + stats.tests_error > 0);

    println!("W3C Test Suite Results:");
    println!("  Total tests: {}", stats.total_tests);
    println!("  Passed: {}", stats.tests_passed);
    println!("  Failed: {}", stats.tests_failed);
    println!("  Skipped: {}", stats.tests_skipped);
    println!("  Error: {}", stats.tests_error);
    println!("  Compliance: {:.1}%", stats.compliance_percentage);
}

#[tokio::test]
async fn test_compliance_report_generation() {
    let config = W3cTestConfig::default();
    let mut runner = W3cTestSuiteRunner::new(config).unwrap();

    // Load manifests and execute tests
    runner.load_manifests().await.unwrap();
    runner.execute_all_tests().await.unwrap();

    // Generate compliance report
    let report = runner.generate_compliance_report();

    // Verify report structure
    assert!(!report.test_suite_version.is_empty());
    assert!(!report.implementation_details.version.is_empty());
    assert!(!report.implementation_details.features.is_empty());

    // Report should have timestamp
    assert!(report.generated_at.timestamp() > 0);

    // Should have summary stats
    // total_tests is always >= 0 by type invariant
}

#[test]
fn test_test_configuration() {
    let config = W3cTestConfig::default();

    // Verify default configuration
    assert!(config.enabled_categories.contains(&TestCategory::Core));
    assert!(config
        .enabled_categories
        .contains(&TestCategory::PropertyPaths));
    assert_eq!(config.test_timeout_seconds, 30);
    assert_eq!(config.max_parallel_tests, 4);
    assert!(!config.verbose_logging);
    assert!(config.output_directory.is_none());
}

#[test]
fn test_test_filtering() {
    use oxirs_shacl::w3c_test_suite::TestFilter;

    let mut config = W3cTestConfig::default();

    // Add test filters
    config.test_filters.push(TestFilter {
        name: "core_only".to_string(),
        test_pattern: "core-".to_string(),
        include: true,
    });

    config.test_filters.push(TestFilter {
        name: "exclude_sparql".to_string(),
        test_pattern: "sparql-".to_string(),
        include: false,
    });

    assert_eq!(config.test_filters.len(), 2);
}

#[tokio::test]
async fn test_manifest_categories() {
    let config = W3cTestConfig::default();
    let mut runner = W3cTestSuiteRunner::new(config).unwrap();

    runner.load_manifests().await.unwrap();

    // Verify we have manifests for different categories
    let categories: HashSet<_> = runner
        .manifests
        .iter()
        .map(|m| m.category.clone())
        .collect();

    assert!(categories.contains(&TestCategory::Core));
    assert!(categories.contains(&TestCategory::PropertyPaths));
    assert!(categories.contains(&TestCategory::LogicalConstraints));
}

#[tokio::test]
async fn test_individual_test_types() {
    use oxirs_shacl::w3c_test_suite::TestType;

    let config = W3cTestConfig::default();
    let mut runner = W3cTestSuiteRunner::new(config).unwrap();

    runner.load_manifests().await.unwrap();

    // Check that we have different test types
    let mut has_validation_tests = false;
    let mut has_core_tests = false;

    for manifest in &runner.manifests {
        for entry in &manifest.entries {
            if entry.test_type == TestType::Validation {
                has_validation_tests = true;
            }
            if entry.id.starts_with("core-") {
                has_core_tests = true;
            }
        }
    }

    assert!(has_validation_tests);
    assert!(has_core_tests);
}

#[tokio::test]
async fn test_expected_results_structure() {
    let config = W3cTestConfig::default();
    let mut runner = W3cTestSuiteRunner::new(config).unwrap();

    runner.load_manifests().await.unwrap();

    // Verify expected results have proper structure
    for manifest in &runner.manifests {
        for entry in &manifest.entries {
            // All test entries should have expected results
            let expected = &entry.expected_result;

            // If violation count is specified, it should be reasonable
            if let Some(count) = expected.violation_count {
                assert!(count < 1000); // Sanity check
            }
        }
    }
}

#[test]
fn test_compliance_assessment_logic() {
    use oxirs_shacl::{w3c_test_suite::*, ValidationReport};

    // Create test entry
    let test_entry = TestEntry {
        id: "test-compliance".to_string(),
        label: "Test compliance assessment".to_string(),
        description: None,
        test_type: TestType::Validation,
        data_graph: None,
        shapes_graph: None,
        expected_result: ExpectedResult {
            conforms: true,
            violation_count: Some(0),
            expected_violations: Vec::new(),
            expected_error: None,
        },
        metadata: std::collections::HashMap::new(),
    };

    // Create matching validation report
    let mut validation_report = ValidationReport::new();
    validation_report.conforms = true;

    // Test compliance assessment
    let config = W3cTestConfig::default();
    let runner = W3cTestSuiteRunner::new(config).unwrap();

    let assessment = runner.assess_compliance(&test_entry, &validation_report);

    assert!(assessment.compliant);
    assert_eq!(assessment.score, 1.0);
    assert!(assessment.issues.is_empty());
}
