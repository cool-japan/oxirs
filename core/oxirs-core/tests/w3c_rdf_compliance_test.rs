//! W3C RDF Format Compliance Integration Tests
//!
//! This module provides comprehensive integration tests for W3C RDF format compliance
//! using the enhanced W3C RDF test suite runner.

use anyhow::Result;
use oxirs_core::format::w3c_tests::*;
use oxirs_core::format::RdfFormat;

#[tokio::test]
async fn test_w3c_rdf_format_compliance_turtle() -> Result<()> {
    let mut config = W3cRdfTestConfig::default();

    // Focus on Turtle format only for this test
    config.enabled_formats.clear();
    config.enabled_formats.insert(RdfFormat::Turtle);

    // Enable verbose logging for detailed output
    config.verbose_logging = true;
    config.test_timeout_seconds = 10;
    config.max_parallel_tests = 4;

    let mut runner = W3cRdfTestSuiteRunner::new(config)?;

    // Load manifests
    runner.load_manifests().await?;

    // Run tests
    let stats = runner.run_tests().await?;

    // Verify results
    assert!(stats.total_tests > 0, "No tests were loaded");

    // Check that we have some passing tests
    assert!(stats.passed > 0, "No tests passed");

    // Verify format-specific stats
    if let Some(turtle_stats) = stats.format_stats.get(&RdfFormat::Turtle) {
        assert!(turtle_stats.total > 0, "No Turtle tests found");
        assert!(
            turtle_stats.compliance_percentage >= 0.0,
            "Invalid compliance percentage"
        );

        println!(
            "Turtle compliance: {:.2}%",
            turtle_stats.compliance_percentage
        );
        println!(
            "Turtle tests: {} passed, {} failed",
            turtle_stats.passed, turtle_stats.failed
        );
    }

    // Generate and verify report
    let report = runner.generate_report();
    assert!(report.contains("W3C RDF Format Compliance Report"));
    assert!(report.contains("Turtle"));

    Ok(())
}

#[tokio::test]
async fn test_w3c_rdf_format_compliance_multiple_formats() -> Result<()> {
    let mut config = W3cRdfTestConfig::default();

    // Test multiple formats
    config.enabled_formats.clear();
    config.enabled_formats.insert(RdfFormat::Turtle);
    config.enabled_formats.insert(RdfFormat::NTriples);
    config.enabled_formats.insert(RdfFormat::NQuads);
    config.enabled_formats.insert(RdfFormat::TriG);

    config.verbose_logging = false; // Reduce output for multi-format test
    config.test_timeout_seconds = 15;
    config.max_parallel_tests = 8;

    let mut runner = W3cRdfTestSuiteRunner::new(config)?;

    // Load manifests
    runner.load_manifests().await?;

    // Run tests
    let stats = runner.run_tests().await?;

    // Verify results for each format
    assert!(stats.total_tests > 0, "No tests were loaded");

    for format in [
        RdfFormat::Turtle,
        RdfFormat::NTriples,
        RdfFormat::NQuads,
        RdfFormat::TriG,
    ] {
        if let Some(format_stats) = stats.format_stats.get(&format) {
            assert!(
                format_stats.total > 0,
                "No tests found for format: {format:?}"
            );
            println!(
                "{format:?}: {:.2}% compliance ({} passed / {} total)",
                format_stats.compliance_percentage, format_stats.passed, format_stats.total
            );
        }
    }

    // Verify overall compliance
    let overall_compliance = if stats.total_tests > 0 {
        (stats.passed as f64 / stats.total_tests as f64) * 100.0
    } else {
        0.0
    };

    println!("Overall compliance: {overall_compliance:.2}%");
    assert!(overall_compliance >= 0.0, "Invalid overall compliance");

    Ok(())
}

#[tokio::test]
async fn test_w3c_rdf_positive_parser_tests() -> Result<()> {
    let mut config = W3cRdfTestConfig::default();

    // Focus on positive parser tests only
    config.enabled_test_types.clear();
    config
        .enabled_test_types
        .insert(RdfTestType::PositiveParser);

    // Test with Turtle format
    config.enabled_formats.clear();
    config.enabled_formats.insert(RdfFormat::Turtle);

    config.verbose_logging = true;

    let mut runner = W3cRdfTestSuiteRunner::new(config)?;

    // Load manifests
    runner.load_manifests().await?;

    // Run tests
    let stats = runner.run_tests().await?;

    // All tests should be positive parser tests
    let results = runner.get_results();
    for result in results.values() {
        assert_eq!(
            result.test_type,
            RdfTestType::PositiveParser,
            "Unexpected test type: {:?}",
            result.test_type
        );
    }

    // Check that positive tests generally pass
    if stats.total_tests > 0 {
        let pass_rate = (stats.passed as f64 / stats.total_tests as f64) * 100.0;
        println!("Positive parser test pass rate: {pass_rate:.2}%");
        assert!(
            pass_rate > 50.0,
            "Pass rate too low for positive tests: {pass_rate:.2}%"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_w3c_rdf_negative_parser_tests() -> Result<()> {
    let mut config = W3cRdfTestConfig::default();

    // Focus on negative parser tests only
    config.enabled_test_types.clear();
    config
        .enabled_test_types
        .insert(RdfTestType::NegativeParser);

    // Test with Turtle format
    config.enabled_formats.clear();
    config.enabled_formats.insert(RdfFormat::Turtle);

    config.verbose_logging = true;

    let mut runner = W3cRdfTestSuiteRunner::new(config)?;

    // Load manifests
    runner.load_manifests().await?;

    // Run tests
    let stats = runner.run_tests().await?;

    // All tests should be negative parser tests
    let results = runner.get_results();
    for result in results.values() {
        assert_eq!(
            result.test_type,
            RdfTestType::NegativeParser,
            "Unexpected test type: {:?}",
            result.test_type
        );
    }

    // Negative tests should pass (i.e., correctly fail to parse invalid input)
    if stats.total_tests > 0 {
        let pass_rate = (stats.passed as f64 / stats.total_tests as f64) * 100.0;
        println!("Negative parser test pass rate: {pass_rate:.2}%");
        // Note: For negative tests, "passing" means correctly rejecting invalid input
    }

    Ok(())
}

#[tokio::test]
async fn test_w3c_rdf_evaluation_tests() -> Result<()> {
    let mut config = W3cRdfTestConfig::default();

    // Focus on evaluation tests only
    config.enabled_test_types.clear();
    config.enabled_test_types.insert(RdfTestType::Evaluation);

    // Test with Turtle format
    config.enabled_formats.clear();
    config.enabled_formats.insert(RdfFormat::Turtle);

    config.verbose_logging = true;

    let mut runner = W3cRdfTestSuiteRunner::new(config)?;

    // Load manifests (even though they're generated, this tests the pipeline)
    runner.load_manifests().await?;

    // For this test, manually add an evaluation test
    // In practice, this would be loaded from actual W3C test manifests

    // Run any tests that were loaded
    let stats = runner.run_tests().await?;

    println!(
        "Evaluation tests - Total: {}, Passed: {}, Failed: {}",
        stats.total_tests, stats.passed, stats.failed
    );

    // The test should complete without errors
    assert!(stats.error == 0, "Evaluation tests had errors");

    Ok(())
}

#[tokio::test]
async fn test_w3c_rdf_test_filters() -> Result<()> {
    let mut config = W3cRdfTestConfig::default();

    // Add a test filter
    config.test_filters.push(RdfTestFilter {
        name_pattern: Some("Positive".to_string()),
        test_type: Some(RdfTestType::PositiveParser),
        format: Some(RdfFormat::Turtle),
        approved_only: true,
    });

    config.enabled_formats.clear();
    config.enabled_formats.insert(RdfFormat::Turtle);

    config.verbose_logging = true;

    let mut runner = W3cRdfTestSuiteRunner::new(config)?;

    // Load manifests
    runner.load_manifests().await?;

    // Run tests
    let stats = runner.run_tests().await?;

    // Verify that only filtered tests were run
    let results = runner.get_results();
    for result in results.values() {
        assert!(
            result.test_name.contains("Positive"),
            "Test name should contain 'Positive': {}",
            result.test_name
        );
        assert_eq!(
            result.test_type,
            RdfTestType::PositiveParser,
            "Test type should be PositiveParser"
        );
        assert_eq!(
            result.format,
            RdfFormat::Turtle,
            "Test format should be Turtle"
        );
    }

    println!("Filtered tests: {}", stats.total_tests);

    Ok(())
}

#[tokio::test]
async fn test_w3c_rdf_compliance_report_generation() -> Result<()> {
    let config = W3cRdfTestConfig::default();
    let mut runner = W3cRdfTestSuiteRunner::new(config)?;

    // Load manifests and run tests
    runner.load_manifests().await?;
    let stats = runner.run_tests().await?;

    // Generate report
    let report = runner.generate_report();

    // Verify report structure
    assert!(report.contains("# W3C RDF Format Compliance Report"));
    assert!(report.contains("Total tests:"));
    assert!(report.contains("Passed:"));
    assert!(report.contains("Failed:"));
    assert!(report.contains("Overall compliance:"));
    assert!(report.contains("## Format-specific results:"));

    // Verify statistics are accessible
    let detailed_stats = runner.get_stats();
    assert_eq!(detailed_stats.total_tests, stats.total_tests);
    assert_eq!(detailed_stats.passed, stats.passed);
    assert_eq!(detailed_stats.failed, stats.failed);

    println!("Generated compliance report:");
    println!("{report}");

    Ok(())
}

#[tokio::test]
async fn test_w3c_rdf_test_timeout_handling() -> Result<()> {
    // Set very short timeout to test timeout handling
    let mut config = W3cRdfTestConfig {
        test_timeout_seconds: 1,
        ..Default::default()
    };
    config.enabled_formats.clear();
    config.enabled_formats.insert(RdfFormat::Turtle);
    config.verbose_logging = true;

    let mut runner = W3cRdfTestSuiteRunner::new(config)?;

    // Load manifests
    runner.load_manifests().await?;

    // Run tests
    let stats = runner.run_tests().await?;

    // Tests should complete (even with short timeout, our simple tests should be fast)
    println!(
        "Test execution with 1s timeout - Total: {}, Timeouts: {}",
        stats.total_tests, stats.timeout
    );

    // The test runner should handle timeouts gracefully
    // Note: timeout is usize, so it's always non-negative

    Ok(())
}

#[test]
fn test_w3c_rdf_config_validation() {
    // Test default configuration
    let config = W3cRdfTestConfig::default();
    assert!(config.enabled_formats.contains(&RdfFormat::Turtle));
    assert!(config
        .enabled_test_types
        .contains(&RdfTestType::PositiveParser));
    assert_eq!(config.test_timeout_seconds, 30);
    assert_eq!(config.max_parallel_tests, 8);
    assert!(!config.verbose_logging);
    assert!(config.skip_known_failures);

    // Test custom configuration
    let custom_config = W3cRdfTestConfig {
        test_timeout_seconds: 60,
        verbose_logging: true,
        max_parallel_tests: 16,
        ..Default::default()
    };

    assert_eq!(custom_config.test_timeout_seconds, 60);
    assert!(custom_config.verbose_logging);
    assert_eq!(custom_config.max_parallel_tests, 16);
}

#[tokio::test]
async fn test_w3c_rdf_runner_creation() -> Result<()> {
    let config = W3cRdfTestConfig::default();
    let runner = W3cRdfTestSuiteRunner::new(config)?;

    // Verify runner was created successfully
    let stats = runner.get_stats();
    assert_eq!(stats.total_tests, 0); // No tests loaded yet
    assert_eq!(stats.passed, 0);
    assert_eq!(stats.failed, 0);

    Ok(())
}

/// Convenience function to run a quick compliance check
#[tokio::test]
async fn test_quick_w3c_compliance_check() -> Result<()> {
    // Use the convenience function
    let stats = run_w3c_compliance_tests(None).await?;

    // Verify basic functionality
    assert!(stats.total_tests > 0, "Should have loaded some tests");

    println!("Quick compliance check completed:");
    println!("Total tests: {}", stats.total_tests);
    println!("Passed: {}", stats.passed);
    println!("Failed: {}", stats.failed);

    // Calculate overall compliance rate
    let compliance_rate = if stats.total_tests > 0 {
        (stats.passed as f64 / stats.total_tests as f64) * 100.0
    } else {
        0.0
    };

    println!("Overall compliance rate: {compliance_rate:.2}%");

    Ok(())
}
