//! SHACL Shape Testing Framework
//!
//! This module provides a comprehensive testing framework for SHACL shapes,
//! allowing developers to write and run test cases for their shape definitions.
//!
//! # Features
//!
//! - Test case definitions with expected outcomes
//! - Positive and negative test cases
//! - Test suite organization
//! - Assertion helpers for common patterns
//! - Test report generation
//! - Integration with CI/CD pipelines
//!
//! # Example
//!
//! ```ignore
//! use oxirs_shacl::testing::{ShapeTestSuite, TestCase, TestResult};
//!
//! let mut suite = ShapeTestSuite::new("Person Shape Tests");
//!
//! suite.add_test(
//!     TestCase::new("valid_person")
//!         .with_data("ex:john a ex:Person ; ex:name 'John' .")
//!         .expect_valid()
//! );
//!
//! suite.add_test(
//!     TestCase::new("missing_name")
//!         .with_data("ex:jane a ex:Person .")
//!         .expect_violation("sh:minCount")
//! );
//!
//! let results = suite.run(&shapes, &store)?;
//! ```

use crate::{Shape, ShapeId, ValidationConfig, ValidationReport, Validator};
use indexmap::IndexMap;
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Test case for SHACL shape validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    /// Test case name
    pub name: String,
    /// Test description
    pub description: Option<String>,
    /// RDF data to validate (in Turtle format)
    pub data: String,
    /// Expected outcome
    pub expected: TestExpectation,
    /// Tags for filtering
    pub tags: Vec<String>,
    /// Focus nodes to validate (optional)
    pub focus_nodes: Vec<String>,
    /// Timeout for this test
    pub timeout: Option<Duration>,
    /// Whether this test should be skipped
    pub skip: bool,
    /// Skip reason
    pub skip_reason: Option<String>,
}

impl TestCase {
    /// Create a new test case
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            data: String::new(),
            expected: TestExpectation::Valid,
            tags: Vec::new(),
            focus_nodes: Vec::new(),
            timeout: None,
            skip: false,
            skip_reason: None,
        }
    }

    /// Set the test description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// Set the RDF data to validate
    pub fn with_data(mut self, data: &str) -> Self {
        self.data = data.to_string();
        self
    }

    /// Add tags to the test
    pub fn with_tags(mut self, tags: &[&str]) -> Self {
        self.tags = tags.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set specific focus nodes to validate
    pub fn with_focus_nodes(mut self, nodes: &[&str]) -> Self {
        self.focus_nodes = nodes.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set timeout for this test
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Skip this test
    pub fn skip(mut self, reason: &str) -> Self {
        self.skip = true;
        self.skip_reason = Some(reason.to_string());
        self
    }

    /// Expect the data to be valid
    pub fn expect_valid(mut self) -> Self {
        self.expected = TestExpectation::Valid;
        self
    }

    /// Expect the data to be invalid
    pub fn expect_invalid(mut self) -> Self {
        self.expected = TestExpectation::Invalid { violations: None };
        self
    }

    /// Expect a specific number of violations
    pub fn expect_violations(mut self, count: usize) -> Self {
        self.expected = TestExpectation::Invalid {
            violations: Some(count),
        };
        self
    }

    /// Expect a specific constraint to be violated
    pub fn expect_violation(mut self, constraint_id: &str) -> Self {
        self.expected = TestExpectation::SpecificViolation {
            constraint_id: constraint_id.to_string(),
            focus_node: None,
        };
        self
    }

    /// Expect a specific constraint to be violated for a specific focus node
    pub fn expect_violation_for(mut self, constraint_id: &str, focus_node: &str) -> Self {
        self.expected = TestExpectation::SpecificViolation {
            constraint_id: constraint_id.to_string(),
            focus_node: Some(focus_node.to_string()),
        };
        self
    }

    /// Expect validation to pass with warnings
    pub fn expect_valid_with_warnings(mut self) -> Self {
        self.expected = TestExpectation::ValidWithWarnings;
        self
    }
}

/// Expected outcome of a test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestExpectation {
    /// Data should conform to all shapes
    Valid,
    /// Data should have violations
    Invalid {
        /// Expected number of violations (None = any number)
        violations: Option<usize>,
    },
    /// Data should violate a specific constraint
    SpecificViolation {
        /// Constraint ID that should be violated
        constraint_id: String,
        /// Focus node that should violate (optional)
        focus_node: Option<String>,
    },
    /// Data should be valid but have warnings
    ValidWithWarnings,
}

/// Result of running a single test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test case name
    pub test_name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Test status
    pub status: TestStatus,
    /// Actual validation report
    pub report: Option<ValidationReport>,
    /// Error message if test failed
    pub error: Option<String>,
    /// Execution time
    pub duration: Duration,
    /// Detailed failure information
    pub failure_details: Option<String>,
}

/// Test execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    /// Test passed
    Passed,
    /// Test failed
    Failed,
    /// Test was skipped
    Skipped,
    /// Test errored (exception/panic)
    Error,
    /// Test timed out
    Timeout,
}

/// Test suite for organizing multiple test cases
#[derive(Debug, Clone)]
pub struct ShapeTestSuite {
    /// Suite name
    pub name: String,
    /// Suite description
    pub description: Option<String>,
    /// Test cases
    pub tests: Vec<TestCase>,
    /// Default timeout for tests
    pub default_timeout: Duration,
    /// Stop on first failure
    pub fail_fast: bool,
    /// Shapes to use for all tests
    pub shapes: Option<IndexMap<ShapeId, Shape>>,
}

impl ShapeTestSuite {
    /// Create a new test suite
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            tests: Vec::new(),
            default_timeout: Duration::from_secs(30),
            fail_fast: false,
            shapes: None,
        }
    }

    /// Set suite description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// Set default timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Enable fail-fast mode
    pub fn fail_fast(mut self) -> Self {
        self.fail_fast = true;
        self
    }

    /// Set shapes to use for all tests
    pub fn with_shapes(mut self, shapes: IndexMap<ShapeId, Shape>) -> Self {
        self.shapes = Some(shapes);
        self
    }

    /// Add a test case
    pub fn add_test(&mut self, test: TestCase) {
        self.tests.push(test);
    }

    /// Add multiple test cases
    pub fn add_tests(&mut self, tests: Vec<TestCase>) {
        self.tests.extend(tests);
    }

    /// Get test count
    pub fn test_count(&self) -> usize {
        self.tests.len()
    }

    /// Get tests by tag
    pub fn get_tests_by_tag(&self, tag: &str) -> Vec<&TestCase> {
        self.tests
            .iter()
            .filter(|t| t.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Run all tests in the suite
    pub fn run(
        &self,
        shapes: &IndexMap<ShapeId, Shape>,
        store_factory: impl Fn(&str) -> Box<dyn Store>,
    ) -> TestSuiteResult {
        let start = Instant::now();
        let mut results = Vec::new();

        let shapes_to_use = self.shapes.as_ref().unwrap_or(shapes);

        for test in &self.tests {
            if self.fail_fast && results.iter().any(|r: &TestResult| !r.passed) {
                // Skip remaining tests
                results.push(TestResult {
                    test_name: test.name.clone(),
                    passed: false,
                    status: TestStatus::Skipped,
                    report: None,
                    error: Some("Skipped due to fail-fast".to_string()),
                    duration: Duration::from_millis(0),
                    failure_details: None,
                });
                continue;
            }

            let result = self.run_test(test, shapes_to_use, &store_factory);
            results.push(result);
        }

        let total_duration = start.elapsed();

        // Calculate summary
        let passed = results
            .iter()
            .filter(|r| r.status == TestStatus::Passed)
            .count();
        let failed = results
            .iter()
            .filter(|r| r.status == TestStatus::Failed)
            .count();
        let skipped = results
            .iter()
            .filter(|r| r.status == TestStatus::Skipped)
            .count();
        let errored = results
            .iter()
            .filter(|r| r.status == TestStatus::Error)
            .count();

        TestSuiteResult {
            suite_name: self.name.clone(),
            results,
            summary: TestSummary {
                total: self.tests.len(),
                passed,
                failed,
                skipped,
                errored,
                duration: total_duration,
                success_rate: if self.tests.is_empty() {
                    1.0
                } else {
                    passed as f64 / self.tests.len() as f64
                },
            },
        }
    }

    /// Run a single test
    fn run_test(
        &self,
        test: &TestCase,
        shapes: &IndexMap<ShapeId, Shape>,
        store_factory: &impl Fn(&str) -> Box<dyn Store>,
    ) -> TestResult {
        let start = Instant::now();

        // Check if skipped
        if test.skip {
            return TestResult {
                test_name: test.name.clone(),
                passed: false,
                status: TestStatus::Skipped,
                report: None,
                error: test.skip_reason.clone(),
                duration: start.elapsed(),
                failure_details: None,
            };
        }

        // Create store and load data
        let store = store_factory(&test.data);

        // Create validator
        let config = ValidationConfig::default();
        let mut validator = Validator::with_config(config);
        for (_, shape) in shapes.iter() {
            let _ = validator.add_shape(shape.clone());
        }

        // Run validation
        let report = match validator.validate_store(&*store, None) {
            Ok(report) => report,
            Err(e) => {
                return TestResult {
                    test_name: test.name.clone(),
                    passed: false,
                    status: TestStatus::Error,
                    report: None,
                    error: Some(e.to_string()),
                    duration: start.elapsed(),
                    failure_details: None,
                };
            }
        };

        // Check expectations
        let (passed, failure_details) = self.check_expectation(test, &report);

        TestResult {
            test_name: test.name.clone(),
            passed,
            status: if passed {
                TestStatus::Passed
            } else {
                TestStatus::Failed
            },
            report: Some(report),
            error: None,
            duration: start.elapsed(),
            failure_details,
        }
    }

    /// Check if the test result matches expectations
    fn check_expectation(
        &self,
        test: &TestCase,
        report: &ValidationReport,
    ) -> (bool, Option<String>) {
        match &test.expected {
            TestExpectation::Valid => {
                if report.conforms() {
                    (true, None)
                } else {
                    let details = format!(
                        "Expected valid, but got {} violation(s): {}",
                        report.violations().len(),
                        report
                            .violations()
                            .iter()
                            .map(|v| v.result_message.clone().unwrap_or_default())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    (false, Some(details))
                }
            }
            TestExpectation::Invalid { violations } => {
                if report.conforms() {
                    (
                        false,
                        Some("Expected invalid, but data conforms".to_string()),
                    )
                } else if let Some(expected_count) = violations {
                    let actual_count = report.violations().len();
                    if actual_count == *expected_count {
                        (true, None)
                    } else {
                        (
                            false,
                            Some(format!(
                                "Expected {} violations, got {}",
                                expected_count, actual_count
                            )),
                        )
                    }
                } else {
                    (true, None)
                }
            }
            TestExpectation::SpecificViolation {
                constraint_id,
                focus_node,
            } => {
                let has_violation = report.violations().iter().any(|v| {
                    let constraint_matches =
                        v.source_constraint_component.0.contains(constraint_id);

                    let node_matches = if let Some(node) = focus_node {
                        v.focus_node.to_string().contains(node)
                    } else {
                        true
                    };

                    constraint_matches && node_matches
                });

                if has_violation {
                    (true, None)
                } else {
                    let details = if let Some(node) = focus_node {
                        format!(
                            "Expected violation '{}' for node '{}', but not found",
                            constraint_id, node
                        )
                    } else {
                        format!("Expected violation '{}', but not found", constraint_id)
                    };
                    (false, Some(details))
                }
            }
            TestExpectation::ValidWithWarnings => {
                if report.conforms() {
                    // Check for warnings (violations with Warning severity)
                    let has_warnings = report
                        .violations()
                        .iter()
                        .any(|v| v.result_severity == crate::Severity::Warning);
                    if has_warnings {
                        (true, None)
                    } else {
                        (false, Some("Expected warnings, but none found".to_string()))
                    }
                } else {
                    (
                        false,
                        Some("Expected valid with warnings, but got errors".to_string()),
                    )
                }
            }
        }
    }
}

/// Result of running a test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteResult {
    /// Suite name
    pub suite_name: String,
    /// Individual test results
    pub results: Vec<TestResult>,
    /// Summary statistics
    pub summary: TestSummary,
}

impl TestSuiteResult {
    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.summary.failed == 0 && self.summary.errored == 0
    }

    /// Get failed tests
    pub fn failed_tests(&self) -> Vec<&TestResult> {
        self.results
            .iter()
            .filter(|r| r.status == TestStatus::Failed)
            .collect()
    }

    /// Generate a report
    pub fn generate_report(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("Test Suite: {}\n", self.suite_name));
        output.push_str(&format!("{}\n\n", "=".repeat(50)));

        // Summary
        output.push_str("Summary:\n");
        output.push_str(&format!("  Total:   {}\n", self.summary.total));
        output.push_str(&format!("  Passed:  {}\n", self.summary.passed));
        output.push_str(&format!("  Failed:  {}\n", self.summary.failed));
        output.push_str(&format!("  Skipped: {}\n", self.summary.skipped));
        output.push_str(&format!("  Errored: {}\n", self.summary.errored));
        output.push_str(&format!(
            "  Success Rate: {:.1}%\n",
            self.summary.success_rate * 100.0
        ));
        output.push_str(&format!("  Duration: {:?}\n\n", self.summary.duration));

        // Test details
        output.push_str("Test Results:\n");
        for result in &self.results {
            let status = match result.status {
                TestStatus::Passed => "✅ PASS",
                TestStatus::Failed => "❌ FAIL",
                TestStatus::Skipped => "⏭️ SKIP",
                TestStatus::Error => "💥 ERROR",
                TestStatus::Timeout => "⏱️ TIMEOUT",
            };

            output.push_str(&format!(
                "  {} {} ({:?})\n",
                status, result.test_name, result.duration
            ));

            if let Some(details) = &result.failure_details {
                output.push_str(&format!("    Details: {}\n", details));
            }

            if let Some(error) = &result.error {
                output.push_str(&format!("    Error: {}\n", error));
            }
        }

        output
    }
}

/// Summary statistics for a test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    /// Total number of tests
    pub total: usize,
    /// Number of passed tests
    pub passed: usize,
    /// Number of failed tests
    pub failed: usize,
    /// Number of skipped tests
    pub skipped: usize,
    /// Number of errored tests
    pub errored: usize,
    /// Total duration
    pub duration: Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
}

/// Assertion helpers for testing
pub struct TestAssertions;

impl TestAssertions {
    /// Assert that a validation report conforms
    pub fn assert_conforms(report: &ValidationReport) {
        assert!(
            report.conforms(),
            "Expected validation to pass, but got {} violation(s)",
            report.violations().len()
        );
    }

    /// Assert that a validation report has violations
    pub fn assert_violates(report: &ValidationReport) {
        assert!(
            !report.conforms(),
            "Expected violations, but validation passed"
        );
    }

    /// Assert a specific number of violations
    pub fn assert_violation_count(report: &ValidationReport, expected: usize) {
        let actual = report.violations().len();
        assert_eq!(
            actual, expected,
            "Expected {} violations, got {}",
            expected, actual
        );
    }

    /// Assert that a specific constraint was violated
    pub fn assert_constraint_violated(report: &ValidationReport, constraint_id: &str) {
        let has_violation = report
            .violations()
            .iter()
            .any(|v| v.source_constraint_component.0.contains(constraint_id));

        assert!(
            has_violation,
            "Expected constraint '{}' to be violated, but it wasn't",
            constraint_id
        );
    }

    /// Assert that no specific constraint was violated
    pub fn assert_constraint_not_violated(report: &ValidationReport, constraint_id: &str) {
        let has_violation = report
            .violations()
            .iter()
            .any(|v| v.source_constraint_component.0.contains(constraint_id));

        assert!(
            !has_violation,
            "Expected constraint '{}' to not be violated, but it was",
            constraint_id
        );
    }

    /// Assert that a specific focus node was targeted
    pub fn assert_focus_node_validated(report: &ValidationReport, focus_node: &str) {
        let was_validated = report
            .violations()
            .iter()
            .any(|v| v.focus_node.to_string().contains(focus_node));

        // Note: This only checks violations, not successful validations
        // A more complete implementation would track all validated nodes
        if !was_validated && !report.conforms() {
            panic!("Focus node '{}' was not found in violations", focus_node);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_builder() {
        let test = TestCase::new("my_test")
            .with_description("Test description")
            .with_data("ex:john a ex:Person .")
            .with_tags(&["person", "basic"])
            .expect_valid();

        assert_eq!(test.name, "my_test");
        assert!(test.description.is_some());
        assert_eq!(test.tags.len(), 2);
        assert!(matches!(test.expected, TestExpectation::Valid));
    }

    #[test]
    fn test_expectation_variants() {
        let test1 = TestCase::new("test1").expect_valid();
        assert!(matches!(test1.expected, TestExpectation::Valid));

        let test2 = TestCase::new("test2").expect_invalid();
        assert!(matches!(
            test2.expected,
            TestExpectation::Invalid { violations: None }
        ));

        let test3 = TestCase::new("test3").expect_violations(5);
        assert!(matches!(
            test3.expected,
            TestExpectation::Invalid {
                violations: Some(5)
            }
        ));

        let test4 = TestCase::new("test4").expect_violation("sh:minCount");
        assert!(matches!(
            test4.expected,
            TestExpectation::SpecificViolation { .. }
        ));
    }

    #[test]
    fn test_suite_creation() {
        let mut suite = ShapeTestSuite::new("My Suite")
            .with_description("Test suite description")
            .with_timeout(Duration::from_secs(60))
            .fail_fast();

        suite.add_test(TestCase::new("test1").expect_valid());
        suite.add_test(TestCase::new("test2").expect_invalid());

        assert_eq!(suite.test_count(), 2);
        assert!(suite.fail_fast);
    }

    #[test]
    fn test_tags_filtering() {
        let mut suite = ShapeTestSuite::new("Tagged Suite");

        suite.add_test(
            TestCase::new("test1")
                .with_tags(&["fast", "unit"])
                .expect_valid(),
        );
        suite.add_test(
            TestCase::new("test2")
                .with_tags(&["slow", "integration"])
                .expect_valid(),
        );
        suite.add_test(
            TestCase::new("test3")
                .with_tags(&["fast", "integration"])
                .expect_valid(),
        );

        let fast_tests = suite.get_tests_by_tag("fast");
        assert_eq!(fast_tests.len(), 2);

        let integration_tests = suite.get_tests_by_tag("integration");
        assert_eq!(integration_tests.len(), 2);
    }

    #[test]
    fn test_skip() {
        let test = TestCase::new("skipped_test")
            .skip("Not implemented yet")
            .expect_valid();

        assert!(test.skip);
        assert!(test.skip_reason.is_some());
    }

    #[test]
    fn test_summary_calculation() {
        let summary = TestSummary {
            total: 10,
            passed: 7,
            failed: 2,
            skipped: 1,
            errored: 0,
            duration: Duration::from_secs(5),
            success_rate: 0.7,
        };

        assert_eq!(
            summary.passed + summary.failed + summary.skipped + summary.errored,
            10
        );
    }
}
