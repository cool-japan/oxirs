//! W3C SHACL Test Suite Integration
//!
//! This module provides comprehensive integration with the official W3C SHACL test suite
//! for validation of SHACL specification compliance. It includes test manifest parsing,
//! automated test execution, and detailed compliance reporting.
//!
//! The W3C SHACL test suite is available at: https://w3c.github.io/data-shapes/data-shapes-test-suite/

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use url::Url;

use oxirs_core::{
    model::{NamedNode, Term, Triple},
    Store, ConcreteStore,
};

use crate::{
    ValidationEngine, ValidationConfig, ValidationReport, Validator,
    Shape, ShapeId, ShaclError,
    shapes::ShapeFactory,
};

/// W3C SHACL test suite runner and compliance checker
#[derive(Debug)]
pub struct W3cTestSuiteRunner {
    /// Base URL for test suite resources
    base_url: Url,
    
    /// Test suite configuration
    config: W3cTestConfig,
    
    /// Loaded test manifests
    manifests: Vec<TestManifest>,
    
    /// Test execution results
    results: HashMap<String, TestResult>,
    
    /// Compliance statistics
    stats: ComplianceStats,
}

/// Configuration for W3C test suite execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct W3cTestConfig {
    /// Test suite base directory or URL
    pub test_suite_location: String,
    
    /// Enable specific test categories
    pub enabled_categories: HashSet<TestCategory>,
    
    /// Test execution timeout in seconds
    pub test_timeout_seconds: u64,
    
    /// Maximum number of parallel test executions
    pub max_parallel_tests: usize,
    
    /// Enable detailed test logging
    pub verbose_logging: bool,
    
    /// Output directory for test reports
    pub output_directory: Option<PathBuf>,
    
    /// Custom test filters
    pub test_filters: Vec<TestFilter>,
}

impl Default for W3cTestConfig {
    fn default() -> Self {
        let mut enabled_categories = HashSet::new();
        enabled_categories.insert(TestCategory::Core);
        enabled_categories.insert(TestCategory::PropertyPaths);
        enabled_categories.insert(TestCategory::NodeShapes);
        enabled_categories.insert(TestCategory::PropertyShapes);
        
        Self {
            test_suite_location: "https://w3c.github.io/data-shapes/data-shapes-test-suite/".to_string(),
            enabled_categories,
            test_timeout_seconds: 30,
            max_parallel_tests: 4,
            verbose_logging: false,
            output_directory: None,
            test_filters: Vec::new(),
        }
    }
}

/// W3C SHACL test categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TestCategory {
    /// Core SHACL constraint tests
    Core,
    /// Property path tests
    PropertyPaths,
    /// Node shape tests
    NodeShapes,
    /// Property shape tests
    PropertyShapes,
    /// Logical constraint tests
    LogicalConstraints,
    /// SPARQL-based constraint tests
    SparqlConstraints,
    /// Advanced feature tests
    Advanced,
    /// Performance and scalability tests
    Performance,
}

/// Test filter for selective test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestFilter {
    /// Filter name
    pub name: String,
    
    /// Test pattern to match
    pub test_pattern: String,
    
    /// Whether this is an inclusion or exclusion filter
    pub include: bool,
}

/// W3C test manifest structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestManifest {
    /// Manifest identifier
    pub id: String,
    
    /// Manifest label/name
    pub label: String,
    
    /// Manifest description
    pub description: Option<String>,
    
    /// Test entries in this manifest
    pub entries: Vec<TestEntry>,
    
    /// Manifest category
    pub category: TestCategory,
    
    /// Manifest source location
    pub source_location: String,
}

/// Individual test entry from W3C test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEntry {
    /// Test identifier
    pub id: String,
    
    /// Test label/name
    pub label: String,
    
    /// Test description
    pub description: Option<String>,
    
    /// Test type (validation, parse error, etc.)
    pub test_type: TestType,
    
    /// Data graph location
    pub data_graph: Option<String>,
    
    /// Shapes graph location
    pub shapes_graph: Option<String>,
    
    /// Expected validation result
    pub expected_result: ExpectedResult,
    
    /// Additional test metadata
    pub metadata: HashMap<String, String>,
}

/// Types of W3C SHACL tests
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestType {
    /// Standard validation test
    Validation,
    /// Parse error test (invalid shapes)
    ParseError,
    /// Feature support test
    FeatureSupport,
    /// Performance test
    Performance,
}

/// Expected test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedResult {
    /// Whether validation should conform
    pub conforms: bool,
    
    /// Expected number of violations (if any)
    pub violation_count: Option<usize>,
    
    /// Expected specific violations
    pub expected_violations: Vec<ExpectedViolation>,
    
    /// Expected error type (for parse error tests)
    pub expected_error: Option<String>,
}

/// Expected violation details for test verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedViolation {
    /// Focus node of expected violation
    pub focus_node: Option<String>,
    
    /// Result path of expected violation
    pub result_path: Option<String>,
    
    /// Violating value
    pub value: Option<String>,
    
    /// Source shape
    pub source_shape: Option<String>,
    
    /// Source constraint component
    pub source_constraint_component: Option<String>,
}

/// Result of executing a W3C test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test entry that was executed
    pub test_entry: TestEntry,
    
    /// Test execution status
    pub status: TestStatus,
    
    /// Actual validation result (if test succeeded)
    pub actual_result: Option<ValidationReport>,
    
    /// Error message (if test failed)
    pub error_message: Option<String>,
    
    /// Test execution time
    pub execution_time_ms: u64,
    
    /// Compliance assessment
    pub compliance: ComplianceAssessment,
    
    /// Test execution timestamp
    pub executed_at: DateTime<Utc>,
}

/// Test execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    /// Test passed with expected results
    Passed,
    /// Test failed with unexpected results
    Failed,
    /// Test was skipped
    Skipped,
    /// Test execution error
    Error,
    /// Test timed out
    Timeout,
}

/// Compliance assessment for individual test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceAssessment {
    /// Overall compliance status
    pub compliant: bool,
    
    /// Specific compliance issues
    pub issues: Vec<ComplianceIssue>,
    
    /// Compliance score (0.0 to 1.0)
    pub score: f64,
    
    /// Detailed assessment notes
    pub notes: Vec<String>,
}

/// Specific compliance issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceIssue {
    /// Issue type
    pub issue_type: ComplianceIssueType,
    
    /// Issue description
    pub description: String,
    
    /// Issue severity
    pub severity: ComplianceIssueSeverity,
    
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Types of compliance issues
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceIssueType {
    /// Incorrect validation result
    IncorrectResult,
    /// Missing violation
    MissingViolation,
    /// Unexpected violation
    UnexpectedViolation,
    /// Incorrect violation details
    IncorrectViolationDetails,
    /// Performance issue
    PerformanceIssue,
    /// Parse error handling
    ParseErrorHandling,
}

/// Severity of compliance issues
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ComplianceIssueSeverity {
    /// Critical compliance failure
    Critical,
    /// Major compliance issue
    Major,
    /// Minor compliance issue
    Minor,
    /// Informational issue
    Info,
}

/// Overall compliance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStats {
    /// Total tests executed
    pub total_tests: usize,
    
    /// Tests passed
    pub tests_passed: usize,
    
    /// Tests failed
    pub tests_failed: usize,
    
    /// Tests skipped
    pub tests_skipped: usize,
    
    /// Tests with errors
    pub tests_error: usize,
    
    /// Overall compliance percentage
    pub compliance_percentage: f64,
    
    /// Compliance by category
    pub compliance_by_category: HashMap<TestCategory, f64>,
    
    /// Most common compliance issues
    pub common_issues: Vec<(ComplianceIssueType, usize)>,
    
    /// Total execution time
    pub total_execution_time_ms: u64,
}

impl W3cTestSuiteRunner {
    /// Create a new W3C test suite runner
    pub fn new(config: W3cTestConfig) -> Result<Self> {
        let base_url = Url::parse(&config.test_suite_location)?;
        
        Ok(Self {
            base_url,
            config,
            manifests: Vec::new(),
            results: HashMap::new(),
            stats: ComplianceStats::default(),
        })
    }
    
    /// Load test manifests from the test suite
    pub async fn load_manifests(&mut self) -> Result<()> {
        // This would typically load manifests from W3C test suite repository
        // For now, we'll create some example manifests to demonstrate the structure
        
        self.manifests.push(self.create_core_constraints_manifest()?);
        self.manifests.push(self.create_property_paths_manifest()?);
        self.manifests.push(self.create_logical_constraints_manifest()?);
        
        Ok(())
    }
    
    /// Execute all loaded tests
    pub async fn execute_all_tests(&mut self) -> Result<ComplianceStats> {
        let start_time = std::time::Instant::now();
        
        // Collect all test entries first to avoid borrowing issues
        let mut test_entries = Vec::new();
        for manifest in &self.manifests {
            if !self.config.enabled_categories.contains(&manifest.category) {
                continue;
            }
            
            for test_entry in &manifest.entries {
                test_entries.push(test_entry.clone());
            }
        }
        
        // Execute tests
        for test_entry in test_entries {
            if self.should_skip_test(&test_entry) {
                self.record_skipped_test(&test_entry);
                continue;
            }
            
            let result = self.execute_test(&test_entry).await?;
            self.results.insert(test_entry.id.clone(), result);
        }
        
        self.stats.total_execution_time_ms = start_time.elapsed().as_millis() as u64;
        self.calculate_compliance_stats();
        
        Ok(self.stats.clone())
    }
    
    /// Execute a single test
    async fn execute_test(&self, test_entry: &TestEntry) -> Result<TestResult> {
        let start_time = std::time::Instant::now();
        
        let result = match self.execute_test_internal(test_entry).await {
            Ok(validation_result) => {
                let compliance = self.assess_compliance(test_entry, &validation_result);
                let status = if compliance.compliant {
                    TestStatus::Passed
                } else {
                    TestStatus::Failed
                };
                
                TestResult {
                    test_entry: test_entry.clone(),
                    status,
                    actual_result: Some(validation_result),
                    error_message: None,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    compliance,
                    executed_at: Utc::now(),
                }
            }
            Err(e) => TestResult {
                test_entry: test_entry.clone(),
                status: TestStatus::Error,
                actual_result: None,
                error_message: Some(e.to_string()),
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                compliance: ComplianceAssessment {
                    compliant: false,
                    issues: vec![ComplianceIssue {
                        issue_type: ComplianceIssueType::ParseErrorHandling,
                        description: format!("Test execution failed: {}", e),
                        severity: ComplianceIssueSeverity::Critical,
                        suggested_fix: Some("Check test data and shapes validity".to_string()),
                    }],
                    score: 0.0,
                    notes: vec!["Test execution failed".to_string()],
                },
                executed_at: Utc::now(),
            }
        };
        
        Ok(result)
    }
    
    /// Internal test execution logic
    async fn execute_test_internal(&self, test_entry: &TestEntry) -> Result<ValidationReport> {
        // Load shapes graph
        let shapes_store = if let Some(shapes_location) = &test_entry.shapes_graph {
            self.load_rdf_store(shapes_location).await?
        } else {
            ConcreteStore::new()?
        };
        
        // Load data graph
        let data_store = if let Some(data_location) = &test_entry.data_graph {
            self.load_rdf_store(data_location).await?
        } else {
            ConcreteStore::new()?
        };
        
        // Parse shapes from shapes graph (placeholder implementation)
        let shapes_vec = self.parse_shapes_from_store(&shapes_store)?;
        
        // Convert Vec to IndexMap for ValidationEngine
        let mut shapes = indexmap::IndexMap::new();
        for shape in shapes_vec {
            shapes.insert(shape.id.clone(), shape);
        }
        
        // Create validation engine
        let config = ValidationConfig::default();
        let mut engine = ValidationEngine::new(&shapes, config);
        
        // Execute validation
        let report = engine.validate_store(&data_store)?;
        
        Ok(report)
    }
    
    /// Load RDF store from location (URL or file path)
    async fn load_rdf_store(&self, _location: &str) -> Result<ConcreteStore> {
        // TODO: Implement proper RDF loading once oxirs-core API is stabilized
        // For now, return an empty store for testing framework structure
        let store = ConcreteStore::new()?;
        Ok(store)
    }
    
    /// Parse shapes from RDF store (placeholder implementation)
    fn parse_shapes_from_store(&self, _store: &ConcreteStore) -> Result<Vec<Shape>> {
        // TODO: Implement proper shape parsing from RDF store
        // For now, return empty shapes for testing framework structure
        Ok(Vec::new())
    }
    
    
    /// Assess compliance for a test result
    fn assess_compliance(&self, test_entry: &TestEntry, actual_result: &ValidationReport) -> ComplianceAssessment {
        let mut issues = Vec::new();
        let mut score: f64 = 1.0;
        
        // Check conformance result
        if actual_result.conforms != test_entry.expected_result.conforms {
            issues.push(ComplianceIssue {
                issue_type: ComplianceIssueType::IncorrectResult,
                description: format!(
                    "Expected conforms: {}, actual: {}",
                    test_entry.expected_result.conforms,
                    actual_result.conforms
                ),
                severity: ComplianceIssueSeverity::Critical,
                suggested_fix: Some("Review validation logic for this constraint type".to_string()),
            });
            score -= 0.5;
        }
        
        // Check violation count if specified
        if let Some(expected_count) = test_entry.expected_result.violation_count {
            let actual_count = actual_result.violations.len();
            if actual_count != expected_count {
                issues.push(ComplianceIssue {
                    issue_type: ComplianceIssueType::IncorrectResult,
                    description: format!(
                        "Expected {} violations, found {}",
                        expected_count,
                        actual_count
                    ),
                    severity: ComplianceIssueSeverity::Major,
                    suggested_fix: Some("Check violation detection and counting logic".to_string()),
                });
                score -= 0.3;
            }
        }
        
        // TODO: Add more detailed violation matching
        
        let compliant = issues.is_empty() || issues.iter().all(|i| i.severity == ComplianceIssueSeverity::Info);
        
        ComplianceAssessment {
            compliant,
            issues,
            score: score.max(0.0),
            notes: vec![format!("Test {}: {}", test_entry.id, test_entry.label)],
        }
    }
    
    /// Check if a test should be skipped based on filters
    fn should_skip_test(&self, test_entry: &TestEntry) -> bool {
        for filter in &self.config.test_filters {
            let matches = test_entry.id.contains(&filter.test_pattern) 
                || test_entry.label.contains(&filter.test_pattern);
            
            if filter.include && !matches {
                return true; // Skip if it doesn't match an include filter
            }
            if !filter.include && matches {
                return true; // Skip if it matches an exclude filter
            }
        }
        
        false
    }
    
    /// Record a skipped test
    fn record_skipped_test(&mut self, test_entry: &TestEntry) {
        let result = TestResult {
            test_entry: test_entry.clone(),
            status: TestStatus::Skipped,
            actual_result: None,
            error_message: Some("Test skipped by filter".to_string()),
            execution_time_ms: 0,
            compliance: ComplianceAssessment {
                compliant: true, // Skipped tests are considered compliant
                issues: Vec::new(),
                score: 1.0,
                notes: vec!["Test skipped".to_string()],
            },
            executed_at: Utc::now(),
        };
        
        self.results.insert(test_entry.id.clone(), result);
    }
    
    /// Calculate overall compliance statistics
    fn calculate_compliance_stats(&mut self) {
        let mut total = 0;
        let mut passed = 0;
        let mut failed = 0;
        let mut skipped = 0;
        let mut error = 0;
        
        let mut category_stats: HashMap<TestCategory, (usize, usize)> = HashMap::new();
        let mut issue_counts: HashMap<ComplianceIssueType, usize> = HashMap::new();
        
        for result in self.results.values() {
            total += 1;
            
            match result.status {
                TestStatus::Passed => passed += 1,
                TestStatus::Failed => failed += 1,
                TestStatus::Skipped => skipped += 1,
                TestStatus::Error | TestStatus::Timeout => error += 1,
            }
            
            // Count issues
            for issue in &result.compliance.issues {
                *issue_counts.entry(issue.issue_type.clone()).or_insert(0) += 1;
            }
        }
        
        let compliance_percentage = if total > 0 {
            (passed as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        
        let common_issues: Vec<(ComplianceIssueType, usize)> = {
            let mut issues: Vec<_> = issue_counts.into_iter().collect();
            issues.sort_by(|a, b| b.1.cmp(&a.1));
            issues
        };
        
        self.stats = ComplianceStats {
            total_tests: total,
            tests_passed: passed,
            tests_failed: failed,
            tests_skipped: skipped,
            tests_error: error,
            compliance_percentage,
            compliance_by_category: HashMap::new(), // TODO: Implement category breakdown
            common_issues,
            total_execution_time_ms: self.stats.total_execution_time_ms,
        };
    }
    
    /// Generate compliance report
    pub fn generate_compliance_report(&self) -> ComplianceReport {
        ComplianceReport {
            summary: self.stats.clone(),
            test_results: self.results.values().cloned().collect(),
            generated_at: Utc::now(),
            test_suite_version: "W3C SHACL Test Suite".to_string(),
            implementation_details: ImplementationDetails {
                version: env!("CARGO_PKG_VERSION").to_string(),
                features: vec!["core".to_string(), "property-paths".to_string()],
                limitations: vec![
                    "SPARQL constraints not fully implemented".to_string(),
                    "Some advanced features pending".to_string(),
                ],
            },
        }
    }
    
    /// Create example core constraints manifest
    fn create_core_constraints_manifest(&self) -> Result<TestManifest> {
        Ok(TestManifest {
            id: "core-constraints".to_string(),
            label: "Core SHACL Constraints".to_string(),
            description: Some("Tests for basic SHACL constraint components".to_string()),
            category: TestCategory::Core,
            source_location: "core/".to_string(),
            entries: vec![
                TestEntry {
                    id: "core-class-001".to_string(),
                    label: "Class constraint - valid".to_string(),
                    description: Some("Test that class constraint accepts valid instances".to_string()),
                    test_type: TestType::Validation,
                    data_graph: Some("core/class-001-data.ttl".to_string()),
                    shapes_graph: Some("core/class-001-shapes.ttl".to_string()),
                    expected_result: ExpectedResult {
                        conforms: true,
                        violation_count: Some(0),
                        expected_violations: Vec::new(),
                        expected_error: None,
                    },
                    metadata: HashMap::new(),
                },
                TestEntry {
                    id: "core-datatype-001".to_string(),
                    label: "Datatype constraint - valid".to_string(),
                    description: Some("Test that datatype constraint accepts valid literals".to_string()),
                    test_type: TestType::Validation,
                    data_graph: Some("core/datatype-001-data.ttl".to_string()),
                    shapes_graph: Some("core/datatype-001-shapes.ttl".to_string()),
                    expected_result: ExpectedResult {
                        conforms: true,
                        violation_count: Some(0),
                        expected_violations: Vec::new(),
                        expected_error: None,
                    },
                    metadata: HashMap::new(),
                },
            ],
        })
    }
    
    /// Create example property paths manifest
    fn create_property_paths_manifest(&self) -> Result<TestManifest> {
        Ok(TestManifest {
            id: "property-paths".to_string(),
            label: "Property Path Tests".to_string(),
            description: Some("Tests for SHACL property path expressions".to_string()),
            category: TestCategory::PropertyPaths,
            source_location: "path/".to_string(),
            entries: vec![
                TestEntry {
                    id: "path-sequence-001".to_string(),
                    label: "Sequence path - valid".to_string(),
                    description: Some("Test sequence property path validation".to_string()),
                    test_type: TestType::Validation,
                    data_graph: Some("path/sequence-001-data.ttl".to_string()),
                    shapes_graph: Some("path/sequence-001-shapes.ttl".to_string()),
                    expected_result: ExpectedResult {
                        conforms: true,
                        violation_count: Some(0),
                        expected_violations: Vec::new(),
                        expected_error: None,
                    },
                    metadata: HashMap::new(),
                },
            ],
        })
    }
    
    /// Create example logical constraints manifest
    fn create_logical_constraints_manifest(&self) -> Result<TestManifest> {
        Ok(TestManifest {
            id: "logical-constraints".to_string(),
            label: "Logical Constraint Tests".to_string(),
            description: Some("Tests for SHACL logical constraints (and, or, not, xone)".to_string()),
            category: TestCategory::LogicalConstraints,
            source_location: "logical/".to_string(),
            entries: vec![
                TestEntry {
                    id: "logical-and-001".to_string(),
                    label: "AND constraint - valid".to_string(),
                    description: Some("Test AND logical constraint with valid data".to_string()),
                    test_type: TestType::Validation,
                    data_graph: Some("logical/and-001-data.ttl".to_string()),
                    shapes_graph: Some("logical/and-001-shapes.ttl".to_string()),
                    expected_result: ExpectedResult {
                        conforms: true,
                        violation_count: Some(0),
                        expected_violations: Vec::new(),
                        expected_error: None,
                    },
                    metadata: HashMap::new(),
                },
            ],
        })
    }
}

/// Complete compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Summary statistics
    pub summary: ComplianceStats,
    
    /// Individual test results
    pub test_results: Vec<TestResult>,
    
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    
    /// Test suite version information
    pub test_suite_version: String,
    
    /// Implementation details
    pub implementation_details: ImplementationDetails,
}

/// Details about the SHACL implementation being tested
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationDetails {
    /// Implementation version
    pub version: String,
    
    /// Supported features
    pub features: Vec<String>,
    
    /// Known limitations
    pub limitations: Vec<String>,
}

impl Default for ComplianceStats {
    fn default() -> Self {
        Self {
            total_tests: 0,
            tests_passed: 0,
            tests_failed: 0,
            tests_skipped: 0,
            tests_error: 0,
            compliance_percentage: 0.0,
            compliance_by_category: HashMap::new(),
            common_issues: Vec::new(),
            total_execution_time_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_w3c_test_suite_runner_creation() {
        let config = W3cTestConfig::default();
        let runner = W3cTestSuiteRunner::new(config);
        assert!(runner.is_ok());
    }
    
    #[test]
    fn test_manifest_creation() {
        let config = W3cTestConfig::default();
        let runner = W3cTestSuiteRunner::new(config).unwrap();
        
        let manifest = runner.create_core_constraints_manifest().unwrap();
        assert_eq!(manifest.id, "core-constraints");
        assert_eq!(manifest.category, TestCategory::Core);
        assert!(!manifest.entries.is_empty());
    }
    
    #[test]
    fn test_compliance_assessment() {
        let test_entry = TestEntry {
            id: "test-001".to_string(),
            label: "Test case".to_string(),
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
            metadata: HashMap::new(),
        };
        
        let mut validation_report = ValidationReport::new();
        validation_report.conforms = true;
        
        let config = W3cTestConfig::default();
        let runner = W3cTestSuiteRunner::new(config).unwrap();
        
        let assessment = runner.assess_compliance(&test_entry, &validation_report);
        assert!(assessment.compliant);
        assert_eq!(assessment.score, 1.0);
    }
    
    #[tokio::test]
    async fn test_manifest_loading() {
        let config = W3cTestConfig::default();
        let mut runner = W3cTestSuiteRunner::new(config).unwrap();
        
        let result = runner.load_manifests().await;
        assert!(result.is_ok());
        assert!(!runner.manifests.is_empty());
    }
}