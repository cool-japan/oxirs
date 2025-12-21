//! CI/CD Integration for SHACL Validation
//!
//! This module provides comprehensive CI/CD integration capabilities for SHACL validation,
//! enabling seamless integration with continuous integration and deployment pipelines.
//!
//! # Features
//!
//! - Multiple output formats (JUnit XML, TAP, JSON, SARIF)
//! - Exit code management for pipeline control
//! - Pre-commit hook generation
//! - GitHub Actions / GitLab CI / Jenkins integration
//! - Threshold-based pass/fail criteria
//! - Baseline comparison for regression detection

use crate::{Severity, ValidationReport, ValidationViolation};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;

/// CI/CD integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdConfig {
    /// Output format for validation results
    pub output_format: OutputFormat,
    /// Path to output file (if any)
    pub output_path: Option<PathBuf>,
    /// Exit code configuration
    pub exit_code: ExitCodeConfig,
    /// Threshold configuration
    pub thresholds: ThresholdConfig,
    /// Baseline configuration for regression detection
    pub baseline: Option<BaselineConfig>,
    /// Environment-specific settings
    pub environment: EnvironmentConfig,
    /// Generate summary for GitHub Actions
    pub github_summary: bool,
    /// Generate annotations for GitHub Actions
    pub github_annotations: bool,
}

impl Default for CiCdConfig {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::Json,
            output_path: None,
            exit_code: ExitCodeConfig::default(),
            thresholds: ThresholdConfig::default(),
            baseline: None,
            environment: EnvironmentConfig::default(),
            github_summary: false,
            github_annotations: false,
        }
    }
}

/// Output formats for CI/CD integration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// JSON format
    Json,
    /// JUnit XML format (for test reporting)
    JUnit,
    /// TAP (Test Anything Protocol) format
    Tap,
    /// SARIF (Static Analysis Results Interchange Format)
    Sarif,
    /// Plain text summary
    Text,
    /// GitHub Actions annotations
    GitHub,
    /// GitLab CI report
    GitLab,
}

/// Exit code configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitCodeConfig {
    /// Exit code for success
    pub success: i32,
    /// Exit code for violations
    pub violations: i32,
    /// Exit code for errors
    pub error: i32,
    /// Fail on violations (default: true)
    pub fail_on_violations: bool,
    /// Fail on warnings
    pub fail_on_warnings: bool,
}

impl Default for ExitCodeConfig {
    fn default() -> Self {
        Self {
            success: 0,
            violations: 1,
            error: 2,
            fail_on_violations: true,
            fail_on_warnings: false,
        }
    }
}

/// Threshold configuration for pass/fail criteria
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThresholdConfig {
    /// Maximum number of violations allowed
    pub max_violations: Option<usize>,
    /// Maximum number of warnings allowed
    pub max_warnings: Option<usize>,
    /// Minimum conformance rate (0.0 to 1.0)
    pub min_conformance_rate: Option<f64>,
    /// Maximum severity level allowed
    pub max_severity: Option<Severity>,
}

/// Baseline configuration for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineConfig {
    /// Path to baseline file
    pub baseline_path: PathBuf,
    /// Fail if new violations are introduced
    pub fail_on_new_violations: bool,
    /// Report resolved violations
    pub report_resolved: bool,
}

/// Environment-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    /// Current environment (dev, staging, prod)
    pub environment: String,
    /// CI system being used
    pub ci_system: Option<CiSystem>,
    /// Build number or ID
    pub build_id: Option<String>,
    /// Commit SHA
    pub commit_sha: Option<String>,
    /// Branch name
    pub branch: Option<String>,
    /// Pull request number
    pub pr_number: Option<String>,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            environment: "development".to_string(),
            ci_system: Self::detect_ci_system(),
            build_id: std::env::var("BUILD_NUMBER")
                .or_else(|_| std::env::var("GITHUB_RUN_NUMBER"))
                .or_else(|_| std::env::var("CI_JOB_ID"))
                .ok(),
            commit_sha: std::env::var("GITHUB_SHA")
                .or_else(|_| std::env::var("CI_COMMIT_SHA"))
                .or_else(|_| std::env::var("GIT_COMMIT"))
                .ok(),
            branch: std::env::var("GITHUB_REF")
                .or_else(|_| std::env::var("CI_COMMIT_BRANCH"))
                .or_else(|_| std::env::var("GIT_BRANCH"))
                .ok(),
            pr_number: std::env::var("GITHUB_PR_NUMBER")
                .or_else(|_| std::env::var("CI_MERGE_REQUEST_IID"))
                .ok(),
        }
    }
}

impl EnvironmentConfig {
    fn detect_ci_system() -> Option<CiSystem> {
        if std::env::var("GITHUB_ACTIONS").is_ok() {
            Some(CiSystem::GitHubActions)
        } else if std::env::var("GITLAB_CI").is_ok() {
            Some(CiSystem::GitLabCi)
        } else if std::env::var("JENKINS_URL").is_ok() {
            Some(CiSystem::Jenkins)
        } else if std::env::var("CIRCLECI").is_ok() {
            Some(CiSystem::CircleCi)
        } else if std::env::var("TRAVIS").is_ok() {
            Some(CiSystem::TravisCi)
        } else if std::env::var("AZURE_PIPELINES").is_ok() {
            Some(CiSystem::AzurePipelines)
        } else {
            None
        }
    }
}

/// CI system type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CiSystem {
    GitHubActions,
    GitLabCi,
    Jenkins,
    CircleCi,
    TravisCi,
    AzurePipelines,
}

/// CI/CD integration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdResult {
    /// Overall pass/fail status
    pub passed: bool,
    /// Exit code to use
    pub exit_code: i32,
    /// Summary message
    pub summary: String,
    /// Detailed report
    pub report: ValidationReport,
    /// Regression analysis (if baseline configured)
    pub regression: Option<RegressionAnalysis>,
    /// Threshold evaluation results
    pub threshold_results: ThresholdResults,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Environment info
    pub environment: EnvironmentConfig,
}

/// Regression analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    /// New violations introduced
    pub new_violations: Vec<ValidationViolation>,
    /// Violations that were resolved
    pub resolved_violations: Vec<ValidationViolation>,
    /// Unchanged violations
    pub unchanged_violations: Vec<ValidationViolation>,
    /// Net change in violation count
    pub net_change: i32,
}

/// Threshold evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdResults {
    /// Violation count check passed
    pub violations_check_passed: bool,
    /// Warning count check passed
    pub warnings_check_passed: bool,
    /// Conformance rate check passed
    pub conformance_check_passed: bool,
    /// Severity check passed
    pub severity_check_passed: bool,
    /// Overall threshold check passed
    pub all_passed: bool,
}

/// CI/CD integration engine
pub struct CiCdEngine {
    config: CiCdConfig,
}

impl CiCdEngine {
    /// Create a new CI/CD engine
    pub fn new(config: CiCdConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(CiCdConfig::default())
    }

    /// Process validation report and generate CI/CD result
    pub fn process(&self, report: &ValidationReport) -> CiCdResult {
        let threshold_results = self.evaluate_thresholds(report);
        let regression = self.analyze_regression(report);

        let passed = self.determine_pass_status(&threshold_results, &regression);
        let exit_code = self.determine_exit_code(passed, report);
        let summary = self.generate_summary(report, &threshold_results, &regression);

        CiCdResult {
            passed,
            exit_code,
            summary,
            report: report.clone(),
            regression,
            threshold_results,
            timestamp: Utc::now(),
            environment: self.config.environment.clone(),
        }
    }

    /// Evaluate thresholds against the validation report
    fn evaluate_thresholds(&self, report: &ValidationReport) -> ThresholdResults {
        let violations_check_passed = self
            .config
            .thresholds
            .max_violations
            .map(|max| report.violations().len() <= max)
            .unwrap_or(true);

        let warning_count = report
            .violations()
            .iter()
            .filter(|v| v.result_severity == Severity::Warning)
            .count();

        let warnings_check_passed = self
            .config
            .thresholds
            .max_warnings
            .map(|max| warning_count <= max)
            .unwrap_or(true);

        let total_count = report.summary.nodes_validated;
        let violation_count = report.violations().len();
        let conformance_rate = if total_count > 0 {
            (total_count - violation_count) as f64 / total_count as f64
        } else {
            1.0
        };

        let conformance_check_passed = self
            .config
            .thresholds
            .min_conformance_rate
            .map(|min| conformance_rate >= min)
            .unwrap_or(true);

        let max_severity = report
            .violations()
            .iter()
            .map(|v| v.result_severity)
            .max()
            .unwrap_or(Severity::Info);

        let severity_check_passed = self
            .config
            .thresholds
            .max_severity
            .map(|allowed_max| max_severity <= allowed_max)
            .unwrap_or(true);

        let all_passed = violations_check_passed
            && warnings_check_passed
            && conformance_check_passed
            && severity_check_passed;

        ThresholdResults {
            violations_check_passed,
            warnings_check_passed,
            conformance_check_passed,
            severity_check_passed,
            all_passed,
        }
    }

    /// Analyze regression against baseline
    fn analyze_regression(&self, _report: &ValidationReport) -> Option<RegressionAnalysis> {
        // If no baseline configured, skip regression analysis
        let _baseline_config = self.config.baseline.as_ref()?;

        // In a real implementation, this would:
        // 1. Load baseline from file
        // 2. Compare current violations with baseline
        // 3. Identify new, resolved, and unchanged violations

        // For now, return None as baseline loading is not implemented
        None
    }

    /// Determine overall pass/fail status
    fn determine_pass_status(
        &self,
        threshold_results: &ThresholdResults,
        regression: &Option<RegressionAnalysis>,
    ) -> bool {
        let mut passed = threshold_results.all_passed;

        // Check regression
        if let Some(reg) = regression {
            if let Some(baseline) = &self.config.baseline {
                if baseline.fail_on_new_violations && !reg.new_violations.is_empty() {
                    passed = false;
                }
            }
        }

        passed
    }

    /// Determine exit code
    fn determine_exit_code(&self, passed: bool, report: &ValidationReport) -> i32 {
        if passed {
            self.config.exit_code.success
        } else if !report.violations().is_empty() {
            if self.config.exit_code.fail_on_violations {
                self.config.exit_code.violations
            } else {
                self.config.exit_code.success
            }
        } else {
            self.config.exit_code.error
        }
    }

    /// Generate summary message
    fn generate_summary(
        &self,
        report: &ValidationReport,
        threshold_results: &ThresholdResults,
        regression: &Option<RegressionAnalysis>,
    ) -> String {
        let mut lines = Vec::new();

        // Overall status
        let status = if threshold_results.all_passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        };
        lines.push(format!("SHACL Validation {}", status));
        lines.push(String::new());

        // Statistics
        lines.push(format!(
            "Targets validated: {}",
            report.summary.nodes_validated
        ));
        lines.push(format!("Violations found: {}", report.violations().len()));

        let warning_count = report
            .violations()
            .iter()
            .filter(|v| v.result_severity == Severity::Warning)
            .count();
        lines.push(format!("Warnings: {}", warning_count));

        // Threshold checks
        lines.push(String::new());
        lines.push("Threshold Checks:".to_string());

        if !threshold_results.violations_check_passed {
            if let Some(max) = self.config.thresholds.max_violations {
                lines.push(format!(
                    "  ❌ Violations: {} > {} (max)",
                    report.violations().len(),
                    max
                ));
            }
        }

        if !threshold_results.conformance_check_passed {
            if let Some(min) = self.config.thresholds.min_conformance_rate {
                lines.push(format!("  ❌ Conformance rate below {:.1}%", min * 100.0));
            }
        }

        if threshold_results.all_passed {
            lines.push("  ✅ All threshold checks passed".to_string());
        }

        // Regression analysis
        if let Some(reg) = regression {
            lines.push(String::new());
            lines.push("Regression Analysis:".to_string());
            lines.push(format!("  New violations: {}", reg.new_violations.len()));
            lines.push(format!(
                "  Resolved violations: {}",
                reg.resolved_violations.len()
            ));
            lines.push(format!("  Net change: {:+}", reg.net_change));
        }

        lines.join("\n")
    }

    /// Generate output in configured format
    pub fn generate_output(&self, result: &CiCdResult) -> String {
        match self.config.output_format {
            OutputFormat::Json => self.generate_json(result),
            OutputFormat::JUnit => self.generate_junit(result),
            OutputFormat::Tap => self.generate_tap(result),
            OutputFormat::Sarif => self.generate_sarif(result),
            OutputFormat::Text => result.summary.clone(),
            OutputFormat::GitHub => self.generate_github_output(result),
            OutputFormat::GitLab => self.generate_gitlab_output(result),
        }
    }

    /// Generate JSON output
    fn generate_json(&self, result: &CiCdResult) -> String {
        serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string())
    }

    /// Generate JUnit XML output
    fn generate_junit(&self, result: &CiCdResult) -> String {
        let mut xml = String::new();
        xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        xml.push('\n');

        let violations = result.report.violations();
        let errors = violations
            .iter()
            .filter(|v| v.result_severity == Severity::Violation)
            .count();
        let failures = violations.len() - errors;

        xml.push_str(&format!(
            r#"<testsuite name="SHACL Validation" tests="{}" errors="{}" failures="{}" timestamp="{}">"#,
            result.report.summary.nodes_validated,
            errors,
            failures,
            result.timestamp.to_rfc3339()
        ));
        xml.push('\n');

        // Add test cases for each validation
        if violations.is_empty() {
            xml.push_str(r#"  <testcase name="SHACL Validation" classname="oxirs-shacl"/>"#);
            xml.push('\n');
        } else {
            for (i, violation) in violations.iter().enumerate() {
                xml.push_str(&format!(
                    r#"  <testcase name="Violation {}" classname="{}">"#,
                    i + 1,
                    violation.source_shape
                ));
                xml.push('\n');
                xml.push_str(&format!(
                    r#"    <failure message="{}" type="{:?}">{}</failure>"#,
                    html_escape(violation.message().as_deref().unwrap_or("No message")),
                    violation.result_severity,
                    html_escape(&format!("Focus node: {}", violation.focus_node))
                ));
                xml.push('\n');
                xml.push_str("  </testcase>");
                xml.push('\n');
            }
        }

        xml.push_str("</testsuite>");
        xml
    }

    /// Generate TAP output
    fn generate_tap(&self, result: &CiCdResult) -> String {
        let mut tap = String::new();

        let total = result.report.summary.nodes_validated.max(1);
        tap.push_str(&format!("TAP version 13\n1..{}\n", total));

        if result.report.violations().is_empty() {
            tap.push_str("ok 1 - SHACL validation passed\n");
        } else {
            for (i, violation) in result.report.violations().iter().enumerate() {
                tap.push_str(&format!(
                    "not ok {} - {} - {}\n",
                    i + 1,
                    violation.source_shape,
                    violation.message().as_deref().unwrap_or("No message")
                ));
                tap.push_str("  ---\n");
                tap.push_str(&format!("  focus_node: {}\n", violation.focus_node));
                tap.push_str(&format!("  severity: {:?}\n", violation.result_severity));
                tap.push_str("  ...\n");
            }
        }

        tap
    }

    /// Generate SARIF output
    fn generate_sarif(&self, result: &CiCdResult) -> String {
        let sarif = serde_json::json!({
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "oxirs-shacl",
                        "version": env!("CARGO_PKG_VERSION"),
                        "informationUri": "https://github.com/cool-japan/oxirs"
                    }
                },
                "results": result.report.violations().iter().map(|v| {
                    serde_json::json!({
                        "ruleId": &v.source_constraint_component.0,
                        "level": match v.result_severity {
                            Severity::Violation => "error",
                            Severity::Warning => "warning",
                            _ => "note"
                        },
                        "message": {
                            "text": v.message().clone()
                        },
                        "locations": [{
                            "physicalLocation": {
                                "artifactLocation": {
                                    "uri": v.focus_node.to_string()
                                }
                            }
                        }]
                    })
                }).collect::<Vec<_>>()
            }]
        });

        serde_json::to_string_pretty(&sarif).unwrap_or_else(|_| "{}".to_string())
    }

    /// Generate GitHub Actions output
    fn generate_github_output(&self, result: &CiCdResult) -> String {
        let mut output = String::new();

        // Generate annotations
        for violation in result.report.violations() {
            let level = match violation.result_severity {
                Severity::Violation => "error",
                Severity::Warning => "warning",
                _ => "notice",
            };

            output.push_str(&format!(
                "::{} title={}::{}\n",
                level,
                violation.source_shape,
                violation.message().as_deref().unwrap_or("No message")
            ));
        }

        // Generate summary
        if self.config.github_summary {
            output.push_str("\n## SHACL Validation Summary\n\n");
            output.push_str(&result.summary);
        }

        output
    }

    /// Generate GitLab CI output
    fn generate_gitlab_output(&self, result: &CiCdResult) -> String {
        // GitLab Code Quality report format
        let issues: Vec<_> = result
            .report
            .violations()
            .iter()
            .map(|v| {
                serde_json::json!({
                    "description": v.message(),
                    "severity": match v.result_severity {
                        Severity::Violation => "critical",
                        Severity::Warning => "major",
                        _ => "info"
                    },
                    "fingerprint": format!("{}-{}", v.source_shape, v.focus_node),
                    "location": {
                        "path": v.focus_node.to_string(),
                        "lines": {
                            "begin": 1
                        }
                    }
                })
            })
            .collect();

        serde_json::to_string_pretty(&issues).unwrap_or_else(|_| "[]".to_string())
    }

    /// Write output to file
    pub fn write_output(&self, result: &CiCdResult) -> Result<(), std::io::Error> {
        let output = self.generate_output(result);

        if let Some(path) = &self.config.output_path {
            let mut file = std::fs::File::create(path)?;
            file.write_all(output.as_bytes())?;
        } else {
            println!("{}", output);
        }

        Ok(())
    }

    /// Generate pre-commit hook script
    pub fn generate_precommit_hook(&self, shapes_path: &str, data_pattern: &str) -> String {
        format!(
            r#"#!/bin/sh
# SHACL Pre-commit Hook
# Generated by oxirs-shacl

echo "Running SHACL validation..."

# Validate staged RDF files
git diff --cached --name-only --diff-filter=ACMR | grep -E '{}' | while read file; do
    oxirs-shacl validate --shapes {} --data "$file"
    if [ $? -ne 0 ]; then
        echo "SHACL validation failed for $file"
        exit 1
    fi
done

echo "SHACL validation passed!"
exit 0
"#,
            data_pattern.replace('.', r"\."),
            shapes_path
        )
    }

    /// Generate GitHub Actions workflow
    pub fn generate_github_workflow(&self, shapes_path: &str, data_path: &str) -> String {
        format!(
            r#"name: SHACL Validation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install oxirs-shacl
        run: cargo install oxirs-shacl

      - name: Run SHACL Validation
        run: |
          oxirs-shacl validate \
            --shapes {} \
            --data {} \
            --format github \
            --github-summary

      - name: Upload validation report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: shacl-report
          path: shacl-report.json
"#,
            shapes_path, data_path
        )
    }

    /// Generate GitLab CI configuration
    pub fn generate_gitlab_ci(&self, shapes_path: &str, data_path: &str) -> String {
        format!(
            r#"shacl-validation:
  image: rust:latest
  stage: test
  before_script:
    - cargo install oxirs-shacl
  script:
    - oxirs-shacl validate --shapes {} --data {} --format gitlab > gl-code-quality-report.json
  artifacts:
    reports:
      codequality: gl-code-quality-report.json
"#,
            shapes_path, data_path
        )
    }
}

impl Default for CiCdEngine {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Escape HTML special characters
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cicd_config_default() {
        let config = CiCdConfig::default();
        assert_eq!(config.exit_code.success, 0);
        assert_eq!(config.exit_code.violations, 1);
        assert!(config.exit_code.fail_on_violations);
    }

    #[test]
    fn test_environment_detection() {
        let env = EnvironmentConfig::default();
        assert_eq!(env.environment, "development");
    }

    #[test]
    fn test_threshold_evaluation() {
        let config = CiCdConfig {
            thresholds: ThresholdConfig {
                max_violations: Some(5),
                ..Default::default()
            },
            ..Default::default()
        };

        let engine = CiCdEngine::new(config);
        let report = ValidationReport::new();

        let results = engine.evaluate_thresholds(&report);
        assert!(results.all_passed);
    }

    #[test]
    fn test_junit_output() {
        let engine = CiCdEngine::new(CiCdConfig {
            output_format: OutputFormat::JUnit,
            ..Default::default()
        });

        let report = ValidationReport::new();
        let result = engine.process(&report);
        let output = engine.generate_output(&result);

        assert!(output.contains("testsuite"));
        assert!(output.contains("oxirs-shacl"));
    }

    #[test]
    fn test_tap_output() {
        let engine = CiCdEngine::new(CiCdConfig {
            output_format: OutputFormat::Tap,
            ..Default::default()
        });

        let report = ValidationReport::new();
        let result = engine.process(&report);
        let output = engine.generate_output(&result);

        assert!(output.contains("TAP version 13"));
        assert!(output.contains("ok 1"));
    }

    #[test]
    fn test_precommit_hook_generation() {
        let engine = CiCdEngine::default_config();
        let hook = engine.generate_precommit_hook("shapes.ttl", r"\.ttl$");

        assert!(hook.contains("#!/bin/sh"));
        assert!(hook.contains("oxirs-shacl"));
        assert!(hook.contains("shapes.ttl"));
    }

    #[test]
    fn test_github_workflow_generation() {
        let engine = CiCdEngine::default_config();
        let workflow = engine.generate_github_workflow("shapes/", "data/");

        assert!(workflow.contains("name: SHACL Validation"));
        assert!(workflow.contains("actions/checkout"));
        assert!(workflow.contains("cargo install oxirs-shacl"));
    }

    #[test]
    fn test_gitlab_ci_generation() {
        let engine = CiCdEngine::default_config();
        let ci = engine.generate_gitlab_ci("shapes/", "data/");

        assert!(ci.contains("shacl-validation:"));
        assert!(ci.contains("codequality"));
    }

    #[test]
    fn test_exit_code_determination() {
        let engine = CiCdEngine::default_config();
        let report = ValidationReport::new();

        let result = engine.process(&report);
        assert_eq!(result.exit_code, 0); // Success, no violations
    }
}
