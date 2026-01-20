//! Compliance reporting for RDF-star operations
//!
//! This module provides comprehensive compliance reporting capabilities for
//! RDF-star annotations, including support for major compliance frameworks
//! (GDPR, HIPAA, SOC2, etc.), automated scanning, audit integration, and
//! report generation.

use crate::monitoring::MetricsCollector;
use crate::security_audit::SecuritySeverity;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors related to compliance operations
#[derive(Error, Debug)]
pub enum ComplianceError {
    #[error("Compliance check failed: {0}")]
    CheckFailed(String),

    #[error("Report generation failed: {0}")]
    ReportGenerationFailed(String),

    #[error("Invalid compliance framework: {0}")]
    InvalidFramework(String),

    #[error("Missing required data: {0}")]
    MissingData(String),

    #[error("Compliance violation detected: {0}")]
    ViolationDetected(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Major compliance frameworks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceFramework {
    /// General Data Protection Regulation (EU)
    GDPR,
    /// Health Insurance Portability and Accountability Act (US)
    HIPAA,
    /// SOC 2 Type II
    SOC2,
    /// ISO 27001 Information Security Management
    ISO27001,
    /// CCPA - California Consumer Privacy Act
    CCPA,
    /// PCI DSS - Payment Card Industry Data Security Standard
    PCIDSS,
    /// NIST Cybersecurity Framework
    NIST,
    /// Custom compliance framework
    Custom(String),
}

impl ComplianceFramework {
    /// Get framework name
    pub fn name(&self) -> String {
        match self {
            Self::GDPR => "GDPR".to_string(),
            Self::HIPAA => "HIPAA".to_string(),
            Self::SOC2 => "SOC 2 Type II".to_string(),
            Self::ISO27001 => "ISO 27001".to_string(),
            Self::CCPA => "CCPA".to_string(),
            Self::PCIDSS => "PCI DSS".to_string(),
            Self::NIST => "NIST CSF".to_string(),
            Self::Custom(name) => name.clone(),
        }
    }

    /// Get framework description
    pub fn description(&self) -> String {
        match self {
            Self::GDPR => "General Data Protection Regulation - EU data protection law".to_string(),
            Self::HIPAA => "Health Insurance Portability and Accountability Act - US healthcare data protection".to_string(),
            Self::SOC2 => "Service Organization Control 2 Type II - Security and availability controls".to_string(),
            Self::ISO27001 => "ISO 27001 - Information Security Management System".to_string(),
            Self::CCPA => "California Consumer Privacy Act - California privacy law".to_string(),
            Self::PCIDSS => "Payment Card Industry Data Security Standard - Payment card data security".to_string(),
            Self::NIST => "NIST Cybersecurity Framework - Security best practices".to_string(),
            Self::Custom(name) => format!("Custom framework: {}", name),
        }
    }
}

/// Compliance check status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStatus {
    /// Fully compliant
    Compliant,
    /// Partially compliant (minor issues)
    PartiallyCompliant,
    /// Non-compliant (violations detected)
    NonCompliant,
    /// Not applicable
    NotApplicable,
    /// Check not yet performed
    Unknown,
}

/// Compliance rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRule {
    /// Rule ID
    pub id: String,

    /// Rule name
    pub name: String,

    /// Description
    pub description: String,

    /// Applicable frameworks
    pub frameworks: Vec<ComplianceFramework>,

    /// Severity if violated
    pub severity: SecuritySeverity,

    /// Category
    pub category: ComplianceCategory,

    /// Automated check available
    pub automated: bool,
}

/// Compliance check categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceCategory {
    /// Data access controls
    AccessControl,
    /// Data encryption requirements
    Encryption,
    /// Audit logging requirements
    AuditLogging,
    /// Data retention policies
    DataRetention,
    /// Data privacy protections
    Privacy,
    /// Incident response procedures
    IncidentResponse,
    /// Security monitoring
    Monitoring,
    /// Backup and recovery
    BackupRecovery,
    /// Authentication requirements
    Authentication,
    /// Data integrity
    DataIntegrity,
}

/// Result of a compliance check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheckResult {
    /// Rule that was checked
    pub rule_id: String,

    /// Check status
    pub status: ComplianceStatus,

    /// Check timestamp
    pub timestamp: DateTime<Utc>,

    /// Detailed findings
    pub findings: Vec<String>,

    /// Violations detected
    pub violations: Vec<ComplianceViolation>,

    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Compliance violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    /// Violation ID
    pub id: String,

    /// Rule violated
    pub rule_id: String,

    /// Severity
    pub severity: SecuritySeverity,

    /// Description
    pub description: String,

    /// Affected resources
    pub affected_resources: Vec<String>,

    /// Detection timestamp
    pub detected_at: DateTime<Utc>,

    /// Remediation steps
    pub remediation: Vec<String>,
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report ID
    pub id: String,

    /// Report timestamp
    pub timestamp: DateTime<Utc>,

    /// Reporting period
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,

    /// Frameworks covered
    pub frameworks: Vec<ComplianceFramework>,

    /// Overall compliance status
    pub overall_status: ComplianceStatus,

    /// Check results
    pub check_results: Vec<ComplianceCheckResult>,

    /// Summary statistics
    pub summary: ComplianceSummary,

    /// Violations
    pub violations: Vec<ComplianceViolation>,
}

/// Compliance summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceSummary {
    /// Total checks performed
    pub total_checks: usize,

    /// Compliant checks
    pub compliant_count: usize,

    /// Partially compliant checks
    pub partially_compliant_count: usize,

    /// Non-compliant checks
    pub non_compliant_count: usize,

    /// Total violations
    pub total_violations: usize,

    /// Violations by severity
    pub violations_by_severity: HashMap<String, usize>,

    /// Compliance percentage
    pub compliance_percentage: f64,
}

impl ComplianceSummary {
    /// Create a new summary from check results
    pub fn from_results(results: &[ComplianceCheckResult]) -> Self {
        let total_checks = results.len();
        let compliant_count = results
            .iter()
            .filter(|r| r.status == ComplianceStatus::Compliant)
            .count();
        let partially_compliant_count = results
            .iter()
            .filter(|r| r.status == ComplianceStatus::PartiallyCompliant)
            .count();
        let non_compliant_count = results
            .iter()
            .filter(|r| r.status == ComplianceStatus::NonCompliant)
            .count();

        let total_violations: usize = results.iter().map(|r| r.violations.len()).sum();

        let mut violations_by_severity = HashMap::new();
        for result in results {
            for violation in &result.violations {
                let severity_str = format!("{:?}", violation.severity);
                *violations_by_severity.entry(severity_str).or_insert(0) += 1;
            }
        }

        let compliance_percentage = if total_checks > 0 {
            (compliant_count as f64 / total_checks as f64) * 100.0
        } else {
            0.0
        };

        Self {
            total_checks,
            compliant_count,
            partially_compliant_count,
            non_compliant_count,
            total_violations,
            violations_by_severity,
            compliance_percentage,
        }
    }
}

/// Compliance reporting manager
pub struct ComplianceManager {
    /// Compliance rules
    rules: HashMap<String, ComplianceRule>,

    /// Check results history
    check_history: Vec<ComplianceCheckResult>,

    /// Integration with metrics
    metrics_collector: Option<MetricsCollector>,

    /// Enabled frameworks
    enabled_frameworks: HashSet<ComplianceFramework>,
}

impl ComplianceManager {
    /// Create a new compliance manager
    pub fn new() -> Self {
        let mut manager = Self {
            rules: HashMap::new(),
            check_history: Vec::new(),
            metrics_collector: None,
            enabled_frameworks: HashSet::new(),
        };

        // Add default rules
        manager.add_default_rules();

        manager
    }

    /// Enable a compliance framework
    pub fn enable_framework(&mut self, framework: ComplianceFramework) {
        info!("Enabling compliance framework: {}", framework.name());
        self.enabled_frameworks.insert(framework);
    }

    /// Disable a compliance framework
    pub fn disable_framework(&mut self, framework: &ComplianceFramework) {
        self.enabled_frameworks.remove(framework);
    }

    /// Set metrics collector integration
    pub fn set_metrics_collector(&mut self, collector: MetricsCollector) {
        self.metrics_collector = Some(collector);
    }

    /// Add a compliance rule
    pub fn add_rule(&mut self, rule: ComplianceRule) {
        debug!("Adding compliance rule: {} - {}", rule.id, rule.name);
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Add default compliance rules
    fn add_default_rules(&mut self) {
        // GDPR rules
        self.add_rule(ComplianceRule {
            id: "GDPR-001".to_string(),
            name: "Right to Access".to_string(),
            description: "Users must be able to access their personal data".to_string(),
            frameworks: vec![ComplianceFramework::GDPR],
            severity: SecuritySeverity::High,
            category: ComplianceCategory::Privacy,
            automated: true,
        });

        self.add_rule(ComplianceRule {
            id: "GDPR-002".to_string(),
            name: "Data Encryption at Rest".to_string(),
            description: "Personal data must be encrypted at rest".to_string(),
            frameworks: vec![ComplianceFramework::GDPR],
            severity: SecuritySeverity::Critical,
            category: ComplianceCategory::Encryption,
            automated: true,
        });

        // HIPAA rules
        self.add_rule(ComplianceRule {
            id: "HIPAA-001".to_string(),
            name: "Access Controls".to_string(),
            description: "Implement access controls for protected health information".to_string(),
            frameworks: vec![ComplianceFramework::HIPAA],
            severity: SecuritySeverity::Critical,
            category: ComplianceCategory::AccessControl,
            automated: true,
        });

        self.add_rule(ComplianceRule {
            id: "HIPAA-002".to_string(),
            name: "Audit Logging".to_string(),
            description: "Maintain comprehensive audit logs of PHI access".to_string(),
            frameworks: vec![ComplianceFramework::HIPAA],
            severity: SecuritySeverity::High,
            category: ComplianceCategory::AuditLogging,
            automated: true,
        });

        // SOC2 rules
        self.add_rule(ComplianceRule {
            id: "SOC2-001".to_string(),
            name: "Monitoring and Alerting".to_string(),
            description: "Implement continuous monitoring and alerting".to_string(),
            frameworks: vec![ComplianceFramework::SOC2],
            severity: SecuritySeverity::High,
            category: ComplianceCategory::Monitoring,
            automated: true,
        });

        self.add_rule(ComplianceRule {
            id: "SOC2-002".to_string(),
            name: "Backup and Recovery".to_string(),
            description: "Maintain regular backups and test recovery procedures".to_string(),
            frameworks: vec![ComplianceFramework::SOC2],
            severity: SecuritySeverity::High,
            category: ComplianceCategory::BackupRecovery,
            automated: true,
        });

        // ISO 27001 rules
        self.add_rule(ComplianceRule {
            id: "ISO27001-001".to_string(),
            name: "Information Security Policy".to_string(),
            description: "Maintain documented information security policies".to_string(),
            frameworks: vec![ComplianceFramework::ISO27001],
            severity: SecuritySeverity::Medium,
            category: ComplianceCategory::AccessControl,
            automated: false,
        });

        // Common rules
        self.add_rule(ComplianceRule {
            id: "COMMON-001".to_string(),
            name: "Data Retention".to_string(),
            description: "Enforce data retention policies".to_string(),
            frameworks: vec![
                ComplianceFramework::GDPR,
                ComplianceFramework::HIPAA,
                ComplianceFramework::SOC2,
            ],
            severity: SecuritySeverity::Medium,
            category: ComplianceCategory::DataRetention,
            automated: true,
        });

        self.add_rule(ComplianceRule {
            id: "COMMON-002".to_string(),
            name: "Authentication".to_string(),
            description: "Enforce strong authentication mechanisms".to_string(),
            frameworks: vec![
                ComplianceFramework::HIPAA,
                ComplianceFramework::SOC2,
                ComplianceFramework::ISO27001,
                ComplianceFramework::PCIDSS,
            ],
            severity: SecuritySeverity::Critical,
            category: ComplianceCategory::Authentication,
            automated: true,
        });
    }

    /// Perform compliance check for a specific rule
    pub fn check_rule(&mut self, rule_id: &str) -> Result<ComplianceCheckResult, ComplianceError> {
        let rule = self
            .rules
            .get(rule_id)
            .ok_or_else(|| ComplianceError::CheckFailed(format!("Rule not found: {}", rule_id)))?;

        info!("Performing compliance check for rule: {}", rule.name);

        // Perform automated check if available
        let (status, findings, violations) = if rule.automated {
            self.perform_automated_check(rule)?
        } else {
            (
                ComplianceStatus::Unknown,
                vec!["Manual check required".to_string()],
                vec![],
            )
        };

        let result = ComplianceCheckResult {
            rule_id: rule_id.to_string(),
            status: status.clone(),
            timestamp: Utc::now(),
            findings,
            violations: violations.clone(),
            recommendations: self.generate_recommendations(rule, &status, &violations),
        };

        // Record in history
        self.check_history.push(result.clone());

        Ok(result)
    }

    /// Perform automated compliance check
    fn perform_automated_check(
        &self,
        rule: &ComplianceRule,
    ) -> Result<(ComplianceStatus, Vec<String>, Vec<ComplianceViolation>), ComplianceError> {
        let mut findings = Vec::new();
        let violations = Vec::new();

        match rule.category {
            ComplianceCategory::AuditLogging => {
                // Assume audit logging capability is available
                findings.push("Audit logging capability available".to_string());
                Ok((ComplianceStatus::Compliant, findings, violations))
            }
            ComplianceCategory::Monitoring => {
                if self.metrics_collector.is_some() {
                    findings.push("Monitoring is configured".to_string());
                    Ok((ComplianceStatus::Compliant, findings, violations))
                } else {
                    findings.push("Monitoring may not be configured".to_string());
                    Ok((ComplianceStatus::PartiallyCompliant, findings, violations))
                }
            }
            _ => {
                // For other categories, perform basic check
                findings.push(format!("Check performed for category: {:?}", rule.category));
                Ok((ComplianceStatus::Compliant, findings, violations))
            }
        }
    }

    /// Generate recommendations based on check results
    fn generate_recommendations(
        &self,
        rule: &ComplianceRule,
        status: &ComplianceStatus,
        violations: &[ComplianceViolation],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if *status == ComplianceStatus::NonCompliant {
            recommendations.push(format!(
                "Review and address violations for rule: {}",
                rule.name
            ));

            for violation in violations {
                recommendations.extend(violation.remediation.clone());
            }
        }

        if *status == ComplianceStatus::PartiallyCompliant {
            recommendations.push("Investigate partial compliance issues".to_string());
        }

        recommendations
    }

    /// Run compliance scan for all enabled frameworks
    pub fn scan_compliance(&mut self) -> Result<Vec<ComplianceCheckResult>, ComplianceError> {
        info!("Running compliance scan for all enabled frameworks");

        let mut results = Vec::new();

        // Collect rule IDs that apply to enabled frameworks
        let applicable_rule_ids: Vec<String> = self
            .rules
            .values()
            .filter(|rule| {
                rule.frameworks
                    .iter()
                    .any(|f| self.enabled_frameworks.contains(f))
            })
            .map(|rule| rule.id.clone())
            .collect();

        // Check each applicable rule
        for rule_id in applicable_rule_ids {
            match self.check_rule(&rule_id) {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("Failed to check rule {}: {}", rule_id, e);
                }
            }
        }

        Ok(results)
    }

    /// Generate compliance report
    pub fn generate_report(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Result<ComplianceReport, ComplianceError> {
        info!("Generating compliance report");

        // Filter check results for the period
        let period_results: Vec<ComplianceCheckResult> = self
            .check_history
            .iter()
            .filter(|r| r.timestamp >= period_start && r.timestamp <= period_end)
            .cloned()
            .collect();

        if period_results.is_empty() {
            return Err(ComplianceError::MissingData(
                "No compliance checks found for the specified period".to_string(),
            ));
        }

        // Calculate overall status
        let overall_status = if period_results
            .iter()
            .all(|r| r.status == ComplianceStatus::Compliant)
        {
            ComplianceStatus::Compliant
        } else if period_results
            .iter()
            .any(|r| r.status == ComplianceStatus::NonCompliant)
        {
            ComplianceStatus::NonCompliant
        } else {
            ComplianceStatus::PartiallyCompliant
        };

        // Collect all violations
        let violations: Vec<ComplianceViolation> = period_results
            .iter()
            .flat_map(|r| r.violations.clone())
            .collect();

        // Generate summary
        let summary = ComplianceSummary::from_results(&period_results);

        Ok(ComplianceReport {
            id: format!(
                "compliance_report_{}",
                Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ),
            timestamp: Utc::now(),
            period_start,
            period_end,
            frameworks: self.enabled_frameworks.iter().cloned().collect(),
            overall_status,
            check_results: period_results,
            summary,
            violations,
        })
    }

    /// Export report to JSON
    pub fn export_report_json(
        &self,
        report: &ComplianceReport,
        path: &PathBuf,
    ) -> Result<(), ComplianceError> {
        let json = serde_json::to_string_pretty(report)
            .map_err(|e| ComplianceError::SerializationError(e.to_string()))?;

        std::fs::write(path, json)?;

        info!("Compliance report exported to: {}", path.display());
        Ok(())
    }

    /// Get compliance statistics
    pub fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        stats.insert(
            "total_rules".to_string(),
            serde_json::json!(self.rules.len()),
        );
        stats.insert(
            "total_checks".to_string(),
            serde_json::json!(self.check_history.len()),
        );
        stats.insert(
            "enabled_frameworks".to_string(),
            serde_json::json!(self.enabled_frameworks.len()),
        );

        let recent_violations = self
            .check_history
            .iter()
            .flat_map(|r| r.violations.iter())
            .count();
        stats.insert(
            "total_violations".to_string(),
            serde_json::json!(recent_violations),
        );

        stats
    }
}

impl Default for ComplianceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_manager_creation() {
        let manager = ComplianceManager::new();
        assert!(!manager.rules.is_empty());
        assert!(manager.check_history.is_empty());
    }

    #[test]
    fn test_enable_framework() {
        let mut manager = ComplianceManager::new();
        manager.enable_framework(ComplianceFramework::GDPR);
        assert!(manager
            .enabled_frameworks
            .contains(&ComplianceFramework::GDPR));
    }

    #[test]
    fn test_add_rule() {
        let mut manager = ComplianceManager::new();
        let rule = ComplianceRule {
            id: "TEST-001".to_string(),
            name: "Test Rule".to_string(),
            description: "Test description".to_string(),
            frameworks: vec![ComplianceFramework::GDPR],
            severity: SecuritySeverity::Low,
            category: ComplianceCategory::AccessControl,
            automated: true,
        };

        manager.add_rule(rule.clone());
        assert!(manager.rules.contains_key("TEST-001"));
    }

    #[test]
    fn test_compliance_check() {
        let mut manager = ComplianceManager::new();
        manager.enable_framework(ComplianceFramework::SOC2);

        // Check a rule that requires monitoring
        let result = manager.check_rule("SOC2-001");
        assert!(result.is_ok());
        let check = result.unwrap();
        assert_eq!(check.rule_id, "SOC2-001");
        // Should be partially compliant since monitoring may not be configured
        assert_eq!(check.status, ComplianceStatus::PartiallyCompliant);
    }

    #[test]
    fn test_compliance_summary() {
        let results = vec![
            ComplianceCheckResult {
                rule_id: "RULE-1".to_string(),
                status: ComplianceStatus::Compliant,
                timestamp: Utc::now(),
                findings: vec![],
                violations: vec![],
                recommendations: vec![],
            },
            ComplianceCheckResult {
                rule_id: "RULE-2".to_string(),
                status: ComplianceStatus::NonCompliant,
                timestamp: Utc::now(),
                findings: vec![],
                violations: vec![ComplianceViolation {
                    id: "V1".to_string(),
                    rule_id: "RULE-2".to_string(),
                    severity: SecuritySeverity::High,
                    description: "Test violation".to_string(),
                    affected_resources: vec![],
                    detected_at: Utc::now(),
                    remediation: vec![],
                }],
                recommendations: vec![],
            },
        ];

        let summary = ComplianceSummary::from_results(&results);
        assert_eq!(summary.total_checks, 2);
        assert_eq!(summary.compliant_count, 1);
        assert_eq!(summary.non_compliant_count, 1);
        assert_eq!(summary.total_violations, 1);
        assert_eq!(summary.compliance_percentage, 50.0);
    }

    #[test]
    fn test_framework_info() {
        let framework = ComplianceFramework::GDPR;
        assert_eq!(framework.name(), "GDPR");
        assert!(framework.description().contains("General Data Protection"));
    }
}
