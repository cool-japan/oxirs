//! Diagnostic data types: levels, severities, results, reports, and repair recommendations.
//!
//! This module defines the core data structures returned by the diagnostic engine:
//! - [`DiagnosticLevel`] and [`Severity`] enums
//! - [`DiagnosticResult`] (per-check result with builder API)
//! - [`DiagnosticReport`] (aggregated report) with [`DiagnosticSummary`] and [`HealthStatus`]
//! - [`RepairRecommendation`] and [`RepairAction`] derived from a report

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Diagnostic level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticLevel {
    /// Quick diagnostics (< 1 second)
    Quick,
    /// Standard diagnostics (1-5 seconds)
    Standard,
    /// Deep diagnostics (5+ seconds, may impact performance)
    Deep,
}

/// Diagnostic severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Informational
    Info,
    /// Warning (potential issue)
    Warning,
    /// Error (needs attention)
    Error,
    /// Critical (immediate action required)
    Critical,
}

/// Diagnostic result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    /// Diagnostic category
    pub category: String,
    /// Diagnostic name
    pub name: String,
    /// Severity level
    pub severity: Severity,
    /// Description
    pub description: String,
    /// Recommended action (if any)
    pub recommendation: Option<String>,
    /// Additional details
    pub details: HashMap<String, String>,
}

impl DiagnosticResult {
    /// Create a new diagnostic result
    pub fn new(category: impl Into<String>, name: impl Into<String>, severity: Severity) -> Self {
        Self {
            category: category.into(),
            name: name.into(),
            severity,
            description: String::new(),
            recommendation: None,
            details: HashMap::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set recommendation
    pub fn with_recommendation(mut self, recommendation: impl Into<String>) -> Self {
        self.recommendation = Some(recommendation.into());
        self
    }

    /// Add a detail
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }
}

/// Comprehensive diagnostic report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    /// Timestamp
    pub timestamp: String,
    /// Diagnostic level used
    pub level: DiagnosticLevel,
    /// Time taken to generate report
    pub duration: Duration,
    /// Overall health status
    pub health_status: HealthStatus,
    /// Individual diagnostic results
    pub results: Vec<DiagnosticResult>,
    /// Summary statistics
    pub summary: DiagnosticSummary,
}

impl DiagnosticReport {
    /// Get results by severity
    pub fn results_by_severity(&self, severity: Severity) -> Vec<&DiagnosticResult> {
        self.results
            .iter()
            .filter(|r| r.severity == severity)
            .collect()
    }

    /// Get critical issues
    pub fn critical_issues(&self) -> Vec<&DiagnosticResult> {
        self.results_by_severity(Severity::Critical)
    }

    /// Get errors
    pub fn errors(&self) -> Vec<&DiagnosticResult> {
        self.results_by_severity(Severity::Error)
    }

    /// Get warnings
    pub fn warnings(&self) -> Vec<&DiagnosticResult> {
        self.results_by_severity(Severity::Warning)
    }

    /// Check if there are any critical issues
    pub fn has_critical_issues(&self) -> bool {
        !self.critical_issues().is_empty()
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        !self.errors().is_empty()
    }

    /// Generate repair recommendations based on diagnostic results
    pub fn repair_recommendations(&self) -> Vec<RepairRecommendation> {
        let mut recommendations = Vec::new();

        for result in &self.results {
            // Check for index inconsistencies
            if result.name.contains("Index") && result.severity >= Severity::Error {
                recommendations.push(RepairRecommendation {
                    issue: result.description.clone(),
                    severity: result.severity,
                    action: RepairAction::RebuildIndexes,
                    estimated_time: "5-30 minutes".to_string(),
                    risk_level: "Medium".to_string(),
                });
            }

            // Check for dictionary issues
            if result.name.contains("Dictionary") && result.severity == Severity::Critical {
                recommendations.push(RepairRecommendation {
                    issue: result.description.clone(),
                    severity: result.severity,
                    action: RepairAction::RestoreFromBackup,
                    estimated_time: "Variable".to_string(),
                    risk_level: "High".to_string(),
                });
            }

            // Check for storage efficiency issues
            if result.name.contains("Storage")
                && result.severity >= Severity::Warning
                && (result.description.contains("overhead") || result.description.contains("bloat"))
            {
                recommendations.push(RepairRecommendation {
                    issue: result.description.clone(),
                    severity: result.severity,
                    action: RepairAction::CompactDatabase,
                    estimated_time: "10-60 minutes".to_string(),
                    risk_level: "Low".to_string(),
                });
            }

            // Check for WAL issues
            if result.name.contains("WAL") && result.severity >= Severity::Warning {
                // Check for large WAL files (case insensitive)
                let desc_lower = result.description.to_lowercase();
                let name_lower = result.name.to_lowercase();
                if desc_lower.contains("large") || name_lower.contains("large") {
                    recommendations.push(RepairRecommendation {
                        issue: result.description.clone(),
                        severity: result.severity,
                        action: RepairAction::CheckpointWal,
                        estimated_time: "1-5 minutes".to_string(),
                        risk_level: "Low".to_string(),
                    });
                }
            }

            // Check for dictionary bloat
            if result.name.contains("Orphaned") {
                recommendations.push(RepairRecommendation {
                    issue: result.description.clone(),
                    severity: result.severity,
                    action: RepairAction::VacuumDictionary,
                    estimated_time: "5-20 minutes".to_string(),
                    risk_level: "Low".to_string(),
                });
            }
        }

        recommendations
    }
}

/// Overall health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All systems operating normally
    Healthy,
    /// Minor issues detected
    Degraded,
    /// Serious issues requiring attention
    Unhealthy,
    /// Critical issues requiring immediate action
    Critical,
}

/// Diagnostic summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticSummary {
    /// Number of info messages
    pub info_count: usize,
    /// Number of warnings
    pub warning_count: usize,
    /// Number of errors
    pub error_count: usize,
    /// Number of critical issues
    pub critical_count: usize,
}

impl DiagnosticSummary {
    pub(crate) fn from_results(results: &[DiagnosticResult]) -> Self {
        let mut info_count = 0;
        let mut warning_count = 0;
        let mut error_count = 0;
        let mut critical_count = 0;

        for result in results {
            match result.severity {
                Severity::Info => info_count += 1,
                Severity::Warning => warning_count += 1,
                Severity::Error => error_count += 1,
                Severity::Critical => critical_count += 1,
            }
        }

        Self {
            info_count,
            warning_count,
            error_count,
            critical_count,
        }
    }
}

/// Repair recommendation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairRecommendation {
    /// Issue to repair
    pub issue: String,
    /// Severity of the issue
    pub severity: Severity,
    /// Repair action
    pub action: RepairAction,
    /// Estimated time
    pub estimated_time: String,
    /// Risk level (Low, Medium, High)
    pub risk_level: String,
}

/// Repair action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RepairAction {
    /// Rebuild indexes
    RebuildIndexes,
    /// Compact database
    CompactDatabase,
    /// Checkpoint WAL
    CheckpointWal,
    /// Vacuum dictionary
    VacuumDictionary,
    /// Restore from backup
    RestoreFromBackup,
    /// Manual intervention required
    ManualIntervention,
}
