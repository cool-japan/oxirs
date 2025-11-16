//! Advanced diagnostic tools for TDB store
//!
//! This module provides comprehensive diagnostic capabilities:
//! - System health checks
//! - Performance analysis
//! - Storage diagnostics
//! - Index consistency checks
//! - Memory usage analysis

use crate::error::Result;
use crate::storage::BufferPoolStats;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

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
    fn from_results(results: &[DiagnosticResult]) -> Self {
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

/// Diagnostic engine
pub struct DiagnosticEngine {
    /// Registered diagnostic checks
    checks: Vec<Box<dyn DiagnosticCheck + Send + Sync>>,
}

/// Trait for diagnostic checks
pub trait DiagnosticCheck: Send + Sync {
    /// Get check name
    fn name(&self) -> &str;

    /// Get check category
    fn category(&self) -> &str;

    /// Minimum level required to run this check
    fn min_level(&self) -> DiagnosticLevel;

    /// Run the diagnostic check
    fn run(&self, context: &DiagnosticContext) -> Result<Vec<DiagnosticResult>>;
}

/// Context for running diagnostics
pub struct DiagnosticContext {
    /// Total number of triples
    pub triple_count: u64,
    /// Buffer pool statistics
    pub buffer_pool_stats: BufferPoolStats,
    /// Dictionary size
    pub dictionary_size: u64,
    /// Total storage size (bytes)
    pub storage_size_bytes: u64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
}

impl DiagnosticEngine {
    /// Create a new diagnostic engine
    pub fn new() -> Self {
        let mut engine = Self { checks: Vec::new() };

        // Register built-in checks
        engine.register(Box::new(BufferPoolCheck));
        engine.register(Box::new(MemoryUsageCheck));
        engine.register(Box::new(StorageEfficiencyCheck));
        engine.register(Box::new(PerformanceCheck));

        engine
    }

    /// Register a diagnostic check
    pub fn register(&mut self, check: Box<dyn DiagnosticCheck + Send + Sync>) {
        self.checks.push(check);
    }

    /// Run diagnostics
    pub fn run(&self, level: DiagnosticLevel, context: &DiagnosticContext) -> DiagnosticReport {
        let start = Instant::now();
        let mut results = Vec::new();

        // Run all applicable checks
        for check in &self.checks {
            if Self::should_run_check(check.min_level(), level) {
                match check.run(context) {
                    Ok(mut check_results) => {
                        results.append(&mut check_results);
                    }
                    Err(e) => {
                        results.push(
                            DiagnosticResult::new(check.category(), check.name(), Severity::Error)
                                .with_description(format!("Diagnostic check failed: {}", e)),
                        );
                    }
                }
            }
        }

        let summary = DiagnosticSummary::from_results(&results);

        // Determine overall health status
        let health_status = if summary.critical_count > 0 {
            HealthStatus::Critical
        } else if summary.error_count > 0 {
            HealthStatus::Unhealthy
        } else if summary.warning_count > 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        DiagnosticReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            level,
            duration: start.elapsed(),
            health_status,
            results,
            summary,
        }
    }

    fn should_run_check(check_level: DiagnosticLevel, requested_level: DiagnosticLevel) -> bool {
        matches!(
            (check_level, requested_level),
            (DiagnosticLevel::Quick, _)
                | (
                    DiagnosticLevel::Standard,
                    DiagnosticLevel::Standard | DiagnosticLevel::Deep
                )
                | (DiagnosticLevel::Deep, DiagnosticLevel::Deep)
        )
    }
}

impl Default for DiagnosticEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Built-in diagnostic checks

/// Buffer pool diagnostic check
struct BufferPoolCheck;

impl DiagnosticCheck for BufferPoolCheck {
    fn name(&self) -> &str {
        "Buffer Pool Health"
    }

    fn category(&self) -> &str {
        "Performance"
    }

    fn min_level(&self) -> DiagnosticLevel {
        DiagnosticLevel::Quick
    }

    fn run(&self, context: &DiagnosticContext) -> Result<Vec<DiagnosticResult>> {
        let mut results = Vec::new();
        let hit_rate = context.buffer_pool_stats.hit_rate();

        // Check hit rate
        if hit_rate < 0.5 {
            results.push(
                DiagnosticResult::new(
                    self.category(),
                    "Low Buffer Pool Hit Rate",
                    Severity::Warning,
                )
                .with_description(format!(
                    "Buffer pool hit rate is {:.2}%, which is below optimal (>90%)",
                    hit_rate * 100.0
                ))
                .with_recommendation(
                    "Consider increasing buffer pool size or analyzing query patterns",
                )
                .with_detail("current_hit_rate", format!("{:.2}%", hit_rate * 100.0))
                .with_detail("target_hit_rate", "90%"),
            );
        } else if hit_rate >= 0.9 {
            results.push(
                DiagnosticResult::new(
                    self.category(),
                    "Optimal Buffer Pool Performance",
                    Severity::Info,
                )
                .with_description(format!("Buffer pool hit rate is {:.2}%", hit_rate * 100.0))
                .with_detail("hit_rate", format!("{:.2}%", hit_rate * 100.0)),
            );
        }

        Ok(results)
    }
}

/// Memory usage diagnostic check
struct MemoryUsageCheck;

impl DiagnosticCheck for MemoryUsageCheck {
    fn name(&self) -> &str {
        "Memory Usage"
    }

    fn category(&self) -> &str {
        "Resources"
    }

    fn min_level(&self) -> DiagnosticLevel {
        DiagnosticLevel::Quick
    }

    fn run(&self, context: &DiagnosticContext) -> Result<Vec<DiagnosticResult>> {
        let mut results = Vec::new();
        let memory_mb = context.memory_usage_bytes / (1024 * 1024);

        // Check if memory usage is excessive (> 1GB)
        if memory_mb > 1024 {
            results.push(
                DiagnosticResult::new(self.category(), "High Memory Usage", Severity::Warning)
                    .with_description(format!("Memory usage is {} MB", memory_mb))
                    .with_recommendation(
                        "Monitor memory growth and consider reducing buffer pool size",
                    )
                    .with_detail("memory_usage_mb", memory_mb.to_string()),
            );
        } else {
            results.push(
                DiagnosticResult::new(self.category(), "Normal Memory Usage", Severity::Info)
                    .with_description(format!("Memory usage is {} MB", memory_mb))
                    .with_detail("memory_usage_mb", memory_mb.to_string()),
            );
        }

        Ok(results)
    }
}

/// Storage efficiency diagnostic check
struct StorageEfficiencyCheck;

impl DiagnosticCheck for StorageEfficiencyCheck {
    fn name(&self) -> &str {
        "Storage Efficiency"
    }

    fn category(&self) -> &str {
        "Storage"
    }

    fn min_level(&self) -> DiagnosticLevel {
        DiagnosticLevel::Standard
    }

    fn run(&self, context: &DiagnosticContext) -> Result<Vec<DiagnosticResult>> {
        let mut results = Vec::new();

        // Calculate storage per triple
        if context.triple_count > 0 {
            let bytes_per_triple = context.storage_size_bytes / context.triple_count;

            if bytes_per_triple > 500 {
                results.push(
                    DiagnosticResult::new(self.category(), "Storage Overhead", Severity::Warning)
                        .with_description(format!(
                            "Storage overhead is {} bytes per triple (expected ~100-200)",
                            bytes_per_triple
                        ))
                        .with_recommendation("Consider running compaction or enabling compression")
                        .with_detail("bytes_per_triple", bytes_per_triple.to_string()),
                );
            } else {
                results.push(
                    DiagnosticResult::new(self.category(), "Efficient Storage", Severity::Info)
                        .with_description(format!(
                            "Storage efficiency: {} bytes per triple",
                            bytes_per_triple
                        ))
                        .with_detail("bytes_per_triple", bytes_per_triple.to_string()),
                );
            }
        }

        Ok(results)
    }
}

/// Performance diagnostic check
struct PerformanceCheck;

impl DiagnosticCheck for PerformanceCheck {
    fn name(&self) -> &str {
        "Performance Metrics"
    }

    fn category(&self) -> &str {
        "Performance"
    }

    fn min_level(&self) -> DiagnosticLevel {
        DiagnosticLevel::Standard
    }

    fn run(&self, context: &DiagnosticContext) -> Result<Vec<DiagnosticResult>> {
        let mut results = Vec::new();

        use std::sync::atomic::Ordering;

        let evictions = context.buffer_pool_stats.evictions.load(Ordering::Relaxed);
        let total_fetches = context
            .buffer_pool_stats
            .total_fetches
            .load(Ordering::Relaxed);

        // Check eviction rate
        if total_fetches > 0 {
            let eviction_rate = (evictions as f64 / total_fetches as f64) * 100.0;

            if eviction_rate > 10.0 {
                results.push(
                    DiagnosticResult::new(self.category(), "High Eviction Rate", Severity::Warning)
                        .with_description(format!(
                            "Buffer pool eviction rate is {:.2}% (target <5%)",
                            eviction_rate
                        ))
                        .with_recommendation("Increase buffer pool size to reduce evictions")
                        .with_detail("eviction_rate", format!("{:.2}%", eviction_rate)),
                );
            } else {
                results.push(
                    DiagnosticResult::new(self.category(), "Normal Eviction Rate", Severity::Info)
                        .with_description(format!("Eviction rate: {:.2}%", eviction_rate))
                        .with_detail("eviction_rate", format!("{:.2}%", eviction_rate)),
                );
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn create_test_context() -> DiagnosticContext {
        DiagnosticContext {
            triple_count: 1000,
            buffer_pool_stats: BufferPoolStats::default(),
            dictionary_size: 5000,
            storage_size_bytes: 200_000,
            memory_usage_bytes: 50_000_000, // 50MB
        }
    }

    #[test]
    fn test_diagnostic_engine_creation() {
        let engine = DiagnosticEngine::new();
        assert!(!engine.checks.is_empty());
    }

    #[test]
    fn test_run_quick_diagnostics() {
        let engine = DiagnosticEngine::new();
        let context = create_test_context();

        let report = engine.run(DiagnosticLevel::Quick, &context);

        assert!(!report.results.is_empty());
        assert_eq!(report.level, DiagnosticLevel::Quick);
    }

    #[test]
    fn test_run_standard_diagnostics() {
        let engine = DiagnosticEngine::new();
        let context = create_test_context();

        let report = engine.run(DiagnosticLevel::Standard, &context);

        assert!(!report.results.is_empty());
        // Standard should run more checks than quick
        let standard_count = report.results.len();
        assert!(standard_count > 0);
    }

    #[test]
    fn test_health_status_determination() {
        let engine = DiagnosticEngine::new();
        let context = create_test_context();

        // Set low hit rate to trigger warnings
        context
            .buffer_pool_stats
            .total_fetches
            .store(1000, std::sync::atomic::Ordering::Relaxed);
        context
            .buffer_pool_stats
            .cache_hits
            .store(400, std::sync::atomic::Ordering::Relaxed);

        let report = engine.run(DiagnosticLevel::Quick, &context);

        // Should have some warnings
        assert!(!report.warnings().is_empty());
    }

    #[test]
    fn test_diagnostic_result_builder() {
        let result = DiagnosticResult::new("Test", "Check", Severity::Warning)
            .with_description("Test description")
            .with_recommendation("Fix it")
            .with_detail("key", "value");

        assert_eq!(result.category, "Test");
        assert_eq!(result.name, "Check");
        assert_eq!(result.severity, Severity::Warning);
        assert_eq!(result.description, "Test description");
        assert_eq!(result.recommendation, Some("Fix it".to_string()));
        assert_eq!(result.details.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_results_by_severity() {
        let engine = DiagnosticEngine::new();
        let context = create_test_context();

        let report = engine.run(DiagnosticLevel::Standard, &context);

        let warnings = report.results_by_severity(Severity::Warning);
        let info = report.results_by_severity(Severity::Info);

        // Should have some results
        assert!(warnings.len() + info.len() > 0);
    }

    #[test]
    fn test_diagnostic_summary() {
        let results = vec![
            DiagnosticResult::new("Cat1", "Check1", Severity::Info),
            DiagnosticResult::new("Cat2", "Check2", Severity::Warning),
            DiagnosticResult::new("Cat3", "Check3", Severity::Error),
            DiagnosticResult::new("Cat4", "Check4", Severity::Critical),
        ];

        let summary = DiagnosticSummary::from_results(&results);

        assert_eq!(summary.info_count, 1);
        assert_eq!(summary.warning_count, 1);
        assert_eq!(summary.error_count, 1);
        assert_eq!(summary.critical_count, 1);
    }

    #[test]
    fn test_buffer_pool_check() {
        let check = BufferPoolCheck;
        let context = create_test_context();

        // Set up low hit rate
        context
            .buffer_pool_stats
            .total_fetches
            .store(1000, std::sync::atomic::Ordering::Relaxed);
        context
            .buffer_pool_stats
            .cache_hits
            .store(400, std::sync::atomic::Ordering::Relaxed);

        let results = check.run(&context).unwrap();

        // Should detect low hit rate
        assert!(!results.is_empty());
        assert_eq!(results[0].severity, Severity::Warning);
    }

    #[test]
    fn test_memory_usage_check() {
        let check = MemoryUsageCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_storage_efficiency_check() {
        let check = StorageEfficiencyCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_performance_check() {
        let check = PerformanceCheck;
        let context = create_test_context();

        context
            .buffer_pool_stats
            .total_fetches
            .store(1000, std::sync::atomic::Ordering::Relaxed);
        context
            .buffer_pool_stats
            .evictions
            .store(50, std::sync::atomic::Ordering::Relaxed);

        let results = check.run(&context).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn test_health_status_from_summary() {
        let engine = DiagnosticEngine::new();
        let context = create_test_context();

        let report = engine.run(DiagnosticLevel::Quick, &context);

        // With good settings, should be healthy or degraded
        assert!(
            report.health_status == HealthStatus::Healthy
                || report.health_status == HealthStatus::Degraded
        );
    }
}
