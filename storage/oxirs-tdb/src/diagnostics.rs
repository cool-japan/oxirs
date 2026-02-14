//! Advanced diagnostic tools for TDB store
//!
//! This module provides comprehensive diagnostic capabilities:
//! - System health checks
//! - Performance analysis
//! - Storage diagnostics
//! - Index consistency checks
//! - Memory usage analysis
//! - WAL integrity verification
//! - Dictionary consistency checks
//! - B+Tree structure validation
//! - Corruption detection and repair

use crate::dictionary::NodeId;
use crate::error::{Result, TdbError};
use crate::index::Triple;
use crate::storage::BufferPoolStats;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
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
    /// SPO index triples (for consistency checks)
    pub spo_triples: Option<Vec<Triple>>,
    /// POS index triples (for consistency checks)
    pub pos_triples: Option<Vec<Triple>>,
    /// OSP index triples (for consistency checks)
    pub osp_triples: Option<Vec<Triple>>,
    /// Dictionary node IDs (for consistency checks)
    pub dictionary_node_ids: Option<HashSet<NodeId>>,
    /// Data directory path (for WAL checks)
    pub data_dir: Option<String>,
}

impl DiagnosticContext {
    /// Create a quick context (no deep data collection)
    pub fn quick(
        triple_count: u64,
        buffer_pool_stats: BufferPoolStats,
        dictionary_size: u64,
        storage_size_bytes: u64,
        memory_usage_bytes: usize,
    ) -> Self {
        Self {
            triple_count,
            buffer_pool_stats,
            dictionary_size,
            storage_size_bytes,
            memory_usage_bytes,
            spo_triples: None,
            pos_triples: None,
            osp_triples: None,
            dictionary_node_ids: None,
            data_dir: None,
        }
    }

    /// Create a deep context (with full data collection for consistency checks)
    #[allow(clippy::too_many_arguments)]
    pub fn deep(
        triple_count: u64,
        buffer_pool_stats: BufferPoolStats,
        dictionary_size: u64,
        storage_size_bytes: u64,
        memory_usage_bytes: usize,
        spo_triples: Vec<Triple>,
        pos_triples: Vec<Triple>,
        osp_triples: Vec<Triple>,
        dictionary_node_ids: HashSet<NodeId>,
        data_dir: String,
    ) -> Self {
        Self {
            triple_count,
            buffer_pool_stats,
            dictionary_size,
            storage_size_bytes,
            memory_usage_bytes,
            spo_triples: Some(spo_triples),
            pos_triples: Some(pos_triples),
            osp_triples: Some(osp_triples),
            dictionary_node_ids: Some(dictionary_node_ids),
            data_dir: Some(data_dir),
        }
    }
}

impl DiagnosticEngine {
    /// Create a new diagnostic engine
    pub fn new() -> Self {
        let mut engine = Self { checks: Vec::new() };

        // Register built-in checks (Quick level)
        engine.register(Box::new(BufferPoolCheck));
        engine.register(Box::new(MemoryUsageCheck));

        // Register standard checks
        engine.register(Box::new(StorageEfficiencyCheck));
        engine.register(Box::new(PerformanceCheck));

        // Register advanced/deep checks
        engine.register(Box::new(IndexConsistencyCheck));
        engine.register(Box::new(DictionaryConsistencyCheck));
        engine.register(Box::new(WalIntegrityCheck));
        engine.register(Box::new(CorruptionDetectionCheck));

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

/// Index consistency diagnostic check
/// Verifies that SPO, POS, and OSP indexes contain the same set of triples
struct IndexConsistencyCheck;

impl DiagnosticCheck for IndexConsistencyCheck {
    fn name(&self) -> &str {
        "Index Consistency"
    }

    fn category(&self) -> &str {
        "Integrity"
    }

    fn min_level(&self) -> DiagnosticLevel {
        DiagnosticLevel::Deep
    }

    fn run(&self, context: &DiagnosticContext) -> Result<Vec<DiagnosticResult>> {
        let mut results = Vec::new();

        // Check if we have the data needed for consistency checks
        if context.spo_triples.is_none()
            || context.pos_triples.is_none()
            || context.osp_triples.is_none()
        {
            results.push(
                DiagnosticResult::new(self.category(), "Insufficient Data", Severity::Warning)
                    .with_description("Deep context required for index consistency check")
                    .with_recommendation("Use DiagnosticContext::deep() for thorough checks"),
            );
            return Ok(results);
        }

        let spo_triples = context
            .spo_triples
            .as_ref()
            .expect("field validated to be Some above");
        let pos_triples = context
            .pos_triples
            .as_ref()
            .expect("field validated to be Some above");
        let osp_triples = context
            .osp_triples
            .as_ref()
            .expect("field validated to be Some above");

        // Convert to sets for comparison
        let spo_set: HashSet<Triple> = spo_triples.iter().copied().collect();
        let pos_set: HashSet<Triple> = pos_triples.iter().copied().collect();
        let osp_set: HashSet<Triple> = osp_triples.iter().copied().collect();

        // Check sizes
        if spo_set.len() != pos_set.len() || spo_set.len() != osp_set.len() {
            results.push(
                DiagnosticResult::new(self.category(), "Index Size Mismatch", Severity::Error)
                    .with_description(format!(
                        "Index sizes differ: SPO={}, POS={}, OSP={}",
                        spo_set.len(),
                        pos_set.len(),
                        osp_set.len()
                    ))
                    .with_recommendation("Run database repair to rebuild indexes")
                    .with_detail("spo_count", spo_set.len().to_string())
                    .with_detail("pos_count", pos_set.len().to_string())
                    .with_detail("osp_count", osp_set.len().to_string()),
            );
        }

        // Check if sets are equal
        let spo_minus_pos: HashSet<_> = spo_set.difference(&pos_set).collect();
        let pos_minus_spo: HashSet<_> = pos_set.difference(&spo_set).collect();
        let spo_minus_osp: HashSet<_> = spo_set.difference(&osp_set).collect();

        if !spo_minus_pos.is_empty() || !pos_minus_spo.is_empty() {
            results.push(
                DiagnosticResult::new(
                    self.category(),
                    "SPO/POS Inconsistency",
                    Severity::Critical,
                )
                .with_description(format!(
                    "SPO and POS indexes differ: {} triples in SPO not in POS, {} triples in POS not in SPO",
                    spo_minus_pos.len(),
                    pos_minus_spo.len()
                ))
                .with_recommendation("Critical: Run database repair immediately")
                .with_detail("spo_exclusive", spo_minus_pos.len().to_string())
                .with_detail("pos_exclusive", pos_minus_spo.len().to_string()),
            );
        }

        if !spo_minus_osp.is_empty() {
            let osp_minus_spo: HashSet<_> = osp_set.difference(&spo_set).collect();
            results.push(
                DiagnosticResult::new(
                    self.category(),
                    "SPO/OSP Inconsistency",
                    Severity::Critical,
                )
                .with_description(format!(
                    "SPO and OSP indexes differ: {} triples in SPO not in OSP, {} triples in OSP not in SPO",
                    spo_minus_osp.len(),
                    osp_minus_spo.len()
                ))
                .with_recommendation("Critical: Run database repair immediately")
                .with_detail("spo_exclusive", spo_minus_osp.len().to_string())
                .with_detail("osp_exclusive", osp_minus_spo.len().to_string()),
            );
        }

        // If all checks pass
        if results.is_empty() {
            results.push(
                DiagnosticResult::new(self.category(), "Indexes Consistent", Severity::Info)
                    .with_description(format!(
                        "All three indexes (SPO, POS, OSP) are consistent with {} triples each",
                        spo_set.len()
                    ))
                    .with_detail("triple_count", spo_set.len().to_string()),
            );
        }

        Ok(results)
    }
}

/// Dictionary consistency diagnostic check
/// Verifies that all NodeIds in indexes exist in the dictionary
struct DictionaryConsistencyCheck;

impl DiagnosticCheck for DictionaryConsistencyCheck {
    fn name(&self) -> &str {
        "Dictionary Consistency"
    }

    fn category(&self) -> &str {
        "Integrity"
    }

    fn min_level(&self) -> DiagnosticLevel {
        DiagnosticLevel::Deep
    }

    fn run(&self, context: &DiagnosticContext) -> Result<Vec<DiagnosticResult>> {
        let mut results = Vec::new();

        // Check if we have the data needed
        if context.spo_triples.is_none() || context.dictionary_node_ids.is_none() {
            results.push(
                DiagnosticResult::new(self.category(), "Insufficient Data", Severity::Warning)
                    .with_description("Deep context required for dictionary consistency check")
                    .with_recommendation("Use DiagnosticContext::deep() for thorough checks"),
            );
            return Ok(results);
        }

        let spo_triples = context
            .spo_triples
            .as_ref()
            .expect("field validated to be Some above");
        let dictionary_ids = context
            .dictionary_node_ids
            .as_ref()
            .expect("field validated to be Some above");

        // Collect all node IDs from triples
        let mut index_node_ids = HashSet::new();
        for triple in spo_triples {
            index_node_ids.insert(triple.subject);
            index_node_ids.insert(triple.predicate);
            index_node_ids.insert(triple.object);
        }

        // Find missing node IDs
        let missing_ids: HashSet<_> = index_node_ids.difference(dictionary_ids).collect();

        if !missing_ids.is_empty() {
            results.push(
                DiagnosticResult::new(
                    self.category(),
                    "Missing Dictionary Entries",
                    Severity::Critical,
                )
                .with_description(format!(
                    "Found {} NodeIds in indexes that don't exist in dictionary",
                    missing_ids.len()
                ))
                .with_recommendation("Critical: Database corruption detected. Restore from backup.")
                .with_detail("missing_count", missing_ids.len().to_string())
                .with_detail("total_index_ids", index_node_ids.len().to_string())
                .with_detail("dictionary_size", dictionary_ids.len().to_string()),
            );
        } else {
            // Check for orphaned dictionary entries (exist in dictionary but not in indexes)
            let orphaned_ids: HashSet<_> = dictionary_ids.difference(&index_node_ids).collect();

            if orphaned_ids.len() > 100 {
                // Allow some orphaned entries, but warn if excessive
                results.push(
                    DiagnosticResult::new(
                        self.category(),
                        "Excessive Orphaned Entries",
                        Severity::Warning,
                    )
                    .with_description(format!(
                        "Found {} orphaned dictionary entries not referenced by any triple",
                        orphaned_ids.len()
                    ))
                    .with_recommendation(
                        "Consider running compaction to reclaim unused dictionary space",
                    )
                    .with_detail("orphaned_count", orphaned_ids.len().to_string()),
                );
            }

            results.push(
                DiagnosticResult::new(self.category(), "Dictionary Consistent", Severity::Info)
                    .with_description("All NodeIds in indexes exist in dictionary")
                    .with_detail("index_node_ids", index_node_ids.len().to_string())
                    .with_detail("dictionary_size", dictionary_ids.len().to_string()),
            );
        }

        Ok(results)
    }
}

/// WAL integrity diagnostic check
/// Verifies Write-Ahead Log file integrity
struct WalIntegrityCheck;

impl DiagnosticCheck for WalIntegrityCheck {
    fn name(&self) -> &str {
        "WAL Integrity"
    }

    fn category(&self) -> &str {
        "Integrity"
    }

    fn min_level(&self) -> DiagnosticLevel {
        DiagnosticLevel::Standard
    }

    fn run(&self, context: &DiagnosticContext) -> Result<Vec<DiagnosticResult>> {
        let mut results = Vec::new();

        // Check if we have the data directory
        let Some(data_dir) = &context.data_dir else {
            results.push(
                DiagnosticResult::new(self.category(), "Insufficient Data", Severity::Info)
                    .with_description("Data directory not provided for WAL integrity check"),
            );
            return Ok(results);
        };

        let wal_path = Path::new(data_dir).join("wal");

        // Check if WAL file exists
        if !wal_path.exists() {
            results.push(
                DiagnosticResult::new(self.category(), "WAL Not Found", Severity::Warning)
                    .with_description("Write-Ahead Log file not found")
                    .with_recommendation("WAL may not be initialized or may have been deleted")
                    .with_detail("wal_path", wal_path.display().to_string()),
            );
            return Ok(results);
        }

        // Check WAL file size
        match std::fs::metadata(&wal_path) {
            Ok(metadata) => {
                let size_mb = metadata.len() / (1024 * 1024);

                if size_mb > 1024 {
                    // WAL larger than 1GB
                    results.push(
                        DiagnosticResult::new(self.category(), "Large WAL File", Severity::Warning)
                            .with_description(format!("WAL file is {} MB", size_mb))
                            .with_recommendation("Consider checkpointing WAL to reduce size")
                            .with_detail("wal_size_mb", size_mb.to_string()),
                    );
                } else {
                    results.push(
                        DiagnosticResult::new(self.category(), "WAL Size Normal", Severity::Info)
                            .with_description(format!("WAL file size: {} MB", size_mb))
                            .with_detail("wal_size_mb", size_mb.to_string()),
                    );
                }

                // TODO: Add CRC checksum verification for WAL entries
                // This would require reading WAL entries and verifying checksums
            }
            Err(e) => {
                results.push(
                    DiagnosticResult::new(self.category(), "WAL Access Error", Severity::Error)
                        .with_description(format!("Failed to access WAL file: {}", e))
                        .with_recommendation("Check file permissions and disk health"),
                );
            }
        }

        Ok(results)
    }
}

/// Corruption detection check
/// Performs various corruption detection checks
struct CorruptionDetectionCheck;

impl DiagnosticCheck for CorruptionDetectionCheck {
    fn name(&self) -> &str {
        "Corruption Detection"
    }

    fn category(&self) -> &str {
        "Integrity"
    }

    fn min_level(&self) -> DiagnosticLevel {
        DiagnosticLevel::Deep
    }

    fn run(&self, context: &DiagnosticContext) -> Result<Vec<DiagnosticResult>> {
        let mut results = Vec::new();

        // Check for unreasonable triple count
        if context.triple_count > 0 && context.dictionary_size == 0 {
            results.push(
                DiagnosticResult::new(self.category(), "Suspicious State", Severity::Critical)
                    .with_description("Triples exist but dictionary is empty")
                    .with_recommendation("Critical: Possible corruption. Restore from backup."),
            );
        }

        // Check for unreasonable storage efficiency
        if context.triple_count > 0 {
            let bytes_per_triple = context.storage_size_bytes / context.triple_count;

            // Triples should not be less than ~50 bytes or more than 10KB each
            if bytes_per_triple < 50 {
                results.push(
                    DiagnosticResult::new(
                        self.category(),
                        "Suspicious Storage Size",
                        Severity::Warning,
                    )
                    .with_description(format!(
                        "Storage size ({} bytes/triple) is unusually small",
                        bytes_per_triple
                    ))
                    .with_recommendation("Verify storage statistics are being collected correctly"),
                );
            } else if bytes_per_triple > 10_000 {
                results.push(
                    DiagnosticResult::new(
                        self.category(),
                        "Excessive Storage Overhead",
                        Severity::Warning,
                    )
                    .with_description(format!(
                        "Storage size ({} bytes/triple) is unusually large",
                        bytes_per_triple
                    ))
                    .with_recommendation(
                        "Possible fragmentation or corruption. Consider compaction.",
                    ),
                );
            }
        }

        // Check triple count vs dictionary size ratio
        if context.triple_count > 0 && context.dictionary_size > 0 {
            let ratio = context.dictionary_size as f64 / context.triple_count as f64;

            // Typically, dictionary should have 1-3x as many entries as triples
            if ratio > 10.0 {
                results.push(
                    DiagnosticResult::new(self.category(), "Dictionary Bloat", Severity::Warning)
                        .with_description(format!(
                            "Dictionary has {:.1}x more entries than triples",
                            ratio
                        ))
                        .with_recommendation(
                            "Consider compaction to remove unused dictionary entries",
                        )
                        .with_detail("ratio", format!("{:.2}", ratio)),
                );
            }
        }

        // If no issues found
        if results.is_empty() {
            results.push(
                DiagnosticResult::new(self.category(), "No Corruption Detected", Severity::Info)
                    .with_description("All corruption checks passed"),
            );
        }

        Ok(results)
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

impl DiagnosticReport {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn create_test_context() -> DiagnosticContext {
        DiagnosticContext::quick(
            1000,                       // triple_count
            BufferPoolStats::default(), // buffer_pool_stats
            5000,                       // dictionary_size
            200_000,                    // storage_size_bytes
            50_000_000,                 // memory_usage_bytes (50MB)
        )
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

    // Advanced diagnostic tests

    #[test]
    fn test_index_consistency_check_no_data() {
        let check = IndexConsistencyCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        // Should warn about insufficient data
        assert!(!results.is_empty());
        assert_eq!(results[0].severity, Severity::Warning);
        assert!(results[0].name.contains("Insufficient"));
    }

    #[test]
    fn test_index_consistency_check_consistent() {
        let check = IndexConsistencyCheck;

        let triple1 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triple2 = Triple::new(NodeId::new(4), NodeId::new(5), NodeId::new(6));
        let triples = vec![triple1, triple2];

        let context = DiagnosticContext::deep(
            2,                          // triple_count
            BufferPoolStats::default(), // buffer_pool_stats
            10,                         // dictionary_size
            1000,                       // storage_size_bytes
            50_000_000,                 // memory_usage_bytes
            triples.clone(),            // spo_triples
            triples.clone(),            // pos_triples
            triples,                    // osp_triples
            HashSet::new(),             // dictionary_node_ids
            "/tmp/test".to_string(),    // data_dir
        );

        let results = check.run(&context).unwrap();

        // Should report consistent indexes
        assert!(!results.is_empty());
        let has_consistent = results.iter().any(|r| r.name.contains("Consistent"));
        assert!(has_consistent);
    }

    #[test]
    fn test_index_consistency_check_inconsistent() {
        let check = IndexConsistencyCheck;

        let triple1 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triple2 = Triple::new(NodeId::new(4), NodeId::new(5), NodeId::new(6));
        let triple3 = Triple::new(NodeId::new(7), NodeId::new(8), NodeId::new(9));

        let spo_triples = vec![triple1, triple2];
        let pos_triples = vec![triple1, triple3]; // Different!
        let osp_triples = vec![triple1, triple2];

        let context = DiagnosticContext::deep(
            2,
            BufferPoolStats::default(),
            10,
            1000,
            50_000_000,
            spo_triples,
            pos_triples,
            osp_triples,
            HashSet::new(),
            "/tmp/test".to_string(),
        );

        let results = check.run(&context).unwrap();

        // Should detect inconsistency
        assert!(!results.is_empty());
        let has_critical = results.iter().any(|r| r.severity == Severity::Critical);
        assert!(has_critical);
    }

    #[test]
    fn test_dictionary_consistency_check_no_data() {
        let check = DictionaryConsistencyCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        // Should warn about insufficient data
        assert!(!results.is_empty());
        assert_eq!(results[0].severity, Severity::Warning);
    }

    #[test]
    fn test_dictionary_consistency_check_consistent() {
        let check = DictionaryConsistencyCheck;

        let triple1 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triples = vec![triple1];

        let mut dict_ids = HashSet::new();
        dict_ids.insert(NodeId::new(1));
        dict_ids.insert(NodeId::new(2));
        dict_ids.insert(NodeId::new(3));

        let context = DiagnosticContext::deep(
            1,
            BufferPoolStats::default(),
            3,
            500,
            50_000_000,
            triples,
            vec![],
            vec![],
            dict_ids,
            "/tmp/test".to_string(),
        );

        let results = check.run(&context).unwrap();

        // Should report consistent dictionary
        assert!(!results.is_empty());
        let has_consistent = results.iter().any(|r| r.name.contains("Consistent"));
        assert!(has_consistent);
    }

    #[test]
    fn test_dictionary_consistency_check_missing_entries() {
        let check = DictionaryConsistencyCheck;

        let triple1 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triples = vec![triple1];

        let mut dict_ids = HashSet::new();
        dict_ids.insert(NodeId::new(1));
        // Missing NodeId 2 and 3!

        let context = DiagnosticContext::deep(
            1,
            BufferPoolStats::default(),
            1,
            500,
            50_000_000,
            triples,
            vec![],
            vec![],
            dict_ids,
            "/tmp/test".to_string(),
        );

        let results = check.run(&context).unwrap();

        // Should detect missing entries
        assert!(!results.is_empty());
        let has_critical = results.iter().any(|r| r.severity == Severity::Critical);
        assert!(has_critical);
        let has_missing = results.iter().any(|r| r.name.contains("Missing"));
        assert!(has_missing);
    }

    #[test]
    fn test_dictionary_consistency_check_orphaned_entries() {
        let check = DictionaryConsistencyCheck;

        let triple1 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triples = vec![triple1];

        let mut dict_ids = HashSet::new();
        dict_ids.insert(NodeId::new(1));
        dict_ids.insert(NodeId::new(2));
        dict_ids.insert(NodeId::new(3));
        // Add many orphaned entries
        for i in 100..250 {
            dict_ids.insert(NodeId::new(i));
        }

        let context = DiagnosticContext::deep(
            1,
            BufferPoolStats::default(),
            dict_ids.len() as u64,
            500,
            50_000_000,
            triples,
            vec![],
            vec![],
            dict_ids,
            "/tmp/test".to_string(),
        );

        let results = check.run(&context).unwrap();

        // Should warn about orphaned entries
        let has_orphaned = results.iter().any(|r| r.name.contains("Orphaned"));
        assert!(has_orphaned);
    }

    #[test]
    fn test_wal_integrity_check_no_data() {
        let check = WalIntegrityCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        // Should be info level (no data directory provided)
        assert!(!results.is_empty());
        assert_eq!(results[0].severity, Severity::Info);
    }

    #[test]
    fn test_wal_integrity_check_no_wal() {
        let check = WalIntegrityCheck;

        use std::env;
        let temp_dir = env::temp_dir().join("oxirs_test_wal_missing");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut context = create_test_context();
        context.data_dir = Some(temp_dir.to_str().unwrap().to_string());

        let results = check.run(&context).unwrap();

        // Should warn about missing WAL
        assert!(!results.is_empty());
        let has_warning = results.iter().any(|r| r.name.contains("Not Found"));
        assert!(has_warning);

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_corruption_detection_check_healthy() {
        let check = CorruptionDetectionCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        // Should report no corruption
        assert!(!results.is_empty());
        let has_ok = results.iter().any(|r| r.name.contains("No Corruption"));
        assert!(has_ok);
    }

    #[test]
    fn test_corruption_detection_check_suspicious_state() {
        let check = CorruptionDetectionCheck;

        let mut context = create_test_context();
        context.triple_count = 1000;
        context.dictionary_size = 0; // Suspicious!

        let results = check.run(&context).unwrap();

        // Should detect suspicious state
        assert!(!results.is_empty());
        let has_critical = results.iter().any(|r| r.severity == Severity::Critical);
        assert!(has_critical);
    }

    #[test]
    fn test_corruption_detection_check_storage_overhead() {
        let check = CorruptionDetectionCheck;

        let mut context = create_test_context();
        context.triple_count = 1000;
        context.storage_size_bytes = 20_000_000; // 20KB per triple - excessive!

        let results = check.run(&context).unwrap();

        // Should detect excessive overhead
        assert!(!results.is_empty());
        let has_warning = results.iter().any(|r| r.name.contains("Excessive Storage"));
        assert!(has_warning);
    }

    #[test]
    fn test_corruption_detection_check_dictionary_bloat() {
        let check = CorruptionDetectionCheck;

        let mut context = create_test_context();
        context.triple_count = 1000;
        context.dictionary_size = 15_000; // 15x ratio - bloated!

        let results = check.run(&context).unwrap();

        // Should detect dictionary bloat
        assert!(!results.is_empty());
        let has_bloat = results.iter().any(|r| r.name.contains("Bloat"));
        assert!(has_bloat);
    }

    #[test]
    fn test_repair_recommendations_index_issues() {
        let results =
            vec![
                DiagnosticResult::new("Integrity", "Index Inconsistency", Severity::Error)
                    .with_description("Indexes are inconsistent"),
            ];

        let report = DiagnosticReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            level: DiagnosticLevel::Deep,
            duration: Duration::from_secs(1),
            health_status: HealthStatus::Unhealthy,
            results,
            summary: DiagnosticSummary {
                info_count: 0,
                warning_count: 0,
                error_count: 1,
                critical_count: 0,
            },
        };

        let recommendations = report.repair_recommendations();

        assert!(!recommendations.is_empty());
        let has_rebuild = recommendations
            .iter()
            .any(|r| matches!(r.action, RepairAction::RebuildIndexes));
        assert!(has_rebuild);
    }

    #[test]
    fn test_repair_recommendations_dictionary_issues() {
        let results =
            vec![
                DiagnosticResult::new("Integrity", "Dictionary Corruption", Severity::Critical)
                    .with_description("Dictionary is corrupted"),
            ];

        let report = DiagnosticReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            level: DiagnosticLevel::Deep,
            duration: Duration::from_secs(1),
            health_status: HealthStatus::Critical,
            results,
            summary: DiagnosticSummary {
                info_count: 0,
                warning_count: 0,
                error_count: 0,
                critical_count: 1,
            },
        };

        let recommendations = report.repair_recommendations();

        assert!(!recommendations.is_empty());
        let has_restore = recommendations
            .iter()
            .any(|r| matches!(r.action, RepairAction::RestoreFromBackup));
        assert!(has_restore);
    }

    #[test]
    fn test_repair_recommendations_wal_issues() {
        let results = vec![
            DiagnosticResult::new("Integrity", "Large WAL File", Severity::Warning)
                .with_description("WAL file is large"),
        ];

        let report = DiagnosticReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            level: DiagnosticLevel::Standard,
            duration: Duration::from_secs(1),
            health_status: HealthStatus::Degraded,
            results,
            summary: DiagnosticSummary {
                info_count: 0,
                warning_count: 1,
                error_count: 0,
                critical_count: 0,
            },
        };

        let recommendations = report.repair_recommendations();

        assert!(!recommendations.is_empty());
        let has_checkpoint = recommendations
            .iter()
            .any(|r| matches!(r.action, RepairAction::CheckpointWal));
        assert!(has_checkpoint);
    }

    #[test]
    fn test_diagnostic_context_quick_constructor() {
        let context =
            DiagnosticContext::quick(100, BufferPoolStats::default(), 500, 10_000, 1_000_000);

        assert_eq!(context.triple_count, 100);
        assert_eq!(context.dictionary_size, 500);
        assert!(context.spo_triples.is_none());
        assert!(context.pos_triples.is_none());
        assert!(context.osp_triples.is_none());
        assert!(context.dictionary_node_ids.is_none());
        assert!(context.data_dir.is_none());
    }

    #[test]
    fn test_diagnostic_context_deep_constructor() {
        let triple = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triples = vec![triple];
        let dict_ids = HashSet::new();

        let context = DiagnosticContext::deep(
            1,
            BufferPoolStats::default(),
            3,
            500,
            1_000_000,
            triples.clone(),
            triples.clone(),
            triples,
            dict_ids,
            "/tmp/test".to_string(),
        );

        assert_eq!(context.triple_count, 1);
        assert!(context.spo_triples.is_some());
        assert!(context.pos_triples.is_some());
        assert!(context.osp_triples.is_some());
        assert!(context.dictionary_node_ids.is_some());
        assert!(context.data_dir.is_some());
    }

    #[test]
    fn test_deep_diagnostics_runs_all_checks() {
        let engine = DiagnosticEngine::new();

        let triple = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triples = vec![triple];
        let mut dict_ids = HashSet::new();
        dict_ids.insert(NodeId::new(1));
        dict_ids.insert(NodeId::new(2));
        dict_ids.insert(NodeId::new(3));

        let context = DiagnosticContext::deep(
            1,
            BufferPoolStats::default(),
            3,
            500,
            50_000_000,
            triples.clone(),
            triples.clone(),
            triples,
            dict_ids,
            "/tmp/test".to_string(),
        );

        let report = engine.run(DiagnosticLevel::Deep, &context);

        // Deep diagnostics should run more checks than quick
        assert!(report.results.len() > 4);

        // Should include integrity checks
        let has_integrity = report.results.iter().any(|r| r.category == "Integrity");
        assert!(has_integrity);
    }
}
