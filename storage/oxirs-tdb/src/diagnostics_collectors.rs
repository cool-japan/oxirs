//! Diagnostic engine and built-in check collectors.
//!
//! This module hosts:
//! - [`DiagnosticEngine`] - registry + runner of [`DiagnosticCheck`] implementations
//! - [`DiagnosticContext`] - the input data fed to each check
//! - The trait [`DiagnosticCheck`] each individual check implements
//! - All built-in checks (buffer pool, memory, storage, performance, index/dictionary
//!   consistency, WAL integrity, corruption detection)

use crate::dictionary::NodeId;
use crate::error::Result;
use crate::index::Triple;
use crate::storage::BufferPoolStats;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::diagnostics_types::{
    DiagnosticLevel, DiagnosticReport, DiagnosticResult, DiagnosticSummary, HealthStatus, Severity,
};

/// Diagnostic engine
pub struct DiagnosticEngine {
    /// Registered diagnostic checks
    pub(crate) checks: Vec<Box<dyn DiagnosticCheck + Send + Sync>>,
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
pub(crate) struct BufferPoolCheck;

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
pub(crate) struct MemoryUsageCheck;

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
pub(crate) struct StorageEfficiencyCheck;

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
        if let Some(bytes_per_triple) = context.storage_size_bytes.checked_div(context.triple_count)
        {
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
pub(crate) struct PerformanceCheck;

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

/// Index consistency diagnostic check.
///
/// Verifies that SPO, POS, and OSP indexes contain the same set of triples.
pub(crate) struct IndexConsistencyCheck;

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

/// Dictionary consistency diagnostic check.
///
/// Verifies that all `NodeId`s in indexes exist in the dictionary.
pub(crate) struct DictionaryConsistencyCheck;

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

/// WAL integrity diagnostic check.
///
/// Verifies Write-Ahead Log file integrity.
pub(crate) struct WalIntegrityCheck;

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

/// Corruption detection check.
///
/// Performs various corruption detection checks.
pub(crate) struct CorruptionDetectionCheck;

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
        if let Some(bytes_per_triple) = context.storage_size_bytes.checked_div(context.triple_count)
        {
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
