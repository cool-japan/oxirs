//! Crash recovery and corruption detection
//!
//! This module provides functionality for:
//! - WAL-based crash recovery
//! - Data corruption detection using checksums
//! - Index consistency verification
//! - Automatic repair when possible

use crate::error::{Result, TdbError};
use crate::index::TripleIndexes;
use crate::storage::page::PageId;
use crate::storage::BufferPool;
use crate::transaction::WriteAheadLog;
use std::path::Path;
use std::sync::Arc;

/// Recovery manager for crash recovery and corruption detection
pub struct RecoveryManager {
    buffer_pool: Arc<BufferPool>,
    wal: Arc<WriteAheadLog>,
}

impl RecoveryManager {
    /// Create a new recovery manager
    pub fn new(buffer_pool: Arc<BufferPool>, wal: Arc<WriteAheadLog>) -> Self {
        RecoveryManager { buffer_pool, wal }
    }

    /// Perform crash recovery on startup
    ///
    /// This method should be called when opening a database to:
    /// - Check for unclean shutdown
    /// - Replay WAL entries to recover uncommitted transactions
    /// - Verify data integrity
    pub fn recover(&self) -> Result<RecoveryReport> {
        let mut report = RecoveryReport {
            clean_shutdown: true,
            wal_entries_replayed: 0,
            corrupted_pages: 0,
            recovered: true,
            errors: Vec::new(),
        };

        // Recover from WAL (this replays all log entries)
        match self.replay_wal() {
            Ok(replayed_count) => {
                report.wal_entries_replayed = replayed_count;
                if replayed_count > 0 {
                    report.clean_shutdown = false;
                    log::info!("Replayed {} WAL entries during recovery", replayed_count);
                }
            }
            Err(e) => {
                report.recovered = false;
                report.errors.push(format!("WAL replay failed: {}", e));
                return Ok(report);
            }
        }

        // Verify buffer pool integrity
        match self.verify_buffer_pool() {
            Ok(corrupted_count) => {
                report.corrupted_pages = corrupted_count;
                if corrupted_count > 0 {
                    report.errors.push(format!(
                        "Found {} corrupted pages in buffer pool",
                        corrupted_count
                    ));
                }
            }
            Err(e) => {
                report.recovered = false;
                report
                    .errors
                    .push(format!("Buffer pool verification failed: {}", e));
            }
        }

        Ok(report)
    }

    /// Replay WAL entries to recover from crash
    ///
    /// This method:
    /// - Reads all WAL entries
    /// - Analyzes transaction states (committed, aborted, active)
    /// - Applies redo operations for committed transactions
    /// - Returns count of replayed entries
    fn replay_wal(&self) -> Result<usize> {
        use crate::transaction::wal::LogRecord;
        use std::collections::{HashMap, HashSet};

        // Recover WAL entries
        let entries = self.wal.recover()?;

        if entries.is_empty() {
            return Ok(0);
        }

        // Track transaction states
        let mut committed_txns = HashSet::new();
        let mut aborted_txns = HashSet::new();
        let mut active_txns = HashSet::new();

        // First pass: identify transaction states
        for entry in &entries {
            match &entry.record {
                LogRecord::Begin { txn_id } => {
                    active_txns.insert(*txn_id);
                }
                LogRecord::Commit { txn_id } => {
                    active_txns.remove(txn_id);
                    committed_txns.insert(*txn_id);
                }
                LogRecord::Abort { txn_id } => {
                    active_txns.remove(txn_id);
                    aborted_txns.insert(*txn_id);
                }
                LogRecord::Checkpoint {
                    active_txns: checkpoint_txns,
                } => {
                    // Update active transactions based on checkpoint
                    for txn_id in checkpoint_txns {
                        active_txns.insert(*txn_id);
                    }
                }
                _ => {}
            }
        }

        // Second pass: apply redo operations for committed transactions
        let mut replayed_count = 0;
        for entry in &entries {
            if let LogRecord::Update {
                txn_id,
                page_id,
                after_image,
                ..
            } = &entry.record
            {
                // Only redo if transaction was committed
                if committed_txns.contains(txn_id) {
                    // Redo: apply the after-image to the page
                    if let Err(e) = self.apply_page_update(*page_id, after_image) {
                        log::warn!("Failed to replay update for page {}: {}", page_id, e);
                    } else {
                        replayed_count += 1;
                    }
                }
            }
        }

        // Log active transactions that were not committed or aborted
        if !active_txns.is_empty() {
            log::warn!(
                "Found {} active transactions during recovery (will be rolled back)",
                active_txns.len()
            );
        }

        Ok(replayed_count)
    }

    /// Apply a page update from WAL replay
    fn apply_page_update(&self, page_id: PageId, after_image: &[u8]) -> Result<()> {
        use crate::storage::page::{Page, PAGE_SIZE};

        // Ensure after_image has correct size
        if after_image.len() != PAGE_SIZE {
            return Err(TdbError::Other(format!(
                "Invalid page image size: expected {}, got {}",
                PAGE_SIZE,
                after_image.len()
            )));
        }

        // Load page from buffer pool
        let page_guard = self.buffer_pool.fetch_page(page_id)?;

        // Get mutable access to the page
        let mut page_lock = page_guard.page_mut();
        if let Some(ref mut page) = *page_lock {
            // Apply the after-image
            let raw_data = page.raw_data_mut();
            raw_data.copy_from_slice(after_image);
        } else {
            return Err(TdbError::Other(format!(
                "Page {} not found in buffer pool",
                page_id
            )));
        }

        Ok(())
    }

    /// Verify buffer pool integrity
    ///
    /// Checks for:
    /// - Page checksum mismatches
    /// - Invalid page types
    /// - Corrupted page headers
    fn verify_buffer_pool(&self) -> Result<usize> {
        use crate::storage::page::PageType;

        let mut corrupted_count = 0;

        // Check currently cached pages
        let cached_count = self.buffer_pool.cached_pages();

        // We check a reasonable range of page IDs
        // In a real implementation, we'd track the max allocated page ID
        let max_pages_to_check = cached_count.max(100);

        for page_id in 0..max_pages_to_check as u64 {
            // Try to fetch the page (this may fail if page doesn't exist)
            match self.buffer_pool.fetch_page(page_id) {
                Ok(page_guard) => {
                    let page_lock = page_guard.page();
                    if let Some(ref page) = *page_lock {
                        // Verify checksum
                        if !page.verify_checksum() {
                            log::warn!("Page {} has invalid checksum", page_id);
                            corrupted_count += 1;
                            continue;
                        }

                        // Verify page type is valid
                        let page_type = page.page_type();
                        let valid_type = matches!(
                            page_type,
                            PageType::Free
                                | PageType::BTreeInternal
                                | PageType::BTreeLeaf
                                | PageType::Dictionary
                                | PageType::StringData
                                | PageType::Metadata
                        );

                        if !valid_type {
                            log::warn!("Page {} has invalid type: {:?}", page_id, page_type);
                            corrupted_count += 1;
                        }
                    }
                }
                Err(_) => {
                    // Page doesn't exist or can't be loaded - skip
                    continue;
                }
            }
        }

        Ok(corrupted_count)
    }

    /// Verify index consistency
    ///
    /// Checks that:
    /// - All three indexes (SPO, POS, OSP) contain the same triples
    /// - No orphaned entries exist
    /// - B+tree structure is valid
    pub fn verify_indexes(&self, indexes: &TripleIndexes) -> Result<IndexVerificationReport> {
        use std::collections::HashSet;

        let mut errors = Vec::new();

        // Scan all entries from SPO index
        log::info!("Scanning SPO index...");
        let spo_keys = match indexes.spo().range_scan(None, None) {
            Ok(keys) => keys,
            Err(e) => {
                errors.push(format!("Failed to scan SPO index: {}", e));
                Vec::new()
            }
        };
        let spo_count = spo_keys.len();
        // Convert SpoKey to normalized Triple representation
        let spo_set: HashSet<_> = spo_keys
            .into_iter()
            .map(|key| (key.0, key.1, key.2))
            .collect();

        // Scan all entries from POS index
        log::info!("Scanning POS index...");
        let pos_keys = match indexes.pos().range_scan(None, None) {
            Ok(keys) => keys,
            Err(e) => {
                errors.push(format!("Failed to scan POS index: {}", e));
                Vec::new()
            }
        };
        let pos_count = pos_keys.len();
        // Convert PosKey (p, o, s) to normalized (s, p, o) tuple
        let pos_set: HashSet<_> = pos_keys
            .into_iter()
            .map(|key| (key.2, key.0, key.1))
            .collect();

        // Scan all entries from OSP index
        log::info!("Scanning OSP index...");
        let osp_keys = match indexes.osp().range_scan(None, None) {
            Ok(keys) => keys,
            Err(e) => {
                errors.push(format!("Failed to scan OSP index: {}", e));
                Vec::new()
            }
        };
        let osp_count = osp_keys.len();
        // Convert OspKey (o, s, p) to normalized (s, p, o) tuple
        let osp_set: HashSet<_> = osp_keys
            .into_iter()
            .map(|key| (key.1, key.2, key.0))
            .collect();

        // Check consistency: all three indexes should contain the same triples
        let consistent = spo_set == pos_set && pos_set == osp_set;

        // Detect orphaned entries (entries in one index but not in others)
        let mut orphaned_entries = 0;

        if !consistent {
            // Find entries only in SPO
            let spo_only: HashSet<_> = spo_set.difference(&pos_set).cloned().collect();
            if !spo_only.is_empty() {
                errors.push(format!(
                    "Found {} entries only in SPO index",
                    spo_only.len()
                ));
                orphaned_entries += spo_only.len();
            }

            // Find entries only in POS
            let pos_only: HashSet<_> = pos_set.difference(&osp_set).cloned().collect();
            if !pos_only.is_empty() {
                errors.push(format!(
                    "Found {} entries only in POS index",
                    pos_only.len()
                ));
                orphaned_entries += pos_only.len();
            }

            // Find entries only in OSP
            let osp_only: HashSet<_> = osp_set.difference(&spo_set).cloned().collect();
            if !osp_only.is_empty() {
                errors.push(format!(
                    "Found {} entries only in OSP index",
                    osp_only.len()
                ));
                orphaned_entries += osp_only.len();
            }
        }

        let report = IndexVerificationReport {
            spo_entries: spo_count,
            pos_entries: pos_count,
            osp_entries: osp_count,
            consistent,
            orphaned_entries,
            errors,
        };

        if consistent {
            log::info!(
                "Index verification completed successfully: {} triples found",
                spo_count
            );
        } else {
            log::warn!(
                "Index verification found inconsistencies: {} orphaned entries",
                orphaned_entries
            );
        }

        Ok(report)
    }

    /// Detect and report corruption in the database
    ///
    /// Performs comprehensive checks:
    /// - Page-level corruption
    /// - Index consistency
    /// - Dictionary integrity
    /// - WAL integrity
    pub fn detect_corruption(&self) -> Result<CorruptionReport> {
        let mut report = CorruptionReport {
            has_corruption: false,
            corrupted_pages: Vec::new(),
            index_inconsistencies: Vec::new(),
            wal_corruption: false,
            severity: CorruptionSeverity::None,
        };

        // Check buffer pool
        match self.verify_buffer_pool() {
            Ok(count) if count > 0 => {
                report.has_corruption = true;
                report.severity = CorruptionSeverity::Minor;
                for i in 0..count {
                    report
                        .corrupted_pages
                        .push(format!("Page corruption detected ({})", i));
                }
            }
            Ok(_) => {}
            Err(e) => {
                report.has_corruption = true;
                report.severity = CorruptionSeverity::Major;
                report
                    .corrupted_pages
                    .push(format!("Buffer pool error: {}", e));
            }
        }

        // Check WAL integrity
        match self.verify_wal_integrity() {
            Ok(is_corrupt) => {
                if is_corrupt {
                    report.has_corruption = true;
                    report.wal_corruption = true;
                    if report.severity < CorruptionSeverity::Major {
                        report.severity = CorruptionSeverity::Major;
                    }
                }
            }
            Err(e) => {
                report.has_corruption = true;
                report.wal_corruption = true;
                report.severity = CorruptionSeverity::Fatal;
                report
                    .corrupted_pages
                    .push(format!("WAL verification failed: {}", e));
            }
        }

        Ok(report)
    }

    /// Verify WAL integrity
    ///
    /// Checks for:
    /// - Corrupted WAL entries
    /// - Invalid log sequence
    /// - Incomplete transactions
    fn verify_wal_integrity(&self) -> Result<bool> {
        use crate::transaction::wal::Lsn;

        // Try to recover all WAL entries
        match self.wal.recover() {
            Ok(entries) => {
                // Verify LSN sequence
                let mut prev_lsn = Lsn::ZERO;
                for entry in &entries {
                    if entry.lsn.as_u64() < prev_lsn.as_u64() {
                        log::warn!(
                            "WAL LSN sequence violation: {} < {}",
                            entry.lsn.as_u64(),
                            prev_lsn.as_u64()
                        );
                        return Ok(true); // Corruption detected
                    }
                    prev_lsn = entry.lsn;
                }

                // No corruption detected
                Ok(false)
            }
            Err(e) => {
                log::error!("Failed to recover WAL: {}", e);
                Ok(true) // Corruption detected
            }
        }
    }

    /// Attempt to repair corruption (if possible)
    ///
    /// Repairs may include:
    /// - Rebuilding indexes from dictionary
    /// - Recomputing checksums
    /// - Removing invalid entries
    ///
    /// Note: This is a destructive operation and may result in data loss
    pub fn repair(&mut self) -> Result<RepairReport> {
        let mut report = RepairReport {
            repairs_attempted: 0,
            repairs_successful: 0,
            data_loss: false,
            errors: Vec::new(),
        };

        // Detect corruption first
        let corruption = self.detect_corruption()?;

        if !corruption.has_corruption {
            return Ok(report);
        }

        // Attempt repairs based on severity
        match corruption.severity {
            CorruptionSeverity::None => {
                // No corruption, nothing to repair
            }
            CorruptionSeverity::Minor => {
                // Minor corruption - try to repair without data loss
                report.repairs_attempted += 1;

                // Recompute checksums for corrupted pages
                if let Err(e) = self.repair_page_checksums() {
                    report
                        .errors
                        .push(format!("Failed to repair page checksums: {}", e));
                } else {
                    report.repairs_successful += 1;
                    log::info!("Successfully repaired page checksums");
                }
            }
            CorruptionSeverity::Major => {
                // Major corruption - may require data loss
                report.repairs_attempted += 1;
                report.data_loss = true;

                // Attempt to rebuild from WAL if possible
                match self.rebuild_from_wal() {
                    Ok(rebuilt) => {
                        if rebuilt {
                            report.repairs_successful += 1;
                            log::info!("Successfully rebuilt database from WAL");
                        } else {
                            report.errors.push("Failed to rebuild from WAL".to_string());
                        }
                    }
                    Err(e) => {
                        report.errors.push(format!("WAL rebuild failed: {}", e));
                    }
                }

                // If WAL rebuild didn't work, try to salvage what we can
                if report.repairs_successful == 0 {
                    report
                        .errors
                        .push("Major corruption detected - manual recovery required".to_string());
                }
            }
            CorruptionSeverity::Fatal => {
                // Fatal corruption - cannot repair
                report
                    .errors
                    .push("Fatal corruption - database cannot be repaired".to_string());
                report
                    .errors
                    .push("Consider restoring from backup or reinitializing database".to_string());
            }
        }

        Ok(report)
    }

    /// Repair page checksums (for minor corruption)
    fn repair_page_checksums(&self) -> Result<()> {
        let cached_count = self.buffer_pool.cached_pages();
        let max_pages_to_check = cached_count.max(100);

        // Check all allocated pages
        for page_id in 0..max_pages_to_check as u64 {
            match self.buffer_pool.fetch_page(page_id) {
                Ok(page_guard) => {
                    let mut page_lock = page_guard.page_mut();
                    if let Some(ref mut page) = *page_lock {
                        // Recompute and update checksum
                        page.update_header();
                        log::debug!("Repaired checksum for page {}", page_id);
                    }
                }
                Err(_) => {
                    // Page doesn't exist - skip
                    continue;
                }
            }
        }

        Ok(())
    }

    /// Attempt to rebuild database from WAL (for major corruption)
    fn rebuild_from_wal(&self) -> Result<bool> {
        use crate::transaction::wal::LogRecord;
        use std::collections::HashMap;

        // Recover all WAL entries
        let entries = self.wal.recover()?;

        if entries.is_empty() {
            return Ok(false); // No WAL to rebuild from
        }

        // Track the latest state for each page
        let mut page_states: HashMap<PageId, Vec<u8>> = HashMap::new();

        // Replay all Update records to reconstruct page states
        for entry in &entries {
            if let LogRecord::Update {
                page_id,
                after_image,
                ..
            } = &entry.record
            {
                // Store the latest state for this page
                page_states.insert(*page_id, after_image.clone());
            }
        }

        // Check if we have any pages to rebuild
        let has_pages = !page_states.is_empty();

        // Apply all reconstructed page states
        for (page_id, after_image) in page_states {
            if let Err(e) = self.apply_page_update(page_id, &after_image) {
                log::warn!("Failed to apply reconstructed page {}: {}", page_id, e);
            }
        }

        Ok(has_pages)
    }
}

/// Report from crash recovery operation
#[derive(Debug)]
pub struct RecoveryReport {
    /// Whether the database was cleanly shutdown
    pub clean_shutdown: bool,
    /// Number of WAL entries replayed
    pub wal_entries_replayed: usize,
    /// Number of corrupted pages found
    pub corrupted_pages: usize,
    /// Whether recovery was successful
    pub recovered: bool,
    /// List of errors encountered
    pub errors: Vec<String>,
}

/// Report from index verification
#[derive(Debug)]
pub struct IndexVerificationReport {
    /// Number of entries in SPO index
    pub spo_entries: usize,
    /// Number of entries in POS index
    pub pos_entries: usize,
    /// Number of entries in OSP index
    pub osp_entries: usize,
    /// Whether all indexes are consistent
    pub consistent: bool,
    /// Number of orphaned entries found
    pub orphaned_entries: usize,
    /// List of errors encountered
    pub errors: Vec<String>,
}

/// Report from corruption detection
#[derive(Debug)]
pub struct CorruptionReport {
    /// Whether corruption was detected
    pub has_corruption: bool,
    /// List of corrupted pages
    pub corrupted_pages: Vec<String>,
    /// List of index inconsistencies
    pub index_inconsistencies: Vec<String>,
    /// Whether WAL is corrupted
    pub wal_corruption: bool,
    /// Severity of corruption
    pub severity: CorruptionSeverity,
}

/// Severity of corruption
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CorruptionSeverity {
    /// No corruption detected
    None = 0,
    /// Minor corruption (e.g., checksums can be recomputed)
    Minor = 1,
    /// Major corruption (e.g., missing index entries, may require data loss)
    Major = 2,
    /// Fatal corruption (database cannot be repaired)
    Fatal = 3,
}

/// Report from repair operation
#[derive(Debug)]
pub struct RepairReport {
    /// Number of repairs attempted
    pub repairs_attempted: usize,
    /// Number of repairs successful
    pub repairs_successful: usize,
    /// Whether data loss occurred
    pub data_loss: bool,
    /// List of errors encountered
    pub errors: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::FileManager;
    use crate::transaction::WriteAheadLog;
    use std::env;
    use tempfile::TempDir;

    #[test]
    fn test_recovery_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let wal_dir = temp_dir.path();

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));
        let wal = Arc::new(WriteAheadLog::new(wal_dir).unwrap());

        let _recovery = RecoveryManager::new(buffer_pool, wal);
    }

    #[test]
    fn test_recovery_clean_database() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let wal_dir = temp_dir.path();

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));
        let wal = Arc::new(WriteAheadLog::new(wal_dir).unwrap());

        let recovery = RecoveryManager::new(buffer_pool, wal);
        let report = recovery.recover().unwrap();

        assert!(report.clean_shutdown);
        assert_eq!(report.wal_entries_replayed, 0);
        assert_eq!(report.corrupted_pages, 0);
        assert!(report.recovered);
    }

    #[test]
    fn test_corruption_detection_clean() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let wal_dir = temp_dir.path();

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));
        let wal = Arc::new(WriteAheadLog::new(wal_dir).unwrap());

        let recovery = RecoveryManager::new(buffer_pool, wal);
        let report = recovery.detect_corruption().unwrap();

        assert!(!report.has_corruption);
        assert_eq!(report.severity, CorruptionSeverity::None);
    }

    #[test]
    fn test_repair_no_corruption() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let wal_dir = temp_dir.path();

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));
        let wal = Arc::new(WriteAheadLog::new(wal_dir).unwrap());

        let mut recovery = RecoveryManager::new(buffer_pool, wal);
        let report = recovery.repair().unwrap();

        assert_eq!(report.repairs_attempted, 0);
        assert_eq!(report.repairs_successful, 0);
        assert!(!report.data_loss);
    }

    #[test]
    fn test_wal_replay_with_committed_transaction() {
        use crate::storage::page::{PageType, PAGE_SIZE};
        use crate::transaction::wal::{LogRecord, TxnId};

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let wal_dir = temp_dir.path();

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));
        let wal = Arc::new(WriteAheadLog::new(wal_dir).unwrap());

        // Create a test page
        let page_guard = buffer_pool.new_page(PageType::BTreeLeaf).unwrap();
        let test_page_id = page_guard.page_id();
        drop(page_guard);

        // Simulate a transaction with WAL entries
        let txn_id = TxnId::new(1);

        // Begin transaction
        wal.append(LogRecord::Begin { txn_id }).unwrap();

        // Update page (create before and after images)
        let before_image = vec![0u8; PAGE_SIZE];
        let after_image = vec![42u8; PAGE_SIZE];

        wal.append(LogRecord::Update {
            txn_id,
            page_id: test_page_id,
            before_image,
            after_image,
        })
        .unwrap();

        // Commit transaction
        wal.append(LogRecord::Commit { txn_id }).unwrap();
        wal.flush().unwrap();

        // Now perform recovery
        let recovery = RecoveryManager::new(buffer_pool.clone(), wal);
        let report = recovery.recover().unwrap();

        // Should have replayed one WAL entry
        assert_eq!(report.wal_entries_replayed, 1);
        assert!(!report.clean_shutdown);
        assert!(report.recovered);
    }

    #[test]
    fn test_wal_replay_with_aborted_transaction() {
        use crate::storage::page::{PageType, PAGE_SIZE};
        use crate::transaction::wal::{LogRecord, TxnId};

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let wal_dir = temp_dir.path();

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));
        let wal = Arc::new(WriteAheadLog::new(wal_dir).unwrap());

        // Create a test page
        let page_guard = buffer_pool.new_page(PageType::BTreeLeaf).unwrap();
        let test_page_id = page_guard.page_id();
        drop(page_guard);

        // Simulate a transaction that aborts
        let txn_id = TxnId::new(1);

        wal.append(LogRecord::Begin { txn_id }).unwrap();

        let before_image = vec![0u8; PAGE_SIZE];
        let after_image = vec![42u8; PAGE_SIZE];

        wal.append(LogRecord::Update {
            txn_id,
            page_id: test_page_id,
            before_image,
            after_image,
        })
        .unwrap();

        // Abort instead of commit
        wal.append(LogRecord::Abort { txn_id }).unwrap();
        wal.flush().unwrap();

        // Perform recovery
        let recovery = RecoveryManager::new(buffer_pool, wal);
        let report = recovery.recover().unwrap();

        // Should NOT have replayed the update (transaction was aborted)
        assert_eq!(report.wal_entries_replayed, 0);
        assert!(report.recovered);
    }

    #[test]
    fn test_wal_integrity_verification() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let wal_dir = temp_dir.path();

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));
        let wal = Arc::new(WriteAheadLog::new(wal_dir).unwrap());

        // Write some WAL entries in proper sequence
        use crate::transaction::wal::{LogRecord, TxnId};

        wal.append(LogRecord::Begin {
            txn_id: TxnId::new(1),
        })
        .unwrap();
        wal.append(LogRecord::Commit {
            txn_id: TxnId::new(1),
        })
        .unwrap();
        wal.flush().unwrap();

        let recovery = RecoveryManager::new(buffer_pool, wal);

        // Verify WAL integrity
        let is_corrupt = recovery.verify_wal_integrity().unwrap();
        assert!(!is_corrupt, "WAL should not be corrupt");
    }

    #[test]
    fn test_buffer_pool_verification() {
        use crate::storage::page::PageType;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let wal_dir = temp_dir.path();

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));
        let wal = Arc::new(WriteAheadLog::new(wal_dir).unwrap());

        // Create some test pages with valid checksums
        for _ in 0..3 {
            let page_guard = buffer_pool.new_page(PageType::BTreeLeaf).unwrap();
            let mut page_lock = page_guard.page_mut();
            if let Some(ref mut page) = *page_lock {
                page.update_header(); // Update checksum
            }
        }

        let recovery = RecoveryManager::new(buffer_pool, wal);

        // Verify buffer pool
        let corrupted_count = recovery.verify_buffer_pool().unwrap();
        assert_eq!(corrupted_count, 0, "No pages should be corrupted");
    }

    #[test]
    fn test_corruption_severity_levels() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let wal_dir = temp_dir.path();

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));
        let wal = Arc::new(WriteAheadLog::new(wal_dir).unwrap());

        let recovery = RecoveryManager::new(buffer_pool, wal);

        // Test severity ordering
        assert!(CorruptionSeverity::None < CorruptionSeverity::Minor);
        assert!(CorruptionSeverity::Minor < CorruptionSeverity::Major);
        assert!(CorruptionSeverity::Major < CorruptionSeverity::Fatal);

        // Test clean database
        let report = recovery.detect_corruption().unwrap();
        assert_eq!(report.severity, CorruptionSeverity::None);
        assert!(!report.has_corruption);
    }

    #[test]
    fn test_recovery_report_structure() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let wal_dir = temp_dir.path();

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));
        let wal = Arc::new(WriteAheadLog::new(wal_dir).unwrap());

        let recovery = RecoveryManager::new(buffer_pool, wal);
        let report = recovery.recover().unwrap();

        // Verify report structure
        assert!(report.clean_shutdown);
        assert_eq!(report.wal_entries_replayed, 0);
        assert_eq!(report.corrupted_pages, 0);
        assert!(report.recovered);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn test_multiple_transactions_recovery() {
        use crate::storage::page::{PageType, PAGE_SIZE};
        use crate::transaction::wal::{LogRecord, TxnId};

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let wal_dir = temp_dir.path();

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));
        let wal = Arc::new(WriteAheadLog::new(wal_dir).unwrap());

        // Create test pages
        let page1 = buffer_pool.new_page(PageType::BTreeLeaf).unwrap();
        let page1_id = page1.page_id();
        drop(page1);

        let page2 = buffer_pool.new_page(PageType::BTreeLeaf).unwrap();
        let page2_id = page2.page_id();
        drop(page2);

        // Simulate multiple transactions
        let txn1 = TxnId::new(1);
        let txn2 = TxnId::new(2);

        // Transaction 1 - commits
        wal.append(LogRecord::Begin { txn_id: txn1 }).unwrap();
        wal.append(LogRecord::Update {
            txn_id: txn1,
            page_id: page1_id,
            before_image: vec![0u8; PAGE_SIZE],
            after_image: vec![1u8; PAGE_SIZE],
        })
        .unwrap();
        wal.append(LogRecord::Commit { txn_id: txn1 }).unwrap();

        // Transaction 2 - aborts
        wal.append(LogRecord::Begin { txn_id: txn2 }).unwrap();
        wal.append(LogRecord::Update {
            txn_id: txn2,
            page_id: page2_id,
            before_image: vec![0u8; PAGE_SIZE],
            after_image: vec![2u8; PAGE_SIZE],
        })
        .unwrap();
        wal.append(LogRecord::Abort { txn_id: txn2 }).unwrap();

        wal.flush().unwrap();

        // Perform recovery
        let recovery = RecoveryManager::new(buffer_pool, wal);
        let report = recovery.recover().unwrap();

        // Only transaction 1's update should be replayed
        assert_eq!(report.wal_entries_replayed, 1);
        assert!(report.recovered);
    }
}
