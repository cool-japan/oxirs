//! Write-Ahead Logging (WAL) for transaction durability
//!
//! This module implements a Write-Ahead Log that ensures durability of transactions
//! by writing all changes to disk before they are applied to the main store.

use crate::model::Quad;
use crate::OxirsError;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Write-Ahead Log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    /// Begin transaction
    Begin { tx_id: u64 },
    /// Insert operation
    Insert { tx_id: u64, quad: Quad },
    /// Delete operation
    Delete { tx_id: u64, quad: Quad },
    /// Commit transaction
    Commit { tx_id: u64 },
    /// Abort transaction
    Abort { tx_id: u64 },
    /// Checkpoint marker
    Checkpoint { tx_id: u64 },
}

/// Write-Ahead Log for transaction durability
pub struct WriteAheadLog {
    /// WAL file path
    wal_path: PathBuf,
    /// Current WAL file
    current_file: BufWriter<File>,
    /// Entry count since last checkpoint
    entry_count: usize,
    /// Checkpoint interval (number of entries)
    checkpoint_interval: usize,
    /// Maximum number of checkpoint files to retain
    max_checkpoints: usize,
}

impl WriteAheadLog {
    /// Create a new Write-Ahead Log
    pub fn new(wal_dir: impl AsRef<Path>) -> Result<Self, OxirsError> {
        let wal_dir = wal_dir.as_ref();

        // Create WAL directory if it doesn't exist
        std::fs::create_dir_all(wal_dir)
            .map_err(|e| OxirsError::Io(format!("Failed to create WAL directory: {}", e)))?;

        let wal_path = wal_dir.join("transaction.wal");

        // Open or create WAL file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)
            .map_err(|e| OxirsError::Io(format!("Failed to open WAL file: {}", e)))?;

        let current_file = BufWriter::new(file);

        Ok(Self {
            wal_path,
            current_file,
            entry_count: 0,
            checkpoint_interval: 1000, // Checkpoint every 1000 entries
            max_checkpoints: 3,        // Keep last 3 checkpoints by default
        })
    }

    /// Create a new Write-Ahead Log with custom configuration
    pub fn with_config(
        wal_dir: impl AsRef<Path>,
        checkpoint_interval: usize,
        max_checkpoints: usize,
    ) -> Result<Self, OxirsError> {
        let wal_dir = wal_dir.as_ref();

        // Create WAL directory if it doesn't exist
        std::fs::create_dir_all(wal_dir)
            .map_err(|e| OxirsError::Io(format!("Failed to create WAL directory: {}", e)))?;

        let wal_path = wal_dir.join("transaction.wal");

        // Open or create WAL file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)
            .map_err(|e| OxirsError::Io(format!("Failed to open WAL file: {}", e)))?;

        let current_file = BufWriter::new(file);

        Ok(Self {
            wal_path,
            current_file,
            entry_count: 0,
            checkpoint_interval,
            max_checkpoints,
        })
    }

    /// Get the WAL file path
    pub fn path(&self) -> &Path {
        &self.wal_path
    }

    /// Append an entry to the WAL
    pub fn append(&mut self, entry: WalEntry) -> Result<(), OxirsError> {
        // Serialize entry to JSON
        let json = serde_json::to_string(&entry)
            .map_err(|e| OxirsError::Serialize(format!("Failed to serialize WAL entry: {}", e)))?;

        // Write entry with newline separator
        writeln!(&mut self.current_file, "{}", json)
            .map_err(|e| OxirsError::Io(format!("Failed to write WAL entry: {}", e)))?;

        self.entry_count += 1;

        // Auto-checkpoint if threshold reached
        if self.entry_count >= self.checkpoint_interval {
            self.checkpoint()?;
        }

        Ok(())
    }

    /// Flush WAL to disk (for durability)
    pub fn flush(&mut self) -> Result<(), OxirsError> {
        self.current_file
            .flush()
            .map_err(|e| OxirsError::Io(format!("Failed to flush WAL: {}", e)))?;

        // Also sync to ensure data is on disk
        self.current_file
            .get_mut()
            .sync_all()
            .map_err(|e| OxirsError::Io(format!("Failed to sync WAL: {}", e)))?;

        Ok(())
    }

    /// Create a checkpoint (truncate old entries)
    pub fn checkpoint(&mut self) -> Result<(), OxirsError> {
        tracing::info!("Creating WAL checkpoint after {} entries", self.entry_count);

        // Flush current buffer
        self.flush()?;

        // Create unique checkpoint filename with timestamp (millisecond precision)
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| OxirsError::Io(format!("Failed to get system time: {}", e)))?
            .as_millis();

        let checkpoint_path = self
            .wal_path
            .with_extension(format!("wal.checkpoint.{}", timestamp));

        // Rename current WAL to checkpoint
        std::fs::rename(&self.wal_path, &checkpoint_path)
            .map_err(|e| OxirsError::Io(format!("Failed to create checkpoint: {}", e)))?;

        // Create new empty WAL file
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.wal_path)
            .map_err(|e| OxirsError::Io(format!("Failed to create new WAL: {}", e)))?;

        self.current_file = BufWriter::new(file);
        self.entry_count = 0;

        // Clean up old checkpoint files
        self.cleanup_old_checkpoints()?;

        Ok(())
    }

    /// Clean up old checkpoint files, keeping only the most recent ones
    fn cleanup_old_checkpoints(&self) -> Result<(), OxirsError> {
        let wal_dir = self
            .wal_path
            .parent()
            .ok_or_else(|| OxirsError::Io("Invalid WAL path".to_string()))?;

        // Find all checkpoint files
        let mut checkpoints = Vec::new();
        let entries = std::fs::read_dir(wal_dir)
            .map_err(|e| OxirsError::Io(format!("Failed to read WAL directory: {}", e)))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| OxirsError::Io(format!("Failed to read directory entry: {}", e)))?;
            let path = entry.path();

            // Check if this is a checkpoint file (ends with a timestamp after .checkpoint)
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.contains(".wal.checkpoint.") {
                    // Get file metadata for sorting by modification time
                    if let Ok(metadata) = entry.metadata() {
                        if let Ok(modified) = metadata.modified() {
                            checkpoints.push((path, modified));
                        }
                    }
                }
            }
        }

        // Sort by modification time (newest first)
        checkpoints.sort_by(|a, b| b.1.cmp(&a.1));

        // Delete old checkpoints beyond max_checkpoints
        if checkpoints.len() > self.max_checkpoints {
            for (old_checkpoint, _) in checkpoints.iter().skip(self.max_checkpoints) {
                tracing::debug!("Removing old checkpoint: {}", old_checkpoint.display());
                if let Err(e) = std::fs::remove_file(old_checkpoint) {
                    tracing::warn!(
                        "Failed to remove old checkpoint {}: {}",
                        old_checkpoint.display(),
                        e
                    );
                }
            }
        }

        Ok(())
    }

    /// Recover from WAL after crash
    ///
    /// This method accepts callback functions to apply inserts and deletes to the store.
    ///
    /// # Arguments
    ///
    /// * `insert_fn` - Callback to insert a quad into the store
    /// * `delete_fn` - Callback to delete a quad from the store
    pub fn recover<F, G>(&self, insert_fn: F, delete_fn: G) -> Result<usize, OxirsError>
    where
        F: FnMut(Quad) -> Result<bool, OxirsError>,
        G: FnMut(&Quad) -> Result<bool, OxirsError>,
    {
        tracing::info!("Recovering from WAL: {}", self.wal_path.display());

        let recovery = WalRecovery::new(&self.wal_path)?;
        recovery.replay(insert_fn, delete_fn)
    }
}

/// WAL recovery after crash
pub struct WalRecovery {
    /// WAL file path
    wal_path: PathBuf,
}

impl WalRecovery {
    /// Create a new WAL recovery handler
    pub fn new(wal_path: impl AsRef<Path>) -> Result<Self, OxirsError> {
        Ok(Self {
            wal_path: wal_path.as_ref().to_path_buf(),
        })
    }

    /// Replay WAL entries to recover state
    ///
    /// This method accepts callback functions to apply inserts and deletes to the store.
    ///
    /// # Arguments
    ///
    /// * `insert_fn` - Callback to insert a quad into the store
    /// * `delete_fn` - Callback to delete a quad from the store
    ///
    /// # Example
    ///
    /// ```ignore
    /// let recovery = WalRecovery::new("./wal/transaction.wal")?;
    /// let mut store = RdfStore::new()?;
    ///
    /// recovery.replay(
    ///     |quad| store.insert_quad(quad),
    ///     |quad| store.remove_quad(quad)
    /// )?;
    /// ```
    pub fn replay<F, G>(&self, mut insert_fn: F, mut delete_fn: G) -> Result<usize, OxirsError>
    where
        F: FnMut(Quad) -> Result<bool, OxirsError>,
        G: FnMut(&Quad) -> Result<bool, OxirsError>,
    {
        if !self.wal_path.exists() {
            tracing::info!("No WAL file found, nothing to recover");
            return Ok(0);
        }

        let file = File::open(&self.wal_path)
            .map_err(|e| OxirsError::Io(format!("Failed to open WAL for recovery: {}", e)))?;

        let reader = BufReader::new(file);
        let mut recovered = 0;
        let mut committed_txs = std::collections::HashSet::new();
        let mut aborted_txs = std::collections::HashSet::new();

        // First pass: identify committed and aborted transactions
        for line in reader.lines() {
            let line =
                line.map_err(|e| OxirsError::Io(format!("Failed to read WAL line: {}", e)))?;

            let entry: WalEntry = serde_json::from_str(&line)
                .map_err(|e| OxirsError::Parse(format!("Failed to parse WAL entry: {}", e)))?;

            match entry {
                WalEntry::Commit { tx_id } => {
                    committed_txs.insert(tx_id);
                }
                WalEntry::Abort { tx_id } => {
                    aborted_txs.insert(tx_id);
                }
                _ => {}
            }
        }

        // Second pass: replay committed transactions
        let file = File::open(&self.wal_path)
            .map_err(|e| OxirsError::Io(format!("Failed to reopen WAL: {}", e)))?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line =
                line.map_err(|e| OxirsError::Io(format!("Failed to read WAL line: {}", e)))?;

            let entry: WalEntry = serde_json::from_str(&line)
                .map_err(|e| OxirsError::Parse(format!("Failed to parse WAL entry: {}", e)))?;

            match entry {
                WalEntry::Insert { tx_id, quad } => {
                    if committed_txs.contains(&tx_id) && !aborted_txs.contains(&tx_id) {
                        // Apply insert to store via callback
                        tracing::debug!("Recovering insert for tx {}: {:?}", tx_id, quad);
                        insert_fn(quad)?;
                        recovered += 1;
                    }
                }
                WalEntry::Delete { tx_id, quad } => {
                    if committed_txs.contains(&tx_id) && !aborted_txs.contains(&tx_id) {
                        // Apply delete to store via callback
                        tracing::debug!("Recovering delete for tx {}: {:?}", tx_id, quad);
                        delete_fn(&quad)?;
                        recovered += 1;
                    }
                }
                _ => {}
            }
        }

        tracing::info!("Recovered {} operations from WAL", recovered);
        Ok(recovered)
    }

    /// Validate WAL integrity
    pub fn validate(&self) -> Result<WalValidation, OxirsError> {
        if !self.wal_path.exists() {
            return Ok(WalValidation {
                is_valid: true,
                total_entries: 0,
                corrupted_entries: 0,
                pending_transactions: 0,
            });
        }

        let file = File::open(&self.wal_path)
            .map_err(|e| OxirsError::Io(format!("Failed to open WAL: {}", e)))?;

        let reader = BufReader::new(file);
        let mut total_entries = 0;
        let mut corrupted_entries = 0;
        let mut active_txs = std::collections::HashSet::new();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => {
                    corrupted_entries += 1;
                    continue;
                }
            };

            let entry: Result<WalEntry, _> = serde_json::from_str(&line);
            match entry {
                Ok(WalEntry::Begin { tx_id }) => {
                    active_txs.insert(tx_id);
                    total_entries += 1;
                }
                Ok(WalEntry::Commit { tx_id }) | Ok(WalEntry::Abort { tx_id }) => {
                    active_txs.remove(&tx_id);
                    total_entries += 1;
                }
                Ok(_) => {
                    total_entries += 1;
                }
                Err(_) => {
                    corrupted_entries += 1;
                }
            }
        }

        Ok(WalValidation {
            is_valid: corrupted_entries == 0,
            total_entries,
            corrupted_entries,
            pending_transactions: active_txs.len(),
        })
    }
}

/// WAL validation result
#[derive(Debug, Clone)]
pub struct WalValidation {
    /// Whether the WAL is valid
    pub is_valid: bool,
    /// Total number of entries
    pub total_entries: usize,
    /// Number of corrupted entries
    pub corrupted_entries: usize,
    /// Number of pending transactions
    pub pending_transactions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GraphName, Literal, NamedNode, Object, Predicate, Subject};
    use tempfile::tempdir;

    fn create_test_quad() -> Quad {
        Quad::new(
            Subject::NamedNode(NamedNode::new("http://s").unwrap()),
            Predicate::NamedNode(NamedNode::new("http://p").unwrap()),
            Object::Literal(Literal::new("value")),
            GraphName::DefaultGraph,
        )
    }

    #[test]
    fn test_wal_creation() -> Result<(), OxirsError> {
        let dir = tempdir().map_err(|e| OxirsError::Io(e.to_string()))?;
        let _wal = WriteAheadLog::new(dir.path())?;

        Ok(())
    }

    #[test]
    fn test_wal_append() -> Result<(), OxirsError> {
        let dir = tempdir().map_err(|e| OxirsError::Io(e.to_string()))?;
        let mut wal = WriteAheadLog::new(dir.path())?;

        let quad = create_test_quad();
        wal.append(WalEntry::Insert { tx_id: 1, quad })?;
        wal.flush()?;

        Ok(())
    }

    #[test]
    fn test_wal_recovery() -> Result<(), OxirsError> {
        let dir = tempdir().map_err(|e| OxirsError::Io(e.to_string()))?;
        let mut wal = WriteAheadLog::new(dir.path())?;

        let quad = create_test_quad();

        // Write some operations
        wal.append(WalEntry::Begin { tx_id: 1 })?;
        wal.append(WalEntry::Insert {
            tx_id: 1,
            quad: quad.clone(),
        })?;
        wal.append(WalEntry::Commit { tx_id: 1 })?;
        wal.flush()?;

        // Recover with mock callbacks
        let mut inserted_quads = Vec::new();
        let mut deleted_quads = Vec::new();

        let recovered = wal.recover(
            |quad| {
                inserted_quads.push(quad);
                Ok(true)
            },
            |quad| {
                deleted_quads.push(quad.clone());
                Ok(true)
            },
        )?;

        assert_eq!(recovered, 1); // One insert operation
        assert_eq!(inserted_quads.len(), 1); // One quad was inserted

        Ok(())
    }

    #[test]
    fn test_wal_validation() -> Result<(), OxirsError> {
        let dir = tempdir().map_err(|e| OxirsError::Io(e.to_string()))?;
        let mut wal = WriteAheadLog::new(dir.path())?;

        let quad = create_test_quad();
        wal.append(WalEntry::Insert { tx_id: 1, quad })?;
        wal.flush()?;

        let recovery = WalRecovery::new(dir.path().join("transaction.wal"))?;
        let validation = recovery.validate()?;

        assert!(validation.is_valid);
        assert_eq!(validation.corrupted_entries, 0);

        Ok(())
    }

    #[test]
    fn test_wal_checkpoint() -> Result<(), OxirsError> {
        let dir = tempdir().map_err(|e| OxirsError::Io(e.to_string()))?;
        let mut wal = WriteAheadLog::new(dir.path())?;

        let quad = create_test_quad();
        for i in 0..10 {
            wal.append(WalEntry::Insert {
                tx_id: i,
                quad: quad.clone(),
            })?;
        }

        wal.checkpoint()?;
        assert_eq!(wal.entry_count, 0);

        Ok(())
    }

    #[test]
    fn test_wal_checkpoint_cleanup() -> Result<(), OxirsError> {
        let dir = tempdir().map_err(|e| OxirsError::Io(e.to_string()))?;

        // Create WAL with max_checkpoints = 2
        let mut wal = WriteAheadLog::with_config(dir.path(), 5, 2)?;

        let quad = create_test_quad();

        // Create multiple checkpoints (more than max_checkpoints)
        for checkpoint_num in 0..5 {
            // Add some entries
            for i in 0..10 {
                wal.append(WalEntry::Insert {
                    tx_id: checkpoint_num * 10 + i,
                    quad: quad.clone(),
                })?;
            }

            // Create checkpoint
            wal.checkpoint()?;

            // Small delay to ensure different timestamps (millisecond precision)
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Count checkpoint files
        let mut checkpoint_count = 0;
        let entries = std::fs::read_dir(dir.path())?;
        for entry in entries {
            let entry = entry?;
            if let Some(filename) = entry.path().file_name().and_then(|n| n.to_str()) {
                if filename.contains(".wal.checkpoint.") {
                    checkpoint_count += 1;
                }
            }
        }

        // Should have only max_checkpoints files (2)
        assert_eq!(
            checkpoint_count, 2,
            "Should keep only 2 most recent checkpoints"
        );

        Ok(())
    }
}
