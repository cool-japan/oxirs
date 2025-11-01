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
        })
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

        // Create new WAL file with .checkpoint extension
        let checkpoint_path = self.wal_path.with_extension("wal.checkpoint");

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

        // TODO: Optionally clean up old checkpoint files

        Ok(())
    }

    /// Recover from WAL after crash
    pub fn recover(&self) -> Result<usize, OxirsError> {
        tracing::info!("Recovering from WAL: {}", self.wal_path.display());

        let recovery = WalRecovery::new(&self.wal_path)?;
        recovery.replay()
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
    pub fn replay(&self) -> Result<usize, OxirsError> {
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
                        // TODO: Apply insert to store
                        tracing::debug!("Recovering insert for tx {}: {:?}", tx_id, quad);
                        recovered += 1;
                    }
                }
                WalEntry::Delete { tx_id, quad } => {
                    if committed_txs.contains(&tx_id) && !aborted_txs.contains(&tx_id) {
                        // TODO: Apply delete to store
                        tracing::debug!("Recovering delete for tx {}: {:?}", tx_id, quad);
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

        // Recover
        let recovered = wal.recover()?;
        assert_eq!(recovered, 1); // One insert operation

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
}
