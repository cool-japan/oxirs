//! Write-Ahead Logging (WAL) for crash recovery
//!
//! This module provides comprehensive write-ahead logging for vector index operations,
//! enabling crash recovery and ensuring data durability. The WAL records all modifications
//! before they are applied to the index, allowing the system to recover from crashes by
//! replaying the log.
//!
//! # Features
//!
//! - Transaction-based logging
//! - Automatic crash recovery
//! - Log compaction and checkpointing
//! - Concurrent write support with proper synchronization
//! - Configurable fsync behavior for performance tuning
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐
//! │ Index Ops   │
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────┐     ┌──────────────┐
//! │ WAL Writer  │────▶│ Log File     │
//! └─────────────┘     └──────────────┘
//!        │                    │
//!        │                    │ (on crash)
//!        ▼                    ▼
//! ┌─────────────┐     ┌──────────────┐
//! │ Index       │◀────│ WAL Recovery │
//! └─────────────┘     └──────────────┘
//! ```

use anyhow::{anyhow, Result};
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// WAL magic number for file format validation
const WAL_MAGIC: &[u8; 4] = b"WALV"; // WAL Vector

/// WAL format version
const WAL_VERSION: u32 = 1;

/// Write-Ahead Log entry representing a single operation
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub enum WalEntry {
    /// Insert a new vector
    Insert {
        id: String,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: u64,
    },
    /// Update an existing vector
    Update {
        id: String,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: u64,
    },
    /// Delete a vector
    Delete { id: String, timestamp: u64 },
    /// Batch operation (multiple entries)
    Batch {
        entries: Vec<WalEntry>,
        timestamp: u64,
    },
    /// Checkpoint marker (all operations before this are persisted)
    Checkpoint {
        sequence_number: u64,
        timestamp: u64,
    },
    /// Transaction begin
    BeginTransaction { transaction_id: u64, timestamp: u64 },
    /// Transaction commit
    CommitTransaction { transaction_id: u64, timestamp: u64 },
    /// Transaction abort
    AbortTransaction { transaction_id: u64, timestamp: u64 },
}

impl WalEntry {
    /// Get the timestamp of this entry
    pub fn timestamp(&self) -> u64 {
        match self {
            WalEntry::Insert { timestamp, .. }
            | WalEntry::Update { timestamp, .. }
            | WalEntry::Delete { timestamp, .. }
            | WalEntry::Batch { timestamp, .. }
            | WalEntry::Checkpoint { timestamp, .. }
            | WalEntry::BeginTransaction { timestamp, .. }
            | WalEntry::CommitTransaction { timestamp, .. }
            | WalEntry::AbortTransaction { timestamp, .. } => *timestamp,
        }
    }

    /// Check if this is a checkpoint entry
    pub fn is_checkpoint(&self) -> bool {
        matches!(self, WalEntry::Checkpoint { .. })
    }
}

/// WAL configuration
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Directory where WAL files are stored
    pub wal_directory: PathBuf,
    /// Maximum size of a single WAL file before rotation (in bytes)
    pub max_file_size: u64,
    /// Whether to call fsync after each write (slower but safer)
    pub sync_on_write: bool,
    /// Checkpoint interval (number of operations)
    pub checkpoint_interval: u64,
    /// Keep this many checkpoint files
    pub checkpoint_retention: usize,
    /// Buffer size for WAL writes
    pub buffer_size: usize,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            wal_directory: PathBuf::from("./wal"),
            max_file_size: 100 * 1024 * 1024, // 100MB
            sync_on_write: false,             // Better performance, acceptable risk
            checkpoint_interval: 10000,
            checkpoint_retention: 3,
            buffer_size: 64 * 1024, // 64KB buffer
        }
    }
}

/// Write-Ahead Log manager
pub struct WalManager {
    config: WalConfig,
    current_file: Arc<Mutex<Option<BufWriter<File>>>>,
    current_file_path: Arc<Mutex<PathBuf>>,
    sequence_number: Arc<Mutex<u64>>,
    last_checkpoint: Arc<Mutex<u64>>,
}

impl WalManager {
    /// Create a new WAL manager
    pub fn new(config: WalConfig) -> Result<Self> {
        // Ensure WAL directory exists
        std::fs::create_dir_all(&config.wal_directory)?;

        let manager = Self {
            config,
            current_file: Arc::new(Mutex::new(None)),
            current_file_path: Arc::new(Mutex::new(PathBuf::new())),
            sequence_number: Arc::new(Mutex::new(0)),
            last_checkpoint: Arc::new(Mutex::new(0)),
        };

        // Open or create the current WAL file
        manager.rotate_wal_file()?;

        Ok(manager)
    }

    /// Append an entry to the WAL
    pub fn append(&self, entry: WalEntry) -> Result<u64> {
        let seq = {
            let mut seq_guard = self.sequence_number.lock().unwrap();
            let seq = *seq_guard;
            *seq_guard += 1;
            seq
        };

        // Write to file
        let needs_checkpoint = {
            let mut file_guard = self.current_file.lock().unwrap();

            if let Some(ref mut writer) = *file_guard {
                // Serialize the entry
                let entry_bytes = bincode::encode_to_vec(&entry, bincode::config::standard())
                    .map_err(|e| anyhow!("Failed to serialize WAL entry: {}", e))?;
                let entry_len = entry_bytes.len() as u32;

                // Write sequence number, length, and data
                writer.write_all(&seq.to_le_bytes())?;
                writer.write_all(&entry_len.to_le_bytes())?;
                writer.write_all(&entry_bytes)?;

                if self.config.sync_on_write {
                    writer.flush()?;
                    writer.get_ref().sync_all()?;
                }

                // Check if file rotation is needed
                let needs_rotation = if let Ok(metadata) = writer.get_ref().metadata() {
                    metadata.len() >= self.config.max_file_size
                } else {
                    false
                };

                if needs_rotation {
                    drop(file_guard);
                    self.rotate_wal_file()?;
                }

                // Check if checkpoint is needed
                let last_checkpoint = *self.last_checkpoint.lock().unwrap();
                seq - last_checkpoint >= self.config.checkpoint_interval
            } else {
                return Err(anyhow!("WAL file not open"));
            }
        };

        // Checkpoint outside of lock
        if needs_checkpoint {
            self.checkpoint(seq)?;
        }

        Ok(seq)
    }

    /// Create a checkpoint
    pub fn checkpoint(&self, sequence_number: u64) -> Result<()> {
        tracing::info!("Creating WAL checkpoint at sequence {}", sequence_number);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let checkpoint_entry = WalEntry::Checkpoint {
            sequence_number,
            timestamp,
        };

        // Write checkpoint directly without going through append() to avoid recursion
        let seq = {
            let mut seq_guard = self.sequence_number.lock().unwrap();
            let seq = *seq_guard;
            *seq_guard += 1;
            seq
        };

        {
            let mut file_guard = self.current_file.lock().unwrap();
            if let Some(ref mut writer) = *file_guard {
                let entry_bytes =
                    bincode::encode_to_vec(&checkpoint_entry, bincode::config::standard())
                        .map_err(|e| anyhow!("Failed to serialize checkpoint entry: {}", e))?;
                let entry_len = entry_bytes.len() as u32;

                writer.write_all(&seq.to_le_bytes())?;
                writer.write_all(&entry_len.to_le_bytes())?;
                writer.write_all(&entry_bytes)?;

                if self.config.sync_on_write {
                    writer.flush()?;
                    writer.get_ref().sync_all()?;
                }
            }
        }

        let mut last_checkpoint = self.last_checkpoint.lock().unwrap();
        *last_checkpoint = sequence_number;

        // Cleanup old WAL files
        self.cleanup_old_files()?;

        Ok(())
    }

    /// Rotate to a new WAL file
    fn rotate_wal_file(&self) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let filename = format!("wal-{:016x}.log", timestamp);
        let filepath = self.config.wal_directory.join(&filename);

        tracing::info!("Rotating WAL to new file: {:?}", filepath);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filepath)?;

        let mut writer = BufWriter::with_capacity(self.config.buffer_size, file);

        // Write WAL file header
        writer.write_all(WAL_MAGIC)?;
        writer.write_all(&WAL_VERSION.to_le_bytes())?;
        writer.write_all(&timestamp.to_le_bytes())?;

        if self.config.sync_on_write {
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }

        let mut file_guard = self.current_file.lock().unwrap();
        let mut path_guard = self.current_file_path.lock().unwrap();

        // Flush and close old file
        if let Some(mut old_writer) = file_guard.take() {
            old_writer.flush()?;
        }

        *file_guard = Some(writer);
        *path_guard = filepath;

        Ok(())
    }

    /// Clean up old WAL files (keep only recent checkpoints)
    fn cleanup_old_files(&self) -> Result<()> {
        let mut wal_files: Vec<_> = std::fs::read_dir(&self.config.wal_directory)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .map(|s| s.starts_with("wal-") && s.ends_with(".log"))
                    .unwrap_or(false)
            })
            .collect();

        // Sort by filename (timestamp-based)
        wal_files.sort_by_key(|entry| entry.file_name());

        // Keep the most recent files
        if wal_files.len() > self.config.checkpoint_retention {
            let to_remove = wal_files.len() - self.config.checkpoint_retention;
            for entry in wal_files.iter().take(to_remove) {
                tracing::info!("Removing old WAL file: {:?}", entry.path());
                std::fs::remove_file(entry.path())?;
            }
        }

        Ok(())
    }

    /// Recover from WAL files
    pub fn recover(&self) -> Result<Vec<WalEntry>> {
        tracing::info!("Starting WAL recovery");

        let mut all_entries = Vec::new();
        let mut last_checkpoint_seq = 0u64;

        // Find all WAL files
        let mut wal_files: Vec<_> = std::fs::read_dir(&self.config.wal_directory)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .map(|s| s.starts_with("wal-") && s.ends_with(".log"))
                    .unwrap_or(false)
            })
            .collect();

        // Sort by filename (timestamp-based)
        wal_files.sort_by_key(|entry| entry.file_name());

        // Read all WAL files
        for entry in wal_files {
            let path = entry.path();
            tracing::debug!("Reading WAL file: {:?}", path);

            let file = File::open(&path)?;
            let mut reader = BufReader::new(file);

            // Verify magic number
            let mut magic = [0u8; 4];
            reader.read_exact(&mut magic)?;
            if &magic != WAL_MAGIC {
                tracing::warn!("Invalid WAL file magic number: {:?}", path);
                continue;
            }

            // Read version
            let mut version_bytes = [0u8; 4];
            reader.read_exact(&mut version_bytes)?;
            let version = u32::from_le_bytes(version_bytes);
            if version != WAL_VERSION {
                tracing::warn!("Unsupported WAL version {} in {:?}", version, path);
                continue;
            }

            // Read file timestamp
            let mut timestamp_bytes = [0u8; 8];
            reader.read_exact(&mut timestamp_bytes)?;

            // Read entries with robust error handling for incomplete writes
            loop {
                // Read sequence number
                let mut seq_bytes = [0u8; 8];
                match reader.read_exact(&mut seq_bytes) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        tracing::debug!("Reached end of WAL file (expected)");
                        break;
                    }
                    Err(e) => return Err(e.into()),
                }
                let seq = u64::from_le_bytes(seq_bytes);

                // Read entry length
                let mut len_bytes = [0u8; 4];
                match reader.read_exact(&mut len_bytes) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        tracing::warn!(
                            "Incomplete entry at sequence {}: missing length field. Skipping rest of file.",
                            seq
                        );
                        break;
                    }
                    Err(e) => return Err(e.into()),
                }
                let len = u32::from_le_bytes(len_bytes);

                // Sanity check on entry length (prevent excessive memory allocation)
                if len > 100_000_000 {
                    // 100MB max entry size
                    tracing::warn!(
                        "Entry at sequence {} has suspicious length {}. Possibly corrupted. Skipping.",
                        seq,
                        len
                    );
                    break;
                }

                // Read entry data
                let mut entry_bytes = vec![0u8; len as usize];
                match reader.read_exact(&mut entry_bytes) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        tracing::warn!(
                            "Incomplete entry at sequence {}: expected {} bytes but reached EOF. Skipping rest of file.",
                            seq,
                            len
                        );
                        break;
                    }
                    Err(e) => return Err(e.into()),
                }

                // Deserialize entry
                let entry: WalEntry =
                    match bincode::decode_from_slice(&entry_bytes, bincode::config::standard()) {
                        Ok((e, _)) => e,
                        Err(e) => {
                            tracing::warn!(
                                "Failed to deserialize entry at sequence {}: {}. Skipping entry.",
                                seq,
                                e
                            );
                            continue; // Skip corrupted entry but continue reading
                        }
                    };

                // Track last checkpoint
                if let WalEntry::Checkpoint {
                    sequence_number, ..
                } = &entry
                {
                    last_checkpoint_seq = *sequence_number;
                }

                all_entries.push((seq, entry));
            }
        }

        // Filter entries after last checkpoint
        // Note: If last_checkpoint_seq == 0 (no checkpoint), recover all entries including seq 0
        // Otherwise, only recover entries strictly after the checkpoint
        let recovered_entries: Vec<_> = all_entries
            .iter()
            .filter(|(seq, _)| {
                if last_checkpoint_seq == 0 {
                    true // No checkpoint, recover everything
                } else {
                    *seq > last_checkpoint_seq // Checkpoint exists, only after it
                }
            })
            .map(|(_, entry)| entry.clone())
            .collect();

        tracing::info!(
            "Recovered {} entries from WAL (after checkpoint {})",
            recovered_entries.len(),
            last_checkpoint_seq
        );

        // Update sequence number based on the maximum sequence number seen
        if let Some((max_seq, _)) = all_entries.iter().max_by_key(|(seq, _)| seq) {
            let mut seq = self.sequence_number.lock().unwrap();
            *seq = max_seq + 1;
        }

        Ok(recovered_entries)
    }

    /// Flush all pending writes to disk
    pub fn flush(&self) -> Result<()> {
        let mut file_guard = self.current_file.lock().unwrap();
        if let Some(ref mut writer) = *file_guard {
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        Ok(())
    }

    /// Get current sequence number
    pub fn current_sequence(&self) -> u64 {
        *self.sequence_number.lock().unwrap()
    }

    /// Get last checkpoint sequence number
    pub fn last_checkpoint_sequence(&self) -> u64 {
        *self.last_checkpoint.lock().unwrap()
    }
}

impl Drop for WalManager {
    fn drop(&mut self) {
        // Ensure all data is flushed on drop
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_wal_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig {
            wal_directory: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let wal = WalManager::new(config).unwrap();
        assert_eq!(wal.current_sequence(), 0);
    }

    #[test]
    fn test_wal_append() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig {
            wal_directory: temp_dir.path().to_path_buf(),
            sync_on_write: true,
            ..Default::default()
        };

        let wal = WalManager::new(config).unwrap();

        let entry = WalEntry::Insert {
            id: "vec1".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
            timestamp: 12345,
        };

        let seq = wal.append(entry).unwrap();
        assert_eq!(seq, 0);
    }

    #[test]
    fn test_wal_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig {
            wal_directory: temp_dir.path().to_path_buf(),
            sync_on_write: true,
            checkpoint_interval: 100,
            ..Default::default()
        };

        // Write some entries
        {
            let wal = WalManager::new(config.clone()).unwrap();

            for i in 0..5 {
                let entry = WalEntry::Insert {
                    id: format!("vec{}", i),
                    vector: vec![i as f32, (i * 2) as f32],
                    metadata: None,
                    timestamp: (i + 1) * 1000, // Use unique timestamps
                };
                wal.append(entry).unwrap();
            }

            wal.flush().unwrap();
            // Ensure Drop is called to flush everything
            drop(wal);
        }

        // Small delay to ensure file is written
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Recover
        {
            let wal = WalManager::new(config).unwrap();
            let recovered = wal.recover().unwrap();

            // Should recover 5 entries
            assert_eq!(
                recovered.len(),
                5,
                "Expected exactly 5 entries, got {}",
                recovered.len()
            );

            // Verify all timestamps are present
            let timestamps: Vec<u64> = recovered.iter().map(|e| e.timestamp()).collect();
            assert_eq!(timestamps, vec![1000, 2000, 3000, 4000, 5000]);
        }
    }

    #[test]
    fn test_wal_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig {
            wal_directory: temp_dir.path().to_path_buf(),
            sync_on_write: true,
            checkpoint_interval: 3,
            ..Default::default()
        };

        let wal = WalManager::new(config).unwrap();

        // Write entries (should trigger checkpoint)
        for i in 0..5 {
            let entry = WalEntry::Insert {
                id: format!("vec{}", i),
                vector: vec![i as f32],
                metadata: None,
                timestamp: i,
            };
            wal.append(entry).unwrap();
        }

        assert!(wal.last_checkpoint_sequence() > 0);
    }

    #[test]
    fn test_wal_batch_operation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig {
            wal_directory: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let wal = WalManager::new(config).unwrap();

        let batch = WalEntry::Batch {
            entries: vec![
                WalEntry::Insert {
                    id: "vec1".to_string(),
                    vector: vec![1.0],
                    metadata: None,
                    timestamp: 1,
                },
                WalEntry::Update {
                    id: "vec2".to_string(),
                    vector: vec![2.0],
                    metadata: None,
                    timestamp: 2,
                },
            ],
            timestamp: 3,
        };

        wal.append(batch).unwrap();
        wal.flush().unwrap();
    }

    #[test]
    fn test_wal_transaction() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig {
            wal_directory: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let wal = WalManager::new(config).unwrap();

        // Begin transaction
        wal.append(WalEntry::BeginTransaction {
            transaction_id: 1,
            timestamp: 100,
        })
        .unwrap();

        // Operations
        wal.append(WalEntry::Insert {
            id: "vec1".to_string(),
            vector: vec![1.0],
            metadata: None,
            timestamp: 101,
        })
        .unwrap();

        // Commit transaction
        wal.append(WalEntry::CommitTransaction {
            transaction_id: 1,
            timestamp: 102,
        })
        .unwrap();

        wal.flush().unwrap();
    }
}
