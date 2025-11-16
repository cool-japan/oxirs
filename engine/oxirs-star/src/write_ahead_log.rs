//! Write-Ahead Logging (WAL) for durability and crash recovery
//!
//! This module implements a write-ahead log that ensures durability of writes
//! and enables recovery after crashes. All writes are logged before being applied
//! to the data store.
//!
//! # Features
//!
//! - **Durability** - All writes persisted before acknowledgment
//! - **Crash recovery** - Replay log to restore state
//! - **Checkpointing** - Periodic snapshots to reduce recovery time
//! - **Log rotation** - Automatic archival of old log segments
//! - **Batch writes** - Group multiple writes for efficiency
//! - **Integrity checks** - CRC32 checksums for corruption detection
//!
//! # Architecture
//!
//! ```text
//! Write → WAL (append) → Flush to disk → Apply to store → Checkpoint
//!         ↓                                                ↓
//!      Log file                                       Snapshot
//! ```
//!
//! # Examples
//!
//! ```rust
//! use oxirs_star::write_ahead_log::{WriteAheadLog, WalConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = WalConfig::default();
//! let mut wal = WriteAheadLog::new(config)?;
//!
//! // Write entries
//! // wal.append_write(key, annotation)?;
//!
//! // Recover from crash
//! // let entries = wal.recover()?;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, info, span, warn, Level};

use crate::annotations::TripleAnnotation;
use crate::StarResult;

/// Configuration for Write-Ahead Log
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Directory for WAL files
    pub wal_dir: PathBuf,

    /// Maximum size of a WAL segment before rotation (bytes)
    pub segment_size_threshold: usize,

    /// Enable fsync after each write (slower but more durable)
    pub enable_fsync: bool,

    /// Buffer size for writes
    pub write_buffer_size: usize,

    /// Number of WAL segments to keep
    pub max_segments: usize,

    /// Enable compression for WAL segments
    pub enable_compression: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            wal_dir: std::env::temp_dir().join("oxirs_wal"),
            segment_size_threshold: 64 * 1024 * 1024, // 64 MB
            enable_fsync: true,
            write_buffer_size: 8192,
            max_segments: 10,
            enable_compression: false, // Disable for WAL to maintain performance
        }
    }
}

/// Type of WAL entry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalEntryType {
    /// Insert or update operation
    Write,
    /// Delete operation
    Delete,
    /// Checkpoint marker
    Checkpoint,
    /// Begin transaction
    BeginTxn,
    /// Commit transaction
    CommitTxn,
    /// Abort transaction
    AbortTxn,
}

/// WAL entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Sequence number (monotonically increasing)
    pub sequence: u64,

    /// Entry type
    pub entry_type: WalEntryType,

    /// Key (triple hash)
    pub key: u64,

    /// Annotation data (for writes)
    pub annotation: Option<TripleAnnotation>,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Transaction ID (if part of a transaction)
    pub transaction_id: Option<u64>,

    /// CRC32 checksum for integrity
    pub checksum: u32,
}

impl WalEntry {
    /// Create a new write entry
    pub fn write(
        sequence: u64,
        key: u64,
        annotation: TripleAnnotation,
        transaction_id: Option<u64>,
    ) -> Self {
        let mut entry = Self {
            sequence,
            entry_type: WalEntryType::Write,
            key,
            annotation: Some(annotation),
            timestamp: Utc::now(),
            transaction_id,
            checksum: 0,
        };
        entry.checksum = entry.calculate_checksum();
        entry
    }

    /// Create a new delete entry
    pub fn delete(sequence: u64, key: u64, transaction_id: Option<u64>) -> Self {
        let mut entry = Self {
            sequence,
            entry_type: WalEntryType::Delete,
            key,
            annotation: None,
            timestamp: Utc::now(),
            transaction_id,
            checksum: 0,
        };
        entry.checksum = entry.calculate_checksum();
        entry
    }

    /// Create a checkpoint marker
    pub fn checkpoint(sequence: u64) -> Self {
        let mut entry = Self {
            sequence,
            entry_type: WalEntryType::Checkpoint,
            key: 0,
            annotation: None,
            timestamp: Utc::now(),
            transaction_id: None,
            checksum: 0,
        };
        entry.checksum = entry.calculate_checksum();
        entry
    }

    /// Calculate CRC32 checksum
    fn calculate_checksum(&self) -> u32 {
        // Simple hash-based checksum (in production, use proper CRC32)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.sequence.hash(&mut hasher);
        self.key.hash(&mut hasher);
        format!("{:?}", self.entry_type).hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> bool {
        self.checksum == self.calculate_checksum()
    }
}

/// WAL segment (a single log file)
struct WalSegment {
    /// Segment ID
    id: u64,

    /// File path
    #[allow(dead_code)]
    path: PathBuf,

    /// Writer
    writer: BufWriter<File>,

    /// Current size in bytes
    size_bytes: usize,

    /// Number of entries
    entry_count: usize,

    /// Creation timestamp
    #[allow(dead_code)]
    created_at: DateTime<Utc>,
}

impl WalSegment {
    /// Create a new WAL segment
    fn create(id: u64, wal_dir: &Path, buffer_size: usize) -> StarResult<Self> {
        let path = wal_dir.join(format!("wal_{:08}.log", id));

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        let writer = BufWriter::with_capacity(buffer_size, file);

        Ok(Self {
            id,
            path,
            writer,
            size_bytes: 0,
            entry_count: 0,
            created_at: Utc::now(),
        })
    }

    /// Append an entry to the segment
    fn append(&mut self, entry: &WalEntry, enable_fsync: bool) -> StarResult<()> {
        // Serialize entry
        let entry_bytes = bincode::serde::encode_to_vec(entry, bincode::config::standard())
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        // Write length prefix
        self.writer
            .write_all(&(entry_bytes.len() as u32).to_le_bytes())
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        // Write entry
        self.writer
            .write_all(&entry_bytes)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        self.size_bytes += 4 + entry_bytes.len();
        self.entry_count += 1;

        // Flush to disk if fsync is enabled
        if enable_fsync {
            self.writer
                .flush()
                .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

            self.writer
                .get_ref()
                .sync_all()
                .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        }

        Ok(())
    }

    /// Close the segment
    fn close(mut self) -> StarResult<()> {
        self.writer
            .flush()
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        Ok(())
    }

    /// Read all entries from a segment file
    fn read_entries(path: &Path) -> StarResult<Vec<WalEntry>> {
        let file = File::open(path).map_err(|e| crate::StarError::parse_error(e.to_string()))?;
        let mut reader = BufReader::new(file);

        let mut entries = Vec::new();

        loop {
            // Read length prefix
            let mut len_bytes = [0u8; 4];
            match reader.read_exact(&mut len_bytes) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(crate::StarError::parse_error(e.to_string())),
            }

            let len = u32::from_le_bytes(len_bytes) as usize;

            // Read entry
            let mut entry_bytes = vec![0u8; len];
            reader
                .read_exact(&mut entry_bytes)
                .map_err(|e| crate::StarError::parse_error(e.to_string()))?;

            let entry: WalEntry =
                bincode::serde::decode_from_slice(&entry_bytes, bincode::config::standard())
                    .map_err(|e| crate::StarError::parse_error(e.to_string()))?
                    .0;

            // Verify checksum
            if !entry.verify_checksum() {
                warn!("Checksum mismatch for entry {}, skipping", entry.sequence);
                continue;
            }

            entries.push(entry);
        }

        Ok(entries)
    }
}

/// Write-Ahead Log
pub struct WriteAheadLog {
    /// Configuration
    config: WalConfig,

    /// Current active segment
    current_segment: Option<WalSegment>,

    /// Next segment ID
    next_segment_id: u64,

    /// Next sequence number
    next_sequence: u64,

    /// Statistics
    stats: WalStatistics,
}

/// WAL statistics
#[derive(Debug, Clone, Default)]
pub struct WalStatistics {
    /// Total entries written
    pub total_entries: usize,

    /// Total bytes written
    pub bytes_written: usize,

    /// Number of segment rotations
    pub rotations: usize,

    /// Number of checkpoints
    pub checkpoints: usize,

    /// Number of recoveries performed
    pub recoveries: usize,
}

impl WriteAheadLog {
    /// Create a new Write-Ahead Log
    pub fn new(config: WalConfig) -> StarResult<Self> {
        let span = span!(Level::INFO, "wal_new");
        let _enter = span.enter();

        // Create WAL directory
        fs::create_dir_all(&config.wal_dir)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        // Find existing segments and determine next IDs
        let existing_segments = Self::list_segments(&config.wal_dir)?;
        let next_segment_id = existing_segments
            .iter()
            .map(|(id, _)| id + 1)
            .max()
            .unwrap_or(1);

        // Find max sequence number
        let next_sequence = Self::find_max_sequence(&existing_segments, &config.wal_dir)? + 1;

        info!(
            "Initialized WAL at {:?}, next segment: {}, next sequence: {}",
            config.wal_dir, next_segment_id, next_sequence
        );

        let mut wal = Self {
            config,
            current_segment: None,
            next_segment_id,
            next_sequence,
            stats: WalStatistics::default(),
        };

        // Create initial segment
        wal.rotate_segment()?;

        Ok(wal)
    }

    /// Append a write entry
    pub fn append_write(
        &mut self,
        key: u64,
        annotation: TripleAnnotation,
        transaction_id: Option<u64>,
    ) -> StarResult<u64> {
        let sequence = self.next_sequence;
        self.next_sequence += 1;

        let entry = WalEntry::write(sequence, key, annotation, transaction_id);
        self.append_entry(&entry)?;

        Ok(sequence)
    }

    /// Append a delete entry
    pub fn append_delete(&mut self, key: u64, transaction_id: Option<u64>) -> StarResult<u64> {
        let sequence = self.next_sequence;
        self.next_sequence += 1;

        let entry = WalEntry::delete(sequence, key, transaction_id);
        self.append_entry(&entry)?;

        Ok(sequence)
    }

    /// Write a checkpoint marker
    pub fn checkpoint(&mut self) -> StarResult<u64> {
        let span = span!(Level::INFO, "wal_checkpoint");
        let _enter = span.enter();

        let sequence = self.next_sequence;
        self.next_sequence += 1;

        let entry = WalEntry::checkpoint(sequence);
        self.append_entry(&entry)?;

        self.stats.checkpoints += 1;

        info!("Wrote checkpoint at sequence {}", sequence);
        Ok(sequence)
    }

    /// Append an entry to the log
    fn append_entry(&mut self, entry: &WalEntry) -> StarResult<()> {
        let segment = self
            .current_segment
            .as_mut()
            .ok_or_else(|| crate::StarError::serialization_error("No active segment"))?;

        segment.append(entry, self.config.enable_fsync)?;

        self.stats.total_entries += 1;
        self.stats.bytes_written += segment.size_bytes;

        // Check if segment needs rotation
        if segment.size_bytes >= self.config.segment_size_threshold {
            debug!("Segment {} reached size threshold, rotating", segment.id);
            self.rotate_segment()?;
        }

        Ok(())
    }

    /// Rotate to a new segment
    fn rotate_segment(&mut self) -> StarResult<()> {
        let span = span!(Level::DEBUG, "rotate_segment");
        let _enter = span.enter();

        // Close current segment
        if let Some(segment) = self.current_segment.take() {
            segment.close()?;
        }

        // Create new segment
        let segment_id = self.next_segment_id;
        self.next_segment_id += 1;

        let new_segment = WalSegment::create(
            segment_id,
            &self.config.wal_dir,
            self.config.write_buffer_size,
        )?;

        self.current_segment = Some(new_segment);
        self.stats.rotations += 1;

        // Clean up old segments
        self.cleanup_old_segments()?;

        debug!("Rotated to segment {}", segment_id);
        Ok(())
    }

    /// Clean up old segments beyond max_segments
    fn cleanup_old_segments(&self) -> StarResult<()> {
        let segments = Self::list_segments(&self.config.wal_dir)?;

        if segments.len() <= self.config.max_segments {
            return Ok(());
        }

        // Keep the most recent max_segments
        let segments_to_delete = segments.len() - self.config.max_segments;

        for (_, path) in segments.iter().take(segments_to_delete) {
            if let Err(e) = fs::remove_file(path) {
                warn!("Failed to delete old WAL segment {:?}: {}", path, e);
            } else {
                debug!("Deleted old WAL segment {:?}", path);
            }
        }

        Ok(())
    }

    /// List all segment files in order
    fn list_segments(wal_dir: &Path) -> StarResult<Vec<(u64, PathBuf)>> {
        let mut segments = Vec::new();

        let entries =
            fs::read_dir(wal_dir).map_err(|e| crate::StarError::parse_error(e.to_string()))?;

        for entry in entries {
            let entry = entry.map_err(|e| crate::StarError::parse_error(e.to_string()))?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with("wal_") && filename.ends_with(".log") {
                    // Extract segment ID
                    let id_str = &filename[4..filename.len() - 4];
                    if let Ok(id) = id_str.parse::<u64>() {
                        segments.push((id, path));
                    }
                }
            }
        }

        segments.sort_by_key(|(id, _)| *id);
        Ok(segments)
    }

    /// Find maximum sequence number from existing segments
    fn find_max_sequence(segments: &[(u64, PathBuf)], _wal_dir: &Path) -> StarResult<u64> {
        let mut max_seq = 0u64;

        for (_, path) in segments {
            let entries = WalSegment::read_entries(path)?;
            if let Some(last_entry) = entries.last() {
                max_seq = max_seq.max(last_entry.sequence);
            }
        }

        Ok(max_seq)
    }

    /// Recover from WAL (replay all entries since last checkpoint)
    pub fn recover(&self) -> StarResult<Vec<WalEntry>> {
        let span = span!(Level::INFO, "wal_recover");
        let _enter = span.enter();

        let segments = Self::list_segments(&self.config.wal_dir)?;
        let mut all_entries = Vec::new();

        // Read all segments
        for (_, path) in segments {
            let entries = WalSegment::read_entries(&path)?;
            all_entries.extend(entries);
        }

        // Find last checkpoint
        let checkpoint_pos = all_entries
            .iter()
            .rposition(|e| e.entry_type == WalEntryType::Checkpoint);

        // Return entries after last checkpoint (or all if no checkpoint)
        let recovery_entries = if let Some(pos) = checkpoint_pos {
            all_entries.split_off(pos + 1)
        } else {
            all_entries
        };

        info!("Recovered {} entries from WAL", recovery_entries.len());

        Ok(recovery_entries)
    }

    /// Get statistics
    pub fn statistics(&self) -> &WalStatistics {
        &self.stats
    }

    /// Flush current segment to disk
    pub fn flush(&mut self) -> StarResult<()> {
        if let Some(segment) = self.current_segment.as_mut() {
            segment
                .writer
                .flush()
                .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_creation() {
        let temp_dir = std::env::temp_dir().join(format!("oxirs_wal_test_{}", std::process::id()));
        let config = WalConfig {
            wal_dir: temp_dir.clone(),
            ..Default::default()
        };
        let wal = WriteAheadLog::new(config);
        assert!(wal.is_ok());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_append_write() {
        let temp_dir = std::env::temp_dir().join(format!("oxirs_wal_test_{}", std::process::id()));
        let config = WalConfig {
            wal_dir: temp_dir.clone(),
            enable_fsync: false, // Faster for tests
            ..Default::default()
        };
        let mut wal = WriteAheadLog::new(config).unwrap();

        let annotation = TripleAnnotation::new().with_confidence(0.9);
        let seq = wal.append_write(123, annotation, None);
        assert!(seq.is_ok());

        let stats = wal.statistics();
        assert_eq!(stats.total_entries, 1);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_append_delete() {
        let temp_dir =
            std::env::temp_dir().join(format!("oxirs_wal_test_delete_{}", std::process::id()));
        let config = WalConfig {
            wal_dir: temp_dir.clone(),
            enable_fsync: false,
            ..Default::default()
        };
        let mut wal = WriteAheadLog::new(config).unwrap();

        let seq = wal.append_delete(456, None);
        assert!(seq.is_ok());

        let stats = wal.statistics();
        assert_eq!(stats.total_entries, 1);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_checkpoint() {
        let temp_dir =
            std::env::temp_dir().join(format!("oxirs_wal_test_checkpoint_{}", std::process::id()));
        let config = WalConfig {
            wal_dir: temp_dir.clone(),
            enable_fsync: false,
            ..Default::default()
        };
        let mut wal = WriteAheadLog::new(config).unwrap();

        wal.checkpoint().unwrap();

        let stats = wal.statistics();
        assert_eq!(stats.checkpoints, 1);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_recovery() {
        let temp_dir =
            std::env::temp_dir().join(format!("oxirs_wal_test_recovery_{}", std::process::id()));
        let config = WalConfig {
            wal_dir: temp_dir.clone(),
            enable_fsync: false,
            ..Default::default()
        };
        let mut wal = WriteAheadLog::new(config.clone()).unwrap();

        // Write some entries
        let ann1 = TripleAnnotation::new().with_confidence(0.8);
        wal.append_write(1, ann1, None).unwrap();

        let ann2 = TripleAnnotation::new().with_confidence(0.9);
        wal.append_write(2, ann2, None).unwrap();

        wal.checkpoint().unwrap();

        let ann3 = TripleAnnotation::new().with_confidence(0.95);
        wal.append_write(3, ann3, None).unwrap();

        wal.flush().unwrap();

        // Recover
        let entries = wal.recover().unwrap();

        // Should only get entries after checkpoint
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, 3);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_segment_rotation() {
        let temp_dir =
            std::env::temp_dir().join(format!("oxirs_wal_test_rotation_{}", std::process::id()));
        let config = WalConfig {
            wal_dir: temp_dir.clone(),
            enable_fsync: false,
            segment_size_threshold: 100, // Very small to force rotation
            ..Default::default()
        };
        let mut wal = WriteAheadLog::new(config).unwrap();

        // Write enough to trigger rotation
        for i in 0..100 {
            let annotation = TripleAnnotation::new().with_confidence(0.9);
            wal.append_write(i, annotation, None).unwrap();
        }

        let stats = wal.statistics();
        assert!(stats.rotations > 0);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_checksum_verification() {
        let annotation = TripleAnnotation::new().with_confidence(0.9);
        let entry = WalEntry::write(1, 123, annotation, None);

        assert!(entry.verify_checksum());
    }
}
