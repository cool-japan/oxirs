//! Write-Ahead Log for crash recovery
//!
//! Provides durability guarantees by logging data points before they
//! are committed to compressed storage.

use crate::error::{TsdbError, TsdbResult};
use crate::series::DataPoint;
use chrono::{DateTime, Utc};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Write-Ahead Log entry
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Series identifier
    pub series_id: u64,
    /// Data point timestamp
    pub timestamp: DateTime<Utc>,
    /// Value
    pub value: f64,
}

impl WalEntry {
    /// Create new WAL entry
    pub fn new(series_id: u64, point: DataPoint) -> Self {
        Self {
            series_id,
            timestamp: point.timestamp,
            value: point.value,
        }
    }

    /// Serialize to binary format
    ///
    /// Format: series_id(8) + timestamp_ms(8) + value(8) = 24 bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(24);

        // Series ID (8 bytes)
        bytes.extend_from_slice(&self.series_id.to_le_bytes());

        // Timestamp as milliseconds since epoch (8 bytes)
        let ts_ms = self.timestamp.timestamp_millis();
        bytes.extend_from_slice(&ts_ms.to_le_bytes());

        // Value (8 bytes)
        bytes.extend_from_slice(&self.value.to_le_bytes());

        bytes
    }

    /// Deserialize from binary format
    pub fn from_bytes(bytes: &[u8]) -> TsdbResult<Self> {
        if bytes.len() != 24 {
            return Err(TsdbError::Wal(format!(
                "WAL entry must be 24 bytes, got {}",
                bytes.len()
            )));
        }

        let series_id = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let ts_ms = i64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let value = f64::from_le_bytes(bytes[16..24].try_into().unwrap());

        let timestamp = DateTime::from_timestamp_millis(ts_ms)
            .ok_or_else(|| TsdbError::Wal(format!("Invalid timestamp: {}", ts_ms)))?;

        Ok(Self {
            series_id,
            timestamp,
            value,
        })
    }
}

/// Write-Ahead Log for durability
pub struct WriteAheadLog {
    /// Path to WAL file
    path: PathBuf,
    /// Writer handle
    writer: BufWriter<File>,
    /// Whether to fsync after each write
    sync_on_write: bool,
    /// Number of entries in current WAL
    entry_count: u64,
}

impl WriteAheadLog {
    /// Create or open a Write-Ahead Log
    ///
    /// # Arguments
    ///
    /// * `path` - Path to WAL file
    /// * `sync_on_write` - If true, fsync after each append
    pub fn new<P: AsRef<Path>>(path: P, sync_on_write: bool) -> TsdbResult<Self> {
        let path = path.as_ref().to_path_buf();

        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        // Count existing entries
        let entry_count = if path.exists() {
            let reader_file = File::open(&path)?;
            let metadata = reader_file.metadata()?;
            metadata.len() / 24 // 24 bytes per entry
        } else {
            0
        };

        Ok(Self {
            path,
            writer: BufWriter::new(file),
            sync_on_write,
            entry_count,
        })
    }

    /// Append a data point to the WAL
    pub fn append(&mut self, series_id: u64, point: DataPoint) -> TsdbResult<()> {
        let entry = WalEntry::new(series_id, point);
        let bytes = entry.to_bytes();

        self.writer.write_all(&bytes)?;

        if self.sync_on_write {
            self.writer.flush()?;
            self.writer.get_ref().sync_all()?;
        }

        self.entry_count += 1;

        Ok(())
    }

    /// Append multiple data points in a batch
    pub fn append_batch(&mut self, entries: &[(u64, DataPoint)]) -> TsdbResult<()> {
        for (series_id, point) in entries {
            let entry = WalEntry::new(*series_id, *point);
            let bytes = entry.to_bytes();
            self.writer.write_all(&bytes)?;
        }

        self.writer.flush()?;

        if self.sync_on_write {
            self.writer.get_ref().sync_all()?;
        }

        self.entry_count += entries.len() as u64;

        Ok(())
    }

    /// Replay all entries in the WAL
    ///
    /// Returns vector of (series_id, DataPoint) tuples
    pub fn replay(&self) -> TsdbResult<Vec<(u64, DataPoint)>> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        let mut buffer = vec![0u8; 24];

        loop {
            match std::io::Read::read_exact(&mut reader, &mut buffer) {
                Ok(_) => {
                    let entry = WalEntry::from_bytes(&buffer)?;
                    entries.push((
                        entry.series_id,
                        DataPoint {
                            timestamp: entry.timestamp,
                            value: entry.value,
                        },
                    ));
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break; // End of file
                }
                Err(e) => return Err(e.into()),
            }
        }

        Ok(entries)
    }

    /// Clear the WAL after successful compaction
    pub fn clear(&mut self) -> TsdbResult<()> {
        // Close current writer
        self.writer.flush()?;

        // Truncate file
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.path)?;

        self.writer = BufWriter::new(file);
        self.entry_count = 0;

        Ok(())
    }

    /// Get number of entries in WAL
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Get WAL file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flush buffers to disk
    pub fn flush(&mut self) -> TsdbResult<()> {
        self.writer.flush()?;

        if self.sync_on_write {
            self.writer.get_ref().sync_all()?;
        }

        Ok(())
    }
}

impl Drop for WriteAheadLog {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use std::env;

    #[test]
    fn test_wal_entry_serialization() {
        let entry = WalEntry {
            series_id: 42,
            timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap(),
            value: 22.5,
        };

        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), 24);

        let recovered = WalEntry::from_bytes(&bytes).unwrap();
        assert_eq!(recovered.series_id, 42);
        assert_eq!(recovered.timestamp, entry.timestamp);
        assert_eq!(recovered.value, 22.5);
    }

    #[test]
    fn test_wal_append_and_replay() {
        let temp_dir = env::temp_dir();
        let wal_path = temp_dir.join("test_wal_append.log");

        // Clean up any existing file
        let _ = std::fs::remove_file(&wal_path);

        {
            let mut wal = WriteAheadLog::new(&wal_path, false).unwrap();

            let timestamp = Utc::now();
            let point1 = DataPoint {
                timestamp,
                value: 10.0,
            };
            let point2 = DataPoint {
                timestamp: timestamp + chrono::Duration::seconds(1),
                value: 20.0,
            };

            wal.append(1, point1).unwrap();
            wal.append(2, point2).unwrap();

            assert_eq!(wal.entry_count(), 2);
        }

        // Replay from disk
        let wal = WriteAheadLog::new(&wal_path, false).unwrap();
        let entries = wal.replay().unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, 1);
        assert_eq!(entries[0].1.value, 10.0);
        assert_eq!(entries[1].0, 2);
        assert_eq!(entries[1].1.value, 20.0);

        // Cleanup
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_wal_clear() {
        let temp_dir = env::temp_dir();
        let wal_path = temp_dir.join("test_wal_clear.log");

        // Clean up any existing file
        let _ = std::fs::remove_file(&wal_path);

        {
            let mut wal = WriteAheadLog::new(&wal_path, false).unwrap();

            let point = DataPoint {
                timestamp: Utc::now(),
                value: 42.0,
            };

            wal.append(1, point).unwrap();
            assert_eq!(wal.entry_count(), 1);

            wal.clear().unwrap();
            assert_eq!(wal.entry_count(), 0);
        }

        // Verify file is empty
        let wal = WriteAheadLog::new(&wal_path, false).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 0);

        // Cleanup
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_wal_batch_append() {
        let temp_dir = env::temp_dir();
        let wal_path = temp_dir.join("test_wal_batch.log");

        // Clean up any existing file
        let _ = std::fs::remove_file(&wal_path);

        {
            let mut wal = WriteAheadLog::new(&wal_path, false).unwrap();

            let base_time = Utc::now();
            let mut batch = Vec::new();

            for i in 0..100 {
                let point = DataPoint {
                    timestamp: base_time + chrono::Duration::seconds(i),
                    value: i as f64,
                };
                batch.push((i as u64, point));
            }

            wal.append_batch(&batch).unwrap();
            assert_eq!(wal.entry_count(), 100);
        }

        // Replay and verify
        let wal = WriteAheadLog::new(&wal_path, false).unwrap();
        let entries = wal.replay().unwrap();

        assert_eq!(entries.len(), 100);
        for (i, (series_id, point)) in entries.iter().enumerate() {
            assert_eq!(*series_id, i as u64);
            assert_eq!(point.value, i as f64);
        }

        // Cleanup
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_wal_fsync() {
        let temp_dir = env::temp_dir();
        let wal_path = temp_dir.join("test_wal_fsync.log");

        // Clean up any existing file
        let _ = std::fs::remove_file(&wal_path);

        {
            let mut wal = WriteAheadLog::new(&wal_path, true).unwrap(); // sync_on_write = true

            let point = DataPoint {
                timestamp: Utc::now(),
                value: 123.456,
            };

            wal.append(1, point).unwrap();
        }

        // Verify data persisted
        let wal = WriteAheadLog::new(&wal_path, false).unwrap();
        let entries = wal.replay().unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, 1);
        assert_eq!(entries[0].1.value, 123.456);

        // Cleanup
        let _ = std::fs::remove_file(&wal_path);
    }
}
