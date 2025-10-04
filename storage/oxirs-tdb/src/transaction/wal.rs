use crate::error::{Result, TdbError};
use crate::storage::page::PageId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Transaction ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TxnId(u64);

impl TxnId {
    /// Create a new transaction ID
    pub const fn new(id: u64) -> Self {
        TxnId(id)
    }

    /// Get the transaction ID as u64
    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    /// Get the next transaction ID
    pub const fn next(&self) -> TxnId {
        TxnId(self.0 + 1)
    }
}

/// Log Sequence Number
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Lsn(u64);

impl Lsn {
    /// Create a new log sequence number
    pub const fn new(lsn: u64) -> Self {
        Lsn(lsn)
    }

    /// Get the LSN as u64
    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    /// Get the next LSN
    pub const fn next(&self) -> Lsn {
        Lsn(self.0 + 1)
    }

    /// Zero LSN constant
    pub const ZERO: Lsn = Lsn(0);
}

/// WAL record type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogRecord {
    /// Transaction begin
    Begin {
        /// Transaction ID
        txn_id: TxnId,
    },
    /// Transaction commit
    Commit {
        /// Transaction ID
        txn_id: TxnId,
    },
    /// Transaction abort
    Abort {
        /// Transaction ID
        txn_id: TxnId,
    },
    /// Page update (redo log)
    Update {
        /// Transaction ID
        txn_id: TxnId,
        /// Page ID being updated
        page_id: PageId,
        /// Page contents before update
        before_image: Vec<u8>,
        /// Page contents after update
        after_image: Vec<u8>,
    },
    /// Checkpoint
    Checkpoint {
        /// Active transaction IDs at checkpoint
        active_txns: Vec<TxnId>,
    },
}

/// WAL entry with LSN and record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Log sequence number
    pub lsn: Lsn,
    /// Log record
    pub record: LogRecord,
}

/// Write-Ahead Log (WAL) implementation
pub struct WriteAheadLog {
    /// Path to WAL file
    wal_path: PathBuf,
    /// WAL file handle
    wal_file: RwLock<File>,
    /// Next LSN to assign
    next_lsn: RwLock<Lsn>,
    /// In-memory log buffer (LSN -> LogEntry)
    log_buffer: RwLock<HashMap<Lsn, LogEntry>>,
    /// Last flushed LSN
    last_flushed_lsn: RwLock<Lsn>,
}

impl WriteAheadLog {
    /// Create a new WAL
    pub fn new<P: AsRef<Path>>(wal_dir: P) -> Result<Self> {
        let wal_path = wal_dir.as_ref().join("wal.log");

        let wal_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .append(true)
            .open(&wal_path)
            .map_err(|e| TdbError::Io(e))?;

        Ok(Self {
            wal_path,
            wal_file: RwLock::new(wal_file),
            next_lsn: RwLock::new(Lsn::ZERO),
            log_buffer: RwLock::new(HashMap::new()),
            last_flushed_lsn: RwLock::new(Lsn::ZERO),
        })
    }

    /// Open an existing WAL and replay logs
    pub fn open<P: AsRef<Path>>(wal_dir: P) -> Result<Self> {
        let wal = Self::new(wal_dir)?;
        wal.recover()?;
        Ok(wal)
    }

    /// Append a log record and return its LSN
    pub fn append(&self, record: LogRecord) -> Result<Lsn> {
        let mut next_lsn = self.next_lsn.write().unwrap();
        let lsn = *next_lsn;
        *next_lsn = next_lsn.next();

        let entry = LogEntry { lsn, record };

        // Add to in-memory buffer
        let mut log_buffer = self.log_buffer.write().unwrap();
        log_buffer.insert(lsn, entry.clone());

        // Write to disk (WAL protocol: log before data)
        let serialized = bincode::serialize(&entry)
            .map_err(|e| TdbError::Serialization(e.to_string()))?;

        let len = (serialized.len() as u32).to_le_bytes();

        let mut wal_file = self.wal_file.write().unwrap();
        wal_file
            .write_all(&len)
            .map_err(|e| TdbError::Io(e))?;
        wal_file
            .write_all(&serialized)
            .map_err(|e| TdbError::Io(e))?;

        Ok(lsn)
    }

    /// Flush WAL to disk (force sync)
    pub fn flush(&self) -> Result<()> {
        let mut wal_file = self.wal_file.write().unwrap();
        wal_file.flush().map_err(|e| TdbError::Io(e))?;
        wal_file.sync_all().map_err(|e| TdbError::Io(e))?;

        let next_lsn = *self.next_lsn.read().unwrap();
        let mut last_flushed = self.last_flushed_lsn.write().unwrap();
        *last_flushed = Lsn::new(next_lsn.as_u64().saturating_sub(1));

        Ok(())
    }

    /// Get log entry by LSN
    pub fn get(&self, lsn: Lsn) -> Option<LogEntry> {
        let log_buffer = self.log_buffer.read().unwrap();
        log_buffer.get(&lsn).cloned()
    }

    /// Truncate WAL up to LSN (after checkpoint)
    pub fn truncate(&self, up_to_lsn: Lsn) -> Result<()> {
        // Remove old entries from buffer
        let mut log_buffer = self.log_buffer.write().unwrap();
        log_buffer.retain(|lsn, _| lsn.as_u64() > up_to_lsn.as_u64());

        // Truncate file (simplified: rewrite entire file)
        let remaining_entries: Vec<_> = log_buffer.values().cloned().collect();

        let mut wal_file = self.wal_file.write().unwrap();
        wal_file
            .set_len(0)
            .map_err(|e| TdbError::Io(e))?;
        wal_file
            .seek(SeekFrom::Start(0))
            .map_err(|e| TdbError::Io(e))?;

        for entry in remaining_entries {
            let serialized = bincode::serialize(&entry)
                .map_err(|e| TdbError::Serialization(e.to_string()))?;
            let len = (serialized.len() as u32).to_le_bytes();

            wal_file.write_all(&len).map_err(|e| TdbError::Io(e))?;
            wal_file
                .write_all(&serialized)
                .map_err(|e| TdbError::Io(e))?;
        }

        wal_file.flush().map_err(|e| TdbError::Io(e))?;
        wal_file.sync_all().map_err(|e| TdbError::Io(e))?;

        Ok(())
    }

    /// Recover from WAL (replay logs)
    pub fn recover(&self) -> Result<Vec<LogEntry>> {
        let mut wal_file = self.wal_file.write().unwrap();
        wal_file
            .seek(SeekFrom::Start(0))
            .map_err(|e| TdbError::Io(e))?;

        let mut entries = Vec::new();
        let mut next_lsn = Lsn::ZERO;

        loop {
            // Read length prefix
            let mut len_buf = [0u8; 4];
            match wal_file.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break; // End of file
                }
                Err(e) => return Err(TdbError::Io(e)),
            }

            let len = u32::from_le_bytes(len_buf) as usize;

            // Read entry
            let mut entry_buf = vec![0u8; len];
            wal_file
                .read_exact(&mut entry_buf)
                .map_err(|e| TdbError::Io(e))?;

            let entry: LogEntry = bincode::deserialize(&entry_buf)
                .map_err(|e| TdbError::Serialization(e.to_string()))?;

            if entry.lsn.as_u64() >= next_lsn.as_u64() {
                next_lsn = entry.lsn.next();
            }

            entries.push(entry.clone());

            // Add to buffer
            let mut log_buffer = self.log_buffer.write().unwrap();
            log_buffer.insert(entry.lsn, entry);
        }

        // Update next LSN
        let mut next_lsn_guard = self.next_lsn.write().unwrap();
        *next_lsn_guard = next_lsn;

        Ok(entries)
    }

    /// Get all log entries (for testing)
    pub fn all_entries(&self) -> Vec<LogEntry> {
        let log_buffer = self.log_buffer.read().unwrap();
        let mut entries: Vec<_> = log_buffer.values().cloned().collect();
        entries.sort_by_key(|e| e.lsn);
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_txn_id() {
        let txn1 = TxnId::new(1);
        let txn2 = txn1.next();
        assert_eq!(txn2.as_u64(), 2);
    }

    #[test]
    fn test_lsn() {
        let lsn1 = Lsn::ZERO;
        let lsn2 = lsn1.next();
        assert_eq!(lsn2.as_u64(), 1);
    }

    #[test]
    fn test_wal_append() {
        let temp_dir = env::temp_dir().join("oxirs_wal_append");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = WriteAheadLog::new(&temp_dir).unwrap();

        let lsn1 = wal
            .append(LogRecord::Begin {
                txn_id: TxnId::new(1),
            })
            .unwrap();
        let lsn2 = wal
            .append(LogRecord::Commit {
                txn_id: TxnId::new(1),
            })
            .unwrap();

        assert_eq!(lsn1.as_u64(), 0);
        assert_eq!(lsn2.as_u64(), 1);

        let entry1 = wal.get(lsn1).unwrap();
        let entry2 = wal.get(lsn2).unwrap();

        assert!(matches!(entry1.record, LogRecord::Begin { .. }));
        assert!(matches!(entry2.record, LogRecord::Commit { .. }));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_wal_flush() {
        let temp_dir = env::temp_dir().join("oxirs_wal_flush");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = WriteAheadLog::new(&temp_dir).unwrap();

        wal.append(LogRecord::Begin {
            txn_id: TxnId::new(1),
        })
        .unwrap();
        wal.flush().unwrap();

        let flushed = *wal.last_flushed_lsn.read().unwrap();
        assert_eq!(flushed.as_u64(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_wal_recover() {
        let temp_dir = env::temp_dir().join("oxirs_wal_recover");
        std::fs::create_dir_all(&temp_dir).unwrap();

        {
            let wal = WriteAheadLog::new(&temp_dir).unwrap();
            wal.append(LogRecord::Begin {
                txn_id: TxnId::new(1),
            })
            .unwrap();
            wal.append(LogRecord::Commit {
                txn_id: TxnId::new(1),
            })
            .unwrap();
            wal.flush().unwrap();
        }

        let wal = WriteAheadLog::open(&temp_dir).unwrap();
        let entries = wal.all_entries();
        assert_eq!(entries.len(), 2);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_wal_truncate() {
        let temp_dir = env::temp_dir().join("oxirs_wal_truncate");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = WriteAheadLog::new(&temp_dir).unwrap();

        let lsn0 = wal
            .append(LogRecord::Begin {
                txn_id: TxnId::new(1),
            })
            .unwrap();
        let _lsn1 = wal
            .append(LogRecord::Commit {
                txn_id: TxnId::new(1),
            })
            .unwrap();
        let lsn2 = wal
            .append(LogRecord::Begin {
                txn_id: TxnId::new(2),
            })
            .unwrap();

        wal.truncate(lsn0).unwrap();

        let entries = wal.all_entries();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().any(|e| e.lsn == lsn2));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_log_record_update() {
        let record = LogRecord::Update {
            txn_id: TxnId::new(1),
            page_id: 10,
            before_image: vec![1, 2, 3],
            after_image: vec![4, 5, 6],
        };

        match record {
            LogRecord::Update {
                txn_id,
                page_id,
                before_image,
                after_image,
            } => {
                assert_eq!(txn_id.as_u64(), 1);
                assert_eq!(page_id, 10);
                assert_eq!(before_image, vec![1, 2, 3]);
                assert_eq!(after_image, vec![4, 5, 6]);
            }
            _ => panic!("Wrong record type"),
        }
    }

    #[test]
    fn test_log_record_checkpoint() {
        let record = LogRecord::Checkpoint {
            active_txns: vec![TxnId::new(1), TxnId::new(2)],
        };

        match record {
            LogRecord::Checkpoint { active_txns } => {
                assert_eq!(active_txns.len(), 2);
            }
            _ => panic!("Wrong record type"),
        }
    }
}
