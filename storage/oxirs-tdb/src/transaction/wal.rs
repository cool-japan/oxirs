use crate::error::{Result, TdbError};
use crate::storage::page::PageId;
use oxicode::Decode;
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
    /// Logical data operation (operation-level redo).
    ///
    /// Written by the store's WAL-integrated write path (see
    /// [`crate::store`]). `payload` is an opaque, store-defined serialization of
    /// the committed operation (an interned triple/quad insert or delete); the
    /// store replays it on open by re-applying the operation to the
    /// reconstructed catalog. Unlike [`LogRecord::Update`] (full-page *physical*
    /// redo, used by the lower-level [`crate::recovery::RecoveryManager`]), this
    /// is a compact operation-level redo record, so each record stays small
    /// (~100 bytes). Note this bounds per-record *size* only: the *number* of
    /// records between checkpoints is bounded by periodic
    /// [`sync`](crate::store::TdbStore::sync) — either an explicit one or the
    /// automatic checkpoint driven by
    /// [`TdbConfig::wal_checkpoint_op_threshold`](crate::store::TdbConfig).
    /// Without any checkpoint the record count (and the in-memory log buffer)
    /// still grows unbounded.
    DataOp {
        /// Transaction ID this operation belongs to. Only replayed if a matching
        /// [`LogRecord::Commit`] for this `txn_id` is present (torn/uncommitted
        /// operations are never replayed).
        txn_id: TxnId,
        /// Opaque, store-encoded redo payload.
        payload: Vec<u8>,
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

/// Upper bound on the on-disk size of a single serialized WAL record body.
///
/// A record body is a serialized [`LogEntry`]; the largest legitimate variant is
/// a full-page [`LogRecord::Update`] carrying a before- and after-image (a few
/// KiB each) plus framing, so 64 MiB is generously above any real record while
/// still bounding the allocation performed for a length prefix read back from
/// disk. A prefix claiming more than this is treated as a torn/garbage tail
/// (see [`WriteAheadLog::recover`]) rather than triggering a multi-gigabyte
/// allocation.
const MAX_WAL_RECORD_LEN: usize = 64 * 1024 * 1024;

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
            .append(true)
            .open(&wal_path)
            .map_err(TdbError::Io)?;

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
        let mut next_lsn = self.next_lsn.write().expect("lock poisoned");
        let lsn = *next_lsn;
        *next_lsn = next_lsn.next();

        let entry = LogEntry { lsn, record };

        // Add to in-memory buffer
        let mut log_buffer = self.log_buffer.write().expect("lock poisoned");
        log_buffer.insert(lsn, entry.clone());

        // Write to disk (WAL protocol: log before data).
        //
        // On-disk record framing is: `[len: u32-le][crc32: u32-le][body: len bytes]`.
        // The CRC covers the serialized body so [`WriteAheadLog::recover`] can
        // distinguish a fully-written record from a torn or bit-flipped tail left
        // by a crash mid-append (append is not atomic; fsync happens in `flush`).
        let serialized = oxicode::serde::encode_to_vec(&entry, oxicode::config::standard())
            .map_err(|e| TdbError::Serialization(e.to_string()))?;

        let len = (serialized.len() as u32).to_le_bytes();
        let crc = crc32fast::hash(&serialized).to_le_bytes();

        let mut wal_file = self.wal_file.write().expect("lock poisoned");
        wal_file.write_all(&len).map_err(TdbError::Io)?;
        wal_file.write_all(&crc).map_err(TdbError::Io)?;
        wal_file.write_all(&serialized).map_err(TdbError::Io)?;

        Ok(lsn)
    }

    /// Flush WAL to disk (force sync)
    pub fn flush(&self) -> Result<()> {
        let mut wal_file = self.wal_file.write().expect("lock poisoned");
        wal_file.flush().map_err(TdbError::Io)?;
        wal_file.sync_all().map_err(TdbError::Io)?;

        let next_lsn = *self.next_lsn.read().expect("lock poisoned");
        let mut last_flushed = self.last_flushed_lsn.write().expect("lock poisoned");
        *last_flushed = Lsn::new(next_lsn.as_u64().saturating_sub(1));

        Ok(())
    }

    /// The next LSN that will be assigned by [`WriteAheadLog::append`].
    ///
    /// All records already written carry an LSN strictly less than this value,
    /// so it is the natural high-water mark to record as a checkpoint LSN in the
    /// superblock (see [`crate::store::TdbStore::sync`]).
    pub fn next_lsn(&self) -> Lsn {
        *self.next_lsn.read().expect("lock poisoned")
    }

    /// Get log entry by LSN
    pub fn get(&self, lsn: Lsn) -> Option<LogEntry> {
        let log_buffer = self.log_buffer.read().expect("lock poisoned");
        log_buffer.get(&lsn).cloned()
    }

    /// Truncate WAL up to LSN (after checkpoint)
    pub fn truncate(&self, up_to_lsn: Lsn) -> Result<()> {
        // Remove old entries from buffer
        let mut log_buffer = self.log_buffer.write().expect("lock poisoned");
        log_buffer.retain(|lsn, _| lsn.as_u64() > up_to_lsn.as_u64());

        // Truncate file (simplified: rewrite entire file)
        let remaining_entries: Vec<_> = log_buffer.values().cloned().collect();

        let mut wal_file = self.wal_file.write().expect("lock poisoned");
        wal_file.set_len(0).map_err(TdbError::Io)?;
        wal_file.seek(SeekFrom::Start(0)).map_err(TdbError::Io)?;

        for entry in remaining_entries {
            let serialized = oxicode::serde::encode_to_vec(&entry, oxicode::config::standard())
                .map_err(|e| TdbError::Serialization(e.to_string()))?;
            let len = (serialized.len() as u32).to_le_bytes();
            let crc = crc32fast::hash(&serialized).to_le_bytes();

            wal_file.write_all(&len).map_err(TdbError::Io)?;
            wal_file.write_all(&crc).map_err(TdbError::Io)?;
            wal_file.write_all(&serialized).map_err(TdbError::Io)?;
        }

        wal_file.flush().map_err(TdbError::Io)?;
        wal_file.sync_all().map_err(TdbError::Io)?;

        Ok(())
    }

    /// Recover from WAL (replay logs).
    ///
    /// A crash while [`WriteAheadLog::append`] was writing the last record is the
    /// normal case (append writes `len`, `crc`, then `body` and only `flush`
    /// fsyncs), so the physical tail of the file may be a *torn* record: a
    /// truncated body, a length prefix with no body, a garbage length, or a
    /// bit-flipped body. Recovery treats the log as ending at the last complete,
    /// CRC-verified record — it stops at the first record it cannot fully read
    /// and verify, and physically truncates the file to that point so the torn
    /// tail is dropped rather than propagating an `UnexpectedEof`/decode error
    /// out of `open()` (which would refuse to open the store after an ordinary
    /// crash). This upholds the crash-recovery contract that committed writes
    /// (records that made it to disk whole) survive a reopen.
    pub fn recover(&self) -> Result<Vec<LogEntry>> {
        let mut wal_file = self.wal_file.write().expect("lock poisoned");
        let file_len = wal_file.metadata().map_err(TdbError::Io)?.len();
        wal_file.seek(SeekFrom::Start(0)).map_err(TdbError::Io)?;

        let mut entries = Vec::new();
        let mut next_lsn = Lsn::ZERO;
        // Byte offset just past the last complete, verified record. Everything
        // after this is a torn tail to be discarded.
        let mut good_offset: u64 = 0;

        loop {
            // --- length prefix ---
            let mut len_buf = [0u8; 4];
            match wal_file.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(TdbError::Io(e)),
            }
            let len = u32::from_le_bytes(len_buf) as usize;

            // Full record occupies 4 (len) + 4 (crc) + len (body) bytes. Reject a
            // zero/implausible/garbage length or a record that would run past the
            // physical end of file: that means the tail is torn. Bounding `len`
            // here also prevents allocating an arbitrarily large buffer from a
            // corrupt prefix.
            let record_end = good_offset.saturating_add(8).saturating_add(len as u64);
            if len == 0 || len > MAX_WAL_RECORD_LEN || record_end > file_len {
                break;
            }

            // --- crc prefix ---
            let mut crc_buf = [0u8; 4];
            match wal_file.read_exact(&mut crc_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(TdbError::Io(e)),
            }
            let stored_crc = u32::from_le_bytes(crc_buf);

            // --- body ---
            let mut entry_buf = vec![0u8; len];
            match wal_file.read_exact(&mut entry_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(TdbError::Io(e)),
            }

            // A CRC mismatch means a torn/bit-flipped tail: stop here.
            if crc32fast::hash(&entry_buf) != stored_crc {
                break;
            }

            // A body that survives the CRC but still fails to decode is likewise
            // treated as end-of-log rather than a hard error.
            let entry: LogEntry =
                match oxicode::serde::decode_from_slice(&entry_buf, oxicode::config::standard()) {
                    Ok((entry, _)) => entry,
                    Err(_) => break,
                };

            good_offset = record_end;

            if entry.lsn.as_u64() >= next_lsn.as_u64() {
                next_lsn = entry.lsn.next();
            }

            entries.push(entry.clone());

            // Add to buffer
            let mut log_buffer = self.log_buffer.write().expect("lock poisoned");
            log_buffer.insert(entry.lsn, entry);
        }

        // Physically drop any torn tail so a subsequent append starts from a
        // clean, fully-valid log and the file does not accumulate garbage.
        if good_offset < file_len {
            wal_file.set_len(good_offset).map_err(TdbError::Io)?;
            wal_file.flush().map_err(TdbError::Io)?;
            wal_file.sync_all().map_err(TdbError::Io)?;
        }

        // Update next LSN
        let mut next_lsn_guard = self.next_lsn.write().expect("lock poisoned");
        *next_lsn_guard = next_lsn;

        Ok(entries)
    }

    /// Get all log entries (for testing)
    pub fn all_entries(&self) -> Vec<LogEntry> {
        let log_buffer = self.log_buffer.read().expect("lock poisoned");
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

        let flushed = *wal
            .last_flushed_lsn
            .read()
            .expect("lock should not be poisoned");
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
    fn regression_wal_torn_tail_truncated_body_recovers_prior_records() {
        // A crash mid-append leaves a length prefix (and maybe crc) but a short
        // body. recover() must recover every complete record before it and drop
        // the torn tail, rather than erroring out of open().
        let temp_dir = env::temp_dir().join("oxirs_wal_torn_body");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let wal_path = temp_dir.join("wal.log");

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

        // Simulate a torn tail: a plausible length prefix + crc, but a body far
        // shorter than the prefix claims.
        {
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            f.write_all(&50u32.to_le_bytes()).unwrap();
            f.write_all(&0u32.to_le_bytes()).unwrap();
            f.write_all(&[7u8; 10]).unwrap();
            f.flush().unwrap();
        }

        let wal = WriteAheadLog::open(&temp_dir).unwrap();
        let entries = wal.all_entries();
        assert_eq!(entries.len(), 2, "torn tail dropped, prior records kept");

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn regression_wal_garbage_length_prefix_is_not_allocated() {
        // A garbage/huge length prefix must be treated as end-of-log (torn tail),
        // never allocated (a naive `vec![0u8; len]` would try to grab 4 GiB).
        let temp_dir = env::temp_dir().join("oxirs_wal_garbage_len");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let wal_path = temp_dir.join("wal.log");

        {
            let wal = WriteAheadLog::new(&temp_dir).unwrap();
            wal.append(LogRecord::Begin {
                txn_id: TxnId::new(1),
            })
            .unwrap();
            wal.flush().unwrap();
        }

        {
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            f.write_all(&u32::MAX.to_le_bytes()).unwrap(); // absurd length
            f.write_all(&[0u8; 4]).unwrap();
            f.flush().unwrap();
        }

        let wal = WriteAheadLog::open(&temp_dir).unwrap();
        assert_eq!(wal.all_entries().len(), 1);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn regression_wal_crc_detects_bitflip_in_tail() {
        // Flipping a byte in the last record's body must be detected by the CRC
        // and that record dropped, leaving the earlier records intact.
        use std::io::Read as _;

        let temp_dir = env::temp_dir().join("oxirs_wal_bitflip");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let wal_path = temp_dir.join("wal.log");

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

        // Flip the final byte of the file (inside the last record's body).
        {
            let mut buf = Vec::new();
            std::fs::File::open(&wal_path)
                .unwrap()
                .read_to_end(&mut buf)
                .unwrap();
            let last = buf.len() - 1;
            buf[last] ^= 0xFF;
            std::fs::write(&wal_path, &buf).unwrap();
        }

        let wal = WriteAheadLog::open(&temp_dir).unwrap();
        assert_eq!(
            wal.all_entries().len(),
            1,
            "bit-flipped tail record dropped, prior record kept"
        );

        std::fs::remove_dir_all(&temp_dir).ok();
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
