//! Write-ahead log of PersistentStorage.
//!
//! Every save of the Raft or application state is journaled under one lock:
//!
//! 1. the new state is appended to `wal.log` as an entry,
//! 2. the state file is replaced atomically,
//! 3. a `Commit` naming the entry's sequence number is appended.
//!
//! A crash between steps 1 and 3 leaves an entry without its commit, and
//! startup recovery writes that state to its file ("redo"). A save whose
//! state file cannot be written removes its entry again, so recovery never
//! redoes a write that reported an error. Once everything in the log is
//! committed and the log has grown past [`RESET_THRESHOLD`], it is emptied.
//!
//! Records are `u64 LE length | oxicode WalEntry`. A record that is cut short
//! or damaged at the end of the log, as a crash during an append leaves it,
//! is moved to `wal.log.torn-<unix ms>` and cut off. Damage that is followed
//! by further records cannot come from a crash; it is an error when crash
//! recovery would replay the log. With crash recovery disabled the log is
//! never replayed, so everything from the damage on is set aside instead.
//! Releases up to 0.4.1 kept appending after a torn record in that mode.

use super::config::StorageConfig;
use super::persistent_files::{self, APP_STATE_FILE, RAFT_STATE_FILE};
use super::types::{RaftState, SnapshotMetadata, WalEntry, WalOperation};
use crate::raft::RdfApp;

use anyhow::{anyhow, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Name of the log inside the data directory.
pub(super) const WAL_FILE: &str = "wal.log";
/// Size past which a fully committed log is emptied.
const RESET_THRESHOLD: u64 = 16 * 1024 * 1024;
/// Size from which [`WalWriter::rotate`] archives the log.
const ROTATE_THRESHOLD: u64 = 100 * 1024 * 1024;
/// Bytes of the length prefix in front of each record.
const LENGTH_PREFIX: u64 = 8;

/// Borrowed form of [`WalOperation`]. It serializes exactly like the owned
/// enum, so a save is journaled without copying the state; the variants must
/// therefore stay in the same order.
#[derive(Serialize)]
pub(super) enum WalOperationRef<'a> {
    WriteRaftState(&'a RaftState),
    WriteAppState(&'a RdfApp),
    // Never journaled, but kept so that the variant indices line up.
    #[allow(dead_code)]
    CreateSnapshot(&'a SnapshotMetadata),
    #[allow(dead_code)]
    TruncateLog(u64),
    Commit(u64),
}

/// Borrowed form of [`WalEntry`], serialized identically.
#[derive(Serialize)]
struct WalEntryRef<'a> {
    sequence: u64,
    timestamp: u64,
    operation: &'a WalOperationRef<'a>,
    checksum: &'a str,
}

/// SHA-256 over the encoded operation, the sequence number and the timestamp.
fn entry_checksum(op_bytes: &[u8], sequence: u64, timestamp: u64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(op_bytes);
    hasher.update(sequence.to_le_bytes());
    hasher.update(timestamp.to_le_bytes());
    hex::encode(hasher.finalize())
}

/// Whether the checksum of `entry` matches its contents.
pub(super) fn verify_wal_entry_checksum(entry: &WalEntry) -> Result<bool> {
    let op_bytes = oxicode::serde::encode_to_vec(&entry.operation, oxicode::config::standard())?;
    Ok(entry_checksum(&op_bytes, entry.sequence, entry.timestamp) == entry.checksum)
}

/// One record for `operation`: its length prefix and the encoded entry.
fn encode_record(sequence: u64, operation: &WalOperationRef<'_>) -> Result<Vec<u8>> {
    let timestamp = persistent_files::unix_secs();
    let op_bytes = oxicode::serde::encode_to_vec(operation, oxicode::config::standard())?;
    let checksum = entry_checksum(&op_bytes, sequence, timestamp);
    let entry = WalEntryRef {
        sequence,
        timestamp,
        operation,
        checksum: &checksum,
    };
    let body = oxicode::serde::encode_to_vec(&entry, oxicode::config::standard())?;
    frame(&body)
}

/// `body` behind its length prefix.
fn frame(body: &[u8]) -> Result<Vec<u8>> {
    let mut record = Vec::with_capacity(body.len() + 8);
    record.extend_from_slice(&u64::try_from(body.len())?.to_le_bytes());
    record.extend_from_slice(body);
    Ok(record)
}

/// What a scan of the log found.
#[derive(Debug, Default)]
pub(super) struct WalScan {
    /// Length of the leading part of the log that holds intact records.
    pub valid_len: u64,
    /// Number of intact records.
    pub records: usize,
    /// Highest sequence number seen.
    pub last_sequence: u64,
    /// The first damaged record, where the scan stopped.
    pub damage: Option<WalDamage>,
}

/// A damaged record found by [`scan_wal`]. It starts at
/// [`WalScan::valid_len`].
#[derive(Debug)]
pub(super) struct WalDamage {
    /// What is wrong with the record.
    pub reason: String,
    /// Bytes that follow the damaged record. Zero means the damage reaches
    /// the end of the log, as a crash during an append leaves it.
    pub trailing: u64,
}

impl WalScan {
    /// The damage as an error, for callers that cannot work around it.
    fn damage_error(&self, path: &Path) -> Option<anyhow::Error> {
        self.damage.as_ref().map(|damage| {
            if damage.trailing == 0 {
                anyhow!(
                    "WAL {} is damaged at its end: {}",
                    path.display(),
                    damage.reason
                )
            } else {
                anyhow!(
                    "WAL {} is damaged: {}, and {} more byte(s) follow it. This is not what a \
                     crash leaves behind, so the log is not used; move it aside to start \
                     without it",
                    path.display(),
                    damage.reason,
                    damage.trailing
                )
            }
        })
    }
}

/// Read the log at `path` record by record and hand every intact entry to
/// `visit`, stopping at the first damaged record. A missing log scans as
/// empty. A record may be as large as the file holds, so whatever a save
/// wrote can be read back.
///
/// # Errors
///
/// Fails on I/O errors; damage is reported in [`WalScan::damage`].
pub(super) fn scan_wal(path: &Path, mut visit: impl FnMut(WalEntry)) -> Result<WalScan> {
    let mut scan = WalScan::default();
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(scan),
        Err(e) => return Err(e.into()),
    };
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    while scan.valid_len < file_len {
        let offset = scan.valid_len;
        let remaining = file_len - offset;
        if remaining < LENGTH_PREFIX {
            scan.damage = Some(WalDamage {
                reason: format!(
                    "{remaining} trailing byte(s) at byte {offset} are too few for a record"
                ),
                trailing: 0,
            });
            break;
        }
        let mut prefix = [0u8; 8];
        reader.read_exact(&mut prefix)?;
        let len = u64::from_le_bytes(prefix);
        let available = remaining - LENGTH_PREFIX;
        if len > available {
            scan.damage = Some(WalDamage {
                reason: format!(
                    "the record at byte {offset} needs {len} bytes but only {available} follow"
                ),
                trailing: 0,
            });
            break;
        }
        let end = offset + LENGTH_PREFIX + len;

        // `len` is at most what the file holds, which bounds the allocation.
        let mut body = vec![0u8; usize::try_from(len)?];
        reader.read_exact(&mut body)?;
        match decode_entry(&body) {
            Ok(entry) => {
                scan.last_sequence = scan.last_sequence.max(entry.sequence);
                scan.records += 1;
                scan.valid_len = end;
                visit(entry);
            }
            Err(reason) => {
                scan.damage = Some(WalDamage {
                    reason: format!("the record at byte {offset} {reason}"),
                    trailing: file_len - end,
                });
                break;
            }
        }
    }
    Ok(scan)
}

/// Decode and verify one record body.
fn decode_entry(body: &[u8]) -> std::result::Result<WalEntry, String> {
    let (entry, used): (WalEntry, usize) =
        oxicode::serde::decode_from_slice(body, oxicode::config::standard())
            .map_err(|e| format!("does not decode ({e})"))?;
    if used != body.len() {
        return Err(format!("has {} byte(s) after its entry", body.len() - used));
    }
    match verify_wal_entry_checksum(&entry) {
        Ok(true) => Ok(entry),
        Ok(false) => Err("fails its checksum".to_string()),
        Err(e) => Err(format!("cannot be checked ({e})")),
    }
}

/// Copy everything in `path` after `valid_len` to a new
/// `wal.log.torn-<unix ms>` file, then cut it off.
fn set_aside_tail(path: &Path, valid_len: u64) -> Result<PathBuf> {
    let mut source = File::open(path)?;
    source.seek(SeekFrom::Start(valid_len))?;
    let (aside, mut target) = create_aside_file(path)?;
    std::io::copy(&mut source, &mut target)?;
    target.sync_all()?;

    let log = OpenOptions::new().write(true).open(path)?;
    log.set_len(valid_len)?;
    log.sync_all()?;
    if let Some(dir) = path.parent() {
        persistent_files::sync_dir(dir)?;
    }
    Ok(aside)
}

/// Create a file next to `path` that does not exist yet.
fn create_aside_file(path: &Path) -> Result<(PathBuf, File)> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis());
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(WAL_FILE);
    let base = format!("{name}.torn-{millis}");
    for attempt in 0..100u32 {
        let candidate = if attempt == 0 {
            path.with_file_name(&base)
        } else {
            path.with_file_name(format!("{base}-{attempt}"))
        };
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(anyhow!(
        "could not create a file to set the torn tail of {} aside",
        path.display()
    ))
}

/// Outcome of [`recover_wal`].
#[derive(Debug)]
pub(super) struct WalRecovery {
    /// Sequence number for the next record.
    pub next_sequence: u64,
    /// Whether a journaled write was redone.
    pub redone: bool,
}

/// Bring the log and the state files up to date before the store opens.
///
/// A torn tail is set aside in any case. With `redo`, the newest Raft and
/// application state entries that were never committed are written to their
/// state files; everything in the log is then reflected on disk, and the log
/// is emptied. Without `redo` the log is never replayed, so damage in its
/// middle is set aside together with everything after it.
///
/// # Errors
///
/// Fails when the log is damaged in the middle and `redo` is set, or on I/O
/// errors.
pub(super) fn recover_wal(
    data_dir: &Path,
    config: &StorageConfig,
    redo: bool,
) -> Result<WalRecovery> {
    let path = data_dir.join(WAL_FILE);

    // The newest entry of each kind, and whether its commit was seen.
    let mut raft: Option<(u64, RaftState, bool)> = None;
    let mut app: Option<(u64, RdfApp, bool)> = None;
    let scan = scan_wal(&path, |entry| match entry.operation {
        WalOperation::WriteRaftState(state) => raft = Some((entry.sequence, state, false)),
        WalOperation::WriteAppState(state) => app = Some((entry.sequence, state, false)),
        WalOperation::Commit(sequence) => {
            for (pending, _, committed) in raft.iter_mut() {
                *committed |= *pending == sequence;
            }
            for (pending, _, committed) in app.iter_mut() {
                *committed |= *pending == sequence;
            }
        }
        WalOperation::CreateSnapshot(_) | WalOperation::TruncateLog(_) => {}
    })?;

    if let Some(damage) = &scan.damage {
        if damage.trailing > 0 && redo {
            // Replaying a log with a hole could redo a stale state.
            if let Some(error) = scan.damage_error(&path) {
                return Err(error);
            }
        }
        let aside = set_aside_tail(&path, scan.valid_len)?;
        tracing::warn!(
            "WAL {}: {}; moved everything from byte {} on to {}",
            path.display(),
            damage.reason,
            scan.valid_len,
            aside.display()
        );
    }

    let mut redone = false;
    if redo {
        if let Some((sequence, state, false)) = &raft {
            persistent_files::write_checksummed(data_dir, RAFT_STATE_FILE, state, config)?;
            tracing::info!("Redid the journaled Raft state write {sequence}");
            redone = true;
        }
        if let Some((sequence, state, false)) = &app {
            persistent_files::write_checksummed(data_dir, APP_STATE_FILE, state, config)?;
            tracing::info!("Redid the journaled application state write {sequence}");
            redone = true;
        }
        if scan.valid_len > 0 {
            let log = OpenOptions::new().write(true).open(&path)?;
            log.set_len(0)?;
            log.sync_all()?;
        }
    }

    Ok(WalRecovery {
        next_sequence: scan.last_sequence + 1,
        redone,
    })
}

/// Appends records to the log. It is held behind one lock for a whole save.
#[derive(Debug)]
pub(super) struct WalWriter {
    path: PathBuf,
    file: File,
    len: u64,
    next_sequence: u64,
    sync: bool,
    /// Set when a failed write could not be undone; the log then ends in a
    /// partial record, which recovery sets aside, and no more is appended.
    broken: Option<String>,
}

impl WalWriter {
    /// Open the log at `path` for appending.
    pub(super) fn open(path: PathBuf, next_sequence: u64, sync: bool) -> Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            path,
            file,
            len,
            next_sequence,
            sync,
            broken: None,
        })
    }

    /// Journal one save: append `operation`, run `apply` (which writes the
    /// state file), then append the commit. When `apply` fails, the entry is
    /// removed again and the error returned.
    pub(super) fn journal(
        &mut self,
        operation: WalOperationRef<'_>,
        apply: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        self.ensure_usable()?;
        let start = self.len;
        let sequence = self.append(&operation)?;
        if let Err(error) = apply() {
            if let Err(undo) = self.truncate_to(start) {
                self.broken = Some(format!("removing record {sequence} failed: {undo}"));
                return Err(error.context(format!(
                    "the WAL record {sequence} of the failed save could not be removed: {undo}"
                )));
            }
            return Err(error);
        }
        // The state file is written, so the save stands even if the commit
        // cannot be recorded: redoing the entry later rewrites the same state.
        if let Err(e) = self.append(&WalOperationRef::Commit(sequence)) {
            tracing::warn!("could not record the commit of WAL record {sequence}: {e}");
            return Ok(());
        }
        if self.len > RESET_THRESHOLD {
            // Every entry in the log is committed at this point.
            self.truncate_to(0)?;
        }
        Ok(())
    }

    /// Rewrite the log, keeping only entries without a commit. Returns the
    /// number of records removed and kept.
    pub(super) fn compact(&mut self) -> Result<(usize, usize)> {
        self.ensure_usable()?;
        let mut pending: Vec<WalEntry> = Vec::new();
        let scan = scan_wal(&self.path, |entry| match entry.operation {
            WalOperation::Commit(sequence) => pending.retain(|p| p.sequence != sequence),
            _ => pending.push(entry),
        })?;
        if let Some(error) = scan.damage_error(&self.path) {
            return Err(error);
        }

        let mut bytes = Vec::new();
        for entry in &pending {
            let body = oxicode::serde::encode_to_vec(entry, oxicode::config::standard())?;
            bytes.extend_from_slice(&frame(&body)?);
        }
        let (dir, name) = self.location()?;
        persistent_files::write_atomic(dir, name, &bytes, self.sync)?;
        // The handle still points at the file that was replaced.
        *self = Self::open(self.path.clone(), self.next_sequence, self.sync)?;
        Ok((scan.records - pending.len(), pending.len()))
    }

    /// Move a log of [`ROTATE_THRESHOLD`] bytes or more to
    /// `wal-<unix secs>.log` and start an empty one.
    pub(super) fn rotate(&mut self) -> Result<Option<PathBuf>> {
        self.ensure_usable()?;
        if self.len < ROTATE_THRESHOLD {
            return Ok(None);
        }
        let archive = self
            .path
            .with_file_name(format!("wal-{}.log", persistent_files::unix_secs()));
        std::fs::rename(&self.path, &archive)?;
        let (dir, _) = self.location()?;
        persistent_files::sync_dir(dir)?;
        *self = Self::open(self.path.clone(), self.next_sequence, self.sync)?;
        Ok(Some(archive))
    }

    /// Count the intact records; any damage is an error.
    pub(super) fn verify(&self) -> Result<usize> {
        let scan = scan_wal(&self.path, |_| {})?;
        match scan.damage_error(&self.path) {
            Some(error) => Err(error),
            None => Ok(scan.records),
        }
    }

    fn append(&mut self, operation: &WalOperationRef<'_>) -> Result<u64> {
        let sequence = self.next_sequence;
        let record = encode_record(sequence, operation)?;
        let sync = self.sync;
        let written = self.file.write_all(&record).and_then(|()| {
            if sync {
                self.file.sync_data()
            } else {
                Ok(())
            }
        });
        if let Err(e) = written {
            // Never leave part of a record behind.
            if let Err(undo) = self.file.set_len(self.len) {
                self.broken = Some(format!("cutting off a partial record failed: {undo}"));
            }
            return Err(
                anyhow::Error::from(e).context(format!("appending to WAL {}", self.path.display()))
            );
        }
        self.len += u64::try_from(record.len())?;
        self.next_sequence += 1;
        Ok(sequence)
    }

    fn truncate_to(&mut self, len: u64) -> Result<()> {
        self.file.set_len(len)?;
        if self.sync {
            self.file.sync_data()?;
        }
        self.len = len;
        Ok(())
    }

    fn ensure_usable(&self) -> Result<()> {
        match &self.broken {
            Some(reason) => Err(anyhow!(
                "WAL {} cannot be written after a failed write ({reason}); restart to recover it",
                self.path.display()
            )),
            None => Ok(()),
        }
    }

    fn location(&self) -> Result<(&Path, &str)> {
        let dir = self
            .path
            .parent()
            .ok_or_else(|| anyhow!("WAL {} has no directory", self.path.display()))?;
        let name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow!("WAL {} has no file name", self.path.display()))?;
        Ok((dir, name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::LogEntry;
    use crate::raft::RdfCommand;

    /// The borrowed operations must encode exactly like the owned ones, or
    /// recovery could not read what a save journaled.
    #[test]
    fn test_borrowed_operations_encode_like_owned_ones() {
        let state = RaftState {
            current_term: 3,
            voted_for: Some(2),
            log: vec![LogEntry::new(
                1,
                3,
                RdfCommand::Insert {
                    subject: "s".into(),
                    predicate: "p".into(),
                    object: "o".into(),
                },
            )],
            commit_index: 1,
            last_applied: 1,
        };
        let mut app = RdfApp::default();
        app.apply_command(&RdfCommand::Insert {
            subject: "a".into(),
            predicate: "b".into(),
            object: "c".into(),
        });
        let cases = [
            (
                WalOperationRef::WriteRaftState(&state),
                WalOperation::WriteRaftState(state.clone()),
            ),
            (
                WalOperationRef::WriteAppState(&app),
                WalOperation::WriteAppState(app.clone()),
            ),
            (WalOperationRef::Commit(9), WalOperation::Commit(9)),
        ];
        for (borrowed, owned) in cases {
            let record = encode_record(5, &borrowed).unwrap();
            let entry = decode_entry(&record[8..]).expect("the record decodes and verifies");
            assert_eq!(entry.sequence, 5);
            assert_eq!(
                oxicode::serde::encode_to_vec(&entry.operation, oxicode::config::standard())
                    .unwrap(),
                oxicode::serde::encode_to_vec(&owned, oxicode::config::standard()).unwrap()
            );
        }
    }
}
