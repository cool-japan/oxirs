//! Durable, restart-surviving backing store for the OpenRaft state that must
//! be persisted for Raft's safety guarantees to hold.
//!
//! `OxirsStorage` (in `raft.rs`) keeps the Raft log, hard state (term/vote/
//! committed), last-applied index, membership, snapshot, and the RDF state
//! machine in memory for the hot read path OpenRaft drives. On its own that is
//! *not* durable: a process restart would lose the persisted vote and any
//! committed-but-unsnapshotted log, which violates Raft's core requirements —
//! a restarted node could vote twice in the same term or lose acknowledged
//! writes.
//!
//! This module provides [`DurableRaftStore`], a file-backed store that
//! `OxirsStorage` writes through to (with `fsync` before acknowledging the
//! operations that require durability — the vote and log appends) and reloads
//! from on startup. The on-disk layout under `<data_dir>/raft/`:
//!
//! - `hard_state.bin` — the `(term, voted_for, committed)` tuple, rewritten
//!   atomically (write-temp + `fsync` + rename) on every change. Small and
//!   fixed-shape, so a full rewrite is cheap and always crash-consistent.
//! - `log.bin` — length-prefixed (`u64` LE) oxicode frames, one per log
//!   entry, appended in order. Truncation (`delete_conflict_logs_since`) and
//!   purge (`purge_logs_upto`) rewrite this file atomically from the caller's
//!   in-memory vector, so it always mirrors the live log exactly.
//! - `state_machine.bin` — the applied `RdfApp` plus `last_applied` and the
//!   applied membership, rewritten atomically whenever entries are applied.
//! - `snapshot.bin` — the current snapshot's raw bytes plus its meta,
//!   rewritten atomically when a snapshot is built or installed.
//!
//! All encoding uses oxicode (COOLJAPAN policy: never bincode).

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use openraft::{BasicNode, Entry, LogId, StoredMembership};
use serde::{Deserialize, Serialize};

use crate::raft::{OxirsNodeId, OxirsTypeConfig, RdfApp};

/// Persisted hard state: `(term, voted_for, committed_log_id)`. Mirrors the
/// in-memory `OxirsStorage::hard_state` tuple exactly.
pub type HardState = (u64, Option<OxirsNodeId>, Option<LogId<OxirsNodeId>>);

/// The persisted state-machine checkpoint: the applied `RdfApp`, the id of the
/// last applied log entry, and the membership as of that entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedStateMachine {
    pub app: RdfApp,
    pub last_applied: Option<LogId<OxirsNodeId>>,
    pub membership: StoredMembership<OxirsNodeId, BasicNode>,
}

/// A persisted snapshot: the raw serialized state-machine bytes plus the meta
/// OpenRaft needs to reason about it (last log id, membership, snapshot id).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSnapshot {
    pub data: Vec<u8>,
    pub last_log_id: Option<LogId<OxirsNodeId>>,
    pub membership: StoredMembership<OxirsNodeId, BasicNode>,
    pub snapshot_id: String,
}

/// Everything reloaded from disk when a durable store is opened on an existing
/// data directory. Empty (all `None`/`default`) for a fresh directory.
#[derive(Debug, Default)]
pub struct LoadedState {
    pub hard_state: Option<HardState>,
    pub log: Vec<Entry<OxirsTypeConfig>>,
    pub state_machine: Option<PersistedStateMachine>,
    pub snapshot: Option<PersistedSnapshot>,
}

/// File-backed durable store for OpenRaft's persistent state. Cheap to
/// `clone` for sharing between the log-store and state-machine halves that
/// `openraft::storage::Adaptor` splits `OxirsStorage` into — it holds only the
/// directory path.
#[derive(Debug, Clone)]
pub struct DurableRaftStore {
    dir: PathBuf,
}

impl DurableRaftStore {
    /// Open (creating if necessary) a durable store rooted at
    /// `<data_dir>/raft/`, and load any previously-persisted state.
    pub fn open(data_dir: impl AsRef<Path>) -> Result<(Self, LoadedState)> {
        let dir = data_dir.as_ref().join("raft");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating raft data dir {}", dir.display()))?;
        let store = Self { dir };
        let loaded = store.load()?;
        Ok((store, loaded))
    }

    fn hard_state_path(&self) -> PathBuf {
        self.dir.join("hard_state.bin")
    }
    fn log_path(&self) -> PathBuf {
        self.dir.join("log.bin")
    }
    fn state_machine_path(&self) -> PathBuf {
        self.dir.join("state_machine.bin")
    }
    fn snapshot_path(&self) -> PathBuf {
        self.dir.join("snapshot.bin")
    }

    /// Read and decode everything currently on disk.
    fn load(&self) -> Result<LoadedState> {
        Ok(LoadedState {
            hard_state: self.load_hard_state()?,
            log: self.load_log()?,
            state_machine: self.load_state_machine()?,
            snapshot: self.load_snapshot()?,
        })
    }

    fn load_hard_state(&self) -> Result<Option<HardState>> {
        decode_file(&self.hard_state_path())
    }

    fn load_state_machine(&self) -> Result<Option<PersistedStateMachine>> {
        decode_file(&self.state_machine_path())
    }

    fn load_snapshot(&self) -> Result<Option<PersistedSnapshot>> {
        decode_file(&self.snapshot_path())
    }

    /// Read the length-prefixed log frames back into a vector. A trailing
    /// partial frame (a crash mid-append) is treated as end-of-log rather than
    /// an error: the entry was never acknowledged, so dropping it is safe and
    /// exactly what Raft expects.
    fn load_log(&self) -> Result<Vec<Entry<OxirsTypeConfig>>> {
        let path = self.log_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let bytes =
            std::fs::read(&path).with_context(|| format!("reading raft log {}", path.display()))?;
        let mut entries = Vec::new();
        let mut offset = 0usize;
        while offset + 8 <= bytes.len() {
            let mut len_buf = [0u8; 8];
            len_buf.copy_from_slice(&bytes[offset..offset + 8]);
            let len = u64::from_le_bytes(len_buf) as usize;
            offset += 8;
            if offset + len > bytes.len() {
                // Truncated final frame from a crash mid-append: stop here.
                break;
            }
            match oxicode::serde::decode_from_slice::<Entry<OxirsTypeConfig>, _>(
                &bytes[offset..offset + len],
                oxicode::config::standard(),
            ) {
                Ok((entry, _)) => entries.push(entry),
                Err(_) => break,
            }
            offset += len;
        }
        Ok(entries)
    }

    /// Persist the hard state (term/vote/committed). Atomic + fsync'd, because
    /// the vote must survive a crash for Raft's single-vote-per-term safety.
    pub fn persist_hard_state(&self, hard_state: &HardState) -> Result<()> {
        atomic_write(&self.hard_state_path(), &encode(hard_state)?)
    }

    /// Append log entries as length-prefixed frames, fsync'd before returning:
    /// an acknowledged append must be durable so a committed entry can never be
    /// lost across a restart.
    pub fn append_log(&self, entries: &[Entry<OxirsTypeConfig>]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut buf = Vec::new();
        for entry in entries {
            let body = encode(entry)?;
            buf.extend_from_slice(&(body.len() as u64).to_le_bytes());
            buf.extend_from_slice(&body);
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.log_path())
            .with_context(|| format!("opening raft log {}", self.log_path().display()))?;
        file.write_all(&buf).context("appending raft log")?;
        file.sync_all().context("fsync raft log")?;
        Ok(())
    }

    /// Rewrite the entire log from the given (already truncated/purged)
    /// in-memory vector. Used by conflict-truncation and purge, where an
    /// append-only file can no longer represent the live log.
    pub fn rewrite_log(&self, entries: &[Entry<OxirsTypeConfig>]) -> Result<()> {
        let mut buf = Vec::new();
        for entry in entries {
            let body = encode(entry)?;
            buf.extend_from_slice(&(body.len() as u64).to_le_bytes());
            buf.extend_from_slice(&body);
        }
        atomic_write(&self.log_path(), &buf)
    }

    /// Persist the applied state-machine checkpoint (RDF state + last-applied +
    /// membership). Atomic; not individually fsync'd on the apply hot path
    /// beyond the atomic rename, since the log is the source of truth and the
    /// state machine can always be rebuilt by replaying it.
    pub fn persist_state_machine(&self, sm: &PersistedStateMachine) -> Result<()> {
        atomic_write(&self.state_machine_path(), &encode(sm)?)
    }

    /// Persist the current snapshot (raw bytes + meta), atomically.
    pub fn persist_snapshot(&self, snapshot: &PersistedSnapshot) -> Result<()> {
        atomic_write(&self.snapshot_path(), &encode(snapshot)?)
    }
}

/// oxicode-encode any serializable value (COOLJAPAN policy: not bincode).
fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    oxicode::serde::encode_to_vec(value, oxicode::config::standard())
        .context("oxicode encode failed")
}

/// Decode an oxicode-encoded file, returning `None` if it does not exist.
fn decode_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Option<T>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.is_empty() {
        return Ok(None);
    }
    let (value, _) = oxicode::serde::decode_from_slice::<T, _>(&bytes, oxicode::config::standard())
        .with_context(|| format!("decoding {}", path.display()))?;
    Ok(Some(value))
}

/// Crash-consistent write: write to a sibling `.tmp` file, `fsync` it, then
/// atomically rename over the target. A crash leaves either the old complete
/// file or the new complete file — never a torn one.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut file = std::fs::File::create(&tmp)
            .with_context(|| format!("creating temp file {}", tmp.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("writing {}", tmp.display()))?;
        file.sync_all()
            .with_context(|| format!("fsync {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use openraft::{CommittedLeaderId, EntryPayload, Membership};
    use std::collections::BTreeMap;

    fn temp_dir(tag: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "oxirs_durable_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        dir
    }

    fn log_id(term: u64, index: u64) -> LogId<OxirsNodeId> {
        LogId::new(CommittedLeaderId::new(term, 0), index)
    }

    /// Persisting hard state, a log, and a state-machine checkpoint, then
    /// re-opening the store on the same directory, must reload identical state
    /// — proving Raft state survives a process restart (the durability the
    /// audit flagged as missing).
    #[test]
    fn regression_durable_state_survives_reopen() {
        let dir = temp_dir("reopen");

        {
            let (store, loaded) = DurableRaftStore::open(&dir).expect("open fresh durable store");
            assert!(loaded.hard_state.is_none());
            assert!(loaded.log.is_empty());
            assert!(loaded.state_machine.is_none());

            // Persist a vote (term 7, voted for node 3) + committed index.
            let hs: HardState = (7, Some(3), Some(log_id(7, 2)));
            store.persist_hard_state(&hs).expect("persist hard state");

            // Append two normal log entries.
            let entries = vec![
                Entry {
                    log_id: log_id(7, 1),
                    payload: EntryPayload::Normal(crate::raft::RdfCommand::Insert {
                        subject: "s".into(),
                        predicate: "p".into(),
                        object: "o".into(),
                    }),
                },
                Entry {
                    log_id: log_id(7, 2),
                    payload: EntryPayload::Blank,
                },
            ];
            store.append_log(&entries).expect("append log");

            // Persist a state-machine checkpoint with real membership.
            let mut app = RdfApp::default();
            app.triples.insert(("s".into(), "p".into(), "o".into()));
            let members: BTreeMap<OxirsNodeId, BasicNode> = [
                (1, BasicNode::new("127.0.0.1:1")),
                (3, BasicNode::new("127.0.0.1:3")),
            ]
            .into_iter()
            .collect();
            let membership = StoredMembership::new(
                Some(log_id(7, 2)),
                Membership::new(vec![members.keys().copied().collect()], members),
            );
            store
                .persist_state_machine(&PersistedStateMachine {
                    app: app.clone(),
                    last_applied: Some(log_id(7, 2)),
                    membership: membership.clone(),
                })
                .expect("persist state machine");
        }

        // Re-open: everything must come back intact.
        let (_store, loaded) = DurableRaftStore::open(&dir).expect("re-open durable store");
        assert_eq!(loaded.hard_state, Some((7, Some(3), Some(log_id(7, 2)))));
        assert_eq!(loaded.log.len(), 2);
        assert_eq!(loaded.log[0].log_id, log_id(7, 1));
        assert_eq!(loaded.log[1].log_id, log_id(7, 2));
        let sm = loaded.state_machine.expect("state machine reloaded");
        assert_eq!(sm.last_applied, Some(log_id(7, 2)));
        assert_eq!(sm.app.triples.len(), 1);
        assert_eq!(sm.membership.voter_ids().collect::<Vec<_>>(), vec![1, 3]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A conflict-truncation / purge rewrite must replace the on-disk log with
    /// exactly the surviving entries.
    #[test]
    fn regression_durable_log_rewrite_truncates() {
        let dir = temp_dir("rewrite");
        let (store, _loaded) = DurableRaftStore::open(&dir).expect("open");

        let entries: Vec<Entry<OxirsTypeConfig>> = (1..=5)
            .map(|i| Entry {
                log_id: log_id(1, i),
                payload: EntryPayload::Blank,
            })
            .collect();
        store.append_log(&entries).expect("append");

        // Keep only the first two (as if truncating conflicts since index 3).
        store.rewrite_log(&entries[..2]).expect("rewrite");

        let (_s, loaded) = DurableRaftStore::open(&dir).expect("re-open");
        assert_eq!(loaded.log.len(), 2);
        assert_eq!(loaded.log[1].log_id.index, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
