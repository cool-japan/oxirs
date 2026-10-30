//! OpenRaft type configuration and storage for [`super::RaftNode`].

use super::*;
use openraft::{
    storage::{LogState, Snapshot},
    ErrorSubject, ErrorVerb, RaftLogReader, RaftSnapshotBuilder, RaftStorage, StorageIOError,
};
use std::io::Cursor;

/// Raft type configuration for OxiRS
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct OxirsTypeConfig;

impl openraft::RaftTypeConfig for OxirsTypeConfig {
    type D = RdfCommand;
    type R = RdfResponse;
    type NodeId = OxirsNodeId;
    type Node = BasicNode;
    type Entry = Entry<Self>;
    type SnapshotData = Cursor<Vec<u8>>;
    type AsyncRuntime = openraft::TokioRuntime;
    type Responder = openraft::impls::OneshotResponder<Self>;
}

/// Raft storage implementation
pub struct OxirsStorage {
    /// Current Raft state
    pub state: Arc<RwLock<RdfApp>>,
    /// Persistent log entries
    pub log: Arc<RwLock<Vec<Entry<OxirsTypeConfig>>>>,
    /// Hard state (term, vote, committed)
    pub hard_state: Arc<RwLock<(u64, Option<OxirsNodeId>, Option<LogId<OxirsNodeId>>)>>,
    /// Last applied log index
    pub last_applied: Arc<RwLock<Option<LogId<OxirsNodeId>>>>,
    /// Last applied membership config, tracked from `EntryPayload::Membership`
    /// entries as they are applied. Required so `last_applied_state` and
    /// `build_snapshot` report the cluster's *actual* membership instead of
    /// a hardcoded empty one — openraft consults this on startup/restart to
    /// know who is in the cluster, and it is embedded in every snapshot.
    pub last_membership: Arc<RwLock<openraft::StoredMembership<OxirsNodeId, BasicNode>>>,
    /// Current snapshot
    pub snapshot: Arc<RwLock<Option<Snapshot<OxirsTypeConfig>>>>,
    /// Optional durable backing store. When `Some`, every mutation to the
    /// vote, log, committed index, state machine, membership, and snapshot
    /// is written through to disk (with `fsync` before acknowledging the
    /// vote and log appends), and the in-memory state above is reloaded
    /// from it on construction via [`OxirsStorage::new_with_dir`]. When
    /// `None` (genuine single-node / test paths), the store is purely
    /// in-memory, exactly as before.
    pub persist: Option<Arc<crate::raft_durable::DurableRaftStore>>,
    /// Applies accumulated since the last durable state-machine checkpoint
    /// (see `STATE_MACHINE_CHECKPOINT_INTERVAL`). Held in an `Arc` so the
    /// count is shared across the clones `openraft::storage::Adaptor` makes
    /// of this store — i.e. it is global to the single state-machine
    /// applier, not per-clone.
    pub apply_since_checkpoint: Arc<AtomicU64>,
}

impl OxirsStorage {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(RdfApp::default())),
            log: Arc::new(RwLock::new(Vec::new())),
            hard_state: Arc::new(RwLock::new((0, None, None))),
            last_applied: Arc::new(RwLock::new(None)),
            last_membership: Arc::new(RwLock::new(openraft::StoredMembership::default())),
            snapshot: Arc::new(RwLock::new(None)),
            persist: None,
            apply_since_checkpoint: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Construct a durable storage backed by `<data_dir>/raft/`, reloading
    /// any previously-persisted Raft state (vote, log, committed index,
    /// applied state machine, membership, snapshot) so a restarted node
    /// rebuilds from disk instead of an empty in-memory `Vec`. This is what
    /// gives the node Raft's required durability across process restarts.
    pub fn new_with_dir(data_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        use crate::raft_durable::DurableRaftStore;
        use std::io::Cursor;

        let (store, loaded) = DurableRaftStore::open(data_dir)?;

        let hard_state = loaded.hard_state.unwrap_or((0, None, None));
        let (last_applied, membership, state) = match loaded.state_machine {
            Some(sm) => (sm.last_applied, sm.membership, sm.app),
            None => (
                None,
                openraft::StoredMembership::default(),
                RdfApp::default(),
            ),
        };
        let snapshot = loaded.snapshot.map(|snap| Snapshot {
            meta: SnapshotMeta {
                last_log_id: snap.last_log_id,
                last_membership: snap.membership,
                snapshot_id: snap.snapshot_id,
            },
            snapshot: Box::new(Cursor::new(snap.data)),
        });

        Ok(Self {
            state: Arc::new(RwLock::new(state)),
            log: Arc::new(RwLock::new(loaded.log)),
            hard_state: Arc::new(RwLock::new(hard_state)),
            last_applied: Arc::new(RwLock::new(last_applied)),
            last_membership: Arc::new(RwLock::new(membership)),
            snapshot: Arc::new(RwLock::new(snapshot)),
            persist: Some(Arc::new(store)),
            apply_since_checkpoint: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Persist the current hard state tuple through the durable store, if
    /// one is configured.
    async fn persist_hard_state(&self) -> Result<(), StorageError<OxirsNodeId>> {
        if let Some(store) = self.persist.as_ref() {
            let hs = *self.hard_state.read().await;
            store
                .persist_hard_state(&hs)
                .map_err(|e| StorageError::IO {
                    source: StorageIOError::new(
                        ErrorSubject::Vote,
                        ErrorVerb::Write,
                        openraft::AnyError::error(e),
                    ),
                })?;
        }
        Ok(())
    }

    /// Persist the applied state-machine checkpoint through the durable
    /// store, if one is configured.
    async fn persist_state_machine(&self) -> Result<(), StorageError<OxirsNodeId>> {
        if let Some(store) = self.persist.as_ref() {
            let sm = crate::raft_durable::PersistedStateMachine {
                app: self.state.read().await.clone(),
                last_applied: *self.last_applied.read().await,
                membership: self.last_membership.read().await.clone(),
            };
            store
                .persist_state_machine(&sm)
                .map_err(|e| StorageError::IO {
                    source: StorageIOError::new(
                        ErrorSubject::StateMachine,
                        ErrorVerb::Write,
                        openraft::AnyError::error(e),
                    ),
                })?;
        }
        Ok(())
    }
}

impl RaftLogReader<OxirsTypeConfig> for OxirsStorage {
    async fn try_get_log_entries<
        RB: std::ops::RangeBounds<u64> + Clone + std::fmt::Debug + Send,
    >(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<OxirsTypeConfig>>, StorageError<OxirsNodeId>> {
        let log = self.log.read().await;
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => u64::MAX,
        };

        let entries = log
            .iter()
            .filter(|entry| entry.log_id.index >= start && entry.log_id.index < end)
            .cloned()
            .collect();
        Ok(entries)
    }
}

impl RaftSnapshotBuilder<OxirsTypeConfig> for OxirsStorage {
    async fn build_snapshot(
        &mut self,
    ) -> Result<Snapshot<OxirsTypeConfig>, StorageError<OxirsNodeId>> {
        let state = self.state.read().await;
        let last_applied = *self.last_applied.read().await;
        // Report the *actual* last-applied membership (tracked as
        // `EntryPayload::Membership` entries are applied in
        // `apply_to_state_machine`), not a hardcoded empty one — a snapshot
        // with an empty membership would tell a node restoring from it that
        // the cluster has no voters at all.
        let last_membership = self.last_membership.read().await.clone();

        let data = serde_json::to_vec(&*state).map_err(|e| StorageError::IO {
            source: StorageIOError::new(
                ErrorSubject::StateMachine,
                ErrorVerb::Write,
                openraft::AnyError::new(&e),
            ),
        })?;
        // `state` is now captured into `data`; release the read guard
        // before `persist_state_machine` (below) re-acquires it.
        drop(state);

        let snapshot_id = format!("snapshot-{}", last_applied.map_or(0, |id| id.index));
        let snapshot = Snapshot {
            meta: SnapshotMeta {
                last_log_id: last_applied,
                last_membership: last_membership.clone(),
                snapshot_id: snapshot_id.clone(),
            },
            snapshot: Box::new(Cursor::new(data.clone())),
        };

        // Durably persist the built snapshot so a restart can serve/restore
        // from it instead of losing it with the process.
        if let Some(store) = self.persist.as_ref() {
            let persisted = crate::raft_durable::PersistedSnapshot {
                data,
                last_log_id: last_applied,
                membership: last_membership,
                snapshot_id,
            };
            store
                .persist_snapshot(&persisted)
                .map_err(|e| StorageError::IO {
                    source: StorageIOError::new(
                        ErrorSubject::Snapshot(None),
                        ErrorVerb::Write,
                        openraft::AnyError::error(e),
                    ),
                })?;
        }

        *self.snapshot.write().await = Some(snapshot.clone());

        // Force a durable state-machine checkpoint at (>=) this snapshot's
        // last-included index. openraft may `purge_logs_upto` that index
        // immediately after building the snapshot; persisting the checkpoint
        // here (with the current, >= snapshot `last_applied`) guarantees the
        // on-disk checkpoint always covers the purge point, so a restart can
        // still replay the surviving log tail. Reset the apply counter since
        // the state machine is now fully persisted.
        self.persist_state_machine().await?;
        self.apply_since_checkpoint.store(0, Ordering::Relaxed);

        Ok(snapshot)
    }
}

impl RaftStorage<OxirsTypeConfig> for OxirsStorage {
    type LogReader = Self;
    type SnapshotBuilder = Self;

    async fn save_committed(
        &mut self,
        committed: Option<LogId<OxirsNodeId>>,
    ) -> Result<(), StorageError<OxirsNodeId>> {
        self.hard_state.write().await.2 = committed;
        self.persist_hard_state().await?;
        Ok(())
    }

    async fn read_committed(
        &mut self,
    ) -> Result<Option<LogId<OxirsNodeId>>, StorageError<OxirsNodeId>> {
        Ok(self.hard_state.read().await.2)
    }

    async fn save_vote(
        &mut self,
        vote: &openraft::Vote<OxirsNodeId>,
    ) -> Result<(), StorageError<OxirsNodeId>> {
        {
            let mut hard_state = self.hard_state.write().await;
            hard_state.0 = vote.leader_id.term;
            hard_state.1 = vote.leader_id.voted_for();
        }
        // The vote MUST be durable before this returns: Raft's single-vote
        // -per-term safety depends on a restarted node remembering it.
        self.persist_hard_state().await?;
        Ok(())
    }

    async fn read_vote(
        &mut self,
    ) -> Result<Option<openraft::Vote<OxirsNodeId>>, StorageError<OxirsNodeId>> {
        let hard_state = self.hard_state.read().await;
        if let Some(node_id) = hard_state.1 {
            Ok(Some(openraft::Vote::new(hard_state.0, node_id)))
        } else {
            Ok(None)
        }
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }

    async fn get_log_state(
        &mut self,
    ) -> Result<LogState<OxirsTypeConfig>, StorageError<OxirsNodeId>> {
        let log = self.log.read().await;
        let last_log_id = log.last().map(|entry| entry.log_id);
        let last_purged_log_id = None; // We don't track purged logs in this simple implementation

        Ok(LogState {
            last_purged_log_id,
            last_log_id,
        })
    }

    async fn append_to_log<I>(&mut self, entries: I) -> Result<(), StorageError<OxirsNodeId>>
    where
        I: IntoIterator<Item = Entry<OxirsTypeConfig>> + Send,
    {
        let appended: Vec<Entry<OxirsTypeConfig>> = entries.into_iter().collect();
        {
            let mut log = self.log.write().await;
            log.extend(appended.iter().cloned());
        }
        // Durably append (fsync) before acknowledging: a committed entry
        // must survive a restart.
        if let Some(store) = self.persist.as_ref() {
            store.append_log(&appended).map_err(|e| StorageError::IO {
                source: StorageIOError::new(
                    ErrorSubject::Logs,
                    ErrorVerb::Write,
                    openraft::AnyError::error(e),
                ),
            })?;
        }
        Ok(())
    }

    async fn delete_conflict_logs_since(
        &mut self,
        log_id: LogId<OxirsNodeId>,
    ) -> Result<(), StorageError<OxirsNodeId>> {
        let mut log = self.log.write().await;
        log.retain(|entry| entry.log_id.index < log_id.index);
        // Append-only can't represent a truncation: rewrite the whole log.
        if let Some(store) = self.persist.as_ref() {
            store.rewrite_log(&log).map_err(|e| StorageError::IO {
                source: StorageIOError::new(
                    ErrorSubject::Logs,
                    ErrorVerb::Delete,
                    openraft::AnyError::error(e),
                ),
            })?;
        }
        Ok(())
    }

    async fn purge_logs_upto(
        &mut self,
        log_id: LogId<OxirsNodeId>,
    ) -> Result<(), StorageError<OxirsNodeId>> {
        let mut log = self.log.write().await;
        log.retain(|entry| entry.log_id.index > log_id.index);
        if let Some(store) = self.persist.as_ref() {
            store.rewrite_log(&log).map_err(|e| StorageError::IO {
                source: StorageIOError::new(
                    ErrorSubject::Logs,
                    ErrorVerb::Delete,
                    openraft::AnyError::error(e),
                ),
            })?;
        }
        Ok(())
    }

    async fn last_applied_state(
        &mut self,
    ) -> Result<
        (
            Option<LogId<OxirsNodeId>>,
            openraft::StoredMembership<OxirsNodeId, BasicNode>,
        ),
        StorageError<OxirsNodeId>,
    > {
        let last_applied = *self.last_applied.read().await;
        let membership = self.last_membership.read().await.clone();
        Ok((last_applied, membership))
    }

    async fn apply_to_state_machine(
        &mut self,
        entries: &[Entry<OxirsTypeConfig>],
    ) -> Result<Vec<RdfResponse>, StorageError<OxirsNodeId>> {
        // openraft requires exactly one response per input entry, at the
        // same index (`debug_assert_eq!(n_entries, n_replies, ...)` in
        // `openraft::core::sm::worker::StateMachineWorker::apply`) — it uses
        // that positional correspondence to route each response back to the
        // client awaiting it. Every entry, not just `Normal`, must therefore
        // push exactly one response; silently skipping `Membership`/`Blank`
        // entries (as this used to) would misalign every response after the
        // first non-`Normal` entry and trip that assert on the very first
        // membership change (e.g. the bootstrap entry from `initialize()`).
        let mut responses = Vec::with_capacity(entries.len());
        {
            let mut state = self.state.write().await;

            for entry in entries {
                let response = match &entry.payload {
                    EntryPayload::Normal(cmd) => state.apply_command(cmd),
                    EntryPayload::Blank => RdfResponse::Success,
                    EntryPayload::Membership(membership) => {
                        // Track the actual last-applied membership so
                        // `last_applied_state`/`build_snapshot` report the real
                        // cluster configuration instead of a hardcoded empty one.
                        *self.last_membership.write().await =
                            openraft::StoredMembership::new(Some(entry.log_id), membership.clone());
                        RdfResponse::Success
                    }
                };
                responses.push(response);
                *self.last_applied.write().await = Some(entry.log_id);
            }
        } // drop the state write guard before persisting (persist reads it)

        // Durably checkpoint the applied state machine only *periodically*.
        // The fsync'd durable log (see `append_to_log`) is the source of
        // truth and already guarantees every committed entry survives a
        // restart; on startup openraft re-applies committed log entries
        // past the persisted checkpoint's `last_applied`. Rewriting the
        // entire, monotonically-growing state machine on every apply was
        // therefore redundant and O(n^2) over a long write burst — each
        // apply re-serialized all prior triples and fsync'd an ever-larger
        // file. Checkpointing every `STATE_MACHINE_CHECKPOINT_INTERVAL`
        // applies bounds restart replay to that many entries while keeping
        // the hot commit path O(1) amortized. (`apply_to_state_machine` is
        // driven single-threaded by openraft's state-machine worker, so the
        // read-modify-write of this counter needs no stronger ordering than
        // `Relaxed`.) `build_snapshot` additionally forces a checkpoint so
        // the on-disk `last_applied` always covers any later
        // `purge_logs_upto` (purge safety).
        let applied = entries.len() as u64;
        let since = self
            .apply_since_checkpoint
            .fetch_add(applied, Ordering::Relaxed)
            + applied;
        if since >= STATE_MACHINE_CHECKPOINT_INTERVAL {
            self.persist_state_machine().await?;
            self.apply_since_checkpoint.store(0, Ordering::Relaxed);
        }

        Ok(responses)
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        self.clone()
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<OxirsNodeId>> {
        Ok(Box::new(Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<OxirsNodeId, BasicNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<OxirsNodeId>> {
        let data = snapshot.get_ref();
        let new_state: RdfApp = serde_json::from_slice(data).map_err(|e| StorageError::IO {
            source: StorageIOError::new(
                ErrorSubject::StateMachine,
                ErrorVerb::Read,
                openraft::AnyError::new(&e),
            ),
        })?;

        *self.state.write().await = new_state;
        *self.last_applied.write().await = meta.last_log_id;
        // A snapshot embeds the membership as of its last-included entry;
        // a follower that catches up via snapshot (rather than individual
        // log entries) must pick that membership up too, or it would
        // still report the hardcoded/stale membership it had before.
        *self.last_membership.write().await = meta.last_membership.clone();

        // Durably persist both the installed snapshot and the resulting
        // state-machine checkpoint so a restart reloads the caught-up state.
        if let Some(store) = self.persist.as_ref() {
            let persisted = crate::raft_durable::PersistedSnapshot {
                data: snapshot.get_ref().clone(),
                last_log_id: meta.last_log_id,
                membership: meta.last_membership.clone(),
                snapshot_id: meta.snapshot_id.clone(),
            };
            store
                .persist_snapshot(&persisted)
                .map_err(|e| StorageError::IO {
                    source: StorageIOError::new(
                        ErrorSubject::Snapshot(None),
                        ErrorVerb::Write,
                        openraft::AnyError::error(e),
                    ),
                })?;
        }
        // The state machine has jumped forward to the installed snapshot's
        // point, and both snapshot and checkpoint are now persisted, so the
        // on-disk checkpoint is fully current — reset the apply counter.
        self.persist_state_machine().await?;
        self.apply_since_checkpoint.store(0, Ordering::Relaxed);

        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<OxirsTypeConfig>>, StorageError<OxirsNodeId>> {
        Ok(self.snapshot.read().await.clone())
    }
}

impl Clone for OxirsStorage {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            log: Arc::clone(&self.log),
            hard_state: Arc::clone(&self.hard_state),
            last_applied: Arc::clone(&self.last_applied),
            last_membership: Arc::clone(&self.last_membership),
            snapshot: Arc::clone(&self.snapshot),
            persist: self.persist.clone(),
            apply_since_checkpoint: Arc::clone(&self.apply_since_checkpoint),
        }
    }
}
