//! # Raft Consensus Implementation
//!
//! Raft consensus algorithm implementation for distributed RDF storage.
//! Uses openraft for production-ready consensus.

use anyhow::Result;
use serde::{Deserialize, Serialize};
#[cfg(feature = "raft")]
use std::collections::HashMap;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "raft")]
use std::net::SocketAddr;
#[cfg(feature = "raft")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
#[cfg(feature = "raft")]
use std::time::Duration;
use tokio::sync::RwLock;

/// Errors specific to Raft consensus construction and lifecycle.
///
/// Multi-node clustering is backed by a real `openraft::Raft` instance: a
/// split log/state-machine storage (`OxirsStorage` wrapped in
/// `openraft::storage::Adaptor`), a dedicated TCP `RaftNetworkFactory`/
/// `RaftNetwork` transport (see `raft_network.rs`), and `Raft::new`. The
/// variants below cover the ways *constructing* that real instance can
/// legitimately fail — they are not a "we can't do this yet" placeholder.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RaftClusterError {
    /// `init_raft` was asked to form a real multi-node cluster (peers beyond
    /// `self` were requested), but this node has no network configuration
    /// (a bind address and/or one or more peer addresses) to build a working
    /// transport with. Call [`RaftNode::set_network`] before `init_raft` for
    /// a multi-node peer set.
    #[error(
        "cannot initialize multi-node Raft consensus for node {node_id} ({peer_count} peer(s) requested): \
         missing network configuration ({detail}). Call `RaftNode::set_network` with this node's bind address \
         and every peer's address before `init_raft`."
    )]
    NetworkNotConfigured {
        node_id: OxirsNodeId,
        peer_count: usize,
        detail: String,
    },

    /// An operation that requires real cross-node consensus was attempted on
    /// a node whose multi-node Raft instance is not currently running (never
    /// constructed, failed to construct, or has since been shut down).
    #[error(
        "node {node_id} cannot service this operation: multi-node Raft consensus is not currently running on \
         this node."
    )]
    ConsensusUnavailable { node_id: OxirsNodeId },
}

/// Global shared storage for testing when Raft is not available
static GLOBAL_SHARED_STORAGE: OnceLock<Arc<RwLock<RdfApp>>> = OnceLock::new();

/// Initialize global shared storage (for testing)
pub fn init_global_shared_storage() -> Arc<RwLock<RdfApp>> {
    GLOBAL_SHARED_STORAGE
        .get_or_init(|| Arc::new(RwLock::new(RdfApp::default())))
        .clone()
}

/// Get global shared storage (for testing)
pub fn get_global_shared_storage() -> Option<Arc<RwLock<RdfApp>>> {
    GLOBAL_SHARED_STORAGE.get().cloned()
}

/// Reset global shared storage (for testing isolation)
pub async fn reset_global_shared_storage() {
    if let Some(storage) = GLOBAL_SHARED_STORAGE.get() {
        let mut state = storage.write().await;
        *state = RdfApp::default();
    }
}

#[cfg(feature = "raft")]
use openraft::{
    BasicNode, Entry, EntryPayload, LogId, Raft, RaftMetrics, SnapshotMeta, StorageError,
};

/// Node ID type for Raft
pub type OxirsNodeId = u64;

/// Raft request ID type
pub type OxirsRequestId = u64;

/// RDF command types that can be replicated
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RdfCommand {
    /// Insert a triple
    Insert {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Delete a triple  
    Delete {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Clear all triples
    Clear,
    /// Begin transaction
    BeginTransaction { tx_id: String },
    /// Commit transaction
    CommitTransaction { tx_id: String },
    /// Rollback transaction
    RollbackTransaction { tx_id: String },
    /// Add a new node to the cluster
    AddNode {
        node_id: OxirsNodeId,
        address: String,
    },
    /// Remove a node from the cluster
    RemoveNode { node_id: OxirsNodeId },
    /// Transfer leadership to another node
    TransferLeadership { target_node: OxirsNodeId },
    /// Force evict a non-responsive node
    ForceEvictNode { node_id: OxirsNodeId },
}

/// RDF response from command execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RdfResponse {
    /// Operation successful
    Success,
    /// Operation failed
    Error(String),
    /// Transaction started
    TransactionStarted { tx_id: String },
    /// Transaction committed
    TransactionCommitted { tx_id: String },
    /// Transaction rolled back
    TransactionRolledBack { tx_id: String },
}

/// Raft application data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RdfApp {
    /// In-memory triple store for demonstration
    /// In production, this would interface with oxirs-tdb
    pub triples: BTreeSet<(String, String, String)>,
    /// Active transactions
    pub transactions: BTreeMap<String, BTreeSet<(String, String, String)>>,
    /// Shard-based storage for distributed operations
    pub shards: BTreeMap<crate::shard::ShardId, BTreeSet<(String, String, String)>>,
    /// Shards marked for deletion
    pub deleted_shards: BTreeSet<crate::shard::ShardId>,
}

impl RdfApp {
    /// Apply a command to the state machine
    pub fn apply_command(&mut self, cmd: &RdfCommand) -> RdfResponse {
        match cmd {
            RdfCommand::Insert {
                subject,
                predicate,
                object,
            } => {
                self.triples
                    .insert((subject.clone(), predicate.clone(), object.clone()));
                RdfResponse::Success
            }
            RdfCommand::Delete {
                subject,
                predicate,
                object,
            } => {
                self.triples
                    .remove(&(subject.clone(), predicate.clone(), object.clone()));
                RdfResponse::Success
            }
            RdfCommand::Clear => {
                self.triples.clear();
                RdfResponse::Success
            }
            RdfCommand::BeginTransaction { tx_id } => {
                self.transactions.insert(tx_id.clone(), BTreeSet::new());
                RdfResponse::TransactionStarted {
                    tx_id: tx_id.clone(),
                }
            }
            RdfCommand::CommitTransaction { tx_id } => {
                if let Some(tx_triples) = self.transactions.remove(tx_id) {
                    self.triples.extend(tx_triples);
                    RdfResponse::TransactionCommitted {
                        tx_id: tx_id.clone(),
                    }
                } else {
                    RdfResponse::Error(format!("Transaction {tx_id} not found"))
                }
            }
            RdfCommand::RollbackTransaction { tx_id } => {
                if self.transactions.remove(tx_id).is_some() {
                    RdfResponse::TransactionRolledBack {
                        tx_id: tx_id.clone(),
                    }
                } else {
                    RdfResponse::Error(format!("Transaction {tx_id} not found"))
                }
            }
            RdfCommand::AddNode { node_id, address } => {
                // Log the configuration change
                tracing::info!("Adding node {} at address {} to cluster", node_id, address);
                RdfResponse::Success
            }
            RdfCommand::RemoveNode { node_id } => {
                // Log the configuration change
                tracing::info!("Removing node {} from cluster", node_id);
                RdfResponse::Success
            }
            RdfCommand::TransferLeadership { target_node } => {
                // Log the leadership transfer
                tracing::info!("Transferring leadership to node {}", target_node);
                RdfResponse::Success
            }
            RdfCommand::ForceEvictNode { node_id } => {
                // Log the forced eviction
                tracing::warn!("Force evicting node {} from cluster", node_id);
                RdfResponse::Success
            }
        }
    }

    /// Get number of triples
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Query triples by pattern (simplified)
    pub fn query(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<(String, String, String)> {
        self.triples
            .iter()
            .filter(|(s, p, o)| {
                subject.map_or(true, |subj| s == subj)
                    && predicate.map_or(true, |pred| p == pred)
                    && object.map_or(true, |obj| o == obj)
            })
            .cloned()
            .collect()
    }

    /// Create a new shard
    pub fn create_shard(&mut self, shard_id: crate::shard::ShardId) {
        self.shards.insert(shard_id, BTreeSet::new());
    }

    /// Delete a shard
    pub fn delete_shard(&mut self, shard_id: crate::shard::ShardId) {
        self.shards.remove(&shard_id);
        self.deleted_shards.remove(&shard_id);
    }

    /// Insert a triple into a specific shard
    pub fn insert_triple_to_shard(
        &mut self,
        shard_id: crate::shard::ShardId,
        triple: oxirs_core::model::Triple,
    ) {
        let triple_tuple = (
            triple.subject().to_string(),
            triple.predicate().to_string(),
            triple.object().to_string(),
        );

        let shard = self.shards.entry(shard_id).or_default();
        shard.insert(triple_tuple);
    }

    /// Delete a triple from a specific shard
    pub fn delete_triple_from_shard(
        &mut self,
        shard_id: crate::shard::ShardId,
        triple: &oxirs_core::model::Triple,
    ) {
        let triple_tuple = (
            triple.subject().to_string(),
            triple.predicate().to_string(),
            triple.object().to_string(),
        );

        if let Some(shard) = self.shards.get_mut(&shard_id) {
            shard.remove(&triple_tuple);
        }
    }

    /// Query triples from a specific shard
    pub fn query_shard(
        &self,
        shard_id: crate::shard::ShardId,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<oxirs_core::model::Triple> {
        if let Some(shard) = self.shards.get(&shard_id) {
            shard
                .iter()
                .filter(|(s, p, o)| {
                    subject.map_or(true, |subj| s == subj)
                        && predicate.map_or(true, |pred| p == pred)
                        && object.map_or(true, |obj| o == obj)
                })
                .filter_map(|(s, p, o)| {
                    // Convert string tuple back to Triple
                    // This is a simplified conversion; in practice you'd want proper parsing
                    use oxirs_core::model::{Literal, NamedNode, Triple};
                    if let (Ok(subj), Ok(pred)) = (NamedNode::new(s), NamedNode::new(p)) {
                        // Try to parse object as NamedNode first, then as Literal
                        if let Ok(obj_node) = NamedNode::new(o) {
                            Some(Triple::new(subj, pred, obj_node))
                        } else {
                            // Treat as literal
                            Some(Triple::new(subj, pred, Literal::new_simple_literal(o)))
                        }
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get shard size in bytes (estimated)
    pub fn get_shard_size(&self, shard_id: crate::shard::ShardId) -> u64 {
        if let Some(shard) = self.shards.get(&shard_id) {
            // Estimate 100 bytes per triple
            (shard.len() * 100) as u64
        } else {
            0
        }
    }

    /// Get shard triple count
    pub fn get_shard_triple_count(&self, shard_id: crate::shard::ShardId) -> usize {
        self.shards.get(&shard_id).map_or(0, |shard| shard.len())
    }

    /// Export all triples from a shard
    pub fn export_shard(&self, shard_id: crate::shard::ShardId) -> Vec<oxirs_core::model::Triple> {
        self.query_shard(shard_id, None, None, None)
    }

    /// Import triples into a shard
    pub fn import_shard(
        &mut self,
        shard_id: crate::shard::ShardId,
        triples: Vec<oxirs_core::model::Triple>,
    ) {
        let shard = self.shards.entry(shard_id).or_default();
        for triple in triples {
            let triple_tuple = (
                triple.subject().to_string(),
                triple.predicate().to_string(),
                triple.object().to_string(),
            );
            shard.insert(triple_tuple);
        }
    }

    /// Get all triples from a shard
    pub fn get_shard_triples(
        &self,
        shard_id: crate::shard::ShardId,
    ) -> Vec<oxirs_core::model::Triple> {
        self.export_shard(shard_id)
    }

    /// Insert multiple triples into a shard
    pub fn insert_triples_to_shard(
        &mut self,
        shard_id: crate::shard::ShardId,
        triples: Vec<oxirs_core::model::Triple>,
    ) {
        let shard = self.shards.entry(shard_id).or_default();
        for triple in triples {
            let triple_tuple = (
                triple.subject().to_string(),
                triple.predicate().to_string(),
                triple.object().to_string(),
            );
            shard.insert(triple_tuple);
        }
    }

    /// Mark a shard for deletion
    pub fn mark_shard_for_deletion(&mut self, shard_id: crate::shard::ShardId) {
        self.deleted_shards.insert(shard_id);
    }
}

#[cfg(feature = "raft")]
mod raft_impl {
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
            }
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

            let snapshot = Snapshot {
                meta: SnapshotMeta {
                    last_log_id: last_applied,
                    last_membership,
                    snapshot_id: format!("snapshot-{}", last_applied.map_or(0, |id| id.index)),
                },
                snapshot: Box::new(Cursor::new(data)),
            };

            *self.snapshot.write().await = Some(snapshot.clone());
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
            let mut hard_state = self.hard_state.write().await;
            hard_state.0 = vote.leader_id.term;
            hard_state.1 = vote.leader_id.voted_for();
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
            let mut log = self.log.write().await;
            log.extend(entries);
            Ok(())
        }

        async fn delete_conflict_logs_since(
            &mut self,
            log_id: LogId<OxirsNodeId>,
        ) -> Result<(), StorageError<OxirsNodeId>> {
            let mut log = self.log.write().await;
            log.retain(|entry| entry.log_id.index < log_id.index);
            Ok(())
        }

        async fn purge_logs_upto(
            &mut self,
            log_id: LogId<OxirsNodeId>,
        ) -> Result<(), StorageError<OxirsNodeId>> {
            let mut log = self.log.write().await;
            log.retain(|entry| entry.log_id.index > log_id.index);
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
            }
        }
    }
}

#[cfg(feature = "raft")]
pub use raft_impl::*;

/// Raft node implementation
pub struct RaftNode {
    node_id: OxirsNodeId,
    #[cfg(feature = "raft")]
    raft: Option<Raft<OxirsTypeConfig>>,
    /// Set (and left set) the first time `init_raft` is asked to form a real
    /// multi-node cluster (peers other than this node were requested).
    /// Distinguishes "multi-node was requested" (so `raft.is_none()` must
    /// never be read as an honest single-node leader, even after a
    /// construction failure or a `shutdown()`) from "multi-node was never
    /// requested" (so `raft.is_none()` legitimately means single-node mode).
    #[cfg(feature = "raft")]
    multi_node_requested: AtomicBool,
    /// Set on this node's first-ever call to `init_raft` for a multi-node
    /// peer set, regardless of whether this node is the designated
    /// bootstrapper (see `init_raft`'s doc comment: only the lowest-ID node
    /// in the member set actually calls `initialize()`). Never cleared by
    /// `shutdown()`, so a later restart's `init_raft` call is recognized as
    /// a rejoin, not a first attempt, and skips `initialize()` even if this
    /// node happens to be the bootstrapper — re-bootstrapping on restart
    /// would be actively harmful (see `init_raft`'s doc comment for why).
    #[cfg(feature = "raft")]
    bootstrap_attempted: AtomicBool,
    /// This node's own bind address for the Raft RPC listener. Set via
    /// [`RaftNode::set_network`] before `init_raft` for a multi-node peer
    /// set; left `None` for genuine single-node deployments that never call
    /// `set_network`.
    #[cfg(feature = "raft")]
    bind_addr: Option<SocketAddr>,
    /// Known network addresses of other cluster members, set via
    /// [`RaftNode::set_network`]. Snapshotted into a fresh
    /// `Arc<RwLock<_>>` for the network factory each time `init_raft`
    /// constructs a new `Raft` instance.
    #[cfg(feature = "raft")]
    peer_addresses: HashMap<OxirsNodeId, SocketAddr>,
    /// Handle to the spawned Raft RPC accept-loop task
    /// (`raft_network::serve_raft_rpc`), so `shutdown` can abort it and free
    /// the listening port for a later restart to rebind.
    #[cfg(feature = "raft")]
    listener_task: Option<tokio::task::JoinHandle<()>>,
    storage: Arc<RwLock<RdfApp>>,
}

/// Outcome of one `raft.initialize()` attempt made while bootstrapping a
/// cluster in [`RaftNode::init_raft`], as classified by
/// [`classify_bootstrap_attempt`].
#[cfg(feature = "raft")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BootstrapDecision {
    /// The retry loop is over: either `initialize()` succeeded, or the
    /// cluster was already initialized by a peer or an earlier attempt
    /// (`InitializeError::NotAllowed`). Either way, calling `initialize()`
    /// again on this node would now be unsafe (see the doc on
    /// `RaftNode::bootstrap_attempted`).
    Done,
    /// A genuine error (not "already initialized"), and attempts remain:
    /// worth retrying after a short backoff.
    Retryable,
    /// A genuine error and no attempts remain: the whole bootstrap has
    /// failed. The cluster was never actually formed, so a future
    /// `init_raft` call must remain free to try again —
    /// `bootstrap_attempted` must NOT be latched.
    Fatal,
}

#[cfg(feature = "raft")]
impl BootstrapDecision {
    /// Whether `RaftNode::bootstrap_attempted` should be latched to `true`
    /// after this decision. Only `Done` makes a future `initialize()` call
    /// on this node genuinely unsafe; `Fatal` means the cluster was never
    /// formed at all, so a future `init_raft` call must remain free to
    /// retry — latching there would permanently strand the node (the bug
    /// this type exists to prevent).
    fn should_latch_bootstrap_attempted(self) -> bool {
        matches!(self, Self::Done)
    }
}

/// Given the outcome of one `raft.initialize()` attempt made while
/// bootstrapping a cluster in [`RaftNode::init_raft`], decide whether the
/// retry loop is done, should keep retrying, or has failed fatally.
///
/// Takes plain `bool`s rather than the real `openraft` error type so this
/// stays pure and synchronous and can be unit-tested directly, without
/// needing to fabricate a real network failure:
/// - `succeeded`: the attempt returned `Ok(())`.
/// - `already_initialized`: the attempt failed with
///   `InitializeError::NotAllowed` — a peer (or an earlier attempt)
///   already initialized the cluster.
/// - `attempts_exhausted`: this was the last attempt the retry loop
///   allows.
#[cfg(feature = "raft")]
fn classify_bootstrap_attempt(
    succeeded: bool,
    already_initialized: bool,
    attempts_exhausted: bool,
) -> BootstrapDecision {
    if succeeded || already_initialized {
        BootstrapDecision::Done
    } else if attempts_exhausted {
        BootstrapDecision::Fatal
    } else {
        BootstrapDecision::Retryable
    }
}

impl RaftNode {
    pub fn new(node_id: OxirsNodeId) -> Self {
        Self {
            node_id,
            #[cfg(feature = "raft")]
            raft: None,
            #[cfg(feature = "raft")]
            multi_node_requested: AtomicBool::new(false),
            #[cfg(feature = "raft")]
            bootstrap_attempted: AtomicBool::new(false),
            #[cfg(feature = "raft")]
            bind_addr: None,
            #[cfg(feature = "raft")]
            peer_addresses: HashMap::new(),
            #[cfg(feature = "raft")]
            listener_task: None,
            storage: Arc::new(RwLock::new(RdfApp::default())),
        }
    }

    /// Configure this node's Raft network: its own bind address, and the
    /// known addresses of every other cluster member. Required before
    /// `init_raft` for any peer set that names a node other than `self` —
    /// without it, `init_raft` cannot build a `RaftNetworkFactory` (no
    /// address to listen on) or reach peers (no addresses to dial), and
    /// returns [`RaftClusterError::NetworkNotConfigured`] rather than
    /// silently falling back to fake single-node "leadership".
    #[cfg(feature = "raft")]
    pub fn set_network(
        &mut self,
        bind_addr: SocketAddr,
        peer_addresses: HashMap<OxirsNodeId, SocketAddr>,
    ) {
        self.bind_addr = Some(bind_addr);
        self.peer_addresses = peer_addresses;
    }

    /// Initialize Raft.
    ///
    /// Real multi-node consensus: a split log/state-machine storage
    /// (`OxirsStorage` wrapped in `openraft::storage::Adaptor`), a dedicated
    /// TCP `RaftNetworkFactory`/`RaftNetwork` transport
    /// (`raft_network::OxirsRaftNetworkFactory`, with its accept loop spawned
    /// via `raft_network::serve_raft_rpc`), and `openraft::Raft::new`.
    ///
    /// Behavior:
    /// - If `peers` names only `self` (or is empty), this is a genuine
    ///   single-node deployment. No `openraft::Raft` instance is
    ///   constructed — the lightweight in-process fallback storage
    ///   (`self.storage`, shared via `GLOBAL_SHARED_STORAGE` in tests) is an
    ///   honest implementation of a one-node "cluster", and real consensus
    ///   for a cluster of one would only add overhead without changing
    ///   observable behavior.
    /// - If `peers` names one or more other nodes, a real multi-node
    ///   instance is constructed as described above. This requires
    ///   [`RaftNode::set_network`] to have been called first; if this node's
    ///   bind address or any peer's address is unknown,
    ///   [`RaftClusterError::NetworkNotConfigured`] is returned instead of
    ///   silently falling back to a fake single-node "leader" (the exact
    ///   anti-pattern a prior build of this module refused to allow — see
    ///   git history — now made unnecessary by actually implementing the
    ///   real thing).
    /// - The very first time construction succeeds, the node whose id is the
    ///   *lowest* in the full member set (`self` plus `peers`) bootstraps the
    ///   cluster by calling `raft.initialize()` with that member set; every
    ///   other node just finishes constructing its `Raft` instance and waits
    ///   to be discovered/replicated to. OpenRaft's own docs for
    ///   `Raft::initialize` state that calling it from *every* node with the
    ///   same member set is safe (whichever call lands first wins, and
    ///   `InitializeError::NotAllowed` on the others just means the cluster
    ///   is already initialized), but that guarantee assumes no node's
    ///   own `elect()` — triggered by its own local `initialize()` — races
    ///   an *incoming* RPC from a peer that bootstrapped first and already
    ///   reached this node's listener; empirically, against openraft
    ///   0.9.24, that race can trip an internal
    ///   `debug_assert!(self.leader.is_none())` inside `following_handler()`.
    ///   Designating a single bootstrapper avoids the race entirely.
    ///   `initialize()` itself is purely local (it appends a membership
    ///   entry and starts an election; it does not require peers to be
    ///   reachable yet — replicating that entry to a quorum happens
    ///   afterwards, via openraft's normal, auto-retrying replication), but
    ///   the call is still retried a few times with a short backoff on any
    ///   other error, in case it races a transient local condition.
    /// - On every *later* call to `init_raft` (i.e. a restart after
    ///   `shutdown()`), `initialize()` is deliberately **not** called again,
    ///   even though this node's storage is rebuilt empty and therefore
    ///   looks "pristine" to openraft (which would otherwise happily accept
    ///   a second bootstrap attempt). Re-bootstrapping here would let this
    ///   node start a new, doomed candidacy — its RequestVote can carry a
    ///   higher term than the real leader's even though its log is behind,
    ///   and any node that observes a higher term must revert to follower —
    ///   which can force a perfectly healthy cluster's real leader to step
    ///   down purely because this one node's local state was reset. Instead
    ///   the node just reconstructs its `Raft` instance and rejoins as an
    ///   ordinary member; the existing leader still lists it in the cluster
    ///   membership and will replicate it back up to date automatically once
    ///   it is reachable again.
    #[cfg(feature = "raft")]
    pub async fn init_raft(&mut self, peers: BTreeSet<OxirsNodeId>) -> Result<()> {
        let other_peers: BTreeSet<OxirsNodeId> = peers
            .into_iter()
            .filter(|&peer| peer != self.node_id)
            .collect();

        if other_peers.is_empty() {
            tracing::info!(
                node_id = self.node_id,
                "Initializing Raft in single-node mode (no other peers requested)"
            );
            return Ok(());
        }

        self.multi_node_requested.store(true, Ordering::SeqCst);

        let Some(bind_addr) = self.bind_addr else {
            return Err(RaftClusterError::NetworkNotConfigured {
                node_id: self.node_id,
                peer_count: other_peers.len(),
                detail: "no bind address set; call set_network() first".to_string(),
            }
            .into());
        };

        // Build the initial member set while validating that every other
        // peer has a known address (one pass does both).
        let mut members: BTreeMap<OxirsNodeId, BasicNode> = BTreeMap::new();
        members.insert(self.node_id, BasicNode::new(bind_addr.to_string()));
        let mut missing_addresses: Vec<OxirsNodeId> = Vec::new();
        for &peer in &other_peers {
            match self.peer_addresses.get(&peer) {
                Some(&addr) => {
                    members.insert(peer, BasicNode::new(addr.to_string()));
                }
                None => missing_addresses.push(peer),
            }
        }
        if !missing_addresses.is_empty() {
            return Err(RaftClusterError::NetworkNotConfigured {
                node_id: self.node_id,
                peer_count: other_peers.len(),
                detail: format!("no address known for peer(s) {missing_addresses:?}"),
            }
            .into());
        }

        // Deliberately more generous than openraft's own defaults
        // (election_timeout 150-300ms, heartbeat 50ms), which are tuned for
        // a low-jitter datacenter LAN. This transport runs real TCP
        // round trips on a shared dev machine that can see many concurrent
        // unrelated cargo/test processes and CPU load averages far above
        // its core count (see project notes); under that kind of scheduling
        // jitter, aggressive timeouts cause spurious "leader unreachable"
        // elections that never let the term settle (a real liveness
        // problem, empirically observed as `ForwardToLeader` errors that
        // persisted across a 10s retry budget). Wider margins trade a bit
        // of failover latency for actually converging.
        let config = Arc::new(
            openraft::Config {
                cluster_name: "oxirs-cluster".to_string(),
                election_timeout_min: 500,
                election_timeout_max: 1000,
                heartbeat_interval: 150,
                ..Default::default()
            }
            .validate()
            .map_err(|e| anyhow::anyhow!("invalid raft config for node {}: {e}", self.node_id))?,
        );

        let store = OxirsStorage::new();
        // Clone the state Arc *before* handing `store` to `Adaptor::new` by
        // value (`OxirsStorage::clone` clones the inner Arcs, so this and the
        // copy inside the adaptor end up sharing the same underlying
        // `RdfApp`). Committed to `self.storage` only after everything below
        // succeeds — see the comment further down.
        let state_arc = Arc::clone(&store.state);
        let (log_store, state_machine) = openraft::storage::Adaptor::new(store);

        let peer_addresses = Arc::new(tokio::sync::RwLock::new(self.peer_addresses.clone()));
        let network =
            crate::raft_network::OxirsRaftNetworkFactory::new(self.node_id, peer_addresses);

        let raft = Raft::new(self.node_id, config, network, log_store, state_machine)
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "node {} failed to construct raft instance: {e}",
                    self.node_id
                )
            })?;

        let listener = tokio::net::TcpListener::bind(bind_addr)
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "node {} failed to bind raft RPC listener on {bind_addr}: {e}",
                    self.node_id
                )
            })?;
        let raft_for_listener = raft.clone();
        let listener_task = tokio::spawn(async move {
            crate::raft_network::serve_raft_rpc(listener, raft_for_listener).await;
        });

        // Construction fully succeeded: commit it to `self`. Read paths
        // (`len`/`query`/`is_empty` below) key off `self.raft.is_some()` to
        // decide whether to read `self.storage` (this node's own applied
        // state) instead of the test-only global fallback, so `self.storage`
        // must not be overwritten before we know a real Raft instance is
        // actually backing it.
        self.storage = state_arc;
        self.listener_task = Some(listener_task);
        self.raft = Some(raft);

        // Only the lowest-ID node in the full member set calls `initialize()`
        // (and only on this node's first-ever `init_raft` call — see
        // `bootstrap_attempted`'s field doc). OpenRaft's own docs for
        // `Raft::initialize` state that calling it from *every* node with
        // the same member set is safe, but that assumes no node's `elect()`
        // (triggered by its own local `initialize()`) races an *incoming*
        // RPC from a peer that bootstrapped first: a faster peer's
        // in-flight RequestVote/AppendEntries can reach this node's already
        // -bound listener before this node's own local `initialize()` call
        // is processed, and `following_handler()`'s
        // `debug_assert!(self.leader.is_none())` can trip when that
        // happens (confirmed empirically against openraft 0.9.24 — see git
        // history for the failure). Designating a single bootstrapper
        // sidesteps the race entirely: every other node just constructs its
        // `Raft` instance and waits to be discovered/replicated to, exactly
        // like the restart/rejoin path below.
        let is_bootstrap_node = members
            .keys()
            .next()
            .is_some_and(|&lowest_id| lowest_id == self.node_id);

        // Read-only check here: whether this attempt should latch
        // `bootstrap_attempted` is decided below, per-outcome, by
        // `classify_bootstrap_attempt` — never unconditionally on entry.
        // Latching unconditionally would strand the node forever the first
        // time every retry below fails for a genuine (non-"already
        // initialized") reason: a future `init_raft` call (e.g. after a
        // later `stop()`/`start()` cycle) would then skip `initialize()`
        // forever, with no recovery path short of recreating the whole
        // `RaftNode`.
        if is_bootstrap_node && !self.bootstrap_attempted.load(Ordering::SeqCst) {
            let raft_ref = self.raft.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "node {} raft instance vanished immediately after construction",
                    self.node_id
                )
            })?;

            const MAX_ATTEMPTS: u32 = 5;
            let mut last_err = None;
            let mut should_latch = false;
            for attempt in 1..=MAX_ATTEMPTS {
                let result = raft_ref.initialize(members.clone()).await;
                let already_initialized = result.as_ref().err().is_some_and(|e| {
                    matches!(
                        e.api_error(),
                        Some(openraft::error::InitializeError::NotAllowed(_))
                    )
                });
                let decision = classify_bootstrap_attempt(
                    result.is_ok(),
                    already_initialized,
                    attempt >= MAX_ATTEMPTS,
                );
                should_latch = decision.should_latch_bootstrap_attempted();

                match decision {
                    BootstrapDecision::Done => {
                        if already_initialized {
                            tracing::info!(
                                node_id = self.node_id,
                                "raft cluster already initialized (by a peer's or a prior attempt's initialize() call)"
                            );
                        } else {
                            tracing::info!(
                                node_id = self.node_id,
                                member_count = members.len(),
                                "bootstrapped raft cluster"
                            );
                        }
                        last_err = None;
                        break;
                    }
                    BootstrapDecision::Fatal => {
                        last_err = result.err();
                        break;
                    }
                    BootstrapDecision::Retryable => {
                        if let Some(e) = result.as_ref().err() {
                            tracing::warn!(
                                node_id = self.node_id,
                                attempt,
                                error = %e,
                                "raft initialize() did not succeed yet, retrying shortly"
                            );
                        }
                        tokio::time::sleep(Duration::from_millis(200)).await;
                    }
                }
            }

            if should_latch {
                self.bootstrap_attempted.store(true, Ordering::SeqCst);
            }

            if let Some(e) = last_err {
                // Every attempt failed for a genuine reason (not "already
                // initialized"): the cluster was never actually formed,
                // and — because this node is the sole designated
                // bootstrapper for this member set (see above) — never
                // will be from this attempt either. `self.raft`/
                // `self.listener_task` were already committed above, but a
                // raft instance that never completed `initialize()` is not
                // part of any cluster and is not going to become part of
                // one on its own. Leaving it in place would: (a)
                // permanently wedge a future retry's `TcpListener::bind`
                // on this now-already-bound address (see `shutdown()`'s
                // doc on why the port must be freed), and (b) let
                // `has_running_raft()` — and therefore
                // `submit_command`/`query`/`len`/`is_empty` — see
                // `self.raft.is_some()` and treat this node as backed by a
                // live consensus instance it does not actually have,
                // inconsistent with the `Err` this call is about to
                // return. Tear both down via the same primitive
                // `shutdown()` uses, so this failure leaves `self` in
                // exactly the state any *other* `init_raft` failure (e.g.
                // a construction or bind failure above) already leaves it
                // in: `raft`/`listener_task` both `None`, safe to retry on
                // a future `init_raft` call.
                if let Err(shutdown_err) = self.shutdown().await {
                    tracing::warn!(
                        node_id = self.node_id,
                        error = %shutdown_err,
                        "failed to cleanly tear down raft instance after bootstrap failure"
                    );
                }
                return Err(anyhow::anyhow!(
                    "node {} failed to bootstrap raft cluster after {MAX_ATTEMPTS} attempt(s): {e}",
                    self.node_id
                ));
            }
        }

        Ok(())
    }

    /// Check if this node is the leader.
    ///
    /// Only honest when either real Raft consensus elected this node, or
    /// no multi-node cluster was ever requested (a genuine single-node
    /// deployment trivially leads itself). If a multi-node cluster was
    /// requested but real consensus could not be constructed, this
    /// deliberately returns `false` instead of masquerading as a leader.
    pub async fn is_leader(&self) -> bool {
        #[cfg(feature = "raft")]
        {
            if let Some(ref raft) = self.raft {
                match raft.metrics().borrow().current_leader {
                    Some(leader) => leader == self.node_id,
                    None => false,
                }
            } else if self.multi_node_requested.load(Ordering::SeqCst) {
                // Multi-node was requested but no Raft instance is currently
                // running (construction failed, or the node has since been
                // `shutdown()`): honestly not a leader.
                false
            } else {
                // No multi-node cluster was ever requested: honest single-node mode.
                true
            }
        }
        #[cfg(not(feature = "raft"))]
        {
            // The "raft" feature is not compiled in at all, so no cluster
            // capability is claimed; single-node behavior is honest here.
            true
        }
    }

    /// Get current term
    pub async fn current_term(&self) -> u64 {
        #[cfg(feature = "raft")]
        {
            if let Some(ref raft) = self.raft {
                raft.metrics().borrow().current_term
            } else {
                0
            }
        }
        #[cfg(not(feature = "raft"))]
        {
            0
        }
    }

    /// Submit a command for replication.
    ///
    /// If a multi-node cluster was requested via `init_raft` but no Raft
    /// instance is currently running (construction failed, or the node has
    /// since been shut down), this returns
    /// [`RaftClusterError::ConsensusUnavailable`] rather than silently
    /// applying the command to local-only fallback storage — a write that
    /// is never replicated to any peer must not be reported as successful.
    pub async fn submit_command(&self, cmd: RdfCommand) -> Result<RdfResponse> {
        #[cfg(feature = "raft")]
        {
            if let Some(ref raft) = self.raft {
                let response = raft
                    .client_write(cmd)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to submit command: {}", e))?;
                Ok(response.data)
            } else if self.multi_node_requested.load(Ordering::SeqCst) {
                Err(RaftClusterError::ConsensusUnavailable {
                    node_id: self.node_id,
                }
                .into())
            } else {
                // Genuine single-node mode: fallback storage honestly
                // represents the only node in the "cluster". Global shared
                // storage exists so multiple in-process RaftNode instances
                // can be exercised together in tests.
                if let Some(shared_storage) = get_global_shared_storage() {
                    let mut state = shared_storage.write().await;
                    Ok(state.apply_command(&cmd))
                } else {
                    let mut state = self.storage.write().await;
                    Ok(state.apply_command(&cmd))
                }
            }
        }
        #[cfg(not(feature = "raft"))]
        {
            // Use global shared storage for testing
            if let Some(shared_storage) = get_global_shared_storage() {
                let mut state = shared_storage.write().await;
                Ok(state.apply_command(&cmd))
            } else {
                let mut state = self.storage.write().await;
                Ok(state.apply_command(&cmd))
            }
        }
    }

    /// Get metrics
    #[cfg(feature = "raft")]
    pub async fn get_metrics(&self) -> Option<RaftMetrics<OxirsNodeId, BasicNode>> {
        self.raft
            .as_ref()
            .map(|raft| raft.metrics().borrow().clone())
    }

    /// This node currently has a real, running multi-node Raft instance —
    /// i.e. `self.storage` is kept in sync with (shares the same `Arc` as)
    /// that instance's own applied state machine (see `init_raft`). Reads
    /// must go through it directly rather than the process-wide
    /// `GLOBAL_SHARED_STORAGE` test fallback below: that global is a trick
    /// so multiple in-process `RaftNode`s can share state in the *simulated*
    /// (non-raft, or single-node) path, and would make every real node's
    /// read return the same answer regardless of actual per-node
    /// replication — defeating the entire point of testing replication.
    #[cfg(feature = "raft")]
    fn has_running_raft(&self) -> bool {
        self.raft.is_some()
    }

    #[cfg(not(feature = "raft"))]
    fn has_running_raft(&self) -> bool {
        false
    }

    /// Query the local state machine
    pub async fn query(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<(String, String, String)> {
        if self.has_running_raft() {
            let state = self.storage.read().await;
            return state.query(subject, predicate, object);
        }
        if let Some(shared_storage) = get_global_shared_storage() {
            let state = shared_storage.read().await;
            state.query(subject, predicate, object)
        } else {
            let state = self.storage.read().await;
            state.query(subject, predicate, object)
        }
    }

    /// Get number of triples
    pub async fn len(&self) -> usize {
        if self.has_running_raft() {
            let state = self.storage.read().await;
            return state.len();
        }
        if let Some(shared_storage) = get_global_shared_storage() {
            let state = shared_storage.read().await;
            state.len()
        } else {
            let state = self.storage.read().await;
            state.len()
        }
    }

    /// Check if store is empty
    pub async fn is_empty(&self) -> bool {
        if self.has_running_raft() {
            let state = self.storage.read().await;
            return state.is_empty();
        }
        if let Some(shared_storage) = get_global_shared_storage() {
            let state = shared_storage.read().await;
            state.is_empty()
        } else {
            let state = self.storage.read().await;
            state.is_empty()
        }
    }

    /// Shutdown the raft node gracefully.
    ///
    /// Abrupt from the cluster's point of view (no leadership transfer is
    /// attempted here — see `ConsensusManager::graceful_shutdown` for that);
    /// this is the primitive both a real graceful shutdown and a simulated
    /// node crash/stop (`ClusterNode::stop`) build on. Frees the Raft RPC
    /// listener's port so a later `init_raft` call (e.g. after `stop()` then
    /// `start()`) can rebind and rejoin the cluster.
    pub async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("Shutting down raft node {}", self.node_id);

        #[cfg(feature = "raft")]
        {
            if let Some(raft) = self.raft.take() {
                // Shutdown the raft instance itself first, so its core loop
                // stops sending heartbeats/replicating (letting peers notice
                // this node is gone and elect a new leader if it was one).
                raft.shutdown()
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to shutdown raft: {}", e))?;
                tracing::info!("Raft instance shutdown completed");
            }
            if let Some(listener_task) = self.listener_task.take() {
                // Abort and await the accept-loop task so the listening port
                // is actually released before this call returns — otherwise
                // a subsequent `init_raft`'s `TcpListener::bind` on the same
                // address could race the still-shutting-down old listener
                // and fail with "address already in use".
                listener_task.abort();
                let _ = listener_task.await;
            }
        }

        // Clear storage reference
        {
            let mut storage = self.storage.write().await;
            *storage = RdfApp::default();
        }

        tracing::info!("Raft node {} shutdown completed", self.node_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdf_command_serialization() {
        let cmd = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };

        let serialized = serde_json::to_string(&cmd).unwrap();
        let deserialized: RdfCommand = serde_json::from_str(&serialized).unwrap();

        assert_eq!(cmd, deserialized);
    }

    #[test]
    fn test_rdf_response_serialization() {
        let response = RdfResponse::Success;

        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: RdfResponse = serde_json::from_str(&serialized).unwrap();

        assert_eq!(response, deserialized);
    }

    #[test]
    fn test_rdf_app_apply_insert() {
        let mut app = RdfApp::default();

        let cmd = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };

        let response = app.apply_command(&cmd);
        assert_eq!(response, RdfResponse::Success);
        assert_eq!(app.len(), 1);
        assert!(!app.is_empty());
    }

    #[test]
    fn test_rdf_app_apply_delete() {
        let mut app = RdfApp::default();

        // Insert first
        let insert_cmd = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        app.apply_command(&insert_cmd);
        assert_eq!(app.len(), 1);

        // Then delete
        let delete_cmd = RdfCommand::Delete {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        let response = app.apply_command(&delete_cmd);
        assert_eq!(response, RdfResponse::Success);
        assert_eq!(app.len(), 0);
        assert!(app.is_empty());
    }

    #[test]
    fn test_rdf_app_apply_clear() {
        let mut app = RdfApp::default();

        // Insert some data
        app.apply_command(&RdfCommand::Insert {
            subject: "s1".to_string(),
            predicate: "p1".to_string(),
            object: "o1".to_string(),
        });
        app.apply_command(&RdfCommand::Insert {
            subject: "s2".to_string(),
            predicate: "p2".to_string(),
            object: "o2".to_string(),
        });
        assert_eq!(app.len(), 2);

        // Clear all
        let response = app.apply_command(&RdfCommand::Clear);
        assert_eq!(response, RdfResponse::Success);
        assert_eq!(app.len(), 0);
        assert!(app.is_empty());
    }

    #[test]
    fn test_rdf_app_transactions() {
        let mut app = RdfApp::default();
        let tx_id = "tx1".to_string();

        // Begin transaction
        let response = app.apply_command(&RdfCommand::BeginTransaction {
            tx_id: tx_id.clone(),
        });
        assert_eq!(
            response,
            RdfResponse::TransactionStarted {
                tx_id: tx_id.clone()
            }
        );
        assert!(app.transactions.contains_key(&tx_id));

        // Commit transaction
        let response = app.apply_command(&RdfCommand::CommitTransaction {
            tx_id: tx_id.clone(),
        });
        assert_eq!(
            response,
            RdfResponse::TransactionCommitted {
                tx_id: tx_id.clone()
            }
        );
        assert!(!app.transactions.contains_key(&tx_id));
    }

    #[test]
    fn test_rdf_app_transaction_rollback() {
        let mut app = RdfApp::default();
        let tx_id = "tx1".to_string();

        // Begin transaction
        app.apply_command(&RdfCommand::BeginTransaction {
            tx_id: tx_id.clone(),
        });
        assert!(app.transactions.contains_key(&tx_id));

        // Rollback transaction
        let response = app.apply_command(&RdfCommand::RollbackTransaction {
            tx_id: tx_id.clone(),
        });
        assert_eq!(
            response,
            RdfResponse::TransactionRolledBack {
                tx_id: tx_id.clone()
            }
        );
        assert!(!app.transactions.contains_key(&tx_id));
    }

    #[test]
    fn test_rdf_app_query() {
        let mut app = RdfApp::default();

        // Insert test data
        app.apply_command(&RdfCommand::Insert {
            subject: "s1".to_string(),
            predicate: "p1".to_string(),
            object: "o1".to_string(),
        });
        app.apply_command(&RdfCommand::Insert {
            subject: "s1".to_string(),
            predicate: "p2".to_string(),
            object: "o2".to_string(),
        });
        app.apply_command(&RdfCommand::Insert {
            subject: "s2".to_string(),
            predicate: "p1".to_string(),
            object: "o3".to_string(),
        });

        // Query all triples
        let results = app.query(None, None, None);
        assert_eq!(results.len(), 3);

        // Query by subject
        let results = app.query(Some("s1"), None, None);
        assert_eq!(results.len(), 2);

        // Query by predicate
        let results = app.query(None, Some("p1"), None);
        assert_eq!(results.len(), 2);

        // Query by object
        let results = app.query(None, None, Some("o1"));
        assert_eq!(results.len(), 1);

        // Query specific triple
        let results = app.query(Some("s1"), Some("p1"), Some("o1"));
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0],
            ("s1".to_string(), "p1".to_string(), "o1".to_string())
        );
    }

    #[tokio::test]
    async fn test_raft_node_creation() {
        let node = RaftNode::new(1);
        assert_eq!(node.node_id, 1);
        // Uninitialized nodes act as leaders in single-node mode
        assert!(node.is_leader().await);
        assert_eq!(node.current_term().await, 0);
        assert_eq!(node.len().await, 0);
        assert!(node.is_empty().await);
    }

    #[tokio::test]
    async fn test_raft_node_local_operations() {
        let node = RaftNode::new(1);

        // Test insert command without Raft
        let cmd = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };

        let response = node.submit_command(cmd).await.unwrap();
        assert_eq!(response, RdfResponse::Success);
        assert_eq!(node.len().await, 1);
        assert!(!node.is_empty().await);

        // Test query
        let results = node.query(Some("s"), Some("p"), Some("o")).await;
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0],
            ("s".to_string(), "p".to_string(), "o".to_string())
        );
    }

    /// `init_raft` must fail loudly (not warn+Ok, and not silently fall back
    /// to fake single-node "leadership") when a real multi-node cluster is
    /// requested without first configuring the network via `set_network` —
    /// there is no bind address to build a `RaftNetworkFactory`/listener
    /// with. Real multi-node consensus *is* implemented (see
    /// `test_multi_node_raft_elects_leader_and_replicates` below); this
    /// covers the still-real precondition failure when setup is incomplete.
    #[cfg(feature = "raft")]
    #[tokio::test]
    async fn test_init_raft_multi_node_without_network_config_fails_loudly() {
        let mut node = RaftNode::new(1);
        let peers: BTreeSet<OxirsNodeId> = [1u64, 2, 3].into_iter().collect();

        let result = node.init_raft(peers).await;
        assert!(
            result.is_err(),
            "requesting a multi-node cluster with no network config must fail, not silently succeed"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("multi-node")
                || err.to_string().contains("network configuration"),
            "error message should clearly explain the missing network configuration: {err}"
        );
    }

    /// After a multi-node `init_raft()` fails (e.g. missing network config),
    /// the node must never claim leadership or accept writes that pretend to
    /// be replicated.
    #[cfg(feature = "raft")]
    #[tokio::test]
    async fn test_node_after_failed_multi_node_init_is_not_a_fake_leader() {
        let mut node = RaftNode::new(42);
        let peers: BTreeSet<OxirsNodeId> = [42u64, 7, 9].into_iter().collect();

        let init_result = node.init_raft(peers).await;
        assert!(init_result.is_err());

        // The node must not silently masquerade as a single-node leader.
        assert!(
            !node.is_leader().await,
            "node must not claim leadership after multi-node Raft init failed"
        );

        // Writes must fail loudly instead of being applied to unreplicated
        // local-only fallback storage.
        let cmd = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        let submit_result = node.submit_command(cmd).await;
        assert!(
            submit_result.is_err(),
            "submit_command must fail rather than fabricate a successful unreplicated write"
        );
    }

    /// Bind three real `RaftNode`s to loopback TCP ports, wire them into one
    /// cluster via `set_network` + `init_raft`, and verify actual OpenRaft
    /// consensus: a leader is elected, a command submitted through it is
    /// really replicated (not just applied to a global test fallback) to
    /// every node's own independently-tracked applied state, and each
    /// node's `is_leader()`/`current_term()` reflect genuine per-node raft
    /// metrics.
    #[cfg(feature = "raft")]
    #[tokio::test]
    async fn test_multi_node_raft_elects_leader_and_replicates() {
        use std::net::TcpListener as StdTcpListener;

        // Reserve 3 free loopback ports synchronously (bind then immediately
        // drop) so every node's address is known before any node starts —
        // real multi-node `init_raft` requires the full peer address map
        // upfront, mirroring how `TestCluster` in the integration tests
        // derives its addresses.
        let free_addr = || {
            let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
            listener.local_addr().expect("local_addr")
        };
        let addrs: BTreeMap<OxirsNodeId, std::net::SocketAddr> = [
            (1u64, free_addr()),
            (2u64, free_addr()),
            (3u64, free_addr()),
        ]
        .into_iter()
        .collect();

        let mut nodes: Vec<RaftNode> = Vec::new();
        for (&id, &addr) in &addrs {
            let mut node = RaftNode::new(id);
            let peers: HashMap<OxirsNodeId, SocketAddr> = addrs
                .iter()
                .filter(|(&peer_id, _)| peer_id != id)
                .map(|(&peer_id, &peer_addr)| (peer_id, peer_addr))
                .collect();
            node.set_network(addr, peers);
            nodes.push(node);
        }

        let all_ids: BTreeSet<OxirsNodeId> = addrs.keys().copied().collect();
        for node in &mut nodes {
            node.init_raft(all_ids.clone())
                .await
                .expect("multi-node init_raft with full network config must succeed");
        }

        // Poll for a leader to emerge (real election takes a nonzero amount
        // of time after initialize()).
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let mut leader_idx = None;
        while tokio::time::Instant::now() < deadline {
            for (idx, node) in nodes.iter().enumerate() {
                if node.is_leader().await {
                    leader_idx = Some(idx);
                    break;
                }
            }
            if leader_idx.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let leader_idx = leader_idx.expect("a leader must be elected within 5s");
        assert!(
            nodes[leader_idx].current_term().await >= 1,
            "an elected leader must have a real (non-zero) term"
        );

        // Submit a command through the real leader and verify it lands on
        // every node's own applied state (not a shared global fallback).
        let cmd = RdfCommand::Insert {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "\"o\"".to_string(),
        };
        let response = nodes[leader_idx]
            .submit_command(cmd)
            .await
            .expect("leader submit_command must succeed");
        assert_eq!(response, RdfResponse::Success);

        for (idx, node) in nodes.iter().enumerate() {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
            let mut count = node.len().await;
            while count != 1 && tokio::time::Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(50)).await;
                count = node.len().await;
            }
            assert_eq!(
                count, 1,
                "node index {idx} did not replicate the committed entry"
            );
        }

        for node in &mut nodes {
            node.shutdown().await.expect("shutdown must succeed");
        }
    }

    /// A genuine single-node deployment (empty peer set, or a peer set
    /// containing only self) must continue to work as an honest one-node
    /// "cluster": init succeeds and the node is its own leader.
    #[cfg(feature = "raft")]
    #[tokio::test]
    async fn test_init_raft_single_node_mode_stays_leader() {
        let mut node = RaftNode::new(5);

        // Explicitly listing only self is equivalent to no peers.
        let peers: BTreeSet<OxirsNodeId> = [5u64].into_iter().collect();
        let result = node.init_raft(peers).await;
        assert!(result.is_ok(), "single-node init must succeed: {result:?}");
        assert!(node.is_leader().await);

        let cmd = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        let response = node.submit_command(cmd).await;
        assert!(
            response.is_ok(),
            "single-node writes must still succeed: {response:?}"
        );
    }

    /// A successful `initialize()` (or one that failed only because a peer
    /// / an earlier attempt already initialized the cluster) is the only
    /// case where re-`initialize()`-ing this node in the future would be
    /// unsafe: `bootstrap_attempted` must be latched, and the retry loop
    /// must stop.
    #[cfg(feature = "raft")]
    #[test]
    fn test_classify_bootstrap_attempt_success_and_already_initialized_are_done_and_latch() {
        let succeeded = classify_bootstrap_attempt(true, false, false);
        assert_eq!(succeeded, BootstrapDecision::Done);
        assert!(succeeded.should_latch_bootstrap_attempted());

        // Whether attempts remain must not matter once we succeeded.
        let succeeded_last_attempt = classify_bootstrap_attempt(true, false, true);
        assert_eq!(succeeded_last_attempt, BootstrapDecision::Done);
        assert!(succeeded_last_attempt.should_latch_bootstrap_attempted());

        let already_initialized = classify_bootstrap_attempt(false, true, false);
        assert_eq!(already_initialized, BootstrapDecision::Done);
        assert!(already_initialized.should_latch_bootstrap_attempted());
    }

    /// This is the regression case for the bootstrap-latch bug: a genuine
    /// failure (not "already initialized") must be retried while attempts
    /// remain, and once every attempt is exhausted, must be reported as
    /// `Fatal` *without* latching `bootstrap_attempted` — latching here is
    /// exactly what permanently stranded a `RaftNode` on a transient
    /// failure before this fix.
    #[cfg(feature = "raft")]
    #[test]
    fn test_classify_bootstrap_attempt_genuine_failure_after_retries_does_not_latch() {
        let mid_retry = classify_bootstrap_attempt(false, false, false);
        assert_eq!(mid_retry, BootstrapDecision::Retryable);
        assert!(!mid_retry.should_latch_bootstrap_attempted());

        let exhausted = classify_bootstrap_attempt(false, false, true);
        assert_eq!(exhausted, BootstrapDecision::Fatal);
        assert!(
            !exhausted.should_latch_bootstrap_attempted(),
            "a genuine failure must leave bootstrap_attempted unlatched so a future \
             init_raft() call can retry"
        );
    }
}
