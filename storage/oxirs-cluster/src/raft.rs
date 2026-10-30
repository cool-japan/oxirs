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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

/// `triple` with its blank nodes in the form `BlankNode::new` gives labels
/// without the `_:` prefix. The N-Triples parser keeps the prefix in the
/// label, and oxirs-core compares labels literally, so `_:b1` read back from
/// a shard would not equal the `b1` that was written.
fn canonical_blank_nodes(triple: oxirs_core::model::Triple) -> oxirs_core::model::Triple {
    use oxirs_core::model::{BlankNode, Object, Subject, Triple};

    let canonical =
        |node: &BlankNode| BlankNode::new(node.local_id()).unwrap_or_else(|_| node.clone());
    let subject = match triple.subject() {
        Subject::BlankNode(node) => Subject::BlankNode(canonical(node)),
        other => other.clone(),
    };
    let object = match triple.object() {
        Object::BlankNode(node) => Object::BlankNode(canonical(node)),
        other => other.clone(),
    };
    Triple::new(subject, triple.predicate().clone(), object)
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
        use oxirs_core::model::{Object, Subject};
        use oxirs_core::parser::{Parser, RdfFormat};
        use oxirs_core::RdfTerm;

        let Some(shard) = self.shards.get(&shard_id) else {
            return Vec::new();
        };
        // Terms are stored in N-Triples syntax (their `Display` form), so a
        // stored triple reads back as one N-Triples line.
        let parser = Parser::new(RdfFormat::NTriples);
        shard
            .iter()
            .filter_map(
                |(s, p, o)| match parser.parse_ntriples_line(&format!("{s} {p} {o} .")) {
                    Ok(Some(quad)) => Some(canonical_blank_nodes(quad.to_triple())),
                    Ok(None) | Err(_) => {
                        tracing::warn!(
                            "shard {shard_id} holds a triple that does not parse: {s} {p} {o}"
                        );
                        None
                    }
                },
            )
            .filter(|triple| {
                // IRIs are matched without angle brackets, other terms in
                // N-Triples syntax, as by the other storage backends.
                let subject_matches = subject.map_or(true, |wanted| match triple.subject() {
                    Subject::NamedNode(node) => node.as_str() == wanted,
                    other => other.to_string() == wanted,
                });
                let predicate_matches =
                    predicate.map_or(true, |wanted| triple.predicate().as_str() == wanted);
                let object_matches = object.map_or(true, |wanted| match triple.object() {
                    Object::NamedNode(node) => node.as_str() == wanted,
                    other => other.to_string() == wanted,
                });
                subject_matches && predicate_matches && object_matches
            })
            .collect()
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

/// How many applied log entries may accumulate before the durable
/// state-machine checkpoint (`state_machine.bin`) is rewritten.
///
/// The checkpoint is **not** the source of truth — the fsync'd durable Raft
/// log is — so it only needs to be rewritten often enough to bound how much of
/// the log a restart must replay (openraft re-applies committed entries past
/// the persisted `last_applied` on startup). Rewriting the whole,
/// monotonically-growing state machine on *every* apply was O(n^2) over a long
/// write burst and forced one extra fsync of an ever-larger file per commit;
/// checkpointing every `STATE_MACHINE_CHECKPOINT_INTERVAL` applies keeps the
/// hot commit path O(1) amortized while capping restart replay at this many
/// entries. `build_snapshot` additionally forces a checkpoint, so the on-disk
/// `last_applied` always covers any subsequent `purge_logs_upto` (purge safety).
#[cfg(feature = "raft")]
pub(crate) const STATE_MACHINE_CHECKPOINT_INTERVAL: u64 = 64;

#[cfg(feature = "raft")]
mod raft_impl;

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
    /// The exact `Arc<RwLock<_>>` peer-address map handed to the running
    /// `OxirsRaftNetworkFactory` in `init_raft`. Kept so dynamic membership
    /// changes (`add_node`) can register a newly-joined node's address into
    /// the *live* factory map, otherwise the leader's outbound RaftNetwork
    /// would have no address to dial the new voter at and replication to it
    /// would never start. `None` until a multi-node `init_raft` succeeds.
    #[cfg(feature = "raft")]
    raft_peer_addresses: Option<Arc<RwLock<HashMap<OxirsNodeId, SocketAddr>>>>,
    /// Directory under which durable Raft state (log/vote/committed/snapshot/
    /// state machine) is persisted. When set, `init_raft` builds an
    /// `OxirsStorage` backed by `<data_dir>/raft/` and reloads persisted state
    /// on startup, so committed writes and the persisted vote survive process
    /// restarts. When `None`, storage is in-memory only (single-node / tests).
    #[cfg(feature = "raft")]
    data_dir: Option<std::path::PathBuf>,
    /// TLS for the Raft RPC transport, set via [`RaftNode::set_tls`]. When
    /// present, every Raft RPC between nodes runs over mutually
    /// authenticated TLS; when absent, over plain TCP.
    #[cfg(feature = "raft")]
    tls: Option<Arc<crate::tls::TlsManager>>,
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
            #[cfg(feature = "raft")]
            raft_peer_addresses: None,
            #[cfg(feature = "raft")]
            data_dir: None,
            #[cfg(feature = "raft")]
            tls: None,
            storage: Arc::new(RwLock::new(RdfApp::default())),
        }
    }

    /// Configure the directory used to durably persist this node's Raft state.
    /// Must be called before `init_raft` to take effect. An empty path is
    /// treated as "no durable directory" (in-memory storage).
    #[cfg(feature = "raft")]
    pub fn set_data_dir(&mut self, dir: std::path::PathBuf) {
        self.data_dir = if dir.as_os_str().is_empty() {
            None
        } else {
            Some(dir)
        };
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

    /// Run all Raft RPC traffic of this node over mutually authenticated TLS.
    ///
    /// Peers are authenticated as `node-{id}` by `tls`'s trust anchors, and
    /// the listener accepts only clients with a trusted certificate. `tls` is
    /// initialized by `init_raft`; if that fails, `init_raft` fails instead
    /// of falling back to plaintext. Must be called before `init_raft`.
    #[cfg(feature = "raft")]
    pub fn set_tls(&mut self, tls: Arc<crate::tls::TlsManager>) {
        self.tls = Some(tls);
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

        match &self.tls {
            Some(tls) => tls.initialize().await.map_err(|e| {
                anyhow::anyhow!(
                    "node {} is configured for TLS, but it could not be initialized; Raft is \
                     not started in plaintext instead: {e}",
                    self.node_id
                )
            })?,
            None => tracing::warn!(
                node_id = self.node_id,
                "Raft RPC traffic between nodes is neither encrypted nor authenticated; \
                 configure TLS for multi-node deployments"
            ),
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

        // Build durable storage when a data directory is configured, reloading
        // any persisted log/vote/committed/snapshot/state-machine so a restart
        // rejoins from disk rather than an empty in-memory Vec. Falls back to
        // in-memory storage for single-node / test deployments.
        let store = match self.data_dir.as_ref() {
            Some(dir) => OxirsStorage::new_with_dir(dir).map_err(|e| {
                anyhow::anyhow!(
                    "node {} failed to open durable raft storage at {}: {e}",
                    self.node_id,
                    dir.display()
                )
            })?,
            None => OxirsStorage::new(),
        };
        // Clone the state Arc *before* handing `store` to `Adaptor::new` by
        // value (`OxirsStorage::clone` clones the inner Arcs, so this and the
        // copy inside the adaptor end up sharing the same underlying
        // `RdfApp`). Committed to `self.storage` only after everything below
        // succeeds — see the comment further down.
        let state_arc = Arc::clone(&store.state);
        let (log_store, state_machine) = openraft::storage::Adaptor::new(store);

        let peer_addresses = Arc::new(tokio::sync::RwLock::new(self.peer_addresses.clone()));
        // Keep a handle to the *same* map the factory dials through, so a later
        // `add_node` can register a new voter's address into it live.
        let raft_peer_addresses = Arc::clone(&peer_addresses);
        let network = crate::raft_network::OxirsRaftNetworkFactory::new(
            self.node_id,
            peer_addresses,
            self.tls.clone(),
        );

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
        let tls_for_listener = self.tls.clone();
        let listener_task = tokio::spawn(async move {
            crate::raft_network::serve_raft_rpc(listener, raft_for_listener, tls_for_listener)
                .await;
        });

        // Construction fully succeeded: commit it to `self`. Read paths
        // (`len`/`query`/`is_empty` below) key off `self.raft.is_some()` to
        // decide whether to read `self.storage` (this node's own applied
        // state) instead of the test-only global fallback, so `self.storage`
        // must not be overwritten before we know a real Raft instance is
        // actually backing it.
        self.storage = state_arc;
        self.listener_task = Some(listener_task);
        self.raft_peer_addresses = Some(raft_peer_addresses);
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

    /// Resolve the running `Raft` handle, or a fail-loud error explaining why
    /// this node cannot service a real cross-node consensus operation. Never
    /// silently succeeds against local-only fallback storage — a membership or
    /// leadership operation that never reaches openraft must not report
    /// success (see the [`RaftClusterError`] variants).
    #[cfg(feature = "raft")]
    fn require_raft(&self, op: &str) -> Result<&Raft<OxirsTypeConfig>> {
        match self.raft.as_ref() {
            Some(raft) => Ok(raft),
            None if self.multi_node_requested.load(Ordering::SeqCst) => {
                Err(RaftClusterError::ConsensusUnavailable {
                    node_id: self.node_id,
                }
                .into())
            }
            None => Err(anyhow::anyhow!(
                "cannot {op}: node {} is not running a multi-node Raft cluster \
                 (single-node mode has no membership to change or leadership to move)",
                self.node_id
            )),
        }
    }

    /// Add a node to the cluster through real OpenRaft reconfiguration.
    ///
    /// This is the genuine joint-consensus path, not a no-op state-machine
    /// command: the new node's address is registered into the live network
    /// factory so the leader can dial it, the node is first added as a
    /// *learner* (blocking until its log catches up), and only then promoted
    /// to a full voter via `change_membership`. Both steps are committed
    /// through openraft, so the added node actually counts toward quorum and
    /// receives replication. Errors from either step are surfaced verbatim.
    #[cfg(feature = "raft")]
    pub async fn add_node(&self, node_id: OxirsNodeId, address: SocketAddr) -> Result<()> {
        let raft = self.require_raft("add node")?;

        // Register the joiner's address into the *live* factory map first, so
        // the replication that `add_learner(.., blocking=true)` waits on has a
        // real endpoint to dial. Without this the learner add would block
        // forever (no address ⇒ RaftNetwork can never reach it).
        if let Some(addrs) = self.raft_peer_addresses.as_ref() {
            addrs.write().await.insert(node_id, address);
        }

        raft.add_learner(node_id, BasicNode::new(address.to_string()), true)
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "node {}: failed to add learner {node_id}: {e}",
                    self.node_id
                )
            })?;

        raft.change_membership(
            openraft::ChangeMembers::AddVoterIds(std::iter::once(node_id).collect()),
            true,
        )
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "node {}: failed to promote {node_id} to voter: {e}",
                self.node_id
            )
        })?;

        tracing::info!(
            node_id = self.node_id,
            added = node_id,
            "committed membership change adding node {node_id} as a voter"
        );
        Ok(())
    }

    /// Remove a node from the cluster through real OpenRaft reconfiguration.
    ///
    /// Commits a `change_membership` that drops `node_id` from the voter set
    /// (and, with `retain = false`, from the cluster entirely) so openraft
    /// stops counting it toward quorum and stops replicating to it — as
    /// opposed to the previous no-op state-machine command that left the
    /// departed node counted as a live voter forever.
    #[cfg(feature = "raft")]
    pub async fn remove_node(&self, node_id: OxirsNodeId) -> Result<()> {
        let raft = self.require_raft("remove node")?;

        raft.change_membership(
            openraft::ChangeMembers::RemoveVoters(std::iter::once(node_id).collect()),
            false,
        )
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "node {}: failed to remove voter {node_id}: {e}",
                self.node_id
            )
        })?;

        if let Some(addrs) = self.raft_peer_addresses.as_ref() {
            addrs.write().await.remove(&node_id);
        }

        tracing::info!(
            node_id = self.node_id,
            removed = node_id,
            "committed membership change removing node {node_id}"
        );
        Ok(())
    }

    /// Transfer leadership to `target_node`.
    ///
    /// OpenRaft 0.9.24 exposes no native leader-transfer call, so this
    /// implements the Raft §3.10 handoff directly over the crate's own Raft
    /// RPC transport: it verifies this node is the current leader and that
    /// `target_node` is a voter whose log has caught up, then sends a
    /// `TimeoutNow` RPC that makes the target immediately start an election
    /// (via `Raft::trigger().elect()` on the receiving side). Because the
    /// target's log is up to date, Raft's election-restriction guarantees it
    /// wins over any stale voter, and this leader steps down when it observes
    /// the target's higher term. Fails loud if this node is not the leader,
    /// the target is not a caught-up voter, or the RPC cannot be delivered —
    /// it never reports a fake success.
    #[cfg(feature = "raft")]
    pub async fn transfer_leadership(&self, target_node: OxirsNodeId) -> Result<()> {
        let raft = self.require_raft("transfer leadership")?;
        let metrics = raft.metrics().borrow().clone();

        if metrics.current_leader != Some(self.node_id) {
            return Err(anyhow::anyhow!(
                "node {} cannot transfer leadership: it is not the current leader (leader is {:?})",
                self.node_id,
                metrics.current_leader
            ));
        }
        if target_node == self.node_id {
            return Ok(());
        }

        // Target must be a voter and its replication must have caught up to
        // this leader's last log id, or the handoff could hand leadership to a
        // node that then cannot be elected (or, worse, loses committed state).
        let voters: BTreeSet<OxirsNodeId> =
            metrics.membership_config.membership().voter_ids().collect();
        if !voters.contains(&target_node) {
            return Err(anyhow::anyhow!(
                "node {} cannot transfer leadership to {target_node}: it is not a voter",
                self.node_id
            ));
        }
        // `caught_up` is the target's matched log id as this leader sees it.
        let caught_up = metrics
            .replication
            .as_ref()
            .and_then(|repl| repl.get(&target_node).copied())
            .flatten();
        match (caught_up, metrics.last_log_index) {
            // Target has replicated up to (or past) this leader's last log.
            (Some(matched), Some(last)) if matched.index >= last => {}
            // Leader has no log yet: nothing for the target to catch up to.
            (_, None) => {}
            _ => {
                return Err(anyhow::anyhow!(
                    "node {} cannot transfer leadership to {target_node}: target has not caught up \
                     (matched={:?}, leader_last={:?})",
                    self.node_id,
                    caught_up,
                    metrics.last_log_index
                ));
            }
        }

        let address = self
            .raft_peer_addresses
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "node {} cannot transfer leadership: no live peer address map",
                    self.node_id
                )
            })?
            .read()
            .await
            .get(&target_node)
            .copied()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "node {} cannot transfer leadership to {target_node}: no known address",
                    self.node_id
                )
            })?;

        crate::raft_network::send_timeout_now(
            self.node_id,
            target_node,
            address,
            self.tls.as_deref(),
        )
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "node {} failed to deliver leadership-transfer signal to {target_node}: {e}",
                self.node_id
            )
        })?;

        tracing::info!(
            node_id = self.node_id,
            target = target_node,
            "sent leadership-transfer (TimeoutNow) signal; target will start an election"
        );
        Ok(())
    }

    /// Trigger a local election immediately (openraft `Trigger::elect`).
    ///
    /// Used by recovery to nudge a quorum-connected cluster that has lost its
    /// leader back into electing one, without the destructive full re-init the
    /// old recovery path performed.
    #[cfg(feature = "raft")]
    pub async fn trigger_election(&self) -> Result<()> {
        let raft = self.require_raft("trigger election")?;
        raft.trigger()
            .elect()
            .await
            .map_err(|e| anyhow::anyhow!("node {}: failed to trigger election: {e}", self.node_id))
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
            // Drop the live factory address-map handle; a fresh `init_raft`
            // installs a new one.
            self.raft_peer_addresses = None;
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
mod tests;
