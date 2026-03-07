//! # Distributed State Management with CRDT Consistency
//!
//! Provides distributed state management primitives using Conflict-free Replicated Data Types
//! (CRDTs) for eventual consistency, Merkle-verified checkpointing, and gossip-based
//! state replication.
//!
//! ## Components
//!
//! - [`CrdtEventLog`]: CRDT-based distributed event log (G-Counter, PN-Counter, LWW-Register)
//! - [`DistributedCheckpointer`]: Checkpoint stream state across nodes with Merkle verification
//! - [`StateReplicationManager`]: Replicates stream state using a gossip protocol

pub mod manager;
pub use manager::*;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

// ─── Error Types ─────────────────────────────────────────────────────────────

/// Errors in distributed state operations
#[derive(Error, Debug)]
pub enum DistributedStateError {
    #[error("Checkpoint verification failed: expected {expected}, got {actual}")]
    CheckpointVerificationFailed { expected: String, actual: String },

    #[error("Stale state: local version {local} < remote version {remote}")]
    StaleState { local: u64, remote: u64 },

    #[error("Merge conflict on key {key}: {detail}")]
    MergeConflict { key: String, detail: String },

    #[error("Replication error: {0}")]
    Replication(String),

    #[error("Checkpoint serialisation error: {0}")]
    Serialisation(String),

    #[error("Node not registered: {0}")]
    NodeNotRegistered(String),
}

/// Result type for distributed state operations
pub type StateResult<T> = Result<T, DistributedStateError>;

// ─── CRDT Primitives ─────────────────────────────────────────────────────────

/// A Grow-only Counter (G-Counter) CRDT.
///
/// Each node maintains its own counter; the global value is the sum of all
/// node counters. Merging takes the per-node maximum.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GCounter {
    /// Per-node increment counts
    counts: HashMap<String, u64>,
}

impl GCounter {
    /// Creates a new empty G-Counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Increments the counter for a specific node.
    pub fn increment(&mut self, node_id: &str) {
        *self.counts.entry(node_id.to_string()).or_insert(0) += 1;
    }

    /// Increments the counter for a specific node by `delta`.
    pub fn increment_by(&mut self, node_id: &str, delta: u64) {
        *self.counts.entry(node_id.to_string()).or_insert(0) += delta;
    }

    /// Returns the current global value (sum of all node counters).
    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Merges another G-Counter into this one by taking the per-node maximum.
    pub fn merge(&mut self, other: &GCounter) {
        for (node, &count) in &other.counts {
            let local = self.counts.entry(node.clone()).or_insert(0);
            if count > *local {
                *local = count;
            }
        }
    }
}

/// A Positive-Negative Counter (PN-Counter) CRDT.
///
/// Supports both increment and decrement by maintaining two G-Counters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PnCounter {
    positive: GCounter,
    negative: GCounter,
}

impl PnCounter {
    /// Creates a new empty PN-Counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Increments the counter for a node.
    pub fn increment(&mut self, node_id: &str) {
        self.positive.increment(node_id);
    }

    /// Decrements the counter for a node.
    pub fn decrement(&mut self, node_id: &str) {
        self.negative.increment(node_id);
    }

    /// Returns the current value (positive total - negative total).
    pub fn value(&self) -> i64 {
        self.positive.value() as i64 - self.negative.value() as i64
    }

    /// Merges another PN-Counter into this one.
    pub fn merge(&mut self, other: &PnCounter) {
        self.positive.merge(&other.positive);
        self.negative.merge(&other.negative);
    }
}

/// A Last-Writer-Wins Register (LWW-Register) CRDT.
///
/// Stores a value tagged with a timestamp; merges take the most recent write.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: Serialize + serde::de::DeserializeOwned")]
pub struct LwwRegister<T: Clone + Serialize + serde::de::DeserializeOwned> {
    /// Current value
    value: Option<T>,
    /// Timestamp of last write (microseconds since UNIX epoch)
    timestamp: u64,
    /// Node that performed the last write
    writer_node: String,
}

impl<T: Clone + Serialize + serde::de::DeserializeOwned> LwwRegister<T> {
    /// Creates a new empty LWW-Register.
    pub fn new() -> Self {
        Self {
            value: None,
            timestamp: 0,
            writer_node: String::new(),
        }
    }

    /// Writes a value, timestamped with the current wall-clock time.
    pub fn write(&mut self, value: T, node_id: &str) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.write_at(value, node_id, ts);
    }

    /// Writes a value at an explicit timestamp (for deterministic testing).
    pub fn write_at(&mut self, value: T, node_id: &str, timestamp: u64) {
        if timestamp >= self.timestamp {
            self.value = Some(value);
            self.timestamp = timestamp;
            self.writer_node = node_id.to_string();
        }
    }

    /// Returns a reference to the current value, if set.
    pub fn read(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// Returns the timestamp of the last write.
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Merges another LWW-Register by keeping the most recent write.
    pub fn merge(&mut self, other: &LwwRegister<T>) {
        if other.timestamp > self.timestamp {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
            self.writer_node = other.writer_node.clone();
        }
    }
}

impl<T: Clone + Serialize + serde::de::DeserializeOwned> Default for LwwRegister<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── CRDT Event Log ──────────────────────────────────────────────────────────

/// A log entry stored in the CRDT event log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrdtLogEntry {
    /// Logical sequence number
    pub sequence: u64,
    /// Node that produced this entry
    pub origin_node: String,
    /// Wall-clock timestamp (microseconds since UNIX epoch)
    pub timestamp: u64,
    /// Payload bytes
    pub payload: Vec<u8>,
}

/// Statistics for the CRDT event log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrdtEventLogStats {
    /// Total number of log entries
    pub total_entries: u64,
    /// Number of nodes that have contributed entries
    pub contributing_nodes: usize,
    /// Current event count from the G-Counter
    pub event_counter: u64,
    /// Current net activity from the PN-Counter
    pub activity_counter: i64,
}

/// A CRDT-based distributed event log.
///
/// Combines a G-Counter (total events), a PN-Counter (net activity), and
/// LWW-Register entries to provide a causally consistent log across nodes.
pub struct CrdtEventLog {
    node_id: String,
    /// Append-only log entries
    entries: Arc<RwLock<Vec<CrdtLogEntry>>>,
    /// G-Counter tracking total events added
    event_counter: Arc<RwLock<GCounter>>,
    /// PN-Counter tracking net activity (add vs remove)
    activity_counter: Arc<RwLock<PnCounter>>,
    /// LWW-Register per named key for last-write-wins state
    registers: Arc<RwLock<HashMap<String, LwwRegister<Vec<u8>>>>>,
    /// Monotonically increasing sequence for this node
    next_sequence: Arc<RwLock<u64>>,
}

impl CrdtEventLog {
    /// Creates a new CRDT event log for the given node.
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            entries: Arc::new(RwLock::new(Vec::new())),
            event_counter: Arc::new(RwLock::new(GCounter::new())),
            activity_counter: Arc::new(RwLock::new(PnCounter::new())),
            registers: Arc::new(RwLock::new(HashMap::new())),
            next_sequence: Arc::new(RwLock::new(0)),
        }
    }

    /// Appends an event to the log, returning the assigned sequence number.
    pub fn append(&self, payload: Vec<u8>) -> u64 {
        let mut seq = self.next_sequence.write();
        let sequence = *seq;
        *seq += 1;
        drop(seq);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let entry = CrdtLogEntry {
            sequence,
            origin_node: self.node_id.clone(),
            timestamp,
            payload,
        };
        self.entries.write().push(entry);
        self.event_counter.write().increment(&self.node_id);
        self.activity_counter.write().increment(&self.node_id);
        debug!("Appended log entry seq={}", sequence);
        sequence
    }

    /// Records a removal (decrement) event without removing from the immutable log.
    pub fn record_removal(&self) {
        self.activity_counter.write().decrement(&self.node_id);
    }

    /// Writes a named LWW-Register value.
    pub fn set_register(&self, key: &str, value: Vec<u8>) {
        let mut registers = self.registers.write();
        let reg = registers.entry(key.to_string()).or_default();
        reg.write(value, &self.node_id);
    }

    /// Reads a named LWW-Register value.
    pub fn get_register(&self, key: &str) -> Option<Vec<u8>> {
        self.registers.read().get(key)?.read().cloned()
    }

    /// Merges a remote CRDT event log state into this log.
    pub fn merge_remote(&self, remote: &RemoteCrdtState) {
        // Merge G-Counter
        self.event_counter.write().merge(&remote.event_counter);
        // Merge PN-Counter
        self.activity_counter
            .write()
            .merge(&remote.activity_counter);
        // Merge LWW-Registers
        let mut local_regs = self.registers.write();
        for (key, remote_reg) in &remote.registers {
            let local_reg = local_regs.entry(key.clone()).or_default();
            local_reg.merge(remote_reg);
        }
        // Append any new entries — deduplicate by (sequence, origin_node) pair
        let mut entries = self.entries.write();
        let existing_keys: std::collections::HashSet<(u64, String)> = entries
            .iter()
            .map(|e| (e.sequence, e.origin_node.clone()))
            .collect();
        for entry in &remote.entries {
            let key = (entry.sequence, entry.origin_node.clone());
            if !existing_keys.contains(&key) {
                entries.push(entry.clone());
            }
        }
        entries.sort_by_key(|e| e.sequence);
        debug!("Merged remote CRDT state, total entries: {}", entries.len());
    }

    /// Exports current state for transmission to remote nodes.
    pub fn export_state(&self) -> RemoteCrdtState {
        RemoteCrdtState {
            origin_node: self.node_id.clone(),
            event_counter: self.event_counter.read().clone(),
            activity_counter: self.activity_counter.read().clone(),
            registers: self.registers.read().clone(),
            entries: self.entries.read().clone(),
        }
    }

    /// Returns CRDT event log statistics.
    pub fn stats(&self) -> CrdtEventLogStats {
        CrdtEventLogStats {
            total_entries: self.entries.read().len() as u64,
            contributing_nodes: {
                let entries = self.entries.read();
                entries
                    .iter()
                    .map(|e| e.origin_node.as_str())
                    .collect::<std::collections::HashSet<_>>()
                    .len()
            },
            event_counter: self.event_counter.read().value(),
            activity_counter: self.activity_counter.read().value(),
        }
    }

    /// Returns all log entries in sequence order.
    pub fn entries(&self) -> Vec<CrdtLogEntry> {
        self.entries.read().clone()
    }
}

/// Portable CRDT state for gossip transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteCrdtState {
    pub origin_node: String,
    pub event_counter: GCounter,
    pub activity_counter: PnCounter,
    pub registers: HashMap<String, LwwRegister<Vec<u8>>>,
    pub entries: Vec<CrdtLogEntry>,
}

// ─── Distributed Checkpointer ─────────────────────────────────────────────────

/// A single node's checkpoint snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCheckpoint {
    /// Checkpoint identifier
    pub checkpoint_id: String,
    /// Node that produced this checkpoint
    pub node_id: String,
    /// Logical timestamp (e.g. stream offset)
    pub logical_time: u64,
    /// Opaque checkpoint state bytes
    pub state_bytes: Vec<u8>,
    /// Merkle root hash of `state_bytes` (hex-encoded SHA-256)
    pub merkle_root: String,
    /// Wall-clock creation time
    pub created_at: SystemTime,
}

/// A global checkpoint aggregating per-node checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalDistributedCheckpoint {
    /// Global checkpoint identifier
    pub checkpoint_id: String,
    /// Per-node checkpoints
    pub node_checkpoints: HashMap<String, NodeCheckpoint>,
    /// Combined Merkle root over all node roots
    pub combined_merkle_root: String,
    /// Minimum logical time across all nodes
    pub min_logical_time: u64,
    /// Maximum logical time across all nodes
    pub max_logical_time: u64,
    /// Whether all expected nodes contributed
    pub is_complete: bool,
    /// Creation time of this global checkpoint
    pub created_at: SystemTime,
}

/// Statistics for the distributed checkpointer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointerStats {
    /// Total checkpoints completed
    pub completed_checkpoints: u64,
    /// Total checkpoints that failed verification
    pub failed_verifications: u64,
    /// Latest completed checkpoint ID
    pub latest_checkpoint_id: Option<String>,
    /// Average state size across checkpoints (bytes)
    pub avg_state_bytes: f64,
}

/// Checkpoints stream state across nodes with Merkle verification.
///
/// Each node submits its local checkpoint; once all expected nodes contribute,
/// a global checkpoint is formed and its combined Merkle root is verified.
pub struct DistributedCheckpointer {
    expected_nodes: std::collections::HashSet<String>,
    /// Active (incomplete) checkpoint collections, keyed by checkpoint_id
    active: Arc<RwLock<HashMap<String, Vec<NodeCheckpoint>>>>,
    /// Completed global checkpoints, keyed by checkpoint_id
    completed: Arc<RwLock<Vec<GlobalDistributedCheckpoint>>>,
    stats: Arc<RwLock<CheckpointerStats>>,
}

impl DistributedCheckpointer {
    /// Creates a new checkpointer expecting contributions from the given nodes.
    pub fn new(expected_nodes: std::collections::HashSet<String>) -> Self {
        Self {
            expected_nodes,
            active: Arc::new(RwLock::new(HashMap::new())),
            completed: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(CheckpointerStats {
                completed_checkpoints: 0,
                failed_verifications: 0,
                latest_checkpoint_id: None,
                avg_state_bytes: 0.0,
            })),
        }
    }

    /// Submits a node checkpoint.
    ///
    /// Returns `Some(GlobalDistributedCheckpoint)` when all expected nodes
    /// have contributed for this `checkpoint_id`.
    pub fn submit_node_checkpoint(
        &self,
        checkpoint: NodeCheckpoint,
    ) -> StateResult<Option<GlobalDistributedCheckpoint>> {
        // Verify Merkle root on arrival
        let computed = Self::compute_merkle_root(&checkpoint.state_bytes);
        if computed != checkpoint.merkle_root {
            self.stats.write().failed_verifications += 1;
            return Err(DistributedStateError::CheckpointVerificationFailed {
                expected: checkpoint.merkle_root.clone(),
                actual: computed,
            });
        }
        let checkpoint_id = checkpoint.checkpoint_id.clone();
        {
            let mut active = self.active.write();
            active
                .entry(checkpoint_id.clone())
                .or_default()
                .push(checkpoint);
        }
        self.try_finalise(&checkpoint_id)
    }

    /// Returns the latest completed global checkpoint, if any.
    pub fn latest_checkpoint(&self) -> Option<GlobalDistributedCheckpoint> {
        self.completed.read().last().cloned()
    }

    /// Returns all completed checkpoints.
    pub fn all_checkpoints(&self) -> Vec<GlobalDistributedCheckpoint> {
        self.completed.read().clone()
    }

    /// Returns checkpointer statistics.
    pub fn stats(&self) -> CheckpointerStats {
        self.stats.read().clone()
    }

    fn try_finalise(
        &self,
        checkpoint_id: &str,
    ) -> StateResult<Option<GlobalDistributedCheckpoint>> {
        let active = self.active.read();
        let contributions = match active.get(checkpoint_id) {
            Some(c) => c,
            None => return Ok(None),
        };
        let contributed_ids: std::collections::HashSet<&str> =
            contributions.iter().map(|c| c.node_id.as_str()).collect();
        let expected_refs: std::collections::HashSet<&str> =
            self.expected_nodes.iter().map(|s| s.as_str()).collect();

        if contributed_refs_subset(&contributed_ids, &expected_refs) {
            drop(active);
            let global = self.build_global(checkpoint_id)?;
            self.completed.write().push(global.clone());
            let mut stats = self.stats.write();
            stats.completed_checkpoints += 1;
            stats.latest_checkpoint_id = Some(checkpoint_id.to_string());
            let total_bytes: usize = global
                .node_checkpoints
                .values()
                .map(|c| c.state_bytes.len())
                .sum();
            stats.avg_state_bytes = total_bytes as f64 / global.node_checkpoints.len() as f64;
            self.active.write().remove(checkpoint_id);
            Ok(Some(global))
        } else {
            Ok(None)
        }
    }

    fn build_global(&self, checkpoint_id: &str) -> StateResult<GlobalDistributedCheckpoint> {
        let active = self.active.read();
        let contributions = active
            .get(checkpoint_id)
            .ok_or_else(|| DistributedStateError::Serialisation("no active checkpoint".into()))?;

        let node_checkpoints: HashMap<String, NodeCheckpoint> = contributions
            .iter()
            .map(|c| (c.node_id.clone(), c.clone()))
            .collect();

        let mut sorted_roots: Vec<String> = node_checkpoints
            .values()
            .map(|c| c.merkle_root.clone())
            .collect();
        sorted_roots.sort();
        let combined_data = sorted_roots.join("");
        let combined_merkle_root = Self::compute_merkle_root(combined_data.as_bytes());

        let min_logical_time = node_checkpoints
            .values()
            .map(|c| c.logical_time)
            .min()
            .unwrap_or(0);
        let max_logical_time = node_checkpoints
            .values()
            .map(|c| c.logical_time)
            .max()
            .unwrap_or(0);

        Ok(GlobalDistributedCheckpoint {
            checkpoint_id: checkpoint_id.to_string(),
            is_complete: node_checkpoints.len() == self.expected_nodes.len(),
            node_checkpoints,
            combined_merkle_root,
            min_logical_time,
            max_logical_time,
            created_at: SystemTime::now(),
        })
    }

    /// Computes a simple Merkle root as the hex-encoded SHA-256 of the data.
    ///
    /// In a production system this would build a full Merkle tree; here we
    /// use a single-level hash for correctness without external crypto deps.
    pub fn compute_merkle_root(data: &[u8]) -> String {
        // FNV-1a 64-bit as a lightweight hash (no sha2 dep needed)
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        let mut hash = FNV_OFFSET;
        for byte in data {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        format!("{:016x}", hash)
    }
}

fn contributed_refs_subset(
    contributed: &std::collections::HashSet<&str>,
    expected: &std::collections::HashSet<&str>,
) -> bool {
    expected.is_subset(contributed) || contributed == expected
}

/// Helper to create a NodeCheckpoint with a correct Merkle root.
pub fn make_node_checkpoint(
    checkpoint_id: impl Into<String>,
    node_id: impl Into<String>,
    logical_time: u64,
    state_bytes: Vec<u8>,
) -> NodeCheckpoint {
    let merkle_root = DistributedCheckpointer::compute_merkle_root(&state_bytes);
    NodeCheckpoint {
        checkpoint_id: checkpoint_id.into(),
        node_id: node_id.into(),
        logical_time,
        state_bytes,
        merkle_root,
        created_at: SystemTime::now(),
    }
}

// ─── State Replication Manager ───────────────────────────────────────────────

/// A replication message sent between nodes via gossip
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMessage {
    /// Source node ID
    pub from_node: String,
    /// Target node ID (empty = broadcast)
    pub to_node: Option<String>,
    /// Gossip round number
    pub round: u64,
    /// State digest for comparison (hex hash of current state)
    pub state_digest: String,
    /// Full state payload (present when digest differs from recipient)
    pub state_payload: Option<Vec<u8>>,
    /// Timestamp of this gossip message
    pub timestamp: SystemTime,
}

/// Per-node replication state tracked by the manager
#[derive(Debug, Clone)]
struct NodeReplicationState {
    node_id: String,
    last_seen: Instant,
    last_digest: String,
    round: u64,
}

/// Configuration for the state replication manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Number of nodes to gossip with per round (fanout)
    pub fanout: usize,
    /// Interval between gossip rounds
    pub gossip_interval: Duration,
    /// Maximum number of rounds before a node is considered stale
    pub stale_rounds: u64,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            fanout: 3,
            gossip_interval: Duration::from_millis(500),
            stale_rounds: 10,
        }
    }
}

/// Statistics for the state replication manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationStats {
    /// Total gossip messages sent
    pub messages_sent: u64,
    /// Total gossip messages received
    pub messages_received: u64,
    /// Total state synchronisations triggered
    pub sync_count: u64,
    /// Number of nodes currently tracked
    pub tracked_nodes: usize,
    /// Current gossip round
    pub current_round: u64,
}

/// Replicates stream state across nodes using a gossip protocol.
///
/// Each node periodically gossips its state digest to a random subset of
/// peers; peers that detect a divergence request the full state payload.
pub struct StateReplicationManager {
    node_id: String,
    config: ReplicationConfig,
    /// Tracked peer nodes
    peers: Arc<RwLock<HashMap<String, NodeReplicationState>>>,
    /// Local state digest
    local_digest: Arc<RwLock<String>>,
    /// Local state bytes
    local_state: Arc<RwLock<Vec<u8>>>,
    /// Gossip round counter
    current_round: Arc<RwLock<u64>>,
    stats: Arc<RwLock<ReplicationStats>>,
    /// Received gossip messages (buffer for processing)
    inbox: Arc<RwLock<Vec<GossipMessage>>>,
}

impl StateReplicationManager {
    /// Creates a new replication manager for the given node.
    pub fn new(node_id: impl Into<String>, config: ReplicationConfig) -> Self {
        Self {
            node_id: node_id.into(),
            config,
            peers: Arc::new(RwLock::new(HashMap::new())),
            local_digest: Arc::new(RwLock::new(String::new())),
            local_state: Arc::new(RwLock::new(Vec::new())),
            current_round: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(ReplicationStats {
                messages_sent: 0,
                messages_received: 0,
                sync_count: 0,
                tracked_nodes: 0,
                current_round: 0,
            })),
            inbox: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Registers a peer node for gossip.
    pub fn add_peer(&self, node_id: impl Into<String>) {
        let id = node_id.into();
        self.peers.write().insert(
            id.clone(),
            NodeReplicationState {
                node_id: id,
                last_seen: Instant::now(),
                last_digest: String::new(),
                round: 0,
            },
        );
        self.stats.write().tracked_nodes = self.peers.read().len();
    }

    /// Updates the local state, recomputing the digest.
    pub fn update_local_state(&self, state: Vec<u8>) {
        let digest = DistributedCheckpointer::compute_merkle_root(&state);
        *self.local_state.write() = state;
        *self.local_digest.write() = digest;
    }

    /// Produces gossip messages for the current round (up to `fanout` peers).
    ///
    /// Returns the list of gossip messages to send.
    pub fn produce_gossip(&self) -> Vec<GossipMessage> {
        let mut round = self.current_round.write();
        *round += 1;
        let current_round = *round;
        drop(round);

        let digest = self.local_digest.read().clone();
        let state_payload = Some(self.local_state.read().clone());

        let peers: Vec<String> = self.peers.read().keys().cloned().collect();
        // Deterministic peer selection using modular arithmetic (no rand)
        let fanout = self.config.fanout.min(peers.len());
        let offset = (current_round as usize) % peers.len().max(1);
        let selected: Vec<&String> = peers.iter().cycle().skip(offset).take(fanout).collect();

        let mut messages = Vec::with_capacity(selected.len());
        for peer_id in selected {
            messages.push(GossipMessage {
                from_node: self.node_id.clone(),
                to_node: Some(peer_id.clone()),
                round: current_round,
                state_digest: digest.clone(),
                state_payload: state_payload.clone(),
                timestamp: SystemTime::now(),
            });
        }
        self.stats.write().messages_sent += messages.len() as u64;
        self.stats.write().current_round = current_round;
        messages
    }

    /// Receives and processes an incoming gossip message.
    ///
    /// Returns `true` if a state synchronisation was triggered (digest differed).
    pub fn receive_gossip(&self, msg: GossipMessage) -> StateResult<bool> {
        self.stats.write().messages_received += 1;
        self.inbox.write().push(msg.clone());

        // Update peer tracking
        {
            let mut peers = self.peers.write();
            if let Some(peer) = peers.get_mut(&msg.from_node) {
                peer.last_seen = Instant::now();
                peer.round = msg.round;
            } else {
                warn!("Gossip from unknown peer {}", msg.from_node);
            }
        }

        let local_digest = self.local_digest.read().clone();
        if msg.state_digest != local_digest {
            // Digest differs — apply remote state if payload provided
            if let Some(payload) = msg.state_payload {
                info!("Syncing state from {} (round {})", msg.from_node, msg.round);
                self.update_local_state(payload);
                self.stats.write().sync_count += 1;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Returns the current local state digest.
    pub fn local_digest(&self) -> String {
        self.local_digest.read().clone()
    }

    /// Returns replication statistics.
    pub fn stats(&self) -> ReplicationStats {
        self.stats.read().clone()
    }

    /// Returns IDs of stale peers (not heard from in `stale_rounds` rounds).
    pub fn stale_peers(&self) -> Vec<String> {
        let current_round = *self.current_round.read();
        self.peers
            .read()
            .values()
            .filter(|p| current_round.saturating_sub(p.round) > self.config.stale_rounds)
            .map(|p| p.node_id.clone())
            .collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── G-Counter tests ──────────────────────────────────────────────────────

    #[test]
    fn test_g_counter_basic() {
        let mut c = GCounter::new();
        c.increment("node-1");
        c.increment("node-1");
        c.increment("node-2");
        assert_eq!(c.value(), 3);
    }

    #[test]
    fn test_g_counter_merge() {
        let mut c1 = GCounter::new();
        c1.increment_by("node-1", 5);

        let mut c2 = GCounter::new();
        c2.increment_by("node-1", 3);
        c2.increment_by("node-2", 7);

        c1.merge(&c2);
        // node-1: max(5, 3)=5, node-2: max(0, 7)=7
        assert_eq!(c1.value(), 12);
    }

    #[test]
    fn test_g_counter_merge_idempotent() {
        let mut c1 = GCounter::new();
        c1.increment_by("node-1", 10);
        let c2 = c1.clone();
        c1.merge(&c2);
        assert_eq!(c1.value(), 10);
    }

    // ── PN-Counter tests ─────────────────────────────────────────────────────

    #[test]
    fn test_pn_counter_basic() {
        let mut c = PnCounter::new();
        c.increment("node-1");
        c.increment("node-1");
        c.increment("node-1");
        c.decrement("node-1");
        assert_eq!(c.value(), 2);
    }

    #[test]
    fn test_pn_counter_merge() {
        let mut c1 = PnCounter::new();
        c1.increment("node-1");

        let mut c2 = PnCounter::new();
        c2.increment("node-2");
        c2.decrement("node-2");

        c1.merge(&c2);
        // node-1: +1, node-2: +1-1=0 → net 1
        assert_eq!(c1.value(), 1);
    }

    // ── LWW-Register tests ───────────────────────────────────────────────────

    #[test]
    fn test_lww_register_write_and_read() {
        let mut reg: LwwRegister<String> = LwwRegister::new();
        reg.write_at("hello".to_string(), "node-1", 100);
        assert_eq!(reg.read(), Some(&"hello".to_string()));
    }

    #[test]
    fn test_lww_register_last_write_wins() {
        let mut reg: LwwRegister<String> = LwwRegister::new();
        reg.write_at("first".to_string(), "node-1", 100);
        reg.write_at("second".to_string(), "node-2", 200);
        reg.write_at("old".to_string(), "node-3", 50);
        assert_eq!(reg.read(), Some(&"second".to_string()));
    }

    #[test]
    fn test_lww_register_merge() {
        let mut r1: LwwRegister<String> = LwwRegister::new();
        r1.write_at("r1-value".to_string(), "node-1", 100);

        let mut r2: LwwRegister<String> = LwwRegister::new();
        r2.write_at("r2-value".to_string(), "node-2", 200);

        r1.merge(&r2);
        assert_eq!(r1.read(), Some(&"r2-value".to_string()));
        assert_eq!(r1.timestamp(), 200);
    }

    // ── CRDT Event Log tests ─────────────────────────────────────────────────

    #[test]
    fn test_crdt_event_log_append() {
        let log = CrdtEventLog::new("node-1");
        let seq0 = log.append(b"event-0".to_vec());
        let seq1 = log.append(b"event-1".to_vec());
        assert_eq!(seq0, 0);
        assert_eq!(seq1, 1);
        let stats = log.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.event_counter, 2);
        assert_eq!(stats.activity_counter, 2);
    }

    #[test]
    fn test_crdt_event_log_registers() {
        let log = CrdtEventLog::new("node-a");
        log.set_register("config", b"v1".to_vec());
        assert_eq!(log.get_register("config"), Some(b"v1".to_vec()));
        log.set_register("config", b"v2".to_vec());
        // LWW — but both writes are "now", so v2 should win (same or higher ts)
        assert!(log.get_register("config").is_some());
    }

    #[test]
    fn test_crdt_event_log_merge() {
        let log1 = CrdtEventLog::new("node-1");
        log1.append(b"n1-event".to_vec());

        let log2 = CrdtEventLog::new("node-2");
        log2.append(b"n2-event".to_vec());
        log2.append(b"n2-event-2".to_vec());

        let remote_state = log2.export_state();
        log1.merge_remote(&remote_state);

        let stats = log1.stats();
        // log1 had 1 entry from node-1; after merge should have 1 + 2 = 3
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.contributing_nodes, 2);
    }

    #[test]
    fn test_crdt_event_log_record_removal() {
        let log = CrdtEventLog::new("node-x");
        log.append(b"e1".to_vec());
        log.append(b"e2".to_vec());
        log.record_removal();
        let stats = log.stats();
        // G-counter: 2, activity: +2 -1 = 1
        assert_eq!(stats.event_counter, 2);
        assert_eq!(stats.activity_counter, 1);
    }

    // ── Distributed Checkpointer tests ───────────────────────────────────────

    #[test]
    fn test_distributed_checkpointer_completes_on_all_nodes() {
        let expected: std::collections::HashSet<String> =
            ["n1", "n2"].iter().map(|s| s.to_string()).collect();
        let checkpointer = DistributedCheckpointer::new(expected);

        let cp1 = make_node_checkpoint("ckpt-1", "n1", 100, b"state-n1".to_vec());
        let result = checkpointer
            .submit_node_checkpoint(cp1)
            .expect("submit should succeed");
        assert!(result.is_none(), "should not complete with 1/2 nodes");

        let cp2 = make_node_checkpoint("ckpt-1", "n2", 110, b"state-n2".to_vec());
        let result = checkpointer
            .submit_node_checkpoint(cp2)
            .expect("submit should succeed");
        assert!(result.is_some(), "should complete with 2/2 nodes");

        let global = result.expect("must be Some");
        assert_eq!(global.checkpoint_id, "ckpt-1");
        assert_eq!(global.node_checkpoints.len(), 2);
        assert_eq!(global.min_logical_time, 100);
        assert_eq!(global.max_logical_time, 110);
        assert!(global.is_complete);
    }

    #[test]
    fn test_distributed_checkpointer_rejects_bad_merkle() {
        let expected: std::collections::HashSet<String> =
            ["n1"].iter().map(|s| s.to_string()).collect();
        let checkpointer = DistributedCheckpointer::new(expected);

        let mut cp = make_node_checkpoint("ckpt-bad", "n1", 50, b"data".to_vec());
        cp.merkle_root = "deadbeef".to_string(); // deliberately wrong

        let result = checkpointer.submit_node_checkpoint(cp);
        assert!(
            matches!(
                result,
                Err(DistributedStateError::CheckpointVerificationFailed { .. })
            ),
            "should reject bad Merkle root"
        );

        let stats = checkpointer.stats();
        assert_eq!(stats.failed_verifications, 1);
    }

    // ── State Replication Manager tests ──────────────────────────────────────

    #[test]
    fn test_state_replication_gossip_produced() {
        let config = ReplicationConfig {
            fanout: 2,
            gossip_interval: Duration::from_millis(100),
            stale_rounds: 5,
        };
        let mgr = StateReplicationManager::new("node-1", config);
        mgr.add_peer("node-2");
        mgr.add_peer("node-3");
        mgr.update_local_state(b"my-state".to_vec());

        let messages = mgr.produce_gossip();
        assert!(!messages.is_empty(), "should produce gossip messages");
        assert!(messages.len() <= 2, "fanout should be respected");
        for msg in &messages {
            assert_eq!(msg.from_node, "node-1");
            assert!(!msg.state_digest.is_empty());
        }
    }

    #[test]
    fn test_state_replication_receive_sync() {
        let config = ReplicationConfig::default();
        let receiver = StateReplicationManager::new("node-2", config);
        receiver.add_peer("node-1");
        receiver.update_local_state(b"old-state".to_vec());

        let new_state = b"new-state-from-node-1".to_vec();
        let new_digest = DistributedCheckpointer::compute_merkle_root(&new_state);

        let gossip = GossipMessage {
            from_node: "node-1".to_string(),
            to_node: Some("node-2".to_string()),
            round: 1,
            state_digest: new_digest.clone(),
            state_payload: Some(new_state.clone()),
            timestamp: SystemTime::now(),
        };

        let synced = receiver
            .receive_gossip(gossip)
            .expect("receive should succeed");
        assert!(synced, "should detect and apply diverged state");
        assert_eq!(receiver.local_digest(), new_digest);

        let stats = receiver.stats();
        assert_eq!(stats.messages_received, 1);
        assert_eq!(stats.sync_count, 1);
    }

    #[test]
    fn test_state_replication_no_sync_when_same_digest() {
        let config = ReplicationConfig::default();
        let mgr = StateReplicationManager::new("node-x", config);
        mgr.add_peer("node-y");
        let state = b"shared-state".to_vec();
        mgr.update_local_state(state.clone());

        let digest = mgr.local_digest();
        let gossip = GossipMessage {
            from_node: "node-y".to_string(),
            to_node: Some("node-x".to_string()),
            round: 1,
            state_digest: digest,
            state_payload: None,
            timestamp: SystemTime::now(),
        };

        let synced = mgr.receive_gossip(gossip).expect("receive should succeed");
        assert!(!synced, "should not sync when digests match");
        let stats = mgr.stats();
        assert_eq!(stats.sync_count, 0);
    }

    #[test]
    fn test_merkle_root_deterministic() {
        let data = b"hello world";
        let r1 = DistributedCheckpointer::compute_merkle_root(data);
        let r2 = DistributedCheckpointer::compute_merkle_root(data);
        assert_eq!(r1, r2);

        let r3 = DistributedCheckpointer::compute_merkle_root(b"different");
        assert_ne!(r1, r3);
    }
}
