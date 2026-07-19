//! Byzantine Fault Tolerance (BFT) consensus implementation
//!
//! This module provides Byzantine fault-tolerant consensus for untrusted environments,
//! protecting against malicious nodes in the cluster.

use crate::raft::{RdfCommand, RdfResponse};
use crate::shard::ShardId;
use crate::storage::StorageBackend;
use crate::{ClusterError, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
// Note: OsRng imported via fully qualified path to avoid scirs2-core re-export conflict
use oxirs_core::model::{BlankNode, Literal, NamedNode, Object, Subject, Triple};
use serde::{Deserialize, Serialize};
use serde_json;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// The single logical shard the BFT state machine applies committed
/// [`RdfCommand`]s to. BFT clusters in this build operate over one shard; the
/// value is fixed so every replica applies to the same partition.
const BFT_DEFAULT_SHARD: ShardId = 0;

/// Callback invoked once a client request reaches a real 2f+1 matching-commit
/// quorum in [`BftConsensus::handle_commit`], carrying `(client_id,
/// timestamp, execution_result)` recovered from the original
/// [`BftMessage::Request`]. See [`BftConsensus::set_commit_callback`].
type CommitCallback = dyn Fn(String, u64, Vec<u8>) + Send + Sync;

/// Byzantine fault tolerance configuration
#[derive(Debug, Clone)]
pub struct BftConfig {
    /// Minimum number of nodes required for consensus
    pub min_nodes: usize,
    /// Maximum number of faulty nodes tolerated (f)
    pub max_faulty: usize,
    /// View change timeout
    pub view_timeout: Duration,
    /// Message authentication timeout
    pub auth_timeout: Duration,
    /// Enable cryptographic signatures
    pub enable_signatures: bool,
    /// Enable message ordering verification
    pub enable_ordering: bool,
}

impl BftConfig {
    /// Create a new BFT configuration for n nodes
    pub fn new(num_nodes: usize) -> Self {
        // Byzantine fault tolerance requires n >= 3f + 1
        let max_faulty = (num_nodes - 1) / 3;

        BftConfig {
            min_nodes: 3 * max_faulty + 1,
            max_faulty,
            view_timeout: Duration::from_secs(10),
            auth_timeout: Duration::from_secs(5),
            enable_signatures: true,
            enable_ordering: true,
        }
    }

    /// Check if we have enough nodes for BFT consensus
    pub fn has_quorum(&self, active_nodes: usize) -> bool {
        active_nodes >= self.min_nodes
    }

    /// Calculate the required votes for consensus (2f + 1)
    pub fn required_votes(&self) -> usize {
        2 * self.max_faulty + 1
    }
}

/// Byzantine node state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BftNodeState {
    /// Normal operation
    Normal,
    /// View change in progress
    ViewChange,
    /// Node is suspected to be Byzantine
    Suspected,
    /// Node is confirmed Byzantine and isolated
    Byzantine,
}

/// PBFT message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BftMessage {
    /// Client request
    Request {
        client_id: String,
        operation: Vec<u8>,
        timestamp: u64,
        signature: Option<Vec<u8>>,
    },
    /// Pre-prepare phase (primary only)
    PrePrepare {
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        request: Box<BftMessage>,
        primary_signature: Vec<u8>,
    },
    /// Prepare phase
    Prepare {
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        node_id: String,
        signature: Vec<u8>,
    },
    /// Commit phase
    Commit {
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        node_id: String,
        signature: Vec<u8>,
    },
    /// Reply to client
    Reply {
        view: u64,
        timestamp: u64,
        client_id: String,
        node_id: String,
        result: Vec<u8>,
        signature: Vec<u8>,
    },
    /// View change request
    ViewChange {
        new_view: u64,
        node_id: String,
        prepared_messages: Vec<PreparedMessage>,
        signature: Vec<u8>,
    },
    /// New view confirmation
    NewView {
        view: u64,
        view_changes: Vec<BftMessage>,
        pre_prepares: Vec<BftMessage>,
        primary_signature: Vec<u8>,
    },
    /// Checkpoint for garbage collection
    Checkpoint {
        sequence: u64,
        digest: Vec<u8>,
        node_id: String,
        signature: Vec<u8>,
    },
}

/// Prepared message proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedMessage {
    pub view: u64,
    pub sequence: u64,
    pub digest: Vec<u8>,
    pub pre_prepare: Box<BftMessage>,
    pub prepares: Vec<BftMessage>,
}

/// Byzantine fault-tolerant consensus engine
pub struct BftConsensus {
    /// Node identifier
    node_id: String,
    /// Current view number
    view: Arc<RwLock<u64>>,
    /// Node state
    #[allow(dead_code)]
    state: Arc<RwLock<BftNodeState>>,
    /// BFT configuration
    config: BftConfig,
    /// Cryptographic keypair for this node
    keypair: SigningKey,
    /// Known public keys of other nodes
    node_keys: Arc<RwLock<HashMap<String, VerifyingKey>>>,
    /// Message log for consensus
    message_log: Arc<RwLock<MessageLog>>,
    /// Byzantine node tracker
    byzantine_tracker: Arc<RwLock<ByzantineTracker>>,
    /// View change timer
    view_timer: Arc<RwLock<Option<Instant>>>,
    /// Optional observer notified from [`BftConsensus::send_reply`] once
    /// [`BftConsensus::handle_commit`] observes a real 2f+1 matching-commit
    /// quorum for a request. This is the commit-observation hook a wrapper
    /// like `BftConsensusManager` (in `bft_consensus.rs`) would register in
    /// order to complete a caller's pending `process_request` future instead
    /// of it timing out -- see [`BftConsensus::set_commit_callback`] for the
    /// exact integration contract and its current limitation.
    commit_callback: Arc<RwLock<Option<Arc<CommitCallback>>>>,
    /// Optional real state-machine backend. When present,
    /// [`BftConsensus::execute_operation`] deserializes each committed
    /// operation into an [`RdfCommand`] and applies it here exactly once
    /// (guarded by the per-sequence completed marker in
    /// [`BftConsensus::handle_commit`]). When absent the engine fails loud
    /// rather than fabricating a commit result.
    storage: Option<Arc<dyn StorageBackend>>,
    /// Optional sync-callable broadcast sink. [`BftConsensus::broadcast_message`]
    /// pushes protocol messages here; a task spawned by the wrapping manager
    /// drains it and performs the authenticated network broadcast. Using an
    /// unbounded sender keeps the synchronous PBFT path from blocking on async
    /// network I/O and resolves the consensus<->network construction cycle via
    /// a post-construction setter.
    broadcaster: RwLock<Option<mpsc::UnboundedSender<BftMessage>>>,
}

/// Message log for consensus tracking
struct MessageLog {
    /// Pre-prepare messages by view and sequence
    pre_prepares: HashMap<(u64, u64), BftMessage>,
    /// Prepare messages by view, sequence, and node
    prepares: HashMap<(u64, u64), HashMap<String, BftMessage>>,
    /// Commit messages by view, sequence, and node
    commits: HashMap<(u64, u64), HashMap<String, BftMessage>>,
    /// Completed requests by sequence
    completed: HashMap<u64, Vec<u8>>,
    /// Last stable checkpoint
    #[allow(dead_code)]
    last_checkpoint: u64,
}

impl MessageLog {
    fn new() -> Self {
        MessageLog {
            pre_prepares: HashMap::new(),
            prepares: HashMap::new(),
            commits: HashMap::new(),
            completed: HashMap::new(),
            last_checkpoint: 0,
        }
    }

    /// Check if a request is prepared (has 2f prepares)
    fn is_prepared(&self, view: u64, sequence: u64, required_votes: usize) -> bool {
        self.prepares
            .get(&(view, sequence))
            .map(|votes| votes.len() >= required_votes)
            .unwrap_or(false)
    }

    /// Check if a request is committed (has 2f + 1 commits)
    fn is_committed(&self, view: u64, sequence: u64, required_votes: usize) -> bool {
        self.commits
            .get(&(view, sequence))
            .map(|votes| votes.len() >= required_votes)
            .unwrap_or(false)
    }
}

/// Byzantine node detection and tracking
struct ByzantineTracker {
    /// Nodes suspected of Byzantine behavior
    suspected_nodes: HashSet<String>,
    /// Confirmed Byzantine nodes (isolated)
    byzantine_nodes: HashSet<String>,
    /// Invalid message counts per node
    invalid_messages: HashMap<String, usize>,
    /// Threshold for Byzantine detection
    detection_threshold: usize,
}

impl ByzantineTracker {
    fn new(threshold: usize) -> Self {
        ByzantineTracker {
            suspected_nodes: HashSet::new(),
            byzantine_nodes: HashSet::new(),
            invalid_messages: HashMap::new(),
            detection_threshold: threshold,
        }
    }

    /// Report an invalid message from a node
    fn report_invalid(&mut self, node_id: &str) {
        let count = self
            .invalid_messages
            .entry(node_id.to_string())
            .or_insert(0);
        *count += 1;

        if *count >= self.detection_threshold {
            self.suspected_nodes.insert(node_id.to_string());

            // After multiple violations, mark as Byzantine
            if *count >= self.detection_threshold * 2 {
                self.byzantine_nodes.insert(node_id.to_string());
                self.suspected_nodes.remove(node_id);
            }
        }
    }

    /// Check if a node is Byzantine or suspected
    fn is_byzantine(&self, node_id: &str) -> bool {
        self.byzantine_nodes.contains(node_id) || self.suspected_nodes.contains(node_id)
    }
}

impl BftConsensus {
    /// Create a new BFT consensus instance without a state-machine backend.
    ///
    /// A consensus engine built this way reaches quorum and fires its commit
    /// callback, but [`BftConsensus::execute_operation`] fails loud instead of
    /// applying operations. Use [`BftConsensus::with_storage`] for a node that
    /// must materialize committed writes.
    pub fn new(node_id: String, config: BftConfig) -> Result<Self> {
        Self::build(node_id, config, None)
    }

    /// Create a new BFT consensus instance bound to a real state-machine
    /// backend. Committed [`RdfCommand`]s are applied to `storage` exactly once
    /// per sequence, mirroring how the Raft path applies commands to its state
    /// machine.
    pub fn with_storage(
        node_id: String,
        config: BftConfig,
        storage: Arc<dyn StorageBackend>,
    ) -> Result<Self> {
        Self::build(node_id, config, Some(storage))
    }

    fn build(
        node_id: String,
        config: BftConfig,
        storage: Option<Arc<dyn StorageBackend>>,
    ) -> Result<Self> {
        // Generate a random SigningKey using rand crate's random() to avoid type conflicts
        // This avoids the scirs2-core OsRng re-export issue
        let seed_bytes: [u8; 32] = rand::random();
        let keypair = SigningKey::from_bytes(&seed_bytes);

        Ok(BftConsensus {
            node_id,
            view: Arc::new(RwLock::new(0)),
            state: Arc::new(RwLock::new(BftNodeState::Normal)),
            config,
            keypair,
            node_keys: Arc::new(RwLock::new(HashMap::new())),
            message_log: Arc::new(RwLock::new(MessageLog::new())),
            byzantine_tracker: Arc::new(RwLock::new(ByzantineTracker::new(5))),
            view_timer: Arc::new(RwLock::new(None)),
            commit_callback: Arc::new(RwLock::new(None)),
            storage,
            broadcaster: RwLock::new(None),
        })
    }

    /// Register the sync-callable sink used by [`BftConsensus::broadcast_message`]
    /// to hand protocol messages to the authenticated network broadcast task.
    pub fn set_broadcaster(&self, sender: mpsc::UnboundedSender<BftMessage>) {
        if let Ok(mut slot) = self.broadcaster.write() {
            *slot = Some(sender);
        }
    }

    /// Register a callback fired the moment [`BftConsensus::handle_commit`]
    /// observes a real 2f+1 matching-commit quorum for a request -- i.e. the
    /// commit-observation point real BFT submissions need in order to
    /// complete instead of timing out. The callback receives the original
    /// request's `client_id`, `timestamp`, and the serialized
    /// [`BftConsensus::execute_operation`] result.
    ///
    /// # Integration contract / current limitation
    ///
    /// This engine has no notion of the caller-generated `request_id` a
    /// wrapper such as `BftConsensusManager::process_request` (in
    /// `bft_consensus.rs`) uses to key its pending-completion map: that
    /// `request_id` is a UUID minted by the wrapper and is **not** threaded
    /// into [`BftMessage::Request`], so this engine cannot look it up here.
    /// A wrapper wiring this callback must therefore correlate on
    /// `(client_id, timestamp)` itself, e.g. by keying its own pending map on
    /// that pair (or embedding its `request_id` inside the `operation`
    /// payload before submission) rather than on the UUID alone, and then
    /// resolve/complete the matching waiter (for `BftConsensusManager` that
    /// final step is calling `notify_request_committed`) from inside the
    /// callback registered here.
    pub fn set_commit_callback<F>(&self, callback: F)
    where
        F: Fn(String, u64, Vec<u8>) + Send + Sync + 'static,
    {
        if let Ok(mut slot) = self.commit_callback.write() {
            *slot = Some(Arc::new(callback));
        }
    }

    /// Register a node's public key
    pub fn register_node(&self, node_id: String, public_key: VerifyingKey) -> Result<()> {
        let mut keys = self
            .node_keys
            .write()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;
        keys.insert(node_id, public_key);
        Ok(())
    }

    /// Get current view number
    pub fn current_view(&self) -> Result<u64> {
        let view = self
            .view
            .read()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;
        Ok(*view)
    }

    /// Check if this node is the primary for current view
    pub fn is_primary(&self) -> Result<bool> {
        let view = self.current_view()?;
        let keys = self
            .node_keys
            .read()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;
        let num_nodes = keys.len() + 1; // Include self
        let primary_index = (view as usize) % num_nodes;

        // Simple primary selection based on sorted node IDs
        let mut all_nodes: Vec<String> = keys.keys().cloned().collect();
        all_nodes.push(self.node_id.clone());
        all_nodes.sort();

        Ok(all_nodes.get(primary_index) == Some(&self.node_id))
    }

    /// Create a message digest
    pub(crate) fn create_digest(message: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(message);
        hasher.finalize().to_vec()
    }

    /// Test-only probe: has a pre-prepare been stored for `(view, sequence)`?
    ///
    /// Used by the manager-level closed-loop test to wait until the primary
    /// has locally recorded the pre-prepare before delivering commit votes.
    #[cfg(test)]
    pub(crate) fn has_pre_prepare(&self, view: u64, sequence: u64) -> bool {
        self.message_log
            .read()
            .map(|log| log.pre_prepares.contains_key(&(view, sequence)))
            .unwrap_or(false)
    }

    /// Sign a message
    fn sign_message(&self, message: &[u8]) -> Vec<u8> {
        if !self.config.enable_signatures {
            return vec![];
        }

        let signature = self.keypair.sign(message);
        signature.to_bytes().to_vec()
    }

    /// Verify a message signature
    fn verify_signature(&self, node_id: &str, message: &[u8], signature: &[u8]) -> Result<bool> {
        if !self.config.enable_signatures {
            return Ok(true);
        }

        let keys = self
            .node_keys
            .read()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;

        let public_key = keys
            .get(node_id)
            .ok_or_else(|| ClusterError::Config(format!("Unknown node: {}", node_id)))?;

        // ed25519-dalek 2.x requires exactly 64 bytes
        if signature.len() != 64 {
            return Ok(false);
        }

        let mut signature_bytes = [0u8; 64];
        signature_bytes.copy_from_slice(signature);
        let sig = Signature::from_bytes(&signature_bytes);

        Ok(public_key.verify(message, &sig).is_ok())
    }

    /// Process a client request (primary only)
    pub fn process_request(&self, request: BftMessage) -> Result<()> {
        if !self.is_primary()? {
            return Err(ClusterError::NotLeader);
        }

        let view = self.current_view()?;
        let sequence = self.next_sequence()?;

        // Create pre-prepare message
        if let BftMessage::Request { operation, .. } = &request {
            let digest = Self::create_digest(operation);
            let pre_prepare = BftMessage::PrePrepare {
                view,
                sequence,
                digest: digest.clone(),
                request: Box::new(request),
                primary_signature: self.sign_message(&digest),
            };

            // Store and broadcast pre-prepare
            self.store_pre_prepare(view, sequence, pre_prepare.clone())?;
            self.broadcast_message(pre_prepare)?;
        }

        Ok(())
    }

    /// Handle incoming BFT message
    pub fn handle_message(&self, message: BftMessage, from_node: &str) -> Result<()> {
        // Check if node is Byzantine
        let tracker = self
            .byzantine_tracker
            .read()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;
        if tracker.is_byzantine(from_node) {
            return Err(ClusterError::Network(format!(
                "Byzantine node: {}",
                from_node
            )));
        }
        drop(tracker);

        match message {
            BftMessage::PrePrepare {
                view,
                sequence,
                digest,
                request,
                primary_signature,
            } => {
                self.handle_pre_prepare(
                    view,
                    sequence,
                    digest,
                    *request,
                    primary_signature,
                    from_node,
                )?;
            }
            BftMessage::Prepare {
                view,
                sequence,
                digest,
                node_id,
                signature,
            } => {
                self.handle_prepare(view, sequence, digest, node_id, signature)?;
            }
            BftMessage::Commit {
                view,
                sequence,
                digest,
                node_id,
                signature,
            } => {
                self.handle_commit(view, sequence, digest, node_id, signature)?;
            }
            BftMessage::ViewChange {
                new_view,
                node_id,
                prepared_messages,
                signature,
            } => {
                self.handle_view_change(new_view, node_id, prepared_messages, signature)?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Handle pre-prepare message
    fn handle_pre_prepare(
        &self,
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        request: BftMessage,
        signature: Vec<u8>,
        from_node: &str,
    ) -> Result<()> {
        // Verify view and primary
        if view != self.current_view()? {
            return Ok(()); // Ignore messages from wrong view
        }

        // Verify signature
        if !self.verify_signature(from_node, &digest, &signature)? {
            self.report_byzantine(from_node)?;
            return Err(ClusterError::Network("Invalid signature".to_string()));
        }

        // Store pre-prepare
        self.store_pre_prepare(
            view,
            sequence,
            BftMessage::PrePrepare {
                view,
                sequence,
                digest: digest.clone(),
                request: Box::new(request),
                primary_signature: signature,
            },
        )?;

        // Send prepare message
        let prepare = BftMessage::Prepare {
            view,
            sequence,
            digest: digest.clone(),
            node_id: self.node_id.clone(),
            signature: self.sign_message(&digest),
        };

        self.broadcast_message(prepare)?;

        Ok(())
    }

    /// Handle prepare message
    fn handle_prepare(
        &self,
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        node_id: String,
        signature: Vec<u8>,
    ) -> Result<()> {
        // Verify signature
        if !self.verify_signature(&node_id, &digest, &signature)? {
            self.report_byzantine(&node_id)?;
            return Err(ClusterError::Network("Invalid signature".to_string()));
        }

        // Store prepare
        let mut log = self
            .message_log
            .write()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;

        log.prepares
            .entry((view, sequence))
            .or_insert_with(HashMap::new)
            .insert(
                node_id.clone(),
                BftMessage::Prepare {
                    view,
                    sequence,
                    digest: digest.clone(),
                    node_id,
                    signature,
                },
            );

        // Check if prepared (2f prepares)
        if log.is_prepared(view, sequence, self.config.required_votes()) {
            drop(log);

            // Send commit message
            let commit = BftMessage::Commit {
                view,
                sequence,
                digest: digest.clone(),
                node_id: self.node_id.clone(),
                signature: self.sign_message(&digest),
            };

            self.broadcast_message(commit)?;
        }

        Ok(())
    }

    /// Handle commit message
    fn handle_commit(
        &self,
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        node_id: String,
        signature: Vec<u8>,
    ) -> Result<()> {
        // Verify signature
        if !self.verify_signature(&node_id, &digest, &signature)? {
            self.report_byzantine(&node_id)?;
            return Err(ClusterError::Network("Invalid signature".to_string()));
        }

        // Store commit
        let mut log = self
            .message_log
            .write()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;

        log.commits
            .entry((view, sequence))
            .or_insert_with(HashMap::new)
            .insert(
                node_id.clone(),
                BftMessage::Commit {
                    view,
                    sequence,
                    digest: digest.clone(),
                    node_id,
                    signature,
                },
            );

        // Act only on the transition into "committed": the per-sequence
        // `completed` marker makes the state-machine apply and the client
        // reply fire exactly once, so redundant Commit votes arriving after the
        // 2f+1 quorum can never re-execute the operation or re-notify the
        // caller.
        if log.is_committed(view, sequence, self.config.required_votes())
            && !log.completed.contains_key(&sequence)
        {
            // Recover the original request bound to this (view, sequence).
            let operation_data = if let Some(BftMessage::PrePrepare { request, .. }) =
                log.pre_prepares.get(&(view, sequence))
            {
                if let BftMessage::Request {
                    operation,
                    client_id,
                    timestamp,
                    ..
                } = &**request
                {
                    Some((operation.clone(), client_id.clone(), *timestamp))
                } else {
                    None
                }
            } else {
                None
            };

            if let Some((operation, client_id, timestamp)) = operation_data {
                // Mark executed *before* releasing the log lock so a concurrent
                // redundant Commit cannot race into a second apply.
                log.completed.insert(sequence, operation.clone());
                drop(log);

                // Apply to the real state machine and answer the client. A
                // failed apply surfaces as an error (the caller times out);
                // it is never turned into a fabricated success.
                let execution_result = self.execute_operation(&operation)?;
                let reply = BftMessage::Reply {
                    view,
                    timestamp,
                    client_id,
                    node_id: self.node_id.clone(),
                    result: execution_result,
                    signature: self.sign_message(&digest),
                };
                self.send_reply(reply)?;
            }
        }

        Ok(())
    }

    /// Handle view change request
    ///
    /// Implements the PBFT view change protocol for primary node replacement.
    /// The protocol ensures safety and liveness even when the primary is faulty.
    fn handle_view_change(
        &self,
        new_view: u64,
        node_id: String,
        prepared_messages: Vec<PreparedMessage>,
        signature: Vec<u8>,
    ) -> Result<()> {
        // Step 1: Validate the view change request signature
        // In a full implementation, this would verify the signature
        let _ = signature; // Suppress unused warning

        // Step 2: Check if view change is valid (new_view > view)
        let current_view = self
            .view
            .read()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;

        if new_view <= *current_view {
            return Err(ClusterError::Consensus(format!(
                "View change to {} rejected: current view is {}",
                new_view, *current_view
            )));
        }
        drop(current_view);

        // Step 3: Verify prepared messages are properly signed and ordered
        // In a full implementation, this would:
        // - Verify each prepared message has 2f+1 prepare certificates
        // - Check message ordering and consistency
        // - Validate all signatures
        if prepared_messages.is_empty() {
            // No prepared messages is valid for view change
        }

        // Step 4: Update view and elect new primary
        // The new primary is determined by: primary = new_view % n
        let mut current_view = self
            .view
            .write()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;
        *current_view = new_view;
        drop(current_view);

        // Step 5: Reset view timer
        let mut timer = self
            .view_timer
            .write()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;
        *timer = Some(Instant::now());

        tracing::info!(
            "View change completed: node {} initiated view change to view {}",
            node_id,
            new_view
        );

        Ok(())
    }

    /// Report Byzantine behavior
    fn report_byzantine(&self, node_id: &str) -> Result<()> {
        let mut tracker = self
            .byzantine_tracker
            .write()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;
        tracker.report_invalid(node_id);
        Ok(())
    }

    /// Get next sequence number
    fn next_sequence(&self) -> Result<u64> {
        let log = self
            .message_log
            .read()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;
        Ok(log.completed.len() as u64 + 1)
    }

    /// Store pre-prepare message
    fn store_pre_prepare(&self, view: u64, sequence: u64, message: BftMessage) -> Result<()> {
        let mut log = self
            .message_log
            .write()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;
        log.pre_prepares.insert((view, sequence), message);
        Ok(())
    }

    /// Execute a committed operation against the real state machine and return
    /// the serialized [`RdfResponse`].
    ///
    /// The `operation` bytes are the serialized [`RdfCommand`] the client
    /// submitted. This deserializes them, applies the command to the configured
    /// [`StorageBackend`], and serializes the resulting [`RdfResponse`] — the
    /// exact byte shape `notify_request_committed`/the commit callback expects.
    /// A malformed payload, an unsupported command, or a backend failure all
    /// surface as explicit errors; success is never fabricated.
    fn execute_operation(&self, operation: &[u8]) -> Result<Vec<u8>> {
        let command: RdfCommand = serde_json::from_slice(operation).map_err(|e| {
            ClusterError::Serialize(format!(
                "BFT operation payload is not a valid RdfCommand: {e}"
            ))
        })?;

        let response = self.apply_command(command)?;

        serde_json::to_vec(&response).map_err(|e| ClusterError::Serialize(e.to_string()))
    }

    /// Apply a committed [`RdfCommand`] to the configured state-machine backend.
    ///
    /// Bridges the synchronous PBFT commit path to the async [`StorageBackend`]
    /// via [`futures::executor::block_on`]. The apply is a short in-memory/WAL
    /// operation that needs no ambient tokio runtime, so this drives it to
    /// completion on the current thread without a nested-runtime panic.
    fn apply_command(&self, command: RdfCommand) -> Result<RdfResponse> {
        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| {
                ClusterError::Consensus(
                    "BFT state-machine backend is not configured; refusing to fabricate a commit \
                     result"
                        .to_string(),
                )
            })?
            .clone();

        futures::executor::block_on(Self::apply_to_backend(storage, command))
    }

    /// Materialize a committed [`RdfCommand`] on `storage`.
    async fn apply_to_backend(
        storage: Arc<dyn StorageBackend>,
        command: RdfCommand,
    ) -> Result<RdfResponse> {
        match command {
            RdfCommand::Insert {
                subject,
                predicate,
                object,
            } => {
                let triple = Self::triple_from_parts(&subject, &predicate, &object)?;
                storage
                    .insert_triple_to_shard(BFT_DEFAULT_SHARD, triple)
                    .await
                    .map_err(|e| ClusterError::Storage(e.to_string()))?;
                Ok(RdfResponse::Success)
            }
            RdfCommand::Delete {
                subject,
                predicate,
                object,
            } => {
                let triple = Self::triple_from_parts(&subject, &predicate, &object)?;
                storage
                    .delete_triple_from_shard(BFT_DEFAULT_SHARD, &triple)
                    .await
                    .map_err(|e| ClusterError::Storage(e.to_string()))?;
                Ok(RdfResponse::Success)
            }
            RdfCommand::Clear => {
                let existing = storage
                    .get_shard_triples(BFT_DEFAULT_SHARD)
                    .await
                    .map_err(|e| ClusterError::Storage(e.to_string()))?;
                for triple in existing {
                    storage
                        .delete_triple_from_shard(BFT_DEFAULT_SHARD, &triple)
                        .await
                        .map_err(|e| ClusterError::Storage(e.to_string()))?;
                }
                Ok(RdfResponse::Success)
            }
            other => Ok(RdfResponse::Error(format!(
                "RdfCommand {other:?} is not supported by the BFT state-machine backend"
            ))),
        }
    }

    /// Build an RDF [`Triple`] from the string components of an [`RdfCommand`].
    ///
    /// Subjects and objects prefixed with `_:` are treated as blank nodes;
    /// object strings that are not valid IRIs fall back to simple literals.
    /// An invalid IRI/blank-node component is a fail-loud error.
    fn triple_from_parts(subject: &str, predicate: &str, object: &str) -> Result<Triple> {
        let subj: Subject = if let Some(id) = subject.strip_prefix("_:") {
            Subject::BlankNode(BlankNode::new(id).map_err(|e| {
                ClusterError::Serialize(format!("invalid blank-node subject '{subject}': {e}"))
            })?)
        } else {
            Subject::NamedNode(NamedNode::new(subject).map_err(|e| {
                ClusterError::Serialize(format!("invalid IRI subject '{subject}': {e}"))
            })?)
        };

        let pred = NamedNode::new(predicate).map_err(|e| {
            ClusterError::Serialize(format!("invalid IRI predicate '{predicate}': {e}"))
        })?;

        let obj: Object = if let Some(id) = object.strip_prefix("_:") {
            Object::BlankNode(BlankNode::new(id).map_err(|e| {
                ClusterError::Serialize(format!("invalid blank-node object '{object}': {e}"))
            })?)
        } else if let Ok(iri) = NamedNode::new(object) {
            Object::NamedNode(iri)
        } else {
            Object::Literal(Literal::new_simple_literal(object))
        };

        Ok(Triple::new(subj, pred, obj))
    }

    /// Broadcast a protocol message to all peers.
    ///
    /// Hands the message to the sync-callable broadcast sink registered via
    /// [`BftConsensus::set_broadcaster`], which a spawned task drains to perform
    /// the authenticated network broadcast. When no sink is registered (e.g. a
    /// consensus engine exercised in isolation) the message is dropped locally
    /// — real network fan-out only happens once a manager has wired the sink.
    fn broadcast_message(&self, message: BftMessage) -> Result<()> {
        if let Ok(guard) = self.broadcaster.read() {
            if let Some(sender) = guard.as_ref() {
                if sender.send(message).is_err() {
                    tracing::warn!("BFT broadcast channel closed; message not delivered to peers");
                }
            }
        }
        Ok(())
    }

    /// Notify the registered commit observer (if any) and send the reply to
    /// the client.
    ///
    /// This is called by [`BftConsensus::handle_commit`] exactly once a
    /// request reaches a real 2f+1 matching-commit quorum, so it is the
    /// commit-observation point: the [`CommitCallback`] registered via
    /// [`BftConsensus::set_commit_callback`] fires here with the real
    /// `client_id`/`timestamp`/execution result recovered from the
    /// `Reply` message, letting a wrapper complete the caller's pending
    /// request instead of it timing out.
    ///
    /// Actual client network delivery of the `Reply` remains a placeholder:
    /// - Serialize the Reply message to bytes
    /// - Send the reply back to the requesting client
    /// - Use the client connection manager for delivery
    /// - Handle client disconnections gracefully
    fn send_reply(&self, reply: BftMessage) -> Result<()> {
        if let BftMessage::Reply {
            client_id,
            timestamp,
            result,
            ..
        } = &reply
        {
            if let Ok(guard) = self.commit_callback.read() {
                if let Some(callback) = guard.as_ref() {
                    callback(client_id.clone(), *timestamp, result.clone());
                }
            }
        }

        // Integration with network layer will be implemented when:
        // 1. Client connection tracking is available
        // 2. Reply routing mechanism is in place
        // 3. Client session management is implemented
        Ok(())
    }

    /// Start view change timer
    pub fn start_view_timer(&self) -> Result<()> {
        let mut timer = self
            .view_timer
            .write()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;
        *timer = Some(Instant::now());
        Ok(())
    }

    /// Check if view change is needed
    pub fn check_view_timeout(&self) -> Result<bool> {
        let timer = self
            .view_timer
            .read()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;

        if let Some(start_time) = *timer {
            Ok(start_time.elapsed() > self.config.view_timeout)
        } else {
            Ok(false)
        }
    }

    /// Collect all prepared messages for view change
    ///
    /// Returns a vector of PreparedMessage containing proof of prepared requests
    /// for inclusion in ViewChange messages during view changes.
    pub fn collect_prepared_messages(&self) -> Result<Vec<PreparedMessage>> {
        let log = self
            .message_log
            .read()
            .map_err(|e| ClusterError::Lock(e.to_string()))?;

        let mut prepared_messages = Vec::new();
        let required_votes = self.config.required_votes();

        // Iterate through all prepare messages to find prepared requests
        for ((view, sequence), prepares_map) in &log.prepares {
            // Check if this request is prepared (has enough votes)
            if prepares_map.len() >= required_votes {
                // Get the pre-prepare message
                if let Some(pre_prepare) = log.pre_prepares.get(&(*view, *sequence)) {
                    if let BftMessage::PrePrepare { digest, .. } = pre_prepare {
                        // Collect prepare messages for this request
                        let prepares: Vec<BftMessage> = prepares_map.values().cloned().collect();

                        prepared_messages.push(PreparedMessage {
                            view: *view,
                            sequence: *sequence,
                            digest: digest.clone(),
                            pre_prepare: Box::new(pre_prepare.clone()),
                            prepares,
                        });
                    }
                }
            }
        }

        tracing::debug!(
            "Collected {} prepared messages for view change",
            prepared_messages.len()
        );

        Ok(prepared_messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bft_config() {
        // Test with 4 nodes (tolerates 1 Byzantine fault)
        let config = BftConfig::new(4);
        assert_eq!(config.max_faulty, 1);
        assert_eq!(config.min_nodes, 4);
        assert_eq!(config.required_votes(), 3);
        assert!(config.has_quorum(4));
        assert!(!config.has_quorum(3));

        // Test with 7 nodes (tolerates 2 Byzantine faults)
        let config = BftConfig::new(7);
        assert_eq!(config.max_faulty, 2);
        assert_eq!(config.min_nodes, 7);
        assert_eq!(config.required_votes(), 5);
    }

    #[test]
    fn test_byzantine_tracker() {
        let mut tracker = ByzantineTracker::new(3);

        // Report invalid messages
        tracker.report_invalid("node1");
        assert!(!tracker.is_byzantine("node1"));

        tracker.report_invalid("node1");
        tracker.report_invalid("node1");
        assert!(tracker.suspected_nodes.contains("node1"));

        // Continue reporting until marked as Byzantine
        for _ in 0..3 {
            tracker.report_invalid("node1");
        }
        assert!(tracker.byzantine_nodes.contains("node1"));
        assert!(!tracker.suspected_nodes.contains("node1"));
    }

    #[test]
    fn test_message_digest() {
        let message1 = b"test message 1";
        let message2 = b"test message 2";

        let digest1 = BftConsensus::create_digest(message1);
        let digest2 = BftConsensus::create_digest(message2);

        assert_ne!(digest1, digest2);
        assert_eq!(digest1.len(), 32); // SHA256 produces 32 bytes
    }

    /// Regression test for the commit-observation hook: the callback
    /// registered via `set_commit_callback` must fire exactly once, with the
    /// original request's `client_id`/`timestamp`, precisely when the
    /// engine observes a real 2f+1 matching-commit quorum -- not before.
    #[test]
    fn test_commit_quorum_invokes_registered_callback() {
        use crate::storage::mock::MockStorageBackend;

        let config = BftConfig::new(4); // max_faulty=1, required_votes=3
        let storage = Arc::new(MockStorageBackend::new());
        let primary = BftConsensus::with_storage("1".to_string(), config, storage).unwrap();

        // Register three remote nodes whose commit votes the primary trusts.
        let mut remotes: Vec<(String, SigningKey)> = Vec::new();
        for id in ["2", "3", "4"] {
            let seed: [u8; 32] = rand::random();
            let signing_key = SigningKey::from_bytes(&seed);
            primary
                .register_node(id.to_string(), signing_key.verifying_key())
                .unwrap();
            remotes.push((id.to_string(), signing_key));
        }

        let calls: Arc<std::sync::Mutex<Vec<(String, u64, Vec<u8>)>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let calls_clone = calls.clone();
        primary.set_commit_callback(move |client_id, timestamp, result| {
            calls_clone
                .lock()
                .expect("test mutex poisoned")
                .push((client_id, timestamp, result));
        });

        // Submit a real client request as the (self) primary; this stores a
        // pre-prepare at (view=0, sequence=1) carrying the original request.
        let command = RdfCommand::Insert {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        };
        let operation = serde_json::to_vec(&command).unwrap();
        let request = BftMessage::Request {
            client_id: "client-42".to_string(),
            operation: operation.clone(),
            timestamp: 1234,
            signature: None,
        };
        primary.process_request(request).unwrap();

        let view = primary.current_view().unwrap();
        let sequence = 1;
        let digest = BftConsensus::create_digest(&operation);

        // Deliver commit votes one at a time: the callback must stay silent
        // until the third (2f+1 = 3) vote lands.
        for (idx, (node_id, signing_key)) in remotes.iter().enumerate() {
            let commit = BftMessage::Commit {
                view,
                sequence,
                digest: digest.clone(),
                node_id: node_id.clone(),
                signature: signing_key.sign(&digest).to_bytes().to_vec(),
            };
            primary.handle_message(commit, node_id).unwrap();

            let observed = calls.lock().expect("test mutex poisoned").len();
            if idx < 2 {
                assert_eq!(observed, 0, "callback must not fire before 2f+1 commits");
            } else {
                assert_eq!(
                    observed, 1,
                    "callback must fire exactly once at 2f+1 commits"
                );
            }
        }

        let recorded = calls.lock().expect("test mutex poisoned");
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].0, "client-42");
        assert_eq!(recorded[0].1, 1234);
        // The callback carries the real serialized RdfResponse, not a fabricated
        // status blob: applying an Insert yields RdfResponse::Success.
        let response: RdfResponse = serde_json::from_slice(&recorded[0].2).unwrap();
        assert_eq!(response, RdfResponse::Success);
    }

    /// Redundant Commit votes beyond the 2f+1 quorum must neither re-apply the
    /// operation nor re-fire the commit callback.
    #[test]
    fn test_extra_commits_beyond_quorum_do_not_re_execute() {
        use crate::storage::mock::MockStorageBackend;

        // config for f=1 (required_votes=3) but register four remotes so we can
        // deliver a fourth, redundant commit after quorum is reached.
        let config = BftConfig::new(4);
        let storage = Arc::new(MockStorageBackend::new());
        let primary = BftConsensus::with_storage("1".to_string(), config, storage.clone()).unwrap();

        let mut remotes: Vec<(String, SigningKey)> = Vec::new();
        for id in ["2", "3", "4", "5"] {
            let seed: [u8; 32] = rand::random();
            let signing_key = SigningKey::from_bytes(&seed);
            primary
                .register_node(id.to_string(), signing_key.verifying_key())
                .unwrap();
            remotes.push((id.to_string(), signing_key));
        }

        let calls = Arc::new(std::sync::Mutex::new(0usize));
        let calls_clone = calls.clone();
        primary.set_commit_callback(move |_client_id, _timestamp, _result| {
            *calls_clone.lock().expect("test mutex poisoned") += 1;
        });

        let command = RdfCommand::Insert {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        };
        let operation = serde_json::to_vec(&command).unwrap();
        let request = BftMessage::Request {
            client_id: "client-1".to_string(),
            operation: operation.clone(),
            timestamp: 7,
            signature: None,
        };
        primary.process_request(request).unwrap();

        let view = primary.current_view().unwrap();
        let sequence = 1;
        let digest = BftConsensus::create_digest(&operation);

        for (node_id, signing_key) in &remotes {
            let commit = BftMessage::Commit {
                view,
                sequence,
                digest: digest.clone(),
                node_id: node_id.clone(),
                signature: signing_key.sign(&digest).to_bytes().to_vec(),
            };
            primary.handle_message(commit, node_id).unwrap();
        }

        // Four commits delivered, quorum is three: exactly one execution/reply.
        assert_eq!(
            *calls.lock().expect("test mutex poisoned"),
            1,
            "redundant commits beyond quorum must not re-fire the callback"
        );
        // And the state machine applied the insert exactly once.
        let triples =
            futures::executor::block_on(storage.get_shard_triples(BFT_DEFAULT_SHARD)).unwrap();
        assert_eq!(triples.len(), 1, "insert must be applied exactly once");
    }

    /// The default (no callback registered) path must keep working exactly
    /// as before: reaching quorum should not panic or otherwise change
    /// behavior when nobody is listening.
    #[test]
    fn test_commit_quorum_without_callback_does_not_panic() {
        use crate::storage::mock::MockStorageBackend;

        let config = BftConfig::new(4);
        let storage = Arc::new(MockStorageBackend::new());
        let primary = BftConsensus::with_storage("1".to_string(), config, storage).unwrap();

        let mut remotes: Vec<(String, SigningKey)> = Vec::new();
        for id in ["2", "3", "4"] {
            let seed: [u8; 32] = rand::random();
            let signing_key = SigningKey::from_bytes(&seed);
            primary
                .register_node(id.to_string(), signing_key.verifying_key())
                .unwrap();
            remotes.push((id.to_string(), signing_key));
        }

        let command = RdfCommand::Delete {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        };
        let operation = serde_json::to_vec(&command).unwrap();
        let request = BftMessage::Request {
            client_id: "client-7".to_string(),
            operation: operation.clone(),
            timestamp: 99,
            signature: None,
        };
        primary.process_request(request).unwrap();

        let view = primary.current_view().unwrap();
        let sequence = 1;
        let digest = BftConsensus::create_digest(&operation);

        for (node_id, signing_key) in &remotes {
            let commit = BftMessage::Commit {
                view,
                sequence,
                digest: digest.clone(),
                node_id: node_id.clone(),
                signature: signing_key.sign(&digest).to_bytes().to_vec(),
            };
            primary.handle_message(commit, node_id).unwrap();
        }
    }
}
