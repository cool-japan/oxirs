//! Byzantine Fault Tolerance (BFT) consensus implementation
//!
//! This module provides Byzantine fault-tolerant consensus for untrusted environments,
//! protecting against malicious nodes in the cluster.

use crate::{ClusterError, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

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
    /// Create a new BFT consensus instance
    pub fn new(node_id: String, config: BftConfig) -> Result<Self> {
        // Generate a random SigningKey using OsRng
        let mut csprng = OsRng;
        let mut seed_bytes = [0u8; 32];
        use rand::RngCore;
        csprng.fill_bytes(&mut seed_bytes);
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
        })
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
    fn create_digest(message: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(message);
        hasher.finalize().to_vec()
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

        // Check if committed (2f + 1 commits)
        if log.is_committed(view, sequence, self.config.required_votes()) {
            // Clone data we need before mutable borrow
            let operation_data = if let Some(pre_prepare) = log.pre_prepares.get(&(view, sequence))
            {
                if let BftMessage::PrePrepare { request, .. } = pre_prepare {
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
                }
            } else {
                None
            };

            // Now we can mutably borrow log
            if let Some((operation, client_id, timestamp)) = operation_data {
                log.completed.insert(sequence, operation.clone());

                // Send reply to client
                let reply = BftMessage::Reply {
                    view,
                    timestamp,
                    client_id,
                    node_id: self.node_id.clone(),
                    result: vec![], // TODO: Actual execution result
                    signature: self.sign_message(&digest),
                };

                drop(log);
                self.send_reply(reply)?;
            }
        }

        Ok(())
    }

    /// Handle view change request
    fn handle_view_change(
        &self,
        _new_view: u64,
        _node_id: String,
        _prepared_messages: Vec<PreparedMessage>,
        _signature: Vec<u8>,
    ) -> Result<()> {
        // TODO: Implement view change protocol
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

    /// Broadcast message to all nodes
    fn broadcast_message(&self, _message: BftMessage) -> Result<()> {
        // TODO: Integrate with network layer
        Ok(())
    }

    /// Send reply to client
    fn send_reply(&self, _reply: BftMessage) -> Result<()> {
        // TODO: Integrate with network layer
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
}
