//! Byzantine Fault Tolerance (BFT) consensus for untrusted environments
//!
//! This module implements PBFT (Practical Byzantine Fault Tolerance) for RDF stores
//! operating in untrusted environments where nodes may act maliciously.

use crate::model::{Triple, NamedNode, BlankNode, Literal};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use sha2::{Sha256, Digest};
use tokio::sync::mpsc;

/// Byzantine fault tolerance configuration
#[derive(Debug, Clone)]
pub struct BftConfig {
    /// Number of tolerated Byzantine failures (f)
    /// System can tolerate f Byzantine nodes out of 3f+1 total nodes
    pub fault_tolerance: usize,
    
    /// View change timeout
    pub view_change_timeout: Duration,
    
    /// Message timeout for consensus rounds
    pub message_timeout: Duration,
    
    /// Checkpoint interval (number of operations)
    pub checkpoint_interval: u64,
    
    /// Maximum log size before compaction
    pub max_log_size: usize,
    
    /// Enable cryptographic signatures
    pub enable_signatures: bool,
}

impl Default for BftConfig {
    fn default() -> Self {
        Self {
            fault_tolerance: 1,  // Tolerate 1 Byzantine node (requires 4 total nodes)
            view_change_timeout: Duration::from_secs(10),
            message_timeout: Duration::from_secs(5),
            checkpoint_interval: 100,
            max_log_size: 10_000,
            enable_signatures: true,
        }
    }
}

/// Node identifier in the BFT cluster
pub type NodeId = u64;

/// View number (incremented on view changes)
pub type ViewNumber = u64;

/// Sequence number for operations
pub type SequenceNumber = u64;

/// BFT consensus phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Initial phase
    Idle,
    /// Pre-prepare phase (primary broadcasts)
    PrePrepare,
    /// Prepare phase (backup nodes agree)
    Prepare,
    /// Commit phase (nodes commit)
    Commit,
    /// View change in progress
    ViewChange,
}

/// BFT message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BftMessage {
    /// Client request
    Request {
        client_id: String,
        operation: RdfOperation,
        timestamp: SystemTime,
        signature: Option<Vec<u8>>,
    },
    
    /// Pre-prepare message from primary
    PrePrepare {
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: Vec<u8>,
        request: Box<BftMessage>,
    },
    
    /// Prepare message from backups
    Prepare {
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: Vec<u8>,
        node_id: NodeId,
    },
    
    /// Commit message
    Commit {
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: Vec<u8>,
        node_id: NodeId,
    },
    
    /// Reply to client
    Reply {
        view: ViewNumber,
        sequence: SequenceNumber,
        client_id: String,
        result: OperationResult,
        timestamp: SystemTime,
    },
    
    /// Checkpoint message
    Checkpoint {
        sequence: SequenceNumber,
        state_digest: Vec<u8>,
        node_id: NodeId,
    },
    
    /// View change message
    ViewChange {
        new_view: ViewNumber,
        node_id: NodeId,
        last_sequence: SequenceNumber,
        checkpoints: Vec<CheckpointProof>,
        prepared_messages: Vec<PreparedProof>,
    },
    
    /// New view message from new primary
    NewView {
        view: ViewNumber,
        view_changes: Vec<BftMessage>,
        pre_prepares: Vec<BftMessage>,
    },
}

/// RDF operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RdfOperation {
    /// Insert a triple
    Insert(SerializableTriple),
    /// Remove a triple
    Remove(SerializableTriple),
    /// Batch insert
    BatchInsert(Vec<SerializableTriple>),
    /// Batch remove
    BatchRemove(Vec<SerializableTriple>),
    /// Read query (doesn't change state)
    Query(String),
}

/// Serializable triple for network transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub object_type: ObjectType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectType {
    NamedNode,
    BlankNode,
    Literal { datatype: Option<String>, language: Option<String> },
}

/// Operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationResult {
    Success,
    Failure(String),
    QueryResult(Vec<SerializableTriple>),
}

/// Checkpoint proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointProof {
    pub sequence: SequenceNumber,
    pub state_digest: Vec<u8>,
    pub signatures: HashMap<NodeId, Vec<u8>>,
}

/// Prepared proof for view changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedProof {
    pub view: ViewNumber,
    pub sequence: SequenceNumber,
    pub digest: Vec<u8>,
    pub pre_prepare: Box<BftMessage>,
    pub prepares: Vec<BftMessage>,
}

/// Byzantine fault tolerant node
pub struct BftNode {
    /// Node configuration
    config: BftConfig,
    
    /// This node's ID
    node_id: NodeId,
    
    /// Current view number
    view: Arc<RwLock<ViewNumber>>,
    
    /// Current phase
    phase: Arc<RwLock<Phase>>,
    
    /// Sequence number counter
    sequence_counter: Arc<Mutex<SequenceNumber>>,
    
    /// Node states (for each view and sequence)
    states: Arc<DashMap<(ViewNumber, SequenceNumber), ConsensusState>>,
    
    /// Message log
    message_log: Arc<RwLock<VecDeque<BftMessage>>>,
    
    /// Checkpoints
    checkpoints: Arc<RwLock<HashMap<SequenceNumber, CheckpointProof>>>,
    
    /// Stable checkpoint
    stable_checkpoint: Arc<RwLock<SequenceNumber>>,
    
    /// Other nodes in the cluster
    nodes: Arc<RwLock<HashMap<NodeId, NodeInfo>>>,
    
    /// Message sender
    message_tx: mpsc::UnboundedSender<(NodeId, BftMessage)>,
    
    /// Message receiver
    message_rx: Arc<Mutex<mpsc::UnboundedReceiver<(NodeId, BftMessage)>>>,
    
    /// RDF state machine
    state_machine: Arc<RwLock<RdfStateMachine>>,
    
    /// View change timer
    view_change_timer: Arc<Mutex<Option<Instant>>>,
}

/// Consensus state for a specific view and sequence
#[derive(Debug, Clone)]
pub struct ConsensusState {
    pub phase: Phase,
    pub request: Option<BftMessage>,
    pub digest: Vec<u8>,
    pub prepares: HashSet<NodeId>,
    pub commits: HashSet<NodeId>,
    pub replied: bool,
}

/// Node information
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: NodeId,
    pub address: String,
    pub public_key: Option<Vec<u8>>,
}

/// RDF state machine
pub struct RdfStateMachine {
    /// Triple store
    triples: HashSet<Triple>,
    
    /// Operation counter
    operation_count: u64,
    
    /// State digest cache
    digest_cache: Option<(u64, Vec<u8>)>,
}

impl BftNode {
    /// Create a new BFT node
    pub fn new(config: BftConfig, node_id: NodeId, nodes: Vec<NodeInfo>) -> Self {
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        
        let mut node_map = HashMap::new();
        for node in nodes {
            node_map.insert(node.id, node);
        }
        
        Self {
            config,
            node_id,
            view: Arc::new(RwLock::new(0)),
            phase: Arc::new(RwLock::new(Phase::Idle)),
            sequence_counter: Arc::new(Mutex::new(0)),
            states: Arc::new(DashMap::new()),
            message_log: Arc::new(RwLock::new(VecDeque::new())),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            stable_checkpoint: Arc::new(RwLock::new(0)),
            nodes: Arc::new(RwLock::new(node_map)),
            message_tx,
            message_rx: Arc::new(Mutex::new(message_rx)),
            state_machine: Arc::new(RwLock::new(RdfStateMachine::new())),
            view_change_timer: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Check if this node is the primary for the current view
    pub fn is_primary(&self) -> bool {
        let view = *self.view.read();
        let num_nodes = self.nodes.read().len() as u64;
        self.node_id == (view % num_nodes)
    }
    
    /// Get the primary node ID for a given view
    pub fn get_primary(&self, view: ViewNumber) -> NodeId {
        let num_nodes = self.nodes.read().len() as u64;
        view % num_nodes
    }
    
    /// Calculate message digest
    fn calculate_digest(message: &BftMessage) -> Vec<u8> {
        let serialized = bincode::serialize(message).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(&serialized);
        hasher.finalize().to_vec()
    }
    
    /// Process incoming message
    pub async fn process_message(&self, from: NodeId, message: BftMessage) -> Result<()> {
        // Log message
        self.log_message(message.clone());
        
        match message {
            BftMessage::Request { .. } => {
                if self.is_primary() {
                    self.handle_client_request(message).await?;
                }
            }
            
            BftMessage::PrePrepare { view, sequence, digest, request } => {
                self.handle_pre_prepare(from, view, sequence, digest, *request).await?;
            }
            
            BftMessage::Prepare { view, sequence, digest, node_id } => {
                self.handle_prepare(view, sequence, digest, node_id).await?;
            }
            
            BftMessage::Commit { view, sequence, digest, node_id } => {
                self.handle_commit(view, sequence, digest, node_id).await?;
            }
            
            BftMessage::Checkpoint { sequence, state_digest, node_id } => {
                self.handle_checkpoint(sequence, state_digest, node_id).await?;
            }
            
            BftMessage::ViewChange { .. } => {
                self.handle_view_change(message).await?;
            }
            
            BftMessage::NewView { .. } => {
                self.handle_new_view(message).await?;
            }
            
            _ => {}
        }
        
        Ok(())
    }
    
    /// Handle client request (primary only)
    async fn handle_client_request(&self, request: BftMessage) -> Result<()> {
        let view = *self.view.read();
        let sequence = {
            let mut counter = self.sequence_counter.lock();
            *counter += 1;
            *counter
        };
        
        let digest = Self::calculate_digest(&request);
        
        // Create pre-prepare message
        let pre_prepare = BftMessage::PrePrepare {
            view,
            sequence,
            digest: digest.clone(),
            request: Box::new(request.clone()),
        };
        
        // Store state
        let state = ConsensusState {
            phase: Phase::PrePrepare,
            request: Some(request),
            digest: digest.clone(),
            prepares: HashSet::new(),
            commits: HashSet::new(),
            replied: false,
        };
        self.states.insert((view, sequence), state);
        
        // Broadcast pre-prepare to all backup nodes
        self.broadcast_message(pre_prepare).await?;
        
        // Move to prepare phase
        self.enter_prepare_phase(view, sequence, digest).await?;
        
        Ok(())
    }
    
    /// Handle pre-prepare message (backup nodes)
    async fn handle_pre_prepare(
        &self,
        from: NodeId,
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: Vec<u8>,
        request: BftMessage,
    ) -> Result<()> {
        // Verify the message is from the primary
        if from != self.get_primary(view) {
            return Err(anyhow!("Pre-prepare not from primary"));
        }
        
        // Verify view number
        if view != *self.view.read() {
            return Ok(()); // Ignore messages from different views
        }
        
        // Verify digest
        let calculated_digest = Self::calculate_digest(&request);
        if digest != calculated_digest {
            return Err(anyhow!("Invalid message digest"));
        }
        
        // Store state
        let state = ConsensusState {
            phase: Phase::PrePrepare,
            request: Some(request),
            digest: digest.clone(),
            prepares: HashSet::new(),
            commits: HashSet::new(),
            replied: false,
        };
        self.states.insert((view, sequence), state);
        
        // Enter prepare phase
        self.enter_prepare_phase(view, sequence, digest).await?;
        
        Ok(())
    }
    
    /// Enter prepare phase
    async fn enter_prepare_phase(
        &self,
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: Vec<u8>,
    ) -> Result<()> {
        // Send prepare message
        let prepare = BftMessage::Prepare {
            view,
            sequence,
            digest,
            node_id: self.node_id,
        };
        
        self.broadcast_message(prepare).await?;
        
        // Update phase
        if let Some(mut state) = self.states.get_mut(&(view, sequence)) {
            state.phase = Phase::Prepare;
        }
        
        Ok(())
    }
    
    /// Handle prepare message
    async fn handle_prepare(
        &self,
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: Vec<u8>,
        node_id: NodeId,
    ) -> Result<()> {
        // Verify view
        if view != *self.view.read() {
            return Ok(());
        }
        
        // Update prepare count
        let should_commit = {
            if let Some(mut state) = self.states.get_mut(&(view, sequence)) {
                if state.digest == digest {
                    state.prepares.insert(node_id);
                    
                    // Check if we have 2f prepares (including our own)
                    state.prepares.len() >= 2 * self.config.fault_tolerance
                } else {
                    false
                }
            } else {
                false
            }
        };
        
        // Enter commit phase if we have enough prepares
        if should_commit {
            self.enter_commit_phase(view, sequence, digest).await?;
        }
        
        Ok(())
    }
    
    /// Enter commit phase
    async fn enter_commit_phase(
        &self,
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: Vec<u8>,
    ) -> Result<()> {
        // Send commit message
        let commit = BftMessage::Commit {
            view,
            sequence,
            digest,
            node_id: self.node_id,
        };
        
        self.broadcast_message(commit).await?;
        
        // Update phase
        if let Some(mut state) = self.states.get_mut(&(view, sequence)) {
            state.phase = Phase::Commit;
        }
        
        Ok(())
    }
    
    /// Handle commit message
    async fn handle_commit(
        &self,
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: Vec<u8>,
        node_id: NodeId,
    ) -> Result<()> {
        // Verify view
        if view != *self.view.read() {
            return Ok(());
        }
        
        // Update commit count and check if we should execute
        let (should_execute, request) = {
            if let Some(mut state) = self.states.get_mut(&(view, sequence)) {
                if state.digest == digest {
                    state.commits.insert(node_id);
                    
                    // Check if we have 2f+1 commits
                    let has_enough = state.commits.len() > 2 * self.config.fault_tolerance;
                    (has_enough && !state.replied, state.request.clone())
                } else {
                    (false, None)
                }
            } else {
                (false, None)
            }
        };
        
        // Execute the operation if we have enough commits
        if should_execute {
            if let Some(request) = request {
                self.execute_operation(view, sequence, request).await?;
            }
        }
        
        Ok(())
    }
    
    /// Execute the operation on the state machine
    async fn execute_operation(
        &self,
        view: ViewNumber,
        sequence: SequenceNumber,
        request: BftMessage,
    ) -> Result<()> {
        if let BftMessage::Request { client_id, operation, .. } = request {
            // Execute on state machine
            let result = self.state_machine.write().execute(operation)?;
            
            // Send reply to client
            let reply = BftMessage::Reply {
                view,
                sequence,
                client_id,
                result,
                timestamp: SystemTime::now(),
            };
            
            // Mark as replied
            if let Some(mut state) = self.states.get_mut(&(view, sequence)) {
                state.replied = true;
            }
            
            // TODO: Send reply to client
            
            // Check if we need a checkpoint
            if sequence % self.config.checkpoint_interval == 0 {
                self.create_checkpoint(sequence).await?;
            }
        }
        
        Ok(())
    }
    
    /// Create a checkpoint
    async fn create_checkpoint(&self, sequence: SequenceNumber) -> Result<()> {
        let state_digest = self.state_machine.write().calculate_digest();
        
        let checkpoint = BftMessage::Checkpoint {
            sequence,
            state_digest,
            node_id: self.node_id,
        };
        
        self.broadcast_message(checkpoint).await?;
        
        Ok(())
    }
    
    /// Handle checkpoint message
    async fn handle_checkpoint(
        &self,
        sequence: SequenceNumber,
        state_digest: Vec<u8>,
        node_id: NodeId,
    ) -> Result<()> {
        let mut checkpoints = self.checkpoints.write();
        
        let proof = checkpoints.entry(sequence).or_insert_with(|| CheckpointProof {
            sequence,
            state_digest: state_digest.clone(),
            signatures: HashMap::new(),
        });
        
        // Add signature (simplified - in real implementation would verify signature)
        proof.signatures.insert(node_id, vec![]);
        
        // Check if we have 2f+1 matching checkpoints
        if proof.signatures.len() > 2 * self.config.fault_tolerance {
            // Update stable checkpoint
            let mut stable = self.stable_checkpoint.write();
            if sequence > *stable {
                *stable = sequence;
                
                // Garbage collect old messages and states
                self.garbage_collect(sequence);
            }
        }
        
        Ok(())
    }
    
    /// Garbage collect old messages and states
    fn garbage_collect(&self, stable_checkpoint: SequenceNumber) {
        // Remove old states
        self.states.retain(|(_, seq), _| *seq > stable_checkpoint);
        
        // Remove old messages from log
        let mut log = self.message_log.write();
        // Keep only recent messages (simplified)
        while log.len() > self.config.max_log_size {
            log.pop_front();
        }
    }
    
    /// Handle view change message
    async fn handle_view_change(&self, message: BftMessage) -> Result<()> {
        // TODO: Implement view change protocol
        // This is complex and involves collecting view change messages,
        // verifying prepared certificates, and transitioning to new view
        Ok(())
    }
    
    /// Handle new view message
    async fn handle_new_view(&self, message: BftMessage) -> Result<()> {
        // TODO: Implement new view validation and transition
        Ok(())
    }
    
    /// Broadcast message to all nodes
    async fn broadcast_message(&self, message: BftMessage) -> Result<()> {
        let nodes = self.nodes.read();
        for (node_id, _) in nodes.iter() {
            if *node_id != self.node_id {
                self.message_tx.send((*node_id, message.clone()))?;
            }
        }
        Ok(())
    }
    
    /// Log message for debugging
    fn log_message(&self, message: BftMessage) {
        let mut log = self.message_log.write();
        log.push_back(message);
        
        // Limit log size
        while log.len() > self.config.max_log_size {
            log.pop_front();
        }
    }
    
    /// Start the node
    pub async fn start(&self) -> Result<()> {
        // Start message processing loop
        let message_rx = self.message_rx.clone();
        let self_clone = self.clone();
        
        tokio::spawn(async move {
            loop {
                // Try to get the next message without holding the lock across await
                let msg = {
                    let mut rx_guard = message_rx.lock();
                    // Use try_recv to avoid blocking
                    rx_guard.try_recv()
                };
                
                let message = match msg {
                    Ok(msg) => Some(msg),
                    Err(mpsc::error::TryRecvError::Empty) => {
                        // No message available, wait a bit
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }
                    Err(mpsc::error::TryRecvError::Disconnected) => None,
                };
                
                match message {
                    Some((from, msg)) => {
                        if let Err(e) = self_clone.process_message(from, msg).await {
                            eprintln!("Error processing message: {}", e);
                        }
                    }
                    None => break, // Channel closed
                }
            }
        });
        
        // Start view change timer
        self.start_view_change_timer().await;
        
        Ok(())
    }
    
    /// Start view change timer
    async fn start_view_change_timer(&self) {
        let view_change_timeout = self.config.view_change_timeout;
        let self_clone = self.clone();
        
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                
                // Check if we need a view change
                let should_change = {
                    let timer = self_clone.view_change_timer.lock();
                    if let Some(last_activity) = *timer {
                        last_activity.elapsed() > view_change_timeout
                    } else {
                        false
                    }
                };
                
                if should_change {
                    // Initiate view change
                    if let Err(e) = self_clone.initiate_view_change().await {
                        eprintln!("Error initiating view change: {}", e);
                    }
                }
            }
        });
    }
    
    /// Initiate view change
    async fn initiate_view_change(&self) -> Result<()> {
        let current_view = *self.view.read();
        let new_view = current_view + 1;
        
        // Collect prepared messages
        let prepared_messages = self.collect_prepared_messages();
        
        // Collect checkpoint proofs
        let checkpoints = self.checkpoints.read()
            .values()
            .cloned()
            .collect();
        
        let view_change = BftMessage::ViewChange {
            new_view,
            node_id: self.node_id,
            last_sequence: *self.stable_checkpoint.read(),
            checkpoints,
            prepared_messages,
        };
        
        self.broadcast_message(view_change).await?;
        
        Ok(())
    }
    
    /// Collect prepared messages for view change
    fn collect_prepared_messages(&self) -> Vec<PreparedProof> {
        let mut prepared = Vec::new();
        
        for entry in self.states.iter() {
            let ((view, sequence), state) = entry.pair();
            
            if state.prepares.len() >= 2 * self.config.fault_tolerance {
                // Create prepared proof
                if let Some(request) = &state.request {
                    let proof = PreparedProof {
                        view: *view,
                        sequence: *sequence,
                        digest: state.digest.clone(),
                        pre_prepare: Box::new(request.clone()),
                        prepares: vec![], // Simplified - would include actual prepare messages
                    };
                    prepared.push(proof);
                }
            }
        }
        
        prepared
    }
}

// Clone implementation for BftNode
impl Clone for BftNode {
    fn clone(&self) -> Self {
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        
        Self {
            config: self.config.clone(),
            node_id: self.node_id,
            view: self.view.clone(),
            phase: self.phase.clone(),
            sequence_counter: self.sequence_counter.clone(),
            states: self.states.clone(),
            message_log: self.message_log.clone(),
            checkpoints: self.checkpoints.clone(),
            stable_checkpoint: self.stable_checkpoint.clone(),
            nodes: self.nodes.clone(),
            message_tx,
            message_rx: Arc::new(Mutex::new(message_rx)),
            state_machine: self.state_machine.clone(),
            view_change_timer: self.view_change_timer.clone(),
        }
    }
}

impl RdfStateMachine {
    /// Create a new RDF state machine
    pub fn new() -> Self {
        Self {
            triples: HashSet::new(),
            operation_count: 0,
            digest_cache: None,
        }
    }
    
    /// Execute an RDF operation
    pub fn execute(&mut self, operation: RdfOperation) -> Result<OperationResult> {
        self.operation_count += 1;
        self.digest_cache = None; // Invalidate cache
        
        match operation {
            RdfOperation::Insert(triple) => {
                let t = self.deserialize_triple(triple)?;
                self.triples.insert(t);
                Ok(OperationResult::Success)
            }
            
            RdfOperation::Remove(triple) => {
                let t = self.deserialize_triple(triple)?;
                self.triples.remove(&t);
                Ok(OperationResult::Success)
            }
            
            RdfOperation::BatchInsert(triples) => {
                for triple in triples {
                    let t = self.deserialize_triple(triple)?;
                    self.triples.insert(t);
                }
                Ok(OperationResult::Success)
            }
            
            RdfOperation::BatchRemove(triples) => {
                for triple in triples {
                    let t = self.deserialize_triple(triple)?;
                    self.triples.remove(&t);
                }
                Ok(OperationResult::Success)
            }
            
            RdfOperation::Query(_query) => {
                // Simplified - would execute SPARQL query
                let results: Vec<SerializableTriple> = self.triples
                    .iter()
                    .take(10) // Limit results
                    .map(|t| self.serialize_triple(t))
                    .collect();
                Ok(OperationResult::QueryResult(results))
            }
        }
    }
    
    /// Calculate state digest
    pub fn calculate_digest(&mut self) -> Vec<u8> {
        // Check cache
        if let Some((count, digest)) = &self.digest_cache {
            if *count == self.operation_count {
                return digest.clone();
            }
        }
        
        // Calculate new digest
        let mut hasher = Sha256::new();
        
        // Sort triples for deterministic digest
        let mut sorted_triples: Vec<_> = self.triples.iter().collect();
        sorted_triples.sort_by_key(|t| {
            (t.subject().to_string(), t.predicate().to_string(), t.object().to_string())
        });
        
        for triple in sorted_triples {
            hasher.update(triple.subject().to_string().as_bytes());
            hasher.update(triple.predicate().to_string().as_bytes());
            hasher.update(triple.object().to_string().as_bytes());
        }
        
        hasher.update(&self.operation_count.to_le_bytes());
        
        let digest = hasher.finalize().to_vec();
        
        // Cache the digest
        self.digest_cache = Some((self.operation_count, digest.clone()));
        
        digest
    }
    
    /// Deserialize a triple from network format
    fn deserialize_triple(&self, st: SerializableTriple) -> Result<Triple> {
        let subject = NamedNode::new(&st.subject)?;
        let predicate = NamedNode::new(&st.predicate)?;
        
        let object = match st.object_type {
            ObjectType::NamedNode => {
                crate::model::Object::NamedNode(NamedNode::new(&st.object)?)
            }
            ObjectType::BlankNode => {
                crate::model::Object::BlankNode(BlankNode::new(&st.object)?)
            }
            ObjectType::Literal { datatype, language } => {
                if let Some(lang) = language {
                    crate::model::Object::Literal(Literal::new_language_tagged_literal(&st.object, &lang)?)
                } else if let Some(dt) = datatype {
                    crate::model::Object::Literal(Literal::new_typed(&st.object, NamedNode::new(&dt)?))
                } else {
                    crate::model::Object::Literal(Literal::new(&st.object))
                }
            }
        };
        
        Ok(Triple::new(subject, predicate, object))
    }
    
    /// Serialize a triple for network transmission
    fn serialize_triple(&self, triple: &Triple) -> SerializableTriple {
        let object_type = match triple.object() {
            crate::model::Object::NamedNode(_) => ObjectType::NamedNode,
            crate::model::Object::BlankNode(_) => ObjectType::BlankNode,
            crate::model::Object::Literal(lit) => ObjectType::Literal {
                datatype: if lit.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    Some(lit.datatype().as_str().to_string())
                } else {
                    None
                },
                language: lit.language().map(|l| l.to_string()),
            },
            _ => ObjectType::NamedNode, // Fallback
        };
        
        SerializableTriple {
            subject: triple.subject().to_string(),
            predicate: triple.predicate().to_string(),
            object: triple.object().to_string(),
            object_type,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_bft_consensus_basic() {
        // Create 4 nodes (tolerating 1 Byzantine failure)
        let config = BftConfig::default();
        
        let nodes = vec![
            NodeInfo { id: 0, address: "node0".to_string(), public_key: None },
            NodeInfo { id: 1, address: "node1".to_string(), public_key: None },
            NodeInfo { id: 2, address: "node2".to_string(), public_key: None },
            NodeInfo { id: 3, address: "node3".to_string(), public_key: None },
        ];
        
        let node0 = BftNode::new(config.clone(), 0, nodes.clone());
        
        // Test primary detection
        assert!(node0.is_primary());
        assert_eq!(node0.get_primary(0), 0);
        assert_eq!(node0.get_primary(1), 1);
        assert_eq!(node0.get_primary(2), 2);
        assert_eq!(node0.get_primary(3), 3);
        assert_eq!(node0.get_primary(4), 0); // Wraps around
    }
    
    #[test]
    fn test_message_digest() {
        let request = BftMessage::Request {
            client_id: "client1".to_string(),
            operation: RdfOperation::Insert(SerializableTriple {
                subject: "http://example.org/s".to_string(),
                predicate: "http://example.org/p".to_string(),
                object: "value".to_string(),
                object_type: ObjectType::Literal { datatype: None, language: None },
            }),
            timestamp: SystemTime::now(),
            signature: None,
        };
        
        let digest1 = BftNode::calculate_digest(&request);
        let digest2 = BftNode::calculate_digest(&request);
        
        assert_eq!(digest1, digest2); // Same message produces same digest
    }
    
    #[test]
    fn test_state_machine() {
        let mut state_machine = RdfStateMachine::new();
        
        // Test insert
        let triple = SerializableTriple {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "value".to_string(),
            object_type: ObjectType::Literal { datatype: None, language: None },
        };
        
        let result = state_machine.execute(RdfOperation::Insert(triple.clone())).unwrap();
        assert!(matches!(result, OperationResult::Success));
        assert_eq!(state_machine.triples.len(), 1);
        
        // Test digest calculation
        let digest1 = state_machine.calculate_digest();
        let digest2 = state_machine.calculate_digest();
        assert_eq!(digest1, digest2); // Cached digest
        
        // Test remove
        let result = state_machine.execute(RdfOperation::Remove(triple)).unwrap();
        assert!(matches!(result, OperationResult::Success));
        assert_eq!(state_machine.triples.len(), 0);
        
        // Digest should change after operation
        let digest3 = state_machine.calculate_digest();
        assert_ne!(digest1, digest3);
    }
    
    #[test]
    fn test_checkpoint_proof() {
        let mut proof = CheckpointProof {
            sequence: 100,
            state_digest: vec![1, 2, 3, 4],
            signatures: HashMap::new(),
        };
        
        // Add signatures
        proof.signatures.insert(0, vec![]);
        proof.signatures.insert(1, vec![]);
        proof.signatures.insert(2, vec![]);
        
        // With f=1, need 2f+1 = 3 signatures
        assert_eq!(proof.signatures.len(), 3);
    }
}