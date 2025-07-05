//! BFT Node implementation and consensus logic

#![allow(dead_code)]

use super::detection::ByzantineDetector;
use super::messages::BftMessage;
use super::state_machine::RdfStateMachine;
use super::types::*;
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

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

    /// Byzantine behavior detection
    byzantine_detector: Arc<RwLock<ByzantineDetector>>,
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
            config: config.clone(),
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
            byzantine_detector: Arc::new(RwLock::new(ByzantineDetector::new(3))), // Default threshold of 3
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

    /// Log a message
    fn log_message(&self, message: BftMessage) {
        let mut log = self.message_log.write();
        log.push_back(message);

        // Trim log if it gets too large
        if log.len() > self.config.max_log_size {
            log.pop_front();
        }
    }

    /// Broadcast message to all other nodes
    async fn broadcast_message(&self, message: BftMessage) -> Result<()> {
        let nodes = self.nodes.read();
        for (&node_id, _) in nodes.iter() {
            if node_id != self.node_id {
                self.message_tx
                    .send((node_id, message.clone()))
                    .map_err(|e| anyhow!("Failed to send message: {}", e))?;
            }
        }
        Ok(())
    }

    /// Process incoming message with enhanced Byzantine detection
    pub async fn process_message(&self, from: NodeId, message: BftMessage) -> Result<()> {
        let start_time = Instant::now();

        // Enhanced Byzantine detection checks
        {
            let mut detector = self.byzantine_detector.write();

            // Check for replay attacks
            let message_hash = Self::calculate_digest(&message);
            if detector.check_replay_attack(from, message_hash.clone()) {
                return Err(anyhow!("Replay attack detected from node {}", from));
            }

            // Monitor resource usage
            detector.monitor_resource_usage(from);

            // Update network partition status
            detector.check_network_partition(from);

            // Check for equivocation (view and sequence dependent)
            if let BftMessage::PrePrepare { view, sequence, .. }
            | BftMessage::Prepare { view, sequence, .. }
            | BftMessage::Commit { view, sequence, .. } = &message
            {
                if detector.check_equivocation(from, *view, *sequence, message_hash) {
                    return Err(anyhow!("Equivocation detected from node {}", from));
                }
            }
        }

        // Log message
        self.log_message(message.clone());

        match message {
            BftMessage::Request { .. } => {
                if self.is_primary() {
                    self.handle_client_request(message).await?;
                }
            }

            BftMessage::PrePrepare {
                view,
                sequence,
                digest,
                request,
            } => {
                self.handle_pre_prepare(from, view, sequence, digest, *request)
                    .await?;
            }

            BftMessage::Prepare {
                view,
                sequence,
                digest,
                node_id,
            } => {
                self.handle_prepare(view, sequence, digest, node_id).await?;
            }

            BftMessage::Commit {
                view,
                sequence,
                digest,
                node_id,
            } => {
                self.handle_commit(view, sequence, digest, node_id).await?;
            }

            BftMessage::Checkpoint {
                sequence,
                state_digest,
                node_id,
            } => {
                self.handle_checkpoint(sequence, state_digest, node_id)
                    .await?;
            }

            BftMessage::ViewChange { .. } => {
                self.handle_view_change(message).await?;
            }

            BftMessage::NewView { .. } => {
                self.handle_new_view(message).await?;
            }

            _ => {}
        }

        // Record timing information for Byzantine detection
        let response_time = start_time.elapsed();
        {
            let mut detector = self.byzantine_detector.write();
            detector.report_timing_anomaly(from, response_time);
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

        // Update commit count and execute if ready
        let should_execute = {
            if let Some(mut state) = self.states.get_mut(&(view, sequence)) {
                if state.digest == digest {
                    state.commits.insert(node_id);

                    // Check if we have 2f+1 commits (including our own)
                    state.commits.len() > 2 * self.config.fault_tolerance
                } else {
                    false
                }
            } else {
                false
            }
        };

        // Execute operation if we have enough commits
        if should_execute {
            self.execute_operation(view, sequence).await?;
        }

        Ok(())
    }

    /// Execute operation after consensus
    async fn execute_operation(&self, view: ViewNumber, sequence: SequenceNumber) -> Result<()> {
        if let Some(state) = self.states.get(&(view, sequence)) {
            if let Some(request) = &state.request {
                if let BftMessage::Request {
                    operation,
                    client_id,
                    ..
                } = request
                {
                    // Execute operation on state machine
                    let result = {
                        let mut sm = self.state_machine.write();
                        sm.execute(operation.clone())?
                    };

                    // Send reply to client
                    let reply = BftMessage::Reply {
                        view,
                        sequence,
                        client_id: client_id.clone(),
                        result,
                        timestamp: std::time::SystemTime::now(),
                    };

                    // In a real implementation, we would send this to the client
                    // For now, we'll just log it
                    self.log_message(reply);

                    // Mark as replied
                    if let Some(mut state) = self.states.get_mut(&(view, sequence)) {
                        state.replied = true;
                    }
                }
            }
        }

        // Check if we should create a checkpoint
        if sequence % self.config.checkpoint_interval == 0 {
            self.create_checkpoint(sequence).await?;
        }

        Ok(())
    }

    /// Create checkpoint
    async fn create_checkpoint(&self, sequence: SequenceNumber) -> Result<()> {
        let state_digest = {
            let sm = self.state_machine.read();
            sm.get_state_digest()
        };

        let checkpoint = BftMessage::Checkpoint {
            sequence,
            state_digest: state_digest.clone(),
            node_id: self.node_id,
        };

        self.broadcast_message(checkpoint).await?;

        // Store checkpoint
        let proof = CheckpointProof {
            sequence,
            state_digest,
            signatures: HashMap::new(), // Would contain actual signatures in real implementation
        };

        self.checkpoints.write().insert(sequence, proof);

        Ok(())
    }

    /// Handle checkpoint message
    async fn handle_checkpoint(
        &self,
        _sequence: SequenceNumber,
        state_digest: Vec<u8>,
        node_id: NodeId,
    ) -> Result<()> {
        // Verify checkpoint against our state
        let our_digest = {
            let sm = self.state_machine.read();
            sm.get_state_digest()
        };

        if state_digest != our_digest {
            // Byzantine detection - inconsistent state
            let mut detector = self.byzantine_detector.write();
            detector.report_inconsistent_pattern(node_id);
            return Err(anyhow!("Inconsistent checkpoint from node {}", node_id));
        }

        Ok(())
    }

    /// Handle view change message
    async fn handle_view_change(&self, _message: BftMessage) -> Result<()> {
        // View change logic would be implemented here
        // This is a complex process involving collecting prepared messages
        // and agreeing on a new primary
        Ok(())
    }

    /// Handle new view message
    async fn handle_new_view(&self, _message: BftMessage) -> Result<()> {
        // New view logic would be implemented here
        // This involves processing the new view and starting consensus
        // with any prepared but uncommitted operations
        Ok(())
    }

    /// Get node status information
    pub fn get_status(&self) -> NodeStatus {
        NodeStatus {
            node_id: self.node_id,
            view: *self.view.read(),
            phase: *self.phase.read(),
            sequence: *self.sequence_counter.lock(),
            suspected_nodes: self.byzantine_detector.read().get_suspected_nodes().clone(),
        }
    }
}

/// Node status information
#[derive(Debug, Clone)]
pub struct NodeStatus {
    pub node_id: NodeId,
    pub view: ViewNumber,
    pub phase: Phase,
    pub sequence: SequenceNumber,
    pub suspected_nodes: HashSet<NodeId>,
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
            byzantine_detector: self.byzantine_detector.clone(),
        }
    }
}
