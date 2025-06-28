//! Byzantine Fault Tolerance (BFT) consensus for untrusted environments
//!
//! This module implements PBFT (Practical Byzantine Fault Tolerance) for RDF stores
//! operating in untrusted environments where nodes may act maliciously.

use crate::model::{BlankNode, Literal, NamedNode, Triple};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
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
            fault_tolerance: 1, // Tolerate 1 Byzantine node (requires 4 total nodes)
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
    Literal {
        datatype: Option<String>,
        language: Option<String>,
    },
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

    /// Byzantine behavior detection
    byzantine_detector: Arc<RwLock<ByzantineDetector>>,
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

/// Byzantine behavior detection system with advanced threat detection
pub struct ByzantineDetector {
    /// Suspected Byzantine nodes
    suspected_nodes: HashSet<NodeId>,

    /// Message timing anomalies with detailed analysis
    timing_anomalies: HashMap<NodeId, TimingAnalysis>,

    /// Signature verification failures
    signature_failures: HashMap<NodeId, usize>,

    /// Inconsistent message patterns
    inconsistent_patterns: HashMap<NodeId, usize>,

    /// Detection threshold
    detection_threshold: usize,

    /// Network partition detection
    partition_detector: PartitionDetector,

    /// Message replay attack detection
    replay_detector: ReplayDetector,

    /// Equivocation detection (sending different messages for same view/sequence)
    equivocation_detector: EquivocationDetector,

    /// Resource exhaustion attack detection
    resource_monitor: ResourceMonitor,

    /// Collusion detection between nodes
    collusion_detector: CollusionDetector,
}

/// Advanced timing analysis for Byzantine detection
#[derive(Debug, Clone)]
pub struct TimingAnalysis {
    /// Recent message timestamps
    message_times: VecDeque<Instant>,
    /// Average response time
    avg_response_time: Duration,
    /// Standard deviation of response times
    response_time_stddev: Duration,
    /// Suspicious timing patterns count
    suspicious_patterns: usize,
}

/// Network partition detection system
#[derive(Debug, Clone)]
pub struct PartitionDetector {
    /// Last communication time with each node
    last_communication: HashMap<NodeId, Instant>,
    /// Suspected partitioned nodes
    partitioned_nodes: HashSet<NodeId>,
    /// Partition timeout threshold
    partition_timeout: Duration,
}

impl PartitionDetector {
    pub fn new() -> Self {
        Self {
            last_communication: HashMap::new(),
            partitioned_nodes: HashSet::new(),
            partition_timeout: Duration::from_secs(30),
        }
    }
}

/// Replay attack detection system
#[derive(Debug, Clone)]
pub struct ReplayDetector {
    /// Recently seen message hashes with timestamps
    seen_messages: HashMap<Vec<u8>, Instant>,
    /// Replay attack threshold
    replay_window: Duration,
    /// Detected replay attempts
    replay_attempts: HashMap<NodeId, usize>,
}

impl ReplayDetector {
    pub fn new() -> Self {
        Self {
            seen_messages: HashMap::new(),
            replay_window: Duration::from_secs(60), // 1 minute window
            replay_attempts: HashMap::new(),
        }
    }
}

/// Equivocation detection system
#[derive(Debug, Clone)]
pub struct EquivocationDetector {
    /// Messages per view/sequence from each node
    node_messages: HashMap<NodeId, HashMap<(ViewNumber, SequenceNumber), Vec<Vec<u8>>>>,
    /// Detected equivocations
    equivocations: HashMap<NodeId, usize>,
}

impl EquivocationDetector {
    pub fn new() -> Self {
        Self {
            node_messages: HashMap::new(),
            equivocations: HashMap::new(),
        }
    }
}

/// Resource exhaustion monitoring
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    /// Message rate per node
    message_rates: HashMap<NodeId, VecDeque<Instant>>,
    /// Rate limit threshold (messages per second)
    rate_limit: f64,
    /// Memory usage tracking
    memory_usage: HashMap<NodeId, usize>,
    /// Detected resource attacks
    resource_attacks: HashMap<NodeId, usize>,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            message_rates: HashMap::new(),
            rate_limit: 100.0, // 100 messages per second default limit
            memory_usage: HashMap::new(),
            resource_attacks: HashMap::new(),
        }
    }
}

/// Collusion detection between Byzantine nodes
#[derive(Debug, Clone)]
pub struct CollusionDetector {
    /// Coordinated behavior patterns
    coordination_patterns: HashMap<Vec<NodeId>, usize>,
    /// Simultaneous actions tracking
    simultaneous_actions: VecDeque<(Instant, Vec<NodeId>)>,
    /// Collusion threshold
    collusion_threshold: usize,
}

impl CollusionDetector {
    pub fn new() -> Self {
        Self {
            coordination_patterns: HashMap::new(),
            simultaneous_actions: VecDeque::new(),
            collusion_threshold: 5, // 5 coordinated actions trigger suspicion
        }
    }
}

impl ByzantineDetector {
    pub fn new(detection_threshold: usize) -> Self {
        Self {
            suspected_nodes: HashSet::new(),
            timing_anomalies: HashMap::new(),
            signature_failures: HashMap::new(),
            inconsistent_patterns: HashMap::new(),
            detection_threshold,
            partition_detector: PartitionDetector::new(),
            replay_detector: ReplayDetector::new(),
            equivocation_detector: EquivocationDetector::new(),
            resource_monitor: ResourceMonitor::new(),
            collusion_detector: CollusionDetector::new(),
        }
    }

    /// Report advanced timing anomaly with detailed analysis
    pub fn report_timing_anomaly(&mut self, node_id: NodeId, response_time: Duration) {
        let now = Instant::now();

        // First, update/create the timing analysis
        {
            let analysis = self
                .timing_anomalies
                .entry(node_id)
                .or_insert_with(|| TimingAnalysis {
                    message_times: VecDeque::new(),
                    avg_response_time: Duration::from_millis(100), // Default
                    response_time_stddev: Duration::from_millis(50),
                    suspicious_patterns: 0,
                });

            analysis.message_times.push_back(now);

            // Keep only recent timing data (last 100 messages)
            while analysis.message_times.len() > 100 {
                analysis.message_times.pop_front();
            }
        }

        // Update statistics (separate borrow)
        self.update_timing_statistics(node_id, response_time);

        // Detect suspicious patterns and update if needed
        let is_suspicious = self.detect_timing_attack(node_id, response_time);
        if is_suspicious {
            if let Some(analysis) = self.timing_anomalies.get_mut(&node_id) {
                analysis.suspicious_patterns += 1;
                if analysis.suspicious_patterns >= self.detection_threshold {
                    self.suspected_nodes.insert(node_id);
                    tracing::warn!("Node {} suspected of timing attacks", node_id);
                }
            }
        }
    }

    /// Detect potential timing-based attacks
    fn detect_timing_attack(&self, node_id: NodeId, response_time: Duration) -> bool {
        if let Some(analysis) = self.timing_anomalies.get(&node_id) {
            // Check for extremely fast responses (potential pre-computation)
            if response_time < Duration::from_millis(1) {
                return true;
            }

            // Check for extremely slow responses (potential DoS)
            if response_time > analysis.avg_response_time + 3 * analysis.response_time_stddev {
                return true;
            }

            // Check for suspiciously regular timing (potential automation)
            if analysis.message_times.len() >= 10 {
                let intervals: Vec<_> = analysis
                    .message_times
                    .iter()
                    .zip(analysis.message_times.iter().skip(1))
                    .map(|(a, b)| b.duration_since(*a))
                    .collect();

                // If all intervals are too similar, it's suspicious
                if let (Some(&min), Some(&max)) = (intervals.iter().min(), intervals.iter().max()) {
                    if max - min < Duration::from_millis(10) && intervals.len() >= 5 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Update timing statistics for a node
    fn update_timing_statistics(&mut self, node_id: NodeId, response_time: Duration) {
        if let Some(analysis) = self.timing_anomalies.get_mut(&node_id) {
            // Simple exponential moving average
            let alpha = 0.1;
            let new_time = response_time.as_millis() as f64;
            let old_avg = analysis.avg_response_time.as_millis() as f64;
            let new_avg = alpha * new_time + (1.0 - alpha) * old_avg;
            analysis.avg_response_time = Duration::from_millis(new_avg as u64);
        }
    }

    pub fn report_signature_failure(&mut self, node_id: NodeId) {
        *self.signature_failures.entry(node_id).or_default() += 1;
        if self.signature_failures[&node_id] >= self.detection_threshold {
            self.suspected_nodes.insert(node_id);
            tracing::warn!("Node {} suspected due to signature failures", node_id);
        }
    }

    pub fn report_inconsistent_pattern(&mut self, node_id: NodeId) {
        *self.inconsistent_patterns.entry(node_id).or_default() += 1;
        if self.inconsistent_patterns[&node_id] >= self.detection_threshold {
            self.suspected_nodes.insert(node_id);
            tracing::warn!("Node {} suspected due to inconsistent patterns", node_id);
        }
    }

    /// Check for message replay attacks
    pub fn check_replay_attack(&mut self, node_id: NodeId, message_hash: Vec<u8>) -> bool {
        let now = Instant::now();

        // Clean old entries
        self.replay_detector
            .seen_messages
            .retain(|_, &mut timestamp| {
                now.duration_since(timestamp) <= self.replay_detector.replay_window
            });

        // Check if message was seen recently
        if let Some(&timestamp) = self.replay_detector.seen_messages.get(&message_hash) {
            if now.duration_since(timestamp) <= self.replay_detector.replay_window {
                *self
                    .replay_detector
                    .replay_attempts
                    .entry(node_id)
                    .or_default() += 1;
                if self.replay_detector.replay_attempts[&node_id] >= self.detection_threshold {
                    self.suspected_nodes.insert(node_id);
                    tracing::warn!("Node {} suspected of replay attacks", node_id);
                }
                return true;
            }
        }

        self.replay_detector.seen_messages.insert(message_hash, now);
        false
    }

    /// Detect equivocation (sending different messages for same view/sequence)
    pub fn check_equivocation(
        &mut self,
        node_id: NodeId,
        view: ViewNumber,
        sequence: SequenceNumber,
        message_hash: Vec<u8>,
    ) -> bool {
        let messages = self
            .equivocation_detector
            .node_messages
            .entry(node_id)
            .or_default()
            .entry((view, sequence))
            .or_default();

        // Check if we've seen a different message for this view/sequence
        if !messages.is_empty() && !messages.contains(&message_hash) {
            *self
                .equivocation_detector
                .equivocations
                .entry(node_id)
                .or_default() += 1;
            if self.equivocation_detector.equivocations[&node_id] >= self.detection_threshold {
                self.suspected_nodes.insert(node_id);
                tracing::warn!("Node {} suspected of equivocation", node_id);
            }
            return true;
        }

        messages.push(message_hash);
        false
    }

    /// Monitor resource usage for DoS attacks
    pub fn monitor_resource_usage(&mut self, node_id: NodeId) -> bool {
        let now = Instant::now();
        let rates = self
            .resource_monitor
            .message_rates
            .entry(node_id)
            .or_default();

        rates.push_back(now);

        // Keep only messages from the last second
        while let Some(&front_time) = rates.front() {
            if now.duration_since(front_time) > Duration::from_secs(1) {
                rates.pop_front();
            } else {
                break;
            }
        }

        // Check if rate exceeds threshold
        let current_rate = rates.len() as f64;
        if current_rate > self.resource_monitor.rate_limit {
            *self
                .resource_monitor
                .resource_attacks
                .entry(node_id)
                .or_default() += 1;
            if self.resource_monitor.resource_attacks[&node_id] >= self.detection_threshold {
                self.suspected_nodes.insert(node_id);
                tracing::warn!("Node {} suspected of resource exhaustion attack", node_id);
            }
            return true;
        }

        false
    }

    /// Detect potential collusion between nodes
    pub fn check_collusion(&mut self, coordinating_nodes: Vec<NodeId>) {
        if coordinating_nodes.len() >= 2 {
            let now = Instant::now();

            // Record simultaneous action
            self.collusion_detector
                .simultaneous_actions
                .push_back((now, coordinating_nodes.clone()));

            // Clean old entries (keep last hour)
            while let Some((timestamp, _)) = self.collusion_detector.simultaneous_actions.front() {
                if now.duration_since(*timestamp) > Duration::from_secs(3600) {
                    self.collusion_detector.simultaneous_actions.pop_front();
                } else {
                    break;
                }
            }

            // Check for repeated coordination
            *self
                .collusion_detector
                .coordination_patterns
                .entry(coordinating_nodes.clone())
                .or_default() += 1;

            if self.collusion_detector.coordination_patterns[&coordinating_nodes]
                >= self.collusion_detector.collusion_threshold
            {
                for &node_id in &coordinating_nodes {
                    self.suspected_nodes.insert(node_id);
                }
                tracing::warn!(
                    "Suspected collusion detected between nodes: {:?}",
                    coordinating_nodes
                );
            }
        }
    }

    /// Check network partition status
    pub fn check_network_partition(&mut self, node_id: NodeId) {
        let now = Instant::now();
        self.partition_detector
            .last_communication
            .insert(node_id, now);

        // Check for partitioned nodes
        for (&id, &last_time) in &self.partition_detector.last_communication {
            if now.duration_since(last_time) > self.partition_detector.partition_timeout {
                self.partition_detector.partitioned_nodes.insert(id);
            } else {
                self.partition_detector.partitioned_nodes.remove(&id);
            }
        }
    }

    /// Get comprehensive threat assessment
    pub fn get_threat_assessment(&self, node_id: NodeId) -> ThreatLevel {
        let mut score = 0;

        if self.suspected_nodes.contains(&node_id) {
            score += 10;
        }

        if let Some(failures) = self.signature_failures.get(&node_id) {
            score += failures * 2;
        }

        if let Some(patterns) = self.inconsistent_patterns.get(&node_id) {
            score += patterns;
        }

        if let Some(replays) = self.replay_detector.replay_attempts.get(&node_id) {
            score += replays * 3;
        }

        if let Some(equivocations) = self.equivocation_detector.equivocations.get(&node_id) {
            score += equivocations * 5;
        }

        if let Some(attacks) = self.resource_monitor.resource_attacks.get(&node_id) {
            score += attacks;
        }

        match score {
            0..=2 => ThreatLevel::Low,
            3..=7 => ThreatLevel::Medium,
            8..=15 => ThreatLevel::High,
            _ => ThreatLevel::Critical,
        }
    }

    pub fn is_suspected(&self, node_id: NodeId) -> bool {
        self.suspected_nodes.contains(&node_id)
    }

    pub fn get_suspected_nodes(&self) -> &HashSet<NodeId> {
        &self.suspected_nodes
    }

    pub fn is_partitioned(&self, node_id: NodeId) -> bool {
        self.partition_detector.partitioned_nodes.contains(&node_id)
    }
}

/// Threat level assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreatLevel {
    Low,
    Medium,
    High,
    Critical,
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
        if let BftMessage::Request {
            client_id,
            operation,
            ..
        } = request
        {
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

        let proof = checkpoints
            .entry(sequence)
            .or_insert_with(|| CheckpointProof {
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
        let message_clone = message.clone();
        if let BftMessage::ViewChange {
            new_view,
            node_id,
            last_sequence,
            ref checkpoints,
            ref prepared_messages,
        } = message
        {
            let current_view = *self.view.read();

            // Only process if this is for a future view
            if new_view <= current_view {
                return Ok(());
            }

            // Verify the view change message
            if !self
                .verify_view_change(&new_view, &node_id, &checkpoints, &prepared_messages)
                .await?
            {
                return Err(anyhow!("Invalid view change message from node {}", node_id));
            }

            // Store the view change message
            self.store_view_change_message(message_clone);

            // Check if we have enough view change messages (2f+1)
            let view_change_count = self.count_view_change_messages(new_view);
            if view_change_count >= 2 * self.config.fault_tolerance + 1 {
                // If we're the new primary, send new view message
                if self.get_primary(new_view) == self.node_id {
                    self.send_new_view(new_view).await?;
                }

                // Transition to new view
                self.transition_to_view(new_view).await?;
            }
        }

        Ok(())
    }

    /// Handle new view message
    async fn handle_new_view(&self, message: BftMessage) -> Result<()> {
        if let BftMessage::NewView {
            view,
            view_changes,
            pre_prepares,
        } = message
        {
            let current_view = *self.view.read();

            // Only process if this is for a future view
            if view <= current_view {
                return Ok(());
            }

            // Verify the new view message
            if !self
                .verify_new_view(&view, &view_changes, &pre_prepares)
                .await?
            {
                return Err(anyhow!("Invalid new view message"));
            }

            // Transition to the new view
            self.transition_to_view(view).await?;

            // Process any pre-prepare messages included
            for pre_prepare in pre_prepares {
                if let BftMessage::PrePrepare {
                    view: pp_view,
                    sequence,
                    digest,
                    request,
                } = pre_prepare
                {
                    if pp_view == view {
                        self.handle_pre_prepare(
                            self.get_primary(view),
                            pp_view,
                            sequence,
                            digest,
                            *request,
                        )
                        .await?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Verify view change message authenticity and validity
    async fn verify_view_change(
        &self,
        new_view: &ViewNumber,
        node_id: &NodeId,
        checkpoints: &[CheckpointProof],
        prepared_messages: &[PreparedProof],
    ) -> Result<bool> {
        // Verify the view number is increasing
        if *new_view <= *self.view.read() {
            return Ok(false);
        }

        // Verify checkpoints
        for checkpoint in checkpoints {
            if checkpoint.signatures.len() < 2 * self.config.fault_tolerance + 1 {
                return Ok(false);
            }
        }

        // Verify prepared messages
        for prepared in prepared_messages {
            if prepared.prepares.len() < 2 * self.config.fault_tolerance {
                return Ok(false);
            }
        }

        // Verify cryptographic signatures if enabled
        if self.config.enable_signatures {
            if !self
                .verify_signatures_for_view_change(node_id, checkpoints, prepared_messages)
                .await?
            {
                let mut detector = self.byzantine_detector.write();
                detector.report_signature_failure(*node_id);
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Verify cryptographic signatures for view change messages
    async fn verify_signatures_for_view_change(
        &self,
        node_id: &NodeId,
        checkpoints: &[CheckpointProof],
        prepared_messages: &[PreparedProof],
    ) -> Result<bool> {
        // Get the public key for the node
        let public_key = {
            let nodes = self.nodes.read();
            let node_info = nodes
                .get(node_id)
                .ok_or_else(|| anyhow!("Unknown node ID: {}", node_id))?;
            node_info.public_key.clone()
        }; // Drop the lock here

        if let Some(public_key) = public_key {
            // Verify checkpoint signatures
            for checkpoint in checkpoints {
                if !self
                    .verify_checkpoint_signatures(checkpoint, &public_key)
                    .await?
                {
                    return Ok(false);
                }
            }

            // Verify prepared message signatures
            for prepared in prepared_messages {
                if !self
                    .verify_prepared_signatures(prepared, &public_key)
                    .await?
                {
                    return Ok(false);
                }
            }

            return Ok(true);
        }

        // If no public key available, skip signature verification but log warning
        tracing::warn!(
            "No public key available for node {}, skipping signature verification",
            node_id
        );
        Ok(true)
    }

    /// Verify signatures in a checkpoint proof
    async fn verify_checkpoint_signatures(
        &self,
        checkpoint: &CheckpointProof,
        _public_key: &[u8],
    ) -> Result<bool> {
        // Simplified signature verification - in real implementation would use proper cryptographic verification
        // For now, just check that we have enough signatures
        let required_sigs = 2 * self.config.fault_tolerance + 1;

        if checkpoint.signatures.len() < required_sigs {
            return Ok(false);
        }

        // In a real implementation, you would:
        // 1. Reconstruct the checkpoint message
        // 2. Hash the message content
        // 3. Verify each signature against the corresponding node's public key
        // 4. Ensure signatures are from valid nodes

        // For demonstration, check that signatures are not empty (basic validation)
        for (signing_node_id, signature) in &checkpoint.signatures {
            {
                let nodes = self.nodes.read();
                if !nodes.contains_key(signing_node_id) {
                    return Ok(false); // Invalid node
                }
            } // Drop the lock here

            // In real implementation: verify signature cryptographically
            if signature.is_empty() {
                return Ok(false); // Empty signature is invalid
            }
        }

        Ok(true)
    }

    /// Verify signatures in prepared message proofs
    async fn verify_prepared_signatures(
        &self,
        prepared: &PreparedProof,
        _public_key: &[u8],
    ) -> Result<bool> {
        // Verify that we have enough prepare messages
        let required_prepares = 2 * self.config.fault_tolerance;

        if prepared.prepares.len() < required_prepares {
            return Ok(false);
        }

        // Verify each prepare message signature
        for prepare_msg in &prepared.prepares {
            if let BftMessage::Prepare {
                view,
                sequence,
                digest,
                node_id,
            } = prepare_msg
            {
                // Verify consistency
                if *view != prepared.view
                    || *sequence != prepared.sequence
                    || *digest != prepared.digest
                {
                    return Ok(false);
                }

                // Verify the node exists
                let nodes = self.nodes.read();
                if !nodes.contains_key(node_id) {
                    return Ok(false);
                }

                // In real implementation: verify cryptographic signature
                // For now, basic validation passed
            } else {
                return Ok(false); // Wrong message type
            }
        }

        Ok(true)
    }

    /// Store view change message for later processing
    fn store_view_change_message(&self, message: BftMessage) {
        // TODO: Store in a dedicated view change message store
        // For now, just log it
        self.log_message(message);
    }

    /// Count view change messages for a specific view
    fn count_view_change_messages(&self, view: ViewNumber) -> usize {
        // TODO: Count actual view change messages from stored messages
        // Simplified implementation - in real system would count from dedicated store
        let log = self.message_log.read();
        log.iter()
            .filter(|msg| {
                if let BftMessage::ViewChange { new_view, .. } = msg {
                    *new_view == view
                } else {
                    false
                }
            })
            .count()
    }

    /// Send new view message (called by new primary)
    async fn send_new_view(&self, view: ViewNumber) -> Result<()> {
        // Collect view change messages
        let view_changes = self.collect_view_change_messages(view);

        // Create pre-prepare messages for any prepared operations
        let pre_prepares = self.create_new_view_pre_prepares(view, &view_changes);

        let new_view_message = BftMessage::NewView {
            view,
            view_changes,
            pre_prepares,
        };

        self.broadcast_message(new_view_message).await
    }

    /// Collect view change messages for a specific view
    fn collect_view_change_messages(&self, view: ViewNumber) -> Vec<BftMessage> {
        let log = self.message_log.read();
        log.iter()
            .filter(|msg| {
                if let BftMessage::ViewChange { new_view, .. } = msg {
                    *new_view == view
                } else {
                    false
                }
            })
            .cloned()
            .collect()
    }

    /// Create pre-prepare messages for new view based on prepared operations
    fn create_new_view_pre_prepares(
        &self,
        view: ViewNumber,
        view_changes: &[BftMessage],
    ) -> Vec<BftMessage> {
        let mut pre_prepares = Vec::new();

        // Collect all prepared operations from view change messages
        for msg in view_changes {
            if let BftMessage::ViewChange {
                prepared_messages, ..
            } = msg
            {
                for prepared in prepared_messages {
                    // Create new pre-prepare with updated view number
                    if let BftMessage::PrePrepare {
                        sequence,
                        digest,
                        request,
                        ..
                    } = prepared.pre_prepare.as_ref()
                    {
                        let new_pre_prepare = BftMessage::PrePrepare {
                            view,
                            sequence: *sequence,
                            digest: digest.clone(),
                            request: request.clone(),
                        };
                        pre_prepares.push(new_pre_prepare);
                    }
                }
            }
        }

        pre_prepares
    }

    /// Verify new view message
    async fn verify_new_view(
        &self,
        view: &ViewNumber,
        view_changes: &[BftMessage],
        pre_prepares: &[BftMessage],
    ) -> Result<bool> {
        // Verify we have enough view change messages (2f+1)
        if view_changes.len() < 2 * self.config.fault_tolerance + 1 {
            return Ok(false);
        }

        // Verify all view change messages are for this view
        for msg in view_changes {
            if let BftMessage::ViewChange { new_view, .. } = msg {
                if *new_view != *view {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }

        // Verify pre-prepare messages are valid
        for msg in pre_prepares {
            if let BftMessage::PrePrepare { view: pp_view, .. } = msg {
                if *pp_view != *view {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Transition to new view
    async fn transition_to_view(&self, new_view: ViewNumber) -> Result<()> {
        let mut view = self.view.write();
        *view = new_view;

        // Reset view change timer
        let mut timer = self.view_change_timer.lock();
        *timer = Some(Instant::now());

        // Reset phase to idle
        let mut phase = self.phase.write();
        *phase = Phase::Idle;

        // Clear any view-specific state
        self.clear_view_state(new_view);

        println!("Node {} transitioned to view {}", self.node_id, new_view);
        Ok(())
    }

    /// Clear view-specific state during view transition
    fn clear_view_state(&self, new_view: ViewNumber) {
        // Remove states from previous views
        self.states.retain(|(view, _), _| *view >= new_view);
    }

    /// Broadcast message to all nodes with Byzantine behavior analysis
    async fn broadcast_message(&self, message: BftMessage) -> Result<()> {
        let mut recipients = Vec::new();

        // Scope the locks to avoid holding them across await points
        {
            let nodes = self.nodes.read();

            for (node_id, _) in nodes.iter() {
                if *node_id != self.node_id {
                    // Check if node is suspected before sending
                    let is_suspected = {
                        let detector = self.byzantine_detector.read();
                        detector.is_suspected(*node_id) || detector.is_partitioned(*node_id)
                    };

                    if !is_suspected {
                        self.message_tx.send((*node_id, message.clone()))?;
                        recipients.push(*node_id);
                    } else {
                        tracing::warn!(
                            "Skipping message broadcast to suspected/partitioned node {}",
                            node_id
                        );
                    }
                }
            }
        } // nodes lock dropped here

        // Analyze broadcast patterns for potential attacks
        self.analyze_broadcast_pattern(&recipients).await;

        Ok(())
    }

    /// Analyze broadcast patterns for Byzantine behavior detection
    async fn analyze_broadcast_pattern(&self, recipients: &[NodeId]) {
        // Check for selective message sending (potential partition attack)
        let total_nodes = {
            let nodes = self.nodes.read();
            nodes.len() - 1 // Exclude self
        };
        let sent_ratio = recipients.len() as f64 / total_nodes as f64;

        if sent_ratio < 0.7 {
            // If sending to less than 70% of nodes, it might be suspicious
            let mut detector = self.byzantine_detector.write();
            detector.report_inconsistent_pattern(self.node_id);
        }

        // Check for collusion patterns - if multiple nodes are consistently excluded
        if recipients.len() < total_nodes {
            let excluded_nodes: Vec<_> = {
                let nodes = self.nodes.read();
                nodes
                    .keys()
                    .filter(|&&id| id != self.node_id && !recipients.contains(&id))
                    .copied()
                    .collect()
            };

            if !excluded_nodes.is_empty() {
                let mut detector = self.byzantine_detector.write();
                detector.check_collusion(excluded_nodes);
            }
        }
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
        let checkpoints = self.checkpoints.read().values().cloned().collect();

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
            byzantine_detector: self.byzantine_detector.clone(),
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
                let results: Vec<SerializableTriple> = self
                    .triples
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
            (
                t.subject().to_string(),
                t.predicate().to_string(),
                t.object().to_string(),
            )
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
            ObjectType::NamedNode => crate::model::Object::NamedNode(NamedNode::new(&st.object)?),
            ObjectType::BlankNode => crate::model::Object::BlankNode(BlankNode::new(&st.object)?),
            ObjectType::Literal { datatype, language } => {
                if let Some(lang) = language {
                    crate::model::Object::Literal(Literal::new_language_tagged_literal(
                        &st.object, &lang,
                    )?)
                } else if let Some(dt) = datatype {
                    crate::model::Object::Literal(Literal::new_typed(
                        &st.object,
                        NamedNode::new(&dt)?,
                    ))
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
            NodeInfo {
                id: 0,
                address: "node0".to_string(),
                public_key: None,
            },
            NodeInfo {
                id: 1,
                address: "node1".to_string(),
                public_key: None,
            },
            NodeInfo {
                id: 2,
                address: "node2".to_string(),
                public_key: None,
            },
            NodeInfo {
                id: 3,
                address: "node3".to_string(),
                public_key: None,
            },
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
                object_type: ObjectType::Literal {
                    datatype: None,
                    language: None,
                },
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
            object_type: ObjectType::Literal {
                datatype: None,
                language: None,
            },
        };

        let result = state_machine
            .execute(RdfOperation::Insert(triple.clone()))
            .unwrap();
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
