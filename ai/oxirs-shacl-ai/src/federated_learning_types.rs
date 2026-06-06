//! Types, structs, enums, and client/server models for Federated Learning.
//!
//! This module contains all the data model definitions used across the
//! federated learning subsystem: nodes, updates, configurations, consensus
//! types, and privacy proof types.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Federated learning node representing a knowledge graph participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedNode {
    /// Unique node identifier
    pub node_id: Uuid,
    /// Network address
    pub address: SocketAddr,
    /// Node reputation score
    pub reputation: f64,
    /// Data contribution score
    pub contribution_score: f64,
    /// Privacy level
    pub privacy_level: PrivacyLevel,
    /// Computational capacity
    pub capacity: ComputationalCapacity,
    /// Last seen timestamp
    pub last_seen: SystemTime,
    /// Node metadata
    pub metadata: NodeMetadata,
}

impl FederatedNode {
    /// Create a new federated node
    pub fn new(address: SocketAddr, privacy_level: PrivacyLevel) -> Self {
        Self {
            node_id: Uuid::new_v4(),
            address,
            reputation: 1.0,
            contribution_score: 0.0,
            privacy_level,
            capacity: ComputationalCapacity::default(),
            last_seen: SystemTime::now(),
            metadata: NodeMetadata::default(),
        }
    }

    /// Check if node is active
    pub fn is_active(&self) -> bool {
        self.last_seen
            .elapsed()
            .unwrap_or(Duration::from_secs(u64::MAX))
            < Duration::from_secs(300) // 5 minutes
    }

    /// Update node activity
    pub fn update_activity(&mut self) {
        self.last_seen = SystemTime::now();
    }

    /// Calculate node trust score
    pub fn trust_score(&self) -> f64 {
        (self.reputation * 0.6 + self.contribution_score * 0.4).clamp(0.0, 1.0)
    }
}

/// Privacy levels for federated learning
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PrivacyLevel {
    /// Full data sharing
    Open,
    /// Share aggregated statistics only
    Statistical,
    /// Share differential privacy data
    DifferentialPrivacy { epsilon: f64 },
    /// Share homomorphically encrypted data
    HomomorphicEncryption,
    /// Maximum privacy with secure multi-party computation
    SecureMultiParty,
}

/// Computational capacity of a federated node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalCapacity {
    /// CPU cores available
    pub cpu_cores: usize,
    /// RAM available in MB
    pub ram_mb: usize,
    /// GPU availability
    pub has_gpu: bool,
    /// Network bandwidth in Mbps
    pub bandwidth_mbps: f64,
    /// Processing power score
    pub processing_score: f64,
}

impl Default for ComputationalCapacity {
    fn default() -> Self {
        Self {
            cpu_cores: 4,
            ram_mb: 8192,
            has_gpu: false,
            bandwidth_mbps: 100.0,
            processing_score: 1.0,
        }
    }
}

/// Node metadata for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Knowledge graph domain
    pub domain: String,
    /// Data size in triples
    pub data_size: usize,
    /// Supported languages
    pub languages: HashSet<String>,
    /// Schema ontologies used
    pub ontologies: HashSet<String>,
    /// Node version
    pub version: String,
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            domain: "general".to_string(),
            data_size: 0,
            languages: HashSet::new(),
            ontologies: HashSet::new(),
            version: "1.0.0".to_string(),
        }
    }
}

/// Federated learning model update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedUpdate {
    /// Update identifier
    pub update_id: Uuid,
    /// Source node
    pub node_id: Uuid,
    /// Model parameters delta
    pub parameter_delta: ModelParameterDelta,
    /// Training metadata
    pub training_metadata: TrainingMetadata,
    /// Privacy proof
    pub privacy_proof: Option<PrivacyProof>,
    /// Digital signature
    pub signature: Vec<u8>,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Model parameter delta for federated updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameterDelta {
    /// Parameter changes
    pub deltas: HashMap<String, f64>,
    /// Gradient information
    pub gradients: HashMap<String, Vec<f64>>,
    /// Learning rate used
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of local epochs
    pub local_epochs: usize,
}

/// Training metadata for federated updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// Number of training samples
    pub sample_count: usize,
    /// Training loss
    pub loss: f64,
    /// Validation accuracy
    pub accuracy: f64,
    /// Training time in seconds
    pub training_time: f64,
    /// Data quality score
    pub data_quality: f64,
}

/// Privacy proof for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyProof {
    /// Proof type
    pub proof_type: PrivacyProofType,
    /// Proof data
    pub proof_data: Vec<u8>,
    /// Verification hash
    pub verification_hash: String,
}

/// Types of privacy proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyProofType {
    DifferentialPrivacy,
    HomomorphicEncryption,
    SecureAggregation,
    ZeroKnowledge,
}

/// Federated learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedLearningConfig {
    /// Minimum number of nodes for consensus
    pub min_nodes_for_consensus: usize,
    /// Update aggregation strategy
    pub aggregation_strategy: AggregationStrategy,
    /// Maximum staleness for updates
    pub max_staleness: Duration,
    /// Byzantine fault tolerance
    pub byzantine_tolerance: f64,
    /// Privacy preservation settings
    pub privacy_config: PrivacyConfig,
}

impl Default for FederatedLearningConfig {
    fn default() -> Self {
        Self {
            min_nodes_for_consensus: 3,
            aggregation_strategy: AggregationStrategy::FederatedAveraging,
            max_staleness: Duration::from_secs(300),
            byzantine_tolerance: 0.33,
            privacy_config: PrivacyConfig::default(),
        }
    }
}

/// Aggregation strategies for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Simple federated averaging
    FederatedAveraging,
    /// Weighted averaging by data size
    WeightedAveraging,
    /// Byzantine-robust aggregation
    ByzantineRobust,
    /// Adaptive aggregation based on node performance
    Adaptive,
}

/// Privacy configuration for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Enable differential privacy
    pub enable_differential_privacy: bool,
    /// Privacy budget (epsilon)
    pub epsilon: f64,
    /// Enable secure aggregation
    pub enable_secure_aggregation: bool,
    /// Enable homomorphic encryption
    pub enable_homomorphic_encryption: bool,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            enable_differential_privacy: true,
            epsilon: 1.0,
            enable_secure_aggregation: false,
            enable_homomorphic_encryption: false,
        }
    }
}

/// Global model state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalModel {
    /// Model version
    pub version: u64,
    /// Learned shapes
    pub shapes: Vec<oxirs_shacl::Shape>,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Last update timestamp
    pub last_updated: SystemTime,
    /// Contributing nodes
    pub contributors: HashSet<Uuid>,
}

impl Default for GlobalModel {
    fn default() -> Self {
        Self {
            version: 1,
            shapes: Vec::new(),
            parameters: HashMap::new(),
            last_updated: SystemTime::now(),
            contributors: HashSet::new(),
        }
    }
}

impl GlobalModel {
    /// Update shapes in the global model
    pub fn update_shapes(&mut self, shapes: Vec<oxirs_shacl::Shape>) {
        self.shapes = shapes;
        self.version += 1;
        self.last_updated = SystemTime::now();
    }
}

/// Consensus algorithms
#[derive(Debug, Clone)]
pub enum ConsensusAlgorithm {
    /// Practical Byzantine Fault Tolerance
    PBFT,
    /// Raft consensus algorithm
    RAFT,
    /// Proof of Stake
    PoS,
}

/// Voting round for consensus
#[derive(Debug, Clone)]
pub struct VotingRound {
    /// Round identifier
    pub round_id: Uuid,
    /// Participating nodes
    pub participants: HashSet<Uuid>,
    /// Votes cast
    pub votes: HashMap<Uuid, Vote>,
    /// Result
    pub result: Option<VotingResult>,
}

/// Vote in consensus round
#[derive(Debug, Clone)]
pub struct Vote {
    /// Voter node ID
    pub voter: Uuid,
    /// Vote content
    pub content: VoteContent,
    /// Signature
    pub signature: Vec<u8>,
}

/// Vote content
#[derive(Debug, Clone)]
pub enum VoteContent {
    Accept(AggregatedParameters),
    Reject(String),
    Abstain,
}

/// Voting result
#[derive(Debug, Clone)]
pub enum VotingResult {
    Accepted(AggregatedParameters),
    Rejected,
    NoConsensus,
}

/// Aggregated parameters from consensus
#[derive(Debug, Clone)]
pub struct AggregatedParameters {
    /// Aggregated model parameters
    pub parameters: HashMap<String, f64>,
    /// Confidence in aggregation
    pub confidence: f64,
    /// Contributing nodes
    pub contributing_nodes: HashSet<Uuid>,
}

/// Federation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationStats {
    /// Total nodes in federation
    pub total_nodes: usize,
    /// Active nodes
    pub active_nodes: usize,
    /// Total updates exchanged
    pub total_updates: usize,
    /// Global model version
    pub global_model_version: u64,
    /// Average node reputation
    pub average_reputation: f64,
    /// Network health score
    pub network_health: f64,
}

/// Byzantine fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineFaultToleranceConfig {
    /// Enable threshold signatures
    pub enable_threshold_signatures: bool,
    /// Signature verification timeout
    pub signature_timeout: Duration,
    /// Byzantine detection sensitivity
    pub detection_sensitivity: f64,
    /// Reputation decay for Byzantine nodes
    pub reputation_decay_rate: f64,
}

impl Default for ByzantineFaultToleranceConfig {
    fn default() -> Self {
        Self {
            enable_threshold_signatures: true,
            signature_timeout: Duration::from_secs(30),
            detection_sensitivity: 0.8,
            reputation_decay_rate: 0.1,
        }
    }
}

/// Threshold signature share
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdSignatureShare {
    pub node_id: Uuid,
    pub signature_share: Vec<u8>,
    pub timestamp: SystemTime,
    pub update_hash: String,
}

/// Byzantine verification result
#[derive(Debug, Clone)]
pub struct ByzantineVerificationResult {
    pub verified: bool,
    pub valid_signatures: usize,
    pub byzantine_nodes: HashSet<Uuid>,
    pub threshold_met: bool,
    pub byzantine_tolerance: bool,
}

/// Consensus message types for HoneyBadger BFT
#[derive(Debug, Clone)]
pub enum ConsensusMessage {
    ReliableBroadcast(ReliableBroadcastMessage),
    BinaryAgreement(BinaryAgreementMessage),
    ThresholdDecryption(ThresholdDecryptionShare),
}

/// Reliable broadcast message
#[derive(Debug, Clone)]
pub struct ReliableBroadcastMessage {
    pub sender: Uuid,
    pub epoch: u64,
    pub data: Vec<u8>,
    pub merkle_proof: Vec<u8>,
}

/// Binary agreement message
#[derive(Debug, Clone)]
pub struct BinaryAgreementMessage {
    pub sender: Uuid,
    pub epoch: u64,
    pub value: bool,
    pub signature: Vec<u8>,
}

/// Threshold decryption share
#[derive(Debug, Clone)]
pub struct ThresholdDecryptionShare {
    pub sender: Uuid,
    pub epoch: u64,
    pub decryption_share: Vec<u8>,
}

/// Secret sharing schemes
#[derive(Debug, Clone)]
pub enum SecretSharingScheme {
    Shamir {
        threshold: usize,
        total_shares: usize,
    },
    Additive {
        num_shares: usize,
    },
}

/// SMPC protocols
#[derive(Debug, Clone, PartialEq)]
pub enum SMPCProtocol {
    SecureAggregation,
    PrivateSetIntersection,
    SecureComparison,
}

/// Secret share
#[derive(Debug, Clone)]
pub struct SecretShare {
    pub party_id: usize,
    pub share_value: f64,
    pub threshold: usize,
    pub total_shares: usize,
}

/// Secure channel for SMPC communication
#[derive(Debug)]
pub struct SecureChannel {
    pub party_id: Uuid,
    pub encryption_key: Vec<u8>,
    pub authentication_key: Vec<u8>,
}

/// Secure aggregation result
#[derive(Debug, Clone)]
pub struct SecureAggregationResult {
    pub aggregated_parameters: HashMap<String, f64>,
    pub participating_parties: HashSet<Uuid>,
    pub security_level: SecurityLevel,
    pub computation_rounds: usize,
}

/// Security level for SMPC
#[derive(Debug, Clone, PartialEq)]
pub enum SecurityLevel {
    Low,
    Medium,
    High,
}

/// Noise mechanisms for differential privacy
#[derive(Debug, Clone, PartialEq)]
pub enum NoiseMechanism {
    Laplace,
    Gaussian,
    Exponential,
}

/// Privacy report for differential privacy application
#[derive(Debug, Clone)]
pub struct PrivacyReport {
    pub epsilon_used: f64,
    pub delta_used: f64,
    pub mechanism_used: NoiseMechanism,
    pub noise_added: f64,
    pub utility_preserved: f64,
}

/// Privacy spending record
#[derive(Debug, Clone)]
pub struct PrivacySpending {
    pub amount: f64,
    pub timestamp: SystemTime,
    pub remaining: f64,
}

/// Privacy budget tracker
#[derive(Debug, Clone)]
pub struct PrivacyBudgetTracker {
    pub(crate) total_budget: f64,
    pub(crate) spent_budget: f64,
    pub(crate) spending_history: Vec<PrivacySpending>,
}
