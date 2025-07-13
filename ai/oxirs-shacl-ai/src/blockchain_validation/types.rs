//! Core types for blockchain validation
//!
//! This module contains the fundamental data structures and enums used
//! throughout the blockchain validation system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use uuid::Uuid;

/// Validation mode for blockchain operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationMode {
    /// On-chain validation using smart contracts
    OnChain,
    /// Off-chain validation with on-chain attestation
    OffChain,
    /// Hybrid validation combining both approaches
    Hybrid,
    /// Zero-knowledge proof validation
    ZeroKnowledge,
    /// Consensus-based validation
    Consensus,
}

/// Cross-chain aggregation strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossChainAggregation {
    /// Union of all validation results
    Union,
    /// Intersection of validation results
    Intersection,
    /// Majority consensus
    Majority,
    /// Weighted consensus based on network stakes
    WeightedConsensus,
    /// Custom aggregation logic
    Custom,
}

/// Privacy levels for blockchain validation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivacyLevel {
    /// Enable zero-knowledge proofs
    pub enable_zk_proofs: bool,
    /// Enable homomorphic encryption
    pub enable_homomorphic_encryption: bool,
    /// Enable secure multi-party computation
    pub enable_smpc: bool,
    /// Anonymity level (0-100)
    pub anonymity_level: u8,
    /// Data minimization enabled
    pub data_minimization: bool,
}

impl Default for PrivacyLevel {
    fn default() -> Self {
        Self {
            enable_zk_proofs: false,
            enable_homomorphic_encryption: false,
            enable_smpc: false,
            anonymity_level: 0,
            data_minimization: true,
        }
    }
}

/// Validation request for blockchain processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRequest {
    /// Unique request identifier
    pub id: Uuid,
    /// Validation data
    pub data: ValidationData,
    /// Shape constraints to validate against
    pub constraints: Vec<ShapeConstraint>,
    /// Validation mode
    pub mode: ValidationMode,
    /// Privacy requirements
    pub privacy_level: PrivacyLevel,
    /// Requesting blockchain network
    pub network_id: String,
    /// Timestamp of the request
    pub timestamp: SystemTime,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// Validation data for blockchain processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationData {
    /// RDF data in N-Triples format
    pub rdf_data: String,
    /// Data graph URI
    pub graph_uri: String,
    /// Data size in bytes
    pub size_bytes: usize,
    /// Data hash for integrity verification
    pub data_hash: String,
    /// Data format specification
    pub format: String,
}

/// Shape constraint definition for blockchain validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeConstraint {
    /// Shape identifier
    pub shape_id: String,
    /// Shape definition in Turtle format
    pub shape_definition: String,
    /// Constraint priority (higher = more important)
    pub priority: u8,
    /// Whether this constraint is mandatory
    pub mandatory: bool,
    /// Associated smart contract address (if any)
    pub contract_address: Option<String>,
}

/// Result of validation submission to blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionResult {
    /// Transaction hash
    pub transaction_hash: String,
    /// Block number where transaction was included
    pub block_number: u64,
    /// Gas used for the transaction
    pub gas_used: u64,
    /// Transaction status
    pub status: TransactionStatus,
    /// Network confirmation count
    pub confirmations: u32,
    /// Estimated finality time
    pub finality_time: Option<SystemTime>,
}

/// Transaction status on blockchain
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionStatus {
    /// Transaction is pending
    Pending,
    /// Transaction confirmed successfully
    Confirmed,
    /// Transaction failed
    Failed,
    /// Transaction reverted
    Reverted,
}

/// Consensus result from distributed validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    /// Consensus algorithm used
    pub algorithm: String,
    /// Participating validators
    pub validators: Vec<ValidatorInfo>,
    /// Consensus outcome
    pub consensus_reached: bool,
    /// Validation result agreed upon
    pub agreed_result: bool,
    /// Confidence score (0.0 - 1.0)
    pub confidence_score: f64,
    /// Number of rounds required
    pub consensus_rounds: u32,
    /// Time taken to reach consensus
    pub consensus_duration_ms: u64,
}

/// Validator information for consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    /// Validator identifier
    pub validator_id: String,
    /// Validator's stake weight
    pub stake_weight: f64,
    /// Validator's vote
    pub vote: bool,
    /// Validator's reputation score
    pub reputation: f64,
    /// Network address
    pub network_address: String,
}

/// Comprehensive blockchain validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockchainValidationResult {
    /// Request identifier
    pub request_id: Uuid,
    /// Overall validation success
    pub success: bool,
    /// Validation details per constraint
    pub constraint_results: HashMap<String, ConstraintValidationResult>,
    /// Submission result to blockchain
    pub submission_result: SubmissionResult,
    /// Consensus result (if applicable)
    pub consensus_result: Option<ConsensusResult>,
    /// Cross-chain validation results
    pub cross_chain_results: Vec<CrossChainValidationResult>,
    /// Privacy-preserving validation result
    pub privacy_result: Option<PrivateValidationResult>,
    /// Smart contract validation results
    pub smart_contract_results: Vec<SmartContractValidationResult>,
    /// Gas costs incurred
    pub total_gas_cost: u64,
    /// Validation timestamp
    pub validation_timestamp: SystemTime,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Result of validating a specific constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintValidationResult {
    /// Constraint identifier
    pub constraint_id: String,
    /// Validation success for this constraint
    pub valid: bool,
    /// Violation messages
    pub violations: Vec<String>,
    /// Constraint evaluation time
    pub evaluation_time_ms: u64,
    /// Gas cost for this constraint
    pub gas_cost: u64,
}

/// Smart contract validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartContractValidationResult {
    /// Contract address
    pub contract_address: String,
    /// Contract validation function called
    pub function_name: String,
    /// Validation result from contract
    pub result: bool,
    /// Return data from contract
    pub return_data: Vec<u8>,
    /// Gas used by contract execution
    pub gas_used: u64,
    /// Contract execution status
    pub execution_status: ContractExecutionStatus,
}

/// Status of smart contract execution
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContractExecutionStatus {
    /// Execution successful
    Success,
    /// Execution failed
    Failed,
    /// Execution reverted
    Reverted,
    /// Out of gas
    OutOfGas,
}

/// Cross-chain validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainValidationResult {
    /// Source chain identifier
    pub chain_id: String,
    /// Chain-specific validation result
    pub validation_result: bool,
    /// Cross-chain message hash
    pub message_hash: String,
    /// Relay confirmation
    pub relay_confirmed: bool,
    /// Validation timestamp on source chain
    pub source_timestamp: SystemTime,
    /// Cross-chain consensus statistics
    pub consensus_stats: CrossChainConsensusStats,
}

/// Cross-chain consensus statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainConsensusStats {
    /// Number of participating chains
    pub participating_chains: u32,
    /// Number of chains that agreed
    pub agreeing_chains: u32,
    /// Cross-chain consensus confidence
    pub cross_chain_confidence: f64,
    /// Time to reach cross-chain consensus
    pub consensus_time_ms: u64,
}

/// Privacy-preserving validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateValidationResult {
    /// Zero-knowledge proof of validation
    pub zk_proof: Option<Vec<u8>>,
    /// Homomorphic encryption result
    pub he_result: Option<Vec<u8>>,
    /// Secure multi-party computation result
    pub smpc_result: Option<bool>,
    /// Privacy level achieved
    pub achieved_privacy_level: PrivacyLevel,
    /// Privacy computation cost
    pub privacy_cost: u64,
}

/// Blockchain event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockchainEvent {
    /// Validation request submitted
    ValidationSubmitted {
        request_id: Uuid,
        network_id: String,
        timestamp: SystemTime,
    },
    /// Validation completed
    ValidationCompleted {
        request_id: Uuid,
        result: BlockchainValidationResult,
    },
    /// Consensus reached
    ConsensusReached {
        request_id: Uuid,
        consensus_result: ConsensusResult,
    },
    /// Cross-chain validation completed
    CrossChainCompleted {
        request_id: Uuid,
        chain_results: Vec<CrossChainValidationResult>,
    },
    /// Smart contract validation completed
    SmartContractCompleted {
        request_id: Uuid,
        contract_results: Vec<SmartContractValidationResult>,
    },
    /// Network error occurred
    NetworkError {
        network_id: String,
        error_message: String,
        timestamp: SystemTime,
    },
}

/// Network performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Network identifier
    pub network_id: String,
    /// Current block height
    pub block_height: u64,
    /// Average block time
    pub avg_block_time_ms: u64,
    /// Network hash rate
    pub hash_rate: f64,
    /// Gas price statistics
    pub gas_price_gwei: f64,
    /// Network congestion level (0.0 - 1.0)
    pub congestion_level: f64,
    /// Active validator count
    pub active_validators: u32,
    /// Network uptime percentage
    pub uptime_percentage: f64,
}

/// Validator performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorPerformance {
    /// Validator identifier
    pub validator_id: String,
    /// Success rate for validations
    pub success_rate: f64,
    /// Average response time
    pub avg_response_time_ms: u64,
    /// Reputation score
    pub reputation_score: f64,
    /// Total validations performed
    pub total_validations: u64,
    /// Stake amount
    pub stake_amount: f64,
    /// Last activity timestamp
    pub last_activity: SystemTime,
}
