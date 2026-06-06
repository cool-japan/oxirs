//! Federated Learning for Distributed SHACL Shape Learning
//!
//! This module implements federated learning capabilities allowing multiple
//! knowledge graph instances to collaboratively learn SHACL shapes while
//! preserving data privacy and enabling distributed intelligence.
//!
//! # Module layout
//!
//! - [`federated_learning_types`]: All structs, enums, and data model types.
//! - [`federated_learning_trainer`]: Training coordinator, consensus, BFT, and HoneyBadger BFT.
//! - [`federated_learning_privacy`]: Differential privacy, SMPC, and noise mechanisms.

// Re-export all public types from the sub-modules so existing code
// that imports from `federated_learning` continues to compile.

pub use crate::federated_learning_types::{
    AggregatedParameters, AggregationStrategy, BinaryAgreementMessage,
    ByzantineFaultToleranceConfig, ByzantineVerificationResult, ComputationalCapacity,
    ConsensusAlgorithm, ConsensusMessage, FederatedLearningConfig, FederatedNode, FederatedUpdate,
    FederationStats, GlobalModel, ModelParameterDelta, NodeMetadata, NoiseMechanism,
    PrivacyBudgetTracker, PrivacyConfig, PrivacyLevel, PrivacyProof, PrivacyProofType,
    PrivacyReport, PrivacySpending, ReliableBroadcastMessage, SMPCProtocol, SecretShare,
    SecretSharingScheme, SecureAggregationResult, SecureChannel, SecurityLevel,
    ThresholdDecryptionShare, ThresholdSignatureShare, TrainingMetadata, Vote, VoteContent,
    VotingResult, VotingRound,
};

pub use crate::federated_learning_trainer::{
    AdvancedByzantineFaultTolerance, ConsensusManager, FederatedLearningCoordinator,
    HoneyBadgerBFT, ReliableBroadcastState, ThresholdEncryption,
};

pub use crate::federated_learning_privacy::{
    AdvancedDifferentialPrivacy, SecureMultiPartyComputation,
};
