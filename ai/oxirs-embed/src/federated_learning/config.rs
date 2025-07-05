//! Configuration structures for federated learning
//!
//! This module contains all configuration types for federated learning including
//! privacy settings, communication protocols, security configurations, and
//! personalization options.

use crate::ModelConfig;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Configuration for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedConfig {
    /// Base model configuration
    pub base_config: ModelConfig,
    /// Number of federated participants
    pub num_participants: usize,
    /// Communication rounds
    pub communication_rounds: usize,
    /// Local training epochs per round
    pub local_epochs: usize,
    /// Minimum participants required for aggregation
    pub min_participants: usize,
    /// Differential privacy configuration
    pub privacy_config: PrivacyConfig,
    /// Aggregation strategy
    pub aggregation_strategy: AggregationStrategy,
    /// Communication optimization
    pub communication_config: CommunicationConfig,
    /// Secure computation settings
    pub security_config: SecurityConfig,
    /// Personalization settings
    pub personalization_config: PersonalizationConfig,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            base_config: ModelConfig::default(),
            num_participants: 10,
            communication_rounds: 100,
            local_epochs: 5,
            min_participants: 5,
            privacy_config: PrivacyConfig::default(),
            aggregation_strategy: AggregationStrategy::FederatedAveraging,
            communication_config: CommunicationConfig::default(),
            security_config: SecurityConfig::default(),
            personalization_config: PersonalizationConfig::default(),
        }
    }
}

/// Privacy-preserving configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Enable differential privacy
    pub enable_differential_privacy: bool,
    /// Privacy budget (epsilon)
    pub epsilon: f64,
    /// Privacy delta parameter
    pub delta: f64,
    /// Noise mechanism
    pub noise_mechanism: NoiseMechanism,
    /// Gradient clipping threshold
    pub clipping_threshold: f64,
    /// Local privacy budget
    pub local_epsilon: f64,
    /// Global privacy budget
    pub global_epsilon: f64,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            enable_differential_privacy: true,
            epsilon: 1.0,
            delta: 1e-5,
            noise_mechanism: NoiseMechanism::Gaussian,
            clipping_threshold: 1.0,
            local_epsilon: 0.5,
            global_epsilon: 0.5,
        }
    }
}

/// Noise mechanisms for differential privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseMechanism {
    /// Gaussian noise mechanism
    Gaussian,
    /// Laplace noise mechanism
    Laplace,
    /// Exponential mechanism
    Exponential,
    /// Sparse vector technique
    SparseVector,
}

/// Aggregation strategies for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Standard federated averaging
    FederatedAveraging,
    /// Weighted federated averaging
    WeightedAveraging,
    /// Secure aggregation
    SecureAggregation,
    /// Robust aggregation (Byzantine-resistant)
    RobustAggregation,
    /// Personalized aggregation
    PersonalizedAggregation,
    /// Hierarchical aggregation
    HierarchicalAggregation,
}

/// Communication optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationConfig {
    /// Enable gradient compression
    pub enable_compression: bool,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Quantization bits
    pub quantization_bits: u8,
    /// Enable sparsification
    pub enable_sparsification: bool,
    /// Sparsity threshold
    pub sparsity_threshold: f64,
    /// Communication protocol
    pub protocol: CommunicationProtocol,
    /// Batch communication
    pub batch_communication: bool,
    /// Communication timeout (seconds)
    pub timeout_seconds: u64,
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_ratio: 0.1,
            quantization_bits: 8,
            enable_sparsification: true,
            sparsity_threshold: 0.01,
            protocol: CommunicationProtocol::Synchronous,
            batch_communication: true,
            timeout_seconds: 300,
        }
    }
}

/// Communication protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationProtocol {
    /// Synchronous communication
    Synchronous,
    /// Asynchronous communication
    Asynchronous,
    /// Semi-synchronous with staleness bounds
    SemiSynchronous { staleness_bound: usize },
    /// Peer-to-peer communication
    PeerToPeer,
}

/// Security configuration for secure computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable homomorphic encryption
    pub enable_homomorphic_encryption: bool,
    /// Encryption scheme
    pub encryption_scheme: EncryptionScheme,
    /// Enable secure multi-party computation
    pub enable_secure_mpc: bool,
    /// Verification mechanisms
    pub verification_mechanisms: Vec<VerificationMechanism>,
    /// Certificate management
    pub certificate_config: CertificateConfig,
    /// Authentication settings
    pub authentication_config: AuthenticationConfig,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_homomorphic_encryption: false,
            encryption_scheme: EncryptionScheme::CKKS,
            enable_secure_mpc: false,
            verification_mechanisms: vec![VerificationMechanism::DigitalSignature],
            certificate_config: CertificateConfig::default(),
            authentication_config: AuthenticationConfig::default(),
        }
    }
}

/// Homomorphic encryption schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionScheme {
    /// CKKS scheme for approximate arithmetic
    CKKS,
    /// BFV scheme for exact arithmetic
    BFV,
    /// SEAL implementation
    SEAL,
    /// HElib implementation
    HElib,
}

/// Verification mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationMechanism {
    /// Digital signatures
    DigitalSignature,
    /// Zero-knowledge proofs
    ZeroKnowledgeProof,
    /// Commitment schemes
    CommitmentScheme,
    /// Hash-based verification
    HashVerification,
}

/// Certificate management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateConfig {
    /// Certificate authority endpoint
    pub ca_endpoint: String,
    /// Certificate validity period (days)
    pub validity_days: u32,
    /// Key length
    pub key_length: u32,
    /// Certificate chain validation
    pub validate_chain: bool,
}

impl Default for CertificateConfig {
    fn default() -> Self {
        Self {
            ca_endpoint: "https://ca.example.com".to_string(),
            validity_days: 365,
            key_length: 2048,
            validate_chain: true,
        }
    }
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Authentication method
    pub method: AuthenticationMethod,
    /// Token expiry time (hours)
    pub token_expiry_hours: u32,
    /// Enable multi-factor authentication
    pub enable_mfa: bool,
    /// Identity provider endpoint
    pub identity_provider: String,
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            method: AuthenticationMethod::OAuth2,
            token_expiry_hours: 24,
            enable_mfa: false,
            identity_provider: "https://idp.example.com".to_string(),
        }
    }
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    /// OAuth 2.0
    OAuth2,
    /// JSON Web Tokens
    JWT,
    /// SAML
    SAML,
    /// Mutual TLS
    MTLS,
    /// API Keys
    ApiKey,
}

/// Personalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationConfig {
    /// Enable personalized models
    pub enable_personalization: bool,
    /// Personalization strategy
    pub strategy: PersonalizationStrategy,
    /// Local adaptation weight
    pub local_adaptation_weight: f64,
    /// Global model weight
    pub global_model_weight: f64,
    /// Personalization layers
    pub personalization_layers: Vec<String>,
    /// Meta-learning configuration
    pub meta_learning_config: MetaLearningConfig,
}

impl Default for PersonalizationConfig {
    fn default() -> Self {
        Self {
            enable_personalization: true,
            strategy: PersonalizationStrategy::LocalAdaptation,
            local_adaptation_weight: 0.3,
            global_model_weight: 0.7,
            personalization_layers: vec!["embedding".to_string(), "output".to_string()],
            meta_learning_config: MetaLearningConfig::default(),
        }
    }
}

/// Personalization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersonalizationStrategy {
    /// Local adaptation of global model
    LocalAdaptation,
    /// Multi-task learning
    MultiTaskLearning,
    /// Meta-learning approach
    MetaLearning,
    /// Mixture of experts
    MixtureOfExperts,
    /// Personalized layers
    PersonalizedLayers,
}

/// Meta-learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// Meta-learning algorithm
    pub algorithm: MetaLearningAlgorithm,
    /// Inner learning rate
    pub inner_learning_rate: f64,
    /// Outer learning rate
    pub outer_learning_rate: f64,
    /// Number of inner steps
    pub inner_steps: usize,
    /// Support set size
    pub support_set_size: usize,
    /// Query set size
    pub query_set_size: usize,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            algorithm: MetaLearningAlgorithm::MAML,
            inner_learning_rate: 0.01,
            outer_learning_rate: 0.001,
            inner_steps: 5,
            support_set_size: 10,
            query_set_size: 5,
        }
    }
}

/// Meta-learning algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaLearningAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML,
    /// Reptile algorithm
    Reptile,
    /// Prototypical networks
    PrototypicalNetworks,
    /// Matching networks
    MatchingNetworks,
    /// Memory-augmented neural networks
    MANN,
}

/// Training configuration for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Convergence criteria
    pub convergence_threshold: f64,
    /// Maximum global iterations
    pub max_global_iterations: usize,
    /// Patience for early stopping
    pub patience: usize,
    /// Learning rate decay
    pub learning_rate_decay: f64,
    /// Minimum learning rate
    pub min_learning_rate: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            convergence_threshold: 1e-6,
            max_global_iterations: 1000,
            patience: 10,
            learning_rate_decay: 0.95,
            min_learning_rate: 1e-6,
        }
    }
}
