//! Participant management for federated learning
//!
//! This module handles participant registration, capability assessment,
//! trust scoring, and status management in federated learning systems.

use chrono::{DateTime, Utc};
use scirs2_core::ndarray_ext::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Federated participant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    /// Participant ID
    pub participant_id: Uuid,
    /// Participant name
    pub name: String,
    /// Network endpoint
    pub endpoint: String,
    /// Public key for verification
    pub public_key: String,
    /// Data statistics
    pub data_stats: DataStatistics,
    /// Capability information
    pub capabilities: ParticipantCapabilities,
    /// Trust score
    pub trust_score: f64,
    /// Last communication time
    pub last_communication: DateTime<Utc>,
    /// Status
    pub status: ParticipantStatus,
}

/// Data statistics for a participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStatistics {
    /// Number of samples
    pub num_samples: usize,
    /// Number of features
    pub num_features: usize,
    /// Data distribution summary
    pub distribution_summary: HashMap<String, f64>,
    /// Data quality metrics
    pub quality_metrics: HashMap<String, f64>,
    /// Privacy budget used
    pub privacy_budget_used: f64,
}

/// Participant capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantCapabilities {
    /// Computational power
    pub compute_power: ComputePower,
    /// Available memory (GB)
    pub available_memory_gb: f64,
    /// Network bandwidth (Mbps)
    pub network_bandwidth_mbps: f64,
    /// Supported algorithms
    pub supported_algorithms: Vec<String>,
    /// Hardware accelerators
    pub hardware_accelerators: Vec<HardwareAccelerator>,
    /// Security features
    pub security_features: Vec<SecurityFeature>,
}

/// Compute power levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputePower {
    /// Low computational resources
    Low,
    /// Medium computational resources
    Medium,
    /// High computational resources
    High,
    /// Very high computational resources
    VeryHigh,
}

/// Hardware accelerators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareAccelerator {
    /// NVIDIA GPU
    GPU,
    /// Google TPU
    TPU,
    /// Intel Neural Compute Stick
    NCS,
    /// ARM Neural Processing Unit
    NPU,
    /// FPGA acceleration
    FPGA,
}

/// Security features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityFeature {
    /// Trusted Execution Environment
    TEE,
    /// Hardware Security Module
    HSM,
    /// Secure Enclave
    SecureEnclave,
    /// Intel SGX
    IntelSGX,
    /// ARM TrustZone
    ARMTrustZone,
}

/// Participant status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ParticipantStatus {
    /// Active and available
    Active,
    /// Temporarily inactive
    Inactive,
    /// Disconnected
    Disconnected,
    /// Suspended due to issues
    Suspended,
    /// Excluded from federation
    Excluded,
}

/// Federated learning round information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedRound {
    /// Round number
    pub round_number: usize,
    /// Round start time
    pub start_time: DateTime<Utc>,
    /// Round end time
    pub end_time: Option<DateTime<Utc>>,
    /// Participating clients
    pub participants: Vec<Uuid>,
    /// Global model parameters
    pub global_parameters: HashMap<String, Array2<f32>>,
    /// Aggregated updates
    pub aggregated_updates: HashMap<String, Array2<f32>>,
    /// Round metrics
    pub metrics: RoundMetrics,
    /// Round status
    pub status: RoundStatus,
}

/// Metrics for a federated learning round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundMetrics {
    /// Number of participating clients
    pub num_participants: usize,
    /// Total training samples
    pub total_samples: usize,
    /// Average local loss
    pub avg_local_loss: f64,
    /// Global model accuracy
    pub global_accuracy: f64,
    /// Communication overhead (bytes)
    pub communication_overhead: u64,
    /// Round duration (seconds)
    pub duration_seconds: f64,
    /// Privacy budget consumed
    pub privacy_budget_consumed: f64,
    /// Convergence metrics
    pub convergence_metrics: ConvergenceMetrics,
}

/// Convergence tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    /// Parameter change magnitude
    pub parameter_change: f64,
    /// Loss improvement
    pub loss_improvement: f64,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Convergence status
    pub convergence_status: ConvergenceStatus,
    /// Estimated rounds to convergence
    pub estimated_rounds_to_convergence: Option<usize>,
}

/// Convergence status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    /// Training is progressing
    Progressing,
    /// Converged to solution
    Converged,
    /// Diverging
    Diverging,
    /// Stagnated
    Stagnated,
    /// Oscillating
    Oscillating,
}

/// Round status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoundStatus {
    /// Round is being initialized
    Initializing,
    /// Training in progress
    Training,
    /// Aggregating updates
    Aggregating,
    /// Round completed successfully
    Completed,
    /// Round failed
    Failed,
    /// Round was aborted
    Aborted,
}

/// Local training statistics for a participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalTrainingStats {
    /// Local training epochs completed
    pub epochs_completed: usize,
    /// Local training time (seconds)
    pub training_time_seconds: f64,
    /// Local loss values
    pub local_loss: f64,
    /// Local accuracy
    pub local_accuracy: f64,
    /// Number of samples used
    pub samples_used: usize,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory usage (GB)
    pub memory_usage_gb: f64,
    /// GPU usage percentage (if available)
    pub gpu_usage_percent: Option<f64>,
    /// Network bandwidth used (Mbps)
    pub network_bandwidth_used_mbps: f64,
}

/// Local model update from a participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalUpdate {
    /// Participant ID
    pub participant_id: Uuid,
    /// Round number
    pub round_number: usize,
    /// Model parameter updates
    pub parameter_updates: HashMap<String, Array2<f32>>,
    /// Number of local samples
    pub num_samples: usize,
    /// Local training statistics
    pub training_stats: LocalTrainingStats,
    /// Update timestamp
    pub timestamp: DateTime<Utc>,
    /// Data selection strategy used
    pub data_selection: DataSelectionStrategy,
}

/// Data selection strategies for local training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSelectionStrategy {
    /// Use all available data
    AllData,
    /// Random sampling
    RandomSampling { sample_rate: f64 },
    /// Stratified sampling
    StratifiedSampling {
        strata_proportions: HashMap<String, f64>,
    },
    /// Active learning selection
    ActiveLearning { uncertainty_threshold: f64 },
    /// Importance sampling
    ImportanceSampling { importance_weights: Vec<f64> },
}

/// Global model state in federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalModelState {
    /// Model parameters
    pub parameters: HashMap<String, Array2<f32>>,
    /// Global training round
    pub global_round: usize,
    /// Model version
    pub model_version: String,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Model performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Participant contributions
    pub participant_contributions: HashMap<Uuid, f64>,
}

/// Local model state for a participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelState {
    /// Participant ID
    pub participant_id: Uuid,
    /// Local model parameters
    pub parameters: HashMap<String, Array2<f32>>,
    /// Personalized layers
    pub personalized_parameters: HashMap<String, Array2<f32>>,
    /// Global round synchronized to
    pub synchronized_round: usize,
    /// Local adaptation steps performed
    pub local_adaptation_steps: usize,
    /// Last synchronization time
    pub last_sync_time: DateTime<Utc>,
}

/// Privacy metrics for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyMetrics {
    /// Total privacy budget spent
    pub total_budget_spent: f64,
    /// Privacy budget per participant
    pub participant_budget_usage: HashMap<Uuid, f64>,
    /// Differential privacy guarantees
    pub dp_guarantees: HashMap<String, f64>,
    /// Privacy violations detected
    pub privacy_violations: Vec<PrivacyViolation>,
    /// Privacy risk assessment
    pub privacy_risk_score: f64,
}

/// Privacy violation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyViolation {
    /// Violation type
    pub violation_type: PrivacyViolationType,
    /// Participant involved
    pub participant_id: Option<Uuid>,
    /// Violation timestamp
    pub timestamp: DateTime<Utc>,
    /// Severity level
    pub severity: ViolationSeverity,
    /// Description
    pub description: String,
    /// Mitigation action taken
    pub mitigation_action: Option<String>,
}

/// Privacy violation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyViolationType {
    /// Budget exceeded
    BudgetExceeded,
    /// Information leakage detected
    InformationLeakage,
    /// Model inversion attack
    ModelInversion,
    /// Membership inference attack
    MembershipInference,
    /// Data reconstruction attack
    DataReconstruction,
}

/// Violation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Federation statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationStats {
    /// Total number of participants
    pub total_participants: usize,
    /// Active participants
    pub active_participants: usize,
    /// Total rounds completed
    pub rounds_completed: usize,
    /// Average round duration
    pub avg_round_duration_seconds: f64,
    /// Total communication overhead
    pub total_communication_overhead_bytes: u64,
    /// Model convergence status
    pub convergence_status: ConvergenceStatus,
    /// Privacy metrics
    pub privacy_metrics: PrivacyMetrics,
    /// System uptime
    pub system_uptime_seconds: u64,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
}
