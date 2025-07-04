//! Type definitions and context structures for swarm neuromorphic networks

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Swarm validation context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmValidationContext {
    /// Swarm coordination mode
    pub coordination_mode: SwarmCoordinationMode,
    /// Task distribution strategy
    pub distribution_strategy: TaskDistributionStrategy,
    /// Communication protocol
    pub communication_protocol: CommunicationProtocol,
    /// Optimization targets
    pub optimization_targets: OptimizationTargets,
}

/// Swarm coordination modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmCoordinationMode {
    Centralized,
    Decentralized,
    Hierarchical,
    SelfOrganizing,
}

/// Task distribution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskDistributionStrategy {
    LoadBalancing,
    SpecializationBased,
    GeographicProximity,
    AdaptiveDynamic,
}

/// Communication protocols for swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationProtocol {
    DirectMessaging,
    PheromoneTrails,
    SignalPropagation,
    HybridCommunication,
}

/// Optimization targets for swarm behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTargets {
    pub minimize_energy_consumption: bool,
    pub maximize_processing_speed: bool,
    pub optimize_communication_efficiency: bool,
    pub enhance_fault_tolerance: bool,
}

/// Unique identifier for swarm nodes
pub type SwarmNodeId = Uuid;

/// Position of swarm node in 3D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPosition {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub velocity: Vector3<f64>,
}

/// Emergent behavior insight from swarm analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentBehaviorInsight {
    /// Insight identifier
    pub insight_id: String,
    /// Type of emergent behavior
    pub behavior_type: EmergentBehaviorType,
    /// Confidence in the insight
    pub confidence: f64,
    /// Description of the behavior
    pub description: String,
    /// Contributing nodes
    pub contributing_nodes: Vec<SwarmNodeId>,
    /// Behavior metrics
    pub metrics: EmergentBehaviorMetrics,
}

/// Types of emergent behaviors in swarms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergentBehaviorType {
    Clustering,
    SynchronizedMovement,
    HierarchicalFormation,
    AdaptiveSpecialization,
    CollectiveDecisionMaking,
    SelfHealing,
    PatternFormation,
}

/// Metrics for emergent behavior analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentBehaviorMetrics {
    pub emergence_strength: f64,
    pub persistence_duration: f64,
    pub spatial_coherence: f64,
    pub temporal_stability: f64,
    pub complexity_measure: f64,
}

/// Types of swarm intelligence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmIntelligenceType {
    /// Individual node intelligence
    Individual,
    /// Collective group intelligence
    Collective,
    /// Hierarchical multi-level intelligence
    Hierarchical,
    /// Distributed consensus intelligence
    Consensus,
    /// Adaptive learning intelligence
    Adaptive,
    /// Emergent pattern intelligence
    Emergent,
}

/// Capabilities of individual swarm nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmNodeCapabilities {
    /// Node processing power
    pub processing_power: f64,
    /// Memory capacity in MB
    pub memory_capacity: f64,
    /// Communication range
    pub communication_range: f64,
    /// Supported algorithms
    pub supported_algorithms: Vec<String>,
    /// Energy efficiency rating
    pub energy_efficiency: f64,
    /// Fault tolerance level
    pub fault_tolerance: f64,
    /// Specialization areas
    pub specializations: Vec<NodeSpecialization>,
}

/// Node specialization areas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeSpecialization {
    DataProcessing,
    Communication,
    Coordination,
    Analysis,
    Learning,
    Storage,
    Monitoring,
    Security,
}