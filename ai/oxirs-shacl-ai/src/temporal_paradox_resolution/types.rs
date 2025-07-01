//! Core types and enums for temporal paradox resolution
//!
//! This module contains the fundamental type definitions used throughout
//! the temporal paradox resolution system.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

/// Types of timelines
#[derive(Debug, Clone)]
pub enum TimelineType {
    /// Original timeline
    Original,
    /// Branched timeline from paradox resolution
    Branched,
    /// Parallel timeline
    Parallel,
    /// Quantum superposition timeline
    QuantumSuperposition,
    /// Causal loop timeline
    CausalLoop,
    /// Paradox-containing timeline
    ParadoxContaining,
    /// Resolved timeline
    Resolved,
    /// Synthetic timeline for testing
    Synthetic,
}

/// Types of temporal dimensions
#[derive(Debug, Clone)]
pub enum TemporalDimensionType {
    /// Linear time dimension
    Linear,
    /// Circular time dimension
    Circular,
    /// Branching time dimension
    Branching,
    /// Quantum time dimension
    Quantum,
    /// Multi-dimensional time
    MultiDimensional,
    /// Transcendent time dimension
    Transcendent,
}

/// Types of causal relationships
#[derive(Debug, Clone)]
pub enum CausalRelationshipType {
    /// Cause precedes effect
    CausePrecedesEffect,
    /// Simultaneous causation
    Simultaneous,
    /// Reverse causation
    ReverseCausation,
    /// Circular causation
    CircularCausation,
    /// Quantum causation
    QuantumCausation,
    /// Acausal relationship
    Acausal,
}

/// Types of timeline connections
#[derive(Debug, Clone)]
pub enum TimelineConnectionType {
    /// Causal connection
    Causal,
    /// Quantum entanglement connection
    QuantumEntanglement,
    /// Branching connection
    Branching,
    /// Parallel connection
    Parallel,
    /// Paradox resolution connection
    ParadoxResolution,
    /// Temporal bridge connection
    TemporalBridge,
}

/// Types of temporal validation
#[derive(Debug, Clone)]
pub enum TemporalValidationType {
    /// Basic temporal consistency
    TemporalConsistency,
    /// Causal relationship validation
    CausalRelationship,
    /// Paradox detection
    ParadoxDetection,
    /// Timeline coherence validation
    TimelineCoherence,
    /// Quantum temporal validation
    QuantumTemporal,
    /// Multi-timeline validation
    MultiTimeline,
    /// Causality loop validation
    CausalityLoop,
}

/// Types of temporal events
#[derive(Debug, Clone)]
pub enum TemporalEventType {
    /// Validation execution
    ValidationExecution,
    /// Constraint evaluation
    ConstraintEvaluation,
    /// Data modification
    DataModification,
    /// Timeline branch
    TimelineBranch,
    /// Paradox creation
    ParadoxCreation,
    /// Paradox resolution
    ParadoxResolution,
    /// Causal intervention
    CausalIntervention,
}

/// Processing state of temporal processor
#[derive(Debug, Clone)]
pub enum TemporalProcessingState {
    /// Idle and ready
    Idle,
    /// Analyzing temporal relationships
    AnalyzingRelationships,
    /// Detecting paradoxes
    DetectingParadoxes,
    /// Resolving paradoxes
    ResolvingParadoxes,
    /// Validating timeline coherence
    ValidatingCoherence,
    /// Processing quantum temporal effects
    ProcessingQuantumEffects,
    /// Error in temporal processing
    Error(String),
}

/// Types of causal loops
#[derive(Debug, Clone)]
pub enum CausalLoopType {
    /// Simple closed loop
    Simple,
    /// Complex nested loops
    Complex,
    /// Quantum superposition loop
    QuantumSuperposition,
    /// Paradoxical loop
    Paradoxical,
    /// Self-consistent loop
    SelfConsistent,
    /// Bootstrap paradox loop
    Bootstrap,
    /// Grandfather paradox loop
    Grandfather,
}

/// Types of loop detection algorithms
#[derive(Debug, Clone)]
pub enum LoopDetectionAlgorithmType {
    /// Graph cycle detection
    GraphCycle,
    /// Temporal pattern analysis
    TemporalPattern,
    /// Causal chain analysis
    CausalChain,
    /// Quantum state analysis
    QuantumState,
    /// Multi-dimensional analysis
    MultiDimensional,
    /// Machine learning detection
    MachineLearning,
}

/// Quantum detection methods
#[derive(Debug, Clone)]
pub enum QuantumDetectionMethod {
    /// Entanglement analysis
    EntanglementAnalysis,
    /// Superposition analysis
    SuperpositionAnalysis,
    /// Quantum interference detection
    QuantumInterference,
    /// Quantum tunneling detection
    QuantumTunneling,
    /// Quantum field fluctuation analysis
    QuantumFieldFluctuation,
}

/// Types of maintenance strategies
#[derive(Debug, Clone)]
pub enum MaintenanceStrategyType {
    /// Preventive maintenance
    Preventive,
    /// Corrective maintenance
    Corrective,
    /// Predictive maintenance
    Predictive,
    /// Quantum maintenance
    Quantum,
    /// Adaptive maintenance
    Adaptive,
}

/// Types of paradox resolution strategies
#[derive(Debug, Clone)]
pub enum ParadoxResolutionStrategyType {
    /// Timeline branching
    TimelineBranching,
    /// Causal loop integration
    CausalLoopIntegration,
    /// Quantum superposition resolution
    QuantumSuperposition,
    /// Paradox isolation
    ParadoxIsolation,
    /// Causal intervention
    CausalIntervention,
    /// Timeline merger
    TimelineMerger,
    /// Quantum decoherence
    QuantumDecoherence,
}

/// Types of resolution steps
#[derive(Debug, Clone)]
pub enum ResolutionStepType {
    /// Analysis step
    Analysis,
    /// Isolation step
    Isolation,
    /// Modification step
    Modification,
    /// Integration step
    Integration,
    /// Verification step
    Verification,
    /// Finalization step
    Finalization,
}

/// Trend direction
#[derive(Debug, Clone)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Declining trend
    Declining,
    /// Stable trend
    Stable,
    /// Fluctuating trend
    Fluctuating,
    /// Unknown trend
    Unknown,
}

/// Types of temporal consistency rules
#[derive(Debug, Clone)]
pub enum TemporalConsistencyRuleType {
    /// Basic causality
    BasicCausality,
    /// Temporal ordering
    TemporalOrdering,
    /// Paradox prevention
    ParadoxPrevention,
    /// Quantum consistency
    QuantumConsistency,
    /// Multi-timeline consistency
    MultiTimelineConsistency,
}

/// Temporal coordinates for timeline positioning
#[derive(Debug, Clone)]
pub struct TemporalCoordinates {
    /// Temporal position
    pub position: f64,
    /// Temporal velocity
    pub velocity: f64,
    /// Temporal acceleration
    pub acceleration: f64,
    /// Temporal dimension
    pub dimension: TemporalDimension,
    /// Quantum temporal state
    pub quantum_state: QuantumTemporalState,
}

/// Temporal dimension representation
#[derive(Debug, Clone)]
pub struct TemporalDimension {
    /// Dimension identifier
    pub id: String,
    /// Dimension type
    pub dimension_type: TemporalDimensionType,
    /// Dimensional properties
    pub properties: HashMap<String, f64>,
    /// Quantum enhancement
    pub quantum_enhancement: f64,
}

/// Quantum temporal state
#[derive(Debug, Clone)]
pub struct QuantumTemporalState {
    /// Quantum temporal amplitudes
    pub amplitudes: Vec<f64>,
    /// Temporal phase information
    pub phases: Vec<f64>,
    /// Temporal coherence time
    pub coherence_time: f64,
    /// Quantum temporal entanglement
    pub entanglement: Vec<String>,
}

/// Temporal event in causality tracking
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    /// Event identifier
    pub id: String,
    /// Event type
    pub event_type: TemporalEventType,
    /// Temporal coordinates
    pub coordinates: TemporalCoordinates,
    /// Event properties
    pub properties: HashMap<String, f64>,
    /// Validation context
    pub validation_context: String,
}

/// Results and metadata types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParadoxDetectionResult {
    /// Paradox identifier
    pub paradox_id: String,
    /// Paradox type
    pub paradox_type: String,
    /// Severity level
    pub severity: f64,
    /// Detection confidence
    pub confidence: f64,
    /// Affected timelines
    pub affected_timelines: Vec<String>,
    /// Recommended resolution strategy
    pub recommended_strategy: Option<String>,
}

/// Result of paradox resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParadoxResolutionResult {
    /// Resolution identifier
    pub resolution_id: String,
    /// Original paradox
    pub paradox_id: String,
    /// Resolution strategy used
    pub strategy_used: String,
    /// Resolution success
    pub success: bool,
    /// Resolution quality
    pub quality: f64,
    /// Side effects
    pub side_effects: Vec<String>,
    /// Resource usage
    pub resource_usage: HashMap<String, f64>,
}

/// Timeline analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineAnalysisResult {
    /// Timeline identifier
    pub timeline_id: String,
    /// Analysis type
    pub analysis_type: String,
    /// Consistency score
    pub consistency_score: f64,
    /// Stability metrics
    pub stability_metrics: HashMap<String, f64>,
    /// Detected anomalies
    pub anomalies: Vec<String>,
}

/// Causality analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalityAnalysisResult {
    /// Analysis identifier
    pub analysis_id: String,
    /// Causal relationships found
    pub relationships: Vec<String>,
    /// Causal consistency score
    pub consistency_score: f64,
    /// Loop detection results
    pub loops_detected: Vec<String>,
    /// Risk assessment
    pub risk_assessment: HashMap<String, f64>,
}

/// Timeline context for validation
#[derive(Debug, Clone)]
pub struct TimelineContext {
    /// Current timeline identifier
    pub current_timeline: String,
    /// Available timelines
    pub available_timelines: Vec<String>,
    /// Temporal constraints
    pub temporal_constraints: Vec<TemporalConstraint>,
    /// Context metadata
    pub metadata: HashMap<String, String>,
}

/// Temporal constraint
#[derive(Debug, Clone)]
pub struct TemporalConstraint {
    /// Constraint identifier
    pub id: String,
    /// Constraint type
    pub constraint_type: String,
    /// Constraint parameters
    pub parameters: HashMap<String, f64>,
    /// Enforcement level
    pub enforcement_level: f64,
}

/// Individual causal relationship
#[derive(Debug, Clone)]
pub struct CausalRelationship {
    /// Relationship identifier
    pub id: String,
    /// Cause event
    pub cause: TemporalEvent,
    /// Effect event
    pub effect: TemporalEvent,
    /// Relationship type
    pub relationship_type: CausalRelationshipType,
    /// Relationship strength
    pub strength: f64,
    /// Temporal delay
    pub temporal_delay: f64,
    /// Certainty level
    pub certainty: f64,
}