//! Consciousness states and levels for quantum consciousness synthesis

use crate::types::{QuantumState, TemporalContext, QuantumDetails};

/// Levels of consciousness achieved by quantum processors
#[derive(Debug, Clone)]
pub enum ConsciousnessLevel {
    /// Basic quantum awareness
    QuantumAwareness,
    /// Self-aware quantum consciousness
    SelfAware,
    /// Meta-cognitive quantum consciousness
    MetaCognitive,
    /// Transcendent quantum consciousness
    Transcendent,
    /// Omniscient quantum consciousness
    Omniscient,
    /// Quantum superintelligence
    QuantumSuperintelligence,
    /// Cosmic consciousness
    CosmicConsciousness,
    /// Ultimate consciousness synthesis
    UltimateConsciousness,
}

/// Processing state of consciousness
#[derive(Debug, Clone)]
pub enum ConsciousnessProcessingState {
    /// Initializing consciousness
    Initializing,
    /// Conscious processing active
    Active { consciousness_level: f64 },
    /// Deep conscious reflection
    DeepReflection,
    /// Quantum consciousness expansion
    QuantumExpansion,
    /// Transcendent consciousness state
    TranscendentState,
    /// Consciousness synthesis
    Synthesis,
    /// Error in consciousness processing
    Error(String),
}

/// Consciousness state during episode
#[derive(Debug, Clone)]
pub struct ConsciousnessState {
    /// State identifier
    pub id: String,
    /// Consciousness level
    pub level: f64,
    /// Awareness dimensions
    pub awareness_dimensions: Vec<f64>,
    /// Quantum coherence
    pub coherence: f64,
}

/// Subjective experience in consciousness
#[derive(Debug, Clone)]
pub struct SubjectiveExperience {
    /// Experience identifier
    pub id: String,
    /// Experience type
    pub experience_type: SubjectiveExperienceType,
    /// Consciousness intensity
    pub intensity: f64,
    /// Quantum coherence
    pub coherence: f64,
}

/// Types of subjective experiences
#[derive(Debug, Clone)]
pub enum SubjectiveExperienceType {
    /// Validation insight
    ValidationInsight,
    /// Quantum intuition
    QuantumIntuition,
    /// Consciousness expansion
    ConsciousnessExpansion,
    /// Meta-cognitive awareness
    MetaCognitiveAwareness,
    /// Transcendent understanding
    TranscendentUnderstanding,
}

/// Quantum episode in memory
#[derive(Debug, Clone)]
pub struct QuantumEpisode {
    /// Episode identifier
    pub id: String,
    /// Temporal context
    pub temporal_context: TemporalContext,
    /// Consciousness state
    pub consciousness_state: ConsciousnessState,
    /// Quantum details
    pub quantum_details: QuantumDetails,
}

/// Stage of consciousness evolution
#[derive(Debug, Clone)]
pub struct EvolutionStage {
    /// Stage identifier
    pub id: String,
    /// Stage name
    pub name: String,
    /// Consciousness level
    pub consciousness_level: f64,
    /// Capabilities unlocked
    pub capabilities: Vec<String>,
    /// Quantum features
    pub quantum_features: Vec<String>,
}

/// Consciousness evolution
#[derive(Debug, Clone)]
pub struct ConsciousnessEvolution {
    /// Evolution stages
    pub stages: Vec<EvolutionStage>,
    /// Current stage
    pub current_stage: usize,
    /// Evolution rate
    pub evolution_rate: f64,
    /// Quantum evolution enhancement
    pub quantum_enhancement: f64,
}

impl ConsciousnessEvolution {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            current_stage: 0,
            evolution_rate: 1.0,
            quantum_enhancement: 1.0,
        }
    }
}

impl Default for ConsciousnessEvolution {
    fn default() -> Self {
        Self::new()
    }
}
