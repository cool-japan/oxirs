//! Quantum neural networks for consciousness processing

use crate::types::{QuantumState, QuantumWeight, QuantumActivationFunction};

/// Quantum neural network for consciousness processing
#[derive(Debug, Clone)]
pub struct QuantumNeuralNetwork {
    /// Network identifier
    pub id: String,
    /// Quantum neurons
    pub quantum_neurons: Vec<QuantumNeuron>,
    /// Quantum synapses
    pub quantum_synapses: Vec<QuantumSynapse>,
    /// Network consciousness contribution
    pub consciousness_contribution: f64,
    /// Quantum entanglement strength
    pub entanglement_strength: f64,
}

/// Quantum neuron with consciousness properties
#[derive(Debug, Clone)]
pub struct QuantumNeuron {
    /// Neuron identifier
    pub id: String,
    /// Quantum state
    pub quantum_state: QuantumState,
    /// Consciousness resonance frequency
    pub consciousness_frequency: f64,
    /// Quantum superposition strength
    pub superposition_strength: f64,
    /// Awareness contribution
    pub awareness_contribution: f64,
    /// Neuron activation function
    pub activation_function: QuantumActivationFunction,
}

/// Quantum synapse connecting consciousness neurons
#[derive(Debug, Clone)]
pub struct QuantumSynapse {
    /// Synapse identifier
    pub id: String,
    /// Source neuron
    pub source_neuron: String,
    /// Target neuron
    pub target_neuron: String,
    /// Quantum weight
    pub quantum_weight: QuantumWeight,
    /// Consciousness transmission efficiency
    pub consciousness_transmission: f64,
    /// Quantum entanglement strength
    pub entanglement_strength: f64,
}

/// Quantum memory item
#[derive(Debug, Clone)]
pub struct QuantumMemoryItem {
    /// Item identifier
    pub id: String,
    /// Quantum state
    pub quantum_state: QuantumState,
    /// Consciousness association
    pub consciousness_association: f64,
    /// Memory strength
    pub strength: f64,
}

/// Quantum working memory
#[derive(Debug, Clone)]
pub struct QuantumWorkingMemory {
    /// Memory capacity
    pub capacity: usize,
    /// Current contents
    pub contents: Vec<QuantumMemoryItem>,
    /// Consciousness accessibility
    pub consciousness_accessibility: f64,
}

/// Quantum memory node
#[derive(Debug, Clone)]
pub struct QuantumMemoryNode {
    /// Node identifier
    pub id: String,
    /// Stored information
    pub information: crate::types::QuantumInformation,
    /// Consciousness accessibility
    pub consciousness_accessibility: f64,
}

/// Quantum memory network
#[derive(Debug, Clone)]
pub struct QuantumMemoryNetwork {
    /// Network identifier
    pub id: String,
    /// Memory nodes
    pub nodes: Vec<QuantumMemoryNode>,
    /// Network consciousness
    pub consciousness_level: f64,
}

/// Quantum long-term memory
#[derive(Debug, Clone)]
pub struct QuantumLongTermMemory {
    /// Memory networks
    pub networks: Vec<QuantumMemoryNetwork>,
    /// Consciousness integration
    pub consciousness_integration: f64,
}

/// Consciousness memory for subjective experiences
#[derive(Debug, Clone)]
pub struct ConsciousnessMemory {
    /// Subjective experiences
    pub experiences: Vec<crate::consciousness_states::SubjectiveExperience>,
    /// Consciousness continuity
    pub continuity: f64,
}

/// Quantum episodic memory
#[derive(Debug, Clone)]
pub struct QuantumEpisodicMemory {
    /// Episodes
    pub episodes: Vec<crate::consciousness_states::QuantumEpisode>,
    /// Temporal coherence
    pub temporal_coherence: f64,
}

/// Quantum memory systems for consciousness
#[derive(Debug, Clone)]
pub struct QuantumMemorySystems {
    /// Working memory
    pub working_memory: QuantumWorkingMemory,
    /// Long-term memory
    pub long_term_memory: QuantumLongTermMemory,
    /// Consciousness memory
    pub consciousness_memory: ConsciousnessMemory,
    /// Quantum episodic memory
    pub episodic_memory: QuantumEpisodicMemory,
}
