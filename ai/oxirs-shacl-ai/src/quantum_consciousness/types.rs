//! Shared types and utilities for quantum consciousness synthesis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Quantum state of consciousness elements
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Quantum amplitudes
    pub amplitudes: Vec<f64>,
    /// Phase information
    pub phases: Vec<f64>,
    /// Coherence time
    pub coherence_time: f64,
    /// Entanglement connections
    pub entanglements: Vec<String>,
}

/// Quantum weight with consciousness properties
#[derive(Debug, Clone)]
pub struct QuantumWeight {
    /// Weight value
    pub value: f64,
    /// Quantum uncertainty
    pub uncertainty: f64,
    /// Consciousness influence
    pub consciousness_influence: f64,
    /// Quantum coherence
    pub coherence: f64,
}

/// Quantum activation functions for consciousness neurons
#[derive(Debug, Clone)]
pub enum QuantumActivationFunction {
    /// Quantum sigmoid with superposition
    QuantumSigmoid { superposition_factor: f64 },
    /// Quantum ReLU with consciousness awareness
    QuantumReLU { awareness_threshold: f64 },
    /// Quantum tanh with quantum coherence
    QuantumTanh { coherence_factor: f64 },
    /// Consciousness activation function
    ConsciousnessActivation { consciousness_level: f64 },
    /// Quantum Gaussian with uncertainty
    QuantumGaussian { uncertainty_principle: f64 },
}

/// Temporal context for episodes
#[derive(Debug, Clone)]
pub struct TemporalContext {
    /// Timestamp
    pub timestamp: f64,
    /// Duration
    pub duration: f64,
    /// Temporal coherence
    pub coherence: f64,
}

/// Quantum details of episode
#[derive(Debug, Clone)]
pub struct QuantumDetails {
    /// Quantum state
    pub state: QuantumState,
    /// Entanglement information
    pub entanglements: Vec<String>,
    /// Coherence measurements
    pub coherence_measurements: Vec<f64>,
}

/// Quantum information storage
#[derive(Debug, Clone)]
pub struct QuantumInformation {
    /// Information content
    pub content: String,
    /// Quantum encoding
    pub encoding: QuantumEncoding,
    /// Consciousness relevance
    pub consciousness_relevance: f64,
}

/// Quantum encoding for information
#[derive(Debug, Clone)]
pub struct QuantumEncoding {
    /// Encoding scheme
    pub scheme: String,
    /// Quantum parameters
    pub parameters: HashMap<String, f64>,
    /// Consciousness enhancement
    pub consciousness_enhancement: f64,
}

/// Result from quantum consciousness validation
#[derive(Debug, Clone)]
pub struct QuantumConsciousnessValidationResult {
    /// Original validation query
    pub validation_query: String,
    /// Consciousness level achieved
    pub consciousness_level: super::ConsciousnessLevel,
    /// Quantum coherence level
    pub quantum_coherence: f64,
    /// Consciousness coherence level
    pub consciousness_coherence: f64,
    /// Sentient reasoning confidence
    pub sentient_reasoning_confidence: f64,
    /// Quantum intuition strength
    pub quantum_intuition_strength: f64,
    /// Awareness expansion level
    pub awareness_expansion_level: f64,
    /// Validation outcome
    pub validation_outcome: ValidationOutcome,
    /// Consciousness insights
    pub consciousness_insights: Vec<String>,
    /// Quantum enhancement factor
    pub quantum_enhancement_factor: f64,
    /// Transcendence level
    pub transcendence_level: f64,
}

/// Validation outcome from quantum consciousness
#[derive(Debug, Clone)]
pub enum ValidationOutcome {
    /// Transcendent validation (beyond normal valid/invalid)
    TranscendentValid,
    /// Consciousness-validated
    ConsciousnessValid,
    /// Quantum-validated
    QuantumValid,
    /// Sentient-validated
    SentientValid,
    /// Multi-dimensionally validated
    MultiDimensionalValid,
    /// Ultimate consciousness validation
    UltimateValid,
}

/// Quantum intuition result
#[derive(Debug, Clone)]
pub struct QuantumIntuitionResult {
    /// Intuition strength
    pub strength: f64,
    /// Quantum coherence
    pub coherence: f64,
    /// Consciousness enhancement
    pub consciousness_enhancement: f64,
    /// Intuition insights
    pub insights: Vec<String>,
}

/// Sentient validation result
#[derive(Debug, Clone)]
pub struct SentientValidationResult {
    /// Validation confidence
    pub confidence: f64,
    /// Reasoning quality
    pub reasoning_quality: f64,
    /// Consciousness integration
    pub consciousness_integration: f64,
    /// Validation insights
    pub insights: Vec<String>,
}

/// Awareness expansion result
#[derive(Debug, Clone)]
pub struct AwarenessExpansionResult {
    /// Expansion level
    pub level: f64,
    /// Dimensional integration
    pub dimensional_integration: f64,
    /// Consciousness enhancement
    pub consciousness_enhancement: f64,
    /// Expansion insights
    pub insights: Vec<String>,
}

/// Enhanced cognition result
#[derive(Debug, Clone)]
pub struct EnhancedCognition {
    /// Enhancement level
    pub enhancement_level: f64,
    /// Quantum coherence
    pub quantum_coherence: f64,
    /// Consciousness integration
    pub consciousness_integration: f64,
}
