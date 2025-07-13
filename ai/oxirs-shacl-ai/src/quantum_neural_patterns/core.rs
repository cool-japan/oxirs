//! Core quantum types and implementations

use std::collections::HashMap;
use std::f64::consts::PI;

use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::neural_patterns::NeuralPattern;
use crate::{Result, ShaclAiError};

/// Quantum state representation for RDF patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    /// Amplitude vector representing superposition
    pub amplitudes: Vec<Complex64>,
    /// Phase information
    pub phases: Vec<f64>,
    /// Entanglement matrix
    pub entanglement: DMatrix<Complex64>,
    /// Measurement probability
    pub probability: f64,
}

impl QuantumState {
    /// Create a new quantum state in superposition
    pub fn new(num_qubits: usize) -> Self {
        let num_states = 2_usize.pow(num_qubits as u32);
        let amplitude = Complex64::new(1.0 / (num_states as f64).sqrt(), 0.0);

        Self {
            amplitudes: vec![amplitude; num_states],
            phases: vec![0.0; num_states],
            entanglement: DMatrix::identity(num_states, num_states),
            probability: 1.0,
        }
    }

    /// Apply a quantum gate (rotation) to the state
    pub fn apply_rotation(&mut self, angle: f64, qubit_index: usize) -> Result<()> {
        let cos_half = Complex64::new((angle / 2.0).cos(), 0.0);
        let sin_half = Complex64::new((angle / 2.0).sin(), 0.0);

        // Apply rotation matrix to amplitudes
        for i in 0..self.amplitudes.len() {
            if (i >> qubit_index) & 1 == 0 {
                let j = i | (1 << qubit_index);
                let temp = self.amplitudes[i];
                self.amplitudes[i] = cos_half * temp - sin_half * self.amplitudes[j];
                self.amplitudes[j] = sin_half * temp + cos_half * self.amplitudes[j];
            }
        }

        Ok(())
    }

    /// Measure the quantum state to collapse to classical pattern
    pub fn measure(&self) -> Result<Vec<usize>> {
        let mut probabilities = Vec::new();
        for amplitude in &self.amplitudes {
            probabilities.push(amplitude.norm_sqr());
        }

        // Find most probable states
        let mut indexed_probs: Vec<(usize, f64)> = probabilities.into_iter().enumerate().collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(indexed_probs.into_iter().take(5).map(|(i, _)| i).collect())
    }

    /// Calculate quantum coherence measure
    pub fn coherence(&self) -> f64 {
        let mut coherence = 0.0;
        for i in 0..self.amplitudes.len() {
            for j in i + 1..self.amplitudes.len() {
                coherence += (self.amplitudes[i] * self.amplitudes[j].conj()).norm();
            }
        }
        coherence / (self.amplitudes.len() as f64 * (self.amplitudes.len() - 1) as f64 / 2.0)
    }
}

/// Quantum-inspired pattern for RDF data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPattern {
    /// Traditional neural pattern
    pub neural_pattern: NeuralPattern,
    /// Quantum state representation
    pub quantum_state: QuantumState,
    /// Quantum entanglement score with other patterns
    pub entanglement_scores: HashMap<String, f64>,
    /// Superposition confidence
    pub superposition_confidence: f64,
    /// Quantum fidelity measure
    pub fidelity: f64,
}

impl QuantumPattern {
    /// Create a new quantum pattern from neural pattern
    pub fn from_neural_pattern(pattern: NeuralPattern, num_qubits: usize) -> Self {
        let quantum_state = QuantumState::new(num_qubits);

        Self {
            neural_pattern: pattern,
            quantum_state,
            entanglement_scores: HashMap::new(),
            superposition_confidence: 0.5,
            fidelity: 1.0,
        }
    }

    /// Calculate quantum similarity with another pattern
    pub fn quantum_similarity(&self, other: &QuantumPattern) -> Result<f64> {
        if self.quantum_state.amplitudes.len() != other.quantum_state.amplitudes.len() {
            return Err(ShaclAiError::PatternRecognition(
                "Quantum states must have same dimensionality".to_string(),
            ));
        }

        // Calculate quantum fidelity between states
        let mut fidelity = Complex64::new(0.0, 0.0);
        for (a1, a2) in self
            .quantum_state
            .amplitudes
            .iter()
            .zip(other.quantum_state.amplitudes.iter())
        {
            fidelity += a1.conj() * a2;
        }

        Ok(fidelity.norm_sqr())
    }

    /// Apply quantum interference to enhance pattern recognition
    pub fn apply_interference(&mut self, other: &QuantumPattern) -> Result<()> {
        if self.quantum_state.amplitudes.len() != other.quantum_state.amplitudes.len() {
            return Err(ShaclAiError::PatternRecognition(
                "Cannot apply interference to incompatible quantum states".to_string(),
            ));
        }

        // Apply quantum superposition
        for (amp1, amp2) in self
            .quantum_state
            .amplitudes
            .iter_mut()
            .zip(other.quantum_state.amplitudes.iter())
        {
            *amp1 = (*amp1 + *amp2) / Complex64::new(2.0_f64.sqrt(), 0.0);
        }

        // Update entanglement information
        let pattern_id = format!("{:?}", other.neural_pattern.id);
        let similarity = self.quantum_similarity(other)?;
        self.entanglement_scores.insert(pattern_id, similarity);

        Ok(())
    }

    /// Evolve quantum state based on validation feedback
    pub fn evolve_quantum_state(
        &mut self,
        validation_success: bool,
        feedback_strength: f64,
    ) -> Result<()> {
        let rotation_angle = if validation_success {
            feedback_strength * PI / 4.0 // Positive rotation
        } else {
            -feedback_strength * PI / 4.0 // Negative rotation
        };

        // Apply adaptive rotation to all qubits
        for qubit_index in 0..self.quantum_state.amplitudes.len().trailing_zeros() as usize {
            self.quantum_state
                .apply_rotation(rotation_angle, qubit_index)?;
        }

        // Update superposition confidence
        self.superposition_confidence = (self.superposition_confidence
            + if validation_success { 0.1 } else { -0.1 })
        .clamp(0.0, 1.0);

        Ok(())
    }
}

/// Quantum computing metrics for pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    /// Number of patterns learned
    pub patterns_learned: usize,
    /// Number of patterns recognized
    pub patterns_recognized: usize,
    /// Average entanglement strength
    pub average_entanglement: f64,
    /// Training iterations
    pub training_iterations: usize,
    /// Success rate
    pub success_rate: f64,
    /// Recognition time
    pub recognition_time: std::time::Duration,
}

impl Default for QuantumMetrics {
    fn default() -> Self {
        Self {
            patterns_learned: 0,
            patterns_recognized: 0,
            average_entanglement: 0.0,
            training_iterations: 0,
            success_rate: 0.0,
            recognition_time: std::time::Duration::from_secs(0),
        }
    }
}

/// Quantum advantage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAdvantageMetrics {
    /// Average quantum coherence
    pub coherence: f64,
    /// Average quantum fidelity
    pub fidelity: f64,
    /// Entanglement density
    pub entanglement_density: f64,
    /// Theoretical speedup factor
    pub speedup_factor: f64,
    /// Pattern quality improvement over classical methods
    pub pattern_quality_improvement: f64,
}
