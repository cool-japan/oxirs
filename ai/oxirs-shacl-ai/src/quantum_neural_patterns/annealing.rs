//! Quantum annealing for pattern optimization

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::core::QuantumPattern;
use crate::Result;

/// Quantum Annealing for Pattern Optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAnnealer {
    /// Annealing schedule
    schedule: Vec<f64>,
    /// Temperature parameter
    temperature: f64,
    /// Optimization objective
    objective_function: String,
    /// Current iteration
    iteration: usize,
}

impl QuantumAnnealer {
    /// Create a new quantum annealer
    pub fn new(max_iterations: usize) -> Self {
        let schedule: Vec<f64> = (0..max_iterations)
            .map(|i| 1.0 - (i as f64) / (max_iterations as f64))
            .collect();

        Self {
            schedule,
            temperature: 1.0,
            objective_function: "pattern_quality".to_string(),
            iteration: 0,
        }
    }

    /// Optimize quantum patterns using quantum annealing
    pub fn anneal_patterns(&mut self, patterns: &mut [QuantumPattern]) -> Result<f64> {
        let mut best_energy = f64::INFINITY;

        for &annealing_parameter in &self.schedule.clone() {
            self.temperature = annealing_parameter;

            // Apply annealing step to each pattern
            for pattern in patterns.iter_mut() {
                self.annealing_step(pattern)?;
            }

            // Calculate system energy
            let current_energy = self.calculate_system_energy(patterns)?;
            if current_energy < best_energy {
                best_energy = current_energy;
            }

            self.iteration += 1;
        }

        Ok(best_energy)
    }

    /// Apply single annealing step
    fn annealing_step(&self, pattern: &mut QuantumPattern) -> Result<()> {
        // Apply transverse field to maintain quantum superposition
        let transverse_field_strength = self.temperature;

        for qubit_idx in 0..pattern.quantum_state.amplitudes.len().trailing_zeros() as usize {
            let rotation_angle = transverse_field_strength * PI / 4.0;
            pattern
                .quantum_state
                .apply_rotation(rotation_angle, qubit_idx)?;
        }

        // Apply longitudinal field based on objective function
        let longitudinal_strength = 1.0 - self.temperature;
        self.apply_objective_field(pattern, longitudinal_strength)?;

        Ok(())
    }

    /// Apply objective-specific field
    fn apply_objective_field(&self, pattern: &mut QuantumPattern, strength: f64) -> Result<()> {
        match self.objective_function.as_str() {
            "pattern_quality" => {
                // Bias towards higher fidelity states
                let quality_bias = pattern.fidelity * strength;
                for amplitude in pattern.quantum_state.amplitudes.iter_mut() {
                    *amplitude *= Complex64::new(1.0 + quality_bias * 0.1, 0.0);
                }
            }
            "entanglement_maximization" => {
                // Bias towards maximally entangled states
                let entanglement_score = pattern.entanglement_scores.values().sum::<f64>()
                    / pattern.entanglement_scores.len().max(1) as f64;
                let entanglement_bias = entanglement_score * strength;
                for amplitude in pattern.quantum_state.amplitudes.iter_mut() {
                    *amplitude *= Complex64::new(1.0 + entanglement_bias * 0.1, 0.0);
                }
            }
            _ => {} // No specific bias
        }
        Ok(())
    }

    /// Calculate total system energy
    fn calculate_system_energy(&self, patterns: &[QuantumPattern]) -> Result<f64> {
        let mut total_energy = 0.0;

        for pattern in patterns {
            // Hamiltonian based on pattern properties
            let kinetic_energy = pattern.quantum_state.coherence();
            let potential_energy = 1.0 - pattern.fidelity;
            let interaction_energy = pattern.entanglement_scores.values().sum::<f64>();

            total_energy += kinetic_energy + potential_energy - interaction_energy;
        }

        Ok(total_energy)
    }

    /// Set optimization objective
    pub fn set_objective(&mut self, objective: String) {
        self.objective_function = objective;
    }

    /// Get current iteration
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
}
