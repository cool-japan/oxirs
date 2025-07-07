//! Quantum streaming types and enums

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Quantum error correction codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumErrorCorrection {
    /// Five-qubit code (perfect for single qubit errors)
    FiveQubitCode,
    /// Shor's 9-qubit code
    Shor9Qubit,
    /// Surface code
    SurfaceCode { distance: usize },
}

/// Quantum state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub amplitudes: Vec<f64>,
    pub phases: Vec<f64>,
    pub entanglement_graph: HashMap<String, Vec<String>>,
}

impl Default for QuantumState {
    fn default() -> Self {
        Self {
            amplitudes: vec![1.0, 0.0], // |0⟩ state
            phases: vec![0.0, 0.0],
            entanglement_graph: HashMap::new(),
        }
    }
}

/// Quantum operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumOperation {
    /// Pauli-X gate
    PauliX,
    /// Pauli-Y gate  
    PauliY,
    /// Pauli-Z gate
    PauliZ,
    /// Hadamard gate
    Hadamard,
    /// CNOT gate
    CNOT { control: usize, target: usize },
    /// Measurement
    Measure,
}

/// Quantum event in the streaming system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumEvent {
    pub id: String,
    pub timestamp: u64, // Unix timestamp in milliseconds
    pub quantum_state: QuantumState,
    pub operation: QuantumOperation,
    pub metadata: HashMap<String, String>,
}

/// Statistics for quantum processing
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct QuantumProcessingStats {
    pub operations_processed: u64,
    pub average_fidelity: f64,
    pub error_rate: f64,
    pub decoherence_time: Duration,
    pub gate_errors: u64,
    pub measurement_errors: u64,
}

/// Quantum stream processor
#[derive(Debug)]
pub struct QuantumStreamProcessor {
    pub quantum_registers: Vec<QuantumState>,
    pub error_correction: QuantumErrorCorrection,
    pub stats: QuantumProcessingStats,
    pub active_operations: Vec<QuantumOperation>,
}

impl Default for QuantumStreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumStreamProcessor {
    pub fn new() -> Self {
        Self {
            quantum_registers: vec![QuantumState::default()],
            error_correction: QuantumErrorCorrection::FiveQubitCode,
            stats: QuantumProcessingStats::default(),
            active_operations: Vec::new(),
        }
    }

    pub async fn process_event(&mut self, event: QuantumEvent) -> Result<()> {
        self.stats.operations_processed += 1;
        self.active_operations.push(event.operation);
        Ok(())
    }

    pub fn get_stats(&self) -> &QuantumProcessingStats {
        &self.stats
    }
}
