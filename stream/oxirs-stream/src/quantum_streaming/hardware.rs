//! Quantum hardware abstractions for stream processing

use serde::{Deserialize, Serialize};

/// Quantum hardware architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumHardware {
    /// Gate-based quantum computer
    GateBased {
        qubits: usize,
        connectivity: Vec<(usize, usize)>,
    },
    /// Quantum annealer
    Annealer {
        qubits: usize,
        coupling_strength: f64,
    },
    /// Photonic quantum computer
    Photonic {
        modes: usize,
        squeezing: f64,
    },
    /// Trapped ion quantum computer
    TrappedIon {
        ions: usize,
        trap_frequency: f64,
    },
    /// Superconducting quantum computer
    Superconducting {
        qubits: usize,
        coherence_time: f64,
    },
}

impl Default for QuantumHardware {
    fn default() -> Self {
        Self::GateBased {
            qubits: 32,
            connectivity: vec![(0, 1), (1, 2), (2, 3)],
        }
    }
}

/// Quantum hardware capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCapabilities {
    pub max_qubits: usize,
    pub gate_fidelity: f64,
    pub coherence_time_ms: f64,
    pub gate_time_ns: f64,
    pub readout_fidelity: f64,
}

impl Default for QuantumCapabilities {
    fn default() -> Self {
        Self {
            max_qubits: 32,
            gate_fidelity: 0.999,
            coherence_time_ms: 100.0,
            gate_time_ns: 10.0,
            readout_fidelity: 0.95,
        }
    }
}