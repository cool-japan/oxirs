//! Quantum processor configuration and basic types

use serde::{Deserialize, Serialize};

/// Quantum processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    pub available_qubits: u32,
    pub coherence_time_microseconds: f64,
    pub gate_fidelity: f64,
    pub measurement_fidelity: f64,
    pub topology: QuantumTopology,
    pub supported_gates: Vec<QuantumGate>,
    pub error_correction_code: ErrorCorrectionCode,
    pub quantum_volume: u64,
    pub max_circuit_depth: u32,
    pub classical_control_overhead_ns: f64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            available_qubits: 20,
            coherence_time_microseconds: 100.0,
            gate_fidelity: 0.999,
            measurement_fidelity: 0.985,
            topology: QuantumTopology::Grid2D,
            supported_gates: vec![
                QuantumGate::PauliX,
                QuantumGate::PauliY,
                QuantumGate::PauliZ,
                QuantumGate::Hadamard,
                QuantumGate::CNOT,
                QuantumGate::CZ,
                QuantumGate::RZ(0.0),
            ],
            error_correction_code: ErrorCorrectionCode::Surface,
            quantum_volume: 64,
            max_circuit_depth: 200,
            classical_control_overhead_ns: 1000.0,
        }
    }
}

/// Quantum processor topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumTopology {
    Linear,
    Ring,
    Grid2D,
    CompleteGraph,
    IonTrap,
    Superconducting,
    PhotonicMesh,
    Custom(String),
}

/// Quantum gates supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumGate {
    // Single-qubit gates
    PauliX,
    PauliY,
    PauliZ,
    Hadamard,
    Phase,
    SPhase,
    TGate,
    RX(f64),
    RY(f64),
    RZ(f64),
    U3(f64, f64, f64),

    // Two-qubit gates
    CNOT,
    CZ,
    SWAP,
    CRX(f64),
    CRY(f64),
    CRZ(f64),

    // Multi-qubit gates
    Toffoli,
    Fredkin,
    CSwap,

    // Specialized gates
    QFT,
    InverseQFT,
    GroverDiffusion,
    Custom(String),
}

/// Error correction codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorCorrectionCode {
    None,
    Repetition,
    Shor,
    Steane,
    Surface,
    ColorCode,
    ToricCode,
    Concatenated,
}
