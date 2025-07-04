//! Quantum circuit representation and operations

use std::collections::HashMap;
use nalgebra::DMatrix;
use num_complex::Complex64;

use super::quantum_config::QuantumGate;

/// Quantum circuit representation
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub circuit_id: String,
    pub qubits: u32,
    pub classical_bits: u32,
    pub gates: Vec<QuantumGateOperation>,
    pub measurements: Vec<MeasurementOperation>,
    pub circuit_depth: u32,
    pub estimated_execution_time_us: f64,
    pub success_probability: f64,
    pub quantum_complexity: QuantumComplexity,
}

impl QuantumCircuit {
    pub fn new(circuit_id: String, qubits: u32, classical_bits: u32) -> Self {
        Self {
            circuit_id,
            qubits,
            classical_bits,
            gates: Vec::new(),
            measurements: Vec::new(),
            circuit_depth: 0,
            estimated_execution_time_us: 0.0,
            success_probability: 1.0,
            quantum_complexity: QuantumComplexity::default(),
        }
    }

    pub fn add_gate(&mut self, gate: QuantumGateOperation) {
        self.gates.push(gate);
        self.circuit_depth = self.calculate_depth();
    }

    pub fn add_measurement(&mut self, measurement: MeasurementOperation) {
        self.measurements.push(measurement);
    }

    fn calculate_depth(&self) -> u32 {
        // Simplified depth calculation
        self.gates.len() as u32
    }
}

/// Quantum gate operation
#[derive(Debug, Clone)]
pub struct QuantumGateOperation {
    pub gate: QuantumGate,
    pub target_qubits: Vec<u32>,
    pub control_qubits: Vec<u32>,
    pub parameters: Vec<f64>,
    pub condition: Option<ClassicalCondition>,
}

impl QuantumGateOperation {
    pub fn new(gate: QuantumGate, target_qubits: Vec<u32>) -> Self {
        Self {
            gate,
            target_qubits,
            control_qubits: Vec::new(),
            parameters: Vec::new(),
            condition: None,
        }
    }

    pub fn with_controls(mut self, control_qubits: Vec<u32>) -> Self {
        self.control_qubits = control_qubits;
        self
    }

    pub fn with_parameters(mut self, parameters: Vec<f64>) -> Self {
        self.parameters = parameters;
        self
    }
}

/// Measurement operation
#[derive(Debug, Clone)]
pub struct MeasurementOperation {
    pub qubit: u32,
    pub classical_bit: u32,
    pub basis: MeasurementBasis,
}

impl MeasurementOperation {
    pub fn new(qubit: u32, classical_bit: u32) -> Self {
        Self {
            qubit,
            classical_bit,
            basis: MeasurementBasis::Computational,
        }
    }

    pub fn with_basis(mut self, basis: MeasurementBasis) -> Self {
        self.basis = basis;
        self
    }
}

/// Measurement basis
#[derive(Debug, Clone)]
pub enum MeasurementBasis {
    Computational,  // Z basis
    Hadamard,      // X basis
    Circular,      // Y basis
    Custom(DMatrix<Complex64>),
}

/// Classical condition for conditional operations
#[derive(Debug, Clone)]
pub struct ClassicalCondition {
    pub register: String,
    pub value: u64,
    pub comparison: ComparisonOperator,
}

/// Comparison operators for classical conditions
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
}

/// Quantum complexity metrics
#[derive(Debug, Clone)]
pub struct QuantumComplexity {
    pub quantum_gate_count: HashMap<String, u32>, // Using String instead of QuantumGate for simplicity
    pub entanglement_entropy: f64,
    pub circuit_expressivity: f64,
    pub barren_plateau_susceptibility: f64,
}

impl Default for QuantumComplexity {
    fn default() -> Self {
        Self {
            quantum_gate_count: HashMap::new(),
            entanglement_entropy: 0.0,
            circuit_expressivity: 0.0,
            barren_plateau_susceptibility: 0.0,
        }
    }
}