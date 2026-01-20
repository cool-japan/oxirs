//! Quantum stream processor type definitions

use serde::{Deserialize, Serialize};
use nalgebra::{Complex, DMatrix, DVector};
use num_complex::Complex64;
use std::collections::HashMap;

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
    Repetition { distance: u32 },
    Shor { qubits: u32 },
    Steane,
    Surface { distance: u32 },
    Color { distance: u32 },
    Topological,
    CSSCodes,
    LDPCCodes,
    TwistedSurface,
    Floquet,
}

/// Measurement basis types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementBasis {
    Computational,
    Hadamard,
    Diagonal,
    Circular,
    Bell,
    Custom(String),
}

/// Comparison operators for quantum conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
}

/// Classical optimization algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassicalOptimizer {
    GradientDescent,
    Adam,
    BFGS,
    NelderMead,
    SimulatedAnnealing,
    GeneticAlgorithm,
    ParticleSwarm,
    QuantumApproximateOptimization,
}

/// Classical machine learning types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassicalMLType {
    LinearRegression,
    LogisticRegression,
    SVM,
    RandomForest,
    NeuralNetwork,
    DeepLearning,
    Clustering,
    PCA,
}

/// Preprocessing steps for quantum data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreprocessingStep {
    Normalization,
    QuantumEncoding,
    FeatureMapping,
    DimensionalityReduction,
    NoiseReduction,
}

/// Postprocessing steps for quantum results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PostprocessingStep {
    ErrorMitigation,
    StateTomography,
    ProcessTomography,
    BenchmarkExtraction,
}

/// Output format for quantum results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Probabilities,
    Amplitudes,
    Measurements,
    Expectation,
    Tomography,
}

/// Quantum circuit representation
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub qubits: u32,
    pub classical_bits: u32,
    pub gates: Vec<(QuantumGate, Vec<u32>)>,
    pub measurements: Vec<(u32, u32)>, // (qubit, classical_bit)
    pub depth: u32,
    pub metadata: HashMap<String, String>,
}

/// Quantum state representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitudes: DVector<Complex64>,
    pub num_qubits: u32,
    pub is_normalized: bool,
    pub entanglement_entropy: f64,
}

/// Quantum measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementResult {
    pub measurements: Vec<u8>,
    pub probabilities: Vec<f64>,
    pub execution_time_ms: f64,
    pub shots: u32,
    pub error_rate: f64,
}

/// Quantum process metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    pub fidelity: f64,
    pub gate_count: u32,
    pub circuit_depth: u32,
    pub execution_time_ms: f64,
    pub error_rate: f64,
    pub quantum_volume: u64,
    pub entanglement_measure: f64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            available_qubits: 16,
            coherence_time_microseconds: 100.0,
            gate_fidelity: 0.999,
            measurement_fidelity: 0.995,
            topology: QuantumTopology::Grid2D,
            supported_gates: vec![
                QuantumGate::PauliX,
                QuantumGate::PauliY,
                QuantumGate::PauliZ,
                QuantumGate::Hadamard,
                QuantumGate::CNOT,
                QuantumGate::RZ(0.0),
            ],
            error_correction_code: ErrorCorrectionCode::Surface { distance: 3 },
            quantum_volume: 32,
            max_circuit_depth: 100,
            classical_control_overhead_ns: 1000.0,
        }
    }
}

impl QuantumCircuit {
    pub fn new(qubits: u32, classical_bits: u32) -> Self {
        Self {
            qubits,
            classical_bits,
            gates: Vec::new(),
            measurements: Vec::new(),
            depth: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn add_gate(&mut self, gate: QuantumGate, qubits: Vec<u32>) {
        self.gates.push((gate, qubits));
        self.depth += 1;
    }

    pub fn add_measurement(&mut self, qubit: u32, classical_bit: u32) {
        self.measurements.push((qubit, classical_bit));
    }
}

impl QuantumState {
    pub fn new(num_qubits: u32) -> Self {
        let size = 2_usize.pow(num_qubits);
        let mut amplitudes = DVector::zeros(size);
        amplitudes[0] = Complex64::new(1.0, 0.0); // |00...0⟩ state
        
        Self {
            amplitudes,
            num_qubits,
            is_normalized: true,
            entanglement_entropy: 0.0,
        }
    }

    pub fn normalize(&mut self) {
        let norm: f64 = self.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum::<f64>()
            .sqrt();
        
        if norm > 0.0 {
            self.amplitudes /= norm;
            self.is_normalized = true;
        }
    }

    pub fn measure_probability(&self, outcome: usize) -> f64 {
        if outcome < self.amplitudes.len() {
            self.amplitudes[outcome].norm_sqr()
        } else {
            0.0
        }
    }
}