//! Advanced Quantum Circuit Implementations for Quantum-Inspired Embeddings
//!
//! This module provides comprehensive quantum circuit implementations including:
//! - Variational Quantum Circuits (VQC)
//! - Quantum Approximate Optimization Algorithm (QAOA)
//! - Variational Quantum Eigensolver (VQE)
//! - Quantum Neural Networks (QNN)
//! - Quantum Feature Maps and Kernels

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Complex number representation for quantum amplitudes
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self {
            real: self.real * scalar,
            imag: self.imag * scalar,
        }
    }
}

/// Quantum gate types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumGate {
    /// Identity gate
    I,
    /// Pauli-X gate (bit flip)
    X,
    /// Pauli-Y gate
    Y,
    /// Pauli-Z gate (phase flip)
    Z,
    /// Hadamard gate
    H,
    /// Phase gate
    S,
    /// T gate
    T,
    /// Rotation around X-axis
    RX(f64),
    /// Rotation around Y-axis
    RY(f64),
    /// Rotation around Z-axis
    RZ(f64),
    /// CNOT gate (controlled-X)
    CNOT(usize, usize), // (control, target)
    /// Controlled-Z gate
    CZ(usize, usize),
    /// Toffoli gate (CCX)
    Toffoli(usize, usize, usize), // (control1, control2, target)
    /// Arbitrary unitary gate
    Unitary(Array2<Complex>),
}

/// Quantum circuit representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Sequence of gates
    pub gates: Vec<QuantumGate>,
    /// Circuit parameters (for variational circuits)
    pub parameters: Vec<f64>,
    /// Parameter mapping to gates
    pub parameter_mapping: HashMap<usize, Vec<usize>>, // gate_index -> parameter_indices
}

impl QuantumCircuit {
    /// Create new quantum circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            parameters: Vec::new(),
            parameter_mapping: HashMap::new(),
        }
    }

    /// Add gate to circuit
    pub fn add_gate(&mut self, gate: QuantumGate) {
        self.gates.push(gate);
    }

    /// Add parameterized gate
    pub fn add_parameterized_gate(&mut self, gate: QuantumGate, param_indices: Vec<usize>) {
        let gate_index = self.gates.len();
        self.gates.push(gate);
        self.parameter_mapping.insert(gate_index, param_indices);
    }

    /// Update circuit parameters
    pub fn set_parameters(&mut self, parameters: Vec<f64>) {
        self.parameters = parameters;
    }

    /// Get gate matrix for single-qubit gates
    pub fn get_single_qubit_matrix(&self, gate: &QuantumGate) -> Array2<Complex> {
        match gate {
            QuantumGate::I => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(1.0, 0.0),
                ],
            )
            .unwrap(),
            QuantumGate::X => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(1.0, 0.0),
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            QuantumGate::Y => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, -1.0),
                    Complex::new(0.0, 1.0),
                    Complex::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            QuantumGate::Z => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(-1.0, 0.0),
                ],
            )
            .unwrap(),
            QuantumGate::H => {
                let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex::new(inv_sqrt2, 0.0),
                        Complex::new(inv_sqrt2, 0.0),
                        Complex::new(inv_sqrt2, 0.0),
                        Complex::new(-inv_sqrt2, 0.0),
                    ],
                )
                .unwrap()
            }
            QuantumGate::S => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 1.0),
                ],
            )
            .unwrap(),
            QuantumGate::T => {
                let phase = Complex::new((PI / 4.0).cos(), (PI / 4.0).sin());
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex::new(1.0, 0.0),
                        Complex::new(0.0, 0.0),
                        Complex::new(0.0, 0.0),
                        phase,
                    ],
                )
                .unwrap()
            }
            QuantumGate::RX(theta) => {
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex::new(cos_half, 0.0),
                        Complex::new(0.0, -sin_half),
                        Complex::new(0.0, -sin_half),
                        Complex::new(cos_half, 0.0),
                    ],
                )
                .unwrap()
            }
            QuantumGate::RY(theta) => {
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex::new(cos_half, 0.0),
                        Complex::new(-sin_half, 0.0),
                        Complex::new(sin_half, 0.0),
                        Complex::new(cos_half, 0.0),
                    ],
                )
                .unwrap()
            }
            QuantumGate::RZ(theta) => {
                let exp_neg = Complex::new((theta / 2.0).cos(), -(theta / 2.0).sin());
                let exp_pos = Complex::new((theta / 2.0).cos(), (theta / 2.0).sin());
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        exp_neg,
                        Complex::new(0.0, 0.0),
                        Complex::new(0.0, 0.0),
                        exp_pos,
                    ],
                )
                .unwrap()
            }
            QuantumGate::Unitary(matrix) => matrix.clone(),
            _ => panic!("Not a single-qubit gate"),
        }
    }
}

/// Quantum state vector simulator
#[derive(Debug, Clone)]
pub struct QuantumSimulator {
    /// Number of qubits
    pub num_qubits: usize,
    /// State vector
    pub state_vector: Vec<Complex>,
}

impl QuantumSimulator {
    /// Create new quantum simulator with |0...0⟩ state
    pub fn new(num_qubits: usize) -> Self {
        let state_size = 2_usize.pow(num_qubits as u32);
        let mut state_vector = vec![Complex::new(0.0, 0.0); state_size];
        state_vector[0] = Complex::new(1.0, 0.0); // |0...0⟩ state

        Self {
            num_qubits,
            state_vector,
        }
    }

    /// Apply single-qubit gate
    pub fn apply_single_qubit_gate(&mut self, gate_matrix: &Array2<Complex>, qubit: usize) {
        let state_size = self.state_vector.len();
        let mut new_state = vec![Complex::new(0.0, 0.0); state_size];

        for i in 0..state_size {
            let bit = (i >> qubit) & 1;
            let i_flip = i ^ (1 << qubit);

            new_state[i] = new_state[i] + gate_matrix[(0, bit)] * self.state_vector[i];
            new_state[i_flip] = new_state[i_flip] + gate_matrix[(1, bit)] * self.state_vector[i];
        }

        self.state_vector = new_state;
    }

    /// Apply CNOT gate
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        let state_size = self.state_vector.len();
        let mut new_state = self.state_vector.clone();

        for i in 0..state_size {
            let control_bit = (i >> control) & 1;
            if control_bit == 1 {
                let target_bit = (i >> target) & 1;
                let i_flip = i ^ (1 << target);
                new_state.swap(i, i_flip);
            }
        }

        self.state_vector = new_state;
    }

    /// Execute quantum circuit
    pub fn execute_circuit(&mut self, circuit: &QuantumCircuit) -> Result<()> {
        for (gate_idx, gate) in circuit.gates.iter().enumerate() {
            match gate {
                QuantumGate::I => {
                    // Identity gate - do nothing
                }
                QuantumGate::X | QuantumGate::Y | QuantumGate::Z | QuantumGate::H | QuantumGate::S | QuantumGate::T => {
                    let matrix = circuit.get_single_qubit_matrix(gate);
                    // Apply to first qubit for now (would need qubit specification in real implementation)
                    self.apply_single_qubit_gate(&matrix, 0);
                }
                QuantumGate::RX(theta) | QuantumGate::RY(theta) | QuantumGate::RZ(theta) => {
                    let mut theta_val = *theta;
                    
                    // Check if this gate uses parameters
                    if let Some(param_indices) = circuit.parameter_mapping.get(&gate_idx) {
                        if !param_indices.is_empty() && param_indices[0] < circuit.parameters.len() {
                            theta_val = circuit.parameters[param_indices[0]];
                        }
                    }

                    let matrix = match gate {
                        QuantumGate::RX(_) => circuit.get_single_qubit_matrix(&QuantumGate::RX(theta_val)),
                        QuantumGate::RY(_) => circuit.get_single_qubit_matrix(&QuantumGate::RY(theta_val)),
                        QuantumGate::RZ(_) => circuit.get_single_qubit_matrix(&QuantumGate::RZ(theta_val)),
                        _ => unreachable!(),
                    };
                    self.apply_single_qubit_gate(&matrix, 0);
                }
                QuantumGate::CNOT(control, target) => {
                    self.apply_cnot(*control, *target);
                }
                QuantumGate::CZ(control, target) => {
                    // Controlled-Z gate
                    let state_size = self.state_vector.len();
                    for i in 0..state_size {
                        let control_bit = (i >> control) & 1;
                        let target_bit = (i >> target) & 1;
                        if control_bit == 1 && target_bit == 1 {
                            self.state_vector[i] = self.state_vector[i] * Complex::new(-1.0, 0.0);
                        }
                    }
                }
                QuantumGate::Toffoli(control1, control2, target) => {
                    // Toffoli gate (CCX)
                    let state_size = self.state_vector.len();
                    let mut new_state = self.state_vector.clone();
                    
                    for i in 0..state_size {
                        let c1_bit = (i >> control1) & 1;
                        let c2_bit = (i >> control2) & 1;
                        if c1_bit == 1 && c2_bit == 1 {
                            let i_flip = i ^ (1 << target);
                            new_state.swap(i, i_flip);
                        }
                    }
                    self.state_vector = new_state;
                }
                QuantumGate::Unitary(matrix) => {
                    // Apply arbitrary unitary (simplified for single qubit)
                    self.apply_single_qubit_gate(matrix, 0);
                }
            }
        }
        Ok(())
    }

    /// Measure qubit in computational basis
    pub fn measure_qubit(&self, qubit: usize) -> (f64, f64) {
        let mut prob_0 = 0.0;
        let mut prob_1 = 0.0;

        for (i, amplitude) in self.state_vector.iter().enumerate() {
            let bit = (i >> qubit) & 1;
            let prob = amplitude.magnitude_squared();
            
            if bit == 0 {
                prob_0 += prob;
            } else {
                prob_1 += prob;
            }
        }

        (prob_0, prob_1)
    }

    /// Get expectation value of Pauli-Z operator on a qubit
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let (prob_0, prob_1) = self.measure_qubit(qubit);
        prob_0 - prob_1
    }

    /// Get all measurement probabilities
    pub fn get_probabilities(&self) -> Vec<f64> {
        self.state_vector
            .iter()
            .map(|amp| amp.magnitude_squared())
            .collect()
    }
}

/// Variational Quantum Eigensolver (VQE) implementation
#[derive(Debug, Clone)]
pub struct VariationalQuantumEigensolver {
    /// Quantum circuit ansatz
    pub ansatz: QuantumCircuit,
    /// Hamiltonian terms (Pauli strings)
    pub hamiltonian: Vec<(f64, Vec<(usize, char)>)>, // (coefficient, [(qubit, pauli_op)])
    /// Current parameters
    pub parameters: Vec<f64>,
    /// Optimization history
    pub energy_history: Vec<f64>,
}

impl VariationalQuantumEigensolver {
    /// Create new VQE instance
    pub fn new(num_qubits: usize, ansatz_depth: usize) -> Self {
        let mut ansatz = QuantumCircuit::new(num_qubits);
        let mut parameters = Vec::new();
        let mut param_idx = 0;

        // Create hardware-efficient ansatz
        for layer in 0..ansatz_depth {
            // RY rotations
            for qubit in 0..num_qubits {
                ansatz.add_parameterized_gate(QuantumGate::RY(0.0), vec![param_idx]);
                parameters.push(0.1 * rand::random::<f64>());
                param_idx += 1;
            }

            // CNOT entangling gates
            for qubit in 0..num_qubits - 1 {
                ansatz.add_gate(QuantumGate::CNOT(qubit, qubit + 1));
            }

            // RZ rotations
            for qubit in 0..num_qubits {
                ansatz.add_parameterized_gate(QuantumGate::RZ(0.0), vec![param_idx]);
                parameters.push(0.1 * rand::random::<f64>());
                param_idx += 1;
            }
        }

        Self {
            ansatz,
            hamiltonian: Vec::new(),
            parameters,
            energy_history: Vec::new(),
        }
    }

    /// Add Hamiltonian term
    pub fn add_hamiltonian_term(&mut self, coefficient: f64, pauli_string: Vec<(usize, char)>) {
        self.hamiltonian.push((coefficient, pauli_string));
    }

    /// Compute energy expectation value
    pub fn compute_energy(&self, parameters: &[f64]) -> Result<f64> {
        let mut total_energy = 0.0;

        for (coefficient, pauli_string) in &self.hamiltonian {
            let mut simulator = QuantumSimulator::new(self.ansatz.num_qubits);
            let mut circuit = self.ansatz.clone();
            circuit.set_parameters(parameters.to_vec());
            
            simulator.execute_circuit(&circuit)?;

            // Compute expectation value for this Pauli string
            let mut expectation = 1.0;
            for &(qubit, pauli_op) in pauli_string {
                match pauli_op {
                    'Z' => expectation *= simulator.expectation_z(qubit),
                    'X' => {
                        // Apply H gate before Z measurement for X
                        let mut sim_copy = simulator.clone();
                        let h_matrix = circuit.get_single_qubit_matrix(&QuantumGate::H);
                        sim_copy.apply_single_qubit_gate(&h_matrix, qubit);
                        expectation *= sim_copy.expectation_z(qubit);
                    }
                    'Y' => {
                        // Apply S†H gate before Z measurement for Y
                        let mut sim_copy = simulator.clone();
                        let s_dag = Array2::from_shape_vec(
                            (2, 2),
                            vec![
                                Complex::new(1.0, 0.0),
                                Complex::new(0.0, 0.0),
                                Complex::new(0.0, 0.0),
                                Complex::new(0.0, -1.0),
                            ],
                        ).unwrap();
                        let h_matrix = circuit.get_single_qubit_matrix(&QuantumGate::H);
                        sim_copy.apply_single_qubit_gate(&s_dag, qubit);
                        sim_copy.apply_single_qubit_gate(&h_matrix, qubit);
                        expectation *= sim_copy.expectation_z(qubit);
                    }
                    'I' => {
                        // Identity - contributes 1.0
                    }
                    _ => return Err(anyhow!("Unknown Pauli operator: {}", pauli_op)),
                }
            }

            total_energy += coefficient * expectation;
        }

        Ok(total_energy)
    }

    /// Optimize parameters using gradient descent
    pub fn optimize(&mut self, max_iterations: usize, learning_rate: f64) -> Result<f64> {
        for iteration in 0..max_iterations {
            let current_energy = self.compute_energy(&self.parameters)?;
            self.energy_history.push(current_energy);

            // Compute gradients using parameter shift rule
            let mut gradients = vec![0.0; self.parameters.len()];
            let shift = PI / 2.0;

            for (i, &param) in self.parameters.iter().enumerate() {
                // Forward shift
                let mut params_plus = self.parameters.clone();
                params_plus[i] = param + shift;
                let energy_plus = self.compute_energy(&params_plus)?;

                // Backward shift
                let mut params_minus = self.parameters.clone();
                params_minus[i] = param - shift;
                let energy_minus = self.compute_energy(&params_minus)?;

                // Gradient using parameter shift rule
                gradients[i] = 0.5 * (energy_plus - energy_minus);
            }

            // Update parameters
            for (i, &gradient) in gradients.iter().enumerate() {
                self.parameters[i] -= learning_rate * gradient;
            }

            // Early convergence check
            if iteration > 10 {
                let recent_energies = &self.energy_history[iteration.saturating_sub(5)..];
                let energy_variance = recent_energies.iter()
                    .map(|&e| (e - current_energy).powi(2))
                    .sum::<f64>() / recent_energies.len() as f64;
                
                if energy_variance < 1e-8 {
                    break;
                }
            }
        }

        self.compute_energy(&self.parameters)
    }
}

/// Quantum Approximate Optimization Algorithm (QAOA) implementation
#[derive(Debug, Clone)]
pub struct QuantumApproximateOptimization {
    /// Number of qubits
    pub num_qubits: usize,
    /// QAOA depth (p layers)
    pub depth: usize,
    /// Cost Hamiltonian
    pub cost_hamiltonian: Vec<(f64, Vec<(usize, char)>)>,
    /// Mixer Hamiltonian (usually X gates)
    pub mixer_hamiltonian: Vec<(f64, Vec<(usize, char)>)>,
    /// Beta parameters (mixer angles)
    pub beta_params: Vec<f64>,
    /// Gamma parameters (cost angles)
    pub gamma_params: Vec<f64>,
    /// Optimization history
    pub cost_history: Vec<f64>,
}

impl QuantumApproximateOptimization {
    /// Create new QAOA instance
    pub fn new(num_qubits: usize, depth: usize) -> Self {
        let mut mixer_hamiltonian = Vec::new();
        
        // Default mixer: sum of X operators
        for qubit in 0..num_qubits {
            mixer_hamiltonian.push((1.0, vec![(qubit, 'X')]));
        }

        Self {
            num_qubits,
            depth,
            cost_hamiltonian: Vec::new(),
            mixer_hamiltonian,
            beta_params: vec![0.1; depth],
            gamma_params: vec![0.1; depth],
            cost_history: Vec::new(),
        }
    }

    /// Add cost term (typically ZZ interactions for MaxCut)
    pub fn add_cost_term(&mut self, coefficient: f64, pauli_string: Vec<(usize, char)>) {
        self.cost_hamiltonian.push((coefficient, pauli_string));
    }

    /// Build QAOA circuit
    pub fn build_circuit(&self) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(self.num_qubits);

        // Initial superposition
        for qubit in 0..self.num_qubits {
            circuit.add_gate(QuantumGate::H);
        }

        // QAOA layers
        for layer in 0..self.depth {
            // Cost unitary e^(-i*γ*H_C)
            for (coefficient, pauli_string) in &self.cost_hamiltonian {
                let angle = self.gamma_params[layer] * coefficient;
                
                // For ZZ terms, decompose into CNOTs and RZ
                if pauli_string.len() == 2 && 
                   pauli_string.iter().all(|(_, op)| *op == 'Z') {
                    let qubit1 = pauli_string[0].0;
                    let qubit2 = pauli_string[1].0;
                    
                    circuit.add_gate(QuantumGate::CNOT(qubit1, qubit2));
                    circuit.add_gate(QuantumGate::RZ(angle));
                    circuit.add_gate(QuantumGate::CNOT(qubit1, qubit2));
                }
            }

            // Mixer unitary e^(-i*β*H_M)
            let beta = self.beta_params[layer];
            for qubit in 0..self.num_qubits {
                circuit.add_gate(QuantumGate::RX(2.0 * beta));
            }
        }

        circuit
    }

    /// Compute cost function expectation
    pub fn compute_cost(&self) -> Result<f64> {
        let circuit = self.build_circuit();
        let mut simulator = QuantumSimulator::new(self.num_qubits);
        simulator.execute_circuit(&circuit)?;

        let mut total_cost = 0.0;
        for (coefficient, pauli_string) in &self.cost_hamiltonian {
            let mut expectation = 1.0;
            for &(qubit, pauli_op) in pauli_string {
                match pauli_op {
                    'Z' => expectation *= simulator.expectation_z(qubit),
                    _ => return Err(anyhow!("QAOA cost function only supports Z operators")),
                }
            }
            total_cost += coefficient * expectation;
        }

        Ok(total_cost)
    }

    /// Optimize QAOA parameters
    pub fn optimize(&mut self, max_iterations: usize, learning_rate: f64) -> Result<f64> {
        for iteration in 0..max_iterations {
            let current_cost = self.compute_cost()?;
            self.cost_history.push(current_cost);

            // Simple gradient descent on parameters
            let mut beta_gradients = vec![0.0; self.depth];
            let mut gamma_gradients = vec![0.0; self.depth];
            let shift = 0.1;

            // Compute gradients for beta parameters
            for i in 0..self.depth {
                // Forward shift
                self.beta_params[i] += shift;
                let cost_plus = self.compute_cost()?;
                
                // Backward shift
                self.beta_params[i] -= 2.0 * shift;
                let cost_minus = self.compute_cost()?;
                
                // Restore original value
                self.beta_params[i] += shift;
                
                beta_gradients[i] = (cost_plus - cost_minus) / (2.0 * shift);
            }

            // Compute gradients for gamma parameters
            for i in 0..self.depth {
                // Forward shift
                self.gamma_params[i] += shift;
                let cost_plus = self.compute_cost()?;
                
                // Backward shift
                self.gamma_params[i] -= 2.0 * shift;
                let cost_minus = self.compute_cost()?;
                
                // Restore original value
                self.gamma_params[i] += shift;
                
                gamma_gradients[i] = (cost_plus - cost_minus) / (2.0 * shift);
            }

            // Update parameters (minimize cost)
            for i in 0..self.depth {
                self.beta_params[i] -= learning_rate * beta_gradients[i];
                self.gamma_params[i] -= learning_rate * gamma_gradients[i];
            }

            // Early convergence
            if iteration > 10 {
                let recent_variance = self.cost_history[iteration.saturating_sub(5)..]
                    .iter()
                    .map(|&c| (c - current_cost).powi(2))
                    .sum::<f64>() / 5.0;
                
                if recent_variance < 1e-8 {
                    break;
                }
            }
        }

        self.compute_cost()
    }
}

/// Quantum Neural Network layer
#[derive(Debug, Clone)]
pub struct QuantumNeuralNetworkLayer {
    /// Number of qubits
    pub num_qubits: usize,
    /// Layer parameters
    pub parameters: Vec<f64>,
    /// Layer type
    pub layer_type: QNNLayerType,
}

/// Types of QNN layers
#[derive(Debug, Clone)]
pub enum QNNLayerType {
    /// Strongly entangling layer
    StronglyEntangling,
    /// Basic entangling layer
    BasicEntangling,
    /// Amplitude embedding layer
    AmplitudeEmbedding,
    /// Angle embedding layer
    AngleEmbedding,
}

impl QuantumNeuralNetworkLayer {
    /// Create new QNN layer
    pub fn new(num_qubits: usize, layer_type: QNNLayerType) -> Self {
        let num_params = match layer_type {
            QNNLayerType::StronglyEntangling => 3 * num_qubits,
            QNNLayerType::BasicEntangling => num_qubits,
            QNNLayerType::AmplitudeEmbedding => 2_usize.pow(num_qubits as u32) - 1,
            QNNLayerType::AngleEmbedding => num_qubits,
        };

        let parameters = (0..num_params)
            .map(|_| 2.0 * PI * rand::random::<f64>())
            .collect();

        Self {
            num_qubits,
            parameters,
            layer_type,
        }
    }

    /// Build circuit for this layer
    pub fn build_circuit(&self, input_data: Option<&[f64]>) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(self.num_qubits);

        match &self.layer_type {
            QNNLayerType::StronglyEntangling => {
                // Strongly entangling layer with RX, RY, RZ rotations and CNOT gates
                for i in 0..self.num_qubits {
                    circuit.add_gate(QuantumGate::RX(self.parameters[3 * i]));
                    circuit.add_gate(QuantumGate::RY(self.parameters[3 * i + 1]));
                    circuit.add_gate(QuantumGate::RZ(self.parameters[3 * i + 2]));
                }

                // Entangling gates
                for i in 0..self.num_qubits {
                    circuit.add_gate(QuantumGate::CNOT(i, (i + 1) % self.num_qubits));
                }
            }
            QNNLayerType::BasicEntangling => {
                // Basic entangling layer with RY rotations
                for i in 0..self.num_qubits {
                    circuit.add_gate(QuantumGate::RY(self.parameters[i]));
                }

                // Circular CNOT gates
                for i in 0..self.num_qubits {
                    circuit.add_gate(QuantumGate::CNOT(i, (i + 1) % self.num_qubits));
                }
            }
            QNNLayerType::AmplitudeEmbedding => {
                if let Some(data) = input_data {
                    // Encode classical data into quantum amplitudes
                    let normalized_data = self.normalize_data(data);
                    // This would require a complex amplitude embedding procedure
                    // For now, use angle embedding as approximation
                    for (i, &value) in normalized_data.iter().enumerate() {
                        if i < self.num_qubits {
                            circuit.add_gate(QuantumGate::RY(value * PI));
                        }
                    }
                }
            }
            QNNLayerType::AngleEmbedding => {
                if let Some(data) = input_data {
                    // Encode data into rotation angles
                    for (i, &value) in data.iter().enumerate() {
                        if i < self.num_qubits {
                            circuit.add_gate(QuantumGate::RY(value * self.parameters[i]));
                        }
                    }
                }
            }
        }

        circuit
    }

    /// Normalize input data
    fn normalize_data(&self, data: &[f64]) -> Vec<f64> {
        let norm = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            data.iter().map(|x| x / norm).collect()
        } else {
            data.to_vec()
        }
    }
}

/// Quantum Neural Network
#[derive(Debug, Clone)]
pub struct QuantumNeuralNetwork {
    /// Number of qubits
    pub num_qubits: usize,
    /// QNN layers
    pub layers: Vec<QuantumNeuralNetworkLayer>,
    /// Output measurement strategy
    pub measurement_strategy: MeasurementStrategy,
    /// Training history
    pub loss_history: Vec<f64>,
}

/// Measurement strategies for QNN output
#[derive(Debug, Clone)]
pub enum MeasurementStrategy {
    /// Expectation values of Pauli-Z on all qubits
    ExpectationZ,
    /// Expectation values of Pauli-X on all qubits
    ExpectationX,
    /// Probability of measuring |0⟩ on all qubits
    Probability0,
    /// Custom Pauli string measurements
    CustomPauli(Vec<(usize, char)>),
}

impl QuantumNeuralNetwork {
    /// Create new QNN
    pub fn new(num_qubits: usize, layer_configs: Vec<QNNLayerType>) -> Self {
        let layers = layer_configs
            .into_iter()
            .map(|layer_type| QuantumNeuralNetworkLayer::new(num_qubits, layer_type))
            .collect();

        Self {
            num_qubits,
            layers,
            measurement_strategy: MeasurementStrategy::ExpectationZ,
            loss_history: Vec::new(),
        }
    }

    /// Forward pass through QNN
    pub fn forward(&self, input_data: &[f64]) -> Result<Vec<f64>> {
        let mut circuit = QuantumCircuit::new(self.num_qubits);

        // Build circuit from all layers
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_circuit = if i == 0 {
                // First layer gets input data
                layer.build_circuit(Some(input_data))
            } else {
                // Subsequent layers use their parameters
                layer.build_circuit(None)
            };

            // Add layer gates to main circuit
            for gate in layer_circuit.gates {
                circuit.add_gate(gate);
            }
        }

        // Execute circuit
        let mut simulator = QuantumSimulator::new(self.num_qubits);
        simulator.execute_circuit(&circuit)?;

        // Perform measurements based on strategy
        let output = match &self.measurement_strategy {
            MeasurementStrategy::ExpectationZ => {
                (0..self.num_qubits)
                    .map(|qubit| simulator.expectation_z(qubit))
                    .collect()
            }
            MeasurementStrategy::ExpectationX => {
                let mut outputs = Vec::new();
                for qubit in 0..self.num_qubits {
                    let mut sim_copy = simulator.clone();
                    let h_matrix = circuit.get_single_qubit_matrix(&QuantumGate::H);
                    sim_copy.apply_single_qubit_gate(&h_matrix, qubit);
                    outputs.push(sim_copy.expectation_z(qubit));
                }
                outputs
            }
            MeasurementStrategy::Probability0 => {
                (0..self.num_qubits)
                    .map(|qubit| simulator.measure_qubit(qubit).0)
                    .collect()
            }
            MeasurementStrategy::CustomPauli(pauli_string) => {
                let mut expectation = 1.0;
                for &(qubit, pauli_op) in pauli_string {
                    match pauli_op {
                        'Z' => expectation *= simulator.expectation_z(qubit),
                        'X' => {
                            let mut sim_copy = simulator.clone();
                            let h_matrix = circuit.get_single_qubit_matrix(&QuantumGate::H);
                            sim_copy.apply_single_qubit_gate(&h_matrix, qubit);
                            expectation *= sim_copy.expectation_z(qubit);
                        }
                        'I' => {} // Identity
                        _ => return Err(anyhow!("Unsupported Pauli operator: {}", pauli_op)),
                    }
                }
                vec![expectation]
            }
        };

        Ok(output)
    }

    /// Train QNN using gradient descent
    pub fn train(
        &mut self,
        training_data: &[(Vec<f64>, Vec<f64>)],
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input, target) in training_data {
                let prediction = self.forward(input)?;
                
                // Mean squared error loss
                let loss = prediction
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>() / prediction.len() as f64;
                
                total_loss += loss;

                // Update parameters using finite difference gradients
                self.update_parameters(input, target, learning_rate)?;
            }

            self.loss_history.push(total_loss / training_data.len() as f64);

            if epoch % 10 == 0 {
                println!("Epoch {}: Average Loss = {:.6}", epoch, self.loss_history.last().unwrap());
            }
        }

        Ok(())
    }

    /// Update parameters using gradient descent
    fn update_parameters(
        &mut self,
        input: &[f64],
        target: &[f64],
        learning_rate: f64,
    ) -> Result<()> {
        let shift = 0.01;

        // Collect all gradients first
        let mut gradients = Vec::new();
        
        // Collect layer information first to avoid borrow conflicts
        let layer_info: Vec<(usize, usize)> = self.layers.iter().enumerate()
            .map(|(idx, layer)| (idx, layer.parameters.len()))
            .collect();
        
        for (layer_idx, param_count) in layer_info {
            let mut layer_gradients = Vec::new();
            
            for i in 0..param_count {
                // Forward shift
                self.layers[layer_idx].parameters[i] += shift;
                let prediction_plus = self.forward(input)?;
                let loss_plus = prediction_plus
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>();

                // Backward shift
                self.layers[layer_idx].parameters[i] -= 2.0 * shift;
                let prediction_minus = self.forward(input)?;
                let loss_minus = prediction_minus
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>();

                // Restore original parameter
                self.layers[layer_idx].parameters[i] += shift;
                
                // Calculate gradient
                let gradient = (loss_plus - loss_minus) / (2.0 * shift);
                layer_gradients.push(gradient);
            }
            gradients.push(layer_gradients);
        }

        // Apply all gradients
        for (layer_idx, layer_gradients) in gradients.iter().enumerate() {
            for (param_idx, gradient) in layer_gradients.iter().enumerate() {
                self.layers[layer_idx].parameters[param_idx] -= learning_rate * gradient;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        
        let sum = a + b;
        assert_eq!(sum.real, 4.0);
        assert_eq!(sum.imag, 6.0);

        let product = a * b;
        assert_eq!(product.real, -5.0); // 1*3 - 2*4
        assert_eq!(product.imag, 10.0); // 1*4 + 2*3
    }

    #[test]
    fn test_quantum_gates() {
        let circuit = QuantumCircuit::new(1);
        
        // Test Pauli-X gate
        let x_matrix = circuit.get_single_qubit_matrix(&QuantumGate::X);
        assert_eq!(x_matrix[(0, 0)].real, 0.0);
        assert_eq!(x_matrix[(0, 1)].real, 1.0);
        assert_eq!(x_matrix[(1, 0)].real, 1.0);
        assert_eq!(x_matrix[(1, 1)].real, 0.0);

        // Test Hadamard gate
        let h_matrix = circuit.get_single_qubit_matrix(&QuantumGate::H);
        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        assert!((h_matrix[(0, 0)].real - inv_sqrt2).abs() < 1e-10);
        assert!((h_matrix[(1, 1)].real + inv_sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_simulator() {
        let mut simulator = QuantumSimulator::new(2);
        
        // Initial state should be |00⟩
        assert_eq!(simulator.state_vector[0], Complex::new(1.0, 0.0));
        assert_eq!(simulator.state_vector[1], Complex::new(0.0, 0.0));
        assert_eq!(simulator.state_vector[2], Complex::new(0.0, 0.0));
        assert_eq!(simulator.state_vector[3], Complex::new(0.0, 0.0));

        // Apply H gate to first qubit
        let circuit = QuantumCircuit::new(2);
        let h_matrix = circuit.get_single_qubit_matrix(&QuantumGate::H);
        simulator.apply_single_qubit_gate(&h_matrix, 0);

        // Should be in superposition (|00⟩ + |10⟩)/√2
        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        assert!((simulator.state_vector[0].real - inv_sqrt2).abs() < 1e-10);
        assert!((simulator.state_vector[2].real - inv_sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        let mut simulator = QuantumSimulator::new(2);
        
        // Prepare |10⟩ state
        let circuit = QuantumCircuit::new(2);
        let x_matrix = circuit.get_single_qubit_matrix(&QuantumGate::X);
        simulator.apply_single_qubit_gate(&x_matrix, 0);
        
        // Apply CNOT
        simulator.apply_cnot(0, 1);
        
        // Should be in |11⟩ state
        assert_eq!(simulator.state_vector[0], Complex::new(0.0, 0.0));
        assert_eq!(simulator.state_vector[1], Complex::new(0.0, 0.0));
        assert_eq!(simulator.state_vector[2], Complex::new(0.0, 0.0));
        assert_eq!(simulator.state_vector[3], Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_vqe_creation() {
        let vqe = VariationalQuantumEigensolver::new(2, 1);
        assert_eq!(vqe.ansatz.num_qubits, 2);
        assert!(!vqe.parameters.is_empty());
    }

    #[test]
    fn test_qaoa_creation() {
        let mut qaoa = QuantumApproximateOptimization::new(3, 2);
        assert_eq!(qaoa.num_qubits, 3);
        assert_eq!(qaoa.depth, 2);
        assert_eq!(qaoa.beta_params.len(), 2);
        assert_eq!(qaoa.gamma_params.len(), 2);

        // Add MaxCut cost terms
        qaoa.add_cost_term(0.5, vec![(0, 'Z'), (1, 'Z')]);
        qaoa.add_cost_term(0.5, vec![(1, 'Z'), (2, 'Z')]);
        
        assert_eq!(qaoa.cost_hamiltonian.len(), 2);
    }

    #[test]
    fn test_qnn_layer() {
        let layer = QuantumNeuralNetworkLayer::new(2, QNNLayerType::BasicEntangling);
        assert_eq!(layer.num_qubits, 2);
        assert_eq!(layer.parameters.len(), 2);

        let circuit = layer.build_circuit(None);
        assert_eq!(circuit.num_qubits, 2);
        assert!(!circuit.gates.is_empty());
    }

    #[test]
    fn test_qnn_forward() {
        let qnn = QuantumNeuralNetwork::new(
            2,
            vec![QNNLayerType::AngleEmbedding, QNNLayerType::BasicEntangling],
        );

        let input = vec![0.5, 0.8];
        let output = qnn.forward(&input).unwrap();

        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_rotation_gates() {
        let circuit = QuantumCircuit::new(1);
        
        // Test RY gate with π/2 rotation
        let ry_matrix = circuit.get_single_qubit_matrix(&QuantumGate::RY(PI / 2.0));
        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        
        assert!((ry_matrix[(0, 0)].real - inv_sqrt2).abs() < 1e-10);
        assert!((ry_matrix[(0, 1)].real + inv_sqrt2).abs() < 1e-10);
        assert!((ry_matrix[(1, 0)].real - inv_sqrt2).abs() < 1e-10);
        assert!((ry_matrix[(1, 1)].real - inv_sqrt2).abs() < 1e-10);
    }
}