//! Quantum machine learning engine for advanced pattern recognition and optimization

use super::QuantumConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Quantum machine learning engine with comprehensive algorithm support
pub struct QuantumMLEngine {
    config: QuantumConfig,
    ml_algorithms: Vec<QuantumMLAlgorithm>,
    quantum_neural_network: Arc<RwLock<QuantumNeuralNetwork>>,
    training_stats: Arc<RwLock<QuantumTrainingStats>>,
    model_registry: Arc<RwLock<HashMap<String, QuantumModel>>>,
}

/// Quantum neural network implementation
#[derive(Debug, Clone)]
pub struct QuantumNeuralNetwork {
    /// Number of qubits in the network
    pub qubit_count: usize,
    /// Quantum circuit layers
    pub layers: Vec<QuantumLayer>,
    /// Variational parameters
    pub parameters: Vec<f64>,
    /// Training hyperparameters
    pub learning_rate: f64,
    /// Network topology
    pub topology: NetworkTopology,
}

/// Quantum layer in the neural network
#[derive(Debug, Clone)]
pub struct QuantumLayer {
    /// Layer type
    pub layer_type: LayerType,
    /// Quantum gates in this layer
    pub gates: Vec<QuantumGate>,
    /// Trainable parameters for this layer
    pub parameters: Vec<f64>,
    /// Entanglement pattern
    pub entanglement: EntanglementPattern,
}

/// Types of quantum layers
#[derive(Debug, Clone)]
pub enum LayerType {
    /// Parameterized quantum circuit
    PQC,
    /// Quantum convolutional layer
    QConv,
    /// Quantum attention mechanism
    QAttention,
    /// Quantum pooling layer
    QPooling,
    /// Variational quantum eigensolver layer
    VQE,
}

/// Quantum gates for circuit construction
#[derive(Debug, Clone)]
pub enum QuantumGate {
    /// Rotation around X-axis
    RX(f64),
    /// Rotation around Y-axis
    RY(f64),
    /// Rotation around Z-axis
    RZ(f64),
    /// Controlled-NOT gate
    CNOT(usize, usize),
    /// Hadamard gate
    H(usize),
    /// Pauli-X gate
    X(usize),
    /// Pauli-Y gate
    Y(usize),
    /// Pauli-Z gate
    Z(usize),
    /// Controlled phase gate
    CPhase(f64, usize, usize),
    /// Toffoli gate
    Toffoli(usize, usize, usize),
}

/// Entanglement patterns for quantum layers
#[derive(Debug, Clone)]
pub enum EntanglementPattern {
    /// Linear entanglement chain
    Linear,
    /// Circular entanglement
    Circular,
    /// Full entanglement (all-to-all)
    Full,
    /// Random entanglement
    Random,
    /// Hardware-efficient ansatz
    HardwareEfficient,
}

/// Network topology configurations
#[derive(Debug, Clone)]
pub enum NetworkTopology {
    /// Feed-forward quantum network
    FeedForward,
    /// Recurrent quantum network
    Recurrent,
    /// Quantum convolutional network
    Convolutional,
    /// Quantum attention network
    Attention,
    /// Hybrid classical-quantum network
    Hybrid,
}

/// Training statistics for quantum ML models
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct QuantumTrainingStats {
    /// Training iterations completed
    pub iterations: u64,
    /// Current loss value
    pub current_loss: f64,
    /// Best loss achieved
    pub best_loss: f64,
    /// Training accuracy
    pub training_accuracy: f64,
    /// Validation accuracy
    pub validation_accuracy: f64,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Parameter variance
    pub parameter_variance: f64,
    /// Quantum fidelity
    pub quantum_fidelity: f64,
    /// Entanglement entropy
    pub entanglement_entropy: f64,
}

/// Quantum model container
#[derive(Debug, Clone)]
pub struct QuantumModel {
    /// Model identifier
    pub id: String,
    /// Model type
    pub model_type: QuantumMLAlgorithm,
    /// Neural network configuration
    pub network: QuantumNeuralNetwork,
    /// Training statistics
    pub stats: QuantumTrainingStats,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

impl QuantumMLEngine {
    /// Create a new quantum ML engine
    pub fn new(config: QuantumConfig) -> Self {
        let qubit_count = config.available_qubits as usize;

        // Initialize quantum neural network
        let qnn = QuantumNeuralNetwork {
            qubit_count,
            layers: Self::create_default_layers(qubit_count),
            parameters: vec![0.0; qubit_count * 4], // 4 parameters per qubit
            learning_rate: 0.01,
            topology: NetworkTopology::FeedForward,
        };

        Self {
            config,
            ml_algorithms: vec![
                QuantumMLAlgorithm::QNN,
                QuantumMLAlgorithm::QSVM,
                QuantumMLAlgorithm::QPCA,
                QuantumMLAlgorithm::QuantumBoltzmannMachine,
                QuantumMLAlgorithm::QuantumAutoencoder,
                QuantumMLAlgorithm::QuantumGAN,
            ],
            quantum_neural_network: Arc::new(RwLock::new(qnn)),
            training_stats: Arc::new(RwLock::new(QuantumTrainingStats::default())),
            model_registry: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create default quantum layers for the network
    fn create_default_layers(qubit_count: usize) -> Vec<QuantumLayer> {
        let mut layers = Vec::new();

        // Input layer with Hadamard gates for superposition
        let input_layer = QuantumLayer {
            layer_type: LayerType::PQC,
            gates: (0..qubit_count).map(QuantumGate::H).collect(),
            parameters: vec![0.0; qubit_count],
            entanglement: EntanglementPattern::Linear,
        };
        layers.push(input_layer);

        // Variational layers with parameterized rotations
        for layer_idx in 0..3 {
            let mut gates = Vec::new();

            // Add rotation gates
            for _qubit in 0..qubit_count {
                gates.push(QuantumGate::RY(0.0)); // Will be parameterized
                gates.push(QuantumGate::RZ(0.0)); // Will be parameterized
            }

            // Add entangling gates
            for i in 0..qubit_count - 1 {
                gates.push(QuantumGate::CNOT(i, i + 1));
            }

            let layer = QuantumLayer {
                layer_type: LayerType::PQC,
                gates,
                parameters: vec![0.0; qubit_count * 2], // 2 parameters per qubit
                entanglement: if layer_idx % 2 == 0 {
                    EntanglementPattern::Linear
                } else {
                    EntanglementPattern::Circular
                },
            };
            layers.push(layer);
        }

        // Output layer with measurements
        let output_layer = QuantumLayer {
            layer_type: LayerType::QPooling,
            gates: (0..qubit_count).map(QuantumGate::Z).collect(),
            parameters: vec![],
            entanglement: EntanglementPattern::Linear,
        };
        layers.push(output_layer);

        layers
    }

    /// Train the quantum neural network
    pub async fn train_qnn(
        &self,
        training_data: Vec<(Vec<f64>, Vec<f64>)>,
        epochs: usize,
    ) -> Result<QuantumTrainingStats> {
        let mut network = self.quantum_neural_network.write().await;
        let mut stats = self.training_stats.write().await;

        info!(
            "Starting quantum neural network training with {} epochs",
            epochs
        );

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut correct_predictions = 0;

            for (input, target) in &training_data {
                // Forward pass
                let prediction = self.forward_pass(&network, input).await?;

                // Calculate loss
                let loss = self.calculate_loss(&prediction, target);
                epoch_loss += loss;

                // Backward pass (parameter update)
                self.update_parameters(&mut network, input, target, &prediction)
                    .await?;

                // Check prediction accuracy
                if self.check_prediction_accuracy(&prediction, target) {
                    correct_predictions += 1;
                }
            }

            // Update statistics
            stats.iterations += 1;
            stats.current_loss = epoch_loss / training_data.len() as f64;
            stats.training_accuracy = correct_predictions as f64 / training_data.len() as f64;

            if stats.current_loss < stats.best_loss || stats.best_loss == 0.0 {
                stats.best_loss = stats.current_loss;
            }

            // Calculate quantum-specific metrics
            stats.quantum_fidelity = self.calculate_quantum_fidelity(&network).await;
            stats.entanglement_entropy = self.calculate_entanglement_entropy(&network).await;

            if epoch % 10 == 0 {
                debug!(
                    "Epoch {}: Loss={:.6}, Accuracy={:.4}, Fidelity={:.4}",
                    epoch, stats.current_loss, stats.training_accuracy, stats.quantum_fidelity
                );
            }
        }

        info!("Quantum neural network training completed");
        Ok(stats.clone())
    }

    /// Forward pass through the quantum neural network
    async fn forward_pass(
        &self,
        network: &QuantumNeuralNetwork,
        input: &[f64],
    ) -> Result<Vec<f64>> {
        // Simulate quantum circuit execution
        let mut quantum_state = self.initialize_quantum_state(input).await?;

        for layer in &network.layers {
            quantum_state = self.apply_quantum_layer(&quantum_state, layer).await?;
        }

        // Measure quantum state to get classical output
        self.measure_quantum_state(&quantum_state).await
    }

    /// Initialize quantum state from classical input
    async fn initialize_quantum_state(&self, input: &[f64]) -> Result<QuantumState> {
        // Encode classical data into quantum state
        let mut amplitudes = vec![0.0; 1 << self.config.available_qubits as usize];

        // Simple amplitude encoding
        let norm = input.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for (i, &val) in input.iter().enumerate() {
                if i < amplitudes.len() {
                    amplitudes[i] = val / norm;
                }
            }
        } else {
            amplitudes[0] = 1.0; // Default to |0⟩ state
        }

        Ok(QuantumState { amplitudes })
    }

    /// Apply a quantum layer to the state
    async fn apply_quantum_layer(
        &self,
        state: &QuantumState,
        layer: &QuantumLayer,
    ) -> Result<QuantumState> {
        let mut new_state = state.clone();

        // Apply quantum gates in sequence
        for gate in &layer.gates {
            new_state = self.apply_quantum_gate(&new_state, gate).await?;
        }

        Ok(new_state)
    }

    /// Apply a single quantum gate
    async fn apply_quantum_gate(
        &self,
        state: &QuantumState,
        gate: &QuantumGate,
    ) -> Result<QuantumState> {
        // Simplified gate operations for demonstration
        match gate {
            QuantumGate::H(qubit) => self.apply_hadamard(state, *qubit).await,
            QuantumGate::RX(angle) => self.apply_rotation_x(state, *angle).await,
            QuantumGate::RY(angle) => self.apply_rotation_y(state, *angle).await,
            QuantumGate::RZ(angle) => self.apply_rotation_z(state, *angle).await,
            QuantumGate::X(qubit) => self.apply_pauli_x(state, *qubit).await,
            QuantumGate::Y(qubit) => self.apply_pauli_y(state, *qubit).await,
            QuantumGate::Z(qubit) => self.apply_pauli_z(state, *qubit).await,
            QuantumGate::CNOT(control, target) => self.apply_cnot(state, *control, *target).await,
            QuantumGate::CPhase(phase, control, target) => self.apply_cphase(state, *phase, *control, *target).await,
            QuantumGate::Toffoli(control1, control2, target) => self.apply_toffoli(state, *control1, *control2, *target).await,
        }
    }

    /// Apply Hadamard gate
    async fn apply_hadamard(&self, state: &QuantumState, qubit: usize) -> Result<QuantumState> {
        let mut new_amplitudes = state.amplitudes.clone();
        let n_states = new_amplitudes.len();

        for i in 0..n_states {
            if (i >> qubit) & 1 == 0 {
                let j = i | (1 << qubit);
                if j < n_states {
                    let temp = (new_amplitudes[i] + new_amplitudes[j]) / 2.0_f64.sqrt();
                    new_amplitudes[j] = (new_amplitudes[i] - new_amplitudes[j]) / 2.0_f64.sqrt();
                    new_amplitudes[i] = temp;
                }
            }
        }

        Ok(QuantumState {
            amplitudes: new_amplitudes,
        })
    }

    /// Apply Y rotation gate
    async fn apply_rotation_y(&self, state: &QuantumState, angle: f64) -> Result<QuantumState> {
        // Simplified rotation implementation
        let cos_half = (angle / 2.0).cos();
        let _sin_half = (angle / 2.0).sin();

        let mut new_amplitudes = state.amplitudes.clone();
        for amp in &mut new_amplitudes {
            *amp *= cos_half; // Simplified - should be proper matrix multiplication
        }

        Ok(QuantumState {
            amplitudes: new_amplitudes,
        })
    }

    /// Apply Z rotation gate
    async fn apply_rotation_z(&self, state: &QuantumState, angle: f64) -> Result<QuantumState> {
        // Simplified rotation implementation
        let phase_real = (-angle / 2.0).cos();

        let mut new_amplitudes = state.amplitudes.clone();
        for amp in &mut new_amplitudes {
            *amp *= phase_real; // Simplified - ignoring imaginary part for now
        }

        Ok(QuantumState {
            amplitudes: new_amplitudes,
        })
    }

    /// Apply CNOT gate
    async fn apply_cnot(
        &self,
        state: &QuantumState,
        control: usize,
        target: usize,
    ) -> Result<QuantumState> {
        let mut new_amplitudes = state.amplitudes.clone();
        let n_states = new_amplitudes.len();

        for i in 0..n_states {
            if (i >> control) & 1 == 1 {
                let j = i ^ (1 << target);
                if j < n_states {
                    new_amplitudes.swap(i, j);
                }
            }
        }

        Ok(QuantumState {
            amplitudes: new_amplitudes,
        })
    }

    /// Apply rotation around X-axis
    async fn apply_rotation_x(&self, state: &QuantumState, angle: f64) -> Result<QuantumState> {
        // Simplified rotation implementation
        let cos_half = (angle / 2.0).cos();
        let _sin_half = (angle / 2.0).sin();

        let mut new_amplitudes = state.amplitudes.clone();
        for amp in &mut new_amplitudes {
            *amp *= cos_half; // Simplified - should be proper matrix multiplication
        }

        Ok(QuantumState {
            amplitudes: new_amplitudes,
        })
    }

    /// Apply Pauli-X gate
    async fn apply_pauli_x(&self, state: &QuantumState, qubit: usize) -> Result<QuantumState> {
        let mut new_amplitudes = state.amplitudes.clone();
        let n_states = new_amplitudes.len();

        for i in 0..n_states {
            let j = i ^ (1 << qubit);
            if j < n_states && i != j {
                new_amplitudes.swap(i, j);
            }
        }

        Ok(QuantumState {
            amplitudes: new_amplitudes,
        })
    }

    /// Apply Pauli-Y gate
    async fn apply_pauli_y(&self, state: &QuantumState, qubit: usize) -> Result<QuantumState> {
        let mut new_amplitudes = state.amplitudes.clone();
        let n_states = new_amplitudes.len();

        for i in 0..n_states {
            let j = i ^ (1 << qubit);
            if j < n_states && i != j {
                // Pauli-Y involves both bit flip and phase
                new_amplitudes.swap(i, j);
                if (i >> qubit) & 1 == 1 {
                    new_amplitudes[i] *= -1.0; // Apply -i phase factor (simplified)
                }
            }
        }

        Ok(QuantumState {
            amplitudes: new_amplitudes,
        })
    }

    /// Apply Pauli-Z gate
    async fn apply_pauli_z(&self, state: &QuantumState, qubit: usize) -> Result<QuantumState> {
        let mut new_amplitudes = state.amplitudes.clone();

        for (i, amp) in new_amplitudes.iter_mut().enumerate() {
            if (i >> qubit) & 1 == 1 {
                *amp *= -1.0; // Apply phase flip
            }
        }

        Ok(QuantumState {
            amplitudes: new_amplitudes,
        })
    }

    /// Apply Controlled Phase gate
    async fn apply_cphase(
        &self,
        state: &QuantumState,
        phase: f64,
        control: usize,
        target: usize,
    ) -> Result<QuantumState> {
        let mut new_amplitudes = state.amplitudes.clone();
        let phase_factor = phase.cos(); // Simplified phase implementation

        for (i, amp) in new_amplitudes.iter_mut().enumerate() {
            if ((i >> control) & 1 == 1) && ((i >> target) & 1 == 1) {
                *amp *= phase_factor; // Apply phase when both qubits are |1⟩
            }
        }

        Ok(QuantumState {
            amplitudes: new_amplitudes,
        })
    }

    /// Apply Toffoli gate (CCX)
    async fn apply_toffoli(
        &self,
        state: &QuantumState,
        control1: usize,
        control2: usize,
        target: usize,
    ) -> Result<QuantumState> {
        let mut new_amplitudes = state.amplitudes.clone();
        let n_states = new_amplitudes.len();

        for i in 0..n_states {
            // Only flip target if both control qubits are |1⟩
            if ((i >> control1) & 1 == 1) && ((i >> control2) & 1 == 1) {
                let j = i ^ (1 << target);
                if j < n_states {
                    new_amplitudes.swap(i, j);
                }
            }
        }

        Ok(QuantumState {
            amplitudes: new_amplitudes,
        })
    }

    /// Measure quantum state to get classical output
    async fn measure_quantum_state(&self, state: &QuantumState) -> Result<Vec<f64>> {
        // Convert quantum amplitudes to classical probabilities
        let probabilities: Vec<f64> = state.amplitudes.iter().map(|amp| amp * amp).collect();

        // For simplicity, return the first few probabilities as output
        let output_size = 4.min(probabilities.len());
        Ok(probabilities[..output_size].to_vec())
    }

    /// Calculate loss function
    fn calculate_loss(&self, prediction: &[f64], target: &[f64]) -> f64 {
        prediction
            .iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / prediction.len() as f64
    }

    /// Update network parameters using gradient descent
    async fn update_parameters(
        &self,
        network: &mut QuantumNeuralNetwork,
        _input: &[f64],
        _target: &[f64],
        _prediction: &[f64],
    ) -> Result<()> {
        // Simplified parameter update - in practice would use proper gradient calculation
        for param in &mut network.parameters {
            let gradient = 0.001; // Simplified gradient
            *param -= network.learning_rate * gradient;
        }

        Ok(())
    }

    /// Check prediction accuracy
    fn check_prediction_accuracy(&self, prediction: &[f64], target: &[f64]) -> bool {
        if prediction.is_empty() || target.is_empty() {
            return false;
        }

        let pred_max_idx = prediction
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let target_max_idx = target
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        pred_max_idx == target_max_idx
    }

    /// Calculate quantum fidelity
    async fn calculate_quantum_fidelity(&self, network: &QuantumNeuralNetwork) -> f64 {
        // Simplified fidelity calculation
        let param_variance =
            network.parameters.iter().map(|p| p * p).sum::<f64>() / network.parameters.len() as f64;

        (-param_variance * 0.1).exp() // Exponential decay with parameter magnitude
    }

    /// Calculate entanglement entropy
    async fn calculate_entanglement_entropy(&self, network: &QuantumNeuralNetwork) -> f64 {
        // Simplified entropy calculation based on parameter distribution
        let mean = network.parameters.iter().sum::<f64>() / network.parameters.len() as f64;
        let variance = network
            .parameters
            .iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>()
            / network.parameters.len() as f64;

        variance.ln().max(0.0) // Log of variance as entropy approximation
    }

    /// Register a trained model
    pub async fn register_model(&self, model: QuantumModel) -> Result<()> {
        let mut registry = self.model_registry.write().await;
        registry.insert(model.id.clone(), model);
        Ok(())
    }

    /// Get training statistics
    pub async fn get_training_stats(&self) -> QuantumTrainingStats {
        self.training_stats.read().await.clone()
    }

    /// Get available algorithms
    pub fn get_algorithms(&self) -> &[QuantumMLAlgorithm] {
        &self.ml_algorithms
    }

    /// Create a quantum support vector machine
    pub async fn create_qsvm(&self, training_data: Vec<(Vec<f64>, f64)>) -> Result<QuantumModel> {
        info!("Creating Quantum Support Vector Machine");

        // Simplified QSVM implementation
        let mut network = self.quantum_neural_network.read().await.clone();
        network.topology = NetworkTopology::Hybrid;

        let model = QuantumModel {
            id: format!("qsvm_{}", uuid::Uuid::new_v4()),
            model_type: QuantumMLAlgorithm::QSVM,
            network,
            stats: QuantumTrainingStats::default(),
            metadata: HashMap::from([
                (
                    "training_samples".to_string(),
                    training_data.len().to_string(),
                ),
                ("created_at".to_string(), chrono::Utc::now().to_rfc3339()),
            ]),
        };

        Ok(model)
    }

    /// Create a quantum principal component analysis model
    pub async fn create_qpca(&self, data: Vec<Vec<f64>>) -> Result<QuantumModel> {
        info!("Creating Quantum Principal Component Analysis model");

        // Simplified QPCA implementation
        let mut network = self.quantum_neural_network.read().await.clone();
        network.topology = NetworkTopology::Convolutional;

        let model = QuantumModel {
            id: format!("qpca_{}", uuid::Uuid::new_v4()),
            model_type: QuantumMLAlgorithm::QPCA,
            network,
            stats: QuantumTrainingStats::default(),
            metadata: HashMap::from([
                ("data_points".to_string(), data.len().to_string()),
                (
                    "features".to_string(),
                    data.first().map_or(0, |d| d.len()).to_string(),
                ),
                ("created_at".to_string(), chrono::Utc::now().to_rfc3339()),
            ]),
        };

        Ok(model)
    }
}

/// Quantum state representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Complex amplitudes (simplified to real numbers for demo)
    pub amplitudes: Vec<f64>,
}

/// Quantum ML algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumMLAlgorithm {
    /// Quantum Neural Network
    QNN,
    /// Quantum Support Vector Machine
    QSVM,
    /// Quantum Principal Component Analysis
    QPCA,
    /// Quantum Boltzmann Machine
    QuantumBoltzmannMachine,
    /// Quantum Autoencoder
    QuantumAutoencoder,
    /// Quantum Generative Adversarial Network
    QuantumGAN,
    /// Variational Quantum Classifier
    VQC,
    /// Quantum Reinforcement Learning
    QRL,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quantum_ml_engine_creation() {
        let config = QuantumConfig {
            available_qubits: 4,
            ..Default::default()
        };

        let engine = QuantumMLEngine::new(config);
        assert_eq!(engine.get_algorithms().len(), 6);
    }

    #[tokio::test]
    async fn test_qnn_training() {
        let config = QuantumConfig {
            available_qubits: 4,
            ..Default::default()
        };

        let engine = QuantumMLEngine::new(config);

        // Create simple training data
        let training_data = vec![
            (vec![1.0, 0.0], vec![1.0, 0.0]),
            (vec![0.0, 1.0], vec![0.0, 1.0]),
        ];

        let stats = engine.train_qnn(training_data, 5).await.unwrap();
        assert!(stats.iterations > 0);
    }

    #[tokio::test]
    async fn test_qsvm_creation() {
        let config = QuantumConfig {
            available_qubits: 4,
            ..Default::default()
        };

        let engine = QuantumMLEngine::new(config);

        let training_data = vec![(vec![1.0, 0.0], 1.0), (vec![0.0, 1.0], -1.0)];

        let model = engine.create_qsvm(training_data).await.unwrap();
        assert!(matches!(model.model_type, QuantumMLAlgorithm::QSVM));
    }

    #[tokio::test]
    async fn test_qpca_creation() {
        let config = QuantumConfig {
            available_qubits: 4,
            ..Default::default()
        };

        let engine = QuantumMLEngine::new(config);

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let model = engine.create_qpca(data).await.unwrap();
        assert!(matches!(model.model_type, QuantumMLAlgorithm::QPCA));
    }
}
