//! # Quantum Communication Module
//!
//! Quantum entanglement-based communication system for ultra-secure streaming
//! with teleportation protocols, error correction, and distributed quantum processing.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info, warn};

use crate::event::StreamEvent;

/// Quantum communication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCommConfig {
    pub max_entangled_pairs: usize,
    pub decoherence_timeout_ms: u64,
    pub error_correction_threshold: f64,
    pub enable_quantum_teleportation: bool,
    pub enable_superdense_coding: bool,
    pub quantum_network_topology: NetworkTopology,
    pub security_protocols: Vec<QuantumSecurityProtocol>,
    pub entanglement_distribution: EntanglementDistribution,
}

/// Quantum network topologies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkTopology {
    FullyConnected,
    Star,
    Ring,
    Mesh,
    Hierarchical,
    AdaptiveHybrid,
}

/// Quantum security protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumSecurityProtocol {
    BB84,
    E91,
    SARG04,
    COW,
    DPS,
    ContinuousVariable,
}

/// Entanglement distribution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntanglementDistribution {
    DirectTransmission,
    EntanglementSwapping,
    QuantumRepeaters,
    SatelliteBased,
    HybridClassicalQuantum,
}

/// Quantum bit (qubit) representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qubit {
    pub id: String,
    pub state: QuantumState,
    pub entanglement_partner: Option<String>,
    pub coherence_time_remaining_ms: u64,
    pub measurement_history: Vec<MeasurementResult>,
    pub created_at: DateTime<Utc>,
    pub last_operation: Option<QuantumOperation>,
}

/// Quantum state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub alpha: Complex64, // |0⟩ amplitude
    pub beta: Complex64,  // |1⟩ amplitude
    pub phase: f64,
    pub purity: f64,   // Measure of quantum state purity
    pub fidelity: f64, // Fidelity with intended state
}

/// Complex number representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Complex64 {
    pub real: f64,
    pub imag: f64,
}

/// Quantum operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumOperation {
    PauliX,
    PauliY,
    PauliZ,
    Hadamard,
    Phase(f64),
    Rotation { axis: String, angle: f64 },
    CNOT { control: String, target: String },
    Measurement { basis: MeasurementBasis },
    Teleportation { target_node: String },
    ErrorCorrection,
    StatePreparation { target_state: QuantumState },
}

/// Measurement basis options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementBasis {
    Computational, // Z basis
    Diagonal,      // X basis
    Circular,      // Y basis
    Custom { theta: f64, phi: f64 },
}

/// Measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementResult {
    pub timestamp: DateTime<Utc>,
    pub basis: MeasurementBasis,
    pub outcome: u8, // 0 or 1
    pub confidence: f64,
    pub post_measurement_state: Option<QuantumState>,
}

/// Entangled pair of qubits
#[derive(Debug, Clone)]
pub struct EntangledPair {
    pub pair_id: String,
    pub qubit_a: Qubit,
    pub qubit_b: Qubit,
    pub entanglement_fidelity: f64,
    pub creation_time: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
    pub usage_count: u64,
    pub bell_state: BellState,
}

/// Bell states for entangled pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BellState {
    PhiPlus,  // |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    PhiMinus, // |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    PsiPlus,  // |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    PsiMinus, // |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
}

/// Quantum communication channel
#[derive(Debug, Clone)]
pub struct QuantumChannel {
    pub channel_id: String,
    pub source_node: String,
    pub destination_node: String,
    pub entangled_pairs: Vec<String>,
    pub channel_fidelity: f64,
    pub transmission_rate_qubits_per_sec: f64,
    pub error_rate: f64,
    pub channel_capacity: f64,
    pub quantum_protocol: QuantumSecurityProtocol,
    pub classical_channel: Option<String>, // For protocol coordination
}

/// Quantum error correction code
#[derive(Debug, Clone)]
pub struct QuantumErrorCorrection {
    pub code_type: ErrorCorrectionCode,
    pub logical_qubits: usize,
    pub physical_qubits: usize,
    pub threshold_error_rate: f64,
    pub correction_rounds: u32,
    pub syndrome_measurements: Vec<SyndromeMeasurement>,
}

/// Error correction codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorCorrectionCode {
    SteaneCode, // 7-qubit CSS code
    ShorCode,   // 9-qubit code
    Surface,    // Surface code
    ColorCode,  // Color code
    BCH,        // BCH codes
    LDPC,       // Low-density parity-check
    Stabilizer, // General stabilizer codes
}

/// Syndrome measurement for error detection
#[derive(Debug, Clone)]
pub struct SyndromeMeasurement {
    pub timestamp: DateTime<Utc>,
    pub stabilizer_generators: Vec<String>,
    pub syndrome_bits: Vec<u8>,
    pub detected_errors: Vec<ErrorType>,
    pub correction_applied: bool,
}

/// Types of quantum errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    BitFlip,
    PhaseFlip,
    Depolarizing,
    AmplitudeDamping,
    PhaseDamping,
    Decoherence,
    Crosstalk,
}

/// Quantum teleportation protocol
#[derive(Debug, Clone)]
pub struct TeleportationProtocol {
    pub protocol_id: String,
    pub source_qubit: String,
    pub entangled_pair: String,
    pub classical_bits: Vec<u8>,
    pub destination_node: String,
    pub fidelity_achieved: f64,
    pub protocol_duration_us: u64,
    pub success: bool,
}

/// Advanced quantum communication system
pub struct QuantumCommSystem {
    config: QuantumCommConfig,
    qubits: RwLock<HashMap<String, Qubit>>,
    entangled_pairs: RwLock<HashMap<String, EntangledPair>>,
    quantum_channels: RwLock<HashMap<String, QuantumChannel>>,
    error_correction: RwLock<HashMap<String, QuantumErrorCorrection>>,
    teleportation_protocols: RwLock<HashMap<String, TeleportationProtocol>>,
    network_topology: RwLock<NetworkTopology>,
    quantum_resources: Semaphore,
    performance_metrics: RwLock<QuantumMetrics>,
}

/// Quantum system performance metrics
#[derive(Debug, Clone, Default)]
pub struct QuantumMetrics {
    pub total_qubits_created: u64,
    pub total_entanglements: u64,
    pub total_teleportations: u64,
    pub successful_teleportations: u64,
    pub average_fidelity: f64,
    pub total_error_corrections: u64,
    pub decoherence_events: u64,
    pub channel_efficiency: f64,
    pub quantum_volume: u64,
}

impl QuantumCommSystem {
    /// Create new quantum communication system
    pub fn new(config: QuantumCommConfig) -> Self {
        let quantum_resources = Semaphore::new(config.max_entangled_pairs);

        Self {
            config,
            qubits: RwLock::new(HashMap::new()),
            entangled_pairs: RwLock::new(HashMap::new()),
            quantum_channels: RwLock::new(HashMap::new()),
            error_correction: RwLock::new(HashMap::new()),
            teleportation_protocols: RwLock::new(HashMap::new()),
            network_topology: RwLock::new(NetworkTopology::AdaptiveHybrid),
            quantum_resources,
            performance_metrics: RwLock::new(QuantumMetrics::default()),
        }
    }

    /// Create entangled pair of qubits
    pub async fn create_entangled_pair(&self, node_a: &str, node_b: &str) -> Result<String> {
        let _permit = self
            .quantum_resources
            .acquire()
            .await
            .map_err(|_| anyhow!("Failed to acquire quantum resources"))?;

        let pair_id = uuid::Uuid::new_v4().to_string();
        let timestamp = Utc::now();

        // Create entangled qubits in Bell state |Φ⁺⟩
        let qubit_a = Qubit {
            id: format!("{}_A", pair_id),
            state: QuantumState {
                alpha: Complex64 {
                    real: 1.0 / 2.0_f64.sqrt(),
                    imag: 0.0,
                },
                beta: Complex64 {
                    real: 1.0 / 2.0_f64.sqrt(),
                    imag: 0.0,
                },
                phase: 0.0,
                purity: 1.0,
                fidelity: 1.0,
            },
            entanglement_partner: Some(format!("{}_B", pair_id)),
            coherence_time_remaining_ms: self.config.decoherence_timeout_ms,
            measurement_history: Vec::new(),
            created_at: timestamp,
            last_operation: None,
        };

        let qubit_b = Qubit {
            id: format!("{}_B", pair_id),
            state: QuantumState {
                alpha: Complex64 {
                    real: 1.0 / 2.0_f64.sqrt(),
                    imag: 0.0,
                },
                beta: Complex64 {
                    real: 1.0 / 2.0_f64.sqrt(),
                    imag: 0.0,
                },
                phase: 0.0,
                purity: 1.0,
                fidelity: 1.0,
            },
            entanglement_partner: Some(format!("{}_A", pair_id)),
            coherence_time_remaining_ms: self.config.decoherence_timeout_ms,
            measurement_history: Vec::new(),
            created_at: timestamp,
            last_operation: None,
        };

        let entangled_pair = EntangledPair {
            pair_id: pair_id.clone(),
            qubit_a: qubit_a.clone(),
            qubit_b: qubit_b.clone(),
            entanglement_fidelity: 1.0,
            creation_time: timestamp,
            last_used: timestamp,
            usage_count: 0,
            bell_state: BellState::PhiPlus,
        };

        // Store qubits and pair
        self.qubits
            .write()
            .await
            .insert(qubit_a.id.clone(), qubit_a);
        self.qubits
            .write()
            .await
            .insert(qubit_b.id.clone(), qubit_b);
        self.entangled_pairs
            .write()
            .await
            .insert(pair_id.clone(), entangled_pair);

        // Update metrics
        let mut metrics = self.performance_metrics.write().await;
        metrics.total_qubits_created += 2;
        metrics.total_entanglements += 1;

        info!(
            "Created entangled pair {} between {} and {}",
            pair_id, node_a, node_b
        );
        Ok(pair_id)
    }

    /// Perform quantum teleportation
    pub async fn quantum_teleport(
        &self,
        source_qubit_id: &str,
        destination_node: &str,
    ) -> Result<TeleportationProtocol> {
        if !self.config.enable_quantum_teleportation {
            return Err(anyhow!("Quantum teleportation is disabled"));
        }

        let start_time = std::time::Instant::now();
        let protocol_id = uuid::Uuid::new_v4().to_string();

        // Find available entangled pair
        let entangled_pair_id = self.find_available_entangled_pair(destination_node).await?;

        // Perform Bell measurement on source qubit and one half of entangled pair
        let classical_bits = self
            .perform_bell_measurement(source_qubit_id, &entangled_pair_id)
            .await?;

        // Calculate fidelity (simplified simulation)
        let fidelity = self.calculate_teleportation_fidelity(&classical_bits).await;

        let protocol = TeleportationProtocol {
            protocol_id: protocol_id.clone(),
            source_qubit: source_qubit_id.to_string(),
            entangled_pair: entangled_pair_id,
            classical_bits,
            destination_node: destination_node.to_string(),
            fidelity_achieved: fidelity,
            protocol_duration_us: start_time.elapsed().as_micros() as u64,
            success: fidelity > 0.8, // Success threshold
        };

        // Store protocol
        self.teleportation_protocols
            .write()
            .await
            .insert(protocol_id.clone(), protocol.clone());

        // Update metrics
        let mut metrics = self.performance_metrics.write().await;
        metrics.total_teleportations += 1;
        if protocol.success {
            metrics.successful_teleportations += 1;
        }
        metrics.average_fidelity =
            (metrics.average_fidelity * (metrics.total_teleportations - 1) as f64 + fidelity)
                / metrics.total_teleportations as f64;

        info!(
            "Quantum teleportation {} completed with fidelity {:.3}",
            protocol_id, fidelity
        );
        Ok(protocol)
    }

    /// Find available entangled pair for destination
    async fn find_available_entangled_pair(&self, destination_node: &str) -> Result<String> {
        let pairs = self.entangled_pairs.read().await;

        for (pair_id, pair) in pairs.iter() {
            // Check if pair is still coherent and available
            let time_elapsed = Utc::now().signed_duration_since(pair.creation_time);
            if time_elapsed.num_milliseconds() < self.config.decoherence_timeout_ms as i64 {
                // Check if either qubit is at destination node (simplified)
                if pair.qubit_b.id.contains(destination_node)
                    || pair.qubit_a.id.contains(destination_node)
                {
                    return Ok(pair_id.clone());
                }
            }
        }

        Err(anyhow!(
            "No available entangled pairs for destination: {}",
            destination_node
        ))
    }

    /// Perform Bell measurement
    async fn perform_bell_measurement(
        &self,
        source_qubit_id: &str,
        entangled_pair_id: &str,
    ) -> Result<Vec<u8>> {
        // Simplified Bell measurement simulation
        let mut classical_bits = Vec::new();

        // Get source qubit state
        let qubits = self.qubits.read().await;
        let source_qubit = qubits
            .get(source_qubit_id)
            .ok_or_else(|| anyhow!("Source qubit not found: {}", source_qubit_id))?;

        // Simulate measurement outcomes based on quantum state
        let prob_00 = source_qubit.state.alpha.real.powi(2) + source_qubit.state.alpha.imag.powi(2);
        let random_value = rand::thread_rng().gen::<f64>();

        if random_value < prob_00 {
            classical_bits.push(0);
            classical_bits.push(0);
        } else if random_value < prob_00 + 0.25 {
            classical_bits.push(0);
            classical_bits.push(1);
        } else if random_value < prob_00 + 0.5 {
            classical_bits.push(1);
            classical_bits.push(0);
        } else {
            classical_bits.push(1);
            classical_bits.push(1);
        }

        debug!("Bell measurement result: {:?}", classical_bits);
        Ok(classical_bits)
    }

    /// Calculate teleportation fidelity
    async fn calculate_teleportation_fidelity(&self, classical_bits: &[u8]) -> f64 {
        // Simplified fidelity calculation
        let base_fidelity = 0.95; // Ideal case
        let error_rate = classical_bits.iter().map(|&b| b as f64).sum::<f64>() * 0.02; // Error per bit

        (base_fidelity - error_rate).max(0.0).min(1.0)
    }

    /// Perform quantum error correction
    pub async fn perform_error_correction(&self, logical_qubit_id: &str) -> Result<()> {
        let correction_id = uuid::Uuid::new_v4().to_string();

        // Create error correction instance
        let error_correction = QuantumErrorCorrection {
            code_type: ErrorCorrectionCode::SteaneCode,
            logical_qubits: 1,
            physical_qubits: 7,
            threshold_error_rate: 0.01,
            correction_rounds: 1,
            syndrome_measurements: Vec::new(),
        };

        // Perform syndrome measurement
        let syndrome = self.measure_syndrome(logical_qubit_id).await?;

        // Apply correction if needed
        if !syndrome.detected_errors.is_empty() {
            self.apply_quantum_correction(logical_qubit_id, &syndrome.detected_errors)
                .await?;
        }

        // Store error correction data
        self.error_correction
            .write()
            .await
            .insert(correction_id, error_correction);

        // Update metrics
        self.performance_metrics
            .write()
            .await
            .total_error_corrections += 1;

        debug!(
            "Quantum error correction performed for {}",
            logical_qubit_id
        );
        Ok(())
    }

    /// Measure error syndrome
    async fn measure_syndrome(&self, qubit_id: &str) -> Result<SyndromeMeasurement> {
        // Simplified syndrome measurement
        let syndrome = SyndromeMeasurement {
            timestamp: Utc::now(),
            stabilizer_generators: vec!["X1X2X3".to_string(), "Z1Z2Z3".to_string()],
            syndrome_bits: vec![0, 1], // Example syndrome
            detected_errors: vec![ErrorType::BitFlip],
            correction_applied: false,
        };

        Ok(syndrome)
    }

    /// Apply quantum error correction
    async fn apply_quantum_correction(&self, qubit_id: &str, errors: &[ErrorType]) -> Result<()> {
        let mut qubits = self.qubits.write().await;
        if let Some(qubit) = qubits.get_mut(qubit_id) {
            for error in errors {
                match error {
                    ErrorType::BitFlip => {
                        // Apply Pauli X correction
                        std::mem::swap(&mut qubit.state.alpha, &mut qubit.state.beta);
                        qubit.last_operation = Some(QuantumOperation::PauliX);
                    }
                    ErrorType::PhaseFlip => {
                        // Apply Pauli Z correction
                        qubit.state.beta.real = -qubit.state.beta.real;
                        qubit.state.beta.imag = -qubit.state.beta.imag;
                        qubit.last_operation = Some(QuantumOperation::PauliZ);
                    }
                    _ => {
                        warn!("Unsupported error type for correction: {:?}", error);
                    }
                }
            }
        }

        debug!("Applied quantum correction for errors: {:?}", errors);
        Ok(())
    }

    /// Establish quantum channel
    pub async fn establish_quantum_channel(
        &self,
        source: &str,
        destination: &str,
    ) -> Result<String> {
        let channel_id = uuid::Uuid::new_v4().to_string();

        // Create entangled pairs for the channel
        let entangled_pair_id = self.create_entangled_pair(source, destination).await?;

        let channel = QuantumChannel {
            channel_id: channel_id.clone(),
            source_node: source.to_string(),
            destination_node: destination.to_string(),
            entangled_pairs: vec![entangled_pair_id],
            channel_fidelity: 0.95,
            transmission_rate_qubits_per_sec: 1000.0,
            error_rate: 0.01,
            channel_capacity: 1.0, // 1 qubit per use
            quantum_protocol: QuantumSecurityProtocol::BB84,
            classical_channel: Some(format!("classical_{}", channel_id)),
        };

        self.quantum_channels
            .write()
            .await
            .insert(channel_id.clone(), channel);

        info!(
            "Established quantum channel {} between {} and {}",
            channel_id, source, destination
        );
        Ok(channel_id)
    }

    /// Send quantum-encrypted event
    pub async fn send_quantum_encrypted_event(
        &self,
        event: &StreamEvent,
        channel_id: &str,
    ) -> Result<Vec<u8>> {
        let channels = self.quantum_channels.read().await;
        let channel = channels
            .get(channel_id)
            .ok_or_else(|| anyhow!("Quantum channel not found: {}", channel_id))?;

        // Serialize event
        let event_data = serde_json::to_vec(event)?;

        // Quantum encrypt using BB84 protocol (simplified)
        let encrypted_data = self.bb84_encrypt(&event_data, channel).await?;

        debug!(
            "Quantum encrypted event {} bytes -> {} bytes",
            event_data.len(),
            encrypted_data.len()
        );
        Ok(encrypted_data)
    }

    /// BB84 quantum key distribution and encryption
    async fn bb84_encrypt(&self, data: &[u8], channel: &QuantumChannel) -> Result<Vec<u8>> {
        // Simplified BB84 implementation
        let mut encrypted = Vec::new();

        for &byte in data {
            // Generate random basis and bit
            let basis = if rand::thread_rng().gen::<bool>() {
                MeasurementBasis::Computational
            } else {
                MeasurementBasis::Diagonal
            };
            let key_bit = rand::thread_rng().gen::<u8>() & 1;

            // XOR encrypt with quantum key
            let encrypted_byte = byte ^ key_bit;
            encrypted.push(encrypted_byte);
        }

        Ok(encrypted)
    }

    /// Get quantum system metrics
    pub async fn get_quantum_metrics(&self) -> QuantumMetrics {
        self.performance_metrics.read().await.clone()
    }

    /// Monitor quantum decoherence
    pub async fn monitor_decoherence(&self) -> Result<Vec<String>> {
        let mut decoherent_qubits = Vec::new();
        let mut qubits = self.qubits.write().await;
        let current_time = Utc::now();

        for (qubit_id, qubit) in qubits.iter_mut() {
            let elapsed_ms = current_time
                .signed_duration_since(qubit.created_at)
                .num_milliseconds() as u64;

            if elapsed_ms > qubit.coherence_time_remaining_ms {
                // Qubit has decoherent
                qubit.state.purity *= 0.5; // Reduce purity
                qubit.state.fidelity *= 0.7; // Reduce fidelity
                decoherent_qubits.push(qubit_id.clone());

                self.performance_metrics.write().await.decoherence_events += 1;
            } else {
                // Update remaining coherence time
                qubit.coherence_time_remaining_ms =
                    qubit.coherence_time_remaining_ms.saturating_sub(elapsed_ms);
            }
        }

        if !decoherent_qubits.is_empty() {
            warn!("Detected decoherence in {} qubits", decoherent_qubits.len());
        }

        Ok(decoherent_qubits)
    }

    /// Cleanup decoherent qubits and pairs
    pub async fn cleanup_decoherent_resources(&self) -> Result<usize> {
        let decoherent_qubits = self.monitor_decoherence().await?;
        let mut cleanup_count = 0;

        // Remove decoherent qubits
        let mut qubits = self.qubits.write().await;
        for qubit_id in &decoherent_qubits {
            qubits.remove(qubit_id);
            cleanup_count += 1;
        }

        // Remove entangled pairs with decoherent qubits
        let mut pairs = self.entangled_pairs.write().await;
        let mut pairs_to_remove = Vec::new();

        for (pair_id, pair) in pairs.iter() {
            if decoherent_qubits.contains(&pair.qubit_a.id)
                || decoherent_qubits.contains(&pair.qubit_b.id)
            {
                pairs_to_remove.push(pair_id.clone());
            }
        }

        for pair_id in pairs_to_remove {
            pairs.remove(&pair_id);
            cleanup_count += 1;
        }

        info!("Cleaned up {} decoherent quantum resources", cleanup_count);
        Ok(cleanup_count)
    }
}

impl Default for QuantumCommConfig {
    fn default() -> Self {
        Self {
            max_entangled_pairs: 100,
            decoherence_timeout_ms: 10000, // 10 seconds
            error_correction_threshold: 0.01,
            enable_quantum_teleportation: true,
            enable_superdense_coding: true,
            quantum_network_topology: NetworkTopology::AdaptiveHybrid,
            security_protocols: vec![QuantumSecurityProtocol::BB84],
            entanglement_distribution: EntanglementDistribution::DirectTransmission,
        }
    }
}

impl Complex64 {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quantum_comm_system_creation() {
        let config = QuantumCommConfig::default();
        let system = QuantumCommSystem::new(config);

        let metrics = system.get_quantum_metrics().await;
        assert_eq!(metrics.total_qubits_created, 0);
    }

    #[tokio::test]
    async fn test_entangled_pair_creation() {
        let config = QuantumCommConfig::default();
        let system = QuantumCommSystem::new(config);

        let pair_id = system
            .create_entangled_pair("node_a", "node_b")
            .await
            .unwrap();
        assert!(!pair_id.is_empty());

        let metrics = system.get_quantum_metrics().await;
        assert_eq!(metrics.total_qubits_created, 2);
        assert_eq!(metrics.total_entanglements, 1);
    }

    #[tokio::test]
    async fn test_quantum_channel_establishment() {
        let config = QuantumCommConfig::default();
        let system = QuantumCommSystem::new(config);

        let channel_id = system
            .establish_quantum_channel("source", "destination")
            .await
            .unwrap();
        assert!(!channel_id.is_empty());
    }

    #[test]
    fn test_complex_number_operations() {
        let c = Complex64::new(3.0, 4.0);
        assert_eq!(c.magnitude_squared(), 25.0);
    }

    #[test]
    fn test_quantum_state_normalization() {
        let state = QuantumState {
            alpha: Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            beta: Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            phase: 0.0,
            purity: 1.0,
            fidelity: 1.0,
        };

        let norm_squared = state.alpha.magnitude_squared() + state.beta.magnitude_squared();
        assert!((norm_squared - 1.0).abs() < 1e-10);
    }
}
