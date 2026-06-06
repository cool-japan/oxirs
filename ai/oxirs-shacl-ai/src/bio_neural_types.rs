//! # Biological Neural Integration — Type Definitions
//!
//! All configuration structs, context types, type aliases, and result types
//! used across the biological neural integration sub-system.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use oxirs_shacl::ValidationReport;

// ── Type aliases ──────────────────────────────────────────────────────────────

pub type CultureId = Uuid;
pub type OrganoidId = Uuid;

// ── Configuration types ───────────────────────────────────────────────────────

/// Configuration for biological neural integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalIntegrationConfig {
    /// Number of target cell culture clusters
    pub target_culture_count: usize,
    /// Number of neurons per culture cluster
    pub neurons_per_culture: usize,
    /// Number of neural organoids to initialize
    pub target_organoid_count: usize,
    /// Cell culture configuration
    pub culture_configuration: CellCultureConfig,
    /// Neural organoid configuration
    pub organoid_configuration: OrganoidConfig,
    /// Bio-electrical signal processing settings
    pub signal_processing_config: SignalProcessingConfig,
    /// Synaptic plasticity parameters
    pub plasticity_config: PlasticityConfig,
    /// Neurotransmitter processing settings
    pub neurotransmitter_config: NeurotransmitterConfig,
}

impl Default for BiologicalIntegrationConfig {
    fn default() -> Self {
        Self {
            target_culture_count: 50,
            neurons_per_culture: 10000,
            target_organoid_count: 10,
            culture_configuration: CellCultureConfig::default(),
            organoid_configuration: OrganoidConfig::default(),
            signal_processing_config: SignalProcessingConfig::default(),
            plasticity_config: PlasticityConfig::default(),
            neurotransmitter_config: NeurotransmitterConfig::default(),
        }
    }
}

/// Cell culture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellCultureConfig {
    /// Culture medium composition
    pub medium_composition: String,
    /// Temperature in Celsius
    pub temperature_celsius: f64,
    /// CO2 concentration percentage
    pub co2_concentration: f64,
    /// pH level
    pub ph_level: f64,
    /// Culture maturation time in days
    pub maturation_days: u32,
}

impl Default for CellCultureConfig {
    fn default() -> Self {
        Self {
            medium_composition: "Neurobasal medium with B27 supplement".to_string(),
            temperature_celsius: 37.0,
            co2_concentration: 5.0,
            ph_level: 7.4,
            maturation_days: 21,
        }
    }
}

/// Neural organoid configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganoidConfig {
    /// Organoid development protocol
    pub development_protocol: String,
    /// Target organoid size in micrometers
    pub target_size_micrometers: f64,
    /// Neural complexity level
    pub neural_complexity_level: u32,
    /// Vascularization enabled
    pub vascularization_enabled: bool,
}

impl Default for OrganoidConfig {
    fn default() -> Self {
        Self {
            development_protocol: "Cerebral organoid protocol with guided differentiation"
                .to_string(),
            target_size_micrometers: 4000.0, // 4mm diameter
            neural_complexity_level: 8,      // High complexity
            vascularization_enabled: true,
        }
    }
}

/// Signal processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalProcessingConfig {
    /// Sampling rate in Hz
    pub sampling_rate_hz: f64,
    /// Signal filtering frequency range
    pub filter_range_hz: (f64, f64),
    /// Noise reduction threshold
    pub noise_threshold: f64,
    /// Action potential detection sensitivity
    pub action_potential_sensitivity: f64,
}

impl Default for SignalProcessingConfig {
    fn default() -> Self {
        Self {
            sampling_rate_hz: 30000.0,        // 30 kHz sampling
            filter_range_hz: (300.0, 3000.0), // Typical neural signal range
            noise_threshold: 50.0,            // 50 µV noise threshold
            action_potential_sensitivity: 0.95,
        }
    }
}

/// Synaptic plasticity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityConfig {
    /// LTP induction threshold
    pub ltp_threshold: f64,
    /// LTD induction threshold
    pub ltd_threshold: f64,
    /// STDP time window in milliseconds
    pub stdp_time_window_ms: f64,
    /// Homeostatic scaling factor
    pub homeostatic_scaling_factor: f64,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            ltp_threshold: 0.8,        // 80% activation threshold for LTP
            ltd_threshold: 0.3,        // 30% activation threshold for LTD
            stdp_time_window_ms: 20.0, // 20ms STDP window
            homeostatic_scaling_factor: 0.1,
        }
    }
}

/// Neurotransmitter processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeurotransmitterConfig {
    /// Dopamine detection sensitivity
    pub dopamine_sensitivity: f64,
    /// Serotonin monitoring precision
    pub serotonin_precision: f64,
    /// Acetylcholine analysis accuracy
    pub acetylcholine_accuracy: f64,
    /// GABA inhibition measurement sensitivity
    pub gaba_sensitivity: f64,
}

impl Default for NeurotransmitterConfig {
    fn default() -> Self {
        Self {
            dopamine_sensitivity: 0.95,
            serotonin_precision: 0.92,
            acetylcholine_accuracy: 0.88,
            gaba_sensitivity: 0.90,
        }
    }
}

// ── Context types ─────────────────────────────────────────────────────────────

/// Biological validation context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalValidationContext {
    /// Cell culture conditions
    pub culture_conditions: CellCultureConditions,
    /// Neural stimulation parameters
    pub stimulation_parameters: NeuralStimulationParameters,
    /// Biological validation mode
    pub validation_mode: BiologicalValidationMode,
    /// Energy efficiency requirements
    pub energy_efficiency_requirements: EnergyEfficiencyRequirements,
}

/// Cell culture conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellCultureConditions {
    /// Current temperature
    pub temperature: f64,
    /// Current pH
    pub ph: f64,
    /// Oxygen concentration
    pub oxygen_concentration: f64,
    /// Nutrient availability
    pub nutrient_availability: f64,
}

/// Neural stimulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralStimulationParameters {
    /// Stimulation frequency in Hz
    pub frequency_hz: f64,
    /// Stimulation amplitude in mV
    pub amplitude_mv: f64,
    /// Pulse duration in microseconds
    pub pulse_duration_us: f64,
    /// Stimulation pattern type
    pub pattern_type: StimulationPattern,
}

/// Stimulation pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StimulationPattern {
    Continuous,
    Burst,
    Theta,
    Gamma,
    Delta,
    Random,
}

/// Biological validation modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiologicalValidationMode {
    /// High-precision low-throughput mode
    HighPrecision,
    /// Balanced precision and throughput
    Balanced,
    /// High-throughput low-precision mode
    HighThroughput,
    /// Adaptive mode based on validation complexity
    Adaptive,
}

/// Energy efficiency requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyEfficiencyRequirements {
    /// Maximum ATP consumption per validation
    pub max_atp_per_validation: f64,
    /// Energy efficiency target (0.0 to 1.0)
    pub efficiency_target: f64,
    /// Metabolic cost constraints
    pub metabolic_cost_constraints: bool,
}

// ── Public result types ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalInitResult {
    pub biological_interfaces_active: bool,
    pub cell_cultures_established: usize,
    pub neural_organoids_initialized: usize,
    pub signal_processing_calibrated: bool,
    pub synaptic_plasticity_active: usize,
    pub neurotransmitter_processing_online: bool,
    pub bio_hybrid_coordination_established: bool,
    pub total_biological_processing_capacity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalValidationResult {
    pub bio_context: BiologicalValidationContext,
    pub biological_processing_efficiency: f64,
    pub bio_electrical_signal_quality: f64,
    pub cell_culture_validation_accuracy: f64,
    pub neural_organoid_reasoning_depth: f64,
    pub synaptic_plasticity_adaptation: f64,
    pub neurotransmitter_signal_strength: f64,
    pub bio_hybrid_decision_confidence: f64,
    pub living_neural_network_coherence: f64,
    pub biological_energy_consumption_atp: f64,
    pub processing_time_biological_seconds: f64,
    pub overall_validation_report: ValidationReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalStatistics {
    pub total_biological_validations: u64,
    pub total_culture_clusters: usize,
    pub total_neural_organoids: usize,
    pub average_processing_time_seconds: f64,
    pub average_processing_efficiency: f64,
    pub total_atp_consumption: f64,
    pub biological_neural_interface_uptime: f64,
    pub synaptic_plasticity_adaptation_rate: f64,
    pub neurotransmitter_signal_quality: f64,
    pub bio_hybrid_coordination_efficiency: f64,
}

// ── Private intermediate result / data types ──────────────────────────────────
// These are crate-internal, simplified-implementation placeholders. Many fields
// are set but read only selectively — that matches the original single-file
// behaviour where private structs don't trigger dead_code warnings. We
// suppress the lint uniformly for this entire section via a submodule.
pub(crate) mod inner {
    #![allow(dead_code)]

    use oxirs_shacl::ValidationReport;

    #[derive(Debug)]
    pub(crate) struct NeuronInterfaceInit {
        pub interfaces_active: bool,
        pub microelectrode_arrays: usize,
        pub patch_clamp_channels: usize,
        pub calcium_imaging_resolution: usize,
        pub optogenetic_control_precision: f64,
    }

    #[derive(Debug)]
    pub(crate) struct CellCultureSetup {
        pub cultures_established: usize,
        pub total_neurons: usize,
        pub culture_viability: f64,
    }

    #[derive(Debug)]
    pub(crate) struct NeuralOrganoidInit {
        pub organoids_initialized: usize,
        pub total_processing_capacity: f64,
        pub neural_network_complexity: f64,
    }

    #[derive(Debug)]
    pub(crate) struct SignalProcessingCalibration {
        pub calibration_accuracy: f64,
        pub action_potential_detection_sensitivity: f64,
        pub synaptic_potential_resolution: f64,
        pub oscillation_analysis_precision: f64,
    }

    #[derive(Debug)]
    pub(crate) struct PlasticityManagementInit {
        pub plasticity_mechanisms_active: usize,
        pub ltp_protocol_efficiency: f64,
        pub ltd_mechanism_precision: f64,
        pub stdp_timing_accuracy: f64,
        pub homeostatic_stability: f64,
    }

    #[derive(Debug)]
    pub(crate) struct NeurotransmitterSetup {
        pub processing_online: bool,
        pub neurotransmitter_types_detected: usize,
        pub release_monitoring_precision: f64,
        pub transmission_analysis_accuracy: f64,
    }

    #[derive(Debug)]
    pub(crate) struct BioHybridCoordinationInit {
        pub coordination_established: bool,
        pub bio_artificial_interfaces: usize,
        pub hybrid_decision_systems: usize,
        pub communication_protocol_efficiency: f64,
    }

    #[derive(Debug)]
    pub(crate) struct BiologicalProcessingResults {
        pub bio_patterns: BiologicalPatterns,
        pub stimulation_efficiency: f64,
        pub neural_response_quality: f64,
        pub biological_processing_speed: f64,
    }

    #[derive(Debug)]
    pub(crate) struct BioElectricalAnalysis {
        pub action_potential_patterns: ActionPotentialPatterns,
        pub synaptic_transmission_quality: f64,
        pub neural_oscillation_coherence: f64,
        pub signal_to_noise_ratio: f64,
    }

    #[derive(Debug)]
    pub(crate) struct CellCultureResults {
        pub cluster_results: Vec<CultureClusterResult>,
        pub overall_culture_accuracy: f64,
        pub culture_network_coherence: f64,
    }

    #[derive(Debug)]
    pub(crate) struct NeuralOrganoidResults {
        pub organoid_results: Vec<OrganoidReasoningResult>,
        pub overall_reasoning_depth: f64,
        pub organoid_network_integration: f64,
    }

    #[derive(Debug)]
    pub(crate) struct SynapticPlasticityResults {
        pub ltp_adaptation_strength: f64,
        pub ltd_adjustment_precision: f64,
        pub stdp_learning_efficiency: f64,
        pub homeostatic_stability: f64,
        pub overall_plasticity_adaptation: f64,
    }

    #[derive(Debug)]
    pub(crate) struct NeurotransmitterValidationResults {
        pub dopamine_reward_signal: f64,
        pub serotonin_confidence_level: f64,
        pub acetylcholine_attention_focus: f64,
        pub gaba_inhibitory_control: f64,
        pub overall_neurotransmitter_strength: f64,
    }

    #[derive(Debug)]
    pub(crate) struct BioHybridDecisionResults {
        pub bio_artificial_integration_quality: f64,
        pub hybrid_decision_confidence: f64,
        pub collaboration_efficiency: f64,
        pub bio_hybrid_coherence: f64,
    }

    #[derive(Debug)]
    pub(crate) struct AggregatedBiologicalResults {
        pub processing_efficiency: f64,
        pub signal_quality: f64,
        pub culture_accuracy: f64,
        pub organoid_reasoning_depth: f64,
        pub plasticity_adaptation: f64,
        pub neurotransmitter_strength: f64,
        pub bio_hybrid_confidence: f64,
        pub living_network_coherence: f64,
        pub energy_consumption_atp: f64,
        pub validation_report: ValidationReport,
    }

    #[derive(Debug)]
    pub(crate) struct BiologicalValidationInputs {
        pub biological_processing: BiologicalProcessingResults,
        pub signal_analysis: BioElectricalAnalysis,
        pub culture_results: CellCultureResults,
        pub organoid_results: NeuralOrganoidResults,
        pub plasticity_results: SynapticPlasticityResults,
        pub neurotransmitter_results: NeurotransmitterValidationResults,
        pub bio_hybrid_results: BioHybridDecisionResults,
    }

    // ── Fine-grained helper result types (component-internal) ─────────────────

    #[derive(Debug)]
    pub(crate) struct MicroelectrodeArraySetup {
        pub arrays_active: bool,
        pub array_count: usize,
    }
    #[derive(Debug)]
    pub(crate) struct PatchClampInit {
        pub channel_count: usize,
    }
    #[derive(Debug)]
    pub(crate) struct CalciumImagingSetup {
        pub resolution: usize,
    }
    #[derive(Debug)]
    pub(crate) struct OptogeneticInit {
        pub precision: f64,
    }
    #[derive(Debug)]
    pub(crate) struct BiologicalPatterns {
        pub pattern_count: usize,
        pub complexity_level: f64,
    }
    #[derive(Debug)]
    pub(crate) struct NeuralStimulationResults {
        pub efficiency: f64,
    }
    #[derive(Debug)]
    pub(crate) struct NeuralResponseRecording {
        pub quality: f64,
        pub processing_speed: f64,
    }
    #[derive(Debug)]
    pub(crate) struct CultureEstablishmentResult {
        pub success: bool,
    }
    #[derive(Debug)]
    pub(crate) struct CultureClusterResult {
        pub validation_accuracy: f64,
    }
    #[derive(Debug)]
    pub(crate) struct OrganoidInitResult {
        pub success: bool,
    }
    #[derive(Debug)]
    pub(crate) struct OrganoidReasoningResult {
        pub reasoning_complexity: f64,
    }
    #[derive(Debug)]
    pub(crate) struct ActionPotentialCalibration {
        pub accuracy: f64,
        pub sensitivity: f64,
    }
    #[derive(Debug)]
    pub(crate) struct SynapticPotentialSetup {
        pub resolution: f64,
    }
    #[derive(Debug)]
    pub(crate) struct OscillationAnalysisInit {
        pub precision: f64,
    }
    #[derive(Debug)]
    pub(crate) struct ActionPotentialPatterns {
        pub pattern_count: usize,
        pub coherence: f64,
    }
    #[derive(Debug)]
    pub(crate) struct SynapticTransmissionAnalysis {
        pub transmission_quality: f64,
    }
    #[derive(Debug)]
    pub(crate) struct NeuralOscillationAnalysis {
        pub coherence: f64,
    }
    #[derive(Debug)]
    pub(crate) struct LTPSetup {
        pub efficiency: f64,
    }
    #[derive(Debug)]
    pub(crate) struct LTDInit {
        pub precision: f64,
    }
    #[derive(Debug)]
    pub(crate) struct STDPSetup {
        pub timing_accuracy: f64,
    }
    #[derive(Debug)]
    pub(crate) struct HomeostaticInit {
        pub stability: f64,
    }
    #[derive(Debug)]
    pub(crate) struct LTPResults {
        pub adaptation_strength: f64,
    }
    #[derive(Debug)]
    pub(crate) struct LTDResults {
        pub adjustment_precision: f64,
    }
    #[derive(Debug)]
    pub(crate) struct STDPResults {
        pub learning_efficiency: f64,
    }
    #[derive(Debug)]
    pub(crate) struct HomeostaticResults {
        pub stability: f64,
    }
    #[derive(Debug)]
    pub(crate) struct NeurotransmitterDetectionInit {
        pub types_detected: usize,
    }
    #[derive(Debug)]
    pub(crate) struct ReleaseMonitoringSetup {
        pub precision: f64,
    }
    #[derive(Debug)]
    pub(crate) struct TransmissionAnalysisInit {
        pub accuracy: f64,
    }
    #[derive(Debug)]
    pub(crate) struct DopamineProcessing {
        pub reward_strength: f64,
    }
    #[derive(Debug)]
    pub(crate) struct SerotoninAnalysis {
        pub confidence_level: f64,
    }
    #[derive(Debug)]
    pub(crate) struct AcetylcholineProcessing {
        pub attention_strength: f64,
    }
    #[derive(Debug)]
    pub(crate) struct GABAAnalysis {
        pub inhibition_strength: f64,
    }
    #[derive(Debug)]
    pub(crate) struct BioArtificialInterfaceSetup {
        pub interface_count: usize,
    }
    #[derive(Debug)]
    pub(crate) struct HybridDecisionSystemInit {
        pub system_count: usize,
    }
    #[derive(Debug)]
    pub(crate) struct CommunicationProtocolSetup {
        pub efficiency: f64,
    }
    #[derive(Debug)]
    pub(crate) struct IntegrationResults {
        pub integration_quality: f64,
    }
    #[derive(Debug)]
    pub(crate) struct DecisionResults {
        pub decision_confidence: f64,
    }
    #[derive(Debug)]
    pub(crate) struct CollaborationOptimization {
        pub efficiency: f64,
    }
}

// Re-export everything from the inner module at crate level for easy import
pub(crate) use inner::{
    AcetylcholineProcessing, ActionPotentialCalibration, ActionPotentialPatterns,
    AggregatedBiologicalResults, BioArtificialInterfaceSetup, BioElectricalAnalysis,
    BioHybridCoordinationInit, BioHybridDecisionResults, BiologicalPatterns,
    BiologicalProcessingResults, BiologicalValidationInputs, CalciumImagingSetup,
    CellCultureResults, CellCultureSetup, CollaborationOptimization, CommunicationProtocolSetup,
    CultureClusterResult, CultureEstablishmentResult, DecisionResults, DopamineProcessing,
    GABAAnalysis, HomeostaticInit, HomeostaticResults, HybridDecisionSystemInit,
    IntegrationResults, LTDInit, LTDResults, LTPResults, LTPSetup, MicroelectrodeArraySetup,
    NeuralOrganoidInit, NeuralOrganoidResults, NeuralOscillationAnalysis, NeuralResponseRecording,
    NeuralStimulationResults, NeuronInterfaceInit, NeurotransmitterDetectionInit,
    NeurotransmitterSetup, NeurotransmitterValidationResults, OptogeneticInit, OrganoidInitResult,
    OrganoidReasoningResult, OscillationAnalysisInit, PatchClampInit, PlasticityManagementInit,
    ReleaseMonitoringSetup, STDPResults, STDPSetup, SerotoninAnalysis, SignalProcessingCalibration,
    SynapticPlasticityResults, SynapticPotentialSetup, SynapticTransmissionAnalysis,
    TransmissionAnalysisInit,
};
