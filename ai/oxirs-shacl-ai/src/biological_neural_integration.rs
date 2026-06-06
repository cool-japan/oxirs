//! # Biological Neural Integration System
//!
//! This module implements direct interface capabilities with actual biological neurons
//! for SHACL validation, creating a hybrid bio-artificial intelligence system that
//! leverages the computational power and efficiency of living neural networks.
//!
//! ## Features
//! - Direct biological neuron interface protocols
//! - Bio-electrical signal interpretation for validation
//! - Living neural network validation clusters
//! - Synaptic plasticity-based constraint learning
//! - Neurotransmitter-based validation signaling
//! - Cell culture-based validation processing
//! - Bio-hybrid artificial intelligence systems
//! - Neural organoid integration for complex reasoning
//!
//! ## Module layout
//! - [`crate::bio_neural_types`]      — all config/context/result type definitions
//! - [`crate::bio_neural_core`]       — `BiologicalNeuralIntegrator` implementation
//! - [`crate::bio_neural_components`] — sub-component struct implementations

// Re-export the public API so callers can use `oxirs_shacl_ai::BiologicalNeuralIntegrator`
// and all associated public types without knowing the internal module layout.
pub use crate::bio_neural_core::BiologicalNeuralIntegrator;
pub use crate::bio_neural_types::{
    BiologicalInitResult, BiologicalIntegrationConfig, BiologicalStatistics,
    BiologicalValidationContext, BiologicalValidationMode, BiologicalValidationResult,
    CellCultureConditions, CellCultureConfig, CultureId, EnergyEfficiencyRequirements,
    NeuralStimulationParameters, NeurotransmitterConfig, OrganoidConfig, OrganoidId,
    PlasticityConfig, SignalProcessingConfig, StimulationPattern,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_biological_neural_integrator_creation() {
        let config = BiologicalIntegrationConfig::default();
        let integrator = BiologicalNeuralIntegrator::new(config);

        assert_eq!(integrator.culture_clusters.len(), 0);
        assert_eq!(integrator.organoid_processors.len(), 0);
    }

    #[tokio::test]
    async fn test_biological_system_initialization() {
        let config = BiologicalIntegrationConfig {
            target_culture_count: 5,
            target_organoid_count: 2,
            ..Default::default()
        };
        let integrator = BiologicalNeuralIntegrator::new(config);

        let result = integrator
            .initialize_biological_system()
            .await
            .expect("should succeed");

        assert!(result.biological_interfaces_active);
        assert!(result.signal_processing_calibrated);
        assert!(result.bio_hybrid_coordination_established);
    }

    #[tokio::test]
    async fn test_biological_validation_context() {
        let bio_context = BiologicalValidationContext {
            culture_conditions: CellCultureConditions {
                temperature: 37.0,
                ph: 7.4,
                oxygen_concentration: 0.21,
                nutrient_availability: 1.0,
            },
            stimulation_parameters: NeuralStimulationParameters {
                frequency_hz: 40.0,
                amplitude_mv: 100.0,
                pulse_duration_us: 200.0,
                pattern_type: StimulationPattern::Gamma,
            },
            validation_mode: BiologicalValidationMode::HighPrecision,
            energy_efficiency_requirements: EnergyEfficiencyRequirements {
                max_atp_per_validation: 1e12,
                efficiency_target: 0.9,
                metabolic_cost_constraints: true,
            },
        };

        assert_eq!(bio_context.culture_conditions.temperature, 37.0);
        assert!(matches!(
            bio_context.validation_mode,
            BiologicalValidationMode::HighPrecision
        ));
    }

    #[tokio::test]
    async fn test_biological_statistics() {
        let config = BiologicalIntegrationConfig::default();
        let integrator = BiologicalNeuralIntegrator::new(config);

        let stats = integrator
            .get_biological_statistics()
            .await
            .expect("should succeed");

        assert_eq!(stats.total_biological_validations, 0);
        assert_eq!(stats.total_culture_clusters, 0);
        assert_eq!(stats.total_neural_organoids, 0);
    }
}
