//! Photonic Computing Engine for OxiRS SHACL-AI
//!
//! This module implements photonic computation capabilities using light-based quantum
//! processing for ultra-fast validation with the speed of light computation and
//! infinite parallel processing through optical interference patterns.

pub mod types;
pub mod engine;

// Re-export main types for easy access
pub use types::{
    WavelengthRange, PhotonicGate, GateType, OpticalProcessingState, ConnectionType,
    PhotonicQubit, PolarizationState, PhotonNumberState, SpatialMode, CoherenceProperties,
    MaterialType, NonlinearProperties, JunctionType, NetworkTopology
};
pub use engine::{
    PhotonicComputingEngine, OpticalProcessingUnit, PhotonicQuantumCircuit,
    InterferenceProcessor, OpticalMemoryBank, PhotonicEntanglementNetwork,
    LightSpeedComputationManager, InterferencePattern, PatternType, ValidationResult
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photonic_engine_creation() {
        let engine = PhotonicComputingEngine::new();
        assert_eq!(engine.light_speed_manager.speed_of_light, 299_792_458.0);
    }

    #[test]
    fn test_entangled_pair_creation() {
        let mut engine = PhotonicComputingEngine::new();
        let result = engine.create_entangled_pair();
        assert!(result.is_ok());
        
        let (qubit1, qubit2) = result.unwrap();
        assert_ne!(qubit1.id, qubit2.id);
        assert_eq!(qubit1.frequency, qubit2.frequency);
    }

    #[test]
    fn test_optical_unit_addition() {
        let engine = PhotonicComputingEngine::new();
        
        let unit = OpticalProcessingUnit {
            id: "test_unit".to_string(),
            wavelength_range: WavelengthRange {
                min_wavelength: 800.0,
                max_wavelength: 1600.0,
            },
            power_level: 1.0,
            coherence_length: 1e-3,
            available_gates: Vec::new(),
            processing_state: OpticalProcessingState::Idle,
            quantum_efficiency: 0.9,
        };

        let result = engine.add_optical_unit(unit);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_processing() {
        let engine = PhotonicComputingEngine::new();
        let validation_data = b"test validation data";
        
        let result = engine.process_shacl_validation(validation_data);
        assert!(result.is_ok());
        
        let validation_result = result.unwrap();
        assert!(validation_result.success);
        assert!(validation_result.processing_time_fs > 0);
        assert!(validation_result.photons_used > 0);
    }
}