//! Quantum error correction

use super::{ErrorCorrectionCode, QuantumConfig};

/// Quantum error correction system
pub struct QuantumErrorCorrection {
    config: QuantumConfig,
    correction_codes: Vec<ErrorCorrectionCode>,
}

impl QuantumErrorCorrection {
    pub fn new(config: QuantumConfig) -> Self {
        Self {
            config: config.clone(),
            correction_codes: vec![config.error_correction_code],
        }
    }
}

/// Error correction metrics
#[derive(Debug, Default)]
pub struct ErrorCorrectionMetrics {
    pub error_rate: f64,
    pub correction_success_rate: f64,
    pub syndrome_detection_time_us: f64,
    pub correction_latency_us: f64,
}
