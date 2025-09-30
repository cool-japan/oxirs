//! Variational quantum algorithms processor

use super::QuantumConfig;

/// Variational processor for VQAs
pub struct VariationalProcessor {
    config: QuantumConfig,
    ansatz_types: Vec<VariationalAnsatz>,
}

impl VariationalProcessor {
    pub fn new(config: QuantumConfig) -> Self {
        Self {
            config,
            ansatz_types: vec![
                VariationalAnsatz::HardwareEfficient,
                VariationalAnsatz::UCCSD,
            ],
        }
    }
}

/// Variational ansatz types
#[derive(Debug, Clone)]
pub enum VariationalAnsatz {
    HardwareEfficient,
    UCCSD,
    RealAmplitudes,
    EfficientSU2,
    Custom,
}
