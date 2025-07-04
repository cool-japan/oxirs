//! Quantum machine learning engine

use super::QuantumConfig;

/// Quantum machine learning engine
pub struct QuantumMLEngine {
    config: QuantumConfig,
    ml_algorithms: Vec<QuantumMLAlgorithm>,
}

impl QuantumMLEngine {
    pub fn new(config: QuantumConfig) -> Self {
        Self {
            config,
            ml_algorithms: vec![
                QuantumMLAlgorithm::QNN,
                QuantumMLAlgorithm::QSVM,
                QuantumMLAlgorithm::QPCA,
            ],
        }
    }
}

/// Quantum ML algorithms
#[derive(Debug, Clone)]
pub enum QuantumMLAlgorithm {
    QNN,
    QSVM,
    QPCA,
    QuantumBoltzmannMachine,
    QuantumAutoencoder,
    QuantumGAN,
}
