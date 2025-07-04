//! Quantum-Inspired Neural Pattern Recognition for SHACL-AI
//!
//! This module implements quantum-inspired algorithms for pattern recognition
//! in RDF data, leveraging quantum superposition and entanglement concepts
//! to enhance shape learning and validation optimization.

pub mod annealing;
pub mod core;
pub mod eigensolver;
pub mod fourier;
pub mod recognizer;
pub mod supremacy;
pub mod teleportation;
pub mod utils;

// Re-export main types for convenience
pub use annealing::QuantumAnnealer;
pub use core::{QuantumMetrics, QuantumPattern, QuantumState};
pub use eigensolver::VariationalQuantumEigensolver;
pub use fourier::QuantumFourierTransform;
pub use recognizer::QuantumNeuralPatternRecognizer;
pub use supremacy::QuantumSupremacyDetector;
pub use teleportation::QuantumTeleportation;
pub use utils::*;

use crate::{Result, ShaclAiError};
