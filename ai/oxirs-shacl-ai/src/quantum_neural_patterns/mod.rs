//! Quantum-Inspired Neural Pattern Recognition for SHACL-AI
//!
//! This module implements quantum-inspired algorithms for pattern recognition
//! in RDF data, leveraging quantum superposition and entanglement concepts
//! to enhance shape learning and validation optimization.

pub mod core;
pub mod teleportation;
pub mod annealing;
pub mod fourier;
pub mod eigensolver;
pub mod supremacy;
pub mod recognizer;
pub mod utils;

// Re-export main types for convenience
pub use core::{QuantumState, QuantumPattern, QuantumMetrics};
pub use teleportation::QuantumTeleportation;
pub use annealing::QuantumAnnealer;
pub use fourier::QuantumFourierTransform;
pub use eigensolver::VariationalQuantumEigensolver;
pub use supremacy::QuantumSupremacyDetector;
pub use recognizer::QuantumNeuralPatternRecognizer;
pub use utils::*;

use crate::{Result, ShaclAiError};