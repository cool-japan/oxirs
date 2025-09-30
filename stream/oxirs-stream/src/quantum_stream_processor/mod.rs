//! # Quantum-Enhanced Stream Processing Module
//!
//! Ultra-advanced quantum computing integration for RDF stream processing with
//! quantum optimization algorithms, quantum machine learning, and quantum-classical
//! hybrid processing for next-generation semantic web applications.

pub mod types;
pub mod gates;
pub mod classical;
pub mod optimization;
pub mod ml;
pub mod entanglement;
pub mod error_correction;
pub mod monitoring;
pub mod processor;

// Re-export main types
pub use types::*;
pub use processor::QuantumStreamProcessor;