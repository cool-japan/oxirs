//! # Evolutionary Neural Architecture System
//!
//! This module implements self-designing neural networks using evolutionary algorithms,
//! neural architecture search (NAS), and genetic programming to automatically discover
//! optimal neural network architectures for SHACL validation tasks.

pub mod core;
pub mod search_engine;
pub mod genetic_programming;
pub mod population;
pub mod performance;
pub mod optimization;
pub mod self_modification;
pub mod types;

// Re-export main types
pub use core::EvolutionaryNeuralArchitecture;
pub use types::*;
pub use search_engine::NeuralArchitectureSearchEngine;
pub use genetic_programming::GeneticProgrammingSystem;
pub use population::{ArchitecturePopulationManager, EvolutionStrategyCoordinator};
pub use performance::ArchitecturePerformanceEvaluator;
pub use optimization::MultiObjectiveOptimizer;
pub use self_modification::SelfModificationEngine;

use crate::{Result, ShaclAiError};