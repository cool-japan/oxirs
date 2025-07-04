//! # Evolutionary Neural Architecture System
//!
//! This module implements self-designing neural networks using evolutionary algorithms,
//! neural architecture search (NAS), and genetic programming to automatically discover
//! optimal neural network architectures for SHACL validation tasks.

pub mod core;
pub mod genetic_programming;
pub mod optimization;
pub mod performance;
pub mod population;
pub mod search_engine;
pub mod self_modification;
pub mod types;

// Re-export main types
pub use core::EvolutionaryNeuralArchitecture;
pub use genetic_programming::GeneticProgrammingSystem;
pub use optimization::MultiObjectiveOptimizer;
pub use performance::ArchitecturePerformanceEvaluator;
pub use population::{ArchitecturePopulationManager, EvolutionStrategyCoordinator};
pub use search_engine::NeuralArchitectureSearchEngine;
pub use self_modification::SelfModificationEngine;
pub use types::*;

use crate::{Result, ShaclAiError};
