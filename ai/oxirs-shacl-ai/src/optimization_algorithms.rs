//! Advanced optimization algorithms for SHACL shape validation.
//!
//! This module re-exports algorithm implementations split across sibling modules:
//! - `opt_algs_evolutionary`: GeneticOptimizer, MultiObjectiveOptimizer, NSGA-II,
//!   DEIndividual, TabuMove, RL/Adaptive supporting types
//! - `opt_algs_swarm`: SimulatedAnnealingOptimizer, ParticleSwarmOptimizer,
//!   BayesianOptimizer, GaussianProcess, search-space types
//! - `opt_algs_tests`: unit tests

pub use crate::opt_algs_evolutionary::*;
pub use crate::opt_algs_swarm::*;
