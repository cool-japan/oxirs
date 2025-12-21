//! Learned vector indexes using neural networks
//!
//! This module implements learned index structures that use neural networks
//! to learn data distributions and provide faster lookups than traditional
//! index structures.
//!
//! ## Key Concepts
//! - **Recursive Model Index (RMI)**: Hierarchy of models for indexing
//! - **Learned CDF**: Neural networks learn cumulative distribution function
//! - **Error Bounds**: Track prediction errors for correctness guarantees
//! - **Hybrid Approach**: Combine learned models with traditional search
//!
//! ## References
//! - "The Case for Learned Index Structures" (Kraska et al., 2018)
//! - "Learning to Hash for Indexing Big Data" (Wang et al., 2016)

pub mod config;
pub mod neural_index;
pub mod rmi;
pub mod training;
pub mod types;

pub use config::{LearnedIndexConfig, ModelArchitecture, TrainingConfig};
pub use neural_index::NeuralVectorIndex;
pub use rmi::{RecursiveModelIndex, RmiStage};
pub use training::{IndexTrainer, TrainingStats};
pub use types::{LearnedIndexError, LearnedIndexResult, PredictionBounds};
