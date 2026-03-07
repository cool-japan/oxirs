//! Graph embedding algorithms for GraphRAG.
//!
//! This module provides standalone graph embedding implementations:
//!
//! - [`node2vec`]: Biased random walk-based Node2Vec embeddings with full
//!   second-order Markov transition probabilities, alias-sampling for O(1)
//!   per-step random walks, and a simplified skip-gram training loop.

pub mod node2vec;

pub use node2vec::{Node2VecConfig, Node2VecEmbedder, Node2VecEmbeddings, Node2VecWalkConfig};
