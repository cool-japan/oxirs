//! GraphSAGE encoder for knowledge-graph entity embeddings.
//!
//! This module implements **phase a** of the hybrid GNN+LLM architecture:
//! a two-layer GraphSAGE encoder with hand-rolled forward + backward passes
//! and an unsupervised link-prediction training objective.
//!
//! # Usage
//!
//! ```rust
//! use oxirs_graphrag::gnn_encoder::{GraphSageConfig, GraphSageEncoder, KgGraph};
//! use scirs2_core::ndarray_ext::Array2;
//!
//! let graph = KgGraph {
//!     num_nodes: 4,
//!     edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
//!     node_features: Array2::zeros((4, 8)),
//! };
//!
//! let config = GraphSageConfig {
//!     input_dim: 8,
//!     hidden_dim: 16,
//!     output_dim: 16,
//!     num_layers: 2,
//!     dropout: 0.0,
//!     k_neighbors: 4,
//!     learning_rate: 0.01,
//! };
//!
//! let encoder = GraphSageEncoder::new(&config).expect("construct");
//! let embeddings = encoder.encode(&graph).expect("encode");
//! assert_eq!(embeddings.embeddings.nrows(), 4);
//! assert_eq!(embeddings.embeddings.ncols(), 16);
//! ```

// ── Original GraphSAGE submodules ───────────────────────────────────────────
pub mod aggregator;
pub mod graphsage;
pub mod sampler;

// ── New GNN encoder submodules (v0.3.1) ─────────────────────────────────────
pub mod adjacency;
pub mod attention;
pub mod message_passing;

// ── Re-exports: original GraphSAGE ─────────────────────────────────────────
pub use graphsage::{
    EntityEmbeddings, GnnError, GnnResult, GraphSageConfig, GraphSageEncoder, KgGraph,
    TrainingHistory,
};

// ── Re-exports: new GNN components ─────────────────────────────────────────
pub use adjacency::{AdjacencyGraph, EdgeList};
pub use attention::ScaledDotProductAttention;
pub use message_passing::{GnnEncoder, GnnEncoderConfig};
