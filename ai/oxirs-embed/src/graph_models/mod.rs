//! Graph neural network embedding models (v0.3.0).
//!
//! This module provides production-ready GNN implementations for
//! knowledge graph and general graph embedding tasks:
//!
//! - [`graphsage`]: GraphSAGE - inductive representation learning via
//!   neighbor sampling with multiple aggregator strategies.
//! - [`gat`]: Graph Attention Networks - multi-head attention for
//!   adaptive neighborhood aggregation.
//!
//! ## Quick Start
//!
//! ### GraphSAGE
//! ```rust,no_run
//! use oxirs_embed::graph_models::graphsage::{Graph, GraphSAGEConfig, GraphSAGEModel};
//!
//! # fn main() -> anyhow::Result<()> {
//! let config = GraphSAGEConfig {
//!     input_dim: 16,
//!     hidden_dims: vec![32],
//!     output_dim: 8,
//!     ..Default::default()
//! };
//! let model = GraphSAGEModel::new(config)?;
//! // Build your graph, then:
//! // let embeddings = model.embed(&graph)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### GAT
//! ```rust,no_run
//! use oxirs_embed::graph_models::gat::{GATConfig, GATModel};
//!
//! # fn main() -> anyhow::Result<()> {
//! let config = GATConfig {
//!     input_dim: 16,
//!     output_head_dim: 8,
//!     num_layers: 2,
//!     ..Default::default()
//! };
//! let model = GATModel::new(config)?;
//! # Ok(())
//! # }
//! ```

pub mod gat;
pub mod graphsage;

// Re-export commonly used types for convenience
pub use gat::{GATConfig, GATEmbeddings, GATModel};
pub use graphsage::{
    AggregatorKind, Graph, GraphSAGEConfig, GraphSAGEEmbeddings, GraphSAGEModel, LSTMAggregator,
    MaxPoolAggregator, MeanAggregator, MeanPoolAggregator, MiniBatchConfig, MiniBatchGraphSAGE,
    TrainingMetrics,
};
