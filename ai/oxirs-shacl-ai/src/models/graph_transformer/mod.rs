//! Graph Transformer Architectures for SHACL Shape Learning
//!
//! Implements two advanced graph transformer architectures:
//!
//! - **Graphormer** (Ying et al. 2021): degree centrality encoding + spatial encoding
//!   (shortest-path distances) + standard transformer blocks.
//! - **GT** (Graph Transformer, Dwivedi & Bresson 2020): Laplacian eigenvector PE
//!   + multi-head attention with sparse adjacency mask.
//!
//! Both use hand-rolled backward passes (no autograd framework).
//! All tensors use `scirs2_core::ndarray_ext::Array2<f64>`.

pub mod attention;
pub mod graphormer;
pub mod gt;
pub mod positional_encoding;

pub use graphormer::GraphormerModel;
pub use gt::GraphTransformerModel;

/// Error type for graph-transformer operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphTransformerError {
    /// Dimension mismatch between expected and actual sizes.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// The adjacency matrix is not square.
    #[error("adjacency matrix is not square: [{rows}, {cols}]")]
    NonSquareAdjacency { rows: usize, cols: usize },

    /// Eigenvector computation failed.
    #[error("eigenvector computation failed: {0}")]
    EigenFailed(String),

    /// Empty graph (no nodes).
    #[error("empty graph: no nodes provided")]
    EmptyGraph,

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),
}
