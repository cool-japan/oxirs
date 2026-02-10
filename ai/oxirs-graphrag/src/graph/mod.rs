//! Graph processing module

pub mod community;
pub mod embeddings;
pub mod subgraph;
pub mod traversal;

pub use community::{CommunityAlgorithm, CommunityConfig, CommunityDetector};
pub use embeddings::{CommunityAwareEmbeddings, CommunityStructure, EmbeddingConfig};
pub use subgraph::SubgraphExtractor;
pub use traversal::GraphTraversal;
