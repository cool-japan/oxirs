//! Retrieval module for GraphRAG

pub mod fusion;
pub mod reranker;
pub mod streaming_sparql;
pub mod streaming_subgraph;

pub use fusion::{FusionStrategy, ResultFuser};
pub use reranker::Reranker;
pub use streaming_sparql::{
    StreamingSparqlConfig, StreamingSparqlRetriever, SubgraphStream as SparqlSubgraphStream,
    TriplePage,
};
pub use streaming_subgraph::{
    StreamConfig, StreamingSubgraphRetriever, SubgraphBatch, SubgraphStream,
};
