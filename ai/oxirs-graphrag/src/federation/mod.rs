//! Federation layer for distributed GraphRAG queries.
//!
//! Provides [`FederatedGraphRag`] for routing queries across multiple remote
//! RAG nodes, [`FederatedIndexBuilder`] for merging and sharding per-node
//! indices, and supporting types.

pub mod distributed;

pub use distributed::{
    FederatedGraphRag, FederatedIndexBuilder, FederatedQuery, FederatedResult, FederationNode,
    FederationRouter, FederationStrategy, IndexShard, LocalIndex, LocalRagEngine, MergedIndex,
    RagResult,
};
