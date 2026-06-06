//! Distributed GraphRAG: federated subgraph expansion across multiple SPARQL endpoints.
//!
//! This module provides the building blocks for querying heterogeneous, geographically
//! distributed knowledge graphs and merging the results into a single coherent subgraph
//! suitable for retrieval-augmented generation.
//!
//! ## Architecture
//!
//! ```text
//! Query Seeds
//!     │
//!     ▼
//! FederatedSubgraphExpander ──► [Endpoint A] ──► subgraph_A
//!     │                    ──► [Endpoint B] ──► subgraph_B   ──► merge + resolve ──► KnowledgeGraph
//!     │                    ──► [Endpoint C] ──► subgraph_C
//!     │
//!     ▼
//! DistributedEntityResolver  (sameAs closure)
//!     │
//!     ▼
//! FederatedContextBuilder    (priority + confidence ranking)
//!     │
//!     ▼
//! RAG context string
//! ```
//!
//! ## Submodule layout
//!
//! | Submodule | Contents |
//! |-----------|----------|
//! `coordinator` | Error types, config, `KnowledgeGraph`, `EndpointExecutor` trait, `DistributedEntityResolver`, `FederatedContextBuilder` |
//! `worker`      | HTTP executor impl, `FederatedSubgraphExpander`, `DistributedGraphRAGMetrics`, SPARQL builders |
//! `distributed_tests` | Integration tests (cfg(test) only) |

pub mod coordinator;
pub mod worker;

#[cfg(test)]
mod distributed_tests;

// ─────────────────────────────────────────────────────────────────────────────
// Flat re-exports — preserve the public API that existed before the split
// ─────────────────────────────────────────────────────────────────────────────

pub use coordinator::{
    ContextOrderingStrategy, DistributedEntityResolver, DistributedError, EndpointAuth,
    EndpointConfig, EndpointExecutor, FederatedContextBuilder, FederatedContextConfig,
    FederatedGraphRAGConfig, KnowledgeGraph,
};

pub use worker::{
    AggregateMetrics, DistributedGraphRAGMetrics, EndpointMetrics, FederatedSubgraphExpander,
    HttpEndpointExecutor,
};
