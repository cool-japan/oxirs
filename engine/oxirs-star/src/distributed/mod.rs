//! Distributed RDF-star query processing modules.
//!
//! - [`federated_star`] – Partition-aware federated SPARQL-star planning
//!   and execution across heterogeneous shards.
//! - [`star_query`] – `DistributedStarQuery`, `ShardRouter`, `ResultMerger`
//!   for splitting quoted-triple queries across shards.

/// Federated SPARQL-star query planning and result aggregation.
pub mod federated_star;

/// Distributed shard-aware SPARQL-star query processing.
pub mod star_query;
