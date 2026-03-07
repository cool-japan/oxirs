//! Routing subsystem for federated SPARQL queries.
//!
//! The `routing` module provides:
//! - [`RegionRouter`] – routes sub-queries to the nearest / best available
//!   SPARQL endpoint based on a live latency matrix.
//! - [`LatencyMatrix`] – exponentially-smoothed inter-region latency tracker.
//! - [`RegionEndpoint`] – metadata for a SPARQL endpoint belonging to a region.
//! - [`RouteRequest`] / [`RouteDecision`] – request/response types.
//! - [`SemanticRouter`] – routes queries based on namespace-prefix analysis.
//! - [`CachingRouter`] – TTL-based cache for routing decisions.

pub mod caching_router;
pub mod region_router;
pub mod semantic_router;

pub use caching_router::{CacheStats, CachingRouter, CachingRouterConfig};
pub use region_router::{
    LatencyMatrix, Region, RegionEndpoint, RegionRouter, RouteDecision, RouteRequest, RouterConfig,
    RouterError,
};
pub use semantic_router::{
    Endpoint, EndpointCapability, EndpointRegistry, SemanticRouter, SemanticRouterError,
};
