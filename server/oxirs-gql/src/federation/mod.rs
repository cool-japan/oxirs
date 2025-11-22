//! GraphQL Federation and Schema Stitching Support
//!
//! This module provides advanced federation capabilities for OxiRS GraphQL, including:
//! - Dynamic service discovery and health monitoring
//! - Remote schema introspection and merging
//! - Schema composition and stitching
//! - Cross-service query planning and optimization
//! - RDF dataset federation
//! - Load balancing and failover
//! - Distributed query tracing across subgraphs
//! - Comprehensive schema validation

pub mod config;
pub mod dataset_federation;
pub mod distributed_tracing;
pub mod enhanced_manager;
pub mod manager;
pub mod query_planner;
pub mod real_time_sync;
pub mod schema_stitcher;
pub mod schema_validation;
pub mod service_discovery;

pub use config::*;
pub use dataset_federation::*;
pub use distributed_tracing::*;
pub use enhanced_manager::*;
pub use manager::*;
pub use query_planner::*;
pub use real_time_sync::*;
pub use schema_stitcher::*;
pub use schema_validation::*;
pub use service_discovery::*;
