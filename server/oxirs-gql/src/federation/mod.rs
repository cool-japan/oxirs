//! GraphQL Federation and Schema Stitching Support
//!
//! This module provides advanced federation capabilities for OxiRS GraphQL, including:
//! - Dynamic service discovery and health monitoring
//! - Remote schema introspection and merging
//! - Schema composition and stitching  
//! - Cross-service query planning and optimization
//! - RDF dataset federation
//! - Load balancing and failover

pub mod config;
pub mod dataset_federation;
pub mod enhanced_manager;
pub mod manager;
pub mod query_planner;
pub mod schema_stitcher;
pub mod service_discovery;

pub use config::*;
pub use dataset_federation::*;
pub use enhanced_manager::*;
pub use manager::*;
pub use query_planner::*;
pub use schema_stitcher::*;
pub use service_discovery::*;
