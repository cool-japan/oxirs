//! GraphQL Federation and Schema Stitching Support
//!
//! This module provides federation capabilities for OxiRS GraphQL, including:
//! - Remote schema introspection
//! - Schema merging and composition  
//! - Cross-service query planning
//! - RDF dataset federation

pub mod config;
pub mod schema_stitcher;
pub mod query_planner;
pub mod dataset_federation;
pub mod manager;

pub use config::*;
pub use schema_stitcher::*;
pub use query_planner::*;
pub use dataset_federation::*;
pub use manager::*;