//! Revolutionary Cluster Optimization Framework for OxiRS Cluster
//!
//! This module integrates the revolutionary AI capabilities developed in oxirs-arq
//! with the distributed cluster system, providing AI-powered consensus optimization,
//! intelligent data distribution, adaptive replication strategies, and unified
//! performance coordination across the distributed infrastructure.
//!
//! ## Refactored Module Structure
//!
//! This module has been refactored from a single 2825-line file into a well-organized
//! modular structure. See `revolutionary_cluster` submodule for implementation details.

mod revolutionary_cluster;

// Re-export everything from the modular implementation
pub use revolutionary_cluster::*;
