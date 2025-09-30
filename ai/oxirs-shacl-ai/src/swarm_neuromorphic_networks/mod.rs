//! Swarm Neuromorphic Networks Module
//!
//! This module provides advanced swarm intelligence and neuromorphic computing
//! capabilities for distributed AI processing and collective decision making.

pub mod config;
pub mod coordination;
pub mod learning;
pub mod network;
pub mod network_management;
pub mod processing;
pub mod resilience;
pub mod results;
pub mod types;

// Re-export key types for convenience
pub use config::*;
pub use coordination::*;
pub use learning::*;
pub use network::*;
pub use network_management::*;
pub use processing::*;
pub use resilience::*;
pub use results::*;
pub use types::*;
