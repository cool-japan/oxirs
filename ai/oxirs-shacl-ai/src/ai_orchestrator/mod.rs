//! AI Orchestrator for Comprehensive Shape Learning
//!
//! This module orchestrates all AI capabilities to provide intelligent,
//! comprehensive SHACL shape learning and validation optimization.

pub mod config;
pub mod core;
pub mod metrics;
pub mod model_selection;
pub mod types;

// Re-export main types and functions
pub use config::*;
pub use core::*;
pub use metrics::*;
pub use model_selection::*;
pub use types::*;