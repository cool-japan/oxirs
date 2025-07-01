//! Intelligent Error Handling System for SHACL-AI
//!
//! This module provides AI-powered error classification, impact assessment,
//! and automated repair suggestions for SHACL validation errors.

pub mod classification;
pub mod config;
pub mod engine;
pub mod impact;
pub mod prevention;
pub mod repair;
pub mod types;

// Re-export main types and functions
pub use classification::*;
pub use config::*;
pub use engine::*;
pub use impact::*;
pub use prevention::*;
pub use repair::*;
pub use types::*;
