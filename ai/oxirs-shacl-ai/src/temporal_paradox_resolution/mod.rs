//! Temporal Paradox Resolution Module
//!
//! This module implements temporal paradox resolution capabilities for handling
//! time-travel related validation scenarios, causality loops, and temporal
//! consistency verification across multiple timelines.

pub mod core;
pub mod timeline;
pub mod types;

// Re-export main types
pub use core::*;
pub use timeline::*;
pub use types::*;
