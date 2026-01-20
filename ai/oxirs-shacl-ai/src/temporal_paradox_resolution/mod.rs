//! Temporal Paradox Resolution Module
//!
//! This module implements temporal paradox resolution capabilities for handling
//! time-travel related validation scenarios, causality loops, and temporal
//! consistency verification across multiple timelines.

pub mod causality;
pub mod core;
pub mod timeline;
pub mod types;

// Re-export main types
pub use causality::CausalDependencyAnalyzer;
pub use core::{
    MultiTimelineValidator, QuantumTemporalEngine, TemporalConsistencyEnforcer,
    TemporalParadoxResolutionEngine, TemporalValidationProcessor, TemporalValidationResult,
};
pub use timeline::{Timeline, TimelineCoherenceManager};
pub use types::*;
