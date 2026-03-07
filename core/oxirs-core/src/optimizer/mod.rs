//! Query optimization strategies for OxiRS.
//!
//! - [`runtime_feedback`]: Adaptive optimizer that learns from execution statistics.

pub mod runtime_feedback;

pub use runtime_feedback::{AdaptiveQueryOptimizer, QueryStats, RuntimeFeedbackStore};
