//! Sophisticated Validation Optimization Strategies
//!
//! This module provides ultra-advanced validation optimization strategies that combine
//! machine learning, quantum computing principles, adaptive algorithms, and sophisticated
//! heuristics to achieve optimal validation performance and accuracy.
//!
//! This is a thin facade that re-exports the public API from cohesive sibling
//! modules:
//! - [`crate::sophisticated_validation_optimization_types`] — configuration,
//!   objectives, results, metrics, contexts, caches, and supporting types.
//! - [`crate::sophisticated_validation_optimization_strategies`] — the
//!   [`OptimizationStrategy`] trait and concrete strategies.
//! - [`crate::sophisticated_validation_optimization_core`] — the
//!   [`SophisticatedValidationOptimizer`] orchestrator and component optimizers.

pub use crate::sophisticated_validation_optimization_core::*;
pub use crate::sophisticated_validation_optimization_strategies::*;
pub use crate::sophisticated_validation_optimization_types::*;
