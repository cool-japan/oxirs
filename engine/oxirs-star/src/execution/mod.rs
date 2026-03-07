//! Parallel and streaming SPARQL-star execution modules.
//!
//! - [`parallel_star`] – Work-stealing scheduler for parallel annotated
//!   triple evaluation.

/// Parallel SPARQL-star evaluation with work-stealing scheduler.
pub mod parallel_star;
