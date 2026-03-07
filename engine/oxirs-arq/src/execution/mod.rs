//! Execution sub-modules for oxirs-arq
//!
//! This module groups execution-related components that complement the
//! existing `executor` module with new parallel and adaptive capabilities.

pub mod parallel_eval;

pub use parallel_eval::{ParallelBgpEvaluator, PatternDependencyGraph, TripleStore};
