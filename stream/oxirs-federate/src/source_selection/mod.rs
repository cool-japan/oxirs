//! # Advanced Source Selection Algorithms
//!
//! This module implements sophisticated algorithms for selecting optimal data sources
//! in federated query processing. It includes triple pattern coverage analysis,
//! predicate-based filtering, range-based selection, and ML-driven source prediction.

pub mod types;
pub mod advanced_selector;
pub mod pattern_coverage;
pub mod predicate_filter;
pub mod range_selector;
pub mod ml_predictor;

// Re-export all types (includes AdvancedSourceSelector, PatternCoverageAnalyzer, etc.)
pub use types::*;
