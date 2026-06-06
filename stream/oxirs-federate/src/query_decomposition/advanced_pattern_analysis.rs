//! Advanced Pattern Analysis for Federated Query Optimization
//!
//! This module re-exports all types from the three sibling sub-modules that
//! together implement advanced pattern analysis:
//! - `advanced_pattern_analysis_consciousness`: Consciousness engine, neural predictor,
//!   adaptive cache, and their supporting config/result types.
//! - `advanced_pattern_analysis_analyzer`: Core `AdvancedPatternAnalyzer`, all result
//!   and config types, and the `MLOptimizationModel`.
//! - `advanced_pattern_analysis_quantum`: Quantum-inspired optimizer and all quantum types.
//!
//! The sibling modules are declared and re-exported through `query_decomposition/mod.rs`.
//! This file exists only for backward-compatibility path references.

pub use super::advanced_pattern_analysis_analyzer::*;
pub use super::advanced_pattern_analysis_consciousness::*;
pub use super::advanced_pattern_analysis_quantum::*;
