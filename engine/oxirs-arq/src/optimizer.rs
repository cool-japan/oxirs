//! Query Optimization Module
//!
//! This module provides query optimization capabilities including
//! rule-based and cost-based optimization passes.

// Re-export all types and functions from the modularized structure
pub use self::optimizer::*;

mod optimizer;