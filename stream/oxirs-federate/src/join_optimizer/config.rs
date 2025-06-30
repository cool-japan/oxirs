//! Join Optimizer Configuration
//!
//! This module contains configuration types for the join optimizer.

use serde::{Deserialize, Serialize};

/// Configuration for the join optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinOptimizerConfig {
    pub enable_star_join_detection: bool,
    pub enable_chain_optimization: bool,
    pub enable_bushy_trees: bool,
    pub max_join_order_enumeration: usize,
    pub cost_threshold_for_reoptimization: f64,
    pub adaptive_execution_enabled: bool,
    pub memory_budget_mb: usize,
    pub parallelism_factor: f64,
}

impl Default for JoinOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_star_join_detection: true,
            enable_chain_optimization: true,
            enable_bushy_trees: true,
            max_join_order_enumeration: 8,
            cost_threshold_for_reoptimization: 0.2,
            adaptive_execution_enabled: true,
            memory_budget_mb: 1024,
            parallelism_factor: 2.0,
        }
    }
}