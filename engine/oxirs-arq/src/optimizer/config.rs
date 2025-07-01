//! Optimizer Configuration
//!
//! Configuration structures and settings for query optimization.

/// Query optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Enable join reordering
    pub join_reordering: bool,
    /// Enable filter pushdown
    pub filter_pushdown: bool,
    /// Enable projection pushdown
    pub projection_pushdown: bool,
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable cost-based optimization
    pub cost_based: bool,
    /// Maximum optimization passes
    pub max_passes: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            join_reordering: true,
            filter_pushdown: true,
            projection_pushdown: true,
            constant_folding: true,
            dead_code_elimination: true,
            cost_based: true,
            max_passes: 10,
        }
    }
}
