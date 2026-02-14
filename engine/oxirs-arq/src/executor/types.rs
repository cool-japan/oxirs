//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;

/// Function registry for custom functions
#[derive(Debug, Clone)]
pub struct FunctionRegistry {}
impl FunctionRegistry {
    pub fn new() -> Self {
        Self {}
    }
}
/// Access path selection for pattern execution
#[derive(Debug, Clone, Copy)]
pub enum AccessPath {
    SubjectIndex,
    PredicateIndex,
    ObjectIndex,
    FullScan,
}
/// Query execution strategy
#[derive(Debug, Clone, Copy, Default)]
pub enum ExecutionStrategy {
    /// Always use serial execution
    Serial,
    /// Always use parallel execution
    Parallel,
    /// Use streaming execution for large results
    Streaming,
    /// Adaptive strategy based on query characteristics
    #[default]
    Adaptive,
}
/// Cached result for query caching
#[derive(Debug, Clone)]
pub struct CachedResult {
    pub solution: Solution,
    pub timestamp: std::time::Instant,
}
