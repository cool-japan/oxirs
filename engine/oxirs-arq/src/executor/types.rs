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

/// Error raised when a SPARQL expression references a function the engine does
/// not implement.
///
/// This is a distinct typed error so filter / `HAVING` row-evaluation loops can
/// tell it apart from ordinary per-row evaluation errors. An unknown function is
/// a whole-query fault that must fail loud — silently dropping the offending
/// rows would return a `200 OK` with a wrongly-shrunk (or empty) result set,
/// violating the no-silent-empty contract. By contrast, a type error on a single
/// row (SPARQL 1.1 §17.3) merely excludes that row and is not an
/// `UnknownFunctionError`.
///
/// Raise it wrapped in [`anyhow::Error`] and recover it with
/// `err.downcast_ref::<UnknownFunctionError>()`. Its `Display` reproduces the
/// engine's long-standing `"Unknown function: {name}"` text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownFunctionError(pub String);

impl std::fmt::Display for UnknownFunctionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unknown function: {}", self.0)
    }
}

impl std::error::Error for UnknownFunctionError {}
