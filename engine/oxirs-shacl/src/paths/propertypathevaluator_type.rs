//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

/// Property path evaluator for finding values along paths
#[derive(Debug)]
pub struct PropertyPathEvaluator {
    /// Cache for path evaluation results
    pub(super) cache: HashMap<String, CachedPathResult>,
    /// Query plan cache for optimized SPARQL path queries
    pub(super) query_plan_cache: HashMap<String, PathQueryPlan>,
    /// Maximum recursion depth for cyclic paths
    pub(super) max_depth: usize,
    /// Maximum number of intermediate results to track
    pub(super) max_intermediate_results: usize,
    /// Path evaluation statistics
    pub(super) stats: PathEvaluationStats,
    /// Cache configuration
    pub(super) cache_config: PathCacheConfig,
}
