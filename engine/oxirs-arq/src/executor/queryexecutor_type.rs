//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub use super::config::{
    ExecutionContext, ParallelConfig, StreamingResultConfig, ThreadPoolConfig,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::types::{CachedResult, ExecutionStrategy, FunctionRegistry};

/// Advanced Query executor with parallel and streaming capabilities
pub struct QueryExecutor {
    pub(super) context: ExecutionContext,
    #[allow(dead_code)]
    pub(super) function_registry: FunctionRegistry,
    pub(super) parallel_executor: Option<Arc<crate::parallel::ParallelExecutor>>,
    #[allow(dead_code)]
    pub(super) result_cache: Arc<RwLock<HashMap<String, CachedResult>>>,
    pub(super) execution_strategy: ExecutionStrategy,
}
