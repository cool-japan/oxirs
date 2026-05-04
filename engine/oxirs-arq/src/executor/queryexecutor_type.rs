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
    /// Adaptive statistics store for runtime feedback to the optimizer.
    pub(super) adaptive_stats: Arc<crate::optimizer::adaptive::AdaptiveStatsStore>,
    /// Optional SLA admission control gate.
    ///
    /// When `Some`, callers can invoke
    /// [`crate::executor::QueryExecutor::execute_for_tenant`] which routes the
    /// query through the admission controller and priority dispatcher before
    /// dispatching to the regular [`crate::executor::QueryExecutor::execute`].
    pub(super) sla_gate: Option<crate::sla_integration::ArqSlaGate>,
    /// Optional runtime resource budget.
    ///
    /// When `Some`, [`crate::executor::QueryExecutor::execute`] checks the
    /// wall-time limit at entry and records one result row per binding in the
    /// returned solution.  Triple-scan hooks are provided by
    /// [`crate::query_governor::ExecutionBudget::record_triple_scan`] but must
    /// be threaded into `execute_single_pattern` by callers that iterate the
    /// store directly (stubbed — see `query_governor` module docs).
    pub(super) execution_budget: Option<std::sync::Arc<crate::query_governor::ExecutionBudget>>,
}
