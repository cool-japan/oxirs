//! # QueryExecutor - execute_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub use super::dataset::{
    convert_property_path, ConcreteStoreDataset, Dataset, DatasetPathAdapter, InMemoryDataset,
};
use crate::algebra::{Algebra, Solution};
pub use crate::executor::stats::ExecutionStats;
use anyhow::Result;
use std::time::Duration;

use super::types::ExecutionStrategy;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Execute algebra expression with optimized strategy selection.
    ///
    /// When a runtime budget has been attached via
    /// [`crate::executor::QueryExecutor::with_budget`], this method:
    ///
    /// 1. Checks the wall-time limit **before** dispatching to the chosen
    ///    execution strategy.
    /// 2. Records one result row for every binding in the returned solution,
    ///    stopping early and propagating the error if the row limit is
    ///    exceeded.
    pub fn execute(
        &mut self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<(Solution, super::stats::ExecutionStats)> {
        // ── Budget: pre-execution wall-time check ────────────────────────
        // Preserve the typed `BudgetExceeded` (via `anyhow::Error::new`, not a
        // stringified `anyhow!("{e}")`) so the fuseki handler can `downcast_ref`
        // it and map a wall-time breach to HTTP 408 rather than a generic 500.
        if let Some(ref budget) = self.execution_budget {
            budget.check_time().map_err(anyhow::Error::new)?;
        }

        let start_time = std::time::Instant::now();
        let strategy = self.choose_execution_strategy(algebra);
        let result = match strategy {
            ExecutionStrategy::Serial => self.execute_serial(algebra, dataset),
            ExecutionStrategy::Parallel => self.execute_parallel(algebra, dataset),
            ExecutionStrategy::Streaming => self.execute_streaming(algebra, dataset),
            ExecutionStrategy::Adaptive => {
                if self.should_use_parallel(algebra) {
                    self.execute_parallel(algebra, dataset)
                } else if self.should_use_streaming(algebra) {
                    self.execute_streaming(algebra, dataset)
                } else {
                    self.execute_serial(algebra, dataset)
                }
            }
        }?;

        // ── Budget: post-execution per-row accounting ────────────────────
        if let Some(ref budget) = self.execution_budget {
            for _ in &result {
                budget.record_result_row().map_err(anyhow::Error::new)?;
            }
        }

        let execution_time = start_time.elapsed();
        let stats = super::stats::ExecutionStats {
            execution_time,
            intermediate_results: 0,
            final_results: result.len(),
            memory_used: self.estimate_memory_usage(&result),
            operations: 1,
            property_path_evaluations: 0,
            time_spent_on_paths: Duration::from_millis(0),
            service_calls: 0,
            time_spent_on_services: Duration::from_millis(0),
            warnings: vec![],
        };
        Ok((result, stats))
    }
    /// Execute using parallel strategy.
    ///
    /// Solution modifiers (ORDER BY / GROUP BY / HAVING / DISTINCT / REDUCED /
    /// PROJECT / SLICE) are evaluated by the same Serial methods the Serial
    /// strategy uses: they run once over an already-materialized solution, so
    /// parallelizing them buys nothing, while the parallel engine's own copies
    /// diverged from SPARQL semantics (datatype-IRI-first ordering, inverted
    /// unbound placement, expression-less GROUP BY keys, fabricated aggregate
    /// datatypes, no XSD casts). Pattern-level nodes still run on the parallel
    /// engine; Join first tries the same bound-join pushdown as Serial.
    pub(super) fn execute_parallel(
        &self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        let Some(ref parallel_executor) = self.parallel_executor else {
            return self.execute_serial(algebra, dataset);
        };
        match algebra {
            Algebra::OrderBy {
                pattern,
                conditions,
            } => {
                let solution = self.execute_parallel(pattern, dataset)?;
                Ok(self.apply_order_by(solution, conditions))
            }
            Algebra::Group {
                pattern,
                variables,
                aggregates,
            } => {
                let solution = self.execute_parallel(pattern, dataset)?;
                self.apply_group_by(solution, variables, aggregates)
            }
            Algebra::Distinct { pattern } => {
                let solution = self.execute_parallel(pattern, dataset)?;
                Ok(self.apply_distinct(solution))
            }
            // REDUCED is a passthrough on the Serial path; keep row counts
            // strategy-independent.
            Algebra::Reduced { pattern } => self.execute_parallel(pattern, dataset),
            Algebra::Project { pattern, variables } => {
                let solution = self.execute_parallel(pattern, dataset)?;
                self.apply_projection(solution, variables)
            }
            Algebra::Slice {
                pattern,
                offset,
                limit,
            } => {
                let solution = self.execute_parallel(pattern, dataset)?;
                Ok(self.apply_slice(solution, *offset, *limit))
            }
            // HAVING needs the dataset-aware Serial condition evaluator
            // (EXISTS, casts, typed errors); delegate the whole node.
            Algebra::Having { .. } => self.execute_serial(algebra, dataset),
            Algebra::Join { left, right } => {
                if let Some(result) = self.try_bound_join(left, right, dataset)? {
                    return Ok(result);
                }
                if let Some(result) = self.try_bound_join(right, left, dataset)? {
                    return Ok(result);
                }
                let mut stats = super::stats::ExecutionStats::default();
                parallel_executor.execute(algebra, dataset, &self.context, &mut stats)
            }
            _ => {
                let mut stats = super::stats::ExecutionStats::default();
                parallel_executor.execute(algebra, dataset, &self.context, &mut stats)
            }
        }
    }
    /// Choose optimal execution strategy
    pub(super) fn choose_execution_strategy(&self, algebra: &Algebra) -> ExecutionStrategy {
        match self.execution_strategy {
            ExecutionStrategy::Adaptive => {
                let complexity = self.estimate_complexity(algebra);
                let estimated_cardinality = self.estimate_cardinality(algebra);
                if estimated_cardinality > 100_000 {
                    ExecutionStrategy::Streaming
                } else if complexity > 5 && self.parallel_executor.is_some() {
                    ExecutionStrategy::Parallel
                } else {
                    ExecutionStrategy::Serial
                }
            }
            strategy => strategy,
        }
    }
    /// Check if query should use parallel execution
    pub(super) fn should_use_parallel(&self, algebra: &Algebra) -> bool {
        self.parallel_executor.is_some()
            && self.estimate_complexity(algebra) > 3
            && self.estimate_cardinality(algebra) > 1000
    }
    /// Check if query should use streaming execution
    pub(super) fn should_use_streaming(&self, algebra: &Algebra) -> bool {
        self.estimate_cardinality(algebra) > 50_000
    }
    /// Estimate memory usage of a solution
    pub(super) fn estimate_memory_usage(&self, solution: &Solution) -> usize {
        solution.len() * 1024
    }
}
