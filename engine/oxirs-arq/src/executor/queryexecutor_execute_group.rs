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
    /// Execute algebra expression with optimized strategy selection
    pub fn execute(
        &mut self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<(Solution, super::stats::ExecutionStats)> {
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
    /// Execute using parallel strategy
    pub(super) fn execute_parallel(
        &self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        if let Some(ref parallel_executor) = self.parallel_executor {
            let mut stats = super::stats::ExecutionStats::default();
            parallel_executor.execute(algebra, dataset, &self.context, &mut stats)
        } else {
            self.execute_serial(algebra, dataset)
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
