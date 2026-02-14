//! # QueryExecutor - accessors Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ExecutionStrategy;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Set execution strategy
    pub fn set_strategy(&mut self, strategy: ExecutionStrategy) {
        self.execution_strategy = strategy;
    }
}
