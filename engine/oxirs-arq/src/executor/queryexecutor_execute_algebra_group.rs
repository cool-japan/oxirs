//! # QueryExecutor - execute_algebra_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{Algebra, Solution};
use anyhow::Result;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Execute algebra expression for update operations
    pub fn execute_algebra(
        &mut self,
        algebra: &Algebra,
        context: &mut crate::algebra::EvaluationContext,
    ) -> Result<Vec<Solution>> {
        let solution = self.execute_serial_algebra(algebra, context)?;
        Ok(vec![solution])
    }
}
