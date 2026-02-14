//! # QueryExecutor - estimate_complexity_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Algebra;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Estimate query complexity
    #[allow(clippy::only_used_in_recursion)]
    pub(super) fn estimate_complexity(&self, algebra: &Algebra) -> usize {
        match algebra {
            Algebra::Bgp(patterns) => patterns.len(),
            Algebra::Join { left, right } => {
                1 + self.estimate_complexity(left) + self.estimate_complexity(right)
            }
            Algebra::Union { left, right } => {
                1 + self.estimate_complexity(left) + self.estimate_complexity(right)
            }
            Algebra::Filter { pattern, .. } => 1 + self.estimate_complexity(pattern),
            _ => 1,
        }
    }
}
