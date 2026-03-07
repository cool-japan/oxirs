//! # QueryExecutor - estimate_cardinality_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Algebra;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Estimate result cardinality
    #[allow(clippy::only_used_in_recursion)]
    pub(super) fn estimate_cardinality(&self, algebra: &Algebra) -> usize {
        match algebra {
            Algebra::Bgp(patterns) => patterns.len() * 1000,
            Algebra::Join { left, right } => {
                (self.estimate_cardinality(left) * self.estimate_cardinality(right)) / 10
            }
            Algebra::Union { left, right } => {
                self.estimate_cardinality(left) + self.estimate_cardinality(right)
            }
            _ => 1000,
        }
    }
}
