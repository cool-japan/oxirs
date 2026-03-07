//! # QueryExecutor - union_solutions_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Union two solutions
    pub(super) fn union_solutions(&self, mut left: Solution, right: Solution) -> Solution {
        left.extend(right);
        left
    }
}
