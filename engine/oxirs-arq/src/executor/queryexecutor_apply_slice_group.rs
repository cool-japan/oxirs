//! # QueryExecutor - apply_slice_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Apply slice (limit/offset) to solution
    pub(super) fn apply_slice(
        &self,
        solution: Solution,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Solution {
        let start = offset.unwrap_or(0);
        let end = if let Some(limit) = limit {
            start + limit
        } else {
            solution.len()
        };
        solution.into_iter().skip(start).take(end - start).collect()
    }
}
