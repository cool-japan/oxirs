//! # QueryExecutor - apply_projection_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;
use anyhow::Result;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Apply projection to solution
    pub(super) fn apply_projection(
        &self,
        solution: Solution,
        variables: &[crate::algebra::Variable],
    ) -> Result<Solution> {
        let var_set: std::collections::HashSet<_> = variables.iter().collect();
        let projected = solution
            .into_iter()
            .map(|binding| {
                binding
                    .into_iter()
                    .filter(|(var, _)| var_set.contains(var))
                    .collect()
            })
            .collect();
        Ok(projected)
    }
}
