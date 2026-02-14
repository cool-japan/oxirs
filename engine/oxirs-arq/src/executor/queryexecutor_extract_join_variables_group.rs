//! # QueryExecutor - extract_join_variables_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Algebra;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Extract join variables from algebra expressions (simplified)
    pub(super) fn extract_join_variables(
        &self,
        _left: &Algebra,
        _right: &Algebra,
    ) -> Vec<crate::algebra::Variable> {
        vec![]
    }
}
