//! # StreamingExecutor - extract_join_variables_group Methods
//!
//! This module contains method implementations for `StreamingExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{Algebra, Variable};

use super::streamingexecutor_type::StreamingExecutor;

impl StreamingExecutor {
    /// Extract variables that are shared between left and right algebra expressions
    pub(super) fn extract_join_variables(&self, left: &Algebra, right: &Algebra) -> Vec<Variable> {
        let left_vars = self.extract_variables_from_algebra(left);
        let right_vars = self.extract_variables_from_algebra(right);
        left_vars
            .into_iter()
            .filter(|var| right_vars.contains(var))
            .collect()
    }
}
