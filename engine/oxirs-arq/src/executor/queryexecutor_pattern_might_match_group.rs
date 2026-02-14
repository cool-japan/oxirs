//! # QueryExecutor - pattern_might_match_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Check if a triple pattern might match given current bindings
    pub(super) fn pattern_might_match(
        &self,
        pattern: &crate::algebra::TriplePattern,
        binding: &crate::algebra::Binding,
    ) -> bool {
        use crate::algebra::Term;
        let subject_bound = match &pattern.subject {
            Term::Variable(var) => binding.contains_key(var),
            _ => true,
        };
        let predicate_bound = match &pattern.predicate {
            Term::Variable(var) => binding.contains_key(var),
            _ => true,
        };
        let object_bound = match &pattern.object {
            Term::Variable(var) => binding.contains_key(var),
            _ => true,
        };
        subject_bound && predicate_bound && object_bound
    }
}
