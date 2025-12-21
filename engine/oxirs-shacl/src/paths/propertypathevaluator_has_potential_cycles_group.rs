//! # PropertyPathEvaluator - has_potential_cycles_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Check if a path has potential for infinite cycles
    #[allow(clippy::only_used_in_recursion)]
    pub(super) fn has_potential_cycles(&self, path: &PropertyPath) -> bool {
        match path {
            PropertyPath::ZeroOrMore(_) | PropertyPath::OneOrMore(_) => true,
            PropertyPath::Sequence(paths) | PropertyPath::Alternative(paths) => {
                paths.iter().any(|p| self.has_potential_cycles(p))
            }
            PropertyPath::Inverse(inner) | PropertyPath::ZeroOrOne(inner) => {
                self.has_potential_cycles(inner)
            }
            PropertyPath::Predicate(_) => false,
        }
    }
}
