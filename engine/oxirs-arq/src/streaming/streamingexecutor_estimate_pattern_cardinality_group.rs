//! # StreamingExecutor - estimate_pattern_cardinality_group Methods
//!
//! This module contains method implementations for `StreamingExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{Term, TriplePattern};

use super::streamingexecutor_type::StreamingExecutor;

impl StreamingExecutor {
    /// Estimate pattern cardinality for streaming optimization
    pub(super) fn estimate_pattern_cardinality(&self, pattern: &TriplePattern) -> usize {
        let mut cardinality = 1000000;
        if !matches!(pattern.subject, Term::Variable(_)) {
            cardinality /= 100;
        }
        if !matches!(pattern.predicate, Term::Variable(_)) {
            cardinality /= 50;
        }
        if !matches!(pattern.object, Term::Variable(_)) {
            cardinality /= 100;
        }
        cardinality.max(1)
    }
}
