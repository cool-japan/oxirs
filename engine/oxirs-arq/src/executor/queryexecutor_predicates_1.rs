//! # QueryExecutor - predicates Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Check if a term is truthy according to SPARQL semantics
    pub(super) fn is_term_truthy(&self, term: &crate::algebra::Term) -> Result<bool> {
        match term {
            crate::algebra::Term::Literal(lit) => Ok(self.is_truthy(lit)),
            _ => Err(anyhow::anyhow!(
                "Cannot evaluate truthiness of non-literal term"
            )),
        }
    }
}
