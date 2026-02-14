//! # QueryExecutor - predicates Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Check if a term represents a numeric literal
    pub(super) fn is_numeric_literal(&self, term: &crate::algebra::Term) -> bool {
        match term {
            crate::algebra::Term::Literal(lit) => {
                if let Some(ref datatype) = lit.datatype {
                    self.is_numeric_datatype(datatype)
                } else {
                    lit.value.parse::<f64>().is_ok()
                }
            }
            _ => false,
        }
    }
}
