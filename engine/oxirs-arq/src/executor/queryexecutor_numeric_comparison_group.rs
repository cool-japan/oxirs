//! # QueryExecutor - numeric_comparison_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Perform numeric comparison between terms
    pub(super) fn numeric_comparison(
        &self,
        left: &crate::algebra::Term,
        right: &crate::algebra::Term,
        op: impl Fn(f64, f64) -> bool,
    ) -> Result<crate::algebra::Term> {
        let left_num = self.extract_numeric_value(left)?;
        let right_num = self.extract_numeric_value(right)?;
        let result = op(left_num, right_num);
        Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
            value: result.to_string(),
            language: None,
            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#boolean",
            )),
        }))
    }
    /// Extract numeric value from a term
    pub(super) fn extract_numeric_value(&self, term: &crate::algebra::Term) -> Result<f64> {
        match term {
            crate::algebra::Term::Literal(lit) => lit
                .value
                .parse::<f64>()
                .map_err(|_| anyhow::anyhow!("Cannot convert literal to number: {}", lit.value)),
            _ => Err(anyhow::anyhow!(
                "Cannot extract numeric value from non-literal term"
            )),
        }
    }
}
