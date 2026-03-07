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
        // Try numeric comparison first
        let left_num_result = self.extract_numeric_value(left);
        let right_num_result = self.extract_numeric_value(right);

        let result = match (left_num_result, right_num_result) {
            (Ok(left_num), Ok(right_num)) => op(left_num, right_num),
            _ => {
                // Fall back to lexicographic string comparison for non-numeric types
                // This handles xsd:string, plain literals, etc.
                let left_str = match left {
                    crate::algebra::Term::Literal(lit) => lit.value.clone(),
                    crate::algebra::Term::Iri(iri) => iri.to_string(),
                    _ => {
                        return Err(anyhow::anyhow!(
                            "Cannot compare non-literal terms: {:?} and {:?}",
                            left,
                            right
                        ))
                    }
                };
                let right_str = match right {
                    crate::algebra::Term::Literal(lit) => lit.value.clone(),
                    crate::algebra::Term::Iri(iri) => iri.to_string(),
                    _ => {
                        return Err(anyhow::anyhow!(
                            "Cannot compare non-literal terms: {:?} and {:?}",
                            left,
                            right
                        ))
                    }
                };
                // Use string comparison via numeric op approximation:
                // Map string ordering to f64: -1.0, 0.0, 1.0
                let ordering = left_str.cmp(&right_str);
                let left_ord = match ordering {
                    std::cmp::Ordering::Less => 0.0f64,
                    std::cmp::Ordering::Equal => 1.0f64,
                    std::cmp::Ordering::Greater => 2.0f64,
                };
                // We need to apply op with string ordering semantics
                // op(left, right) where left < right: op(0.0, 1.0)
                // op(left, right) where left = right: op(1.0, 1.0)
                // op(left, right) where left > right: op(2.0, 1.0)
                op(left_ord, 1.0)
            }
        };
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
