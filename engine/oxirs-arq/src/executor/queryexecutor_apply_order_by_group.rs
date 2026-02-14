//! # QueryExecutor - apply_order_by_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Apply order by to solution
    pub(super) fn apply_order_by(
        &self,
        mut solution: Solution,
        conditions: &[crate::algebra::OrderCondition],
    ) -> Solution {
        if conditions.is_empty() {
            return solution;
        }
        solution.sort_by(|a, b| {
            for condition in conditions {
                let val_a = self.evaluate_order_expression(&condition.expr, a);
                let val_b = self.evaluate_order_expression(&condition.expr, b);
                let cmp = match (val_a, val_b) {
                    (
                        Some(crate::algebra::Term::Literal(lit_a)),
                        Some(crate::algebra::Term::Literal(lit_b)),
                    ) => {
                        if let (Ok(num_a), Ok(num_b)) =
                            (lit_a.value.parse::<f64>(), lit_b.value.parse::<f64>())
                        {
                            num_a
                                .partial_cmp(&num_b)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            lit_a.value.cmp(&lit_b.value)
                        }
                    }
                    (
                        Some(crate::algebra::Term::Iri(iri_a)),
                        Some(crate::algebra::Term::Iri(iri_b)),
                    ) => iri_a.as_str().cmp(iri_b.as_str()),
                    (Some(a), Some(b)) => format!("{a}").cmp(&format!("{b}")),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                };
                let result = if condition.ascending {
                    cmp
                } else {
                    cmp.reverse()
                };
                if result != std::cmp::Ordering::Equal {
                    return result;
                }
            }
            std::cmp::Ordering::Equal
        });
        solution
    }
    /// Evaluate an expression for ordering
    pub(super) fn evaluate_order_expression(
        &self,
        expr: &crate::algebra::Expression,
        binding: &std::collections::HashMap<crate::algebra::Variable, crate::algebra::Term>,
    ) -> Option<crate::algebra::Term> {
        match expr {
            crate::algebra::Expression::Variable(var) => binding.get(var).cloned(),
            _ => None,
        }
    }
}
