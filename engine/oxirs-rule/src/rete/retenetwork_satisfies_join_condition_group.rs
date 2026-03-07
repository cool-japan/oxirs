//! # ReteNetwork - satisfies_join_condition_group Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Term;
use anyhow::Result;

use super::types::{JoinCondition, Token};

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Check if two tokens satisfy the join condition
    pub(super) fn satisfies_join_condition(
        &self,
        left_token: &Token,
        right_token: &Token,
        join_condition: &JoinCondition,
    ) -> Result<bool> {
        for (left_var, right_var) in &join_condition.constraints {
            let left_value = left_token.bindings.get(left_var);
            let right_value = right_token.bindings.get(right_var);
            match (left_value, right_value) {
                (Some(left_val), Some(right_val)) => {
                    if !self.terms_equal(left_val, right_val) {
                        return Ok(false);
                    }
                }
                (None, None) => continue,
                _ => return Ok(false),
            }
        }
        for filter in &join_condition.filters {
            if !self.apply_filter(filter, left_token, right_token)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    /// Check if two terms are equal
    pub(super) fn terms_equal(&self, term1: &Term, term2: &Term) -> bool {
        match (term1, term2) {
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            (Term::Variable(v1), Term::Variable(v2)) => v1 == v2,
            _ => false,
        }
    }
    /// Apply a filter condition
    pub(super) fn apply_filter(
        &self,
        _filter: &str,
        _left_token: &Token,
        _right_token: &Token,
    ) -> Result<bool> {
        Ok(true)
    }
}
