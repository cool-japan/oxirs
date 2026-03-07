//! # ReteNetwork - convert_filter_to_join_condition_group Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{RuleAtom, Term};

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Convert a filter RuleAtom to a JoinCondition for the enhanced beta join
    pub(super) fn convert_filter_to_join_condition(
        &self,
        atom: &RuleAtom,
    ) -> Option<crate::rete_enhanced::JoinCondition> {
        match atom {
            RuleAtom::GreaterThan { left, right } => self.create_comparison_condition(
                left,
                right,
                crate::rete_enhanced::ComparisonOp::Greater,
            ),
            RuleAtom::LessThan { left, right } => self.create_comparison_condition(
                left,
                right,
                crate::rete_enhanced::ComparisonOp::Less,
            ),
            RuleAtom::NotEqual { left, right } => self.create_comparison_condition(
                left,
                right,
                crate::rete_enhanced::ComparisonOp::NotEqual,
            ),
            _ => None,
        }
    }
    /// Create a comparison condition from two terms
    pub(super) fn create_comparison_condition(
        &self,
        left: &Term,
        right: &Term,
        op: crate::rete_enhanced::ComparisonOp,
    ) -> Option<crate::rete_enhanced::JoinCondition> {
        match (left, right) {
            (Term::Variable(left_var), Term::Variable(right_var)) => {
                Some(crate::rete_enhanced::JoinCondition::VarComparison {
                    left_var: left_var.clone(),
                    right_var: right_var.clone(),
                    op,
                })
            }
            (Term::Variable(var), constant) | (constant, Term::Variable(var)) => {
                Some(crate::rete_enhanced::JoinCondition::VarConstComparison {
                    var: var.clone(),
                    constant: constant.clone(),
                    op,
                })
            }
            _ => None,
        }
    }
}
