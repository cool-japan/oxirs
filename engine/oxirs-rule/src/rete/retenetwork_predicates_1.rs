//! # ReteNetwork - predicates Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::RuleAtom;

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Check if a RuleAtom is a filter condition (comparison or builtin test)
    /// Filter conditions should not create alpha nodes but instead be added as
    /// conditions on beta joins.
    pub(super) fn is_filter_condition(&self, atom: &RuleAtom) -> bool {
        matches!(
            atom,
            RuleAtom::GreaterThan { .. } | RuleAtom::LessThan { .. } | RuleAtom::NotEqual { .. }
        )
    }
}
