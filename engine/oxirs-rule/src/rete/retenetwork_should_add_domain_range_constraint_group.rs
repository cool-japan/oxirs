//! # ReteNetwork - should_add_domain_range_constraint_group Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{RuleAtom, Term};

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Check if domain/range constraint should be added
    pub(super) fn should_add_domain_range_constraint(
        &self,
        left: &RuleAtom,
        right: &RuleAtom,
    ) -> bool {
        match (left, right) {
            (
                RuleAtom::Triple {
                    predicate: Term::Constant(p1),
                    ..
                },
                RuleAtom::Triple {
                    predicate: Term::Constant(p2),
                    ..
                },
            ) => {
                p1.contains("domain")
                    || p1.contains("range")
                    || p2.contains("domain")
                    || p2.contains("range")
            }
            _ => false,
        }
    }
}
