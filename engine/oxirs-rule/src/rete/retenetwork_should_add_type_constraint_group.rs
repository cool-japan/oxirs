//! # ReteNetwork - should_add_type_constraint_group Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{RuleAtom, Term};

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Check if type constraint should be added
    pub(super) fn should_add_type_constraint(&self, left: &RuleAtom, right: &RuleAtom) -> bool {
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
                p1 == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                    || p2 == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            }
            _ => false,
        }
    }
}
