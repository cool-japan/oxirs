//! # ReteNetwork - apply_substitution_group Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::forward::Substitution;
use crate::{RuleAtom, Term};
use anyhow::Result;

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Apply substitution to an atom
    pub(super) fn apply_substitution(
        &self,
        atom: &RuleAtom,
        substitution: &Substitution,
    ) -> Result<RuleAtom> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => Ok(RuleAtom::Triple {
                subject: self.substitute_term(subject, substitution),
                predicate: self.substitute_term(predicate, substitution),
                object: self.substitute_term(object, substitution),
            }),
            RuleAtom::Builtin { name, args } => {
                let substituted_args = args
                    .iter()
                    .map(|arg| self.substitute_term(arg, substitution))
                    .collect();
                Ok(RuleAtom::Builtin {
                    name: name.clone(),
                    args: substituted_args,
                })
            }
            RuleAtom::NotEqual { left, right } => Ok(RuleAtom::NotEqual {
                left: self.substitute_term(left, substitution),
                right: self.substitute_term(right, substitution),
            }),
            RuleAtom::GreaterThan { left, right } => Ok(RuleAtom::GreaterThan {
                left: self.substitute_term(left, substitution),
                right: self.substitute_term(right, substitution),
            }),
            RuleAtom::LessThan { left, right } => Ok(RuleAtom::LessThan {
                left: self.substitute_term(left, substitution),
                right: self.substitute_term(right, substitution),
            }),
        }
    }
    /// Substitute variables in a term
    #[allow(clippy::only_used_in_recursion)]
    pub(super) fn substitute_term(&self, term: &Term, substitution: &Substitution) -> Term {
        match term {
            Term::Variable(var) => substitution
                .get(var)
                .cloned()
                .unwrap_or_else(|| term.clone()),
            Term::Function { name, args } => {
                let substituted_args = args
                    .iter()
                    .map(|arg| self.substitute_term(arg, substitution))
                    .collect();
                Term::Function {
                    name: name.clone(),
                    args: substituted_args,
                }
            }
            _ => term.clone(),
        }
    }
}
