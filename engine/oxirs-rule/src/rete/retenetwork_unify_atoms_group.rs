//! # ReteNetwork - unify_atoms_group Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::forward::Substitution;
use crate::{RuleAtom, Term};
use anyhow::Result;

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Unify two atoms with given substitution
    pub(super) fn unify_atoms(
        &self,
        atom1: &RuleAtom,
        atom2: &RuleAtom,
        substitution: &Substitution,
    ) -> Result<Option<Substitution>> {
        match (atom1, atom2) {
            (
                RuleAtom::Triple {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                RuleAtom::Triple {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                let mut sub = substitution.clone();
                if self.unify_terms(s1, s2, &mut sub)?
                    && self.unify_terms(p1, p2, &mut sub)?
                    && self.unify_terms(o1, o2, &mut sub)?
                {
                    Ok(Some(sub))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
    /// Unify two terms
    pub(super) fn unify_terms(
        &self,
        term1: &Term,
        term2: &Term,
        substitution: &mut Substitution,
    ) -> Result<bool> {
        match (term1, term2) {
            (Term::Variable(var), term) | (term, Term::Variable(var)) => {
                if let Some(existing) = substitution.get(var) {
                    Ok(self.terms_equal(existing, term))
                } else {
                    substitution.insert(var.clone(), term.clone());
                    Ok(true)
                }
            }
            (Term::Constant(c1), Term::Constant(c2)) => Ok(c1 == c2),
            (Term::Literal(l1), Term::Literal(l2)) => Ok(l1 == l2),
            (Term::Constant(c), Term::Literal(l)) | (Term::Literal(l), Term::Constant(c)) => {
                Ok(c == l)
            }
            _ => Ok(false),
        }
    }
}
