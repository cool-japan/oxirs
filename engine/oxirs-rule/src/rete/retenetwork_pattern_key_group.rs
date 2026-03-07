//! # ReteNetwork - pattern_key_group Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{RuleAtom, Term};

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Generate a unique key for a pattern
    pub(super) fn pattern_key(&self, pattern: &RuleAtom) -> String {
        match pattern {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                format!(
                    "triple:{}:{}:{}",
                    self.term_key(subject),
                    self.term_key(predicate),
                    self.term_key(object)
                )
            }
            RuleAtom::Builtin { name, args } => {
                let arg_keys: Vec<String> = args.iter().map(|arg| self.term_key(arg)).collect();
                format!("builtin:{}:{}", name, arg_keys.join(","))
            }
            RuleAtom::NotEqual { left, right } => {
                format!("notequal:{}:{}", self.term_key(left), self.term_key(right))
            }
            RuleAtom::GreaterThan { left, right } => {
                format!(
                    "greaterthan:{}:{}",
                    self.term_key(left),
                    self.term_key(right)
                )
            }
            RuleAtom::LessThan { left, right } => {
                format!("lessthan:{}:{}", self.term_key(left), self.term_key(right))
            }
        }
    }
    /// Generate a key for a term
    #[allow(clippy::only_used_in_recursion)]
    pub(super) fn term_key(&self, term: &Term) -> String {
        match term {
            Term::Variable(v) => format!("?{v}"),
            Term::Constant(c) => format!("c:{c}"),
            Term::Literal(l) => format!("l:{l}"),
            Term::Function { name, args } => {
                let arg_keys: Vec<String> = args.iter().map(|arg| self.term_key(arg)).collect();
                format!("f:{name}({})", arg_keys.join(","))
            }
        }
    }
}
