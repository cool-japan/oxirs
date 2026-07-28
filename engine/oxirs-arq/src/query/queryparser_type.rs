//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Variable;
use std::collections::{HashMap, HashSet};

use super::types::Token;

/// Query parser implementation
pub struct QueryParser {
    pub(super) tokens: Vec<Token>,
    pub(super) position: usize,
    pub(super) prefixes: HashMap<String, String>,
    pub(super) base_iri: Option<String>,
    pub(super) variables: HashSet<Variable>,
    /// Monotonic counter minting unique labels for the anonymous blank nodes of
    /// `[ … ]` blank-node property lists and `( … )` RDF collections.
    pub(super) blank_node_counter: usize,
    /// Whether the parser is currently reading a CONSTRUCT template (as opposed
    /// to a WHERE graph pattern). It flips how anonymous blank nodes are lowered:
    /// a template mints real `Term::BlankNode`s (`instantiate_construct` then
    /// mints a fresh one per solution row), while a WHERE pattern mints fresh
    /// non-distinguished `Term::Variable`s (the store matches a query blank node
    /// as an existential variable, never as a specific stored blank node).
    pub(super) in_construct_template: bool,
}
