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
    #[allow(dead_code)]
    pub(super) blank_node_counter: usize,
}
