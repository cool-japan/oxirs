//! # PropertyPath - Trait Implementations
//!
//! This module contains trait implementations for `PropertyPath`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl std::fmt::Display for PropertyPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PropertyPath::Predicate(pred) => write!(f, "{}", pred.as_str()),
            PropertyPath::Inverse(path) => write!(f, "^{path}"),
            PropertyPath::Sequence(paths) => {
                let path_strs: Vec<String> = paths.iter().map(|p| p.to_string()).collect();
                write!(f, "{}", path_strs.join("/"))
            }
            PropertyPath::Alternative(paths) => {
                let path_strs: Vec<String> = paths.iter().map(|p| p.to_string()).collect();
                write!(f, "({})", path_strs.join("|"))
            }
            PropertyPath::ZeroOrMore(path) => write!(f, "{path}*"),
            PropertyPath::OneOrMore(path) => write!(f, "{path}+"),
            PropertyPath::ZeroOrOne(path) => write!(f, "{path}?"),
        }
    }
}
