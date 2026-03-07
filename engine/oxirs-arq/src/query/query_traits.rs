//! # Query - Trait Implementations
//!
//! This module contains trait implementations for `Query`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::{Query, QueryType};

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.query_type {
            QueryType::Select => {
                write!(f, "SELECT ")?;
                if self.distinct {
                    write!(f, "DISTINCT ")?;
                } else if self.reduced {
                    write!(f, "REDUCED ")?;
                }
                if self.select_variables.is_empty() {
                    write!(f, "* ")?;
                } else {
                    for (i, var) in self.select_variables.iter().enumerate() {
                        if i > 0 {
                            write!(f, " ")?;
                        }
                        write!(f, "?{var}", var = var.as_str())?;
                    }
                    write!(f, " ")?;
                }
                write!(f, "WHERE {{ {:?} }}", self.where_clause)?;
            }
            QueryType::Construct => {
                write!(f, "CONSTRUCT {{ ")?;
                for (i, pattern) in self.construct_template.iter().enumerate() {
                    if i > 0 {
                        write!(f, " . ")?;
                    }
                    write!(f, "{pattern:?}")?;
                }
                write!(f, " }} WHERE {{ {:?} }}", self.where_clause)?;
            }
            QueryType::Ask => {
                write!(f, "ASK WHERE {{ {:?} }}", self.where_clause)?;
            }
            QueryType::Describe => {
                write!(f, "DESCRIBE ")?;
                if self.select_variables.is_empty() {
                    write!(f, "* ")?;
                } else {
                    for (i, var) in self.select_variables.iter().enumerate() {
                        if i > 0 {
                            write!(f, " ")?;
                        }
                        write!(f, "?{var}", var = var.as_str())?;
                    }
                    write!(f, " ")?;
                }
                write!(f, "WHERE {{ {:?} }}", self.where_clause)?;
            }
        }
        if let Some(limit) = self.limit {
            write!(f, " LIMIT {limit}")?;
        }
        if let Some(offset) = self.offset {
            write!(f, " OFFSET {offset}")?;
        }
        Ok(())
    }
}
