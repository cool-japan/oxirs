//! # Store - query_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Execute a SPARQL query against the default dataset
    pub fn query(&self, sparql: &str) -> FusekiResult<QueryResult> {
        self.query_dataset(sparql, None)
    }
}
