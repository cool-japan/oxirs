//! # Store - update_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Execute a SPARQL update against the default dataset
    pub fn update(&self, sparql: &str) -> FusekiResult<UpdateResult> {
        self.update_dataset(sparql, None)
    }
}
