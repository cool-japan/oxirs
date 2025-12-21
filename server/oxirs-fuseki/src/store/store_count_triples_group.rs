//! # Store - count_triples_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Count triples in a specific dataset or the default dataset
    /// Returns 0 if the dataset doesn't exist or on error
    pub fn count_triples(&self, dataset_name: &str) -> usize {
        let dataset_opt = if dataset_name == "default" || dataset_name.is_empty() {
            Some(Arc::clone(&self.default_store))
        } else {
            self.datasets
                .read()
                .ok()
                .and_then(|datasets| datasets.get(dataset_name).cloned())
        };
        match dataset_opt {
            Some(store) => store
                .read()
                .ok()
                .and_then(|guard| guard.len().ok())
                .unwrap_or(0),
            None => 0,
        }
    }
}
