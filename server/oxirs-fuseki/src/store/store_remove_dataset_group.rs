//! # Store - remove_dataset_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Remove a named dataset
    pub fn remove_dataset(&self, name: &str) -> FusekiResult<bool> {
        let mut datasets = self
            .datasets
            .write()
            .map_err(|e| FusekiError::store(format!("Failed to acquire write lock: {e}")))?;
        let removed = datasets.remove(name).is_some();
        if removed {
            info!("Removed dataset: '{}'", name);
        }
        Ok(removed)
    }
}
