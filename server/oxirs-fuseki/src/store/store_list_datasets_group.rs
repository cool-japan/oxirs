//! # Store - list_datasets_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// List all dataset names
    pub fn list_datasets(&self) -> FusekiResult<Vec<String>> {
        let datasets = self
            .datasets
            .read()
            .map_err(|e| FusekiError::store(format!("Failed to acquire read lock: {e}")))?;
        Ok(datasets.keys().cloned().collect())
    }
}
