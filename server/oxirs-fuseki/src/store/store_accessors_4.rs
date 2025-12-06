//! # Store - accessors Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Get the latest change ID
    pub async fn get_latest_change_id(&self) -> FusekiResult<u64> {
        let metadata = self.metadata.read().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata read lock: {e}"))
        })?;
        Ok(metadata.last_change_id)
    }
}
