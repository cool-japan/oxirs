//! # Store - accessors Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Get changes since a specific timestamp for WebSocket notifications
    pub async fn get_changes_since(
        &self,
        since: chrono::DateTime<chrono::Utc>,
    ) -> FusekiResult<Vec<StoreChange>> {
        let metadata = self.metadata.read().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata read lock: {e}"))
        })?;
        let recent_changes = metadata
            .change_log
            .iter()
            .filter(|change| change.timestamp > since)
            .cloned()
            .collect();
        Ok(recent_changes)
    }
}
