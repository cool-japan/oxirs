//! # Store - cleanup_old_changes_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Clear old changes from the log
    pub async fn cleanup_old_changes(
        &self,
        older_than: chrono::DateTime<chrono::Utc>,
    ) -> FusekiResult<usize> {
        let mut metadata = self.metadata.write().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata write lock: {e}"))
        })?;
        let initial_count = metadata.change_log.len();
        metadata
            .change_log
            .retain(|change| change.timestamp > older_than);
        let removed_count = initial_count - metadata.change_log.len();
        if removed_count > 0 {
            debug!("Cleaned up {} old change log entries", removed_count);
        }
        Ok(removed_count)
    }
}
