//! # Store - watch_changes_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Watch for changes (used by WebSocket subscriptions)
    pub async fn watch_changes(&self, since_id: u64) -> FusekiResult<Vec<StoreChange>> {
        let metadata = self.metadata.read().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata read lock: {e}"))
        })?;
        let new_changes = metadata
            .change_log
            .iter()
            .filter(|change| change.id > since_id)
            .cloned()
            .collect();
        Ok(new_changes)
    }
}
