//! # Store - accessors Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Get changes with advanced filtering
    pub async fn get_changes_filtered(
        &self,
        params: ChangeDetectionParams,
    ) -> FusekiResult<Vec<StoreChange>> {
        let metadata = self.metadata.read().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata read lock: {e}"))
        })?;
        let mut changes: Vec<StoreChange> = metadata
            .change_log
            .iter()
            .filter(|change| {
                if change.timestamp <= params.since {
                    return false;
                }
                if let Some(ref graphs) = params.graphs {
                    if !change.affected_graphs.iter().any(|g| graphs.contains(g)) {
                        return false;
                    }
                }
                if let Some(ref op_types) = params.operation_types {
                    if !op_types.contains(&change.operation_type) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();
        if let Some(limit) = params.limit {
            if changes.len() > limit {
                changes.truncate(limit);
            }
        }
        Ok(changes)
    }
}
