//! # Store - clear_default_graph_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Clear the default graph
    pub(super) fn clear_default_graph(
        &self,
        store: &mut dyn CoreStore,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let default_quads = store
            .find_quads(
                None,
                None,
                None,
                Some(&oxirs_core::model::GraphName::DefaultGraph),
            )
            .map_err(|e| {
                FusekiError::update_execution(format!("Failed to query default graph: {e}"))
            })?;
        let mut deleted_count = 0;
        for quad in default_quads {
            if store.remove_quad(&quad).map_err(|e| {
                FusekiError::update_execution(format!(
                    "Failed to remove quad from default graph: {e}"
                ))
            })? {
                deleted_count += 1;
            }
        }
        info!("Cleared default graph: {} quads removed", deleted_count);
        Ok((
            "CLEAR DEFAULT",
            0,
            deleted_count,
            vec!["default".to_string()],
        ))
    }
    /// Helper: Clear a graph by name string
    pub(super) fn clear_graph_by_name(
        &self,
        store: &mut dyn CoreStore,
        graph_name: &str,
    ) -> FusekiResult<usize> {
        if graph_name == "default" {
            let result = self.clear_default_graph(store)?;
            Ok(result.2)
        } else {
            let result = self.clear_named_graph(store, graph_name)?;
            Ok(result.2)
        }
    }
}
