//! # Store - accessors Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Helper: Get all quads from a graph (by name string)
    pub(super) fn get_quads_from_graph(
        &self,
        store: &dyn CoreStore,
        graph_name: &str,
    ) -> FusekiResult<Vec<Quad>> {
        let graph_name_obj = self.graph_name_from_string(graph_name)?;
        let quads = store
            .find_quads(None, None, None, Some(&graph_name_obj))
            .map_err(|e| FusekiError::update_execution(format!("Failed to query graph: {e}")))?;
        Ok(quads)
    }
}
