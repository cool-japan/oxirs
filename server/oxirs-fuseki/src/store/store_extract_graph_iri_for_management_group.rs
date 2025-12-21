//! # Store - extract_graph_iri_for_management_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Helper: Extract graph IRI for graph management operations
    pub(super) fn extract_graph_iri_for_management(
        &self,
        sparql: &str,
        operation: &str,
    ) -> FusekiResult<String> {
        let operation_upper = operation.to_uppercase();
        let sparql_upper = sparql.to_uppercase();
        let op_pos = sparql_upper.find(&operation_upper).ok_or_else(|| {
            FusekiError::update_execution(format!("Operation '{operation}' not found"))
        })?;
        let remaining = &sparql[op_pos + operation.len()..];
        let search_start = if remaining.trim_start().to_uppercase().starts_with("SILENT") {
            &remaining[6..]
        } else {
            remaining
        };
        let search_upper = search_start.to_uppercase();
        let iri_start_pos = if let Some(graph_pos) = search_upper.find("GRAPH") {
            &search_start[graph_pos + 5..]
        } else {
            search_start
        };
        let open_bracket = iri_start_pos
            .find('<')
            .ok_or_else(|| FusekiError::update_execution("Graph IRI not found".to_string()))?;
        let close_bracket = iri_start_pos[open_bracket + 1..].find('>').ok_or_else(|| {
            FusekiError::update_execution("Graph IRI closing '>' not found".to_string())
        })?;
        let iri = &iri_start_pos[open_bracket + 1..open_bracket + 1 + close_bracket];
        Ok(iri.trim().to_string())
    }
    /// Helper: Parse graph management statement (COPY/MOVE/ADD)
    pub(super) fn parse_graph_management_statement(
        &self,
        sparql: &str,
        operation: &str,
    ) -> FusekiResult<(String, String)> {
        let sparql_upper = sparql.to_uppercase();
        let to_pos = sparql_upper
            .find(" TO ")
            .ok_or_else(|| FusekiError::update_execution("TO keyword not found".to_string()))?;
        let source_part = &sparql[..to_pos];
        let target_part = &sparql[to_pos + 4..];
        let source = if source_part.to_uppercase().contains("DEFAULT") {
            "default".to_string()
        } else if let Ok(iri) = self.extract_graph_iri_for_management(source_part, operation) {
            iri
        } else {
            return Err(FusekiError::update_execution(
                "Failed to parse source graph".to_string(),
            ));
        };
        let target = if target_part.to_uppercase().contains("DEFAULT") {
            "default".to_string()
        } else {
            let open_bracket = target_part.find('<').ok_or_else(|| {
                FusekiError::update_execution("Target graph IRI not found".to_string())
            })?;
            let close_bracket = target_part[open_bracket + 1..].find('>').ok_or_else(|| {
                FusekiError::update_execution("Target graph IRI closing '>' not found".to_string())
            })?;
            target_part[open_bracket + 1..open_bracket + 1 + close_bracket]
                .trim()
                .to_string()
        };
        Ok((source, target))
    }
}
