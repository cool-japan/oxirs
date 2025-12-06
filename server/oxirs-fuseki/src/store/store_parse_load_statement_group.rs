//! # Store - parse_load_statement_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Parse LOAD statement to extract source IRI and optional target graph
    /// Syntax: LOAD <sourceIRI> [INTO GRAPH <targetIRI>]
    pub(super) fn parse_load_statement(
        &self,
        sparql: &str,
    ) -> FusekiResult<(String, Option<String>)> {
        let sparql_upper = sparql.to_uppercase();
        let load_pos = sparql_upper
            .find("LOAD")
            .ok_or_else(|| FusekiError::update_execution("LOAD keyword not found".to_string()))?;
        let remaining = &sparql[load_pos + 4..];
        let source_start = remaining.find('<').ok_or_else(|| {
            FusekiError::update_execution("Source IRI not found (expected <IRI>)".to_string())
        })?;
        let source_end = remaining[source_start + 1..].find('>').ok_or_else(|| {
            FusekiError::update_execution("Source IRI closing '>' not found".to_string())
        })?;
        let source_iri = remaining[source_start + 1..source_start + 1 + source_end]
            .trim()
            .to_string();
        let after_source = &remaining[source_start + 1 + source_end + 1..];
        let after_source_upper = after_source.to_uppercase();
        let target_graph = if let Some(into_pos) = after_source_upper.find("INTO GRAPH") {
            let graph_part = &after_source[into_pos + 10..];
            let graph_start = graph_part.find('<').ok_or_else(|| {
                FusekiError::update_execution(
                    "Target graph IRI not found after INTO GRAPH".to_string(),
                )
            })?;
            let graph_end = graph_part[graph_start + 1..].find('>').ok_or_else(|| {
                FusekiError::update_execution("Target graph IRI closing '>' not found".to_string())
            })?;
            let graph_iri = graph_part[graph_start + 1..graph_start + 1 + graph_end]
                .trim()
                .to_string();
            Some(graph_iri)
        } else {
            None
        };
        Ok((source_iri, target_graph))
    }
}
