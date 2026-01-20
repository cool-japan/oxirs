//! # Store - extract_graph_iri_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Extract graph IRI from CLEAR GRAPH statement
    pub(super) fn extract_graph_iri(&self, sparql: &str) -> FusekiResult<Option<String>> {
        let sparql_upper = sparql.to_uppercase();
        if let Some(start_pos) = sparql_upper.find("CLEAR GRAPH") {
            let remaining = &sparql[start_pos + "CLEAR GRAPH".len()..];
            if let Some(open_bracket) = remaining.find('<') {
                if let Some(close_bracket) = remaining[open_bracket + 1..].find('>') {
                    let iri = &remaining[open_bracket + 1..open_bracket + 1 + close_bracket];
                    return Ok(Some(iri.trim().to_string()));
                }
            }
            let tokens: Vec<&str> = remaining.split_whitespace().collect();
            if let Some(first_token) = tokens.first() {
                if !first_token.starts_with('<') && first_token.contains(':') {
                    warn!("Prefixed graph names not fully supported yet: {first_token}");
                    return Ok(Some(format!("urn:x-local:{first_token}")));
                }
            }
        }
        Ok(None)
    }
}
