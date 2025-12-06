//! # SparqlCapabilities - Trait Implementations
//!
//! This module contains trait implementations for `SparqlCapabilities`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{SparqlCapabilities, SparqlVersion};
use std::collections::HashSet;

impl Default for SparqlCapabilities {
    fn default() -> Self {
        Self {
            sparql_version: SparqlVersion::V11,
            result_formats: vec![
                "application/sparql-results+json".to_string(),
                "application/sparql-results+xml".to_string(),
            ]
            .into_iter()
            .collect(),
            graph_formats: vec!["text/turtle".to_string(), "application/rdf+xml".to_string()]
                .into_iter()
                .collect(),
            custom_functions: HashSet::new(),
            max_query_complexity: Some(1000),
            supports_federation: true,
            supports_update: false,
            supports_named_graphs: true,
            supports_full_text_search: false,
            supports_geospatial: false,
            supports_rdf_star: false,
            service_description: None,
        }
    }
}
