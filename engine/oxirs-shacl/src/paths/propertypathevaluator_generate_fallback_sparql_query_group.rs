//! # PropertyPathEvaluator - generate_fallback_sparql_query_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Generate fallback SPARQL query for programmatic evaluation
    pub(super) fn generate_fallback_sparql_query(
        &self,
        start_node: &Term,
        _path: &PropertyPath,
        graph_name: Option<&str>,
    ) -> Result<String> {
        let start_term = format_term_for_sparql(start_node)?;
        let query = if let Some(graph) = graph_name {
            format!(
                "# Fallback query for complex path evaluation\nSELECT DISTINCT ?candidate WHERE {{\n  GRAPH <{graph}> {{\n    {start_term} ?p ?candidate .\n  }}\n}}\nLIMIT 100"
            )
        } else {
            format!(
                "# Fallback query for complex path evaluation\nSELECT DISTINCT ?candidate WHERE {{\n  {start_term} ?p ?candidate .\n}}\nLIMIT 100"
            )
        };
        Ok(query)
    }
}
