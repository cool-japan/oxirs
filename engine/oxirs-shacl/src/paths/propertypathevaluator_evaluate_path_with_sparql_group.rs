//! # PropertyPathEvaluator - evaluate_path_with_sparql_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Evaluate a property path using optimized SPARQL property path query
    pub(super) fn evaluate_path_with_sparql(
        &self,
        store: &dyn Store,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let sparql_path = path.to_sparql_path()?;
        let mut values = Vec::new();
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?value WHERE {{
                    GRAPH <{}> {{
                        {} {} ?value .
                    }}
                }}
                ORDER BY ?value
            "#,
                graph,
                format_term_for_sparql(start_node)?,
                sparql_path
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?value WHERE {{
                    {} {} ?value .
                }}
                ORDER BY ?value
            "#,
                format_term_for_sparql(start_node)?,
                sparql_path
            )
        };
        match self.execute_path_query(store, &query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(value) = binding.get("value") {
                            values.push(value.clone());
                        }
                    }
                }
            }
            Err(e) => {
                return Err(ShaclError::PropertyPath(format!(
                    "SPARQL property path query failed: {e}"
                )));
            }
        }
        tracing::debug!(
            "Found {} values using SPARQL property path: {}",
            values.len(),
            sparql_path
        );
        Ok(values)
    }
}
