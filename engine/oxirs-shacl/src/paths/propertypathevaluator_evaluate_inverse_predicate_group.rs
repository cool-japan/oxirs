//! # PropertyPathEvaluator - evaluate_inverse_predicate_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Evaluate inverse of a simple predicate
    pub(super) fn evaluate_inverse_predicate(
        &self,
        store: &dyn Store,
        start_node: &Term,
        predicate: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut values = Vec::new();
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?value WHERE {{
                    GRAPH <{}> {{
                        ?value <{}> {} .
                    }}
                }}
            "#,
                graph,
                predicate.as_str(),
                format_term_for_sparql(start_node)?
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?value WHERE {{
                    ?value <{}> {} .
                }}
            "#,
                predicate.as_str(),
                format_term_for_sparql(start_node)?
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
                tracing::error!("Failed to execute inverse predicate path query: {}", e);
                values = self
                    .evaluate_inverse_predicate_direct(store, start_node, predicate, graph_name)?;
            }
        }
        tracing::debug!(
            "Found {} values for inverse predicate path {}",
            values.len(),
            predicate.as_str()
        );
        Ok(values)
    }
}
