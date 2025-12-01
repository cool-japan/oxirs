//! # PropertyPathEvaluator - evaluate_predicate_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Evaluate a simple predicate path
    pub(super) fn evaluate_predicate(
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
                        {} <{}> ?value .
                    }}
                }}
            "#,
                graph,
                format_term_for_sparql(start_node)?,
                predicate.as_str()
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?value WHERE {{
                    {} <{}> ?value .
                }}
            "#,
                format_term_for_sparql(start_node)?,
                predicate.as_str()
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
                tracing::error!("Failed to execute predicate path query: {}", e);
                values =
                    self.evaluate_predicate_direct(store, start_node, predicate, graph_name)?;
            }
        }
        tracing::debug!(
            "Found {} values for predicate path {}",
            values.len(),
            predicate.as_str()
        );
        Ok(values)
    }
}
