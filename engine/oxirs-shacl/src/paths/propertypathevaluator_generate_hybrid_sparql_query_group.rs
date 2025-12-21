//! # PropertyPathEvaluator - generate_hybrid_sparql_query_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Generate hybrid SPARQL query that combines multiple strategies
    pub(super) fn generate_hybrid_sparql_query(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        match path {
            PropertyPath::Sequence(paths) => self.generate_sequence_sparql_query(
                start_node,
                paths,
                graph_name,
                optimization_hints,
            ),
            PropertyPath::Alternative(paths) => {
                self.generate_union_sparql_query(start_node, paths, graph_name, optimization_hints)
            }
            PropertyPath::ZeroOrMore(_) | PropertyPath::OneOrMore(_) => self
                .generate_recursive_sparql_query(start_node, path, graph_name, optimization_hints),
            _ => self.generate_native_sparql_path_query(
                start_node,
                path,
                graph_name,
                optimization_hints,
            ),
        }
    }
}
