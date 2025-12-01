//! # PropertyPathEvaluator - evaluate_path_impl_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Internal implementation of path evaluation
    pub(super) fn evaluate_path_impl(
        &self,
        store: &dyn Store,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        if depth > self.max_depth {
            return Err(ShaclError::PropertyPath(format!(
                "Maximum recursion depth {} exceeded for property path evaluation",
                self.max_depth
            )));
        }
        match path {
            PropertyPath::Predicate(predicate) => {
                self.evaluate_predicate(store, start_node, predicate, graph_name)
            }
            PropertyPath::Inverse(inner_path) => {
                self.evaluate_inverse(store, start_node, inner_path, graph_name, depth)
            }
            PropertyPath::Sequence(paths) => {
                self.evaluate_sequence(store, start_node, paths, graph_name, depth)
            }
            PropertyPath::Alternative(paths) => {
                self.evaluate_alternative(store, start_node, paths, graph_name, depth)
            }
            PropertyPath::ZeroOrMore(inner_path) => {
                self.evaluate_zero_or_more(store, start_node, inner_path, graph_name, depth)
            }
            PropertyPath::OneOrMore(inner_path) => {
                self.evaluate_one_or_more(store, start_node, inner_path, graph_name, depth)
            }
            PropertyPath::ZeroOrOne(inner_path) => {
                self.evaluate_zero_or_one(store, start_node, inner_path, graph_name, depth)
            }
        }
    }
    /// Evaluate an inverse path
    fn evaluate_inverse(
        &self,
        store: &dyn Store,
        start_node: &Term,
        inner_path: &PropertyPath,
        graph_name: Option<&str>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        match inner_path {
            PropertyPath::Predicate(predicate) => {
                self.evaluate_inverse_predicate(store, start_node, predicate, graph_name)
            }
            _ => self.evaluate_complex_inverse(store, start_node, inner_path, graph_name, depth),
        }
    }
    /// Evaluate a zero-or-one path (path?)
    fn evaluate_zero_or_one(
        &self,
        store: &dyn Store,
        start_node: &Term,
        inner_path: &PropertyPath,
        graph_name: Option<&str>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        let mut result = vec![start_node.clone()];
        let path_values =
            self.evaluate_path_impl(store, start_node, inner_path, graph_name, depth + 1)?;
        result.extend(path_values);
        let unique_result: HashSet<_> = result.into_iter().collect();
        Ok(unique_result.into_iter().collect())
    }
}
