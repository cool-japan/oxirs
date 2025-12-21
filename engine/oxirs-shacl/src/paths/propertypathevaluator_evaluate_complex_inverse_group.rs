//! # PropertyPathEvaluator - evaluate_complex_inverse_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Evaluate inverse of a complex path
    pub(super) fn evaluate_complex_inverse(
        &self,
        store: &dyn Store,
        start_node: &Term,
        inner_path: &PropertyPath,
        graph_name: Option<&str>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        let mut result = Vec::new();
        let mut candidates: HashSet<Term> = HashSet::new();
        let candidate_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?candidate WHERE {{
                    GRAPH <{graph}> {{
                        ?candidate ?p ?o .
                    }}
                }}
                LIMIT 1000
            "#
            )
        } else {
            "SELECT DISTINCT ?candidate WHERE { ?candidate ?p ?o . } LIMIT 1000".to_string()
        };
        match self.execute_path_query(store, &candidate_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(candidate) = binding.get("candidate") {
                            candidates.insert(candidate.clone());
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to get candidates for complex inverse path: {}", e);
                candidates = self.get_candidates_direct(store, graph_name)?;
            }
        }
        tracing::debug!(
            "Testing {} candidates for complex inverse path",
            candidates.len()
        );
        for candidate in candidates {
            if depth > self.max_depth - 5 {
                tracing::warn!("Stopping complex inverse path evaluation due to depth limit");
                break;
            }
            match self.evaluate_path_impl(store, &candidate, inner_path, graph_name, depth + 1) {
                Ok(path_values) => {
                    if path_values.contains(start_node) {
                        result.push(candidate);
                    }
                }
                Err(e) => {
                    tracing::debug!("Failed to evaluate path for candidate: {}", e);
                }
            }
            if result.len() > 100 {
                tracing::warn!("Limiting complex inverse path results to 100 items");
                break;
            }
        }
        tracing::debug!("Complex inverse path found {} results", result.len());
        Ok(result)
    }
    /// Evaluate a sequence path (path1 / path2 / ...)
    pub(super) fn evaluate_sequence(
        &self,
        store: &dyn Store,
        start_node: &Term,
        paths: &[PropertyPath],
        graph_name: Option<&str>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        if paths.is_empty() {
            return Ok(vec![start_node.clone()]);
        }
        let mut current_nodes = vec![start_node.clone()];
        for path in paths {
            let mut next_nodes = Vec::new();
            for node in &current_nodes {
                let path_values =
                    self.evaluate_path_impl(store, node, path, graph_name, depth + 1)?;
                next_nodes.extend(path_values);
                if next_nodes.len() > self.max_intermediate_results {
                    return Err(ShaclError::PropertyPath(format!(
                        "Too many intermediate results ({}), limit is {}",
                        next_nodes.len(),
                        self.max_intermediate_results
                    )));
                }
            }
            current_nodes = next_nodes;
            if current_nodes.is_empty() {
                break;
            }
        }
        Ok(current_nodes)
    }
    /// Evaluate an alternative path (path1 | path2 | ...)
    pub(super) fn evaluate_alternative(
        &self,
        store: &dyn Store,
        start_node: &Term,
        paths: &[PropertyPath],
        graph_name: Option<&str>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        let mut all_values = HashSet::new();
        for path in paths {
            let path_values =
                self.evaluate_path_impl(store, start_node, path, graph_name, depth + 1)?;
            all_values.extend(path_values);
        }
        Ok(all_values.into_iter().collect())
    }
    /// Evaluate a zero-or-more path (path*)
    pub(super) fn evaluate_zero_or_more(
        &self,
        store: &dyn Store,
        start_node: &Term,
        inner_path: &PropertyPath,
        graph_name: Option<&str>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        result.insert(start_node.clone());
        queue.push_back(start_node.clone());
        visited.insert(start_node.clone());
        while let Some(current) = queue.pop_front() {
            if result.len() > self.max_intermediate_results {
                return Err(ShaclError::PropertyPath(format!(
                    "Too many results in zero-or-more path ({}), limit is {}",
                    result.len(),
                    self.max_intermediate_results
                )));
            }
            let path_values =
                self.evaluate_path_impl(store, &current, inner_path, graph_name, depth + 1)?;
            for value in path_values {
                if !visited.contains(&value) {
                    visited.insert(value.clone());
                    result.insert(value.clone());
                    queue.push_back(value);
                }
            }
        }
        Ok(result.into_iter().collect())
    }
    /// Evaluate a one-or-more path (path+)
    pub(super) fn evaluate_one_or_more(
        &self,
        store: &dyn Store,
        start_node: &Term,
        inner_path: &PropertyPath,
        graph_name: Option<&str>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let initial_values =
            self.evaluate_path_impl(store, start_node, inner_path, graph_name, depth + 1)?;
        for value in initial_values {
            result.insert(value.clone());
            queue.push_back(value.clone());
            visited.insert(value);
        }
        while let Some(current) = queue.pop_front() {
            if result.len() > self.max_intermediate_results {
                return Err(ShaclError::PropertyPath(format!(
                    "Too many results in one-or-more path ({}), limit is {}",
                    result.len(),
                    self.max_intermediate_results
                )));
            }
            let path_values =
                self.evaluate_path_impl(store, &current, inner_path, graph_name, depth + 1)?;
            for value in path_values {
                if !visited.contains(&value) {
                    visited.insert(value.clone());
                    result.insert(value.clone());
                    queue.push_back(value);
                }
            }
        }
        Ok(result.into_iter().collect())
    }
    /// Execute a path query using oxirs-core query engine
    pub(super) fn execute_path_query(
        &self,
        store: &dyn Store,
        query: &str,
    ) -> Result<oxirs_core::query::QueryResult> {
        use oxirs_core::query::QueryEngine;
        let query_engine = QueryEngine::new();
        tracing::debug!("Executing path query: {}", query);
        let parser = oxirs_core::query::parser::SparqlParser::new();
        let parsed_query = parser
            .parse(query)
            .map_err(|e| ShaclError::PropertyPath(format!("Query parsing failed: {e}")))?;
        let result = query_engine
            .execute_query(&parsed_query, store)
            .map_err(|e| ShaclError::PropertyPath(format!("Path query execution failed: {e}")))?;
        Ok(result)
    }
}
