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

        // Paginate the candidate-subject scan (OFFSET/LIMIT) instead of taking a
        // single `LIMIT 1000` snapshot, so graphs with more than 1000 distinct
        // subjects are still fully considered. `max_intermediate_results` is a
        // real, error-producing hard limit: if the candidate set grows past it
        // we fail loudly instead of silently returning an incomplete result.
        const PAGE_SIZE: usize = 1000;
        let mut offset: usize = 0;
        loop {
            let candidate_query = if let Some(graph) = graph_name {
                format!(
                    r#"
                    SELECT DISTINCT ?candidate WHERE {{
                        GRAPH <{graph}> {{
                            ?candidate ?p ?o .
                        }}
                    }}
                    LIMIT {PAGE_SIZE} OFFSET {offset}
                "#
                )
            } else {
                format!(
                    "SELECT DISTINCT ?candidate WHERE {{ ?candidate ?p ?o . }} LIMIT {PAGE_SIZE} OFFSET {offset}"
                )
            };

            match self.execute_path_query(store, &candidate_query) {
                Ok(results) => {
                    let mut page_count = 0usize;
                    if let oxirs_core::query::QueryResult::Select {
                        variables: _,
                        bindings,
                    } = results
                    {
                        page_count = bindings.len();
                        for binding in bindings {
                            if let Some(candidate) = binding.get("candidate") {
                                candidates.insert(candidate.clone());
                            }
                        }
                    }

                    if page_count < PAGE_SIZE {
                        // Fewer rows than requested page size: scan exhausted.
                        break;
                    }
                    offset = offset.saturating_add(PAGE_SIZE);
                }
                Err(e) => {
                    tracing::error!("Failed to get candidates for complex inverse path: {}", e);
                    candidates = self.get_candidates_direct(store, graph_name)?;
                    break;
                }
            }

            if candidates.len() > self.max_intermediate_results {
                return Err(ShaclError::PropertyPath(format!(
                    "Complex inverse path candidate scan exceeded the limit of {} distinct \
                     subjects; the graph is too large to fully evaluate this inverse path \
                     without truncation",
                    self.max_intermediate_results
                )));
            }
        }

        if candidates.len() > self.max_intermediate_results {
            return Err(ShaclError::PropertyPath(format!(
                "Complex inverse path candidate set exceeded the limit of {} distinct subjects; \
                 the graph is too large to fully evaluate this inverse path without truncation",
                self.max_intermediate_results
            )));
        }

        tracing::debug!(
            "Testing {} candidates for complex inverse path",
            candidates.len()
        );
        for candidate in candidates {
            if depth > self.max_depth.saturating_sub(5) {
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
            if result.len() > self.max_intermediate_results {
                return Err(ShaclError::PropertyPath(format!(
                    "Complex inverse path result set exceeded the limit of {} items; refusing \
                     to silently truncate results",
                    self.max_intermediate_results
                )));
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

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::{
        model::{GraphName, Quad},
        ConcreteStore,
    };

    fn nn(iri: &str) -> NamedNode {
        NamedNode::new(iri).expect("valid IRI")
    }

    /// The candidate-subject scan must fail loudly (a real, error-producing
    /// hard limit) instead of silently truncating results when the number of
    /// distinct candidate subjects exceeds `max_intermediate_results`.
    #[test]
    fn regression_complex_inverse_candidate_scan_hard_limit_is_fail_loud() {
        let store = ConcreteStore::new().expect("store");
        // 5 distinct candidate subjects, each with one arbitrary triple.
        for i in 0..5 {
            store
                .insert_quad(Quad::new(
                    nn(&format!("http://example.org/subject{i}")),
                    nn("http://example.org/p"),
                    nn("http://example.org/o"),
                    GraphName::DefaultGraph,
                ))
                .expect("insert");
        }

        // max_intermediate_results = 3 forces the candidate-scan hard limit
        // (5 candidates > 3) to trip instead of silently truncating.
        let evaluator = PropertyPathEvaluator::with_limits(50, 3);
        let inner_path = PropertyPath::Predicate(nn("http://example.org/knows"));
        let start_node = Term::NamedNode(nn("http://example.org/target"));

        let result = evaluator.evaluate_complex_inverse(&store, &start_node, &inner_path, None, 0);

        assert!(
            result.is_err(),
            "exceeding the candidate-scan hard limit must fail loudly, not silently truncate"
        );
    }

    /// Below the hard limit, evaluation must still succeed and correctly
    /// identify matches (regression guard against the fix being overly
    /// aggressive).
    #[test]
    fn regression_complex_inverse_below_limit_finds_real_match() {
        let store = ConcreteStore::new().expect("store");
        let target = nn("http://example.org/target");
        let matching_candidate = nn("http://example.org/matches");
        let non_matching_candidate = nn("http://example.org/nomatch");
        let knows = nn("http://example.org/knows");

        // matching_candidate --knows--> target
        store
            .insert_quad(Quad::new(
                matching_candidate.clone(),
                knows.clone(),
                target.clone(),
                GraphName::DefaultGraph,
            ))
            .expect("insert matching triple");
        // non_matching_candidate has some unrelated triple, so it's a
        // candidate but does not reach `target` via `knows`.
        store
            .insert_quad(Quad::new(
                non_matching_candidate,
                nn("http://example.org/other"),
                nn("http://example.org/somethingElse"),
                GraphName::DefaultGraph,
            ))
            .expect("insert unrelated triple");

        let evaluator = PropertyPathEvaluator::with_limits(50, 10_000);
        let inner_path = PropertyPath::Predicate(knows);
        let start_node = Term::NamedNode(target);

        let result = evaluator
            .evaluate_complex_inverse(&store, &start_node, &inner_path, None, 0)
            .expect("evaluation should succeed below the hard limit");

        assert!(
            result.contains(&Term::NamedNode(matching_candidate)),
            "the real inverse match must be found, got {result:?}"
        );
    }
}
