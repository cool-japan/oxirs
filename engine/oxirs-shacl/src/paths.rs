//! SHACL property path implementation
//!
//! This module handles property path evaluation according to SHACL specification.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use oxirs_core::{
    model::{BlankNode, NamedNode, RdfTerm, Term, Triple},
    OxirsError, Store,
};

use crate::{Result, ShaclError};

/// SHACL property path types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyPath {
    /// Simple property path (single predicate)
    Predicate(NamedNode),

    /// Inverse property path (^predicate)
    Inverse(Box<PropertyPath>),

    /// Sequence path (path1 / path2 / ...)
    Sequence(Vec<PropertyPath>),

    /// Alternative path (path1 | path2 | ...)
    Alternative(Vec<PropertyPath>),

    /// Zero-or-more path (path*)
    ZeroOrMore(Box<PropertyPath>),

    /// One-or-more path (path+)
    OneOrMore(Box<PropertyPath>),

    /// Zero-or-one path (path?)
    ZeroOrOne(Box<PropertyPath>),
}

impl PropertyPath {
    /// Create a simple predicate path
    pub fn predicate(predicate: NamedNode) -> Self {
        PropertyPath::Predicate(predicate)
    }

    /// Create an inverse path
    pub fn inverse(path: PropertyPath) -> Self {
        PropertyPath::Inverse(Box::new(path))
    }

    /// Create a sequence path
    pub fn sequence(paths: Vec<PropertyPath>) -> Self {
        PropertyPath::Sequence(paths)
    }

    /// Create an alternative path
    pub fn alternative(paths: Vec<PropertyPath>) -> Self {
        PropertyPath::Alternative(paths)
    }

    /// Create a zero-or-more path
    pub fn zero_or_more(path: PropertyPath) -> Self {
        PropertyPath::ZeroOrMore(Box::new(path))
    }

    /// Create a one-or-more path
    pub fn one_or_more(path: PropertyPath) -> Self {
        PropertyPath::OneOrMore(Box::new(path))
    }

    /// Create a zero-or-one path
    pub fn zero_or_one(path: PropertyPath) -> Self {
        PropertyPath::ZeroOrOne(Box::new(path))
    }

    /// Check if this is a simple predicate path
    pub fn is_predicate(&self) -> bool {
        matches!(self, PropertyPath::Predicate(_))
    }

    /// Get the predicate if this is a simple predicate path
    pub fn as_predicate(&self) -> Option<&NamedNode> {
        match self {
            PropertyPath::Predicate(p) => Some(p),
            _ => None,
        }
    }

    /// Check if this path involves complex operations (non-predicate)
    pub fn is_complex(&self) -> bool {
        !self.is_predicate()
    }

    /// Generate SPARQL property path syntax for this path
    pub fn to_sparql_path(&self) -> Result<String> {
        match self {
            PropertyPath::Predicate(predicate) => Ok(format!("<{}>", predicate.as_str())),
            PropertyPath::Inverse(inner_path) => Ok(format!("^({})", inner_path.to_sparql_path()?)),
            PropertyPath::Sequence(paths) => {
                let path_strs: Result<Vec<String>> =
                    paths.iter().map(|p| p.to_sparql_path()).collect();
                Ok(path_strs?.join(" / "))
            }
            PropertyPath::Alternative(paths) => {
                let path_strs: Result<Vec<String>> =
                    paths.iter().map(|p| p.to_sparql_path()).collect();
                Ok(format!("({})", path_strs?.join(" | ")))
            }
            PropertyPath::ZeroOrMore(inner_path) => {
                Ok(format!("({})*", inner_path.to_sparql_path()?))
            }
            PropertyPath::OneOrMore(inner_path) => {
                Ok(format!("({})+", inner_path.to_sparql_path()?))
            }
            PropertyPath::ZeroOrOne(inner_path) => {
                Ok(format!("({})?", inner_path.to_sparql_path()?))
            }
        }
    }

    /// Check if this path can be efficiently represented as a SPARQL property path
    pub fn can_use_sparql_path(&self) -> bool {
        match self {
            PropertyPath::Predicate(_) => true,
            PropertyPath::Inverse(inner) => inner.can_use_sparql_path(),
            PropertyPath::Sequence(paths) => {
                paths.len() <= 5 && paths.iter().all(|p| p.can_use_sparql_path())
            }
            PropertyPath::Alternative(paths) => {
                paths.len() <= 10 && paths.iter().all(|p| p.can_use_sparql_path())
            }
            PropertyPath::ZeroOrMore(_) => true, // Can be expensive but supported
            PropertyPath::OneOrMore(_) => true,
            PropertyPath::ZeroOrOne(inner) => inner.can_use_sparql_path(),
        }
    }

    /// Estimate the complexity of this path for optimization
    pub fn complexity(&self) -> usize {
        match self {
            PropertyPath::Predicate(_) => 1,
            PropertyPath::Inverse(path) => path.complexity() + 1,
            PropertyPath::Sequence(paths) => {
                paths.iter().map(|p| p.complexity()).sum::<usize>() + 1
            }
            PropertyPath::Alternative(paths) => {
                paths.iter().map(|p| p.complexity()).max().unwrap_or(0) + 1
            }
            PropertyPath::ZeroOrMore(path) => path.complexity() * 10, // High complexity due to recursion
            PropertyPath::OneOrMore(path) => path.complexity() * 8,
            PropertyPath::ZeroOrOne(path) => path.complexity() + 1,
        }
    }
}

/// Property path evaluator for finding values along paths
#[derive(Debug)]
pub struct PropertyPathEvaluator {
    /// Cache for path evaluation results
    cache: HashMap<String, Vec<Term>>,

    /// Maximum recursion depth for cyclic paths
    max_depth: usize,

    /// Maximum number of intermediate results to track
    max_intermediate_results: usize,
}

impl PropertyPathEvaluator {
    /// Create a new property path evaluator
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            max_depth: 50,
            max_intermediate_results: 10000,
        }
    }

    /// Create a new evaluator with custom limits
    pub fn with_limits(max_depth: usize, max_intermediate_results: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_depth,
            max_intermediate_results,
        }
    }

    /// Evaluate a property path from a starting node
    pub fn evaluate_path(
        &mut self,
        store: &dyn Store,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let cache_key = self.create_cache_key(start_node, path, graph_name);

        // Check cache first
        if let Some(cached_result) = self.cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }

        // Try to use optimized SPARQL property path query first
        let result = if path.can_use_sparql_path() {
            match self.evaluate_path_with_sparql(store, start_node, path, graph_name) {
                Ok(results) => results,
                Err(e) => {
                    tracing::debug!(
                        "SPARQL property path failed, falling back to programmatic evaluation: {}",
                        e
                    );
                    self.evaluate_path_impl(store, start_node, path, graph_name, 0)?
                }
            }
        } else {
            self.evaluate_path_impl(store, start_node, path, graph_name, 0)?
        };

        // Cache the result
        self.cache.insert(cache_key, result.clone());

        Ok(result)
    }

    /// Evaluate multiple paths from the same starting node
    pub fn evaluate_multiple_paths(
        &mut self,
        store: &dyn Store,
        start_node: &Term,
        paths: &[PropertyPath],
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut all_values = HashSet::new();

        for path in paths {
            let values = self.evaluate_path(store, start_node, path, graph_name)?;
            all_values.extend(values);
        }

        Ok(all_values.into_iter().collect())
    }

    /// Evaluate a property path using optimized SPARQL property path query
    fn evaluate_path_with_sparql(
        &self,
        store: &dyn Store,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let sparql_path = path.to_sparql_path()?;
        let mut values = Vec::new();

        // Generate optimized SPARQL query using property paths
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

        // Execute the optimized SPARQL query
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
                    "SPARQL property path query failed: {}",
                    e
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

    /// Internal implementation of path evaluation
    fn evaluate_path_impl(
        &self,
        store: &dyn Store,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        // Check recursion depth
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

    /// Evaluate a simple predicate path
    fn evaluate_predicate(
        &self,
        store: &dyn Store,
        start_node: &Term,
        predicate: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut values = Vec::new();

        // Create a SPARQL query to find all values connected via the predicate
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

        // Execute the SPARQL query using oxirs-core query engine
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
                // Fallback to direct store querying
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
                // For inverse of simple predicate, find subjects where start_node is object
                self.evaluate_inverse_predicate(store, start_node, predicate, graph_name)
            }
            _ => {
                // For complex paths, we need to find all nodes that reach start_node via the path
                self.evaluate_complex_inverse(store, start_node, inner_path, graph_name, depth)
            }
        }
    }

    /// Evaluate inverse of a simple predicate
    fn evaluate_inverse_predicate(
        &self,
        store: &dyn Store,
        start_node: &Term,
        predicate: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut values = Vec::new();

        // Create a SPARQL query to find all values that connect to start_node via the predicate
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

        // Execute the SPARQL query using oxirs-core query engine
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
                // Fallback to direct store querying
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

    /// Evaluate inverse of a complex path
    fn evaluate_complex_inverse(
        &self,
        store: &dyn Store,
        start_node: &Term,
        inner_path: &PropertyPath,
        graph_name: Option<&str>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        // This is computationally expensive - we need to find all nodes that can reach start_node
        // We'll implement this using a recursive approach with caching

        let mut result = Vec::new();
        let mut candidates: HashSet<Term> = HashSet::new();

        // Strategy: Find potential candidates and test if they reach start_node via the path
        // This is expensive but necessary for correctness

        // First, collect all potential starting nodes from the graph
        let candidate_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?candidate WHERE {{
                    GRAPH <{}> {{
                        ?candidate ?p ?o .
                    }}
                }}
                LIMIT 1000
            "#,
                graph
            )
        } else {
            "SELECT DISTINCT ?candidate WHERE { ?candidate ?p ?o . } LIMIT 1000".to_string()
        };

        // Execute query to get candidates
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
                // Use fallback: get candidates from direct store query
                candidates = self.get_candidates_direct(store, graph_name)?;
            }
        }

        tracing::debug!(
            "Testing {} candidates for complex inverse path",
            candidates.len()
        );

        // For each candidate, test if it reaches start_node via the inner path
        for candidate in candidates {
            // Limit recursion to prevent infinite loops
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
                    // Continue with other candidates
                }
            }

            // Limit result size to prevent memory issues
            if result.len() > 100 {
                tracing::warn!("Limiting complex inverse path results to 100 items");
                break;
            }
        }

        tracing::debug!("Complex inverse path found {} results", result.len());
        Ok(result)
    }

    /// Evaluate a sequence path (path1 / path2 / ...)
    fn evaluate_sequence(
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

                // Check if we exceed intermediate results limit
                if next_nodes.len() > self.max_intermediate_results {
                    return Err(ShaclError::PropertyPath(format!(
                        "Too many intermediate results ({}), limit is {}",
                        next_nodes.len(),
                        self.max_intermediate_results
                    )));
                }
            }

            current_nodes = next_nodes;

            // If no results at any step, the sequence fails
            if current_nodes.is_empty() {
                break;
            }
        }

        Ok(current_nodes)
    }

    /// Evaluate an alternative path (path1 | path2 | ...)
    fn evaluate_alternative(
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
    fn evaluate_zero_or_more(
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

        // Include the starting node (zero iterations)
        result.insert(start_node.clone());
        queue.push_back(start_node.clone());
        visited.insert(start_node.clone());

        while let Some(current) = queue.pop_front() {
            // Check limits
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
    fn evaluate_one_or_more(
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

        // Start with one iteration (don't include starting node)
        let initial_values =
            self.evaluate_path_impl(store, start_node, inner_path, graph_name, depth + 1)?;

        for value in initial_values {
            result.insert(value.clone());
            queue.push_back(value.clone());
            visited.insert(value);
        }

        while let Some(current) = queue.pop_front() {
            // Check limits
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

    /// Evaluate a zero-or-one path (path?)
    fn evaluate_zero_or_one(
        &self,
        store: &dyn Store,
        start_node: &Term,
        inner_path: &PropertyPath,
        graph_name: Option<&str>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        let mut result = vec![start_node.clone()]; // Include starting node (zero iterations)

        let path_values =
            self.evaluate_path_impl(store, start_node, inner_path, graph_name, depth + 1)?;
        result.extend(path_values);

        // Remove duplicates
        let unique_result: HashSet<_> = result.into_iter().collect();
        Ok(unique_result.into_iter().collect())
    }

    /// Execute a path query using oxirs-core query engine
    fn execute_path_query(
        &self,
        store: &dyn Store,
        query: &str,
    ) -> Result<oxirs_core::query::QueryResult> {
        use oxirs_core::query::QueryEngine;

        let query_engine = QueryEngine::new();

        tracing::debug!("Executing path query: {}", query);

        // Parse the query string into a Query object
        let parser = oxirs_core::query::parser::SparqlParser::new();
        let parsed_query = parser
            .parse(query)
            .map_err(|e| ShaclError::PropertyPath(format!("Query parsing failed: {}", e)))?;

        let result = query_engine
            .execute_query(&parsed_query, store)
            .map_err(|e| ShaclError::PropertyPath(format!("Path query execution failed: {}", e)))?;

        Ok(result)
    }

    /// Fallback method to evaluate predicate using direct store queries
    fn evaluate_predicate_direct(
        &self,
        store: &dyn Store,
        start_node: &Term,
        predicate: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::model::{GraphName, Object, Predicate, Subject};

        let subject = match start_node {
            Term::NamedNode(node) => Subject::NamedNode(node.clone()),
            Term::BlankNode(node) => Subject::BlankNode(node.clone()),
            _ => {
                return Err(ShaclError::PropertyPath(
                    "Invalid subject term for predicate path".to_string(),
                ))
            }
        };

        let predicate_term = Predicate::NamedNode(predicate.clone());

        let graph_filter = match graph_name {
            Some(g) => {
                Some(GraphName::NamedNode(NamedNode::new(g).map_err(|e| {
                    ShaclError::Core(OxirsError::Parse(e.to_string()))
                })?))
            }
            None => None,
        };

        let quads = store
            .find_quads(
                Some(&subject),
                Some(&predicate_term),
                None, // Any object
                graph_filter.as_ref(),
            )
            .map_err(|e| ShaclError::Core(e))?;

        let values: Vec<Term> = quads
            .into_iter()
            .map(|quad| Term::from(quad.object().clone()))
            .collect();

        tracing::debug!(
            "Direct store query found {} values for predicate path",
            values.len()
        );
        Ok(values)
    }

    /// Fallback method to evaluate inverse predicate using direct store queries
    fn evaluate_inverse_predicate_direct(
        &self,
        store: &dyn Store,
        start_node: &Term,
        predicate: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::model::{GraphName, Object, Predicate, Subject};

        let object = match start_node {
            Term::NamedNode(node) => Object::NamedNode(node.clone()),
            Term::BlankNode(node) => Object::BlankNode(node.clone()),
            Term::Literal(literal) => Object::Literal(literal.clone()),
            _ => {
                return Err(ShaclError::PropertyPath(
                    "Invalid object term for inverse predicate path".to_string(),
                ))
            }
        };

        let predicate_term = Predicate::NamedNode(predicate.clone());

        let graph_filter = match graph_name {
            Some(g) => {
                Some(GraphName::NamedNode(NamedNode::new(g).map_err(|e| {
                    ShaclError::Core(OxirsError::Parse(e.to_string()))
                })?))
            }
            None => None,
        };

        let quads = store
            .find_quads(
                None, // Any subject
                Some(&predicate_term),
                Some(&object),
                graph_filter.as_ref(),
            )
            .map_err(|e| ShaclError::Core(e))?;

        let values: Vec<Term> = quads
            .into_iter()
            .map(|quad| Term::from(quad.subject().clone()))
            .collect();

        tracing::debug!(
            "Direct store query found {} values for inverse predicate path",
            values.len()
        );
        Ok(values)
    }

    /// Fallback method to get candidate nodes using direct store queries
    fn get_candidates_direct(
        &self,
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<HashSet<Term>> {
        use oxirs_core::model::GraphName;

        let mut candidates = HashSet::new();

        let graph_filter = match graph_name {
            Some(g) => {
                Some(GraphName::NamedNode(NamedNode::new(g).map_err(|e| {
                    ShaclError::Core(OxirsError::Parse(e.to_string()))
                })?))
            }
            None => None,
        };

        // Get all subjects from the store (limited to prevent memory issues)
        let quads = store
            .find_quads(
                None, // Any subject
                None, // Any predicate
                None, // Any object
                graph_filter.as_ref(),
            )
            .map_err(|e| ShaclError::Core(e))?;

        // Collect unique subjects, limited to prevent memory explosion
        for quad in quads.into_iter().take(1000) {
            candidates.insert(Term::from(quad.subject().clone()));
        }

        tracing::debug!(
            "Direct store query found {} candidate nodes",
            candidates.len()
        );
        Ok(candidates)
    }

    /// Create a cache key for path evaluation
    fn create_cache_key(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        start_node.as_str().hash(&mut hasher);
        path.hash(&mut hasher);
        graph_name.hash(&mut hasher);

        format!("path_eval_{}", hasher.finish())
    }

    /// Clear the evaluation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> PathCacheStats {
        PathCacheStats {
            entries: self.cache.len(),
            total_values: self.cache.values().map(|v| v.len()).sum(),
        }
    }

    /// Set maximum recursion depth
    pub fn set_max_depth(&mut self, max_depth: usize) {
        self.max_depth = max_depth;
    }

    /// Set maximum intermediate results limit
    pub fn set_max_intermediate_results(&mut self, max_results: usize) {
        self.max_intermediate_results = max_results;
    }

    /// Analyze and optimize a property path for better performance
    pub fn optimize_path(&self, path: &PropertyPath) -> OptimizedPropertyPath {
        let complexity = path.complexity();
        let can_use_sparql = path.can_use_sparql_path();
        let estimated_cost = self.estimate_path_cost(path);

        let optimization_strategy = if can_use_sparql && complexity <= 5 {
            PathOptimizationStrategy::SparqlPath
        } else if complexity > 50 {
            PathOptimizationStrategy::Programmatic
        } else {
            PathOptimizationStrategy::Hybrid
        };

        OptimizedPropertyPath {
            original_path: path.clone(),
            optimization_strategy,
            estimated_complexity: complexity,
            estimated_cost,
            can_use_sparql_path: can_use_sparql,
        }
    }

    /// Generate an optimized SPARQL query for property path evaluation
    pub fn generate_optimized_sparql_query(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        let optimized_path = self.optimize_path(path);

        match optimized_path.optimization_strategy {
            PathOptimizationStrategy::SparqlPath => self.generate_native_sparql_path_query(
                start_node,
                path,
                graph_name,
                optimization_hints,
            ),
            PathOptimizationStrategy::Programmatic => {
                // For programmatic evaluation, generate query for simple fallback
                self.generate_fallback_sparql_query(start_node, path, graph_name)
            }
            PathOptimizationStrategy::Hybrid => {
                self.generate_hybrid_sparql_query(start_node, path, graph_name, optimization_hints)
            }
        }
    }

    /// Generate native SPARQL property path query
    fn generate_native_sparql_path_query(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        let start_term = format_term_for_sparql(start_node)?;

        // Build query with optimizations
        let mut query_parts = Vec::new();

        // Add prefixes if needed
        query_parts.push(self.generate_common_prefixes());

        // Add main SELECT clause with optimizations
        let select_clause = if optimization_hints.parallel_threshold > 0 {
            "SELECT DISTINCT ?value # HINT: PARALLEL"
        } else {
            "SELECT DISTINCT ?value"
        };
        query_parts.push(select_clause.to_string());

        // Special handling for sequence paths - use intermediate variables for better debugging
        let where_clause = if let PropertyPath::Sequence(paths) = path {
            return self.generate_sequence_sparql_query(
                start_node,
                paths,
                graph_name,
                optimization_hints,
            );
        } else if matches!(path, PropertyPath::Alternative(_)) {
            self.generate_union_based_alternative_query(start_node, path, graph_name)?
        } else {
            let sparql_path = path.to_sparql_path()?;
            if let Some(graph) = graph_name {
                format!(
                    "WHERE {{\n  GRAPH <{}> {{\n    {} {} ?value .\n  }}\n}}",
                    graph, start_term, sparql_path
                )
            } else {
                format!("WHERE {{\n  {} {} ?value .\n}}", start_term, sparql_path)
            }
        };
        query_parts.push(where_clause);

        // Add optimization hints
        if path.complexity() > 10 {
            query_parts.push("# HINT: USE_INDEX(property_path_index)".to_string());
        }

        // Add ordering for deterministic results
        query_parts.push("ORDER BY ?value".to_string());

        // Add limits if configured
        if optimization_hints.max_intermediate_results < usize::MAX {
            query_parts.push(format!(
                "LIMIT {}",
                optimization_hints.max_intermediate_results
            ));
        }

        Ok(query_parts.join("\n"))
    }

    /// Generate UNION-based query for alternative paths
    fn generate_union_based_alternative_query(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
    ) -> Result<String> {
        if let PropertyPath::Alternative(paths) = path {
            let start_term = format_term_for_sparql(start_node)?;
            let mut union_parts = Vec::new();

            for alternative_path in paths {
                let path_sparql = alternative_path.to_sparql_path()?;
                let triple_pattern = if let Some(graph) = graph_name {
                    format!(
                        "    GRAPH <{}> {{ {} {} ?value . }}",
                        graph, start_term, path_sparql
                    )
                } else {
                    format!("    {} {} ?value .", start_term, path_sparql)
                };
                union_parts.push(format!("  {{\n{}\n  }}", triple_pattern));
            }

            Ok(format!("WHERE {{\n{}\n}}", union_parts.join("\n  UNION\n")))
        } else {
            Err(ShaclError::PropertyPath(
                "Expected alternative path for UNION query generation".to_string(),
            ))
        }
    }

    /// Generate hybrid SPARQL query that combines multiple strategies
    fn generate_hybrid_sparql_query(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        match path {
            PropertyPath::Sequence(paths) => {
                // For sequences, generate optimized multi-step query
                self.generate_sequence_sparql_query(
                    start_node,
                    paths,
                    graph_name,
                    optimization_hints,
                )
            }
            PropertyPath::Alternative(paths) => {
                // For alternatives, generate UNION query
                self.generate_union_sparql_query(start_node, paths, graph_name, optimization_hints)
            }
            PropertyPath::ZeroOrMore(_) | PropertyPath::OneOrMore(_) => {
                // For recursive paths, generate specialized recursive query
                self.generate_recursive_sparql_query(
                    start_node,
                    path,
                    graph_name,
                    optimization_hints,
                )
            }
            _ => {
                // For other paths, use native SPARQL path
                self.generate_native_sparql_path_query(
                    start_node,
                    path,
                    graph_name,
                    optimization_hints,
                )
            }
        }
    }

    /// Generate optimized SPARQL query for sequence paths
    fn generate_sequence_sparql_query(
        &self,
        start_node: &Term,
        paths: &[PropertyPath],
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        let start_term = format_term_for_sparql(start_node)?;
        let mut query_parts = Vec::new();

        // Add prefixes
        query_parts.push(self.generate_common_prefixes());

        // Add SELECT clause
        query_parts.push("SELECT DISTINCT ?value".to_string());

        // Generate intermediate variables
        let mut variables = vec!["?start".to_string()];
        for i in 1..paths.len() {
            variables.push(format!("?inter{}", i));
        }
        variables.push("?value".to_string());

        // Build WHERE clause with sequence
        let mut where_parts = Vec::new();
        where_parts.push(format!("BIND({} AS ?start)", start_term));

        for (i, path) in paths.iter().enumerate() {
            let from_var = &variables[i];
            let to_var = &variables[i + 1];
            let sparql_path = path.to_sparql_path()?;

            let triple_pattern = if let Some(graph) = graph_name {
                format!(
                    "GRAPH <{}> {{ {} {} {} . }}",
                    graph, from_var, sparql_path, to_var
                )
            } else {
                format!("{} {} {} .", from_var, sparql_path, to_var)
            };

            where_parts.push(triple_pattern);
        }

        query_parts.push(format!("WHERE {{\n  {}\n}}", where_parts.join("\n  ")));

        // Add optimizations
        if paths.len() > 3 {
            query_parts.push("# HINT: OPTIMIZE_JOIN_ORDER".to_string());
        }

        query_parts.push("ORDER BY ?value".to_string());

        if optimization_hints.max_intermediate_results < usize::MAX {
            query_parts.push(format!(
                "LIMIT {}",
                optimization_hints.max_intermediate_results
            ));
        }

        Ok(query_parts.join("\n"))
    }

    /// Generate optimized SPARQL query for alternative paths using UNION
    fn generate_union_sparql_query(
        &self,
        start_node: &Term,
        paths: &[PropertyPath],
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        let start_term = format_term_for_sparql(start_node)?;
        let mut query_parts = Vec::new();

        // Add prefixes
        query_parts.push(self.generate_common_prefixes());

        // Add SELECT clause
        query_parts.push("SELECT DISTINCT ?value".to_string());

        // Build UNION query for alternatives
        let mut union_parts = Vec::new();

        for path in paths {
            let sparql_path = path.to_sparql_path()?;
            let union_part = if let Some(graph) = graph_name {
                format!(
                    "{{ GRAPH <{}> {{ {} {} ?value . }} }}",
                    graph, start_term, sparql_path
                )
            } else {
                format!("{{ {} {} ?value . }}", start_term, sparql_path)
            };
            union_parts.push(union_part);
        }

        query_parts.push(format!(
            "WHERE {{\n  {}\n}}",
            union_parts.join("\n  UNION\n  ")
        ));

        // Add optimizations for UNION queries
        if paths.len() > 5 {
            query_parts.push("# HINT: OPTIMIZE_UNION".to_string());
        }

        query_parts.push("ORDER BY ?value".to_string());

        if optimization_hints.max_intermediate_results < usize::MAX {
            query_parts.push(format!(
                "LIMIT {}",
                optimization_hints.max_intermediate_results
            ));
        }

        Ok(query_parts.join("\n"))
    }

    /// Generate specialized SPARQL query for recursive paths
    fn generate_recursive_sparql_query(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        let start_term = format_term_for_sparql(start_node)?;
        let mut query_parts = Vec::new();

        // Add prefixes
        query_parts.push(self.generate_common_prefixes());

        // Add SELECT clause
        query_parts.push("SELECT DISTINCT ?value".to_string());

        // Generate recursive query based on path type
        let where_clause = match path {
            PropertyPath::ZeroOrMore(inner_path) => {
                let inner_sparql = inner_path.to_sparql_path()?;

                if let Some(graph) = graph_name {
                    format!(
                        "WHERE {{\n  {{\n    BIND({} AS ?value)\n  }}\n  UNION\n  {{\n    GRAPH <{}> {{\n      {} {}+ ?value .\n    }}\n  }}\n}}",
                        start_term, graph, start_term, inner_sparql
                    )
                } else {
                    format!(
                        "WHERE {{\n  {{\n    BIND({} AS ?value)\n  }}\n  UNION\n  {{\n    {} {}+ ?value .\n  }}\n}}",
                        start_term, start_term, inner_sparql
                    )
                }
            }
            PropertyPath::OneOrMore(inner_path) => {
                let inner_sparql = inner_path.to_sparql_path()?;

                if let Some(graph) = graph_name {
                    format!(
                        "WHERE {{\n  GRAPH <{}> {{\n    {} {}+ ?value .\n  }}\n}}",
                        graph, start_term, inner_sparql
                    )
                } else {
                    format!("WHERE {{\n  {} {}+ ?value .\n}}", start_term, inner_sparql)
                }
            }
            _ => {
                return Err(ShaclError::PropertyPath(
                    "Invalid recursive path type".to_string(),
                ));
            }
        };

        query_parts.push(where_clause);

        // Add recursion safety limits
        query_parts.push(format!(
            "# HINT: MAX_RECURSION_DEPTH {}",
            optimization_hints.max_recursion_depth
        ));

        query_parts.push("ORDER BY ?value".to_string());

        // Limit results for recursive queries to prevent explosion
        let recursive_limit = optimization_hints.max_intermediate_results.min(1000);
        query_parts.push(format!("LIMIT {}", recursive_limit));

        Ok(query_parts.join("\n"))
    }

    /// Generate fallback SPARQL query for programmatic evaluation
    fn generate_fallback_sparql_query(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
    ) -> Result<String> {
        // For complex paths that can't be efficiently expressed in SPARQL,
        // generate a simple query to get initial candidates
        let start_term = format_term_for_sparql(start_node)?;

        let query = if let Some(graph) = graph_name {
            format!(
                "# Fallback query for complex path evaluation\nSELECT DISTINCT ?candidate WHERE {{\n  GRAPH <{}> {{\n    {} ?p ?candidate .\n  }}\n}}\nLIMIT 100",
                graph, start_term
            )
        } else {
            format!(
                "# Fallback query for complex path evaluation\nSELECT DISTINCT ?candidate WHERE {{\n  {} ?p ?candidate .\n}}\nLIMIT 100",
                start_term
            )
        };

        Ok(query)
    }

    /// Generate common SPARQL prefixes
    fn generate_common_prefixes(&self) -> String {
        r#"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX sh: <http://www.w3.org/ns/shacl#>"#
            .to_string()
    }

    /// Generate query plan for complex property path evaluation
    pub fn generate_query_plan(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<PropertyPathQueryPlan> {
        let optimized_path = self.optimize_path(path);
        let query =
            self.generate_optimized_sparql_query(start_node, path, graph_name, optimization_hints)?;

        let execution_strategy = match optimized_path.optimization_strategy {
            PathOptimizationStrategy::SparqlPath => PathExecutionStrategy::DirectSparql,
            PathOptimizationStrategy::Programmatic => PathExecutionStrategy::Programmatic,
            PathOptimizationStrategy::Hybrid => PathExecutionStrategy::HybridExecution,
        };

        Ok(PropertyPathQueryPlan {
            query,
            execution_strategy,
            estimated_cost: optimized_path.estimated_cost,
            estimated_complexity: optimized_path.estimated_complexity,
            optimization_hints: optimization_hints.clone(),
            cache_key: self.create_cache_key(start_node, path, graph_name),
        })
    }

    /// Validate a property path for correctness and performance
    pub fn validate_property_path(&self, path: &PropertyPath) -> PathValidationResult {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        // Check complexity
        let complexity = path.complexity();
        if complexity > 100 {
            warnings.push(format!(
                "High complexity path ({}), consider simplification",
                complexity
            ));
        }

        // Check for potential performance issues
        match path {
            PropertyPath::ZeroOrMore(_) => {
                warnings.push("Zero-or-more paths can be expensive on large datasets".to_string());
            }
            PropertyPath::OneOrMore(_) => {
                warnings.push("One-or-more paths can be expensive on large datasets".to_string());
            }
            PropertyPath::Sequence(paths) => {
                if paths.len() > 10 {
                    warnings.push(format!(
                        "Long sequence path ({} steps), consider breaking into smaller parts",
                        paths.len()
                    ));
                }
                if paths.is_empty() {
                    errors.push("Empty sequence path is invalid".to_string());
                }
            }
            PropertyPath::Alternative(paths) => {
                if paths.len() > 20 {
                    warnings.push(format!(
                        "Many alternatives ({}), consider grouping",
                        paths.len()
                    ));
                }
                if paths.len() < 2 {
                    errors.push("Alternative path must have at least 2 alternatives".to_string());
                }
            }
            _ => {}
        }

        // Check SPARQL compatibility
        if !path.can_use_sparql_path() {
            warnings.push(
                "Path cannot use native SPARQL property paths, will use programmatic evaluation"
                    .to_string(),
            );
        }

        // Check for recursive structures that might cause infinite loops
        if self.has_potential_cycles(path) {
            warnings.push(
                "Path has potential for infinite recursion, ensure proper depth limits".to_string(),
            );
        }

        PathValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            complexity,
            can_use_sparql: path.can_use_sparql_path(),
            estimated_cost: self.estimate_path_cost(path),
        }
    }

    /// Check if a path has potential for infinite cycles
    fn has_potential_cycles(&self, path: &PropertyPath) -> bool {
        match path {
            PropertyPath::ZeroOrMore(_) | PropertyPath::OneOrMore(_) => true,
            PropertyPath::Sequence(paths) | PropertyPath::Alternative(paths) => {
                paths.iter().any(|p| self.has_potential_cycles(p))
            }
            PropertyPath::Inverse(inner) | PropertyPath::ZeroOrOne(inner) => {
                self.has_potential_cycles(inner)
            }
            PropertyPath::Predicate(_) => false,
        }
    }

    /// Estimate the cost of evaluating a property path
    fn estimate_path_cost(&self, path: &PropertyPath) -> f64 {
        match path {
            PropertyPath::Predicate(_) => 1.0,
            PropertyPath::Inverse(inner) => inner.complexity() as f64 * 1.5,
            PropertyPath::Sequence(paths) => {
                paths
                    .iter()
                    .map(|p| self.estimate_path_cost(p))
                    .sum::<f64>()
                    * 1.2
            }
            PropertyPath::Alternative(paths) => {
                paths
                    .iter()
                    .map(|p| self.estimate_path_cost(p))
                    .sum::<f64>()
                    * 0.8
            }
            PropertyPath::ZeroOrMore(_) => 100.0, // Very expensive
            PropertyPath::OneOrMore(_) => 80.0,   // Expensive
            PropertyPath::ZeroOrOne(inner) => self.estimate_path_cost(inner) * 1.1,
        }
    }

    /// Get performance statistics for path evaluation
    pub fn get_performance_stats(&self) -> PropertyPathStats {
        PropertyPathStats {
            cache_entries: self.cache.len(),
            total_cached_results: self.cache.values().map(|v| v.len()).sum(),
        }
    }
}

impl Default for PropertyPathEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about property path evaluation cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathCacheStats {
    pub entries: usize,
    pub total_values: usize,
}

/// Optimization strategies for property path evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PathOptimizationStrategy {
    /// Use native SPARQL property path queries
    SparqlPath,
    /// Use programmatic evaluation
    Programmatic,
    /// Use hybrid approach (SPARQL for simple parts, programmatic for complex)
    Hybrid,
}

/// An optimized property path with analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedPropertyPath {
    pub original_path: PropertyPath,
    pub optimization_strategy: PathOptimizationStrategy,
    pub estimated_complexity: usize,
    pub estimated_cost: f64,
    pub can_use_sparql_path: bool,
}

/// Performance statistics for property path evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyPathStats {
    pub cache_entries: usize,
    pub total_cached_results: usize,
}

/// Property path optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathOptimizationHints {
    /// Cache simple predicate path results
    pub cache_simple_paths: bool,

    /// Cache complex path results
    pub cache_complex_paths: bool,

    /// Maximum cache size for path results
    pub max_cache_size: usize,

    /// Parallel evaluation threshold
    pub parallel_threshold: usize,

    /// Maximum recursion depth for cyclic paths
    pub max_recursion_depth: usize,

    /// Maximum intermediate results
    pub max_intermediate_results: usize,
}

impl Default for PathOptimizationHints {
    fn default() -> Self {
        Self {
            cache_simple_paths: true,
            cache_complex_paths: false, // Complex paths change frequently
            max_cache_size: 5000,
            parallel_threshold: 100,
            max_recursion_depth: 50,
            max_intermediate_results: 10000,
        }
    }
}

/// Property path validation context
#[derive(Debug, Clone)]
pub struct PathValidationContext {
    /// Current recursion depth
    pub depth: usize,

    /// Visited nodes (for cycle detection)
    pub visited: HashSet<Term>,

    /// Path being evaluated
    pub current_path: PropertyPath,

    /// Performance statistics
    pub stats: PathEvaluationStats,
}

impl PathValidationContext {
    pub fn new(path: PropertyPath) -> Self {
        Self {
            depth: 0,
            visited: HashSet::new(),
            current_path: path,
            stats: PathEvaluationStats::default(),
        }
    }
}

/// Property path evaluation performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathEvaluationStats {
    pub total_evaluations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub total_values_found: usize,
    pub avg_values_per_evaluation: f64,
    pub max_recursion_depth_reached: usize,
}

/// Execution strategy for property path evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PathExecutionStrategy {
    /// Execute using direct SPARQL property path queries
    DirectSparql,
    /// Execute using programmatic evaluation
    Programmatic,
    /// Execute using hybrid approach
    HybridExecution,
}

/// Query plan for property path evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyPathQueryPlan {
    /// Generated SPARQL query
    pub query: String,
    /// Execution strategy to use
    pub execution_strategy: PathExecutionStrategy,
    /// Estimated execution cost
    pub estimated_cost: f64,
    /// Estimated complexity
    pub estimated_complexity: usize,
    /// Optimization hints applied
    pub optimization_hints: PathOptimizationHints,
    /// Cache key for this plan
    pub cache_key: String,
}

/// Validation result for property paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathValidationResult {
    /// Whether the path is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Path complexity
    pub complexity: usize,
    /// Whether path can use SPARQL
    pub can_use_sparql: bool,
    /// Estimated cost
    pub estimated_cost: f64,
}

impl PathEvaluationStats {
    pub fn record_evaluation(&mut self, values_found: usize, cache_hit: bool, depth: usize) {
        self.total_evaluations += 1;
        self.total_values_found += values_found;

        if cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }

        if depth > self.max_recursion_depth_reached {
            self.max_recursion_depth_reached = depth;
        }

        self.avg_values_per_evaluation =
            self.total_values_found as f64 / self.total_evaluations as f64;
    }

    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_evaluations as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_path_creation() {
        let predicate = NamedNode::new("http://example.org/knows").unwrap();
        let path = PropertyPath::predicate(predicate.clone());

        assert!(path.is_predicate());
        assert_eq!(path.as_predicate(), Some(&predicate));
        assert!(!path.is_complex());
        assert_eq!(path.complexity(), 1);
    }

    #[test]
    fn test_inverse_path() {
        let predicate = NamedNode::new("http://example.org/knows").unwrap();
        let path = PropertyPath::inverse(PropertyPath::predicate(predicate));

        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 2);
    }

    #[test]
    fn test_sequence_path() {
        let pred1 = NamedNode::new("http://example.org/knows").unwrap();
        let pred2 = NamedNode::new("http://example.org/friend").unwrap();

        let path = PropertyPath::sequence(vec![
            PropertyPath::predicate(pred1),
            PropertyPath::predicate(pred2),
        ]);

        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 3); // 1 + 1 + 1 for sequence overhead
    }

    #[test]
    fn test_zero_or_more_path() {
        let predicate = NamedNode::new("http://example.org/knows").unwrap();
        let path = PropertyPath::zero_or_more(PropertyPath::predicate(predicate));

        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 10); // High complexity due to recursion
    }

    #[test]
    fn test_alternative_path() {
        let pred1 = NamedNode::new("http://example.org/knows").unwrap();
        let pred2 = NamedNode::new("http://example.org/friend").unwrap();

        let path = PropertyPath::alternative(vec![
            PropertyPath::predicate(pred1),
            PropertyPath::predicate(pred2),
        ]);

        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 2); // max(1, 1) + 1 for alternative overhead
    }

    #[test]
    fn test_path_evaluator_creation() {
        let evaluator = PropertyPathEvaluator::new();
        assert_eq!(evaluator.max_depth, 50);
        assert_eq!(evaluator.max_intermediate_results, 10000);

        let custom_evaluator = PropertyPathEvaluator::with_limits(100, 5000);
        assert_eq!(custom_evaluator.max_depth, 100);
        assert_eq!(custom_evaluator.max_intermediate_results, 5000);
    }

    #[test]
    fn test_path_evaluation_stats() {
        let mut stats = PathEvaluationStats::default();

        stats.record_evaluation(5, false, 2); // cache miss, depth 2
        stats.record_evaluation(3, true, 1); // cache hit, depth 1
        stats.record_evaluation(7, false, 3); // cache miss, depth 3

        assert_eq!(stats.total_evaluations, 3);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 2);
        assert_eq!(stats.total_values_found, 15);
        assert_eq!(stats.avg_values_per_evaluation, 5.0);
        assert_eq!(stats.max_recursion_depth_reached, 3);
        assert_eq!(stats.cache_hit_rate(), 1.0 / 3.0);
    }

    #[test]
    fn test_sparql_path_query_generation() {
        let evaluator = PropertyPathEvaluator::new();
        let start_node = Term::NamedNode(NamedNode::new("http://example.org/person1").unwrap());
        let predicate = NamedNode::new("http://example.org/knows").unwrap();
        let path = PropertyPath::predicate(predicate);
        let hints = PathOptimizationHints::default();

        let query = evaluator
            .generate_optimized_sparql_query(&start_node, &path, None, &hints)
            .unwrap();

        assert!(query.contains("SELECT DISTINCT ?value"));
        assert!(query.contains("<http://example.org/person1>"));
        assert!(query.contains("<http://example.org/knows>"));
        assert!(query.contains("PREFIX"));
    }

    #[test]
    fn test_sequence_path_query_generation() {
        let evaluator = PropertyPathEvaluator::new();
        let start_node = Term::NamedNode(NamedNode::new("http://example.org/person1").unwrap());

        let pred1 = NamedNode::new("http://example.org/knows").unwrap();
        let pred2 = NamedNode::new("http://example.org/friend").unwrap();

        let path = PropertyPath::sequence(vec![
            PropertyPath::predicate(pred1),
            PropertyPath::predicate(pred2),
        ]);

        let hints = PathOptimizationHints::default();

        let query = evaluator
            .generate_optimized_sparql_query(&start_node, &path, None, &hints)
            .unwrap();

        assert!(query.contains("SELECT DISTINCT ?value"));
        assert!(query.contains("?start"));
        assert!(query.contains("?inter1"));
        assert!(query.contains("BIND"));
    }

    #[test]
    fn test_alternative_path_query_generation() {
        let evaluator = PropertyPathEvaluator::new();
        let start_node = Term::NamedNode(NamedNode::new("http://example.org/person1").unwrap());

        let pred1 = NamedNode::new("http://example.org/knows").unwrap();
        let pred2 = NamedNode::new("http://example.org/friend").unwrap();

        let path = PropertyPath::alternative(vec![
            PropertyPath::predicate(pred1),
            PropertyPath::predicate(pred2),
        ]);

        let hints = PathOptimizationHints::default();

        let query = evaluator
            .generate_optimized_sparql_query(&start_node, &path, None, &hints)
            .unwrap();

        assert!(query.contains("SELECT DISTINCT ?value"));
        assert!(query.contains("UNION"));
        assert!(query.contains("<http://example.org/knows>"));
        assert!(query.contains("<http://example.org/friend>"));
    }

    #[test]
    fn test_recursive_path_query_generation() {
        let evaluator = PropertyPathEvaluator::new();
        let start_node = Term::NamedNode(NamedNode::new("http://example.org/person1").unwrap());

        let predicate = NamedNode::new("http://example.org/knows").unwrap();
        let path = PropertyPath::one_or_more(PropertyPath::predicate(predicate));

        let hints = PathOptimizationHints::default();

        let query = evaluator
            .generate_optimized_sparql_query(&start_node, &path, None, &hints)
            .unwrap();

        assert!(query.contains("SELECT DISTINCT ?value"));
        assert!(query.contains("+"));
        assert!(query.contains("MAX_RECURSION_DEPTH"));
    }

    #[test]
    fn test_path_optimization() {
        let evaluator = PropertyPathEvaluator::new();

        // Simple path should use SPARQL
        let simple_path =
            PropertyPath::predicate(NamedNode::new("http://example.org/knows").unwrap());
        let optimized = evaluator.optimize_path(&simple_path);
        assert_eq!(
            optimized.optimization_strategy,
            PathOptimizationStrategy::SparqlPath
        );

        // Complex sequence should use hybrid
        let complex_path = PropertyPath::sequence(vec![
            PropertyPath::predicate(NamedNode::new("http://example.org/knows").unwrap()),
            PropertyPath::zero_or_more(PropertyPath::predicate(
                NamedNode::new("http://example.org/friend").unwrap(),
            )),
        ]);
        let optimized = evaluator.optimize_path(&complex_path);
        assert_eq!(
            optimized.optimization_strategy,
            PathOptimizationStrategy::Hybrid
        );
    }

    #[test]
    fn test_path_validation() {
        let evaluator = PropertyPathEvaluator::new();

        // Valid simple path
        let valid_path =
            PropertyPath::predicate(NamedNode::new("http://example.org/knows").unwrap());
        let result = evaluator.validate_property_path(&valid_path);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());

        // Invalid empty sequence
        let invalid_path = PropertyPath::sequence(vec![]);
        let result = evaluator.validate_property_path(&invalid_path);
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());

        // Path with warnings
        let warning_path = PropertyPath::zero_or_more(PropertyPath::predicate(
            NamedNode::new("http://example.org/knows").unwrap(),
        ));
        let result = evaluator.validate_property_path(&warning_path);
        assert!(result.is_valid);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_query_plan_generation() {
        let evaluator = PropertyPathEvaluator::new();
        let start_node = Term::NamedNode(NamedNode::new("http://example.org/person1").unwrap());
        let path = PropertyPath::predicate(NamedNode::new("http://example.org/knows").unwrap());
        let hints = PathOptimizationHints::default();

        let plan = evaluator
            .generate_query_plan(&start_node, &path, None, &hints)
            .unwrap();

        assert!(!plan.query.is_empty());
        assert_eq!(plan.execution_strategy, PathExecutionStrategy::DirectSparql);
        assert!(plan.estimated_cost > 0.0);
        assert!(!plan.cache_key.is_empty());
    }
}

/// Format a term for use in SPARQL queries
fn format_term_for_sparql(term: &Term) -> Result<String> {
    match term {
        Term::NamedNode(node) => Ok(format!("<{}>", node.as_str())),
        Term::BlankNode(node) => Ok(format!("_:{}", node.as_str())),
        Term::Literal(literal) => {
            // Format literal with proper escaping and datatype/language tags
            let value = literal.value().replace('\\', "\\\\").replace('"', "\\\"");

            let datatype = literal.datatype();
            if datatype.as_str() == "http://www.w3.org/2001/XMLSchema#string" {
                // Simple string literals don't need datatype annotation
                Ok(format!("\"{}\"", value))
            } else {
                Ok(format!("\"{}\"^^<{}>", value, datatype.as_str()))
            }
        }
        Term::Variable(var) => Ok(format!("?{}", var.name())),
        Term::QuotedTriple(_) => Err(ShaclError::PropertyPath(
            "Quoted triples not supported in property path queries".to_string(),
        )),
    }
}
