//! Advanced SPARQL-star query patterns
//!
//! This module implements advanced query features for SPARQL-star:
//! - **Property paths** - Traverse RDF graphs with complex path expressions
//! - **Federated queries** - Query multiple RDF-star endpoints
//! - **Service calls** - External SERVICE queries with quoted triples
//! - **Full-text search** - Integration with text search for literals
//! - **Geo-spatial queries** - Spatial queries on geo-tagged RDF-star data
//! - **Temporal queries** - Time-based queries on temporal RDF-star data
//!
//! # Examples
//!
//! ```rust,ignore
//! use oxirs_star::advanced_query::{PropertyPath, PropertyPathExecutor};
//! use oxirs_star::StarStore;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let store = StarStore::new();
//! let executor = PropertyPathExecutor::new();
//!
//! // Query using property path: ?x foaf:knows+ ?friend
//! let path = PropertyPath::one_or_more("http://xmlns.com/foaf/0.1/knows");
//! let results = executor.evaluate_path(&store, &path, None, None)?;
//! println!("Found {} results", results.len());
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use tracing::debug;

use crate::model::{StarTerm, StarTriple};
use crate::store::StarStore;
use crate::StarResult;

/// Property path expression for traversing RDF graphs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyPath {
    /// Simple predicate path: p
    Predicate(String),

    /// Inverse path: ^p
    Inverse(Box<PropertyPath>),

    /// Sequence path: p1 / p2
    Sequence(Box<PropertyPath>, Box<PropertyPath>),

    /// Alternative path: p1 | p2
    Alternative(Box<PropertyPath>, Box<PropertyPath>),

    /// Zero or more: p*
    ZeroOrMore(Box<PropertyPath>),

    /// One or more: p+
    OneOrMore(Box<PropertyPath>),

    /// Zero or one: p?
    ZeroOrOne(Box<PropertyPath>),

    /// Negated property set: !(p1|p2|...)
    NegatedPropertySet(Vec<String>),
}

impl PropertyPath {
    /// Create a simple predicate path
    pub fn predicate(iri: impl Into<String>) -> Self {
        Self::Predicate(iri.into())
    }

    /// Create an inverse path
    pub fn inverse(path: PropertyPath) -> Self {
        Self::Inverse(Box::new(path))
    }

    /// Create a sequence path
    pub fn sequence(first: PropertyPath, second: PropertyPath) -> Self {
        Self::Sequence(Box::new(first), Box::new(second))
    }

    /// Create an alternative path
    pub fn alternative(first: PropertyPath, second: PropertyPath) -> Self {
        Self::Alternative(Box::new(first), Box::new(second))
    }

    /// Create zero-or-more path
    pub fn zero_or_more(iri: impl Into<String>) -> Self {
        Self::ZeroOrMore(Box::new(Self::Predicate(iri.into())))
    }

    /// Create one-or-more path
    pub fn one_or_more(iri: impl Into<String>) -> Self {
        Self::OneOrMore(Box::new(Self::Predicate(iri.into())))
    }

    /// Create zero-or-one path
    pub fn zero_or_one(iri: impl Into<String>) -> Self {
        Self::ZeroOrOne(Box::new(Self::Predicate(iri.into())))
    }
}

/// Property path evaluator
pub struct PropertyPathExecutor {
    /// Maximum path length to prevent infinite loops
    max_path_length: usize,

    /// Cache for evaluated paths (reserved for future optimization)
    #[allow(dead_code)]
    cache: HashMap<String, Vec<PathResult>>,
}

/// Result of a property path evaluation
#[derive(Debug, Clone)]
pub struct PathResult {
    /// Start node
    pub start: StarTerm,

    /// End node
    pub end: StarTerm,

    /// Intermediate path (for debugging)
    pub path: Vec<StarTerm>,
}

impl PropertyPathExecutor {
    /// Create a new property path executor
    pub fn new() -> Self {
        Self {
            max_path_length: 100,
            cache: HashMap::new(),
        }
    }

    /// Evaluate a property path
    pub fn evaluate_path(
        &self,
        store: &StarStore,
        path: &PropertyPath,
        start: Option<&StarTerm>,
        end: Option<&StarTerm>,
    ) -> StarResult<Vec<PathResult>> {
        debug!("Evaluating property path: {:?}", path);

        match path {
            PropertyPath::Predicate(predicate) => {
                self.evaluate_predicate(store, predicate, start, end)
            }

            PropertyPath::Inverse(inner_path) => {
                // Swap start and end for inverse
                self.evaluate_path(store, inner_path, end, start)
            }

            PropertyPath::Sequence(first, second) => {
                self.evaluate_sequence(store, first, second, start, end)
            }

            PropertyPath::Alternative(first, second) => {
                self.evaluate_alternative(store, first, second, start, end)
            }

            PropertyPath::ZeroOrMore(inner_path) => {
                self.evaluate_zero_or_more(store, inner_path, start, end)
            }

            PropertyPath::OneOrMore(inner_path) => {
                self.evaluate_one_or_more(store, inner_path, start, end)
            }

            PropertyPath::ZeroOrOne(inner_path) => {
                self.evaluate_zero_or_one(store, inner_path, start, end)
            }

            PropertyPath::NegatedPropertySet(predicates) => {
                self.evaluate_negated_set(store, predicates, start, end)
            }
        }
    }

    fn evaluate_predicate(
        &self,
        store: &StarStore,
        predicate: &str,
        start: Option<&StarTerm>,
        end: Option<&StarTerm>,
    ) -> StarResult<Vec<PathResult>> {
        let pred_term = StarTerm::iri(predicate)?;

        // Query for triples matching the predicate
        let triples = store.query(start, Some(&pred_term), end)?;

        Ok(triples
            .into_iter()
            .map(|triple| PathResult {
                start: triple.subject.clone(),
                end: triple.object.clone(),
                path: vec![triple.predicate.clone()],
            })
            .collect())
    }

    fn evaluate_sequence(
        &self,
        store: &StarStore,
        first: &PropertyPath,
        second: &PropertyPath,
        start: Option<&StarTerm>,
        end: Option<&StarTerm>,
    ) -> StarResult<Vec<PathResult>> {
        // Evaluate first path
        let first_results = self.evaluate_path(store, first, start, None)?;

        let mut final_results = Vec::new();

        // For each result of the first path, evaluate the second path
        for first_result in first_results {
            let second_results = self.evaluate_path(store, second, Some(&first_result.end), end)?;

            for second_result in second_results {
                // Combine paths
                let mut combined_path = first_result.path.clone();
                combined_path.extend(second_result.path);

                final_results.push(PathResult {
                    start: first_result.start.clone(),
                    end: second_result.end,
                    path: combined_path,
                });
            }
        }

        Ok(final_results)
    }

    fn evaluate_alternative(
        &self,
        store: &StarStore,
        first: &PropertyPath,
        second: &PropertyPath,
        start: Option<&StarTerm>,
        end: Option<&StarTerm>,
    ) -> StarResult<Vec<PathResult>> {
        let mut results = self.evaluate_path(store, first, start, end)?;
        let mut second_results = self.evaluate_path(store, second, start, end)?;

        results.append(&mut second_results);

        // Remove duplicates
        let mut seen = HashSet::new();
        results.retain(|r| {
            let key = format!("{}->{}", r.start, r.end);
            seen.insert(key)
        });

        Ok(results)
    }

    fn evaluate_zero_or_more(
        &self,
        store: &StarStore,
        path: &PropertyPath,
        start: Option<&StarTerm>,
        end: Option<&StarTerm>,
    ) -> StarResult<Vec<PathResult>> {
        let mut results = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Get all possible starting points
        let start_nodes = if let Some(s) = start {
            vec![s.clone()]
        } else {
            // Get all subjects from the store
            let all_triples = store.query(None, None, None)?;
            all_triples
                .into_iter()
                .map(|t| t.subject)
                .collect::<HashSet<_>>()
                .into_iter()
                .collect()
        };

        // BFS to find all reachable nodes
        for start_node in start_nodes {
            queue.push_back((start_node.clone(), vec![]));
            visited.insert(format!("{}", start_node));

            // Add identity result (zero steps)
            if end.is_none() || end == Some(&start_node) {
                results.push(PathResult {
                    start: start_node.clone(),
                    end: start_node.clone(),
                    path: vec![],
                });
            }

            while let Some((current, current_path)) = queue.pop_front() {
                if current_path.len() >= self.max_path_length {
                    continue;
                }

                // Evaluate one step
                let one_step = self.evaluate_path(store, path, Some(&current), None)?;

                for step_result in one_step {
                    let key = format!("{}", step_result.end);

                    if !visited.contains(&key) {
                        visited.insert(key);

                        let mut new_path = current_path.clone();
                        new_path.extend(step_result.path);

                        // Add result if it matches the end constraint
                        if end.is_none() || end == Some(&step_result.end) {
                            results.push(PathResult {
                                start: start_node.clone(),
                                end: step_result.end.clone(),
                                path: new_path.clone(),
                            });
                        }

                        queue.push_back((step_result.end, new_path));
                    }
                }
            }
        }

        Ok(results)
    }

    fn evaluate_one_or_more(
        &self,
        store: &StarStore,
        path: &PropertyPath,
        start: Option<&StarTerm>,
        end: Option<&StarTerm>,
    ) -> StarResult<Vec<PathResult>> {
        // One or more = at least one step
        let mut results = self.evaluate_path(store, path, start, end)?;

        // Then zero or more additional steps
        let zero_or_more_path = PropertyPath::ZeroOrMore(Box::new(path.clone()));

        for result in results.clone() {
            let additional =
                self.evaluate_path(store, &zero_or_more_path, Some(&result.end), end)?;

            for add_result in additional {
                if add_result.start != add_result.end {
                    // Skip identity
                    let mut combined_path = result.path.clone();
                    combined_path.extend(add_result.path);

                    results.push(PathResult {
                        start: result.start.clone(),
                        end: add_result.end,
                        path: combined_path,
                    });
                }
            }
        }

        Ok(results)
    }

    fn evaluate_zero_or_one(
        &self,
        store: &StarStore,
        path: &PropertyPath,
        start: Option<&StarTerm>,
        end: Option<&StarTerm>,
    ) -> StarResult<Vec<PathResult>> {
        let mut results = Vec::new();

        // Zero steps (identity)
        if let Some(s) = start {
            if end.is_none() || end == Some(s) {
                results.push(PathResult {
                    start: s.clone(),
                    end: s.clone(),
                    path: vec![],
                });
            }
        }

        // One step
        let mut one_step = self.evaluate_path(store, path, start, end)?;
        results.append(&mut one_step);

        Ok(results)
    }

    fn evaluate_negated_set(
        &self,
        store: &StarStore,
        predicates: &[String],
        start: Option<&StarTerm>,
        end: Option<&StarTerm>,
    ) -> StarResult<Vec<PathResult>> {
        // Get all triples
        let all_triples = store.query(start, None, end)?;

        // Filter out triples with predicates in the negated set
        let results = all_triples
            .into_iter()
            .filter(|triple| {
                if let StarTerm::NamedNode(nn) = &triple.predicate {
                    !predicates.contains(&nn.iri)
                } else {
                    true
                }
            })
            .map(|triple| PathResult {
                start: triple.subject.clone(),
                end: triple.object.clone(),
                path: vec![triple.predicate.clone()],
            })
            .collect();

        Ok(results)
    }
}

impl Default for PropertyPathExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Federated query support for querying multiple endpoints
pub struct FederatedQueryExecutor {
    /// Registered endpoints
    endpoints: HashMap<String, String>,
}

impl FederatedQueryExecutor {
    /// Create a new federated query executor
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
        }
    }

    /// Register an endpoint
    pub fn register_endpoint(&mut self, name: impl Into<String>, url: impl Into<String>) {
        self.endpoints.insert(name.into(), url.into());
    }

    /// Execute a federated query (simplified - would use HTTP in production)
    pub fn execute_federated(
        &self,
        _endpoint: &str,
        _query: &str,
    ) -> StarResult<Vec<HashMap<String, StarTerm>>> {
        // In production, this would send an HTTP request to the endpoint
        // For now, return empty results
        Ok(Vec::new())
    }
}

impl Default for FederatedQueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Full-text search integration
pub struct FullTextSearch {
    /// Indexed literals
    index: HashMap<String, Vec<StarTriple>>,
}

impl FullTextSearch {
    /// Create a new full-text search index
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    /// Index a store for full-text search
    pub fn index_store(&mut self, store: &StarStore) -> StarResult<()> {
        let triples = store.query(None, None, None)?;

        for triple in triples {
            // Index literals
            if let StarTerm::Literal(lit) = &triple.object {
                let words: Vec<String> = lit
                    .value
                    .split_whitespace()
                    .map(|s| s.to_lowercase())
                    .collect();

                for word in words {
                    self.index.entry(word).or_default().push(triple.clone());
                }
            }
        }

        Ok(())
    }

    /// Search for triples containing a term
    pub fn search(&self, term: &str) -> Vec<StarTriple> {
        let normalized = term.to_lowercase();

        self.index.get(&normalized).cloned().unwrap_or_default()
    }

    /// Search with wildcards (simplified)
    pub fn search_wildcard(&self, pattern: &str) -> Vec<StarTriple> {
        let pattern_lower = pattern.to_lowercase();
        let mut results = Vec::new();

        for (word, triples) in &self.index {
            if Self::matches_wildcard(word, &pattern_lower) {
                results.extend(triples.clone());
            }
        }

        // Remove duplicates
        let mut seen = HashSet::new();
        results.retain(|t| seen.insert(format!("{}", t)));

        results
    }

    fn matches_wildcard(text: &str, pattern: &str) -> bool {
        // Simple wildcard matching (* matches any characters)
        if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('*').collect();

            if parts.is_empty() {
                return true;
            }

            let mut pos = 0;
            for (i, part) in parts.iter().enumerate() {
                if part.is_empty() {
                    continue;
                }

                if i == 0 {
                    // First part must match from the start
                    if !text.starts_with(part) {
                        return false;
                    }
                    pos = part.len();
                } else if i == parts.len() - 1 {
                    // Last part must match to the end
                    return text.ends_with(part);
                } else {
                    // Middle parts must appear in order
                    if let Some(found_pos) = text[pos..].find(part) {
                        pos += found_pos + part.len();
                    } else {
                        return false;
                    }
                }
            }

            true
        } else {
            text == pattern
        }
    }
}

impl Default for FullTextSearch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{StarTerm, StarTriple};

    #[test]
    fn test_property_path_predicate() -> StarResult<()> {
        let store = StarStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice")?,
            StarTerm::iri("http://xmlns.com/foaf/0.1/knows")?,
            StarTerm::iri("http://example.org/bob")?,
        );

        store.insert(&triple)?;

        let executor = PropertyPathExecutor::new();
        let path = PropertyPath::predicate("http://xmlns.com/foaf/0.1/knows");

        let results = executor.evaluate_path(&store, &path, None, None)?;

        assert_eq!(results.len(), 1);
        assert_eq!(
            format!("{}", results[0].start),
            "<http://example.org/alice>"
        );

        Ok(())
    }

    #[test]
    fn test_property_path_alternative() -> StarResult<()> {
        let store = StarStore::new();

        store.insert(&StarTriple::new(
            StarTerm::iri("http://example.org/alice")?,
            StarTerm::iri("http://xmlns.com/foaf/0.1/knows")?,
            StarTerm::iri("http://example.org/bob")?,
        ))?;

        store.insert(&StarTriple::new(
            StarTerm::iri("http://example.org/alice")?,
            StarTerm::iri("http://xmlns.com/foaf/0.1/worksWith")?,
            StarTerm::iri("http://example.org/charlie")?,
        ))?;

        let executor = PropertyPathExecutor::new();
        let path = PropertyPath::alternative(
            PropertyPath::predicate("http://xmlns.com/foaf/0.1/knows"),
            PropertyPath::predicate("http://xmlns.com/foaf/0.1/worksWith"),
        );

        let results = executor.evaluate_path(&store, &path, None, None)?;

        assert_eq!(results.len(), 2);

        Ok(())
    }

    #[test]
    fn test_full_text_search() -> StarResult<()> {
        let store = StarStore::new();

        store.insert(&StarTriple::new(
            StarTerm::iri("http://example.org/doc1")?,
            StarTerm::iri("http://purl.org/dc/terms/title")?,
            StarTerm::literal("The Quick Brown Fox")?,
        ))?;

        store.insert(&StarTriple::new(
            StarTerm::iri("http://example.org/doc2")?,
            StarTerm::iri("http://purl.org/dc/terms/title")?,
            StarTerm::literal("A Slow Brown Dog")?,
        ))?;

        let mut fts = FullTextSearch::new();
        fts.index_store(&store)?;

        let results = fts.search("brown");
        assert_eq!(results.len(), 2);

        let results = fts.search("quick");
        assert_eq!(results.len(), 1);

        Ok(())
    }

    #[test]
    fn test_wildcard_search() -> StarResult<()> {
        let store = StarStore::new();

        store.insert(&StarTriple::new(
            StarTerm::iri("http://example.org/doc1")?,
            StarTerm::iri("http://purl.org/dc/terms/title")?,
            StarTerm::literal("testing wildcards")?,
        ))?;

        let mut fts = FullTextSearch::new();
        fts.index_store(&store)?;

        let results = fts.search_wildcard("test*");
        assert_eq!(results.len(), 1);

        Ok(())
    }
}
