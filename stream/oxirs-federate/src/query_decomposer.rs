//! Federated SPARQL query decomposition.
//!
//! Splits a flat list of triple patterns across registered SPARQL endpoints
//! using declared predicate/graph capabilities, estimates join costs, and
//! provides a utility for merging heterogeneous result sets.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// One term in a triple pattern — either a bound value or a query variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternTerm {
    /// A SPARQL query variable (without the leading `?`).
    Variable(String),
    /// A fully-qualified IRI.
    Iri(String),
    /// A literal string value.
    Literal(String),
}

impl PatternTerm {
    /// Returns `true` if this term is a variable.
    pub fn is_variable(&self) -> bool {
        matches!(self, PatternTerm::Variable(_))
    }

    /// Returns the inner string regardless of variant.
    pub fn value(&self) -> &str {
        match self {
            PatternTerm::Variable(s) | PatternTerm::Iri(s) | PatternTerm::Literal(s) => s.as_str(),
        }
    }
}

/// A single RDF triple pattern with subject, predicate, and object terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriplePattern {
    /// Subject term.
    pub subject: PatternTerm,
    /// Predicate term.
    pub predicate: PatternTerm,
    /// Object term.
    pub object: PatternTerm,
}

impl TriplePattern {
    /// Create a new triple pattern.
    pub fn new(subject: PatternTerm, predicate: PatternTerm, object: PatternTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Number of unbound variables in this pattern.
    pub fn variable_count(&self) -> usize {
        [&self.subject, &self.predicate, &self.object]
            .iter()
            .filter(|t| t.is_variable())
            .count()
    }
}

/// A sub-query assigned to a single endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct SubQuery {
    /// Patterns routed to `endpoint`.
    pub patterns: Vec<TriplePattern>,
    /// SPARQL endpoint URL.
    pub endpoint: String,
    /// Optional SPARQL FILTER clauses that apply to this sub-query.
    pub filters: Vec<String>,
}

impl SubQuery {
    /// Create an empty sub-query for a given endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            patterns: Vec::new(),
            endpoint: endpoint.into(),
            filters: Vec::new(),
        }
    }

    /// Add a pattern.
    pub fn add_pattern(&mut self, pattern: TriplePattern) {
        self.patterns.push(pattern);
    }

    /// Add a FILTER expression.
    pub fn add_filter(&mut self, filter: impl Into<String>) {
        self.filters.push(filter.into());
    }
}

/// The result of decomposing a set of triple patterns across endpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct DecomposedQuery {
    /// Sub-queries, one per targeted endpoint.
    pub sub_queries: Vec<SubQuery>,
    /// Suggested join order (indices into `sub_queries`).
    pub join_order: Vec<usize>,
    /// Estimated total execution cost (lower is better).
    pub estimated_cost: f64,
}

/// Capability advertisement for a single SPARQL endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointCapability {
    /// SPARQL endpoint URL.
    pub endpoint: String,
    /// Predicates the endpoint handles (IRIs).
    pub predicates: Vec<String>,
    /// Named graphs hosted at this endpoint.
    pub graphs: Vec<String>,
}

impl EndpointCapability {
    /// Create a capability descriptor with no predicates or graphs.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            predicates: Vec::new(),
            graphs: Vec::new(),
        }
    }

    /// Builder: add a handled predicate IRI.
    pub fn with_predicate(mut self, pred: impl Into<String>) -> Self {
        self.predicates.push(pred.into());
        self
    }

    /// Builder: add a hosted named graph IRI.
    pub fn with_graph(mut self, graph: impl Into<String>) -> Self {
        self.graphs.push(graph.into());
        self
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// QueryDecomposer
// ──────────────────────────────────────────────────────────────────────────────

/// Routes triple patterns to SPARQL endpoints based on declared capabilities.
///
/// Each endpoint advertises the predicates and named graphs it handles.
/// `decompose` assigns every pattern to the most specific endpoint it can find
/// and groups them into [`SubQuery`] values.  Patterns that no endpoint claims
/// are routed to the first registered endpoint as a fallback; if no endpoints
/// are registered the sub-query list is empty.
#[derive(Debug, Default)]
pub struct QueryDecomposer {
    capabilities: Vec<EndpointCapability>,
}

impl QueryDecomposer {
    /// Create a decomposer with no registered endpoints.
    pub fn new() -> Self {
        Self {
            capabilities: Vec::new(),
        }
    }

    /// Register an endpoint's capabilities.
    pub fn add_endpoint(&mut self, capability: EndpointCapability) {
        self.capabilities.push(capability);
    }

    /// Return the number of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.capabilities.len()
    }

    /// Find which endpoint can handle `pattern`.
    ///
    /// Matching priority:
    /// 1. The predicate is an IRI that appears in `EndpointCapability::predicates`.
    /// 2. Otherwise the first endpoint (catch-all).
    /// 3. `None` if no endpoints are registered.
    pub fn assign_pattern(&self, pattern: &TriplePattern) -> Option<&str> {
        if self.capabilities.is_empty() {
            return None;
        }
        // Try predicate-based matching first.
        if let PatternTerm::Iri(pred_iri) = &pattern.predicate {
            for cap in &self.capabilities {
                if cap.predicates.iter().any(|p| p == pred_iri) {
                    return Some(cap.endpoint.as_str());
                }
            }
        }
        // Fallback: first endpoint.
        Some(self.capabilities[0].endpoint.as_str())
    }

    /// Decompose a list of triple patterns into endpoint-specific sub-queries.
    ///
    /// The join order is heuristically sorted so that sub-queries with fewer
    /// variables (more selective) come first, reducing intermediate result set
    /// sizes.
    pub fn decompose(&self, patterns: Vec<TriplePattern>) -> DecomposedQuery {
        if self.capabilities.is_empty() {
            return DecomposedQuery {
                sub_queries: Vec::new(),
                join_order: Vec::new(),
                estimated_cost: 0.0,
            };
        }

        // Accumulate patterns per endpoint URL.
        let mut endpoint_map: HashMap<String, SubQuery> = HashMap::new();

        for pattern in patterns {
            let endpoint_url = self
                .assign_pattern(&pattern)
                .unwrap_or_else(|| self.capabilities[0].endpoint.as_str())
                .to_owned();

            endpoint_map
                .entry(endpoint_url.clone())
                .or_insert_with(|| SubQuery::new(endpoint_url))
                .add_pattern(pattern);
        }

        // Collect into a stable Vec sorted by endpoint URL for determinism.
        let mut sub_queries: Vec<SubQuery> = endpoint_map.into_values().collect();
        sub_queries.sort_by(|a, b| a.endpoint.cmp(&b.endpoint));

        // Heuristic join order: prefer sub-queries with fewer variables
        // (more selective / smaller expected result sets).
        let mut indexed: Vec<(usize, usize)> = sub_queries
            .iter()
            .enumerate()
            .map(|(i, sq)| {
                let var_count: usize = sq.patterns.iter().map(|p| p.variable_count()).sum();
                (i, var_count)
            })
            .collect();
        indexed.sort_by_key(|&(_, vars)| vars);
        let join_order: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();

        // Simple cost: sum of (pattern_count × avg_variable_count) per sub-query.
        let estimated_cost = sub_queries
            .iter()
            .map(|sq| {
                let var_sum: usize = sq.patterns.iter().map(|p| p.variable_count()).sum();
                var_sum as f64
            })
            .sum();

        DecomposedQuery {
            sub_queries,
            join_order,
            estimated_cost,
        }
    }

    /// Merge multiple result rows from different sub-queries into a unified set.
    ///
    /// Rows from all sub-queries are concatenated.  Duplicate rows (identical
    /// variable binding maps) are removed.
    pub fn merge_results(
        results: Vec<Vec<HashMap<String, String>>>,
    ) -> Vec<HashMap<String, String>> {
        let mut merged: Vec<HashMap<String, String>> = Vec::new();
        for batch in results {
            for row in batch {
                if !merged.contains(&row) {
                    merged.push(row);
                }
            }
        }
        merged
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn var(name: &str) -> PatternTerm {
        PatternTerm::Variable(name.to_owned())
    }

    fn iri(value: &str) -> PatternTerm {
        PatternTerm::Iri(value.to_owned())
    }

    fn lit(value: &str) -> PatternTerm {
        PatternTerm::Literal(value.to_owned())
    }

    fn pat(s: PatternTerm, p: PatternTerm, o: PatternTerm) -> TriplePattern {
        TriplePattern::new(s, p, o)
    }

    fn row(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // ── PatternTerm ───────────────────────────────────────────────────────────

    #[test]
    fn test_pattern_term_is_variable() {
        assert!(var("x").is_variable());
        assert!(!iri("http://a").is_variable());
        assert!(!lit("hello").is_variable());
    }

    #[test]
    fn test_pattern_term_value() {
        assert_eq!(var("x").value(), "x");
        assert_eq!(iri("http://a").value(), "http://a");
        assert_eq!(lit("hello").value(), "hello");
    }

    #[test]
    fn test_pattern_term_equality() {
        assert_eq!(var("x"), var("x"));
        assert_ne!(var("x"), iri("x"));
    }

    // ── TriplePattern ─────────────────────────────────────────────────────────

    #[test]
    fn test_triple_pattern_variable_count_all_vars() {
        let p = pat(var("s"), var("p"), var("o"));
        assert_eq!(p.variable_count(), 3);
    }

    #[test]
    fn test_triple_pattern_variable_count_no_vars() {
        let p = pat(iri("s"), iri("p"), lit("o"));
        assert_eq!(p.variable_count(), 0);
    }

    #[test]
    fn test_triple_pattern_variable_count_mixed() {
        let p = pat(var("s"), iri("p"), var("o"));
        assert_eq!(p.variable_count(), 2);
    }

    // ── SubQuery ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sub_query_new_is_empty() {
        let sq = SubQuery::new("http://ep1");
        assert!(sq.patterns.is_empty());
        assert!(sq.filters.is_empty());
    }

    #[test]
    fn test_sub_query_add_pattern() {
        let mut sq = SubQuery::new("ep");
        sq.add_pattern(pat(var("s"), iri("p"), var("o")));
        assert_eq!(sq.patterns.len(), 1);
    }

    #[test]
    fn test_sub_query_add_filter() {
        let mut sq = SubQuery::new("ep");
        sq.add_filter("?age > 18");
        assert_eq!(sq.filters.len(), 1);
        assert_eq!(sq.filters[0], "?age > 18");
    }

    // ── EndpointCapability ────────────────────────────────────────────────────

    #[test]
    fn test_endpoint_capability_new() {
        let cap = EndpointCapability::new("http://ep");
        assert_eq!(cap.endpoint, "http://ep");
        assert!(cap.predicates.is_empty());
        assert!(cap.graphs.is_empty());
    }

    #[test]
    fn test_endpoint_capability_builder() {
        let cap = EndpointCapability::new("ep")
            .with_predicate("http://p1")
            .with_predicate("http://p2")
            .with_graph("http://g1");
        assert_eq!(cap.predicates.len(), 2);
        assert_eq!(cap.graphs.len(), 1);
    }

    // ── QueryDecomposer construction ──────────────────────────────────────────

    #[test]
    fn test_decomposer_new_empty() {
        let d = QueryDecomposer::new();
        assert_eq!(d.endpoint_count(), 0);
    }

    #[test]
    fn test_add_endpoint_increments_count() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(EndpointCapability::new("ep1"));
        d.add_endpoint(EndpointCapability::new("ep2"));
        assert_eq!(d.endpoint_count(), 2);
    }

    // ── assign_pattern ────────────────────────────────────────────────────────

    #[test]
    fn test_assign_pattern_no_endpoints_returns_none() {
        let d = QueryDecomposer::new();
        let p = pat(var("s"), iri("http://p"), var("o"));
        assert!(d.assign_pattern(&p).is_none());
    }

    #[test]
    fn test_assign_pattern_matches_predicate() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(EndpointCapability::new("http://ep1").with_predicate("http://knows"));
        d.add_endpoint(EndpointCapability::new("http://ep2").with_predicate("http://age"));

        let p1 = pat(var("s"), iri("http://knows"), var("o"));
        let p2 = pat(var("s"), iri("http://age"), var("o"));

        assert_eq!(d.assign_pattern(&p1), Some("http://ep1"));
        assert_eq!(d.assign_pattern(&p2), Some("http://ep2"));
    }

    #[test]
    fn test_assign_pattern_fallback_to_first() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(EndpointCapability::new("http://fallback"));

        let p = pat(var("s"), iri("http://unknown"), var("o"));
        assert_eq!(d.assign_pattern(&p), Some("http://fallback"));
    }

    #[test]
    fn test_assign_pattern_variable_predicate_uses_fallback() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(EndpointCapability::new("http://ep1").with_predicate("http://p"));

        let p = pat(var("s"), var("pred"), var("o"));
        // Variable predicate — can't match; falls back to first endpoint.
        assert_eq!(d.assign_pattern(&p), Some("http://ep1"));
    }

    // ── decompose ─────────────────────────────────────────────────────────────

    #[test]
    fn test_decompose_empty_patterns_no_endpoints() {
        let d = QueryDecomposer::new();
        let result = d.decompose(vec![]);
        assert!(result.sub_queries.is_empty());
        assert!(result.join_order.is_empty());
    }

    #[test]
    fn test_decompose_empty_patterns_with_endpoint() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(EndpointCapability::new("http://ep"));
        let result = d.decompose(vec![]);
        assert!(result.sub_queries.is_empty());
    }

    #[test]
    fn test_decompose_routes_to_single_endpoint() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(EndpointCapability::new("http://ep1").with_predicate("http://p"));

        let patterns = vec![
            pat(var("s"), iri("http://p"), var("o")),
            pat(var("s"), iri("http://p"), lit("lit")),
        ];

        let result = d.decompose(patterns);
        assert_eq!(result.sub_queries.len(), 1);
        assert_eq!(result.sub_queries[0].endpoint, "http://ep1");
        assert_eq!(result.sub_queries[0].patterns.len(), 2);
    }

    #[test]
    fn test_decompose_routes_to_multiple_endpoints() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(EndpointCapability::new("http://ep-a").with_predicate("http://type"));
        d.add_endpoint(EndpointCapability::new("http://ep-b").with_predicate("http://name"));

        let patterns = vec![
            pat(var("s"), iri("http://type"), iri("http://Person")),
            pat(var("s"), iri("http://name"), var("n")),
        ];

        let result = d.decompose(patterns);
        assert_eq!(result.sub_queries.len(), 2);
    }

    #[test]
    fn test_decompose_join_order_is_valid_indices() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(EndpointCapability::new("http://ep1").with_predicate("http://p1"));
        d.add_endpoint(EndpointCapability::new("http://ep2").with_predicate("http://p2"));

        let patterns = vec![
            pat(var("s"), iri("http://p1"), var("o")),
            pat(var("s"), iri("http://p2"), lit("fixed")),
        ];

        let result = d.decompose(patterns);
        let n = result.sub_queries.len();
        assert_eq!(result.join_order.len(), n);
        for &idx in &result.join_order {
            assert!(idx < n, "join_order index {} out of range", idx);
        }
    }

    #[test]
    fn test_decompose_estimated_cost_non_negative() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(EndpointCapability::new("http://ep"));

        let patterns = vec![pat(var("s"), var("p"), var("o"))];
        let result = d.decompose(patterns);
        assert!(result.estimated_cost >= 0.0);
    }

    #[test]
    fn test_decompose_selective_pattern_comes_first_in_join_order() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(EndpointCapability::new("http://ep1").with_predicate("http://a"));
        d.add_endpoint(EndpointCapability::new("http://ep2").with_predicate("http://b"));

        // ep1 gets an all-variable pattern (3 vars), ep2 gets a selective pattern (1 var)
        let patterns = vec![
            pat(var("s"), iri("http://a"), var("o")),        // 2 vars
            pat(iri("http://x"), iri("http://b"), var("n")), // 1 var
        ];

        let result = d.decompose(patterns);
        // The join order should put the sub-query with fewer variables first
        let first_idx = result.join_order[0];
        let last_idx = result.join_order[result.join_order.len() - 1];
        let first_vars: usize = result.sub_queries[first_idx]
            .patterns
            .iter()
            .map(|p| p.variable_count())
            .sum();
        let last_vars: usize = result.sub_queries[last_idx]
            .patterns
            .iter()
            .map(|p| p.variable_count())
            .sum();
        assert!(first_vars <= last_vars);
    }

    // ── merge_results ─────────────────────────────────────────────────────────

    #[test]
    fn test_merge_results_empty() {
        let merged = QueryDecomposer::merge_results(vec![]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_results_single_batch() {
        let batch = vec![row(&[("x", "1")]), row(&[("x", "2")])];
        let merged = QueryDecomposer::merge_results(vec![batch]);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merge_results_deduplicates() {
        let row1 = row(&[("x", "1")]);
        let batch1 = vec![row1.clone(), row(&[("x", "2")])];
        let batch2 = vec![row1.clone(), row(&[("x", "3")])];
        let merged = QueryDecomposer::merge_results(vec![batch1, batch2]);
        // row1 is deduplicated
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_merge_results_multiple_batches_no_overlap() {
        let b1 = vec![row(&[("a", "1")])];
        let b2 = vec![row(&[("a", "2")])];
        let b3 = vec![row(&[("a", "3")])];
        let merged = QueryDecomposer::merge_results(vec![b1, b2, b3]);
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_merge_results_preserves_all_fields() {
        let r = row(&[("x", "1"), ("y", "hello")]);
        let merged = QueryDecomposer::merge_results(vec![vec![r.clone()]]);
        assert_eq!(merged[0]["x"], "1");
        assert_eq!(merged[0]["y"], "hello");
    }

    // ── end-to-end ────────────────────────────────────────────────────────────

    #[test]
    fn test_end_to_end_two_endpoints() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(
            EndpointCapability::new("http://dbpedia.org/sparql")
                .with_predicate("http://dbpedia.org/ontology/birthPlace"),
        );
        d.add_endpoint(
            EndpointCapability::new("http://wikidata.org/sparql")
                .with_predicate("http://wikidata.org/prop/P569"),
        );

        let patterns = vec![
            pat(
                var("person"),
                iri("http://dbpedia.org/ontology/birthPlace"),
                var("place"),
            ),
            pat(
                var("person"),
                iri("http://wikidata.org/prop/P569"),
                var("dob"),
            ),
        ];

        let result = d.decompose(patterns);
        assert_eq!(result.sub_queries.len(), 2);
        assert_eq!(result.join_order.len(), 2);

        let ep_urls: Vec<&str> = result
            .sub_queries
            .iter()
            .map(|sq| sq.endpoint.as_str())
            .collect();
        assert!(ep_urls.contains(&"http://dbpedia.org/sparql"));
        assert!(ep_urls.contains(&"http://wikidata.org/sparql"));
    }

    #[test]
    fn test_assign_pattern_literal_predicate_uses_fallback() {
        let mut d = QueryDecomposer::new();
        d.add_endpoint(EndpointCapability::new("http://ep"));
        let p = pat(var("s"), lit("some-literal"), var("o"));
        // Literal predicates never match; fall back to first endpoint.
        assert_eq!(d.assign_pattern(&p), Some("http://ep"));
    }
}
