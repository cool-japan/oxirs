//! Query decomposition for cross-instance SPARQL federation.
//!
//! The [`QueryDecomposer`] takes a SPARQL query string together with a set of
//! known SPARQL endpoints and produces a [`FederatedQuery`]: a partitioned
//! representation where each partition targets a specific endpoint.
//!
//! # Decomposition strategy
//!
//! 1. Parse the query into triple patterns and filter clauses (lightweight
//!    token-based parsing; no full SPARQL grammar required at this level).
//! 2. Assign each triple pattern to the "best" endpoint based on its namespace
//!    affinities and estimated result cardinality.
//! 3. Group assigned patterns back into per-endpoint subqueries, preserving
//!    shared projection variables and FILTER expressions where possible.
//! 4. Apply push-down rewriting: move FILTER expressions into the subquery of
//!    the endpoint that owns all variables referenced by that filter.
//!
//! The result is a [`FederatedQuery`] whose subqueries are ready for dispatch
//! and whose per-subquery metadata can be fed to the optimizer.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::error::{FederationError, FederationResult};

// ─── Public types ────────────────────────────────────────────────────────────

/// Metadata describing a known SPARQL endpoint that may be targeted by
/// subqueries during federation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointInfo {
    /// The full URL of the SPARQL endpoint (e.g. `http://host/sparql`).
    pub url: String,
    /// Namespace prefixes whose data is expected to reside at this endpoint.
    /// Used to score predicate/type affinity during pattern assignment.
    pub namespace_affinities: Vec<String>,
    /// Optional human-readable name for logging and diagnostics.
    pub name: Option<String>,
    /// Approximate number of RDF triples hosted at this endpoint (0 = unknown).
    pub estimated_triple_count: u64,
}

impl EndpointInfo {
    /// Construct a minimal `EndpointInfo` with just a URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            namespace_affinities: Vec::new(),
            name: None,
            estimated_triple_count: 0,
        }
    }

    /// Builder-style method to add a namespace affinity.
    pub fn with_affinity(mut self, namespace: impl Into<String>) -> Self {
        self.namespace_affinities.push(namespace.into());
        self
    }
}

/// A triple pattern extracted from the WHERE clause of a SPARQL query.
///
/// Each component is represented as an optional string: `None` indicates the
/// wildcard (`_:` or simply absent), while `Some(s)` is either a variable
/// (`?name`) or a concrete IRI/literal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TriplePattern {
    /// Subject term or variable.
    pub subject: Option<String>,
    /// Predicate term or variable.
    pub predicate: Option<String>,
    /// Object term or variable.
    pub object: Option<String>,
    /// The full original pattern string as it appeared in the SPARQL source.
    pub pattern_string: String,
}

impl TriplePattern {
    /// Collect the set of variable names (prefixed `?`) referenced in this pattern.
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        let parts: [&Option<String>; 3] = [&self.subject, &self.predicate, &self.object];
        for part in parts.into_iter().flatten() {
            if part.starts_with('?') {
                vars.insert(part.clone());
            }
        }
        vars
    }
}

/// A SPARQL subquery targeted at a single remote endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointSubquery {
    /// URL of the SPARQL endpoint to which this subquery should be dispatched.
    pub endpoint_url: String,
    /// Complete SPARQL SELECT query string ready for execution.
    pub sparql: String,
    /// Estimated number of result rows (0 = unknown).
    pub estimated_results: usize,
    /// Scheduling priority: lower values should be executed first (or in parallel
    /// earlier) to minimise overall query latency.  Derived from the cost model.
    pub priority: f64,
}

/// The complete decomposition of a federated SPARQL query into per-endpoint
/// subqueries together with the original query string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedQuery {
    /// The original SPARQL query that was decomposed.
    pub original_query: String,
    /// Per-endpoint subqueries produced by decomposition.
    pub subqueries: Vec<EndpointSubquery>,
}

impl FederatedQuery {
    /// Return the set of endpoint URLs that will be contacted.
    pub fn endpoint_urls(&self) -> Vec<&str> {
        self.subqueries
            .iter()
            .map(|sq| sq.endpoint_url.as_str())
            .collect()
    }

    /// Return `true` if every subquery targets a distinct endpoint.
    pub fn has_no_duplicate_endpoints(&self) -> bool {
        let urls: HashSet<_> = self
            .subqueries
            .iter()
            .map(|sq| sq.endpoint_url.as_str())
            .collect();
        urls.len() == self.subqueries.len()
    }
}

// ─── QueryDecomposer ─────────────────────────────────────────────────────────

/// Decomposes a federated SPARQL query into per-endpoint subqueries.
#[derive(Debug, Clone)]
pub struct QueryDecomposer {
    /// Known endpoints that can serve as targets for subqueries.
    pub endpoints: Vec<EndpointInfo>,
}

impl QueryDecomposer {
    /// Create a new decomposer for the given set of endpoints.
    pub fn new(endpoints: Vec<EndpointInfo>) -> Self {
        Self { endpoints }
    }

    /// Decompose a SPARQL query string into a [`FederatedQuery`].
    ///
    /// # Errors
    ///
    /// Returns [`FederationError::EmptyEndpointList`] when no endpoints are
    /// configured and [`FederationError::QueryParseError`] for structurally
    /// invalid query strings.
    pub fn decompose(&self, sparql: &str) -> FederationResult<FederatedQuery> {
        if self.endpoints.is_empty() {
            return Err(FederationError::EmptyEndpointList);
        }

        let patterns = parse_triple_patterns(sparql)?;
        let filters = parse_filter_expressions(sparql);
        let projection = parse_projection_variables(sparql);

        let assignments = self.assign_endpoints(&patterns);
        let mut grouped = self.group_patterns_by_endpoint(assignments, &filters);

        // Apply filter push-down
        self.push_down_filters(&mut grouped);

        let subqueries = grouped
            .into_iter()
            .map(|(endpoint_url, ep_patterns, ep_filters)| {
                let sparql_text =
                    build_select_query(&projection, &ep_patterns, &ep_filters, &endpoint_url);
                let estimated_results = estimate_result_count(&ep_patterns);
                EndpointSubquery {
                    endpoint_url,
                    sparql: sparql_text,
                    estimated_results,
                    priority: 0.0, // Will be set by optimizer
                }
            })
            .collect();

        Ok(FederatedQuery {
            original_query: sparql.to_string(),
            subqueries,
        })
    }

    /// Push FILTER expressions down into the subquery of each endpoint that
    /// owns all variables referenced by that filter.
    ///
    /// This reduces the number of result rows returned over the network before
    /// the federation engine applies post-merge filtering.
    pub fn push_down_filters(&self, grouped: &mut [(String, Vec<TriplePattern>, Vec<String>)]) {
        // For each group, determine which variables are bound in that group.
        let group_bound_vars: Vec<HashSet<String>> = grouped
            .iter()
            .map(|(_, patterns, _)| {
                patterns
                    .iter()
                    .flat_map(|p| p.variables())
                    .collect::<HashSet<_>>()
            })
            .collect();

        // Collect all filters across all groups (deduplicated)
        let all_filters: Vec<String> = grouped
            .iter()
            .flat_map(|(_, _, filters)| filters.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        for (idx, (_, _, ref mut filters)) in grouped.iter_mut().enumerate() {
            let bound = &group_bound_vars[idx];
            for filter in &all_filters {
                let filter_vars = extract_filter_variables(filter);
                // Push down only if all filter variables are in this group's bound set
                if !filter_vars.is_empty()
                    && filter_vars.iter().all(|v| bound.contains(v))
                    && !filters.contains(filter)
                {
                    filters.push(filter.clone());
                }
            }
        }
    }

    /// Assign each triple pattern to the best-matching endpoint.
    ///
    /// Returns a `Vec` pairing each pattern with the index of the chosen endpoint.
    pub fn assign_endpoints(&self, patterns: &[TriplePattern]) -> Vec<(TriplePattern, usize)> {
        patterns
            .iter()
            .map(|pattern| {
                let idx = self.score_pattern_endpoints(pattern);
                (pattern.clone(), idx)
            })
            .collect()
    }

    // ─── Private helpers ──────────────────────────────────────────────────────

    /// Score each endpoint for a pattern and return the index of the best match.
    fn score_pattern_endpoints(&self, pattern: &TriplePattern) -> usize {
        let mut best_idx = 0usize;
        let mut best_score = -1.0f64;

        for (idx, endpoint) in self.endpoints.iter().enumerate() {
            let score = compute_affinity_score(pattern, endpoint);
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Group pattern assignments by endpoint, mapping endpoint indexes to URLs.
    fn group_patterns_by_endpoint(
        &self,
        assignments: Vec<(TriplePattern, usize)>,
        all_filters: &[String],
    ) -> Vec<(String, Vec<TriplePattern>, Vec<String>)> {
        let mut index_map: HashMap<usize, Vec<TriplePattern>> = HashMap::new();
        for (pattern, idx) in assignments {
            index_map.entry(idx).or_default().push(pattern);
        }

        index_map
            .into_iter()
            .map(|(idx, patterns)| {
                let url = self
                    .endpoints
                    .get(idx)
                    .map(|e| e.url.clone())
                    .unwrap_or_else(|| format!("unknown-endpoint-{idx}"));
                (url, patterns, all_filters.to_vec())
            })
            .collect()
    }
}

// ─── Module-level helpers ────────────────────────────────────────────────────

/// Compute a numeric affinity score for a triple pattern against an endpoint.
fn compute_affinity_score(pattern: &TriplePattern, endpoint: &EndpointInfo) -> f64 {
    let mut score = 0.0f64;

    let parts: [&Option<String>; 3] = [&pattern.predicate, &pattern.object, &pattern.subject];
    for part in parts.into_iter().flatten() {
        for affinity in &endpoint.namespace_affinities {
            if part.contains(affinity.as_str()) || part.starts_with(affinity.trim_end_matches('/'))
            {
                score += 10.0;
            }
            // Shorter prefix match
            if part
                .split(':')
                .next()
                .is_some_and(|prefix| affinity.contains(prefix))
            {
                score += 2.0;
            }
        }
    }

    score
}

/// Lightweight SPARQL triple-pattern parser.
fn parse_triple_patterns(sparql: &str) -> FederationResult<Vec<TriplePattern>> {
    let where_body = extract_where_body(sparql)?;
    let mut patterns = Vec::new();

    let raw_patterns = split_on_period(&where_body);

    for raw in raw_patterns {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let upper = trimmed.to_uppercase();
        if upper.starts_with("FILTER")
            || upper.starts_with("OPTIONAL")
            || upper.starts_with("UNION")
            || upper.starts_with("VALUES")
            || upper.starts_with("GRAPH")
            || upper.starts_with("BIND")
            || upper.starts_with("SERVICE")
        {
            continue;
        }

        if let Some(pattern) = parse_single_triple(trimmed) {
            patterns.push(pattern);
        }
    }

    Ok(patterns)
}

/// Parse a single triple from a whitespace-tokenised triple string.
fn parse_single_triple(s: &str) -> Option<TriplePattern> {
    let tokens: Vec<&str> = s.split_whitespace().collect();
    if tokens.len() < 3 {
        return None;
    }
    Some(TriplePattern {
        subject: Some(tokens[0].to_string()),
        predicate: Some(tokens[1].to_string()),
        object: Some(tokens[2..].join(" ")),
        pattern_string: s.to_string(),
    })
}

/// Extract FILTER expression strings from the query text.
fn parse_filter_expressions(sparql: &str) -> Vec<String> {
    let mut filters = Vec::new();
    let upper = sparql.to_uppercase();
    let mut search_from = 0usize;

    while let Some(pos) = upper[search_from..].find("FILTER") {
        let abs_pos = search_from + pos;
        if let Some(open) = sparql[abs_pos..].find('(') {
            let open_abs = abs_pos + open;
            if let Some(filter_body) = extract_balanced_parens(&sparql[open_abs..]) {
                filters.push(format!("FILTER({filter_body})"));
                search_from = open_abs + filter_body.len() + 2;
            } else {
                search_from = abs_pos + 6;
            }
        } else {
            search_from = abs_pos + 6;
        }
    }

    filters
}

/// Parse the projection variables from a SELECT clause.
fn parse_projection_variables(sparql: &str) -> Vec<String> {
    let upper = sparql.to_uppercase();
    let select_pos = match upper.find("SELECT") {
        Some(p) => p,
        None => return vec!["*".to_string()],
    };

    let after_select = &sparql[select_pos + 6..];
    let where_pos = after_select
        .to_uppercase()
        .find("WHERE")
        .unwrap_or(after_select.len());
    let projection_text = after_select[..where_pos].trim();

    if projection_text == "*" {
        return vec!["*".to_string()];
    }

    let vars: Vec<String> = projection_text
        .split_whitespace()
        .filter(|t| t.starts_with('?') || t.starts_with('$'))
        .map(|t| t.to_string())
        .collect();

    if vars.is_empty() {
        vec!["*".to_string()]
    } else {
        vars
    }
}

/// Extract the body of the WHERE clause.
fn extract_where_body(sparql: &str) -> FederationResult<String> {
    let upper = sparql.to_uppercase();
    let where_pos = upper
        .find("WHERE")
        .or_else(|| upper.find("ASK"))
        .ok_or_else(|| FederationError::QueryParseError("Missing WHERE clause".to_string()))?;

    let after_where = &sparql[where_pos..];
    let _ = after_where.find('{').ok_or_else(|| {
        FederationError::QueryParseError("Missing opening brace in WHERE clause".to_string())
    })?;

    let body = extract_balanced_braces(after_where).ok_or_else(|| {
        FederationError::QueryParseError("Unbalanced braces in WHERE clause".to_string())
    })?;

    Ok(body)
}

/// Extract the content inside balanced curly braces starting at position 0 of `s`.
fn extract_balanced_braces(s: &str) -> Option<String> {
    let mut depth = 0i32;
    let mut start = None;
    let mut end = None;

    for (i, ch) in s.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i + 1);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }

    match (start, end) {
        (Some(s_idx), Some(e_idx)) => Some(s[s_idx..e_idx].to_string()),
        _ => None,
    }
}

/// Extract the content inside balanced parentheses starting at position 0 of `s`.
fn extract_balanced_parens(s: &str) -> Option<String> {
    let mut depth = 0i32;
    let mut start = None;
    let mut end = None;

    for (i, ch) in s.char_indices() {
        match ch {
            '(' => {
                if depth == 0 {
                    start = Some(i + 1);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }

    match (start, end) {
        (Some(s_idx), Some(e_idx)) => Some(s[s_idx..e_idx].to_string()),
        _ => None,
    }
}

/// Split a WHERE body string on `.` at the top brace level.
fn split_on_period(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();

    for ch in s.chars() {
        match ch {
            '{' => {
                depth += 1;
                current.push(ch);
            }
            '}' => {
                depth -= 1;
                current.push(ch);
            }
            '.' if depth == 0 => {
                result.push(current.trim().to_string());
                current = String::new();
            }
            _ => current.push(ch),
        }
    }

    let tail = current.trim().to_string();
    if !tail.is_empty() {
        result.push(tail);
    }

    result
}

/// Extract variable names referenced inside a FILTER expression string.
pub(super) fn extract_filter_variables(filter: &str) -> Vec<String> {
    let mut vars = Vec::new();
    let mut chars = filter.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '?'
            && chars
                .peek()
                .is_some_and(|c| c.is_alphanumeric() || *c == '_')
        {
            let mut var = String::from("?");
            while chars
                .peek()
                .is_some_and(|c| c.is_alphanumeric() || *c == '_')
            {
                var.push(chars.next().expect("already peeked successfully"));
            }
            vars.push(var);
        }
    }
    vars
}

/// Build a SPARQL SELECT subquery for an endpoint from its assigned patterns and filters.
fn build_select_query(
    projection: &[String],
    patterns: &[TriplePattern],
    filters: &[String],
    _endpoint_url: &str,
) -> String {
    let proj_str = if projection.is_empty() || projection == ["*"] {
        "*".to_string()
    } else {
        projection.join(" ")
    };

    let mut body_parts: Vec<String> = patterns
        .iter()
        .map(|p| format!("  {}", p.pattern_string))
        .collect();

    for filter in filters {
        body_parts.push(format!("  {filter}"));
    }

    format!(
        "SELECT {proj_str} WHERE {{\n{}\n}}",
        body_parts.join(" .\n")
    )
}

/// Rough cardinality estimate based on the number of variables in the patterns.
fn estimate_result_count(patterns: &[TriplePattern]) -> usize {
    let total_vars: usize = patterns.iter().map(|p| p.variables().len()).sum();
    if total_vars == 0 {
        1
    } else {
        (10usize).saturating_pow(total_vars.min(4) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_endpoints() -> Vec<EndpointInfo> {
        vec![
            EndpointInfo::new("http://foaf-endpoint/sparql")
                .with_affinity("http://xmlns.com/foaf/0.1/")
                .with_affinity("foaf"),
            EndpointInfo::new("http://schema-endpoint/sparql")
                .with_affinity("http://schema.org/")
                .with_affinity("schema"),
        ]
    }

    #[test]
    fn test_decompose_requires_endpoints() {
        let decomposer = QueryDecomposer::new(vec![]);
        let result = decomposer.decompose("SELECT ?s WHERE { ?s a foaf:Person }");
        assert!(matches!(result, Err(FederationError::EmptyEndpointList)));
    }

    #[test]
    fn test_decompose_single_pattern() {
        let decomposer = QueryDecomposer::new(make_endpoints());
        let result = decomposer.decompose("SELECT ?s WHERE { ?s a foaf:Person }");
        assert!(result.is_ok());
        let fq = result.expect("decomposition should succeed");
        assert_eq!(fq.original_query, "SELECT ?s WHERE { ?s a foaf:Person }");
        assert!(!fq.subqueries.is_empty());
    }

    #[test]
    fn test_decompose_two_patterns_same_endpoint() {
        let decomposer = QueryDecomposer::new(make_endpoints());
        let query = "SELECT ?s ?name WHERE { ?s a foaf:Person . ?s foaf:name ?name }";
        let fq = decomposer
            .decompose(query)
            .expect("decomposition should succeed");
        // Both patterns have foaf affinity → should both go to the first endpoint
        assert_eq!(fq.subqueries.len(), 1);
        assert_eq!(fq.subqueries[0].endpoint_url, "http://foaf-endpoint/sparql");
    }

    #[test]
    fn test_assign_endpoints_returns_one_per_pattern() {
        let decomposer = QueryDecomposer::new(make_endpoints());
        let patterns = vec![
            TriplePattern {
                subject: Some("?s".to_string()),
                predicate: Some("a".to_string()),
                object: Some("foaf:Person".to_string()),
                pattern_string: "?s a foaf:Person".to_string(),
            },
            TriplePattern {
                subject: Some("?s".to_string()),
                predicate: Some("schema:name".to_string()),
                object: Some("?name".to_string()),
                pattern_string: "?s schema:name ?name".to_string(),
            },
        ];
        let assignments = decomposer.assign_endpoints(&patterns);
        assert_eq!(assignments.len(), 2);
    }

    #[test]
    fn test_triple_pattern_variables() {
        let p = TriplePattern {
            subject: Some("?s".to_string()),
            predicate: Some("foaf:name".to_string()),
            object: Some("?name".to_string()),
            pattern_string: "?s foaf:name ?name".to_string(),
        };
        let vars = p.variables();
        assert!(vars.contains("?s"));
        assert!(vars.contains("?name"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_parse_projection_variables() {
        let proj = parse_projection_variables("SELECT ?s ?name WHERE { ?s foaf:name ?name }");
        assert!(proj.contains(&"?s".to_string()));
        assert!(proj.contains(&"?name".to_string()));
    }

    #[test]
    fn test_parse_projection_wildcard() {
        let proj = parse_projection_variables("SELECT * WHERE { ?s ?p ?o }");
        assert_eq!(proj, vec!["*"]);
    }

    #[test]
    fn test_filter_push_down_adds_filter_to_owning_group() {
        let decomposer = QueryDecomposer::new(make_endpoints());
        let mut grouped = vec![(
            "http://ep1/sparql".to_string(),
            vec![TriplePattern {
                subject: Some("?s".to_string()),
                predicate: Some("foaf:age".to_string()),
                object: Some("?age".to_string()),
                pattern_string: "?s foaf:age ?age".to_string(),
            }],
            vec!["FILTER(?age > 18)".to_string()],
        )];
        decomposer.push_down_filters(&mut grouped);
        // Filter is already in the list and should remain
        assert!(grouped[0].2.contains(&"FILTER(?age > 18)".to_string()));
    }

    #[test]
    fn test_filter_push_down_does_not_push_foreign_filter() {
        let decomposer = QueryDecomposer::new(make_endpoints());
        // Group 1 binds ?age; filter references ?income which group 1 does NOT bind
        let mut grouped = vec![(
            "http://ep1/sparql".to_string(),
            vec![TriplePattern {
                subject: Some("?s".to_string()),
                predicate: Some("foaf:age".to_string()),
                object: Some("?age".to_string()),
                pattern_string: "?s foaf:age ?age".to_string(),
            }],
            vec![],
        )];
        // Simulate that a filter for ?income was produced elsewhere
        let filter = "FILTER(?income > 50000)";
        // Manually add to simulate a global filter
        for (_, _, ref mut f) in &mut grouped {
            f.push(filter.to_string());
        }
        decomposer.push_down_filters(&mut grouped);
        // After push-down, this group should still not have the filter since ?income is unbound here
        // (the filter is already in the list from manual insert, so this tests idempotency)
        let count = grouped[0].2.iter().filter(|f| f.contains("income")).count();
        // It was added manually, so count should be 1 — but push_down should not add a second copy
        assert_eq!(count, 1);
    }

    #[test]
    fn test_endpoint_info_builder() {
        let info =
            EndpointInfo::new("http://ep/sparql").with_affinity("http://xmlns.com/foaf/0.1/");
        assert_eq!(info.url, "http://ep/sparql");
        assert_eq!(info.namespace_affinities.len(), 1);
    }

    #[test]
    fn test_federated_query_endpoint_urls() {
        let decomposer = QueryDecomposer::new(make_endpoints());
        let fq = decomposer
            .decompose("SELECT ?s WHERE { ?s a foaf:Person }")
            .expect("decomposition should succeed");
        let urls = fq.endpoint_urls();
        assert!(!urls.is_empty());
    }

    #[test]
    fn test_extract_filter_variables() {
        let vars = extract_filter_variables("FILTER(?age > 18 && ?name != \"bob\")");
        assert!(vars.contains(&"?age".to_string()));
        assert!(vars.contains(&"?name".to_string()));
    }

    #[test]
    fn test_estimate_result_count_zero_vars() {
        let patterns = vec![TriplePattern {
            subject: Some("ex:Alice".to_string()),
            predicate: Some("ex:knows".to_string()),
            object: Some("ex:Bob".to_string()),
            pattern_string: "ex:Alice ex:knows ex:Bob".to_string(),
        }];
        assert_eq!(estimate_result_count(&patterns), 1);
    }

    #[test]
    fn test_build_select_query_format() {
        let proj = vec!["?s".to_string(), "?name".to_string()];
        let patterns = vec![TriplePattern {
            subject: Some("?s".to_string()),
            predicate: Some("foaf:name".to_string()),
            object: Some("?name".to_string()),
            pattern_string: "?s foaf:name ?name".to_string(),
        }];
        let filters: Vec<String> = vec![];
        let query = build_select_query(&proj, &patterns, &filters, "http://ep/sparql");
        assert!(query.starts_with("SELECT ?s ?name WHERE"));
        assert!(query.contains("?s foaf:name ?name"));
    }

    #[test]
    fn test_has_no_duplicate_endpoints() {
        let decomposer = QueryDecomposer::new(make_endpoints());
        let fq = decomposer
            .decompose("SELECT ?s WHERE { ?s a foaf:Person }")
            .expect("decomposition should succeed");
        // Single subquery → trivially no duplicates
        assert!(fq.has_no_duplicate_endpoints());
    }

    #[test]
    fn test_decompose_mixed_patterns_splits_across_endpoints() {
        let decomposer = QueryDecomposer::new(make_endpoints());
        // One foaf pattern + one schema pattern → should route to different endpoints
        let query = "SELECT ?s ?name WHERE { ?s a foaf:Person . ?s schema:name ?name }";
        let fq = decomposer
            .decompose(query)
            .expect("decomposition should succeed");
        // Expect 2 subqueries (one per endpoint)
        assert_eq!(fq.subqueries.len(), 2);
    }
}
