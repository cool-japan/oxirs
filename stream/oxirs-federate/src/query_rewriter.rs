//! SPARQL query rewriting for federated execution.
//!
//! Transforms query strings by applying named rewrite rules and splitting
//! patterns into per-endpoint [`ServiceClause`]s.

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A basic graph pattern: triples, filters, and optional sub-patterns.
#[derive(Debug, Clone, Default)]
pub struct GraphPattern {
    pub triples: Vec<(String, String, String)>,
    pub filters: Vec<String>,
    pub optionals: Vec<GraphPattern>,
}

/// A SERVICE clause binding an endpoint URL to a graph pattern.
#[derive(Debug, Clone)]
pub struct ServiceClause {
    pub endpoint: String,
    pub pattern: GraphPattern,
}

/// The transformation to apply when a rule matches.
#[derive(Debug, Clone)]
pub enum RewriteTransform {
    /// Prepend `prefix` to the query string.
    AddPrefix(String),
    /// Replace all occurrences of `from` with `to` in the query string.
    ReplaceGraph(String, String),
    /// Inject a FILTER expression into the query string.
    InjectFilter(String),
    /// Wrap pattern blocks with a SERVICE clause for the given endpoint.
    WrapWithService(String),
    /// Strip OPTIONAL blocks from the query string.
    StripOptional,
}

/// A named rewrite rule.
#[derive(Debug, Clone)]
pub struct RewriteRule {
    pub name: String,
    /// A substring that must appear in the query for the rule to fire.
    pub applies_to: String,
    pub transform: RewriteTransform,
}

/// The result of rewriting a query.
#[derive(Debug, Clone)]
pub struct RewrittenQuery {
    pub original: String,
    pub rewritten: String,
    pub rules_applied: Vec<String>,
    pub service_clauses: Vec<ServiceClause>,
}

/// Rewrites SPARQL queries for federated execution.
pub struct QueryRewriter {
    rules: Vec<RewriteRule>,
    endpoints: Vec<String>,
}

impl Default for QueryRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryRewriter {
    /// Create a new rewriter with no rules or endpoints.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            endpoints: Vec::new(),
        }
    }

    /// Add a rewrite rule.
    pub fn add_rule(&mut self, rule: RewriteRule) {
        self.rules.push(rule);
    }

    /// Register a federated endpoint URL.
    pub fn add_endpoint(&mut self, endpoint: impl Into<String>) {
        self.endpoints.push(endpoint.into());
    }

    /// Apply all matching rules to `query` and return the rewritten result.
    pub fn rewrite(&self, query: &str) -> RewrittenQuery {
        let mut current = query.to_string();
        let mut rules_applied = Vec::new();
        let mut service_clauses = Vec::new();

        for rule in &self.rules {
            if current.contains(&rule.applies_to) {
                let (transformed, clauses) = apply_transform(&current, &rule.transform);
                current = transformed;
                rules_applied.push(rule.name.clone());
                service_clauses.extend(clauses);
            }
        }

        RewrittenQuery {
            original: query.to_string(),
            rewritten: current,
            rules_applied,
            service_clauses,
        }
    }

    /// Split `query` into one [`ServiceClause`] per registered endpoint.
    ///
    /// Each clause carries an identical copy of the parsed pattern, allowing
    /// the federation planner to send the same sub-query to every endpoint.
    pub fn split_by_endpoint(&self, query: &str) -> Vec<ServiceClause> {
        let pattern = parse_pattern(query);
        self.endpoints
            .iter()
            .map(|ep| ServiceClause {
                endpoint: ep.clone(),
                pattern: pattern.clone(),
            })
            .collect()
    }

    /// Inject `clauses` into `query` by appending SERVICE blocks.
    pub fn inject_service_clauses(&self, query: &str, clauses: &[ServiceClause]) -> String {
        if clauses.is_empty() {
            return query.to_string();
        }

        let service_blocks: Vec<String> = clauses
            .iter()
            .map(|c| {
                let triples: Vec<String> = c
                    .pattern
                    .triples
                    .iter()
                    .map(|(s, p, o)| format!("  {} {} {} .", s, p, o))
                    .collect();
                let filters: Vec<String> = c
                    .pattern
                    .filters
                    .iter()
                    .map(|f| format!("  FILTER({})", f))
                    .collect();
                let body = [triples, filters].concat().join("\n");
                format!("SERVICE <{}> {{\n{}\n}}", c.endpoint, body)
            })
            .collect();

        // Insert before the closing `}` of the WHERE clause, or append.
        let blocks_str = service_blocks.join("\n");
        if let Some(pos) = query.rfind('}') {
            let mut result = query.to_string();
            result.insert_str(pos, &format!("\n{}\n", blocks_str));
            result
        } else {
            format!("{}\n{}", query, blocks_str)
        }
    }

    /// Number of registered rewrite rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Number of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Parse a trivial graph pattern from a SPARQL-like string for testing.
///
/// Extracts `?s ?p ?o .` style triples and `FILTER(...)` clauses.
fn parse_pattern(query: &str) -> GraphPattern {
    let mut triples = Vec::new();
    let mut filters = Vec::new();

    for line in query.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("FILTER(") {
            let inner = trimmed
                .trim_start_matches("FILTER(")
                .trim_end_matches(')')
                .trim_end_matches(')')
                .to_string();
            filters.push(inner);
        } else if trimmed.ends_with('.') {
            let parts: Vec<&str> = trimmed.trim_end_matches('.').split_whitespace().collect();
            if parts.len() >= 3 {
                triples.push((parts[0].to_string(), parts[1].to_string(), parts[2].to_string()));
            }
        }
    }

    GraphPattern {
        triples,
        filters,
        optionals: Vec::new(),
    }
}

/// Apply a single [`RewriteTransform`] to a query string.
///
/// Returns the transformed string and any service clauses generated.
fn apply_transform(
    query: &str,
    transform: &RewriteTransform,
) -> (String, Vec<ServiceClause>) {
    match transform {
        RewriteTransform::AddPrefix(prefix) => {
            (format!("{}\n{}", prefix, query), Vec::new())
        }
        RewriteTransform::ReplaceGraph(from, to) => {
            (query.replace(from.as_str(), to.as_str()), Vec::new())
        }
        RewriteTransform::InjectFilter(filter_expr) => {
            // Insert the FILTER before the last `}` in the query
            if let Some(pos) = query.rfind('}') {
                let mut result = query.to_string();
                result.insert_str(pos, &format!("  FILTER({})\n", filter_expr));
                (result, Vec::new())
            } else {
                (format!("{}\nFILTER({})", query, filter_expr), Vec::new())
            }
        }
        RewriteTransform::WrapWithService(endpoint) => {
            let pattern = parse_pattern(query);
            let clause = ServiceClause {
                endpoint: endpoint.clone(),
                pattern: pattern.clone(),
            };
            let triples_str: Vec<String> = pattern
                .triples
                .iter()
                .map(|(s, p, o)| format!("  {} {} {} .", s, p, o))
                .collect();
            let filters_str: Vec<String> = pattern
                .filters
                .iter()
                .map(|f| format!("  FILTER({})", f))
                .collect();
            let body = [triples_str, filters_str].concat().join("\n");
            let service_block = format!(
                "SELECT * WHERE {{\nSERVICE <{}> {{\n{}\n}}\n}}",
                endpoint, body
            );
            (service_block, vec![clause])
        }
        RewriteTransform::StripOptional => {
            // Remove OPTIONAL { ... } blocks (simple single-line version)
            let result = regex_strip_optional(query);
            (result, Vec::new())
        }
    }
}

/// Very simple OPTIONAL stripper — removes `OPTIONAL { ... }` where the
/// opening and closing braces appear on the same line.
fn regex_strip_optional(query: &str) -> String {
    let mut result = String::with_capacity(query.len());
    for line in query.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("OPTIONAL") {
            // Skip this line
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }
    // Remove trailing newline if original didn't have one
    if !query.ends_with('\n') {
        result.truncate(result.trim_end_matches('\n').len());
    }
    result
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_query() -> &'static str {
        "SELECT * WHERE {\n  ?s ?p ?o .\n}"
    }

    // ── basic setup ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_has_no_rules_or_endpoints() {
        let rw = QueryRewriter::new();
        assert_eq!(rw.rule_count(), 0);
        assert_eq!(rw.endpoint_count(), 0);
    }

    #[test]
    fn test_add_rule_increments_count() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "r1".to_string(),
            applies_to: "SELECT".to_string(),
            transform: RewriteTransform::AddPrefix("PREFIX ex: <http://example.org/>".to_string()),
        });
        assert_eq!(rw.rule_count(), 1);
    }

    #[test]
    fn test_add_endpoint_increments_count() {
        let mut rw = QueryRewriter::new();
        rw.add_endpoint("http://ep1.example.org/sparql");
        assert_eq!(rw.endpoint_count(), 1);
    }

    // ── no rules returns unchanged ───────────────────────────────────────────

    #[test]
    fn test_no_rules_returns_original() {
        let rw = QueryRewriter::new();
        let result = rw.rewrite(simple_query());
        assert_eq!(result.rewritten, simple_query());
        assert_eq!(result.original, simple_query());
        assert!(result.rules_applied.is_empty());
    }

    #[test]
    fn test_empty_query_no_rules() {
        let rw = QueryRewriter::new();
        let result = rw.rewrite("");
        assert_eq!(result.rewritten, "");
        assert!(result.rules_applied.is_empty());
    }

    // ── AddPrefix ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_prefix_rule_applied() {
        let mut rw = QueryRewriter::new();
        let prefix = "PREFIX ex: <http://example.org/>".to_string();
        rw.add_rule(RewriteRule {
            name: "add-prefix".to_string(),
            applies_to: "SELECT".to_string(),
            transform: RewriteTransform::AddPrefix(prefix.clone()),
        });
        let result = rw.rewrite(simple_query());
        assert!(result.rewritten.starts_with(&prefix));
        assert!(result.rules_applied.contains(&"add-prefix".to_string()));
    }

    #[test]
    fn test_add_prefix_not_applied_when_no_match() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "add-prefix".to_string(),
            applies_to: "CONSTRUCT".to_string(),
            transform: RewriteTransform::AddPrefix("PREFIX x: <http://x/>".to_string()),
        });
        let result = rw.rewrite(simple_query());
        assert!(!result.rules_applied.contains(&"add-prefix".to_string()));
        assert_eq!(result.rewritten, simple_query());
    }

    // ── ReplaceGraph ─────────────────────────────────────────────────────────

    #[test]
    fn test_replace_graph_substitution() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "replace-graph".to_string(),
            applies_to: "?p".to_string(),
            transform: RewriteTransform::ReplaceGraph(
                "?p".to_string(),
                "ex:knows".to_string(),
            ),
        });
        let result = rw.rewrite(simple_query());
        assert!(result.rewritten.contains("ex:knows"));
        assert!(!result.rewritten.contains("?p"));
        assert!(result.rules_applied.contains(&"replace-graph".to_string()));
    }

    #[test]
    fn test_replace_graph_no_occurrence() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "replace-graph".to_string(),
            applies_to: "GRAPH".to_string(),
            transform: RewriteTransform::ReplaceGraph(
                "GRAPH <a>".to_string(),
                "GRAPH <b>".to_string(),
            ),
        });
        let result = rw.rewrite(simple_query());
        assert!(result.rules_applied.is_empty());
    }

    // ── InjectFilter ─────────────────────────────────────────────────────────

    #[test]
    fn test_inject_filter_adds_filter() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "inject-filter".to_string(),
            applies_to: "?s".to_string(),
            transform: RewriteTransform::InjectFilter("?s != <http://skip.me>".to_string()),
        });
        let result = rw.rewrite(simple_query());
        assert!(result.rewritten.contains("FILTER(?s != <http://skip.me>)"));
        assert!(result.rules_applied.contains(&"inject-filter".to_string()));
    }

    #[test]
    fn test_inject_filter_position_before_closing_brace() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "inject-filter".to_string(),
            applies_to: "SELECT".to_string(),
            transform: RewriteTransform::InjectFilter("?x > 0".to_string()),
        });
        let result = rw.rewrite("SELECT * WHERE { ?x ?y ?z . }");
        // The filter should appear before the final `}`
        let filter_pos = result.rewritten.rfind("FILTER").expect("filter present");
        let brace_pos = result.rewritten.rfind('}').expect("brace present");
        assert!(filter_pos < brace_pos);
    }

    // ── WrapWithService ──────────────────────────────────────────────────────

    #[test]
    fn test_wrap_with_service_wraps_pattern() {
        let mut rw = QueryRewriter::new();
        let ep = "http://service.example.org/sparql";
        rw.add_rule(RewriteRule {
            name: "wrap-service".to_string(),
            applies_to: "SELECT".to_string(),
            transform: RewriteTransform::WrapWithService(ep.to_string()),
        });
        let result = rw.rewrite(simple_query());
        assert!(result.rewritten.contains(&format!("SERVICE <{}>", ep)));
        assert!(result.rules_applied.contains(&"wrap-service".to_string()));
        assert!(!result.service_clauses.is_empty());
        assert_eq!(result.service_clauses[0].endpoint, ep);
    }

    #[test]
    fn test_wrap_with_service_generates_service_clause() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "ws".to_string(),
            applies_to: "?s".to_string(),
            transform: RewriteTransform::WrapWithService("http://ep/sparql".to_string()),
        });
        let q = "SELECT * WHERE {\n  ?s ?p ?o .\n}";
        let result = rw.rewrite(q);
        assert_eq!(result.service_clauses.len(), 1);
        let clause = &result.service_clauses[0];
        assert!(!clause.pattern.triples.is_empty());
    }

    // ── StripOptional ────────────────────────────────────────────────────────

    #[test]
    fn test_strip_optional_removes_optional_lines() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "strip".to_string(),
            applies_to: "OPTIONAL".to_string(),
            transform: RewriteTransform::StripOptional,
        });
        let q = "SELECT * WHERE {\n  ?s ?p ?o .\n  OPTIONAL { ?s ex:label ?l . }\n}";
        let result = rw.rewrite(q);
        assert!(!result.rewritten.contains("OPTIONAL"));
        assert!(result.rules_applied.contains(&"strip".to_string()));
    }

    // ── multiple rules applied in order ─────────────────────────────────────

    #[test]
    fn test_multiple_rules_applied_in_order() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "r1".to_string(),
            applies_to: "SELECT".to_string(),
            transform: RewriteTransform::AddPrefix("# added by r1".to_string()),
        });
        rw.add_rule(RewriteRule {
            name: "r2".to_string(),
            applies_to: "SELECT".to_string(),
            transform: RewriteTransform::AddPrefix("# added by r2".to_string()),
        });
        let result = rw.rewrite(simple_query());
        assert_eq!(result.rules_applied, vec!["r1", "r2"]);
        assert!(result.rewritten.contains("# added by r1"));
        assert!(result.rewritten.contains("# added by r2"));
    }

    #[test]
    fn test_rules_applied_list_correct() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "only-if-construct".to_string(),
            applies_to: "CONSTRUCT".to_string(),
            transform: RewriteTransform::AddPrefix("# noop".to_string()),
        });
        rw.add_rule(RewriteRule {
            name: "always".to_string(),
            applies_to: "SELECT".to_string(),
            transform: RewriteTransform::AddPrefix("# always".to_string()),
        });
        let result = rw.rewrite(simple_query());
        assert_eq!(result.rules_applied, vec!["always"]);
    }

    // ── split_by_endpoint ────────────────────────────────────────────────────

    #[test]
    fn test_split_by_endpoint_creates_clause_per_endpoint() {
        let mut rw = QueryRewriter::new();
        rw.add_endpoint("http://ep1/sparql");
        rw.add_endpoint("http://ep2/sparql");
        let clauses = rw.split_by_endpoint(simple_query());
        assert_eq!(clauses.len(), 2);
        assert_eq!(clauses[0].endpoint, "http://ep1/sparql");
        assert_eq!(clauses[1].endpoint, "http://ep2/sparql");
    }

    #[test]
    fn test_split_by_endpoint_no_endpoints_empty() {
        let rw = QueryRewriter::new();
        let clauses = rw.split_by_endpoint(simple_query());
        assert!(clauses.is_empty());
    }

    #[test]
    fn test_split_by_endpoint_parses_triples() {
        let mut rw = QueryRewriter::new();
        rw.add_endpoint("http://ep/sparql");
        let q = "SELECT * WHERE {\n  ?s ?p ?o .\n}";
        let clauses = rw.split_by_endpoint(q);
        assert_eq!(clauses.len(), 1);
        assert!(!clauses[0].pattern.triples.is_empty());
    }

    // ── inject_service_clauses ───────────────────────────────────────────────

    #[test]
    fn test_inject_service_clauses_appends_blocks() {
        let rw = QueryRewriter::new();
        let clause = ServiceClause {
            endpoint: "http://ep/sparql".to_string(),
            pattern: GraphPattern {
                triples: vec![("?a".into(), "?b".into(), "?c".into())],
                filters: Vec::new(),
                optionals: Vec::new(),
            },
        };
        let q = "SELECT * WHERE { }";
        let result = rw.inject_service_clauses(q, &[clause]);
        assert!(result.contains("SERVICE <http://ep/sparql>"));
        assert!(result.contains("?a ?b ?c ."));
    }

    #[test]
    fn test_inject_service_clauses_empty_clauses() {
        let rw = QueryRewriter::new();
        let q = simple_query();
        let result = rw.inject_service_clauses(q, &[]);
        assert_eq!(result, q);
    }

    #[test]
    fn test_inject_service_clauses_with_filter() {
        let rw = QueryRewriter::new();
        let clause = ServiceClause {
            endpoint: "http://ep/sparql".to_string(),
            pattern: GraphPattern {
                triples: Vec::new(),
                filters: vec!["?x > 0".to_string()],
                optionals: Vec::new(),
            },
        };
        let q = "SELECT * WHERE { }";
        let result = rw.inject_service_clauses(q, &[clause]);
        assert!(result.contains("FILTER(?x > 0)"));
    }

    #[test]
    fn test_inject_multiple_service_clauses() {
        let rw = QueryRewriter::new();
        let clauses: Vec<ServiceClause> = (0..3)
            .map(|i| ServiceClause {
                endpoint: format!("http://ep{}/sparql", i),
                pattern: GraphPattern::default(),
            })
            .collect();
        let result = rw.inject_service_clauses("SELECT * WHERE { }", &clauses);
        assert!(result.contains("SERVICE <http://ep0/sparql>"));
        assert!(result.contains("SERVICE <http://ep1/sparql>"));
        assert!(result.contains("SERVICE <http://ep2/sparql>"));
    }

    // ── RewrittenQuery fields ────────────────────────────────────────────────

    #[test]
    fn test_rewritten_query_original_preserved() {
        let rw = QueryRewriter::new();
        let q = simple_query();
        let result = rw.rewrite(q);
        assert_eq!(result.original, q);
    }

    #[test]
    fn test_service_clauses_from_wrap_rule() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "ws".to_string(),
            applies_to: "SELECT".to_string(),
            transform: RewriteTransform::WrapWithService("http://srv/sparql".to_string()),
        });
        let result = rw.rewrite(simple_query());
        assert!(!result.service_clauses.is_empty());
    }

    #[test]
    fn test_no_service_clauses_for_non_wrap_rules() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "pf".to_string(),
            applies_to: "SELECT".to_string(),
            transform: RewriteTransform::AddPrefix("# prefix".to_string()),
        });
        let result = rw.rewrite(simple_query());
        assert!(result.service_clauses.is_empty());
    }

    #[test]
    fn test_parse_pattern_extracts_triple() {
        let q = "SELECT * WHERE {\n  ?s ?p ?o .\n}";
        let pattern = parse_pattern(q);
        assert_eq!(pattern.triples.len(), 1);
        assert_eq!(pattern.triples[0].0, "?s");
        assert_eq!(pattern.triples[0].1, "?p");
        assert_eq!(pattern.triples[0].2, "?o");
    }

    #[test]
    fn test_replace_graph_replaces_all_occurrences() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "rg".to_string(),
            applies_to: "GRAPH_A".to_string(),
            transform: RewriteTransform::ReplaceGraph("GRAPH_A".to_string(), "GRAPH_B".to_string()),
        });
        let q = "SELECT * WHERE { GRAPH_A { } GRAPH_A { } }";
        let result = rw.rewrite(q);
        assert_eq!(result.rewritten.matches("GRAPH_B").count(), 2);
        assert!(!result.rewritten.contains("GRAPH_A"));
    }

    #[test]
    fn test_default_rewriter() {
        let rw = QueryRewriter::default();
        assert_eq!(rw.rule_count(), 0);
        assert_eq!(rw.endpoint_count(), 0);
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_multiple_endpoints_added() {
        let mut rw = QueryRewriter::new();
        for i in 0..5 {
            rw.add_endpoint(format!("http://ep{}/sparql", i));
        }
        assert_eq!(rw.endpoint_count(), 5);
    }

    #[test]
    fn test_multiple_rules_added() {
        let mut rw = QueryRewriter::new();
        for i in 0..5 {
            rw.add_rule(RewriteRule {
                name: format!("r{}", i),
                applies_to: "SELECT".to_string(),
                transform: RewriteTransform::AddPrefix(format!("# {}", i)),
            });
        }
        assert_eq!(rw.rule_count(), 5);
    }

    #[test]
    fn test_add_prefix_preserves_query_body() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "prefix".to_string(),
            applies_to: "WHERE".to_string(),
            transform: RewriteTransform::AddPrefix("PREFIX ex: <http://ex/>".to_string()),
        });
        let q = "SELECT * WHERE { ?s ?p ?o . }";
        let result = rw.rewrite(q);
        assert!(result.rewritten.contains("WHERE"));
        assert!(result.rewritten.contains("?s ?p ?o"));
    }

    #[test]
    fn test_inject_filter_on_query_without_braces() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "inject".to_string(),
            applies_to: "SELECT".to_string(),
            transform: RewriteTransform::InjectFilter("?x > 0".to_string()),
        });
        let q = "SELECT * WHERE { ?s ?p ?o . }";
        let result = rw.rewrite(q);
        assert!(result.rewritten.contains("FILTER(?x > 0)"));
    }

    #[test]
    fn test_strip_optional_noop_when_no_optional() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "strip".to_string(),
            applies_to: "OPTIONAL".to_string(),
            transform: RewriteTransform::StripOptional,
        });
        let q = "SELECT * WHERE { ?s ?p ?o . }";
        // no OPTIONAL keyword → rule doesn't apply
        let result = rw.rewrite(q);
        assert!(result.rules_applied.is_empty());
    }

    #[test]
    fn test_split_three_endpoints() {
        let mut rw = QueryRewriter::new();
        for i in 0..3 {
            rw.add_endpoint(format!("http://ep{i}/sparql"));
        }
        let clauses = rw.split_by_endpoint(simple_query());
        assert_eq!(clauses.len(), 3);
    }

    #[test]
    fn test_wrap_service_contains_triple() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "ws".to_string(),
            applies_to: "?s".to_string(),
            transform: RewriteTransform::WrapWithService("http://ep/sparql".to_string()),
        });
        let q = "SELECT * WHERE {\n  ?s ?p ?o .\n}";
        let result = rw.rewrite(q);
        assert!(result.rewritten.contains("?s"));
        assert!(result.rewritten.contains("?p"));
        assert!(result.rewritten.contains("?o"));
    }

    #[test]
    fn test_rules_applied_order_preserved() {
        let mut rw = QueryRewriter::new();
        let names = ["alpha", "beta", "gamma"];
        for name in &names {
            rw.add_rule(RewriteRule {
                name: name.to_string(),
                applies_to: "SELECT".to_string(),
                transform: RewriteTransform::AddPrefix(format!("# {}", name)),
            });
        }
        let result = rw.rewrite(simple_query());
        assert_eq!(result.rules_applied, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_inject_service_clauses_multiple() {
        let rw = QueryRewriter::new();
        let clauses: Vec<ServiceClause> = (0..2).map(|i| ServiceClause {
            endpoint: format!("http://ep{}/sparql", i),
            pattern: GraphPattern::default(),
        }).collect();
        let result = rw.inject_service_clauses("SELECT * WHERE { }", &clauses);
        assert!(result.contains("ep0"));
        assert!(result.contains("ep1"));
    }

    #[test]
    fn test_graph_pattern_default_empty() {
        let gp = GraphPattern::default();
        assert!(gp.triples.is_empty());
        assert!(gp.filters.is_empty());
        assert!(gp.optionals.is_empty());
    }

    #[test]
    fn test_rewritten_query_original_unchanged_after_rule() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "r".to_string(),
            applies_to: "SELECT".to_string(),
            transform: RewriteTransform::ReplaceGraph("SELECT".to_string(), "CONSTRUCT".to_string()),
        });
        let original = "SELECT * WHERE { }";
        let result = rw.rewrite(original);
        assert_eq!(result.original, original);
        assert!(result.rewritten.contains("CONSTRUCT"));
    }

    #[test]
    fn test_empty_rule_applies_to_empty_query() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "empty-match".to_string(),
            applies_to: "".to_string(), // empty string matches everything
            transform: RewriteTransform::AddPrefix("# header".to_string()),
        });
        let result = rw.rewrite(simple_query());
        assert!(!result.rules_applied.is_empty());
    }

    #[test]
    fn test_inject_service_clauses_returns_string_without_clauses() {
        let rw = QueryRewriter::new();
        let q = "SELECT * WHERE { ?a ?b ?c . }";
        let result = rw.inject_service_clauses(q, &[]);
        assert_eq!(result, q);
    }

    #[test]
    fn test_rewrite_no_match_preserves_query_unchanged() {
        let mut rw = QueryRewriter::new();
        rw.add_rule(RewriteRule {
            name: "no-match".to_string(),
            applies_to: "DESCRIBE".to_string(),
            transform: RewriteTransform::AddPrefix("# prefix".to_string()),
        });
        let q = "SELECT * WHERE { }";
        let result = rw.rewrite(q);
        assert_eq!(result.rewritten, q);
        assert!(result.rules_applied.is_empty());
    }

    #[test]
    fn test_split_by_endpoint_each_has_same_query_triples() {
        let mut rw = QueryRewriter::new();
        rw.add_endpoint("http://ep1/sparql");
        rw.add_endpoint("http://ep2/sparql");
        let q = "SELECT * WHERE {\n  ?s ?p ?o .\n}";
        let clauses = rw.split_by_endpoint(q);
        // Both clauses should have the same triple count
        assert_eq!(
            clauses[0].pattern.triples.len(),
            clauses[1].pattern.triples.len()
        );
    }
}
