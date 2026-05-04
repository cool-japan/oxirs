use crate::error::{AppError, AppResult};
use serde::{Deserialize, Serialize};

/// A single triple pattern in a SPARQL query graph pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriplePattern {
    pub id: String,
    /// Subject: `"?s"`, `"<http://...>"`, or blank node.
    pub subject: String,
    /// Predicate: `"?p"`, `"<http://...>"`, `"a"` (rdf:type shorthand), or `prefix:local`.
    pub predicate: String,
    /// Object: `"?o"`, `"<http://...>"`, literal, or blank node.
    pub object: String,
    /// Canvas position X (pixels).
    pub x: f32,
    /// Canvas position Y (pixels).
    pub y: f32,
}

/// A FILTER expression node in the visual builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterNode {
    pub id: String,
    /// SPARQL expression string, e.g. `"?age > 18"`.
    pub expression: String,
    /// Canvas position X (pixels).
    pub x: f32,
    /// Canvas position Y (pixels).
    pub y: f32,
}

/// A join edge connecting two triple patterns via a shared variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinEdge {
    pub from_pattern_id: String,
    pub to_pattern_id: String,
    /// The shared variable that joins the two patterns.
    pub variable: String,
}

/// The full visual query graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryGraph {
    pub patterns: Vec<TriplePattern>,
    pub filters: Vec<FilterNode>,
    pub edges: Vec<JoinEdge>,
    /// Variables in the SELECT clause; if empty, all variables are collected automatically.
    pub select_vars: Vec<String>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

/// Result of SPARQL generation from a [`QueryGraph`].
#[derive(Debug, Serialize, Deserialize)]
pub struct GeneratedQuery {
    pub sparql: String,
    pub warnings: Vec<String>,
}

/// Outcome of [`validate_sparql`]: structural validity check.
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

/// Convert a visual [`QueryGraph`] to a SPARQL SELECT query string.
#[tauri::command]
pub fn generate_sparql(graph: QueryGraph) -> AppResult<GeneratedQuery> {
    let mut warnings = Vec::new();

    // Determine select variables: use explicit list or collect from patterns.
    let select_vars: Vec<String> = if graph.select_vars.is_empty() {
        collect_variables(&graph.patterns)
    } else {
        graph.select_vars.clone()
    };

    if select_vars.is_empty() {
        warnings.push("No variables found; query will select nothing.".to_string());
    }

    // Build SELECT clause.
    let select_clause = if select_vars.is_empty() {
        "SELECT *".to_string()
    } else {
        format!("SELECT {}", select_vars.join(" "))
    };

    // Build WHERE clause body.
    let mut where_parts: Vec<String> = Vec::new();

    for p in &graph.patterns {
        let subj = format_term(&p.subject);
        let pred = format_term(&p.predicate);
        let obj = format_term(&p.object);
        where_parts.push(format!("  {} {} {} .", subj, pred, obj));
    }

    for f in &graph.filters {
        if !f.expression.trim().is_empty() {
            where_parts.push(format!("  FILTER ( {} )", f.expression.trim()));
        }
    }

    let where_block = if where_parts.is_empty() {
        "  # No patterns added yet".to_string()
    } else {
        where_parts.join("\n")
    };

    let mut sparql = format!("{}\nWHERE {{\n{}\n}}", select_clause, where_block);

    if let Some(limit) = graph.limit {
        sparql.push_str(&format!("\nLIMIT {}", limit));
    }
    if let Some(offset) = graph.offset {
        sparql.push_str(&format!("\nOFFSET {}", offset));
    }

    Ok(GeneratedQuery { sparql, warnings })
}

/// Validate a SPARQL query string (structural check only — not full parsing).
#[tauri::command]
pub fn validate_sparql(query: String) -> AppResult<ValidationResult> {
    let trimmed = query.trim().to_uppercase();

    if trimmed.is_empty() {
        return Ok(ValidationResult {
            valid: false,
            error: Some("Query is empty".to_string()),
        });
    }

    // Basic structural checks.
    if !trimmed.starts_with("SELECT")
        && !trimmed.starts_with("CONSTRUCT")
        && !trimmed.starts_with("ASK")
        && !trimmed.starts_with("DESCRIBE")
    {
        return Ok(ValidationResult {
            valid: false,
            error: Some("Query must start with SELECT, CONSTRUCT, ASK, or DESCRIBE".to_string()),
        });
    }

    if !trimmed.contains("WHERE") {
        return Ok(ValidationResult {
            valid: false,
            error: Some("Query must contain a WHERE clause".to_string()),
        });
    }

    let open = query.chars().filter(|&c| c == '{').count();
    let close = query.chars().filter(|&c| c == '}').count();
    if open != close {
        return Ok(ValidationResult {
            valid: false,
            error: Some(format!("Unbalanced braces: {} {{ vs {} }}", open, close)),
        });
    }

    Ok(ValidationResult {
        valid: true,
        error: None,
    })
}

/// Parse a SPARQL query string and extract its triple patterns for visualization.
///
/// Uses a minimal line-by-line parser to extract the WHERE block body.
/// Full SPARQL parsing is handled by the oxirs-arq engine; this command is
/// intended for round-tripping simple queries in the visual builder.
#[tauri::command]
pub fn parse_sparql_to_graph(query: String) -> AppResult<QueryGraph> {
    let upper = query.to_uppercase();
    let where_start = upper
        .find("WHERE")
        .ok_or_else(|| AppError::Chat("No WHERE clause found".to_string()))?;

    let after_where = &query[where_start + 5..];
    let block_start = after_where
        .find('{')
        .ok_or_else(|| AppError::Chat("No '{' after WHERE".to_string()))?;
    let block = &after_where[block_start + 1..];
    let block_end = block.rfind('}').unwrap_or(block.len());
    let body = &block[..block_end];

    let mut patterns = Vec::new();
    let mut y = 50.0f32;

    for (i, line) in body.lines().enumerate() {
        // Strip leading/trailing whitespace, then remove trailing period and any
        // remaining whitespace (e.g. "?s ?p ?o ." → "?s ?p ?o").
        let trimmed = line.trim().trim_end_matches('.').trim();
        if trimmed.is_empty() || trimmed.to_uppercase().starts_with("FILTER") {
            continue;
        }
        let parts: Vec<&str> = trimmed.splitn(3, ' ').collect();
        if parts.len() == 3 {
            patterns.push(TriplePattern {
                id: format!("p{}", i),
                subject: parts[0].trim().to_string(),
                predicate: parts[1].trim().to_string(),
                object: parts[2].trim().to_string(),
                x: 100.0,
                y,
            });
            y += 80.0;
        }
    }

    Ok(QueryGraph {
        patterns,
        filters: Vec::new(),
        edges: Vec::new(),
        select_vars: Vec::new(),
        limit: None,
        offset: None,
    })
}

/// Return a list of example SPARQL queries for the UI's "Examples" dropdown.
///
/// Each entry is a `(label, sparql)` tuple.
#[tauri::command]
pub fn get_example_queries() -> Vec<(String, String)> {
    vec![
        (
            "All triples (100)".to_string(),
            "SELECT ?s ?p ?o\nWHERE {\n  ?s ?p ?o .\n}\nLIMIT 100".to_string(),
        ),
        (
            "Classes".to_string(),
            "SELECT DISTINCT ?class\nWHERE {\n  ?s a ?class .\n}\nORDER BY ?class".to_string(),
        ),
        (
            "Properties of a subject".to_string(),
            "SELECT ?p ?o\nWHERE {\n  <http://example.org/subject> ?p ?o .\n}".to_string(),
        ),
        (
            "Count by type".to_string(),
            "SELECT ?type (COUNT(?s) AS ?count)\nWHERE {\n  ?s a ?type .\n}\nGROUP BY ?type\nORDER BY DESC(?count)"
                .to_string(),
        ),
    ]
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Collect all `?variable` names from a slice of triple patterns, deduplicating.
fn collect_variables(patterns: &[TriplePattern]) -> Vec<String> {
    let mut vars: Vec<String> = Vec::new();
    for p in patterns {
        for term in [&p.subject, &p.predicate, &p.object] {
            if term.starts_with('?') && !vars.contains(term) {
                vars.push(term.clone());
            }
        }
    }
    vars
}

/// Format a SPARQL term for output.
///
/// Handles:
/// - `?variable` — passed through unchanged
/// - `<URI>` — passed through unchanged
/// - `"literal"` / `'literal'` — passed through unchanged
/// - `a` — RDF rdf:type shorthand, passed through unchanged
/// - `prefix:local` — passed through unchanged (contains `:`)
/// - bare word — wrapped as `?word` (treats unknown bare tokens as variables)
/// - empty string — replaced with `?_`
fn format_term(term: &str) -> String {
    let t = term.trim();
    if t == "a" {
        // RDF rdf:type shorthand — preserve as-is.
        return "a".to_string();
    }
    if t.starts_with('?') || t.starts_with('<') || t.starts_with('"') || t.starts_with('\'') {
        return t.to_string();
    }
    if t.contains(':') {
        // prefix:local or full IRI without angle brackets.
        return t.to_string();
    }
    if !t.is_empty() {
        // Bare word treated as variable.
        return format!("?{}", t);
    }
    "?_".to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_graph() -> QueryGraph {
        QueryGraph {
            patterns: Vec::new(),
            filters: Vec::new(),
            edges: Vec::new(),
            select_vars: Vec::new(),
            limit: None,
            offset: None,
        }
    }

    fn triple(s: &str, p: &str, o: &str) -> TriplePattern {
        TriplePattern {
            id: "p0".to_string(),
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            x: 0.0,
            y: 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // generate_sparql
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_sparql_empty_graph() {
        let g = empty_graph();
        let r = generate_sparql(g).expect("generate_sparql failed");
        assert!(r.sparql.contains("SELECT *"));
        assert!(r.sparql.contains("WHERE"));
        assert!(!r.warnings.is_empty());
    }

    #[test]
    fn test_generate_sparql_single_pattern() {
        let mut g = empty_graph();
        g.patterns.push(triple("?s", "a", "?type"));
        let r = generate_sparql(g).expect("generate_sparql failed");
        // "a" must be preserved as the rdf:type shorthand, not turned into "?a".
        assert!(
            r.sparql.contains("?s a ?type ."),
            "expected '?s a ?type .' in:\n{}",
            r.sparql
        );
    }

    #[test]
    fn test_generate_sparql_with_filter() {
        let mut g = empty_graph();
        g.patterns.push(triple("?s", "?p", "?o"));
        g.filters.push(FilterNode {
            id: "f0".to_string(),
            expression: "?o > 10".to_string(),
            x: 0.0,
            y: 0.0,
        });
        let r = generate_sparql(g).expect("generate_sparql failed");
        assert!(r.sparql.contains("FILTER ( ?o > 10 )"));
    }

    #[test]
    fn test_generate_sparql_with_limit() {
        let mut g = empty_graph();
        g.limit = Some(100);
        let r = generate_sparql(g).expect("generate_sparql failed");
        assert!(r.sparql.contains("LIMIT 100"));
    }

    #[test]
    fn test_generate_sparql_with_offset() {
        let mut g = empty_graph();
        g.limit = Some(10);
        g.offset = Some(20);
        let r = generate_sparql(g).expect("generate_sparql failed");
        assert!(r.sparql.contains("LIMIT 10"));
        assert!(r.sparql.contains("OFFSET 20"));
    }

    #[test]
    fn test_generate_sparql_explicit_select_vars() {
        let mut g = empty_graph();
        g.patterns.push(triple("?s", "?p", "?o"));
        g.select_vars = vec!["?s".to_string(), "?o".to_string()];
        let r = generate_sparql(g).expect("generate_sparql failed");
        assert!(r.sparql.contains("SELECT ?s ?o"));
    }

    #[test]
    fn test_generate_sparql_multiple_patterns() {
        let mut g = empty_graph();
        g.patterns.push(triple("?s", "a", "?class"));
        g.patterns
            .push(triple("?s", "<http://xmlns.com/foaf/0.1/name>", "?name"));
        let r = generate_sparql(g).expect("generate_sparql failed");
        assert!(r.sparql.contains("?s a ?class ."));
        assert!(r.sparql.contains("<http://xmlns.com/foaf/0.1/name>"));
    }

    #[test]
    fn test_generate_sparql_prefix_predicate() {
        let mut g = empty_graph();
        g.patterns.push(triple("?s", "foaf:name", "?name"));
        let r = generate_sparql(g).expect("generate_sparql failed");
        // foaf:name contains ':' so it must be preserved as-is.
        assert!(r.sparql.contains("foaf:name"));
        // Must NOT be turned into a variable.
        assert!(!r.sparql.contains("?foaf:name"));
    }

    #[test]
    fn test_generate_sparql_empty_filter_skipped() {
        let mut g = empty_graph();
        g.patterns.push(triple("?s", "?p", "?o"));
        g.filters.push(FilterNode {
            id: "f0".to_string(),
            expression: "   ".to_string(), // whitespace only — should be skipped
            x: 0.0,
            y: 0.0,
        });
        let r = generate_sparql(g).expect("generate_sparql failed");
        assert!(!r.sparql.contains("FILTER"));
    }

    // -----------------------------------------------------------------------
    // validate_sparql
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_sparql_valid() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . }".to_string();
        let r = validate_sparql(q).expect("validate_sparql failed");
        assert!(r.valid);
        assert!(r.error.is_none());
    }

    #[test]
    fn test_validate_sparql_empty() {
        let r = validate_sparql(String::new()).expect("validate_sparql failed");
        assert!(!r.valid);
        assert!(r.error.is_some());
    }

    #[test]
    fn test_validate_sparql_no_where() {
        let r = validate_sparql("SELECT ?s { ?s ?p ?o . }".to_string())
            .expect("validate_sparql failed");
        assert!(!r.valid);
        let msg = r.error.as_deref().unwrap_or("");
        assert!(msg.contains("WHERE"), "expected WHERE mention in: {msg}");
    }

    #[test]
    fn test_validate_sparql_unbalanced_braces() {
        let r = validate_sparql("SELECT ?s WHERE { ?s ?p ?o .".to_string())
            .expect("validate_sparql failed");
        assert!(!r.valid);
    }

    #[test]
    fn test_validate_sparql_construct() {
        let q = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o . }".to_string();
        let r = validate_sparql(q).expect("validate_sparql failed");
        assert!(r.valid);
    }

    #[test]
    fn test_validate_sparql_ask() {
        let q = "ASK WHERE { ?s ?p ?o . }".to_string();
        let r = validate_sparql(q).expect("validate_sparql failed");
        assert!(r.valid);
    }

    #[test]
    fn test_validate_sparql_describe() {
        let q = "DESCRIBE <http://example.org/s> WHERE { ?s ?p ?o . }".to_string();
        let r = validate_sparql(q).expect("validate_sparql failed");
        assert!(r.valid);
    }

    #[test]
    fn test_validate_sparql_unknown_keyword() {
        let r = validate_sparql("INSERT DATA { <s> <p> <o> }".to_string())
            .expect("validate_sparql failed");
        assert!(!r.valid);
    }

    // -----------------------------------------------------------------------
    // get_example_queries
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_example_queries_nonempty() {
        let examples = get_example_queries();
        assert!(examples.len() >= 4);
        for (name, sparql) in &examples {
            assert!(!name.is_empty(), "example name must not be empty");
            assert!(
                sparql.contains("WHERE"),
                "example '{name}' must contain WHERE"
            );
        }
    }

    #[test]
    fn test_get_example_queries_are_valid() {
        for (name, sparql) in get_example_queries() {
            let r = validate_sparql(sparql).expect("validate_sparql failed");
            assert!(r.valid, "example '{name}' failed validation: {:?}", r.error);
        }
    }

    // -----------------------------------------------------------------------
    // parse_sparql_to_graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_sparql_to_graph_single_triple() {
        let q = "SELECT ?s ?p ?o\nWHERE {\n  ?s ?p ?o .\n}".to_string();
        let g = parse_sparql_to_graph(q).expect("parse_sparql_to_graph failed");
        assert_eq!(g.patterns.len(), 1);
        assert_eq!(g.patterns[0].subject, "?s");
        assert_eq!(g.patterns[0].predicate, "?p");
        assert_eq!(g.patterns[0].object, "?o");
    }

    #[test]
    fn test_parse_sparql_no_where_fails() {
        let r = parse_sparql_to_graph("SELECT ?s { ?s ?p ?o }".to_string());
        assert!(r.is_err(), "expected error for missing WHERE keyword");
    }

    #[test]
    fn test_parse_sparql_to_graph_multiple_triples() {
        let q = "SELECT ?s ?name\nWHERE {\n  ?s a ?class .\n  ?s foaf:name ?name .\n}".to_string();
        let g = parse_sparql_to_graph(q).expect("parse_sparql_to_graph failed");
        assert_eq!(g.patterns.len(), 2);
    }

    #[test]
    fn test_parse_sparql_to_graph_y_increments() {
        let q = "SELECT ?s ?p ?o\nWHERE {\n  ?s ?p ?o .\n  ?s ?p ?o .\n}".to_string();
        let g = parse_sparql_to_graph(q).expect("parse_sparql_to_graph failed");
        assert_eq!(g.patterns.len(), 2);
        // Each successive pattern should have a higher Y position.
        assert!(g.patterns[1].y > g.patterns[0].y);
    }

    // -----------------------------------------------------------------------
    // collect_variables
    // -----------------------------------------------------------------------

    #[test]
    fn test_collect_variables_deduplicates() {
        let patterns = vec![triple("?s", "?p", "?o"), triple("?s", "a", "?type")];
        let vars = collect_variables(&patterns);
        let count = vars.iter().filter(|v| *v == "?s").count();
        assert_eq!(count, 1, "?s should appear only once in: {vars:?}");
        // "a" is not a variable — must not appear in vars.
        assert!(
            !vars.iter().any(|v| v == "a"),
            "\"a\" must not be listed as a variable"
        );
    }

    #[test]
    fn test_collect_variables_empty_patterns() {
        let vars = collect_variables(&[]);
        assert!(vars.is_empty());
    }

    // -----------------------------------------------------------------------
    // format_term
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_term_variable() {
        assert_eq!(format_term("?s"), "?s");
    }

    #[test]
    fn test_format_term_uri() {
        assert_eq!(
            format_term("<http://example.org/>"),
            "<http://example.org/>"
        );
    }

    #[test]
    fn test_format_term_rdf_type_shorthand() {
        assert_eq!(format_term("a"), "a");
    }

    #[test]
    fn test_format_term_prefix_local() {
        assert_eq!(format_term("foaf:name"), "foaf:name");
    }

    #[test]
    fn test_format_term_literal() {
        assert_eq!(format_term("\"hello\""), "\"hello\"");
    }

    #[test]
    fn test_format_term_bare_word_becomes_variable() {
        assert_eq!(format_term("age"), "?age");
    }

    #[test]
    fn test_format_term_empty_becomes_placeholder() {
        assert_eq!(format_term(""), "?_");
    }
}
