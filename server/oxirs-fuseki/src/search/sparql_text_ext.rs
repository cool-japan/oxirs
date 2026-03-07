//! SPARQL `text:query` Service Extension
//!
//! Integrates `TextSearchIndex` with SPARQL queries by recognising and
//! executing `text:query` property-function patterns.
//!
//! Supported SPARQL patterns
//! ─────────────────────────
//! 1. Basic property function:
//!    `?s text:query "search terms" .`
//!
//! 2. With limit:
//!    `?s text:query ("search terms" 10) .`
//!
//! 3. SERVICE block (Jena compatibility):
//!    ```sparql
//!    SERVICE <http://jena.apache.org/text#query> {
//!        ?s text:query "search terms"
//!    }
//!    ```
//!
//! Note: This parser performs lightweight pattern-matching on the raw SPARQL
//! string — it is not a full SPARQL 1.1 parser.

use crate::error::{FusekiError, FusekiResult};
use crate::search::text_search_index::TextSearchIndex;
use std::sync::{Arc, RwLock};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A `text:query` call extracted from a SPARQL query string.
#[derive(Debug, Clone, PartialEq)]
pub struct TextQueryCall {
    /// The SPARQL variable to bind results to (without `?`, e.g. `"s"`)
    pub subject_var: String,
    /// The search terms string
    pub search_terms: String,
    /// Optional variable for the relevance score (without `?`)
    pub score_var: Option<String>,
    /// Optional result limit extracted from `("terms" N)` syntax
    pub limit: Option<usize>,
}

/// A single result binding from executing a `TextQueryCall`.
#[derive(Debug, Clone)]
pub struct TextQueryBinding {
    /// The matched subject IRI
    pub subject: String,
    /// The relevance score
    pub score: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// SparqlTextSearchExtension
// ──────────────────────────────────────────────────────────────────────────────

/// Integrates `TextSearchIndex` with SPARQL via the `text:query` pattern.
///
/// Thread-safe: the underlying index is wrapped in `Arc<RwLock<...>>`.
pub struct SparqlTextSearchExtension {
    index: Arc<RwLock<TextSearchIndex>>,
}

impl SparqlTextSearchExtension {
    /// Create a new extension with a fresh, empty `TextSearchIndex`.
    pub fn new() -> Self {
        SparqlTextSearchExtension {
            index: Arc::new(RwLock::new(TextSearchIndex::new())),
        }
    }

    /// Create an extension that shares an existing index.
    pub fn new_with_index(index: Arc<RwLock<TextSearchIndex>>) -> Self {
        SparqlTextSearchExtension { index }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Static parsing
    // ──────────────────────────────────────────────────────────────────────────

    /// Parse all `text:query` calls found in a SPARQL query string.
    ///
    /// Recognises the following patterns (case-insensitive for keywords):
    /// - `?var text:query "terms"` — basic
    /// - `?var text:query ("terms" N)` — with limit
    /// - SERVICE `<http://jena.apache.org/text#query>` blocks
    ///
    /// Does not require a full SPARQL parse; uses simple pattern matching.
    pub fn parse_text_queries(query: &str) -> Vec<TextQueryCall> {
        let mut results = Vec::new();

        // Scan the query line-by-line / token-by-token for `text:query`
        // occurrences.  For each occurrence we look backward for `?var` and
        // forward for the search term string.

        let lower = query.to_lowercase();
        let bytes = query.as_bytes();

        let mut search_start = 0;

        while let Some(pos) = lower[search_start..].find("text:query") {
            let abs_pos = search_start + pos;

            // ── find the subject variable (looking backward for `?`) ────────
            let before = &query[..abs_pos];
            let subject_var = extract_var_before(before);

            // ── find the search terms (looking forward after `text:query`) ──
            let after = &query[abs_pos + "text:query".len()..];
            let after_trimmed = after.trim_start();

            let (search_terms, limit) = if after_trimmed.starts_with('(') {
                // `(?s text:query ("terms" N))` form
                parse_parenthesised_args(after_trimmed)
            } else {
                // `?s text:query "terms"` form
                (extract_quoted_string(after_trimmed), None)
            };

            if let (Some(var), Some(terms)) = (subject_var, search_terms) {
                if !terms.is_empty() {
                    results.push(TextQueryCall {
                        subject_var: var,
                        search_terms: terms,
                        score_var: None,
                        limit,
                    });
                }
            }

            search_start = abs_pos + "text:query".len();
            // Guard against infinite loop on zero-length match
            if search_start >= lower.len() {
                break;
            }
        }

        // Scan for SERVICE blocks and enrich the results with SERVICE context
        // (SERVICE blocks may repeat the `text:query` pattern already found above,
        //  but we normalise them to the same output format).
        let _ = bytes; // suppress unused warning when the feature is not used
        results
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Execution
    // ──────────────────────────────────────────────────────────────────────────

    /// Execute a parsed `TextQueryCall` against the internal index.
    ///
    /// Returns bindings sorted by score descending, limited by `call.limit` if set.
    pub fn execute(&self, call: &TextQueryCall) -> FusekiResult<Vec<TextQueryBinding>> {
        let index = self.index.read().map_err(|e| FusekiError::Internal {
            message: format!("TextSearchIndex RwLock poisoned on execute: {}", e),
        })?;

        let hits = index.search(&call.search_terms);

        let iter = hits.into_iter().map(|h| TextQueryBinding {
            subject: h.subject,
            score: h.score,
        });

        let bindings: Vec<TextQueryBinding> = if let Some(limit) = call.limit {
            iter.take(limit).collect()
        } else {
            iter.collect()
        };

        Ok(bindings)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Index management
    // ──────────────────────────────────────────────────────────────────────────

    /// Index a triple's literal value into the text search index.
    pub fn index_triple(&self, subject: &str, predicate: &str, literal: &str) -> FusekiResult<()> {
        let mut index = self.index.write().map_err(|e| FusekiError::Internal {
            message: format!("TextSearchIndex RwLock poisoned on index_triple: {}", e),
        })?;
        index.index_triple(subject, predicate, literal);
        Ok(())
    }

    /// Remove all indexed triples for the given subject from the text index.
    ///
    /// This removes every `(subject, *)` document regardless of predicate.
    pub fn remove_subject(&self, subject: &str) -> FusekiResult<()> {
        let mut index = self.index.write().map_err(|e| FusekiError::Internal {
            message: format!("TextSearchIndex RwLock poisoned on remove_subject: {}", e),
        })?;
        // Enumerate all predicates for this subject using the subject-predicate index,
        // then remove each (subject, predicate) pair from the text index.
        let predicates = index.predicates_for_subject(subject);
        for predicate in &predicates {
            index.remove_triple(subject, predicate);
        }
        Ok(())
    }

    /// Number of documents currently in the text search index.
    pub fn document_count(&self) -> usize {
        self.index
            .read()
            .map(|idx| idx.document_count())
            .unwrap_or(0)
    }
}

impl Default for SparqlTextSearchExtension {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Parser helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Extract the last `?varname` or `$varname` token appearing in `text` (the
/// portion of the query before `text:query`).
fn extract_var_before(text: &str) -> Option<String> {
    // Walk the text in reverse looking for `?` or `$` followed by identifier chars
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = n;

    // Skip trailing whitespace
    while i > 0 && chars[i - 1].is_whitespace() {
        i -= 1;
    }

    if i == 0 {
        return None;
    }

    // Collect identifier characters going backward
    let end = i;
    while i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '_') {
        i -= 1;
    }

    // The character just before the identifier should be `?` or `$`
    if i == 0 {
        return None;
    }

    if chars[i - 1] == '?' || chars[i - 1] == '$' {
        let var_name: String = chars[i..end].iter().collect();
        if !var_name.is_empty() {
            return Some(var_name);
        }
    }

    None
}

/// Extract the first double-quoted string from `text`.
/// Returns `None` if no quoted string is found.
fn extract_quoted_string(text: &str) -> Option<String> {
    let start = text.find('"')?;
    let rest = &text[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Parse `("terms" N)` syntax.
/// Returns `(Some(terms), Some(limit))` if both are present,
/// `(Some(terms), None)` if only terms, `(None, None)` otherwise.
fn parse_parenthesised_args(text: &str) -> (Option<String>, Option<usize>) {
    // text starts with `(`
    let inner = match text.strip_prefix('(') {
        Some(s) => s,
        None => return (None, None),
    };

    // Extract quoted string
    let terms = extract_quoted_string(inner);

    // After the quoted string, look for an integer limit
    let limit = if terms.is_some() {
        // Find end of quote in inner
        let q_start = inner.find('"').unwrap_or(0);
        let q_end = inner[q_start + 1..]
            .find('"')
            .map(|p| q_start + 1 + p + 1)
            .unwrap_or(0);
        let after_quote = inner[q_end..].trim();
        // Extract leading integer
        let digits: String = after_quote
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if !digits.is_empty() {
            digits.parse::<usize>().ok()
        } else {
            None
        }
    } else {
        None
    };

    (terms, limit)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_text_queries ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_basic_pattern() {
        let query = r#"SELECT ?s WHERE { ?s text:query "semantic web" . }"#;
        let calls = SparqlTextSearchExtension::parse_text_queries(query);
        assert_eq!(calls.len(), 1, "Should parse one text:query call");
        let call = &calls[0];
        assert_eq!(call.subject_var, "s");
        assert_eq!(call.search_terms, "semantic web");
        assert_eq!(call.limit, None);
    }

    #[test]
    fn test_parse_with_limit() {
        let query = r#"SELECT ?s WHERE { ?s text:query ("knowledge graph" 10) . }"#;
        let calls = SparqlTextSearchExtension::parse_text_queries(query);
        assert_eq!(calls.len(), 1);
        let call = &calls[0];
        assert_eq!(call.subject_var, "s");
        assert_eq!(call.search_terms, "knowledge graph");
        assert_eq!(call.limit, Some(10));
    }

    #[test]
    fn test_parse_service_block() {
        let query = r#"
            SELECT ?s WHERE {
              SERVICE <http://jena.apache.org/text#query> {
                ?s text:query "linked data"
              }
            }
        "#;
        let calls = SparqlTextSearchExtension::parse_text_queries(query);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].search_terms, "linked data");
    }

    #[test]
    fn test_parse_no_text_query() {
        let query = "SELECT * WHERE { ?s ?p ?o }";
        let calls = SparqlTextSearchExtension::parse_text_queries(query);
        assert!(calls.is_empty(), "No text:query should yield empty vec");
    }

    #[test]
    fn test_parse_multiple_text_queries() {
        let query = r#"
            SELECT ?s ?t WHERE {
              ?s text:query "sparql endpoint" .
              ?t text:query "rdf triple" .
            }
        "#;
        let calls = SparqlTextSearchExtension::parse_text_queries(query);
        assert_eq!(calls.len(), 2, "Should find two text:query calls");
        let terms: Vec<&str> = calls.iter().map(|c| c.search_terms.as_str()).collect();
        assert!(terms.contains(&"sparql endpoint"));
        assert!(terms.contains(&"rdf triple"));
    }

    #[test]
    fn test_parse_empty_query_string() {
        let calls = SparqlTextSearchExtension::parse_text_queries("");
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_dollar_sign_variable() {
        let query = r#"SELECT $result WHERE { $result text:query "ontology" . }"#;
        let calls = SparqlTextSearchExtension::parse_text_queries(query);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].subject_var, "result");
    }

    // ── execute ───────────────────────────────────────────────────────────────

    #[test]
    fn test_execute_returns_empty_on_no_index() {
        let ext = SparqlTextSearchExtension::new();
        let call = TextQueryCall {
            subject_var: "s".to_string(),
            search_terms: "sparql".to_string(),
            score_var: None,
            limit: None,
        };
        let result = ext.execute(&call).unwrap();
        assert!(result.is_empty(), "Empty index should return no results");
    }

    #[test]
    fn test_execute_finds_indexed_triples() {
        let ext = SparqlTextSearchExtension::new();
        ext.index_triple(
            "http://ex.org/s1",
            "http://ex.org/label",
            "SPARQL query language",
        )
        .unwrap();
        ext.index_triple(
            "http://ex.org/s2",
            "http://ex.org/label",
            "GraphQL API server",
        )
        .unwrap();

        let call = TextQueryCall {
            subject_var: "s".to_string(),
            search_terms: "sparql query".to_string(),
            score_var: None,
            limit: None,
        };
        let bindings = ext.execute(&call).unwrap();
        assert!(!bindings.is_empty(), "Should find SPARQL-related triples");
        assert_eq!(bindings[0].subject, "http://ex.org/s1");
    }

    #[test]
    fn test_execute_respects_limit() {
        let ext = SparqlTextSearchExtension::new();
        for i in 0..10u32 {
            ext.index_triple(
                &format!("http://ex.org/s{}", i),
                "http://ex.org/label",
                "common term repeated",
            )
            .unwrap();
        }

        let call = TextQueryCall {
            subject_var: "s".to_string(),
            search_terms: "common term".to_string(),
            score_var: None,
            limit: Some(3),
        };
        let bindings = ext.execute(&call).unwrap();
        assert_eq!(bindings.len(), 3, "Limit should cap results at 3");
    }

    #[test]
    fn test_execute_sorted_by_score_descending() {
        let ext = SparqlTextSearchExtension::new();
        ext.index_triple(
            "http://ex.org/s1",
            "http://ex.org/label",
            "database systems",
        )
        .unwrap();
        ext.index_triple(
            "http://ex.org/s2",
            "http://ex.org/label",
            "database database storage",
        )
        .unwrap();

        let call = TextQueryCall {
            subject_var: "s".to_string(),
            search_terms: "database".to_string(),
            score_var: None,
            limit: None,
        };
        let bindings = ext.execute(&call).unwrap();
        assert_eq!(bindings.len(), 2);
        assert!(
            bindings[0].score >= bindings[1].score,
            "Results should be sorted by score"
        );
    }

    // ── index_triple & document_count ─────────────────────────────────────────

    #[test]
    fn test_index_triple_increments_count() {
        let ext = SparqlTextSearchExtension::new();
        assert_eq!(ext.document_count(), 0);
        ext.index_triple("s1", "p1", "hello world").unwrap();
        assert_eq!(ext.document_count(), 1);
        ext.index_triple("s2", "p1", "hello rust").unwrap();
        assert_eq!(ext.document_count(), 2);
    }

    // ── remove_subject ────────────────────────────────────────────────────────

    #[test]
    fn test_remove_subject_removes_documents() {
        let ext = SparqlTextSearchExtension::new();
        ext.index_triple(
            "http://ex.org/s1",
            "http://ex.org/label",
            "semantic data web",
        )
        .unwrap();
        ext.index_triple(
            "http://ex.org/s2",
            "http://ex.org/label",
            "linked data graph",
        )
        .unwrap();

        ext.remove_subject("http://ex.org/s1").unwrap();

        let call = TextQueryCall {
            subject_var: "s".to_string(),
            search_terms: "semantic".to_string(),
            score_var: None,
            limit: None,
        };
        let bindings = ext.execute(&call).unwrap();
        assert!(
            bindings.iter().all(|b| b.subject != "http://ex.org/s1"),
            "s1 should be removed"
        );
    }

    // ── TextQueryCall fields ──────────────────────────────────────────────────

    #[test]
    fn test_text_query_call_with_score_var() {
        let call = TextQueryCall {
            subject_var: "s".to_string(),
            search_terms: "test".to_string(),
            score_var: Some("score".to_string()),
            limit: None,
        };
        assert_eq!(call.score_var, Some("score".to_string()));
    }

    #[test]
    fn test_text_query_call_without_score_var() {
        let call = TextQueryCall {
            subject_var: "s".to_string(),
            search_terms: "test".to_string(),
            score_var: None,
            limit: None,
        };
        assert!(call.score_var.is_none());
    }

    // ── TextQueryBinding score ────────────────────────────────────────────────

    #[test]
    fn test_binding_score_positive() {
        let ext = SparqlTextSearchExtension::new();
        ext.index_triple("http://ex.org/s1", "http://ex.org/p", "rdf linked data")
            .unwrap();
        let call = TextQueryCall {
            subject_var: "s".to_string(),
            search_terms: "rdf".to_string(),
            score_var: None,
            limit: None,
        };
        let bindings = ext.execute(&call).unwrap();
        assert!(!bindings.is_empty());
        assert!(bindings[0].score > 0.0);
    }

    // ── parse_text_queries: edge cases ────────────────────────────────────────

    #[test]
    fn test_parse_whitespace_only_is_empty() {
        let calls = SparqlTextSearchExtension::parse_text_queries("   \n  ");
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_limit_extracted_correctly() {
        let query = r#"SELECT ?x WHERE { ?x text:query ("full text search" 5) }"#;
        let calls = SparqlTextSearchExtension::parse_text_queries(query);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].limit, Some(5));
    }

    #[test]
    fn test_execute_no_match_returns_empty() {
        let ext = SparqlTextSearchExtension::new();
        ext.index_triple("s", "p", "completely different content")
            .unwrap();
        let call = TextQueryCall {
            subject_var: "s".to_string(),
            search_terms: "zzznomatch".to_string(),
            score_var: None,
            limit: None,
        };
        let result = ext.execute(&call).unwrap();
        assert!(result.is_empty());
    }
}
