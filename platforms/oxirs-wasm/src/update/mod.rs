//! SPARQL UPDATE operations for OxiRS WASM
//!
//! Implements a subset of SPARQL 1.1 Update:
//! - INSERT DATA { triples }
//! - DELETE DATA { triples }
//! - INSERT { template } WHERE { pattern }
//! - DELETE { template } WHERE { pattern }
//! - CLEAR
//! - DROP

use crate::error::{WasmError, WasmResult};
use crate::store::OxiRSStore;
use std::collections::HashMap;

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// A parsed SPARQL UPDATE operation
#[derive(Debug, Clone)]
pub enum UpdateOperation {
    /// INSERT DATA { triples }
    InsertData(Vec<RawTriple>),
    /// DELETE DATA { triples }
    DeleteData(Vec<RawTriple>),
    /// INSERT { template } WHERE { pattern }
    InsertWhere {
        template: Vec<TemplateTriple>,
        where_patterns: Vec<RawPattern>,
    },
    /// DELETE { template } WHERE { pattern }
    DeleteWhere {
        template: Vec<TemplateTriple>,
        where_patterns: Vec<RawPattern>,
    },
    /// CLEAR – removes all triples
    Clear,
    /// DROP – alias for CLEAR in the embedded store
    Drop,
}

/// A fully resolved triple (no variables)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl RawTriple {
    /// Create a new raw triple
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// A triple that may contain variable references (`?var`) in subject/predicate/object
#[derive(Debug, Clone)]
pub struct TemplateTriple {
    pub subject: TemplateTerm,
    pub predicate: TemplateTerm,
    pub object: TemplateTerm,
}

/// A template term is either a variable or a concrete value
#[derive(Debug, Clone)]
pub enum TemplateTerm {
    Variable(String),
    Value(String),
}

impl TemplateTerm {
    /// Resolve against a binding map, returning the concrete value.
    /// Variables not found in the map are left as the raw variable string.
    pub fn resolve(&self, binding: &HashMap<String, String>) -> String {
        match self {
            TemplateTerm::Variable(v) => {
                binding.get(v).cloned().unwrap_or_else(|| format!("?{}", v))
            }
            TemplateTerm::Value(v) => v.clone(),
        }
    }
}

/// A triple pattern used in WHERE clauses (may contain variables)
#[derive(Debug, Clone)]
pub struct RawPattern {
    pub subject: TemplateTerm,
    pub predicate: TemplateTerm,
    pub object: TemplateTerm,
}

// -----------------------------------------------------------------------
// UpdateParser
// -----------------------------------------------------------------------

/// Parses SPARQL UPDATE strings into [`UpdateOperation`] values
pub struct UpdateParser;

impl UpdateParser {
    /// Parse a SPARQL UPDATE string into an [`UpdateOperation`].
    pub fn parse(sparql: &str) -> WasmResult<UpdateOperation> {
        let s = sparql.trim();
        let upper = s.to_uppercase();

        if upper.starts_with("INSERT DATA") {
            let body = extract_brace_body(s, "INSERT DATA")?;
            let triples = parse_data_block(&body)?;
            Ok(UpdateOperation::InsertData(triples))
        } else if upper.starts_with("DELETE DATA") {
            let body = extract_brace_body(s, "DELETE DATA")?;
            let triples = parse_data_block(&body)?;
            Ok(UpdateOperation::DeleteData(triples))
        } else if upper.starts_with("INSERT") && upper.contains("WHERE") {
            let (template_body, where_body) = split_template_where(s, "INSERT")?;
            let template = parse_template_block(&template_body)?;
            let where_patterns = parse_pattern_block(&where_body)?;
            Ok(UpdateOperation::InsertWhere {
                template,
                where_patterns,
            })
        } else if upper.starts_with("DELETE") && upper.contains("WHERE") {
            let (template_body, where_body) = split_template_where(s, "DELETE")?;
            let template = parse_template_block(&template_body)?;
            let where_patterns = parse_pattern_block(&where_body)?;
            Ok(UpdateOperation::DeleteWhere {
                template,
                where_patterns,
            })
        } else if upper.starts_with("CLEAR") {
            Ok(UpdateOperation::Clear)
        } else if upper.starts_with("DROP") {
            Ok(UpdateOperation::Drop)
        } else {
            Err(WasmError::QueryError(format!(
                "Unknown or unsupported UPDATE operation: {}",
                &s[..s.len().min(80)]
            )))
        }
    }
}

// -----------------------------------------------------------------------
// UpdateExecutor
// -----------------------------------------------------------------------

/// Executes [`UpdateOperation`] values against an [`OxiRSStore`]
pub struct UpdateExecutor;

impl UpdateExecutor {
    /// Apply an already-parsed [`UpdateOperation`] and return the number of
    /// triples affected (inserted or deleted).
    pub fn execute(op: &UpdateOperation, store: &mut OxiRSStore) -> WasmResult<u32> {
        match op {
            UpdateOperation::InsertData(triples) => {
                let mut count = 0u32;
                for t in triples {
                    if store.insert(&t.subject, &t.predicate, &t.object) {
                        count += 1;
                    }
                }
                Ok(count)
            }

            UpdateOperation::DeleteData(triples) => {
                let mut count = 0u32;
                for t in triples {
                    if store.delete(&t.subject, &t.predicate, &t.object) {
                        count += 1;
                    }
                }
                Ok(count)
            }

            UpdateOperation::InsertWhere {
                template,
                where_patterns,
            } => {
                let bindings = evaluate_where_patterns(where_patterns, store)?;
                let mut count = 0u32;
                for binding in &bindings {
                    for t in template {
                        let s = t.subject.resolve(binding);
                        let p = t.predicate.resolve(binding);
                        let o = t.object.resolve(binding);
                        if store.insert(&s, &p, &o) {
                            count += 1;
                        }
                    }
                }
                Ok(count)
            }

            UpdateOperation::DeleteWhere {
                template,
                where_patterns,
            } => {
                let bindings = evaluate_where_patterns(where_patterns, store)?;
                // Collect all triples to delete first to avoid borrow conflicts
                let mut to_delete: Vec<RawTriple> = Vec::new();
                for binding in &bindings {
                    for t in template {
                        let s = t.subject.resolve(binding);
                        let p = t.predicate.resolve(binding);
                        let o = t.object.resolve(binding);
                        to_delete.push(RawTriple::new(s, p, o));
                    }
                }
                let mut count = 0u32;
                for t in to_delete {
                    if store.delete(&t.subject, &t.predicate, &t.object) {
                        count += 1;
                    }
                }
                Ok(count)
            }

            UpdateOperation::Clear | UpdateOperation::Drop => {
                let before = store.size() as u32;
                store.clear();
                Ok(before)
            }
        }
    }
}

// -----------------------------------------------------------------------
// Top-level convenience function
// -----------------------------------------------------------------------

/// Parse and execute a SPARQL UPDATE string against the given store.
///
/// Returns the number of triples affected (inserted or deleted).
pub fn execute_update(sparql: &str, store: &mut OxiRSStore) -> WasmResult<u32> {
    let op = UpdateParser::parse(sparql)?;
    UpdateExecutor::execute(&op, store)
}

// -----------------------------------------------------------------------
// Internal helpers – pattern evaluation
// -----------------------------------------------------------------------

/// Simple SPARQL pattern evaluation for WHERE clauses in UPDATE.
/// Returns all bindings that match the pattern list.
fn evaluate_where_patterns(
    patterns: &[RawPattern],
    store: &OxiRSStore,
) -> WasmResult<Vec<HashMap<String, String>>> {
    let mut results: Vec<HashMap<String, String>> = vec![HashMap::new()];

    for pattern in patterns {
        let mut new_results = Vec::new();
        for binding in &results {
            for triple in store.all_triples() {
                if matches_term(&pattern.subject, &triple.subject, binding)
                    && matches_term(&pattern.predicate, &triple.predicate, binding)
                    && matches_term(&pattern.object, &triple.object, binding)
                {
                    let mut new_binding = binding.clone();
                    bind_variable(&pattern.subject, &triple.subject, &mut new_binding);
                    bind_variable(&pattern.predicate, &triple.predicate, &mut new_binding);
                    bind_variable(&pattern.object, &triple.object, &mut new_binding);
                    new_results.push(new_binding);
                }
            }
        }
        results = new_results;
    }

    Ok(results)
}

fn matches_term(term: &TemplateTerm, value: &str, binding: &HashMap<String, String>) -> bool {
    match term {
        TemplateTerm::Variable(var) => {
            if let Some(bound) = binding.get(var) {
                bound == value
            } else {
                true
            }
        }
        TemplateTerm::Value(v) => v == value,
    }
}

fn bind_variable(term: &TemplateTerm, value: &str, binding: &mut HashMap<String, String>) {
    if let TemplateTerm::Variable(var) = term {
        binding
            .entry(var.clone())
            .or_insert_with(|| value.to_string());
    }
}

// -----------------------------------------------------------------------
// Internal helpers – parsing
// -----------------------------------------------------------------------

/// Extract the content of the `{ }` block that follows a keyword
fn extract_brace_body(sparql: &str, keyword: &str) -> WasmResult<String> {
    let upper = sparql.to_uppercase();
    let kw_upper = keyword.to_uppercase();
    let start = upper
        .find(&kw_upper)
        .ok_or_else(|| WasmError::QueryError(format!("Expected keyword '{}'", keyword)))?
        + keyword.len();

    let after = &sparql[start..];
    let open = after
        .find('{')
        .ok_or_else(|| WasmError::QueryError(format!("Missing '{{' after '{}'", keyword)))?;
    let inner_start = open + 1;

    let chars: Vec<char> = after[inner_start..].chars().collect();
    let mut depth = 1usize;
    let mut pos = 0usize;
    while pos < chars.len() && depth > 0 {
        match chars[pos] {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        if depth > 0 {
            pos += 1;
        }
    }

    if depth != 0 {
        return Err(WasmError::QueryError("Unmatched '{' in UPDATE".to_string()));
    }

    Ok(chars[..pos].iter().collect())
}

/// Split `KEYWORD { template } WHERE { where_body }` into the two bodies
fn split_template_where(sparql: &str, keyword: &str) -> WasmResult<(String, String)> {
    let upper = sparql.to_uppercase();
    let kw_upper = keyword.to_uppercase();

    let kw_start = upper
        .find(&kw_upper)
        .ok_or_else(|| WasmError::QueryError(format!("Expected keyword '{}'", keyword)))?;
    let after_kw = &sparql[kw_start + keyword.len()..];

    // First brace block = template
    let open1 = after_kw
        .find('{')
        .ok_or_else(|| WasmError::QueryError(format!("Missing '{{' after '{}'", keyword)))?;
    let after_kw_inner = &after_kw[open1 + 1..];
    let chars1: Vec<char> = after_kw_inner.chars().collect();
    let mut depth = 1usize;
    let mut pos1 = 0usize;
    while pos1 < chars1.len() && depth > 0 {
        match chars1[pos1] {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        if depth > 0 {
            pos1 += 1;
        }
    }
    let template_body: String = chars1[..pos1].iter().collect();

    // Everything after the closing '}' of the template
    let consumed = kw_start + keyword.len() + open1 + 1 + pos1 + 1;
    let rest = &sparql[consumed..];

    // Find WHERE keyword then second brace block
    let rest_upper = rest.to_uppercase();
    let where_pos = rest_upper
        .find("WHERE")
        .ok_or_else(|| WasmError::QueryError("Missing WHERE clause".to_string()))?;
    let after_where = &rest[where_pos + 5..];
    let open2 = after_where
        .find('{')
        .ok_or_else(|| WasmError::QueryError("Missing '{{' after WHERE".to_string()))?;
    let after_where_inner = &after_where[open2 + 1..];
    let chars2: Vec<char> = after_where_inner.chars().collect();
    let mut depth2 = 1usize;
    let mut pos2 = 0usize;
    while pos2 < chars2.len() && depth2 > 0 {
        match chars2[pos2] {
            '{' => depth2 += 1,
            '}' => depth2 -= 1,
            _ => {}
        }
        if depth2 > 0 {
            pos2 += 1;
        }
    }
    let where_body: String = chars2[..pos2].iter().collect();

    Ok((template_body, where_body))
}

/// Parse a DATA block (no variables) into [`RawTriple`] values
fn parse_data_block(body: &str) -> WasmResult<Vec<RawTriple>> {
    let mut triples = Vec::new();

    for stmt in body.split('.') {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = stmt.split_whitespace().collect();
        if tokens.len() >= 3 {
            let s = iri_from_token(tokens[0]);
            let p = if tokens[1] == "a" {
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()
            } else {
                iri_from_token(tokens[1])
            };
            let obj_raw = tokens[2..].join(" ");
            let o = object_from_token(&obj_raw);
            triples.push(RawTriple::new(s, p, o));
        }
    }

    Ok(triples)
}

/// Parse a template block into [`TemplateTriple`] values (may contain variables)
fn parse_template_block(body: &str) -> WasmResult<Vec<TemplateTriple>> {
    let mut triples = Vec::new();

    for stmt in body.split('.') {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = stmt.split_whitespace().collect();
        if tokens.len() >= 3 {
            let s = template_term_from(tokens[0]);
            let p = if tokens[1] == "a" {
                TemplateTerm::Value("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
            } else {
                template_term_from(tokens[1])
            };
            let obj_raw = tokens[2..].join(" ");
            let o = template_term_from(&obj_raw);
            triples.push(TemplateTriple {
                subject: s,
                predicate: p,
                object: o,
            });
        }
    }

    Ok(triples)
}

/// Parse a WHERE pattern block into [`RawPattern`] values (may contain variables)
fn parse_pattern_block(body: &str) -> WasmResult<Vec<RawPattern>> {
    let mut patterns = Vec::new();

    for stmt in body.split('.') {
        let stmt = stmt.trim();
        if stmt.is_empty() || stmt.to_uppercase().starts_with("FILTER") {
            continue;
        }
        let tokens: Vec<&str> = stmt.split_whitespace().collect();
        if tokens.len() >= 3 {
            let s = template_term_from(tokens[0]);
            let p = if tokens[1] == "a" {
                TemplateTerm::Value("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
            } else {
                template_term_from(tokens[1])
            };
            let obj_raw = tokens[2..].join(" ");
            let o = template_term_from(&obj_raw);
            patterns.push(RawPattern {
                subject: s,
                predicate: p,
                object: o,
            });
        }
    }

    Ok(patterns)
}

fn template_term_from(token: &str) -> TemplateTerm {
    let t = token.trim();
    if t.starts_with('?') || t.starts_with('$') {
        TemplateTerm::Variable(t.trim_start_matches(['?', '$']).to_string())
    } else {
        TemplateTerm::Value(iri_from_token(t))
    }
}

fn iri_from_token(token: &str) -> String {
    let t = token.trim();
    if t.starts_with('<') && t.ends_with('>') {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

fn object_from_token(token: &str) -> String {
    let t = token.trim();
    if t.starts_with('<') && t.ends_with('>') {
        t[1..t.len() - 1].to_string()
    } else {
        // Literals are kept as-is (including surrounding quotes)
        t.to_string()
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> OxiRSStore {
        OxiRSStore::new()
    }

    // INSERT DATA

    #[test]
    fn test_insert_data_single() {
        let mut store = make_store();
        let n = execute_update(
            "INSERT DATA { <http://a> <http://b> <http://c> }",
            &mut store,
        )
        .expect("execute");
        assert_eq!(n, 1);
        assert!(store.contains("http://a", "http://b", "http://c"));
    }

    #[test]
    fn test_insert_data_multiple() {
        let mut store = make_store();
        let n = execute_update(
            "INSERT DATA { <http://a> <http://b> <http://c> . <http://x> <http://y> <http://z> }",
            &mut store,
        )
        .expect("execute");
        assert_eq!(n, 2);
        assert_eq!(store.size(), 2);
    }

    #[test]
    fn test_insert_data_no_duplicates() {
        let mut store = make_store();
        execute_update(
            "INSERT DATA { <http://a> <http://b> <http://c> }",
            &mut store,
        )
        .expect("first");
        let n = execute_update(
            "INSERT DATA { <http://a> <http://b> <http://c> }",
            &mut store,
        )
        .expect("second");
        assert_eq!(n, 0);
        assert_eq!(store.size(), 1);
    }

    #[test]
    fn test_insert_data_type_shortcut() {
        let mut store = make_store();
        let n = execute_update(
            "INSERT DATA { <http://alice> a <http://Person> }",
            &mut store,
        )
        .expect("execute");
        assert_eq!(n, 1);
        assert!(store.contains(
            "http://alice",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://Person"
        ));
    }

    // DELETE DATA

    #[test]
    fn test_delete_data_existing() {
        let mut store = make_store();
        store.insert("http://a", "http://b", "http://c");
        let n = execute_update(
            "DELETE DATA { <http://a> <http://b> <http://c> }",
            &mut store,
        )
        .expect("execute");
        assert_eq!(n, 1);
        assert_eq!(store.size(), 0);
    }

    #[test]
    fn test_delete_data_nonexistent() {
        let mut store = make_store();
        let n = execute_update(
            "DELETE DATA { <http://a> <http://b> <http://c> }",
            &mut store,
        )
        .expect("execute");
        assert_eq!(n, 0);
    }

    // INSERT WHERE

    #[test]
    fn test_insert_where() {
        let mut store = make_store();
        store.insert("http://alice", "http://knows", "http://bob");

        let n = execute_update(
            "INSERT { ?s <http://friend> ?o } WHERE { ?s <http://knows> ?o }",
            &mut store,
        )
        .expect("execute");
        assert_eq!(n, 1);
        assert!(store.contains("http://alice", "http://friend", "http://bob"));
    }

    // DELETE WHERE

    #[test]
    fn test_delete_where() {
        let mut store = make_store();
        store.insert("http://alice", "http://name", "Alice");
        store.insert("http://bob", "http://name", "Bob");
        store.insert("http://alice", "http://age", "30");

        let n = execute_update(
            "DELETE { ?s <http://name> ?o } WHERE { ?s <http://name> ?o }",
            &mut store,
        )
        .expect("execute");
        assert_eq!(n, 2);
        assert!(!store.contains("http://alice", "http://name", "Alice"));
        assert!(!store.contains("http://bob", "http://name", "Bob"));
        assert!(store.contains("http://alice", "http://age", "30"));
    }

    // CLEAR

    #[test]
    fn test_clear() {
        let mut store = make_store();
        store.insert("http://a", "http://b", "http://c");
        store.insert("http://x", "http://y", "http://z");

        let n = execute_update("CLEAR", &mut store).expect("execute");
        assert_eq!(n, 2);
        assert_eq!(store.size(), 0);
    }

    #[test]
    fn test_drop() {
        let mut store = make_store();
        store.insert("http://a", "http://b", "http://c");

        let n = execute_update("DROP", &mut store).expect("execute");
        assert_eq!(n, 1);
        assert_eq!(store.size(), 0);
    }

    #[test]
    fn test_unknown_operation() {
        let mut store = make_store();
        assert!(execute_update("LOAD <http://example.org/data.ttl>", &mut store).is_err());
    }
}
