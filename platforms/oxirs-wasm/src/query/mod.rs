//! SPARQL query execution for WASM
//!
//! Supports:
//! - SELECT, ASK, CONSTRUCT
//! - Basic triple patterns
//! - OPTIONAL (LEFT JOIN semantics)
//! - UNION
//! - FILTER: LANG(), STR(), DATATYPE(), BOUND(), regex(), isIRI(), isLiteral(), isBlank()
//!   STRSTARTS(), STRENDS(), CONTAINS(), STRLEN(), logical &&/||/!
//! - FILTER EXISTS / FILTER NOT EXISTS
//! - Property paths: /, |, *, +, ?, ^, !()
//! - JSON-LD output
//! - LIMIT / OFFSET
//! - ORDER BY

pub mod aggregates;
pub mod construct;
pub mod filter;
pub mod jsonld;
pub mod property_path;
pub mod subquery;

use crate::error::{WasmError, WasmResult};
use crate::store::OxiRSStore;
use crate::Triple;
use aggregates::{
    parse_aggregate_expr, parse_group_by, AggregateEvaluator, AggregateProjection, GroupByClause,
};
use filter::{
    extract_lang, extract_literal_value, find_top_level_comma, parse_filter_inner, FilterExpr,
};
use property_path::{parse_property_path, PropertyPath};
use std::collections::HashMap;
use subquery::{join_with_subquery, try_extract_subquery, SubqueryEvaluator};

// Re-export JSON-LD for convenience
pub use jsonld::{serialize_jsonld, serialize_jsonld_with_prefixes};

// -----------------------------------------------------------------------
// Public entry points
// -----------------------------------------------------------------------

/// Execute a SPARQL SELECT query
pub fn execute_select(
    sparql: &str,
    store: &OxiRSStore,
) -> WasmResult<Vec<HashMap<String, String>>> {
    let query = parse_select_query(sparql)?;
    evaluate_select(&query, store)
}

/// Execute a SPARQL ASK query
pub fn execute_ask(sparql: &str, store: &OxiRSStore) -> WasmResult<bool> {
    let query = parse_ask_query(sparql)?;
    evaluate_ask(&query, store)
}

/// Execute a SPARQL CONSTRUCT query
pub fn execute_construct(sparql: &str, store: &OxiRSStore) -> WasmResult<Vec<Triple>> {
    let query = parse_select_query(sparql)?;
    let bindings = evaluate_select(&query, store)?;

    let mut triples = Vec::new();
    for binding in bindings {
        if let (Some(s), Some(p), Some(o)) = (
            binding.get("s").or(binding.get("subject")),
            binding.get("p").or(binding.get("predicate")),
            binding.get("o").or(binding.get("object")),
        ) {
            triples.push(Triple::new(s, p, o));
        }
    }

    Ok(triples)
}

// -----------------------------------------------------------------------
// Core types (pub(crate) so update and named_graph modules can use them)
// -----------------------------------------------------------------------

/// A parsed SELECT/ASK/CONSTRUCT query
struct SelectQuery {
    variables: Vec<String>,
    patterns: Vec<GraphPattern>,
    limit: Option<usize>,
    offset: Option<usize>,
    order_by: Vec<OrderCondition>,
    distinct: bool,
    /// GROUP BY clause (None if no grouping)
    group_by: Option<GroupByClause>,
    /// Aggregate projections from SELECT clause (e.g. COUNT(?o) AS ?count)
    agg_projections: Vec<AggregateProjection>,
}

/// ORDER BY condition
#[derive(Debug, Clone)]
struct OrderCondition {
    variable: String,
    ascending: bool,
}

/// A graph pattern – the unit of evaluation in SPARQL algebra
#[derive(Debug, Clone)]
pub(crate) enum GraphPattern {
    /// Basic triple pattern
    Triple(TriplePattern),
    /// Property path pattern: ?s path ?o
    PropertyPath {
        subject: PatternTerm,
        path: PropertyPath,
        object: PatternTerm,
    },
    /// OPTIONAL { patterns }
    Optional(Vec<GraphPattern>),
    /// { left } UNION { right }
    Union(Vec<GraphPattern>, Vec<GraphPattern>),
    /// FILTER expression
    Filter(FilterExpr),
    /// FILTER EXISTS { inner_patterns }
    FilterExists {
        negated: bool,
        inner: Vec<GraphPattern>,
    },
    /// VALUES inline data
    Values {
        variables: Vec<String>,
        rows: Vec<HashMap<String, String>>,
    },
    /// Subquery: { SELECT ... WHERE { ... } }
    Subquery(String),
}

/// A triple pattern with variable or concrete term positions
#[derive(Debug, Clone)]
pub(crate) struct TriplePattern {
    pub(crate) subject: PatternTerm,
    pub(crate) predicate: PatternTerm,
    pub(crate) object: PatternTerm,
}

/// Either a variable (`?name`) or a concrete RDF value
#[derive(Debug, Clone)]
pub(crate) enum PatternTerm {
    Variable(String),
    Value(String),
}

impl PatternTerm {
    pub(crate) fn matches(&self, value: &str, bindings: &HashMap<String, String>) -> bool {
        match self {
            PatternTerm::Variable(name) => {
                if let Some(bound_value) = bindings.get(name) {
                    bound_value == value
                } else {
                    true
                }
            }
            PatternTerm::Value(v) => v == value,
        }
    }

    /// Bind this term in the given binding, returning the resolved value or None if
    /// a variable that isn't bound yet (meaning "any")
    pub(crate) fn resolve<'a>(&'a self, bindings: &'a HashMap<String, String>) -> Option<&'a str> {
        match self {
            PatternTerm::Variable(name) => bindings.get(name.as_str()).map(|s| s.as_str()),
            PatternTerm::Value(v) => Some(v.as_str()),
        }
    }
}

// -----------------------------------------------------------------------
// Token types for the pattern parser
// -----------------------------------------------------------------------

enum Token {
    Optional(String),
    Union(String, String),
    Filter(String),
    FilterExists { negated: bool, inner: String },
    Values(String),
    Statement(String),
}

// -----------------------------------------------------------------------
// Tokenizer
// -----------------------------------------------------------------------

fn tokenize_top_level(body: &str) -> Vec<Token> {
    let chars: Vec<char> = body.chars().collect();
    let mut tokens: Vec<Token> = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        // Skip whitespace
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }

        // Comment
        if chars[i] == '#' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        // Read a word to check for keywords
        let (word, word_end) = read_word(&chars, i);
        let word_upper = word.to_uppercase();

        // OPTIONAL { ... }
        if word_upper == "OPTIONAL" {
            let brace_start = find_next_char(&chars, word_end, '{');
            if let Some(bs) = brace_start {
                let (block, end) = read_brace_block(&chars, bs);
                tokens.push(Token::Optional(block));
                i = end;
                continue;
            }
        }

        // FILTER NOT EXISTS { ... }
        if word_upper == "FILTER" {
            // Peek ahead for NOT EXISTS or EXISTS or regular FILTER(...)
            let after = skip_whitespace(&chars, word_end);

            // Check "NOT"
            let (next_word, next_end) = read_word(&chars, after);
            if next_word.to_uppercase() == "NOT" {
                let after2 = skip_whitespace(&chars, next_end);
                let (next2_word, next2_end) = read_word(&chars, after2);
                if next2_word.to_uppercase() == "EXISTS" {
                    let brace_start = find_next_char(&chars, next2_end, '{');
                    if let Some(bs) = brace_start {
                        let (block, end) = read_brace_block(&chars, bs);
                        tokens.push(Token::FilterExists {
                            negated: true,
                            inner: block,
                        });
                        i = end;
                        continue;
                    }
                }
            }

            // Check "EXISTS"
            if next_word.to_uppercase() == "EXISTS" {
                let brace_start = find_next_char(&chars, next_end, '{');
                if let Some(bs) = brace_start {
                    let (block, end) = read_brace_block(&chars, bs);
                    tokens.push(Token::FilterExists {
                        negated: false,
                        inner: block,
                    });
                    i = end;
                    continue;
                }
            }

            // Regular FILTER(...)
            if after < chars.len() && chars[after] == '(' {
                let (paren_content, end) = read_paren_group(&chars, after);
                tokens.push(Token::Filter(paren_content));
                i = end;
                continue;
            }
        }

        // VALUES
        if word_upper == "VALUES" {
            let (values_str, end) = read_values_block(&chars, word_end);
            tokens.push(Token::Values(values_str));
            i = end;
            continue;
        }

        // { left } UNION { right }
        if chars[i] == '{' {
            let (left_block, left_end) = read_brace_block(&chars, i);
            let after_left = skip_whitespace(&chars, left_end);
            let (maybe_union, union_end) = read_word(&chars, after_left);
            if maybe_union.to_uppercase() == "UNION" {
                let after_union = skip_whitespace(&chars, union_end);
                if after_union < chars.len() && chars[after_union] == '{' {
                    let (right_block, right_end) = read_brace_block(&chars, after_union);
                    tokens.push(Token::Union(left_block, right_block));
                    i = right_end;
                    continue;
                }
            }
            // Not a UNION — treat as a nested group (inline into statement)
            tokens.push(Token::Statement(left_block));
            i = left_end;
            continue;
        }

        // Regular statement (triple pattern)
        let (stmt, end) = read_statement_from(&chars, i);
        if !stmt.trim().is_empty() {
            tokens.push(Token::Statement(stmt));
        }
        i = end;
    }

    tokens
}

fn find_next_char(chars: &[char], from: usize, target: char) -> Option<usize> {
    let mut i = from;
    while i < chars.len() {
        if chars[i] == target {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn skip_whitespace(chars: &[char], from: usize) -> usize {
    let mut i = from;
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    i
}

fn read_brace_block(chars: &[char], start: usize) -> (String, usize) {
    debug_assert_eq!(chars[start], '{');
    let mut depth = 0usize;
    let mut i = start;
    let mut content = String::new();
    let mut in_string = false;
    let mut in_angle = false;

    while i < chars.len() {
        let c = chars[i];
        if in_string {
            content.push(c);
            if c == '"' {
                in_string = false;
            }
        } else if in_angle {
            content.push(c);
            if c == '>' {
                in_angle = false;
            }
        } else {
            match c {
                '"' => {
                    content.push(c);
                    in_string = true;
                }
                '<' => {
                    content.push(c);
                    in_angle = true;
                }
                '{' => {
                    depth += 1;
                    if depth > 1 {
                        content.push(c);
                    }
                }
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return (content, i + 1);
                    }
                    content.push(c);
                }
                _ => content.push(c),
            }
        }
        i += 1;
    }
    (content, i)
}

fn read_word(chars: &[char], start: usize) -> (String, usize) {
    let mut i = start;
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    let word_start = i;
    while i < chars.len()
        && !chars[i].is_whitespace()
        && chars[i] != '{'
        && chars[i] != '}'
        && chars[i] != '('
        && chars[i] != ')'
        && chars[i] != '.'
    {
        i += 1;
    }
    let word: String = chars[word_start..i].iter().collect();
    (word, i)
}

fn read_paren_group(chars: &[char], start: usize) -> (String, usize) {
    debug_assert_eq!(chars[start], '(');
    let mut depth = 0usize;
    let mut i = start;
    let mut content = String::new();
    let mut in_string = false;

    while i < chars.len() {
        let c = chars[i];
        if in_string {
            content.push(c);
            if c == '"' {
                in_string = false;
            }
        } else {
            match c {
                '"' => {
                    in_string = true;
                    content.push(c);
                }
                '(' => {
                    depth += 1;
                    if depth > 1 {
                        content.push(c);
                    }
                }
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return (content, i + 1);
                    }
                    content.push(c);
                }
                _ => content.push(c),
            }
        }
        i += 1;
    }
    (content, i)
}

fn read_values_block(chars: &[char], start: usize) -> (String, usize) {
    // VALUES ?var { values... } or VALUES (?var1 ?var2) { (val1 val2)... }
    // We return a string in the form: `<var_decls> { <values_body> }`
    // so that parse_values_block can find the `{`.
    let mut content = String::new();

    // Scan forward to find the opening brace
    let brace_start = match find_next_char(chars, start, '{') {
        Some(bs) => bs,
        None => return (content, start),
    };

    // Include variable declarations (between start and brace_start)
    for c in &chars[start..brace_start] {
        content.push(*c);
    }

    // Read the brace block content (strips outer braces)
    let (block, end) = read_brace_block(chars, brace_start);
    // Re-add braces so parse_values_block can find them
    content.push('{');
    content.push_str(&block);
    content.push('}');
    (content, end)
}

/// Read a triple pattern statement (up to the next `.` or end of block)
fn read_statement_from(chars: &[char], start: usize) -> (String, usize) {
    let mut i = start;
    let mut content = String::new();
    let mut in_string = false;
    let mut in_angle = false;
    let mut depth_brace = 0usize;
    let mut depth_paren = 0usize;

    while i < chars.len() {
        let c = chars[i];

        if in_string {
            content.push(c);
            if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if in_angle {
            content.push(c);
            if c == '>' {
                in_angle = false;
            }
            i += 1;
            continue;
        }

        match c {
            '"' => {
                in_string = true;
                content.push(c);
            }
            '<' => {
                in_angle = true;
                content.push(c);
            }
            '{' => {
                depth_brace += 1;
                content.push(c);
            }
            '}' => {
                if depth_brace == 0 {
                    break;
                }
                depth_brace -= 1;
                content.push(c);
            }
            '(' => {
                depth_paren += 1;
                content.push(c);
            }
            ')' => {
                if depth_paren == 0 {
                    break;
                }
                depth_paren -= 1;
                content.push(c);
            }
            '.' if depth_brace == 0 && depth_paren == 0 => {
                // Statement terminator (only at top level)
                i += 1;
                // But peek if next non-ws word is OPTIONAL/UNION/FILTER — then stop here
                break;
            }
            ';' if depth_brace == 0 && depth_paren == 0 => {
                // Predicate-object list shortcut — end current statement segment
                i += 1;
                break;
            }
            _ => {
                // Check for keyword boundaries (OPTIONAL/UNION/FILTER at start of line)
                if depth_brace == 0 && depth_paren == 0 {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        // Peek if current char starts a keyword
                        let (word_ahead, _) = read_word(chars, i);
                        let wu = word_ahead.to_uppercase();
                        if wu == "OPTIONAL" || wu == "UNION" || wu == "FILTER" || wu == "VALUES" {
                            break;
                        }
                    }
                }
                content.push(c);
            }
        }
        i += 1;
    }
    (content, i)
}

// -----------------------------------------------------------------------
// Graph pattern parser
// -----------------------------------------------------------------------

/// Parse graph patterns from the body of a WHERE clause
pub(crate) fn parse_graph_patterns(body: &str) -> WasmResult<Vec<GraphPattern>> {
    let tokens = tokenize_top_level(body);
    let mut patterns: Vec<GraphPattern> = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Optional(inner) => {
                let inner_patterns = parse_graph_patterns(inner)?;
                patterns.push(GraphPattern::Optional(inner_patterns));
                i += 1;
            }

            Token::Union(left_str, right_str) => {
                let left = parse_graph_patterns(left_str)?;
                let right = parse_graph_patterns(right_str)?;
                patterns.push(GraphPattern::Union(left, right));
                i += 1;
            }

            Token::Filter(filter_str) => {
                if let Some(expr) = parse_filter_inner(filter_str) {
                    patterns.push(GraphPattern::Filter(expr));
                }
                i += 1;
            }

            Token::FilterExists { negated, inner } => {
                let inner_patterns = parse_graph_patterns(inner)?;
                patterns.push(GraphPattern::FilterExists {
                    negated: *negated,
                    inner: inner_patterns,
                });
                i += 1;
            }

            Token::Values(values_str) => {
                if let Some(values_pattern) = parse_values_block(values_str) {
                    patterns.push(values_pattern);
                }
                i += 1;
            }

            Token::Statement(stmt) => {
                let stmt = stmt.trim();
                if !stmt.is_empty() {
                    // Check if this is a subquery: { SELECT ... WHERE { ... } }
                    if let Some(sq) = try_extract_subquery(stmt) {
                        patterns.push(GraphPattern::Subquery(sq));
                    }
                    // Check if this is a property path pattern (?s path ?o)
                    else if let Some(pp_pattern) = try_parse_property_path_pattern(stmt) {
                        patterns.push(pp_pattern);
                    } else if let Some(tp) = parse_triple_tokens(stmt) {
                        patterns.push(GraphPattern::Triple(tp));
                    }
                }
                i += 1;
            }
        }
    }

    Ok(patterns)
}

/// Try to parse a triple pattern as a property path: `?s path ?o`
///
/// Returns None if the predicate is a simple IRI or variable (not a path expression)
fn try_parse_property_path_pattern(s: &str) -> Option<GraphPattern> {
    let tokens = tokenize_triple(s);
    if tokens.len() < 3 {
        return None;
    }
    let pred_token = &tokens[1];

    // Check if predicate contains path operators
    if !pred_token.contains('/')
        && !pred_token.contains('|')
        && !pred_token.ends_with('*')
        && !pred_token.ends_with('+')
        && !pred_token.ends_with('?')
        && !pred_token.starts_with('^')
        && !pred_token.starts_with("!(")
        && pred_token != "a"
    {
        return None;
    }

    let path = parse_property_path(pred_token)?;
    let subject = parse_pattern_term(&tokens[0]);
    let obj_raw = tokens[2..].join(" ");
    let object = parse_pattern_term(&obj_raw);

    Some(GraphPattern::PropertyPath {
        subject,
        path,
        object,
    })
}

/// Parse a VALUES block: `?var { "val1" "val2" }`  or  `(?var1 ?var2) { ... }`
fn parse_values_block(s: &str) -> Option<GraphPattern> {
    let s = s.trim();

    // Find opening brace
    let brace_pos = s.find('{')?;
    let vars_str = s[..brace_pos].trim();
    let values_body = s[brace_pos + 1..s.rfind('}').unwrap_or(s.len())].trim();

    // Parse variable names
    let variables: Vec<String> = if vars_str.starts_with('(') && vars_str.ends_with(')') {
        vars_str[1..vars_str.len() - 1]
            .split_whitespace()
            .filter(|t| t.starts_with('?') || t.starts_with('$'))
            .map(|t| t.trim_start_matches(['?', '$']).to_string())
            .collect()
    } else {
        vars_str
            .split_whitespace()
            .filter(|t| t.starts_with('?') || t.starts_with('$'))
            .map(|t| t.trim_start_matches(['?', '$']).to_string())
            .collect()
    };

    if variables.is_empty() {
        return None;
    }

    // Parse value rows
    let mut rows: Vec<HashMap<String, String>> = Vec::new();

    if variables.len() == 1 {
        // Single var: each token in body is a value
        let values: Vec<&str> = values_body.split_whitespace().collect();
        for val in values {
            if val == "UNDEF" {
                rows.push(HashMap::new());
            } else {
                let mut row = HashMap::new();
                // Normalize IRIs: strip angle brackets
                let normalized = if val.starts_with('<') && val.ends_with('>') {
                    val[1..val.len() - 1].to_string()
                } else {
                    val.to_string()
                };
                row.insert(variables[0].clone(), normalized);
                rows.push(row);
            }
        }
    } else {
        // Multiple vars: values are grouped in ( ... ) tuples
        let tuple_str = values_body;
        let mut pos = 0usize;
        let chars: Vec<char> = tuple_str.chars().collect();
        while pos < chars.len() {
            if chars[pos] == '(' {
                // Read tuple
                let mut depth = 0usize;
                let mut tuple_content = String::new();
                let start = pos;
                while pos < chars.len() {
                    match chars[pos] {
                        '(' => {
                            depth += 1;
                            if depth > 1 {
                                tuple_content.push('(');
                            }
                        }
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                pos += 1;
                                break;
                            }
                            tuple_content.push(')');
                        }
                        c => {
                            if depth > 0 {
                                tuple_content.push(c);
                            }
                        }
                    }
                    pos += 1;
                }
                let _ = start;
                let vals: Vec<&str> = tuple_content.split_whitespace().collect();
                let mut row = HashMap::new();
                for (j, var) in variables.iter().enumerate() {
                    if let Some(val) = vals.get(j) {
                        if *val != "UNDEF" {
                            row.insert(var.clone(), val.to_string());
                        }
                    }
                }
                rows.push(row);
            } else {
                pos += 1;
            }
        }
    }

    Some(GraphPattern::Values { variables, rows })
}

/// Parse a triple pattern from a whitespace-delimited string
fn parse_triple_tokens(s: &str) -> Option<TriplePattern> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    let tokens = tokenize_triple(s);
    if tokens.len() < 3 {
        return None;
    }

    let subject = parse_pattern_term(&tokens[0]);
    let predicate = if tokens[1] == "a" {
        PatternTerm::Value("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
    } else {
        parse_pattern_term(&tokens[1])
    };
    // Object may contain language/datatype annotations – take the rest
    let obj_raw = tokens[2..].join(" ");
    let object = parse_pattern_term(&obj_raw);

    Some(TriplePattern {
        subject,
        predicate,
        object,
    })
}

/// Tokenise a single triple string respecting `<...>` and `"..."` quoting.
fn tokenize_triple(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut in_angle = false;
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if in_string {
            current.push(c);
            if c == '"' {
                in_string = false;
            }
        } else if in_angle {
            current.push(c);
            if c == '>' {
                in_angle = false;
            }
        } else {
            match c {
                '"' => {
                    current.push(c);
                    in_string = true;
                }
                '<' => {
                    current.push(c);
                    in_angle = true;
                }
                ' ' | '\t' | '\n' | '\r' => {
                    let tok = current.trim().to_string();
                    if !tok.is_empty() {
                        tokens.push(tok);
                    }
                    current = String::new();
                }
                _ => current.push(c),
            }
        }
        i += 1;
    }
    let tok = current.trim().to_string();
    if !tok.is_empty() {
        tokens.push(tok);
    }
    tokens
}

/// Parse a pattern term (variable or IRI/literal)
pub(crate) fn parse_pattern_term(term: &str) -> PatternTerm {
    let term = term.trim();
    if term.starts_with('?') || term.starts_with('$') {
        PatternTerm::Variable(term.trim_start_matches(['?', '$']).to_string())
    } else if term.starts_with('<') && term.ends_with('>') {
        PatternTerm::Value(term[1..term.len() - 1].to_string())
    } else {
        PatternTerm::Value(term.to_string())
    }
}

// -----------------------------------------------------------------------
// Top-level query parsers
// -----------------------------------------------------------------------

/// Parse a SELECT query string into a [`SelectQuery`]
fn parse_select_query(sparql: &str) -> WasmResult<SelectQuery> {
    let sparql = sparql.trim();
    let upper = sparql.to_uppercase();

    let select_start = upper
        .find("SELECT")
        .ok_or_else(|| WasmError::QueryError("No SELECT clause".to_string()))?;

    let where_start = upper
        .find("WHERE")
        .ok_or_else(|| WasmError::QueryError("No WHERE clause".to_string()))?;

    let select_clause = sparql[select_start + 6..where_start].trim();

    // Detect SELECT DISTINCT — only at the start of the clause, not inside aggregates
    let select_upper = select_clause.to_uppercase();
    // DISTINCT at the very beginning (possibly after whitespace) indicates SELECT DISTINCT
    let distinct = select_upper.trim_start().starts_with("DISTINCT");
    // Strip the leading DISTINCT keyword only (not inside aggregate expressions)
    let select_clause = if distinct {
        let trimmed = select_clause.trim_start();
        let after = trimmed.to_uppercase();
        if after.starts_with("DISTINCT") {
            trimmed[8..].to_string()
        } else {
            select_clause.to_string()
        }
    } else {
        select_clause.to_string()
    };
    let select_clause = select_clause.as_str();

    // Parse aggregate projections from SELECT clause: (FUNC(?x) AS ?alias)
    // Also collect plain variable names
    let (variables, agg_projections) = parse_select_clause(select_clause);

    // Find the opening `{` of the WHERE body
    let where_open = sparql[where_start..]
        .find('{')
        .ok_or_else(|| WasmError::QueryError("No WHERE body '{'".to_string()))?
        + where_start;

    let where_body = extract_braces_at(sparql, where_open)?;

    let limit = parse_modifier(sparql, "LIMIT");
    let offset = parse_modifier(sparql, "OFFSET");

    // Find the end of the WHERE clause to look for ORDER BY / GROUP BY after it
    let after_where = where_open + where_body.len() + 2; // +2 for { and }
    let tail = if after_where < sparql.len() {
        &sparql[after_where..]
    } else {
        ""
    };
    let order_by = parse_order_by(tail);
    let group_by = parse_group_by(tail);

    let patterns = parse_graph_patterns(&where_body)?;

    Ok(SelectQuery {
        variables,
        patterns,
        limit,
        offset,
        order_by,
        distinct,
        group_by,
        agg_projections,
    })
}

/// Parse a SELECT clause into (plain_variables, aggregate_projections).
///
/// Handles:
/// - `?x ?y ?z` → plain variables
/// - `(COUNT(?o) AS ?count) ?x` → one agg projection + one plain variable
/// - `*` → empty (wildcard)
fn parse_select_clause(clause: &str) -> (Vec<String>, Vec<AggregateProjection>) {
    let clause = clause.trim();
    if clause == "*" {
        return (vec![], vec![]);
    }

    let mut variables: Vec<String> = Vec::new();
    let mut agg_projections: Vec<AggregateProjection> = Vec::new();

    // Tokenise: split by whitespace but keep parenthesised groups together
    let chars: Vec<char> = clause.chars().collect();
    let mut pos = 0;

    while pos < chars.len() {
        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }
        if pos >= chars.len() {
            break;
        }

        if chars[pos] == '(' {
            // Read entire paren group
            let mut depth = 0usize;
            let start = pos;
            while pos < chars.len() {
                match chars[pos] {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            pos += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                pos += 1;
            }
            let token: String = chars[start..pos].iter().collect();
            // Try as aggregate projection
            if let Some(proj) = parse_aggregate_expr(&token) {
                agg_projections.push(proj);
            }
        } else {
            // Read plain token (variable or keyword)
            let start = pos;
            while pos < chars.len() && !chars[pos].is_whitespace() && chars[pos] != '(' {
                pos += 1;
            }
            let token: String = chars[start..pos].iter().collect();
            if token.starts_with('?') || token.starts_with('$') {
                variables.push(token.trim_start_matches(['?', '$']).to_string());
            }
        }
    }

    (variables, agg_projections)
}

/// Parse ASK query by rewriting as SELECT
fn parse_ask_query(sparql: &str) -> WasmResult<SelectQuery> {
    let original_upper = sparql.to_uppercase();
    let ask_pos = original_upper
        .find("ASK")
        .ok_or_else(|| WasmError::QueryError("No ASK keyword".to_string()))?;
    let rewritten = format!(
        "{}SELECT * WHERE{}",
        &sparql[..ask_pos],
        &sparql[ask_pos + 3..]
    );
    parse_select_query(&rewritten)
}

/// Parse ORDER BY clause: `ORDER BY ?var` or `ORDER BY DESC(?var)`
fn parse_order_by(after_where: &str) -> Vec<OrderCondition> {
    let upper = after_where.to_uppercase();
    let Some(order_pos) = upper.find("ORDER") else {
        return vec![];
    };
    let rest = &after_where[order_pos + 5..];
    let upper_rest = rest.to_uppercase();
    let Some(by_pos) = upper_rest.find("BY") else {
        return vec![];
    };
    let conditions_str = &rest[by_pos + 2..];

    // Cut off at LIMIT/OFFSET/end
    let cutoff = {
        let u = conditions_str.to_uppercase();
        u.find("LIMIT")
            .or_else(|| u.find("OFFSET"))
            .unwrap_or(conditions_str.len())
    };
    let conditions_str = &conditions_str[..cutoff];

    let mut result = Vec::new();
    for token in conditions_str.split_whitespace() {
        let token_upper = token.to_uppercase();
        if token_upper.starts_with("DESC(") && token_upper.ends_with(')') {
            let var = token[5..token.len() - 1]
                .trim()
                .trim_start_matches(['?', '$'])
                .to_string();
            result.push(OrderCondition {
                variable: var,
                ascending: false,
            });
        } else if token_upper.starts_with("ASC(") && token_upper.ends_with(')') {
            let var = token[4..token.len() - 1]
                .trim()
                .trim_start_matches(['?', '$'])
                .to_string();
            result.push(OrderCondition {
                variable: var,
                ascending: true,
            });
        } else if token.starts_with('?') || token.starts_with('$') {
            let var = token.trim_start_matches(['?', '$']).to_string();
            result.push(OrderCondition {
                variable: var,
                ascending: true,
            });
        }
    }
    result
}

/// Extract the content inside the `{...}` block that starts at `open_pos`
///
/// Handles string literals (`"..."`) and IRI angle brackets (`<http://...>`).
/// Comparison operators like `<` and `>` (not followed by a URI scheme) are
/// treated as plain characters, not angle bracket delimiters.
fn extract_braces_at(s: &str, open_pos: usize) -> WasmResult<String> {
    let chars: Vec<char> = s[open_pos + 1..].chars().collect();
    let mut depth = 1usize;
    let mut pos = 0usize;
    let mut in_string = false;
    let mut in_angle = false;

    while pos < chars.len() && depth > 0 {
        let c = chars[pos];
        if in_string {
            if c == '"' {
                in_string = false;
            }
        } else if in_angle {
            if c == '>' {
                in_angle = false;
            }
        } else {
            match c {
                '"' => in_string = true,
                '<' => {
                    // Only treat as IRI angle bracket if followed by a non-whitespace,
                    // non-digit character (i.e., part of an IRI, not a comparison)
                    let next = chars.get(pos + 1).copied();
                    if let Some(nc) = next {
                        if !nc.is_whitespace() && !nc.is_ascii_digit() && nc != '=' && nc != '>' {
                            in_angle = true;
                        }
                    }
                }
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        if depth > 0 {
            pos += 1;
        }
    }
    if depth != 0 {
        return Err(WasmError::QueryError("Unmatched '{' in query".to_string()));
    }
    Ok(chars[..pos].iter().collect())
}

/// Parse modifier like LIMIT or OFFSET
fn parse_modifier(sparql: &str, modifier: &str) -> Option<usize> {
    let upper = sparql.to_uppercase();
    let idx = upper.find(modifier)?;
    let rest = &sparql[idx + modifier.len()..];
    let num_str: String = rest
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num_str.parse().ok()
}

// -----------------------------------------------------------------------
// Evaluation
// -----------------------------------------------------------------------

type Bindings = Vec<HashMap<String, String>>;

/// Evaluate a SELECT query against the store
fn evaluate_select(query: &SelectQuery, store: &OxiRSStore) -> WasmResult<Bindings> {
    let mut results: Bindings = vec![HashMap::new()];

    for pattern in &query.patterns {
        results = evaluate_pattern(pattern, results, store)?;
    }

    // Apply GROUP BY + aggregates if present
    if let Some(group_by) = &query.group_by {
        results = AggregateEvaluator::apply(&results, group_by, &query.agg_projections)?;
    } else if !query.agg_projections.is_empty() {
        // Aggregate without GROUP BY → single group over all rows.
        // SPARQL 1.1 spec: even with zero input rows, one output row is produced.
        let implicit_group = GroupByClause {
            variables: vec![],
            having: None,
        };
        results = AggregateEvaluator::apply(&results, &implicit_group, &query.agg_projections)?;
        // If results is still empty (no input rows), emit one row with zero-value aggregates
        if results.is_empty() {
            let mut zero_row = HashMap::new();
            for agg in &query.agg_projections {
                let zero_val = match &agg.func {
                    aggregates::AggregateFunc::Count { .. } => "0".to_string(),
                    aggregates::AggregateFunc::Sum { .. } => "0".to_string(),
                    aggregates::AggregateFunc::Avg { .. } => "0".to_string(),
                    aggregates::AggregateFunc::Min { .. } => "".to_string(),
                    aggregates::AggregateFunc::Max { .. } => "".to_string(),
                    aggregates::AggregateFunc::GroupConcat { .. } => "".to_string(),
                    aggregates::AggregateFunc::Sample { .. } => "".to_string(),
                };
                zero_row.insert(agg.alias.clone(), zero_val);
            }
            results = vec![zero_row];
        }
    }

    // ORDER BY (before offset/limit, on full bindings)
    if !query.order_by.is_empty() {
        let order_by = &query.order_by;
        results.sort_by(|a, b| {
            for cond in order_by {
                let av = a.get(&cond.variable).map(|s| s.as_str()).unwrap_or("");
                let bv = b.get(&cond.variable).map(|s| s.as_str()).unwrap_or("");
                let av_num = extract_literal_value(av).parse::<f64>();
                let bv_num = extract_literal_value(bv).parse::<f64>();
                let ord = match (av_num, bv_num) {
                    (Ok(an), Ok(bn)) => an.partial_cmp(&bn).unwrap_or(std::cmp::Ordering::Equal),
                    _ => av.cmp(bv),
                };
                let ord = if cond.ascending { ord } else { ord.reverse() };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });
    }

    // Build the full projection list: plain variables + aggregate aliases
    let proj_vars: Vec<String> = {
        let mut v = query.variables.clone();
        for agg in &query.agg_projections {
            if !v.contains(&agg.alias) {
                v.push(agg.alias.clone());
            }
        }
        v
    };

    // Project variables (before DISTINCT so dedup is on projected output)
    if !proj_vars.is_empty() {
        results = results
            .into_iter()
            .map(|binding| {
                let mut projected = HashMap::new();
                for var in &proj_vars {
                    if let Some(value) = binding.get(var) {
                        projected.insert(var.clone(), value.clone());
                    }
                }
                projected
            })
            .collect();
    }

    // DISTINCT (applied after projection)
    if query.distinct {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        results.retain(|b| {
            // Sort keys for deterministic key generation
            let mut pairs: Vec<_> = b.iter().collect();
            pairs.sort_by_key(|(k, _)| k.as_str());
            let key = format!("{:?}", pairs);
            seen.insert(key)
        });
    }

    // OFFSET
    if let Some(offset) = query.offset {
        if offset >= results.len() {
            results.clear();
        } else {
            results = results.into_iter().skip(offset).collect();
        }
    }

    // LIMIT
    if let Some(limit) = query.limit {
        results.truncate(limit);
    }

    Ok(results)
}

/// Evaluate a single graph pattern
pub(crate) fn evaluate_pattern(
    pattern: &GraphPattern,
    input: Bindings,
    store: &OxiRSStore,
) -> WasmResult<Bindings> {
    match pattern {
        GraphPattern::Triple(tp) => evaluate_triple_pattern(tp, input, store),
        GraphPattern::PropertyPath {
            subject,
            path,
            object,
        } => evaluate_property_path(subject, path, object, input, store),
        GraphPattern::Optional(inner) => evaluate_optional(inner, input, store),
        GraphPattern::Union(left, right) => evaluate_union(left, right, &input, store),
        GraphPattern::Filter(expr) => Ok(input.into_iter().filter(|b| expr.evaluate(b)).collect()),
        GraphPattern::FilterExists { negated, inner } => {
            evaluate_filter_exists(*negated, inner, input, store)
        }
        GraphPattern::Values { variables, rows } => evaluate_values(variables, rows, input),
        GraphPattern::Subquery(subquery_str) => {
            let evaluator = SubqueryEvaluator::new(store);
            let sub_results = evaluator.evaluate(subquery_str)?;
            Ok(join_with_subquery(input, sub_results))
        }
    }
}

/// Evaluate a basic triple pattern (inner join)
fn evaluate_triple_pattern(
    tp: &TriplePattern,
    input: Bindings,
    store: &OxiRSStore,
) -> WasmResult<Bindings> {
    let mut output = Vec::new();

    for binding in &input {
        // Use indexes when possible for performance
        let triples_iter: Box<dyn Iterator<Item = _>> = match (
            tp.subject.resolve(binding),
            tp.predicate.resolve(binding),
            tp.object.resolve(binding),
        ) {
            (Some(s), Some(p), Some(o)) => {
                // All bound: existence check
                Box::new(
                    store
                        .all_triples()
                        .filter(move |t| t.subject == s && t.predicate == p && t.object == o),
                )
            }
            (Some(s), _, _) => Box::new(store.all_triples().filter(move |t| t.subject == s)),
            (_, Some(p), _) => Box::new(store.all_triples().filter(move |t| t.predicate == p)),
            _ => Box::new(store.all_triples()),
        };

        for triple in triples_iter {
            if tp.subject.matches(&triple.subject, binding)
                && tp.predicate.matches(&triple.predicate, binding)
                && tp.object.matches(&triple.object, binding)
            {
                let mut new_binding = binding.clone();
                if let PatternTerm::Variable(name) = &tp.subject {
                    new_binding.insert(name.clone(), triple.subject.clone());
                }
                if let PatternTerm::Variable(name) = &tp.predicate {
                    new_binding.insert(name.clone(), triple.predicate.clone());
                }
                if let PatternTerm::Variable(name) = &tp.object {
                    new_binding.insert(name.clone(), triple.object.clone());
                }
                output.push(new_binding);
            }
        }
    }

    Ok(output)
}

/// Evaluate a property path pattern
fn evaluate_property_path(
    subject: &PatternTerm,
    path: &PropertyPath,
    object: &PatternTerm,
    input: Bindings,
    store: &OxiRSStore,
) -> WasmResult<Bindings> {
    let mut output = Vec::new();

    for binding in &input {
        match (subject.resolve(binding), object.resolve(binding)) {
            (Some(subj_val), Some(obj_val)) => {
                // Both bound: check if object is reachable from subject via path
                let reached = path.evaluate(subj_val, store);
                if reached.iter().any(|r| r == obj_val) {
                    output.push(binding.clone());
                }
            }
            (Some(subj_val), None) => {
                // Subject bound, object unbound: enumerate reachable objects
                let reached = path.evaluate(subj_val, store);
                for obj in reached {
                    let mut new_binding = binding.clone();
                    if let PatternTerm::Variable(name) = object {
                        new_binding.insert(name.clone(), obj);
                    }
                    output.push(new_binding);
                }
            }
            (None, Some(obj_val)) => {
                // Object bound, subject unbound: enumerate subjects that can reach object
                let subjects = path.evaluate_reverse(obj_val, store);
                for subj in subjects {
                    let mut new_binding = binding.clone();
                    if let PatternTerm::Variable(name) = subject {
                        new_binding.insert(name.clone(), subj);
                    }
                    output.push(new_binding);
                }
            }
            (None, None) => {
                // Both unbound: enumerate all subject-object pairs via path
                // For each unique subject in the store, evaluate the path
                let subjects: std::collections::HashSet<String> =
                    store.all_triples().map(|t| t.subject.clone()).collect();
                for subj_val in subjects {
                    let reached = path.evaluate(&subj_val, store);
                    for obj in reached {
                        let mut new_binding = binding.clone();
                        if let PatternTerm::Variable(name) = subject {
                            new_binding.insert(name.clone(), subj_val.clone());
                        }
                        if let PatternTerm::Variable(name) = object {
                            new_binding.insert(name.clone(), obj);
                        }
                        output.push(new_binding);
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Evaluate OPTIONAL (left outer join)
fn evaluate_optional(
    inner_patterns: &[GraphPattern],
    input: Bindings,
    store: &OxiRSStore,
) -> WasmResult<Bindings> {
    let mut output = Vec::new();

    for binding in input {
        let mut inner: Bindings = vec![binding.clone()];
        for p in inner_patterns {
            inner = evaluate_pattern(p, inner, store)?;
        }
        if inner.is_empty() {
            output.push(binding);
        } else {
            output.extend(inner);
        }
    }

    Ok(output)
}

/// Evaluate UNION: each branch is evaluated from a fresh binding (not row-by-row)
fn evaluate_union(
    left: &[GraphPattern],
    right: &[GraphPattern],
    _input: &Bindings,
    store: &OxiRSStore,
) -> WasmResult<Bindings> {
    let fresh: Bindings = vec![HashMap::new()];

    let mut left_results: Bindings = fresh.clone();
    for p in left {
        left_results = evaluate_pattern(p, left_results, store)?;
    }

    let mut right_results: Bindings = fresh;
    for p in right {
        right_results = evaluate_pattern(p, right_results, store)?;
    }

    left_results.extend(right_results);
    Ok(left_results)
}

/// Evaluate FILTER EXISTS / FILTER NOT EXISTS
fn evaluate_filter_exists(
    negated: bool,
    inner: &[GraphPattern],
    input: Bindings,
    store: &OxiRSStore,
) -> WasmResult<Bindings> {
    let mut output = Vec::new();

    for binding in input {
        let mut inner_result: Bindings = vec![binding.clone()];
        for p in inner {
            inner_result = evaluate_pattern(p, inner_result, store)?;
        }
        let has_result = !inner_result.is_empty();
        let passes = if negated { !has_result } else { has_result };
        if passes {
            output.push(binding);
        }
    }

    Ok(output)
}

/// Evaluate VALUES inline data (join with value rows)
fn evaluate_values(
    variables: &[String],
    rows: &[HashMap<String, String>],
    input: Bindings,
) -> WasmResult<Bindings> {
    let mut output = Vec::new();

    for binding in &input {
        for row in rows {
            // Check compatibility
            let mut compatible = true;
            for var in variables {
                if let (Some(bound_val), Some(row_val)) = (binding.get(var), row.get(var)) {
                    if bound_val != row_val {
                        compatible = false;
                        break;
                    }
                }
            }
            if compatible {
                let mut new_binding = binding.clone();
                for (k, v) in row {
                    new_binding.entry(k.clone()).or_insert_with(|| v.clone());
                }
                output.push(new_binding);
            }
        }
    }

    Ok(output)
}

fn evaluate_ask(query: &SelectQuery, store: &OxiRSStore) -> WasmResult<bool> {
    let results = evaluate_select(query, store)?;
    Ok(!results.is_empty())
}
// -----------------------------------------------------------------------
// Tests (extracted to query_tests.rs for file size compliance)
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // All main query tests are in query_tests.rs (included below)
    include!("query_tests.rs");

    // Additional tests for new aggregate/subquery features in mod.rs

    #[test]
    fn test_aggregate_select_clause_parsing() {
        let (vars, agg) = parse_select_clause("(COUNT(*) AS ?cnt) ?s");
        assert_eq!(vars, vec!["s"]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].alias, "cnt");
    }

    #[test]
    fn test_select_distinct_not_stripped_from_aggregate() {
        let (_, agg) = parse_select_clause("(COUNT(DISTINCT ?x) AS ?ux)");
        assert_eq!(agg.len(), 1);
        if let aggregates::AggregateFunc::Count { distinct, .. } = &agg[0].func {
            assert!(*distinct);
        } else {
            panic!("expected Count aggregate");
        }
    }

    #[test]
    fn test_parse_select_clause_wildcard() {
        let (vars, agg) = parse_select_clause("*");
        assert!(vars.is_empty());
        assert!(agg.is_empty());
    }

    #[test]
    fn test_parse_select_clause_plain_vars() {
        let (vars, agg) = parse_select_clause("?s ?p ?o");
        assert_eq!(vars, vec!["s", "p", "o"]);
        assert!(agg.is_empty());
    }
}
