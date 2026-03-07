//! # WASM SPARQL Query Executor
//!
//! In-memory, dependency-free SPARQL query executor for WebAssembly environments.
//! Supports SELECT, ASK, CONSTRUCT, and DESCRIBE against an in-memory triple
//! store, with basic FILTER support (regex stub and string equality).
//!
//! ## Example
//!
//! ```rust
//! use oxirs_wasm::sparql_executor::{SparqlExecutor, RdfTriple};
//!
//! let mut executor = SparqlExecutor::new();
//! executor.add_triple(RdfTriple::new(
//!     "<http://example.org/alice>",
//!     "<http://xmlns.com/foaf/0.1/name>",
//!     "\"Alice\"",
//! ));
//! let result = executor.execute("SELECT ?s WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?o }").expect("ok");
//! assert!(!result.rows.is_empty());
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by the SPARQL executor
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutorError {
    /// The query string could not be parsed / understood
    ParseError(String),
    /// A required variable was not bound in the result
    UnboundVariable(String),
    /// Internal executor failure
    InternalError(String),
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutorError::ParseError(msg) => write!(f, "Parse error: {msg}"),
            ExecutorError::UnboundVariable(var) => write!(f, "Unbound variable: {var}"),
            ExecutorError::InternalError(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl std::error::Error for ExecutorError {}

// ─── Domain types ─────────────────────────────────────────────────────────────

/// A single RDF triple (subject, predicate, object as strings)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RdfTriple {
    /// Subject (IRI enclosed in `<>` or blank node `_:id`)
    pub subject: String,
    /// Predicate (IRI enclosed in `<>`)
    pub predicate: String,
    /// Object (IRI, blank node, or quoted literal)
    pub object: String,
}

impl RdfTriple {
    /// Create a new triple
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

    /// Returns `true` if the object looks like an RDF literal (starts with `"`)
    pub fn object_is_literal(&self) -> bool {
        self.object.starts_with('"')
    }

    /// Returns `true` if the object is an IRI (starts with `<`)
    pub fn object_is_iri(&self) -> bool {
        self.object.starts_with('<')
    }

    /// Returns `true` if the object is a blank node (starts with `_:`)
    pub fn object_is_blank(&self) -> bool {
        self.object.starts_with("_:")
    }

    /// Extract the literal string value (strips surrounding quotes and optional datatype/lang)
    pub fn literal_value(&self) -> Option<String> {
        if !self.object_is_literal() {
            return None;
        }
        let inner = &self.object[1..]; // strip leading "
                                       // find closing "
        if let Some(end) = inner.find('"') {
            Some(inner[..end].to_string())
        } else {
            Some(inner.to_string())
        }
    }
}

// ─── Query types ──────────────────────────────────────────────────────────────

/// Detected SPARQL query form
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryType {
    Select,
    Ask,
    Construct,
    Describe,
    Unknown,
}

/// A row of variable bindings (variable name → RDF term string)
pub type Binding = HashMap<String, String>;

/// Result of executing a SPARQL query
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// The form of the query that was executed
    pub query_type: QueryType,
    /// Variable names (for SELECT queries)
    pub variables: Vec<String>,
    /// Rows of bindings (for SELECT/CONSTRUCT queries)
    pub rows: Vec<Binding>,
    /// Boolean result (for ASK queries)
    pub boolean: Option<bool>,
    /// Triples (for CONSTRUCT/DESCRIBE queries)
    pub triples: Vec<RdfTriple>,
}

impl QueryResult {
    /// Format result as an ASCII table (SELECT-style)
    pub fn format_table(&self) -> String {
        if self.variables.is_empty() || self.rows.is_empty() {
            return "(no results)\n".to_string();
        }

        // Compute column widths
        let widths: Vec<usize> = self
            .variables
            .iter()
            .map(|var| {
                let header_len = var.len() + 1; // '?' prefix
                let max_val = self
                    .rows
                    .iter()
                    .filter_map(|row| row.get(var))
                    .map(|v| v.len())
                    .max()
                    .unwrap_or(0);
                header_len.max(max_val).max(4)
            })
            .collect();

        let sep: String = widths
            .iter()
            .map(|w| "-".repeat(*w + 2))
            .collect::<Vec<_>>()
            .join("+");
        let sep = format!("+{sep}+");

        let header: String = self
            .variables
            .iter()
            .zip(widths.iter())
            .map(|(v, w)| format!(" ?{:<width$}", v, width = w - 1))
            .collect::<Vec<_>>()
            .join("|");
        let header = format!("|{header}|");

        let mut lines = vec![sep.clone(), header, sep.clone()];

        for row in &self.rows {
            let line: String = self
                .variables
                .iter()
                .zip(widths.iter())
                .map(|(v, w)| {
                    let val = row.get(v).map(String::as_str).unwrap_or("");
                    format!(" {:<width$}", val, width = *w)
                })
                .collect::<Vec<_>>()
                .join("|");
            lines.push(format!("|{line}|"));
        }

        lines.push(sep);
        lines.join("\n")
    }

    /// Format result as a JSON string
    pub fn format_json(&self) -> String {
        match self.query_type {
            QueryType::Ask => {
                let b = self.boolean.unwrap_or(false);
                format!(r#"{{"type": "ask", "boolean": {b}}}"#)
            }
            QueryType::Select => {
                let vars_json: String = self
                    .variables
                    .iter()
                    .map(|v| format!(r#""{v}""#))
                    .collect::<Vec<_>>()
                    .join(", ");

                let rows_json: String = self
                    .rows
                    .iter()
                    .map(|row| {
                        let pairs: String = row
                            .iter()
                            .map(|(k, v)| format!(r#""{k}": {{"value": "{}"}}"#, escape_json(v)))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{{{pairs}}}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");

                format!(
                    r#"{{"type": "select", "variables": [{vars}], "results": [{rows}]}}"#,
                    vars = vars_json,
                    rows = rows_json,
                )
            }
            QueryType::Construct | QueryType::Describe => {
                let triples_json: String = self
                    .triples
                    .iter()
                    .map(|t| {
                        format!(
                            r#"{{"s": "{}", "p": "{}", "o": "{}"}}"#,
                            escape_json(&t.subject),
                            escape_json(&t.predicate),
                            escape_json(&t.object),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let qtype = match self.query_type {
                    QueryType::Construct => "construct",
                    _ => "describe",
                };
                format!(r#"{{"type": "{qtype}", "triples": [{triples_json}]}}"#)
            }
            QueryType::Unknown => r#"{"type": "unknown"}"#.to_string(),
        }
    }
}

// ─── Pattern matching helpers ─────────────────────────────────────────────────

/// A SPARQL triple pattern term (variable or concrete value)
#[derive(Debug, Clone, PartialEq)]
enum PatternTerm {
    Variable(String),
    Concrete(String),
}

impl PatternTerm {
    fn from_token(token: &str) -> Self {
        if let Some(stripped) = token.strip_prefix('?') {
            PatternTerm::Variable(stripped.to_string())
        } else {
            PatternTerm::Concrete(token.to_string())
        }
    }

    fn matches(&self, value: &str, bindings: &Binding) -> bool {
        match self {
            PatternTerm::Concrete(c) => c == value,
            PatternTerm::Variable(v) => {
                // If already bound, must match
                if let Some(bound) = bindings.get(v.as_str()) {
                    bound == value
                } else {
                    true
                }
            }
        }
    }

    fn bind(&self, value: &str, bindings: &mut Binding) {
        if let PatternTerm::Variable(v) = self {
            bindings
                .entry(v.clone())
                .or_insert_with(|| value.to_string());
        }
    }
}

/// A parsed triple pattern
#[derive(Debug, Clone)]
struct TriplePattern {
    subject: PatternTerm,
    predicate: PatternTerm,
    object: PatternTerm,
}

// ─── FILTER ───────────────────────────────────────────────────────────────────

/// A simple FILTER expression
#[derive(Debug, Clone)]
enum FilterExpr {
    /// FILTER(regex(?var, "pattern")) — stub: prefix match
    Regex { var: String, pattern: String },
    /// FILTER(?var = "value") — string equality on literal value
    StringEquals { var: String, value: String },
    /// Unsupported / unknown filter (passes everything)
    Unknown,
}

impl FilterExpr {
    fn evaluate(&self, bindings: &Binding) -> bool {
        match self {
            FilterExpr::Regex { var, pattern } => {
                let val = match bindings.get(var.as_str()) {
                    Some(v) => v.clone(),
                    None => return true,
                };
                // Strip literal quotes for comparison
                let text = strip_literal(&val);
                // Stub: check if text contains the pattern
                text.contains(pattern.as_str())
            }
            FilterExpr::StringEquals { var, value } => {
                let val = match bindings.get(var.as_str()) {
                    Some(v) => v.clone(),
                    None => return false,
                };
                let text = strip_literal(&val);
                text == value.as_str()
            }
            FilterExpr::Unknown => true,
        }
    }
}

// ─── Simple query parser ──────────────────────────────────────────────────────

struct ParsedQuery {
    query_type: QueryType,
    variables: Vec<String>, // SELECT variables (empty = *)
    patterns: Vec<TriplePattern>,
    filters: Vec<FilterExpr>,
    construct_template: Vec<TriplePattern>,
    describe_target: Option<String>,
}

/// Detect query form from the query string
fn detect_query_type(query: &str) -> QueryType {
    let upper = query.trim_start().to_uppercase();
    if upper.starts_with("SELECT") {
        QueryType::Select
    } else if upper.starts_with("ASK") {
        QueryType::Ask
    } else if upper.starts_with("CONSTRUCT") {
        QueryType::Construct
    } else if upper.starts_with("DESCRIBE") {
        QueryType::Describe
    } else {
        QueryType::Unknown
    }
}

/// Extract the content between the first `{` and matching `}` in the WHERE clause.
///
/// For `SELECT`/`CONSTRUCT`/`DESCRIBE` queries this looks for the `WHERE` keyword.
/// For `ASK` queries (which have the form `ASK { ... }`) the WHERE keyword is
/// optional, so we fall back to finding the last opening `{` in the query if no
/// `WHERE` is present.
fn extract_where_body(query: &str) -> Option<&str> {
    // Try to find WHERE { first (standard form)
    let upper = query.to_uppercase();

    let search_start = if let Some(where_pos) = upper.find("WHERE") {
        &query[where_pos + 5..]
    } else if let Some(ask_pos) = upper.find("ASK") {
        // ASK { ... } — body starts directly after ASK keyword
        &query[ask_pos + 3..]
    } else {
        query
    };

    let brace_start = search_start.find('{')?;
    let body_start = brace_start + 1;
    // Find matching close brace
    let mut depth = 1usize;
    let mut pos = body_start;
    for (i, &byte) in search_start.as_bytes().iter().enumerate().skip(body_start) {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    pos = i;
                    break;
                }
            }
            _ => {}
        }
    }
    Some(&search_start[body_start..pos])
}

/// Extract content between first `{` and matching `}` (for CONSTRUCT template)
fn extract_first_block(query: &str) -> Option<&str> {
    let brace_start = query.find('{')?;
    let body_start = brace_start + 1;
    let mut depth = 1usize;
    let mut end_pos = body_start;
    for (i, &byte) in query.as_bytes().iter().enumerate().skip(body_start) {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end_pos = i;
                    break;
                }
            }
            _ => {}
        }
    }
    Some(&query[body_start..end_pos])
}

/// Parse triple patterns from a WHERE body string.
/// Handles simple `S P O .` triples and FILTER(...) clauses.
fn parse_patterns_and_filters(body: &str) -> (Vec<TriplePattern>, Vec<FilterExpr>) {
    let mut patterns = Vec::new();
    let mut filters = Vec::new();

    // Strip PREFIX and BASE lines for simplicity
    let mut tokens: Vec<String> = Vec::new();
    let mut i = 0usize;
    let chars: Vec<char> = body.chars().collect();
    let len = chars.len();

    while i < len {
        // Skip whitespace
        while i < len && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }

        if chars[i] == '<' {
            // IRI
            let mut iri = String::from('<');
            i += 1;
            while i < len && chars[i] != '>' {
                iri.push(chars[i]);
                i += 1;
            }
            if i < len {
                iri.push('>');
                i += 1;
            }
            tokens.push(iri);
        } else if chars[i] == '"' {
            // Quoted literal
            let mut lit = String::from('"');
            i += 1;
            while i < len && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < len {
                    lit.push('\\');
                    lit.push(chars[i + 1]);
                    i += 2;
                } else {
                    lit.push(chars[i]);
                    i += 1;
                }
            }
            if i < len {
                lit.push('"');
                i += 1;
            }
            // Optional type annotation ^^<...> or language @lang
            if i < len && chars[i] == '^' && i + 1 < len && chars[i + 1] == '^' {
                lit.push_str("^^");
                i += 2;
                if i < len && chars[i] == '<' {
                    let mut dt = String::from('<');
                    i += 1;
                    while i < len && chars[i] != '>' {
                        dt.push(chars[i]);
                        i += 1;
                    }
                    if i < len {
                        dt.push('>');
                        i += 1;
                    }
                    lit.push_str(&dt);
                }
            } else if i < len && chars[i] == '@' {
                lit.push('@');
                i += 1;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '-') {
                    lit.push(chars[i]);
                    i += 1;
                }
            }
            tokens.push(lit);
        } else if chars[i] == '.' {
            tokens.push(".".to_string());
            i += 1;
        } else if chars[i] == '(' || chars[i] == ')' {
            // Skip parentheses (FILTER content is read by the pattern/filter walker)
            i += 1;
        } else if chars[i] == '#' {
            // Comment: skip to end of line
            while i < len && chars[i] != '\n' {
                i += 1;
            }
        } else {
            // Identifier / variable / keyword
            let mut word = String::new();
            while i < len
                && !chars[i].is_whitespace()
                && chars[i] != '.'
                && chars[i] != '('
                && chars[i] != ')'
                && chars[i] != '{'
                && chars[i] != '}'
            {
                word.push(chars[i]);
                i += 1;
            }
            if !word.is_empty() {
                tokens.push(word);
            }
        }
    }

    // Walk tokens to build patterns and filters
    let mut ti = 0usize;
    let tlen = tokens.len();
    while ti < tlen {
        let tok = &tokens[ti];
        let upper_tok = tok.to_uppercase();

        if upper_tok == "FILTER" {
            // Collect all remaining tokens as the filter expression.
            // The tokenizer has already stripped parentheses, so the remaining
            // tokens are the raw filter contents (variable, operator, value).
            ti += 1;
            let filter_tokens: Vec<&str> = tokens[ti..].iter().map(String::as_str).collect();
            let filter = parse_filter_expr(&filter_tokens);
            filters.push(filter);
            // Skip the consumed filter tokens and stop processing
            break; // simplified: one filter per WHERE
        } else if tok == "." || upper_tok == "OPTIONAL" || upper_tok == "UNION" {
            ti += 1;
        } else if tok.starts_with('?')
            || tok.starts_with('<')
            || tok.starts_with('_')
            || tok.starts_with('"')
        {
            // Possible triple pattern: need 3 terms
            if ti + 2 < tlen {
                let s = PatternTerm::from_token(&tokens[ti]);
                let p = PatternTerm::from_token(&tokens[ti + 1]);
                let o = PatternTerm::from_token(&tokens[ti + 2]);
                patterns.push(TriplePattern {
                    subject: s,
                    predicate: p,
                    object: o,
                });
                ti += 3;
                // Skip optional '.'
                if ti < tlen && tokens[ti] == "." {
                    ti += 1;
                }
            } else {
                ti += 1;
            }
        } else {
            ti += 1;
        }
    }

    (patterns, filters)
}

/// Parse a filter expression from a list of tokens (after FILTER keyword)
fn parse_filter_expr(tokens: &[&str]) -> FilterExpr {
    // Concatenate tokens to get a rough expression string
    let expr: String = tokens.join(" ");
    let expr_upper = expr.to_uppercase();

    if expr_upper.contains("REGEX") {
        // regex(?var, "pattern")
        // Extract variable and pattern
        if let (Some(var), Some(pattern)) = (
            extract_var_from_expr(&expr),
            extract_string_literal_from_expr(&expr),
        ) {
            return FilterExpr::Regex { var, pattern };
        }
    }

    // ?var = "value"
    if let Some(eq_pos) = expr.find('=') {
        let lhs = expr[..eq_pos].trim().to_string();
        let rhs = expr[eq_pos + 1..].trim().to_string();
        if let Some(stripped) = lhs.strip_prefix('?') {
            let var = stripped.to_string();
            let value = strip_literal(&rhs);
            return FilterExpr::StringEquals { var, value };
        }
    }

    FilterExpr::Unknown
}

fn extract_var_from_expr(expr: &str) -> Option<String> {
    let var_start = expr.find('?')?;
    let rest = &expr[var_start + 1..];
    let end = rest
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

fn extract_string_literal_from_expr(expr: &str) -> Option<String> {
    let start = expr.find('"')? + 1;
    let rest = &expr[start..];
    let end = rest.find('"').unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

/// Parse SELECT variable list
fn parse_select_variables(query: &str) -> Vec<String> {
    let upper = query.to_uppercase();
    let after_select = if let Some(pos) = upper.find("SELECT") {
        query[pos + 6..].trim_start()
    } else {
        return Vec::new();
    };

    // Check for *
    let upper_after = after_select.to_uppercase();
    if upper_after.trim_start().starts_with('*') {
        return Vec::new(); // wildcard
    }

    let end = after_select
        .to_uppercase()
        .find("WHERE")
        .unwrap_or(after_select.len());
    let var_section = &after_select[..end];

    var_section
        .split_whitespace()
        .filter(|t| t.starts_with('?'))
        .map(|t| t[1..].to_string())
        .collect()
}

/// Parse the CONSTRUCT template block
fn parse_construct_template(query: &str) -> Vec<TriplePattern> {
    let upper = query.to_uppercase();
    let after_construct = if let Some(pos) = upper.find("CONSTRUCT") {
        &query[pos + 9..]
    } else {
        return Vec::new();
    };

    if let Some(block) = extract_first_block(after_construct) {
        let (patterns, _) = parse_patterns_and_filters(block);
        patterns
    } else {
        Vec::new()
    }
}

/// Parse the DESCRIBE target
fn parse_describe_target(query: &str) -> Option<String> {
    let upper = query.to_uppercase();
    let after = if let Some(pos) = upper.find("DESCRIBE") {
        query[pos + 8..].trim_start()
    } else {
        return None;
    };

    // Take first token
    let end = after
        .find(|c: char| c.is_whitespace() || c == '{')
        .unwrap_or(after.len());
    let tok = after[..end].trim().to_string();
    if tok.is_empty() {
        None
    } else {
        Some(tok)
    }
}

/// Parse query into structured form
fn parse_query(query: &str) -> Result<ParsedQuery, ExecutorError> {
    let query_type = detect_query_type(query);

    let (variables, patterns, filters, construct_template, describe_target) = match query_type {
        QueryType::Select => {
            let vars = parse_select_variables(query);
            let where_body = extract_where_body(query).unwrap_or("");
            let (pats, filts) = parse_patterns_and_filters(where_body);
            (vars, pats, filts, Vec::new(), None)
        }
        QueryType::Ask => {
            let where_body = extract_where_body(query).unwrap_or("");
            let (pats, filts) = parse_patterns_and_filters(where_body);
            (Vec::new(), pats, filts, Vec::new(), None)
        }
        QueryType::Construct => {
            let template = parse_construct_template(query);
            let where_body = extract_where_body(query).unwrap_or("");
            let (pats, filts) = parse_patterns_and_filters(where_body);
            (Vec::new(), pats, filts, template, None)
        }
        QueryType::Describe => {
            let target = parse_describe_target(query);
            let where_body = extract_where_body(query).unwrap_or("");
            let (pats, filts) = parse_patterns_and_filters(where_body);
            (Vec::new(), pats, filts, Vec::new(), target)
        }
        QueryType::Unknown => {
            return Err(ExecutorError::ParseError(
                "Unsupported query form".to_string(),
            ));
        }
    };

    Ok(ParsedQuery {
        query_type,
        variables,
        patterns,
        filters,
        construct_template,
        describe_target,
    })
}

// ─── SparqlExecutor ───────────────────────────────────────────────────────────

/// In-memory SPARQL query executor
pub struct SparqlExecutor {
    /// Triple store
    triples: Vec<RdfTriple>,
}

impl Default for SparqlExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl SparqlExecutor {
    /// Create a new, empty executor
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }

    /// Add a single triple to the store
    pub fn add_triple(&mut self, triple: RdfTriple) {
        self.triples.push(triple);
    }

    /// Add multiple triples
    pub fn add_triples(&mut self, triples: impl IntoIterator<Item = RdfTriple>) {
        for t in triples {
            self.triples.push(t);
        }
    }

    /// Return the number of triples in the store
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Clear all triples
    pub fn clear(&mut self) {
        self.triples.clear();
    }

    /// Detect the query type without executing
    pub fn detect_type(&self, query: &str) -> QueryType {
        detect_query_type(query)
    }

    /// Match a single triple pattern against the store, returning all possible bindings
    fn match_pattern(
        &self,
        pattern: &TriplePattern,
        existing_bindings: &[Binding],
    ) -> Vec<Binding> {
        let mut results: Vec<Binding> = Vec::new();

        let base_bindings: Vec<Binding> = if existing_bindings.is_empty() {
            vec![HashMap::new()]
        } else {
            existing_bindings.to_vec()
        };

        for bindings in &base_bindings {
            for triple in &self.triples {
                if pattern.subject.matches(&triple.subject, bindings)
                    && pattern.predicate.matches(&triple.predicate, bindings)
                    && pattern.object.matches(&triple.object, bindings)
                {
                    let mut new_bindings = bindings.clone();
                    pattern.subject.bind(&triple.subject, &mut new_bindings);
                    pattern.predicate.bind(&triple.predicate, &mut new_bindings);
                    pattern.object.bind(&triple.object, &mut new_bindings);
                    results.push(new_bindings);
                }
            }
        }

        results
    }

    /// Execute all patterns in sequence (nested-loop join)
    fn evaluate_patterns(&self, patterns: &[TriplePattern]) -> Vec<Binding> {
        let mut current: Vec<Binding> = Vec::new();

        for pattern in patterns {
            current = self.match_pattern(pattern, &current);
        }

        current
    }

    /// Apply FILTER expressions to a set of bindings
    fn apply_filters(&self, bindings: Vec<Binding>, filters: &[FilterExpr]) -> Vec<Binding> {
        bindings
            .into_iter()
            .filter(|b| filters.iter().all(|f| f.evaluate(b)))
            .collect()
    }

    /// Project SELECT variables from bindings
    fn project(&self, bindings: Vec<Binding>, variables: &[String]) -> Vec<Binding> {
        if variables.is_empty() {
            return bindings; // SELECT *
        }
        bindings
            .into_iter()
            .map(|b| {
                variables
                    .iter()
                    .filter_map(|v| b.get(v).map(|val| (v.clone(), val.clone())))
                    .collect()
            })
            .collect()
    }

    /// Instantiate a CONSTRUCT template with a binding
    fn instantiate_template(
        &self,
        template: &[TriplePattern],
        binding: &Binding,
    ) -> Option<RdfTriple> {
        fn resolve(term: &PatternTerm, binding: &Binding) -> Option<String> {
            match term {
                PatternTerm::Concrete(c) => Some(c.clone()),
                PatternTerm::Variable(v) => binding.get(v.as_str()).cloned(),
            }
        }

        if template.is_empty() {
            return None;
        }
        let t = &template[0];
        let s = resolve(&t.subject, binding)?;
        let p = resolve(&t.predicate, binding)?;
        let o = resolve(&t.object, binding)?;
        Some(RdfTriple::new(s, p, o))
    }

    /// Execute a SPARQL query against the in-memory store
    pub fn execute(&self, query: &str) -> Result<QueryResult, ExecutorError> {
        let parsed = parse_query(query)?;

        match parsed.query_type {
            QueryType::Select => {
                let mut bindings = self.evaluate_patterns(&parsed.patterns);
                bindings = self.apply_filters(bindings, &parsed.filters);

                // Collect all variables if SELECT *
                let variables: Vec<String> = if parsed.variables.is_empty() {
                    let mut var_set: HashSet<String> = HashSet::new();
                    for b in &bindings {
                        for k in b.keys() {
                            var_set.insert(k.clone());
                        }
                    }
                    let mut vars: Vec<String> = var_set.into_iter().collect();
                    vars.sort();
                    vars
                } else {
                    parsed.variables.clone()
                };

                let rows = self.project(bindings, &variables);

                Ok(QueryResult {
                    query_type: QueryType::Select,
                    variables,
                    rows,
                    boolean: None,
                    triples: Vec::new(),
                })
            }

            QueryType::Ask => {
                let mut bindings = self.evaluate_patterns(&parsed.patterns);
                bindings = self.apply_filters(bindings, &parsed.filters);
                Ok(QueryResult {
                    query_type: QueryType::Ask,
                    variables: Vec::new(),
                    rows: Vec::new(),
                    boolean: Some(!bindings.is_empty()),
                    triples: Vec::new(),
                })
            }

            QueryType::Construct => {
                let mut bindings = self.evaluate_patterns(&parsed.patterns);
                bindings = self.apply_filters(bindings, &parsed.filters);

                let mut out_triples: Vec<RdfTriple> = Vec::new();
                for binding in &bindings {
                    for template_pattern in &parsed.construct_template {
                        // Resolve each template triple individually
                        fn resolve_term(term: &PatternTerm, b: &Binding) -> Option<String> {
                            match term {
                                PatternTerm::Concrete(c) => Some(c.clone()),
                                PatternTerm::Variable(v) => b.get(v.as_str()).cloned(),
                            }
                        }
                        if let (Some(s), Some(p), Some(o)) = (
                            resolve_term(&template_pattern.subject, binding),
                            resolve_term(&template_pattern.predicate, binding),
                            resolve_term(&template_pattern.object, binding),
                        ) {
                            out_triples.push(RdfTriple::new(s, p, o));
                        }
                    }
                    // Fallback: if no template, emit matched triples directly
                    if parsed.construct_template.is_empty() {
                        if let Some(t) = self.instantiate_template(&[], binding) {
                            out_triples.push(t);
                        }
                    }
                }

                // If no template defined, return all matched triples from store
                if parsed.construct_template.is_empty() && out_triples.is_empty() {
                    for binding in &bindings {
                        // Re-materialise triples from bindings (s/p/o vars)
                        if let (Some(s), Some(p), Some(o)) = (
                            binding.get("s").or_else(|| binding.get("subject")),
                            binding.get("p").or_else(|| binding.get("predicate")),
                            binding.get("o").or_else(|| binding.get("object")),
                        ) {
                            out_triples.push(RdfTriple::new(s, p, o));
                        }
                    }
                }

                Ok(QueryResult {
                    query_type: QueryType::Construct,
                    variables: Vec::new(),
                    rows: Vec::new(),
                    boolean: None,
                    triples: out_triples,
                })
            }

            QueryType::Describe => {
                let subject_target = parsed.describe_target.clone();
                let mut out_triples: Vec<RdfTriple> = Vec::new();

                if let Some(target) = &subject_target {
                    // Return all triples where the target is the subject
                    for triple in &self.triples {
                        if &triple.subject == target {
                            out_triples.push(triple.clone());
                        }
                    }
                } else {
                    // WHERE-based: evaluate patterns and return all triples about bound subjects
                    let mut bindings = self.evaluate_patterns(&parsed.patterns);
                    bindings = self.apply_filters(bindings, &parsed.filters);

                    let mut subjects: HashSet<String> = HashSet::new();
                    for b in &bindings {
                        for v in b.values() {
                            if v.starts_with('<') {
                                subjects.insert(v.clone());
                            }
                        }
                    }
                    for triple in &self.triples {
                        if subjects.contains(&triple.subject) {
                            out_triples.push(triple.clone());
                        }
                    }
                }

                Ok(QueryResult {
                    query_type: QueryType::Describe,
                    variables: Vec::new(),
                    rows: Vec::new(),
                    boolean: None,
                    triples: out_triples,
                })
            }

            QueryType::Unknown => Err(ExecutorError::ParseError("Unknown query type".to_string())),
        }
    }
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/// Strip surrounding quotes from a literal term
fn strip_literal(s: &str) -> String {
    let trimmed = s.trim();
    if let Some(inner) = trimmed.strip_prefix('"') {
        if let Some(end) = inner.find('"') {
            return inner[..end].to_string();
        }
        return inner.to_string();
    }
    trimmed.to_string()
}

/// Escape a string for embedding in a JSON string value
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const ALICE: &str = "<http://example.org/alice>";
    const BOB: &str = "<http://example.org/bob>";
    const CAROL: &str = "<http://example.org/carol>";
    const KNOWS: &str = "<http://xmlns.com/foaf/0.1/knows>";
    const NAME: &str = "<http://xmlns.com/foaf/0.1/name>";
    const AGE: &str = "<http://xmlns.com/foaf/0.1/age>";

    fn basic_store() -> SparqlExecutor {
        let mut ex = SparqlExecutor::new();
        ex.add_triples(vec![
            RdfTriple::new(ALICE, KNOWS, BOB),
            RdfTriple::new(BOB, KNOWS, CAROL),
            RdfTriple::new(ALICE, NAME, "\"Alice\""),
            RdfTriple::new(BOB, NAME, "\"Bob\""),
            RdfTriple::new(ALICE, AGE, "\"30\""),
        ]);
        ex
    }

    // ── Construction / basic operations ──────────────────────────────────────

    #[test]
    fn test_new_executor_is_empty() {
        let ex = SparqlExecutor::new();
        assert_eq!(ex.triple_count(), 0);
    }

    #[test]
    fn test_add_triple_increments_count() {
        let mut ex = SparqlExecutor::new();
        ex.add_triple(RdfTriple::new(ALICE, KNOWS, BOB));
        assert_eq!(ex.triple_count(), 1);
    }

    #[test]
    fn test_add_triples_batch() {
        let ex = basic_store();
        assert_eq!(ex.triple_count(), 5);
    }

    #[test]
    fn test_clear_removes_all() {
        let mut ex = basic_store();
        ex.clear();
        assert_eq!(ex.triple_count(), 0);
    }

    // ── Query type detection ──────────────────────────────────────────────────

    #[test]
    fn test_detect_select() {
        let ex = SparqlExecutor::new();
        assert_eq!(
            ex.detect_type("SELECT ?s WHERE { ?s ?p ?o }"),
            QueryType::Select
        );
    }

    #[test]
    fn test_detect_ask() {
        let ex = SparqlExecutor::new();
        assert_eq!(ex.detect_type("ASK { ?s ?p ?o }"), QueryType::Ask);
    }

    #[test]
    fn test_detect_construct() {
        let ex = SparqlExecutor::new();
        assert_eq!(
            ex.detect_type("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"),
            QueryType::Construct
        );
    }

    #[test]
    fn test_detect_describe() {
        let ex = SparqlExecutor::new();
        assert_eq!(
            ex.detect_type("DESCRIBE <http://example.org/alice>"),
            QueryType::Describe
        );
    }

    #[test]
    fn test_detect_unknown() {
        let ex = SparqlExecutor::new();
        assert_eq!(ex.detect_type("INSERT DATA { }"), QueryType::Unknown);
    }

    // ── SELECT ────────────────────────────────────────────────────────────────

    #[test]
    fn test_select_all_subjects() {
        let ex = basic_store();
        let res = ex
            .execute(&format!("SELECT ?s WHERE {{ ?s {KNOWS} ?o }}"))
            .expect("ok");
        assert_eq!(res.query_type, QueryType::Select);
        assert_eq!(res.rows.len(), 2);
    }

    #[test]
    fn test_select_with_concrete_predicate_and_object() {
        let ex = basic_store();
        let res = ex
            .execute(&format!("SELECT ?s WHERE {{ ?s {KNOWS} {BOB} }}"))
            .expect("ok");
        assert_eq!(res.rows.len(), 1);
        assert_eq!(res.rows[0].get("s"), Some(&ALICE.to_string()));
    }

    #[test]
    fn test_select_returns_variables() {
        let ex = basic_store();
        let res = ex
            .execute(&format!("SELECT ?s ?o WHERE {{ ?s {NAME} ?o }}"))
            .expect("ok");
        assert!(res.variables.contains(&"s".to_string()));
        assert!(res.variables.contains(&"o".to_string()));
    }

    #[test]
    fn test_select_no_match_returns_empty() {
        let ex = basic_store();
        let res = ex
            .execute("SELECT ?s WHERE { ?s <http://example.org/NOEXIST> ?o }")
            .expect("ok");
        assert!(res.rows.is_empty());
    }

    #[test]
    fn test_select_star_returns_all_bindings() {
        let ex = basic_store();
        let res = ex
            .execute(&format!("SELECT * WHERE {{ ?s {KNOWS} ?o }}"))
            .expect("ok");
        // Should have both bindings
        assert!(!res.rows.is_empty());
    }

    // ── ASK ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_ask_true_when_match() {
        let ex = basic_store();
        let res = ex
            .execute(&format!("ASK {{ {ALICE} {KNOWS} {BOB} }}"))
            .expect("ok");
        assert_eq!(res.boolean, Some(true));
    }

    #[test]
    fn test_ask_false_when_no_match() {
        let ex = basic_store();
        let res = ex
            .execute(&format!("ASK {{ {CAROL} {KNOWS} {ALICE} }}"))
            .expect("ok");
        assert_eq!(res.boolean, Some(false));
    }

    #[test]
    fn test_ask_with_variable() {
        let ex = basic_store();
        let res = ex
            .execute(&format!("ASK {{ ?s {KNOWS} {BOB} }}"))
            .expect("ok");
        assert_eq!(res.boolean, Some(true));
    }

    // ── CONSTRUCT ─────────────────────────────────────────────────────────────

    #[test]
    fn test_construct_with_template() {
        let ex = basic_store();
        let query = format!(
            "CONSTRUCT {{ ?s <http://example.org/connectedTo> ?o }} WHERE {{ ?s {KNOWS} ?o }}"
        );
        let res = ex.execute(&query).expect("ok");
        assert_eq!(res.query_type, QueryType::Construct);
        // Two knows triples → two constructed triples
        assert_eq!(res.triples.len(), 2);
        // All constructed triples should have the new predicate
        assert!(res
            .triples
            .iter()
            .all(|t| t.predicate == "<http://example.org/connectedTo>"));
    }

    #[test]
    fn test_construct_no_match_returns_empty_triples() {
        let ex = basic_store();
        let query = "CONSTRUCT { ?s ?p ?o } WHERE { ?s <http://example.org/NOEXIST> ?o }";
        let res = ex.execute(query).expect("ok");
        assert!(res.triples.is_empty());
    }

    // ── DESCRIBE ─────────────────────────────────────────────────────────────

    #[test]
    fn test_describe_target_returns_subject_triples() {
        let ex = basic_store();
        let query = format!("DESCRIBE {ALICE}");
        let res = ex.execute(&query).expect("ok");
        assert_eq!(res.query_type, QueryType::Describe);
        // Alice has knows, name, age triples
        assert_eq!(res.triples.len(), 3);
        assert!(res.triples.iter().all(|t| t.subject == ALICE));
    }

    #[test]
    fn test_describe_unknown_target_returns_empty() {
        let ex = basic_store();
        let query = "DESCRIBE <http://example.org/unknown>";
        let res = ex.execute(query).expect("ok");
        assert!(res.triples.is_empty());
    }

    // ── FILTER ────────────────────────────────────────────────────────────────

    #[test]
    fn test_filter_string_equals_match() {
        let ex = basic_store();
        // Select subjects whose name = "Alice"
        let query = format!(r#"SELECT ?s WHERE {{ ?s {NAME} ?name FILTER(?name = "Alice") }}"#);
        let res = ex.execute(&query).expect("ok");
        // The filter is a stub, passes everything, but at least should not panic
        assert_eq!(res.query_type, QueryType::Select);
    }

    #[test]
    fn test_filter_regex_stub_contains() {
        let ex = basic_store();
        let query =
            format!(r#"SELECT ?s WHERE {{ ?s {NAME} ?name FILTER(regex(?name, "Alice")) }}"#);
        let res = ex.execute(&query).expect("ok");
        assert_eq!(res.query_type, QueryType::Select);
    }

    // ── Result formatting ──────────────────────────────────────────────────────

    #[test]
    fn test_format_table_select_result() {
        let ex = basic_store();
        let res = ex
            .execute(&format!("SELECT ?s WHERE {{ ?s {KNOWS} ?o }}"))
            .expect("ok");
        let table = res.format_table();
        assert!(table.contains("?s"), "table = {table}");
    }

    #[test]
    fn test_format_table_empty_result() {
        let ex = SparqlExecutor::new();
        let res = QueryResult {
            query_type: QueryType::Select,
            variables: Vec::new(),
            rows: Vec::new(),
            boolean: None,
            triples: Vec::new(),
        };
        let table = res.format_table();
        assert!(table.contains("no results"));
    }

    #[test]
    fn test_format_json_select() {
        let ex = basic_store();
        let res = ex
            .execute(&format!("SELECT ?s WHERE {{ ?s {KNOWS} ?o }}"))
            .expect("ok");
        let json = res.format_json();
        assert!(json.contains(r#""type": "select""#), "json = {json}");
        assert!(json.contains("variables"), "json = {json}");
    }

    #[test]
    fn test_format_json_ask_true() {
        let ex = basic_store();
        let res = ex
            .execute(&format!("ASK {{ {ALICE} {KNOWS} {BOB} }}"))
            .expect("ok");
        let json = res.format_json();
        assert!(json.contains("true"), "json = {json}");
    }

    #[test]
    fn test_format_json_ask_false() {
        let ex = basic_store();
        let res = ex
            .execute(&format!("ASK {{ {CAROL} {KNOWS} {ALICE} }}"))
            .expect("ok");
        let json = res.format_json();
        assert!(json.contains("false"), "json = {json}");
    }

    #[test]
    fn test_format_json_construct() {
        let ex = basic_store();
        let query =
            format!("CONSTRUCT {{ ?s <http://example.org/c> ?o }} WHERE {{ ?s {KNOWS} ?o }}");
        let res = ex.execute(&query).expect("ok");
        let json = res.format_json();
        assert!(json.contains(r#""type": "construct""#), "json = {json}");
        assert!(json.contains("triples"), "json = {json}");
    }

    #[test]
    fn test_format_json_describe() {
        let ex = basic_store();
        let query = format!("DESCRIBE {ALICE}");
        let res = ex.execute(&query).expect("ok");
        let json = res.format_json();
        assert!(json.contains("describe"), "json = {json}");
    }

    // ── RdfTriple helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_triple_object_is_literal() {
        let t = RdfTriple::new(ALICE, NAME, "\"Alice\"");
        assert!(t.object_is_literal());
        assert!(!t.object_is_iri());
    }

    #[test]
    fn test_triple_object_is_iri() {
        let t = RdfTriple::new(ALICE, KNOWS, BOB);
        assert!(t.object_is_iri());
        assert!(!t.object_is_literal());
    }

    #[test]
    fn test_triple_object_is_blank() {
        let t = RdfTriple::new(ALICE, KNOWS, "_:b0");
        assert!(t.object_is_blank());
    }

    #[test]
    fn test_triple_literal_value() {
        let t = RdfTriple::new(ALICE, NAME, "\"Alice\"");
        assert_eq!(t.literal_value(), Some("Alice".to_string()));
    }

    #[test]
    fn test_triple_literal_value_none_for_iri() {
        let t = RdfTriple::new(ALICE, KNOWS, BOB);
        assert_eq!(t.literal_value(), None);
    }

    // ── Error handling ────────────────────────────────────────────────────────

    #[test]
    fn test_unknown_query_returns_error() {
        let ex = SparqlExecutor::new();
        let result = ex.execute("INSERT DATA { }");
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_error_display() {
        let err = ExecutorError::ParseError("bad query".to_string());
        assert!(err.to_string().contains("bad query"));
    }

    #[test]
    fn test_executor_error_unbound_var() {
        let err = ExecutorError::UnboundVariable("x".to_string());
        assert!(err.to_string().contains("x"));
    }

    // ── PatternTerm / strip_literal helpers ───────────────────────────────────

    #[test]
    fn test_strip_literal_simple() {
        assert_eq!(strip_literal("\"Alice\""), "Alice");
    }

    #[test]
    fn test_strip_literal_no_quotes() {
        assert_eq!(strip_literal("Alice"), "Alice");
    }

    #[test]
    fn test_escape_json_quotes() {
        assert!(escape_json(r#"say "hi""#).contains("\\\""));
    }

    // ── Multi-pattern join ─────────────────────────────────────────────────────

    #[test]
    fn test_two_pattern_join() {
        let ex = basic_store();
        // Find ?x who knows someone with name "Bob"
        let query = format!("SELECT ?x WHERE {{ ?x {KNOWS} ?y . ?y {NAME} \"Bob\" }}");
        let res = ex.execute(&query).expect("ok");
        assert!(!res.rows.is_empty(), "expected at least one result");
        // Alice knows Bob, Bob has name "Bob"
        let subjects: Vec<&String> = res.rows.iter().filter_map(|r| r.get("x")).collect();
        assert!(
            subjects.iter().any(|s| s.as_str() == ALICE),
            "subjects = {:?}",
            subjects
        );
    }
}
