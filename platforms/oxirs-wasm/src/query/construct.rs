//! Enhanced SPARQL CONSTRUCT query support for WASM
//!
//! Implements proper CONSTRUCT template parsing per SPARQL 1.1 specification,
//! including:
//!
//! - **Template triple patterns**: Parse CONSTRUCT { ?s ?p ?o . ?x ?y ?z } templates
//! - **Blank node generation**: Scoped blank node identifiers per solution mapping
//! - **Literal propagation**: Correctly propagate language tags and datatypes
//! - **Deduplication**: Remove duplicate triples from output
//! - **CONSTRUCT WHERE shorthand**: When template equals WHERE body
//! - **Multi-format serialization**: N-Triples, Turtle, and JSON-LD output
//! - **Statistics**: Track template expansion metrics

use super::{
    evaluate_pattern, parse_graph_patterns, parse_pattern_term, GraphPattern, PatternTerm,
    TriplePattern,
};
use crate::error::{WasmError, WasmResult};
use crate::store::OxiRSStore;
use crate::Triple;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for CONSTRUCT query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstructConfig {
    /// Whether to deduplicate output triples (default: true).
    pub deduplicate: bool,
    /// Maximum number of output triples (None = unlimited).
    pub max_triples: Option<usize>,
    /// Whether to track construction statistics (default: true).
    pub collect_stats: bool,
    /// Blank node prefix for generated blank nodes.
    pub blank_node_prefix: String,
}

impl Default for ConstructConfig {
    fn default() -> Self {
        Self {
            deduplicate: true,
            max_triples: None,
            collect_stats: true,
            blank_node_prefix: "b".to_string(),
        }
    }
}

// ─────────────────────────────────────────────
// Template types
// ─────────────────────────────────────────────

/// A parsed CONSTRUCT query with template and WHERE clause.
#[derive(Debug, Clone)]
pub struct ConstructQuery {
    /// Template triple patterns to instantiate per solution.
    pub template: Vec<TemplateTriple>,
    /// WHERE clause graph patterns.
    pub(crate) where_patterns: Vec<GraphPattern>,
    /// PREFIX declarations (prefix -> IRI).
    pub prefixes: HashMap<String, String>,
    /// LIMIT modifier.
    pub limit: Option<usize>,
    /// OFFSET modifier.
    pub offset: Option<usize>,
}

/// A triple pattern in the CONSTRUCT template.
#[derive(Debug, Clone)]
pub struct TemplateTriple {
    /// Subject: variable, IRI, or blank node.
    pub subject: TemplateTerm,
    /// Predicate: variable or IRI.
    pub predicate: TemplateTerm,
    /// Object: variable, IRI, blank node, or literal.
    pub object: TemplateTerm,
}

/// A term in a CONSTRUCT template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateTerm {
    /// A SPARQL variable (?name).
    Variable(String),
    /// An IRI reference.
    Iri(String),
    /// A blank node identifier.
    BlankNode(String),
    /// A plain literal.
    Literal(String),
    /// A language-tagged literal.
    LangLiteral { value: String, lang: String },
    /// A datatype-tagged literal.
    TypedLiteral { value: String, datatype: String },
}

impl TemplateTerm {
    /// Instantiate this template term using the given solution mapping.
    ///
    /// Returns None if a variable is unbound (the entire triple is skipped
    /// per SPARQL 1.1 spec section 16.2).
    fn instantiate(
        &self,
        bindings: &HashMap<String, String>,
        blank_scope: &mut HashMap<String, String>,
        blank_counter: &mut u64,
        prefix: &str,
    ) -> Option<String> {
        match self {
            TemplateTerm::Variable(name) => bindings.get(name).cloned(),
            TemplateTerm::Iri(iri) => Some(iri.clone()),
            TemplateTerm::BlankNode(label) => {
                // Scoped blank nodes: each solution mapping gets unique blank node IDs
                let entry = blank_scope.entry(label.clone()).or_insert_with(|| {
                    *blank_counter += 1;
                    format!("_:{}{}", prefix, blank_counter)
                });
                Some(entry.clone())
            }
            TemplateTerm::Literal(val) => Some(format!("\"{}\"", val)),
            TemplateTerm::LangLiteral { value, lang } => Some(format!("\"{}\"@{}", value, lang)),
            TemplateTerm::TypedLiteral { value, datatype } => {
                Some(format!("\"{}\"^^<{}>", value, datatype))
            }
        }
    }
}

// ─────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────

/// Statistics from a CONSTRUCT query execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstructStats {
    /// Number of solution mappings from WHERE clause.
    pub solution_count: usize,
    /// Number of template triples per solution.
    pub template_triple_count: usize,
    /// Total triples before deduplication.
    pub raw_triple_count: usize,
    /// Total triples after deduplication.
    pub deduped_triple_count: usize,
    /// Number of triples skipped due to unbound variables.
    pub skipped_unbound: usize,
    /// Number of blank nodes generated.
    pub blank_nodes_generated: u64,
}

// ─────────────────────────────────────────────
// CONSTRUCT engine
// ─────────────────────────────────────────────

/// Engine for executing SPARQL CONSTRUCT queries.
pub struct ConstructEngine {
    config: ConstructConfig,
}

impl ConstructEngine {
    /// Create a new CONSTRUCT engine with default configuration.
    pub fn new() -> Self {
        Self {
            config: ConstructConfig::default(),
        }
    }

    /// Create a new CONSTRUCT engine with the given configuration.
    pub fn with_config(config: ConstructConfig) -> Self {
        Self { config }
    }

    /// Execute a CONSTRUCT query against the store.
    pub fn execute(
        &self,
        sparql: &str,
        store: &OxiRSStore,
    ) -> WasmResult<(Vec<Triple>, ConstructStats)> {
        let query = parse_construct_query(sparql)?;
        self.execute_parsed(&query, store)
    }

    /// Execute a pre-parsed CONSTRUCT query.
    pub fn execute_parsed(
        &self,
        query: &ConstructQuery,
        store: &OxiRSStore,
    ) -> WasmResult<(Vec<Triple>, ConstructStats)> {
        let mut stats = ConstructStats {
            template_triple_count: query.template.len(),
            ..Default::default()
        };

        // Evaluate WHERE clause to get solution mappings
        let mut solutions: Vec<HashMap<String, String>> = vec![HashMap::new()];
        for pattern in &query.where_patterns {
            solutions = evaluate_pattern(pattern, solutions, store)?;
        }

        // Apply OFFSET
        if let Some(offset) = query.offset {
            if offset >= solutions.len() {
                solutions.clear();
            } else {
                solutions = solutions.into_iter().skip(offset).collect();
            }
        }

        // Apply LIMIT
        if let Some(limit) = query.limit {
            solutions.truncate(limit);
        }

        stats.solution_count = solutions.len();

        // Instantiate template for each solution
        let mut blank_counter: u64 = 0;
        let mut all_triples: Vec<(String, String, String)> = Vec::new();

        for solution in &solutions {
            // Each solution gets its own blank node scope
            let mut blank_scope: HashMap<String, String> = HashMap::new();

            for template_triple in &query.template {
                let s_opt = template_triple.subject.instantiate(
                    solution,
                    &mut blank_scope,
                    &mut blank_counter,
                    &self.config.blank_node_prefix,
                );
                let p_opt = template_triple.predicate.instantiate(
                    solution,
                    &mut blank_scope,
                    &mut blank_counter,
                    &self.config.blank_node_prefix,
                );
                let o_opt = template_triple.object.instantiate(
                    solution,
                    &mut blank_scope,
                    &mut blank_counter,
                    &self.config.blank_node_prefix,
                );

                match (s_opt, p_opt, o_opt) {
                    (Some(s), Some(p), Some(o)) => {
                        all_triples.push((s, p, o));
                    }
                    _ => {
                        stats.skipped_unbound += 1;
                    }
                }
            }
        }

        stats.raw_triple_count = all_triples.len();
        stats.blank_nodes_generated = blank_counter;

        // Deduplicate if configured
        let result_triples = if self.config.deduplicate {
            let mut seen: HashSet<(String, String, String)> = HashSet::new();
            let mut deduped = Vec::new();
            for triple in all_triples {
                if seen.insert(triple.clone()) {
                    deduped.push(Triple::new(&triple.0, &triple.1, &triple.2));
                }
            }
            deduped
        } else {
            all_triples
                .into_iter()
                .map(|(s, p, o)| Triple::new(&s, &p, &o))
                .collect()
        };

        stats.deduped_triple_count = result_triples.len();

        // Apply max_triples limit
        let result_triples = if let Some(max) = self.config.max_triples {
            result_triples.into_iter().take(max).collect()
        } else {
            result_triples
        };

        Ok((result_triples, stats))
    }
}

impl Default for ConstructEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
// Serialization
// ─────────────────────────────────────────────

/// Output format for CONSTRUCT results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstructOutputFormat {
    /// N-Triples format.
    NTriples,
    /// Turtle format (with prefix abbreviation).
    Turtle,
    /// JSON-LD format.
    JsonLd,
}

/// Serialize CONSTRUCT result triples to a string in the given format.
pub fn serialize_construct(
    triples: &[Triple],
    format: ConstructOutputFormat,
    prefixes: &HashMap<String, String>,
) -> WasmResult<String> {
    match format {
        ConstructOutputFormat::NTriples => serialize_ntriples(triples),
        ConstructOutputFormat::Turtle => serialize_turtle(triples, prefixes),
        ConstructOutputFormat::JsonLd => serialize_construct_jsonld(triples),
    }
}

/// Serialize triples as N-Triples.
fn serialize_ntriples(triples: &[Triple]) -> WasmResult<String> {
    let mut output = String::new();
    for triple in triples {
        let s = format_nt_term(&triple.subject());
        let p = format_nt_term(&triple.predicate());
        let o = format_nt_object(&triple.object());
        output.push_str(&format!("{} {} {} .\n", s, p, o));
    }
    Ok(output)
}

/// Format an N-Triples subject/predicate term.
fn format_nt_term(term: &str) -> String {
    if term.starts_with("_:")
        || (term.starts_with('<') && term.ends_with('>'))
        || term.starts_with('"')
    {
        term.to_string()
    } else {
        format!("<{}>", term)
    }
}

/// Format an N-Triples object term (handles literals).
fn format_nt_object(term: &str) -> String {
    if term.starts_with('"')
        || term.starts_with("_:")
        || (term.starts_with('<') && term.ends_with('>'))
    {
        term.to_string()
    } else {
        format!("<{}>", term)
    }
}

/// Serialize triples as Turtle with prefix abbreviation.
fn serialize_turtle(triples: &[Triple], prefixes: &HashMap<String, String>) -> WasmResult<String> {
    let mut output = String::new();

    // Write prefix declarations
    for (prefix, iri) in prefixes {
        output.push_str(&format!("@prefix {}: <{}> .\n", prefix, iri));
    }
    if !prefixes.is_empty() {
        output.push('\n');
    }

    // Group triples by subject for Turtle abbreviation
    let mut by_subject: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for triple in triples {
        let s = triple.subject();
        if let Some(entry) = by_subject.iter_mut().find(|(subj, _)| *subj == s) {
            entry.1.push((triple.predicate(), triple.object()));
        } else {
            by_subject.push((s, vec![(triple.predicate(), triple.object())]));
        }
    }

    for (subject, po_pairs) in &by_subject {
        let s = abbreviate_term(subject, prefixes);
        let mut first = true;
        for (predicate, object) in po_pairs {
            let p = if predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                "a".to_string()
            } else {
                abbreviate_term(predicate, prefixes)
            };
            let o = abbreviate_object(object, prefixes);

            if first {
                output.push_str(&format!("{} {} {}", s, p, o));
                first = false;
            } else {
                output.push_str(&format!(" ;\n    {} {}", p, o));
            }
        }
        output.push_str(" .\n");
    }

    Ok(output)
}

/// Abbreviate an IRI using prefix mappings.
fn abbreviate_term(term: &str, prefixes: &HashMap<String, String>) -> String {
    if term.starts_with("_:") {
        return term.to_string();
    }
    for (prefix, iri) in prefixes {
        if let Some(local) = term.strip_prefix(iri.as_str()) {
            return format!("{}:{}", prefix, local);
        }
    }
    format!("<{}>", term)
}

/// Abbreviate an object term (handles literals).
fn abbreviate_object(term: &str, prefixes: &HashMap<String, String>) -> String {
    if term.starts_with('"') || term.starts_with("_:") {
        term.to_string()
    } else {
        abbreviate_term(term, prefixes)
    }
}

/// Serialize triples as a simple JSON-LD array.
fn serialize_construct_jsonld(triples: &[Triple]) -> WasmResult<String> {
    let mut nodes: Vec<serde_json::Value> = Vec::new();

    // Group by subject
    let mut by_subject: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for triple in triples {
        by_subject
            .entry(triple.subject())
            .or_default()
            .push((triple.predicate(), triple.object()));
    }

    for (subject, po_pairs) in &by_subject {
        let mut node = serde_json::Map::new();
        node.insert(
            "@id".to_string(),
            serde_json::Value::String(subject.clone()),
        );

        for (predicate, object) in po_pairs {
            let key = if predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                "@type".to_string()
            } else {
                predicate.clone()
            };

            let value = if object.starts_with('"') {
                // Extract literal value
                let val = extract_literal_content(object);
                serde_json::json!([{"@value": val}])
            } else {
                serde_json::json!([{"@id": object}])
            };

            node.insert(key, value);
        }

        nodes.push(serde_json::Value::Object(node));
    }

    serde_json::to_string_pretty(&nodes).map_err(|e| WasmError::SerializationError(e.to_string()))
}

/// Extract the value content from a literal string like `"hello"@en` or `"42"^^<xsd:int>`.
fn extract_literal_content(literal: &str) -> String {
    let s = literal.trim();
    if let Some(stripped) = s.strip_prefix('"') {
        // Find the closing quote
        if let Some(end_quote) = stripped.find('"') {
            return stripped[..end_quote].to_string();
        }
    }
    s.to_string()
}

// ─────────────────────────────────────────────
// CONSTRUCT query parser
// ─────────────────────────────────────────────

/// Parse a SPARQL CONSTRUCT query string.
pub fn parse_construct_query(sparql: &str) -> WasmResult<ConstructQuery> {
    let sparql = sparql.trim();
    let upper = sparql.to_uppercase();

    // Parse PREFIX declarations
    let prefixes = parse_prefixes(sparql);

    // Find CONSTRUCT keyword
    let construct_pos = upper
        .find("CONSTRUCT")
        .ok_or_else(|| WasmError::QueryError("No CONSTRUCT keyword found".to_string()))?;

    // Check for CONSTRUCT WHERE shorthand
    let after_construct = &sparql[construct_pos + 9..];
    let after_upper = after_construct.trim().to_uppercase();

    if after_upper.starts_with("WHERE") {
        // CONSTRUCT WHERE { ... } shorthand: template = WHERE body
        return parse_construct_where_shorthand(sparql, construct_pos, &prefixes);
    }

    // Standard: CONSTRUCT { template } WHERE { body }
    let template_open = sparql[construct_pos + 9..]
        .find('{')
        .ok_or_else(|| WasmError::QueryError("No CONSTRUCT template body '{'".to_string()))?
        + construct_pos
        + 9;

    let template_body = extract_braces(sparql, template_open)?;
    let template = parse_template_triples(&template_body)?;

    // Find WHERE clause
    let after_template = template_open + template_body.len() + 2;
    let rest = &sparql[after_template..];
    let rest_upper = rest.to_uppercase();

    let where_pos = rest_upper
        .find("WHERE")
        .ok_or_else(|| WasmError::QueryError("No WHERE clause in CONSTRUCT query".to_string()))?;

    let where_open = rest[where_pos + 5..]
        .find('{')
        .ok_or_else(|| WasmError::QueryError("No WHERE body '{'".to_string()))?
        + where_pos
        + 5;

    let where_body = extract_braces(rest, where_open)?;
    let where_patterns = parse_graph_patterns(&where_body)?;

    // Parse LIMIT/OFFSET from tail
    let limit = parse_modifier_from(sparql, "LIMIT");
    let offset = parse_modifier_from(sparql, "OFFSET");

    Ok(ConstructQuery {
        template,
        where_patterns,
        prefixes,
        limit,
        offset,
    })
}

/// Parse the CONSTRUCT WHERE shorthand form.
fn parse_construct_where_shorthand(
    sparql: &str,
    construct_pos: usize,
    prefixes: &HashMap<String, String>,
) -> WasmResult<ConstructQuery> {
    let after_construct = &sparql[construct_pos + 9..];
    let trimmed = after_construct.trim();
    let upper = trimmed.to_uppercase();

    let where_start = upper
        .find("WHERE")
        .ok_or_else(|| WasmError::QueryError("No WHERE keyword".to_string()))?;

    let brace_start = trimmed[where_start + 5..]
        .find('{')
        .ok_or_else(|| WasmError::QueryError("No WHERE body '{'".to_string()))?
        + where_start
        + 5;

    let body = extract_braces(trimmed, brace_start)?;

    // In CONSTRUCT WHERE, the template is the same as the WHERE body
    let template = parse_template_triples(&body)?;
    let where_patterns = parse_graph_patterns(&body)?;

    let limit = parse_modifier_from(sparql, "LIMIT");
    let offset = parse_modifier_from(sparql, "OFFSET");

    Ok(ConstructQuery {
        template,
        where_patterns,
        prefixes: prefixes.clone(),
        limit,
        offset,
    })
}

/// Parse PREFIX declarations from a SPARQL query.
fn parse_prefixes(sparql: &str) -> HashMap<String, String> {
    let mut prefixes = HashMap::new();
    let upper = sparql.to_uppercase();
    let mut search_from = 0;

    while let Some(pos) = upper[search_from..].find("PREFIX") {
        let abs_pos = search_from + pos;
        let rest = &sparql[abs_pos + 6..];
        let trimmed = rest.trim_start();

        // Parse prefix:iri pattern
        if let Some(colon_pos) = trimmed.find(':') {
            let prefix = trimmed[..colon_pos].trim().to_string();
            let after_colon = trimmed[colon_pos + 1..].trim_start();

            if let Some(stripped) = after_colon.strip_prefix('<') {
                if let Some(end) = stripped.find('>') {
                    let iri = stripped[..end].to_string();
                    prefixes.insert(prefix, iri);
                }
            }
        }

        search_from = abs_pos + 6;
    }

    prefixes
}

/// Parse template triple patterns from a CONSTRUCT template body.
fn parse_template_triples(body: &str) -> WasmResult<Vec<TemplateTriple>> {
    let mut triples = Vec::new();
    let body = body.trim();
    if body.is_empty() {
        return Ok(triples);
    }

    // Split on '.' but respect quoted strings and angle brackets
    let statements = split_template_statements(body);

    for stmt in &statements {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }

        let tokens = tokenize_template(stmt);
        if tokens.len() < 3 {
            continue;
        }

        // Handle predicate-object lists with ';'
        let subject = parse_template_term(&tokens[0]);
        let mut i = 1;
        while i + 1 < tokens.len() {
            let predicate = if tokens[i] == "a" {
                TemplateTerm::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
            } else {
                parse_template_term(&tokens[i])
            };

            // Object may include language tag or datatype
            let object = parse_template_term(&tokens[i + 1]);

            triples.push(TemplateTriple {
                subject: subject.clone(),
                predicate,
                object,
            });

            i += 2;
            // Skip ';' separator for predicate-object lists
            if i < tokens.len() && tokens[i] == ";" {
                i += 1;
            }
        }
    }

    Ok(triples)
}

/// Split template body into statements on '.', respecting quotes and angle brackets.
fn split_template_statements(body: &str) -> Vec<String> {
    let chars: Vec<char> = body.chars().collect();
    let mut statements = Vec::new();
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
                '.' => {
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        statements.push(trimmed);
                    }
                    current = String::new();
                }
                _ => current.push(c),
            }
        }
        i += 1;
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        statements.push(trimmed);
    }

    statements
}

/// Tokenize a template statement, respecting quotes and angle brackets.
fn tokenize_template(s: &str) -> Vec<String> {
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
                // Check for language tag or datatype suffix
                if i + 1 < chars.len() {
                    if chars[i + 1] == '@' {
                        // Language tag: consume @lang
                        i += 1;
                        current.push(chars[i]); // '@'
                        i += 1;
                        while i < chars.len()
                            && !chars[i].is_whitespace()
                            && chars[i] != '.'
                            && chars[i] != ';'
                        {
                            current.push(chars[i]);
                            i += 1;
                        }
                        let tok = current.trim().to_string();
                        if !tok.is_empty() {
                            tokens.push(tok);
                        }
                        current = String::new();
                        continue;
                    } else if chars[i + 1] == '^' && i + 2 < chars.len() && chars[i + 2] == '^' {
                        // Datatype: consume ^^<datatype>
                        current.push('^');
                        current.push('^');
                        i += 3;
                        if i < chars.len() && chars[i] == '<' {
                            current.push('<');
                            i += 1;
                            while i < chars.len() && chars[i] != '>' {
                                current.push(chars[i]);
                                i += 1;
                            }
                            if i < chars.len() {
                                current.push('>');
                                i += 1;
                            }
                        }
                        let tok = current.trim().to_string();
                        if !tok.is_empty() {
                            tokens.push(tok);
                        }
                        current = String::new();
                        continue;
                    }
                }
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
                ';' => {
                    let tok = current.trim().to_string();
                    if !tok.is_empty() {
                        tokens.push(tok);
                    }
                    tokens.push(";".to_string());
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

/// Parse a single template term.
fn parse_template_term(term: &str) -> TemplateTerm {
    let term = term.trim();

    // Variable
    if term.starts_with('?') || term.starts_with('$') {
        return TemplateTerm::Variable(term.trim_start_matches(['?', '$']).to_string());
    }

    // Blank node
    if let Some(label) = term.strip_prefix("_:") {
        return TemplateTerm::BlankNode(label.to_string());
    }

    // IRI in angle brackets
    if term.starts_with('<') && term.ends_with('>') {
        return TemplateTerm::Iri(term[1..term.len() - 1].to_string());
    }

    // Language-tagged literal: "value"@lang
    if term.starts_with('"') && term.contains("\"@") {
        if let Some(at_pos) = term.rfind("\"@") {
            let value = term[1..at_pos].to_string();
            let lang = term[at_pos + 2..].to_string();
            return TemplateTerm::LangLiteral { value, lang };
        }
    }

    // Typed literal: "value"^^<datatype>
    if term.starts_with('"') && term.contains("\"^^<") {
        if let Some(caret_pos) = term.find("\"^^<") {
            let value = term[1..caret_pos].to_string();
            let dt_start = caret_pos + 4;
            let dt_end = term.len().saturating_sub(1);
            if dt_end > dt_start {
                let datatype = term[dt_start..dt_end].to_string();
                return TemplateTerm::TypedLiteral { value, datatype };
            }
        }
    }

    // Plain literal
    if term.starts_with('"') && term.ends_with('"') && term.len() >= 2 {
        return TemplateTerm::Literal(term[1..term.len() - 1].to_string());
    }

    // The 'a' keyword for rdf:type
    if term == "a" {
        return TemplateTerm::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string());
    }

    // Bare IRI (no angle brackets)
    TemplateTerm::Iri(term.to_string())
}

/// Extract content from a braced block.
fn extract_braces(s: &str, open_pos: usize) -> WasmResult<String> {
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
        return Err(WasmError::QueryError(
            "Unmatched '{' in CONSTRUCT query".to_string(),
        ));
    }
    Ok(chars[..pos].iter().collect())
}

/// Parse LIMIT/OFFSET modifiers.
fn parse_modifier_from(sparql: &str, modifier: &str) -> Option<usize> {
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

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_social_store() -> OxiRSStore {
        let mut store = OxiRSStore::new();
        store.insert("http://ex/alice", "http://ex/knows", "http://ex/bob");
        store.insert("http://ex/alice", "http://ex/name", "\"Alice\"");
        store.insert("http://ex/bob", "http://ex/knows", "http://ex/carol");
        store.insert("http://ex/bob", "http://ex/name", "\"Bob\"");
        store.insert("http://ex/carol", "http://ex/name", "\"Carol\"");
        store.insert(
            "http://ex/alice",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://ex/Person",
        );
        store.insert(
            "http://ex/bob",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://ex/Person",
        );
        store
    }

    // ── Config tests ──

    #[test]
    fn test_config_default() {
        let config = ConstructConfig::default();
        assert!(config.deduplicate);
        assert!(config.max_triples.is_none());
        assert!(config.collect_stats);
        assert_eq!(config.blank_node_prefix, "b");
    }

    #[test]
    fn test_config_custom() {
        let config = ConstructConfig {
            deduplicate: false,
            max_triples: Some(100),
            collect_stats: false,
            blank_node_prefix: "gen".to_string(),
        };
        assert!(!config.deduplicate);
        assert_eq!(config.max_triples, Some(100));
    }

    // ── Template term parsing ──

    #[test]
    fn test_parse_variable_term() {
        let term = parse_template_term("?name");
        assert_eq!(term, TemplateTerm::Variable("name".to_string()));
    }

    #[test]
    fn test_parse_dollar_variable_term() {
        let term = parse_template_term("$x");
        assert_eq!(term, TemplateTerm::Variable("x".to_string()));
    }

    #[test]
    fn test_parse_iri_term() {
        let term = parse_template_term("<http://example.org/foo>");
        assert_eq!(
            term,
            TemplateTerm::Iri("http://example.org/foo".to_string())
        );
    }

    #[test]
    fn test_parse_blank_node_term() {
        let term = parse_template_term("_:b0");
        assert_eq!(term, TemplateTerm::BlankNode("b0".to_string()));
    }

    #[test]
    fn test_parse_plain_literal() {
        let term = parse_template_term("\"hello\"");
        assert_eq!(term, TemplateTerm::Literal("hello".to_string()));
    }

    #[test]
    fn test_parse_lang_literal() {
        let term = parse_template_term("\"hello\"@en");
        assert_eq!(
            term,
            TemplateTerm::LangLiteral {
                value: "hello".to_string(),
                lang: "en".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_typed_literal() {
        let term = parse_template_term("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>");
        assert_eq!(
            term,
            TemplateTerm::TypedLiteral {
                value: "42".to_string(),
                datatype: "http://www.w3.org/2001/XMLSchema#integer".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_a_keyword() {
        let term = parse_template_term("a");
        assert_eq!(
            term,
            TemplateTerm::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
        );
    }

    // ── Template parsing ──

    #[test]
    fn test_parse_template_single_triple() {
        let body = "?s <http://ex/p> ?o";
        let triples = parse_template_triples(body).expect("parse");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, TemplateTerm::Variable("s".to_string()));
    }

    #[test]
    fn test_parse_template_multiple_triples() {
        let body = "?s <http://ex/p> ?o . ?s <http://ex/q> ?z";
        let triples = parse_template_triples(body).expect("parse");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_parse_template_with_blank_nodes() {
        let body = "_:b0 <http://ex/p> ?o . _:b0 <http://ex/q> _:b1";
        let triples = parse_template_triples(body).expect("parse");
        assert_eq!(triples.len(), 2);
        assert_eq!(
            triples[0].subject,
            TemplateTerm::BlankNode("b0".to_string())
        );
    }

    #[test]
    fn test_parse_template_empty_body() {
        let triples = parse_template_triples("").expect("parse");
        assert!(triples.is_empty());
    }

    // ── CONSTRUCT query parsing ──

    #[test]
    fn test_parse_construct_basic() {
        let sparql = "CONSTRUCT { ?s <http://ex/p> ?o } WHERE { ?s <http://ex/knows> ?o }";
        let query = parse_construct_query(sparql).expect("parse");
        assert_eq!(query.template.len(), 1);
        assert!(!query.where_patterns.is_empty());
    }

    #[test]
    fn test_parse_construct_with_prefix() {
        let sparql = r#"
            PREFIX ex: <http://ex/>
            CONSTRUCT { ?s ex:p ?o }
            WHERE { ?s ex:knows ?o }
        "#;
        let query = parse_construct_query(sparql).expect("parse");
        assert!(query.prefixes.contains_key("ex"));
        assert_eq!(
            query.prefixes.get("ex").map(|s| s.as_str()),
            Some("http://ex/")
        );
    }

    #[test]
    fn test_parse_construct_where_shorthand() {
        let sparql = "CONSTRUCT WHERE { ?s <http://ex/knows> ?o }";
        let query = parse_construct_query(sparql).expect("parse");
        assert_eq!(query.template.len(), 1);
        assert!(!query.where_patterns.is_empty());
    }

    #[test]
    fn test_parse_construct_with_limit() {
        let sparql = "CONSTRUCT { ?s <http://ex/p> ?o } WHERE { ?s <http://ex/knows> ?o } LIMIT 5";
        let query = parse_construct_query(sparql).expect("parse");
        assert_eq!(query.limit, Some(5));
    }

    #[test]
    fn test_parse_construct_with_offset() {
        let sparql = "CONSTRUCT { ?s <http://ex/p> ?o } WHERE { ?s <http://ex/knows> ?o } OFFSET 2";
        let query = parse_construct_query(sparql).expect("parse");
        assert_eq!(query.offset, Some(2));
    }

    // ── Engine execution tests ──

    #[test]
    fn test_construct_basic_execution() {
        let store = make_social_store();
        let engine = ConstructEngine::new();
        let sparql = "CONSTRUCT { ?s <http://ex/friendOf> ?o } WHERE { ?s <http://ex/knows> ?o }";
        let (triples, stats) = engine.execute(sparql, &store).expect("execute");
        assert_eq!(triples.len(), 2); // alice->bob, bob->carol
        assert_eq!(stats.solution_count, 2);
        assert_eq!(stats.template_triple_count, 1);
    }

    #[test]
    fn test_construct_multi_template() {
        let store = make_social_store();
        let engine = ConstructEngine::new();
        let sparql = r#"
            CONSTRUCT {
                ?s <http://ex/friendOf> ?o .
                ?o <http://ex/knownBy> ?s
            } WHERE {
                ?s <http://ex/knows> ?o
            }
        "#;
        let (triples, stats) = engine.execute(sparql, &store).expect("execute");
        assert_eq!(triples.len(), 4); // 2 solutions x 2 template triples
        assert_eq!(stats.template_triple_count, 2);
    }

    #[test]
    fn test_construct_deduplication() {
        let mut store = OxiRSStore::new();
        store.insert("http://ex/a", "http://ex/p", "http://ex/b");
        store.insert("http://ex/a", "http://ex/q", "http://ex/b");

        let engine = ConstructEngine::new();
        // Both solutions map ?s to the same value, so the constructed triple is identical
        let sparql =
            "CONSTRUCT { <http://ex/a> <http://ex/r> <http://ex/b> } WHERE { ?s ?p <http://ex/b> }";
        let (triples, stats) = engine.execute(sparql, &store).expect("execute");
        assert_eq!(triples.len(), 1); // Deduplicated
        assert_eq!(stats.raw_triple_count, 2);
        assert_eq!(stats.deduped_triple_count, 1);
    }

    #[test]
    fn test_construct_no_deduplication() {
        let mut store = OxiRSStore::new();
        store.insert("http://ex/a", "http://ex/p", "http://ex/b");
        store.insert("http://ex/a", "http://ex/q", "http://ex/b");

        let config = ConstructConfig {
            deduplicate: false,
            ..Default::default()
        };
        let engine = ConstructEngine::with_config(config);
        let sparql =
            "CONSTRUCT { <http://ex/a> <http://ex/r> <http://ex/b> } WHERE { ?s ?p <http://ex/b> }";
        let (triples, _) = engine.execute(sparql, &store).expect("execute");
        assert_eq!(triples.len(), 2); // Not deduplicated
    }

    #[test]
    fn test_construct_unbound_variable_skipped() {
        let mut store = OxiRSStore::new();
        store.insert("http://ex/a", "http://ex/p", "http://ex/b");

        let engine = ConstructEngine::new();
        // ?name is unbound -> triple should be skipped
        let sparql = r#"
            CONSTRUCT {
                ?s <http://ex/named> ?name .
                ?s <http://ex/linked> ?o
            } WHERE {
                ?s <http://ex/p> ?o
            }
        "#;
        let (triples, stats) = engine.execute(sparql, &store).expect("execute");
        assert_eq!(triples.len(), 1); // Only the ?s linked ?o triple
        assert_eq!(stats.skipped_unbound, 1);
    }

    #[test]
    fn test_construct_blank_node_scoping() {
        let mut store = OxiRSStore::new();
        store.insert("http://ex/alice", "http://ex/knows", "http://ex/bob");
        store.insert("http://ex/carol", "http://ex/knows", "http://ex/dave");

        let engine = ConstructEngine::new();
        let sparql = r#"
            CONSTRUCT {
                _:node <http://ex/from> ?s .
                _:node <http://ex/to> ?o
            } WHERE {
                ?s <http://ex/knows> ?o
            }
        "#;
        let (triples, stats) = engine.execute(sparql, &store).expect("execute");
        // 2 solutions x 2 template triples = 4 triples
        assert_eq!(triples.len(), 4);
        // Each solution's _:node should be different
        let subjects: Vec<String> = triples.iter().map(|t| t.subject()).collect();
        let unique_blanks: HashSet<&String> =
            subjects.iter().filter(|s| s.starts_with("_:")).collect();
        assert_eq!(unique_blanks.len(), 2); // Two distinct blank nodes
        assert!(stats.blank_nodes_generated >= 2);
    }

    #[test]
    fn test_construct_with_limit() {
        let store = make_social_store();
        let engine = ConstructEngine::new();
        let sparql = "CONSTRUCT { ?s <http://ex/f> ?o } WHERE { ?s <http://ex/knows> ?o } LIMIT 1";
        let (triples, stats) = engine.execute(sparql, &store).expect("execute");
        assert_eq!(triples.len(), 1);
        assert_eq!(stats.solution_count, 1);
    }

    #[test]
    fn test_construct_with_offset() {
        let store = make_social_store();
        let engine = ConstructEngine::new();
        let sparql = "CONSTRUCT { ?s <http://ex/f> ?o } WHERE { ?s <http://ex/knows> ?o } OFFSET 1";
        let (triples, stats) = engine.execute(sparql, &store).expect("execute");
        assert_eq!(triples.len(), 1);
        assert_eq!(stats.solution_count, 1);
    }

    #[test]
    fn test_construct_max_triples_limit() {
        let store = make_social_store();
        let config = ConstructConfig {
            max_triples: Some(1),
            ..Default::default()
        };
        let engine = ConstructEngine::with_config(config);
        let sparql = "CONSTRUCT { ?s <http://ex/f> ?o } WHERE { ?s <http://ex/knows> ?o }";
        let (triples, _) = engine.execute(sparql, &store).expect("execute");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_construct_empty_result() {
        let store = OxiRSStore::new();
        let engine = ConstructEngine::new();
        let sparql = "CONSTRUCT { ?s <http://ex/p> ?o } WHERE { ?s <http://ex/nonexistent> ?o }";
        let (triples, stats) = engine.execute(sparql, &store).expect("execute");
        assert!(triples.is_empty());
        assert_eq!(stats.solution_count, 0);
    }

    #[test]
    fn test_construct_where_shorthand_execution() {
        let store = make_social_store();
        let engine = ConstructEngine::new();
        let sparql = "CONSTRUCT WHERE { ?s <http://ex/knows> ?o }";
        let (triples, _) = engine.execute(sparql, &store).expect("execute");
        assert_eq!(triples.len(), 2);
    }

    // ── Serialization tests ──

    #[test]
    fn test_serialize_ntriples() {
        let triples = vec![
            Triple::new("http://ex/alice", "http://ex/knows", "http://ex/bob"),
            Triple::new("http://ex/alice", "http://ex/name", "\"Alice\""),
        ];
        let output = serialize_ntriples(&triples).expect("serialize");
        assert!(output.contains("<http://ex/alice>"));
        assert!(output.contains("<http://ex/knows>"));
        assert!(output.contains("<http://ex/bob>"));
        assert!(output.contains("\"Alice\""));
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_serialize_turtle_with_prefixes() {
        let triples = vec![Triple::new(
            "http://ex/alice",
            "http://ex/knows",
            "http://ex/bob",
        )];
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://ex/".to_string());
        let output = serialize_turtle(&triples, &prefixes).expect("serialize");
        assert!(output.contains("@prefix ex: <http://ex/>"));
        assert!(output.contains("ex:alice"));
        assert!(output.contains("ex:knows"));
        assert!(output.contains("ex:bob"));
    }

    #[test]
    fn test_serialize_turtle_rdf_type_abbreviation() {
        let triples = vec![Triple::new(
            "http://ex/alice",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://ex/Person",
        )];
        let prefixes = HashMap::new();
        let output = serialize_turtle(&triples, &prefixes).expect("serialize");
        assert!(output.contains(" a "));
    }

    #[test]
    fn test_serialize_turtle_subject_grouping() {
        let triples = vec![
            Triple::new("http://ex/alice", "http://ex/knows", "http://ex/bob"),
            Triple::new("http://ex/alice", "http://ex/name", "\"Alice\""),
        ];
        let prefixes = HashMap::new();
        let output = serialize_turtle(&triples, &prefixes).expect("serialize");
        // Both predicates should be grouped under the same subject with ';'
        assert!(output.contains(';'));
    }

    #[test]
    fn test_serialize_jsonld() {
        let triples = vec![Triple::new(
            "http://ex/alice",
            "http://ex/knows",
            "http://ex/bob",
        )];
        let output = serialize_construct_jsonld(&triples).expect("serialize");
        assert!(output.contains("@id"));
        assert!(output.contains("http://ex/alice"));
    }

    #[test]
    fn test_serialize_construct_all_formats() {
        let triples = vec![Triple::new("http://ex/a", "http://ex/b", "http://ex/c")];
        let prefixes = HashMap::new();

        let nt =
            serialize_construct(&triples, ConstructOutputFormat::NTriples, &prefixes).expect("nt");
        assert!(nt.contains("<http://ex/a>"));

        let ttl =
            serialize_construct(&triples, ConstructOutputFormat::Turtle, &prefixes).expect("ttl");
        assert!(ttl.contains("<http://ex/a>"));

        let jld = serialize_construct(&triples, ConstructOutputFormat::JsonLd, &prefixes)
            .expect("jsonld");
        assert!(jld.contains("http://ex/a"));
    }

    // ── Blank node N-Triples serialization ──

    #[test]
    fn test_serialize_ntriples_blank_nodes() {
        let triples = vec![Triple::new("_:b1", "http://ex/p", "http://ex/o")];
        let output = serialize_ntriples(&triples).expect("serialize");
        assert!(output.contains("_:b1"));
    }

    // ── PREFIX parsing ──

    #[test]
    fn test_parse_prefixes_single() {
        let sparql = "PREFIX ex: <http://example.org/> SELECT * WHERE { ?s ?p ?o }";
        let prefixes = parse_prefixes(sparql);
        assert_eq!(
            prefixes.get("ex").map(|s| s.as_str()),
            Some("http://example.org/")
        );
    }

    #[test]
    fn test_parse_prefixes_multiple() {
        let sparql = r#"
            PREFIX ex: <http://example.org/>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT * WHERE { ?s ?p ?o }
        "#;
        let prefixes = parse_prefixes(sparql);
        assert_eq!(prefixes.len(), 2);
        assert!(prefixes.contains_key("ex"));
        assert!(prefixes.contains_key("foaf"));
    }

    #[test]
    fn test_parse_prefixes_empty() {
        let sparql = "SELECT * WHERE { ?s ?p ?o }";
        let prefixes = parse_prefixes(sparql);
        assert!(prefixes.is_empty());
    }

    // ── Extract literal content ──

    #[test]
    fn test_extract_literal_content_plain() {
        assert_eq!(extract_literal_content("\"hello\""), "hello");
    }

    #[test]
    fn test_extract_literal_content_lang() {
        assert_eq!(extract_literal_content("\"hello\"@en"), "hello");
    }

    #[test]
    fn test_extract_literal_content_typed() {
        assert_eq!(
            extract_literal_content("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"),
            "42"
        );
    }

    // ── Stats ──

    #[test]
    fn test_construct_stats_tracking() {
        let store = make_social_store();
        let engine = ConstructEngine::new();
        let sparql = r#"
            CONSTRUCT {
                ?s <http://ex/friendOf> ?o .
                ?o <http://ex/knownBy> ?s
            } WHERE {
                ?s <http://ex/knows> ?o
            }
        "#;
        let (_, stats) = engine.execute(sparql, &store).expect("execute");
        assert_eq!(stats.solution_count, 2);
        assert_eq!(stats.template_triple_count, 2);
        assert_eq!(stats.raw_triple_count, 4);
        assert_eq!(stats.deduped_triple_count, 4);
        assert_eq!(stats.skipped_unbound, 0);
    }

    // ── Abbreviation ──

    #[test]
    fn test_abbreviate_term_with_prefix() {
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());
        assert_eq!(
            abbreviate_term("http://example.org/alice", &prefixes),
            "ex:alice"
        );
    }

    #[test]
    fn test_abbreviate_term_no_match() {
        let prefixes = HashMap::new();
        assert_eq!(
            abbreviate_term("http://other.org/foo", &prefixes),
            "<http://other.org/foo>"
        );
    }

    #[test]
    fn test_abbreviate_blank_node() {
        let prefixes = HashMap::new();
        assert_eq!(abbreviate_term("_:b1", &prefixes), "_:b1");
    }

    // ── ConstructEngine default ──

    #[test]
    fn test_engine_default() {
        let engine = ConstructEngine::default();
        assert!(engine.config.deduplicate);
    }

    // ── Config serialization ──

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = ConstructConfig {
            deduplicate: false,
            max_triples: Some(50),
            collect_stats: true,
            blank_node_prefix: "test".to_string(),
        };
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: ConstructConfig = serde_json::from_str(&json).expect("deserialize");
        assert!(!deserialized.deduplicate);
        assert_eq!(deserialized.max_triples, Some(50));
        assert_eq!(deserialized.blank_node_prefix, "test");
    }

    #[test]
    fn test_stats_serialization_roundtrip() {
        let stats = ConstructStats {
            solution_count: 10,
            template_triple_count: 3,
            raw_triple_count: 30,
            deduped_triple_count: 25,
            skipped_unbound: 2,
            blank_nodes_generated: 5,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        let deserialized: ConstructStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.solution_count, 10);
        assert_eq!(deserialized.deduped_triple_count, 25);
    }

    // ── Split template statements ──

    #[test]
    fn test_split_template_single() {
        let stmts = split_template_statements("?s <http://p> ?o");
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn test_split_template_multiple() {
        let stmts = split_template_statements("?s <http://p> ?o . ?a <http://q> ?b");
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn test_split_template_quoted_dot() {
        let stmts = split_template_statements("?s <http://p> \"hello. world\"");
        assert_eq!(stmts.len(), 1); // Dot inside quotes should not split
    }

    // ── Tokenize template ──

    #[test]
    fn test_tokenize_template_basic() {
        let tokens = tokenize_template("?s <http://p> ?o");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], "?s");
        assert_eq!(tokens[1], "<http://p>");
        assert_eq!(tokens[2], "?o");
    }

    #[test]
    fn test_tokenize_template_with_semicolon() {
        let tokens = tokenize_template("?s <http://p> ?o ; <http://q> ?z");
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[3], ";");
    }
}
