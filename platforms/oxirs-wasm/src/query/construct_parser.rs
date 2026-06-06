//! Parser for SPARQL CONSTRUCT query strings.
//!
//! Implements CONSTRUCT template parsing per SPARQL 1.1, including the
//! `CONSTRUCT WHERE` shorthand, PREFIX declarations, and LIMIT/OFFSET
//! modifiers.

use super::construct_types::{ConstructQuery, TemplateTerm, TemplateTriple};
use super::parse_graph_patterns;
use crate::error::{WasmError, WasmResult};
use std::collections::HashMap;

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
pub(crate) fn parse_prefixes(sparql: &str) -> HashMap<String, String> {
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
pub(crate) fn parse_template_triples(body: &str) -> WasmResult<Vec<TemplateTriple>> {
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
pub(crate) fn split_template_statements(body: &str) -> Vec<String> {
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
pub(crate) fn tokenize_template(s: &str) -> Vec<String> {
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
pub(crate) fn parse_template_term(term: &str) -> TemplateTerm {
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
