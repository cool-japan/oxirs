//! # Import Command Runner
//!
//! Core import execution: format dispatch, helper utilities, and the main
//! `ImportCommand` struct with its format-detection entry point.

use std::collections::HashMap;

use super::import_command_formats::{
    parse_csv_impl, parse_jsonld_impl, parse_nquads_impl, parse_ntriples_impl, parse_rdfxml_impl,
    parse_trig_impl, parse_turtle_impl,
};
use super::import_command_types::{ImportError, ImportFormat, ImportResult, Triple};

// ---------------------------------------------------------------------------
// ImportCommand
// ---------------------------------------------------------------------------

/// Stateless multi-format RDF importer.
pub struct ImportCommand;

impl ImportCommand {
    // -----------------------------------------------------------------------
    // Main entry point
    // -----------------------------------------------------------------------

    /// Parse `input` using the specified `format`.
    ///
    /// # Errors
    /// Returns `ImportError::EmptyInput` if the trimmed input is empty.
    pub fn import(input: &str, format: ImportFormat) -> Result<ImportResult, ImportError> {
        if input.trim().is_empty() {
            return Err(ImportError::EmptyInput);
        }
        match format {
            ImportFormat::NTriples => parse_ntriples_impl(input),
            ImportFormat::NQuads => parse_nquads_impl(input),
            ImportFormat::Turtle => parse_turtle_impl(input),
            ImportFormat::TriG => parse_trig_impl(input),
            ImportFormat::Csv => parse_csv_impl(input),
            ImportFormat::JsonLd => parse_jsonld_impl(input),
            ImportFormat::RdfXml => parse_rdfxml_impl(input),
        }
    }

    // -----------------------------------------------------------------------
    // Per-format public dispatch wrappers
    // -----------------------------------------------------------------------

    /// Parse N-Triples: `<subject> <predicate> <object> .` one per line.
    pub fn parse_ntriples(input: &str) -> Result<ImportResult, ImportError> {
        parse_ntriples_impl(input)
    }

    /// Parse N-Quads: `<s> <p> <o> [<g>] .` one per line.
    pub fn parse_nquads(input: &str) -> Result<ImportResult, ImportError> {
        parse_nquads_impl(input)
    }

    /// Parse simplified Turtle.
    pub fn parse_turtle(input: &str) -> Result<ImportResult, ImportError> {
        parse_turtle_impl(input)
    }

    /// Parse simplified TriG.
    pub fn parse_trig(input: &str) -> Result<ImportResult, ImportError> {
        parse_trig_impl(input)
    }

    /// Parse CSV.
    pub fn parse_csv(input: &str) -> Result<ImportResult, ImportError> {
        parse_csv_impl(input)
    }

    /// Parse JSON-LD.
    pub fn parse_jsonld(input: &str) -> Result<ImportResult, ImportError> {
        parse_jsonld_impl(input)
    }

    /// Parse RDF/XML.
    pub fn parse_rdfxml(input: &str) -> Result<ImportResult, ImportError> {
        parse_rdfxml_impl(input)
    }

    // -----------------------------------------------------------------------
    // Format detection
    // -----------------------------------------------------------------------

    /// Sniff the content of `input` to guess its format.
    pub fn detect_format(input: &str) -> Option<ImportFormat> {
        let trimmed = input.trim_start();
        if trimmed.starts_with("@prefix")
            || trimmed.starts_with("@base")
            || (trimmed.starts_with('<') && trimmed.contains('>'))
                && !trimmed.contains("<?xml")
                && !trimmed.contains("<rdf:")
        {
            let first_line = trimmed.lines().next().unwrap_or("");
            if first_line.starts_with("@prefix") || first_line.starts_with("@base") {
                return Some(ImportFormat::Turtle);
            }
        }
        if trimmed.starts_with('@') {
            return Some(ImportFormat::Turtle);
        }
        if trimmed.starts_with('{')
            || trimmed.contains("\"@context\"")
            || trimmed.contains("\"@id\"")
        {
            return Some(ImportFormat::JsonLd);
        }
        if trimmed.starts_with("<?xml")
            || trimmed.starts_with("<rdf:RDF")
            || trimmed.contains("<rdf:Description")
        {
            return Some(ImportFormat::RdfXml);
        }
        for line in trimmed.lines().take(20) {
            let l = line.trim();
            if l.to_uppercase().starts_with("GRAPH") && l.contains('<') {
                return Some(ImportFormat::TriG);
            }
        }
        let first_line = trimmed.lines().next().unwrap_or("").to_lowercase();
        if first_line.contains("subject")
            && first_line.contains("predicate")
            && first_line.contains("object")
        {
            return Some(ImportFormat::Csv);
        }
        let sample = trimmed.lines().next().unwrap_or("");
        let iri_count = sample.matches('<').count();
        if iri_count >= 4 {
            return Some(ImportFormat::NQuads);
        }
        if iri_count >= 2 && sample.ends_with('.') {
            return Some(ImportFormat::NTriples);
        }
        None
    }

    // -----------------------------------------------------------------------
    // Internal helpers — exposed publicly for tests
    // -----------------------------------------------------------------------

    /// Strip angle brackets from an IRI token: `<http://...>` → `http://...`
    pub fn strip_iri(s: &str) -> &str {
        let s = s.trim();
        if s.starts_with('<') && s.ends_with('>') {
            &s[1..s.len() - 1]
        } else {
            s
        }
    }

    /// Unescape common N-Triples escape sequences in a literal string.
    pub fn unescape_literal(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('t') => result.push('\t'),
                    Some('r') => result.push('\r'),
                    Some('"') => result.push('"'),
                    Some('\\') => result.push('\\'),
                    Some('u') => {
                        let hex: String = chars.by_ref().take(4).collect();
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(ch) = char::from_u32(code) {
                                result.push(ch);
                                continue;
                            }
                        }
                        result.push_str(&format!("\\u{}", hex));
                    }
                    Some('U') => {
                        let hex: String = chars.by_ref().take(8).collect();
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(ch) = char::from_u32(code) {
                                result.push(ch);
                                continue;
                            }
                        }
                        result.push_str(&format!("\\U{}", hex));
                    }
                    Some(other) => {
                        result.push('\\');
                        result.push(other);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(c);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Internal line-level parser helpers (pub(crate) for use by formats module)
// ---------------------------------------------------------------------------

/// Tokenise an N-Triples / N-Quads line, respecting quoted literals.
pub(crate) fn tokenise_nt_line(line: &str) -> Vec<String> {
    let line = line.trim_end_matches('.');
    let line = line.trim();
    let mut tokens = Vec::new();
    let mut chars = line.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' => {
                chars.next();
            }
            '<' => {
                let mut tok = String::from('<');
                chars.next();
                for ch in chars.by_ref() {
                    tok.push(ch);
                    if ch == '>' {
                        break;
                    }
                }
                tokens.push(tok);
            }
            '"' => {
                let mut tok = String::from('"');
                chars.next();
                let mut escaped = false;
                for ch in chars.by_ref() {
                    tok.push(ch);
                    if escaped {
                        escaped = false;
                    } else if ch == '\\' {
                        escaped = true;
                    } else if ch == '"' {
                        break;
                    }
                }
                if let Some(&next) = chars.peek() {
                    if next == '^' || next == '@' {
                        tok.push(next);
                        chars.next();
                        for ch in chars.by_ref() {
                            if ch == ' ' || ch == '\t' {
                                break;
                            }
                            tok.push(ch);
                        }
                    }
                }
                tokens.push(tok);
            }
            '_' => {
                let mut tok = String::new();
                for ch in chars.by_ref() {
                    if ch == ' ' || ch == '\t' {
                        break;
                    }
                    tok.push(ch);
                }
                tokens.push(tok);
            }
            '.' => {
                chars.next();
            }
            _ => {
                let mut tok = String::new();
                for ch in chars.by_ref() {
                    if ch == ' ' || ch == '\t' {
                        break;
                    }
                    tok.push(ch);
                }
                tokens.push(tok);
            }
        }
    }
    tokens
}

/// Convert an N-Triples term token to a string value.
pub(crate) fn parse_nt_term(token: &str) -> Result<String, String> {
    let t = token.trim();
    if t.starts_with('<') && t.ends_with('>') {
        return Ok(t[1..t.len() - 1].to_string());
    }
    if t.starts_with("_:") {
        return Ok(t.to_string());
    }
    if t.starts_with('"') {
        return Ok(ImportCommand::unescape_literal(
            &t[1..t.rfind('"').unwrap_or(t.len())],
        ));
    }
    Err(format!("unrecognised term: {}", t))
}

/// Parse a Turtle `@prefix px: <iri> .` or SPARQL `PREFIX px: <iri>` line.
pub(crate) fn parse_prefix_decl(line: &str) -> Option<(String, String)> {
    let line = line
        .trim_start_matches("@prefix")
        .trim_start_matches("PREFIX")
        .trim()
        .trim_end_matches('.');
    let colon = line.find(':')?;
    let prefix = line[..colon].trim().to_string();
    let rest = line[colon + 1..].trim();
    if rest.starts_with('<') && rest.ends_with('>') {
        return Some((prefix, rest[1..rest.len() - 1].to_string()));
    }
    None
}

/// Parse a Turtle triple line, expanding prefixed names with `prefixes`.
pub(crate) fn parse_turtle_triple(
    line: &str,
    prefixes: &HashMap<String, String>,
) -> Result<Option<Triple>, String> {
    let line = line.trim_end_matches(['.', ';', ','].as_ref()).trim();
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let tokens = tokenise_turtle_line(line, prefixes);
    if tokens.len() < 3 {
        if tokens.is_empty() {
            return Ok(None);
        }
        return Err(format!("need 3 terms, got {}", tokens.len()));
    }
    Ok(Some(Triple {
        subject: tokens[0].clone(),
        predicate: tokens[1].clone(),
        object: tokens[2].clone(),
        graph: None,
    }))
}

/// Simplified Turtle tokeniser that expands `prefix:local` names.
pub(crate) fn tokenise_turtle_line(line: &str, prefixes: &HashMap<String, String>) -> Vec<String> {
    let mut tokens = Vec::new();
    for token in line.split_whitespace() {
        let tok = token.trim_end_matches(['.', ';', ','].as_ref());
        if tok.is_empty() {
            continue;
        }
        if tok.starts_with('<') && tok.ends_with('>') {
            tokens.push(tok[1..tok.len() - 1].to_string());
        } else if tok.starts_with('"') {
            tokens.push(tok.trim_matches('"').to_string());
        } else if tok.contains(':') && !tok.starts_with("http") {
            let colon = tok.find(':').unwrap_or(tok.len());
            let pfx = &tok[..colon];
            let local = &tok[colon + 1..];
            if let Some(ns) = prefixes.get(pfx) {
                tokens.push(format!("{}{}", ns, local));
            } else {
                tokens.push(tok.to_string());
            }
        } else {
            tokens.push(tok.to_string());
        }
    }
    tokens
}

/// Extract the named-graph IRI from a `GRAPH <iri> {` line.
pub(crate) fn extract_graph_iri(line: &str) -> Option<String> {
    let start = line.find('<')? + 1;
    let end = line[start..].find('>')? + start;
    Some(line[start..end].to_string())
}

/// Extract an XML attribute value: `attr="value"`.
pub(crate) fn extract_xml_attr(text: &str, attr: &str) -> Option<String> {
    let search = format!("{}=\"", attr);
    let start = text.find(&search)? + search.len();
    let end = text[start..].find('"')? + start;
    Some(text[start..end].to_string())
}

/// Extract a JSON key-value string pair from a line: `"key": "value"`.
pub(crate) fn extract_json_string_pair(line: &str) -> Option<(String, String)> {
    let colon = line.find(":")?;
    let key_part = line[..colon].trim().trim_matches('"').to_string();
    let val_part = line[colon + 1..].trim();
    let value = if val_part.starts_with('"') {
        val_part.trim_matches(['"', ','].as_ref()).to_string()
    } else {
        return None;
    };
    Some((key_part, value))
}

/// Extract the string value from a JSON `"key": "value"` segment.
pub(crate) fn extract_json_string_value(s: &str) -> Option<String> {
    let start = s.find('"')? + 1;
    let end = s[start..].find('"')? + start;
    Some(s[start..end].to_string())
}

/// Find the position of the matching `}` for an opening `{` at position 0.
pub(crate) fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Find the start of the enclosing JSON object (last `{` before pos).
pub(crate) fn find_obj_start(s: &str) -> usize {
    s.rfind('{').map(|p| p + 1).unwrap_or(0)
}

/// Find the end of the current JSON object (next `}`).
pub(crate) fn find_obj_end(s: &str) -> Option<usize> {
    s.find('}').map(|p| p + 1)
}
