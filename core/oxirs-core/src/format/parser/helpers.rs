//! Common parsing utilities shared across RDF formats
//!
//! Note: Some functions are reserved for future native RDF parser implementation

#![allow(dead_code)]

use super::super::error::{ParseResult, RdfParseError, RdfSyntaxError};
use crate::model::Literal;
use crate::NamedNode;
use std::collections::HashMap;

/// Helper to tokenize N-Quads/N-Triples line respecting quotes
pub(super) fn tokenize_nquads_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();
    let mut in_quotes = false;
    let mut in_angle_brackets = false;
    let mut escape_next = false;

    for ch in line.chars() {
        if escape_next {
            current_token.push(ch);
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if in_quotes => {
                current_token.push(ch);
                escape_next = true;
            }
            '"' => {
                current_token.push(ch);
                in_quotes = !in_quotes;
            }
            '<' if !in_quotes => {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                current_token.push(ch);
                in_angle_brackets = true;
            }
            '>' if !in_quotes && in_angle_brackets => {
                current_token.push(ch);
                in_angle_brackets = false;
                tokens.push(current_token.clone());
                current_token.clear();
            }
            ' ' | '\t' if !in_quotes && !in_angle_brackets => {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
            }
            _ => {
                current_token.push(ch);
            }
        }
    }

    if !current_token.is_empty() {
        tokens.push(current_token);
    }

    tokens
}

/// Parse a prefix declaration (optimized)
pub(super) fn parse_prefix_declaration(line: &str) -> ParseResult<Option<(String, String)>> {
    // @prefix prefix: <iri> .
    if !line.starts_with("@prefix") && !line.to_uppercase().starts_with("PREFIX") {
        return Ok(None);
    }

    let line = line
        .trim_start_matches("@prefix")
        .trim_start_matches("PREFIX")
        .trim();

    if let Some(colon_pos) = line.find(':') {
        let prefix = line[..colon_pos].trim().to_string();
        let rest = line[colon_pos + 1..].trim();

        if let Some(start) = rest.find('<') {
            if let Some(end) = rest.find('>') {
                let iri = rest[start + 1..end].trim().to_string();
                return Ok(Some((prefix, iri)));
            }
        }
    }

    Ok(None)
}

/// Parse a base declaration (optimized)
pub(super) fn parse_base_declaration(line: &str) -> ParseResult<Option<String>> {
    // @base <iri> .
    if !line.starts_with("@base") && !line.to_uppercase().starts_with("BASE") {
        return Ok(None);
    }

    let line = line
        .trim_start_matches("@base")
        .trim_start_matches("BASE")
        .trim();

    if let Some(start) = line.find('<') {
        if let Some(end) = line.find('>') {
            let iri = line[start + 1..end].trim().to_string();
            return Ok(Some(iri));
        }
    }

    Ok(None)
}

/// Expand a prefixed name (optimized to reduce allocations)
pub(super) fn expand_prefixed_name(
    prefixed: &str,
    prefixes: &HashMap<String, String>,
) -> ParseResult<String> {
    if let Some(colon_pos) = prefixed.find(':') {
        let prefix = &prefixed[..colon_pos];
        let local_name = &prefixed[colon_pos + 1..];

        if let Some(namespace) = prefixes.get(prefix) {
            let mut result = String::with_capacity(namespace.len() + local_name.len());
            result.push_str(namespace);
            result.push_str(local_name);
            Ok(result)
        } else {
            Err(RdfParseError::Syntax(RdfSyntaxError {
                message: format!("Unknown prefix: '{}'", prefix),
                position: None,
                context: None,
            }))
        }
    } else {
        Ok(prefixed.to_string())
    }
}

/// Resolve a relative IRI against a base (optimized to reduce allocations)
pub(super) fn resolve_iri(iri: &str, base: Option<&str>) -> ParseResult<String> {
    if iri.contains("://") {
        Ok(iri.to_string())
    } else if let Some(base_iri) = base {
        if base_iri.ends_with('/') {
            let mut result = String::with_capacity(base_iri.len() + iri.len());
            result.push_str(base_iri);
            result.push_str(iri);
            Ok(result)
        } else {
            let mut result = String::with_capacity(base_iri.len() + 1 + iri.len());
            result.push_str(base_iri);
            result.push('/');
            result.push_str(iri);
            Ok(result)
        }
    } else {
        Ok(iri.to_string())
    }
}

/// Parse a literal from N-Quads/N-Triples format
pub(super) fn parse_literal_from_nquads(literal_str: &str) -> ParseResult<Literal> {
    if !literal_str.starts_with('"') {
        return Err(RdfParseError::Syntax(RdfSyntaxError {
            message: format!("Invalid literal: must start with quote: {}", literal_str),
            position: None,
            context: None,
        }));
    }

    let mut end_quote = 1;
    let mut escaped = false;
    let chars: Vec<char> = literal_str.chars().collect();

    while end_quote < chars.len() {
        if escaped {
            escaped = false;
        } else if chars[end_quote] == '\\' {
            escaped = true;
        } else if chars[end_quote] == '"' {
            break;
        }
        end_quote += 1;
    }

    if end_quote >= chars.len() {
        return Err(RdfParseError::Syntax(RdfSyntaxError {
            message: format!("Unterminated literal: {}", literal_str),
            position: None,
            context: None,
        }));
    }

    let value_str: String = chars[1..end_quote].iter().collect();
    let value = unescape_literal(&value_str)?;
    let remainder = &literal_str[end_quote + 1..];

    if let Some(lang_tag) = remainder.strip_prefix('@') {
        Ok(Literal::new_lang(
            value,
            lang_tag.trim_end_matches('.').trim(),
        )?)
    } else if let Some(datatype_str) = remainder.strip_prefix("^^") {
        let datatype_str = datatype_str.trim_end_matches('.').trim();
        if datatype_str.starts_with('<') && datatype_str.ends_with('>') {
            let datatype_iri = &datatype_str[1..datatype_str.len() - 1];
            let datatype = NamedNode::new(datatype_iri)?;
            Ok(Literal::new_typed(value, datatype))
        } else {
            Err(RdfParseError::Syntax(RdfSyntaxError {
                message: format!("Invalid datatype IRI: {}", datatype_str),
                position: None,
                context: None,
            }))
        }
    } else if remainder.trim().is_empty() || remainder.trim() == "." {
        Ok(Literal::new(value))
    } else {
        Err(RdfParseError::Syntax(RdfSyntaxError {
            message: format!("Invalid literal syntax: {}", literal_str),
            position: None,
            context: None,
        }))
    }
}

fn unescape_literal(s: &str) -> ParseResult<String> {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('t') => result.push('\t'),
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    if hex.len() != 4 {
                        return Err(RdfParseError::Syntax(RdfSyntaxError {
                            message: "Invalid Unicode escape: expected 4 hex digits".to_string(),
                            position: None,
                            context: None,
                        }));
                    }
                    let code_point = u32::from_str_radix(&hex, 16).map_err(|_| {
                        RdfParseError::Syntax(RdfSyntaxError {
                            message: format!("Invalid hex digits in Unicode escape: {}", hex),
                            position: None,
                            context: None,
                        })
                    })?;
                    let unicode_char = char::from_u32(code_point).ok_or_else(|| {
                        RdfParseError::Syntax(RdfSyntaxError {
                            message: format!("Invalid Unicode code point: {:X}", code_point),
                            position: None,
                            context: None,
                        })
                    })?;
                    result.push(unicode_char);
                }
                Some('U') => {
                    let hex: String = chars.by_ref().take(8).collect();
                    if hex.len() != 8 {
                        return Err(RdfParseError::Syntax(RdfSyntaxError {
                            message: "Invalid Unicode escape: expected 8 hex digits".to_string(),
                            position: None,
                            context: None,
                        }));
                    }
                    let code_point = u32::from_str_radix(&hex, 16).map_err(|_| {
                        RdfParseError::Syntax(RdfSyntaxError {
                            message: format!("Invalid hex digits in Unicode escape: {}", hex),
                            position: None,
                            context: None,
                        })
                    })?;
                    let unicode_char = char::from_u32(code_point).ok_or_else(|| {
                        RdfParseError::Syntax(RdfSyntaxError {
                            message: format!("Invalid Unicode code point: {:X}", code_point),
                            position: None,
                            context: None,
                        })
                    })?;
                    result.push(unicode_char);
                }
                Some(other) => {
                    return Err(RdfParseError::Syntax(RdfSyntaxError {
                        message: format!("Invalid escape sequence: \\{}", other),
                        position: None,
                        context: None,
                    }));
                }
                None => {
                    return Err(RdfParseError::Syntax(RdfSyntaxError {
                        message: "Incomplete escape sequence at end of literal".to_string(),
                        position: None,
                        context: None,
                    }));
                }
            }
        } else {
            result.push(ch);
        }
    }

    Ok(result)
}

/// Convert oxrdf Quad to oxirs Quad (for use with oxttl/oxrdfxml/oxjsonld parsers)
pub(super) fn convert_quad(oxrdf_quad: oxrdf::Quad) -> ParseResult<crate::model::Quad> {
    use crate::model::{BlankNode, GraphName, Object, Predicate, Subject};

    // Convert subject
    let subject = match oxrdf_quad.subject {
        oxrdf::Subject::NamedNode(n) => Subject::NamedNode(
            NamedNode::new(n.as_str())
                .map_err(|e| RdfParseError::invalid_iri(format!("{}: {}", n.as_str(), e)))?,
        ),
        oxrdf::Subject::BlankNode(b) => Subject::BlankNode(
            BlankNode::new(b.as_str())
                .map_err(|e| RdfParseError::InvalidBlankNode(format!("{}: {}", b.as_str(), e)))?,
        ),
    };

    // Convert predicate
    let predicate =
        Predicate::NamedNode(NamedNode::new(oxrdf_quad.predicate.as_str()).map_err(|e| {
            RdfParseError::invalid_iri(format!("{}: {}", oxrdf_quad.predicate.as_str(), e))
        })?);

    // Convert object
    let object = match oxrdf_quad.object {
        oxrdf::Term::NamedNode(n) => Object::NamedNode(
            NamedNode::new(n.as_str())
                .map_err(|e| RdfParseError::invalid_iri(format!("{}: {}", n.as_str(), e)))?,
        ),
        oxrdf::Term::BlankNode(b) => Object::BlankNode(
            BlankNode::new(b.as_str())
                .map_err(|e| RdfParseError::InvalidBlankNode(format!("{}: {}", b.as_str(), e)))?,
        ),
        oxrdf::Term::Literal(l) => {
            let lit = if let Some(lang) = l.language() {
                Literal::new_language_tagged_literal(l.value(), lang)
                    .map_err(|e| RdfParseError::InvalidLanguageTag(format!("{}: {}", lang, e)))?
            } else if l.datatype()
                != oxrdf::NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#string")
            {
                Literal::new_typed_literal(
                    l.value(),
                    NamedNode::new(l.datatype().as_str()).map_err(|e| {
                        RdfParseError::invalid_iri(format!("{}: {}", l.datatype().as_str(), e))
                    })?,
                )
            } else {
                Literal::new(l.value())
            };
            Object::Literal(lit)
        }
    };

    // Convert graph name
    let graph_name = match oxrdf_quad.graph_name {
        oxrdf::GraphName::DefaultGraph => GraphName::DefaultGraph,
        oxrdf::GraphName::NamedNode(n) => GraphName::NamedNode(
            NamedNode::new(n.as_str())
                .map_err(|e| RdfParseError::invalid_iri(format!("{}: {}", n.as_str(), e)))?,
        ),
        oxrdf::GraphName::BlankNode(b) => GraphName::BlankNode(
            BlankNode::new(b.as_str())
                .map_err(|e| RdfParseError::InvalidBlankNode(format!("{}: {}", b.as_str(), e)))?,
        ),
    };

    Ok(crate::model::Quad::new(
        subject, predicate, object, graph_name,
    ))
}
