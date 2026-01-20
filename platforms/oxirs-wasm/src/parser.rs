//! RDF parsers for WASM

use crate::error::{WasmError, WasmResult};
use crate::store::OxiRSStore;
use std::collections::HashMap;

/// Internal triple for parsing
#[derive(Debug, Clone)]
pub(crate) struct ParsedTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Parse Turtle format (simplified)
pub fn parse_turtle(turtle: &str) -> WasmResult<Vec<ParsedTriple>> {
    let mut triples = Vec::new();
    let mut prefixes: HashMap<String, String> = HashMap::new();
    let mut base: Option<String> = None;

    let mut current_subject: Option<String> = None;
    let mut current_predicate: Option<String> = None;

    for line in turtle.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Handle @prefix
        if line.starts_with("@prefix") || line.starts_with("PREFIX") {
            if let Some((prefix, uri)) = parse_prefix_line(line) {
                prefixes.insert(prefix, uri);
            }
            continue;
        }

        // Handle @base
        if line.starts_with("@base") || line.starts_with("BASE") {
            if let Some(b) = parse_base_line(line) {
                base = Some(b);
            }
            continue;
        }

        // Parse triple components
        let tokens = tokenize_turtle_line(line);

        for token in &tokens {
            if token == "." {
                current_subject = None;
                current_predicate = None;
            } else if token == ";" {
                current_predicate = None;
            } else if token == "," {
                // Keep subject and predicate, add object
            } else if token == "a" {
                // rdf:type shorthand
                current_predicate =
                    Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string());
            } else if current_subject.is_none() {
                current_subject = Some(expand_term(token, &prefixes, &base));
            } else if current_predicate.is_none() {
                current_predicate = Some(expand_term(token, &prefixes, &base));
            } else {
                let object = expand_term(token, &prefixes, &base);
                if let (Some(s), Some(p)) = (&current_subject, &current_predicate) {
                    triples.push(ParsedTriple {
                        subject: s.clone(),
                        predicate: p.clone(),
                        object,
                    });
                }
            }
        }
    }

    Ok(triples)
}

/// Parse N-Triples format
pub fn parse_ntriples(ntriples: &str) -> WasmResult<Vec<ParsedTriple>> {
    let mut triples = Vec::new();

    for line in ntriples.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse N-Triple line
        if let Some(triple) = parse_ntriple_line(line) {
            triples.push(triple);
        }
    }

    Ok(triples)
}

/// Parse a single N-Triple line
fn parse_ntriple_line(line: &str) -> Option<ParsedTriple> {
    let mut chars = line.chars().peekable();
    let mut parts: Vec<String> = Vec::new();

    while chars.peek().is_some() {
        skip_whitespace(&mut chars);

        if chars.peek() == Some(&'.') {
            chars.next();
            break;
        }

        if chars.peek() == Some(&'<') {
            // IRI
            chars.next(); // consume '<'
            let mut iri = String::new();
            while let Some(&c) = chars.peek() {
                if c == '>' {
                    chars.next();
                    break;
                }
                iri.push(c);
                chars.next();
            }
            parts.push(iri);
        } else if chars.peek() == Some(&'"') {
            // Literal
            chars.next(); // consume '"'
            let mut literal = String::new();
            let mut escaped = false;
            while let Some(&c) = chars.peek() {
                if escaped {
                    literal.push(c);
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    chars.next();
                    break;
                } else {
                    literal.push(c);
                }
                chars.next();
            }

            // Check for language tag or datatype
            if chars.peek() == Some(&'@') {
                chars.next();
                let mut lang = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || c == '.' {
                        break;
                    }
                    lang.push(c);
                    chars.next();
                }
                literal = format!("\"{}\"@{}", literal, lang);
            } else if chars.peek() == Some(&'^') {
                chars.next();
                if chars.peek() == Some(&'^') {
                    chars.next();
                }
                // Skip datatype IRI parsing for simplicity
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || c == '.' {
                        break;
                    }
                    chars.next();
                }
            } else {
                literal = format!("\"{}\"", literal);
            }

            parts.push(literal);
        } else if chars.peek() == Some(&'_') {
            // Blank node
            let mut bnode = String::from("_:");
            chars.next(); // consume '_'
            chars.next(); // consume ':'
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == '.' {
                    break;
                }
                bnode.push(c);
                chars.next();
            }
            parts.push(bnode);
        } else {
            chars.next();
        }
    }

    if parts.len() >= 3 {
        Some(ParsedTriple {
            subject: parts[0].clone(),
            predicate: parts[1].clone(),
            object: parts[2].clone(),
        })
    } else {
        None
    }
}

fn skip_whitespace<I: Iterator<Item = char>>(chars: &mut std::iter::Peekable<I>) {
    while let Some(&c) = chars.peek() {
        if !c.is_whitespace() {
            break;
        }
        chars.next();
    }
}

/// Parse @prefix line
fn parse_prefix_line(line: &str) -> Option<(String, String)> {
    let line = line
        .trim_start_matches("@prefix")
        .trim_start_matches("PREFIX")
        .trim();

    let parts: Vec<&str> = line.splitn(2, ':').collect();
    if parts.len() != 2 {
        return None;
    }

    let prefix = parts[0].trim().to_string();
    let uri = parts[1]
        .trim()
        .trim_start_matches('<')
        .trim_end_matches('>')
        .trim_end_matches('.')
        .trim()
        .trim_start_matches('<')
        .trim_end_matches('>')
        .to_string();

    Some((prefix, uri))
}

/// Parse @base line
fn parse_base_line(line: &str) -> Option<String> {
    let line = line
        .trim_start_matches("@base")
        .trim_start_matches("BASE")
        .trim()
        .trim_end_matches('.')
        .trim();

    if line.starts_with('<') && line.ends_with('>') {
        Some(line[1..line.len() - 1].to_string())
    } else {
        None
    }
}

/// Tokenize a Turtle line
fn tokenize_turtle_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_iri = false;
    let mut in_literal = false;
    let mut escaped = false;

    for c in line.chars() {
        if escaped {
            current.push(c);
            escaped = false;
            continue;
        }

        if c == '\\' {
            escaped = true;
            current.push(c);
            continue;
        }

        if c == '<' && !in_literal {
            in_iri = true;
            current.push(c);
        } else if c == '>' && in_iri {
            current.push(c);
            in_iri = false;
            tokens.push(current.clone());
            current.clear();
        } else if c == '"' && !in_iri {
            if in_literal {
                current.push(c);
                in_literal = false;
            } else {
                in_literal = true;
                current.push(c);
            }
        } else if c.is_whitespace() && !in_iri && !in_literal {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
        } else if (c == '.' || c == ';' || c == ',') && !in_iri && !in_literal {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            tokens.push(c.to_string());
        } else {
            current.push(c);
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Expand a term using prefixes and base
fn expand_term(term: &str, prefixes: &HashMap<String, String>, base: &Option<String>) -> String {
    // Full IRI
    if term.starts_with('<') && term.ends_with('>') {
        return term[1..term.len() - 1].to_string();
    }

    // Prefixed name
    if let Some(idx) = term.find(':') {
        let prefix = &term[..idx];
        let local = &term[idx + 1..];

        if let Some(ns) = prefixes.get(prefix) {
            return format!("{}{}", ns, local);
        }
    }

    // Relative IRI
    if let Some(b) = base {
        if !term.contains(':') {
            return format!("{}{}", b, term);
        }
    }

    term.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ntriples() {
        let nt = r#"
<http://example.org/s> <http://example.org/p> <http://example.org/o> .
<http://example.org/s> <http://example.org/name> "Alice" .
"#;

        let triples = parse_ntriples(nt).unwrap();
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_parse_turtle() {
        let ttl = r#"
@prefix : <http://example.org/> .
:alice :knows :bob .
:bob :name "Bob" .
"#;

        let triples = parse_turtle(ttl).unwrap();
        assert!(triples.len() >= 2);
    }
}
