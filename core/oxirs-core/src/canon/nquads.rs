//! N-Quads serialization for URDNA2015 canonical form.
//!
//! Implements the canonical N-Quads serialization as described in
//! https://www.w3.org/TR/n-quads/ and used by URDNA2015
//! (https://www.w3.org/TR/rdf-canon/#serialize-canonical-form).
//!
//! Key rules for canonical N-Quads:
//! - Blank nodes are replaced by their canonical identifier (e.g. `_:c14n0`)
//! - Literals use the canonical lexical form with explicit datatype IRI
//! - One quad per line, terminated by ` .\n`
//! - The complete document is the sorted concatenation of all canonical quad lines

use std::collections::HashMap;

use super::types::{QuadTerm, RdfQuad};

/// Escape special characters in a string literal value per N-Quads spec.
///
/// Escapes: `\\`, `"`, `\n`, `\r`, `\t`, and control characters as `\uXXXX`.
pub fn escape_literal(s: &str) -> String {
    let mut buf = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '\\' => buf.push_str("\\\\"),
            '"' => buf.push_str("\\\""),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            c if c.is_control() => {
                // Unicode escape for remaining control chars
                buf.push_str(&format!("\\u{:04X}", c as u32));
            }
            c => buf.push(c),
        }
    }
    buf
}

/// Serialize a single [`QuadTerm`] to its N-Quads representation.
///
/// `mapping` is the canonical blank node identifier mapping (`original → c14nN`).
/// If `mapping` is `None`, blank nodes are serialized as-is (useful for
/// intermediate hash computations using placeholder IDs).
pub fn term_to_nquad(term: &QuadTerm, mapping: &HashMap<String, String>) -> String {
    match term {
        QuadTerm::Iri(iri) => format!("<{}>", iri),

        QuadTerm::BlankNode(id) => {
            // Look up canonical identifier; fall back to the original name prefixed
            // with `_:` so that intermediate placeholders also work correctly.
            if let Some(canonical) = mapping.get(id.as_str()) {
                canonical.clone()
            } else {
                // The blank node is a placeholder string like `_:a` already
                if id.starts_with("_:") {
                    id.clone()
                } else {
                    format!("_:{}", id)
                }
            }
        }

        QuadTerm::Literal {
            value,
            datatype,
            language,
        } => {
            let escaped = escape_literal(value);
            if let Some(lang) = language {
                format!("\"{}\"@{}", escaped, lang)
            } else {
                format!("\"{}\"^^<{}>", escaped, datatype)
            }
        }
    }
}

/// Serialize a single [`RdfQuad`] to its canonical N-Quads line.
///
/// The line format is:
/// ```text
/// <subject> <predicate> <object> [<graph>] .
/// ```
/// The trailing newline is NOT included; the caller is responsible for joining
/// lines with `\n` and appending a final `\n` if required.
pub fn quad_to_nquad(quad: &RdfQuad, mapping: &HashMap<String, String>) -> String {
    let s = term_to_nquad(&quad.subject, mapping);
    let p = term_to_nquad(&quad.predicate, mapping);
    let o = term_to_nquad(&quad.object, mapping);

    if let Some(graph) = &quad.graph {
        let g = term_to_nquad(graph, mapping);
        format!("{} {} {} {} .", s, p, o, g)
    } else {
        format!("{} {} {} .", s, p, o)
    }
}

/// Serialize a single quad using placeholder blank node names.
///
/// Used during the first-degree hash step where:
/// - the blank node under consideration is replaced by `_:a`
/// - all other blank nodes are replaced by `_:z`
pub fn quad_to_nquad_with_placeholders(
    quad: &RdfQuad,
    reference_bnode: &str,
    issued: &HashMap<String, String>,
) -> String {
    let serialize_term = |term: &QuadTerm| -> String {
        match term {
            QuadTerm::Iri(iri) => format!("<{}>", iri),
            QuadTerm::BlankNode(id) => {
                if id == reference_bnode {
                    "_:a".to_string()
                } else if let Some(canonical) = issued.get(id.as_str()) {
                    canonical.clone()
                } else {
                    "_:z".to_string()
                }
            }
            QuadTerm::Literal {
                value,
                datatype,
                language,
            } => {
                let escaped = escape_literal(value);
                if let Some(lang) = language {
                    format!("\"{}\"@{}", escaped, lang)
                } else {
                    format!("\"{}\"^^<{}>", escaped, datatype)
                }
            }
        }
    };

    let s = serialize_term(&quad.subject);
    let p = serialize_term(&quad.predicate);
    let o = serialize_term(&quad.object);

    if let Some(graph) = &quad.graph {
        let g = serialize_term(graph);
        format!("{} {} {} {} .", s, p, o, g)
    } else {
        format!("{} {} {} .", s, p, o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canon::types::{QuadTerm, RdfQuad};

    fn empty_map() -> HashMap<String, String> {
        HashMap::new()
    }

    #[test]
    fn test_escape_plain_string() {
        assert_eq!(escape_literal("hello"), "hello");
    }

    #[test]
    fn test_escape_special_chars() {
        assert_eq!(escape_literal("a\nb"), "a\\nb");
        assert_eq!(escape_literal("a\rb"), "a\\rb");
        assert_eq!(escape_literal("a\tb"), "a\\tb");
        assert_eq!(escape_literal("a\\b"), "a\\\\b");
        assert_eq!(escape_literal("say \"hi\""), "say \\\"hi\\\"");
    }

    #[test]
    fn test_iri_term() {
        let t = QuadTerm::Iri("http://example.org/s".to_string());
        assert_eq!(term_to_nquad(&t, &empty_map()), "<http://example.org/s>");
    }

    #[test]
    fn test_literal_with_datatype() {
        let t = QuadTerm::Literal {
            value: "42".to_string(),
            datatype: "http://www.w3.org/2001/XMLSchema#integer".to_string(),
            language: None,
        };
        assert_eq!(
            term_to_nquad(&t, &empty_map()),
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"
        );
    }

    #[test]
    fn test_literal_with_language() {
        let t = QuadTerm::Literal {
            value: "Hello".to_string(),
            datatype: "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".to_string(),
            language: Some("en".to_string()),
        };
        assert_eq!(term_to_nquad(&t, &empty_map()), "\"Hello\"@en");
    }

    #[test]
    fn test_blank_node_mapped() {
        let t = QuadTerm::BlankNode("b0".to_string());
        let mut map = HashMap::new();
        map.insert("b0".to_string(), "_:c14n0".to_string());
        assert_eq!(term_to_nquad(&t, &map), "_:c14n0");
    }

    #[test]
    fn test_quad_to_nquad_default_graph() {
        let quad = RdfQuad {
            subject: QuadTerm::Iri("http://example.org/s".to_string()),
            predicate: QuadTerm::Iri("http://example.org/p".to_string()),
            object: QuadTerm::Iri("http://example.org/o".to_string()),
            graph: None,
        };
        let line = quad_to_nquad(&quad, &empty_map());
        assert_eq!(
            line,
            "<http://example.org/s> <http://example.org/p> <http://example.org/o> ."
        );
    }

    #[test]
    fn test_quad_to_nquad_named_graph() {
        let quad = RdfQuad {
            subject: QuadTerm::Iri("http://s".to_string()),
            predicate: QuadTerm::Iri("http://p".to_string()),
            object: QuadTerm::Iri("http://o".to_string()),
            graph: Some(QuadTerm::Iri("http://g".to_string())),
        };
        let line = quad_to_nquad(&quad, &empty_map());
        assert_eq!(line, "<http://s> <http://p> <http://o> <http://g> .");
    }
}
