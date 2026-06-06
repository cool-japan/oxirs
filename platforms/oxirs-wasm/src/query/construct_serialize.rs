//! Serialization of CONSTRUCT result triples to N-Triples, Turtle, and JSON-LD.

use crate::error::{WasmError, WasmResult};
use crate::Triple;
use std::collections::HashMap;

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
pub(crate) fn serialize_ntriples(triples: &[Triple]) -> WasmResult<String> {
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
pub(crate) fn serialize_turtle(
    triples: &[Triple],
    prefixes: &HashMap<String, String>,
) -> WasmResult<String> {
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
pub(crate) fn abbreviate_term(term: &str, prefixes: &HashMap<String, String>) -> String {
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
pub(crate) fn serialize_construct_jsonld(triples: &[Triple]) -> WasmResult<String> {
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
pub(crate) fn extract_literal_content(literal: &str) -> String {
    let s = literal.trim();
    if let Some(stripped) = s.strip_prefix('"') {
        // Find the closing quote
        if let Some(end_quote) = stripped.find('"') {
            return stripped[..end_quote].to_string();
        }
    }
    s.to_string()
}
