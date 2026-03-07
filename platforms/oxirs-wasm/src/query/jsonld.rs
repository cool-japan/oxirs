//! JSON-LD output format for OxiRS WASM
//!
//! Converts an RDF graph into JSON-LD compact form.
//!
//! Supports:
//! - `@context` with namespace prefix bindings
//! - `@id` for subject IRIs
//! - `@type` for `rdf:type` predicate
//! - `@language` for lang-tagged literals
//! - `@value` / `@type` for typed literals
//! - Grouping: multiple triples with the same subject are merged into one JSON object
//! - Array values when a property has multiple objects

use crate::store::{InternalTriple, OxiRSStore};
use std::collections::HashMap;

/// The rdf:type IRI
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

/// Standard namespace prefixes included in every @context
static DEFAULT_PREFIXES: &[(&str, &str)] = &[
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("schema", "https://schema.org/"),
    ("foaf", "http://xmlns.com/foaf/0.1/"),
    ("dc", "http://purl.org/dc/elements/1.1/"),
    ("dcterms", "http://purl.org/dc/terms/"),
];

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

/// Serialize all triples in the store to a JSON-LD document string.
///
/// The output is a compact JSON-LD document with:
/// - `@context` containing standard namespace prefix bindings
/// - `@graph` array of subject-grouped RDF objects
///
/// # Example output
/// ```json
/// {
///   "@context": { "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", ... },
///   "@graph": [
///     {
///       "@id": "http://example.org/alice",
///       "http://example.org/name": [{"@value": "Alice"}],
///       "http://example.org/knows": [{"@id": "http://example.org/bob"}]
///     }
///   ]
/// }
/// ```
pub fn serialize_jsonld(store: &OxiRSStore) -> String {
    let triples: Vec<&InternalTriple> = store.all_triples().collect();
    serialize_triples_jsonld(&triples, &[])
}

/// Serialize with additional custom prefix bindings.
///
/// `extra_prefixes` is a slice of `("prefix", "IRI")` pairs.
pub fn serialize_jsonld_with_prefixes(
    store: &OxiRSStore,
    extra_prefixes: &[(&str, &str)],
) -> String {
    let triples: Vec<&InternalTriple> = store.all_triples().collect();
    serialize_triples_jsonld(&triples, extra_prefixes)
}

/// Core serialization: given a list of triples, produce JSON-LD text.
pub(crate) fn serialize_triples_jsonld(
    triples: &[&InternalTriple],
    extra_prefixes: &[(&str, &str)],
) -> String {
    // Build prefix map
    let mut prefixes: HashMap<String, String> = DEFAULT_PREFIXES
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    for (k, v) in extra_prefixes {
        prefixes.insert(k.to_string(), v.to_string());
    }

    // Group triples by subject
    let mut subject_map: HashMap<String, Vec<&InternalTriple>> = HashMap::new();
    let mut subject_order: Vec<String> = Vec::new();
    for triple in triples {
        let entry = subject_map
            .entry(triple.subject.clone())
            .or_insert_with(|| {
                subject_order.push(triple.subject.clone());
                Vec::new()
            });
        entry.push(triple);
    }

    // Build @context
    let context_str = build_context(&prefixes);

    // Build @graph
    let mut graph_items: Vec<String> = Vec::new();
    for subject in &subject_order {
        if let Some(group) = subject_map.get(subject) {
            let item = build_subject_object(subject, group, &prefixes);
            graph_items.push(item);
        }
    }

    if graph_items.is_empty() {
        format!("{{\n  \"@context\": {context_str},\n  \"@graph\": []\n}}")
    } else {
        let graph_str = graph_items.join(",\n    ");
        format!("{{\n  \"@context\": {context_str},\n  \"@graph\": [\n    {graph_str}\n  ]\n}}")
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Build the `@context` JSON object string
fn build_context(prefixes: &HashMap<String, String>) -> String {
    let mut pairs: Vec<String> = prefixes
        .iter()
        .map(|(k, v)| format!("    \"{k}\": \"{v}\""))
        .collect();
    pairs.sort(); // deterministic output
    format!("{{\n{}\n  }}", pairs.join(",\n"))
}

/// Build a single JSON-LD object for a subject and its predicate-object pairs
fn build_subject_object(
    subject: &str,
    triples: &[&InternalTriple],
    prefixes: &HashMap<String, String>,
) -> String {
    let id = compact_iri(subject, prefixes);
    let mut props: HashMap<String, Vec<String>> = HashMap::new();
    let mut prop_order: Vec<String> = Vec::new();

    for triple in triples {
        let pred_compact = compact_iri(&triple.predicate, prefixes);
        let obj_json = object_to_json(&triple.object, prefixes);

        let entry = props.entry(pred_compact.clone()).or_insert_with(|| {
            prop_order.push(pred_compact);
            Vec::new()
        });
        entry.push(obj_json);
    }

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("\"@id\": \"{id}\""));

    for key in &prop_order {
        if let Some(values) = props.get(key) {
            // rdf:type uses @type shorthand
            let json_key = if key == "rdf:type" || key == "a" || key == RDF_TYPE {
                "@type".to_string()
            } else {
                key.clone()
            };

            if values.len() == 1 {
                lines.push(format!("\"{json_key}\": {}", values[0]));
            } else {
                let arr = values.join(", ");
                lines.push(format!("\"{json_key}\": [{arr}]"));
            }
        }
    }

    let inner = lines.join(",\n      ");
    format!("{{\n      {inner}\n    }}")
}

/// Convert an RDF term to a compact IRI using prefix bindings
pub(crate) fn compact_iri(iri: &str, prefixes: &HashMap<String, String>) -> String {
    // Try each prefix
    for (prefix, base) in prefixes {
        if let Some(local) = iri.strip_prefix(base.as_str()) {
            if !local.is_empty() && !local.contains('/') && !local.contains('#') {
                return format!("{prefix}:{local}");
            }
        }
    }
    // No prefix matched — return as-is
    iri.to_string()
}

/// Convert an RDF object term to JSON-LD value notation
pub(crate) fn object_to_json(term: &str, prefixes: &HashMap<String, String>) -> String {
    if term.starts_with('"') {
        // Literal: "value", "value"@lang, or "value"^^<datatype>
        parse_literal_to_json(term)
    } else if let Some(id) = term.strip_prefix("_:") {
        // Blank node
        format!("{{\"@id\": \"_:{id}\"}}")
    } else {
        // IRI
        let compact = compact_iri(term, prefixes);
        format!("{{\"@id\": \"{compact}\"}}")
    }
}

/// Parse a literal RDF term and produce JSON-LD value notation
fn parse_literal_to_json(term: &str) -> String {
    // Find the closing quote
    let chars: Vec<char> = term.chars().collect();
    let mut pos = 1usize;
    while pos < chars.len() && chars[pos] != '"' {
        if chars[pos] == '\\' {
            pos += 1; // skip escape char
        }
        pos += 1;
    }

    let value: String = chars[1..pos].iter().collect();
    let value_escaped = escape_json_string(&value);

    if pos + 1 >= chars.len() {
        // Plain literal with no annotation
        return format!("{{\"@value\": \"{value_escaped}\"}}");
    }

    let rest: String = chars[pos + 1..].iter().collect();

    if let Some(lang) = rest.strip_prefix('@') {
        // Language-tagged literal
        format!("{{\"@value\": \"{value_escaped}\", \"@language\": \"{lang}\"}}")
    } else if let Some(dt_raw) = rest.strip_prefix("^^") {
        // Typed literal
        let datatype = if dt_raw.starts_with('<') && dt_raw.ends_with('>') {
            dt_raw[1..dt_raw.len() - 1].to_string()
        } else {
            dt_raw.to_string()
        };
        // Use compact datatype IRI for well-known types
        let short_dt = compact_xsd_type(&datatype);
        format!("{{\"@value\": \"{value_escaped}\", \"@type\": \"{short_dt}\"}}")
    } else {
        format!("{{\"@value\": \"{value_escaped}\"}}")
    }
}

/// Compact XSD datatypes to short form
fn compact_xsd_type(datatype: &str) -> String {
    let xsd = "http://www.w3.org/2001/XMLSchema#";
    if let Some(local) = datatype.strip_prefix(xsd) {
        format!("xsd:{local}")
    } else {
        datatype.to_string()
    }
}

/// Escape special JSON characters in a string
fn escape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c => result.push(c),
        }
    }
    result
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::OxiRSStore;

    fn make_store() -> OxiRSStore {
        let mut store = OxiRSStore::new();
        store.insert(
            "http://example.org/alice",
            "http://example.org/name",
            "\"Alice\"",
        );
        store.insert(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        );
        store.insert(
            "http://example.org/bob",
            "http://example.org/name",
            "\"Bob\"",
        );
        store
    }

    #[test]
    fn test_jsonld_contains_context() {
        let store = make_store();
        let output = serialize_jsonld(&store);
        assert!(output.contains("\"@context\""));
        assert!(output.contains("\"rdf\""));
    }

    #[test]
    fn test_jsonld_contains_graph() {
        let store = make_store();
        let output = serialize_jsonld(&store);
        assert!(output.contains("\"@graph\""));
    }

    #[test]
    fn test_jsonld_contains_subject_ids() {
        let store = make_store();
        let output = serialize_jsonld(&store);
        assert!(output.contains("alice"));
        assert!(output.contains("bob"));
    }

    #[test]
    fn test_jsonld_literal_value() {
        let store = make_store();
        let output = serialize_jsonld(&store);
        assert!(output.contains("\"@value\""));
        assert!(output.contains("Alice"));
    }

    #[test]
    fn test_jsonld_iri_object() {
        let store = make_store();
        let output = serialize_jsonld(&store);
        assert!(output.contains("\"@id\""));
    }

    #[test]
    fn test_jsonld_lang_literal() {
        let mut store = OxiRSStore::new();
        store.insert(
            "http://example.org/s",
            "http://example.org/p",
            "\"hello\"@en",
        );
        let output = serialize_jsonld(&store);
        assert!(output.contains("\"@language\""));
        assert!(output.contains("\"en\""));
    }

    #[test]
    fn test_jsonld_typed_literal() {
        let mut store = OxiRSStore::new();
        store.insert(
            "http://example.org/s",
            "http://example.org/age",
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>",
        );
        let output = serialize_jsonld(&store);
        assert!(output.contains("\"@type\""));
        assert!(output.contains("xsd:integer"));
    }

    #[test]
    fn test_jsonld_empty_store() {
        let store = OxiRSStore::new();
        let output = serialize_jsonld(&store);
        assert!(output.contains("\"@graph\""));
        assert!(output.contains("[]"));
    }

    #[test]
    fn test_jsonld_with_rdf_type() {
        let mut store = OxiRSStore::new();
        store.insert(
            "http://example.org/alice",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://xmlns.com/foaf/0.1/Person",
        );
        let output = serialize_jsonld(&store);
        assert!(output.contains("@type") || output.contains("rdf:type"));
    }

    #[test]
    fn test_jsonld_blank_node() {
        let mut store = OxiRSStore::new();
        store.insert("http://example.org/s", "http://example.org/p", "_:b0");
        let output = serialize_jsonld(&store);
        assert!(output.contains("_:b0") || output.contains("b0"));
    }

    #[test]
    fn test_jsonld_with_custom_prefixes() {
        let mut store = OxiRSStore::new();
        store.insert(
            "http://example.org/alice",
            "http://example.org/name",
            "\"Alice\"",
        );
        let output = serialize_jsonld_with_prefixes(&store, &[("ex", "http://example.org/")]);
        assert!(output.contains("\"ex\""));
    }

    #[test]
    fn test_compact_iri_standard_prefix() {
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());
        let compact = compact_iri("http://example.org/alice", &prefixes);
        assert_eq!(compact, "ex:alice");
    }

    #[test]
    fn test_compact_iri_no_match() {
        let prefixes = HashMap::new();
        let compact = compact_iri("http://unknown.org/foo", &prefixes);
        assert_eq!(compact, "http://unknown.org/foo");
    }

    #[test]
    fn test_escape_json_string() {
        let s = "Hello \"World\"\nTab\there";
        let escaped = escape_json_string(s);
        assert!(escaped.contains("\\\""));
        assert!(escaped.contains("\\n"));
        assert!(escaped.contains("\\t"));
    }

    #[test]
    fn test_jsonld_valid_json_structure() {
        let store = make_store();
        let output = serialize_jsonld(&store);
        // Basic JSON structure checks
        assert!(output.starts_with('{'));
        assert!(output.ends_with('}'));
        assert!(output.contains("@context"));
        assert!(output.contains("@graph"));
    }

    #[test]
    fn test_jsonld_multiple_objects_same_predicate() {
        let mut store = OxiRSStore::new();
        store.insert(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        );
        store.insert(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/carol",
        );
        let output = serialize_jsonld(&store);
        // Both bob and carol should appear
        assert!(output.contains("bob"));
        assert!(output.contains("carol"));
    }
}
