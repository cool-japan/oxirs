//! Integration tests for PREFIX declaration parsing
//!
//! Tests single-line and multi-line PREFIX support

use oxirs_core::sparql::extract_and_expand_prefixes;

#[test]
fn test_single_line_prefix() {
    let query =
        "PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?name WHERE { ?person foaf:name ?name }";
    let (prefixes, expanded) = extract_and_expand_prefixes(query).unwrap();

    assert_eq!(prefixes.len(), 1);
    assert_eq!(
        prefixes.get("foaf"),
        Some(&"http://xmlns.com/foaf/0.1/".to_string())
    );
    assert!(expanded.contains("<http://xmlns.com/foaf/0.1/name>"));
    assert!(!expanded.contains("PREFIX"));
}

#[test]
fn test_multi_line_prefix() {
    let query = r#"PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?person WHERE {
  ?person rdf:type foaf:Person .
}"#;

    let (prefixes, expanded) = extract_and_expand_prefixes(query).unwrap();

    assert_eq!(prefixes.len(), 2);
    assert_eq!(
        prefixes.get("foaf"),
        Some(&"http://xmlns.com/foaf/0.1/".to_string())
    );
    assert_eq!(
        prefixes.get("rdf"),
        Some(&"http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string())
    );
    assert!(expanded.contains("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"));
    assert!(expanded.contains("<http://xmlns.com/foaf/0.1/Person>"));
    assert!(!expanded.contains("PREFIX"));
}

#[test]
fn test_mixed_case_prefix() {
    let query =
        "prefix foaf: <http://xmlns.com/foaf/0.1/> SELECT ?name WHERE { ?person foaf:name ?name }";
    let (prefixes, expanded) = extract_and_expand_prefixes(query).unwrap();

    assert_eq!(prefixes.len(), 1);
    assert!(expanded.contains("<http://xmlns.com/foaf/0.1/name>"));
}

#[test]
fn test_no_prefix() {
    let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
    let (prefixes, expanded) = extract_and_expand_prefixes(query).unwrap();

    assert_eq!(prefixes.len(), 0);
    assert_eq!(expanded, query);
}

#[test]
fn test_prefix_with_hash() {
    let query = "PREFIX ex: <http://example.org/#> SELECT ?name WHERE { ?person ex:name ?name }";
    let (prefixes, expanded) = extract_and_expand_prefixes(query).unwrap();

    assert_eq!(prefixes.len(), 1);
    assert!(expanded.contains("<http://example.org/#name>"));
}

#[test]
fn test_prefix_with_slash() {
    let query = "PREFIX ex: <http://example.org/> SELECT ?name WHERE { ?person ex:name ?name }";
    let (prefixes, expanded) = extract_and_expand_prefixes(query).unwrap();

    assert_eq!(prefixes.len(), 1);
    assert!(expanded.contains("<http://example.org/name>"));
}

#[test]
fn test_multiple_prefixes_same_line() {
    let query = "PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX ex: <http://example.org/> SELECT ?name WHERE { ?person foaf:name ?name }";
    let (prefixes, _expanded) = extract_and_expand_prefixes(query).unwrap();

    assert_eq!(prefixes.len(), 2);
    assert_eq!(
        prefixes.get("foaf"),
        Some(&"http://xmlns.com/foaf/0.1/".to_string())
    );
    assert_eq!(prefixes.get("ex"), Some(&"http://example.org/".to_string()));
}
