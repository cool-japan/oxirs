#![cfg(test)]

use super::handler::negotiate_vocab_format;
use super::metadata::*;
use super::registry::*;
use super::serializer::*;

#[test]
fn test_register_vocab() {
    let mut r = VocabularyRegistry::new();
    let e = VocabularyEntry::new("foaf", "http://xmlns.com/foaf/0.1/", "FOAF");
    r.register(e);
    assert_eq!(r.len(), 1);
    assert!(!r.is_empty());
}

#[test]
fn test_get_vocab() {
    let mut r = VocabularyRegistry::new();
    r.register(VocabularyEntry::new(
        "dc",
        "http://purl.org/dc/terms/",
        "Dublin Core",
    ));
    assert_eq!(r.get("dc").unwrap().title, "Dublin Core");
}

#[test]
fn test_missing_vocab() {
    let r = VocabularyRegistry::new();
    assert!(r.get("nonexistent").is_none());
}

#[test]
fn test_remove_vocab() {
    let mut r = VocabularyRegistry::new();
    r.register(VocabularyEntry::new(
        "rdf",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "RDF",
    ));
    assert!(r.remove("rdf").is_some());
    assert!(r.is_empty());
}

#[test]
fn test_vocab_with_description() {
    let e = VocabularyEntry::new("a", "http://example.org/a#", "A").with_description("description");
    assert_eq!(e.description, "description");
}

#[test]
fn test_vocab_with_contributors() {
    let e = VocabularyEntry::new("a", "http://example.org/a#", "A")
        .with_contributor("Alice")
        .with_contributor("Bob");
    assert_eq!(e.contributors.len(), 2);
    assert_eq!(e.contributors[0], "Alice");
    assert_eq!(e.contributors[1], "Bob");
}

#[test]
fn test_register_replaces_existing() {
    let mut r = VocabularyRegistry::new();
    r.register(VocabularyEntry::new("a", "ns1", "Title 1"));
    let previous = r.register(VocabularyEntry::new("a", "ns2", "Title 2"));
    assert!(previous.is_some());
    assert_eq!(r.len(), 1);
    assert_eq!(r.get("a").unwrap().namespace, "ns2");
}

#[test]
fn test_registry_all_returns_entries() {
    let mut r = VocabularyRegistry::new();
    r.register(VocabularyEntry::new("a", "ns1", "A"));
    r.register(VocabularyEntry::new("b", "ns2", "B"));
    assert_eq!(r.all().len(), 2);
}

#[test]
fn test_build_metadata() {
    let e = VocabularyEntry::new("a", "http://example.org/a#", "A").with_description("desc");
    let m = build_metadata(&e, 42);
    assert_eq!(m.concept_count, 42);
    assert_eq!(m.title, "A");
}

#[test]
fn test_serialize_html() {
    let m = VocabularyMetadata {
        id: "test".into(),
        namespace: "http://example.org/".into(),
        title: "Test".into(),
        description: "desc".into(),
        contributors: vec![],
        concept_count: 5,
    };
    let html = serialize_metadata(&m, VocabFormat::Html);
    assert!(html.contains("Test"));
    assert!(html.contains("5"));
}

#[test]
fn test_serialize_html_escapes_input() {
    let m = VocabularyMetadata {
        id: "test".into(),
        namespace: "http://example.org/".into(),
        title: "Title <with> &chars".into(),
        description: "desc".into(),
        contributors: vec![],
        concept_count: 0,
    };
    let html = serialize_metadata(&m, VocabFormat::Html);
    assert!(html.contains("&lt;with&gt;"));
    assert!(html.contains("&amp;chars"));
}

#[test]
fn test_serialize_turtle() {
    let m = VocabularyMetadata {
        id: "test".into(),
        namespace: "http://example.org/".into(),
        title: "Test".into(),
        description: "desc".into(),
        contributors: vec![],
        concept_count: 5,
    };
    let ttl = serialize_metadata(&m, VocabFormat::Turtle);
    assert!(ttl.contains("@prefix dcterms"));
    assert!(ttl.contains("\"Test\""));
}

#[test]
fn test_serialize_turtle_escapes_quotes() {
    let m = VocabularyMetadata {
        id: "test".into(),
        namespace: "http://example.org/".into(),
        title: "She said \"hello\"".into(),
        description: "desc".into(),
        contributors: vec![],
        concept_count: 0,
    };
    let ttl = serialize_metadata(&m, VocabFormat::Turtle);
    assert!(ttl.contains("She said \\\"hello\\\""));
}

#[test]
fn test_serialize_jsonld() {
    let m = VocabularyMetadata {
        id: "test".into(),
        namespace: "http://example.org/".into(),
        title: "Test".into(),
        description: "desc".into(),
        contributors: vec![],
        concept_count: 5,
    };
    let jsonld = serialize_metadata(&m, VocabFormat::JsonLd);
    assert!(jsonld.contains("Test"));
    assert!(jsonld.contains("5"));
}

#[test]
fn test_mime_types() {
    assert_eq!(VocabFormat::Html.mime_type(), "text/html");
    assert_eq!(VocabFormat::JsonLd.mime_type(), "application/ld+json");
    assert_eq!(VocabFormat::Turtle.mime_type(), "text/turtle");
}

#[test]
fn test_negotiate_vocab_format_html_default() {
    let headers = axum::http::HeaderMap::new();
    assert_eq!(negotiate_vocab_format(&headers), VocabFormat::Html);
}

#[test]
fn test_negotiate_vocab_format_jsonld() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("accept", "application/ld+json".parse().unwrap());
    assert_eq!(negotiate_vocab_format(&headers), VocabFormat::JsonLd);
}

#[test]
fn test_negotiate_vocab_format_turtle() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("accept", "text/turtle".parse().unwrap());
    assert_eq!(negotiate_vocab_format(&headers), VocabFormat::Turtle);
}
