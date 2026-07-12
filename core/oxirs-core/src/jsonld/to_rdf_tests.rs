//! Tests for the JSON-LD to RDF conversion helpers.

#![cfg(test)]

use super::to_rdf::JsonLdParser;
use super::to_rdf_converter::{canonicalize_json_number, RdfJsonNumber};
use crate::model::RdfTerm;
use crate::vocab::rdf;

#[test]
fn test_canonicalize_json_number() {
    assert_eq!(
        canonicalize_json_number("12", false),
        Some(RdfJsonNumber::Integer("12".into()))
    );
    assert_eq!(
        canonicalize_json_number("-12", false),
        Some(RdfJsonNumber::Integer("-12".into()))
    );
    assert_eq!(
        canonicalize_json_number("1", true),
        Some(RdfJsonNumber::Double("1.0E0".into()))
    );
    assert_eq!(
        canonicalize_json_number("1", true),
        Some(RdfJsonNumber::Double("1.0E0".into()))
    );
    assert_eq!(
        canonicalize_json_number("+1", true),
        Some(RdfJsonNumber::Double("1.0E0".into()))
    );
    assert_eq!(
        canonicalize_json_number("-1", true),
        Some(RdfJsonNumber::Double("-1.0E0".into()))
    );
    assert_eq!(
        canonicalize_json_number("12", true),
        Some(RdfJsonNumber::Double("1.2E1".into()))
    );
    assert_eq!(
        canonicalize_json_number("-12", true),
        Some(RdfJsonNumber::Double("-1.2E1".into()))
    );
    assert_eq!(
        canonicalize_json_number("12.3456E3", false),
        Some(RdfJsonNumber::Double("1.23456E4".into()))
    );
    assert_eq!(
        canonicalize_json_number("12.3456e3", false),
        Some(RdfJsonNumber::Double("1.23456E4".into()))
    );
    assert_eq!(
        canonicalize_json_number("-12.3456E3", false),
        Some(RdfJsonNumber::Double("-1.23456E4".into()))
    );
    assert_eq!(
        canonicalize_json_number("12.34E-3", false),
        Some(RdfJsonNumber::Double("1.234E-2".into()))
    );
    assert_eq!(
        canonicalize_json_number("12.340E-3", false),
        Some(RdfJsonNumber::Double("1.234E-2".into()))
    );
    assert_eq!(
        canonicalize_json_number("0.01234E-1", false),
        Some(RdfJsonNumber::Double("1.234E-3".into()))
    );
    assert_eq!(
        canonicalize_json_number("1.0", false),
        Some(RdfJsonNumber::Integer("1".into()))
    );
    assert_eq!(
        canonicalize_json_number("1.0E0", false),
        Some(RdfJsonNumber::Integer("1".into()))
    );
    assert_eq!(
        canonicalize_json_number("0.01E2", false),
        Some(RdfJsonNumber::Integer("1".into()))
    );
    assert_eq!(
        canonicalize_json_number("1E2", false),
        Some(RdfJsonNumber::Integer("100".into()))
    );
    assert_eq!(
        canonicalize_json_number("1E21", false),
        Some(RdfJsonNumber::Double("1.0E21".into()))
    );
    assert_eq!(
        canonicalize_json_number("0", false),
        Some(RdfJsonNumber::Integer("0".into()))
    );
    assert_eq!(
        canonicalize_json_number("0", true),
        Some(RdfJsonNumber::Double("0.0E0".into()))
    );
    assert_eq!(
        canonicalize_json_number("-0", true),
        Some(RdfJsonNumber::Double("-0.0E0".into()))
    );
    assert_eq!(
        canonicalize_json_number("0E-10", true),
        Some(RdfJsonNumber::Double("0.0E0".into()))
    );
}

/// Helper: parse a JSON-LD document slice and collect all quads, panicking on errors.
fn parse_jsonld(jsonld: &str) -> Vec<crate::model::Quad> {
    JsonLdParser::new()
        .for_slice(jsonld.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .expect("JSON-LD parse must succeed")
}

/// Verify that `@set` container produces flat quads (no rdf:first/rdf:rest/rdf:nil).
///
/// A `@set` is semantically an unordered set whose members are emitted as
/// individual triples sharing the same subject+predicate — identical to the
/// behaviour of a plain JSON array.  It must NOT produce the linked-list
/// structure that `@list` does.
#[test]
fn test_set_container_emits_flat_triples_not_list() {
    let jsonld = r#"[{
      "@id": "http://example.org/s",
      "http://example.org/p": {
        "@set": [
          {"@value": "a"},
          {"@value": "b"},
          {"@value": "c"}
        ]
      }
    }]"#;

    let quads = parse_jsonld(jsonld);

    // Must have exactly three quads, one per member.
    assert_eq!(
        quads.len(),
        3,
        "Expected 3 flat quads for @set with 3 members, got {}: {quads:#?}",
        quads.len()
    );

    // None of the predicates must be rdf:first, rdf:rest, or rdf:nil.
    let rdf_list_predicates = [rdf::FIRST.as_str(), rdf::REST.as_str(), rdf::NIL.as_str()];
    for quad in &quads {
        let pred = quad.predicate().as_str();
        assert!(
            !rdf_list_predicates.contains(&pred),
            "@set must not produce list predicates but found '{pred}' in {quad}"
        );
    }
}

/// Verify that multiple literal values in a `@set` all share the same subject and predicate.
#[test]
fn test_set_container_shared_subject_and_predicate() {
    let jsonld = r#"[{
      "@id": "http://example.org/subject",
      "http://example.org/knows": {
        "@set": [
          {"@id": "http://example.org/alice"},
          {"@id": "http://example.org/bob"}
        ]
      }
    }]"#;

    let quads = parse_jsonld(jsonld);

    assert_eq!(
        quads.len(),
        2,
        "Expected 2 quads for @set with 2 object IRIs, got {}: {quads:#?}",
        quads.len()
    );

    for quad in &quads {
        assert_eq!(
            quad.subject().as_str(),
            "http://example.org/subject",
            "All @set quads must share subject"
        );
        assert_eq!(
            quad.predicate().as_str(),
            "http://example.org/knows",
            "All @set quads must share predicate"
        );
    }

    // Objects must be the two IRIs.
    let objects: Vec<&str> = quads.iter().map(|q| q.object().as_str()).collect();
    assert!(
        objects.contains(&"http://example.org/alice"),
        "alice must appear in objects"
    );
    assert!(
        objects.contains(&"http://example.org/bob"),
        "bob must appear in objects"
    );
}

/// Verify that an empty `@set` produces no quads at all.
#[test]
fn test_set_container_empty_produces_no_quads() {
    let jsonld = r#"[{
      "@id": "http://example.org/s",
      "http://example.org/p": {
        "@set": []
      }
    }]"#;

    let quads = parse_jsonld(jsonld);
    assert!(
        quads.is_empty(),
        "Empty @set must produce zero quads, got: {quads:#?}"
    );
}

/// Verify that `@list` still produces the rdf:first/rdf:rest/rdf:nil linked-list structure,
/// ensuring the @set fix did not accidentally break @list handling.
#[test]
fn test_list_container_still_uses_rdf_list_structure() {
    let jsonld = r#"[{
      "@id": "http://example.org/s",
      "http://example.org/p": {
        "@list": [
          {"@value": "x"},
          {"@value": "y"}
        ]
      }
    }]"#;

    let quads = parse_jsonld(jsonld);

    // A two-element rdf:List produces:
    //   s p _:b0 .
    //   _:b0 rdf:first "x" .
    //   _:b0 rdf:rest _:b1 .
    //   _:b1 rdf:first "y" .
    //   _:b1 rdf:rest rdf:nil .
    // That's exactly 5 quads.
    assert_eq!(
        quads.len(),
        5,
        "Expected 5 quads for @list with 2 elements (rdf:List structure), got {}: {quads:#?}",
        quads.len()
    );

    let predicates: Vec<&str> = quads.iter().map(|q| q.predicate().as_str()).collect();
    assert!(
        predicates.contains(&rdf::FIRST.as_str()),
        "rdf:first must appear in @list quads"
    );
    assert!(
        predicates.contains(&rdf::REST.as_str()),
        "rdf:rest must appear in @list quads"
    );
}
