//! Tests for the JSON-LD to RDF conversion helpers.

#![cfg(test)]

use super::to_rdf_converter::{canonicalize_json_number, RdfJsonNumber};

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
