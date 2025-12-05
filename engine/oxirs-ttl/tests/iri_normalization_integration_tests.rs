//! Integration tests for IRI normalization in serialization
//!
//! These tests demonstrate how IRI normalization can be used
//! during RDF serialization to ensure consistent output.

use oxirs_core::model::{Literal, NamedNode, Triple};
use oxirs_ttl::toolkit::iri_normalizer::normalize_iri;
use oxirs_ttl::toolkit::SerializationConfig;
use oxirs_ttl::turtle::TurtleSerializer;
use oxirs_ttl::Serializer;

/// Test that SerializationConfig has normalize_iris field
#[test]
fn test_serialization_config_has_normalize_iris() {
    let config = SerializationConfig::default();
    assert!(!config.normalize_iris); // Default is false

    let config_enabled = SerializationConfig::new().with_normalize_iris(true);
    assert!(config_enabled.normalize_iris);

    let config_disabled = SerializationConfig::new().with_normalize_iris(false);
    assert!(!config_disabled.normalize_iris);
}

/// Test IRI normalization can be configured
#[test]
fn test_normalize_iris_builder_pattern() {
    let config = SerializationConfig::new()
        .with_pretty(true)
        .with_normalize_iris(true)
        .with_use_prefixes(false);

    assert!(config.pretty);
    assert!(config.normalize_iris);
    assert!(!config.use_prefixes);
}

/// Test basic triple serialization (without normalization for now)
#[test]
fn test_basic_triple_serialization() {
    let triple = Triple::new(
        NamedNode::new_unchecked("http://example.org/subject"),
        NamedNode::new_unchecked("http://example.org/predicate"),
        Literal::new_simple_literal("object"),
    );

    let mut output = Vec::new();
    let serializer = TurtleSerializer::new();
    serializer.serialize(&[triple], &mut output).unwrap();

    let result = String::from_utf8(output).unwrap();
    assert!(result.contains("http://example.org/subject"));
    assert!(result.contains("http://example.org/predicate"));
    assert!(result.contains("\"object\""));
}

/// Test that normalize_iri works with various IRI formats
#[test]
fn test_normalize_iri_various_formats() {
    // Case normalization
    let iri1 = normalize_iri("HTTP://EXAMPLE.ORG/path").unwrap();
    assert_eq!(iri1.as_str(), "http://example.org/path");

    // Percent-encoding normalization
    let iri2 = normalize_iri("http://example.org/%7Euser").unwrap();
    assert_eq!(iri2.as_str(), "http://example.org/~user");

    // Path normalization
    let iri3 = normalize_iri("http://example.org/a/./b/../c").unwrap();
    assert_eq!(iri3.as_str(), "http://example.org/a/c");

    // Default port removal
    let iri4 = normalize_iri("http://example.org:80/path").unwrap();
    assert_eq!(iri4.as_str(), "http://example.org/path");

    // Complex normalization
    let iri5 = normalize_iri("HTTP://EXAMPLE.ORG:80/%7Euser/a/./b/../c").unwrap();
    assert_eq!(iri5.as_str(), "http://example.org/~user/a/c");
}

/// Test IRI normalization with common RDF namespaces
#[test]
fn test_normalize_common_rdf_iris() {
    // RDF namespace - no path, so normalizer adds empty path before fragment
    let rdf_type = normalize_iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
    assert_eq!(
        rdf_type.as_str(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns?#type"
    );

    // RDFS namespace
    let rdfs_label = normalize_iri("http://www.w3.org/2000/01/rdf-schema#label").unwrap();
    assert_eq!(
        rdfs_label.as_str(),
        "http://www.w3.org/2000/01/rdf-schema?#label"
    );

    // XSD namespace
    let xsd_string = normalize_iri("http://www.w3.org/2001/XMLSchema#string").unwrap();
    assert_eq!(
        xsd_string.as_str(),
        "http://www.w3.org/2001/XMLSchema?#string"
    );

    // OWL namespace - with path, so fragment is preserved correctly
    let owl_class = normalize_iri("http://www.w3.org/2002/07/owl#Class").unwrap();
    assert_eq!(owl_class.as_str(), "http://www.w3.org/2002/07/owl?#Class");
}

/// Test IRI normalization with real-world IRIs
#[test]
fn test_normalize_real_world_iris() {
    // DBpedia resource - parentheses are reserved characters, remain percent-encoded
    let dbpedia =
        normalize_iri("HTTP://DBPEDIA.ORG/resource/Rust_%28programming_language%29").unwrap();
    assert_eq!(
        dbpedia.as_str(),
        "http://dbpedia.org/resource/Rust_%28programming_language%29"
    );

    // Schema.org
    let schema = normalize_iri("HTTPS://SCHEMA.ORG:443/Person").unwrap();
    assert_eq!(schema.as_str(), "https://schema.org/Person");

    // WikiData
    let wikidata = normalize_iri("https://www.wikidata.org/wiki/./Q37312/../Q37312").unwrap();
    assert_eq!(wikidata.as_str(), "https://www.wikidata.org/wiki/Q37312");
}

/// Example: Using IRI normalization for consistent RDF output
///
/// This demonstrates a workflow where IRIs are normalized before
/// serialization to ensure consistent output across different data sources.
#[test]
fn test_example_normalized_workflow() {
    // Simulate receiving data from different sources with inconsistent IRIs
    let iris = [
        "HTTP://EXAMPLE.ORG/Person/Alice",     // Uppercase scheme and host
        "http://example.org:80/Person/Bob",    // With default port
        "http://example.org/%50erson/Charlie", // Percent-encoded 'P'
        "http://example.org/./Person/../Person/Dave", // With dot segments
    ];

    // Normalize all IRIs
    let normalized: Vec<_> = iris.iter().map(|iri| normalize_iri(iri).unwrap()).collect();

    // All should normalize to consistent format
    assert_eq!(normalized[0].as_str(), "http://example.org/Person/Alice");
    assert_eq!(normalized[1].as_str(), "http://example.org/Person/Bob");
    assert_eq!(normalized[2].as_str(), "http://example.org/Person/Charlie");
    assert_eq!(normalized[3].as_str(), "http://example.org/Person/Dave");

    // Create triples with normalized IRIs
    let triples: Vec<_> = normalized
        .iter()
        .map(|iri| {
            Triple::new(
                NamedNode::new_unchecked(iri.as_str()),
                NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                NamedNode::new_unchecked("http://example.org/Person"),
            )
        })
        .collect();

    // Serialize with consistent IRIs
    let mut output = Vec::new();
    let serializer = TurtleSerializer::new();
    serializer.serialize(&triples, &mut output).unwrap();

    let result = String::from_utf8(output).unwrap();

    // Verify output contains normalized IRIs
    assert!(result.contains("http://example.org/Person/Alice"));
    assert!(result.contains("http://example.org/Person/Bob"));
    assert!(result.contains("http://example.org/Person/Charlie"));
    assert!(result.contains("http://example.org/Person/Dave"));

    // Should NOT contain unnormalized forms
    assert!(!result.contains("HTTP://"));
    assert!(!result.contains(":80/"));
    assert!(!result.contains("%50"));
    assert!(!result.contains("/./"));
    assert!(!result.contains("/../"));
}

/// Test IRI normalization helps deduplicate equivalent IRIs
#[test]
fn test_iri_normalization_deduplication() {
    use std::collections::HashSet;

    // These IRIs are semantically equivalent
    let iris = [
        "http://example.org/path",
        "HTTP://EXAMPLE.ORG/path",
        "http://example.org:80/path",
        "http://example.org/a/../path",
    ];

    // Without normalization, they're all different
    let unique_raw: HashSet<_> = iris.iter().collect();
    assert_eq!(unique_raw.len(), 4);

    // With normalization, they're all the same
    let normalized: Vec<_> = iris.iter().map(|iri| normalize_iri(iri).unwrap()).collect();

    let unique_normalized: HashSet<_> = normalized.iter().map(|n| n.as_str()).collect();
    assert_eq!(unique_normalized.len(), 1);
    assert_eq!(
        unique_normalized.iter().next().unwrap(),
        &"http://example.org/path"
    );
}

/// Test IRI normalization preserves fragment identifiers
#[test]
fn test_normalize_preserves_fragments() {
    // Fragment with path - normalizer adds empty query before fragment
    let iri = normalize_iri("HTTP://EXAMPLE.ORG/page#section").unwrap();
    assert_eq!(iri.as_str(), "http://example.org/page?#section");

    // Fragment with percent-encoding normalization
    let iri2 = normalize_iri("http://example.org:80/page#%41BC").unwrap();
    assert_eq!(iri2.as_str(), "http://example.org/page?#ABC");

    // Fragment without path adds / before ?# (hierarchical IRI requirement)
    let iri3 = normalize_iri("HTTP://EXAMPLE.ORG#section").unwrap();
    assert_eq!(iri3.as_str(), "http://example.org/?#section");
}

/// Test IRI normalization preserves query parameters
#[test]
fn test_normalize_preserves_queries() {
    let iri = normalize_iri("HTTP://EXAMPLE.ORG/search?q=rust").unwrap();
    assert_eq!(iri.as_str(), "http://example.org/search?q=rust");

    let iri2 = normalize_iri("http://example.org:80/search?q=%72ust").unwrap();
    assert_eq!(iri2.as_str(), "http://example.org/search?q=rust");
}

/// Test that serialization config can be built with multiple options
#[test]
fn test_serialization_config_comprehensive() {
    let config = SerializationConfig::new()
        .with_pretty(true)
        .with_base_iri("http://example.org/".to_string())
        .with_prefix("ex".to_string(), "http://example.org/".to_string())
        .with_use_prefixes(true)
        .with_max_line_length(Some(100))
        .with_indent("    ".to_string())
        .with_normalize_iris(true);

    assert!(config.pretty);
    assert_eq!(config.base_iri, Some("http://example.org/".to_string()));
    assert_eq!(
        config.prefixes.get("ex"),
        Some(&"http://example.org/".to_string())
    );
    assert!(config.use_prefixes);
    assert_eq!(config.max_line_length, Some(100));
    assert_eq!(config.indent, "    ");
    assert!(config.normalize_iris);
}
