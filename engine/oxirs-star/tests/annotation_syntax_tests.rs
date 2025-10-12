//! Tests for RDF-star annotation syntax ({| |})
//!
//! This module tests the W3C RDF-star annotation syntax feature.
//! Annotation syntax allows attaching metadata to triples in a concise way:
//!
//! ```turtle
//! :alice :age 30 {| :certainty 0.9; :source :survey |} .
//! ```
//!
//! This expands to multiple triples:
//! ```turtle
//! <<:alice :age 30>> :certainty 0.9 .
//! <<:alice :age 30>> :source :survey .
//! ```

use oxirs_star::parser::{StarFormat, StarParser};

#[test]
fn test_annotation_syntax_single_property() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:age "30" {| ex:certainty "0.9" |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse annotation syntax");

    // Should have 1 triple: the annotation triple
    // The base triple is NOT asserted (referential opacity)
    assert_eq!(graph.len(), 1, "Should have exactly 1 triple");

    // Check that the triple is an annotation (subject is quoted triple)
    let triples = graph.triples();
    assert_eq!(triples.len(), 1);
    let annotation_triple = &triples[0];

    assert!(
        annotation_triple.subject.is_quoted_triple(),
        "Subject should be a quoted triple"
    );

    // Verify the annotation predicate
    let predicate_iri = annotation_triple
        .predicate
        .as_named_node()
        .expect("Predicate should be a named node")
        .iri
        .as_str();
    assert_eq!(predicate_iri, "http://example.org/certainty");

    // Verify the annotation value
    let object_literal = annotation_triple
        .object
        .as_literal()
        .expect("Object should be a literal");
    assert_eq!(object_literal.value, "0.9");
}

#[test]
fn test_annotation_syntax_multiple_properties() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:age "30" {| ex:certainty "0.9"; ex:source ex:survey2023 |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse annotation syntax");

    // Should have 2 triples: one for each annotation property
    assert_eq!(graph.len(), 2, "Should have 2 annotation triples");

    // Both triples should have the same quoted triple as subject
    let triples = graph.triples();
    assert!(
        triples[0].subject.is_quoted_triple(),
        "First triple subject should be quoted triple"
    );
    assert!(
        triples[1].subject.is_quoted_triple(),
        "Second triple subject should be quoted triple"
    );

    // Verify both annotation predicates
    let predicates: Vec<String> = triples
        .iter()
        .map(|t| {
            t.predicate
                .as_named_node()
                .expect("Predicate should be named node")
                .iri
                .clone()
        })
        .collect();

    assert!(predicates.contains(&"http://example.org/certainty".to_string()));
    assert!(predicates.contains(&"http://example.org/source".to_string()));
}

#[test]
fn test_annotation_syntax_three_properties() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:bob ex:height "180" {|
            ex:certainty "0.95";
            ex:source ex:healthRecord;
            ex:timestamp "2023-10-12T10:00:00Z"
        |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse annotation syntax");

    assert_eq!(graph.len(), 3, "Should have 3 annotation triples");

    // Verify all subjects are the same quoted triple
    let triples = graph.triples();
    let first_subject = &triples[0].subject;
    assert!(first_subject.is_quoted_triple());

    for triple in triples {
        assert_eq!(
            &triple.subject, first_subject,
            "All subjects should be identical"
        );
    }
}

#[test]
fn test_annotation_syntax_with_iri_object() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .
        @prefix prov: <http://www.w3.org/ns/prov#> .

        ex:alice ex:knows ex:bob {| prov:wasDerivedFrom ex:socialNetwork |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse annotation syntax");

    assert_eq!(graph.len(), 1);

    let triple = &graph.triples()[0];
    assert!(triple.subject.is_quoted_triple());
    assert!(
        triple.object.is_named_node(),
        "Annotation object should be IRI"
    );
}

#[test]
fn test_annotation_syntax_empty_annotation_block() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:age "30" {|  |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Empty annotation block should be allowed");

    // Empty annotation block should result in no annotation triples
    assert_eq!(
        graph.len(),
        0,
        "Empty annotation block should produce no triples"
    );
}

#[test]
fn test_annotation_syntax_whitespace_handling() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:age "30"{|ex:certainty "0.9"|}.
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Should handle annotation without spaces");

    assert_eq!(graph.len(), 1);
}

#[test]
fn test_annotation_syntax_with_literals() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        ex:alice ex:salary "50000"^^xsd:integer {|
            ex:confidence "0.95"^^xsd:decimal;
            ex:year "2023"^^xsd:gYear
        |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse annotations with typed literals");

    assert_eq!(graph.len(), 2);

    // Verify typed literals in annotations
    let triples = graph.triples();
    for triple in triples {
        assert!(
            triple.object.is_literal(),
            "Annotation values should be literals"
        );
    }
}

#[test]
fn test_annotation_syntax_with_language_tags() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:name "Alice"@en {| ex:source "Census"@en; ex:verified "yes"@en |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse annotations with language tags");

    assert_eq!(graph.len(), 2);
}

#[test]
fn test_annotation_syntax_multiple_statements() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:age "30" {| ex:certainty "0.9" |} .
        ex:bob ex:age "25" {| ex:certainty "0.85"; ex:source ex:survey |} .
        ex:charlie ex:name "Charlie" {| ex:verified "true" |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse multiple annotated statements");

    // 1 + 2 + 1 = 4 annotation triples total
    assert_eq!(graph.len(), 4);

    // Count unique quoted triples
    let quoted_triples: std::collections::HashSet<_> = graph
        .triples()
        .iter()
        .filter(|t| t.subject.is_quoted_triple())
        .map(|t| &t.subject)
        .collect();

    assert_eq!(
        quoted_triples.len(),
        3,
        "Should have 3 unique quoted triples"
    );
}

#[test]
fn test_annotation_syntax_nested_quoted_triple() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        << ex:alice ex:says << ex:bob ex:age "30" >> >> ex:certainty "0.8" {|
            ex:context ex:conversation;
            ex:timestamp "2023-10-12"
        |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse nested quoted triple with annotation");

    // Should have 3 triples:
    // 1. The base triple (nested quoted triple statement)
    // 2-3. Two annotation triples
    assert_eq!(graph.len(), 3);
}

#[test]
fn test_annotation_syntax_trailing_semicolon() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:age "30" {| ex:certainty "0.9"; |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Trailing semicolon should be handled gracefully");

    // Should parse correctly, ignoring empty statement after semicolon
    assert_eq!(graph.len(), 1);
}

#[test]
fn test_annotation_syntax_error_missing_closing_delimiter() {
    use oxirs_star::StarConfig;

    // Use strict mode to ensure errors are caught
    let parser = StarParser::with_config(StarConfig {
        strict_mode: true,
        ..Default::default()
    });

    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:age "30" {| ex:certainty "0.9" .
    "#;

    let result = parser.parse_str(data, StarFormat::TurtleStar);

    assert!(
        result.is_err(),
        "Should fail with missing closing delimiter"
    );
}

#[test]
fn test_annotation_syntax_error_delimiters_wrong_order() {
    use oxirs_star::StarConfig;

    // Use strict mode to ensure errors are caught
    let parser = StarParser::with_config(StarConfig {
        strict_mode: true,
        ..Default::default()
    });

    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:age "30" |} ex:certainty "0.9" {| .
    "#;

    let result = parser.parse_str(data, StarFormat::TurtleStar);

    assert!(
        result.is_err(),
        "Should fail with delimiters in wrong order"
    );
}

#[test]
fn test_annotation_syntax_mixed_with_regular_triples() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:knows ex:bob .
        ex:alice ex:age "30" {| ex:certainty "0.9" |} .
        ex:bob ex:name "Bob" .
        << ex:bob ex:name "Bob" >> ex:source ex:database .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse mix of regular and annotated triples");

    // 2 regular triples + 1 annotation + 1 quoted triple statement = 4 triples
    assert_eq!(graph.len(), 4);
}

#[test]
fn test_annotation_syntax_blank_node_predicates() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:age "30" {| _:b1 "metadata" |} .
    "#;

    // Blank nodes are NOT allowed as predicates in RDF or RDF-star
    // The parser should reject this or skip the malformed annotation
    let result = parser.parse_str(data, StarFormat::TurtleStar);

    // In non-strict mode, the parser might skip the invalid annotation
    // and return an empty graph
    if let Ok(graph) = result {
        assert_eq!(
            graph.len(),
            0,
            "Invalid annotation with blank node predicate should be skipped"
        );
    } else {
        // In strict mode, this would error
        assert!(result.is_err(), "Should reject blank node predicate");
    }
}

#[test]
fn test_annotation_syntax_integration_with_complex_base_triple() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        ex:alice ex:measurement "75.5"^^xsd:decimal {|
            ex:unit "kg";
            ex:timestamp "2023-10-12T14:30:00Z"^^xsd:dateTime;
            ex:device ex:scale42;
            ex:accuracy "±0.1"
        |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse complex annotated triple");

    assert_eq!(graph.len(), 4, "Should have 4 annotation triples");

    // Verify all annotations reference the same base triple
    let triples = graph.triples();
    let first_quoted = &triples[0].subject;

    for triple in triples {
        assert_eq!(
            &triple.subject, first_quoted,
            "All annotations should reference same base triple"
        );
    }
}

#[test]
fn test_annotation_syntax_stress_many_properties() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:data "value" {|
            ex:p1 "v1"; ex:p2 "v2"; ex:p3 "v3"; ex:p4 "v4"; ex:p5 "v5";
            ex:p6 "v6"; ex:p7 "v7"; ex:p8 "v8"; ex:p9 "v9"; ex:p10 "v10"
        |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse many annotation properties");

    assert_eq!(graph.len(), 10, "Should have 10 annotation triples");
}

#[test]
fn test_annotation_syntax_ntriples_star_format() {
    let parser = StarParser::new();

    // N-Triples-star doesn't support prefixes or annotation syntax
    // This test documents expected behavior
    let data = r#"
        << <http://example.org/alice> <http://example.org/age> "30" >> <http://example.org/certainty> "0.9" .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::NTriplesStar)
        .expect("N-Triples-star should parse quoted triples");

    assert_eq!(graph.len(), 1);
}

#[test]
fn test_annotation_syntax_preserves_quoted_triple_structure() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:age "30" {| ex:certainty "0.9" |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse");

    let triple = &graph.triples()[0];

    // Extract the quoted triple
    let quoted_triple = triple
        .subject
        .as_quoted_triple()
        .expect("Subject should be quoted triple");

    // Verify the structure of the base triple
    assert!(quoted_triple.subject.is_named_node());
    assert!(quoted_triple.predicate.is_named_node());
    assert!(quoted_triple.object.is_literal());

    let subject_iri = quoted_triple.subject.as_named_node().unwrap().iri.as_str();
    let predicate_iri = quoted_triple
        .predicate
        .as_named_node()
        .unwrap()
        .iri
        .as_str();
    let object_value = quoted_triple.object.as_literal().unwrap().value.as_str();

    assert_eq!(subject_iri, "http://example.org/alice");
    assert_eq!(predicate_iri, "http://example.org/age");
    assert_eq!(object_value, "30");
}

#[test]
fn test_annotation_syntax_count_quoted_triples() {
    let parser = StarParser::new();
    let data = r#"
        @prefix ex: <http://example.org/> .

        ex:alice ex:age "30" {| ex:certainty "0.9" |} .
        ex:bob ex:age "25" {| ex:certainty "0.8" |} .
    "#;

    let graph = parser
        .parse_str(data, StarFormat::TurtleStar)
        .expect("Failed to parse");

    // Check the quoted triple counting method
    let quoted_count = graph.count_quoted_triples();
    assert!(quoted_count >= 2, "Should count at least 2 quoted triples");
}
