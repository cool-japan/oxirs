//! Tests for RDF-star annotation syntax ({| |})
//!
//! This module tests the W3C RDF-star annotation syntax feature.
//! Annotation syntax allows attaching metadata to triples in a concise way:
//!
//! ```turtle
//! :alice :age 30 {| :certainty 0.9; :source :survey |} .
//! ```
//!
//! Per the W3C RDF-star annotation-syntax equivalence this asserts the base
//! triple AND the annotation triples:
//! ```turtle
//! :alice :age 30 .
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

    // Per the W3C RDF-star annotation-syntax equivalence, `s p o {| q v |} .`
    // expands to `s p o . <<s p o>> q v .` — the base triple IS asserted in
    // addition to the annotation triple. (Referential opacity applies to a
    // bare `<< >>` quoted-triple TERM, not to the `{| |}` annotation sugar.)
    // So we expect 2 triples: base + 1 annotation.
    assert_eq!(graph.len(), 2, "Should have base triple + 1 annotation");

    // Locate the annotation triple (its subject is the quoted base triple)
    let triples = graph.triples();
    let annotation_triple = triples
        .iter()
        .find(|t| t.subject.is_quoted_triple())
        .expect("Annotation triple with quoted-triple subject should exist");

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

    // W3C expansion: base triple + one annotation triple per property.
    assert_eq!(graph.len(), 3, "Should have base triple + 2 annotations");

    // The two annotation triples share the quoted base triple as subject.
    let triples = graph.triples();
    let annotation_triples: Vec<_> = triples
        .iter()
        .filter(|t| t.subject.is_quoted_triple())
        .collect();
    assert_eq!(
        annotation_triples.len(),
        2,
        "Should have exactly 2 annotation triples"
    );

    // Verify both annotation predicates
    let predicates: Vec<String> = annotation_triples
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

    // W3C expansion: 1 base triple + 3 annotation triples.
    assert_eq!(graph.len(), 4, "Should have base triple + 3 annotations");

    // All annotation triples (quoted-triple subjects) share the same subject.
    let triples = graph.triples();
    let annotation_triples: Vec<_> = triples
        .iter()
        .filter(|t| t.subject.is_quoted_triple())
        .collect();
    assert_eq!(annotation_triples.len(), 3, "Should have 3 annotations");

    let first_subject = &annotation_triples[0].subject;
    assert!(first_subject.is_quoted_triple());
    for triple in &annotation_triples {
        assert_eq!(
            &triple.subject, first_subject,
            "All annotation subjects should be identical"
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

    // W3C expansion: base triple + 1 annotation triple.
    assert_eq!(graph.len(), 2);

    let triples = graph.triples();
    let triple = triples
        .iter()
        .find(|t| t.subject.is_quoted_triple())
        .expect("Annotation triple with quoted-triple subject should exist");
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

    // Per the W3C annotation-syntax equivalence, the base triple is asserted
    // even when the annotation block is empty (`s p o {| |} .` == `s p o .`),
    // so exactly the base triple remains and no annotation triples are added.
    assert_eq!(
        graph.len(),
        1,
        "Empty annotation block still asserts the base triple"
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

    // W3C expansion: base triple + 1 annotation triple.
    assert_eq!(graph.len(), 2);
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

    // W3C expansion: base triple + 2 annotation triples.
    assert_eq!(graph.len(), 3);

    // Every object here (base "50000"^^xsd:integer plus the two annotation
    // values) is a literal.
    let triples = graph.triples();
    for triple in triples {
        assert!(
            triple.object.is_literal(),
            "All object values should be literals"
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

    // W3C expansion: base triple + 2 annotation triples.
    assert_eq!(graph.len(), 3);
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

    // W3C expansion asserts each base triple plus its annotations:
    //   (1 base + 1 ann) + (1 base + 2 ann) + (1 base + 1 ann) = 7 triples.
    assert_eq!(graph.len(), 7);

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

    // Ignoring the empty statement after the semicolon, the W3C expansion is
    // base triple + 1 annotation triple.
    assert_eq!(graph.len(), 2);
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

    // 2 regular triples + (1 base + 1 annotation from the {| |} statement)
    // + 1 quoted-triple statement = 5 triples.
    assert_eq!(graph.len(), 5);
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

    // In non-strict mode the base triple is still asserted (per the W3C
    // annotation-syntax equivalence) while the invalid blank-node-predicate
    // annotation is skipped, leaving only the base triple.
    if let Ok(graph) = result {
        assert_eq!(
            graph.len(),
            1,
            "Base triple is asserted; invalid blank-node-predicate annotation is skipped"
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

    // W3C expansion: 1 base triple + 4 annotation triples.
    assert_eq!(graph.len(), 5, "Should have base triple + 4 annotations");

    // Verify all annotation triples reference the same quoted base triple.
    let triples = graph.triples();
    let annotation_triples: Vec<_> = triples
        .iter()
        .filter(|t| t.subject.is_quoted_triple())
        .collect();
    assert_eq!(annotation_triples.len(), 4, "Should have 4 annotations");

    let first_quoted = &annotation_triples[0].subject;
    for triple in &annotation_triples {
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

    // W3C expansion: 1 base triple + 10 annotation triples.
    assert_eq!(graph.len(), 11, "Should have base triple + 10 annotations");
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

    // The graph now contains both the asserted base triple and the annotation
    // triple; select the annotation triple, whose subject is the quoted base.
    let triples = graph.triples();
    let triple = triples
        .iter()
        .find(|t| t.subject.is_quoted_triple())
        .expect("Annotation triple with quoted-triple subject should exist");

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
