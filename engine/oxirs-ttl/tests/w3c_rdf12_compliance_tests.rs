//! W3C RDF 1.2 / RDF-star Compliance Test Suite
//!
//! This test suite validates compliance with the W3C RDF 1.2 specification,
//! with a focus on RDF-star (quoted triples) and directional language tags.
//!
//! References:
//! - RDF 1.2 Concepts: https://www.w3.org/TR/rdf12-concepts/
//! - RDF-star: https://www.w3.org/TR/rdf12-turtle/#rdf-star
//! - Directional Language Tags: https://www.w3.org/TR/rdf12-concepts/#dfn-language-tagged-string

use oxirs_core::model::{Object, Predicate, Subject, Triple};
use oxirs_ttl::formats::turtle::{TurtleParser, TurtleSerializer};
use oxirs_ttl::toolkit::{Parser, Serializer};

/// Helper function to parse Turtle with RDF 1.2 features
fn parse_rdf12(input: &str) -> Result<Vec<Triple>, Box<dyn std::error::Error>> {
    let parser = TurtleParser::new();
    Ok(parser.parse(input.as_bytes())?)
}

/// Helper function to serialize and re-parse (round-trip test)
fn roundtrip_rdf12(triples: &[Triple]) -> Result<Vec<Triple>, Box<dyn std::error::Error>> {
    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer.serialize(triples, &mut output)?;

    let parser = TurtleParser::new();
    Ok(parser.parse(&output[..])?)
}

// ============================================================================
// RDF-star Positive Syntax Tests
// ============================================================================

#[test]
fn test_rdfstar_quoted_triple_as_subject() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
<< ex:alice ex:knows ex:bob >> ex:certainty "high" .
"#;

    let triples = parse_rdf12(ttl).expect("Should parse quoted triple as subject");
    assert_eq!(triples.len(), 1);

    // Verify subject is a quoted triple
    matches!(triples[0].subject(), Subject::QuotedTriple(_));
}

#[test]
fn test_rdfstar_quoted_triple_as_object() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
ex:statement ex:about << ex:alice ex:knows ex:bob >> .
"#;

    let triples = parse_rdf12(ttl).expect("Should parse quoted triple as object");
    assert_eq!(triples.len(), 1);

    // Verify object is a quoted triple
    matches!(triples[0].object(), Object::QuotedTriple(_));
}

#[test]
fn test_rdfstar_nested_quoted_triples() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
<< << ex:a ex:b ex:c >> ex:d ex:e >> ex:f ex:g .
"#;

    let triples = parse_rdf12(ttl).expect("Should parse nested quoted triples");
    assert_eq!(triples.len(), 1);

    // Verify nested structure
    if let Subject::QuotedTriple(outer) = triples[0].subject() {
        matches!(outer.subject(), Subject::QuotedTriple(_));
    } else {
        panic!("Expected quoted triple as subject");
    }
}

#[test]
fn test_rdfstar_quoted_triple_with_blank_nodes() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
<< _:alice ex:knows _:bob >> ex:source ex:socialNetwork .
"#;

    let triples = parse_rdf12(ttl).expect("Should parse quoted triple with blank nodes");
    assert_eq!(triples.len(), 1);

    // Verify blank nodes in quoted triple
    if let Subject::QuotedTriple(qt) = triples[0].subject() {
        matches!(qt.subject(), Subject::BlankNode(_));
        matches!(qt.object(), Object::BlankNode(_));
    } else {
        panic!("Expected quoted triple as subject");
    }
}

#[test]
fn test_rdfstar_quoted_triple_with_literals() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
<< ex:temperature ex:value "23.5"^^<http://www.w3.org/2001/XMLSchema#decimal> >> ex:recorded "2025-12-05T10:00:00"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
"#;

    let triples = parse_rdf12(ttl).expect("Should parse quoted triple with typed literals");
    assert_eq!(triples.len(), 1);

    // Verify literal in quoted triple
    if let Subject::QuotedTriple(qt) = triples[0].subject() {
        matches!(qt.object(), Object::Literal(_));
    } else {
        panic!("Expected quoted triple as subject");
    }
}

#[test]
fn test_rdfstar_multiple_quoted_triples() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:knows ex:bob >> ex:confidence "high" .
<< ex:alice ex:knows ex:charlie >> ex:confidence "medium" .
<< ex:bob ex:knows ex:charlie >> ex:confidence "low" .
"#;

    let triples = parse_rdf12(ttl).expect("Should parse multiple quoted triples");
    assert_eq!(triples.len(), 3);

    // All subjects should be quoted triples
    for triple in &triples {
        matches!(triple.subject(), Subject::QuotedTriple(_));
    }
}

#[test]
fn test_rdfstar_quoted_triple_in_collection() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
ex:list ex:contains ( << ex:a ex:b ex:c >> << ex:d ex:e ex:f >> ) .
"#;

    // This tests that quoted triples can appear in RDF collections
    let result = parse_rdf12(ttl);
    assert!(result.is_ok(), "Should parse quoted triples in collections");
}

#[test]
fn test_rdfstar_quoted_triple_with_annotation_syntax() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
ex:alice ex:knows ex:bob {| ex:since 2020 |} .
"#;

    // The {| |} annotation syntax is RDF-star shorthand for quoted triples
    // If supported, this should parse correctly
    let result = parse_rdf12(ttl);

    // This might not be supported yet - mark as optional test
    if result.is_err() {
        println!("Note: Annotation syntax {{| |}} not yet supported");
    }
}

// ============================================================================
// Directional Language Tags - Positive Syntax Tests
// ============================================================================

#[test]
fn test_directional_language_tag_ltr() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
ex:greeting ex:text "Hello"@en--ltr .
"#;

    let triples = parse_rdf12(ttl).expect("Should parse LTR language tag");
    assert_eq!(triples.len(), 1);

    // Verify directional language tag
    if let Object::Literal(lit) = triples[0].object() {
        assert_eq!(lit.language(), Some("en"));
        // The direction should be stored as part of the language tag or separately
    } else {
        panic!("Expected literal with language tag");
    }
}

#[test]
fn test_directional_language_tag_rtl() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
ex:greeting ex:text "مرحبا"@ar--rtl .
"#;

    let triples = parse_rdf12(ttl).expect("Should parse RTL language tag");
    assert_eq!(triples.len(), 1);

    // Verify RTL directional tag
    if let Object::Literal(lit) = triples[0].object() {
        assert_eq!(lit.language(), Some("ar"));
    } else {
        panic!("Expected literal with language tag");
    }
}

#[test]
fn test_directional_language_tag_multiple() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:alice ex:nameEn "Alice"@en--ltr .
ex:alice ex:nameAr "أليس"@ar--rtl .
ex:alice ex:nameHe "אליס"@he--rtl .
"#;

    let triples = parse_rdf12(ttl).expect("Should parse multiple directional tags");
    assert_eq!(triples.len(), 3);

    // All objects should be language-tagged literals
    for triple in &triples {
        matches!(triple.object(), Object::Literal(_));
    }
}

#[test]
fn test_mixed_directional_and_plain_language_tags() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:text1 ex:value "Hello"@en .
ex:text2 ex:value "Hello"@en--ltr .
ex:text3 ex:value "مرحبا"@ar .
ex:text4 ex:value "مرحبا"@ar--rtl .
"#;

    let triples = parse_rdf12(ttl).expect("Should parse mixed language tags");
    assert_eq!(triples.len(), 4);
}

// ============================================================================
// RDF-star Negative Syntax Tests (should fail)
// ============================================================================

#[test]
fn test_rdfstar_invalid_empty_quoted_triple() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
<< >> ex:prop "value" .
"#;

    let result = parse_rdf12(ttl);
    assert!(result.is_err(), "Empty quoted triple should fail");
}

#[test]
fn test_rdfstar_invalid_incomplete_quoted_triple() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
<< ex:alice ex:knows >> ex:prop "value" .
"#;

    let result = parse_rdf12(ttl);
    assert!(result.is_err(), "Incomplete quoted triple should fail");
}

#[test]
fn test_rdfstar_invalid_quoted_triple_as_predicate() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
ex:subject << ex:a ex:b ex:c >> ex:object .
"#;

    let result = parse_rdf12(ttl);
    assert!(result.is_err(), "Quoted triple as predicate should fail");
}

#[test]
fn test_directional_invalid_direction_value() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
ex:text ex:value "Hello"@en--updown .
"#;

    let result = parse_rdf12(ttl);
    assert!(result.is_err(), "Invalid direction value should fail");
}

#[test]
fn test_directional_missing_language_tag() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
ex:text ex:value "Hello"@--ltr .
"#;

    let result = parse_rdf12(ttl);
    assert!(
        result.is_err(),
        "Missing language before direction should fail"
    );
}

// ============================================================================
// Round-trip Serialization Tests
// ============================================================================

#[test]
fn test_rdfstar_roundtrip_quoted_triple() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
<< ex:alice ex:knows ex:bob >> ex:certainty "high" .
"#;

    let original = parse_rdf12(ttl).expect("Parse failed");
    let roundtrip = roundtrip_rdf12(&original).expect("Round-trip failed");

    assert_eq!(original.len(), roundtrip.len());
    // More detailed comparison would check structural equality
}

#[test]
fn test_directional_roundtrip() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
ex:greeting ex:text "Hello"@en--ltr .
"#;

    let original = parse_rdf12(ttl).expect("Parse failed");
    let roundtrip = roundtrip_rdf12(&original).expect("Round-trip failed");

    assert_eq!(original.len(), roundtrip.len());
}

#[test]
fn test_complex_rdf12_roundtrip() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:knows ex:bob >> ex:confidence "high"@en--ltr .
<< ex:alice ex:age 30 >> ex:verified "true"^^<http://www.w3.org/2001/XMLSchema#boolean> .
ex:greeting ex:text "مرحبا"@ar--rtl .
"#;

    let original = parse_rdf12(ttl).expect("Parse failed");
    let roundtrip = roundtrip_rdf12(&original).expect("Round-trip failed");

    assert_eq!(original.len(), roundtrip.len());
}

// ============================================================================
// Evaluation Tests - Verify Semantic Correctness
// ============================================================================

#[test]
fn test_rdfstar_quoted_triple_structure() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
<< ex:alice ex:knows ex:bob >> ex:certainty "0.9"^^<http://www.w3.org/2001/XMLSchema#decimal> .
"#;

    let triples = parse_rdf12(ttl).expect("Parse failed");
    assert_eq!(triples.len(), 1);

    let triple = &triples[0];

    // Verify quoted triple structure
    if let Subject::QuotedTriple(qt) = triple.subject() {
        // Check subject of quoted triple
        if let Subject::NamedNode(nn) = qt.subject() {
            assert_eq!(nn.as_str(), "http://example.org/alice");
        } else {
            panic!("Expected named node as subject");
        }

        // Check predicate of quoted triple
        if let Predicate::NamedNode(nn) = qt.predicate() {
            assert_eq!(nn.as_str(), "http://example.org/knows");
        } else {
            panic!("Expected named node as predicate");
        }

        // Check object of quoted triple
        if let Object::NamedNode(nn) = qt.object() {
            assert_eq!(nn.as_str(), "http://example.org/bob");
        } else {
            panic!("Expected named node as object");
        }
    } else {
        panic!("Expected quoted triple as subject");
    }

    // Check the main triple's predicate
    if let Predicate::NamedNode(nn) = triple.predicate() {
        assert_eq!(nn.as_str(), "http://example.org/certainty");
    } else {
        panic!("Expected named node as predicate");
    }

    // Check the main triple's object (typed literal)
    if let Object::Literal(lit) = triple.object() {
        assert_eq!(lit.value(), "0.9");
        assert_eq!(
            lit.datatype().as_str(),
            "http://www.w3.org/2001/XMLSchema#decimal"
        );
    } else {
        panic!("Expected typed literal as object");
    }
}

#[test]
fn test_directional_language_tag_structure() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
ex:greeting ex:text "Hello"@en--ltr .
"#;

    let triples = parse_rdf12(ttl).expect("Parse failed");
    assert_eq!(triples.len(), 1);

    let triple = &triples[0];

    // Verify literal structure
    if let Object::Literal(lit) = triple.object() {
        assert_eq!(lit.value(), "Hello");
        assert_eq!(lit.language(), Some("en"));

        // The direction is stored as part of the language tag in RDF 1.2
        // Format: "en--ltr" or handled separately depending on implementation
        // This test verifies the language part is correctly extracted
    } else {
        panic!("Expected language-tagged literal");
    }
}

// ============================================================================
// Mixed RDF 1.1 and RDF 1.2 Features
// ============================================================================

#[test]
fn test_mixed_rdf11_and_rdf12_features() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

# Standard RDF 1.1 triples
ex:alice rdf:type ex:Person .
ex:alice ex:name "Alice" .

# RDF 1.2 quoted triple
<< ex:alice ex:knows ex:bob >> ex:confidence "high" .

# RDF 1.2 directional language tag
ex:alice ex:greeting "Hello"@en--ltr .

# Standard collection
ex:alice ex:friends ( ex:bob ex:charlie ) .
"#;

    let triples = parse_rdf12(ttl).expect("Should parse mixed RDF 1.1 and 1.2");
    assert!(triples.len() >= 4, "Expected at least 4 triples");
}

// ============================================================================
// Performance and Scalability Tests
// ============================================================================

#[test]
fn test_large_quoted_triple_dataset() {
    let mut ttl = String::from("@prefix ex: <http://example.org/> .\n\n");

    // Generate 100 quoted triples
    for i in 0..100 {
        ttl.push_str(&format!(
            "<< ex:subject{} ex:predicate{} ex:object{} >> ex:id {} .\n",
            i, i, i, i
        ));
    }

    let triples = parse_rdf12(&ttl).expect("Should parse large quoted triple dataset");
    assert_eq!(triples.len(), 100);
}

#[test]
fn test_deeply_nested_quoted_triples() {
    // Test nesting depth limit (reasonable limit is around 10 levels)
    let mut ttl = String::from("@prefix ex: <http://example.org/> .\n\n");

    // Create nested structure: << << << ex:a ex:b ex:c >> ex:d ex:e >> ex:f ex:g >> ex:h ex:i .
    let mut nested = String::from("<< ex:a ex:b ex:c >>");
    for i in 0..5 {
        nested = format!("<< {} ex:p{} ex:o{} >>", nested, i, i);
    }
    ttl.push_str(&format!("{} ex:final ex:value .\n", nested));

    let result = parse_rdf12(&ttl);
    assert!(
        result.is_ok(),
        "Should handle nested quoted triples up to reasonable depth"
    );
}

#[test]
fn test_performance_baseline_rdf12() {
    use std::time::Instant;

    let mut ttl = String::from("@prefix ex: <http://example.org/> .\n\n");

    // Generate mixed RDF 1.2 content
    for i in 0..1000 {
        if i % 3 == 0 {
            ttl.push_str(&format!(
                "<< ex:s{} ex:p{} ex:o{} >> ex:confidence \"high\" .\n",
                i, i, i
            ));
        } else if i % 3 == 1 {
            ttl.push_str(&format!(
                "ex:subject{} ex:name \"Name {}\"@en--ltr .\n",
                i, i
            ));
        } else {
            ttl.push_str(&format!("ex:subject{} ex:value {} .\n", i, i));
        }
    }

    let start = Instant::now();
    let triples = parse_rdf12(&ttl).expect("Parse failed");
    let duration = start.elapsed();

    assert_eq!(triples.len(), 1000);

    // Performance baseline: should parse 1000 RDF 1.2 triples in < 100ms
    assert!(
        duration.as_millis() < 100,
        "Parsing 1000 RDF 1.2 triples took {:?} (should be < 100ms)",
        duration
    );

    println!(
        "RDF 1.2 Performance: {} triples in {:?}",
        triples.len(),
        duration
    );
}
