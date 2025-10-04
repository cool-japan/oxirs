//! Integration tests for RDF parsers
//!
//! Tests comprehensive parser functionality for all supported formats

use oxirs_core::format::{RdfFormat, RdfParser};
use oxirs_core::model::{Object, Subject};
use oxirs_core::RdfTerm;
use std::io::Cursor;

#[test]
fn test_turtle_parser_basic() {
    let turtle_data = r#"
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

        ex:subject1 ex:predicate1 "literal value" .
        ex:subject2 rdf:type ex:Type1 .
        ex:subject3 ex:predicate2 ex:object1 .
    "#;

    let parser = RdfParser::new(RdfFormat::Turtle);
    let cursor = Cursor::new(turtle_data.as_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse Turtle");

    assert_eq!(quads.len(), 3, "Should parse 3 triples");

    // Check first triple
    let quad1 = &quads[0];
    assert!(matches!(
        quad1.subject(),
        Subject::NamedNode(n) if n.as_str() == "http://example.org/subject1"
    ));
    assert_eq!(quad1.predicate().as_str(), "http://example.org/predicate1");
    assert!(matches!(quad1.object(), Object::Literal(l) if l.value() == "literal value"));
}

#[test]
fn test_turtle_parser_prefixes() {
    let turtle_data = r#"
        @prefix ex: <http://example.org/> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .
        @base <http://base.example.org/> .

        ex:alice foaf:name "Alice" ;
                 foaf:knows ex:bob .

        ex:bob foaf:name "Bob" .
    "#;

    let parser = RdfParser::new(RdfFormat::Turtle);
    let cursor = Cursor::new(turtle_data.as_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse Turtle with prefixes");

    assert!(quads.len() >= 3, "Should parse at least 3 triples");

    // Check that prefixes were expanded correctly
    for quad in &quads {
        if let Subject::NamedNode(n) = quad.subject() {
            assert!(
                n.as_str().starts_with("http://"),
                "Subject should be expanded: {}",
                n.as_str()
            );
        }
        assert!(
            quad.predicate().as_str().starts_with("http://"),
            "Predicate should be expanded: {}",
            quad.predicate().as_str()
        );
    }
}

#[test]
fn test_turtle_parser_literals() {
    let turtle_data = r#"
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        ex:entity1 ex:name "Alice"@en .
        ex:entity2 ex:age "25"^^xsd:integer .
        ex:entity3 ex:description """Multi-line
        literal value""" .
    "#;

    let parser = RdfParser::new(RdfFormat::Turtle);
    let cursor = Cursor::new(turtle_data.as_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse Turtle literals");

    assert_eq!(
        quads.len(),
        3,
        "Should parse 3 triples with different literal types"
    );

    // Check language-tagged literal
    let quad1 = &quads[0];
    if let Object::Literal(lit) = quad1.object() {
        assert_eq!(lit.value(), "Alice");
        assert_eq!(lit.language(), Some("en"));
    } else {
        panic!("Expected language-tagged literal");
    }
}

#[test]
fn test_rdfxml_parser_basic() {
    let rdfxml_data = r#"<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:ex="http://example.org/">
            <rdf:Description rdf:about="http://example.org/subject1">
                <ex:predicate1>literal value</ex:predicate1>
            </rdf:Description>
            <rdf:Description rdf:about="http://example.org/subject2">
                <rdf:type rdf:resource="http://example.org/Type1"/>
            </rdf:Description>
        </rdf:RDF>
    "#;

    let parser = RdfParser::new(RdfFormat::RdfXml);
    let cursor = Cursor::new(rdfxml_data.as_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse RDF/XML");

    assert!(quads.len() >= 2, "Should parse at least 2 triples");

    // Check that subjects are properly parsed
    for quad in &quads {
        if let Subject::NamedNode(n) = quad.subject() {
            assert!(
                n.as_str().starts_with("http://example.org/subject"),
                "Subject should be parsed correctly: {}",
                n.as_str()
            );
        }
    }
}

#[test]
fn test_rdfxml_parser_typed_nodes() {
    let rdfxml_data = r#"<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:ex="http://example.org/">
            <ex:Person rdf:about="http://example.org/alice">
                <ex:name>Alice</ex:name>
                <ex:age>25</ex:age>
            </ex:Person>
        </rdf:RDF>
    "#;

    let parser = RdfParser::new(RdfFormat::RdfXml);
    let cursor = Cursor::new(rdfxml_data.as_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse RDF/XML with typed nodes");

    // Should have at least 3 triples: rdf:type, name, age
    assert!(quads.len() >= 3, "Should parse type and properties");

    // Check for rdf:type triple
    let has_type = quads
        .iter()
        .any(|q| q.predicate().as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    assert!(has_type, "Should have rdf:type triple");
}

#[test]
fn test_jsonld_parser_basic() {
    let jsonld_data = r#"
    {
        "@context": {
            "ex": "http://example.org/"
        },
        "@id": "http://example.org/subject1",
        "ex:predicate1": "literal value",
        "@type": "ex:Type1"
    }
    "#;

    use oxirs_core::format::JsonLdProfileSet;
    let parser = RdfParser::new(RdfFormat::JsonLd {
        profile: JsonLdProfileSet::empty(),
    });
    let cursor = Cursor::new(jsonld_data.as_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse JSON-LD");

    assert!(
        quads.len() >= 2,
        "Should parse at least 2 triples (type + property)"
    );

    // Check for rdf:type triple
    let has_type = quads
        .iter()
        .any(|q| q.predicate().as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    assert!(has_type, "Should have rdf:type triple");
}

#[test]
fn test_jsonld_parser_array() {
    let jsonld_data = r#"
    [
        {
            "@id": "http://example.org/alice",
            "@type": "http://example.org/Person",
            "http://example.org/name": "Alice"
        },
        {
            "@id": "http://example.org/bob",
            "@type": "http://example.org/Person",
            "http://example.org/name": "Bob"
        }
    ]
    "#;

    use oxirs_core::format::JsonLdProfileSet;
    let parser = RdfParser::new(RdfFormat::JsonLd {
        profile: JsonLdProfileSet::empty(),
    });
    let cursor = Cursor::new(jsonld_data.as_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse JSON-LD array");

    // Should have at least 4 triples (2 types + 2 names)
    assert!(quads.len() >= 4, "Should parse multiple objects from array");
}

#[test]
fn test_jsonld_parser_literals() {
    let jsonld_data = r#"
    {
        "@id": "http://example.org/entity1",
        "http://example.org/name": {
            "@value": "Alice",
            "@language": "en"
        },
        "http://example.org/age": {
            "@value": "25",
            "@type": "http://www.w3.org/2001/XMLSchema#integer"
        }
    }
    "#;

    use oxirs_core::format::JsonLdProfileSet;
    let parser = RdfParser::new(RdfFormat::JsonLd {
        profile: JsonLdProfileSet::empty(),
    });
    let cursor = Cursor::new(jsonld_data.as_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse JSON-LD literals");

    assert_eq!(quads.len(), 2, "Should parse 2 triples with literals");

    // Check language-tagged literal
    let has_lang = quads.iter().any(|q| {
        if let Object::Literal(lit) = q.object() {
            lit.language() == Some("en")
        } else {
            false
        }
    });
    assert!(has_lang, "Should have language-tagged literal");
}

#[test]
fn test_parser_error_resilience() {
    // Test with invalid Turtle data in lenient mode
    let invalid_turtle = r#"
        @prefix ex: <http://example.org/> .

        ex:subject1 ex:predicate1 "valid triple" .
        this is invalid turtle syntax
        ex:subject2 ex:predicate2 "another valid triple" .
    "#;

    let parser = RdfParser::new(RdfFormat::Turtle).lenient();
    let cursor = Cursor::new(invalid_turtle.as_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_default();

    // In lenient mode, should parse valid triples and skip invalid ones
    assert!(
        quads.len() >= 1,
        "Should parse at least some valid triples in lenient mode"
    );
}

#[test]
fn test_format_round_trip_turtle() {
    use oxirs_core::format::RdfSerializer;

    let original_data = r#"
        @prefix ex: <http://example.org/> .

        ex:subject1 ex:predicate1 "literal value" .
        ex:subject2 ex:predicate2 ex:object1 .
    "#;

    // Parse
    let parser = RdfParser::new(RdfFormat::Turtle);
    let cursor = Cursor::new(original_data.as_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse");

    // Serialize back
    let output = Vec::new();
    let mut serializer = RdfSerializer::new(RdfFormat::Turtle).for_writer(output);
    for quad in &quads {
        serializer
            .serialize_quad(quad.as_ref())
            .expect("Failed to serialize");
    }
    let output = serializer.finish().expect("Failed to finish");

    // Parse again
    let parser2 = RdfParser::new(RdfFormat::Turtle);
    let cursor2 = Cursor::new(output.clone());
    let quads2: Vec<_> = parser2
        .for_reader(cursor2)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to re-parse");

    assert_eq!(
        quads.len(),
        quads2.len(),
        "Round-trip should preserve triple count"
    );
}

#[test]
fn test_cross_format_conversion() {
    use oxirs_core::format::RdfSerializer;

    // Start with Turtle
    let turtle_data = r#"
        @prefix ex: <http://example.org/> .
        ex:subject ex:predicate "value" .
    "#;

    let parser = RdfParser::new(RdfFormat::Turtle);
    let cursor = Cursor::new(turtle_data.as_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse Turtle");

    // Convert to N-Triples
    let ntriples_output = Vec::new();
    let mut serializer = RdfSerializer::new(RdfFormat::NTriples).for_writer(ntriples_output);
    for quad in &quads {
        serializer
            .serialize_quad(quad.as_ref())
            .expect("Failed to serialize to N-Triples");
    }
    let ntriples_output = serializer.finish().expect("Failed to finish");

    // Parse N-Triples
    let parser2 = RdfParser::new(RdfFormat::NTriples);
    let cursor2 = Cursor::new(ntriples_output.clone());
    let quads2: Vec<_> = parser2
        .for_reader(cursor2)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse N-Triples");

    assert_eq!(
        quads.len(),
        quads2.len(),
        "Cross-format conversion should preserve triples"
    );
}

#[test]
fn test_large_turtle_file() {
    // Generate large Turtle data
    let mut turtle_data = String::from("@prefix ex: <http://example.org/> .\n\n");
    for i in 0..1000 {
        turtle_data.push_str(&format!(
            "ex:subject{} ex:predicate{} \"value{}\" .\n",
            i,
            i % 10,
            i
        ));
    }

    let parser = RdfParser::new(RdfFormat::Turtle);
    let cursor = Cursor::new(turtle_data.into_bytes());
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to parse large Turtle file");

    assert_eq!(quads.len(), 1000, "Should parse all 1000 triples");

    // Verify some triples
    for (i, quad) in quads.iter().enumerate() {
        if let Subject::NamedNode(n) = quad.subject() {
            assert_eq!(n.as_str(), format!("http://example.org/subject{}", i));
        }
        if let Object::Literal(lit) = quad.object() {
            assert_eq!(lit.value(), format!("value{}", i));
        }
    }
}
