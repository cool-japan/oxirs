//! Tests for RDF 1.2 features (RDF-star and directional language tags)

use oxirs_core::model::{NamedNode, Object, Predicate, Subject};
use oxirs_ttl::formats::turtle::TurtleParser;
use oxirs_ttl::toolkit::Parser;

#[test]
fn test_quoted_triple_as_subject() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:knows ex:bob >> ex:certainty "high" .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 1);

    // Check that subject is a quoted triple
    match triples[0].subject() {
        Subject::QuotedTriple(qt) => {
            assert_eq!(
                qt.subject(),
                &Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap())
            );
            assert_eq!(
                qt.predicate(),
                &Predicate::NamedNode(NamedNode::new("http://example.org/knows").unwrap())
            );
            assert_eq!(
                qt.object(),
                &Object::NamedNode(NamedNode::new("http://example.org/bob").unwrap())
            );
        }
        _ => panic!("Expected quoted triple as subject"),
    }

    // Check the predicate and object
    assert_eq!(
        triples[0].predicate(),
        &Predicate::NamedNode(NamedNode::new("http://example.org/certainty").unwrap())
    );
}

#[test]
fn test_quoted_triple_as_object() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:statement ex:says << ex:alice ex:knows ex:bob >> .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 1);

    // Check that object is a quoted triple
    match triples[0].object() {
        Object::QuotedTriple(qt) => {
            assert_eq!(
                qt.subject(),
                &Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap())
            );
            assert_eq!(
                qt.predicate(),
                &Predicate::NamedNode(NamedNode::new("http://example.org/knows").unwrap())
            );
            assert_eq!(
                qt.object(),
                &Object::NamedNode(NamedNode::new("http://example.org/bob").unwrap())
            );
        }
        _ => panic!("Expected quoted triple as object"),
    }
}

#[test]
fn test_nested_quoted_triples() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< << ex:alice ex:knows ex:bob >> ex:certainty "high" >> ex:source ex:researcher .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 1);

    // Check that subject is a quoted triple containing another quoted triple
    match triples[0].subject() {
        Subject::QuotedTriple(outer_qt) => match outer_qt.subject() {
            Subject::QuotedTriple(inner_qt) => {
                assert_eq!(
                    inner_qt.subject(),
                    &Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap())
                );
            }
            _ => panic!("Expected nested quoted triple"),
        },
        _ => panic!("Expected quoted triple as subject"),
    }
}

#[test]
fn test_quoted_triple_with_blank_nodes() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< _:b1 ex:knows _:b2 >> ex:verified true .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 1);

    // Check that quoted triple contains blank nodes
    match triples[0].subject() {
        Subject::QuotedTriple(qt) => {
            assert!(matches!(qt.subject(), Subject::BlankNode(_)));
            assert!(matches!(qt.object(), Object::BlankNode(_)));
        }
        _ => panic!("Expected quoted triple as subject"),
    }
}

#[test]
fn test_quoted_triple_with_literals() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:age 30 >> ex:confidence 0.95 .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 1);

    // Check that quoted triple contains a literal object
    match triples[0].subject() {
        Subject::QuotedTriple(qt) => {
            assert!(matches!(qt.object(), Object::Literal(_)));
        }
        _ => panic!("Expected quoted triple as subject"),
    }
}

#[test]
fn test_multiple_quoted_triples() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:knows ex:bob >> ex:certainty "high" .
<< ex:alice ex:knows ex:charlie >> ex:certainty "low" .
<< ex:bob ex:likes ex:alice >> ex:verified true .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 3);

    // Verify all subjects are quoted triples
    for triple in &triples {
        assert!(matches!(triple.subject(), Subject::QuotedTriple(_)));
    }
}

#[test]
fn test_quoted_triple_with_prefixed_names() {
    let ttl = r#"
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

<< ex:alice foaf:knows ex:bob >> ex:certainty "medium" .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 1);

    match triples[0].subject() {
        Subject::QuotedTriple(qt) => {
            assert_eq!(
                qt.predicate(),
                &Predicate::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/knows").unwrap())
            );
        }
        _ => panic!("Expected quoted triple as subject"),
    }
}

#[test]
fn test_quoted_triple_roundtrip() {
    use oxirs_ttl::formats::turtle::TurtleSerializer;
    use oxirs_ttl::toolkit::Serializer;

    let ttl = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:knows ex:bob >> ex:certainty "high" .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize(&triples, &mut output)
        .expect("Serialization failed");

    let output_str = String::from_utf8(output).unwrap();

    // Parse the serialized output
    let triples2 = parser
        .parse(output_str.as_bytes())
        .expect("Re-parse failed");

    assert_eq!(triples.len(), triples2.len());
    assert_eq!(triples[0], triples2[0]);
}

#[test]
fn test_quoted_triple_with_collection() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:likes ( ex:apple ex:banana ex:orange ) >> ex:verified true .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    // Collection creates additional triples, so we expect more than 1
    assert!(triples.len() > 1);

    // Find the main triple with the quoted triple
    let main_triple = triples
        .iter()
        .find(|t| matches!(t.subject(), Subject::QuotedTriple(_)))
        .expect("No quoted triple found");

    match main_triple.subject() {
        Subject::QuotedTriple(qt) => {
            assert_eq!(
                qt.predicate(),
                &Predicate::NamedNode(NamedNode::new("http://example.org/likes").unwrap())
            );
        }
        _ => panic!("Expected quoted triple"),
    }
}

#[test]
fn test_quoted_triple_with_blank_node_property_list() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:knows [ ex:name "Bob" ; ex:age 30 ] >> ex:verified true .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    // Blank node property list creates additional triples
    assert!(triples.len() > 1);

    // Find the main triple
    let main_triple = triples
        .iter()
        .find(|t| matches!(t.subject(), Subject::QuotedTriple(_)))
        .expect("No quoted triple found");

    match main_triple.subject() {
        Subject::QuotedTriple(qt) => {
            assert!(matches!(qt.object(), Object::BlankNode(_)));
        }
        _ => panic!("Expected quoted triple"),
    }
}

#[test]
fn test_error_unclosed_quoted_triple() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:knows ex:bob ex:verified true .
"#;

    let parser = TurtleParser::new();
    let result = parser.parse(ttl.as_bytes());

    assert!(result.is_err());
}

#[test]
fn test_error_empty_quoted_triple() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< >> ex:empty true .
"#;

    let parser = TurtleParser::new();
    let result = parser.parse(ttl.as_bytes());

    assert!(result.is_err());
}

// ============================================================================
// Directional Language Tag Tests (RDF 1.2)
// ============================================================================

#[test]
#[cfg(feature = "rdf-12")]
fn test_directional_language_tag_ltr() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:poem ex:title "Hello World"@en--ltr .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 1);

    match triples[0].object() {
        Object::Literal(lit) => {
            assert_eq!(lit.value(), "Hello World");
            assert_eq!(lit.language(), Some("en"));
            assert_eq!(
                lit.direction(),
                Some(oxirs_core::model::literal::BaseDirection::Ltr)
            );
        }
        _ => panic!("Expected literal object"),
    }
}

#[test]
#[cfg(feature = "rdf-12")]
fn test_directional_language_tag_rtl() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:poem ex:title "مرحبا"@ar--rtl .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 1);

    match triples[0].object() {
        Object::Literal(lit) => {
            assert_eq!(lit.value(), "مرحبا");
            assert_eq!(lit.language(), Some("ar"));
            assert_eq!(
                lit.direction(),
                Some(oxirs_core::model::literal::BaseDirection::Rtl)
            );
        }
        _ => panic!("Expected literal object"),
    }
}

#[test]
#[cfg(feature = "rdf-12")]
fn test_multiple_directional_language_tags() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:doc1 ex:text "Hello"@en--ltr .
ex:doc2 ex:text "שלום"@he--rtl .
ex:doc3 ex:text "Bonjour"@fr--ltr .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 3);

    // First triple: English LTR
    match triples[0].object() {
        Object::Literal(lit) => {
            assert_eq!(lit.value(), "Hello");
            assert_eq!(lit.language(), Some("en"));
            assert_eq!(
                lit.direction(),
                Some(oxirs_core::model::literal::BaseDirection::Ltr)
            );
        }
        _ => panic!("Expected literal object"),
    }

    // Second triple: Hebrew RTL
    match triples[1].object() {
        Object::Literal(lit) => {
            assert_eq!(lit.value(), "שלום");
            assert_eq!(lit.language(), Some("he"));
            assert_eq!(
                lit.direction(),
                Some(oxirs_core::model::literal::BaseDirection::Rtl)
            );
        }
        _ => panic!("Expected literal object"),
    }

    // Third triple: French LTR
    match triples[2].object() {
        Object::Literal(lit) => {
            assert_eq!(lit.value(), "Bonjour");
            assert_eq!(lit.language(), Some("fr"));
            assert_eq!(
                lit.direction(),
                Some(oxirs_core::model::literal::BaseDirection::Ltr)
            );
        }
        _ => panic!("Expected literal object"),
    }
}

#[test]
#[cfg(feature = "rdf-12")]
fn test_directional_language_tag_roundtrip() {
    use oxirs_ttl::formats::turtle::TurtleSerializer;
    use oxirs_ttl::toolkit::Serializer;

    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:text1 ex:title "Hello"@en--ltr .
ex:text2 ex:title "مرحبا"@ar--rtl .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 2);

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize(&triples, &mut output)
        .expect("Serialization failed");

    let output_str = String::from_utf8(output).unwrap();

    // Parse the serialized output
    let triples2 = parser
        .parse(output_str.as_bytes())
        .expect("Re-parse failed");

    assert_eq!(triples.len(), triples2.len());
    assert_eq!(triples[0], triples2[0]);
    assert_eq!(triples[1], triples2[1]);
}

#[test]
#[cfg(feature = "rdf-12")]
fn test_mixed_language_tags() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:doc1 ex:text "Plain"@en .
ex:doc2 ex:text "Directional"@en--ltr .
ex:doc3 ex:text "Another"@fr .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 3);

    // First: plain language tag
    match triples[0].object() {
        Object::Literal(lit) => {
            assert_eq!(lit.language(), Some("en"));
            assert_eq!(lit.direction(), None);
        }
        _ => panic!("Expected literal"),
    }

    // Second: directional language tag
    match triples[1].object() {
        Object::Literal(lit) => {
            assert_eq!(lit.language(), Some("en"));
            assert_eq!(
                lit.direction(),
                Some(oxirs_core::model::literal::BaseDirection::Ltr)
            );
        }
        _ => panic!("Expected literal"),
    }

    // Third: plain language tag
    match triples[2].object() {
        Object::Literal(lit) => {
            assert_eq!(lit.language(), Some("fr"));
            assert_eq!(lit.direction(), None);
        }
        _ => panic!("Expected literal"),
    }
}

#[test]
#[cfg(feature = "rdf-12")]
fn test_directional_language_tag_with_quoted_triple() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:says "Hello"@en--ltr >> ex:verified true .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 1);

    match triples[0].subject() {
        Subject::QuotedTriple(qt) => match qt.object() {
            Object::Literal(lit) => {
                assert_eq!(lit.value(), "Hello");
                assert_eq!(lit.language(), Some("en"));
                assert_eq!(
                    lit.direction(),
                    Some(oxirs_core::model::literal::BaseDirection::Ltr)
                );
            }
            _ => panic!("Expected literal in quoted triple"),
        },
        _ => panic!("Expected quoted triple"),
    }
}

#[test]
fn test_error_invalid_direction() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:text ex:value "Invalid"@en--invalid .
"#;

    let parser = TurtleParser::new();
    let result = parser.parse(ttl.as_bytes());

    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Invalid direction") || err_msg.contains("invalid"));
}

#[test]
#[cfg(not(feature = "rdf-12"))]
fn test_directional_language_tag_requires_feature() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:text ex:value "Hello"@en--ltr .
"#;

    let parser = TurtleParser::new();
    let result = parser.parse(ttl.as_bytes());

    // Without rdf-12 feature, this should fail
    assert!(result.is_err());
}
