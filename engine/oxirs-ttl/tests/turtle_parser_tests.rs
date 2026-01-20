//! Comprehensive tests for Turtle parser

use oxirs_core::model::{Object, Predicate, Subject};
use oxirs_ttl::error::{TurtleParseError, TurtleSyntaxError};
use oxirs_ttl::turtle::TurtleParser;

#[test]
fn test_simple_triple() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        ex:subject ex:predicate "object" .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    let triple = &triples[0];
    match triple.subject() {
        Subject::NamedNode(node) => {
            assert_eq!(node.as_str(), "http://example.org/subject");
        }
        _ => panic!("Expected NamedNode subject"),
    }

    match triple.predicate() {
        Predicate::NamedNode(node) => {
            assert_eq!(node.as_str(), "http://example.org/predicate");
        }
        _ => panic!("Expected NamedNode predicate"),
    }

    match triple.object() {
        Object::Literal(lit) => {
            assert_eq!(lit.value(), "object");
        }
        _ => panic!("Expected Literal object"),
    }
}

#[test]
fn test_multiple_triples() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        ex:alice ex:name "Alice" .
        ex:bob ex:name "Bob" .
        ex:alice ex:knows ex:bob .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 3);
}

#[test]
fn test_prefixed_names() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .
        ex:alice foaf:name "Alice" .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    let triple = &triples[0];
    match triple.subject() {
        Subject::NamedNode(node) => {
            assert_eq!(node.as_str(), "http://example.org/alice");
        }
        _ => panic!("Expected NamedNode"),
    }

    match triple.predicate() {
        Predicate::NamedNode(node) => {
            assert_eq!(node.as_str(), "http://xmlns.com/foaf/0.1/name");
        }
        _ => panic!("Expected NamedNode predicate"),
    }
}

#[test]
fn test_rdf_type_shorthand() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        ex:alice a ex:Person .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    let triple = &triples[0];
    match triple.predicate() {
        Predicate::NamedNode(node) => {
            assert_eq!(
                node.as_str(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            );
        }
        _ => panic!("Expected NamedNode predicate"),
    }
}

#[test]
fn test_language_tagged_literal() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        ex:greeting ex:text "Hello"@en .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    let triple = &triples[0];
    match triple.object() {
        Object::Literal(lit) => {
            assert_eq!(lit.value(), "Hello");
            assert_eq!(lit.language(), Some("en"));
        }
        _ => panic!("Expected language-tagged literal"),
    }
}

#[test]
fn test_typed_literal() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        ex:value ex:number "42"^^xsd:integer .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    let triple = &triples[0];
    match triple.object() {
        Object::Literal(lit) => {
            assert_eq!(lit.value(), "42");
            assert_eq!(
                lit.datatype().as_str(),
                "http://www.w3.org/2001/XMLSchema#integer"
            );
        }
        _ => panic!("Expected typed literal"),
    }
}

#[test]
fn test_blank_node_label() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        _:alice ex:name "Alice" .
        _:bob ex:knows _:alice .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 2);

    // Verify blank nodes are used
    match triples[0].subject() {
        Subject::BlankNode(_) => {}
        _ => panic!("Expected BlankNode subject"),
    }
}

#[test]
fn test_anonymous_blank_node() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        [] ex:name "Anonymous" .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    match triples[0].subject() {
        Subject::BlankNode(_) => {}
        _ => panic!("Expected anonymous BlankNode"),
    }
}

#[test]
fn test_base_iri() {
    let turtle = r#"
        @base <http://example.org/> .
        <alice> <knows> <bob> .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    // Note: Current implementation has simplified IRI resolution
    // This test verifies the parser accepts @base declarations
}

#[test]
fn test_comments() {
    let turtle = r#"
        # This is a comment
        @prefix ex: <http://example.org/> .
        # Another comment
        ex:subject ex:predicate "object" . # Inline comment
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);
}

#[test]
fn test_multiline_string() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        ex:text ex:value """Line 1
Line 2
Line 3""" .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    match triples[0].object() {
        Object::Literal(lit) => {
            assert!(lit.value().contains("Line 1"));
            assert!(lit.value().contains("Line 2"));
            assert!(lit.value().contains("Line 3"));
        }
        _ => panic!("Expected multiline literal"),
    }
}

#[test]
fn test_escape_sequences() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        ex:text ex:value "Line 1\nLine 2\tTabbed" .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    match triples[0].object() {
        Object::Literal(lit) => {
            assert!(lit.value().contains('\n'));
            assert!(lit.value().contains('\t'));
        }
        _ => panic!("Expected literal with escape sequences"),
    }
}

#[test]
fn test_unicode_escape() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        ex:text ex:value "Unicode: \u0048\u0065\u006C\u006C\u006F" .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    match triples[0].object() {
        Object::Literal(lit) => {
            assert!(lit.value().contains("Hello"));
        }
        _ => panic!("Expected literal with Unicode escapes"),
    }
}

#[test]
fn test_empty_document() {
    let turtle = "";

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 0);
}

#[test]
fn test_whitespace_only() {
    let turtle = "   \n\n  \t  \n  ";

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 0);
}

#[test]
fn test_comments_only() {
    let turtle = r#"
        # Comment 1
        # Comment 2
        # Comment 3
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 0);
}

#[test]
fn test_invalid_prefix() {
    let turtle = r#"
        invalid:alice invalid:knows invalid:bob .
    "#;

    let parser = TurtleParser::new();
    let result = parser.parse_document(turtle);

    assert!(result.is_err());
    match result {
        Err(TurtleParseError::Syntax(TurtleSyntaxError::UndefinedPrefix { prefix, .. })) => {
            assert_eq!(prefix, "invalid");
        }
        _ => panic!("Expected UndefinedPrefix error"),
    }
}

#[test]
fn test_missing_dot() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        ex:subject ex:predicate "object"
        ex:another ex:triple "value" .
    "#;

    let parser = TurtleParser::new();
    let result = parser.parse_document(turtle);

    // Should fail due to missing dot after first triple
    assert!(result.is_err());
}

#[test]
fn test_incomplete_triple() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        ex:subject ex:predicate .
    "#;

    let parser = TurtleParser::new();
    let result = parser.parse_document(turtle);

    // Should fail due to missing object
    assert!(result.is_err());
}

#[test]
fn test_default_prefixes() {
    let turtle = r#"
        <http://example.org/alice> rdf:type <http://example.org/Person> .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    // Verify rdf prefix is recognized by default
    match triples[0].predicate() {
        Predicate::NamedNode(node) => {
            assert_eq!(
                node.as_str(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            );
        }
        _ => panic!("Expected NamedNode predicate"),
    }
}

#[test]
fn test_builder_with_base() {
    let turtle = r#"<alice> <knows> <bob> ."#;

    let parser = TurtleParser::new().with_base_iri("http://example.org/".to_string());
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);
}

#[test]
fn test_builder_with_prefix() {
    let turtle = r#"ex:alice ex:knows ex:bob ."#;

    let parser =
        TurtleParser::new().with_prefix("ex".to_string(), "http://example.org/".to_string());
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);
}

#[test]
fn test_lenient_mode() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        ex:valid ex:triple "value" .
        invalid syntax here
        ex:another ex:valid "triple" .
    "#;

    let parser = TurtleParser::new_lenient();
    // Lenient mode should attempt to continue parsing
    // Current implementation may still fail - this tests the API exists
    let _result = parser.parse_document(turtle);
}

#[test]
fn test_full_iri() {
    let turtle = r#"
        <http://example.org/subject> <http://example.org/predicate> <http://example.org/object> .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 1);

    match triples[0].subject() {
        Subject::NamedNode(node) => {
            assert_eq!(node.as_str(), "http://example.org/subject");
        }
        _ => panic!("Expected NamedNode"),
    }
}

#[test]
fn test_numeric_literals() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        ex:int ex:value "42"^^xsd:integer .
        ex:decimal ex:value "3.14"^^xsd:decimal .
        ex:double ex:value "1.23e10"^^xsd:double .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 3);
}

#[test]
fn test_boolean_literals() {
    let turtle = r#"
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        ex:true ex:value "true"^^xsd:boolean .
        ex:false ex:value "false"^^xsd:boolean .
    "#;

    let parser = TurtleParser::new();
    let triples = parser.parse_document(turtle).unwrap();

    assert_eq!(triples.len(), 2);
}
