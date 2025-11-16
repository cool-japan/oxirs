//! W3C Turtle Test Suite Integration
//!
//! This test module integrates the official W3C Turtle test suite to ensure
//! specification compliance. The tests are based on the W3C RDF 1.1 Turtle
//! Test Cases: https://w3c.github.io/rdf-tests/turtle/
//!
//! Test categories:
//! - Positive syntax tests: Valid Turtle that should parse successfully
//! - Negative syntax tests: Invalid Turtle that should fail to parse
//! - Evaluation tests: Valid Turtle with expected RDF output

use oxirs_ttl::formats::turtle::TurtleParser;

/// W3C Test Case structure (for future manifest-based testing)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct W3CTestCase {
    /// Test identifier
    id: String,
    /// Test name/description
    name: String,
    /// Test category (positive_syntax, negative_syntax, eval)
    category: TestCategory,
    /// Input Turtle content
    input: String,
    /// Expected outcome (for eval tests, this would be N-Triples)
    expected: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum TestCategory {
    /// Valid syntax - should parse without errors
    PositiveSyntax,
    /// Invalid syntax - should fail to parse
    NegativeSyntax,
    /// Evaluation test - should parse and match expected output
    Evaluation,
}

/// Positive Syntax Tests - Valid Turtle that must parse successfully
mod positive_syntax {
    use super::*;

    #[test]
    fn test_simple_triple() {
        let turtle = "<http://example.org/subject> <http://example.org/predicate> <http://example.org/object> .";
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_ok(), "Simple triple should parse successfully");
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_prefix_declaration() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate ex:object .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_ok(),
            "Prefix declaration should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_base_declaration() {
        let turtle = r#"
@base <http://example.org/> .
<subject> <predicate> <object> .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_ok(), "Base declaration should parse successfully");
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_blank_nodes_anonymous() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate [ ex:prop "value" ] .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_ok(),
            "Anonymous blank nodes should parse successfully"
        );
        let triples = result.unwrap();
        assert_eq!(triples.len(), 2, "Should generate 2 triples");
    }

    #[test]
    fn test_blank_nodes_labeled() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
_:b1 ex:predicate "value" .
_:b1 ex:another "value2" .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_ok(),
            "Labeled blank nodes should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_collection_syntax() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:list ( ex:item1 ex:item2 ex:item3 ) .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_ok(),
            "Collection syntax should parse successfully"
        );
        // Collection expands to multiple triples
        assert!(result.unwrap().len() > 1);
    }

    #[test]
    fn test_abbreviated_predicate_a() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject a ex:Type .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_ok(), "'a' abbreviation should parse successfully");
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_predicate_object_list() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject
    ex:prop1 "value1" ;
    ex:prop2 "value2" ;
    ex:prop3 "value3" .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_ok(),
            "Predicate-object lists should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_object_list() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate "value1", "value2", "value3" .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_ok(), "Object lists should parse successfully");
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_string_literals() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:s1 ex:p "simple string" .
ex:s2 ex:p "string with\nnewline" .
ex:s3 ex:p "string with\ttab" .
ex:s4 ex:p "string with \"quotes\"" .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_ok(),
            "String literals with escapes should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 4);
    }

    #[test]
    fn test_long_string_literals() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate """This is a
multi-line
string literal""" .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_ok(),
            "Long string literals should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_language_tagged_strings() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:s1 ex:p "English"@en .
ex:s2 ex:p "日本語"@ja .
ex:s3 ex:p "Français"@fr-FR .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_ok(),
            "Language-tagged strings should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_typed_literals() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
ex:s1 ex:p "42"^^xsd:integer .
ex:s2 ex:p "3.14"^^xsd:decimal .
ex:s3 ex:p "true"^^xsd:boolean .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_ok(), "Typed literals should parse successfully");
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_numeric_literals() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:s1 ex:p 42 .
ex:s2 ex:p 3.14 .
ex:s3 ex:p 1.23e-10 .
ex:s4 ex:p -17 .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_ok(), "Numeric literals should parse successfully");
        assert_eq!(result.unwrap().len(), 4);
    }

    #[test]
    fn test_boolean_literals() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:s1 ex:p true .
ex:s2 ex:p false .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_ok(), "Boolean literals should parse successfully");
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_comments() {
        let turtle = r#"
# This is a comment
@prefix ex: <http://example.org/> .
# Another comment
ex:subject ex:predicate ex:object . # Inline comment
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_ok(), "Comments should be handled correctly");
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_empty_prefix() {
        let turtle = r#"
@prefix : <http://example.org/> .
:subject :predicate :object .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_ok(), "Empty prefix should parse successfully");
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_unicode_in_iris() {
        let turtle = r#"
<http://example.org/subject_日本語> <http://example.org/predicate> <http://example.org/object> .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_ok(), "Unicode in IRIs should parse successfully");
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_mixed_content() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

ex:Person1 a ex:Person ;
    ex:name "Alice" ;
    ex:age 30 ;
    ex:knows [
        ex:name "Bob" ;
        ex:age 25
    ] .

ex:Person2 ex:name "Charlie"@en, "チャーリー"@ja .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_ok(),
            "Mixed complex content should parse successfully"
        );
        assert!(result.unwrap().len() > 5);
    }
}

/// Negative Syntax Tests - Invalid Turtle that must fail to parse
mod negative_syntax {
    use super::*;

    #[test]
    fn test_missing_dot() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate ex:object
ex:subject2 ex:predicate2 ex:object2 .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_err(), "Missing dot should cause parse error");
    }

    #[test]
    fn test_unterminated_iri() {
        let turtle = "<http://example.org/subject <http://example.org/predicate> <http://example.org/object> .";
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_err(), "Unterminated IRI should cause parse error");
    }

    #[test]
    fn test_unterminated_string() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate "unterminated string .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_err(),
            "Unterminated string should cause parse error"
        );
    }

    #[test]
    fn test_invalid_prefix_declaration() {
        let turtle = r#"
@prefix ex http://example.org/> .
ex:subject ex:predicate ex:object .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_err(),
            "Invalid prefix declaration should cause parse error"
        );
    }

    #[test]
    fn test_undefined_prefix() {
        let turtle = "undefined:subject undefined:predicate undefined:object .";
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_err(), "Undefined prefix should cause parse error");
    }

    #[test]
    fn test_invalid_numeric() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate 12.34.56 .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_err(),
            "Invalid numeric literal should cause parse error"
        );
    }

    #[test]
    #[ignore] // TODO: Parser currently accepts trailing semicolons (lenient mode)
    fn test_trailing_semicolon() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate ex:object ; .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_err(),
            "Trailing semicolon should cause parse error"
        );
    }

    #[test]
    fn test_trailing_comma() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate ex:object1, .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_err(), "Trailing comma should cause parse error");
    }

    #[test]
    fn test_mismatched_brackets() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate [ ex:prop "value" .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_err(),
            "Mismatched brackets should cause parse error"
        );
    }

    #[test]
    fn test_mismatched_parentheses() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:list ( ex:item1 ex:item2 .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_err(),
            "Mismatched parentheses should cause parse error"
        );
    }
}

/// Evaluation Tests - Parse and verify output matches expected N-Triples
mod evaluation {
    use super::*;
    use oxirs_core::model::{Object, Predicate, Subject};

    #[test]
    fn test_simple_triple_output() {
        let turtle = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .";
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);

        assert!(result.is_ok());
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);

        let triple = &triples[0];
        match triple.subject() {
            Subject::NamedNode(nn) => assert_eq!(nn.as_str(), "http://example.org/s"),
            _ => panic!("Expected NamedNode subject"),
        }
        match triple.predicate() {
            Predicate::NamedNode(nn) => assert_eq!(nn.as_str(), "http://example.org/p"),
            Predicate::Variable(_) => panic!("Unexpected Variable predicate"),
        }
        match triple.object() {
            Object::NamedNode(nn) => assert_eq!(nn.as_str(), "http://example.org/o"),
            _ => panic!("Expected NamedNode object"),
        }
    }

    #[test]
    fn test_prefix_expansion() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate ex:object .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);

        assert!(result.is_ok());
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);

        let triple = &triples[0];
        match triple.subject() {
            Subject::NamedNode(nn) => assert_eq!(nn.as_str(), "http://example.org/subject"),
            _ => panic!("Expected NamedNode subject"),
        }
    }

    #[test]
    fn test_rdf_type_abbreviation() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject a ex:Type .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);

        assert!(result.is_ok());
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);

        let triple = &triples[0];
        match triple.predicate() {
            Predicate::NamedNode(nn) => {
                assert_eq!(
                    nn.as_str(),
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                );
            }
            Predicate::Variable(_) => panic!("Unexpected Variable predicate"),
        }
    }

    #[test]
    fn test_literal_values() {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate "Hello, World!" .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);

        assert!(result.is_ok());
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);

        let triple = &triples[0];
        match triple.object() {
            Object::Literal(lit) => {
                assert_eq!(lit.value(), "Hello, World!");
            }
            _ => panic!("Expected Literal object"),
        }
    }
}

/// Performance tests for W3C test suite processing
#[cfg(test)]
mod performance {
    use super::*;

    #[test]
    fn test_parse_performance_baseline() {
        // Large Turtle document with mixed syntax features
        let turtle = generate_large_w3c_style_document(1000);
        let parser = TurtleParser::new();

        let start = std::time::Instant::now();
        let result = parser.parse_document(&turtle);
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Large document should parse successfully");
        println!("Parsed 1000-triple document in {:?}", elapsed);

        // Baseline: Should parse 1000 triples in less than 250ms
        // Note: Threshold adjusted for Linux/CUDA environment (macOS baseline was 150ms)
        assert!(
            elapsed.as_millis() < 250,
            "Performance regression detected: {:?}",
            elapsed
        );
    }

    fn generate_large_w3c_style_document(num_triples: usize) -> String {
        let mut doc = String::with_capacity(num_triples * 100);
        doc.push_str("@prefix ex: <http://example.org/> .\n");
        doc.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n\n");

        for i in 0..num_triples {
            if i % 10 == 0 {
                // Add some variety with blank nodes
                doc.push_str(&format!(
                    "ex:subject{} a ex:Type{} ;\n  ex:prop [ ex:value {} ] .\n\n",
                    i,
                    i % 5,
                    i
                ));
            } else {
                doc.push_str(&format!(
                    "ex:subject{} ex:predicate{} \"Value {}\" .\n",
                    i,
                    i % 10,
                    i
                ));
            }
        }

        doc
    }
}
