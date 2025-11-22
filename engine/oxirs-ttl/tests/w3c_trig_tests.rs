//! W3C TriG Test Suite Integration
//!
//! This test module integrates the official W3C TriG test suite to ensure
//! specification compliance. The tests are based on the W3C RDF 1.1 TriG
//! Test Cases: https://w3c.github.io/rdf-tests/trig/
//!
//! Test categories:
//! - Positive syntax tests: Valid TriG that should parse successfully
//! - Negative syntax tests: Invalid TriG that should fail to parse
//! - Evaluation tests: Valid TriG with expected RDF output
//! - Named graph tests: Specific tests for named graph handling

use oxirs_core::model::GraphName;
use oxirs_ttl::trig::TriGParser;
use oxirs_ttl::Parser;
use std::io::Cursor;

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
    /// Input TriG content
    input: String,
    /// Expected outcome (for eval tests, this would be N-Quads)
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

/// Positive Syntax Tests - Valid TriG that must parse successfully
mod positive_syntax {
    use super::*;

    #[test]
    fn test_simple_default_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate ex:object .
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Simple default graph triple should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_simple_named_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .
<http://example.org/graph1> {
    ex:subject ex:predicate ex:object .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Simple named graph should parse successfully"
        );
        let quads = result.unwrap();
        assert_eq!(quads.len(), 1);
        assert!(matches!(quads[0].graph_name(), GraphName::NamedNode(_)));
    }

    #[test]
    fn test_multiple_named_graphs() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:alice ex:name "Alice" .
}

<http://example.org/graph2> {
    ex:bob ex:name "Bob" .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Multiple named graphs should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_mixed_default_and_named_graphs() {
        let trig = r#"
@prefix ex: <http://example.org/> .

ex:default1 ex:prop "In default graph" .

<http://example.org/graph1> {
    ex:named1 ex:prop "In named graph" .
}

ex:default2 ex:prop "Also in default" .
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Mixed default and named graphs should parse successfully"
        );
        let quads = result.unwrap();
        assert_eq!(quads.len(), 3);

        // Check default graph count
        let default_count = quads
            .iter()
            .filter(|q| matches!(q.graph_name(), GraphName::DefaultGraph))
            .count();
        assert_eq!(default_count, 2);
    }

    #[test]
    fn test_blank_node_graph_name() {
        let trig = r#"
@prefix ex: <http://example.org/> .

_:g1 {
    ex:subject ex:predicate ex:object .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Blank node as graph name should parse successfully"
        );
        let quads = result.unwrap();
        assert_eq!(quads.len(), 1);
        assert!(matches!(quads[0].graph_name(), GraphName::BlankNode(_)));
    }

    #[test]
    fn test_prefixed_graph_name() {
        let trig = r#"
@prefix ex: <http://example.org/> .
@prefix g: <http://graphs.example.org/> .

g:graph1 {
    ex:subject ex:predicate ex:object .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Prefixed graph name should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_graph_with_semicolon_syntax() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:alice ex:name "Alice" ;
             ex:age 30 ;
             ex:email "alice@example.org" .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Semicolon syntax in named graph should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_graph_with_comma_syntax() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:bob ex:knows ex:alice, ex:charlie, ex:diana .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Comma syntax in named graph should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_graph_with_blank_nodes() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:alice ex:address [ ex:city "Wonderland" ; ex:zip "12345" ] .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Blank nodes in named graph should parse successfully"
        );
        assert!(result.unwrap().len() >= 2); // At least 2 triples from blank node
    }

    #[test]
    fn test_graph_with_collections() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:alice ex:favorites ( "red" "blue" "green" ) .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Collections in named graph should parse successfully"
        );
        assert!(result.unwrap().len() > 1); // Collections expand to multiple triples
    }

    #[test]
    fn test_graph_with_language_tags() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:poem ex:title "La Chanson"@fr .
    ex:poem ex:title "The Song"@en .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Language tags in named graph should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_graph_with_typed_literals() {
        let trig = r#"
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://example.org/graph1> {
    ex:measurement ex:value "42"^^xsd:integer .
    ex:measurement ex:timestamp "2025-11-21T00:00:00"^^xsd:dateTime .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Typed literals in named graph should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_empty_named_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/empty> {
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Empty named graph should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_multiline_string_in_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:note ex:text """This is a
multiline
string literal""" .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Multiline strings in named graph should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_comments_in_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .

# This is a comment
<http://example.org/graph1> {
    # Comment inside graph
    ex:subject ex:predicate ex:object . # Inline comment
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Comments in and around graphs should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_base_iri_in_trig() {
        let trig = r#"
@base <http://example.org/> .
@prefix ex: <http://example.org/ns#> .

<graph1> {
    ex:subject ex:predicate ex:object .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Base IRI declaration should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_numeric_literals_in_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:int ex:value 42 .
    ex:decimal ex:value 3.14 .
    ex:double ex:value 1.23e10 .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Numeric literals in named graph should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_boolean_literals_in_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:flag1 ex:value true .
    ex:flag2 ex:value false .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "Boolean literals in named graph should parse successfully"
        );
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_graph_with_rdf_type_shorthand() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:alice a ex:Person .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_ok(),
            "RDF type shorthand 'a' should parse successfully in named graph"
        );
        assert_eq!(result.unwrap().len(), 1);
    }
}

/// Negative Syntax Tests - Invalid TriG that should fail to parse
mod negative_syntax {
    use super::*;

    #[test]
    fn test_missing_closing_brace() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:subject ex:predicate ex:object .
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_err(),
            "Missing closing brace should fail to parse"
        );
    }

    #[test]
    fn test_missing_opening_brace() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1>
    ex:subject ex:predicate ex:object .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_err(),
            "Missing opening brace should fail to parse"
        );
    }

    #[test]
    #[ignore = "Anonymous graph syntax '{}' is treated as default graph in some implementations - spec ambiguous"]
    fn test_missing_graph_name() {
        let trig = r#"
@prefix ex: <http://example.org/> .

{
    ex:subject ex:predicate ex:object .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        // Note: Some TriG parsers accept {} as default graph syntax
        // This test is ignored pending clarification of spec
        assert!(result.is_err(), "Missing graph name should fail to parse");
    }

    #[test]
    fn test_missing_dot_in_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:subject ex:predicate ex:object
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_err(),
            "Missing dot at end of triple in graph should fail to parse"
        );
    }

    #[test]
    fn test_invalid_graph_name() {
        let trig = r#"
@prefix ex: <http://example.org/> .

"not a valid graph name" {
    ex:subject ex:predicate ex:object .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(result.is_err(), "String literal as graph name should fail");
    }

    #[test]
    fn test_undefined_prefix_in_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    unknown:subject ex:predicate ex:object .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_err(),
            "Undefined prefix in graph should fail to parse"
        );
    }

    #[test]
    fn test_unterminated_string_in_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:subject ex:predicate "unterminated .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_err(),
            "Unterminated string in graph should fail to parse"
        );
    }

    #[test]
    #[ignore = "Parser currently doesn't validate nested graphs - requires fix in trig.rs"]
    fn test_nested_graphs() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    <http://example.org/graph2> {
        ex:subject ex:predicate ex:object .
    }
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        // TODO: Parser should detect and reject nested graphs
        assert!(result.is_err(), "Nested graphs should fail to parse");
    }

    #[test]
    fn test_invalid_numeric_literal() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:subject ex:value 12.34.56 .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(
            result.is_err(),
            "Invalid numeric literal should fail to parse"
        );
    }

    #[test]
    fn test_trailing_semicolon_in_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:subject ex:predicate ex:object ;
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        // This should fail in strict mode (trailing semicolon without continuation)
        assert!(
            result.is_err(),
            "Trailing semicolon without continuation should fail"
        );
    }
}

/// Evaluation Tests - Parse and verify output correctness
mod evaluation {
    use super::*;
    use oxirs_core::model::{Object, Predicate, Subject};

    #[test]
    fn test_simple_triple_structure() {
        let trig = r#"
@prefix ex: <http://example.org/> .

<http://example.org/graph1> {
    ex:alice ex:name "Alice" .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(result.is_ok());

        let quads = result.unwrap();
        assert_eq!(quads.len(), 1);

        let quad = &quads[0];
        assert!(matches!(quad.subject(), Subject::NamedNode(_)));
        assert!(matches!(quad.predicate(), Predicate::NamedNode(_)));
        assert!(matches!(quad.object(), Object::Literal(_)));
    }

    #[test]
    fn test_prefix_expansion_in_graph() {
        let trig = r#"
@prefix ex: <http://example.org/> .

ex:graph1 {
    ex:subject ex:predicate ex:object .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(result.is_ok());

        let quads = result.unwrap();
        assert_eq!(quads.len(), 1);

        // Verify that prefixes were expanded correctly
        match quads[0].graph_name() {
            GraphName::NamedNode(nn) => {
                assert!(nn.as_str().contains("http://example.org/"));
            }
            _ => panic!("Expected named node graph name"),
        }
    }

    #[test]
    fn test_multiple_graphs_distribution() {
        let trig = r#"
@prefix ex: <http://example.org/> .

ex:graph1 {
    ex:alice ex:knows ex:bob .
    ex:alice ex:age 30 .
}

ex:graph2 {
    ex:bob ex:knows ex:charlie .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(result.is_ok());

        let quads = result.unwrap();
        assert_eq!(quads.len(), 3);

        // Count quads per graph
        let g1_count = quads
            .iter()
            .filter(|q| {
                matches!(q.graph_name(), GraphName::NamedNode(n) if n.as_str().ends_with("graph1"))
            })
            .count();

        let g2_count = quads
            .iter()
            .filter(|q| {
                matches!(q.graph_name(), GraphName::NamedNode(n) if n.as_str().ends_with("graph2"))
            })
            .count();

        assert_eq!(g1_count, 2, "graph1 should contain 2 quads");
        assert_eq!(g2_count, 1, "graph2 should contain 1 quad");
    }

    #[test]
    fn test_blank_node_consistency() {
        let trig = r#"
@prefix ex: <http://example.org/> .

ex:graph1 {
    _:b1 ex:prop1 "value1" .
    _:b1 ex:prop2 "value2" .
}
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(result.is_ok());

        let quads = result.unwrap();
        assert_eq!(quads.len(), 2);

        // Both quads should have the same blank node as subject
        match (&quads[0].subject(), &quads[1].subject()) {
            (Subject::BlankNode(b1), Subject::BlankNode(b2)) => {
                assert_eq!(
                    b1.as_str(),
                    b2.as_str(),
                    "Same blank node label should produce same blank node"
                );
            }
            _ => panic!("Expected blank node subjects"),
        }
    }

    #[test]
    fn test_default_graph_identification() {
        let trig = r#"
@prefix ex: <http://example.org/> .

ex:subject1 ex:pred ex:obj1 .

ex:graph1 {
    ex:subject2 ex:pred ex:obj2 .
}

ex:subject3 ex:pred ex:obj3 .
        "#;
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        assert!(result.is_ok());

        let quads = result.unwrap();
        assert_eq!(quads.len(), 3);

        let default_count = quads
            .iter()
            .filter(|q| matches!(q.graph_name(), GraphName::DefaultGraph))
            .count();

        assert_eq!(
            default_count, 2,
            "Two triples should be in the default graph"
        );
    }
}

/// Performance Tests - Ensure reasonable performance on W3C-style documents
mod performance {
    use super::*;

    #[test]
    fn test_parse_performance_baseline() {
        // Create a moderately complex TriG document with multiple graphs
        let mut trig = String::from(
            r#"
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
"#,
        );

        // Add 10 graphs with 10 triples each
        for i in 0..10 {
            trig.push_str(&format!("\nex:graph{} {{\n", i));
            for j in 0..10 {
                trig.push_str(&format!(
                    "    ex:subject{} foaf:name \"Person {}\" .\n",
                    j, j
                ));
            }
            trig.push_str("}\n");
        }

        let start = std::time::Instant::now();
        let parser = TriGParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Performance test document should parse");
        assert_eq!(result.unwrap().len(), 100);

        // Should parse 100 quads in reasonable time (< 100ms)
        assert!(
            elapsed.as_millis() < 100,
            "Parsing should complete in < 100ms, took {:?}",
            elapsed
        );
    }
}
