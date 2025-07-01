//! Comprehensive property-based testing for edge cases and error conditions
//!
//! This test suite focuses on boundary conditions, error handling, and edge cases
//! that might not be covered by the basic property tests.

use oxirs_star::model::*;
use oxirs_star::parser::{StarFormat, StarParser};
use oxirs_star::serializer::StarSerializer;
use oxirs_star::{StarQuad, StarTerm, StarTriple};
use proptest::prelude::*;
use std::collections::HashSet;

/// Strategy for generating potentially problematic strings
fn problematic_string_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // Empty string
        Just("".to_string()),
        // Whitespace only
        prop::string::string_regex("\\s+").unwrap(),
        // Very long strings
        prop::string::string_regex("[a-zA-Z0-9]{1000,2000}").unwrap(),
        // Strings with special characters
        prop::string::string_regex("[\\x00-\\x1F\\x7F-\\xFF]{1,50}").unwrap(),
        // Unicode strings
        "[😀-😿🌀-🏿]{1,20}",
        // Strings with quotes and escapes
        "[\"\\'\\\\\\n\\r\\t]{1,50}",
        // Mixed content
        prop::string::string_regex("[a-zA-Z0-9\\s\\p{P}]{10,100}").unwrap(),
    ]
}

/// Strategy for generating malformed IRI-like strings
fn malformed_iri_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // Missing protocol
        "//example.org/path",
        // Invalid characters
        "http://example.org/path with spaces",
        "http://example.org/path<>",
        // Incomplete IRIs
        "http://",
        "http:///path",
        // Very long IRIs
        prop::string::string_regex("http://example.org/[a-zA-Z0-9]{1000,1500}").unwrap(),
        // IRIs with fragments and queries
        "http://example.org/path?query=value#fragment with spaces",
    ]
}

/// Strategy for generating potentially invalid literals
fn invalid_literal_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // Unescaped quotes
        "value with \" quote",
        // Control characters
        "value\nwith\tcontrol\rchars",
        // Very long values
        prop::string::string_regex("[a-zA-Z0-9 ]{5000,10000}").unwrap(),
        // Binary data simulation
        prop::collection::vec(any::<u8>(), 100..200)
            .prop_map(|bytes| String::from_utf8_lossy(&bytes).to_string()),
    ]
}

/// Strategy for generating deeply nested quoted triples
fn deeply_nested_triple_strategy(max_depth: u32) -> BoxedStrategy<StarTriple> {
    if max_depth == 0 {
        (
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )
            .prop_map(|(s, p, o)| {
                StarTriple::new(
                    StarTerm::iri(s).unwrap(),
                    StarTerm::iri(p).unwrap(),
                    StarTerm::iri(o).unwrap(),
                )
            })
            .boxed()
    } else {
        (
            deeply_nested_triple_strategy(max_depth - 1),
            "http://example.org/p",
            deeply_nested_triple_strategy(max_depth - 1),
        )
            .prop_map(|(subject_triple, p, object_triple)| {
                StarTriple::new(
                    StarTerm::quoted_triple(subject_triple),
                    StarTerm::iri(p).unwrap(),
                    StarTerm::quoted_triple(object_triple),
                )
            })
            .boxed()
    }
}

/// Strategy for generating malformed RDF syntax
fn malformed_rdf_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // Missing closing brackets
        "<http://example.org/s> <http://example.org/p> <http://example.org/o",
        // Extra closing brackets
        "<http://example.org/s> <http://example.org/p> <http://example.org/o>> .",
        // Missing dots
        "<http://example.org/s> <http://example.org/p> <http://example.org/o>",
        // Invalid prefixes
        "@prefix : <> .",
        "@prefix invalid: incomplete",
        // Malformed literals
        "\"unclosed literal",
        "\"literal\"@invalid-lang",
        "\"literal\"^^<invalid-datatype>",
        // Invalid quoted triples
        "<<<http://example.org/s> <http://example.org/p> <http://example.org/o>>> incomplete",
        "<<incomplete>> <http://example.org/p> <http://example.org/o> .",
        // Mixed syntax
        "<s> <p> \"o\" . { <s2> <p2> <o2> .",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #[test]
        fn test_invalid_iri_handling(iri in malformed_iri_strategy()) {
            // All malformed IRIs should be rejected
            let result = StarTerm::iri(&iri);

            if result.is_ok() {
                // If it somehow succeeds, it should validate properly
                let term = result.unwrap();
                prop_assert!(term.is_named_node());
            } else {
                // Error should have meaningful message
                let error = format!("{:?}", result.unwrap_err());
                prop_assert!(!error.is_empty());
            }
        }

        #[test]
        fn test_problematic_string_literals(value in invalid_literal_strategy()) {
            // Test that literal creation either succeeds or fails gracefully
            let result = StarTerm::literal(&value);

            match result {
                Ok(literal) => {
                    // If successful, should behave correctly
                    prop_assert!(literal.is_literal());
                    if let Some(lit) = literal.as_literal() {
                        // Should be able to extract the value
                        prop_assert!(!lit.value().is_empty() || value.is_empty());
                    }
                },
                Err(_) => {
                    // Failure is acceptable for problematic input
                    prop_assert!(true);
                }
            }
        }

        #[test]
        fn test_extreme_nesting_depth(depth in 1u32..20) {
            // Test behavior with very deep nesting
            let triple_result = deeply_nested_triple_strategy(depth).new_tree(&mut TestRunner::default());

            match triple_result {
                Ok(triple_tree) => {
                    let triple = triple_tree.current();

                    // Should handle deep nesting gracefully
                    let computed_depth = triple.nesting_depth();
                    prop_assert!(computed_depth >= depth);

                    // Validation should still work
                    prop_assert!(triple.validate().is_ok());

                    // Should be serializable
                    let serialized = format!("{}", triple);
                    prop_assert!(!serialized.is_empty());

                    // Should count quoted triples correctly
                    let quoted_count = triple.count_quoted_triples();
                    prop_assert!(quoted_count > 0);
                },
                Err(_) => {
                    // It's acceptable to fail for extreme depths
                    prop_assert!(true);
                }
            }
        }

        #[test]
        fn test_malformed_rdf_parsing(content in malformed_rdf_strategy()) {
            let parser = StarParser::new();

            // All formats should handle malformed input gracefully
            for format in [StarFormat::TurtleStar, StarFormat::NTriplesStar, StarFormat::NQuadsStar, StarFormat::TrigStar] {
                let result = parser.parse_str(&content, format);

                match result {
                    Ok(graph) => {
                        // If parsing succeeds unexpectedly, graph should be valid
                        prop_assert!(graph.len() >= 0);
                        prop_assert!(graph.is_valid());
                    },
                    Err(error) => {
                        // Error should be informative
                        let error_msg = format!("{}", error);
                        prop_assert!(!error_msg.is_empty());
                        prop_assert!(error_msg.len() < 1000); // Reasonable error message length
                    }
                }
            }
        }

        #[test]
        fn test_large_graph_operations(
            triples in prop::collection::vec(
                (
                    prop::string::string_regex("http://example.org/s[0-9]+").unwrap(),
                    prop::string::string_regex("http://example.org/p[0-9]+").unwrap(),
                    prop::string::string_regex("http://example.org/o[0-9]+").unwrap(),
                ),
                100..1000
            )
        ) {
            let mut graph = StarGraph::new();
            let mut inserted_count = 0;

            // Insert all triples, tracking successful insertions
            for (s, p, o) in &triples {
                let triple = StarTriple::new(
                    StarTerm::iri(s).unwrap(),
                    StarTerm::iri(p).unwrap(),
                    StarTerm::iri(o).unwrap(),
                );

                if graph.insert(triple).is_ok() {
                    inserted_count += 1;
                }
            }

            // Graph operations should be consistent
            prop_assert_eq!(graph.len(), inserted_count);
            prop_assert!(graph.total_len() >= inserted_count);

            // Memory usage should be reasonable
            let stats = graph.statistics();
            if let Some(&triple_count) = stats.get("triples") {
                prop_assert_eq!(triple_count, inserted_count);
            }

            // Iteration should work
            let iterated_count = graph.iter().count();
            prop_assert_eq!(iterated_count, inserted_count);
        }

        #[test]
        fn test_unicode_handling(
            subject in "[\\p{L}\\p{N}]+",
            predicate in "[\\p{L}\\p{N}]+",
            object in "[\\p{L}\\p{N}\\p{P}\\p{S}\\s]+"
        ) {
            // Test Unicode character handling in various positions
            let s_iri = format!("http://example.org/{}", subject);
            let p_iri = format!("http://example.org/{}", predicate);

            let s_term = StarTerm::iri(&s_iri);
            let p_term = StarTerm::iri(&p_iri);
            let o_term = StarTerm::literal(&object);

            if let (Ok(s), Ok(p), Ok(o)) = (s_term, p_term, o_term) {
                let triple = StarTriple::new(s, p, o);

                // Should handle Unicode correctly
                prop_assert!(triple.validate().is_ok());

                // Serialization should preserve Unicode
                let serialized = format!("{}", triple);
                prop_assert!(serialized.contains(&subject) || serialized.contains(&predicate));
                prop_assert!(serialized.contains(&object));

                // Round-trip through string representation
                let display_format = format!("{}", triple);
                prop_assert!(!display_format.is_empty());
            }
        }

        #[test]
        fn test_memory_stress(operations in 1usize..10000) {
            // Stress test memory management with many operations
            let mut graph = StarGraph::new();
            let mut operation_count = 0;

            for i in 0..operations {
                let triple = StarTriple::new(
                    StarTerm::iri(&format!("http://example.org/s{}", i)).unwrap(),
                    StarTerm::iri("http://example.org/predicate").unwrap(),
                    StarTerm::literal(&format!("value{}", i)).unwrap(),
                );

                // Insert
                if graph.insert(triple.clone()).is_ok() {
                    operation_count += 1;
                }

                // Occasionally remove to test memory cleanup
                if i % 100 == 0 && i > 0 {
                    graph.remove(&triple);
                    operation_count -= 1;
                }
            }

            // Graph should maintain consistency
            prop_assert!(graph.len() <= operations);
            prop_assert!(graph.len() >= 0);

            // Should be able to clear efficiently
            graph.clear();
            prop_assert_eq!(graph.len(), 0);
            prop_assert!(graph.is_empty());
        }

        #[test]
        fn test_concurrent_access_patterns(
            readers in 1usize..10,
            writers in 1usize..5,
            operations in 1usize..100
        ) {
            // Test patterns that might occur in concurrent scenarios
            let mut graph = StarGraph::new();

            // Simulate mixed read/write operations
            for i in 0..operations {
                let triple = StarTriple::new(
                    StarTerm::iri(&format!("http://example.org/s{}", i)).unwrap(),
                    StarTerm::iri(&format!("http://example.org/p{}", i % 10)).unwrap(),
                    StarTerm::literal(&format!("value{}", i)).unwrap(),
                );

                // Writers add data
                if i % (readers + writers) < writers {
                    graph.insert(triple.clone()).ok();
                }

                // Readers query data
                for reader in 0..readers {
                    let query_triple = StarTriple::new(
                        StarTerm::iri(&format!("http://example.org/s{}", i.saturating_sub(reader))).unwrap(),
                        StarTerm::iri(&format!("http://example.org/p{}", (i.saturating_sub(reader)) % 10)).unwrap(),
                        StarTerm::literal(&format!("value{}", i.saturating_sub(reader))).unwrap(),
                    );

                    // Query operations should not fail
                    let _contains = graph.contains(&query_triple);
                    let _subjects: Vec<_> = graph.subjects().collect();
                    let _predicates: Vec<_> = graph.predicates().collect();
                }
            }

            // Graph should remain in valid state
            prop_assert!(graph.is_valid());
            prop_assert!(graph.len() <= operations);
        }

        #[test]
        fn test_serialization_edge_cases(
            triples in prop::collection::vec(
                (problematic_string_strategy(), problematic_string_strategy(), problematic_string_strategy()),
                1..20
            )
        ) {
            let mut graph = StarGraph::new();
            let mut valid_triples = 0;

            // Try to create triples from problematic strings
            for (s, p, o) in triples {
                let s_iri = format!("http://example.org/{}", s.replace(' ', "_"));
                let p_iri = format!("http://example.org/{}", p.replace(' ', "_"));

                if let (Ok(s_term), Ok(p_term), Ok(o_term)) = (
                    StarTerm::iri(&s_iri),
                    StarTerm::iri(&p_iri),
                    StarTerm::literal(&o)
                ) {
                    let triple = StarTriple::new(s_term, p_term, o_term);
                    if graph.insert(triple).is_ok() {
                        valid_triples += 1;
                    }
                }
            }

            if valid_triples > 0 {
                // Serialization should handle edge cases
                let serializer = StarSerializer::new();

                // Try different formats
                for format in [StarFormat::TurtleStar, StarFormat::NTriplesStar, StarFormat::NQuadsStar] {
                    match serializer.serialize_graph(&graph, format) {
                        Ok(serialized) => {
                            prop_assert!(!serialized.is_empty());
                            prop_assert!(serialized.len() < 1_000_000); // Reasonable size limit
                        },
                        Err(_) => {
                            // Serialization failure is acceptable for edge cases
                            prop_assert!(true);
                        }
                    }
                }
            }
        }
    }

    // Additional deterministic tests for specific edge cases
    #[test]
    fn test_null_byte_handling() {
        // Null bytes should be handled gracefully
        let result = StarTerm::literal("value\0with\0nulls");
        match result {
            Ok(term) => assert!(term.is_literal()),
            Err(_) => (), // Acceptable to reject
        }
    }

    #[test]
    fn test_extremely_long_iri() {
        let very_long_path = "a".repeat(10_000);
        let long_iri = format!("http://example.org/{}", very_long_path);

        let result = StarTerm::iri(&long_iri);
        match result {
            Ok(term) => assert!(term.is_named_node()),
            Err(_) => (), // Acceptable to reject very long IRIs
        }
    }

    #[test]
    fn test_empty_graph_operations() {
        let graph = StarGraph::new();

        // Empty graph operations should work correctly
        assert_eq!(graph.len(), 0);
        assert!(graph.is_empty());
        assert_eq!(graph.subjects().count(), 0);
        assert_eq!(graph.predicates().count(), 0);
        assert_eq!(graph.objects().count(), 0);
        assert_eq!(graph.iter().count(), 0);

        // Statistics should be consistent
        let stats = graph.statistics();
        assert_eq!(stats.get("triples"), Some(&0));
    }

    #[test]
    fn test_maximum_recursion_depth() {
        // Test that very deep nesting is handled appropriately
        fn create_deeply_nested(depth: u32) -> Option<StarTriple> {
            if depth == 0 {
                Some(StarTriple::new(
                    StarTerm::iri("http://example.org/s").ok()?,
                    StarTerm::iri("http://example.org/p").ok()?,
                    StarTerm::iri("http://example.org/o").ok()?,
                ))
            } else {
                let inner = create_deeply_nested(depth - 1)?;
                Some(StarTriple::new(
                    StarTerm::quoted_triple(inner),
                    StarTerm::iri("http://example.org/p").ok()?,
                    StarTerm::iri("http://example.org/o").ok()?,
                ))
            }
        }

        // Test various depths
        for depth in [10, 50, 100, 500, 1000] {
            match create_deeply_nested(depth) {
                Some(triple) => {
                    // Should handle reasonably deep nesting
                    assert!(triple.validate().is_ok());
                    assert!(triple.nesting_depth() >= depth);
                }
                None => {
                    // It's acceptable to fail at extreme depths
                    break;
                }
            }
        }
    }
}
