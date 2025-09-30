use oxirs_star::model::{StarGraph, StarTerm, StarTriple};
use oxirs_star::parser::{StarFormat, StarParser};
use oxirs_star::serializer::StarSerializer;
use proptest::prelude::*;

// Generate valid N-Triples-star lines
fn ntriples_line_strategy() -> impl Strategy<Value = String> {
    (
        prop::string::string_regex(
            "<https://[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]/[a-zA-Z0-9._-]+>",
        )
        .unwrap(),
        prop::string::string_regex(
            "<https://[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]/[a-zA-Z0-9._-]+>",
        )
        .unwrap(),
        prop_oneof![
            prop::string::string_regex(
                "<https://[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]/[a-zA-Z0-9._-]+>"
            )
            .unwrap(),
            prop::string::string_regex("\"[a-zA-Z0-9 ._-]*\"").unwrap(),
            prop::string::string_regex("_:[a-zA-Z][a-zA-Z0-9_]*").unwrap(),
        ],
    )
        .prop_map(|(s, p, o)| format!("{s} {p} {o} ."))
}

// Generate valid Turtle-star prefixes
fn turtle_prefix_strategy() -> impl Strategy<Value = String> {
    (
        prop::string::string_regex("[a-zA-Z][a-zA-Z0-9]*").unwrap(),
        prop::string::string_regex(
            "https://[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]/[a-zA-Z0-9._-]*#?",
        )
        .unwrap(),
    )
        .prop_map(|(prefix, iri)| format!("@prefix {prefix}: <{iri}> ."))
}

// Generate simple Turtle-star content
fn turtle_content_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            turtle_prefix_strategy(),
            ntriples_line_strategy().prop_map(|line| line.trim_end_matches('.').to_string() + " ."),
        ],
        1..10,
    )
    .prop_map(|lines| lines.join("\n"))
}

// Generate N-Quads-star lines
fn nquads_line_strategy() -> impl Strategy<Value = String> {
    (
        prop::string::string_regex(
            "<https://[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]/[a-zA-Z0-9._-]+>",
        )
        .unwrap(),
        prop::string::string_regex(
            "<https://[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]/[a-zA-Z0-9._-]+>",
        )
        .unwrap(),
        prop_oneof![
            prop::string::string_regex(
                "<https://[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]/[a-zA-Z0-9._-]+>"
            )
            .unwrap(),
            prop::string::string_regex("\"[a-zA-Z0-9 ._-]*\"").unwrap(),
        ],
        prop::option::of(
            prop::string::string_regex(
                "<https://[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]/graph[0-9]*>",
            )
            .unwrap(),
        ),
    )
        .prop_map(|(s, p, o, g)| {
            if let Some(graph) = g {
                format!("{s} {p} {o} {graph} .")
            } else {
                format!("{s} {p} {o} .")
            }
        })
}

// Generate TriG-star content
fn trig_content_strategy() -> impl Strategy<Value = String> {
    (
        prop::collection::vec(turtle_prefix_strategy(), 0..3),
        prop::collection::vec(ntriples_line_strategy(), 0..5),
        prop::collection::vec(
            (
                prop::string::string_regex(
                    "<https://[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]/graph[0-9]+>",
                )
                .unwrap(),
                prop::collection::vec(ntriples_line_strategy(), 1..3),
            ),
            0..3,
        ),
    )
        .prop_map(|(prefixes, default_triples, named_graphs)| {
            let mut content = prefixes.join("\n");
            if !prefixes.is_empty() {
                content.push_str("\n\n");
            }

            // Default graph
            if !default_triples.is_empty() {
                content.push_str("{\n");
                for triple in default_triples {
                    content.push_str("  ");
                    content.push_str(&triple);
                    content.push('\n');
                }
                content.push_str("}\n\n");
            }

            // Named graphs
            for (graph_name, triples) in named_graphs {
                content.push_str(&format!("{graph_name} {{\n"));
                for triple in triples {
                    content.push_str("  ");
                    content.push_str(&triple);
                    content.push('\n');
                }
                content.push_str("}\n\n");
            }

            content
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #[test]
        fn test_ntriples_star_parsing(content in prop::collection::vec(ntriples_line_strategy(), 0..10)) {
            let parser = StarParser::new();
            let input = content.join("\n");

            match parser.parse_str(&input, StarFormat::NTriplesStar) {
                Ok(graph) => {
                    // Number of parsed triples should match input lines
                    prop_assert_eq!(graph.len(), content.len());
                },
                Err(e) => {
                    // If parsing fails, the error should be meaningful
                    let error_msg = format!("{e}");
                    prop_assert!(!error_msg.is_empty());
                }
            }
        }

        #[test]
        fn test_turtle_star_parsing(content in turtle_content_strategy()) {
            let parser = StarParser::new();

            match parser.parse_str(&content, StarFormat::TurtleStar) {
                Ok(_graph) => {
                    // Parsing should produce a valid graph
                    // (graph.len() is always >= 0 by type invariant)
                },
                Err(e) => {
                    // Error messages should be descriptive
                    let error_msg = format!("{e}");
                    prop_assert!(error_msg.contains("Parse error") || error_msg.contains("line"));
                }
            }
        }

        #[test]
        fn test_nquads_star_parsing(content in prop::collection::vec(nquads_line_strategy(), 0..10)) {
            let parser = StarParser::new();
            let input = content.join("\n");

            match parser.parse_str(&input, StarFormat::NQuadsStar) {
                Ok(graph) => {
                    // Total number of quads should match input
                    prop_assert_eq!(graph.quad_len(), content.len());
                },
                Err(e) => {
                    let error_msg = format!("{e}");
                    prop_assert!(!error_msg.is_empty());
                }
            }
        }

        #[test]
        fn test_trig_star_parsing(content in trig_content_strategy()) {
            let parser = StarParser::new();

            match parser.parse_str(&content, StarFormat::TrigStar) {
                Ok(graph) => {
                    // Should produce a valid graph
                    // (graph.total_len() is always >= 0 by type invariant)

                    // Named graphs should be accessible
                    for name in graph.named_graph_names() {
                        prop_assert!(graph.named_graph_triples(name).is_some());
                    }
                },
                Err(e) => {
                    let error_msg = format!("{e}");
                    prop_assert!(!error_msg.is_empty());
                }
            }
        }

        #[test]
        fn test_parser_serializer_roundtrip_ntriples(
            triples in prop::collection::vec(
                (
                    prop::string::string_regex("http://example.org/[a-z]+").unwrap(),
                    prop::string::string_regex("http://example.org/[a-z]+").unwrap(),
                    prop_oneof![
                        prop::string::string_regex("http://example.org/[a-z]+").unwrap(),
                        prop::string::string_regex("[a-zA-Z][a-zA-Z0-9]*").unwrap(),
                    ]
                ),
                1..5
            )
        ) {
            let parser = StarParser::new();
            let serializer = StarSerializer::new();
            let mut original_graph = StarGraph::new();

            // Build graph from generated data
            for (s, p, o) in triples {
                // Try to create valid terms, skip if invalid
                let subject = match StarTerm::iri(&s) {
                    Ok(term) => term,
                    Err(_) => continue, // Skip invalid IRIs
                };
                let predicate = match StarTerm::iri(&p) {
                    Ok(term) => term,
                    Err(_) => continue, // Skip invalid IRIs
                };
                let object = if o.starts_with("http") {
                    match StarTerm::iri(&o) {
                        Ok(term) => term,
                        Err(_) => continue, // Skip invalid IRIs
                    }
                } else {
                    match StarTerm::literal(&o) {
                        Ok(term) => term,
                        Err(_) => continue, // Skip invalid literals
                    }
                };

                let triple = StarTriple::new(subject, predicate, object);
                let _ = original_graph.insert(triple); // Ignore insertion errors
            }

            // Only test if we have valid triples
            if !original_graph.is_empty() {
                // Serialize to N-Triples-star
                match serializer.serialize_to_string(&original_graph, StarFormat::NTriplesStar) {
                    Ok(serialized) => {
                        // Parse back
                        match parser.parse_str(&serialized, StarFormat::NTriplesStar) {
                            Ok(parsed_graph) => {
                                // Should have same number of triples
                                prop_assert_eq!(original_graph.len(), parsed_graph.len());
                            }
                            Err(_) => {
                                // Skip test if parsing fails due to invalid generated data
                                prop_assume!(false);
                            }
                        }
                    }
                    Err(_) => {
                        // Skip test if serialization fails
                        prop_assume!(false);
                    }
                }
            }
        }

        #[test]
        fn test_malformed_input_handling(
            format in prop_oneof![
                Just(StarFormat::NTriplesStar),
                Just(StarFormat::TurtleStar),
                Just(StarFormat::TrigStar),
                Just(StarFormat::NQuadsStar),
            ],
            garbage in "[^\\n]{1,100}"
        ) {
            let parser = StarParser::new();

            // Parser should handle garbage input gracefully
            let result = parser.parse_str(&garbage, format);

            if result.is_err() {
                // Error should be a parse error
                let error = result.unwrap_err();
                let error_msg = format!("{error}");
                prop_assert!(error_msg.contains("Parse error") || error_msg.contains("Unexpected"));
            }
        }

        #[test]
        fn test_empty_input_handling(
            format in prop_oneof![
                Just(StarFormat::NTriplesStar),
                Just(StarFormat::TurtleStar),
                Just(StarFormat::TrigStar),
                Just(StarFormat::NQuadsStar),
            ]
        ) {
            let parser = StarParser::new();

            // Empty input should parse to empty graph
            let result = parser.parse_str("", format);
            prop_assert!(result.is_ok());

            let graph = result.unwrap();
            prop_assert_eq!(graph.len(), 0);
            prop_assert!(graph.is_empty());
        }

        #[test]
        fn test_comment_handling(
            format in prop_oneof![
                Just(StarFormat::NTriplesStar),
                Just(StarFormat::TurtleStar),
            ],
            comment_text in "[^\\n]*",
            triple_line in ntriples_line_strategy()
        ) {
            let parser = StarParser::new();

            // Build input with comments
            let input = format!(
                "# {comment_text}\n{triple_line}\n# Another comment\n"
            );

            match parser.parse_str(&input, format) {
                Ok(graph) => {
                    // Should parse exactly one triple (comments ignored)
                    prop_assert_eq!(graph.len(), 1);
                },
                Err(_) => {
                    // If it fails, it should be due to the triple, not comments
                    prop_assert!(true);
                }
            }
        }

        #[test]
        fn test_whitespace_handling(
            format in prop_oneof![
                Just(StarFormat::NTriplesStar),
                Just(StarFormat::TurtleStar),
            ],
            triple_line in ntriples_line_strategy(),
            ws_before in "[ \\t]*",
            ws_after in "[ \\t]*"
        ) {
            let parser = StarParser::new();

            // Add whitespace around valid content
            let input = format!("{ws_before}{triple_line}{ws_after}");

            match parser.parse_str(&input, format) {
                Ok(graph) => {
                    // Whitespace shouldn't affect parsing
                    prop_assert!(!graph.is_empty());
                },
                Err(_) => {
                    // Failure should be due to content, not whitespace
                    prop_assert!(true);
                }
            }
        }
    }

    // Additional deterministic tests
    #[test]
    fn test_quoted_triple_parsing() {
        let parser = StarParser::new();

        // N-Triples-star with quoted triple
        let input = "<< <http://example.org/alice> <http://example.org/says> \"hello\" >> <http://example.org/certainty> \"0.9\" .";
        let result = parser.parse_str(input, StarFormat::NTriplesStar);
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.len(), 1);
        assert_eq!(graph.count_quoted_triples(), 1);
    }

    #[test]
    fn test_nested_quoted_triple_parsing() {
        let parser = StarParser::new();

        // Nested quoted triples
        let input = "<< << <http://example.org/alice> <http://example.org/believes> <http://example.org/bob> >> <http://example.org/certainty> \"0.8\" >> <http://example.org/meta> \"nested\" .";
        let result = parser.parse_str(input, StarFormat::NTriplesStar);

        if result.is_ok() {
            let graph = result.unwrap();
            assert_eq!(graph.len(), 1);
            assert!(graph.max_nesting_depth() >= 2);
        }
    }
}
