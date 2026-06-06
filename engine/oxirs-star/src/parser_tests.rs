//! Tests for RDF-star parsing across all supported formats.
//!
//! Tests that require access to private types (ParseContext, parse_term,
//! tokenize_triple) remain in the inline `#[cfg(test)]` block within parser.rs,
//! where they can access private items via `super::*`.
//!
//! This module contains tests that only use the public API of `StarParser`.

#[cfg(test)]
mod tests {
    use crate::parser::StarParser;
    use crate::parser_ast::StarFormat;
    use crate::StarConfig;

    #[test]
    fn test_simple_triple_parsing() {
        let parser = StarParser::new();
        let data = r#"
            <http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .
        "#;

        let graph = parser.parse_str(data, StarFormat::NTriplesStar).unwrap();
        assert_eq!(graph.len(), 1);

        let triples = graph.triples();
        let triple = &triples[0];
        assert!(triple.subject.is_named_node());
        assert!(triple.predicate.is_named_node());
        assert!(triple.object.is_named_node());
    }

    #[test]
    fn test_quoted_triple_parsing() {
        let parser = StarParser::new();
        let data = r#"
            << <http://example.org/alice> <http://example.org/age> "25" >> <http://example.org/certainty> "0.9" .
        "#;

        let graph = parser.parse_str(data, StarFormat::NTriplesStar).unwrap();
        assert_eq!(graph.len(), 1);

        let triples = graph.triples();
        let triple = &triples[0];
        assert!(triple.subject.is_quoted_triple());
        assert!(triple.predicate.is_named_node());
        assert!(triple.object.is_literal());
    }

    #[test]
    fn test_turtle_star_with_prefixes() {
        let parser = StarParser::new();
        let data = r#"
            @prefix ex: <http://example.org/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .

            ex:alice foaf:knows ex:bob .
            << ex:alice foaf:age "25" >> ex:certainty "high" .
        "#;

        let graph = parser.parse_str(data, StarFormat::TurtleStar).unwrap();
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn test_error_handling() {
        let parser = StarParser::new();

        // Invalid format
        let result = parser.parse_str("invalid data", StarFormat::NTriplesStar);
        assert!(result.is_err());

        // Unclosed quoted triple
        let result = parser.parse_str(
            r#"<< <s> <p> <o> <certainty> "high" ."#,
            StarFormat::NTriplesStar,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_nquads_star_parsing() {
        let parser = StarParser::new();

        // Simple quad
        let nquads = r#"<http://example.org/s> <http://example.org/p> "test" <http://example.org/g> .
<http://example.org/s2> <http://example.org/p2> <http://example.org/o2> ."#;

        let result = parser.parse_str(nquads, StarFormat::NQuadsStar).unwrap();
        assert_eq!(result.quad_len(), 2);
        assert_eq!(result.total_len(), 2);

        // Quad with quoted triple
        let nquads_with_quoted = r#"<< <http://example.org/alice> <http://example.org/says> "hello" >> <http://example.org/certainty> "0.9" <http://example.org/provenance> ."#;

        let result = parser
            .parse_str(nquads_with_quoted, StarFormat::NQuadsStar)
            .unwrap();
        assert_eq!(result.quad_len(), 1);
        assert!(result.count_quoted_triples() > 0);
    }

    #[test]
    fn test_trig_star_parsing() {
        let parser = StarParser::new();

        // Simple TriG with named graphs
        let trig = r#"
@prefix ex: <http://example.org/> .

{
    ex:alice ex:knows ex:bob .
}

ex:graph1 {
    ex:charlie ex:likes ex:dave .
    << ex:charlie ex:likes ex:dave >> ex:certainty "0.8" .
}
"#;

        let result = parser.parse_str(trig, StarFormat::TrigStar).unwrap();
        assert_eq!(result.quad_len(), 3);
        assert_eq!(result.named_graph_names().len(), 1);

        // Test default graph
        let default_triples = result.triples();
        assert_eq!(default_triples.len(), 1);

        // Test named graph
        let graph_name = result.named_graph_names()[0];
        let named_triples = result.named_graph_triples(graph_name).unwrap();
        assert_eq!(named_triples.len(), 2);
    }

    #[test]
    fn test_trig_star_error_recovery() {
        let config = StarConfig {
            strict_mode: false,
            ..Default::default()
        }; // Enable error recovery
        let parser = StarParser::with_config(config);

        // TriG with errors that should be recoverable
        let trig_with_errors = r#"
@prefix ex: <http://example.org/> .

# Valid triple
ex:alice ex:knows ex:bob .

# Invalid triple (missing object) - should be skipped
ex:charlie ex:likes .

# Valid graph block
ex:graph1 {
    ex:dave ex:age "30" .
}

# Unclosed graph block - should report error but continue
ex:graph2 {
    ex:eve ex:age "25" .
    # Missing closing brace

# Valid triple after error
ex:frank ex:knows ex:grace .
"#;

        let result = parser.parse_str(trig_with_errors, StarFormat::TrigStar);
        // Should parse successfully with errors logged
        assert!(result.is_ok());
        let graph = result.unwrap();
        // Should have parsed the valid triples
        assert!(graph.quad_len() >= 3); // At least the valid triples
    }

    #[test]
    fn test_nquads_star_with_blank_nodes() {
        let parser = StarParser::new();

        let nquads = r#"_:b1 <http://example.org/p> "test" <http://example.org/g> .
<< _:b1 <http://example.org/says> "hello" >> <http://example.org/certainty> "0.9" _:g1 ."#;

        let result = parser.parse_str(nquads, StarFormat::NQuadsStar).unwrap();
        assert_eq!(result.quad_len(), 2);
    }
}
