//! Utility functions for RDF-star serialization

use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::parser::StarFormat;
use crate::{StarError, StarResult};
use super::super::config::SerializationOptions;
use super::StarSerializer;

impl StarSerializer {
    /// Check if a graph is suitable for pretty printing
    pub fn can_pretty_print(&self, graph: &StarGraph) -> bool {
        // Simple heuristic: if graph is small and doesn't have deeply nested quoted triples
        graph.len() < 1000 && graph.max_nesting_depth() < 3
    }

    /// Estimate serialized size for a graph
    pub fn estimate_size(&self, graph: &StarGraph, format: StarFormat) -> usize {
        let base_size_per_triple = match format {
            StarFormat::TurtleStar => 50,   // Turtle is more compact
            StarFormat::NTriplesStar => 80, // N-Triples uses full IRIs
            StarFormat::TrigStar => 60,     // TriG has graph context
            StarFormat::NQuadsStar => 90,   // N-Quads uses full IRIs + graph
            StarFormat::JsonLdStar => 120,  // JSON-LD has overhead from JSON structure
        };

        let quoted_triple_multiplier = 1.5; // Quoted triples add overhead

        let mut total_size = graph.len() * base_size_per_triple;

        // Add overhead for quoted triples
        let quoted_count = graph.count_quoted_triples();
        total_size +=
            (quoted_count as f64 * quoted_triple_multiplier * base_size_per_triple as f64) as usize;

        total_size
    }

    /// Validate that a graph can be serialized in the given format
    pub fn validate_for_format(&self, graph: &StarGraph, format: StarFormat) -> StarResult<()> {
        // Check nesting depth
        let max_depth = graph.max_nesting_depth();
        if max_depth > self.config.max_nesting_depth {
            return Err(StarError::serialization_error(format!(
                "Graph nesting depth {} exceeds maximum {}",
                max_depth, self.config.max_nesting_depth
            )));
        }

        // Format-specific validation
        match format {
            StarFormat::TurtleStar | StarFormat::NTriplesStar => {
                // These formats support quoted triples in any position
                Ok(())
            }
            StarFormat::TrigStar | StarFormat::NQuadsStar => {
                // Validate quad-specific constraints
                self.validate_quad_constraints(graph, format)
            }
            StarFormat::JsonLdStar => {
                // JSON-LD-star supports quoted triples as annotations
                Ok(())
            }
        }
    }

    /// Validate quad-specific constraints for TriG-star and N-Quads-star formats
    fn validate_quad_constraints(&self, graph: &StarGraph, format: StarFormat) -> StarResult<()> {
        match format {
            StarFormat::NQuadsStar => {
                // Validate N-Quads-star specific constraints
                for quad in graph.quads() {
                    // Validate subject (must be IRI or blank node, can be quoted triple)
                    match &quad.subject {
                        StarTerm::NamedNode(_) | StarTerm::BlankNode(_) => {}
                        StarTerm::QuotedTriple(inner_triple) => {
                            // Validate the quoted triple structure
                            self.validate_quoted_triple_structure(inner_triple)?;
                        }
                        StarTerm::Literal(_) => {
                            return Err(StarError::serialization_error(
                                "N-Quads-star: Literals cannot be subjects in quads".to_string(),
                            ));
                        }
                        StarTerm::Variable(_) => {
                            return Err(StarError::serialization_error(
                                "N-Quads-star: Variables cannot be serialized in concrete data"
                                    .to_string(),
                            ));
                        }
                    }

                    // Validate predicate (must be IRI only)
                    match &quad.predicate {
                        StarTerm::NamedNode(_) => {}
                        _ => {
                            return Err(StarError::serialization_error(
                                "N-Quads-star: Predicates must be IRIs".to_string(),
                            ));
                        }
                    }

                    // Validate object (can be any term including quoted triples)
                    match &quad.object {
                        StarTerm::QuotedTriple(inner_triple) => {
                            self.validate_quoted_triple_structure(inner_triple)?;
                        }
                        StarTerm::Variable(_) => {
                            return Err(StarError::serialization_error(
                                "N-Quads-star: Variables cannot be serialized in concrete data"
                                    .to_string(),
                            ));
                        }
                        _ => {} // Other terms are valid as objects
                    }

                    // Validate graph component if present
                    if let Some(ref graph_term) = quad.graph {
                        match graph_term {
                            StarTerm::NamedNode(_) | StarTerm::BlankNode(_) => {}
                            StarTerm::QuotedTriple(_) => {
                                return Err(StarError::serialization_error(
                                    "N-Quads-star: Quoted triples cannot be used as graph names"
                                        .to_string(),
                                ));
                            }
                            StarTerm::Literal(_) => {
                                return Err(StarError::serialization_error(
                                    "N-Quads-star: Literals cannot be used as graph names"
                                        .to_string(),
                                ));
                            }
                            StarTerm::Variable(_) => {
                                return Err(StarError::serialization_error(
                                    "N-Quads-star: Variables cannot be serialized in concrete data"
                                        .to_string(),
                                ));
                            }
                        }
                    }
                }
            }
            StarFormat::TrigStar => {
                // Validate TriG-star specific constraints

                // Check that all graph names are valid
                for graph_name in graph.named_graph_names() {
                    if graph_name.is_empty() {
                        return Err(StarError::serialization_error(
                            "TriG-star: Empty graph names are not allowed".to_string(),
                        ));
                    }

                    // Validate that graph name is a valid IRI or blank node identifier
                    if !graph_name.starts_with("http://")
                        && !graph_name.starts_with("https://")
                        && !graph_name.starts_with("_:")
                        && !graph_name.starts_with("urn:")
                    {
                        return Err(StarError::serialization_error(format!(
                            "TriG-star: Invalid graph name format: {graph_name}"
                        )));
                    }
                }

                // Validate all triples in default and named graphs
                for triple in graph.triples() {
                    self.validate_triple_for_trig(triple)?;
                }

                // Validate triples in named graphs
                for graph_name in graph.named_graph_names() {
                    if let Some(named_triples) = graph.named_graph_triples(graph_name) {
                        for triple in named_triples {
                            self.validate_triple_for_trig(triple)?;
                        }
                    }
                }

                // Check for excessive graph nesting (TriG typically doesn't support nested graphs)
                let graph_count = graph.named_graph_names().len();
                if graph_count > 1000 {
                    return Err(StarError::serialization_error(
                        format!("TriG-star: Too many named graphs ({graph_count}), consider using streaming serialization")
                    ));
                }
            }
            _ => {
                return Err(StarError::serialization_error(
                    "Internal error: validate_quad_constraints called for non-quad format"
                        .to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Validate a quoted triple structure for proper nesting and term constraints
    #[allow(clippy::only_used_in_recursion)]
    fn validate_quoted_triple_structure(&self, triple: &StarTriple) -> StarResult<()> {
        // Validate subject of quoted triple
        match &triple.subject {
            StarTerm::Literal(_) => {
                return Err(StarError::serialization_error(
                    "Quoted triple: Literals cannot be subjects".to_string(),
                ));
            }
            StarTerm::Variable(_) => {
                return Err(StarError::serialization_error(
                    "Quoted triple: Variables cannot be serialized in concrete data".to_string(),
                ));
            }
            StarTerm::QuotedTriple(nested) => {
                // Recursively validate nested quoted triples
                self.validate_quoted_triple_structure(nested)?;
            }
            _ => {} // NamedNode and BlankNode are valid
        }

        // Validate predicate (must be IRI)
        match &triple.predicate {
            StarTerm::NamedNode(_) => {}
            _ => {
                return Err(StarError::serialization_error(
                    "Quoted triple: Predicates must be IRIs".to_string(),
                ));
            }
        }

        // Validate object
        match &triple.object {
            StarTerm::Variable(_) => {
                return Err(StarError::serialization_error(
                    "Quoted triple: Variables cannot be serialized in concrete data".to_string(),
                ));
            }
            StarTerm::QuotedTriple(nested) => {
                // Recursively validate nested quoted triples
                self.validate_quoted_triple_structure(nested)?;
            }
            _ => {} // All other terms are valid as objects
        }

        Ok(())
    }

    /// Validate a triple for TriG-star format constraints
    fn validate_triple_for_trig(&self, triple: &StarTriple) -> StarResult<()> {
        // Validate subject
        match &triple.subject {
            StarTerm::Literal(_) => {
                return Err(StarError::serialization_error(
                    "TriG-star: Literals cannot be subjects".to_string(),
                ));
            }
            StarTerm::Variable(_) => {
                return Err(StarError::serialization_error(
                    "TriG-star: Variables cannot be serialized in concrete data".to_string(),
                ));
            }
            StarTerm::QuotedTriple(inner) => {
                self.validate_quoted_triple_structure(inner)?;
            }
            _ => {} // NamedNode and BlankNode are valid
        }

        // Validate predicate (must be IRI)
        match &triple.predicate {
            StarTerm::NamedNode(_) => {}
            _ => {
                return Err(StarError::serialization_error(
                    "TriG-star: Predicates must be IRIs".to_string(),
                ));
            }
        }

        // Validate object
        match &triple.object {
            StarTerm::Variable(_) => {
                return Err(StarError::serialization_error(
                    "TriG-star: Variables cannot be serialized in concrete data".to_string(),
                ));
            }
            StarTerm::QuotedTriple(inner) => {
                self.validate_quoted_triple_structure(inner)?;
            }
            _ => {} // All other terms are valid as objects
        }

        Ok(())
    }

    /// Serialize a graph to string using the specified format and options
    pub fn serialize_graph(
        &self,
        graph: &StarGraph,
        format: StarFormat,
        _options: &SerializationOptions,
    ) -> StarResult<String> {
        // For now, this is a wrapper around serialize_to_string
        // In a more complete implementation, this would use the options parameter
        self.serialize_to_string(graph, format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;
    use crate::parser::StarParser;
    use crate::StarQuad;
    use crate::serializer::CompressionType;

    #[test]
    fn test_simple_triple_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );

        graph.insert(triple).unwrap();

        // Test N-Triples-star
        let result = serializer
            .serialize_to_string(&graph, StarFormat::NTriplesStar)
            .unwrap();
        assert!(result.contains("<http://example.org/alice>"));
        assert!(result.contains("<http://example.org/knows>"));
        assert!(result.contains("<http://example.org/bob>"));
        assert!(result.ends_with(" .\n"));

        // Test Turtle-star
        let result = serializer
            .serialize_to_string(&graph, StarFormat::TurtleStar)
            .unwrap();
        assert!(result.contains("@prefix"));
    }

    #[test]
    fn test_quoted_triple_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        graph.insert(outer).unwrap();

        let result = serializer
            .serialize_to_string(&graph, StarFormat::NTriplesStar)
            .unwrap();
        assert!(result.contains("<<"));
        assert!(result.contains(">>"));
        assert!(result.contains("\"25\""));
        assert!(result.contains("\"0.9\""));
    }

    #[test]
    fn test_literal_escaping() {
        let test_cases = vec![
            ("simple", "simple"),
            ("with\nnewline", "with\\nnewline"),
            ("with\ttab", "with\\ttab"),
            ("with\"quote", "with\\\"quote"),
            ("with\\backslash", "with\\\\backslash"),
        ];

        for (input, expected) in test_cases {
            let escaped = StarSerializer::escape_literal(input);
            assert_eq!(escaped, expected);
        }
    }

    #[test]
    fn test_literal_with_language_and_datatype() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Literal with language tag
        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/resource").unwrap(),
            StarTerm::iri("http://example.org/label").unwrap(),
            StarTerm::literal_with_language("Hello", "en").unwrap(),
        );

        // Literal with datatype
        let triple2 = StarTriple::new(
            StarTerm::iri("http://example.org/resource").unwrap(),
            StarTerm::iri("http://example.org/count").unwrap(),
            StarTerm::literal_with_datatype("42", "http://www.w3.org/2001/XMLSchema#integer")
                .unwrap(),
        );

        graph.insert(triple1).unwrap();
        graph.insert(triple2).unwrap();

        let result = serializer
            .serialize_to_string(&graph, StarFormat::NTriplesStar)
            .unwrap();
        assert!(result.contains("\"Hello\"@en"));
        assert!(result.contains("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"));
    }

    #[test]
    fn test_format_validation() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create deeply nested quoted triples
        let mut current_triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        // Nest it multiple times
        for _ in 0..15 {
            current_triple = StarTriple::new(
                StarTerm::quoted_triple(current_triple),
                StarTerm::iri("http://example.org/meta").unwrap(),
                StarTerm::literal("value").unwrap(),
            );
        }

        graph.insert(current_triple).unwrap();

        // Should fail validation due to excessive nesting
        let result = serializer.validate_for_format(&graph, StarFormat::NTriplesStar);
        assert!(result.is_err());
    }

    #[test]
    fn test_size_estimation() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        graph.insert(triple).unwrap();

        let turtle_size = serializer.estimate_size(&graph, StarFormat::TurtleStar);
        let ntriples_size = serializer.estimate_size(&graph, StarFormat::NTriplesStar);

        // N-Triples should be larger due to full IRIs
        assert!(ntriples_size > turtle_size);
    }

    #[test]
    fn test_enhanced_trig_star_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Add triple to default graph
        let default_triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );
        graph.insert(default_triple).unwrap();

        // Add quad to named graph
        let named_quad = StarQuad::new(
            StarTerm::iri("http://example.org/charlie").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
            Some(StarTerm::iri("http://example.org/graph1").unwrap()),
        );
        graph.insert_quad(named_quad).unwrap();

        let result = serializer
            .serialize_to_string(&graph, StarFormat::TrigStar)
            .unwrap();

        // Should contain prefix declarations
        assert!(result.contains("@prefix"));

        // Should contain default graph block
        assert!(result.contains("{"));
        assert!(result.contains("alice"));

        // Should contain named graph declaration
        assert!(result.contains("http://example.org/graph1"));
        assert!(result.contains("charlie"));
    }

    #[test]
    fn test_enhanced_nquads_star_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Add triple to default graph
        let default_triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );
        graph.insert(default_triple).unwrap();

        // Add quad to named graph
        let named_quad = StarQuad::new(
            StarTerm::iri("http://example.org/charlie").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
            Some(StarTerm::iri("http://example.org/graph1").unwrap()),
        );
        graph.insert_quad(named_quad).unwrap();

        let result = serializer
            .serialize_to_string(&graph, StarFormat::NQuadsStar)
            .unwrap();

        // Should contain default graph triple (3 terms)
        assert!(result.contains(
            "<http://example.org/alice> <http://example.org/knows> <http://example.org/bob> ."
        ));

        // Should contain named graph quad (4 terms)
        assert!(result.contains("<http://example.org/charlie> <http://example.org/age> \"30\" <http://example.org/graph1> ."));
    }

    #[test]
    fn test_quoted_triple_serialization_roundtrip() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create a complex quoted triple structure
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/says").unwrap(),
            StarTerm::literal("Hello").unwrap(),
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.95").unwrap(),
        );

        graph.insert(outer).unwrap();

        // Test all formats
        for format in [
            StarFormat::TurtleStar,
            StarFormat::NTriplesStar,
            StarFormat::TrigStar,
            StarFormat::NQuadsStar,
        ] {
            let serialized = serializer.serialize_to_string(&graph, format).unwrap();

            // Should contain quoted triple markers
            assert!(serialized.contains("<<"));
            assert!(serialized.contains(">>"));

            // Should contain the nested content
            assert!(serialized.contains("alice"));
            assert!(serialized.contains("says"));
            assert!(serialized.contains("Hello"));
            assert!(serialized.contains("certainty"));
        }
    }

    #[test]
    fn test_nquads_star_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Add triples to default graph
        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );
        graph.insert(triple1).unwrap();

        // Add quad with named graph
        let quad1 = StarQuad::new(
            StarTerm::iri("http://example.org/charlie").unwrap(),
            StarTerm::iri("http://example.org/likes").unwrap(),
            StarTerm::iri("http://example.org/dave").unwrap(),
            Some(StarTerm::iri("http://example.org/graph1").unwrap()),
        );
        graph.insert_quad(quad1).unwrap();

        // Add quad with quoted triple
        let quoted = StarTriple::new(
            StarTerm::iri("http://example.org/eve").unwrap(),
            StarTerm::iri("http://example.org/says").unwrap(),
            StarTerm::literal("hello").unwrap(),
        );
        let quad2 = StarQuad::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.8").unwrap(),
            Some(StarTerm::iri("http://example.org/graph2").unwrap()),
        );
        graph.insert_quad(quad2).unwrap();

        let serialized = serializer
            .serialize_to_string(&graph, StarFormat::NQuadsStar)
            .unwrap();

        // Verify each quad is on its own line
        let lines: Vec<&str> = serialized
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 3);

        // Verify graph contexts are present
        assert!(serialized.contains("<http://example.org/graph1>"));
        assert!(serialized.contains("<http://example.org/graph2>"));

        // Verify quoted triple syntax
        assert!(serialized.contains("<< "));
        assert!(serialized.contains(" >>"));
    }

    #[test]
    fn test_trig_star_serialization_with_multiple_graphs() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Add to default graph
        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
        );
        graph.insert(triple1).unwrap();

        // Add to named graph 1
        let quad1 = StarQuad::new(
            StarTerm::iri("http://example.org/bob").unwrap(),
            StarTerm::iri("http://example.org/likes").unwrap(),
            StarTerm::iri("http://example.org/coffee").unwrap(),
            Some(StarTerm::iri("http://example.org/preferences").unwrap()),
        );
        graph.insert_quad(quad1).unwrap();

        // Add quoted triple to named graph 2
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/charlie").unwrap(),
            StarTerm::iri("http://example.org/believes").unwrap(),
            StarTerm::literal("earth is round").unwrap(),
        );
        let quad2 = StarQuad::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/confidence").unwrap(),
            StarTerm::literal("1.0").unwrap(),
            Some(StarTerm::iri("http://example.org/beliefs").unwrap()),
        );
        graph.insert_quad(quad2).unwrap();

        let serialized = serializer
            .serialize_to_string(&graph, StarFormat::TrigStar)
            .unwrap();

        // Verify prefixes are included
        assert!(serialized.contains("@prefix"));

        // Verify default graph block
        assert!(serialized.contains("{\n"));
        assert!(serialized.contains("alice"));

        // Verify named graph blocks
        assert!(serialized.contains("<http://example.org/preferences> {"));
        assert!(serialized.contains("<http://example.org/beliefs> {"));

        // Verify quoted triple in TriG format
        assert!(serialized.contains("<<"));
        assert!(serialized.contains(">>"));
    }

    #[test]
    fn test_streaming_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create a larger graph for streaming test
        for i in 0..1000 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{i}")).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal(&format!("value{i}")).unwrap(),
            );
            graph.insert(triple).unwrap();
        }

        let mut output = Vec::new();
        serializer
            .serialize_streaming(&graph, &mut output, StarFormat::NTriplesStar, 100)
            .unwrap();

        let output_str = String::from_utf8(output).unwrap();
        let lines: Vec<&str> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 1000);

        // Verify each line is a valid N-Triples statement
        for line in lines {
            assert!(line.ends_with(" ."));
            assert!(line.contains("http://example.org/"));
        }
    }

    #[test]
    fn test_parallel_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create a graph suitable for parallel processing
        for i in 0..500 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/subject{i}")).unwrap(),
                StarTerm::iri("http://example.org/predicate").unwrap(),
                StarTerm::iri(&format!("http://example.org/object{i}")).unwrap(),
            );
            graph.insert(triple).unwrap();
        }

        let output = Box::leak(Box::new(Vec::new()));
        let output_ptr = output as *const Vec<u8>;
        serializer
            .serialize_parallel(&graph, output, StarFormat::NTriplesStar, 4, 100)
            .unwrap();

        // Safe because we know the serialize method completed and output is still valid
        let output_data = unsafe { &*output_ptr };
        let output_str = String::from_utf8(output_data.clone()).unwrap();
        let lines: Vec<&str> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 500);
    }

    #[test]
    fn test_serialization_with_options() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create test graph
        for i in 0..100 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{i}")).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal(&format!("test{i}")).unwrap(),
            );
            graph.insert(triple).unwrap();
        }

        let options = SerializationOptions {
            streaming: true,
            batch_size: 25,
            buffer_size: 1024,
            ..Default::default()
        };

        let output = Box::leak(Box::new(Vec::new()));
        let output_ptr = output as *const Vec<u8>;
        serializer
            .serialize_with_options(&graph, output, StarFormat::NTriplesStar, &options)
            .unwrap();

        // Safe because we know the serialize method completed and output is still valid
        let output_data = unsafe { &*output_ptr };
        let output_str = String::from_utf8(output_data.clone()).unwrap();
        let lines: Vec<&str> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 100);
    }

    #[test]
    fn test_optimized_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create a complex graph with quoted triples
        for i in 0..50 {
            let inner = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/alice{i}")).unwrap(),
                StarTerm::iri("http://example.org/says").unwrap(),
                StarTerm::literal(&format!("statement{i}")).unwrap(),
            );
            let outer = StarTriple::new(
                StarTerm::quoted_triple(inner),
                StarTerm::iri("http://example.org/certainty").unwrap(),
                StarTerm::literal("0.9").unwrap(),
            );
            graph.insert(outer).unwrap();
        }

        let output = Box::leak(Box::new(Vec::new()));
        let output_ptr = output as *const Vec<u8>;
        serializer
            .serialize_optimized(&graph, output, StarFormat::NTriplesStar)
            .unwrap();

        // Safe because we know the serialize method completed and output is still valid
        let output_data = unsafe { &*output_ptr };
        let output_str = String::from_utf8(output_data.clone()).unwrap();

        // Should contain quoted triple syntax
        assert!(output_str.contains("<<"));
        assert!(output_str.contains(">>"));

        let lines: Vec<&str> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 50);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        for i in 0..100 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{i}")).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal(&format!("value{i}")).unwrap(),
            );
            graph.insert(triple).unwrap();
        }

        let options = SerializationOptions::default();
        let memory_estimate =
            serializer.estimate_memory_usage(&graph, StarFormat::NTriplesStar, &options);

        // Should provide reasonable estimate (not zero, not excessive)
        assert!(memory_estimate > 1000);
        assert!(memory_estimate < 10_000_000);

        let streaming_options = SerializationOptions {
            streaming: true,
            ..Default::default()
        };
        let streaming_estimate =
            serializer.estimate_memory_usage(&graph, StarFormat::NTriplesStar, &streaming_options);

        // Streaming should use reasonable memory (may not be less for small datasets due to overhead)
        assert!(streaming_estimate > 0);
        assert!(streaming_estimate < 10_000_000);
    }

    // TODO: Re-enable when ChunkedIterator is implemented
    // #[test]
    // fn test_chunked_iterator() {
    //     let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    //     let chunked = ChunkedIterator::new(data.into_iter(), 3);
    //
    //     let chunks: Vec<_> = chunked.collect();
    //     assert_eq!(chunks.len(), 4);
    //     assert_eq!(chunks[0], vec![1, 2, 3]);
    //     assert_eq!(chunks[1], vec![4, 5, 6]);
    //     assert_eq!(chunks[2], vec![7, 8, 9]);
    //     assert_eq!(chunks[3], vec![10]);
    // }

    #[test]
    fn test_compression_type_selection() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::literal("test").unwrap(),
        );
        graph.insert(triple).unwrap();

        // Test different compression types (placeholder implementations)
        for compression in [
            CompressionType::None,
            CompressionType::Gzip,
            CompressionType::Zstd,
            CompressionType::Lz4,
        ] {
            let options = SerializationOptions {
                compression,
                ..Default::default()
            };

            let output = Box::leak(Box::new(Vec::new()));
            let result = serializer.serialize_with_options(
                &graph,
                output,
                StarFormat::NTriplesStar,
                &options,
            );

            // Should not fail even with unimplemented compression
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let parser = StarParser::new();
        let serializer = StarSerializer::new();
        let mut original_graph = StarGraph::new();

        // Create complex graph with various features
        let simple = StarTriple::new(
            StarTerm::iri("http://example.org/s1").unwrap(),
            StarTerm::iri("http://example.org/p1").unwrap(),
            StarTerm::literal("test").unwrap(),
        );
        original_graph.insert(simple).unwrap();

        // Quoted triple
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/says").unwrap(),
            StarTerm::literal("hello").unwrap(),
        );
        let quoted = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );
        original_graph.insert(quoted).unwrap();

        // Test roundtrip for each format
        for format in [
            StarFormat::TurtleStar,
            StarFormat::NTriplesStar,
            StarFormat::TrigStar,
            StarFormat::NQuadsStar,
        ] {
            let serialized = serializer
                .serialize_to_string(&original_graph, format)
                .unwrap();
            let parsed_graph = parser.parse_str(&serialized, format).unwrap();

            // Verify same number of triples
            assert_eq!(
                original_graph.total_len(),
                parsed_graph.total_len(),
                "Roundtrip failed for format {format:?}"
            );
        }
    }

    #[test]
    fn test_quad_validation() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create valid quad
        let valid_quad = StarQuad::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
            Some(StarTerm::iri("http://example.org/graph1").unwrap()),
        );
        graph.insert_quad(valid_quad).unwrap();

        // Valid quad should pass validation
        assert!(serializer
            .validate_for_format(&graph, StarFormat::NQuadsStar)
            .is_ok());
        assert!(serializer
            .validate_for_format(&graph, StarFormat::TrigStar)
            .is_ok());

        // Test validation methods directly without going through graph insertion
        // (since graph insertion itself validates and rejects invalid quads)

        // Test quoted triple validation with invalid subject (literal)
        let invalid_inner = StarTriple::new(
            StarTerm::literal("invalid_subject").unwrap(), // Invalid: literal as subject
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::literal("object").unwrap(),
        );

        // Test the quoted triple validation directly
        let result = serializer.validate_quoted_triple_structure(&invalid_inner);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Literals cannot be subjects"));

        // Test quoted triple validation with invalid predicate
        let invalid_predicate_triple = StarTriple::new(
            StarTerm::iri("http://example.org/subject").unwrap(),
            StarTerm::literal("invalid_predicate").unwrap(), // Invalid: literal as predicate
            StarTerm::literal("object").unwrap(),
        );

        let result = serializer.validate_quoted_triple_structure(&invalid_predicate_triple);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Predicates must be IRIs"));

        // Test TriG validation directly
        let invalid_trig_triple = StarTriple::new(
            StarTerm::variable("invalid_subject").unwrap(), // Invalid: variable in concrete data
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::literal("object").unwrap(),
        );

        let result = serializer.validate_triple_for_trig(&invalid_trig_triple);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Variables cannot be serialized"));
    }

    #[test]
    fn test_quoted_triple_validation() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create valid quoted triple
        let inner_triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/says").unwrap(),
            StarTerm::literal("hello").unwrap(),
        );
        let quoted_triple = StarTriple::new(
            StarTerm::quoted_triple(inner_triple),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );
        graph.insert(quoted_triple).unwrap();

        // Should pass validation for all formats
        assert!(serializer
            .validate_for_format(&graph, StarFormat::TurtleStar)
            .is_ok());
        assert!(serializer
            .validate_for_format(&graph, StarFormat::NTriplesStar)
            .is_ok());
        assert!(serializer
            .validate_for_format(&graph, StarFormat::TrigStar)
            .is_ok());
        assert!(serializer
            .validate_for_format(&graph, StarFormat::NQuadsStar)
            .is_ok());

        // Test invalid quoted triple with literal as subject
        let mut invalid_graph = StarGraph::new();
        let invalid_inner = StarTriple::new(
            StarTerm::literal("invalid_subject").unwrap(), // Invalid: literal as subject
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::literal("object").unwrap(),
        );
        let invalid_quoted = StarTriple::new(
            StarTerm::quoted_triple(invalid_inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );
        invalid_graph.insert(invalid_quoted).unwrap();

        // Should fail validation
        let result = serializer.validate_for_format(&invalid_graph, StarFormat::NQuadsStar);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Literals cannot be subjects"));
    }

    #[test]
    fn test_trig_star_graph_name_validation() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Add quad with valid graph name
        let valid_quad = StarQuad::new(
            StarTerm::iri("http://example.org/subject").unwrap(),
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::iri("http://example.org/object").unwrap(),
            Some(StarTerm::iri("http://example.org/valid_graph").unwrap()),
        );
        graph.insert_quad(valid_quad).unwrap();

        // Should pass validation
        assert!(serializer
            .validate_for_format(&graph, StarFormat::TrigStar)
            .is_ok());

        // Test with blank node graph name
        let mut graph_with_bnode = StarGraph::new();
        let bnode_quad = StarQuad::new(
            StarTerm::iri("http://example.org/subject").unwrap(),
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::iri("http://example.org/object").unwrap(),
            Some(StarTerm::blank_node("graph1").unwrap()),
        );
        graph_with_bnode.insert_quad(bnode_quad).unwrap();

        // Should also pass validation (blank nodes are valid graph names)
        assert!(serializer
            .validate_for_format(&graph_with_bnode, StarFormat::TrigStar)
            .is_ok());
    }
}
