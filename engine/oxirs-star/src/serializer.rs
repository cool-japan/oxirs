//! RDF-star serialization implementations for various formats.
//!
//! This module provides serializers for RDF-star formats including:
//! - Turtle-star (*.ttls)
//! - N-Triples-star (*.nts)  
//! - TriG-star (*.trigs)
//! - N-Quads-star (*.nqs)

use std::collections::HashMap;
use std::io::{BufWriter, Write};

use tracing::{debug, span, Level};

use crate::model::{StarGraph, StarQuad, StarTerm, StarTriple};
use crate::parser::StarFormat;
use crate::{StarConfig, StarError, StarResult};

/// Context for serialization with namespace prefixes and formatting options
#[derive(Debug, Default)]
struct SerializationContext {
    /// Namespace prefixes for compact representation
    prefixes: HashMap<String, String>,
    /// Base IRI for relative references
    base_iri: Option<String>,
    /// Pretty printing with indentation
    pretty_print: bool,
    /// Current indentation level
    indent_level: usize,
    /// Indentation string (spaces or tabs)
    indent_string: String,
}

impl SerializationContext {
    fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base_iri: None,
            pretty_print: true,
            indent_level: 0,
            indent_string: "  ".to_string(),
        }
    }

    /// Add a namespace prefix
    fn add_prefix(&mut self, prefix: &str, namespace: &str) {
        self.prefixes
            .insert(prefix.to_string(), namespace.to_string());
    }

    /// Get current indentation
    fn current_indent(&self) -> String {
        self.indent_string.repeat(self.indent_level)
    }

    /// Increase indentation level
    fn increase_indent(&mut self) {
        self.indent_level += 1;
    }

    /// Decrease indentation level
    fn decrease_indent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    /// Try to compress IRI using prefixes
    fn compress_iri(&self, iri: &str) -> String {
        for (prefix, namespace) in &self.prefixes {
            if iri.starts_with(namespace) {
                let local = &iri[namespace.len()..];
                return format!("{}:{}", prefix, local);
            }
        }

        // Return full IRI if no prefix match
        format!("<{}>", iri)
    }
}

/// RDF-star serializer with support for multiple formats
pub struct StarSerializer {
    config: StarConfig,
}

impl StarSerializer {
    /// Create a new serializer with default configuration
    pub fn new() -> Self {
        Self {
            config: StarConfig::default(),
        }
    }

    /// Create a new serializer with custom configuration
    pub fn with_config(config: StarConfig) -> Self {
        Self { config }
    }

    /// Serialize a StarGraph to a writer in the specified format
    pub fn serialize<W: Write>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
    ) -> StarResult<()> {
        let span = span!(Level::INFO, "serialize_rdf_star", format = ?format);
        let _enter = span.enter();

        match format {
            StarFormat::TurtleStar => self.serialize_turtle_star(graph, writer),
            StarFormat::NTriplesStar => self.serialize_ntriples_star(graph, writer),
            StarFormat::TrigStar => self.serialize_trig_star(graph, writer),
            StarFormat::NQuadsStar => self.serialize_nquads_star(graph, writer),
        }
    }

    /// Serialize to string in the specified format
    pub fn serialize_to_string(&self, graph: &StarGraph, format: StarFormat) -> StarResult<String> {
        let mut buffer = Vec::new();
        self.serialize(graph, &mut buffer, format)?;
        String::from_utf8(buffer).map_err(|e| StarError::SerializationError(e.to_string()))
    }

    /// Serialize to Turtle-star format
    pub fn serialize_turtle_star<W: Write>(&self, graph: &StarGraph, writer: W) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_turtle_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let mut context = SerializationContext::new();

        // Add common prefixes
        self.add_common_prefixes(&mut context);

        // Write prefixes
        self.write_turtle_prefixes(&mut buf_writer, &context)?;

        // Write triples
        for triple in graph.triples() {
            self.write_turtle_triple(&mut buf_writer, triple, &context)?;
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::SerializationError(e.to_string()))?;
        debug!("Serialized {} triples in Turtle-star format", graph.len());
        Ok(())
    }

    /// Write Turtle-star prefixes
    fn write_turtle_prefixes<W: Write>(
        &self,
        writer: &mut W,
        context: &SerializationContext,
    ) -> StarResult<()> {
        for (prefix, namespace) in &context.prefixes {
            writeln!(writer, "@prefix {}: <{}> .", prefix, namespace)
                .map_err(|e| StarError::SerializationError(e.to_string()))?;
        }

        if !context.prefixes.is_empty() {
            writeln!(writer).map_err(|e| StarError::SerializationError(e.to_string()))?;
        }

        Ok(())
    }

    /// Write a single Turtle-star triple
    fn write_turtle_triple<W: Write>(
        &self,
        writer: &mut W,
        triple: &StarTriple,
        context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = self.format_term(&triple.subject, context)?;
        let predicate_str = self.format_term(&triple.predicate, context)?;
        let object_str = self.format_term(&triple.object, context)?;

        if context.pretty_print {
            writeln!(
                writer,
                "{}{} {} {} .",
                context.current_indent(),
                subject_str,
                predicate_str,
                object_str
            )
            .map_err(|e| StarError::SerializationError(e.to_string()))?;
        } else {
            writeln!(writer, "{} {} {} .", subject_str, predicate_str, object_str)
                .map_err(|e| StarError::SerializationError(e.to_string()))?;
        }

        Ok(())
    }

    /// Serialize to N-Triples-star format
    pub fn serialize_ntriples_star<W: Write>(
        &self,
        graph: &StarGraph,
        writer: W,
    ) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_ntriples_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let context = SerializationContext::new(); // N-Triples doesn't use prefixes

        for triple in graph.triples() {
            self.write_ntriples_triple(&mut buf_writer, triple, &context)?;
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::SerializationError(e.to_string()))?;
        debug!(
            "Serialized {} triples in N-Triples-star format",
            graph.len()
        );
        Ok(())
    }

    /// Write a single N-Triples-star triple
    fn write_ntriples_triple<W: Write>(
        &self,
        writer: &mut W,
        triple: &StarTriple,
        context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = self.format_term_ntriples(&triple.subject)?;
        let predicate_str = self.format_term_ntriples(&triple.predicate)?;
        let object_str = self.format_term_ntriples(&triple.object)?;

        writeln!(writer, "{} {} {} .", subject_str, predicate_str, object_str)
            .map_err(|e| StarError::SerializationError(e.to_string()))?;

        Ok(())
    }

    /// Serialize to TriG-star format (with named graphs)
    pub fn serialize_trig_star<W: Write>(&self, graph: &StarGraph, writer: W) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_trig_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let mut context = SerializationContext::new();
        context.pretty_print = true;

        // Add common prefixes
        self.add_common_prefixes(&mut context);

        // Write prefixes first
        self.write_turtle_prefixes(&mut buf_writer, &context)?;

        // Serialize default graph if it has triples
        if !graph.triples().is_empty() {
            writeln!(buf_writer, "{{").map_err(|e| StarError::SerializationError(e.to_string()))?;

            context.increase_indent();
            for triple in graph.triples() {
                self.write_turtle_triple(&mut buf_writer, triple, &context)?;
            }
            context.decrease_indent();

            writeln!(buf_writer, "}}").map_err(|e| StarError::SerializationError(e.to_string()))?;
            writeln!(buf_writer).map_err(|e| StarError::SerializationError(e.to_string()))?;
        }

        // Serialize named graphs
        for graph_name in graph.named_graph_names() {
            if let Some(named_triples) = graph.named_graph_triples(graph_name) {
                if !named_triples.is_empty() {
                    // Write graph declaration
                    let graph_term = self.parse_graph_name(graph_name, &context)?;
                    writeln!(buf_writer, "{} {{", graph_term)
                        .map_err(|e| StarError::SerializationError(e.to_string()))?;

                    context.increase_indent();
                    for triple in named_triples {
                        self.write_turtle_triple(&mut buf_writer, triple, &context)?;
                    }
                    context.decrease_indent();

                    writeln!(buf_writer, "}}")
                        .map_err(|e| StarError::SerializationError(e.to_string()))?;
                    writeln!(buf_writer)
                        .map_err(|e| StarError::SerializationError(e.to_string()))?;
                }
            }
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::SerializationError(e.to_string()))?;
        debug!(
            "Serialized {} quads ({} total triples) in TriG-star format",
            graph.quad_len(),
            graph.total_len()
        );
        Ok(())
    }

    /// Serialize to N-Quads-star format
    pub fn serialize_nquads_star<W: Write>(&self, graph: &StarGraph, writer: W) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_nquads_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let context = SerializationContext::new(); // N-Quads doesn't use prefixes

        // Serialize all quads from the graph (including both default and named graphs)
        for quad in graph.quads() {
            self.write_nquads_quad_complete(&mut buf_writer, quad, &context)?;
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::SerializationError(e.to_string()))?;
        debug!(
            "Serialized {} quads ({} total triples) in N-Quads-star format",
            graph.quad_len(),
            graph.total_len()
        );
        Ok(())
    }

    /// Write a single N-Quads-star quad with proper graph context
    fn write_nquads_quad_complete<W: Write>(
        &self,
        writer: &mut W,
        quad: &StarQuad,
        _context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = self.format_term_ntriples(&quad.subject)?;
        let predicate_str = self.format_term_ntriples(&quad.predicate)?;
        let object_str = self.format_term_ntriples(&quad.object)?;

        if let Some(ref graph_term) = quad.graph {
            // Named graph quad
            let graph_str = self.format_term_ntriples(graph_term)?;
            writeln!(
                writer,
                "{} {} {} {} .",
                subject_str, predicate_str, object_str, graph_str
            )
            .map_err(|e| StarError::SerializationError(e.to_string()))?;
        } else {
            // Default graph quad (triple)
            writeln!(writer, "{} {} {} .", subject_str, predicate_str, object_str)
                .map_err(|e| StarError::SerializationError(e.to_string()))?;
        }

        Ok(())
    }

    /// Write a single N-Quads-star quad (triple + optional graph) - legacy method
    fn write_nquads_quad<W: Write>(
        &self,
        writer: &mut W,
        triple: &StarTriple,
        _context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = self.format_term_ntriples(&triple.subject)?;
        let predicate_str = self.format_term_ntriples(&triple.predicate)?;
        let object_str = self.format_term_ntriples(&triple.object)?;

        // Default graph (no graph component)
        writeln!(writer, "{} {} {} .", subject_str, predicate_str, object_str)
            .map_err(|e| StarError::SerializationError(e.to_string()))?;

        Ok(())
    }

    /// Format a StarTerm for Turtle-star (with prefix compression)
    fn format_term(&self, term: &StarTerm, context: &SerializationContext) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(context.compress_iri(&node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", Self::escape_literal(&literal.value));

                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{}", lang));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^{}", context.compress_iri(&datatype.iri)));
                }

                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = self.format_term(&triple.subject, context)?;
                let predicate = self.format_term(&triple.predicate, context)?;
                let object = self.format_term(&triple.object, context)?;
                Ok(format!("<< {} {} {} >>", subject, predicate, object))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }

    /// Format a StarTerm for N-Triples-star (full IRIs, no prefixes)
    fn format_term_ntriples(&self, term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(format!("<{}>", node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", Self::escape_literal(&literal.value));

                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{}", lang));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^<{}>", datatype.iri));
                }

                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = self.format_term_ntriples(&triple.subject)?;
                let predicate = self.format_term_ntriples(&triple.predicate)?;
                let object = self.format_term_ntriples(&triple.object)?;
                Ok(format!("<< {} {} {} >>", subject, predicate, object))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }

    /// Escape special characters in literals
    fn escape_literal(value: &str) -> String {
        value
            .replace('\\', "\\\\")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
            .replace('"', "\\\"")
    }

    /// Add common namespace prefixes
    fn add_common_prefixes(&self, context: &mut SerializationContext) {
        context.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        context.add_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#");
        context.add_prefix("owl", "http://www.w3.org/2002/07/owl#");
        context.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");
        context.add_prefix("foaf", "http://xmlns.com/foaf/0.1/");
        context.add_prefix("dc", "http://purl.org/dc/terms/");
    }

    /// Parse a graph name string back to a term for TriG serialization
    fn parse_graph_name(
        &self,
        graph_name: &str,
        context: &SerializationContext,
    ) -> StarResult<String> {
        if graph_name.starts_with("_:") {
            // Blank node graph name
            Ok(graph_name.to_string())
        } else {
            // Named node graph name - compress with prefixes if possible
            Ok(context.compress_iri(graph_name))
        }
    }
}

impl Default for StarSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for serialization
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
            return Err(StarError::SerializationError(format!(
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
                // TODO: Add quad-specific validation when implemented
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;
    use crate::parser::StarParser;

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
                "Roundtrip failed for format {:?}",
                format
            );
        }
    }
}
