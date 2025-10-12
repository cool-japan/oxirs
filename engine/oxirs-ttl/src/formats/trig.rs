//! TriG format parser and serializer
//!
//! TriG is an extension of Turtle that supports named graphs.
//! Syntax examples:
//! - Default graph: `ex:alice ex:knows ex:bob .`
//! - Named graph: `ex:graph1 { ex:alice ex:age 30 . }`
//! - GRAPH syntax: `GRAPH ex:graph2 { ex:alice ex:worksFor ex:company . }`

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::toolkit::{Parser, Serializer};
use oxirs_core::model::{GraphName, Literal, NamedNode, Quad, Triple};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};

/// TriG parser with support for named graphs
#[derive(Debug, Clone)]
pub struct TriGParser {
    /// Whether to continue parsing after errors
    pub lenient: bool,
    /// Base IRI for resolving relative IRIs
    pub base_iri: Option<String>,
    /// Initial prefix declarations
    pub prefixes: HashMap<String, String>,
}

impl Default for TriGParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TriGParser {
    /// Create a new TriG parser
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();

        // Add common prefixes
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );

        Self {
            lenient: false,
            base_iri: None,
            prefixes,
        }
    }

    /// Parse TriG content into quads
    fn parse_trig_content<R: BufRead>(&self, reader: R) -> TurtleResult<Vec<Quad>> {
        let mut quads = Vec::new();
        let mut current_graph = GraphName::DefaultGraph;
        let content = {
            let mut buffer = String::new();
            let mut reader = reader;
            reader.read_to_string(&mut buffer).map_err(|e| {
                TurtleParseError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?;
            buffer
        };

        // Simple line-by-line parsing for basic TriG support
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                i += 1;
                continue;
            }

            // Handle prefix declarations
            if line.starts_with("@prefix") {
                // Parse prefix declaration - delegate to turtle parser logic
                i += 1;
                continue;
            }

            // Handle base declarations
            if line.starts_with("@base") {
                // Parse base declaration
                i += 1;
                continue;
            }

            // Check for named graph start
            if line.contains('{') && !line.ends_with('.') {
                // Parse graph name
                let graph_part = line.split('{').next().unwrap_or("").trim();

                if graph_part.starts_with("GRAPH") {
                    // GRAPH <iri> { syntax
                    let graph_iri = graph_part.strip_prefix("GRAPH").unwrap_or("").trim();
                    current_graph = self.parse_graph_name(graph_iri)?;
                } else if !graph_part.is_empty() {
                    // <iri> { syntax
                    current_graph = self.parse_graph_name(graph_part)?;
                } else {
                    current_graph = GraphName::DefaultGraph;
                }

                i += 1;
                continue;
            }

            // Check for graph end
            if line.trim() == "}" {
                current_graph = GraphName::DefaultGraph;
                i += 1;
                continue;
            }

            // Parse triple and convert to quad
            if line.ends_with('.') {
                if let Ok(triple) = self.parse_simple_triple(line) {
                    let quad = Quad::new(
                        triple.subject().clone(),
                        triple.predicate().clone(),
                        triple.object().clone(),
                        current_graph.clone(),
                    );
                    quads.push(quad);
                }
            }

            i += 1;
        }

        Ok(quads)
    }

    /// Parse a simple triple from a line (basic implementation)
    fn parse_simple_triple(&self, line: &str) -> TurtleResult<Triple> {
        // This is a simplified parser - in production you'd want full Turtle parsing
        let line = line.trim_end_matches('.').trim();
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 3 {
            return Err(TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                message: "Invalid triple syntax".to_string(),
                position: TextPosition::start(),
            }));
        }

        // Very basic parsing - this should use proper Turtle tokenization
        let subject = self.parse_term_as_subject(parts[0])?;
        let predicate = self.parse_term_as_predicate(parts[1])?;
        let object = self.parse_term_as_object(&parts[2..].join(" "))?;

        Ok(Triple::new(subject, predicate, object))
    }

    /// Parse a graph name from string
    fn parse_graph_name(&self, graph_str: &str) -> TurtleResult<GraphName> {
        let graph_str = graph_str.trim();

        if graph_str.starts_with('<') && graph_str.ends_with('>') {
            let iri = graph_str.trim_start_matches('<').trim_end_matches('>');
            let named_node = NamedNode::new(iri).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid IRI: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(GraphName::NamedNode(named_node))
        } else if graph_str.contains(':') {
            // Handle prefixed names
            let expanded = self.expand_prefixed_name(graph_str)?;
            let named_node = NamedNode::new(&expanded).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid IRI: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(GraphName::NamedNode(named_node))
        } else {
            Err(TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid graph name: {graph_str}"),
                position: TextPosition::start(),
            }))
        }
    }

    /// Expand a prefixed name (simplified implementation)
    fn expand_prefixed_name(&self, prefixed: &str) -> TurtleResult<String> {
        if let Some(colon_pos) = prefixed.find(':') {
            let prefix = &prefixed[..colon_pos];
            let local = &prefixed[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                Ok(format!("{namespace}{local}"))
            } else {
                Err(TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Unknown prefix: {prefix}"),
                    position: TextPosition::start(),
                }))
            }
        } else {
            Err(TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid prefixed name: {prefixed}"),
                position: TextPosition::start(),
            }))
        }
    }

    /// Parse term as subject (simplified)
    fn parse_term_as_subject(&self, term: &str) -> TurtleResult<oxirs_core::model::Subject> {
        use oxirs_core::model::{BlankNode, Subject};

        if term.starts_with('<') && term.ends_with('>') {
            let iri = term.trim_start_matches('<').trim_end_matches('>');
            let named_node = NamedNode::new(iri).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid IRI: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(Subject::NamedNode(named_node))
        } else if let Some(stripped) = term.strip_prefix("_:") {
            let blank_node = BlankNode::new(stripped).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid blank node: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(Subject::BlankNode(blank_node))
        } else if term.contains(':') {
            let expanded = self.expand_prefixed_name(term)?;
            let named_node = NamedNode::new(&expanded).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid IRI: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(Subject::NamedNode(named_node))
        } else {
            Err(TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid subject: {term}"),
                position: TextPosition::start(),
            }))
        }
    }

    /// Parse term as predicate (simplified)
    fn parse_term_as_predicate(&self, term: &str) -> TurtleResult<oxirs_core::model::Predicate> {
        use oxirs_core::model::Predicate;

        if term == "a" {
            // Handle rdf:type abbreviation
            let rdf_type =
                NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
            Ok(Predicate::NamedNode(rdf_type))
        } else if term.starts_with('<') && term.ends_with('>') {
            let iri = term.trim_start_matches('<').trim_end_matches('>');
            let named_node = NamedNode::new(iri).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid IRI: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(Predicate::NamedNode(named_node))
        } else if term.contains(':') {
            let expanded = self.expand_prefixed_name(term)?;
            let named_node = NamedNode::new(&expanded).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid IRI: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(Predicate::NamedNode(named_node))
        } else {
            Err(TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid predicate: {term}"),
                position: TextPosition::start(),
            }))
        }
    }

    /// Parse term as object (simplified)
    fn parse_term_as_object(&self, term: &str) -> TurtleResult<oxirs_core::model::Object> {
        use oxirs_core::model::{BlankNode, Literal, Object};

        let term = term.trim();

        if term.starts_with('"') && term.ends_with('"') {
            // String literal
            let content = &term[1..term.len() - 1];
            let literal = Literal::new_simple_literal(content);
            Ok(Object::Literal(literal))
        } else if term.starts_with('<') && term.ends_with('>') {
            let iri = term.trim_start_matches('<').trim_end_matches('>');
            let named_node = NamedNode::new(iri).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid IRI: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(Object::NamedNode(named_node))
        } else if let Some(stripped) = term.strip_prefix("_:") {
            let blank_node = BlankNode::new(stripped).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid blank node: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(Object::BlankNode(blank_node))
        } else if term.contains(':') {
            let expanded = self.expand_prefixed_name(term)?;
            let named_node = NamedNode::new(&expanded).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid IRI: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(Object::NamedNode(named_node))
        } else {
            // Try to parse as a typed literal
            let literal = Literal::new_simple_literal(term);
            Ok(Object::Literal(literal))
        }
    }
}

impl Parser<Quad> for TriGParser {
    fn parse<R: Read>(&self, reader: R) -> TurtleResult<Vec<Quad>> {
        let buf_reader = BufReader::new(reader);
        self.parse_trig_content(buf_reader)
    }

    fn for_reader<R: BufRead>(&self, reader: R) -> Box<dyn Iterator<Item = TurtleResult<Quad>>> {
        // For streaming, we'll return all quads at once for now
        // A proper implementation would use a streaming parser
        match self.parse_trig_content(reader) {
            Ok(quads) => Box::new(quads.into_iter().map(Ok)),
            Err(e) => Box::new(std::iter::once(Err(e))),
        }
    }
}

/// TriG serializer with support for named graphs
#[derive(Debug, Clone)]
pub struct TriGSerializer {
    /// Base IRI for relative IRI serialization
    pub base_iri: Option<String>,
    /// Prefix declarations for compact serialization
    pub prefixes: HashMap<String, String>,
}

impl Default for TriGSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl TriGSerializer {
    /// Create a new TriG serializer
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();

        // Add common prefixes
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );

        Self {
            base_iri: None,
            prefixes,
        }
    }

    /// Serialize a quad to TriG format
    fn serialize_quad<W: Write>(&self, quad: &Quad, writer: &mut W) -> TurtleResult<()> {
        // Serialize subject
        self.serialize_subject(quad.subject(), writer)?;
        write!(writer, " ").map_err(TurtleParseError::Io)?;

        // Serialize predicate
        self.serialize_predicate(quad.predicate(), writer)?;
        write!(writer, " ").map_err(TurtleParseError::Io)?;

        // Serialize object
        self.serialize_object(quad.object(), writer)?;

        Ok(())
    }

    /// Serialize a subject term
    fn serialize_subject<W: Write>(
        &self,
        subject: &oxirs_core::model::Subject,
        writer: &mut W,
    ) -> TurtleResult<()> {
        use oxirs_core::model::Subject;

        match subject {
            Subject::NamedNode(node) => {
                self.serialize_named_node(node, writer)?;
            }
            Subject::BlankNode(node) => {
                write!(writer, "_:{}", node.as_str()).map_err(TurtleParseError::Io)?;
            }
            Subject::QuotedTriple(triple) => {
                // RDF-star quoted triple
                write!(writer, "<< ").map_err(TurtleParseError::Io)?;
                self.serialize_subject(triple.subject(), writer)?;
                write!(writer, " ").map_err(TurtleParseError::Io)?;
                self.serialize_predicate(triple.predicate(), writer)?;
                write!(writer, " ").map_err(TurtleParseError::Io)?;
                self.serialize_object(triple.object(), writer)?;
                write!(writer, " >>").map_err(TurtleParseError::Io)?;
            }
            Subject::Variable(var) => {
                write!(writer, "?{}", var.name()).map_err(TurtleParseError::Io)?;
            }
        }
        Ok(())
    }

    /// Serialize a predicate term
    fn serialize_predicate<W: Write>(
        &self,
        predicate: &oxirs_core::model::Predicate,
        writer: &mut W,
    ) -> TurtleResult<()> {
        use oxirs_core::model::Predicate;

        match predicate {
            Predicate::NamedNode(node) => {
                // Check for rdf:type abbreviation
                if node.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                    write!(writer, "a").map_err(TurtleParseError::Io)?;
                } else {
                    self.serialize_named_node(node, writer)?;
                }
            }
            Predicate::Variable(var) => {
                write!(writer, "?{}", var.name()).map_err(TurtleParseError::Io)?;
            }
        }
        Ok(())
    }

    /// Serialize an object term
    fn serialize_object<W: Write>(
        &self,
        object: &oxirs_core::model::Object,
        writer: &mut W,
    ) -> TurtleResult<()> {
        use oxirs_core::model::Object;

        match object {
            Object::NamedNode(node) => {
                self.serialize_named_node(node, writer)?;
            }
            Object::BlankNode(node) => {
                write!(writer, "_:{}", node.as_str()).map_err(TurtleParseError::Io)?;
            }
            Object::Literal(literal) => {
                self.serialize_literal(literal, writer)?;
            }
            Object::QuotedTriple(triple) => {
                // RDF-star quoted triple
                write!(writer, "<< ").map_err(TurtleParseError::Io)?;
                self.serialize_subject(triple.subject(), writer)?;
                write!(writer, " ").map_err(TurtleParseError::Io)?;
                self.serialize_predicate(triple.predicate(), writer)?;
                write!(writer, " ").map_err(TurtleParseError::Io)?;
                self.serialize_object(triple.object(), writer)?;
                write!(writer, " >>").map_err(TurtleParseError::Io)?;
            }
            Object::Variable(var) => {
                write!(writer, "?{}", var.name()).map_err(TurtleParseError::Io)?;
            }
        }
        Ok(())
    }

    /// Serialize a named node, using prefixes if possible
    fn serialize_named_node<W: Write>(&self, node: &NamedNode, writer: &mut W) -> TurtleResult<()> {
        let iri = node.as_str();

        // Try to use a prefix
        for (prefix, namespace) in &self.prefixes {
            if iri.starts_with(namespace) {
                let local = &iri[namespace.len()..];
                write!(writer, "{prefix}:{local}").map_err(TurtleParseError::Io)?;
                return Ok(());
            }
        }

        // Use full IRI
        write!(writer, "<{iri}>").map_err(TurtleParseError::Io)?;
        Ok(())
    }

    /// Serialize a literal
    fn serialize_literal<W: Write>(&self, literal: &Literal, writer: &mut W) -> TurtleResult<()> {
        let value = literal.value();

        // Escape quotes in the value
        let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
        write!(writer, "\"{escaped}\"").map_err(TurtleParseError::Io)?;

        // Add language tag if present
        if let Some(lang) = literal.language() {
            write!(writer, "@{lang}").map_err(TurtleParseError::Io)?;
        }
        // Add datatype if not a string literal
        else if literal.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
            write!(writer, "^^").map_err(TurtleParseError::Io)?;
            self.serialize_named_node(&literal.datatype().into_owned(), writer)?;
        }

        Ok(())
    }

    /// Group quads by graph for efficient serialization
    fn group_quads_by_graph<'a>(
        &self,
        quads: &'a [Quad],
    ) -> std::collections::BTreeMap<GraphName, Vec<&'a Quad>> {
        let mut grouped = std::collections::BTreeMap::new();

        for quad in quads {
            grouped
                .entry(quad.graph_name().clone())
                .or_insert_with(Vec::new)
                .push(quad);
        }

        grouped
    }
}

impl Serializer<Quad> for TriGSerializer {
    fn serialize<W: Write>(&self, quads: &[Quad], mut writer: W) -> TurtleResult<()> {
        // Write prefix declarations
        for (prefix, namespace) in &self.prefixes {
            writeln!(writer, "@prefix {prefix}: <{namespace}> .").map_err(TurtleParseError::Io)?;
        }

        if !self.prefixes.is_empty() {
            writeln!(writer).map_err(TurtleParseError::Io)?;
        }

        // Group quads by graph
        let grouped = self.group_quads_by_graph(quads);

        for (graph_name, graph_quads) in grouped {
            match graph_name {
                GraphName::DefaultGraph => {
                    // Serialize default graph triples directly
                    for quad in graph_quads {
                        self.serialize_quad(quad, &mut writer)?;
                        writeln!(writer, " .").map_err(TurtleParseError::Io)?;
                    }
                }
                GraphName::NamedNode(node) => {
                    // Named graph
                    self.serialize_named_node(&node, &mut writer)?;
                    writeln!(writer, " {{").map_err(TurtleParseError::Io)?;

                    for quad in graph_quads {
                        write!(writer, "    ").map_err(TurtleParseError::Io)?;
                        self.serialize_quad(quad, &mut writer)?;
                        writeln!(writer, " .").map_err(TurtleParseError::Io)?;
                    }

                    writeln!(writer, "}}").map_err(TurtleParseError::Io)?;
                }
                GraphName::BlankNode(node) => {
                    // Blank node graph
                    writeln!(writer, "_:{} {{", node.as_str()).map_err(TurtleParseError::Io)?;

                    for quad in graph_quads {
                        write!(writer, "    ").map_err(TurtleParseError::Io)?;
                        self.serialize_quad(quad, &mut writer)?;
                        writeln!(writer, " .").map_err(TurtleParseError::Io)?;
                    }

                    writeln!(writer, "}}").map_err(TurtleParseError::Io)?;
                }
                GraphName::Variable(_) => {
                    // Variables shouldn't appear in concrete data
                    return Err(TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                        message: "Cannot serialize variable graph names".to_string(),
                        position: TextPosition::start(),
                    }));
                }
            }

            writeln!(writer).map_err(TurtleParseError::Io)?;
        }

        Ok(())
    }

    fn serialize_item<W: Write>(&self, quad: &Quad, mut writer: W) -> TurtleResult<()> {
        match quad.graph_name() {
            GraphName::DefaultGraph => {
                // Default graph - serialize as simple triple
                self.serialize_quad(quad, &mut writer)?;
                writeln!(writer, " .").map_err(TurtleParseError::Io)?;
            }
            GraphName::NamedNode(node) => {
                // Named graph - use GRAPH syntax for individual quad
                write!(writer, "GRAPH ").map_err(TurtleParseError::Io)?;
                self.serialize_named_node(node, &mut writer)?;
                write!(writer, " {{ ").map_err(TurtleParseError::Io)?;
                self.serialize_quad(quad, &mut writer)?;
                writeln!(writer, " . }}").map_err(TurtleParseError::Io)?;
            }
            GraphName::BlankNode(node) => {
                // Blank node graph
                write!(writer, "_:{} {{ ", node.as_str()).map_err(TurtleParseError::Io)?;
                self.serialize_quad(quad, &mut writer)?;
                writeln!(writer, " . }}").map_err(TurtleParseError::Io)?;
            }
            GraphName::Variable(_) => {
                return Err(TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: "Cannot serialize variable graph names".to_string(),
                    position: TextPosition::start(),
                }));
            }
        }
        Ok(())
    }
}
