//! TriG format parser and serializer
//!
//! TriG (Terse RDF Triple Language + Named Graphs) is an extension of Turtle
//! that adds support for named graphs. It allows you to group triples into
//! different graphs within a single document.
//!
//! # Format Overview
//!
//! - **Default Graph**: Triples outside any graph block
//! - **Named Graphs**: `<graph-iri> { triples... }`
//! - **GRAPH Keyword**: `GRAPH <graph-iri> { triples... }`
//! - **Prefixes**: Same as Turtle (`@prefix ex: <http://example.org/> .`)
//! - **Base IRI**: Same as Turtle (`@base <http://example.org/> .`)
//!
//! # Examples
//!
//! ## Basic TriG Parsing
//!
//! ```rust
//! use oxirs_ttl::trig::TriGParser;
//! use oxirs_ttl::Parser;
//! use std::io::Cursor;
//!
//! let trig_data = r#"
//! @prefix ex: <http://example.org/> .
//! ex:alice ex:knows ex:bob .
//! ex:graph1 {
//!     ex:alice ex:age 30 .
//!     ex:bob ex:age 28 .
//! }
//! "#;
//!
//! let parser = TriGParser::new();
//! let quads = parser.parse(Cursor::new(trig_data))?;
//! assert!(quads.len() >= 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Multiple Named Graphs
//!
//! ```rust
//! use oxirs_ttl::trig::TriGParser;
//! use oxirs_ttl::Parser;
//! use std::io::Cursor;
//!
//! let trig_data = r#"
//! @prefix ex: <http://example.org/> .
//! @prefix foaf: <http://xmlns.com/foaf/0.1/> .
//!
//! ex:people {
//!     ex:alice foaf:name "Alice" .
//!     ex:bob foaf:name "Bob" .
//! }
//!
//! ex:connections {
//!     ex:alice foaf:knows ex:bob .
//!     ex:bob foaf:knows ex:charlie .
//! }
//! "#;
//!
//! let parser = TriGParser::new();
//! let quads = parser.parse(Cursor::new(trig_data))?;
//!
//! // Check that we have quads in different graphs
//! let graphs: std::collections::HashSet<_> = quads.iter()
//!     .map(|q| q.graph_name())
//!     .collect();
//! assert!(graphs.len() >= 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Using GRAPH Keyword
//!
//! ```rust
//! use oxirs_ttl::trig::TriGParser;
//! use oxirs_ttl::Parser;
//! use std::io::Cursor;
//!
//! let trig_data = r#"
//! @prefix ex: <http://example.org/> .
//!
//! GRAPH ex:metadata {
//!     ex:document ex:createdAt "2025-11-29" .
//!     ex:document ex:author "Alice" .
//! }
//! "#;
//!
//! let parser = TriGParser::new();
//! let quads = parser.parse(Cursor::new(trig_data))?;
//! assert_eq!(quads.len(), 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Advanced Turtle Syntax in Graphs
//!
//! ```rust
//! use oxirs_ttl::trig::TriGParser;
//! use oxirs_ttl::Parser;
//! use std::io::Cursor;
//!
//! let trig_data = r#"
//! @prefix ex: <http://example.org/> .
//! ex:graph1 {
//!     ex:alice ex:name "Alice" ;
//!              ex:age 30 ;
//!              ex:city "Wonderland" .
//!     ex:bob ex:knows ex:alice, ex:charlie .
//!     ex:list ex:items (1 2 3 4 5) .
//! }
//! "#;
//!
//! let parser = TriGParser::new();
//! let quads = parser.parse(Cursor::new(trig_data))?;
//! assert!(quads.len() >= 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Default and Named Graphs Mixed
//!
//! ```rust
//! use oxirs_ttl::trig::TriGParser;
//! use oxirs_ttl::Parser;
//! use std::io::Cursor;
//!
//! let trig_data = r#"
//! @prefix ex: <http://example.org/> .
//! ex:alice ex:type ex:Person .
//! ex:bob ex:type ex:Person .
//! ex:relationships {
//!     ex:alice ex:knows ex:bob .
//! }
//! ex:charlie ex:type ex:Person .
//! "#;
//!
//! let parser = TriGParser::new();
//! let quads = parser.parse(Cursor::new(trig_data))?;
//!
//! let default_graph_count = quads.iter()
//!     .filter(|q| q.graph_name().is_default_graph())
//!     .count();
//! assert!(default_graph_count >= 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::formats::turtle::TurtleParser;
use crate::toolkit::{Parser, Serializer};
use oxirs_core::model::{BlankNode, GraphName, Literal, NamedNode, Quad, Triple};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};

/// TriG parser with support for named graphs
///
/// TriG extends Turtle with the ability to group triples into named graphs.
/// This parser supports both default graph triples and named graph syntax.
///
/// # Examples
///
/// ```rust
/// use oxirs_ttl::trig::TriGParser;
/// use oxirs_ttl::Parser;
/// use std::io::Cursor;
///
/// let trig = r#"
/// @prefix ex: <http://example.org/> .
/// ex:graph1 {
///     ex:s ex:p ex:o .
/// }
/// "#;
///
/// let parser = TriGParser::new();
/// let quads = parser.parse(Cursor::new(trig))?;
/// assert!(!quads.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
        let mut graph_depth = 0; // Track whether we're inside a named graph
        let mut prefixes = self.prefixes.clone();
        let mut base_iri = self.base_iri.clone();

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
                // Parse: @prefix ex: <http://example.org/> .
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let prefix = parts[1].trim_end_matches(':');
                    let iri = parts[2]
                        .trim_start_matches('<')
                        .trim_end_matches('>')
                        .trim_end_matches('.');
                    // Resolve prefix IRI against base if it's relative
                    let resolved_iri = Self::resolve_iri(iri, &base_iri);
                    prefixes.insert(prefix.to_string(), resolved_iri);
                }
                i += 1;
                continue;
            }

            // Handle base declarations
            if line.starts_with("@base") {
                // Parse: @base <http://example.org/> .
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let iri = parts[1]
                        .trim_start_matches('<')
                        .trim_end_matches('>')
                        .trim_end_matches('.');
                    base_iri = Some(iri.to_string());
                }
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
                    current_graph = self
                        .parse_graph_name_with_prefixes_and_base(graph_iri, &prefixes, &base_iri)?;
                } else if !graph_part.is_empty() {
                    // <iri> { syntax
                    current_graph = self.parse_graph_name_with_prefixes_and_base(
                        graph_part, &prefixes, &base_iri,
                    )?;
                } else {
                    current_graph = GraphName::DefaultGraph;
                }

                graph_depth += 1;
                i += 1;
                continue;
            }

            // Check for graph end
            if line.trim() == "}" {
                if graph_depth == 0 {
                    return Err(TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                        message: "Unexpected closing brace '}'".to_string(),
                        position: TextPosition::default(),
                    }));
                }
                graph_depth -= 1;
                current_graph = GraphName::DefaultGraph;
                i += 1;
                continue;
            }

            // Accumulate multi-line statements (handle semicolons and commas)
            if !line.is_empty()
                && !line.starts_with('@')
                && !line.contains('{')
                && line.trim() != "}"
            {
                // Strip inline comments from the line
                let line_without_comment = if let Some(comment_pos) = line.find('#') {
                    line[..comment_pos].trim_end()
                } else {
                    line
                };

                let mut statement = line_without_comment.to_string();

                // Track if we're inside a multiline string literal
                let count_triple_quotes =
                    |s: &str| s.matches("\"\"\"").count() + s.matches("'''").count();

                // Continue accumulating lines until we find one ending with '.'
                // but only if we're not inside a multiline string literal
                while i + 1 < lines.len() {
                    let inside_multiline = count_triple_quotes(&statement) % 2 == 1;
                    if !inside_multiline && statement.trim_end().ends_with('.') {
                        break;
                    }
                    i += 1;
                    let next_line = lines[i].trim();
                    // Stop if we hit a graph closing brace
                    if next_line == "}" {
                        i -= 1; // Back up so the main loop processes the '}'
                        break;
                    }
                    if !next_line.is_empty() && !next_line.starts_with('#') {
                        // Strip inline comments from the next line as well
                        let next_line_without_comment =
                            if let Some(comment_pos) = next_line.find('#') {
                                next_line[..comment_pos].trim_end()
                            } else {
                                next_line
                            };

                        if !next_line_without_comment.is_empty() {
                            // Preserve newlines for multiline string literals
                            statement.push('\n');
                            statement.push_str(next_line_without_comment);
                        }
                    }
                }

                // Parse the complete statement using Turtle parser
                if statement.trim_end().ends_with('.') {
                    match self.parse_triple_with_turtle(&statement, &prefixes, &base_iri) {
                        Ok(triples) => {
                            for triple in triples {
                                let quad = Quad::new(
                                    triple.subject().clone(),
                                    triple.predicate().clone(),
                                    triple.object().clone(),
                                    current_graph.clone(),
                                );
                                quads.push(quad);
                            }
                        }
                        Err(_e) if self.lenient => {
                            // In lenient mode, skip failed triples
                        }
                        Err(e) => return Err(e),
                    }
                } else if !statement.trim().is_empty() && !self.lenient {
                    // Statement doesn't end with '.' - this is invalid syntax
                    return Err(TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                        message: format!("Statement must end with '.': {}", statement.trim()),
                        position: TextPosition::default(),
                    }));
                }
            }

            i += 1;
        }

        // Check for unclosed graphs
        if graph_depth > 0 && !self.lenient {
            return Err(TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                message: "Unclosed graph: missing closing brace '}'".to_string(),
                position: TextPosition::default(),
            }));
        }

        Ok(quads)
    }

    /// Parse triple(s) using Turtle parser for full syntax support
    fn parse_triple_with_turtle(
        &self,
        content: &str,
        prefixes: &HashMap<String, String>,
        base_iri: &Option<String>,
    ) -> TurtleResult<Vec<Triple>> {
        // Create a complete Turtle document by prepending prefix/base declarations
        let mut document = String::new();

        // Add base declaration if present
        if let Some(base) = base_iri {
            document.push_str(&format!("@base <{}> .\n", base));
        }

        // Add prefix declarations
        for (prefix, iri) in prefixes {
            document.push_str(&format!("@prefix {}: <{}> .\n", prefix, iri));
        }

        // Add the statement
        document.push_str(content);

        let mut turtle_parser = TurtleParser::new();
        if self.lenient {
            turtle_parser.lenient = true;
        }

        turtle_parser.parse_document(&document)
    }

    // Legacy parsing methods - kept for potential future use but currently unused
    // as we now use TurtleParser for all triple parsing
    #[allow(dead_code)]
    fn parse_simple_triple(&self, line: &str) -> TurtleResult<Triple> {
        self.parse_simple_triple_with_prefixes(line, &self.prefixes)
    }

    #[allow(dead_code)]
    fn parse_simple_triple_with_prefixes(
        &self,
        line: &str,
        prefixes: &HashMap<String, String>,
    ) -> TurtleResult<Triple> {
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
        let subject = self.parse_term_as_subject_with_prefixes(parts[0], prefixes)?;
        let predicate = self.parse_term_as_predicate_with_prefixes(parts[1], prefixes)?;
        let object = self.parse_term_as_object_with_prefixes(&parts[2..].join(" "), prefixes)?;

        Ok(Triple::new(subject, predicate, object))
    }

    #[allow(dead_code)]
    fn parse_graph_name(&self, graph_str: &str) -> TurtleResult<GraphName> {
        self.parse_graph_name_with_prefixes(graph_str, &self.prefixes)
    }

    /// Parse a graph name with provided prefixes
    fn parse_graph_name_with_prefixes(
        &self,
        graph_str: &str,
        prefixes: &HashMap<String, String>,
    ) -> TurtleResult<GraphName> {
        self.parse_graph_name_with_prefixes_and_base(graph_str, prefixes, &None)
    }

    /// Parse a graph name with provided prefixes and base IRI
    fn parse_graph_name_with_prefixes_and_base(
        &self,
        graph_str: &str,
        prefixes: &HashMap<String, String>,
        base_iri: &Option<String>,
    ) -> TurtleResult<GraphName> {
        let graph_str = graph_str.trim();

        if graph_str.starts_with('<') && graph_str.ends_with('>') {
            let iri = graph_str.trim_start_matches('<').trim_end_matches('>');
            // Resolve relative IRI against base if needed
            let resolved_iri = Self::resolve_iri(iri, base_iri);
            let named_node = NamedNode::new(&resolved_iri).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid IRI: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(GraphName::NamedNode(named_node))
        } else if graph_str.starts_with("_:") {
            // Handle blank nodes as graph names
            let label = graph_str.trim_start_matches("_:");
            let blank_node = BlankNode::new(label).map_err(|e| {
                TurtleParseError::Syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid blank node: {e}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(GraphName::BlankNode(blank_node))
        } else if graph_str.contains(':') {
            // Handle prefixed names
            let expanded = Self::expand_prefixed_name_static(graph_str, prefixes)?;
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

    /// Resolve a potentially relative IRI against a base IRI
    fn resolve_iri(iri: &str, base_iri: &Option<String>) -> String {
        // If IRI is already absolute (has a scheme), return as-is
        if iri.contains("://")
            || iri.starts_with("http:")
            || iri.starts_with("https:")
            || iri.starts_with("urn:")
        {
            return iri.to_string();
        }

        // If we have a base IRI, resolve against it
        if let Some(base) = base_iri {
            // Simple resolution: concatenate base + relative
            // For a production system, you'd want proper IRI resolution per RFC 3986
            if base.ends_with('/') || base.ends_with('#') {
                format!("{}{}", base, iri)
            } else {
                format!("{}/{}", base, iri)
            }
        } else {
            // No base IRI, return as-is
            iri.to_string()
        }
    }

    #[allow(dead_code)]
    fn expand_prefixed_name(&self, prefixed: &str) -> TurtleResult<String> {
        Self::expand_prefixed_name_static(prefixed, &self.prefixes)
    }

    /// Static version of expand_prefixed_name
    fn expand_prefixed_name_static(
        prefixed: &str,
        prefixes: &HashMap<String, String>,
    ) -> TurtleResult<String> {
        if let Some(colon_pos) = prefixed.find(':') {
            let prefix = &prefixed[..colon_pos];
            let local = &prefixed[colon_pos + 1..];

            if let Some(namespace) = prefixes.get(prefix) {
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

    #[allow(dead_code)]
    fn parse_term_as_subject(&self, term: &str) -> TurtleResult<oxirs_core::model::Subject> {
        self.parse_term_as_subject_with_prefixes(term, &self.prefixes)
    }

    #[allow(dead_code)]
    fn parse_term_as_subject_with_prefixes(
        &self,
        term: &str,
        prefixes: &HashMap<String, String>,
    ) -> TurtleResult<oxirs_core::model::Subject> {
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
            let expanded = Self::expand_prefixed_name_static(term, prefixes)?;
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

    #[allow(dead_code)]
    fn parse_term_as_predicate(&self, term: &str) -> TurtleResult<oxirs_core::model::Predicate> {
        self.parse_term_as_predicate_with_prefixes(term, &self.prefixes)
    }

    #[allow(dead_code)]
    fn parse_term_as_predicate_with_prefixes(
        &self,
        term: &str,
        prefixes: &HashMap<String, String>,
    ) -> TurtleResult<oxirs_core::model::Predicate> {
        use oxirs_core::model::Predicate;

        if term == "a" {
            // Handle rdf:type abbreviation
            let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                .expect("valid IRI");
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
            let expanded = Self::expand_prefixed_name_static(term, prefixes)?;
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

    #[allow(dead_code)]
    fn parse_term_as_object(&self, term: &str) -> TurtleResult<oxirs_core::model::Object> {
        self.parse_term_as_object_with_prefixes(term, &self.prefixes)
    }

    #[allow(dead_code)]
    fn parse_term_as_object_with_prefixes(
        &self,
        term: &str,
        prefixes: &HashMap<String, String>,
    ) -> TurtleResult<oxirs_core::model::Object> {
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
            let expanded = Self::expand_prefixed_name_static(term, prefixes)?;
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
