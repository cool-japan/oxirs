//! Turtle Format Parser and Serializer
//!
//! Extracted and adapted from OxiGraph oxttl with OxiRS enhancements.
//! Based on W3C Turtle specification: <https://www.w3.org/TR/turtle/>

use super::error::SerializeResult;
use super::error::{ParseResult, RdfParseError};
use super::serializer::QuadSerializer;
use crate::model::{QuadRef, Triple, TripleRef};
use std::collections::HashMap;
use std::io::{Read, Write};

/// Turtle parser implementation
#[derive(Debug, Clone)]
pub struct TurtleParser {
    lenient: bool,
    base_iri: Option<String>,
    prefixes: HashMap<String, String>,
}

impl TurtleParser {
    /// Create a new Turtle parser
    pub fn new() -> Self {
        Self {
            lenient: false,
            base_iri: None,
            prefixes: HashMap::new(),
        }
    }

    /// Enable lenient parsing (skip some validations)
    pub fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }

    /// Set base IRI for resolving relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        self.base_iri = Some(base_iri.into());
        self
    }

    /// Add a namespace prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.insert(prefix.into(), iri.into());
        self
    }

    /// Parse Turtle from a reader
    pub fn parse_reader<R: Read>(&self, mut reader: R) -> ParseResult<Vec<Triple>> {
        // Read all data from the reader
        let mut buffer = String::new();
        reader.read_to_string(&mut buffer)?;

        // Use the string parser (handles basic Turtle syntax)
        // Note: Current implementation handles simple triples, prefixes, and base directives
        // Advanced Turtle features (collections, lists, multi-line literals) are partially supported
        self.parse_str(&buffer)
    }

    /// Parse Turtle from a byte slice
    pub fn parse_slice(&self, slice: &[u8]) -> ParseResult<Vec<Triple>> {
        // Convert to string and parse
        // Note: Future optimization could use zero-copy parsing with byte-level operations
        let content = std::str::from_utf8(slice)
            .map_err(|e| RdfParseError::syntax(format!("Invalid UTF-8: {e}")))?;

        self.parse_str(content)
    }

    /// Parse Turtle from a string
    pub fn parse_str(&self, input: &str) -> ParseResult<Vec<Triple>> {
        use super::parser::helpers::convert_quad;
        use std::io::Cursor;

        // Build oxttl parser with configuration
        let mut oxttl_parser = oxttl::TurtleParser::new();

        // Apply base IRI if set
        if let Some(ref base) = self.base_iri {
            oxttl_parser = oxttl_parser
                .with_base_iri(base.as_str())
                .unwrap_or_else(|_| oxttl::TurtleParser::new());
        }

        // Enable lenient mode if requested
        if self.lenient {
            oxttl_parser = oxttl_parser.lenient();
        }

        // Parse and collect triples
        let reader = Cursor::new(input.as_bytes());
        let mut triples = Vec::new();

        for result in oxttl_parser.for_reader(reader) {
            match result {
                Ok(triple) => {
                    // Convert oxrdf Triple to oxirs Triple via Quad
                    let quad = oxrdf::Quad::new(
                        triple.subject,
                        triple.predicate,
                        triple.object,
                        oxrdf::GraphName::DefaultGraph,
                    );
                    let oxirs_quad = convert_quad(quad)?;
                    triples.push(oxirs_quad.to_triple());
                }
                Err(e) => {
                    if !self.lenient {
                        return Err(RdfParseError::syntax(e.to_string()));
                    }
                    // In lenient mode, skip errors
                }
            }
        }

        Ok(triples)
    }

    /// Get current prefixes
    pub fn prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }

    /// Get current base IRI
    pub fn base_iri(&self) -> Option<&str> {
        self.base_iri.as_deref()
    }

    /// Check if lenient parsing is enabled
    pub fn is_lenient(&self) -> bool {
        self.lenient
    }
}

impl Default for TurtleParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Turtle serializer implementation
#[derive(Debug, Clone)]
pub struct TurtleSerializer {
    base_iri: Option<String>,
    prefixes: HashMap<String, String>,
    pretty: bool,
}

impl TurtleSerializer {
    /// Create a new Turtle serializer
    pub fn new() -> Self {
        Self {
            base_iri: None,
            prefixes: HashMap::new(),
            pretty: false,
        }
    }

    /// Set base IRI for generating relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        self.base_iri = Some(base_iri.into());
        self
    }

    /// Add a namespace prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.insert(prefix.into(), iri.into());
        self
    }

    /// Enable pretty formatting
    pub fn pretty(mut self) -> Self {
        self.pretty = true;
        self
    }

    /// Create a writer-based serializer
    pub fn for_writer<W: Write>(self, writer: W) -> WriterTurtleSerializer<W> {
        WriterTurtleSerializer::new(writer, self)
    }

    /// Serialize triples to a string
    pub fn serialize_to_string(&self, triples: &[Triple]) -> SerializeResult<String> {
        let mut buffer = Vec::new();
        {
            let mut serializer = self.clone().for_writer(&mut buffer);
            for triple in triples {
                serializer.serialize_triple(triple.as_ref())?;
            }
            serializer.finish()?;
        }
        String::from_utf8(buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Get the prefixes
    pub fn prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }

    /// Get the base IRI
    pub fn base_iri(&self) -> Option<&str> {
        self.base_iri.as_deref()
    }

    /// Check if pretty formatting is enabled
    pub fn is_pretty(&self) -> bool {
        self.pretty
    }
}

impl Default for TurtleSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Writer-based Turtle serializer
pub struct WriterTurtleSerializer<W: Write> {
    writer: W,
    config: TurtleSerializer,
    headers_written: bool,
}

impl<W: Write> WriterTurtleSerializer<W> {
    /// Create a new writer serializer
    pub fn new(writer: W, config: TurtleSerializer) -> Self {
        Self {
            writer,
            config,
            headers_written: false,
        }
    }

    /// Serialize a triple
    pub fn serialize_triple(&mut self, triple: TripleRef<'_>) -> SerializeResult<()> {
        self.ensure_headers_written()?;

        // Subject serialization
        let subject_str = self.serialize_subject(triple.subject())?;

        // Predicate serialization
        let predicate_str = self.serialize_predicate(triple.predicate())?;

        // Object serialization
        let object_str = self.serialize_object(triple.object())?;

        // Write the triple with proper formatting
        if self.config.pretty {
            writeln!(self.writer, "{subject_str} {predicate_str} {object_str} .")?;
        } else {
            writeln!(self.writer, "{subject_str} {predicate_str} {object_str}.")?;
        }

        Ok(())
    }

    /// Serialize a subject (NamedNode, BlankNode, or Variable)
    fn serialize_subject(&self, subject: crate::model::SubjectRef<'_>) -> SerializeResult<String> {
        use crate::model::SubjectRef;

        match subject {
            SubjectRef::NamedNode(node) => self.serialize_named_node(node.into()),
            SubjectRef::BlankNode(node) => {
                let node_str = node.as_str();
                Ok(format!("_:{node_str}"))
            }
            SubjectRef::Variable(var) => {
                let var_str = var.as_str();
                Ok(format!("?{var_str}"))
            }
        }
    }

    /// Serialize a predicate (NamedNode or Variable)
    fn serialize_predicate(
        &self,
        predicate: crate::model::PredicateRef<'_>,
    ) -> SerializeResult<String> {
        use crate::model::PredicateRef;

        match predicate {
            PredicateRef::NamedNode(node) => {
                // Check for rdf:type shorthand
                if node.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                    Ok("a".to_string())
                } else {
                    self.serialize_named_node(node.into())
                }
            }
            PredicateRef::Variable(var) => {
                let var_str = var.as_str();
                Ok(format!("?{var_str}"))
            }
        }
    }

    /// Serialize an object (NamedNode, BlankNode, Literal, or Variable)
    fn serialize_object(&self, object: crate::model::ObjectRef<'_>) -> SerializeResult<String> {
        use crate::model::ObjectRef;

        match object {
            ObjectRef::NamedNode(node) => self.serialize_named_node(node.into()),
            ObjectRef::BlankNode(node) => {
                let node_str = node.as_str();
                Ok(format!("_:{node_str}"))
            }
            ObjectRef::Literal(literal) => self.serialize_literal(literal),
            ObjectRef::Variable(var) => {
                let var_str = var.as_str();
                Ok(format!("?{var_str}"))
            }
        }
    }

    /// Serialize a named node with prefix abbreviation
    fn serialize_named_node(
        &self,
        node: crate::model::NamedNodeRef<'_>,
    ) -> SerializeResult<String> {
        let iri = node.as_str();

        // Try to find a matching prefix
        for (prefix, namespace) in &self.config.prefixes {
            if iri.starts_with(namespace) {
                let local = &iri[namespace.len()..];
                // Check if local part is valid for prefixed name
                if is_valid_local_name(local) {
                    return Ok(format!("{prefix}:{local}"));
                }
            }
        }

        // Fall back to full IRI in angle brackets
        Ok(format!("<{iri}>"))
    }

    /// Serialize a literal
    fn serialize_literal(&self, literal: &crate::model::Literal) -> SerializeResult<String> {
        let value = literal.value();

        // Escape special characters in the string
        let escaped_value = escape_turtle_string(value);

        // Handle language tag
        if let Some(lang) = literal.language() {
            return Ok(format!("\"{escaped_value}\"@{lang}"));
        }

        // Handle datatype
        let datatype = literal.datatype();
        if datatype.as_str() == "http://www.w3.org/2001/XMLSchema#string" {
            // XSD string is the default, no need to specify
            Ok(format!("\"{escaped_value}\""))
        } else {
            // Serialize datatype as IRI
            let datatype_str = self.serialize_named_node(datatype)?;
            Ok(format!("\"{escaped_value}\"^^{datatype_str}"))
        }
    }

    /// Finish serialization and return the writer
    pub fn finish(self) -> SerializeResult<W> {
        Ok(self.writer)
    }

    /// Ensure headers (prefixes, base) are written
    fn ensure_headers_written(&mut self) -> SerializeResult<()> {
        if self.headers_written {
            return Ok(());
        }

        // Write base directive
        if let Some(base) = &self.config.base_iri {
            writeln!(self.writer, "@base <{base}> .")?;
        }

        // Write prefix directives
        for (prefix, iri) in &self.config.prefixes {
            writeln!(self.writer, "@prefix {prefix}: <{iri}> .")?;
        }

        // Add blank line after headers if we wrote any
        if self.config.base_iri.is_some() || !self.config.prefixes.is_empty() {
            writeln!(self.writer)?;
        }

        self.headers_written = true;
        Ok(())
    }
}

impl<W: Write> QuadSerializer<W> for WriterTurtleSerializer<W> {
    fn serialize_quad(&mut self, quad: QuadRef<'_>) -> SerializeResult<()> {
        // Turtle only supports default graph, so ignore named graphs
        if quad.graph_name().is_default_graph() {
            self.serialize_triple(quad.triple())
        } else {
            // Could log a warning here about ignoring named graph
            Ok(())
        }
    }

    fn finish(self: Box<Self>) -> SerializeResult<W> {
        Ok(self.writer)
    }
}

/// Check if a string is a valid local name for Turtle prefixed names
fn is_valid_local_name(local: &str) -> bool {
    if local.is_empty() {
        return true; // Empty local names are allowed
    }

    // First character must be a name start char or underscore
    let first_char = local
        .chars()
        .next()
        .expect("local name validated to be non-empty");
    if !is_pn_chars_base(first_char) && first_char != '_' {
        return false;
    }

    // Rest of characters must be name chars, underscore, dot, or hyphen
    for ch in local.chars().skip(1) {
        if !is_pn_chars(ch) && ch != '.' && ch != '-' {
            return false;
        }
    }

    // Cannot end with a dot
    !local.ends_with('.')
}

/// Check if character is a PN_CHARS_BASE (per Turtle grammar)
fn is_pn_chars_base(ch: char) -> bool {
    ch.is_ascii_alphabetic()
        || ('\u{00C0}'..='\u{00D6}').contains(&ch)
        || ('\u{00D8}'..='\u{00F6}').contains(&ch)
        || ('\u{00F8}'..='\u{02FF}').contains(&ch)
        || ('\u{0370}'..='\u{037D}').contains(&ch)
        || ('\u{037F}'..='\u{1FFF}').contains(&ch)
        || ('\u{200C}'..='\u{200D}').contains(&ch)
        || ('\u{2070}'..='\u{218F}').contains(&ch)
        || ('\u{2C00}'..='\u{2FEF}').contains(&ch)
        || ('\u{3001}'..='\u{D7FF}').contains(&ch)
        || ('\u{F900}'..='\u{FDCF}').contains(&ch)
        || ('\u{FDF0}'..='\u{FFFD}').contains(&ch)
}

/// Check if character is a PN_CHARS (per Turtle grammar)
fn is_pn_chars(ch: char) -> bool {
    is_pn_chars_base(ch)
        || ch == '_'
        || ch.is_ascii_digit()
        || ch == '\u{00B7}'
        || ('\u{0300}'..='\u{036F}').contains(&ch)
        || ('\u{203F}'..='\u{2040}').contains(&ch)
}

/// Escape special characters in Turtle strings
fn escape_turtle_string(input: &str) -> String {
    let mut result = String::with_capacity(input.len());

    for ch in input.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\x08' => result.push_str("\\b"), // backspace
            '\x0C' => result.push_str("\\f"), // form feed
            c if c.is_control() => {
                // Escape other control characters as Unicode escape sequences
                let code = c as u32;
                result.push_str(&format!("\\u{code:04X}"));
            }
            c => result.push(c),
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turtle_parser_creation() {
        let parser = TurtleParser::new();
        assert!(!parser.is_lenient());
        assert!(parser.base_iri().is_none());
        assert!(parser.prefixes().is_empty());
    }

    #[test]
    fn test_turtle_parser_configuration() {
        let parser = TurtleParser::new()
            .lenient()
            .with_base_iri("http://example.org/")
            .with_prefix("ex", "http://example.org/ns#");

        assert!(parser.is_lenient());
        assert_eq!(parser.base_iri(), Some("http://example.org/"));
        assert_eq!(
            parser.prefixes().get("ex"),
            Some(&"http://example.org/ns#".to_string())
        );
    }

    #[test]
    fn test_turtle_serializer_creation() {
        let serializer = TurtleSerializer::new();
        assert!(!serializer.is_pretty());
        assert!(serializer.base_iri().is_none());
        assert!(serializer.prefixes().is_empty());
    }

    #[test]
    fn test_turtle_serializer_configuration() {
        let serializer = TurtleSerializer::new()
            .pretty()
            .with_base_iri("http://example.org/")
            .with_prefix("ex", "http://example.org/ns#");

        assert!(serializer.is_pretty());
        assert_eq!(serializer.base_iri(), Some("http://example.org/"));
        assert_eq!(
            serializer.prefixes().get("ex"),
            Some(&"http://example.org/ns#".to_string())
        );
    }

    #[test]
    fn test_empty_turtle_parsing() {
        let parser = TurtleParser::new();
        let result = parser.parse_str("");
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_turtle_comments() {
        let parser = TurtleParser::new();
        let turtle = "# This is a comment\n# Another comment";
        let result = parser.parse_str(turtle);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
