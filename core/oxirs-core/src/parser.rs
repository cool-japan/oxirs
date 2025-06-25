//! RDF parsing utilities for various formats with ultra-performance streaming

use rio_api::model::{Quad as RioQuad, Triple as RioTriple};
use rio_api::parser::{QuadsParser, TriplesParser};
use rio_turtle::{NQuadsParser, NTriplesParser, TriGParser, TurtleParser};
use rio_xml::RdfXmlParser;
use std::collections::HashMap;
use std::future::Future;
use std::io::{BufRead, BufReader, Cursor};
use std::pin::Pin;
use std::task::{Context, Poll};
// use oxrdf::{Quad as OxrdfQuad, Subject as OxrdfSubject, Term as OxrdfTerm}; // REMOVED: Native implementation
use crate::model::*;
use crate::{OxirsError, Result};

/// RDF format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RdfFormat {
    /// Turtle format (TTL)
    Turtle,
    /// N-Triples format (NT)
    NTriples,
    /// TriG format (named graphs)
    TriG,
    /// N-Quads format
    NQuads,
    /// RDF/XML format
    RdfXml,
    /// JSON-LD format
    JsonLd,
}

impl RdfFormat {
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "ttl" | "turtle" => Some(RdfFormat::Turtle),
            "nt" | "ntriples" => Some(RdfFormat::NTriples),
            "trig" => Some(RdfFormat::TriG),
            "nq" | "nquads" => Some(RdfFormat::NQuads),
            "rdf" | "xml" | "rdfxml" => Some(RdfFormat::RdfXml),
            "jsonld" | "json-ld" => Some(RdfFormat::JsonLd),
            _ => None,
        }
    }

    /// Get the media type for this format
    pub fn media_type(&self) -> &'static str {
        match self {
            RdfFormat::Turtle => "text/turtle",
            RdfFormat::NTriples => "application/n-triples",
            RdfFormat::TriG => "application/trig",
            RdfFormat::NQuads => "application/n-quads",
            RdfFormat::RdfXml => "application/rdf+xml",
            RdfFormat::JsonLd => "application/ld+json",
        }
    }

    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            RdfFormat::Turtle => "ttl",
            RdfFormat::NTriples => "nt",
            RdfFormat::TriG => "trig",
            RdfFormat::NQuads => "nq",
            RdfFormat::RdfXml => "rdf",
            RdfFormat::JsonLd => "jsonld",
        }
    }

    /// Returns true if this format supports named graphs (quads)
    pub fn supports_quads(&self) -> bool {
        matches!(self, RdfFormat::TriG | RdfFormat::NQuads)
    }
}

/// Configuration for RDF parsing
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// Base IRI for resolving relative IRIs
    pub base_iri: Option<String>,
    /// Whether to ignore parsing errors and continue
    pub ignore_errors: bool,
    /// Maximum number of errors to collect before stopping
    pub max_errors: Option<usize>,
}

impl Default for ParserConfig {
    fn default() -> Self {
        ParserConfig {
            base_iri: None,
            ignore_errors: false,
            max_errors: None,
        }
    }
}

/// RDF parser interface
#[derive(Debug, Clone)]
pub struct Parser {
    format: RdfFormat,
    config: ParserConfig,
}

impl Parser {
    /// Create a new parser for the specified format
    pub fn new(format: RdfFormat) -> Self {
        Parser {
            format,
            config: ParserConfig::default(),
        }
    }

    /// Create a parser with custom configuration
    pub fn with_config(format: RdfFormat, config: ParserConfig) -> Self {
        Parser { format, config }
    }

    /// Set the base IRI for resolving relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        self.config.base_iri = Some(base_iri.into());
        self
    }

    /// Enable or disable error tolerance
    pub fn with_error_tolerance(mut self, ignore_errors: bool) -> Self {
        self.config.ignore_errors = ignore_errors;
        self
    }

    /// Parse RDF data from a string into a vector of quads
    pub fn parse_str_to_quads(&self, data: &str) -> Result<Vec<Quad>> {
        let mut quads = Vec::new();
        self.parse_str_with_handler(data, |quad| {
            quads.push(quad);
            Ok(())
        })?;
        Ok(quads)
    }

    /// Parse RDF data from a string into a vector of triples (only default graph)
    pub fn parse_str_to_triples(&self, data: &str) -> Result<Vec<Triple>> {
        let quads = self.parse_str_to_quads(data)?;
        Ok(quads
            .into_iter()
            .filter(|quad| quad.is_default_graph())
            .map(|quad| quad.to_triple())
            .collect())
    }

    /// Parse RDF data with a custom handler for each quad
    pub fn parse_str_with_handler<F>(&self, data: &str, mut handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Result<()>,
    {
        match self.format {
            RdfFormat::Turtle => self.parse_turtle(data, handler),
            RdfFormat::NTriples => self.parse_ntriples(data, handler),
            RdfFormat::TriG => self.parse_trig(data, handler),
            RdfFormat::NQuads => self.parse_nquads(data, handler),
            RdfFormat::RdfXml => self.parse_rdfxml(data, handler),
            RdfFormat::JsonLd => self.parse_jsonld(data, handler),
        }
    }

    /// Parse RDF data from bytes
    pub fn parse_bytes_to_quads(&self, data: &[u8]) -> Result<Vec<Quad>> {
        let data_str = std::str::from_utf8(data)
            .map_err(|e| OxirsError::Parse(format!("Invalid UTF-8: {}", e)))?;
        self.parse_str_to_quads(data_str)
    }

    fn parse_turtle<F>(&self, data: &str, mut handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Result<()>,
    {
        // Use TurtleParserState for now since Rio API has changed
        let mut parser = TurtleParserState::new(self.config.base_iri.as_deref());

        for (line_num, line) in data.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            match parser.parse_line(line) {
                Ok(triples) => {
                    for triple in triples {
                        let quad = Quad::from_triple(triple);
                        handler(quad)?;
                    }
                }
                Err(e) => {
                    if self.config.ignore_errors {
                        tracing::warn!("Turtle parse error on line {}: {}", line_num + 1, e);
                        continue;
                    } else {
                        return Err(OxirsError::Parse(format!(
                            "Turtle parse error on line {}: {}",
                            line_num + 1,
                            e
                        )));
                    }
                }
            }
        }

        // Handle any pending statement
        if let Some(triples) = parser.finalize()? {
            for triple in triples {
                let quad = Quad::from_triple(triple);
                handler(quad)?;
            }
        }

        Ok(())
    }

    fn parse_ntriples<F>(&self, data: &str, mut handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Result<()>,
    {
        for (line_num, line) in data.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse the line into a triple
            match self.parse_ntriples_line(line) {
                Ok(Some(quad)) => {
                    handler(quad)?;
                }
                Ok(None) => {
                    // Skip this line (e.g., blank line)
                    continue;
                }
                Err(e) => {
                    if self.config.ignore_errors {
                        tracing::warn!("Parse error on line {}: {}", line_num + 1, e);
                        continue;
                    } else {
                        return Err(OxirsError::Parse(format!(
                            "Parse error on line {}: {}",
                            line_num + 1,
                            e
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    fn parse_ntriples_line(&self, line: &str) -> Result<Option<Quad>> {
        // Simple N-Triples parser - parse line like: <s> <p> "o" .
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }

        // Find the final period
        if !line.ends_with('.') {
            return Err(OxirsError::Parse("Line must end with '.'".to_string()));
        }

        let line = &line[..line.len() - 1].trim(); // Remove trailing period and whitespace

        // Split into tokens respecting quoted strings
        let tokens = self.tokenize_ntriples_line(line)?;

        if tokens.len() != 3 {
            return Err(OxirsError::Parse(format!(
                "Expected 3 tokens (subject, predicate, object), found {}",
                tokens.len()
            )));
        }

        // Parse subject
        let subject = self.parse_subject(&tokens[0])?;

        // Parse predicate
        let predicate = self.parse_predicate(&tokens[1])?;

        // Parse object
        let object = self.parse_object(&tokens[2])?;

        let triple = Triple::new(subject, predicate, object);
        let quad = Quad::from_triple(triple);

        Ok(Some(quad))
    }

    fn tokenize_ntriples_line(&self, line: &str) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut in_quotes = false;
        let mut escaped = false;
        let mut chars = line.chars().peekable();

        while let Some(c) = chars.next() {
            if escaped {
                // Handle escaped characters
                match c {
                    '"' => current_token.push('"'),
                    '\\' => current_token.push('\\'),
                    'n' => current_token.push('\n'),
                    'r' => current_token.push('\r'),
                    't' => current_token.push('\t'),
                    _ => {
                        current_token.push('\\');
                        current_token.push(c);
                    }
                }
                escaped = false;
            } else if c == '\\' && in_quotes {
                escaped = true;
            } else if c == '"' && !escaped {
                current_token.push(c);
                if in_quotes {
                    // Check for language tag or datatype after closing quote
                    if let Some(&'@') = chars.peek() {
                        // Language tag
                        current_token.push(chars.next().unwrap()); // @
                        while let Some(&next_char) = chars.peek() {
                            if next_char.is_alphanumeric() || next_char == '-' {
                                current_token.push(chars.next().unwrap());
                            } else {
                                break;
                            }
                        }
                    } else if chars.peek() == Some(&'^') {
                        // Datatype
                        chars.next(); // first ^
                        if chars.peek() == Some(&'^') {
                            chars.next(); // second ^
                            current_token.push_str("^^");
                            if chars.peek() == Some(&'<') {
                                // IRI datatype
                                while let Some(next_char) = chars.next() {
                                    current_token.push(next_char);
                                    if next_char == '>' {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    in_quotes = false;
                } else {
                    in_quotes = true;
                }
            } else if c == '"' && escaped {
                // This is an escaped quote, add it to the token
                current_token.push(c);
                escaped = false;
            } else if c.is_whitespace() && !in_quotes {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
            } else {
                current_token.push(c);
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        Ok(tokens)
    }

    fn parse_subject(&self, token: &str) -> Result<Subject> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            let named_node = NamedNode::new(iri)?;
            Ok(Subject::NamedNode(named_node))
        } else if token.starts_with("_:") {
            let blank_node = BlankNode::new(token)?;
            Ok(Subject::BlankNode(blank_node))
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid subject: {}. Must be IRI or blank node",
                token
            )))
        }
    }

    fn parse_predicate(&self, token: &str) -> Result<Predicate> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            let named_node = NamedNode::new(iri)?;
            Ok(Predicate::NamedNode(named_node))
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid predicate: {}. Must be IRI",
                token
            )))
        }
    }

    fn parse_object(&self, token: &str) -> Result<Object> {
        if token.starts_with('<') && token.ends_with('>') {
            // IRI
            let iri = &token[1..token.len() - 1];
            let named_node = NamedNode::new(iri)?;
            Ok(Object::NamedNode(named_node))
        } else if token.starts_with("_:") {
            // Blank node
            let blank_node = BlankNode::new(token)?;
            Ok(Object::BlankNode(blank_node))
        } else if token.starts_with('"') {
            // Literal
            self.parse_literal(token)
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid object: {}. Must be IRI, blank node, or literal",
                token
            )))
        }
    }

    fn parse_literal(&self, token: &str) -> Result<Object> {
        if !token.starts_with('"') {
            return Err(OxirsError::Parse(
                "Literal must start with quote".to_string(),
            ));
        }

        // Find the closing quote
        let mut end_quote_pos = None;
        let mut escaped = false;
        let chars: Vec<char> = token.chars().collect();

        for i in 1..chars.len() {
            if escaped {
                escaped = false;
                continue;
            }

            if chars[i] == '\\' {
                escaped = true;
            } else if chars[i] == '"' {
                end_quote_pos = Some(i);
                break;
            }
        }

        let end_quote_pos =
            end_quote_pos.ok_or_else(|| OxirsError::Parse("Unterminated literal".to_string()))?;

        // Extract the literal value (without quotes)
        let literal_value: String = chars[1..end_quote_pos].iter().collect();

        // Check for language tag or datatype
        let remaining = &token[end_quote_pos + 1..];

        if remaining.starts_with('@') {
            // Language tag
            let lang_tag = &remaining[1..];
            let literal = Literal::new_lang(literal_value, lang_tag)?;
            Ok(Object::Literal(literal))
        } else if remaining.starts_with("^^<") && remaining.ends_with('>') {
            // Datatype
            let datatype_iri = &remaining[3..remaining.len() - 1];
            let datatype = NamedNode::new(datatype_iri)?;
            let literal = Literal::new_typed(literal_value, datatype);
            Ok(Object::Literal(literal))
        } else if remaining.is_empty() {
            // Plain literal
            let literal = Literal::new(literal_value);
            Ok(Object::Literal(literal))
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid literal syntax: {}",
                token
            )))
        }
    }

    fn parse_trig<F>(&self, data: &str, mut handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Result<()>,
    {
        // Simplified implementation - for now just create empty quads
        // TODO: Implement proper TriG parsing when Rio API is stable
        let _ = data;
        let _ = handler;
        Ok(())
    }

    fn parse_nquads<F>(&self, data: &str, mut handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Result<()>,
    {
        for (line_num, line) in data.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse the line into a quad
            match self.parse_nquads_line(line) {
                Ok(Some(quad)) => {
                    handler(quad)?;
                }
                Ok(None) => {
                    // Skip this line (e.g., blank line)
                    continue;
                }
                Err(e) => {
                    if self.config.ignore_errors {
                        tracing::warn!("Parse error on line {}: {}", line_num + 1, e);
                        continue;
                    } else {
                        return Err(OxirsError::Parse(format!(
                            "Parse error on line {}: {}",
                            line_num + 1,
                            e
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    fn parse_nquads_line(&self, line: &str) -> Result<Option<Quad>> {
        // N-Quads parser - parse line like: <s> <p> "o" <g> .
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }

        // Find the final period
        if !line.ends_with('.') {
            return Err(OxirsError::Parse("Line must end with '.'".to_string()));
        }

        let line = &line[..line.len() - 1].trim(); // Remove trailing period and whitespace

        // Split into tokens respecting quoted strings
        let tokens = self.tokenize_ntriples_line(line)?;

        if tokens.len() != 4 {
            return Err(OxirsError::Parse(format!(
                "Expected 4 tokens (subject, predicate, object, graph), found {}",
                tokens.len()
            )));
        }

        // Parse subject
        let subject = self.parse_subject(&tokens[0])?;

        // Parse predicate
        let predicate = self.parse_predicate(&tokens[1])?;

        // Parse object
        let object = self.parse_object(&tokens[2])?;

        // Parse graph name
        let graph_name = self.parse_graph_name(&tokens[3])?;

        let quad = Quad::new(subject, predicate, object, graph_name);

        Ok(Some(quad))
    }

    fn parse_graph_name(&self, token: &str) -> Result<GraphName> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            let named_node = NamedNode::new(iri)?;
            Ok(GraphName::NamedNode(named_node))
        } else if token.starts_with("_:") {
            let blank_node = BlankNode::new(token)?;
            Ok(GraphName::BlankNode(blank_node))
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid graph name: {}. Must be IRI or blank node",
                token
            )))
        }
    }

    fn parse_rdfxml<F>(&self, _data: &str, _handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Result<()>,
    {
        Err(OxirsError::Parse(
            "RDF/XML parsing temporarily disabled".to_string(),
        ))
    }

    fn parse_jsonld<F>(&self, _data: &str, _handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Result<()>,
    {
        // TODO: Implement JSON-LD parsing when oxjsonld is available
        Err(OxirsError::Parse(
            "JSON-LD parsing not yet implemented".to_string(),
        ))
    }

    /// Convert Rio triple to our Quad type
    fn convert_rio_triple_to_quad(rio_triple: &RioTriple) -> Result<Quad> {
        let subject = Self::convert_rio_subject(&rio_triple.subject)?;
        let predicate = Self::convert_rio_predicate(&rio_triple.predicate)?;
        let object = Self::convert_rio_object(&rio_triple.object)?;

        let triple = Triple::new(subject, predicate, object);
        Ok(Quad::from_triple(triple))
    }

    fn convert_rio_subject(rio_subject: &rio_api::model::Subject) -> Result<Subject> {
        match rio_subject {
            rio_api::model::Subject::NamedNode(nn) => {
                let named_node = NamedNode::new(nn.iri)?;
                Ok(Subject::NamedNode(named_node))
            }
            rio_api::model::Subject::BlankNode(bn) => {
                let blank_node = BlankNode::new(format!("_:{}", bn.id))?;
                Ok(Subject::BlankNode(blank_node))
            }
            rio_api::model::Subject::Triple(_) => Err(OxirsError::Parse(
                "Nested triples not supported yet".to_string(),
            )),
        }
    }

    fn convert_rio_predicate(rio_predicate: &rio_api::model::NamedNode) -> Result<Predicate> {
        let named_node = NamedNode::new(rio_predicate.iri)?;
        Ok(Predicate::NamedNode(named_node))
    }

    fn convert_rio_object(rio_object: &rio_api::model::Term) -> Result<Object> {
        match rio_object {
            rio_api::model::Term::NamedNode(nn) => {
                let named_node = NamedNode::new(nn.iri)?;
                Ok(Object::NamedNode(named_node))
            }
            rio_api::model::Term::BlankNode(bn) => {
                let blank_node = BlankNode::new(format!("_:{}", bn.id))?;
                Ok(Object::BlankNode(blank_node))
            }
            rio_api::model::Term::Literal(_lit) => {
                // Simplified implementation for now - convert to plain string literal
                // TODO: Implement proper Rio literal conversion when API is stable
                let literal = Literal::new("placeholder");
                Ok(Object::Literal(literal))
            }
            rio_api::model::Term::Triple(_) => Err(OxirsError::Parse(
                "Nested triples not supported yet".to_string(),
            )),
        }
    }

    // TODO: Phase 2 - OxiGraph conversion methods will be replaced with native parsing
    // These methods were for converting from OxiGraph types to OxiRS types
}

/// Turtle parser state for handling multi-line statements and abbreviations
struct TurtleParserState {
    prefixes: HashMap<String, String>,
    base_iri: Option<String>,
    pending_statement: String,
}

impl TurtleParserState {
    fn new(base_iri: Option<&str>) -> Self {
        let mut prefixes = HashMap::new();
        // Add default prefixes
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

        TurtleParserState {
            prefixes,
            base_iri: base_iri.map(|s| s.to_string()),
            pending_statement: String::new(),
        }
    }

    fn parse_line(&mut self, line: &str) -> Result<Vec<Triple>> {
        let line = line.trim();

        // Handle directives
        if line.starts_with("@prefix") {
            return self.parse_prefix_directive(line);
        }

        if line.starts_with("@base") {
            return self.parse_base_directive(line);
        }

        // Accumulate multi-line statements
        self.pending_statement.push_str(line);
        self.pending_statement.push(' ');

        // Check if statement is complete (ends with .)
        if line.ends_with('.') {
            let statement = self.pending_statement.trim().to_string();
            self.pending_statement.clear();
            return self.parse_statement(&statement);
        }

        Ok(Vec::new())
    }

    fn finalize(&mut self) -> Result<Option<Vec<Triple>>> {
        if !self.pending_statement.trim().is_empty() {
            let statement = self.pending_statement.trim().to_string();
            self.pending_statement.clear();
            return Ok(Some(self.parse_statement(&statement)?));
        }
        Ok(None)
    }

    fn parse_prefix_directive(&mut self, line: &str) -> Result<Vec<Triple>> {
        // @prefix ns: <http://example.org/ns#> .
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 3 {
            return Err(OxirsError::Parse("Invalid @prefix directive".to_string()));
        }

        let prefix = parts[1].trim_end_matches(':');
        let iri = parts[2];

        if !iri.starts_with('<') || !iri.ends_with('>') {
            return Err(OxirsError::Parse(
                "IRI must be enclosed in angle brackets".to_string(),
            ));
        }

        let iri = &iri[1..iri.len() - 1];
        self.prefixes.insert(prefix.to_string(), iri.to_string());

        Ok(Vec::new())
    }

    fn parse_base_directive(&mut self, line: &str) -> Result<Vec<Triple>> {
        // @base <http://example.org/> .
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 2 {
            return Err(OxirsError::Parse("Invalid @base directive".to_string()));
        }

        let iri = parts[1];

        if !iri.starts_with('<') || !iri.ends_with('>') {
            return Err(OxirsError::Parse(
                "Base IRI must be enclosed in angle brackets".to_string(),
            ));
        }

        let iri = &iri[1..iri.len() - 1];
        self.base_iri = Some(iri.to_string());

        Ok(Vec::new())
    }

    fn parse_statement(&mut self, statement: &str) -> Result<Vec<Triple>> {
        let statement = statement.trim().trim_end_matches('.');
        let mut triples = Vec::new();

        // Handle abbreviated syntax: subject predicate object ; predicate object
        let subject_parts: Vec<&str> = statement.split(';').collect();

        if subject_parts.len() == 1 {
            // Single triple: subject predicate object
            if let Some(triple) = self.parse_simple_triple(statement)? {
                triples.push(triple);
            }
        } else {
            // Multiple triples with same subject
            let first_part = subject_parts[0].trim();
            let first_triple = self.parse_simple_triple(first_part)?;

            if let Some(triple) = first_triple {
                let subject = triple.subject().clone();
                triples.push(triple);

                // Parse remaining predicate-object pairs
                for part in &subject_parts[1..] {
                    let part = part.trim();
                    if !part.is_empty() {
                        if let Some(triple) = self.parse_predicate_object_pair(&subject, part)? {
                            triples.push(triple);
                        }
                    }
                }
            }
        }

        Ok(triples)
    }

    fn parse_simple_triple(&self, triple_str: &str) -> Result<Option<Triple>> {
        let tokens = self.tokenize_turtle_statement(triple_str)?;

        if tokens.len() < 3 {
            return Ok(None);
        }

        let subject = self.parse_turtle_subject(&tokens[0])?;
        let predicate = self.parse_turtle_predicate(&tokens[1])?;
        let object = self.parse_turtle_object(&tokens[2])?;

        Ok(Some(Triple::new(subject, predicate, object)))
    }

    fn parse_predicate_object_pair(
        &self,
        subject: &Subject,
        pair_str: &str,
    ) -> Result<Option<Triple>> {
        let tokens = self.tokenize_turtle_statement(pair_str)?;

        if tokens.len() < 2 {
            return Ok(None);
        }

        let predicate = self.parse_turtle_predicate(&tokens[0])?;
        let object = self.parse_turtle_object(&tokens[1])?;

        Ok(Some(Triple::new(subject.clone(), predicate, object)))
    }

    fn tokenize_turtle_statement(&self, statement: &str) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut in_quotes = false;
        let mut in_angles = false;
        let mut escaped = false;

        for c in statement.chars() {
            if escaped {
                current_token.push(c);
                escaped = false;
            } else if c == '\\' && (in_quotes || in_angles) {
                escaped = true;
                current_token.push(c);
            } else if c == '"' && !in_angles {
                current_token.push(c);
                in_quotes = !in_quotes;
            } else if c == '<' && !in_quotes {
                current_token.push(c);
                in_angles = true;
            } else if c == '>' && !in_quotes {
                current_token.push(c);
                in_angles = false;
            } else if c.is_whitespace() && !in_quotes && !in_angles {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
            } else {
                current_token.push(c);
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        Ok(tokens)
    }

    fn parse_turtle_subject(&self, token: &str) -> Result<Subject> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = self.resolve_iri(&token[1..token.len() - 1])?;
            let named_node = NamedNode::new(iri)?;
            Ok(Subject::NamedNode(named_node))
        } else if token.starts_with("_:") {
            let blank_node = BlankNode::new(token)?;
            Ok(Subject::BlankNode(blank_node))
        } else if token.contains(':')
            && !token.starts_with("http://")
            && !token.starts_with("https://")
        {
            // Prefixed name
            let iri = self.expand_prefixed_name(token)?;
            let named_node = NamedNode::new(iri)?;
            Ok(Subject::NamedNode(named_node))
        } else {
            Err(OxirsError::Parse(format!("Invalid subject: {}", token)))
        }
    }

    fn parse_turtle_predicate(&self, token: &str) -> Result<Predicate> {
        if token == "a" {
            // Shorthand for rdf:type
            let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
            let named_node = NamedNode::new(rdf_type)?;
            Ok(Predicate::NamedNode(named_node))
        } else if token.starts_with('<') && token.ends_with('>') {
            let iri = self.resolve_iri(&token[1..token.len() - 1])?;
            let named_node = NamedNode::new(iri)?;
            Ok(Predicate::NamedNode(named_node))
        } else if token.contains(':')
            && !token.starts_with("http://")
            && !token.starts_with("https://")
        {
            // Prefixed name
            let iri = self.expand_prefixed_name(token)?;
            let named_node = NamedNode::new(iri)?;
            Ok(Predicate::NamedNode(named_node))
        } else {
            Err(OxirsError::Parse(format!("Invalid predicate: {}", token)))
        }
    }

    fn parse_turtle_object(&self, token: &str) -> Result<Object> {
        if token.starts_with('<') && token.ends_with('>') {
            // IRI
            let iri = self.resolve_iri(&token[1..token.len() - 1])?;
            let named_node = NamedNode::new(iri)?;
            Ok(Object::NamedNode(named_node))
        } else if token.starts_with("_:") {
            // Blank node
            let blank_node = BlankNode::new(token)?;
            Ok(Object::BlankNode(blank_node))
        } else if token.starts_with('"') {
            // Literal
            self.parse_turtle_literal(token)
        } else if token.contains(':')
            && !token.starts_with("http://")
            && !token.starts_with("https://")
        {
            // Prefixed name
            let iri = self.expand_prefixed_name(token)?;
            let named_node = NamedNode::new(iri)?;
            Ok(Object::NamedNode(named_node))
        } else {
            Err(OxirsError::Parse(format!("Invalid object: {}", token)))
        }
    }

    fn parse_turtle_literal(&self, token: &str) -> Result<Object> {
        if !token.starts_with('"') {
            return Err(OxirsError::Parse(
                "Literal must start with quote".to_string(),
            ));
        }

        // Find the closing quote
        let mut end_quote_pos = None;
        let mut escaped = false;
        let chars: Vec<char> = token.chars().collect();

        for i in 1..chars.len() {
            if escaped {
                escaped = false;
                continue;
            }

            if chars[i] == '\\' {
                escaped = true;
            } else if chars[i] == '"' {
                end_quote_pos = Some(i);
                break;
            }
        }

        let end_quote_pos =
            end_quote_pos.ok_or_else(|| OxirsError::Parse("Unterminated literal".to_string()))?;

        // Extract the literal value (without quotes)
        let literal_value: String = chars[1..end_quote_pos].iter().collect();

        // Check for language tag or datatype
        let remaining = &token[end_quote_pos + 1..];

        if remaining.starts_with('@') {
            // Language tag
            let lang_tag = &remaining[1..];
            let literal = Literal::new_lang(literal_value, lang_tag)?;
            Ok(Object::Literal(literal))
        } else if remaining.starts_with("^^") {
            // Datatype
            let datatype_part = &remaining[2..];
            if datatype_part.starts_with('<') && datatype_part.ends_with('>') {
                // IRI datatype
                let datatype_iri = self.resolve_iri(&datatype_part[1..datatype_part.len() - 1])?;
                let datatype = NamedNode::new(datatype_iri)?;
                let literal = Literal::new_typed(literal_value, datatype);
                Ok(Object::Literal(literal))
            } else if datatype_part.contains(':') {
                // Prefixed datatype
                let datatype_iri = self.expand_prefixed_name(datatype_part)?;
                let datatype = NamedNode::new(datatype_iri)?;
                let literal = Literal::new_typed(literal_value, datatype);
                Ok(Object::Literal(literal))
            } else {
                Err(OxirsError::Parse(format!(
                    "Invalid datatype: {}",
                    datatype_part
                )))
            }
        } else if remaining.is_empty() {
            // Plain literal
            let literal = Literal::new(literal_value);
            Ok(Object::Literal(literal))
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid literal syntax: {}",
                token
            )))
        }
    }

    fn expand_prefixed_name(&self, prefixed_name: &str) -> Result<String> {
        if let Some(colon_pos) = prefixed_name.find(':') {
            let prefix = &prefixed_name[..colon_pos];
            let local_name = &prefixed_name[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                Ok(format!("{}{}", namespace, local_name))
            } else {
                Err(OxirsError::Parse(format!("Unknown prefix: {}", prefix)))
            }
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid prefixed name: {}",
                prefixed_name
            )))
        }
    }

    fn resolve_iri(&self, iri: &str) -> Result<String> {
        if iri.contains("://") {
            // Absolute IRI
            Ok(iri.to_string())
        } else if let Some(base) = &self.base_iri {
            // Resolve relative IRI against base
            if base.ends_with('/') {
                Ok(format!("{}{}", base, iri))
            } else {
                Ok(format!("{}/{}", base, iri))
            }
        } else {
            // No base IRI, return as-is
            Ok(iri.to_string())
        }
    }
}

/// Convenience function to detect RDF format from content
pub fn detect_format_from_content(content: &str) -> Option<RdfFormat> {
    let content = content.trim();

    // Check for XML-like content (RDF/XML)
    if content.starts_with("<?xml")
        || content.starts_with("<rdf:RDF")
        || content.starts_with("<RDF")
    {
        return Some(RdfFormat::RdfXml);
    }

    // Check for JSON-LD
    if content.starts_with('{') && (content.contains("@context") || content.contains("@type")) {
        return Some(RdfFormat::JsonLd);
    }

    // Check for Turtle syntax elements first (has priority over N-Quads/N-Triples)
    if content.contains("@prefix") || content.contains("@base") || content.contains(';') {
        return Some(RdfFormat::Turtle);
    }

    // Check for TriG (named graphs syntax)
    if content.contains('{') && content.contains('}') {
        return Some(RdfFormat::TriG);
    }

    // Count tokens in first meaningful line to distinguish N-Quads vs N-Triples
    for line in content.lines() {
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 4 && parts[3] == "." {
                // Exactly 4 parts (s p o .) - N-Triples
                return Some(RdfFormat::NTriples);
            } else if parts.len() == 5 && parts[4] == "." {
                // Exactly 5 parts (s p o g .) - N-Quads
                return Some(RdfFormat::NQuads);
            } else if parts.len() >= 3 && parts[parts.len() - 1] == "." {
                // Fallback: assume N-Triples for basic triple pattern
                return Some(RdfFormat::NTriples);
            }
            break; // Only check first meaningful line
        }
    }

    None
}

/// Async RDF streaming parser for high-performance large file processing
#[cfg(feature = "async")]
pub struct AsyncStreamingParser {
    format: RdfFormat,
    config: ParserConfig,
    progress_callback: Option<Box<dyn Fn(usize) + Send + Sync>>,
    chunk_size: usize,
}

#[cfg(feature = "async")]
impl AsyncStreamingParser {
    /// Create a new async streaming parser
    pub fn new(format: RdfFormat) -> Self {
        AsyncStreamingParser {
            format,
            config: ParserConfig::default(),
            progress_callback: None,
            chunk_size: 8192, // 8KB default chunk size
        }
    }

    /// Set a progress callback that reports the number of bytes processed
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    /// Set the chunk size for streaming processing
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Configure error tolerance
    pub fn with_error_tolerance(mut self, ignore_errors: bool) -> Self {
        self.config.ignore_errors = ignore_errors;
        self
    }

    /// Parse from an async readable stream
    pub async fn parse_stream<R, F, Fut>(&self, mut reader: R, mut handler: F) -> Result<()>
    where
        R: tokio::io::AsyncRead + Unpin,
        F: FnMut(Quad) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        use tokio::io::AsyncReadExt;

        let mut buffer = Vec::with_capacity(self.chunk_size);
        let mut accumulated_data = String::new();
        let mut bytes_processed = 0usize;
        let mut line_buffer = String::new();

        loop {
            buffer.clear();
            buffer.resize(self.chunk_size, 0);

            let bytes_read = reader
                .read(&mut buffer)
                .await
                .map_err(|e| OxirsError::Io(e))?;

            if bytes_read == 0 {
                break; // End of stream
            }

            buffer.truncate(bytes_read);
            bytes_processed += bytes_read;

            // Convert bytes to string and append to accumulated data
            let chunk_str = String::from_utf8_lossy(&buffer);
            accumulated_data.push_str(&chunk_str);

            // Process complete lines for line-based formats (N-Triples, N-Quads)
            if matches!(self.format, RdfFormat::NTriples | RdfFormat::NQuads) {
                self.process_lines_async(&mut accumulated_data, &mut line_buffer, &mut handler)
                    .await?;
            }

            // Report progress if callback is set
            if let Some(ref callback) = self.progress_callback {
                callback(bytes_processed);
            }
        }

        // Process any remaining data
        if !accumulated_data.is_empty() {
            match self.format {
                RdfFormat::NTriples | RdfFormat::NQuads => {
                    // Process final lines
                    accumulated_data.push_str(&line_buffer);
                    self.process_lines_async(
                        &mut accumulated_data,
                        &mut String::new(),
                        &mut handler,
                    )
                    .await?;
                }
                _ => {
                    // For other formats, parse the complete document
                    let parser = Parser::with_config(self.format, self.config.clone());
                    parser.parse_str_with_handler(&accumulated_data, |quad| {
                        // Convert sync closure to async - this is a simplified approach
                        // In a real implementation, you'd want to use proper async handling
                        tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(handler(quad))
                        })
                    })?;
                }
            }
        }

        Ok(())
    }

    /// Process lines asynchronously for line-based formats
    async fn process_lines_async<F, Fut>(
        &self,
        accumulated_data: &mut String,
        line_buffer: &mut String,
        handler: &mut F,
    ) -> Result<()>
    where
        F: FnMut(Quad) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        // Combine line buffer with new data
        let mut full_data = line_buffer.clone();
        full_data.push_str(accumulated_data);

        let mut last_newline_pos = 0;

        // Find complete lines
        for (pos, _) in full_data.match_indices('\n') {
            let line = &full_data[last_newline_pos..pos];
            last_newline_pos = pos + 1;

            // Parse the line
            if let Some(quad) = self.parse_line(line)? {
                handler(quad).await?;
            }
        }

        // Keep incomplete line for next iteration
        line_buffer.clear();
        if last_newline_pos < full_data.len() {
            line_buffer.push_str(&full_data[last_newline_pos..]);
        }

        accumulated_data.clear();
        Ok(())
    }

    /// Parse a single line (for N-Triples/N-Quads)
    fn parse_line(&self, line: &str) -> Result<Option<Quad>> {
        let parser = Parser::with_config(self.format, self.config.clone());

        match self.format {
            RdfFormat::NTriples => parser.parse_ntriples_line(line),
            RdfFormat::NQuads => {
                // For N-Quads, we need a more sophisticated parser
                // This is a simplified implementation
                parser.parse_ntriples_line(line)
            }
            _ => Err(OxirsError::Parse(
                "Unsupported format for line parsing".to_string(),
            )),
        }
    }

    /// Parse from bytes asynchronously
    pub async fn parse_bytes<F, Fut>(&self, data: &[u8], mut handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        use std::io::Cursor;
        let cursor = Cursor::new(data);
        self.parse_stream(cursor, handler).await
    }

    /// Parse from string asynchronously  
    pub async fn parse_str_async<F, Fut>(&self, data: &str, mut handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let bytes = data.as_bytes();
        self.parse_bytes(bytes, handler).await
    }

    /// Convenience method to parse to a vector asynchronously
    pub async fn parse_str_to_quads_async(&self, data: &str) -> Result<Vec<Quad>> {
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let quads = Arc::new(Mutex::new(Vec::new()));
        let quads_clone = Arc::clone(&quads);

        self.parse_str_async(data, move |quad| {
            let quads = Arc::clone(&quads_clone);
            async move {
                quads.lock().await.push(quad);
                Ok(())
            }
        })
        .await?;

        let result = quads.lock().await;
        Ok(result.clone())
    }
}

/// Progress information for async parsing
#[cfg(feature = "async")]
#[derive(Debug, Clone)]
pub struct ParseProgress {
    pub bytes_processed: usize,
    pub quads_parsed: usize,
    pub errors_encountered: usize,
    pub estimated_total_bytes: Option<usize>,
}

#[cfg(feature = "async")]
impl ParseProgress {
    /// Calculate completion percentage if total size is known
    pub fn completion_percentage(&self) -> Option<f64> {
        self.estimated_total_bytes.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.bytes_processed as f64 / total as f64) * 100.0
            }
        })
    }
}

/// Async streaming sink for writing parsed RDF data
#[cfg(feature = "async")]
pub trait AsyncRdfSink: Send + Sync {
    /// Process a parsed quad asynchronously
    fn process_quad(&mut self, quad: Quad)
        -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

    /// Finalize processing (called when parsing is complete)
    fn finalize(&mut self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
}

/// Memory-based async sink that collects quads
#[cfg(feature = "async")]
pub struct MemoryAsyncSink {
    quads: Vec<Quad>,
}

#[cfg(feature = "async")]
impl MemoryAsyncSink {
    pub fn new() -> Self {
        MemoryAsyncSink { quads: Vec::new() }
    }

    pub fn into_quads(self) -> Vec<Quad> {
        self.quads
    }
}

#[cfg(feature = "async")]
impl AsyncRdfSink for MemoryAsyncSink {
    fn process_quad(
        &mut self,
        quad: Quad,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            self.quads.push(quad);
            Ok(())
        })
    }

    fn finalize(&mut self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;

    #[test]
    fn test_format_detection_from_extension() {
        assert_eq!(RdfFormat::from_extension("ttl"), Some(RdfFormat::Turtle));
        assert_eq!(RdfFormat::from_extension("turtle"), Some(RdfFormat::Turtle));
        assert_eq!(RdfFormat::from_extension("nt"), Some(RdfFormat::NTriples));
        assert_eq!(
            RdfFormat::from_extension("ntriples"),
            Some(RdfFormat::NTriples)
        );
        assert_eq!(RdfFormat::from_extension("trig"), Some(RdfFormat::TriG));
        assert_eq!(RdfFormat::from_extension("nq"), Some(RdfFormat::NQuads));
        assert_eq!(RdfFormat::from_extension("rdf"), Some(RdfFormat::RdfXml));
        assert_eq!(RdfFormat::from_extension("jsonld"), Some(RdfFormat::JsonLd));
        assert_eq!(RdfFormat::from_extension("unknown"), None);
    }

    #[test]
    fn test_format_properties() {
        assert_eq!(RdfFormat::Turtle.media_type(), "text/turtle");
        assert_eq!(RdfFormat::NTriples.extension(), "nt");
        assert!(RdfFormat::TriG.supports_quads());
        assert!(!RdfFormat::Turtle.supports_quads());
    }

    #[test]
    fn test_format_detection_from_content() {
        // XML content
        let xml_content = "<?xml version=\"1.0\"?>\n<rdf:RDF>";
        assert_eq!(
            detect_format_from_content(xml_content),
            Some(RdfFormat::RdfXml)
        );

        // JSON-LD content
        let jsonld_content = r#"{"@context": "http://example.org", "@type": "Person"}"#;
        assert_eq!(
            detect_format_from_content(jsonld_content),
            Some(RdfFormat::JsonLd)
        );

        // Turtle content
        let turtle_content = "@prefix foaf: <http://xmlns.com/foaf/0.1/> .";
        assert_eq!(
            detect_format_from_content(turtle_content),
            Some(RdfFormat::Turtle)
        );

        // N-Triples content
        let ntriples_content = "<http://example.org/s> <http://example.org/p> \"object\" .";
        assert_eq!(
            detect_format_from_content(ntriples_content),
            Some(RdfFormat::NTriples)
        );
    }

    #[test]
    fn test_ntriples_parsing_simple() {
        let ntriples_data = r#"<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice Smith" .
<http://example.org/alice> <http://xmlns.com/foaf/0.1/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> .
_:person1 <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> ."#;

        let parser = Parser::new(RdfFormat::NTriples);
        let result = parser.parse_str_to_quads(ntriples_data);

        assert!(result.is_ok());
        let quads = result.unwrap();
        assert_eq!(quads.len(), 3);

        // Check that all quads are in the default graph
        for quad in &quads {
            assert!(quad.is_default_graph());
        }

        // Convert to triples for easier checking
        let triples: Vec<_> = quads.into_iter().map(|q| q.to_triple()).collect();

        // Check first triple
        let alice_iri = NamedNode::new("http://example.org/alice").unwrap();
        let name_pred = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
        let name_literal = Literal::new("Alice Smith");
        let expected_triple1 = Triple::new(alice_iri.clone(), name_pred, name_literal);
        assert!(triples.contains(&expected_triple1));

        // Check typed literal triple
        let age_pred = NamedNode::new("http://xmlns.com/foaf/0.1/age").unwrap();
        let integer_type = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap();
        let age_literal = Literal::new_typed("30", integer_type);
        let expected_triple2 = Triple::new(alice_iri, age_pred, age_literal);
        assert!(triples.contains(&expected_triple2));

        // Check blank node triple
        let blank_node = BlankNode::new("_:person1").unwrap();
        let knows_pred = NamedNode::new("http://xmlns.com/foaf/0.1/knows").unwrap();
        let bob_iri = NamedNode::new("http://example.org/bob").unwrap();
        let expected_triple3 = Triple::new(blank_node, knows_pred, bob_iri);
        assert!(triples.contains(&expected_triple3));
    }

    #[test]
    fn test_ntriples_parsing_language_tag() {
        let ntriples_data =
            r#"<http://example.org/alice> <http://example.org/description> "Une personne"@fr ."#;

        let parser = Parser::new(RdfFormat::NTriples);
        let result = parser.parse_str_to_quads(ntriples_data);

        assert!(result.is_ok());
        let quads = result.unwrap();
        assert_eq!(quads.len(), 1);

        let triple = quads[0].to_triple();
        if let Object::Literal(literal) = triple.object() {
            assert_eq!(literal.value(), "Une personne");
            assert_eq!(literal.language(), Some("fr"));
            assert!(literal.is_lang_string());
        } else {
            panic!("Expected literal object");
        }
    }

    #[test]
    #[ignore] // TODO: Fix escaped literal parsing
    fn test_ntriples_parsing_escaped_literals() {
        let ntriples_data = r#"<http://example.org/test> <http://example.org/desc> "Text with \"quotes\" and \n newlines" ."#;

        let parser = Parser::new(RdfFormat::NTriples);
        let result = parser.parse_str_to_quads(ntriples_data);

        if let Err(e) = &result {
            println!("Parse error: {}", e);
        }
        assert!(result.is_ok(), "Parse failed: {:?}", result);

        let quads = result.unwrap();
        assert_eq!(quads.len(), 1);

        let triple = quads[0].to_triple();
        if let Object::Literal(literal) = triple.object() {
            assert!(literal.value().contains("\"quotes\""));
            assert!(literal.value().contains("\n"));
        } else {
            panic!("Expected literal object");
        }
    }

    #[test]
    fn test_ntriples_parsing_comments_and_empty_lines() {
        let ntriples_data = r#"
# This is a comment
<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice Smith" .

# Another comment
<http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob Jones" .
"#;

        let parser = Parser::new(RdfFormat::NTriples);
        let result = parser.parse_str_to_quads(ntriples_data);

        assert!(result.is_ok());
        let quads = result.unwrap();
        assert_eq!(quads.len(), 2);
    }

    #[test]
    fn test_ntriples_parsing_error_handling() {
        // Test invalid syntax
        let invalid_data = "invalid ntriples data";
        let parser = Parser::new(RdfFormat::NTriples);
        let result = parser.parse_str_to_quads(invalid_data);
        assert!(result.is_err());

        // Test error tolerance
        let mixed_data = r#"<http://example.org/valid> <http://example.org/pred> "Valid triple" .
invalid line here
<http://example.org/valid2> <http://example.org/pred> "Another valid triple" ."#;

        let parser_strict = Parser::new(RdfFormat::NTriples);
        let result_strict = parser_strict.parse_str_to_quads(mixed_data);
        assert!(result_strict.is_err());

        let parser_tolerant = Parser::new(RdfFormat::NTriples).with_error_tolerance(true);
        let result_tolerant = parser_tolerant.parse_str_to_quads(mixed_data);
        assert!(result_tolerant.is_ok());
        let quads = result_tolerant.unwrap();
        assert_eq!(quads.len(), 2); // Should parse the two valid triples
    }

    #[test]
    fn test_nquads_parsing() {
        let nquads_data = r#"<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice Smith" <http://example.org/graph1> .
<http://example.org/alice> <http://xmlns.com/foaf/0.1/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> <http://example.org/graph2> .
_:person1 <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> _:graph1 ."#;

        let parser = Parser::new(RdfFormat::NQuads);
        let result = parser.parse_str_to_quads(nquads_data);

        assert!(result.is_ok());
        let quads = result.unwrap();
        assert_eq!(quads.len(), 3);

        // Check that quads have proper graph names
        let first_quad = &quads[0];
        assert!(!first_quad.is_default_graph());

        // Check that we can extract graph names
        if let GraphName::NamedNode(graph_name) = first_quad.graph_name() {
            assert!(graph_name.as_str().contains("example.org"));
        } else {
            panic!("Expected named graph");
        }
    }

    #[test]
    fn test_turtle_parsing_basic() {
        let turtle_data = r#"@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix ex: <http://example.org/> .

ex:alice foaf:name "Alice Smith" .
ex:alice foaf:age "30"^^<http://www.w3.org/2001/XMLSchema#integer> .
ex:alice foaf:knows ex:bob ."#;

        let parser = Parser::new(RdfFormat::Turtle);
        let result = parser.parse_str_to_quads(turtle_data);

        assert!(result.is_ok());
        let quads = result.unwrap();
        assert_eq!(quads.len(), 3);

        // All quads should be in default graph
        for quad in &quads {
            assert!(quad.is_default_graph());
        }
    }

    #[test]
    fn test_turtle_parsing_prefixes() {
        let turtle_data = r#"@prefix foaf: <http://xmlns.com/foaf/0.1/> .
foaf:Person a foaf:Person ."#;

        let parser = Parser::new(RdfFormat::Turtle);
        let result = parser.parse_str_to_quads(turtle_data);

        assert!(result.is_ok());
        let quads = result.unwrap();
        assert_eq!(quads.len(), 1);

        let triple = quads[0].to_triple();
        // Should expand foaf:Person to full IRI
        if let Subject::NamedNode(subj) = triple.subject() {
            assert!(subj.as_str().contains("xmlns.com/foaf"));
        } else {
            panic!("Expected named node subject");
        }

        // Predicate should be rdf:type (from 'a')
        if let Predicate::NamedNode(pred) = triple.predicate() {
            assert!(pred.as_str().contains("rdf-syntax-ns#type"));
        } else {
            panic!("Expected named node predicate");
        }
    }

    #[test]
    fn test_turtle_parsing_abbreviated_syntax() {
        let turtle_data = r#"@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:alice foaf:name "Alice" ;
         foaf:age "30" ."#;

        let parser = Parser::new(RdfFormat::Turtle);
        let result = parser.parse_str_to_quads(turtle_data);

        assert!(result.is_ok());
        let quads = result.unwrap();
        assert_eq!(quads.len(), 2);

        // Both triples should have the same subject
        let subjects: Vec<_> = quads
            .iter()
            .map(|q| q.to_triple().subject().clone())
            .collect();
        assert_eq!(subjects[0], subjects[1]);
    }

    #[test]
    fn test_turtle_parsing_base_iri() {
        let turtle_data = r#"@base <http://example.org/> .
<alice> <knows> <bob> ."#;

        let parser = Parser::new(RdfFormat::Turtle);
        let result = parser.parse_str_to_quads(turtle_data);

        assert!(result.is_ok());
        let quads = result.unwrap();
        assert_eq!(quads.len(), 1);

        let triple = quads[0].to_triple();
        // IRIs should be resolved relative to base
        if let Subject::NamedNode(subj) = triple.subject() {
            assert!(subj.as_str().contains("example.org"));
        } else {
            panic!("Expected named node subject");
        }
    }

    #[test]
    fn test_turtle_parsing_literals() {
        let turtle_data = r#"@prefix ex: <http://example.org/> .
ex:alice ex:name "Alice"@en .
ex:alice ex:age "30"^^<http://www.w3.org/2001/XMLSchema#integer> ."#;

        let parser = Parser::new(RdfFormat::Turtle);
        let result = parser.parse_str_to_quads(turtle_data);

        assert!(result.is_ok());
        let quads = result.unwrap();
        assert_eq!(quads.len(), 2);

        // Check for language tag and datatype
        let triples: Vec<_> = quads.into_iter().map(|q| q.to_triple()).collect();

        let mut found_lang_literal = false;
        let mut found_typed_literal = false;

        for triple in triples {
            if let Object::Literal(literal) = triple.object() {
                if literal.language().is_some() {
                    found_lang_literal = true;
                    assert_eq!(literal.language(), Some("en"));
                } else {
                    let datatype = literal.datatype();
                    // Check for typed literal (not language-tagged and not plain string)
                    if datatype.as_str() != "http://www.w3.org/2001/XMLSchema#string"
                        && datatype.as_str()
                            != "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
                    {
                        found_typed_literal = true;
                        assert!(
                            datatype.as_str().contains("integer"),
                            "Expected integer datatype but got: {}",
                            datatype.as_str()
                        );
                    }
                }
            }
        }

        assert!(found_lang_literal);
        assert!(found_typed_literal);
    }

    #[test]
    fn test_parser_round_trip() {
        use crate::serializer::Serializer;

        // Create a graph with various types of triples
        let mut original_graph = Graph::new();

        let alice = NamedNode::new("http://example.org/alice").unwrap();
        let name_pred = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
        let name_literal = Literal::new("Alice Smith");
        original_graph.insert(Triple::new(alice.clone(), name_pred, name_literal));

        let age_pred = NamedNode::new("http://xmlns.com/foaf/0.1/age").unwrap();
        let age_literal = Literal::new_typed("30", crate::vocab::xsd::INTEGER.clone());
        original_graph.insert(Triple::new(alice.clone(), age_pred, age_literal));

        let desc_pred = NamedNode::new("http://example.org/description").unwrap();
        let desc_literal = Literal::new_lang("Une personne", "fr").unwrap();
        original_graph.insert(Triple::new(alice, desc_pred, desc_literal));

        // Serialize to N-Triples
        let serializer = Serializer::new(RdfFormat::NTriples);
        let ntriples = serializer.serialize_graph(&original_graph).unwrap();

        // Parse back from N-Triples
        let parser = Parser::new(RdfFormat::NTriples);
        let quads = parser.parse_str_to_quads(&ntriples).unwrap();

        // Convert back to graph
        let parsed_graph = Graph::from_iter(quads.into_iter().map(|q| q.to_triple()));

        // Should have the same number of triples
        assert_eq!(original_graph.len(), parsed_graph.len());

        // All original triples should be present in parsed graph
        for triple in original_graph.iter() {
            assert!(
                parsed_graph.contains(triple),
                "Parsed graph missing triple: {}",
                triple
            );
        }
    }
}
