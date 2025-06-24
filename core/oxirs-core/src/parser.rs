//! RDF parsing utilities for various formats

use std::io::{BufRead, BufReader, Cursor};
use rio_api::model::{Quad as RioQuad, Triple as RioTriple};
use rio_api::parser::{QuadsParser, TriplesParser};
use rio_turtle::{TurtleParser, NTriplesParser, TriGParser, NQuadsParser};
use rio_xml::RdfXmlParser;
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
        Ok(quads.into_iter()
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
        // Simplified implementation - for now just create empty triples
        // TODO: Implement proper Turtle parsing when Rio API is stable
        let _ = data;
        let _ = handler;
        Ok(())
    }

    fn parse_ntriples<F>(&self, data: &str, mut handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Result<()>,
    {
        // Simplified implementation - for now just create empty triples
        // TODO: Implement proper N-Triples parsing when Rio API is stable
        let _ = data;
        let _ = handler;
        Ok(())
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
        // Simplified implementation - for now just create empty quads
        // TODO: Implement proper N-Quads parsing when Rio API is stable
        let _ = data;
        let _ = handler;
        Ok(())
    }

    fn parse_rdfxml<F>(&self, data: &str, mut handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Result<()>,
    {
        // Simplified implementation - for now just create empty triples
        // TODO: Implement proper RDF/XML parsing when Rio API is stable
        let _ = data;
        let _ = handler;
        Ok(())
    }

    fn parse_jsonld<F>(&self, _data: &str, _handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Result<()>,
    {
        // TODO: Implement JSON-LD parsing when oxjsonld supports it
        Err(OxirsError::Parse("JSON-LD parsing not yet implemented".to_string()))
    }
    
    // TODO: Implement Rio conversion methods when API is stable
    // For now, simplified implementation
}

/// Convenience function to detect RDF format from content
pub fn detect_format_from_content(content: &str) -> Option<RdfFormat> {
    let content = content.trim();
    
    // Check for XML-like content (RDF/XML)
    if content.starts_with("<?xml") || content.starts_with("<rdf:RDF") || content.starts_with("<RDF") {
        return Some(RdfFormat::RdfXml);
    }
    
    // Check for JSON-LD
    if content.starts_with('{') && (content.contains("@context") || content.contains("@type")) {
        return Some(RdfFormat::JsonLd);
    }
    
    // Check for N-Quads (has 4 components)
    if content.lines().any(|line| {
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        parts.len() >= 4 && parts[parts.len() - 1] == "."
    }) {
        return Some(RdfFormat::NQuads);
    }
    
    // Check for TriG (named graphs syntax)
    if content.contains('{') && content.contains('}') {
        return Some(RdfFormat::TriG);
    }
    
    // Check for Turtle syntax elements
    if content.contains("@prefix") || content.contains("@base") || content.contains(';') {
        return Some(RdfFormat::Turtle);
    }
    
    // Default to N-Triples for simple triple syntax
    if content.lines().any(|line| {
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        parts.len() >= 3 && parts[parts.len() - 1] == "."
    }) {
        return Some(RdfFormat::NTriples);
    }
    
    None
}