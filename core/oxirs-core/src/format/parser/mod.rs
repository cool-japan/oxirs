//! Unified RDF Parser Interface
//!
//! Provides a consistent API for parsing all supported RDF formats.
//! Extracted and adapted from OxiGraph with OxiRS enhancements.

mod helpers;
mod jsonld;
mod ntriples;
mod rdfxml;
mod turtle;

use super::error::ParseResult;
use super::format::RdfFormat;
use crate::model::Quad;
use std::io::Read;

/// Result type for quad parsing operations
pub type QuadParseResult = ParseResult<Quad>;

/// Iterator over parsed quads from a reader
pub struct ReaderQuadParser<'a, R: Read> {
    inner: Box<dyn Iterator<Item = QuadParseResult> + Send + 'a>,
    _phantom: std::marker::PhantomData<R>,
}

impl<'a, R: Read> ReaderQuadParser<'a, R> {
    /// Create a new reader parser
    pub fn new(iter: Box<dyn Iterator<Item = QuadParseResult> + Send + 'a>) -> Self {
        Self {
            inner: iter,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, R: Read> Iterator for ReaderQuadParser<'a, R> {
    type Item = QuadParseResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Iterator over parsed quads from a byte slice
pub struct SliceQuadParser<'a> {
    inner: Box<dyn Iterator<Item = QuadParseResult> + 'a>,
}

impl<'a> SliceQuadParser<'a> {
    /// Create a new slice parser
    pub fn new(iter: Box<dyn Iterator<Item = QuadParseResult> + 'a>) -> Self {
        Self { inner: iter }
    }
}

impl<'a> Iterator for SliceQuadParser<'a> {
    type Item = QuadParseResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Unified RDF parser supporting all formats
#[derive(Debug, Clone)]
pub struct RdfParser {
    format: RdfFormat,
    base_iri: Option<String>,
    prefixes: std::collections::HashMap<String, String>,
    lenient: bool,
}

impl RdfParser {
    /// Create a new parser for the specified format
    pub fn new(format: RdfFormat) -> Self {
        Self {
            format,
            base_iri: None,
            prefixes: std::collections::HashMap::new(),
            lenient: false,
        }
    }

    /// Set the base IRI for resolving relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        self.base_iri = Some(base_iri.into());
        self
    }

    /// Add a namespace prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.insert(prefix.into(), iri.into());
        self
    }

    /// Enable lenient parsing (skip some validations for performance)
    pub fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }

    /// Parse from a reader
    pub fn for_reader<R: Read + Send + 'static>(self, reader: R) -> ReaderQuadParser<'static, R> {
        match self.format {
            RdfFormat::Turtle => turtle::parse_reader(self, reader),
            RdfFormat::NTriples => ntriples::parse_ntriples_reader(self, reader),
            RdfFormat::NQuads => ntriples::parse_nquads_reader(self, reader),
            RdfFormat::TriG => ntriples::parse_trig_reader(self, reader),
            RdfFormat::RdfXml => rdfxml::parse_reader(self, reader),
            RdfFormat::JsonLd { .. } => jsonld::parse_reader(self, reader),
            RdfFormat::N3 => ntriples::parse_n3_reader(self, reader),
        }
    }

    /// Parse from a byte slice
    pub fn for_slice<'a>(self, slice: &'a [u8]) -> SliceQuadParser<'a> {
        match self.format {
            RdfFormat::Turtle => turtle::parse_slice(self, slice),
            RdfFormat::NTriples => ntriples::parse_ntriples_slice(self, slice),
            RdfFormat::NQuads => ntriples::parse_nquads_slice(self, slice),
            RdfFormat::TriG => ntriples::parse_trig_slice(self, slice),
            RdfFormat::RdfXml => rdfxml::parse_slice(self, slice),
            RdfFormat::JsonLd { .. } => jsonld::parse_slice(self, slice),
            RdfFormat::N3 => ntriples::parse_n3_slice(self, slice),
        }
    }

    /// Get the format being parsed
    pub fn format(&self) -> RdfFormat {
        self.format.clone()
    }

    /// Get the base IRI
    pub fn base_iri(&self) -> Option<&str> {
        self.base_iri.as_deref()
    }

    /// Get the prefixes
    pub fn prefixes(&self) -> &std::collections::HashMap<String, String> {
        &self.prefixes
    }

    /// Check if lenient parsing is enabled
    pub fn is_lenient(&self) -> bool {
        self.lenient
    }
}

impl Default for RdfParser {
    fn default() -> Self {
        Self::new(RdfFormat::Turtle)
    }
}
