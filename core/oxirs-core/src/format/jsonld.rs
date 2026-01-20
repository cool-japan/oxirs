//! JSON-LD Format Parser and Serializer
//!
//! Wrapper around the full JSON-LD implementation in oxirs-core/jsonld module.
//! Based on W3C JSON-LD specifications: <https://www.w3.org/TR/json-ld/>

use super::error::SerializeResult;
use super::error::{ParseResult, RdfParseError};
use super::format::JsonLdProfileSet;
use super::serializer::QuadSerializer;
use crate::jsonld;
use crate::model::{Quad, QuadRef};
use std::io::{Read, Write};

/// JSON-LD parser implementation
///
/// This wraps the full JSON-LD parser from the jsonld module and provides
/// a simplified interface for the format abstraction layer.
// Removed Debug, Clone - inner types don't implement them
pub struct JsonLdParser {
    inner: jsonld::JsonLdParser,
    profile: JsonLdProfileSet,
}

impl JsonLdParser {
    /// Create a new JSON-LD parser
    pub fn new() -> Self {
        Self {
            inner: jsonld::JsonLdParser::new(),
            profile: JsonLdProfileSet::empty(),
        }
    }

    /// Set the JSON-LD processing profile
    pub fn with_profile(mut self, profile: JsonLdProfileSet) -> Self {
        self.profile = profile.clone();
        // Convert format::JsonLdProfileSet to jsonld::JsonLdProfileSet
        let jsonld_profile_set = profile.to_jsonld_profile_set();
        self.inner = self.inner.with_profile(jsonld_profile_set);
        self
    }

    /// Set base IRI for resolving relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, RdfParseError> {
        let base_iri_str = base_iri.into();
        self.inner = self
            .inner
            .with_base_iri(base_iri_str.clone())
            .map_err(|e| RdfParseError::syntax(format!("Invalid base IRI: {e}")))?;
        Ok(self)
    }

    /// Assume lenient parsing (skip some validations)
    pub fn lenient(mut self) -> Self {
        self.inner = self.inner.lenient();
        self
    }

    /// Parse JSON-LD from a reader
    pub fn parse_reader<R: Read>(&self, reader: R) -> ParseResult<Vec<Quad>> {
        // Use the actual JSON-LD parser implementation
        self.inner
            .clone()
            .for_reader(reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| RdfParseError::syntax(format!("JSON-LD parse error: {e}")))
    }

    /// Parse JSON-LD from a byte slice
    pub fn parse_slice(&self, slice: &[u8]) -> ParseResult<Vec<Quad>> {
        // Use the actual JSON-LD parser implementation
        self.inner
            .clone()
            .for_slice(slice)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| RdfParseError::syntax(format!("JSON-LD parse error: {e}")))
    }

    /// Parse JSON-LD from a string
    pub fn parse_str(&self, input: &str) -> ParseResult<Vec<Quad>> {
        self.parse_slice(input.as_bytes())
    }

    /// Get the processing profile
    pub fn profile(&self) -> &JsonLdProfileSet {
        &self.profile
    }
}

impl Default for JsonLdParser {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON-LD serializer implementation
///
/// This wraps the full JSON-LD serializer from the jsonld module and provides
/// a simplified interface for the format abstraction layer.
#[derive(Clone)]
pub struct JsonLdSerializer {
    inner: jsonld::JsonLdSerializer,
    profile: JsonLdProfileSet,
}

impl JsonLdSerializer {
    /// Create a new JSON-LD serializer
    pub fn new() -> Self {
        Self {
            inner: jsonld::JsonLdSerializer::new(),
            profile: JsonLdProfileSet::empty(),
        }
    }

    /// Set the JSON-LD processing profile
    pub fn with_profile(mut self, profile: JsonLdProfileSet) -> Self {
        self.profile = profile;
        self
    }

    /// Add a prefix to the serializer
    pub fn with_prefix(
        mut self,
        prefix: impl Into<String>,
        iri: impl Into<String>,
    ) -> Result<Self, RdfParseError> {
        self.inner = self
            .inner
            .with_prefix(prefix, iri)
            .map_err(|e| RdfParseError::syntax(format!("Invalid prefix IRI: {e}")))?;
        Ok(self)
    }

    /// Set base IRI for generating relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, RdfParseError> {
        self.inner = self
            .inner
            .with_base_iri(base_iri)
            .map_err(|e| RdfParseError::syntax(format!("Invalid base IRI: {e}")))?;
        Ok(self)
    }

    /// Enable pretty formatting (no-op for JSON-LD streaming serializer)
    ///
    /// The underlying JSON-LD serializer always produces compact streaming output.
    /// This method is provided for API compatibility with other serializers.
    pub fn pretty(self) -> Self {
        // JSON-LD streaming serializer doesn't support pretty printing
        self
    }

    /// Create a writer-based serializer
    pub fn for_writer<W: Write>(self, writer: W) -> WriterJsonLdSerializer<W> {
        WriterJsonLdSerializer::new(writer, self)
    }

    /// Serialize quads to a JSON-LD string
    pub fn serialize_to_string(&self, quads: &[Quad]) -> SerializeResult<String> {
        let mut buffer = Vec::new();
        {
            let mut serializer = self.clone().for_writer(&mut buffer);
            for quad in quads {
                serializer.serialize_quad(quad.as_ref())?;
            }
            serializer.finish()?;
        }
        String::from_utf8(buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Get the processing profile
    pub fn profile(&self) -> &JsonLdProfileSet {
        &self.profile
    }
}

impl Default for JsonLdSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Writer-based JSON-LD serializer
///
/// This wraps the actual JSON-LD writer serializer implementation
pub struct WriterJsonLdSerializer<W: Write> {
    inner: jsonld::WriterJsonLdSerializer<W>,
}

impl<W: Write> WriterJsonLdSerializer<W> {
    /// Create a new writer serializer
    pub fn new(writer: W, config: JsonLdSerializer) -> Self {
        Self {
            inner: config.inner.for_writer(writer),
        }
    }

    /// Serialize a quad
    pub fn serialize_quad(&mut self, quad: QuadRef<'_>) -> SerializeResult<()> {
        self.inner
            .serialize_quad(quad)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Finish serialization and return the writer
    pub fn finish(self) -> SerializeResult<W> {
        Box::new(self.inner)
            .finish()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

impl<W: Write> QuadSerializer<W> for WriterJsonLdSerializer<W> {
    fn serialize_quad(&mut self, quad: QuadRef<'_>) -> SerializeResult<()> {
        self.serialize_quad(quad)
    }

    fn finish(self: Box<Self>) -> SerializeResult<W> {
        (*self).finish()
    }
}

/// JSON-LD context management utilities
///
/// For advanced context operations, use the full jsonld module directly.
/// This module provides basic re-exports for convenience.
pub mod context {
    /// Re-export context types from the main jsonld module
    pub use crate::jsonld::{JsonLdLoadDocumentOptions, JsonLdRemoteDocument};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::format::{JsonLdProfile, JsonLdProfileSet};
    use crate::model::*;
    use crate::vocab::rdf;

    #[test]
    fn test_jsonld_parser_creation() {
        let parser = JsonLdParser::new();
        assert!(parser.profile().profiles().is_empty());
    }

    #[test]
    fn test_jsonld_parser_configuration() {
        let profile = JsonLdProfileSet::from_profile(JsonLdProfile::Expanded);

        let parser = JsonLdParser::new()
            .with_profile(profile.clone())
            .with_base_iri("http://example.org/")
            .unwrap();

        assert_eq!(parser.profile(), &profile);
    }

    #[test]
    fn test_jsonld_serializer_creation() {
        let serializer = JsonLdSerializer::new();
        assert!(serializer.profile().profiles().is_empty());
    }

    #[test]
    fn test_jsonld_serializer_configuration() {
        let profile = JsonLdProfileSet::from_profile(JsonLdProfile::Compacted);

        let serializer = JsonLdSerializer::new()
            .with_profile(profile.clone())
            .with_prefix("schema", "http://schema.org/")
            .unwrap()
            .with_base_iri("http://example.org/")
            .unwrap();

        assert_eq!(serializer.profile(), &profile);
    }

    #[test]
    fn test_empty_json_parsing() {
        let parser = JsonLdParser::new();
        let result = parser.parse_str("{}");
        assert!(result.is_ok());
        // Empty JSON-LD documents might produce no quads
    }

    #[test]
    fn test_invalid_json_parsing() {
        let parser = JsonLdParser::new();
        let result = parser.parse_str("invalid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_jsonld_roundtrip() {
        // Create a simple quad
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = rdf::TYPE.clone();
        let object = NamedNode::new("http://example.org/Type").unwrap();
        let quad = Quad::new(subject, predicate, object, GraphName::DefaultGraph);

        // Serialize to JSON-LD
        let serializer = JsonLdSerializer::new()
            .with_prefix("ex", "http://example.org/")
            .unwrap();
        let json_ld = serializer.serialize_to_string(std::slice::from_ref(&quad));
        assert!(json_ld.is_ok());

        // Parse back
        let parser = JsonLdParser::new();
        let parsed_quads = parser.parse_str(&json_ld.unwrap());
        assert!(parsed_quads.is_ok());

        // Verify we got at least one quad back
        let quads = parsed_quads.unwrap();
        assert!(!quads.is_empty());
    }
}
