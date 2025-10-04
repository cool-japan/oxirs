//! Unified RDF Serializer Interface
//!
//! Provides a consistent API for serializing to all supported RDF formats.
//! Extracted and adapted from OxiGraph with OxiRS enhancements.

use super::error::FormatError;
pub use super::error::SerializeResult;
use super::format::RdfFormat;
use crate::model::{Quad, QuadRef, Triple, TripleRef};
use std::collections::HashMap;
use std::io::Write;

/// Result type for quad serialization operations
pub type QuadSerializeResult = SerializeResult<()>;

/// Writer-based quad serializer
pub struct WriterQuadSerializer<W: Write> {
    inner: Box<dyn QuadSerializer<W>>,
}

impl<W: Write> WriterQuadSerializer<W> {
    /// Create a new writer serializer
    pub fn new(serializer: Box<dyn QuadSerializer<W>>) -> Self {
        Self { inner: serializer }
    }

    /// Serialize a quad
    pub fn serialize_quad<'a>(&mut self, quad: impl Into<QuadRef<'a>>) -> QuadSerializeResult {
        self.inner.serialize_quad(quad.into())
    }

    /// Serialize a triple (placed in default graph)
    pub fn serialize_triple<'a>(
        &mut self,
        triple: impl Into<TripleRef<'a>>,
    ) -> QuadSerializeResult {
        let quad = triple.into().in_graph(None);
        self.serialize_quad(quad)
    }

    /// Serialize multiple quads
    pub fn serialize_quads<I>(&mut self, quads: I) -> QuadSerializeResult
    where
        I: IntoIterator,
        I::Item: Into<QuadRef<'static>>,
    {
        for quad in quads {
            self.inner.serialize_quad(quad.into())?;
        }
        Ok(())
    }

    /// Finish serialization and return the writer
    pub fn finish(self) -> SerializeResult<W> {
        self.inner.finish()
    }
}

/// Trait for serializing quads to a writer
pub trait QuadSerializer<W: Write> {
    /// Serialize a quad
    fn serialize_quad(&mut self, quad: QuadRef<'_>) -> QuadSerializeResult;

    /// Finish serialization and return the writer
    fn finish(self: Box<Self>) -> SerializeResult<W>;
}

/// Extension trait for bulk serialization operations
pub trait QuadSerializerExt<W: Write>: QuadSerializer<W> {
    /// Serialize multiple quads
    fn serialize_quads<I>(&mut self, quads: I) -> QuadSerializeResult
    where
        I: IntoIterator,
        I::Item: Into<QuadRef<'static>>,
    {
        for quad in quads {
            self.serialize_quad(quad.into())?;
        }
        Ok(())
    }
}

/// Blanket implementation for all QuadSerializer types
impl<W: Write, T: QuadSerializer<W>> QuadSerializerExt<W> for T {}

/// Unified RDF serializer supporting all formats
pub struct RdfSerializer {
    format: RdfFormat,
    base_iri: Option<String>,
    prefixes: HashMap<String, String>,
    pretty: bool,
}

impl RdfSerializer {
    /// Create a new serializer for the specified format
    pub fn new(format: RdfFormat) -> Self {
        Self {
            format,
            base_iri: None,
            prefixes: HashMap::new(),
            pretty: false,
        }
    }

    /// Set the base IRI for relative IRI generation
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        self.base_iri = Some(base_iri.into());
        self
    }

    /// Add a namespace prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.insert(prefix.into(), iri.into());
        self
    }

    /// Enable pretty formatting (indentation, line breaks)
    pub fn pretty(mut self) -> Self {
        self.pretty = true;
        self
    }

    /// Create a writer-based serializer
    pub fn for_writer<W: Write + 'static>(self, writer: W) -> WriterQuadSerializer<W> {
        match self.format {
            RdfFormat::Turtle => self.create_turtle_serializer(writer),
            RdfFormat::NTriples => self.create_ntriples_serializer(writer),
            RdfFormat::NQuads => self.create_nquads_serializer(writer),
            RdfFormat::TriG => self.create_trig_serializer(writer),
            RdfFormat::RdfXml => self.create_rdfxml_serializer(writer),
            RdfFormat::JsonLd { .. } => self.create_jsonld_serializer(writer),
            RdfFormat::N3 => self.create_n3_serializer(writer),
        }
    }

    /// Get the format being serialized
    pub fn format(&self) -> RdfFormat {
        self.format.clone()
    }

    /// Get the base IRI
    pub fn base_iri(&self) -> Option<&str> {
        self.base_iri.as_deref()
    }

    /// Get the prefixes
    pub fn prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }

    /// Check if pretty formatting is enabled
    pub fn is_pretty(&self) -> bool {
        self.pretty
    }

    // Format-specific serializer implementations

    fn create_turtle_serializer<W: Write + 'static>(self, writer: W) -> WriterQuadSerializer<W> {
        // Use existing Turtle serializer implementation
        let mut turtle_serializer = super::turtle::TurtleSerializer::new();

        // Apply configuration
        if let Some(base) = self.base_iri {
            turtle_serializer = turtle_serializer.with_base_iri(&base);
        }
        for (prefix, iri) in self.prefixes {
            turtle_serializer = turtle_serializer.with_prefix(&prefix, &iri);
        }
        if self.pretty {
            turtle_serializer = turtle_serializer.pretty();
        }

        WriterQuadSerializer::new(Box::new(turtle_serializer.for_writer(writer)))
    }

    fn create_ntriples_serializer<W: Write + 'static>(self, writer: W) -> WriterQuadSerializer<W> {
        // Use existing N-Triples serializer implementation
        let ntriples_serializer = super::ntriples::NTriplesSerializer::new().for_writer(writer);
        WriterQuadSerializer::new(Box::new(ntriples_serializer))
    }

    fn create_nquads_serializer<W: Write + 'static>(self, writer: W) -> WriterQuadSerializer<W> {
        // Use N-Quads serializer implementation
        let nquads_serializer = super::nquads::NQuadsSerializer::new().for_writer(writer);
        WriterQuadSerializer::new(Box::new(nquads_serializer))
    }

    fn create_trig_serializer<W: Write + 'static>(self, writer: W) -> WriterQuadSerializer<W> {
        // Use TriG serializer implementation
        let mut trig_serializer = super::trig::TriGSerializer::new();

        // Apply configuration
        if let Some(base) = self.base_iri {
            trig_serializer = trig_serializer.with_base_iri(&base);
        }
        for (prefix, iri) in self.prefixes {
            trig_serializer = trig_serializer.with_prefix(&prefix, &iri);
        }
        if self.pretty {
            trig_serializer = trig_serializer.pretty();
        }

        WriterQuadSerializer::new(Box::new(trig_serializer.for_writer(writer)))
    }

    fn create_rdfxml_serializer<W: Write + 'static>(self, writer: W) -> WriterQuadSerializer<W> {
        // Use existing RDF/XML serializer implementation
        let mut rdfxml_serializer = super::rdfxml::RdfXmlSerializer::new();

        // Apply configuration
        if self.pretty {
            rdfxml_serializer = rdfxml_serializer.pretty();
        }

        WriterQuadSerializer::new(Box::new(rdfxml_serializer.for_writer(writer)))
    }

    fn create_jsonld_serializer<W: Write + 'static>(self, writer: W) -> WriterQuadSerializer<W> {
        // Use existing JSON-LD serializer implementation
        let mut jsonld_serializer = super::jsonld::JsonLdSerializer::new();

        // Apply configuration
        if self.pretty {
            jsonld_serializer = jsonld_serializer.pretty();
        }

        WriterQuadSerializer::new(Box::new(jsonld_serializer.for_writer(writer)))
    }

    fn create_n3_serializer<W: Write + 'static>(self, writer: W) -> WriterQuadSerializer<W> {
        // Use N3 serializer implementation
        let mut n3_serializer = super::n3::N3Serializer::new();

        // Apply configuration
        if let Some(base) = self.base_iri {
            n3_serializer = n3_serializer.with_base_iri(&base);
        }
        for (prefix, iri) in self.prefixes {
            n3_serializer = n3_serializer.with_prefix(&prefix, &iri);
        }
        if self.pretty {
            n3_serializer = n3_serializer.pretty();
        }

        WriterQuadSerializer::new(Box::new(n3_serializer.for_writer(writer)))
    }
}

impl Default for RdfSerializer {
    fn default() -> Self {
        Self::new(RdfFormat::default())
    }
}

/// Serialization configuration for fine-grained control
#[derive(Debug, Clone)]
pub struct SerializeConfig {
    /// Use compact formatting
    pub compact: bool,
    /// Indentation string for pretty formatting
    pub indent: String,
    /// Line ending style
    pub line_ending: LineEnding,
    /// Maximum line length for wrapping
    pub max_line_length: Option<usize>,
    /// Sort output by subject/predicate
    pub sort_output: bool,
    /// Include comments in output
    pub include_comments: bool,
    /// Validate output during serialization
    pub validate_output: bool,
}

/// Line ending styles
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineEnding {
    /// Unix-style line endings (\n)
    Unix,
    /// Windows-style line endings (\r\n)
    Windows,
    /// Mac-style line endings (\r)
    Mac,
    /// Platform-default line endings
    Platform,
}

impl Default for SerializeConfig {
    fn default() -> Self {
        Self {
            compact: false,
            indent: "  ".to_string(),
            line_ending: LineEnding::Platform,
            max_line_length: None,
            sort_output: false,
            include_comments: false,
            validate_output: true,
        }
    }
}

impl LineEnding {
    /// Get the line ending string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unix => "\n",
            Self::Windows => "\r\n",
            Self::Mac => "\r",
            Self::Platform => {
                #[cfg(windows)]
                return "\r\n";
                #[cfg(not(windows))]
                return "\n";
            }
        }
    }
}

/// Advanced serializer with configuration support
pub struct ConfigurableSerializer {
    serializer: RdfSerializer,
    config: SerializeConfig,
}

impl ConfigurableSerializer {
    /// Create a new configurable serializer
    pub fn new(format: RdfFormat, config: SerializeConfig) -> Self {
        Self {
            serializer: RdfSerializer::new(format),
            config,
        }
    }

    /// Set base IRI
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        self.serializer = self.serializer.with_base_iri(base_iri);
        self
    }

    /// Add prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.serializer = self.serializer.with_prefix(prefix, iri);
        self
    }

    /// Create a writer with configuration
    pub fn for_writer<W: Write + 'static>(self, writer: W) -> WriterQuadSerializer<W> {
        // Apply configuration settings and create serializer
        let mut serializer = self.serializer;

        if !self.config.compact {
            serializer = serializer.pretty();
        }

        // TODO: Apply other configuration options
        serializer.for_writer(writer)
    }

    /// Get the configuration
    pub fn config(&self) -> &SerializeConfig {
        &self.config
    }

    /// Get the serializer
    pub fn serializer(&self) -> &RdfSerializer {
        &self.serializer
    }
}

/// Simple serialization functions for common use cases
pub mod simple {
    use super::*;

    /// Serialize triples to a string in the specified format
    pub fn serialize_triples_to_string(
        triples: &[Triple],
        format: RdfFormat,
    ) -> Result<String, FormatError> {
        let buffer = Vec::new();
        let mut serializer = RdfSerializer::new(format).for_writer(buffer);
        for triple in triples {
            serializer.serialize_triple(triple.as_ref())?;
        }
        let buffer = serializer.finish()?;
        String::from_utf8(buffer).map_err(|e| FormatError::invalid_data(e.to_string()))
    }

    /// Serialize quads to a string in the specified format
    pub fn serialize_quads_to_string(
        quads: &[Quad],
        format: RdfFormat,
    ) -> Result<String, FormatError> {
        let buffer = Vec::new();
        let mut serializer = RdfSerializer::new(format).for_writer(buffer);
        for quad in quads {
            serializer.serialize_quad(quad.as_ref())?;
        }
        let buffer = serializer.finish()?;
        String::from_utf8(buffer).map_err(|e| FormatError::invalid_data(e.to_string()))
    }

    /// Serialize triples to Turtle string
    pub fn serialize_turtle(triples: &[Triple]) -> Result<String, FormatError> {
        serialize_triples_to_string(triples, RdfFormat::Turtle)
    }

    /// Serialize triples to N-Triples string
    pub fn serialize_ntriples(triples: &[Triple]) -> Result<String, FormatError> {
        serialize_triples_to_string(triples, RdfFormat::NTriples)
    }

    /// Serialize quads to N-Quads string
    pub fn serialize_nquads(quads: &[Quad]) -> Result<String, FormatError> {
        serialize_quads_to_string(quads, RdfFormat::NQuads)
    }

    /// Serialize quads to TriG string
    pub fn serialize_trig(quads: &[Quad]) -> Result<String, FormatError> {
        serialize_quads_to_string(quads, RdfFormat::TriG)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serializer_creation() {
        let serializer = RdfSerializer::new(RdfFormat::Turtle);
        assert_eq!(serializer.format(), RdfFormat::Turtle);
        assert!(serializer.base_iri().is_none());
        assert!(serializer.prefixes().is_empty());
        assert!(!serializer.is_pretty());
    }

    #[test]
    fn test_serializer_configuration() {
        let serializer = RdfSerializer::new(RdfFormat::Turtle)
            .with_base_iri("http://example.org/")
            .with_prefix("ex", "http://example.org/ns#")
            .pretty();

        assert_eq!(serializer.base_iri(), Some("http://example.org/"));
        assert_eq!(
            serializer.prefixes().get("ex"),
            Some(&"http://example.org/ns#".to_string())
        );
        assert!(serializer.is_pretty());
    }

    #[test]
    fn test_configurable_serializer() {
        let config = SerializeConfig {
            compact: true,
            sort_output: true,
            ..Default::default()
        };

        let serializer = ConfigurableSerializer::new(RdfFormat::NQuads, config);
        assert!(serializer.config().compact);
        assert!(serializer.config().sort_output);
    }

    #[test]
    fn test_serialize_config_default() {
        let config = SerializeConfig::default();
        assert!(!config.compact);
        assert_eq!(config.indent, "  ");
        assert_eq!(config.line_ending, LineEnding::Platform);
        assert_eq!(config.max_line_length, None);
        assert!(!config.sort_output);
        assert!(!config.include_comments);
        assert!(config.validate_output);
    }

    #[test]
    fn test_line_ending() {
        assert_eq!(LineEnding::Unix.as_str(), "\n");
        assert_eq!(LineEnding::Windows.as_str(), "\r\n");
        assert_eq!(LineEnding::Mac.as_str(), "\r");
        // Platform depends on the compilation target
    }
}
