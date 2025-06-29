//! RDF Format Support Module
//!
//! Phase 3 of OxiGraph extraction: Complete format support for all major RDF serializations.
//! Extracted and adapted from OxiGraph format libraries with OxiRS enhancements.
//!
//! Provides unified parsing and serialization for:
//! - Turtle (.ttl)
//! - N-Triples (.nt)
//! - N-Quads (.nq)
//! - TriG (.trig)
//! - RDF/XML (.rdf, .xml)
//! - JSON-LD (.jsonld)
//! - N3 (.n3)

pub mod error;
pub mod format;
pub mod jsonld;
pub mod n3_lexer;
pub mod ntriples;
pub mod parser;
pub mod rdfxml;
pub mod serializer;
pub mod toolkit;
pub mod turtle;
pub mod turtle_grammar;

// Re-export key types
pub use error::{FormatError, RdfParseError, RdfSyntaxError, TextPosition};
pub use format::RdfFormat;
pub use parser::{QuadParseResult, RdfParser, ReaderQuadParser, SliceQuadParser};
pub use serializer::{QuadSerializeResult, RdfSerializer, WriterQuadSerializer};

// Format-specific re-exports
pub use format::{JsonLdProfile, JsonLdProfileSet};
pub use jsonld::{JsonLdParser, JsonLdSerializer};
pub use ntriples::{NTriplesParser, NTriplesSerializer};
pub use rdfxml::{RdfXmlParser, RdfXmlSerializer};
pub use turtle::{TurtleParser, TurtleSerializer};

use crate::model::{Quad, Triple};
use crate::OxirsError;
use std::io::{Read, Write};

/// Result type for format operations
pub type FormatResult<T> = Result<T, FormatError>;

/// Trait for format detection from content or metadata
pub trait FormatDetection {
    /// Detect format from file extension
    fn from_extension(extension: &str) -> Option<RdfFormat>;

    /// Detect format from media type
    fn from_media_type(media_type: &str) -> Option<RdfFormat>;

    /// Detect format from content analysis (magic bytes, syntax patterns)
    fn from_content(content: &[u8]) -> Option<RdfFormat>;

    /// Detect format from filename
    fn from_filename(filename: &str) -> Option<RdfFormat> {
        std::path::Path::new(filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(Self::from_extension)
    }
}

/// Unified RDF format handler combining parsing and serialization
pub struct FormatHandler {
    format: RdfFormat,
}

impl FormatHandler {
    /// Create a new format handler for the specified format
    pub fn new(format: RdfFormat) -> Self {
        Self { format }
    }

    /// Parse RDF from a reader into quads
    pub fn parse_quads<R: Read + Send>(&self, reader: R) -> FormatResult<Vec<Quad>> {
        let parser = RdfParser::new(self.format.clone());
        let mut quads = Vec::new();

        for quad_result in parser.for_reader(reader) {
            quads.push(quad_result?);
        }

        Ok(quads)
    }

    /// Parse RDF from a reader into triples (only default graph)
    pub fn parse_triples<R: Read + Send>(&self, reader: R) -> FormatResult<Vec<Triple>> {
        let quads = self.parse_quads(reader)?;
        Ok(quads
            .into_iter()
            .filter_map(|quad| quad.triple_in_default_graph())
            .collect())
    }

    /// Serialize quads to a writer
    pub fn serialize_quads<W: Write + 'static>(
        &self,
        writer: W,
        quads: &[Quad],
    ) -> FormatResult<()> {
        let mut serializer = RdfSerializer::new(self.format.clone()).for_writer(writer);

        for quad in quads {
            serializer.serialize_quad(quad.as_ref())?;
        }

        serializer.finish()?;
        Ok(())
    }

    /// Serialize triples to a writer (places in default graph)
    pub fn serialize_triples<W: Write + 'static>(
        &self,
        writer: W,
        triples: &[Triple],
    ) -> FormatResult<()> {
        let quads: Vec<Quad> = triples.iter().map(|triple| triple.clone().into()).collect();
        self.serialize_quads(writer, &quads)
    }

    /// Get the format
    pub fn format(&self) -> RdfFormat {
        self.format.clone()
    }
}

impl FormatDetection for FormatHandler {
    fn from_extension(extension: &str) -> Option<RdfFormat> {
        RdfFormat::from_extension(extension)
    }

    fn from_media_type(media_type: &str) -> Option<RdfFormat> {
        RdfFormat::from_media_type(media_type)
    }

    fn from_content(content: &[u8]) -> Option<RdfFormat> {
        // Simple heuristics for format detection
        let content_str = std::str::from_utf8(content).ok()?;
        let content_lower = content_str.to_lowercase();

        // Check for XML-like structures (RDF/XML)
        if content_lower.contains("<?xml") || content_lower.contains("<rdf:") {
            return Some(RdfFormat::RdfXml);
        }

        // Check for JSON-LD
        if content_lower.trim_start().starts_with('{')
            && (content_lower.contains("@context") || content_lower.contains("@type"))
        {
            return Some(RdfFormat::JsonLd {
                profile: JsonLdProfileSet::empty(),
            });
        }

        // Check for Turtle-family formats
        if content_lower.contains("@prefix") || content_lower.contains("@base") {
            if content_lower.contains("graph") {
                return Some(RdfFormat::TriG);
            }
            return Some(RdfFormat::Turtle);
        }

        // Check for N-Quads (4 terms per line)
        let lines: Vec<&str> = content_str.lines().take(10).collect();
        if lines.iter().any(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            parts.len() >= 4 && line.ends_with(" .")
        }) {
            return Some(RdfFormat::NQuads);
        }

        // Check for N-Triples (3 terms per line)
        if lines.iter().any(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            parts.len() >= 3 && line.ends_with(" .")
        }) {
            return Some(RdfFormat::NTriples);
        }

        None
    }
}

/// Convert OxiRS errors to format errors
impl From<OxirsError> for FormatError {
    fn from(err: OxirsError) -> Self {
        FormatError::InvalidData(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection_from_extension() {
        assert_eq!(
            FormatHandler::from_extension("ttl"),
            Some(RdfFormat::Turtle)
        );
        assert_eq!(
            FormatHandler::from_extension("nt"),
            Some(RdfFormat::NTriples)
        );
        assert_eq!(
            FormatHandler::from_extension("jsonld"),
            Some(RdfFormat::JsonLd {
                profile: JsonLdProfileSet::empty()
            })
        );
        assert_eq!(FormatHandler::from_extension("unknown"), None);
    }

    #[test]
    fn test_format_detection_from_content() {
        let turtle_content = b"@prefix ex: <http://example.org/> .\nex:foo ex:bar ex:baz .";
        assert_eq!(
            FormatHandler::from_content(turtle_content),
            Some(RdfFormat::Turtle)
        );

        let jsonld_content = br#"{"@context": "http://example.org/", "@type": "Person"}"#;
        assert_eq!(
            FormatHandler::from_content(jsonld_content),
            Some(RdfFormat::JsonLd {
                profile: JsonLdProfileSet::empty()
            })
        );

        let rdfxml_content = b"<?xml version=\"1.0\"?>\n<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">";
        assert_eq!(
            FormatHandler::from_content(rdfxml_content),
            Some(RdfFormat::RdfXml)
        );
    }
}
