//! Format Conversion Utilities
//!
//! This module provides utilities for converting between different RDF formats.
//! Supports both in-memory and streaming conversions with format auto-detection.
//!
//! # Examples
//!
//! ## Basic Format Conversion
//!
//! ```rust
//! use oxirs_ttl::toolkit::format_converter::FormatConverter;
//! use oxirs_ttl::toolkit::RdfFormat;
//!
//! let turtle_data = r#"
//! @prefix ex: <http://example.org/> .
//! ex:subject ex:predicate "object" .
//! "#;
//!
//! let converter = FormatConverter::new();
//! let ntriples = converter.convert_string(
//!     turtle_data,
//!     RdfFormat::Turtle,
//!     RdfFormat::NTriples
//! )?;
//!
//! assert!(ntriples.contains("<http://example.org/subject>"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Streaming Conversion
//!
//! ```rust
//! use oxirs_ttl::toolkit::format_converter::FormatConverter;
//! use oxirs_ttl::toolkit::RdfFormat;
//! use std::io::Cursor;
//!
//! let turtle_data = b"<http://s> <http://p> <http://o> .";
//! let input = Cursor::new(turtle_data);
//! let mut output = Vec::new();
//!
//! let converter = FormatConverter::new();
//! converter.convert_stream(
//!     input,
//!     &mut output,
//!     RdfFormat::NTriples,
//!     RdfFormat::Turtle
//! )?;
//!
//! let result = String::from_utf8(output)?;
//! assert!(result.contains("<http://s>"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::TurtleParseError;
use crate::formats::nquads::{NQuadsParser, NQuadsSerializer};
use crate::formats::ntriples::{NTriplesParser, NTriplesSerializer};
use crate::formats::trig::TriGParser;
use crate::formats::turtle::{TurtleParser, TurtleSerializer};
use crate::toolkit::{Parser, RdfFormat, SerializationConfig, Serializer};
use oxirs_core::model::{Quad, Triple};
use std::io::{BufRead, BufReader, Write};

/// Result type for format conversion operations
pub type ConversionResult<T> = Result<T, ConversionError>;

/// Errors that can occur during format conversion
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    /// Parse error during conversion
    #[error("Parse error: {0}")]
    ParseError(#[from] TurtleParseError),

    /// I/O error during conversion
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Unsupported format combination
    #[error("Unsupported conversion from {0:?} to {1:?}")]
    UnsupportedConversion(RdfFormat, RdfFormat),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Configuration for format conversion
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// Serialization configuration
    pub serialization: SerializationConfig,
    /// Whether to preserve prefixes when possible
    pub preserve_prefixes: bool,
    /// Whether to use lenient parsing
    pub lenient_parsing: bool,
    /// Batch size for streaming conversion
    pub batch_size: usize,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            serialization: SerializationConfig::default(),
            preserve_prefixes: true,
            lenient_parsing: false,
            batch_size: 10_000,
        }
    }
}

impl ConversionConfig {
    /// Create a new conversion config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable lenient parsing
    pub fn with_lenient(mut self, lenient: bool) -> Self {
        self.lenient_parsing = lenient;
        self
    }

    /// Set whether to preserve prefixes
    pub fn with_preserve_prefixes(mut self, preserve: bool) -> Self {
        self.preserve_prefixes = preserve;
        self
    }

    /// Set batch size for streaming
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set serialization configuration
    pub fn with_serialization(mut self, config: SerializationConfig) -> Self {
        self.serialization = config;
        self
    }
}

/// Format converter for RDF data
///
/// Provides methods for converting between different RDF formats.
#[derive(Debug)]
pub struct FormatConverter {
    config: ConversionConfig,
}

impl FormatConverter {
    /// Create a new format converter with default configuration
    pub fn new() -> Self {
        Self {
            config: ConversionConfig::default(),
        }
    }

    /// Create a converter with custom configuration
    pub fn with_config(config: ConversionConfig) -> Self {
        Self { config }
    }

    /// Convert a string from one format to another
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_ttl::toolkit::format_converter::FormatConverter;
    /// use oxirs_ttl::toolkit::RdfFormat;
    ///
    /// let converter = FormatConverter::new();
    /// let turtle = r#"<http://s> <http://p> <http://o> ."#;
    /// let ntriples = converter.convert_string(
    ///     turtle,
    ///     RdfFormat::Turtle,
    ///     RdfFormat::NTriples
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn convert_string(
        &self,
        input: &str,
        from: RdfFormat,
        to: RdfFormat,
    ) -> ConversionResult<String> {
        let mut output = Vec::new();
        let input_bytes = input.as_bytes().to_vec();
        let cursor = std::io::Cursor::new(input_bytes);
        self.convert_stream(cursor, &mut output, from, to)?;
        String::from_utf8(output).map_err(|e| {
            ConversionError::SerializationError(format!("Invalid UTF-8 output: {}", e))
        })
    }

    /// Convert between formats using streams
    ///
    /// This is the most memory-efficient method for large datasets.
    pub fn convert_stream<R: BufRead + 'static, W: Write>(
        &self,
        input: R,
        output: &mut W,
        from: RdfFormat,
        to: RdfFormat,
    ) -> ConversionResult<()> {
        // Check if formats support triples or quads
        let from_has_quads = matches!(from, RdfFormat::NQuads | RdfFormat::TriG);
        let to_has_quads = matches!(to, RdfFormat::NQuads | RdfFormat::TriG);

        if from_has_quads && !to_has_quads {
            // Convert quads to triples (extract default graph)
            let quads = self.parse_quads(input, from)?;
            let triples: Vec<Triple> = quads
                .into_iter()
                .filter_map(|q| match q.graph_name() {
                    oxirs_core::model::GraphName::DefaultGraph => Some(Triple::new(
                        q.subject().clone(),
                        q.predicate().clone(),
                        q.object().clone(),
                    )),
                    _ => None,
                })
                .collect();
            self.serialize_triples(&triples, output, to)?;
        } else if !from_has_quads && to_has_quads {
            // Convert triples to quads (add to default graph)
            let triples = self.parse_triples(input, from)?;
            let quads: Vec<Quad> = triples
                .into_iter()
                .map(|t| {
                    Quad::new(
                        t.subject().clone(),
                        t.predicate().clone(),
                        t.object().clone(),
                        oxirs_core::model::GraphName::DefaultGraph,
                    )
                })
                .collect();
            self.serialize_quads(&quads, output, to)?;
        } else if from_has_quads && to_has_quads {
            // Both use quads
            let quads = self.parse_quads(input, from)?;
            self.serialize_quads(&quads, output, to)?;
        } else {
            // Both use triples
            let triples = self.parse_triples(input, from)?;
            self.serialize_triples(&triples, output, to)?;
        }

        Ok(())
    }

    /// Parse triples from input
    fn parse_triples<R: BufRead + 'static>(
        &self,
        input: R,
        format: RdfFormat,
    ) -> ConversionResult<Vec<Triple>> {
        match format {
            RdfFormat::Turtle => {
                let parser = if self.config.lenient_parsing {
                    TurtleParser::new_lenient()
                } else {
                    TurtleParser::new()
                };
                parser
                    .for_reader(input)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(ConversionError::from)
            }
            RdfFormat::NTriples => {
                let parser = NTriplesParser::new();
                parser
                    .for_reader(input)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(ConversionError::from)
            }
            _ => Err(ConversionError::UnsupportedConversion(
                format,
                RdfFormat::Turtle,
            )),
        }
    }

    /// Parse quads from input
    fn parse_quads<R: BufRead + 'static>(
        &self,
        input: R,
        format: RdfFormat,
    ) -> ConversionResult<Vec<Quad>> {
        match format {
            RdfFormat::NQuads => {
                let parser = NQuadsParser::new();
                parser
                    .for_reader(input)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(ConversionError::from)
            }
            RdfFormat::TriG => {
                let parser = TriGParser::new();
                parser
                    .for_reader(input)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(ConversionError::from)
            }
            _ => Err(ConversionError::UnsupportedConversion(
                format,
                RdfFormat::TriG,
            )),
        }
    }

    /// Serialize triples to output
    fn serialize_triples<W: Write>(
        &self,
        triples: &[Triple],
        output: &mut W,
        format: RdfFormat,
    ) -> ConversionResult<()> {
        match format {
            RdfFormat::Turtle => {
                let serializer = TurtleSerializer::with_config(self.config.serialization.clone());
                serializer
                    .serialize(triples, output)
                    .map_err(|e| ConversionError::SerializationError(e.to_string()))?;
            }
            RdfFormat::NTriples => {
                let serializer = NTriplesSerializer::new();
                serializer
                    .serialize(triples, output)
                    .map_err(|e| ConversionError::SerializationError(e.to_string()))?;
            }
            _ => {
                return Err(ConversionError::UnsupportedConversion(
                    RdfFormat::Turtle,
                    format,
                ))
            }
        }
        Ok(())
    }

    /// Serialize quads to output
    fn serialize_quads<W: Write>(
        &self,
        quads: &[Quad],
        output: &mut W,
        format: RdfFormat,
    ) -> ConversionResult<()> {
        match format {
            RdfFormat::NQuads => {
                let serializer = NQuadsSerializer::new();
                serializer
                    .serialize(quads, output)
                    .map_err(|e| ConversionError::SerializationError(e.to_string()))?;
            }
            RdfFormat::TriG => {
                // TriG serialization not yet implemented in serializer
                // For now, serialize as N-Quads
                let serializer = NQuadsSerializer::new();
                serializer
                    .serialize(quads, output)
                    .map_err(|e| ConversionError::SerializationError(e.to_string()))?;
            }
            _ => {
                return Err(ConversionError::UnsupportedConversion(
                    RdfFormat::TriG,
                    format,
                ))
            }
        }
        Ok(())
    }

    /// Convert a file from one format to another
    pub fn convert_file(
        &self,
        input_path: &str,
        output_path: &str,
        from: RdfFormat,
        to: RdfFormat,
    ) -> ConversionResult<ConversionStats> {
        let input = std::fs::File::open(input_path)?;
        let reader = BufReader::new(input);

        let mut output = std::fs::File::create(output_path)?;

        let start = std::time::Instant::now();
        self.convert_stream(reader, &mut output, from, to)?;
        let duration = start.elapsed();

        Ok(ConversionStats {
            duration,
            items_processed: 0, // TODO: track this
        })
    }
}

impl Default for FormatConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a conversion operation
#[derive(Debug, Clone)]
pub struct ConversionStats {
    /// Time taken for conversion
    pub duration: std::time::Duration,
    /// Number of items processed
    pub items_processed: usize,
}

impl ConversionStats {
    /// Get throughput in items per second
    pub fn throughput(&self) -> f64 {
        if self.duration.as_secs_f64() > 0.0 {
            self.items_processed as f64 / self.duration.as_secs_f64()
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turtle_to_ntriples() {
        let converter = FormatConverter::new();
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate "object" .
        "#;

        let result = converter
            .convert_string(turtle, RdfFormat::Turtle, RdfFormat::NTriples)
            .expect("conversion should succeed");

        assert!(result.contains("<http://example.org/subject>"));
        assert!(result.contains("<http://example.org/predicate>"));
        assert!(result.contains("\"object\""));
    }

    #[test]
    fn test_ntriples_to_turtle() {
        let converter = FormatConverter::new();
        let ntriples = "<http://example.org/s> <http://example.org/p> \"o\" .";

        let result = converter
            .convert_string(ntriples, RdfFormat::NTriples, RdfFormat::Turtle)
            .expect("conversion should succeed");

        assert!(result.contains("<http://example.org/s>"));
    }

    #[test]
    fn test_streaming_conversion() {
        let converter = FormatConverter::new();
        let turtle = b"<http://s> <http://p> <http://o> ." as &[u8];
        let mut output = Vec::new();

        converter
            .convert_stream(turtle, &mut output, RdfFormat::Turtle, RdfFormat::NTriples)
            .expect("conversion should succeed");

        let result = String::from_utf8(output).expect("valid UTF-8");
        assert!(result.contains("<http://s>"));
    }

    #[test]
    fn test_config_builder() {
        let config = ConversionConfig::new()
            .with_lenient(true)
            .with_preserve_prefixes(false)
            .with_batch_size(5000);

        assert!(config.lenient_parsing);
        assert!(!config.preserve_prefixes);
        assert_eq!(config.batch_size, 5000);
    }
}
