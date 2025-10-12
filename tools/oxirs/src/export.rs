//! Data export utilities

use std::io::Write;
use std::path::Path;

/// Supported export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportFormat {
    /// Turtle (Terse RDF Triple Language)
    Turtle,
    /// N-Triples
    NTriples,
    /// RDF/XML
    RdfXml,
    /// JSON-LD
    JsonLd,
    /// TriG (Turtle with named graphs)
    TriG,
    /// N-Quads
    NQuads,
}

impl ExportFormat {
    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::Turtle => "ttl",
            ExportFormat::NTriples => "nt",
            ExportFormat::RdfXml => "rdf",
            ExportFormat::JsonLd => "jsonld",
            ExportFormat::TriG => "trig",
            ExportFormat::NQuads => "nq",
        }
    }

    /// Get MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ExportFormat::Turtle => "text/turtle",
            ExportFormat::NTriples => "application/n-triples",
            ExportFormat::RdfXml => "application/rdf+xml",
            ExportFormat::JsonLd => "application/ld+json",
            ExportFormat::TriG => "application/trig",
            ExportFormat::NQuads => "application/n-quads",
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportFormat::Turtle => write!(f, "Turtle"),
            ExportFormat::NTriples => write!(f, "N-Triples"),
            ExportFormat::RdfXml => write!(f, "RDF/XML"),
            ExportFormat::JsonLd => write!(f, "JSON-LD"),
            ExportFormat::TriG => write!(f, "TriG"),
            ExportFormat::NQuads => write!(f, "N-Quads"),
        }
    }
}

/// Export configuration options
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Output format
    pub format: ExportFormat,
    /// Pretty print output
    pub pretty: bool,
    /// Include base URI
    pub base_uri: Option<String>,
    /// Include prefixes (for applicable formats)
    pub use_prefixes: bool,
    /// Compression level (0-9, if applicable)
    pub compression: Option<u8>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Turtle,
            pretty: true,
            base_uri: None,
            use_prefixes: true,
            compression: None,
        }
    }
}

/// Export data to various formats
pub struct Exporter {
    config: ExportConfig,
}

impl Exporter {
    /// Create a new exporter with default configuration
    pub fn new() -> Self {
        Self {
            config: ExportConfig::default(),
        }
    }

    /// Create a new exporter with custom configuration
    pub fn with_config(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Set the export format
    pub fn with_format(mut self, format: ExportFormat) -> Self {
        self.config.format = format;
        self
    }

    /// Enable or disable pretty printing
    pub fn with_pretty(mut self, pretty: bool) -> Self {
        self.config.pretty = pretty;
        self
    }

    /// Set base URI
    pub fn with_base_uri(mut self, base_uri: Option<String>) -> Self {
        self.config.base_uri = base_uri;
        self
    }

    /// Export RDF data to a writer
    /// The data parameter should be RDF data in N-Triples or N-Quads format
    /// which will be parsed and then serialized to the target format
    pub fn export_to_writer<W: Write + 'static>(
        &self,
        data: &str,
        writer: W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use oxirs_core::format::{JsonLdProfileSet, RdfFormat as CoreFormat, RdfSerializer};
        use oxirs_core::parser::{Parser, RdfFormat as ParserFormat};

        // Parse input data (assume N-Triples/N-Quads format)
        let input_format = if data.lines().any(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            parts.len() == 5 && parts[4] == "."
        }) {
            ParserFormat::NQuads
        } else {
            ParserFormat::NTriples
        };

        let parser = Parser::new(input_format);
        let quads = parser.parse_str_to_quads(data)?;

        // Convert our ExportFormat to oxirs_core::format::RdfFormat
        let output_format = match self.config.format {
            ExportFormat::Turtle => CoreFormat::Turtle,
            ExportFormat::NTriples => CoreFormat::NTriples,
            ExportFormat::RdfXml => CoreFormat::RdfXml,
            ExportFormat::JsonLd => CoreFormat::JsonLd {
                profile: JsonLdProfileSet::empty(),
            },
            ExportFormat::TriG => CoreFormat::TriG,
            ExportFormat::NQuads => CoreFormat::NQuads,
        };

        // Create serializer with configuration (before for_writer)
        let mut serializer = RdfSerializer::new(output_format);

        // Add common prefixes if requested
        if self.config.use_prefixes {
            serializer = serializer
                .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
                .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
                .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
                .with_prefix("owl", "http://www.w3.org/2002/07/owl#");
        }

        // Set base URI if provided
        if let Some(base) = &self.config.base_uri {
            serializer = serializer.with_base_iri(base);
        }

        // Enable pretty printing if requested
        if self.config.pretty {
            serializer = serializer.pretty();
        }

        // Now create the writer serializer
        let mut writer_serializer = serializer.for_writer(writer);

        // Serialize all quads
        for quad in &quads {
            writer_serializer.serialize_quad(quad.as_ref())?;
        }

        // Finalize serialization
        writer_serializer.finish()?;

        Ok(())
    }

    /// Export RDF data to a file
    pub fn export_to_file<P: AsRef<Path>>(
        &self,
        data: &str,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        self.export_to_writer(data, file)
    }

    /// Export RDF data to a string
    pub fn export_to_string(&self, data: &str) -> Result<String, Box<dyn std::error::Error>> {
        use oxirs_core::format::{JsonLdProfileSet, RdfFormat as CoreFormat, RdfSerializer};
        use oxirs_core::parser::{Parser, RdfFormat as ParserFormat};
        use std::io::Cursor;

        // Parse input data
        let input_format = if data.lines().any(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            parts.len() == 5 && parts[4] == "."
        }) {
            ParserFormat::NQuads
        } else {
            ParserFormat::NTriples
        };

        let parser = Parser::new(input_format);
        let quads = parser.parse_str_to_quads(data)?;

        // Convert format
        let output_format = match self.config.format {
            ExportFormat::Turtle => CoreFormat::Turtle,
            ExportFormat::NTriples => CoreFormat::NTriples,
            ExportFormat::RdfXml => CoreFormat::RdfXml,
            ExportFormat::JsonLd => CoreFormat::JsonLd {
                profile: JsonLdProfileSet::empty(),
            },
            ExportFormat::TriG => CoreFormat::TriG,
            ExportFormat::NQuads => CoreFormat::NQuads,
        };

        // Create serializer
        let mut serializer = RdfSerializer::new(output_format);
        if self.config.use_prefixes {
            serializer = serializer
                .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
                .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
                .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
                .with_prefix("owl", "http://www.w3.org/2002/07/owl#");
        }
        if let Some(base) = &self.config.base_uri {
            serializer = serializer.with_base_iri(base);
        }
        if self.config.pretty {
            serializer = serializer.pretty();
        }

        // Serialize to Cursor<Vec<u8>> which owns the buffer
        let cursor = Cursor::new(Vec::new());
        let mut writer_serializer = serializer.for_writer(cursor);
        for quad in &quads {
            writer_serializer.serialize_quad(quad.as_ref())?;
        }
        let cursor = writer_serializer.finish()?;

        // Extract the buffer from the cursor
        let buffer = cursor.into_inner();
        Ok(String::from_utf8(buffer)?)
    }

    /// Get the current configuration
    pub fn config(&self) -> &ExportConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: ExportConfig) {
        self.config = config;
    }
}

impl Default for Exporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Turtle.extension(), "ttl");
        assert_eq!(ExportFormat::NTriples.extension(), "nt");
        assert_eq!(ExportFormat::JsonLd.extension(), "jsonld");
    }

    #[test]
    fn test_export_format_mime_type() {
        assert_eq!(ExportFormat::Turtle.mime_type(), "text/turtle");
        assert_eq!(ExportFormat::JsonLd.mime_type(), "application/ld+json");
    }

    #[test]
    fn test_exporter_creation() {
        let exporter = Exporter::new();
        assert_eq!(exporter.config().format, ExportFormat::Turtle);
        assert!(exporter.config().pretty);
    }

    #[test]
    fn test_exporter_with_format() {
        let exporter = Exporter::new().with_format(ExportFormat::JsonLd);
        assert_eq!(exporter.config().format, ExportFormat::JsonLd);
    }

    #[test]
    fn test_export_to_string() {
        let exporter = Exporter::new().with_format(ExportFormat::NTriples);
        // Provide valid N-Triples data
        let test_data = "<http://example.org/subject> <http://example.org/predicate> \"object\" .";
        let result = exporter.export_to_string(test_data).unwrap();
        // Should contain the triple data (serialized)
        assert!(result.contains("http://example.org/subject"));
    }
}
