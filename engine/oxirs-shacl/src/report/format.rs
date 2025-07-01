//\! Report output formats and related functionality

use serde::{Deserialize, Serialize};
use std::fmt;

/// Report output formats supported by the SHACL validator
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    /// Turtle/TTL format
    Turtle,
    /// JSON-LD format
    JsonLd,
    /// RDF/XML format
    RdfXml,
    /// N-Triples format
    NTriples,
    /// JSON format (non-RDF)
    Json,
    /// HTML format with styling
    Html,
    /// CSV format for tabular data
    Csv,
    /// Plain text summary
    Text,
    /// YAML format
    Yaml,
}

impl fmt::Display for ReportFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReportFormat::Turtle => write!(f, "turtle"),
            ReportFormat::JsonLd => write!(f, "jsonld"),
            ReportFormat::RdfXml => write!(f, "rdfxml"),
            ReportFormat::NTriples => write!(f, "ntriples"),
            ReportFormat::Json => write!(f, "json"),
            ReportFormat::Html => write!(f, "html"),
            ReportFormat::Csv => write!(f, "csv"),
            ReportFormat::Text => write!(f, "text"),
            ReportFormat::Yaml => write!(f, "yaml"),
        }
    }
}

impl ReportFormat {
    /// Get the file extension for this format
    pub fn file_extension(&self) -> &'static str {
        match self {
            ReportFormat::Turtle => "ttl",
            ReportFormat::JsonLd => "jsonld",
            ReportFormat::RdfXml => "rdf",
            ReportFormat::NTriples => "nt",
            ReportFormat::Json => "json",
            ReportFormat::Html => "html",
            ReportFormat::Csv => "csv",
            ReportFormat::Text => "txt",
            ReportFormat::Yaml => "yaml",
        }
    }

    /// Get the MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ReportFormat::Turtle => "text/turtle",
            ReportFormat::JsonLd => "application/ld+json",
            ReportFormat::RdfXml => "application/rdf+xml",
            ReportFormat::NTriples => "application/n-triples",
            ReportFormat::Json => "application/json",
            ReportFormat::Html => "text/html",
            ReportFormat::Csv => "text/csv",
            ReportFormat::Text => "text/plain",
            ReportFormat::Yaml => "application/yaml",
        }
    }

    /// Parse format from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "turtle" | "ttl" => Some(ReportFormat::Turtle),
            "jsonld" | "json-ld" => Some(ReportFormat::JsonLd),
            "rdfxml" | "rdf-xml" | "xml" => Some(ReportFormat::RdfXml),
            "ntriples" | "nt" => Some(ReportFormat::NTriples),
            "json" => Some(ReportFormat::Json),
            "html" => Some(ReportFormat::Html),
            "csv" => Some(ReportFormat::Csv),
            "text" | "txt" => Some(ReportFormat::Text),
            "yaml" | "yml" => Some(ReportFormat::Yaml),
            _ => None,
        }
    }

    /// Get all supported formats
    pub fn all_formats() -> Vec<Self> {
        vec![
            ReportFormat::Turtle,
            ReportFormat::JsonLd,
            ReportFormat::RdfXml,
            ReportFormat::NTriples,
            ReportFormat::Json,
            ReportFormat::Html,
            ReportFormat::Csv,
            ReportFormat::Text,
            ReportFormat::Yaml,
        ]
    }

    /// Check if this is an RDF format
    pub fn is_rdf_format(&self) -> bool {
        matches!(
            self,
            ReportFormat::Turtle
                | ReportFormat::JsonLd
                | ReportFormat::RdfXml
                | ReportFormat::NTriples
        )
    }

    /// Check if this is a structured data format
    pub fn is_structured(&self) -> bool {
        matches!(
            self,
            ReportFormat::Json
                | ReportFormat::JsonLd
                | ReportFormat::Yaml
                | ReportFormat::RdfXml
                | ReportFormat::Turtle
                | ReportFormat::NTriples
        )
    }

    /// Check if this format supports styling/formatting
    pub fn supports_styling(&self) -> bool {
        matches!(self, ReportFormat::Html)
    }
}

impl Default for ReportFormat {
    fn default() -> Self {
        ReportFormat::Json
    }
}

/// Configuration for report formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    /// Output format
    pub format: ReportFormat,
    /// Include detailed violation information
    pub include_details: bool,
    /// Include summary statistics
    pub include_summary: bool,
    /// Include metadata
    pub include_metadata: bool,
    /// Pretty-print output (where applicable)
    pub pretty_print: bool,
    /// Include timestamps
    pub include_timestamps: bool,
    /// Maximum number of violations to include
    pub max_violations: Option<usize>,
    /// Include validation performance metrics
    pub include_performance: bool,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            format: ReportFormat::default(),
            include_details: true,
            include_summary: true,
            include_metadata: true,
            pretty_print: true,
            include_timestamps: true,
            max_violations: None,
            include_performance: false,
        }
    }
}

impl ReportConfig {
    /// Create a minimal configuration
    pub fn minimal() -> Self {
        Self {
            include_details: false,
            include_summary: false,
            include_metadata: false,
            pretty_print: false,
            include_timestamps: false,
            include_performance: false,
            ..Default::default()
        }
    }

    /// Create a detailed configuration
    pub fn detailed() -> Self {
        Self {
            include_details: true,
            include_summary: true,
            include_metadata: true,
            include_performance: true,
            ..Default::default()
        }
    }

    /// Set the output format
    pub fn with_format(mut self, format: ReportFormat) -> Self {
        self.format = format;
        self
    }

    /// Set maximum violations
    pub fn with_max_violations(mut self, max: usize) -> Self {
        self.max_violations = Some(max);
        self
    }

    /// Enable or disable pretty printing
    pub fn with_pretty_print(mut self, pretty: bool) -> Self {
        self.pretty_print = pretty;
        self
    }
}
