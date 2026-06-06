//! # Import Command Types
//!
//! Core types for the multi-format RDF importer: format enum, triple, result, and error types.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ImportFormat
// ---------------------------------------------------------------------------

/// Supported RDF input formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImportFormat {
    /// Turtle (`.ttl`)
    Turtle,
    /// N-Triples (`.nt`)
    NTriples,
    /// N-Quads (`.nq`)
    NQuads,
    /// JSON-LD (simplified, `.jsonld`)
    JsonLd,
    /// RDF/XML (simplified, `.rdf`)
    RdfXml,
    /// TriG (`.trig`)
    TriG,
    /// CSV with `subject,predicate,object[,graph]` header (`.csv`)
    Csv,
}

impl ImportFormat {
    /// Detect format from a file-extension string (case-insensitive).
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "ttl" => Some(ImportFormat::Turtle),
            "nt" => Some(ImportFormat::NTriples),
            "nq" => Some(ImportFormat::NQuads),
            "jsonld" | "json-ld" | "json" => Some(ImportFormat::JsonLd),
            "rdf" | "owl" | "xml" => Some(ImportFormat::RdfXml),
            "trig" => Some(ImportFormat::TriG),
            "csv" => Some(ImportFormat::Csv),
            _ => None,
        }
    }

    /// Detect format from a MIME-type string.
    pub fn from_mime_type(mime: &str) -> Option<Self> {
        match mime.to_lowercase().split(';').next().unwrap_or("").trim() {
            "text/turtle" => Some(ImportFormat::Turtle),
            "application/n-triples" => Some(ImportFormat::NTriples),
            "application/n-quads" => Some(ImportFormat::NQuads),
            "application/ld+json" => Some(ImportFormat::JsonLd),
            "application/rdf+xml" => Some(ImportFormat::RdfXml),
            "application/trig" => Some(ImportFormat::TriG),
            "text/csv" => Some(ImportFormat::Csv),
            _ => None,
        }
    }

    /// Canonical file extension (without leading `.`).
    pub fn extension(&self) -> &'static str {
        match self {
            ImportFormat::Turtle => "ttl",
            ImportFormat::NTriples => "nt",
            ImportFormat::NQuads => "nq",
            ImportFormat::JsonLd => "jsonld",
            ImportFormat::RdfXml => "rdf",
            ImportFormat::TriG => "trig",
            ImportFormat::Csv => "csv",
        }
    }

    /// MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            ImportFormat::Turtle => "text/turtle",
            ImportFormat::NTriples => "application/n-triples",
            ImportFormat::NQuads => "application/n-quads",
            ImportFormat::JsonLd => "application/ld+json",
            ImportFormat::RdfXml => "application/rdf+xml",
            ImportFormat::TriG => "application/trig",
            ImportFormat::Csv => "text/csv",
        }
    }
}

// ---------------------------------------------------------------------------
// Triple
// ---------------------------------------------------------------------------

/// An RDF triple (or quad when `graph` is `Some`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// Named-graph IRI (for N-Quads and TriG).
    pub graph: Option<String>,
}

// ---------------------------------------------------------------------------
// ImportResult
// ---------------------------------------------------------------------------

/// The result of a successful parse operation.
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Parsed triples.
    pub triples: Vec<Triple>,
    /// Prefix declarations found in the input.
    pub prefixes: HashMap<String, String>,
    /// Distinct named-graph IRIs found.
    pub graphs: Vec<String>,
    /// Non-fatal parse warnings.
    pub warnings: Vec<String>,
    /// Format that was used for parsing.
    pub format_detected: ImportFormat,
}

impl ImportResult {
    /// Number of triples parsed.
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Number of distinct named graphs.
    pub fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    /// Returns `true` if any non-fatal warnings were generated.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ImportError
// ---------------------------------------------------------------------------

/// Errors produced by `ImportCommand`.
#[derive(Debug)]
pub enum ImportError {
    /// The requested format is not supported.
    UnsupportedFormat(String),
    /// The input is syntactically invalid.
    ParseError(String),
    /// The input is empty.
    EmptyInput,
    /// A triple could not be constructed from the parsed values.
    InvalidTriple(String),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::UnsupportedFormat(s) => write!(f, "Unsupported format: {}", s),
            ImportError::ParseError(s) => write!(f, "Parse error: {}", s),
            ImportError::EmptyInput => write!(f, "Input is empty"),
            ImportError::InvalidTriple(s) => write!(f, "Invalid triple: {}", s),
        }
    }
}

impl std::error::Error for ImportError {}
