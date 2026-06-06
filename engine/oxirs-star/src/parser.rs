//! RDF-star parsing implementations for various formats.
//!
//! This module provides parsers for RDF-star formats including:
//! - Turtle-star (*.ttls)
//! - N-Triples-star (*.nts)
//! - TriG-star (*.trigs)
//! - N-Quads-star (*.nqs)
//! - JSON-LD-star (*.jlds)
//!
//! The implementation is split across sibling modules for maintainability:
//! - [`crate::parser_lexer`]: tokenization, term parsing, and pattern primitives.
//! - [`crate::parser_statements`]: Turtle-star and N-Triples-star statement parsing.
//! - [`crate::parser_rdfstar`]: TriG-star, N-Quads-star, and JSON-LD-star parsing.
//!
//! This file hosts the [`StarParser`] type, shared submodule wiring, and the
//! public-facing top-level entry points.

// Internal parser submodules (visible to sibling files)
#[path = "parser/context.rs"]
pub(crate) mod context;
#[path = "parser/jsonld.rs"]
pub(crate) mod jsonld;
#[path = "parser/simd_scanner.rs"]
pub(crate) mod simd_scanner;
#[path = "parser/tokenizer.rs"]
pub(crate) mod tokenizer;

use std::io::Read;

use tracing::{span, Level};

use crate::model::StarGraph;
use crate::{StarConfig, StarError, StarResult};

// Import parser context types
use context::ParseContext;
use simd_scanner::SimdScanner;

// Re-export public error types
pub use context::{ErrorSeverity as PublicErrorSeverity, ParseError as PublicParseError};

// Re-export StarFormat from the ast sibling module
pub use crate::parser_ast::StarFormat;

/// RDF-star parser with support for multiple formats
pub struct StarParser {
    pub(crate) config: StarConfig,
    pub(crate) simd_scanner: SimdScanner,
}

impl StarParser {
    /// Create a new parser with default configuration
    pub fn new() -> Self {
        Self {
            config: StarConfig::default(),
            simd_scanner: SimdScanner::new(),
        }
    }

    /// Create a new parser with custom configuration
    pub fn with_config(config: StarConfig) -> Self {
        Self {
            config,
            simd_scanner: SimdScanner::new(),
        }
    }

    /// Set strict mode for parsing
    pub fn set_strict_mode(&mut self, strict: bool) {
        self.config.strict_mode = strict;
    }

    /// Set error recovery mode
    pub fn set_error_recovery(&mut self, _enabled: bool) {
        // This is a no-op for now since error recovery is handled per-parse
        // but we keep the method for API compatibility
    }

    /// Get parsing errors (returns empty for stateless parser)
    pub fn get_errors(&self) -> Vec<StarError> {
        // The current parser implementation doesn't store errors
        // since they're returned through the Result type
        Vec::new()
    }

    /// Access the parser configuration (used by sibling modules)
    pub(crate) fn config(&self) -> &StarConfig {
        &self.config
    }

    /// Access the SIMD scanner (used by sibling modules)
    pub(crate) fn simd_scanner(&self) -> &SimdScanner {
        &self.simd_scanner
    }

    /// Get parsing context configured for this parser
    pub(crate) fn create_parse_context(&self) -> ParseContext {
        ParseContext::with_config(self.config.strict_mode, true) // Enable error recovery by default
    }

    /// Parse RDF-star data from a reader
    pub fn parse<R: Read>(&self, reader: R, format: StarFormat) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "parse_rdf_star", format = ?format);
        let _enter = span.enter();

        match format {
            StarFormat::TurtleStar => self.parse_turtle_star(reader),
            StarFormat::NTriplesStar => self.parse_ntriples_star(reader),
            StarFormat::TrigStar => self.parse_trig_star(reader),
            StarFormat::NQuadsStar => self.parse_nquads_star(reader),
            StarFormat::JsonLdStar => self.parse_jsonld_star(reader),
        }
    }

    /// Parse RDF-star from string
    pub fn parse_str(&self, data: &str, format: StarFormat) -> StarResult<StarGraph> {
        self.parse(data.as_bytes(), format)
    }
}

impl Default for StarParser {
    fn default() -> Self {
        Self::new()
    }
}

// Tests live in `crate::parser_inline_tests` and `crate::parser_tests`.
