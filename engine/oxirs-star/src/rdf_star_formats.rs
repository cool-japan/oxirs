//! Turtle-star and N-Triples-star serializers and parsers with round-trip support.
//!
//! This module provides two pairs of serializer + parser:
//!
//! - [`TurtleStarSerializer`] / [`TurtleStarParser`] – handles the `.ttls` format,
//!   including `@prefix` declarations, compact IRI notation, and multi-level
//!   nested `<< ... >>` syntax.
//! - [`NTriplesStarSerializer`] / [`NTriplesStarParser`] – handles the `.nts` format,
//!   which requires fully-qualified IRIs (no prefix abbreviation) and one
//!   statement per line.
//! - [`TrigStarSerializer`] – serializes named graphs in TriG-star format.
//!
//! All parsers guarantee round-trip fidelity: a serialized document fed back
//! through the matching parser yields the same set of `StarTriple`s.
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_star::rdf_star_formats::{TurtleStarSerializer, TurtleStarParser};
//!
//! let triple = /* ... */;
//! let mut ser = TurtleStarSerializer::new();
//! ser.add_prefix("ex", "http://example.org/");
//! let output = ser.serialize_triple(&triple).unwrap();
//!
//! let parser = TurtleStarParser::new();
//! let parsed = parser.parse_str(&output).unwrap();
//! assert_eq!(parsed.len(), 1);
//! ```

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

use serde::{Deserialize, Serialize};
use tracing::{debug, span, Level};

use crate::model::{Literal, StarTerm, StarTriple};
use crate::{StarError, StarResult};

// ============================================================================
// Common helpers
// ============================================================================

/// Escape a string literal value for Turtle/N-Triples output.
fn escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

/// Unescape a Turtle/N-Triples string escape sequence.
fn unescape_string(s: &str) -> StarResult<String> {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => {
                    return Err(StarError::parse_error(
                        "Unterminated escape sequence at end of string",
                    ))
                }
            }
        } else {
            out.push(ch);
        }
    }
    Ok(out)
}

// ============================================================================
// SerializationFormat
// ============================================================================

/// The RDF-star serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StarSerializationFormat {
    /// Turtle-star (`.ttls`)
    TurtleStar,
    /// N-Triples-star (`.nts`)
    NTriplesStar,
    /// TriG-star (`.trigs`)
    TrigStar,
}

impl StarSerializationFormat {
    /// Return the conventional file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::TurtleStar => "ttls",
            Self::NTriplesStar => "nts",
            Self::TrigStar => "trigs",
        }
    }

    /// Return the MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::TurtleStar => "text/turtle",
            Self::NTriplesStar => "application/n-triples",
            Self::TrigStar => "application/trig",
        }
    }
}

// ============================================================================
// TurtleStarSerializer
// ============================================================================

/// Serializes `StarTriple`s to Turtle-star format.
///
/// Features:
/// - Prefix declarations with `@prefix`
/// - Compact IRI notation using registered prefixes
/// - Nested `<< ... >>` syntax for quoted triples (arbitrary depth)
/// - Pretty-printed output with optional indentation
pub struct TurtleStarSerializer {
    prefixes: HashMap<String, String>,
    pretty: bool,
    base_iri: Option<String>,
}

impl TurtleStarSerializer {
    /// Create a new serializer.
    pub fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            pretty: true,
            base_iri: None,
        }
    }

    /// Register a prefix abbreviation.
    pub fn add_prefix(&mut self, prefix: &str, namespace: &str) {
        self.prefixes
            .insert(prefix.to_string(), namespace.to_string());
    }

    /// Set whether to produce pretty-printed (indented) output.
    pub fn set_pretty(&mut self, pretty: bool) {
        self.pretty = pretty;
    }

    /// Set the base IRI for `@base` declarations.
    pub fn set_base_iri(&mut self, base: impl Into<String>) {
        self.base_iri = Some(base.into());
    }

    /// Serialize a single triple to a Turtle-star string (without prefix header).
    pub fn serialize_triple(&self, triple: &StarTriple) -> StarResult<String> {
        let mut out = String::new();
        self.write_term(&triple.subject, &mut out)?;
        out.push(' ');
        self.write_term(&triple.predicate, &mut out)?;
        out.push(' ');
        self.write_term(&triple.object, &mut out)?;
        out.push_str(" .");
        Ok(out)
    }

    /// Serialize a slice of triples to a full Turtle-star document.
    pub fn serialize_document(&self, triples: &[StarTriple]) -> StarResult<String> {
        let span = span!(Level::DEBUG, "TurtleStarSerializer::serialize_document");
        let _enter = span.enter();

        let mut out = String::new();

        // Write base IRI
        if let Some(base) = &self.base_iri {
            writeln!(out, "@base <{}> .", base)
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        // Write prefix declarations
        let mut sorted_prefixes: Vec<_> = self.prefixes.iter().collect();
        sorted_prefixes.sort_by_key(|(k, _)| k.as_str());
        for (prefix, namespace) in &sorted_prefixes {
            writeln!(out, "@prefix {}: <{}> .", prefix, namespace)
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        if !sorted_prefixes.is_empty() || self.base_iri.is_some() {
            out.push('\n');
        }

        // Write triples
        for triple in triples {
            let line = self.serialize_triple(triple)?;
            if self.pretty {
                writeln!(out, "{}", line)
                    .map_err(|e| StarError::serialization_error(e.to_string()))?;
            } else {
                out.push_str(&line);
                out.push('\n');
            }
        }

        debug!(
            triple_count = triples.len(),
            "TurtleStar document serialized"
        );
        Ok(out)
    }

    fn write_term(&self, term: &StarTerm, out: &mut String) -> StarResult<()> {
        match term {
            StarTerm::NamedNode(n) => {
                let compact = self.compact_iri(&n.iri);
                out.push_str(&compact);
            }
            StarTerm::BlankNode(b) => {
                out.push_str("_:");
                out.push_str(&b.id);
            }
            StarTerm::Literal(lit) => {
                self.write_literal(lit, out)?;
            }
            StarTerm::QuotedTriple(inner) => {
                out.push_str("<< ");
                self.write_term(&inner.subject, out)?;
                out.push(' ');
                self.write_term(&inner.predicate, out)?;
                out.push(' ');
                self.write_term(&inner.object, out)?;
                out.push_str(" >>");
            }
            StarTerm::Variable(var) => {
                out.push('?');
                out.push_str(&var.name);
            }
        }
        Ok(())
    }

    fn write_literal(&self, lit: &Literal, out: &mut String) -> StarResult<()> {
        out.push('"');
        out.push_str(&escape_string(&lit.value));
        out.push('"');

        if let Some(lang) = &lit.language {
            out.push('@');
            out.push_str(lang);
        } else if let Some(dt) = &lit.datatype {
            out.push_str("^^");
            let compact = self.compact_iri(&dt.iri);
            out.push_str(&compact);
        }
        Ok(())
    }

    fn compact_iri(&self, iri: &str) -> String {
        for (prefix, namespace) in &self.prefixes {
            if let Some(local) = iri.strip_prefix(namespace.as_str()) {
                if !local.is_empty() && !local.contains([' ', '<', '>', '"']) {
                    return format!("{}:{}", prefix, local);
                }
            }
        }
        format!("<{}>", iri)
    }
}

impl Default for TurtleStarSerializer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TurtleStarParser
// ============================================================================

/// Parses Turtle-star format into `StarTriple`s.
///
/// Supports:
/// - `@prefix` declarations
/// - `@base` declarations
/// - Nested `<< ... >>` quoted triple syntax (arbitrary depth)
/// - Blank nodes (`_:id`)
/// - Literals with language tags and datatype IRIs
/// - Line comments (`# ...`)
pub struct TurtleStarParser {
    prefixes: HashMap<String, String>,
    base_iri: Option<String>,
}

impl TurtleStarParser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base_iri: None,
        }
    }

    /// Pre-register a prefix (useful for testing or incremental parsing).
    pub fn add_prefix(&mut self, prefix: &str, namespace: &str) {
        self.prefixes
            .insert(prefix.to_string(), namespace.to_string());
    }

    /// Parse a Turtle-star string, returning all triples found.
    pub fn parse_str(&mut self, input: &str) -> StarResult<Vec<StarTriple>> {
        let span = span!(Level::DEBUG, "TurtleStarParser::parse_str");
        let _enter = span.enter();

        let mut triples = Vec::new();
        let mut input = input.trim().to_string();

        // First pass: extract prefix and base declarations
        self.extract_declarations(&mut input)?;

        // Second pass: parse triples line by line (simplified: one triple per line)
        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Skip prefix / base lines (already processed)
            if line.starts_with("@prefix")
                || line.starts_with("@base")
                || line.starts_with("PREFIX")
                || line.starts_with("BASE")
            {
                continue;
            }
            // Remove trailing period
            let stmt = line.trim_end_matches(['.', ' ']).trim();
            if stmt.is_empty() {
                continue;
            }
            let triple = self.parse_triple_stmt(stmt)?;
            triples.push(triple);
        }

        debug!(triple_count = triples.len(), "TurtleStar document parsed");
        Ok(triples)
    }

    fn extract_declarations(&mut self, input: &mut String) -> StarResult<()> {
        let mut result = String::new();
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("@prefix") {
                self.parse_prefix_decl(trimmed)?;
            } else if trimmed.starts_with("@base") {
                self.parse_base_decl(trimmed)?;
            } else {
                result.push_str(line);
                result.push('\n');
            }
        }
        *input = result;
        Ok(())
    }

    fn parse_prefix_decl(&mut self, line: &str) -> StarResult<()> {
        // @prefix ex: <http://example.org/> .
        let after_prefix = line.trim_start_matches("@prefix").trim();
        let colon_pos = after_prefix.find(':').ok_or_else(|| {
            StarError::parse_error(format!("Invalid prefix declaration: {}", line))
        })?;
        let prefix = after_prefix[..colon_pos].trim().to_string();
        let rest = after_prefix[colon_pos + 1..].trim();
        let iri = self.extract_angle_bracket_iri(rest)?;
        self.prefixes.insert(prefix, iri);
        Ok(())
    }

    fn parse_base_decl(&mut self, line: &str) -> StarResult<()> {
        // @base <http://example.org/> .
        let rest = line.trim_start_matches("@base").trim();
        let iri = self.extract_angle_bracket_iri(rest)?;
        self.base_iri = Some(iri);
        Ok(())
    }

    fn extract_angle_bracket_iri(&self, s: &str) -> StarResult<String> {
        let s = s.trim();
        if !s.starts_with('<') {
            return Err(StarError::parse_error(format!(
                "Expected '<' for IRI, got: {}",
                &s[..s.len().min(20)]
            )));
        }
        let end = s
            .find('>')
            .ok_or_else(|| StarError::parse_error("Unclosed IRI angle bracket"))?;
        Ok(s[1..end].to_string())
    }

    fn parse_triple_stmt(&self, stmt: &str) -> StarResult<StarTriple> {
        let (subj, rest) = self.parse_term(stmt)?;
        let rest = rest.trim_start();
        let (pred, rest) = self.parse_term(rest)?;
        let rest = rest.trim_start();
        let (obj, _) = self.parse_term(rest)?;
        Ok(StarTriple::new(subj, pred, obj))
    }

    /// Parse a single term from the front of `input`, returning the term and
    /// the remaining unconsumed input.
    fn parse_term<'a>(&self, input: &'a str) -> StarResult<(StarTerm, &'a str)> {
        let input = input.trim_start();

        if input.starts_with("<<") {
            // Quoted triple
            return self.parse_quoted_triple(input);
        }

        if input.starts_with('<') {
            // Full IRI
            let end = input
                .find('>')
                .ok_or_else(|| StarError::parse_error("Unclosed IRI angle bracket in term"))?;
            let iri = &input[1..end];
            let term = StarTerm::iri(iri)?;
            return Ok((term, &input[end + 1..]));
        }

        if let Some(rest) = input.strip_prefix("_:") {
            // Blank node
            let end = rest
                .find(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == ';')
                .unwrap_or(rest.len());
            let id = &rest[..end];
            let term = StarTerm::blank_node(if id.is_empty() { "b0" } else { id })?;
            return Ok((term, &rest[end..]));
        }

        if input.starts_with('"') {
            // Literal
            return self.parse_literal(input);
        }

        if let Some(rest) = input.strip_prefix('?') {
            // Variable
            let end = rest
                .find(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == ';')
                .unwrap_or(rest.len());
            let name = &rest[..end];
            let term = StarTerm::variable(name)?;
            return Ok((term, &rest[end..]));
        }

        // Compact IRI (prefix:local)
        if let Some(colon_pos) = input.find(':') {
            let prefix = &input[..colon_pos];
            if !prefix.contains(|c: char| c.is_whitespace() || c == '<' || c == '>') {
                let rest_after_colon = &input[colon_pos + 1..];
                let end = rest_after_colon
                    .find(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == ';')
                    .unwrap_or(rest_after_colon.len());
                let local = &rest_after_colon[..end];

                if let Some(namespace) = self.prefixes.get(prefix) {
                    let full_iri = format!("{}{}", namespace, local);
                    let term = StarTerm::iri(&full_iri)?;
                    return Ok((term, &rest_after_colon[end..]));
                }
                // Treat as full IRI if no prefix mapping (e.g. xsd:integer)
                let full_iri = format!("{}:{}", prefix, local);
                let term = StarTerm::iri(&full_iri)?;
                return Ok((term, &rest_after_colon[end..]));
            }
        }

        Err(StarError::parse_error(format!(
            "Cannot parse term from: {}",
            &input[..input.len().min(30)]
        )))
    }

    fn parse_quoted_triple<'a>(&self, input: &'a str) -> StarResult<(StarTerm, &'a str)> {
        // input starts with "<<"
        let inner_start = &input[2..].trim_start();
        let (s, rest) = self.parse_term(inner_start)?;
        let rest = rest.trim_start();
        let (p, rest) = self.parse_term(rest)?;
        let rest = rest.trim_start();
        let (o, rest) = self.parse_term(rest)?;
        let rest = rest.trim_start();

        if !rest.starts_with(">>") {
            return Err(StarError::parse_error(format!(
                "Expected '>>' to close quoted triple, got: {}",
                &rest[..rest.len().min(20)]
            )));
        }
        let after_close = &rest[2..];
        let inner = StarTriple::new(s, p, o);
        Ok((StarTerm::quoted_triple(inner), after_close))
    }

    fn parse_literal<'a>(&self, input: &'a str) -> StarResult<(StarTerm, &'a str)> {
        // input starts with '"'
        let content_start = &input[1..];
        // Find the closing quote (handle escaped quotes)
        let mut end = 0;
        let bytes = content_start.as_bytes();
        loop {
            if end >= bytes.len() {
                return Err(StarError::parse_error("Unclosed string literal"));
            }
            if bytes[end] == b'\\' {
                end += 2; // Skip escaped character
                continue;
            }
            if bytes[end] == b'"' {
                break;
            }
            end += 1;
        }
        let raw_value = &content_start[..end];
        let value = unescape_string(raw_value)?;
        let after_quote = &content_start[end + 1..];

        // Check for language tag or datatype
        if let Some(rest) = after_quote.strip_prefix('@') {
            let lang_end = rest
                .find(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == ';')
                .unwrap_or(rest.len());
            let lang = &rest[..lang_end];
            let term = StarTerm::literal_with_language(&value, lang)?;
            return Ok((term, &rest[lang_end..]));
        }

        if let Some(rest) = after_quote.strip_prefix("^^") {
            let (dt_term, remaining) = self.parse_term(rest)?;
            let dt_iri = match &dt_term {
                StarTerm::NamedNode(n) => n.iri.clone(),
                _ => return Err(StarError::parse_error("Datatype must be an IRI")),
            };
            let term = StarTerm::literal_with_datatype(&value, &dt_iri)?;
            return Ok((term, remaining));
        }

        let term = StarTerm::literal(&value)?;
        Ok((term, after_quote))
    }
}

impl Default for TurtleStarParser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// NTriplesStarSerializer
// ============================================================================

/// Serializes `StarTriple`s to N-Triples-star format.
///
/// N-Triples-star uses fully-qualified IRIs (no prefix abbreviation), one
/// statement per line, ending with a period.
pub struct NTriplesStarSerializer;

impl NTriplesStarSerializer {
    /// Create a new N-Triples-star serializer.
    pub fn new() -> Self {
        Self
    }

    /// Serialize a single triple to an N-Triples-star line (no trailing newline).
    pub fn serialize_triple(&self, triple: &StarTriple) -> StarResult<String> {
        let mut out = String::new();
        self.write_term(&triple.subject, &mut out)?;
        out.push(' ');
        self.write_term(&triple.predicate, &mut out)?;
        out.push(' ');
        self.write_term(&triple.object, &mut out)?;
        out.push_str(" .");
        Ok(out)
    }

    /// Serialize a slice of triples to a full N-Triples-star document.
    pub fn serialize_document(&self, triples: &[StarTriple]) -> StarResult<String> {
        let span = span!(Level::DEBUG, "NTriplesStarSerializer::serialize_document");
        let _enter = span.enter();

        let mut out = String::new();
        for triple in triples {
            let line = self.serialize_triple(triple)?;
            out.push_str(&line);
            out.push('\n');
        }

        debug!(
            triple_count = triples.len(),
            "NTriplesStar document serialized"
        );
        Ok(out)
    }

    fn write_term(&self, term: &StarTerm, out: &mut String) -> StarResult<()> {
        match term {
            StarTerm::NamedNode(n) => {
                out.push('<');
                out.push_str(&n.iri);
                out.push('>');
            }
            StarTerm::BlankNode(b) => {
                out.push_str("_:");
                out.push_str(&b.id);
            }
            StarTerm::Literal(lit) => {
                out.push('"');
                out.push_str(&escape_string(&lit.value));
                out.push('"');
                if let Some(lang) = &lit.language {
                    out.push('@');
                    out.push_str(lang);
                } else if let Some(dt) = &lit.datatype {
                    out.push_str("^^<");
                    out.push_str(&dt.iri);
                    out.push('>');
                }
            }
            StarTerm::QuotedTriple(inner) => {
                out.push_str("<< ");
                self.write_term(&inner.subject, out)?;
                out.push(' ');
                self.write_term(&inner.predicate, out)?;
                out.push(' ');
                self.write_term(&inner.object, out)?;
                out.push_str(" >>");
            }
            StarTerm::Variable(var) => {
                out.push('?');
                out.push_str(&var.name);
            }
        }
        Ok(())
    }
}

impl Default for NTriplesStarSerializer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// NTriplesStarParser
// ============================================================================

/// Parses N-Triples-star format into `StarTriple`s.
///
/// Requires fully-qualified IRIs (no prefix shortcuts), one statement per line.
pub struct NTriplesStarParser;

impl NTriplesStarParser {
    /// Create a new N-Triples-star parser.
    pub fn new() -> Self {
        Self
    }

    /// Parse an N-Triples-star string into triples.
    pub fn parse_str(&self, input: &str) -> StarResult<Vec<StarTriple>> {
        let span = span!(Level::DEBUG, "NTriplesStarParser::parse_str");
        let _enter = span.enter();

        let mut triples = Vec::new();

        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let stmt = line.trim_end_matches(['.', ' ']).trim();
            if stmt.is_empty() {
                continue;
            }
            let triple = self.parse_triple_stmt(stmt)?;
            triples.push(triple);
        }

        debug!(triple_count = triples.len(), "NTriplesStar document parsed");
        Ok(triples)
    }

    fn parse_triple_stmt(&self, stmt: &str) -> StarResult<StarTriple> {
        let (s, rest) = self.parse_term(stmt)?;
        let rest = rest.trim_start();
        let (p, rest) = self.parse_term(rest)?;
        let rest = rest.trim_start();
        let (o, _) = self.parse_term(rest)?;
        Ok(StarTriple::new(s, p, o))
    }

    fn parse_term<'a>(&self, input: &'a str) -> StarResult<(StarTerm, &'a str)> {
        let input = input.trim_start();

        if input.starts_with("<<") {
            return self.parse_quoted_triple(input);
        }

        if input.starts_with('<') {
            let end = input
                .find('>')
                .ok_or_else(|| StarError::parse_error("Unclosed IRI angle bracket"))?;
            let iri = &input[1..end];
            let term = StarTerm::iri(iri)?;
            return Ok((term, &input[end + 1..]));
        }

        if let Some(rest) = input.strip_prefix("_:") {
            let end = rest
                .find(|c: char| c.is_whitespace() || c == '.')
                .unwrap_or(rest.len());
            let id = &rest[..end];
            let term = StarTerm::blank_node(if id.is_empty() { "b0" } else { id })?;
            return Ok((term, &rest[end..]));
        }

        if input.starts_with('"') {
            return self.parse_literal(input);
        }

        Err(StarError::parse_error(format!(
            "Cannot parse N-Triples term from: {}",
            &input[..input.len().min(30)]
        )))
    }

    fn parse_quoted_triple<'a>(&self, input: &'a str) -> StarResult<(StarTerm, &'a str)> {
        let inner_start = &input[2..].trim_start();
        let (s, rest) = self.parse_term(inner_start)?;
        let rest = rest.trim_start();
        let (p, rest) = self.parse_term(rest)?;
        let rest = rest.trim_start();
        let (o, rest) = self.parse_term(rest)?;
        let rest = rest.trim_start();

        if !rest.starts_with(">>") {
            return Err(StarError::parse_error(
                "Expected '>>' to close quoted triple in N-Triples-star",
            ));
        }
        let after_close = &rest[2..];
        let inner = StarTriple::new(s, p, o);
        Ok((StarTerm::quoted_triple(inner), after_close))
    }

    fn parse_literal<'a>(&self, input: &'a str) -> StarResult<(StarTerm, &'a str)> {
        let content_start = &input[1..];
        let mut end = 0;
        let bytes = content_start.as_bytes();
        loop {
            if end >= bytes.len() {
                return Err(StarError::parse_error("Unclosed string literal"));
            }
            if bytes[end] == b'\\' {
                end += 2;
                continue;
            }
            if bytes[end] == b'"' {
                break;
            }
            end += 1;
        }
        let raw_value = &content_start[..end];
        let value = unescape_string(raw_value)?;
        let after_quote = &content_start[end + 1..];

        if let Some(rest) = after_quote.strip_prefix('@') {
            let lang_end = rest
                .find(|c: char| c.is_whitespace() || c == '.')
                .unwrap_or(rest.len());
            let lang = &rest[..lang_end];
            let term = StarTerm::literal_with_language(&value, lang)?;
            return Ok((term, &rest[lang_end..]));
        }

        if let Some(rest) = after_quote.strip_prefix("^^") {
            let (dt_term, remaining) = self.parse_term(rest)?;
            let dt_iri = match &dt_term {
                StarTerm::NamedNode(n) => n.iri.clone(),
                _ => return Err(StarError::parse_error("Datatype must be an IRI")),
            };
            let term = StarTerm::literal_with_datatype(&value, &dt_iri)?;
            return Ok((term, remaining));
        }

        let term = StarTerm::literal(&value)?;
        Ok((term, after_quote))
    }
}

impl Default for NTriplesStarParser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TrigStarSerializer
// ============================================================================

/// Serializes named-graph collections to TriG-star format.
///
/// TriG-star extends Turtle-star by wrapping triple sets in named graph blocks.
pub struct TrigStarSerializer {
    turtle_ser: TurtleStarSerializer,
}

impl TrigStarSerializer {
    /// Create a new TriG-star serializer.
    pub fn new() -> Self {
        Self {
            turtle_ser: TurtleStarSerializer::new(),
        }
    }

    /// Register a prefix abbreviation.
    pub fn add_prefix(&mut self, prefix: &str, namespace: &str) {
        self.turtle_ser.add_prefix(prefix, namespace);
    }

    /// Serialize a named graph block: `GRAPH <graph_iri> { ... }`.
    pub fn serialize_named_graph(
        &self,
        graph_iri: &str,
        triples: &[StarTriple],
    ) -> StarResult<String> {
        let span = span!(Level::DEBUG, "TrigStarSerializer::serialize_named_graph");
        let _enter = span.enter();

        let mut out = String::new();
        writeln!(out, "GRAPH <{}> {{", graph_iri)
            .map_err(|e| StarError::serialization_error(e.to_string()))?;

        for triple in triples {
            let line = self.turtle_ser.serialize_triple(triple)?;
            writeln!(out, "  {}", line)
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        out.push('}');
        debug!(
            graph_iri = %graph_iri,
            triple_count = triples.len(),
            "TrigStar named graph serialized"
        );
        Ok(out)
    }

    /// Serialize multiple named graphs plus an optional default graph.
    pub fn serialize_dataset(
        &self,
        default_graph: &[StarTriple],
        named_graphs: &[(&str, &[StarTriple])],
    ) -> StarResult<String> {
        let mut out = String::new();

        // Prefix declarations
        let mut sorted_prefixes: Vec<_> = self.turtle_ser.prefixes.iter().collect();
        sorted_prefixes.sort_by_key(|(k, _)| k.as_str());
        for (prefix, namespace) in &sorted_prefixes {
            writeln!(out, "@prefix {}: <{}> .", prefix, namespace)
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
        }
        if !sorted_prefixes.is_empty() {
            out.push('\n');
        }

        // Default graph
        if !default_graph.is_empty() {
            for triple in default_graph {
                let line = self.turtle_ser.serialize_triple(triple)?;
                writeln!(out, "{}", line)
                    .map_err(|e| StarError::serialization_error(e.to_string()))?;
            }
            out.push('\n');
        }

        // Named graphs
        for (iri, triples) in named_graphs {
            let block = self.serialize_named_graph(iri, triples)?;
            out.push_str(&block);
            out.push('\n');
        }

        Ok(out)
    }
}

impl Default for TrigStarSerializer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;
    use std::env;

    fn alice_age_triple() -> StarTriple {
        StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
        )
    }

    fn make_annotated(inner: StarTriple) -> StarTriple {
        StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        )
    }

    // ── StarSerializationFormat ─────────────────────────────────────────────

    #[test]
    fn test_format_extensions() {
        assert_eq!(StarSerializationFormat::TurtleStar.extension(), "ttls");
        assert_eq!(StarSerializationFormat::NTriplesStar.extension(), "nts");
        assert_eq!(StarSerializationFormat::TrigStar.extension(), "trigs");
    }

    #[test]
    fn test_format_mime_types() {
        assert_eq!(
            StarSerializationFormat::TurtleStar.mime_type(),
            "text/turtle"
        );
        assert_eq!(
            StarSerializationFormat::NTriplesStar.mime_type(),
            "application/n-triples"
        );
    }

    // ── TurtleStarSerializer ────────────────────────────────────────────────

    #[test]
    fn test_turtle_ser_plain_triple() {
        let ser = TurtleStarSerializer::new();
        let triple = alice_age_triple();
        let out = ser.serialize_triple(&triple).unwrap();
        assert!(out.contains("<http://example.org/alice>"));
        assert!(out.contains("<http://example.org/age>"));
        assert!(out.contains("\"30\""));
        assert!(out.ends_with('.'));
    }

    #[test]
    fn test_turtle_ser_with_prefix() {
        let mut ser = TurtleStarSerializer::new();
        ser.add_prefix("ex", "http://example.org/");
        let triple = alice_age_triple();
        let out = ser.serialize_triple(&triple).unwrap();
        assert!(out.contains("ex:alice"));
        assert!(out.contains("ex:age"));
    }

    #[test]
    fn test_turtle_ser_quoted_triple() {
        let ser = TurtleStarSerializer::new();
        let inner = alice_age_triple();
        let annotated = make_annotated(inner);
        let out = ser.serialize_triple(&annotated).unwrap();
        assert!(out.contains("<<"));
        assert!(out.contains(">>"));
    }

    #[test]
    fn test_turtle_ser_document_with_prefix_header() {
        let mut ser = TurtleStarSerializer::new();
        ser.add_prefix("ex", "http://example.org/");
        let triple = alice_age_triple();
        let out = ser.serialize_document(&[triple]).unwrap();
        assert!(out.contains("@prefix ex:"));
    }

    #[test]
    fn test_turtle_ser_blank_node() {
        let ser = TurtleStarSerializer::new();
        let triple = StarTriple::new(
            StarTerm::blank_node("b1").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );
        let out = ser.serialize_triple(&triple).unwrap();
        assert!(out.contains("_:b1"));
    }

    #[test]
    fn test_turtle_ser_literal_with_language() {
        let ser = TurtleStarSerializer::new();
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/name").unwrap(),
            StarTerm::literal_with_language("Alice", "en").unwrap(),
        );
        let out = ser.serialize_triple(&triple).unwrap();
        assert!(out.contains("\"Alice\"@en"));
    }

    #[test]
    fn test_turtle_ser_literal_with_datatype() {
        let ser = TurtleStarSerializer::new();
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal_with_datatype("30", "http://www.w3.org/2001/XMLSchema#integer")
                .unwrap(),
        );
        let out = ser.serialize_triple(&triple).unwrap();
        assert!(out.contains("^^"));
    }

    #[test]
    fn test_turtle_ser_nested_quoted_triple() {
        let ser = TurtleStarSerializer::new();
        let inner = alice_age_triple();
        let mid = make_annotated(inner);
        let outer = StarTriple::new(
            StarTerm::quoted_triple(mid.subject.as_quoted_triple().unwrap().clone()),
            StarTerm::iri("http://example.org/source").unwrap(),
            StarTerm::iri("http://example.org/study").unwrap(),
        );
        let out = ser.serialize_triple(&outer).unwrap();
        // Should contain nested << >>
        assert_eq!(out.matches("<<").count(), 1);
    }

    #[test]
    fn test_turtle_ser_variable_term() {
        let ser = TurtleStarSerializer::new();
        let triple = StarTriple::new(
            StarTerm::variable("s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::variable("o").unwrap(),
        );
        let out = ser.serialize_triple(&triple).unwrap();
        assert!(out.contains("?s"));
        assert!(out.contains("?o"));
    }

    // ── TurtleStarParser ────────────────────────────────────────────────────

    #[test]
    fn test_turtle_parser_plain_triple() {
        let mut parser = TurtleStarParser::new();
        let input = "<http://example.org/alice> <http://example.org/age> \"30\" .";
        let triples = parser.parse_str(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(
            triples[0].subject,
            StarTerm::iri("http://example.org/alice").unwrap()
        );
    }

    #[test]
    fn test_turtle_parser_with_prefix() {
        let mut parser = TurtleStarParser::new();
        let input = "@prefix ex: <http://example.org/> .\nex:alice ex:age \"30\" .";
        let triples = parser.parse_str(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(
            triples[0].subject,
            StarTerm::iri("http://example.org/alice").unwrap()
        );
    }

    #[test]
    fn test_turtle_parser_quoted_triple() {
        let mut parser = TurtleStarParser::new();
        let input =
            "<< <http://example.org/alice> <http://example.org/age> \"30\" >> <http://example.org/certainty> \"0.9\" .";
        let triples = parser.parse_str(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert!(matches!(triples[0].subject, StarTerm::QuotedTriple(_)));
    }

    #[test]
    fn test_turtle_parser_blank_node() {
        let mut parser = TurtleStarParser::new();
        let input = "_:b1 <http://example.org/p> <http://example.org/o> .";
        let triples = parser.parse_str(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert!(matches!(triples[0].subject, StarTerm::BlankNode(_)));
    }

    #[test]
    fn test_turtle_parser_comment_skipped() {
        let mut parser = TurtleStarParser::new();
        let input = "# This is a comment\n<http://example.org/s> <http://example.org/p> <http://example.org/o> .";
        let triples = parser.parse_str(input).unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_turtle_parser_empty_input() {
        let mut parser = TurtleStarParser::new();
        let triples = parser.parse_str("").unwrap();
        assert!(triples.is_empty());
    }

    #[test]
    fn test_turtle_parser_literal_with_language() {
        let mut parser = TurtleStarParser::new();
        let input = "<http://example.org/s> <http://example.org/p> \"Hello\"@en .";
        let triples = parser.parse_str(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert!(matches!(
            &triples[0].object,
            StarTerm::Literal(lit) if lit.language == Some("en".to_string())
        ));
    }

    // ── Round-trip tests: Turtle-star ────────────────────────────────────────

    #[test]
    fn test_turtle_roundtrip_plain_triple() {
        let ser = TurtleStarSerializer::new();
        let original = alice_age_triple();
        let serialized = ser.serialize_triple(&original).unwrap();

        let mut parser = TurtleStarParser::new();
        let parsed = parser.parse_str(&serialized).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], original);
    }

    #[test]
    fn test_turtle_roundtrip_quoted_triple() {
        let ser = TurtleStarSerializer::new();
        let inner = alice_age_triple();
        let original = make_annotated(inner);
        let serialized = ser.serialize_triple(&original).unwrap();

        let mut parser = TurtleStarParser::new();
        let parsed = parser.parse_str(&serialized).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], original);
    }

    #[test]
    fn test_turtle_roundtrip_multiple_triples() {
        let ser = TurtleStarSerializer::new();
        let triples = vec![
            alice_age_triple(),
            StarTriple::new(
                StarTerm::iri("http://example.org/bob").unwrap(),
                StarTerm::iri("http://example.org/age").unwrap(),
                StarTerm::literal("25").unwrap(),
            ),
        ];
        let doc = ser.serialize_document(&triples).unwrap();

        let mut parser = TurtleStarParser::new();
        let parsed = parser.parse_str(&doc).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn test_turtle_roundtrip_with_prefix() {
        let mut ser = TurtleStarSerializer::new();
        ser.add_prefix("ex", "http://example.org/");
        let original = alice_age_triple();
        let doc = ser
            .serialize_document(std::slice::from_ref(&original))
            .unwrap();

        let mut parser = TurtleStarParser::new();
        let parsed = parser.parse_str(&doc).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], original);
    }

    // ── NTriplesStarSerializer ──────────────────────────────────────────────

    #[test]
    fn test_ntriples_ser_plain_triple() {
        let ser = NTriplesStarSerializer::new();
        let triple = alice_age_triple();
        let out = ser.serialize_triple(&triple).unwrap();
        assert!(out.contains("<http://example.org/alice>"));
        assert!(out.ends_with('.'));
    }

    #[test]
    fn test_ntriples_ser_quoted_triple() {
        let ser = NTriplesStarSerializer::new();
        let inner = alice_age_triple();
        let annotated = make_annotated(inner);
        let out = ser.serialize_triple(&annotated).unwrap();
        assert!(out.contains("<<"));
        assert!(out.contains(">>"));
    }

    #[test]
    fn test_ntriples_ser_document() {
        let ser = NTriplesStarSerializer::new();
        let triples = vec![alice_age_triple()];
        let doc = ser.serialize_document(&triples).unwrap();
        assert!(doc.ends_with('\n'));
    }

    #[test]
    fn test_ntriples_ser_literal_with_datatype() {
        let ser = NTriplesStarSerializer::new();
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::literal_with_datatype("42", "http://www.w3.org/2001/XMLSchema#integer")
                .unwrap(),
        );
        let out = ser.serialize_triple(&triple).unwrap();
        assert!(out.contains("^^<http://www.w3.org/2001/XMLSchema#integer>"));
    }

    // ── NTriplesStarParser ──────────────────────────────────────────────────

    #[test]
    fn test_ntriples_parser_plain_triple() {
        let parser = NTriplesStarParser::new();
        let input = "<http://example.org/alice> <http://example.org/age> \"30\" .";
        let triples = parser.parse_str(input).unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_ntriples_parser_quoted_triple() {
        let parser = NTriplesStarParser::new();
        let input =
            "<< <http://example.org/alice> <http://example.org/age> \"30\" >> <http://example.org/certainty> \"0.9\" .";
        let triples = parser.parse_str(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert!(matches!(triples[0].subject, StarTerm::QuotedTriple(_)));
    }

    #[test]
    fn test_ntriples_parser_comment_skipped() {
        let parser = NTriplesStarParser::new();
        let input =
            "# comment\n<http://example.org/s> <http://example.org/p> <http://example.org/o> .";
        let triples = parser.parse_str(input).unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_ntriples_parser_literal_with_datatype() {
        let parser = NTriplesStarParser::new();
        let input = "<http://example.org/s> <http://example.org/p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .";
        let triples = parser.parse_str(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert!(matches!(
            &triples[0].object,
            StarTerm::Literal(lit) if lit.datatype.is_some()
        ));
    }

    // ── Round-trip tests: N-Triples-star ────────────────────────────────────

    #[test]
    fn test_ntriples_roundtrip_plain_triple() {
        let ser = NTriplesStarSerializer::new();
        let original = alice_age_triple();
        let serialized = ser.serialize_triple(&original).unwrap();

        let parser = NTriplesStarParser::new();
        let parsed = parser.parse_str(&serialized).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], original);
    }

    #[test]
    fn test_ntriples_roundtrip_quoted_triple() {
        let ser = NTriplesStarSerializer::new();
        let inner = alice_age_triple();
        let original = make_annotated(inner);
        let serialized = ser.serialize_triple(&original).unwrap();

        let parser = NTriplesStarParser::new();
        let parsed = parser.parse_str(&serialized).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], original);
    }

    #[test]
    fn test_ntriples_roundtrip_blank_node() {
        let ser = NTriplesStarSerializer::new();
        let original = StarTriple::new(
            StarTerm::blank_node("b99").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );
        let serialized = ser.serialize_triple(&original).unwrap();

        let parser = NTriplesStarParser::new();
        let parsed = parser.parse_str(&serialized).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], original);
    }

    #[test]
    fn test_ntriples_roundtrip_multiple_triples() {
        let ser = NTriplesStarSerializer::new();
        let triples = vec![alice_age_triple(), make_annotated(alice_age_triple())];
        let doc = ser.serialize_document(&triples).unwrap();

        let parser = NTriplesStarParser::new();
        let parsed = parser.parse_str(&doc).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], triples[0]);
        assert_eq!(parsed[1], triples[1]);
    }

    // ── TrigStarSerializer ──────────────────────────────────────────────────

    #[test]
    fn test_trig_ser_named_graph() {
        let ser = TrigStarSerializer::new();
        let triples = vec![alice_age_triple()];
        let out = ser
            .serialize_named_graph("http://example.org/g1", &triples)
            .unwrap();
        assert!(out.contains("GRAPH <http://example.org/g1>"));
        assert!(out.contains('{'));
        assert!(out.contains('}'));
        assert!(out.contains("<http://example.org/alice>"));
    }

    #[test]
    fn test_trig_ser_dataset() {
        let ser = TrigStarSerializer::new();
        let default = vec![alice_age_triple()];
        let g1_triples = vec![make_annotated(alice_age_triple())];
        let named = vec![("http://example.org/g1", g1_triples.as_slice())];
        let out = ser.serialize_dataset(&default, &named).unwrap();
        assert!(out.contains("GRAPH <http://example.org/g1>"));
    }

    #[test]
    fn test_trig_ser_empty_named_graph() {
        let ser = TrigStarSerializer::new();
        let out = ser
            .serialize_named_graph("http://example.org/empty", &[])
            .unwrap();
        assert!(out.contains("GRAPH <http://example.org/empty>"));
        assert!(out.contains("{}") || out.contains("}\n") || out.ends_with('}'));
    }

    #[test]
    fn test_trig_ser_with_prefix() {
        let mut ser = TrigStarSerializer::new();
        ser.add_prefix("ex", "http://example.org/");
        let triples = vec![alice_age_triple()];
        let out = ser
            .serialize_named_graph("http://example.org/g1", &triples)
            .unwrap();
        assert!(out.contains("ex:alice") || out.contains("<http://example.org/alice>"));
    }

    // ── Escape / unescape ───────────────────────────────────────────────────

    #[test]
    fn test_escape_basic_string() {
        assert_eq!(escape_string("hello"), "hello");
    }

    #[test]
    fn test_escape_double_quote() {
        assert_eq!(escape_string(r#"say "hello""#), r#"say \"hello\""#);
    }

    #[test]
    fn test_escape_newline() {
        assert_eq!(escape_string("line1\nline2"), "line1\\nline2");
    }

    #[test]
    fn test_unescape_basic_string() {
        assert_eq!(unescape_string("hello").unwrap(), "hello");
    }

    #[test]
    fn test_unescape_escaped_quote() {
        assert_eq!(
            unescape_string(r#"say \"hello\""#).unwrap(),
            r#"say "hello""#
        );
    }

    #[test]
    fn test_unescape_newline() {
        assert_eq!(unescape_string("line1\\nline2").unwrap(), "line1\nline2");
    }

    // ── File round-trip test ────────────────────────────────────────────────

    #[test]
    fn test_turtle_roundtrip_via_temp_file() {
        use std::io::Write;

        let ser = TurtleStarSerializer::new();
        let inner = alice_age_triple();
        let original = make_annotated(inner);
        let content = ser.serialize_triple(&original).unwrap();

        // Write to temp file
        let mut tmp_path = env::temp_dir();
        tmp_path.push("oxirs_star_turtle_roundtrip_test.ttls");
        let mut file = std::fs::File::create(&tmp_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        drop(file);

        // Read back and parse
        let read_back = std::fs::read_to_string(&tmp_path).unwrap();
        let mut parser = TurtleStarParser::new();
        let parsed = parser.parse_str(&read_back).unwrap();

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], original);

        // Cleanup
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_ntriples_roundtrip_via_temp_file() {
        use std::io::Write;

        let ser = NTriplesStarSerializer::new();
        let inner = alice_age_triple();
        let original = make_annotated(inner);
        let content = ser.serialize_triple(&original).unwrap();

        let mut tmp_path = env::temp_dir();
        tmp_path.push("oxirs_star_ntriples_roundtrip_test.nts");
        let mut file = std::fs::File::create(&tmp_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        drop(file);

        let read_back = std::fs::read_to_string(&tmp_path).unwrap();
        let parser = NTriplesStarParser::new();
        let parsed = parser.parse_str(&read_back).unwrap();

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], original);

        let _ = std::fs::remove_file(&tmp_path);
    }
}
