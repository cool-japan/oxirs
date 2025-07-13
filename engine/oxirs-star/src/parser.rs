//! RDF-star parsing implementations for various formats.
//!
//! This module provides parsers for RDF-star formats including:
//! - Turtle-star (*.ttls)
//! - N-Triples-star (*.nts)
//! - TriG-star (*.trigs)
//! - N-Quads-star (*.nqs)

use std::collections::HashMap;
use std::fmt;
use std::io::{BufRead, BufReader, Read};
use std::str::FromStr;

use anyhow::{Context, Result};
use tracing::{debug, span, Level};

use crate::model::{BlankNode, Literal, NamedNode, StarGraph, StarQuad, StarTerm, StarTriple};
use crate::{StarConfig, StarError, StarResult};

/// RDF-star format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StarFormat {
    /// Turtle-star format
    TurtleStar,
    /// N-Triples-star format
    NTriplesStar,
    /// TriG-star format (named graphs)
    TrigStar,
    /// N-Quads-star format
    NQuadsStar,
    /// JSON-LD-star format
    JsonLdStar,
}

/// Enhanced state tracking for TriG-star parsing
#[derive(Debug, Default)]
struct TrigParserState {
    /// Current graph context
    current_graph: Option<StarTerm>,
    /// Nesting level (for tracking braces)
    brace_depth: usize,
    /// Whether we're inside a graph block
    in_graph_block: bool,
    /// Buffer for accumulating multi-line graph names
    graph_name_buffer: String,
    /// Whether we're parsing a graph name declaration
    parsing_graph_name: bool,
}

impl TrigParserState {
    fn new() -> Self {
        Self::default()
    }

    fn reset_graph_context(&mut self) {
        self.current_graph = None;
        self.in_graph_block = false;
        self.brace_depth = 0;
        self.graph_name_buffer.clear();
        self.parsing_graph_name = false;
    }

    fn enter_graph_block(&mut self, graph: Option<StarTerm>) {
        self.current_graph = graph;
        self.in_graph_block = true;
        self.brace_depth += 1;
        self.parsing_graph_name = false;
        self.graph_name_buffer.clear();
    }

    fn exit_graph_block(&mut self) -> bool {
        if self.brace_depth > 0 {
            self.brace_depth -= 1;
            if self.brace_depth == 0 {
                self.reset_graph_context();
                return true;
            }
        }
        false
    }
}

impl FromStr for StarFormat {
    type Err = StarError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "turtle-star" | "ttls" => Ok(StarFormat::TurtleStar),
            "ntriples-star" | "nts" => Ok(StarFormat::NTriplesStar),
            "trig-star" | "trigs" => Ok(StarFormat::TrigStar),
            "nquads-star" | "nqs" => Ok(StarFormat::NQuadsStar),
            "json-ld-star" | "jsonld-star" | "jlds" => Ok(StarFormat::JsonLdStar),
            _ => Err(StarError::parse_error(format!("Unknown format: {s}"))),
        }
    }
}

/// Parser context for maintaining state during parsing
#[derive(Debug, Default)]
struct ParseContext {
    /// Namespace prefixes
    prefixes: HashMap<String, String>,
    /// Current base IRI
    base_iri: Option<String>,
    /// Current graph name (for TriG/N-Quads)
    #[allow(dead_code)]
    current_graph: Option<StarTerm>,
    /// Blank node counter
    blank_node_counter: usize,
    /// Current line number for error reporting
    line_number: usize,
    /// Current column position for error reporting
    column_position: usize,
    /// Strict parsing mode (reject any malformed input)
    strict_mode: bool,
    /// Error recovery mode (try to continue parsing after errors)
    error_recovery: bool,
    /// Accumulated parsing errors for batch reporting
    parsing_errors: Vec<ParseError>,
}

/// Enhanced error information for parser errors
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Error message
    pub message: String,
    /// Line number where error occurred
    pub line: usize,
    /// Column position where error occurred
    pub column: usize,
    /// Context around the error (nearby text)
    pub context: String,
    /// Severity level
    pub severity: ErrorSeverity,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSeverity {
    /// Warning - parsing can continue
    Warning,
    /// Error - current statement failed but parsing can continue
    Error,
    /// Fatal - parsing must stop
    Fatal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Warning => write!(f, "Warning"),
            ErrorSeverity::Error => write!(f, "Error"),
            ErrorSeverity::Fatal => write!(f, "Fatal"),
        }
    }
}

impl ParseContext {
    fn new() -> Self {
        Self::default()
    }

    /// Create context with configuration
    fn with_config(strict_mode: bool, error_recovery: bool) -> Self {
        Self {
            strict_mode,
            error_recovery,
            ..Default::default()
        }
    }

    /// Update position tracking
    fn update_position(&mut self, line: usize, column: usize) {
        self.line_number = line;
        self.column_position = column;
    }

    /// Add a parsing error with context
    fn add_error(&mut self, message: String, context: String, severity: ErrorSeverity) {
        let error = ParseError {
            message,
            line: self.line_number,
            column: self.column_position,
            context,
            severity,
        };
        self.parsing_errors.push(error);
    }

    /// Check if fatal errors occurred
    fn has_fatal_errors(&self) -> bool {
        self.parsing_errors
            .iter()
            .any(|e| e.severity == ErrorSeverity::Fatal)
    }

    /// Get all errors
    fn get_errors(&self) -> &[ParseError] {
        &self.parsing_errors
    }

    /// Clear errors
    #[allow(dead_code)]
    fn clear_errors(&mut self) {
        self.parsing_errors.clear();
    }

    /// Generate a new blank node identifier
    fn next_blank_node(&mut self) -> String {
        self.blank_node_counter += 1;
        format!("_:b{}", self.blank_node_counter)
    }

    /// Resolve a prefixed name to full IRI with enhanced error reporting
    fn resolve_prefix(&mut self, prefixed: &str) -> StarResult<String> {
        if let Some(colon_pos) = prefixed.find(':') {
            let prefix = &prefixed[..colon_pos];
            let local = &prefixed[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                Ok(format!("{namespace}{local}"))
            } else {
                let error_msg = format!("Unknown prefix: '{prefix}'");
                let context = format!("in prefixed name '{prefixed}'");

                if self.strict_mode {
                    self.add_error(error_msg.clone(), context, ErrorSeverity::Fatal);
                    Err(StarError::parse_error(error_msg))
                } else {
                    self.add_error(error_msg.clone(), context, ErrorSeverity::Warning);
                    // In non-strict mode, return the prefixed name as-is
                    Ok(prefixed.to_string())
                }
            }
        } else {
            let error_msg = format!("Invalid prefixed name: '{prefixed}'");
            self.add_error(
                error_msg.clone(),
                prefixed.to_string(),
                ErrorSeverity::Error,
            );
            Err(StarError::parse_error(error_msg))
        }
    }

    /// Resolve relative IRI against base
    fn resolve_relative(&self, iri: &str) -> String {
        if let Some(ref base) = self.base_iri {
            // Simple relative IRI resolution (not fully RFC compliant)
            if iri.starts_with('#') {
                format!("{base}{iri}")
            } else {
                iri.to_string()
            }
        } else {
            iri.to_string()
        }
    }

    /// Try to recover from parsing error
    fn try_recover_from_error(&mut self, error_context: &str) -> bool {
        if !self.error_recovery {
            return false;
        }

        // Add recovery attempt to error log
        let recovery_msg = format!("Attempting error recovery from: {error_context}");
        self.add_error(
            recovery_msg,
            error_context.to_string(),
            ErrorSeverity::Warning,
        );

        // Simple recovery strategies could be implemented here
        // For now, just indicate that recovery is possible
        true
    }
}

/// RDF-star parser with support for multiple formats
pub struct StarParser {
    config: StarConfig,
}

impl StarParser {
    /// Create a new parser with default configuration
    pub fn new() -> Self {
        Self {
            config: StarConfig::default(),
        }
    }

    /// Create a new parser with custom configuration
    pub fn with_config(config: StarConfig) -> Self {
        Self { config }
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

    /// Get parsing context configured for this parser
    fn create_parse_context(&self) -> ParseContext {
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

    /// Parse Turtle-star format with enhanced error handling
    pub fn parse_turtle_star<R: Read>(&self, reader: R) -> StarResult<StarGraph> {
        let span = span!(Level::DEBUG, "parse_turtle_star");
        let _enter = span.enter();

        let mut graph = StarGraph::new();
        let mut context = self.create_parse_context();
        let buf_reader = BufReader::new(reader);

        for (line_num, line_result) in buf_reader.lines().enumerate() {
            let line = line_result.map_err(|e| StarError::parse_error(e.to_string()))?;
            let line = line.trim();

            // Update position tracking
            context.update_position(line_num + 1, 0);

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Try to parse the line with error recovery
            match self.parse_turtle_line_safe(line, &mut context, &mut graph) {
                Ok(_) => {}
                Err(e) => {
                    let error_context = format!("line {}: {}", line_num + 1, line);
                    context.add_error(
                        format!("Parse error: {e}"),
                        error_context,
                        if context.strict_mode {
                            ErrorSeverity::Fatal
                        } else {
                            ErrorSeverity::Error
                        },
                    );

                    if context.strict_mode || context.has_fatal_errors() {
                        return Err(StarError::parse_error(format!(
                            "Parsing failed at line {} with {} errors. First error: {}",
                            line_num + 1,
                            context.get_errors().len(),
                            context
                                .get_errors()
                                .first()
                                .map(|e| &e.message)
                                .unwrap_or(&"Unknown error".to_string())
                        )));
                    }

                    // Try error recovery
                    if !context.try_recover_from_error(&format!("line {}", line_num + 1)) {
                        debug!("Error recovery failed for line {}", line_num + 1);
                    }
                }
            }
        }

        // Report any accumulated errors
        if !context.get_errors().is_empty() {
            debug!(
                "Parsing completed with {} warnings/errors",
                context.get_errors().len()
            );
            for error in context.get_errors() {
                debug!(
                    "Line {}, Column {}: {} ({})",
                    error.line, error.column, error.message, error.context
                );
            }
        }

        debug!("Parsed {} triples in Turtle-star format", graph.len());
        Ok(graph)
    }

    /// Parse a single Turtle-star line with enhanced error handling
    fn parse_turtle_line_safe(
        &self,
        line: &str,
        context: &mut ParseContext,
        graph: &mut StarGraph,
    ) -> StarResult<()> {
        // Handle directives
        if line.starts_with("@prefix") {
            return self.parse_prefix_directive_safe(line, context);
        }

        if line.starts_with("@base") {
            return self.parse_base_directive_safe(line, context);
        }

        // Parse triple
        if let Some(stripped) = line.strip_suffix('.') {
            let triple_str = stripped.trim();
            match self.parse_triple_pattern_safe(triple_str, context) {
                Ok(triple) => {
                    graph.insert(triple).map_err(|e| {
                        StarError::parse_error(format!("Failed to insert triple: {e}"))
                    })?;
                }
                Err(e) => {
                    let error_msg = format!("Failed to parse triple pattern '{triple_str}': {e}");
                    context.add_error(error_msg.clone(), line.to_string(), ErrorSeverity::Error);

                    if context.strict_mode {
                        return Err(StarError::parse_error(error_msg));
                    }
                    // In non-strict mode, log error and continue
                }
            }
        } else if !line.trim().is_empty() {
            // Non-empty line that doesn't end with '.' - potential malformed statement
            let error_msg = format!("Malformed statement (missing terminating '.'): {line}");
            context.add_error(error_msg.clone(), line.to_string(), ErrorSeverity::Warning);

            if context.strict_mode {
                return Err(StarError::parse_error(error_msg));
            }
        }

        Ok(())
    }

    /// Parse a single Turtle-star line (legacy method)
    #[allow(dead_code)]
    fn parse_turtle_line(
        &self,
        line: &str,
        context: &mut ParseContext,
        graph: &mut StarGraph,
    ) -> Result<()> {
        // Handle directives
        if line.starts_with("@prefix") {
            self.parse_prefix_directive(line, context)?;
            return Ok(());
        }

        if line.starts_with("@base") {
            self.parse_base_directive(line, context)?;
            return Ok(());
        }

        // Parse triple
        if let Some(stripped) = line.strip_suffix('.') {
            let triple_str = stripped.trim();
            let triple = self.parse_triple_pattern(triple_str, context)?;
            graph.insert(triple)?;
        }

        Ok(())
    }

    /// Parse a single TriG-star line
    #[allow(dead_code)]
    fn parse_trig_line(
        &self,
        line: &str,
        context: &mut ParseContext,
        graph: &mut StarGraph,
        current_graph: &mut Option<StarTerm>,
        in_graph_block: &mut bool,
        brace_count: &mut usize,
    ) -> Result<()> {
        // Handle directives (same as Turtle)
        if line.starts_with("@prefix") {
            self.parse_prefix_directive(line, context)?;
            return Ok(());
        }

        if line.starts_with("@base") {
            self.parse_base_directive(line, context)?;
            return Ok(());
        }

        // Handle graph declarations
        if line.contains('{') && !*in_graph_block {
            // Start of named graph block
            let graph_part = line.split('{').next().unwrap().trim();
            if !graph_part.is_empty() {
                let graph_term = self.parse_term(graph_part, context)?;
                *current_graph = Some(graph_term);
            } else {
                // Default graph
                *current_graph = None;
            }
            *in_graph_block = true;
            *brace_count = line.chars().filter(|&c| c == '{').count();
        }

        // Handle closing braces
        if line.contains('}') {
            let close_braces = line.chars().filter(|&c| c == '}').count();
            if close_braces >= *brace_count {
                *in_graph_block = false;
                *current_graph = None;
                *brace_count = 0;
            } else {
                *brace_count -= close_braces;
            }
        }

        // Parse triples within the line (excluding graph declaration part)
        let triple_part = if line.contains('{') {
            line.split('{').nth(1).unwrap_or("")
        } else {
            line
        };

        if triple_part.trim().ends_with('.') && !triple_part.trim().is_empty() {
            let triple_str = triple_part.trim();
            let triple_str = &triple_str[..triple_str.len() - 1].trim();
            if !triple_str.is_empty() {
                let triple = self.parse_triple_pattern(triple_str, context)?;
                graph.insert(triple)?;
            }
        }

        Ok(())
    }

    /// Parse @prefix directive with enhanced error handling
    fn parse_prefix_directive_safe(
        &self,
        line: &str,
        context: &mut ParseContext,
    ) -> StarResult<()> {
        // @prefix prefix: <namespace> .
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let prefix_name = parts[1].trim_end_matches(':');
            let namespace = parts[2].trim_matches(['<', '>', '.']);

            // Validate prefix format
            if prefix_name.is_empty() {
                let error_msg = "Empty prefix name in @prefix directive".to_string();
                context.add_error(error_msg.clone(), line.to_string(), ErrorSeverity::Error);
                if context.strict_mode {
                    return Err(StarError::parse_error(error_msg));
                }
            } else if namespace.is_empty() {
                let error_msg = "Empty namespace in @prefix directive".to_string();
                context.add_error(error_msg.clone(), line.to_string(), ErrorSeverity::Error);
                if context.strict_mode {
                    return Err(StarError::parse_error(error_msg));
                }
            } else {
                context
                    .prefixes
                    .insert(prefix_name.to_string(), namespace.to_string());
            }
        } else {
            let error_msg = format!("Malformed @prefix directive: {line}");
            context.add_error(error_msg.clone(), line.to_string(), ErrorSeverity::Error);
            if context.strict_mode {
                return Err(StarError::parse_error(error_msg));
            }
        }
        Ok(())
    }

    /// Parse @base directive with enhanced error handling
    fn parse_base_directive_safe(&self, line: &str, context: &mut ParseContext) -> StarResult<()> {
        // @base <iri> .
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let base_iri = parts[1].trim_matches(['<', '>', '.']);
            if base_iri.is_empty() {
                let error_msg = "Empty base IRI in @base directive".to_string();
                context.add_error(error_msg.clone(), line.to_string(), ErrorSeverity::Error);
                if context.strict_mode {
                    return Err(StarError::parse_error(error_msg));
                }
            } else {
                context.base_iri = Some(base_iri.to_string());
            }
        } else {
            let error_msg = format!("Malformed @base directive: {line}");
            context.add_error(error_msg.clone(), line.to_string(), ErrorSeverity::Error);
            if context.strict_mode {
                return Err(StarError::parse_error(error_msg));
            }
        }
        Ok(())
    }

    /// Parse triple pattern with enhanced error handling
    fn parse_triple_pattern_safe(
        &self,
        pattern: &str,
        context: &mut ParseContext,
    ) -> StarResult<StarTriple> {
        self.parse_triple_pattern_safe_with_depth(pattern, context, 0)
    }

    /// Parse term with enhanced error handling
    fn parse_term_safe(&self, term_str: &str, context: &mut ParseContext) -> StarResult<StarTerm> {
        self.parse_term_safe_with_depth(term_str, context, 0)
    }

    /// Parse term with depth tracking for quoted triples
    fn parse_term_safe_with_depth(
        &self,
        term_str: &str,
        context: &mut ParseContext,
        depth: usize,
    ) -> StarResult<StarTerm> {
        const MAX_QUOTED_TRIPLE_DEPTH: usize = 100;

        let term_str = term_str.trim();

        // Check for empty input
        if term_str.is_empty() {
            let error_msg = "Empty term in triple".to_string();
            context.add_error(
                error_msg.clone(),
                term_str.to_string(),
                ErrorSeverity::Error,
            );
            return Err(StarError::parse_error(error_msg));
        }

        // Quoted triple: << ... >>
        if term_str.starts_with("<<") && term_str.ends_with(">>") {
            // Check depth limit to prevent stack overflow
            if depth >= MAX_QUOTED_TRIPLE_DEPTH {
                let error_msg = format!(
                    "Maximum quoted triple nesting depth ({MAX_QUOTED_TRIPLE_DEPTH}) exceeded"
                );
                context.add_error(
                    error_msg.clone(),
                    term_str.to_string(),
                    ErrorSeverity::Error,
                );
                return Err(StarError::parse_error(error_msg));
            }

            let inner = &term_str[2..term_str.len() - 2].trim();

            // Check for empty quoted triple
            if inner.is_empty() {
                let error_msg = "Empty quoted triple content".to_string();
                context.add_error(
                    error_msg.clone(),
                    term_str.to_string(),
                    ErrorSeverity::Error,
                );
                if context.strict_mode {
                    return Err(StarError::parse_error(error_msg));
                }
                // In non-strict mode, create a placeholder error term
                return self
                    .parse_term("<urn:error:empty-quoted-triple>", context)
                    .map_err(|e| StarError::parse_error(e.to_string()));
            }

            // Validate proper quoted triple delimiters
            if !self.validate_quoted_triple_delimiters(term_str) {
                let error_msg = "Malformed quoted triple delimiters".to_string();
                context.add_error(
                    error_msg.clone(),
                    term_str.to_string(),
                    ErrorSeverity::Error,
                );
                if context.strict_mode {
                    return Err(StarError::parse_error(error_msg));
                }
            }

            match self.parse_triple_pattern_safe_with_depth(inner, context, depth + 1) {
                Ok(inner_triple) => return Ok(StarTerm::quoted_triple(inner_triple)),
                Err(e) => {
                    let error_msg = format!("Failed to parse quoted triple content: {e}");
                    context.add_error(
                        error_msg.clone(),
                        term_str.to_string(),
                        ErrorSeverity::Error,
                    );
                    if context.strict_mode {
                        return Err(StarError::parse_error(error_msg));
                    }
                    // In non-strict mode, try to parse as regular term
                    // but with better error context
                    return self.parse_term_fallback(term_str, context, &error_msg);
                }
            }
        }

        // Continue with other term types...
        self.parse_term(term_str, context)
            .map_err(|e| StarError::parse_error(e.to_string()))
    }

    /// Validate quoted triple delimiter structure
    fn validate_quoted_triple_delimiters(&self, term_str: &str) -> bool {
        let mut depth = 0;
        let mut chars = term_str.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                '<' if chars.peek() == Some(&'<') => {
                    chars.next(); // consume second '<'
                    depth += 1;
                }
                '>' if chars.peek() == Some(&'>') => {
                    chars.next(); // consume second '>'
                    depth -= 1;
                    if depth < 0 {
                        return false; // More closing than opening
                    }
                }
                _ => {}
            }
        }

        depth == 0 // Should be balanced
    }

    /// Parse triple pattern with depth tracking
    fn parse_triple_pattern_safe_with_depth(
        &self,
        pattern: &str,
        context: &mut ParseContext,
        depth: usize,
    ) -> StarResult<StarTriple> {
        match self.tokenize_triple(pattern) {
            Ok(terms) => {
                if terms.len() != 3 {
                    let error_msg = format!(
                        "Triple must have exactly 3 terms, found {} (depth: {})",
                        terms.len(),
                        depth
                    );
                    context.add_error(error_msg.clone(), pattern.to_string(), ErrorSeverity::Error);
                    return Err(StarError::parse_error(error_msg));
                }

                let subject = self.parse_term_safe_with_depth(&terms[0], context, depth)?;
                let predicate = self.parse_term_safe_with_depth(&terms[1], context, depth)?;
                let object = self.parse_term_safe_with_depth(&terms[2], context, depth)?;

                let triple = StarTriple::new(subject, predicate, object);

                // Validate the constructed triple
                if let Err(validation_error) = triple.validate() {
                    let error_msg = format!("Invalid triple at depth {depth}: {validation_error}");
                    context.add_error(error_msg.clone(), pattern.to_string(), ErrorSeverity::Error);
                    return Err(StarError::parse_error(error_msg));
                }

                Ok(triple)
            }
            Err(e) => {
                let error_msg = format!("Failed to tokenize triple at depth {depth}: {e}");
                context.add_error(error_msg.clone(), pattern.to_string(), ErrorSeverity::Error);
                Err(StarError::parse_error(error_msg))
            }
        }
    }

    /// Fallback parsing with better error context
    fn parse_term_fallback(
        &self,
        term_str: &str,
        context: &mut ParseContext,
        original_error: &str,
    ) -> StarResult<StarTerm> {
        match self.parse_term(term_str, context) {
            Ok(term) => {
                // Log a warning that we fell back to regular parsing
                context.add_error(
                    format!(
                        "Quoted triple parsing failed, parsed as regular term: {original_error}"
                    ),
                    term_str.to_string(),
                    ErrorSeverity::Warning,
                );
                Ok(term)
            }
            Err(fallback_error) => {
                let combined_error = format!("Both quoted triple and regular term parsing failed. Quoted triple error: {original_error}. Regular term error: {fallback_error}");
                context.add_error(
                    combined_error.clone(),
                    term_str.to_string(),
                    ErrorSeverity::Error,
                );
                Err(StarError::parse_error(combined_error))
            }
        }
    }

    /// Parse @prefix directive (legacy)
    fn parse_prefix_directive(&self, line: &str, context: &mut ParseContext) -> Result<()> {
        // @prefix prefix: <namespace> .
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let prefix_name = parts[1].trim_end_matches(':');
            let namespace = parts[2].trim_matches(['<', '>', '.']);
            context
                .prefixes
                .insert(prefix_name.to_string(), namespace.to_string());
        }
        Ok(())
    }

    /// Parse @base directive (legacy)
    fn parse_base_directive(&self, line: &str, context: &mut ParseContext) -> Result<()> {
        // @base <iri> .
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let base_iri = parts[1].trim_matches(['<', '>', '.']);
            context.base_iri = Some(base_iri.to_string());
        }
        Ok(())
    }

    /// Parse N-Triples-star format
    pub fn parse_ntriples_star<R: Read>(&self, reader: R) -> StarResult<StarGraph> {
        let span = span!(Level::DEBUG, "parse_ntriples_star");
        let _enter = span.enter();

        let mut graph = StarGraph::new();
        let mut context = ParseContext::new();
        let buf_reader = BufReader::new(reader);

        for (line_num, line_result) in buf_reader.lines().enumerate() {
            let line = line_result.map_err(|e| StarError::parse_error(e.to_string()))?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(stripped) = line.strip_suffix('.') {
                let triple_str = &stripped.trim();
                let triple = self
                    .parse_triple_pattern(triple_str, &mut context)
                    .with_context(|| format!("Error parsing line {}: {}", line_num + 1, line))
                    .map_err(|e| StarError::parse_error(e.to_string()))?;
                graph.insert(triple)?;
            } else {
                // In N-Triples-star, all non-empty lines must be valid triples ending with '.'
                return Err(StarError::parse_error(format!(
                    "Invalid N-Triples-star line {}: {}",
                    line_num + 1,
                    line
                )));
            }
        }

        debug!("Parsed {} triples in N-Triples-star format", graph.len());
        Ok(graph)
    }

    /// Parse TriG-star format (with named graphs) - Enhanced version
    pub fn parse_trig_star<R: Read>(&self, reader: R) -> StarResult<StarGraph> {
        let span = span!(Level::DEBUG, "parse_trig_star");
        let _enter = span.enter();

        let mut graph = StarGraph::new();
        let mut context = self.create_parse_context();
        let buf_reader = BufReader::new(reader);

        // Enhanced state tracking for TriG parsing
        let mut trig_state = TrigParserState::new();
        let mut accumulated_line = String::new();
        let mut error_count = 0;
        let max_errors = self.config.max_parse_errors.unwrap_or(100);

        for (line_num, line_result) in buf_reader.lines().enumerate() {
            let line = line_result.map_err(|e| StarError::parse_error(e.to_string()))?;
            let line = line.trim();

            // Update position tracking
            context.update_position(line_num + 1, 0);

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Handle multi-line statements
            accumulated_line.push_str(line);
            accumulated_line.push(' ');

            // Check if we have a complete statement
            if self.is_complete_trig_statement(&accumulated_line, &mut trig_state) {
                match self.parse_complete_trig_statement(
                    accumulated_line.trim(),
                    &mut context,
                    &mut graph,
                    &mut trig_state,
                ) {
                    Ok(_) => {
                        // Successfully parsed
                        accumulated_line.clear();
                    }
                    Err(e) => {
                        error_count += 1;
                        let error_context =
                            format!("line {}: {}", line_num + 1, accumulated_line.trim());
                        context.add_error(
                            format!("TriG-star parse error: {e}"),
                            error_context,
                            if context.strict_mode {
                                ErrorSeverity::Fatal
                            } else {
                                ErrorSeverity::Error
                            },
                        );

                        if context.strict_mode || error_count >= max_errors {
                            return Err(StarError::parse_error(format!(
                                "TriG-star parsing failed at line {} with {} errors. First error: {}",
                                line_num + 1,
                                context.get_errors().len(),
                                context.get_errors().first().map(|e| &e.message).unwrap_or(&"Unknown error".to_string())
                            )));
                        }

                        // Try to recover by resetting the accumulated line and state
                        accumulated_line.clear();

                        // If we're in a nested structure, try to recover the state
                        if trig_state.brace_depth > 0 {
                            debug!("Attempting to recover from error in nested graph block");
                            // Don't reset state completely, just try to continue
                        } else {
                            // Reset to a clean state for top-level errors
                            trig_state = TrigParserState::new();
                        }
                    }
                }
            }
        }

        // Handle any remaining incomplete statement
        if !accumulated_line.trim().is_empty() {
            context.add_error(
                "Incomplete TriG-star statement at end of input".to_string(),
                accumulated_line.trim().to_string(),
                ErrorSeverity::Error,
            );
            if context.strict_mode {
                return Err(StarError::parse_error(format!(
                    "Incomplete TriG-star statement at end of input: {}",
                    accumulated_line.trim()
                )));
            }
        }

        // Check for unclosed graph blocks
        if trig_state.in_graph_block {
            context.add_error(
                format!(
                    "Unclosed graph block at end of input. Missing {} closing brace(s)",
                    trig_state.brace_depth
                ),
                "End of file".to_string(),
                ErrorSeverity::Error,
            );
            if context.strict_mode {
                return Err(StarError::parse_error(
                    "Unclosed graph block at end of input".to_string(),
                ));
            }
        }

        // Report any accumulated errors
        if !context.get_errors().is_empty() {
            debug!(
                "TriG-star parsing completed with {} warnings/errors",
                context.get_errors().len()
            );
            for error in context.get_errors() {
                debug!(
                    "Line {}, Column {}: {} [{}] ({})",
                    error.line, error.column, error.message, error.severity, error.context
                );
            }
        }

        debug!(
            "Parsed {} quads ({} total triples) in TriG-star format with {} errors",
            graph.quad_len(),
            graph.total_len(),
            context.get_errors().len()
        );
        Ok(graph)
    }

    /// Parse N-Quads-star format with enhanced error handling and streaming support
    pub fn parse_nquads_star<R: Read>(&self, reader: R) -> StarResult<StarGraph> {
        let span = span!(Level::DEBUG, "parse_nquads_star");
        let _enter = span.enter();

        let mut graph = StarGraph::new();
        let mut context = self.create_parse_context();
        let buf_reader = BufReader::new(reader);

        for (line_num, line_result) in buf_reader.lines().enumerate() {
            let line = line_result.map_err(|e| StarError::parse_error(e.to_string()))?;
            let line = line.trim();

            // Update position tracking
            context.update_position(line_num + 1, 0);

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(stripped) = line.strip_suffix('.') {
                let quad_str = &stripped.trim();
                match self.parse_quad_pattern_safe(quad_str, &mut context) {
                    Ok(quad) => {
                        graph.insert_quad(quad)?;
                    }
                    Err(e) => {
                        let error_context = format!("line {}: {}", line_num + 1, line);
                        context.add_error(
                            format!("Parse error: {e}"),
                            error_context,
                            if context.strict_mode {
                                ErrorSeverity::Fatal
                            } else {
                                ErrorSeverity::Error
                            },
                        );

                        if context.strict_mode || context.has_fatal_errors() {
                            return Err(StarError::parse_error(format!(
                                "N-Quads-star parsing failed at line {} with {} errors. First error: {}",
                                line_num + 1,
                                context.get_errors().len(),
                                context.get_errors().first().map(|e| &e.message).unwrap_or(&"Unknown error".to_string())
                            )));
                        }

                        // Try error recovery
                        if !context.try_recover_from_error(&format!("line {}", line_num + 1)) {
                            debug!(
                                "Error recovery failed for N-Quads-star line {}",
                                line_num + 1
                            );
                        }
                    }
                }
            } else {
                // In N-Quads-star, all non-empty lines must be valid quads ending with '.'
                let error_msg = format!(
                    "Invalid N-Quads-star line {}: missing terminating '.'",
                    line_num + 1
                );
                context.add_error(error_msg.clone(), line.to_string(), ErrorSeverity::Error);

                if context.strict_mode {
                    return Err(StarError::parse_error(error_msg));
                }
            }
        }

        // Report any accumulated errors
        if !context.get_errors().is_empty() {
            debug!(
                "N-Quads-star parsing completed with {} warnings/errors",
                context.get_errors().len()
            );
            for error in context.get_errors() {
                debug!(
                    "Line {}, Column {}: {} ({})",
                    error.line, error.column, error.message, error.context
                );
            }
        }

        debug!(
            "Parsed {} quads ({} total triples) in N-Quads-star format",
            graph.quad_len(),
            graph.total_len()
        );
        Ok(graph)
    }

    /// Parse a quad pattern with enhanced error handling
    fn parse_quad_pattern_safe(
        &self,
        pattern: &str,
        context: &mut ParseContext,
    ) -> StarResult<StarQuad> {
        match self.tokenize_quad(pattern) {
            Ok(terms) => {
                if terms.len() < 3 || terms.len() > 4 {
                    let error_msg = format!("Quad must have 3 or 4 terms, found {}", terms.len());
                    context.add_error(error_msg.clone(), pattern.to_string(), ErrorSeverity::Error);
                    return Err(StarError::parse_error(error_msg));
                }

                let subject = self.parse_term_safe(&terms[0], context)?;
                let predicate = self.parse_term_safe(&terms[1], context)?;
                let object = self.parse_term_safe(&terms[2], context)?;

                // Graph is optional in N-Quads (default graph if omitted)
                let graph = if terms.len() == 4 {
                    Some(self.parse_term_safe(&terms[3], context)?)
                } else {
                    None
                };

                let quad = StarQuad::new(subject, predicate, object, graph);

                // Validate the constructed quad
                if let Err(validation_error) = quad.validate() {
                    let error_msg = format!("Invalid quad: {validation_error}");
                    context.add_error(error_msg.clone(), pattern.to_string(), ErrorSeverity::Error);
                    return Err(StarError::parse_error(error_msg));
                }

                Ok(quad)
            }
            Err(e) => {
                let error_msg = format!("Failed to tokenize quad: {e}");
                context.add_error(error_msg.clone(), pattern.to_string(), ErrorSeverity::Error);
                Err(StarError::parse_error(error_msg))
            }
        }
    }

    /// Parse a quad pattern (subject predicate object graph)
    #[allow(dead_code)]
    fn parse_quad_pattern(&self, pattern: &str, context: &mut ParseContext) -> Result<StarQuad> {
        let terms = self.tokenize_quad(pattern)?;

        if terms.len() < 3 || terms.len() > 4 {
            return Err(anyhow::anyhow!(
                "Quad must have 3 or 4 terms, found {}",
                terms.len()
            ));
        }

        let subject = self.parse_term(&terms[0], context)?;
        let predicate = self.parse_term(&terms[1], context)?;
        let object = self.parse_term(&terms[2], context)?;

        // Graph is optional in N-Quads (default graph if omitted)
        let graph = if terms.len() == 4 {
            Some(self.parse_term(&terms[3], context)?)
        } else {
            None
        };

        Ok(StarQuad {
            subject,
            predicate,
            object,
            graph,
        })
    }

    /// Parse a triple pattern (subject predicate object)
    fn parse_triple_pattern(
        &self,
        pattern: &str,
        context: &mut ParseContext,
    ) -> Result<StarTriple> {
        let terms = self.tokenize_triple(pattern)?;

        if terms.len() != 3 {
            return Err(anyhow::anyhow!(
                "Triple must have exactly 3 terms, found {}",
                terms.len()
            ));
        }

        let subject = self.parse_term(&terms[0], context)?;
        let predicate = self.parse_term(&terms[1], context)?;
        let object = self.parse_term(&terms[2], context)?;

        let triple = StarTriple::new(subject, predicate, object);
        triple
            .validate()
            .map_err(|e| anyhow::anyhow!("Invalid triple: {}", e))?;

        Ok(triple)
    }

    /// Tokenize a triple into its three components, handling quoted triples
    fn tokenize_triple(&self, pattern: &str) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut chars = pattern.chars().peekable();
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;

        while let Some(ch) = chars.next() {
            if escape_next {
                current_token.push(ch);
                escape_next = false;
                continue;
            }

            match ch {
                '\\' if in_string => {
                    escape_next = true;
                    current_token.push(ch);
                }
                '"' => {
                    in_string = !in_string;
                    current_token.push(ch);
                }
                '<' if !in_string && chars.peek() == Some(&'<') => {
                    // Start of quoted triple
                    chars.next(); // consume second '<'
                    depth += 1;
                    current_token.push_str("<<");
                }
                '>' if !in_string && chars.peek() == Some(&'>') => {
                    // End of quoted triple
                    chars.next(); // consume second '>'
                    depth -= 1;
                    current_token.push_str(">>");
                }
                ' ' | '\t' if !in_string && depth == 0 => {
                    // Whitespace at top level - end of token
                    if !current_token.trim().is_empty() {
                        tokens.push(current_token.trim().to_string());
                        current_token.clear();
                    }
                }
                _ => {
                    current_token.push(ch);
                }
            }
        }

        // Add final token
        if !current_token.trim().is_empty() {
            tokens.push(current_token.trim().to_string());
        }

        Ok(tokens)
    }

    /// Tokenize a quad pattern (similar to triple but allows 4 terms)
    fn tokenize_quad(&self, pattern: &str) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut chars = pattern.chars().peekable();
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;

        while let Some(ch) = chars.next() {
            if escape_next {
                current_token.push(ch);
                escape_next = false;
                continue;
            }

            match ch {
                '\\' if in_string => {
                    escape_next = true;
                    current_token.push(ch);
                }
                '"' => {
                    in_string = !in_string;
                    current_token.push(ch);
                }
                '<' if !in_string && chars.peek() == Some(&'<') => {
                    // Start of quoted triple
                    chars.next(); // consume second '<'
                    depth += 1;
                    current_token.push_str("<<");
                }
                '>' if !in_string && chars.peek() == Some(&'>') => {
                    // End of quoted triple
                    chars.next(); // consume second '>'
                    depth -= 1;
                    current_token.push_str(">>");
                }
                ' ' | '\t' if !in_string && depth == 0 => {
                    // Whitespace at top level - end of token
                    if !current_token.trim().is_empty() {
                        tokens.push(current_token.trim().to_string());
                        current_token.clear();
                    }
                }
                _ => {
                    current_token.push(ch);
                }
            }
        }

        // Add final token
        if !current_token.trim().is_empty() {
            tokens.push(current_token.trim().to_string());
        }

        Ok(tokens)
    }

    /// Parse a single term (IRI, blank node, literal, or quoted triple)
    fn parse_term(&self, term_str: &str, context: &mut ParseContext) -> Result<StarTerm> {
        let term_str = term_str.trim();

        // Quoted triple: << ... >>
        if term_str.starts_with("<<") && term_str.ends_with(">>") {
            let inner = &term_str[2..term_str.len() - 2];
            let inner_triple = self.parse_triple_pattern(inner, context)?;
            return Ok(StarTerm::quoted_triple(inner_triple));
        }

        // IRI: <...> or prefixed name
        if term_str.starts_with('<') && term_str.ends_with('>') {
            let iri = &term_str[1..term_str.len() - 1];
            let resolved = context.resolve_relative(iri);
            return StarTerm::iri(&resolved).map_err(|e| anyhow::anyhow!("Invalid IRI: {}", e));
        }

        // Prefixed name
        if term_str.contains(':') && !term_str.starts_with('_') && !term_str.starts_with('"') {
            let resolved = context.resolve_prefix(term_str)?;
            return StarTerm::iri(&resolved).map_err(|e| anyhow::anyhow!("Invalid IRI: {}", e));
        }

        // Blank node: _:id
        if let Some(id) = term_str.strip_prefix("_:") {
            return StarTerm::blank_node(id)
                .map_err(|e| anyhow::anyhow!("Invalid blank node: {}", e));
        }

        // Literal: "value"@lang or "value"^^<datatype>
        if term_str.starts_with('"') {
            return self.parse_literal(term_str, context);
        }

        // Variable: ?name (for SPARQL-star)
        if let Some(name) = term_str.strip_prefix('?') {
            return StarTerm::variable(name)
                .map_err(|e| anyhow::anyhow!("Invalid variable: {}", e));
        }

        Err(anyhow::anyhow!("Unrecognized term format: {}", term_str))
    }

    /// Parse a literal term with optional language tag or datatype
    fn parse_literal(&self, literal_str: &str, context: &mut ParseContext) -> Result<StarTerm> {
        let mut chars = literal_str.chars().peekable();
        let mut value = String::new();
        let mut escape_next = false;

        // Skip opening quote
        if chars.next() != Some('"') {
            return Err(anyhow::anyhow!("Literal must start with quote"));
        }
        let mut in_string = true;

        // Parse value
        for ch in chars.by_ref() {
            if escape_next {
                value.push(ch);
                escape_next = false;
                continue;
            }

            match ch {
                '\\' => {
                    escape_next = true;
                }
                '"' => {
                    in_string = false;
                    break;
                }
                _ => {
                    value.push(ch);
                }
            }
        }

        if in_string {
            return Err(anyhow::anyhow!("Unterminated string literal"));
        }

        // Check for language tag or datatype
        let remaining: String = chars.collect();

        if let Some(lang) = remaining.strip_prefix('@') {
            Ok(StarTerm::literal_with_language(&value, lang)
                .map_err(|e| anyhow::anyhow!("Invalid literal: {}", e))?)
        } else if let Some(datatype_str) = remaining.strip_prefix("^^") {
            let datatype = if let Some(stripped) = datatype_str
                .strip_prefix('<')
                .and_then(|s| s.strip_suffix('>'))
            {
                stripped.to_string()
            } else {
                context.resolve_prefix(datatype_str)?
            };
            Ok(StarTerm::literal_with_datatype(&value, &datatype)
                .map_err(|e| anyhow::anyhow!("Invalid literal: {}", e))?)
        } else {
            Ok(StarTerm::literal(&value).map_err(|e| anyhow::anyhow!("Invalid literal: {}", e))?)
        }
    }

    /// Check if a TriG statement is complete (enhanced version)
    fn is_complete_trig_statement(&self, statement: &str, state: &mut TrigParserState) -> bool {
        let trimmed = statement.trim();

        // Handle directives (always complete on one line)
        if trimmed.starts_with("@prefix") || trimmed.starts_with("@base") {
            return trimmed.ends_with('.');
        }

        // Handle graph block closing
        if trimmed == "}" {
            return true;
        }

        let mut brace_count: i32 = 0;
        let mut in_string = false;
        let mut escape_next = false;
        let mut quoted_triple_depth: i32 = 0;
        let mut chars = statement.chars().peekable();

        while let Some(ch) = chars.next() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match ch {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '<' if !in_string => {
                    // Check for quoted triple start
                    if chars.peek() == Some(&'<') {
                        chars.next(); // consume second '<'
                        quoted_triple_depth += 1;
                    }
                }
                '>' if !in_string && quoted_triple_depth > 0 => {
                    // Check for quoted triple end
                    if chars.peek() == Some(&'>') {
                        chars.next(); // consume second '>'
                        quoted_triple_depth = quoted_triple_depth.saturating_sub(1);
                    }
                }
                '{' if !in_string && quoted_triple_depth == 0 => {
                    brace_count += 1;
                    if !state.in_graph_block {
                        state.parsing_graph_name = false;
                    }
                }
                '}' if !in_string && quoted_triple_depth == 0 => {
                    brace_count = brace_count.saturating_sub(1);
                }
                '.' if !in_string && quoted_triple_depth == 0 && brace_count == 0 => {
                    // Statement ends with a dot and we're not in any nested structure
                    return true;
                }
                _ => {}
            }
        }

        // Special handling for graph declarations
        if brace_count > 0 && !state.in_graph_block {
            // We have an opening brace - this is a complete graph declaration start
            return true;
        }

        // Check for complete graph block
        if brace_count == 0 && state.in_graph_block && trimmed.ends_with('}') {
            return true;
        }

        false
    }

    /// Parse a complete TriG statement (enhanced version with proper named graph support)
    fn parse_complete_trig_statement(
        &self,
        statement: &str,
        context: &mut ParseContext,
        graph: &mut StarGraph,
        state: &mut TrigParserState,
    ) -> Result<()> {
        let statement = statement.trim();

        // Handle directives
        if statement.starts_with("@prefix") {
            self.parse_prefix_directive(statement, context)?;
            return Ok(());
        }

        if statement.starts_with("@base") {
            self.parse_base_directive(statement, context)?;
            return Ok(());
        }

        // Handle graph blocks
        if statement.contains('{') {
            return self.parse_graph_block(statement, context, graph, state);
        }

        // Handle graph block closing
        if statement == "}" {
            state.exit_graph_block();
            return Ok(());
        }

        // Handle regular triples
        if let Some(stripped) = statement.strip_suffix('.') {
            let triple_str = &stripped.trim();
            if !triple_str.is_empty() {
                let triple = self.parse_triple_pattern(triple_str, context)?;

                // Create quad with current graph context
                let quad = StarQuad::new(
                    triple.subject,
                    triple.predicate,
                    triple.object,
                    state.current_graph.clone(),
                );

                graph.insert_quad(quad)?;
            }
        }

        Ok(())
    }

    /// Parse a graph block declaration (enhanced version)
    /// Parse graph name with error recovery
    fn parse_graph_name_with_recovery(
        &self,
        graph_name: &str,
        context: &mut ParseContext,
    ) -> StarResult<StarTerm> {
        let graph_name = graph_name.trim();

        // Try different graph name formats with recovery

        // Check for quoted graph names (common mistake)
        if graph_name.starts_with('"') && graph_name.ends_with('"') && graph_name.len() >= 2 {
            let inner = &graph_name[1..graph_name.len() - 1];
            context.add_error(
                format!("Graph names should not be quoted strings. Converting '{inner}' to IRI"),
                graph_name.to_string(),
                ErrorSeverity::Warning,
            );
            // Try to convert to IRI
            if inner.contains(':') || inner.starts_with("http") {
                return StarTerm::iri(inner);
            }
        }

        // Parse normally
        self.parse_term_safe(graph_name, context)
    }

    fn parse_graph_block(
        &self,
        statement: &str,
        context: &mut ParseContext,
        graph: &mut StarGraph,
        state: &mut TrigParserState,
    ) -> Result<()> {
        // Find the opening brace
        if let Some(brace_pos) = statement.find('{') {
            let graph_part = statement[..brace_pos].trim();
            let content_part = &statement[brace_pos + 1..];

            // Parse graph name if present
            let graph_term = if graph_part.is_empty() {
                None // Default graph
            } else {
                // Enhanced graph name parsing with better error handling
                match self.parse_graph_name_with_recovery(graph_part, context) {
                    Ok(term) => {
                        // Validate that the graph name is appropriate (IRI or blank node)
                        match &term {
                            StarTerm::NamedNode(_) | StarTerm::BlankNode(_) => Some(term),
                            _ => {
                                let error_msg = format!(
                                    "Graph name must be an IRI or blank node, found: {term:?}"
                                );
                                context.add_error(
                                    error_msg.clone(),
                                    statement.to_string(),
                                    ErrorSeverity::Error,
                                );
                                if context.strict_mode {
                                    return Err(anyhow::anyhow!(error_msg));
                                }
                                None // Continue with default graph in non-strict mode
                            }
                        }
                    }
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Failed to parse graph name '{}': {}",
                            graph_part,
                            e
                        ));
                    }
                }
            };

            // Check for nested graph declarations
            if state.in_graph_block {
                debug!(
                    "Warning: Nested graph declaration detected. Previous graph: {:?}, New graph: {:?}",
                    state.current_graph, graph_term
                );
                // Store the previous graph context for potential recovery
            }

            state.enter_graph_block(graph_term);

            // Parse any triples in the same line
            let remaining_content = content_part.trim();
            if !remaining_content.is_empty() && !remaining_content.starts_with('}') {
                // Handle triples on the same line as graph declaration
                for triple_candidate in remaining_content.split('.') {
                    let triple_str = triple_candidate.trim();
                    if !triple_str.is_empty() && !triple_str.starts_with('}') {
                        match self.parse_triple_pattern_safe(triple_str, context) {
                            Ok(triple) => {
                                let quad = StarQuad::new(
                                    triple.subject,
                                    triple.predicate,
                                    triple.object,
                                    state.current_graph.clone(),
                                );
                                graph.insert_quad(quad)?;
                            }
                            Err(e) => {
                                // Log error but continue parsing
                                debug!(
                                    "Error parsing triple in graph block: '{}' - Error: {}",
                                    triple_str, e
                                );
                                context.add_error(
                                    format!("Failed to parse triple in graph block: {e}"),
                                    triple_str.to_string(),
                                    ErrorSeverity::Warning,
                                );
                            }
                        }
                    }
                }
            }

            // Handle closing brace on same line
            if remaining_content.contains('}') {
                state.exit_graph_block();
            }
        }

        Ok(())
    }

    /// Get parsing errors (placeholder implementation)
    /// Parse JSON-LD-star format with RDF-star extension
    pub fn parse_jsonld_star<R: Read>(&self, reader: R) -> StarResult<StarGraph> {
        let span = span!(Level::DEBUG, "parse_jsonld_star");
        let _enter = span.enter();

        let mut graph = StarGraph::new();
        let mut context = self.create_parse_context();

        // Read the entire JSON content
        let mut content = String::new();
        let mut buf_reader = BufReader::new(reader);
        buf_reader.read_to_string(&mut content).map_err(|e| {
            StarError::parse_error(format!("Failed to read JSON-LD-star content: {e}"))
        })?;

        // Parse JSON
        let json_value: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| StarError::parse_error(format!("Invalid JSON: {e}")))?;

        // Process JSON-LD-star document
        self.process_jsonld_star_value(&json_value, &mut graph, &mut context)?;

        debug!(
            "Parsed JSON-LD-star: {} quads, {} errors",
            graph.quad_len(),
            context.parsing_errors.len()
        );

        if !context.parsing_errors.is_empty() && context.strict_mode {
            return Err(StarError::parse_error(format!(
                "JSON-LD-star parsing failed with {} errors",
                context.parsing_errors.len()
            )));
        }

        Ok(graph)
    }

    /// Process a JSON-LD-star value recursively
    fn process_jsonld_star_value(
        &self,
        value: &serde_json::Value,
        graph: &mut StarGraph,
        context: &mut ParseContext,
    ) -> StarResult<()> {
        match value {
            serde_json::Value::Object(obj) => {
                self.process_jsonld_star_object(obj, graph, context)?;
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.process_jsonld_star_value(item, graph, context)?;
                }
            }
            _ => {
                // Skip primitive values at top level
            }
        }
        Ok(())
    }

    /// Process a JSON-LD-star object
    fn process_jsonld_star_object(
        &self,
        obj: &serde_json::Map<String, serde_json::Value>,
        graph: &mut StarGraph,
        context: &mut ParseContext,
    ) -> StarResult<()> {
        // Check for quoted triple annotation (RDF-star extension)
        if obj.contains_key("@annotation") {
            return self.process_quoted_triple_annotation(obj, graph, context);
        }

        // Extract subject (either @id or generate blank node)
        let subject = if let Some(id_value) = obj.get("@id") {
            match id_value {
                serde_json::Value::String(id) => StarTerm::iri(id)?,
                _ => return Err(StarError::parse_error("@id must be a string".to_string())),
            }
        } else {
            StarTerm::BlankNode(BlankNode {
                id: context.next_blank_node(),
            })
        };

        // Process properties
        for (key, value) in obj {
            if key.starts_with('@') {
                // Skip JSON-LD keywords for now
                continue;
            }

            let predicate = StarTerm::iri(key)?;

            // Process property values
            self.process_property_values(&subject, &predicate, value, graph, context)?;
        }

        Ok(())
    }

    /// Process property values (which may be arrays or single values)
    fn process_property_values(
        &self,
        subject: &StarTerm,
        predicate: &StarTerm,
        value: &serde_json::Value,
        graph: &mut StarGraph,
        context: &mut ParseContext,
    ) -> StarResult<()> {
        match value {
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.create_triple_from_value(subject, predicate, item, graph, context)?;
                }
            }
            _ => {
                self.create_triple_from_value(subject, predicate, value, graph, context)?;
            }
        }
        Ok(())
    }

    /// Create a triple from a JSON-LD value
    fn create_triple_from_value(
        &self,
        subject: &StarTerm,
        predicate: &StarTerm,
        value: &serde_json::Value,
        graph: &mut StarGraph,
        context: &mut ParseContext,
    ) -> StarResult<()> {
        let object = match value {
            serde_json::Value::String(s) => {
                if s.starts_with("http://") || s.starts_with("https://") || s.contains(':') {
                    StarTerm::iri(s)?
                } else {
                    StarTerm::Literal(Literal {
                        value: s.clone(),
                        datatype: None,
                        language: None,
                    })
                }
            }
            serde_json::Value::Object(obj) => {
                if let Some(id_value) = obj.get("@id") {
                    // Reference to another resource
                    if let serde_json::Value::String(id) = id_value {
                        StarTerm::iri(id)?
                    } else {
                        return Err(StarError::parse_error("@id must be a string".to_string()));
                    }
                } else if let Some(value_str) = obj.get("@value") {
                    // Typed literal
                    let value_str = match value_str {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Number(n) => n.to_string(),
                        serde_json::Value::Bool(b) => b.to_string(),
                        _ => return Err(StarError::parse_error("Invalid @value".to_string())),
                    };

                    let datatype = obj
                        .get("@type")
                        .and_then(|v| v.as_str())
                        .map(|s| NamedNode { iri: s.to_string() });

                    let language = obj
                        .get("@language")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    StarTerm::Literal(Literal {
                        value: value_str,
                        language,
                        datatype,
                    })
                } else {
                    // Nested object - process recursively and return blank node
                    let blank_node = StarTerm::BlankNode(BlankNode {
                        id: context.next_blank_node(),
                    });
                    self.process_jsonld_star_object(obj, graph, context)?;
                    blank_node
                }
            }
            serde_json::Value::Number(n) => StarTerm::Literal(Literal {
                value: n.to_string(),
                datatype: Some(NamedNode {
                    iri: "http://www.w3.org/2001/XMLSchema#decimal".to_string(),
                }),
                language: None,
            }),
            serde_json::Value::Bool(b) => StarTerm::Literal(Literal {
                value: b.to_string(),
                datatype: Some(NamedNode {
                    iri: "http://www.w3.org/2001/XMLSchema#boolean".to_string(),
                }),
                language: None,
            }),
            _ => {
                return Err(StarError::parse_error(
                    "Unsupported JSON value type".to_string(),
                ));
            }
        };

        // Create and insert quad
        let quad = StarQuad::new(subject.clone(), predicate.clone(), object, None);
        graph.insert_quad(quad)?;

        Ok(())
    }

    /// Process quoted triple annotation (RDF-star extension for JSON-LD)
    fn process_quoted_triple_annotation(
        &self,
        obj: &serde_json::Map<String, serde_json::Value>,
        graph: &mut StarGraph,
        context: &mut ParseContext,
    ) -> StarResult<()> {
        let annotation = obj
            .get("@annotation")
            .ok_or_else(|| StarError::parse_error("Missing @annotation".to_string()))?;

        match annotation {
            serde_json::Value::Object(ann_obj) => {
                // Extract the quoted triple
                let subject_val = ann_obj.get("subject").ok_or_else(|| {
                    StarError::parse_error("Missing subject in annotation".to_string())
                })?;
                let predicate_val = ann_obj.get("predicate").ok_or_else(|| {
                    StarError::parse_error("Missing predicate in annotation".to_string())
                })?;
                let object_val = ann_obj.get("object").ok_or_else(|| {
                    StarError::parse_error("Missing object in annotation".to_string())
                })?;

                let subject = self.json_value_to_star_term(subject_val)?;
                let predicate = self.json_value_to_star_term(predicate_val)?;
                let object = self.json_value_to_star_term(object_val)?;

                // Create quoted triple
                let quoted_triple =
                    StarTerm::QuotedTriple(Box::new(StarTriple::new(subject, predicate, object)));

                // Process annotation properties
                for (prop_key, prop_value) in obj {
                    if prop_key == "@annotation" {
                        continue;
                    }

                    let annotation_predicate = StarTerm::iri(prop_key)?;
                    self.process_property_values(
                        &quoted_triple,
                        &annotation_predicate,
                        prop_value,
                        graph,
                        context,
                    )?;
                }
            }
            _ => {
                return Err(StarError::parse_error(
                    "@annotation must be an object".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Convert JSON value to StarTerm
    fn json_value_to_star_term(&self, value: &serde_json::Value) -> StarResult<StarTerm> {
        match value {
            serde_json::Value::String(s) => {
                if s.starts_with("_:") {
                    Ok(StarTerm::BlankNode(BlankNode { id: s.clone() }))
                } else {
                    StarTerm::iri(s)
                }
            }
            serde_json::Value::Object(obj) => {
                if let Some(id) = obj.get("@id").and_then(|v| v.as_str()) {
                    StarTerm::iri(id)
                } else if let Some(value) = obj.get("@value").and_then(|v| v.as_str()) {
                    let datatype = obj
                        .get("@type")
                        .and_then(|v| v.as_str())
                        .map(|s| NamedNode { iri: s.to_string() });
                    let language = obj
                        .get("@language")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    Ok(StarTerm::Literal(Literal {
                        value: value.to_string(),
                        datatype,
                        language,
                    }))
                } else {
                    Err(StarError::parse_error("Invalid object term".to_string()))
                }
            }
            _ => Err(StarError::parse_error("Invalid term value".to_string())),
        }
    }
}

impl Default for StarParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_triple_parsing() {
        let parser = StarParser::new();
        let data = r#"
            <http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .
        "#;

        let graph = parser.parse_str(data, StarFormat::NTriplesStar).unwrap();
        assert_eq!(graph.len(), 1);

        let triples = graph.triples();
        let triple = &triples[0];
        assert!(triple.subject.is_named_node());
        assert!(triple.predicate.is_named_node());
        assert!(triple.object.is_named_node());
    }

    #[test]
    fn test_quoted_triple_parsing() {
        let parser = StarParser::new();
        let data = r#"
            << <http://example.org/alice> <http://example.org/age> "25" >> <http://example.org/certainty> "0.9" .
        "#;

        let graph = parser.parse_str(data, StarFormat::NTriplesStar).unwrap();
        assert_eq!(graph.len(), 1);

        let triples = graph.triples();
        let triple = &triples[0];
        assert!(triple.subject.is_quoted_triple());
        assert!(triple.predicate.is_named_node());
        assert!(triple.object.is_literal());
    }

    #[test]
    fn test_turtle_star_with_prefixes() {
        let parser = StarParser::new();
        let data = r#"
            @prefix ex: <http://example.org/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .

            ex:alice foaf:knows ex:bob .
            << ex:alice foaf:age "25" >> ex:certainty "high" .
        "#;

        let graph = parser.parse_str(data, StarFormat::TurtleStar).unwrap();
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn test_literal_parsing() {
        let parser = StarParser::new();
        let mut context = ParseContext::new();

        // Simple literal
        let term = parser.parse_term(r#""hello""#, &mut context).unwrap();
        assert!(term.is_literal());

        // Literal with language tag
        let term = parser.parse_term(r#""hello"@en"#, &mut context).unwrap();
        assert!(term.is_literal());
        if let Some(literal) = term.as_literal() {
            assert_eq!(literal.language, Some("en".to_string()));
        }

        // Literal with datatype
        let term = parser
            .parse_term(
                r#""25"^^<http://www.w3.org/2001/XMLSchema#integer>"#,
                &mut context,
            )
            .unwrap();
        assert!(term.is_literal());
    }

    #[test]
    fn test_tokenization() {
        let parser = StarParser::new();

        // Simple triple
        let tokens = parser.tokenize_triple(r#"<s> <p> <o>"#).unwrap();
        assert_eq!(tokens, vec!["<s>", "<p>", "<o>"]);

        // Quoted triple as subject
        let tokens = parser
            .tokenize_triple(r#"<< <s> <p> <o> >> <certainty> "high""#)
            .unwrap();
        assert_eq!(
            tokens,
            vec!["<< <s> <p> <o> >>", "<certainty>", r#""high""#]
        );
    }

    #[test]
    fn test_error_handling() {
        let parser = StarParser::new();

        // Invalid format
        let result = parser.parse_str("invalid data", StarFormat::NTriplesStar);
        assert!(result.is_err());

        // Unclosed quoted triple
        let result = parser.parse_str(
            r#"<< <s> <p> <o> <certainty> "high" ."#,
            StarFormat::NTriplesStar,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_nquads_star_parsing() {
        let parser = StarParser::new();

        // Simple quad
        let nquads = r#"<http://example.org/s> <http://example.org/p> "test" <http://example.org/g> .
<http://example.org/s2> <http://example.org/p2> <http://example.org/o2> ."#;

        let result = parser.parse_str(nquads, StarFormat::NQuadsStar).unwrap();
        assert_eq!(result.quad_len(), 2);
        assert_eq!(result.total_len(), 2);

        // Quad with quoted triple
        let nquads_with_quoted = r#"<< <http://example.org/alice> <http://example.org/says> "hello" >> <http://example.org/certainty> "0.9" <http://example.org/provenance> ."#;

        let result = parser
            .parse_str(nquads_with_quoted, StarFormat::NQuadsStar)
            .unwrap();
        assert_eq!(result.quad_len(), 1);
        assert!(result.count_quoted_triples() > 0);
    }

    #[test]
    fn test_trig_star_parsing() {
        let parser = StarParser::new();

        // Simple TriG with named graphs
        let trig = r#"
@prefix ex: <http://example.org/> .

{
    ex:alice ex:knows ex:bob .
}

ex:graph1 {
    ex:charlie ex:likes ex:dave .
    << ex:charlie ex:likes ex:dave >> ex:certainty "0.8" .
}
"#;

        let result = parser.parse_str(trig, StarFormat::TrigStar).unwrap();
        assert_eq!(result.quad_len(), 3);
        assert_eq!(result.named_graph_names().len(), 1);

        // Test default graph
        let default_triples = result.triples();
        assert_eq!(default_triples.len(), 1);

        // Test named graph
        let graph_name = result.named_graph_names()[0];
        let named_triples = result.named_graph_triples(graph_name).unwrap();
        assert_eq!(named_triples.len(), 2);
    }

    #[test]
    fn test_trig_star_error_recovery() {
        let config = StarConfig {
            strict_mode: false,
            ..Default::default()
        }; // Enable error recovery
        let parser = StarParser::with_config(config);

        // TriG with errors that should be recoverable
        let trig_with_errors = r#"
@prefix ex: <http://example.org/> .

# Valid triple
ex:alice ex:knows ex:bob .

# Invalid triple (missing object) - should be skipped
ex:charlie ex:likes .

# Valid graph block
ex:graph1 {
    ex:dave ex:age "30" .
}

# Unclosed graph block - should report error but continue
ex:graph2 {
    ex:eve ex:age "25" .
    # Missing closing brace

# Valid triple after error
ex:frank ex:knows ex:grace .
"#;

        let result = parser.parse_str(trig_with_errors, StarFormat::TrigStar);
        // Should parse successfully with errors logged
        assert!(result.is_ok());
        let graph = result.unwrap();
        // Should have parsed the valid triples
        assert!(graph.quad_len() >= 3); // At least the valid triples
    }

    #[test]
    fn test_nquads_star_with_blank_nodes() {
        let parser = StarParser::new();

        let nquads = r#"_:b1 <http://example.org/p> "test" <http://example.org/g> .
<< _:b1 <http://example.org/says> "hello" >> <http://example.org/certainty> "0.9" _:g1 ."#;

        let result = parser.parse_str(nquads, StarFormat::NQuadsStar).unwrap();
        assert_eq!(result.quad_len(), 2);
    }
}
