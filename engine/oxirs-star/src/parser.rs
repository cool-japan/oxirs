//! RDF-star parsing implementations for various formats.
//!
//! This module provides parsers for RDF-star formats including:
//! - Turtle-star (*.ttls)
//! - N-Triples-star (*.nts)
//! - TriG-star (*.trigs)
//! - N-Quads-star (*.nqs)

// Internal parser submodules
#[path = "parser/context.rs"]
mod context;
#[path = "parser/jsonld.rs"]
mod jsonld;
#[path = "parser/simd_scanner.rs"]
mod simd_scanner;
#[path = "parser/tokenizer.rs"]
mod tokenizer;

use std::io::{BufRead, BufReader, Read};
use std::str::FromStr;

use anyhow::{Context, Result};
use tracing::{debug, span, Level};

use crate::model::{StarGraph, StarQuad, StarTerm, StarTriple};
use crate::{StarConfig, StarError, StarResult};

// Import parser context types
use context::{ErrorSeverity, ParseContext, TrigParserState};
use simd_scanner::SimdScanner;

// Re-export public error types
pub use context::{ErrorSeverity as PublicErrorSeverity, ParseError as PublicParseError};

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

// TrigParserState has been moved to parser/context.rs

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

// ParseContext, ParseError, and ErrorSeverity have been moved to parser/context.rs

/// RDF-star parser with support for multiple formats
pub struct StarParser {
    config: StarConfig,
    simd_scanner: SimdScanner,
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

    /// Parse Turtle-star format with enhanced error handling and multi-line statement support
    pub fn parse_turtle_star<R: Read>(&self, reader: R) -> StarResult<StarGraph> {
        let span = span!(Level::DEBUG, "parse_turtle_star");
        let _enter = span.enter();

        let mut graph = StarGraph::new();
        let mut context = self.create_parse_context();
        let buf_reader = BufReader::new(reader);

        // Accumulator for multi-line statements
        let mut accumulated_line = String::new();
        let mut error_count = 0;
        let max_errors = self.config.max_parse_errors.unwrap_or(100);

        for (line_num, line_result) in buf_reader.lines().enumerate() {
            let line = line_result.map_err(|e| StarError::parse_error(e.to_string()))?;
            // Use SIMD-accelerated trimming
            let line = self.simd_scanner.trim(&line);

            // Update position tracking
            context.update_position(line_num + 1, 0);

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Strip inline comments (but respect strings)
            let line = tokenizer::strip_inline_comment(line);
            // Use SIMD-accelerated trimming
            let line = self.simd_scanner.trim(line);

            // Skip if line becomes empty after comment stripping
            if line.is_empty() {
                continue;
            }

            // Check if line looks incomplete after comment stripping (no period, not a directive, not in a special block)
            // This handles cases like "ex:charlie ex:knows  # missing object"
            // But allow lines that are part of multi-line blocks (annotation blocks or quoted triples)
            // Use SIMD-accelerated pattern detection
            let in_annotation_block = self
                .simd_scanner
                .contains_annotation_block(&accumulated_line)
                || self.simd_scanner.contains_annotation_block(line);
            let in_quoted_triple = self.simd_scanner.contains_quoted_triple(&accumulated_line);

            if !line.ends_with('.')
                && !line.ends_with('{')
                && !line.ends_with('}')
                && !in_annotation_block
                && !in_quoted_triple
            {
                // Check if line contains incomplete quoted triple using SIMD
                let has_incomplete_quoted_triple = if self.simd_scanner.contains_quoted_triple(line)
                {
                    // Use SIMD-accelerated counting
                    !self.simd_scanner.is_quoted_balanced(line)
                } else {
                    false
                };

                // This line appears incomplete and might have been truncated by inline comment
                // Skip it in error recovery mode if:
                // 1. It's not a directive (doesn't start with '@'), AND
                // 2. It's incomplete (doesn't end with '.', '{', '}' and isn't part of annotation/quoted block)
                if !line.starts_with('@') && !context.strict_mode {
                    // This is an incomplete statement - skip it in error recovery mode
                    let reason = if has_incomplete_quoted_triple {
                        "incomplete quoted triple"
                    } else {
                        "incomplete triple (possibly truncated by comment)"
                    };
                    context.add_error(
                        format!("Skipping {}: {line}", reason),
                        line.to_string(),
                        ErrorSeverity::Warning,
                    );
                    continue;
                }
            }

            // Handle directives immediately (they're always single-line)
            if line.starts_with("@prefix") || line.starts_with("@base") {
                match self.parse_turtle_line_safe(line, &mut context, &mut graph) {
                    Ok(_) => continue,
                    Err(e) => {
                        error_count += 1;
                        let line_number = line_num + 1;
                        let error_context = format!("line {line_number}: {line}");
                        context.add_error(
                            format!("Parse error: {e}"),
                            error_context,
                            if context.strict_mode {
                                ErrorSeverity::Fatal
                            } else {
                                ErrorSeverity::Error
                            },
                        );

                        if context.strict_mode || error_count >= max_errors {
                            return Err(StarError::parse_error(format!(
                                "Parsing failed at line {} with {} errors",
                                line_num + 1,
                                context.get_errors().len()
                            )));
                        }
                        continue;
                    }
                }
            }

            // Accumulate lines for multi-line statements
            accumulated_line.push_str(line);
            accumulated_line.push(' ');

            // Check if we have a complete statement (ends with '.')
            if tokenizer::is_complete_turtle_statement(&accumulated_line) {
                match self.parse_turtle_line_safe(accumulated_line.trim(), &mut context, &mut graph)
                {
                    Ok(_) => {
                        accumulated_line.clear();
                    }
                    Err(e) => {
                        error_count += 1;
                        let line_number = line_num + 1;
                        let error_context =
                            format!("line {line_number}: {}", accumulated_line.trim());
                        context.add_error(
                            format!("Parse error: {e}"),
                            error_context,
                            if context.strict_mode {
                                ErrorSeverity::Fatal
                            } else {
                                ErrorSeverity::Error
                            },
                        );

                        if context.strict_mode || error_count >= max_errors {
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

                        // Clear accumulated line and try to recover
                        accumulated_line.clear();
                    }
                }
            }
        }

        // Handle any remaining incomplete statement
        if !accumulated_line.trim().is_empty() {
            context.add_error(
                "Incomplete Turtle-star statement at end of input".to_string(),
                accumulated_line.trim().to_string(),
                ErrorSeverity::Error,
            );
            if context.strict_mode {
                return Err(StarError::parse_error(format!(
                    "Incomplete Turtle-star statement at end of input: {}",
                    accumulated_line.trim()
                )));
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

            // Check for annotation syntax {| ... |}
            if triple_str.contains("{|") {
                return self.parse_triple_with_annotation(triple_str, context, graph);
            }

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

    /// Parse triple with annotation syntax: subject predicate object {| annotation properties |} .
    ///
    /// Example: :alice :age 30 {| :certainty 0.9; :source :survey |} .
    /// Expands to:
    ///   <<:alice :age 30>> :certainty 0.9 .
    ///   <<:alice :age 30>> :source :survey .
    fn parse_triple_with_annotation(
        &self,
        statement: &str,
        context: &mut ParseContext,
        graph: &mut StarGraph,
    ) -> StarResult<()> {
        // Find annotation block delimiters
        let annotation_start = statement.find("{|").ok_or_else(|| {
            StarError::parse_error("Missing annotation start delimiter {|".to_string())
        })?;

        let annotation_end = statement.find("|}").ok_or_else(|| {
            StarError::parse_error("Missing annotation end delimiter |}".to_string())
        })?;

        if annotation_end < annotation_start {
            return Err(StarError::parse_error(
                "Annotation delimiters in wrong order".to_string(),
            ));
        }

        // Extract base triple and annotation block
        let base_triple_str = statement[..annotation_start].trim();
        let annotation_block = statement[annotation_start + 2..annotation_end].trim();

        // Parse the base triple
        let base_triple = self.parse_triple_pattern_safe(base_triple_str, context)?;

        // If the base triple has a quoted triple as subject, insert it
        // This is a regular assertion about a quoted triple, not just metadata
        if base_triple.subject.is_quoted_triple() {
            graph.insert(base_triple.clone())?;
        }

        // Create quoted triple from base triple for annotations
        let quoted_triple = StarTerm::quoted_triple(base_triple.clone());

        // Parse annotation properties (separated by semicolons)
        let annotation_statements = annotation_block.split(';');

        for ann_stmt in annotation_statements {
            let ann_stmt = ann_stmt.trim();
            if ann_stmt.is_empty() {
                continue;
            }

            // Parse annotation as predicate-object pair
            let ann_tokens = self
                .tokenize_triple(&format!("_:dummy {ann_stmt}"))
                .map_err(|e| StarError::parse_error(e.to_string()))?;
            if ann_tokens.len() != 3 {
                let error_msg = format!(
                    "Invalid annotation property: expected predicate object, found {} terms",
                    ann_tokens.len() - 1
                );
                context.add_error(
                    error_msg.clone(),
                    ann_stmt.to_string(),
                    ErrorSeverity::Error,
                );
                if context.strict_mode {
                    return Err(StarError::parse_error(error_msg));
                }
                continue;
            }

            // Parse annotation predicate and object (skip dummy subject at index 0)
            let ann_predicate = self.parse_term_safe(&ann_tokens[1], context)?;
            let ann_object = self.parse_term_safe(&ann_tokens[2], context)?;

            // Create annotation triple: <<base triple>> annotation_predicate annotation_object
            let annotation_triple =
                StarTriple::new(quoted_triple.clone(), ann_predicate, ann_object);

            // Insert annotation triple
            graph.insert(annotation_triple)?;
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
            if tokenizer::is_complete_trig_statement(&accumulated_line, &mut trig_state) {
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
        tokenizer::tokenize_triple(pattern)
    }

    /// Tokenize a quad pattern (similar to triple but allows 4 terms)
    fn tokenize_quad(&self, pattern: &str) -> Result<Vec<String>> {
        tokenizer::tokenize_quad(pattern)
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
        jsonld::process_jsonld_star_value(value, graph, context)
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
