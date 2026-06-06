//! TriG-star, N-Quads-star, and JSON-LD-star parsing.
//!
//! This sibling module of `crate::parser` contains parsers for the
//! quad-oriented RDF-star serializations (TriG-star and N-Quads-star) plus
//! the JSON-LD-star variant.

use std::io::{BufRead, BufReader, Read};

use anyhow::Result;
use tracing::{debug, span, Level};

use crate::model::{StarGraph, StarQuad, StarTerm};
use crate::parser::context::{ErrorSeverity, ParseContext, TrigParserState};
use crate::parser::jsonld;
use crate::parser::tokenizer;
use crate::parser::StarParser;
use crate::{StarError, StarResult};

impl StarParser {
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
        let max_errors = self.config().max_parse_errors.unwrap_or(100);

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

    /// Parse a single TriG-star line
    #[allow(dead_code)]
    pub(crate) fn parse_trig_line(
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
            let graph_part = line
                .split('{')
                .next()
                .expect("split always returns at least one element")
                .trim();
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

    /// Parse a complete TriG statement (enhanced version with proper named graph support)
    pub(crate) fn parse_complete_trig_statement(
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
    pub(crate) fn parse_graph_name_with_recovery(
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

    pub(crate) fn parse_graph_block(
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
    pub(crate) fn process_jsonld_star_value(
        &self,
        value: &serde_json::Value,
        graph: &mut StarGraph,
        context: &mut ParseContext,
    ) -> StarResult<()> {
        jsonld::process_jsonld_star_value(value, graph, context)
    }
}
