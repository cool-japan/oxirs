//! Statement-level parsing for Turtle-star and N-Triples-star formats.
//!
//! This sibling module of `crate::parser` contains the line-oriented and
//! statement-oriented parsing for the Turtle-star and N-Triples-star
//! serializations, including directive handling and annotation expansion.

use std::io::{BufRead, BufReader, Read};

use anyhow::{Context, Result};
use tracing::{debug, span, Level};

use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::parser::context::{ErrorSeverity, ParseContext};
use crate::parser::tokenizer;
use crate::parser::StarParser;
use crate::{StarError, StarResult};

impl StarParser {
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
        let max_errors = self.config().max_parse_errors.unwrap_or(100);

        for (line_num, line_result) in buf_reader.lines().enumerate() {
            let line = line_result.map_err(|e| StarError::parse_error(e.to_string()))?;
            // Use SIMD-accelerated trimming
            let line = self.simd_scanner().trim(&line);

            // Update position tracking
            context.update_position(line_num + 1, 0);

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Strip inline comments (but respect strings)
            let line = tokenizer::strip_inline_comment(line);
            // Use SIMD-accelerated trimming
            let line = self.simd_scanner().trim(line);

            // Skip if line becomes empty after comment stripping
            if line.is_empty() {
                continue;
            }

            // Check if line looks incomplete after comment stripping (no period, not a directive, not in a special block)
            // This handles cases like "ex:charlie ex:knows  # missing object"
            // But allow lines that are part of multi-line blocks (annotation blocks or quoted triples)
            // Use SIMD-accelerated pattern detection
            let in_annotation_block = self
                .simd_scanner()
                .contains_annotation_block(&accumulated_line)
                || self.simd_scanner().contains_annotation_block(line);
            let in_quoted_triple = self
                .simd_scanner()
                .contains_quoted_triple(&accumulated_line);

            if !line.ends_with('.')
                && !line.ends_with('{')
                && !line.ends_with('}')
                && !in_annotation_block
                && !in_quoted_triple
            {
                // Check if line contains incomplete quoted triple using SIMD
                let has_incomplete_quoted_triple =
                    if self.simd_scanner().contains_quoted_triple(line) {
                        // Use SIMD-accelerated counting
                        !self.simd_scanner().is_quoted_balanced(line)
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
    pub(crate) fn parse_turtle_line_safe(
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
    pub(crate) fn parse_triple_with_annotation(
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

        // The annotation shorthand `{| |}` is syntactic sugar that expands to
        // BOTH the base triple assertion AND the reification annotation
        // triple(s) about it: `<s> <p> <o> .` plus `<<s p o>> ap ao .`.
        // Always assert the base triple, regardless of what term its subject
        // is (plain term or quoted triple).
        graph.insert(base_triple.clone())?;

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
    pub(crate) fn parse_turtle_line(
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

    /// Parse @prefix directive with enhanced error handling
    pub(crate) fn parse_prefix_directive_safe(
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
    pub(crate) fn parse_base_directive_safe(
        &self,
        line: &str,
        context: &mut ParseContext,
    ) -> StarResult<()> {
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

    /// Parse @prefix directive (legacy)
    pub(crate) fn parse_prefix_directive(
        &self,
        line: &str,
        context: &mut ParseContext,
    ) -> Result<()> {
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
    pub(crate) fn parse_base_directive(
        &self,
        line: &str,
        context: &mut ParseContext,
    ) -> Result<()> {
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
}
