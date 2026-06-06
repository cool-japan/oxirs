//! Lexer and low-level parsing primitives for RDF-star.
//!
//! This sibling module of `crate::parser` contains the lexical analysis helpers
//! and triple/quad/term parsing primitives that operate on individual tokens.

use anyhow::Result;

use crate::model::{StarQuad, StarTerm, StarTriple};
use crate::parser::context::{ErrorSeverity, ParseContext};
use crate::parser::tokenizer;
use crate::parser::StarParser;
use crate::{StarError, StarResult};

impl StarParser {
    /// Parse triple pattern with enhanced error handling
    pub(crate) fn parse_triple_pattern_safe(
        &self,
        pattern: &str,
        context: &mut ParseContext,
    ) -> StarResult<StarTriple> {
        self.parse_triple_pattern_safe_with_depth(pattern, context, 0)
    }

    /// Parse term with enhanced error handling
    pub(crate) fn parse_term_safe(
        &self,
        term_str: &str,
        context: &mut ParseContext,
    ) -> StarResult<StarTerm> {
        self.parse_term_safe_with_depth(term_str, context, 0)
    }

    /// Parse term with depth tracking for quoted triples
    pub(crate) fn parse_term_safe_with_depth(
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
    pub(crate) fn validate_quoted_triple_delimiters(&self, term_str: &str) -> bool {
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
    pub(crate) fn parse_triple_pattern_safe_with_depth(
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
    pub(crate) fn parse_term_fallback(
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

    /// Parse a quad pattern with enhanced error handling
    pub(crate) fn parse_quad_pattern_safe(
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
    pub(crate) fn parse_quad_pattern(
        &self,
        pattern: &str,
        context: &mut ParseContext,
    ) -> Result<StarQuad> {
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
    pub(crate) fn parse_triple_pattern(
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
    pub(crate) fn tokenize_triple(&self, pattern: &str) -> Result<Vec<String>> {
        tokenizer::tokenize_triple(pattern)
    }

    /// Tokenize a quad pattern (similar to triple but allows 4 terms)
    pub(crate) fn tokenize_quad(&self, pattern: &str) -> Result<Vec<String>> {
        tokenizer::tokenize_quad(pattern)
    }

    /// Parse a single term (IRI, blank node, literal, or quoted triple)
    pub(crate) fn parse_term(
        &self,
        term_str: &str,
        context: &mut ParseContext,
    ) -> Result<StarTerm> {
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
    pub(crate) fn parse_literal(
        &self,
        literal_str: &str,
        context: &mut ParseContext,
    ) -> Result<StarTerm> {
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
}
