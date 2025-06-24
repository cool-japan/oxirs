//! RDF-star parsing implementations for various formats.
//!
//! This module provides parsers for RDF-star formats including:
//! - Turtle-star (*.ttls)
//! - N-Triples-star (*.nts)  
//! - TriG-star (*.trigs)
//! - N-Quads-star (*.nqs)

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::str::FromStr;

use anyhow::{Context, Result};
use tracing::{debug, error, span, Level};

use crate::model::{NamedNode, StarGraph, StarQuad, StarTerm, StarTriple};
use crate::{StarConfig, StarError, StarResult};

/// RDF-star format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StarFormat {
    /// Turtle-star format
    TurtleStar,
    /// N-Triples-star format
    NTriplesStar,
    /// TriG-star format (named graphs)
    TrigStar,
    /// N-Quads-star format
    NQuadsStar,
}

impl FromStr for StarFormat {
    type Err = StarError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "turtle-star" | "ttls" => Ok(StarFormat::TurtleStar),
            "ntriples-star" | "nts" => Ok(StarFormat::NTriplesStar),
            "trig-star" | "trigs" => Ok(StarFormat::TrigStar),
            "nquads-star" | "nqs" => Ok(StarFormat::NQuadsStar),
            _ => Err(StarError::ParseError(format!("Unknown format: {}", s))),
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
    current_graph: Option<StarTerm>,
    /// Blank node counter
    blank_node_counter: usize,
}

impl ParseContext {
    fn new() -> Self {
        Self::default()
    }

    /// Generate a new blank node identifier
    fn next_blank_node(&mut self) -> String {
        self.blank_node_counter += 1;
        format!("_:b{}", self.blank_node_counter)
    }

    /// Resolve a prefixed name to full IRI
    fn resolve_prefix(&self, prefixed: &str) -> StarResult<String> {
        if let Some(colon_pos) = prefixed.find(':') {
            let prefix = &prefixed[..colon_pos];
            let local = &prefixed[colon_pos + 1..];
            
            if let Some(namespace) = self.prefixes.get(prefix) {
                Ok(format!("{}{}", namespace, local))
            } else {
                Err(StarError::ParseError(format!("Unknown prefix: {}", prefix)))
            }
        } else {
            Err(StarError::ParseError(format!("Invalid prefixed name: {}", prefixed)))
        }
    }

    /// Resolve relative IRI against base
    fn resolve_relative(&self, iri: &str) -> String {
        if let Some(ref base) = self.base_iri {
            // Simple relative IRI resolution (not fully RFC compliant)
            if iri.starts_with('#') {
                format!("{}{}", base, iri)
            } else {
                iri.to_string()
            }
        } else {
            iri.to_string()
        }
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

    /// Parse RDF-star data from a reader
    pub fn parse<R: Read>(&self, reader: R, format: StarFormat) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "parse_rdf_star", format = ?format);
        let _enter = span.enter();

        match format {
            StarFormat::TurtleStar => self.parse_turtle_star(reader),
            StarFormat::NTriplesStar => self.parse_ntriples_star(reader),
            StarFormat::TrigStar => self.parse_trig_star(reader),
            StarFormat::NQuadsStar => self.parse_nquads_star(reader),
        }
    }

    /// Parse RDF-star from string
    pub fn parse_str(&self, data: &str, format: StarFormat) -> StarResult<StarGraph> {
        self.parse(data.as_bytes(), format)
    }

    /// Parse Turtle-star format
    pub fn parse_turtle_star<R: Read>(&self, reader: R) -> StarResult<StarGraph> {
        let span = span!(Level::DEBUG, "parse_turtle_star");
        let _enter = span.enter();

        let mut graph = StarGraph::new();
        let mut context = ParseContext::new();
        let buf_reader = BufReader::new(reader);

        for (line_num, line_result) in buf_reader.lines().enumerate() {
            let line = line_result.map_err(|e| StarError::ParseError(e.to_string()))?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            self.parse_turtle_line(line, &mut context, &mut graph)
                .with_context(|| format!("Error parsing line {}: {}", line_num + 1, line))
                .map_err(|e| StarError::ParseError(e.to_string()))?;
        }

        debug!("Parsed {} triples in Turtle-star format", graph.len());
        Ok(graph)
    }

    /// Parse a single Turtle-star line
    fn parse_turtle_line(&self, line: &str, context: &mut ParseContext, graph: &mut StarGraph) -> Result<()> {
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
        if line.ends_with('.') {
            let triple_str = &line[..line.len() - 1].trim();
            let triple = self.parse_triple_pattern(triple_str, context)?;
            graph.insert(triple)?;
        }

        Ok(())
    }

    /// Parse a single TriG-star line
    fn parse_trig_line(
        &self, 
        line: &str, 
        context: &mut ParseContext, 
        graph: &mut StarGraph,
        current_graph: &mut Option<StarTerm>,
        in_graph_block: &mut bool,
        brace_count: &mut usize
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
                context.current_graph = current_graph.clone();
            } else {
                // Default graph
                *current_graph = None;
                context.current_graph = None;
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
                context.current_graph = None;
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

    /// Parse @prefix directive
    fn parse_prefix_directive(&self, line: &str, context: &mut ParseContext) -> Result<()> {
        // @prefix prefix: <namespace> .
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let prefix_name = parts[1].trim_end_matches(':');
            let namespace = parts[2].trim_matches(['<', '>', '.']);
            context.prefixes.insert(prefix_name.to_string(), namespace.to_string());
        }
        Ok(())
    }

    /// Parse @base directive
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
            let line = line_result.map_err(|e| StarError::ParseError(e.to_string()))?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.ends_with('.') {
                let triple_str = &line[..line.len() - 1].trim();
                let triple = self.parse_triple_pattern(triple_str, &mut context)
                    .with_context(|| format!("Error parsing line {}: {}", line_num + 1, line))
                    .map_err(|e| StarError::ParseError(e.to_string()))?;
                graph.insert(triple)?;
            } else {
                // In N-Triples-star, all non-empty lines must be valid triples ending with '.'
                return Err(StarError::ParseError(format!("Invalid N-Triples-star line {}: {}", line_num + 1, line)));
            }
        }

        debug!("Parsed {} triples in N-Triples-star format", graph.len());
        Ok(graph)
    }

    /// Parse TriG-star format (with named graphs)
    pub fn parse_trig_star<R: Read>(&self, reader: R) -> StarResult<StarGraph> {
        let span = span!(Level::DEBUG, "parse_trig_star");
        let _enter = span.enter();

        let mut graph = StarGraph::new();
        let mut context = ParseContext::new();
        let buf_reader = BufReader::new(reader);

        let mut current_graph: Option<StarTerm> = None;
        let mut in_graph_block = false;
        let mut brace_count = 0;

        for (line_num, line_result) in buf_reader.lines().enumerate() {
            let line = line_result.map_err(|e| StarError::ParseError(e.to_string()))?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            self.parse_trig_line(line, &mut context, &mut graph, &mut current_graph, &mut in_graph_block, &mut brace_count)
                .with_context(|| format!("Error parsing TriG-star line {}: {}", line_num + 1, line))
                .map_err(|e| StarError::ParseError(e.to_string()))?;
        }

        debug!("Parsed {} triples in TriG-star format", graph.len());
        Ok(graph)
    }

    /// Parse N-Quads-star format
    pub fn parse_nquads_star<R: Read>(&self, reader: R) -> StarResult<StarGraph> {
        let span = span!(Level::DEBUG, "parse_nquads_star");
        let _enter = span.enter();

        let mut graph = StarGraph::new();
        let mut context = ParseContext::new();
        let buf_reader = BufReader::new(reader);

        for (line_num, line_result) in buf_reader.lines().enumerate() {
            let line = line_result.map_err(|e| StarError::ParseError(e.to_string()))?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.ends_with('.') {
                let quad_str = &line[..line.len() - 1].trim();
                let quad = self.parse_quad_pattern(quad_str, &mut context)
                    .with_context(|| format!("Error parsing line {}: {}", line_num + 1, line))
                    .map_err(|e| StarError::ParseError(e.to_string()))?;
                
                // Insert the quad into the graph with its graph component
                if let Some(graph_name) = quad.graph {
                    // For named graphs, we can store them with graph context
                    // For now, we'll add all quads to the default graph
                    // TODO: Implement proper named graph support in StarGraph
                    let triple = StarTriple::new(quad.subject, quad.predicate, quad.object);
                    graph.insert(triple)?;
                } else {
                    // Default graph
                    let triple = StarTriple::new(quad.subject, quad.predicate, quad.object);
                    graph.insert(triple)?;
                }
            } else {
                // In N-Quads-star, all non-empty lines must be valid quads ending with '.'
                return Err(StarError::ParseError(format!("Invalid N-Quads-star line {}: {}", line_num + 1, line)));
            }
        }

        debug!("Parsed {} triples in N-Quads-star format", graph.len());
        Ok(graph)
    }

    /// Parse a quad pattern (subject predicate object graph)
    fn parse_quad_pattern(&self, pattern: &str, context: &mut ParseContext) -> Result<StarQuad> {
        let terms = self.tokenize_quad(pattern)?;
        
        if terms.len() < 3 || terms.len() > 4 {
            return Err(anyhow::anyhow!("Quad must have 3 or 4 terms, found {}", terms.len()));
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
    fn parse_triple_pattern(&self, pattern: &str, context: &mut ParseContext) -> Result<StarTriple> {
        let terms = self.tokenize_triple(pattern)?;
        
        if terms.len() != 3 {
            return Err(anyhow::anyhow!("Triple must have exactly 3 terms, found {}", terms.len()));
        }

        let subject = self.parse_term(&terms[0], context)?;
        let predicate = self.parse_term(&terms[1], context)?;
        let object = self.parse_term(&terms[2], context)?;

        let triple = StarTriple::new(subject, predicate, object);
        triple.validate().map_err(|e| anyhow::anyhow!("Invalid triple: {}", e))?;

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
            return Ok(StarTerm::iri(&resolved).map_err(|e| anyhow::anyhow!("Invalid IRI: {}", e))?);
        }

        // Prefixed name
        if term_str.contains(':') && !term_str.starts_with('_') && !term_str.starts_with('"') {
            let resolved = context.resolve_prefix(term_str)?;
            return Ok(StarTerm::iri(&resolved).map_err(|e| anyhow::anyhow!("Invalid IRI: {}", e))?);
        }

        // Blank node: _:id
        if term_str.starts_with("_:") {
            let id = &term_str[2..];
            return Ok(StarTerm::blank_node(id).map_err(|e| anyhow::anyhow!("Invalid blank node: {}", e))?);
        }

        // Literal: "value"@lang or "value"^^<datatype>
        if term_str.starts_with('"') {
            return self.parse_literal(term_str, context);
        }

        // Variable: ?name (for SPARQL-star)
        if term_str.starts_with('?') {
            let name = &term_str[1..];
            return Ok(StarTerm::variable(name).map_err(|e| anyhow::anyhow!("Invalid variable: {}", e))?);
        }

        Err(anyhow::anyhow!("Unrecognized term format: {}", term_str))
    }

    /// Parse a literal term with optional language tag or datatype
    fn parse_literal(&self, literal_str: &str, context: &ParseContext) -> Result<StarTerm> {
        let mut chars = literal_str.chars().peekable();
        let mut value = String::new();
        let mut in_string = false;
        let mut escape_next = false;

        // Skip opening quote
        if chars.next() != Some('"') {
            return Err(anyhow::anyhow!("Literal must start with quote"));
        }
        in_string = true;

        // Parse value
        while let Some(ch) = chars.next() {
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
        
        if remaining.starts_with('@') {
            let lang = &remaining[1..];
            Ok(StarTerm::literal_with_language(&value, lang).map_err(|e| anyhow::anyhow!("Invalid literal: {}", e))?)
        } else if remaining.starts_with("^^") {
            let datatype_str = &remaining[2..];
            let datatype = if datatype_str.starts_with('<') && datatype_str.ends_with('>') {
                datatype_str[1..datatype_str.len() - 1].to_string()
            } else {
                context.resolve_prefix(datatype_str)?
            };
            Ok(StarTerm::literal_with_datatype(&value, &datatype).map_err(|e| anyhow::anyhow!("Invalid literal: {}", e))?)
        } else {
            Ok(StarTerm::literal(&value).map_err(|e| anyhow::anyhow!("Invalid literal: {}", e))?)
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
        let term = parser.parse_term(r#""25"^^<http://www.w3.org/2001/XMLSchema#integer>"#, &mut context).unwrap();
        assert!(term.is_literal());
    }

    #[test]
    fn test_tokenization() {
        let parser = StarParser::new();
        
        // Simple triple
        let tokens = parser.tokenize_triple(r#"<s> <p> <o>"#).unwrap();
        assert_eq!(tokens, vec!["<s>", "<p>", "<o>"]);

        // Quoted triple as subject
        let tokens = parser.tokenize_triple(r#"<< <s> <p> <o> >> <certainty> "high""#).unwrap();
        assert_eq!(tokens, vec!["<< <s> <p> <o> >>", "<certainty>", r#""high""#]);
    }

    #[test]
    fn test_error_handling() {
        let parser = StarParser::new();
        
        // Invalid format
        let result = parser.parse_str("invalid data", StarFormat::NTriplesStar);
        assert!(result.is_err());

        // Unclosed quoted triple
        let result = parser.parse_str(r#"<< <s> <p> <o> <certainty> "high" ."#, StarFormat::NTriplesStar);
        assert!(result.is_err());
    }
}
