//! N3 format parser
//!
//! N3 (Notation3) is a superset of Turtle that adds support for variables,
//! rules, and formulae. This implementation provides basic support for the
//! Turtle subset of N3.

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::toolkit::Parser;
use oxirs_core::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject, Triple};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};

/// N3 parser with basic Turtle subset support
#[derive(Debug, Clone)]
pub struct N3Parser {
    /// Whether to continue parsing after errors
    pub lenient: bool,
    /// Base IRI for resolving relative IRIs
    pub base_iri: Option<String>,
    /// Prefix declarations
    pub prefixes: HashMap<String, String>,
}

impl Default for N3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl N3Parser {
    /// Create a new N3 parser
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();

        // Add standard prefixes
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );

        Self {
            lenient: false,
            base_iri: None,
            prefixes,
        }
    }

    /// Create a lenient N3 parser that continues after errors
    pub fn new_lenient() -> Self {
        Self {
            lenient: true,
            ..Self::new()
        }
    }

    /// Parse N3 content focusing on the Turtle subset
    fn parse_n3_content<R: BufRead>(&self, mut reader: R) -> TurtleResult<Vec<Triple>> {
        let mut content = String::new();
        reader
            .read_to_string(&mut content)
            .map_err(TurtleParseError::io)?;

        let mut triples = Vec::new();
        let mut current_prefixes = self.prefixes.clone();

        // Simple line-by-line parsing
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                i += 1;
                continue;
            }

            // Handle prefix declarations
            if line.starts_with("@prefix") {
                if let Some(parsed_prefix) = self.parse_prefix_declaration(line)? {
                    current_prefixes.insert(parsed_prefix.0, parsed_prefix.1);
                }
                i += 1;
                continue;
            }

            // Handle base declarations
            if line.starts_with("@base") {
                // Parse base - simplified implementation
                i += 1;
                continue;
            }

            // Skip N3-specific constructs that we don't fully support yet
            if line.contains("=>") || line.contains("<=") || line.contains("=") {
                // Skip rules and implications for now
                i += 1;
                continue;
            }

            if line.contains("{") && line.contains("}") {
                // Skip formulae for now
                i += 1;
                continue;
            }

            // Parse regular triple statements
            if line.ends_with('.') || line.ends_with(';') || line.ends_with(',') {
                match self.parse_statement(line, &current_prefixes) {
                    Ok(parsed_triples) => {
                        triples.extend(parsed_triples);
                    }
                    Err(e) => {
                        if !self.lenient {
                            return Err(e);
                        }
                        // Continue parsing in lenient mode
                    }
                }
            }

            i += 1;
        }

        Ok(triples)
    }

    /// Parse a prefix declaration
    fn parse_prefix_declaration(&self, line: &str) -> TurtleResult<Option<(String, String)>> {
        // @prefix ex: <http://example.org/> .
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 && parts[0] == "@prefix" {
            let prefix = parts[1].trim_end_matches(':');
            let namespace = parts[2].trim_start_matches('<').trim_end_matches('>');
            Ok(Some((prefix.to_string(), namespace.to_string())))
        } else {
            Ok(None)
        }
    }

    /// Parse a statement (simplified for Turtle subset)
    fn parse_statement(
        &self,
        line: &str,
        prefixes: &HashMap<String, String>,
    ) -> TurtleResult<Vec<Triple>> {
        let line = line
            .trim_end_matches('.')
            .trim_end_matches(';')
            .trim_end_matches(',')
            .trim();

        // Very basic triple parsing
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(vec![]); // Skip malformed statements in lenient mode
        }

        let subject = self.parse_term_as_subject(parts[0], prefixes)?;
        let predicate = self.parse_term_as_predicate(parts[1], prefixes)?;

        // Join remaining parts for the object (might contain spaces)
        let object_str = parts[2..].join(" ");
        let object = self.parse_term_as_object(&object_str, prefixes)?;

        Ok(vec![Triple::new(subject, predicate, object)])
    }

    /// Parse a term as a subject
    fn parse_term_as_subject(
        &self,
        term: &str,
        prefixes: &HashMap<String, String>,
    ) -> TurtleResult<Subject> {
        if term.starts_with('<') && term.ends_with('>') {
            // Absolute IRI
            let iri = &term[1..term.len() - 1];
            let named_node = NamedNode::new(iri).map_err(TurtleParseError::model)?;
            Ok(Subject::NamedNode(named_node))
        } else if let Some(stripped) = term.strip_prefix("_:") {
            // Blank node
            let blank_node = BlankNode::new(stripped).map_err(TurtleParseError::model)?;
            Ok(Subject::BlankNode(blank_node))
        } else if term.contains(':')
            && !term.starts_with("http://")
            && !term.starts_with("https://")
        {
            // Prefixed name
            let expanded = self.expand_prefixed_name(term, prefixes)?;
            let named_node = NamedNode::new(&expanded).map_err(TurtleParseError::model)?;
            Ok(Subject::NamedNode(named_node))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid subject: {term}"),
                position: TextPosition::default(),
            }))
        }
    }

    /// Parse a term as a predicate
    fn parse_term_as_predicate(
        &self,
        term: &str,
        prefixes: &HashMap<String, String>,
    ) -> TurtleResult<Predicate> {
        if term == "a" {
            // rdf:type shorthand
            let rdf_type =
                NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
            Ok(Predicate::NamedNode(rdf_type))
        } else if term.starts_with('<') && term.ends_with('>') {
            // Absolute IRI
            let iri = &term[1..term.len() - 1];
            let named_node = NamedNode::new(iri).map_err(TurtleParseError::model)?;
            Ok(Predicate::NamedNode(named_node))
        } else if term.contains(':')
            && !term.starts_with("http://")
            && !term.starts_with("https://")
        {
            // Prefixed name
            let expanded = self.expand_prefixed_name(term, prefixes)?;
            let named_node = NamedNode::new(&expanded).map_err(TurtleParseError::model)?;
            Ok(Predicate::NamedNode(named_node))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid predicate: {term}"),
                position: TextPosition::default(),
            }))
        }
    }

    /// Parse a term as an object
    fn parse_term_as_object(
        &self,
        term: &str,
        prefixes: &HashMap<String, String>,
    ) -> TurtleResult<Object> {
        let term = term.trim();

        if term.starts_with('"') {
            // String literal (simplified parsing)
            self.parse_simple_literal(term)
        } else if term.starts_with('<') && term.ends_with('>') {
            // Absolute IRI
            let iri = &term[1..term.len() - 1];
            let named_node = NamedNode::new(iri).map_err(TurtleParseError::model)?;
            Ok(Object::NamedNode(named_node))
        } else if let Some(stripped) = term.strip_prefix("_:") {
            // Blank node
            let blank_node = BlankNode::new(stripped).map_err(TurtleParseError::model)?;
            Ok(Object::BlankNode(blank_node))
        } else if term.contains(':')
            && !term.starts_with("http://")
            && !term.starts_with("https://")
        {
            // Prefixed name
            let expanded = self.expand_prefixed_name(term, prefixes)?;
            let named_node = NamedNode::new(&expanded).map_err(TurtleParseError::model)?;
            Ok(Object::NamedNode(named_node))
        } else if term.chars().all(|c| c.is_ascii_digit()) {
            // Integer literal
            let literal = Literal::new_typed_literal(
                term,
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            );
            Ok(Object::Literal(literal))
        } else if term.parse::<f64>().is_ok() {
            // Decimal literal
            let literal = Literal::new_typed_literal(
                term,
                NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal").unwrap(),
            );
            Ok(Object::Literal(literal))
        } else {
            // Treat as simple string literal
            let literal = Literal::new_simple_literal(term);
            Ok(Object::Literal(literal))
        }
    }

    /// Parse a simple string literal
    fn parse_simple_literal(&self, term: &str) -> TurtleResult<Object> {
        if !term.starts_with('"') {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Literal must start with quote".to_string(),
                position: TextPosition::default(),
            }));
        }

        // Find the closing quote (simplified - doesn't handle all escape sequences)
        if let Some(end_pos) = term[1..].find('"') {
            let content = &term[1..end_pos + 1];
            let remainder = &term[end_pos + 2..];

            if remainder.is_empty() {
                // Simple string literal
                let literal = Literal::new_simple_literal(content);
                Ok(Object::Literal(literal))
            } else if let Some(lang) = remainder.strip_prefix('@') {
                // Language-tagged literal
                let literal = Literal::new_language_tagged_literal(content, lang).map_err(|e| {
                    TurtleParseError::syntax(TurtleSyntaxError::Generic {
                        message: format!("Invalid language tag: {e}"),
                        position: TextPosition::default(),
                    })
                })?;
                Ok(Object::Literal(literal))
            } else if let Some(datatype_str) = remainder.strip_prefix("^^") {
                // Typed literal
                let datatype_iri = if datatype_str.starts_with('<') && datatype_str.ends_with('>') {
                    &datatype_str[1..datatype_str.len() - 1]
                } else {
                    datatype_str
                };
                let datatype = NamedNode::new(datatype_iri).map_err(TurtleParseError::model)?;
                let literal = Literal::new_typed_literal(content, datatype);
                Ok(Object::Literal(literal))
            } else {
                // Unknown suffix, treat as simple literal
                let literal = Literal::new_simple_literal(&term[1..end_pos + 1]);
                Ok(Object::Literal(literal))
            }
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Unterminated string literal".to_string(),
                position: TextPosition::default(),
            }))
        }
    }

    /// Expand a prefixed name to a full IRI
    fn expand_prefixed_name(
        &self,
        prefixed: &str,
        prefixes: &HashMap<String, String>,
    ) -> TurtleResult<String> {
        if let Some(colon_pos) = prefixed.find(':') {
            let prefix = &prefixed[..colon_pos];
            let local = &prefixed[colon_pos + 1..];

            if let Some(namespace) = prefixes.get(prefix) {
                Ok(format!("{namespace}{local}"))
            } else {
                Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: format!("Unknown prefix: {prefix}"),
                    position: TextPosition::default(),
                }))
            }
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid prefixed name: {prefixed}"),
                position: TextPosition::default(),
            }))
        }
    }
}

impl Parser<Triple> for N3Parser {
    fn parse<R: Read>(&self, reader: R) -> TurtleResult<Vec<Triple>> {
        let buf_reader = BufReader::new(reader);
        self.parse_n3_content(buf_reader)
    }

    fn for_reader<R: BufRead>(&self, reader: R) -> Box<dyn Iterator<Item = TurtleResult<Triple>>> {
        // For streaming, return all triples at once for now
        // A proper implementation would use a streaming parser
        match self.parse_n3_content(reader) {
            Ok(triples) => Box::new(triples.into_iter().map(Ok)),
            Err(e) => Box::new(std::iter::once(Err(e))),
        }
    }
}

/// N3 streaming iterator
pub struct N3Iterator {
    triples: std::vec::IntoIter<Triple>,
}

impl N3Iterator {
    pub fn new<R: BufRead>(reader: R, parser: &N3Parser) -> TurtleResult<Self> {
        let triples = parser.parse_n3_content(reader)?;
        Ok(Self {
            triples: triples.into_iter(),
        })
    }
}

impl Iterator for N3Iterator {
    type Item = TurtleResult<Triple>;

    fn next(&mut self) -> Option<Self::Item> {
        self.triples.next().map(Ok)
    }
}
