//! Unified RDF Parser Interface
//!
//! Provides a consistent API for parsing all supported RDF formats.
//! Extracted and adapted from OxiGraph with OxiRS enhancements.

use super::error::{ParseResult, RdfParseError, RdfSyntaxError, TextPosition};
use super::format::RdfFormat;
use super::n3_lexer::N3Token;
use crate::model::{Literal, Object, Predicate, Quad, Subject, Triple};
use crate::GraphName;
use std::collections::HashMap;
use std::io::Read;

/// Result type for quad parsing operations
pub type QuadParseResult = ParseResult<Quad>;

/// Result type for triple parsing operations  
pub type TripleParseResult = ParseResult<Triple>;

/// Iterator over parsed quads from a reader
pub struct ReaderQuadParser<R: Read> {
    inner: Box<dyn Iterator<Item = QuadParseResult> + Send>,
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Read> ReaderQuadParser<R> {
    /// Create a new reader parser
    pub fn new(iter: Box<dyn Iterator<Item = QuadParseResult> + Send>) -> Self {
        Self {
            inner: iter,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<R: Read> Iterator for ReaderQuadParser<R> {
    type Item = QuadParseResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Iterator over parsed quads from a byte slice
pub struct SliceQuadParser<'a> {
    inner: Box<dyn Iterator<Item = QuadParseResult> + 'a>,
}

impl<'a> SliceQuadParser<'a> {
    /// Create a new slice parser
    pub fn new(iter: Box<dyn Iterator<Item = QuadParseResult> + 'a>) -> Self {
        Self { inner: iter }
    }
}

impl<'a> Iterator for SliceQuadParser<'a> {
    type Item = QuadParseResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Unified RDF parser supporting all formats
#[derive(Debug, Clone)]
pub struct RdfParser {
    format: RdfFormat,
    base_iri: Option<String>,
    prefixes: HashMap<String, String>,
    lenient: bool,
}

impl RdfParser {
    /// Create a new parser for the specified format
    pub fn new(format: RdfFormat) -> Self {
        Self {
            format,
            base_iri: None,
            prefixes: HashMap::new(),
            lenient: false,
        }
    }

    /// Set the base IRI for resolving relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        self.base_iri = Some(base_iri.into());
        self
    }

    /// Add a namespace prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.insert(prefix.into(), iri.into());
        self
    }

    /// Enable lenient parsing (skip some validations for performance)
    pub fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }

    /// Parse from a reader
    pub fn for_reader<R: Read + Send>(self, reader: R) -> ReaderQuadParser<R> {
        match self.format {
            RdfFormat::Turtle => self.parse_turtle_reader(reader),
            RdfFormat::NTriples => self.parse_ntriples_reader(reader),
            RdfFormat::NQuads => self.parse_nquads_reader(reader),
            RdfFormat::TriG => self.parse_trig_reader(reader),
            RdfFormat::RdfXml => self.parse_rdfxml_reader(reader),
            RdfFormat::JsonLd { .. } => self.parse_jsonld_reader(reader),
            RdfFormat::N3 => self.parse_n3_reader(reader),
        }
    }

    /// Parse from a byte slice
    pub fn for_slice<'a>(self, slice: &'a [u8]) -> SliceQuadParser<'a> {
        match self.format {
            RdfFormat::Turtle => self.parse_turtle_slice(slice),
            RdfFormat::NTriples => self.parse_ntriples_slice(slice),
            RdfFormat::NQuads => self.parse_nquads_slice(slice),
            RdfFormat::TriG => self.parse_trig_slice(slice),
            RdfFormat::RdfXml => self.parse_rdfxml_slice(slice),
            RdfFormat::JsonLd { .. } => self.parse_jsonld_slice(slice),
            RdfFormat::N3 => self.parse_n3_slice(slice),
        }
    }

    /// Get the format being parsed
    pub fn format(&self) -> RdfFormat {
        self.format.clone()
    }

    /// Get the base IRI
    pub fn base_iri(&self) -> Option<&str> {
        self.base_iri.as_deref()
    }

    /// Get the prefixes
    pub fn prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }

    /// Check if lenient parsing is enabled
    pub fn is_lenient(&self) -> bool {
        self.lenient
    }

    /// Parse N-Quads content
    fn parse_nquads_content(&self, content: &str) -> ParseResult<Vec<Quad>> {
        use crate::model::{BlankNode, NamedNode, Quad};

        let mut quads = Vec::new();
        let mut line_number = 1;

        for line in content.lines() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') {
                line_number += 1;
                continue;
            }

            // Parse quad: <s> <p> <o> <g> .
            if !trimmed.ends_with('.') {
                if !self.lenient {
                    return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        "N-Quads line must end with '.'".to_string(),
                        TextPosition::new(line_number, trimmed.len(), 0),
                    )));
                }
                line_number += 1;
                continue;
            }

            let line_without_dot = trimmed[..trimmed.len() - 1].trim();
            let parts: Vec<&str> = line_without_dot.split_whitespace().collect();

            if parts.len() < 3 || parts.len() > 4 {
                if !self.lenient {
                    return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        "N-Quads line must have 3 or 4 terms".to_string(),
                        TextPosition::new(line_number, 1, 0),
                    )));
                }
                line_number += 1;
                continue;
            }

            // Parse subject
            let subject = if parts[0].starts_with('<') && parts[0].ends_with('>') {
                let iri = &parts[0][1..parts[0].len() - 1];
                Subject::NamedNode(NamedNode::new(iri)?)
            } else if parts[0].starts_with("_:") {
                let label = &parts[0][2..];
                Subject::BlankNode(BlankNode::new(label)?)
            } else {
                if !self.lenient {
                    return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        "Invalid subject format".to_string(),
                        TextPosition::new(line_number, 1, 0),
                    )));
                }
                line_number += 1;
                continue;
            };

            // Parse predicate
            let predicate = if parts[1].starts_with('<') && parts[1].ends_with('>') {
                let iri = &parts[1][1..parts[1].len() - 1];
                Predicate::NamedNode(NamedNode::new(iri)?)
            } else {
                if !self.lenient {
                    return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        "Invalid predicate format".to_string(),
                        TextPosition::new(line_number, 1, 0),
                    )));
                }
                line_number += 1;
                continue;
            };

            // Parse object
            let object = if parts[2].starts_with('<') && parts[2].ends_with('>') {
                let iri = &parts[2][1..parts[2].len() - 1];
                Object::NamedNode(NamedNode::new(iri)?)
            } else if parts[2].starts_with("_:") {
                let label = &parts[2][2..];
                Object::BlankNode(BlankNode::new(label)?)
            } else if parts[2].starts_with('"') {
                let literal_str = parts[2];
                let literal = self.parse_literal_from_nquads(literal_str)?;
                Object::Literal(literal)
            } else {
                if !self.lenient {
                    return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        "Invalid object format".to_string(),
                        TextPosition::new(line_number, 1, 0),
                    )));
                }
                line_number += 1;
                continue;
            };

            // Parse graph (optional)
            let graph = if parts.len() == 4 {
                if parts[3].starts_with('<') && parts[3].ends_with('>') {
                    let iri = &parts[3][1..parts[3].len() - 1];
                    GraphName::NamedNode(NamedNode::new(iri)?)
                } else {
                    if !self.lenient {
                        return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                            "Invalid graph format".to_string(),
                            TextPosition::new(line_number, 1, 0),
                        )));
                    }
                    GraphName::DefaultGraph
                }
            } else {
                GraphName::DefaultGraph
            };

            let quad = Quad::new(subject, predicate, object, graph);
            quads.push(quad);
            line_number += 1;
        }

        Ok(quads)
    }

    /// Parse TriG content
    fn parse_trig_content(&self, content: &str) -> ParseResult<Vec<Quad>> {
        use crate::format::n3_lexer::{N3Lexer, N3Token};
        use crate::format::toolkit::{StringBuffer, TokenRecognizer};
        use crate::model::{NamedNode, Quad};

        let mut buffer = StringBuffer::new(content.to_string());
        let mut lexer = N3Lexer::new();
        let mut quads = Vec::new();
        let mut prefixes: HashMap<String, String> = HashMap::new();
        let mut current_graph = GraphName::DefaultGraph;

        // Tokenize the input
        let mut tokens = Vec::new();
        loop {
            match lexer.recognize_next_token(&mut buffer, &mut TextPosition::start())? {
                Some(N3Token::Eof) => break,
                Some(token) => tokens.push(token),
                None => break,
            }
        }

        let mut i = 0;
        while i < tokens.len() {
            match &tokens[i] {
                N3Token::Prefix => {
                    // Handle @prefix directive
                    if i + 3 < tokens.len() {
                        if let (
                            N3Token::PrefixedName {
                                prefix: Some(prefix),
                                ..
                            },
                            N3Token::Iri(iri),
                        ) = (&tokens[i + 1], &tokens[i + 2])
                        {
                            prefixes.insert(prefix.clone(), iri.clone());
                            i += 4; // Skip prefix, name, IRI, dot
                            continue;
                        }
                    }
                    i += 1;
                }
                N3Token::Base => {
                    // Handle @base directive - skip for now
                    i += 3; // Skip base, IRI, dot
                }
                N3Token::Iri(graph_iri) => {
                    // Named graph block
                    current_graph = GraphName::NamedNode(NamedNode::new(graph_iri)?);
                    i += 1;
                    if i < tokens.len() && tokens[i] == N3Token::LeftBrace {
                        i += 1; // Skip {
                                // Parse triples in this graph
                        while i < tokens.len() && tokens[i] != N3Token::RightBrace {
                            if let Some(triple) =
                                self.parse_trig_triple(&tokens, &mut i, &prefixes)?
                            {
                                let quad = Quad::new(
                                    triple.subject().clone(),
                                    triple.predicate().clone(),
                                    triple.object().clone(),
                                    current_graph.clone(),
                                );
                                quads.push(quad);
                            }
                        }
                        if i < tokens.len() {
                            i += 1; // Skip }
                        }
                    }
                }
                N3Token::LeftBrace => {
                    // Anonymous graph block
                    current_graph = GraphName::DefaultGraph;
                    i += 1;
                    while i < tokens.len() && tokens[i] != N3Token::RightBrace {
                        if let Some(triple) = self.parse_trig_triple(&tokens, &mut i, &prefixes)? {
                            let quad = Quad::new(
                                triple.subject().clone(),
                                triple.predicate().clone(),
                                triple.object().clone(),
                                current_graph.clone(),
                            );
                            quads.push(quad);
                        }
                    }
                    if i < tokens.len() {
                        i += 1; // Skip }
                    }
                }
                _ => {
                    // Parse triple in default graph
                    if let Some(triple) = self.parse_trig_triple(&tokens, &mut i, &prefixes)? {
                        let quad = Quad::new(
                            triple.subject().clone(),
                            triple.predicate().clone(),
                            triple.object().clone(),
                            current_graph.clone(),
                        );
                        quads.push(quad);
                    }
                }
            }
        }

        Ok(quads)
    }

    /// Parse N3 content
    fn parse_n3_content(&self, content: &str) -> ParseResult<Vec<Quad>> {
        use crate::format::n3_lexer::{N3Lexer, N3Token};
        use crate::format::toolkit::{StringBuffer, TokenRecognizer};
        use crate::model::Quad;

        let mut buffer = StringBuffer::new(content.to_string());
        let mut lexer = N3Lexer::new();
        let mut quads = Vec::new();
        let mut prefixes: HashMap<String, String> = HashMap::new();

        // Tokenize the input
        let mut tokens = Vec::new();
        loop {
            match lexer.recognize_next_token(&mut buffer, &mut TextPosition::start())? {
                Some(N3Token::Eof) => break,
                Some(token) => tokens.push(token),
                None => break,
            }
        }

        let mut i = 0;
        while i < tokens.len() {
            match &tokens[i] {
                N3Token::Prefix => {
                    // Handle @prefix directive
                    if i + 3 < tokens.len() {
                        if let (
                            N3Token::PrefixedName {
                                prefix: Some(prefix),
                                ..
                            },
                            N3Token::Iri(iri),
                        ) = (&tokens[i + 1], &tokens[i + 2])
                        {
                            prefixes.insert(prefix.clone(), iri.clone());
                            i += 4; // Skip prefix, name, IRI, dot
                            continue;
                        }
                    }
                    i += 1;
                }
                N3Token::Base => {
                    // Handle @base directive - skip for now
                    i += 3; // Skip base, IRI, dot
                }
                _ => {
                    // Parse triple
                    if let Some(triple) = self.parse_n3_triple(&tokens, &mut i, &prefixes)? {
                        let quad = Quad::new(
                            triple.subject().clone(),
                            triple.predicate().clone(),
                            triple.object().clone(),
                            GraphName::DefaultGraph,
                        );
                        quads.push(quad);
                    }
                }
            }
        }

        Ok(quads)
    }

    /// Helper to parse literal from N-Quads format
    fn parse_literal_from_nquads(&self, literal_str: &str) -> ParseResult<Literal> {
        use crate::model::{Literal, NamedNode};

        // Simple literal parsing for N-Quads
        if let Some(quote_end) = literal_str[1..].find('"') {
            let value = &literal_str[1..quote_end + 1];
            let remainder = &literal_str[quote_end + 2..];

            if let Some(lang) = remainder.strip_prefix('@') {
                // Language literal
                Literal::new_language_tagged_literal(value, lang).map_err(|e| {
                    RdfParseError::Syntax(RdfSyntaxError::with_position(
                        format!("Invalid language literal: {e}"),
                        TextPosition::start(),
                    ))
                })
            } else if remainder.starts_with("^^") && remainder.len() > 3 {
                // Typed literal
                let datatype_str = &remainder[3..remainder.len() - 1]; // Remove ^^< and >
                let datatype = NamedNode::new(datatype_str).map_err(|e| {
                    RdfParseError::Syntax(RdfSyntaxError::with_position(
                        format!("Invalid datatype: {e}"),
                        TextPosition::start(),
                    ))
                })?;
                Ok(Literal::new_typed_literal(value, datatype))
            } else {
                // Simple literal
                Ok(Literal::new(value))
            }
        } else {
            Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Invalid literal format".to_string(),
                TextPosition::start(),
            )))
        }
    }

    /// Helper to parse TriG triple
    fn parse_trig_triple(
        &self,
        tokens: &[N3Token],
        i: &mut usize,
        prefixes: &HashMap<String, String>,
    ) -> ParseResult<Option<Triple>> {
        use crate::model::Triple;

        if *i + 2 >= tokens.len() {
            return Ok(None);
        }

        // Parse subject (may contain quoted triples)
        let subject = self.token_to_subject_with_quoted(tokens, i, prefixes)?;
        if *i < tokens.len()
            && !matches!(
                tokens[*i],
                N3Token::QuotedTripleStart | N3Token::QuotedTripleEnd
            )
        {
            *i += 1;
        }

        // Parse predicate
        let predicate = self.token_to_predicate(&tokens[*i], prefixes)?;
        *i += 1;

        // Parse object (may contain quoted triples)
        let object = self.token_to_object_with_quoted(tokens, i, prefixes)?;
        if *i < tokens.len()
            && !matches!(
                tokens[*i],
                N3Token::QuotedTripleStart | N3Token::QuotedTripleEnd | N3Token::Dot
            )
        {
            *i += 1;
        }

        // Skip dot if present
        if *i < tokens.len() && tokens[*i] == N3Token::Dot {
            *i += 1;
        }

        Ok(Some(Triple::new(subject, predicate, object)))
    }

    /// Helper to parse N3 triple
    fn parse_n3_triple(
        &self,
        tokens: &[N3Token],
        i: &mut usize,
        prefixes: &HashMap<String, String>,
    ) -> ParseResult<Option<Triple>> {
        // N3 parsing is similar to TriG but without graph support
        self.parse_trig_triple(tokens, i, prefixes)
    }

    /// Convert N3Token to Subject
    #[allow(dead_code)]
    fn token_to_subject(
        &self,
        token: &N3Token,
        prefixes: &HashMap<String, String>,
    ) -> ParseResult<Subject> {
        use crate::model::{BlankNode, NamedNode};

        match token {
            N3Token::Iri(iri) => Ok(Subject::NamedNode(NamedNode::new(iri)?)),
            N3Token::PrefixedName { prefix, local } => {
                let full_iri = if let Some(prefix_str) = prefix {
                    if let Some(namespace) = prefixes.get(prefix_str) {
                        format!("{namespace}{local}")
                    } else {
                        return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                            format!("Unknown prefix: {prefix_str}"),
                            TextPosition::start(),
                        )));
                    }
                } else {
                    local.clone() // Default namespace
                };
                Ok(Subject::NamedNode(NamedNode::new(full_iri)?))
            }
            N3Token::BlankNode(label) => Ok(Subject::BlankNode(BlankNode::new(label)?)),
            _ => Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Invalid subject token".to_string(),
                TextPosition::start(),
            ))),
        }
    }

    /// Convert N3Token to Subject with quoted triple support
    fn token_to_subject_with_quoted(
        &self,
        tokens: &[N3Token],
        i: &mut usize,
        prefixes: &HashMap<String, String>,
    ) -> ParseResult<Subject> {
        use crate::model::{BlankNode, NamedNode};

        match &tokens[*i] {
            N3Token::Iri(iri) => Ok(Subject::NamedNode(NamedNode::new(iri)?)),
            N3Token::PrefixedName { prefix, local } => {
                let full_iri = if let Some(prefix_str) = prefix {
                    if let Some(namespace) = prefixes.get(prefix_str) {
                        format!("{namespace}{local}")
                    } else {
                        return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                            format!("Unknown prefix: {prefix_str}"),
                            TextPosition::start(),
                        )));
                    }
                } else {
                    local.clone() // Default namespace
                };
                Ok(Subject::NamedNode(NamedNode::new(full_iri)?))
            }
            N3Token::BlankNode(label) => Ok(Subject::BlankNode(BlankNode::new(label)?)),
            N3Token::QuotedTripleStart => {
                let quoted_triple = self.parse_quoted_triple(tokens, i, prefixes)?;
                Ok(Subject::QuotedTriple(Box::new(quoted_triple)))
            }
            _ => Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Invalid subject token".to_string(),
                TextPosition::start(),
            ))),
        }
    }

    /// Convert N3Token to Predicate
    fn token_to_predicate(
        &self,
        token: &N3Token,
        prefixes: &HashMap<String, String>,
    ) -> ParseResult<Predicate> {
        use crate::model::NamedNode;

        match token {
            N3Token::Iri(iri) => Ok(Predicate::NamedNode(NamedNode::new(iri)?)),
            N3Token::PrefixedName { prefix, local } => {
                let full_iri = if let Some(prefix_str) = prefix {
                    if let Some(namespace) = prefixes.get(prefix_str) {
                        format!("{namespace}{local}")
                    } else {
                        return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                            format!("Unknown prefix: {prefix_str}"),
                            TextPosition::start(),
                        )));
                    }
                } else {
                    local.clone() // Default namespace
                };
                Ok(Predicate::NamedNode(NamedNode::new(full_iri)?))
            }
            N3Token::A => {
                // 'a' is shorthand for rdf:type
                Ok(Predicate::NamedNode(NamedNode::new(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                )?))
            }
            _ => Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Invalid predicate token".to_string(),
                TextPosition::start(),
            ))),
        }
    }

    /// Convert N3Token to Object
    #[allow(dead_code)]
    fn token_to_object(
        &self,
        token: &N3Token,
        prefixes: &HashMap<String, String>,
    ) -> ParseResult<Object> {
        use crate::model::{BlankNode, Literal, NamedNode};

        match token {
            N3Token::Iri(iri) => Ok(Object::NamedNode(NamedNode::new(iri)?)),
            N3Token::PrefixedName { prefix, local } => {
                let full_iri = if let Some(prefix_str) = prefix {
                    if let Some(namespace) = prefixes.get(prefix_str) {
                        format!("{namespace}{local}")
                    } else {
                        return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                            format!("Unknown prefix: {prefix_str}"),
                            TextPosition::start(),
                        )));
                    }
                } else {
                    local.clone() // Default namespace
                };
                Ok(Object::NamedNode(NamedNode::new(full_iri)?))
            }
            N3Token::BlankNode(label) => Ok(Object::BlankNode(BlankNode::new(label)?)),
            N3Token::Literal {
                value,
                datatype,
                language,
            } => {
                let literal = if let Some(lang) = language {
                    Literal::new_language_tagged_literal(value, lang)?
                } else if let Some(dt) = datatype {
                    let datatype_node = NamedNode::new(dt)?;
                    Literal::new_typed_literal(value, datatype_node)
                } else {
                    Literal::new(value)
                };
                Ok(Object::Literal(literal))
            }
            N3Token::Integer(value) => {
                let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    value.to_string(),
                    datatype,
                )))
            }
            N3Token::Decimal(value) => {
                let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal")?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    value.to_string(),
                    datatype,
                )))
            }
            N3Token::Double(value) => {
                let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#double")?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    value.to_string(),
                    datatype,
                )))
            }
            N3Token::True => {
                let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    "true", datatype,
                )))
            }
            N3Token::False => {
                let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    "false", datatype,
                )))
            }
            N3Token::QuotedTripleStart => {
                // This should be handled by a separate quoted triple parser
                Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                    "Quoted triple parsing not handled in object context".to_string(),
                    TextPosition::start(),
                )))
            }
            _ => Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Invalid object token".to_string(),
                TextPosition::start(),
            ))),
        }
    }

    /// Convert N3Token to Object with quoted triple support
    fn token_to_object_with_quoted(
        &self,
        tokens: &[N3Token],
        i: &mut usize,
        prefixes: &HashMap<String, String>,
    ) -> ParseResult<Object> {
        use crate::model::{BlankNode, Literal, NamedNode};

        match &tokens[*i] {
            N3Token::Iri(iri) => Ok(Object::NamedNode(NamedNode::new(iri)?)),
            N3Token::PrefixedName { prefix, local } => {
                let full_iri = if let Some(prefix_str) = prefix {
                    if let Some(namespace) = prefixes.get(prefix_str) {
                        format!("{namespace}{local}")
                    } else {
                        return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                            format!("Unknown prefix: {prefix_str}"),
                            TextPosition::start(),
                        )));
                    }
                } else {
                    local.clone() // Default namespace
                };
                Ok(Object::NamedNode(NamedNode::new(full_iri)?))
            }
            N3Token::BlankNode(label) => Ok(Object::BlankNode(BlankNode::new(label)?)),
            N3Token::Literal {
                value,
                datatype,
                language,
            } => {
                let literal = if let Some(lang) = language {
                    Literal::new_language_tagged_literal(value, lang)?
                } else if let Some(dt) = datatype {
                    let datatype_node = NamedNode::new(dt)?;
                    Literal::new_typed_literal(value, datatype_node)
                } else {
                    Literal::new(value)
                };
                Ok(Object::Literal(literal))
            }
            N3Token::Integer(value) => {
                let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    value.to_string(),
                    datatype,
                )))
            }
            N3Token::Decimal(value) => {
                let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal")?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    value.to_string(),
                    datatype,
                )))
            }
            N3Token::Double(value) => {
                let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#double")?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    value.to_string(),
                    datatype,
                )))
            }
            N3Token::True => {
                let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    "true", datatype,
                )))
            }
            N3Token::False => {
                let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    "false", datatype,
                )))
            }
            N3Token::QuotedTripleStart => {
                let quoted_triple = self.parse_quoted_triple(tokens, i, prefixes)?;
                Ok(Object::QuotedTriple(Box::new(quoted_triple)))
            }
            _ => Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Invalid object token".to_string(),
                TextPosition::start(),
            ))),
        }
    }

    /// Parse a quoted triple from tokens
    #[allow(dead_code)]
    fn parse_quoted_triple(
        &self,
        tokens: &[N3Token],
        i: &mut usize,
        prefixes: &HashMap<String, String>,
    ) -> ParseResult<crate::model::star::QuotedTriple> {
        use crate::model::{star::QuotedTriple, Triple};

        // Expect QuotedTripleStart
        if *i >= tokens.len() || tokens[*i] != N3Token::QuotedTripleStart {
            return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Expected quoted triple start <<".to_string(),
                TextPosition::start(),
            )));
        }
        *i += 1;

        // Parse subject (may contain nested quoted triples)
        let subject = self.token_to_subject_with_quoted(tokens, i, prefixes)?;
        if *i < tokens.len() && tokens[*i] != N3Token::QuotedTripleStart {
            *i += 1;
        }

        // Parse predicate
        let predicate = self.token_to_predicate(&tokens[*i], prefixes)?;
        *i += 1;

        // Parse object (may contain nested quoted triples)
        let object = self.token_to_object_with_quoted(tokens, i, prefixes)?;
        if *i < tokens.len() && tokens[*i] != N3Token::QuotedTripleEnd {
            *i += 1;
        }

        // Expect QuotedTripleEnd
        if *i >= tokens.len() || tokens[*i] != N3Token::QuotedTripleEnd {
            return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Expected quoted triple end >>".to_string(),
                TextPosition::start(),
            )));
        }
        *i += 1;

        let triple = Triple::new(subject, predicate, object);
        Ok(QuotedTriple::new(triple))
    }

    // Format-specific parser implementations

    fn parse_turtle_reader<R: Read + Send>(self, _reader: R) -> ReaderQuadParser<R> {
        // TODO: Implement actual Turtle parsing using extracted oxttl components
        // For now, return empty iterator
        ReaderQuadParser::new(Box::new(std::iter::empty()))
    }

    fn parse_turtle_slice<'a>(self, _slice: &'a [u8]) -> SliceQuadParser<'a> {
        // TODO: Implement actual Turtle parsing using extracted oxttl components
        SliceQuadParser::new(Box::new(std::iter::empty()))
    }

    fn parse_ntriples_reader<R: Read + Send>(self, _reader: R) -> ReaderQuadParser<R> {
        // TODO: Implement N-Triples parsing
        ReaderQuadParser::new(Box::new(std::iter::empty()))
    }

    fn parse_ntriples_slice<'a>(self, _slice: &'a [u8]) -> SliceQuadParser<'a> {
        // TODO: Implement N-Triples parsing
        SliceQuadParser::new(Box::new(std::iter::empty()))
    }

    fn parse_nquads_reader<R: Read + Send>(self, reader: R) -> ReaderQuadParser<R> {
        let mut buf_reader = std::io::BufReader::new(reader);
        let mut quads = Vec::new();

        // Read all lines and parse
        let mut content = String::new();
        if buf_reader.read_to_string(&mut content).is_ok() {
            if let Ok(parsed_quads) = self.parse_nquads_content(&content) {
                quads = parsed_quads;
            }
        }

        ReaderQuadParser::new(Box::new(quads.into_iter().map(Ok)))
    }

    fn parse_nquads_slice<'a>(self, slice: &'a [u8]) -> SliceQuadParser<'a> {
        let content = std::str::from_utf8(slice).unwrap_or("");
        let quads = self.parse_nquads_content(content).unwrap_or_default();
        SliceQuadParser::new(Box::new(quads.into_iter().map(Ok)))
    }

    fn parse_trig_reader<R: Read + Send>(self, reader: R) -> ReaderQuadParser<R> {
        let mut buf_reader = std::io::BufReader::new(reader);
        let mut quads = Vec::new();

        // Read all content and parse
        let mut content = String::new();
        if buf_reader.read_to_string(&mut content).is_ok() {
            if let Ok(parsed_quads) = self.parse_trig_content(&content) {
                quads = parsed_quads;
            }
        }

        ReaderQuadParser::new(Box::new(quads.into_iter().map(Ok)))
    }

    fn parse_trig_slice<'a>(self, slice: &'a [u8]) -> SliceQuadParser<'a> {
        let content = std::str::from_utf8(slice).unwrap_or("");
        let quads = self.parse_trig_content(content).unwrap_or_default();
        SliceQuadParser::new(Box::new(quads.into_iter().map(Ok)))
    }

    fn parse_rdfxml_reader<R: Read + Send>(self, _reader: R) -> ReaderQuadParser<R> {
        // TODO: Implement RDF/XML parsing using extracted oxrdfxml components
        ReaderQuadParser::new(Box::new(std::iter::empty()))
    }

    fn parse_rdfxml_slice<'a>(self, _slice: &'a [u8]) -> SliceQuadParser<'a> {
        // TODO: Implement RDF/XML parsing
        SliceQuadParser::new(Box::new(std::iter::empty()))
    }

    fn parse_jsonld_reader<R: Read + Send>(self, _reader: R) -> ReaderQuadParser<R> {
        // TODO: Implement JSON-LD parsing using extracted oxjsonld components
        ReaderQuadParser::new(Box::new(std::iter::empty()))
    }

    fn parse_jsonld_slice<'a>(self, _slice: &'a [u8]) -> SliceQuadParser<'a> {
        // TODO: Implement JSON-LD parsing
        SliceQuadParser::new(Box::new(std::iter::empty()))
    }

    fn parse_n3_reader<R: Read + Send>(self, reader: R) -> ReaderQuadParser<R> {
        let mut buf_reader = std::io::BufReader::new(reader);
        let mut quads = Vec::new();

        // Read all content and parse
        let mut content = String::new();
        if buf_reader.read_to_string(&mut content).is_ok() {
            if let Ok(parsed_quads) = self.parse_n3_content(&content) {
                quads = parsed_quads;
            }
        }

        ReaderQuadParser::new(Box::new(quads.into_iter().map(Ok)))
    }

    fn parse_n3_slice<'a>(self, slice: &'a [u8]) -> SliceQuadParser<'a> {
        let content = std::str::from_utf8(slice).unwrap_or("");
        let quads = self.parse_n3_content(content).unwrap_or_default();
        SliceQuadParser::new(Box::new(quads.into_iter().map(Ok)))
    }
}

impl Default for RdfParser {
    fn default() -> Self {
        Self::new(RdfFormat::default())
    }
}

/// Parsing configuration for fine-grained control
#[derive(Debug, Clone)]
pub struct ParseConfig {
    /// Maximum number of triples/quads to parse (None = unlimited)
    pub max_items: Option<usize>,
    /// Enable parallel parsing for large files
    pub parallel: bool,
    /// Number of parallel threads (None = auto-detect)
    pub thread_count: Option<usize>,
    /// Buffer size for streaming parsing
    pub buffer_size: usize,
    /// Validate IRIs strictly
    pub strict_iri_validation: bool,
    /// Validate literals strictly
    pub strict_literal_validation: bool,
    /// Continue parsing on recoverable errors
    pub continue_on_error: bool,
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            max_items: None,
            parallel: false,
            thread_count: None,
            buffer_size: 8192,
            strict_iri_validation: true,
            strict_literal_validation: true,
            continue_on_error: false,
        }
    }
}

/// Advanced parser with configuration support
pub struct ConfigurableParser {
    parser: RdfParser,
    config: ParseConfig,
}

impl ConfigurableParser {
    /// Create a new configurable parser
    pub fn new(format: RdfFormat, config: ParseConfig) -> Self {
        Self {
            parser: RdfParser::new(format),
            config,
        }
    }

    /// Set base IRI
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        self.parser = self.parser.with_base_iri(base_iri);
        self
    }

    /// Add prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.parser = self.parser.with_prefix(prefix, iri);
        self
    }

    /// Parse with configuration
    pub fn parse_slice<'a>(&self, slice: &'a [u8]) -> SliceQuadParser<'a> {
        // Apply configuration settings and parse
        let mut parser = self.parser.clone();

        if !self.config.strict_iri_validation || !self.config.strict_literal_validation {
            parser = parser.lenient();
        }

        // TODO: Apply other configuration options
        parser.for_slice(slice)
    }

    /// Get the configuration
    pub fn config(&self) -> &ParseConfig {
        &self.config
    }

    /// Get the parser
    pub fn parser(&self) -> &RdfParser {
        &self.parser
    }
}

/// Simple parsing functions for common use cases
pub mod simple {
    use super::*;

    /// Parse triples from a string in the specified format
    pub fn parse_triples_from_str(input: &str, format: RdfFormat) -> ParseResult<Vec<Triple>> {
        let parser = RdfParser::new(format);
        let mut triples = Vec::new();

        for quad_result in parser.for_slice(input.as_bytes()) {
            let quad = quad_result?;
            if let Some(triple) = quad.triple_in_default_graph() {
                triples.push(triple);
            }
        }

        Ok(triples)
    }

    /// Parse quads from a string in the specified format
    pub fn parse_quads_from_str(input: &str, format: RdfFormat) -> ParseResult<Vec<Quad>> {
        let parser = RdfParser::new(format);
        let mut quads = Vec::new();

        for quad_result in parser.for_slice(input.as_bytes()) {
            quads.push(quad_result?);
        }

        Ok(quads)
    }

    /// Parse triples from Turtle string
    pub fn parse_turtle(input: &str) -> ParseResult<Vec<Triple>> {
        parse_triples_from_str(input, RdfFormat::Turtle)
    }

    /// Parse triples from N-Triples string
    pub fn parse_ntriples(input: &str) -> ParseResult<Vec<Triple>> {
        parse_triples_from_str(input, RdfFormat::NTriples)
    }

    /// Parse quads from N-Quads string
    pub fn parse_nquads(input: &str) -> ParseResult<Vec<Quad>> {
        parse_quads_from_str(input, RdfFormat::NQuads)
    }

    /// Parse quads from TriG string
    pub fn parse_trig(input: &str) -> ParseResult<Vec<Quad>> {
        parse_quads_from_str(input, RdfFormat::TriG)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = RdfParser::new(RdfFormat::Turtle);
        assert_eq!(parser.format(), RdfFormat::Turtle);
        assert!(parser.base_iri().is_none());
        assert!(parser.prefixes().is_empty());
        assert!(!parser.is_lenient());
    }

    #[test]
    fn test_parser_configuration() {
        let parser = RdfParser::new(RdfFormat::Turtle)
            .with_base_iri("http://example.org/")
            .with_prefix("ex", "http://example.org/ns#")
            .lenient();

        assert_eq!(parser.base_iri(), Some("http://example.org/"));
        assert_eq!(
            parser.prefixes().get("ex"),
            Some(&"http://example.org/ns#".to_string())
        );
        assert!(parser.is_lenient());
    }

    #[test]
    fn test_configurable_parser() {
        let config = ParseConfig {
            max_items: Some(1000),
            parallel: true,
            ..Default::default()
        };

        let parser = ConfigurableParser::new(RdfFormat::NQuads, config);
        assert_eq!(parser.config().max_items, Some(1000));
        assert!(parser.config().parallel);
    }

    #[test]
    fn test_parse_config_default() {
        let config = ParseConfig::default();
        assert_eq!(config.max_items, None);
        assert!(!config.parallel);
        assert_eq!(config.buffer_size, 8192);
        assert!(config.strict_iri_validation);
        assert!(config.strict_literal_validation);
        assert!(!config.continue_on_error);
    }
}
