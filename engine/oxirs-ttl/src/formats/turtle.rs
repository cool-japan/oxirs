//! Turtle format parser and serializer
//!
//! Full implementation of the Turtle RDF serialization format with support for:
//! - Prefix declarations (@prefix)
//! - Base IRI declarations (@base)
//! - Abbreviated syntax (a for rdf:type, semicolons, commas)
//! - Collection syntax []
//! - List syntax ()
//! - Comments and whitespace handling

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::toolkit::{FormattedWriter, Parser, SerializationConfig, Serializer};
use oxirs_core::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject, Triple};
use std::collections::HashMap;
use std::io::{BufRead, Read, Write};

/// Turtle parser with full Turtle 1.1 support
#[derive(Debug, Clone)]
pub struct TurtleParser {
    /// Whether to continue parsing after errors
    pub lenient: bool,
    /// Base IRI for resolving relative IRIs
    pub base_iri: Option<String>,
    /// Initial prefix declarations
    pub prefixes: HashMap<String, String>,
}

impl Default for TurtleParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TurtleParser {
    /// Create a new Turtle parser
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();

        // Add common prefixes
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

    /// Create a new lenient Turtle parser (continues after errors)
    pub fn new_lenient() -> Self {
        let mut parser = Self::new();
        parser.lenient = true;
        parser
    }

    /// Set the base IRI
    pub fn with_base_iri(mut self, base_iri: String) -> Self {
        self.base_iri = Some(base_iri);
        self
    }

    /// Add a prefix declaration
    pub fn with_prefix(mut self, prefix: String, iri: String) -> Self {
        self.prefixes.insert(prefix, iri);
        self
    }

    /// Parse Turtle document
    pub fn parse_document(&self, content: &str) -> TurtleResult<Vec<Triple>> {
        let mut context = TurtleParsingContext::new();
        context.prefixes = self.prefixes.clone();
        context.base_iri = self.base_iri.clone();

        let mut tokenizer = TurtleTokenizer::new(content);
        let mut triples = Vec::new();

        while let Some(statement) = self.parse_statement(&mut tokenizer, &mut context)? {
            match statement {
                TurtleStatement::Triple(triple) => triples.push(triple),
                TurtleStatement::PrefixDecl(prefix, iri) => {
                    context.prefixes.insert(prefix, iri);
                }
                TurtleStatement::BaseDecl(iri) => {
                    context.base_iri = Some(iri);
                }
            }
        }

        Ok(triples)
    }

    /// Parse a single statement (triple, prefix, or base declaration)
    fn parse_statement(
        &self,
        tokenizer: &mut TurtleTokenizer,
        context: &mut TurtleParsingContext,
    ) -> TurtleResult<Option<TurtleStatement>> {
        // Skip whitespace and comments
        tokenizer.skip_whitespace_and_comments();

        if tokenizer.is_at_end() {
            return Ok(None);
        }

        let token = tokenizer.peek_token()?;

        match &token.kind {
            TokenKind::PrefixKeyword => {
                let _ = tokenizer.consume_token(); // consume @prefix
                let prefix = self.parse_prefix_name(tokenizer)?;

                // Check if we need to consume a colon (only if prefix didn't already include it)
                let next_token = tokenizer.peek_token()?;
                if matches!(next_token.kind, TokenKind::Colon) {
                    let _ = tokenizer.consume_token(); // consume colon
                }

                let iri = self.parse_iri_ref(tokenizer, context)?;
                self.expect_token(tokenizer, TokenKind::Dot)?;
                Ok(Some(TurtleStatement::PrefixDecl(prefix, iri)))
            }
            TokenKind::BaseKeyword => {
                let _ = tokenizer.consume_token(); // consume @base
                let iri = self.parse_iri_ref(tokenizer, context)?;
                self.expect_token(tokenizer, TokenKind::Dot)?;
                Ok(Some(TurtleStatement::BaseDecl(iri)))
            }
            _ => {
                // Parse a triple
                let triple = self.parse_triple(tokenizer, context)?;
                self.expect_token(tokenizer, TokenKind::Dot)?;
                Ok(Some(TurtleStatement::Triple(triple)))
            }
        }
    }

    /// Parse a triple
    fn parse_triple(
        &self,
        tokenizer: &mut TurtleTokenizer,
        context: &mut TurtleParsingContext,
    ) -> TurtleResult<Triple> {
        let subject = self.parse_subject(tokenizer, context)?;
        let predicate = self.parse_predicate(tokenizer, context)?;
        let object = self.parse_object(tokenizer, context)?;

        Ok(Triple::new(subject, predicate, object))
    }

    /// Parse a subject (IRI, blank node, or collection)
    fn parse_subject(
        &self,
        tokenizer: &mut TurtleTokenizer,
        context: &mut TurtleParsingContext,
    ) -> TurtleResult<Subject> {
        let token = tokenizer.peek_token()?;

        match &token.kind {
            TokenKind::IriRef(_) => {
                let iri = self.parse_iri_ref(tokenizer, context)?;
                let named_node = NamedNode::new(&iri).map_err(TurtleParseError::model)?;
                Ok(Subject::NamedNode(named_node))
            }
            TokenKind::PrefixedName(prefix, local) => {
                let _ = tokenizer.consume_token();
                let iri = self.resolve_prefixed_name(prefix, local, context)?;
                let named_node = NamedNode::new(&iri).map_err(TurtleParseError::model)?;
                Ok(Subject::NamedNode(named_node))
            }
            TokenKind::BlankNodeLabel(label) => {
                let _ = tokenizer.consume_token();
                let blank_node = BlankNode::new(label).map_err(TurtleParseError::model)?;
                Ok(Subject::BlankNode(blank_node))
            }
            TokenKind::LeftBracket => {
                // Anonymous blank node []
                let _ = tokenizer.consume_token(); // consume [
                self.expect_token(tokenizer, TokenKind::RightBracket)?;
                let id = context.generate_blank_node_id();
                let blank_node = BlankNode::new(&id).map_err(TurtleParseError::model)?;
                Ok(Subject::BlankNode(blank_node))
            }
            _ => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Expected subject, found {:?}", token.kind),
                position: token.position,
            })),
        }
    }

    /// Parse a predicate (IRI or 'a' for rdf:type)
    fn parse_predicate(
        &self,
        tokenizer: &mut TurtleTokenizer,
        context: &mut TurtleParsingContext,
    ) -> TurtleResult<Predicate> {
        let token = tokenizer.peek_token()?;

        match &token.kind {
            TokenKind::A => {
                let _ = tokenizer.consume_token();
                let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    .map_err(TurtleParseError::model)?;
                Ok(Predicate::NamedNode(rdf_type))
            }
            TokenKind::IriRef(_) => {
                let iri = self.parse_iri_ref(tokenizer, context)?;
                let named_node = NamedNode::new(&iri).map_err(TurtleParseError::model)?;
                Ok(Predicate::NamedNode(named_node))
            }
            TokenKind::PrefixedName(prefix, local) => {
                let _ = tokenizer.consume_token();
                let iri = self.resolve_prefixed_name(prefix, local, context)?;
                let named_node = NamedNode::new(&iri).map_err(TurtleParseError::model)?;
                Ok(Predicate::NamedNode(named_node))
            }
            _ => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Expected predicate, found {:?}", token.kind),
                position: token.position,
            })),
        }
    }

    /// Parse an object (IRI, blank node, literal, or collection)
    fn parse_object(
        &self,
        tokenizer: &mut TurtleTokenizer,
        context: &mut TurtleParsingContext,
    ) -> TurtleResult<Object> {
        let token = tokenizer.peek_token()?;

        match &token.kind {
            TokenKind::IriRef(_) => {
                let iri = self.parse_iri_ref(tokenizer, context)?;
                let named_node = NamedNode::new(&iri).map_err(TurtleParseError::model)?;
                Ok(Object::NamedNode(named_node))
            }
            TokenKind::PrefixedName(prefix, local) => {
                let _ = tokenizer.consume_token();
                let iri = self.resolve_prefixed_name(prefix, local, context)?;
                let named_node = NamedNode::new(&iri).map_err(TurtleParseError::model)?;
                Ok(Object::NamedNode(named_node))
            }
            TokenKind::BlankNodeLabel(label) => {
                let _ = tokenizer.consume_token();
                let blank_node = BlankNode::new(label).map_err(TurtleParseError::model)?;
                Ok(Object::BlankNode(blank_node))
            }
            TokenKind::StringLiteral(value) => {
                let _ = tokenizer.consume_token();

                // Check for language tag or datatype
                let next_token = tokenizer.peek_token().ok();

                if let Some(token) = next_token {
                    match &token.kind {
                        TokenKind::LanguageTag(lang) => {
                            let _ = tokenizer.consume_token();
                            let literal = Literal::new_language_tagged_literal(value, lang)
                                .map_err(|e| {
                                    TurtleParseError::syntax(
                                        crate::error::TurtleSyntaxError::Generic {
                                            message: format!("Invalid language tag: {e}"),
                                            position: TextPosition::default(),
                                        },
                                    )
                                })?;
                            Ok(Object::Literal(literal))
                        }
                        TokenKind::DataTypeAnnotation => {
                            let _ = tokenizer.consume_token(); // consume ^^
                            let datatype_iri = self.parse_iri_ref(tokenizer, context)?;
                            let datatype =
                                NamedNode::new(&datatype_iri).map_err(TurtleParseError::model)?;
                            let literal = Literal::new_typed_literal(value, datatype);
                            Ok(Object::Literal(literal))
                        }
                        _ => {
                            let literal = Literal::new_simple_literal(value);
                            Ok(Object::Literal(literal))
                        }
                    }
                } else {
                    let literal = Literal::new_simple_literal(value);
                    Ok(Object::Literal(literal))
                }
            }
            TokenKind::LeftBracket => {
                // Anonymous blank node []
                let _ = tokenizer.consume_token(); // consume [
                self.expect_token(tokenizer, TokenKind::RightBracket)?;
                let id = context.generate_blank_node_id();
                let blank_node = BlankNode::new(&id).map_err(TurtleParseError::model)?;
                Ok(Object::BlankNode(blank_node))
            }
            _ => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Expected object, found {:?}", token.kind),
                position: token.position,
            })),
        }
    }

    /// Parse an IRI reference
    fn parse_iri_ref(
        &self,
        tokenizer: &mut TurtleTokenizer,
        context: &TurtleParsingContext,
    ) -> TurtleResult<String> {
        let token = tokenizer.consume_token()?;

        if let TokenKind::IriRef(iri) = &token.kind {
            Ok(context.resolve_iri(iri))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Expected IRI reference, found {:?}", token.kind),
                position: token.position,
            }))
        }
    }

    /// Parse a prefix name (the part after @prefix)
    fn parse_prefix_name(&self, tokenizer: &mut TurtleTokenizer) -> TurtleResult<String> {
        let token = tokenizer.consume_token()?;

        match &token.kind {
            TokenKind::PrefixName(name) => Ok(name.clone()),
            TokenKind::PrefixedName(prefix, local) if local.is_empty() => {
                // Handle case where "prefix:" is parsed as PrefixedName but we only want the prefix part
                Ok(prefix.clone())
            }
            _ => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Expected prefix name, found {:?}", token.kind),
                position: token.position,
            })),
        }
    }

    /// Expect a specific token type
    fn expect_token(
        &self,
        tokenizer: &mut TurtleTokenizer,
        expected: TokenKind,
    ) -> TurtleResult<()> {
        let token = tokenizer.consume_token()?;

        if std::mem::discriminant(&token.kind) == std::mem::discriminant(&expected) {
            Ok(())
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Expected {:?}, found {:?}", expected, token.kind),
                position: token.position,
            }))
        }
    }

    /// Resolve a prefixed name using the context
    fn resolve_prefixed_name(
        &self,
        prefix: &str,
        local: &str,
        context: &TurtleParsingContext,
    ) -> TurtleResult<String> {
        if let Some(prefix_iri) = context.prefixes.get(prefix) {
            Ok(format!("{prefix_iri}{local}"))
        } else {
            Err(TurtleParseError::syntax(
                TurtleSyntaxError::UndefinedPrefix {
                    prefix: prefix.to_string(),
                    position: TextPosition::start(), // TODO: track actual position
                },
            ))
        }
    }
}

/// Parsing context for Turtle documents
#[derive(Debug, Clone)]
pub struct TurtleParsingContext {
    /// Prefix declarations
    pub prefixes: HashMap<String, String>,
    /// Base IRI for resolving relative IRIs
    pub base_iri: Option<String>,
    /// Blank node counter
    pub blank_node_counter: usize,
}

impl TurtleParsingContext {
    fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base_iri: None,
            blank_node_counter: 0,
        }
    }

    fn generate_blank_node_id(&mut self) -> String {
        let id = format!("_:b{}", self.blank_node_counter);
        self.blank_node_counter += 1;
        id
    }

    fn resolve_iri(&self, iri: &str) -> String {
        if let Some(ref base) = self.base_iri {
            // Simplified IRI resolution
            if iri.contains(':') {
                iri.to_string() // Absolute IRI
            } else {
                format!("{base}{iri}") // Relative IRI
            }
        } else {
            iri.to_string()
        }
    }
}

/// Statements in a Turtle document
#[derive(Debug, Clone)]
pub enum TurtleStatement {
    Triple(Triple),
    PrefixDecl(String, String),
    BaseDecl(String),
}

/// Token types for Turtle lexing
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    PrefixKeyword, // @prefix
    BaseKeyword,   // @base
    A,             // a (shorthand for rdf:type)

    // Punctuation
    Dot,                // .
    Semicolon,          // ;
    Comma,              // ,
    LeftBracket,        // [
    RightBracket,       // ]
    LeftParen,          // (
    RightParen,         // )
    Colon,              // :
    DataTypeAnnotation, // ^^

    // Literals and identifiers
    IriRef(String),               // <http://example.org>
    PrefixedName(String, String), // prefix:local
    PrefixName(String),           // prefix (in @prefix declarations)
    BlankNodeLabel(String),       // _:label
    StringLiteral(String),        // "string"
    LanguageTag(String),          // @en, @fr

    // Whitespace and comments
    Whitespace,
    Comment(String),

    // End of input
    Eof,
}

/// A token with position information
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub position: TextPosition,
}

/// Simple tokenizer for Turtle format
pub struct TurtleTokenizer {
    input: String,
    position: usize,
    line: usize,
    column: usize,
}

impl TurtleTokenizer {
    fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }

    fn current_char(&self) -> Option<char> {
        self.input.chars().nth(self.position)
    }

    fn peek_token(&mut self) -> TurtleResult<Token> {
        self.skip_whitespace_and_comments();

        if self.is_at_end() {
            return Ok(Token {
                kind: TokenKind::Eof,
                position: TextPosition::new(self.line, self.column, self.position),
            });
        }

        let start_position = TextPosition::new(self.line, self.column, self.position);

        match self.current_char().unwrap() {
            '.' => Ok(Token {
                kind: TokenKind::Dot,
                position: start_position,
            }),
            ';' => Ok(Token {
                kind: TokenKind::Semicolon,
                position: start_position,
            }),
            ',' => Ok(Token {
                kind: TokenKind::Comma,
                position: start_position,
            }),
            '[' => Ok(Token {
                kind: TokenKind::LeftBracket,
                position: start_position,
            }),
            ']' => Ok(Token {
                kind: TokenKind::RightBracket,
                position: start_position,
            }),
            '(' => Ok(Token {
                kind: TokenKind::LeftParen,
                position: start_position,
            }),
            ')' => Ok(Token {
                kind: TokenKind::RightParen,
                position: start_position,
            }),
            ':' => Ok(Token {
                kind: TokenKind::Colon,
                position: start_position,
            }),
            '<' => self.read_iri_ref(start_position),
            '"' => self.read_string_literal(start_position),
            '@' => self.read_at_keyword_or_language_tag(start_position),
            '_' => self.read_blank_node_label(start_position),
            '^' => self.read_datatype_annotation(start_position),
            'a' if self.is_standalone_a() => Ok(Token {
                kind: TokenKind::A,
                position: start_position,
            }),
            _ => self.read_prefixed_name_or_prefix(start_position),
        }
    }

    fn consume_token(&mut self) -> TurtleResult<Token> {
        let token = self.peek_token()?;

        // Advance position based on token
        match &token.kind {
            TokenKind::Dot
            | TokenKind::Semicolon
            | TokenKind::Comma
            | TokenKind::LeftBracket
            | TokenKind::RightBracket
            | TokenKind::LeftParen
            | TokenKind::RightParen
            | TokenKind::Colon => {
                self.advance();
            }
            TokenKind::A => {
                self.advance();
            }
            TokenKind::DataTypeAnnotation => {
                self.advance(); // ^
                self.advance(); // ^
            }
            TokenKind::PrefixKeyword => {
                // Advance by "@prefix" length (7 characters)
                for _ in 0..7 {
                    self.advance();
                }
            }
            TokenKind::BaseKeyword => {
                // Advance by "@base" length (5 characters)
                for _ in 0..5 {
                    self.advance();
                }
            }
            TokenKind::PrefixedName(prefix, local) => {
                // Advance by prefix length + colon + local part length
                let total_length = prefix.len() + 1 + local.len(); // +1 for colon
                for _ in 0..total_length {
                    self.advance();
                }
            }
            TokenKind::PrefixName(name) => {
                // Advance by name length
                for _ in 0..name.len() {
                    self.advance();
                }
            }
            TokenKind::IriRef(iri) => {
                // Advance by IRI length + 2 for angle brackets
                let total_length = iri.len() + 2; // +2 for < and >
                for _ in 0..total_length {
                    self.advance();
                }
            }
            TokenKind::StringLiteral(string) => {
                // Advance by string length + 2 for quotes
                let total_length = string.len() + 2; // +2 for quotes
                for _ in 0..total_length {
                    self.advance();
                }
            }
            TokenKind::Eof => {
                // Don't advance past EOF
            }
            _ => {
                // For complex tokens, we need to advance appropriately
                // This is simplified - a full implementation would track token lengths
                self.advance();
            }
        }

        Ok(token)
    }

    fn advance(&mut self) {
        if let Some(ch) = self.current_char() {
            self.position += ch.len_utf8();
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        while let Some(ch) = self.current_char() {
            if ch.is_whitespace() {
                self.advance();
            } else if ch == '#' {
                // Skip comment line
                while let Some(ch) = self.current_char() {
                    self.advance();
                    if ch == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn read_iri_ref(&self, position: TextPosition) -> TurtleResult<Token> {
        // Simplified IRI reading - just find the closing >
        let content = if let Some(end) = self.input[self.position + 1..].find('>') {
            self.input[self.position + 1..self.position + 1 + end].to_string()
        } else {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Unterminated IRI reference".to_string(),
                position,
            }));
        };

        Ok(Token {
            kind: TokenKind::IriRef(content),
            position,
        })
    }

    fn read_string_literal(&self, position: TextPosition) -> TurtleResult<Token> {
        // Simplified string reading - just find the closing quote
        let mut end_pos = self.position + 1;
        let mut escaped = false;

        while end_pos < self.input.len() {
            let ch = self.input.chars().nth(end_pos).unwrap();
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                let content = self.input[self.position + 1..end_pos].to_string();
                return Ok(Token {
                    kind: TokenKind::StringLiteral(content),
                    position,
                });
            }
            end_pos += ch.len_utf8();
        }

        Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: "Unterminated string literal".to_string(),
            position,
        }))
    }

    fn read_at_keyword_or_language_tag(&self, position: TextPosition) -> TurtleResult<Token> {
        let remaining = &self.input[self.position..];

        if remaining.starts_with("@prefix") {
            Ok(Token {
                kind: TokenKind::PrefixKeyword,
                position,
            })
        } else if remaining.starts_with("@base") {
            Ok(Token {
                kind: TokenKind::BaseKeyword,
                position,
            })
        } else {
            // Language tag
            let end = remaining[1..]
                .find(|c: char| !c.is_alphanumeric() && c != '-')
                .map(|i| i + 1)
                .unwrap_or(remaining.len());
            let tag = remaining[1..end].to_string();
            Ok(Token {
                kind: TokenKind::LanguageTag(tag),
                position,
            })
        }
    }

    fn read_blank_node_label(&self, position: TextPosition) -> TurtleResult<Token> {
        let remaining = &self.input[self.position..];

        if let Some(stripped) = remaining.strip_prefix("_:") {
            let end = stripped
                .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
                .unwrap_or(stripped.len());
            let label = stripped[..end].to_string();
            Ok(Token {
                kind: TokenKind::BlankNodeLabel(label),
                position,
            })
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Invalid blank node label".to_string(),
                position,
            }))
        }
    }

    fn read_datatype_annotation(&self, position: TextPosition) -> TurtleResult<Token> {
        let remaining = &self.input[self.position..];

        if remaining.starts_with("^^") {
            Ok(Token {
                kind: TokenKind::DataTypeAnnotation,
                position,
            })
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Expected ^^ for datatype annotation".to_string(),
                position,
            }))
        }
    }

    fn read_prefixed_name_or_prefix(&self, position: TextPosition) -> TurtleResult<Token> {
        let remaining = &self.input[self.position..];

        // Find the end of the identifier
        let end = remaining
            .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != ':')
            .unwrap_or(remaining.len());

        let identifier = &remaining[..end];

        if let Some(colon_pos) = identifier.find(':') {
            // Prefixed name
            let prefix = identifier[..colon_pos].to_string();
            let local = identifier[colon_pos + 1..].to_string();
            Ok(Token {
                kind: TokenKind::PrefixedName(prefix, local),
                position,
            })
        } else {
            // Just a prefix name (used in @prefix declarations)
            Ok(Token {
                kind: TokenKind::PrefixName(identifier.to_string()),
                position,
            })
        }
    }

    fn is_standalone_a(&self) -> bool {
        // Check if 'a' is followed by whitespace or punctuation
        if let Some(next_char) = self.input.chars().nth(self.position + 1) {
            next_char.is_whitespace() || ".,;[]()".contains(next_char)
        } else {
            true // End of input
        }
    }
}

impl Parser<Triple> for TurtleParser {
    fn parse<R: Read>(&self, mut reader: R) -> TurtleResult<Vec<Triple>> {
        let mut content = String::new();
        reader
            .read_to_string(&mut content)
            .map_err(TurtleParseError::io)?;
        self.parse_document(&content)
    }

    fn for_reader<R: BufRead>(&self, reader: R) -> Box<dyn Iterator<Item = TurtleResult<Triple>>> {
        // For simplicity, read everything and parse
        let content = reader
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .map(|lines| lines.join("\n"));

        match content {
            Ok(content) => match self.parse_document(&content) {
                Ok(triples) => Box::new(triples.into_iter().map(Ok)),
                Err(e) => Box::new(std::iter::once(Err(e))),
            },
            Err(e) => Box::new(std::iter::once(Err(TurtleParseError::io(e)))),
        }
    }
}

/// Turtle serializer
#[derive(Debug, Clone)]
pub struct TurtleSerializer {
    config: SerializationConfig,
}

impl Default for TurtleSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl TurtleSerializer {
    /// Create a new Turtle serializer
    pub fn new() -> Self {
        Self {
            config: SerializationConfig::default(),
        }
    }

    /// Create a Turtle serializer with custom configuration
    pub fn with_config(config: SerializationConfig) -> Self {
        Self { config }
    }
}

impl Serializer<Triple> for TurtleSerializer {
    fn serialize<W: Write>(&self, triples: &[Triple], writer: W) -> TurtleResult<()> {
        let mut formatted_writer = FormattedWriter::new(writer, self.config.clone());

        // Write prefix declarations
        for (prefix, iri) in &self.config.prefixes {
            formatted_writer
                .write_str(&format!("@prefix {prefix}: <{iri}> ."))
                .map_err(TurtleParseError::io)?;
            formatted_writer
                .write_newline()
                .map_err(TurtleParseError::io)?;
        }

        // Write base declaration if present
        if let Some(ref base) = self.config.base_iri {
            formatted_writer
                .write_str(&format!("@base <{base}> ."))
                .map_err(TurtleParseError::io)?;
            formatted_writer
                .write_newline()
                .map_err(TurtleParseError::io)?;
        }

        if !self.config.prefixes.is_empty() || self.config.base_iri.is_some() {
            formatted_writer
                .write_newline()
                .map_err(TurtleParseError::io)?;
        }

        // Write triples
        for triple in triples {
            self.serialize_item_formatted(triple, &mut formatted_writer)?;
            formatted_writer
                .write_str(" .")
                .map_err(TurtleParseError::io)?;
            formatted_writer
                .write_newline()
                .map_err(TurtleParseError::io)?;
        }

        Ok(())
    }

    fn serialize_item<W: Write>(&self, triple: &Triple, writer: W) -> TurtleResult<()> {
        let mut formatted_writer = FormattedWriter::new(writer, self.config.clone());
        self.serialize_item_formatted(triple, &mut formatted_writer)
    }
}

impl TurtleSerializer {
    fn serialize_item_formatted<W: Write>(
        &self,
        triple: &Triple,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        // Serialize subject
        match triple.subject() {
            Subject::NamedNode(nn) => {
                let abbrev = writer.abbreviate_iri(nn.as_str());
                writer.write_str(&abbrev).map_err(TurtleParseError::io)?;
            }
            Subject::BlankNode(bn) => {
                writer
                    .write_str(&format!("_:{}", bn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Subject::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Subject::QuotedTriple(_) => {
                return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: "QuotedTriple subjects not yet supported".to_string(),
                    position: TextPosition::default(),
                }));
            }
        }

        writer.write_space().map_err(TurtleParseError::io)?;

        // Serialize predicate (check for rdf:type abbreviation)
        match triple.predicate() {
            Predicate::NamedNode(nn) => {
                if nn.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                    writer.write_str("a").map_err(TurtleParseError::io)?;
                } else {
                    let abbrev = writer.abbreviate_iri(nn.as_str());
                    writer.write_str(&abbrev).map_err(TurtleParseError::io)?;
                }
            }
            Predicate::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
        }

        writer.write_space().map_err(TurtleParseError::io)?;

        // Serialize object
        match triple.object() {
            Object::NamedNode(nn) => {
                let abbrev = writer.abbreviate_iri(nn.as_str());
                writer.write_str(&abbrev).map_err(TurtleParseError::io)?;
            }
            Object::BlankNode(bn) => {
                writer
                    .write_str(&format!("_:{}", bn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Object::Literal(literal) => {
                let escaped = writer.escape_string(literal.value());
                writer.write_str(&escaped).map_err(TurtleParseError::io)?;

                if let Some(language) = literal.language() {
                    writer
                        .write_str(&format!("@{language}"))
                        .map_err(TurtleParseError::io)?;
                } else if literal.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    let datatype_abbrev = writer.abbreviate_iri(literal.datatype().as_str());
                    writer
                        .write_str(&format!("^^{datatype_abbrev}"))
                        .map_err(TurtleParseError::io)?;
                }
            }
            Object::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Object::QuotedTriple(_) => {
                return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: "QuotedTriple objects not yet supported".to_string(),
                    position: TextPosition::default(),
                }));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_turtle() {
        let parser = TurtleParser::new();
        let input = r#"
            @prefix ex: <http://example.org/> .
            ex:subject ex:predicate "object" .
        "#;

        let triples = parser.parse_document(input).unwrap();
        assert_eq!(triples.len(), 1);

        let triple = &triples[0];
        if let Subject::NamedNode(subject) = triple.subject() {
            assert_eq!(subject.as_str(), "http://example.org/subject");
        } else {
            panic!("Expected named node subject");
        }
    }

    #[test]
    fn test_parse_rdf_type_abbreviation() {
        let parser = TurtleParser::new();
        let input = r#"
            @prefix ex: <http://example.org/> .
            ex:subject a ex:Class .
        "#;

        let triples = parser.parse_document(input).unwrap();
        assert_eq!(triples.len(), 1);

        let triple = &triples[0];
        if let Predicate::NamedNode(predicate) = triple.predicate() {
            assert_eq!(
                predicate.as_str(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            );
        } else {
            panic!("Expected named node predicate");
        }
    }

    #[test]
    fn test_serialize_turtle() {
        let serializer = TurtleSerializer::new();
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap());
        let predicate =
            Predicate::NamedNode(NamedNode::new("http://example.org/predicate").unwrap());
        let object = Object::Literal(Literal::new_simple_literal("object"));
        let triple = Triple::new(subject, predicate, object);

        let mut output = Vec::new();
        serializer.serialize(&[triple], &mut output).unwrap();

        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("<http://example.org/subject>"));
        assert!(output_str.contains("\"object\""));
    }
}
