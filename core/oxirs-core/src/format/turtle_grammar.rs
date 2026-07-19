//! Turtle Grammar Recognizer
//!
//! **Status: unfinished / not wired up.** This module was meant to implement
//! the W3C Turtle grammar specification as a [`RuleRecognizer`] that builds
//! RDF triples from N3/Turtle token streams, but [`TurtleGrammarRecognizer`]
//! never actually recognizes anything and [`TurtleParser::parse_str`] /
//! [`TurtleParser::parse_reader`] never parsed the input they were given.
//! Rather than silently returning `Ok(vec![])` (i.e. silently discarding
//! every triple in the document, indistinguishable from "empty document"),
//! every entry point now returns an explicit
//! [`RdfParseError::UnsupportedFeature`] error. The module is `pub(crate)`
//! (see `format/mod.rs`) so it cannot be reached from outside this crate.
//!
//! For real Turtle parsing use [`crate::format::turtle::TurtleParser`] (the
//! oxttl-backed implementation) or [`crate::parser::Parser`] with
//! `RdfFormat::Turtle`.

#![allow(dead_code)]

use super::error::{ParseResult, RdfParseError, RdfSyntaxError, TextPosition};
use super::n3_lexer::N3Token;
use super::toolkit::{Parser, RuleRecognizer};
use crate::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject, Triple};
use std::collections::HashMap;

/// AST node types for Turtle grammar
#[derive(Debug, Clone, PartialEq)]
pub enum TurtleNode {
    Triple(Triple),
    PrefixDeclaration { prefix: String, iri: String },
    BaseDeclaration { iri: String },
    Comment(String),
}

/// Turtle parser context for prefix management and base IRI resolution
#[derive(Debug, Clone)]
pub struct TurtleContext {
    /// Current base IRI for relative IRI resolution
    pub base_iri: Option<String>,
    /// Prefix declarations mapping prefix -> IRI
    pub prefixes: HashMap<String, String>,
    /// Auto-generated blank node counter
    pub blank_node_counter: u64,
    /// Current position for error reporting
    pub position: TextPosition,
}

impl Default for TurtleContext {
    fn default() -> Self {
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
            base_iri: None,
            prefixes,
            blank_node_counter: 0,
            position: TextPosition::start(),
        }
    }
}

impl TurtleContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve a prefixed name to a full IRI
    pub fn resolve_prefixed_name(&self, prefix: Option<&str>, local: &str) -> ParseResult<String> {
        match prefix {
            Some(prefix) => match self.prefixes.get(prefix) {
                Some(base_iri) => Ok(format!("{base_iri}{local}")),
                None => Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                    format!("Undefined prefix: {prefix}"),
                    self.position,
                ))),
            },
            None => {
                // Default prefix (empty prefix)
                match self.prefixes.get("") {
                    Some(base_iri) => Ok(format!("{base_iri}{local}")),
                    None => Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        "No default prefix defined".to_string(),
                        self.position,
                    ))),
                }
            }
        }
    }

    /// Resolve a relative IRI against the base IRI
    pub fn resolve_iri(&self, iri: &str) -> ParseResult<String> {
        if self.is_absolute_iri(iri) {
            Ok(iri.to_string())
        } else {
            match &self.base_iri {
                Some(base) => Ok(self.resolve_relative_iri(base, iri)),
                None => Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                    format!("Relative IRI without base: {iri}"),
                    self.position,
                ))),
            }
        }
    }

    /// Generate a new anonymous blank node
    pub fn generate_blank_node(&mut self) -> BlankNode {
        self.blank_node_counter += 1;
        BlankNode::new(format!("_:gen{}", self.blank_node_counter))
            .expect("generated blank node format is always valid")
    }

    /// Check if an IRI is absolute (has scheme)
    fn is_absolute_iri(&self, iri: &str) -> bool {
        iri.contains(':') && !iri.starts_with(':')
    }

    /// Resolve relative IRI against base IRI
    fn resolve_relative_iri(&self, base: &str, relative: &str) -> String {
        if relative.is_empty() {
            return base.to_string();
        }

        // Simple implementation - in production would use proper URI resolution
        if base.ends_with('/') || base.ends_with('#') {
            format!("{base}{relative}")
        } else {
            format!("{base}/{relative}")
        }
    }
}

/// Turtle grammar recognizer state machine
#[derive(Debug, Clone, PartialEq)]
pub enum TurtleGrammarState {
    /// Expecting statement (triple, directive, or comment)
    ExpectingStatement,
    /// Processing prefix declaration
    PrefixDeclaration { prefix: Option<String> },
    /// Processing base declaration
    BaseDeclaration,
    /// Processing triple with subject
    TripleWithSubject { subject: Subject },
    /// Processing predicate-object list
    PredicateObjectList {
        subject: Subject,
        predicates: Vec<(Predicate, Vec<Object>)>,
    },
    /// Processing object list for current predicate
    ObjectList {
        subject: Subject,
        predicate: Predicate,
        objects: Vec<Object>,
    },
    /// Processing blank node property list
    BlankNodePropertyList {
        properties: Vec<(Predicate, Vec<Object>)>,
    },
    /// Processing collection (RDF list)
    Collection { items: Vec<Object> },
    /// Error recovery state
    ErrorRecovery,
}

/// Turtle grammar recognizer implementation
#[derive(Debug, Clone)]
pub struct TurtleGrammarRecognizer {
    state: TurtleGrammarState,
}

impl Default for TurtleGrammarRecognizer {
    fn default() -> Self {
        Self {
            state: TurtleGrammarState::ExpectingStatement,
        }
    }
}

impl TurtleGrammarRecognizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a term (subject, predicate, or object) from a token
    fn parse_term(&self, token: &N3Token, context: &mut TurtleContext) -> ParseResult<Object> {
        match token {
            N3Token::Iri(iri) => {
                let resolved_iri = context.resolve_iri(iri)?;
                Ok(Object::NamedNode(
                    NamedNode::new(resolved_iri)
                        .map_err(|e| RdfParseError::internal(e.to_string()))?,
                ))
            }
            N3Token::PrefixedName { prefix, local } => {
                let iri = context.resolve_prefixed_name(prefix.as_deref(), local)?;
                Ok(Object::NamedNode(
                    NamedNode::new(iri).map_err(|e| RdfParseError::internal(e.to_string()))?,
                ))
            }
            N3Token::BlankNode(label) => Ok(Object::BlankNode(
                BlankNode::new(label.clone())
                    .map_err(|e| RdfParseError::internal(e.to_string()))?,
            )),
            N3Token::Literal {
                value,
                datatype,
                language,
            } => {
                let literal: Literal = match (datatype, language) {
                    (Some(dt), None) => {
                        let dt_iri = context.resolve_iri(dt)?;
                        Literal::new_typed_literal(
                            value,
                            NamedNode::new(dt_iri)
                                .map_err(|e| RdfParseError::internal(e.to_string()))?,
                        )
                    }
                    (None, Some(lang)) => Literal::new_language_tagged_literal(value, lang)
                        .map_err(|e| RdfParseError::InvalidLanguageTag(e.to_string()))?,
                    (None, None) => Literal::new_simple_literal(value),
                    (Some(_), Some(_)) => {
                        return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                            "Literal cannot have both datatype and language tag".to_string(),
                            context.position,
                        )));
                    }
                };
                Ok(Object::Literal(literal))
            }
            N3Token::Integer(i) => {
                let xsd_integer = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .map_err(|e| RdfParseError::internal(e.to_string()))?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    i.to_string(),
                    xsd_integer,
                )))
            }
            N3Token::Decimal(d) => {
                let xsd_decimal = NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal")
                    .map_err(|e| RdfParseError::internal(e.to_string()))?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    d.to_string(),
                    xsd_decimal,
                )))
            }
            N3Token::Double(d) => {
                let xsd_double = NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .map_err(|e| RdfParseError::internal(e.to_string()))?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    d.to_string(),
                    xsd_double,
                )))
            }
            N3Token::True => {
                let xsd_boolean = NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")
                    .map_err(|e| RdfParseError::internal(e.to_string()))?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    "true",
                    xsd_boolean,
                )))
            }
            N3Token::False => {
                let xsd_boolean = NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")
                    .map_err(|e| RdfParseError::internal(e.to_string()))?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    "false",
                    xsd_boolean,
                )))
            }
            N3Token::A => {
                // 'a' is shorthand for rdf:type
                let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    .map_err(|e| RdfParseError::internal(e.to_string()))?;
                Ok(Object::NamedNode(rdf_type))
            }
            _ => Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                format!("Unexpected token in term position: {token:?}"),
                context.position,
            ))),
        }
    }

    /// Parse a subject from a token
    fn parse_subject(&self, token: &N3Token, context: &mut TurtleContext) -> ParseResult<Subject> {
        match self.parse_term(token, context)? {
            Object::NamedNode(n) => Ok(Subject::NamedNode(n)),
            Object::BlankNode(b) => Ok(Subject::BlankNode(b)),
            _ => Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Invalid subject: must be IRI or blank node".to_string(),
                context.position,
            ))),
        }
    }

    /// Parse a predicate from a token
    fn parse_predicate(
        &self,
        token: &N3Token,
        context: &mut TurtleContext,
    ) -> ParseResult<Predicate> {
        match self.parse_term(token, context)? {
            Object::NamedNode(n) => Ok(Predicate::NamedNode(n)),
            _ => Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Invalid predicate: must be IRI".to_string(),
                context.position,
            ))),
        }
    }
}

impl RuleRecognizer<TurtleNode> for TurtleGrammarRecognizer {
    fn recognize_next_node<Token>(
        &mut self,
        _parser: &mut Parser<Token>,
    ) -> ParseResult<Option<TurtleNode>> {
        // This recognizer was never finished: it does not process tokens
        // according to `self.state` and cannot legitimately return
        // `TurtleNode::Triple`/`PrefixDeclaration`/etc. Returning `Ok(None)`
        // here would be indistinguishable from "end of input reached
        // successfully with nothing left to recognize", which silently
        // drops every statement in the document. Fail loudly instead.
        Err(RdfParseError::unsupported(
            "turtle_grammar::TurtleGrammarRecognizer is not implemented; use \
             crate::format::turtle::TurtleParser or crate::parser::Parser \
             (RdfFormat::Turtle) instead",
        ))
    }
}

/// High-level Turtle parser combining lexer and grammar recognizer
pub struct TurtleParser {
    context: TurtleContext,
}

impl TurtleParser {
    pub fn new() -> Self {
        Self {
            context: TurtleContext::new(),
        }
    }

    /// Parse Turtle from a string into triples
    ///
    /// # Errors
    ///
    /// Always returns [`RdfParseError::UnsupportedFeature`]: this recognizer
    /// is unfinished (see module docs) and must not silently report success
    /// with zero triples for non-empty input. Use
    /// [`crate::format::turtle::TurtleParser`] instead.
    pub fn parse_str(&mut self, _input: &str) -> ParseResult<Vec<Triple>> {
        Err(RdfParseError::unsupported(
            "turtle_grammar::TurtleParser::parse_str is not implemented; use \
             crate::format::turtle::TurtleParser or crate::parser::Parser \
             (RdfFormat::Turtle) instead",
        ))
    }

    /// Parse Turtle from a reader into triples
    ///
    /// # Errors
    ///
    /// Always returns [`RdfParseError::UnsupportedFeature`]; see
    /// [`Self::parse_str`].
    pub fn parse_reader<R: std::io::Read>(&mut self, _reader: R) -> ParseResult<Vec<Triple>> {
        Err(RdfParseError::unsupported(
            "turtle_grammar::TurtleParser::parse_reader is not implemented; use \
             crate::format::turtle::TurtleParser or crate::parser::Parser \
             (RdfFormat::Turtle) instead",
        ))
    }

    /// Set base IRI for relative IRI resolution
    pub fn set_base_iri(&mut self, base_iri: String) {
        self.context.base_iri = Some(base_iri);
    }

    /// Add a prefix declaration
    pub fn add_prefix(&mut self, prefix: String, iri: String) {
        self.context.prefixes.insert(prefix, iri);
    }

    /// Get the current context (for inspection)
    pub fn context(&self) -> &TurtleContext {
        &self.context
    }
}

impl Default for TurtleParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turtle_context_prefix_resolution() {
        let context = TurtleContext::new();

        // Test standard prefix resolution
        let resolved = context
            .resolve_prefixed_name(Some("rdf"), "type")
            .expect("prefix resolution should succeed");
        assert_eq!(resolved, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");

        // Test undefined prefix
        assert!(context
            .resolve_prefixed_name(Some("undefined"), "test")
            .is_err());
    }

    #[test]
    fn test_turtle_context_iri_resolution() {
        let mut context = TurtleContext::new();
        context.base_iri = Some("http://example.org/".to_string());

        // Test absolute IRI (should remain unchanged)
        let resolved = context
            .resolve_iri("http://other.org/test")
            .expect("operation should succeed");
        assert_eq!(resolved, "http://other.org/test");

        // Test relative IRI resolution
        let resolved = context
            .resolve_iri("relative")
            .expect("operation should succeed");
        assert_eq!(resolved, "http://example.org/relative");

        // Test relative IRI without base (should error)
        context.base_iri = None;
        assert!(context.resolve_iri("relative").is_err());
    }

    #[test]
    fn test_blank_node_generation() {
        let mut context = TurtleContext::new();

        let bn1 = context.generate_blank_node();
        let bn2 = context.generate_blank_node();

        assert_ne!(bn1, bn2);
        assert!(bn1.to_string().starts_with("_:gen"));
        assert!(bn2.to_string().starts_with("_:gen"));
    }

    #[test]
    fn test_turtle_parser_creation() {
        let parser = TurtleParser::new();
        assert!(parser.context.prefixes.contains_key("rdf"));
        assert!(parser.context.prefixes.contains_key("xsd"));
    }

    /// Regression test for the P1 finding: `parse_str` must never silently
    /// report success with zero triples for non-empty input (that would be
    /// indistinguishable from "parsed an empty document" and cause silent
    /// data loss for any real caller). It must fail loudly instead.
    #[test]
    fn test_parse_str_fails_loudly_instead_of_silently_dropping_input() {
        let mut parser = TurtleParser::new();
        let result = parser.parse_str("<http://example.org/s> <http://example.org/p> \"o\" .");
        assert!(
            result.is_err(),
            "unfinished recognizer must error, not silently return Ok(empty)"
        );
        assert!(matches!(
            result.unwrap_err(),
            RdfParseError::UnsupportedFeature(_)
        ));
    }

    #[test]
    fn test_parse_reader_fails_loudly_instead_of_silently_dropping_input() {
        let mut parser = TurtleParser::new();
        let data = b"<http://example.org/s> <http://example.org/p> \"o\" .";
        let result = parser.parse_reader(&data[..]);
        assert!(
            result.is_err(),
            "unfinished recognizer must error, not silently return Ok(empty)"
        );
        assert!(matches!(
            result.unwrap_err(),
            RdfParseError::UnsupportedFeature(_)
        ));
    }

    #[test]
    fn test_recognize_next_node_fails_loudly() {
        use super::super::n3_lexer::N3Token;
        use super::super::toolkit::Parser as ToolkitParser;

        let mut recognizer = TurtleGrammarRecognizer::new();
        // The recognizer never reads from the parser before erroring, so an
        // empty token buffer is sufficient to exercise the fail-loudly path.
        let mut toolkit_parser: ToolkitParser<N3Token> = ToolkitParser::new(Vec::new());
        let result = recognizer.recognize_next_node(&mut toolkit_parser);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RdfParseError::UnsupportedFeature(_)
        ));
    }
}
