//! Turtle parser implementation
//!
//! Provides the main `TurtleParser` struct for parsing Turtle RDF documents.
//! Supports full Turtle 1.1 syntax including:
//! - Prefix declarations (@prefix)
//! - Base IRI declarations (@base)
//! - Abbreviated syntax (a for rdf:type, semicolons, commas)
//! - Blank node property lists []
//! - RDF collections ()
//! - RDF-star quoted triples (<< >>)
//! - Comments and whitespace handling

use super::tokenizer::TurtleTokenizer;
use super::types::{TokenKind, TurtleParsingContext, TurtleStatement};
use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::toolkit::Parser;
#[cfg(feature = "rdf-12")]
use oxirs_core::model::literal::BaseDirection;
use oxirs_core::model::{BlankNode, Literal, Object, Predicate, QuotedTriple, Subject, Triple};
use std::collections::HashMap;
use std::io::{BufRead, Read};

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
        let mut errors = Vec::new();

        loop {
            let statement_result = self.parse_statement(&mut tokenizer, &mut context);

            match statement_result {
                Ok(Some(statement)) => match statement {
                    TurtleStatement::Triple(triple) => triples.push(triple),
                    TurtleStatement::Triples(mut triple_list) => triples.append(&mut triple_list),
                    TurtleStatement::PrefixDecl(prefix, iri) => {
                        context.prefixes.insert(prefix, iri);
                    }
                    TurtleStatement::BaseDecl(iri) => {
                        context.base_iri = Some(iri);
                    }
                },
                Ok(None) => {
                    // End of document
                    break;
                }
                Err(e) => {
                    if self.lenient {
                        // In lenient mode, collect errors and continue
                        errors.push(e);
                        // Try to skip to next statement by consuming tokens until we find a period
                        self.skip_to_next_statement(&mut tokenizer);
                    } else {
                        // In strict mode, fail immediately
                        return Err(e);
                    }
                }
            }
        }

        // If we collected any errors in lenient mode, return them all
        if !errors.is_empty() {
            Err(TurtleParseError::multiple(errors))
        } else {
            Ok(triples)
        }
    }

    /// Skip to the next statement after an error (used in lenient mode)
    fn skip_to_next_statement(&self, tokenizer: &mut TurtleTokenizer) {
        // Efficiently scan for the next period or newline without parsing tokens
        // This avoids the expensive O(n^2) behavior of calling peek_token() from every position
        while !tokenizer.is_at_end() {
            if let Some(ch) = tokenizer.current_char() {
                if ch == '.' {
                    // Found a period, consume it and stop
                    tokenizer.advance();
                    break;
                } else if ch == '\n' {
                    // Also break on newlines to avoid scanning too far
                    tokenizer.advance();
                    // After a newline, skip whitespace to get to the next statement
                    tokenizer.skip_whitespace_and_comments();
                    break;
                } else {
                    tokenizer.advance();
                }
            } else {
                break;
            }
        }
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

        let (token, _) = tokenizer.peek_token()?;

        match &token.kind {
            TokenKind::PrefixKeyword => {
                let _ = tokenizer.consume_token(); // consume @prefix
                let prefix = self.parse_prefix_name(tokenizer)?;

                // Check if we need to consume a colon (only if prefix didn't already include it)
                let (next_token, _) = tokenizer.peek_token()?;
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
                // Parse triple(s) - may return multiple triples due to semicolon/comma syntax
                let triples = self.parse_triple(tokenizer, context)?;
                self.expect_token(tokenizer, TokenKind::Dot)?;

                if triples.is_empty() {
                    Ok(None) // Empty triple list
                } else if triples.len() == 1 {
                    Ok(Some(TurtleStatement::Triple(
                        triples
                            .into_iter()
                            .next()
                            .expect("iterator should have next element"),
                    )))
                } else {
                    Ok(Some(TurtleStatement::Triples(triples)))
                }
            }
        }
    }

    /// Parse one or more triples (supports semicolon and comma syntax)
    fn parse_triple(
        &self,
        tokenizer: &mut TurtleTokenizer,
        context: &mut TurtleParsingContext,
    ) -> TurtleResult<Vec<Triple>> {
        let subject = self.parse_subject(tokenizer, context)?;
        let mut triples = Vec::new();

        // Collect any pending triples from blank node property lists in the subject
        triples.append(&mut context.pending_triples);

        // Parse predicate-object lists (separated by semicolons)
        loop {
            // Check for empty predicate-object list (subject followed by dot)
            let (token, _) = tokenizer.peek_token()?;
            if matches!(token.kind, TokenKind::Dot) {
                break;
            }

            let predicate = self.parse_predicate(tokenizer, context)?;

            // Parse object list (separated by commas)
            loop {
                let object = self.parse_object(tokenizer, context)?;

                // Collect any pending triples from blank node property lists in the object
                triples.append(&mut context.pending_triples);

                triples.push(Triple::new(subject.clone(), predicate.clone(), object));

                // Check for comma (more objects for same predicate)
                let (token, _) = tokenizer.peek_token()?;
                if matches!(token.kind, TokenKind::Comma) {
                    let _ = tokenizer.consume_token(); // consume comma
                    continue;
                } else {
                    break;
                }
            }

            // Check for semicolon (more predicates for same subject)
            let (token, _) = tokenizer.peek_token()?;
            if matches!(token.kind, TokenKind::Semicolon) {
                let _ = tokenizer.consume_token(); // consume semicolon
                                                   // Check if there's another predicate or if it's just trailing semicolon
                let (next_token, _) = tokenizer.peek_token()?;
                if matches!(next_token.kind, TokenKind::Dot) {
                    break; // Trailing semicolon before dot
                }
                continue;
            } else {
                break;
            }
        }

        Ok(triples)
    }

    /// Parse a subject (IRI, blank node, or collection)
    fn parse_subject(
        &self,
        tokenizer: &mut TurtleTokenizer,
        context: &mut TurtleParsingContext,
    ) -> TurtleResult<Subject> {
        let (token, _) = tokenizer.peek_token()?;

        match &token.kind {
            TokenKind::IriRef(_) => {
                let iri = self.parse_iri_ref(tokenizer, context)?;
                let named_node = context
                    .create_named_node(&iri)
                    .map_err(TurtleParseError::model)?;
                Ok(Subject::NamedNode(named_node))
            }
            TokenKind::PrefixedName(prefix, local) => {
                let position = token.position;
                let _ = tokenizer.consume_token();
                let iri = self.resolve_prefixed_name(prefix, local, context, position)?;
                let named_node = context
                    .create_named_node(&iri)
                    .map_err(TurtleParseError::model)?;
                Ok(Subject::NamedNode(named_node))
            }
            TokenKind::BlankNodeLabel(label) => {
                let _ = tokenizer.consume_token();
                let blank_node = BlankNode::new(label).map_err(TurtleParseError::model)?;
                Ok(Subject::BlankNode(blank_node))
            }
            TokenKind::LeftBracket => {
                // Blank node with property list: [ pred obj ; pred2 obj2 ]
                let _ = tokenizer.consume_token(); // consume [

                // Generate blank node ID
                let id = context.generate_blank_node_id();
                let blank_node = BlankNode::new(&id).map_err(TurtleParseError::model)?;

                // Check if this is an empty blank node [] or has properties
                let (next_token, _) = tokenizer.peek_token()?;
                if matches!(next_token.kind, TokenKind::RightBracket) {
                    let _ = tokenizer.consume_token(); // consume ]
                    return Ok(Subject::BlankNode(blank_node));
                }

                // Parse property list inside brackets (predicate-object pairs)
                // These will be added to the context's pending triples
                self.parse_blank_node_property_list(
                    tokenizer,
                    context,
                    Subject::BlankNode(blank_node.clone()),
                )?;

                self.expect_token(tokenizer, TokenKind::RightBracket)?;
                Ok(Subject::BlankNode(blank_node))
            }
            TokenKind::DoubleLessThan => {
                // Quoted triple: << subject predicate object >> - RDF 1.2 (RDF-star)
                let _ = tokenizer.consume_token(); // consume <<

                // Parse the inner triple
                let inner_subject = self.parse_subject(tokenizer, context)?;
                let inner_predicate = self.parse_predicate(tokenizer, context)?;
                let inner_object = self.parse_object(tokenizer, context)?;

                // Expect closing >>
                self.expect_token(tokenizer, TokenKind::DoubleGreaterThan)?;

                // Create the quoted triple
                let inner_triple = Triple::new(inner_subject, inner_predicate, inner_object);
                let quoted_triple = QuotedTriple::new(inner_triple);

                Ok(Subject::QuotedTriple(Box::new(quoted_triple)))
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
        let (token, _) = tokenizer.peek_token()?;

        match &token.kind {
            TokenKind::A => {
                let _ = tokenizer.consume_token();
                let rdf_type = context
                    .create_named_node("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    .map_err(TurtleParseError::model)?;
                Ok(Predicate::NamedNode(rdf_type))
            }
            TokenKind::IriRef(_) => {
                let iri = self.parse_iri_ref(tokenizer, context)?;
                let named_node = context
                    .create_named_node(&iri)
                    .map_err(TurtleParseError::model)?;
                Ok(Predicate::NamedNode(named_node))
            }
            TokenKind::PrefixedName(prefix, local) => {
                let position = token.position;
                let _ = tokenizer.consume_token();
                let iri = self.resolve_prefixed_name(prefix, local, context, position)?;
                let named_node = context
                    .create_named_node(&iri)
                    .map_err(TurtleParseError::model)?;
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
        let (token, _) = tokenizer.peek_token()?;

        match &token.kind {
            TokenKind::IriRef(_) => {
                let iri = self.parse_iri_ref(tokenizer, context)?;
                let named_node = context
                    .create_named_node(&iri)
                    .map_err(TurtleParseError::model)?;
                Ok(Object::NamedNode(named_node))
            }
            TokenKind::PrefixedName(prefix, local) => {
                let position = token.position;
                let _ = tokenizer.consume_token();
                let iri = self.resolve_prefixed_name(prefix, local, context, position)?;
                let named_node = context
                    .create_named_node(&iri)
                    .map_err(TurtleParseError::model)?;
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

                if let Some((token, _)) = next_token {
                    match &token.kind {
                        TokenKind::LanguageTag(lang, direction) => {
                            let _ = tokenizer.consume_token();

                            #[cfg(feature = "rdf-12")]
                            let literal = if let Some(dir) = direction {
                                // RDF 1.2 directional language tag
                                let base_direction = match dir.as_str() {
                                    "ltr" => BaseDirection::Ltr,
                                    "rtl" => BaseDirection::Rtl,
                                    _ => {
                                        return Err(TurtleParseError::syntax(
                                            TurtleSyntaxError::Generic {
                                                message: format!("Invalid direction: {dir}"),
                                                position: token.position,
                                            },
                                        ));
                                    }
                                };
                                Literal::new_directional_language_tagged_literal(
                                    value,
                                    lang,
                                    base_direction,
                                )
                                .map_err(|e| {
                                    TurtleParseError::syntax(TurtleSyntaxError::Generic {
                                        message: format!("Invalid directional language tag: {e}"),
                                        position: token.position,
                                    })
                                })?
                            } else {
                                Literal::new_language_tagged_literal(value, lang).map_err(|e| {
                                    TurtleParseError::syntax(
                                        crate::error::TurtleSyntaxError::Generic {
                                            message: format!("Invalid language tag: {e}"),
                                            position: TextPosition::default(),
                                        },
                                    )
                                })?
                            };

                            #[cfg(not(feature = "rdf-12"))]
                            let literal = {
                                if direction.is_some() {
                                    return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                                        message: "Directional language tags require the 'rdf-12' feature".to_string(),
                                        position: token.position,
                                    }));
                                }
                                Literal::new_language_tagged_literal(value, lang).map_err(|e| {
                                    TurtleParseError::syntax(
                                        crate::error::TurtleSyntaxError::Generic {
                                            message: format!("Invalid language tag: {e}"),
                                            position: TextPosition::default(),
                                        },
                                    )
                                })?
                            };

                            Ok(Object::Literal(literal))
                        }
                        TokenKind::DataTypeAnnotation => {
                            let _ = tokenizer.consume_token(); // consume ^^

                            // Parse datatype IRI or prefixed name
                            let (datatype_token, _) = tokenizer.peek_token()?;
                            let datatype_iri = match &datatype_token.kind {
                                TokenKind::IriRef(_) => self.parse_iri_ref(tokenizer, context)?,
                                TokenKind::PrefixedName(prefix, local) => {
                                    let position = datatype_token.position;
                                    let _ = tokenizer.consume_token();
                                    self.resolve_prefixed_name(prefix, local, context, position)?
                                }
                                _ => {
                                    return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                                        message: format!("Expected IRI or prefixed name for datatype, found {:?}", datatype_token.kind),
                                        position: datatype_token.position,
                                    }));
                                }
                            };

                            let datatype = context
                                .create_named_node(&datatype_iri)
                                .map_err(TurtleParseError::model)?;
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
                // Blank node with property list: [ pred obj ; pred2 obj2 ]
                let _ = tokenizer.consume_token(); // consume [

                // Generate blank node ID
                let id = context.generate_blank_node_id();
                let blank_node = BlankNode::new(&id).map_err(TurtleParseError::model)?;

                // Check if this is an empty blank node [] or has properties
                let (next_token, _) = tokenizer.peek_token()?;
                if matches!(next_token.kind, TokenKind::RightBracket) {
                    let _ = tokenizer.consume_token(); // consume ]
                    return Ok(Object::BlankNode(blank_node));
                }

                // Parse property list inside brackets
                self.parse_blank_node_property_list(
                    tokenizer,
                    context,
                    Subject::BlankNode(blank_node.clone()),
                )?;

                self.expect_token(tokenizer, TokenKind::RightBracket)?;
                Ok(Object::BlankNode(blank_node))
            }
            TokenKind::Boolean(value) => {
                let _ = tokenizer.consume_token();
                let xsd_boolean = context
                    .create_named_node("http://www.w3.org/2001/XMLSchema#boolean")
                    .map_err(TurtleParseError::model)?;
                let literal = Literal::new_typed_literal(value.to_string(), xsd_boolean);
                Ok(Object::Literal(literal))
            }
            TokenKind::Integer(value) => {
                let _ = tokenizer.consume_token();
                let xsd_integer = context
                    .create_named_node("http://www.w3.org/2001/XMLSchema#integer")
                    .map_err(TurtleParseError::model)?;
                let literal = Literal::new_typed_literal(value, xsd_integer);
                Ok(Object::Literal(literal))
            }
            TokenKind::Decimal(value) => {
                let _ = tokenizer.consume_token();
                let xsd_decimal = context
                    .create_named_node("http://www.w3.org/2001/XMLSchema#decimal")
                    .map_err(TurtleParseError::model)?;
                let literal = Literal::new_typed_literal(value, xsd_decimal);
                Ok(Object::Literal(literal))
            }
            TokenKind::Double(value) => {
                let _ = tokenizer.consume_token();
                let xsd_double = context
                    .create_named_node("http://www.w3.org/2001/XMLSchema#double")
                    .map_err(TurtleParseError::model)?;
                let literal = Literal::new_typed_literal(value, xsd_double);
                Ok(Object::Literal(literal))
            }
            TokenKind::LeftParen => {
                // RDF Collection: ( item1 item2 item3 )
                self.parse_collection(tokenizer, context)
            }
            TokenKind::DoubleLessThan => {
                // Quoted triple: << subject predicate object >> - RDF 1.2 (RDF-star)
                let _ = tokenizer.consume_token(); // consume <<

                // Parse the inner triple
                let inner_subject = self.parse_subject(tokenizer, context)?;
                let inner_predicate = self.parse_predicate(tokenizer, context)?;
                let inner_object = self.parse_object(tokenizer, context)?;

                // Expect closing >>
                self.expect_token(tokenizer, TokenKind::DoubleGreaterThan)?;

                // Create the quoted triple
                let inner_triple = Triple::new(inner_subject, inner_predicate, inner_object);
                let quoted_triple = QuotedTriple::new(inner_triple);

                Ok(Object::QuotedTriple(Box::new(quoted_triple)))
            }
            _ => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Expected object, found {:?}", token.kind),
                position: token.position,
            })),
        }
    }

    /// Parse RDF collection: ( item1 item2 item3 )
    fn parse_collection(
        &self,
        tokenizer: &mut TurtleTokenizer,
        context: &mut TurtleParsingContext,
    ) -> TurtleResult<Object> {
        let _ = tokenizer.consume_token(); // consume (

        // Check for empty collection ()
        let (next_token, _) = tokenizer.peek_token()?;
        if matches!(next_token.kind, TokenKind::RightParen) {
            let _ = tokenizer.consume_token(); // consume )
                                               // Empty collection is rdf:nil
            let rdf_nil = context
                .create_named_node("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil")
                .map_err(TurtleParseError::model)?;
            return Ok(Object::NamedNode(rdf_nil));
        }

        // Parse collection items and build RDF list structure
        let rdf_first = context
            .create_named_node("http://www.w3.org/1999/02/22-rdf-syntax-ns#first")
            .map_err(TurtleParseError::model)?;
        let rdf_rest = context
            .create_named_node("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest")
            .map_err(TurtleParseError::model)?;
        let rdf_nil = context
            .create_named_node("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil")
            .map_err(TurtleParseError::model)?;

        // First item's blank node
        let first_id = context.generate_blank_node_id();
        let first_bn = BlankNode::new(&first_id).map_err(TurtleParseError::model)?;
        let mut current_bn = first_bn.clone();

        loop {
            // Parse collection item
            let item = self.parse_object(tokenizer, context)?;

            // Create triple: current_bn rdf:first item
            let triple = Triple::new(
                Subject::BlankNode(current_bn.clone()),
                Predicate::NamedNode(rdf_first.clone()),
                item,
            );
            context.pending_triples.push(triple);

            // Check for more items or end of collection
            let (next_token, _) = tokenizer.peek_token()?;
            if matches!(next_token.kind, TokenKind::RightParen) {
                let _ = tokenizer.consume_token(); // consume )
                                                   // Last item: current_bn rdf:rest rdf:nil
                let triple = Triple::new(
                    Subject::BlankNode(current_bn),
                    Predicate::NamedNode(rdf_rest),
                    Object::NamedNode(rdf_nil),
                );
                context.pending_triples.push(triple);
                break;
            } else {
                // More items: create next blank node
                let next_id = context.generate_blank_node_id();
                let next_bn = BlankNode::new(&next_id).map_err(TurtleParseError::model)?;

                // current_bn rdf:rest next_bn
                let triple = Triple::new(
                    Subject::BlankNode(current_bn),
                    Predicate::NamedNode(rdf_rest.clone()),
                    Object::BlankNode(next_bn.clone()),
                );
                context.pending_triples.push(triple);

                current_bn = next_bn;
            }
        }

        Ok(Object::BlankNode(first_bn))
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
        let (token, _) = tokenizer.peek_token()?;

        match &token.kind {
            TokenKind::Colon => {
                // Empty prefix case: @prefix : <...>
                // Don't consume the colon here - it will be consumed by the caller
                Ok(String::new())
            }
            TokenKind::PrefixName(name) => {
                let _ = tokenizer.consume_token();
                Ok(name.clone())
            }
            TokenKind::PrefixedName(prefix, local) if local.is_empty() => {
                // Handle case where "prefix:" is parsed as PrefixedName but we only want the prefix part
                let _ = tokenizer.consume_token();
                Ok(prefix.clone())
            }
            _ => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Expected prefix name, found {:?}", token.kind),
                position: token.position,
            })),
        }
    }

    /// Parse predicate-object pairs inside blank node brackets
    fn parse_blank_node_property_list(
        &self,
        tokenizer: &mut TurtleTokenizer,
        context: &mut TurtleParsingContext,
        subject: Subject,
    ) -> TurtleResult<()> {
        // Parse predicate-object lists (separated by semicolons)
        loop {
            // Check for end of property list
            let (token, _) = tokenizer.peek_token()?;
            if matches!(token.kind, TokenKind::RightBracket) {
                break;
            }

            let predicate = self.parse_predicate(tokenizer, context)?;

            // Parse object list (separated by commas)
            loop {
                let object = self.parse_object(tokenizer, context)?;
                let triple = Triple::new(subject.clone(), predicate.clone(), object);
                context.pending_triples.push(triple);

                // Check for comma (more objects for same predicate)
                let (token, _) = tokenizer.peek_token()?;
                if matches!(token.kind, TokenKind::Comma) {
                    let _ = tokenizer.consume_token(); // consume comma
                    continue;
                } else {
                    break;
                }
            }

            // Check for semicolon (more predicates for same subject)
            let (token, _) = tokenizer.peek_token()?;
            if matches!(token.kind, TokenKind::Semicolon) {
                let _ = tokenizer.consume_token(); // consume semicolon
                                                   // Check if there's another predicate or if it's just trailing semicolon
                let (next_token, _) = tokenizer.peek_token()?;
                if matches!(next_token.kind, TokenKind::RightBracket) {
                    break; // Trailing semicolon before ]
                }
                continue;
            } else {
                break;
            }
        }

        Ok(())
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
        position: TextPosition,
    ) -> TurtleResult<String> {
        if let Some(prefix_iri) = context.prefixes.get(prefix) {
            Ok(format!("{prefix_iri}{local}"))
        } else {
            Err(TurtleParseError::syntax(
                TurtleSyntaxError::UndefinedPrefix {
                    prefix: prefix.to_string(),
                    position,
                },
            ))
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
