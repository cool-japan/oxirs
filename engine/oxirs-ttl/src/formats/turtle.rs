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
use crate::toolkit::{FormattedWriter, Parser, SerializationConfig, Serializer, StringInterner};
#[cfg(feature = "rdf-12")]
use oxirs_core::model::literal::BaseDirection;
use oxirs_core::model::{
    BlankNode, Literal, NamedNode, Object, Predicate, QuotedTriple, Subject, Triple,
};
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
        // This avoids the expensive O(n²) behavior of calling peek_token() from every position
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
                        triples.into_iter().next().unwrap(),
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

/// Parsing context for Turtle documents
#[derive(Debug, Clone)]
pub struct TurtleParsingContext {
    /// Prefix declarations
    pub prefixes: HashMap<String, String>,
    /// Base IRI for resolving relative IRIs
    pub base_iri: Option<String>,
    /// Blank node counter
    pub blank_node_counter: usize,
    /// Pending triples from blank node property lists
    pub pending_triples: Vec<Triple>,
    /// String interner for deduplicating IRIs, prefixes, and language tags
    pub string_interner: StringInterner,
}

impl TurtleParsingContext {
    fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base_iri: None,
            blank_node_counter: 0,
            pending_triples: Vec::new(),
            string_interner: StringInterner::with_common_namespaces(),
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

    /// Create a NamedNode with string interning for efficient memory usage
    fn create_named_node(&mut self, iri: &str) -> Result<NamedNode, oxirs_core::error::OxirsError> {
        // Intern the IRI to deduplicate common strings
        let interned = self.string_interner.intern(iri);
        // Create NamedNode with the interned string (dereference Arc to get &str)
        // Note: This still clones, but the interner reduces allocations overall
        NamedNode::new(interned.as_str())
    }
}

/// Statements in a Turtle document
#[derive(Debug, Clone)]
pub enum TurtleStatement {
    /// Single RDF triple statement
    Triple(Triple),
    /// Multiple RDF triples (from semicolon/comma syntax)
    Triples(Vec<Triple>),
    /// Prefix declaration statement
    PrefixDecl(String, String),
    /// Base URI declaration statement
    BaseDecl(String),
}

/// Token types for Turtle lexing
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    /// @prefix keyword
    PrefixKeyword, // @prefix
    /// @base keyword
    BaseKeyword, // @base
    /// 'a' shorthand for rdf:type
    A, // a (shorthand for rdf:type)

    // Punctuation
    /// Dot punctuation (.)
    Dot, // .
    /// Semicolon punctuation (;)
    Semicolon, // ;
    /// Comma punctuation (,)
    Comma, // ,
    /// Left bracket ([)
    LeftBracket, // [
    /// Right bracket (])
    RightBracket, // ]
    /// Left parenthesis (()
    LeftParen, // (
    /// Right parenthesis ())
    RightParen, // )
    /// Colon punctuation (:)
    Colon, // :
    /// Data type annotation (^^)
    DataTypeAnnotation, // ^^
    /// Double less-than for quoted triples (<<) - RDF 1.2
    DoubleLessThan, // <<
    /// Double greater-than for quoted triples (>>) - RDF 1.2
    DoubleGreaterThan, // >>

    // Literals and identifiers
    /// IRI reference enclosed in angle brackets
    IriRef(String), // <http://example.org>
    /// Prefixed name with prefix and local parts
    PrefixedName(String, String), // prefix:local
    /// Prefix name used in @prefix declarations
    PrefixName(String), // prefix (in @prefix declarations)
    /// Blank node label
    BlankNodeLabel(String), // _:label
    /// String literal value
    StringLiteral(String), // "string"
    /// Language tag for literals, with optional direction for RDF 1.2
    LanguageTag(String, Option<String>), // @en, @en--ltr, @ar--rtl
    /// Boolean literal (true or false)
    Boolean(bool), // true, false
    /// Integer literal
    Integer(String), // 42, -5
    /// Decimal literal
    Decimal(String), // 3.14, -0.5
    /// Double literal (scientific notation)
    Double(String), // 1.5e10, -3.2E-5

    // Whitespace and comments
    /// Whitespace characters
    Whitespace,
    /// Comment text
    Comment(String),

    // End of input
    /// End of input marker
    Eof,
}

/// A token with position information
#[derive(Debug, Clone)]
pub struct Token {
    /// The type of token
    pub kind: TokenKind,
    /// Position in the input text
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
        // Use byte slicing for proper UTF-8 handling
        if self.position >= self.input.len() {
            None
        } else {
            self.input[self.position..].chars().next()
        }
    }

    fn peek_token(&mut self) -> TurtleResult<(Token, usize)> {
        self.skip_whitespace_and_comments();

        if self.is_at_end() {
            return Ok((
                Token {
                    kind: TokenKind::Eof,
                    position: TextPosition::new(self.line, self.column, self.position),
                },
                0,
            ));
        }

        let start_position = TextPosition::new(self.line, self.column, self.position);

        match self.current_char().unwrap() {
            '.' => Ok((
                Token {
                    kind: TokenKind::Dot,
                    position: start_position,
                },
                1,
            )),
            ';' => Ok((
                Token {
                    kind: TokenKind::Semicolon,
                    position: start_position,
                },
                1,
            )),
            ',' => Ok((
                Token {
                    kind: TokenKind::Comma,
                    position: start_position,
                },
                1,
            )),
            '[' => Ok((
                Token {
                    kind: TokenKind::LeftBracket,
                    position: start_position,
                },
                1,
            )),
            ']' => Ok((
                Token {
                    kind: TokenKind::RightBracket,
                    position: start_position,
                },
                1,
            )),
            '(' => Ok((
                Token {
                    kind: TokenKind::LeftParen,
                    position: start_position,
                },
                1,
            )),
            ')' => Ok((
                Token {
                    kind: TokenKind::RightParen,
                    position: start_position,
                },
                1,
            )),
            ':' => {
                // Check if this is an empty prefix name like :alice
                let remaining = &self.input[self.position + 1..];
                if let Some(first_char) = remaining.chars().next() {
                    if first_char.is_alphabetic() || first_char == '_' {
                        // This is an empty prefix name - read the local part
                        return self.read_empty_prefix_name(start_position);
                    }
                }
                // Otherwise, it's just a colon
                Ok((
                    Token {
                        kind: TokenKind::Colon,
                        position: start_position,
                    },
                    1,
                ))
            }
            '<' => {
                // Check for << (quoted triple start) - RDF 1.2
                if self.position + 1 < self.input.len() {
                    let next_char = self.input[self.position + 1..].chars().next();
                    if next_char == Some('<') {
                        return Ok((
                            Token {
                                kind: TokenKind::DoubleLessThan,
                                position: start_position,
                            },
                            2,
                        ));
                    }
                }
                // Otherwise, it's an IRI reference
                self.read_iri_ref(start_position)
            }
            '>' => {
                // Check for >> (quoted triple end) - RDF 1.2
                if self.position + 1 < self.input.len() {
                    let next_char = self.input[self.position + 1..].chars().next();
                    if next_char == Some('>') {
                        return Ok((
                            Token {
                                kind: TokenKind::DoubleGreaterThan,
                                position: start_position,
                            },
                            2,
                        ));
                    }
                }
                // Single > is an error (unexpected character)
                Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: "Unexpected character: '>'".to_string(),
                    position: start_position,
                }))
            }
            '"' => self.read_string_literal(start_position),
            '@' => self.read_at_keyword_or_language_tag(start_position),
            '_' => self.read_blank_node_label(start_position),
            '^' => self.read_datatype_annotation(start_position),
            'a' if self.is_standalone_a() => Ok((
                Token {
                    kind: TokenKind::A,
                    position: start_position,
                },
                1,
            )),
            '+' | '-' | '0'..='9' => self.read_numeric_literal(start_position),
            _ => {
                // Check for boolean keywords (true/false) or prefixed names
                let remaining = &self.input[self.position..];
                if remaining.starts_with("true") && self.is_keyword_boundary(4) {
                    Ok((
                        Token {
                            kind: TokenKind::Boolean(true),
                            position: start_position,
                        },
                        4,
                    ))
                } else if remaining.starts_with("false") && self.is_keyword_boundary(5) {
                    Ok((
                        Token {
                            kind: TokenKind::Boolean(false),
                            position: start_position,
                        },
                        5,
                    ))
                } else {
                    self.read_prefixed_name_or_prefix(start_position)
                }
            }
        }
    }

    fn consume_token(&mut self) -> TurtleResult<Token> {
        let (token, raw_length) = self.peek_token()?;

        // Advance position by raw byte count
        // We need to advance character-by-character to update line/column correctly
        let target_position = self.position + raw_length;
        while self.position < target_position && !self.is_at_end() {
            self.advance();
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

    fn read_iri_ref(&self, position: TextPosition) -> TurtleResult<(Token, usize)> {
        // Simplified IRI reading - just find the closing >
        let (content, raw_length) = if let Some(end) = self.input[self.position + 1..].find('>') {
            let content = self.input[self.position + 1..self.position + 1 + end].to_string();
            // raw_length is end (bytes to '>') + 2 (for '<' and '>')
            (content, end + 2)
        } else {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Unterminated IRI reference".to_string(),
                position,
            }));
        };

        Ok((
            Token {
                kind: TokenKind::IriRef(content),
                position,
            },
            raw_length,
        ))
    }

    fn read_string_literal(&self, position: TextPosition) -> TurtleResult<(Token, usize)> {
        // Check for multiline string (""")
        let remaining = &self.input[self.position..];
        if remaining.starts_with("\"\"\"") {
            return self.read_multiline_string_literal(position);
        }

        // Regular string reading with escape sequence processing
        let mut end_pos = self.position + 1;
        let mut escaped = false;

        while end_pos < self.input.len() {
            let ch = self.input[end_pos..].chars().next().unwrap();
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                let raw_content = &self.input[self.position + 1..end_pos];
                let content = self.process_escape_sequences(raw_content)?;
                let raw_length = end_pos - self.position + 1; // +1 for closing quote
                return Ok((
                    Token {
                        kind: TokenKind::StringLiteral(content),
                        position,
                    },
                    raw_length,
                ));
            }
            end_pos += ch.len_utf8();
        }

        Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: "Unterminated string literal".to_string(),
            position,
        }))
    }

    fn read_multiline_string_literal(
        &self,
        position: TextPosition,
    ) -> TurtleResult<(Token, usize)> {
        // Skip opening """
        let mut end_pos = self.position + 3;

        while end_pos + 2 < self.input.len() {
            if &self.input[end_pos..end_pos + 3] == "\"\"\"" {
                let raw_content = &self.input[self.position + 3..end_pos];
                let content = self.process_escape_sequences(raw_content)?;
                let raw_length = end_pos - self.position + 3; // +3 for closing """
                return Ok((
                    Token {
                        kind: TokenKind::StringLiteral(content),
                        position,
                    },
                    raw_length,
                ));
            }
            end_pos += 1;
        }

        Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: "Unterminated multiline string literal".to_string(),
            position,
        }))
    }

    fn process_escape_sequences(&self, input: &str) -> TurtleResult<String> {
        let mut result = String::with_capacity(input.len());
        let mut chars = input.chars();

        while let Some(ch) = chars.next() {
            if ch == '\\' {
                if let Some(next) = chars.next() {
                    match next {
                        't' => result.push('\t'),
                        'n' => result.push('\n'),
                        'r' => result.push('\r'),
                        '"' => result.push('"'),
                        '\'' => result.push('\''),
                        '\\' => result.push('\\'),
                        'u' => {
                            // \uXXXX - 4 hex digits
                            let hex: String = chars.by_ref().take(4).collect();
                            if hex.len() == 4 {
                                if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                    if let Some(unicode_char) = char::from_u32(code) {
                                        result.push(unicode_char);
                                    } else {
                                        return Err(TurtleParseError::syntax(
                                            TurtleSyntaxError::InvalidUnicode {
                                                codepoint: code,
                                                position: TextPosition::default(),
                                            },
                                        ));
                                    }
                                } else {
                                    return Err(TurtleParseError::syntax(
                                        TurtleSyntaxError::InvalidEscape {
                                            sequence: format!("u{hex}"),
                                            position: TextPosition::default(),
                                        },
                                    ));
                                }
                            } else {
                                return Err(TurtleParseError::syntax(
                                    TurtleSyntaxError::InvalidEscape {
                                        sequence: format!("u{hex}"),
                                        position: TextPosition::default(),
                                    },
                                ));
                            }
                        }
                        'U' => {
                            // \UXXXXXXXX - 8 hex digits
                            let hex: String = chars.by_ref().take(8).collect();
                            if hex.len() == 8 {
                                if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                    if let Some(unicode_char) = char::from_u32(code) {
                                        result.push(unicode_char);
                                    } else {
                                        return Err(TurtleParseError::syntax(
                                            TurtleSyntaxError::InvalidUnicode {
                                                codepoint: code,
                                                position: TextPosition::default(),
                                            },
                                        ));
                                    }
                                } else {
                                    return Err(TurtleParseError::syntax(
                                        TurtleSyntaxError::InvalidEscape {
                                            sequence: format!("U{hex}"),
                                            position: TextPosition::default(),
                                        },
                                    ));
                                }
                            } else {
                                return Err(TurtleParseError::syntax(
                                    TurtleSyntaxError::InvalidEscape {
                                        sequence: format!("U{hex}"),
                                        position: TextPosition::default(),
                                    },
                                ));
                            }
                        }
                        _ => {
                            // Unknown escape sequence - just include it as-is
                            result.push('\\');
                            result.push(next);
                        }
                    }
                } else {
                    result.push('\\');
                }
            } else {
                result.push(ch);
            }
        }

        Ok(result)
    }

    fn read_at_keyword_or_language_tag(
        &self,
        position: TextPosition,
    ) -> TurtleResult<(Token, usize)> {
        let remaining = &self.input[self.position..];

        if remaining.starts_with("@prefix") {
            Ok((
                Token {
                    kind: TokenKind::PrefixKeyword,
                    position,
                },
                7, // "@prefix" length
            ))
        } else if remaining.starts_with("@base") {
            Ok((
                Token {
                    kind: TokenKind::BaseKeyword,
                    position,
                },
                5, // "@base" length
            ))
        } else {
            // Language tag (possibly with direction for RDF 1.2)
            let end = remaining[1..]
                .find(|c: char| !c.is_alphanumeric() && c != '-')
                .map(|i| i + 1)
                .unwrap_or(remaining.len());
            let tag_with_dir = &remaining[1..end];

            // Check for RDF 1.2 directional language tag: @lang--dir
            let (tag, direction, raw_length) =
                if let Some(double_dash_pos) = tag_with_dir.find("--") {
                    let language = tag_with_dir[..double_dash_pos].to_string();
                    let dir = &tag_with_dir[double_dash_pos + 2..];

                    // Validate direction is either "ltr" or "rtl"
                    if dir != "ltr" && dir != "rtl" {
                        return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                            message: format!("Invalid direction '{}'. Must be 'ltr' or 'rtl'", dir),
                            position,
                        }));
                    }

                    (language, Some(dir.to_string()), end)
                } else {
                    (tag_with_dir.to_string(), None, end)
                };

            Ok((
                Token {
                    kind: TokenKind::LanguageTag(tag, direction),
                    position,
                },
                raw_length,
            ))
        }
    }

    fn read_blank_node_label(&self, position: TextPosition) -> TurtleResult<(Token, usize)> {
        let remaining = &self.input[self.position..];

        if let Some(stripped) = remaining.strip_prefix("_:") {
            let end = stripped
                .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
                .unwrap_or(stripped.len());
            let label = stripped[..end].to_string();
            let raw_length = 2 + end; // "_:" + label
            Ok((
                Token {
                    kind: TokenKind::BlankNodeLabel(label),
                    position,
                },
                raw_length,
            ))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Invalid blank node label".to_string(),
                position,
            }))
        }
    }

    fn read_datatype_annotation(&self, position: TextPosition) -> TurtleResult<(Token, usize)> {
        let remaining = &self.input[self.position..];

        if remaining.starts_with("^^") {
            Ok((
                Token {
                    kind: TokenKind::DataTypeAnnotation,
                    position,
                },
                2, // "^^" length
            ))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Expected ^^ for datatype annotation".to_string(),
                position,
            }))
        }
    }

    fn read_prefixed_name_or_prefix(&self, position: TextPosition) -> TurtleResult<(Token, usize)> {
        let remaining = &self.input[self.position..];

        // Find the end of the identifier
        let end = remaining
            .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != ':')
            .unwrap_or(remaining.len());

        let identifier = &remaining[..end];
        let raw_length = end;

        if let Some(colon_pos) = identifier.find(':') {
            // Prefixed name
            let prefix = identifier[..colon_pos].to_string();
            let local = identifier[colon_pos + 1..].to_string();
            Ok((
                Token {
                    kind: TokenKind::PrefixedName(prefix, local),
                    position,
                },
                raw_length,
            ))
        } else {
            // Just a prefix name (used in @prefix declarations)
            Ok((
                Token {
                    kind: TokenKind::PrefixName(identifier.to_string()),
                    position,
                },
                raw_length,
            ))
        }
    }

    fn read_empty_prefix_name(&self, position: TextPosition) -> TurtleResult<(Token, usize)> {
        // Skip the initial colon
        let remaining = &self.input[self.position + 1..];

        // Find the end of the local part
        let end = remaining
            .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .unwrap_or(remaining.len());

        let local = &remaining[..end];
        let raw_length = end + 1; // +1 for the initial colon

        Ok((
            Token {
                kind: TokenKind::PrefixedName(String::new(), local.to_string()),
                position,
            },
            raw_length,
        ))
    }

    fn is_standalone_a(&self) -> bool {
        // Check if 'a' is followed by whitespace or punctuation
        if let Some(next_char) = self.input.chars().nth(self.position + 1) {
            next_char.is_whitespace() || ".,;[]()".contains(next_char)
        } else {
            true // End of input
        }
    }

    fn is_keyword_boundary(&self, keyword_len: usize) -> bool {
        // Check if keyword is followed by whitespace, punctuation, or end of input
        if self.position + keyword_len >= self.input.len() {
            return true; // End of input
        }
        if let Some(next_char) = self.input[self.position + keyword_len..].chars().next() {
            next_char.is_whitespace() || ".,;[]()".contains(next_char)
        } else {
            true
        }
    }

    fn read_numeric_literal(&self, position: TextPosition) -> TurtleResult<(Token, usize)> {
        let remaining = &self.input[self.position..];
        let mut end = 0;
        let mut has_decimal_point = false;
        let mut has_exponent = false;

        // Handle optional sign
        if remaining.starts_with('+') || remaining.starts_with('-') {
            end += 1;
        }

        // Read digits before decimal point or exponent
        while end < remaining.len() {
            let ch = remaining.chars().nth(end).unwrap();
            if ch.is_ascii_digit() {
                end += 1;
            } else {
                break;
            }
        }

        // Check for decimal point
        if end < remaining.len() && remaining.chars().nth(end) == Some('.') {
            // Make sure it's not the end-of-statement dot
            if end + 1 < remaining.len() {
                let next_ch = remaining.chars().nth(end + 1).unwrap();
                if next_ch.is_ascii_digit() {
                    has_decimal_point = true;
                    end += 1; // Skip the decimal point

                    // Read fractional digits
                    while end < remaining.len() {
                        let ch = remaining.chars().nth(end).unwrap();
                        if ch.is_ascii_digit() {
                            end += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        // Check for exponent (e or E)
        if end < remaining.len() {
            let ch = remaining.chars().nth(end).unwrap();
            if ch == 'e' || ch == 'E' {
                has_exponent = true;
                end += 1;

                // Handle optional exponent sign
                if end < remaining.len() {
                    let sign_ch = remaining.chars().nth(end).unwrap();
                    if sign_ch == '+' || sign_ch == '-' {
                        end += 1;
                    }
                }

                // Read exponent digits
                let exponent_start = end;
                while end < remaining.len() {
                    let ch = remaining.chars().nth(end).unwrap();
                    if ch.is_ascii_digit() {
                        end += 1;
                    } else {
                        break;
                    }
                }

                // Ensure we have at least one digit in the exponent
                if end == exponent_start {
                    return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                        message: "Invalid numeric literal: exponent requires digits".to_string(),
                        position,
                    }));
                }
            }
        }

        if end == 0 || (end == 1 && (remaining.starts_with('+') || remaining.starts_with('-'))) {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Invalid numeric literal: no digits found".to_string(),
                position,
            }));
        }

        let literal_str = remaining[..end].to_string();
        let token_kind = if has_exponent {
            TokenKind::Double(literal_str)
        } else if has_decimal_point {
            TokenKind::Decimal(literal_str)
        } else {
            TokenKind::Integer(literal_str)
        };

        Ok((
            Token {
                kind: token_kind,
                position,
            },
            end,
        ))
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

    /// Create a Turtle serializer with auto-generated prefixes from the triples
    pub fn with_auto_prefixes(triples: &[Triple]) -> Self {
        let prefixes = Self::auto_generate_prefixes(triples);
        let config = SerializationConfig::default().with_use_prefixes(true);

        let mut config_with_prefixes = config;
        config_with_prefixes.prefixes = prefixes;

        Self {
            config: config_with_prefixes,
        }
    }

    /// Auto-detect and generate common prefixes from a set of triples
    pub fn auto_generate_prefixes(triples: &[Triple]) -> HashMap<String, String> {
        let mut iri_counts: HashMap<String, usize> = HashMap::new();

        // Count IRI namespace occurrences
        for triple in triples {
            // Count subject namespace
            if let Subject::NamedNode(nn) = triple.subject() {
                if let Some(namespace) = Self::extract_namespace(nn.as_str()) {
                    *iri_counts.entry(namespace).or_insert(0) += 1;
                }
            }

            // Count predicate namespace
            if let Predicate::NamedNode(nn) = triple.predicate() {
                if let Some(namespace) = Self::extract_namespace(nn.as_str()) {
                    *iri_counts.entry(namespace).or_insert(0) += 1;
                }
            }

            // Count object namespace
            if let Object::NamedNode(nn) = triple.object() {
                if let Some(namespace) = Self::extract_namespace(nn.as_str()) {
                    *iri_counts.entry(namespace).or_insert(0) += 1;
                }
            }
        }

        // Generate prefixes for namespaces used more than once
        let mut prefixes = HashMap::new();
        let mut prefix_counter = 1;

        for (namespace, count) in iri_counts {
            if count > 1 {
                // Try to generate a meaningful prefix from the namespace
                let prefix = Self::suggest_prefix(&namespace, prefix_counter);
                prefixes.insert(prefix, namespace);
                prefix_counter += 1;
            }
        }

        // Add common well-known prefixes if they're used
        Self::add_well_known_prefixes(&mut prefixes, triples);

        prefixes
    }

    /// Extract namespace from an IRI (everything up to the last # or /)
    fn extract_namespace(iri: &str) -> Option<String> {
        // Find the last occurrence of # or /
        let last_separator = iri.rfind(['#', '/'])?;
        Some(iri[..=last_separator].to_string())
    }

    /// Suggest a prefix name based on the namespace
    fn suggest_prefix(namespace: &str, counter: usize) -> String {
        // Try to extract a meaningful part from the namespace
        if namespace.contains("example.org") {
            return "ex".to_string();
        } else if namespace.contains("w3.org/1999/02/22-rdf-syntax-ns#") {
            return "rdf".to_string();
        } else if namespace.contains("w3.org/2000/01/rdf-schema#") {
            return "rdfs".to_string();
        } else if namespace.contains("w3.org/2002/07/owl#") {
            return "owl".to_string();
        } else if namespace.contains("xmlns.com/foaf") {
            return "foaf".to_string();
        } else if namespace.contains("purl.org/dc") {
            return "dc".to_string();
        } else if namespace.contains("schema.org") {
            return "schema".to_string();
        }

        // Generic prefix
        format!("ns{counter}")
    }

    /// Add well-known prefixes if they're actually used in the triples
    fn add_well_known_prefixes(prefixes: &mut HashMap<String, String>, triples: &[Triple]) {
        let well_known = [
            ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
            ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
            ("xsd", "http://www.w3.org/2001/XMLSchema#"),
            ("owl", "http://www.w3.org/2002/07/owl#"),
        ];

        for (prefix, iri) in &well_known {
            // Check if this namespace is used
            let used = triples.iter().any(|t| Self::triple_uses_namespace(t, iri));

            if used && !prefixes.values().any(|v| v == iri) {
                prefixes.insert(prefix.to_string(), iri.to_string());
            }
        }
    }

    /// Check if a triple uses a specific namespace
    fn triple_uses_namespace(triple: &Triple, namespace: &str) -> bool {
        // Check subject
        if let Subject::NamedNode(nn) = triple.subject() {
            if nn.as_str().starts_with(namespace) {
                return true;
            }
        }

        // Check predicate
        if let Predicate::NamedNode(nn) = triple.predicate() {
            if nn.as_str().starts_with(namespace) {
                return true;
            }
        }

        // Check object
        if let Object::NamedNode(nn) = triple.object() {
            if nn.as_str().starts_with(namespace) {
                return true;
            }
        }

        false
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
            Subject::QuotedTriple(qt) => {
                // RDF 1.2: << s p o >> syntax
                writer.write_str("<< ").map_err(TurtleParseError::io)?;
                Self::serialize_quoted_triple(qt, writer)?;
                writer.write_str(" >>").map_err(TurtleParseError::io)?;
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
                    // Check for directional language tag (RDF 1.2)
                    #[cfg(feature = "rdf-12")]
                    if let Some(direction) = literal.direction() {
                        writer
                            .write_str(&format!("@{language}--{direction}"))
                            .map_err(TurtleParseError::io)?;
                    } else {
                        writer
                            .write_str(&format!("@{language}"))
                            .map_err(TurtleParseError::io)?;
                    }

                    #[cfg(not(feature = "rdf-12"))]
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
            Object::QuotedTriple(qt) => {
                // RDF 1.2: << s p o >> syntax
                writer.write_str("<< ").map_err(TurtleParseError::io)?;
                Self::serialize_quoted_triple(qt, writer)?;
                writer.write_str(" >>").map_err(TurtleParseError::io)?;
            }
        }

        Ok(())
    }

    /// Helper method to serialize a quoted triple (RDF 1.2 / RDF-star)
    fn serialize_quoted_triple<W: Write>(
        qt: &QuotedTriple,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        // Serialize inner subject
        match qt.subject() {
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
            Subject::QuotedTriple(inner_qt) => {
                // Nested quoted triple
                writer.write_str("<< ").map_err(TurtleParseError::io)?;
                Self::serialize_quoted_triple(inner_qt, writer)?;
                writer.write_str(" >>").map_err(TurtleParseError::io)?;
            }
        }

        writer.write_space().map_err(TurtleParseError::io)?;

        // Serialize inner predicate
        match qt.predicate() {
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

        // Serialize inner object
        match qt.object() {
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
                    // Check for directional language tag (RDF 1.2)
                    #[cfg(feature = "rdf-12")]
                    if let Some(direction) = literal.direction() {
                        writer
                            .write_str(&format!("@{language}--{direction}"))
                            .map_err(TurtleParseError::io)?;
                    } else {
                        writer
                            .write_str(&format!("@{language}"))
                            .map_err(TurtleParseError::io)?;
                    }

                    #[cfg(not(feature = "rdf-12"))]
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
            Object::QuotedTriple(inner_qt) => {
                // Nested quoted triple
                writer.write_str("<< ").map_err(TurtleParseError::io)?;
                Self::serialize_quoted_triple(inner_qt, writer)?;
                writer.write_str(" >>").map_err(TurtleParseError::io)?;
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
#[cfg(test)]
mod test_language_tags {
    use super::*;

    #[test]
    fn test_single_language_tag() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:greeting ex:text "Hello"@en .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_ok(),
            "Single language tag should work: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_multiple_language_tags_semicolon() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:book ex:title "Hello"@en ; ex:title "Bonjour"@fr .
        "#;
        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(
            result.is_ok(),
            "Multiple language tags with semicolon should work: {:?}",
            result.err()
        );
    }
}
