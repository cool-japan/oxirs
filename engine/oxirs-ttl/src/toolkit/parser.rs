//! Generic parser framework for converting tokens to RDF elements
//!
//! This module provides the rule-based parsing infrastructure that works
//! with tokenizers to produce RDF triples and quads.

use crate::error::{RuleRecognizerError, TurtleParseError, TurtleResult};
use crate::toolkit::lexer::{TokenOrLineJump, TokenRecognizer};
// use oxirs_core::model::{Quad, Triple};
use std::io::{BufRead, Read};
use std::marker::PhantomData;

/// A rule recognizer that converts token streams into RDF elements
pub trait RuleRecognizer {
    /// The token recognizer this rule works with
    type TokenRecognizer: TokenRecognizer;

    /// The output produced (Triple, Quad, etc.)
    type Output;

    /// Parsing context (prefixes, base IRI, etc.)
    type Context;

    /// Recognize the next RDF element from a token
    fn recognize_next(
        self,
        token: TokenOrLineJump<<Self::TokenRecognizer as TokenRecognizer>::Token<'_>>,
        context: &mut Self::Context,
        results: &mut Vec<Self::Output>,
        errors: &mut Vec<RuleRecognizerError>,
    ) -> Self;
}

/// A streaming parser that combines tokenization and rule recognition
pub struct StreamingParser<R, T: crate::toolkit::lexer::TokenRecognizer, P: RuleRecognizer> {
    tokenizer: crate::toolkit::lexer::StreamingTokenizer<R, T>,
    rule_recognizer: P,
    context: P::Context,
    _phantom: PhantomData<P>,
}

impl<R: BufRead, T: TokenRecognizer, P: RuleRecognizer<TokenRecognizer = T>>
    StreamingParser<R, T, P>
{
    /// Create a new streaming parser
    pub fn new(
        tokenizer: crate::toolkit::lexer::StreamingTokenizer<R, T>,
        rule_recognizer: P,
        context: P::Context,
    ) -> Self {
        Self {
            tokenizer,
            rule_recognizer,
            context,
            _phantom: PhantomData,
        }
    }
}

impl<R: BufRead, T: TokenRecognizer, P: RuleRecognizer<TokenRecognizer = T>> Iterator
    for StreamingParser<R, T, P>
where
    P: Clone,
{
    type Item = TurtleResult<P::Output>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.tokenizer.next() {
                None => return None, // EOF
                Some(Err(e)) => {
                    return Some(Err(TurtleParseError::syntax(
                        crate::error::TurtleSyntaxError::Generic {
                            message: e.to_string(),
                            position: self.tokenizer.position(),
                        },
                    )))
                }
                Some(Ok(token)) => {
                    let mut results = Vec::new();
                    let mut errors = Vec::new();

                    self.rule_recognizer = self.rule_recognizer.clone().recognize_next(
                        token,
                        &mut self.context,
                        &mut results,
                        &mut errors,
                    );

                    // Handle errors
                    if !errors.is_empty() {
                        return Some(Err(TurtleParseError::syntax(
                            crate::error::TurtleSyntaxError::Generic {
                                message: format!("Rule recognition error: {:?}", errors[0]),
                                position: self.tokenizer.position(),
                            },
                        )));
                    }

                    // Return first result if any
                    if let Some(result) = results.into_iter().next() {
                        return Some(Ok(result));
                    }

                    // Continue if no results (e.g., whitespace, comments)
                }
            }
        }
    }
}

/// A generic parser trait for all RDF formats
pub trait Parser<Output> {
    /// Parse from a reader
    fn parse<R: Read>(&self, reader: R) -> TurtleResult<Vec<Output>>;

    /// Create an iterator for streaming parsing
    fn for_reader<R: BufRead + 'static>(
        &self,
        reader: R,
    ) -> Box<dyn Iterator<Item = TurtleResult<Output>>>;
}

/// Async parser trait for Tokio integration
#[cfg(feature = "async-tokio")]
pub trait AsyncParser<Output> {
    /// Parse from an async reader
    async fn parse_async<R: tokio::io::AsyncRead + Unpin>(
        &self,
        reader: R,
    ) -> TurtleResult<Vec<Output>>;

    /// Create an async stream for streaming parsing
    fn for_async_reader<R: tokio::io::AsyncBufRead + Unpin>(
        &self,
        reader: R,
    ) -> Box<dyn futures::Stream<Item = TurtleResult<Output>> + Unpin>;
}

/// Context for parsing operations
#[derive(Debug, Clone, Default)]
pub struct ParsingContext {
    /// Base IRI for resolving relative IRIs
    pub base_iri: Option<String>,
    /// Prefix declarations
    pub prefixes: std::collections::HashMap<String, String>,
    /// Blank node ID generator state
    pub blank_node_counter: usize,
}

impl ParsingContext {
    /// Create a new parsing context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base IRI
    pub fn with_base_iri(mut self, base_iri: String) -> Self {
        self.base_iri = Some(base_iri);
        self
    }

    /// Add a prefix declaration
    pub fn add_prefix(&mut self, prefix: String, iri: String) {
        self.prefixes.insert(prefix, iri);
    }

    /// Resolve a prefixed name
    pub fn resolve_prefixed_name(&self, prefix: &str, local: &str) -> Option<String> {
        self.prefixes.get(prefix).map(|iri| format!("{iri}{local}"))
    }

    /// Generate a new blank node ID
    pub fn generate_blank_node_id(&mut self) -> String {
        let id = format!("_:b{}", self.blank_node_counter);
        self.blank_node_counter += 1;
        id
    }

    /// Resolve a relative IRI against the base IRI
    pub fn resolve_iri(&self, iri: &str) -> String {
        if let Some(ref base) = self.base_iri {
            // Simple resolution - in practice would use proper IRI resolution
            if iri.starts_with('#') || iri.starts_with('/') {
                format!("{base}{iri}")
            } else {
                iri.to_string()
            }
        } else {
            iri.to_string()
        }
    }
}
