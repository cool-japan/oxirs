//! RDF-star annotation syntax module.
//!
//! Implements the `{| ... |}` shorthand annotation syntax from the RDF-star specification.
//!
//! ## Background
//!
//! Turtle-star allows annotation shorthand syntax:
//! ```text
//! <subject> <predicate> <object> {| <ann_pred> <ann_obj> |} .
//! ```
//!
//! This is equivalent to the explicit form:
//! ```text
//! << <subject> <predicate> <object> >> <ann_pred> <ann_obj> .
//! ```
//!
//! The `{| ... |}` block introduces annotation triples about the preceding triple.
//!
//! ## Modules
//! - [`tokenizer`] — tokenize `{| ... |}` blocks
//! - [`expander`] — expand annotation shorthand to explicit triples

pub mod expander;
pub mod tests;
pub mod tokenizer;

pub use expander::{annotations_to_turtle, expand_annotations, to_explicit_turtle};
pub use tokenizer::{find_annotation_blocks, tokenize_annotation_block, AnnotationToken};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error types for annotation syntax parsing
#[derive(Debug, Clone, Error)]
pub enum AnnotationSyntaxError {
    /// Unexpected token during parsing
    #[error("unexpected token: expected {expected}, got {got}")]
    UnexpectedToken { expected: String, got: String },

    /// Annotation block was opened but never closed
    #[error("unclosed annotation block: missing '|}}' terminator")]
    UnclosedAnnotation,

    /// Subject term is not valid in this context
    #[error("invalid subject: {0}")]
    InvalidSubject(String),

    /// Predicate term is not valid (e.g. blank node as predicate)
    #[error("invalid predicate: {0}")]
    InvalidPredicate(String),

    /// Empty annotation block `{| |}` is not allowed
    #[error("empty annotation block")]
    EmptyAnnotation,
}

/// A literal value with optional datatype or language tag
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotationLiteral {
    /// Lexical value of the literal
    pub value: String,
    /// Optional XSD datatype URI
    pub datatype: Option<String>,
    /// Optional BCP47 language tag
    pub language: Option<String>,
}

/// A term in an RDF-star triple (subject, predicate, or object position)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StarTerm {
    /// An IRI / named node
    NamedNode(String),
    /// A blank node with identifier
    BlankNode(String),
    /// A literal value
    Literal(AnnotationLiteral),
    /// A quoted triple (nested `<< s p o >>`)
    QuotedTriple(Box<RdfStarTriple>),
}

impl StarTerm {
    /// Return the IRI if this is a named node, otherwise `None`
    pub fn as_named_node(&self) -> Option<&str> {
        if let Self::NamedNode(iri) = self {
            Some(iri.as_str())
        } else {
            None
        }
    }

    /// Return the blank-node identifier if this is a blank node, otherwise `None`
    pub fn as_blank_node(&self) -> Option<&str> {
        if let Self::BlankNode(id) = self {
            Some(id.as_str())
        } else {
            None
        }
    }

    /// Serialize this term to Turtle-star notation
    pub fn to_turtle(&self) -> String {
        match self {
            Self::NamedNode(iri) => format!("<{}>", iri),
            Self::BlankNode(id) => format!("_:{}", id),
            Self::Literal(lit) => {
                if let Some(lang) = &lit.language {
                    format!("\"{}\"@{}", escape_turtle_string(&lit.value), lang)
                } else if let Some(dt) = &lit.datatype {
                    format!("\"{}\"^^<{}>", escape_turtle_string(&lit.value), dt)
                } else {
                    format!("\"{}\"", escape_turtle_string(&lit.value))
                }
            }
            Self::QuotedTriple(triple) => {
                format!(
                    "<< {} {} {} >>",
                    triple.subject.to_turtle(),
                    triple.predicate.to_turtle(),
                    triple.object.to_turtle()
                )
            }
        }
    }
}

/// An RDF-star triple (subject-predicate-object), used both as base triples
/// and as quoted/nested triples within annotation contexts
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RdfStarTriple {
    pub subject: StarTerm,
    pub predicate: StarTerm,
    pub object: StarTerm,
}

impl RdfStarTriple {
    /// Construct a new RDF-star triple
    pub fn new(subject: StarTerm, predicate: StarTerm, object: StarTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Serialize to Turtle-star notation without annotation shorthand
    pub fn to_turtle(&self) -> String {
        format!(
            "{} {} {} .",
            self.subject.to_turtle(),
            self.predicate.to_turtle(),
            self.object.to_turtle()
        )
    }

    /// Serialize as a quoted triple reference `<< s p o >>`
    pub fn to_quoted(&self) -> String {
        format!(
            "<< {} {} {} >>",
            self.subject.to_turtle(),
            self.predicate.to_turtle(),
            self.object.to_turtle()
        )
    }
}

/// A predicate-object pair inside an annotation block
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotationPair {
    /// Predicate of the annotation (must be a named node IRI)
    pub predicate: String,
    /// Object of the annotation
    pub object: AnnotationValue,
}

/// Object value in an annotation pair
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnnotationValue {
    /// An IRI
    NamedNode(String),
    /// A blank node
    BlankNode(String),
    /// A literal
    Literal(AnnotationLiteral),
}

impl AnnotationValue {
    /// Serialize to Turtle-star notation
    pub fn to_turtle(&self) -> String {
        match self {
            Self::NamedNode(iri) => format!("<{}>", iri),
            Self::BlankNode(id) => format!("_:{}", id),
            Self::Literal(lit) => {
                if let Some(lang) = &lit.language {
                    format!("\"{}\"@{}", escape_turtle_string(&lit.value), lang)
                } else if let Some(dt) = &lit.datatype {
                    format!("\"{}\"^^<{}>", escape_turtle_string(&lit.value), dt)
                } else {
                    format!("\"{}\"", escape_turtle_string(&lit.value))
                }
            }
        }
    }
}

/// A base triple together with its annotation pairs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedTriple {
    /// The base triple being annotated
    pub base: RdfStarTriple,
    /// Annotation predicate-object pairs collected from the `{| ... |}` block
    pub annotations: Vec<AnnotationPair>,
}

impl AnnotatedTriple {
    /// Construct an annotated triple
    pub fn new(base: RdfStarTriple, annotations: Vec<AnnotationPair>) -> Self {
        Self { base, annotations }
    }
}

/// Main struct for parsing annotation blocks from token streams
pub struct AnnotationParser;

impl AnnotationParser {
    /// Create a new `AnnotationParser`
    pub fn new() -> Self {
        Self
    }

    /// Parse an annotation block from the input string.
    ///
    /// The input is expected to be the Turtle-star text of a single annotated
    /// triple — for example:
    /// ```text
    /// <s> <p> <o> {| <ap> <ao> |}
    /// ```
    pub fn parse_annotated_triple(
        &self,
        input: &str,
    ) -> Result<AnnotatedTriple, AnnotationSyntaxError> {
        // Locate the `{|` delimiter
        let annotation_start = input
            .find("{|")
            .ok_or(AnnotationSyntaxError::UnclosedAnnotation)?;

        let base_str = input[..annotation_start].trim();
        let rest = &input[annotation_start + 2..]; // skip "{|"

        let annotation_end = rest
            .rfind("|}")
            .ok_or(AnnotationSyntaxError::UnclosedAnnotation)?;
        let annotation_body = rest[..annotation_end].trim();

        // Parse base triple
        let base = self.parse_triple(base_str)?;

        // Tokenize and parse annotation pairs
        let tokens = tokenize_annotation_block(annotation_body)?;
        let annotations = self.parse_annotation_pairs(&tokens)?;

        Ok(AnnotatedTriple::new(base, annotations))
    }

    // ---- private helpers ----

    fn parse_triple(&self, input: &str) -> Result<RdfStarTriple, AnnotationSyntaxError> {
        // Handle quoted-triple subjects: << s p o >> p2 o2
        if let Some(inner) = input.strip_prefix("<<") {
            let inner = inner.trim_start();
            let close = inner
                .rfind(">>")
                .ok_or_else(|| AnnotationSyntaxError::InvalidSubject(input.to_string()))?;
            let quoted_str = inner[..close].trim();
            let quoted = self.parse_triple(quoted_str)?;
            let after = inner[close + 2..].trim();

            let (pred_str, obj_str) =
                split_two_terms(after).ok_or_else(|| AnnotationSyntaxError::UnexpectedToken {
                    expected: "predicate object".to_string(),
                    got: after.to_string(),
                })?;

            let subject = StarTerm::QuotedTriple(Box::new(quoted));
            let predicate = self.parse_predicate_term(pred_str)?;
            let object = self.parse_object_term(obj_str)?;
            return Ok(RdfStarTriple::new(subject, predicate, object));
        }

        let (subj_str, rest) =
            split_first_term(input).ok_or_else(|| AnnotationSyntaxError::UnexpectedToken {
                expected: "subject".to_string(),
                got: input.to_string(),
            })?;

        let (pred_str, obj_str) =
            split_two_terms(rest.trim()).ok_or_else(|| AnnotationSyntaxError::UnexpectedToken {
                expected: "predicate object".to_string(),
                got: rest.to_string(),
            })?;

        let subject = self.parse_subject_term(subj_str)?;
        let predicate = self.parse_predicate_term(pred_str)?;
        let object = self.parse_object_term(obj_str)?;

        Ok(RdfStarTriple::new(subject, predicate, object))
    }

    fn parse_subject_term(&self, s: &str) -> Result<StarTerm, AnnotationSyntaxError> {
        self.parse_term(s)
    }

    fn parse_predicate_term(&self, s: &str) -> Result<StarTerm, AnnotationSyntaxError> {
        let term = self.parse_term(s)?;
        // Predicates must be named nodes
        if matches!(term, StarTerm::BlankNode(_)) {
            return Err(AnnotationSyntaxError::InvalidPredicate(
                "blank node cannot be used as predicate".to_string(),
            ));
        }
        Ok(term)
    }

    fn parse_object_term(&self, s: &str) -> Result<StarTerm, AnnotationSyntaxError> {
        self.parse_term(s)
    }

    fn parse_term(&self, s: &str) -> Result<StarTerm, AnnotationSyntaxError> {
        let s = s.trim().trim_end_matches('.');
        let s = s.trim();

        if s.starts_with('<') {
            // Named node
            let iri = s
                .strip_prefix('<')
                .and_then(|t| t.strip_suffix('>'))
                .ok_or_else(|| AnnotationSyntaxError::UnexpectedToken {
                    expected: "IRI like <...>".to_string(),
                    got: s.to_string(),
                })?;
            return Ok(StarTerm::NamedNode(iri.to_string()));
        }

        if let Some(rest) = s.strip_prefix("_:") {
            return Ok(StarTerm::BlankNode(rest.to_string()));
        }

        if s.starts_with('"') {
            return self.parse_literal_term(s);
        }

        Err(AnnotationSyntaxError::UnexpectedToken {
            expected: "<IRI>, _:blank, or \"literal\"".to_string(),
            got: s.to_string(),
        })
    }

    fn parse_literal_term(&self, s: &str) -> Result<StarTerm, AnnotationSyntaxError> {
        // Strip leading '"'
        let rest = s
            .strip_prefix('"')
            .ok_or_else(|| AnnotationSyntaxError::UnexpectedToken {
                expected: "string literal".to_string(),
                got: s.to_string(),
            })?;

        // Find closing '"' (handle escapes)
        let mut chars = rest.char_indices();
        let mut close_pos = None;
        let mut escaped = false;
        for (i, c) in chars.by_ref() {
            if escaped {
                escaped = false;
                continue;
            }
            if c == '\\' {
                escaped = true;
                continue;
            }
            if c == '"' {
                close_pos = Some(i);
                break;
            }
        }
        let close_pos = close_pos.ok_or_else(|| AnnotationSyntaxError::UnexpectedToken {
            expected: "closing quote for literal".to_string(),
            got: s.to_string(),
        })?;

        let value = rest[..close_pos].to_string();
        let after = rest[close_pos + 1..].trim();

        if let Some(lang_rest) = after.strip_prefix('@') {
            // Language-tagged literal
            let lang = lang_rest.trim().to_string();
            return Ok(StarTerm::Literal(AnnotationLiteral {
                value,
                language: Some(lang),
                datatype: None,
            }));
        }

        if let Some(dt_rest) = after.strip_prefix("^^") {
            // Typed literal
            let dt_rest = dt_rest.trim();
            let datatype = if dt_rest.starts_with('<') {
                dt_rest
                    .strip_prefix('<')
                    .and_then(|t| t.strip_suffix('>'))
                    .unwrap_or(dt_rest)
                    .to_string()
            } else {
                dt_rest.to_string()
            };
            return Ok(StarTerm::Literal(AnnotationLiteral {
                value,
                language: None,
                datatype: Some(datatype),
            }));
        }

        // Plain string literal
        Ok(StarTerm::Literal(AnnotationLiteral {
            value,
            language: None,
            datatype: None,
        }))
    }

    fn parse_annotation_pairs(
        &self,
        tokens: &[AnnotationToken],
    ) -> Result<Vec<AnnotationPair>, AnnotationSyntaxError> {
        if tokens.is_empty() {
            return Err(AnnotationSyntaxError::EmptyAnnotation);
        }

        let mut pairs = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            // Expect a predicate (NamedNode)
            let predicate = match &tokens[i] {
                AnnotationToken::NamedNode(iri) => iri.clone(),
                AnnotationToken::Dot | AnnotationToken::Semicolon | AnnotationToken::Comma => {
                    i += 1;
                    continue;
                }
                other => {
                    return Err(AnnotationSyntaxError::InvalidPredicate(format!(
                        "expected named node as predicate, got {:?}",
                        other
                    )));
                }
            };
            i += 1;

            // Expect an object
            if i >= tokens.len() {
                return Err(AnnotationSyntaxError::UnexpectedToken {
                    expected: "object after predicate".to_string(),
                    got: "end of tokens".to_string(),
                });
            }

            let object = match &tokens[i] {
                AnnotationToken::NamedNode(iri) => AnnotationValue::NamedNode(iri.clone()),
                AnnotationToken::BlankNode(id) => AnnotationValue::BlankNode(id.clone()),
                AnnotationToken::Literal(value, lang, datatype) => {
                    AnnotationValue::Literal(AnnotationLiteral {
                        value: value.clone(),
                        language: lang.clone(),
                        datatype: datatype.clone(),
                    })
                }
                other => {
                    return Err(AnnotationSyntaxError::UnexpectedToken {
                        expected: "object value".to_string(),
                        got: format!("{:?}", other),
                    });
                }
            };
            i += 1;

            pairs.push(AnnotationPair { predicate, object });

            // Optional separator: `;` (new predicate-object pair), `,` (same predicate),
            // or implicit by continuing to next named node
            if i < tokens.len() {
                match &tokens[i] {
                    AnnotationToken::Semicolon | AnnotationToken::Comma | AnnotationToken::Dot => {
                        i += 1;
                    }
                    _ => {}
                }
            }
        }

        if pairs.is_empty() {
            return Err(AnnotationSyntaxError::EmptyAnnotation);
        }

        Ok(pairs)
    }
}

impl Default for AnnotationParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---- utility functions ----

/// Escape special characters in Turtle string literals
fn escape_turtle_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Split an input string into the first term and the remainder.
/// Handles `<...>`, `_:...`, `"..."`, and `"..."@lang` / `"..."^^<dt>`.
fn split_first_term(input: &str) -> Option<(&str, &str)> {
    let input = input.trim();

    if input.is_empty() {
        return None;
    }

    if input.starts_with('<') {
        let end = input.find('>')?;
        return Some((&input[..end + 1], &input[end + 1..]));
    }

    if let Some(rest) = input.strip_prefix("_:") {
        // blank node: up to next whitespace
        let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
        // "_:" is 2 bytes, so term occupies [0..end+2], remainder starts at end+2
        return Some((&input[..end + 2], &input[end + 2..]));
    }

    if input.starts_with('"') {
        // find closing quote
        let rest = input.strip_prefix('"').expect("starts_with checked above");
        let mut escaped = false;
        let mut close = None;
        for (i, c) in rest.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            if c == '\\' {
                escaped = true;
                continue;
            }
            if c == '"' {
                close = Some(i);
                break;
            }
        }
        let close = close?;
        let after = &rest[close + 1..];

        // Include optional @lang or ^^<dt>
        let extra_end = if let Some(lang_rest) = after.strip_prefix('@') {
            1 + lang_rest
                .find(|c: char| c.is_whitespace())
                .unwrap_or(lang_rest.len())
        } else if let Some(dt_rest) = after.strip_prefix("^^<") {
            let dt_end = dt_rest.find('>')?;
            3 + dt_end + 1
        } else {
            0
        };

        let term_end = 1 + close + 1 + extra_end; // leading '"' + content + closing '"' + suffix
        return Some((&input[..term_end], &input[term_end..]));
    }

    None
}

/// Split a string into exactly two whitespace-separated terms.
/// Returns `(first_term, second_term)` stripping surrounding whitespace.
fn split_two_terms(input: &str) -> Option<(&str, &str)> {
    let input = input.trim();
    let (first, rest) = split_first_term(input)?;
    let rest = rest.trim();
    if rest.is_empty() {
        return None;
    }
    let (second, _) = split_first_term(rest)?;
    Some((first, second))
}
