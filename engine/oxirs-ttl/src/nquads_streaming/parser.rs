//! N-Quads line parser: convert tokens into [`StreamedQuad`] values.

use crate::nquads_streaming::{
    lexer::{NQuadsLexer, Token},
    NQuadsParseError, StreamedLiteral, StreamedQuad, StreamedTerm,
};

/// Parse one N-Quads line.
///
/// - Returns `Ok(None)` for blank lines and comment-only lines.
/// - Returns `Ok(Some(quad))` for valid N-Quads statements.
/// - Returns `Err(_)` for syntactically invalid lines.
pub fn parse_line(line: &str, line_num: usize) -> Result<Option<StreamedQuad>, NQuadsParseError> {
    let trimmed = line.trim();

    // Skip blank lines and comments
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(None);
    }

    let tokens = NQuadsLexer::tokenize_line(trimmed, line_num)?;

    if tokens.is_empty() {
        return Ok(None);
    }

    // Filter out structural tokens that are handled inline (Dot is at the end)
    // Expected token sequences:
    //   Triple:  <subject> <predicate> <object> .
    //   Quad:    <subject> <predicate> <object> <graph> .
    //
    // The lexer already folds @lang and ^^<dt> into StringLiteral tokens so
    // the token stream at this point contains only:
    //   IriRef | BlankNodeLabel | StringLiteral | Dot
    // (Caret and At should never appear here; they were consumed by the lexer.)

    // Collect term tokens, verify trailing Dot
    let mut term_tokens: Vec<&Token> = Vec::new();
    let mut dot_seen = false;

    for token in &tokens {
        match token {
            Token::Dot => {
                dot_seen = true;
            }
            Token::Caret | Token::At => {
                // These should never appear after lexer folding; treat as error.
                return Err(NQuadsParseError::InvalidLine {
                    line: line_num,
                    message: format!("Unexpected structural token: {:?}", token),
                });
            }
            _ => {
                if dot_seen {
                    return Err(NQuadsParseError::InvalidLine {
                        line: line_num,
                        message: "Unexpected token after '.'".to_string(),
                    });
                }
                term_tokens.push(token);
            }
        }
    }

    if !dot_seen {
        return Err(NQuadsParseError::InvalidLine {
            line: line_num,
            message: "N-Quads statement must end with '.'".to_string(),
        });
    }

    // We expect 3 terms (triple) or 4 terms (quad)
    match term_tokens.len() {
        3 => {
            let subject = parse_term(term_tokens[0], line_num)?;
            validate_subject(&subject, line_num)?;

            let predicate = parse_term(term_tokens[1], line_num)?;
            validate_predicate(&predicate, line_num)?;

            let object = parse_term(term_tokens[2], line_num)?;

            Ok(Some(StreamedQuad {
                subject,
                predicate,
                object,
                graph_name: None,
            }))
        }
        4 => {
            let subject = parse_term(term_tokens[0], line_num)?;
            validate_subject(&subject, line_num)?;

            let predicate = parse_term(term_tokens[1], line_num)?;
            validate_predicate(&predicate, line_num)?;

            let object = parse_term(term_tokens[2], line_num)?;

            let graph_name = parse_term(term_tokens[3], line_num)?;
            validate_graph_name(&graph_name, line_num)?;

            Ok(Some(StreamedQuad {
                subject,
                predicate,
                object,
                graph_name: Some(graph_name),
            }))
        }
        n => Err(NQuadsParseError::InvalidLine {
            line: line_num,
            message: format!("Expected 3 or 4 terms before '.', got {}", n),
        }),
    }
}

/// Convert a [`Token`] into a [`StreamedTerm`].
///
/// This is a public function so callers can re-use it for custom processing.
pub fn parse_term(token: &Token, line_num: usize) -> Result<StreamedTerm, NQuadsParseError> {
    match token {
        Token::IriRef(iri) => Ok(StreamedTerm::NamedNode(iri.clone())),
        Token::BlankNodeLabel(label) => Ok(StreamedTerm::BlankNode(label.clone())),
        Token::StringLiteral {
            value,
            lang,
            datatype,
        } => Ok(StreamedTerm::Literal(StreamedLiteral {
            value: value.clone(),
            datatype: datatype.clone(),
            language: lang.clone(),
        })),
        other => Err(NQuadsParseError::InvalidLine {
            line: line_num,
            message: format!("Cannot convert token {:?} to RDF term", other),
        }),
    }
}

// ============================================================================
// Validators
// ============================================================================

/// Subject must be a NamedNode or BlankNode (not a Literal).
fn validate_subject(term: &StreamedTerm, line_num: usize) -> Result<(), NQuadsParseError> {
    match term {
        StreamedTerm::NamedNode(_) | StreamedTerm::BlankNode(_) => Ok(()),
        StreamedTerm::Literal(_) => Err(NQuadsParseError::InvalidLine {
            line: line_num,
            message: "Subject must be a named node or blank node, not a literal".to_string(),
        }),
    }
}

/// Predicate must be a NamedNode only (not BlankNode, not Literal).
fn validate_predicate(term: &StreamedTerm, line_num: usize) -> Result<(), NQuadsParseError> {
    match term {
        StreamedTerm::NamedNode(_) => Ok(()),
        StreamedTerm::BlankNode(_) => Err(NQuadsParseError::InvalidLine {
            line: line_num,
            message: "Predicate must be a named node (IRIs only), not a blank node".to_string(),
        }),
        StreamedTerm::Literal(_) => Err(NQuadsParseError::InvalidLine {
            line: line_num,
            message: "Predicate must be a named node (IRIs only), not a literal".to_string(),
        }),
    }
}

/// Graph name must be a NamedNode or BlankNode (not a Literal).
fn validate_graph_name(term: &StreamedTerm, line_num: usize) -> Result<(), NQuadsParseError> {
    match term {
        StreamedTerm::NamedNode(_) | StreamedTerm::BlankNode(_) => Ok(()),
        StreamedTerm::Literal(_) => Err(NQuadsParseError::InvalidLine {
            line: line_num,
            message: "Graph name must be a named node or blank node, not a literal".to_string(),
        }),
    }
}
