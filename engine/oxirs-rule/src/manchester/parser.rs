//! Recursive-descent parser for OWL Manchester Syntax.
//!
//! Implements the following grammar (precedence: `or` < `and` < `not` <
//! property restrictions and primary terms):
//!
//! ```text
//! expr     := or_expr
//! or_expr  := and_expr ('or'  and_expr)*
//! and_expr := not_expr ('and' not_expr)*
//! not_expr := 'not' primary | primary
//! primary  := ident rest
//!           | '{' ident+ '}'
//!           | '(' expr ')'
//! rest     := 'some'    primary
//!           | 'only'    primary
//!           | 'min'     Number primary?
//!           | 'max'     Number primary?
//!           | 'exactly' Number primary?
//!           | 'value'   ident
//!           | ε
//! ```
//!
//! The parser returns a fully-structured [`ManchesterExpr`] on success or a
//! [`ManchesterError::ParseError`] on failure.

use super::{
    lexer::{tokenize, Token},
    ManchesterError, ManchesterExpr,
};

/// Internal parser state.  The token stream is stored as a slice-like view via
/// an index cursor so that we never need to copy tokens.
struct Parser {
    tokens: Vec<(Token, usize)>,
    /// Index of the *current* (not yet consumed) token.
    cursor: usize,
}

impl Parser {
    fn new(tokens: Vec<(Token, usize)>) -> Self {
        Self { tokens, cursor: 0 }
    }

    // ─── Cursor helpers ────────────────────────────────────────────────────────

    /// Peek at the current token without consuming it.
    fn peek(&self) -> &Token {
        &self.tokens[self.cursor].0
    }

    /// Consume and return the current token, advancing the cursor.
    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.cursor].0;
        if self.cursor + 1 < self.tokens.len() {
            self.cursor += 1;
        }
        tok
    }

    /// Consume the current token and return the index *before* advancing
    /// (used when we need the token value after the advance call).
    fn consume(&mut self) -> Token {
        let tok = self.tokens[self.cursor].0.clone();
        if self.cursor + 1 < self.tokens.len() {
            self.cursor += 1;
        }
        tok
    }

    /// Produce a [`ManchesterError::ParseError`] anchored at the current token.
    fn err(&self, msg: impl Into<String>) -> ManchesterError {
        ManchesterError::ParseError {
            pos: self.cursor,
            msg: msg.into(),
        }
    }

    // ─── Grammar rules ─────────────────────────────────────────────────────────

    /// `expr := or_expr`
    fn parse_expr(&mut self) -> Result<ManchesterExpr, ManchesterError> {
        self.parse_or_expr()
    }

    /// `or_expr := and_expr ('or' and_expr)*`
    fn parse_or_expr(&mut self) -> Result<ManchesterExpr, ManchesterError> {
        let first = self.parse_and_expr()?;
        if !matches!(self.peek(), Token::Or) {
            return Ok(first);
        }
        let mut arms = vec![first];
        while matches!(self.peek(), Token::Or) {
            self.advance(); // consume `or`
            arms.push(self.parse_and_expr()?);
        }
        Ok(ManchesterExpr::Or(arms))
    }

    /// `and_expr := not_expr ('and' not_expr)*`
    fn parse_and_expr(&mut self) -> Result<ManchesterExpr, ManchesterError> {
        let first = self.parse_not_expr()?;
        if !matches!(self.peek(), Token::And) {
            return Ok(first);
        }
        let mut arms = vec![first];
        while matches!(self.peek(), Token::And) {
            self.advance(); // consume `and`
            arms.push(self.parse_not_expr()?);
        }
        Ok(ManchesterExpr::And(arms))
    }

    /// `not_expr := 'not' primary | primary`
    fn parse_not_expr(&mut self) -> Result<ManchesterExpr, ManchesterError> {
        if matches!(self.peek(), Token::Not) {
            self.advance(); // consume `not`
            let inner = self.parse_primary()?;
            return Ok(ManchesterExpr::Not(Box::new(inner)));
        }
        self.parse_primary()
    }

    /// ```text
    /// primary := ident rest
    ///          | '{' ident+ '}'
    ///          | '(' expr ')'
    /// ```
    fn parse_primary(&mut self) -> Result<ManchesterExpr, ManchesterError> {
        match self.peek().clone() {
            Token::Ident(_) => {
                let name = match self.consume() {
                    Token::Ident(s) => s,
                    _ => unreachable!(),
                };
                self.parse_rest(name)
            }
            Token::LBrace => {
                self.advance(); // consume `{`
                let mut individuals: Vec<String> = Vec::new();
                loop {
                    match self.peek().clone() {
                        Token::RBrace => {
                            self.advance(); // consume `}`
                            break;
                        }
                        Token::Ident(_) => {
                            let ind = match self.consume() {
                                Token::Ident(s) => s,
                                _ => unreachable!(),
                            };
                            individuals.push(ind);
                        }
                        Token::Eof => {
                            return Err(self.err("unexpected end of input inside `{…}`"));
                        }
                        other => {
                            return Err(self.err(format!(
                                "expected identifier or `}}` inside `{{…}}`, got {other}"
                            )));
                        }
                    }
                }
                if individuals.is_empty() {
                    return Err(ManchesterError::ParseError {
                        pos: self.cursor,
                        msg: "`{…}` must contain at least one individual".to_string(),
                    });
                }
                Ok(ManchesterExpr::OneOf(individuals))
            }
            Token::LParen => {
                self.advance(); // consume `(`
                let inner = self.parse_expr()?;
                match self.peek() {
                    Token::RParen => {
                        self.advance(); // consume `)`
                    }
                    other => {
                        return Err(self.err(format!("expected `)`, got {other}")));
                    }
                }
                Ok(inner)
            }
            Token::Eof => Err(self.err("unexpected end of input — expected a class expression")),
            other => Err(self.err(format!("expected class name, `{{`, or `(`, got {other}"))),
        }
    }

    /// Parse the optional continuation after an identifier:
    ///
    /// ```text
    /// rest := 'some'    primary
    ///       | 'only'    primary
    ///       | 'min'     Number primary?
    ///       | 'max'     Number primary?
    ///       | 'exactly' Number primary?
    ///       | 'value'   ident
    ///       | ε          → Class(name)
    /// ```
    fn parse_rest(&mut self, property: String) -> Result<ManchesterExpr, ManchesterError> {
        match self.peek().clone() {
            Token::Some => {
                self.advance();
                let filler = self.parse_primary()?;
                Ok(ManchesterExpr::Some {
                    property,
                    filler: Box::new(filler),
                })
            }
            Token::Only => {
                self.advance();
                let filler = self.parse_primary()?;
                Ok(ManchesterExpr::Only {
                    property,
                    filler: Box::new(filler),
                })
            }
            Token::Min => {
                self.advance();
                let cardinality = self.expect_number()?;
                let filler = self.maybe_primary()?;
                Ok(ManchesterExpr::Min {
                    property,
                    cardinality,
                    filler: filler.map(Box::new),
                })
            }
            Token::Max => {
                self.advance();
                let cardinality = self.expect_number()?;
                let filler = self.maybe_primary()?;
                Ok(ManchesterExpr::Max {
                    property,
                    cardinality,
                    filler: filler.map(Box::new),
                })
            }
            Token::Exactly => {
                self.advance();
                let cardinality = self.expect_number()?;
                let filler = self.maybe_primary()?;
                Ok(ManchesterExpr::Exactly {
                    property,
                    cardinality,
                    filler: filler.map(Box::new),
                })
            }
            Token::Value => {
                self.advance();
                let individual = match self.peek().clone() {
                    Token::Ident(_) => match self.consume() {
                        Token::Ident(s) => s,
                        _ => unreachable!(),
                    },
                    other => {
                        return Err(self.err(format!(
                            "expected individual name after `value`, got {other}"
                        )));
                    }
                };
                Ok(ManchesterExpr::HasValue {
                    property,
                    individual,
                })
            }
            // ε — bare class name
            _ => Ok(ManchesterExpr::Class(property)),
        }
    }

    // ─── Helpers ───────────────────────────────────────────────────────────────

    /// Consume and return the current token if it is a [`Token::Number`];
    /// otherwise return a parse error.
    fn expect_number(&mut self) -> Result<u32, ManchesterError> {
        match self.peek().clone() {
            Token::Number(n) => {
                self.advance();
                Ok(n)
            }
            other => Err(self.err(format!("expected a number, got {other}"))),
        }
    }

    /// Attempt to parse an optional primary expression.
    ///
    /// Returns `Some(expr)` if the current token can start a primary, or
    /// `None` if it cannot (without consuming any tokens).
    fn maybe_primary(&mut self) -> Result<Option<ManchesterExpr>, ManchesterError> {
        match self.peek() {
            Token::Ident(_) | Token::LBrace | Token::LParen => Ok(Some(self.parse_primary()?)),
            _ => Ok(None),
        }
    }
}

/// Parse an OWL Manchester Syntax class expression string.
///
/// The whole input must be consumed; any trailing tokens produce an error.
///
/// # Errors
///
/// Returns [`ManchesterError::LexError`] if tokenization fails, or
/// [`ManchesterError::ParseError`] if the token stream does not conform to the
/// Manchester Syntax grammar.
pub fn parse(input: &str) -> Result<ManchesterExpr, ManchesterError> {
    if input.trim().is_empty() {
        return Err(ManchesterError::ParseError {
            pos: 0,
            msg: "input is empty — expected a class expression".to_string(),
        });
    }

    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;

    // Ensure the entire input has been consumed
    match parser.peek() {
        Token::Eof => {}
        other => {
            return Err(ManchesterError::ParseError {
                pos: parser.cursor,
                msg: format!(
                    "unexpected token {other} after class expression — expected end of input"
                ),
            });
        }
    }

    Ok(expr)
}
