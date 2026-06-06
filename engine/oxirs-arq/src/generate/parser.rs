//! Hand-rolled parser for the SPARQL-Generate query language.
//!
//! Supports the following grammar subset:
//!
//! ```text
//! generate_query  ::= prefix_decl* 'GENERATE' '{' template_clause* '}'
//!                     ('ITERATOR' string_literal)?
//!                     'WHERE' '{' where_body '}'
//!
//! prefix_decl     ::= 'PREFIX' prefix_name ':' '<' iri '>'
//!
//! template_clause ::= string_literal '?' ident string_literal
//!                   | '?' ident
//!                   | string_literal
//!
//! where_clause    ::= 'WHERE' '{' .* '}'   (captured verbatim)
//! ```
//!
//! String literals may be single-quoted (`'...'`), double-quoted (`"..."`),
//! or triple-quoted (`'''...'''` / `"""..."""`).

use super::ast::{GenerateLiteral, GenerateQuery, TemplateClause};
use super::GenerateError;

// ─────────────────────────────────────────────────────────────────────────────
// Token
// ─────────────────────────────────────────────────────────────────────────────

/// Elementary tokens produced by the SPARQL-Generate tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// A reserved keyword: `GENERATE`, `WHERE`, `PREFIX`, `ITERATOR`, etc.
    Keyword(String),
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `?` — begins a variable reference in template clauses.
    QuestionMark,
    /// A plain identifier (variable name, prefix label, …).
    Ident(String),
    /// A string literal, with the surrounding quotes stripped.
    StringLit(String),
    /// Signals the end of the token stream.
    Eof,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer
// ─────────────────────────────────────────────────────────────────────────────

/// Tokenize a SPARQL-Generate query string into a flat list of `Token`s.
///
/// The tokenizer is deliberately lenient — it skips whitespace and line
/// comments (`#…`) and does not validate keyword casing beyond upper-casing
/// identifiers for keyword detection.
pub fn tokenize(input: &str) -> Vec<Token> {
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0usize;
    let mut tokens = Vec::new();

    while pos < chars.len() {
        // ── Skip whitespace ────────────────────────────────────────────────
        if chars[pos].is_whitespace() {
            pos += 1;
            continue;
        }

        // ── Line comment: # … \n ───────────────────────────────────────────
        if chars[pos] == '#' {
            while pos < chars.len() && chars[pos] != '\n' {
                pos += 1;
            }
            continue;
        }

        // ── Single-char tokens ─────────────────────────────────────────────
        if chars[pos] == '{' {
            tokens.push(Token::LBrace);
            pos += 1;
            continue;
        }
        if chars[pos] == '}' {
            tokens.push(Token::RBrace);
            pos += 1;
            continue;
        }
        if chars[pos] == '?' {
            tokens.push(Token::QuestionMark);
            pos += 1;
            continue;
        }

        // ── Skip colon and angle brackets (used in PREFIX decls) ───────────
        if chars[pos] == ':' || chars[pos] == '<' || chars[pos] == '>' {
            pos += 1;
            continue;
        }

        // ── Triple-quoted string literals: """…""" or '''…''' ─────────────
        if pos + 2 < chars.len()
            && ((chars[pos] == '"' && chars[pos + 1] == '"' && chars[pos + 2] == '"')
                || (chars[pos] == '\'' && chars[pos + 1] == '\'' && chars[pos + 2] == '\''))
        {
            let delim = chars[pos];
            pos += 3; // skip opening triple-quote
            let start = pos;
            while pos + 2 < chars.len()
                && !(chars[pos] == delim && chars[pos + 1] == delim && chars[pos + 2] == delim)
            {
                pos += 1;
            }
            let s: String = chars[start..pos].iter().collect();
            pos += 3; // skip closing triple-quote
            tokens.push(Token::StringLit(s));
            continue;
        }

        // ── Double-quoted string: "…" ──────────────────────────────────────
        if chars[pos] == '"' {
            pos += 1; // skip opening quote
            let start = pos;
            while pos < chars.len() && chars[pos] != '"' {
                if chars[pos] == '\\' && pos + 1 < chars.len() {
                    pos += 1; // skip escape character
                }
                pos += 1;
            }
            let s: String = chars[start..pos].iter().collect();
            pos += 1; // skip closing quote
            tokens.push(Token::StringLit(s));
            continue;
        }

        // ── Single-quoted string: '…' ──────────────────────────────────────
        if chars[pos] == '\'' {
            pos += 1; // skip opening quote
            let start = pos;
            while pos < chars.len() && chars[pos] != '\'' {
                if chars[pos] == '\\' && pos + 1 < chars.len() {
                    pos += 1;
                }
                pos += 1;
            }
            let s: String = chars[start..pos].iter().collect();
            pos += 1; // skip closing quote
            tokens.push(Token::StringLit(s));
            continue;
        }

        // ── Identifiers and keywords ───────────────────────────────────────
        if chars[pos].is_alphabetic() || chars[pos] == '_' {
            let start = pos;
            while pos < chars.len()
                && (chars[pos].is_alphanumeric() || chars[pos] == '_' || chars[pos] == '-')
            {
                pos += 1;
            }
            let word: String = chars[start..pos].iter().collect();
            let upper = word.to_uppercase();
            let tok = match upper.as_str() {
                "GENERATE" | "WHERE" | "PREFIX" | "ITERATOR" => Token::Keyword(upper),
                _ => Token::Ident(word),
            };
            tokens.push(tok);
            continue;
        }

        // ── Skip any other punctuation silently ────────────────────────────
        pos += 1;
    }

    tokens.push(Token::Eof);
    tokens
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser helpers
// ─────────────────────────────────────────────────────────────────────────────

/// A simple cursor-based view into a token slice.
struct Cursor<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Peek at the current token without consuming it.
    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    /// Consume and return the current token.
    fn next(&mut self) -> &Token {
        let tok = self.tokens.get(self.pos).unwrap_or(&Token::Eof);
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    /// Consume a `Keyword(kw)` token or return a `ParseError`.
    fn expect_keyword(&mut self, kw: &str) -> Result<(), GenerateError> {
        let pos = self.pos;
        let tok = self.next().clone();
        match tok {
            Token::Keyword(ref k) if k.eq_ignore_ascii_case(kw) => Ok(()),
            other => Err(GenerateError::ParseError {
                pos,
                msg: format!("expected keyword {kw}, got {other:?}"),
            }),
        }
    }

    /// Consume a `LBrace` or return a `ParseError`.
    fn expect_lbrace(&mut self) -> Result<(), GenerateError> {
        let pos = self.pos;
        let tok = self.next().clone();
        match tok {
            Token::LBrace => Ok(()),
            other => Err(GenerateError::ParseError {
                pos,
                msg: format!("expected '{{', got {other:?}"),
            }),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a SPARQL-Generate query from `input` and return a `GenerateQuery`.
///
/// # Errors
///
/// Returns `GenerateError::ParseError` if the input does not conform to the
/// supported grammar subset.
pub fn parse(input: &str) -> Result<GenerateQuery, GenerateError> {
    let tokens = tokenize(input);
    let mut cursor = Cursor::new(&tokens);

    // ── Optional PREFIX declarations ─────────────────────────────────────────
    let mut prefix_decls: Vec<(String, String)> = Vec::new();
    loop {
        // Check for PREFIX keyword without holding a borrow past next().
        let is_prefix = matches!(cursor.peek(), Token::Keyword(kw) if kw == "PREFIX");
        if !is_prefix {
            break;
        }
        cursor.next(); // consume PREFIX

        let pos_before_label = cursor.pos;
        let label_tok = cursor.next().clone();
        let label = match label_tok {
            Token::Ident(s) => s,
            other => {
                return Err(GenerateError::ParseError {
                    pos: pos_before_label,
                    msg: format!("expected prefix label after PREFIX, got {other:?}"),
                })
            }
        };

        // The IRI is captured as a StringLit token by the tokenizer
        // (angle brackets are stripped).
        let pos_before_iri = cursor.pos;
        let iri_tok = cursor.next().clone();
        let iri = match iri_tok {
            Token::StringLit(s) | Token::Ident(s) => s,
            other => {
                return Err(GenerateError::ParseError {
                    pos: pos_before_iri,
                    msg: format!("expected IRI after prefix label, got {other:?}"),
                })
            }
        };
        prefix_decls.push((label, iri));
    }

    // ── GENERATE keyword ─────────────────────────────────────────────────────
    cursor.expect_keyword("GENERATE")?;

    // ── GENERATE { template_clauses* } ──────────────────────────────────────
    cursor.expect_lbrace()?;

    // Collect tokens inside the GENERATE { } block (excluding the closing brace).
    let block_start = cursor.pos;
    let mut depth = 1usize;
    while depth > 0 {
        match cursor.peek() {
            Token::LBrace => {
                depth += 1;
                cursor.next();
            }
            Token::RBrace => {
                depth -= 1;
                if depth > 0 {
                    cursor.next();
                }
                // At depth==0 we leave the RBrace in place so expect_rbrace can consume it.
            }
            Token::Eof => {
                return Err(GenerateError::ParseError {
                    pos: cursor.pos,
                    msg: "unexpected EOF inside GENERATE block".to_string(),
                })
            }
            _ => {
                cursor.next();
            }
        }
    }
    let block_end = cursor.pos; // points to the closing RBrace token
    let block_tokens = &tokens[block_start..block_end];

    // Consume the closing `}`
    let pos_before_rbrace = cursor.pos;
    let rbrace_tok = cursor.next().clone();
    match rbrace_tok {
        Token::RBrace => {}
        other => {
            return Err(GenerateError::ParseError {
                pos: pos_before_rbrace,
                msg: format!("expected '}}' to close GENERATE block, got {other:?}"),
            })
        }
    }

    let template = parse_template_block(block_tokens)?;

    // ── Optional ITERATOR clause ──────────────────────────────────────────────
    let is_iterator = matches!(cursor.peek(), Token::Keyword(kw) if kw == "ITERATOR");
    let iterator = if is_iterator {
        cursor.next(); // consume ITERATOR
        let iter_tok = cursor.next().clone();
        match iter_tok {
            Token::StringLit(s) | Token::Ident(s) => Some(s),
            _ => None,
        }
    } else {
        None
    };

    // ── WHERE { body } ───────────────────────────────────────────────────────
    // We extract the WHERE body as a raw string from the original input.
    let where_body = parse_where_body(input)?;

    Ok(GenerateQuery {
        prefix_decls,
        template,
        where_body,
        iterator,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Template block parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse the token slice inside a GENERATE `{ ... }` block into a list of
/// `TemplateClause`s.
///
/// Grammar (per clause):
/// ```text
/// template_clause ::= string_literal '?' ident string_literal  -- prefix + var + suffix
///                   | string_literal '?' ident                  -- prefix + var
///                   | '?' ident string_literal                  -- var + suffix
///                   | '?' ident                                  -- bare variable
///                   | string_literal                            -- static text only
/// ```
fn parse_template_block(tokens: &[Token]) -> Result<Vec<TemplateClause>, GenerateError> {
    let mut clauses = Vec::new();
    let mut pos = 0usize;

    while pos < tokens.len() {
        match &tokens[pos] {
            // ── Clause starting with a string literal ──────────────────────
            Token::StringLit(prefix_text) => {
                let prefix_text = prefix_text.clone();
                pos += 1;
                // Peek: is the next token a `?` (var reference)?
                if pos < tokens.len() {
                    if let Token::QuestionMark = &tokens[pos] {
                        pos += 1; // consume `?`
                        let var_name = match tokens.get(pos) {
                            Some(Token::Ident(v)) => {
                                pos += 1;
                                v.clone()
                            }
                            other => {
                                return Err(GenerateError::ParseError {
                                    pos,
                                    msg: format!("expected identifier after '?', got {other:?}"),
                                })
                            }
                        };
                        // Optional suffix string literal
                        let suffix = if pos < tokens.len() {
                            if let Token::StringLit(s) = &tokens[pos] {
                                let s = s.clone();
                                pos += 1;
                                Some(s)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        clauses.push(TemplateClause {
                            prefix: Some(prefix_text),
                            expr: GenerateLiteral::Var(var_name),
                            suffix,
                        });
                        continue;
                    }
                }
                // No variable follows — emit a pure text clause.
                clauses.push(TemplateClause {
                    prefix: None,
                    expr: GenerateLiteral::Text(prefix_text),
                    suffix: None,
                });
            }

            // ── Clause starting with `?` (bare variable or var+suffix) ─────
            Token::QuestionMark => {
                pos += 1; // consume `?`
                let var_name = match tokens.get(pos) {
                    Some(Token::Ident(v)) => {
                        pos += 1;
                        v.clone()
                    }
                    other => {
                        return Err(GenerateError::ParseError {
                            pos,
                            msg: format!("expected identifier after '?', got {other:?}"),
                        })
                    }
                };
                // Optional suffix string literal
                let suffix = if pos < tokens.len() {
                    if let Token::StringLit(s) = &tokens[pos] {
                        let s = s.clone();
                        pos += 1;
                        Some(s)
                    } else {
                        None
                    }
                } else {
                    None
                };
                clauses.push(TemplateClause {
                    prefix: None,
                    expr: GenerateLiteral::Var(var_name),
                    suffix,
                });
            }

            // ── Skip unexpected tokens inside the block ────────────────────
            _ => {
                pos += 1;
            }
        }
    }

    Ok(clauses)
}

// ─────────────────────────────────────────────────────────────────────────────
// WHERE body extractor
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the raw WHERE body from the original input string.
///
/// We locate `WHERE` (case-insensitive) followed by `{`, then capture
/// everything up to (but not including) the matching closing `}`, respecting
/// nested braces.
fn parse_where_body(input: &str) -> Result<String, GenerateError> {
    // Find 'WHERE' keyword (case-insensitive).
    let upper = input.to_uppercase();
    let where_pos = upper
        .find("WHERE")
        .ok_or_else(|| GenerateError::ParseError {
            pos: 0,
            msg: "missing WHERE clause".to_string(),
        })?;

    // Find the `{` after WHERE.
    let after_where = &input[where_pos + 5..]; // skip "WHERE"
    let brace_offset = after_where
        .find('{')
        .ok_or_else(|| GenerateError::ParseError {
            pos: where_pos,
            msg: "expected '{' after WHERE".to_string(),
        })?;

    let body_start = where_pos + 5 + brace_offset + 1; // one past the `{`
    let chars: Vec<char> = input[body_start..].chars().collect();
    let mut depth = 1usize;
    let mut byte_len = 0usize;

    for ch in &chars {
        if depth == 0 {
            break;
        }
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
        byte_len += ch.len_utf8();
    }

    if depth != 0 {
        return Err(GenerateError::ParseError {
            pos: body_start,
            msg: "unclosed '{' in WHERE clause".to_string(),
        });
    }

    Ok(input[body_start..body_start + byte_len].trim().to_string())
}
