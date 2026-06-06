//! Hand-rolled recursive-descent parser for the GQL subset.
//!
//! Grammar supported:
//! ```text
//! GqlQuery     ::= MATCH GraphPattern (WHERE Predicate)? RETURN ReturnClause
//! GraphPattern ::= NodePattern (EdgePattern NodePattern)*
//! NodePattern  ::= '(' VarOpt LabelFilter? PropFilter? ')'
//! EdgePattern  ::= '-[' VarOpt LabelFilter? ']->'
//!                | '<-[' VarOpt LabelFilter? ']-'
//! VarOpt       ::= Ident?
//! LabelFilter  ::= ':' Ident
//! PropFilter   ::= '{' PropKV (',' PropKV)* '}'
//! PropKV       ::= Ident ':' Literal
//! ReturnClause ::= Ident (',' Ident)*
//! Predicate    ::= Ident '.' Ident '=' Literal
//! Literal      ::= '"' [^"]* '"' | NUMBER | 'true' | 'false'
//! ```

use super::ast::{
    EdgeDirection, EdgePattern, GqlLiteral, GqlPredicate, GqlQuery, NodePattern, PathSegment,
};
use crate::gql::GqlTranslateError;

// ─────────────────────────────────────────────────────────────────────────────
// Tokeniser
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal token types produced by the lexer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Token {
    /// An identifier or keyword (case-insensitive keywords are matched later).
    Ident(String),
    /// A double-quoted string literal (content already unescaped).
    Str(String),
    /// An integer literal.
    Int(i64),
    /// A floating-point literal.
    Float(f64),
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `:`
    Colon,
    /// `.`
    Dot,
    /// `,`
    Comma,
    /// `=`
    Eq,
    /// `-`
    Minus,
    /// `>`
    Gt,
    /// `<`
    Lt,
}

/// Tokenise the input string, returning a list of `(position, Token)` pairs.
///
/// Positions are byte offsets into `src`.
pub(crate) fn tokenise(src: &str) -> Result<Vec<(usize, Token)>, GqlTranslateError> {
    let chars: Vec<char> = src.chars().collect();
    let mut pos = 0usize;
    let mut tokens = Vec::new();

    // Byte offset helper: since we iterate over chars, we need to track the
    // corresponding byte offset.
    let byte_offsets: Vec<usize> = {
        let mut offs = vec![0usize];
        let mut running = 0usize;
        for c in &chars {
            running += c.len_utf8();
            offs.push(running);
        }
        offs
    };

    macro_rules! byte_pos {
        ($char_idx:expr) => {
            byte_offsets[$char_idx]
        };
    }

    while pos < chars.len() {
        let c = chars[pos];

        // Skip whitespace.
        if c.is_whitespace() {
            pos += 1;
            continue;
        }

        // Skip single-line comments (# … \n)
        if c == '#' {
            while pos < chars.len() && chars[pos] != '\n' {
                pos += 1;
            }
            continue;
        }

        let start = byte_pos!(pos);

        // String literal
        if c == '"' {
            pos += 1; // consume opening quote
            let mut s = String::new();
            loop {
                if pos >= chars.len() {
                    return Err(GqlTranslateError::ParseError {
                        pos: byte_pos!(pos),
                        msg: "Unterminated string literal".to_string(),
                    });
                }
                let ch = chars[pos];
                if ch == '"' {
                    pos += 1; // consume closing quote
                    break;
                }
                if ch == '\\' {
                    pos += 1;
                    if pos >= chars.len() {
                        return Err(GqlTranslateError::ParseError {
                            pos: byte_pos!(pos),
                            msg: "Escape sequence at end of input".to_string(),
                        });
                    }
                    match chars[pos] {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        '"' => s.push('"'),
                        '\\' => s.push('\\'),
                        other => {
                            s.push('\\');
                            s.push(other);
                        }
                    }
                    pos += 1;
                } else {
                    s.push(ch);
                    pos += 1;
                }
            }
            tokens.push((start, Token::Str(s)));
            continue;
        }

        // Number literal (integer or float), possibly negative only when
        // tokenised in context (the `-` sign is handled separately and
        // context-free sign handling occurs in the parser).
        if c.is_ascii_digit() {
            let num_start = pos;
            while pos < chars.len() && chars[pos].is_ascii_digit() {
                pos += 1;
            }
            // Check for decimal point.
            if pos + 1 < chars.len() && chars[pos] == '.' && chars[pos + 1].is_ascii_digit() {
                pos += 1; // consume '.'
                while pos < chars.len() && chars[pos].is_ascii_digit() {
                    pos += 1;
                }
                let raw: String = chars[num_start..pos].iter().collect();
                let v: f64 = raw.parse().map_err(|_| GqlTranslateError::ParseError {
                    pos: start,
                    msg: format!("Invalid float literal: {raw}"),
                })?;
                tokens.push((start, Token::Float(v)));
            } else {
                let raw: String = chars[num_start..pos].iter().collect();
                let v: i64 = raw.parse().map_err(|_| GqlTranslateError::ParseError {
                    pos: start,
                    msg: format!("Invalid integer literal: {raw}"),
                })?;
                tokens.push((start, Token::Int(v)));
            }
            continue;
        }

        // Identifier / keyword.
        if c.is_alphabetic() || c == '_' {
            let id_start = pos;
            while pos < chars.len() && (chars[pos].is_alphanumeric() || chars[pos] == '_') {
                pos += 1;
            }
            let id: String = chars[id_start..pos].iter().collect();
            tokens.push((start, Token::Ident(id)));
            continue;
        }

        // Single-character punctuation.
        let tok = match c {
            '(' => Token::LParen,
            ')' => Token::RParen,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            '{' => Token::LBrace,
            '}' => Token::RBrace,
            ':' => Token::Colon,
            '.' => Token::Dot,
            ',' => Token::Comma,
            '=' => Token::Eq,
            '-' => Token::Minus,
            '>' => Token::Gt,
            '<' => Token::Lt,
            other => {
                return Err(GqlTranslateError::ParseError {
                    pos: start,
                    msg: format!("Unexpected character: {other:?}"),
                });
            }
        };
        pos += 1;
        tokens.push((start, tok));
    }

    Ok(tokens)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser state
// ─────────────────────────────────────────────────────────────────────────────

/// Recursive-descent parser that consumes a token stream.
pub(crate) struct Parser {
    tokens: Vec<(usize, Token)>,
    /// Index of the next token to be consumed.
    cursor: usize,
    /// Auto-increment counter for generating fresh anonymous variables.
    anon_counter: usize,
}

impl Parser {
    pub(crate) fn new(tokens: Vec<(usize, Token)>) -> Self {
        Self {
            tokens,
            cursor: 0,
            anon_counter: 0,
        }
    }

    // ── Low-level helpers ───────────────────────────────────────────────────

    /// Current byte position (used for error messages).
    fn current_pos(&self) -> usize {
        self.tokens
            .get(self.cursor)
            .map(|(p, _)| *p)
            .unwrap_or_else(|| {
                // End of token stream — use position of last token + 1.
                self.tokens.last().map(|(p, _)| p + 1).unwrap_or(0)
            })
    }

    /// Peek at the current token without consuming it.
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.cursor).map(|(_, t)| t)
    }

    /// Peek two tokens ahead (used for look-ahead disambiguation).
    fn peek2(&self) -> Option<&Token> {
        self.tokens.get(self.cursor + 1).map(|(_, t)| t)
    }

    /// Consume and return the current token.
    fn advance(&mut self) -> Option<&Token> {
        if self.cursor < self.tokens.len() {
            let tok = &self.tokens[self.cursor].1;
            self.cursor += 1;
            Some(tok)
        } else {
            None
        }
    }

    /// Expect a specific token variant; consume it or return a parse error.
    fn expect(&mut self, expected: &Token) -> Result<(), GqlTranslateError> {
        match self.peek() {
            Some(t) if std::mem::discriminant(t) == std::mem::discriminant(expected) => {
                self.advance();
                Ok(())
            }
            Some(t) => Err(GqlTranslateError::ParseError {
                pos: self.current_pos(),
                msg: format!("Expected {expected:?}, found {t:?}"),
            }),
            None => Err(GqlTranslateError::ParseError {
                pos: self.current_pos(),
                msg: format!("Expected {expected:?}, found end of input"),
            }),
        }
    }

    /// Expect an identifier keyword (case-insensitive) and consume it.
    fn expect_keyword(&mut self, kw: &str) -> Result<(), GqlTranslateError> {
        match self.peek() {
            Some(Token::Ident(id)) if id.to_uppercase() == kw.to_uppercase() => {
                self.advance();
                Ok(())
            }
            Some(t) => Err(GqlTranslateError::ParseError {
                pos: self.current_pos(),
                msg: format!("Expected keyword '{kw}', found {t:?}"),
            }),
            None => Err(GqlTranslateError::ParseError {
                pos: self.current_pos(),
                msg: format!("Expected keyword '{kw}', found end of input"),
            }),
        }
    }

    /// Check whether the upcoming token is an identifier matching `kw`
    /// (case-insensitive) without consuming it.
    fn peek_keyword(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Token::Ident(id)) if id.to_uppercase() == kw.to_uppercase())
    }

    /// Consume the next identifier token and return its text, or error.
    fn consume_ident(&mut self) -> Result<String, GqlTranslateError> {
        // Peek first to avoid borrow conflicts when constructing error messages.
        match self.peek() {
            Some(Token::Ident(_)) => {
                // SAFETY: we just confirmed it is Ident.
                if let Some(Token::Ident(id)) = self.advance() {
                    Ok(id.clone())
                } else {
                    unreachable!("peek confirmed Ident variant")
                }
            }
            Some(t) => {
                let msg = format!("Expected identifier, found {t:?}");
                let pos = self.current_pos();
                Err(GqlTranslateError::ParseError { pos, msg })
            }
            None => Err(GqlTranslateError::ParseError {
                pos: self.current_pos(),
                msg: "Expected identifier, found end of input".to_string(),
            }),
        }
    }

    /// Generate a fresh anonymous variable name like `_anon0`.
    fn fresh_anon(&mut self) -> String {
        let n = self.anon_counter;
        self.anon_counter += 1;
        format!("_anon{n}")
    }

    // ── Top-level parse ─────────────────────────────────────────────────────

    /// Parse a complete GQL query.
    pub(crate) fn parse_query(&mut self) -> Result<GqlQuery, GqlTranslateError> {
        self.expect_keyword("MATCH")?;

        let match_pattern = self.parse_graph_pattern()?;

        // Optional WHERE clause.
        let where_pred = if self.peek_keyword("WHERE") {
            self.advance(); // consume WHERE
            Some(self.parse_predicate()?)
        } else {
            None
        };

        self.expect_keyword("RETURN")?;
        let return_vars = self.parse_return_clause()?;

        if self.cursor < self.tokens.len() {
            // Extra tokens remaining — not necessarily an error for a subset
            // parser, but flag it to aid debugging.
            let pos = self.current_pos();
            return Err(GqlTranslateError::ParseError {
                pos,
                msg: format!("Unexpected token after RETURN clause: {:?}", self.peek()),
            });
        }

        Ok(GqlQuery {
            match_pattern,
            where_pred,
            return_vars,
        })
    }

    // ── Graph pattern ───────────────────────────────────────────────────────

    /// Parse `GraphPattern ::= NodePattern (EdgePattern NodePattern)*`.
    fn parse_graph_pattern(&mut self) -> Result<Vec<PathSegment>, GqlTranslateError> {
        let mut segments = Vec::new();
        segments.push(PathSegment::Node(self.parse_node_pattern()?));

        loop {
            // Decide whether the next token begins an edge pattern.
            // Forward edge starts with `-[`.
            // Backward edge starts with `<-[`.
            let is_forward_edge = matches!(self.peek(), Some(Token::Minus))
                && matches!(self.peek2(), Some(Token::LBracket));

            let is_backward_edge = matches!(self.peek(), Some(Token::Lt))
                && matches!(self.peek2(), Some(Token::Minus));

            if !is_forward_edge && !is_backward_edge {
                break;
            }

            segments.push(PathSegment::Edge(self.parse_edge_pattern()?));
            segments.push(PathSegment::Node(self.parse_node_pattern()?));
        }

        Ok(segments)
    }

    // ── Node pattern ────────────────────────────────────────────────────────

    /// Parse `NodePattern ::= '(' VarOpt LabelFilter? PropFilter? ')'`.
    fn parse_node_pattern(&mut self) -> Result<NodePattern, GqlTranslateError> {
        self.expect(&Token::LParen)?;

        // VarOpt: an identifier that is NOT a keyword and is not `_`.
        // `_` means anonymous.
        let var = match self.peek() {
            Some(Token::Ident(id)) if !self.is_structural_keyword(id) => {
                let id = id.clone();
                self.advance();
                if id == "_" {
                    // Anonymous — generate a fresh blank variable so translations
                    // still produce well-formed SPARQL.
                    Some(self.fresh_anon())
                } else {
                    Some(id)
                }
            }
            _ => None,
        };

        // LabelFilter: `:` Ident
        let label = if matches!(self.peek(), Some(Token::Colon)) {
            self.advance(); // consume `:`
            Some(self.consume_ident()?)
        } else {
            None
        };

        // PropFilter: `{` … `}`
        let props = if matches!(self.peek(), Some(Token::LBrace)) {
            self.parse_prop_filter()?
        } else {
            Vec::new()
        };

        self.expect(&Token::RParen)?;

        Ok(NodePattern { var, label, props })
    }

    // ── Edge pattern ────────────────────────────────────────────────────────

    /// Parse a directed edge pattern.
    ///
    /// ```text
    /// EdgePattern ::= '-[' VarOpt LabelFilter? ']->'
    ///               | '<-[' VarOpt LabelFilter? ']-'
    /// ```
    fn parse_edge_pattern(&mut self) -> Result<EdgePattern, GqlTranslateError> {
        let direction = if matches!(self.peek(), Some(Token::Minus)) {
            // Forward: `-[…]->`
            self.advance(); // consume `-`
            self.expect(&Token::LBracket)?;
            EdgeDirection::Forward
        } else {
            // Backward: `<-[…]-`
            self.expect(&Token::Lt)?;
            self.expect(&Token::Minus)?;
            self.expect(&Token::LBracket)?;
            EdgeDirection::Backward
        };

        // VarOpt
        let var = match self.peek() {
            Some(Token::Ident(id)) if !self.is_structural_keyword(id) => {
                let id = id.clone();
                self.advance();
                if id == "_" {
                    Some(self.fresh_anon())
                } else {
                    Some(id)
                }
            }
            _ => None,
        };

        // LabelFilter
        let label = if matches!(self.peek(), Some(Token::Colon)) {
            self.advance(); // consume `:`
            Some(self.consume_ident()?)
        } else {
            None
        };

        self.expect(&Token::RBracket)?;

        // Consume the closing arrow.
        match direction {
            EdgeDirection::Forward => {
                // Expect `->`
                self.expect(&Token::Minus)?;
                self.expect(&Token::Gt)?;
            }
            EdgeDirection::Backward => {
                // Expect `-`
                self.expect(&Token::Minus)?;
            }
        }

        Ok(EdgePattern {
            var,
            label,
            direction,
        })
    }

    // ── Property filter ─────────────────────────────────────────────────────

    /// Parse `PropFilter ::= '{' PropKV (',' PropKV)* '}'`.
    fn parse_prop_filter(&mut self) -> Result<Vec<(String, GqlLiteral)>, GqlTranslateError> {
        self.expect(&Token::LBrace)?;
        let mut kvs = Vec::new();

        if !matches!(self.peek(), Some(Token::RBrace)) {
            kvs.push(self.parse_prop_kv()?);
            while matches!(self.peek(), Some(Token::Comma)) {
                self.advance(); // consume `,`
                kvs.push(self.parse_prop_kv()?);
            }
        }

        self.expect(&Token::RBrace)?;
        Ok(kvs)
    }

    /// Parse `PropKV ::= Ident ':' Literal`.
    fn parse_prop_kv(&mut self) -> Result<(String, GqlLiteral), GqlTranslateError> {
        let key = self.consume_ident()?;
        self.expect(&Token::Colon)?;
        let val = self.parse_literal()?;
        Ok((key, val))
    }

    // ── WHERE predicate ─────────────────────────────────────────────────────

    /// Parse `Predicate ::= Ident '.' Ident '=' Literal`.
    fn parse_predicate(&mut self) -> Result<GqlPredicate, GqlTranslateError> {
        let var = self.consume_ident()?;
        self.expect(&Token::Dot)?;
        let prop = self.consume_ident()?;
        self.expect(&Token::Eq)?;
        let value = self.parse_literal()?;
        Ok(GqlPredicate { var, prop, value })
    }

    // ── RETURN clause ───────────────────────────────────────────────────────

    /// Parse `ReturnClause ::= Ident (',' Ident)*`.
    fn parse_return_clause(&mut self) -> Result<Vec<String>, GqlTranslateError> {
        let first = self.consume_ident()?;
        let mut vars = vec![first];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.advance(); // consume `,`
            vars.push(self.consume_ident()?);
        }
        Ok(vars)
    }

    // ── Literal ─────────────────────────────────────────────────────────────

    /// Parse `Literal ::= '"' … '"' | NUMBER | 'true' | 'false'`.
    ///
    /// Handles an optional unary minus before a numeric literal so that
    /// negative numbers like `-5` parse correctly inside property filters.
    fn parse_literal(&mut self) -> Result<GqlLiteral, GqlTranslateError> {
        // Check for unary minus before a number.
        let negate = if matches!(self.peek(), Some(Token::Minus)) {
            // Only consume the minus if the *next* token is a number.
            if matches!(self.peek2(), Some(Token::Int(_)) | Some(Token::Float(_))) {
                self.advance(); // consume `-`
                true
            } else {
                false
            }
        } else {
            false
        };

        // Peek to build error messages without borrow conflicts.
        match self.peek() {
            Some(Token::Str(_)) => {
                if let Some(Token::Str(s)) = self.advance() {
                    Ok(GqlLiteral::Str(s.clone()))
                } else {
                    unreachable!()
                }
            }
            Some(Token::Int(_)) => {
                if let Some(Token::Int(n)) = self.advance() {
                    let v = if negate { -n } else { *n };
                    Ok(GqlLiteral::Int(v))
                } else {
                    unreachable!()
                }
            }
            Some(Token::Float(_)) => {
                if let Some(Token::Float(f)) = self.advance() {
                    let v = if negate { -f } else { *f };
                    Ok(GqlLiteral::Float(v))
                } else {
                    unreachable!()
                }
            }
            Some(Token::Ident(id)) => {
                // Clone before mutably advancing.
                let id_lower = id.to_lowercase();
                let pos = self.current_pos();
                self.advance(); // consume
                match id_lower.as_str() {
                    "true" => Ok(GqlLiteral::Bool(true)),
                    "false" => Ok(GqlLiteral::Bool(false)),
                    other => Err(GqlTranslateError::ParseError {
                        pos,
                        msg: format!("Expected literal, found identifier '{other}'"),
                    }),
                }
            }
            Some(t) => {
                let msg = format!("Expected literal, found {t:?}");
                let pos = self.current_pos();
                Err(GqlTranslateError::ParseError { pos, msg })
            }
            None => Err(GqlTranslateError::ParseError {
                pos: self.current_pos(),
                msg: "Expected literal, found end of input".to_string(),
            }),
        }
    }

    // ── Helper ───────────────────────────────────────────────────────────────

    /// Returns `true` for identifiers that are structural GQL keywords and
    /// therefore cannot serve as variable names.
    fn is_structural_keyword(&self, id: &str) -> bool {
        matches!(
            id.to_uppercase().as_str(),
            "MATCH" | "WHERE" | "RETURN" | "TRUE" | "FALSE"
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a GQL query string into a [`GqlQuery`] AST.
pub fn parse_gql(src: &str) -> Result<GqlQuery, GqlTranslateError> {
    let tokens = tokenise(src)?;
    let mut parser = Parser::new(tokens);
    parser.parse_query()
}
