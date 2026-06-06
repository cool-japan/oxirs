//! TriG lexer — tokenizer for the TriG format.
//!
//! Handles all terminal symbols defined in the TriG grammar:
//! - IRI references: `<...>`
//! - Prefixed names: `prefix:local`
//! - Blank node labels: `_:label`
//! - Anonymous blank nodes: `[]`
//! - String literals (single, double, long, with lang/datatype)
//! - `@prefix` / `@base` / `PREFIX` / `BASE` directives
//! - `{` `}` `.` `;` `,`
//! - `a` (rdf:type shorthand)
//! - Numeric literals (integer, decimal, double)
//! - Boolean literals (`true`, `false`)

use std::io::BufRead;

use crate::trig_streaming::TriGParseError;

// ============================================================================
// Token type
// ============================================================================

/// A single lexical token produced by the TriG lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum TriGToken {
    /// `<iri>` — an absolute or relative IRI reference.
    IriRef(String),
    /// `prefix:local` — a CURIE with the given prefix and local name.
    PrefixedName(String, String),
    /// `_:label` — a named blank node.
    BlankNodeLabel(String),
    /// `[]` — an anonymous blank node.
    AnonBlankNode,
    /// A string literal, with optional language tag or datatype IRI.
    StringLiteral {
        /// The unescaped string value.
        value: String,
        /// Optional BCP 47 language tag.
        lang: Option<String>,
        /// Optional datatype IRI (angle brackets stripped).
        datatype: Option<String>,
    },
    /// `@prefix` directive keyword (label already parsed).
    Prefix(String),
    /// `@base` directive keyword (IRI already parsed).
    Base(String),
    /// `{` — open graph block.
    LBrace,
    /// `}` — close graph block.
    RBrace,
    /// `.` — statement terminator.
    Dot,
    /// `;` — predicate-object list separator.
    Semicolon,
    /// `,` — object list separator.
    Comma,
    /// `a` — shorthand for `rdf:type`.
    A,
    /// `true` literal.
    True,
    /// `false` literal.
    False,
    /// Integer literal.
    Integer(i64),
    /// Decimal literal.
    Decimal(f64),
    /// Double (floating-point) literal.
    Double(f64),
    /// `[` — open blank node property list.
    LBracket,
    /// `]` — close blank node property list.
    RBracket,
}

// ============================================================================
// Lexer
// ============================================================================

/// TriG tokenizer with single-token lookahead.
pub struct TriGLexer<R: BufRead> {
    reader: R,
    /// Current line number (1-based).
    line: usize,
    /// Buffer holding the current line.
    buf: String,
    /// Current byte offset within `buf`.
    pos: usize,
    /// One-token lookahead.
    peeked: Option<TriGToken>,
}

impl<R: BufRead> TriGLexer<R> {
    /// Create a new lexer wrapping the given buffered reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line: 0,
            buf: String::new(),
            pos: 0,
            peeked: None,
        }
    }

    /// Return the current 1-based line number.
    pub fn line(&self) -> usize {
        self.line
    }

    /// Peek at the next token without consuming it.
    pub fn peek(&mut self) -> Result<Option<&TriGToken>, TriGParseError> {
        if self.peeked.is_none() {
            self.peeked = self.read_next_token()?;
        }
        Ok(self.peeked.as_ref())
    }

    /// Consume and return the next token.
    pub fn next_token(&mut self) -> Result<Option<TriGToken>, TriGParseError> {
        if let Some(tok) = self.peeked.take() {
            return Ok(Some(tok));
        }
        self.read_next_token()
    }

    // -----------------------------------------------------------------------
    // Internal tokenization — no `loop` around the outer match so clippy is happy
    // -----------------------------------------------------------------------

    fn read_next_token(&mut self) -> Result<Option<TriGToken>, TriGParseError> {
        // Skip whitespace and comments, filling the buffer from the reader.
        self.skip_whitespace_and_comments()?;

        // EOF?
        if self.current_char().is_none() {
            return Ok(None);
        }

        let ch = match self.current_char() {
            Some(c) => c,
            None => return Ok(None),
        };

        match ch {
            // IRI reference
            '<' => self.read_iri_ref().map(Some),

            // Blank node label.
            '_' if self.peek_next_char() == Some(':') => self.read_blank_node_label().map(Some),

            // Anonymous blank node `[]` or blank node property list `[`.
            '[' => {
                self.advance();
                // Check immediately for ] (anonymous blank node).
                self.skip_inline_whitespace();
                if self.current_char() == Some(']') {
                    self.advance();
                    Ok(Some(TriGToken::AnonBlankNode))
                } else {
                    Ok(Some(TriGToken::LBracket))
                }
            }

            ']' => {
                self.advance();
                Ok(Some(TriGToken::RBracket))
            }

            // String literal.
            '"' | '\'' => self.read_string_literal().map(Some),

            // Graph block delimiters.
            '{' => {
                self.advance();
                Ok(Some(TriGToken::LBrace))
            }
            '}' => {
                self.advance();
                Ok(Some(TriGToken::RBrace))
            }

            // Statement terminator.
            '.' => {
                self.advance();
                Ok(Some(TriGToken::Dot))
            }

            // Separators.
            ';' => {
                self.advance();
                Ok(Some(TriGToken::Semicolon))
            }
            ',' => {
                self.advance();
                Ok(Some(TriGToken::Comma))
            }

            // @-keywords.
            '@' => self.read_at_keyword().map(Some),

            // Numeric literals (also handles +/- sign).
            c if c.is_ascii_digit() || c == '+' || c == '-' => self.read_numeric().map(Some),

            // Named node prefix or keyword.
            c if c.is_alphabetic() || c == '_' => self.read_name_or_keyword().map(Some),

            other => {
                let line = self.line;
                Err(TriGParseError::InvalidToken {
                    line,
                    message: format!("Unexpected character: {:?}", other),
                })
            }
        }
    }

    // -----------------------------------------------------------------------
    // Whitespace / comment skipping
    // -----------------------------------------------------------------------

    fn skip_whitespace_and_comments(&mut self) -> Result<(), TriGParseError> {
        'outer: loop {
            // If buf is exhausted, read next line.
            while self.pos >= self.buf.len() {
                self.buf.clear();
                self.pos = 0;
                let n = self.reader.read_line(&mut self.buf)?;
                if n == 0 {
                    return Ok(()); // EOF
                }
                self.line += 1;
            }

            if let Some(ch) = self.current_char() {
                if ch.is_whitespace() {
                    self.advance();
                    continue 'outer;
                }
                if ch == '#' {
                    // Consume rest of line.
                    while self.current_char().is_some_and(|c| c != '\n') {
                        self.advance();
                    }
                    continue 'outer;
                }
            }
            break;
        }
        Ok(())
    }

    fn skip_inline_whitespace(&mut self) {
        while matches!(self.current_char(), Some(' ') | Some('\t')) {
            self.advance();
        }
    }

    // -----------------------------------------------------------------------
    // Token readers
    // -----------------------------------------------------------------------

    fn read_iri_ref(&mut self) -> Result<TriGToken, TriGParseError> {
        // Consume '<'.
        self.advance();
        let mut iri = String::new();
        loop {
            self.ensure_line()?;
            match self.current_char() {
                None => {
                    return Err(TriGParseError::InvalidToken {
                        line: self.line,
                        message: "Unterminated IRI reference".to_string(),
                    });
                }
                Some('>') => {
                    self.advance();
                    return Ok(TriGToken::IriRef(iri));
                }
                Some('\\') => {
                    self.advance();
                    let escaped = self.read_escape()?;
                    iri.push(escaped);
                }
                Some(ch) => {
                    iri.push(ch);
                    self.advance();
                }
            }
        }
    }

    fn read_blank_node_label(&mut self) -> Result<TriGToken, TriGParseError> {
        // Consume '_:'.
        self.advance(); // '_'
        self.advance(); // ':'
        let mut label = String::new();
        loop {
            match self.current_char() {
                Some(c) if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' => {
                    label.push(c);
                    self.advance();
                }
                _ => break,
            }
        }
        // Trim trailing '.'.
        while label.ends_with('.') {
            label.pop();
        }
        if label.is_empty() {
            return Err(TriGParseError::InvalidToken {
                line: self.line,
                message: "Empty blank node label".to_string(),
            });
        }
        Ok(TriGToken::BlankNodeLabel(label))
    }

    fn read_string_literal(&mut self) -> Result<TriGToken, TriGParseError> {
        let quote_char = self.current_char().unwrap_or('"');
        self.advance(); // consume first quote

        // Check for long string (triple-quoted).
        if self.current_char() == Some(quote_char) {
            self.advance();
            if self.current_char() == Some(quote_char) {
                self.advance(); // third quote — long string
                let value = self.read_long_string(quote_char)?;
                return self.read_literal_suffix(value);
            }
            // Two quotes consumed — empty short string followed by another token.
            // The second quote closed the string; return empty literal.
            return self.read_literal_suffix(String::new());
        }

        let value = self.read_short_string(quote_char)?;
        self.read_literal_suffix(value)
    }

    fn read_short_string(&mut self, quote: char) -> Result<String, TriGParseError> {
        let mut s = String::new();
        loop {
            self.ensure_line()?;
            match self.current_char() {
                None => {
                    return Err(TriGParseError::InvalidToken {
                        line: self.line,
                        message: "Unterminated string literal".to_string(),
                    });
                }
                Some(c) if c == quote => {
                    self.advance();
                    return Ok(s);
                }
                Some('\n') | Some('\r') => {
                    return Err(TriGParseError::InvalidToken {
                        line: self.line,
                        message: "Newline in short string literal".to_string(),
                    });
                }
                Some('\\') => {
                    self.advance();
                    let ch = self.read_escape()?;
                    s.push(ch);
                }
                Some(c) => {
                    s.push(c);
                    self.advance();
                }
            }
        }
    }

    fn read_long_string(&mut self, quote: char) -> Result<String, TriGParseError> {
        let mut s = String::new();
        loop {
            // Long strings can span lines.
            if self.pos >= self.buf.len() {
                self.buf.clear();
                self.pos = 0;
                let n = self.reader.read_line(&mut self.buf)?;
                if n == 0 {
                    return Err(TriGParseError::InvalidToken {
                        line: self.line,
                        message: "Unterminated long string literal".to_string(),
                    });
                }
                self.line += 1;
            }
            match self.current_char() {
                None => {
                    return Err(TriGParseError::InvalidToken {
                        line: self.line,
                        message: "Unterminated long string literal".to_string(),
                    });
                }
                Some(c) if c == quote => {
                    self.advance();
                    // Check for closing triple quote.
                    if self.current_char() == Some(quote) {
                        self.advance();
                        if self.current_char() == Some(quote) {
                            self.advance();
                            return Ok(s);
                        }
                        s.push(quote);
                    }
                    s.push(quote);
                }
                Some('\\') => {
                    self.advance();
                    let ch = self.read_escape()?;
                    s.push(ch);
                }
                Some(c) => {
                    s.push(c);
                    self.advance();
                }
            }
        }
    }

    fn read_literal_suffix(&mut self, value: String) -> Result<TriGToken, TriGParseError> {
        match self.current_char() {
            Some('@') => {
                self.advance();
                let mut lang = String::new();
                loop {
                    match self.current_char() {
                        Some(c) if c.is_alphanumeric() || c == '-' => {
                            lang.push(c);
                            self.advance();
                        }
                        _ => break,
                    }
                }
                Ok(TriGToken::StringLiteral {
                    value,
                    lang: Some(lang),
                    datatype: None,
                })
            }
            Some('^') => {
                self.advance(); // first ^
                if self.current_char() != Some('^') {
                    return Err(TriGParseError::InvalidToken {
                        line: self.line,
                        message: "Expected '^^' for datatype annotation".to_string(),
                    });
                }
                self.advance(); // second ^
                // Read datatype IRI.
                let dt = if self.current_char() == Some('<') {
                    match self.read_iri_ref()? {
                        TriGToken::IriRef(iri) => iri,
                        _ => unreachable!(),
                    }
                } else {
                    // Could be a prefixed name datatype.
                    match self.read_name_or_keyword()? {
                        TriGToken::PrefixedName(p, l) => format!("{}:{}", p, l),
                        other => {
                            return Err(TriGParseError::InvalidToken {
                                line: self.line,
                                message: format!("Expected IRI for datatype, got {:?}", other),
                            });
                        }
                    }
                };
                Ok(TriGToken::StringLiteral {
                    value,
                    lang: None,
                    datatype: Some(dt),
                })
            }
            _ => Ok(TriGToken::StringLiteral {
                value,
                lang: None,
                datatype: None,
            }),
        }
    }

    fn read_at_keyword(&mut self) -> Result<TriGToken, TriGParseError> {
        self.advance(); // consume '@'
        let mut kw = String::new();
        while let Some(c) = self.current_char() {
            if !c.is_alphabetic() {
                break;
            }
            kw.push(c);
            self.advance();
        }
        match kw.as_str() {
            "prefix" => {
                // Read prefix label.
                self.skip_whitespace_and_comments()?;
                let mut label = String::new();
                while let Some(c) = self.current_char() {
                    if !(c.is_alphanumeric() || c == '_' || c == '-') {
                        break;
                    }
                    label.push(c);
                    self.advance();
                }
                if self.current_char() == Some(':') {
                    self.advance(); // consume ':'
                }
                Ok(TriGToken::Prefix(label))
            }
            "base" => {
                // Read base IRI.
                self.skip_whitespace_and_comments()?;
                match self.read_iri_ref()? {
                    TriGToken::IriRef(iri) => Ok(TriGToken::Base(iri)),
                    _ => Err(TriGParseError::InvalidToken {
                        line: self.line,
                        message: "@base requires an IRI reference".to_string(),
                    }),
                }
            }
            other => Err(TriGParseError::InvalidToken {
                line: self.line,
                message: format!("Unknown @-keyword: @{}", other),
            }),
        }
    }

    fn read_numeric(&mut self) -> Result<TriGToken, TriGParseError> {
        let mut num = String::new();
        // Optional sign.
        if let Some(c @ ('+' | '-')) = self.current_char() {
            num.push(c);
            self.advance();
        }
        // Integer part.
        while let Some(c) = self.current_char() {
            if !c.is_ascii_digit() {
                break;
            }
            num.push(c);
            self.advance();
        }
        // Decimal or double.
        let mut is_decimal = false;
        let mut is_double = false;
        if self.current_char() == Some('.') {
            is_decimal = true;
            num.push('.');
            self.advance();
            while let Some(c) = self.current_char() {
                if !c.is_ascii_digit() {
                    break;
                }
                num.push(c);
                self.advance();
            }
        }
        if let Some(c @ ('e' | 'E')) = self.current_char() {
            is_double = true;
            num.push(c);
            self.advance();
            if let Some(c @ ('+' | '-')) = self.current_char() {
                num.push(c);
                self.advance();
            }
            while let Some(c) = self.current_char() {
                if !c.is_ascii_digit() {
                    break;
                }
                num.push(c);
                self.advance();
            }
        }

        if is_double || (is_decimal && num.contains('e')) {
            let f: f64 = num.parse().map_err(|_| TriGParseError::InvalidToken {
                line: self.line,
                message: format!("Invalid double literal: {}", num),
            })?;
            Ok(TriGToken::Double(f))
        } else if is_decimal {
            let f: f64 = num.parse().map_err(|_| TriGParseError::InvalidToken {
                line: self.line,
                message: format!("Invalid decimal literal: {}", num),
            })?;
            Ok(TriGToken::Decimal(f))
        } else {
            let i: i64 = num.parse().map_err(|_| TriGParseError::InvalidToken {
                line: self.line,
                message: format!("Invalid integer literal: {}", num),
            })?;
            Ok(TriGToken::Integer(i))
        }
    }

    fn read_name_or_keyword(&mut self) -> Result<TriGToken, TriGParseError> {
        let mut name = String::new();
        while let Some(c) = self.current_char() {
            if !(c.is_alphanumeric() || c == '_' || c == '-' || c == '.') {
                break;
            }
            name.push(c);
            self.advance();
        }
        // Trim trailing dots (they may be statement terminators).
        while name.ends_with('.') {
            name.pop();
            self.pos -= 1; // put '.' back
        }

        // Handle SPARQL-style PREFIX / BASE (case-insensitive).
        if name.eq_ignore_ascii_case("prefix") {
            self.skip_whitespace_and_comments()?;
            let mut label = String::new();
            while let Some(c) = self.current_char() {
                if !(c.is_alphanumeric() || c == '_' || c == '-') {
                    break;
                }
                label.push(c);
                self.advance();
            }
            if self.current_char() == Some(':') {
                self.advance();
            }
            return Ok(TriGToken::Prefix(label));
        }
        if name.eq_ignore_ascii_case("base") {
            self.skip_whitespace_and_comments()?;
            return match self.read_iri_ref()? {
                TriGToken::IriRef(iri) => Ok(TriGToken::Base(iri)),
                _ => Err(TriGParseError::InvalidToken {
                    line: self.line,
                    message: "BASE requires an IRI reference".to_string(),
                }),
            };
        }

        // Keyword: a, true, false.
        if name == "a" && !matches!(self.current_char(), Some(':')) {
            return Ok(TriGToken::A);
        }
        if name == "true" {
            return Ok(TriGToken::True);
        }
        if name == "false" {
            return Ok(TriGToken::False);
        }

        // Prefixed name: name:local.
        if self.current_char() == Some(':') {
            self.advance(); // consume ':'
            let mut local = String::new();
            while let Some(c) = self.current_char() {
                if !(c.is_alphanumeric()
                    || c == '_'
                    || c == '-'
                    || c == '.'
                    || c == '/'
                    || c == '#')
                {
                    break;
                }
                local.push(c);
                self.advance();
            }
            // Trim trailing dots.
            while local.ends_with('.') {
                local.pop();
                self.pos -= 1;
            }
            return Ok(TriGToken::PrefixedName(name, local));
        }

        if name.is_empty() {
            return Err(TriGParseError::InvalidToken {
                line: self.line,
                message: "Empty name token".to_string(),
            });
        }

        // Bare name with no colon — treat as prefixed name with empty prefix.
        Ok(TriGToken::PrefixedName(String::new(), name))
    }

    fn read_escape(&mut self) -> Result<char, TriGParseError> {
        match self.current_char() {
            Some('n') => { self.advance(); Ok('\n') }
            Some('t') => { self.advance(); Ok('\t') }
            Some('r') => { self.advance(); Ok('\r') }
            Some('"') => { self.advance(); Ok('"') }
            Some('\'') => { self.advance(); Ok('\'') }
            Some('\\') => { self.advance(); Ok('\\') }
            Some('/') => { self.advance(); Ok('/') }
            Some('u') => {
                self.advance();
                self.read_unicode_escape(4)
            }
            Some('U') => {
                self.advance();
                self.read_unicode_escape(8)
            }
            Some(other) => Err(TriGParseError::InvalidToken {
                line: self.line,
                message: format!("Invalid escape sequence: \\{}", other),
            }),
            None => Err(TriGParseError::InvalidToken {
                line: self.line,
                message: "Unterminated escape sequence".to_string(),
            }),
        }
    }

    fn read_unicode_escape(&mut self, digits: usize) -> Result<char, TriGParseError> {
        let mut hex = String::new();
        for _ in 0..digits {
            match self.current_char() {
                Some(c) if c.is_ascii_hexdigit() => {
                    hex.push(c);
                    self.advance();
                }
                other => {
                    return Err(TriGParseError::InvalidToken {
                        line: self.line,
                        message: format!("Expected hex digit in Unicode escape, got {:?}", other),
                    });
                }
            }
        }
        let code_point = u32::from_str_radix(&hex, 16).map_err(|_| TriGParseError::InvalidToken {
            line: self.line,
            message: format!("Invalid Unicode code point: {}", hex),
        })?;
        char::from_u32(code_point).ok_or_else(|| TriGParseError::InvalidToken {
            line: self.line,
            message: format!("Invalid Unicode scalar value: U+{:04X}", code_point),
        })
    }

    // -----------------------------------------------------------------------
    // Buffer navigation helpers
    // -----------------------------------------------------------------------

    /// Return the character at the current position, or `None` at EOF.
    fn current_char(&self) -> Option<char> {
        self.buf[self.pos..].chars().next()
    }

    /// Return the character after the current one.
    fn peek_next_char(&self) -> Option<char> {
        let mut chars = self.buf[self.pos..].chars();
        chars.next(); // skip current
        chars.next()
    }

    /// Advance past the current character.
    fn advance(&mut self) {
        if self.pos < self.buf.len() {
            let ch_len = self.buf[self.pos..].chars().next().map_or(0, |c| c.len_utf8());
            self.pos += ch_len;
        }
    }

    /// Ensure the current line buffer has content (reads next line if exhausted).
    fn ensure_line(&mut self) -> Result<(), TriGParseError> {
        while self.pos >= self.buf.len() {
            self.buf.clear();
            self.pos = 0;
            let n = self.reader.read_line(&mut self.buf)?;
            if n == 0 {
                return Ok(()); // EOF is valid mid-token only for long strings
            }
            self.line += 1;
        }
        Ok(())
    }
}
