// Jena Rule Language lexer — hand-written, no external dependencies.
// Tokenizes .rules files following Apache Jena syntax.

use std::fmt;

/// All tokens produced by the JRL lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `->`  (forward arrow)
    Arrow,
    /// `<-`  (backward arrow)
    BackArrow,
    /// `:`
    Colon,
    /// `.`
    Dot,
    /// `@prefix`
    AtPrefix,
    /// `?name` — variable reference
    Variable(String),
    /// `<http://...>` — full IRI
    Iri(String),
    /// `prefix:local` — prefixed name
    PrefixedName(String, String),
    /// `"text"` — string literal
    StringLit(String),
    /// Integer literal
    IntLit(i64),
    /// Floating-point literal
    FloatLit(f64),
    /// Plain identifier (rule names, builtin names, bare terms)
    Ident(String),
    /// End of input
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::Arrow => write!(f, "->"),
            Token::BackArrow => write!(f, "<-"),
            Token::Colon => write!(f, ":"),
            Token::Dot => write!(f, "."),
            Token::AtPrefix => write!(f, "@prefix"),
            Token::Variable(s) => write!(f, "?{}", s),
            Token::Iri(s) => write!(f, "<{}>", s),
            Token::PrefixedName(p, l) => write!(f, "{}:{}", p, l),
            Token::StringLit(s) => write!(f, "\"{}\"", s),
            Token::IntLit(n) => write!(f, "{}", n),
            Token::FloatLit(n) => write!(f, "{}", n),
            Token::Ident(s) => write!(f, "{}", s),
            Token::Eof => write!(f, "<EOF>"),
        }
    }
}

/// Error produced by the lexer, with source position.
#[derive(Debug)]
pub struct LexError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LexError at {}:{}: {}",
            self.line, self.col, self.message
        )
    }
}

impl std::error::Error for LexError {}

/// Token together with its source position.
#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub line: usize,
    pub col: usize,
}

/// Hand-written lexer for Jena Rule Language.
pub struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.input.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.input.get(self.pos).copied()?;
        self.pos += 1;
        if ch == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    /// Skip whitespace (spaces, tabs, newlines, carriage returns).
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == b' ' || ch == b'\t' || ch == b'\n' || ch == b'\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Skip a line comment starting with `#`.
    /// Guard: caller must have already seen `#`.
    fn skip_line_comment(&mut self) {
        while let Some(ch) = self.advance() {
            if ch == b'\n' {
                break;
            }
        }
    }

    /// Skip whitespace and `#` comments.
    fn skip_trivia(&mut self) {
        loop {
            self.skip_whitespace();
            if self.peek() == Some(b'#') {
                self.advance(); // consume `#`
                self.skip_line_comment();
            } else {
                break;
            }
        }
    }

    /// Lex a string literal starting after the opening `"`.
    fn lex_string(&mut self) -> Result<Token, LexError> {
        let start_line = self.line;
        let start_col = self.col;
        let mut buf = String::new();
        loop {
            match self.advance() {
                None => {
                    return Err(LexError {
                        message: "Unterminated string literal".to_string(),
                        line: start_line,
                        col: start_col,
                    });
                }
                Some(b'"') => break,
                Some(b'\\') => match self.advance() {
                    Some(b'"') => buf.push('"'),
                    Some(b'\\') => buf.push('\\'),
                    Some(b'n') => buf.push('\n'),
                    Some(b't') => buf.push('\t'),
                    Some(b'r') => buf.push('\r'),
                    Some(other) => {
                        buf.push('\\');
                        buf.push(other as char);
                    }
                    None => {
                        return Err(LexError {
                            message: "Unterminated escape sequence".to_string(),
                            line: self.line,
                            col: self.col,
                        });
                    }
                },
                Some(ch) => buf.push(ch as char),
            }
        }
        Ok(Token::StringLit(buf))
    }

    /// Lex a full IRI `<...>` starting after the opening `<`.
    fn lex_iri(&mut self) -> Result<Token, LexError> {
        let start_line = self.line;
        let start_col = self.col;
        let mut buf = String::new();
        loop {
            match self.advance() {
                None => {
                    return Err(LexError {
                        message: "Unterminated IRI".to_string(),
                        line: start_line,
                        col: start_col,
                    });
                }
                Some(b'>') => break,
                Some(ch) => buf.push(ch as char),
            }
        }
        Ok(Token::Iri(buf))
    }

    /// Lex a variable `?name` starting after `?`.
    fn lex_variable(&mut self) -> Token {
        let mut name = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == b'_' {
                self.advance();
                name.push(ch as char);
            } else {
                break;
            }
        }
        Token::Variable(name)
    }

    /// Lex an identifier or number starting at the current byte.
    /// Returns `@prefix`, `Ident`, `PrefixedName`, `IntLit`, or `FloatLit`.
    fn lex_ident_or_number(&mut self, first: u8) -> Result<Token, LexError> {
        let line = self.line;
        let col = self.col;
        let mut buf = String::new();
        buf.push(first as char);

        // Accumulate alphanumeric + `-`, `_`, `.` (may be part of a prefixed name local part)
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == b'_' || ch == b'-' || ch == b'.' {
                self.advance();
                buf.push(ch as char);
            } else {
                break;
            }
        }

        // @prefix keyword
        if buf == "@prefix" {
            return Ok(Token::AtPrefix);
        }

        // If it starts with `@` it's an unknown at-keyword (treat as ident)
        if buf.starts_with('@') {
            return Ok(Token::Ident(buf));
        }

        // Check for prefixed name: peek for `:` that isn't `:-` or `::`
        // A prefixed name is  `prefix:local`; prefix is already in buf.
        if self.peek() == Some(b':') {
            // Peek past `:` to see if it's a local name character or empty local
            let next2 = self.peek2();
            // `:` followed by alphanumeric/_ starts a prefixed name.
            // `:` followed by `-` could be Drools/other; treat as separator if alone.
            if next2
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'\'' || c == b'.')
            {
                self.advance(); // consume `:`
                let prefix = buf.clone();
                let mut local = String::new();
                while let Some(ch) = self.peek() {
                    if ch.is_ascii_alphanumeric()
                        || ch == b'_'
                        || ch == b'-'
                        || ch == b'.'
                        || ch == b'\''
                    {
                        self.advance();
                        local.push(ch as char);
                    } else {
                        break;
                    }
                }
                return Ok(Token::PrefixedName(prefix, local));
            } else if next2.map_or(true, |c| {
                // `:` by itself (or followed by whitespace/bracket): treat as Ident + upcoming Colon
                c == b' ' || c == b'\t' || c == b'\n' || c == b']' || c == b')'
            }) {
                // Don't consume `:` — emit just the ident; the colon will be next token
                return Ok(Token::Ident(buf));
            }
        }

        // Attempt numeric parsing
        if first.is_ascii_digit() || first == b'-' || first == b'+' {
            // Remove trailing `.` that was speculatively consumed (can appear in decimals)
            // We already included them; try parse
            if let Ok(n) = buf.parse::<i64>() {
                return Ok(Token::IntLit(n));
            }
            if let Ok(f) = buf.parse::<f64>() {
                return Ok(Token::FloatLit(f));
            }
            // Not numeric after all — fall through to Ident
        }

        // Handle special `@prefix` where `@` was the first char
        if first == b'@' && buf == "@prefix" {
            return Ok(Token::AtPrefix);
        }

        // Otherwise plain identifier
        if buf.is_empty() {
            return Err(LexError {
                message: format!("Unexpected character: {:?}", first as char),
                line,
                col,
            });
        }

        Ok(Token::Ident(buf))
    }

    /// Lex the next token (skips trivia automatically).
    pub fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_trivia();
        let line = self.line;
        let col = self.col;

        let ch = match self.advance() {
            None => return Ok(Token::Eof),
            Some(c) => c,
        };

        match ch {
            b'[' => Ok(Token::LBracket),
            b']' => Ok(Token::RBracket),
            b'(' => Ok(Token::LParen),
            b')' => Ok(Token::RParen),
            b'.' => Ok(Token::Dot),
            b':' => Ok(Token::Colon),
            b'"' => self.lex_string(),
            b'?' => Ok(self.lex_variable()),
            b'<' => {
                // Could be `<-` (backward arrow) or IRI `<http://...>`
                if self.peek() == Some(b'-') {
                    self.advance(); // consume `-`
                    Ok(Token::BackArrow)
                } else {
                    self.lex_iri()
                }
            }
            b'-' => {
                // `->` arrow or negative number or just `-`
                if self.peek() == Some(b'>') {
                    self.advance(); // consume `>`
                    Ok(Token::Arrow)
                } else if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.lex_ident_or_number(ch)
                } else {
                    Ok(Token::Ident("-".to_string()))
                }
            }
            b'@' => {
                // @prefix
                let mut buf = String::from("@");
                while let Some(c) = self.peek() {
                    if c.is_ascii_alphabetic() {
                        self.advance();
                        buf.push(c as char);
                    } else {
                        break;
                    }
                }
                if buf == "@prefix" {
                    Ok(Token::AtPrefix)
                } else {
                    Ok(Token::Ident(buf))
                }
            }
            other => {
                if other.is_ascii_alphanumeric() || other == b'_' || other == b'+' {
                    self.lex_ident_or_number(other)
                } else {
                    Err(LexError {
                        message: format!("Unexpected character: {:?}", other as char),
                        line,
                        col,
                    })
                }
            }
        }
    }

    /// Tokenize the entire input into a flat vec (excluding Eof).
    pub fn tokenize(input: &'a str) -> Result<Vec<SpannedToken>, LexError> {
        let mut lexer = Self::new(input);
        let mut tokens = Vec::new();
        loop {
            let line = lexer.line;
            let col = lexer.col;
            let tok = lexer.next_token()?;
            if tok == Token::Eof {
                break;
            }
            tokens.push(SpannedToken {
                token: tok,
                line,
                col,
            });
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(input: &str) -> Vec<Token> {
        Lexer::tokenize(input)
            .unwrap()
            .into_iter()
            .map(|st| st.token)
            .collect()
    }

    #[test]
    fn test_tokenize_variable() {
        let toks = tokens("?x ?foo_bar");
        assert_eq!(toks[0], Token::Variable("x".to_string()));
        assert_eq!(toks[1], Token::Variable("foo_bar".to_string()));
    }

    #[test]
    fn test_tokenize_iri() {
        let toks = tokens("<http://example.org/Foo>");
        assert_eq!(toks[0], Token::Iri("http://example.org/Foo".to_string()));
    }

    #[test]
    fn test_tokenize_prefixed_name() {
        let toks = tokens("rdf:type");
        assert_eq!(
            toks[0],
            Token::PrefixedName("rdf".to_string(), "type".to_string())
        );
    }

    #[test]
    fn test_tokenize_arrow() {
        let toks = tokens("->");
        assert_eq!(toks[0], Token::Arrow);
    }

    #[test]
    fn test_tokenize_back_arrow() {
        let toks = tokens("<-");
        assert_eq!(toks[0], Token::BackArrow);
    }

    #[test]
    fn test_tokenize_brackets() {
        let toks = tokens("[ ]");
        assert_eq!(toks[0], Token::LBracket);
        assert_eq!(toks[1], Token::RBracket);
    }

    #[test]
    fn test_tokenize_comment_ignored() {
        // Comments beginning with # should produce no tokens
        let toks = tokens("# this is a comment\n?x");
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0], Token::Variable("x".to_string()));
    }

    #[test]
    fn test_tokenize_string_literal() {
        let toks = tokens(r#""hello world""#);
        assert_eq!(toks[0], Token::StringLit("hello world".to_string()));
    }

    #[test]
    fn test_tokenize_int_literal() {
        let toks = tokens("42");
        assert_eq!(toks[0], Token::IntLit(42));
    }

    #[test]
    fn test_tokenize_at_prefix() {
        let toks = tokens("@prefix");
        assert_eq!(toks[0], Token::AtPrefix);
    }

    #[test]
    fn test_tokenize_colon() {
        // bare `:` after a space (not part of a prefixed name)
        let toks = tokens("[rule1 : (?x p ?y)]");
        assert!(toks.contains(&Token::Colon));
    }

    #[test]
    fn test_tokenize_multiline_no_crash() {
        let input = "# comment\n[rule1: (?a ex:p ?b) -> (?a ex:q ?b)]\n";
        let result = Lexer::tokenize(input);
        assert!(result.is_ok());
    }
}
