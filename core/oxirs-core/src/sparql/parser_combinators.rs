//! Parser combinators for SPARQL token stream processing
//!
//! Provides a simple, composable parser combinator library for processing
//! SPARQL token streams. Supports keywords, IRIs, variables, literals,
//! punctuation, optional matches, zero-or-more repetitions, and ordered choice.

/// The kind of a SPARQL token
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    /// SPARQL keyword (SELECT, WHERE, PREFIX, etc.)
    Keyword,
    /// Full IRI enclosed in angle brackets: <http://…>
    Iri,
    /// Prefixed name: prefix:local
    PrefixedName,
    /// Variable: ?name or $name
    Variable,
    /// Literal: "text", 42, 3.14, true
    Literal,
    /// Punctuation: { } ( ) , ; . etc.
    Punctuation,
    /// Whitespace (spaces/tabs/newlines)
    Whitespace,
    /// Comment (# …)
    Comment,
    /// End of token stream
    Eof,
}

/// A single SPARQL token
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The kind of this token
    pub kind: TokenKind,
    /// The raw text value of the token
    pub value: String,
    /// Byte offset of the token's first character in the source
    pub position: usize,
}

impl Token {
    /// Create a new token
    pub fn new(kind: TokenKind, value: impl Into<String>, position: usize) -> Self {
        Token {
            kind,
            value: value.into(),
            position,
        }
    }
}

/// An immutable, cloneable view of a token stream with a cursor position
#[derive(Debug, Clone)]
pub struct TokenStream {
    tokens: Vec<Token>,
    pos: usize,
}

/// Result type for parser combinators
pub type ParseResult<T> = Result<(T, TokenStream), ParseError>;

/// Error produced when a combinator fails
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Human-readable description of the failure
    pub message: String,
    /// Byte position in the source where the error occurred
    pub position: usize,
}

impl ParseError {
    /// Construct a new parse error
    pub fn new(message: impl Into<String>, position: usize) -> Self {
        ParseError {
            message: message.into(),
            position,
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at {}: {}", self.position, self.message)
    }
}

impl std::error::Error for ParseError {}

// ─── TokenStream ─────────────────────────────────────────────────────────────

impl TokenStream {
    /// Create a new token stream from a vector of tokens
    pub fn new(tokens: Vec<Token>) -> Self {
        TokenStream { tokens, pos: 0 }
    }

    /// Peek at the current token without advancing
    pub fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    /// Consume the current token, returning it and the advanced stream
    pub fn next(mut self) -> (Option<Token>, TokenStream) {
        if self.pos < self.tokens.len() {
            let tok = self.tokens[self.pos].clone();
            self.pos += 1;
            (Some(tok), self)
        } else {
            (None, self)
        }
    }

    /// Return true if no tokens remain (excluding Eof sentinel)
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Number of tokens remaining (excluding any trailing Eof token)
    pub fn remaining(&self) -> usize {
        let total = self.tokens.len();
        if total == 0 {
            return 0;
        }
        // Count non-EOF tokens from current pos
        let remaining_tokens = &self.tokens[self.pos..];
        remaining_tokens
            .iter()
            .filter(|t| t.kind != TokenKind::Eof)
            .count()
    }

    /// Current cursor position (index into token slice)
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Return the byte offset of the current token (or end-of-source)
    pub fn byte_offset(&self) -> usize {
        self.tokens.get(self.pos).map(|t| t.position).unwrap_or(0)
    }
}

// ─── Combinator functions ─────────────────────────────────────────────────────

/// Expect the next token to be a keyword matching `keyword` (case-insensitive)
pub fn expect_keyword(stream: TokenStream, keyword: &str) -> ParseResult<()> {
    match stream.peek() {
        Some(tok) if tok.kind == TokenKind::Keyword && tok.value.eq_ignore_ascii_case(keyword) => {
            let (_, rest) = stream.next();
            Ok(((), rest))
        }
        Some(tok) => Err(ParseError::new(
            format!(
                "expected keyword '{}', found {:?} '{}'",
                keyword, tok.kind, tok.value
            ),
            tok.position,
        )),
        None => Err(ParseError::new(
            format!("expected keyword '{}', reached end of stream", keyword),
            0,
        )),
    }
}

/// Expect the next token to be an IRI; returns the IRI string (without angle brackets)
pub fn expect_iri(stream: TokenStream) -> ParseResult<String> {
    match stream.peek() {
        Some(tok) if tok.kind == TokenKind::Iri => {
            let value = tok.value.clone();
            let (_, rest) = stream.next();
            Ok((value, rest))
        }
        Some(tok) if tok.kind == TokenKind::PrefixedName => {
            let value = tok.value.clone();
            let (_, rest) = stream.next();
            Ok((value, rest))
        }
        Some(tok) => Err(ParseError::new(
            format!("expected IRI, found {:?} '{}'", tok.kind, tok.value),
            tok.position,
        )),
        None => Err(ParseError::new("expected IRI, reached end of stream", 0)),
    }
}

/// Expect the next token to be a variable; returns the variable name (without ? or $)
pub fn expect_variable(stream: TokenStream) -> ParseResult<String> {
    match stream.peek() {
        Some(tok) if tok.kind == TokenKind::Variable => {
            let value = tok.value.clone();
            let (_, rest) = stream.next();
            Ok((value, rest))
        }
        Some(tok) => Err(ParseError::new(
            format!("expected variable, found {:?} '{}'", tok.kind, tok.value),
            tok.position,
        )),
        None => Err(ParseError::new(
            "expected variable, reached end of stream",
            0,
        )),
    }
}

/// Try to apply `f`; on failure, return `None` without consuming any tokens
pub fn optional<T, F>(stream: TokenStream, f: F) -> ParseResult<Option<T>>
where
    F: Fn(TokenStream) -> ParseResult<T>,
{
    let snapshot = stream.clone();
    match f(stream) {
        Ok((value, rest)) => Ok((Some(value), rest)),
        Err(_) => Ok((None, snapshot)),
    }
}

/// Apply `f` repeatedly until it fails; return all successful results
/// The stream is advanced only for successful applications
pub fn many0<T, F>(stream: TokenStream, f: F) -> ParseResult<Vec<T>>
where
    F: Fn(TokenStream) -> ParseResult<T>,
{
    let mut results = Vec::new();
    let mut current = stream;
    loop {
        let snapshot = current.clone();
        match f(current) {
            Ok((value, rest)) => {
                results.push(value);
                current = rest;
            }
            Err(_) => {
                current = snapshot;
                break;
            }
        }
    }
    Ok((results, current))
}

/// Try each parser in order; return the first success (backtracking on failure)
pub fn choice<T>(
    stream: TokenStream,
    parsers: Vec<Box<dyn Fn(TokenStream) -> ParseResult<T>>>,
) -> ParseResult<T> {
    let mut last_err = ParseError::new("no alternatives in choice", stream.byte_offset());
    for parser in &parsers {
        let snapshot = stream.clone();
        match parser(snapshot) {
            Ok(result) => return Ok(result),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

// ─── Tokenizer ───────────────────────────────────────────────────────────────

/// Known SPARQL keywords (subset used in SPARQL 1.1/1.2)
const SPARQL_KEYWORDS: &[&str] = &[
    "BASE",
    "PREFIX",
    "SELECT",
    "DISTINCT",
    "REDUCED",
    "CONSTRUCT",
    "DESCRIBE",
    "ASK",
    "FROM",
    "NAMED",
    "WHERE",
    "ORDER",
    "BY",
    "ASC",
    "DESC",
    "LIMIT",
    "OFFSET",
    "HAVING",
    "GROUP",
    "UNION",
    "OPTIONAL",
    "MINUS",
    "GRAPH",
    "SERVICE",
    "BIND",
    "VALUES",
    "FILTER",
    "EXISTS",
    "NOT",
    "IN",
    "AS",
    "SEPARATOR",
    "COUNT",
    "SUM",
    "MIN",
    "MAX",
    "AVG",
    "SAMPLE",
    "REGEX",
    "LANG",
    "DATATYPE",
    "IRI",
    "URI",
    "BNODE",
    "STR",
    "STRDT",
    "STRLANG",
    "TRUE",
    "FALSE",
    "UNDEF",
    "LOAD",
    "CLEAR",
    "DROP",
    "CREATE",
    "ADD",
    "MOVE",
    "COPY",
    "INSERT",
    "DELETE",
    "WITH",
    "USING",
    "DATA",
    "INTO",
    "ALL",
    "DEFAULT",
    "SILENT",
    "UPDATE",
    "SPARQL",
];

/// Tokenizer for SPARQL source text
pub struct Tokenizer;

impl Tokenizer {
    /// Tokenize the full input, including whitespace and comments
    pub fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let start = i;
            let ch = chars[i];

            // Whitespace
            if ch.is_whitespace() {
                let mut end = i;
                while end < chars.len() && chars[end].is_whitespace() {
                    end += 1;
                }
                let value: String = chars[start..end].iter().collect();
                tokens.push(Token::new(TokenKind::Whitespace, value, start));
                i = end;
                continue;
            }

            // Comment
            if ch == '#' {
                let mut end = i;
                while end < chars.len() && chars[end] != '\n' {
                    end += 1;
                }
                let value: String = chars[start..end].iter().collect();
                tokens.push(Token::new(TokenKind::Comment, value, start));
                i = end;
                continue;
            }

            // IRI: <…>  (but not <= which is a comparison operator)
            if ch == '<' && !(i + 1 < chars.len() && chars[i + 1] == '=') {
                let mut end = i + 1;
                while end < chars.len() && chars[end] != '>' {
                    if chars[end] == '\n' || chars[end] == ' ' {
                        return Err(ParseError::new(
                            "unterminated IRI: unexpected whitespace inside angle brackets",
                            start,
                        ));
                    }
                    end += 1;
                }
                if end >= chars.len() {
                    return Err(ParseError::new("unterminated IRI: missing '>'", start));
                }
                end += 1; // consume '>'
                let value: String = chars[start..end].iter().collect();
                tokens.push(Token::new(TokenKind::Iri, value, start));
                i = end;
                continue;
            }

            // String literal: "…" or '…'
            if ch == '"' || ch == '\'' {
                let quote = ch;
                // Check for triple-quoted
                let triple = i + 2 < chars.len() && chars[i + 1] == quote && chars[i + 2] == quote;
                let (delim_len, close_seq): (usize, Vec<char>) = if triple {
                    (3, vec![quote, quote, quote])
                } else {
                    (1, vec![quote])
                };
                let mut end = i + delim_len;
                loop {
                    if end + close_seq.len() > chars.len() {
                        return Err(ParseError::new("unterminated string literal", start));
                    }
                    let window: Vec<char> = chars[end..end + close_seq.len()].to_vec();
                    if window == close_seq {
                        end += close_seq.len();
                        break;
                    }
                    if chars[end] == '\\' {
                        end += 2; // skip escape
                    } else {
                        end += 1;
                    }
                }
                // Optional language tag or datatype
                if end < chars.len() && chars[end] == '@' {
                    end += 1;
                    while end < chars.len() && (chars[end].is_alphanumeric() || chars[end] == '-') {
                        end += 1;
                    }
                } else if end + 1 < chars.len() && chars[end] == '^' && chars[end + 1] == '^' {
                    end += 2;
                    if end < chars.len() && chars[end] == '<' {
                        while end < chars.len() && chars[end] != '>' {
                            end += 1;
                        }
                        if end < chars.len() {
                            end += 1;
                        }
                    } else {
                        // prefixed datatype
                        while end < chars.len()
                            && (chars[end].is_alphanumeric()
                                || chars[end] == ':'
                                || chars[end] == '_')
                        {
                            end += 1;
                        }
                    }
                }
                let value: String = chars[start..end].iter().collect();
                tokens.push(Token::new(TokenKind::Literal, value, start));
                i = end;
                continue;
            }

            // Variable: ?name or $name
            if ch == '?' || ch == '$' {
                let mut end = i + 1;
                while end < chars.len() && (chars[end].is_alphanumeric() || chars[end] == '_') {
                    end += 1;
                }
                let value: String = chars[start..end].iter().collect();
                tokens.push(Token::new(TokenKind::Variable, value, start));
                i = end;
                continue;
            }

            // Numeric literal
            if ch.is_ascii_digit()
                || (ch == '-' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit())
            {
                let mut end = i;
                if chars[end] == '-' {
                    end += 1;
                }
                while end < chars.len() && chars[end].is_ascii_digit() {
                    end += 1;
                }
                if end < chars.len() && chars[end] == '.' {
                    end += 1;
                    while end < chars.len() && chars[end].is_ascii_digit() {
                        end += 1;
                    }
                }
                // Optional exponent
                if end < chars.len() && (chars[end] == 'e' || chars[end] == 'E') {
                    end += 1;
                    if end < chars.len() && (chars[end] == '+' || chars[end] == '-') {
                        end += 1;
                    }
                    while end < chars.len() && chars[end].is_ascii_digit() {
                        end += 1;
                    }
                }
                let value: String = chars[start..end].iter().collect();
                tokens.push(Token::new(TokenKind::Literal, value, start));
                i = end;
                continue;
            }

            // Keyword or prefixed name or bare identifier
            if ch.is_alphabetic() || ch == '_' {
                let mut end = i;
                while end < chars.len()
                    && (chars[end].is_alphanumeric() || chars[end] == '_' || chars[end] == '-')
                {
                    end += 1;
                }
                let word: String = chars[start..end].iter().collect();

                // Check for prefix:local
                if end < chars.len() && chars[end] == ':' {
                    end += 1; // consume ':'
                              // local part (may be empty)
                    while end < chars.len()
                        && (chars[end].is_alphanumeric()
                            || chars[end] == '_'
                            || chars[end] == '-'
                            || chars[end] == '.')
                    {
                        end += 1;
                    }
                    let full: String = chars[start..end].iter().collect();
                    tokens.push(Token::new(TokenKind::PrefixedName, full, start));
                    i = end;
                    continue;
                }

                // Check keyword (case-insensitive)
                if SPARQL_KEYWORDS
                    .iter()
                    .any(|kw| kw.eq_ignore_ascii_case(&word))
                {
                    tokens.push(Token::new(TokenKind::Keyword, word, start));
                } else {
                    // bare identifier — treat as literal for simplicity
                    tokens.push(Token::new(TokenKind::Literal, word, start));
                }
                i = end;
                continue;
            }

            // Punctuation and operators
            let punct_chars: &[char] = &[
                '{', '}', '(', ')', '[', ']', '.', ',', ';', '|', '/', '^', '+', '*', '!', '=',
                '<', '>', '&', '@',
            ];
            if punct_chars.contains(&ch) {
                // Handle two-character operators
                let two: String = if i + 1 < chars.len() {
                    chars[i..i + 2].iter().collect()
                } else {
                    String::new()
                };
                if matches!(two.as_str(), "!=" | "<=" | ">=" | "&&" | "||" | "^^") {
                    tokens.push(Token::new(TokenKind::Punctuation, two, start));
                    i += 2;
                } else {
                    tokens.push(Token::new(TokenKind::Punctuation, ch.to_string(), start));
                    i += 1;
                }
                continue;
            }

            return Err(ParseError::new(
                format!("unexpected character '{}'", ch),
                start,
            ));
        }

        tokens.push(Token::new(TokenKind::Eof, "", input.len()));
        Ok(tokens)
    }

    /// Tokenize and strip whitespace and comment tokens
    pub fn tokenize_filtered(input: &str) -> Result<Vec<Token>, ParseError> {
        let tokens = Self::tokenize(input)?;
        Ok(tokens
            .into_iter()
            .filter(|t| t.kind != TokenKind::Whitespace && t.kind != TokenKind::Comment)
            .collect())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tokenizer tests ──────────────────────────────────────────────────────

    #[test]
    fn test_tokenize_keyword_select() {
        let tokens = Tokenizer::tokenize("SELECT").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Keyword);
        assert_eq!(tokens[0].value, "SELECT");
    }

    #[test]
    fn test_tokenize_keyword_case_insensitive() {
        let tokens = Tokenizer::tokenize("select").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Keyword);
    }

    #[test]
    fn test_tokenize_keyword_where() {
        let tokens = Tokenizer::tokenize("WHERE").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Keyword);
        assert_eq!(tokens[0].value, "WHERE");
    }

    #[test]
    fn test_tokenize_keyword_prefix() {
        let tokens = Tokenizer::tokenize("PREFIX").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Keyword);
    }

    #[test]
    fn test_tokenize_keyword_optional() {
        let tokens = Tokenizer::tokenize("OPTIONAL").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Keyword);
    }

    #[test]
    fn test_tokenize_iri() {
        let tokens = Tokenizer::tokenize("<http://example.org/foo>").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Iri);
        assert_eq!(tokens[0].value, "<http://example.org/foo>");
        assert_eq!(tokens[0].position, 0);
    }

    #[test]
    fn test_tokenize_iri_position() {
        let tokens = Tokenizer::tokenize("  <http://example.org/>").unwrap();
        let iri = tokens.iter().find(|t| t.kind == TokenKind::Iri).unwrap();
        assert_eq!(iri.position, 2);
    }

    #[test]
    fn test_tokenize_variable_question_mark() {
        let tokens = Tokenizer::tokenize("?name").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Variable);
        assert_eq!(tokens[0].value, "?name");
    }

    #[test]
    fn test_tokenize_variable_dollar() {
        let tokens = Tokenizer::tokenize("$subject").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Variable);
        assert_eq!(tokens[0].value, "$subject");
    }

    #[test]
    fn test_tokenize_string_literal_double_quote() {
        let tokens = Tokenizer::tokenize("\"hello\"").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Literal);
        assert_eq!(tokens[0].value, "\"hello\"");
    }

    #[test]
    fn test_tokenize_string_literal_single_quote() {
        let tokens = Tokenizer::tokenize("'world'").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Literal);
    }

    #[test]
    fn test_tokenize_numeric_literal_integer() {
        let tokens = Tokenizer::tokenize("42").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Literal);
        assert_eq!(tokens[0].value, "42");
    }

    #[test]
    fn test_tokenize_numeric_literal_float() {
        let tokens = Tokenizer::tokenize("3.14").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Literal);
        assert_eq!(tokens[0].value, "3.14");
    }

    #[test]
    fn test_tokenize_prefixed_name() {
        let tokens = Tokenizer::tokenize("rdf:type").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::PrefixedName);
        assert_eq!(tokens[0].value, "rdf:type");
    }

    #[test]
    fn test_tokenize_prefixed_name_empty_local() {
        let tokens = Tokenizer::tokenize("ex:").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::PrefixedName);
    }

    #[test]
    fn test_tokenize_punctuation_brace() {
        let tokens = Tokenizer::tokenize("{").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Punctuation);
        assert_eq!(tokens[0].value, "{");
    }

    #[test]
    fn test_tokenize_punctuation_dot() {
        let tokens = Tokenizer::tokenize(".").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Punctuation);
        assert_eq!(tokens[0].value, ".");
    }

    #[test]
    fn test_tokenize_whitespace() {
        let tokens = Tokenizer::tokenize("   ").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Whitespace);
    }

    #[test]
    fn test_tokenize_comment() {
        let tokens = Tokenizer::tokenize("# this is a comment\n").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Comment);
        assert!(tokens[0].value.starts_with('#'));
    }

    #[test]
    fn test_tokenize_eof_appended() {
        let tokens = Tokenizer::tokenize("SELECT").unwrap();
        assert_eq!(tokens.last().unwrap().kind, TokenKind::Eof);
    }

    #[test]
    fn test_tokenize_multiple_tokens() {
        let tokens =
            Tokenizer::tokenize_filtered("SELECT ?x WHERE { ?x rdf:type <http://a.org/A> }")
                .unwrap();
        let kinds: Vec<&TokenKind> = tokens.iter().map(|t| &t.kind).collect();
        assert!(kinds.contains(&&TokenKind::Keyword));
        assert!(kinds.contains(&&TokenKind::Variable));
        assert!(kinds.contains(&&TokenKind::PrefixedName));
        assert!(kinds.contains(&&TokenKind::Iri));
        assert!(kinds.contains(&&TokenKind::Punctuation));
    }

    #[test]
    fn test_tokenize_filtered_removes_whitespace() {
        let all = Tokenizer::tokenize("SELECT ?x").unwrap();
        let filtered = Tokenizer::tokenize_filtered("SELECT ?x").unwrap();
        assert!(all.len() > filtered.len());
        assert!(!filtered.iter().any(|t| t.kind == TokenKind::Whitespace));
    }

    #[test]
    fn test_tokenize_filtered_removes_comments() {
        let filtered = Tokenizer::tokenize_filtered("SELECT # comment\n?x").unwrap();
        assert!(!filtered.iter().any(|t| t.kind == TokenKind::Comment));
    }

    #[test]
    fn test_tokenize_string_with_language_tag() {
        let tokens = Tokenizer::tokenize("\"hello\"@en").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Literal);
        assert!(tokens[0].value.contains("@en"));
    }

    #[test]
    fn test_tokenize_unterminated_iri_error() {
        let result = Tokenizer::tokenize("<http://unclosed");
        assert!(result.is_err());
    }

    // ── TokenStream tests ────────────────────────────────────────────────────

    #[test]
    fn test_stream_peek_first_token() {
        let tokens = Tokenizer::tokenize_filtered("SELECT").unwrap();
        let stream = TokenStream::new(tokens);
        let tok = stream.peek().unwrap();
        assert_eq!(tok.kind, TokenKind::Keyword);
    }

    #[test]
    fn test_stream_peek_empty() {
        let stream = TokenStream::new(vec![]);
        assert!(stream.peek().is_none());
    }

    #[test]
    fn test_stream_next_advances() {
        let tokens = Tokenizer::tokenize_filtered("SELECT ?x").unwrap();
        let stream = TokenStream::new(tokens);
        let (tok, rest) = stream.next();
        assert!(tok.is_some());
        assert_eq!(tok.unwrap().kind, TokenKind::Keyword);
        let (tok2, _) = rest.next();
        assert_eq!(tok2.unwrap().kind, TokenKind::Variable);
    }

    #[test]
    fn test_stream_remaining_count() {
        let tokens = Tokenizer::tokenize_filtered("SELECT ?x WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        // SELECT ?x WHERE + EOF = 4, but remaining excludes EOF
        assert_eq!(stream.remaining(), 3);
    }

    #[test]
    fn test_stream_is_empty_after_consuming_all() {
        let tokens = Tokenizer::tokenize_filtered("SELECT").unwrap();
        let stream = TokenStream::new(tokens);
        let (_, rest) = stream.next(); // consume SELECT
        let (_, rest2) = rest.next(); // consume EOF
        assert!(rest2.is_empty());
    }

    #[test]
    fn test_stream_position_zero_initially() {
        let tokens = Tokenizer::tokenize_filtered("WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        assert_eq!(stream.position(), 0);
    }

    #[test]
    fn test_stream_position_advances() {
        let tokens = Tokenizer::tokenize_filtered("SELECT ?x").unwrap();
        let stream = TokenStream::new(tokens);
        let (_, rest) = stream.next();
        assert_eq!(rest.position(), 1);
    }

    // ── expect_keyword tests ─────────────────────────────────────────────────

    #[test]
    fn test_expect_keyword_success() {
        let tokens = Tokenizer::tokenize_filtered("SELECT").unwrap();
        let stream = TokenStream::new(tokens);
        let result = expect_keyword(stream, "SELECT");
        assert!(result.is_ok());
    }

    #[test]
    fn test_expect_keyword_case_insensitive() {
        let tokens = Tokenizer::tokenize_filtered("select").unwrap();
        let stream = TokenStream::new(tokens);
        assert!(expect_keyword(stream, "SELECT").is_ok());
    }

    #[test]
    fn test_expect_keyword_wrong_keyword() {
        let tokens = Tokenizer::tokenize_filtered("WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        let result = expect_keyword(stream, "SELECT");
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("SELECT"));
    }

    #[test]
    fn test_expect_keyword_not_a_keyword() {
        let tokens = Tokenizer::tokenize_filtered("?x").unwrap();
        let stream = TokenStream::new(tokens);
        let result = expect_keyword(stream, "SELECT");
        assert!(result.is_err());
    }

    #[test]
    fn test_expect_keyword_consumes_token() {
        let tokens = Tokenizer::tokenize_filtered("SELECT WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        let (_, rest) = expect_keyword(stream, "SELECT").unwrap();
        assert!(expect_keyword(rest, "WHERE").is_ok());
    }

    // ── expect_iri tests ─────────────────────────────────────────────────────

    #[test]
    fn test_expect_iri_success() {
        let tokens = Tokenizer::tokenize_filtered("<http://example.org/>").unwrap();
        let stream = TokenStream::new(tokens);
        let result = expect_iri(stream);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().0, "<http://example.org/>");
    }

    #[test]
    fn test_expect_iri_prefixed_name() {
        let tokens = Tokenizer::tokenize_filtered("rdf:type").unwrap();
        let stream = TokenStream::new(tokens);
        let result = expect_iri(stream);
        assert!(result.is_ok());
    }

    #[test]
    fn test_expect_iri_failure_on_variable() {
        let tokens = Tokenizer::tokenize_filtered("?x").unwrap();
        let stream = TokenStream::new(tokens);
        let result = expect_iri(stream);
        assert!(result.is_err());
    }

    // ── expect_variable tests ────────────────────────────────────────────────

    #[test]
    fn test_expect_variable_success() {
        let tokens = Tokenizer::tokenize_filtered("?subject").unwrap();
        let stream = TokenStream::new(tokens);
        let result = expect_variable(stream);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().0, "?subject");
    }

    #[test]
    fn test_expect_variable_dollar_prefix() {
        let tokens = Tokenizer::tokenize_filtered("$pred").unwrap();
        let stream = TokenStream::new(tokens);
        let result = expect_variable(stream);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().0, "$pred");
    }

    #[test]
    fn test_expect_variable_failure_on_keyword() {
        let tokens = Tokenizer::tokenize_filtered("SELECT").unwrap();
        let stream = TokenStream::new(tokens);
        let result = expect_variable(stream);
        assert!(result.is_err());
    }

    // ── optional tests ───────────────────────────────────────────────────────

    #[test]
    fn test_optional_hit() {
        let tokens = Tokenizer::tokenize_filtered("SELECT").unwrap();
        let stream = TokenStream::new(tokens);
        let (result, _) = optional(stream, |s| expect_keyword(s, "SELECT")).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_optional_miss_returns_none() {
        let tokens = Tokenizer::tokenize_filtered("WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        let (result, rest) = optional(stream, |s| expect_keyword(s, "SELECT")).unwrap();
        assert!(result.is_none());
        // Stream should not have advanced
        assert_eq!(rest.position(), 0);
    }

    #[test]
    fn test_optional_miss_does_not_advance_stream() {
        let tokens = Tokenizer::tokenize_filtered("?x").unwrap();
        let stream = TokenStream::new(tokens);
        let pos_before = stream.position();
        let (_, rest) = optional(stream, |s| expect_keyword(s, "SELECT")).unwrap();
        assert_eq!(rest.position(), pos_before);
    }

    // ── many0 tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_many0_zero_matches() {
        let tokens = Tokenizer::tokenize_filtered("WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        let (results, rest) = many0(stream, |s| expect_keyword(s, "SELECT")).unwrap();
        assert_eq!(results.len(), 0);
        assert_eq!(rest.position(), 0);
    }

    #[test]
    fn test_many0_one_match() {
        let tokens = Tokenizer::tokenize_filtered("SELECT WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        let (results, _) = many0(stream, |s| expect_keyword(s, "SELECT")).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_many0_multiple_matches() {
        let tokens = Tokenizer::tokenize_filtered("SELECT SELECT SELECT WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        let (results, rest) = many0(stream, |s| expect_keyword(s, "SELECT")).unwrap();
        assert_eq!(results.len(), 3);
        // remaining should show WHERE + EOF
        assert!(rest.remaining() >= 1);
    }

    #[test]
    fn test_many0_variables() {
        let tokens = Tokenizer::tokenize_filtered("?a ?b ?c WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        let (vars, _) = many0(stream, expect_variable).unwrap();
        assert_eq!(vars.len(), 3);
        assert_eq!(vars[0], "?a");
        assert_eq!(vars[1], "?b");
        assert_eq!(vars[2], "?c");
    }

    // ── choice tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_choice_first_alternative() {
        let tokens = Tokenizer::tokenize_filtered("SELECT").unwrap();
        let stream = TokenStream::new(tokens);
        let parsers: Vec<Box<dyn Fn(TokenStream) -> ParseResult<&'static str>>> = vec![
            Box::new(|s| expect_keyword(s, "SELECT").map(|(_, r)| ("SELECT", r))),
            Box::new(|s| expect_keyword(s, "ASK").map(|(_, r)| ("ASK", r))),
        ];
        let (result, _) = choice(stream, parsers).unwrap();
        assert_eq!(result, "SELECT");
    }

    #[test]
    fn test_choice_second_alternative() {
        let tokens = Tokenizer::tokenize_filtered("ASK").unwrap();
        let stream = TokenStream::new(tokens);
        let parsers: Vec<Box<dyn Fn(TokenStream) -> ParseResult<&'static str>>> = vec![
            Box::new(|s| expect_keyword(s, "SELECT").map(|(_, r)| ("SELECT", r))),
            Box::new(|s| expect_keyword(s, "ASK").map(|(_, r)| ("ASK", r))),
        ];
        let (result, _) = choice(stream, parsers).unwrap();
        assert_eq!(result, "ASK");
    }

    #[test]
    fn test_choice_no_match_returns_error() {
        let tokens = Tokenizer::tokenize_filtered("WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        let parsers: Vec<Box<dyn Fn(TokenStream) -> ParseResult<&'static str>>> = vec![
            Box::new(|s| expect_keyword(s, "SELECT").map(|(_, r)| ("SELECT", r))),
            Box::new(|s| expect_keyword(s, "ASK").map(|(_, r)| ("ASK", r))),
        ];
        assert!(choice(stream, parsers).is_err());
    }

    #[test]
    fn test_choice_empty_parsers_returns_error() {
        let tokens = Tokenizer::tokenize_filtered("SELECT").unwrap();
        let stream = TokenStream::new(tokens);
        let parsers: Vec<Box<dyn Fn(TokenStream) -> ParseResult<String>>> = vec![];
        assert!(choice(stream, parsers).is_err());
    }

    // ── ParseError tests ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_error_position() {
        let tokens = Tokenizer::tokenize_filtered("?x").unwrap();
        let stream = TokenStream::new(tokens);
        let err = expect_keyword(stream, "SELECT").unwrap_err();
        assert_eq!(err.position, 0);
    }

    #[test]
    fn test_parse_error_message_contains_expected() {
        let tokens = Tokenizer::tokenize_filtered("?x").unwrap();
        let stream = TokenStream::new(tokens);
        let err = expect_keyword(stream, "SELECT").unwrap_err();
        assert!(err.message.contains("SELECT"));
    }

    #[test]
    fn test_parse_error_display() {
        let err = ParseError::new("test error", 42);
        let display = format!("{}", err);
        assert!(display.contains("42"));
        assert!(display.contains("test error"));
    }

    // ── Composite parsing tests ──────────────────────────────────────────────

    #[test]
    fn test_parse_simple_triple_pattern() {
        // Parse: ?s rdf:type ?o
        let tokens = Tokenizer::tokenize_filtered("?s rdf:type ?o").unwrap();
        let stream = TokenStream::new(tokens);

        let (subj, rest) = expect_variable(stream).unwrap();
        let (pred, rest) = expect_iri(rest).unwrap();
        let (obj, _) = expect_variable(rest).unwrap();

        assert_eq!(subj, "?s");
        assert_eq!(pred, "rdf:type");
        assert_eq!(obj, "?o");
    }

    #[test]
    fn test_parse_select_query_skeleton() {
        let tokens = Tokenizer::tokenize_filtered("SELECT ?x WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        let (_, rest) = expect_keyword(stream, "SELECT").unwrap();
        let (vars, rest) = many0(rest, expect_variable).unwrap();
        let (_, _) = expect_keyword(rest, "WHERE").unwrap();
        assert_eq!(vars, vec!["?x"]);
    }

    #[test]
    fn test_token_new() {
        let tok = Token::new(TokenKind::Keyword, "SELECT", 0);
        assert_eq!(tok.kind, TokenKind::Keyword);
        assert_eq!(tok.value, "SELECT");
        assert_eq!(tok.position, 0);
    }

    #[test]
    fn test_parse_error_new() {
        let err = ParseError::new("oops", 5);
        assert_eq!(err.position, 5);
        assert_eq!(err.message, "oops");
    }

    #[test]
    fn test_tokenize_two_char_operator_neq() {
        let tokens = Tokenizer::tokenize("!=").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Punctuation);
        assert_eq!(tokens[0].value, "!=");
    }

    #[test]
    fn test_tokenize_two_char_operator_leq() {
        let tokens = Tokenizer::tokenize("<=").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Punctuation);
        assert_eq!(tokens[0].value, "<=");
    }

    #[test]
    fn test_tokenize_keyword_filter() {
        let tokens = Tokenizer::tokenize("FILTER").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Keyword);
    }

    #[test]
    fn test_tokenize_keyword_bind() {
        let tokens = Tokenizer::tokenize("BIND").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Keyword);
    }

    #[test]
    fn test_stream_clone_independence() {
        let tokens = Tokenizer::tokenize_filtered("SELECT WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        let clone = stream.clone();
        let (_, advanced) = stream.next();
        // Original clone should still be at position 0
        assert_eq!(clone.position(), 0);
        assert_eq!(advanced.position(), 1);
    }

    #[test]
    fn test_many0_with_iri() {
        let tokens = Tokenizer::tokenize_filtered("<http://a.org/> <http://b.org/> ?x").unwrap();
        let stream = TokenStream::new(tokens);
        let (iris, rest) = many0(stream, expect_iri).unwrap();
        assert_eq!(iris.len(), 2);
        assert_eq!(iris[0], "<http://a.org/>");
        assert_eq!(iris[1], "<http://b.org/>");
        // ?x should still be next
        assert_eq!(rest.peek().unwrap().kind, TokenKind::Variable);
    }

    #[test]
    fn test_optional_iri_hit() {
        let tokens = Tokenizer::tokenize_filtered("<http://example.org/> WHERE").unwrap();
        let stream = TokenStream::new(tokens);
        let (result, _) = optional(stream, expect_iri).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_optional_variable_miss_on_keyword() {
        let tokens = Tokenizer::tokenize_filtered("SELECT").unwrap();
        let stream = TokenStream::new(tokens);
        let (result, rest) = optional(stream, expect_variable).unwrap();
        assert!(result.is_none());
        assert_eq!(rest.position(), 0);
    }
}
