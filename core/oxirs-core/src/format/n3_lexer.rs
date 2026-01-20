//! N3/Turtle Lexer Implementation
//!
//! Provides tokenization for the N3 family of formats (Turtle, TriG, N3)
//! Extracted and adapted from OxiGraph with OxiRS enhancements.

use super::error::{ParseResult, RdfParseError, RdfSyntaxError, TextPosition};
use super::toolkit::{char_utils, BufferProvider, TokenRecognizer};
use std::fmt;

/// Token types for N3/Turtle formats
#[derive(Debug, Clone, PartialEq)]
pub enum N3Token {
    // Punctuation
    Dot,          // .
    Semicolon,    // ;
    Comma,        // ,
    LeftBracket,  // [
    RightBracket, // ]
    LeftParen,    // (
    RightParen,   // )
    LeftBrace,    // {
    RightBrace,   // }

    // Operators
    Prefix, // @prefix
    Base,   // @base
    A,      // a (shorthand for rdf:type)

    // Terms
    Iri(String), // <http://example.org>
    PrefixedName {
        // prefix:localName
        prefix: Option<String>,
        local: String,
    },
    BlankNode(String), // _:label
    Literal {
        value: String,
        datatype: Option<String>,
        language: Option<String>,
    },
    Variable(String), // ?var or $var (for SPARQL compatibility)

    // RDF-star support
    QuotedTripleStart, // <<
    QuotedTripleEnd,   // >>

    // Special values
    True,  // true
    False, // false

    // Numeric literals
    Integer(i64),
    Decimal(f64),
    Double(f64),

    // Comments and whitespace (usually skipped)
    Comment(String),
    Whitespace,

    // End of input
    Eof,
}

impl fmt::Display for N3Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            N3Token::Dot => write!(f, "."),
            N3Token::Semicolon => write!(f, ";"),
            N3Token::Comma => write!(f, ","),
            N3Token::LeftBracket => write!(f, "["),
            N3Token::RightBracket => write!(f, "]"),
            N3Token::LeftParen => write!(f, "("),
            N3Token::RightParen => write!(f, ")"),
            N3Token::LeftBrace => write!(f, "{{"),
            N3Token::RightBrace => write!(f, "}}"),
            N3Token::Prefix => write!(f, "@prefix"),
            N3Token::Base => write!(f, "@base"),
            N3Token::A => write!(f, "a"),
            N3Token::Iri(iri) => write!(f, "<{iri}>"),
            N3Token::PrefixedName {
                prefix: Some(prefix),
                local,
            } => write!(f, "{prefix}:{local}"),
            N3Token::PrefixedName {
                prefix: None,
                local,
            } => write!(f, ":{local}"),
            N3Token::BlankNode(label) => write!(f, "_:{label}"),
            N3Token::Literal {
                value,
                datatype: Some(dt),
                language: None,
            } => write!(f, "\"{value}\"^^<{dt}>"),
            N3Token::Literal {
                value,
                datatype: None,
                language: Some(lang),
            } => write!(f, "\"{value}\"@{lang}"),
            N3Token::Literal {
                value,
                datatype: None,
                language: None,
            } => write!(f, "\"{value}\""),
            N3Token::Literal {
                value,
                datatype: Some(dt),
                language: Some(lang),
            } => write!(f, "\"{value}\"@{lang}^^<{dt}>"),
            N3Token::Variable(var) => write!(f, "?{var}"),
            N3Token::QuotedTripleStart => write!(f, "<<"),
            N3Token::QuotedTripleEnd => write!(f, ">>"),
            N3Token::True => write!(f, "true"),
            N3Token::False => write!(f, "false"),
            N3Token::Integer(i) => write!(f, "{i}"),
            N3Token::Decimal(d) => write!(f, "{d}"),
            N3Token::Double(d) => write!(f, "{d}"),
            N3Token::Comment(comment) => write!(f, "# {comment}"),
            N3Token::Whitespace => write!(f, " "),
            N3Token::Eof => write!(f, "EOF"),
        }
    }
}

/// N3/Turtle lexer implementation
#[derive(Debug, Clone)]
pub struct N3Lexer {
    /// Skip whitespace and comments
    pub skip_whitespace: bool,
    /// Parse variables (for SPARQL compatibility)
    pub parse_variables: bool,
}

impl Default for N3Lexer {
    fn default() -> Self {
        Self {
            skip_whitespace: true,
            parse_variables: false,
        }
    }
}

impl N3Lexer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_variables(mut self) -> Self {
        self.parse_variables = true;
        self
    }

    pub fn with_whitespace(mut self) -> Self {
        self.skip_whitespace = false;
        self
    }

    /// Read an IRI enclosed in angle brackets
    fn read_iri(&self, buffer: &mut dyn BufferProvider) -> ParseResult<String> {
        let mut iri = String::new();

        // Skip opening '<'
        buffer.advance();

        while let Some(ch) = buffer.current() {
            match ch {
                '>' => {
                    buffer.advance();
                    return Ok(iri);
                }
                '\\' => {
                    buffer.advance();
                    match buffer.current() {
                        Some('u') => {
                            buffer.advance();
                            let unicode = self.read_unicode_escape(buffer, 4)?;
                            iri.push(unicode);
                        }
                        Some('U') => {
                            buffer.advance();
                            let unicode = self.read_unicode_escape(buffer, 8)?;
                            iri.push(unicode);
                        }
                        Some(escaped) => {
                            // Handle other escape sequences
                            match escaped {
                                't' => iri.push('\t'),
                                'n' => iri.push('\n'),
                                'r' => iri.push('\r'),
                                '\\' => iri.push('\\'),
                                '>' => iri.push('>'),
                                _ => {
                                    return Err(RdfParseError::Syntax(
                                        RdfSyntaxError::with_position(
                                            format!("Invalid IRI escape sequence: \\{escaped}"),
                                            *buffer.position(),
                                        ),
                                    ));
                                }
                            }
                            buffer.advance();
                        }
                        None => {
                            return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                                "Unexpected end of IRI".to_string(),
                                *buffer.position(),
                            )));
                        }
                    }
                }
                ch if char_utils::is_iri_char(ch) => {
                    iri.push(ch);
                    buffer.advance();
                }
                _ => {
                    return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        format!("Invalid character in IRI: '{ch}'"),
                        *buffer.position(),
                    )));
                }
            }
        }

        Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
            "Unclosed IRI".to_string(),
            *buffer.position(),
        )))
    }

    /// Read a Unicode escape sequence
    fn read_unicode_escape(
        &self,
        buffer: &mut dyn BufferProvider,
        digits: usize,
    ) -> ParseResult<char> {
        let mut unicode_str = String::new();

        for _ in 0..digits {
            match buffer.current() {
                Some(ch) if char_utils::is_hex_digit(ch) => {
                    unicode_str.push(ch);
                    buffer.advance();
                }
                Some(ch) => {
                    return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        format!("Invalid hex digit in Unicode escape: '{ch}'"),
                        *buffer.position(),
                    )));
                }
                None => {
                    return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        "Unexpected end of Unicode escape".to_string(),
                        *buffer.position(),
                    )));
                }
            }
        }

        let code_point = u32::from_str_radix(&unicode_str, 16).map_err(|_| {
            RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Invalid Unicode code point".to_string(),
                *buffer.position(),
            ))
        })?;

        char::from_u32(code_point).ok_or_else(|| {
            RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Invalid Unicode code point".to_string(),
                *buffer.position(),
            ))
        })
    }

    /// Read a prefixed name (prefix:localName or :localName)
    fn read_prefixed_name(&self, buffer: &mut dyn BufferProvider) -> ParseResult<N3Token> {
        let mut prefix = String::new();

        // Read prefix part (before colon)
        while let Some(ch) = buffer.current() {
            if ch == ':' {
                buffer.advance();
                break;
            } else if char_utils::is_pn_chars(ch) {
                prefix.push(ch);
                buffer.advance();
            } else {
                return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                    format!("Invalid character in prefix: '{ch}'"),
                    *buffer.position(),
                )));
            }
        }

        // Read local name part (after colon)
        let mut local = String::new();
        while let Some(ch) = buffer.current() {
            if char_utils::is_pn_chars(ch) || ch == '.' {
                local.push(ch);
                buffer.advance();
            } else {
                break;
            }
        }

        let prefix_opt = if prefix.is_empty() {
            None
        } else {
            Some(prefix)
        };

        Ok(N3Token::PrefixedName {
            prefix: prefix_opt,
            local,
        })
    }

    /// Read a blank node identifier
    fn read_blank_node(&self, buffer: &mut dyn BufferProvider) -> ParseResult<String> {
        // Skip '_:'
        buffer.advance(); // skip '_'
        if buffer.current() != Some(':') {
            return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                "Expected ':' after '_' in blank node".to_string(),
                *buffer.position(),
            )));
        }
        buffer.advance(); // skip ':'

        let mut label = String::new();

        // First character must be PN_CHARS_BASE or digit
        match buffer.current() {
            Some(ch) if char_utils::is_pn_chars_base(ch) || char_utils::is_digit(ch) => {
                label.push(ch);
                buffer.advance();
            }
            Some(ch) => {
                return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                    format!("Invalid first character in blank node label: '{ch}'"),
                    *buffer.position(),
                )));
            }
            None => {
                return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                    "Expected blank node label after '_:'".to_string(),
                    *buffer.position(),
                )));
            }
        }

        // Rest of the characters
        while let Some(ch) = buffer.current() {
            if char_utils::is_pn_chars(ch) {
                label.push(ch);
                buffer.advance();
            } else {
                break;
            }
        }

        Ok(label)
    }

    /// Read a string literal
    fn read_string_literal(
        &self,
        buffer: &mut dyn BufferProvider,
        quote_char: char,
    ) -> ParseResult<N3Token> {
        buffer.advance(); // Skip opening quote

        let mut value = String::new();
        let mut triple_quoted = false;

        // Check for triple quotes
        if buffer.current() == Some(quote_char) {
            buffer.advance();
            if buffer.current() == Some(quote_char) {
                buffer.advance();
                triple_quoted = true;
            } else {
                // Empty string with single quote
                return self.read_literal_suffix(buffer, value);
            }
        }

        while let Some(ch) = buffer.current() {
            match ch {
                c if c == quote_char => {
                    if triple_quoted {
                        // Check for triple quote ending
                        buffer.advance();
                        if buffer.current() == Some(quote_char) {
                            buffer.advance();
                            if buffer.current() == Some(quote_char) {
                                buffer.advance();
                                break; // End of triple-quoted string
                            } else {
                                // Just two quotes, continue
                                value.push(quote_char);
                                value.push(quote_char);
                            }
                        } else {
                            // Just one quote, continue
                            value.push(quote_char);
                        }
                    } else {
                        buffer.advance();
                        break; // End of single-quoted string
                    }
                }
                '\\' => {
                    buffer.advance();
                    match buffer.current() {
                        Some('t') => value.push('\t'),
                        Some('n') => value.push('\n'),
                        Some('r') => value.push('\r'),
                        Some('b') => value.push('\u{0008}'),
                        Some('f') => value.push('\u{000C}'),
                        Some('"') => value.push('"'),
                        Some('\'') => value.push('\''),
                        Some('\\') => value.push('\\'),
                        Some('u') => {
                            buffer.advance();
                            let unicode = self.read_unicode_escape(buffer, 4)?;
                            value.push(unicode);
                            continue; // Don't advance again
                        }
                        Some('U') => {
                            buffer.advance();
                            let unicode = self.read_unicode_escape(buffer, 8)?;
                            value.push(unicode);
                            continue; // Don't advance again
                        }
                        Some(other) => {
                            return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                                format!("Invalid escape sequence: \\{other}"),
                                *buffer.position(),
                            )));
                        }
                        None => {
                            return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                                "Unexpected end of string literal".to_string(),
                                *buffer.position(),
                            )));
                        }
                    }
                    buffer.advance();
                }
                '\n' | '\r' if !triple_quoted => {
                    return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        "Newline in single-quoted string literal".to_string(),
                        *buffer.position(),
                    )));
                }
                _ => {
                    value.push(ch);
                    buffer.advance();
                }
            }
        }

        self.read_literal_suffix(buffer, value)
    }

    /// Read datatype or language tag suffix for literals
    fn read_literal_suffix(
        &self,
        buffer: &mut dyn BufferProvider,
        value: String,
    ) -> ParseResult<N3Token> {
        let mut datatype = None;
        let mut language = None;

        // Check for language tag (@lang) or datatype (^^<iri>)
        match buffer.current() {
            Some('@') => {
                buffer.advance();
                let mut lang = String::new();
                while let Some(ch) = buffer.current() {
                    if ch.is_ascii_alphanumeric() || ch == '-' {
                        lang.push(ch);
                        buffer.advance();
                    } else {
                        break;
                    }
                }
                language = Some(lang);
            }
            Some('^') => {
                buffer.advance();
                if buffer.current() == Some('^') {
                    buffer.advance();
                    if buffer.current() == Some('<') {
                        let dt = self.read_iri(buffer)?;
                        datatype = Some(dt);
                    } else {
                        return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                            "Expected '<' after '^^' in datatype".to_string(),
                            *buffer.position(),
                        )));
                    }
                } else {
                    return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        "Expected '^' after first '^' in datatype".to_string(),
                        *buffer.position(),
                    )));
                }
            }
            _ => {} // No suffix
        }

        Ok(N3Token::Literal {
            value,
            datatype,
            language,
        })
    }

    /// Read a numeric literal
    fn read_numeric(&self, buffer: &mut dyn BufferProvider) -> ParseResult<N3Token> {
        let mut number_str = String::new();
        let mut has_decimal = false;
        let mut has_exponent = false;

        // Handle optional sign
        if matches!(buffer.current(), Some('+') | Some('-')) {
            number_str.push(buffer.current().unwrap());
            buffer.advance();
        }

        // Read digits before decimal or exponent
        while let Some(ch) = buffer.current() {
            if char_utils::is_digit(ch) {
                number_str.push(ch);
                buffer.advance();
            } else {
                break;
            }
        }

        // Check for decimal point
        if buffer.current() == Some('.') {
            has_decimal = true;
            number_str.push('.');
            buffer.advance();

            // Read digits after decimal
            while let Some(ch) = buffer.current() {
                if char_utils::is_digit(ch) {
                    number_str.push(ch);
                    buffer.advance();
                } else {
                    break;
                }
            }
        }

        // Check for exponent
        if matches!(buffer.current(), Some('e') | Some('E')) {
            has_exponent = true;
            number_str.push(buffer.current().unwrap());
            buffer.advance();

            // Handle optional sign in exponent
            if matches!(buffer.current(), Some('+') | Some('-')) {
                number_str.push(buffer.current().unwrap());
                buffer.advance();
            }

            // Read exponent digits
            while let Some(ch) = buffer.current() {
                if char_utils::is_digit(ch) {
                    number_str.push(ch);
                    buffer.advance();
                } else {
                    break;
                }
            }
        }

        // Parse the number based on its format
        if has_exponent {
            // Double
            let value = number_str.parse::<f64>().map_err(|_| {
                RdfParseError::Syntax(RdfSyntaxError::with_position(
                    format!("Invalid double literal: {number_str}"),
                    *buffer.position(),
                ))
            })?;
            Ok(N3Token::Double(value))
        } else if has_decimal {
            // Decimal
            let value = number_str.parse::<f64>().map_err(|_| {
                RdfParseError::Syntax(RdfSyntaxError::with_position(
                    format!("Invalid decimal literal: {number_str}"),
                    *buffer.position(),
                ))
            })?;
            Ok(N3Token::Decimal(value))
        } else {
            // Integer
            let value = number_str.parse::<i64>().map_err(|_| {
                RdfParseError::Syntax(RdfSyntaxError::with_position(
                    format!("Invalid integer literal: {number_str}"),
                    *buffer.position(),
                ))
            })?;
            Ok(N3Token::Integer(value))
        }
    }

    /// Read a keyword or identifier
    fn read_keyword(&self, buffer: &mut dyn BufferProvider) -> ParseResult<N3Token> {
        let mut keyword = String::new();

        while let Some(ch) = buffer.current() {
            if char_utils::is_pn_chars(ch) {
                keyword.push(ch);
                buffer.advance();
            } else {
                break;
            }
        }

        match keyword.as_str() {
            "true" => Ok(N3Token::True),
            "false" => Ok(N3Token::False),
            "a" => Ok(N3Token::A),
            _ => {
                // Check if it's a prefixed name
                if buffer.current() == Some(':') {
                    buffer.advance();
                    let mut local = String::new();
                    while let Some(ch) = buffer.current() {
                        if char_utils::is_pn_chars(ch) || ch == '.' {
                            local.push(ch);
                            buffer.advance();
                        } else {
                            break;
                        }
                    }
                    Ok(N3Token::PrefixedName {
                        prefix: Some(keyword),
                        local,
                    })
                } else {
                    // Just a keyword/identifier - treat as prefixed name without prefix
                    Ok(N3Token::PrefixedName {
                        prefix: None,
                        local: keyword,
                    })
                }
            }
        }
    }

    /// Read a comment
    fn read_comment(&self, buffer: &mut dyn BufferProvider) -> ParseResult<String> {
        buffer.advance(); // Skip '#'

        let mut comment = String::new();
        while let Some(ch) = buffer.current() {
            if ch == '\n' || ch == '\r' {
                break;
            }
            comment.push(ch);
            buffer.advance();
        }

        Ok(comment)
    }

    /// Skip whitespace
    fn skip_whitespace(&self, buffer: &mut dyn BufferProvider) {
        while let Some(ch) = buffer.current() {
            if char_utils::is_whitespace(ch) {
                buffer.advance();
            } else {
                break;
            }
        }
    }
}

impl TokenRecognizer for N3Lexer {
    type Token = N3Token;
    fn recognize_next_token(
        &mut self,
        buffer: &mut dyn BufferProvider,
        _position: &mut TextPosition,
    ) -> ParseResult<Option<N3Token>> {
        loop {
            match buffer.current() {
                None => return Ok(Some(N3Token::Eof)),

                Some(ch) if char_utils::is_whitespace(ch) => {
                    if self.skip_whitespace {
                        self.skip_whitespace(buffer);
                        continue;
                    } else {
                        buffer.advance();
                        return Ok(Some(N3Token::Whitespace));
                    }
                }

                Some('#') => {
                    if self.skip_whitespace {
                        self.read_comment(buffer)?;
                        continue;
                    } else {
                        let comment = self.read_comment(buffer)?;
                        return Ok(Some(N3Token::Comment(comment)));
                    }
                }

                // Punctuation
                Some('.') => {
                    buffer.advance();
                    return Ok(Some(N3Token::Dot));
                }
                Some(';') => {
                    buffer.advance();
                    return Ok(Some(N3Token::Semicolon));
                }
                Some(',') => {
                    buffer.advance();
                    return Ok(Some(N3Token::Comma));
                }
                Some('[') => {
                    buffer.advance();
                    return Ok(Some(N3Token::LeftBracket));
                }
                Some(']') => {
                    buffer.advance();
                    return Ok(Some(N3Token::RightBracket));
                }
                Some('(') => {
                    buffer.advance();
                    return Ok(Some(N3Token::LeftParen));
                }
                Some(')') => {
                    buffer.advance();
                    return Ok(Some(N3Token::RightParen));
                }
                Some('{') => {
                    buffer.advance();
                    return Ok(Some(N3Token::LeftBrace));
                }
                Some('}') => {
                    buffer.advance();
                    return Ok(Some(N3Token::RightBrace));
                }

                // IRI or quoted triple start
                Some('<') => {
                    // Check if this is the start of a quoted triple <<
                    if buffer.peek() == Some('<') {
                        buffer.advance(); // First <
                        buffer.advance(); // Second <
                        return Ok(Some(N3Token::QuotedTripleStart));
                    } else {
                        let iri = self.read_iri(buffer)?;
                        return Ok(Some(N3Token::Iri(iri)));
                    }
                }

                // Quoted triple end
                Some('>') => {
                    // Check if this is the end of a quoted triple >>
                    if buffer.peek() == Some('>') {
                        buffer.advance(); // First >
                        buffer.advance(); // Second >
                        return Ok(Some(N3Token::QuotedTripleEnd));
                    } else {
                        return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                            "Unexpected '>' character".to_string(),
                            *buffer.position(),
                        )));
                    }
                }

                // Blank node
                Some('_') => {
                    let label = self.read_blank_node(buffer)?;
                    return Ok(Some(N3Token::BlankNode(label)));
                }

                // String literals
                Some('"') => {
                    let literal = self.read_string_literal(buffer, '"')?;
                    return Ok(Some(literal));
                }
                Some('\'') => {
                    let literal = self.read_string_literal(buffer, '\'')?;
                    return Ok(Some(literal));
                }

                // Variables (for SPARQL compatibility)
                Some('?') | Some('$') if self.parse_variables => {
                    let _var_char = buffer.current().unwrap();
                    buffer.advance();
                    let mut var_name = String::new();
                    while let Some(ch) = buffer.current() {
                        if char_utils::is_pn_chars(ch) {
                            var_name.push(ch);
                            buffer.advance();
                        } else {
                            break;
                        }
                    }
                    return Ok(Some(N3Token::Variable(var_name)));
                }

                // Directives
                Some('@') => {
                    buffer.advance();
                    let mut directive = String::new();
                    while let Some(ch) = buffer.current() {
                        if char_utils::is_pn_chars(ch) {
                            directive.push(ch);
                            buffer.advance();
                        } else {
                            break;
                        }
                    }

                    match directive.as_str() {
                        "prefix" => return Ok(Some(N3Token::Prefix)),
                        "base" => return Ok(Some(N3Token::Base)),
                        _ => {
                            return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                                format!("Unknown directive: @{directive}"),
                                *buffer.position(),
                            )));
                        }
                    }
                }

                // Numeric literals or prefixed names starting with digit
                Some(ch) if char_utils::is_numeric_start(ch) => {
                    let token = self.read_numeric(buffer)?;
                    return Ok(Some(token));
                }

                // Prefixed names starting with ':'
                Some(':') => {
                    let token = self.read_prefixed_name(buffer)?;
                    return Ok(Some(token));
                }

                // Keywords and prefixed names
                Some(ch) if char_utils::is_pn_chars_base(ch) => {
                    let token = self.read_keyword(buffer)?;
                    return Ok(Some(token));
                }

                Some(ch) => {
                    return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                        format!("Unexpected character: '{ch}'"),
                        *buffer.position(),
                    )));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::toolkit::StringBuffer;
    use super::*;

    fn tokenize_string(input: &str) -> ParseResult<Vec<N3Token>> {
        let mut buffer = StringBuffer::new(input.to_string());
        let mut lexer = N3Lexer::new();
        let mut tokens = Vec::new();

        loop {
            match lexer.recognize_next_token(&mut buffer, &mut TextPosition::start())? {
                Some(N3Token::Eof) => break,
                Some(token) => tokens.push(token),
                None => break,
            }
        }

        Ok(tokens)
    }

    #[test]
    fn test_basic_punctuation() {
        let tokens = tokenize_string(". ; , [ ] ( ) { }").unwrap();
        assert_eq!(
            tokens,
            vec![
                N3Token::Dot,
                N3Token::Semicolon,
                N3Token::Comma,
                N3Token::LeftBracket,
                N3Token::RightBracket,
                N3Token::LeftParen,
                N3Token::RightParen,
                N3Token::LeftBrace,
                N3Token::RightBrace,
            ]
        );
    }

    #[test]
    fn test_iri() {
        let tokens = tokenize_string("<http://example.org>").unwrap();
        assert_eq!(tokens, vec![N3Token::Iri("http://example.org".to_string())]);
    }

    #[test]
    fn test_prefixed_name() {
        let tokens = tokenize_string("ex:name :name").unwrap();
        assert_eq!(
            tokens,
            vec![
                N3Token::PrefixedName {
                    prefix: Some("ex".to_string()),
                    local: "name".to_string()
                },
                N3Token::PrefixedName {
                    prefix: None,
                    local: "name".to_string()
                },
            ]
        );
    }

    #[test]
    fn test_blank_node() {
        let tokens = tokenize_string("_:blank1").unwrap();
        assert_eq!(tokens, vec![N3Token::BlankNode("blank1".to_string())]);
    }

    #[test]
    fn test_string_literal() {
        let tokens = tokenize_string("\"hello world\"").unwrap();
        assert_eq!(
            tokens,
            vec![N3Token::Literal {
                value: "hello world".to_string(),
                datatype: None,
                language: None,
            }]
        );
    }

    #[test]
    fn test_string_literal_with_language() {
        let tokens = tokenize_string("\"hello\"@en").unwrap();
        assert_eq!(
            tokens,
            vec![N3Token::Literal {
                value: "hello".to_string(),
                datatype: None,
                language: Some("en".to_string()),
            }]
        );
    }

    #[test]
    fn test_string_literal_with_datatype() {
        let tokens = tokenize_string("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>").unwrap();
        assert_eq!(
            tokens,
            vec![N3Token::Literal {
                value: "42".to_string(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                language: None,
            }]
        );
    }

    #[test]
    fn test_numeric_literals() {
        let tokens = tokenize_string("42 3.14 1.5e10").unwrap();
        assert_eq!(
            tokens,
            vec![
                N3Token::Integer(42),
                #[allow(clippy::approx_constant)]
                N3Token::Decimal(3.14),
                N3Token::Double(1.5e10),
            ]
        );
    }

    #[test]
    fn test_boolean_literals() {
        let tokens = tokenize_string("true false").unwrap();
        assert_eq!(tokens, vec![N3Token::True, N3Token::False]);
    }

    #[test]
    fn test_directives() {
        let tokens = tokenize_string("@prefix @base").unwrap();
        assert_eq!(tokens, vec![N3Token::Prefix, N3Token::Base]);
    }

    #[test]
    fn test_type_shorthand() {
        let tokens = tokenize_string("a").unwrap();
        assert_eq!(tokens, vec![N3Token::A]);
    }

    #[test]
    fn test_variables() {
        let mut lexer = N3Lexer::new().with_variables();
        let mut buffer = StringBuffer::new("?x $y".to_string());
        let mut tokens = Vec::new();

        loop {
            match lexer
                .recognize_next_token(&mut buffer, &mut TextPosition::start())
                .unwrap()
            {
                Some(N3Token::Eof) => break,
                Some(token) => tokens.push(token),
                None => break,
            }
        }

        assert_eq!(
            tokens,
            vec![
                N3Token::Variable("x".to_string()),
                N3Token::Variable("y".to_string()),
            ]
        );
    }
}
