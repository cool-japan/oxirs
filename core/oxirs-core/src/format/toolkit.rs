//! Generic parsing toolkit for RDF formats
//!
//! This module provides reusable parsing infrastructure including lexers, parsers,
//! and error handling utilities adapted from OxiGraph's parsing toolkit.

use super::error::{ParseResult, RdfParseError, TextPosition};
use std::io::BufRead;

/// A generic lexer that tokenizes input streams
#[allow(dead_code)]
pub struct Lexer<B, TR> {
    buffer: B,
    tokenizer: TR,
    position: TextPosition,
    current_char: Option<char>,
    peek_char: Option<char>,
}

/// Trait for tokenizing rules
pub trait TokenRecognizer {
    type Token;

    /// Recognize the next token from the current position
    fn recognize_next_token(
        &mut self,
        buffer: &mut dyn BufferProvider,
        position: &mut TextPosition,
    ) -> ParseResult<Option<Self::Token>>;
}

/// Trait for parsing rules that build AST nodes
pub trait RuleRecognizer<Node> {
    /// Recognize the next rule/node from token stream
    fn recognize_next_node<Token>(
        &mut self,
        parser: &mut Parser<Token>,
    ) -> ParseResult<Option<Node>>;
}

/// Buffer provider trait for reading characters
pub trait BufferProvider {
    /// Get the current character without advancing
    fn current(&self) -> Option<char>;

    /// Get the next character without advancing
    fn peek(&self) -> Option<char>;

    /// Advance to the next character and return it
    fn advance(&mut self) -> Option<char>;

    /// Get current position for error reporting
    fn position(&self) -> &TextPosition;

    /// Update position tracking
    fn update_position(&mut self, ch: char);
}

/// String buffer implementation
pub struct StringBuffer {
    content: String,
    position: TextPosition,
    current: Option<char>,
    peek: Option<char>,
    char_position: usize, // Current position in chars
}

impl StringBuffer {
    pub fn new(content: String) -> Self {
        let mut buffer = Self {
            content,
            position: TextPosition::start(),
            current: None,
            peek: None,
            char_position: 0,
        };
        // Initialize current and peek
        buffer.current = buffer.get_char_at(0);
        buffer.peek = buffer.get_char_at(1);
        buffer
    }

    fn get_char_at(&self, index: usize) -> Option<char> {
        self.content.chars().nth(index)
    }
}

impl BufferProvider for StringBuffer {
    fn current(&self) -> Option<char> {
        self.current
    }

    fn peek(&self) -> Option<char> {
        self.peek
    }

    fn advance(&mut self) -> Option<char> {
        if let Some(ch) = self.current {
            self.update_position(ch);
        }

        self.current = self.peek;
        self.char_position += 1;
        self.peek = self.get_char_at(self.char_position + 1);
        self.current
    }

    fn position(&self) -> &TextPosition {
        &self.position
    }

    fn update_position(&mut self, ch: char) {
        match ch {
            '\n' => {
                self.position.line += 1;
                self.position.column = 1;
                self.position.offset += 1;
            }
            '\r' => {
                // Handle Windows line endings
                if self.peek == Some('\n') {
                    // Don't increment position for \r if followed by \n
                } else {
                    self.position.line += 1;
                    self.position.column = 1;
                }
                self.position.offset += 1;
            }
            _ => {
                self.position.column += 1;
                self.position.offset += 1;
            }
        }
    }
}

/// Reader buffer implementation for streaming
pub struct ReaderBuffer<R: BufRead> {
    reader: R,
    position: TextPosition,
    current: Option<char>,
    peek: Option<char>,
    char_buffer: Vec<char>,
    buffer_pos: usize,
}

impl<R: BufRead> ReaderBuffer<R> {
    pub fn new(reader: R) -> ParseResult<Self> {
        let mut buffer = Self {
            reader,
            position: TextPosition::start(),
            current: None,
            peek: None,
            char_buffer: Vec::new(),
            buffer_pos: 0,
        };

        buffer.fill_buffer()?;
        buffer.advance(); // Load first character
        Ok(buffer)
    }

    fn fill_buffer(&mut self) -> ParseResult<()> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => Ok(()), // EOF
            Ok(_) => {
                self.char_buffer.extend(line.chars());
                Ok(())
            }
            Err(e) => Err(RdfParseError::Io(e)),
        }
    }

    #[allow(dead_code)]
    fn ensure_chars_available(&mut self) -> ParseResult<()> {
        if self.buffer_pos + 1 >= self.char_buffer.len() {
            self.fill_buffer()?;
        }
        Ok(())
    }
}

impl<R: BufRead> BufferProvider for ReaderBuffer<R> {
    fn current(&self) -> Option<char> {
        self.current
    }

    fn peek(&self) -> Option<char> {
        self.peek
    }

    fn advance(&mut self) -> Option<char> {
        if let Some(ch) = self.current {
            self.update_position(ch);
        }

        self.current = self.peek;

        // Try to get next peek character
        self.buffer_pos += 1;
        if self.buffer_pos < self.char_buffer.len() {
            self.peek = Some(self.char_buffer[self.buffer_pos]);
        } else {
            // Try to read more
            if self.fill_buffer().is_ok() && self.buffer_pos < self.char_buffer.len() {
                self.peek = Some(self.char_buffer[self.buffer_pos]);
            } else {
                self.peek = None;
            }
        }

        self.current
    }

    fn position(&self) -> &TextPosition {
        &self.position
    }

    fn update_position(&mut self, ch: char) {
        match ch {
            '\n' => {
                self.position.line += 1;
                self.position.column = 1;
                self.position.offset += 1;
            }
            '\r' => {
                if self.peek == Some('\n') {
                    // Don't increment position for \r if followed by \n
                } else {
                    self.position.line += 1;
                    self.position.column = 1;
                }
                self.position.offset += 1;
            }
            _ => {
                self.position.column += 1;
                self.position.offset += 1;
            }
        }
    }
}

impl<B: BufferProvider, TR> Lexer<B, TR> {
    pub fn new(buffer: B, tokenizer: TR) -> Self {
        Self {
            buffer,
            tokenizer,
            position: TextPosition::start(),
            current_char: None,
            peek_char: None,
        }
    }
}

impl<B: BufferProvider, TR: TokenRecognizer> Lexer<B, TR> {
    /// Get the next token from the input
    pub fn next_token(&mut self) -> ParseResult<Option<TR::Token>> {
        self.tokenizer
            .recognize_next_token(&mut self.buffer, &mut self.position)
    }

    /// Get current position for error reporting
    pub fn position(&self) -> &TextPosition {
        self.buffer.position()
    }
}

/// Generic parser combining lexer with grammar rules
pub struct Parser<Token> {
    tokens: Vec<Token>,
    position: usize,
}

impl<Token> Parser<Token> {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    /// Peek at current token without consuming
    pub fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    /// Advance and return current token
    pub fn next_token(&mut self) -> Option<&Token> {
        if self.position < self.tokens.len() {
            let token = &self.tokens[self.position];
            self.position += 1;
            Some(token)
        } else {
            None
        }
    }

    /// Check if we're at the end of input
    pub fn is_at_end(&self) -> bool {
        self.position >= self.tokens.len()
    }

    /// Get current position in token stream
    pub fn token_position(&self) -> usize {
        self.position
    }

    /// Reset to a previous position (for backtracking)
    pub fn reset_to(&mut self, pos: usize) {
        self.position = pos.min(self.tokens.len());
    }
}

/// Utility functions for character classification
pub mod char_utils {
    /// Check if character is whitespace in Turtle/N3
    pub fn is_whitespace(ch: char) -> bool {
        matches!(ch, ' ' | '\t' | '\n' | '\r')
    }

    /// Check if character can start an IRI
    pub fn is_iri_start(ch: char) -> bool {
        ch == '<'
    }

    /// Check if character can be in an IRI
    pub fn is_iri_char(ch: char) -> bool {
        !matches!(
            ch,
            '<' | '>' | '"' | '{' | '}' | '|' | '^' | '`' | '\\' | '\x00'..='\x20'
        )
    }

    /// Check if character can start a blank node
    pub fn is_blank_node_start(ch: char) -> bool {
        ch == '_'
    }

    /// Check if character can start a prefix name
    pub fn is_pn_chars_base(ch: char) -> bool {
        matches!(ch, 'A'..='Z' | 'a'..='z' | '\u{00C0}'..='\u{00D6}' | '\u{00D8}'..='\u{00F6}' | '\u{00F8}'..='\u{02FF}' | '\u{0370}'..='\u{037D}' | '\u{037F}'..='\u{1FFF}' | '\u{200C}'..='\u{200D}' | '\u{2070}'..='\u{218F}' | '\u{2C00}'..='\u{2FEF}' | '\u{3001}'..='\u{D7FF}' | '\u{F900}'..='\u{FDCF}' | '\u{FDF0}'..='\u{FFFD}')
    }

    /// Check if character can be in a prefix name (after first character)
    pub fn is_pn_chars(ch: char) -> bool {
        is_pn_chars_base(ch)
            || matches!(ch, '-' | '0'..='9' | '\u{00B7}' | '\u{0300}'..='\u{036F}' | '\u{203F}'..='\u{2040}')
    }

    /// Check if character can start a numeric literal
    pub fn is_numeric_start(ch: char) -> bool {
        matches!(ch, '0'..='9' | '+' | '-' | '.')
    }

    /// Check if character is a digit
    pub fn is_digit(ch: char) -> bool {
        ch.is_ascii_digit()
    }

    /// Check if character is hexadecimal
    pub fn is_hex_digit(ch: char) -> bool {
        ch.is_ascii_hexdigit()
    }
}

/// Utility functions for string processing
pub mod string_utils {
    use super::ParseResult;
    use crate::format::error::{RdfParseError, RdfSyntaxError, TextPosition};

    /// Unescape string literal with Turtle escape sequences
    pub fn unescape_string(input: &str, position: &TextPosition) -> ParseResult<String> {
        let mut result = String::new();
        let mut chars = input.chars();

        while let Some(ch) = chars.next() {
            if ch == '\\' {
                match chars.next() {
                    Some('t') => result.push('\t'),
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('b') => result.push('\u{0008}'),
                    Some('f') => result.push('\u{000C}'),
                    Some('"') => result.push('"'),
                    Some('\'') => result.push('\''),
                    Some('\\') => result.push('\\'),
                    Some('u') => {
                        // Unicode escape \uXXXX
                        let mut unicode_chars = String::new();
                        for _ in 0..4 {
                            match chars.next() {
                                Some(c) if c.is_ascii_hexdigit() => unicode_chars.push(c),
                                _ => {
                                    return Err(RdfParseError::Syntax(
                                        RdfSyntaxError::with_position(
                                            "Invalid Unicode escape sequence".to_string(),
                                            *position,
                                        ),
                                    ))
                                }
                            }
                        }
                        let code_point = u32::from_str_radix(&unicode_chars, 16).map_err(|_| {
                            RdfParseError::Syntax(RdfSyntaxError::with_position(
                                "Invalid Unicode code point".to_string(),
                                *position,
                            ))
                        })?;
                        match char::from_u32(code_point) {
                            Some(unicode_char) => result.push(unicode_char),
                            None => {
                                return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                                    "Invalid Unicode code point".to_string(),
                                    *position,
                                )))
                            }
                        }
                    }
                    Some('U') => {
                        // Unicode escape \UXXXXXXXX
                        let mut unicode_chars = String::new();
                        for _ in 0..8 {
                            match chars.next() {
                                Some(c) if c.is_ascii_hexdigit() => unicode_chars.push(c),
                                _ => {
                                    return Err(RdfParseError::Syntax(
                                        RdfSyntaxError::with_position(
                                            "Invalid Unicode escape sequence".to_string(),
                                            *position,
                                        ),
                                    ))
                                }
                            }
                        }
                        let code_point = u32::from_str_radix(&unicode_chars, 16).map_err(|_| {
                            RdfParseError::Syntax(RdfSyntaxError::with_position(
                                "Invalid Unicode code point".to_string(),
                                *position,
                            ))
                        })?;
                        match char::from_u32(code_point) {
                            Some(unicode_char) => result.push(unicode_char),
                            None => {
                                return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                                    "Invalid Unicode code point".to_string(),
                                    *position,
                                )))
                            }
                        }
                    }
                    Some(other) => {
                        return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                            format!("Invalid escape sequence: \\{other}"),
                            *position,
                        )));
                    }
                    None => {
                        return Err(RdfParseError::Syntax(RdfSyntaxError::with_position(
                            "Incomplete escape sequence".to_string(),
                            *position,
                        )));
                    }
                }
            } else {
                result.push(ch);
            }
        }

        Ok(result)
    }

    /// Escape string for Turtle output
    pub fn escape_string(input: &str) -> String {
        let mut result = String::new();
        for ch in input.chars() {
            match ch {
                '\t' => result.push_str("\\t"),
                '\n' => result.push_str("\\n"),
                '\r' => result.push_str("\\r"),
                '\u{0008}' => result.push_str("\\b"),
                '\u{000C}' => result.push_str("\\f"),
                '"' => result.push_str("\\\""),
                '\\' => result.push_str("\\\\"),
                c if c.is_control() => {
                    if (c as u32) <= 0xFFFF {
                        result.push_str(&format!("\\u{:04X}", c as u32));
                    } else {
                        result.push_str(&format!("\\U{:08X}", c as u32));
                    }
                }
                c => result.push(c),
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::char_utils::*;
    use super::string_utils::*;
    use super::*;

    #[test]
    fn test_string_buffer() {
        let mut buffer = StringBuffer::new("hello\nworld".to_string());

        assert_eq!(buffer.current(), Some('h'));
        assert_eq!(buffer.peek(), Some('e'));

        buffer.advance();
        assert_eq!(buffer.current(), Some('e'));
        assert_eq!(buffer.position().column, 2);

        // Advance to newline
        for _ in 0..4 {
            buffer.advance();
        }
        assert_eq!(buffer.current(), Some('\n'));
        assert_eq!(buffer.position().line, 1);
        assert_eq!(buffer.position().column, 6);

        buffer.advance();
        assert_eq!(buffer.current(), Some('w'));
        assert_eq!(buffer.position().line, 2);
        assert_eq!(buffer.position().column, 1);
    }

    #[test]
    fn test_char_classification() {
        assert!(is_whitespace(' '));
        assert!(is_whitespace('\t'));
        assert!(is_whitespace('\n'));
        assert!(!is_whitespace('a'));

        assert!(is_iri_start('<'));
        assert!(!is_iri_start('a'));

        assert!(is_pn_chars_base('A'));
        assert!(is_pn_chars_base('z'));
        assert!(!is_pn_chars_base('1'));

        assert!(is_pn_chars('A'));
        assert!(is_pn_chars('1'));
        assert!(is_pn_chars('-'));

        assert!(is_numeric_start('1'));
        assert!(is_numeric_start('+'));
        assert!(is_numeric_start('.'));
        assert!(!is_numeric_start('a'));
    }

    #[test]
    fn test_string_escaping() {
        let position = TextPosition::start();

        // Test basic escapes
        assert_eq!(
            unescape_string("hello\\nworld", &position).unwrap(),
            "hello\nworld"
        );
        assert_eq!(
            unescape_string("say \\\"hello\\\"", &position).unwrap(),
            "say \"hello\""
        );

        // Test Unicode escapes
        assert_eq!(unescape_string("\\u0041", &position).unwrap(), "A");
        assert_eq!(unescape_string("\\U00000041", &position).unwrap(), "A");

        // Test escape string
        assert_eq!(escape_string("hello\nworld"), "hello\\nworld");
        assert_eq!(escape_string("say \"hello\""), "say \\\"hello\\\"");
    }
}
