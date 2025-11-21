//! Tokenizer for Turtle RDF format.
//!
//! This module provides a simple tokenizer for parsing Turtle (Terse RDF Triple Language) documents.
//! The tokenizer breaks down Turtle syntax into discrete tokens that can be consumed by the parser.
//!
//! ## Features
//!
//! - Handles all standard Turtle syntax elements (IRIs, literals, prefixed names, blank nodes)
//! - Supports RDF 1.1 and RDF 1.2 features (quoted triples, directional language tags)
//! - Processes Unicode escape sequences in string literals
//! - Tracks line and column positions for error reporting
//! - Handles multiline string literals (triple-quoted strings)
//!
//! ## Usage
//!
//! The tokenizer is typically used internally by the `TurtleParser`:
//!
//! ```ignore
//! use crate::formats::turtle::tokenizer::TurtleTokenizer;
//!
//! let mut tokenizer = TurtleTokenizer::new(turtle_input);
//! while !tokenizer.is_at_end() {
//!     let token = tokenizer.consume_token()?;
//!     // Process token...
//! }
//! ```

use super::types::{Token, TokenKind};
use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};

/// Simple tokenizer for Turtle format.
///
/// The tokenizer maintains the current position in the input and provides methods
/// to consume tokens one at a time. It handles whitespace, comments, and all
/// Turtle syntax elements.
pub(crate) struct TurtleTokenizer {
    input: String,
    position: usize,
    line: usize,
    column: usize,
}

impl TurtleTokenizer {
    /// Creates a new tokenizer for the given input string.
    pub(crate) fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    /// Returns true if the tokenizer has reached the end of input.
    pub(crate) fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }

    /// Returns the current character without advancing the position.
    pub(crate) fn current_char(&self) -> Option<char> {
        // Use byte slicing for proper UTF-8 handling
        if self.position >= self.input.len() {
            None
        } else {
            self.input[self.position..].chars().next()
        }
    }

    /// Peeks at the next token without consuming it.
    /// Returns the token and the raw byte length to advance.
    pub(crate) fn peek_token(&mut self) -> TurtleResult<(Token, usize)> {
        self.skip_whitespace_and_comments();

        if self.is_at_end() {
            return Ok((
                Token {
                    kind: TokenKind::Eof,
                    position: TextPosition::new(self.line, self.column, self.position),
                },
                0,
            ));
        }

        let start_position = TextPosition::new(self.line, self.column, self.position);

        match self.current_char().unwrap() {
            '.' => Ok((
                Token {
                    kind: TokenKind::Dot,
                    position: start_position,
                },
                1,
            )),
            ';' => Ok((
                Token {
                    kind: TokenKind::Semicolon,
                    position: start_position,
                },
                1,
            )),
            ',' => Ok((
                Token {
                    kind: TokenKind::Comma,
                    position: start_position,
                },
                1,
            )),
            '[' => Ok((
                Token {
                    kind: TokenKind::LeftBracket,
                    position: start_position,
                },
                1,
            )),
            ']' => Ok((
                Token {
                    kind: TokenKind::RightBracket,
                    position: start_position,
                },
                1,
            )),
            '(' => Ok((
                Token {
                    kind: TokenKind::LeftParen,
                    position: start_position,
                },
                1,
            )),
            ')' => Ok((
                Token {
                    kind: TokenKind::RightParen,
                    position: start_position,
                },
                1,
            )),
            ':' => {
                // Check if this is an empty prefix name like :alice
                let remaining = &self.input[self.position + 1..];
                if let Some(first_char) = remaining.chars().next() {
                    if first_char.is_alphabetic() || first_char == '_' {
                        // This is an empty prefix name - read the local part
                        return self.read_empty_prefix_name(start_position);
                    }
                }
                // Otherwise, it's just a colon
                Ok((
                    Token {
                        kind: TokenKind::Colon,
                        position: start_position,
                    },
                    1,
                ))
            }
            '<' => {
                // Check for << (quoted triple start) - RDF 1.2
                if self.position + 1 < self.input.len() {
                    let next_char = self.input[self.position + 1..].chars().next();
                    if next_char == Some('<') {
                        return Ok((
                            Token {
                                kind: TokenKind::DoubleLessThan,
                                position: start_position,
                            },
                            2,
                        ));
                    }
                }
                // Otherwise, it's an IRI reference
                self.read_iri_ref(start_position)
            }
            '>' => {
                // Check for >> (quoted triple end) - RDF 1.2
                if self.position + 1 < self.input.len() {
                    let next_char = self.input[self.position + 1..].chars().next();
                    if next_char == Some('>') {
                        return Ok((
                            Token {
                                kind: TokenKind::DoubleGreaterThan,
                                position: start_position,
                            },
                            2,
                        ));
                    }
                }
                // Single > is an error (unexpected character)
                Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: "Unexpected character: '>'".to_string(),
                    position: start_position,
                }))
            }
            '"' => self.read_string_literal(start_position),
            '@' => self.read_at_keyword_or_language_tag(start_position),
            '_' => self.read_blank_node_label(start_position),
            '^' => self.read_datatype_annotation(start_position),
            'a' if self.is_standalone_a() => Ok((
                Token {
                    kind: TokenKind::A,
                    position: start_position,
                },
                1,
            )),
            '+' | '-' | '0'..='9' => self.read_numeric_literal(start_position),
            _ => {
                // Check for boolean keywords (true/false) or prefixed names
                let remaining = &self.input[self.position..];
                if remaining.starts_with("true") && self.is_keyword_boundary(4) {
                    Ok((
                        Token {
                            kind: TokenKind::Boolean(true),
                            position: start_position,
                        },
                        4,
                    ))
                } else if remaining.starts_with("false") && self.is_keyword_boundary(5) {
                    Ok((
                        Token {
                            kind: TokenKind::Boolean(false),
                            position: start_position,
                        },
                        5,
                    ))
                } else {
                    self.read_prefixed_name_or_prefix(start_position)
                }
            }
        }
    }

    /// Consumes and returns the next token.
    pub(crate) fn consume_token(&mut self) -> TurtleResult<Token> {
        let (token, raw_length) = self.peek_token()?;

        // Advance position by raw byte count
        // We need to advance character-by-character to update line/column correctly
        let target_position = self.position + raw_length;
        while self.position < target_position && !self.is_at_end() {
            self.advance();
        }

        Ok(token)
    }

    /// Advances the position by one character, updating line and column tracking.
    pub(crate) fn advance(&mut self) {
        if let Some(ch) = self.current_char() {
            self.position += ch.len_utf8();
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
    }

    /// Skips whitespace and comments in the input.
    pub(crate) fn skip_whitespace_and_comments(&mut self) {
        while let Some(ch) = self.current_char() {
            if ch.is_whitespace() {
                self.advance();
            } else if ch == '#' {
                // Skip comment line
                while let Some(ch) = self.current_char() {
                    self.advance();
                    if ch == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Reads an IRI reference (e.g., <http://example.org/>).
    pub(crate) fn read_iri_ref(&self, position: TextPosition) -> TurtleResult<(Token, usize)> {
        // Simplified IRI reading - just find the closing >
        let (content, raw_length) = if let Some(end) = self.input[self.position + 1..].find('>') {
            let content = self.input[self.position + 1..self.position + 1 + end].to_string();
            // raw_length is end (bytes to '>') + 2 (for '<' and '>')
            (content, end + 2)
        } else {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Unterminated IRI reference".to_string(),
                position,
            }));
        };

        Ok((
            Token {
                kind: TokenKind::IriRef(content),
                position,
            },
            raw_length,
        ))
    }

    /// Reads a string literal (single or double quoted).
    pub(crate) fn read_string_literal(
        &self,
        position: TextPosition,
    ) -> TurtleResult<(Token, usize)> {
        // Check for multiline string (""")
        let remaining = &self.input[self.position..];
        if remaining.starts_with("\"\"\"") {
            return self.read_multiline_string_literal(position);
        }

        // Regular string reading with escape sequence processing
        let mut end_pos = self.position + 1;
        let mut escaped = false;

        while end_pos < self.input.len() {
            let ch = self.input[end_pos..].chars().next().unwrap();
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                let raw_content = &self.input[self.position + 1..end_pos];
                let content = self.process_escape_sequences(raw_content)?;
                let raw_length = end_pos - self.position + 1; // +1 for closing quote
                return Ok((
                    Token {
                        kind: TokenKind::StringLiteral(content),
                        position,
                    },
                    raw_length,
                ));
            }
            end_pos += ch.len_utf8();
        }

        Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: "Unterminated string literal".to_string(),
            position,
        }))
    }

    /// Reads a multiline string literal (triple-quoted).
    pub(crate) fn read_multiline_string_literal(
        &self,
        position: TextPosition,
    ) -> TurtleResult<(Token, usize)> {
        // Skip opening """
        let mut end_pos = self.position + 3;

        while end_pos + 2 < self.input.len() {
            if &self.input[end_pos..end_pos + 3] == "\"\"\"" {
                let raw_content = &self.input[self.position + 3..end_pos];
                let content = self.process_escape_sequences(raw_content)?;
                let raw_length = end_pos - self.position + 3; // +3 for closing """
                return Ok((
                    Token {
                        kind: TokenKind::StringLiteral(content),
                        position,
                    },
                    raw_length,
                ));
            }
            end_pos += 1;
        }

        Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: "Unterminated multiline string literal".to_string(),
            position,
        }))
    }

    /// Processes escape sequences in a string (e.g., \n, \t, \uXXXX).
    pub(crate) fn process_escape_sequences(&self, input: &str) -> TurtleResult<String> {
        let mut result = String::with_capacity(input.len());
        let mut chars = input.chars();

        while let Some(ch) = chars.next() {
            if ch == '\\' {
                if let Some(next) = chars.next() {
                    match next {
                        't' => result.push('\t'),
                        'n' => result.push('\n'),
                        'r' => result.push('\r'),
                        '"' => result.push('"'),
                        '\'' => result.push('\''),
                        '\\' => result.push('\\'),
                        'u' => {
                            // \uXXXX - 4 hex digits
                            let hex: String = chars.by_ref().take(4).collect();
                            if hex.len() == 4 {
                                if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                    if let Some(unicode_char) = char::from_u32(code) {
                                        result.push(unicode_char);
                                    } else {
                                        return Err(TurtleParseError::syntax(
                                            TurtleSyntaxError::InvalidUnicode {
                                                codepoint: code,
                                                position: TextPosition::default(),
                                            },
                                        ));
                                    }
                                } else {
                                    return Err(TurtleParseError::syntax(
                                        TurtleSyntaxError::InvalidEscape {
                                            sequence: format!("u{hex}"),
                                            position: TextPosition::default(),
                                        },
                                    ));
                                }
                            } else {
                                return Err(TurtleParseError::syntax(
                                    TurtleSyntaxError::InvalidEscape {
                                        sequence: format!("u{hex}"),
                                        position: TextPosition::default(),
                                    },
                                ));
                            }
                        }
                        'U' => {
                            // \UXXXXXXXX - 8 hex digits
                            let hex: String = chars.by_ref().take(8).collect();
                            if hex.len() == 8 {
                                if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                    if let Some(unicode_char) = char::from_u32(code) {
                                        result.push(unicode_char);
                                    } else {
                                        return Err(TurtleParseError::syntax(
                                            TurtleSyntaxError::InvalidUnicode {
                                                codepoint: code,
                                                position: TextPosition::default(),
                                            },
                                        ));
                                    }
                                } else {
                                    return Err(TurtleParseError::syntax(
                                        TurtleSyntaxError::InvalidEscape {
                                            sequence: format!("U{hex}"),
                                            position: TextPosition::default(),
                                        },
                                    ));
                                }
                            } else {
                                return Err(TurtleParseError::syntax(
                                    TurtleSyntaxError::InvalidEscape {
                                        sequence: format!("U{hex}"),
                                        position: TextPosition::default(),
                                    },
                                ));
                            }
                        }
                        _ => {
                            // Unknown escape sequence - just include it as-is
                            result.push('\\');
                            result.push(next);
                        }
                    }
                } else {
                    result.push('\\');
                }
            } else {
                result.push(ch);
            }
        }

        Ok(result)
    }

    /// Reads an @ keyword or language tag (e.g., @prefix, @base, @en).
    pub(crate) fn read_at_keyword_or_language_tag(
        &self,
        position: TextPosition,
    ) -> TurtleResult<(Token, usize)> {
        let remaining = &self.input[self.position..];

        if remaining.starts_with("@prefix") {
            Ok((
                Token {
                    kind: TokenKind::PrefixKeyword,
                    position,
                },
                7, // "@prefix" length
            ))
        } else if remaining.starts_with("@base") {
            Ok((
                Token {
                    kind: TokenKind::BaseKeyword,
                    position,
                },
                5, // "@base" length
            ))
        } else {
            // Language tag (possibly with direction for RDF 1.2)
            let end = remaining[1..]
                .find(|c: char| !c.is_alphanumeric() && c != '-')
                .map(|i| i + 1)
                .unwrap_or(remaining.len());
            let tag_with_dir = &remaining[1..end];

            // Check for RDF 1.2 directional language tag: @lang--dir
            let (tag, direction, raw_length) =
                if let Some(double_dash_pos) = tag_with_dir.find("--") {
                    let language = tag_with_dir[..double_dash_pos].to_string();
                    let dir = &tag_with_dir[double_dash_pos + 2..];

                    // Validate direction is either "ltr" or "rtl"
                    if dir != "ltr" && dir != "rtl" {
                        return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                            message: format!("Invalid direction '{}'. Must be 'ltr' or 'rtl'", dir),
                            position,
                        }));
                    }

                    (language, Some(dir.to_string()), end)
                } else {
                    (tag_with_dir.to_string(), None, end)
                };

            Ok((
                Token {
                    kind: TokenKind::LanguageTag(tag, direction),
                    position,
                },
                raw_length,
            ))
        }
    }

    /// Reads a blank node label (e.g., _:b0).
    pub(crate) fn read_blank_node_label(
        &self,
        position: TextPosition,
    ) -> TurtleResult<(Token, usize)> {
        let remaining = &self.input[self.position..];

        if let Some(stripped) = remaining.strip_prefix("_:") {
            let end = stripped
                .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
                .unwrap_or(stripped.len());
            let label = stripped[..end].to_string();
            let raw_length = 2 + end; // "_:" + label
            Ok((
                Token {
                    kind: TokenKind::BlankNodeLabel(label),
                    position,
                },
                raw_length,
            ))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Invalid blank node label".to_string(),
                position,
            }))
        }
    }

    /// Reads a datatype annotation (^^).
    pub(crate) fn read_datatype_annotation(
        &self,
        position: TextPosition,
    ) -> TurtleResult<(Token, usize)> {
        let remaining = &self.input[self.position..];

        if remaining.starts_with("^^") {
            Ok((
                Token {
                    kind: TokenKind::DataTypeAnnotation,
                    position,
                },
                2, // "^^" length
            ))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Expected ^^ for datatype annotation".to_string(),
                position,
            }))
        }
    }

    /// Reads a prefixed name or prefix declaration (e.g., foaf:name, foaf).
    pub(crate) fn read_prefixed_name_or_prefix(
        &self,
        position: TextPosition,
    ) -> TurtleResult<(Token, usize)> {
        let remaining = &self.input[self.position..];

        // Find the end of the identifier
        let end = remaining
            .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != ':')
            .unwrap_or(remaining.len());

        let identifier = &remaining[..end];
        let raw_length = end;

        if let Some(colon_pos) = identifier.find(':') {
            // Prefixed name
            let prefix = identifier[..colon_pos].to_string();
            let local = identifier[colon_pos + 1..].to_string();
            Ok((
                Token {
                    kind: TokenKind::PrefixedName(prefix, local),
                    position,
                },
                raw_length,
            ))
        } else {
            // Just a prefix name (used in @prefix declarations)
            Ok((
                Token {
                    kind: TokenKind::PrefixName(identifier.to_string()),
                    position,
                },
                raw_length,
            ))
        }
    }

    /// Reads an empty prefix name (e.g., :alice).
    pub(crate) fn read_empty_prefix_name(
        &self,
        position: TextPosition,
    ) -> TurtleResult<(Token, usize)> {
        // Skip the initial colon
        let remaining = &self.input[self.position + 1..];

        // Find the end of the local part
        let end = remaining
            .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .unwrap_or(remaining.len());

        let local = &remaining[..end];
        let raw_length = end + 1; // +1 for the initial colon

        Ok((
            Token {
                kind: TokenKind::PrefixedName(String::new(), local.to_string()),
                position,
            },
            raw_length,
        ))
    }

    /// Checks if 'a' is a standalone keyword (not part of an identifier).
    pub(crate) fn is_standalone_a(&self) -> bool {
        // Check if 'a' is followed by whitespace or punctuation
        if let Some(next_char) = self.input.chars().nth(self.position + 1) {
            next_char.is_whitespace() || ".,;[]()".contains(next_char)
        } else {
            true // End of input
        }
    }

    /// Checks if a keyword is followed by a boundary character.
    pub(crate) fn is_keyword_boundary(&self, keyword_len: usize) -> bool {
        // Check if keyword is followed by whitespace, punctuation, or end of input
        if self.position + keyword_len >= self.input.len() {
            return true; // End of input
        }
        if let Some(next_char) = self.input[self.position + keyword_len..].chars().next() {
            next_char.is_whitespace() || ".,;[]()".contains(next_char)
        } else {
            true
        }
    }

    /// Reads a numeric literal (integer, decimal, or double).
    pub(crate) fn read_numeric_literal(
        &self,
        position: TextPosition,
    ) -> TurtleResult<(Token, usize)> {
        let remaining = &self.input[self.position..];
        let mut end = 0;
        let mut has_decimal_point = false;
        let mut has_exponent = false;

        // Handle optional sign
        if remaining.starts_with('+') || remaining.starts_with('-') {
            end += 1;
        }

        // Read digits before decimal point or exponent
        while end < remaining.len() {
            let ch = remaining.chars().nth(end).unwrap();
            if ch.is_ascii_digit() {
                end += 1;
            } else {
                break;
            }
        }

        // Check for decimal point
        if end < remaining.len() && remaining.chars().nth(end) == Some('.') {
            // Make sure it's not the end-of-statement dot
            if end + 1 < remaining.len() {
                let next_ch = remaining.chars().nth(end + 1).unwrap();
                if next_ch.is_ascii_digit() {
                    has_decimal_point = true;
                    end += 1; // Skip the decimal point

                    // Read fractional digits
                    while end < remaining.len() {
                        let ch = remaining.chars().nth(end).unwrap();
                        if ch.is_ascii_digit() {
                            end += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        // Check for exponent (e or E)
        if end < remaining.len() {
            let ch = remaining.chars().nth(end).unwrap();
            if ch == 'e' || ch == 'E' {
                has_exponent = true;
                end += 1;

                // Handle optional exponent sign
                if end < remaining.len() {
                    let sign_ch = remaining.chars().nth(end).unwrap();
                    if sign_ch == '+' || sign_ch == '-' {
                        end += 1;
                    }
                }

                // Read exponent digits
                let exponent_start = end;
                while end < remaining.len() {
                    let ch = remaining.chars().nth(end).unwrap();
                    if ch.is_ascii_digit() {
                        end += 1;
                    } else {
                        break;
                    }
                }

                // Ensure we have at least one digit in the exponent
                if end == exponent_start {
                    return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                        message: "Invalid numeric literal: exponent requires digits".to_string(),
                        position,
                    }));
                }
            }
        }

        if end == 0 || (end == 1 && (remaining.starts_with('+') || remaining.starts_with('-'))) {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Invalid numeric literal: no digits found".to_string(),
                position,
            }));
        }

        let literal_str = remaining[..end].to_string();
        let token_kind = if has_exponent {
            TokenKind::Double(literal_str)
        } else if has_decimal_point {
            TokenKind::Decimal(literal_str)
        } else {
            TokenKind::Integer(literal_str)
        };

        Ok((
            Token {
                kind: token_kind,
                position,
            },
            end,
        ))
    }
}
