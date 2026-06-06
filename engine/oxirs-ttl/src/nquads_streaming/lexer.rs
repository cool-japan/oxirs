//! N-Quads lexer: tokenize a single N-Quads line into [`Token`]s.
//!
//! The lexer handles:
//! - IRI references: `<...>`
//! - Blank node labels: `_:label`
//! - String literals: `"value"`, `"value"@lang`, `"value"^^<datatype>`
//! - Escape sequences: `\n`, `\t`, `\r`, `\\`, `\"`, `\uXXXX`, `\UXXXXXXXX`
//! - Structural tokens: `.` (end-of-statement), `^^` (type annotation), `@` (lang tag)

use crate::nquads_streaming::NQuadsParseError;

/// A single lexical token from an N-Quads line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// `<iri>` — an IRI reference with angle brackets stripped.
    IriRef(String),
    /// `_:label` — a blank node label (the `_:` prefix is stripped).
    BlankNodeLabel(String),
    /// A string literal with optional language tag or datatype.
    StringLiteral {
        /// The lexical form of the string literal.
        value: String,
        /// Optional BCP 47 language tag (e.g., `"en"`, `"fr-CA"`).
        lang: Option<String>,
        /// Optional datatype IRI (already stripped of angle brackets).
        datatype: Option<String>,
    },
    /// `.` — statement terminator.
    Dot,
    /// `^^` — datatype annotation separator.
    Caret,
    /// `@` — language tag prefix.
    At,
}

/// N-Quads line tokenizer.
pub struct NQuadsLexer;

impl NQuadsLexer {
    /// Tokenize an N-Quads line into a `Vec<Token>`.
    ///
    /// - Returns `Ok(vec![])` for blank or comment-only lines.
    /// - Returns an error if the line contains invalid syntax.
    pub fn tokenize_line(line: &str, line_num: usize) -> Result<Vec<Token>, NQuadsParseError> {
        let mut tokens = Vec::new();
        let bytes = line.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            // Skip whitespace
            if bytes[i].is_ascii_whitespace() {
                i += 1;
                continue;
            }

            // Comment
            if bytes[i] == b'#' {
                break;
            }

            // IRI reference: <...>
            if bytes[i] == b'<' {
                let (iri, consumed) = Self::read_iri(line, i, line_num)?;
                tokens.push(Token::IriRef(iri));
                i += consumed;
                continue;
            }

            // Blank node: _:label
            if i + 1 < bytes.len() && bytes[i] == b'_' && bytes[i + 1] == b':' {
                let (label, consumed) = Self::read_blank_node(line, i, line_num)?;
                tokens.push(Token::BlankNodeLabel(label));
                i += consumed;
                continue;
            }

            // String literal: "..."
            if bytes[i] == b'"' {
                let (tok, consumed) = Self::read_string_literal(line, i, line_num)?;
                tokens.push(tok);
                i += consumed;
                continue;
            }

            // Dot statement terminator
            if bytes[i] == b'.' {
                tokens.push(Token::Dot);
                i += 1;
                continue;
            }

            // Caret pair: ^^
            if i + 1 < bytes.len() && bytes[i] == b'^' && bytes[i + 1] == b'^' {
                tokens.push(Token::Caret);
                i += 2;
                continue;
            }

            // At sign: @
            if bytes[i] == b'@' {
                tokens.push(Token::At);
                i += 1;
                continue;
            }

            return Err(NQuadsParseError::InvalidLine {
                line: line_num,
                message: format!(
                    "Unexpected character '{}' at position {}",
                    bytes[i] as char, i
                ),
            });
        }

        Ok(tokens)
    }

    /// Read an IRI reference starting at `start` (which must be `<`).
    ///
    /// Returns `(iri_string, bytes_consumed)`.
    fn read_iri(
        line: &str,
        start: usize,
        line_num: usize,
    ) -> Result<(String, usize), NQuadsParseError> {
        let bytes = line.as_bytes();
        debug_assert_eq!(bytes[start], b'<');
        let mut i = start + 1;
        let mut iri = String::new();

        loop {
            if i >= bytes.len() {
                return Err(NQuadsParseError::InvalidIri {
                    line: line_num,
                    iri: iri.clone(),
                });
            }

            match bytes[i] {
                b'>' => {
                    return Ok((iri, i - start + 1));
                }
                b'\\' => {
                    let (ch, skip) = Self::read_escape(line, i, line_num)?;
                    iri.push(ch);
                    i += skip;
                }
                b => {
                    // Reject disallowed characters per N-Quads spec
                    // (control chars, space, `"`, `{`, `}`, `|`, `^`, `` ` ``)
                    if b < 0x21 || matches!(b, b'"' | b'{' | b'}' | b'|' | b'^' | b'`') {
                        return Err(NQuadsParseError::InvalidIri {
                            line: line_num,
                            iri: iri.clone(),
                        });
                    }
                    // Decode multi-byte UTF-8 character
                    let (ch, char_bytes) = Self::decode_utf8_char(line, i, line_num)?;
                    iri.push(ch);
                    i += char_bytes;
                }
            }
        }
    }

    /// Read a blank node label starting at `start` (which must be `_`).
    ///
    /// Returns `(label, bytes_consumed)` where label excludes the `_:` prefix.
    fn read_blank_node(
        line: &str,
        start: usize,
        line_num: usize,
    ) -> Result<(String, usize), NQuadsParseError> {
        let bytes = line.as_bytes();
        debug_assert_eq!(bytes[start], b'_');
        debug_assert_eq!(bytes[start + 1], b':');
        let mut i = start + 2;
        let mut label = String::new();

        while i < bytes.len() {
            let b = bytes[i];
            if b.is_ascii_whitespace() || b == b'.' || b == b',' || b == b';' {
                break;
            }
            let (ch, char_bytes) = Self::decode_utf8_char(line, i, line_num)?;
            label.push(ch);
            i += char_bytes;
        }

        if label.is_empty() {
            return Err(NQuadsParseError::InvalidBlankNode {
                line: line_num,
                name: String::new(),
            });
        }

        Ok((label, i - start))
    }

    /// Read a string literal starting at `start` (which must be `"`).
    ///
    /// Returns `(Token, bytes_consumed)`.
    fn read_string_literal(
        line: &str,
        start: usize,
        line_num: usize,
    ) -> Result<(Token, usize), NQuadsParseError> {
        let bytes = line.as_bytes();
        debug_assert_eq!(bytes[start], b'"');
        let mut i = start + 1;
        let mut value = String::new();

        // Read the literal value
        loop {
            if i >= bytes.len() {
                return Err(NQuadsParseError::InvalidLiteral {
                    line: line_num,
                    message: "Unterminated string literal".to_string(),
                });
            }
            match bytes[i] {
                b'"' => {
                    i += 1;
                    break;
                }
                b'\\' => {
                    let (ch, skip) = Self::read_escape(line, i, line_num)?;
                    value.push(ch);
                    i += skip;
                }
                _ => {
                    let (ch, char_bytes) = Self::decode_utf8_char(line, i, line_num)?;
                    value.push(ch);
                    i += char_bytes;
                }
            }
        }

        // Skip whitespace after closing quote
        let after_close = i;

        // Check for @lang or ^^<datatype>
        // Peek at non-whitespace
        let mut j = after_close;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }

        if j < bytes.len() && bytes[j] == b'@' {
            // Language tag
            j += 1; // skip '@'
            let lang_start = j;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'-') {
                j += 1;
            }
            let lang = &line[lang_start..j];
            if lang.is_empty() {
                return Err(NQuadsParseError::InvalidLiteral {
                    line: line_num,
                    message: "Empty language tag".to_string(),
                });
            }
            return Ok((
                Token::StringLiteral {
                    value,
                    lang: Some(lang.to_string()),
                    datatype: None,
                },
                j - start,
            ));
        }

        if j + 1 < bytes.len() && bytes[j] == b'^' && bytes[j + 1] == b'^' {
            // Datatype IRI
            j += 2; // skip '^^'
                    // Skip whitespace
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j >= bytes.len() || bytes[j] != b'<' {
                return Err(NQuadsParseError::InvalidLiteral {
                    line: line_num,
                    message: "Expected '<' after '^^'".to_string(),
                });
            }
            let (datatype_iri, iri_len) = Self::read_iri(line, j, line_num)?;
            return Ok((
                Token::StringLiteral {
                    value,
                    lang: None,
                    datatype: Some(datatype_iri),
                },
                j + iri_len - start,
            ));
        }

        // Plain literal (no annotation)
        Ok((
            Token::StringLiteral {
                value,
                lang: None,
                datatype: None,
            },
            after_close - start,
        ))
    }

    /// Read a single escape sequence starting at `start` (which must be `\`).
    ///
    /// Returns `(char, bytes_consumed_including_backslash)`.
    fn read_escape(
        line: &str,
        start: usize,
        line_num: usize,
    ) -> Result<(char, usize), NQuadsParseError> {
        let bytes = line.as_bytes();
        debug_assert_eq!(bytes[start], b'\\');
        if start + 1 >= bytes.len() {
            return Err(NQuadsParseError::InvalidLiteral {
                line: line_num,
                message: "Incomplete escape sequence".to_string(),
            });
        }
        match bytes[start + 1] {
            b'n' => Ok(('\n', 2)),
            b't' => Ok(('\t', 2)),
            b'r' => Ok(('\r', 2)),
            b'\\' => Ok(('\\', 2)),
            b'"' => Ok(('"', 2)),
            b'\'' => Ok(('\'', 2)),
            b'u' => {
                // \uXXXX – exactly 4 hex digits
                if start + 5 >= bytes.len() {
                    return Err(NQuadsParseError::InvalidLiteral {
                        line: line_num,
                        message: "\\uXXXX requires 4 hex digits".to_string(),
                    });
                }
                let hex = &line[start + 2..start + 6];
                let code_point =
                    u32::from_str_radix(hex, 16).map_err(|_| NQuadsParseError::InvalidLiteral {
                        line: line_num,
                        message: format!("Invalid \\u escape: \\u{}", hex),
                    })?;
                let ch =
                    char::from_u32(code_point).ok_or_else(|| NQuadsParseError::InvalidLiteral {
                        line: line_num,
                        message: format!("Invalid Unicode code point U+{:04X}", code_point),
                    })?;
                Ok((ch, 6))
            }
            b'U' => {
                // \UXXXXXXXX – exactly 8 hex digits
                if start + 9 >= bytes.len() {
                    return Err(NQuadsParseError::InvalidLiteral {
                        line: line_num,
                        message: "\\UXXXXXXXX requires 8 hex digits".to_string(),
                    });
                }
                let hex = &line[start + 2..start + 10];
                let code_point =
                    u32::from_str_radix(hex, 16).map_err(|_| NQuadsParseError::InvalidLiteral {
                        line: line_num,
                        message: format!("Invalid \\U escape: \\U{}", hex),
                    })?;
                let ch =
                    char::from_u32(code_point).ok_or_else(|| NQuadsParseError::InvalidLiteral {
                        line: line_num,
                        message: format!("Invalid Unicode code point U+{:08X}", code_point),
                    })?;
                Ok((ch, 10))
            }
            other => Err(NQuadsParseError::InvalidLiteral {
                line: line_num,
                message: format!("Unknown escape sequence: \\{}", other as char),
            }),
        }
    }

    /// Decode a UTF-8 character at byte position `pos` in `line`.
    ///
    /// Returns `(char, byte_length)`.
    fn decode_utf8_char(
        line: &str,
        pos: usize,
        line_num: usize,
    ) -> Result<(char, usize), NQuadsParseError> {
        let slice = &line[pos..];
        let ch = slice
            .chars()
            .next()
            .ok_or_else(|| NQuadsParseError::InvalidLine {
                line: line_num,
                message: format!("Invalid UTF-8 at byte position {}", pos),
            })?;
        Ok((ch, ch.len_utf8()))
    }
}
