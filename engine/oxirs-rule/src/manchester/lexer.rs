//! Byte-level tokenizer for OWL Manchester Syntax.
//!
//! Converts a Manchester Syntax string into a flat sequence of [`Token`]s
//! together with their byte offsets, enabling precise error reporting.
//!
//! Keywords are **case-sensitive**: `and`, `or`, `not`, `some`, `only`,
//! `min`, `max`, `exactly`, `value`.

use super::ManchesterError;

/// A single lexical token produced by the Manchester Syntax tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Identifier: alphanumeric characters, `_`, or `:`.
    /// Colon enables prefixed names such as `owl:Thing`.
    Ident(String),
    /// Keyword `and`
    And,
    /// Keyword `or`
    Or,
    /// Keyword `not`
    Not,
    /// Keyword `some`
    Some,
    /// Keyword `only`
    Only,
    /// Keyword `min`
    Min,
    /// Keyword `max`
    Max,
    /// Keyword `exactly`
    Exactly,
    /// Keyword `value`
    Value,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// Non-negative integer literal
    Number(u32),
    /// Sentinel: end of token stream
    Eof,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Ident(s) => write!(f, "identifier `{s}`"),
            Token::And => write!(f, "`and`"),
            Token::Or => write!(f, "`or`"),
            Token::Not => write!(f, "`not`"),
            Token::Some => write!(f, "`some`"),
            Token::Only => write!(f, "`only`"),
            Token::Min => write!(f, "`min`"),
            Token::Max => write!(f, "`max`"),
            Token::Exactly => write!(f, "`exactly`"),
            Token::Value => write!(f, "`value`"),
            Token::LBrace => write!(f, "`{{`"),
            Token::RBrace => write!(f, "`}}`"),
            Token::LParen => write!(f, "`(`"),
            Token::RParen => write!(f, "`)`"),
            Token::Number(n) => write!(f, "number `{n}`"),
            Token::Eof => write!(f, "end of input"),
        }
    }
}

/// Classify a raw identifier string as either a keyword token or a generic
/// [`Token::Ident`].
fn classify_ident(raw: &str) -> Token {
    match raw {
        "and" => Token::And,
        "or" => Token::Or,
        "not" => Token::Not,
        "some" => Token::Some,
        "only" => Token::Only,
        "min" => Token::Min,
        "max" => Token::Max,
        "exactly" => Token::Exactly,
        "value" => Token::Value,
        _ => Token::Ident(raw.to_string()),
    }
}

/// Returns `true` for byte values that are valid **identifier body** characters.
///
/// Identifiers consist of ASCII alphanumerics, underscores, or colons.
/// Colons allow prefixed names (`owl:Thing`, `ex:MyClass`).
#[inline]
fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b':'
}

/// Returns `true` for byte values that begin a new identifier token.
#[inline]
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

/// Tokenize a Manchester Syntax string.
///
/// Returns a [`Vec`] of `(Token, byte_offset)` pairs.  The byte offset is the
/// position of the **first byte** of the token in the original input.  An
/// [`Token::Eof`] sentinel is appended at the end.
///
/// # Errors
///
/// Returns [`ManchesterError::LexError`] if an unexpected character is
/// encountered, or if a decimal literal overflows `u32`.
pub fn tokenize(input: &str) -> Result<Vec<(Token, usize)>, ManchesterError> {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut tokens: Vec<(Token, usize)> = Vec::with_capacity(len / 3 + 8);

    while pos < len {
        // Skip ASCII whitespace
        if bytes[pos].is_ascii_whitespace() {
            pos += 1;
            continue;
        }

        let start = pos;
        let b = bytes[pos];

        if is_ident_start(b) {
            // Consume the full identifier (including ':' for prefixed names)
            while pos < len && is_ident_char(bytes[pos]) {
                pos += 1;
            }
            let raw = &input[start..pos];
            tokens.push((classify_ident(raw), start));
        } else if b.is_ascii_digit() {
            while pos < len && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            let raw = &input[start..pos];
            let n: u32 = raw.parse().map_err(|_| ManchesterError::LexError {
                pos: start,
                msg: format!("integer literal `{raw}` overflows u32"),
            })?;
            tokens.push((Token::Number(n), start));
        } else {
            pos += 1; // consume the single-byte punctuation (or emit error)
            match b {
                b'{' => tokens.push((Token::LBrace, start)),
                b'}' => tokens.push((Token::RBrace, start)),
                b'(' => tokens.push((Token::LParen, start)),
                b')' => tokens.push((Token::RParen, start)),
                other => {
                    return Err(ManchesterError::LexError {
                        pos: start,
                        msg: format!("unexpected character `{}`", other as char),
                    });
                }
            }
        }
    }

    tokens.push((Token::Eof, len));
    Ok(tokens)
}
