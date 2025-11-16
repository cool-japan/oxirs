//! Zero-copy parsing utilities for efficient RDF processing
//!
//! This module provides zero-copy parsing techniques to minimize string allocations
//! during RDF parsing. By using `Cow<str>` and direct slice references, we can
//! significantly reduce memory allocations for frequently-occurring patterns.

use std::borrow::Cow;
use std::collections::HashMap;

/// Zero-copy IRI reference parser
///
/// Parses IRI references from input slices without allocating when possible.
/// Uses `Cow<str>` to return borrowed slices when no decoding is needed,
/// and only allocates when escape sequences need to be decoded.
///
/// # Example
///
/// ```
/// use oxirs_ttl::toolkit::ZeroCopyIriParser;
///
/// let mut parser = ZeroCopyIriParser::new();
///
/// // Simple IRI - no allocation (borrowed)
/// let iri = parser.parse_iri_ref(b"<http://example.org/>").unwrap();
/// // Verify it's borrowed
/// assert!(matches!(iri, std::borrow::Cow::Borrowed(_)));
///
/// // IRI with escape - allocates (owned)
/// let iri2 = parser.parse_iri_ref(b"<http://example.org/sp%20ace>").unwrap();
/// assert!(matches!(iri2, std::borrow::Cow::Owned(_)));
/// ```
#[derive(Debug, Clone)]
pub struct ZeroCopyIriParser {
    /// Cache of decoded IRIs to avoid re-decoding common patterns
    decode_cache: HashMap<Vec<u8>, String>,
}

impl ZeroCopyIriParser {
    /// Create a new zero-copy IRI parser
    pub fn new() -> Self {
        Self {
            decode_cache: HashMap::with_capacity(256),
        }
    }

    /// Create with pre-allocated cache capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            decode_cache: HashMap::with_capacity(capacity),
        }
    }

    /// Parse an IRI reference from bytes, returning a zero-copy result when possible
    ///
    /// Returns `Cow::Borrowed` when the IRI has no escape sequences.
    /// Returns `Cow::Owned` when escape sequences need to be decoded.
    pub fn parse_iri_ref<'a>(&mut self, input: &'a [u8]) -> Result<Cow<'a, str>, ParseError> {
        // Check for opening/closing angle brackets
        if input.len() < 2 || input[0] != b'<' || input[input.len() - 1] != b'>' {
            return Err(ParseError::InvalidIriFormat);
        }

        // Get the content between < and >
        let content = &input[1..input.len() - 1];

        // Fast path: Check if content needs decoding
        if !Self::needs_decoding(content) {
            // No escape sequences - return borrowed string
            match std::str::from_utf8(content) {
                Ok(s) => Ok(Cow::Borrowed(s)),
                Err(_) => Err(ParseError::InvalidUtf8),
            }
        } else {
            // Has escape sequences - check cache first
            if let Some(cached) = self.decode_cache.get(content) {
                return Ok(Cow::Owned(cached.clone()));
            }

            // Decode and cache
            let decoded = Self::decode_iri(content)?;
            self.decode_cache.insert(content.to_vec(), decoded.clone());
            Ok(Cow::Owned(decoded))
        }
    }

    /// Check if IRI content contains escape sequences that need decoding
    #[inline]
    fn needs_decoding(content: &[u8]) -> bool {
        content.iter().any(|&b| b == b'%' || b == b'\\')
    }

    /// Decode percent-encoded and escaped characters in IRI
    fn decode_iri(content: &[u8]) -> Result<String, ParseError> {
        let mut result = String::with_capacity(content.len());
        let mut i = 0;

        while i < content.len() {
            match content[i] {
                b'%' => {
                    // Percent-encoded character
                    if i + 2 >= content.len() {
                        return Err(ParseError::InvalidEscape);
                    }

                    let hex = &content[i + 1..i + 3];
                    let byte = Self::decode_hex_byte(hex)?;
                    result.push(byte as char);
                    i += 3;
                }
                b'\\' => {
                    // Escape sequence
                    if i + 1 >= content.len() {
                        return Err(ParseError::InvalidEscape);
                    }

                    let escaped = match content[i + 1] {
                        b't' => '\t',
                        b'n' => '\n',
                        b'r' => '\r',
                        b'\\' => '\\',
                        b'>' => '>',
                        c => c as char,
                    };
                    result.push(escaped);
                    i += 2;
                }
                b => {
                    result.push(b as char);
                    i += 1;
                }
            }
        }

        Ok(result)
    }

    /// Decode a two-character hex sequence to a byte
    #[inline]
    fn decode_hex_byte(hex: &[u8]) -> Result<u8, ParseError> {
        if hex.len() != 2 {
            return Err(ParseError::InvalidEscape);
        }

        let high = Self::hex_digit(hex[0])?;
        let low = Self::hex_digit(hex[1])?;

        Ok((high << 4) | low)
    }

    /// Convert a hex digit character to its numeric value
    #[inline]
    fn hex_digit(c: u8) -> Result<u8, ParseError> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => Err(ParseError::InvalidHexDigit),
        }
    }

    /// Clear the decode cache
    pub fn clear_cache(&mut self) {
        self.decode_cache.clear();
    }

    /// Get the number of cached entries
    pub fn cache_size(&self) -> usize {
        self.decode_cache.len()
    }

    /// Shrink the cache to fit current usage
    pub fn shrink_cache(&mut self) {
        self.decode_cache.shrink_to_fit();
    }
}

impl Default for ZeroCopyIriParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse error types for zero-copy parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Invalid IRI format (missing angle brackets)
    InvalidIriFormat,
    /// Invalid UTF-8 encoding
    InvalidUtf8,
    /// Invalid escape sequence
    InvalidEscape,
    /// Invalid hexadecimal digit
    InvalidHexDigit,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::InvalidIriFormat => write!(f, "Invalid IRI format"),
            ParseError::InvalidUtf8 => write!(f, "Invalid UTF-8 encoding"),
            ParseError::InvalidEscape => write!(f, "Invalid escape sequence"),
            ParseError::InvalidHexDigit => write!(f, "Invalid hexadecimal digit"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Zero-copy literal parser
///
/// Parses string literals without allocating when possible.
#[derive(Debug, Clone)]
pub struct ZeroCopyLiteralParser;

impl ZeroCopyLiteralParser {
    /// Create a new zero-copy literal parser
    pub fn new() -> Self {
        Self
    }

    /// Parse a string literal, returning zero-copy result when possible
    ///
    /// Returns `Cow::Borrowed` when the literal has no escape sequences.
    /// Returns `Cow::Owned` when escape sequences need to be decoded.
    pub fn parse_string_literal<'a>(&self, input: &'a [u8]) -> Result<Cow<'a, str>, ParseError> {
        // Check for quotes
        if input.len() < 2 {
            return Err(ParseError::InvalidIriFormat);
        }

        let quote = input[0];
        if quote != b'"' && quote != b'\'' {
            return Err(ParseError::InvalidIriFormat);
        }

        // Check for matching closing quote
        if input[input.len() - 1] != quote {
            return Err(ParseError::InvalidIriFormat);
        }

        let content = &input[1..input.len() - 1];

        // Fast path: no escapes
        if !content.contains(&b'\\') {
            match std::str::from_utf8(content) {
                Ok(s) => Ok(Cow::Borrowed(s)),
                Err(_) => Err(ParseError::InvalidUtf8),
            }
        } else {
            // Has escapes - decode
            Ok(Cow::Owned(Self::decode_string(content)?))
        }
    }

    /// Decode escape sequences in a string literal
    fn decode_string(content: &[u8]) -> Result<String, ParseError> {
        let mut result = String::with_capacity(content.len());
        let mut i = 0;

        while i < content.len() {
            if content[i] == b'\\' {
                if i + 1 >= content.len() {
                    return Err(ParseError::InvalidEscape);
                }

                let escaped = match content[i + 1] {
                    b't' => '\t',
                    b'n' => '\n',
                    b'r' => '\r',
                    b'\\' => '\\',
                    b'"' => '"',
                    b'\'' => '\'',
                    b'u' => {
                        // Unicode escape \uXXXX
                        if i + 5 >= content.len() {
                            return Err(ParseError::InvalidEscape);
                        }
                        let hex = &content[i + 2..i + 6];
                        let codepoint = Self::decode_unicode_4(hex)?;
                        i += 6; // Skip \uXXXX
                        result.push(codepoint);
                        continue;
                    }
                    b'U' => {
                        // Unicode escape \UXXXXXXXX
                        if i + 9 >= content.len() {
                            return Err(ParseError::InvalidEscape);
                        }
                        let hex = &content[i + 2..i + 10];
                        let codepoint = Self::decode_unicode_8(hex)?;
                        i += 10; // Skip \UXXXXXXXX
                        result.push(codepoint);
                        continue;
                    }
                    c => c as char,
                };

                result.push(escaped);
                i += 2;
            } else {
                result.push(content[i] as char);
                i += 1;
            }
        }

        Ok(result)
    }

    /// Decode 4-digit Unicode escape sequence
    fn decode_unicode_4(hex: &[u8]) -> Result<char, ParseError> {
        if hex.len() != 4 {
            return Err(ParseError::InvalidEscape);
        }

        let mut value = 0u32;
        for &byte in hex {
            value = (value << 4) | ZeroCopyIriParser::hex_digit(byte)? as u32;
        }

        char::from_u32(value).ok_or(ParseError::InvalidEscape)
    }

    /// Decode 8-digit Unicode escape sequence
    fn decode_unicode_8(hex: &[u8]) -> Result<char, ParseError> {
        if hex.len() != 8 {
            return Err(ParseError::InvalidEscape);
        }

        let mut value = 0u32;
        for &byte in hex {
            value = (value << 4) | ZeroCopyIriParser::hex_digit(byte)? as u32;
        }

        char::from_u32(value).ok_or(ParseError::InvalidEscape)
    }
}

impl Default for ZeroCopyLiteralParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_iri_no_allocation() {
        let mut parser = ZeroCopyIriParser::new();
        let iri = parser.parse_iri_ref(b"<http://example.org/>").unwrap();

        // Should be borrowed (no allocation)
        assert!(matches!(iri, Cow::Borrowed(_)));
        assert_eq!(iri, "http://example.org/");
    }

    #[test]
    fn test_iri_with_escape_allocates() {
        let mut parser = ZeroCopyIriParser::new();
        let iri = parser
            .parse_iri_ref(b"<http://example.org/sp%20ace>")
            .unwrap();

        // Should be owned (allocation for decoding)
        assert!(matches!(iri, Cow::Owned(_)));
        assert_eq!(iri, "http://example.org/sp ace");
    }

    #[test]
    fn test_iri_cache() {
        let mut parser = ZeroCopyIriParser::new();

        // Parse same IRI twice
        parser
            .parse_iri_ref(b"<http://example.org/sp%20ace>")
            .unwrap();
        parser
            .parse_iri_ref(b"<http://example.org/sp%20ace>")
            .unwrap();

        // Should have one cached entry
        assert_eq!(parser.cache_size(), 1);
    }

    #[test]
    fn test_invalid_iri_format() {
        let mut parser = ZeroCopyIriParser::new();

        // Missing closing bracket
        assert!(parser.parse_iri_ref(b"<http://example.org/").is_err());

        // Missing opening bracket
        assert!(parser.parse_iri_ref(b"http://example.org/>").is_err());

        // No brackets
        assert!(parser.parse_iri_ref(b"http://example.org/").is_err());
    }

    #[test]
    fn test_string_literal_no_allocation() {
        let parser = ZeroCopyLiteralParser::new();
        let literal = parser.parse_string_literal(b"\"hello world\"").unwrap();

        // Should be borrowed (no escape sequences)
        assert!(matches!(literal, Cow::Borrowed(_)));
        assert_eq!(literal, "hello world");
    }

    #[test]
    fn test_string_literal_with_escapes() {
        let parser = ZeroCopyLiteralParser::new();
        let literal = parser.parse_string_literal(b"\"hello\\nworld\"").unwrap();

        // Should be owned (escape sequence decoded)
        assert!(matches!(literal, Cow::Owned(_)));
        assert_eq!(literal, "hello\nworld");
    }

    #[test]
    fn test_string_literal_unicode_escape() {
        let parser = ZeroCopyLiteralParser::new();

        // \u0041 = 'A'
        let literal = parser.parse_string_literal(b"\"\\u0041BC\"").unwrap();
        assert_eq!(literal, "ABC");

        // \U00000041 = 'A'
        let literal = parser.parse_string_literal(b"\"\\U00000041BC\"").unwrap();
        assert_eq!(literal, "ABC");
    }

    #[test]
    fn test_string_literal_mixed_quotes() {
        let parser = ZeroCopyLiteralParser::new();

        // Double quotes
        let literal = parser.parse_string_literal(b"\"test\"").unwrap();
        assert_eq!(literal, "test");

        // Single quotes
        let literal = parser.parse_string_literal(b"'test'").unwrap();
        assert_eq!(literal, "test");
    }

    #[test]
    fn test_hex_digit_decoding() {
        assert_eq!(ZeroCopyIriParser::hex_digit(b'0').unwrap(), 0);
        assert_eq!(ZeroCopyIriParser::hex_digit(b'9').unwrap(), 9);
        assert_eq!(ZeroCopyIriParser::hex_digit(b'a').unwrap(), 10);
        assert_eq!(ZeroCopyIriParser::hex_digit(b'f').unwrap(), 15);
        assert_eq!(ZeroCopyIriParser::hex_digit(b'A').unwrap(), 10);
        assert_eq!(ZeroCopyIriParser::hex_digit(b'F').unwrap(), 15);

        assert!(ZeroCopyIriParser::hex_digit(b'g').is_err());
        assert!(ZeroCopyIriParser::hex_digit(b'Z').is_err());
    }

    #[test]
    fn test_percent_encoding() {
        let mut parser = ZeroCopyIriParser::new();

        // Space encoded as %20
        let iri = parser.parse_iri_ref(b"<http://example.org/%20>").unwrap();
        assert_eq!(iri, "http://example.org/ ");

        // Multiple percent-encoded characters
        let iri = parser
            .parse_iri_ref(b"<http://example.org/%20%21%22>")
            .unwrap();
        assert_eq!(iri, "http://example.org/ !\"");
    }

    #[test]
    fn test_clear_cache() {
        let mut parser = ZeroCopyIriParser::new();

        parser
            .parse_iri_ref(b"<http://example.org/sp%20ace>")
            .unwrap();
        assert_eq!(parser.cache_size(), 1);

        parser.clear_cache();
        assert_eq!(parser.cache_size(), 0);
    }
}
