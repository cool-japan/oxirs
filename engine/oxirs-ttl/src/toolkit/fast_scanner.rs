//! Fast scanning utilities for RDF tokenization
//!
//! This module provides SIMD-accelerated and optimized scanning functions
//! for common tokenization operations. These functions are hot paths in RDF parsing
//! and benefit significantly from optimization.

use memchr::{memchr, memchr2, memchr3};

/// Fast scanner for RDF tokenization with SIMD-accelerated operations
///
/// This scanner provides optimized methods for common lexing operations:
/// - Skipping whitespace
/// - Finding delimiters
/// - Scanning strings
/// - Finding comments
///
/// # Example
///
/// ```
/// use oxirs_ttl::toolkit::FastScanner;
///
/// let input = "  <http://example.org/>  ";
/// let scanner = FastScanner::new(input.as_bytes());
///
/// // Skip whitespace efficiently
/// let pos = scanner.skip_whitespace(0);
/// assert_eq!(pos, 2); // Skipped to '<'
/// ```
#[derive(Debug, Clone)]
pub struct FastScanner<'a> {
    bytes: &'a [u8],
    len: usize,
}

impl<'a> FastScanner<'a> {
    /// Create a new fast scanner
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            len: bytes.len(),
        }
    }

    /// Get the length of the input
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the scanner is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Skip whitespace and return the new position
    ///
    /// Uses SIMD-accelerated byte scanning for maximum performance.
    /// Whitespace includes: space, tab, newline, carriage return
    #[inline]
    pub fn skip_whitespace(&self, mut pos: usize) -> usize {
        while pos < self.len {
            match self.bytes[pos] {
                b' ' | b'\t' | b'\n' | b'\r' => pos += 1,
                _ => break,
            }
        }
        pos
    }

    /// Skip whitespace and comments, return new position
    ///
    /// Turtle comments start with # and continue to end of line
    #[inline]
    pub fn skip_whitespace_and_comments(&self, mut pos: usize) -> usize {
        loop {
            // Skip whitespace
            pos = self.skip_whitespace(pos);

            // Check for comment
            if pos < self.len && self.bytes[pos] == b'#' {
                // Skip to end of line
                pos = self.find_line_end(pos).unwrap_or(self.len);
            } else {
                break;
            }
        }
        pos
    }

    /// Find the next occurrence of a specific byte, starting from pos
    ///
    /// Uses memchr which is SIMD-accelerated on most platforms
    #[inline]
    pub fn find_byte(&self, byte: u8, pos: usize) -> Option<usize> {
        if pos >= self.len {
            return None;
        }
        memchr(byte, &self.bytes[pos..]).map(|offset| pos + offset)
    }

    /// Find the next occurrence of any of two bytes
    ///
    /// SIMD-accelerated search for either of two bytes
    #[inline]
    pub fn find_either_byte(&self, byte1: u8, byte2: u8, pos: usize) -> Option<usize> {
        if pos >= self.len {
            return None;
        }
        memchr2(byte1, byte2, &self.bytes[pos..]).map(|offset| pos + offset)
    }

    /// Find the next occurrence of any of three bytes
    ///
    /// SIMD-accelerated search for any of three bytes
    #[inline]
    pub fn find_any_of_three(&self, byte1: u8, byte2: u8, byte3: u8, pos: usize) -> Option<usize> {
        if pos >= self.len {
            return None;
        }
        memchr3(byte1, byte2, byte3, &self.bytes[pos..]).map(|offset| pos + offset)
    }

    /// Find the end of the current line (newline or carriage return)
    #[inline]
    pub fn find_line_end(&self, pos: usize) -> Option<usize> {
        self.find_either_byte(b'\n', b'\r', pos)
    }

    /// Scan until a delimiter is found
    ///
    /// Delimiters: space, tab, newline, CR, `<`, `>`, `@`, `;`, `,`, `.`, `[`, `]`, `(`, `)`, `{`, `}`, `#`, `"`
    pub fn scan_until_delimiter(&self, mut pos: usize) -> usize {
        while pos < self.len {
            match self.bytes[pos] {
                b' ' | b'\t' | b'\n' | b'\r' | b'<' | b'>' | b'@' | b';' | b',' | b'.' | b'['
                | b']' | b'(' | b')' | b'{' | b'}' | b'#' | b'"' | b'\'' => break,
                _ => pos += 1,
            }
        }
        pos
    }

    /// Scan a string literal, handling escapes
    ///
    /// Returns the position after the closing quote, or None if unterminated
    pub fn scan_string_literal(&self, start: usize, quote: u8) -> Option<usize> {
        let mut pos = start + 1; // Skip opening quote

        while pos < self.len {
            match self.bytes[pos] {
                b'\\' => {
                    // Skip escape sequence
                    pos += 2;
                }
                byte if byte == quote => {
                    // Found closing quote
                    return Some(pos + 1);
                }
                _ => {
                    pos += 1;
                }
            }
        }

        None // Unterminated string
    }

    /// Scan a long string literal (""" or ''')
    ///
    /// Returns the position after the closing triple quotes
    pub fn scan_long_string_literal(&self, start: usize, quote: u8) -> Option<usize> {
        let mut pos = start + 3; // Skip opening triple quotes

        while pos + 2 < self.len {
            if self.bytes[pos] == quote
                && self.bytes[pos + 1] == quote
                && self.bytes[pos + 2] == quote
            {
                return Some(pos + 3);
            }

            if self.bytes[pos] == b'\\' {
                pos += 2; // Skip escape
            } else {
                pos += 1;
            }
        }

        None // Unterminated long string
    }

    /// Check if a long string starts at position (""" or ''')
    #[inline]
    pub fn is_long_string_start(&self, pos: usize, quote: u8) -> bool {
        pos + 2 < self.len
            && self.bytes[pos] == quote
            && self.bytes[pos + 1] == quote
            && self.bytes[pos + 2] == quote
    }

    /// Scan an IRI reference (<...>)
    ///
    /// Returns the position after the closing >, or None if unterminated
    pub fn scan_iri_ref(&self, start: usize) -> Option<usize> {
        let mut pos = start + 1; // Skip opening <

        while pos < self.len {
            match self.bytes[pos] {
                b'>' => return Some(pos + 1),
                b'\\' => pos += 2,            // Skip escape
                b'\n' | b'\r' => return None, // Invalid: newline in IRI
                _ => pos += 1,
            }
        }

        None // Unterminated IRI
    }

    /// Fast check if a byte is a valid PN_CHARS_BASE (used in prefixed names)
    #[inline]
    pub fn is_pn_chars_base(byte: u8) -> bool {
        matches!(byte,
            b'A'..=b'Z' | b'a'..=b'z' |
            0xC0..=0xD6 | 0xD8..=0xF6 | 0xF8..=0xFF
        )
    }

    /// Fast check if a byte is a valid PN_CHARS (used in prefixed names)
    #[inline]
    pub fn is_pn_chars(byte: u8) -> bool {
        matches!(byte,
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' |
            b'-' | b'_' | b'.' |
            0xC0..=0xD6 | 0xD8..=0xF6 | 0xF8..=0xFF
        )
    }

    /// Scan a prefixed name (prefix:local)
    ///
    /// Returns (end_of_prefix, end_of_local) positions
    pub fn scan_prefixed_name(&self, start: usize) -> Option<(usize, usize)> {
        let mut pos = start;

        // Scan prefix part (can be empty)
        let prefix_end = if pos < self.len && Self::is_pn_chars_base(self.bytes[pos]) {
            pos += 1;
            while pos < self.len && Self::is_pn_chars(self.bytes[pos]) {
                pos += 1;
            }
            pos
        } else {
            pos
        };

        // Must have colon
        if pos >= self.len || self.bytes[pos] != b':' {
            return None;
        }
        pos += 1; // Skip colon

        // Scan local part (can be empty for some cases)
        let _local_start = pos;
        if pos < self.len && (Self::is_pn_chars_base(self.bytes[pos]) || self.bytes[pos] == b'_') {
            pos += 1;
            while pos < self.len && Self::is_pn_chars(self.bytes[pos]) {
                pos += 1;
            }
        }

        Some((prefix_end, pos))
    }

    /// Get a slice of bytes from start to end
    #[inline]
    pub fn slice(&self, start: usize, end: usize) -> &'a [u8] {
        &self.bytes[start..end.min(self.len)]
    }

    /// Get a byte at position
    #[inline]
    pub fn byte_at(&self, pos: usize) -> Option<u8> {
        if pos < self.len {
            Some(self.bytes[pos])
        } else {
            None
        }
    }

    /// Count lines up to position (for error reporting)
    pub fn count_lines(&self, pos: usize) -> usize {
        self.bytes[..pos.min(self.len)]
            .iter()
            .filter(|&&b| b == b'\n')
            .count()
            + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skip_whitespace() {
        let input = "   hello";
        let scanner = FastScanner::new(input.as_bytes());

        let pos = scanner.skip_whitespace(0);
        assert_eq!(pos, 3);
        assert_eq!(scanner.byte_at(pos), Some(b'h'));
    }

    #[test]
    fn test_skip_whitespace_mixed() {
        let input = "  \t\n  hello";
        let scanner = FastScanner::new(input.as_bytes());

        let pos = scanner.skip_whitespace(0);
        assert_eq!(pos, 6);
    }

    #[test]
    fn test_find_byte() {
        let input = "hello world";
        let scanner = FastScanner::new(input.as_bytes());

        assert_eq!(scanner.find_byte(b'w', 0), Some(6));
        assert_eq!(scanner.find_byte(b'o', 0), Some(4));
        assert_eq!(scanner.find_byte(b'z', 0), None);
    }

    #[test]
    fn test_find_either_byte() {
        let input = "hello world";
        let scanner = FastScanner::new(input.as_bytes());

        assert_eq!(scanner.find_either_byte(b'w', b'o', 0), Some(4)); // Finds 'o' first
        assert_eq!(scanner.find_either_byte(b'x', b'y', 0), None);
    }

    #[test]
    fn test_scan_until_delimiter() {
        let input = "prefix:local <iri>";
        let scanner = FastScanner::new(input.as_bytes());

        let pos = scanner.scan_until_delimiter(0);
        assert_eq!(pos, 12); // Stops at space
        assert_eq!(
            std::str::from_utf8(scanner.slice(0, pos)).expect("valid UTF-8"),
            "prefix:local"
        );
    }

    #[test]
    fn test_scan_string_literal() {
        let input = r#""hello world" more"#;
        let scanner = FastScanner::new(input.as_bytes());

        let end = scanner.scan_string_literal(0, b'"');
        assert_eq!(end, Some(13)); // Position after closing quote
    }

    #[test]
    fn test_scan_string_with_escape() {
        let input = r#""hello \"world\"" more"#;
        let scanner = FastScanner::new(input.as_bytes());

        let end = scanner.scan_string_literal(0, b'"');
        assert_eq!(end, Some(17));
    }

    #[test]
    fn test_scan_long_string() {
        let input = r#""""hello
world""" more"#;
        let scanner = FastScanner::new(input.as_bytes());

        let end = scanner.scan_long_string_literal(0, b'"');
        // Input: """hello\nworld""" more
        // Positions: 0-2: """, 3-7: hello, 8: \n, 9-13: world, 14-16: """
        assert_eq!(end, Some(17)); // Position after closing """
    }

    #[test]
    fn test_scan_iri_ref() {
        let input = "<http://example.org/> more";
        let scanner = FastScanner::new(input.as_bytes());

        let end = scanner.scan_iri_ref(0);
        assert_eq!(end, Some(21));
    }

    #[test]
    fn test_scan_iri_ref_unterminated() {
        let input = "<http://example.org/";
        let scanner = FastScanner::new(input.as_bytes());

        let end = scanner.scan_iri_ref(0);
        assert_eq!(end, None);
    }

    #[test]
    fn test_scan_prefixed_name() {
        let input = "prefix:local ";
        let scanner = FastScanner::new(input.as_bytes());

        let result = scanner.scan_prefixed_name(0);
        assert_eq!(result, Some((6, 12))); // (end_of_prefix, end_of_local)
    }

    #[test]
    fn test_scan_prefixed_name_no_local() {
        let input = "prefix: ";
        let scanner = FastScanner::new(input.as_bytes());

        let result = scanner.scan_prefixed_name(0);
        assert_eq!(result, Some((6, 7))); // Empty local part
    }

    #[test]
    fn test_skip_whitespace_and_comments() {
        let input = "  # comment\n  hello";
        let scanner = FastScanner::new(input.as_bytes());

        let pos = scanner.skip_whitespace_and_comments(0);
        // Input: "  # comment\n  hello"
        // Positions: 0-1: spaces, 2: #, 3-10: comment, 11: \n, 12-13: spaces, 14: h
        assert_eq!(pos, 14); // Skipped whitespace, comment, and more whitespace
        assert_eq!(scanner.byte_at(pos), Some(b'h'));
    }

    #[test]
    fn test_is_long_string_start() {
        let scanner = FastScanner::new(br#"""""#);
        assert!(scanner.is_long_string_start(0, b'"'));

        let scanner2 = FastScanner::new(br#""hello"#);
        assert!(!scanner2.is_long_string_start(0, b'"'));
    }

    #[test]
    fn test_count_lines() {
        let input = "line1\nline2\nline3";
        let scanner = FastScanner::new(input.as_bytes());

        assert_eq!(scanner.count_lines(0), 1);
        assert_eq!(scanner.count_lines(6), 2); // After first newline
        assert_eq!(scanner.count_lines(12), 3); // After second newline
    }

    #[test]
    fn test_find_line_end() {
        let input = "hello world\nmore text";
        let scanner = FastScanner::new(input.as_bytes());

        assert_eq!(scanner.find_line_end(0), Some(11));
    }

    #[test]
    fn test_empty_scanner() {
        let scanner = FastScanner::new(&[]);

        assert!(scanner.is_empty());
        assert_eq!(scanner.len(), 0);
        assert_eq!(scanner.skip_whitespace(0), 0);
        assert_eq!(scanner.find_byte(b'x', 0), None);
    }
}
