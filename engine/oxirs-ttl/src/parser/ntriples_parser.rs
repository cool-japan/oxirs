//! Standalone N-Triples and N-Quads parser
//!
//! These parsers operate on the lightweight [`RdfTerm`] type from
//! [`crate::writer`] and do not depend on `oxirs-core`'s model types.
//! They follow the W3C specifications:
//! - N-Triples: <https://www.w3.org/TR/n-triples/>
//! - N-Quads: <https://www.w3.org/TR/n-quads/>
//!
//! # Examples
//!
//! ## N-Triples
//!
//! ```rust
//! use oxirs_ttl::parser::{NTriplesLiteParser, ParseError};
//!
//! let mut parser = NTriplesLiteParser::new();
//! let input = "<http://example.org/s> <http://example.org/p> \"hello\" .\n\
//! <http://example.org/s> <http://example.org/p2> _:b0 .\n";
//! let triples = parser.parse_str(input)?;
//! assert_eq!(triples.len(), 2);
//! # Ok::<(), ParseError>(())
//! ```
//!
//! ## N-Quads
//!
//! ```rust
//! use oxirs_ttl::parser::{NQuadsLiteParser, ParseError};
//!
//! let mut parser = NQuadsLiteParser::new();
//! let quads = parser.parse_str(
//!     "<http://s> <http://p> <http://o> <http://g> .\n"
//! )?;
//! assert_eq!(quads.len(), 1);
//! assert_eq!(quads[0].3.as_ref().map(|t| t.value.as_str()), Some("http://g"));
//! # Ok::<(), ParseError>(())
//! ```

use crate::writer::RdfTerm;

// ─── Error type ─────────────────────────────────────────────────────────────

/// Parsing error produced by [`NTriplesLiteParser`] and [`NQuadsLiteParser`]
#[derive(Debug, Clone)]
pub struct ParseError {
    /// 1-based line number where the error occurred
    pub line: usize,
    /// Human-readable description
    pub message: String,
}

impl ParseError {
    fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

// ─── Triple type alias ───────────────────────────────────────────────────────

/// A parsed N-Triple: `(subject, predicate, object)`
pub type NTriple = (RdfTerm, RdfTerm, RdfTerm);

/// A parsed N-Quad: `(subject, predicate, object, graph_name?)`
pub type NQuad = (RdfTerm, RdfTerm, RdfTerm, Option<RdfTerm>);

// ─── N-Triples parser ────────────────────────────────────────────────────────

/// Standalone N-Triples parser using [`RdfTerm`]
///
/// Tracks the current line number across multiple [`parse_str`](Self::parse_str) calls
/// so that errors always report accurate positions.
#[derive(Debug, Clone)]
pub struct NTriplesLiteParser {
    line_count: usize,
}

impl Default for NTriplesLiteParser {
    fn default() -> Self {
        Self::new()
    }
}

impl NTriplesLiteParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self { line_count: 0 }
    }

    /// Reset the internal line counter
    pub fn reset(&mut self) {
        self.line_count = 0;
    }

    /// Parse a complete N-Triples document
    pub fn parse_str(&mut self, input: &str) -> Result<Vec<NTriple>, ParseError> {
        let mut triples = Vec::new();
        for raw_line in input.lines() {
            self.line_count += 1;
            if let Some(triple) = self.parse_line(raw_line)? {
                triples.push(triple);
            }
        }
        Ok(triples)
    }

    /// Parse a single N-Triples line.
    ///
    /// Returns `Ok(None)` for blank lines and comment lines.
    pub fn parse_line(&mut self, line: &str) -> Result<Option<NTriple>, ParseError> {
        let line = strip_comment(line).trim();
        if line.is_empty() {
            return Ok(None);
        }

        let mut pos = 0;

        let subject = parse_term(line, &mut pos, self.line_count)?;
        skip_whitespace(line, &mut pos);

        let predicate = parse_iri_term(line, &mut pos, self.line_count)?;
        skip_whitespace(line, &mut pos);

        let object = parse_term(line, &mut pos, self.line_count)?;
        skip_whitespace(line, &mut pos);

        expect_dot(line, &mut pos, self.line_count)?;

        Ok(Some((subject, predicate, object)))
    }
}

// ─── N-Quads parser ──────────────────────────────────────────────────────────

/// Standalone N-Quads parser using [`RdfTerm`]
#[derive(Debug, Clone)]
pub struct NQuadsLiteParser {
    line_count: usize,
}

impl Default for NQuadsLiteParser {
    fn default() -> Self {
        Self::new()
    }
}

impl NQuadsLiteParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self { line_count: 0 }
    }

    /// Reset the internal line counter
    pub fn reset(&mut self) {
        self.line_count = 0;
    }

    /// Parse a complete N-Quads document
    pub fn parse_str(&mut self, input: &str) -> Result<Vec<NQuad>, ParseError> {
        let mut quads = Vec::new();
        for raw_line in input.lines() {
            self.line_count += 1;
            if let Some(quad) = self.parse_line(raw_line)? {
                quads.push(quad);
            }
        }
        Ok(quads)
    }

    /// Parse a single N-Quads line.
    ///
    /// The optional fourth field is the named graph IRI or blank node.
    pub fn parse_line(&mut self, line: &str) -> Result<Option<NQuad>, ParseError> {
        let line = strip_comment(line).trim();
        if line.is_empty() {
            return Ok(None);
        }

        let mut pos = 0;

        let subject = parse_term(line, &mut pos, self.line_count)?;
        skip_whitespace(line, &mut pos);

        let predicate = parse_iri_term(line, &mut pos, self.line_count)?;
        skip_whitespace(line, &mut pos);

        let object = parse_term(line, &mut pos, self.line_count)?;
        skip_whitespace(line, &mut pos);

        // Optional graph name before the dot
        let graph_name = if pos < line.len() && !line[pos..].starts_with('.') {
            let g = parse_graph_name(line, &mut pos, self.line_count)?;
            skip_whitespace(line, &mut pos);
            Some(g)
        } else {
            None
        };

        expect_dot(line, &mut pos, self.line_count)?;

        Ok(Some((subject, predicate, object, graph_name)))
    }
}

// ─── Low-level parsing helpers ───────────────────────────────────────────────

/// Strip a `#`-prefixed comment, respecting quoted strings and IRI brackets
fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    let mut in_iri = false;
    let mut escaped = false;

    for (i, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' if !in_iri => in_string = !in_string,
            '<' if !in_string => in_iri = true,
            '>' if !in_string => in_iri = false,
            '#' if !in_string && !in_iri => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Advance `pos` past ASCII whitespace
fn skip_whitespace(s: &str, pos: &mut usize) {
    while *pos < s.len() {
        let byte = s.as_bytes()[*pos];
        if byte == b' ' || byte == b'\t' {
            *pos += 1;
        } else {
            break;
        }
    }
}

/// Expect and consume a `.` terminator
fn expect_dot(s: &str, pos: &mut usize, line: usize) -> Result<(), ParseError> {
    skip_whitespace(s, pos);
    if *pos >= s.len() {
        return Err(ParseError::new(line, "expected '.' but found end of line"));
    }
    if s.as_bytes()[*pos] != b'.' {
        let found = &s[*pos..(*pos + 1).min(s.len())];
        return Err(ParseError::new(
            line,
            format!("expected '.', found '{found}'"),
        ));
    }
    *pos += 1;
    Ok(())
}

/// Parse any valid N-Triples term (IRI, blank node, or literal)
fn parse_term(s: &str, pos: &mut usize, line: usize) -> Result<RdfTerm, ParseError> {
    if *pos >= s.len() {
        return Err(ParseError::new(line, "unexpected end of input"));
    }
    match s.as_bytes()[*pos] {
        b'<' => parse_iri(s, pos, line),
        b'_' => parse_blank_node(s, pos, line),
        b'"' => parse_literal(s, pos, line),
        other => Err(ParseError::new(
            line,
            format!(
                "unexpected character '{}' at position {pos}",
                char::from(other)
            ),
        )),
    }
}

/// Parse a subject/predicate that must be an IRI (no literals or blank nodes for predicate)
fn parse_iri_term(s: &str, pos: &mut usize, line: usize) -> Result<RdfTerm, ParseError> {
    if *pos >= s.len() {
        return Err(ParseError::new(line, "expected IRI"));
    }
    if s.as_bytes()[*pos] != b'<' {
        return Err(ParseError::new(
            line,
            format!(
                "expected '<' for predicate IRI, found '{}'",
                char::from(s.as_bytes()[*pos])
            ),
        ));
    }
    parse_iri(s, pos, line)
}

/// Parse an IRI enclosed in `<…>`
fn parse_iri(s: &str, pos: &mut usize, line: usize) -> Result<RdfTerm, ParseError> {
    debug_assert_eq!(s.as_bytes()[*pos], b'<');
    *pos += 1; // skip '<'

    let start = *pos;
    let bytes = s.as_bytes();

    while *pos < bytes.len() {
        match bytes[*pos] {
            b'>' => {
                let iri = &s[start..*pos];
                *pos += 1; // skip '>'
                return Ok(RdfTerm::iri(iri));
            }
            b'\\' => {
                // IRI escape sequences (e.g. \uXXXX) — advance past the backslash;
                // for simplicity we let the value contain the raw escape.
                *pos += 1;
                if *pos < bytes.len() {
                    *pos += 1;
                }
            }
            _ => *pos += 1,
        }
    }

    Err(ParseError::new(line, "unterminated IRI (missing '>')"))
}

/// Parse a blank node `_:label`
fn parse_blank_node(s: &str, pos: &mut usize, line: usize) -> Result<RdfTerm, ParseError> {
    if s.len() < *pos + 2 || &s[*pos..*pos + 2] != "_:" {
        return Err(ParseError::new(line, "invalid blank node: expected '_:'"));
    }
    *pos += 2; // skip "_:"

    let start = *pos;
    let bytes = s.as_bytes();

    while *pos < bytes.len() {
        let b = bytes[*pos];
        if b == b' ' || b == b'\t' || b == b'.' || b == b',' || b == b';' {
            break;
        }
        *pos += 1;
    }

    if *pos == start {
        return Err(ParseError::new(line, "blank node label is empty"));
    }

    Ok(RdfTerm::blank_node(&s[start..*pos]))
}

/// Parse a literal `"…"` with optional `@lang` or `^^<datatype>`
fn parse_literal(s: &str, pos: &mut usize, line: usize) -> Result<RdfTerm, ParseError> {
    debug_assert_eq!(s.as_bytes()[*pos], b'"');
    *pos += 1; // skip opening quote

    let mut value = String::new();
    let bytes = s.as_bytes();

    loop {
        if *pos >= bytes.len() {
            return Err(ParseError::new(line, "unterminated string literal"));
        }
        match bytes[*pos] {
            b'"' => {
                *pos += 1; // skip closing quote
                break;
            }
            b'\\' => {
                *pos += 1;
                if *pos >= bytes.len() {
                    return Err(ParseError::new(line, "unexpected end after '\\'"));
                }
                let ch = unescape_sequence(bytes[*pos], s, pos, line)?;
                value.push(ch);
            }
            _ => {
                // Collect a full UTF-8 character
                let ch = s[*pos..]
                    .chars()
                    .next()
                    .ok_or_else(|| ParseError::new(line, "invalid UTF-8 in literal"))?;
                value.push(ch);
                *pos += ch.len_utf8();
            }
        }
    }

    // Check for language tag or datatype annotation
    if *pos < bytes.len() && bytes[*pos] == b'@' {
        *pos += 1;
        let lang_start = *pos;
        while *pos < bytes.len() && (bytes[*pos].is_ascii_alphanumeric() || bytes[*pos] == b'-') {
            *pos += 1;
        }
        let lang = &s[lang_start..*pos];
        if lang.is_empty() {
            return Err(ParseError::new(line, "empty language tag after '@'"));
        }
        return Ok(RdfTerm::lang_literal(value, lang));
    }

    if *pos + 1 < bytes.len() && bytes[*pos] == b'^' && bytes[*pos + 1] == b'^' {
        *pos += 2;
        if *pos >= bytes.len() || bytes[*pos] != b'<' {
            return Err(ParseError::new(line, "expected '<' after '^^'"));
        }
        let dt_term = parse_iri(s, pos, line)?;
        return Ok(RdfTerm::typed_literal(value, dt_term.value));
    }

    Ok(RdfTerm::simple_literal(value))
}

/// Resolve a single escape byte following a `\`
fn unescape_sequence(byte: u8, s: &str, pos: &mut usize, line: usize) -> Result<char, ParseError> {
    *pos += 1; // move past the escape character
    match byte {
        b't' => Ok('\t'),
        b'n' => Ok('\n'),
        b'r' => Ok('\r'),
        b'"' => Ok('"'),
        b'\'' => Ok('\''),
        b'\\' => Ok('\\'),
        b'u' => {
            // \uXXXX
            let hex = read_hex_digits(s, pos, 4, line)?;
            let cp = u32::from_str_radix(&hex, 16)
                .map_err(|_| ParseError::new(line, format!("invalid \\u escape: {hex}")))?;
            char::from_u32(cp).ok_or_else(|| {
                ParseError::new(line, format!("invalid Unicode code point U+{cp:04X}"))
            })
        }
        b'U' => {
            // \UXXXXXXXX
            let hex = read_hex_digits(s, pos, 8, line)?;
            let cp = u32::from_str_radix(&hex, 16)
                .map_err(|_| ParseError::new(line, format!("invalid \\U escape: {hex}")))?;
            char::from_u32(cp).ok_or_else(|| {
                ParseError::new(line, format!("invalid Unicode code point U+{cp:08X}"))
            })
        }
        other => Err(ParseError::new(
            line,
            format!("unknown escape '\\{}'", char::from(other)),
        )),
    }
}

/// Read exactly `n` hex digits from `s` starting at `*pos`, advancing `*pos`
fn read_hex_digits(s: &str, pos: &mut usize, n: usize, line: usize) -> Result<String, ParseError> {
    if *pos + n > s.len() {
        return Err(ParseError::new(
            line,
            format!("expected {n} hex digits but found end of input"),
        ));
    }
    let hex = &s[*pos..*pos + n];
    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ParseError::new(
            line,
            format!("expected {n} hex digits, found: {hex}"),
        ));
    }
    *pos += n;
    Ok(hex.to_string())
}

/// Parse a graph name (IRI or blank node) for N-Quads
fn parse_graph_name(s: &str, pos: &mut usize, line: usize) -> Result<RdfTerm, ParseError> {
    if *pos >= s.len() {
        return Err(ParseError::new(line, "expected graph name"));
    }
    match s.as_bytes()[*pos] {
        b'<' => parse_iri(s, pos, line),
        b'_' => parse_blank_node(s, pos, line),
        other => Err(ParseError::new(
            line,
            format!(
                "invalid graph name: expected '<' or '_', found '{}'",
                char::from(other)
            ),
        )),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::TermType;

    // ── NTriplesLiteParser ─────────────────────────────────────────────────

    #[test]
    fn test_ntriples_simple_iri_triple() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("<http://s> <http://p> <http://o> .")
            .expect("parse should succeed");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].0.value, "http://s");
        assert_eq!(triples[0].1.value, "http://p");
        assert_eq!(triples[0].2.value, "http://o");
    }

    #[test]
    fn test_ntriples_simple_literal() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("<http://s> <http://p> \"hello\" .")
            .expect("parse should succeed");
        assert_eq!(triples[0].2.value, "hello");
        assert_eq!(
            triples[0].2.term_type,
            TermType::Literal {
                datatype: None,
                lang: None
            }
        );
    }

    #[test]
    fn test_ntriples_language_literal() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("<http://s> <http://p> \"bonjour\"@fr .")
            .expect("parse should succeed");
        assert_eq!(
            triples[0].2.term_type,
            TermType::Literal {
                datatype: None,
                lang: Some("fr".to_string())
            }
        );
    }

    #[test]
    fn test_ntriples_typed_literal() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("<http://s> <http://p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .")
            .expect("parse should succeed");
        assert_eq!(
            triples[0].2.term_type,
            TermType::Literal {
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                lang: None,
            }
        );
    }

    #[test]
    fn test_ntriples_blank_node_subject() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("_:b0 <http://p> <http://o> .")
            .expect("parse should succeed");
        assert_eq!(triples[0].0.term_type, TermType::BlankNode);
        assert_eq!(triples[0].0.value, "b0");
    }

    #[test]
    fn test_ntriples_blank_node_object() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("<http://s> <http://p> _:b1 .")
            .expect("parse should succeed");
        assert_eq!(triples[0].2.term_type, TermType::BlankNode);
        assert_eq!(triples[0].2.value, "b1");
    }

    #[test]
    fn test_ntriples_comment_lines_skipped() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str(
                "# this is a comment\n<http://s> <http://p> <http://o> .\n# another comment\n",
            )
            .expect("parse should succeed");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_ntriples_escape_sequences() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("<http://s> <http://p> \"line1\\nline2\\ttab\" .")
            .expect("parse should succeed");
        assert_eq!(triples[0].2.value, "line1\nline2\ttab");
    }

    #[test]
    fn test_ntriples_unicode_escape() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("<http://s> <http://p> \"caf\\u00E9\" .")
            .expect("parse should succeed");
        assert_eq!(triples[0].2.value, "café");
    }

    #[test]
    fn test_ntriples_multiple_lines() {
        let input = "<http://s> <http://p1> \"a\" .\n<http://s> <http://p2> \"b\" .\n";
        let mut p = NTriplesLiteParser::new();
        let triples = p.parse_str(input).expect("parse should succeed");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_ntriples_error_missing_dot() {
        let mut p = NTriplesLiteParser::new();
        let result = p.parse_str("<http://s> <http://p> <http://o>");
        assert!(result.is_err(), "missing dot should fail");
    }

    #[test]
    fn test_ntriples_inline_comment() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("<http://s> <http://p> <http://o> . # inline comment")
            .expect("parse should succeed");
        assert_eq!(triples.len(), 1);
    }

    // ── NQuadsLiteParser ──────────────────────────────────────────────────

    #[test]
    fn test_nquads_with_graph() {
        let mut p = NQuadsLiteParser::new();
        let quads = p
            .parse_str("<http://s> <http://p> <http://o> <http://g> .")
            .expect("parse should succeed");
        assert_eq!(quads.len(), 1);
        let (s, pred, o, g) = &quads[0];
        assert_eq!(s.value, "http://s");
        assert_eq!(pred.value, "http://p");
        assert_eq!(o.value, "http://o");
        assert_eq!(g.as_ref().map(|t| t.value.as_str()), Some("http://g"));
    }

    #[test]
    fn test_nquads_without_graph() {
        let mut p = NQuadsLiteParser::new();
        let quads = p
            .parse_str("<http://s> <http://p> <http://o> .")
            .expect("parse should succeed");
        assert_eq!(quads[0].3, None);
    }

    #[test]
    fn test_nquads_blank_node_graph() {
        let mut p = NQuadsLiteParser::new();
        let quads = p
            .parse_str("<http://s> <http://p> <http://o> _:g0 .")
            .expect("parse should succeed");
        assert_eq!(
            quads[0].3.as_ref().map(|t| &t.term_type),
            Some(&TermType::BlankNode)
        );
    }

    #[test]
    fn test_nquads_literal_object() {
        let mut p = NQuadsLiteParser::new();
        let quads = p
            .parse_str("<http://s> <http://p> \"value\" <http://g> .")
            .expect("parse should succeed");
        assert_eq!(quads[0].2.value, "value");
    }

    #[test]
    fn test_nquads_multiple_quads() {
        let input = concat!(
            "<http://s1> <http://p> <http://o1> <http://g> .\n",
            "<http://s2> <http://p> <http://o2> .\n",
        );
        let mut p = NQuadsLiteParser::new();
        let quads = p.parse_str(input).expect("parse should succeed");
        assert_eq!(quads.len(), 2);
        assert!(quads[0].3.is_some());
        assert!(quads[1].3.is_none());
    }

    // ── Additional NTriplesLiteParser tests ────────────────────────────────

    #[test]
    fn test_ntriples_large_unicode_escape() {
        // \UXXXXXXXX — 8-digit code point for characters outside the BMP
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("<http://s> <http://p> \"\\U0001F600\" .")
            .expect("\\U escape parse should succeed");
        // U+1F600 is the "grinning face" emoji
        assert_eq!(triples[0].2.value, "\u{1F600}");
    }

    #[test]
    fn test_ntriples_reset_clears_line_counter() {
        let mut p = NTriplesLiteParser::new();
        // Parse two lines to advance the counter
        p.parse_str("<http://s> <http://p> <http://o> .\n<http://s2> <http://p2> <http://o2> .")
            .expect("initial parse should succeed");
        assert_eq!(p.line_count, 2, "line counter should be 2 after two lines");
        p.reset();
        assert_eq!(p.line_count, 0, "line counter should reset to 0");
    }

    #[test]
    fn test_ntriples_line_counter_accumulates_across_calls() {
        let mut p = NTriplesLiteParser::new();
        p.parse_str("<http://s> <http://p> <http://o> .")
            .expect("first parse_str should succeed");
        let line_after_first = p.line_count;
        // A second call continues counting from where it left off
        p.parse_str("<http://s2> <http://p2> <http://o2> .")
            .expect("second parse_str should succeed");
        assert!(
            p.line_count > line_after_first,
            "line counter should keep growing across parse_str calls"
        );
    }

    #[test]
    fn test_ntriples_error_unterminated_iri() {
        let mut p = NTriplesLiteParser::new();
        let result = p.parse_str("<http://s> <http://p> <http://o");
        assert!(result.is_err(), "unterminated IRI should fail");
        let err = result.expect_err("should be an error");
        assert!(
            err.message.contains("unterminated IRI") || err.message.contains('>'),
            "error message should mention IRI termination: {err}"
        );
    }

    #[test]
    fn test_ntriples_error_blank_node_empty_label() {
        let mut p = NTriplesLiteParser::new();
        // "_: " has a space immediately after "_:" — label is empty
        let result = p.parse_str("_: <http://p> <http://o> .");
        assert!(result.is_err(), "blank node with empty label should fail");
    }

    #[test]
    fn test_ntriples_error_unknown_escape_sequence() {
        let mut p = NTriplesLiteParser::new();
        // \q is not a valid escape sequence
        let result = p.parse_str("<http://s> <http://p> \"bad\\qescape\" .");
        assert!(result.is_err(), "unknown escape sequence should fail");
        let err = result.expect_err("should be an error");
        assert!(
            err.message.contains("unknown escape") || err.message.contains('q'),
            "error message should mention the bad escape: {err}"
        );
    }

    #[test]
    fn test_ntriples_empty_input_produces_no_triples() {
        let mut p = NTriplesLiteParser::new();
        let triples = p.parse_str("").expect("empty input should succeed");
        assert!(triples.is_empty(), "empty input should produce no triples");
    }

    #[test]
    fn test_ntriples_only_whitespace_and_comments() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("   \n# comment\n\n   \n")
            .expect("whitespace-only input should succeed");
        assert!(
            triples.is_empty(),
            "whitespace and comment-only input produces no triples"
        );
    }

    #[test]
    fn test_ntriples_literal_with_embedded_quote() {
        // Escaped double-quote inside a literal
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("<http://s> <http://p> \"say \\\"hello\\\"\" .")
            .expect("escaped quotes should parse successfully");
        assert_eq!(triples[0].2.value, "say \"hello\"");
    }

    #[test]
    fn test_ntriples_literal_with_backslash() {
        let mut p = NTriplesLiteParser::new();
        let triples = p
            .parse_str("<http://s> <http://p> \"back\\\\slash\" .")
            .expect("escaped backslash should parse successfully");
        assert_eq!(triples[0].2.value, "back\\slash");
    }

    #[test]
    fn test_ntriples_roundtrip_via_display() {
        // Parse a triple and re-serialize it using RdfTerm::Display, then re-parse
        let mut p = NTriplesLiteParser::new();
        let original = "<http://example.org/s> <http://example.org/p> \"hello\"@en .";
        let triples = p.parse_str(original).expect("initial parse should succeed");
        assert_eq!(triples.len(), 1);

        // Re-serialize using Display
        let (s, pred, o) = &triples[0];
        let serialized = format!("{s} {pred} {o} .");

        // Re-parse the serialized form
        let mut p2 = NTriplesLiteParser::new();
        let roundtrip = p2
            .parse_str(&serialized)
            .expect("roundtrip parse should succeed");
        assert_eq!(roundtrip.len(), 1);
        assert_eq!(roundtrip[0].0, triples[0].0);
        assert_eq!(roundtrip[0].1, triples[0].1);
        assert_eq!(roundtrip[0].2, triples[0].2);
    }

    // ── Additional NQuadsLiteParser tests ─────────────────────────────────

    #[test]
    fn test_nquads_reset_clears_line_counter() {
        let mut p = NQuadsLiteParser::new();
        p.parse_str("<http://s> <http://p> <http://o> <http://g> .")
            .expect("parse should succeed");
        assert_eq!(p.line_count, 1);
        p.reset();
        assert_eq!(p.line_count, 0, "reset should zero the line counter");
    }

    #[test]
    fn test_nquads_language_tagged_literal_in_quad() {
        let mut p = NQuadsLiteParser::new();
        let quads = p
            .parse_str("<http://s> <http://p> \"hola\"@es <http://g> .")
            .expect("lang-tagged literal in quad should parse");
        assert_eq!(quads.len(), 1);
        assert_eq!(
            quads[0].2.term_type,
            TermType::Literal {
                datatype: None,
                lang: Some("es".to_string())
            }
        );
        assert_eq!(quads[0].2.value, "hola");
    }

    #[test]
    fn test_nquads_typed_literal_in_quad() {
        let mut p = NQuadsLiteParser::new();
        let quads = p
            .parse_str(
                "<http://s> <http://p> \"3.14\"^^<http://www.w3.org/2001/XMLSchema#decimal> <http://g> .",
            )
            .expect("typed literal in quad should parse");
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].2.value, "3.14");
        assert_eq!(
            quads[0].2.term_type,
            TermType::Literal {
                datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
                lang: None,
            }
        );
    }

    #[test]
    fn test_nquads_error_missing_dot() {
        let mut p = NQuadsLiteParser::new();
        let result = p.parse_str("<http://s> <http://p> <http://o> <http://g>");
        assert!(result.is_err(), "missing dot in N-Quad should fail");
    }
}
