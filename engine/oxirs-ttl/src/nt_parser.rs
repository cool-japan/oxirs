//! N-Triples and N-Quads streaming parser (v1.1.0 round 14)
//!
//! Implements a strict N-Triples 1.1 and N-Quads 1.1 parser according to
//! the W3C specifications:
//!   - <https://www.w3.org/TR/n-triples/>
//!   - <https://www.w3.org/TR/n-quads/>
//!
//! Supports streaming line-by-line parsing as well as full document parsing.

use std::fmt;

/// A single token in an N-Triples / N-Quads document.
#[derive(Debug, Clone, PartialEq)]
pub enum NtToken {
    /// An IRI reference enclosed in angle brackets, e.g. `<http://example.org/foo>`
    Iri(String),
    /// A blank node, e.g. `_:b0`
    BlankNode(String),
    /// An RDF literal with an optional datatype IRI or language tag.
    Literal {
        /// The lexical form of the literal (the unescaped string between the quotes).
        value: String,
        /// The datatype IRI, e.g. `http://www.w3.org/2001/XMLSchema#integer`, if present.
        datatype: Option<String>,
        /// The BCP 47 language tag, e.g. `"en"` or `"en-US"`, if present.
        lang: Option<String>,
    },
    /// The statement-terminating dot `.`
    Dot,
    /// A named graph identifier (used in N-Quads)
    GraphName(String),
}

/// A single N-Triples statement.
#[derive(Debug, Clone, PartialEq)]
pub struct NtTriple {
    /// The subject term (IRI or blank node).
    pub subject: NtToken,
    /// The predicate term (must be an IRI).
    pub predicate: NtToken,
    /// The object term (IRI, blank node, or literal).
    pub object: NtToken,
}

/// A single N-Quads statement, optionally associated with a named graph.
#[derive(Debug, Clone, PartialEq)]
pub struct NtQuad {
    /// The subject-predicate-object triple component of the quad.
    pub triple: NtTriple,
    /// The optional named graph identifier (IRI or blank node mapped to a `GraphName` token).
    pub graph: Option<NtToken>,
}

/// Errors that can occur during N-Triples / N-Quads parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum NtError {
    /// The subject position contains a token type that is not allowed there (e.g. a literal).
    InvalidSubject,
    /// The predicate position contains a token type that is not allowed there (e.g. a blank node).
    InvalidPredicate,
    /// The object position contains a token type that is not allowed there.
    InvalidObject,
    /// The statement-terminating `.` was not found at the expected position.
    MissingDot,
    /// A literal token is syntactically malformed; the inner string describes the cause.
    InvalidLiteral(String),
    /// An IRI reference is syntactically malformed; the inner string describes the cause.
    InvalidIri(String),
    /// The input ended before a complete statement could be parsed.
    UnexpectedEof,
}

impl fmt::Display for NtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NtError::InvalidSubject => write!(f, "Invalid subject term"),
            NtError::InvalidPredicate => write!(f, "Invalid predicate term"),
            NtError::InvalidObject => write!(f, "Invalid object term"),
            NtError::MissingDot => write!(f, "Missing terminating dot '.'"),
            NtError::InvalidLiteral(msg) => write!(f, "Invalid literal: {msg}"),
            NtError::InvalidIri(msg) => write!(f, "Invalid IRI: {msg}"),
            NtError::UnexpectedEof => write!(f, "Unexpected end of input"),
        }
    }
}

impl std::error::Error for NtError {}

/// N-Triples and N-Quads streaming parser.
///
/// Each `NtParser` instance is stateless — all state is held per-call.
/// This makes the parser thread-safe and re-entrant.
#[derive(Debug, Default)]
pub struct NtParser;

impl NtParser {
    /// Create a new parser instance.
    pub fn new() -> Self {
        NtParser
    }

    /// Parse a single N-Triples line into an [`NtTriple`].
    ///
    /// Blank lines and lines starting with `#` return `Err(NtError::UnexpectedEof)`
    /// to indicate that callers should skip the line.  Use `parse_document` to
    /// handle multi-line documents transparently.
    pub fn parse_triple(&self, line: &str) -> Result<NtTriple, NtError> {
        let tokens = self.tokenize(line.trim())?;
        self.tokens_to_triple(&tokens)
    }

    /// Parse a single N-Quads line into an [`NtQuad`].
    pub fn parse_quad(&self, line: &str) -> Result<NtQuad, NtError> {
        let tokens = self.tokenize(line.trim())?;
        self.tokens_to_quad(&tokens)
    }

    /// Parse a complete N-Triples document (multiple lines, possibly with
    /// blank lines and comment lines starting with `#`).
    pub fn parse_document(&self, input: &str) -> Result<Vec<NtTriple>, NtError> {
        let mut triples = Vec::new();
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let triple = self.parse_triple(trimmed)?;
            triples.push(triple);
        }
        Ok(triples)
    }

    /// Parse a complete N-Quads document.
    pub fn parse_quads_document(&self, input: &str) -> Result<Vec<NtQuad>, NtError> {
        let mut quads = Vec::new();
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let quad = self.parse_quad(trimmed)?;
            quads.push(quad);
        }
        Ok(quads)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Tokenize one logical line, returning a sequence of raw tokens.
    fn tokenize(&self, input: &str) -> Result<Vec<RawToken>, NtError> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = input.chars().collect();
        let mut pos = 0usize;

        loop {
            // Skip whitespace
            while pos < chars.len() && chars[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if pos >= chars.len() {
                break;
            }

            match chars[pos] {
                '#' => {
                    // Rest of line is a comment — stop tokenizing
                    break;
                }
                '<' => {
                    let (iri, next_pos) = self.parse_iri(&chars, pos)?;
                    tokens.push(RawToken::Iri(iri));
                    pos = next_pos;
                }
                '_' if chars.get(pos + 1) == Some(&':') => {
                    let (bn, next_pos) = self.parse_blank_node(&chars, pos)?;
                    tokens.push(RawToken::BlankNode(bn));
                    pos = next_pos;
                }
                '"' => {
                    let (lit, next_pos) = self.parse_literal(&chars, pos)?;
                    tokens.push(RawToken::Literal(lit));
                    pos = next_pos;
                }
                '.' => {
                    tokens.push(RawToken::Dot);
                    pos += 1;
                }
                c => {
                    return Err(NtError::InvalidIri(format!(
                        "Unexpected character '{c}' at position {pos}"
                    )));
                }
            }
        }

        Ok(tokens)
    }

    /// Parse `<iri>` starting at `pos`.  Returns the IRI string and the next
    /// position after the closing `>`.
    fn parse_iri(&self, chars: &[char], start: usize) -> Result<(String, usize), NtError> {
        debug_assert_eq!(chars[start], '<');
        let mut pos = start + 1;
        let mut value = String::new();
        let mut escape_next = false;

        while pos < chars.len() {
            let c = chars[pos];
            if escape_next {
                match c {
                    'u' => {
                        // \uXXXX
                        let (ch, next) = self.parse_unicode_escape(chars, pos + 1, 4)?;
                        value.push(ch);
                        pos = next;
                        escape_next = false;
                        continue;
                    }
                    'U' => {
                        // \UXXXXXXXX
                        let (ch, next) = self.parse_unicode_escape(chars, pos + 1, 8)?;
                        value.push(ch);
                        pos = next;
                        escape_next = false;
                        continue;
                    }
                    _ => {
                        return Err(NtError::InvalidIri(format!(
                            "Unknown escape sequence '\\{c}' in IRI"
                        )));
                    }
                }
            }

            match c {
                '\\' => {
                    escape_next = true;
                    pos += 1;
                }
                '>' => {
                    // Closing bracket found
                    return Ok((value, pos + 1));
                }
                '\0'..='\x20' => {
                    return Err(NtError::InvalidIri(format!(
                        "Illegal character '{c:?}' inside IRI"
                    )));
                }
                _ => {
                    value.push(c);
                    pos += 1;
                }
            }
        }
        Err(NtError::UnexpectedEof)
    }

    /// Parse `_:label` starting at `pos`.
    fn parse_blank_node(&self, chars: &[char], start: usize) -> Result<(String, usize), NtError> {
        debug_assert_eq!(chars[start], '_');
        debug_assert_eq!(chars[start + 1], ':');
        let mut pos = start + 2;
        let mut label = String::new();

        while pos < chars.len() {
            let c = chars[pos];
            if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' {
                label.push(c);
                pos += 1;
            } else {
                break;
            }
        }

        if label.is_empty() {
            return Err(NtError::InvalidIri(
                "Blank node label must not be empty".to_string(),
            ));
        }
        // Trailing dots are not allowed
        if label.ends_with('.') {
            return Err(NtError::InvalidIri(
                "Blank node label must not end with a dot".to_string(),
            ));
        }
        Ok((label, pos))
    }

    /// Parse a string literal starting with `"` at `pos`.
    fn parse_literal(&self, chars: &[char], start: usize) -> Result<(RawLiteral, usize), NtError> {
        debug_assert_eq!(chars[start], '"');
        let mut pos = start + 1;
        let mut value = String::new();
        let mut escape_next = false;

        // Parse the quoted string
        loop {
            if pos >= chars.len() {
                return Err(NtError::UnexpectedEof);
            }
            let c = chars[pos];

            if escape_next {
                let escaped = match c {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '"' => '"',
                    '\'' => '\'',
                    '\\' => '\\',
                    'u' => {
                        let (ch, next) = self.parse_unicode_escape(chars, pos + 1, 4)?;
                        value.push(ch);
                        pos = next;
                        escape_next = false;
                        continue;
                    }
                    'U' => {
                        let (ch, next) = self.parse_unicode_escape(chars, pos + 1, 8)?;
                        value.push(ch);
                        pos = next;
                        escape_next = false;
                        continue;
                    }
                    _ => {
                        return Err(NtError::InvalidLiteral(format!(
                            "Unknown escape sequence '\\{c}'"
                        )));
                    }
                };
                value.push(escaped);
                pos += 1;
                escape_next = false;
                continue;
            }

            match c {
                '\\' => {
                    escape_next = true;
                    pos += 1;
                }
                '"' => {
                    pos += 1;
                    break;
                }
                '\n' | '\r' => {
                    return Err(NtError::InvalidLiteral(
                        "Unescaped newline in literal".to_string(),
                    ));
                }
                _ => {
                    value.push(c);
                    pos += 1;
                }
            }
        }

        // Check for datatype or language tag
        let mut datatype: Option<String> = None;
        let mut lang: Option<String> = None;

        // Skip whitespace between the closing quote and annotation
        // (N-Triples technically forbids this, but we stay lenient)
        if pos < chars.len() {
            match chars[pos] {
                '^' if chars.get(pos + 1) == Some(&'^') => {
                    pos += 2;
                    if pos >= chars.len() || chars[pos] != '<' {
                        return Err(NtError::InvalidLiteral(
                            "Expected '<' after '^^'".to_string(),
                        ));
                    }
                    let (iri, next_pos) = self.parse_iri(chars, pos)?;
                    datatype = Some(iri);
                    pos = next_pos;
                }
                '@' => {
                    pos += 1;
                    let mut tag = String::new();
                    while pos < chars.len() {
                        let c = chars[pos];
                        if c.is_alphanumeric() || c == '-' {
                            tag.push(c);
                            pos += 1;
                        } else {
                            break;
                        }
                    }
                    if tag.is_empty() {
                        return Err(NtError::InvalidLiteral(
                            "Language tag must not be empty".to_string(),
                        ));
                    }
                    lang = Some(tag);
                }
                _ => {}
            }
        }

        Ok((
            RawLiteral {
                value,
                datatype,
                lang,
            },
            pos,
        ))
    }

    /// Parse N hex digits starting at `pos` as a Unicode code point.
    fn parse_unicode_escape(
        &self,
        chars: &[char],
        start: usize,
        n: usize,
    ) -> Result<(char, usize), NtError> {
        if start + n > chars.len() {
            return Err(NtError::UnexpectedEof);
        }
        let hex: String = chars[start..start + n].iter().collect();
        let code = u32::from_str_radix(&hex, 16)
            .map_err(|_| NtError::InvalidLiteral(format!("Invalid Unicode escape: \\u{hex}")))?;
        let ch = char::from_u32(code).ok_or_else(|| {
            NtError::InvalidLiteral(format!("Invalid Unicode code point: U+{code:X}"))
        })?;
        Ok((ch, start + n))
    }

    /// Convert a flat token list to an [`NtTriple`].
    fn tokens_to_triple(&self, tokens: &[RawToken]) -> Result<NtTriple, NtError> {
        // Expected shape: subject predicate object .
        if tokens.len() < 4 {
            return Err(NtError::UnexpectedEof);
        }

        let subject = self.to_subject_token(&tokens[0])?;
        let predicate = self.to_predicate_token(&tokens[1])?;
        let object = self.to_object_token(&tokens[2])?;

        // 4th token must be a dot (trailing tokens after Dot are ignored — e.g. comments)
        if tokens[3] != RawToken::Dot {
            return Err(NtError::MissingDot);
        }

        Ok(NtTriple {
            subject,
            predicate,
            object,
        })
    }

    /// Convert a flat token list to an [`NtQuad`].
    fn tokens_to_quad(&self, tokens: &[RawToken]) -> Result<NtQuad, NtError> {
        if tokens.len() < 4 {
            return Err(NtError::UnexpectedEof);
        }

        let subject = self.to_subject_token(&tokens[0])?;
        let predicate = self.to_predicate_token(&tokens[1])?;
        let object = self.to_object_token(&tokens[2])?;

        // 4th token is either a graph name (IRI or blank node) or the dot.
        let graph: Option<NtToken>;
        let dot_idx: usize;

        match &tokens[3] {
            RawToken::Dot => {
                graph = None;
                dot_idx = 3;
            }
            RawToken::Iri(iri) => {
                graph = Some(NtToken::GraphName(iri.clone()));
                dot_idx = 4;
            }
            RawToken::BlankNode(bn) => {
                graph = Some(NtToken::GraphName(format!("_:{bn}")));
                dot_idx = 4;
            }
            _ => {
                return Err(NtError::MissingDot);
            }
        }

        if tokens.get(dot_idx) != Some(&RawToken::Dot) {
            return Err(NtError::MissingDot);
        }

        Ok(NtQuad {
            triple: NtTriple {
                subject,
                predicate,
                object,
            },
            graph,
        })
    }

    fn to_subject_token(&self, raw: &RawToken) -> Result<NtToken, NtError> {
        match raw {
            RawToken::Iri(s) => Ok(NtToken::Iri(s.clone())),
            RawToken::BlankNode(s) => Ok(NtToken::BlankNode(s.clone())),
            _ => Err(NtError::InvalidSubject),
        }
    }

    fn to_predicate_token(&self, raw: &RawToken) -> Result<NtToken, NtError> {
        match raw {
            RawToken::Iri(s) => Ok(NtToken::Iri(s.clone())),
            _ => Err(NtError::InvalidPredicate),
        }
    }

    fn to_object_token(&self, raw: &RawToken) -> Result<NtToken, NtError> {
        match raw {
            RawToken::Iri(s) => Ok(NtToken::Iri(s.clone())),
            RawToken::BlankNode(s) => Ok(NtToken::BlankNode(s.clone())),
            RawToken::Literal(lit) => Ok(NtToken::Literal {
                value: lit.value.clone(),
                datatype: lit.datatype.clone(),
                lang: lit.lang.clone(),
            }),
            _ => Err(NtError::InvalidObject),
        }
    }
}

// ---------------------------------------------------------------------------
// Private intermediate types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum RawToken {
    Iri(String),
    BlankNode(String),
    Literal(RawLiteral),
    Dot,
}

#[derive(Debug, Clone, PartialEq)]
struct RawLiteral {
    value: String,
    datatype: Option<String>,
    lang: Option<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> NtParser {
        NtParser::new()
    }

    // -- Basic triple parsing ------------------------------------------------

    #[test]
    fn test_basic_iri_triple() {
        let p = parser();
        let line = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .";
        let triple = p.parse_triple(line).expect("parse should succeed");
        assert_eq!(triple.subject, NtToken::Iri("http://example.org/s".into()));
        assert_eq!(
            triple.predicate,
            NtToken::Iri("http://example.org/p".into())
        );
        assert_eq!(triple.object, NtToken::Iri("http://example.org/o".into()));
    }

    #[test]
    fn test_simple_string_literal() {
        let p = parser();
        let line = r#"<http://example.org/s> <http://example.org/p> "hello" ."#;
        let triple = p.parse_triple(line).expect("parse should succeed");
        assert_eq!(
            triple.object,
            NtToken::Literal {
                value: "hello".into(),
                datatype: None,
                lang: None,
            }
        );
    }

    #[test]
    fn test_typed_literal_xsd_integer() {
        let p = parser();
        let line = r#"<http://a.example/s> <http://a.example/p> "42"^^<http://www.w3.org/2001/XMLSchema#integer> ."#;
        let triple = p.parse_triple(line).expect("parse should succeed");
        assert_eq!(
            triple.object,
            NtToken::Literal {
                value: "42".into(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
                lang: None,
            }
        );
    }

    #[test]
    fn test_typed_literal_xsd_string() {
        let p = parser();
        let line = r#"<http://a.example/s> <http://a.example/p> "text"^^<http://www.w3.org/2001/XMLSchema#string> ."#;
        let triple = p.parse_triple(line).expect("parse should succeed");
        if let NtToken::Literal {
            datatype: Some(dt), ..
        } = triple.object
        {
            assert!(dt.contains("XMLSchema#string"));
        } else {
            panic!("Expected typed literal");
        }
    }

    #[test]
    fn test_language_tagged_literal() {
        let p = parser();
        let line = r#"<http://a.example/s> <http://a.example/p> "Guten Tag"@de ."#;
        let triple = p.parse_triple(line).expect("parse should succeed");
        assert_eq!(
            triple.object,
            NtToken::Literal {
                value: "Guten Tag".into(),
                datatype: None,
                lang: Some("de".into()),
            }
        );
    }

    #[test]
    fn test_language_tagged_with_region() {
        let p = parser();
        let line = r#"<http://a.example/s> <http://a.example/p> "Hello"@en-US ."#;
        let triple = p.parse_triple(line).expect("parse should succeed");
        if let NtToken::Literal {
            lang: Some(tag), ..
        } = triple.object
        {
            assert_eq!(tag, "en-US");
        } else {
            panic!("Expected language tagged literal");
        }
    }

    #[test]
    fn test_blank_node_subject() {
        let p = parser();
        let line = r#"_:b0 <http://a.example/p> <http://a.example/o> ."#;
        let triple = p.parse_triple(line).expect("parse should succeed");
        assert_eq!(triple.subject, NtToken::BlankNode("b0".into()));
    }

    #[test]
    fn test_blank_node_object() {
        let p = parser();
        let line = r#"<http://a.example/s> <http://a.example/p> _:b1 ."#;
        let triple = p.parse_triple(line).expect("parse should succeed");
        assert_eq!(triple.object, NtToken::BlankNode("b1".into()));
    }

    #[test]
    fn test_both_blank_nodes() {
        let p = parser();
        let line = r#"_:alice <http://a.example/knows> _:bob ."#;
        let triple = p.parse_triple(line).expect("parse should succeed");
        assert_eq!(triple.subject, NtToken::BlankNode("alice".into()));
        assert_eq!(triple.object, NtToken::BlankNode("bob".into()));
    }

    #[test]
    fn test_unicode_escape_in_iri() {
        let p = parser();
        // \u0041 == 'A'
        let line = r"<http://example.org/\u0041> <http://example.org/p> <http://example.org/o> .";
        let triple = p.parse_triple(line).expect("parse should succeed");
        assert_eq!(triple.subject, NtToken::Iri("http://example.org/A".into()));
    }

    #[test]
    fn test_unicode_escape_in_literal() {
        let p = parser();
        let line = r#"<http://a/s> <http://a/p> "caf\u00E9" ."#;
        let triple = p.parse_triple(line).expect("parse should succeed");
        if let NtToken::Literal { value, .. } = triple.object {
            assert_eq!(value, "café");
        } else {
            panic!("Expected literal");
        }
    }

    #[test]
    fn test_escape_sequences_in_literal() {
        let p = parser();
        let line = r#"<http://a/s> <http://a/p> "line1\nline2\ttab" ."#;
        let triple = p.parse_triple(line).expect("parse should succeed");
        if let NtToken::Literal { value, .. } = triple.object {
            assert!(value.contains('\n'));
            assert!(value.contains('\t'));
        } else {
            panic!("Expected literal");
        }
    }

    #[test]
    fn test_escaped_quote_in_literal() {
        let p = parser();
        let line = r#"<http://a/s> <http://a/p> "say \"hello\"" ."#;
        let triple = p.parse_triple(line).expect("parse should succeed");
        if let NtToken::Literal { value, .. } = triple.object {
            assert_eq!(value, r#"say "hello""#);
        } else {
            panic!("Expected literal");
        }
    }

    // -- Comment / blank line skipping ---------------------------------------

    #[test]
    fn test_blank_lines_skipped() {
        let p = parser();
        let doc = "\n\n<http://a/s> <http://a/p> <http://a/o> .\n\n";
        let triples = p.parse_document(doc).expect("should parse");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_comment_lines_skipped() {
        let p = parser();
        let doc = "# This is a comment\n<http://a/s> <http://a/p> <http://a/o> .";
        let triples = p.parse_document(doc).expect("should parse");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_mixed_comments_and_triples() {
        let p = parser();
        let doc = concat!(
            "# comment\n",
            "<http://a/s1> <http://a/p> <http://a/o1> .\n",
            "# another\n",
            "<http://a/s2> <http://a/p> <http://a/o2> .\n",
        );
        let triples = p.parse_document(doc).expect("should parse");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_empty_document() {
        let p = parser();
        let triples = p.parse_document("").expect("should parse");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_comment_only_document() {
        let p = parser();
        let triples = p
            .parse_document("# just a comment\n")
            .expect("should parse");
        assert!(triples.is_empty());
    }

    // -- Error cases ---------------------------------------------------------

    #[test]
    fn test_missing_dot() {
        let p = parser();
        let line = "<http://a/s> <http://a/p> <http://a/o>";
        assert_eq!(p.parse_triple(line), Err(NtError::UnexpectedEof));
    }

    #[test]
    fn test_invalid_predicate_blank_node() {
        let p = parser();
        let line = "<http://a/s> _:b0 <http://a/o> .";
        assert_eq!(p.parse_triple(line), Err(NtError::InvalidPredicate));
    }

    #[test]
    fn test_invalid_predicate_literal() {
        let p = parser();
        let line = r#"<http://a/s> "literal" <http://a/o> ."#;
        assert_eq!(p.parse_triple(line), Err(NtError::InvalidPredicate));
    }

    #[test]
    fn test_invalid_subject_literal() {
        let p = parser();
        let line = r#""literal" <http://a/p> <http://a/o> ."#;
        assert_eq!(p.parse_triple(line), Err(NtError::InvalidSubject));
    }

    #[test]
    fn test_unclosed_iri() {
        let p = parser();
        let line = "<http://a/s <http://a/p> <http://a/o> .";
        // Should return an error (unexpected EOF or illegal char)
        assert!(p.parse_triple(line).is_err());
    }

    #[test]
    fn test_empty_lang_tag_error() {
        let p = parser();
        let line = r#"<http://a/s> <http://a/p> "text"@ ."#;
        assert!(p.parse_triple(line).is_err());
    }

    #[test]
    fn test_too_few_tokens() {
        let p = parser();
        let line = "<http://a/s> <http://a/p> .";
        assert!(p.parse_triple(line).is_err());
    }

    // -- N-Quads tests -------------------------------------------------------

    #[test]
    fn test_nquad_with_graph() {
        let p = parser();
        let line = "<http://a/s> <http://a/p> <http://a/o> <http://a/g> .";
        let quad = p.parse_quad(line).expect("should parse quad");
        assert_eq!(quad.graph, Some(NtToken::GraphName("http://a/g".into())));
        assert_eq!(quad.triple.subject, NtToken::Iri("http://a/s".into()));
    }

    #[test]
    fn test_nquad_without_graph() {
        let p = parser();
        let line = "<http://a/s> <http://a/p> <http://a/o> .";
        let quad = p.parse_quad(line).expect("should parse quad");
        assert!(quad.graph.is_none());
    }

    #[test]
    fn test_nquads_document_multiple() {
        let p = parser();
        let doc = concat!(
            "<http://a/s1> <http://a/p> <http://a/o1> <http://g1> .\n",
            "<http://a/s2> <http://a/p> <http://a/o2> <http://g2> .\n",
        );
        let quads = p.parse_quads_document(doc).expect("should parse");
        assert_eq!(quads.len(), 2);
    }

    #[test]
    fn test_nquad_blank_node_subject() {
        let p = parser();
        let line = "_:node1 <http://a/p> <http://a/o> <http://a/g> .";
        let quad = p.parse_quad(line).expect("should parse quad");
        assert_eq!(quad.triple.subject, NtToken::BlankNode("node1".into()));
    }

    #[test]
    fn test_nquad_literal_object() {
        let p = parser();
        let line = r#"<http://a/s> <http://a/p> "value"^^<http://www.w3.org/2001/XMLSchema#string> <http://a/g> ."#;
        let quad = p.parse_quad(line).expect("should parse quad");
        if let NtToken::Literal { value, .. } = &quad.triple.object {
            assert_eq!(value, "value");
        } else {
            panic!("Expected literal object");
        }
    }

    #[test]
    fn test_nquads_with_comments() {
        let p = parser();
        let doc = concat!(
            "# comment\n",
            "<http://a/s> <http://a/p> <http://a/o> <http://a/g> .\n",
            "\n",
        );
        let quads = p.parse_quads_document(doc).expect("should parse");
        assert_eq!(quads.len(), 1);
    }

    // -- Multi-triple documents ----------------------------------------------

    #[test]
    fn test_document_multiple_triples() {
        let p = parser();
        let doc = concat!(
            "<http://a/s1> <http://a/p1> <http://a/o1> .\n",
            "<http://a/s2> <http://a/p2> <http://a/o2> .\n",
            "<http://a/s3> <http://a/p3> <http://a/o3> .\n",
        );
        let triples = p.parse_document(doc).expect("should parse");
        assert_eq!(triples.len(), 3);
    }

    #[test]
    fn test_predicate_is_iri() {
        let p = parser();
        let line = "<http://a/s> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://a/o> .";
        let triple = p.parse_triple(line).expect("should parse");
        assert_eq!(
            triple.predicate,
            NtToken::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into())
        );
    }

    #[test]
    fn test_literal_backslash_escape() {
        let p = parser();
        let line = r#"<http://a/s> <http://a/p> "back\\slash" ."#;
        let triple = p.parse_triple(line).expect("should parse");
        if let NtToken::Literal { value, .. } = triple.object {
            assert_eq!(value, r"back\slash");
        } else {
            panic!("Expected literal");
        }
    }

    #[test]
    fn test_xsd_boolean_typed_literal() {
        let p = parser();
        let line =
            r#"<http://a/s> <http://a/p> "true"^^<http://www.w3.org/2001/XMLSchema#boolean> ."#;
        let triple = p.parse_triple(line).expect("should parse");
        if let NtToken::Literal {
            datatype: Some(dt),
            value,
            ..
        } = triple.object
        {
            assert_eq!(value, "true");
            assert!(dt.ends_with("#boolean"));
        } else {
            panic!("Expected typed literal");
        }
    }

    #[test]
    fn test_xsd_decimal_typed_literal() {
        let p = parser();
        let line =
            r#"<http://a/s> <http://a/p> "3.14"^^<http://www.w3.org/2001/XMLSchema#decimal> ."#;
        let triple = p.parse_triple(line).expect("should parse");
        if let NtToken::Literal { value, .. } = triple.object {
            assert_eq!(value, "3.14");
        } else {
            panic!("Expected typed literal");
        }
    }

    #[test]
    fn test_literal_with_carriage_return_escape() {
        let p = parser();
        let line = r#"<http://a/s> <http://a/p> "hello\rworld" ."#;
        let triple = p.parse_triple(line).expect("should parse");
        if let NtToken::Literal { value, .. } = triple.object {
            assert!(value.contains('\r'));
        } else {
            panic!("Expected literal");
        }
    }

    #[test]
    fn test_large_unicode_escape_in_literal() {
        let p = parser();
        let line = r#"<http://a/s> <http://a/p> "\U0001F600" ."#;
        let triple = p.parse_triple(line).expect("should parse");
        if let NtToken::Literal { value, .. } = triple.object {
            assert!(value.contains('\u{1F600}'));
        } else {
            panic!("Expected literal");
        }
    }

    #[test]
    fn test_blank_node_with_alphanumeric_label() {
        let p = parser();
        let line = "_:node123 <http://a/p> <http://a/o> .";
        let triple = p.parse_triple(line).expect("should parse");
        assert_eq!(triple.subject, NtToken::BlankNode("node123".into()));
    }

    #[test]
    fn test_blank_node_with_hyphen_label() {
        let p = parser();
        let line = "_:my-node <http://a/p> <http://a/o> .";
        let triple = p.parse_triple(line).expect("should parse");
        assert_eq!(triple.subject, NtToken::BlankNode("my-node".into()));
    }

    #[test]
    fn test_nttoken_dot_variant() {
        // Confirm Dot is a valid token
        let dot = NtToken::Dot;
        assert_eq!(format!("{dot:?}"), "Dot");
    }

    #[test]
    fn test_nterror_display_invalid_subject() {
        let err = NtError::InvalidSubject;
        assert!(err.to_string().contains("Invalid subject"));
    }

    #[test]
    fn test_nterror_display_invalid_literal() {
        let err = NtError::InvalidLiteral("bad".into());
        assert!(err.to_string().contains("bad"));
    }

    #[test]
    fn test_nterror_display_missing_dot() {
        let err = NtError::MissingDot;
        assert!(err.to_string().contains("dot"));
    }

    #[test]
    fn test_nterror_display_unexpected_eof() {
        let err = NtError::UnexpectedEof;
        assert!(err.to_string().contains("Unexpected"));
    }

    #[test]
    fn test_nterror_display_invalid_iri() {
        let err = NtError::InvalidIri("ctl".into());
        assert!(err.to_string().contains("IRI"));
    }

    #[test]
    fn test_inline_comment_after_dot() {
        // Comment after the dot should be silently ignored
        let p = parser();
        let line = "<http://a/s> <http://a/p> <http://a/o> . # comment";
        let triple = p.parse_triple(line).expect("should parse");
        assert_eq!(triple.subject, NtToken::Iri("http://a/s".into()));
    }

    #[test]
    fn test_nquad_missing_dot_returns_error() {
        let p = parser();
        let line = "<http://a/s> <http://a/p> <http://a/o> <http://a/g>";
        assert!(p.parse_quad(line).is_err());
    }

    #[test]
    fn test_nterror_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(NtError::UnexpectedEof);
        assert!(err.to_string().contains("Unexpected"));
    }

    #[test]
    fn test_parser_is_default() {
        let _p = NtParser;
    }

    #[test]
    fn test_many_triples_document() {
        let p = parser();
        let lines: Vec<String> = (0..50)
            .map(|i| format!("<http://a/s{i}> <http://a/p> <http://a/o{i}> ."))
            .collect();
        let doc = lines.join("\n");
        let triples = p.parse_document(&doc).expect("should parse");
        assert_eq!(triples.len(), 50);
    }

    #[test]
    fn test_nterror_invalid_predicate_display() {
        let err = NtError::InvalidPredicate;
        assert!(err.to_string().contains("predicate"));
    }

    #[test]
    fn test_nterror_invalid_object_display() {
        let err = NtError::InvalidObject;
        assert!(err.to_string().contains("object"));
    }
}
