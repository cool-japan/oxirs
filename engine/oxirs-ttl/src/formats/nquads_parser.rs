//! W3C N-Quads parser and serialiser.
//!
//! Implements the N-Quads 1.1 grammar:
//! <https://www.w3.org/TR/n-quads/>

// ── Types ─────────────────────────────────────────────────────────────────────

/// An N-Quads term: IRI, blank node, or literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NTerm {
    /// An IRI term (e.g. `<http://example.org/>`).
    Iri(String),
    /// A blank node term (e.g. `_:b0`).
    BlankNode(String),
    /// A literal term with an optional datatype IRI or language tag.
    Literal {
        /// The lexical form of the literal.
        value: String,
        /// Optional datatype IRI (absolute, without angle brackets).
        datatype: Option<String>,
        /// Optional BCP-47 language tag.
        lang: Option<String>,
    },
}

impl NTerm {
    /// Serialize back to N-Quads notation.
    fn to_nquads_string(&self) -> String {
        match self {
            NTerm::Iri(iri) => format!("<{}>", iri),
            NTerm::BlankNode(label) => format!("_:{}", label),
            NTerm::Literal {
                value,
                datatype,
                lang,
            } => {
                let escaped = escape_string(value);
                if let Some(lang_tag) = lang {
                    format!("\"{}\"@{}", escaped, lang_tag)
                } else if let Some(dt) = datatype {
                    format!("\"{}\"^^<{}>", escaped, dt)
                } else {
                    format!("\"{}\"", escaped)
                }
            }
        }
    }
}

/// A single N-Quad.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NQuad {
    /// The subject term (IRI or blank node).
    pub subject: NTerm,
    /// The predicate term (must be an IRI).
    pub predicate: NTerm,
    /// The object term (IRI, blank node, or literal).
    pub object: NTerm,
    /// Optional graph name (IRI or blank node).
    pub graph: Option<NTerm>,
}

/// Errors that can occur during parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NQuadsError {
    /// An unparseable line (line number + message).
    InvalidLine(usize, String),
    /// An invalid IRI reference.
    InvalidIri(String),
    /// An invalid literal value.
    InvalidLiteral(String),
    /// An invalid blank-node label.
    InvalidBlankNode(String),
}

impl std::fmt::Display for NQuadsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NQuadsError::InvalidLine(n, msg) => write!(f, "Line {}: {}", n, msg),
            NQuadsError::InvalidIri(msg) => write!(f, "Invalid IRI: {}", msg),
            NQuadsError::InvalidLiteral(msg) => write!(f, "Invalid literal: {}", msg),
            NQuadsError::InvalidBlankNode(msg) => write!(f, "Invalid blank node: {}", msg),
        }
    }
}

impl std::error::Error for NQuadsError {}

// ── Parser ────────────────────────────────────────────────────────────────────

/// W3C N-Quads parser.
pub struct NQuadsParser;

impl NQuadsParser {
    /// Create a new parser instance.
    pub fn new() -> Self {
        NQuadsParser
    }

    /// Parse a complete N-Quads document.
    pub fn parse(&self, input: &str) -> Result<Vec<NQuad>, NQuadsError> {
        let mut quads = Vec::new();
        for (idx, raw_line) in input.lines().enumerate() {
            let line_num = idx + 1;
            if let Some(quad) = self.parse_line(raw_line, line_num)? {
                quads.push(quad);
            }
        }
        Ok(quads)
    }

    /// Parse a single line. Returns `None` for blank lines and comments.
    pub fn parse_line(&self, line: &str, line_num: usize) -> Result<Option<NQuad>, NQuadsError> {
        let trimmed = line.trim();
        // Empty line or comment.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return Ok(None);
        }

        let rest = trimmed;

        // Parse subject.
        let (subject, rest) = self
            .parse_term(rest)
            .map_err(|e| NQuadsError::InvalidLine(line_num, format!("subject: {}", e)))?;
        let rest = rest.trim_start();

        // Validate subject: must be IRI or blank node.
        match &subject {
            NTerm::Iri(_) | NTerm::BlankNode(_) => {}
            _ => {
                return Err(NQuadsError::InvalidLine(
                    line_num,
                    "subject must be IRI or blank node".to_string(),
                ))
            }
        }

        // Parse predicate.
        let (predicate, rest) = self
            .parse_term(rest)
            .map_err(|e| NQuadsError::InvalidLine(line_num, format!("predicate: {}", e)))?;
        let rest = rest.trim_start();

        // Validate predicate: must be IRI.
        match &predicate {
            NTerm::Iri(_) => {}
            _ => {
                return Err(NQuadsError::InvalidLine(
                    line_num,
                    "predicate must be an IRI".to_string(),
                ))
            }
        }

        // Parse object.
        let (object, rest) = self
            .parse_term(rest)
            .map_err(|e| NQuadsError::InvalidLine(line_num, format!("object: {}", e)))?;
        let rest = rest.trim_start();

        // Optional graph name.
        let (graph, rest) = if rest.starts_with('<') || rest.starts_with("_:") {
            let (g, r) = self
                .parse_term(rest)
                .map_err(|e| NQuadsError::InvalidLine(line_num, format!("graph: {}", e)))?;
            match &g {
                NTerm::Iri(_) | NTerm::BlankNode(_) => {}
                _ => {
                    return Err(NQuadsError::InvalidLine(
                        line_num,
                        "graph must be IRI or blank node".to_string(),
                    ))
                }
            }
            (Some(g), r)
        } else {
            (None, rest)
        };

        let rest = rest.trim_start();

        // Expect terminating `.`
        if !rest.starts_with('.') {
            return Err(NQuadsError::InvalidLine(
                line_num,
                format!("expected '.', found {:?}", rest),
            ));
        }

        Ok(Some(NQuad {
            subject,
            predicate,
            object,
            graph,
        }))
    }

    /// Parse one N-Quads term from the front of `s`.
    /// Returns `(term, remaining_input)`.
    pub fn parse_term<'a>(&self, s: &'a str) -> Result<(NTerm, &'a str), NQuadsError> {
        let s = s.trim_start();
        if s.is_empty() {
            return Err(NQuadsError::InvalidLine(
                0,
                "unexpected end of input".to_string(),
            ));
        }
        if s.starts_with('<') {
            let (iri, rest) = self.parse_iri(s)?;
            Ok((NTerm::Iri(iri), rest))
        } else if s.starts_with("_:") {
            let (label, rest) = self.parse_blank_node(s)?;
            Ok((NTerm::BlankNode(label), rest))
        } else if s.starts_with('"') {
            self.parse_literal(s)
        } else {
            Err(NQuadsError::InvalidLine(
                0,
                format!("unexpected token: {:?}", &s[..s.len().min(20)]),
            ))
        }
    }

    /// Parse `<iri>` from the front of `s`. Returns `(iri_string, rest)`.
    pub fn parse_iri<'a>(&self, s: &'a str) -> Result<(String, &'a str), NQuadsError> {
        if !s.starts_with('<') {
            return Err(NQuadsError::InvalidIri("expected '<'".to_string()));
        }
        let content = &s[1..];
        let end = content
            .find('>')
            .ok_or_else(|| NQuadsError::InvalidIri("unclosed IRI '<'".to_string()))?;
        let iri = unescape_iri(&content[..end])?;
        let rest = &content[end + 1..];
        Ok((iri, rest))
    }

    /// Parse `_:label` from the front of `s`. Returns `(label, rest)`.
    pub fn parse_blank_node<'a>(&self, s: &'a str) -> Result<(String, &'a str), NQuadsError> {
        if !s.starts_with("_:") {
            return Err(NQuadsError::InvalidBlankNode("expected '_:'".to_string()));
        }
        let label_start = &s[2..];
        // Label: [A-Za-z0-9_] then [A-Za-z0-9_\-.]*
        if label_start.is_empty() {
            return Err(NQuadsError::InvalidBlankNode(
                "empty blank node label".to_string(),
            ));
        }
        let first = label_start.chars().next().unwrap_or('\0');
        if !first.is_ascii_alphanumeric() && first != '_' {
            return Err(NQuadsError::InvalidBlankNode(format!(
                "invalid first char in blank node label: {:?}",
                first
            )));
        }
        let end = label_start
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '.')
            .unwrap_or(label_start.len());
        let label = &label_start[..end];
        // Strip trailing dots (per spec).
        let label = label.trim_end_matches('.');
        Ok((label.to_string(), &label_start[end..]))
    }

    /// Parse a string literal (plain, lang-tagged, or typed) from `s`.
    pub fn parse_literal<'a>(&self, s: &'a str) -> Result<(NTerm, &'a str), NQuadsError> {
        if !s.starts_with('"') {
            return Err(NQuadsError::InvalidLiteral("expected '\"'".to_string()));
        }

        // Find the closing unescaped quote.
        let content = &s[1..];
        let (raw_value, after_quote) = scan_string_content(content)?;
        let value = unescape_string(&raw_value)?;

        let rest = after_quote; // after the closing `"`

        if let Some(stripped_hat) = rest.strip_prefix("^^") {
            // Typed literal.
            let type_part = stripped_hat.trim_start();
            if !type_part.starts_with('<') {
                return Err(NQuadsError::InvalidLiteral(
                    "expected '<' after '^^'".to_string(),
                ));
            }
            let (datatype, remaining) = self.parse_iri(type_part)?;
            Ok((
                NTerm::Literal {
                    value,
                    datatype: Some(datatype),
                    lang: None,
                },
                remaining,
            ))
        } else if let Some(lang_start) = rest.strip_prefix('@') {
            // Language-tagged literal.
            let end = lang_start
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '-')
                .unwrap_or(lang_start.len());
            if end == 0 {
                return Err(NQuadsError::InvalidLiteral(
                    "empty language tag".to_string(),
                ));
            }
            let lang = &lang_start[..end];
            let remaining = &lang_start[end..];
            Ok((
                NTerm::Literal {
                    value,
                    datatype: None,
                    lang: Some(lang.to_string()),
                },
                remaining,
            ))
        } else {
            Ok((
                NTerm::Literal {
                    value,
                    datatype: None,
                    lang: None,
                },
                rest,
            ))
        }
    }

    /// Serialize a slice of N-Quads to an N-Quads document string.
    pub fn serialize(quads: &[NQuad]) -> String {
        let mut out = String::new();
        for quad in quads {
            out.push_str(&quad.subject.to_nquads_string());
            out.push(' ');
            out.push_str(&quad.predicate.to_nquads_string());
            out.push(' ');
            out.push_str(&quad.object.to_nquads_string());
            if let Some(g) = &quad.graph {
                out.push(' ');
                out.push_str(&g.to_nquads_string());
            }
            out.push_str(" .\n");
        }
        out
    }
}

impl Default for NQuadsParser {
    fn default() -> Self {
        Self::new()
    }
}

// ── String helpers ────────────────────────────────────────────────────────────

/// Scan a string literal body (after the opening `"`) until the closing `"`.
/// Returns `(raw_body, rest_after_closing_quote)`.
fn scan_string_content(s: &str) -> Result<(String, &str), NQuadsError> {
    scan_string_content_bytes(s)
}

/// Byte-indexed version that avoids the char_indices complexity above.
fn scan_string_content_bytes(s: &str) -> Result<(String, &str), NQuadsError> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut raw = String::new();
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                // Closing quote.
                return Ok((raw, &s[i + 1..]));
            }
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    return Err(NQuadsError::InvalidLiteral(
                        "trailing backslash".to_string(),
                    ));
                }
                raw.push('\\');
                raw.push(bytes[i] as char);
                i += 1;
            }
            _ => {
                // Multi-byte UTF-8: let str::chars handle it by finding the char.
                let ch_str = &s[i..];
                let ch = ch_str.chars().next().unwrap_or('\0');
                raw.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    Err(NQuadsError::InvalidLiteral(
        "unclosed string literal".to_string(),
    ))
}

/// Unescape escape sequences in a literal value.
fn unescape_string(s: &str) -> Result<String, NQuadsError> {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    let code = u32::from_str_radix(&hex, 16).map_err(|_| {
                        NQuadsError::InvalidLiteral(format!("invalid \\u escape: {}", hex))
                    })?;
                    let ch = char::from_u32(code).ok_or_else(|| {
                        NQuadsError::InvalidLiteral(format!("invalid codepoint: {}", code))
                    })?;
                    out.push(ch);
                }
                Some('U') => {
                    let hex: String = chars.by_ref().take(8).collect();
                    let code = u32::from_str_radix(&hex, 16).map_err(|_| {
                        NQuadsError::InvalidLiteral(format!("invalid \\U escape: {}", hex))
                    })?;
                    let ch = char::from_u32(code).ok_or_else(|| {
                        NQuadsError::InvalidLiteral(format!("invalid codepoint: {}", code))
                    })?;
                    out.push(ch);
                }
                Some(other) => {
                    return Err(NQuadsError::InvalidLiteral(format!(
                        "unknown escape: \\{}",
                        other
                    )))
                }
                None => {
                    return Err(NQuadsError::InvalidLiteral(
                        "trailing backslash".to_string(),
                    ))
                }
            }
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

/// Unescape IRI escape sequences (only `\uXXXX` and `\UXXXXXXXX`).
fn unescape_iri(s: &str) -> Result<String, NQuadsError> {
    // IRIs in N-Quads rarely contain backslash escapes but we handle the spec.
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    let code = u32::from_str_radix(&hex, 16).map_err(|_| {
                        NQuadsError::InvalidIri(format!("invalid \\u escape: {}", hex))
                    })?;
                    let ch = char::from_u32(code).ok_or_else(|| {
                        NQuadsError::InvalidIri(format!("invalid codepoint: {}", code))
                    })?;
                    out.push(ch);
                }
                Some('U') => {
                    let hex: String = chars.by_ref().take(8).collect();
                    let code = u32::from_str_radix(&hex, 16).map_err(|_| {
                        NQuadsError::InvalidIri(format!("invalid \\U escape: {}", hex))
                    })?;
                    let ch = char::from_u32(code).ok_or_else(|| {
                        NQuadsError::InvalidIri(format!("invalid codepoint: {}", code))
                    })?;
                    out.push(ch);
                }
                Some(other) => {
                    return Err(NQuadsError::InvalidIri(format!(
                        "unknown IRI escape: \\{}",
                        other
                    )))
                }
                None => return Err(NQuadsError::InvalidIri("trailing backslash".to_string())),
            }
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

/// Escape special characters for N-Quads string output.
fn escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> NQuadsParser {
        NQuadsParser::new()
    }

    // ── Empty / comment lines ─────────────────────────────────────────────

    #[test]
    fn test_empty_line() {
        let r = p().parse_line("", 1).expect("ok");
        assert!(r.is_none());
    }

    #[test]
    fn test_whitespace_only() {
        let r = p().parse_line("   \t  ", 1).expect("ok");
        assert!(r.is_none());
    }

    #[test]
    fn test_comment_line() {
        let r = p().parse_line("# this is a comment", 1).expect("ok");
        assert!(r.is_none());
    }

    // ── IRI parsing ───────────────────────────────────────────────────────

    #[test]
    fn test_parse_iri_basic() {
        let (iri, rest) = p().parse_iri("<http://example.org/>").expect("ok");
        assert_eq!(iri, "http://example.org/");
        assert!(rest.is_empty());
    }

    #[test]
    fn test_parse_iri_unclosed() {
        assert!(p().parse_iri("<http://unclosed").is_err());
    }

    #[test]
    fn test_parse_iri_no_bracket() {
        assert!(p().parse_iri("http://no-bracket").is_err());
    }

    // ── Blank node parsing ────────────────────────────────────────────────

    #[test]
    fn test_parse_blank_node_basic() {
        let (label, rest) = p().parse_blank_node("_:b0 ").expect("ok");
        assert_eq!(label, "b0");
        assert_eq!(rest.trim(), "");
    }

    #[test]
    fn test_parse_blank_node_alphanumeric() {
        let (label, _) = p().parse_blank_node("_:abc123").expect("ok");
        assert_eq!(label, "abc123");
    }

    #[test]
    fn test_parse_blank_node_missing_prefix() {
        assert!(p().parse_blank_node("noprefix").is_err());
    }

    #[test]
    fn test_parse_blank_node_empty_label() {
        assert!(p().parse_blank_node("_:").is_err());
    }

    // ── Literal parsing ───────────────────────────────────────────────────

    #[test]
    fn test_parse_plain_literal() {
        let (term, _) = p().parse_literal("\"hello\"").expect("ok");
        assert_eq!(
            term,
            NTerm::Literal {
                value: "hello".to_string(),
                datatype: None,
                lang: None
            }
        );
    }

    #[test]
    fn test_parse_lang_literal() {
        let (term, _) = p().parse_literal("\"hello\"@en").expect("ok");
        assert_eq!(
            term,
            NTerm::Literal {
                value: "hello".to_string(),
                datatype: None,
                lang: Some("en".to_string())
            }
        );
    }

    #[test]
    fn test_parse_typed_literal() {
        let (term, _) = p()
            .parse_literal("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>")
            .expect("ok");
        match term {
            NTerm::Literal {
                value, datatype, ..
            } => {
                assert_eq!(value, "42");
                assert_eq!(
                    datatype.expect("should succeed"),
                    "http://www.w3.org/2001/XMLSchema#integer"
                );
            }
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn test_parse_literal_with_escape() {
        let (term, _) = p().parse_literal("\"hello\\nworld\"").expect("ok");
        match term {
            NTerm::Literal { value, .. } => assert_eq!(value, "hello\nworld"),
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn test_parse_literal_unclosed() {
        assert!(p().parse_literal("\"unclosed").is_err());
    }

    // ── Full triple / quad parsing ────────────────────────────────────────

    #[test]
    fn test_simple_triple() {
        let line = "<http://s> <http://p> <http://o> .";
        let quad = p().parse_line(line, 1).expect("ok").expect("some");
        assert_eq!(quad.subject, NTerm::Iri("http://s".to_string()));
        assert_eq!(quad.predicate, NTerm::Iri("http://p".to_string()));
        assert_eq!(quad.object, NTerm::Iri("http://o".to_string()));
        assert!(quad.graph.is_none());
    }

    #[test]
    fn test_quad_with_graph() {
        let line = "<http://s> <http://p> <http://o> <http://g> .";
        let quad = p().parse_line(line, 1).expect("ok").expect("some");
        assert_eq!(quad.graph, Some(NTerm::Iri("http://g".to_string())));
    }

    #[test]
    fn test_blank_node_subject() {
        let line = "_:b0 <http://p> <http://o> .";
        let quad = p().parse_line(line, 1).expect("ok").expect("some");
        assert_eq!(quad.subject, NTerm::BlankNode("b0".to_string()));
    }

    #[test]
    fn test_literal_object() {
        let line = "<http://s> <http://p> \"Alice\" .";
        let quad = p().parse_line(line, 1).expect("ok").expect("some");
        assert_eq!(
            quad.object,
            NTerm::Literal {
                value: "Alice".to_string(),
                datatype: None,
                lang: None
            }
        );
    }

    #[test]
    fn test_lang_literal_object() {
        let line = "<http://s> <http://p> \"Alice\"@en .";
        let quad = p().parse_line(line, 1).expect("ok").expect("some");
        match &quad.object {
            NTerm::Literal { lang, .. } => assert_eq!(lang.as_deref(), Some("en")),
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn test_typed_literal_object() {
        let line = "<http://s> <http://p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .";
        let quad = p().parse_line(line, 1).expect("ok").expect("some");
        match &quad.object {
            NTerm::Literal { datatype, .. } => {
                assert!(datatype.is_some())
            }
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn test_missing_dot() {
        let line = "<http://s> <http://p> <http://o>";
        assert!(p().parse_line(line, 1).is_err());
    }

    // ── Multi-line document parsing ───────────────────────────────────────

    #[test]
    fn test_parse_document_multi_line() {
        let doc = "<http://s1> <http://p> <http://o1> .\n\
                   # comment\n\
                   <http://s2> <http://p> <http://o2> .\n";
        let quads = p().parse(doc).expect("ok");
        assert_eq!(quads.len(), 2);
    }

    #[test]
    fn test_parse_empty_document() {
        let quads = p().parse("").expect("ok");
        assert!(quads.is_empty());
    }

    // ── Serializer ────────────────────────────────────────────────────────

    #[test]
    fn test_serialize_basic() {
        let quads = vec![NQuad {
            subject: NTerm::Iri("http://s".to_string()),
            predicate: NTerm::Iri("http://p".to_string()),
            object: NTerm::Iri("http://o".to_string()),
            graph: None,
        }];
        let out = NQuadsParser::serialize(&quads);
        assert!(out.contains("<http://s>"));
        assert!(out.ends_with(".\n"));
    }

    #[test]
    fn test_serialize_with_graph() {
        let quads = vec![NQuad {
            subject: NTerm::Iri("http://s".to_string()),
            predicate: NTerm::Iri("http://p".to_string()),
            object: NTerm::Iri("http://o".to_string()),
            graph: Some(NTerm::Iri("http://g".to_string())),
        }];
        let out = NQuadsParser::serialize(&quads);
        assert!(out.contains("<http://g>"));
    }

    #[test]
    fn test_serialize_literal() {
        let quads = vec![NQuad {
            subject: NTerm::Iri("http://s".to_string()),
            predicate: NTerm::Iri("http://p".to_string()),
            object: NTerm::Literal {
                value: "hello".to_string(),
                datatype: None,
                lang: Some("en".to_string()),
            },
            graph: None,
        }];
        let out = NQuadsParser::serialize(&quads);
        assert!(out.contains("\"hello\"@en"));
    }

    // ── Roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_simple_quad() {
        let original = vec![NQuad {
            subject: NTerm::Iri("http://example.org/s".to_string()),
            predicate: NTerm::Iri("http://example.org/p".to_string()),
            object: NTerm::Literal {
                value: "test value".to_string(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                lang: None,
            },
            graph: Some(NTerm::Iri("http://example.org/g".to_string())),
        }];
        let serialized = NQuadsParser::serialize(&original);
        let parsed = p().parse(&serialized).expect("ok");
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_roundtrip_blank_node() {
        let original = vec![NQuad {
            subject: NTerm::BlankNode("node1".to_string()),
            predicate: NTerm::Iri("http://p".to_string()),
            object: NTerm::Iri("http://o".to_string()),
            graph: None,
        }];
        let serialized = NQuadsParser::serialize(&original);
        let parsed = p().parse(&serialized).expect("ok");
        assert_eq!(original, parsed);
    }

    // ── Error cases ───────────────────────────────────────────────────────

    #[test]
    fn test_error_literal_as_subject() {
        let line = "\"literal\" <http://p> <http://o> .";
        assert!(p().parse_line(line, 1).is_err());
    }

    #[test]
    fn test_error_literal_as_predicate() {
        let line = "<http://s> \"literal\" <http://o> .";
        assert!(p().parse_line(line, 1).is_err());
    }

    #[test]
    fn test_unicode_escape_in_literal() {
        let line = "<http://s> <http://p> \"caf\\u00E9\" .";
        let quad = p().parse_line(line, 1).expect("ok").expect("some");
        match &quad.object {
            NTerm::Literal { value, .. } => assert_eq!(value, "café"),
            _ => panic!("expected literal"),
        }
    }
}
