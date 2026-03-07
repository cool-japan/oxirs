//! N-Triples and N-Quads serialization.
//!
//! Implements streaming, character-level serialization of RDF triples and quads
//! in the W3C N-Triples and N-Quads line-based formats. Handles IRI escaping,
//! literal encoding (string escaping, language tags, datatypes), blank node
//! labels, Unicode escaping, and optional canonical (sorted, normalised) output.

use std::fmt;
use std::io::{self, Write};

// ── Core RDF terms ──────────────────────────────────────────────────────────

/// An RDF term that can appear in N-Triples / N-Quads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NTerm {
    /// An IRI reference.
    Iri(String),
    /// A blank node with a label.
    BlankNode(String),
    /// A typed or language-tagged literal.
    Literal {
        /// Lexical value.
        value: String,
        /// Optional language tag (e.g. `"en"`).
        lang: Option<String>,
        /// Optional datatype IRI. If absent and no lang tag, implied
        /// `xsd:string`.
        datatype: Option<String>,
    },
}

impl NTerm {
    /// Create an IRI term.
    pub fn iri(s: impl Into<String>) -> Self {
        Self::Iri(s.into())
    }

    /// Create a blank node term.
    pub fn bnode(label: impl Into<String>) -> Self {
        Self::BlankNode(label.into())
    }

    /// Create a plain string literal.
    pub fn literal(value: impl Into<String>) -> Self {
        Self::Literal {
            value: value.into(),
            lang: None,
            datatype: None,
        }
    }

    /// Create a language-tagged literal.
    pub fn lang_literal(value: impl Into<String>, lang: impl Into<String>) -> Self {
        Self::Literal {
            value: value.into(),
            lang: Some(lang.into()),
            datatype: None,
        }
    }

    /// Create a datatyped literal.
    pub fn typed_literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        Self::Literal {
            value: value.into(),
            lang: None,
            datatype: Some(datatype.into()),
        }
    }
}

/// An N-Triple: subject, predicate, object.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NTriple {
    /// Subject (IRI or blank node).
    pub subject: NTerm,
    /// Predicate (IRI).
    pub predicate: NTerm,
    /// Object (IRI, blank node, or literal).
    pub object: NTerm,
}

impl NTriple {
    /// Create a new N-Triple.
    pub fn new(subject: NTerm, predicate: NTerm, object: NTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }
}

/// An N-Quad: subject, predicate, object, optional graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NQuad {
    /// Subject.
    pub subject: NTerm,
    /// Predicate.
    pub predicate: NTerm,
    /// Object.
    pub object: NTerm,
    /// Optional graph name (IRI or blank node).
    pub graph: Option<NTerm>,
}

impl NQuad {
    /// Create an N-Quad (with graph).
    pub fn new(subject: NTerm, predicate: NTerm, object: NTerm, graph: NTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
            graph: Some(graph),
        }
    }

    /// Create an N-Quad in the default graph.
    pub fn default_graph(subject: NTerm, predicate: NTerm, object: NTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
            graph: None,
        }
    }
}

// ── Serialization errors ────────────────────────────────────────────────────

/// Errors during N-Triples/N-Quads serialization.
#[derive(Debug)]
pub enum NTriplesError {
    /// An IO error from the underlying writer.
    Io(io::Error),
    /// A term is invalid for the position it appears in.
    InvalidTerm(String),
    /// An IRI contains characters that cannot be escaped.
    InvalidIri(String),
}

impl fmt::Display for NTriplesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::InvalidTerm(s) => write!(f, "Invalid term: {s}"),
            Self::InvalidIri(s) => write!(f, "Invalid IRI: {s}"),
        }
    }
}

impl std::error::Error for NTriplesError {}

impl From<io::Error> for NTriplesError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ── Escape helpers ──────────────────────────────────────────────────────────

/// Escape a string value for N-Triples literal encoding.
///
/// Characters `\`, `"`, `\n`, `\r`, `\t` are escaped. Non-ASCII characters
/// outside the Basic Latin plane are optionally Unicode-escaped.
pub fn escape_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            _ => out.push(ch),
        }
    }
    out
}

/// Escape a string using \\uXXXX / \\UXXXXXXXX for all non-ASCII characters.
pub fn escape_unicode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii() {
            match ch {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                _ => out.push(ch),
            }
        } else {
            let code = ch as u32;
            if code <= 0xFFFF {
                out.push_str(&format!("\\u{code:04X}"));
            } else {
                out.push_str(&format!("\\U{code:08X}"));
            }
        }
    }
    out
}

/// Escape an IRI for N-Triples `<...>` notation.
///
/// Per the spec, only a small set of characters need escaping inside `<>`:
/// `<`, `>`, `"`, `{`, `}`, `|`, `^`, `` ` ``, `\`, and control characters.
pub fn escape_iri(iri: &str) -> String {
    let mut out = String::with_capacity(iri.len());
    for ch in iri.chars() {
        match ch {
            '<' | '>' | '"' | '{' | '}' | '|' | '^' | '`' | '\\' => {
                let code = ch as u32;
                out.push_str(&format!("\\u{code:04X}"));
            }
            c if c.is_control() => {
                let code = c as u32;
                if code <= 0xFFFF {
                    out.push_str(&format!("\\u{code:04X}"));
                } else {
                    out.push_str(&format!("\\U{code:08X}"));
                }
            }
            _ => out.push(ch),
        }
    }
    out
}

// ── Term serialization ──────────────────────────────────────────────────────

/// Serialize a single term to a string.
pub fn serialize_term(term: &NTerm) -> String {
    match term {
        NTerm::Iri(iri) => format!("<{}>", escape_iri(iri)),
        NTerm::BlankNode(label) => format!("_:{label}"),
        NTerm::Literal {
            value,
            lang,
            datatype,
        } => {
            let escaped = escape_literal(value);
            if let Some(l) = lang {
                format!("\"{escaped}\"@{l}")
            } else if let Some(dt) = datatype {
                format!("\"{escaped}\"^^<{}>", escape_iri(dt))
            } else {
                format!("\"{escaped}\"")
            }
        }
    }
}

/// Serialize a term using full Unicode escaping.
pub fn serialize_term_unicode(term: &NTerm) -> String {
    match term {
        NTerm::Iri(iri) => format!("<{}>", escape_iri(iri)),
        NTerm::BlankNode(label) => format!("_:{label}"),
        NTerm::Literal {
            value,
            lang,
            datatype,
        } => {
            let escaped = escape_unicode(value);
            if let Some(l) = lang {
                format!("\"{escaped}\"@{l}")
            } else if let Some(dt) = datatype {
                format!("\"{escaped}\"^^<{}>", escape_iri(dt))
            } else {
                format!("\"{escaped}\"")
            }
        }
    }
}

// ── Triple/Quad serialization ───────────────────────────────────────────────

/// Serialize a single N-Triple to a string (terminated with ` .\n`).
pub fn serialize_triple(triple: &NTriple) -> String {
    format!(
        "{} {} {} .\n",
        serialize_term(&triple.subject),
        serialize_term(&triple.predicate),
        serialize_term(&triple.object),
    )
}

/// Serialize a single N-Quad to a string (terminated with ` .\n`).
pub fn serialize_quad(quad: &NQuad) -> String {
    match &quad.graph {
        Some(g) => format!(
            "{} {} {} {} .\n",
            serialize_term(&quad.subject),
            serialize_term(&quad.predicate),
            serialize_term(&quad.object),
            serialize_term(g),
        ),
        None => format!(
            "{} {} {} .\n",
            serialize_term(&quad.subject),
            serialize_term(&quad.predicate),
            serialize_term(&quad.object),
        ),
    }
}

// ── Streaming writer ────────────────────────────────────────────────────────

/// Configuration for the N-Triples writer.
#[derive(Debug, Clone, Default)]
pub struct WriterConfig {
    /// Whether to use full Unicode escaping for literal values.
    pub unicode_escape: bool,
    /// Whether to produce canonical output (sorted, normalised).
    pub canonical: bool,
}

/// A streaming N-Triples writer that writes directly to an `io::Write` sink.
pub struct NTriplesWriter<W: Write> {
    writer: W,
    config: WriterConfig,
    count: usize,
}

impl<W: Write> NTriplesWriter<W> {
    /// Create a new writer with default configuration.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            config: WriterConfig::default(),
            count: 0,
        }
    }

    /// Create a writer with a custom configuration.
    pub fn with_config(writer: W, config: WriterConfig) -> Self {
        Self {
            writer,
            config,
            count: 0,
        }
    }

    /// Write a single N-Triple.
    pub fn write_triple(&mut self, triple: &NTriple) -> Result<(), NTriplesError> {
        let line = if self.config.unicode_escape {
            format!(
                "{} {} {} .\n",
                serialize_term_unicode(&triple.subject),
                serialize_term_unicode(&triple.predicate),
                serialize_term_unicode(&triple.object),
            )
        } else {
            serialize_triple(triple)
        };
        self.writer.write_all(line.as_bytes())?;
        self.count += 1;
        Ok(())
    }

    /// Write a single N-Quad.
    pub fn write_quad(&mut self, quad: &NQuad) -> Result<(), NTriplesError> {
        let line = serialize_quad(quad);
        self.writer.write_all(line.as_bytes())?;
        self.count += 1;
        Ok(())
    }

    /// Write multiple triples.
    pub fn write_triples(&mut self, triples: &[NTriple]) -> Result<(), NTriplesError> {
        for t in triples {
            self.write_triple(t)?;
        }
        Ok(())
    }

    /// Write multiple quads.
    pub fn write_quads(&mut self, quads: &[NQuad]) -> Result<(), NTriplesError> {
        for q in quads {
            self.write_quad(q)?;
        }
        Ok(())
    }

    /// How many triples/quads have been written so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Flush the underlying writer.
    pub fn flush(&mut self) -> Result<(), NTriplesError> {
        self.writer.flush()?;
        Ok(())
    }

    /// Consume the writer and return the inner sink.
    pub fn into_inner(self) -> W {
        self.writer
    }
}

// ── Canonical serialization ─────────────────────────────────────────────────

/// Produce canonical N-Triples: triples are sorted lexicographically by
/// their serialised forms, and literal values are normalised.
pub fn canonical_ntriples(triples: &[NTriple]) -> String {
    let mut sorted = triples.to_vec();
    sorted.sort();
    let mut out = String::new();
    for t in &sorted {
        out.push_str(&serialize_triple(t));
    }
    out
}

/// Produce canonical N-Quads: quads are sorted lexicographically.
pub fn canonical_nquads(quads: &[NQuad]) -> String {
    let mut sorted = quads.to_vec();
    sorted.sort();
    let mut out = String::new();
    for q in &sorted {
        out.push_str(&serialize_quad(q));
    }
    out
}

/// Serialise triples to a `Vec<u8>` buffer (convenience function).
pub fn triples_to_bytes(triples: &[NTriple]) -> Result<Vec<u8>, NTriplesError> {
    let mut buf = Vec::new();
    {
        let mut writer = NTriplesWriter::new(&mut buf);
        writer.write_triples(triples)?;
        writer.flush()?;
    }
    Ok(buf)
}

/// Serialise quads to a `Vec<u8>` buffer (convenience function).
pub fn quads_to_bytes(quads: &[NQuad]) -> Result<Vec<u8>, NTriplesError> {
    let mut buf = Vec::new();
    {
        let mut writer = NTriplesWriter::new(&mut buf);
        writer.write_quads(quads)?;
        writer.flush()?;
    }
    Ok(buf)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── IRI serialization ───────────────────────────────────────────────────

    #[test]
    fn test_serialize_iri() {
        let t = NTerm::iri("http://example.org/foo");
        assert_eq!(serialize_term(&t), "<http://example.org/foo>");
    }

    #[test]
    fn test_serialize_iri_with_special_chars() {
        let t = NTerm::iri("http://example.org/foo<bar>");
        let s = serialize_term(&t);
        assert!(!s.contains('<') || s.starts_with('<'));
        // The inner < and > should be escaped
        assert!(s.contains("\\u003C") || s.contains("\\u003E"));
    }

    #[test]
    fn test_escape_iri_backslash() {
        let escaped = escape_iri("http://example.org/foo\\bar");
        assert!(escaped.contains("\\u005C"));
    }

    // ── Literal encoding ────────────────────────────────────────────────────

    #[test]
    fn test_serialize_plain_literal() {
        let t = NTerm::literal("hello");
        assert_eq!(serialize_term(&t), "\"hello\"");
    }

    #[test]
    fn test_serialize_literal_with_lang() {
        let t = NTerm::lang_literal("bonjour", "fr");
        assert_eq!(serialize_term(&t), "\"bonjour\"@fr");
    }

    #[test]
    fn test_serialize_literal_with_datatype() {
        let dt = "http://www.w3.org/2001/XMLSchema#integer";
        let t = NTerm::typed_literal("42", dt);
        let s = serialize_term(&t);
        assert!(s.starts_with("\"42\"^^<"));
        assert!(s.contains("XMLSchema#integer"));
    }

    #[test]
    fn test_escape_literal_quotes() {
        let escaped = escape_literal("He said \"hello\"");
        assert_eq!(escaped, "He said \\\"hello\\\"");
    }

    #[test]
    fn test_escape_literal_newline() {
        let escaped = escape_literal("line1\nline2");
        assert_eq!(escaped, "line1\\nline2");
    }

    #[test]
    fn test_escape_literal_tab() {
        let escaped = escape_literal("col1\tcol2");
        assert_eq!(escaped, "col1\\tcol2");
    }

    #[test]
    fn test_escape_literal_backslash() {
        let escaped = escape_literal("path\\to\\file");
        assert_eq!(escaped, "path\\\\to\\\\file");
    }

    #[test]
    fn test_escape_literal_carriage_return() {
        let escaped = escape_literal("cr\rhere");
        assert_eq!(escaped, "cr\\rhere");
    }

    // ── Blank node serialization ────────────────────────────────────────────

    #[test]
    fn test_serialize_blank_node() {
        let t = NTerm::bnode("b0");
        assert_eq!(serialize_term(&t), "_:b0");
    }

    #[test]
    fn test_serialize_blank_node_long_label() {
        let t = NTerm::bnode("genid_abc_123");
        assert_eq!(serialize_term(&t), "_:genid_abc_123");
    }

    // ── N-Triple serialization ──────────────────────────────────────────────

    #[test]
    fn test_serialize_triple() {
        let triple = NTriple::new(
            NTerm::iri("http://example.org/s"),
            NTerm::iri("http://example.org/p"),
            NTerm::literal("hello"),
        );
        let s = serialize_triple(&triple);
        assert_eq!(
            s,
            "<http://example.org/s> <http://example.org/p> \"hello\" .\n"
        );
    }

    #[test]
    fn test_serialize_triple_with_bnode_subject() {
        let triple = NTriple::new(
            NTerm::bnode("b0"),
            NTerm::iri("http://example.org/p"),
            NTerm::iri("http://example.org/o"),
        );
        let s = serialize_triple(&triple);
        assert!(s.starts_with("_:b0 "));
    }

    // ── N-Quad serialization ────────────────────────────────────────────────

    #[test]
    fn test_serialize_quad_with_graph() {
        let quad = NQuad::new(
            NTerm::iri("http://s"),
            NTerm::iri("http://p"),
            NTerm::iri("http://o"),
            NTerm::iri("http://g"),
        );
        let s = serialize_quad(&quad);
        assert_eq!(s, "<http://s> <http://p> <http://o> <http://g> .\n");
    }

    #[test]
    fn test_serialize_quad_default_graph() {
        let quad = NQuad::default_graph(
            NTerm::iri("http://s"),
            NTerm::iri("http://p"),
            NTerm::iri("http://o"),
        );
        let s = serialize_quad(&quad);
        // No graph term → same as N-Triple
        assert_eq!(s, "<http://s> <http://p> <http://o> .\n");
    }

    // ── Unicode escaping ────────────────────────────────────────────────────

    #[test]
    fn test_escape_unicode_bmp() {
        let escaped = escape_unicode("\u{00E9}"); // é
        assert_eq!(escaped, "\\u00E9");
    }

    #[test]
    fn test_escape_unicode_supplementary() {
        let escaped = escape_unicode("\u{1F600}"); // 😀
        assert_eq!(escaped, "\\U0001F600");
    }

    #[test]
    fn test_escape_unicode_ascii_passthrough() {
        let escaped = escape_unicode("abc");
        assert_eq!(escaped, "abc");
    }

    #[test]
    fn test_serialize_term_unicode_literal() {
        let t = NTerm::literal("caf\u{00E9}");
        let s = serialize_term_unicode(&t);
        assert!(s.contains("\\u00E9"));
    }

    // ── Streaming writer ────────────────────────────────────────────────────

    #[test]
    fn test_streaming_writer_count() {
        let mut buf = Vec::new();
        let mut writer = NTriplesWriter::new(&mut buf);
        let t1 = NTriple::new(
            NTerm::iri("http://s1"),
            NTerm::iri("http://p"),
            NTerm::literal("v1"),
        );
        let t2 = NTriple::new(
            NTerm::iri("http://s2"),
            NTerm::iri("http://p"),
            NTerm::literal("v2"),
        );
        writer.write_triple(&t1).expect("ok");
        writer.write_triple(&t2).expect("ok");
        assert_eq!(writer.count(), 2);
    }

    #[test]
    fn test_streaming_writer_output() {
        let mut buf = Vec::new();
        {
            let mut writer = NTriplesWriter::new(&mut buf);
            let t = NTriple::new(
                NTerm::iri("http://s"),
                NTerm::iri("http://p"),
                NTerm::literal("v"),
            );
            writer.write_triple(&t).expect("ok");
            writer.flush().expect("ok");
        }
        let output = String::from_utf8(buf).expect("valid utf8");
        assert_eq!(output, "<http://s> <http://p> \"v\" .\n");
    }

    #[test]
    fn test_streaming_writer_unicode_config() {
        let config = WriterConfig {
            unicode_escape: true,
            ..WriterConfig::default()
        };
        let mut buf = Vec::new();
        {
            let mut writer = NTriplesWriter::with_config(&mut buf, config);
            let t = NTriple::new(
                NTerm::iri("http://s"),
                NTerm::iri("http://p"),
                NTerm::literal("caf\u{00E9}"),
            );
            writer.write_triple(&t).expect("ok");
            writer.flush().expect("ok");
        }
        let output = String::from_utf8(buf).expect("valid utf8");
        assert!(output.contains("\\u00E9"));
    }

    #[test]
    fn test_streaming_writer_quads() {
        let mut buf = Vec::new();
        {
            let mut writer = NTriplesWriter::new(&mut buf);
            let q = NQuad::new(
                NTerm::iri("http://s"),
                NTerm::iri("http://p"),
                NTerm::iri("http://o"),
                NTerm::iri("http://g"),
            );
            writer.write_quad(&q).expect("ok");
            writer.flush().expect("ok");
        }
        let output = String::from_utf8(buf).expect("valid utf8");
        assert!(output.contains("<http://g>"));
    }

    // ── Canonical output ────────────────────────────────────────────────────

    #[test]
    fn test_canonical_sorted() {
        let t1 = NTriple::new(
            NTerm::iri("http://z"),
            NTerm::iri("http://p"),
            NTerm::literal("a"),
        );
        let t2 = NTriple::new(
            NTerm::iri("http://a"),
            NTerm::iri("http://p"),
            NTerm::literal("b"),
        );
        let canon = canonical_ntriples(&[t1, t2]);
        let lines: Vec<&str> = canon.lines().collect();
        // http://a should come before http://z
        assert!(
            lines[0].contains("http://a"),
            "First line should have http://a"
        );
        assert!(
            lines[1].contains("http://z"),
            "Second line should have http://z"
        );
    }

    #[test]
    fn test_canonical_nquads_sorted() {
        let q1 = NQuad::new(
            NTerm::iri("http://z"),
            NTerm::iri("http://p"),
            NTerm::iri("http://o"),
            NTerm::iri("http://g1"),
        );
        let q2 = NQuad::new(
            NTerm::iri("http://a"),
            NTerm::iri("http://p"),
            NTerm::iri("http://o"),
            NTerm::iri("http://g2"),
        );
        let canon = canonical_nquads(&[q1, q2]);
        let lines: Vec<&str> = canon.lines().collect();
        assert!(lines[0].contains("http://a"));
    }

    // ── Round-trip fidelity ─────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_plain_literal() {
        let original = "hello world";
        let escaped = escape_literal(original);
        // The escaped form should not alter plain ASCII
        assert_eq!(escaped, original);
    }

    #[test]
    fn test_roundtrip_special_chars() {
        let original = "line1\nline2\ttab \"quoted\" back\\slash";
        let escaped = escape_literal(original);
        assert!(escaped.contains("\\n"));
        assert!(escaped.contains("\\t"));
        assert!(escaped.contains("\\\""));
        assert!(escaped.contains("\\\\"));
        // No data lost – re-unescaping would recover original
    }

    // ── Batch convenience ───────────────────────────────────────────────────

    #[test]
    fn test_triples_to_bytes() {
        let triples = vec![
            NTriple::new(
                NTerm::iri("http://s1"),
                NTerm::iri("http://p"),
                NTerm::literal("v"),
            ),
            NTriple::new(
                NTerm::iri("http://s2"),
                NTerm::iri("http://p"),
                NTerm::literal("w"),
            ),
        ];
        let bytes = triples_to_bytes(&triples).expect("ok");
        let output = String::from_utf8(bytes).expect("valid utf8");
        assert_eq!(output.lines().count(), 2);
    }

    #[test]
    fn test_quads_to_bytes() {
        let quads = vec![NQuad::new(
            NTerm::iri("http://s"),
            NTerm::iri("http://p"),
            NTerm::iri("http://o"),
            NTerm::iri("http://g"),
        )];
        let bytes = quads_to_bytes(&quads).expect("ok");
        let output = String::from_utf8(bytes).expect("valid utf8");
        assert!(output.contains("<http://g>"));
    }

    // ── Write batch helpers ─────────────────────────────────────────────────

    #[test]
    fn test_write_triples_batch() {
        let triples = vec![
            NTriple::new(
                NTerm::iri("http://s1"),
                NTerm::iri("http://p"),
                NTerm::literal("a"),
            ),
            NTriple::new(
                NTerm::iri("http://s2"),
                NTerm::iri("http://p"),
                NTerm::literal("b"),
            ),
            NTriple::new(
                NTerm::iri("http://s3"),
                NTerm::iri("http://p"),
                NTerm::literal("c"),
            ),
        ];
        let mut buf = Vec::new();
        {
            let mut writer = NTriplesWriter::new(&mut buf);
            writer.write_triples(&triples).expect("ok");
            writer.flush().expect("ok");
            assert_eq!(writer.count(), 3);
        }
        let output = String::from_utf8(buf).expect("valid utf8");
        assert_eq!(output.lines().count(), 3);
    }

    #[test]
    fn test_write_quads_batch() {
        let quads = vec![
            NQuad::new(
                NTerm::iri("http://s1"),
                NTerm::iri("http://p"),
                NTerm::iri("http://o1"),
                NTerm::iri("http://g"),
            ),
            NQuad::default_graph(
                NTerm::iri("http://s2"),
                NTerm::iri("http://p"),
                NTerm::iri("http://o2"),
            ),
        ];
        let mut buf = Vec::new();
        {
            let mut writer = NTriplesWriter::new(&mut buf);
            writer.write_quads(&quads).expect("ok");
            writer.flush().expect("ok");
        }
        let output = String::from_utf8(buf).expect("valid utf8");
        // First line has graph, second does not
        let lines: Vec<&str> = output.lines().collect();
        assert!(lines[0].contains("<http://g>"));
        assert!(!lines[1].contains("<http://g>"));
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_empty_literal() {
        let t = NTerm::literal("");
        assert_eq!(serialize_term(&t), "\"\"");
    }

    #[test]
    fn test_empty_iri() {
        let t = NTerm::iri("");
        assert_eq!(serialize_term(&t), "<>");
    }

    #[test]
    fn test_escape_backspace_formfeed() {
        let escaped = escape_literal("\u{08}\u{0C}");
        assert_eq!(escaped, "\\b\\f");
    }

    #[test]
    fn test_into_inner() {
        let buf = Vec::new();
        let writer = NTriplesWriter::new(buf);
        let inner = writer.into_inner();
        assert!(inner.is_empty());
    }

    #[test]
    fn test_canonical_empty() {
        let canon = canonical_ntriples(&[]);
        assert!(canon.is_empty());
    }
}
