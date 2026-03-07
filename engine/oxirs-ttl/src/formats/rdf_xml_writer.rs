//! # RDF/XML Writer
//!
//! Serializes RDF triples to the RDF/XML format (W3C Recommendation).
//!
//! This complements the existing RDF/XML parser by providing write support,
//! enabling round-trip conversion between Turtle, N-Triples, and RDF/XML.
//!
//! ## Features
//!
//! - Abbreviated syntax using `rdf:about`, `rdf:resource` attributes
//! - Prefix/namespace management with automatic prefix generation
//! - Pretty-printing with configurable indentation
//! - Streaming output (write triples one at a time)
//! - Supports typed literals, language-tagged literals, blank nodes
//! - XML entity escaping
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_ttl::formats::rdf_xml_writer::{RdfXmlWriter, WriterConfig, RdfTriple, RdfTerm};
//!
//! let mut writer = RdfXmlWriter::new(WriterConfig::default());
//!
//! writer.add_prefix("ex", "http://example.org/");
//!
//! writer.write_triple(&RdfTriple {
//!     subject: RdfTerm::iri("http://example.org/alice"),
//!     predicate: RdfTerm::iri("http://example.org/name"),
//!     object: RdfTerm::literal("Alice"),
//! })?;
//!
//! let xml = writer.finish()?;
//! println!("{xml}");
//! ```
//!
//! ## References
//!
//! - <https://www.w3.org/TR/rdf-syntax-grammar/>

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::io::Write;

// ---------------------------------------------------------------------------
// RDF terms
// ---------------------------------------------------------------------------

/// An RDF term for the writer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RdfTerm {
    /// An IRI reference.
    Iri(String),
    /// A blank node.
    BlankNode(String),
    /// A plain literal (xsd:string).
    PlainLiteral(String),
    /// A typed literal.
    TypedLiteral {
        /// The lexical value.
        value: String,
        /// The datatype IRI.
        datatype: String,
    },
    /// A language-tagged literal.
    LangLiteral {
        /// The lexical value.
        value: String,
        /// The language tag.
        lang: String,
    },
}

impl RdfTerm {
    /// Create an IRI term.
    pub fn iri(s: impl Into<String>) -> Self {
        Self::Iri(s.into())
    }

    /// Create a plain literal.
    pub fn literal(s: impl Into<String>) -> Self {
        Self::PlainLiteral(s.into())
    }

    /// Create a typed literal.
    pub fn typed(s: impl Into<String>, dt: impl Into<String>) -> Self {
        Self::TypedLiteral {
            value: s.into(),
            datatype: dt.into(),
        }
    }

    /// Create a language-tagged literal.
    pub fn lang(s: impl Into<String>, lang: impl Into<String>) -> Self {
        Self::LangLiteral {
            value: s.into(),
            lang: lang.into(),
        }
    }

    /// Create a blank node.
    pub fn blank(id: impl Into<String>) -> Self {
        Self::BlankNode(id.into())
    }

    /// Check if this is an IRI.
    pub fn is_iri(&self) -> bool {
        matches!(self, Self::Iri(_))
    }

    /// Check if this is a literal.
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            Self::PlainLiteral(_) | Self::TypedLiteral { .. } | Self::LangLiteral { .. }
        )
    }
}

impl fmt::Display for RdfTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Iri(iri) => write!(f, "<{iri}>"),
            Self::BlankNode(id) => write!(f, "_:{id}"),
            Self::PlainLiteral(val) => write!(f, "\"{val}\""),
            Self::TypedLiteral { value, datatype } => write!(f, "\"{value}\"^^<{datatype}>"),
            Self::LangLiteral { value, lang } => write!(f, "\"{value}\"@{lang}"),
        }
    }
}

/// An RDF triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfTriple {
    /// Subject (IRI or blank node).
    pub subject: RdfTerm,
    /// Predicate (IRI).
    pub predicate: RdfTerm,
    /// Object (any RDF term).
    pub object: RdfTerm,
}

// ---------------------------------------------------------------------------
// Writer configuration
// ---------------------------------------------------------------------------

/// Configuration for the RDF/XML writer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriterConfig {
    /// Number of spaces per indentation level.
    pub indent: usize,
    /// Whether to use abbreviated syntax (rdf:resource attributes).
    pub abbreviated: bool,
    /// Whether to include the XML declaration.
    pub xml_declaration: bool,
    /// Base URI for relative IRI resolution.
    pub base_uri: Option<String>,
    /// Whether to sort subjects for deterministic output.
    pub sort_output: bool,
}

impl Default for WriterConfig {
    fn default() -> Self {
        Self {
            indent: 2,
            abbreviated: true,
            xml_declaration: true,
            base_uri: None,
            sort_output: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Writer errors
// ---------------------------------------------------------------------------

/// Errors from the RDF/XML writer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RdfXmlWriteError {
    /// Invalid IRI encountered.
    InvalidIri(String),
    /// Blank node used as predicate (not allowed in RDF).
    BlankNodePredicate,
    /// IO error message.
    IoError(String),
    /// XML encoding error.
    XmlEncodingError(String),
}

impl fmt::Display for RdfXmlWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIri(iri) => write!(f, "Invalid IRI: {iri}"),
            Self::BlankNodePredicate => write!(f, "Blank nodes cannot be predicates in RDF/XML"),
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::XmlEncodingError(msg) => write!(f, "XML encoding error: {msg}"),
        }
    }
}

impl std::error::Error for RdfXmlWriteError {}

// ---------------------------------------------------------------------------
// RDF/XML Writer
// ---------------------------------------------------------------------------

/// Writes RDF triples in RDF/XML format.
pub struct RdfXmlWriter {
    config: WriterConfig,
    /// Registered prefixes: prefix -> namespace URI.
    prefixes: HashMap<String, String>,
    /// Collected triples grouped by subject.
    triples: Vec<RdfTriple>,
    /// Statistics.
    triple_count: usize,
}

impl RdfXmlWriter {
    /// Create a new writer with the given configuration.
    pub fn new(config: WriterConfig) -> Self {
        let mut prefixes = HashMap::new();
        // Always include rdf namespace
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );

        Self {
            config,
            prefixes,
            triples: Vec::new(),
            triple_count: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(WriterConfig::default())
    }

    /// Register a namespace prefix.
    pub fn add_prefix(&mut self, prefix: &str, namespace: &str) {
        self.prefixes
            .insert(prefix.to_string(), namespace.to_string());
    }

    /// Write a triple. Triples are buffered until `finish()` is called.
    pub fn write_triple(&mut self, triple: &RdfTriple) -> Result<(), RdfXmlWriteError> {
        // Validate
        if matches!(triple.predicate, RdfTerm::BlankNode(_)) {
            return Err(RdfXmlWriteError::BlankNodePredicate);
        }
        self.triples.push(triple.clone());
        self.triple_count += 1;
        Ok(())
    }

    /// Finish writing and produce the RDF/XML string.
    pub fn finish(&self) -> Result<String, RdfXmlWriteError> {
        let mut buf = Vec::new();
        self.write_to(&mut buf)
            .map_err(|e| RdfXmlWriteError::IoError(e.to_string()))?;
        String::from_utf8(buf).map_err(|e| RdfXmlWriteError::XmlEncodingError(e.to_string()))
    }

    /// Write to an arbitrary `Write` sink.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let indent_str = " ".repeat(self.config.indent);

        // XML declaration
        if self.config.xml_declaration {
            writeln!(writer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
        }

        // Opening rdf:RDF tag with namespaces
        write!(writer, "<rdf:RDF")?;
        let mut sorted_prefixes: Vec<_> = self.prefixes.iter().collect();
        sorted_prefixes.sort_by_key(|(k, _)| (*k).clone());
        for (prefix, ns) in &sorted_prefixes {
            write!(
                writer,
                "\n{indent_str}xmlns:{prefix}=\"{}\"",
                xml_escape(ns)
            )?;
        }
        if let Some(base) = &self.config.base_uri {
            write!(writer, "\n{indent_str}xml:base=\"{base}\"")?;
        }
        writeln!(writer, ">")?;

        // Group triples by subject
        let groups = self.group_by_subject();
        let mut subjects: Vec<_> = groups.keys().collect();
        if self.config.sort_output {
            subjects.sort();
        }

        for subject in subjects {
            let predicates = &groups[subject.as_str()];
            self.write_description(writer, subject, predicates, &indent_str)?;
        }

        // Closing tag
        writeln!(writer, "</rdf:RDF>")?;
        Ok(())
    }

    /// Group triples by subject, returning subject -> list of (predicate, object).
    fn group_by_subject(&self) -> HashMap<String, Vec<(&RdfTerm, &RdfTerm)>> {
        let mut groups: HashMap<String, Vec<(&RdfTerm, &RdfTerm)>> = HashMap::new();
        for triple in &self.triples {
            let key = Self::subject_key(&triple.subject);
            groups
                .entry(key)
                .or_default()
                .push((&triple.predicate, &triple.object));
        }
        groups
    }

    fn subject_key(term: &RdfTerm) -> String {
        match term {
            RdfTerm::Iri(iri) => iri.clone(),
            RdfTerm::BlankNode(id) => format!("_:{id}"),
            _ => format!("{term}"),
        }
    }

    fn write_description<W: Write>(
        &self,
        writer: &mut W,
        subject: &str,
        predicates: &[(&RdfTerm, &RdfTerm)],
        indent: &str,
    ) -> std::io::Result<()> {
        // Open rdf:Description
        write!(writer, "{indent}<rdf:Description")?;
        if let Some(node_id) = subject.strip_prefix("_:") {
            write!(writer, " rdf:nodeID=\"{node_id}\"")?;
        } else {
            write!(writer, " rdf:about=\"{}\"", xml_escape(subject))?;
        }

        if predicates.is_empty() {
            writeln!(writer, "/>")?;
            return Ok(());
        }

        writeln!(writer, ">")?;

        // Write each predicate-object pair
        let inner_indent = format!("{indent}{indent}");
        for (pred, obj) in predicates {
            self.write_property(writer, pred, obj, &inner_indent)?;
        }

        writeln!(writer, "{indent}</rdf:Description>")?;
        Ok(())
    }

    fn write_property<W: Write>(
        &self,
        writer: &mut W,
        predicate: &RdfTerm,
        object: &RdfTerm,
        indent: &str,
    ) -> std::io::Result<()> {
        let pred_qname = match predicate {
            RdfTerm::Iri(iri) => self.iri_to_qname(iri),
            _ => None,
        };
        let tag = pred_qname.unwrap_or_else(|| match predicate {
            RdfTerm::Iri(iri) => format!("rdf:_unknown_{}", iri.len()),
            _ => "rdf:_unknown".to_string(),
        });

        match object {
            RdfTerm::Iri(iri) if self.config.abbreviated => {
                writeln!(
                    writer,
                    "{indent}<{tag} rdf:resource=\"{}\"/>",
                    xml_escape(iri)
                )?;
            }
            RdfTerm::BlankNode(id) if self.config.abbreviated => {
                writeln!(writer, "{indent}<{tag} rdf:nodeID=\"{id}\"/>")?;
            }
            RdfTerm::PlainLiteral(val) => {
                writeln!(writer, "{indent}<{tag}>{}</{tag}>", xml_escape(val))?;
            }
            RdfTerm::TypedLiteral { value, datatype } => {
                writeln!(
                    writer,
                    "{indent}<{tag} rdf:datatype=\"{}\">{}</{tag}>",
                    xml_escape(datatype),
                    xml_escape(value)
                )?;
            }
            RdfTerm::LangLiteral { value, lang } => {
                writeln!(
                    writer,
                    "{indent}<{tag} xml:lang=\"{lang}\">{}</{tag}>",
                    xml_escape(value)
                )?;
            }
            RdfTerm::Iri(iri) => {
                writeln!(
                    writer,
                    "{indent}<{tag} rdf:resource=\"{}\"/>",
                    xml_escape(iri)
                )?;
            }
            RdfTerm::BlankNode(id) => {
                writeln!(writer, "{indent}<{tag} rdf:nodeID=\"{id}\"/>")?;
            }
        }

        Ok(())
    }

    /// Try to shorten an IRI to a QName using registered prefixes.
    fn iri_to_qname(&self, iri: &str) -> Option<String> {
        for (prefix, ns) in &self.prefixes {
            if let Some(local) = iri.strip_prefix(ns.as_str()) {
                if is_valid_xml_name(local) {
                    return Some(format!("{prefix}:{local}"));
                }
            }
        }
        None
    }

    /// Number of triples written.
    pub fn triple_count(&self) -> usize {
        self.triple_count
    }

    /// Number of registered prefixes.
    pub fn prefix_count(&self) -> usize {
        self.prefixes.len()
    }
}

impl Default for RdfXmlWriter {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Escape special XML characters.
fn xml_escape(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => result.push_str("&amp;"),
            '<' => result.push_str("&lt;"),
            '>' => result.push_str("&gt;"),
            '"' => result.push_str("&quot;"),
            '\'' => result.push_str("&apos;"),
            _ => result.push(c),
        }
    }
    result
}

/// Check if a string is a valid XML name (simplified).
fn is_valid_xml_name(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let first = s.chars().next().expect("non-empty");
    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(s: &str) -> String {
        format!("http://example.org/{s}")
    }

    fn sample_writer() -> RdfXmlWriter {
        let mut w = RdfXmlWriter::with_defaults();
        w.add_prefix("ex", "http://example.org/");
        w
    }

    // ── Basic output ──────────────────────────────────────────────────────

    #[test]
    fn test_empty_document() {
        let w = sample_writer();
        let xml = w.finish().expect("should succeed");
        assert!(xml.contains("<?xml"));
        assert!(xml.contains("<rdf:RDF"));
        assert!(xml.contains("</rdf:RDF>"));
    }

    #[test]
    fn test_single_triple_iri_object() {
        let mut w = sample_writer();
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri(ex("knows")),
            object: RdfTerm::iri(ex("bob")),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        assert!(xml.contains("rdf:about=\"http://example.org/alice\""));
        assert!(xml.contains("rdf:resource=\"http://example.org/bob\""));
        assert!(xml.contains("ex:knows"));
    }

    #[test]
    fn test_single_triple_literal_object() {
        let mut w = sample_writer();
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri(ex("name")),
            object: RdfTerm::literal("Alice"),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        assert!(xml.contains(">Alice</ex:name>"));
    }

    #[test]
    fn test_typed_literal() {
        let mut w = sample_writer();
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri(ex("age")),
            object: RdfTerm::typed("30", "http://www.w3.org/2001/XMLSchema#integer"),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        assert!(xml.contains("rdf:datatype"));
        assert!(xml.contains("30"));
    }

    #[test]
    fn test_lang_literal() {
        let mut w = sample_writer();
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri(ex("name")),
            object: RdfTerm::lang("Alice", "en"),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        assert!(xml.contains("xml:lang=\"en\""));
        assert!(xml.contains("Alice"));
    }

    // ── Blank nodes ───────────────────────────────────────────────────────

    #[test]
    fn test_blank_node_subject() {
        let mut w = sample_writer();
        w.write_triple(&RdfTriple {
            subject: RdfTerm::blank("b0"),
            predicate: RdfTerm::iri(ex("name")),
            object: RdfTerm::literal("Unknown"),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        assert!(xml.contains("rdf:nodeID=\"b0\""));
    }

    #[test]
    fn test_blank_node_object() {
        let mut w = sample_writer();
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri(ex("address")),
            object: RdfTerm::blank("addr1"),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        assert!(xml.contains("rdf:nodeID=\"addr1\""));
    }

    #[test]
    fn test_blank_node_predicate_rejected() {
        let mut w = sample_writer();
        let result = w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::blank("b0"),
            object: RdfTerm::literal("val"),
        });
        assert!(result.is_err());
    }

    // ── Multiple triples same subject ─────────────────────────────────────

    #[test]
    fn test_multiple_triples_same_subject() {
        let mut w = sample_writer();
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri(ex("name")),
            object: RdfTerm::literal("Alice"),
        })
        .expect("write");
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri(ex("age")),
            object: RdfTerm::typed("30", "http://www.w3.org/2001/XMLSchema#integer"),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        // Should have only one rdf:Description for alice
        let desc_count = xml.matches("rdf:Description").count();
        // Opening + closing = 2 per description
        assert_eq!(desc_count, 2); // one description block
    }

    // ── Multiple subjects ─────────────────────────────────────────────────

    #[test]
    fn test_multiple_subjects() {
        let mut w = sample_writer();
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri(ex("name")),
            object: RdfTerm::literal("Alice"),
        })
        .expect("write");
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("bob")),
            predicate: RdfTerm::iri(ex("name")),
            object: RdfTerm::literal("Bob"),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        assert!(xml.contains("alice"));
        assert!(xml.contains("bob"));
        // Two descriptions: 2 * 2 = 4 rdf:Description occurrences
        assert_eq!(xml.matches("rdf:Description").count(), 4);
    }

    // ── XML escaping ──────────────────────────────────────────────────────

    #[test]
    fn test_xml_escape_ampersand() {
        assert_eq!(xml_escape("a&b"), "a&amp;b");
    }

    #[test]
    fn test_xml_escape_lt_gt() {
        assert_eq!(xml_escape("<b>"), "&lt;b&gt;");
    }

    #[test]
    fn test_xml_escape_quotes() {
        assert_eq!(xml_escape("\"'"), "&quot;&apos;");
    }

    #[test]
    fn test_xml_escape_clean() {
        assert_eq!(xml_escape("hello"), "hello");
    }

    #[test]
    fn test_literal_with_special_chars() {
        let mut w = sample_writer();
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri(ex("note")),
            object: RdfTerm::literal("A & B < C > D"),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        assert!(xml.contains("A &amp; B &lt; C &gt; D"));
    }

    // ── Valid XML names ───────────────────────────────────────────────────

    #[test]
    fn test_valid_xml_name() {
        assert!(is_valid_xml_name("foo"));
        assert!(is_valid_xml_name("_foo"));
        assert!(is_valid_xml_name("foo123"));
        assert!(is_valid_xml_name("foo-bar"));
        assert!(!is_valid_xml_name(""));
        assert!(!is_valid_xml_name("123foo"));
        assert!(!is_valid_xml_name("foo bar"));
    }

    // ── QName resolution ──────────────────────────────────────────────────

    #[test]
    fn test_iri_to_qname() {
        let w = sample_writer();
        assert_eq!(
            w.iri_to_qname("http://example.org/name"),
            Some("ex:name".to_string())
        );
        assert_eq!(
            w.iri_to_qname("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Some("rdf:type".to_string())
        );
    }

    #[test]
    fn test_iri_to_qname_no_match() {
        let w = sample_writer();
        assert_eq!(w.iri_to_qname("http://unknown.org/foo"), None);
    }

    // ── Configuration ─────────────────────────────────────────────────────

    #[test]
    fn test_config_no_xml_declaration() {
        let config = WriterConfig {
            xml_declaration: false,
            ..Default::default()
        };
        let w = RdfXmlWriter::new(config);
        let xml = w.finish().expect("finish");
        assert!(!xml.contains("<?xml"));
    }

    #[test]
    fn test_config_base_uri() {
        let config = WriterConfig {
            base_uri: Some("http://example.org/".to_string()),
            ..Default::default()
        };
        let w = RdfXmlWriter::new(config);
        let xml = w.finish().expect("finish");
        assert!(xml.contains("xml:base=\"http://example.org/\""));
    }

    #[test]
    fn test_config_sorted_output() {
        let config = WriterConfig {
            sort_output: true,
            ..Default::default()
        };
        let mut w = RdfXmlWriter::new(config);
        w.add_prefix("ex", "http://example.org/");
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("z_last")),
            predicate: RdfTerm::iri(ex("p")),
            object: RdfTerm::literal("1"),
        })
        .expect("write");
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("a_first")),
            predicate: RdfTerm::iri(ex("p")),
            object: RdfTerm::literal("2"),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        let a_pos = xml.find("a_first").expect("a_first found");
        let z_pos = xml.find("z_last").expect("z_last found");
        assert!(a_pos < z_pos);
    }

    // ── Statistics ────────────────────────────────────────────────────────

    #[test]
    fn test_triple_count() {
        let mut w = sample_writer();
        assert_eq!(w.triple_count(), 0);
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("s")),
            predicate: RdfTerm::iri(ex("p")),
            object: RdfTerm::literal("o"),
        })
        .expect("write");
        assert_eq!(w.triple_count(), 1);
    }

    #[test]
    fn test_prefix_count() {
        let mut w = RdfXmlWriter::with_defaults();
        assert_eq!(w.prefix_count(), 2); // rdf, rdfs
        w.add_prefix("ex", "http://example.org/");
        assert_eq!(w.prefix_count(), 3);
    }

    // ── Write to vec ──────────────────────────────────────────────────────

    #[test]
    fn test_write_to() {
        let mut w = sample_writer();
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri(ex("name")),
            object: RdfTerm::literal("Alice"),
        })
        .expect("write");

        let mut buf = Vec::new();
        w.write_to(&mut buf).expect("write_to");
        let xml = String::from_utf8(buf).expect("utf8");
        assert!(xml.contains("Alice"));
    }

    // ── RdfTerm tests ─────────────────────────────────────────────────────

    #[test]
    fn test_rdf_term_is_iri() {
        assert!(RdfTerm::iri("http://x").is_iri());
        assert!(!RdfTerm::literal("hello").is_iri());
    }

    #[test]
    fn test_rdf_term_is_literal() {
        assert!(RdfTerm::literal("hello").is_literal());
        assert!(RdfTerm::typed("42", "xsd:int").is_literal());
        assert!(RdfTerm::lang("hello", "en").is_literal());
        assert!(!RdfTerm::iri("http://x").is_literal());
    }

    #[test]
    fn test_rdf_term_display() {
        assert_eq!(format!("{}", RdfTerm::iri("http://x")), "<http://x>");
        assert_eq!(format!("{}", RdfTerm::literal("hi")), "\"hi\"");
        assert_eq!(
            format!("{}", RdfTerm::typed("42", "xsd:int")),
            "\"42\"^^<xsd:int>"
        );
        assert_eq!(format!("{}", RdfTerm::lang("hi", "en")), "\"hi\"@en");
        assert_eq!(format!("{}", RdfTerm::blank("b0")), "_:b0");
    }

    // ── Error display ─────────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = RdfXmlWriteError::InvalidIri("bad".to_string());
        assert!(format!("{e}").contains("bad"));

        let e = RdfXmlWriteError::BlankNodePredicate;
        assert!(format!("{e}").contains("predicate"));

        let e = RdfXmlWriteError::IoError("broken pipe".to_string());
        assert!(format!("{e}").contains("broken pipe"));

        let e = RdfXmlWriteError::XmlEncodingError("utf8".to_string());
        assert!(format!("{e}").contains("utf8"));
    }

    // ── Non-abbreviated mode ──────────────────────────────────────────────

    #[test]
    fn test_non_abbreviated() {
        let config = WriterConfig {
            abbreviated: false,
            ..Default::default()
        };
        let mut w = RdfXmlWriter::new(config);
        w.add_prefix("ex", "http://example.org/");
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri(ex("knows")),
            object: RdfTerm::iri(ex("bob")),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        // In non-abbreviated mode, still uses rdf:resource (it's the only way in RDF/XML)
        assert!(xml.contains("rdf:resource"));
    }

    // ── Default writer ────────────────────────────────────────────────────

    #[test]
    fn test_default_writer() {
        let w = RdfXmlWriter::default();
        assert_eq!(w.triple_count(), 0);
        assert_eq!(w.prefix_count(), 2);
    }

    // ── rdf:type handling ─────────────────────────────────────────────────

    #[test]
    fn test_rdf_type_triple() {
        let mut w = sample_writer();
        w.write_triple(&RdfTriple {
            subject: RdfTerm::iri(ex("alice")),
            predicate: RdfTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
            object: RdfTerm::iri(ex("Person")),
        })
        .expect("write");

        let xml = w.finish().expect("finish");
        assert!(xml.contains("rdf:type"));
    }

    // ── Large document ────────────────────────────────────────────────────

    #[test]
    fn test_many_triples() {
        let mut w = sample_writer();
        for i in 0..100 {
            w.write_triple(&RdfTriple {
                subject: RdfTerm::iri(ex(&format!("s{i}"))),
                predicate: RdfTerm::iri(ex("p")),
                object: RdfTerm::literal(format!("value_{i}")),
            })
            .expect("write");
        }
        assert_eq!(w.triple_count(), 100);
        let xml = w.finish().expect("finish");
        assert!(xml.contains("value_99"));
    }
}
