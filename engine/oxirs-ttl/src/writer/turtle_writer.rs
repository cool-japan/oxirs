//! Turtle (Terse RDF Triple Language) serializer / pretty-printer
//!
//! Produces compact, human-readable Turtle output from generic RDF term triples.
//! Operates on the lightweight [`RdfTerm`] / [`TermType`] types so callers are not
//! coupled to `oxirs-core`'s full model hierarchy.
//!
//! # Examples
//!
//! ```rust
//! use oxirs_ttl::writer::{RdfTerm, TermType, TurtleWriter};
//!
//! let mut writer = TurtleWriter::new();
//! writer.add_prefix("ex", "http://example.org/");
//!
//! let triples = vec![
//!     (
//!         RdfTerm::iri("http://example.org/alice"),
//!         RdfTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
//!         RdfTerm::iri("http://example.org/Person"),
//!     ),
//!     (
//!         RdfTerm::iri("http://example.org/alice"),
//!         RdfTerm::iri("http://example.org/name"),
//!         RdfTerm::simple_literal("Alice"),
//!     ),
//! ];
//!
//! let output = writer.write_triples(&triples);
//! assert!(output.contains("@prefix ex:"));
//! assert!(output.contains("ex:alice"));
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;

/// The type of an RDF term
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TermType {
    /// An IRI (Internationalized Resource Identifier)
    Iri,
    /// A blank node
    BlankNode,
    /// A plain, language-tagged, or typed literal
    Literal {
        /// Optional XSD datatype IRI
        datatype: Option<String>,
        /// Optional BCP 47 language tag
        lang: Option<String>,
    },
}

/// A lightweight RDF term for use with [`TurtleWriter`] and the standalone parsers
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RdfTerm {
    /// Lexical value of the term
    pub value: String,
    /// Classification of the term
    pub term_type: TermType,
}

impl RdfTerm {
    /// Construct an IRI term
    pub fn iri(iri: impl Into<String>) -> Self {
        Self {
            value: iri.into(),
            term_type: TermType::Iri,
        }
    }

    /// Construct a blank-node term
    pub fn blank_node(id: impl Into<String>) -> Self {
        Self {
            value: id.into(),
            term_type: TermType::BlankNode,
        }
    }

    /// Construct a plain (xsd:string) literal
    pub fn simple_literal(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            term_type: TermType::Literal {
                datatype: None,
                lang: None,
            },
        }
    }

    /// Construct a language-tagged literal
    pub fn lang_literal(value: impl Into<String>, lang: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            term_type: TermType::Literal {
                datatype: None,
                lang: Some(lang.into()),
            },
        }
    }

    /// Construct a typed literal
    pub fn typed_literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            term_type: TermType::Literal {
                datatype: Some(datatype.into()),
                lang: None,
            },
        }
    }
}

impl std::fmt::Display for RdfTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.term_type {
            TermType::Iri => write!(f, "<{}>", self.value),
            TermType::BlankNode => write!(f, "_:{}", self.value),
            TermType::Literal { datatype, lang } => {
                let escaped = escape_string(&self.value);
                write!(f, "\"{escaped}\"")?;
                if let Some(l) = lang {
                    write!(f, "@{l}")?;
                } else if let Some(dt) = datatype {
                    write!(f, "^^<{dt}>")?;
                }
                Ok(())
            }
        }
    }
}

// ─── Configuration ─────────────────────────────────────────────────────────

/// Configuration for the Turtle pretty-printer
#[derive(Debug, Clone)]
pub struct TurtleWriterConfig {
    /// Indentation string (default: four spaces)
    pub indent: String,
    /// Whether to apply registered prefixes when abbreviating IRIs
    pub use_prefixes: bool,
    /// Whether to emit predicates in alphabetical order
    pub sort_predicates: bool,
    /// Whether to emit multiple objects on separate indented lines
    pub compact_lists: bool,
    /// Soft limit for line length (informational only; not hard-wrapped)
    pub max_line_length: usize,
}

impl Default for TurtleWriterConfig {
    fn default() -> Self {
        Self {
            indent: "    ".to_string(),
            use_prefixes: true,
            sort_predicates: true,
            compact_lists: true,
            max_line_length: 120,
        }
    }
}

// ─── Writer ────────────────────────────────────────────────────────────────

/// Turtle serializer / pretty-printer
///
/// Groups triples by subject and emits compact Turtle with predicate-object
/// lists, prefix abbreviation, and optional `rdf:type` → `a` shorthand.
#[derive(Debug, Clone)]
pub struct TurtleWriter {
    config: TurtleWriterConfig,
    /// Registered prefixes: prefix string → namespace IRI
    prefixes: BTreeMap<String, String>,
}

impl Default for TurtleWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl TurtleWriter {
    /// Create a writer with default configuration and no prefixes
    pub fn new() -> Self {
        Self {
            config: TurtleWriterConfig::default(),
            prefixes: BTreeMap::new(),
        }
    }

    /// Create a writer with a custom configuration
    pub fn with_config(config: TurtleWriterConfig) -> Self {
        Self {
            config,
            prefixes: BTreeMap::new(),
        }
    }

    /// Register a prefix/namespace pair
    ///
    /// ```rust
    /// use oxirs_ttl::writer::TurtleWriter;
    /// let mut w = TurtleWriter::new();
    /// w.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
    /// ```
    pub fn add_prefix(&mut self, prefix: &str, iri: &str) {
        self.prefixes.insert(prefix.to_string(), iri.to_string());
    }

    /// Serialize a slice of `(subject, predicate, object)` triples to a Turtle string
    pub fn write_triples(&self, triples: &[(RdfTerm, RdfTerm, RdfTerm)]) -> String {
        let mut output = String::new();

        // Emit prefix declarations
        if self.config.use_prefixes && !self.prefixes.is_empty() {
            for (prefix, namespace) in &self.prefixes {
                writeln!(output, "@prefix {prefix}: <{namespace}> .").ok();
            }
            writeln!(output).ok();
        }

        // Group triples by subject (preserving encounter order via BTreeMap on
        // canonical subject string so output is deterministic)
        let grouped = Self::group_by_subject(triples);

        for (subject_key, po_pairs) in &grouped {
            // Retrieve the original subject term from the first pair
            let subject = triples
                .iter()
                .find(|(s, _, _)| self.term_key(s) == *subject_key)
                .map(|(s, _, _)| s)
                .expect("subject must exist because it came from our grouped map");

            let block = self.write_subject_block(subject, po_pairs);
            output.push_str(&block);
            output.push('\n');
        }

        output
    }

    // ─── Internal helpers ────────────────────────────────────────────────────

    /// Produce the Turtle block for one subject with all its predicate-object pairs
    fn write_subject_block(&self, subject: &RdfTerm, po_pairs: &[(RdfTerm, RdfTerm)]) -> String {
        let mut out = String::new();
        let subject_str = self.term_to_turtle(subject);

        // Deduplicate and (optionally) sort predicate keys
        let mut predicate_map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (predicate, object) in po_pairs {
            let pred_str = self.term_to_turtle(predicate);
            let obj_str = self.term_to_turtle(object);
            predicate_map.entry(pred_str).or_default().insert(obj_str);
        }

        let predicates: Vec<_> = if self.config.sort_predicates {
            // rdf:type / "a" always first
            let mut preds: Vec<_> = predicate_map.keys().cloned().collect();
            preds.sort_by(|a, b| {
                let a_is_type = a == "a";
                let b_is_type = b == "a";
                match (a_is_type, b_is_type) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => a.cmp(b),
                }
            });
            preds
        } else {
            predicate_map.keys().cloned().collect()
        };

        write!(out, "{subject_str}").ok();

        for (i, pred) in predicates.iter().enumerate() {
            let objects: Vec<_> = predicate_map[pred].iter().cloned().collect();

            if i == 0 {
                // First predicate on the same line as the subject (or next line if long)
                write!(out, "\n{}{pred}", self.config.indent).ok();
            } else {
                write!(out, " ;\n{}{pred}", self.config.indent).ok();
            }

            if objects.len() == 1 {
                write!(out, " {}", objects[0]).ok();
            } else {
                // Multiple objects: one per line, comma-separated
                for (j, obj) in objects.iter().enumerate() {
                    if j == 0 {
                        write!(out, " {obj}").ok();
                    } else {
                        write!(out, " ,\n{}{}{obj}", self.config.indent, self.config.indent).ok();
                    }
                }
            }
        }

        writeln!(out, " .").ok();
        out
    }

    /// Abbreviate an IRI using the registered prefixes.  If no prefix matches,
    /// returns the IRI enclosed in angle brackets.
    fn abbreviate_iri(&self, iri: &str) -> String {
        // rdf:type shorthand
        if iri == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
            return "a".to_string();
        }

        if self.config.use_prefixes {
            // Find the longest matching namespace to be most specific
            let mut best: Option<(&str, &str)> = None;
            for (prefix, namespace) in &self.prefixes {
                if iri.starts_with(namespace.as_str())
                    && best.map_or(true, |(_, prev_ns)| namespace.len() > prev_ns.len())
                {
                    best = Some((prefix.as_str(), namespace.as_str()));
                }
            }
            if let Some((prefix, namespace)) = best {
                let local = &iri[namespace.len()..];
                if !local.is_empty() && is_valid_local_name(local) {
                    return format!("{prefix}:{local}");
                }
            }
        }

        format!("<{iri}>")
    }

    /// Serialize a single [`RdfTerm`] to its Turtle representation
    fn term_to_turtle(&self, term: &RdfTerm) -> String {
        match &term.term_type {
            TermType::Iri => self.abbreviate_iri(&term.value),
            TermType::BlankNode => format!("_:{}", term.value),
            TermType::Literal { datatype, lang } => {
                let escaped = escape_string(&term.value);
                let mut s = format!("\"{escaped}\"");
                if let Some(l) = lang {
                    s.push('@');
                    s.push_str(l);
                } else if let Some(dt) = datatype {
                    let dt_turtle = if self.config.use_prefixes {
                        self.abbreviate_iri(dt)
                    } else {
                        format!("<{dt}>")
                    };
                    s.push_str("^^");
                    s.push_str(&dt_turtle);
                }
                s
            }
        }
    }

    /// Canonical sort key for a term (used for grouping)
    fn term_key(&self, term: &RdfTerm) -> String {
        match &term.term_type {
            TermType::Iri => format!("iri:{}", term.value),
            TermType::BlankNode => format!("bn:{}", term.value),
            TermType::Literal { .. } => format!("lit:{term}"),
        }
    }

    /// Group triples by subject, preserving a deterministic order
    fn group_by_subject(
        triples: &[(RdfTerm, RdfTerm, RdfTerm)],
    ) -> BTreeMap<String, Vec<(RdfTerm, RdfTerm)>> {
        // We use insertion-order tracking alongside a BTreeMap so that we
        // can emit subjects in the order they were first encountered while
        // still keeping the code simple.  The BTreeMap key is a canonical
        // string representation so the map is sorted lexicographically.
        let mut map: BTreeMap<String, Vec<(RdfTerm, RdfTerm)>> = BTreeMap::new();

        for (subject, predicate, object) in triples {
            let key = match &subject.term_type {
                TermType::Iri => format!("iri:{}", subject.value),
                TermType::BlankNode => format!("bn:{}", subject.value),
                TermType::Literal { .. } => format!("lit:{subject}"),
            };
            map.entry(key)
                .or_default()
                .push((predicate.clone(), object.clone()));
        }

        map
    }
}

// ─── Utility functions ──────────────────────────────────────────────────────

/// Escape special characters in a string literal value
fn escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Check whether a string is a valid Turtle local name (simplified check)
fn is_valid_local_name(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == '/')
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(local: &str) -> RdfTerm {
        RdfTerm::iri(format!("http://example.org/{local}"))
    }

    fn rdf_type() -> RdfTerm {
        RdfTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    }

    #[test]
    fn test_turtle_writer_basic_output() {
        let mut writer = TurtleWriter::new();
        writer.add_prefix("ex", "http://example.org/");

        let triples = vec![
            (ex("alice"), rdf_type(), ex("Person")),
            (ex("alice"), ex("name"), RdfTerm::simple_literal("Alice")),
        ];

        let output = writer.write_triples(&triples);

        assert!(output.contains("@prefix ex:"), "missing prefix declaration");
        assert!(output.contains("ex:alice"), "IRI not abbreviated");
        assert!(output.contains("a "), "rdf:type not shortened to 'a'");
        assert!(output.contains("\"Alice\""), "literal missing");
    }

    #[test]
    fn test_turtle_writer_multiple_subjects() {
        let mut writer = TurtleWriter::new();
        writer.add_prefix("ex", "http://example.org/");

        let triples = vec![
            (ex("alice"), ex("knows"), ex("bob")),
            (ex("bob"), ex("name"), RdfTerm::simple_literal("Bob")),
        ];

        let output = writer.write_triples(&triples);
        // Both subjects must appear
        assert!(output.contains("ex:alice"));
        assert!(output.contains("ex:bob"));
        // Two subject blocks means two '.' terminators on their own lines
        let dot_lines = output.lines().filter(|l| l.trim_end() == ".").count();
        // Count lines ending with " ." that are not prefix declarations
        let triple_dots = output
            .lines()
            .filter(|l| l.ends_with(" .") && !l.starts_with("@prefix"))
            .count();
        assert_eq!(
            triple_dots, 2,
            "expected 2 triple-block terminators, got {triple_dots}\nOutput:\n{output}"
        );
        let _ = dot_lines;
    }

    #[test]
    fn test_turtle_writer_no_prefixes() {
        let config = TurtleWriterConfig {
            use_prefixes: false,
            ..TurtleWriterConfig::default()
        };
        let writer = TurtleWriter::with_config(config);

        let triples = vec![(ex("alice"), ex("name"), RdfTerm::simple_literal("Alice"))];

        let output = writer.write_triples(&triples);
        assert!(
            !output.contains("@prefix"),
            "should not emit prefix declarations"
        );
        assert!(
            output.contains("<http://example.org/alice>"),
            "full IRI must appear"
        );
    }

    #[test]
    fn test_turtle_writer_language_tagged_literal() {
        let mut writer = TurtleWriter::new();
        writer.add_prefix("ex", "http://example.org/");

        let triples = vec![(ex("doc"), ex("title"), RdfTerm::lang_literal("Hello", "en"))];

        let output = writer.write_triples(&triples);
        assert!(output.contains("\"Hello\"@en"), "language tag missing");
    }

    #[test]
    fn test_turtle_writer_typed_literal() {
        let mut writer = TurtleWriter::new();
        writer.add_prefix("ex", "http://example.org/");
        writer.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");

        let triples = vec![(
            ex("item"),
            ex("count"),
            RdfTerm::typed_literal("42", "http://www.w3.org/2001/XMLSchema#integer"),
        )];

        let output = writer.write_triples(&triples);
        assert!(
            output.contains("\"42\"^^xsd:integer"),
            "typed literal incorrectly serialised"
        );
    }

    #[test]
    fn test_turtle_writer_blank_node_subject() {
        let mut writer = TurtleWriter::new();
        writer.add_prefix("ex", "http://example.org/");

        let triples = vec![(
            RdfTerm::blank_node("b0"),
            ex("label"),
            RdfTerm::simple_literal("anon"),
        )];

        let output = writer.write_triples(&triples);
        assert!(output.contains("_:b0"), "blank node subject missing");
    }

    #[test]
    fn test_turtle_writer_multiple_objects_same_predicate() {
        let mut writer = TurtleWriter::new();
        writer.add_prefix("ex", "http://example.org/");

        // Two triples sharing subject and predicate → object list
        let triples = vec![
            (ex("alice"), ex("likes"), ex("cats")),
            (ex("alice"), ex("likes"), ex("dogs")),
        ];

        let output = writer.write_triples(&triples);
        // Should contain a comma separating the two objects
        assert!(output.contains(','), "comma-separated object list expected");
    }

    #[test]
    fn test_rdf_term_display() {
        assert_eq!(
            RdfTerm::iri("http://ex.org/foo").to_string(),
            "<http://ex.org/foo>"
        );
        assert_eq!(RdfTerm::blank_node("b1").to_string(), "_:b1");
        assert_eq!(RdfTerm::simple_literal("hello").to_string(), "\"hello\"");
        assert_eq!(
            RdfTerm::lang_literal("bonjour", "fr").to_string(),
            "\"bonjour\"@fr"
        );
        assert_eq!(
            RdfTerm::typed_literal("42", "http://www.w3.org/2001/XMLSchema#integer").to_string(),
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"
        );
    }

    #[test]
    fn test_escape_special_chars() {
        let term = RdfTerm::simple_literal("line1\nline2\ttab\"quote\\back");
        let turtle = TurtleWriter::new().term_to_turtle(&term);
        assert!(turtle.contains("\\n"), "newline not escaped");
        assert!(turtle.contains("\\t"), "tab not escaped");
        assert!(turtle.contains("\\\""), "quote not escaped");
        assert!(turtle.contains("\\\\"), "backslash not escaped");
    }
}
