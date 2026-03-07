//! # Export Command — Multi-format RDF Data Exporter
//!
//! Provides `DataExporter` for serialising RDF triples to Turtle, N-Triples,
//! N-Quads, JSON-LD, RDF/XML, TriG, and CSV formats.
//!
//! # Example
//!
//! ```rust
//! use oxirs::commands::export_command::{DataExporter, ExportFormat, ExportOptions, Triple};
//! use std::collections::HashMap;
//!
//! let triples = vec![
//!     Triple { subject: "http://a.org/s".into(), predicate: "http://a.org/p".into(),
//!              object: "http://a.org/o".into(), graph: None },
//! ];
//! let prefixes = HashMap::new();
//! let opts = ExportOptions {
//!     format: ExportFormat::NTriples,
//!     ..Default::default()
//! };
//! let out = DataExporter::export(&triples, &prefixes, &opts).expect("export ok");
//! assert!(out.contains("<http://a.org/s>"));
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the exporter
#[derive(Debug, Clone, PartialEq)]
pub enum ExportError {
    /// The format string is not recognised
    UnknownFormat(String),
    /// A triple has invalid structure
    InvalidTriple(String),
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExportError::UnknownFormat(s) => write!(f, "Unknown export format: {s}"),
            ExportError::InvalidTriple(s) => write!(f, "Invalid triple: {s}"),
        }
    }
}

impl std::error::Error for ExportError {}

// ---------------------------------------------------------------------------
// Export format
// ---------------------------------------------------------------------------

/// Supported RDF serialisation formats
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExportFormat {
    /// Turtle (`.ttl`)
    Turtle,
    /// N-Triples (`.nt`)
    NTriples,
    /// N-Quads (`.nq`)
    NQuads,
    /// JSON-LD (`.jsonld`)
    JsonLd,
    /// RDF/XML (`.rdf`)
    RdfXml,
    /// TriG (`.trig`)
    Trig,
    /// CSV (`.csv`)
    Csv,
}

impl ExportFormat {
    /// File extension for this format (without leading `.`)
    pub fn extension(&self) -> &str {
        match self {
            ExportFormat::Turtle => "ttl",
            ExportFormat::NTriples => "nt",
            ExportFormat::NQuads => "nq",
            ExportFormat::JsonLd => "jsonld",
            ExportFormat::RdfXml => "rdf",
            ExportFormat::Trig => "trig",
            ExportFormat::Csv => "csv",
        }
    }

    /// MIME type for this format
    pub fn mime_type(&self) -> &str {
        match self {
            ExportFormat::Turtle => "text/turtle",
            ExportFormat::NTriples => "application/n-triples",
            ExportFormat::NQuads => "application/n-quads",
            ExportFormat::JsonLd => "application/ld+json",
            ExportFormat::RdfXml => "application/rdf+xml",
            ExportFormat::Trig => "application/trig",
            ExportFormat::Csv => "text/csv",
        }
    }

    /// Parse from a string, case-insensitively
    pub fn parse(s: &str) -> Result<ExportFormat, ExportError> {
        match s.to_lowercase().trim() {
            "turtle" | "ttl" => Ok(ExportFormat::Turtle),
            "ntriples" | "n-triples" | "nt" => Ok(ExportFormat::NTriples),
            "nquads" | "n-quads" | "nq" => Ok(ExportFormat::NQuads),
            "jsonld" | "json-ld" | "json_ld" => Ok(ExportFormat::JsonLd),
            "rdfxml" | "rdf-xml" | "rdf/xml" | "rdf" => Ok(ExportFormat::RdfXml),
            "trig" => Ok(ExportFormat::Trig),
            "csv" => Ok(ExportFormat::Csv),
            other => Err(ExportError::UnknownFormat(other.to_string())),
        }
    }
}

// ---------------------------------------------------------------------------
// Triple
// ---------------------------------------------------------------------------

/// An RDF triple with an optional named graph
#[derive(Debug, Clone)]
pub struct Triple {
    /// Subject IRI or blank node
    pub subject: String,
    /// Predicate IRI
    pub predicate: String,
    /// Object IRI, blank node, or literal
    pub object: String,
    /// Named graph IRI (optional; present for quads)
    pub graph: Option<String>,
}

// ---------------------------------------------------------------------------
// Export options
// ---------------------------------------------------------------------------

/// Options controlling the export behaviour
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Target serialisation format
    pub format: ExportFormat,
    /// Enable pretty-printing (grouping, indentation)
    pub pretty_print: bool,
    /// Base IRI for relative IRI resolution
    pub base_iri: Option<String>,
    /// Emit PREFIX / @prefix declarations
    pub include_prefixes: bool,
    /// Compress output (future use)
    pub compress: bool,
    /// Maximum number of triples to export
    pub limit: Option<usize>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Turtle,
            pretty_print: true,
            base_iri: None,
            include_prefixes: true,
            compress: false,
            limit: None,
        }
    }
}

// ---------------------------------------------------------------------------
// DataExporter
// ---------------------------------------------------------------------------

/// Stateless RDF data exporter
pub struct DataExporter;

impl DataExporter {
    // -----------------------------------------------------------------------
    // Dispatcher
    // -----------------------------------------------------------------------

    /// Export triples using the format specified in `options`.
    pub fn export(
        triples: &[Triple],
        prefixes: &HashMap<String, String>,
        options: &ExportOptions,
    ) -> Result<String, ExportError> {
        let limited = Self::apply_limit(triples, options.limit);
        match options.format {
            ExportFormat::Turtle => Ok(Self::export_turtle(limited, prefixes, options)),
            ExportFormat::NTriples => Ok(Self::export_ntriples(limited)),
            ExportFormat::NQuads => Ok(Self::export_nquads(limited)),
            ExportFormat::JsonLd => Ok(Self::export_jsonld(limited, prefixes)),
            ExportFormat::RdfXml => Ok(Self::export_rdfxml(limited, prefixes, options)),
            ExportFormat::Trig => Ok(Self::export_trig(limited, prefixes, options)),
            ExportFormat::Csv => Ok(Self::export_csv(limited, true)),
        }
    }

    // -----------------------------------------------------------------------
    // Limit
    // -----------------------------------------------------------------------

    /// Slice triples to respect an optional upper bound.
    pub fn apply_limit(triples: &[Triple], limit: Option<usize>) -> &[Triple] {
        match limit {
            Some(n) if n < triples.len() => &triples[..n],
            _ => triples,
        }
    }

    // -----------------------------------------------------------------------
    // IRI abbreviation
    // -----------------------------------------------------------------------

    /// Shorten `iri` using the longest matching prefix namespace.
    ///
    /// Returns the prefixed form (e.g. `rdf:type`) or the original `iri` if no
    /// prefix matches.
    pub fn abbreviate_iri(iri: &str, prefixes: &HashMap<String, String>) -> String {
        // Find the longest matching namespace
        let mut best: Option<(&str, &str)> = None; // (prefix_name, namespace)
        for (name, namespace) in prefixes {
            if iri.starts_with(namespace.as_str()) {
                match best {
                    None => best = Some((name, namespace)),
                    Some((_, prev_ns)) if namespace.len() > prev_ns.len() => {
                        best = Some((name, namespace));
                    }
                    _ => {}
                }
            }
        }
        if let Some((name, namespace)) = best {
            let local = &iri[namespace.len()..];
            if !local.is_empty() {
                return format!("{name}:{local}");
            }
        }
        iri.to_string()
    }

    // -----------------------------------------------------------------------
    // Turtle
    // -----------------------------------------------------------------------

    /// Serialise to Turtle format.
    ///
    /// Triples are grouped by subject.  If `pretty_print` is set, each
    /// predicate-object pair is placed on its own line.
    pub fn export_turtle(
        triples: &[Triple],
        prefixes: &HashMap<String, String>,
        options: &ExportOptions,
    ) -> String {
        let mut out = String::new();

        // Prefix declarations
        if options.include_prefixes {
            for (name, namespace) in prefixes {
                out.push_str(&format!("@prefix {name}: <{namespace}> .\n"));
            }
            if !prefixes.is_empty() {
                out.push('\n');
            }
        }

        // Base IRI
        if let Some(ref base) = options.base_iri {
            out.push_str(&format!("@base <{base}> .\n\n"));
        }

        // Group by subject
        let mut grouped: Vec<(String, Vec<(&Triple,)>)> = Vec::new();
        for triple in triples {
            let subj = &triple.subject;
            if let Some(entry) = grouped.iter_mut().find(|(s, _)| s == subj) {
                entry.1.push((triple,));
            } else {
                grouped.push((subj.clone(), vec![(triple,)]));
            }
        }

        for (subject, group) in &grouped {
            let subj_repr = Self::turtle_term(subject, prefixes);
            if options.pretty_print {
                out.push_str(&subj_repr);
                out.push('\n');
                let n = group.len();
                for (idx, (triple,)) in group.iter().enumerate() {
                    let pred = Self::turtle_term(&triple.predicate, prefixes);
                    let obj = Self::turtle_object(&triple.object, prefixes);
                    let sep = if idx + 1 < n { " ;" } else { " ." };
                    out.push_str(&format!("    {pred} {obj}{sep}\n"));
                }
            } else {
                for (triple,) in group {
                    let pred = Self::turtle_term(&triple.predicate, prefixes);
                    let obj = Self::turtle_object(&triple.object, prefixes);
                    out.push_str(&format!("{subj_repr} {pred} {obj} .\n"));
                }
            }
            out.push('\n');
        }

        out.trim_end().to_string()
    }

    // -----------------------------------------------------------------------
    // N-Triples
    // -----------------------------------------------------------------------

    /// Serialise to N-Triples (one triple per line, full IRIs).
    pub fn export_ntriples(triples: &[Triple]) -> String {
        let mut out = String::new();
        for triple in triples {
            let s = Self::nt_term(&triple.subject);
            let p = Self::nt_iri(&triple.predicate);
            let o = Self::nt_object(&triple.object);
            out.push_str(&format!("{s} {p} {o} .\n"));
        }
        out
    }

    // -----------------------------------------------------------------------
    // N-Quads
    // -----------------------------------------------------------------------

    /// Serialise to N-Quads (includes graph IRI when present).
    pub fn export_nquads(triples: &[Triple]) -> String {
        let mut out = String::new();
        for triple in triples {
            let s = Self::nt_term(&triple.subject);
            let p = Self::nt_iri(&triple.predicate);
            let o = Self::nt_object(&triple.object);
            if let Some(ref g) = triple.graph {
                let gn = Self::nt_iri(g);
                out.push_str(&format!("{s} {p} {o} {gn} .\n"));
            } else {
                out.push_str(&format!("{s} {p} {o} .\n"));
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // JSON-LD (hand-rolled)
    // -----------------------------------------------------------------------

    /// Serialise to JSON-LD (simple @graph array).
    pub fn export_jsonld(triples: &[Triple], prefixes: &HashMap<String, String>) -> String {
        let mut out = String::from("{\n");

        // @context
        if !prefixes.is_empty() {
            out.push_str("  \"@context\": {\n");
            let entries: Vec<String> = prefixes
                .iter()
                .map(|(name, ns)| format!("    \"{name}\": \"{ns}\""))
                .collect();
            out.push_str(&entries.join(",\n"));
            out.push_str("\n  },\n");
        }

        // @graph
        out.push_str("  \"@graph\": [\n");

        // Group by subject
        let mut grouped: HashMap<&str, Vec<&Triple>> = HashMap::new();
        for triple in triples {
            grouped.entry(&triple.subject).or_default().push(triple);
        }

        let mut subject_blocks: Vec<String> = Vec::new();
        for (subject, group) in &grouped {
            let mut block = String::from("    {\n");
            block.push_str(&format!("      \"@id\": \"{subject}\""));

            // Group predicates
            let mut pred_map: HashMap<&str, Vec<String>> = HashMap::new();
            for triple in group {
                pred_map
                    .entry(&triple.predicate)
                    .or_default()
                    .push(jsonld_value(&triple.object, prefixes));
            }

            for (pred, values) in &pred_map {
                let abbrev = Self::abbreviate_iri(pred, prefixes);
                block.push_str(&format!(",\n      \"{abbrev}\": "));
                if values.len() == 1 {
                    block.push_str(&values[0]);
                } else {
                    block.push('[');
                    block.push_str(&values.join(", "));
                    block.push(']');
                }
            }

            block.push_str("\n    }");
            subject_blocks.push(block);
        }

        out.push_str(&subject_blocks.join(",\n"));
        out.push_str("\n  ]\n}");
        out
    }

    // -----------------------------------------------------------------------
    // RDF/XML (hand-rolled)
    // -----------------------------------------------------------------------

    /// Serialise to RDF/XML format.
    pub fn export_rdfxml(
        triples: &[Triple],
        prefixes: &HashMap<String, String>,
        _options: &ExportOptions,
    ) -> String {
        let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<rdf:RDF\n");
        out.push_str("  xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"");

        for (name, ns) in prefixes {
            out.push_str(&format!("\n  xmlns:{name}=\"{ns}\""));
        }
        out.push_str(">\n\n");

        // Group by subject
        let mut grouped: Vec<(String, Vec<&Triple>)> = Vec::new();
        for triple in triples {
            if let Some(entry) = grouped.iter_mut().find(|(s, _)| s == &triple.subject) {
                entry.1.push(triple);
            } else {
                grouped.push((triple.subject.clone(), vec![triple]));
            }
        }

        for (subject, group) in &grouped {
            out.push_str(&format!("  <rdf:Description rdf:about=\"{subject}\">\n"));
            for triple in group {
                let pred_local =
                    Self::abbreviate_iri(&triple.predicate, prefixes).replace(':', "__");
                let pred_tag = if pred_local.contains("__") {
                    pred_local.replacen("__", ":", 1)
                } else {
                    format!("rdf:_{pred_local}")
                };
                let obj = xml_escape(&triple.object);
                out.push_str(&format!("    <{pred_tag}>{obj}</{pred_tag}>\n"));
            }
            out.push_str("  </rdf:Description>\n\n");
        }

        out.push_str("</rdf:RDF>");
        out
    }

    // -----------------------------------------------------------------------
    // TriG
    // -----------------------------------------------------------------------

    /// Serialise to TriG format.
    ///
    /// Triples without a graph appear in the default graph block.
    /// Named-graph triples are grouped by graph IRI.
    pub fn export_trig(
        triples: &[Triple],
        prefixes: &HashMap<String, String>,
        options: &ExportOptions,
    ) -> String {
        let mut out = String::new();

        if options.include_prefixes {
            for (name, namespace) in prefixes {
                out.push_str(&format!("@prefix {name}: <{namespace}> .\n"));
            }
            if !prefixes.is_empty() {
                out.push('\n');
            }
        }

        // Partition: default graph vs. named graphs
        let (default_triples, named_triples): (Vec<&Triple>, Vec<&Triple>) =
            triples.iter().partition(|t| t.graph.is_none());

        // Default graph
        if !default_triples.is_empty() {
            out.push_str("{\n");
            for triple in &default_triples {
                let s = Self::turtle_term(&triple.subject, prefixes);
                let p = Self::turtle_term(&triple.predicate, prefixes);
                let o = Self::turtle_object(&triple.object, prefixes);
                out.push_str(&format!("  {s} {p} {o} .\n"));
            }
            out.push_str("}\n\n");
        }

        // Named graphs
        let mut graph_groups: Vec<(String, Vec<&Triple>)> = Vec::new();
        for triple in named_triples {
            let gname = triple.graph.as_deref().unwrap_or("").to_string();
            if let Some(entry) = graph_groups.iter_mut().find(|(g, _)| g == &gname) {
                entry.1.push(triple);
            } else {
                graph_groups.push((gname, vec![triple]));
            }
        }

        for (graph, group) in &graph_groups {
            let g_repr = Self::turtle_term(graph, prefixes);
            out.push_str(&format!("{g_repr} {{\n"));
            for triple in group {
                let s = Self::turtle_term(&triple.subject, prefixes);
                let p = Self::turtle_term(&triple.predicate, prefixes);
                let o = Self::turtle_object(&triple.object, prefixes);
                out.push_str(&format!("  {s} {p} {o} .\n"));
            }
            out.push_str("}\n\n");
        }

        out.trim_end().to_string()
    }

    // -----------------------------------------------------------------------
    // CSV
    // -----------------------------------------------------------------------

    /// Serialise to CSV with header `subject,predicate,object[,graph]`.
    pub fn export_csv(triples: &[Triple], include_graph: bool) -> String {
        let mut out = String::new();
        if include_graph {
            out.push_str("subject,predicate,object,graph\n");
        } else {
            out.push_str("subject,predicate,object\n");
        }
        for triple in triples {
            let s = csv_escape(&triple.subject);
            let p = csv_escape(&triple.predicate);
            let o = csv_escape(&triple.object);
            if include_graph {
                let g = triple.graph.as_deref().map(csv_escape).unwrap_or_default();
                out.push_str(&format!("{s},{p},{o},{g}\n"));
            } else {
                out.push_str(&format!("{s},{p},{o}\n"));
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Represent an IRI or blank node as a Turtle term, abbreviating with prefixes.
    fn turtle_term(value: &str, prefixes: &HashMap<String, String>) -> String {
        if value.starts_with("_:") {
            // Blank node
            return value.to_string();
        }
        let abbrev = Self::abbreviate_iri(value, prefixes);
        if abbrev == value {
            format!("<{value}>")
        } else {
            abbrev
        }
    }

    /// Represent an object term in Turtle (handles literals, IRIs, blank nodes).
    fn turtle_object(value: &str, prefixes: &HashMap<String, String>) -> String {
        if value.starts_with('"') || value.starts_with('\'') {
            // Already a literal
            value.to_string()
        } else if value.starts_with("_:") {
            value.to_string()
        } else {
            Self::turtle_term(value, prefixes)
        }
    }

    /// N-Triples representation for a subject or predicate (IRI or blank node).
    fn nt_term(value: &str) -> String {
        if value.starts_with("_:") {
            value.to_string()
        } else {
            format!("<{value}>")
        }
    }

    /// N-Triples representation for a predicate (always IRI).
    fn nt_iri(value: &str) -> String {
        format!("<{value}>")
    }

    /// N-Triples representation for an object (IRI, blank node, or literal).
    fn nt_object(value: &str) -> String {
        if value.starts_with('"') || value.starts_with("_:") {
            value.to_string()
        } else {
            format!("<{value}>")
        }
    }
}

// ---------------------------------------------------------------------------
// Format-specific helpers (free functions)
// ---------------------------------------------------------------------------

/// Escape special XML characters
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// CSV-quote a field if it contains commas, quotes or newlines
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Produce a JSON-LD value string for an object term
fn jsonld_value(value: &str, prefixes: &HashMap<String, String>) -> String {
    if value.starts_with('"') {
        // Literal — wrap in {"@value": ...}
        format!("{{\"@value\": {value}}}")
    } else {
        let abbrev = DataExporter::abbreviate_iri(value, prefixes);
        format!("{{\"@id\": \"{abbrev}\"}}")
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triples() -> Vec<Triple> {
        vec![
            Triple {
                subject: "http://example.org/alice".into(),
                predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into(),
                object: "http://example.org/Person".into(),
                graph: None,
            },
            Triple {
                subject: "http://example.org/alice".into(),
                predicate: "http://xmlns.com/foaf/0.1/name".into(),
                object: "\"Alice\"".into(),
                graph: None,
            },
            Triple {
                subject: "http://example.org/bob".into(),
                predicate: "http://xmlns.com/foaf/0.1/name".into(),
                object: "\"Bob\"".into(),
                graph: None,
            },
        ]
    }

    fn sample_prefixes() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert(
            "rdf".into(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".into(),
        );
        m.insert("foaf".into(), "http://xmlns.com/foaf/0.1/".into());
        m.insert("ex".into(), "http://example.org/".into());
        m
    }

    // --- ExportFormat ---

    #[test]
    fn test_format_extension_turtle() {
        assert_eq!(ExportFormat::Turtle.extension(), "ttl");
    }

    #[test]
    fn test_format_extension_ntriples() {
        assert_eq!(ExportFormat::NTriples.extension(), "nt");
    }

    #[test]
    fn test_format_extension_nquads() {
        assert_eq!(ExportFormat::NQuads.extension(), "nq");
    }

    #[test]
    fn test_format_extension_jsonld() {
        assert_eq!(ExportFormat::JsonLd.extension(), "jsonld");
    }

    #[test]
    fn test_format_extension_rdfxml() {
        assert_eq!(ExportFormat::RdfXml.extension(), "rdf");
    }

    #[test]
    fn test_format_extension_trig() {
        assert_eq!(ExportFormat::Trig.extension(), "trig");
    }

    #[test]
    fn test_format_extension_csv() {
        assert_eq!(ExportFormat::Csv.extension(), "csv");
    }

    #[test]
    fn test_format_mime_turtle() {
        assert_eq!(ExportFormat::Turtle.mime_type(), "text/turtle");
    }

    #[test]
    fn test_format_mime_ntriples() {
        assert_eq!(ExportFormat::NTriples.mime_type(), "application/n-triples");
    }

    #[test]
    fn test_format_mime_csv() {
        assert_eq!(ExportFormat::Csv.mime_type(), "text/csv");
    }

    #[test]
    fn test_format_from_str_turtle() {
        assert_eq!(ExportFormat::parse("turtle").unwrap(), ExportFormat::Turtle);
        assert_eq!(ExportFormat::parse("TTL").unwrap(), ExportFormat::Turtle);
    }

    #[test]
    fn test_format_from_str_ntriples() {
        assert_eq!(
            ExportFormat::parse("ntriples").unwrap(),
            ExportFormat::NTriples
        );
        assert_eq!(
            ExportFormat::parse("n-triples").unwrap(),
            ExportFormat::NTriples
        );
        assert_eq!(ExportFormat::parse("nt").unwrap(), ExportFormat::NTriples);
    }

    #[test]
    fn test_format_from_str_jsonld() {
        assert_eq!(ExportFormat::parse("jsonld").unwrap(), ExportFormat::JsonLd);
        assert_eq!(
            ExportFormat::parse("json-ld").unwrap(),
            ExportFormat::JsonLd
        );
    }

    #[test]
    fn test_format_from_str_unknown() {
        let err = ExportFormat::parse("xyz").unwrap_err();
        assert!(matches!(err, ExportError::UnknownFormat(_)));
    }

    #[test]
    fn test_format_from_str_csv() {
        assert_eq!(ExportFormat::parse("csv").unwrap(), ExportFormat::Csv);
        assert_eq!(ExportFormat::parse("CSV").unwrap(), ExportFormat::Csv);
    }

    #[test]
    fn test_format_from_str_rdfxml() {
        assert_eq!(ExportFormat::parse("rdfxml").unwrap(), ExportFormat::RdfXml);
        assert_eq!(ExportFormat::parse("rdf").unwrap(), ExportFormat::RdfXml);
    }

    // --- apply_limit ---

    #[test]
    fn test_apply_limit_some() {
        let triples = sample_triples();
        let limited = DataExporter::apply_limit(&triples, Some(2));
        assert_eq!(limited.len(), 2);
    }

    #[test]
    fn test_apply_limit_none() {
        let triples = sample_triples();
        let limited = DataExporter::apply_limit(&triples, None);
        assert_eq!(limited.len(), 3);
    }

    #[test]
    fn test_apply_limit_larger_than_input() {
        let triples = sample_triples();
        let limited = DataExporter::apply_limit(&triples, Some(100));
        assert_eq!(limited.len(), 3);
    }

    #[test]
    fn test_apply_limit_zero() {
        let triples = sample_triples();
        let limited = DataExporter::apply_limit(&triples, Some(0));
        assert_eq!(limited.len(), 0);
    }

    // --- abbreviate_iri ---

    #[test]
    fn test_abbreviate_iri_matches() {
        let prefixes = sample_prefixes();
        let abbrev = DataExporter::abbreviate_iri("http://xmlns.com/foaf/0.1/name", &prefixes);
        assert_eq!(abbrev, "foaf:name");
    }

    #[test]
    fn test_abbreviate_iri_no_match() {
        let prefixes = sample_prefixes();
        let abbrev = DataExporter::abbreviate_iri("http://unknown.org/foo", &prefixes);
        assert_eq!(abbrev, "http://unknown.org/foo");
    }

    #[test]
    fn test_abbreviate_iri_longest_prefix() {
        let mut prefixes = HashMap::new();
        prefixes.insert("a".into(), "http://example.org/".into());
        prefixes.insert("b".into(), "http://example.org/ns/".into());
        let abbrev = DataExporter::abbreviate_iri("http://example.org/ns/Foo", &prefixes);
        assert_eq!(abbrev, "b:Foo");
    }

    // --- N-Triples ---

    #[test]
    fn test_export_ntriples_format() {
        let triples = sample_triples();
        let out = DataExporter::export_ntriples(&triples);
        assert!(out.contains("<http://example.org/alice>"));
        assert!(out.contains(" .\n"));
    }

    #[test]
    fn test_export_ntriples_literal() {
        let triples = vec![Triple {
            subject: "http://a.org/s".into(),
            predicate: "http://a.org/p".into(),
            object: "\"hello\"".into(),
            graph: None,
        }];
        let out = DataExporter::export_ntriples(&triples);
        assert!(out.contains("\"hello\""));
    }

    #[test]
    fn test_export_ntriples_blank_node() {
        let triples = vec![Triple {
            subject: "_:b1".into(),
            predicate: "http://a.org/p".into(),
            object: "http://a.org/o".into(),
            graph: None,
        }];
        let out = DataExporter::export_ntriples(&triples);
        assert!(out.contains("_:b1"));
    }

    // --- N-Quads ---

    #[test]
    fn test_export_nquads_with_graph() {
        let triples = vec![Triple {
            subject: "http://a.org/s".into(),
            predicate: "http://a.org/p".into(),
            object: "http://a.org/o".into(),
            graph: Some("http://a.org/g".into()),
        }];
        let out = DataExporter::export_nquads(&triples);
        assert!(out.contains("<http://a.org/g>"));
    }

    #[test]
    fn test_export_nquads_without_graph() {
        let triples = vec![Triple {
            subject: "http://a.org/s".into(),
            predicate: "http://a.org/p".into(),
            object: "http://a.org/o".into(),
            graph: None,
        }];
        let out = DataExporter::export_nquads(&triples);
        // Should be: <s> <p> <o> .  (no 4th element before dot)
        let line = out.lines().next().unwrap_or("");
        let parts: Vec<&str> = line.split_whitespace().collect();
        // s p o .
        assert_eq!(parts.len(), 4);
    }

    // --- Turtle ---

    #[test]
    fn test_export_turtle_prefix_declarations() {
        let triples = sample_triples();
        let prefixes = sample_prefixes();
        let opts = ExportOptions {
            format: ExportFormat::Turtle,
            include_prefixes: true,
            ..Default::default()
        };
        let out = DataExporter::export_turtle(&triples, &prefixes, &opts);
        assert!(out.contains("@prefix"));
    }

    #[test]
    fn test_export_turtle_abbreviates_iri() {
        let triples = sample_triples();
        let prefixes = sample_prefixes();
        let opts = ExportOptions {
            format: ExportFormat::Turtle,
            pretty_print: false,
            ..Default::default()
        };
        let out = DataExporter::export_turtle(&triples, &prefixes, &opts);
        assert!(out.contains("foaf:name") || out.contains("ex:"));
    }

    #[test]
    fn test_export_turtle_no_prefixes() {
        let triples = sample_triples();
        let opts = ExportOptions {
            format: ExportFormat::Turtle,
            include_prefixes: false,
            ..Default::default()
        };
        let out = DataExporter::export_turtle(&triples, &HashMap::new(), &opts);
        assert!(!out.contains("@prefix"));
        assert!(out.contains("<http://example.org/alice>"));
    }

    // --- JSON-LD ---

    #[test]
    fn test_export_jsonld_contains_context() {
        let prefixes = sample_prefixes();
        let out = DataExporter::export_jsonld(&sample_triples(), &prefixes);
        assert!(out.contains("\"@context\""));
    }

    #[test]
    fn test_export_jsonld_contains_graph() {
        let out = DataExporter::export_jsonld(&sample_triples(), &HashMap::new());
        assert!(out.contains("\"@graph\""));
    }

    #[test]
    fn test_export_jsonld_contains_id() {
        let out = DataExporter::export_jsonld(&sample_triples(), &HashMap::new());
        assert!(out.contains("\"@id\""));
    }

    // --- CSV ---

    #[test]
    fn test_export_csv_header_with_graph() {
        let out = DataExporter::export_csv(&sample_triples(), true);
        assert!(out.starts_with("subject,predicate,object,graph\n"));
    }

    #[test]
    fn test_export_csv_header_without_graph() {
        let out = DataExporter::export_csv(&sample_triples(), false);
        assert!(out.starts_with("subject,predicate,object\n"));
    }

    #[test]
    fn test_export_csv_row_count() {
        let out = DataExporter::export_csv(&sample_triples(), false);
        // 1 header + 3 data rows
        assert_eq!(out.lines().count(), 4);
    }

    #[test]
    fn test_export_csv_comma_in_value() {
        let triples = vec![Triple {
            subject: "http://a.org/s".into(),
            predicate: "http://a.org/p".into(),
            object: "\"val,ue\"".into(),
            graph: None,
        }];
        let out = DataExporter::export_csv(&triples, false);
        // The literal itself contains a comma; the CSV field should be quoted
        assert!(out.contains('"'));
    }

    // --- RDF/XML ---

    #[test]
    fn test_export_rdfxml_header() {
        let out = DataExporter::export_rdfxml(
            &sample_triples(),
            &sample_prefixes(),
            &ExportOptions::default(),
        );
        assert!(out.contains("<?xml version=\"1.0\""));
        assert!(out.contains("<rdf:RDF"));
        assert!(out.contains("</rdf:RDF>"));
    }

    #[test]
    fn test_export_rdfxml_subject() {
        let out = DataExporter::export_rdfxml(
            &sample_triples(),
            &HashMap::new(),
            &ExportOptions::default(),
        );
        assert!(out.contains("http://example.org/alice"));
    }

    // --- TriG ---

    #[test]
    fn test_export_trig_default_graph() {
        let triples = sample_triples();
        let opts = ExportOptions::default();
        let out = DataExporter::export_trig(&triples, &HashMap::new(), &opts);
        // Default-graph triples go inside bare { }
        assert!(out.contains("{\n"));
    }

    #[test]
    fn test_export_trig_named_graph() {
        let triples = vec![Triple {
            subject: "http://a.org/s".into(),
            predicate: "http://a.org/p".into(),
            object: "http://a.org/o".into(),
            graph: Some("http://a.org/graph1".into()),
        }];
        let opts = ExportOptions::default();
        let out = DataExporter::export_trig(&triples, &HashMap::new(), &opts);
        assert!(out.contains("http://a.org/graph1"));
    }

    // --- Dispatch ---

    #[test]
    fn test_export_dispatch_ntriples() {
        let opts = ExportOptions {
            format: ExportFormat::NTriples,
            ..Default::default()
        };
        let out = DataExporter::export(&sample_triples(), &HashMap::new(), &opts).unwrap();
        assert!(out.contains(" .\n"));
    }

    #[test]
    fn test_export_dispatch_csv() {
        let opts = ExportOptions {
            format: ExportFormat::Csv,
            ..Default::default()
        };
        let out = DataExporter::export(&sample_triples(), &HashMap::new(), &opts).unwrap();
        assert!(out.starts_with("subject,"));
    }

    #[test]
    fn test_export_dispatch_limit() {
        let opts = ExportOptions {
            format: ExportFormat::NTriples,
            limit: Some(1),
            ..Default::default()
        };
        let out = DataExporter::export(&sample_triples(), &HashMap::new(), &opts).unwrap();
        assert_eq!(out.lines().count(), 1);
    }

    // --- Error types ---

    #[test]
    fn test_error_unknown_format_display() {
        let e = ExportError::UnknownFormat("xyz".into());
        assert!(e.to_string().contains("xyz"));
    }

    #[test]
    fn test_error_invalid_triple_display() {
        let e = ExportError::InvalidTriple("bad triple".into());
        assert!(e.to_string().contains("bad triple"));
    }
}
