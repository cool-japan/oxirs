//! # Turtle Pretty Printer with Prefix Analysis
//!
//! Analyses IRI frequency, suggests optimal prefix declarations, and outputs
//! well-formatted Turtle with aligned predicates.
//!
//! ## Features
//!
//! - **IRI frequency analysis**: counts IRI namespaces and suggests prefixes
//! - **Prefix suggestion**: automatically picks short prefixes for common namespaces
//! - **Aligned predicates**: aligns predicate columns for readability
//! - **Subject grouping**: groups triples by subject
//! - **Configurable indentation and line width
//!
//! ## Usage
//!
//! ```rust
//! use oxirs_ttl::pretty_printer::{TurtlePrettyPrinter, PrettyPrinterConfig, RawTriple};
//!
//! let triples = vec![
//!     RawTriple::new(
//!         "http://example.org/alice",
//!         "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
//!         "http://xmlns.com/foaf/0.1/Person",
//!     ),
//!     RawTriple::new(
//!         "http://example.org/alice",
//!         "http://xmlns.com/foaf/0.1/name",
//!         "\"Alice\"",
//!     ),
//! ];
//!
//! let printer = TurtlePrettyPrinter::new();
//! let output = printer.format(&triples);
//! assert!(output.contains("@prefix"));
//! ```

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// RawTriple
// ---------------------------------------------------------------------------

/// A simple triple with string components.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RawTriple {
    /// Subject IRI or blank node.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object IRI, literal, or blank node.
    pub object: String,
}

impl RawTriple {
    /// Create a new raw triple.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// PrefixSuggestion
// ---------------------------------------------------------------------------

/// A suggested prefix declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixSuggestion {
    /// Short prefix name (e.g. "foaf").
    pub prefix: String,
    /// Namespace IRI (e.g. `"http://xmlns.com/foaf/0.1/"`).
    pub namespace: String,
    /// Number of IRIs that use this namespace.
    pub usage_count: usize,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the pretty printer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrettyPrinterConfig {
    /// Indentation string for predicates under a subject.
    pub indent: String,
    /// Whether to align predicate columns.
    pub align_predicates: bool,
    /// Minimum namespace usage count to suggest a prefix.
    pub min_prefix_usage: usize,
    /// Whether to sort subjects alphabetically.
    pub sort_subjects: bool,
    /// Whether to use `a` shorthand for rdf:type.
    pub use_a_shorthand: bool,
    /// Maximum line width (best-effort).
    pub max_line_width: usize,
    /// Custom prefix overrides (prefix -> namespace).
    pub custom_prefixes: HashMap<String, String>,
}

impl Default for PrettyPrinterConfig {
    fn default() -> Self {
        Self {
            indent: "    ".to_string(),
            align_predicates: true,
            min_prefix_usage: 1,
            sort_subjects: true,
            use_a_shorthand: true,
            max_line_width: 80,
            custom_prefixes: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Well-known prefixes
// ---------------------------------------------------------------------------

fn well_known_prefixes() -> Vec<(&'static str, &'static str)> {
    vec![
        ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
        ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
        ("xsd", "http://www.w3.org/2001/XMLSchema#"),
        ("owl", "http://www.w3.org/2002/07/owl#"),
        ("foaf", "http://xmlns.com/foaf/0.1/"),
        ("dc", "http://purl.org/dc/elements/1.1/"),
        ("dcterms", "http://purl.org/dc/terms/"),
        ("skos", "http://www.w3.org/2004/02/skos/core#"),
        ("schema", "http://schema.org/"),
        ("sh", "http://www.w3.org/ns/shacl#"),
        ("geo", "http://www.opengis.net/ont/geosparql#"),
        ("prov", "http://www.w3.org/ns/prov#"),
        ("dcat", "http://www.w3.org/ns/dcat#"),
        ("void", "http://rdfs.org/ns/void#"),
        ("doap", "http://usefulinc.com/ns/doap#"),
        ("vcard", "http://www.w3.org/2006/vcard/ns#"),
    ]
}

// ---------------------------------------------------------------------------
// TurtlePrettyPrinter
// ---------------------------------------------------------------------------

/// The Turtle pretty printer.
pub struct TurtlePrettyPrinter {
    config: PrettyPrinterConfig,
}

impl Default for TurtlePrettyPrinter {
    fn default() -> Self {
        Self::new()
    }
}

impl TurtlePrettyPrinter {
    /// Create a printer with default configuration.
    pub fn new() -> Self {
        Self {
            config: PrettyPrinterConfig::default(),
        }
    }

    /// Create a printer with custom configuration.
    pub fn with_config(config: PrettyPrinterConfig) -> Self {
        Self { config }
    }

    /// Analyse IRI frequency and suggest prefix declarations.
    pub fn analyse_prefixes(&self, triples: &[RawTriple]) -> Vec<PrefixSuggestion> {
        let mut namespace_counts: HashMap<String, usize> = HashMap::new();

        for triple in triples {
            if let Some(ns) = extract_namespace(&triple.subject) {
                *namespace_counts.entry(ns).or_insert(0) += 1;
            }
            if let Some(ns) = extract_namespace(&triple.predicate) {
                *namespace_counts.entry(ns).or_insert(0) += 1;
            }
            if let Some(ns) = extract_namespace(&triple.object) {
                *namespace_counts.entry(ns).or_insert(0) += 1;
            }
        }

        let well_known: HashMap<&str, &str> = well_known_prefixes()
            .into_iter()
            .map(|(p, ns)| (ns, p))
            .collect();

        let mut suggestions: Vec<PrefixSuggestion> = Vec::new();
        let mut used_prefixes: HashMap<String, bool> = HashMap::new();

        // Add custom prefixes first
        for (prefix, ns) in &self.config.custom_prefixes {
            if let Some(count) = namespace_counts.get(ns) {
                suggestions.push(PrefixSuggestion {
                    prefix: prefix.clone(),
                    namespace: ns.clone(),
                    usage_count: *count,
                });
                used_prefixes.insert(ns.clone(), true);
            }
        }

        // Sort namespaces by usage (descending)
        let mut sorted_ns: Vec<(String, usize)> = namespace_counts.into_iter().collect();
        sorted_ns.sort_by(|a, b| b.1.cmp(&a.1));

        let mut prefix_counter = 0_usize;

        for (ns, count) in &sorted_ns {
            if count < &self.config.min_prefix_usage {
                continue;
            }
            if used_prefixes.contains_key(ns) {
                continue;
            }

            let prefix = if let Some(known) = well_known.get(ns.as_str()) {
                known.to_string()
            } else {
                // Generate a prefix like ns0, ns1, ...
                let p = format!("ns{prefix_counter}");
                prefix_counter += 1;
                p
            };

            suggestions.push(PrefixSuggestion {
                prefix,
                namespace: ns.clone(),
                usage_count: *count,
            });
            used_prefixes.insert(ns.clone(), true);
        }

        suggestions
    }

    /// Format triples as pretty-printed Turtle.
    pub fn format(&self, triples: &[RawTriple]) -> String {
        let suggestions = self.analyse_prefixes(triples);
        let prefix_map: HashMap<String, String> = suggestions
            .iter()
            .map(|s| (s.namespace.clone(), s.prefix.clone()))
            .collect();

        let mut out = String::new();

        // Emit prefix declarations
        if !suggestions.is_empty() {
            for s in &suggestions {
                out.push_str(&format!("@prefix {}: <{}> .\n", s.prefix, s.namespace));
            }
            out.push('\n');
        }

        // Group by subject
        let mut subject_groups: BTreeMap<String, Vec<&RawTriple>> = BTreeMap::new();
        for triple in triples {
            subject_groups
                .entry(triple.subject.clone())
                .or_default()
                .push(triple);
        }

        let subjects: Vec<String> = if self.config.sort_subjects {
            subject_groups.keys().cloned().collect()
        } else {
            // Preserve insertion order from BTreeMap (sorted anyway)
            subject_groups.keys().cloned().collect()
        };

        for (si, subject) in subjects.iter().enumerate() {
            let triples_for_subject = &subject_groups[subject];
            let compact_subject = self.compact_iri(subject, &prefix_map);

            // Compute max predicate length for alignment
            let max_pred_len = if self.config.align_predicates {
                triples_for_subject
                    .iter()
                    .map(|t| {
                        let p = self.compact_predicate(&t.predicate, &prefix_map);
                        p.len()
                    })
                    .max()
                    .unwrap_or(0)
            } else {
                0
            };

            for (i, triple) in triples_for_subject.iter().enumerate() {
                let pred = self.compact_predicate(&triple.predicate, &prefix_map);
                let obj = self.compact_object(&triple.object, &prefix_map);

                if i == 0 {
                    out.push_str(&compact_subject);
                } else {
                    out.push_str(&self.config.indent);
                }

                if self.config.align_predicates && i > 0 {
                    out.push_str(&format!("{:width$}", pred, width = max_pred_len));
                } else if i == 0 {
                    out.push(' ');
                    if self.config.align_predicates {
                        out.push_str(&format!("{:width$}", pred, width = max_pred_len));
                    } else {
                        out.push_str(&pred);
                    }
                } else {
                    out.push_str(&pred);
                }

                out.push(' ');
                out.push_str(&obj);

                let is_last = i == triples_for_subject.len() - 1;
                if is_last {
                    out.push_str(" .\n");
                } else {
                    out.push_str(" ;\n");
                }
            }

            if si < subjects.len() - 1 {
                out.push('\n');
            }
        }

        out
    }

    /// Count the number of distinct namespaces in the triples.
    pub fn count_namespaces(&self, triples: &[RawTriple]) -> usize {
        let suggestions = self.analyse_prefixes(triples);
        suggestions.len()
    }

    // -- internal helpers --

    fn compact_iri(&self, iri: &str, prefix_map: &HashMap<String, String>) -> String {
        for (ns, prefix) in prefix_map {
            if iri.starts_with(ns.as_str()) {
                let local = &iri[ns.len()..];
                return format!("{prefix}:{local}");
            }
        }
        format!("<{iri}>")
    }

    fn compact_predicate(&self, iri: &str, prefix_map: &HashMap<String, String>) -> String {
        if self.config.use_a_shorthand && iri == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
            return "a".to_string();
        }
        self.compact_iri(iri, prefix_map)
    }

    fn compact_object(&self, obj: &str, prefix_map: &HashMap<String, String>) -> String {
        if obj.starts_with('"') {
            // literal
            return obj.to_string();
        }
        if obj.starts_with("_:") {
            return obj.to_string();
        }
        self.compact_iri(obj, prefix_map)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Extract namespace from an IRI (everything up to and including the last `#` or `/`).
fn extract_namespace(iri: &str) -> Option<String> {
    if iri.starts_with('"') || iri.starts_with("_:") {
        return None;
    }
    // Try hash first, then slash
    if let Some(pos) = iri.rfind('#') {
        Some(iri[..=pos].to_string())
    } else {
        iri.rfind('/').map(|pos| iri[..=pos].to_string())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn foaf_triples() -> Vec<RawTriple> {
        vec![
            RawTriple::new(
                "http://example.org/alice",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://xmlns.com/foaf/0.1/Person",
            ),
            RawTriple::new(
                "http://example.org/alice",
                "http://xmlns.com/foaf/0.1/name",
                "\"Alice\"",
            ),
            RawTriple::new(
                "http://example.org/alice",
                "http://xmlns.com/foaf/0.1/age",
                "\"30\"",
            ),
            RawTriple::new(
                "http://example.org/bob",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://xmlns.com/foaf/0.1/Person",
            ),
            RawTriple::new(
                "http://example.org/bob",
                "http://xmlns.com/foaf/0.1/name",
                "\"Bob\"",
            ),
        ]
    }

    // -- extract_namespace --

    #[test]
    fn test_extract_namespace_hash() {
        let ns = extract_namespace("http://xmlns.com/foaf/0.1/Person");
        assert_eq!(ns, Some("http://xmlns.com/foaf/0.1/".to_string()));
    }

    #[test]
    fn test_extract_namespace_with_hash_char() {
        let ns = extract_namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        assert_eq!(
            ns,
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string())
        );
    }

    #[test]
    fn test_extract_namespace_literal() {
        assert_eq!(extract_namespace("\"hello\""), None);
    }

    #[test]
    fn test_extract_namespace_blank_node() {
        assert_eq!(extract_namespace("_:b0"), None);
    }

    #[test]
    fn test_extract_namespace_no_separator() {
        assert_eq!(extract_namespace("justAstring"), None);
    }

    // -- prefix analysis --

    #[test]
    fn test_analyse_prefixes_empty() {
        let printer = TurtlePrettyPrinter::new();
        let suggestions = printer.analyse_prefixes(&[]);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_analyse_prefixes_finds_foaf() {
        let printer = TurtlePrettyPrinter::new();
        let suggestions = printer.analyse_prefixes(&foaf_triples());
        assert!(suggestions.iter().any(|s| s.prefix == "foaf"));
    }

    #[test]
    fn test_analyse_prefixes_finds_rdf() {
        let printer = TurtlePrettyPrinter::new();
        let suggestions = printer.analyse_prefixes(&foaf_triples());
        assert!(suggestions.iter().any(|s| s.prefix == "rdf"));
    }

    #[test]
    fn test_analyse_prefixes_unknown_namespace() {
        let printer = TurtlePrettyPrinter::new();
        let triples = vec![RawTriple::new(
            "http://custom.example.com/foo/bar",
            "http://custom.example.com/foo/pred",
            "http://custom.example.com/foo/obj",
        )];
        let suggestions = printer.analyse_prefixes(&triples);
        assert!(!suggestions.is_empty());
        // Should get a generated prefix like ns0
        assert!(suggestions.iter().any(|s| s.prefix.starts_with("ns")));
    }

    #[test]
    fn test_analyse_prefixes_custom_override() {
        let config = PrettyPrinterConfig {
            custom_prefixes: {
                let mut m = HashMap::new();
                m.insert("ex".to_string(), "http://example.org/".to_string());
                m
            },
            ..Default::default()
        };
        let printer = TurtlePrettyPrinter::with_config(config);
        let suggestions = printer.analyse_prefixes(&foaf_triples());
        assert!(suggestions.iter().any(|s| s.prefix == "ex"));
    }

    #[test]
    fn test_analyse_prefixes_min_usage_filter() {
        let config = PrettyPrinterConfig {
            min_prefix_usage: 100,
            ..Default::default()
        };
        let printer = TurtlePrettyPrinter::with_config(config);
        let triples = vec![RawTriple::new(
            "http://rare.example.org/x",
            "http://rare.example.org/p",
            "http://rare.example.org/o",
        )];
        let suggestions = printer.analyse_prefixes(&triples);
        assert!(suggestions.is_empty());
    }

    // -- format output --

    #[test]
    fn test_format_contains_prefix_declarations() {
        let printer = TurtlePrettyPrinter::new();
        let output = printer.format(&foaf_triples());
        assert!(output.contains("@prefix"));
        assert!(output.contains("foaf:"));
    }

    #[test]
    fn test_format_groups_by_subject() {
        let printer = TurtlePrettyPrinter::new();
        let output = printer.format(&foaf_triples());
        // Alice's triples should use semicolons
        assert!(output.contains(";"));
        assert!(output.contains("."));
    }

    #[test]
    fn test_format_uses_a_shorthand() {
        let printer = TurtlePrettyPrinter::new();
        let output = printer.format(&foaf_triples());
        assert!(output.contains(" a "));
    }

    #[test]
    fn test_format_no_a_shorthand() {
        let config = PrettyPrinterConfig {
            use_a_shorthand: false,
            ..Default::default()
        };
        let printer = TurtlePrettyPrinter::with_config(config);
        let triples = vec![RawTriple::new(
            "http://example.org/alice",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://xmlns.com/foaf/0.1/Person",
        )];
        let output = printer.format(&triples);
        assert!(!output.contains(" a "));
    }

    #[test]
    fn test_format_empty_triples() {
        let printer = TurtlePrettyPrinter::new();
        let output = printer.format(&[]);
        assert!(output.is_empty() || output.trim().is_empty());
    }

    #[test]
    fn test_format_single_triple() {
        let printer = TurtlePrettyPrinter::new();
        let triples = vec![RawTriple::new(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )];
        let output = printer.format(&triples);
        assert!(output.contains("."));
    }

    #[test]
    fn test_format_preserves_literals() {
        let printer = TurtlePrettyPrinter::new();
        let triples = vec![RawTriple::new(
            "http://example.org/s",
            "http://example.org/p",
            "\"hello world\"",
        )];
        let output = printer.format(&triples);
        assert!(output.contains("\"hello world\""));
    }

    #[test]
    fn test_format_preserves_blank_nodes() {
        let printer = TurtlePrettyPrinter::new();
        let triples = vec![RawTriple::new(
            "_:b0",
            "http://example.org/p",
            "http://example.org/o",
        )];
        let output = printer.format(&triples);
        assert!(output.contains("_:b0"));
    }

    // -- compact_iri --

    #[test]
    fn test_compact_iri_known_prefix() {
        let printer = TurtlePrettyPrinter::new();
        let mut map = HashMap::new();
        map.insert("http://xmlns.com/foaf/0.1/".to_string(), "foaf".to_string());
        assert_eq!(
            printer.compact_iri("http://xmlns.com/foaf/0.1/Person", &map),
            "foaf:Person"
        );
    }

    #[test]
    fn test_compact_iri_unknown() {
        let printer = TurtlePrettyPrinter::new();
        let map = HashMap::new();
        assert_eq!(
            printer.compact_iri("http://example.org/test", &map),
            "<http://example.org/test>"
        );
    }

    // -- count_namespaces --

    #[test]
    fn test_count_namespaces() {
        let printer = TurtlePrettyPrinter::new();
        let count = printer.count_namespaces(&foaf_triples());
        assert!(count >= 2); // At least foaf and rdf
    }

    // -- PrettyPrinterConfig --

    #[test]
    fn test_default_config() {
        let config = PrettyPrinterConfig::default();
        assert_eq!(config.indent, "    ");
        assert!(config.align_predicates);
        assert!(config.use_a_shorthand);
        assert!(config.sort_subjects);
    }

    // -- RawTriple --

    #[test]
    fn test_raw_triple_new() {
        let t = RawTriple::new("s", "p", "o");
        assert_eq!(t.subject, "s");
        assert_eq!(t.predicate, "p");
        assert_eq!(t.object, "o");
    }

    #[test]
    fn test_raw_triple_eq() {
        let a = RawTriple::new("s", "p", "o");
        let b = RawTriple::new("s", "p", "o");
        assert_eq!(a, b);
    }

    // -- PrefixSuggestion --

    #[test]
    fn test_prefix_suggestion_usage_count() {
        let printer = TurtlePrettyPrinter::new();
        let suggestions = printer.analyse_prefixes(&foaf_triples());
        let foaf_suggestion = suggestions.iter().find(|s| s.prefix == "foaf");
        assert!(foaf_suggestion.is_some());
        assert!(foaf_suggestion.map(|s| s.usage_count).unwrap_or(0) > 0);
    }

    // -- well-known prefixes --

    #[test]
    fn test_well_known_prefixes_not_empty() {
        assert!(!well_known_prefixes().is_empty());
    }

    #[test]
    fn test_well_known_contains_rdf() {
        let wk = well_known_prefixes();
        assert!(wk.iter().any(|(p, _)| *p == "rdf"));
    }

    // -- multi-namespace scenario --

    #[test]
    fn test_many_namespaces() {
        let triples = vec![
            RawTriple::new(
                "http://example.org/s",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://xmlns.com/foaf/0.1/Person",
            ),
            RawTriple::new(
                "http://example.org/s",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "\"Test\"",
            ),
            RawTriple::new(
                "http://example.org/s",
                "http://purl.org/dc/terms/title",
                "\"Title\"",
            ),
        ];
        let printer = TurtlePrettyPrinter::new();
        let output = printer.format(&triples);
        assert!(output.contains("rdf:") || output.contains(" a "));
        assert!(output.contains("rdfs:"));
        assert!(output.contains("dcterms:"));
    }

    // -- alignment --

    #[test]
    fn test_no_alignment() {
        let config = PrettyPrinterConfig {
            align_predicates: false,
            ..Default::default()
        };
        let printer = TurtlePrettyPrinter::with_config(config);
        let triples = vec![
            RawTriple::new(
                "http://example.org/s",
                "http://xmlns.com/foaf/0.1/name",
                "\"Alice\"",
            ),
            RawTriple::new(
                "http://example.org/s",
                "http://xmlns.com/foaf/0.1/age",
                "\"30\"",
            ),
        ];
        let output = printer.format(&triples);
        assert!(output.contains("foaf:name"));
    }

    // -- edge cases --

    #[test]
    fn test_format_many_triples() {
        let triples: Vec<RawTriple> = (0..50)
            .map(|i| {
                RawTriple::new(
                    format!("http://example.org/s{i}"),
                    "http://example.org/p",
                    format!("http://example.org/o{i}"),
                )
            })
            .collect();
        let printer = TurtlePrettyPrinter::new();
        let output = printer.format(&triples);
        assert!(!output.is_empty());
        // Count lines that are NOT prefix declarations ending with " ."
        // Prefix lines look like "@prefix ...: <...> ."
        let dot_count = output
            .lines()
            .filter(|l| l.ends_with(" .") && !l.starts_with("@prefix"))
            .count();
        assert_eq!(dot_count, 50);
    }
}
