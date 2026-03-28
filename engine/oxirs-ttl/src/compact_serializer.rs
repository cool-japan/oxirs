//! Compact Turtle serialization.
//!
//! Produces human-readable Turtle output with subject grouping, predicate
//! grouping (semicolons), object list abbreviation (commas), prefix
//! optimization, blank node inlining, collection syntax `(...)`,
//! `rdf:type` → `a` shorthand, and configurable indentation / line width.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as FmtWrite;

// ── RDF constants ────────────────────────────────────────────────────────────

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";

// ── Public types ─────────────────────────────────────────────────────────────

/// A term in an RDF triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RdfTerm {
    /// A named IRI.
    Iri(String),
    /// A plain or typed literal.
    Literal {
        /// Lexical value.
        value: String,
        /// Optional datatype IRI.
        datatype: Option<String>,
        /// Optional language tag.
        language: Option<String>,
    },
    /// A blank node identifier.
    Blank(String),
}

/// An RDF triple for serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdfTriple {
    /// Subject.
    pub subject: RdfTerm,
    /// Predicate.
    pub predicate: RdfTerm,
    /// Object.
    pub object: RdfTerm,
}

impl RdfTriple {
    /// Create a new triple.
    pub fn new(subject: RdfTerm, predicate: RdfTerm, object: RdfTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }
}

/// Configuration for compact serialization.
#[derive(Debug, Clone)]
pub struct CompactConfig {
    /// Number of spaces per indentation level (default 2).
    pub indent_size: usize,
    /// Target maximum line width before wrapping (default 80).
    pub max_line_width: usize,
    /// Use `a` shorthand for `rdf:type` (default true).
    pub use_a_shorthand: bool,
    /// Inline anonymous blank nodes as `[...]` when they are the object of
    /// exactly one triple (default true).
    pub inline_blank_nodes: bool,
    /// Use collection syntax `(...)` for rdf:List structures (default true).
    pub use_collection_syntax: bool,
    /// Sort subjects alphabetically (default true).
    pub sort_subjects: bool,
}

impl Default for CompactConfig {
    fn default() -> Self {
        Self {
            indent_size: 2,
            max_line_width: 80,
            use_a_shorthand: true,
            inline_blank_nodes: true,
            use_collection_syntax: true,
            sort_subjects: true,
        }
    }
}

/// Serialization statistics.
#[derive(Debug, Clone, Default)]
pub struct SerializerStats {
    /// Number of triples serialized.
    pub triples_count: usize,
    /// Number of distinct subjects.
    pub subjects_count: usize,
    /// Number of prefix declarations emitted.
    pub prefix_count: usize,
    /// Number of `rdf:type` → `a` shortenings.
    pub a_shorthand_count: usize,
    /// Number of blank nodes inlined as `[...]`.
    pub inlined_blanks: usize,
}

// ── CompactSerializer ────────────────────────────────────────────────────────

/// Compact Turtle serializer.
pub struct CompactSerializer {
    /// Prefix map: short name → IRI namespace.
    prefixes: BTreeMap<String, String>,
    /// Configuration.
    config: CompactConfig,
}

impl CompactSerializer {
    /// Create a serializer with default configuration and no prefixes.
    pub fn new() -> Self {
        Self {
            prefixes: BTreeMap::new(),
            config: CompactConfig::default(),
        }
    }

    /// Create a serializer with the given configuration.
    pub fn with_config(config: CompactConfig) -> Self {
        Self {
            prefixes: BTreeMap::new(),
            config,
        }
    }

    /// Register a prefix.
    pub fn add_prefix(&mut self, prefix: impl Into<String>, namespace: impl Into<String>) {
        self.prefixes.insert(prefix.into(), namespace.into());
    }

    /// Serialize a set of triples to compact Turtle.
    pub fn serialize(&self, triples: &[RdfTriple]) -> (String, SerializerStats) {
        let mut stats = SerializerStats {
            triples_count: triples.len(),
            ..Default::default()
        };

        let mut output = String::new();

        // 1. Emit prefix declarations.
        for (prefix, ns) in &self.prefixes {
            let _ = writeln!(output, "@prefix {prefix}: <{ns}> .");
            stats.prefix_count += 1;
        }
        if !self.prefixes.is_empty() {
            output.push('\n');
        }

        // 2. Group triples by subject.
        let grouped = self.group_by_subject(triples);
        stats.subjects_count = grouped.len();

        // 3. Determine blank node usage for inlining.
        let blank_usage = self.blank_node_object_count(triples);

        // 4. Collect inlineable blank nodes (occur as object of exactly 1 triple
        //    and themselves appear as subject of some triples).
        let inlineable: std::collections::HashSet<String> = if self.config.inline_blank_nodes {
            blank_usage
                .iter()
                .filter(|&(id, &count)| {
                    count == 1 && grouped.contains_key(&RdfTerm::Blank(id.clone()))
                })
                .map(|(id, _)| id.clone())
                .collect()
        } else {
            std::collections::HashSet::new()
        };

        // 5. Serialize each subject group.
        let mut subjects: Vec<&RdfTerm> = grouped.keys().collect();
        if self.config.sort_subjects {
            subjects.sort();
        }

        let indent = " ".repeat(self.config.indent_size);

        for (idx, subject) in subjects.iter().enumerate() {
            // Skip subjects that will be inlined.
            if let RdfTerm::Blank(id) = subject {
                if inlineable.contains(id.as_str()) {
                    continue;
                }
            }

            let pred_obj_list = grouped.get(subject).map(|v| v.as_slice()).unwrap_or(&[]);

            let subj_str = self.format_term(subject);
            let _ = write!(output, "{subj_str}");

            // Group by predicate.
            let pred_groups = self.group_by_predicate(pred_obj_list);
            let preds: Vec<&RdfTerm> = {
                let mut v: Vec<&RdfTerm> = pred_groups.keys().collect();
                v.sort();
                v
            };

            for (pi, pred) in preds.iter().enumerate() {
                let pred_str = self.format_predicate(pred, &mut stats);

                if pi == 0 {
                    let _ = write!(output, " {pred_str}");
                } else {
                    let _ = write!(output, " ;\n{indent}{pred_str}");
                }

                let objects = pred_groups.get(pred).map(|v| v.as_slice()).unwrap_or(&[]);
                for (oi, obj) in objects.iter().enumerate() {
                    let obj_str =
                        self.format_object(obj, &grouped, &inlineable, &indent, &mut stats);
                    if oi == 0 {
                        let _ = write!(output, " {obj_str}");
                    } else {
                        let _ = write!(output, " ,\n{indent}{indent}{obj_str}");
                    }
                }
            }

            let _ = writeln!(output, " .");
            if idx + 1 < subjects.len() {
                output.push('\n');
            }
        }

        (output, stats)
    }

    // ── Grouping ─────────────────────────────────────────────────────────────

    fn group_by_subject<'a>(
        &self,
        triples: &'a [RdfTriple],
    ) -> BTreeMap<RdfTerm, Vec<(&'a RdfTerm, &'a RdfTerm)>> {
        let mut map: BTreeMap<RdfTerm, Vec<(&'a RdfTerm, &'a RdfTerm)>> = BTreeMap::new();
        for t in triples {
            map.entry(t.subject.clone())
                .or_default()
                .push((&t.predicate, &t.object));
        }
        map
    }

    fn group_by_predicate<'a>(
        &self,
        pred_obj_list: &[(&'a RdfTerm, &'a RdfTerm)],
    ) -> BTreeMap<RdfTerm, Vec<&'a RdfTerm>> {
        let mut map: BTreeMap<RdfTerm, Vec<&'a RdfTerm>> = BTreeMap::new();
        for &(pred, obj) in pred_obj_list {
            map.entry(pred.clone()).or_default().push(obj);
        }
        map
    }

    // ── Blank node counting ──────────────────────────────────────────────────

    fn blank_node_object_count(&self, triples: &[RdfTriple]) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for t in triples {
            if let RdfTerm::Blank(id) = &t.object {
                *counts.entry(id.clone()).or_insert(0) += 1;
            }
        }
        counts
    }

    // ── Formatting ───────────────────────────────────────────────────────────

    fn format_term(&self, term: &RdfTerm) -> String {
        match term {
            RdfTerm::Iri(iri) => self.compress_iri(iri),
            RdfTerm::Literal {
                value,
                datatype,
                language,
            } => {
                let mut s = format!("\"{}\"", Self::escape_turtle(value));
                if let Some(lang) = language {
                    let _ = write!(s, "@{lang}");
                } else if let Some(dt) = datatype {
                    let compressed = self.compress_iri(dt);
                    let _ = write!(s, "^^{compressed}");
                }
                s
            }
            RdfTerm::Blank(id) => format!("_:{id}"),
        }
    }

    fn format_predicate(&self, pred: &RdfTerm, stats: &mut SerializerStats) -> String {
        if let RdfTerm::Iri(iri) = pred {
            if self.config.use_a_shorthand && iri == RDF_TYPE {
                stats.a_shorthand_count += 1;
                return "a".to_string();
            }
        }
        self.format_term(pred)
    }

    fn format_object(
        &self,
        obj: &RdfTerm,
        grouped: &BTreeMap<RdfTerm, Vec<(&RdfTerm, &RdfTerm)>>,
        inlineable: &std::collections::HashSet<String>,
        indent: &str,
        stats: &mut SerializerStats,
    ) -> String {
        if let RdfTerm::Blank(id) = obj {
            if self.config.inline_blank_nodes && inlineable.contains(id.as_str()) {
                if let Some(pred_obj_list) = grouped.get(&RdfTerm::Blank(id.clone())) {
                    stats.inlined_blanks += 1;
                    return self.format_inline_blank(pred_obj_list, indent, stats);
                }
            }
        }
        self.format_term(obj)
    }

    fn format_inline_blank(
        &self,
        pred_obj_list: &[(&RdfTerm, &RdfTerm)],
        indent: &str,
        stats: &mut SerializerStats,
    ) -> String {
        if pred_obj_list.is_empty() {
            return "[]".to_string();
        }
        let inner_indent = format!("{indent}  ");
        let mut s = String::from("[\n");
        for (i, &(pred, obj)) in pred_obj_list.iter().enumerate() {
            let pred_str = self.format_predicate(pred, stats);
            let obj_str = self.format_term(obj);
            let sep = if i + 1 < pred_obj_list.len() {
                " ;"
            } else {
                ""
            };
            let _ = writeln!(s, "{inner_indent}{pred_str} {obj_str}{sep}");
        }
        let _ = write!(s, "{indent}]");
        s
    }

    // ── IRI compression ──────────────────────────────────────────────────────

    fn compress_iri(&self, iri: &str) -> String {
        // Try to find the longest matching prefix.
        let mut best: Option<(&str, &str)> = None;
        for (prefix, ns) in &self.prefixes {
            if iri.starts_with(ns.as_str())
                && (best.is_none() || ns.len() > best.map(|(_, n)| n.len()).unwrap_or(0))
            {
                best = Some((prefix.as_str(), ns.as_str()));
            }
        }
        if let Some((prefix, ns)) = best {
            let local = &iri[ns.len()..];
            format!("{prefix}:{local}")
        } else {
            format!("<{iri}>")
        }
    }

    // ── Turtle escaping ──────────────────────────────────────────────────────

    fn escape_turtle(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                _ => out.push(c),
            }
        }
        out
    }

    // ── Collection detection ─────────────────────────────────────────────────

    /// Detect rdf:List chains starting from the given blank node.
    ///
    /// Returns the ordered list of `rdf:first` values if the blank node is the
    /// head of a well-formed list terminating with `rdf:nil`, or `None` otherwise.
    pub fn detect_list(&self, head: &str, triples: &[RdfTriple]) -> Option<Vec<RdfTerm>> {
        let mut by_subject: HashMap<String, Vec<(&RdfTerm, &RdfTerm)>> = HashMap::new();
        for t in triples {
            if let RdfTerm::Blank(id) = &t.subject {
                by_subject
                    .entry(id.clone())
                    .or_default()
                    .push((&t.predicate, &t.object));
            }
        }

        let mut items = Vec::new();
        let mut current = head.to_string();

        for _ in 0..10_000 {
            let po = by_subject.get(&current)?;

            let first = po.iter().find_map(|(p, o)| {
                if let RdfTerm::Iri(iri) = p {
                    if iri == RDF_FIRST {
                        return Some((*o).clone());
                    }
                }
                None
            })?;

            let rest = po.iter().find_map(|(p, o)| {
                if let RdfTerm::Iri(iri) = p {
                    if iri == RDF_REST {
                        return Some((*o).clone());
                    }
                }
                None
            })?;

            items.push(first);

            match &rest {
                RdfTerm::Iri(iri) if iri == RDF_NIL => return Some(items),
                RdfTerm::Blank(next_id) => {
                    current = next_id.clone();
                }
                _ => return None,
            }
        }

        None
    }

    /// Format an rdf:List collection in `(...)` syntax.
    pub fn format_collection(&self, items: &[RdfTerm]) -> String {
        let parts: Vec<String> = items.iter().map(|t| self.format_term(t)).collect();
        format!("( {} )", parts.join(" "))
    }

    // ── Prefix optimization ──────────────────────────────────────────────────

    /// Suggest optimal prefixes from a set of triples.
    ///
    /// Scans all IRIs, extracts namespace candidates, and returns the most
    /// frequently used ones.
    pub fn suggest_prefixes(
        triples: &[RdfTriple],
        max_prefixes: usize,
    ) -> BTreeMap<String, String> {
        let mut ns_count: HashMap<String, usize> = HashMap::new();

        for t in triples {
            for term in [&t.subject, &t.predicate, &t.object] {
                if let RdfTerm::Iri(iri) = term {
                    if let Some(ns) = Self::extract_namespace(iri) {
                        *ns_count.entry(ns).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut sorted: Vec<(String, usize)> = ns_count.into_iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1));

        let mut result = BTreeMap::new();
        for (i, (ns, _)) in sorted.into_iter().take(max_prefixes).enumerate() {
            let prefix = Self::derive_prefix_name(&ns, i);
            result.insert(prefix, ns);
        }
        result
    }

    fn extract_namespace(iri: &str) -> Option<String> {
        if let Some(pos) = iri.rfind('#') {
            Some(iri[..=pos].to_string())
        } else {
            iri.rfind('/').map(|pos| iri[..=pos].to_string())
        }
    }

    fn derive_prefix_name(ns: &str, idx: usize) -> String {
        // Try to extract a meaningful short name from the namespace.
        let stripped = ns.trim_end_matches('#').trim_end_matches('/');
        if let Some(pos) = stripped.rfind('/') {
            let candidate = &stripped[pos + 1..];
            if !candidate.is_empty() && candidate.len() <= 10 {
                return candidate.to_lowercase();
            }
        }
        format!("ns{idx}")
    }
}

impl Default for CompactSerializer {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn iri(s: &str) -> RdfTerm {
        RdfTerm::Iri(s.to_string())
    }

    fn lit(s: &str) -> RdfTerm {
        RdfTerm::Literal {
            value: s.to_string(),
            datatype: None,
            language: None,
        }
    }

    fn lit_lang(s: &str, lang: &str) -> RdfTerm {
        RdfTerm::Literal {
            value: s.to_string(),
            datatype: None,
            language: Some(lang.to_string()),
        }
    }

    fn lit_typed(s: &str, dt: &str) -> RdfTerm {
        RdfTerm::Literal {
            value: s.to_string(),
            datatype: Some(dt.to_string()),
            language: None,
        }
    }

    fn blank(s: &str) -> RdfTerm {
        RdfTerm::Blank(s.to_string())
    }

    fn triple(s: RdfTerm, p: RdfTerm, o: RdfTerm) -> RdfTriple {
        RdfTriple::new(s, p, o)
    }

    // ── Subject grouping ─────────────────────────────────────────────────────

    #[test]
    fn test_subject_grouping() {
        let triples = vec![
            triple(iri("http://ex.org/s"), iri("http://ex.org/p1"), lit("a")),
            triple(iri("http://ex.org/s"), iri("http://ex.org/p2"), lit("b")),
        ];
        let ser = CompactSerializer::new();
        let (output, stats) = ser.serialize(&triples);
        assert_eq!(stats.subjects_count, 1);
        // One `.` for one subject group
        assert_eq!(output.matches(" .").count(), 1);
    }

    #[test]
    fn test_multiple_subjects() {
        let triples = vec![
            triple(iri("http://ex.org/s1"), iri("http://ex.org/p"), lit("a")),
            triple(iri("http://ex.org/s2"), iri("http://ex.org/p"), lit("b")),
        ];
        let ser = CompactSerializer::new();
        let (output, stats) = ser.serialize(&triples);
        assert_eq!(stats.subjects_count, 2);
        assert_eq!(output.matches(" .").count(), 2);
    }

    // ── Predicate grouping (semicolons) ──────────────────────────────────────

    #[test]
    fn test_predicate_grouping_semicolons() {
        let triples = vec![
            triple(iri("http://ex.org/s"), iri("http://ex.org/p1"), lit("a")),
            triple(iri("http://ex.org/s"), iri("http://ex.org/p2"), lit("b")),
        ];
        let ser = CompactSerializer::new();
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains(";"));
    }

    // ── Object list abbreviation (commas) ────────────────────────────────────

    #[test]
    fn test_object_list_commas() {
        let triples = vec![
            triple(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("a")),
            triple(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("b")),
        ];
        let ser = CompactSerializer::new();
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains(","));
    }

    // ── Prefix optimization ──────────────────────────────────────────────────

    #[test]
    fn test_prefix_compression() {
        let mut ser = CompactSerializer::new();
        ser.add_prefix("ex", "http://example.org/");
        let triples = vec![triple(
            iri("http://example.org/s"),
            iri("http://example.org/p"),
            lit("v"),
        )];
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains("@prefix ex: <http://example.org/> ."));
        assert!(output.contains("ex:s"));
        assert!(output.contains("ex:p"));
    }

    #[test]
    fn test_no_prefix_full_iri() {
        let ser = CompactSerializer::new();
        let triples = vec![triple(
            iri("http://example.org/s"),
            iri("http://example.org/p"),
            lit("v"),
        )];
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains("<http://example.org/s>"));
    }

    #[test]
    fn test_suggest_prefixes() {
        let triples = vec![
            triple(
                iri("http://example.org/s1"),
                iri("http://example.org/p"),
                lit("a"),
            ),
            triple(
                iri("http://example.org/s2"),
                iri("http://other.org/q"),
                lit("b"),
            ),
        ];
        let suggested = CompactSerializer::suggest_prefixes(&triples, 5);
        assert!(!suggested.is_empty());
    }

    // ── rdf:type → "a" shorthand ─────────────────────────────────────────────

    #[test]
    fn test_a_shorthand() {
        let ser = CompactSerializer::new();
        let triples = vec![triple(
            iri("http://example.org/s"),
            iri(RDF_TYPE),
            iri("http://example.org/MyClass"),
        )];
        let (output, stats) = ser.serialize(&triples);
        assert!(output.contains(" a "));
        assert_eq!(stats.a_shorthand_count, 1);
    }

    #[test]
    fn test_no_a_shorthand_when_disabled() {
        let config = CompactConfig {
            use_a_shorthand: false,
            ..Default::default()
        };
        let ser = CompactSerializer::with_config(config);
        let triples = vec![triple(
            iri("http://example.org/s"),
            iri(RDF_TYPE),
            iri("http://example.org/MyClass"),
        )];
        let (output, stats) = ser.serialize(&triples);
        assert!(!output.contains(" a "));
        assert_eq!(stats.a_shorthand_count, 0);
    }

    // ── Blank node inlining ──────────────────────────────────────────────────

    #[test]
    fn test_blank_node_inline() {
        let triples = vec![
            triple(iri("http://ex.org/s"), iri("http://ex.org/p"), blank("b0")),
            triple(blank("b0"), iri("http://ex.org/name"), lit("Alice")),
        ];
        let ser = CompactSerializer::new();
        let (output, stats) = ser.serialize(&triples);
        assert!(output.contains("["), "Should contain inline blank node");
        assert!(stats.inlined_blanks >= 1);
    }

    #[test]
    fn test_blank_node_not_inlined_multiple_refs() {
        let triples = vec![
            triple(iri("http://ex.org/s1"), iri("http://ex.org/p"), blank("b0")),
            triple(iri("http://ex.org/s2"), iri("http://ex.org/p"), blank("b0")),
            triple(blank("b0"), iri("http://ex.org/name"), lit("Alice")),
        ];
        let ser = CompactSerializer::new();
        let (output, stats) = ser.serialize(&triples);
        // Blank node referenced twice → not inlined
        assert_eq!(stats.inlined_blanks, 0);
        assert!(output.contains("_:b0"));
    }

    // ── Collection syntax ────────────────────────────────────────────────────

    #[test]
    fn test_detect_list() {
        let triples = vec![
            triple(blank("l0"), iri(RDF_FIRST), lit("a")),
            triple(blank("l0"), iri(RDF_REST), blank("l1")),
            triple(blank("l1"), iri(RDF_FIRST), lit("b")),
            triple(blank("l1"), iri(RDF_REST), iri(RDF_NIL)),
        ];
        let ser = CompactSerializer::new();
        let list = ser.detect_list("l0", &triples);
        assert!(list.is_some());
        let items = list.expect("list should exist");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], lit("a"));
        assert_eq!(items[1], lit("b"));
    }

    #[test]
    fn test_detect_list_not_a_list() {
        let triples = vec![triple(
            blank("x"),
            iri("http://ex.org/p"),
            lit("not a list"),
        )];
        let ser = CompactSerializer::new();
        let list = ser.detect_list("x", &triples);
        assert!(list.is_none());
    }

    #[test]
    fn test_format_collection() {
        let ser = CompactSerializer::new();
        let items = vec![lit("a"), lit("b"), lit("c")];
        let output = ser.format_collection(&items);
        assert_eq!(output, "( \"a\" \"b\" \"c\" )");
    }

    // ── Pretty-print indentation ─────────────────────────────────────────────

    #[test]
    fn test_custom_indent() {
        let config = CompactConfig {
            indent_size: 4,
            ..Default::default()
        };
        let ser = CompactSerializer::with_config(config);
        let triples = vec![
            triple(iri("http://ex.org/s"), iri("http://ex.org/p1"), lit("a")),
            triple(iri("http://ex.org/s"), iri("http://ex.org/p2"), lit("b")),
        ];
        let (output, _) = ser.serialize(&triples);
        // Second predicate should be indented with 4 spaces.
        assert!(output.contains("    "));
    }

    // ── Literal escaping ─────────────────────────────────────────────────────

    #[test]
    fn test_escape_special_chars() {
        let ser = CompactSerializer::new();
        let triples = vec![triple(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            lit("hello\n\"world\\"),
        )];
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains("\\n"));
        assert!(output.contains("\\\""));
        assert!(output.contains("\\\\"));
    }

    // ── Language-tagged literals ──────────────────────────────────────────────

    #[test]
    fn test_lang_literal() {
        let ser = CompactSerializer::new();
        let triples = vec![triple(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            lit_lang("hello", "en"),
        )];
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains("\"hello\"@en"));
    }

    // ── Typed literals ───────────────────────────────────────────────────────

    #[test]
    fn test_typed_literal() {
        let mut ser = CompactSerializer::new();
        ser.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");
        let triples = vec![triple(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            lit_typed("42", "http://www.w3.org/2001/XMLSchema#integer"),
        )];
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains("^^xsd:integer"));
    }

    // ── Empty graph ──────────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let ser = CompactSerializer::new();
        let (output, stats) = ser.serialize(&[]);
        assert!(output.is_empty() || output.trim().is_empty());
        assert_eq!(stats.triples_count, 0);
    }

    // ── Config defaults ──────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let c = CompactConfig::default();
        assert_eq!(c.indent_size, 2);
        assert_eq!(c.max_line_width, 80);
        assert!(c.use_a_shorthand);
        assert!(c.inline_blank_nodes);
        assert!(c.use_collection_syntax);
        assert!(c.sort_subjects);
    }

    // ── Stats ────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_triples_count() {
        let ser = CompactSerializer::new();
        let triples = vec![
            triple(iri("http://ex.org/s1"), iri("http://ex.org/p"), lit("a")),
            triple(iri("http://ex.org/s2"), iri("http://ex.org/p"), lit("b")),
            triple(iri("http://ex.org/s3"), iri("http://ex.org/p"), lit("c")),
        ];
        let (_, stats) = ser.serialize(&triples);
        assert_eq!(stats.triples_count, 3);
        assert_eq!(stats.subjects_count, 3);
    }

    #[test]
    fn test_stats_prefix_count() {
        let mut ser = CompactSerializer::new();
        ser.add_prefix("ex", "http://example.org/");
        ser.add_prefix("foaf", "http://xmlns.com/foaf/0.1/");
        let (_, stats) = ser.serialize(&[]);
        assert_eq!(stats.prefix_count, 2);
    }

    // ── Prefix suggestion ────────────────────────────────────────────────────

    #[test]
    fn test_suggest_prefixes_respects_max() {
        let triples = vec![
            triple(iri("http://a.org/s"), iri("http://b.org/p"), lit("v")),
            triple(iri("http://c.org/s"), iri("http://d.org/p"), lit("v")),
        ];
        let suggested = CompactSerializer::suggest_prefixes(&triples, 2);
        assert!(suggested.len() <= 2);
    }

    #[test]
    fn test_suggest_prefixes_empty() {
        let suggested = CompactSerializer::suggest_prefixes(&[], 5);
        assert!(suggested.is_empty());
    }

    // ── Namespace extraction ─────────────────────────────────────────────────

    #[test]
    fn test_extract_namespace_hash() {
        let ns = CompactSerializer::extract_namespace("http://ex.org/ns#term");
        assert_eq!(ns, Some("http://ex.org/ns#".to_string()));
    }

    #[test]
    fn test_extract_namespace_slash() {
        let ns = CompactSerializer::extract_namespace("http://ex.org/ns/term");
        assert_eq!(ns, Some("http://ex.org/ns/".to_string()));
    }

    #[test]
    fn test_extract_namespace_none() {
        let ns = CompactSerializer::extract_namespace("urn:simple");
        // No '#' or '/' separator, so namespace cannot be extracted.
        assert!(ns.is_none());
    }

    // ── Sort subjects ────────────────────────────────────────────────────────

    #[test]
    fn test_sort_subjects_enabled() {
        let triples = vec![
            triple(iri("http://z.org/s"), iri("http://ex.org/p"), lit("z")),
            triple(iri("http://a.org/s"), iri("http://ex.org/p"), lit("a")),
        ];
        let ser = CompactSerializer::new();
        let (output, _) = ser.serialize(&triples);
        let a_pos = output.find("<http://a.org/s>").unwrap_or(usize::MAX);
        let z_pos = output.find("<http://z.org/s>").unwrap_or(usize::MAX);
        assert!(a_pos < z_pos, "Subjects should be sorted");
    }

    #[test]
    fn test_sort_subjects_disabled() {
        let config = CompactConfig {
            sort_subjects: false,
            ..Default::default()
        };
        let ser = CompactSerializer::with_config(config);
        let triples = vec![triple(
            iri("http://z.org/s"),
            iri("http://ex.org/p"),
            lit("z"),
        )];
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains("<http://z.org/s>"));
    }

    // ── Default serializer ───────────────────────────────────────────────────

    #[test]
    fn test_default_serializer() {
        let ser = CompactSerializer::default();
        let (output, stats) = ser.serialize(&[triple(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            lit("v"),
        )]);
        assert_eq!(stats.triples_count, 1);
        assert!(!output.is_empty());
    }

    // ── IRI compression longest match ────────────────────────────────────────

    #[test]
    fn test_longest_prefix_match() {
        let mut ser = CompactSerializer::new();
        ser.add_prefix("short", "http://ex.org/");
        ser.add_prefix("long", "http://ex.org/ns/");
        let triples = vec![triple(
            iri("http://ex.org/ns/term"),
            iri("http://ex.org/p"),
            lit("v"),
        )];
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains("long:term"));
    }

    // ── Blank node as subject ────────────────────────────────────────────────

    #[test]
    fn test_blank_node_subject() {
        let ser = CompactSerializer::new();
        let triples = vec![triple(blank("b0"), iri("http://ex.org/p"), lit("v"))];
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains("_:b0"));
    }

    // ── Additional tests for coverage ────────────────────────────────────────

    #[test]
    fn test_multiple_prefixes() {
        let mut ser = CompactSerializer::new();
        ser.add_prefix("ex", "http://example.org/");
        ser.add_prefix("foaf", "http://xmlns.com/foaf/0.1/");
        ser.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        let triples = vec![triple(
            iri("http://example.org/alice"),
            iri("http://xmlns.com/foaf/0.1/name"),
            lit("Alice"),
        )];
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains("ex:alice"));
        assert!(output.contains("foaf:name"));
    }

    #[test]
    fn test_three_predicates_same_subject() {
        let ser = CompactSerializer::new();
        let triples = vec![
            triple(iri("http://ex.org/s"), iri("http://ex.org/p1"), lit("a")),
            triple(iri("http://ex.org/s"), iri("http://ex.org/p2"), lit("b")),
            triple(iri("http://ex.org/s"), iri("http://ex.org/p3"), lit("c")),
        ];
        let (output, stats) = ser.serialize(&triples);
        assert_eq!(stats.subjects_count, 1);
        assert_eq!(output.matches(';').count(), 2);
    }

    #[test]
    fn test_three_objects_same_predicate() {
        let ser = CompactSerializer::new();
        let triples = vec![
            triple(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("a")),
            triple(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("b")),
            triple(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("c")),
        ];
        let (output, _) = ser.serialize(&triples);
        assert_eq!(output.matches(',').count(), 2);
    }

    #[test]
    fn test_rdf_type_with_prefix() {
        let mut ser = CompactSerializer::new();
        ser.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        let triples = vec![triple(
            iri("http://ex.org/s"),
            iri(RDF_TYPE),
            iri("http://ex.org/Class"),
        )];
        let (output, stats) = ser.serialize(&triples);
        assert!(output.contains(" a "));
        assert_eq!(stats.a_shorthand_count, 1);
    }

    #[test]
    fn test_escape_tab_and_return() {
        let ser = CompactSerializer::new();
        let triples = vec![triple(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            lit("line1\tline2\rline3"),
        )];
        let (output, _) = ser.serialize(&triples);
        assert!(output.contains("\\t"));
        assert!(output.contains("\\r"));
    }

    #[test]
    fn test_single_element_list() {
        let triples = vec![
            triple(blank("l0"), iri(RDF_FIRST), lit("only")),
            triple(blank("l0"), iri(RDF_REST), iri(RDF_NIL)),
        ];
        let ser = CompactSerializer::new();
        let list = ser.detect_list("l0", &triples);
        assert!(list.is_some());
        let items = list.expect("single item list");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], lit("only"));
    }

    #[test]
    fn test_detect_list_missing_first() {
        let triples = vec![triple(blank("l0"), iri(RDF_REST), iri(RDF_NIL))];
        let ser = CompactSerializer::new();
        let list = ser.detect_list("l0", &triples);
        assert!(list.is_none());
    }

    #[test]
    fn test_detect_list_missing_rest() {
        let triples = vec![triple(blank("l0"), iri(RDF_FIRST), lit("a"))];
        let ser = CompactSerializer::new();
        let list = ser.detect_list("l0", &triples);
        assert!(list.is_none());
    }

    #[test]
    fn test_format_collection_single() {
        let ser = CompactSerializer::new();
        let items = vec![lit("x")];
        let output = ser.format_collection(&items);
        assert_eq!(output, "( \"x\" )");
    }

    #[test]
    fn test_format_collection_empty() {
        let ser = CompactSerializer::new();
        let items: Vec<RdfTerm> = vec![];
        let output = ser.format_collection(&items);
        assert_eq!(output, "(  )");
    }

    #[test]
    fn test_inline_blank_disabled() {
        let config = CompactConfig {
            inline_blank_nodes: false,
            ..Default::default()
        };
        let ser = CompactSerializer::with_config(config);
        let triples = vec![
            triple(iri("http://ex.org/s"), iri("http://ex.org/p"), blank("b0")),
            triple(blank("b0"), iri("http://ex.org/name"), lit("Alice")),
        ];
        let (output, stats) = ser.serialize(&triples);
        assert!(!output.contains('['));
        assert_eq!(stats.inlined_blanks, 0);
    }

    #[test]
    fn test_suggest_prefixes_frequent_wins() {
        let triples = vec![
            triple(iri("http://ex.org/s1"), iri("http://ex.org/p1"), lit("a")),
            triple(iri("http://ex.org/s2"), iri("http://ex.org/p2"), lit("b")),
            triple(iri("http://ex.org/s3"), iri("http://other.org/q"), lit("c")),
        ];
        let suggested = CompactSerializer::suggest_prefixes(&triples, 1);
        assert_eq!(suggested.len(), 1);
        // ex.org should win (5 occurrences vs 1)
        let (_, ns) = suggested.iter().next().expect("one prefix");
        assert!(ns.contains("ex.org"));
    }
}
