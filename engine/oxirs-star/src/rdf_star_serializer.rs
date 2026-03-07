// RDF-star serialization to multiple formats (v1.1.0 round 11)
//
// Supports Turtle-star, N-Triples-star, JSON-LD-star, and Notation3-star output.

use std::collections::HashMap;

/// A plain (non-quoted) triple used as the base of a quoted triple
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseTriple {
    pub s: String,
    pub p: String,
    pub o: String,
}

impl BaseTriple {
    /// Create a new base triple
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            s: s.into(),
            p: p.into(),
            o: o.into(),
        }
    }
}

/// An RDF-star term: IRI, Literal, BlankNode, or QuotedTriple
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RdfStarTerm {
    Iri(String),
    Literal(String),
    BlankNode(String),
    QuotedTriple(Box<BaseTriple>),
}

/// An RDF-star triple (subject and object can be quoted triples)
#[derive(Debug, Clone)]
pub struct RdfStarTriple {
    pub s: RdfStarTerm,
    pub p: String,
    pub o: RdfStarTerm,
}

impl RdfStarTriple {
    /// Convenience constructor for plain IRI-IRI-IRI triples
    pub fn plain(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            s: RdfStarTerm::Iri(s.into()),
            p: p.into(),
            o: RdfStarTerm::Iri(o.into()),
        }
    }

    /// Convenience constructor for quoted-subject triples
    pub fn with_quoted_subject(qt: BaseTriple, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            s: RdfStarTerm::QuotedTriple(Box::new(qt)),
            p: p.into(),
            o: RdfStarTerm::Iri(o.into()),
        }
    }

    /// Convenience constructor for quoted-object triples
    pub fn with_quoted_object(s: impl Into<String>, p: impl Into<String>, qt: BaseTriple) -> Self {
        Self {
            s: RdfStarTerm::Iri(s.into()),
            p: p.into(),
            o: RdfStarTerm::QuotedTriple(Box::new(qt)),
        }
    }
}

/// The target serialization format
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializationFormat {
    TurtleStar,
    NTriplesStar,
    JsonLdStar,
    Notation3Star,
}

/// Options that influence serialization output
#[derive(Debug, Clone)]
pub struct SerializationOptions {
    /// Pretty-print (indent / line breaks)
    pub pretty: bool,
    /// Optional base IRI for Turtle @base directive
    pub base_iri: Option<String>,
    /// Prefix map for Turtle prefix declarations
    pub prefixes: HashMap<String, String>,
}

impl SerializationOptions {
    pub fn new() -> Self {
        Self {
            pretty: false,
            base_iri: None,
            prefixes: HashMap::new(),
        }
    }

    pub fn with_pretty(mut self) -> Self {
        self.pretty = true;
        self
    }

    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.insert(prefix.into(), iri.into());
        self
    }

    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base_iri = Some(base.into());
        self
    }
}

impl Default for SerializationOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Stateless serializer for RDF-star triples
pub struct RdfStarSerializer;

impl RdfStarSerializer {
    /// Serialize using the specified format and options
    pub fn serialize(
        triples: &[RdfStarTriple],
        format: SerializationFormat,
        opts: &SerializationOptions,
    ) -> String {
        match format {
            SerializationFormat::TurtleStar => Self::serialize_turtle_star(triples, opts),
            SerializationFormat::NTriplesStar => Self::serialize_ntriples_star(triples),
            SerializationFormat::JsonLdStar => Self::serialize_jsonld_star(triples, opts),
            SerializationFormat::Notation3Star => Self::serialize_notation3_star(triples, opts),
        }
    }

    // ── Turtle-star ────────────────────────────────────────────────────────

    /// Serialize to Turtle-star format.
    /// Quoted triples are represented with `<< s p o >>` syntax.
    pub fn serialize_turtle_star(triples: &[RdfStarTriple], opts: &SerializationOptions) -> String {
        let mut out = String::new();

        // Prefix declarations
        let mut sorted_prefixes: Vec<(&String, &String)> = opts.prefixes.iter().collect();
        sorted_prefixes.sort_by_key(|(k, _)| k.as_str());
        for (prefix, iri) in &sorted_prefixes {
            out.push_str(&format!("@prefix {prefix}: <{iri}> .\n"));
        }
        if let Some(base) = &opts.base_iri {
            out.push_str(&format!("@base <{base}> .\n"));
        }
        if !sorted_prefixes.is_empty() || opts.base_iri.is_some() {
            out.push('\n');
        }

        for triple in triples {
            let s = Self::term_to_turtle(&triple.s, &opts.prefixes);
            let p = Self::abbreviate_iri(&triple.p, &opts.prefixes);
            let o = Self::term_to_turtle(&triple.o, &opts.prefixes);
            if opts.pretty {
                out.push_str(&format!("{s}\n    {p}\n    {o} .\n\n"));
            } else {
                out.push_str(&format!("{s} {p} {o} .\n"));
            }
        }

        out
    }

    // ── N-Triples-star ─────────────────────────────────────────────────────

    /// Serialize to N-Triples-star format (one triple per line).
    /// Quoted triples use `<< s p o >>` notation.
    pub fn serialize_ntriples_star(triples: &[RdfStarTriple]) -> String {
        let mut out = String::new();
        for triple in triples {
            let s = Self::term_to_ntriples(&triple.s);
            let p = format!("<{}>", triple.p);
            let o = Self::term_to_ntriples(&triple.o);
            out.push_str(&format!("{s} {p} {o} .\n"));
        }
        out
    }

    // ── JSON-LD-star ───────────────────────────────────────────────────────

    /// Serialize to a JSON-LD-star representation.
    /// Quoted triples are represented as nested `@annotation` objects.
    pub fn serialize_jsonld_star(
        triples: &[RdfStarTriple],
        _opts: &SerializationOptions,
    ) -> String {
        let mut graph_entries: Vec<String> = Vec::new();

        for triple in triples {
            let s_str = Self::term_to_jsonld(&triple.s);
            let p_str = format!("\"{}\"", triple.p);
            let o_str = Self::term_to_jsonld(&triple.o);

            let entry = format!("  {{\"@id\": {s_str}, {p_str}: [{o_str}]}}");
            graph_entries.push(entry);
        }

        format!(
            "{{\n  \"@graph\": [\n{}\n  ]\n}}",
            graph_entries.join(",\n")
        )
    }

    // ── Notation3-star ─────────────────────────────────────────────────────

    /// Serialize to Notation3-star format (N3-like with `{{ s p o }}` for quoted triples)
    pub fn serialize_notation3_star(
        triples: &[RdfStarTriple],
        opts: &SerializationOptions,
    ) -> String {
        let mut out = String::new();
        for (pfx, iri) in &opts.prefixes {
            out.push_str(&format!("@prefix {pfx}: <{iri}> .\n"));
        }
        for triple in triples {
            let s = Self::term_to_n3(&triple.s, &opts.prefixes);
            let p = Self::abbreviate_iri(&triple.p, &opts.prefixes);
            let o = Self::term_to_n3(&triple.o, &opts.prefixes);
            out.push_str(&format!("{s} {p} {o} .\n"));
        }
        out
    }

    // ── Term renderers ─────────────────────────────────────────────────────

    /// Render a term to Turtle-star notation
    pub fn term_to_turtle(term: &RdfStarTerm, prefixes: &HashMap<String, String>) -> String {
        match term {
            RdfStarTerm::Iri(iri) => Self::abbreviate_iri(iri, prefixes),
            RdfStarTerm::Literal(lit) => {
                // If already quoted, keep as-is; otherwise add surrounding quotes
                if lit.starts_with('"') {
                    lit.clone()
                } else {
                    format!("\"{}\"", lit)
                }
            }
            RdfStarTerm::BlankNode(bn) => format!("_:{bn}"),
            RdfStarTerm::QuotedTriple(bt) => {
                let bs = Self::abbreviate_iri(&bt.s, prefixes);
                let bp = Self::abbreviate_iri(&bt.p, prefixes);
                let bo = Self::abbreviate_iri(&bt.o, prefixes);
                format!("<< {bs} {bp} {bo} >>")
            }
        }
    }

    /// Render a term to N-Triples-star notation (no prefix abbreviation)
    pub fn term_to_ntriples(term: &RdfStarTerm) -> String {
        match term {
            RdfStarTerm::Iri(iri) => format!("<{iri}>"),
            RdfStarTerm::Literal(lit) => {
                if lit.starts_with('"') {
                    lit.clone()
                } else {
                    format!("\"{}\"", lit)
                }
            }
            RdfStarTerm::BlankNode(bn) => format!("_:{bn}"),
            RdfStarTerm::QuotedTriple(bt) => {
                format!("<< <{}> <{}> <{}> >>", bt.s, bt.p, bt.o)
            }
        }
    }

    /// Render a term to JSON-LD notation
    fn term_to_jsonld(term: &RdfStarTerm) -> String {
        match term {
            RdfStarTerm::Iri(iri) => format!("\"{}\"", iri),
            RdfStarTerm::Literal(lit) => format!("{{\"@value\": \"{}\"}}", lit),
            RdfStarTerm::BlankNode(bn) => format!("{{\"@id\": \"_:{}\"}}", bn),
            RdfStarTerm::QuotedTriple(bt) => {
                format!(
                    "{{\"@annotation\": {{\"@subject\": \"{}\", \"@predicate\": \"{}\", \"@object\": \"{}\"}}}}",
                    bt.s, bt.p, bt.o
                )
            }
        }
    }

    /// Render a term to N3-like notation (using `{{ }}` for quoted triples)
    fn term_to_n3(term: &RdfStarTerm, prefixes: &HashMap<String, String>) -> String {
        match term {
            RdfStarTerm::QuotedTriple(bt) => {
                let bs = Self::abbreviate_iri(&bt.s, prefixes);
                let bp = Self::abbreviate_iri(&bt.p, prefixes);
                let bo = Self::abbreviate_iri(&bt.o, prefixes);
                format!("{{ {bs} {bp} {bo} }}")
            }
            other => Self::term_to_turtle(other, prefixes),
        }
    }

    /// Abbreviate an IRI using the prefix map; falls back to `<iri>` syntax.
    fn abbreviate_iri(iri: &str, prefixes: &HashMap<String, String>) -> String {
        // Try longest prefix first
        let mut best: Option<(usize, &str, &str)> = None;
        for (prefix, ns) in prefixes {
            if iri.starts_with(ns.as_str()) && ns.len() > best.map_or(0, |(l, _, _)| l) {
                best = Some((ns.len(), prefix.as_str(), iri));
            }
        }
        if let Some((len, prefix, full)) = best {
            format!("{}:{}", prefix, &full[len..])
        } else {
            format!("<{iri}>")
        }
    }

    // ── Predicate helpers ──────────────────────────────────────────────────

    /// True if `term` is a `QuotedTriple` variant
    pub fn is_quoted_triple(term: &RdfStarTerm) -> bool {
        matches!(term, RdfStarTerm::QuotedTriple(_))
    }

    /// Total number of triples in the slice
    pub fn triple_count(triples: &[RdfStarTriple]) -> usize {
        triples.len()
    }

    /// Number of triples where either the subject or object is a quoted triple
    pub fn quoted_triple_count(triples: &[RdfStarTriple]) -> usize {
        triples
            .iter()
            .filter(|t| Self::is_quoted_triple(&t.s) || Self::is_quoted_triple(&t.o))
            .count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn no_opts() -> SerializationOptions {
        SerializationOptions::new()
    }

    fn with_prefix(prefix: &str, iri: &str) -> SerializationOptions {
        SerializationOptions::new().with_prefix(prefix, iri)
    }

    // ── BaseTriple ────────────────────────────────────────────────────────

    #[test]
    fn test_base_triple_new() {
        let bt = BaseTriple::new("s", "p", "o");
        assert_eq!(bt.s, "s");
        assert_eq!(bt.p, "p");
        assert_eq!(bt.o, "o");
    }

    // ── RdfStarTerm ───────────────────────────────────────────────────────

    #[test]
    fn test_is_quoted_triple_true() {
        let t = RdfStarTerm::QuotedTriple(Box::new(BaseTriple::new("s", "p", "o")));
        assert!(RdfStarSerializer::is_quoted_triple(&t));
    }

    #[test]
    fn test_is_quoted_triple_false_for_iri() {
        let t = RdfStarTerm::Iri("http://example.org/x".to_string());
        assert!(!RdfStarSerializer::is_quoted_triple(&t));
    }

    #[test]
    fn test_is_quoted_triple_false_for_literal() {
        let t = RdfStarTerm::Literal("hello".to_string());
        assert!(!RdfStarSerializer::is_quoted_triple(&t));
    }

    #[test]
    fn test_is_quoted_triple_false_for_blank_node() {
        let t = RdfStarTerm::BlankNode("b1".to_string());
        assert!(!RdfStarSerializer::is_quoted_triple(&t));
    }

    // ── triple_count / quoted_triple_count ────────────────────────────────

    #[test]
    fn test_triple_count_empty() {
        assert_eq!(RdfStarSerializer::triple_count(&[]), 0);
    }

    #[test]
    fn test_triple_count_non_empty() {
        let t1 = RdfStarTriple::plain("s", "p", "o");
        let t2 = RdfStarTriple::plain("a", "b", "c");
        assert_eq!(RdfStarSerializer::triple_count(&[t1, t2]), 2);
    }

    #[test]
    fn test_quoted_triple_count_none() {
        let triples = vec![RdfStarTriple::plain("s", "p", "o")];
        assert_eq!(RdfStarSerializer::quoted_triple_count(&triples), 0);
    }

    #[test]
    fn test_quoted_triple_count_subject() {
        let qt = BaseTriple::new("s", "p", "o");
        let t = RdfStarTriple::with_quoted_subject(qt, "q", "obj");
        assert_eq!(RdfStarSerializer::quoted_triple_count(&[t]), 1);
    }

    #[test]
    fn test_quoted_triple_count_object() {
        let qt = BaseTriple::new("s", "p", "o");
        let t = RdfStarTriple::with_quoted_object("subj", "q", qt);
        assert_eq!(RdfStarSerializer::quoted_triple_count(&[t]), 1);
    }

    #[test]
    fn test_quoted_triple_count_mixed() {
        let qt = BaseTriple::new("s", "p", "o");
        let t1 = RdfStarTriple::with_quoted_subject(qt.clone(), "q", "obj");
        let t2 = RdfStarTriple::plain("a", "b", "c");
        assert_eq!(RdfStarSerializer::quoted_triple_count(&[t1, t2]), 1);
    }

    // ── term_to_ntriples ──────────────────────────────────────────────────

    #[test]
    fn test_term_to_ntriples_iri() {
        let t = RdfStarTerm::Iri("http://example.org/x".to_string());
        assert_eq!(
            RdfStarSerializer::term_to_ntriples(&t),
            "<http://example.org/x>"
        );
    }

    #[test]
    fn test_term_to_ntriples_literal_unquoted() {
        let t = RdfStarTerm::Literal("hello".to_string());
        assert_eq!(RdfStarSerializer::term_to_ntriples(&t), "\"hello\"");
    }

    #[test]
    fn test_term_to_ntriples_literal_pre_quoted() {
        let t = RdfStarTerm::Literal("\"world\"".to_string());
        assert_eq!(RdfStarSerializer::term_to_ntriples(&t), "\"world\"");
    }

    #[test]
    fn test_term_to_ntriples_blank_node() {
        let t = RdfStarTerm::BlankNode("b42".to_string());
        assert_eq!(RdfStarSerializer::term_to_ntriples(&t), "_:b42");
    }

    #[test]
    fn test_term_to_ntriples_quoted_triple() {
        let bt = BaseTriple::new("s", "p", "o");
        let t = RdfStarTerm::QuotedTriple(Box::new(bt));
        let out = RdfStarSerializer::term_to_ntriples(&t);
        assert!(out.starts_with("<< "));
        assert!(out.ends_with(" >>"));
        assert!(out.contains("<s>") && out.contains("<p>") && out.contains("<o>"));
    }

    // ── term_to_turtle ────────────────────────────────────────────────────

    #[test]
    fn test_term_to_turtle_iri_no_prefix() {
        let t = RdfStarTerm::Iri("http://example.org/x".to_string());
        let out = RdfStarSerializer::term_to_turtle(&t, &HashMap::new());
        assert_eq!(out, "<http://example.org/x>");
    }

    #[test]
    fn test_term_to_turtle_iri_with_prefix() {
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());
        let t = RdfStarTerm::Iri("http://example.org/Alice".to_string());
        let out = RdfStarSerializer::term_to_turtle(&t, &prefixes);
        assert_eq!(out, "ex:Alice");
    }

    #[test]
    fn test_term_to_turtle_literal() {
        let t = RdfStarTerm::Literal("42".to_string());
        let out = RdfStarSerializer::term_to_turtle(&t, &HashMap::new());
        assert_eq!(out, "\"42\"");
    }

    #[test]
    fn test_term_to_turtle_blank_node() {
        let t = RdfStarTerm::BlankNode("node1".to_string());
        let out = RdfStarSerializer::term_to_turtle(&t, &HashMap::new());
        assert_eq!(out, "_:node1");
    }

    #[test]
    fn test_term_to_turtle_quoted_triple_syntax() {
        let bt = BaseTriple::new(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        );
        let t = RdfStarTerm::QuotedTriple(Box::new(bt));
        let out = RdfStarSerializer::term_to_turtle(&t, &HashMap::new());
        assert!(out.starts_with("<< "));
        assert!(out.ends_with(" >>"));
    }

    // ── serialize_ntriples_star ────────────────────────────────────────────

    #[test]
    fn test_ntriples_star_one_per_line() {
        let triples = vec![
            RdfStarTriple::plain("http://s", "http://p", "http://o"),
            RdfStarTriple::plain("http://a", "http://b", "http://c"),
        ];
        let out = RdfStarSerializer::serialize_ntriples_star(&triples);
        assert_eq!(out.lines().count(), 2);
    }

    #[test]
    fn test_ntriples_star_simple_triple() {
        let t = RdfStarTriple::plain("http://s", "http://p", "http://o");
        let out = RdfStarSerializer::serialize_ntriples_star(&[t]);
        assert!(out.contains("<http://s>"));
        assert!(out.contains("<http://p>"));
        assert!(out.contains("<http://o>"));
        assert!(out.trim_end().ends_with(" ."));
    }

    #[test]
    fn test_ntriples_star_quoted_object() {
        let qt = BaseTriple::new("http://s2", "http://p2", "http://o2");
        let t = RdfStarTriple::with_quoted_object("http://a", "http://b", qt);
        let out = RdfStarSerializer::serialize_ntriples_star(&[t]);
        assert!(out.contains("<< "));
        assert!(out.contains(" >>"));
    }

    #[test]
    fn test_ntriples_star_quoted_subject() {
        let qt = BaseTriple::new("http://s2", "http://p2", "http://o2");
        let t = RdfStarTriple::with_quoted_subject(qt, "http://p", "http://o");
        let out = RdfStarSerializer::serialize_ntriples_star(&[t]);
        assert!(out.contains("<< "));
    }

    // ── serialize_turtle_star ──────────────────────────────────────────────

    #[test]
    fn test_turtle_star_simple_triple() {
        let t = RdfStarTriple::plain(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        );
        let out = RdfStarSerializer::serialize_turtle_star(&[t], &no_opts());
        assert!(out.contains("<http://example.org/s>"));
        assert!(out.trim_end_matches('\n').ends_with('.') || out.contains(".\n"));
    }

    #[test]
    fn test_turtle_star_prefix_declaration() {
        let opts = with_prefix("ex", "http://example.org/");
        let t = RdfStarTriple::plain(
            "http://example.org/Alice",
            "http://example.org/knows",
            "http://example.org/Bob",
        );
        let out = RdfStarSerializer::serialize_turtle_star(&[t], &opts);
        assert!(out.contains("@prefix ex:"));
        assert!(out.contains("ex:Alice"));
        assert!(out.contains("ex:Bob"));
    }

    #[test]
    fn test_turtle_star_prefix_abbreviation() {
        let opts = with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        let t = RdfStarTriple::plain(
            "http://example.org/x",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/C",
        );
        let out = RdfStarSerializer::serialize_turtle_star(&[t], &opts);
        assert!(out.contains("rdf:type"));
    }

    #[test]
    fn test_turtle_star_quoted_triple_double_angle() {
        let qt = BaseTriple::new("http://s", "http://p", "http://o");
        let t = RdfStarTriple::with_quoted_subject(qt, "http://cert", "http://val");
        let out = RdfStarSerializer::serialize_turtle_star(&[t], &no_opts());
        assert!(out.contains("<< "));
        assert!(out.contains(" >>"));
    }

    #[test]
    fn test_turtle_star_pretty_mode() {
        let t = RdfStarTriple::plain("http://s", "http://p", "http://o");
        let opts = SerializationOptions::new().with_pretty();
        let out = RdfStarSerializer::serialize_turtle_star(&[t], &opts);
        // Pretty mode should add newlines/indentation
        assert!(out.contains('\n'));
    }

    // ── serialize_jsonld_star ──────────────────────────────────────────────

    #[test]
    fn test_jsonld_star_has_graph_key() {
        let t = RdfStarTriple::plain("http://s", "http://p", "http://o");
        let out = RdfStarSerializer::serialize_jsonld_star(&[t], &no_opts());
        assert!(out.contains("@graph"));
    }

    #[test]
    fn test_jsonld_star_subject_appears() {
        let t = RdfStarTriple::plain("http://s", "http://p", "http://o");
        let out = RdfStarSerializer::serialize_jsonld_star(&[t], &no_opts());
        assert!(out.contains("http://s"));
    }

    #[test]
    fn test_jsonld_star_quoted_triple_annotation() {
        let qt = BaseTriple::new("http://s2", "http://p2", "http://o2");
        let t = RdfStarTriple::with_quoted_object("http://a", "http://b", qt);
        let out = RdfStarSerializer::serialize_jsonld_star(&[t], &no_opts());
        assert!(out.contains("@annotation"));
    }

    // ── dispatch via serialize() ──────────────────────────────────────────

    #[test]
    fn test_serialize_dispatch_ntriples() {
        let t = RdfStarTriple::plain("http://s", "http://p", "http://o");
        let out = RdfStarSerializer::serialize(&[t], SerializationFormat::NTriplesStar, &no_opts());
        assert!(out.contains("<http://s>"));
    }

    #[test]
    fn test_serialize_dispatch_turtle() {
        let t = RdfStarTriple::plain("http://s", "http://p", "http://o");
        let out = RdfStarSerializer::serialize(&[t], SerializationFormat::TurtleStar, &no_opts());
        assert!(out.contains("http://s"));
    }

    #[test]
    fn test_serialize_dispatch_jsonld() {
        let t = RdfStarTriple::plain("http://s", "http://p", "http://o");
        let out = RdfStarSerializer::serialize(&[t], SerializationFormat::JsonLdStar, &no_opts());
        assert!(out.contains("@graph"));
    }

    #[test]
    fn test_serialize_dispatch_notation3() {
        let t = RdfStarTriple::plain("http://s", "http://p", "http://o");
        let out =
            RdfStarSerializer::serialize(&[t], SerializationFormat::Notation3Star, &no_opts());
        assert!(out.contains("http://s"));
    }

    #[test]
    fn test_ntriples_star_empty_input() {
        let out = RdfStarSerializer::serialize_ntriples_star(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_turtle_star_empty_input() {
        let out = RdfStarSerializer::serialize_turtle_star(&[], &no_opts());
        assert!(out.trim().is_empty());
    }

    #[test]
    fn test_term_to_ntriples_blank_node_format() {
        let t = RdfStarTerm::BlankNode("x1".to_string());
        assert_eq!(RdfStarSerializer::term_to_ntriples(&t), "_:x1");
    }

    #[test]
    fn test_triple_count_large() {
        let triples: Vec<_> = (0..100)
            .map(|i| RdfStarTriple::plain(format!("s{i}"), format!("p{i}"), format!("o{i}")))
            .collect();
        assert_eq!(RdfStarSerializer::triple_count(&triples), 100);
    }

    #[test]
    fn test_quoted_triple_count_both_quoted() {
        let qt_s = BaseTriple::new("s1", "p1", "o1");
        let qt_o = BaseTriple::new("s2", "p2", "o2");
        let t = RdfStarTriple {
            s: RdfStarTerm::QuotedTriple(Box::new(qt_s)),
            p: "http://p".to_string(),
            o: RdfStarTerm::QuotedTriple(Box::new(qt_o)),
        };
        assert_eq!(RdfStarSerializer::quoted_triple_count(&[t]), 1);
    }

    #[test]
    fn test_serialization_format_equality() {
        assert_eq!(
            SerializationFormat::TurtleStar,
            SerializationFormat::TurtleStar
        );
        assert_ne!(
            SerializationFormat::NTriplesStar,
            SerializationFormat::JsonLdStar
        );
    }

    #[test]
    fn test_turtle_star_base_iri_declared() {
        let opts = SerializationOptions::new().with_base("http://base.example.org/");
        let t = RdfStarTriple::plain("http://s", "http://p", "http://o");
        let out = RdfStarSerializer::serialize_turtle_star(&[t], &opts);
        assert!(out.contains("@base"));
        assert!(out.contains("http://base.example.org/"));
    }

    #[test]
    fn test_serialization_options_default() {
        let opts = SerializationOptions::default();
        assert!(!opts.pretty);
        assert!(opts.base_iri.is_none());
        assert!(opts.prefixes.is_empty());
    }
}
