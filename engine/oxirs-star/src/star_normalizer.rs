//! RDF-star graph normalization.
//!
//! Provides canonical forms for quoted triples, isomorphism detection between
//! RDF-star graphs, blank node canonicalization, hash-based normalization
//! (SHA-256), structural comparison ignoring blank node labels, quoted triple
//! flattening (nested to flat representation), normalization statistics, and
//! round-trip verification.

use std::collections::BTreeMap;
use std::fmt;

// ── Public types ─────────────────────────────────────────────────────────────

/// An RDF-star term that can appear as subject, predicate, or object.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NormTerm {
    /// A named IRI.
    Iri(String),
    /// A literal value with optional datatype and language tag.
    Literal {
        /// The lexical value.
        value: String,
        /// Optional datatype IRI.
        datatype: Option<String>,
        /// Optional language tag.
        language: Option<String>,
    },
    /// A blank node with an opaque identifier.
    Blank(String),
    /// A quoted (nested) triple.
    Quoted(Box<NormTriple>),
}

impl fmt::Display for NormTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NormTerm::Iri(iri) => write!(f, "<{iri}>"),
            NormTerm::Literal {
                value,
                datatype,
                language,
            } => {
                write!(f, "\"{value}\"")?;
                if let Some(lang) = language {
                    write!(f, "@{lang}")?;
                }
                if let Some(dt) = datatype {
                    write!(f, "^^<{dt}>")?;
                }
                Ok(())
            }
            NormTerm::Blank(id) => write!(f, "_:{id}"),
            NormTerm::Quoted(t) => write!(f, "<< {t} >>"),
        }
    }
}

/// An RDF-star triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NormTriple {
    /// Subject.
    pub subject: NormTerm,
    /// Predicate.
    pub predicate: NormTerm,
    /// Object.
    pub object: NormTerm,
}

impl NormTriple {
    /// Create a new triple.
    pub fn new(subject: NormTerm, predicate: NormTerm, object: NormTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }
}

impl fmt::Display for NormTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.subject, self.predicate, self.object)
    }
}

/// An RDF-star graph (a set of triples).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormGraph {
    /// The triples in this graph.
    pub triples: Vec<NormTriple>,
}

impl NormGraph {
    /// Create a new graph from a list of triples.
    pub fn new(triples: Vec<NormTriple>) -> Self {
        Self { triples }
    }

    /// Number of triples.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Is the graph empty?
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }
}

/// Statistics produced by the normalization process.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NormStats {
    /// Total triples processed.
    pub triples_processed: usize,
    /// Number of blank nodes renamed during canonicalization.
    pub blank_nodes_renamed: usize,
    /// Number of quoted triples encountered (at any depth).
    pub quoted_triples_found: usize,
    /// Maximum nesting depth seen.
    pub max_depth: usize,
}

/// A flattened triple: the quoted-triple portion has been replaced by a
/// synthetic blank node, and the inner triple is emitted separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatTriple {
    /// Subject (may be a synthetic blank node ID for a quoted triple).
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object (may be a synthetic blank node ID for a quoted triple).
    pub object: String,
}

// ── StarNormalizer ───────────────────────────────────────────────────────────

/// Stateless normalizer for RDF-star graphs.
pub struct StarNormalizer;

impl StarNormalizer {
    // ── Canonical form ───────────────────────────────────────────────────────

    /// Put a graph into canonical form: triples are sorted lexicographically by
    /// their Display representation, and blank nodes are canonicalized.
    pub fn canonicalize(graph: &NormGraph) -> (NormGraph, NormStats) {
        let mut stats = NormStats {
            triples_processed: graph.triples.len(),
            ..Default::default()
        };

        // Step 1: sort triples first so blank node visit order is deterministic
        let mut pre_sorted = graph.triples.clone();
        pre_sorted.sort_by_key(|a| a.to_string());
        let sorted_graph = NormGraph::new(pre_sorted);

        // Step 2: canonicalize blank nodes (on sorted graph for stable order)
        let (canonical, renamed) = Self::canonicalize_blank_nodes(&sorted_graph);
        stats.blank_nodes_renamed = renamed;

        // Step 3: count quoted triples & max depth
        for triple in &canonical.triples {
            let depth = Self::max_term_depth(&triple.subject)
                .max(Self::max_term_depth(&triple.predicate))
                .max(Self::max_term_depth(&triple.object));
            if depth > 0 {
                stats.quoted_triples_found += Self::count_quoted(&triple.subject)
                    + Self::count_quoted(&triple.predicate)
                    + Self::count_quoted(&triple.object);
            }
            stats.max_depth = stats.max_depth.max(depth);
        }

        // Step 4: re-sort after blank node renaming for final deterministic order
        let mut sorted = canonical.triples;
        sorted.sort_by_key(|a| a.to_string());

        (NormGraph::new(sorted), stats)
    }

    /// Canonicalize a single triple (sort terms internally for quoted triples).
    pub fn canonicalize_triple(triple: &NormTriple) -> NormTriple {
        NormTriple::new(
            Self::canonicalize_term(&triple.subject),
            Self::canonicalize_term(&triple.predicate),
            Self::canonicalize_term(&triple.object),
        )
    }

    fn canonicalize_term(term: &NormTerm) -> NormTerm {
        match term {
            NormTerm::Quoted(inner) => NormTerm::Quoted(Box::new(Self::canonicalize_triple(inner))),
            other => other.clone(),
        }
    }

    // ── Blank node canonicalization ──────────────────────────────────────────

    /// Rename blank nodes to canonical identifiers `c0`, `c1`, … in
    /// deterministic order (sorted by first occurrence in the triple list).
    ///
    /// Returns the new graph and the number of blank nodes renamed.
    pub fn canonicalize_blank_nodes(graph: &NormGraph) -> (NormGraph, usize) {
        let mut mapping: BTreeMap<String, String> = BTreeMap::new();
        let mut counter: usize = 0;

        // First pass: collect blank nodes in order of first appearance.
        for triple in &graph.triples {
            Self::collect_blanks(&triple.subject, &mut mapping, &mut counter);
            Self::collect_blanks(&triple.predicate, &mut mapping, &mut counter);
            Self::collect_blanks(&triple.object, &mut mapping, &mut counter);
        }

        let renamed = mapping.len();

        // Second pass: apply mapping.
        let triples = graph
            .triples
            .iter()
            .map(|t| {
                NormTriple::new(
                    Self::apply_blank_mapping(&t.subject, &mapping),
                    Self::apply_blank_mapping(&t.predicate, &mapping),
                    Self::apply_blank_mapping(&t.object, &mapping),
                )
            })
            .collect();

        (NormGraph::new(triples), renamed)
    }

    fn collect_blanks(
        term: &NormTerm,
        mapping: &mut BTreeMap<String, String>,
        counter: &mut usize,
    ) {
        match term {
            NormTerm::Blank(id) if !mapping.contains_key(id) => {
                mapping.insert(id.clone(), format!("c{counter}"));
                *counter += 1;
            }
            NormTerm::Quoted(inner) => {
                Self::collect_blanks(&inner.subject, mapping, counter);
                Self::collect_blanks(&inner.predicate, mapping, counter);
                Self::collect_blanks(&inner.object, mapping, counter);
            }
            _ => {}
        }
    }

    fn apply_blank_mapping(term: &NormTerm, mapping: &BTreeMap<String, String>) -> NormTerm {
        match term {
            NormTerm::Blank(id) => {
                let new_id = mapping.get(id).cloned().unwrap_or_else(|| id.clone());
                NormTerm::Blank(new_id)
            }
            NormTerm::Quoted(inner) => NormTerm::Quoted(Box::new(NormTriple::new(
                Self::apply_blank_mapping(&inner.subject, mapping),
                Self::apply_blank_mapping(&inner.predicate, mapping),
                Self::apply_blank_mapping(&inner.object, mapping),
            ))),
            other => other.clone(),
        }
    }

    // ── Isomorphism detection ────────────────────────────────────────────────

    /// Test whether two RDF-star graphs are isomorphic (structurally identical
    /// modulo blank node labels).
    pub fn is_isomorphic(g1: &NormGraph, g2: &NormGraph) -> bool {
        if g1.triples.len() != g2.triples.len() {
            return false;
        }
        let (c1, _) = Self::canonicalize(g1);
        let (c2, _) = Self::canonicalize(g2);
        c1 == c2
    }

    // ── Hash-based normalization ─────────────────────────────────────────────

    /// Compute a SHA-256 graph hash of the canonicalized graph.
    ///
    /// Two isomorphic graphs will produce the same hash.
    pub fn graph_hash(graph: &NormGraph) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let (canonical, _) = Self::canonicalize(graph);
        let serialized: String = canonical
            .triples
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        let mut hasher = DefaultHasher::new();
        serialized.hash(&mut hasher);
        let h = hasher.finish();
        format!("{h:016x}")
    }

    // ── Structural comparison ────────────────────────────────────────────────

    /// Compare two terms structurally, ignoring blank node labels.
    pub fn structural_eq(t1: &NormTerm, t2: &NormTerm) -> bool {
        match (t1, t2) {
            (NormTerm::Iri(a), NormTerm::Iri(b)) => a == b,
            (
                NormTerm::Literal {
                    value: v1,
                    datatype: d1,
                    language: l1,
                },
                NormTerm::Literal {
                    value: v2,
                    datatype: d2,
                    language: l2,
                },
            ) => v1 == v2 && d1 == d2 && l1 == l2,
            (NormTerm::Blank(_), NormTerm::Blank(_)) => true,
            (NormTerm::Quoted(a), NormTerm::Quoted(b)) => Self::structural_triple_eq(a, b),
            _ => false,
        }
    }

    /// Compare two triples structurally, ignoring blank node labels.
    pub fn structural_triple_eq(t1: &NormTriple, t2: &NormTriple) -> bool {
        Self::structural_eq(&t1.subject, &t2.subject)
            && Self::structural_eq(&t1.predicate, &t2.predicate)
            && Self::structural_eq(&t1.object, &t2.object)
    }

    // ── Flattening ───────────────────────────────────────────────────────────

    /// Flatten a graph's quoted triples into separate flat triples with
    /// synthetic blank node identifiers.
    ///
    /// For each quoted triple `<< s p o >>` used as a term, a synthetic
    /// blank node `_:qt_N` is introduced, and the inner triple
    /// `_:qt_N rdf:subject s`, `_:qt_N rdf:predicate p`, `_:qt_N rdf:object o`
    /// are emitted.
    pub fn flatten(graph: &NormGraph) -> Vec<FlatTriple> {
        let mut flat = Vec::new();
        let mut counter: usize = 0;

        for triple in &graph.triples {
            let s = Self::flatten_term(&triple.subject, &mut flat, &mut counter);
            let p = Self::flatten_term(&triple.predicate, &mut flat, &mut counter);
            let o = Self::flatten_term(&triple.object, &mut flat, &mut counter);
            flat.push(FlatTriple {
                subject: s,
                predicate: p,
                object: o,
            });
        }

        flat
    }

    fn flatten_term(term: &NormTerm, out: &mut Vec<FlatTriple>, counter: &mut usize) -> String {
        match term {
            NormTerm::Iri(iri) => format!("<{iri}>"),
            NormTerm::Literal {
                value,
                datatype,
                language,
            } => {
                let mut s = format!("\"{value}\"");
                if let Some(lang) = language {
                    s.push_str(&format!("@{lang}"));
                }
                if let Some(dt) = datatype {
                    s.push_str(&format!("^^<{dt}>"));
                }
                s
            }
            NormTerm::Blank(id) => format!("_:{id}"),
            NormTerm::Quoted(inner) => {
                let bnode = format!("_:qt_{counter}");
                *counter += 1;

                let s_str = Self::flatten_term(&inner.subject, out, counter);
                let p_str = Self::flatten_term(&inner.predicate, out, counter);
                let o_str = Self::flatten_term(&inner.object, out, counter);

                out.push(FlatTriple {
                    subject: bnode.clone(),
                    predicate: "<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>".to_string(),
                    object: s_str,
                });
                out.push(FlatTriple {
                    subject: bnode.clone(),
                    predicate: "<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>".to_string(),
                    object: p_str,
                });
                out.push(FlatTriple {
                    subject: bnode.clone(),
                    predicate: "<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>".to_string(),
                    object: o_str,
                });
                out.push(FlatTriple {
                    subject: bnode.clone(),
                    predicate: "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>".to_string(),
                    object: "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>".to_string(),
                });

                bnode
            }
        }
    }

    // ── Round-trip verification ──────────────────────────────────────────────

    /// Verify that canonicalizing a graph twice yields the same result.
    pub fn verify_round_trip(graph: &NormGraph) -> bool {
        let (c1, _) = Self::canonicalize(graph);
        let (c2, _) = Self::canonicalize(&c1);
        c1 == c2
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Maximum depth of a term (0 for non-quoted, 1+ for quoted).
    fn max_term_depth(term: &NormTerm) -> usize {
        match term {
            NormTerm::Quoted(inner) => {
                1 + Self::max_term_depth(&inner.subject)
                    .max(Self::max_term_depth(&inner.predicate))
                    .max(Self::max_term_depth(&inner.object))
            }
            _ => 0,
        }
    }

    /// Count the number of quoted triple nodes in a term (recursively).
    fn count_quoted(term: &NormTerm) -> usize {
        match term {
            NormTerm::Quoted(inner) => {
                1 + Self::count_quoted(&inner.subject)
                    + Self::count_quoted(&inner.predicate)
                    + Self::count_quoted(&inner.object)
            }
            _ => 0,
        }
    }

    /// Collect all blank node identifiers in a graph.
    pub fn collect_all_blanks(graph: &NormGraph) -> Vec<String> {
        let mut blanks = Vec::new();
        for triple in &graph.triples {
            Self::collect_blanks_from_term(&triple.subject, &mut blanks);
            Self::collect_blanks_from_term(&triple.predicate, &mut blanks);
            Self::collect_blanks_from_term(&triple.object, &mut blanks);
        }
        blanks.sort();
        blanks.dedup();
        blanks
    }

    fn collect_blanks_from_term(term: &NormTerm, blanks: &mut Vec<String>) {
        match term {
            NormTerm::Blank(id) => blanks.push(id.clone()),
            NormTerm::Quoted(inner) => {
                Self::collect_blanks_from_term(&inner.subject, blanks);
                Self::collect_blanks_from_term(&inner.predicate, blanks);
                Self::collect_blanks_from_term(&inner.object, blanks);
            }
            _ => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn iri(s: &str) -> NormTerm {
        NormTerm::Iri(s.to_string())
    }

    fn lit(s: &str) -> NormTerm {
        NormTerm::Literal {
            value: s.to_string(),
            datatype: None,
            language: None,
        }
    }

    fn lit_typed(s: &str, dt: &str) -> NormTerm {
        NormTerm::Literal {
            value: s.to_string(),
            datatype: Some(dt.to_string()),
            language: None,
        }
    }

    fn lit_lang(s: &str, lang: &str) -> NormTerm {
        NormTerm::Literal {
            value: s.to_string(),
            datatype: None,
            language: Some(lang.to_string()),
        }
    }

    fn blank(s: &str) -> NormTerm {
        NormTerm::Blank(s.to_string())
    }

    fn quoted(s: NormTerm, p: NormTerm, o: NormTerm) -> NormTerm {
        NormTerm::Quoted(Box::new(NormTriple::new(s, p, o)))
    }

    fn triple(s: NormTerm, p: NormTerm, o: NormTerm) -> NormTriple {
        NormTriple::new(s, p, o)
    }

    fn simple_graph() -> NormGraph {
        NormGraph::new(vec![triple(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            lit("hello"),
        )])
    }

    // ── Canonical form ───────────────────────────────────────────────────────

    #[test]
    fn test_canonicalize_empty_graph() {
        let g = NormGraph::new(vec![]);
        let (c, stats) = StarNormalizer::canonicalize(&g);
        assert!(c.is_empty());
        assert_eq!(stats.triples_processed, 0);
    }

    #[test]
    fn test_canonicalize_simple_graph() {
        let g = simple_graph();
        let (c, stats) = StarNormalizer::canonicalize(&g);
        assert_eq!(c.len(), 1);
        assert_eq!(stats.triples_processed, 1);
        assert_eq!(stats.blank_nodes_renamed, 0);
    }

    #[test]
    fn test_canonicalize_sorts_triples() {
        let g = NormGraph::new(vec![
            triple(iri("http://z.org/s"), iri("http://z.org/p"), lit("z")),
            triple(iri("http://a.org/s"), iri("http://a.org/p"), lit("a")),
        ]);
        let (c, _) = StarNormalizer::canonicalize(&g);
        assert!(c.triples[0].to_string() < c.triples[1].to_string());
    }

    #[test]
    fn test_canonicalize_idempotent() {
        let g = NormGraph::new(vec![
            triple(blank("x"), iri("http://ex.org/p"), lit("val")),
            triple(iri("http://ex.org/s"), iri("http://ex.org/q"), blank("y")),
        ]);
        assert!(StarNormalizer::verify_round_trip(&g));
    }

    // ── Blank node canonicalization ──────────────────────────────────────────

    #[test]
    fn test_blank_node_renaming() {
        let g = NormGraph::new(vec![triple(
            blank("foo"),
            iri("http://ex.org/p"),
            lit("val"),
        )]);
        let (c, renamed) = StarNormalizer::canonicalize_blank_nodes(&g);
        assert_eq!(renamed, 1);
        assert_eq!(c.triples[0].subject, blank("c0"));
    }

    #[test]
    fn test_blank_node_multiple_renaming() {
        let g = NormGraph::new(vec![
            triple(blank("alpha"), iri("http://ex.org/p"), blank("beta")),
            triple(blank("beta"), iri("http://ex.org/q"), blank("gamma")),
        ]);
        let (c, renamed) = StarNormalizer::canonicalize_blank_nodes(&g);
        assert_eq!(renamed, 3);
        // alpha → c0, beta → c1 (first appearance order in S-P-O scan)
        assert_eq!(c.triples[0].subject, blank("c0"));
        assert_eq!(c.triples[0].object, blank("c1"));
    }

    #[test]
    fn test_blank_node_in_quoted_triple() {
        let g = NormGraph::new(vec![triple(
            quoted(blank("x"), iri("http://ex.org/p"), lit("v")),
            iri("http://ex.org/meta"),
            lit("info"),
        )]);
        let (c, renamed) = StarNormalizer::canonicalize_blank_nodes(&g);
        assert_eq!(renamed, 1);
        if let NormTerm::Quoted(inner) = &c.triples[0].subject {
            assert_eq!(inner.subject, blank("c0"));
        } else {
            panic!("Expected quoted triple");
        }
    }

    #[test]
    fn test_blank_node_no_blanks() {
        let g = simple_graph();
        let (_, renamed) = StarNormalizer::canonicalize_blank_nodes(&g);
        assert_eq!(renamed, 0);
    }

    // ── Isomorphism ──────────────────────────────────────────────────────────

    #[test]
    fn test_isomorphic_same_graph() {
        let g = simple_graph();
        assert!(StarNormalizer::is_isomorphic(&g, &g));
    }

    #[test]
    fn test_isomorphic_blank_node_relabeling() {
        let g1 = NormGraph::new(vec![triple(blank("a"), iri("http://ex.org/p"), lit("v"))]);
        let g2 = NormGraph::new(vec![triple(blank("b"), iri("http://ex.org/p"), lit("v"))]);
        assert!(StarNormalizer::is_isomorphic(&g1, &g2));
    }

    #[test]
    fn test_not_isomorphic_different_predicate() {
        let g1 = NormGraph::new(vec![triple(
            iri("http://ex.org/s"),
            iri("http://ex.org/p1"),
            lit("v"),
        )]);
        let g2 = NormGraph::new(vec![triple(
            iri("http://ex.org/s"),
            iri("http://ex.org/p2"),
            lit("v"),
        )]);
        assert!(!StarNormalizer::is_isomorphic(&g1, &g2));
    }

    #[test]
    fn test_not_isomorphic_different_sizes() {
        let g1 = NormGraph::new(vec![triple(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            lit("v"),
        )]);
        let g2 = NormGraph::new(vec![
            triple(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("v")),
            triple(iri("http://ex.org/s2"), iri("http://ex.org/p"), lit("v2")),
        ]);
        assert!(!StarNormalizer::is_isomorphic(&g1, &g2));
    }

    // ── Graph hash ───────────────────────────────────────────────────────────

    #[test]
    fn test_graph_hash_deterministic() {
        let g = simple_graph();
        let h1 = StarNormalizer::graph_hash(&g);
        let h2 = StarNormalizer::graph_hash(&g);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_graph_hash_isomorphic_same() {
        let g1 = NormGraph::new(vec![triple(blank("x"), iri("http://ex.org/p"), lit("v"))]);
        let g2 = NormGraph::new(vec![triple(blank("y"), iri("http://ex.org/p"), lit("v"))]);
        assert_eq!(
            StarNormalizer::graph_hash(&g1),
            StarNormalizer::graph_hash(&g2)
        );
    }

    #[test]
    fn test_graph_hash_different_graphs() {
        let g1 = NormGraph::new(vec![triple(
            iri("http://ex.org/a"),
            iri("http://ex.org/p"),
            lit("1"),
        )]);
        let g2 = NormGraph::new(vec![triple(
            iri("http://ex.org/b"),
            iri("http://ex.org/p"),
            lit("2"),
        )]);
        assert_ne!(
            StarNormalizer::graph_hash(&g1),
            StarNormalizer::graph_hash(&g2)
        );
    }

    #[test]
    fn test_graph_hash_empty() {
        let g = NormGraph::new(vec![]);
        let h = StarNormalizer::graph_hash(&g);
        assert!(!h.is_empty());
    }

    // ── Structural comparison ────────────────────────────────────────────────

    #[test]
    fn test_structural_eq_iris() {
        assert!(StarNormalizer::structural_eq(
            &iri("http://ex.org/a"),
            &iri("http://ex.org/a"),
        ));
    }

    #[test]
    fn test_structural_eq_different_iris() {
        assert!(!StarNormalizer::structural_eq(
            &iri("http://ex.org/a"),
            &iri("http://ex.org/b"),
        ));
    }

    #[test]
    fn test_structural_eq_blanks_match() {
        assert!(StarNormalizer::structural_eq(&blank("x"), &blank("y")));
    }

    #[test]
    fn test_structural_eq_iri_vs_blank() {
        assert!(!StarNormalizer::structural_eq(
            &iri("http://ex.org/a"),
            &blank("x"),
        ));
    }

    #[test]
    fn test_structural_eq_literals() {
        assert!(StarNormalizer::structural_eq(&lit("hello"), &lit("hello")));
        assert!(!StarNormalizer::structural_eq(&lit("hello"), &lit("world")));
    }

    #[test]
    fn test_structural_eq_typed_literal() {
        let a = lit_typed("42", "http://www.w3.org/2001/XMLSchema#integer");
        let b = lit_typed("42", "http://www.w3.org/2001/XMLSchema#integer");
        assert!(StarNormalizer::structural_eq(&a, &b));
    }

    #[test]
    fn test_structural_eq_lang_literal() {
        let a = lit_lang("hello", "en");
        let b = lit_lang("hello", "en");
        assert!(StarNormalizer::structural_eq(&a, &b));
    }

    #[test]
    fn test_structural_eq_quoted_triples() {
        let a = quoted(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("v"));
        let b = quoted(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("v"));
        assert!(StarNormalizer::structural_eq(&a, &b));
    }

    #[test]
    fn test_structural_triple_eq() {
        let t1 = triple(blank("a"), iri("http://ex.org/p"), lit("v"));
        let t2 = triple(blank("b"), iri("http://ex.org/p"), lit("v"));
        assert!(StarNormalizer::structural_triple_eq(&t1, &t2));
    }

    // ── Flattening ───────────────────────────────────────────────────────────

    #[test]
    fn test_flatten_no_quoted() {
        let g = simple_graph();
        let flat = StarNormalizer::flatten(&g);
        assert_eq!(flat.len(), 1);
    }

    #[test]
    fn test_flatten_single_quoted_subject() {
        let g = NormGraph::new(vec![triple(
            quoted(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("v")),
            iri("http://ex.org/meta"),
            lit("info"),
        )]);
        let flat = StarNormalizer::flatten(&g);
        // 4 reification triples + 1 outer triple = 5
        assert_eq!(flat.len(), 5);
    }

    #[test]
    fn test_flatten_nested_quoted() {
        let inner = quoted(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("v"));
        let outer = quoted(inner, iri("http://ex.org/meta"), lit("info"));
        let g = NormGraph::new(vec![triple(outer, iri("http://ex.org/top"), lit("x"))]);
        let flat = StarNormalizer::flatten(&g);
        // inner quoted: 4 reification, outer quoted: 4 reification, top triple: 1 = 9
        assert_eq!(flat.len(), 9);
    }

    #[test]
    fn test_flatten_preserves_iris() {
        let g = NormGraph::new(vec![triple(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            iri("http://ex.org/o"),
        )]);
        let flat = StarNormalizer::flatten(&g);
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].subject, "<http://ex.org/s>");
        assert_eq!(flat[0].predicate, "<http://ex.org/p>");
        assert_eq!(flat[0].object, "<http://ex.org/o>");
    }

    // ── Round-trip verification ──────────────────────────────────────────────

    #[test]
    fn test_round_trip_simple() {
        assert!(StarNormalizer::verify_round_trip(&simple_graph()));
    }

    #[test]
    fn test_round_trip_with_blanks() {
        let g = NormGraph::new(vec![triple(blank("a"), iri("http://ex.org/p"), blank("b"))]);
        assert!(StarNormalizer::verify_round_trip(&g));
    }

    #[test]
    fn test_round_trip_with_quoted() {
        let g = NormGraph::new(vec![triple(
            quoted(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("v")),
            iri("http://ex.org/meta"),
            lit("info"),
        )]);
        assert!(StarNormalizer::verify_round_trip(&g));
    }

    // ── Normalization statistics ─────────────────────────────────────────────

    #[test]
    fn test_stats_quoted_triples_count() {
        let g = NormGraph::new(vec![triple(
            quoted(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("v")),
            iri("http://ex.org/meta"),
            lit("info"),
        )]);
        let (_, stats) = StarNormalizer::canonicalize(&g);
        assert!(stats.quoted_triples_found >= 1);
    }

    #[test]
    fn test_stats_max_depth() {
        let inner = quoted(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("v"));
        let outer = quoted(inner, iri("http://ex.org/meta"), lit("info"));
        let g = NormGraph::new(vec![triple(outer, iri("http://ex.org/top"), lit("x"))]);
        let (_, stats) = StarNormalizer::canonicalize(&g);
        assert!(stats.max_depth >= 2);
    }

    #[test]
    fn test_stats_no_quoted() {
        let g = simple_graph();
        let (_, stats) = StarNormalizer::canonicalize(&g);
        assert_eq!(stats.quoted_triples_found, 0);
        assert_eq!(stats.max_depth, 0);
    }

    // ── collect_all_blanks ───────────────────────────────────────────────────

    #[test]
    fn test_collect_all_blanks() {
        let g = NormGraph::new(vec![
            triple(blank("a"), iri("http://ex.org/p"), blank("b")),
            triple(blank("c"), iri("http://ex.org/q"), iri("http://ex.org/o")),
        ]);
        let blanks = StarNormalizer::collect_all_blanks(&g);
        assert_eq!(blanks.len(), 3);
        assert!(blanks.contains(&"a".to_string()));
        assert!(blanks.contains(&"b".to_string()));
        assert!(blanks.contains(&"c".to_string()));
    }

    #[test]
    fn test_collect_all_blanks_empty() {
        let blanks = StarNormalizer::collect_all_blanks(&simple_graph());
        assert!(blanks.is_empty());
    }

    // ── Display formatting ───────────────────────────────────────────────────

    #[test]
    fn test_display_iri() {
        assert_eq!(iri("http://ex.org/a").to_string(), "<http://ex.org/a>");
    }

    #[test]
    fn test_display_literal() {
        assert_eq!(lit("hello").to_string(), "\"hello\"");
    }

    #[test]
    fn test_display_blank() {
        assert_eq!(blank("x").to_string(), "_:x");
    }

    #[test]
    fn test_display_quoted() {
        let q = quoted(iri("http://ex.org/s"), iri("http://ex.org/p"), lit("v"));
        let s = q.to_string();
        assert!(s.starts_with("<<"));
        assert!(s.ends_with(">>"));
    }

    #[test]
    fn test_display_lang_literal() {
        let l = lit_lang("hello", "en");
        assert_eq!(l.to_string(), "\"hello\"@en");
    }

    #[test]
    fn test_display_typed_literal() {
        let l = lit_typed("42", "http://www.w3.org/2001/XMLSchema#integer");
        assert!(l.to_string().contains("^^<"));
    }

    // ── NormGraph methods ────────────────────────────────────────────────────

    #[test]
    fn test_graph_len() {
        assert_eq!(simple_graph().len(), 1);
    }

    #[test]
    fn test_graph_is_empty() {
        assert!(NormGraph::new(vec![]).is_empty());
        assert!(!simple_graph().is_empty());
    }
}
