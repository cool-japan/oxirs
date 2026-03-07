//! Mapping between RDF reification and RDF-star quoted triples.
//!
//! RDF reification (RDF 1.1) represents statements about statements by
//! introducing a blank node typed as `rdf:Statement` with four properties:
//! `rdf:type`, `rdf:Statement`, `rdf:subject`, `rdf:predicate`, and `rdf:object`.
//!
//! RDF-star (W3C Draft, 2023) extends the RDF data model to allow triples
//! to be used directly as subjects or objects of other triples (quoted triples).
//!
//! This module bridges the two representations, enabling lossless round-trip
//! conversion and interoperability between reification-based and RDF-star graphs.

use std::collections::HashMap;

// ── Vocabulary constants ──────────────────────────────────────────────────────

/// The `rdf:type` predicate IRI.
pub const RDF_TYPE: &str = "rdf:type";
/// The `rdf:Statement` class IRI.
pub const RDF_STATEMENT: &str = "rdf:Statement";
/// The `rdf:subject` predicate IRI.
pub const RDF_SUBJECT: &str = "rdf:subject";
/// The `rdf:predicate` predicate IRI.
pub const RDF_PREDICATE: &str = "rdf:predicate";
/// The `rdf:object` predicate IRI.
pub const RDF_OBJECT: &str = "rdf:object";

// ── RdfTriple ─────────────────────────────────────────────────────────────────

/// A standard RDF triple consisting of subject, predicate, and object strings.
///
/// Subjects and objects may be IRIs, blank-node identifiers (`_:…`), or literal
/// strings. Predicates must be IRIs according to the RDF specification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RdfTriple {
    /// The subject of the triple (IRI or blank node).
    pub subject: String,
    /// The predicate of the triple (IRI).
    pub predicate: String,
    /// The object of the triple (IRI, blank node, or literal).
    pub object: String,
}

impl RdfTriple {
    /// Construct an `RdfTriple` from three string-like values.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        RdfTriple {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

// ── ReificationGroup ─────────────────────────────────────────────────────────

/// An RDF reification group: an `rdf:Statement` node describing a single triple.
///
/// In canonical RDF reification, a statement node `S` typed as `rdf:Statement`
/// carries four triples:
/// ```text
/// S rdf:type      rdf:Statement .
/// S rdf:subject   <s> .
/// S rdf:predicate <p> .
/// S rdf:object    <o> .
/// ```
/// The `annotations` field holds any *additional* predicate–object pairs
/// asserted about `statement_node` (e.g. provenance, certainty).
#[derive(Debug, Clone, PartialEq)]
pub struct ReificationGroup {
    /// The blank node or IRI acting as the `rdf:Statement` resource.
    pub statement_node: String,
    /// The triple being reified.
    pub triple: RdfTriple,
    /// Additional predicate–object annotations on `statement_node`.
    pub annotations: Vec<(String, String)>,
}

impl ReificationGroup {
    /// Construct a `ReificationGroup` with no annotations.
    pub fn new(statement_node: impl Into<String>, triple: RdfTriple) -> Self {
        ReificationGroup {
            statement_node: statement_node.into(),
            triple,
            annotations: Vec::new(),
        }
    }

    /// Construct a `ReificationGroup` with annotations.
    pub fn with_annotations(
        statement_node: impl Into<String>,
        triple: RdfTriple,
        annotations: Vec<(String, String)>,
    ) -> Self {
        ReificationGroup {
            statement_node: statement_node.into(),
            triple,
            annotations,
        }
    }
}

// ── QuotedTripleWithAnnotations ───────────────────────────────────────────────

/// An RDF-star quoted triple together with metadata annotations.
///
/// The `triple` field holds the underlying statement, and `annotations`
/// holds predicate–object pairs that annotate the quoted triple itself.
#[derive(Debug, Clone, PartialEq)]
pub struct QuotedTripleWithAnnotations {
    /// The quoted triple (`<< s p o >>`).
    pub triple: RdfTriple,
    /// Predicate–object annotations on the quoted triple.
    pub annotations: Vec<(String, String)>,
}

impl QuotedTripleWithAnnotations {
    /// Construct a `QuotedTripleWithAnnotations` with no annotations.
    pub fn new(triple: RdfTriple) -> Self {
        QuotedTripleWithAnnotations {
            triple,
            annotations: Vec::new(),
        }
    }

    /// Construct a `QuotedTripleWithAnnotations` with annotations.
    pub fn with_annotations(triple: RdfTriple, annotations: Vec<(String, String)>) -> Self {
        QuotedTripleWithAnnotations {
            triple,
            annotations,
        }
    }
}

// ── ReificationMapper ─────────────────────────────────────────────────────────

/// Converts between RDF reification groups and RDF-star quoted triples.
///
/// The mapper is stateless; all methods are free functions or take `&self`
/// for a consistent API surface.
pub struct ReificationMapper;

impl Default for ReificationMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl ReificationMapper {
    /// Create a new `ReificationMapper`.
    pub fn new() -> Self {
        ReificationMapper
    }

    // ── Core conversion methods ─────────────────────────────────────────────

    /// Convert an RDF reification group into a quoted triple with annotations.
    ///
    /// The `statement_node` identifier is discarded in the output because
    /// RDF-star does not require a surrogate identifier for the quoted triple.
    pub fn reification_to_rdf_star(group: &ReificationGroup) -> QuotedTripleWithAnnotations {
        QuotedTripleWithAnnotations {
            triple: group.triple.clone(),
            annotations: group.annotations.clone(),
        }
    }

    /// Convert a quoted triple with annotations into an RDF reification group.
    ///
    /// `statement_id` becomes the `statement_node` of the resulting group.
    /// Pass a blank-node identifier (e.g. `"_:b0"`) or a full IRI.
    pub fn rdf_star_to_reification(
        quoted: &QuotedTripleWithAnnotations,
        statement_id: &str,
    ) -> ReificationGroup {
        ReificationGroup {
            statement_node: statement_id.to_owned(),
            triple: quoted.triple.clone(),
            annotations: quoted.annotations.clone(),
        }
    }

    /// Expand a `ReificationGroup` into the four canonical reification triples.
    ///
    /// The four triples produced are:
    /// 1. `<statement_node> rdf:type rdf:Statement`
    /// 2. `<statement_node> rdf:subject <s>`
    /// 3. `<statement_node> rdf:predicate <p>`
    /// 4. `<statement_node> rdf:object <o>`
    ///
    /// Annotation triples (if any) are **not** included in the output; callers
    /// that need them should iterate `group.annotations` directly.
    pub fn to_reification_triples(group: &ReificationGroup) -> Vec<RdfTriple> {
        let node = &group.statement_node;
        vec![
            RdfTriple::new(node, RDF_TYPE, RDF_STATEMENT),
            RdfTriple::new(node, RDF_SUBJECT, &group.triple.subject),
            RdfTriple::new(node, RDF_PREDICATE, &group.triple.predicate),
            RdfTriple::new(node, RDF_OBJECT, &group.triple.object),
        ]
    }

    /// Check whether `triples` contains the four reification triples for `node`.
    ///
    /// A complete reification requires all four RDF reification predicates to be
    /// present with `node` as subject:
    /// - `rdf:type rdf:Statement`
    /// - `rdf:subject <any>`
    /// - `rdf:predicate <any>`
    /// - `rdf:object <any>`
    pub fn is_complete_reification(triples: &[RdfTriple], node: &str) -> bool {
        let mut has_type = false;
        let mut has_subject = false;
        let mut has_predicate = false;
        let mut has_object = false;

        for t in triples {
            if t.subject != node {
                continue;
            }
            match t.predicate.as_str() {
                p if p == RDF_TYPE && t.object == RDF_STATEMENT => has_type = true,
                p if p == RDF_SUBJECT => has_subject = true,
                p if p == RDF_PREDICATE => has_predicate = true,
                p if p == RDF_OBJECT => has_object = true,
                _ => {}
            }
        }

        has_type && has_subject && has_predicate && has_object
    }

    // ── Extended helpers (preserved from previous implementation) ───────────

    /// Generate a fresh blank node identifier `_:b{n}` from a counter.
    ///
    /// `counter` is incremented after use so each call produces a unique id.
    pub fn generate_blank_node(counter: &mut u64) -> String {
        let id = format!("_:b{counter}");
        *counter += 1;
        id
    }

    /// Scan `triples` for complete reification patterns and reconstruct groups.
    ///
    /// Returns all `ReificationGroup` values found in `triples`.
    pub fn from_reification_triples(triples: &[RdfTriple]) -> Vec<ReificationGroup> {
        // Collect candidate statement ids (subjects of rdf:type rdf:Statement).
        let stmt_ids: Vec<String> = triples
            .iter()
            .filter(|t| t.predicate == RDF_TYPE && t.object == RDF_STATEMENT)
            .map(|t| t.subject.clone())
            .collect();

        let mut result = Vec::new();

        for stmt_id in stmt_ids {
            let props: HashMap<&str, &str> = triples
                .iter()
                .filter(|t| t.subject == stmt_id)
                .filter_map(|t| {
                    let key = match t.predicate.as_str() {
                        p if p == RDF_SUBJECT => "s",
                        p if p == RDF_PREDICATE => "p",
                        p if p == RDF_OBJECT => "o",
                        _ => return None,
                    };
                    Some((key, t.object.as_str()))
                })
                .collect();

            if let (Some(&s), Some(&p), Some(&o)) = (props.get("s"), props.get("p"), props.get("o"))
            {
                // Collect additional annotations (predicates other than the 4 core ones)
                let annotations: Vec<(String, String)> = triples
                    .iter()
                    .filter(|t| {
                        t.subject == stmt_id
                            && t.predicate != RDF_TYPE
                            && t.predicate != RDF_SUBJECT
                            && t.predicate != RDF_PREDICATE
                            && t.predicate != RDF_OBJECT
                    })
                    .map(|t| (t.predicate.clone(), t.object.clone()))
                    .collect();

                result.push(ReificationGroup::with_annotations(
                    stmt_id.clone(),
                    RdfTriple::new(s, p, o),
                    annotations,
                ));
            }
        }

        result
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RdfTriple ─────────────────────────────────────────────────────────

    #[test]
    fn rdf_triple_new() {
        let t = RdfTriple::new("http://s", "http://p", "http://o");
        assert_eq!(t.subject, "http://s");
        assert_eq!(t.predicate, "http://p");
        assert_eq!(t.object, "http://o");
    }

    #[test]
    fn rdf_triple_equality() {
        let t1 = RdfTriple::new("s", "p", "o");
        let t2 = RdfTriple::new("s", "p", "o");
        assert_eq!(t1, t2);
    }

    #[test]
    fn rdf_triple_clone() {
        let t = RdfTriple::new("s", "p", "o");
        assert_eq!(t.clone(), t);
    }

    // ── ReificationGroup ──────────────────────────────────────────────────

    #[test]
    fn reification_group_new_no_annotations() {
        let t = RdfTriple::new("http://s", "http://p", "http://o");
        let g = ReificationGroup::new("_:b0", t.clone());
        assert_eq!(g.statement_node, "_:b0");
        assert_eq!(g.triple, t);
        assert!(g.annotations.is_empty());
    }

    #[test]
    fn reification_group_with_annotations() {
        let t = RdfTriple::new("http://s", "http://p", "http://o");
        let anns = vec![("http://certainty".to_owned(), "0.9".to_owned())];
        let g = ReificationGroup::with_annotations("_:stmt1", t, anns.clone());
        assert_eq!(g.annotations, anns);
    }

    // ── QuotedTripleWithAnnotations ───────────────────────────────────────

    #[test]
    fn quoted_triple_new_no_annotations() {
        let t = RdfTriple::new("s", "p", "o");
        let q = QuotedTripleWithAnnotations::new(t.clone());
        assert_eq!(q.triple, t);
        assert!(q.annotations.is_empty());
    }

    #[test]
    fn quoted_triple_with_annotations() {
        let t = RdfTriple::new("s", "p", "o");
        let anns = vec![("http://source".to_owned(), "http://db".to_owned())];
        let q = QuotedTripleWithAnnotations::with_annotations(t, anns.clone());
        assert_eq!(q.annotations, anns);
    }

    // ── reification_to_rdf_star ───────────────────────────────────────────

    #[test]
    fn reification_to_rdf_star_basic() {
        let t = RdfTriple::new("http://alice", "http://age", "30");
        let g = ReificationGroup::new("_:stmt", t.clone());
        let q = ReificationMapper::reification_to_rdf_star(&g);
        assert_eq!(q.triple, t);
        assert!(q.annotations.is_empty());
    }

    #[test]
    fn reification_to_rdf_star_preserves_annotations() {
        let t = RdfTriple::new("http://s", "http://p", "http://o");
        let anns = vec![
            ("http://certainty".to_owned(), "0.95".to_owned()),
            ("http://source".to_owned(), "http://wiki".to_owned()),
        ];
        let g = ReificationGroup::with_annotations("_:x", t.clone(), anns.clone());
        let q = ReificationMapper::reification_to_rdf_star(&g);
        assert_eq!(q.triple, t);
        assert_eq!(q.annotations, anns);
    }

    #[test]
    fn reification_to_rdf_star_discards_statement_node() {
        let t = RdfTriple::new("s", "p", "o");
        let g1 = ReificationGroup::new("_:node_a", t.clone());
        let g2 = ReificationGroup::new("_:node_b", t.clone());
        assert_eq!(
            ReificationMapper::reification_to_rdf_star(&g1),
            ReificationMapper::reification_to_rdf_star(&g2)
        );
    }

    // ── rdf_star_to_reification ───────────────────────────────────────────

    #[test]
    fn rdf_star_to_reification_basic() {
        let t = RdfTriple::new("http://alice", "http://age", "30");
        let q = QuotedTripleWithAnnotations::new(t.clone());
        let g = ReificationMapper::rdf_star_to_reification(&q, "_:b0");
        assert_eq!(g.statement_node, "_:b0");
        assert_eq!(g.triple, t);
        assert!(g.annotations.is_empty());
    }

    #[test]
    fn rdf_star_to_reification_custom_id() {
        let t = RdfTriple::new("s", "p", "o");
        let q = QuotedTripleWithAnnotations::new(t);
        let g = ReificationMapper::rdf_star_to_reification(&q, "http://example.org/stmt1");
        assert_eq!(g.statement_node, "http://example.org/stmt1");
    }

    #[test]
    fn rdf_star_to_reification_preserves_annotations() {
        let t = RdfTriple::new("s", "p", "o");
        let anns = vec![("http://p2".to_owned(), "val".to_owned())];
        let q = QuotedTripleWithAnnotations::with_annotations(t, anns.clone());
        let g = ReificationMapper::rdf_star_to_reification(&q, "_:id");
        assert_eq!(g.annotations, anns);
    }

    // ── to_reification_triples ────────────────────────────────────────────

    #[test]
    fn to_reification_triples_produces_exactly_four() {
        let t = RdfTriple::new("http://s", "http://p", "http://o");
        let g = ReificationGroup::new("_:stmt", t);
        let triples = ReificationMapper::to_reification_triples(&g);
        assert_eq!(triples.len(), 4);
    }

    #[test]
    fn to_reification_triples_rdf_type() {
        let t = RdfTriple::new("http://s", "http://p", "http://o");
        let g = ReificationGroup::new("_:stmt", t);
        let triples = ReificationMapper::to_reification_triples(&g);
        assert!(triples.iter().any(|tr| {
            tr.subject == "_:stmt" && tr.predicate == RDF_TYPE && tr.object == RDF_STATEMENT
        }));
    }

    #[test]
    fn to_reification_triples_rdf_subject() {
        let t = RdfTriple::new("http://alice", "http://p", "http://o");
        let g = ReificationGroup::new("_:s", t);
        let triples = ReificationMapper::to_reification_triples(&g);
        assert!(triples.iter().any(|tr| {
            tr.subject == "_:s" && tr.predicate == RDF_SUBJECT && tr.object == "http://alice"
        }));
    }

    #[test]
    fn to_reification_triples_rdf_predicate() {
        let t = RdfTriple::new("http://s", "http://knows", "http://o");
        let g = ReificationGroup::new("_:n", t);
        let triples = ReificationMapper::to_reification_triples(&g);
        assert!(triples.iter().any(|tr| {
            tr.subject == "_:n" && tr.predicate == RDF_PREDICATE && tr.object == "http://knows"
        }));
    }

    #[test]
    fn to_reification_triples_rdf_object() {
        let t = RdfTriple::new("http://s", "http://p", "\"hello\"");
        let g = ReificationGroup::new("_:r", t);
        let triples = ReificationMapper::to_reification_triples(&g);
        assert!(triples.iter().any(|tr| {
            tr.subject == "_:r" && tr.predicate == RDF_OBJECT && tr.object == "\"hello\""
        }));
    }

    #[test]
    fn to_reification_triples_all_subjects_are_node() {
        let t = RdfTriple::new("http://s", "http://p", "http://o");
        let g = ReificationGroup::new("http://node1", t);
        let triples = ReificationMapper::to_reification_triples(&g);
        assert!(triples.iter().all(|tr| tr.subject == "http://node1"));
    }

    // ── is_complete_reification ───────────────────────────────────────────

    #[test]
    fn is_complete_reification_true() {
        let t = RdfTriple::new("http://s", "http://p", "http://o");
        let g = ReificationGroup::new("_:stmt", t);
        let triples = ReificationMapper::to_reification_triples(&g);
        assert!(ReificationMapper::is_complete_reification(
            &triples, "_:stmt"
        ));
    }

    #[test]
    fn is_complete_reification_false_missing_type() {
        let triples = vec![
            RdfTriple::new("_:n", RDF_SUBJECT, "http://s"),
            RdfTriple::new("_:n", RDF_PREDICATE, "http://p"),
            RdfTriple::new("_:n", RDF_OBJECT, "http://o"),
        ];
        assert!(!ReificationMapper::is_complete_reification(&triples, "_:n"));
    }

    #[test]
    fn is_complete_reification_false_missing_subject() {
        let triples = vec![
            RdfTriple::new("_:n", RDF_TYPE, RDF_STATEMENT),
            RdfTriple::new("_:n", RDF_PREDICATE, "http://p"),
            RdfTriple::new("_:n", RDF_OBJECT, "http://o"),
        ];
        assert!(!ReificationMapper::is_complete_reification(&triples, "_:n"));
    }

    #[test]
    fn is_complete_reification_false_missing_predicate() {
        let triples = vec![
            RdfTriple::new("_:n", RDF_TYPE, RDF_STATEMENT),
            RdfTriple::new("_:n", RDF_SUBJECT, "http://s"),
            RdfTriple::new("_:n", RDF_OBJECT, "http://o"),
        ];
        assert!(!ReificationMapper::is_complete_reification(&triples, "_:n"));
    }

    #[test]
    fn is_complete_reification_false_missing_object() {
        let triples = vec![
            RdfTriple::new("_:n", RDF_TYPE, RDF_STATEMENT),
            RdfTriple::new("_:n", RDF_SUBJECT, "http://s"),
            RdfTriple::new("_:n", RDF_PREDICATE, "http://p"),
        ];
        assert!(!ReificationMapper::is_complete_reification(&triples, "_:n"));
    }

    #[test]
    fn is_complete_reification_empty_slice() {
        assert!(!ReificationMapper::is_complete_reification(&[], "_:n"));
    }

    #[test]
    fn is_complete_reification_wrong_node() {
        let t = RdfTriple::new("http://s", "http://p", "http://o");
        let g = ReificationGroup::new("_:correct_node", t);
        let triples = ReificationMapper::to_reification_triples(&g);
        assert!(!ReificationMapper::is_complete_reification(
            &triples,
            "_:wrong_node"
        ));
    }

    #[test]
    fn is_complete_reification_with_extra_triples() {
        let t = RdfTriple::new("http://s", "http://p", "http://o");
        let g = ReificationGroup::new("_:s", t);
        let mut triples = ReificationMapper::to_reification_triples(&g);
        triples.push(RdfTriple::new("_:s", "http://certainty", "0.9"));
        assert!(ReificationMapper::is_complete_reification(&triples, "_:s"));
    }

    // ── Round-trip ────────────────────────────────────────────────────────

    #[test]
    fn round_trip_reification_to_star_and_back() {
        let t = RdfTriple::new("http://ex/alice", "http://ex/age", "\"30\"^^xsd:integer");
        let original = ReificationGroup::new("_:b42", t);

        let quoted = ReificationMapper::reification_to_rdf_star(&original);
        let restored = ReificationMapper::rdf_star_to_reification(&quoted, "_:b42");

        assert_eq!(restored.triple, original.triple);
        assert_eq!(restored.annotations, original.annotations);
        assert_eq!(restored.statement_node, "_:b42");
    }

    #[test]
    fn round_trip_with_annotations() {
        let t = RdfTriple::new("http://s", "http://p", "http://o");
        let anns = vec![
            ("http://certain".to_owned(), "0.8".to_owned()),
            ("http://src".to_owned(), "http://db1".to_owned()),
        ];
        let original = ReificationGroup::with_annotations("_:x", t, anns);

        let quoted = ReificationMapper::reification_to_rdf_star(&original);
        let restored = ReificationMapper::rdf_star_to_reification(&quoted, "_:x");

        assert_eq!(restored.triple, original.triple);
        assert_eq!(restored.annotations, original.annotations);
    }

    // ── Empty annotations ─────────────────────────────────────────────────

    #[test]
    fn reification_to_star_empty_annotations() {
        let t = RdfTriple::new("s", "p", "o");
        let g = ReificationGroup::new("_:n", t);
        let q = ReificationMapper::reification_to_rdf_star(&g);
        assert!(q.annotations.is_empty());
    }

    #[test]
    fn rdf_star_to_reification_empty_annotations() {
        let t = RdfTriple::new("s", "p", "o");
        let q = QuotedTripleWithAnnotations::new(t);
        let g = ReificationMapper::rdf_star_to_reification(&q, "_:n");
        assert!(g.annotations.is_empty());
    }

    // ── Default impl ──────────────────────────────────────────────────────

    #[test]
    fn reification_mapper_default() {
        let _m: ReificationMapper = Default::default();
    }

    // ── from_reification_triples (extended helper) ────────────────────────

    #[test]
    fn from_reification_triples_single() {
        let t = RdfTriple::new("http://alice", "http://knows", "http://bob");
        let g = ReificationGroup::new("_:s0", t);
        let triples = ReificationMapper::to_reification_triples(&g);
        let recovered = ReificationMapper::from_reification_triples(&triples);
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].triple.subject, "http://alice");
    }

    #[test]
    fn from_reification_triples_empty() {
        let recovered = ReificationMapper::from_reification_triples(&[]);
        assert!(recovered.is_empty());
    }

    // ── generate_blank_node ───────────────────────────────────────────────

    #[test]
    fn generate_blank_node_increments() {
        let mut counter = 0u64;
        let b0 = ReificationMapper::generate_blank_node(&mut counter);
        let b1 = ReificationMapper::generate_blank_node(&mut counter);
        assert_eq!(b0, "_:b0");
        assert_eq!(b1, "_:b1");
    }
}
