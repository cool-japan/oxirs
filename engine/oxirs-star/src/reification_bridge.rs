//! Enhanced Reification Bridge: bidirectional conversion between legacy RDF
//! reification (`rdf:Statement`) and RDF-star quoted triples.
//!
//! The [`ReificationBridge`] type provides:
//! - `reification_to_star`: convert a graph containing `rdf:Statement` quads
//!   into an RDF-star graph.
//! - `star_to_reification`: convert an RDF-star graph back to standard RDF
//!   reification patterns.
//! - Validation of both representations.
//! - Detection of incomplete or malformed reification structures.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::reification::vocab;
use crate::{StarError, StarResult};

// ============================================================================
// BridgeConfig
// ============================================================================

/// Configuration for the reification bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Base IRI prefix for generating reification statement identifiers.
    pub base_iri: String,
    /// Whether to preserve annotation triples (predicates other than
    /// `rdf:subject/predicate/object/type`) as meta-triples in the star graph.
    pub preserve_annotations: bool,
    /// Whether to assert the base triple automatically when converting
    /// `reification_to_star` (RDF-star referential opacity: the base triple
    /// is NOT automatically asserted from a reification).
    pub assert_base_triple: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            base_iri: "http://reification.example/stmt/".to_string(),
            preserve_annotations: true,
            assert_base_triple: false, // W3C RDF-star opacity by default.
        }
    }
}

// ============================================================================
// Reification record
// ============================================================================

/// Internal record for one `rdf:Statement` reification cluster.
#[derive(Debug, Clone)]
struct ReificationRecord {
    /// IRI or blank-node identifier for the rdf:Statement node.
    stmt_node: StarTerm,
    /// `rdf:subject` value (if found).
    subject: Option<StarTerm>,
    /// `rdf:predicate` value (if found).
    predicate: Option<StarTerm>,
    /// `rdf:object` value (if found).
    object: Option<StarTerm>,
    /// Any additional annotation triples (pred, obj) pairs.
    annotations: Vec<(StarTerm, StarTerm)>,
    /// Whether `rdf:type rdf:Statement` was found.
    has_type: bool,
}

impl ReificationRecord {
    fn new(stmt_node: StarTerm) -> Self {
        Self {
            stmt_node,
            subject: None,
            predicate: None,
            object: None,
            annotations: Vec::new(),
            has_type: false,
        }
    }

    fn is_complete(&self) -> bool {
        self.subject.is_some() && self.predicate.is_some() && self.object.is_some()
    }

    fn to_star_triples(&self, assert_base: bool) -> StarResult<Vec<StarTriple>> {
        let subj = self
            .subject
            .clone()
            .ok_or_else(|| missing_component("rdf:subject", &self.stmt_node))?;
        let pred = self
            .predicate
            .clone()
            .ok_or_else(|| missing_component("rdf:predicate", &self.stmt_node))?;
        let obj = self
            .object
            .clone()
            .ok_or_else(|| missing_component("rdf:object", &self.stmt_node))?;

        let base = StarTriple::new(subj, pred, obj);
        let mut result = Vec::new();

        if assert_base {
            result.push(base.clone());
        }

        for (ann_pred, ann_obj) in &self.annotations {
            let meta = StarTriple::new(
                StarTerm::quoted_triple(base.clone()),
                ann_pred.clone(),
                ann_obj.clone(),
            );
            result.push(meta);
        }

        if result.is_empty() {
            // No annotations — just produce a meta-triple with rdf:subject as
            // a sentinel so the quoted triple at least appears.
            let meta = StarTriple::new(
                StarTerm::quoted_triple(base.clone()),
                StarTerm::iri(vocab::RDF_SUBJECT).map_err(|e| StarError::QueryError {
                    message: e.to_string(),
                    query_fragment: None,
                    position: None,
                    suggestion: None,
                })?,
                self.stmt_node.clone(),
            );
            if assert_base {
                // base already pushed, just add meta.
                result.push(meta);
            } else {
                result.push(meta);
            }
        }

        Ok(result)
    }
}

fn missing_component(component: &str, stmt_node: &StarTerm) -> StarError {
    StarError::QueryError {
        message: format!(
            "Incomplete reification: missing {component} for statement node {stmt_node:?}"
        ),
        query_fragment: None,
        position: None,
        suggestion: Some(format!("Add {component} triple to the reification")),
    }
}

// ============================================================================
// ReificationBridge
// ============================================================================

/// Bidirectional bridge between legacy RDF reification and RDF-star quoted triples.
///
/// # Example
///
/// ```rust,ignore
/// let bridge = ReificationBridge::new(BridgeConfig::default());
///
/// // Build a graph with rdf:Statement triples.
/// let mut reif_graph = StarGraph::new();
/// // ... fill with rdf:Statement triples ...
///
/// // Convert to RDF-star.
/// let star_graph = bridge.reification_to_star(&reif_graph)?;
/// ```
#[derive(Debug, Clone)]
pub struct ReificationBridge {
    config: BridgeConfig,
    /// Monotone counter for generating statement IRIs.
    counter: u32,
}

impl ReificationBridge {
    /// Create a new bridge with the given configuration.
    pub fn new(config: BridgeConfig) -> Self {
        Self { config, counter: 0 }
    }

    /// Create a bridge with default configuration.
    pub fn default_config() -> Self {
        Self::new(BridgeConfig::default())
    }

    // ------------------------------------------------------------------
    // reification_to_star
    // ------------------------------------------------------------------

    /// Convert a graph using standard RDF reification into an RDF-star graph.
    ///
    /// Scans for all `rdf:Statement` nodes and their associated
    /// `rdf:subject`, `rdf:predicate`, `rdf:object` triples.
    ///
    /// Non-reification triples (those whose subject has no `rdf:type
    /// rdf:Statement`) are copied verbatim into the output graph.
    pub fn reification_to_star(&self, reif_graph: &StarGraph) -> StarResult<StarGraph> {
        let records = self.collect_reification_records(reif_graph);
        let reification_nodes: HashSet<_> = records.keys().cloned().collect();

        let mut star_graph = StarGraph::new();

        // Convert each complete reification record.
        for record in records.values() {
            if record.is_complete() {
                let triples = record.to_star_triples(self.config.assert_base_triple)?;
                for t in triples {
                    let _ = star_graph.insert(t);
                }
            }
        }

        // Copy non-reification triples.
        for triple in reif_graph.triples() {
            if !is_reification_triple(triple, &reification_nodes) {
                let _ = star_graph.insert(triple.clone());
            }
        }

        Ok(star_graph)
    }

    // ------------------------------------------------------------------
    // star_to_reification
    // ------------------------------------------------------------------

    /// Convert an RDF-star graph into a standard RDF reification graph.
    ///
    /// For each triple that has a quoted triple as its subject, a new
    /// `rdf:Statement` node is created (or reused if the same quoted triple
    /// appeared before).
    pub fn star_to_reification(&mut self, star_graph: &StarGraph) -> StarResult<StarGraph> {
        let mut reif_graph = StarGraph::new();
        // Map from canonical triple string → statement node IRI.
        let mut triple_to_stmt: HashMap<String, String> = HashMap::new();

        for triple in star_graph.triples() {
            match &triple.subject {
                StarTerm::QuotedTriple(inner) => {
                    let stmt_iri = self.get_or_create_stmt_iri(&triple_to_stmt, inner);
                    triple_to_stmt
                        .entry(triple_key(inner))
                        .or_insert_with(|| stmt_iri.clone());

                    // rdf:type rdf:Statement
                    let _ = reif_graph.insert(StarTriple::new(
                        StarTerm::iri(&stmt_iri)?,
                        StarTerm::iri(vocab::RDF_TYPE)?,
                        StarTerm::iri(vocab::RDF_STATEMENT)?,
                    ));
                    // rdf:subject
                    let _ = reif_graph.insert(StarTriple::new(
                        StarTerm::iri(&stmt_iri)?,
                        StarTerm::iri(vocab::RDF_SUBJECT)?,
                        inner.subject.clone(),
                    ));
                    // rdf:predicate
                    let _ = reif_graph.insert(StarTriple::new(
                        StarTerm::iri(&stmt_iri)?,
                        StarTerm::iri(vocab::RDF_PREDICATE)?,
                        inner.predicate.clone(),
                    ));
                    // rdf:object
                    let _ = reif_graph.insert(StarTriple::new(
                        StarTerm::iri(&stmt_iri)?,
                        StarTerm::iri(vocab::RDF_OBJECT)?,
                        inner.object.clone(),
                    ));
                    // Annotation triple.
                    let _ = reif_graph.insert(StarTriple::new(
                        StarTerm::iri(&stmt_iri)?,
                        triple.predicate.clone(),
                        triple.object.clone(),
                    ));
                }
                _ => {
                    // Regular triple — copy as-is.
                    let _ = reif_graph.insert(triple.clone());
                }
            }
        }

        Ok(reif_graph)
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    /// Validate that a graph only contains complete reification structures.
    ///
    /// Returns `Err` if any `rdf:Statement` node is missing `rdf:subject`,
    /// `rdf:predicate`, or `rdf:object`.
    pub fn validate_reification_graph(&self, graph: &StarGraph) -> StarResult<()> {
        let records = self.collect_reification_records(graph);
        for (node, record) in &records {
            if !record.is_complete() {
                return Err(StarError::QueryError {
                    message: format!("Incomplete reification for node {node:?}"),
                    query_fragment: None,
                    position: None,
                    suggestion: Some(
                        "Ensure rdf:subject, rdf:predicate, rdf:object are all present".into(),
                    ),
                });
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Collect all `rdf:Statement` node records from a graph.
    fn collect_reification_records(&self, graph: &StarGraph) -> HashMap<String, ReificationRecord> {
        let mut records: HashMap<String, ReificationRecord> = HashMap::new();

        for triple in graph.triples() {
            let subj_key = term_display_key(&triple.subject);

            match triple.predicate.as_named_node().map(|n| n.iri.as_str()) {
                Some(iri) if iri == vocab::RDF_TYPE => {
                    if triple.object.as_named_node().map(|n| n.iri.as_str())
                        == Some(vocab::RDF_STATEMENT)
                    {
                        let rec = records
                            .entry(subj_key.clone())
                            .or_insert_with(|| ReificationRecord::new(triple.subject.clone()));
                        rec.has_type = true;
                    }
                }
                Some(iri) if iri == vocab::RDF_SUBJECT => {
                    let rec = records
                        .entry(subj_key.clone())
                        .or_insert_with(|| ReificationRecord::new(triple.subject.clone()));
                    rec.subject = Some(triple.object.clone());
                }
                Some(iri) if iri == vocab::RDF_PREDICATE => {
                    let rec = records
                        .entry(subj_key.clone())
                        .or_insert_with(|| ReificationRecord::new(triple.subject.clone()));
                    rec.predicate = Some(triple.object.clone());
                }
                Some(iri) if iri == vocab::RDF_OBJECT => {
                    let rec = records
                        .entry(subj_key.clone())
                        .or_insert_with(|| ReificationRecord::new(triple.subject.clone()));
                    rec.object = Some(triple.object.clone());
                }
                _ => {
                    // Potential annotation triple — we'll collect it even if
                    // the stmt node hasn't appeared yet.
                    if self.config.preserve_annotations {
                        let rec = records
                            .entry(subj_key.clone())
                            .or_insert_with(|| ReificationRecord::new(triple.subject.clone()));
                        rec.annotations
                            .push((triple.predicate.clone(), triple.object.clone()));
                    }
                }
            }
        }

        records
    }

    /// Get or create a statement IRI for the given inner triple.
    fn get_or_create_stmt_iri(
        &mut self,
        existing: &HashMap<String, String>,
        inner: &StarTriple,
    ) -> String {
        let key = triple_key(inner);
        if let Some(iri) = existing.get(&key) {
            return iri.clone();
        }
        self.counter += 1;
        format!("{}{}", self.config.base_iri, self.counter)
    }
}

/// Canonical string key for a triple (for deduplication).
fn triple_key(triple: &StarTriple) -> String {
    format!(
        "{:?}|{:?}|{:?}",
        triple.subject, triple.predicate, triple.object
    )
}

fn term_display_key(term: &StarTerm) -> String {
    format!("{term:?}")
}

/// Returns `true` if the triple is a reification structural triple (its
/// subject is a known reification statement node).
fn is_reification_triple(triple: &StarTriple, reification_nodes: &HashSet<String>) -> bool {
    let key = term_display_key(&triple.subject);
    if !reification_nodes.contains(&key) {
        return false;
    }
    // Check that the predicate is one of the four reification predicates.
    matches!(
        triple.predicate.as_named_node().map(|n| n.iri.as_str()),
        Some(p)
            if p == vocab::RDF_TYPE
                || p == vocab::RDF_SUBJECT
                || p == vocab::RDF_PREDICATE
                || p == vocab::RDF_OBJECT
    )
}

// ============================================================================
// Convenience free functions (public API)
// ============================================================================

/// Convert a reification graph to RDF-star using default settings.
pub fn reification_to_star(reif_graph: &StarGraph) -> StarResult<StarGraph> {
    ReificationBridge::default_config().reification_to_star(reif_graph)
}

/// Convert an RDF-star graph to reification using default settings.
pub fn star_to_reification(star_graph: &StarGraph) -> StarResult<StarGraph> {
    ReificationBridge::default_config().star_to_reification(star_graph)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{StarGraph, StarTerm, StarTriple};
    use crate::reification::vocab;

    fn iri(s: &str) -> StarTerm {
        StarTerm::iri(s).expect("valid IRI")
    }

    fn lit(s: &str) -> StarTerm {
        StarTerm::literal(s).expect("ok")
    }

    fn triple(s: &str, p: &str, o: &str) -> StarTriple {
        StarTriple::new(iri(s), iri(p), iri(o))
    }

    /// Build a complete standard RDF reification graph for `s p o` with
    /// statement node `stmt` and an extra annotation `ann_pred ann_obj`.
    fn build_reification_graph(
        stmt: &str,
        s: &str,
        p: &str,
        o: &str,
        ann_pred: Option<(&str, &str)>,
    ) -> StarGraph {
        let mut g = StarGraph::new();
        let _ = g.insert(StarTriple::new(
            iri(stmt),
            iri(vocab::RDF_TYPE),
            iri(vocab::RDF_STATEMENT),
        ));
        let _ = g.insert(StarTriple::new(iri(stmt), iri(vocab::RDF_SUBJECT), iri(s)));
        let _ = g.insert(StarTriple::new(
            iri(stmt),
            iri(vocab::RDF_PREDICATE),
            iri(p),
        ));
        let _ = g.insert(StarTriple::new(iri(stmt), iri(vocab::RDF_OBJECT), iri(o)));
        if let Some((ap, ao)) = ann_pred {
            let _ = g.insert(StarTriple::new(iri(stmt), iri(ap), iri(ao)));
        }
        g
    }

    // ------------------------------------------------------------------
    // reification_to_star
    // ------------------------------------------------------------------

    #[test]
    fn test_reification_to_star_basic() {
        let graph = build_reification_graph(
            "http://ex.org/stmt1",
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
            None,
        );
        let bridge = ReificationBridge::default_config();
        let star = bridge.reification_to_star(&graph).expect("convert ok");
        // Should have one meta-triple.
        assert!(!star.is_empty());
        let t = &star.triples()[0];
        assert!(
            t.subject.is_quoted_triple(),
            "subject should be a quoted triple"
        );
    }

    #[test]
    fn test_reification_to_star_with_annotation() {
        let graph = build_reification_graph(
            "http://ex.org/stmt1",
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
            Some(("http://ex.org/certainty", "http://ex.org/high")),
        );
        let bridge = ReificationBridge::default_config();
        let star = bridge.reification_to_star(&graph).expect("convert ok");

        // Should have at least one meta-triple with certainty annotation.
        let has_cert = star
            .triples()
            .iter()
            .any(|t| t.predicate == iri("http://ex.org/certainty"));
        assert!(has_cert, "annotation should appear as meta-triple");
    }

    #[test]
    fn test_reification_to_star_assert_base_triple() {
        let graph = build_reification_graph(
            "http://ex.org/stmt1",
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
            None,
        );
        let cfg = BridgeConfig {
            assert_base_triple: true,
            ..BridgeConfig::default()
        };
        let bridge = ReificationBridge::new(cfg);
        let star = bridge.reification_to_star(&graph).expect("convert ok");

        let base = triple(
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
        );
        assert!(star.contains(&base), "base triple should be asserted");
    }

    #[test]
    fn test_reification_to_star_no_assert_base_by_default() {
        let graph = build_reification_graph(
            "http://ex.org/stmt1",
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
            None,
        );
        let bridge = ReificationBridge::default_config();
        let star = bridge.reification_to_star(&graph).expect("convert ok");

        let base = triple(
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
        );
        // By default referential opacity means the base triple is NOT asserted.
        assert!(
            !star.contains(&base),
            "base triple should not be auto-asserted"
        );
    }

    // ------------------------------------------------------------------
    // star_to_reification
    // ------------------------------------------------------------------

    #[test]
    fn test_star_to_reification_basic() {
        let mut star_graph = StarGraph::new();
        let inner = triple(
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            iri("http://ex.org/certainty"),
            lit("0.9"),
        );
        let _ = star_graph.insert(meta);

        let mut bridge = ReificationBridge::default_config();
        let reif = bridge.star_to_reification(&star_graph).expect("convert ok");

        // Must contain rdf:subject triple.
        let has_subj = reif
            .triples()
            .iter()
            .any(|t| t.predicate == iri(vocab::RDF_SUBJECT));
        assert!(has_subj, "must have rdf:subject triple");
    }

    #[test]
    fn test_star_to_reification_annotation_preserved() {
        let mut star_graph = StarGraph::new();
        let inner = triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        let meta = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            iri("http://ex.org/source"),
            iri("http://ex.org/db"),
        );
        let _ = star_graph.insert(meta);

        let mut bridge = ReificationBridge::default_config();
        let reif = bridge.star_to_reification(&star_graph).expect("ok");

        let has_source = reif
            .triples()
            .iter()
            .any(|t| t.predicate == iri("http://ex.org/source"));
        assert!(has_source);
    }

    #[test]
    fn test_star_to_reification_multiple_meta() {
        let mut star_graph = StarGraph::new();
        let inner = triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        let _ = star_graph.insert(StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            iri("http://ex.org/cert"),
            lit("0.9"),
        ));
        let _ = star_graph.insert(StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            iri("http://ex.org/source"),
            iri("http://ex.org/db"),
        ));

        let mut bridge = ReificationBridge::default_config();
        let reif = bridge.star_to_reification(&star_graph).expect("ok");
        // Must have at least two annotation triples.
        assert!(reif.len() >= 2);
    }

    // ------------------------------------------------------------------
    // Round-trip
    // ------------------------------------------------------------------

    #[test]
    fn test_round_trip_star_to_reif_to_star() {
        let mut star_graph = StarGraph::new();
        let inner = triple(
            "http://ex.org/alice",
            "http://ex.org/knows",
            "http://ex.org/bob",
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            iri("http://ex.org/certainty"),
            lit("0.9"),
        );
        let _ = star_graph.insert(meta);

        // star → reification → star
        let mut bridge = ReificationBridge::default_config();
        let reif = bridge.star_to_reification(&star_graph).expect("star→reif");
        let bridge2 = ReificationBridge::default_config();
        let recovered = bridge2.reification_to_star(&reif).expect("reif→star");

        // Recovered graph should contain a triple with quoted subject.
        let has_quoted = recovered
            .triples()
            .iter()
            .any(|t| t.subject.is_quoted_triple());
        assert!(has_quoted, "round-trip must preserve quoted triple");
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    #[test]
    fn test_validate_complete_reification() {
        let graph = build_reification_graph(
            "http://ex.org/stmt1",
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
            None,
        );
        let bridge = ReificationBridge::default_config();
        assert!(bridge.validate_reification_graph(&graph).is_ok());
    }

    #[test]
    fn test_validate_incomplete_reification() {
        let mut g = StarGraph::new();
        // Only rdf:type and rdf:subject, missing predicate and object.
        let _ = g.insert(StarTriple::new(
            iri("http://ex.org/stmt"),
            iri(vocab::RDF_TYPE),
            iri(vocab::RDF_STATEMENT),
        ));
        let _ = g.insert(StarTriple::new(
            iri("http://ex.org/stmt"),
            iri(vocab::RDF_SUBJECT),
            iri("http://ex.org/alice"),
        ));

        let bridge = ReificationBridge::default_config();
        assert!(bridge.validate_reification_graph(&g).is_err());
    }

    // ------------------------------------------------------------------
    // Convenience functions
    // ------------------------------------------------------------------

    #[test]
    fn test_free_function_reification_to_star() {
        let graph = build_reification_graph(
            "http://ex.org/stmt1",
            "http://ex.org/s",
            "http://ex.org/p",
            "http://ex.org/o",
            None,
        );
        let star = reification_to_star(&graph).expect("ok");
        assert!(!star.is_empty());
    }

    #[test]
    fn test_free_function_star_to_reification() {
        let mut star = StarGraph::new();
        let inner = triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        let _ = star.insert(StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://ex.org/cert"),
            lit("0.9"),
        ));
        let reif = star_to_reification(&star).expect("ok");
        assert!(!reif.is_empty());
    }

    #[test]
    fn test_non_reification_triples_passed_through() {
        let mut graph = build_reification_graph(
            "http://ex.org/stmt1",
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
            None,
        );
        // Add an unrelated triple.
        let unrelated = triple("http://ex.org/x", "http://ex.org/y", "http://ex.org/z");
        let _ = graph.insert(unrelated.clone());

        let bridge = ReificationBridge::default_config();
        let star = bridge.reification_to_star(&graph).expect("ok");
        assert!(
            star.contains(&unrelated),
            "unrelated triple must pass through"
        );
    }
}
