//! # RDF-star Annotation Paths
//!
//! Traverses chains of annotations on quoted triples (meta-annotations).
//! For example, given:
//!
//! ```text
//! << :alice :age 30 >> :certainty 0.9 .
//! << << :alice :age 30 >> :certainty 0.9 >> :source :census .
//! ```
//!
//! An annotation path can walk from the base statement through each layer
//! of meta-annotation, collecting provenance, confidence, or other metadata
//! along the way.
//!
//! ## Features
//!
//! - Walk annotation chains of arbitrary depth
//! - Collect annotations matching a predicate filter
//! - Compute chain statistics (depth, breadth)
//! - Build annotation path expressions for SPARQL-star queries
//!
//! ## Usage
//!
//! ```rust
//! use oxirs_star::annotation_paths::{AnnotationPath, AnnotationStore, AnnotationEntry};
//! use oxirs_star::model::{StarTriple, StarTerm};
//!
//! let mut store = AnnotationStore::new();
//!
//! let base = StarTriple::new(
//!     StarTerm::iri("http://ex.org/alice").unwrap(),
//!     StarTerm::iri("http://ex.org/age").unwrap(),
//!     StarTerm::literal("30").unwrap(),
//! );
//! store.add_annotation(&base, "http://ex.org/certainty", "0.9");
//!
//! let path = AnnotationPath::new(&store);
//! let chain = path.walk(&base);
//! assert_eq!(chain.depth(), 1);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::model::{StarTerm, StarTriple};

// ---------------------------------------------------------------------------
// AnnotationEntry
// ---------------------------------------------------------------------------

/// A single annotation on a statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationEntry {
    /// The predicate IRI of the annotation.
    pub predicate: String,
    /// The value of the annotation.
    pub value: String,
    /// The annotated triple (serialised key form).
    pub annotated_triple_key: String,
}

// ---------------------------------------------------------------------------
// AnnotationStore
// ---------------------------------------------------------------------------

/// An in-memory store of annotations keyed by a canonical triple string.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnnotationStore {
    /// Map from canonical triple key -> list of annotations.
    annotations: HashMap<String, Vec<AnnotationEntry>>,
    /// Map from annotation key (the annotation itself as a triple) -> list of meta-annotations.
    meta_annotations: HashMap<String, Vec<AnnotationEntry>>,
}

impl AnnotationStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an annotation on a base triple.
    pub fn add_annotation(&mut self, triple: &StarTriple, predicate: &str, value: &str) {
        let key = triple_key(triple);
        let entry = AnnotationEntry {
            predicate: predicate.to_string(),
            value: value.to_string(),
            annotated_triple_key: key.clone(),
        };
        self.annotations.entry(key).or_default().push(entry);
    }

    /// Add a meta-annotation (annotation on an annotation).
    pub fn add_meta_annotation(
        &mut self,
        base_triple: &StarTriple,
        annotation_predicate: &str,
        meta_predicate: &str,
        meta_value: &str,
    ) {
        // The annotation itself acts as a "statement"
        let base_key = triple_key(base_triple);
        let annotation_key = format!("<< {base_key} >> {annotation_predicate}");
        let entry = AnnotationEntry {
            predicate: meta_predicate.to_string(),
            value: meta_value.to_string(),
            annotated_triple_key: annotation_key.clone(),
        };
        self.meta_annotations
            .entry(annotation_key)
            .or_default()
            .push(entry);
    }

    /// Get direct annotations on a triple.
    pub fn get_annotations(&self, triple: &StarTriple) -> Vec<&AnnotationEntry> {
        let key = triple_key(triple);
        self.annotations
            .get(&key)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get meta-annotations on an annotation.
    pub fn get_meta_annotations(
        &self,
        base_triple: &StarTriple,
        annotation_predicate: &str,
    ) -> Vec<&AnnotationEntry> {
        let base_key = triple_key(base_triple);
        let annotation_key = format!("<< {base_key} >> {annotation_predicate}");
        self.meta_annotations
            .get(&annotation_key)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Total number of annotations.
    pub fn total_annotations(&self) -> usize {
        self.annotations.values().map(|v| v.len()).sum::<usize>()
            + self
                .meta_annotations
                .values()
                .map(|v| v.len())
                .sum::<usize>()
    }

    /// Number of annotated triples.
    pub fn num_annotated_triples(&self) -> usize {
        self.annotations.len()
    }

    /// Check if a triple has any annotations.
    pub fn has_annotations(&self, triple: &StarTriple) -> bool {
        let key = triple_key(triple);
        self.annotations.contains_key(&key)
    }

    /// Filter annotations by predicate.
    pub fn annotations_by_predicate(
        &self,
        triple: &StarTriple,
        predicate: &str,
    ) -> Vec<&AnnotationEntry> {
        self.get_annotations(triple)
            .into_iter()
            .filter(|e| e.predicate == predicate)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// AnnotationChain
// ---------------------------------------------------------------------------

/// A chain of annotations from a base triple through meta-annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationChain {
    /// The base triple key.
    pub base_key: String,
    /// Levels of annotations. Level 0 = direct annotations, Level 1 = meta-annotations, etc.
    pub levels: Vec<Vec<AnnotationEntry>>,
}

impl AnnotationChain {
    /// Depth of the chain (number of annotation levels).
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Total number of annotation entries across all levels.
    pub fn total_entries(&self) -> usize {
        self.levels.iter().map(|l| l.len()).sum()
    }

    /// Whether the chain is empty (no annotations found).
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty() || self.levels.iter().all(|l| l.is_empty())
    }

    /// Get annotations at a specific level.
    pub fn at_level(&self, level: usize) -> &[AnnotationEntry] {
        self.levels.get(level).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Flatten all entries in order.
    pub fn flatten(&self) -> Vec<&AnnotationEntry> {
        self.levels.iter().flat_map(|l| l.iter()).collect()
    }
}

// ---------------------------------------------------------------------------
// AnnotationPath
// ---------------------------------------------------------------------------

/// Walk annotation chains on triples.
pub struct AnnotationPath<'a> {
    store: &'a AnnotationStore,
    max_depth: usize,
}

impl<'a> AnnotationPath<'a> {
    /// Create a new annotation path walker.
    pub fn new(store: &'a AnnotationStore) -> Self {
        Self {
            store,
            max_depth: 10,
        }
    }

    /// Set maximum traversal depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Walk the full annotation chain from a base triple.
    pub fn walk(&self, triple: &StarTriple) -> AnnotationChain {
        let base_key = triple_key(triple);
        let mut levels: Vec<Vec<AnnotationEntry>> = Vec::new();

        // Level 0: direct annotations
        let direct = self.store.get_annotations(triple);
        if direct.is_empty() {
            return AnnotationChain {
                base_key,
                levels: Vec::new(),
            };
        }
        let level0: Vec<AnnotationEntry> = direct.into_iter().cloned().collect();
        levels.push(level0);

        // Subsequent levels: meta-annotations
        for depth in 1..self.max_depth {
            let prev_level = &levels[depth - 1];
            let mut next_level = Vec::new();
            for entry in prev_level {
                let meta_key = format!("<< {} >> {}", entry.annotated_triple_key, entry.predicate);
                if let Some(metas) = self.store.meta_annotations.get(&meta_key) {
                    next_level.extend(metas.iter().cloned());
                }
            }
            if next_level.is_empty() {
                break;
            }
            levels.push(next_level);
        }

        AnnotationChain { base_key, levels }
    }

    /// Walk and filter by a specific predicate at all levels.
    pub fn walk_with_predicate(&self, triple: &StarTriple, predicate: &str) -> AnnotationChain {
        let full = self.walk(triple);
        let filtered_levels: Vec<Vec<AnnotationEntry>> = full
            .levels
            .into_iter()
            .map(|level| {
                level
                    .into_iter()
                    .filter(|e| e.predicate == predicate)
                    .collect::<Vec<_>>()
            })
            .filter(|l| !l.is_empty())
            .collect();
        AnnotationChain {
            base_key: full.base_key,
            levels: filtered_levels,
        }
    }

    /// Collect all values for a predicate at any depth.
    pub fn collect_values(&self, triple: &StarTriple, predicate: &str) -> Vec<String> {
        let chain = self.walk(triple);
        chain
            .flatten()
            .into_iter()
            .filter(|e| e.predicate == predicate)
            .map(|e| e.value.clone())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// AnnotationPathExpression
// ---------------------------------------------------------------------------

/// A path expression describing how to navigate annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationPathExpression {
    /// The predicates to follow at each step.
    pub steps: Vec<String>,
}

impl AnnotationPathExpression {
    /// Create a single-step path.
    pub fn single(predicate: impl Into<String>) -> Self {
        Self {
            steps: vec![predicate.into()],
        }
    }

    /// Create a multi-step path.
    pub fn multi(predicates: Vec<String>) -> Self {
        Self { steps: predicates }
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether this is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Generate a SPARQL-star-like path expression string.
    pub fn to_sparql_star_pattern(&self) -> String {
        if self.steps.is_empty() {
            return String::new();
        }
        let mut parts = Vec::new();
        for (i, step) in self.steps.iter().enumerate() {
            if i == 0 {
                parts.push(format!("<< ?s ?p ?o >> <{step}> ?ann0"));
            } else {
                parts.push(format!("<< ?ann{} >> <{step}> ?ann{}", i - 1, i));
            }
        }
        parts.join(" . ")
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn triple_key(triple: &StarTriple) -> String {
    format!(
        "{} {} {}",
        term_key(&triple.subject),
        term_key(&triple.predicate),
        term_key(&triple.object)
    )
}

fn term_key(term: &StarTerm) -> String {
    match term {
        StarTerm::NamedNode(nn) => format!("<{}>", nn.iri),
        StarTerm::BlankNode(bn) => format!("_:{}", bn.id),
        StarTerm::Literal(lit) => {
            let mut s = format!("\"{}\"", lit.value);
            if let Some(ref lang) = lit.language {
                s.push_str(&format!("@{lang}"));
            }
            if let Some(ref dt) = lit.datatype {
                s.push_str(&format!("^^<{}>", dt.iri));
            }
            s
        }
        StarTerm::QuotedTriple(qt) => {
            format!("<< {} >>", triple_key(qt))
        }
        StarTerm::Variable(v) => format!("?{}", v.name),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn base_triple() -> StarTriple {
        StarTriple::new(
            StarTerm::iri("http://ex.org/alice").expect("iri"),
            StarTerm::iri("http://ex.org/age").expect("iri"),
            StarTerm::literal("30").expect("literal"),
        )
    }

    fn base_triple2() -> StarTriple {
        StarTriple::new(
            StarTerm::iri("http://ex.org/bob").expect("iri"),
            StarTerm::iri("http://ex.org/age").expect("iri"),
            StarTerm::literal("25").expect("literal"),
        )
    }

    // -- AnnotationStore --

    #[test]
    fn test_empty_store() {
        let store = AnnotationStore::new();
        assert_eq!(store.total_annotations(), 0);
        assert_eq!(store.num_annotated_triples(), 0);
    }

    #[test]
    fn test_add_annotation() {
        let mut store = AnnotationStore::new();
        let triple = base_triple();
        store.add_annotation(&triple, "http://ex.org/certainty", "0.9");
        assert_eq!(store.total_annotations(), 1);
        assert!(store.has_annotations(&triple));
    }

    #[test]
    fn test_get_annotations() {
        let mut store = AnnotationStore::new();
        let triple = base_triple();
        store.add_annotation(&triple, "http://ex.org/certainty", "0.9");
        store.add_annotation(&triple, "http://ex.org/source", "census");
        let anns = store.get_annotations(&triple);
        assert_eq!(anns.len(), 2);
    }

    #[test]
    fn test_no_annotations_for_unknown_triple() {
        let store = AnnotationStore::new();
        let triple = base_triple();
        assert!(!store.has_annotations(&triple));
        assert!(store.get_annotations(&triple).is_empty());
    }

    #[test]
    fn test_annotations_by_predicate() {
        let mut store = AnnotationStore::new();
        let triple = base_triple();
        store.add_annotation(&triple, "http://ex.org/certainty", "0.9");
        store.add_annotation(&triple, "http://ex.org/source", "census");
        let filtered = store.annotations_by_predicate(&triple, "http://ex.org/certainty");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].value, "0.9");
    }

    #[test]
    fn test_add_meta_annotation() {
        let mut store = AnnotationStore::new();
        let triple = base_triple();
        store.add_annotation(&triple, "http://ex.org/certainty", "0.9");
        store.add_meta_annotation(
            &triple,
            "http://ex.org/certainty",
            "http://ex.org/method",
            "statistical",
        );
        let metas = store.get_meta_annotations(&triple, "http://ex.org/certainty");
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].predicate, "http://ex.org/method");
    }

    #[test]
    fn test_num_annotated_triples() {
        let mut store = AnnotationStore::new();
        store.add_annotation(&base_triple(), "http://ex.org/c", "0.9");
        store.add_annotation(&base_triple2(), "http://ex.org/c", "0.8");
        assert_eq!(store.num_annotated_triples(), 2);
    }

    // -- AnnotationPath walk --

    #[test]
    fn test_walk_empty_store() {
        let store = AnnotationStore::new();
        let path = AnnotationPath::new(&store);
        let chain = path.walk(&base_triple());
        assert!(chain.is_empty());
        assert_eq!(chain.depth(), 0);
    }

    #[test]
    fn test_walk_single_level() {
        let mut store = AnnotationStore::new();
        let triple = base_triple();
        store.add_annotation(&triple, "http://ex.org/certainty", "0.9");
        let path = AnnotationPath::new(&store);
        let chain = path.walk(&triple);
        assert_eq!(chain.depth(), 1);
        assert_eq!(chain.total_entries(), 1);
    }

    #[test]
    fn test_walk_two_levels() {
        let mut store = AnnotationStore::new();
        let triple = base_triple();
        store.add_annotation(&triple, "http://ex.org/certainty", "0.9");
        store.add_meta_annotation(
            &triple,
            "http://ex.org/certainty",
            "http://ex.org/source",
            "census",
        );
        let path = AnnotationPath::new(&store);
        let chain = path.walk(&triple);
        assert_eq!(chain.depth(), 2);
        assert_eq!(chain.at_level(0).len(), 1);
        assert_eq!(chain.at_level(1).len(), 1);
    }

    #[test]
    fn test_walk_with_predicate_filter() {
        let mut store = AnnotationStore::new();
        let triple = base_triple();
        store.add_annotation(&triple, "http://ex.org/certainty", "0.9");
        store.add_annotation(&triple, "http://ex.org/source", "census");
        let path = AnnotationPath::new(&store);
        let chain = path.walk_with_predicate(&triple, "http://ex.org/certainty");
        assert_eq!(chain.total_entries(), 1);
    }

    #[test]
    fn test_collect_values() {
        let mut store = AnnotationStore::new();
        let triple = base_triple();
        store.add_annotation(&triple, "http://ex.org/certainty", "0.9");
        store.add_annotation(&triple, "http://ex.org/certainty", "0.8");
        store.add_annotation(&triple, "http://ex.org/source", "census");
        let path = AnnotationPath::new(&store);
        let vals = path.collect_values(&triple, "http://ex.org/certainty");
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&"0.9".to_string()));
        assert!(vals.contains(&"0.8".to_string()));
    }

    #[test]
    fn test_max_depth_limit() {
        let store = AnnotationStore::new();
        let path = AnnotationPath::new(&store).with_max_depth(1);
        assert_eq!(path.max_depth, 1);
    }

    // -- AnnotationChain --

    #[test]
    fn test_chain_flatten() {
        let mut store = AnnotationStore::new();
        let triple = base_triple();
        store.add_annotation(&triple, "http://ex.org/c1", "v1");
        store.add_annotation(&triple, "http://ex.org/c2", "v2");
        let path = AnnotationPath::new(&store);
        let chain = path.walk(&triple);
        assert_eq!(chain.flatten().len(), 2);
    }

    #[test]
    fn test_chain_at_level_out_of_bounds() {
        let chain = AnnotationChain {
            base_key: "test".to_string(),
            levels: Vec::new(),
        };
        assert!(chain.at_level(5).is_empty());
    }

    // -- AnnotationPathExpression --

    #[test]
    fn test_path_expression_single() {
        let expr = AnnotationPathExpression::single("http://ex.org/certainty");
        assert_eq!(expr.len(), 1);
        assert!(!expr.is_empty());
    }

    #[test]
    fn test_path_expression_multi() {
        let expr = AnnotationPathExpression::multi(vec![
            "http://ex.org/certainty".to_string(),
            "http://ex.org/source".to_string(),
        ]);
        assert_eq!(expr.len(), 2);
    }

    #[test]
    fn test_path_expression_empty() {
        let expr = AnnotationPathExpression::multi(Vec::new());
        assert!(expr.is_empty());
        assert_eq!(expr.to_sparql_star_pattern(), "");
    }

    #[test]
    fn test_path_expression_sparql_single() {
        let expr = AnnotationPathExpression::single("http://ex.org/certainty");
        let pattern = expr.to_sparql_star_pattern();
        assert!(pattern.contains("<< ?s ?p ?o >>"));
        assert!(pattern.contains("<http://ex.org/certainty>"));
        assert!(pattern.contains("?ann0"));
    }

    #[test]
    fn test_path_expression_sparql_multi() {
        let expr = AnnotationPathExpression::multi(vec![
            "http://ex.org/certainty".to_string(),
            "http://ex.org/source".to_string(),
        ]);
        let pattern = expr.to_sparql_star_pattern();
        assert!(pattern.contains("?ann0"));
        assert!(pattern.contains("?ann1"));
    }

    // -- triple_key / term_key --

    #[test]
    fn test_triple_key_iri() {
        let triple = base_triple();
        let key = triple_key(&triple);
        assert!(key.contains("<http://ex.org/alice>"));
        assert!(key.contains("<http://ex.org/age>"));
    }

    #[test]
    fn test_term_key_literal() {
        let term = StarTerm::literal("hello").expect("literal");
        let key = term_key(&term);
        assert!(key.contains("hello"));
    }

    #[test]
    fn test_term_key_typed_literal() {
        let term =
            StarTerm::literal_with_datatype("42", "http://www.w3.org/2001/XMLSchema#integer")
                .expect("typed lit");
        let key = term_key(&term);
        assert!(key.contains("42"));
        assert!(key.contains("XMLSchema#integer"));
    }

    #[test]
    fn test_term_key_lang_literal() {
        let term = StarTerm::literal_with_language("hello", "en").expect("lang lit");
        let key = term_key(&term);
        assert!(key.contains("@en"));
    }

    #[test]
    fn test_term_key_blank_node() {
        let term = StarTerm::blank_node("b0").expect("bnode");
        let key = term_key(&term);
        assert_eq!(key, "_:b0");
    }

    #[test]
    fn test_term_key_quoted_triple() {
        let inner = StarTriple::new(
            StarTerm::iri("http://ex.org/s").expect("iri"),
            StarTerm::iri("http://ex.org/p").expect("iri"),
            StarTerm::iri("http://ex.org/o").expect("iri"),
        );
        let term = StarTerm::QuotedTriple(Box::new(inner));
        let key = term_key(&term);
        assert!(key.contains("<<"));
        assert!(key.contains(">>"));
    }

    // -- multiple triples with different annotations --

    #[test]
    fn test_separate_triple_annotations() {
        let mut store = AnnotationStore::new();
        store.add_annotation(&base_triple(), "http://ex.org/c", "0.9");
        store.add_annotation(&base_triple2(), "http://ex.org/c", "0.8");
        let path = AnnotationPath::new(&store);

        let chain1 = path.walk(&base_triple());
        let chain2 = path.walk(&base_triple2());
        assert_eq!(chain1.total_entries(), 1);
        assert_eq!(chain2.total_entries(), 1);
        assert_eq!(chain1.at_level(0)[0].value, "0.9");
        assert_eq!(chain2.at_level(0)[0].value, "0.8");
    }
}
