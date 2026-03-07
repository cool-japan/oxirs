//! # RDF-star Graph Merger
//!
//! Merges named graphs containing RDF-star triples (including quoted triples of the form
//! `<<subject predicate object>>`). Supports union, intersection, and difference strategies
//! as well as conflict resolution policies for subject/predicate pairs with differing objects.

use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────────────
// Triple
// ─────────────────────────────────────────────────────────────────────────────

/// An RDF (or RDF-star) triple. Subjects and objects may be quoted triples of
/// the form `<<s p o>>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    /// Subject IRI, blank node, or quoted triple `<<s p o>>`.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object IRI, literal, blank node, or quoted triple `<<s p o>>`.
    pub object: String,
}

impl Triple {
    /// Create a new `Triple`.
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

    /// Return `true` if the subject is a quoted triple (`<<…>>`).
    pub fn subject_is_quoted(&self) -> bool {
        self.subject.starts_with("<<") && self.subject.ends_with(">>")
    }

    /// Return `true` if the object is a quoted triple (`<<…>>`).
    pub fn object_is_quoted(&self) -> bool {
        self.object.starts_with("<<") && self.object.ends_with(">>")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NamedGraph
// ─────────────────────────────────────────────────────────────────────────────

/// A named RDF graph containing a deduplicated set of triples.
#[derive(Debug, Clone)]
pub struct NamedGraph {
    /// Graph name (IRI or blank node identifier).
    pub name: String,
    /// The triples contained in this graph.
    pub triples: Vec<Triple>,
    /// Internal dedup set — keys are (subject, predicate, object).
    seen: HashSet<(String, String, String)>,
}

impl NamedGraph {
    /// Create an empty `NamedGraph` with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            triples: Vec::new(),
            seen: HashSet::new(),
        }
    }

    /// Insert a triple, returning `true` if it was new (not a duplicate).
    pub fn insert(&mut self, t: Triple) -> bool {
        let key = (t.subject.clone(), t.predicate.clone(), t.object.clone());
        if self.seen.insert(key) {
            self.triples.push(t);
            true
        } else {
            false
        }
    }

    /// Return the number of triples in this graph.
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Return `true` if the graph contains no triples.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Test whether a specific triple is present.
    pub fn contains(&self, t: &Triple) -> bool {
        let key = (t.subject.clone(), t.predicate.clone(), t.object.clone());
        self.seen.contains(&key)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MergeStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy used when merging two graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Include all triples from both graphs (set union).
    Union,
    /// Include only triples present in both graphs (set intersection).
    Intersection,
    /// Include only triples present in graph A but not in graph B (set difference A \ B).
    Difference,
}

// ─────────────────────────────────────────────────────────────────────────────
// ConflictResolution
// ─────────────────────────────────────────────────────────────────────────────

/// How to resolve conflicts where two triples share the same subject and predicate
/// but have different objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictResolution {
    /// Keep the triple from graph A (first encountered).
    KeepFirst,
    /// Keep the triple from graph B (last encountered).
    KeepLast,
    /// Keep both triples (resulting graph may have multiple objects per s+p pair).
    KeepBoth,
    /// Skip both conflicting triples.
    Skip,
}

// ─────────────────────────────────────────────────────────────────────────────
// MergeResult
// ─────────────────────────────────────────────────────────────────────────────

/// The result of a merge operation.
#[derive(Debug)]
pub struct MergeResult {
    /// The merged graph.
    pub graph: NamedGraph,
    /// Number of subject+predicate conflicts encountered.
    pub conflicts: usize,
    /// Number of triples added to the result.
    pub added: usize,
    /// Number of triples skipped (due to conflicts or strategy).
    pub skipped: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphMerger
// ─────────────────────────────────────────────────────────────────────────────

/// Merges `NamedGraph` instances using configurable strategies.
#[derive(Debug, Default)]
pub struct GraphMerger;

impl GraphMerger {
    /// Create a new `GraphMerger`.
    pub fn new() -> Self {
        Self
    }

    /// Merge graphs `a` and `b` using the given strategy.
    ///
    /// The resulting graph is named `"merged"`.
    pub fn merge(&self, a: &NamedGraph, b: &NamedGraph, strategy: MergeStrategy) -> MergeResult {
        let mut result = NamedGraph::new("merged");
        let mut added = 0usize;
        let mut skipped = 0usize;

        match strategy {
            MergeStrategy::Union => {
                // All triples from A
                for t in &a.triples {
                    if result.insert(t.clone()) {
                        added += 1;
                    }
                }
                // All triples from B (duplicates are deduped by NamedGraph::insert)
                for t in &b.triples {
                    if result.insert(t.clone()) {
                        added += 1;
                    } else {
                        skipped += 1;
                    }
                }
            }

            MergeStrategy::Intersection => {
                // Only triples present in both.
                for t in &a.triples {
                    if b.contains(t) {
                        if result.insert(t.clone()) {
                            added += 1;
                        }
                    } else {
                        skipped += 1;
                    }
                }
            }

            MergeStrategy::Difference => {
                // Triples in A that are NOT in B.
                for t in &a.triples {
                    if !b.contains(t) {
                        if result.insert(t.clone()) {
                            added += 1;
                        }
                    } else {
                        skipped += 1;
                    }
                }
            }
        }

        MergeResult {
            graph: result,
            conflicts: 0,
            added,
            skipped,
        }
    }

    /// Merge multiple graphs using the given strategy.
    ///
    /// Graphs are merged left-to-right (fold). Returns an empty result if `graphs` is empty.
    pub fn merge_many(&self, graphs: &[NamedGraph], strategy: MergeStrategy) -> MergeResult {
        if graphs.is_empty() {
            return MergeResult {
                graph: NamedGraph::new("merged"),
                conflicts: 0,
                added: 0,
                skipped: 0,
            };
        }

        if graphs.len() == 1 {
            let mut result = NamedGraph::new("merged");
            let mut added = 0;
            for t in &graphs[0].triples {
                if result.insert(t.clone()) {
                    added += 1;
                }
            }
            return MergeResult {
                graph: result,
                conflicts: 0,
                added,
                skipped: 0,
            };
        }

        // Accumulate: merge the first result graph with each subsequent graph.
        let first = self.merge(&graphs[0], &graphs[1], strategy.clone());
        let mut acc = first;

        for next in &graphs[2..] {
            let merged = self.merge(&acc.graph, next, strategy.clone());
            acc = MergeResult {
                graph: merged.graph,
                conflicts: acc.conflicts + merged.conflicts,
                added: acc.added + merged.added,
                skipped: acc.skipped + merged.skipped,
            };
        }

        acc
    }

    /// Return the list of "conflicting" triples: those that share the same subject
    /// and predicate in `a` and `b` but have different objects.
    pub fn find_conflicts(&self, a: &NamedGraph, b: &NamedGraph) -> Vec<Triple> {
        // Build a (subject, predicate) → object map for B.
        let mut b_map: std::collections::HashMap<(&str, &str), HashSet<&str>> =
            std::collections::HashMap::new();
        for t in &b.triples {
            b_map
                .entry((t.subject.as_str(), t.predicate.as_str()))
                .or_default()
                .insert(t.object.as_str());
        }

        let mut conflicts = Vec::new();
        for t in &a.triples {
            if let Some(b_objects) = b_map.get(&(t.subject.as_str(), t.predicate.as_str())) {
                // Conflict if B has the same s+p but a different object.
                if !b_objects.contains(t.object.as_str()) {
                    conflicts.push(t.clone());
                }
            }
        }

        conflicts
    }

    /// Resolve conflicts between graphs A and B using the specified resolution strategy.
    ///
    /// Non-conflicting triples are always included. For conflicting triples (same s+p,
    /// different o) the `ConflictResolution` policy is applied.
    pub fn resolve(
        &self,
        a: &NamedGraph,
        b: &NamedGraph,
        resolution: ConflictResolution,
    ) -> NamedGraph {
        let mut result = NamedGraph::new("resolved");

        // Build s+p → Vec<object> map for both graphs.
        let mut a_sp: std::collections::HashMap<(&str, &str), Vec<&Triple>> =
            std::collections::HashMap::new();
        for t in &a.triples {
            a_sp.entry((t.subject.as_str(), t.predicate.as_str()))
                .or_default()
                .push(t);
        }

        let mut b_sp: std::collections::HashMap<(&str, &str), Vec<&Triple>> =
            std::collections::HashMap::new();
        for t in &b.triples {
            b_sp.entry((t.subject.as_str(), t.predicate.as_str()))
                .or_default()
                .push(t);
        }

        // Collect all unique (s, p) pairs.
        let mut sp_pairs: HashSet<(&str, &str)> = HashSet::new();
        sp_pairs.extend(a_sp.keys());
        sp_pairs.extend(b_sp.keys());

        for sp in sp_pairs {
            let a_triples = a_sp.get(&sp).map(|v| v.as_slice()).unwrap_or(&[]);
            let b_triples = b_sp.get(&sp).map(|v| v.as_slice()).unwrap_or(&[]);

            // If only one side has this s+p, no conflict.
            if a_triples.is_empty() {
                for t in b_triples {
                    result.insert((*t).clone());
                }
                continue;
            }
            if b_triples.is_empty() {
                for t in a_triples {
                    result.insert((*t).clone());
                }
                continue;
            }

            // Check if there is an actual conflict (differing objects).
            let a_objects: HashSet<&str> = a_triples.iter().map(|t| t.object.as_str()).collect();
            let b_objects: HashSet<&str> = b_triples.iter().map(|t| t.object.as_str()).collect();
            let conflict = a_objects != b_objects;

            if !conflict {
                // No conflict: include all (deduped by NamedGraph::insert).
                for t in a_triples {
                    result.insert((*t).clone());
                }
            } else {
                match resolution {
                    ConflictResolution::KeepFirst => {
                        for t in a_triples {
                            result.insert((*t).clone());
                        }
                    }
                    ConflictResolution::KeepLast => {
                        for t in b_triples {
                            result.insert((*t).clone());
                        }
                    }
                    ConflictResolution::KeepBoth => {
                        for t in a_triples {
                            result.insert((*t).clone());
                        }
                        for t in b_triples {
                            result.insert((*t).clone());
                        }
                    }
                    ConflictResolution::Skip => {
                        // Neither side is included.
                    }
                }
            }
        }

        result
    }

    /// Return triples that are present in both `a` and `b` (set intersection).
    pub fn common_triples(&self, a: &NamedGraph, b: &NamedGraph) -> Vec<Triple> {
        a.triples
            .iter()
            .filter(|t| b.contains(t))
            .cloned()
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    fn graph(name: &str, triples: &[Triple]) -> NamedGraph {
        let mut g = NamedGraph::new(name);
        for triple in triples {
            g.insert(triple.clone());
        }
        g
    }

    // ── Triple ────────────────────────────────────────────────────────────────

    #[test]
    fn test_triple_new() {
        let tr = t("s", "p", "o");
        assert_eq!(tr.subject, "s");
        assert_eq!(tr.predicate, "p");
        assert_eq!(tr.object, "o");
    }

    #[test]
    fn test_triple_quoted_subject() {
        let tr = t("<<s p o>>", "cert", "0.9");
        assert!(tr.subject_is_quoted());
        assert!(!tr.object_is_quoted());
    }

    #[test]
    fn test_triple_quoted_object() {
        let tr = t("source", "cites", "<<a b c>>");
        assert!(!tr.subject_is_quoted());
        assert!(tr.object_is_quoted());
    }

    #[test]
    fn test_triple_not_quoted() {
        let tr = t("ex:s", "ex:p", "ex:o");
        assert!(!tr.subject_is_quoted());
        assert!(!tr.object_is_quoted());
    }

    // ── NamedGraph ────────────────────────────────────────────────────────────

    #[test]
    fn test_named_graph_new() {
        let g = NamedGraph::new("http://example.org/g1");
        assert_eq!(g.name, "http://example.org/g1");
        assert_eq!(g.triple_count(), 0);
        assert!(g.is_empty());
    }

    #[test]
    fn test_named_graph_insert_new() {
        let mut g = NamedGraph::new("g");
        assert!(g.insert(t("s", "p", "o")));
        assert_eq!(g.triple_count(), 1);
    }

    #[test]
    fn test_named_graph_insert_duplicate() {
        let mut g = NamedGraph::new("g");
        assert!(g.insert(t("s", "p", "o")));
        assert!(!g.insert(t("s", "p", "o")));
        assert_eq!(g.triple_count(), 1);
    }

    #[test]
    fn test_named_graph_contains() {
        let mut g = NamedGraph::new("g");
        let tr = t("s", "p", "o");
        g.insert(tr.clone());
        assert!(g.contains(&tr));
        assert!(!g.contains(&t("s", "p", "other")));
    }

    // ── Union ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_union_disjoint() {
        let a = graph("a", &[t("s1", "p", "o1")]);
        let b = graph("b", &[t("s2", "p", "o2")]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 2);
    }

    #[test]
    fn test_union_identical() {
        let tr = t("s", "p", "o");
        let a = graph("a", std::slice::from_ref(&tr));
        let b = graph("b", std::slice::from_ref(&tr));
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 1);
    }

    #[test]
    fn test_union_partial_overlap() {
        let shared = t("s", "p", "o");
        let a = graph("a", &[shared.clone(), t("s1", "p", "o1")]);
        let b = graph("b", &[shared.clone(), t("s2", "p", "o2")]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 3);
    }

    #[test]
    fn test_union_with_empty_b() {
        let a = graph("a", &[t("s", "p", "o")]);
        let b = graph("b", &[]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 1);
    }

    #[test]
    fn test_union_both_empty() {
        let a = graph("a", &[]);
        let b = graph("b", &[]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 0);
    }

    // ── Intersection ──────────────────────────────────────────────────────────

    #[test]
    fn test_intersection_shared() {
        let shared = t("s", "p", "o");
        let a = graph("a", &[shared.clone(), t("unique_a", "p", "o")]);
        let b = graph("b", &[shared.clone(), t("unique_b", "p", "o")]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Intersection);
        assert_eq!(result.graph.triple_count(), 1);
        assert!(result.graph.contains(&shared));
    }

    #[test]
    fn test_intersection_disjoint() {
        let a = graph("a", &[t("s1", "p", "o1")]);
        let b = graph("b", &[t("s2", "p", "o2")]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Intersection);
        assert_eq!(result.graph.triple_count(), 0);
    }

    #[test]
    fn test_intersection_identical() {
        let tr = t("s", "p", "o");
        let a = graph("a", std::slice::from_ref(&tr));
        let b = graph("b", std::slice::from_ref(&tr));
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Intersection);
        assert_eq!(result.graph.triple_count(), 1);
    }

    #[test]
    fn test_intersection_empty_a() {
        let a = graph("a", &[]);
        let b = graph("b", &[t("s", "p", "o")]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Intersection);
        assert_eq!(result.graph.triple_count(), 0);
    }

    // ── Difference ────────────────────────────────────────────────────────────

    #[test]
    fn test_difference_disjoint() {
        let a = graph("a", &[t("s1", "p", "o1")]);
        let b = graph("b", &[t("s2", "p", "o2")]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Difference);
        assert_eq!(result.graph.triple_count(), 1);
        assert!(result.graph.contains(&t("s1", "p", "o1")));
    }

    #[test]
    fn test_difference_identical() {
        let tr = t("s", "p", "o");
        let a = graph("a", std::slice::from_ref(&tr));
        let b = graph("b", std::slice::from_ref(&tr));
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Difference);
        assert_eq!(result.graph.triple_count(), 0);
    }

    #[test]
    fn test_difference_partial_overlap() {
        let shared = t("s", "p", "shared");
        let only_a = t("s", "p", "only_a");
        let a = graph("a", &[shared.clone(), only_a.clone()]);
        let b = graph("b", &[shared.clone(), t("s2", "p", "b_only")]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Difference);
        assert_eq!(result.graph.triple_count(), 1);
        assert!(result.graph.contains(&only_a));
    }

    // ── merge_many ────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_many_empty_slice() {
        let merger = GraphMerger::new();
        let result = merger.merge_many(&[], MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 0);
    }

    #[test]
    fn test_merge_many_single() {
        let g = graph("g", &[t("s", "p", "o")]);
        let merger = GraphMerger::new();
        let result = merger.merge_many(&[g], MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 1);
    }

    #[test]
    fn test_merge_many_three_union() {
        let g1 = graph("g1", &[t("s1", "p", "o1")]);
        let g2 = graph("g2", &[t("s2", "p", "o2")]);
        let g3 = graph("g3", &[t("s3", "p", "o3")]);
        let merger = GraphMerger::new();
        let result = merger.merge_many(&[g1, g2, g3], MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 3);
    }

    #[test]
    fn test_merge_many_three_intersection() {
        let shared = t("s", "p", "shared");
        let g1 = graph("g1", &[shared.clone(), t("only1", "p", "o")]);
        let g2 = graph("g2", &[shared.clone(), t("only2", "p", "o")]);
        let g3 = graph("g3", &[shared.clone(), t("only3", "p", "o")]);
        let merger = GraphMerger::new();
        let result = merger.merge_many(&[g1, g2, g3], MergeStrategy::Intersection);
        assert_eq!(result.graph.triple_count(), 1);
        assert!(result.graph.contains(&shared));
    }

    // ── find_conflicts ────────────────────────────────────────────────────────

    #[test]
    fn test_find_conflicts_none() {
        let a = graph("a", &[t("s", "p", "o")]);
        let b = graph("b", &[t("s", "p", "o")]);
        let merger = GraphMerger::new();
        let conflicts = merger.find_conflicts(&a, &b);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_find_conflicts_detected() {
        let a = graph("a", &[t("s", "p", "o_a")]);
        let b = graph("b", &[t("s", "p", "o_b")]);
        let merger = GraphMerger::new();
        let conflicts = merger.find_conflicts(&a, &b);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].subject, "s");
        assert_eq!(conflicts[0].predicate, "p");
        assert_eq!(conflicts[0].object, "o_a");
    }

    #[test]
    fn test_find_conflicts_disjoint_sp() {
        let a = graph("a", &[t("s1", "p1", "o1")]);
        let b = graph("b", &[t("s2", "p2", "o2")]);
        let merger = GraphMerger::new();
        let conflicts = merger.find_conflicts(&a, &b);
        assert!(conflicts.is_empty());
    }

    // ── resolve ───────────────────────────────────────────────────────────────

    #[test]
    fn test_resolve_keep_first() {
        let a = graph("a", &[t("s", "p", "from_a")]);
        let b = graph("b", &[t("s", "p", "from_b")]);
        let merger = GraphMerger::new();
        let resolved = merger.resolve(&a, &b, ConflictResolution::KeepFirst);
        assert_eq!(resolved.triple_count(), 1);
        assert!(resolved.contains(&t("s", "p", "from_a")));
    }

    #[test]
    fn test_resolve_keep_last() {
        let a = graph("a", &[t("s", "p", "from_a")]);
        let b = graph("b", &[t("s", "p", "from_b")]);
        let merger = GraphMerger::new();
        let resolved = merger.resolve(&a, &b, ConflictResolution::KeepLast);
        assert_eq!(resolved.triple_count(), 1);
        assert!(resolved.contains(&t("s", "p", "from_b")));
    }

    #[test]
    fn test_resolve_keep_both() {
        let a = graph("a", &[t("s", "p", "from_a")]);
        let b = graph("b", &[t("s", "p", "from_b")]);
        let merger = GraphMerger::new();
        let resolved = merger.resolve(&a, &b, ConflictResolution::KeepBoth);
        assert_eq!(resolved.triple_count(), 2);
    }

    #[test]
    fn test_resolve_skip() {
        let a = graph("a", &[t("s", "p", "from_a")]);
        let b = graph("b", &[t("s", "p", "from_b")]);
        let merger = GraphMerger::new();
        let resolved = merger.resolve(&a, &b, ConflictResolution::Skip);
        assert_eq!(resolved.triple_count(), 0);
    }

    #[test]
    fn test_resolve_non_conflicting_always_included() {
        let a = graph("a", &[t("s", "p", "same"), t("only_a", "p", "v")]);
        let b = graph("b", &[t("s", "p", "same"), t("only_b", "p", "v")]);
        let merger = GraphMerger::new();
        let resolved = merger.resolve(&a, &b, ConflictResolution::Skip);
        // No conflict for s+p="same", so that triple is kept.
        // "only_a" and "only_b" have no counterpart in the other graph, so no conflict.
        assert!(resolved.contains(&t("s", "p", "same")));
    }

    // ── common_triples ────────────────────────────────────────────────────────

    #[test]
    fn test_common_triples_none() {
        let a = graph("a", &[t("s1", "p", "o1")]);
        let b = graph("b", &[t("s2", "p", "o2")]);
        let merger = GraphMerger::new();
        let common = merger.common_triples(&a, &b);
        assert!(common.is_empty());
    }

    #[test]
    fn test_common_triples_some() {
        let shared = t("s", "p", "o");
        let a = graph("a", &[shared.clone(), t("unique", "p", "u")]);
        let b = graph("b", std::slice::from_ref(&shared));
        let merger = GraphMerger::new();
        let common = merger.common_triples(&a, &b);
        assert_eq!(common.len(), 1);
        assert_eq!(common[0], shared);
    }

    #[test]
    fn test_common_triples_all() {
        let trs = vec![t("s1", "p", "o1"), t("s2", "p", "o2")];
        let a = graph("a", &trs);
        let b = graph("b", &trs);
        let merger = GraphMerger::new();
        let common = merger.common_triples(&a, &b);
        assert_eq!(common.len(), 2);
    }

    // ── Quoted triples ────────────────────────────────────────────────────────

    #[test]
    fn test_union_with_quoted_subjects() {
        let quoted_s = "<<ex:s ex:p ex:o>>";
        let a = graph("a", &[t(quoted_s, "ex:certainty", "0.9")]);
        let b = graph("b", &[t(quoted_s, "ex:certainty", "0.9")]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 1);
    }

    #[test]
    fn test_union_with_quoted_objects() {
        let a = graph("a", &[t("ex:claim", "ex:refers", "<<a b c>>")]);
        let b = graph("b", &[t("ex:fact", "ex:refers", "<<x y z>>")]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 2);
    }

    #[test]
    fn test_find_conflicts_with_quoted_triples() {
        let quoted = "<<ex:s ex:age 30>>";
        let a = graph("a", &[t(quoted, "ex:source", "db_a")]);
        let b = graph("b", &[t(quoted, "ex:source", "db_b")]);
        let merger = GraphMerger::new();
        let conflicts = merger.find_conflicts(&a, &b);
        assert_eq!(conflicts.len(), 1);
    }

    // ── Empty graph edge cases ─────────────────────────────────────────────────

    #[test]
    fn test_merge_empty_a_union() {
        let a = graph("a", &[]);
        let b = graph("b", &[t("s", "p", "o")]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 1);
    }

    #[test]
    fn test_merge_empty_b_union() {
        let a = graph("a", &[t("s", "p", "o")]);
        let b = graph("b", &[]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Union);
        assert_eq!(result.graph.triple_count(), 1);
    }

    #[test]
    fn test_merge_empty_both_intersection() {
        let a = graph("a", &[]);
        let b = graph("b", &[]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Intersection);
        assert_eq!(result.graph.triple_count(), 0);
    }

    #[test]
    fn test_merge_empty_both_difference() {
        let a = graph("a", &[]);
        let b = graph("b", &[]);
        let merger = GraphMerger::new();
        let result = merger.merge(&a, &b, MergeStrategy::Difference);
        assert_eq!(result.graph.triple_count(), 0);
    }

    #[test]
    fn test_resolve_both_empty() {
        let a = graph("a", &[]);
        let b = graph("b", &[]);
        let merger = GraphMerger::new();
        let resolved = merger.resolve(&a, &b, ConflictResolution::KeepFirst);
        assert_eq!(resolved.triple_count(), 0);
    }

    #[test]
    fn test_common_triples_empty_graphs() {
        let a = graph("a", &[]);
        let b = graph("b", &[]);
        let merger = GraphMerger::new();
        let common = merger.common_triples(&a, &b);
        assert!(common.is_empty());
    }
}
