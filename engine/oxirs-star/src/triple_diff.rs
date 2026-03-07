//! RDF / RDF-star graph diff utilities.
//!
//! Provides:
//! * Added / removed triple detection
//! * RDF Patch format serialisation (`A`/`D` lines)
//! * Patch application to a base graph
//! * Symmetric difference
//! * Diff statistics
//! * Blank-node-aware isomorphic diff (simple bijection search)

use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Triple representation
// ---------------------------------------------------------------------------

/// A single RDF triple (or quoted triple) represented as three string terms.
///
/// Terms are serialised as:
/// * IRI: `<http://example.org/foo>`
/// * Blank node: `_:b0`
/// * Literal: `"hello"` or `"42"^^<xsd:integer>`
/// * Quoted triple: `<< s p o >>`
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
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

    /// Return `true` when the triple contains at least one blank-node term.
    pub fn has_blank_node(&self) -> bool {
        is_blank(&self.subject) || is_blank(&self.predicate) || is_blank(&self.object)
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {} .", self.subject, self.predicate, self.object)
    }
}

fn is_blank(term: &str) -> bool {
    term.starts_with("_:")
}

// ---------------------------------------------------------------------------
// RDF Graph
// ---------------------------------------------------------------------------

/// An in-memory RDF graph (set of triples).
#[derive(Debug, Clone, Default)]
pub struct Graph {
    triples: HashSet<Triple>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a triple.  Returns `true` if it was newly added.
    pub fn insert(&mut self, triple: Triple) -> bool {
        self.triples.insert(triple)
    }

    /// Remove a triple.  Returns `true` if it was present.
    pub fn remove(&mut self, triple: &Triple) -> bool {
        self.triples.remove(triple)
    }

    /// Return `true` when the graph contains the given triple.
    pub fn contains(&self, triple: &Triple) -> bool {
        self.triples.contains(triple)
    }

    /// Number of triples.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Return `true` when the graph has no triples.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Iterator over all triples.
    pub fn iter(&self) -> impl Iterator<Item = &Triple> {
        self.triples.iter()
    }

    /// Sorted triple list (for deterministic output).
    pub fn sorted(&self) -> Vec<Triple> {
        let mut v: Vec<Triple> = self.triples.iter().cloned().collect();
        v.sort();
        v
    }
}

// ---------------------------------------------------------------------------
// Diff result
// ---------------------------------------------------------------------------

/// The diff between two graphs.
#[derive(Debug, Clone)]
pub struct GraphDiff {
    /// Triples present in the new graph but not in the old.
    pub added: Vec<Triple>,
    /// Triples present in the old graph but not in the new.
    pub removed: Vec<Triple>,
    /// Triples present in both graphs (unchanged).
    pub unchanged_count: usize,
}

impl GraphDiff {
    /// Number of added triples.
    pub fn added_count(&self) -> usize {
        self.added.len()
    }

    /// Number of removed triples.
    pub fn removed_count(&self) -> usize {
        self.removed.len()
    }

    /// Whether the diff is empty (no changes).
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Diff statistics
// ---------------------------------------------------------------------------

/// Summary statistics for a diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffStats {
    pub added: usize,
    pub removed: usize,
    pub unchanged: usize,
    /// Total triples in the old graph.
    pub old_size: usize,
    /// Total triples in the new graph.
    pub new_size: usize,
}

impl DiffStats {
    pub fn change_ratio(&self) -> f64 {
        let total = self.old_size.max(self.new_size);
        if total == 0 {
            0.0
        } else {
            (self.added + self.removed) as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Core diff computation
// ---------------------------------------------------------------------------

/// Compute the diff between `old_graph` and `new_graph` without any blank-node
/// renaming (ground triples only or already-canonicalized blank nodes).
pub fn compute_diff(old_graph: &Graph, new_graph: &Graph) -> GraphDiff {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut unchanged_count = 0usize;

    for triple in new_graph.iter() {
        if old_graph.contains(triple) {
            unchanged_count += 1;
        } else {
            added.push(triple.clone());
        }
    }
    for triple in old_graph.iter() {
        if !new_graph.contains(triple) {
            removed.push(triple.clone());
        }
    }
    added.sort();
    removed.sort();
    GraphDiff {
        added,
        removed,
        unchanged_count,
    }
}

/// Compute diff statistics for two graphs.
pub fn diff_stats(old_graph: &Graph, new_graph: &Graph) -> DiffStats {
    let diff = compute_diff(old_graph, new_graph);
    DiffStats {
        added: diff.added_count(),
        removed: diff.removed_count(),
        unchanged: diff.unchanged_count,
        old_size: old_graph.len(),
        new_size: new_graph.len(),
    }
}

// ---------------------------------------------------------------------------
// Symmetric difference
// ---------------------------------------------------------------------------

/// Return the symmetric difference: triples that are in exactly one of the two
/// graphs (neither in both nor in neither).
pub fn symmetric_difference(a: &Graph, b: &Graph) -> Vec<Triple> {
    let mut result: Vec<Triple> = a
        .iter()
        .filter(|t| !b.contains(t))
        .chain(b.iter().filter(|t| !a.contains(t)))
        .cloned()
        .collect();
    result.sort();
    result
}

// ---------------------------------------------------------------------------
// RDF Patch format
// ---------------------------------------------------------------------------

/// A single operation in an RDF Patch document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchOp {
    /// `A s p o .`  — add triple.
    Add(Triple),
    /// `D s p o .`  — delete triple.
    Delete(Triple),
}

impl fmt::Display for PatchOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatchOp::Add(t) => write!(f, "A {} {} {} .", t.subject, t.predicate, t.object),
            PatchOp::Delete(t) => write!(f, "D {} {} {} .", t.subject, t.predicate, t.object),
        }
    }
}

/// An ordered sequence of patch operations.
#[derive(Debug, Clone, Default)]
pub struct RdfPatch {
    pub ops: Vec<PatchOp>,
}

impl RdfPatch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Return `true` when the patch has no operations.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl std::fmt::Display for RdfPatch {
    /// Serialise the patch to a multi-line string in RDF Patch text format.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self
            .ops
            .iter()
            .map(|op| op.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// Patch generation
// ---------------------------------------------------------------------------

/// Generate an RDF Patch that transforms `old_graph` into `new_graph`.
///
/// Additions come first (sorted), then deletions (sorted).
pub fn generate_patch(old_graph: &Graph, new_graph: &Graph) -> RdfPatch {
    let diff = compute_diff(old_graph, new_graph);
    let mut ops = Vec::with_capacity(diff.added.len() + diff.removed.len());
    for t in diff.added {
        ops.push(PatchOp::Add(t));
    }
    for t in diff.removed {
        ops.push(PatchOp::Delete(t));
    }
    RdfPatch { ops }
}

// ---------------------------------------------------------------------------
// Patch application
// ---------------------------------------------------------------------------

/// Error returned when patch application fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchError {
    /// A delete operation targeted a triple that is not in the graph.
    TripleNotFound(Triple),
}

impl fmt::Display for PatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatchError::TripleNotFound(t) => write!(f, "triple not found: {t}"),
        }
    }
}

/// Apply an RDF Patch to `base`, returning the updated graph or an error.
///
/// Operations are applied in order.  A delete of a non-existent triple
/// is reported as `PatchError::TripleNotFound`.
pub fn apply_patch(base: &Graph, patch: &RdfPatch) -> Result<Graph, PatchError> {
    let mut graph = base.clone();
    for op in &patch.ops {
        match op {
            PatchOp::Add(t) => {
                graph.insert(t.clone());
            }
            PatchOp::Delete(t) => {
                if !graph.remove(t) {
                    return Err(PatchError::TripleNotFound(t.clone()));
                }
            }
        }
    }
    Ok(graph)
}

/// Lenient patch application: ignore delete-of-missing silently.
pub fn apply_patch_lenient(base: &Graph, patch: &RdfPatch) -> Graph {
    let mut graph = base.clone();
    for op in &patch.ops {
        match op {
            PatchOp::Add(t) => {
                graph.insert(t.clone());
            }
            PatchOp::Delete(t) => {
                graph.remove(t);
            }
        }
    }
    graph
}

// ---------------------------------------------------------------------------
// Blank-node-aware isomorphic diff
// ---------------------------------------------------------------------------

/// Return `true` when `a` and `b` are isomorphic under some bijection of blank
/// node labels.
///
/// Uses a simple exhaustive bijection search that is exponential in the number
/// of distinct blank nodes but practical for small graphs (< ~20 blank nodes).
pub fn are_isomorphic(a: &Graph, b: &Graph) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let a_blanks = collect_blank_nodes(a);
    let b_blanks = collect_blank_nodes(b);
    if a_blanks.len() != b_blanks.len() {
        return false;
    }
    if a_blanks.is_empty() {
        // No blank nodes — simple set equality
        return a.iter().all(|t| b.contains(t));
    }
    let a_blanks_vec: Vec<String> = a_blanks.into_iter().collect();
    let b_blanks_vec: Vec<String> = b_blanks.into_iter().collect();
    try_bijections(a, b, &a_blanks_vec, &b_blanks_vec, HashMap::new())
}

fn collect_blank_nodes(g: &Graph) -> HashSet<String> {
    let mut set = HashSet::new();
    for t in g.iter() {
        if is_blank(&t.subject) {
            set.insert(t.subject.clone());
        }
        if is_blank(&t.object) {
            set.insert(t.object.clone());
        }
    }
    set
}

/// Recursively try all bijections from `a_blanks` to `b_blanks`.
fn try_bijections(
    a: &Graph,
    b: &Graph,
    a_blanks: &[String],
    b_blanks: &[String],
    mapping: HashMap<String, String>,
) -> bool {
    if mapping.len() == a_blanks.len() {
        // Complete mapping — check whether A maps exactly onto B
        return a.iter().all(|t| b.contains(&rename_triple(t, &mapping)));
    }
    let idx = mapping.len();
    let a_blank = &a_blanks[idx];
    let used: HashSet<&String> = mapping.values().collect();
    for b_blank in b_blanks {
        if used.contains(b_blank) {
            continue;
        }
        let mut next = mapping.clone();
        next.insert(a_blank.clone(), b_blank.clone());
        if try_bijections(a, b, a_blanks, b_blanks, next) {
            return true;
        }
    }
    false
}

fn rename_triple(t: &Triple, mapping: &HashMap<String, String>) -> Triple {
    Triple::new(
        mapping
            .get(&t.subject)
            .cloned()
            .unwrap_or_else(|| t.subject.clone()),
        mapping
            .get(&t.predicate)
            .cloned()
            .unwrap_or_else(|| t.predicate.clone()),
        mapping
            .get(&t.object)
            .cloned()
            .unwrap_or_else(|| t.object.clone()),
    )
}

/// Compute a diff that is blank-node-aware: triples that are isomorphically
/// equivalent are counted as unchanged.
///
/// This is a coarse version: it partitions ground triples first, then
/// checks the remaining blank-node triples for graph isomorphism.
pub fn compute_isomorphic_diff(old_graph: &Graph, new_graph: &Graph) -> GraphDiff {
    // Separate ground from blank-node triples in each graph
    let (old_ground, old_bnode): (Graph, Graph) = partition(old_graph);
    let (new_ground, new_bnode): (Graph, Graph) = partition(new_graph);

    // Ground triples: exact match
    let ground_diff = compute_diff(&old_ground, &new_ground);

    // Blank-node triples: isomorphic match (treat as unchanged if graphs are isomorphic)
    let (bnode_added, bnode_removed, bnode_unchanged) = if are_isomorphic(&old_bnode, &new_bnode) {
        (vec![], vec![], old_bnode.len())
    } else {
        let d = compute_diff(&old_bnode, &new_bnode);
        let uc = d.unchanged_count;
        (d.added, d.removed, uc)
    };

    let mut added = ground_diff.added;
    added.extend(bnode_added);
    added.sort();

    let mut removed = ground_diff.removed;
    removed.extend(bnode_removed);
    removed.sort();

    GraphDiff {
        added,
        removed,
        unchanged_count: ground_diff.unchanged_count + bnode_unchanged,
    }
}

fn partition(g: &Graph) -> (Graph, Graph) {
    let mut ground = Graph::new();
    let mut bnode = Graph::new();
    for t in g.iter() {
        if t.has_blank_node() {
            bnode.insert(t.clone());
        } else {
            ground.insert(t.clone());
        }
    }
    (ground, bnode)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    fn g(triples: &[Triple]) -> Graph {
        let mut graph = Graph::new();
        for triple in triples {
            graph.insert(triple.clone());
        }
        graph
    }

    // -----------------------------------------------------------------------
    // Graph basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_graph_insert_remove() {
        let mut graph = Graph::new();
        let triple = t("<s>", "<p>", "<o>");
        assert!(graph.insert(triple.clone()));
        assert!(graph.contains(&triple));
        assert!(graph.remove(&triple));
        assert!(!graph.contains(&triple));
    }

    #[test]
    fn test_graph_len_and_is_empty() {
        let mut graph = Graph::new();
        assert!(graph.is_empty());
        graph.insert(t("<s>", "<p>", "<o>"));
        assert_eq!(graph.len(), 1);
        assert!(!graph.is_empty());
    }

    // -----------------------------------------------------------------------
    // compute_diff
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_added_triples() {
        let old = g(&[t("<s>", "<p>", "<o1>")]);
        let new = g(&[t("<s>", "<p>", "<o1>"), t("<s>", "<p>", "<o2>")]);
        let diff = compute_diff(&old, &new);
        assert_eq!(diff.added_count(), 1);
        assert_eq!(diff.removed_count(), 0);
        assert_eq!(diff.unchanged_count, 1);
        assert_eq!(diff.added[0], t("<s>", "<p>", "<o2>"));
    }

    #[test]
    fn test_diff_removed_triples() {
        let old = g(&[t("<s>", "<p>", "<o1>"), t("<s>", "<p>", "<o2>")]);
        let new = g(&[t("<s>", "<p>", "<o1>")]);
        let diff = compute_diff(&old, &new);
        assert_eq!(diff.added_count(), 0);
        assert_eq!(diff.removed_count(), 1);
        assert_eq!(diff.removed[0], t("<s>", "<p>", "<o2>"));
    }

    #[test]
    fn test_diff_unchanged() {
        let old = g(&[t("<s>", "<p>", "<o>")]);
        let new = g(&[t("<s>", "<p>", "<o>")]);
        let diff = compute_diff(&old, &new);
        assert!(diff.is_empty());
        assert_eq!(diff.unchanged_count, 1);
    }

    #[test]
    fn test_diff_both_added_and_removed() {
        let old = g(&[t("<s>", "<p>", "<a>"), t("<s>", "<p>", "<b>")]);
        let new = g(&[t("<s>", "<p>", "<b>"), t("<s>", "<p>", "<c>")]);
        let diff = compute_diff(&old, &new);
        assert_eq!(diff.added, vec![t("<s>", "<p>", "<c>")]);
        assert_eq!(diff.removed, vec![t("<s>", "<p>", "<a>")]);
        assert_eq!(diff.unchanged_count, 1);
    }

    #[test]
    fn test_diff_empty_graphs() {
        let old = Graph::new();
        let new = Graph::new();
        let diff = compute_diff(&old, &new);
        assert!(diff.is_empty());
    }

    // -----------------------------------------------------------------------
    // diff_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_stats() {
        let old = g(&[t("<s>", "<p>", "<a>"), t("<s>", "<p>", "<b>")]);
        let new = g(&[t("<s>", "<p>", "<b>"), t("<s>", "<p>", "<c>")]);
        let stats = diff_stats(&old, &new);
        assert_eq!(stats.added, 1);
        assert_eq!(stats.removed, 1);
        assert_eq!(stats.unchanged, 1);
        assert_eq!(stats.old_size, 2);
        assert_eq!(stats.new_size, 2);
    }

    #[test]
    fn test_diff_stats_change_ratio() {
        let old = g(&[t("<s>", "<p>", "<a>"), t("<s>", "<p>", "<b>")]);
        let new = g(&[t("<s>", "<p>", "<a>"), t("<s>", "<p>", "<c>")]);
        let stats = diff_stats(&old, &new);
        // 1 added + 1 removed out of 2 max = 1.0
        assert!((stats.change_ratio() - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // symmetric_difference
    // -----------------------------------------------------------------------

    #[test]
    fn test_symmetric_difference() {
        let a = g(&[t("<s>", "<p>", "<a>"), t("<s>", "<p>", "<b>")]);
        let b = g(&[t("<s>", "<p>", "<b>"), t("<s>", "<p>", "<c>")]);
        let sym = symmetric_difference(&a, &b);
        assert_eq!(sym.len(), 2);
        assert!(sym.contains(&t("<s>", "<p>", "<a>")));
        assert!(sym.contains(&t("<s>", "<p>", "<c>")));
    }

    #[test]
    fn test_symmetric_difference_identical() {
        let a = g(&[t("<s>", "<p>", "<o>")]);
        let b = g(&[t("<s>", "<p>", "<o>")]);
        assert!(symmetric_difference(&a, &b).is_empty());
    }

    // -----------------------------------------------------------------------
    // generate_patch / apply_patch
    // -----------------------------------------------------------------------

    #[test]
    fn test_patch_roundtrip() {
        let old = g(&[t("<s>", "<p>", "<a>"), t("<s>", "<p>", "<b>")]);
        let new = g(&[t("<s>", "<p>", "<b>"), t("<s>", "<p>", "<c>")]);
        let patch = generate_patch(&old, &new);
        let result = apply_patch(&old, &patch).expect("patch should apply cleanly");
        assert_eq!(result.len(), new.len());
        for t in new.iter() {
            assert!(result.contains(t));
        }
    }

    #[test]
    fn test_patch_to_string() {
        let old = g(&[t("<s>", "<p>", "<a>")]);
        let new = g(&[t("<s>", "<p>", "<b>")]);
        let patch = generate_patch(&old, &new);
        let s = patch.to_string();
        assert!(s.contains('A') || s.contains('D'));
    }

    #[test]
    fn test_patch_delete_missing_returns_error() {
        let base = g(&[t("<s>", "<p>", "<a>")]);
        let mut patch = RdfPatch::new();
        patch
            .ops
            .push(PatchOp::Delete(t("<s>", "<p>", "<missing>")));
        let result = apply_patch(&base, &patch);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_patch_lenient_delete_missing() {
        let base = g(&[t("<s>", "<p>", "<a>")]);
        let mut patch = RdfPatch::new();
        patch
            .ops
            .push(PatchOp::Delete(t("<s>", "<p>", "<missing>")));
        let result = apply_patch_lenient(&base, &patch);
        assert!(result.contains(&t("<s>", "<p>", "<a>")));
    }

    #[test]
    fn test_patch_empty() {
        let old = g(&[t("<s>", "<p>", "<o>")]);
        let new = g(&[t("<s>", "<p>", "<o>")]);
        let patch = generate_patch(&old, &new);
        assert!(patch.is_empty());
    }

    // -----------------------------------------------------------------------
    // PatchOp display
    // -----------------------------------------------------------------------

    #[test]
    fn test_patch_op_display_add() {
        let op = PatchOp::Add(t("<s>", "<p>", "<o>"));
        assert_eq!(op.to_string(), "A <s> <p> <o> .");
    }

    #[test]
    fn test_patch_op_display_delete() {
        let op = PatchOp::Delete(t("<s>", "<p>", "<o>"));
        assert_eq!(op.to_string(), "D <s> <p> <o> .");
    }

    // -----------------------------------------------------------------------
    // Blank-node-aware isomorphic diff
    // -----------------------------------------------------------------------

    #[test]
    fn test_are_isomorphic_ground_equal() {
        let a = g(&[t("<s>", "<p>", "<o>")]);
        let b = g(&[t("<s>", "<p>", "<o>")]);
        assert!(are_isomorphic(&a, &b));
    }

    #[test]
    fn test_are_isomorphic_ground_different() {
        let a = g(&[t("<s>", "<p>", "<o1>")]);
        let b = g(&[t("<s>", "<p>", "<o2>")]);
        assert!(!are_isomorphic(&a, &b));
    }

    #[test]
    fn test_are_isomorphic_blank_nodes_matching() {
        let a = g(&[t("_:b0", "<p>", "<o>")]);
        let b = g(&[t("_:x", "<p>", "<o>")]);
        assert!(are_isomorphic(&a, &b));
    }

    #[test]
    fn test_are_isomorphic_blank_nodes_no_match() {
        let a = g(&[t("_:b0", "<p>", "<o1>")]);
        let b = g(&[t("_:x", "<p>", "<o2>")]);
        assert!(!are_isomorphic(&a, &b));
    }

    #[test]
    fn test_isomorphic_diff_blank_nodes_unchanged() {
        let old = g(&[t("_:b0", "<p>", "<o>")]);
        let new = g(&[t("_:b1", "<p>", "<o>")]);
        // Isomorphically equal → no changes
        let diff = compute_isomorphic_diff(&old, &new);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
    }

    // -----------------------------------------------------------------------
    // Triple display and has_blank_node
    // -----------------------------------------------------------------------

    #[test]
    fn test_triple_display() {
        let triple = t("<s>", "<p>", "<o>");
        assert_eq!(triple.to_string(), "<s> <p> <o> .");
    }

    #[test]
    fn test_has_blank_node() {
        assert!(t("_:b0", "<p>", "<o>").has_blank_node());
        assert!(!t("<s>", "<p>", "<o>").has_blank_node());
    }

    // -----------------------------------------------------------------------
    // Additional edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_graph_insert_duplicate_returns_false() {
        let mut graph = Graph::new();
        let triple = t("<s>", "<p>", "<o>");
        assert!(graph.insert(triple.clone()));
        assert!(!graph.insert(triple)); // second insert returns false
    }

    #[test]
    fn test_graph_remove_absent_returns_false() {
        let mut graph = Graph::new();
        let triple = t("<s>", "<p>", "<o>");
        assert!(!graph.remove(&triple));
    }

    #[test]
    fn test_graph_sorted_order() {
        let mut graph = Graph::new();
        graph.insert(t("<s>", "<p>", "<c>"));
        graph.insert(t("<s>", "<p>", "<a>"));
        graph.insert(t("<s>", "<p>", "<b>"));
        let sorted = graph.sorted();
        assert_eq!(sorted[0], t("<s>", "<p>", "<a>"));
        assert_eq!(sorted[2], t("<s>", "<p>", "<c>"));
    }

    #[test]
    fn test_diff_old_empty() {
        let old = Graph::new();
        let new = g(&[t("<s>", "<p>", "<o>")]);
        let diff = compute_diff(&old, &new);
        assert_eq!(diff.added_count(), 1);
        assert_eq!(diff.removed_count(), 0);
    }

    #[test]
    fn test_diff_new_empty() {
        let old = g(&[t("<s>", "<p>", "<o>")]);
        let new = Graph::new();
        let diff = compute_diff(&old, &new);
        assert_eq!(diff.removed_count(), 1);
        assert_eq!(diff.added_count(), 0);
    }

    #[test]
    fn test_diff_stats_change_ratio_zero() {
        let old = g(&[t("<s>", "<p>", "<o>")]);
        let new = g(&[t("<s>", "<p>", "<o>")]);
        let stats = diff_stats(&old, &new);
        assert!((stats.change_ratio()).abs() < 1e-9);
    }

    #[test]
    fn test_symmetric_difference_a_subset_of_b() {
        let a = g(&[t("<s>", "<p>", "<o>")]);
        let b = g(&[t("<s>", "<p>", "<o>"), t("<s>", "<p>", "<x>")]);
        let sym = symmetric_difference(&a, &b);
        assert_eq!(sym.len(), 1);
        assert_eq!(sym[0], t("<s>", "<p>", "<x>"));
    }

    #[test]
    fn test_symmetric_difference_both_disjoint() {
        let a = g(&[t("<s>", "<p>", "<a>")]);
        let b = g(&[t("<s>", "<p>", "<b>")]);
        let sym = symmetric_difference(&a, &b);
        assert_eq!(sym.len(), 2);
    }

    #[test]
    fn test_patch_add_only() {
        let old = Graph::new();
        let new = g(&[t("<s>", "<p>", "<o>")]);
        let patch = generate_patch(&old, &new);
        assert_eq!(patch.len(), 1);
        assert!(matches!(&patch.ops[0], PatchOp::Add(_)));
    }

    #[test]
    fn test_patch_delete_only() {
        let old = g(&[t("<s>", "<p>", "<o>")]);
        let new = Graph::new();
        let patch = generate_patch(&old, &new);
        assert_eq!(patch.len(), 1);
        assert!(matches!(&patch.ops[0], PatchOp::Delete(_)));
    }

    #[test]
    fn test_apply_patch_add() {
        let base = Graph::new();
        let mut patch = RdfPatch::new();
        patch.ops.push(PatchOp::Add(t("<s>", "<p>", "<o>")));
        let result = apply_patch(&base, &patch).expect("apply");
        assert!(result.contains(&t("<s>", "<p>", "<o>")));
    }

    #[test]
    fn test_apply_patch_delete() {
        let base = g(&[t("<s>", "<p>", "<o>")]);
        let mut patch = RdfPatch::new();
        patch.ops.push(PatchOp::Delete(t("<s>", "<p>", "<o>")));
        let result = apply_patch(&base, &patch).expect("apply");
        assert!(!result.contains(&t("<s>", "<p>", "<o>")));
    }

    #[test]
    fn test_patch_error_display() {
        let err = PatchError::TripleNotFound(t("<s>", "<p>", "<o>"));
        let s = err.to_string();
        assert!(s.contains("not found"));
    }

    #[test]
    fn test_patch_is_empty_initially() {
        let patch = RdfPatch::new();
        assert!(patch.is_empty());
        assert_eq!(patch.len(), 0);
    }

    #[test]
    fn test_are_isomorphic_different_sizes() {
        let a = g(&[t("<s>", "<p>", "<o1>"), t("<s>", "<p>", "<o2>")]);
        let b = g(&[t("<s>", "<p>", "<o1>")]);
        assert!(!are_isomorphic(&a, &b));
    }

    #[test]
    fn test_are_isomorphic_two_blank_nodes() {
        // a: _:b0 <p> <o> and _:b1 <q> <o>
        // b: _:x  <p> <o> and _:y  <q> <o>  — isomorphic
        let a = g(&[t("_:b0", "<p>", "<o>"), t("_:b1", "<q>", "<o>")]);
        let b = g(&[t("_:x", "<p>", "<o>"), t("_:y", "<q>", "<o>")]);
        assert!(are_isomorphic(&a, &b));
    }

    #[test]
    fn test_triple_ordering() {
        let t1 = t("<a>", "<p>", "<o>");
        let t2 = t("<b>", "<p>", "<o>");
        assert!(t1 < t2);
    }

    #[test]
    fn test_triple_equality() {
        let t1 = t("<s>", "<p>", "<o>");
        let t2 = t("<s>", "<p>", "<o>");
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_triple_inequality() {
        let t1 = t("<s>", "<p>", "<o1>");
        let t2 = t("<s>", "<p>", "<o2>");
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_compute_isomorphic_diff_ground_and_bnode_mixed() {
        let old = g(&[t("<s>", "<p>", "<a>"), t("_:b0", "<q>", "<x>")]);
        let new = g(&[t("<s>", "<p>", "<b>"), t("_:b1", "<q>", "<x>")]);
        let diff = compute_isomorphic_diff(&old, &new);
        // Ground: <a> removed, <b> added; bnode part is isomorphic (unchanged)
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.removed.len(), 1);
    }
}
