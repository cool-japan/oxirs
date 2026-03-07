//! RDF dataset diff computation and patch application.
//!
//! Provides tools for computing the difference between two RDF datasets,
//! applying patches, inverting diffs, and composing multiple diffs.

use std::collections::HashSet;

/// A single RDF triple represented as (subject, predicate, object) strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    /// Subject IRI or blank node identifier.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object IRI, blank node, or literal value.
    pub object: String,
}

impl Triple {
    /// Construct a new triple from string-convertible values.
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

/// Statistics about a diff operation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiffStats {
    /// Number of triples present in `after` but not in `before`.
    pub added_count: usize,
    /// Number of triples present in `before` but not in `after`.
    pub removed_count: usize,
    /// Number of triples present in both datasets (unchanged).
    pub unchanged_count: usize,
}

/// The difference between two RDF datasets.
#[derive(Debug, Clone)]
pub struct DatasetDiff {
    /// Triples added (in `after` but not `before`).
    pub added: Vec<Triple>,
    /// Triples removed (in `before` but not `after`).
    pub removed: Vec<Triple>,
    /// Summary statistics.
    pub stats: DiffStats,
}

/// Engine for computing and manipulating RDF dataset diffs.
pub struct RdfDiffEngine;

impl RdfDiffEngine {
    /// Compute the diff between `before` and `after` triple sets.
    ///
    /// - `added`   = triples in `after` that are not in `before`
    /// - `removed` = triples in `before` that are not in `after`
    pub fn compute(before: &[Triple], after: &[Triple]) -> DatasetDiff {
        let before_set: HashSet<&Triple> = before.iter().collect();
        let after_set: HashSet<&Triple> = after.iter().collect();

        let added: Vec<Triple> = after
            .iter()
            .filter(|t| !before_set.contains(t))
            .cloned()
            .collect();

        let removed: Vec<Triple> = before
            .iter()
            .filter(|t| !after_set.contains(t))
            .cloned()
            .collect();

        let unchanged_count = before.iter().filter(|t| after_set.contains(t)).count();

        let stats = DiffStats {
            added_count: added.len(),
            removed_count: removed.len(),
            unchanged_count,
        };

        DatasetDiff {
            added,
            removed,
            stats,
        }
    }

    /// Apply `diff` to `base`, returning the resulting dataset.
    ///
    /// Removes all triples listed in `diff.removed` from `base`, then appends
    /// all triples in `diff.added`, de-duplicating the result.
    pub fn apply_diff(mut base: Vec<Triple>, diff: &DatasetDiff) -> Vec<Triple> {
        let removed_set: HashSet<&Triple> = diff.removed.iter().collect();
        base.retain(|t| !removed_set.contains(t));

        for triple in &diff.added {
            if !base.contains(triple) {
                base.push(triple.clone());
            }
        }

        base
    }

    /// Invert a diff — swap `added` and `removed` so applying the result
    /// undoes the original diff.
    pub fn invert(diff: DatasetDiff) -> DatasetDiff {
        let stats = DiffStats {
            added_count: diff.removed.len(),
            removed_count: diff.added.len(),
            unchanged_count: diff.stats.unchanged_count,
        };
        DatasetDiff {
            added: diff.removed,
            removed: diff.added,
            stats,
        }
    }

    /// Compose two diffs into a single diff representing the net effect of
    /// applying `d1` followed by `d2`.
    ///
    /// Net effect:
    /// - Net added   = (d1.added ∪ d2.added) \ d2.removed
    /// - Net removed = (d1.removed ∪ d2.removed) \ d2.added
    pub fn compose(d1: DatasetDiff, d2: DatasetDiff) -> DatasetDiff {
        let d2_removed_set: HashSet<&Triple> = d2.removed.iter().collect();
        let d2_added_set: HashSet<&Triple> = d2.added.iter().collect();

        // Net added: items added in d1 not removed in d2, union items added in d2.
        let mut net_added: Vec<Triple> = d1
            .added
            .iter()
            .filter(|t| !d2_removed_set.contains(t))
            .cloned()
            .collect();
        for t in &d2.added {
            if !net_added.contains(t) {
                net_added.push(t.clone());
            }
        }

        // Net removed: items removed in d1 not re-added in d2, union items removed in d2.
        let mut net_removed: Vec<Triple> = d1
            .removed
            .iter()
            .filter(|t| !d2_added_set.contains(t))
            .cloned()
            .collect();
        for t in &d2.removed {
            if !net_removed.contains(t) {
                net_removed.push(t.clone());
            }
        }

        let stats = DiffStats {
            added_count: net_added.len(),
            removed_count: net_removed.len(),
            unchanged_count: 0, // unknown after composition
        };

        DatasetDiff {
            added: net_added,
            removed: net_removed,
            stats,
        }
    }

    /// Return `true` if `diff` represents no change (no added and no removed triples).
    pub fn is_empty(diff: &DatasetDiff) -> bool {
        diff.added.is_empty() && diff.removed.is_empty()
    }
}

/// A sequence of diffs that can be applied as a single patch.
#[derive(Debug, Clone, Default)]
pub struct DatasetPatch {
    /// Ordered list of diffs in this patch.
    pub patches: Vec<DatasetDiff>,
}

impl DatasetPatch {
    /// Create a new, empty patch.
    pub fn new() -> Self {
        Self {
            patches: Vec::new(),
        }
    }

    /// Append a diff to the end of this patch.
    pub fn add_diff(&mut self, diff: DatasetDiff) {
        self.patches.push(diff);
    }

    /// Apply all diffs in order against `base` and return the result.
    pub fn apply_all(&self, mut base: Vec<Triple>) -> Vec<Triple> {
        for diff in &self.patches {
            base = RdfDiffEngine::apply_diff(base, diff);
        }
        base
    }

    /// Total number of diffs in this patch.
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    /// Return `true` if the patch contains no diffs.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    // --- Triple tests ---

    #[test]
    fn test_triple_equality() {
        let a = t("s", "p", "o");
        let b = t("s", "p", "o");
        assert_eq!(a, b);
    }

    #[test]
    fn test_triple_inequality() {
        let a = t("s", "p", "o1");
        let b = t("s", "p", "o2");
        assert_ne!(a, b);
    }

    // --- RdfDiffEngine::compute tests ---

    #[test]
    fn test_diff_empty_datasets() {
        let diff = RdfDiffEngine::compute(&[], &[]);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert_eq!(diff.stats.unchanged_count, 0);
    }

    #[test]
    fn test_diff_all_added() {
        let before: Vec<Triple> = vec![];
        let after = vec![t("s", "p", "o")];
        let diff = RdfDiffEngine::compute(&before, &after);
        assert_eq!(diff.added.len(), 1);
        assert!(diff.removed.is_empty());
        assert_eq!(diff.stats.added_count, 1);
    }

    #[test]
    fn test_diff_all_removed() {
        let before = vec![t("s", "p", "o")];
        let after: Vec<Triple> = vec![];
        let diff = RdfDiffEngine::compute(&before, &after);
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.stats.removed_count, 1);
    }

    #[test]
    fn test_diff_no_change() {
        let dataset = vec![t("s", "p", "o")];
        let diff = RdfDiffEngine::compute(&dataset, &dataset);
        assert!(RdfDiffEngine::is_empty(&diff));
        assert_eq!(diff.stats.unchanged_count, 1);
    }

    #[test]
    fn test_diff_mixed() {
        let before = vec![t("s", "p", "o1"), t("s", "p", "o2")];
        let after = vec![t("s", "p", "o2"), t("s", "p", "o3")];
        let diff = RdfDiffEngine::compute(&before, &after);
        assert_eq!(diff.added, vec![t("s", "p", "o3")]);
        assert_eq!(diff.removed, vec![t("s", "p", "o1")]);
        assert_eq!(diff.stats.unchanged_count, 1);
    }

    // --- RdfDiffEngine::apply_diff tests ---

    #[test]
    fn test_apply_diff_add_triples() {
        let base = vec![t("s", "p", "o1")];
        let diff = RdfDiffEngine::compute(&base, &[t("s", "p", "o1"), t("s", "p", "o2")]);
        let result = RdfDiffEngine::apply_diff(base, &diff);
        assert!(result.contains(&t("s", "p", "o1")));
        assert!(result.contains(&t("s", "p", "o2")));
    }

    #[test]
    fn test_apply_diff_remove_triples() {
        let base = vec![t("s", "p", "o1"), t("s", "p", "o2")];
        let diff = RdfDiffEngine::compute(&base, &[t("s", "p", "o1")]);
        let result = RdfDiffEngine::apply_diff(base, &diff);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&t("s", "p", "o1")));
    }

    #[test]
    fn test_apply_diff_no_change() {
        let base = vec![t("s", "p", "o")];
        let diff = RdfDiffEngine::compute(&base, &base.clone());
        let result = RdfDiffEngine::apply_diff(base.clone(), &diff);
        assert_eq!(result, base);
    }

    // --- RdfDiffEngine::invert tests ---

    #[test]
    fn test_invert_diff() {
        let before = vec![t("s", "p", "o1")];
        let after = vec![t("s", "p", "o2")];
        let diff = RdfDiffEngine::compute(&before, &after);
        let inv = RdfDiffEngine::invert(diff);
        assert_eq!(inv.added, vec![t("s", "p", "o1")]);
        assert_eq!(inv.removed, vec![t("s", "p", "o2")]);
        assert_eq!(inv.stats.added_count, 1);
        assert_eq!(inv.stats.removed_count, 1);
    }

    #[test]
    fn test_invert_roundtrip() {
        let base = vec![t("a", "b", "c"), t("d", "e", "f")];
        let modified = vec![t("a", "b", "c"), t("x", "y", "z")];
        let diff = RdfDiffEngine::compute(&base, &modified);
        let inv = RdfDiffEngine::invert(diff);

        let restored = RdfDiffEngine::apply_diff(modified.clone(), &inv);
        // restored should equal base
        assert!(restored.contains(&t("a", "b", "c")));
        assert!(restored.contains(&t("d", "e", "f")));
        assert!(!restored.contains(&t("x", "y", "z")));
    }

    // --- RdfDiffEngine::compose tests ---

    #[test]
    fn test_compose_empty_diffs() {
        let d1 = RdfDiffEngine::compute(&[], &[]);
        let d2 = RdfDiffEngine::compute(&[], &[]);
        let composed = RdfDiffEngine::compose(d1, d2);
        assert!(RdfDiffEngine::is_empty(&composed));
    }

    #[test]
    fn test_compose_two_diffs() {
        // d1: add t1; d2: add t2
        let d1 = RdfDiffEngine::compute(&[], &[t("s", "p", "o1")]);
        let d2 = RdfDiffEngine::compute(
            &[t("s", "p", "o1")],
            &[t("s", "p", "o1"), t("s", "p", "o2")],
        );
        let composed = RdfDiffEngine::compose(d1, d2);
        // Net: both t1 and t2 should be in added
        assert!(composed.added.contains(&t("s", "p", "o1")));
        assert!(composed.added.contains(&t("s", "p", "o2")));
    }

    #[test]
    fn test_compose_add_then_remove() {
        // d1 adds t1; d2 removes t1 → net effect is nothing
        let d1 = RdfDiffEngine::compute(&[], &[t("s", "p", "o1")]);
        let d2 = RdfDiffEngine::compute(&[t("s", "p", "o1")], &[]);
        let composed = RdfDiffEngine::compose(d1, d2);
        // t1 should not be in net_added
        assert!(!composed.added.contains(&t("s", "p", "o1")));
        // t1 is in net_removed
        assert!(composed.removed.contains(&t("s", "p", "o1")));
    }

    // --- RdfDiffEngine::is_empty tests ---

    #[test]
    fn test_is_empty_true() {
        let diff = RdfDiffEngine::compute(&[t("s", "p", "o")], &[t("s", "p", "o")]);
        assert!(RdfDiffEngine::is_empty(&diff));
    }

    #[test]
    fn test_is_empty_false() {
        let diff = RdfDiffEngine::compute(&[], &[t("s", "p", "o")]);
        assert!(!RdfDiffEngine::is_empty(&diff));
    }

    // --- DatasetPatch tests ---

    #[test]
    fn test_patch_new_is_empty() {
        let patch = DatasetPatch::new();
        assert!(patch.is_empty());
        assert_eq!(patch.len(), 0);
    }

    #[test]
    fn test_patch_apply_single_diff() {
        let mut patch = DatasetPatch::new();
        let diff = RdfDiffEngine::compute(&[], &[t("s", "p", "o")]);
        patch.add_diff(diff);

        let result = patch.apply_all(vec![]);
        assert_eq!(result, vec![t("s", "p", "o")]);
    }

    #[test]
    fn test_patch_apply_multiple_diffs() {
        let mut patch = DatasetPatch::new();
        // Step 1: add t1
        patch.add_diff(RdfDiffEngine::compute(&[], &[t("s", "p", "o1")]));
        // Step 2: add t2 to existing set
        patch.add_diff(RdfDiffEngine::compute(
            &[t("s", "p", "o1")],
            &[t("s", "p", "o1"), t("s", "p", "o2")],
        ));

        let result = patch.apply_all(vec![]);
        assert!(result.contains(&t("s", "p", "o1")));
        assert!(result.contains(&t("s", "p", "o2")));
    }

    #[test]
    fn test_patch_apply_all_on_empty_patch() {
        let patch = DatasetPatch::new();
        let base = vec![t("s", "p", "o")];
        let result = patch.apply_all(base.clone());
        assert_eq!(result, base);
    }

    #[test]
    fn test_patch_len() {
        let mut patch = DatasetPatch::new();
        assert_eq!(patch.len(), 0);
        patch.add_diff(RdfDiffEngine::compute(&[], &[]));
        assert_eq!(patch.len(), 1);
        patch.add_diff(RdfDiffEngine::compute(&[], &[]));
        assert_eq!(patch.len(), 2);
    }
}
