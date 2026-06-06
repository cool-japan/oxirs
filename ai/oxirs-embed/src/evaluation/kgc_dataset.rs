//! Knowledge Graph Completion dataset abstractions.
//!
//! This module provides [`KgcDataset`], which holds the canonical
//! train / valid / test split used by FB15k-237 / WN18RR style benchmarks,
//! together with:
//!
//! - A compact [`EvaluationTriple`] that stores entity/relation strings.
//! - A tiny **synthetic** dataset (`tiny_synthetic()`) for unit testing
//!   without network access.
//! - TSV loading (`from_tsv`) for real benchmark files.
//! - A helper to collect all positive triples for filtered evaluation.

use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────────────
// EvaluationTriple
// ─────────────────────────────────────────────────────────────────────────────

/// A single (head, relation, tail) triple stored as owned strings.
///
/// The strings mirror whatever vocabulary the dataset uses (e.g. Freebase
/// mid's, WordNet synsets, or short symbolic names in synthetic datasets).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvaluationTriple {
    /// Head entity (subject).
    pub head: String,
    /// Relation type (predicate).
    pub relation: String,
    /// Tail entity (object).
    pub tail: String,
}

impl EvaluationTriple {
    /// Construct a new triple from string slices.
    pub fn new(
        head: impl Into<String>,
        relation: impl Into<String>,
        tail: impl Into<String>,
    ) -> Self {
        Self {
            head: head.into(),
            relation: relation.into(),
            tail: tail.into(),
        }
    }

    /// Convert to the canonical `(head, relation, tail)` tuple owned form.
    pub fn as_tuple(&self) -> (&str, &str, &str) {
        (&self.head, &self.relation, &self.tail)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KgcDataset
// ─────────────────────────────────────────────────────────────────────────────

/// A knowledge graph completion dataset split into train / valid / test sets.
///
/// The `entity_set` and `relation_set` fields contain the union of all
/// entities and relations across all three splits, so that filtered ranking
/// can use the full vocabulary.
#[derive(Debug, Clone)]
pub struct KgcDataset {
    /// Training triples (used for model training).
    pub train: Vec<EvaluationTriple>,
    /// Validation triples.
    pub valid: Vec<EvaluationTriple>,
    /// Test triples (held-out evaluation split).
    pub test: Vec<EvaluationTriple>,
    /// Union of all entity strings across all splits.
    pub entity_set: HashSet<String>,
    /// Union of all relation strings across all splits.
    pub relation_set: HashSet<String>,
}

impl KgcDataset {
    /// Build the dataset from pre-split triple vectors.
    ///
    /// Entity and relation vocabularies are computed automatically from the
    /// union of all three splits.
    pub fn from_splits(
        train: Vec<EvaluationTriple>,
        valid: Vec<EvaluationTriple>,
        test: Vec<EvaluationTriple>,
    ) -> Self {
        let mut entity_set = HashSet::new();
        let mut relation_set = HashSet::new();

        for triple in train.iter().chain(valid.iter()).chain(test.iter()) {
            entity_set.insert(triple.head.clone());
            entity_set.insert(triple.tail.clone());
            relation_set.insert(triple.relation.clone());
        }

        Self {
            train,
            valid,
            test,
            entity_set,
            relation_set,
        }
    }

    /// Build a tiny 4-entity / 3-relation synthetic dataset for unit tests.
    ///
    /// Graph structure (conceptually a small social / geographical KG):
    ///
    /// ```text
    /// alice   --[knows]-->   bob
    /// alice   --[knows]-->   carol
    /// bob     --[knows]-->   carol
    /// alice   --[lives_in]--> paris
    /// bob     --[lives_in]--> london
    /// carol   --[works_at]--> acme
    /// paris   --[located_in]--> france
    /// london  --[located_in]--> uk
    /// ```
    ///
    /// Train/valid/test split: 6 / 1 / 1 triples.
    pub fn tiny_synthetic() -> Self {
        let all: Vec<EvaluationTriple> = vec![
            EvaluationTriple::new("alice", "knows", "bob"),
            EvaluationTriple::new("alice", "knows", "carol"),
            EvaluationTriple::new("bob", "knows", "carol"),
            EvaluationTriple::new("alice", "lives_in", "paris"),
            EvaluationTriple::new("bob", "lives_in", "london"),
            EvaluationTriple::new("carol", "works_at", "acme"),
            EvaluationTriple::new("paris", "located_in", "france"),
            EvaluationTriple::new("london", "located_in", "uk"),
        ];

        // Deterministic split: last 1 triple → test, penultimate → valid, rest → train
        let n = all.len();
        let train_end = n.saturating_sub(2);
        let valid_end = n.saturating_sub(1);

        let train = all[..train_end].to_vec();
        let valid = all[train_end..valid_end].to_vec();
        let test = all[valid_end..].to_vec();

        Self::from_splits(train, valid, test)
    }

    /// Load a dataset from three parallel TSV files (one per split).
    ///
    /// Each file must contain one triple per line in the format:
    ///
    /// ```text
    /// <head>\t<relation>\t<tail>
    /// ```
    ///
    /// Blank lines and lines starting with `#` are silently skipped.
    ///
    /// `train_tsv`, `valid_tsv`, `test_tsv` — file contents (not paths),
    /// passed as string slices so the caller controls I/O (making this
    /// trivially testable without temporary files).
    pub fn from_tsv(train_tsv: &str, valid_tsv: &str, test_tsv: &str) -> Self {
        let parse = |tsv: &str| -> Vec<EvaluationTriple> {
            tsv.lines()
                .filter(|l| {
                    let trimmed = l.trim();
                    !trimmed.is_empty() && !trimmed.starts_with('#')
                })
                .filter_map(|line| {
                    let mut parts = line.splitn(3, '\t');
                    let head = parts.next()?.trim();
                    let relation = parts.next()?.trim();
                    let tail = parts.next()?.trim();
                    if head.is_empty() || relation.is_empty() || tail.is_empty() {
                        return None;
                    }
                    Some(EvaluationTriple::new(head, relation, tail))
                })
                .collect()
        };

        let train = parse(train_tsv);
        let valid = parse(valid_tsv);
        let test = parse(test_tsv);

        Self::from_splits(train, valid, test)
    }

    /// Return every positive triple across **all** splits as an owned
    /// `HashSet<(String, String, String)>`.
    ///
    /// Used during filtered evaluation to identify which entity substitutions
    /// are known positives and should be removed from the ranking before
    /// computing the rank of the target.
    pub fn all_positives(&self) -> HashSet<(String, String, String)> {
        self.train
            .iter()
            .chain(self.valid.iter())
            .chain(self.test.iter())
            .map(|t| (t.head.clone(), t.relation.clone(), t.tail.clone()))
            .collect()
    }

    /// Return the sorted entity vocabulary as a `Vec<String>`.
    ///
    /// Sorting ensures a deterministic entity-to-index mapping, which is
    /// important for reproducible filtered-ranking computations.
    pub fn sorted_entities(&self) -> Vec<String> {
        let mut v: Vec<String> = self.entity_set.iter().cloned().collect();
        v.sort_unstable();
        v
    }

    /// Return the sorted relation vocabulary as a `Vec<String>`.
    pub fn sorted_relations(&self) -> Vec<String> {
        let mut v: Vec<String> = self.relation_set.iter().cloned().collect();
        v.sort_unstable();
        v
    }

    /// Total number of triples across all splits.
    pub fn total_triples(&self) -> usize {
        self.train.len() + self.valid.len() + self.test.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: tiny_synthetic builds non-empty train/valid/test ──────────
    #[test]
    fn test_tiny_synthetic_non_empty_splits() {
        let ds = KgcDataset::tiny_synthetic();
        assert!(!ds.train.is_empty(), "train should be non-empty");
        assert!(!ds.valid.is_empty(), "valid should be non-empty");
        assert!(!ds.test.is_empty(), "test should be non-empty");
    }

    // ── Test 2: entity_set contains expected entities ─────────────────────
    #[test]
    fn test_entity_set_populated() {
        let ds = KgcDataset::tiny_synthetic();
        for entity in &[
            "alice", "bob", "carol", "paris", "london", "acme", "france", "uk",
        ] {
            assert!(
                ds.entity_set.contains(*entity),
                "entity_set should contain '{entity}'"
            );
        }
    }

    // ── Test 3: relation_set contains expected relations ───────────────────
    #[test]
    fn test_relation_set_populated() {
        let ds = KgcDataset::tiny_synthetic();
        for rel in &["knows", "lives_in", "works_at", "located_in"] {
            assert!(
                ds.relation_set.contains(*rel),
                "relation_set should contain '{rel}'"
            );
        }
    }

    // ── Test 4: all_positives returns all unique triples ──────────────────
    #[test]
    fn test_all_positives_coverage() {
        let ds = KgcDataset::tiny_synthetic();
        let positives = ds.all_positives();
        // Every triple in every split must appear in all_positives.
        for t in ds.train.iter().chain(ds.valid.iter()).chain(ds.test.iter()) {
            let key = (t.head.clone(), t.relation.clone(), t.tail.clone());
            assert!(
                positives.contains(&key),
                "all_positives missing ({}, {}, {})",
                t.head,
                t.relation,
                t.tail
            );
        }
        // Size equals total_triples (no duplicates in tiny_synthetic).
        assert_eq!(positives.len(), ds.total_triples());
    }

    // ── Test 5: from_tsv parses correctly ─────────────────────────────────
    #[test]
    fn test_from_tsv_parsing() {
        let train_tsv = "alice\tknows\tbob\nbob\tknows\tcarol\n";
        let valid_tsv = "alice\tlives_in\tparis\n";
        let test_tsv = "bob\tlives_in\tlondon\n";
        let ds = KgcDataset::from_tsv(train_tsv, valid_tsv, test_tsv);
        assert_eq!(ds.train.len(), 2);
        assert_eq!(ds.valid.len(), 1);
        assert_eq!(ds.test.len(), 1);
        assert!(ds.entity_set.contains("alice"));
        assert!(ds.relation_set.contains("knows"));
    }

    // ── Test 6: from_tsv skips blank lines and comments ───────────────────
    #[test]
    fn test_from_tsv_skips_blanks_and_comments() {
        let train_tsv = "# header\nalice\tknows\tbob\n\n# another comment\nbob\tknows\tcarol\n";
        let ds = KgcDataset::from_tsv(train_tsv, "", "");
        assert_eq!(ds.train.len(), 2, "should parse exactly 2 data lines");
    }

    // ── Test 7: sorted_entities is deterministic and sorted ───────────────
    #[test]
    fn test_sorted_entities_is_sorted() {
        let ds = KgcDataset::tiny_synthetic();
        let sorted = ds.sorted_entities();
        let mut copy = sorted.clone();
        copy.sort_unstable();
        assert_eq!(sorted, copy, "sorted_entities should return sorted output");
    }

    // ── Test 8: total_triples sum is consistent ────────────────────────────
    #[test]
    fn test_total_triples_consistency() {
        let ds = KgcDataset::tiny_synthetic();
        assert_eq!(
            ds.total_triples(),
            ds.train.len() + ds.valid.len() + ds.test.len()
        );
    }
}
