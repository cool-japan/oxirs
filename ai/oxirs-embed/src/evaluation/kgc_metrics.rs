//! Standard Knowledge Graph Completion (KGC) evaluation metrics.
//!
//! Implements the canonical evaluation protocol used in FB15k-237 / WN18RR
//! benchmarks:
//!
//! - **Mean Rank (MR)** — average rank of the correct entity across all
//!   queries.  Lower is better.
//! - **Mean Reciprocal Rank (MRR)** — mean of 1/rank.  Higher is better.
//! - **Hits@K** — fraction of queries where the correct entity appears in the
//!   top-K candidates.  Standard K values are 1, 3, 10.
//!
//! Both raw (all-entity) and *filtered* (known-positives removed before
//! ranking) variants are computed.  Filtered metrics are the de-facto
//! standard in the literature.
//!
//! # Example
//!
//! ```rust
//! use oxirs_embed::evaluation::kgc_metrics::EvaluationMetrics;
//!
//! let ranks = vec![1, 2, 3];
//! let filtered = vec![1, 1, 2];
//! let m = EvaluationMetrics::compute(&ranks, &filtered);
//! assert!((m.mean_reciprocal_rank - (1.0 / 3.0 + 0.5 + 1.0 / 3.0) / 3.0).abs() < 1e-9);
//! ```

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// HitsAtK helper
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Hits@K fraction from a slice of 1-based ranks.
///
/// Returns the proportion of queries where rank ≤ k.  Returns `0.0` when
/// `ranks` is empty.
pub fn hits_at_k(ranks: &[usize], k: usize) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let hits = ranks.iter().filter(|&&r| r <= k).count();
    hits as f64 / ranks.len() as f64
}

/// Compute the Mean Rank from a slice of 1-based ranks.
///
/// Returns `0.0` when `ranks` is empty.
pub fn mean_rank(ranks: &[usize]) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let total: usize = ranks.iter().sum();
    total as f64 / ranks.len() as f64
}

/// Compute the Mean Reciprocal Rank (MRR) from a slice of 1-based ranks.
///
/// Returns `0.0` when `ranks` is empty.
pub fn mean_reciprocal_rank(ranks: &[usize]) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let total: f64 = ranks.iter().map(|&r| 1.0 / r as f64).sum();
    total / ranks.len() as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// FilteredRanking
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the *filtered* rank of a target entity.
///
/// Standard filtered evaluation (Bordes et al. 2013):  after scoring all
/// candidate entities, remove every candidate that forms a *known* positive
/// triple with the fixed (head, relation) or (relation, tail) context,
/// **except** for the target itself.  The rank of the target among the
/// remaining candidates is the filtered rank.
///
/// `all_scores` — `(entity_index, score)` pairs for all entities, sorted
///   **descending** by score (highest score = rank 1).
/// `target_idx` — index of the correct entity in `all_scores`.
/// `known_true_indices` — set of entity indices that are known positives
///   (other than the target).  These are removed before ranking.
///
/// Returns the 1-based filtered rank of the target.
pub fn compute_filtered_rank(
    all_scores: &[(usize, f64)],
    target_entity: usize,
    known_true_indices: &std::collections::HashSet<usize>,
) -> usize {
    // Walk scores in descending order; stop once we find the target.
    // Skip any known-true entity (other than the target itself).
    let mut filtered_rank = 0usize;
    for &(entity_idx, _score) in all_scores {
        let is_other_positive =
            known_true_indices.contains(&entity_idx) && entity_idx != target_entity;
        if is_other_positive {
            continue;
        }
        filtered_rank += 1;
        if entity_idx == target_entity {
            return filtered_rank;
        }
    }
    // If target not found at all, return the total count + 1 (worst rank).
    all_scores.len() + 1
}

// ─────────────────────────────────────────────────────────────────────────────
// EvaluationMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated KGC evaluation metrics — both raw and filtered variants.
///
/// Computed by [`EvaluationMetrics::compute`] from two parallel slices of
/// 1-based ranks: one raw (over all entities) and one filtered (known
/// positives removed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    /// Mean rank (raw, all entities).  Lower is better.
    pub mean_rank: f64,
    /// Mean Reciprocal Rank (raw).  Higher is better.
    pub mean_reciprocal_rank: f64,
    /// Hits@1 (raw) — fraction where rank = 1.
    pub hits_at_1: f64,
    /// Hits@3 (raw) — fraction where rank ≤ 3.
    pub hits_at_3: f64,
    /// Hits@10 (raw) — fraction where rank ≤ 10.
    pub hits_at_10: f64,

    /// Mean rank (filtered).  Lower is better.
    pub filtered_mean_rank: f64,
    /// Mean Reciprocal Rank (filtered).  Higher is better.
    pub filtered_mrr: f64,
    /// Hits@1 (filtered).
    pub filtered_hits_at_1: f64,
    /// Hits@3 (filtered).
    pub filtered_hits_at_3: f64,
    /// Hits@10 (filtered).
    pub filtered_hits_at_10: f64,

    /// Total number of test triples evaluated (head + tail = 2 × test set).
    pub num_test_triples: usize,
}

impl EvaluationMetrics {
    /// Compute all metrics from parallel slices of 1-based ranks.
    ///
    /// `ranks` and `filtered_ranks` must have the same length ≥ 1.
    /// Each entry corresponds to one (head-replaced or tail-replaced) query.
    /// The slices may contain `num_queries = 2 × |test_set|` entries because
    /// both head-prediction and tail-prediction queries are aggregated
    /// together before being passed in.
    ///
    /// # Panics
    ///
    /// Panics if `ranks` and `filtered_ranks` have different lengths.
    pub fn compute(ranks: &[usize], filtered_ranks: &[usize]) -> Self {
        assert_eq!(
            ranks.len(),
            filtered_ranks.len(),
            "ranks and filtered_ranks must have the same length"
        );

        let num = ranks.len();

        Self {
            mean_rank: mean_rank(ranks),
            mean_reciprocal_rank: mean_reciprocal_rank(ranks),
            hits_at_1: hits_at_k(ranks, 1),
            hits_at_3: hits_at_k(ranks, 3),
            hits_at_10: hits_at_k(ranks, 10),

            filtered_mean_rank: mean_rank(filtered_ranks),
            filtered_mrr: mean_reciprocal_rank(filtered_ranks),
            filtered_hits_at_1: hits_at_k(filtered_ranks, 1),
            filtered_hits_at_3: hits_at_k(filtered_ranks, 3),
            filtered_hits_at_10: hits_at_k(filtered_ranks, 10),

            num_test_triples: num,
        }
    }

    /// Convenience: build all-zero metrics (useful as a sentinel).
    pub fn zero() -> Self {
        Self {
            mean_rank: 0.0,
            mean_reciprocal_rank: 0.0,
            hits_at_1: 0.0,
            hits_at_3: 0.0,
            hits_at_10: 0.0,
            filtered_mean_rank: 0.0,
            filtered_mrr: 0.0,
            filtered_hits_at_1: 0.0,
            filtered_hits_at_3: 0.0,
            filtered_hits_at_10: 0.0,
            num_test_triples: 0,
        }
    }

    /// Pretty-print metrics in a compact tabular format.
    pub fn display(&self) -> String {
        format!(
            "KGC Evaluation Metrics ({} queries)\n\
             ─────────────────────────────────────────\n\
             Metric          Raw         Filtered\n\
             Mean Rank     {:>10.2}  {:>10.2}\n\
             MRR           {:>10.4}  {:>10.4}\n\
             Hits@1        {:>10.4}  {:>10.4}\n\
             Hits@3        {:>10.4}  {:>10.4}\n\
             Hits@10       {:>10.4}  {:>10.4}",
            self.num_test_triples,
            self.mean_rank,
            self.filtered_mean_rank,
            self.mean_reciprocal_rank,
            self.filtered_mrr,
            self.hits_at_1,
            self.filtered_hits_at_1,
            self.hits_at_3,
            self.filtered_hits_at_3,
            self.hits_at_10,
            self.filtered_hits_at_10,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: compute on known ranks → correct MRR ──────────────────────
    #[test]
    fn test_compute_known_ranks_correct_mrr() {
        // ranks [1, 2, 4]: MRR = (1 + 0.5 + 0.25) / 3 = 1.75/3 ≈ 0.58333
        let ranks = vec![1usize, 2, 4];
        let filtered = vec![1usize, 2, 4]; // same for simplicity
        let m = EvaluationMetrics::compute(&ranks, &filtered);
        let expected_mrr = (1.0 + 0.5 + 0.25) / 3.0;
        assert!(
            (m.mean_reciprocal_rank - expected_mrr).abs() < 1e-12,
            "MRR expected {expected_mrr:.6}, got {:.6}",
            m.mean_reciprocal_rank
        );
    }

    // ── Test 2: MRR = 1.0 when all ranks = 1 ─────────────────────────────
    #[test]
    fn test_mrr_all_rank_one() {
        let ranks = vec![1usize; 10];
        let m = EvaluationMetrics::compute(&ranks, &ranks.clone());
        assert!(
            (m.mean_reciprocal_rank - 1.0).abs() < 1e-12,
            "expected MRR = 1.0, got {}",
            m.mean_reciprocal_rank
        );
    }

    // ── Test 3: MRR = 0.5 when all ranks = 2 ─────────────────────────────
    #[test]
    fn test_mrr_all_rank_two() {
        let ranks = vec![2usize; 6];
        let m = EvaluationMetrics::compute(&ranks, &ranks.clone());
        assert!(
            (m.mean_reciprocal_rank - 0.5).abs() < 1e-12,
            "expected MRR = 0.5, got {}",
            m.mean_reciprocal_rank
        );
    }

    // ── Test 4: Hits@K correctness ────────────────────────────────────────
    #[test]
    fn test_hits_at_k_correctness() {
        let ranks = vec![1usize, 2, 3, 5, 11, 12];
        // Hits@1: rank ≤ 1 → only rank=1 → 1/6
        assert!((hits_at_k(&ranks, 1) - 1.0 / 6.0).abs() < 1e-12);
        // Hits@3: ranks ≤ 3 → 1,2,3 → 3/6
        assert!((hits_at_k(&ranks, 3) - 3.0 / 6.0).abs() < 1e-12);
        // Hits@10: ranks ≤ 10 → 1,2,3,5 → 4/6
        assert!((hits_at_k(&ranks, 10) - 4.0 / 6.0).abs() < 1e-12);
    }

    // ── Test 5: mean_rank helper ──────────────────────────────────────────
    #[test]
    fn test_mean_rank_helper() {
        let ranks = vec![2usize, 4, 6];
        // MR = (2 + 4 + 6) / 3 = 4
        assert!((mean_rank(&ranks) - 4.0).abs() < 1e-12);
    }

    // ── Test 6: empty slice returns 0.0 ───────────────────────────────────
    #[test]
    fn test_empty_slices_return_zero() {
        assert_eq!(hits_at_k(&[], 1), 0.0);
        assert_eq!(mean_rank(&[]), 0.0);
        assert_eq!(mean_reciprocal_rank(&[]), 0.0);
    }

    // ── Test 7: compute preserves num_test_triples ────────────────────────
    #[test]
    fn test_num_test_triples_field() {
        let ranks = vec![1usize, 3, 5, 7];
        let m = EvaluationMetrics::compute(&ranks, &ranks.clone());
        assert_eq!(m.num_test_triples, 4);
    }

    // ── Test 8: filtered_rank skips known positives ────────────────────────
    #[test]
    fn test_compute_filtered_rank_skips_positives() {
        // Scores descending: entity 0 → 10.0, entity 1 → 8.0, entity 2 → 6.0
        // target = 2; entity 1 is a known positive (other than target)
        // raw rank of 2 = 3 (position 3 in sorted list)
        // filtered rank of 2 = 2 (entity 1 skipped)
        let scores = vec![(0usize, 10.0_f64), (1, 8.0), (2, 6.0)];
        let mut known: std::collections::HashSet<usize> = std::collections::HashSet::new();
        known.insert(1);
        let raw = scores.iter().position(|&(e, _)| e == 2).unwrap() + 1;
        let filtered = compute_filtered_rank(&scores, 2, &known);
        assert_eq!(raw, 3, "raw rank should be 3");
        assert_eq!(filtered, 2, "filtered rank should be 2 (entity 1 removed)");
    }

    // ── Test 9: filtered_rank = raw rank when no other positives ──────────
    #[test]
    fn test_filtered_rank_equals_raw_when_no_other_positives() {
        let scores = vec![(0usize, 5.0_f64), (1, 3.0), (2, 1.0)];
        let known: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let raw = scores.iter().position(|&(e, _)| e == 2).unwrap() + 1;
        let filtered = compute_filtered_rank(&scores, 2, &known);
        assert_eq!(raw, filtered, "no other positives → filtered equals raw");
    }

    // ── Test 10: EvaluationMetrics display produces non-empty string ───────
    #[test]
    fn test_display_non_empty() {
        let ranks = vec![1usize, 2, 3];
        let m = EvaluationMetrics::compute(&ranks, &ranks.clone());
        let s = m.display();
        assert!(!s.is_empty(), "display() should produce a non-empty string");
        assert!(s.contains("MRR"), "display string should mention MRR");
    }
}
