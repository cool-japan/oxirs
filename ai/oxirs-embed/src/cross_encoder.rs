//! Cross-encoder re-ranker for embedding search results.
//!
//! Implements a lightweight simulation of cross-encoder scoring using
//! token-overlap (Jaccard) similarity.  In a production system the `score`
//! function would call a transformer model; here it is kept deterministic and
//! dependency-free for testing purposes.

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A (query, document) pair submitted for re-ranking, together with the
/// initial retrieval score produced by an upstream embedding model.
#[derive(Debug, Clone)]
pub struct CandidatePair {
    /// The user query.
    pub query: String,
    /// The candidate document / passage.
    pub document: String,
    /// Score produced by the first-stage retrieval model.
    pub initial_score: f32,
}

/// The outcome of re-ranking a single candidate document.
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// The candidate document text.
    pub document: String,
    /// Score from the first-stage retrieval model.
    pub initial_score: f32,
    /// Score assigned by the cross-encoder.
    pub rerank_score: f32,
    /// 1-based rank in the sorted result list (lower is better).
    pub rank: usize,
}

/// Configuration for a `CrossEncoder` instance.
#[derive(Debug, Clone)]
pub struct CrossEncoderConfig {
    /// Maximum token length (currently advisory; ignored in the simulation).
    pub max_length: usize,
    /// When `true`, scores are min-max normalised to `[0, 1]` before being
    /// returned.
    pub normalize_scores: bool,
    /// How many pairs to process per batch.
    pub batch_size: usize,
}

impl Default for CrossEncoderConfig {
    fn default() -> Self {
        CrossEncoderConfig {
            max_length: 512,
            normalize_scores: false,
            batch_size: 32,
        }
    }
}

/// Stateful cross-encoder that tracks the total number of pairs scored.
pub struct CrossEncoder {
    config: CrossEncoderConfig,
    total_scored: u64,
}

impl CrossEncoder {
    /// Create a new `CrossEncoder` with the given configuration.
    pub fn new(config: CrossEncoderConfig) -> Self {
        CrossEncoder {
            config,
            total_scored: 0,
        }
    }

    /// Score a single `(query, document)` pair using token-overlap similarity.
    pub fn score(&mut self, pair: &CandidatePair) -> f32 {
        self.total_scored += 1;
        token_overlap_score(&pair.query, &pair.document)
    }

    /// Score multiple pairs at once.  Respects `batch_size` but processes
    /// sequentially in the simulation (no concurrency overhead).
    pub fn score_batch(&mut self, pairs: &[CandidatePair]) -> Vec<f32> {
        pairs.iter().map(|p| self.score(p)).collect()
    }

    /// Re-rank `candidates` against `query`.
    ///
    /// Returns a list of [`RerankResult`] sorted by `rerank_score` descending,
    /// with `rank` fields assigned from 1.
    pub fn rerank(
        &mut self,
        query: &str,
        candidates: &[String],
        initial_scores: &[f32],
    ) -> Vec<RerankResult> {
        let n = candidates.len().min(initial_scores.len());
        let pairs: Vec<CandidatePair> = (0..n)
            .map(|i| CandidatePair {
                query: query.to_string(),
                document: candidates[i].clone(),
                initial_score: initial_scores[i],
            })
            .collect();

        let mut raw_scores = self.score_batch(&pairs);

        if self.config.normalize_scores {
            raw_scores = normalize_scores(&raw_scores);
        }

        let mut results: Vec<RerankResult> = (0..n)
            .map(|i| RerankResult {
                document: candidates[i].clone(),
                initial_score: initial_scores[i],
                rerank_score: raw_scores[i],
                rank: 0, // filled below
            })
            .collect();

        // Sort descending by rerank_score.
        results.sort_by(|a, b| {
            b.rerank_score
                .partial_cmp(&a.rerank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign 1-based ranks.
        for (idx, r) in results.iter_mut().enumerate() {
            r.rank = idx + 1;
        }

        results
    }

    /// Like `rerank` but truncates the output to the top-`k` results.
    pub fn top_k(
        &mut self,
        query: &str,
        candidates: &[String],
        initial_scores: &[f32],
        k: usize,
    ) -> Vec<RerankResult> {
        let mut all = self.rerank(query, candidates, initial_scores);
        all.truncate(k);
        all
    }

    /// Total number of individual pairs that have been scored so far.
    pub fn total_scored(&self) -> u64 {
        self.total_scored
    }
}

// ---------------------------------------------------------------------------
// Internal helpers (also exposed for tests)
// ---------------------------------------------------------------------------

/// Jaccard similarity over whitespace-tokenised sets.
///
/// Returns `1.0` for identical strings (even empty ones) and `0.0` for
/// completely disjoint token sets.
pub(crate) fn token_overlap_score(a: &str, b: &str) -> f32 {
    let set_a: HashSet<&str> = a.split_whitespace().collect();
    let set_b: HashSet<&str> = b.split_whitespace().collect();

    if set_a.is_empty() && set_b.is_empty() {
        // Both empty → treat as identical.
        return 1.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Min-max normalise a slice of scores to `[0, 1]`.
///
/// If all scores are equal the output is all-zeros (undefined range).
pub(crate) fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let min = scores.iter().copied().fold(f32::MAX, f32::min);
    let max = scores.iter().copied().fold(f32::MIN, f32::max);

    let range = max - min;
    if range == 0.0 {
        return vec![0.0; scores.len()];
    }
    scores.iter().map(|&s| (s - min) / range).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_encoder() -> CrossEncoder {
        CrossEncoder::new(CrossEncoderConfig::default())
    }

    fn norming_encoder() -> CrossEncoder {
        CrossEncoder::new(CrossEncoderConfig {
            normalize_scores: true,
            ..CrossEncoderConfig::default()
        })
    }

    // --- token_overlap_score ---

    #[test]
    fn test_token_overlap_identical_strings() {
        let score = token_overlap_score("the quick brown fox", "the quick brown fox");
        assert!(
            (score - 1.0).abs() < 1e-6,
            "identical strings should score 1.0"
        );
    }

    #[test]
    fn test_token_overlap_disjoint_strings() {
        let score = token_overlap_score("apple orange", "banana grape");
        assert!(
            (score - 0.0).abs() < 1e-6,
            "disjoint strings should score 0.0"
        );
    }

    #[test]
    fn test_token_overlap_partial_match() {
        let score = token_overlap_score("the fox", "the cat");
        // intersection={the}, union={the,fox,cat} → 1/3
        assert!((score - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_token_overlap_both_empty() {
        let score = token_overlap_score("", "");
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_token_overlap_one_empty() {
        let score = token_overlap_score("hello", "");
        assert!((score - 0.0).abs() < 1e-6);
    }

    // --- normalize_scores ---

    #[test]
    fn test_normalize_scores_range() {
        let scores = vec![0.1f32, 0.5, 0.9];
        let norm = normalize_scores(&scores);
        // All normalised values must lie in [0, 1].
        for &v in &norm {
            assert!(v >= 0.0, "normalised value {v} is below 0");
            assert!(v <= 1.0, "normalised value {v} is above 1");
        }
    }

    #[test]
    fn test_normalize_scores_min_is_zero() {
        let scores = vec![2.0f32, 4.0, 6.0];
        let norm = normalize_scores(&scores);
        assert!((norm[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_scores_max_is_one() {
        let scores = vec![2.0f32, 4.0, 6.0];
        let norm = normalize_scores(&scores);
        assert!((norm[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_scores_all_equal() {
        let scores = vec![3.0f32, 3.0, 3.0];
        let norm = normalize_scores(&scores);
        assert!(norm.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_normalize_scores_empty() {
        let norm = normalize_scores(&[]);
        assert!(norm.is_empty());
    }

    // --- CrossEncoder::score ---

    #[test]
    fn test_score_identical() {
        let mut enc = default_encoder();
        let pair = CandidatePair {
            query: "foo bar".into(),
            document: "foo bar".into(),
            initial_score: 0.9,
        };
        let s = enc.score(&pair);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_disjoint() {
        let mut enc = default_encoder();
        let pair = CandidatePair {
            query: "apple".into(),
            document: "banana".into(),
            initial_score: 0.1,
        };
        let s = enc.score(&pair);
        assert!((s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_increments_total_scored() {
        let mut enc = default_encoder();
        assert_eq!(enc.total_scored(), 0);
        let pair = CandidatePair {
            query: "x".into(),
            document: "y".into(),
            initial_score: 0.0,
        };
        enc.score(&pair);
        assert_eq!(enc.total_scored(), 1);
    }

    // --- CrossEncoder::score_batch ---

    #[test]
    fn test_score_batch_length_matches_input() {
        let mut enc = default_encoder();
        let pairs: Vec<CandidatePair> = (0..5)
            .map(|i| CandidatePair {
                query: format!("query {i}"),
                document: format!("doc {i}"),
                initial_score: 0.5,
            })
            .collect();
        let scores = enc.score_batch(&pairs);
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn test_score_batch_increments_total_scored() {
        let mut enc = default_encoder();
        let pairs: Vec<CandidatePair> = (0..10)
            .map(|i| CandidatePair {
                query: "q".into(),
                document: format!("d {i}"),
                initial_score: 0.0,
            })
            .collect();
        enc.score_batch(&pairs);
        assert_eq!(enc.total_scored(), 10);
    }

    // --- CrossEncoder::rerank ---

    #[test]
    fn test_rerank_sorted_descending() {
        let mut enc = default_encoder();
        let candidates = vec![
            "apple".to_string(),
            "apple banana".to_string(),
            "apple banana cherry".to_string(),
        ];
        let query = "apple banana cherry";
        let initial = vec![0.3, 0.6, 0.9];
        let results = enc.rerank(query, &candidates, &initial);
        // Results should be sorted by rerank_score descending.
        for w in results.windows(2) {
            assert!(w[0].rerank_score >= w[1].rerank_score);
        }
    }

    #[test]
    fn test_rerank_rank_field_correct() {
        let mut enc = default_encoder();
        let candidates = vec!["a b c".to_string(), "x y z".to_string()];
        let results = enc.rerank("a b c", &candidates, &[0.5, 0.5]);
        assert_eq!(results[0].rank, 1);
        assert_eq!(results[1].rank, 2);
    }

    #[test]
    fn test_rerank_empty_candidates() {
        let mut enc = default_encoder();
        let results = enc.rerank("query", &[], &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rerank_total_scored_increments() {
        let mut enc = default_encoder();
        let docs: Vec<String> = (0..3).map(|i| format!("doc {i}")).collect();
        let scores: Vec<f32> = (0..3).map(|i| i as f32 * 0.1).collect();
        enc.rerank("q", &docs, &scores);
        assert_eq!(enc.total_scored(), 3);
    }

    // --- CrossEncoder::top_k ---

    #[test]
    fn test_top_k_limits_output() {
        let mut enc = default_encoder();
        let docs: Vec<String> = (0..10).map(|i| format!("word{i} text")).collect();
        let initial: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let results = enc.top_k("word5 text", &docs, &initial, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_top_k_returns_all_when_k_exceeds_count() {
        let mut enc = default_encoder();
        let docs = vec!["a".to_string(), "b".to_string()];
        let results = enc.top_k("a", &docs, &[0.5, 0.2], 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_top_k_rank_starts_at_one() {
        let mut enc = default_encoder();
        let docs = vec!["hello world".to_string(), "foo bar".to_string()];
        let results = enc.top_k("hello world", &docs, &[0.5, 0.5], 2);
        assert_eq!(results[0].rank, 1);
    }

    // --- normalize_scores integration ---

    #[test]
    fn test_rerank_with_normalisation_range() {
        let mut enc = norming_encoder();
        let docs = vec!["a b".to_string(), "c d".to_string(), "e f".to_string()];
        let initial = vec![0.1, 0.5, 0.9];
        let results = enc.rerank("a b", &docs, &initial);
        for r in &results {
            assert!(r.rerank_score >= 0.0 && r.rerank_score <= 1.0);
        }
    }
}
