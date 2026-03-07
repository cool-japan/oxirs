//! Result reranking for retrieval pipelines.
//!
//! Provides [`Reranker`]: a flexible reranking engine that supports
//! cross-encoder scoring simulation, BM25 reranking, reciprocal rank fusion,
//! score normalisation, top-k selection, threshold filtering, and batch
//! reranking with rich statistics.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_embed::reranker::{Reranker, RerankerConfig, RerankMethod, Document};
//!
//! let config = RerankerConfig {
//!     method: RerankMethod::Bm25,
//!     top_k: 3,
//!     score_threshold: Some(0.0),
//!     normalize_scores: true,
//! };
//! let reranker = Reranker::new(config);
//!
//! let docs = vec![
//!     Document { id: "d1".into(), text: "Rust programming language".into(), initial_score: 0.6 },
//!     Document { id: "d2".into(), text: "Rust toolchain cargo".into(), initial_score: 0.4 },
//! ];
//! let results = reranker.rerank("cargo build", &docs);
//! assert!(!results.is_empty());
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A document candidate to be reranked.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    /// Unique identifier.
    pub id: String,
    /// Full text of the document.
    pub text: String,
    /// Score assigned by the initial retriever (e.g. cosine similarity).
    pub initial_score: f64,
}

/// A ranked result produced by the reranker.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedResult {
    /// Document identifier.
    pub id: String,
    /// Reranking score (semantic of the value depends on the chosen method).
    pub score: f64,
    /// 1-based rank position in the final result set.
    pub rank: usize,
    /// Score delta compared with initial retriever score.
    pub rank_shift: f64,
}

/// Reranking method to apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RerankMethod {
    /// Simulated cross-encoder scoring: score(query, doc) via token overlap
    /// and length normalisation.
    CrossEncoder,
    /// BM25 reranking using TF×IDF approximation.
    Bm25,
    /// Reciprocal Rank Fusion — combines the initial retriever rank with a
    /// secondary BM25 ranking.
    ReciprocalRankFusion,
}

/// Configuration for the [`Reranker`].
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Reranking algorithm to use.
    pub method: RerankMethod,
    /// Maximum number of results to return after reranking.
    pub top_k: usize,
    /// Minimum score to keep a result (applied after normalisation if enabled).
    /// `None` means no threshold.
    pub score_threshold: Option<f64>,
    /// When `true`, final scores are min-max normalised to `[0, 1]`.
    pub normalize_scores: bool,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            method: RerankMethod::Bm25,
            top_k: 10,
            score_threshold: None,
            normalize_scores: false,
        }
    }
}

/// Score distribution statistics over a reranked set.
#[derive(Debug, Clone)]
pub struct RerankStats {
    /// Number of documents in the reranked list.
    pub count: usize,
    /// Minimum score in the result set.
    pub min_score: f64,
    /// Maximum score in the result set.
    pub max_score: f64,
    /// Arithmetic mean of scores.
    pub mean_score: f64,
    /// Standard deviation of scores.
    pub std_dev: f64,
    /// Mean absolute rank shift relative to initial order.
    pub mean_rank_shift: f64,
}

/// Batch input for reranking multiple queries at once.
#[derive(Debug, Clone)]
pub struct BatchRerankInput {
    /// Query string.
    pub query: String,
    /// Candidate documents for this query.
    pub documents: Vec<Document>,
}

/// Batch output for a single query's reranked results.
#[derive(Debug, Clone)]
pub struct BatchRerankOutput {
    /// The original query string.
    pub query: String,
    /// Ranked results for this query.
    pub results: Vec<RankedResult>,
    /// Statistics over the results.
    pub stats: RerankStats,
}

// ─────────────────────────────────────────────────────────────────────────────
// BM25 helpers
// ─────────────────────────────────────────────────────────────────────────────

/// BM25 tuning parameters.
const BM25_K1: f64 = 1.5;
const BM25_B: f64 = 0.75;

/// Tokenise text into lowercase terms.
fn tokenise(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Build term-frequency map for a token list.
fn term_freq(tokens: &[String]) -> HashMap<String, usize> {
    let mut tf = HashMap::new();
    for t in tokens {
        *tf.entry(t.clone()).or_insert(0) += 1;
    }
    tf
}

/// Compute IDF for a term given document frequency and corpus size.
/// Uses the classic Okapi BM25 IDF formula.
fn idf(doc_freq: usize, num_docs: usize) -> f64 {
    let n = num_docs as f64;
    let df = doc_freq as f64;
    ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
}

/// Score a single document against a query using BM25.
fn bm25_score(
    query_terms: &[String],
    doc_tokens: &[String],
    df_map: &HashMap<String, usize>,
    num_docs: usize,
    avg_dl: f64,
) -> f64 {
    let tf_map = term_freq(doc_tokens);
    let dl = doc_tokens.len() as f64;
    let mut score = 0.0_f64;
    for term in query_terms {
        let tf = *tf_map.get(term).unwrap_or(&0) as f64;
        if tf == 0.0 {
            continue;
        }
        let df = *df_map.get(term).unwrap_or(&0);
        let idf_val = idf(df, num_docs);
        let numerator = tf * (BM25_K1 + 1.0);
        let denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avg_dl.max(1.0));
        score += idf_val * numerator / denominator;
    }
    score
}

// ─────────────────────────────────────────────────────────────────────────────
// Score normalisation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Min-max normalise a slice of scores to `[0, 1]`.
fn min_max_normalize(scores: &[f64]) -> Vec<f64> {
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < f64::EPSILON {
        return vec![1.0; scores.len()];
    }
    scores.iter().map(|s| (s - min) / range).collect()
}

/// Z-score normalise a slice of scores (mean 0, std 1).
pub fn z_score_normalize(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    if std < f64::EPSILON {
        return vec![0.0; scores.len()];
    }
    scores.iter().map(|s| (s - mean) / std).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Reranker
// ─────────────────────────────────────────────────────────────────────────────

/// Reranker supporting cross-encoder simulation, BM25, and reciprocal rank
/// fusion.
pub struct Reranker {
    config: RerankerConfig,
}

impl Reranker {
    /// Create a new reranker with the given configuration.
    pub fn new(config: RerankerConfig) -> Self {
        Self { config }
    }

    /// Create a reranker with the default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RerankerConfig::default())
    }

    /// Rerank `docs` for `query` and return top-k [`RankedResult`]s.
    pub fn rerank(&self, query: &str, docs: &[Document]) -> Vec<RankedResult> {
        if docs.is_empty() {
            return Vec::new();
        }
        let scores = match self.config.method {
            RerankMethod::CrossEncoder => self.cross_encoder_scores(query, docs),
            RerankMethod::Bm25 => self.bm25_scores(query, docs),
            RerankMethod::ReciprocalRankFusion => self.rrf_scores(query, docs),
        };

        self.finalize(docs, scores)
    }

    /// Rerank multiple (query, docs) pairs in a single call.
    pub fn rerank_batch(&self, inputs: &[BatchRerankInput]) -> Vec<BatchRerankOutput> {
        inputs
            .iter()
            .map(|input| {
                let results = self.rerank(&input.query, &input.documents);
                let stats = self.compute_stats(&results);
                BatchRerankOutput {
                    query: input.query.clone(),
                    results,
                    stats,
                }
            })
            .collect()
    }

    /// Return statistics for the given ranked results.
    pub fn compute_stats(&self, results: &[RankedResult]) -> RerankStats {
        if results.is_empty() {
            return RerankStats {
                count: 0,
                min_score: 0.0,
                max_score: 0.0,
                mean_score: 0.0,
                std_dev: 0.0,
                mean_rank_shift: 0.0,
            };
        }
        let n = results.len() as f64;
        let scores: Vec<f64> = results.iter().map(|r| r.score).collect();
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_score = scores.iter().sum::<f64>() / n;
        let variance = scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let mean_rank_shift = results.iter().map(|r| r.rank_shift.abs()).sum::<f64>() / n;
        RerankStats {
            count: results.len(),
            min_score,
            max_score,
            mean_score,
            std_dev,
            mean_rank_shift,
        }
    }

    // ── Private scoring methods ──────────────────────────────────────────────

    /// Simulated cross-encoder: token-overlap score normalised by query length.
    fn cross_encoder_scores(&self, query: &str, docs: &[Document]) -> Vec<f64> {
        let query_tokens: Vec<String> = tokenise(query);
        let q_set: std::collections::HashSet<String> = query_tokens.iter().cloned().collect();
        docs.iter()
            .map(|doc| {
                let doc_tokens = tokenise(&doc.text);
                if q_set.is_empty() || doc_tokens.is_empty() {
                    return 0.0;
                }
                let matches = doc_tokens.iter().filter(|t| q_set.contains(*t)).count();
                let tf_norm = matches as f64 / doc_tokens.len() as f64;
                let idf_weight = (matches as f64 + 1.0).ln() / (q_set.len() as f64 + 1.0).ln();
                // Blend with initial score for cross-encoder flavour
                0.6 * tf_norm + 0.2 * idf_weight + 0.2 * doc.initial_score
            })
            .collect()
    }

    /// BM25 scores over the document corpus.
    fn bm25_scores(&self, query: &str, docs: &[Document]) -> Vec<f64> {
        let query_terms = tokenise(query);
        let tokenised: Vec<Vec<String>> = docs.iter().map(|d| tokenise(&d.text)).collect();
        let num_docs = docs.len();
        let total_len: usize = tokenised.iter().map(|t| t.len()).sum();
        let avg_dl = total_len as f64 / num_docs as f64;

        // Build document-frequency map
        let mut df_map: HashMap<String, usize> = HashMap::new();
        for toks in &tokenised {
            let unique: std::collections::HashSet<&String> = toks.iter().collect();
            for t in unique {
                *df_map.entry(t.clone()).or_insert(0) += 1;
            }
        }

        tokenised
            .iter()
            .map(|toks| bm25_score(&query_terms, toks, &df_map, num_docs, avg_dl))
            .collect()
    }

    /// Reciprocal Rank Fusion: combines initial retriever rank + BM25 rank.
    ///
    /// RRF formula: `score = Σ 1 / (k + rank_i)` where `k = 60` (standard).
    fn rrf_scores(&self, query: &str, docs: &[Document]) -> Vec<f64> {
        const K: f64 = 60.0;

        // Rank by initial score (descending)
        let n = docs.len();
        let mut initial_order: Vec<usize> = (0..n).collect();
        initial_order.sort_by(|&a, &b| {
            docs[b]
                .initial_score
                .partial_cmp(&docs[a].initial_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut rank_initial = vec![0usize; n];
        for (rank, &idx) in initial_order.iter().enumerate() {
            rank_initial[idx] = rank + 1;
        }

        // Rank by BM25 score (descending)
        let bm25 = self.bm25_scores(query, docs);
        let mut bm25_order: Vec<usize> = (0..n).collect();
        bm25_order.sort_by(|&a, &b| {
            bm25[b]
                .partial_cmp(&bm25[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut rank_bm25 = vec![0usize; n];
        for (rank, &idx) in bm25_order.iter().enumerate() {
            rank_bm25[idx] = rank + 1;
        }

        (0..n)
            .map(|i| 1.0 / (K + rank_initial[i] as f64) + 1.0 / (K + rank_bm25[i] as f64))
            .collect()
    }

    /// Apply normalisation, threshold filtering, and top-k selection.
    fn finalize(&self, docs: &[Document], raw_scores: Vec<f64>) -> Vec<RankedResult> {
        // Build initial ranks from initial_score ordering
        let n = docs.len();
        let mut initial_order: Vec<usize> = (0..n).collect();
        initial_order.sort_by(|&a, &b| {
            docs[b]
                .initial_score
                .partial_cmp(&docs[a].initial_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut initial_rank = vec![0usize; n];
        for (rank, &idx) in initial_order.iter().enumerate() {
            initial_rank[idx] = rank + 1;
        }

        // Optionally normalise
        let final_scores = if self.config.normalize_scores {
            min_max_normalize(&raw_scores)
        } else {
            raw_scores.clone()
        };

        // Sort descending by reranked score
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            final_scores[b]
                .partial_cmp(&final_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut results: Vec<RankedResult> = order
            .iter()
            .enumerate()
            .map(|(new_rank, &idx)| {
                let rank_shift = initial_rank[idx] as f64 - (new_rank + 1) as f64;
                RankedResult {
                    id: docs[idx].id.clone(),
                    score: final_scores[idx],
                    rank: new_rank + 1,
                    rank_shift,
                }
            })
            .collect();

        // Apply score threshold
        if let Some(threshold) = self.config.score_threshold {
            results.retain(|r| r.score >= threshold);
        }

        // Apply top-k
        results.truncate(self.config.top_k);

        // Re-assign rank after truncation
        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i + 1;
        }

        results
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_docs() -> Vec<Document> {
        vec![
            Document {
                id: "d1".into(),
                text: "Rust is a systems programming language".into(),
                initial_score: 0.9,
            },
            Document {
                id: "d2".into(),
                text: "Python is a high-level scripting language".into(),
                initial_score: 0.7,
            },
            Document {
                id: "d3".into(),
                text: "Cargo is the Rust package manager and build tool".into(),
                initial_score: 0.5,
            },
            Document {
                id: "d4".into(),
                text: "JavaScript runs in the browser".into(),
                initial_score: 0.3,
            },
            Document {
                id: "d5".into(),
                text: "Rust ownership model ensures memory safety".into(),
                initial_score: 0.6,
            },
        ]
    }

    // ── Cross-encoder ────────────────────────────────────────────────────────

    #[test]
    fn test_cross_encoder_rerank_returns_results() {
        let config = RerankerConfig {
            method: RerankMethod::CrossEncoder,
            top_k: 3,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust systems language", &sample_docs());
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_cross_encoder_ranks_rust_docs_higher() {
        let config = RerankerConfig {
            method: RerankMethod::CrossEncoder,
            top_k: 5,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust programming", &sample_docs());
        // First result should be a Rust-related document
        assert!(results[0].id == "d1" || results[0].id == "d3" || results[0].id == "d5");
    }

    #[test]
    fn test_cross_encoder_rank_order() {
        let config = RerankerConfig {
            method: RerankMethod::CrossEncoder,
            top_k: 5,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust", &sample_docs());
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.rank, i + 1);
        }
    }

    #[test]
    fn test_cross_encoder_scores_non_negative() {
        let reranker = Reranker::with_defaults();
        let results = reranker.rerank("cargo build", &sample_docs());
        for r in &results {
            assert!(r.score >= 0.0);
        }
    }

    // ── BM25 ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_bm25_rerank_basic() {
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 5,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust ownership memory", &sample_docs());
        assert!(!results.is_empty());
    }

    #[test]
    fn test_bm25_top_k_respected() {
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 2,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("language programming", &sample_docs());
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_bm25_term_frequency_effect() {
        // A document containing the query term more often should score higher
        let docs = vec![
            Document {
                id: "rare".into(),
                text: "Rust is a language".into(),
                initial_score: 0.5,
            },
            Document {
                id: "frequent".into(),
                text: "Rust Rust Rust Rust Rust performance systems Rust".into(),
                initial_score: 0.5,
            },
        ];
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 2,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust", &docs);
        // The "frequent" document should rank first
        assert_eq!(results[0].id, "frequent");
    }

    #[test]
    fn test_bm25_scores_are_non_negative() {
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 5,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("systems", &sample_docs());
        for r in &results {
            assert!(r.score >= 0.0);
        }
    }

    // ── Reciprocal Rank Fusion ───────────────────────────────────────────────

    #[test]
    fn test_rrf_rerank_returns_results() {
        let config = RerankerConfig {
            method: RerankMethod::ReciprocalRankFusion,
            top_k: 5,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust cargo build", &sample_docs());
        assert!(!results.is_empty());
    }

    #[test]
    fn test_rrf_scores_positive() {
        let config = RerankerConfig {
            method: RerankMethod::ReciprocalRankFusion,
            top_k: 5,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("language", &sample_docs());
        for r in &results {
            assert!(r.score > 0.0);
        }
    }

    #[test]
    fn test_rrf_top_k_applied() {
        let config = RerankerConfig {
            method: RerankMethod::ReciprocalRankFusion,
            top_k: 2,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("language", &sample_docs());
        assert!(results.len() <= 2);
    }

    // ── Score normalisation ──────────────────────────────────────────────────

    #[test]
    fn test_min_max_normalize_range() {
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 5,
            normalize_scores: true,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust", &sample_docs());
        for r in &results {
            assert!(r.score >= 0.0 && r.score <= 1.0 + 1e-10);
        }
    }

    #[test]
    fn test_z_score_normalize_identity_for_equal_values() {
        let scores = vec![5.0, 5.0, 5.0];
        let normalized = z_score_normalize(&scores);
        for v in normalized {
            assert!((v - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_z_score_normalize_basic() {
        let scores = vec![1.0, 2.0, 3.0];
        let normalized = z_score_normalize(&scores);
        assert_eq!(normalized.len(), 3);
        // Mean of normalised should be ~0
        let mean: f64 = normalized.iter().sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_min_max_normalize_empty() {
        let scores: Vec<f64> = vec![];
        let normalized = min_max_normalize(&scores);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_min_max_normalize_single_value() {
        let scores = vec![3.7];
        let normalized = min_max_normalize(&scores);
        assert_eq!(normalized.len(), 1);
        assert!((normalized[0] - 1.0).abs() < 1e-10);
    }

    // ── Score threshold ──────────────────────────────────────────────────────

    #[test]
    fn test_score_threshold_filters_low_scores() {
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 10,
            normalize_scores: true,
            score_threshold: Some(0.5),
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust", &sample_docs());
        for r in &results {
            assert!(r.score >= 0.5);
        }
    }

    #[test]
    fn test_score_threshold_zero_keeps_all_non_negative() {
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 10,
            score_threshold: Some(0.0),
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust", &sample_docs());
        for r in &results {
            assert!(r.score >= 0.0);
        }
    }

    // ── Top-k ────────────────────────────────────────────────────────────────

    #[test]
    fn test_top_k_one() {
        let config = RerankerConfig {
            method: RerankMethod::CrossEncoder,
            top_k: 1,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust", &sample_docs());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rank, 1);
    }

    #[test]
    fn test_top_k_larger_than_docs() {
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 100,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust", &sample_docs());
        assert!(results.len() <= sample_docs().len());
    }

    // ── Rank shift ───────────────────────────────────────────────────────────

    #[test]
    fn test_rank_shift_computed() {
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 5,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("ownership memory", &sample_docs());
        // rank_shift values must be finite
        for r in &results {
            assert!(r.rank_shift.is_finite());
        }
    }

    // ── Empty input ──────────────────────────────────────────────────────────

    #[test]
    fn test_empty_docs_returns_empty() {
        let reranker = Reranker::with_defaults();
        let results = reranker.rerank("Rust", &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_query_bm25() {
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 5,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("", &sample_docs());
        // All BM25 scores will be 0 — we still get (up to) 5 results
        assert!(results.len() <= 5);
    }

    // ── Batch reranking ──────────────────────────────────────────────────────

    #[test]
    fn test_batch_rerank_multiple_queries() {
        let reranker = Reranker::with_defaults();
        let inputs = vec![
            BatchRerankInput {
                query: "Rust".into(),
                documents: sample_docs(),
            },
            BatchRerankInput {
                query: "Python".into(),
                documents: sample_docs(),
            },
        ];
        let outputs = reranker.rerank_batch(&inputs);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].query, "Rust");
        assert_eq!(outputs[1].query, "Python");
    }

    #[test]
    fn test_batch_rerank_stats_populated() {
        let reranker = Reranker::with_defaults();
        let inputs = vec![BatchRerankInput {
            query: "Rust".into(),
            documents: sample_docs(),
        }];
        let outputs = reranker.rerank_batch(&inputs);
        let stats = &outputs[0].stats;
        assert!(stats.count > 0);
        assert!(stats.max_score >= stats.min_score);
    }

    #[test]
    fn test_batch_rerank_empty_inputs() {
        let reranker = Reranker::with_defaults();
        let outputs = reranker.rerank_batch(&[]);
        assert!(outputs.is_empty());
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    #[test]
    fn test_compute_stats_empty_results() {
        let reranker = Reranker::with_defaults();
        let stats = reranker.compute_stats(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean_score, 0.0);
    }

    #[test]
    fn test_compute_stats_single_result() {
        let reranker = Reranker::with_defaults();
        let results = vec![RankedResult {
            id: "d1".into(),
            score: 0.75,
            rank: 1,
            rank_shift: 0.0,
        }];
        let stats = reranker.compute_stats(&results);
        assert_eq!(stats.count, 1);
        assert!((stats.min_score - 0.75).abs() < 1e-10);
        assert!((stats.max_score - 0.75).abs() < 1e-10);
        assert!((stats.mean_score - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_std_dev() {
        let reranker = Reranker::with_defaults();
        let results = vec![
            RankedResult {
                id: "a".into(),
                score: 1.0,
                rank: 1,
                rank_shift: 0.0,
            },
            RankedResult {
                id: "b".into(),
                score: 3.0,
                rank: 2,
                rank_shift: 0.0,
            },
        ];
        let stats = reranker.compute_stats(&results);
        assert!((stats.std_dev - 1.0).abs() < 1e-10);
    }

    // ── Config defaults ──────────────────────────────────────────────────────

    #[test]
    fn test_reranker_config_default() {
        let cfg = RerankerConfig::default();
        assert_eq!(cfg.method, RerankMethod::Bm25);
        assert_eq!(cfg.top_k, 10);
        assert!(cfg.score_threshold.is_none());
        assert!(!cfg.normalize_scores);
    }

    #[test]
    fn test_tokenise_lowercases_and_strips_punct() {
        let tokens = tokenise("Hello, World! Rust.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"rust".to_string()));
    }

    #[test]
    fn test_z_score_normalize_empty() {
        let result = z_score_normalize(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_idf_formula() {
        // When df == 1 and num_docs == 10: idf = ln((10-1+0.5)/(1+0.5)+1) = ln(7.33) ≈ 1.99
        let v = idf(1, 10);
        assert!(v > 0.0);
    }

    #[test]
    fn test_rerank_rank_contiguous() {
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 5,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust", &sample_docs());
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i + 1);
        }
    }

    #[test]
    fn test_cross_encoder_single_doc() {
        let docs = vec![Document {
            id: "only".into(),
            text: "Rust is great".into(),
            initial_score: 0.8,
        }];
        let config = RerankerConfig {
            method: RerankMethod::CrossEncoder,
            top_k: 1,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust", &docs);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "only");
    }

    #[test]
    fn test_rrf_single_doc() {
        let docs = vec![Document {
            id: "only".into(),
            text: "unique content here".into(),
            initial_score: 1.0,
        }];
        let config = RerankerConfig {
            method: RerankMethod::ReciprocalRankFusion,
            top_k: 1,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("content", &docs);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_term_freq_counts_correctly() {
        let tokens = tokenise("rust rust cargo");
        let tf = term_freq(&tokens);
        assert_eq!(*tf.get("rust").unwrap_or(&0), 2);
        assert_eq!(*tf.get("cargo").unwrap_or(&0), 1);
    }

    #[test]
    fn test_document_clone() {
        let doc = Document {
            id: "d1".into(),
            text: "Rust language".into(),
            initial_score: 0.9,
        };
        let cloned = doc.clone();
        assert_eq!(cloned.id, "d1");
        assert!((cloned.initial_score - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_ranked_result_fields() {
        let r = RankedResult {
            id: "x".into(),
            score: 0.5,
            rank: 2,
            rank_shift: -1.0,
        };
        assert_eq!(r.id, "x");
        assert_eq!(r.rank, 2);
        assert!((r.rank_shift + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_rerank_stats_count_matches_results() {
        let reranker = Reranker::with_defaults();
        let inputs = vec![BatchRerankInput {
            query: "language".into(),
            documents: sample_docs(),
        }];
        let outputs = reranker.rerank_batch(&inputs);
        assert_eq!(outputs[0].stats.count, outputs[0].results.len());
    }

    #[test]
    fn test_rerank_descending_score_order() {
        let config = RerankerConfig {
            method: RerankMethod::Bm25,
            top_k: 5,
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let results = reranker.rerank("Rust", &sample_docs());
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score - 1e-10);
        }
    }
}
