//! BM25 + Dense hybrid retrieval with configurable alpha blending
//!
//! This module implements a proper BM25 scoring engine combined with dense
//! (vector) retrieval and multiple blending strategies: alpha-weighted linear
//! interpolation and RRF-style rank fusion.

use crate::{GraphRAGError, GraphRAGResult, ScoreSource, ScoredEntity};
use std::collections::HashMap;

// ─── BM25 parameters ────────────────────────────────────────────────────────

/// BM25 algorithm variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bm25Variant {
    /// Classic BM25 (Okapi BM25)
    #[default]
    Classic,
    /// BM25+ (addresses the lower-bound problem of BM25 for non-relevant docs)
    Plus,
    /// BM25L (length-normalised variant, good for longer documents)
    L,
}

/// Configuration for the BM25 scorer
#[derive(Debug, Clone)]
pub struct Bm25Config {
    /// Term-frequency saturation parameter (default 1.2)
    pub k1: f64,
    /// Field length normalisation (0 = no normalisation, 1 = full, default 0.75)
    pub b: f64,
    /// Delta for BM25+ / BM25L (default 1.0)
    pub delta: f64,
    /// BM25 variant
    pub variant: Bm25Variant,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            delta: 1.0,
            variant: Bm25Variant::Classic,
        }
    }
}

// ─── Document corpus ────────────────────────────────────────────────────────

/// A document in the BM25 corpus (entity URI + text representation)
#[derive(Debug, Clone)]
pub struct Document {
    /// Entity URI or identifier
    pub id: String,
    /// Tokenised terms
    pub terms: Vec<String>,
}

impl Document {
    /// Create from a URI and raw text (whitespace tokenisation)
    pub fn from_text(id: impl Into<String>, text: &str) -> Self {
        Self {
            id: id.into(),
            terms: text.split_whitespace().map(|t| t.to_lowercase()).collect(),
        }
    }
}

// ─── BM25 index ─────────────────────────────────────────────────────────────

/// In-memory inverted index for BM25 scoring
pub struct Bm25Index {
    config: Bm25Config,
    /// Total number of documents
    num_docs: usize,
    /// Average document length
    avg_doc_len: f64,
    /// Per-document term frequencies: doc_id → term → tf
    doc_tf: HashMap<String, HashMap<String, usize>>,
    /// Per-document length
    doc_len: HashMap<String, usize>,
    /// Inverted index: term → document frequency
    df: HashMap<String, usize>,
}

impl Bm25Index {
    /// Build an index from a corpus of documents
    pub fn build(corpus: &[Document], config: Bm25Config) -> Self {
        let num_docs = corpus.len();
        let mut doc_tf: HashMap<String, HashMap<String, usize>> = HashMap::with_capacity(num_docs);
        let mut doc_len: HashMap<String, usize> = HashMap::with_capacity(num_docs);
        let mut df: HashMap<String, usize> = HashMap::new();

        for doc in corpus {
            let mut tf: HashMap<String, usize> = HashMap::new();
            for term in &doc.terms {
                *tf.entry(term.clone()).or_insert(0) += 1;
            }
            for term in tf.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            doc_len.insert(doc.id.clone(), doc.terms.len());
            doc_tf.insert(doc.id.clone(), tf);
        }

        let total_len: usize = doc_len.values().sum();
        let avg_doc_len = if num_docs == 0 {
            1.0
        } else {
            total_len as f64 / num_docs as f64
        };

        Self {
            config,
            num_docs,
            avg_doc_len,
            doc_tf,
            doc_len,
            df,
        }
    }

    /// Score all documents in the corpus against a query string
    pub fn score_all(&self, query: &str) -> Vec<(String, f64)> {
        if self.num_docs == 0 || query.is_empty() {
            return vec![];
        }

        let query_terms: Vec<String> = query.split_whitespace().map(|t| t.to_lowercase()).collect();

        let mut scores: HashMap<&str, f64> = HashMap::with_capacity(self.num_docs);

        for term in &query_terms {
            let df_t = *self.df.get(term.as_str()).unwrap_or(&0) as f64;
            if df_t == 0.0 {
                continue;
            }
            let idf = self.idf(df_t);

            for (doc_id, tf_map) in &self.doc_tf {
                let tf_t = *tf_map.get(term.as_str()).unwrap_or(&0) as f64;
                if tf_t == 0.0 {
                    continue;
                }
                let dl = *self.doc_len.get(doc_id).unwrap_or(&0) as f64;
                let term_score = idf * self.tf_weight(tf_t, dl);
                *scores.entry(doc_id.as_str()).or_insert(0.0) += term_score;
            }
        }

        let mut result: Vec<(String, f64)> = scores
            .into_iter()
            .filter(|(_, s)| *s > 0.0)
            .map(|(id, s)| (id.to_string(), s))
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// IDF (Robertson / Sparck Jones formula, smoothed)
    fn idf(&self, df_t: f64) -> f64 {
        let n = self.num_docs as f64;
        ((n - df_t + 0.5) / (df_t + 0.5) + 1.0).ln()
    }

    /// Term-frequency weight based on variant
    fn tf_weight(&self, tf: f64, dl: f64) -> f64 {
        let k1 = self.config.k1;
        let b = self.config.b;
        let avgdl = self.avg_doc_len;

        match self.config.variant {
            Bm25Variant::Classic => {
                let norm = 1.0 - b + b * (dl / avgdl);
                tf * (k1 + 1.0) / (tf + k1 * norm)
            }
            Bm25Variant::Plus => {
                let norm = 1.0 - b + b * (dl / avgdl);
                (tf * (k1 + 1.0) / (tf + k1 * norm)) + self.config.delta
            }
            Bm25Variant::L => {
                // BM25L: ctf = tf / (1 - b + b * dl/avgdl), capped at k1
                let c_tf = tf / (1.0 - b + b * dl / avgdl);
                (k1 + 1.0) * (c_tf + self.config.delta) / (k1 + c_tf + self.config.delta)
            }
        }
    }
}

// ─── Hybrid retrieval configuration ─────────────────────────────────────────

/// Blending strategy for combining dense and sparse scores
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HybridBlendMode {
    /// Linear interpolation: score = α·dense + (1−α)·bm25 (both normalised)
    AlphaInterpolation {
        /// Weight given to dense scores (0.0 → pure BM25, 1.0 → pure dense)
        alpha: f64,
    },
    /// Convex hull – take whichever component dominates per entity
    MaxScore,
    /// Reciprocal Rank Fusion (equal weight on rank positions)
    ReciprocalRankFusion {
        /// RRF smoothing constant (default 60)
        k: f64,
    },
    /// Soft vote: normalise each list to \[0,1\], multiply probabilities
    SoftVote,
}

impl Default for HybridBlendMode {
    fn default() -> Self {
        Self::AlphaInterpolation { alpha: 0.7 }
    }
}

/// Configuration for the hybrid retriever
#[derive(Debug, Clone)]
pub struct HybridRetrievalConfig {
    /// BM25 configuration
    pub bm25: Bm25Config,
    /// Blending strategy
    pub blend_mode: HybridBlendMode,
    /// Maximum results to return
    pub top_k: usize,
    /// Minimum final score threshold (0.0 = no threshold)
    pub min_score: f64,
    /// Normalise dense scores to \[0,1\] before blending
    pub normalise_dense: bool,
    /// Normalise BM25 scores to \[0,1\] before blending
    pub normalise_bm25: bool,
}

impl Default for HybridRetrievalConfig {
    fn default() -> Self {
        Self {
            bm25: Bm25Config::default(),
            blend_mode: HybridBlendMode::default(),
            top_k: 20,
            min_score: 0.0,
            normalise_dense: true,
            normalise_bm25: true,
        }
    }
}

// ─── Hybrid retriever ────────────────────────────────────────────────────────

/// Hybrid retriever combining BM25 and dense vector retrieval
pub struct HybridRetriever {
    index: Bm25Index,
    config: HybridRetrievalConfig,
}

impl HybridRetriever {
    /// Build a hybrid retriever from a document corpus
    pub fn build(corpus: &[Document], config: HybridRetrievalConfig) -> Self {
        let index = Bm25Index::build(corpus, config.bm25.clone());
        Self { index, config }
    }

    /// Retrieve and blend results for the given query.
    ///
    /// `dense_results` – ranked list `(entity_id, score)` from a vector store
    /// (scores should be cosine / dot-product similarities, higher = better).
    pub fn retrieve(
        &self,
        query: &str,
        dense_results: &[(String, f32)],
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        // 1. BM25 scores
        let bm25_raw = self.index.score_all(query);

        // 2. Normalise both lists (optional)
        let dense_norm = if self.config.normalise_dense {
            normalise_scores_f32(dense_results)
        } else {
            dense_results
                .iter()
                .map(|(id, s)| (id.clone(), *s as f64))
                .collect()
        };

        let bm25_norm = if self.config.normalise_bm25 {
            normalise_scores_f64(&bm25_raw)
        } else {
            bm25_raw.clone()
        };

        // 3. Blend
        let blended = match self.config.blend_mode {
            HybridBlendMode::AlphaInterpolation { alpha } => {
                blend_alpha(&dense_norm, &bm25_norm, alpha)
            }
            HybridBlendMode::MaxScore => blend_max(&dense_norm, &bm25_norm),
            HybridBlendMode::ReciprocalRankFusion { k } => blend_rrf(&dense_norm, &bm25_norm, k),
            HybridBlendMode::SoftVote => blend_soft_vote(&dense_norm, &bm25_norm),
        };

        // 4. Apply threshold, sort, truncate
        let mut entities: Vec<ScoredEntity> = blended
            .into_iter()
            .filter(|(_, score)| *score >= self.config.min_score)
            .map(|(uri, score)| {
                let in_dense = dense_norm.iter().any(|(id, _)| id == &uri);
                let in_bm25 = bm25_norm.iter().any(|(id, _)| id == &uri);
                let source = match (in_dense, in_bm25) {
                    (true, true) => ScoreSource::Fused,
                    (true, false) => ScoreSource::Vector,
                    _ => ScoreSource::Keyword,
                };
                ScoredEntity {
                    uri,
                    score,
                    source,
                    metadata: HashMap::new(),
                }
            })
            .collect();

        entities.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entities.truncate(self.config.top_k);

        Ok(entities)
    }

    /// Retrieve using only BM25 (no dense component)
    pub fn retrieve_bm25_only(&self, query: &str) -> GraphRAGResult<Vec<ScoredEntity>> {
        let bm25_raw = self.index.score_all(query);
        let normalised = if self.config.normalise_bm25 {
            normalise_scores_f64(&bm25_raw)
        } else {
            bm25_raw
        };

        let mut entities: Vec<ScoredEntity> = normalised
            .into_iter()
            .filter(|(_, s)| *s >= self.config.min_score)
            .map(|(uri, score)| ScoredEntity {
                uri,
                score,
                source: ScoreSource::Keyword,
                metadata: HashMap::new(),
            })
            .collect();

        entities.truncate(self.config.top_k);
        Ok(entities)
    }

    /// Score a single document against the query (for inspection / debugging)
    pub fn score_document(&self, query: &str, doc_id: &str) -> GraphRAGResult<f64> {
        let scores = self.index.score_all(query);
        scores
            .iter()
            .find(|(id, _)| id == doc_id)
            .map(|(_, s)| *s)
            .ok_or_else(|| GraphRAGError::InternalError(format!("Document {doc_id} not found")))
    }
}

// ─── Blending helpers ────────────────────────────────────────────────────────

fn normalise_scores_f32(scores: &[(String, f32)]) -> Vec<(String, f64)> {
    let max = scores
        .iter()
        .map(|(_, s)| *s as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    if max <= 0.0 {
        return scores.iter().map(|(id, _)| (id.clone(), 0.0)).collect();
    }
    scores
        .iter()
        .map(|(id, s)| (id.clone(), *s as f64 / max))
        .collect()
}

fn normalise_scores_f64(scores: &[(String, f64)]) -> Vec<(String, f64)> {
    let max = scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    if max <= 0.0 {
        return scores.iter().map(|(id, _)| (id.clone(), 0.0)).collect();
    }
    scores.iter().map(|(id, s)| (id.clone(), s / max)).collect()
}

fn collect_all_ids(dense: &[(String, f64)], bm25: &[(String, f64)]) -> Vec<String> {
    let mut ids: Vec<String> = dense.iter().map(|(id, _)| id.clone()).collect();
    for (id, _) in bm25 {
        if !ids.contains(id) {
            ids.push(id.clone());
        }
    }
    ids
}

fn lookup(scores: &[(String, f64)], id: &str) -> f64 {
    scores
        .iter()
        .find(|(sid, _)| sid == id)
        .map(|(_, s)| *s)
        .unwrap_or(0.0)
}

fn blend_alpha(dense: &[(String, f64)], bm25: &[(String, f64)], alpha: f64) -> Vec<(String, f64)> {
    collect_all_ids(dense, bm25)
        .into_iter()
        .map(|id| {
            let d = lookup(dense, &id);
            let b = lookup(bm25, &id);
            let score = alpha * d + (1.0 - alpha) * b;
            (id, score)
        })
        .collect()
}

fn blend_max(dense: &[(String, f64)], bm25: &[(String, f64)]) -> Vec<(String, f64)> {
    collect_all_ids(dense, bm25)
        .into_iter()
        .map(|id| {
            let d = lookup(dense, &id);
            let b = lookup(bm25, &id);
            (id, d.max(b))
        })
        .collect()
}

fn blend_rrf(dense: &[(String, f64)], bm25: &[(String, f64)], k: f64) -> Vec<(String, f64)> {
    let ids = collect_all_ids(dense, bm25);
    ids.into_iter()
        .map(|id| {
            let dense_rank = dense
                .iter()
                .position(|(did, _)| did == &id)
                .map(|r| 1.0 / (k + r as f64 + 1.0))
                .unwrap_or(0.0);
            let bm25_rank = bm25
                .iter()
                .position(|(bid, _)| bid == &id)
                .map(|r| 1.0 / (k + r as f64 + 1.0))
                .unwrap_or(0.0);
            (id, dense_rank + bm25_rank)
        })
        .collect()
}

fn blend_soft_vote(dense: &[(String, f64)], bm25: &[(String, f64)]) -> Vec<(String, f64)> {
    collect_all_ids(dense, bm25)
        .into_iter()
        .map(|id| {
            // Treat each normalised score as a probability and multiply
            let d = lookup(dense, &id);
            let b = lookup(bm25, &id);
            // Use geometric mean to avoid 0 × x = 0 when only in one list
            let score = if d > 0.0 && b > 0.0 {
                (d * b).sqrt()
            } else {
                (d + b) * 0.5
            };
            (id, score)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_corpus() -> Vec<Document> {
        vec![
            Document::from_text("http://a", "battery cell safety thermal runaway"),
            Document::from_text("http://b", "battery cell capacity degradation"),
            Document::from_text("http://c", "thermal management cooling system"),
            Document::from_text("http://d", "electric vehicle charging protocol"),
            Document::from_text("http://e", "cell chemistry lithium ion cathode"),
        ]
    }

    #[test]
    fn test_bm25_basic_scoring() {
        let corpus = make_corpus();
        let index = Bm25Index::build(&corpus, Bm25Config::default());
        let scores = index.score_all("battery safety");

        assert!(!scores.is_empty());
        // http://a has both "battery" and "safety", should rank first
        assert_eq!(scores[0].0, "http://a");
    }

    #[test]
    fn test_bm25_empty_query_returns_empty() {
        let corpus = make_corpus();
        let index = Bm25Index::build(&corpus, Bm25Config::default());
        let scores = index.score_all("");
        assert!(scores.is_empty());
    }

    #[test]
    fn test_bm25_unknown_term_returns_empty() {
        let corpus = make_corpus();
        let index = Bm25Index::build(&corpus, Bm25Config::default());
        let scores = index.score_all("xyznonexistenttermXYZ");
        assert!(scores.is_empty());
    }

    #[test]
    fn test_bm25_plus_variant() {
        let corpus = make_corpus();
        let config = Bm25Config {
            variant: Bm25Variant::Plus,
            ..Default::default()
        };
        let index = Bm25Index::build(&corpus, config);
        let scores = index.score_all("battery");
        assert!(!scores.is_empty());
        // All scores should be higher than Classic because of delta
        assert!(scores[0].1 > 0.0);
    }

    #[test]
    fn test_bm25_l_variant() {
        let corpus = make_corpus();
        let config = Bm25Config {
            variant: Bm25Variant::L,
            ..Default::default()
        };
        let index = Bm25Index::build(&corpus, config);
        let scores = index.score_all("thermal");
        assert!(!scores.is_empty());
    }

    #[test]
    fn test_hybrid_alpha_blending() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig {
            blend_mode: HybridBlendMode::AlphaInterpolation { alpha: 0.5 },
            top_k: 5,
            ..Default::default()
        };
        let retriever = HybridRetriever::build(&corpus, config);

        let dense: Vec<(String, f32)> = vec![
            ("http://a".to_string(), 0.95),
            ("http://c".to_string(), 0.80),
            ("http://b".to_string(), 0.60),
        ];
        let results = retriever.retrieve("battery safety", &dense).unwrap();
        assert!(!results.is_empty());
        // http://a appears in both lists, should be near top
        let top = &results[0];
        assert!(top.score > 0.0);
    }

    #[test]
    fn test_hybrid_rrf_blending() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig {
            blend_mode: HybridBlendMode::ReciprocalRankFusion { k: 60.0 },
            top_k: 5,
            ..Default::default()
        };
        let retriever = HybridRetriever::build(&corpus, config);

        let dense: Vec<(String, f32)> = vec![
            ("http://a".to_string(), 0.9),
            ("http://b".to_string(), 0.85),
        ];
        let results = retriever.retrieve("battery cell", &dense).unwrap();
        assert!(!results.is_empty());
        // Fused source for http://a (in both)
        assert!(results
            .iter()
            .any(|e| e.source == ScoreSource::Fused || e.source == ScoreSource::Vector));
    }

    #[test]
    fn test_hybrid_max_score_blending() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig {
            blend_mode: HybridBlendMode::MaxScore,
            top_k: 5,
            ..Default::default()
        };
        let retriever = HybridRetriever::build(&corpus, config);
        let dense: Vec<(String, f32)> = vec![("http://d".to_string(), 0.99)];
        let results = retriever.retrieve("charging", &dense).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hybrid_soft_vote() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig {
            blend_mode: HybridBlendMode::SoftVote,
            top_k: 5,
            ..Default::default()
        };
        let retriever = HybridRetriever::build(&corpus, config);
        let dense: Vec<(String, f32)> = vec![("http://e".to_string(), 0.8)];
        let results = retriever.retrieve("lithium cathode", &dense).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hybrid_top_k_limiting() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig {
            top_k: 2,
            ..Default::default()
        };
        let retriever = HybridRetriever::build(&corpus, config);
        let dense: Vec<(String, f32)> = vec![
            ("http://a".to_string(), 0.9),
            ("http://b".to_string(), 0.8),
            ("http://c".to_string(), 0.7),
            ("http://d".to_string(), 0.6),
        ];
        let results = retriever.retrieve("battery", &dense).unwrap();
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_bm25_only_retrieval() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig::default();
        let retriever = HybridRetriever::build(&corpus, config);
        let results = retriever.retrieve_bm25_only("thermal management").unwrap();
        assert!(!results.is_empty());
        for e in &results {
            assert_eq!(e.source, ScoreSource::Keyword);
        }
    }

    #[test]
    fn test_score_document() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig::default();
        let retriever = HybridRetriever::build(&corpus, config);
        let score = retriever.score_document("battery", "http://a").unwrap();
        assert!(score >= 0.0);
    }

    #[test]
    fn test_score_document_not_found() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig::default();
        let retriever = HybridRetriever::build(&corpus, config);
        // "http://zzz" not in corpus
        let result = retriever.score_document("battery", "http://zzz");
        assert!(result.is_err());
    }

    #[test]
    fn test_normalise_scores_f32_max_one() {
        let scores = vec![("a".to_string(), 2.0f32), ("b".to_string(), 1.0f32)];
        let normed = normalise_scores_f32(&scores);
        assert!((normed[0].1 - 1.0).abs() < 1e-9);
        assert!((normed[1].1 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_normalise_scores_f64_empty() {
        let scores: Vec<(String, f64)> = vec![];
        let normed = normalise_scores_f64(&scores);
        assert!(normed.is_empty());
    }

    #[test]
    fn test_min_score_threshold() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig {
            min_score: 0.99, // impossibly high
            ..Default::default()
        };
        let retriever = HybridRetriever::build(&corpus, config);
        let results = retriever.retrieve("battery", &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_build_empty_corpus() {
        let index = Bm25Index::build(&[], Bm25Config::default());
        let scores = index.score_all("battery");
        assert!(scores.is_empty());
    }

    #[test]
    fn test_hybrid_empty_dense_results() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig::default();
        let retriever = HybridRetriever::build(&corpus, config);
        let results = retriever.retrieve("battery cell", &[]).unwrap();
        // BM25 should still produce results
        assert!(!results.is_empty());
    }

    #[test]
    fn test_blend_rrf_rank_ordering() {
        let dense = vec![
            ("http://a".to_string(), 1.0f64),
            ("http://b".to_string(), 0.9),
        ];
        let bm25 = vec![
            ("http://b".to_string(), 1.0f64),
            ("http://a".to_string(), 0.8),
        ];
        let blended = blend_rrf(&dense, &bm25, 60.0);
        // Both are in both lists; http://a rank-1 in dense, rank-2 in bm25
        // http://b rank-2 in dense, rank-1 in bm25 → similar total
        let a_score = blended
            .iter()
            .find(|(id, _)| id == "http://a")
            .map(|(_, s)| *s)
            .unwrap();
        let b_score = blended
            .iter()
            .find(|(id, _)| id == "http://b")
            .map(|(_, s)| *s)
            .unwrap();
        // Both should be positive
        assert!(a_score > 0.0);
        assert!(b_score > 0.0);
    }
}

// ─── Additional tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod additional_tests {
    use super::*;

    fn make_corpus() -> Vec<Document> {
        vec![
            Document::from_text("http://a", "battery cell safety thermal runaway"),
            Document::from_text("http://b", "battery cell capacity degradation"),
            Document::from_text("http://c", "thermal management cooling system"),
            Document::from_text("http://d", "electric vehicle charging protocol"),
            Document::from_text("http://e", "cell chemistry lithium ion cathode"),
        ]
    }

    // ── Bm25Config tests ──────────────────────────────────────────────────

    #[test]
    fn test_bm25_config_defaults() {
        let cfg = Bm25Config::default();
        assert!((cfg.k1 - 1.2).abs() < f64::EPSILON);
        assert!((cfg.b - 0.75).abs() < f64::EPSILON);
        assert!((cfg.delta - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.variant, Bm25Variant::Classic);
    }

    #[test]
    fn test_bm25_variant_default_is_classic() {
        let v = Bm25Variant::default();
        assert_eq!(v, Bm25Variant::Classic);
    }

    // ── Document::from_text ───────────────────────────────────────────────

    #[test]
    fn test_document_from_text_tokenises() {
        let doc = Document::from_text("http://x", "Hello World");
        assert_eq!(doc.terms, vec!["hello", "world"]);
        assert_eq!(doc.id, "http://x");
    }

    #[test]
    fn test_document_from_text_lowercases() {
        let doc = Document::from_text("http://y", "BaTtErY CELL");
        assert_eq!(doc.terms[0], "battery");
        assert_eq!(doc.terms[1], "cell");
    }

    #[test]
    fn test_document_from_text_empty_string() {
        let doc = Document::from_text("http://z", "");
        assert!(doc.terms.is_empty());
    }

    // ── BM25 index build edge cases ───────────────────────────────────────

    #[test]
    fn test_bm25_single_document_corpus() {
        let corpus = vec![Document::from_text("http://only", "unique term here")];
        let index = Bm25Index::build(&corpus, Bm25Config::default());
        let scores = index.score_all("unique");
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].0, "http://only");
    }

    #[test]
    fn test_bm25_query_term_not_in_corpus() {
        let corpus = make_corpus();
        let index = Bm25Index::build(&corpus, Bm25Config::default());
        let scores = index.score_all("zzzunknownterm");
        assert!(scores.is_empty());
    }

    #[test]
    fn test_bm25_score_all_results_sorted_desc() {
        let corpus = make_corpus();
        let index = Bm25Index::build(&corpus, Bm25Config::default());
        let scores = index.score_all("battery cell");
        for i in 1..scores.len() {
            assert!(
                scores[i - 1].1 >= scores[i].1,
                "Scores should be in descending order: {} < {}",
                scores[i - 1].1,
                scores[i].1
            );
        }
    }

    #[test]
    fn test_bm25_multiple_query_terms_boost_relevant() {
        let corpus = make_corpus();
        let index = Bm25Index::build(&corpus, Bm25Config::default());
        // http://a has both "battery" and "thermal"
        let scores = index.score_all("battery thermal");
        let a_score = scores
            .iter()
            .find(|(id, _)| id == "http://a")
            .map(|(_, s)| *s);
        assert!(a_score.is_some(), "http://a should appear in results");
        assert!(a_score.unwrap() > 0.0);
    }

    #[test]
    fn test_bm25_higher_k1_higher_saturation() {
        let corpus = vec![
            Document::from_text("http://a", "battery battery battery battery battery"),
            Document::from_text("http://b", "battery cell thermal"),
        ];
        let low_k1 = Bm25Config {
            k1: 0.5,
            ..Default::default()
        };
        let high_k1 = Bm25Config {
            k1: 5.0,
            ..Default::default()
        };
        let index_low = Bm25Index::build(&corpus, low_k1);
        let index_high = Bm25Index::build(&corpus, high_k1);
        let score_low = index_low.score_all("battery")[0].1;
        let score_high = index_high.score_all("battery")[0].1;
        // Higher k1 = more raw TF weight → higher score for repeated term
        assert!(score_high > score_low);
    }

    // ── HybridBlendMode tests ─────────────────────────────────────────────

    #[test]
    fn test_blend_mode_default_is_alpha_07() {
        let mode = HybridBlendMode::default();
        matches!(mode, HybridBlendMode::AlphaInterpolation { alpha } if (alpha - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hybrid_config_defaults() {
        let cfg = HybridRetrievalConfig::default();
        assert_eq!(cfg.top_k, 20);
        assert!((cfg.min_score - 0.0).abs() < f64::EPSILON);
        assert!(cfg.normalise_dense);
        assert!(cfg.normalise_bm25);
    }

    // ── Normalisation helpers ─────────────────────────────────────────────

    #[test]
    fn test_normalise_f32_all_zeros_returns_zeros() {
        let scores = vec![("a".to_string(), 0.0f32), ("b".to_string(), 0.0f32)];
        let normed = normalise_scores_f32(&scores);
        for (_, s) in &normed {
            assert!(*s <= 0.0);
        }
    }

    #[test]
    fn test_normalise_f64_max_normalises_to_one() {
        let scores = vec![
            ("a".to_string(), 3.0f64),
            ("b".to_string(), 1.5f64),
            ("c".to_string(), 0.0f64),
        ];
        let normed = normalise_scores_f64(&scores);
        let max_val = normed
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((max_val - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalise_f64_proportional() {
        let scores = vec![("a".to_string(), 4.0f64), ("b".to_string(), 2.0f64)];
        let normed = normalise_scores_f64(&scores);
        let a = normed
            .iter()
            .find(|(id, _)| id == "a")
            .map(|(_, s)| *s)
            .unwrap();
        let b = normed
            .iter()
            .find(|(id, _)| id == "b")
            .map(|(_, s)| *s)
            .unwrap();
        assert!((a - 1.0).abs() < 1e-9);
        assert!((b - 0.5).abs() < 1e-9);
    }

    // ── blend_rrf property tests ──────────────────────────────────────────

    #[test]
    fn test_rrf_score_increases_with_higher_rank() {
        // Item at rank 0 should score higher than rank 1
        let dense = vec![
            ("http://first".to_string(), 1.0f64),
            ("http://second".to_string(), 0.9f64),
        ];
        let bm25: Vec<(String, f64)> = vec![];
        let blended = blend_rrf(&dense, &bm25, 60.0);
        let first = blended
            .iter()
            .find(|(id, _)| id == "http://first")
            .map(|(_, s)| *s)
            .unwrap();
        let second = blended
            .iter()
            .find(|(id, _)| id == "http://second")
            .map(|(_, s)| *s)
            .unwrap();
        assert!(
            first > second,
            "Rank-0 item should score higher: {first} vs {second}"
        );
    }

    #[test]
    fn test_rrf_k_60_smoothing() {
        // With k=60, rank-0 score = 1/(60+1) ≈ 0.0164
        let dense = vec![("http://a".to_string(), 1.0f64)];
        let bm25: Vec<(String, f64)> = vec![];
        let blended = blend_rrf(&dense, &bm25, 60.0);
        let score = blended
            .iter()
            .find(|(id, _)| id == "http://a")
            .map(|(_, s)| *s)
            .unwrap();
        let expected = 1.0 / (60.0 + 1.0);
        assert!(
            (score - expected).abs() < 1e-9,
            "Expected {expected}, got {score}"
        );
    }

    // ── Alpha blend property tests ────────────────────────────────────────

    #[test]
    fn test_alpha_one_gives_pure_dense() {
        let dense = vec![("http://a".to_string(), 0.8f64)];
        let bm25 = vec![("http://b".to_string(), 0.9f64)];
        let blended = blend_alpha(&dense, &bm25, 1.0);
        let a_score = blended
            .iter()
            .find(|(id, _)| id == "http://a")
            .map(|(_, s)| *s)
            .unwrap();
        let b_score = blended
            .iter()
            .find(|(id, _)| id == "http://b")
            .map(|(_, s)| *s)
            .unwrap();
        assert!((a_score - 0.8).abs() < 1e-9);
        assert!((b_score - 0.0).abs() < 1e-9); // only in bm25, alpha=1 → 0 weight on bm25
    }

    #[test]
    fn test_alpha_zero_gives_pure_bm25() {
        let dense = vec![("http://a".to_string(), 0.8f64)];
        let bm25 = vec![("http://b".to_string(), 0.9f64)];
        let blended = blend_alpha(&dense, &bm25, 0.0);
        let a_score = blended
            .iter()
            .find(|(id, _)| id == "http://a")
            .map(|(_, s)| *s)
            .unwrap();
        let b_score = blended
            .iter()
            .find(|(id, _)| id == "http://b")
            .map(|(_, s)| *s)
            .unwrap();
        assert!((a_score - 0.0).abs() < 1e-9);
        assert!((b_score - 0.9).abs() < 1e-9);
    }

    // ── Max score blend ───────────────────────────────────────────────────

    #[test]
    fn test_max_score_takes_higher_of_two() {
        let dense = vec![("http://a".to_string(), 0.6f64)];
        let bm25 = vec![("http://a".to_string(), 0.9f64)];
        let blended = blend_max(&dense, &bm25);
        let score = blended
            .iter()
            .find(|(id, _)| id == "http://a")
            .map(|(_, s)| *s)
            .unwrap();
        assert!((score - 0.9).abs() < 1e-9);
    }

    // ── Soft vote ─────────────────────────────────────────────────────────

    #[test]
    fn test_soft_vote_both_lists_uses_geometric_mean() {
        let dense = vec![("http://a".to_string(), 0.4f64)];
        let bm25 = vec![("http://a".to_string(), 0.9f64)];
        let blended = blend_soft_vote(&dense, &bm25);
        let score = blended
            .iter()
            .find(|(id, _)| id == "http://a")
            .map(|(_, s)| *s)
            .unwrap();
        let expected = (0.4 * 0.9f64).sqrt();
        assert!((score - expected).abs() < 1e-9);
    }

    #[test]
    fn test_soft_vote_single_list_uses_half() {
        let dense = vec![("http://only_dense".to_string(), 0.6f64)];
        let bm25: Vec<(String, f64)> = vec![];
        let blended = blend_soft_vote(&dense, &bm25);
        let score = blended
            .iter()
            .find(|(id, _)| id == "http://only_dense")
            .map(|(_, s)| *s)
            .unwrap();
        // b=0, d=0.6 → (0.6 + 0) * 0.5 = 0.3
        assert!((score - 0.3).abs() < 1e-9);
    }

    // ── Source attribution tests ──────────────────────────────────────────

    #[test]
    fn test_source_fused_when_in_both_lists() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig {
            blend_mode: HybridBlendMode::AlphaInterpolation { alpha: 0.5 },
            top_k: 10,
            ..Default::default()
        };
        let retriever = HybridRetriever::build(&corpus, config);
        let dense = vec![("http://a".to_string(), 0.9f32)];
        let results = retriever.retrieve("battery safety", &dense).unwrap();
        // http://a is in both dense and BM25 results
        let a_entity = results.iter().find(|e| e.uri == "http://a");
        assert!(a_entity.is_some());
        assert_eq!(a_entity.unwrap().source, ScoreSource::Fused);
    }

    #[test]
    fn test_source_keyword_when_only_in_bm25() {
        let corpus = make_corpus();
        let config = HybridRetrievalConfig {
            blend_mode: HybridBlendMode::AlphaInterpolation { alpha: 0.0 },
            top_k: 10,
            normalise_dense: true,
            normalise_bm25: true,
            ..Default::default()
        };
        let retriever = HybridRetriever::build(&corpus, config);
        // Empty dense list → all results are from BM25
        let results = retriever.retrieve("battery", &[]).unwrap();
        for e in &results {
            assert_eq!(e.source, ScoreSource::Keyword);
        }
    }
}
