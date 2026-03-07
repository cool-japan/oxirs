//! Standalone BM25 scorer and RRF fuser with configurable weights
//!
//! This module provides:
//! - `HybridSearchConfig` – configurable weights for BM25 + dense fusion
//! - `BM25Scorer` – lightweight BM25 scorer (document index + search)
//! - `RrfFuser` – Reciprocal Rank Fusion combining two ranked lists

use std::collections::HashMap;

// ── HybridSearchConfig ────────────────────────────────────────────────────────

/// Configuration for hybrid BM25 + dense retrieval fusion
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// Weight applied to BM25 scores (0.0–1.0; does not need to sum to 1 with dense_weight)
    pub bm25_weight: f64,
    /// Weight applied to dense (vector) scores
    pub dense_weight: f64,
    /// RRF smoothing constant k (typical value: 60)
    pub rrf_k: u32,
    /// BM25 term-frequency saturation (k1 parameter)
    pub bm25_k1: f64,
    /// BM25 length normalisation (b parameter)
    pub bm25_b: f64,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            bm25_weight: 0.4,
            dense_weight: 0.6,
            rrf_k: 60,
            bm25_k1: 1.2,
            bm25_b: 0.75,
        }
    }
}

// ── BM25Scorer ────────────────────────────────────────────────────────────────

/// In-memory BM25 scorer for graph text retrieval.
///
/// Supports document indexing and ranked retrieval.
pub struct BM25Scorer {
    config: HybridSearchConfig,
    /// Per-document term frequencies: doc_id -> term -> count
    doc_tf: HashMap<String, HashMap<String, usize>>,
    /// Per-document lengths
    doc_len: HashMap<String, usize>,
    /// Inverted index: term -> document frequency
    df: HashMap<String, usize>,
    /// Total document count
    num_docs: usize,
    /// Average document length
    avg_doc_len: f64,
}

impl BM25Scorer {
    /// Create a new empty scorer with the given config.
    pub fn new(config: HybridSearchConfig) -> Self {
        Self {
            config,
            doc_tf: HashMap::new(),
            doc_len: HashMap::new(),
            df: HashMap::new(),
            num_docs: 0,
            avg_doc_len: 1.0,
        }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(HybridSearchConfig::default())
    }

    /// Index a document identified by `doc_id` with the given term list.
    ///
    /// Calling this multiple times with the same `doc_id` overwrites the previous entry.
    pub fn index_document(&mut self, doc_id: &str, terms: &[&str]) {
        // Remove previous entry if re-indexing
        if self.doc_tf.contains_key(doc_id) {
            let old_tf = self.doc_tf.remove(doc_id).unwrap_or_default();
            for term in old_tf.keys() {
                if let Some(count) = self.df.get_mut(term.as_str()) {
                    *count = count.saturating_sub(1);
                }
            }
            self.doc_len.remove(doc_id);
            self.num_docs = self.num_docs.saturating_sub(1);
        }

        let mut tf: HashMap<String, usize> = HashMap::new();
        for &term in terms {
            *tf.entry(term.to_lowercase()).or_insert(0) += 1;
        }
        for term in tf.keys() {
            *self.df.entry(term.clone()).or_insert(0) += 1;
        }
        self.doc_len.insert(doc_id.to_string(), terms.len());
        self.doc_tf.insert(doc_id.to_string(), tf);
        self.num_docs += 1;

        // Recompute average document length
        let total: usize = self.doc_len.values().sum();
        self.avg_doc_len = if self.num_docs == 0 {
            1.0
        } else {
            total as f64 / self.num_docs as f64
        };
    }

    /// Compute the BM25 score for a single document against query terms.
    ///
    /// Returns 0.0 if the document is not indexed.
    pub fn score(&self, query_terms: &[&str], doc_terms: &[&str], avg_doc_len: f64) -> f64 {
        // Compute ad-hoc TF for doc_terms
        let mut local_tf: HashMap<String, usize> = HashMap::new();
        for &t in doc_terms {
            *local_tf.entry(t.to_lowercase()).or_insert(0) += 1;
        }
        let dl = doc_terms.len() as f64;
        let k1 = self.config.bm25_k1;
        let b = self.config.bm25_b;
        let n = (self.num_docs.max(1)) as f64;

        let mut score = 0.0;
        for &qt in query_terms {
            let qt_lower = qt.to_lowercase();
            let df_t = *self.df.get(&qt_lower).unwrap_or(&0) as f64;
            if df_t == 0.0 {
                continue;
            }
            let idf = ((n - df_t + 0.5) / (df_t + 0.5) + 1.0).ln();
            let tf_t = *local_tf.get(&qt_lower).unwrap_or(&0) as f64;
            let norm = 1.0 - b + b * (dl / avg_doc_len.max(1.0));
            let tf_weight = tf_t * (k1 + 1.0) / (tf_t + k1 * norm);
            score += idf * tf_weight;
        }
        score
    }

    /// Search all indexed documents and return ranked results.
    ///
    /// Returns `Vec<(doc_id, score)>` sorted descending by score.
    pub fn search(&self, query: &[&str], top_k: usize) -> Vec<(String, f64)> {
        if self.num_docs == 0 || query.is_empty() {
            return vec![];
        }

        let k1 = self.config.bm25_k1;
        let b = self.config.bm25_b;
        let n = self.num_docs as f64;
        let avgdl = self.avg_doc_len;

        let query_lower: Vec<String> = query.iter().map(|t| t.to_lowercase()).collect();
        let mut scores: HashMap<&str, f64> = HashMap::new();

        for qt in &query_lower {
            let df_t = *self.df.get(qt.as_str()).unwrap_or(&0) as f64;
            if df_t == 0.0 {
                continue;
            }
            let idf = ((n - df_t + 0.5) / (df_t + 0.5) + 1.0).ln();
            for (doc_id, tf_map) in &self.doc_tf {
                let tf_t = *tf_map.get(qt.as_str()).unwrap_or(&0) as f64;
                if tf_t == 0.0 {
                    continue;
                }
                let dl = *self.doc_len.get(doc_id.as_str()).unwrap_or(&0) as f64;
                let norm = 1.0 - b + b * (dl / avgdl);
                let tf_weight = tf_t * (k1 + 1.0) / (tf_t + k1 * norm);
                *scores.entry(doc_id.as_str()).or_insert(0.0) += idf * tf_weight;
            }
        }

        let mut result: Vec<(String, f64)> = scores
            .into_iter()
            .filter(|(_, s)| *s > 0.0)
            .map(|(id, s)| (id.to_string(), s))
            .collect();

        result.sort_by(|a, b_| b_.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if top_k > 0 {
            result.truncate(top_k);
        }
        result
    }

    /// Total number of indexed documents
    pub fn doc_count(&self) -> usize {
        self.num_docs
    }
}

// ── RrfFuser ─────────────────────────────────────────────────────────────────

/// Reciprocal Rank Fusion combining BM25 and dense retrieval results.
///
/// Formula: `score(d) = bm25_w * 1/(k+rank_bm25) + dense_w * 1/(k+rank_dense)`
pub struct RrfFuser {
    config: HybridSearchConfig,
}

impl RrfFuser {
    /// Create a new fuser with the given config.
    pub fn new(config: HybridSearchConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(HybridSearchConfig::default())
    }

    /// Fuse BM25 and dense results using Reciprocal Rank Fusion.
    ///
    /// Both input lists are assumed to be pre-sorted descending by score.
    /// Returns a merged ranked list sorted descending by fused score.
    pub fn fuse(
        &self,
        bm25_results: &[(String, f64)],
        dense_results: &[(String, f64)],
        config: &HybridSearchConfig,
    ) -> Vec<(String, f64)> {
        let k = config.rrf_k as f64;
        let bm25_w = config.bm25_weight;
        let dense_w = config.dense_weight;

        // Collect all unique IDs
        let mut all_ids: Vec<String> = bm25_results.iter().map(|(id, _)| id.clone()).collect();
        for (id, _) in dense_results {
            if !all_ids.contains(id) {
                all_ids.push(id.clone());
            }
        }

        let mut scored: Vec<(String, f64)> = all_ids
            .into_iter()
            .map(|id| {
                let bm25_rrf = bm25_results
                    .iter()
                    .position(|(did, _)| did == &id)
                    .map(|r| bm25_w / (k + r as f64 + 1.0))
                    .unwrap_or(0.0);
                let dense_rrf = dense_results
                    .iter()
                    .position(|(did, _)| did == &id)
                    .map(|r| dense_w / (k + r as f64 + 1.0))
                    .unwrap_or(0.0);
                (id, bm25_rrf + dense_rrf)
            })
            .collect();

        scored.sort_by(|a, b_| b_.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scorer_with_docs() -> BM25Scorer {
        let mut scorer = BM25Scorer::with_defaults();
        scorer.index_document("doc:a", &["battery", "safety", "thermal", "runaway"]);
        scorer.index_document("doc:b", &["battery", "capacity", "degradation"]);
        scorer.index_document("doc:c", &["thermal", "management", "cooling"]);
        scorer.index_document("doc:d", &["electric", "vehicle", "charging"]);
        scorer
    }

    // ── HybridSearchConfig ────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = HybridSearchConfig::default();
        assert!((cfg.bm25_weight - 0.4).abs() < f64::EPSILON);
        assert!((cfg.dense_weight - 0.6).abs() < f64::EPSILON);
        assert_eq!(cfg.rrf_k, 60);
        assert!((cfg.bm25_k1 - 1.2).abs() < f64::EPSILON);
        assert!((cfg.bm25_b - 0.75).abs() < f64::EPSILON);
    }

    // ── BM25Scorer::index_document ────────────────────────────────────────

    #[test]
    fn test_index_document_count() {
        let scorer = make_scorer_with_docs();
        assert_eq!(scorer.doc_count(), 4);
    }

    #[test]
    fn test_index_empty_scorer_count_zero() {
        let scorer = BM25Scorer::with_defaults();
        assert_eq!(scorer.doc_count(), 0);
    }

    // ── BM25Scorer::search ────────────────────────────────────────────────

    #[test]
    fn test_search_returns_relevant_doc() {
        let scorer = make_scorer_with_docs();
        let results = scorer.search(&["battery", "safety"], 10);
        assert!(!results.is_empty());
        // doc:a has both "battery" and "safety", should rank first
        assert_eq!(results[0].0, "doc:a");
    }

    #[test]
    fn test_search_empty_query_returns_empty() {
        let scorer = make_scorer_with_docs();
        let results = scorer.search(&[], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_unknown_term_returns_empty() {
        let scorer = make_scorer_with_docs();
        let results = scorer.search(&["xyznonsense"], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_top_k_limits_results() {
        let scorer = make_scorer_with_docs();
        let results = scorer.search(&["battery", "thermal", "electric"], 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_search_results_sorted_descending() {
        let scorer = make_scorer_with_docs();
        let results = scorer.search(&["battery"], 10);
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 >= results[i].1,
                "Results not sorted: {} < {}",
                results[i - 1].1,
                results[i].1
            );
        }
    }

    // ── BM25Scorer::score ─────────────────────────────────────────────────

    #[test]
    fn test_score_single_term_positive() {
        let scorer = make_scorer_with_docs();
        let s = scorer.score(&["battery"], &["battery", "safety"], 4.0);
        assert!(s > 0.0);
    }

    #[test]
    fn test_score_no_overlap_zero() {
        let scorer = make_scorer_with_docs();
        // Query term "xyz" not in df → score = 0
        let s = scorer.score(&["xyz"], &["battery", "safety"], 4.0);
        assert!((s - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_multiple_matching_terms_higher() {
        let scorer = make_scorer_with_docs();
        let s1 = scorer.score(&["battery"], &["battery"], 1.0);
        let s2 = scorer.score(&["battery", "thermal"], &["battery", "thermal"], 2.0);
        // Two-term match should produce higher raw score
        assert!(s2 > s1, "s2={s2} should be > s1={s1}");
    }

    // ── BM25Scorer re-indexing ────────────────────────────────────────────

    #[test]
    fn test_reindex_document_updates_count() {
        let mut scorer = BM25Scorer::with_defaults();
        scorer.index_document("doc:x", &["alpha", "beta"]);
        scorer.index_document("doc:x", &["gamma", "delta"]); // re-index same doc
        assert_eq!(scorer.doc_count(), 1); // still 1 doc
                                           // "alpha"/"beta" should no longer score
        let r = scorer.search(&["alpha"], 10);
        assert!(r.is_empty());
        let r = scorer.search(&["gamma"], 10);
        assert!(!r.is_empty());
    }

    // ── RrfFuser::fuse ───────────────────────────────────────────────────

    #[test]
    fn test_rrf_fuse_combined_score_positive() {
        let fuser = RrfFuser::with_defaults();
        let cfg = HybridSearchConfig::default();
        let bm25 = vec![("doc:a".to_string(), 1.0), ("doc:b".to_string(), 0.5)];
        let dense = vec![("doc:a".to_string(), 0.9), ("doc:c".to_string(), 0.7)];
        let result = fuser.fuse(&bm25, &dense, &cfg);
        assert!(!result.is_empty());
        // doc:a is in both lists → highest fused score
        assert_eq!(result[0].0, "doc:a");
    }

    #[test]
    fn test_rrf_fuse_sorted_descending() {
        let fuser = RrfFuser::with_defaults();
        let cfg = HybridSearchConfig::default();
        let bm25 = vec![
            ("doc:a".to_string(), 1.0),
            ("doc:b".to_string(), 0.8),
            ("doc:c".to_string(), 0.6),
        ];
        let dense = vec![("doc:b".to_string(), 1.0), ("doc:a".to_string(), 0.7)];
        let result = fuser.fuse(&bm25, &dense, &cfg);
        for i in 1..result.len() {
            assert!(
                result[i - 1].1 >= result[i].1,
                "Not sorted: {}:{} >= {}:{}",
                result[i - 1].0,
                result[i - 1].1,
                result[i].0,
                result[i].1
            );
        }
    }

    #[test]
    fn test_rrf_fuse_empty_bm25_uses_dense_only() {
        let fuser = RrfFuser::with_defaults();
        let cfg = HybridSearchConfig::default();
        let bm25: Vec<(String, f64)> = vec![];
        let dense = vec![("doc:x".to_string(), 0.9)];
        let result = fuser.fuse(&bm25, &dense, &cfg);
        assert!(!result.is_empty());
        assert_eq!(result[0].0, "doc:x");
    }

    #[test]
    fn test_rrf_fuse_empty_dense_uses_bm25_only() {
        let fuser = RrfFuser::with_defaults();
        let cfg = HybridSearchConfig::default();
        let bm25 = vec![("doc:y".to_string(), 0.9)];
        let dense: Vec<(String, f64)> = vec![];
        let result = fuser.fuse(&bm25, &dense, &cfg);
        assert!(!result.is_empty());
        assert_eq!(result[0].0, "doc:y");
    }

    #[test]
    fn test_rrf_fuse_both_empty_returns_empty() {
        let fuser = RrfFuser::with_defaults();
        let cfg = HybridSearchConfig::default();
        let result = fuser.fuse(&[], &[], &cfg);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rrf_k_affects_scores() {
        let fuser = RrfFuser::with_defaults();
        let cfg_low_k = HybridSearchConfig {
            rrf_k: 1,
            ..Default::default()
        };
        let cfg_high_k = HybridSearchConfig {
            rrf_k: 1000,
            ..Default::default()
        };
        let bm25 = vec![("doc:a".to_string(), 1.0)];
        let dense: Vec<(String, f64)> = vec![];
        let low_k_result = fuser.fuse(&bm25, &dense, &cfg_low_k);
        let high_k_result = fuser.fuse(&bm25, &dense, &cfg_high_k);
        // Lower k gives higher RRF scores (less smoothing)
        assert!(
            low_k_result[0].1 > high_k_result[0].1,
            "Low k should give higher score"
        );
    }

    #[test]
    fn test_rrf_weight_affects_rank() {
        // Give much higher weight to BM25; doc:bm25only should score well
        let fuser = RrfFuser::with_defaults();
        let cfg = HybridSearchConfig {
            bm25_weight: 0.99,
            dense_weight: 0.01,
            rrf_k: 60,
            ..Default::default()
        };
        let bm25 = vec![("doc:bm25only".to_string(), 1.0)];
        let dense = vec![("doc:denseonly".to_string(), 1.0)];
        let result = fuser.fuse(&bm25, &dense, &cfg);
        assert_eq!(result[0].0, "doc:bm25only");
    }

    #[test]
    fn test_rrf_deduplicates_ids() {
        let fuser = RrfFuser::with_defaults();
        let cfg = HybridSearchConfig::default();
        let bm25 = vec![("doc:shared".to_string(), 1.0)];
        let dense = vec![("doc:shared".to_string(), 1.0)];
        let result = fuser.fuse(&bm25, &dense, &cfg);
        // doc:shared should appear exactly once
        let count = result.iter().filter(|(id, _)| id == "doc:shared").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_rrf_new_and_with_defaults_equivalent_k() {
        let cfg = HybridSearchConfig::default();
        let rrf_k = cfg.rrf_k;
        let fuser1 = RrfFuser::new(cfg);
        let fuser2 = RrfFuser::with_defaults();
        assert_eq!(fuser1.config.rrf_k, fuser2.config.rrf_k);
        assert_eq!(fuser1.config.rrf_k, rrf_k);
    }
}
