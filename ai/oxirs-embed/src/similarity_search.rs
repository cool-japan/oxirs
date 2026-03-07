//! Multi-metric similarity search over embedding collections.
//!
//! Supports Cosine, L2, Dot-product, and Manhattan distance metrics,
//! with top-k selection, minimum similarity filtering, and label filtering.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// An embedding record stored in the index.
#[derive(Debug, Clone)]
pub struct EmbeddingRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub label: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// The distance / similarity metric to use for search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimilarityMetric {
    Cosine,
    L2,
    DotProduct,
    Manhattan,
}

/// Parameters for a single search query.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub vector: Vec<f32>,
    pub metric: SimilarityMetric,
    /// Maximum number of hits to return.
    pub top_k: usize,
    /// Minimum score threshold (results below this are excluded).
    pub min_similarity: Option<f32>,
    /// If set, only records whose label matches are considered.
    pub filter_label: Option<String>,
}

/// A single search result.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: String,
    /// Score meaning depends on the metric:
    /// - Cosine / DotProduct: higher is better.
    /// - L2 / Manhattan: negated distance (higher = closer).
    pub score: f32,
    pub label: Option<String>,
}

/// An in-memory similarity search index over [`EmbeddingRecord`]s.
pub struct SimilaritySearchIndex {
    records: Vec<EmbeddingRecord>,
}

impl Default for SimilaritySearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SimilaritySearchIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Add a record to the index.
    pub fn add(&mut self, record: EmbeddingRecord) {
        self.records.push(record);
    }

    /// Remove the record with the given `id`.
    ///
    /// Returns `true` if a record was removed.
    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(pos) = self.records.iter().position(|r| r.id == id) {
            self.records.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Search the index and return up to `top_k` hits sorted by score
    /// (best first).
    pub fn search(&self, query: &SearchQuery) -> Vec<SearchHit> {
        let min_sim = query.min_similarity.unwrap_or(f32::NEG_INFINITY);

        let mut scored: Vec<(f32, &EmbeddingRecord)> = self
            .records
            .iter()
            .filter(|r| {
                if let Some(ref lbl) = query.filter_label {
                    r.label.as_deref() == Some(lbl.as_str())
                } else {
                    true
                }
            })
            .filter_map(|r| {
                if r.vector.len() != query.vector.len() {
                    return None;
                }
                let score = compute_score(&query.metric, &query.vector, &r.vector);
                if score >= min_sim {
                    Some((score, r))
                } else {
                    None
                }
            })
            .collect();

        // Sort descending by score (best first)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(query.top_k)
            .map(|(score, r)| SearchHit {
                id: r.id.clone(),
                score,
                label: r.label.clone(),
            })
            .collect()
    }

    /// Run multiple queries and return one result list per query.
    pub fn search_batch(&self, queries: &[SearchQuery]) -> Vec<Vec<SearchHit>> {
        queries.iter().map(|q| self.search(q)).collect()
    }

    /// Return the single nearest neighbour (or `None` if the index is empty).
    pub fn nearest_neighbor(
        &self,
        vector: &[f32],
        metric: SimilarityMetric,
    ) -> Option<SearchHit> {
        let query = SearchQuery {
            vector: vector.to_vec(),
            metric,
            top_k: 1,
            min_similarity: None,
            filter_label: None,
        };
        self.search(&query).into_iter().next()
    }

    /// Total number of records in the index.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Dimensionality inferred from the first record, or `None` if the index
    /// is empty.
    pub fn dimension(&self) -> Option<usize> {
        self.records.first().map(|r| r.vector.len())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Metric implementations
// ──────────────────────────────────────────────────────────────────────────────

/// Dispatch to the correct metric function.
fn compute_score(metric: &SimilarityMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        SimilarityMetric::Cosine => cosine_similarity(a, b),
        SimilarityMetric::L2 => -l2_distance(a, b),
        SimilarityMetric::DotProduct => dot_product(a, b),
        SimilarityMetric::Manhattan => -manhattan_distance(a, b),
    }
}

/// Cosine similarity ∈ [−1, 1].
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

/// Euclidean (L2) distance ≥ 0.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Dot product (inner product).
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Manhattan (L1) distance ≥ 0.
fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(id: &str, v: Vec<f32>, label: Option<&str>) -> EmbeddingRecord {
        EmbeddingRecord {
            id: id.to_string(),
            vector: v,
            label: label.map(|s| s.to_string()),
            metadata: HashMap::new(),
        }
    }

    fn simple_query(v: Vec<f32>, metric: SimilarityMetric, k: usize) -> SearchQuery {
        SearchQuery {
            vector: v,
            metric,
            top_k: k,
            min_similarity: None,
            filter_label: None,
        }
    }

    fn index_123() -> SimilaritySearchIndex {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![1.0, 0.0, 0.0], Some("a")));
        idx.add(make_record("r2", vec![0.0, 1.0, 0.0], Some("b")));
        idx.add(make_record("r3", vec![0.0, 0.0, 1.0], Some("a")));
        idx
    }

    // ── basic setup ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_index_empty() {
        let idx = SimilaritySearchIndex::new();
        assert_eq!(idx.record_count(), 0);
        assert!(idx.dimension().is_none());
    }

    #[test]
    fn test_add_increments_count() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![1.0], None));
        assert_eq!(idx.record_count(), 1);
    }

    #[test]
    fn test_dimension_from_first_record() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![1.0, 2.0, 3.0], None));
        assert_eq!(idx.dimension(), Some(3));
    }

    // ── remove ───────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_record() {
        let mut idx = index_123();
        assert!(idx.remove("r2"));
        assert_eq!(idx.record_count(), 2);
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut idx = index_123();
        assert!(!idx.remove("r999"));
    }

    #[test]
    fn test_remove_only_element() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![1.0], None));
        assert!(idx.remove("r1"));
        assert_eq!(idx.record_count(), 0);
    }

    // ── cosine metric ────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical_vector_max_score() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![1.0, 0.0, 0.0], None));
        let hits = idx.search(&simple_query(vec![1.0, 0.0, 0.0], SimilarityMetric::Cosine, 1));
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_orthogonal_vectors_zero_score() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![1.0, 0.0], None));
        let hits = idx.search(&simple_query(vec![0.0, 1.0], SimilarityMetric::Cosine, 1));
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_result_ordering() {
        let idx = index_123();
        // Query aligned with r1
        let hits = idx.search(&simple_query(
            vec![1.0, 0.0, 0.0],
            SimilarityMetric::Cosine,
            3,
        ));
        assert_eq!(hits[0].id, "r1");
        assert!(hits[0].score > hits[1].score);
    }

    // ── L2 metric ────────────────────────────────────────────────────────────

    #[test]
    fn test_l2_identical_vector_highest_score() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![1.0, 2.0, 3.0], None));
        let hits = idx.search(&simple_query(vec![1.0, 2.0, 3.0], SimilarityMetric::L2, 1));
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 0.0).abs() < 1e-5); // negated distance = 0
    }

    #[test]
    fn test_l2_closer_record_ranked_higher() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("far", vec![10.0, 0.0], None));
        idx.add(make_record("near", vec![1.0, 0.0], None));
        let hits = idx.search(&simple_query(vec![0.0, 0.0], SimilarityMetric::L2, 2));
        assert_eq!(hits[0].id, "near");
    }

    // ── DotProduct metric ────────────────────────────────────────────────────

    #[test]
    fn test_dot_product_correct() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![2.0, 3.0], None));
        idx.add(make_record("r2", vec![1.0, 1.0], None));
        let hits = idx.search(&simple_query(vec![1.0, 1.0], SimilarityMetric::DotProduct, 2));
        // r1: 2+3=5, r2: 1+1=2
        assert_eq!(hits[0].id, "r1");
        assert!((hits[0].score - 5.0).abs() < 1e-5);
    }

    // ── Manhattan metric ─────────────────────────────────────────────────────

    #[test]
    fn test_manhattan_identical_vector() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![3.0, 4.0], None));
        let hits = idx.search(&simple_query(vec![3.0, 4.0], SimilarityMetric::Manhattan, 1));
        assert!((hits[0].score - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_manhattan_closer_ranked_higher() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("close", vec![1.0, 1.0], None));
        idx.add(make_record("far", vec![5.0, 5.0], None));
        let hits = idx.search(&simple_query(vec![0.0, 0.0], SimilarityMetric::Manhattan, 2));
        assert_eq!(hits[0].id, "close");
    }

    // ── top_k ────────────────────────────────────────────────────────────────

    #[test]
    fn test_top_k_limits_results() {
        let idx = index_123();
        let hits = idx.search(&simple_query(vec![1.0, 0.0, 0.0], SimilarityMetric::Cosine, 1));
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_top_k_larger_than_index() {
        let idx = index_123();
        let hits = idx.search(&simple_query(vec![1.0, 0.0, 0.0], SimilarityMetric::Cosine, 100));
        assert_eq!(hits.len(), 3);
    }

    // ── min_similarity ───────────────────────────────────────────────────────

    #[test]
    fn test_min_similarity_filters_low_scores() {
        let idx = index_123();
        let q = SearchQuery {
            vector: vec![1.0, 0.0, 0.0],
            metric: SimilarityMetric::Cosine,
            top_k: 10,
            min_similarity: Some(0.9),
            filter_label: None,
        };
        let hits = idx.search(&q);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "r1");
    }

    #[test]
    fn test_min_similarity_none_returns_all() {
        let idx = index_123();
        let q = SearchQuery {
            vector: vec![1.0, 0.0, 0.0],
            metric: SimilarityMetric::Cosine,
            top_k: 10,
            min_similarity: None,
            filter_label: None,
        };
        let hits = idx.search(&q);
        assert_eq!(hits.len(), 3);
    }

    // ── filter_label ─────────────────────────────────────────────────────────

    #[test]
    fn test_filter_label_restricts_results() {
        let idx = index_123();
        let q = SearchQuery {
            vector: vec![0.0, 0.0, 1.0],
            metric: SimilarityMetric::Cosine,
            top_k: 10,
            min_similarity: None,
            filter_label: Some("a".to_string()),
        };
        let hits = idx.search(&q);
        assert!(hits.iter().all(|h| h.label.as_deref() == Some("a")));
    }

    #[test]
    fn test_filter_label_no_match_returns_empty() {
        let idx = index_123();
        let q = SearchQuery {
            vector: vec![1.0, 0.0, 0.0],
            metric: SimilarityMetric::Cosine,
            top_k: 10,
            min_similarity: None,
            filter_label: Some("zzz".to_string()),
        };
        let hits = idx.search(&q);
        assert!(hits.is_empty());
    }

    // ── nearest_neighbor ─────────────────────────────────────────────────────

    #[test]
    fn test_nearest_neighbor_returns_closest() {
        let idx = index_123();
        let nn = idx.nearest_neighbor(&[1.0, 0.0, 0.0], SimilarityMetric::Cosine).unwrap();
        assert_eq!(nn.id, "r1");
    }

    #[test]
    fn test_nearest_neighbor_empty_index_returns_none() {
        let idx = SimilaritySearchIndex::new();
        assert!(idx.nearest_neighbor(&[1.0, 0.0], SimilarityMetric::L2).is_none());
    }

    // ── search_batch ─────────────────────────────────────────────────────────

    #[test]
    fn test_search_batch_returns_one_list_per_query() {
        let idx = index_123();
        let queries = vec![
            simple_query(vec![1.0, 0.0, 0.0], SimilarityMetric::Cosine, 1),
            simple_query(vec![0.0, 1.0, 0.0], SimilarityMetric::Cosine, 1),
        ];
        let results = idx.search_batch(&queries);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].id, "r1");
        assert_eq!(results[1][0].id, "r2");
    }

    #[test]
    fn test_search_batch_empty_queries() {
        let idx = index_123();
        let results = idx.search_batch(&[]);
        assert!(results.is_empty());
    }

    // ── empty index ──────────────────────────────────────────────────────────

    #[test]
    fn test_search_empty_index_returns_empty() {
        let idx = SimilaritySearchIndex::new();
        let hits = idx.search(&simple_query(vec![1.0, 0.0], SimilarityMetric::Cosine, 5));
        assert!(hits.is_empty());
    }

    // ── SearchHit ordering ───────────────────────────────────────────────────

    #[test]
    fn test_hits_sorted_descending() {
        let idx = index_123();
        let hits = idx.search(&simple_query(
            vec![1.0, 0.1, 0.0],
            SimilarityMetric::Cosine,
            3,
        ));
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    // ── dimension mismatch ───────────────────────────────────────────────────

    #[test]
    fn test_dimension_mismatch_skipped() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![1.0, 2.0, 3.0], None));
        // Query has different dimension — record should be skipped
        let hits = idx.search(&simple_query(vec![1.0, 2.0], SimilarityMetric::Cosine, 5));
        assert!(hits.is_empty());
    }

    // ── label in hits ────────────────────────────────────────────────────────

    #[test]
    fn test_hit_label_carried_through() {
        let idx = index_123();
        let hits = idx.search(&simple_query(
            vec![0.0, 1.0, 0.0],
            SimilarityMetric::Cosine,
            1,
        ));
        assert_eq!(hits[0].label.as_deref(), Some("b"));
    }

    #[test]
    fn test_hit_no_label_is_none() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![1.0], None));
        let hits = idx.search(&simple_query(vec![1.0], SimilarityMetric::DotProduct, 1));
        assert!(hits[0].label.is_none());
    }

    // ── metric helper functions ──────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![3.0_f32, 4.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_distance_zero() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!(l2_distance(&v, &v) < 1e-5);
    }

    #[test]
    fn test_dot_product_known_value() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_manhattan_distance_known() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![3.0_f32, 4.0];
        assert!((manhattan_distance(&a, &b) - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_default_index() {
        let idx = SimilaritySearchIndex::default();
        assert_eq!(idx.record_count(), 0);
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_add_multiple_records() {
        let mut idx = SimilaritySearchIndex::new();
        for i in 0..10 {
            idx.add(make_record(&format!("r{i}"), vec![i as f32, 0.0], None));
        }
        assert_eq!(idx.record_count(), 10);
    }

    #[test]
    fn test_remove_after_multiple_adds() {
        let mut idx = index_123();
        assert!(idx.remove("r1"));
        assert!(idx.remove("r3"));
        assert_eq!(idx.record_count(), 1);
        assert!(!idx.remove("r1")); // already gone
    }

    #[test]
    fn test_search_returns_correct_ids() {
        let idx = index_123();
        let hits = idx.search(&simple_query(vec![0.0, 1.0, 0.0], SimilarityMetric::Cosine, 3));
        let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        assert!(ids.contains(&"r2")); // r2 = [0,1,0], best match
    }

    #[test]
    fn test_cosine_negative_vectors() {
        let a = vec![-1.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_l2_known_distance() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        assert!((l2_distance(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_manhattan_single_dimension() {
        let a = vec![5.0_f32];
        let b = vec![2.0_f32];
        assert!((manhattan_distance(&a, &b) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_zero_vectors() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        assert!((dot_product(&a, &b)).abs() < 1e-5);
    }

    #[test]
    fn test_nearest_neighbor_l2() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("close", vec![1.0, 0.0], None));
        idx.add(make_record("far", vec![100.0, 0.0], None));
        let nn = idx.nearest_neighbor(&[0.0, 0.0], SimilarityMetric::L2).unwrap();
        assert_eq!(nn.id, "close");
    }

    #[test]
    fn test_min_similarity_exact_match_threshold() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("r1", vec![1.0, 0.0], None));
        let q = SearchQuery {
            vector: vec![1.0, 0.0],
            metric: SimilarityMetric::Cosine,
            top_k: 10,
            min_similarity: Some(1.0), // exact 1.0 should still be included
            filter_label: None,
        };
        let hits = idx.search(&q);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_search_batch_empty_each_query() {
        let idx = SimilaritySearchIndex::new();
        let queries = vec![
            simple_query(vec![1.0], SimilarityMetric::Cosine, 5),
            simple_query(vec![1.0], SimilarityMetric::L2, 5),
        ];
        let results = idx.search_batch(&queries);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_empty());
        assert!(results[1].is_empty());
    }

    #[test]
    fn test_dimension_updates_with_records() {
        let mut idx = SimilaritySearchIndex::new();
        assert!(idx.dimension().is_none());
        idx.add(make_record("r1", vec![1.0, 2.0], None));
        assert_eq!(idx.dimension(), Some(2));
    }

    #[test]
    fn test_remove_decrements_count() {
        let mut idx = index_123();
        let before = idx.record_count();
        idx.remove("r1");
        assert_eq!(idx.record_count(), before - 1);
    }

    #[test]
    fn test_nearest_neighbor_manhattan() {
        let mut idx = SimilaritySearchIndex::new();
        idx.add(make_record("near", vec![1.0, 1.0], None));
        idx.add(make_record("far", vec![10.0, 10.0], None));
        let nn = idx
            .nearest_neighbor(&[0.0, 0.0], SimilarityMetric::Manhattan)
            .unwrap();
        assert_eq!(nn.id, "near");
    }
}
