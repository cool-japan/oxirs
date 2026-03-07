//! Vector embedding similarity metrics and nearest-neighbour utilities.
//!
//! Provides `EmbeddingSimilarity` with multiple distance / similarity metrics
//! and a `top_k` search over an in-memory corpus.

// ── SimilarityMetric ──────────────────────────────────────────────────────────

/// A distance or similarity measure between two embedding vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Cosine similarity (1 = identical direction, -1 = opposite).
    Cosine,
    /// Raw dot product.
    DotProduct,
    /// Euclidean (L2) distance — converted to a similarity score as `1 / (1 + d)`.
    Euclidean,
    /// Manhattan (L1) distance — similarity as `1 / (1 + d)`.
    Manhattan,
    /// Chebyshev (L∞) distance — similarity as `1 / (1 + d)`.
    Chebyshev,
}

// ── SimilarityResult ──────────────────────────────────────────────────────────

/// A single result from a nearest-neighbour search.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityResult {
    /// Index of the corpus vector.
    pub index: usize,
    /// Similarity score (higher = more similar for all metrics).
    pub score: f64,
    /// Optional human-readable label.
    pub label: Option<String>,
}

// ── EmbeddingSimilarity ───────────────────────────────────────────────────────

/// Utility functions for vector embedding similarity.
pub struct EmbeddingSimilarity;

impl EmbeddingSimilarity {
    // ── Individual metrics ────────────────────────────────────────────────────

    /// Cosine similarity between two vectors.
    ///
    /// Returns `0.0` if either vector is the zero vector.
    pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
        let dot = Self::dot_product(a, b);
        let norm_a = Self::l2_norm(a);
        let norm_b = Self::l2_norm(b);
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Raw dot product of two vectors.
    pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Euclidean distance converted to a similarity score `1 / (1 + distance)`.
    pub fn euclidean(a: &[f64], b: &[f64]) -> f64 {
        let dist: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        1.0 / (1.0 + dist)
    }

    /// Manhattan (L1) distance converted to a similarity score `1 / (1 + distance)`.
    pub fn manhattan(a: &[f64], b: &[f64]) -> f64 {
        let dist: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        1.0 / (1.0 + dist)
    }

    /// Chebyshev (L∞) distance converted to a similarity score `1 / (1 + distance)`.
    pub fn chebyshev(a: &[f64], b: &[f64]) -> f64 {
        let dist = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        1.0 / (1.0 + dist)
    }

    // ── Dispatch ──────────────────────────────────────────────────────────────

    /// Compute the similarity between `a` and `b` using the given `metric`.
    pub fn compute(a: &[f64], b: &[f64], metric: SimilarityMetric) -> f64 {
        match metric {
            SimilarityMetric::Cosine => Self::cosine(a, b),
            SimilarityMetric::DotProduct => Self::dot_product(a, b),
            SimilarityMetric::Euclidean => Self::euclidean(a, b),
            SimilarityMetric::Manhattan => Self::manhattan(a, b),
            SimilarityMetric::Chebyshev => Self::chebyshev(a, b),
        }
    }

    // ── Top-k search ──────────────────────────────────────────────────────────

    /// Return the top-`k` most similar vectors from `corpus` relative to `query`.
    ///
    /// Results are sorted descending by score (most similar first).
    /// If `k` exceeds `corpus.len()`, all entries are returned.
    pub fn top_k(
        query: &[f64],
        corpus: &[Vec<f64>],
        k: usize,
        metric: SimilarityMetric,
    ) -> Vec<SimilarityResult> {
        let mut scored: Vec<SimilarityResult> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| SimilarityResult {
                index: i,
                score: Self::compute(query, v, metric),
                label: None,
            })
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored
    }

    // ── Normalisation ─────────────────────────────────────────────────────────

    /// L2-normalise a vector (return a unit vector).
    ///
    /// If the input is the zero vector, returns a zero vector.
    pub fn normalize(v: &[f64]) -> Vec<f64> {
        let norm = Self::l2_norm(v);
        if norm == 0.0 {
            return vec![0.0; v.len()];
        }
        v.iter().map(|x| x / norm).collect()
    }

    // ── Pairwise matrix ───────────────────────────────────────────────────────

    /// Compute the N×N pairwise similarity matrix for `corpus`.
    ///
    /// `result[i][j]` is the similarity between `corpus[i]` and `corpus[j]`.
    pub fn pairwise(corpus: &[Vec<f64>], metric: SimilarityMetric) -> Vec<Vec<f64>> {
        let n = corpus.len();
        let mut matrix = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = Self::compute(&corpus[i], &corpus[j], metric);
            }
        }
        matrix
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn l2_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    // ── cosine ────────────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let s = EmbeddingSimilarity::cosine(&v, &v);
        assert!(approx_eq(s, 1.0));
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let s = EmbeddingSimilarity::cosine(&a, &b);
        assert!(approx_eq(s, -1.0));
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let s = EmbeddingSimilarity::cosine(&a, &b);
        assert!(approx_eq(s, 0.0));
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let s = EmbeddingSimilarity::cosine(&a, &b);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_cosine_range() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let s = EmbeddingSimilarity::cosine(&a, &b);
        assert!((-1.0..=1.0).contains(&s));
    }

    // ── dot_product ───────────────────────────────────────────────────────────

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = EmbeddingSimilarity::dot_product(&a, &b);
        assert!(approx_eq(d, 32.0));
    }

    #[test]
    fn test_dot_product_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(approx_eq(EmbeddingSimilarity::dot_product(&a, &b), 0.0));
    }

    #[test]
    fn test_dot_product_negative() {
        let a = vec![1.0, -1.0];
        let b = vec![1.0, 1.0];
        assert!(approx_eq(EmbeddingSimilarity::dot_product(&a, &b), 0.0));
    }

    // ── euclidean ─────────────────────────────────────────────────────────────

    #[test]
    fn test_euclidean_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let s = EmbeddingSimilarity::euclidean(&v, &v);
        assert!(approx_eq(s, 1.0)); // distance=0 → similarity=1
    }

    #[test]
    fn test_euclidean_unit_apart() {
        let a = vec![0.0];
        let b = vec![1.0];
        // distance = 1 → similarity = 0.5
        let s = EmbeddingSimilarity::euclidean(&a, &b);
        assert!(approx_eq(s, 0.5));
    }

    #[test]
    fn test_euclidean_positive() {
        let a = vec![1.0, 2.0];
        let b = vec![4.0, 6.0];
        let s = EmbeddingSimilarity::euclidean(&a, &b);
        assert!(s > 0.0 && s < 1.0);
    }

    // ── manhattan ─────────────────────────────────────────────────────────────

    #[test]
    fn test_manhattan_identical() {
        let v = vec![1.0, 2.0];
        let s = EmbeddingSimilarity::manhattan(&v, &v);
        assert!(approx_eq(s, 1.0));
    }

    #[test]
    fn test_manhattan_unit_apart() {
        let a = vec![0.0];
        let b = vec![1.0];
        assert!(approx_eq(EmbeddingSimilarity::manhattan(&a, &b), 0.5));
    }

    #[test]
    fn test_manhattan_positive() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        // L1 distance = 7, similarity = 1/8
        let s = EmbeddingSimilarity::manhattan(&a, &b);
        assert!(approx_eq(s, 1.0 / 8.0));
    }

    // ── chebyshev ─────────────────────────────────────────────────────────────

    #[test]
    fn test_chebyshev_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let s = EmbeddingSimilarity::chebyshev(&v, &v);
        assert!(approx_eq(s, 1.0));
    }

    #[test]
    fn test_chebyshev_picks_max() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 5.0];
        // Chebyshev distance = max(1, 5) = 5, similarity = 1/6
        let s = EmbeddingSimilarity::chebyshev(&a, &b);
        assert!(approx_eq(s, 1.0 / 6.0));
    }

    #[test]
    fn test_chebyshev_positive() {
        let a = vec![1.0, 2.0];
        let b = vec![4.0, 3.0];
        let s = EmbeddingSimilarity::chebyshev(&a, &b);
        assert!(s > 0.0 && s < 1.0);
    }

    // ── compute (dispatch) ────────────────────────────────────────────────────

    #[test]
    fn test_compute_cosine() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        let s = EmbeddingSimilarity::compute(&a, &b, SimilarityMetric::Cosine);
        assert!(approx_eq(s, 1.0));
    }

    #[test]
    fn test_compute_dot_product() {
        let a = vec![2.0, 3.0];
        let b = vec![4.0, 5.0];
        let s = EmbeddingSimilarity::compute(&a, &b, SimilarityMetric::DotProduct);
        assert!(approx_eq(s, 23.0));
    }

    #[test]
    fn test_compute_euclidean() {
        let a = vec![0.0];
        let b = vec![1.0];
        let s = EmbeddingSimilarity::compute(&a, &b, SimilarityMetric::Euclidean);
        assert!(approx_eq(s, 0.5));
    }

    #[test]
    fn test_compute_manhattan() {
        let a = vec![0.0];
        let b = vec![1.0];
        let s = EmbeddingSimilarity::compute(&a, &b, SimilarityMetric::Manhattan);
        assert!(approx_eq(s, 0.5));
    }

    #[test]
    fn test_compute_chebyshev() {
        let a = vec![0.0, 0.0];
        let b = vec![2.0, 3.0];
        let s = EmbeddingSimilarity::compute(&a, &b, SimilarityMetric::Chebyshev);
        assert!(approx_eq(s, 1.0 / 4.0)); // max dist = 3, sim = 1/4
    }

    // ── normalize ─────────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_unit_length() {
        let v = vec![3.0, 4.0];
        let n = EmbeddingSimilarity::normalize(&v);
        let norm: f64 = n.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(approx_eq(norm, 1.0));
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0];
        let n = EmbeddingSimilarity::normalize(&v);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_normalize_already_unit() {
        let v = vec![1.0, 0.0];
        let n = EmbeddingSimilarity::normalize(&v);
        assert!(approx_eq(n[0], 1.0));
        assert!(approx_eq(n[1], 0.0));
    }

    #[test]
    fn test_normalize_preserves_direction() {
        let v = vec![1.0, 1.0];
        let n = EmbeddingSimilarity::normalize(&v);
        assert!(approx_eq(n[0], n[1]));
    }

    // ── top_k ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_top_k_returns_k_results() {
        let query = vec![1.0, 0.0];
        let corpus = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.5, 0.5],
        ];
        let results = EmbeddingSimilarity::top_k(&query, &corpus, 2, SimilarityMetric::Cosine);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_top_k_sorted_descending() {
        let query = vec![1.0, 0.0];
        let corpus = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let results = EmbeddingSimilarity::top_k(&query, &corpus, 3, SimilarityMetric::Cosine);
        for i in 0..results.len() - 1 {
            assert!(results[i].score >= results[i + 1].score);
        }
    }

    #[test]
    fn test_top_k_best_is_identical() {
        let query = vec![1.0, 2.0, 3.0];
        let corpus = vec![vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 1.0]];
        let results = EmbeddingSimilarity::top_k(&query, &corpus, 1, SimilarityMetric::Cosine);
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_top_k_empty_corpus() {
        let query = vec![1.0, 0.0];
        let results = EmbeddingSimilarity::top_k(&query, &[], 5, SimilarityMetric::Euclidean);
        assert!(results.is_empty());
    }

    #[test]
    fn test_top_k_k_larger_than_corpus() {
        let query = vec![1.0];
        let corpus = vec![vec![1.0], vec![2.0]];
        let results = EmbeddingSimilarity::top_k(&query, &corpus, 100, SimilarityMetric::Euclidean);
        assert_eq!(results.len(), 2);
    }

    // ── pairwise ──────────────────────────────────────────────────────────────

    #[test]
    fn test_pairwise_dimensions() {
        let corpus = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let m = EmbeddingSimilarity::pairwise(&corpus, SimilarityMetric::Cosine);
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].len(), 3);
    }

    #[test]
    fn test_pairwise_diagonal_is_max_cosine() {
        let corpus = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let m = EmbeddingSimilarity::pairwise(&corpus, SimilarityMetric::Cosine);
        // Diagonal should be 1.0 (identical vectors)
        assert!(approx_eq(m[0][0], 1.0));
        assert!(approx_eq(m[1][1], 1.0));
    }

    #[test]
    fn test_pairwise_symmetric_cosine() {
        let corpus = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let m = EmbeddingSimilarity::pairwise(&corpus, SimilarityMetric::Cosine);
        assert!(approx_eq(m[0][1], m[1][0]));
    }

    #[test]
    fn test_pairwise_empty_corpus() {
        let m = EmbeddingSimilarity::pairwise(&[], SimilarityMetric::Cosine);
        assert!(m.is_empty());
    }

    // ── SimilarityResult ──────────────────────────────────────────────────────

    #[test]
    fn test_similarity_result_fields() {
        let r = SimilarityResult {
            index: 5,
            score: 0.95,
            label: Some("example".to_string()),
        };
        assert_eq!(r.index, 5);
        assert!((r.score - 0.95).abs() < EPS);
        assert_eq!(r.label, Some("example".to_string()));
    }

    #[test]
    fn test_similarity_result_no_label() {
        let r = SimilarityResult {
            index: 0,
            score: 1.0,
            label: None,
        };
        assert!(r.label.is_none());
    }

    #[test]
    fn test_similarity_result_clone() {
        let r = SimilarityResult {
            index: 1,
            score: 0.5,
            label: None,
        };
        assert_eq!(r, r.clone());
    }

    // ── SimilarityMetric ──────────────────────────────────────────────────────

    #[test]
    fn test_metric_copy() {
        let m = SimilarityMetric::Cosine;
        let m2 = m;
        assert_eq!(m, m2);
    }

    #[test]
    fn test_metric_debug() {
        let s = format!("{:?}", SimilarityMetric::DotProduct);
        assert!(s.contains("DotProduct"));
    }

    #[test]
    fn test_chebyshev_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = EmbeddingSimilarity::compute(&a, &a, SimilarityMetric::Chebyshev);
        // Distance = 0 → similarity = 1/(1+0) = 1.0
        assert!(approx_eq(sim, 1.0));
    }

    #[test]
    fn test_manhattan_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = EmbeddingSimilarity::compute(&a, &b, SimilarityMetric::Manhattan);
        // L1 distance = 2 → similarity = 1/3
        assert!((sim - 1.0 / 3.0).abs() < EPS);
    }

    #[test]
    fn test_dot_product_negative_components() {
        let a = vec![-1.0, -1.0];
        let b = vec![-1.0, -1.0];
        let sim = EmbeddingSimilarity::compute(&a, &b, SimilarityMetric::DotProduct);
        // dot(a, b) = 1+1 = 2
        assert!(approx_eq(sim, 2.0));
    }

    #[test]
    fn test_top_k_all_metrics() {
        let query = vec![1.0, 0.0];
        let corpus = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        for metric in [
            SimilarityMetric::Cosine,
            SimilarityMetric::Euclidean,
            SimilarityMetric::Manhattan,
            SimilarityMetric::Chebyshev,
        ] {
            let results = EmbeddingSimilarity::top_k(&query, &corpus, 2, metric);
            assert_eq!(
                results.len(),
                2,
                "metric {:?} should return 2 results",
                metric
            );
        }
    }

    #[test]
    fn test_similarity_result_debug() {
        let r = SimilarityResult {
            index: 0,
            score: 0.9,
            label: None,
        };
        let s = format!("{r:?}");
        assert!(s.contains("SimilarityResult"));
    }
}
