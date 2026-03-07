//! Multi-vector product search combining multiple embedding sub-vectors.
//!
//! A `ProductSearchIndex` stores `MultiVector` items (each item has multiple
//! sub-vectors of potentially different dimensions) and provides search
//! functionality that computes per-sub-vector scores and combines them.

// ── Types ─────────────────────────────────────────────────────────────────────

/// An item in the index.  `vectors[i]` is the i-th sub-vector.
#[derive(Debug, Clone)]
pub struct MultiVector {
    pub id: usize,
    pub vectors: Vec<Vec<f32>>,
}

/// The distance metric used for scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Squared Euclidean distance (lower is better → negated for combined score).
    L2,
    /// Cosine similarity (higher is better).
    Cosine,
    /// Raw dot-product (higher is better).
    DotProduct,
}

/// Configuration for the index.
#[derive(Debug, Clone)]
pub struct ProductSearchConfig {
    /// Number of sub-vectors per item.
    pub sub_dimensions: usize,
    /// Distance metric used during search.
    pub distance_metric: DistanceMetric,
}

/// A single search result candidate.
#[derive(Debug, Clone)]
pub struct SearchCandidate {
    /// The item's id.
    pub id: usize,
    /// One score per sub-vector.
    pub scores: Vec<f32>,
    /// The arithmetic mean of `scores`.
    pub combined_score: f32,
}

// ── ProductSearchIndex ────────────────────────────────────────────────────────

/// Multi-vector product search index.
pub struct ProductSearchIndex {
    config: ProductSearchConfig,
    items: Vec<MultiVector>,
}

impl ProductSearchIndex {
    /// Create an empty index.
    pub fn new(config: ProductSearchConfig) -> Self {
        Self {
            config,
            items: Vec::new(),
        }
    }

    /// Insert an item into the index.
    pub fn insert(&mut self, item: MultiVector) {
        self.items.push(item);
    }

    /// Search for the `k` nearest items to `query` across all sub-vectors.
    ///
    /// The combined score is the mean of per-sub-vector scores.  Items are
    /// returned sorted by combined score (descending for similarity metrics).
    pub fn search(&self, query: &MultiVector, k: usize) -> Vec<SearchCandidate> {
        let mut candidates: Vec<SearchCandidate> = self
            .items
            .iter()
            .filter_map(|item| self.score_all(query, item))
            .collect();

        // Sort: higher combined score = better (for L2 we negate so still descending)
        candidates.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(k);
        candidates
    }

    /// Search using only the sub-vector at index `sub_idx`.
    pub fn search_sub(&self, query_sub: &[f32], sub_idx: usize, k: usize) -> Vec<SearchCandidate> {
        let mut candidates: Vec<SearchCandidate> = self
            .items
            .iter()
            .filter_map(|item| {
                let item_sub = item.vectors.get(sub_idx)?;
                if item_sub.len() != query_sub.len() {
                    return None;
                }
                let score = self.compute_score(query_sub, item_sub);
                Some(SearchCandidate {
                    id: item.id,
                    scores: vec![score],
                    combined_score: score,
                })
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(k);
        candidates
    }

    /// Number of items in the index.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// The configured number of sub-dimensions.
    pub fn sub_dimension_count(&self) -> usize {
        self.config.sub_dimensions
    }

    /// Remove an item by id.  Returns `true` if the item existed.
    pub fn remove(&mut self, id: usize) -> bool {
        let before = self.items.len();
        self.items.retain(|item| item.id != id);
        self.items.len() < before
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Score `query` against `item` across all matching sub-vectors.
    fn score_all(&self, query: &MultiVector, item: &MultiVector) -> Option<SearchCandidate> {
        let n_subs = query.vectors.len().min(item.vectors.len());
        if n_subs == 0 {
            return None;
        }
        let mut scores: Vec<f32> = Vec::with_capacity(n_subs);
        for i in 0..n_subs {
            let qv = &query.vectors[i];
            let iv = &item.vectors[i];
            if qv.len() != iv.len() {
                return None;
            }
            scores.push(self.compute_score(qv, iv));
        }
        let combined_score = scores.iter().sum::<f32>() / scores.len() as f32;
        Some(SearchCandidate {
            id: item.id,
            scores,
            combined_score,
        })
    }

    /// Compute a single sub-vector score according to the configured metric.
    fn compute_score(&self, a: &[f32], b: &[f32]) -> f32 {
        match &self.config.distance_metric {
            DistanceMetric::L2 => -l2_distance(a, b),
            DistanceMetric::Cosine => cosine_sim(a, b),
            DistanceMetric::DotProduct => dot_product(a, b),
        }
    }
}

// ── Distance/similarity functions ────────────────────────────────────────────

/// Euclidean L2 distance.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Cosine similarity in [−1, 1].
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Raw dot product.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn vec1(v: &[f32]) -> Vec<Vec<f32>> {
        vec![v.to_vec()]
    }

    fn vec2(v1: &[f32], v2: &[f32]) -> Vec<Vec<f32>> {
        vec![v1.to_vec(), v2.to_vec()]
    }

    fn cfg(metric: DistanceMetric) -> ProductSearchConfig {
        ProductSearchConfig {
            sub_dimensions: 1,
            distance_metric: metric,
        }
    }

    fn mv(id: usize, vecs: Vec<Vec<f32>>) -> MultiVector {
        MultiVector { id, vectors: vecs }
    }

    // ── l2_distance ───────────────────────────────────────────────────────────

    #[test]
    fn test_l2_distance_zero() {
        assert!((l2_distance(&[1.0, 2.0], &[1.0, 2.0])).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_known() {
        // 3-4-5 triangle
        assert!((l2_distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-5);
    }

    // ── cosine_sim ────────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_sim_identical() {
        let v = [1.0f32, 0.0, 0.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        assert!((cosine_sim(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_opposite() {
        assert!((cosine_sim(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_zero_vector() {
        assert_eq!(cosine_sim(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    // ── dot_product ───────────────────────────────────────────────────────────

    #[test]
    fn test_dot_product_basic() {
        assert!((dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_zero() {
        assert_eq!(dot_product(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    // ── insert / item_count ───────────────────────────────────────────────────

    #[test]
    fn test_insert_increments_count() {
        let mut idx = ProductSearchIndex::new(cfg(DistanceMetric::L2));
        idx.insert(mv(1, vec1(&[1.0])));
        assert_eq!(idx.item_count(), 1);
    }

    #[test]
    fn test_insert_multiple() {
        let mut idx = ProductSearchIndex::new(cfg(DistanceMetric::L2));
        idx.insert(mv(1, vec1(&[1.0])));
        idx.insert(mv(2, vec1(&[2.0])));
        assert_eq!(idx.item_count(), 2);
    }

    // ── sub_dimension_count ───────────────────────────────────────────────────

    #[test]
    fn test_sub_dimension_count() {
        let idx = ProductSearchIndex::new(ProductSearchConfig {
            sub_dimensions: 3,
            distance_metric: DistanceMetric::Cosine,
        });
        assert_eq!(idx.sub_dimension_count(), 3);
    }

    // ── search L2 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_search_l2_nearest_neighbor() {
        let mut idx = ProductSearchIndex::new(cfg(DistanceMetric::L2));
        idx.insert(mv(1, vec1(&[0.0])));
        idx.insert(mv(2, vec1(&[10.0])));
        let q = mv(0, vec1(&[0.5]));
        let results = idx.search(&q, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1); // closer to 0.0 than 10.0
    }

    #[test]
    fn test_search_l2_same_vector_best_score() {
        let mut idx = ProductSearchIndex::new(cfg(DistanceMetric::L2));
        idx.insert(mv(1, vec1(&[1.0, 2.0, 3.0])));
        idx.insert(mv(2, vec1(&[10.0, 10.0, 10.0])));
        let q = mv(0, vec1(&[1.0, 2.0, 3.0]));
        let results = idx.search(&q, 2);
        assert_eq!(results[0].id, 1); // exact match
    }

    #[test]
    fn test_search_l2_k_limit() {
        let mut idx = ProductSearchIndex::new(cfg(DistanceMetric::L2));
        for i in 0..10usize {
            idx.insert(mv(i, vec1(&[i as f32])));
        }
        let q = mv(99, vec1(&[0.0]));
        let results = idx.search(&q, 3);
        assert_eq!(results.len(), 3);
    }

    // ── search Cosine ─────────────────────────────────────────────────────────

    #[test]
    fn test_search_cosine_identical_is_top() {
        let mut idx = ProductSearchIndex::new(cfg(DistanceMetric::Cosine));
        idx.insert(mv(1, vec1(&[1.0, 0.0])));
        idx.insert(mv(2, vec1(&[0.0, 1.0])));
        let q = mv(0, vec1(&[1.0, 0.0]));
        let results = idx.search(&q, 2);
        assert_eq!(results[0].id, 1);
    }

    // ── search DotProduct ─────────────────────────────────────────────────────

    #[test]
    fn test_search_dot_product() {
        let mut idx = ProductSearchIndex::new(cfg(DistanceMetric::DotProduct));
        idx.insert(mv(1, vec1(&[1.0, 2.0])));
        idx.insert(mv(2, vec1(&[3.0, 4.0])));
        let q = mv(0, vec1(&[1.0, 1.0]));
        let results = idx.search(&q, 2);
        // dot([1,1],[3,4])=7 > dot([1,1],[1,2])=3 → item 2 first
        assert_eq!(results[0].id, 2);
    }

    // ── multi-vector combination ──────────────────────────────────────────────

    #[test]
    fn test_search_multi_vector_combination() {
        let mut idx = ProductSearchIndex::new(ProductSearchConfig {
            sub_dimensions: 2,
            distance_metric: DistanceMetric::Cosine,
        });
        // Item 1: very similar in sub0, very dissimilar in sub1
        idx.insert(mv(1, vec2(&[1.0, 0.0], &[0.0, 1.0])));
        // Item 2: similar in both sub0 and sub1
        idx.insert(mv(2, vec2(&[1.0, 0.0], &[1.0, 0.0])));
        let q = mv(0, vec2(&[1.0, 0.0], &[1.0, 0.0]));
        let results = idx.search(&q, 2);
        // Item 2 should have higher combined score
        assert_eq!(results[0].id, 2);
    }

    #[test]
    fn test_search_candidate_scores_count_equals_sub_vectors() {
        let mut idx = ProductSearchIndex::new(ProductSearchConfig {
            sub_dimensions: 3,
            distance_metric: DistanceMetric::Cosine,
        });
        idx.insert(mv(1, vec![vec![1.0], vec![1.0], vec![1.0]]));
        let q = mv(0, vec![vec![1.0], vec![1.0], vec![1.0]]);
        let results = idx.search(&q, 1);
        assert_eq!(results[0].scores.len(), 3);
    }

    // ── search_sub ────────────────────────────────────────────────────────────

    #[test]
    fn test_search_sub_single_dimension() {
        let mut idx = ProductSearchIndex::new(ProductSearchConfig {
            sub_dimensions: 2,
            distance_metric: DistanceMetric::L2,
        });
        idx.insert(mv(1, vec2(&[0.0], &[10.0])));
        idx.insert(mv(2, vec2(&[5.0], &[10.0])));
        let results = idx.search_sub(&[0.0], 0, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1); // item 1 has [0.0] in sub0
    }

    #[test]
    fn test_search_sub_k_limit() {
        let mut idx = ProductSearchIndex::new(ProductSearchConfig {
            sub_dimensions: 1,
            distance_metric: DistanceMetric::Cosine,
        });
        for i in 0..5usize {
            idx.insert(mv(i, vec1(&[i as f32 + 1.0])));
        }
        let results = idx.search_sub(&[1.0], 0, 2);
        assert_eq!(results.len(), 2);
    }

    // ── remove ────────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_item() {
        let mut idx = ProductSearchIndex::new(cfg(DistanceMetric::L2));
        idx.insert(mv(42, vec1(&[1.0])));
        assert!(idx.remove(42));
        assert_eq!(idx.item_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_item() {
        let mut idx = ProductSearchIndex::new(cfg(DistanceMetric::L2));
        assert!(!idx.remove(99));
    }

    #[test]
    fn test_remove_does_not_affect_other_items() {
        let mut idx = ProductSearchIndex::new(cfg(DistanceMetric::L2));
        idx.insert(mv(1, vec1(&[1.0])));
        idx.insert(mv(2, vec1(&[2.0])));
        idx.remove(1);
        assert_eq!(idx.item_count(), 1);
        let q = mv(0, vec1(&[2.0]));
        let results = idx.search(&q, 1);
        assert_eq!(results[0].id, 2);
    }

    // ── empty index ───────────────────────────────────────────────────────────

    #[test]
    fn test_search_empty_index() {
        let idx = ProductSearchIndex::new(cfg(DistanceMetric::L2));
        let q = mv(0, vec1(&[1.0]));
        let results = idx.search(&q, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_sub_empty_index() {
        let idx = ProductSearchIndex::new(cfg(DistanceMetric::L2));
        let results = idx.search_sub(&[1.0], 0, 5);
        assert!(results.is_empty());
    }

    // ── combined_score correctness ────────────────────────────────────────────

    #[test]
    fn test_combined_score_is_mean_of_scores() {
        let mut idx = ProductSearchIndex::new(ProductSearchConfig {
            sub_dimensions: 2,
            distance_metric: DistanceMetric::Cosine,
        });
        idx.insert(mv(1, vec2(&[1.0, 0.0], &[1.0, 0.0])));
        let q = mv(0, vec2(&[1.0, 0.0], &[1.0, 0.0]));
        let results = idx.search(&q, 1);
        let c = &results[0];
        let expected = c.scores.iter().sum::<f32>() / c.scores.len() as f32;
        assert!((c.combined_score - expected).abs() < 1e-5);
    }
}
