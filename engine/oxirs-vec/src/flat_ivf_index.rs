//! Flat IVF (Inverted File Index) for approximate nearest-neighbour search.
//!
//! Uses k-means clustering (Lloyd's algorithm) to partition the vector space
//! into `num_cells` Voronoi cells.  Search probes the `n_probe` nearest cells.

// ── Types ─────────────────────────────────────────────────────────────────────

/// A cluster centroid.
#[derive(Debug, Clone)]
pub struct Centroid {
    pub id: usize,
    pub vector: Vec<f32>,
}

/// One inverted-file cell: the centroid id, plus all vectors assigned to it.
#[derive(Debug, Clone, Default)]
pub struct IvfCell {
    pub centroid_id: usize,
    pub vector_ids: Vec<u64>,
    pub vectors: Vec<Vec<f32>>,
}

/// A single search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
}

/// Flat IVF index.
pub struct FlatIvfIndex {
    pub dim: usize,
    pub num_cells: usize,
    pub cells: Vec<IvfCell>,
    pub centroids: Vec<Centroid>,
}

impl FlatIvfIndex {
    /// Create an untrained index with `num_cells` cells and vectors of dimension `dim`.
    pub fn new(dim: usize, num_cells: usize) -> Self {
        let cells: Vec<IvfCell> = (0..num_cells)
            .map(|id| IvfCell {
                centroid_id: id,
                vector_ids: Vec::new(),
                vectors: Vec::new(),
            })
            .collect();
        FlatIvfIndex {
            dim,
            num_cells,
            cells,
            centroids: Vec::new(),
        }
    }

    // ── Training ──────────────────────────────────────────────────────────

    /// Train the index using Lloyd's k-means (up to 20 iterations) on `vectors`.
    ///
    /// Centroids are initialised by selecting evenly-spaced samples from the input.
    pub fn train(&mut self, vectors: &[Vec<f32>]) {
        if vectors.is_empty() || self.num_cells == 0 {
            return;
        }
        let k = self.num_cells.min(vectors.len());

        // --- Initialise centroids from evenly-spaced samples ---
        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|i| {
                let idx = (i * vectors.len()) / k;
                vectors[idx].clone()
            })
            .collect();

        // --- Lloyd's iterations ---
        for _ in 0..20 {
            // Assign each vector to its nearest centroid.
            let assignments: Vec<usize> = vectors
                .iter()
                .map(|v| Self::nearest_centroid_from_list(&centroids, v))
                .collect();

            // Recompute centroids as cluster means.
            let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0f32; self.dim]; k];
            let mut counts: Vec<usize> = vec![0; k];

            for (v, &c) in vectors.iter().zip(assignments.iter()) {
                for (d, x) in new_centroids[c].iter_mut().zip(v.iter()) {
                    *d += x;
                }
                counts[c] += 1;
            }

            let mut converged = true;
            for c in 0..k {
                if counts[c] == 0 {
                    // Keep old centroid if empty cluster.
                    new_centroids[c] = centroids[c].clone();
                } else {
                    for d in new_centroids[c].iter_mut() {
                        *d /= counts[c] as f32;
                    }
                }
                let change = Self::l2_distance(&centroids[c], &new_centroids[c]);
                if change > 1e-6 {
                    converged = false;
                }
            }
            centroids = new_centroids;
            if converged {
                break;
            }
        }

        // Store trained centroids.
        self.centroids = centroids
            .into_iter()
            .enumerate()
            .map(|(id, vector)| Centroid { id, vector })
            .collect();

        // Reset cells with updated centroids.
        self.cells = (0..k)
            .map(|id| IvfCell {
                centroid_id: id,
                vector_ids: Vec::new(),
                vectors: Vec::new(),
            })
            .collect();
        self.num_cells = k;
    }

    // ── Insertion ─────────────────────────────────────────────────────────

    /// Insert a vector with the given id into the nearest cell.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        let cell_idx = self.nearest_centroid(&vector);
        let cell = &mut self.cells[cell_idx];
        cell.vector_ids.push(id);
        cell.vectors.push(vector);
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Search for the `k` nearest neighbours of `query`, probing `n_probe` cells.
    pub fn search(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<SearchResult> {
        if self.centroids.is_empty() || k == 0 {
            return Vec::new();
        }

        let n_probe = n_probe.min(self.num_cells);

        // Find the `n_probe` nearest centroids.
        let mut centroid_dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .map(|c| (c.id, Self::l2_distance(query, &c.vector)))
            .collect();
        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Collect candidates from the top `n_probe` cells.
        let mut candidates: Vec<SearchResult> = Vec::new();
        for (cell_id, _) in centroid_dists.iter().take(n_probe) {
            let cell = &self.cells[*cell_id];
            for (vec_id, vec) in cell.vector_ids.iter().zip(cell.vectors.iter()) {
                let dist = Self::l2_distance(query, vec);
                candidates.push(SearchResult {
                    id: *vec_id,
                    distance: dist,
                });
            }
        }

        // Sort by distance and return top-k.
        candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(k);
        candidates
    }

    // ── Removal ───────────────────────────────────────────────────────────

    /// Remove the vector with `id` from the index. Returns `true` if found.
    pub fn remove(&mut self, id: u64) -> bool {
        for cell in &mut self.cells {
            if let Some(pos) = cell.vector_ids.iter().position(|&x| x == id) {
                cell.vector_ids.remove(pos);
                cell.vectors.remove(pos);
                return true;
            }
        }
        false
    }

    // ── Metadata ──────────────────────────────────────────────────────────

    /// Total number of vectors in the index.
    pub fn len(&self) -> usize {
        self.cells.iter().map(|c| c.vector_ids.len()).sum()
    }

    /// Returns `true` if no vectors are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Return the index of the nearest cell for `vec`.
    pub fn nearest_centroid(&self, vec: &[f32]) -> usize {
        if self.centroids.is_empty() {
            // Fall back to modulo assignment before training.
            return 0;
        }
        Self::nearest_centroid_from_list(
            &self
                .centroids
                .iter()
                .map(|c| c.vector.clone())
                .collect::<Vec<_>>(),
            vec,
        )
    }

    fn nearest_centroid_from_list(centroids: &[Vec<f32>], vec: &[f32]) -> usize {
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, Self::l2_distance(vec, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute the squared L2 distance between two equal-length slices.
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, val: f32) -> Vec<f32> {
        vec![val; dim]
    }

    // ── Construction ──────────────────────────────────────────────────────

    #[test]
    fn test_new_index() {
        let idx = FlatIvfIndex::new(4, 3);
        assert_eq!(idx.dim, 4);
        assert_eq!(idx.num_cells, 3);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    // ── Training ──────────────────────────────────────────────────────────

    #[test]
    fn test_train_basic() {
        let mut idx = FlatIvfIndex::new(2, 2);
        let vecs: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        idx.train(&vecs);
        assert_eq!(idx.centroids.len(), 2);
    }

    #[test]
    fn test_train_empty() {
        let mut idx = FlatIvfIndex::new(2, 3);
        idx.train(&[]);
        assert!(idx.centroids.is_empty());
    }

    #[test]
    fn test_train_fewer_vectors_than_cells() {
        let mut idx = FlatIvfIndex::new(2, 10);
        let vecs = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
        idx.train(&vecs);
        assert!(idx.centroids.len() <= 2);
    }

    // ── Insert / len / is_empty ───────────────────────────────────────────

    #[test]
    fn test_insert_and_len() {
        let mut idx = FlatIvfIndex::new(2, 2);
        let vecs = vec![vec![0.0f32, 0.0], vec![10.0, 10.0]];
        idx.train(&vecs);
        idx.insert(1, vec![0.0, 0.0]);
        idx.insert(2, vec![10.0, 10.0]);
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
    }

    #[test]
    fn test_insert_many() {
        let mut idx = FlatIvfIndex::new(1, 3);
        let vecs: Vec<Vec<f32>> = (0..30).map(|i| vec![i as f32]).collect();
        idx.train(&vecs);
        for i in 0u64..30 {
            idx.insert(i, vec![i as f32]);
        }
        assert_eq!(idx.len(), 30);
    }

    // ── Remove ────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing() {
        let mut idx = FlatIvfIndex::new(2, 2);
        idx.train(&[vec![0.0f32, 0.0], vec![5.0, 5.0]]);
        idx.insert(42, vec![0.0, 0.0]);
        assert!(idx.remove(42));
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut idx = FlatIvfIndex::new(2, 2);
        idx.train(&[vec![0.0f32, 0.0], vec![5.0, 5.0]]);
        assert!(!idx.remove(999));
    }

    #[test]
    fn test_remove_and_search() {
        let mut idx = FlatIvfIndex::new(1, 2);
        idx.train(&[vec![0.0f32], vec![10.0]]);
        idx.insert(1, vec![0.0]);
        idx.insert(2, vec![10.0]);
        idx.remove(1);
        let results = idx.search(&[0.0], 10, 2);
        assert!(!results.iter().any(|r| r.id == 1));
    }

    // ── Search ────────────────────────────────────────────────────────────

    #[test]
    fn test_search_nearest() {
        let mut idx = FlatIvfIndex::new(1, 2);
        let train_vecs = vec![vec![0.0f32], vec![100.0]];
        idx.train(&train_vecs);
        idx.insert(0, vec![0.0]);
        idx.insert(1, vec![1.0]);
        idx.insert(2, vec![100.0]);
        let results = idx.search(&[0.5], 1, 1);
        assert_eq!(results.len(), 1);
        // Either 0 or 1 is nearest; both are close to 0.5.
        assert!(results[0].id == 0 || results[0].id == 1);
    }

    #[test]
    fn test_search_k_results() {
        let mut idx = FlatIvfIndex::new(1, 2);
        let vecs: Vec<Vec<f32>> = vec![vec![0.0], vec![100.0]];
        idx.train(&vecs);
        for i in 0u64..5 {
            idx.insert(i, vec![i as f32]);
        }
        let results = idx.search(&[0.0], 3, 2);
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_search_k_0_returns_empty() {
        let mut idx = FlatIvfIndex::new(1, 2);
        idx.train(&[vec![0.0f32], vec![1.0]]);
        idx.insert(0, vec![0.0]);
        let results = idx.search(&[0.0], 0, 1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_empty_index() {
        let idx = FlatIvfIndex::new(2, 3);
        let results = idx.search(&[0.0, 0.0], 5, 2);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_n_probe_all_cells() {
        let mut idx = FlatIvfIndex::new(1, 3);
        let train_vecs: Vec<Vec<f32>> = vec![vec![0.0], vec![5.0], vec![10.0]];
        idx.train(&train_vecs);
        idx.insert(0, vec![0.0]);
        idx.insert(1, vec![5.0]);
        idx.insert(2, vec![10.0]);
        let results = idx.search(&[5.0], 3, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_sorted_by_distance() {
        let mut idx = FlatIvfIndex::new(1, 2);
        idx.train(&[vec![0.0f32], vec![10.0]]);
        idx.insert(0, vec![0.0]);
        idx.insert(1, vec![3.0]);
        idx.insert(2, vec![10.0]);
        let results = idx.search(&[0.0], 3, 2);
        for i in 1..results.len() {
            assert!(results[i - 1].distance <= results[i].distance);
        }
    }

    // ── l2_distance ───────────────────────────────────────────────────────

    #[test]
    fn test_l2_distance_zero() {
        let a = vec![1.0f32, 2.0, 3.0];
        assert!((FlatIvfIndex::l2_distance(&a, &a)).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_unit_vector() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 0.0];
        assert!((FlatIvfIndex::l2_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_symmetric() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let d1 = FlatIvfIndex::l2_distance(&a, &b);
        let d2 = FlatIvfIndex::l2_distance(&b, &a);
        assert!((d1 - d2).abs() < 1e-6);
    }

    // ── nearest_centroid ──────────────────────────────────────────────────

    #[test]
    fn test_nearest_centroid_basic() {
        let mut idx = FlatIvfIndex::new(1, 2);
        idx.train(&[vec![0.0f32], vec![100.0]]);
        let near_zero = idx.nearest_centroid(&[1.0]);
        let near_hundred = idx.nearest_centroid(&[99.0]);
        assert_ne!(near_zero, near_hundred);
    }

    // ── n_probe variation ─────────────────────────────────────────────────

    #[test]
    fn test_n_probe_1_vs_all() {
        let mut idx = FlatIvfIndex::new(1, 4);
        let tv: Vec<Vec<f32>> = vec![vec![0.0], vec![10.0], vec![20.0], vec![30.0]];
        idx.train(&tv);
        for i in 0..8u64 {
            idx.insert(i, vec![(i as f32) * 5.0]);
        }
        let r1 = idx.search(&[15.0], 8, 1);
        let r_all = idx.search(&[15.0], 8, 4);
        // Probing all cells should find at least as many results.
        assert!(r_all.len() >= r1.len());
    }

    // ── 2D vectors ────────────────────────────────────────────────────────

    #[test]
    fn test_2d_cluster_separation() {
        let mut idx = FlatIvfIndex::new(2, 2);
        let tv = vec![
            vec![0.0f32, 0.0],
            vec![0.5, 0.5],
            vec![100.0, 100.0],
            vec![100.5, 100.5],
        ];
        idx.train(&tv);
        idx.insert(10, vec![0.2, 0.2]);
        idx.insert(11, vec![100.2, 100.2]);

        let results = idx.search(&[0.1, 0.1], 1, 1);
        if !results.is_empty() {
            assert_eq!(results[0].id, 10);
        }
    }

    #[test]
    fn test_exact_match() {
        let mut idx = FlatIvfIndex::new(3, 2);
        idx.train(&[vec![1.0f32, 2.0, 3.0], vec![10.0, 20.0, 30.0]]);
        idx.insert(99, vec![5.0, 5.0, 5.0]);
        let query = vec![5.0f32, 5.0, 5.0];
        let results = idx.search(&query, 1, 2);
        assert!(!results.is_empty());
        assert!((results[0].distance).abs() < 1e-5);
        assert_eq!(results[0].id, 99);
    }
}
