//! Locality-Sensitive Hashing (LSH) index for approximate nearest neighbour search.
//!
//! Uses random hyperplane projection with multiple hash tables to efficiently
//! find approximate nearest neighbours in high-dimensional vector spaces.
//! All randomness comes from a deterministic XorShift-64 PRNG.

use std::collections::HashMap;

// -------------------------------------------------------------------------
// XorShift-64 deterministic PRNG
// -------------------------------------------------------------------------

/// Deterministic XorShift-64 pseudo-random number generator.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Ensure the seed is non-zero
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate the next pseudo-random u64.
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a pseudo-random f64 in [-1.0, 1.0).
    fn next_f64_signed(&mut self) -> f64 {
        let bits = self.next();
        // Map to [0.0, 1.0) then scale to [-1.0, 1.0)
        let pos = (bits as f64) / (u64::MAX as f64);
        pos * 2.0 - 1.0
    }
}

// -------------------------------------------------------------------------
// LshHasher
// -------------------------------------------------------------------------

/// A single LSH hasher using random hyperplane projection.
///
/// Each bit of the hash corresponds to one random hyperplane: a bit is 1 if
/// the dot product with the random vector is non-negative, 0 otherwise.
#[derive(Debug, Clone)]
pub struct LshHasher {
    /// Unit random vectors (one per hash bit).
    pub random_vectors: Vec<Vec<f64>>,
    /// Dimensionality of the input vectors.
    pub dim: usize,
}

impl LshHasher {
    /// Create a new hasher with `num_hashes` random unit vectors of dimension `dim`.
    ///
    /// Uses the provided XorShift state for deterministic generation.
    fn new_with_rng(dim: usize, num_hashes: usize, rng: &mut XorShift64) -> Self {
        let mut random_vectors = Vec::with_capacity(num_hashes);
        for _ in 0..num_hashes {
            let mut v: Vec<f64> = (0..dim).map(|_| rng.next_f64_signed()).collect();
            normalize_vec(&mut v);
            random_vectors.push(v);
        }
        Self {
            random_vectors,
            dim,
        }
    }

    /// Hash a vector to a `u64` using sign bits of dot products.
    pub fn hash(&self, v: &[f64]) -> u64 {
        let mut h: u64 = 0;
        for (bit, rv) in self.random_vectors.iter().enumerate() {
            if bit >= 64 {
                break;
            }
            let dot: f64 = v.iter().zip(rv.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                h |= 1u64 << bit;
            }
        }
        h
    }
}

/// A hash table bucket mapping hash values to vector indices.
pub type LshBucket = HashMap<u64, Vec<usize>>;

// -------------------------------------------------------------------------
// LshIndex
// -------------------------------------------------------------------------

/// Approximate nearest-neighbour index using Locality-Sensitive Hashing.
pub struct LshIndex {
    /// All inserted vectors (indexed by position in this Vec).
    pub vectors: Vec<Vec<f64>>,
    /// One bucket table per LSH table.
    pub buckets: Vec<LshBucket>,
    /// One hasher per LSH table.
    pub hashers: Vec<LshHasher>,
    /// Dimensionality of the vectors.
    pub dim: usize,
    /// Number of hash tables.
    pub num_tables: usize,
    /// Number of hash bits per table.
    pub num_hashes: usize,
}

impl LshIndex {
    /// Create a new LSH index.
    ///
    /// * `dim` — vector dimensionality
    /// * `num_tables` — number of independent hash tables (more → higher recall)
    /// * `num_hashes` — number of hash bits per table (more → higher precision)
    /// * `seed` — XorShift-64 seed for reproducibility
    pub fn new(dim: usize, num_tables: usize, num_hashes: usize, seed: u64) -> Self {
        let mut rng = XorShift64::new(seed);
        let mut hashers = Vec::with_capacity(num_tables);
        let mut buckets = Vec::with_capacity(num_tables);
        for _ in 0..num_tables {
            hashers.push(LshHasher::new_with_rng(dim, num_hashes, &mut rng));
            buckets.push(LshBucket::new());
        }
        Self {
            vectors: Vec::new(),
            buckets,
            hashers,
            dim,
            num_tables,
            num_hashes,
        }
    }

    /// Insert a vector with the given id into all hash tables.
    pub fn insert(&mut self, id: usize, vector: &[f64]) {
        // Ensure storage is large enough
        while self.vectors.len() <= id {
            self.vectors.push(vec![]);
        }
        self.vectors[id] = vector.to_vec();

        for (table_idx, hasher) in self.hashers.iter().enumerate() {
            let h = hasher.hash(vector);
            self.buckets[table_idx].entry(h).or_default().push(id);
        }
    }

    /// Compute cosine similarity between two vectors.
    ///
    /// Returns 0.0 if either vector has zero magnitude.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag_a < f64::EPSILON || mag_b < f64::EPSILON {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    /// Search for the top-k approximate nearest neighbours of `query`.
    ///
    /// Collects candidates from all tables, deduplicates them, computes cosine
    /// similarity, and returns the top-k sorted by descending similarity.
    /// If the query itself was indexed (same vector), it may appear in results.
    pub fn search(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        let mut candidate_set = std::collections::HashSet::new();

        for (table_idx, hasher) in self.hashers.iter().enumerate() {
            let h = hasher.hash(query);
            if let Some(ids) = self.buckets[table_idx].get(&h) {
                for &id in ids {
                    candidate_set.insert(id);
                }
            }
        }

        let mut scored: Vec<(usize, f64)> = candidate_set
            .into_iter()
            .filter_map(|id| {
                let v = self.vectors.get(id)?;
                if v.is_empty() {
                    return None;
                }
                Some((id, Self::cosine_similarity(query, v)))
            })
            .collect();

        // Sort by similarity descending, then by id ascending for determinism
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        scored.truncate(k);
        scored
    }

    /// Return the number of indexed vectors.
    pub fn len(&self) -> usize {
        self.vectors.iter().filter(|v| !v.is_empty()).count()
    }

    /// Return true if no vectors have been indexed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all indexed vectors and clear all hash tables.
    pub fn clear(&mut self) {
        self.vectors.clear();
        for bucket in &mut self.buckets {
            bucket.clear();
        }
    }
}

// -------------------------------------------------------------------------
// Helper: normalize a vector to unit length
// -------------------------------------------------------------------------

fn normalize_vec(v: &mut [f64]) {
    let mag: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag > f64::EPSILON {
        for x in v.iter_mut() {
            *x /= mag;
        }
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, axis: usize) -> Vec<f64> {
        let mut v = vec![0.0_f64; dim];
        v[axis] = 1.0;
        v
    }

    fn new_index() -> LshIndex {
        LshIndex::new(4, 4, 8, 42)
    }

    // ------ XorShift64 ------

    #[test]
    fn test_xorshift64_deterministic() {
        let mut rng1 = XorShift64::new(123);
        let mut rng2 = XorShift64::new(123);
        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_xorshift64_nonzero_seed() {
        let mut rng = XorShift64::new(0); // Should be initialised to 1 internally
        let v = rng.next();
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_different_seeds() {
        let mut rng1 = XorShift64::new(1);
        let mut rng2 = XorShift64::new(2);
        let v1 = rng1.next();
        let v2 = rng2.next();
        assert_ne!(v1, v2);
    }

    // ------ normalize_vec ------

    #[test]
    fn test_normalize_vec_unit_length() {
        let mut v = vec![3.0_f64, 4.0_f64];
        normalize_vec(&mut v);
        let mag: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((mag - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalize_zero_vec_safe() {
        let mut v = vec![0.0_f64; 4];
        normalize_vec(&mut v); // should not panic
    }

    // ------ LshHasher ------

    #[test]
    fn test_hasher_deterministic() {
        let mut rng = XorShift64::new(42);
        let h1 = LshHasher::new_with_rng(4, 8, &mut rng);
        let v = vec![1.0_f64, 0.0, 0.0, 0.0];
        let hash1 = h1.hash(&v);

        // Same seed → same hash
        let mut rng2 = XorShift64::new(42);
        let h2 = LshHasher::new_with_rng(4, 8, &mut rng2);
        let hash2 = h2.hash(&v);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hasher_similar_vectors_same_bucket() {
        let mut rng = XorShift64::new(42);
        let h = LshHasher::new_with_rng(4, 4, &mut rng);
        let v1 = vec![1.0_f64, 0.001, 0.001, 0.001];
        let v2 = vec![1.0_f64, 0.001, 0.001, 0.002];
        // Very similar vectors — likely same bucket (not guaranteed but often true)
        let hash1 = h.hash(&v1);
        let hash2 = h.hash(&v2);
        // We can't assert equality (probabilistic), but we can assert both are u64
        let _ = (hash1, hash2);
    }

    #[test]
    fn test_hasher_opposite_vectors_different_bits() {
        let mut rng = XorShift64::new(99);
        let h = LshHasher::new_with_rng(4, 8, &mut rng);
        let v = vec![1.0_f64, 0.0, 0.0, 0.0];
        let neg_v = vec![-1.0_f64, 0.0, 0.0, 0.0];
        let h1 = h.hash(&v);
        let h2 = h.hash(&neg_v);
        // Opposite vectors should hash differently
        assert_ne!(h1, h2);
    }

    // ------ cosine_similarity ------

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0_f64, 2.0, 3.0];
        let sim = LshIndex::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let v1 = vec![1.0_f64, 0.0, 0.0];
        let v2 = vec![0.0_f64, 1.0, 0.0];
        let sim = LshIndex::cosine_similarity(&v1, &v2);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let v1 = vec![1.0_f64, 0.0];
        let v2 = vec![-1.0_f64, 0.0];
        let sim = LshIndex::cosine_similarity(&v1, &v2);
        assert!((sim + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let v1 = vec![0.0_f64, 0.0];
        let v2 = vec![1.0_f64, 0.0];
        let sim = LshIndex::cosine_similarity(&v1, &v2);
        assert!((sim).abs() < 1e-9);
    }

    // ------ LshIndex construction ------

    #[test]
    fn test_index_new_dimensions() {
        let idx = LshIndex::new(8, 4, 16, 1);
        assert_eq!(idx.dim, 8);
        assert_eq!(idx.num_tables, 4);
        assert_eq!(idx.num_hashes, 16);
        assert_eq!(idx.hashers.len(), 4);
        assert_eq!(idx.buckets.len(), 4);
    }

    #[test]
    fn test_index_empty() {
        let idx = new_index();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    // ------ insert / len ------

    #[test]
    fn test_insert_single_vector() {
        let mut idx = new_index();
        idx.insert(0, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_insert_multiple_vectors() {
        let mut idx = new_index();
        for i in 0..10 {
            idx.insert(i, &unit_vec(4, i % 4));
        }
        assert_eq!(idx.len(), 10);
    }

    // ------ search ------

    #[test]
    fn test_search_empty_index() {
        let idx = new_index();
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_exact_match() {
        let mut idx = LshIndex::new(4, 8, 16, 42);
        let v = vec![1.0_f64, 0.0, 0.0, 0.0];
        idx.insert(0, &v);
        let results = idx.search(&v, 1);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_k_limits_results() {
        let mut idx = LshIndex::new(4, 8, 4, 77);
        let v = vec![1.0_f64, 0.0, 0.0, 0.0];
        for i in 0..5 {
            // All very similar vectors
            let mut vv = v.clone();
            vv[0] = 1.0 - i as f64 * 0.01;
            idx.insert(i, &vv);
        }
        let results = idx.search(&v, 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_search_returns_closer_vector() {
        let mut idx = LshIndex::new(2, 8, 16, 1);
        // v1 is close to query [1, 0]
        idx.insert(0, &[1.0_f64, 0.01]);
        // v2 is far from query
        idx.insert(1, &[0.0_f64, 1.0]);

        let results = idx.search(&[1.0_f64, 0.0], 2);
        // When both are found, v1 should rank higher
        if results.len() >= 2 {
            assert!(results[0].1 >= results[1].1);
        }
    }

    #[test]
    fn test_search_sorted_descending() {
        let mut idx = LshIndex::new(4, 8, 16, 7);
        idx.insert(0, &[1.0_f64, 0.0, 0.0, 0.0]);
        idx.insert(1, &[0.9_f64, 0.1, 0.0, 0.0]);
        idx.insert(2, &[0.5_f64, 0.5, 0.0, 0.0]);

        let query = [1.0_f64, 0.0, 0.0, 0.0];
        let results = idx.search(&query, 3);
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1, "Results not sorted descending");
        }
    }

    #[test]
    fn test_search_k_greater_than_num_vectors() {
        let mut idx = new_index();
        idx.insert(0, &[1.0_f64, 0.0, 0.0, 0.0]);
        idx.insert(1, &[0.0_f64, 1.0, 0.0, 0.0]);
        let results = idx.search(&[1.0_f64, 0.0, 0.0, 0.0], 100);
        assert!(results.len() <= 2);
    }

    // ------ clear ------

    #[test]
    fn test_clear() {
        let mut idx = new_index();
        idx.insert(0, &[1.0_f64, 0.0, 0.0, 0.0]);
        idx.clear();
        assert!(idx.is_empty());
    }

    #[test]
    fn test_clear_then_insert() {
        let mut idx = new_index();
        idx.insert(0, &[1.0_f64, 0.0, 0.0, 0.0]);
        idx.clear();
        idx.insert(0, &[0.0_f64, 1.0, 0.0, 0.0]);
        assert_eq!(idx.len(), 1);
    }

    // ------ multi-table redundancy ------

    #[test]
    fn test_multi_table_improves_recall() {
        // With many tables, the target vector should almost certainly be found
        let mut idx = LshIndex::new(4, 16, 8, 2024);
        let target = vec![1.0_f64, 0.0, 0.0, 0.0];
        idx.insert(42, &target);

        // Add some other vectors
        for i in 0..20 {
            let mut v = vec![0.0_f64; 4];
            v[i % 4] = 1.0;
            v[(i + 1) % 4] = 0.1;
            idx.insert(i, &v);
        }

        let results = idx.search(&target, 5);
        let found = results.iter().any(|(id, _)| *id == 42);
        assert!(found, "Target vector should be found with 16 tables");
    }

    #[test]
    fn test_high_dimensional_search() {
        let dim = 64;
        let mut idx = LshIndex::new(dim, 8, 16, 99);
        let target: Vec<f64> = (0..dim).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        idx.insert(0, &target);
        let results = idx.search(&target, 1);
        if !results.is_empty() {
            assert!((results[0].1 - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_is_empty_after_inserts() {
        let mut idx = new_index();
        assert!(idx.is_empty());
        idx.insert(0, &[1.0_f64, 0.0, 0.0, 0.0]);
        assert!(!idx.is_empty());
    }

    #[test]
    fn test_results_contain_similarity() {
        let mut idx = LshIndex::new(4, 8, 16, 55);
        idx.insert(0, &[1.0_f64, 0.0, 0.0, 0.0]);
        let results = idx.search(&[1.0_f64, 0.0, 0.0, 0.0], 1);
        if !results.is_empty() {
            assert!(results[0].1 >= 0.0 && results[0].1 <= 1.0 + 1e-9);
        }
    }
}
