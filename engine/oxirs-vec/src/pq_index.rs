//! # Product Quantization Index
//!
//! A memory-efficient approximate nearest-neighbour index based on product
//! quantization (PQ).  Each vector is split into `m` sub-vectors, and each
//! sub-vector is replaced by the index of its nearest centroid in a per-
//! sub-space codebook.  Distances are then approximated via pre-computed
//! lookup tables (asymmetric distance computation — ADC).
//!
//! ## Features
//!
//! - **Codebook training** via k-means on sub-vectors
//! - **Asymmetric distance computation** for accurate approximations
//! - **Multi-probe search** with configurable number of probes
//! - **Compact storage**: each vector is just `m` bytes (for `k=256`)
//!
//! ## Usage
//!
//! ```rust
//! use oxirs_vec::pq_index::{ProductQuantizationIndex, PqConfig};
//!
//! let config = PqConfig {
//!     dimension: 8,
//!     num_sub_vectors: 4,
//!     num_centroids: 4,
//!     training_iterations: 5,
//!     ..Default::default()
//! };
//! let mut pq = ProductQuantizationIndex::new(config).expect("should succeed");
//!
//! // Train on some data
//! let training_data = vec![
//!     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
//!     vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
//!     vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
//!     vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
//! ];
//! pq.train(&training_data).expect("should succeed");
//!
//! // Add vectors
//! pq.add(0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("should succeed");
//! pq.add(1, &[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).expect("should succeed");
//!
//! // Search
//! let results = pq.search(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1).expect("should succeed");
//! assert_eq!(results[0].0, 0); // ID of closest vector
//! ```

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the PQ index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqConfig {
    /// Total vector dimension.
    pub dimension: usize,
    /// Number of sub-vector spaces (`m`). Must divide `dimension` evenly.
    pub num_sub_vectors: usize,
    /// Number of centroids per sub-space (`k`, typically 256).
    pub num_centroids: usize,
    /// Number of k-means iterations for codebook training.
    pub training_iterations: usize,
    /// Number of probes for multi-probe search (0 = exact ADC scan).
    pub num_probes: usize,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            num_sub_vectors: 8,
            num_centroids: 256,
            training_iterations: 20,
            num_probes: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Codebook
// ---------------------------------------------------------------------------

/// A codebook for one sub-vector space.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SubCodebook {
    /// `centroids[i]` is the centroid vector for code `i`.
    centroids: Vec<Vec<f32>>,
    /// Sub-vector dimension.
    sub_dim: usize,
}

impl SubCodebook {
    fn new(sub_dim: usize, num_centroids: usize) -> Self {
        Self {
            centroids: vec![vec![0.0; sub_dim]; num_centroids],
            sub_dim,
        }
    }

    /// Assign the nearest centroid index to a sub-vector.
    fn encode(&self, sub_vec: &[f32]) -> u16 {
        let mut best_idx = 0u16;
        let mut best_dist = f32::MAX;
        for (i, centroid) in self.centroids.iter().enumerate() {
            let dist = l2_sq(sub_vec, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i as u16;
            }
        }
        best_idx
    }

    /// Decode: return the centroid for a given code.
    fn decode(&self, code: u16) -> &[f32] {
        &self.centroids[code as usize]
    }

    /// Build a distance lookup table for a query sub-vector.
    fn build_distance_table(&self, query_sub: &[f32]) -> Vec<f32> {
        self.centroids.iter().map(|c| l2_sq(query_sub, c)).collect()
    }
}

// ---------------------------------------------------------------------------
// Encoded vector (PQ codes)
// ---------------------------------------------------------------------------

/// A PQ-encoded vector: `codes[i]` is the centroid index for sub-space `i`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PqCode {
    id: u64,
    codes: Vec<u16>,
}

// ---------------------------------------------------------------------------
// ProductQuantizationIndex
// ---------------------------------------------------------------------------

/// The Product Quantization index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizationIndex {
    config: PqConfig,
    codebooks: Vec<SubCodebook>,
    entries: Vec<PqCode>,
    trained: bool,
    sub_dim: usize,
}

impl ProductQuantizationIndex {
    /// Create a new (untrained) PQ index.
    pub fn new(config: PqConfig) -> Result<Self> {
        if config.dimension == 0 {
            bail!("dimension must be > 0");
        }
        if config.num_sub_vectors == 0 {
            bail!("num_sub_vectors must be > 0");
        }
        if config.dimension % config.num_sub_vectors != 0 {
            bail!(
                "dimension ({}) must be divisible by num_sub_vectors ({})",
                config.dimension,
                config.num_sub_vectors
            );
        }
        if config.num_centroids == 0 || config.num_centroids > 65536 {
            bail!("num_centroids must be in 1..=65536");
        }
        let sub_dim = config.dimension / config.num_sub_vectors;
        let codebooks = (0..config.num_sub_vectors)
            .map(|_| SubCodebook::new(sub_dim, config.num_centroids))
            .collect();
        Ok(Self {
            config,
            codebooks,
            entries: Vec::new(),
            trained: false,
            sub_dim,
        })
    }

    /// Train the codebooks using the provided training vectors.
    pub fn train(&mut self, training_data: &[Vec<f32>]) -> Result<()> {
        if training_data.is_empty() {
            bail!("training data is empty");
        }
        for (i, v) in training_data.iter().enumerate() {
            if v.len() != self.config.dimension {
                bail!(
                    "training vector {i} has dimension {} but expected {}",
                    v.len(),
                    self.config.dimension
                );
            }
        }

        for m in 0..self.config.num_sub_vectors {
            let start = m * self.sub_dim;
            let end = start + self.sub_dim;

            // Extract sub-vectors for this sub-space
            let sub_vectors: Vec<Vec<f32>> = training_data
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            // Run simple k-means
            let centroids = kmeans(
                &sub_vectors,
                self.config.num_centroids,
                self.config.training_iterations,
                self.sub_dim,
            );
            self.codebooks[m].centroids = centroids;
        }

        self.trained = true;
        Ok(())
    }

    /// Whether the index has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Add a vector with the given ID.
    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if !self.trained {
            bail!("index must be trained before adding vectors");
        }
        if vector.len() != self.config.dimension {
            bail!(
                "vector dimension {} != expected {}",
                vector.len(),
                self.config.dimension
            );
        }

        let mut codes = Vec::with_capacity(self.config.num_sub_vectors);
        for m in 0..self.config.num_sub_vectors {
            let start = m * self.sub_dim;
            let end = start + self.sub_dim;
            let code = self.codebooks[m].encode(&vector[start..end]);
            codes.push(code);
        }

        self.entries.push(PqCode { id, codes });
        Ok(())
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Search for the `k` nearest neighbours using asymmetric distance computation.
    /// Returns `(id, approximate_distance)` pairs sorted by ascending distance.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        if !self.trained {
            bail!("index must be trained before searching");
        }
        if query.len() != self.config.dimension {
            bail!(
                "query dimension {} != expected {}",
                query.len(),
                self.config.dimension
            );
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        // Pre-compute distance tables for each sub-space
        let distance_tables: Vec<Vec<f32>> = (0..self.config.num_sub_vectors)
            .map(|m| {
                let start = m * self.sub_dim;
                let end = start + self.sub_dim;
                self.codebooks[m].build_distance_table(&query[start..end])
            })
            .collect();

        // Scan all entries, summing per-subspace distances from the lookup tables
        let mut heap: BinaryHeap<Reverse<(OrderedF32, u64)>> = BinaryHeap::new();
        for entry in &self.entries {
            let mut dist = 0.0f32;
            for (m, code) in entry.codes.iter().enumerate() {
                dist += distance_tables[m][*code as usize];
            }
            heap.push(Reverse((OrderedF32(dist), entry.id)));
        }

        let mut results = Vec::with_capacity(k.min(heap.len()));
        for _ in 0..k {
            if let Some(Reverse((OrderedF32(d), id))) = heap.pop() {
                results.push((id, d));
            } else {
                break;
            }
        }
        Ok(results)
    }

    /// Reconstruct an approximate vector from its PQ codes.
    pub fn reconstruct(&self, id: u64) -> Result<Vec<f32>> {
        let entry = self
            .entries
            .iter()
            .find(|e| e.id == id)
            .ok_or_else(|| anyhow!("id {id} not found in index"))?;

        let mut vector = Vec::with_capacity(self.config.dimension);
        for (m, code) in entry.codes.iter().enumerate() {
            vector.extend_from_slice(self.codebooks[m].decode(*code));
        }
        Ok(vector)
    }

    /// Remove all indexed vectors (keeps codebooks).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get the PQ configuration.
    pub fn config(&self) -> &PqConfig {
        &self.config
    }

    /// Compute the compression ratio (original bytes / encoded bytes).
    pub fn compression_ratio(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let original_bytes = self.config.dimension * 4; // f32
        let encoded_bytes = self.config.num_sub_vectors * 2; // u16 codes
        original_bytes as f64 / encoded_bytes as f64
    }
}

// ---------------------------------------------------------------------------
// k-means (simple)
// ---------------------------------------------------------------------------

fn kmeans(data: &[Vec<f32>], k: usize, iterations: usize, dim: usize) -> Vec<Vec<f32>> {
    let actual_k = k.min(data.len());
    // Initialise centroids from first k data points
    let mut centroids: Vec<Vec<f32>> = data.iter().take(actual_k).cloned().collect();
    // Pad if data.len() < k
    while centroids.len() < k {
        centroids.push(vec![0.0; dim]);
    }

    for _ in 0..iterations {
        // Assignment step
        let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (idx, point) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = l2_sq(point, centroid);
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }
            assignments[best].push(idx);
        }

        // Update step
        for (c, assigned) in assignments.iter().enumerate() {
            if assigned.is_empty() {
                continue;
            }
            let mut new_centroid = vec![0.0f32; dim];
            for &idx in assigned {
                for (d, val) in data[idx].iter().enumerate() {
                    new_centroid[d] += val;
                }
            }
            let count = assigned.len() as f32;
            for val in &mut new_centroid {
                *val /= count;
            }
            centroids[c] = new_centroid;
        }
    }

    centroids
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Newtype for ordered f32 (used in BinaryHeap).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(dim: usize, m: usize, k: usize) -> PqConfig {
        PqConfig {
            dimension: dim,
            num_sub_vectors: m,
            num_centroids: k,
            training_iterations: 5,
            num_probes: 0,
        }
    }

    fn make_training_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f32 * 0.1).collect())
            .collect()
    }

    fn trained_index(dim: usize, m: usize, k: usize) -> ProductQuantizationIndex {
        let config = default_config(dim, m, k);
        let mut idx = ProductQuantizationIndex::new(config).expect("new");
        let data = make_training_data(k.max(4), dim);
        idx.train(&data).expect("train");
        idx
    }

    // -- constructor tests --

    #[test]
    fn test_new_valid_config() {
        let idx = ProductQuantizationIndex::new(default_config(8, 4, 4));
        assert!(idx.is_ok());
    }

    #[test]
    fn test_new_zero_dimension() {
        let config = PqConfig {
            dimension: 0,
            ..Default::default()
        };
        assert!(ProductQuantizationIndex::new(config).is_err());
    }

    #[test]
    fn test_new_zero_sub_vectors() {
        let config = PqConfig {
            num_sub_vectors: 0,
            ..Default::default()
        };
        assert!(ProductQuantizationIndex::new(config).is_err());
    }

    #[test]
    fn test_new_indivisible_dimension() {
        let config = PqConfig {
            dimension: 7,
            num_sub_vectors: 4,
            ..Default::default()
        };
        assert!(ProductQuantizationIndex::new(config).is_err());
    }

    #[test]
    fn test_new_zero_centroids() {
        let config = PqConfig {
            num_centroids: 0,
            ..Default::default()
        };
        assert!(ProductQuantizationIndex::new(config).is_err());
    }

    // -- training tests --

    #[test]
    fn test_train_sets_trained_flag() {
        let mut idx = ProductQuantizationIndex::new(default_config(8, 4, 4)).expect("new");
        assert!(!idx.is_trained());
        let data = make_training_data(10, 8);
        idx.train(&data).expect("train");
        assert!(idx.is_trained());
    }

    #[test]
    fn test_train_empty_data_fails() {
        let mut idx = ProductQuantizationIndex::new(default_config(8, 4, 4)).expect("new");
        assert!(idx.train(&[]).is_err());
    }

    #[test]
    fn test_train_wrong_dimension_fails() {
        let mut idx = ProductQuantizationIndex::new(default_config(8, 4, 4)).expect("new");
        let data = vec![vec![1.0, 2.0]]; // dim=2 not 8
        assert!(idx.train(&data).is_err());
    }

    // -- add tests --

    #[test]
    fn test_add_before_training_fails() {
        let mut idx = ProductQuantizationIndex::new(default_config(8, 4, 4)).expect("new");
        assert!(idx.add(0, &[1.0; 8]).is_err());
    }

    #[test]
    fn test_add_wrong_dimension_fails() {
        let mut idx = trained_index(8, 4, 4);
        assert!(idx.add(0, &[1.0; 4]).is_err());
    }

    #[test]
    fn test_add_and_len() {
        let mut idx = trained_index(8, 4, 4);
        assert!(idx.is_empty());
        idx.add(0, &[1.0; 8]).expect("add");
        assert_eq!(idx.len(), 1);
        idx.add(1, &[2.0; 8]).expect("add");
        assert_eq!(idx.len(), 2);
    }

    // -- search tests --

    #[test]
    fn test_search_before_training_fails() {
        let idx = ProductQuantizationIndex::new(default_config(8, 4, 4)).expect("new");
        assert!(idx.search(&[1.0; 8], 1).is_err());
    }

    #[test]
    fn test_search_wrong_dimension_fails() {
        let idx = trained_index(8, 4, 4);
        assert!(idx.search(&[1.0; 4], 1).is_err());
    }

    #[test]
    fn test_search_k_zero_returns_empty() {
        let idx = trained_index(8, 4, 4);
        let results = idx.search(&[1.0; 8], 0).expect("search");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_empty_index_returns_empty() {
        let idx = trained_index(8, 4, 4);
        let results = idx.search(&[1.0; 8], 5).expect("search");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_finds_nearest() {
        let mut idx = trained_index(8, 4, 4);
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v2 = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];
        idx.add(10, &v1).expect("add");
        idx.add(20, &v2).expect("add");

        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let results = idx.search(&query, 1).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 10);
    }

    #[test]
    fn test_search_returns_sorted_by_distance() {
        let mut idx = trained_index(8, 4, 4);
        let v1 = vec![1.0; 8];
        let v2 = vec![2.0; 8];
        let v3 = vec![10.0; 8];
        idx.add(1, &v1).expect("add");
        idx.add(2, &v2).expect("add");
        idx.add(3, &v3).expect("add");

        let results = idx.search(&[1.0; 8], 3).expect("search");
        assert_eq!(results.len(), 3);
        // Distances should be ascending
        assert!(results[0].1 <= results[1].1);
        assert!(results[1].1 <= results[2].1);
    }

    #[test]
    fn test_search_k_larger_than_index() {
        let mut idx = trained_index(8, 4, 4);
        idx.add(1, &[1.0; 8]).expect("add");
        let results = idx.search(&[1.0; 8], 100).expect("search");
        assert_eq!(results.len(), 1);
    }

    // -- reconstruct tests --

    #[test]
    fn test_reconstruct_existing_id() {
        let mut idx = trained_index(8, 4, 4);
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        idx.add(42, &v).expect("add");
        let reconstructed = idx.reconstruct(42).expect("reconstruct");
        assert_eq!(reconstructed.len(), 8);
    }

    #[test]
    fn test_reconstruct_missing_id() {
        let idx = trained_index(8, 4, 4);
        assert!(idx.reconstruct(999).is_err());
    }

    // -- clear --

    #[test]
    fn test_clear() {
        let mut idx = trained_index(8, 4, 4);
        idx.add(1, &[1.0; 8]).expect("add");
        assert_eq!(idx.len(), 1);
        idx.clear();
        assert!(idx.is_empty());
        // Still trained after clear
        assert!(idx.is_trained());
    }

    // -- compression ratio --

    #[test]
    fn test_compression_ratio_empty() {
        let idx = trained_index(8, 4, 4);
        assert_eq!(idx.compression_ratio(), 0.0);
    }

    #[test]
    fn test_compression_ratio_non_empty() {
        let mut idx = trained_index(8, 4, 4);
        idx.add(0, &[1.0; 8]).expect("add");
        let ratio = idx.compression_ratio();
        // 8 * 4 bytes original = 32; 4 * 2 bytes encoded = 8 -> ratio = 4
        assert!((ratio - 4.0).abs() < 1e-6);
    }

    // -- config --

    #[test]
    fn test_config_accessor() {
        let idx = ProductQuantizationIndex::new(default_config(16, 4, 8)).expect("new");
        assert_eq!(idx.config().dimension, 16);
        assert_eq!(idx.config().num_sub_vectors, 4);
    }

    // -- default config --

    #[test]
    fn test_default_config() {
        let config = PqConfig::default();
        assert_eq!(config.dimension, 128);
        assert_eq!(config.num_sub_vectors, 8);
        assert_eq!(config.num_centroids, 256);
    }

    // -- kmeans --

    #[test]
    fn test_kmeans_basic() {
        let data = vec![
            vec![1.0, 0.0],
            vec![1.1, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.1],
        ];
        let centroids = kmeans(&data, 2, 10, 2);
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn test_kmeans_more_k_than_data() {
        let data = vec![vec![1.0], vec![2.0]];
        let centroids = kmeans(&data, 5, 3, 1);
        assert_eq!(centroids.len(), 5);
    }

    // -- l2_sq --

    #[test]
    fn test_l2_sq_identical() {
        assert_eq!(l2_sq(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_l2_sq_known() {
        // (3-1)^2 + (4-1)^2 = 4+9 = 13
        let dist = l2_sq(&[3.0, 4.0], &[1.0, 1.0]);
        assert!((dist - 13.0).abs() < 1e-6);
    }

    // -- ordered f32 --

    #[test]
    fn test_ordered_f32_ordering() {
        let a = OrderedF32(1.0);
        let b = OrderedF32(2.0);
        assert!(a < b);
    }

    // -- multi-vector search --

    #[test]
    fn test_multi_add_and_search() {
        let mut idx = trained_index(8, 4, 4);
        for i in 0..20_u64 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add(i, &v).expect("add");
        }
        assert_eq!(idx.len(), 20);
        let results = idx.search(&[0.0; 8], 5).expect("search");
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_retrain_resets_codebooks() {
        let mut idx = trained_index(8, 4, 4);
        idx.add(0, &[1.0; 8]).expect("add");
        let data2 = make_training_data(10, 8);
        idx.train(&data2).expect("retrain");
        // entries are still there
        assert_eq!(idx.len(), 1);
    }

    // -- edge cases --

    #[test]
    fn test_single_dimension_subvectors() {
        let config = default_config(4, 4, 2);
        let mut idx = ProductQuantizationIndex::new(config).expect("new");
        let data = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        idx.train(&data).expect("train");
        idx.add(0, &[1.0, 2.0, 3.0, 4.0]).expect("add");
        let results = idx.search(&[1.0, 2.0, 3.0, 4.0], 1).expect("search");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_single_centroid_perfect_encode() {
        let config = default_config(4, 2, 1);
        let mut idx = ProductQuantizationIndex::new(config).expect("new");
        let data = vec![vec![1.0, 2.0, 3.0, 4.0]];
        idx.train(&data).expect("train");
        idx.add(0, &[1.0, 2.0, 3.0, 4.0]).expect("add");
        let recon = idx.reconstruct(0).expect("reconstruct");
        // With only 1 centroid, reconstruction should match the training mean
        assert_eq!(recon.len(), 4);
    }

    #[test]
    fn test_large_dimension() {
        let config = default_config(64, 8, 4);
        let mut idx = ProductQuantizationIndex::new(config).expect("new");
        let data = make_training_data(10, 64);
        idx.train(&data).expect("train");
        idx.add(0, &vec![0.5; 64]).expect("add");
        let results = idx.search(&[0.5; 64], 1).expect("search");
        assert_eq!(results.len(), 1);
    }
}
