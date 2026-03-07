//! Quantized embedding cache with scalar quantization, product quantization,
//! and asymmetric distance computation.
//!
//! `QuantizedEmbeddingCache` stores compressed int8 codes for each vector,
//! supporting two compression schemes:
//!
//! * **Scalar Quantization (SQ)** – maps each fp32 scalar to an int8 value
//!   using per-dimension or global min/max ranges.
//! * **Product Quantization (PQ)** – splits the vector into sub-spaces and
//!   quantizes each sub-space to a centroid index, enabling very high
//!   compression ratios.
//!
//! Distance is computed **asymmetrically**: the query is kept in fp32 while
//! the database codes are decompressed on-the-fly, giving better accuracy
//! than comparing compressed codes directly.
//!
//! Compression ratio and distance accuracy metrics are tracked automatically.
//!
//! # Pure Rust Policy
//!
//! No unsafe code, no C/Fortran FFI, no CUDA runtime calls.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── quantization scheme ────────────────────────────────────────────────────

/// Which compression scheme to use in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// Scalar quantization: one int8 per dimension.
    Scalar,
    /// Product quantization: one centroid index per sub-space.
    Product,
}

// ── scalar quantization helpers ────────────────────────────────────────────

/// Per-dimension parameters for scalar quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarDimParams {
    /// Minimum fp32 value seen during training.
    pub min: f32,
    /// Maximum fp32 value seen during training.
    pub max: f32,
    /// Precomputed scale = 255.0 / (max - min).
    pub scale: f32,
}

impl ScalarDimParams {
    fn new(min: f32, max: f32) -> Self {
        let range = max - min;
        let scale = if range > 1e-9 { 255.0 / range } else { 1.0 };
        Self { min, max, scale }
    }

    /// Quantize a single fp32 scalar to u8.
    #[inline]
    pub fn quantize(&self, v: f32) -> u8 {
        ((v - self.min) * self.scale).clamp(0.0, 255.0) as u8
    }

    /// Dequantize a u8 back to fp32.
    #[inline]
    pub fn dequantize(&self, code: u8) -> f32 {
        self.min + (code as f32) / self.scale
    }
}

// ── product quantization helpers ───────────────────────────────────────────

/// A single PQ codebook: `n_centroids` centroids, each of dimension `sub_dim`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCodebook {
    /// Number of centroids.
    pub n_centroids: usize,
    /// Dimension of each centroid (= total_dim / n_subspaces).
    pub sub_dim: usize,
    /// Flattened centroids: `n_centroids × sub_dim` fp32 values.
    pub centroids: Vec<f32>,
}

impl PqCodebook {
    /// Build a codebook from training sub-vectors using a simple k-means variant.
    fn train(sub_vectors: &[Vec<f32>], n_centroids: usize, max_iters: usize) -> Self {
        let sub_dim = if sub_vectors.is_empty() {
            0
        } else {
            sub_vectors[0].len()
        };

        if sub_vectors.is_empty() || n_centroids == 0 || sub_dim == 0 {
            return Self {
                n_centroids,
                sub_dim,
                centroids: Vec::new(),
            };
        }

        let actual_k = n_centroids.min(sub_vectors.len());

        // Initialise centroids from the first `actual_k` training vectors
        let mut centroids: Vec<Vec<f32>> = sub_vectors.iter().take(actual_k).cloned().collect();

        for _ in 0..max_iters {
            // Assignment step
            let mut assignments: Vec<usize> = Vec::with_capacity(sub_vectors.len());
            for sv in sub_vectors {
                let best = centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, euclidean_sq_slice(sv, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                assignments.push(best);
            }

            // Update step
            let mut new_centroids = vec![vec![0.0_f32; sub_dim]; actual_k];
            let mut counts = vec![0usize; actual_k];
            for (sv, &asgn) in sub_vectors.iter().zip(&assignments) {
                for (d, &v) in sv.iter().enumerate() {
                    new_centroids[asgn][d] += v;
                }
                counts[asgn] += 1;
            }
            for (c, count) in new_centroids.iter_mut().zip(&counts) {
                if *count > 0 {
                    for v in c.iter_mut() {
                        *v /= *count as f32;
                    }
                }
            }
            centroids = new_centroids;
        }

        let flat: Vec<f32> = centroids.into_iter().flatten().collect();
        Self {
            n_centroids: actual_k,
            sub_dim,
            centroids: flat,
        }
    }

    /// Find the nearest centroid index for a sub-vector.
    pub fn encode(&self, sub_vec: &[f32]) -> u8 {
        let best = (0..self.n_centroids)
            .map(|i| {
                let offset = i * self.sub_dim;
                let centroid = &self.centroids[offset..offset + self.sub_dim];
                (i, euclidean_sq_slice(sub_vec, centroid))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (best & 0xFF) as u8
    }

    /// Decode a centroid index back to a sub-vector slice.
    pub fn decode(&self, code: u8) -> &[f32] {
        let i = (code as usize).min(self.n_centroids.saturating_sub(1));
        let offset = i * self.sub_dim;
        &self.centroids[offset..offset + self.sub_dim]
    }
}

// ── cache config ───────────────────────────────────────────────────────────

/// Configuration for `QuantizedEmbeddingCache`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedCacheConfig {
    /// Which quantization scheme to use.
    pub scheme: QuantizationScheme,
    /// For SQ: number of bits (currently fixed at 8).
    pub sq_bits: u8,
    /// For PQ: number of sub-spaces.
    pub pq_n_subspaces: usize,
    /// For PQ: number of centroids per codebook.
    pub pq_n_centroids: usize,
    /// For PQ: number of k-means iterations during training.
    pub pq_max_iters: usize,
    /// Whether to normalize vectors before quantization.
    pub normalize: bool,
    /// Maximum number of training samples to use.
    pub max_training_samples: usize,
}

impl Default for QuantizedCacheConfig {
    fn default() -> Self {
        Self {
            scheme: QuantizationScheme::Scalar,
            sq_bits: 8,
            pq_n_subspaces: 8,
            pq_n_centroids: 256,
            pq_max_iters: 25,
            normalize: false,
            max_training_samples: 10_000,
        }
    }
}

// ── metrics ────────────────────────────────────────────────────────────────

/// Compression ratio and distance accuracy metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Number of vectors stored.
    pub vector_count: usize,
    /// Dimensionality of stored vectors.
    pub dimensions: usize,
    /// Bytes used for compressed codes.
    pub compressed_bytes: usize,
    /// Bytes that fp32 vectors would occupy.
    pub uncompressed_bytes: usize,
    /// `uncompressed_bytes / compressed_bytes`.
    pub compression_ratio: f64,
    /// Mean absolute error between original and reconstructed vectors
    /// (measured on the stored vectors, sampled during `train`).
    pub mean_reconstruction_error: f32,
    /// Number of queries served.
    pub queries_served: u64,
    /// Cumulative distance computations.
    pub distance_computations: u64,
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            vector_count: 0,
            dimensions: 0,
            compressed_bytes: 0,
            uncompressed_bytes: 0,
            compression_ratio: 0.0,
            mean_reconstruction_error: 0.0,
            queries_served: 0,
            distance_computations: 0,
        }
    }
}

// ── internal storage ───────────────────────────────────────────────────────

/// A stored compressed code, keyed by arbitrary string ID.
#[derive(Debug, Clone)]
struct CompressedCode {
    /// One byte per code slot (SQ: one per dimension; PQ: one per sub-space).
    codes: Vec<u8>,
    /// Optional user-defined metadata.
    metadata: HashMap<String, String>,
}

// ── main struct ────────────────────────────────────────────────────────────

/// Quantized embedding cache with scalar or product quantization and asymmetric
/// distance computation for compressed similarity search.
pub struct QuantizedEmbeddingCache {
    config: QuantizedCacheConfig,
    dimensions: usize,
    // SQ parameters
    sq_params: Vec<ScalarDimParams>,
    // PQ codebooks (one per sub-space)
    pq_codebooks: Vec<PqCodebook>,
    // Stored compressed codes
    codes: Vec<CompressedCode>,
    id_to_idx: HashMap<String, usize>,
    idx_to_id: Vec<String>,
    // Metrics
    metrics: CacheMetrics,
}

impl QuantizedEmbeddingCache {
    /// Create a new, untrained cache.
    pub fn new(config: QuantizedCacheConfig, dimensions: usize) -> Self {
        Self {
            config,
            dimensions,
            sq_params: Vec::new(),
            pq_codebooks: Vec::new(),
            codes: Vec::new(),
            id_to_idx: HashMap::new(),
            idx_to_id: Vec::new(),
            metrics: CacheMetrics {
                dimensions,
                ..Default::default()
            },
        }
    }

    // ── training ──────────────────────────────────────────────────────────

    /// Train quantization parameters from `training_vectors`.
    ///
    /// Must be called before any calls to `add` or `search`.
    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> Result<()> {
        if training_vectors.is_empty() {
            return Err(anyhow!("No training vectors provided"));
        }
        let dim = training_vectors[0].len();
        if dim != self.dimensions {
            return Err(anyhow!(
                "Training vector dim {} ≠ cache dim {}",
                dim,
                self.dimensions
            ));
        }

        let limit = training_vectors.len().min(self.config.max_training_samples);
        let raw_samples = &training_vectors[..limit];

        // When normalization is enabled, normalize training samples so that
        // the quantizer learns the min/max range of normalized vectors.
        let normalized_storage: Vec<Vec<f32>>;
        let samples: &[Vec<f32>] = if self.config.normalize {
            normalized_storage = raw_samples.iter().map(|v| normalize_vec(v)).collect();
            &normalized_storage
        } else {
            raw_samples
        };

        match self.config.scheme {
            QuantizationScheme::Scalar => self.train_scalar(samples)?,
            QuantizationScheme::Product => self.train_product(samples)?,
        }

        // Measure reconstruction error on training samples
        let error = self.measure_reconstruction_error(samples);
        self.metrics.mean_reconstruction_error = error;

        Ok(())
    }

    fn train_scalar(&mut self, samples: &[Vec<f32>]) -> Result<()> {
        let mut dim_mins = vec![f32::INFINITY; self.dimensions];
        let mut dim_maxs = vec![f32::NEG_INFINITY; self.dimensions];

        for v in samples {
            for (d, &val) in v.iter().enumerate() {
                dim_mins[d] = dim_mins[d].min(val);
                dim_maxs[d] = dim_maxs[d].max(val);
            }
        }

        self.sq_params = dim_mins
            .into_iter()
            .zip(dim_maxs)
            .map(|(mn, mx)| ScalarDimParams::new(mn, mx))
            .collect();

        Ok(())
    }

    fn train_product(&mut self, samples: &[Vec<f32>]) -> Result<()> {
        let n_sub = self.config.pq_n_subspaces;
        if self.dimensions % n_sub != 0 {
            return Err(anyhow!(
                "dimensions ({}) must be divisible by pq_n_subspaces ({})",
                self.dimensions,
                n_sub
            ));
        }
        let sub_dim = self.dimensions / n_sub;

        self.pq_codebooks = (0..n_sub)
            .map(|s| {
                let sub_vecs: Vec<Vec<f32>> = samples
                    .iter()
                    .map(|v| v[s * sub_dim..(s + 1) * sub_dim].to_vec())
                    .collect();
                PqCodebook::train(
                    &sub_vecs,
                    self.config.pq_n_centroids,
                    self.config.pq_max_iters,
                )
            })
            .collect();

        Ok(())
    }

    fn measure_reconstruction_error(&self, samples: &[Vec<f32>]) -> f32 {
        let limit = samples.len().min(200);
        let mut total = 0.0_f32;
        for v in &samples[..limit] {
            let normalized = if self.config.normalize {
                normalize_vec(v)
            } else {
                v.clone()
            };
            let codes = self.encode_vector(&normalized);
            let reconstructed = self.decode_codes(&codes);
            let err: f32 = normalized
                .iter()
                .zip(&reconstructed)
                .map(|(&a, &b)| (a - b).abs())
                .sum::<f32>()
                / self.dimensions as f32;
            total += err;
        }
        total / limit as f32
    }

    // ── insert / retrieve ──────────────────────────────────────────────────

    /// Compress and store a vector by `id`.
    pub fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        self.add_with_metadata(id, vector, HashMap::new())
    }

    /// Compress and store a vector with metadata.
    pub fn add_with_metadata(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        if self.is_untrained() {
            return Err(anyhow!("Cache not trained; call train() first"));
        }
        if vector.len() != self.dimensions {
            return Err(anyhow!(
                "Vector dim {} ≠ cache dim {}",
                vector.len(),
                self.dimensions
            ));
        }
        if self.id_to_idx.contains_key(&id) {
            return Err(anyhow!("ID '{}' already in cache", id));
        }

        let normalized = if self.config.normalize {
            normalize_vec(&vector)
        } else {
            vector
        };
        let codes = self.encode_vector(&normalized);
        let idx = self.codes.len();

        self.codes.push(CompressedCode { codes, metadata });
        self.id_to_idx.insert(id.clone(), idx);
        self.idx_to_id.push(id);

        // Update metrics
        let code_len = self.code_length();
        self.metrics.vector_count += 1;
        self.metrics.compressed_bytes += code_len;
        self.metrics.uncompressed_bytes += self.dimensions * 4;
        self.metrics.compression_ratio =
            self.metrics.uncompressed_bytes as f64 / self.metrics.compressed_bytes.max(1) as f64;

        Ok(())
    }

    /// Retrieve the decompressed (reconstructed) vector for `id`.
    pub fn get(&self, id: &str) -> Option<Vec<f32>> {
        let idx = *self.id_to_idx.get(id)?;
        Some(self.decode_codes(&self.codes[idx].codes))
    }

    /// Number of vectors in the cache.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Returns `true` if no vectors are stored.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    // ── asymmetric search ──────────────────────────────────────────────────

    /// Find the `k` nearest cached vectors to `query` using asymmetric distance.
    ///
    /// The query is kept in fp32; each database code is decompressed on-the-fly
    /// and Euclidean distance is computed.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if self.is_untrained() {
            return Err(anyhow!("Cache not trained"));
        }
        if query.len() != self.dimensions {
            return Err(anyhow!(
                "Query dim {} ≠ cache dim {}",
                query.len(),
                self.dimensions
            ));
        }

        let normalized_query = if self.config.normalize {
            normalize_vec(query)
        } else {
            query.to_vec()
        };

        let mut distances: Vec<(usize, f32)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(i, code)| {
                let reconstructed = self.decode_codes(&code.codes);
                let dist = euclidean_sq_slice(&normalized_query, &reconstructed).sqrt();
                (i, dist)
            })
            .collect();

        self.metrics.distance_computations += self.codes.len() as u64;
        self.metrics.queries_served += 1;

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        Ok(distances
            .into_iter()
            .map(|(i, d)| (self.idx_to_id[i].clone(), d))
            .collect())
    }

    /// Return a snapshot of current metrics.
    pub fn metrics(&self) -> &CacheMetrics {
        &self.metrics
    }

    /// Access the configuration.
    pub fn config(&self) -> &QuantizedCacheConfig {
        &self.config
    }

    // ── encoding / decoding ────────────────────────────────────────────────

    fn encode_vector(&self, vector: &[f32]) -> Vec<u8> {
        match self.config.scheme {
            QuantizationScheme::Scalar => vector
                .iter()
                .zip(&self.sq_params)
                .map(|(&v, params)| params.quantize(v))
                .collect(),
            QuantizationScheme::Product => {
                let n_sub = self.pq_codebooks.len();
                if n_sub == 0 {
                    return Vec::new();
                }
                let sub_dim = self.dimensions / n_sub;
                (0..n_sub)
                    .map(|s| {
                        let sub = &vector[s * sub_dim..(s + 1) * sub_dim];
                        self.pq_codebooks[s].encode(sub)
                    })
                    .collect()
            }
        }
    }

    fn decode_codes(&self, codes: &[u8]) -> Vec<f32> {
        match self.config.scheme {
            QuantizationScheme::Scalar => codes
                .iter()
                .zip(&self.sq_params)
                .map(|(&code, params)| params.dequantize(code))
                .collect(),
            QuantizationScheme::Product => {
                let n_sub = self.pq_codebooks.len();
                if n_sub == 0 {
                    return Vec::new();
                }
                let mut out = Vec::with_capacity(self.dimensions);
                for (s, &code) in (0..n_sub).zip(codes.iter()) {
                    out.extend_from_slice(self.pq_codebooks[s].decode(code));
                }
                out
            }
        }
    }

    /// Number of code bytes per vector.
    fn code_length(&self) -> usize {
        match self.config.scheme {
            QuantizationScheme::Scalar => self.dimensions, // 1 byte per dim
            QuantizationScheme::Product => self.config.pq_n_subspaces,
        }
    }

    fn is_untrained(&self) -> bool {
        match self.config.scheme {
            QuantizationScheme::Scalar => self.sq_params.is_empty(),
            QuantizationScheme::Product => self.pq_codebooks.is_empty(),
        }
    }
}

// ── free functions ─────────────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length slices.
#[inline]
fn euclidean_sq_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// L2-normalize a vector (returns original if norm is near zero).
fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm < 1e-9 {
        v.to_vec()
    } else {
        v.iter().map(|&x| x / norm).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sq_cache(dims: usize) -> QuantizedEmbeddingCache {
        let config = QuantizedCacheConfig {
            scheme: QuantizationScheme::Scalar,
            ..Default::default()
        };
        QuantizedEmbeddingCache::new(config, dims)
    }

    fn make_pq_cache(dims: usize, n_sub: usize) -> QuantizedEmbeddingCache {
        let config = QuantizedCacheConfig {
            scheme: QuantizationScheme::Product,
            pq_n_subspaces: n_sub,
            pq_n_centroids: 8,
            pq_max_iters: 5,
            ..Default::default()
        };
        QuantizedEmbeddingCache::new(config, dims)
    }

    fn training_vecs(n: usize, dims: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dims).map(|d| (i * dims + d) as f32 * 0.01).collect())
            .collect()
    }

    // ── SQ training ────────────────────────────────────────────────────────

    #[test]
    fn test_sq_train_succeeds() {
        let mut cache = make_sq_cache(4);
        let samples = training_vecs(50, 4);
        assert!(cache.train(&samples).is_ok());
        assert_eq!(cache.sq_params.len(), 4);
    }

    #[test]
    fn test_sq_train_empty_fails() {
        let mut cache = make_sq_cache(4);
        assert!(cache.train(&[]).is_err());
    }

    #[test]
    fn test_sq_train_wrong_dim_fails() {
        let mut cache = make_sq_cache(4);
        let samples = vec![vec![1.0_f32; 8]];
        assert!(cache.train(&samples).is_err());
    }

    #[test]
    fn test_sq_untrained_add_fails() {
        let mut cache = make_sq_cache(4);
        let err = cache.add("k".to_string(), vec![0.0; 4]);
        assert!(err.is_err());
    }

    // ── SQ add / get ───────────────────────────────────────────────────────

    #[test]
    fn test_sq_add_and_get() {
        let mut cache = make_sq_cache(4);
        let samples = training_vecs(50, 4);
        cache.train(&samples).unwrap();
        cache
            .add("v0".to_string(), vec![0.1, 0.2, 0.3, 0.4])
            .unwrap();
        let reconstructed = cache.get("v0");
        assert!(reconstructed.is_some());
        let r = reconstructed.unwrap();
        assert_eq!(r.len(), 4);
        // Reconstruction should be close to original
        for (orig, rec) in [0.1_f32, 0.2, 0.3, 0.4].iter().zip(&r) {
            assert!((orig - rec).abs() < 0.05, "Reconstruction error too large");
        }
    }

    #[test]
    fn test_sq_duplicate_id_fails() {
        let mut cache = make_sq_cache(4);
        cache.train(&training_vecs(10, 4)).unwrap();
        cache.add("k".to_string(), vec![0.0; 4]).unwrap();
        assert!(cache.add("k".to_string(), vec![1.0; 4]).is_err());
    }

    #[test]
    fn test_sq_get_missing_returns_none() {
        let mut cache = make_sq_cache(4);
        cache.train(&training_vecs(10, 4)).unwrap();
        assert!(cache.get("absent").is_none());
    }

    // ── SQ search ──────────────────────────────────────────────────────────

    #[test]
    fn test_sq_search_returns_nearest() {
        let mut cache = make_sq_cache(2);
        let samples = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![5.0, 0.0]];
        cache.train(&samples).unwrap();
        cache.add("origin".to_string(), vec![0.0, 0.0]).unwrap();
        cache.add("near".to_string(), vec![0.5, 0.0]).unwrap();
        cache.add("far".to_string(), vec![5.0, 0.0]).unwrap();

        let results = cache.search(&[0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "origin");
    }

    #[test]
    fn test_sq_search_top_k_ordering() {
        let mut cache = make_sq_cache(1);
        let samples: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32]).collect();
        cache.train(&samples).unwrap();
        for i in 0..10_u32 {
            cache.add(format!("v{}", i), vec![i as f32]).unwrap();
        }
        let results = cache.search(&[5.0], 3).unwrap();
        assert!(results.len() <= 3);
        // Results should be ascending distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-6);
        }
    }

    #[test]
    fn test_sq_search_empty_cache() {
        let mut cache = make_sq_cache(4);
        cache.train(&training_vecs(10, 4)).unwrap();
        let results = cache.search(&[0.0; 4], 5).unwrap();
        assert!(results.is_empty());
    }

    // ── SQ metrics ─────────────────────────────────────────────────────────

    #[test]
    fn test_sq_compression_ratio_greater_than_one() {
        let mut cache = make_sq_cache(32);
        cache.train(&training_vecs(100, 32)).unwrap();
        for i in 0..10 {
            cache.add(format!("v{}", i), vec![0.5; 32]).unwrap();
        }
        let m = cache.metrics();
        assert!(m.compression_ratio > 1.0);
        // 32 dims × 4 bytes fp32 vs 32 dims × 1 byte u8 → ratio ≈ 4
        assert!(
            (m.compression_ratio - 4.0).abs() < 0.5,
            "SQ ratio should be ~4"
        );
    }

    #[test]
    fn test_sq_metrics_vector_count() {
        let mut cache = make_sq_cache(4);
        cache.train(&training_vecs(10, 4)).unwrap();
        for i in 0..5 {
            cache.add(format!("v{}", i), vec![i as f32; 4]).unwrap();
        }
        assert_eq!(cache.metrics().vector_count, 5);
    }

    #[test]
    fn test_sq_queries_served_increments() {
        let mut cache = make_sq_cache(4);
        cache.train(&training_vecs(10, 4)).unwrap();
        cache.add("a".to_string(), vec![0.0; 4]).unwrap();
        cache.search(&[0.0; 4], 1).unwrap();
        cache.search(&[0.0; 4], 1).unwrap();
        assert_eq!(cache.metrics().queries_served, 2);
    }

    #[test]
    fn test_sq_reconstruction_error_reasonable() {
        let mut cache = make_sq_cache(4);
        let samples = training_vecs(100, 4);
        cache.train(&samples).unwrap();
        // For 8-bit SQ, reconstruction error should be small
        assert!(cache.metrics().mean_reconstruction_error < 0.1);
    }

    // ── PQ training ────────────────────────────────────────────────────────

    #[test]
    fn test_pq_train_succeeds() {
        let mut cache = make_pq_cache(8, 2);
        let samples = training_vecs(50, 8);
        assert!(cache.train(&samples).is_ok());
        assert_eq!(cache.pq_codebooks.len(), 2);
    }

    #[test]
    fn test_pq_train_indivisible_dims_fails() {
        let mut cache = make_pq_cache(7, 3); // 7 not divisible by 3
        let samples = training_vecs(30, 7);
        assert!(cache.train(&samples).is_err());
    }

    #[test]
    fn test_pq_add_and_get() {
        let mut cache = make_pq_cache(8, 2);
        let samples = training_vecs(50, 8);
        cache.train(&samples).unwrap();
        cache.add("v0".to_string(), vec![0.1; 8]).unwrap();
        let r = cache.get("v0").unwrap();
        assert_eq!(r.len(), 8);
    }

    #[test]
    fn test_pq_compression_ratio() {
        let mut cache = make_pq_cache(16, 4); // 4 sub-spaces
        cache.train(&training_vecs(50, 16)).unwrap();
        for i in 0..8 {
            cache.add(format!("v{}", i), vec![0.5; 16]).unwrap();
        }
        let m = cache.metrics();
        // 16 dims × 4 bytes = 64 bytes uncompressed; 4 codes × 1 byte = 4 bytes compressed → ratio = 16
        assert!(m.compression_ratio > 4.0, "PQ ratio should be > 4");
    }

    #[test]
    fn test_pq_search() {
        let mut cache = make_pq_cache(8, 2);
        let samples = training_vecs(50, 8);
        cache.train(&samples).unwrap();
        cache.add("a".to_string(), vec![0.0; 8]).unwrap();
        cache.add("b".to_string(), vec![10.0; 8]).unwrap();
        let results = cache.search(&[0.1; 8], 1).unwrap();
        assert!(!results.is_empty());
    }

    // ── normalization ──────────────────────────────────────────────────────

    #[test]
    fn test_normalized_vectors_stored_as_unit_length() {
        let config = QuantizedCacheConfig {
            scheme: QuantizationScheme::Scalar,
            normalize: true,
            ..Default::default()
        };
        let mut cache = QuantizedEmbeddingCache::new(config, 4);
        let long_vecs: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32 + 1.0, i as f32 + 2.0, 0.0, 0.0])
            .collect();
        cache.train(&long_vecs).unwrap();
        cache
            .add("v".to_string(), vec![3.0, 4.0, 0.0, 0.0])
            .unwrap();
        let r = cache.get("v").unwrap();
        let norm: f32 = r.iter().map(|&x| x * x).sum::<f32>().sqrt();
        // Reconstructed vector should be approximately unit length (quantization error allowed)
        assert!((norm - 1.0).abs() < 0.1, "norm={}, expected ~1.0", norm);
    }

    // ── config accessors ───────────────────────────────────────────────────

    #[test]
    fn test_config_accessors() {
        let config = QuantizedCacheConfig {
            scheme: QuantizationScheme::Product,
            pq_n_subspaces: 4,
            pq_n_centroids: 16,
            ..Default::default()
        };
        let cache = QuantizedEmbeddingCache::new(config, 8);
        assert_eq!(cache.config().pq_n_subspaces, 4);
        assert_eq!(cache.config().pq_n_centroids, 16);
    }

    #[test]
    fn test_is_empty_initially() {
        let mut cache = make_sq_cache(4);
        cache.train(&training_vecs(10, 4)).unwrap();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_len_after_adds() {
        let mut cache = make_sq_cache(4);
        cache.train(&training_vecs(10, 4)).unwrap();
        for i in 0..5 {
            cache.add(format!("v{}", i), vec![0.0; 4]).unwrap();
        }
        assert_eq!(cache.len(), 5);
    }

    // ── add_with_metadata ──────────────────────────────────────────────────

    #[test]
    fn test_add_with_metadata() {
        let mut cache = make_sq_cache(4);
        cache.train(&training_vecs(10, 4)).unwrap();
        let mut meta = HashMap::new();
        meta.insert("tag".to_string(), "test".to_string());
        cache
            .add_with_metadata("m".to_string(), vec![0.0; 4], meta)
            .unwrap();
        assert_eq!(cache.len(), 1);
    }

    // ── scalar dim params ──────────────────────────────────────────────────

    #[test]
    fn test_scalar_dim_params_roundtrip() {
        let params = ScalarDimParams::new(-1.0, 1.0);
        let q = params.quantize(0.0);
        let r = params.dequantize(q);
        assert!((r - 0.0).abs() < 0.02);
    }

    #[test]
    fn test_scalar_dim_params_extremes() {
        let params = ScalarDimParams::new(0.0, 1.0);
        assert_eq!(params.quantize(0.0), 0);
        assert_eq!(params.quantize(1.0), 255);
    }
}
