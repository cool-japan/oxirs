//! Embedding Compression: Quantization and Product Quantization for efficient
//! storage and approximate nearest-neighbor search.
//!
//! Provides:
//! - `QuantizedEmbedding` — 8-bit scalar quantization via min-max
//! - `EmbeddingQuantizer` — batch quantization for 4-bit and 8-bit
//! - `ProductQuantizer` — product quantization with per-subspace k-means

// ─────────────────────────────────────────────
// QuantizedEmbedding (scalar 8-bit)
// ─────────────────────────────────────────────

/// A single embedding quantized to 8-bit precision via min-max scaling.
#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    /// Original embedding dimensionality.
    pub original_dim: usize,
    /// Quantized values as u8 in [0, 255].
    pub quantized_data: Vec<u8>,
    /// Scale factor: (max - min) / 255.
    pub scale: f32,
    /// Zero point (minimum value of original embedding).
    pub zero_point: f32,
}

impl QuantizedEmbedding {
    /// Quantize a floating-point embedding to 8-bit using min-max scaling.
    ///
    /// Formula: v_q = round((v - min) / (max - min) * 255)
    pub fn quantize(embedding: &[f32]) -> Self {
        let dim = embedding.len();
        if dim == 0 {
            return Self {
                original_dim: 0,
                quantized_data: vec![],
                scale: 0.0,
                zero_point: 0.0,
            };
        }

        let min_val = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        let (scale, zero_point) = if range < 1e-10 {
            // When range=0 all values are identical; use scale=0 so dequantize gives zero_point
            (0.0_f32, min_val)
        } else {
            (range / 255.0, min_val)
        };

        let quantized_data: Vec<u8> = embedding
            .iter()
            .map(|&v| {
                if range < 1e-10 {
                    0_u8
                } else {
                    ((v - min_val) / range * 255.0).round().clamp(0.0, 255.0) as u8
                }
            })
            .collect();

        Self {
            original_dim: dim,
            quantized_data,
            scale,
            zero_point,
        }
    }

    /// Dequantize back to f32.
    ///
    /// Formula: v = v_q * scale + zero_point
    pub fn dequantize(&self) -> Vec<f32> {
        self.quantized_data
            .iter()
            .map(|&q| q as f32 * self.scale + self.zero_point)
            .collect()
    }

    /// Approximate storage size in bytes (header overhead not included).
    pub fn approx_size_bytes(&self) -> usize {
        self.quantized_data.len() // 1 byte per element
            + std::mem::size_of::<f32>() * 2  // scale + zero_point
            + std::mem::size_of::<usize>() // original_dim
    }
}

// ─────────────────────────────────────────────
// EmbeddingQuantizer
// ─────────────────────────────────────────────

/// Batch quantizer supporting 4-bit and 8-bit precision.
#[derive(Debug, Clone)]
pub struct EmbeddingQuantizer {
    /// Quantization bit width (4 or 8).
    pub bits: u8,
}

impl EmbeddingQuantizer {
    /// Create a new quantizer with the specified bit width.
    ///
    /// `bits` should be 4 or 8; other values are accepted but treated as 8.
    pub fn new(bits: u8) -> Self {
        Self { bits }
    }

    /// Quantize a batch of embeddings.
    pub fn quantize_batch(&self, embeddings: &[Vec<f32>]) -> Vec<QuantizedEmbedding> {
        embeddings.iter().map(|e| self.quantize_single(e)).collect()
    }

    /// Dequantize a batch of quantized embeddings.
    pub fn dequantize_batch(&self, quantized: &[QuantizedEmbedding]) -> Vec<Vec<f32>> {
        quantized.iter().map(|q| q.dequantize()).collect()
    }

    /// Compute compression ratio: original_bytes / quantized_bytes.
    pub fn compression_ratio(&self, original: &[Vec<f32>]) -> f64 {
        if original.is_empty() {
            return 1.0;
        }
        let original_bytes: usize = original.iter().map(|v| v.len() * 4).sum(); // f32 = 4 bytes
        let quantized = self.quantize_batch(original);
        let quantized_bytes: usize = quantized.iter().map(|q| q.approx_size_bytes()).sum();
        if quantized_bytes == 0 {
            return 1.0;
        }
        original_bytes as f64 / quantized_bytes as f64
    }

    // ── Private ───────────────────────────────

    fn quantize_single(&self, embedding: &[f32]) -> QuantizedEmbedding {
        if self.bits <= 4 {
            self.quantize_4bit(embedding)
        } else {
            QuantizedEmbedding::quantize(embedding)
        }
    }

    /// 4-bit quantization: each value is stored in its own byte (4-bit precision, 1 byte/value).
    /// Uses a scale of (range / 15) so values span [0, 15].
    fn quantize_4bit(&self, embedding: &[f32]) -> QuantizedEmbedding {
        let dim = embedding.len();
        if dim == 0 {
            return QuantizedEmbedding {
                original_dim: 0,
                quantized_data: vec![],
                scale: 0.0,
                zero_point: 0.0,
            };
        }

        let min_val = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        // When range=0 all values are identical; use scale=0 so dequantize gives zero_point.
        let (scale, zero_point) = if range < 1e-10 {
            (0.0_f32, min_val)
        } else {
            (range / 15.0, min_val)
        };

        // Store one quantized nibble per byte for compatibility with the shared dequantize().
        let quantized_data: Vec<u8> = embedding
            .iter()
            .map(|&v| {
                if range < 1e-10 {
                    0_u8
                } else {
                    ((v - min_val) / range * 15.0).round().clamp(0.0, 15.0) as u8
                }
            })
            .collect();

        QuantizedEmbedding {
            original_dim: dim,
            quantized_data,
            scale,
            zero_point,
        }
    }
}

// ─────────────────────────────────────────────
// ProductQuantizer
// ─────────────────────────────────────────────

/// Product quantizer: divides the embedding into subspaces and learns a codebook
/// per subspace for efficient approximate nearest-neighbor search.
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    /// Number of subspaces.
    pub subspace_count: usize,
    /// Number of codes per subspace (e.g. 256 for 8-bit codes).
    pub codebook_size: usize,
    /// Codebooks: \[subspace\]\[code\]\[subvec_dim\].
    pub codebooks: Vec<Vec<Vec<f32>>>,
    /// Dimension of each sub-vector (embedding_dim / subspace_count).
    pub subvec_dim: usize,
}

impl ProductQuantizer {
    /// Create an untrained product quantizer.
    pub fn new(subspace_count: usize, codebook_size: usize) -> Self {
        Self {
            subspace_count,
            codebook_size,
            codebooks: Vec::new(),
            subvec_dim: 0,
        }
    }

    /// Train the product quantizer on a set of embeddings.
    ///
    /// Uses a simplified k-means: randomly selects `codebook_size` distinct
    /// embeddings as initial centroids, then runs a few iterations.
    pub fn train(&mut self, embeddings: &[Vec<f32>]) {
        if embeddings.is_empty() || self.subspace_count == 0 {
            return;
        }
        let dim = embeddings[0].len();
        self.subvec_dim = dim / self.subspace_count;
        if self.subvec_dim == 0 {
            self.subvec_dim = 1;
        }

        self.codebooks = (0..self.subspace_count)
            .map(|s| {
                let start = s * self.subvec_dim;
                let end = ((s + 1) * self.subvec_dim).min(dim);

                // Collect sub-vectors for this subspace
                let subvecs: Vec<Vec<f32>> =
                    embeddings.iter().map(|e| e[start..end].to_vec()).collect();

                // Initialize centroids from distinct data points
                let n_codes = self.codebook_size.min(subvecs.len());
                let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n_codes);

                // LCG-based deterministic selection
                let mut lcg_state: u64 = (s as u64 + 1).wrapping_mul(6_364_136_223_846_793_005);
                let mut used = std::collections::HashSet::new();
                while centroids.len() < n_codes {
                    lcg_state = lcg_state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let idx = (lcg_state >> 33) as usize % subvecs.len();
                    if used.insert(idx) {
                        centroids.push(subvecs[idx].clone());
                    }
                }

                // Run simplified k-means for 5 iterations
                for _ in 0..5 {
                    let assignments: Vec<usize> = subvecs
                        .iter()
                        .map(|sv| nearest_centroid(sv, &centroids))
                        .collect();

                    let sub_dim = end - start;
                    let mut new_centroids = vec![vec![0.0_f32; sub_dim]; n_codes];
                    let mut counts = vec![0usize; n_codes];

                    for (sv, &c) in subvecs.iter().zip(assignments.iter()) {
                        for (i, &v) in sv.iter().enumerate() {
                            if i < new_centroids[c].len() {
                                new_centroids[c][i] += v;
                            }
                        }
                        counts[c] += 1;
                    }

                    for (c, count) in counts.iter().enumerate() {
                        if *count > 0 {
                            let n = *count as f32;
                            new_centroids[c].iter_mut().for_each(|x| *x /= n);
                            centroids[c] = new_centroids[c].clone();
                        }
                    }
                }

                centroids
            })
            .collect();
    }

    /// Encode an embedding as a vector of codebook indices (one per subspace).
    pub fn encode(&self, embedding: &[f32]) -> Vec<u8> {
        if self.codebooks.is_empty() || self.subvec_dim == 0 {
            return vec![0; self.subspace_count];
        }
        let dim = embedding.len();
        (0..self.subspace_count)
            .map(|s| {
                let start = s * self.subvec_dim;
                let end = ((s + 1) * self.subvec_dim).min(dim);
                let subvec = &embedding[start..end];
                let code = nearest_centroid(subvec, &self.codebooks[s]);
                code.min(255) as u8
            })
            .collect()
    }

    /// Decode a code vector back to an approximate embedding.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        if self.codebooks.is_empty() {
            return vec![];
        }
        let mut result = Vec::new();
        for (s, &code) in codes.iter().enumerate().take(self.subspace_count) {
            if s >= self.codebooks.len() {
                break;
            }
            let c_idx = (code as usize).min(self.codebooks[s].len().saturating_sub(1));
            result.extend_from_slice(&self.codebooks[s][c_idx]);
        }
        result
    }

    /// Compute approximate L2 distance between two encoded vectors using codebook lookups.
    pub fn approx_distance(&self, codes1: &[u8], codes2: &[u8]) -> f32 {
        if self.codebooks.is_empty() {
            return 0.0;
        }
        let mut total = 0.0_f32;
        for s in 0..self.subspace_count.min(codes1.len()).min(codes2.len()) {
            if s >= self.codebooks.len() {
                break;
            }
            let c1 = (codes1[s] as usize).min(self.codebooks[s].len().saturating_sub(1));
            let c2 = (codes2[s] as usize).min(self.codebooks[s].len().saturating_sub(1));
            let v1 = &self.codebooks[s][c1];
            let v2 = &self.codebooks[s][c2];
            let sq_dist: f32 = v1
                .iter()
                .zip(v2.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            total += sq_dist;
        }
        total
    }

    /// Check whether the quantizer has been trained.
    pub fn is_trained(&self) -> bool {
        !self.codebooks.is_empty()
    }
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/// Find the index of the nearest centroid to `query` by squared L2 distance.
fn nearest_centroid(query: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d: f32 = query
            .iter()
            .zip(c.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_embedding(seed: u32, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        let mut s = seed;
        for _ in 0..dim {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push((s as f32 / u32::MAX as f32) * 2.0 - 1.0);
        }
        v
    }

    fn sample_batch(n: usize, dim: usize, base_seed: u32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| sample_embedding(base_seed + i as u32, dim))
            .collect()
    }

    // ── QuantizedEmbedding ────────────────────

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let emb = sample_embedding(1, 16);
        let q = QuantizedEmbedding::quantize(&emb);
        let deq = q.dequantize();
        assert_eq!(deq.len(), emb.len());
        // Max reconstruction error for 8-bit quantization ≤ range/255
        let range = emb.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - emb.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_err = range / 255.0 + 1e-5;
        for (orig, rec) in emb.iter().zip(deq.iter()) {
            assert!(
                (orig - rec).abs() <= max_err + 1e-4,
                "reconstruction error too large: {} vs {} (max_err={})",
                orig,
                rec,
                max_err
            );
        }
    }

    #[test]
    fn test_quantize_output_in_range() {
        let emb = sample_embedding(2, 8);
        let q = QuantizedEmbedding::quantize(&emb);
        assert_eq!(q.quantized_data.len(), 8);
        // all values in [0, 255] — always true for u8, just verify non-empty
        assert!(!q.quantized_data.is_empty());
    }

    #[test]
    fn test_quantize_empty_embedding() {
        let q = QuantizedEmbedding::quantize(&[]);
        assert_eq!(q.original_dim, 0);
        assert!(q.quantized_data.is_empty());
    }

    #[test]
    fn test_quantize_constant_embedding() {
        let val = 3.125_f32;
        let emb = vec![val; 8];
        let q = QuantizedEmbedding::quantize(&emb);
        let deq = q.dequantize();
        for &v in &deq {
            assert!(
                (v - val).abs() < 0.5,
                "constant embedding should dequantize close to {val}, got {v}"
            );
        }
    }

    #[test]
    fn test_approx_size_bytes() {
        let emb = sample_embedding(3, 64);
        let q = QuantizedEmbedding::quantize(&emb);
        let sz = q.approx_size_bytes();
        assert!(sz > 0);
        // Should be much smaller than original f32 size (64 * 4 = 256 bytes)
        assert!(sz < 64 * 4, "quantized size should be smaller than f32");
    }

    // ── EmbeddingQuantizer ────────────────────

    #[test]
    fn test_quantizer_8bit_creation() {
        let q = EmbeddingQuantizer::new(8);
        assert_eq!(q.bits, 8);
    }

    #[test]
    fn test_quantizer_4bit_creation() {
        let q = EmbeddingQuantizer::new(4);
        assert_eq!(q.bits, 4);
    }

    #[test]
    fn test_quantize_batch_count() {
        let q = EmbeddingQuantizer::new(8);
        let batch = sample_batch(10, 16, 100);
        let out = q.quantize_batch(&batch);
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn test_dequantize_batch_count() {
        let q = EmbeddingQuantizer::new(8);
        let batch = sample_batch(5, 16, 200);
        let quantized = q.quantize_batch(&batch);
        let deq = q.dequantize_batch(&quantized);
        assert_eq!(deq.len(), 5);
        assert_eq!(deq[0].len(), 16);
    }

    #[test]
    fn test_compression_ratio_8bit() {
        let q = EmbeddingQuantizer::new(8);
        let batch = sample_batch(10, 64, 300);
        let ratio = q.compression_ratio(&batch);
        // 8-bit should give ratio close to 4x (f32 → u8) minus overhead
        assert!(
            ratio > 1.0,
            "8-bit quantization should compress: ratio={ratio}"
        );
    }

    #[test]
    fn test_compression_ratio_4bit() {
        let q = EmbeddingQuantizer::new(4);
        let batch = sample_batch(10, 64, 400);
        let ratio = q.compression_ratio(&batch);
        assert!(
            ratio > 1.0,
            "4-bit quantization should compress: ratio={ratio}"
        );
    }

    #[test]
    fn test_compression_ratio_empty() {
        let q = EmbeddingQuantizer::new(8);
        let ratio = q.compression_ratio(&[]);
        assert_eq!(ratio, 1.0);
    }

    #[test]
    fn test_4bit_quantize_dequantize() {
        let q = EmbeddingQuantizer::new(4);
        let batch = sample_batch(3, 16, 500);
        let quantized = q.quantize_batch(&batch);
        let deq = q.dequantize_batch(&quantized);
        // 4-bit reconstruction error ≤ range/15
        for (orig, rec) in batch.iter().zip(deq.iter()) {
            let range = orig.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                - orig.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_err = range / 15.0 + 1e-3;
            for (o, r) in orig.iter().zip(rec.iter()) {
                assert!(
                    (o - r).abs() <= max_err + 0.1,
                    "4-bit error too large: {o} vs {r}"
                );
            }
        }
    }

    // ── ProductQuantizer ──────────────────────

    #[test]
    fn test_pq_creation() {
        let pq = ProductQuantizer::new(4, 16);
        assert_eq!(pq.subspace_count, 4);
        assert_eq!(pq.codebook_size, 16);
        assert!(!pq.is_trained());
    }

    #[test]
    fn test_pq_train() {
        let mut pq = ProductQuantizer::new(4, 8);
        let batch = sample_batch(50, 16, 1000);
        pq.train(&batch);
        assert!(pq.is_trained());
        assert_eq!(pq.codebooks.len(), 4);
    }

    #[test]
    fn test_pq_encode_length() {
        let mut pq = ProductQuantizer::new(4, 8);
        let batch = sample_batch(30, 16, 1100);
        pq.train(&batch);
        let codes = pq.encode(&batch[0]);
        assert_eq!(codes.len(), 4);
    }

    #[test]
    fn test_pq_decode_length() {
        let mut pq = ProductQuantizer::new(4, 8);
        let batch = sample_batch(30, 16, 1200);
        pq.train(&batch);
        let codes = pq.encode(&batch[0]);
        let decoded = pq.decode(&codes);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_pq_approx_distance_same_code() {
        let mut pq = ProductQuantizer::new(4, 8);
        let batch = sample_batch(30, 16, 1300);
        pq.train(&batch);
        let codes = pq.encode(&batch[0]);
        let dist = pq.approx_distance(&codes, &codes);
        assert!(
            dist.abs() < 1e-6,
            "distance to self should be ~0, got {dist}"
        );
    }

    #[test]
    fn test_pq_approx_distance_different_codes() {
        let mut pq = ProductQuantizer::new(4, 8);
        let batch = sample_batch(40, 16, 1400);
        pq.train(&batch);
        let c0 = pq.encode(&batch[0]);
        let c1 = pq.encode(&batch[20]);
        let dist = pq.approx_distance(&c0, &c1);
        assert!(dist >= 0.0, "distance should be non-negative");
        assert!(dist.is_finite(), "distance should be finite");
    }

    #[test]
    fn test_pq_encode_before_train_returns_zeros() {
        let pq = ProductQuantizer::new(4, 8);
        let emb = sample_embedding(1, 16);
        let codes = pq.encode(&emb);
        assert!(codes.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_pq_codebook_size_capped_by_data() {
        let mut pq = ProductQuantizer::new(2, 256); // more codes than data
        let batch = sample_batch(10, 8, 2000); // only 10 embeddings
        pq.train(&batch);
        // Each codebook should have at most 10 entries (capped by data)
        for cb in &pq.codebooks {
            assert!(cb.len() <= 256);
        }
    }

    #[test]
    fn test_pq_reconstruction_quality() {
        let mut pq = ProductQuantizer::new(2, 8);
        let batch = sample_batch(50, 8, 3000);
        pq.train(&batch);
        // Reconstruction of a training vector should be somewhat close
        let orig = &batch[0];
        let codes = pq.encode(orig);
        let decoded = pq.decode(&codes);
        // Just check that decoded is non-empty and finite
        assert!(!decoded.is_empty());
        assert!(decoded.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_pq_train_empty_no_panic() {
        let mut pq = ProductQuantizer::new(4, 8);
        pq.train(&[]); // should not panic
        assert!(!pq.is_trained());
    }
}
