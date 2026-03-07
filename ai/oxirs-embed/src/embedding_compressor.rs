//! Random projection for embedding dimensionality reduction.
//!
//! Implements Achlioptas (2003) sparse random projection for efficient
//! dimensionality reduction of embedding vectors while approximately
//! preserving pairwise distances.

/// Configuration for random projection compression.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Original embedding dimension.
    pub input_dim: usize,
    /// Target (compressed) dimension.
    pub output_dim: usize,
    /// Random seed for reproducible projections.
    pub seed: u64,
}

/// Random projection matrix compressor using Achlioptas (2003) sparse projection.
///
/// Each entry of the projection matrix is +sqrt(3), 0, or -sqrt(3)
/// with probabilities 1/6, 2/3, 1/6 respectively.
pub struct EmbeddingCompressor {
    config: CompressionConfig,
    /// Projection matrix [output_dim x input_dim].
    projection: Vec<Vec<f32>>,
}

/// Simple LCG random number generator for seeded, reproducible projections.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance and return next u64 in [0, u64::MAX].
    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Knuth TAOCP Vol 2
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return a value in [0, 5] uniformly.
    fn next_sixths(&mut self) -> u64 {
        self.next_u64() % 6
    }
}

impl EmbeddingCompressor {
    /// Build compressor with a sparse random projection matrix (Achlioptas, 2003).
    ///
    /// Each matrix entry is:
    /// - +sqrt(3) with probability 1/6
    /// - 0        with probability 2/3
    /// - -sqrt(3) with probability 1/6
    pub fn new(config: CompressionConfig) -> Self {
        let scale = (3.0_f32).sqrt();
        let mut rng = LcgRng::new(config.seed);
        let mut projection = Vec::with_capacity(config.output_dim);

        for _ in 0..config.output_dim {
            let mut row = Vec::with_capacity(config.input_dim);
            for _ in 0..config.input_dim {
                let val = match rng.next_sixths() {
                    0 => scale,   // prob 1/6
                    5 => -scale,  // prob 1/6
                    _ => 0.0_f32, // prob 4/6 = 2/3
                };
                row.push(val);
            }
            projection.push(row);
        }

        Self { config, projection }
    }

    /// Compress a single embedding vector.
    ///
    /// Returns an error if the input length does not match `input_dim`.
    pub fn compress(&self, embedding: &[f32]) -> Result<Vec<f32>, String> {
        if embedding.len() != self.config.input_dim {
            return Err(format!(
                "Expected embedding of length {}, got {}",
                self.config.input_dim,
                embedding.len()
            ));
        }

        let scale = 1.0_f32 / (self.config.output_dim as f32).sqrt();
        let compressed = self
            .projection
            .iter()
            .map(|row| {
                let dot: f32 = row.iter().zip(embedding.iter()).map(|(r, e)| r * e).sum();
                dot * scale
            })
            .collect();

        Ok(compressed)
    }

    /// Compress a batch of embeddings.
    ///
    /// Returns an error if any embedding has incorrect length.
    pub fn compress_batch(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        embeddings.iter().map(|e| self.compress(e)).collect()
    }

    /// Approximate the similarity preservation ratio between two vectors.
    ///
    /// Computes cosine similarity in both original and compressed spaces and
    /// returns the ratio (compressed / original). Should be close to 1.0 for
    /// high-dimensional inputs (Johnson-Lindenstrauss lemma).
    pub fn similarity_preservation_ratio(&self, a: &[f32], b: &[f32]) -> Result<f32, String> {
        let original_sim = cosine_similarity(a, b)?;
        let a_comp = self.compress(a)?;
        let b_comp = self.compress(b)?;
        let compressed_sim = cosine_similarity(&a_comp, &b_comp)?;

        // Avoid division by zero; if original similarity is ~0, return compressed_sim
        if original_sim.abs() < 1e-9 {
            return Ok(compressed_sim.abs());
        }

        Ok(compressed_sim / original_sim)
    }

    /// Return a reference to the compression configuration.
    pub fn config(&self) -> &CompressionConfig {
        &self.config
    }

    /// Return the compression ratio (input_dim / output_dim).
    pub fn compression_ratio(&self) -> f32 {
        self.config.input_dim as f32 / self.config.output_dim as f32
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
    }
    if a.is_empty() {
        return Err("Cannot compute cosine similarity of empty vectors".to_string());
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-9 || norm_b < 1e-9 {
        return Ok(0.0);
    }

    Ok(dot / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(input_dim: usize, output_dim: usize, seed: u64) -> CompressionConfig {
        CompressionConfig {
            input_dim,
            output_dim,
            seed,
        }
    }

    fn make_vec(dim: usize, val: f32) -> Vec<f32> {
        vec![val; dim]
    }

    fn unit_vec(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[idx] = 1.0;
        v
    }

    // --- Construction ---

    #[test]
    fn test_new_creates_correct_projection_dims() {
        let cfg = make_config(128, 32, 42);
        let c = EmbeddingCompressor::new(cfg);
        assert_eq!(c.projection.len(), 32);
        for row in &c.projection {
            assert_eq!(row.len(), 128);
        }
    }

    #[test]
    fn test_new_entries_are_valid_achlioptas_values() {
        let scale = (3.0_f32).sqrt();
        let cfg = make_config(64, 16, 7);
        let c = EmbeddingCompressor::new(cfg);
        for row in &c.projection {
            for &v in row {
                assert!(
                    (v - scale).abs() < 1e-6 || v.abs() < 1e-6 || (v + scale).abs() < 1e-6,
                    "Unexpected value: {v}"
                );
            }
        }
    }

    #[test]
    fn test_seed_reproducibility() {
        let cfg1 = make_config(64, 16, 99);
        let cfg2 = make_config(64, 16, 99);
        let c1 = EmbeddingCompressor::new(cfg1);
        let c2 = EmbeddingCompressor::new(cfg2);
        assert_eq!(c1.projection, c2.projection);
    }

    #[test]
    fn test_different_seeds_produce_different_matrices() {
        let c1 = EmbeddingCompressor::new(make_config(64, 16, 1));
        let c2 = EmbeddingCompressor::new(make_config(64, 16, 2));
        // With high probability the matrices differ
        assert_ne!(c1.projection, c2.projection);
    }

    // --- compress ---

    #[test]
    fn test_compress_output_length_equals_output_dim() {
        let cfg = make_config(128, 32, 0);
        let c = EmbeddingCompressor::new(cfg);
        let v = make_vec(128, 1.0);
        let out = c.compress(&v).expect("compress should succeed");
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn test_compress_wrong_input_length_returns_error() {
        let cfg = make_config(128, 32, 0);
        let c = EmbeddingCompressor::new(cfg);
        let v = make_vec(64, 1.0);
        let result = c.compress(&v);
        assert!(result.is_err());
    }

    #[test]
    fn test_compress_zero_vector() {
        let cfg = make_config(64, 16, 5);
        let c = EmbeddingCompressor::new(cfg);
        let v = make_vec(64, 0.0);
        let out = c.compress(&v).expect("compress should succeed");
        for &x in &out {
            assert!((x).abs() < 1e-9, "Expected zero vector, got {x}");
        }
    }

    #[test]
    fn test_compress_single_dim_input() {
        let cfg = make_config(1, 1, 0);
        let c = EmbeddingCompressor::new(cfg);
        let v = vec![2.0_f32];
        let out = c.compress(&v).expect("compress should succeed");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_compress_exact_output_dimension() {
        for (input_dim, output_dim) in [(256, 64), (512, 128), (100, 50)] {
            let cfg = make_config(input_dim, output_dim, 42);
            let c = EmbeddingCompressor::new(cfg);
            let v = make_vec(input_dim, 1.0);
            let out = c.compress(&v).expect("compress ok");
            assert_eq!(out.len(), output_dim);
        }
    }

    #[test]
    fn test_compress_is_deterministic() {
        let cfg = make_config(64, 16, 13);
        let c = EmbeddingCompressor::new(cfg);
        let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let out1 = c.compress(&v).expect("ok");
        let out2 = c.compress(&v).expect("ok");
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_compress_linearity_scalar_multiple() {
        let cfg = make_config(32, 8, 17);
        let c = EmbeddingCompressor::new(cfg);
        let v: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let out1 = c.compress(&v).expect("ok");
        let v2: Vec<f32> = v.iter().map(|&x| x * 2.0).collect();
        let out2 = c.compress(&v2).expect("ok");
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((b - 2.0 * a).abs() < 1e-5, "Linearity failed: {a} vs {b}");
        }
    }

    #[test]
    fn test_compress_unit_vector() {
        let cfg = make_config(32, 8, 11);
        let c = EmbeddingCompressor::new(cfg);
        let v = unit_vec(32, 0);
        let out = c.compress(&v).expect("ok");
        assert_eq!(out.len(), 8);
    }

    // --- compress_batch ---

    #[test]
    fn test_compress_batch_correct_count() {
        let cfg = make_config(64, 16, 0);
        let c = EmbeddingCompressor::new(cfg);
        let batch: Vec<Vec<f32>> = (0..5).map(|_| make_vec(64, 1.0)).collect();
        let result = c.compress_batch(&batch).expect("ok");
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_compress_batch_each_output_length() {
        let cfg = make_config(64, 16, 0);
        let c = EmbeddingCompressor::new(cfg);
        let batch: Vec<Vec<f32>> = (0..3).map(|_| make_vec(64, 1.0)).collect();
        let result = c.compress_batch(&batch).expect("ok");
        for out in &result {
            assert_eq!(out.len(), 16);
        }
    }

    #[test]
    fn test_compress_batch_empty() {
        let cfg = make_config(64, 16, 0);
        let c = EmbeddingCompressor::new(cfg);
        let result = c.compress_batch(&[]).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_compress_batch_error_on_wrong_size() {
        let cfg = make_config(64, 16, 0);
        let c = EmbeddingCompressor::new(cfg);
        let batch = vec![make_vec(64, 1.0), make_vec(32, 1.0)];
        let result = c.compress_batch(&batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_compress_batch_single_element() {
        let cfg = make_config(64, 16, 0);
        let c = EmbeddingCompressor::new(cfg);
        let batch = vec![make_vec(64, 0.5)];
        let result = c.compress_batch(&batch).expect("ok");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 16);
    }

    #[test]
    fn test_compress_batch_matches_individual_compress() {
        let cfg = make_config(32, 8, 55);
        let c = EmbeddingCompressor::new(cfg);
        let v1: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let individual1 = c.compress(&v1).expect("ok");
        let individual2 = c.compress(&v2).expect("ok");
        let batch = c.compress_batch(&[v1, v2]).expect("ok");
        assert_eq!(batch[0], individual1);
        assert_eq!(batch[1], individual2);
    }

    // --- compression_ratio ---

    #[test]
    fn test_compression_ratio_basic() {
        let cfg = make_config(128, 32, 0);
        let c = EmbeddingCompressor::new(cfg);
        let ratio = c.compression_ratio();
        assert!((ratio - 4.0).abs() < 1e-6, "Expected 4.0, got {ratio}");
    }

    #[test]
    fn test_compression_ratio_no_compression() {
        let cfg = make_config(64, 64, 0);
        let c = EmbeddingCompressor::new(cfg);
        assert!((c.compression_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compression_ratio_high() {
        let cfg = make_config(512, 8, 0);
        let c = EmbeddingCompressor::new(cfg);
        assert!((c.compression_ratio() - 64.0).abs() < 1e-6);
    }

    // --- config ---

    #[test]
    fn test_config_returns_correct_input_dim() {
        let cfg = make_config(100, 25, 42);
        let c = EmbeddingCompressor::new(cfg);
        assert_eq!(c.config().input_dim, 100);
    }

    #[test]
    fn test_config_returns_correct_output_dim() {
        let cfg = make_config(100, 25, 42);
        let c = EmbeddingCompressor::new(cfg);
        assert_eq!(c.config().output_dim, 25);
    }

    #[test]
    fn test_config_returns_correct_seed() {
        let cfg = make_config(100, 25, 42);
        let c = EmbeddingCompressor::new(cfg);
        assert_eq!(c.config().seed, 42);
    }

    // --- similarity_preservation_ratio ---

    #[test]
    fn test_similarity_preservation_ratio_in_range() {
        let cfg = make_config(128, 32, 42);
        let c = EmbeddingCompressor::new(cfg);
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.2).cos()).collect();
        let ratio = c.similarity_preservation_ratio(&a, &b).expect("ok");
        // The ratio can be large when original similarity is near zero;
        // just verify it is finite.
        assert!(ratio.is_finite(), "Ratio should be finite: {ratio}");
    }

    #[test]
    fn test_similarity_preservation_parallel_vectors() {
        let cfg = make_config(64, 16, 7);
        let c = EmbeddingCompressor::new(cfg);
        let a = make_vec(64, 1.0);
        let b = make_vec(64, 2.0); // parallel to a
                                   // Both original and compressed cosine sim should be 1.0
        let ratio = c.similarity_preservation_ratio(&a, &b).expect("ok");
        // ratio should be approximately 1.0 (1.0 / 1.0)
        assert!((ratio - 1.0).abs() < 0.5, "Expected ~1.0, got {ratio}");
    }

    #[test]
    fn test_similarity_preservation_wrong_length() {
        let cfg = make_config(64, 16, 7);
        let c = EmbeddingCompressor::new(cfg);
        let a = make_vec(64, 1.0);
        let b = make_vec(32, 1.0); // wrong length
        let result = c.similarity_preservation_ratio(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_similarity_preservation_zero_vector() {
        let cfg = make_config(32, 8, 3);
        let c = EmbeddingCompressor::new(cfg);
        let a = make_vec(32, 0.0); // zero vector
        let b = make_vec(32, 1.0);
        // Should not panic; original cosine similarity is 0
        let result = c.similarity_preservation_ratio(&a, &b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_similarity_preservation_identical_vectors() {
        let cfg = make_config(64, 16, 9);
        let c = EmbeddingCompressor::new(cfg);
        let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let result = c.similarity_preservation_ratio(&a, &a);
        assert!(result.is_ok());
        // Ratio should be close to 1.0 (cosine similarity of identical vectors in both spaces)
        let ratio = result.expect("ok");
        assert!((0.0..=2.0).contains(&ratio), "ratio={ratio}");
    }

    // --- Edge cases ---

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_minimum_dimensions() {
        let cfg = make_config(1, 1, 0);
        let c = EmbeddingCompressor::new(cfg);
        let out = c.compress(&[3.14]).expect("ok");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_large_dimension() {
        let cfg = make_config(1024, 128, 42);
        let c = EmbeddingCompressor::new(cfg);
        let v = make_vec(1024, 0.5);
        let out = c.compress(&v).expect("ok");
        assert_eq!(out.len(), 128);
    }

    #[test]
    fn test_seed_zero() {
        let cfg = make_config(32, 8, 0);
        let c = EmbeddingCompressor::new(cfg);
        let v = make_vec(32, 1.0);
        let out = c.compress(&v).expect("ok");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_projection_sparsity() {
        // Achlioptas matrix should be ~2/3 zeros
        let cfg = make_config(300, 100, 12345);
        let c = EmbeddingCompressor::new(cfg);
        let total: usize = c.projection.len() * c.projection[0].len();
        let zeros: usize = c
            .projection
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v.abs() < 1e-9)
            .count();
        let zero_fraction = zeros as f64 / total as f64;
        // Should be roughly 2/3 zeros (allow ±15% slack)
        assert!(
            zero_fraction > 0.50 && zero_fraction < 0.80,
            "Expected ~2/3 zeros, got {zero_fraction:.3}"
        );
    }

    #[test]
    fn test_batch_size_large() {
        let cfg = make_config(64, 16, 42);
        let c = EmbeddingCompressor::new(cfg);
        let batch: Vec<Vec<f32>> = (0..100).map(|_| make_vec(64, 0.5)).collect();
        let result = c.compress_batch(&batch).expect("ok");
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_different_seeds_compress_differently() {
        let v = make_vec(64, 1.0);
        let c1 = EmbeddingCompressor::new(make_config(64, 16, 1));
        let c2 = EmbeddingCompressor::new(make_config(64, 16, 2));
        let out1 = c1.compress(&v).expect("ok");
        let out2 = c2.compress(&v).expect("ok");
        // With very high probability, outputs differ
        assert_ne!(out1, out2);
    }

    #[test]
    fn test_config_clone() {
        let cfg = make_config(64, 16, 99);
        let c = EmbeddingCompressor::new(cfg.clone());
        assert_eq!(c.config().input_dim, cfg.input_dim);
        assert_eq!(c.config().output_dim, cfg.output_dim);
        assert_eq!(c.config().seed, cfg.seed);
    }

    #[test]
    fn test_debug_format_config() {
        let cfg = make_config(64, 16, 42);
        let debug_str = format!("{cfg:?}");
        assert!(debug_str.contains("64"));
        assert!(debug_str.contains("16"));
    }
}
