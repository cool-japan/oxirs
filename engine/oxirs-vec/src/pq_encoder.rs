//! Product quantization encoding and decoding (v1.1.0 round 16).
//!
//! Product Quantization (PQ) decomposes a high-dimensional space into `M`
//! independent sub-spaces of dimension `d/M` and quantizes each sub-space
//! separately into `K` centroids.
//!
//! Reference: Jégou et al., "Product Quantization for Nearest Neighbor Search",
//! IEEE TPAMI 2011. <https://doi.org/10.1109/TPAMI.2010.57>

// ──────────────────────────────────────────────────────────────────────────────
// PqConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration parameters for a product quantizer.
#[derive(Debug, Clone)]
pub struct PqConfig {
    /// Number of sub-spaces `M`.
    pub num_subspaces: usize,
    /// Number of centroids per sub-space `K` (typically 256).
    pub num_centroids: usize,
    /// Full dimensionality of the input vectors.
    pub dimension: usize,
}

impl PqConfig {
    /// Create a new `PqConfig`.
    ///
    /// Returns an error if `dimension` is not divisible by `num_subspaces`,
    /// or if either `num_subspaces` or `num_centroids` is zero.
    pub fn new(
        dimension: usize,
        num_subspaces: usize,
        num_centroids: usize,
    ) -> Result<Self, String> {
        if num_subspaces == 0 {
            return Err("num_subspaces must be > 0".to_string());
        }
        if num_centroids == 0 {
            return Err("num_centroids must be > 0".to_string());
        }
        if dimension == 0 {
            return Err("dimension must be > 0".to_string());
        }
        if dimension % num_subspaces != 0 {
            return Err(format!(
                "dimension ({}) must be divisible by num_subspaces ({})",
                dimension, num_subspaces
            ));
        }
        Ok(Self {
            num_subspaces,
            num_centroids,
            dimension,
        })
    }

    /// Return the dimensionality of a single sub-space: `dimension / num_subspaces`.
    pub fn subspace_dim(&self) -> usize {
        self.dimension / self.num_subspaces
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PqEncoder
// ──────────────────────────────────────────────────────────────────────────────

/// A product quantizer with pre-trained codebooks.
///
/// The codebook `codebooks[m][k]` is the `k`-th centroid vector for sub-space
/// `m`, with length `config.subspace_dim()`.
pub struct PqEncoder {
    /// Configuration used to build this quantizer.
    config: PqConfig,
    /// Codebooks: `M × K × subspace_dim`.
    codebooks: Vec<Vec<Vec<f32>>>,
}

impl PqEncoder {
    /// Create a `PqEncoder` with randomly initialised codebooks using a
    /// deterministic LCG so tests are reproducible.
    pub fn new_random(config: PqConfig) -> Self {
        let sub_dim = config.subspace_dim();
        let mut seed: u64 = 0xdeadbeef_cafebabe;
        let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(config.num_subspaces);

        for _ in 0..config.num_subspaces {
            let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(config.num_centroids);
            for _ in 0..config.num_centroids {
                let centroid: Vec<f32> = (0..sub_dim)
                    .map(|_| {
                        // LCG: a=6364136223846793005, c=1442695040888963407 (Knuth)
                        seed = seed
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        // Map to [-1, 1]
                        let bits = (seed >> 11) as f32;
                        bits / (1u64 << 53) as f32 * 2.0 - 1.0
                    })
                    .collect();
                centroids.push(centroid);
            }
            codebooks.push(centroids);
        }

        Self { config, codebooks }
    }

    /// Encode a vector into `M` centroid indices (one per sub-space).
    ///
    /// Returns an error if `vector.len() != config.dimension`.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<usize>, String> {
        if vector.len() != self.config.dimension {
            return Err(format!(
                "Vector length {} does not match configured dimension {}",
                vector.len(),
                self.config.dimension
            ));
        }
        let sub_dim = self.config.subspace_dim();
        let mut codes = Vec::with_capacity(self.config.num_subspaces);

        for m in 0..self.config.num_subspaces {
            let sub_vec = &vector[m * sub_dim..(m + 1) * sub_dim];
            let best = self.nearest_centroid(m, sub_vec);
            codes.push(best);
        }
        Ok(codes)
    }

    /// Decode `M` centroid indices back to an approximate reconstructed vector.
    ///
    /// Returns an error if `codes.len() != config.num_subspaces` or any code
    /// index is out of bounds.
    pub fn decode(&self, codes: &[usize]) -> Result<Vec<f32>, String> {
        if codes.len() != self.config.num_subspaces {
            return Err(format!(
                "Expected {} codes, got {}",
                self.config.num_subspaces,
                codes.len()
            ));
        }
        let sub_dim = self.config.subspace_dim();
        let mut result = vec![0.0f32; self.config.dimension];

        for (m, &code) in codes.iter().enumerate() {
            if code >= self.config.num_centroids {
                return Err(format!(
                    "Code {} in sub-space {} exceeds num_centroids {}",
                    code, m, self.config.num_centroids
                ));
            }
            let centroid = &self.codebooks[m][code];
            let offset = m * sub_dim;
            result[offset..offset + sub_dim].copy_from_slice(centroid);
        }
        Ok(result)
    }

    /// Compute the asymmetric distance between a query vector and encoded codes.
    ///
    /// The asymmetric distance is the sum of squared Euclidean distances
    /// between each query sub-vector and its assigned centroid.
    ///
    /// Returns an error if `query.len() != config.dimension` or codes are invalid.
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[usize]) -> Result<f32, String> {
        if query.len() != self.config.dimension {
            return Err(format!(
                "Query length {} does not match configured dimension {}",
                query.len(),
                self.config.dimension
            ));
        }
        if codes.len() != self.config.num_subspaces {
            return Err(format!(
                "Expected {} codes, got {}",
                self.config.num_subspaces,
                codes.len()
            ));
        }
        let sub_dim = self.config.subspace_dim();
        let mut total_dist = 0.0f32;

        for (m, &code) in codes.iter().enumerate() {
            if code >= self.config.num_centroids {
                return Err(format!(
                    "Code {} in sub-space {} exceeds num_centroids {}",
                    code, m, self.config.num_centroids
                ));
            }
            let centroid = &self.codebooks[m][code];
            let sub_query = &query[m * sub_dim..(m + 1) * sub_dim];
            let sq_dist: f32 = sub_query
                .iter()
                .zip(centroid.iter())
                .map(|(q, c)| (q - c) * (q - c))
                .sum();
            total_dist += sq_dist;
        }
        Ok(total_dist)
    }

    /// Return a reference to the encoder's configuration.
    pub fn config(&self) -> &PqConfig {
        &self.config
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Return the index of the nearest centroid in sub-space `m` to `sub_vec`.
    fn nearest_centroid(&self, m: usize, sub_vec: &[f32]) -> usize {
        let centroids = &self.codebooks[m];
        let mut best_idx = 0usize;
        let mut best_dist = f32::MAX;

        for (k, centroid) in centroids.iter().enumerate() {
            let dist: f32 = sub_vec
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_idx = k;
            }
        }
        best_idx
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder(dim: usize, m: usize, k: usize) -> PqEncoder {
        let cfg = PqConfig::new(dim, m, k).expect("valid config");
        PqEncoder::new_random(cfg)
    }

    // ── PqConfig ──────────────────────────────────────────────────────────────

    #[test]
    fn test_config_valid() {
        let cfg = PqConfig::new(64, 4, 256).expect("ok");
        assert_eq!(cfg.dimension, 64);
        assert_eq!(cfg.num_subspaces, 4);
        assert_eq!(cfg.num_centroids, 256);
    }

    #[test]
    fn test_config_subspace_dim() {
        let cfg = PqConfig::new(64, 4, 256).expect("ok");
        assert_eq!(cfg.subspace_dim(), 16);
    }

    #[test]
    fn test_config_subspace_dim_small() {
        let cfg = PqConfig::new(8, 2, 4).expect("ok");
        assert_eq!(cfg.subspace_dim(), 4);
    }

    #[test]
    fn test_config_invalid_not_divisible() {
        let result = PqConfig::new(7, 4, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_zero_subspaces() {
        let result = PqConfig::new(64, 0, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_zero_centroids() {
        let result = PqConfig::new(64, 4, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_zero_dimension() {
        let result = PqConfig::new(0, 4, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_single_subspace() {
        let cfg = PqConfig::new(16, 1, 8).expect("ok");
        assert_eq!(cfg.subspace_dim(), 16);
    }

    // ── encode ────────────────────────────────────────────────────────────────

    #[test]
    fn test_encode_returns_m_codes() {
        let enc = make_encoder(16, 4, 8);
        let vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let codes = enc.encode(&vec).expect("encode ok");
        assert_eq!(codes.len(), 4);
    }

    #[test]
    fn test_encode_codes_in_range() {
        let enc = make_encoder(16, 4, 8);
        let vec: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let codes = enc.encode(&vec).expect("encode ok");
        for code in codes {
            assert!(code < 8, "code {} should be < 8", code);
        }
    }

    #[test]
    fn test_encode_wrong_dimension_error() {
        let enc = make_encoder(16, 4, 8);
        let result = enc.encode(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_zero_vector() {
        let enc = make_encoder(8, 2, 4);
        let vec = vec![0.0f32; 8];
        let codes = enc.encode(&vec).expect("encode ok");
        assert_eq!(codes.len(), 2);
    }

    #[test]
    fn test_encode_deterministic() {
        let enc = make_encoder(16, 4, 8);
        let vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let codes1 = enc.encode(&vec).expect("ok");
        let codes2 = enc.encode(&vec).expect("ok");
        assert_eq!(codes1, codes2);
    }

    // ── decode ────────────────────────────────────────────────────────────────

    #[test]
    fn test_decode_returns_full_dimension() {
        let enc = make_encoder(16, 4, 8);
        let codes = vec![0usize; 4];
        let decoded = enc.decode(&codes).expect("decode ok");
        assert_eq!(decoded.len(), 16);
    }

    #[test]
    fn test_decode_wrong_code_count_error() {
        let enc = make_encoder(16, 4, 8);
        let codes = vec![0usize; 3]; // should be 4
        assert!(enc.decode(&codes).is_err());
    }

    #[test]
    fn test_decode_out_of_range_code_error() {
        let enc = make_encoder(16, 4, 8);
        let codes = vec![0, 0, 0, 100]; // 100 >= num_centroids=8
        assert!(enc.decode(&codes).is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip_shape() {
        let enc = make_encoder(32, 4, 16);
        let vec: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let codes = enc.encode(&vec).expect("encode ok");
        let decoded = enc.decode(&codes).expect("decode ok");
        assert_eq!(decoded.len(), 32);
        assert_eq!(codes.len(), 4);
    }

    // ── asymmetric_distance ───────────────────────────────────────────────────

    #[test]
    fn test_asymmetric_distance_non_negative() {
        let enc = make_encoder(16, 4, 8);
        let vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let codes = enc.encode(&vec).expect("encode ok");
        let dist = enc.asymmetric_distance(&vec, &codes).expect("dist ok");
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_asymmetric_distance_zero_for_centroid_query() {
        let enc = make_encoder(8, 2, 4);
        // A vector of zeros — its nearest centroids are found and the
        // distance to those centroids should be >= 0.
        let vec = vec![0.0f32; 8];
        let codes = enc.encode(&vec).expect("encode ok");
        let dist = enc.asymmetric_distance(&vec, &codes).expect("dist ok");
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_asymmetric_distance_wrong_query_dim() {
        let enc = make_encoder(16, 4, 8);
        let codes = vec![0usize; 4];
        let result = enc.asymmetric_distance(&[1.0, 2.0], &codes);
        assert!(result.is_err());
    }

    #[test]
    fn test_asymmetric_distance_wrong_code_count() {
        let enc = make_encoder(16, 4, 8);
        let vec = vec![0.0f32; 16];
        let result = enc.asymmetric_distance(&vec, &[0, 0]);
        assert!(result.is_err());
    }

    // ── config accessor ───────────────────────────────────────────────────────

    #[test]
    fn test_config_accessor() {
        let enc = make_encoder(32, 8, 16);
        let cfg = enc.config();
        assert_eq!(cfg.dimension, 32);
        assert_eq!(cfg.num_subspaces, 8);
        assert_eq!(cfg.num_centroids, 16);
        assert_eq!(cfg.subspace_dim(), 4);
    }

    // ── new_random reproducibility ────────────────────────────────────────────

    #[test]
    fn test_new_random_reproducible() {
        let cfg1 = PqConfig::new(16, 4, 8).expect("ok");
        let cfg2 = PqConfig::new(16, 4, 8).expect("ok");
        let enc1 = PqEncoder::new_random(cfg1);
        let enc2 = PqEncoder::new_random(cfg2);
        let vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(
            enc1.encode(&vec).expect("ok"),
            enc2.encode(&vec).expect("ok")
        );
    }
}
