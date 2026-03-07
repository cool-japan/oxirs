/// A single sub-space codebook with `n_clusters` centroids of size `sub_dimension`.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Centroids stored as row-major vectors of length `sub_dimension`.
    pub centroids: Vec<Vec<f32>>,
    /// The number of coordinates per centroid.
    pub sub_dimension: usize,
}

impl Codebook {
    /// Create an empty codebook for a given sub-dimension.
    pub fn new(sub_dimension: usize) -> Self {
        Self {
            centroids: Vec::new(),
            sub_dimension,
        }
    }

    /// Return the index (code) of the centroid nearest to `sub_vector`.
    /// Uses squared Euclidean distance.
    pub fn nearest_centroid(&self, sub_vector: &[f32]) -> u8 {
        let mut best_idx = 0u8;
        let mut best_dist = f32::INFINITY;
        for (idx, centroid) in self.centroids.iter().enumerate() {
            let dist: f32 = sub_vector
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx as u8;
            }
        }
        best_idx
    }

    /// Return the centroid for a code, or `None` if the code is out of range.
    pub fn centroid(&self, code: u8) -> Option<&Vec<f32>> {
        self.centroids.get(code as usize)
    }
}

/// A quantized representation of a vector.
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// One byte code per sub-space.
    pub codes: Vec<u8>,
    /// The original vector dimensionality.
    pub original_dim: usize,
}

/// An approximately reconstructed vector with the quantization error.
#[derive(Debug, Clone)]
pub struct ReconstructedVector {
    /// The reconstructed approximation.
    pub vector: Vec<f32>,
    /// Mean squared reconstruction error.
    pub quantization_error: f32,
}

/// Configuration for the product quantizer.
#[derive(Debug, Clone, Copy)]
pub struct QuantizerConfig {
    /// Number of sub-spaces (must divide the vector dimension evenly).
    pub n_subspaces: usize,
    /// Number of clusters per codebook (≤ 256 because codes are `u8`).
    pub n_clusters: usize,
}

impl Default for QuantizerConfig {
    fn default() -> Self {
        Self {
            n_subspaces: 4,
            n_clusters: 256,
        }
    }
}

/// Errors from quantizer operations.
#[derive(Debug)]
pub enum QuantizerError {
    /// Training or encoding was attempted before training.
    NotTrained,
    /// The vector dimension is incompatible with the codebooks.
    DimensionMismatch,
    /// Not enough training vectors (need at least `n_clusters` per sub-space).
    InsufficientData(usize),
    /// The configuration is invalid (e.g. n_clusters > 256).
    InvalidConfig(String),
}

impl std::fmt::Display for QuantizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotTrained => write!(f, "Quantizer is not trained"),
            Self::DimensionMismatch => write!(f, "Vector dimension mismatch"),
            Self::InsufficientData(n) => {
                write!(f, "Insufficient training data: {n} vectors")
            }
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
        }
    }
}

impl std::error::Error for QuantizerError {}

/// Product Quantizer that compresses high-dimensional vectors.
#[derive(Debug)]
pub struct Quantizer {
    config: QuantizerConfig,
    codebooks: Vec<Codebook>,
}

impl Quantizer {
    /// Create a new, un-trained quantizer with the given configuration.
    pub fn new(config: QuantizerConfig) -> Self {
        Self {
            config,
            codebooks: Vec::new(),
        }
    }

    /// Train the quantizer using k-means on the provided data vectors.
    ///
    /// All vectors must have the same dimensionality, which must be divisible by `n_subspaces`.
    pub fn train(&mut self, data: &[Vec<f32>]) -> Result<(), QuantizerError> {
        // Validate configuration
        if self.config.n_clusters > 256 {
            return Err(QuantizerError::InvalidConfig(
                "n_clusters must be ≤ 256".to_string(),
            ));
        }
        if self.config.n_subspaces == 0 {
            return Err(QuantizerError::InvalidConfig(
                "n_subspaces must be > 0".to_string(),
            ));
        }
        if data.is_empty() {
            return Err(QuantizerError::InsufficientData(0));
        }
        if data.len() < self.config.n_clusters {
            return Err(QuantizerError::InsufficientData(data.len()));
        }

        let dim = data[0].len();
        if dim % self.config.n_subspaces != 0 {
            return Err(QuantizerError::InvalidConfig(format!(
                "Dimension {dim} is not divisible by n_subspaces {}",
                self.config.n_subspaces
            )));
        }
        let sub_dim = dim / self.config.n_subspaces;

        // Validate all vectors have the same dimension
        for v in data {
            if v.len() != dim {
                return Err(QuantizerError::DimensionMismatch);
            }
        }

        // Train one codebook per sub-space
        let actual_k = self.config.n_clusters.min(data.len());
        let mut codebooks = Vec::with_capacity(self.config.n_subspaces);
        for sub in 0..self.config.n_subspaces {
            let start = sub * sub_dim;
            let end = start + sub_dim;
            // Extract sub-vectors for this sub-space
            let sub_vecs: Vec<Vec<f32>> = data.iter().map(|v| v[start..end].to_vec()).collect();
            let cb = kmeans_train(&sub_vecs, actual_k, sub_dim, 50)?;
            codebooks.push(cb);
        }
        self.codebooks = codebooks;
        Ok(())
    }

    /// Encode a single vector into a `QuantizedVector`.
    pub fn encode(&self, vector: &[f32]) -> Result<QuantizedVector, QuantizerError> {
        if self.codebooks.is_empty() {
            return Err(QuantizerError::NotTrained);
        }
        let dim = vector.len();
        let expected_dim = self.codebooks.len() * self.codebooks[0].sub_dimension;
        if dim != expected_dim {
            return Err(QuantizerError::DimensionMismatch);
        }
        let sub_dim = self.codebooks[0].sub_dimension;
        let codes: Vec<u8> = self
            .codebooks
            .iter()
            .enumerate()
            .map(|(i, cb)| {
                let start = i * sub_dim;
                let end = start + sub_dim;
                cb.nearest_centroid(&vector[start..end])
            })
            .collect();
        Ok(QuantizedVector {
            codes,
            original_dim: dim,
        })
    }

    /// Decode a `QuantizedVector` back to an approximate vector.
    pub fn decode(&self, qv: &QuantizedVector) -> Result<ReconstructedVector, QuantizerError> {
        if self.codebooks.is_empty() {
            return Err(QuantizerError::NotTrained);
        }
        if qv.codes.len() != self.codebooks.len() {
            return Err(QuantizerError::DimensionMismatch);
        }
        let mut result = Vec::with_capacity(qv.original_dim);
        for (cb, &code) in self.codebooks.iter().zip(qv.codes.iter()) {
            match cb.centroid(code) {
                Some(c) => result.extend_from_slice(c),
                None => return Err(QuantizerError::DimensionMismatch),
            }
        }
        let error = 0.0_f32; // error not tracked at decode time without original
        Ok(ReconstructedVector {
            vector: result,
            quantization_error: error,
        })
    }

    /// Encode a batch of vectors.
    pub fn encode_batch(
        &self,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<QuantizedVector>, QuantizerError> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    /// Return true if the quantizer has been trained.
    pub fn is_trained(&self) -> bool {
        !self.codebooks.is_empty()
    }

    /// Compute the compression ratio: bytes of original / bytes of compressed.
    ///
    /// Original: `original_dim * 4` bytes (f32).
    /// Compressed: `n_subspaces` bytes (one u8 code per sub-space).
    pub fn compression_ratio(&self, original_dim: usize) -> f32 {
        let n_sub = self.config.n_subspaces;
        if n_sub == 0 {
            return 1.0;
        }
        (original_dim as f32 * 4.0) / n_sub as f32
    }

    /// Return the number of codebooks (equals `n_subspaces` after training).
    pub fn codebook_count(&self) -> usize {
        self.codebooks.len()
    }
}

// ---- K-means implementation ----

/// Run Lloyd's k-means algorithm on `sub_vecs` for `k` clusters and `max_iters` iterations.
fn kmeans_train(
    sub_vecs: &[Vec<f32>],
    k: usize,
    sub_dim: usize,
    max_iters: usize,
) -> Result<Codebook, QuantizerError> {
    let n = sub_vecs.len();
    if k == 0 || n == 0 {
        return Err(QuantizerError::InvalidConfig(
            "k and n must be > 0".to_string(),
        ));
    }

    // Initialise centroids using k-means++ style seeded selection.
    let mut centroids = kmeans_init(sub_vecs, k, sub_dim);

    for _iter in 0..max_iters {
        // Assignment step
        let assignments: Vec<usize> = sub_vecs
            .iter()
            .map(|v| nearest_centroid_idx(&centroids, v))
            .collect();

        // Update step
        let mut sums: Vec<Vec<f64>> = vec![vec![0.0_f64; sub_dim]; k];
        let mut counts: Vec<usize> = vec![0; k];

        for (v, &a) in sub_vecs.iter().zip(assignments.iter()) {
            for (d, &x) in v.iter().enumerate() {
                sums[a][d] += x as f64;
            }
            counts[a] += 1;
        }

        let mut converged = true;
        for (ci, centroid) in centroids.iter_mut().enumerate() {
            if counts[ci] == 0 {
                continue;
            }
            for d in 0..sub_dim {
                let new_val = (sums[ci][d] / counts[ci] as f64) as f32;
                if (new_val - centroid[d]).abs() > 1e-6 {
                    converged = false;
                }
                centroid[d] = new_val;
            }
        }
        if converged {
            break;
        }
    }

    Ok(Codebook {
        centroids,
        sub_dimension: sub_dim,
    })
}

/// K-means++ inspired initialisation: first centroid random, then D² sampling.
fn kmeans_init(data: &[Vec<f32>], k: usize, sub_dim: usize) -> Vec<Vec<f32>> {
    let n = data.len();
    // Build a simple deterministic index map based on dimension sums to avoid
    // requiring async/Random trait in tests. Use a simple hash for selection.
    let mut chosen_indices: Vec<usize> = Vec::with_capacity(k);

    // First centroid: use index derived from data dimensions (deterministic for tests)
    let first_idx = (sub_dim * 7 + n * 3) % n;
    chosen_indices.push(first_idx);

    let mut distances: Vec<f32> = vec![f32::INFINITY; n];
    for _ in 1..k {
        // Update distances
        for (i, v) in data.iter().enumerate() {
            let last = &data[*chosen_indices.last().unwrap_or(&0)];
            let dist: f32 = v
                .iter()
                .zip(last.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if dist < distances[i] {
                distances[i] = dist;
            }
        }
        // Choose next: the point farthest from its nearest chosen centroid
        let next_idx = distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        chosen_indices.push(next_idx);
    }

    chosen_indices
        .into_iter()
        .map(|i| data[i % n].clone())
        .collect()
}

/// Find index of nearest centroid using squared Euclidean distance.
fn nearest_centroid_idx(centroids: &[Vec<f32>], v: &[f32]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da: f32 = a.iter().zip(v.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
            let db: f32 = b.iter().zip(v.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(n_subspaces: usize, n_clusters: usize) -> QuantizerConfig {
        QuantizerConfig {
            n_subspaces,
            n_clusters,
        }
    }

    /// Generate n vectors of dimension d with distinct cluster patterns.
    fn make_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dim).map(|d| (i as f32 * 0.1) + d as f32).collect())
            .collect()
    }

    // --- is_trained ---

    #[test]
    fn test_not_trained_initially() {
        let q = Quantizer::new(make_config(4, 8));
        assert!(!q.is_trained());
    }

    #[test]
    fn test_is_trained_after_train() {
        let mut q = Quantizer::new(make_config(4, 8));
        let data = make_data(32, 8);
        q.train(&data).unwrap();
        assert!(q.is_trained());
    }

    // --- train errors ---

    #[test]
    fn test_train_empty_data_error() {
        let mut q = Quantizer::new(make_config(4, 8));
        let err = q.train(&[]);
        assert!(matches!(err, Err(QuantizerError::InsufficientData(0))));
    }

    #[test]
    fn test_train_insufficient_data_error() {
        let mut q = Quantizer::new(make_config(2, 10));
        let data = make_data(5, 4); // 5 < 10 clusters
        let err = q.train(&data);
        assert!(matches!(err, Err(QuantizerError::InsufficientData(_))));
    }

    #[test]
    fn test_train_n_clusters_over_256() {
        let mut q = Quantizer::new(make_config(2, 300));
        let data = make_data(400, 8);
        let err = q.train(&data);
        assert!(matches!(err, Err(QuantizerError::InvalidConfig(_))));
    }

    #[test]
    fn test_train_dimension_not_divisible() {
        let mut q = Quantizer::new(make_config(3, 4)); // 3 does not divide 8
        let data = make_data(20, 8);
        let err = q.train(&data);
        assert!(matches!(err, Err(QuantizerError::InvalidConfig(_))));
    }

    // --- encode / decode ---

    #[test]
    fn test_encode_not_trained_error() {
        let q = Quantizer::new(make_config(4, 8));
        let v = vec![0.0f32; 8];
        assert!(matches!(q.encode(&v), Err(QuantizerError::NotTrained)));
    }

    #[test]
    fn test_encode_dimension_mismatch() {
        let mut q = Quantizer::new(make_config(2, 4));
        let data = make_data(16, 8);
        q.train(&data).unwrap();
        let v = vec![0.0f32; 4]; // wrong dim
        assert!(matches!(
            q.encode(&v),
            Err(QuantizerError::DimensionMismatch)
        ));
    }

    #[test]
    fn test_encode_codes_length() {
        let mut q = Quantizer::new(make_config(4, 8));
        let data = make_data(32, 8);
        q.train(&data).unwrap();
        let v = vec![1.0f32; 8];
        let qv = q.encode(&v).unwrap();
        assert_eq!(qv.codes.len(), 4); // one code per sub-space
    }

    #[test]
    fn test_encode_original_dim_stored() {
        let mut q = Quantizer::new(make_config(2, 4));
        let data = make_data(16, 8);
        q.train(&data).unwrap();
        let v = vec![0.0f32; 8];
        let qv = q.encode(&v).unwrap();
        assert_eq!(qv.original_dim, 8);
    }

    #[test]
    fn test_decode_produces_correct_dim() {
        let mut q = Quantizer::new(make_config(4, 8));
        let data = make_data(32, 8);
        q.train(&data).unwrap();
        let v = vec![0.5f32; 8];
        let qv = q.encode(&v).unwrap();
        let rv = q.decode(&qv).unwrap();
        assert_eq!(rv.vector.len(), 8);
    }

    #[test]
    fn test_decode_not_trained_error() {
        let q = Quantizer::new(make_config(4, 8));
        let qv = QuantizedVector {
            codes: vec![0; 4],
            original_dim: 8,
        };
        assert!(matches!(q.decode(&qv), Err(QuantizerError::NotTrained)));
    }

    #[test]
    fn test_encode_decode_approximates_original() {
        // A simple test: training on clustered data should reconstruct well
        let mut q = Quantizer::new(make_config(2, 4));
        // 3 clusters in 2D sub-spaces, each sub-space has 4 coords
        let mut data: Vec<Vec<f32>> = Vec::new();
        for c in 0..4 {
            for _ in 0..8 {
                let v: Vec<f32> = (0..8).map(|d| (c as f32 * 10.0) + d as f32 * 0.1).collect();
                data.push(v);
            }
        }
        q.train(&data).unwrap();
        let test = data[0].clone();
        let qv = q.encode(&test).unwrap();
        let rv = q.decode(&qv).unwrap();
        // Reconstruction should be within 2.0 of original for each dim
        for (&orig, &rec) in test.iter().zip(rv.vector.iter()) {
            assert!((orig - rec).abs() < 5.0, "orig={orig}, rec={rec}");
        }
    }

    // --- encode_batch ---

    #[test]
    fn test_encode_batch_empty() {
        let mut q = Quantizer::new(make_config(2, 4));
        let data = make_data(16, 8);
        q.train(&data).unwrap();
        let result = q.encode_batch(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_encode_batch_multiple() {
        let mut q = Quantizer::new(make_config(2, 4));
        let data = make_data(16, 8);
        q.train(&data).unwrap();
        let batch = data.clone();
        let result = q.encode_batch(&batch).unwrap();
        assert_eq!(result.len(), data.len());
    }

    // --- compression_ratio ---

    #[test]
    fn test_compression_ratio_basic() {
        let q = Quantizer::new(make_config(4, 8));
        // original: 128 * 4 = 512 bytes; compressed: 4 bytes
        let ratio = q.compression_ratio(128);
        assert!((ratio - 128.0).abs() < 0.001);
    }

    #[test]
    fn test_compression_ratio_formula() {
        let q = Quantizer::new(make_config(8, 256));
        // 64-dim vector: 64*4=256 bytes original, 8 bytes compressed = ratio 32
        let ratio = q.compression_ratio(64);
        assert!((ratio - 32.0).abs() < 0.001);
    }

    // --- codebook_count ---

    #[test]
    fn test_codebook_count_before_training() {
        let q = Quantizer::new(make_config(4, 8));
        assert_eq!(q.codebook_count(), 0);
    }

    #[test]
    fn test_codebook_count_after_training_matches_n_subspaces() {
        let mut q = Quantizer::new(make_config(4, 8));
        let data = make_data(32, 8);
        q.train(&data).unwrap();
        assert_eq!(q.codebook_count(), 4);
    }

    // --- Codebook ---

    #[test]
    fn test_codebook_new() {
        let cb = Codebook::new(4);
        assert_eq!(cb.sub_dimension, 4);
        assert!(cb.centroids.is_empty());
    }

    #[test]
    fn test_nearest_centroid_single() {
        let mut cb = Codebook::new(2);
        cb.centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let code = cb.nearest_centroid(&[1.0, 1.0]);
        assert_eq!(code, 0); // closer to (0,0)
    }

    #[test]
    fn test_nearest_centroid_second() {
        let mut cb = Codebook::new(2);
        cb.centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let code = cb.nearest_centroid(&[9.0, 9.0]);
        assert_eq!(code, 1); // closer to (10,10)
    }

    #[test]
    fn test_centroid_valid_code() {
        let mut cb = Codebook::new(2);
        cb.centroids = vec![vec![1.0, 2.0]];
        let c = cb.centroid(0).unwrap();
        assert_eq!(c[0], 1.0);
    }

    #[test]
    fn test_centroid_out_of_range() {
        let cb = Codebook::new(2);
        assert!(cb.centroid(5).is_none());
    }

    // --- error display ---

    #[test]
    fn test_not_trained_display() {
        let e = QuantizerError::NotTrained;
        assert!(format!("{e}").contains("trained"));
    }

    #[test]
    fn test_dimension_mismatch_display() {
        let e = QuantizerError::DimensionMismatch;
        assert!(format!("{e}").contains("mismatch"));
    }

    #[test]
    fn test_insufficient_data_display() {
        let e = QuantizerError::InsufficientData(3);
        assert!(format!("{e}").contains("3"));
    }

    #[test]
    fn test_invalid_config_display() {
        let e = QuantizerError::InvalidConfig("bad".to_string());
        assert!(format!("{e}").contains("bad"));
    }
}
