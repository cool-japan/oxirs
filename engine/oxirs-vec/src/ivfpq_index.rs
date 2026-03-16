// Inverted File Index with Product Quantization (IVF-PQ) compound index
// Added in v1.1.0 Round 7

/// Configuration for an IVF-PQ index.
#[derive(Debug, Clone)]
pub struct IvfPqConfig {
    /// Number of coarse clusters (IVF lists).
    pub nlist: usize,
    /// Number of PQ sub-quantizers (must divide dimension evenly).
    pub m: usize,
    /// Number of centroids per sub-quantizer.
    pub k_per_sub: usize,
    /// Number of coarse clusters to probe at query time.
    pub nprobe: usize,
    /// Vector dimension.
    pub dimension: usize,
}

impl IvfPqConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), IvfPqError> {
        if self.m == 0 {
            return Err(IvfPqError::InvalidConfig("m must be > 0".to_string()));
        }
        if self.dimension == 0 {
            return Err(IvfPqError::InvalidConfig(
                "dimension must be > 0".to_string(),
            ));
        }
        if self.dimension % self.m != 0 {
            return Err(IvfPqError::InvalidConfig(format!(
                "dimension ({}) must be divisible by m ({})",
                self.dimension, self.m
            )));
        }
        if self.nlist == 0 {
            return Err(IvfPqError::InvalidConfig("nlist must be > 0".to_string()));
        }
        if self.nprobe == 0 {
            return Err(IvfPqError::InvalidConfig("nprobe must be > 0".to_string()));
        }
        if self.nprobe > self.nlist {
            return Err(IvfPqError::InvalidConfig(format!(
                "nprobe ({}) must be <= nlist ({})",
                self.nprobe, self.nlist
            )));
        }
        if self.k_per_sub == 0 {
            return Err(IvfPqError::InvalidConfig(
                "k_per_sub must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Errors from IVF-PQ operations.
#[derive(Debug)]
pub enum IvfPqError {
    DimensionMismatch { expected: usize, got: usize },
    NotTrained,
    InvalidConfig(String),
    InsufficientData(String),
}

impl std::fmt::Display for IvfPqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IvfPqError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            }
            IvfPqError::NotTrained => write!(f, "Index has not been trained yet"),
            IvfPqError::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            IvfPqError::InsufficientData(msg) => write!(f, "Insufficient data: {msg}"),
        }
    }
}

impl std::error::Error for IvfPqError {}

/// IVF-PQ approximate nearest neighbor index.
pub struct IvfPqIndex {
    config: IvfPqConfig,
    /// Coarse centroids: nlist vectors of size `dimension`.
    coarse_centroids: Vec<Vec<f64>>,
    /// Per-cluster inverted lists: (vector_id, pq_codes).
    inverted_lists: Vec<Vec<(u64, Vec<u8>)>>,
    /// PQ codebook: m sub-quantizers, each has k_per_sub × (dimension/m) centroids.
    pq_codebook: Vec<Vec<Vec<f64>>>,
    is_trained: bool,
    next_id: u64,
}

impl IvfPqIndex {
    /// Create a new (untrained) IVF-PQ index with the given configuration.
    pub fn new(config: IvfPqConfig) -> Result<Self, IvfPqError> {
        config.validate()?;
        let nlist = config.nlist;
        Ok(Self {
            config,
            coarse_centroids: Vec::new(),
            inverted_lists: vec![Vec::new(); nlist],
            pq_codebook: Vec::new(),
            is_trained: false,
            next_id: 0,
        })
    }

    /// Train the index: build coarse centroids (k-means) and PQ codebook.
    pub fn train(&mut self, vectors: &[Vec<f64>]) -> Result<(), IvfPqError> {
        if vectors.is_empty() {
            return Err(IvfPqError::InsufficientData(
                "Need at least 1 vector to train".to_string(),
            ));
        }
        let n = vectors.len();
        let dim = self.config.dimension;

        // Validate dimensions
        for v in vectors.iter() {
            if v.len() != dim {
                return Err(IvfPqError::DimensionMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
        }

        let nlist = self.config.nlist.min(n); // can't have more clusters than vectors
        let m = self.config.m;
        let k_per_sub = self.config.k_per_sub;
        let sub_dim = dim / m;

        // Step 1: Train coarse centroids (k-means on full vectors)
        self.coarse_centroids = Self::kmeans(vectors, nlist, dim, 10);

        // Step 2: Compute residuals and train PQ codebook
        // For each vector, find its nearest coarse centroid, compute residual
        let residuals: Vec<Vec<f64>> = vectors
            .iter()
            .map(|v| {
                let nearest = self.find_nearest_centroid_trained(v);
                let centroid = &self.coarse_centroids[nearest];
                v.iter().zip(centroid.iter()).map(|(a, b)| a - b).collect()
            })
            .collect();

        // Step 3: For each sub-quantizer, train k_per_sub centroids on residual slices
        let mut pq_codebook = Vec::with_capacity(m);
        for sub_idx in 0..m {
            let start = sub_idx * sub_dim;
            let end = start + sub_dim;
            let sub_data: Vec<Vec<f64>> =
                residuals.iter().map(|r| r[start..end].to_vec()).collect();
            let k = k_per_sub.min(sub_data.len());
            let centroids = Self::kmeans(&sub_data, k, sub_dim, 5);
            pq_codebook.push(centroids);
        }
        self.pq_codebook = pq_codebook;
        self.is_trained = true;

        // Resize inverted lists to actual nlist
        let actual_nlist = self.coarse_centroids.len();
        self.inverted_lists = vec![Vec::new(); actual_nlist];
        Ok(())
    }

    /// Add a vector to the trained index.
    pub fn add(&mut self, vector: &[f64]) -> Result<u64, IvfPqError> {
        if !self.is_trained {
            return Err(IvfPqError::NotTrained);
        }
        let dim = self.config.dimension;
        if vector.len() != dim {
            return Err(IvfPqError::DimensionMismatch {
                expected: dim,
                got: vector.len(),
            });
        }
        let cluster_idx = self.find_nearest_centroid(vector);
        let centroid = &self.coarse_centroids[cluster_idx];
        let residual: Vec<f64> = vector
            .iter()
            .zip(centroid.iter())
            .map(|(a, b)| a - b)
            .collect();
        let codes = self.encode_residual(&residual);
        let id = self.next_id;
        self.next_id += 1;
        self.inverted_lists[cluster_idx].push((id, codes));
        Ok(id)
    }

    /// Add multiple vectors in batch.
    pub fn add_batch(&mut self, vectors: &[Vec<f64>]) -> Result<Vec<u64>, IvfPqError> {
        vectors.iter().map(|v| self.add(v)).collect()
    }

    /// Search for the k nearest neighbors of a query vector.
    ///
    /// Returns (id, approximate_distance) pairs sorted by distance (ascending).
    pub fn search(&self, query: &[f64], k: usize) -> Result<Vec<(u64, f64)>, IvfPqError> {
        if !self.is_trained {
            return Err(IvfPqError::NotTrained);
        }
        let dim = self.config.dimension;
        if query.len() != dim {
            return Err(IvfPqError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        // Find nprobe nearest coarse centroids
        let nprobe = self.config.nprobe.min(self.coarse_centroids.len());
        let mut centroid_dists: Vec<(usize, f64)> = self
            .coarse_centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, Self::l2_distance(query, c)))
            .collect();
        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute PQ distance tables for the query residual against each top cluster
        let sub_dim = dim / self.config.m;
        let m = self.config.m;

        let mut candidates: Vec<(u64, f64)> = Vec::new();

        for &(cluster_idx, _) in centroid_dists.iter().take(nprobe) {
            let centroid = &self.coarse_centroids[cluster_idx];
            let residual: Vec<f64> = query
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| a - b)
                .collect();

            // Build distance tables: for each sub-quantizer, precompute dist to each code centroid
            let dist_tables: Vec<Vec<f64>> = (0..m)
                .map(|sub_idx| {
                    let start = sub_idx * sub_dim;
                    let q_sub = &residual[start..start + sub_dim];
                    self.pq_codebook[sub_idx]
                        .iter()
                        .map(|code_centroid| Self::l2_distance(q_sub, code_centroid))
                        .collect()
                })
                .collect();

            for &(id, ref codes) in &self.inverted_lists[cluster_idx] {
                // Approximate distance via PQ lookup
                let dist: f64 = codes
                    .iter()
                    .enumerate()
                    .map(|(sub_idx, &code)| {
                        let code_idx = code as usize;
                        dist_tables[sub_idx]
                            .get(code_idx)
                            .copied()
                            .unwrap_or(f64::MAX)
                    })
                    .sum();
                candidates.push((id, dist));
            }
        }

        // Sort candidates by distance and take top k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        Ok(candidates)
    }

    /// Total number of vectors added to the index.
    pub fn size(&self) -> usize {
        self.inverted_lists.iter().map(|l| l.len()).sum()
    }

    /// Whether the index has been trained.
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Find the nearest coarse centroid index for a vector (only callable after training).
    pub fn find_nearest_centroid(&self, vector: &[f64]) -> usize {
        self.find_nearest_centroid_trained(vector)
    }

    fn find_nearest_centroid_trained(&self, vector: &[f64]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (i, centroid) in self.coarse_centroids.iter().enumerate() {
            let d = Self::l2_distance(vector, centroid);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Encode a residual vector using the PQ codebook.
    pub fn encode_residual(&self, residual: &[f64]) -> Vec<u8> {
        let sub_dim = self.config.dimension / self.config.m;
        let m = self.config.m;
        let mut codes = Vec::with_capacity(m);
        for sub_idx in 0..m {
            let start = sub_idx * sub_dim;
            let sub = &residual[start..start + sub_dim];
            // Find nearest centroid in PQ codebook for this sub-quantizer
            let mut best_code = 0u8;
            let mut best_dist = f64::MAX;
            for (code_idx, centroid) in self.pq_codebook[sub_idx].iter().enumerate() {
                let d = Self::l2_distance(sub, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_code = (code_idx & 0xFF) as u8;
                }
            }
            codes.push(best_code);
        }
        codes
    }

    /// L2 squared distance between two equal-length slices.
    pub fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
    }

    /// Simple k-means with random initialization (first k vectors) and `iters` iterations.
    pub fn kmeans(data: &[Vec<f64>], k: usize, dim: usize, iters: usize) -> Vec<Vec<f64>> {
        if data.is_empty() || k == 0 {
            return Vec::new();
        }
        let k = k.min(data.len());

        // Initialize centroids: evenly spaced through data
        let mut centroids: Vec<Vec<f64>> =
            (0..k).map(|i| data[i * data.len() / k].clone()).collect();

        for _ in 0..iters {
            // Assign step
            let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
            for (idx, point) in data.iter().enumerate() {
                let best = centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| (ci, Self::l2_distance(point, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(ci, _)| ci)
                    .unwrap_or(0);
                clusters[best].push(idx);
            }

            // Update step
            let mut new_centroids = Vec::with_capacity(k);
            for (ci, members) in clusters.iter().enumerate() {
                if members.is_empty() {
                    // Keep old centroid if cluster is empty
                    new_centroids.push(centroids[ci].clone());
                } else {
                    let mut centroid = vec![0.0f64; dim];
                    for &idx in members {
                        for (d, val) in centroid.iter_mut().zip(data[idx].iter()) {
                            *d += val;
                        }
                    }
                    let count = members.len() as f64;
                    for d in centroid.iter_mut() {
                        *d /= count;
                    }
                    new_centroids.push(centroid);
                }
            }
            centroids = new_centroids;
        }
        centroids
    }
}

#[cfg(test)]
mod tests {
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;

    fn make_config(dim: usize, nlist: usize, m: usize, k: usize, nprobe: usize) -> IvfPqConfig {
        IvfPqConfig {
            nlist,
            m,
            k_per_sub: k,
            nprobe,
            dimension: dim,
        }
    }

    fn make_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
        // Simple LCG pseudo-random for reproducibility
        let mut state = seed;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    // ---- new / config validation ----

    #[test]
    fn test_new_valid_config() {
        let config = make_config(8, 4, 2, 4, 2);
        assert!(IvfPqIndex::new(config).is_ok());
    }

    #[test]
    fn test_new_m_not_divides_dimension() {
        let config = make_config(7, 4, 3, 4, 2); // 7 % 3 != 0
        assert!(matches!(
            IvfPqIndex::new(config),
            Err(IvfPqError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_new_m_zero() {
        let config = make_config(8, 4, 0, 4, 2);
        assert!(matches!(
            IvfPqIndex::new(config),
            Err(IvfPqError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_new_nlist_zero() {
        let config = make_config(8, 0, 2, 4, 2);
        assert!(matches!(
            IvfPqIndex::new(config),
            Err(IvfPqError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_new_nprobe_gt_nlist() {
        let config = make_config(8, 2, 2, 4, 5); // nprobe=5 > nlist=2
        assert!(matches!(
            IvfPqIndex::new(config),
            Err(IvfPqError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_new_dimension_zero() {
        let config = make_config(0, 4, 0, 4, 2);
        assert!(matches!(
            IvfPqIndex::new(config),
            Err(IvfPqError::InvalidConfig(_))
        ));
    }

    // ---- is_trained / train ----

    #[test]
    fn test_not_trained_initially() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let index = IvfPqIndex::new(config)?;
        assert!(!index.is_trained());
        Ok(())
    }

    #[test]
    fn test_train_basic() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let vectors = make_random_vectors(20, 8, 42);
        index.train(&vectors)?;
        assert!(index.is_trained());
        Ok(())
    }

    #[test]
    fn test_train_too_few_vectors() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        // 0 vectors → error
        let result = index.train(&[]);
        assert!(matches!(result, Err(IvfPqError::InsufficientData(_))));
        Ok(())
    }

    #[test]
    fn test_train_dimension_mismatch() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let vectors = vec![vec![1.0, 2.0, 3.0]]; // dim=3, not 8
        let result = index.train(&vectors);
        assert!(matches!(result, Err(IvfPqError::DimensionMismatch { .. })));
        Ok(())
    }

    // ---- add ----

    #[test]
    fn test_add_before_training_error() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let v = vec![0.0; 8];
        let result = index.add(&v);
        assert!(matches!(result, Err(IvfPqError::NotTrained)));
        Ok(())
    }

    #[test]
    fn test_add_after_training() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let vectors = make_random_vectors(20, 8, 1);
        index.train(&vectors)?;
        let id = index.add(&vectors[0])?;
        assert_eq!(id, 0);
        assert_eq!(index.size(), 1);
        Ok(())
    }

    #[test]
    fn test_add_dimension_mismatch() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let vectors = make_random_vectors(20, 8, 2);
        index.train(&vectors)?;
        let bad_v = vec![1.0, 2.0]; // wrong dim
        let result = index.add(&bad_v);
        assert!(matches!(result, Err(IvfPqError::DimensionMismatch { .. })));
        Ok(())
    }

    // ---- add_batch ----

    #[test]
    fn test_add_batch() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let train_data = make_random_vectors(20, 8, 3);
        index.train(&train_data)?;
        let add_data = make_random_vectors(5, 8, 4);
        let ids = index.add_batch(&add_data)?;
        assert_eq!(ids.len(), 5);
        assert_eq!(index.size(), 5);
        Ok(())
    }

    // ---- size ----

    #[test]
    fn test_size_starts_at_zero() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let vectors = make_random_vectors(20, 8, 5);
        index.train(&vectors)?;
        assert_eq!(index.size(), 0);
        Ok(())
    }

    #[test]
    fn test_size_after_adding() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let vectors = make_random_vectors(20, 8, 6);
        index.train(&vectors)?;
        for v in &vectors {
            index.add(v)?;
        }
        assert_eq!(index.size(), 20);
        Ok(())
    }

    // ---- search ----

    #[test]
    fn test_search_before_training_error() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let index = IvfPqIndex::new(config)?;
        let q = vec![0.0; 8];
        let result = index.search(&q, 5);
        assert!(matches!(result, Err(IvfPqError::NotTrained)));
        Ok(())
    }

    #[test]
    fn test_search_empty_index() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let vectors = make_random_vectors(20, 8, 7);
        index.train(&vectors)?;
        let q = vec![0.0; 8];
        let results = index.search(&q, 5)?;
        assert!(results.is_empty());
        Ok(())
    }

    #[test]
    fn test_search_returns_k_results() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let vectors = make_random_vectors(50, 8, 8);
        index.train(&vectors)?;
        for v in &vectors {
            index.add(v)?;
        }
        let q = vec![0.0; 8];
        let results = index.search(&q, 10)?;
        assert!(results.len() <= 10);
        assert!(!results.is_empty());
        Ok(())
    }

    #[test]
    fn test_search_sorted_by_distance() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let vectors = make_random_vectors(30, 8, 9);
        index.train(&vectors)?;
        for v in &vectors {
            index.add(v)?;
        }
        let q = vec![0.0; 8];
        let results = index.search(&q, 10)?;
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1,
                "Results not sorted: {} > {}",
                results[i - 1].1,
                results[i].1
            );
        }
        Ok(())
    }

    #[test]
    fn test_search_dimension_mismatch() -> Result<()> {
        let config = make_config(8, 4, 2, 4, 2);
        let mut index = IvfPqIndex::new(config)?;
        let vectors = make_random_vectors(20, 8, 10);
        index.train(&vectors)?;
        let bad_q = vec![1.0, 2.0]; // wrong dim
        let result = index.search(&bad_q, 5);
        assert!(matches!(result, Err(IvfPqError::DimensionMismatch { .. })));
        Ok(())
    }

    // ---- l2_distance ----

    #[test]
    fn test_l2_distance_zero() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(IvfPqIndex::l2_distance(&a, &a) < 1e-10);
    }

    #[test]
    fn test_l2_distance_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = IvfPqIndex::l2_distance(&a, &b);
        assert!((d - 25.0).abs() < 1e-10); // 3^2 + 4^2 = 25
    }

    // ---- IvfPqError display ----

    #[test]
    fn test_error_display() {
        let e = IvfPqError::DimensionMismatch {
            expected: 8,
            got: 4,
        };
        assert!(format!("{e}").contains("8"));
        let e2 = IvfPqError::NotTrained;
        assert!(!format!("{e2}").is_empty());
        let e3 = IvfPqError::InvalidConfig("m".to_string());
        assert!(format!("{e3}").contains("m"));
        let e4 = IvfPqError::InsufficientData("need more".to_string());
        assert!(format!("{e4}").contains("need more"));
    }

    // ---- IvfPqConfig validation ----

    #[test]
    fn test_config_validation_valid() {
        let config = make_config(8, 4, 2, 4, 2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_k_per_sub_zero() {
        let config = IvfPqConfig {
            nlist: 4,
            m: 2,
            k_per_sub: 0,
            nprobe: 2,
            dimension: 8,
        };
        assert!(matches!(
            config.validate(),
            Err(IvfPqError::InvalidConfig(_))
        ));
    }
}
