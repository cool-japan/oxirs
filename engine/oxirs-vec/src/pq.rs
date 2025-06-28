//! Product Quantization (PQ) for memory-efficient vector compression and search
//!
//! PQ divides high-dimensional vectors into subvectors and quantizes each subvector
//! independently using k-means clustering. This achieves high compression ratios
//! while maintaining reasonable search accuracy.

use crate::{Vector, VectorIndex};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Configuration for Product Quantization
#[derive(Debug, Clone, PartialEq)]
pub struct PQConfig {
    /// Number of subquantizers (vector is split into this many parts)
    pub n_subquantizers: usize,
    /// Number of centroids per subquantizer (typically 256 for 8-bit codes)
    pub n_centroids: usize,
    /// Number of bits per subquantizer (determines n_centroids: 2^n_bits)
    pub n_bits: usize,
    /// Number of iterations for k-means training
    pub max_iterations: usize,
    /// Convergence threshold for k-means
    pub convergence_threshold: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Enable residual quantization for better accuracy
    pub enable_residual_quantization: bool,
    /// Number of residual quantization levels
    pub residual_levels: usize,
    /// Enable multi-codebook quantization
    pub enable_multi_codebook: bool,
    /// Number of codebooks for multi-codebook quantization
    pub num_codebooks: usize,
    /// Enable symmetric distance computation
    pub enable_symmetric_distance: bool,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            n_subquantizers: 8,
            n_centroids: 256,
            n_bits: 8, // 2^8 = 256 centroids
            max_iterations: 50,
            convergence_threshold: 1e-4,
            seed: None,
            enable_residual_quantization: false,
            residual_levels: 2,
            enable_multi_codebook: false,
            num_codebooks: 2,
            enable_symmetric_distance: false,
        }
    }
}

impl PQConfig {
    /// Create a new PQConfig with specified bits per subquantizer
    pub fn with_bits(n_subquantizers: usize, n_bits: usize) -> Self {
        Self {
            n_subquantizers,
            n_centroids: 1 << n_bits, // 2^n_bits
            n_bits,
            max_iterations: 50,
            convergence_threshold: 1e-4,
            seed: None,
            enable_residual_quantization: false,
            residual_levels: 2,
            enable_multi_codebook: false,
            num_codebooks: 2,
            enable_symmetric_distance: false,
        }
    }

    /// Create a configuration with residual quantization enabled
    pub fn with_residual_quantization(
        n_subquantizers: usize,
        n_bits: usize,
        residual_levels: usize,
    ) -> Self {
        Self {
            n_subquantizers,
            n_centroids: 1 << n_bits,
            n_bits,
            enable_residual_quantization: true,
            residual_levels,
            ..Default::default()
        }
    }

    /// Create a configuration with multi-codebook quantization enabled
    pub fn with_multi_codebook(
        n_subquantizers: usize,
        n_bits: usize,
        num_codebooks: usize,
    ) -> Self {
        Self {
            n_subquantizers,
            n_centroids: 1 << n_bits,
            n_bits,
            enable_multi_codebook: true,
            num_codebooks,
            ..Default::default()
        }
    }

    /// Create a configuration with all enhancements enabled
    pub fn enhanced(n_subquantizers: usize, n_bits: usize) -> Self {
        Self {
            n_subquantizers,
            n_centroids: 1 << n_bits,
            n_bits,
            enable_residual_quantization: true,
            residual_levels: 2,
            enable_multi_codebook: true,
            num_codebooks: 2,
            enable_symmetric_distance: true,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.n_centroids != (1 << self.n_bits) {
            return Err(anyhow!(
                "n_centroids {} doesn't match 2^n_bits ({})",
                self.n_centroids,
                1 << self.n_bits
            ));
        }
        if self.n_subquantizers == 0 {
            return Err(anyhow!("n_subquantizers must be greater than 0"));
        }
        if self.n_bits == 0 || self.n_bits > 16 {
            return Err(anyhow!("n_bits must be between 1 and 16"));
        }
        if self.enable_residual_quantization && self.residual_levels == 0 {
            return Err(anyhow!(
                "residual_levels must be greater than 0 when residual quantization is enabled"
            ));
        }
        if self.enable_multi_codebook && self.num_codebooks < 2 {
            return Err(anyhow!(
                "num_codebooks must be at least 2 when multi-codebook quantization is enabled"
            ));
        }
        Ok(())
    }
}

/// A single subquantizer that handles a portion of the vector dimensions
#[derive(Debug, Clone)]
struct SubQuantizer {
    /// Start dimension (inclusive)
    start_dim: usize,
    /// End dimension (exclusive)
    end_dim: usize,
    /// Centroids for this subquantizer
    centroids: Vec<Vec<f32>>,
}

impl SubQuantizer {
    fn new(start_dim: usize, end_dim: usize, n_centroids: usize) -> Self {
        Self {
            start_dim,
            end_dim,
            centroids: Vec::with_capacity(n_centroids),
        }
    }

    /// Extract subvector from full vector
    fn extract_subvector(&self, vector: &[f32]) -> Vec<f32> {
        vector[self.start_dim..self.end_dim].to_vec()
    }

    /// Train this subquantizer on subvectors
    fn train(&mut self, subvectors: &[Vec<f32>], config: &PQConfig) -> Result<()> {
        if subvectors.is_empty() {
            return Err(anyhow!("Cannot train subquantizer with empty data"));
        }

        let dims = subvectors[0].len();

        // Initialize centroids using k-means++
        self.centroids = self.initialize_centroids_kmeans_plus_plus(subvectors, config)?;

        // Run k-means
        let mut iteration = 0;
        let mut prev_error = f32::INFINITY;

        while iteration < config.max_iterations {
            // Assign points to nearest centroids
            let mut clusters: Vec<Vec<&Vec<f32>>> = vec![Vec::new(); config.n_centroids];

            for subvector in subvectors {
                let nearest_idx = self.find_nearest_centroid(subvector)?;
                clusters[nearest_idx].push(subvector);
            }

            // Update centroids
            let mut total_error = 0.0;
            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let new_centroid = self.compute_centroid(cluster, dims);
                    total_error += self.euclidean_distance(&self.centroids[i], &new_centroid);
                    self.centroids[i] = new_centroid;
                }
            }

            // Check convergence
            if (prev_error - total_error).abs() < config.convergence_threshold {
                break;
            }

            prev_error = total_error;
            iteration += 1;
        }

        Ok(())
    }

    /// Initialize centroids using k-means++
    fn initialize_centroids_kmeans_plus_plus(
        &self,
        subvectors: &[Vec<f32>],
        config: &PQConfig,
    ) -> Result<Vec<Vec<f32>>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        config.seed.unwrap_or(42).hash(&mut hasher);
        let mut rng_state = hasher.finish();

        let mut centroids = Vec::with_capacity(config.n_centroids);

        // Choose first centroid randomly
        let first_idx = (rng_state as usize) % subvectors.len();
        centroids.push(subvectors[first_idx].clone());

        // Choose remaining centroids
        while centroids.len() < config.n_centroids {
            let mut distances = Vec::with_capacity(subvectors.len());
            let mut sum_distances = 0.0;

            // Calculate distance to nearest centroid for each point
            for subvector in subvectors {
                let min_dist = centroids
                    .iter()
                    .map(|c| self.euclidean_distance(subvector, c))
                    .fold(f32::INFINITY, |a, b| a.min(b));

                distances.push(min_dist * min_dist);
                sum_distances += min_dist * min_dist;
            }

            // Choose next centroid
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let threshold = (rng_state as f32 / u64::MAX as f32) * sum_distances;

            let mut cumulative = 0.0;
            for (i, &dist) in distances.iter().enumerate() {
                cumulative += dist;
                if cumulative >= threshold {
                    centroids.push(subvectors[i].clone());
                    break;
                }
            }
        }

        Ok(centroids)
    }

    /// Compute centroid of a cluster
    fn compute_centroid(&self, cluster: &[&Vec<f32>], dims: usize) -> Vec<f32> {
        if cluster.is_empty() {
            return vec![0.0; dims];
        }

        let mut sum = vec![0.0; dims];
        for vector in cluster {
            for (i, &val) in vector.iter().enumerate() {
                sum[i] += val;
            }
        }

        let count = cluster.len() as f32;
        for val in &mut sum {
            *val /= count;
        }

        sum
    }

    /// Find nearest centroid for a subvector
    fn find_nearest_centroid(&self, subvector: &[f32]) -> Result<usize> {
        if self.centroids.is_empty() {
            return Err(anyhow!("No centroids available"));
        }

        let mut min_distance = f32::INFINITY;
        let mut nearest_idx = 0;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let distance = self.euclidean_distance(subvector, centroid);
            if distance < min_distance {
                min_distance = distance;
                nearest_idx = i;
            }
        }

        Ok(nearest_idx)
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Encode a subvector to its nearest centroid index
    fn encode(&self, subvector: &[f32]) -> Result<u8> {
        if self.centroids.len() > 256 {
            return Err(anyhow!("Too many centroids for u8 encoding"));
        }

        let idx = self.find_nearest_centroid(subvector)?;
        Ok(idx as u8)
    }

    /// Decode a centroid index back to a subvector
    fn decode(&self, code: u8) -> Result<Vec<f32>> {
        let idx = code as usize;
        if idx >= self.centroids.len() {
            return Err(anyhow!("Invalid code: {}", code));
        }
        Ok(self.centroids[idx].clone())
    }
}

/// Enhanced codes structure for advanced PQ features
#[derive(Debug, Clone)]
pub struct EnhancedCodes {
    /// Primary quantization codes
    pub primary: Vec<u8>,
    /// Residual quantization codes (one per level)
    pub residual: Vec<Vec<u8>>,
    /// Multi-codebook quantization codes (one per codebook)
    pub multi_codebook: Vec<Vec<u8>>,
}

/// Enhanced Product Quantization index with residual and multi-codebook support
#[derive(Debug, Clone)]
pub struct PQIndex {
    config: PQConfig,
    /// Primary subquantizers
    subquantizers: Vec<SubQuantizer>,
    /// Residual quantizers (for each level)
    residual_quantizers: Vec<Vec<SubQuantizer>>,
    /// Multi-codebook quantizers
    multi_codebook_quantizers: Vec<Vec<SubQuantizer>>,
    /// Encoded vectors (primary codes)
    codes: Vec<(String, Vec<u8>)>,
    /// Residual codes (for each level)
    residual_codes: Vec<Vec<(String, Vec<u8>)>>,
    /// Multi-codebook codes
    multi_codebook_codes: Vec<Vec<(String, Vec<u8>)>>,
    /// Distance lookup tables for symmetric distance computation
    distance_tables: Option<Vec<Vec<Vec<f32>>>>,
    /// URI to index mapping
    uri_to_id: HashMap<String, usize>,
    /// Vector dimensions
    dimensions: Option<usize>,
    /// Whether the index has been trained
    is_trained: bool,
}

impl PQIndex {
    /// Create a new PQ index
    pub fn new(config: PQConfig) -> Self {
        Self {
            residual_quantizers: vec![Vec::new(); config.residual_levels],
            multi_codebook_quantizers: vec![Vec::new(); config.num_codebooks],
            residual_codes: vec![Vec::new(); config.residual_levels],
            multi_codebook_codes: vec![Vec::new(); config.num_codebooks],
            distance_tables: None,
            config,
            subquantizers: Vec::new(),
            codes: Vec::new(),
            uri_to_id: HashMap::new(),
            dimensions: None,
            is_trained: false,
        }
    }

    /// Train the PQ index with training vectors
    pub fn train(&mut self, training_vectors: &[Vector]) -> Result<()> {
        if training_vectors.is_empty() {
            return Err(anyhow!("Cannot train PQ with empty training set"));
        }

        // Validate dimensions
        let dims = training_vectors[0].dimensions;
        if !training_vectors.iter().all(|v| v.dimensions == dims) {
            return Err(anyhow!(
                "All training vectors must have the same dimensions"
            ));
        }

        if dims % self.config.n_subquantizers != 0 {
            return Err(anyhow!(
                "Vector dimensions {} must be divisible by n_subquantizers {}",
                dims,
                self.config.n_subquantizers
            ));
        }

        self.dimensions = Some(dims);
        let subdim = dims / self.config.n_subquantizers;

        // Initialize subquantizers
        self.subquantizers.clear();
        for i in 0..self.config.n_subquantizers {
            let start = i * subdim;
            let end = start + subdim;
            self.subquantizers
                .push(SubQuantizer::new(start, end, self.config.n_centroids));
        }

        // Extract training data as f32
        let training_data: Vec<Vec<f32>> = training_vectors.iter().map(|v| v.as_f32()).collect();

        // Train each subquantizer
        for sq in self.subquantizers.iter_mut() {
            // Extract subvectors for this subquantizer
            let subvectors: Vec<Vec<f32>> = training_data
                .iter()
                .map(|v| sq.extract_subvector(v))
                .collect();

            sq.train(&subvectors, &self.config)?;
        }

        // Train residual quantizers if enabled
        if self.config.enable_residual_quantization {
            self.train_residual_quantizers(&training_data)?;
        }

        // Train multi-codebook quantizers if enabled
        if self.config.enable_multi_codebook {
            self.train_multi_codebook_quantizers(&training_data)?;
        }

        // Build distance tables for symmetric distance computation if enabled
        if self.config.enable_symmetric_distance {
            self.build_distance_tables()?;
        }

        self.is_trained = true;
        Ok(())
    }

    /// Train residual quantizers for improved accuracy
    fn train_residual_quantizers(&mut self, training_data: &[Vec<f32>]) -> Result<()> {
        let subdim = self.dimensions.unwrap() / self.config.n_subquantizers;

        // Start with residuals from the primary quantizers
        let mut current_residuals = training_data.to_vec();

        for level in 0..self.config.residual_levels {
            // Compute residuals from previous level
            if level == 0 {
                // Compute residuals from primary quantizers
                for (i, vector) in training_data.iter().enumerate() {
                    let primary_codes = self.encode_primary_vector(vector)?;
                    let reconstructed = self.decode_primary_codes(&primary_codes)?;

                    // Compute residual
                    let residual: Vec<f32> = vector
                        .iter()
                        .zip(reconstructed.iter())
                        .map(|(a, b)| a - b)
                        .collect();
                    current_residuals[i] = residual;
                }
            } else {
                // Compute residuals from previous residual level
                for (i, residual) in current_residuals.clone().iter().enumerate() {
                    let residual_codes = self.encode_residual_vector(residual, level - 1)?;
                    let reconstructed_residual =
                        self.decode_residual_codes(&residual_codes, level - 1)?;

                    let new_residual: Vec<f32> = residual
                        .iter()
                        .zip(reconstructed_residual.iter())
                        .map(|(a, b)| a - b)
                        .collect();
                    current_residuals[i] = new_residual;
                }
            }

            // Initialize residual subquantizers for this level
            self.residual_quantizers[level].clear();
            for i in 0..self.config.n_subquantizers {
                let start = i * subdim;
                let end = start + subdim;
                self.residual_quantizers[level].push(SubQuantizer::new(
                    start,
                    end,
                    self.config.n_centroids,
                ));
            }

            // Train each residual subquantizer
            for sq in self.residual_quantizers[level].iter_mut() {
                let subvectors: Vec<Vec<f32>> = current_residuals
                    .iter()
                    .map(|v| sq.extract_subvector(v))
                    .collect();

                sq.train(&subvectors, &self.config)?;
            }
        }

        Ok(())
    }

    /// Train multi-codebook quantizers for better coverage
    fn train_multi_codebook_quantizers(&mut self, training_data: &[Vec<f32>]) -> Result<()> {
        let subdim = self.dimensions.unwrap() / self.config.n_subquantizers;

        for codebook_idx in 0..self.config.num_codebooks {
            // Initialize subquantizers for this codebook
            self.multi_codebook_quantizers[codebook_idx].clear();
            for i in 0..self.config.n_subquantizers {
                let start = i * subdim;
                let end = start + subdim;
                self.multi_codebook_quantizers[codebook_idx].push(SubQuantizer::new(
                    start,
                    end,
                    self.config.n_centroids,
                ));
            }

            // Use different initialization for each codebook
            let mut modified_config = self.config.clone();
            modified_config.seed = self.config.seed.map(|s| s + codebook_idx as u64);

            // Train each subquantizer in this codebook
            for sq in self.multi_codebook_quantizers[codebook_idx].iter_mut() {
                let subvectors: Vec<Vec<f32>> = training_data
                    .iter()
                    .map(|v| sq.extract_subvector(v))
                    .collect();

                sq.train(&subvectors, &modified_config)?;
            }
        }

        Ok(())
    }

    /// Build distance lookup tables for symmetric distance computation
    fn build_distance_tables(&mut self) -> Result<()> {
        let mut tables = Vec::new();

        for sq_idx in 0..self.config.n_subquantizers {
            let sq = &self.subquantizers[sq_idx];
            let mut sq_table = Vec::new();

            // Build distance table between all pairs of centroids
            for i in 0..sq.centroids.len() {
                let mut centroid_distances = Vec::new();
                for j in 0..sq.centroids.len() {
                    let distance = sq.euclidean_distance(&sq.centroids[i], &sq.centroids[j]);
                    centroid_distances.push(distance);
                }
                sq_table.push(centroid_distances);
            }
            tables.push(sq_table);
        }

        self.distance_tables = Some(tables);
        Ok(())
    }

    /// Helper method to encode with primary quantizers only
    fn encode_primary_vector(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let mut codes = Vec::with_capacity(self.subquantizers.len());

        for sq in &self.subquantizers {
            let subvec = sq.extract_subvector(vector);
            let code = sq.encode(&subvec)?;
            codes.push(code);
        }

        Ok(codes)
    }

    /// Helper method to decode primary codes
    fn decode_primary_codes(&self, codes: &[u8]) -> Result<Vec<f32>> {
        let mut reconstructed = Vec::new();

        for (sq, &code) in self.subquantizers.iter().zip(codes.iter()) {
            let subvec = sq.decode(code)?;
            reconstructed.extend(subvec);
        }

        Ok(reconstructed)
    }

    /// Helper method to encode with residual quantizers
    fn encode_residual_vector(&self, vector: &[f32], level: usize) -> Result<Vec<u8>> {
        if level >= self.residual_quantizers.len() {
            return Err(anyhow!("Invalid residual level: {}", level));
        }

        let mut codes = Vec::with_capacity(self.residual_quantizers[level].len());

        for sq in &self.residual_quantizers[level] {
            let subvec = sq.extract_subvector(vector);
            let code = sq.encode(&subvec)?;
            codes.push(code);
        }

        Ok(codes)
    }

    /// Helper method to decode residual codes
    fn decode_residual_codes(&self, codes: &[u8], level: usize) -> Result<Vec<f32>> {
        if level >= self.residual_quantizers.len() {
            return Err(anyhow!("Invalid residual level: {}", level));
        }

        let mut reconstructed = Vec::new();

        for (sq, &code) in self.residual_quantizers[level].iter().zip(codes.iter()) {
            let subvec = sq.decode(code)?;
            reconstructed.extend(subvec);
        }

        Ok(reconstructed)
    }

    /// Encode a vector into PQ codes
    fn encode_vector(&self, vector: &Vector) -> Result<Vec<u8>> {
        if !self.is_trained {
            return Err(anyhow!("PQ index must be trained before encoding"));
        }

        let vector_f32 = vector.as_f32();
        let mut codes = Vec::with_capacity(self.subquantizers.len());

        for sq in &self.subquantizers {
            let subvec = sq.extract_subvector(&vector_f32);
            let code = sq.encode(&subvec)?;
            codes.push(code);
        }

        Ok(codes)
    }

    /// Decode PQ codes back to an approximate vector
    fn decode_codes(&self, codes: &[u8]) -> Result<Vector> {
        if codes.len() != self.subquantizers.len() {
            return Err(anyhow!("Invalid code length"));
        }

        let mut reconstructed = Vec::new();

        for (sq, &code) in self.subquantizers.iter().zip(codes.iter()) {
            let subvec = sq.decode(code)?;
            reconstructed.extend(subvec);
        }

        Ok(Vector::new(reconstructed))
    }

    /// Public method to encode a vector (for OPQ)
    pub fn encode(&self, vector: &Vector) -> Result<Vec<u8>> {
        self.encode_vector(vector)
    }

    /// Public method to decode codes (for OPQ)
    pub fn decode(&self, codes: &[u8]) -> Result<Vector> {
        self.decode_codes(codes)
    }

    /// Reconstruct a vector by encoding and then decoding (for OPQ)
    pub fn reconstruct(&self, vector: &Vector) -> Result<Vector> {
        let codes = self.encode_vector(vector)?;
        self.decode_codes(&codes)
    }

    /// Compute asymmetric distance between a query vector and PQ codes
    fn asymmetric_distance(&self, query: &Vector, codes: &[u8]) -> Result<f32> {
        let query_f32 = query.as_f32();
        let mut total_distance = 0.0;

        for (sq, &code) in self.subquantizers.iter().zip(codes.iter()) {
            let query_subvec = sq.extract_subvector(&query_f32);
            let centroid = &sq.centroids[code as usize];

            // Compute squared distance to avoid sqrt
            let dist: f32 = query_subvec
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();

            total_distance += dist;
        }

        Ok(total_distance.sqrt())
    }

    /// Enhanced encoding with residual and multi-codebook support
    fn encode_vector_enhanced(&self, vector: &Vector) -> Result<EnhancedCodes> {
        if !self.is_trained {
            return Err(anyhow!("PQ index must be trained before encoding"));
        }

        let vector_f32 = vector.as_f32();

        // Primary encoding
        let primary_codes = self.encode_primary_vector(&vector_f32)?;

        // Residual encoding if enabled
        let mut residual_codes = Vec::new();
        if self.config.enable_residual_quantization {
            let mut current_residual = vector_f32.clone();

            // Compute residual from primary quantization
            let primary_reconstructed = self.decode_primary_codes(&primary_codes)?;
            current_residual = current_residual
                .iter()
                .zip(primary_reconstructed.iter())
                .map(|(a, b)| a - b)
                .collect();

            // Encode residuals at each level
            for level in 0..self.config.residual_levels {
                let level_codes = self.encode_residual_vector(&current_residual, level)?;
                residual_codes.push(level_codes.clone());

                // Update residual for next level
                if level < self.config.residual_levels - 1 {
                    let level_reconstructed = self.decode_residual_codes(&level_codes, level)?;
                    current_residual = current_residual
                        .iter()
                        .zip(level_reconstructed.iter())
                        .map(|(a, b)| a - b)
                        .collect();
                }
            }
        }

        // Multi-codebook encoding if enabled
        let mut multi_codebook_codes = Vec::new();
        if self.config.enable_multi_codebook {
            for codebook_idx in 0..self.config.num_codebooks {
                let mut codes =
                    Vec::with_capacity(self.multi_codebook_quantizers[codebook_idx].len());

                for sq in &self.multi_codebook_quantizers[codebook_idx] {
                    let subvec = sq.extract_subvector(&vector_f32);
                    let code = sq.encode(&subvec)?;
                    codes.push(code);
                }
                multi_codebook_codes.push(codes);
            }
        }

        Ok(EnhancedCodes {
            primary: primary_codes,
            residual: residual_codes,
            multi_codebook: multi_codebook_codes,
        })
    }

    /// Symmetric distance computation between two sets of codes
    fn symmetric_distance(&self, codes1: &[u8], codes2: &[u8]) -> Result<f32> {
        if !self.config.enable_symmetric_distance {
            return Err(anyhow!("Symmetric distance computation not enabled"));
        }

        let distance_tables = self
            .distance_tables
            .as_ref()
            .ok_or_else(|| anyhow!("Distance tables not built"))?;

        if codes1.len() != codes2.len() || codes1.len() != self.config.n_subquantizers {
            return Err(anyhow!("Invalid code lengths for symmetric distance"));
        }

        let mut total_distance = 0.0;

        for (sq_idx, (&code1, &code2)) in codes1.iter().zip(codes2.iter()).enumerate() {
            let distance = distance_tables[sq_idx][code1 as usize][code2 as usize];
            total_distance += distance * distance; // Squared distance
        }

        Ok(total_distance.sqrt())
    }

    /// Enhanced distance computation with residual and multi-codebook support
    fn enhanced_distance(&self, query: &Vector, enhanced_codes: &EnhancedCodes) -> Result<f32> {
        // Start with primary distance
        let mut total_distance = self.asymmetric_distance(query, &enhanced_codes.primary)?;

        // Add residual distances if enabled
        if self.config.enable_residual_quantization && !enhanced_codes.residual.is_empty() {
            let query_f32 = query.as_f32();
            let mut current_residual = query_f32.clone();

            // Compute residual from primary quantization
            let primary_reconstructed = self.decode_primary_codes(&enhanced_codes.primary)?;
            current_residual = current_residual
                .iter()
                .zip(primary_reconstructed.iter())
                .map(|(a, b)| a - b)
                .collect();

            // Add distance from each residual level
            for (level, residual_codes) in enhanced_codes.residual.iter().enumerate() {
                let mut residual_distance = 0.0;

                for (sq, &code) in self.residual_quantizers[level]
                    .iter()
                    .zip(residual_codes.iter())
                {
                    let query_subvec = sq.extract_subvector(&current_residual);
                    let centroid = &sq.centroids[code as usize];

                    let dist: f32 = query_subvec
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();

                    residual_distance += dist;
                }

                total_distance += residual_distance.sqrt() * 0.5; // Weight residual distances

                // Update residual for next level
                if level < enhanced_codes.residual.len() - 1 {
                    let level_reconstructed = self.decode_residual_codes(residual_codes, level)?;
                    current_residual = current_residual
                        .iter()
                        .zip(level_reconstructed.iter())
                        .map(|(a, b)| a - b)
                        .collect();
                }
            }
        }

        // For multi-codebook, use the minimum distance across codebooks
        if self.config.enable_multi_codebook && !enhanced_codes.multi_codebook.is_empty() {
            let mut min_codebook_distance = f32::INFINITY;

            for codes in &enhanced_codes.multi_codebook {
                let codebook_distance = self.asymmetric_distance(query, codes)?;
                min_codebook_distance = min_codebook_distance.min(codebook_distance);
            }

            // Use the minimum as a refinement
            total_distance = total_distance.min(min_codebook_distance);
        }

        Ok(total_distance)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if let Some(dims) = self.dimensions {
            // Original: dims * 4 bytes (f32)
            // Compressed: n_subquantizers bytes
            (dims as f32 * 4.0) / (self.config.n_subquantizers as f32)
        } else {
            0.0
        }
    }

    /// Get index statistics
    pub fn stats(&self) -> PQStats {
        PQStats {
            n_vectors: self.codes.len(),
            n_subquantizers: self.config.n_subquantizers,
            n_centroids: self.config.n_centroids,
            is_trained: self.is_trained,
            dimensions: self.dimensions,
            compression_ratio: self.compression_ratio(),
            memory_usage_bytes: self.estimate_memory_usage(),
        }
    }

    /// Estimate memory usage in bytes
    fn estimate_memory_usage(&self) -> usize {
        let codebook_size = self
            .subquantizers
            .iter()
            .map(|sq| sq.centroids.len() * (sq.end_dim - sq.start_dim) * 4)
            .sum::<usize>();

        let codes_size = self.codes.len() * self.config.n_subquantizers;

        codebook_size + codes_size
    }

    /// Check if the index is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Compute distance between query and encoded vector (for IVF compatibility)
    pub fn compute_distance(&self, query: &Vector, codes: &[u8]) -> Result<f32> {
        self.asymmetric_distance(query, codes)
    }

    /// Decode codes to vector (for IVF compatibility)
    pub fn decode_vector(&self, codes: &[u8]) -> Result<Vector> {
        self.decode_codes(codes)
    }
}

impl VectorIndex for PQIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        if !self.is_trained {
            return Err(anyhow!("PQ index must be trained before inserting vectors"));
        }

        // Validate dimensions
        if let Some(dims) = self.dimensions {
            if vector.dimensions != dims {
                return Err(anyhow!(
                    "Vector dimensions {} don't match index dimensions {}",
                    vector.dimensions,
                    dims
                ));
            }
        }

        // Encode the vector
        let codes = self.encode_vector(&vector)?;

        // Store the codes
        let id = self.codes.len();
        self.uri_to_id.insert(uri.clone(), id);
        self.codes.push((uri, codes));

        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        if !self.is_trained {
            return Err(anyhow!("PQ index must be trained before searching"));
        }

        // Compute distances to all vectors
        let mut distances: Vec<(String, f32)> = self
            .codes
            .iter()
            .map(|(uri, codes)| {
                let dist = self
                    .asymmetric_distance(query, codes)
                    .unwrap_or(f32::INFINITY);
                (uri.clone(), dist)
            })
            .collect();

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        // Convert distances to similarities
        Ok(distances
            .into_iter()
            .map(|(uri, dist)| (uri, 1.0 / (1.0 + dist)))
            .collect())
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        if !self.is_trained {
            return Err(anyhow!("PQ index must be trained before searching"));
        }

        let mut results = Vec::new();

        for (uri, codes) in &self.codes {
            let dist = self.asymmetric_distance(query, codes)?;
            let similarity = 1.0 / (1.0 + dist);

            if similarity >= threshold {
                results.push((uri.clone(), similarity));
            }
        }

        // Sort by similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(results)
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        // PQ doesn't store original vectors, only codes
        // Would need to decode, but that returns an approximation
        None
    }
}

impl PQIndex {
    /// Public search method for use by OPQ and other modules
    pub fn search(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        self.search_knn(query, k)
    }
}

/// Statistics for PQ index
#[derive(Debug, Clone)]
pub struct PQStats {
    pub n_vectors: usize,
    pub n_subquantizers: usize,
    pub n_centroids: usize,
    pub is_trained: bool,
    pub dimensions: Option<usize>,
    pub compression_ratio: f32,
    pub memory_usage_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_basic() {
        let config = PQConfig {
            n_subquantizers: 2,
            n_centroids: 4,
            ..Default::default()
        };

        let mut index = PQIndex::new(config);

        // Create training vectors
        let training_vectors = vec![
            Vector::new(vec![1.0, 0.0, 0.0, 1.0]),
            Vector::new(vec![0.0, 1.0, 1.0, 0.0]),
            Vector::new(vec![-1.0, 0.0, 0.0, -1.0]),
            Vector::new(vec![0.0, -1.0, -1.0, 0.0]),
            Vector::new(vec![0.5, 0.5, 0.5, 0.5]),
            Vector::new(vec![-0.5, -0.5, -0.5, -0.5]),
        ];

        // Train the index
        index.train(&training_vectors).unwrap();
        assert!(index.is_trained);

        // Insert vectors
        for (i, vec) in training_vectors.iter().enumerate() {
            index.insert(format!("vec{}", i), vec.clone()).unwrap();
        }

        // Search for nearest neighbors
        let query = Vector::new(vec![0.9, 0.1, 0.1, 0.9]);
        let results = index.search_knn(&query, 3).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_pq_compression() {
        let config = PQConfig {
            n_subquantizers: 4,
            n_centroids: 16,
            ..Default::default()
        };

        let mut index = PQIndex::new(config);

        // Create 128-dimensional vectors
        let dims = 128;
        let training_vectors: Vec<Vector> = (0..100)
            .map(|i| {
                let values: Vec<f32> = (0..dims).map(|j| ((i + j) as f32).sin()).collect();
                Vector::new(values)
            })
            .collect();

        // Train and check compression ratio
        index.train(&training_vectors).unwrap();

        let compression_ratio = index.compression_ratio();
        assert_eq!(compression_ratio, 128.0); // 128*4 bytes -> 4 bytes

        let stats = index.stats();
        assert_eq!(stats.n_subquantizers, 4);
        assert_eq!(stats.n_centroids, 16);
        assert_eq!(stats.dimensions, Some(128));
    }

    #[test]
    fn test_pq_reconstruction() {
        let config = PQConfig {
            n_subquantizers: 2,
            n_centroids: 8,
            ..Default::default()
        };

        let mut index = PQIndex::new(config);

        // Simple training set
        let training_vectors = vec![
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![-1.0, 0.0]),
            Vector::new(vec![0.0, -1.0]),
        ];

        index.train(&training_vectors).unwrap();

        // Encode and decode a vector
        let original = Vector::new(vec![0.7, 0.7]);
        let codes = index.encode_vector(&original).unwrap();
        let reconstructed = index.decode_codes(&codes).unwrap();

        // Check that reconstruction is reasonable (not exact due to quantization)
        let dist = original.euclidean_distance(&reconstructed).unwrap();
        assert!(dist < 1.0); // Should be reasonably close
    }

    #[test]
    fn test_pq_residual_quantization() {
        let config = PQConfig::with_residual_quantization(2, 3, 2); // 2 subquantizers, 3 bits, 2 residual levels
        let mut index = PQIndex::new(config);

        // Create training vectors
        let training_vectors = vec![
            Vector::new(vec![1.0, 0.0, 0.0, 1.0]),
            Vector::new(vec![0.0, 1.0, 1.0, 0.0]),
            Vector::new(vec![-1.0, 0.0, 0.0, -1.0]),
            Vector::new(vec![0.0, -1.0, -1.0, 0.0]),
            Vector::new(vec![0.5, 0.5, 0.5, 0.5]),
            Vector::new(vec![-0.5, -0.5, -0.5, -0.5]),
        ];

        // Train the index with residual quantization
        index.train(&training_vectors).unwrap();
        assert!(index.is_trained());
        assert_eq!(index.residual_quantizers.len(), 2);

        // Test enhanced encoding
        let test_vector = Vector::new(vec![0.7, 0.3, 0.3, 0.7]);
        let enhanced_codes = index.encode_vector_enhanced(&test_vector).unwrap();

        assert!(!enhanced_codes.primary.is_empty());
        assert_eq!(enhanced_codes.residual.len(), 2);
        assert!(enhanced_codes.multi_codebook.is_empty()); // Multi-codebook not enabled
    }

    #[test]
    fn test_pq_multi_codebook() {
        let config = PQConfig::with_multi_codebook(2, 3, 3); // 2 subquantizers, 3 bits, 3 codebooks
        let mut index = PQIndex::new(config);

        // Create training vectors
        let training_vectors = vec![
            Vector::new(vec![1.0, 0.0, 0.0, 1.0]),
            Vector::new(vec![0.0, 1.0, 1.0, 0.0]),
            Vector::new(vec![-1.0, 0.0, 0.0, -1.0]),
            Vector::new(vec![0.0, -1.0, -1.0, 0.0]),
            Vector::new(vec![0.5, 0.5, 0.5, 0.5]),
            Vector::new(vec![-0.5, -0.5, -0.5, -0.5]),
        ];

        // Train the index with multi-codebook quantization
        index.train(&training_vectors).unwrap();
        assert!(index.is_trained());
        assert_eq!(index.multi_codebook_quantizers.len(), 3);

        // Test enhanced encoding
        let test_vector = Vector::new(vec![0.7, 0.3, 0.3, 0.7]);
        let enhanced_codes = index.encode_vector_enhanced(&test_vector).unwrap();

        assert!(!enhanced_codes.primary.is_empty());
        assert!(enhanced_codes.residual.is_empty()); // Residual not enabled
        assert_eq!(enhanced_codes.multi_codebook.len(), 3);
    }

    #[test]
    fn test_pq_symmetric_distance() {
        let mut config = PQConfig::default();
        config.enable_symmetric_distance = true;
        config.n_subquantizers = 2;
        config.n_centroids = 4;

        let mut index = PQIndex::new(config);

        // Create training vectors
        let training_vectors = vec![
            Vector::new(vec![1.0, 0.0, 0.0, 1.0]),
            Vector::new(vec![0.0, 1.0, 1.0, 0.0]),
            Vector::new(vec![-1.0, 0.0, 0.0, -1.0]),
            Vector::new(vec![0.0, -1.0, -1.0, 0.0]),
        ];

        // Train the index
        index.train(&training_vectors).unwrap();
        assert!(index.distance_tables.is_some());

        // Test symmetric distance computation
        let codes1 = vec![0, 1];
        let codes2 = vec![1, 0];
        let distance = index.symmetric_distance(&codes1, &codes2).unwrap();

        assert!(distance >= 0.0);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_pq_enhanced_features() {
        let config = PQConfig::enhanced(2, 3); // All features enabled
        let mut index = PQIndex::new(config);

        // Create training vectors
        let training_vectors = vec![
            Vector::new(vec![1.0, 0.0, 0.0, 1.0]),
            Vector::new(vec![0.0, 1.0, 1.0, 0.0]),
            Vector::new(vec![-1.0, 0.0, 0.0, -1.0]),
            Vector::new(vec![0.0, -1.0, -1.0, 0.0]),
            Vector::new(vec![0.5, 0.5, 0.5, 0.5]),
            Vector::new(vec![-0.5, -0.5, -0.5, -0.5]),
        ];

        // Train with all enhanced features
        index.train(&training_vectors).unwrap();
        assert!(index.is_trained());

        // Verify all features are initialized
        assert!(!index.residual_quantizers.is_empty());
        assert!(!index.multi_codebook_quantizers.is_empty());
        assert!(index.distance_tables.is_some());

        // Test enhanced encoding and distance computation
        let test_vector = Vector::new(vec![0.7, 0.3, 0.3, 0.7]);
        let enhanced_codes = index.encode_vector_enhanced(&test_vector).unwrap();
        let enhanced_distance = index
            .enhanced_distance(&test_vector, &enhanced_codes)
            .unwrap();

        assert!(enhanced_distance >= 0.0);
        assert!(enhanced_distance.is_finite());

        // Enhanced distance should be more accurate (smaller) than basic asymmetric distance
        let basic_distance = index
            .asymmetric_distance(&test_vector, &enhanced_codes.primary)
            .unwrap();
        assert!(enhanced_distance <= basic_distance * 1.1); // Allow some tolerance
    }

    #[test]
    fn test_pq_config_validation() {
        // Test valid enhanced config
        let config = PQConfig::enhanced(4, 8);
        assert!(config.validate().is_ok());

        // Test invalid residual config
        let mut invalid_config = PQConfig::default();
        invalid_config.enable_residual_quantization = true;
        invalid_config.residual_levels = 0;
        assert!(invalid_config.validate().is_err());

        // Test invalid multi-codebook config
        let mut invalid_config = PQConfig::default();
        invalid_config.enable_multi_codebook = true;
        invalid_config.num_codebooks = 1;
        assert!(invalid_config.validate().is_err());
    }
}
