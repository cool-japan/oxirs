//! Inverted File (IVF) index implementation for approximate nearest neighbor search
//!
//! IVF is a clustering-based indexing method that partitions the vector space into
//! Voronoi cells. Each cell has a centroid, and vectors are assigned to their nearest
//! centroid. During search, only a subset of cells are examined, greatly reducing
//! search time at the cost of some accuracy.

use crate::{Vector, VectorIndex, VectorPrecision, pq::{PQConfig, PQIndex}};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Quantization strategy for residuals
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationStrategy {
    /// No quantization - store full residuals
    None,
    /// Single-level product quantization
    ProductQuantization(PQConfig),
    /// Residual quantization with multiple levels
    ResidualQuantization {
        levels: usize,
        pq_configs: Vec<PQConfig>,
    },
    /// Multi-codebook quantization for improved accuracy
    MultiCodebook {
        num_codebooks: usize,
        pq_configs: Vec<PQConfig>,
    },
}

/// Configuration for IVF index
#[derive(Debug, Clone)]
pub struct IvfConfig {
    /// Number of clusters (Voronoi cells)
    pub n_clusters: usize,
    /// Number of probes during search (cells to examine)
    pub n_probes: usize,
    /// Maximum iterations for k-means clustering
    pub max_iterations: usize,
    /// Convergence threshold for k-means
    pub convergence_threshold: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Quantization strategy for residuals
    pub quantization: QuantizationStrategy,
    /// Enable residual quantization for compression (deprecated - use quantization field)
    pub enable_residual_quantization: bool,
    /// Product quantization configuration for residuals (deprecated - use quantization field)
    pub pq_config: Option<PQConfig>,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            n_clusters: 256,
            n_probes: 8,
            max_iterations: 100,
            convergence_threshold: 1e-4,
            seed: None,
            quantization: QuantizationStrategy::None,
            enable_residual_quantization: false,
            pq_config: None,
        }
    }
}

/// Storage format for vectors in inverted lists
#[derive(Debug, Clone)]
enum VectorStorage {
    /// Store full vectors
    Full(Vector),
    /// Store quantized residuals with PQ codes
    Quantized(Vec<u8>),
    /// Store multi-level residual quantization codes
    MultiLevelQuantized {
        levels: Vec<Vec<u8>>, // PQ codes for each quantization level
        final_residual: Option<Vector>, // Optional final unquantized residual
    },
    /// Store multi-codebook quantization codes
    MultiCodebook {
        codebooks: Vec<Vec<u8>>, // PQ codes from different codebooks
        weights: Vec<f32>, // Weights for combining codebook predictions
    },
}

/// Inverted list storing vectors for a single cluster
#[derive(Debug, Clone)]
struct InvertedList {
    /// Vectors in this cluster with their storage format
    vectors: Vec<(String, VectorStorage)>,
    /// Quantization strategy used for this list
    quantization: QuantizationStrategy,
    /// Product quantizer for single-level quantization
    pq_index: Option<PQIndex>,
    /// Multiple PQ indexes for multi-level residual quantization
    multi_level_pq: Vec<PQIndex>,
    /// Multiple PQ indexes for multi-codebook quantization
    multi_codebook_pq: Vec<PQIndex>,
    /// Codebook weights for multi-codebook quantization
    codebook_weights: Vec<f32>,
}

impl InvertedList {
    fn new() -> Self {
        Self {
            vectors: Vec::new(),
            quantization: QuantizationStrategy::None,
            pq_index: None,
            multi_level_pq: Vec::new(),
            multi_codebook_pq: Vec::new(),
            codebook_weights: Vec::new(),
        }
    }
    
    fn new_with_quantization(quantization: QuantizationStrategy) -> Result<Self> {
        let mut list = Self {
            vectors: Vec::new(),
            quantization: quantization.clone(),
            pq_index: None,
            multi_level_pq: Vec::new(),
            multi_codebook_pq: Vec::new(),
            codebook_weights: Vec::new(),
        };
        
        match quantization {
            QuantizationStrategy::None => {},
            QuantizationStrategy::ProductQuantization(pq_config) => {
                list.pq_index = Some(PQIndex::new(pq_config));
            },
            QuantizationStrategy::ResidualQuantization { levels: _, ref pq_configs } => {
                for pq_config in pq_configs {
                    list.multi_level_pq.push(PQIndex::new(pq_config.clone()));
                }
            },
            QuantizationStrategy::MultiCodebook { num_codebooks, ref pq_configs } => {
                for pq_config in pq_configs {
                    list.multi_codebook_pq.push(PQIndex::new(pq_config.clone()));
                }
                // Initialize equal weights for all codebooks
                list.codebook_weights = vec![1.0 / num_codebooks as f32; num_codebooks];
            },
        }
        
        Ok(list)
    }
    
    // Deprecated - use new_with_quantization instead
    fn new_with_pq(pq_config: PQConfig) -> Result<Self> {
        Self::new_with_quantization(QuantizationStrategy::ProductQuantization(pq_config))
    }

    fn add_full(&mut self, uri: String, vector: Vector) {
        self.vectors.push((uri, VectorStorage::Full(vector)));
    }
    
    fn add_residual(&mut self, uri: String, residual: Vector, centroid: &Vector) -> Result<()> {
        match &self.quantization {
            QuantizationStrategy::ProductQuantization(_) => {
                if let Some(ref mut pq_index) = self.pq_index {
                    // Train PQ on residuals if not already trained
                    if !pq_index.is_trained() {
                        let training_residuals = vec![residual.clone()];
                        pq_index.train(&training_residuals)?;
                    }
                    
                    let codes = pq_index.encode(&residual)?;
                    self.vectors.push((uri, VectorStorage::Quantized(codes)));
                } else {
                    return Err(anyhow!("PQ index not initialized for residual quantization"));
                }
            },
            QuantizationStrategy::ResidualQuantization { levels, .. } => {
                self.add_multi_level_residual(uri, residual, *levels)?;
            },
            QuantizationStrategy::MultiCodebook { .. } => {
                self.add_multi_codebook(uri, residual)?;
            },
            QuantizationStrategy::None => {
                self.add_full(uri, residual);
            },
        }
        Ok(())
    }
    
    /// Add vector using multi-level residual quantization
    fn add_multi_level_residual(&mut self, uri: String, mut residual: Vector, levels: usize) -> Result<()> {
        let mut level_codes = Vec::new();
        
        for level in 0..levels.min(self.multi_level_pq.len()) {
            // Train this level's PQ if not already trained
            if !self.multi_level_pq[level].is_trained() {
                let training_residuals = vec![residual.clone()];
                self.multi_level_pq[level].train(&training_residuals)?;
            }
            
            // Encode residual at this level
            let codes = self.multi_level_pq[level].encode(&residual)?;
            level_codes.push(codes);
            
            // Compute and subtract the quantized approximation to get next level residual
            let approximation = self.multi_level_pq[level].decode_vector(&level_codes[level])?;
            residual = residual.subtract(&approximation)?;
        }
        
        // Store the final residual if we haven't exhausted all levels
        let final_residual = if level_codes.len() < levels {
            Some(residual)
        } else {
            None
        };
        
        self.vectors.push((uri, VectorStorage::MultiLevelQuantized {
            levels: level_codes,
            final_residual,
        }));
        
        Ok(())
    }
    
    /// Add vector using multi-codebook quantization
    fn add_multi_codebook(&mut self, uri: String, residual: Vector) -> Result<()> {
        let mut codebook_codes = Vec::new();
        
        for (i, pq_index) in self.multi_codebook_pq.iter_mut().enumerate() {
            // Train this codebook's PQ if not already trained
            if !pq_index.is_trained() {
                let training_residuals = vec![residual.clone()];
                pq_index.train(&training_residuals)?;
            }
            
            // Encode residual with this codebook
            let codes = pq_index.encode(&residual)?;
            codebook_codes.push(codes);
        }
        
        self.vectors.push((uri, VectorStorage::MultiCodebook {
            codebooks: codebook_codes,
            weights: self.codebook_weights.clone(),
        }));
        
        Ok(())
    }

    fn search(&self, query: &Vector, centroid: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let mut distances: Vec<(String, f32)> = Vec::new();
        let query_residual = query.subtract(centroid)?;
        
        for (uri, storage) in &self.vectors {
            let distance = match storage {
                VectorStorage::Full(vec) => {
                    query.euclidean_distance(vec).unwrap_or(f32::INFINITY)
                }
                VectorStorage::Quantized(codes) => {
                    if let Some(ref pq_index) = self.pq_index {
                        pq_index.compute_distance(&query_residual, codes)?
                    } else {
                        f32::INFINITY
                    }
                }
                VectorStorage::MultiLevelQuantized { levels, final_residual } => {
                    self.compute_multi_level_distance(&query_residual, levels, final_residual)?
                }
                VectorStorage::MultiCodebook { codebooks, weights } => {
                    self.compute_multi_codebook_distance(&query_residual, codebooks, weights)?
                }
            };
            distances.push((uri.clone(), distance));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        
        // Convert distances to similarities (1 / (1 + distance))
        Ok(distances
            .into_iter()
            .map(|(uri, dist)| (uri, 1.0 / (1.0 + dist)))
            .collect())
    }
    
    /// Compute distance for multi-level residual quantization
    fn compute_multi_level_distance(
        &self, 
        query_residual: &Vector, 
        level_codes: &[Vec<u8>], 
        final_residual: &Option<Vector>
    ) -> Result<f32> {
        let mut reconstructed_residual = Vector::new(vec![0.0; query_residual.dimensions]);
        
        // Reconstruct vector from quantized levels
        for (level, codes) in level_codes.iter().enumerate() {
            if level < self.multi_level_pq.len() {
                let level_reconstruction = self.multi_level_pq[level].decode_vector(codes)?;
                reconstructed_residual = reconstructed_residual.add(&level_reconstruction)?;
            }
        }
        
        // Add final unquantized residual if present
        if let Some(final_res) = final_residual {
            reconstructed_residual = reconstructed_residual.add(final_res)?;
        }
        
        // Compute distance between query residual and reconstructed residual
        query_residual.euclidean_distance(&reconstructed_residual)
    }
    
    /// Compute distance for multi-codebook quantization
    fn compute_multi_codebook_distance(
        &self, 
        query_residual: &Vector, 
        codebook_codes: &[Vec<u8>], 
        weights: &[f32]
    ) -> Result<f32> {
        let mut weighted_distance = 0.0;
        let mut total_weight = 0.0;
        
        // Compute weighted combination of distances from all codebooks
        for (i, codes) in codebook_codes.iter().enumerate() {
            if i < self.multi_codebook_pq.len() && i < weights.len() {
                let codebook_distance = self.multi_codebook_pq[i].compute_distance(query_residual, codes)?;
                weighted_distance += weights[i] * codebook_distance;
                total_weight += weights[i];
            }
        }
        
        // Normalize by total weight
        if total_weight > 0.0 {
            Ok(weighted_distance / total_weight)
        } else {
            Ok(f32::INFINITY)
        }
    }
    
    /// Train product quantizer on collected residuals
    fn train_pq(&mut self, residuals: &[Vector]) -> Result<()> {
        match &self.quantization {
            QuantizationStrategy::ProductQuantization(_) => {
                if let Some(ref mut pq_index) = self.pq_index {
                    pq_index.train(residuals)?;
                }
            },
            QuantizationStrategy::ResidualQuantization { levels, .. } => {
                self.train_multi_level_pq(residuals, *levels)?;
            },
            QuantizationStrategy::MultiCodebook { .. } => {
                self.train_multi_codebook_pq(residuals)?;
            },
            QuantizationStrategy::None => {},
        }
        Ok(())
    }
    
    /// Train multi-level residual quantization
    fn train_multi_level_pq(&mut self, residuals: &[Vector], levels: usize) -> Result<()> {
        let mut current_residuals = residuals.to_vec();
        
        for level in 0..levels.min(self.multi_level_pq.len()) {
            // Train PQ at this level
            self.multi_level_pq[level].train(&current_residuals)?;
            
            // Compute residuals for next level by subtracting quantized approximation
            let mut next_residuals = Vec::new();
            for residual in &current_residuals {
                let codes = self.multi_level_pq[level].encode(residual)?;
                let approximation = self.multi_level_pq[level].decode_vector(&codes)?;
                let next_residual = residual.subtract(&approximation)?;
                next_residuals.push(next_residual);
            }
            current_residuals = next_residuals;
        }
        
        Ok(())
    }
    
    /// Train multi-codebook quantization
    fn train_multi_codebook_pq(&mut self, residuals: &[Vector]) -> Result<()> {
        // Train each codebook independently on the same residuals
        for pq_index in &mut self.multi_codebook_pq {
            pq_index.train(residuals)?;
        }
        
        // Optionally, optimize codebook weights based on reconstruction quality
        self.optimize_codebook_weights(residuals)?;
        
        Ok(())
    }
    
    /// Optimize weights for multi-codebook quantization
    fn optimize_codebook_weights(&mut self, residuals: &[Vector]) -> Result<()> {
        if self.multi_codebook_pq.is_empty() || residuals.is_empty() {
            return Ok(());
        }
        
        let num_codebooks = self.multi_codebook_pq.len();
        let mut reconstruction_errors = vec![0.0; num_codebooks];
        
        // Compute reconstruction error for each codebook
        for (i, pq_index) in self.multi_codebook_pq.iter().enumerate() {
            let mut total_error = 0.0;
            for residual in residuals {
                let codes = pq_index.encode(residual)?;
                let reconstruction = pq_index.decode_vector(&codes)?;
                let error = residual.euclidean_distance(&reconstruction).unwrap_or(f32::INFINITY);
                total_error += error;
            }
            reconstruction_errors[i] = total_error / residuals.len() as f32;
        }
        
        // Compute weights inversely proportional to reconstruction error
        let max_error = reconstruction_errors.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_error > 0.0 {
            let mut total_weight = 0.0;
            for i in 0..num_codebooks {
                // Higher weight for lower error
                self.codebook_weights[i] = (max_error - reconstruction_errors[i] + 1e-6) / max_error;
                total_weight += self.codebook_weights[i];
            }
            
            // Normalize weights
            if total_weight > 0.0 {
                for weight in &mut self.codebook_weights {
                    *weight /= total_weight;
                }
            }
        }
        
        Ok(())
    }
    
    /// Get statistics about this inverted list
    fn stats(&self) -> InvertedListStats {
        let mut full_vectors = 0;
        let mut quantized_vectors = 0;
        let mut multi_level_vectors = 0;
        let mut multi_codebook_vectors = 0;
        
        for (_, storage) in &self.vectors {
            match storage {
                VectorStorage::Full(_) => full_vectors += 1,
                VectorStorage::Quantized(_) => quantized_vectors += 1,
                VectorStorage::MultiLevelQuantized { .. } => {
                    quantized_vectors += 1;
                    multi_level_vectors += 1;
                },
                VectorStorage::MultiCodebook { .. } => {
                    quantized_vectors += 1;
                    multi_codebook_vectors += 1;
                },
            }
        }
        
        let total_vectors = self.vectors.len();
        let compression_ratio = if total_vectors > 0 {
            quantized_vectors as f32 / total_vectors as f32
        } else {
            0.0
        };
        
        InvertedListStats {
            total_vectors,
            full_vectors,
            quantized_vectors,
            compression_ratio,
            multi_level_vectors,
            multi_codebook_vectors,
            quantization_strategy: self.quantization.clone(),
        }
    }
}

/// Statistics for an inverted list
#[derive(Debug, Clone)]
pub struct InvertedListStats {
    pub total_vectors: usize,
    pub full_vectors: usize,
    pub quantized_vectors: usize,
    pub compression_ratio: f32,
    pub multi_level_vectors: usize,
    pub multi_codebook_vectors: usize,
    pub quantization_strategy: QuantizationStrategy,
}

/// IVF index for approximate nearest neighbor search
pub struct IvfIndex {
    config: IvfConfig,
    /// Cluster centroids
    centroids: Vec<Vector>,
    /// Inverted lists (one per cluster)
    inverted_lists: Vec<Arc<RwLock<InvertedList>>>,
    /// Dimensions of vectors
    dimensions: Option<usize>,
    /// Total number of vectors
    n_vectors: usize,
    /// Whether the index has been trained
    is_trained: bool,
}

impl IvfIndex {
    /// Create a new IVF index
    pub fn new(config: IvfConfig) -> Result<Self> {
        let mut inverted_lists = Vec::with_capacity(config.n_clusters);
        
        // Determine quantization strategy (backward compatibility support)
        let quantization = if config.enable_residual_quantization {
            if let Some(ref pq_config) = config.pq_config {
                QuantizationStrategy::ProductQuantization(pq_config.clone())
            } else {
                return Err(anyhow!("PQ config required when residual quantization is enabled"));
            }
        } else {
            config.quantization.clone()
        };
        
        for _ in 0..config.n_clusters {
            let inverted_list = Arc::new(RwLock::new(InvertedList::new_with_quantization(quantization.clone())?));
            inverted_lists.push(inverted_list);
        }

        Ok(Self {
            config,
            centroids: Vec::new(),
            inverted_lists,
            dimensions: None,
            n_vectors: 0,
            is_trained: false,
        })
    }
    
    /// Create a new IVF index with product quantization
    pub fn new_with_product_quantization(
        n_clusters: usize, 
        n_probes: usize, 
        pq_config: PQConfig
    ) -> Result<Self> {
        let config = IvfConfig {
            n_clusters,
            n_probes,
            quantization: QuantizationStrategy::ProductQuantization(pq_config),
            ..Default::default()
        };
        Self::new(config)
    }
    
    /// Create a new IVF index with multi-level residual quantization
    pub fn new_with_multi_level_quantization(
        n_clusters: usize,
        n_probes: usize,
        levels: usize,
        pq_configs: Vec<PQConfig>
    ) -> Result<Self> {
        if pq_configs.len() < levels {
            return Err(anyhow!("Number of PQ configs must be at least equal to levels"));
        }
        
        let config = IvfConfig {
            n_clusters,
            n_probes,
            quantization: QuantizationStrategy::ResidualQuantization { levels, pq_configs },
            ..Default::default()
        };
        Self::new(config)
    }
    
    /// Create a new IVF index with multi-codebook quantization
    pub fn new_with_multi_codebook_quantization(
        n_clusters: usize,
        n_probes: usize,
        num_codebooks: usize,
        pq_configs: Vec<PQConfig>
    ) -> Result<Self> {
        if pq_configs.len() != num_codebooks {
            return Err(anyhow!("Number of PQ configs must equal number of codebooks"));
        }
        
        let config = IvfConfig {
            n_clusters,
            n_probes,
            quantization: QuantizationStrategy::MultiCodebook { num_codebooks, pq_configs },
            ..Default::default()
        };
        Self::new(config)
    }
    
    /// Create a new IVF index with residual quantization enabled (deprecated)
    pub fn new_with_residual_quantization(
        n_clusters: usize, 
        n_probes: usize, 
        pq_config: PQConfig
    ) -> Result<Self> {
        Self::new_with_product_quantization(n_clusters, n_probes, pq_config)
    }

    /// Train the index with a sample of vectors
    pub fn train(&mut self, training_vectors: &[Vector]) -> Result<()> {
        if training_vectors.is_empty() {
            return Err(anyhow!("Cannot train IVF index with empty training set"));
        }

        // Validate dimensions
        let dims = training_vectors[0].dimensions;
        if !training_vectors.iter().all(|v| v.dimensions == dims) {
            return Err(anyhow!("All training vectors must have the same dimensions"));
        }

        self.dimensions = Some(dims);

        // Initialize centroids using k-means++
        self.centroids = self.initialize_centroids_kmeans_plus_plus(training_vectors)?;

        // Run k-means clustering
        let mut iteration = 0;
        let mut prev_error = f32::INFINITY;

        while iteration < self.config.max_iterations {
            // Assign vectors to nearest centroids
            let mut clusters: Vec<Vec<&Vector>> = vec![Vec::new(); self.config.n_clusters];
            
            for vector in training_vectors {
                let nearest_idx = self.find_nearest_centroid(vector)?;
                clusters[nearest_idx].push(vector);
            }

            // Update centroids
            let mut total_error = 0.0;
            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let new_centroid = self.compute_centroid(cluster);
                    total_error += self.centroids[i]
                        .euclidean_distance(&new_centroid)
                        .unwrap_or(0.0);
                    self.centroids[i] = new_centroid;
                }
            }

            // Check convergence
            if (prev_error - total_error).abs() < self.config.convergence_threshold {
                break;
            }

            prev_error = total_error;
            iteration += 1;
        }

        self.is_trained = true;
        
        // Train quantization if enabled
        if !matches!(self.config.quantization, QuantizationStrategy::None) || self.config.enable_residual_quantization {
            self.train_residual_quantization(training_vectors)?;
        }
        
        Ok(())
    }
    
    /// Train residual quantization on all clusters
    fn train_residual_quantization(&mut self, training_vectors: &[Vector]) -> Result<()> {
        // Collect residuals for each cluster
        let mut cluster_residuals: Vec<Vec<Vector>> = vec![Vec::new(); self.config.n_clusters];
        
        for vector in training_vectors {
            let cluster_idx = self.find_nearest_centroid(vector)?;
            let centroid = &self.centroids[cluster_idx];
            let residual = vector.subtract(centroid)?;
            cluster_residuals[cluster_idx].push(residual);
        }
        
        // Train PQ for each cluster that has enough residuals
        for (cluster_idx, residuals) in cluster_residuals.iter().enumerate() {
            if residuals.len() > 10 { // Minimum threshold for training
                let mut list = self.inverted_lists[cluster_idx].write().unwrap();
                list.train_pq(residuals)?;
            }
        }
        
        Ok(())
    }

    /// Initialize centroids using k-means++ algorithm
    fn initialize_centroids_kmeans_plus_plus(&self, vectors: &[Vector]) -> Result<Vec<Vector>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.config.seed.unwrap_or(42).hash(&mut hasher);
        let mut rng_state = hasher.finish();

        let mut centroids = Vec::with_capacity(self.config.n_clusters);

        // Choose first centroid randomly
        let first_idx = (rng_state as usize) % vectors.len();
        centroids.push(vectors[first_idx].clone());

        // Choose remaining centroids
        while centroids.len() < self.config.n_clusters {
            let mut distances = Vec::with_capacity(vectors.len());
            let mut sum_distances = 0.0;

            // Calculate distance to nearest centroid for each vector
            for vector in vectors {
                let min_dist = centroids
                    .iter()
                    .map(|c| vector.euclidean_distance(c).unwrap_or(f32::INFINITY))
                    .fold(f32::INFINITY, |a, b| a.min(b));
                
                distances.push(min_dist * min_dist); // Square for k-means++
                sum_distances += min_dist * min_dist;
            }

            // Choose next centroid with probability proportional to squared distance
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let threshold = (rng_state as f32 / u64::MAX as f32) * sum_distances;

            let mut cumulative = 0.0;
            for (i, &dist) in distances.iter().enumerate() {
                cumulative += dist;
                if cumulative >= threshold {
                    centroids.push(vectors[i].clone());
                    break;
                }
            }
        }

        Ok(centroids)
    }

    /// Compute centroid of a cluster
    fn compute_centroid(&self, cluster: &[&Vector]) -> Vector {
        if cluster.is_empty() {
            return Vector::new(vec![0.0; self.dimensions.unwrap_or(0)]);
        }

        let dims = cluster[0].dimensions;
        let mut sum = vec![0.0; dims];

        for vector in cluster {
            let values = vector.as_f32();
            for (i, &val) in values.iter().enumerate() {
                sum[i] += val;
            }
        }

        let count = cluster.len() as f32;
        for val in &mut sum {
            *val /= count;
        }

        Vector::new(sum)
    }

    /// Find the nearest centroid for a vector
    fn find_nearest_centroid(&self, vector: &Vector) -> Result<usize> {
        if self.centroids.is_empty() {
            return Err(anyhow!("No centroids available"));
        }

        let mut min_distance = f32::INFINITY;
        let mut nearest_idx = 0;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let distance = vector.euclidean_distance(centroid)?;
            if distance < min_distance {
                min_distance = distance;
                nearest_idx = i;
            }
        }

        Ok(nearest_idx)
    }

    /// Find the n_probes nearest centroids for a query
    fn find_nearest_centroids(&self, query: &Vector, n_probes: usize) -> Result<Vec<usize>> {
        let mut distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| {
                let dist = query.euclidean_distance(centroid).unwrap_or(f32::INFINITY);
                (i, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        Ok(distances
            .into_iter()
            .take(n_probes.min(self.centroids.len()))
            .map(|(i, _)| i)
            .collect())
    }

    /// Get comprehensive statistics about the IVF index including compression info
    pub fn stats(&self) -> IvfStats {
        let mut total_list_stats = InvertedListStats {
            total_vectors: 0,
            full_vectors: 0,
            quantized_vectors: 0,
            compression_ratio: 0.0,
            multi_level_vectors: 0,
            multi_codebook_vectors: 0,
            quantization_strategy: QuantizationStrategy::None,
        };
        
        let mut cluster_stats = Vec::new();
        let mut vectors_per_cluster = Vec::new();
        let mut non_empty_clusters = 0;
        
        for list in &self.inverted_lists {
            let list_guard = list.read().unwrap();
            let stats = list_guard.stats();
            
            total_list_stats.total_vectors += stats.total_vectors;
            total_list_stats.full_vectors += stats.full_vectors;
            total_list_stats.quantized_vectors += stats.quantized_vectors;
            
            vectors_per_cluster.push(stats.total_vectors);
            if stats.total_vectors > 0 {
                non_empty_clusters += 1;
            }
            
            cluster_stats.push(stats);
        }
        
        // Calculate overall compression ratio
        if total_list_stats.total_vectors > 0 {
            total_list_stats.compression_ratio = 
                total_list_stats.quantized_vectors as f32 / total_list_stats.total_vectors as f32;
        }
        
        let avg_vectors_per_cluster = if self.config.n_clusters > 0 {
            self.n_vectors as f32 / self.config.n_clusters as f32
        } else {
            0.0
        };
        
        IvfStats {
            n_clusters: self.config.n_clusters,
            n_probes: self.config.n_probes,
            n_vectors: self.n_vectors,
            is_trained: self.is_trained,
            dimensions: self.dimensions,
            vectors_per_cluster,
            avg_vectors_per_cluster,
            non_empty_clusters,
            enable_residual_quantization: self.config.enable_residual_quantization,
            quantization_strategy: self.config.quantization.clone(),
            compression_stats: Some(total_list_stats),
            cluster_stats,
        }
    }
}

impl VectorIndex for IvfIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        if !self.is_trained {
            return Err(anyhow!("IVF index must be trained before inserting vectors"));
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

        // Find nearest centroid
        let cluster_idx = self.find_nearest_centroid(&vector)?;
        let centroid = &self.centroids[cluster_idx];
        
        let mut list = self.inverted_lists[cluster_idx].write().unwrap();
        
        // Handle quantization based on strategy
        match &self.config.quantization {
            QuantizationStrategy::None => {
                if self.config.enable_residual_quantization {
                    // Backward compatibility: use residual quantization
                    let residual = vector.subtract(centroid)?;
                    list.add_residual(uri, residual, centroid)?;
                } else {
                    list.add_full(uri, vector);
                }
            },
            _ => {
                // Use new quantization strategies
                let residual = vector.subtract(centroid)?;
                list.add_residual(uri, residual, centroid)?;
            }
        }
        
        self.n_vectors += 1;
        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        if !self.is_trained {
            return Err(anyhow!("IVF index must be trained before searching"));
        }

        // Find nearest centroids to probe
        let probe_indices = self.find_nearest_centroids(query, self.config.n_probes)?;

        // Search in selected inverted lists
        let mut all_results = Vec::new();
        for idx in probe_indices {
            let list = self.inverted_lists[idx].read().unwrap();
            let centroid = &self.centroids[idx];
            let mut results = list.search(query, centroid, k)?;
            all_results.append(&mut results);
        }

        // Sort and truncate to k results
        all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        all_results.truncate(k);

        Ok(all_results)
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        if !self.is_trained {
            return Err(anyhow!("IVF index must be trained before searching"));
        }

        // Find nearest centroids to probe
        let probe_indices = self.find_nearest_centroids(query, self.config.n_probes)?;

        // Search in selected inverted lists
        let mut all_results = Vec::new();
        for idx in probe_indices {
            let list = self.inverted_lists[idx].read().unwrap();
            let centroid = &self.centroids[idx];
            let results = list.search(query, centroid, self.n_vectors)?;
            
            // Filter by threshold
            for (uri, similarity) in results {
                if similarity >= threshold {
                    all_results.push((uri, similarity));
                }
            }
        }

        // Sort by similarity
        all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(all_results)
    }
    
    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        // IVF doesn't maintain a direct URI to vector mapping
        // Would need to search through all inverted lists
        // For efficiency, we return None
        None
    }
}

/// Statistics for IVF index
#[derive(Debug, Clone)]
pub struct IvfStats {
    pub n_vectors: usize,
    pub n_clusters: usize,
    pub n_probes: usize,
    pub is_trained: bool,
    pub dimensions: Option<usize>,
    pub vectors_per_cluster: Vec<usize>,
    pub avg_vectors_per_cluster: f32,
    pub non_empty_clusters: usize,
    pub enable_residual_quantization: bool,
    pub quantization_strategy: QuantizationStrategy,
    pub compression_stats: Option<InvertedListStats>,
    pub cluster_stats: Vec<InvertedListStats>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_basic() {
        let config = IvfConfig {
            n_clusters: 4,
            n_probes: 2,
            ..Default::default()
        };
        
        let mut index = IvfIndex::new(config).unwrap();

        // Create training vectors
        let training_vectors = vec![
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![-1.0, 0.0]),
            Vector::new(vec![0.0, -1.0]),
            Vector::new(vec![0.5, 0.5]),
            Vector::new(vec![-0.5, 0.5]),
            Vector::new(vec![-0.5, -0.5]),
            Vector::new(vec![0.5, -0.5]),
        ];

        // Train the index
        index.train(&training_vectors).unwrap();
        assert!(index.is_trained);

        // Insert vectors
        for (i, vec) in training_vectors.iter().enumerate() {
            index.insert(format!("vec{}", i), vec.clone()).unwrap();
        }

        // Search for nearest neighbors
        let query = Vector::new(vec![0.9, 0.1]);
        let results = index.search_knn(&query, 3).unwrap();
        
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        
        // The first result should be vec0 (closest to [1.0, 0.0])
        assert_eq!(results[0].0, "vec0");
    }

    #[test]
    fn test_ivf_threshold_search() {
        let config = IvfConfig {
            n_clusters: 2,
            n_probes: 2,
            ..Default::default()
        };
        
        let mut index = IvfIndex::new(config).unwrap();

        // Create and train with vectors
        let training_vectors = vec![
            Vector::new(vec![1.0, 0.0, 0.0]),
            Vector::new(vec![0.0, 1.0, 0.0]),
            Vector::new(vec![0.0, 0.0, 1.0]),
            Vector::new(vec![0.5, 0.5, 0.0]),
        ];

        index.train(&training_vectors).unwrap();

        // Insert vectors
        index.insert("v1".to_string(), training_vectors[0].clone()).unwrap();
        index.insert("v2".to_string(), training_vectors[1].clone()).unwrap();
        index.insert("v3".to_string(), training_vectors[2].clone()).unwrap();
        index.insert("v4".to_string(), training_vectors[3].clone()).unwrap();

        // Search with threshold
        let query = Vector::new(vec![0.9, 0.1, 0.0]);
        let results = index.search_threshold(&query, 0.5).unwrap();

        assert!(!results.is_empty());
        // Should find vectors with similarity >= 0.5
        for (_, similarity) in &results {
            assert!(*similarity >= 0.5);
        }
    }

    #[test]
    fn test_ivf_stats() {
        let config = IvfConfig {
            n_clusters: 3,
            n_probes: 1,
            ..Default::default()
        };
        
        let mut index = IvfIndex::new(config).unwrap();

        // Train with simple vectors
        let training_vectors = vec![
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![-1.0, -1.0]),
        ];

        index.train(&training_vectors).unwrap();

        // Insert some vectors
        index.insert("a".to_string(), Vector::new(vec![1.1, 0.1])).unwrap();
        index.insert("b".to_string(), Vector::new(vec![0.1, 1.1])).unwrap();

        let stats = index.stats();
        assert_eq!(stats.n_vectors, 2);
        assert_eq!(stats.n_clusters, 3);
        assert!(stats.is_trained);
        assert_eq!(stats.dimensions, Some(2));
    }

    #[test]
    fn test_ivf_multi_level_quantization() {
        use crate::pq::PQConfig;
        
        // Create PQ configs for different levels
        let pq_config_1 = PQConfig {
            n_subquantizers: 2,
            n_bits: 8,
            ..Default::default()
        };
        let pq_config_2 = PQConfig {
            n_subquantizers: 2,
            n_bits: 4,
            ..Default::default()
        };
        
        let mut index = IvfIndex::new_with_multi_level_quantization(
            4, 2, 2, vec![pq_config_1, pq_config_2]
        ).unwrap();

        // Create training vectors
        let training_vectors = vec![
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
            Vector::new(vec![0.0, 0.0, 1.0, 0.0]),
            Vector::new(vec![0.0, 0.0, 0.0, 1.0]),
            Vector::new(vec![0.5, 0.5, 0.0, 0.0]),
            Vector::new(vec![0.0, 0.0, 0.5, 0.5]),
        ];

        // Train the index
        index.train(&training_vectors).unwrap();
        assert!(index.is_trained);

        // Insert vectors
        for (i, vec) in training_vectors.iter().enumerate() {
            index.insert(format!("vec{}", i), vec.clone()).unwrap();
        }

        // Search for nearest neighbors
        let query = Vector::new(vec![0.9, 0.1, 0.0, 0.0]);
        let results = index.search_knn(&query, 3).unwrap();
        
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        
        // Check stats
        let stats = index.stats();
        assert!(matches!(stats.quantization_strategy, QuantizationStrategy::ResidualQuantization { .. }));
        if let Some(compression_stats) = &stats.compression_stats {
            assert!(compression_stats.multi_level_vectors > 0);
        }
    }

    #[test]
    fn test_ivf_multi_codebook_quantization() {
        use crate::pq::PQConfig;
        
        // Create PQ configs for different codebooks
        let pq_config_1 = PQConfig {
            n_subquantizers: 2,
            n_bits: 8,
            ..Default::default()
        };
        let pq_config_2 = PQConfig {
            n_subquantizers: 2,  
            n_bits: 8,
            ..Default::default()
        };
        
        let mut index = IvfIndex::new_with_multi_codebook_quantization(
            4, 2, 2, vec![pq_config_1, pq_config_2]
        ).unwrap();

        // Create training vectors
        let training_vectors = vec![
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
            Vector::new(vec![0.0, 0.0, 1.0, 0.0]),
            Vector::new(vec![0.0, 0.0, 0.0, 1.0]),
            Vector::new(vec![0.5, 0.5, 0.5, 0.5]),
        ];

        // Train the index
        index.train(&training_vectors).unwrap();
        assert!(index.is_trained);

        // Insert vectors
        for (i, vec) in training_vectors.iter().enumerate() {
            index.insert(format!("vec{}", i), vec.clone()).unwrap();
        }

        // Search for nearest neighbors
        let query = Vector::new(vec![0.9, 0.1, 0.0, 0.0]);
        let results = index.search_knn(&query, 2).unwrap();
        
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        
        // Check stats
        let stats = index.stats();
        assert!(matches!(stats.quantization_strategy, QuantizationStrategy::MultiCodebook { .. }));
        if let Some(compression_stats) = &stats.compression_stats {
            assert!(compression_stats.multi_codebook_vectors > 0);
        }
    }

    #[test]
    fn test_quantization_strategies() {
        use crate::pq::PQConfig;
        
        let pq_config = PQConfig::default();
        
        // Test different quantization strategies
        let strategies = vec![
            QuantizationStrategy::None,
            QuantizationStrategy::ProductQuantization(pq_config.clone()),
            QuantizationStrategy::ResidualQuantization {
                levels: 2,
                pq_configs: vec![pq_config.clone(), pq_config.clone()],
            },
            QuantizationStrategy::MultiCodebook {
                num_codebooks: 2,
                pq_configs: vec![pq_config.clone(), pq_config.clone()],
            },
        ];
        
        for strategy in strategies {
            let config = IvfConfig {
                n_clusters: 2,
                n_probes: 1,
                quantization: strategy.clone(),
                ..Default::default()
            };
            
            let index = IvfIndex::new(config);
            assert!(index.is_ok(), "Failed to create index with strategy: {:?}", strategy);
        }
    }
}