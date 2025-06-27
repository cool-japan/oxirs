//! Product Quantization (PQ) for memory-efficient vector compression and search
//!
//! PQ divides high-dimensional vectors into subvectors and quantizes each subvector
//! independently using k-means clustering. This achieves high compression ratios
//! while maintaining reasonable search accuracy.

use crate::{Vector, VectorIndex};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Configuration for Product Quantization
#[derive(Debug, Clone)]
pub struct PQConfig {
    /// Number of subquantizers (vector is split into this many parts)
    pub n_subquantizers: usize,
    /// Number of centroids per subquantizer (typically 256 for 8-bit codes)
    pub n_centroids: usize,
    /// Number of iterations for k-means training
    pub max_iterations: usize,
    /// Convergence threshold for k-means
    pub convergence_threshold: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            n_subquantizers: 8,
            n_centroids: 256,
            max_iterations: 50,
            convergence_threshold: 1e-4,
            seed: None,
        }
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

/// Product Quantization index
pub struct PQIndex {
    config: PQConfig,
    /// Subquantizers, one for each part of the vector
    subquantizers: Vec<SubQuantizer>,
    /// Encoded vectors (each vector is represented as n_subquantizers bytes)
    codes: Vec<(String, Vec<u8>)>,
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
            return Err(anyhow!("All training vectors must have the same dimensions"));
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
            self.subquantizers.push(SubQuantizer::new(start, end, self.config.n_centroids));
        }

        // Extract training data as f32
        let training_data: Vec<Vec<f32>> = training_vectors
            .iter()
            .map(|v| v.as_f32())
            .collect();

        // Train each subquantizer
        for (i, sq) in self.subquantizers.iter_mut().enumerate() {
            // Extract subvectors for this subquantizer
            let subvectors: Vec<Vec<f32>> = training_data
                .iter()
                .map(|v| sq.extract_subvector(v))
                .collect();

            sq.train(&subvectors, &self.config)?;
        }

        self.is_trained = true;
        Ok(())
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
        let codebook_size = self.subquantizers.iter()
            .map(|sq| sq.centroids.len() * (sq.end_dim - sq.start_dim) * 4)
            .sum::<usize>();
        
        let codes_size = self.codes.len() * self.config.n_subquantizers;
        
        codebook_size + codes_size
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
        let mut distances: Vec<(String, f32)> = self.codes
            .iter()
            .map(|(uri, codes)| {
                let dist = self.asymmetric_distance(query, codes).unwrap_or(f32::INFINITY);
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
                let values: Vec<f32> = (0..dims)
                    .map(|j| ((i + j) as f32).sin())
                    .collect();
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
}