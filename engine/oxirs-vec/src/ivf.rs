//! Inverted File (IVF) index implementation for approximate nearest neighbor search
//!
//! IVF is a clustering-based indexing method that partitions the vector space into
//! Voronoi cells. Each cell has a centroid, and vectors are assigned to their nearest
//! centroid. During search, only a subset of cells are examined, greatly reducing
//! search time at the cost of some accuracy.

use crate::{Vector, VectorIndex, VectorPrecision};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            n_clusters: 256,
            n_probes: 8,
            max_iterations: 100,
            convergence_threshold: 1e-4,
            seed: None,
        }
    }
}

/// Inverted list storing vectors for a single cluster
#[derive(Debug, Clone)]
struct InvertedList {
    /// Vectors in this cluster
    vectors: Vec<(String, Vector)>,
}

impl InvertedList {
    fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }

    fn add(&mut self, uri: String, vector: Vector) {
        self.vectors.push((uri, vector));
    }

    fn search(&self, query: &Vector, k: usize) -> Vec<(String, f32)> {
        let mut distances: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(uri, vec)| {
                let distance = query.euclidean_distance(vec).unwrap_or(f32::INFINITY);
                (uri.clone(), distance)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        
        // Convert distances to similarities (1 / (1 + distance))
        distances
            .into_iter()
            .map(|(uri, dist)| (uri, 1.0 / (1.0 + dist)))
            .collect()
    }
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
    pub fn new(config: IvfConfig) -> Self {
        let mut inverted_lists = Vec::with_capacity(config.n_clusters);
        for _ in 0..config.n_clusters {
            inverted_lists.push(Arc::new(RwLock::new(InvertedList::new())));
        }

        Self {
            config,
            centroids: Vec::new(),
            inverted_lists,
            dimensions: None,
            n_vectors: 0,
            is_trained: false,
        }
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

    /// Get index statistics
    pub fn stats(&self) -> IvfStats {
        let vectors_per_cluster: Vec<usize> = self
            .inverted_lists
            .iter()
            .map(|list| list.read().unwrap().vectors.len())
            .collect();

        let total_vectors: usize = vectors_per_cluster.iter().sum();
        let avg_vectors_per_cluster = if self.config.n_clusters > 0 {
            total_vectors as f32 / self.config.n_clusters as f32
        } else {
            0.0
        };

        let non_empty_clusters = vectors_per_cluster.iter().filter(|&&n| n > 0).count();

        IvfStats {
            n_vectors: self.n_vectors,
            n_clusters: self.config.n_clusters,
            n_probes: self.config.n_probes,
            is_trained: self.is_trained,
            dimensions: self.dimensions,
            vectors_per_cluster,
            avg_vectors_per_cluster,
            non_empty_clusters,
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

        // Find nearest centroid and add to corresponding inverted list
        let cluster_idx = self.find_nearest_centroid(&vector)?;
        let mut list = self.inverted_lists[cluster_idx].write().unwrap();
        list.add(uri, vector);
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
            let mut results = list.search(query, k);
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
            for (uri, vec) in &list.vectors {
                let distance = query.euclidean_distance(vec)?;
                let similarity = 1.0 / (1.0 + distance);
                if similarity >= threshold {
                    all_results.push((uri.clone(), similarity));
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
        
        let mut index = IvfIndex::new(config);

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
        
        let mut index = IvfIndex::new(config);

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
        
        let mut index = IvfIndex::new(config);

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
}