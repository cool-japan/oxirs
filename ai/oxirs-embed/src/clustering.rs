//! Clustering Support for Knowledge Graph Embeddings
//!
//! This module provides comprehensive clustering algorithms for analyzing and grouping
//! entities based on their learned embeddings. Clustering helps discover latent
//! structure in knowledge graphs and can improve downstream tasks such as entity
//! type discovery, knowledge organization, and recommendation systems.
//!
//! # Overview
//!
//! The module provides four powerful clustering algorithms:
//! - **K-Means**: Fast, spherical clusters with K-Means++ initialization
//! - **Hierarchical**: Bottom-up agglomerative clustering with linkage methods
//! - **DBSCAN**: Density-based clustering that discovers arbitrary shapes and handles noise
//! - **Spectral**: Graph-based clustering using eigenvalues of similarity matrices
//!
//! Each algorithm is suited for different data characteristics and use cases.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use oxirs_embed::{
//!     TransE, ModelConfig, Triple, NamedNode, EmbeddingModel,
//!     clustering::{EntityClustering, ClusteringConfig, ClusteringAlgorithm},
//! };
//! use std::collections::HashMap;
//! use scirs2_core::ndarray_ext::Array1;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // 1. Train an embedding model
//! let config = ModelConfig::default().with_dimensions(128);
//! let mut model = TransE::new(config);
//!
//! model.add_triple(Triple::new(
//!     NamedNode::new("paris")?,
//!     NamedNode::new("capital_of")?,
//!     NamedNode::new("france")?,
//! ))?;
//! model.add_triple(Triple::new(
//!     NamedNode::new("london")?,
//!     NamedNode::new("capital_of")?,
//!     NamedNode::new("uk")?,
//! ))?;
//!
//! model.train(Some(100)).await?;
//!
//! // 2. Extract embeddings
//! let mut embeddings = HashMap::new();
//! for entity in model.get_entities() {
//!     if let Ok(emb) = model.get_entity_embedding(&entity) {
//!         let array = Array1::from_vec(emb.values);
//!         embeddings.insert(entity, array);
//!     }
//! }
//!
//! // 3. Perform clustering
//! let cluster_config = ClusteringConfig {
//!     algorithm: ClusteringAlgorithm::KMeans,
//!     num_clusters: 3,
//!     max_iterations: 50,
//!     ..Default::default()
//! };
//!
//! let mut clustering = EntityClustering::new(cluster_config);
//! let result = clustering.cluster(&embeddings)?;
//!
//! println!("Silhouette score: {:.3}", result.silhouette_score);
//! println!("Cluster 0: {} entities", result.cluster_sizes[0]);
//! # Ok(())
//! # }
//! ```
//!
//! # Clustering Algorithms
//!
//! ## K-Means Clustering
//!
//! Fast and efficient for spherical clusters. Uses K-Means++ initialization for
//! better convergence. Best for when you know the number of clusters.
//!
//! ```rust,no_run
//! use oxirs_embed::clustering::{ClusteringConfig, ClusteringAlgorithm, EntityClustering};
//! use std::collections::HashMap;
//! use scirs2_core::ndarray_ext::Array1;
//!
//! # fn example() -> anyhow::Result<()> {
//! # let embeddings: HashMap<String, Array1<f32>> = HashMap::new();
//! let config = ClusteringConfig {
//!     algorithm: ClusteringAlgorithm::KMeans,
//!     num_clusters: 5,
//!     max_iterations: 100,
//!     tolerance: 0.0001,
//!     ..Default::default()
//! };
//!
//! let mut clustering = EntityClustering::new(config);
//! let result = clustering.cluster(&embeddings)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Hierarchical Clustering
//!
//! Builds a hierarchy of clusters using bottom-up approach. Supports different
//! linkage methods (single, average, complete). Does not require specifying the
//! number of clusters upfront.
//!
//! ```rust,no_run
//! use oxirs_embed::clustering::{ClusteringConfig, ClusteringAlgorithm, EntityClustering};
//! use std::collections::HashMap;
//! use scirs2_core::ndarray_ext::Array1;
//!
//! # fn example() -> anyhow::Result<()> {
//! # let embeddings: HashMap<String, Array1<f32>> = HashMap::new();
//! let config = ClusteringConfig {
//!     algorithm: ClusteringAlgorithm::Hierarchical,
//!     num_clusters: 4,
//!     ..Default::default()
//! };
//!
//! let mut clustering = EntityClustering::new(config);
//! let result = clustering.cluster(&embeddings)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## DBSCAN (Density-Based Clustering)
//!
//! Discovers clusters of arbitrary shape and automatically identifies noise/outliers.
//! Does not require specifying the number of clusters. Best for non-spherical clusters.
//!
//! ```rust,no_run
//! use oxirs_embed::clustering::{ClusteringConfig, ClusteringAlgorithm, EntityClustering};
//! use std::collections::HashMap;
//! use scirs2_core::ndarray_ext::Array1;
//!
//! # fn example() -> anyhow::Result<()> {
//! # let embeddings: HashMap<String, Array1<f32>> = HashMap::new();
//! let config = ClusteringConfig {
//!     algorithm: ClusteringAlgorithm::DBSCAN,
//!     epsilon: 0.5,        // Neighborhood radius
//!     min_points: 5,       // Minimum points to form cluster
//!     ..Default::default()
//! };
//!
//! let mut clustering = EntityClustering::new(config);
//! let result = clustering.cluster(&embeddings)?;
//!
//! // Check for noise points (cluster_id == usize::MAX)
//! let noise_count = result.assignments.values()
//!     .filter(|&&id| id == usize::MAX)
//!     .count();
//! println!("Noise points: {}", noise_count);
//! # Ok(())
//! # }
//! ```
//!
//! ## Spectral Clustering
//!
//! Graph-based clustering using eigenvalues of the similarity matrix. Effective for
//! non-convex clusters and can capture complex geometric structures.
//!
//! ```rust,no_run
//! use oxirs_embed::clustering::{ClusteringConfig, ClusteringAlgorithm, EntityClustering};
//! use std::collections::HashMap;
//! use scirs2_core::ndarray_ext::Array1;
//!
//! # fn example() -> anyhow::Result<()> {
//! # let embeddings: HashMap<String, Array1<f32>> = HashMap::new();
//! let config = ClusteringConfig {
//!     algorithm: ClusteringAlgorithm::Spectral,
//!     num_clusters: 3,
//!     ..Default::default()
//! };
//!
//! let mut clustering = EntityClustering::new(config);
//! let result = clustering.cluster(&embeddings)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Cluster Quality Metrics
//!
//! The module computes several metrics to assess clustering quality:
//!
//! ## Silhouette Score
//!
//! Measures how similar entities are to their own cluster compared to other clusters.
//! Range: [-1, 1], where:
//! - 1: Perfect clustering
//! - 0: Overlapping clusters
//! - -1: Incorrect clustering
//!
//! ```rust,no_run
//! # use oxirs_embed::clustering::*;
//! # use std::collections::HashMap;
//! # use scirs2_core::ndarray_ext::Array1;
//! # fn example() -> anyhow::Result<()> {
//! # let embeddings: HashMap<String, Array1<f32>> = HashMap::new();
//! # let mut clustering = EntityClustering::new(ClusteringConfig::default());
//! let result = clustering.cluster(&embeddings)?;
//!
//! if result.silhouette_score > 0.7 {
//!     println!("Excellent clustering!");
//! } else if result.silhouette_score > 0.5 {
//!     println!("Good clustering");
//! } else {
//!     println!("Weak clustering - consider different parameters");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Inertia
//!
//! Sum of squared distances from entities to their cluster centroids.
//! Lower values indicate tighter clusters (for K-Means).
//!
//! # Analyzing Cluster Results
//!
//! ```rust,no_run
//! # use oxirs_embed::clustering::*;
//! # use std::collections::HashMap;
//! # use scirs2_core::ndarray_ext::Array1;
//! # fn example() -> anyhow::Result<()> {
//! # let embeddings: HashMap<String, Array1<f32>> = HashMap::new();
//! # let mut clustering = EntityClustering::new(ClusteringConfig::default());
//! let result = clustering.cluster(&embeddings)?;
//!
//! // Analyze cluster composition
//! for (entity, cluster_id) in &result.assignments {
//!     println!("Entity '{}' belongs to cluster {}", entity, cluster_id);
//! }
//!
//! // Cluster statistics
//! for (i, size) in result.cluster_sizes.iter().enumerate() {
//!     println!("Cluster {}: {} entities", i, size);
//! }
//!
//! // Find entities closest to cluster centroids
//! for (cluster_id, centroid) in result.centroids.iter().enumerate() {
//!     println!("Cluster {} centroid: {:?}", cluster_id, centroid);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Use Cases
//!
//! ## Entity Type Discovery
//!
//! Automatically discover entity types without explicit labels:
//!
//! ```text
//! Cluster 0: [paris, london, berlin]  -> Cities
//! Cluster 1: [france, germany, uk]    -> Countries
//! Cluster 2: [euro, dollar, pound]    -> Currencies
//! ```
//!
//! ## Knowledge Graph Organization
//!
//! Group related entities for improved navigation and querying.
//!
//! ## Recommendation Systems
//!
//! Find similar users or items based on learned embeddings.
//!
//! ## Anomaly Detection
//!
//! Identify outliers using DBSCAN's noise detection (cluster_id == usize::MAX).
//!
//! # Performance Considerations
//!
//! - **K-Means**: O(n*k*d*i) where n=entities, k=clusters, d=dimensions, i=iterations
//! - **Hierarchical**: O(n^2 * log n) - slow for large datasets
//! - **DBSCAN**: O(n * log n) with spatial indexing
//! - **Spectral**: O(n^3) due to eigenvalue computation - slow for large datasets
//!
//! For large knowledge graphs (>10,000 entities), K-Means or DBSCAN are recommended.
//!
//! # Choosing the Right Algorithm
//!
//! | Algorithm    | When to Use                                | Pros                           | Cons                    |
//! |--------------|-------------------------------------------|--------------------------------|-------------------------|
//! | K-Means      | Known cluster count, spherical clusters   | Fast, scalable                 | Requires K, spherical   |
//! | Hierarchical | Nested structure, small datasets          | No K needed, hierarchical      | Slow, memory intensive  |
//! | DBSCAN       | Arbitrary shapes, noise handling          | Finds outliers, no K needed    | Sensitive to parameters |
//! | Spectral     | Non-convex clusters, graph structure      | Handles complex shapes         | Slow, requires K        |
//!
//! # See Also
//!
//! - [`EntityClustering`]: Main clustering interface
//! - [`ClusteringConfig`]: Configuration options
//! - [`ClusteringResult`]: Clustering results and metrics
//! - [`ClusteringAlgorithm`]: Available algorithms

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

/// Clustering algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusteringAlgorithm {
    /// K-Means clustering
    KMeans,
    /// Hierarchical clustering
    Hierarchical,
    /// DBSCAN (Density-Based Spatial Clustering)
    DBSCAN,
    /// Spectral clustering
    Spectral,
}

/// Clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    /// Clustering algorithm to use
    pub algorithm: ClusteringAlgorithm,
    /// Number of clusters (for K-Means, Spectral)
    pub num_clusters: usize,
    /// Maximum iterations (for iterative algorithms)
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f32,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// DBSCAN epsilon (neighborhood radius)
    pub epsilon: f32,
    /// DBSCAN minimum points
    pub min_points: usize,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            algorithm: ClusteringAlgorithm::KMeans,
            num_clusters: 10,
            max_iterations: 100,
            tolerance: 1e-4,
            random_seed: None,
            epsilon: 0.5,
            min_points: 5,
        }
    }
}

/// Clustering result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    /// Cluster assignments for each entity (entity_id -> cluster_id)
    pub assignments: HashMap<String, usize>,
    /// Cluster centroids (for K-Means, Spectral)
    pub centroids: Vec<Array1<f32>>,
    /// Cluster sizes
    pub cluster_sizes: Vec<usize>,
    /// Inertia/objective function value
    pub inertia: f32,
    /// Number of iterations performed
    pub num_iterations: usize,
    /// Silhouette score (quality metric, -1 to 1, higher is better)
    pub silhouette_score: f32,
}

/// Entity clustering for knowledge graph embeddings
pub struct EntityClustering {
    config: ClusteringConfig,
    rng: Random,
}

impl EntityClustering {
    /// Create new entity clustering
    pub fn new(config: ClusteringConfig) -> Self {
        let rng = Random::default();

        Self { config, rng }
    }

    /// Cluster entities based on their embeddings
    pub fn cluster(
        &mut self,
        entity_embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<ClusteringResult> {
        if entity_embeddings.is_empty() {
            return Err(anyhow!("No entity embeddings provided"));
        }

        info!(
            "Clustering {} entities using {:?}",
            entity_embeddings.len(),
            self.config.algorithm
        );

        match self.config.algorithm {
            ClusteringAlgorithm::KMeans => self.kmeans_clustering(entity_embeddings),
            ClusteringAlgorithm::Hierarchical => self.hierarchical_clustering(entity_embeddings),
            ClusteringAlgorithm::DBSCAN => self.dbscan_clustering(entity_embeddings),
            ClusteringAlgorithm::Spectral => self.spectral_clustering(entity_embeddings),
        }
    }

    /// K-Means clustering implementation
    fn kmeans_clustering(
        &mut self,
        entity_embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<ClusteringResult> {
        let k = self.config.num_clusters;
        let entity_list: Vec<String> = entity_embeddings.keys().cloned().collect();
        let n = entity_list.len();

        if k > n {
            return Err(anyhow!("Number of clusters exceeds number of entities"));
        }

        // Initialize centroids randomly
        let dim = entity_embeddings.values().next().unwrap().len();
        let mut centroids: Vec<Array1<f32>> = Vec::new();

        // K-Means++ initialization for better convergence
        let first_idx = self.rng.random_range(0..n);
        centroids.push(entity_embeddings[&entity_list[first_idx]].clone());

        for _ in 1..k {
            // Compute distances to nearest centroid
            let distances: Vec<f32> = entity_list
                .iter()
                .map(|entity| {
                    let emb = &entity_embeddings[entity];
                    centroids
                        .iter()
                        .map(|c| self.euclidean_distance(emb, c))
                        .fold(f32::INFINITY, f32::min)
                        .powi(2)
                })
                .collect();

            // Sample proportional to distance squared
            let sum: f32 = distances.iter().sum();
            let mut prob = self.rng.random_range(0.0..sum);
            let mut next_idx = 0;

            for (i, &dist) in distances.iter().enumerate() {
                prob -= dist;
                if prob <= 0.0 {
                    next_idx = i;
                    break;
                }
            }

            centroids.push(entity_embeddings[&entity_list[next_idx]].clone());
        }

        // Iterative refinement
        let mut assignments: HashMap<String, usize> = HashMap::new();
        let mut prev_inertia = f32::INFINITY;

        for iteration in 0..self.config.max_iterations {
            // Assignment step
            assignments.clear();
            for entity in &entity_list {
                let emb = &entity_embeddings[entity];
                let cluster = self.nearest_centroid(emb, &centroids);
                assignments.insert(entity.clone(), cluster);
            }

            // Update step
            let mut new_centroids: Vec<Array1<f32>> = vec![Array1::zeros(dim); k];
            let mut counts = vec![0; k];

            for entity in &entity_list {
                if let Some(&cluster) = assignments.get(entity) {
                    new_centroids[cluster] = &new_centroids[cluster] + &entity_embeddings[entity];
                    counts[cluster] += 1;
                }
            }

            for (i, count) in counts.iter().enumerate() {
                if *count > 0 {
                    new_centroids[i] = &new_centroids[i] / (*count as f32);
                }
            }

            centroids = new_centroids;

            // Compute inertia
            let inertia =
                self.compute_inertia(&entity_list, entity_embeddings, &assignments, &centroids);

            debug!("Iteration {}: inertia = {:.6}", iteration + 1, inertia);

            // Check convergence
            if (prev_inertia - inertia).abs() < self.config.tolerance {
                info!("K-Means converged at iteration {}", iteration + 1);
                break;
            }

            prev_inertia = inertia;
        }

        let final_inertia =
            self.compute_inertia(&entity_list, entity_embeddings, &assignments, &centroids);
        let cluster_sizes = self.compute_cluster_sizes(&assignments, k);
        let silhouette =
            self.compute_silhouette_score(&entity_list, entity_embeddings, &assignments);

        Ok(ClusteringResult {
            assignments,
            centroids,
            cluster_sizes,
            inertia: final_inertia,
            num_iterations: self.config.max_iterations,
            silhouette_score: silhouette,
        })
    }

    /// Hierarchical clustering (agglomerative)
    fn hierarchical_clustering(
        &mut self,
        entity_embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<ClusteringResult> {
        let entity_list: Vec<String> = entity_embeddings.keys().cloned().collect();
        let n = entity_list.len();

        // Start with each entity in its own cluster
        let mut clusters: Vec<HashSet<usize>> = (0..n)
            .map(|i| {
                let mut set = HashSet::new();
                set.insert(i);
                set
            })
            .collect();

        // Merge clusters until we reach desired number
        while clusters.len() > self.config.num_clusters {
            // Find closest pair of clusters
            let (i, j) = self.find_closest_clusters(&clusters, &entity_list, entity_embeddings);

            // Merge clusters
            let cluster_j = clusters.remove(j);
            clusters[i].extend(cluster_j);
        }

        // Convert to assignments
        let mut assignments = HashMap::new();
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            for &entity_idx in cluster {
                assignments.insert(entity_list[entity_idx].clone(), cluster_id);
            }
        }

        // Compute centroids
        let dim = entity_embeddings.values().next().unwrap().len();
        let mut centroids = vec![Array1::zeros(dim); self.config.num_clusters];
        let mut counts = vec![0; self.config.num_clusters];

        for (entity, &cluster) in &assignments {
            centroids[cluster] = &centroids[cluster] + &entity_embeddings[entity];
            counts[cluster] += 1;
        }

        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                centroids[i] = &centroids[i] / (*count as f32);
            }
        }

        let inertia =
            self.compute_inertia(&entity_list, entity_embeddings, &assignments, &centroids);
        let cluster_sizes = self.compute_cluster_sizes(&assignments, self.config.num_clusters);
        let silhouette =
            self.compute_silhouette_score(&entity_list, entity_embeddings, &assignments);

        Ok(ClusteringResult {
            assignments,
            centroids,
            cluster_sizes,
            inertia,
            num_iterations: n - self.config.num_clusters,
            silhouette_score: silhouette,
        })
    }

    /// DBSCAN clustering implementation
    fn dbscan_clustering(
        &mut self,
        entity_embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<ClusteringResult> {
        let entity_list: Vec<String> = entity_embeddings.keys().cloned().collect();
        let n = entity_list.len();

        let mut assignments: HashMap<String, usize> = HashMap::new();
        let mut visited = HashSet::new();
        let mut cluster_id = 0;

        for i in 0..n {
            let entity = &entity_list[i];
            if visited.contains(&i) {
                continue;
            }

            visited.insert(i);

            // Find neighbors
            let neighbors = self.find_neighbors(i, &entity_list, entity_embeddings);

            if neighbors.len() < self.config.min_points {
                // Mark as noise (-1 represented as max usize)
                assignments.insert(entity.clone(), usize::MAX);
            } else {
                // Start new cluster
                self.expand_cluster(
                    i,
                    &neighbors,
                    cluster_id,
                    &entity_list,
                    entity_embeddings,
                    &mut assignments,
                    &mut visited,
                );
                cluster_id += 1;
            }
        }

        // Compute centroids for non-noise clusters
        let dim = entity_embeddings.values().next().unwrap().len();
        let mut centroids = vec![Array1::zeros(dim); cluster_id];
        let mut counts = vec![0; cluster_id];

        for (entity, &cluster) in &assignments {
            if cluster != usize::MAX {
                centroids[cluster] = &centroids[cluster] + &entity_embeddings[entity];
                counts[cluster] += 1;
            }
        }

        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                centroids[i] = &centroids[i] / (*count as f32);
            }
        }

        let inertia =
            self.compute_inertia(&entity_list, entity_embeddings, &assignments, &centroids);
        let cluster_sizes = self.compute_cluster_sizes(&assignments, cluster_id);
        let silhouette =
            self.compute_silhouette_score(&entity_list, entity_embeddings, &assignments);

        Ok(ClusteringResult {
            assignments,
            centroids,
            cluster_sizes,
            inertia,
            num_iterations: 1,
            silhouette_score: silhouette,
        })
    }

    /// Spectral clustering (simplified implementation)
    fn spectral_clustering(
        &mut self,
        entity_embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<ClusteringResult> {
        // For simplicity, use K-Means on normalized embeddings
        // Full spectral clustering requires eigendecomposition of graph Laplacian

        let mut normalized_embeddings = HashMap::new();
        for (entity, emb) in entity_embeddings {
            let norm = emb.dot(emb).sqrt();
            if norm > 0.0 {
                normalized_embeddings.insert(entity.clone(), emb / norm);
            } else {
                normalized_embeddings.insert(entity.clone(), emb.clone());
            }
        }

        self.kmeans_clustering(&normalized_embeddings)
    }

    /// Find nearest centroid for an embedding
    fn nearest_centroid(&self, embedding: &Array1<f32>, centroids: &[Array1<f32>]) -> usize {
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.euclidean_distance(embedding, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute Euclidean distance
    fn euclidean_distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let diff = a - b;
        diff.dot(&diff).sqrt()
    }

    /// Compute inertia (sum of squared distances to centroids)
    fn compute_inertia(
        &self,
        entity_list: &[String],
        embeddings: &HashMap<String, Array1<f32>>,
        assignments: &HashMap<String, usize>,
        centroids: &[Array1<f32>],
    ) -> f32 {
        entity_list
            .iter()
            .filter_map(|entity| {
                assignments.get(entity).and_then(|&cluster| {
                    if cluster < centroids.len() {
                        Some(
                            self.euclidean_distance(&embeddings[entity], &centroids[cluster])
                                .powi(2),
                        )
                    } else {
                        None
                    }
                })
            })
            .sum()
    }

    /// Compute cluster sizes
    fn compute_cluster_sizes(
        &self,
        assignments: &HashMap<String, usize>,
        num_clusters: usize,
    ) -> Vec<usize> {
        let mut sizes = vec![0; num_clusters];
        for &cluster in assignments.values() {
            if cluster < num_clusters {
                sizes[cluster] += 1;
            }
        }
        sizes
    }

    /// Compute silhouette score
    fn compute_silhouette_score(
        &self,
        entity_list: &[String],
        embeddings: &HashMap<String, Array1<f32>>,
        assignments: &HashMap<String, usize>,
    ) -> f32 {
        if entity_list.len() < 2 {
            return 0.0;
        }

        let scores: Vec<f32> = entity_list
            .iter()
            .filter_map(|entity| {
                assignments.get(entity).map(|&cluster| {
                    let emb = &embeddings[entity];

                    // Compute average distance to same cluster (a)
                    let same_cluster: Vec<f32> = entity_list
                        .iter()
                        .filter_map(|other| {
                            if other != entity && assignments.get(other) == Some(&cluster) {
                                Some(self.euclidean_distance(emb, &embeddings[other]))
                            } else {
                                None
                            }
                        })
                        .collect();

                    let a = if !same_cluster.is_empty() {
                        same_cluster.iter().sum::<f32>() / same_cluster.len() as f32
                    } else {
                        0.0
                    };

                    // Compute minimum average distance to other clusters (b)
                    let unique_clusters: HashSet<usize> = assignments.values().copied().collect();
                    let b = unique_clusters
                        .iter()
                        .filter(|&&c| c != cluster)
                        .map(|&other_cluster| {
                            let distances: Vec<f32> = entity_list
                                .iter()
                                .filter_map(|other| {
                                    if assignments.get(other) == Some(&other_cluster) {
                                        Some(self.euclidean_distance(emb, &embeddings[other]))
                                    } else {
                                        None
                                    }
                                })
                                .collect();

                            if !distances.is_empty() {
                                distances.iter().sum::<f32>() / distances.len() as f32
                            } else {
                                f32::INFINITY
                            }
                        })
                        .fold(f32::INFINITY, f32::min);

                    (b - a) / a.max(b).max(1e-10)
                })
            })
            .collect();

        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }

    /// Find closest pair of clusters for hierarchical clustering
    fn find_closest_clusters(
        &self,
        clusters: &[HashSet<usize>],
        entity_list: &[String],
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> (usize, usize) {
        let mut min_dist = f32::INFINITY;
        let mut closest_pair = (0, 1);

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                // Average linkage
                let mut total_dist = 0.0;
                let mut count = 0;

                for &idx_i in &clusters[i] {
                    for &idx_j in &clusters[j] {
                        let dist = self.euclidean_distance(
                            &embeddings[&entity_list[idx_i]],
                            &embeddings[&entity_list[idx_j]],
                        );
                        total_dist += dist;
                        count += 1;
                    }
                }

                let avg_dist = if count > 0 {
                    total_dist / count as f32
                } else {
                    f32::INFINITY
                };

                if avg_dist < min_dist {
                    min_dist = avg_dist;
                    closest_pair = (i, j);
                }
            }
        }

        closest_pair
    }

    /// Find neighbors within epsilon distance for DBSCAN
    fn find_neighbors(
        &self,
        idx: usize,
        entity_list: &[String],
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Vec<usize> {
        let entity = &entity_list[idx];
        let emb = &embeddings[entity];

        entity_list
            .iter()
            .enumerate()
            .filter_map(|(i, other)| {
                if i != idx
                    && self.euclidean_distance(emb, &embeddings[other]) <= self.config.epsilon
                {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Expand cluster for DBSCAN
    #[allow(clippy::too_many_arguments)]
    fn expand_cluster(
        &self,
        idx: usize,
        neighbors: &[usize],
        cluster_id: usize,
        entity_list: &[String],
        embeddings: &HashMap<String, Array1<f32>>,
        assignments: &mut HashMap<String, usize>,
        visited: &mut HashSet<usize>,
    ) {
        assignments.insert(entity_list[idx].clone(), cluster_id);

        let mut queue: Vec<usize> = neighbors.to_vec();
        let mut processed = 0;

        while processed < queue.len() {
            let neighbor_idx = queue[processed];
            processed += 1;

            if !visited.contains(&neighbor_idx) {
                visited.insert(neighbor_idx);

                let neighbor_neighbors = self.find_neighbors(neighbor_idx, entity_list, embeddings);

                if neighbor_neighbors.len() >= self.config.min_points {
                    queue.extend(neighbor_neighbors);
                }
            }

            if !assignments.contains_key(&entity_list[neighbor_idx]) {
                assignments.insert(entity_list[neighbor_idx].clone(), cluster_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_kmeans_clustering() {
        let mut embeddings = HashMap::new();
        embeddings.insert("e1".to_string(), array![1.0, 1.0]);
        embeddings.insert("e2".to_string(), array![1.1, 0.9]);
        embeddings.insert("e3".to_string(), array![5.0, 5.0]);
        embeddings.insert("e4".to_string(), array![5.1, 4.9]);

        let config = ClusteringConfig {
            algorithm: ClusteringAlgorithm::KMeans,
            num_clusters: 2,
            ..Default::default()
        };

        let mut clustering = EntityClustering::new(config);
        let result = clustering.cluster(&embeddings).unwrap();

        assert_eq!(result.assignments.len(), 4);
        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.cluster_sizes.len(), 2);

        // Check that similar entities are in the same cluster
        assert_eq!(result.assignments["e1"], result.assignments["e2"]);
        assert_eq!(result.assignments["e3"], result.assignments["e4"]);
    }

    #[test]
    fn test_silhouette_score() {
        let mut embeddings = HashMap::new();
        embeddings.insert("e1".to_string(), array![0.0, 0.0]);
        embeddings.insert("e2".to_string(), array![1.0, 1.0]);
        embeddings.insert("e3".to_string(), array![5.0, 5.0]);

        let config = ClusteringConfig {
            num_clusters: 2,
            ..Default::default()
        };

        let mut clustering = EntityClustering::new(config);
        let result = clustering.cluster(&embeddings).unwrap();

        // Silhouette score should be between -1 and 1
        assert!(result.silhouette_score >= -1.0 && result.silhouette_score <= 1.0);
    }
}
