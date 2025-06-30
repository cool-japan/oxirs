//! Advanced clustering algorithms for vector similarity and resource grouping
//!
//! This module provides various clustering algorithms for the SPARQL vec:cluster function:
//! - K-means clustering
//! - DBSCAN (Density-Based Spatial Clustering)
//! - Hierarchical clustering (Agglomerative)
//! - Spectral clustering
//! - Community detection for graph clustering

use crate::{similarity::SimilarityMetric, Vector};
use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Clustering algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClusteringAlgorithm {
    /// K-means clustering
    KMeans,
    /// DBSCAN density-based clustering
    DBSCAN,
    /// Hierarchical agglomerative clustering
    Hierarchical,
    /// Spectral clustering
    Spectral,
    /// Community detection (for graph-based clustering)
    Community,
    /// Threshold-based similarity clustering
    Similarity,
}

/// Clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    /// Clustering algorithm to use
    pub algorithm: ClusteringAlgorithm,
    /// Number of clusters (for k-means, spectral)
    pub num_clusters: Option<usize>,
    /// Similarity threshold (for DBSCAN, similarity clustering)
    pub similarity_threshold: f32,
    /// Minimum cluster size (for DBSCAN)
    pub min_cluster_size: usize,
    /// Distance metric to use
    pub distance_metric: SimilarityMetric,
    /// Maximum iterations (for iterative algorithms)
    pub max_iterations: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Convergence tolerance
    pub tolerance: f32,
    /// Linkage criterion for hierarchical clustering
    pub linkage: LinkageCriterion,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            algorithm: ClusteringAlgorithm::KMeans,
            num_clusters: Some(3),
            similarity_threshold: 0.7,
            min_cluster_size: 3,
            distance_metric: SimilarityMetric::Cosine,
            max_iterations: 100,
            random_seed: None,
            tolerance: 1e-4,
            linkage: LinkageCriterion::Average,
        }
    }
}

/// Linkage criteria for hierarchical clustering
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LinkageCriterion {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage (average distance)
    Average,
    /// Ward linkage (minimize within-cluster variance)
    Ward,
}

/// Cluster result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    /// Cluster ID
    pub id: usize,
    /// Resource IDs in this cluster
    pub members: Vec<String>,
    /// Cluster centroid (if applicable)
    pub centroid: Option<Vector>,
    /// Cluster statistics
    pub stats: ClusterStats,
}

/// Cluster statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    /// Number of members
    pub size: usize,
    /// Average intra-cluster similarity
    pub avg_intra_similarity: f32,
    /// Cluster density (for DBSCAN)
    pub density: f32,
    /// Silhouette score for this cluster
    pub silhouette_score: f32,
}

/// Clustering result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    /// All clusters found
    pub clusters: Vec<Cluster>,
    /// Noise points (for DBSCAN)
    pub noise: Vec<String>,
    /// Overall clustering quality metrics
    pub quality_metrics: ClusteringQualityMetrics,
    /// Algorithm used
    pub algorithm: ClusteringAlgorithm,
    /// Configuration used
    pub config: ClusteringConfig,
}

/// Quality metrics for clustering results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringQualityMetrics {
    /// Silhouette score (-1 to 1, higher is better)
    pub silhouette_score: f32,
    /// Davies-Bouldin index (lower is better)
    pub davies_bouldin_index: f32,
    /// Calinski-Harabasz index (higher is better)
    pub calinski_harabasz_index: f32,
    /// Within-cluster sum of squares
    pub within_cluster_ss: f32,
    /// Between-cluster sum of squares
    pub between_cluster_ss: f32,
}

/// Main clustering engine
pub struct ClusteringEngine {
    config: ClusteringConfig,
}

impl ClusteringEngine {
    pub fn new(config: ClusteringConfig) -> Self {
        Self { config }
    }

    /// Cluster a set of resources with their embeddings
    pub fn cluster(&self, resources: &[(String, Vector)]) -> Result<ClusteringResult> {
        if resources.is_empty() {
            return Ok(ClusteringResult {
                clusters: Vec::new(),
                noise: Vec::new(),
                quality_metrics: ClusteringQualityMetrics::default(),
                algorithm: self.config.algorithm,
                config: self.config.clone(),
            });
        }

        match self.config.algorithm {
            ClusteringAlgorithm::KMeans => self.kmeans_clustering(resources),
            ClusteringAlgorithm::DBSCAN => self.dbscan_clustering(resources),
            ClusteringAlgorithm::Hierarchical => self.hierarchical_clustering(resources),
            ClusteringAlgorithm::Spectral => self.spectral_clustering(resources),
            ClusteringAlgorithm::Community => self.community_detection(resources),
            ClusteringAlgorithm::Similarity => self.similarity_clustering(resources),
        }
    }

    /// K-means clustering implementation
    fn kmeans_clustering(&self, resources: &[(String, Vector)]) -> Result<ClusteringResult> {
        let k = self.config.num_clusters.unwrap_or(3);
        if k >= resources.len() {
            return Err(anyhow!(
                "Number of clusters must be less than number of resources"
            ));
        }

        let mut rng = if let Some(seed) = self.config.random_seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_seed(rand::random())
        };

        // Initialize centroids using k-means++
        let mut centroids = self.initialize_centroids_kmeans_plus_plus(resources, k, &mut rng)?;
        let mut assignments = vec![0; resources.len()];
        let mut prev_assignments = vec![usize::MAX; resources.len()];

        for iteration in 0..self.config.max_iterations {
            // Assign points to closest centroids
            for (i, (_, vector)) in resources.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_distance = f32::INFINITY;

                for (cluster_id, centroid) in centroids.iter().enumerate() {
                    let distance = self.calculate_distance(vector, centroid)?;
                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = cluster_id;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Check for convergence
            if assignments == prev_assignments {
                break;
            }

            // Update centroids
            for cluster_id in 0..k {
                let cluster_vectors: Vec<&Vector> = resources
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == cluster_id)
                    .map(|(_, (_, vector))| vector)
                    .collect();

                if !cluster_vectors.is_empty() {
                    centroids[cluster_id] = self.compute_centroid(&cluster_vectors)?;
                }
            }

            prev_assignments = assignments.clone();

            if iteration > 0 && iteration % 10 == 0 {
                println!(
                    "K-means iteration {}/{}",
                    iteration, self.config.max_iterations
                );
            }
        }

        // Build clusters from assignments
        let mut clusters = Vec::new();
        for cluster_id in 0..k {
            let members: Vec<String> = resources
                .iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == cluster_id)
                .map(|(_, (resource_id, _))| resource_id.clone())
                .collect();

            if !members.is_empty() {
                let cluster_vectors: Vec<&Vector> = resources
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == cluster_id)
                    .map(|(_, (_, vector))| vector)
                    .collect();

                let stats = self.compute_cluster_stats(&cluster_vectors)?;

                clusters.push(Cluster {
                    id: cluster_id,
                    members,
                    centroid: Some(centroids[cluster_id].clone()),
                    stats,
                });
            }
        }

        let quality_metrics = self.compute_quality_metrics(resources, &clusters)?;

        Ok(ClusteringResult {
            clusters,
            noise: Vec::new(),
            quality_metrics,
            algorithm: ClusteringAlgorithm::KMeans,
            config: self.config.clone(),
        })
    }

    /// DBSCAN clustering implementation
    fn dbscan_clustering(&self, resources: &[(String, Vector)]) -> Result<ClusteringResult> {
        let eps = 1.0 - self.config.similarity_threshold; // Convert similarity to distance
        let min_pts = self.config.min_cluster_size;

        let mut visited = vec![false; resources.len()];
        let mut cluster_assignments = vec![None; resources.len()];
        let mut cluster_id = 0;
        let mut noise_points = Vec::new();

        for i in 0..resources.len() {
            if visited[i] {
                continue;
            }
            visited[i] = true;

            let neighbors = self.find_neighbors(resources, i, eps)?;

            if neighbors.len() < min_pts {
                noise_points.push(resources[i].0.clone());
            } else {
                let mut cluster_queue = VecDeque::new();
                cluster_queue.push_back(i);
                cluster_assignments[i] = Some(cluster_id);

                while let Some(point_idx) = cluster_queue.pop_front() {
                    let point_neighbors = self.find_neighbors(resources, point_idx, eps)?;

                    if point_neighbors.len() >= min_pts {
                        for &neighbor_idx in &point_neighbors {
                            if !visited[neighbor_idx] {
                                visited[neighbor_idx] = true;
                                cluster_queue.push_back(neighbor_idx);
                            }
                            if cluster_assignments[neighbor_idx].is_none() {
                                cluster_assignments[neighbor_idx] = Some(cluster_id);
                            }
                        }
                    }
                }
                cluster_id += 1;
            }
        }

        // Build clusters from assignments
        let mut clusters = Vec::new();
        for cid in 0..cluster_id {
            let members: Vec<String> = resources
                .iter()
                .enumerate()
                .filter(|(i, _)| cluster_assignments[*i] == Some(cid))
                .map(|(_, (resource_id, _))| resource_id.clone())
                .collect();

            if !members.is_empty() {
                let cluster_vectors: Vec<&Vector> = resources
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| cluster_assignments[*i] == Some(cid))
                    .map(|(_, (_, vector))| vector)
                    .collect();

                let stats = self.compute_cluster_stats(&cluster_vectors)?;
                let centroid = if !cluster_vectors.is_empty() {
                    Some(self.compute_centroid(&cluster_vectors)?)
                } else {
                    None
                };

                clusters.push(Cluster {
                    id: cid,
                    members,
                    centroid,
                    stats,
                });
            }
        }

        let quality_metrics = self.compute_quality_metrics(resources, &clusters)?;

        Ok(ClusteringResult {
            clusters,
            noise: noise_points,
            quality_metrics,
            algorithm: ClusteringAlgorithm::DBSCAN,
            config: self.config.clone(),
        })
    }

    /// Hierarchical clustering implementation (agglomerative)
    fn hierarchical_clustering(&self, resources: &[(String, Vector)]) -> Result<ClusteringResult> {
        let target_clusters = self.config.num_clusters.unwrap_or(3);

        // Initialize each point as its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..resources.len()).map(|i| vec![i]).collect();

        // Compute initial distance matrix
        let mut distance_matrix = self.compute_distance_matrix(resources)?;

        // Merge clusters until we reach the target number
        while clusters.len() > target_clusters {
            let (min_i, min_j) = self.find_closest_clusters(&clusters, &distance_matrix)?;

            // Merge clusters
            let cluster_j = clusters.remove(min_j.max(min_i));
            clusters[min_i.min(min_j)].extend(cluster_j);

            // Update distance matrix
            self.update_distance_matrix(
                &mut distance_matrix,
                &clusters,
                min_i.min(min_j),
                resources,
            )?;
        }

        // Build final cluster results
        let mut result_clusters = Vec::new();
        for (cluster_id, cluster_indices) in clusters.iter().enumerate() {
            let members: Vec<String> = cluster_indices
                .iter()
                .map(|&idx| resources[idx].0.clone())
                .collect();

            let cluster_vectors: Vec<&Vector> = cluster_indices
                .iter()
                .map(|&idx| &resources[idx].1)
                .collect();

            let stats = self.compute_cluster_stats(&cluster_vectors)?;
            let centroid = if !cluster_vectors.is_empty() {
                Some(self.compute_centroid(&cluster_vectors)?)
            } else {
                None
            };

            result_clusters.push(Cluster {
                id: cluster_id,
                members,
                centroid,
                stats,
            });
        }

        let quality_metrics = self.compute_quality_metrics(resources, &result_clusters)?;

        Ok(ClusteringResult {
            clusters: result_clusters,
            noise: Vec::new(),
            quality_metrics,
            algorithm: ClusteringAlgorithm::Hierarchical,
            config: self.config.clone(),
        })
    }

    /// Placeholder for spectral clustering
    fn spectral_clustering(&self, resources: &[(String, Vector)]) -> Result<ClusteringResult> {
        // For now, fall back to k-means
        // TODO: Implement proper spectral clustering with eigenvalue decomposition
        println!("Spectral clustering not yet fully implemented, falling back to k-means");
        self.kmeans_clustering(resources)
    }

    /// Placeholder for community detection
    fn community_detection(&self, resources: &[(String, Vector)]) -> Result<ClusteringResult> {
        // For now, fall back to similarity clustering
        // TODO: Implement graph-based community detection algorithms like Louvain
        println!(
            "Community detection not yet fully implemented, falling back to similarity clustering"
        );
        self.similarity_clustering(resources)
    }

    /// Simple similarity-based clustering
    fn similarity_clustering(&self, resources: &[(String, Vector)]) -> Result<ClusteringResult> {
        let threshold = self.config.similarity_threshold;
        let mut clusters = Vec::new();
        let mut assigned = vec![false; resources.len()];
        let mut cluster_id = 0;

        for i in 0..resources.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster_members = vec![i];
            assigned[i] = true;

            // Find all similar vectors
            for j in (i + 1)..resources.len() {
                if assigned[j] {
                    continue;
                }

                let similarity = self.calculate_similarity(&resources[i].1, &resources[j].1)?;
                if similarity >= threshold {
                    cluster_members.push(j);
                    assigned[j] = true;
                }
            }

            let members: Vec<String> = cluster_members
                .iter()
                .map(|&idx| resources[idx].0.clone())
                .collect();

            let cluster_vectors: Vec<&Vector> = cluster_members
                .iter()
                .map(|&idx| &resources[idx].1)
                .collect();

            let stats = self.compute_cluster_stats(&cluster_vectors)?;
            let centroid = if !cluster_vectors.is_empty() {
                Some(self.compute_centroid(&cluster_vectors)?)
            } else {
                None
            };

            clusters.push(Cluster {
                id: cluster_id,
                members,
                centroid,
                stats,
            });

            cluster_id += 1;
        }

        let quality_metrics = self.compute_quality_metrics(resources, &clusters)?;

        Ok(ClusteringResult {
            clusters,
            noise: Vec::new(),
            quality_metrics,
            algorithm: ClusteringAlgorithm::Similarity,
            config: self.config.clone(),
        })
    }

    // Helper methods

    /// Initialize centroids using k-means++
    fn initialize_centroids_kmeans_plus_plus(
        &self,
        resources: &[(String, Vector)],
        k: usize,
        rng: &mut impl Rng,
    ) -> Result<Vec<Vector>> {
        let mut centroids = Vec::new();

        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..resources.len());
        centroids.push(resources[first_idx].1.clone());

        // Choose remaining centroids with probability proportional to squared distance
        for _ in 1..k {
            let mut distances = Vec::new();
            let mut total_distance = 0.0;

            for (_, vector) in resources {
                let min_dist_sq = centroids
                    .iter()
                    .map(|centroid| {
                        self.calculate_distance(vector, centroid)
                            .unwrap_or(f32::INFINITY)
                    })
                    .fold(f32::INFINITY, f32::min)
                    .powi(2);
                distances.push(min_dist_sq);
                total_distance += min_dist_sq;
            }

            let target = rng.gen::<f32>() * total_distance;
            let mut cumulative = 0.0;

            for (i, &dist) in distances.iter().enumerate() {
                cumulative += dist;
                if cumulative >= target {
                    centroids.push(resources[i].1.clone());
                    break;
                }
            }
        }

        Ok(centroids)
    }

    /// Calculate distance between two vectors
    fn calculate_distance(&self, v1: &Vector, v2: &Vector) -> Result<f32> {
        match self.config.distance_metric {
            SimilarityMetric::Cosine => Ok(1.0 - v1.cosine_similarity(v2)?),
            SimilarityMetric::Euclidean => v1.euclidean_distance(v2),
            SimilarityMetric::Manhattan => v1.manhattan_distance(v2),
            _ => Ok(1.0 - v1.cosine_similarity(v2)?), // Default to cosine
        }
    }

    /// Calculate similarity between two vectors
    fn calculate_similarity(&self, v1: &Vector, v2: &Vector) -> Result<f32> {
        match self.config.distance_metric {
            SimilarityMetric::Cosine => v1.cosine_similarity(v2),
            SimilarityMetric::Euclidean => {
                let dist = v1.euclidean_distance(v2)?;
                Ok(1.0 / (1.0 + dist))
            }
            SimilarityMetric::Manhattan => {
                let dist = v1.manhattan_distance(v2)?;
                Ok(1.0 / (1.0 + dist))
            }
            _ => v1.cosine_similarity(v2), // Default to cosine
        }
    }

    /// Find neighbors within distance eps
    fn find_neighbors(
        &self,
        resources: &[(String, Vector)],
        point_idx: usize,
        eps: f32,
    ) -> Result<Vec<usize>> {
        let mut neighbors = Vec::new();
        let point = &resources[point_idx].1;

        for (i, (_, vector)) in resources.iter().enumerate() {
            if i != point_idx {
                let distance = self.calculate_distance(point, vector)?;
                if distance <= eps {
                    neighbors.push(i);
                }
            }
        }

        Ok(neighbors)
    }

    /// Compute centroid of vectors
    fn compute_centroid(&self, vectors: &[&Vector]) -> Result<Vector> {
        if vectors.is_empty() {
            return Err(anyhow!("Cannot compute centroid of empty vector set"));
        }

        let dim = vectors[0].dimensions;
        let mut centroid_data = vec![0.0; dim];

        for vector in vectors {
            let data = vector.as_f32();
            for (i, &value) in data.iter().enumerate() {
                centroid_data[i] += value;
            }
        }

        let count = vectors.len() as f32;
        for value in &mut centroid_data {
            *value /= count;
        }

        Ok(Vector::new(centroid_data))
    }

    /// Compute cluster statistics
    fn compute_cluster_stats(&self, vectors: &[&Vector]) -> Result<ClusterStats> {
        if vectors.is_empty() {
            return Ok(ClusterStats {
                size: 0,
                avg_intra_similarity: 0.0,
                density: 0.0,
                silhouette_score: 0.0,
            });
        }

        let size = vectors.len();
        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        // Calculate average intra-cluster similarity
        for i in 0..vectors.len() {
            for j in (i + 1)..vectors.len() {
                let similarity = self.calculate_similarity(vectors[i], vectors[j])?;
                total_similarity += similarity;
                pair_count += 1;
            }
        }

        let avg_intra_similarity = if pair_count > 0 {
            total_similarity / pair_count as f32
        } else {
            1.0 // Single point cluster
        };

        Ok(ClusterStats {
            size,
            avg_intra_similarity,
            density: avg_intra_similarity, // Simplified density measure
            silhouette_score: 0.0,         // TODO: Implement proper silhouette calculation
        })
    }

    /// Compute distance matrix for hierarchical clustering
    fn compute_distance_matrix(&self, resources: &[(String, Vector)]) -> Result<Vec<Vec<f32>>> {
        let n = resources.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let distance = self.calculate_distance(&resources[i].1, &resources[j].1)?;
                matrix[i][j] = distance;
                matrix[j][i] = distance;
            }
        }

        Ok(matrix)
    }

    /// Find closest clusters for hierarchical clustering
    fn find_closest_clusters(
        &self,
        clusters: &[Vec<usize>],
        distance_matrix: &[Vec<f32>],
    ) -> Result<(usize, usize)> {
        let mut min_distance = f32::INFINITY;
        let mut closest_pair = (0, 1);

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let distance = self.cluster_distance(&clusters[i], &clusters[j], distance_matrix);
                if distance < min_distance {
                    min_distance = distance;
                    closest_pair = (i, j);
                }
            }
        }

        Ok(closest_pair)
    }

    /// Calculate distance between clusters based on linkage criterion
    fn cluster_distance(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        distance_matrix: &[Vec<f32>],
    ) -> f32 {
        match self.config.linkage {
            LinkageCriterion::Single => {
                // Minimum distance
                cluster1
                    .iter()
                    .flat_map(|&i| cluster2.iter().map(move |&j| distance_matrix[i][j]))
                    .fold(f32::INFINITY, f32::min)
            }
            LinkageCriterion::Complete => {
                // Maximum distance
                cluster1
                    .iter()
                    .flat_map(|&i| cluster2.iter().map(move |&j| distance_matrix[i][j]))
                    .fold(0.0, f32::max)
            }
            LinkageCriterion::Average => {
                // Average distance
                let mut total = 0.0;
                let mut count = 0;
                for &i in cluster1 {
                    for &j in cluster2 {
                        total += distance_matrix[i][j];
                        count += 1;
                    }
                }
                if count > 0 {
                    total / count as f32
                } else {
                    0.0
                }
            }
            LinkageCriterion::Ward => {
                // Simplified Ward linkage (should consider cluster variance)
                self.cluster_distance(cluster1, cluster2, distance_matrix)
            }
        }
    }

    /// Update distance matrix after merging clusters
    fn update_distance_matrix(
        &self,
        distance_matrix: &mut Vec<Vec<f32>>,
        clusters: &[Vec<usize>],
        merged_cluster: usize,
        resources: &[(String, Vector)],
    ) -> Result<()> {
        // Simplified update - could be more efficient
        let new_matrix = self.compute_distance_matrix(resources)?;
        *distance_matrix = new_matrix;
        Ok(())
    }

    /// Compute clustering quality metrics
    fn compute_quality_metrics(
        &self,
        resources: &[(String, Vector)],
        clusters: &[Cluster],
    ) -> Result<ClusteringQualityMetrics> {
        // Simplified quality metrics - in practice these would be more sophisticated
        let mut within_cluster_ss = 0.0;
        let mut silhouette_scores = Vec::new();

        for cluster in clusters {
            if cluster.members.len() > 1 {
                let cluster_vectors: Vec<&Vector> = cluster
                    .members
                    .iter()
                    .filter_map(|member| {
                        resources
                            .iter()
                            .find(|(id, _)| id == member)
                            .map(|(_, v)| v)
                    })
                    .collect();

                if let Some(ref centroid) = cluster.centroid {
                    for vector in &cluster_vectors {
                        let dist = self.calculate_distance(vector, centroid)?;
                        within_cluster_ss += dist * dist;
                    }
                }
            }
        }

        let silhouette_score = if !silhouette_scores.is_empty() {
            silhouette_scores.iter().sum::<f32>() / silhouette_scores.len() as f32
        } else {
            0.0
        };

        Ok(ClusteringQualityMetrics {
            silhouette_score,
            davies_bouldin_index: 0.0,    // TODO: Implement
            calinski_harabasz_index: 0.0, // TODO: Implement
            within_cluster_ss,
            between_cluster_ss: 0.0, // TODO: Implement
        })
    }
}

impl Default for ClusteringQualityMetrics {
    fn default() -> Self {
        Self {
            silhouette_score: 0.0,
            davies_bouldin_index: 0.0,
            calinski_harabasz_index: 0.0,
            within_cluster_ss: 0.0,
            between_cluster_ss: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_clustering() {
        let config = ClusteringConfig {
            algorithm: ClusteringAlgorithm::KMeans,
            num_clusters: Some(2),
            random_seed: Some(42),
            distance_metric: SimilarityMetric::Euclidean, // Use Euclidean for proper distance calculation
            ..Default::default()
        };

        let engine = ClusteringEngine::new(config);

        let resources = vec![
            ("res1".to_string(), Vector::new(vec![1.0, 1.0, 1.0])),
            ("res2".to_string(), Vector::new(vec![1.1, 1.1, 1.1])),
            ("res3".to_string(), Vector::new(vec![10.0, 10.0, 10.0])),
            ("res4".to_string(), Vector::new(vec![10.1, 10.1, 10.1])),
        ];

        let result = engine.cluster(&resources).unwrap();

        assert_eq!(result.clusters.len(), 2);
        assert!(result.noise.is_empty());
    }

    #[test]
    fn test_dbscan_clustering() {
        let config = ClusteringConfig {
            algorithm: ClusteringAlgorithm::DBSCAN,
            similarity_threshold: 0.9,
            min_cluster_size: 2,
            ..Default::default()
        };

        let engine = ClusteringEngine::new(config);

        let resources = vec![
            ("res1".to_string(), Vector::new(vec![1.0, 1.0, 1.0])),
            ("res2".to_string(), Vector::new(vec![1.1, 1.1, 1.1])),
            ("res3".to_string(), Vector::new(vec![10.0, 10.0, 10.0])),
        ];

        let result = engine.cluster(&resources).unwrap();
        assert!(result.clusters.len() <= 2);
    }

    #[test]
    fn test_similarity_clustering() {
        let config = ClusteringConfig {
            algorithm: ClusteringAlgorithm::Similarity,
            similarity_threshold: 0.95,
            ..Default::default()
        };

        let engine = ClusteringEngine::new(config);

        let resources = vec![
            ("res1".to_string(), Vector::new(vec![1.0, 0.0, 0.0])),
            ("res2".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
            ("res3".to_string(), Vector::new(vec![0.0, 0.0, 1.0])),
        ];

        let result = engine.cluster(&resources).unwrap();
        // Should have 3 clusters since vectors are orthogonal
        assert_eq!(result.clusters.len(), 3);
    }
}
