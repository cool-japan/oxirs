//! Embedding Visualization Tools
//!
//! This module provides tools for visualizing knowledge graph embeddings in 2D/3D space
//! using dimensionality reduction techniques like t-SNE, UMAP, and PCA.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::prelude::{Normal, Random};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Dimensionality reduction method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionMethod {
    /// Principal Component Analysis
    PCA,
    /// t-Distributed Stochastic Neighbor Embedding
    TSNE,
    /// Uniform Manifold Approximation and Projection
    UMAP,
    /// Random Projection (fast but less accurate)
    RandomProjection,
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Dimensionality reduction method
    pub method: ReductionMethod,
    /// Target dimensions (2 or 3)
    pub target_dims: usize,
    /// Perplexity for t-SNE (typically 5-50)
    pub tsne_perplexity: f32,
    /// Learning rate for t-SNE
    pub tsne_learning_rate: f32,
    /// Number of iterations for iterative methods
    pub max_iterations: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Number of neighbors for UMAP
    pub umap_n_neighbors: usize,
    /// Minimum distance for UMAP
    pub umap_min_dist: f32,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            method: ReductionMethod::PCA,
            target_dims: 2,
            tsne_perplexity: 30.0,
            tsne_learning_rate: 200.0,
            max_iterations: 1000,
            random_seed: None,
            umap_n_neighbors: 15,
            umap_min_dist: 0.1,
        }
    }
}

/// Visualization result with 2D/3D coordinates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationResult {
    /// Entity ID to coordinates mapping
    pub coordinates: HashMap<String, Vec<f32>>,
    /// Number of dimensions
    pub dimensions: usize,
    /// Method used
    pub method: ReductionMethod,
    /// Explained variance (for PCA)
    pub explained_variance: Option<Vec<f32>>,
    /// Final stress/loss (for t-SNE)
    pub final_loss: Option<f32>,
}

/// Embedding visualizer
pub struct EmbeddingVisualizer {
    config: VisualizationConfig,
    rng: Random,
}

impl EmbeddingVisualizer {
    /// Create new embedding visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        let rng = Random::default();

        info!(
            "Initialized embedding visualizer: method={:?}, target_dims={}",
            config.method, config.target_dims
        );

        Self { config, rng }
    }

    /// Visualize embeddings
    pub fn visualize(
        &mut self,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<VisualizationResult> {
        if embeddings.is_empty() {
            return Err(anyhow!("No embeddings to visualize"));
        }

        if self.config.target_dims != 2 && self.config.target_dims != 3 {
            return Err(anyhow!("Target dimensions must be 2 or 3"));
        }

        info!("Visualizing {} embeddings", embeddings.len());

        match self.config.method {
            ReductionMethod::PCA => self.pca(embeddings),
            ReductionMethod::TSNE => self.tsne(embeddings),
            ReductionMethod::UMAP => self.umap_approximate(embeddings),
            ReductionMethod::RandomProjection => self.random_projection(embeddings),
        }
    }

    /// PCA dimensionality reduction
    fn pca(&mut self, embeddings: &HashMap<String, Array1<f32>>) -> Result<VisualizationResult> {
        let entity_list: Vec<String> = embeddings.keys().cloned().collect();
        let n = entity_list.len();
        let d = embeddings
            .values()
            .next()
            .expect("embeddings should not be empty")
            .len();

        // Build data matrix (n x d)
        let mut data_matrix = Array2::zeros((n, d));
        for (i, entity) in entity_list.iter().enumerate() {
            let emb = &embeddings[entity];
            for j in 0..d {
                data_matrix[[i, j]] = emb[j];
            }
        }

        // Center the data
        let mean = self.compute_mean(&data_matrix);
        for i in 0..n {
            for j in 0..d {
                data_matrix[[i, j]] -= mean[j];
            }
        }

        // Compute covariance matrix (d x d)
        let cov_matrix = self.compute_covariance(&data_matrix);

        // Find principal components using power iteration
        let (eigenvectors, eigenvalues) =
            self.power_iteration_top_k(&cov_matrix, self.config.target_dims)?;

        // Project data onto principal components
        let mut coordinates = HashMap::new();
        for (i, entity) in entity_list.iter().enumerate() {
            let mut projected = vec![0.0; self.config.target_dims];
            for k in 0..self.config.target_dims {
                let mut dot_product = 0.0;
                for j in 0..d {
                    dot_product += data_matrix[[i, j]] * eigenvectors[[j, k]];
                }
                projected[k] = dot_product;
            }
            coordinates.insert(entity.clone(), projected);
        }

        // Compute explained variance
        let total_variance: f32 = eigenvalues.iter().sum();
        let explained_variance: Vec<f32> =
            eigenvalues.iter().map(|&ev| ev / total_variance).collect();

        info!(
            "PCA complete: explained variance = {:?}",
            explained_variance
        );

        Ok(VisualizationResult {
            coordinates,
            dimensions: self.config.target_dims,
            method: ReductionMethod::PCA,
            explained_variance: Some(explained_variance),
            final_loss: None,
        })
    }

    /// t-SNE dimensionality reduction (simplified implementation)
    fn tsne(&mut self, embeddings: &HashMap<String, Array1<f32>>) -> Result<VisualizationResult> {
        let entity_list: Vec<String> = embeddings.keys().cloned().collect();
        let n = entity_list.len();

        // Initialize low-dimensional representation randomly
        let dist = Normal::new(0.0, 0.01)
            .expect("Normal distribution should be valid for these parameters");
        let mut y = Array2::from_shape_fn((n, self.config.target_dims), |_| self.rng.sample(dist));

        // Compute pairwise affinities in high-dimensional space
        let p = self.compute_affinities(embeddings, &entity_list);

        // Gradient descent
        let mut final_loss = 0.0;
        for iteration in 0..self.config.max_iterations {
            // Compute pairwise affinities in low-dimensional space
            let q = self.compute_low_dim_affinities(&y);

            // Compute gradient
            let grad = self.compute_tsne_gradient(&y, &p, &q);

            // Update positions
            for i in 0..n {
                for j in 0..self.config.target_dims {
                    y[[i, j]] -= self.config.tsne_learning_rate * grad[[i, j]];
                }
            }

            // Compute KL divergence (loss)
            if iteration % 100 == 0 {
                final_loss = self.compute_kl_divergence(&p, &q);
                debug!("t-SNE iteration {}: loss = {:.6}", iteration, final_loss);
            }
        }

        // Extract coordinates
        let mut coordinates = HashMap::new();
        for (i, entity) in entity_list.iter().enumerate() {
            let mut coords = vec![0.0; self.config.target_dims];
            for j in 0..self.config.target_dims {
                coords[j] = y[[i, j]];
            }
            coordinates.insert(entity.clone(), coords);
        }

        info!("t-SNE complete: final loss = {:.6}", final_loss);

        Ok(VisualizationResult {
            coordinates,
            dimensions: self.config.target_dims,
            method: ReductionMethod::TSNE,
            explained_variance: None,
            final_loss: Some(final_loss),
        })
    }

    /// UMAP (simplified/approximate implementation)
    fn umap_approximate(
        &mut self,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<VisualizationResult> {
        // For a full UMAP implementation, we'd need complex graph construction and optimization
        // This is a simplified approximation using PCA followed by force-directed layout

        info!("Using approximate UMAP (PCA + refinement)");

        // Start with PCA
        let mut result = self.pca(embeddings)?;

        // Apply force-directed refinement
        let entity_list: Vec<String> = embeddings.keys().cloned().collect();
        let n = entity_list.len();

        // Build k-nearest neighbor graph
        let knn_graph =
            self.build_knn_graph(embeddings, &entity_list, self.config.umap_n_neighbors);

        // Refine positions using force-directed layout
        for _iteration in 0..100 {
            for i in 0..n {
                let entity = &entity_list[i];
                let pos = &result.coordinates[entity].clone();

                // Attractive forces from neighbors
                let mut force = vec![0.0; self.config.target_dims];
                for &neighbor_idx in &knn_graph[i] {
                    let neighbor = &entity_list[neighbor_idx];
                    let neighbor_pos = &result.coordinates[neighbor];

                    for d in 0..self.config.target_dims {
                        let diff = neighbor_pos[d] - pos[d];
                        force[d] += diff * 0.01; // Attraction
                    }
                }

                // Update position
                let coords = result
                    .coordinates
                    .get_mut(entity)
                    .expect("entity should exist in coordinates");
                for d in 0..self.config.target_dims {
                    coords[d] += force[d];
                }
            }
        }

        result.method = ReductionMethod::UMAP;
        info!("Approximate UMAP complete");

        Ok(result)
    }

    /// Random projection (fast dimensionality reduction)
    fn random_projection(
        &mut self,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<VisualizationResult> {
        let entity_list: Vec<String> = embeddings.keys().cloned().collect();
        let d = embeddings
            .values()
            .next()
            .expect("embeddings should not be empty")
            .len();

        // Generate random projection matrix
        let dist = Normal::new(0.0, 1.0)
            .expect("Normal distribution should be valid for these parameters");
        let projection_matrix =
            Array2::from_shape_fn((d, self.config.target_dims), |_| self.rng.sample(dist));

        // Project each embedding
        let mut coordinates = HashMap::new();
        for entity in &entity_list {
            let emb = &embeddings[entity];
            let mut projected = vec![0.0; self.config.target_dims];

            for k in 0..self.config.target_dims {
                let mut dot_product = 0.0;
                for j in 0..d {
                    dot_product += emb[j] * projection_matrix[[j, k]];
                }
                projected[k] = dot_product;
            }

            coordinates.insert(entity.clone(), projected);
        }

        info!("Random projection complete");

        Ok(VisualizationResult {
            coordinates,
            dimensions: self.config.target_dims,
            method: ReductionMethod::RandomProjection,
            explained_variance: None,
            final_loss: None,
        })
    }

    /// Compute mean of each dimension
    fn compute_mean(&self, data: &Array2<f32>) -> Vec<f32> {
        let n = data.nrows();
        let d = data.ncols();
        let mut mean = vec![0.0; d];

        for j in 0..d {
            for i in 0..n {
                mean[j] += data[[i, j]];
            }
            mean[j] /= n as f32;
        }

        mean
    }

    /// Compute covariance matrix
    fn compute_covariance(&self, data: &Array2<f32>) -> Array2<f32> {
        let n = data.nrows() as f32;
        let d = data.ncols();
        let mut cov = Array2::zeros((d, d));

        for i in 0..d {
            for j in 0..d {
                let mut sum = 0.0;
                for k in 0..data.nrows() {
                    sum += data[[k, i]] * data[[k, j]];
                }
                cov[[i, j]] = sum / (n - 1.0);
            }
        }

        cov
    }

    /// Power iteration to find top-k eigenvectors
    fn power_iteration_top_k(
        &mut self,
        matrix: &Array2<f32>,
        k: usize,
    ) -> Result<(Array2<f32>, Vec<f32>)> {
        let d = matrix.nrows();
        let mut eigenvectors = Array2::zeros((d, k));
        let mut eigenvalues = Vec::new();

        let mut working_matrix = matrix.clone();

        for component in 0..k {
            // Initialize random vector
            let dist = Normal::new(0.0f32, 1.0f32)
                .expect("Normal distribution should be valid for these parameters");
            let mut v = Array1::from_shape_fn(d, |_| self.rng.sample(dist));

            // Power iteration
            for _ in 0..100 {
                // Multiply by matrix
                let mut new_v = Array1::<f32>::zeros(d);
                for i in 0..d {
                    for j in 0..d {
                        new_v[i] += working_matrix[[i, j]] * v[j];
                    }
                }

                // Normalize
                let norm = new_v.dot(&new_v).sqrt();
                if norm > 0.0 {
                    v = new_v / norm;
                }
            }

            // Compute eigenvalue
            let mut av = Array1::<f32>::zeros(d);
            for i in 0..d {
                for j in 0..d {
                    av[i] += working_matrix[[i, j]] * v[j];
                }
            }
            let eigenvalue = v.dot(&av);
            eigenvalues.push(eigenvalue);

            // Store eigenvector
            for i in 0..d {
                eigenvectors[[i, component]] = v[i];
            }

            // Deflate matrix for next component
            for i in 0..d {
                for j in 0..d {
                    working_matrix[[i, j]] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        Ok((eigenvectors, eigenvalues))
    }

    /// Compute affinities for t-SNE
    fn compute_affinities(
        &self,
        embeddings: &HashMap<String, Array1<f32>>,
        entity_list: &[String],
    ) -> Array2<f32> {
        let n = entity_list.len();
        let mut p = Array2::zeros((n, n));

        // Compute pairwise distances
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dist = self.euclidean_distance(
                        &embeddings[&entity_list[i]],
                        &embeddings[&entity_list[j]],
                    );
                    // Gaussian kernel
                    p[[i, j]] = (-dist * dist / (2.0 * self.config.tsne_perplexity)).exp();
                }
            }

            // Normalize row
            let row_sum: f32 = (0..n).map(|j| p[[i, j]]).sum();
            if row_sum > 0.0 {
                for j in 0..n {
                    p[[i, j]] /= row_sum;
                }
            }
        }

        // Symmetrize
        for i in 0..n {
            for j in 0..n {
                p[[i, j]] = (p[[i, j]] + p[[j, i]]) / (2.0 * n as f32);
            }
        }

        p
    }

    /// Compute low-dimensional affinities
    fn compute_low_dim_affinities(&self, y: &Array2<f32>) -> Array2<f32> {
        let n = y.nrows();
        let mut q = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let mut dist_sq = 0.0;
                    for k in 0..y.ncols() {
                        let diff = y[[i, k]] - y[[j, k]];
                        dist_sq += diff * diff;
                    }
                    q[[i, j]] = 1.0 / (1.0 + dist_sq);
                }
            }
        }

        // Normalize
        let sum: f32 = q.iter().sum();
        if sum > 0.0 {
            q /= sum;
        }

        q
    }

    /// Compute t-SNE gradient
    fn compute_tsne_gradient(
        &self,
        y: &Array2<f32>,
        p: &Array2<f32>,
        q: &Array2<f32>,
    ) -> Array2<f32> {
        let n = y.nrows();
        let d = y.ncols();
        let mut grad = Array2::zeros((n, d));

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let pq_diff = p[[i, j]] - q[[i, j]];
                    let q_val = q[[i, j]];

                    for k in 0..d {
                        let y_diff = y[[i, k]] - y[[j, k]];
                        grad[[i, k]] += 4.0 * pq_diff * y_diff * q_val;
                    }
                }
            }
        }

        grad
    }

    /// Compute KL divergence
    fn compute_kl_divergence(&self, p: &Array2<f32>, q: &Array2<f32>) -> f32 {
        let mut kl = 0.0;
        for i in 0..p.nrows() {
            for j in 0..p.ncols() {
                if p[[i, j]] > 0.0 && q[[i, j]] > 0.0 {
                    kl += p[[i, j]] * (p[[i, j]] / q[[i, j]]).ln();
                }
            }
        }
        kl
    }

    /// Euclidean distance between two embeddings
    fn euclidean_distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let diff = a - b;
        diff.dot(&diff).sqrt()
    }

    /// Build k-nearest neighbor graph
    fn build_knn_graph(
        &self,
        embeddings: &HashMap<String, Array1<f32>>,
        entity_list: &[String],
        k: usize,
    ) -> Vec<Vec<usize>> {
        let n = entity_list.len();
        let mut knn_graph = Vec::new();

        for i in 0..n {
            let entity = &entity_list[i];
            let emb = &embeddings[entity];

            // Compute distances to all other entities
            let mut distances: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let other_emb = &embeddings[&entity_list[j]];
                    let dist = self.euclidean_distance(emb, other_emb);
                    (j, dist)
                })
                .collect();

            // Sort by distance and take top-k
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let neighbors: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();
            knn_graph.push(neighbors);
        }

        knn_graph
    }

    /// Export visualization to JSON
    pub fn export_json(&self, result: &VisualizationResult) -> Result<String> {
        serde_json::to_string_pretty(result)
            .map_err(|e| anyhow!("Failed to serialize visualization: {}", e))
    }

    /// Export visualization to CSV
    pub fn export_csv(&self, result: &VisualizationResult) -> Result<String> {
        let mut csv = String::from("entity");
        for i in 0..result.dimensions {
            csv.push_str(&format!(",dim{}", i + 1));
        }
        csv.push('\n');

        for (entity, coords) in &result.coordinates {
            csv.push_str(entity);
            for coord in coords {
                csv.push_str(&format!(",{}", coord));
            }
            csv.push('\n');
        }

        Ok(csv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_pca_visualization() {
        let mut embeddings = HashMap::new();
        embeddings.insert("e1".to_string(), array![1.0, 0.0, 0.0, 0.0]);
        embeddings.insert("e2".to_string(), array![0.0, 1.0, 0.0, 0.0]);
        embeddings.insert("e3".to_string(), array![0.0, 0.0, 1.0, 0.0]);
        embeddings.insert("e4".to_string(), array![0.0, 0.0, 0.0, 1.0]);

        let config = VisualizationConfig {
            method: ReductionMethod::PCA,
            target_dims: 2,
            ..Default::default()
        };

        let mut visualizer = EmbeddingVisualizer::new(config);
        let result = visualizer.visualize(&embeddings).unwrap();

        assert_eq!(result.coordinates.len(), 4);
        assert_eq!(result.dimensions, 2);
        assert!(result.explained_variance.is_some());
    }

    #[test]
    fn test_random_projection() {
        let mut embeddings = HashMap::new();
        for i in 0..10 {
            let emb = Array1::from_vec(vec![i as f32; 100]);
            embeddings.insert(format!("e{}", i), emb);
        }

        let config = VisualizationConfig {
            method: ReductionMethod::RandomProjection,
            target_dims: 3,
            ..Default::default()
        };

        let mut visualizer = EmbeddingVisualizer::new(config);
        let result = visualizer.visualize(&embeddings).unwrap();

        assert_eq!(result.coordinates.len(), 10);
        assert_eq!(result.dimensions, 3);
    }

    #[test]
    fn test_export_csv() {
        let mut coordinates = HashMap::new();
        coordinates.insert("e1".to_string(), vec![1.0, 2.0]);
        coordinates.insert("e2".to_string(), vec![3.0, 4.0]);

        let result = VisualizationResult {
            coordinates,
            dimensions: 2,
            method: ReductionMethod::PCA,
            explained_variance: None,
            final_loss: None,
        };

        let config = VisualizationConfig::default();
        let visualizer = EmbeddingVisualizer::new(config);
        let csv = visualizer.export_csv(&result).unwrap();

        assert!(csv.contains("entity,dim1,dim2"));
        assert!(csv.contains("e1,1,2"));
    }
}
