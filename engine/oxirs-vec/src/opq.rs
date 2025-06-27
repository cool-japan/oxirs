//! Optimized Product Quantization (OPQ) implementation
//!
//! OPQ learns an optimal rotation matrix that minimizes quantization error
//! by aligning the data before applying product quantization. This typically
//! provides better compression quality than standard PQ.

use crate::{Vector, pq::{PQConfig, PQIndex}};
use anyhow::{anyhow, Result};
use nalgebra::{DMatrix, DVector, SVD};
use oxirs_core::parallel::*;
use std::collections::HashMap;

/// Configuration for Optimized Product Quantization
#[derive(Debug, Clone)]
pub struct OPQConfig {
    /// Base PQ configuration
    pub pq_config: PQConfig,
    /// Number of iterations for alternating optimization
    pub n_iterations: usize,
    /// Whether to center data before rotation
    pub center_data: bool,
    /// Regularization parameter for rotation matrix
    pub regularization: f32,
}

impl Default for OPQConfig {
    fn default() -> Self {
        Self {
            pq_config: PQConfig::default(),
            n_iterations: 10,
            center_data: true,
            regularization: 0.0,
        }
    }
}

/// Optimized Product Quantization index
pub struct OPQIndex {
    /// Configuration
    config: OPQConfig,
    /// Rotation matrix R (d x d)
    rotation_matrix: Option<DMatrix<f32>>,
    /// Data mean for centering
    data_mean: Option<DVector<f32>>,
    /// Underlying PQ index
    pq_index: PQIndex,
    /// Whether the model is trained
    is_trained: bool,
}

impl OPQIndex {
    /// Create a new OPQ index
    pub fn new(config: OPQConfig) -> Self {
        Self {
            pq_index: PQIndex::new(config.pq_config.clone()),
            config,
            rotation_matrix: None,
            data_mean: None,
            is_trained: false,
        }
    }

    /// Train the OPQ model using alternating optimization
    pub fn train(&mut self, vectors: &[Vector]) -> Result<()> {
        if vectors.is_empty() {
            return Err(anyhow!("Cannot train OPQ with empty data"));
        }

        let n_samples = vectors.len();
        let dimensions = vectors[0].dimensions;

        // Convert vectors to matrix (samples x dimensions)
        let mut data_matrix = DMatrix::zeros(n_samples, dimensions);
        for (i, vector) in vectors.iter().enumerate() {
            let vec_f32 = vector.as_f32();
            for (j, &val) in vec_f32.iter().enumerate() {
                data_matrix[(i, j)] = val;
            }
        }

        // Center data if requested
        if self.config.center_data {
            let mean = self.compute_mean(&data_matrix);
            self.center_data_matrix(&mut data_matrix, &mean);
            self.data_mean = Some(mean);
        }

        // Initialize rotation matrix as identity
        let mut rotation = DMatrix::identity(dimensions, dimensions);

        // Alternating optimization
        for iteration in 0..self.config.n_iterations {
            println!("OPQ iteration {}/{}", iteration + 1, self.config.n_iterations);

            // Step 1: Fix R, optimize C (codebooks)
            let rotated_data = self.apply_rotation(&data_matrix, &rotation);
            let rotated_vectors = self.matrix_to_vectors(&rotated_data);
            
            // Train PQ on rotated data
            self.pq_index.train(&rotated_vectors)?;

            // Step 2: Fix C, optimize R
            rotation = self.optimize_rotation(&data_matrix, &rotated_vectors)?;

            // Compute reconstruction error for monitoring
            let error = self.compute_reconstruction_error(&data_matrix, &rotation)?;
            println!("Reconstruction error: {}", error);
        }

        self.rotation_matrix = Some(rotation);
        self.is_trained = true;

        Ok(())
    }

    /// Compute mean of data matrix
    fn compute_mean(&self, data: &DMatrix<f32>) -> DVector<f32> {
        let n_samples = data.nrows() as f32;
        let mut mean = DVector::zeros(data.ncols());
        
        for i in 0..data.ncols() {
            mean[i] = data.column(i).sum() / n_samples;
        }
        
        mean
    }

    /// Center data matrix by subtracting mean
    fn center_data_matrix(&self, data: &mut DMatrix<f32>, mean: &DVector<f32>) {
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                data[(i, j)] -= mean[j];
            }
        }
    }

    /// Apply rotation matrix to data
    fn apply_rotation(&self, data: &DMatrix<f32>, rotation: &DMatrix<f32>) -> DMatrix<f32> {
        data * rotation.transpose()
    }

    /// Convert matrix back to vectors
    fn matrix_to_vectors(&self, matrix: &DMatrix<f32>) -> Vec<Vector> {
        let mut vectors = Vec::with_capacity(matrix.nrows());
        
        for i in 0..matrix.nrows() {
            let row: Vec<f32> = matrix.row(i).iter().cloned().collect();
            vectors.push(Vector::new(row));
        }
        
        vectors
    }

    /// Optimize rotation matrix using SVD
    fn optimize_rotation(&self, data: &DMatrix<f32>, rotated_vectors: &[Vector]) -> Result<DMatrix<f32>> {
        // Reconstruct data using current codebooks
        let mut reconstructed = DMatrix::zeros(data.nrows(), data.ncols());
        
        for (i, vector) in rotated_vectors.iter().enumerate() {
            // Encode and decode to get reconstruction
            if let Ok(reconstructed_vec) = self.pq_index.reconstruct(vector) {
                let rec_f32 = reconstructed_vec.as_f32();
                for (j, &val) in rec_f32.iter().enumerate() {
                    reconstructed[(i, j)] = val;
                }
            }
        }

        // Solve orthogonal Procrustes problem: min ||X - Y*R||_F
        // Solution: R = U*V^T where X^T*Y = U*S*V^T
        let correlation = data.transpose() * &reconstructed;
        
        // Add regularization if needed
        let mut reg_correlation = correlation.clone();
        if self.config.regularization > 0.0 {
            for i in 0..reg_correlation.ncols().min(reg_correlation.nrows()) {
                reg_correlation[(i, i)] += self.config.regularization;
            }
        }

        // Compute SVD
        let svd = SVD::new(reg_correlation, true, true);
        let u = svd.u.ok_or_else(|| anyhow!("SVD failed to compute U"))?;
        let v_t = svd.v_t.ok_or_else(|| anyhow!("SVD failed to compute V^T"))?;

        // Optimal rotation is U * V^T
        Ok(u * v_t)
    }

    /// Compute reconstruction error
    fn compute_reconstruction_error(&self, data: &DMatrix<f32>, rotation: &DMatrix<f32>) -> Result<f32> {
        let rotated = self.apply_rotation(data, rotation);
        let rotated_vecs = self.matrix_to_vectors(&rotated);
        
        let mut total_error = 0.0;
        for (i, vec) in rotated_vecs.iter().enumerate() {
            if let Ok(reconstructed) = self.pq_index.reconstruct(vec) {
                let rec_f32 = reconstructed.as_f32();
                for (j, &val) in rec_f32.iter().enumerate() {
                    let diff = rotated[(i, j)] - val;
                    total_error += diff * diff;
                }
            }
        }
        
        Ok((total_error / (data.nrows() * data.ncols()) as f32).sqrt())
    }

    /// Encode a vector using OPQ
    pub fn encode(&self, vector: &Vector) -> Result<Vec<u8>> {
        if !self.is_trained {
            return Err(anyhow!("OPQ index must be trained before encoding"));
        }

        // Apply centering and rotation
        let transformed = self.transform_vector(vector)?;
        
        // Use PQ to encode
        self.pq_index.encode(&transformed)
    }

    /// Decode PQ codes to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Result<Vector> {
        if !self.is_trained {
            return Err(anyhow!("OPQ index must be trained before decoding"));
        }

        // Decode using PQ
        let rotated = self.pq_index.decode(codes)?;
        
        // Apply inverse transformation
        self.inverse_transform_vector(&rotated)
    }

    /// Transform vector: center and rotate
    fn transform_vector(&self, vector: &Vector) -> Result<Vector> {
        let rotation = self.rotation_matrix.as_ref()
            .ok_or_else(|| anyhow!("Rotation matrix not initialized"))?;
        
        let vec_f32 = vector.as_f32();
        let mut vec_dv = DVector::from_vec(vec_f32.to_vec());
        
        // Center if needed
        if let Some(ref mean) = self.data_mean {
            vec_dv -= mean;
        }
        
        // Apply rotation
        let rotated = rotation.transpose() * vec_dv;
        
        Ok(Vector::new(rotated.iter().cloned().collect()))
    }

    /// Inverse transform: rotate back and uncenter
    fn inverse_transform_vector(&self, vector: &Vector) -> Result<Vector> {
        let rotation = self.rotation_matrix.as_ref()
            .ok_or_else(|| anyhow!("Rotation matrix not initialized"))?;
        
        let vec_f32 = vector.as_f32();
        let vec_dv = DVector::from_vec(vec_f32.to_vec());
        
        // Apply inverse rotation
        let unrotated = rotation * vec_dv;
        
        // Uncenter if needed
        let mut result = unrotated;
        if let Some(ref mean) = self.data_mean {
            result += mean;
        }
        
        Ok(Vector::new(result.iter().cloned().collect()))
    }

    /// Search for nearest neighbors using asymmetric distance computation
    pub fn search(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        if !self.is_trained {
            return Err(anyhow!("OPQ index must be trained before searching"));
        }

        // Transform query
        let transformed_query = self.transform_vector(query)?;
        
        // Use PQ search
        self.pq_index.search(&transformed_query, k)
    }

    /// Get compression statistics
    pub fn stats(&self) -> OPQStats {
        let pq_stats = self.pq_index.stats();
        
        OPQStats {
            pq_stats,
            is_trained: self.is_trained,
            has_rotation: self.rotation_matrix.is_some(),
            rotation_rank: self.rotation_matrix.as_ref()
                .map(|r| r.rank(1e-6))
                .unwrap_or(0),
        }
    }
}

/// Statistics for OPQ index
#[derive(Debug, Clone)]
pub struct OPQStats {
    pub pq_stats: crate::pq::PQStats,
    pub is_trained: bool,
    pub has_rotation: bool,
    pub rotation_rank: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorIndex;

    #[test]
    fn test_opq_basic() -> Result<()> {
        let config = OPQConfig {
            pq_config: PQConfig {
                n_subquantizers: 4,
                n_centroids: 16,
                ..Default::default()
            },
            n_iterations: 3,
            ..Default::default()
        };

        let mut opq = OPQIndex::new(config);

        // Create test data
        let vectors: Vec<Vector> = (0..100)
            .map(|i| {
                let values: Vec<f32> = (0..16)
                    .map(|j| (i as f32 * 0.1 + j as f32) % 10.0)
                    .collect();
                Vector::new(values)
            })
            .collect();

        // Train OPQ
        opq.train(&vectors)?;

        // Test encoding/decoding
        let test_vec = Vector::new(vec![1.0; 16]);
        let codes = opq.encode(&test_vec)?;
        let reconstructed = opq.decode(&codes)?;

        assert_eq!(reconstructed.dimensions, 16);

        Ok(())
    }

    #[test]
    fn test_opq_search() -> Result<()> {
        let config = OPQConfig::default();
        let mut opq = OPQIndex::new(config);

        // Create and train on random vectors
        let vectors: Vec<Vector> = (0..50)
            .map(|i| {
                let values: Vec<f32> = (0..8)
                    .map(|j| ((i * j) as f32).sin())
                    .collect();
                Vector::new(values)
            })
            .collect();

        opq.train(&vectors)?;

        // Add vectors to index
        for (i, vec) in vectors.iter().enumerate() {
            opq.pq_index.insert(format!("vec_{}", i), vec.clone())?;
        }

        // Search
        let query = Vector::new(vec![0.5; 8]);
        let results = opq.search(&query, 5)?;

        assert_eq!(results.len(), 5);

        Ok(())
    }
}