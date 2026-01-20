//! Scalar Quantization (SQ) for efficient vector compression
//!
//! This module implements Scalar Quantization, a simpler and faster alternative to
//! Product Quantization (PQ) that quantizes each vector dimension independently.
//!
//! SQ is particularly useful when:
//! - Training time needs to be minimal
//! - Simple, predictable compression is preferred
//! - Memory reduction is more important than extreme accuracy
//! - Real-time index updates are required (no retraining needed)

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Scalar quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqConfig {
    /// Number of bits per scalar (4 or 8)
    pub bits: u8,
    /// Quantization mode
    pub mode: QuantizationMode,
    /// Whether to normalize vectors before quantization
    pub normalize: bool,
    /// Number of training vectors to use for range estimation
    pub training_samples: usize,
}

impl Default for SqConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            mode: QuantizationMode::Uniform,
            normalize: false,
            training_samples: 10_000,
        }
    }
}

/// Quantization mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum QuantizationMode {
    /// Uniform quantization across global min/max
    Uniform,
    /// Per-dimension quantization with individual min/max
    PerDimension,
    /// Quantization using mean and standard deviation (more robust to outliers)
    MeanStd,
}

/// Quantization parameters for a single dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Minimum value in the dimension
    pub min: f32,
    /// Maximum value in the dimension
    pub max: f32,
    /// Scale factor for quantization
    pub scale: f32,
    /// Offset for quantization
    pub offset: f32,
}

impl QuantizationParams {
    /// Create quantization parameters from min/max values
    pub fn from_range(min: f32, max: f32, bits: u8) -> Self {
        let levels = (1 << bits) - 1;
        let range = max - min;
        let scale = if range > 1e-8 {
            levels as f32 / range
        } else {
            1.0
        };

        Self {
            min,
            max,
            scale,
            offset: min,
        }
    }

    /// Create parameters from mean and standard deviation (3-sigma range)
    pub fn from_mean_std(mean: f32, std: f32, bits: u8) -> Self {
        let min = mean - 3.0 * std;
        let max = mean + 3.0 * std;
        Self::from_range(min, max, bits)
    }

    /// Quantize a value
    pub fn quantize(&self, value: f32) -> u8 {
        let normalized = (value - self.offset) * self.scale;
        normalized.clamp(0.0, 255.0) as u8
    }

    /// Dequantize a value
    pub fn dequantize(&self, quantized: u8) -> f32 {
        (quantized as f32 / self.scale) + self.offset
    }
}

/// Scalar quantization index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqStats {
    /// Number of vectors in index
    pub vector_count: usize,
    /// Vector dimensionality
    pub dimensions: usize,
    /// Number of bits per scalar
    pub bits: u8,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Average quantization error
    pub avg_quantization_error: f32,
}

/// Scalar quantization vector index
pub struct SqIndex {
    config: SqConfig,
    dimensions: usize,
    quantization_params: Vec<QuantizationParams>,
    quantized_vectors: Vec<Vec<u8>>,
    uri_to_id: HashMap<String, usize>,
    id_to_uri: Vec<String>,
}

impl SqIndex {
    /// Create a new SQ index
    pub fn new(config: SqConfig, dimensions: usize) -> Self {
        Self {
            config,
            dimensions,
            quantization_params: Vec::new(),
            quantized_vectors: Vec::new(),
            uri_to_id: HashMap::new(),
            id_to_uri: Vec::new(),
        }
    }

    /// Train quantization parameters from training vectors
    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> Result<()> {
        if training_vectors.is_empty() {
            return Err(anyhow!("No training vectors provided"));
        }

        let dim = training_vectors[0].len();
        if dim != self.dimensions {
            return Err(anyhow!(
                "Training vector dimensions ({}) don't match index dimensions ({})",
                dim,
                self.dimensions
            ));
        }

        // Limit training samples
        let sample_count = training_vectors.len().min(self.config.training_samples);
        let samples = &training_vectors[..sample_count];

        match self.config.mode {
            QuantizationMode::Uniform => {
                self.train_uniform(samples)?;
            }
            QuantizationMode::PerDimension => {
                self.train_per_dimension(samples)?;
            }
            QuantizationMode::MeanStd => {
                self.train_mean_std(samples)?;
            }
        }

        tracing::info!(
            "Trained SQ index: mode={:?}, bits={}, samples={}, dimensions={}",
            self.config.mode,
            self.config.bits,
            sample_count,
            self.dimensions
        );

        Ok(())
    }

    /// Train uniform quantization (single global range)
    fn train_uniform(&mut self, samples: &[Vec<f32>]) -> Result<()> {
        let mut global_min = f32::INFINITY;
        let mut global_max = f32::NEG_INFINITY;

        for vector in samples {
            for &value in vector {
                global_min = global_min.min(value);
                global_max = global_max.max(value);
            }
        }

        let params = QuantizationParams::from_range(global_min, global_max, self.config.bits);
        self.quantization_params = vec![params; self.dimensions];

        Ok(())
    }

    /// Train per-dimension quantization
    fn train_per_dimension(&mut self, samples: &[Vec<f32>]) -> Result<()> {
        let mut dim_mins = vec![f32::INFINITY; self.dimensions];
        let mut dim_maxs = vec![f32::NEG_INFINITY; self.dimensions];

        for vector in samples {
            for (d, &value) in vector.iter().enumerate() {
                dim_mins[d] = dim_mins[d].min(value);
                dim_maxs[d] = dim_maxs[d].max(value);
            }
        }

        self.quantization_params = dim_mins
            .into_iter()
            .zip(dim_maxs)
            .map(|(min, max)| QuantizationParams::from_range(min, max, self.config.bits))
            .collect();

        Ok(())
    }

    /// Train using mean and standard deviation
    fn train_mean_std(&mut self, samples: &[Vec<f32>]) -> Result<()> {
        let n = samples.len() as f32;
        let mut dim_means = vec![0.0; self.dimensions];
        let mut dim_stds = vec![0.0; self.dimensions];

        // Calculate means
        for vector in samples {
            for (d, &value) in vector.iter().enumerate() {
                dim_means[d] += value;
            }
        }
        for mean in &mut dim_means {
            *mean /= n;
        }

        // Calculate standard deviations
        for vector in samples {
            for (d, &value) in vector.iter().enumerate() {
                let diff = value - dim_means[d];
                dim_stds[d] += diff * diff;
            }
        }
        for std in &mut dim_stds {
            *std = (*std / n).sqrt();
        }

        self.quantization_params = dim_means
            .into_iter()
            .zip(dim_stds)
            .map(|(mean, std)| QuantizationParams::from_mean_std(mean, std, self.config.bits))
            .collect();

        Ok(())
    }

    /// Add a vector to the index
    pub fn add(&mut self, uri: String, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(anyhow!(
                "Vector dimensions ({}) don't match index dimensions ({})",
                vector.len(),
                self.dimensions
            ));
        }

        if self.quantization_params.is_empty() {
            return Err(anyhow!(
                "Index not trained. Call train() before adding vectors."
            ));
        }

        let quantized = self.quantize_vector(&vector);
        let id = self.quantized_vectors.len();

        self.uri_to_id.insert(uri.clone(), id);
        self.id_to_uri.push(uri);
        self.quantized_vectors.push(quantized);

        Ok(())
    }

    /// Quantize a vector
    fn quantize_vector(&self, vector: &[f32]) -> Vec<u8> {
        vector
            .iter()
            .zip(&self.quantization_params)
            .map(|(&value, params)| params.quantize(value))
            .collect()
    }

    /// Dequantize a vector
    fn dequantize_vector(&self, quantized: &[u8]) -> Vec<f32> {
        quantized
            .iter()
            .zip(&self.quantization_params)
            .map(|(&q, params)| params.dequantize(q))
            .collect()
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if query.len() != self.dimensions {
            return Err(anyhow!(
                "Query dimensions ({}) don't match index dimensions ({})",
                query.len(),
                self.dimensions
            ));
        }

        if self.quantized_vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Quantize query
        let query_quantized = self.quantize_vector(query);

        // Compute distances
        let mut distances: Vec<(usize, f32)> = self
            .quantized_vectors
            .iter()
            .enumerate()
            .map(|(id, vec)| {
                let dist = self.asymmetric_distance(&query_quantized, vec);
                (id, dist)
            })
            .collect();

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k with URIs
        Ok(distances
            .into_iter()
            .take(k)
            .map(|(id, dist)| (self.id_to_uri[id].clone(), dist))
            .collect())
    }

    /// Compute asymmetric distance (query vs quantized vector)
    /// This provides better accuracy than symmetric distance
    fn asymmetric_distance(&self, query_quantized: &[u8], db_quantized: &[u8]) -> f32 {
        query_quantized
            .iter()
            .zip(db_quantized)
            .zip(&self.quantization_params)
            .map(|((&q1, &q2), params)| {
                let v1 = params.dequantize(q1);
                let v2 = params.dequantize(q2);
                let diff = v1 - v2;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Get statistics about the index
    pub fn stats(&self) -> SqStats {
        let vector_count = self.quantized_vectors.len();
        let bits_per_vector = self.dimensions * self.config.bits as usize;
        let bytes_per_vector = (bits_per_vector + 7) / 8;
        let memory_bytes = vector_count * bytes_per_vector;

        let original_bytes = vector_count * self.dimensions * 4; // f32 = 4 bytes
        let compression_ratio = if memory_bytes > 0 {
            original_bytes as f32 / memory_bytes as f32
        } else {
            0.0
        };

        SqStats {
            vector_count,
            dimensions: self.dimensions,
            bits: self.config.bits,
            compression_ratio,
            memory_bytes,
            avg_quantization_error: self.estimate_quantization_error(),
        }
    }

    /// Estimate average quantization error
    fn estimate_quantization_error(&self) -> f32 {
        if self.quantized_vectors.is_empty() {
            return 0.0;
        }

        let sample_size = self.quantized_vectors.len().min(100);
        let mut total_error = 0.0;

        for quantized in self.quantized_vectors.iter().take(sample_size) {
            let dequantized = self.dequantize_vector(quantized);
            let reconstructed_quantized = self.quantize_vector(&dequantized);

            // Error is difference between original quantized and re-quantized
            let error: f32 = quantized
                .iter()
                .zip(&reconstructed_quantized)
                .map(|(&a, &b)| (a as f32 - b as f32).abs())
                .sum();

            total_error += error / self.dimensions as f32;
        }

        total_error / sample_size as f32
    }

    /// Get vector by URI
    pub fn get(&self, uri: &str) -> Option<Vec<f32>> {
        self.uri_to_id
            .get(uri)
            .and_then(|&id| self.quantized_vectors.get(id))
            .map(|q| self.dequantize_vector(q))
    }

    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.quantized_vectors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.quantized_vectors.is_empty()
    }

    /// Get configuration
    pub fn config(&self) -> &SqConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_params() {
        let params = QuantizationParams::from_range(0.0, 1.0, 8);
        assert_eq!(params.quantize(0.0), 0);
        assert_eq!(params.quantize(1.0), 255);
        assert_eq!(params.quantize(0.5), 127);

        let dequantized = params.dequantize(127);
        assert!((dequantized - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_sq_index_creation() {
        let config = SqConfig::default();
        let index = SqIndex::new(config, 128);
        assert_eq!(index.dimensions, 128);
        assert!(index.is_empty());
    }

    #[test]
    fn test_sq_training() {
        let config = SqConfig {
            bits: 8,
            mode: QuantizationMode::PerDimension,
            ..Default::default()
        };

        let mut index = SqIndex::new(config, 4);

        let training_data = vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
        ];

        assert!(index.train(&training_data).is_ok());
        assert_eq!(index.quantization_params.len(), 4);
    }

    #[test]
    fn test_sq_add_and_search() {
        let config = SqConfig::default();
        let mut index = SqIndex::new(config, 4);

        let training_data = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0],
        ];

        index.train(&training_data).unwrap();

        index
            .add("vec1".to_string(), vec![0.1, 0.1, 0.1, 0.1])
            .unwrap();
        index
            .add("vec2".to_string(), vec![0.9, 0.9, 0.9, 0.9])
            .unwrap();
        index
            .add("vec3".to_string(), vec![1.8, 1.8, 1.8, 1.8])
            .unwrap();

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "vec1");
    }

    #[test]
    fn test_sq_stats() {
        let config = SqConfig {
            bits: 4,
            ..Default::default()
        };
        let mut index = SqIndex::new(config, 128);

        let training_data: Vec<Vec<f32>> =
            (0..100).map(|_| (0..128).map(|_| 0.5).collect()).collect();

        index.train(&training_data).unwrap();

        for i in 0..10 {
            index.add(format!("vec{}", i), vec![0.5; 128]).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.vector_count, 10);
        assert_eq!(stats.dimensions, 128);
        assert_eq!(stats.bits, 4);
        assert!(stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_different_quantization_modes() {
        let dimensions = 4;
        let training_data = vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
        ];

        // Test Uniform mode
        let mut index_uniform = SqIndex::new(
            SqConfig {
                mode: QuantizationMode::Uniform,
                ..Default::default()
            },
            dimensions,
        );
        assert!(index_uniform.train(&training_data).is_ok());

        // Test PerDimension mode
        let mut index_per_dim = SqIndex::new(
            SqConfig {
                mode: QuantizationMode::PerDimension,
                ..Default::default()
            },
            dimensions,
        );
        assert!(index_per_dim.train(&training_data).is_ok());

        // Test MeanStd mode
        let mut index_mean_std = SqIndex::new(
            SqConfig {
                mode: QuantizationMode::MeanStd,
                ..Default::default()
            },
            dimensions,
        );
        assert!(index_mean_std.train(&training_data).is_ok());
    }
}
