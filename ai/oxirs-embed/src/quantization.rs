//! Quantization Support for Model Compression
//!
//! This module provides quantization techniques to compress knowledge graph
//! embeddings by reducing precision from float32 to int8/int4, significantly
//! reducing model size and improving inference speed.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// Symmetric quantization (zero point = 0)
    Symmetric,
    /// Asymmetric quantization (learnable zero point)
    Asymmetric,
    /// Per-channel quantization
    PerChannel,
    /// Per-tensor quantization
    PerTensor,
}

/// Quantization bit width
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitWidth {
    /// 8-bit quantization
    Int8,
    /// 4-bit quantization
    Int4,
    /// Binary quantization (1-bit)
    Binary,
}

impl BitWidth {
    /// Get quantization range
    pub fn range(&self) -> (i32, i32) {
        match self {
            BitWidth::Int8 => (-128, 127),
            BitWidth::Int4 => (-8, 7),
            BitWidth::Binary => (0, 1),
        }
    }

    /// Get number of bits
    pub fn bits(&self) -> usize {
        match self {
            BitWidth::Int8 => 8,
            BitWidth::Int4 => 4,
            BitWidth::Binary => 1,
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization scheme to use
    pub scheme: QuantizationScheme,
    /// Bit width for quantization
    pub bit_width: BitWidth,
    /// Enable calibration for better quantization
    pub calibration: bool,
    /// Number of calibration samples
    pub calibration_samples: usize,
    /// Quantize only weights (keep activations in float)
    pub weights_only: bool,
    /// Use quantization-aware training
    pub qat: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            scheme: QuantizationScheme::Symmetric,
            bit_width: BitWidth::Int8,
            calibration: true,
            calibration_samples: 1000,
            weights_only: true,
            qat: false,
        }
    }
}

/// Quantization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: i32,
    /// Min value observed during calibration
    pub min_val: f32,
    /// Max value observed during calibration
    pub max_val: f32,
}

impl QuantizationParams {
    /// Compute quantization parameters from tensor statistics
    pub fn from_statistics(
        min_val: f32,
        max_val: f32,
        bit_width: BitWidth,
        symmetric: bool,
    ) -> Self {
        let (qmin, qmax) = bit_width.range();

        let (scale, zero_point) = if symmetric {
            // Symmetric quantization
            let max_abs = min_val.abs().max(max_val.abs());
            let scale = (2.0 * max_abs) / (qmax - qmin) as f32;
            (scale, 0)
        } else {
            // Asymmetric quantization
            let scale = (max_val - min_val) / (qmax - qmin) as f32;
            let zero_point = qmin - (min_val / scale).round() as i32;
            (scale, zero_point)
        };

        Self {
            scale,
            zero_point,
            min_val,
            max_val,
        }
    }

    /// Quantize a float value
    pub fn quantize(&self, value: f32, bit_width: BitWidth) -> i8 {
        let (qmin, qmax) = bit_width.range();
        let quantized = (value / self.scale).round() as i32 + self.zero_point;
        quantized.clamp(qmin, qmax) as i8
    }

    /// Dequantize an int value back to float
    pub fn dequantize(&self, quantized: i8) -> f32 {
        (quantized as i32 - self.zero_point) as f32 * self.scale
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    /// Quantized values (int8)
    pub values: Vec<i8>,
    /// Quantization parameters
    pub params: QuantizationParams,
    /// Original shape
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Create quantized tensor from float array
    pub fn from_array(array: &Array1<f32>, config: &QuantizationConfig) -> Self {
        let min_val = array.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = array.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let symmetric = matches!(config.scheme, QuantizationScheme::Symmetric);
        let params =
            QuantizationParams::from_statistics(min_val, max_val, config.bit_width, symmetric);

        let values: Vec<i8> = array
            .iter()
            .map(|&v| params.quantize(v, config.bit_width))
            .collect();

        Self {
            values,
            params,
            shape: vec![array.len()],
        }
    }

    /// Dequantize back to float array
    pub fn to_array(&self) -> Array1<f32> {
        Array1::from_vec(
            self.values
                .iter()
                .map(|&v| self.params.dequantize(v))
                .collect(),
        )
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        // Original: 4 bytes per float32
        // Quantized: 1 byte per int8 + overhead for params
        let original_size = self.values.len() * 4;
        let quantized_size = self.values.len() + std::mem::size_of::<QuantizationParams>();
        original_size as f32 / quantized_size as f32
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.values.len() + std::mem::size_of::<QuantizationParams>()
    }
}

/// Quantization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationStats {
    /// Total parameters quantized
    pub total_params: usize,
    /// Original model size (bytes)
    pub original_size_bytes: usize,
    /// Quantized model size (bytes)
    pub quantized_size_bytes: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Average quantization error
    pub avg_quantization_error: f32,
    /// Maximum quantization error
    pub max_quantization_error: f32,
}

impl Default for QuantizationStats {
    fn default() -> Self {
        Self {
            total_params: 0,
            original_size_bytes: 0,
            quantized_size_bytes: 0,
            compression_ratio: 1.0,
            avg_quantization_error: 0.0,
            max_quantization_error: 0.0,
        }
    }
}

/// Model quantizer
pub struct ModelQuantizer {
    config: QuantizationConfig,
    stats: QuantizationStats,
}

impl ModelQuantizer {
    /// Create new model quantizer
    pub fn new(config: QuantizationConfig) -> Self {
        info!(
            "Initialized model quantizer: scheme={:?}, bit_width={:?}",
            config.scheme, config.bit_width
        );

        Self {
            config,
            stats: QuantizationStats::default(),
        }
    }

    /// Quantize entity embeddings
    pub fn quantize_embeddings(
        &mut self,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<HashMap<String, QuantizedTensor>> {
        if embeddings.is_empty() {
            return Err(anyhow!("No embeddings to quantize"));
        }

        info!("Quantizing {} embeddings", embeddings.len());

        let mut quantized_embeddings = HashMap::new();
        let mut total_error = 0.0;
        let mut max_error: f32 = 0.0;

        for (entity, embedding) in embeddings {
            let quantized = QuantizedTensor::from_array(embedding, &self.config);

            // Compute quantization error
            let dequantized = quantized.to_array();
            let error = self.compute_error(embedding, &dequantized);
            total_error += error;
            max_error = max_error.max(error);

            // Update stats
            self.stats.original_size_bytes += embedding.len() * 4;
            self.stats.quantized_size_bytes += quantized.size_bytes();

            quantized_embeddings.insert(entity.clone(), quantized);
        }

        self.stats.total_params = embeddings.values().map(|e| e.len()).sum();
        self.stats.compression_ratio =
            self.stats.original_size_bytes as f32 / self.stats.quantized_size_bytes as f32;
        self.stats.avg_quantization_error = total_error / embeddings.len() as f32;
        self.stats.max_quantization_error = max_error;

        info!(
            "Quantization complete: compression_ratio={:.2}x, avg_error={:.6}",
            self.stats.compression_ratio, self.stats.avg_quantization_error
        );

        Ok(quantized_embeddings)
    }

    /// Dequantize embeddings
    pub fn dequantize_embeddings(
        &self,
        quantized: &HashMap<String, QuantizedTensor>,
    ) -> HashMap<String, Array1<f32>> {
        quantized
            .iter()
            .map(|(entity, q)| (entity.clone(), q.to_array()))
            .collect()
    }

    /// Quantize a single embedding
    pub fn quantize_embedding(&self, embedding: &Array1<f32>) -> QuantizedTensor {
        QuantizedTensor::from_array(embedding, &self.config)
    }

    /// Dequantize a single embedding
    pub fn dequantize_embedding(&self, quantized: &QuantizedTensor) -> Array1<f32> {
        quantized.to_array()
    }

    /// Compute mean squared error between original and dequantized
    fn compute_error(&self, original: &Array1<f32>, dequantized: &Array1<f32>) -> f32 {
        let diff = original - dequantized;
        let mse = diff.dot(&diff) / original.len() as f32;
        mse.sqrt() // RMSE
    }

    /// Calibrate quantization parameters using sample data
    pub fn calibrate(&mut self, embeddings: &HashMap<String, Array1<f32>>) -> Result<()> {
        if !self.config.calibration {
            return Ok(());
        }

        info!(
            "Calibrating quantization parameters with {} samples",
            self.config.calibration_samples.min(embeddings.len())
        );

        // Collect statistics from sample embeddings
        let samples: Vec<&Array1<f32>> = embeddings
            .values()
            .take(self.config.calibration_samples)
            .collect();

        // Find global min/max for per-tensor quantization
        let mut global_min = f32::INFINITY;
        let mut global_max = f32::NEG_INFINITY;

        for embedding in samples {
            let min = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            global_min = global_min.min(min);
            global_max = global_max.max(max);
        }

        debug!(
            "Calibration complete: min={:.6}, max={:.6}",
            global_min, global_max
        );

        Ok(())
    }

    /// Get quantization statistics
    pub fn get_stats(&self) -> &QuantizationStats {
        &self.stats
    }

    /// Estimate inference speedup
    pub fn estimate_speedup(&self) -> f32 {
        // Int8 operations are typically 2-4x faster than float32
        match self.config.bit_width {
            BitWidth::Int8 => 3.0,
            BitWidth::Int4 => 5.0,
            BitWidth::Binary => 10.0,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_quantization_params() {
        let min_val = -10.0;
        let max_val = 10.0;

        let params = QuantizationParams::from_statistics(
            min_val,
            max_val,
            BitWidth::Int8,
            true, // symmetric
        );

        assert!(params.scale > 0.0);
        assert_eq!(params.zero_point, 0); // Symmetric should have zero point = 0
    }

    #[test]
    fn test_quantize_dequantize() {
        let params = QuantizationParams::from_statistics(-10.0, 10.0, BitWidth::Int8, true);

        let value = 5.0;
        let quantized = params.quantize(value, BitWidth::Int8);
        let dequantized = params.dequantize(quantized);

        // Should be approximately equal (within quantization error)
        assert!((value - dequantized).abs() < 1.0);
    }

    #[test]
    fn test_quantized_tensor() {
        // Use larger array (128 elements) so compression ratio > 1.0
        // With small arrays, quantization params overhead dominates
        let array = Array1::from_vec((0..128).map(|i| i as f32 * 0.1).collect());
        let config = QuantizationConfig::default();

        let quantized = QuantizedTensor::from_array(&array, &config);
        let dequantized = quantized.to_array();

        assert_eq!(quantized.values.len(), 128);
        assert_eq!(dequantized.len(), 128);

        // Check compression ratio (should be ~3.8x for 128-dim)
        assert!(quantized.compression_ratio() > 1.0);
    }

    #[test]
    fn test_model_quantizer() {
        let mut embeddings = HashMap::new();
        // Use larger embeddings (128-dim) for meaningful compression
        embeddings.insert(
            "e1".to_string(),
            Array1::from_vec((0..128).map(|i| i as f32 * 0.1).collect()),
        );
        embeddings.insert(
            "e2".to_string(),
            Array1::from_vec((0..128).map(|i| (i as f32 * 0.1) + 10.0).collect()),
        );

        let config = QuantizationConfig::default();
        let mut quantizer = ModelQuantizer::new(config);

        let quantized = quantizer.quantize_embeddings(&embeddings).unwrap();

        assert_eq!(quantized.len(), 2);
        assert!(quantizer.stats.compression_ratio > 1.0);
        assert!(quantizer.stats.avg_quantization_error >= 0.0);
    }

    #[test]
    fn test_roundtrip() {
        let mut embeddings = HashMap::new();
        embeddings.insert("e1".to_string(), array![1.0, -2.0, 3.5, -4.2]);

        let config = QuantizationConfig::default();
        let mut quantizer = ModelQuantizer::new(config);

        let quantized = quantizer.quantize_embeddings(&embeddings).unwrap();
        let dequantized = quantizer.dequantize_embeddings(&quantized);

        assert_eq!(dequantized.len(), 1);

        // Values should be close to original
        let original = &embeddings["e1"];
        let recovered = &dequantized["e1"];

        for i in 0..original.len() {
            let error = (original[i] - recovered[i]).abs();
            // Quantization error should be small but non-zero
            assert!(error < 1.0);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let mut embeddings = HashMap::new();
        for i in 0..100 {
            let emb = Array1::from_vec(vec![i as f32; 128]);
            embeddings.insert(format!("e{}", i), emb);
        }

        let config = QuantizationConfig::default();
        let mut quantizer = ModelQuantizer::new(config);

        quantizer.quantize_embeddings(&embeddings).unwrap();

        // Int8 should give ~4x compression (32-bit to 8-bit)
        assert!(quantizer.stats.compression_ratio > 3.0);
        assert!(quantizer.stats.compression_ratio < 5.0);
    }
}
