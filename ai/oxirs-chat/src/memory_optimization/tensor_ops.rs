//! Memory-efficient tensor operations

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Memory-efficient tensor representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryEfficientTensor {
    /// Full precision (f32)
    F32(Vec<f32>),

    /// Half precision (f16) - 50% memory reduction
    F16(Vec<u16>),

    /// Quantized (i8) - 75% memory reduction
    I8 {
        data: Vec<i8>,
        scale: f32,
        zero_point: f32, // Stores min value for dequantization
    },

    /// Sparse (only non-zero values)
    Sparse {
        indices: Vec<usize>,
        values: Vec<f32>,
        size: usize,
    },
}

impl MemoryEfficientTensor {
    /// Convert to f32 for computation
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Self::F32(data) => data.clone(),
            Self::F16(data) => data.iter().map(|x| f16_to_f32(*x)).collect(),
            Self::I8 {
                data,
                scale,
                zero_point,
            } => data
                .iter()
                .map(|x| (*x as f32 + 128.0) * scale + zero_point)
                .collect(),
            Self::Sparse {
                indices,
                values,
                size,
            } => {
                let mut result = vec![0.0f32; *size];
                for (idx, val) in indices.iter().zip(values.iter()) {
                    result[*idx] = *val;
                }
                result
            }
        }
    }

    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            Self::F32(data) => data.len() * 4,
            Self::F16(data) => data.len() * 2,
            Self::I8 { data, .. } => data.len() + 8, // data + scale (f32) + zero_point (f32)
            Self::Sparse {
                indices, values, ..
            } => indices.len() * 8 + values.len() * 4 + 8,
        }
    }
}

/// Tensor optimizer for memory efficiency
pub struct TensorOptimizer {
    use_low_precision: bool,
    quantization_enabled: bool,
    sparse_threshold: f32,
}

impl TensorOptimizer {
    pub fn new(use_low_precision: bool) -> Self {
        Self {
            use_low_precision,
            quantization_enabled: false,
            sparse_threshold: 0.5, // 50% sparsity threshold
        }
    }

    /// Enable quantization for even more memory savings
    pub fn with_quantization(mut self) -> Self {
        self.quantization_enabled = true;
        self
    }

    /// Set sparsity threshold (0.0 to 1.0)
    pub fn with_sparse_threshold(mut self, threshold: f32) -> Self {
        self.sparse_threshold = threshold;
        self
    }

    /// Optimize tensor representation
    pub fn optimize(&self, tensor: &[f32]) -> Result<MemoryEfficientTensor> {
        // Check for sparsity
        let sparsity = calculate_sparsity(tensor);
        if sparsity >= self.sparse_threshold {
            return Ok(self.to_sparse(tensor));
        }

        // Use quantization if enabled
        if self.quantization_enabled {
            return Ok(self.quantize_i8(tensor));
        }

        // Use half precision if enabled
        if self.use_low_precision {
            return Ok(self.to_f16(tensor));
        }

        // Default: keep as f32
        Ok(MemoryEfficientTensor::F32(tensor.to_vec()))
    }

    fn to_f16(&self, tensor: &[f32]) -> MemoryEfficientTensor {
        let data: Vec<u16> = tensor.iter().map(|x| f32_to_f16(*x)).collect();
        MemoryEfficientTensor::F16(data)
    }

    fn quantize_i8(&self, tensor: &[f32]) -> MemoryEfficientTensor {
        let min = tensor.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = tensor.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let scale = (max - min) / 255.0;
        let _zero_point = 0i8; // Use 0 as zero point for simplicity

        let data: Vec<i8> = tensor
            .iter()
            .map(|x| (((x - min) / scale).round() as i32 - 128).clamp(-128, 127) as i8)
            .collect();

        MemoryEfficientTensor::I8 {
            data,
            scale,
            zero_point: min, // Store min as zero_point for dequantization
        }
    }

    fn to_sparse(&self, tensor: &[f32]) -> MemoryEfficientTensor {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (idx, &val) in tensor.iter().enumerate() {
            if val.abs() > 1e-6 {
                indices.push(idx);
                values.push(val);
            }
        }

        MemoryEfficientTensor::Sparse {
            indices,
            values,
            size: tensor.len(),
        }
    }

    /// Calculate memory savings
    pub fn memory_savings(&self, original: &[f32], optimized: &MemoryEfficientTensor) -> f64 {
        let original_size = original.len() * 4; // f32 = 4 bytes
        let optimized_size = optimized.memory_size();

        1.0 - (optimized_size as f64 / original_size as f64)
    }
}

/// Calculate sparsity (ratio of zero elements)
fn calculate_sparsity(tensor: &[f32]) -> f32 {
    if tensor.is_empty() {
        return 0.0;
    }

    let zeros = tensor.iter().filter(|x| x.abs() < 1e-6).count();
    zeros as f32 / tensor.len() as f32
}

/// Simple f32 to f16 conversion (simplified, not IEEE 754 compliant)
fn f32_to_f16(value: f32) -> u16 {
    // Simplified conversion (for demonstration)
    // In production, use proper IEEE 754 half-precision conversion
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = (bits >> 13) & 0x3FF;

    if exp == 0 {
        return sign as u16;
    }

    let exp_adj = exp - 127 + 15;
    if exp_adj >= 31 {
        return (sign | 0x7C00) as u16; // Infinity
    }
    if exp_adj <= 0 {
        return sign as u16; // Zero
    }

    (sign | ((exp_adj as u32) << 10) | frac) as u16
}

/// Simple f16 to f32 conversion
fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value >> 15) & 1) as u32;
    let exp = ((value >> 10) & 0x1F) as i32;
    let frac = (value & 0x3FF) as u32;

    if exp == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }

    if exp == 31 {
        return if frac == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }

    let exp_adj = exp - 15 + 127;
    let bits = (sign << 31) | ((exp_adj as u32) << 23) | (frac << 13);
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_tensor() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = MemoryEfficientTensor::F32(data.clone());

        let recovered = tensor.to_f32();
        assert_eq!(recovered, data);
        assert_eq!(tensor.memory_size(), 12); // 3 * 4 bytes
    }

    #[test]
    fn test_f16_tensor() {
        let optimizer = TensorOptimizer::new(true);
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let optimized = optimizer.optimize(&data).unwrap();
        assert_eq!(optimized.memory_size(), 8); // 4 * 2 bytes

        let recovered = optimized.to_f32();
        assert_eq!(recovered.len(), data.len());
    }

    #[test]
    fn test_quantized_tensor() {
        let optimizer = TensorOptimizer::new(false).with_quantization();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let optimized = optimizer.optimize(&data).unwrap();
        let recovered = optimized.to_f32();

        assert_eq!(recovered.len(), data.len());
        // Values should be close but not exact due to quantization
        // i8 quantization introduces more error, so use larger tolerance
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.5, "Expected {} but got {}", a, b);
        }
    }

    #[test]
    fn test_sparse_tensor() {
        let optimizer = TensorOptimizer::new(false).with_sparse_threshold(0.5);
        let data = vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0]; // 66% sparse

        let optimized = optimizer.optimize(&data).unwrap();
        assert!(matches!(optimized, MemoryEfficientTensor::Sparse { .. }));

        let recovered = optimized.to_f32();
        assert_eq!(recovered.len(), data.len());

        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn test_calculate_sparsity() {
        let sparse = vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0];
        let sparsity = calculate_sparsity(&sparse);
        assert!((sparsity - 0.666).abs() < 0.01);

        let dense = vec![1.0, 2.0, 3.0, 4.0];
        let sparsity_dense = calculate_sparsity(&dense);
        assert_eq!(sparsity_dense, 0.0);
    }

    #[test]
    fn test_memory_savings() {
        let optimizer = TensorOptimizer::new(false).with_quantization();
        let data = vec![1.0; 1000];

        let optimized = optimizer.optimize(&data).unwrap();
        let savings = optimizer.memory_savings(&data, &optimized);

        assert!(savings > 0.7); // Should save >70% with i8 quantization
    }
}
