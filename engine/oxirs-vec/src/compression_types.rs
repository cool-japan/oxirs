//! Codec types, compression stats, config enums, error types for vector compression.

use crate::{Vector, VectorData, VectorError};
use half::f16;

/// Compression method selection
#[derive(Debug, Clone, Default)]
pub enum CompressionMethod {
    #[default]
    None,
    Zstd {
        level: i32,
    },
    Quantization {
        bits: u8,
    },
    ProductQuantization {
        subvectors: usize,
        codebook_size: usize,
    },
    Pca {
        components: usize,
    },
    Adaptive {
        quality_level: AdaptiveQuality,
        analysis_samples: usize,
    },
}

/// Adaptive quality level for adaptive compression
#[derive(Debug, Clone)]
pub enum AdaptiveQuality {
    Fast,      // Prioritize speed, moderate compression
    Balanced,  // Balance speed and compression ratio
    BestRatio, // Prioritize compression ratio over speed
}

/// Trait implemented by all vector compressors
pub trait VectorCompressor: Send + Sync {
    fn compress(&self, vector: &Vector) -> Result<Vec<u8>, VectorError>;
    fn decompress(&self, data: &[u8], dimensions: usize) -> Result<Vector, VectorError>;
    fn compression_ratio(&self) -> f32;
}

/// Performance metrics for a compressor instance
#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    pub vectors_compressed: usize,
    pub total_original_size: usize,
    pub total_compressed_size: usize,
    pub compression_time_ms: f64,
    pub decompression_time_ms: f64,
    pub current_ratio: f32,
    pub method_switches: usize,
}

impl Default for CompressionMetrics {
    fn default() -> Self {
        Self {
            vectors_compressed: 0,
            total_original_size: 0,
            total_compressed_size: 0,
            compression_time_ms: 0.0,
            decompression_time_ms: 0.0,
            current_ratio: 1.0,
            method_switches: 0,
        }
    }
}

/// Vector characteristics analysis for adaptive compression selection
#[derive(Debug, Clone)]
pub struct VectorAnalysis {
    pub sparsity: f32,
    pub range: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub entropy: f32,
    pub dominant_patterns: Vec<f32>,
    pub recommended_method: CompressionMethod,
    pub expected_ratio: f32,
}

impl VectorAnalysis {
    pub fn analyze(vectors: &[Vector], quality: &AdaptiveQuality) -> Result<Self, VectorError> {
        if vectors.is_empty() {
            return Err(VectorError::InvalidDimensions(
                "No vectors to analyze".to_string(),
            ));
        }

        let mut all_values = Vec::new();
        let mut dimensions = 0;

        for vector in vectors {
            let values = match &vector.values {
                VectorData::F32(v) => v.clone(),
                VectorData::F64(v) => v.iter().map(|&x| x as f32).collect(),
                VectorData::F16(v) => v.iter().map(|&x| f16::from_bits(x).to_f32()).collect(),
                VectorData::I8(v) => v.iter().map(|&x| x as f32).collect(),
                VectorData::Binary(_) => {
                    return Ok(Self::binary_analysis(vectors.len()));
                }
            };
            if dimensions == 0 {
                dimensions = values.len();
            }
            all_values.extend(values);
        }

        if all_values.is_empty() {
            return Err(VectorError::InvalidDimensions(
                "No values to analyze".to_string(),
            ));
        }

        let min_val = all_values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = all_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;

        let variance =
            all_values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / all_values.len() as f32;
        let std_dev = variance.sqrt();

        let epsilon = std_dev * 0.01;
        let near_zero_count = all_values.iter().filter(|&&x| x.abs() < epsilon).count();
        let sparsity = near_zero_count as f32 / all_values.len() as f32;

        let entropy = Self::calculate_entropy(&all_values);
        let dominant_patterns = Self::find_dominant_patterns(&all_values);

        let (recommended_method, expected_ratio) =
            Self::select_optimal_method(sparsity, range, std_dev, entropy, dimensions, quality);

        Ok(Self {
            sparsity,
            range,
            mean,
            std_dev,
            entropy,
            dominant_patterns,
            recommended_method,
            expected_ratio,
        })
    }

    fn binary_analysis(_vector_count: usize) -> Self {
        Self {
            sparsity: 0.0,
            range: 1.0,
            mean: 0.5,
            std_dev: 0.5,
            entropy: 1.0,
            dominant_patterns: vec![0.0, 1.0],
            recommended_method: CompressionMethod::Zstd { level: 1 },
            expected_ratio: 0.125,
        }
    }

    fn calculate_entropy(values: &[f32]) -> f32 {
        let mut histogram = std::collections::HashMap::new();
        let bins = 64;

        if values.is_empty() {
            return 0.0;
        }

        let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        if range == 0.0 {
            return 0.0;
        }

        for &value in values {
            let bin = ((value - min_val) / range * (bins - 1) as f32) as usize;
            let bin = bin.min(bins - 1);
            *histogram.entry(bin).or_insert(0) += 1;
        }

        let total = values.len() as f32;
        let mut entropy = 0.0;

        for count in histogram.values() {
            let probability = *count as f32 / total;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }

        entropy
    }

    fn find_dominant_patterns(values: &[f32]) -> Vec<f32> {
        let mut value_counts = std::collections::HashMap::new();

        for &value in values {
            let quantized = (value * 1000.0).round() / 1000.0;
            *value_counts.entry(quantized.to_bits()).or_insert(0) += 1;
        }

        let mut patterns: Vec<_> = value_counts.into_iter().collect();
        patterns.sort_by_key(|b| std::cmp::Reverse(b.1));

        patterns
            .into_iter()
            .take(5)
            .map(|(bits, _)| f32::from_bits(bits))
            .collect()
    }

    pub(crate) fn select_optimal_method(
        sparsity: f32,
        range: f32,
        std_dev: f32,
        entropy: f32,
        dimensions: usize,
        quality: &AdaptiveQuality,
    ) -> (CompressionMethod, f32) {
        if sparsity > 0.7 {
            return match quality {
                AdaptiveQuality::Fast => (CompressionMethod::Zstd { level: 1 }, 0.3),
                AdaptiveQuality::Balanced => (CompressionMethod::Zstd { level: 6 }, 0.2),
                AdaptiveQuality::BestRatio => (CompressionMethod::Zstd { level: 19 }, 0.15),
            };
        }

        if entropy < 2.0 {
            return match quality {
                AdaptiveQuality::Fast => (CompressionMethod::Zstd { level: 3 }, 0.4),
                AdaptiveQuality::Balanced => (CompressionMethod::Zstd { level: 9 }, 0.3),
                AdaptiveQuality::BestRatio => (CompressionMethod::Zstd { level: 22 }, 0.2),
            };
        }

        if range < 2.0 && std_dev < 0.5 {
            return match quality {
                AdaptiveQuality::Fast => (CompressionMethod::Quantization { bits: 8 }, 0.25),
                AdaptiveQuality::Balanced => (CompressionMethod::Quantization { bits: 6 }, 0.1875),
                AdaptiveQuality::BestRatio => (CompressionMethod::Quantization { bits: 4 }, 0.125),
            };
        }

        if dimensions > 128 {
            let components = match quality {
                AdaptiveQuality::Fast => dimensions * 7 / 10,
                AdaptiveQuality::Balanced => dimensions / 2,
                AdaptiveQuality::BestRatio => dimensions / 3,
            };
            return (
                CompressionMethod::Pca { components },
                components as f32 / dimensions as f32,
            );
        }

        match quality {
            AdaptiveQuality::Fast => (CompressionMethod::Zstd { level: 3 }, 0.6),
            AdaptiveQuality::Balanced => (CompressionMethod::Zstd { level: 6 }, 0.5),
            AdaptiveQuality::BestRatio => (CompressionMethod::Zstd { level: 12 }, 0.4),
        }
    }
}
