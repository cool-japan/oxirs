//! Streaming compress/decompress, block I/O, AdaptiveCompressor, and factory.

use super::compression_codecs::{
    NoOpCompressor, PcaCompressor, ProductQuantizer, ScalarQuantizer, ZstdCompressor,
};
use super::compression_types::{
    AdaptiveQuality, CompressionMethod, CompressionMetrics, VectorAnalysis, VectorCompressor,
};
use crate::{Vector, VectorError};

// ─────────────────────────────────────────────────────────────────────────────
// Method equivalence check
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn methods_equivalent(method1: &CompressionMethod, method2: &CompressionMethod) -> bool {
    match (method1, method2) {
        (CompressionMethod::None, CompressionMethod::None) => true,
        (CompressionMethod::Zstd { level: l1 }, CompressionMethod::Zstd { level: l2 }) => {
            (l1 - l2).abs() <= 2
        }
        (
            CompressionMethod::Quantization { bits: b1 },
            CompressionMethod::Quantization { bits: b2 },
        ) => b1 == b2,
        (CompressionMethod::Pca { components: c1 }, CompressionMethod::Pca { components: c2 }) => {
            ((*c1 as i32) - (*c2 as i32)).abs() <= (*c1 as i32) / 10
        }
        _ => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────

/// Create a boxed compressor from the given method description
pub fn create_compressor(method: &CompressionMethod) -> Box<dyn VectorCompressor> {
    match method {
        CompressionMethod::None => Box::new(NoOpCompressor),
        CompressionMethod::Zstd { level } => Box::new(ZstdCompressor::new(*level)),
        CompressionMethod::Quantization { bits } => Box::new(ScalarQuantizer::new(*bits)),
        CompressionMethod::Pca { components } => Box::new(PcaCompressor::new(*components)),
        CompressionMethod::ProductQuantization {
            subvectors,
            codebook_size,
        } => Box::new(ProductQuantizer::new(*subvectors, *codebook_size)),
        CompressionMethod::Adaptive {
            quality_level,
            analysis_samples,
        } => Box::new(AdaptiveCompressor::new(
            quality_level.clone(),
            *analysis_samples,
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveCompressor
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptive compressor that automatically selects the best compression method
pub struct AdaptiveCompressor {
    pub(crate) quality_level: AdaptiveQuality,
    pub(crate) analysis_samples: usize,
    pub(crate) current_method: Option<Box<dyn VectorCompressor>>,
    pub(crate) analysis_cache: Option<VectorAnalysis>,
    pub(crate) performance_metrics: CompressionMetrics,
}

impl AdaptiveCompressor {
    pub fn new(quality_level: AdaptiveQuality, analysis_samples: usize) -> Self {
        Self {
            quality_level,
            analysis_samples: analysis_samples.max(10),
            current_method: None,
            analysis_cache: None,
            performance_metrics: CompressionMetrics::default(),
        }
    }

    pub fn with_fast_quality() -> Self {
        Self::new(AdaptiveQuality::Fast, 50)
    }

    pub fn with_balanced_quality() -> Self {
        Self::new(AdaptiveQuality::Balanced, 100)
    }

    pub fn with_best_ratio() -> Self {
        Self::new(AdaptiveQuality::BestRatio, 200)
    }

    /// Analyze sample vectors and optimize compression method
    pub fn optimize_for_vectors(&mut self, sample_vectors: &[Vector]) -> Result<(), VectorError> {
        if sample_vectors.is_empty() {
            return Ok(());
        }

        let start_time = std::time::Instant::now();

        let samples_to_analyze = sample_vectors.len().min(self.analysis_samples);
        let analysis_vectors = &sample_vectors[..samples_to_analyze];

        let analysis = VectorAnalysis::analyze(analysis_vectors, &self.quality_level)?;

        let should_switch = match (&self.current_method, &self.analysis_cache) {
            (Some(_), Some(cached)) => {
                !methods_equivalent(&cached.recommended_method, &analysis.recommended_method)
            }
            _ => true,
        };

        if should_switch {
            self.current_method = Some(create_compressor(&analysis.recommended_method));
            self.performance_metrics.method_switches += 1;
        }

        self.analysis_cache = Some(analysis);

        let analysis_time = start_time.elapsed().as_secs_f64() * 1000.0;
        tracing::debug!("Adaptive compression analysis took {:.2}ms", analysis_time);

        Ok(())
    }

    pub fn get_metrics(&self) -> &CompressionMetrics {
        &self.performance_metrics
    }

    pub fn get_analysis(&self) -> Option<&VectorAnalysis> {
        self.analysis_cache.as_ref()
    }

    /// Adaptively re-analyze and potentially switch compression method
    pub fn adaptive_reanalysis(&mut self, recent_vectors: &[Vector]) -> Result<bool, VectorError> {
        if recent_vectors.len() < self.analysis_samples / 4 {
            return Ok(false);
        }

        let old_method = self
            .analysis_cache
            .as_ref()
            .map(|a| a.recommended_method.clone());

        self.optimize_for_vectors(recent_vectors)?;

        let method_changed = match (old_method, &self.analysis_cache) {
            (Some(old), Some(new)) => !methods_equivalent(&old, &new.recommended_method),
            _ => false,
        };

        Ok(method_changed)
    }
}

impl VectorCompressor for AdaptiveCompressor {
    fn compress(&self, vector: &Vector) -> Result<Vec<u8>, VectorError> {
        if let Some(compressor) = &self.current_method {
            let start = std::time::Instant::now();
            let result = compressor.compress(vector);
            let _compression_time = start.elapsed().as_secs_f64() * 1000.0;
            result
        } else {
            let no_op = NoOpCompressor;
            no_op.compress(vector)
        }
    }

    fn decompress(&self, data: &[u8], dimensions: usize) -> Result<Vector, VectorError> {
        if let Some(compressor) = &self.current_method {
            let start = std::time::Instant::now();
            let result = compressor.decompress(data, dimensions);
            let _decompression_time = start.elapsed().as_secs_f64() * 1000.0;
            result
        } else {
            let no_op = NoOpCompressor;
            no_op.decompress(data, dimensions)
        }
    }

    fn compression_ratio(&self) -> f32 {
        if let Some(compressor) = &self.current_method {
            compressor.compression_ratio()
        } else {
            1.0
        }
    }
}
