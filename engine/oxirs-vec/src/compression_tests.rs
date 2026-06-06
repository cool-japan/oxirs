//! Tests for compression codecs and adaptive compression.

#[cfg(test)]
mod tests {
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    use crate::compression::{
        AdaptiveCompressor, AdaptiveQuality, CompressionMethod, PcaCompressor, ProductQuantizer,
        ScalarQuantizer, VectorAnalysis, VectorCompressor, ZstdCompressor,
    };
    use crate::compression_io::methods_equivalent;
    use crate::{Vector, VectorError};

    #[test]
    fn test_zstd_compression() -> Result<()> {
        let vector = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let compressor = ZstdCompressor::new(3);

        let compressed = compressor.compress(&vector)?;
        let decompressed = compressor.decompress(&compressed, 5)?;

        let orig = vector.as_f32();
        let dec = decompressed.as_f32();
        assert_eq!(orig.len(), dec.len());
        for (a, b) in orig.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_scalar_quantization() -> Result<()> {
        let vector = Vector::new(vec![0.1, 0.5, 0.9, 0.3, 0.7]);
        let mut quantizer = ScalarQuantizer::new(8);
        quantizer.train(std::slice::from_ref(&vector))?;

        let compressed = quantizer.compress(&vector)?;
        let decompressed = quantizer.decompress(&compressed, 5)?;

        assert!(compressed.len() < 20);

        let orig = vector.as_f32();
        let dec = decompressed.as_f32();
        assert_eq!(orig.len(), dec.len());
        for (a, b) in orig.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 0.01);
        }
        Ok(())
    }

    #[test]
    fn test_pca_compression() -> Result<()> {
        let vectors = vec![
            Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            Vector::new(vec![2.0, 3.0, 4.0, 5.0, 6.0]),
            Vector::new(vec![3.0, 4.0, 5.0, 6.0, 7.0]),
        ];

        let mut pca = PcaCompressor::new(3);
        pca.train(&vectors)?;

        let compressed = pca.compress(&vectors[0])?;
        let decompressed = pca.decompress(&compressed, 5)?;

        let dec = decompressed.as_f32();
        assert_eq!(dec.len(), 5);
        Ok(())
    }

    #[test]
    fn test_adaptive_compression_sparse_data() -> Result<()> {
        let vectors = vec![
            Vector::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
            Vector::new(vec![0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
        ];

        let mut adaptive = AdaptiveCompressor::with_balanced_quality();
        adaptive.optimize_for_vectors(&vectors)?;

        let analysis = adaptive
            .get_analysis()
            .ok_or("get_analysis returned None")?;
        assert!(analysis.sparsity > 0.5);

        let compressed = adaptive.compress(&vectors[0])?;
        let decompressed = adaptive.decompress(&compressed, 10)?;

        let orig = vectors[0].as_f32();
        let dec = decompressed.as_f32();
        assert_eq!(orig.len(), dec.len());
        Ok(())
    }

    #[test]
    fn test_adaptive_compression_quantizable_data() -> Result<()> {
        let vectors = vec![
            Vector::new(vec![0.1, 0.2, 0.3, 0.4, 0.5]),
            Vector::new(vec![0.2, 0.3, 0.4, 0.5, 0.6]),
            Vector::new(vec![0.3, 0.4, 0.5, 0.6, 0.7]),
        ];

        let mut adaptive = AdaptiveCompressor::with_balanced_quality();
        adaptive.optimize_for_vectors(&vectors)?;

        let analysis = adaptive
            .get_analysis()
            .ok_or("get_analysis returned None")?;
        assert!(analysis.range < 1.0);

        let compressed = adaptive.compress(&vectors[0])?;
        let decompressed = adaptive.decompress(&compressed, 5)?;

        let orig = vectors[0].as_f32();
        let dec = decompressed.as_f32();
        assert_eq!(orig.len(), dec.len());

        assert!(adaptive.compression_ratio() < 0.5);
        Ok(())
    }

    #[test]
    fn test_adaptive_compression_high_dimensional() -> Result<()> {
        let mut vectors = Vec::new();
        for i in 0..10 {
            let mut data = vec![0.0; 200];
            for (j, item) in data.iter_mut().enumerate().take(200) {
                *item = (i * j) as f32 * 0.01;
            }
            vectors.push(Vector::new(data));
        }

        let mut adaptive = AdaptiveCompressor::with_best_ratio();
        adaptive.optimize_for_vectors(&vectors)?;

        let analysis = adaptive
            .get_analysis()
            .ok_or("get_analysis returned None")?;

        match &analysis.recommended_method {
            CompressionMethod::Pca { components } => {
                assert!(*components < 200);
            }
            _ => {
                assert!(matches!(
                    analysis.recommended_method,
                    CompressionMethod::Pca { .. }
                        | CompressionMethod::Quantization { .. }
                        | CompressionMethod::Zstd { .. }
                ));
            }
        }

        let original = &vectors[0];
        println!("Original vector length: {}", original.dimensions);
        println!("Recommended method: {:?}", analysis.recommended_method);

        let compressed = adaptive.compress(original)?;
        println!("Compressed size: {} bytes", compressed.len());

        assert!(!compressed.is_empty());
        assert!(compressed.len() < original.dimensions * 4);

        match &analysis.recommended_method {
            CompressionMethod::Pca { components } => {
                assert!(*components < original.dimensions);
                println!(
                    "PCA compression: {} → {} components",
                    original.dimensions, components
                );
            }
            _ => {
                let decompressed = adaptive.decompress(&compressed, original.dimensions)?;
                let dec = decompressed.as_f32();
                let orig = original.as_f32();
                assert_eq!(dec.len(), orig.len());
            }
        }
        Ok(())
    }

    #[test]
    fn test_adaptive_compression_method_switching() -> Result<()> {
        let mut adaptive = AdaptiveCompressor::with_fast_quality();

        let sparse_vectors = vec![
            Vector::new(vec![0.0, 0.0, 1.0, 0.0, 0.0]),
            Vector::new(vec![0.0, 2.0, 0.0, 0.0, 0.0]),
        ];
        adaptive.optimize_for_vectors(&sparse_vectors)?;
        let initial_switches = adaptive.get_metrics().method_switches;

        let dense_vectors = vec![
            Vector::new(vec![0.1, 0.2, 0.3, 0.4, 0.5]),
            Vector::new(vec![0.2, 0.3, 0.4, 0.5, 0.6]),
        ];
        adaptive.optimize_for_vectors(&dense_vectors)?;

        assert!(adaptive.get_metrics().method_switches > initial_switches);
        Ok(())
    }

    #[test]
    fn test_vector_analysis() -> Result<()> {
        let vectors = vec![
            Vector::new(vec![1.0, 2.0, 3.0]),
            Vector::new(vec![2.0, 3.0, 4.0]),
            Vector::new(vec![3.0, 4.0, 5.0]),
        ];

        let analysis = VectorAnalysis::analyze(&vectors, &AdaptiveQuality::Balanced)?;

        assert!(analysis.mean > 0.0);
        assert!(analysis.std_dev > 0.0);
        assert!(analysis.range > 0.0);
        assert!(analysis.entropy >= 0.0);
        assert!(!analysis.dominant_patterns.is_empty());
        assert!(analysis.expected_ratio > 0.0 && analysis.expected_ratio <= 1.0);
        Ok(())
    }

    #[test]
    fn test_compression_method_equivalence() {
        assert!(methods_equivalent(
            &CompressionMethod::Zstd { level: 5 },
            &CompressionMethod::Zstd { level: 6 }
        ));

        assert!(!methods_equivalent(
            &CompressionMethod::Zstd { level: 1 },
            &CompressionMethod::Zstd { level: 10 }
        ));

        assert!(methods_equivalent(
            &CompressionMethod::Quantization { bits: 8 },
            &CompressionMethod::Quantization { bits: 8 }
        ));

        assert!(!methods_equivalent(
            &CompressionMethod::Zstd { level: 5 },
            &CompressionMethod::Quantization { bits: 8 }
        ));
    }

    #[test]
    fn test_adaptive_compressor_convenience_constructors() {
        let fast = AdaptiveCompressor::with_fast_quality();
        assert!(matches!(fast.quality_level, AdaptiveQuality::Fast));

        let balanced = AdaptiveCompressor::with_balanced_quality();
        assert!(matches!(balanced.quality_level, AdaptiveQuality::Balanced));

        let best = AdaptiveCompressor::with_best_ratio();
        assert!(matches!(best.quality_level, AdaptiveQuality::BestRatio));
    }

    #[test]
    fn test_product_quantization() -> Result<()> {
        let vectors = vec![
            Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            Vector::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            Vector::new(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            Vector::new(vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
        ];

        let mut pq = ProductQuantizer::new(4, 4);
        pq.train(&vectors)?;

        let original = &vectors[0];
        let compressed = pq.compress(original)?;
        let decompressed = pq.decompress(&compressed, 8)?;

        assert_eq!(decompressed.dimensions, original.dimensions);

        let ratio = pq.compression_ratio();
        assert!(
            ratio > 0.0 && ratio < 1.0,
            "Compression ratio should be between 0 and 1, got {ratio}"
        );

        for vector in &vectors {
            let compressed = pq.compress(vector)?;
            let decompressed = pq.decompress(&compressed, vector.dimensions)?;
            assert_eq!(decompressed.dimensions, vector.dimensions);
        }
        Ok(())
    }

    #[test]
    fn test_product_quantization_invalid_dimensions() {
        let vectors = vec![Vector::new(vec![1.0, 2.0, 3.0])];

        let mut pq = ProductQuantizer::new(4, 4);
        let result = pq.train(&vectors);

        assert!(result.is_err());
        if let Err(VectorError::InvalidDimensions(_)) = result {
            // Expected error type
        } else {
            panic!("Expected InvalidDimensions error");
        }
    }
}
