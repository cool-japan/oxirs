//! Tests for the advanced benchmarking framework.

#[cfg(test)]
mod tests {
    use crate::bench_metrics::{
        AdvancedBenchmarkConfig, BuildTimeMetrics, IndexSizeMetrics, LatencyMetrics,
        PerformanceMetrics, ThroughputMetrics,
    };
    use crate::bench_runner::{AdvancedBenchmarkSuite, StatisticalAnalyzer};
    use crate::Vector;
    use anyhow::Result;
    use std::collections::HashMap;

    #[test]
    fn test_advanced_benchmark_config() {
        let config = AdvancedBenchmarkConfig::new();
        assert_eq!(config.confidence_level, 0.95);
        assert_eq!(config.min_runs, 10);

        let ann_config = AdvancedBenchmarkConfig::ann_benchmarks_compatible();
        assert!(ann_config.ann_benchmarks_mode);
    }

    #[test]
    fn test_dataset_analysis() -> Result<()> {
        let config = AdvancedBenchmarkConfig::new();
        let suite = AdvancedBenchmarkSuite::new(config);

        let vectors = vec![
            Vector::new(vec![1.0, 0.0, 0.0]),
            Vector::new(vec![0.0, 1.0, 0.0]),
            Vector::new(vec![0.0, 0.0, 1.0]),
        ];

        let stats = suite.compute_dataset_statistics(&vectors)?;
        assert_eq!(stats.vector_count, 3);
        assert_eq!(stats.dimensions, 3);
        assert!(stats.mean_magnitude > 0.0);
        Ok(())
    }

    #[test]
    fn test_statistical_analyzer() -> Result<()> {
        let analyzer = StatisticalAnalyzer::new(0.95, 10, 2.0);

        let latency = LatencyMetrics {
            mean_ms: 1.0,
            std_ms: 0.1,
            percentiles: HashMap::new(),
            distribution: vec![
                0.9, 1.0, 1.1, 0.95, 1.05, 0.98, 1.02, 0.92, 1.08, 0.97, 1.03,
            ],
            max_ms: 1.1,
            min_ms: 0.9,
        };

        let performance = PerformanceMetrics {
            latency,
            throughput: ThroughputMetrics {
                qps: 1000.0,
                batch_qps: HashMap::new(),
                concurrent_qps: HashMap::new(),
                saturation_qps: 1200.0,
            },
            build_time: BuildTimeMetrics {
                total_seconds: 10.0,
                per_vector_ms: 0.1,
                allocation_seconds: 1.0,
                construction_seconds: 8.0,
                optimization_seconds: 1.0,
            },
            index_size: IndexSizeMetrics {
                total_bytes: 1024,
                per_vector_bytes: 100.0,
                overhead_ratio: 0.2,
                compression_ratio: 0.8,
                serialized_bytes: 800,
            },
        };

        let stats = analyzer.analyze_metrics(&performance)?;
        assert_eq!(stats.sample_size, 11);
        assert!(stats.confidence_intervals.contains_key("mean_latency_ms"));
        Ok(())
    }
}
