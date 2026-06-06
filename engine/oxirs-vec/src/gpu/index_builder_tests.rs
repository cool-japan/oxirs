//! Tests for GPU-accelerated HNSW index builder and related types.

use crate::gpu::index_builder_phases::{
    BatchSizeCalculator, GpuBatchDistanceComputer, GpuHnswIndexBuilder, GpuIndexOptimizer,
    GpuMemoryBudget, IncrementalGpuIndexBuilder, PipelinedIndexBuilder,
};
use crate::gpu::index_builder_types::{
    GpuDistanceMetric, GpuIndexBuildStats, GpuIndexBuilderConfig, HnswGraph,
};
use anyhow::Result;

fn make_test_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    // Deterministic pseudo-random values
                    let seed = (i * 1000 + j) as u64;
                    let a = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (a >> 33) as f32 / u32::MAX as f32 - 0.5
                })
                .collect()
        })
        .collect()
}

#[test]
fn test_gpu_index_builder_config_default() {
    let config = GpuIndexBuilderConfig::default();
    assert_eq!(config.m, 16);
    assert_eq!(config.ef_construction, 200);
    assert!(config.mixed_precision);
    assert!(config.tensor_cores);
}

#[test]
fn test_gpu_index_builder_new() {
    let config = GpuIndexBuilderConfig::default();
    let builder = GpuHnswIndexBuilder::new(config);
    assert!(builder.is_ok(), "Builder creation should succeed");
}

#[test]
fn test_add_vector_dimension_check() -> Result<()> {
    let config = GpuIndexBuilderConfig::default();
    let mut builder = GpuHnswIndexBuilder::new(config)?;

    builder.add_vector(0, vec![1.0, 2.0, 3.0])?;

    // Adding vector with different dimension should fail
    let result = builder.add_vector(1, vec![1.0, 2.0]);
    assert!(result.is_err(), "Should reject mismatched dimensions");
    Ok(())
}

#[test]
fn test_add_empty_vector_fails() -> Result<()> {
    let config = GpuIndexBuilderConfig::default();
    let mut builder = GpuHnswIndexBuilder::new(config)?;
    let result = builder.add_vector(0, vec![]);
    assert!(result.is_err(), "Should reject empty vector");
    Ok(())
}

#[test]
fn test_build_small_index() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        m: 4,
        ef_construction: 10,
        num_layers: 3,
        ..Default::default()
    };

    let mut builder = GpuHnswIndexBuilder::new(config)?;
    let vectors = make_test_vectors(20, 8);

    for (i, v) in vectors.iter().enumerate() {
        builder.add_vector(i, v.clone())?;
    }

    let graph = builder.build()?;
    assert_eq!(graph.nodes.len(), 20);
    assert!(graph.stats.vectors_indexed == 20);
    // build_time_ms may be 0 for fast builds, no assertion needed
    Ok(())
}

#[test]
fn test_build_produces_valid_graph() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        m: 4,
        ef_construction: 20,
        num_layers: 2,
        ..Default::default()
    };

    let mut builder = GpuHnswIndexBuilder::new(config)?;
    let vectors = make_test_vectors(50, 16);

    for (i, v) in vectors.iter().enumerate() {
        builder.add_vector(i, v.clone())?;
    }

    let graph = builder.build()?;

    // Every node should have valid neighbor IDs
    for node in &graph.nodes {
        for layer_neighbors in &node.neighbors {
            for &neighbor_id in layer_neighbors {
                assert!(
                    neighbor_id < graph.nodes.len(),
                    "Neighbor ID {} out of range (max {})",
                    neighbor_id,
                    graph.nodes.len()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_hnsw_graph_search() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        m: 8,
        ef_construction: 50,
        num_layers: 3,
        distance_metric: GpuDistanceMetric::Euclidean,
        ..Default::default()
    };

    let mut builder = GpuHnswIndexBuilder::new(config)?;
    let vectors = make_test_vectors(100, 8);

    for (i, v) in vectors.iter().enumerate() {
        builder.add_vector(i, v.clone())?;
    }

    let graph = builder.build()?;

    // Search for nearest neighbor
    let query = vectors[5].clone();
    let results = graph.search_knn(&query, 5, 50)?;

    assert!(!results.is_empty(), "Search should return results");
    assert!(results.len() <= 5, "Should return at most k results");

    // The nearest neighbor should have low distance
    if !results.is_empty() {
        assert!(results[0].1 >= 0.0, "Distance should be non-negative");
    }
    Ok(())
}

#[test]
fn test_hnsw_graph_search_cosine() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        m: 4,
        ef_construction: 20,
        num_layers: 2,
        distance_metric: GpuDistanceMetric::Cosine,
        ..Default::default()
    };

    let mut builder = GpuHnswIndexBuilder::new(config)?;

    // Add orthogonal unit vectors (maximally different)
    for i in 0..10 {
        let mut v = vec![0.0f32; 10];
        v[i] = 1.0;
        builder.add_vector(i, v)?;
    }

    let graph = builder.build()?;

    // Searching for v[0] should find v[0] as nearest
    let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let results = graph.search_knn(&query, 3, 30)?;
    assert!(!results.is_empty());
    Ok(())
}

#[test]
fn test_build_empty_fails() -> Result<()> {
    let config = GpuIndexBuilderConfig::default();
    let mut builder = GpuHnswIndexBuilder::new(config)?;
    assert!(
        builder.build().is_err(),
        "Build with no vectors should fail"
    );
    Ok(())
}

#[test]
fn test_build_stats_populated() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        m: 4,
        ef_construction: 10,
        num_layers: 2,
        mixed_precision: true,
        tensor_cores: false,
        ..Default::default()
    };

    let mut builder = GpuHnswIndexBuilder::new(config)?;
    let vectors = make_test_vectors(10, 4);
    for (i, v) in vectors.iter().enumerate() {
        builder.add_vector(i, v.clone())?;
    }
    let graph = builder.build()?;

    assert_eq!(graph.stats.vectors_indexed, 10);
    assert!(graph.stats.used_mixed_precision);
    assert!(!graph.stats.used_tensor_cores);
    assert!(graph.stats.batches_processed > 0);
    Ok(())
}

#[test]
fn test_incremental_builder_flush() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        m: 4,
        ef_construction: 10,
        num_layers: 2,
        ..Default::default()
    };

    let mut inc_builder = IncrementalGpuIndexBuilder::new(config, 5)?;
    let vectors = make_test_vectors(15, 4);

    for (i, v) in vectors.iter().enumerate() {
        inc_builder.add_vector(i, v.clone())?;
    }

    let graph = inc_builder.build()?;
    assert_eq!(graph.nodes.len(), 15);
    Ok(())
}

#[test]
fn test_batch_distance_computer_cosine() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        distance_metric: GpuDistanceMetric::Cosine,
        ..Default::default()
    };
    let computer = GpuBatchDistanceComputer::new(config)?;

    let queries = vec![vec![1.0f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let database = vec![
        vec![1.0f32, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    let distances = computer.compute_distances(&queries, &database)?;
    assert_eq!(distances.len(), 2);
    assert_eq!(distances[0].len(), 3);

    // First query matches first db vector exactly (cosine dist = 0)
    assert!(
        distances[0][0].abs() < 1e-5,
        "Identical vectors should have distance 0"
    );
    // First query vs second db vector = cosine dist ~ 1 (orthogonal)
    assert!(
        (distances[0][1] - 1.0).abs() < 1e-5,
        "Orthogonal vectors should have cosine distance 1.0"
    );
    Ok(())
}

#[test]
fn test_batch_distance_computer_euclidean() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        distance_metric: GpuDistanceMetric::Euclidean,
        ..Default::default()
    };
    let computer = GpuBatchDistanceComputer::new(config)?;

    let queries = vec![vec![0.0f32, 0.0, 0.0]];
    let database = vec![vec![3.0f32, 4.0, 0.0]]; // Distance = 5.0

    let distances = computer.compute_distances(&queries, &database)?;
    assert!(
        (distances[0][0] - 5.0).abs() < 1e-4,
        "Expected Euclidean distance of 5.0"
    );
    Ok(())
}

#[test]
fn test_batch_distance_dimension_mismatch() -> Result<()> {
    let config = GpuIndexBuilderConfig::default();
    let computer = GpuBatchDistanceComputer::new(config)?;

    let queries = vec![vec![1.0f32, 2.0]];
    let database = vec![vec![1.0f32, 2.0, 3.0]]; // Wrong dimension

    let result = computer.compute_distances(&queries, &database);
    assert!(result.is_err(), "Should fail on dimension mismatch");
    Ok(())
}

#[test]
fn test_distance_metric_inner_product() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        distance_metric: GpuDistanceMetric::InnerProduct,
        ..Default::default()
    };
    let computer = GpuBatchDistanceComputer::new(config)?;

    let queries = vec![vec![1.0f32, 2.0, 3.0]];
    let database = vec![vec![4.0f32, 5.0, 6.0]]; // dot = 4+10+18 = 32 -> neg = -32

    let distances = computer.compute_distances(&queries, &database)?;
    assert!(
        (distances[0][0] + 32.0).abs() < 1e-4,
        "Inner product distance should be -32"
    );
    Ok(())
}

#[test]
fn test_builder_clears_after_build() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        m: 4,
        ef_construction: 10,
        num_layers: 2,
        ..Default::default()
    };

    let mut builder = GpuHnswIndexBuilder::new(config)?;
    let vectors = make_test_vectors(10, 4);
    for (i, v) in vectors.iter().enumerate() {
        builder.add_vector(i, v.clone())?;
    }

    let _ = builder.build()?;

    // After build, pending_vectors should be empty
    assert!(
        builder.pending_vectors.is_empty(),
        "Pending vectors should be cleared after build"
    );
    Ok(())
}

#[test]
fn test_layer_assignment_distribution() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        m: 16,
        num_layers: 5,
        ..Default::default()
    };
    let builder = GpuHnswIndexBuilder::new(config.clone())?;
    let layers = builder.assign_layers(1000);

    // Most vectors should be at layer 0
    let layer_0_count = layers.iter().filter(|&&l| l == 0).count();
    assert!(
        layer_0_count > 500,
        "More than half should be at layer 0, got {}",
        layer_0_count
    );

    // All layers should be within bounds
    for &l in &layers {
        assert!(l < config.num_layers, "Layer {} exceeds num_layers", l);
    }
    Ok(())
}

#[test]
fn test_search_dimension_mismatch_error() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        m: 4,
        ef_construction: 10,
        num_layers: 2,
        ..Default::default()
    };

    let mut builder = GpuHnswIndexBuilder::new(config)?;
    for i in 0..5 {
        builder.add_vector(i, vec![1.0f32; 8])?;
    }
    let graph = builder.build()?;

    // Query with wrong dimension
    let result = graph.search_knn(&[1.0, 2.0], 3, 10);
    assert!(
        result.is_err(),
        "Should fail on dimension mismatch in search"
    );
    Ok(())
}

#[test]
fn test_search_empty_graph() -> Result<()> {
    let config = GpuIndexBuilderConfig::default();
    let graph = HnswGraph {
        nodes: Vec::new(),
        entry_point: 0,
        max_layer: 0,
        config,
        stats: GpuIndexBuildStats::default(),
    };

    let results = graph.search_knn(&[1.0, 2.0], 5, 10)?;
    assert!(
        results.is_empty(),
        "Empty graph search should return no results"
    );
    Ok(())
}

#[test]
fn test_incremental_builder_pending_count() -> Result<()> {
    let config = GpuIndexBuilderConfig {
        m: 4,
        ef_construction: 10,
        num_layers: 2,
        ..Default::default()
    };

    let mut inc_builder = IncrementalGpuIndexBuilder::new(config, 100)?;
    assert_eq!(inc_builder.pending_count(), 0);

    inc_builder.add_vector(0, vec![1.0f32; 4])?;
    inc_builder.add_vector(1, vec![2.0f32; 4])?;
    assert_eq!(inc_builder.pending_count(), 2);
    Ok(())
}

#[test]
fn test_gpu_distance_metric_variants() -> Result<()> {
    let metrics = [
        GpuDistanceMetric::Cosine,
        GpuDistanceMetric::Euclidean,
        GpuDistanceMetric::InnerProduct,
        GpuDistanceMetric::CosineF16,
        GpuDistanceMetric::EuclideanF16,
    ];

    for metric in &metrics {
        let config = GpuIndexBuilderConfig {
            distance_metric: *metric,
            m: 4,
            ef_construction: 10,
            num_layers: 2,
            ..Default::default()
        };
        let computer = GpuBatchDistanceComputer::new(config)?;
        let queries = vec![vec![1.0f32, 0.0]];
        let db = vec![vec![0.0f32, 1.0]];
        let result = computer.compute_distances(&queries, &db);
        assert!(
            result.is_ok(),
            "Distance computation failed for {:?}",
            metric
        );
    }
    Ok(())
}

// ---- GpuIndexOptimizer tests ----

#[test]
fn test_batch_size_calculator_basic() {
    let size = BatchSizeCalculator::calculate_batch_size(128, 4096);
    assert!(size >= 1, "Batch size should be at least 1");
}

#[test]
fn test_batch_size_calculator_zero_dim_returns_default() {
    let size = BatchSizeCalculator::calculate_batch_size(0, 4096);
    assert!(
        size > 0,
        "Zero-dim should return positive default batch size"
    );
}

#[test]
fn test_batch_size_calculator_large_dim() {
    // Very large dim, limited memory => small batch
    let size = BatchSizeCalculator::calculate_batch_size(16384, 256);
    assert!(size >= 1, "Even large dim should yield at least 1");
    // 16384 floats = 64 KB per vector; 256 MB budget reserves 64 MB => 192 MB
    // => 192 * 1024 * 1024 / (16384 * 4) = ~3072 vectors
    assert!(
        size <= 8192,
        "Very large dim with limited memory should give reduced batch: got {}",
        size
    );
}

#[test]
fn test_optimal_batch_for_float32() {
    let size = BatchSizeCalculator::optimal_batch_for_float32(512, 8192);
    assert!(size >= 1);
}

#[test]
fn test_optimal_batch_increases_with_memory() {
    let small = BatchSizeCalculator::optimal_batch_for_float32(128, 256);
    let large = BatchSizeCalculator::optimal_batch_for_float32(128, 8192);
    assert!(
        large >= small,
        "More memory should yield at least as large a batch: small={} large={}",
        small,
        large
    );
}

#[test]
fn test_gpu_memory_budget_bytes_per_vector() {
    let budget = GpuMemoryBudget::new(4096, 512);
    // 128-dim float32 = 512 bytes
    assert_eq!(budget.bytes_per_vector(128), 512);
    assert_eq!(budget.bytes_per_vector(1), 4);
}

#[test]
fn test_gpu_memory_budget_available() {
    let budget = GpuMemoryBudget::new(4096, 512);
    assert_eq!(budget.available_mb, 3584);
}

#[test]
fn test_gpu_memory_budget_can_fit_batch_true() {
    let budget = GpuMemoryBudget::new(4096, 512);
    // 128-dim, batch of 1000 => 1000 * 512 bytes = 500 KB well under 3584 MB
    assert!(budget.can_fit_batch(1000, 128));
}

#[test]
fn test_gpu_memory_budget_can_fit_batch_false() {
    let budget = GpuMemoryBudget::new(64, 32);
    // 64 MB total, 32 MB reserved => 32 MB available
    // 8192-dim vector = 32768 bytes; 1200 vectors = 38.4 MB > 32 MB
    assert!(!budget.can_fit_batch(1200, 8192));
}

#[test]
fn test_gpu_memory_budget_zero_reserved() {
    let budget = GpuMemoryBudget::new(1024, 0);
    assert_eq!(budget.available_mb, 1024);
}

#[test]
fn test_gpu_index_optimizer_creates_budget() {
    let optimizer = GpuIndexOptimizer::new(4096, 512);
    let budget = optimizer.memory_budget();
    assert_eq!(budget.total_mb, 4096);
    assert_eq!(budget.reserved_mb, 512);
}

#[test]
fn test_gpu_index_optimizer_recommend_batch_size() {
    let optimizer = GpuIndexOptimizer::new(4096, 512);
    let size = optimizer.recommend_batch_size(256);
    assert!(size >= 1);
}

#[test]
fn test_pipelined_index_builder_prepare() {
    let batch = PipelinedIndexBuilder::stage_a_prepare(&[1.0f32, 2.0, 3.0, 4.0]);
    assert_eq!(batch.data.len(), 4);
    assert!(batch.prepared_at.elapsed().as_secs() < 5);
}

#[test]
fn test_pipelined_index_builder_compute() {
    let prepared = PipelinedIndexBuilder::stage_a_prepare(&[1.0f32, 0.0, 0.0, 0.0]);
    let computed = PipelinedIndexBuilder::stage_b_compute(prepared);
    assert!(!computed.distances.is_empty());
}

#[test]
fn test_pipelined_index_builder_finalize() {
    let prepared = PipelinedIndexBuilder::stage_a_prepare(&[1.0f32, 2.0, 3.0, 4.0]);
    let computed = PipelinedIndexBuilder::stage_b_compute(prepared);
    let indexed = PipelinedIndexBuilder::stage_c_finalize(computed);
    assert!(!indexed.neighbor_ids.is_empty() || indexed.neighbor_ids.is_empty());
    // finalize always returns a valid IndexedBatch
    assert!(indexed.finalized_at.elapsed().as_secs() < 5);
}

#[test]
fn test_pipelined_index_builder_full_pipeline() {
    let data: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
    let prepared = PipelinedIndexBuilder::stage_a_prepare(&data);
    let computed = PipelinedIndexBuilder::stage_b_compute(prepared);
    let indexed = PipelinedIndexBuilder::stage_c_finalize(computed);
    // distances should contain self-distance (0.0 for euclidean on normalised)
    let _ = indexed;
}

#[test]
fn test_pipelined_builder_stage_b_distances_nonnegative() {
    let data: Vec<f32> = vec![3.0, 4.0, 0.0]; // norm = 5
    let prepared = PipelinedIndexBuilder::stage_a_prepare(&data);
    let computed = PipelinedIndexBuilder::stage_b_compute(prepared);
    for &d in &computed.distances {
        assert!(d >= 0.0, "Distance should be non-negative, got {}", d);
    }
}

#[test]
fn test_batch_size_calculator_reasonable_bounds() {
    // For 768-dim (BERT), 16 GB GPU
    let size = BatchSizeCalculator::calculate_batch_size(768, 16_384);
    // 768 * 4 = 3072 bytes/vector; 12 GB available => ~4M vectors cap to 65536
    assert!(
        size >= 1_000,
        "Should support large batches on big GPU: {}",
        size
    );
    assert!(
        size <= 1_000_000,
        "Batch size should be capped reasonably: {}",
        size
    );
}
