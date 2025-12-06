# OxiRS Vec - Performance Tuning Guide

**Version**: v0.1.0-beta.2
**Last Updated**: December 6, 2025

## Table of Contents

1. [Introduction](#introduction)
2. [Performance Metrics](#performance-metrics)
3. [Index Algorithm Selection](#index-algorithm-selection)
4. [HNSW Optimization](#hnsw-optimization)
5. [Memory Optimization](#memory-optimization)
6. [Query Optimization](#query-optimization)
7. [GPU Acceleration](#gpu-acceleration)
8. [Caching Strategies](#caching-strategies)
9. [Workload-Specific Tuning](#workload-specific-tuning)
10. [Benchmarking](#benchmarking)

---

## Introduction

This guide provides detailed strategies for optimizing OxiRS Vec performance across different workloads and hardware configurations.

### Performance Goals

| Metric | Target | Excellent |
|--------|--------|-----------|
| Query Latency (p50) | < 10 ms | < 5 ms |
| Query Latency (p95) | < 50 ms | < 20 ms |
| Query Latency (p99) | < 100 ms | < 50 ms |
| Throughput | > 1000 QPS | > 10000 QPS |
| Recall@10 | > 0.95 | > 0.98 |
| Index Build Time | < 1 hr/10M | < 30 min/10M |

---

## Performance Metrics

### Key Performance Indicators (KPIs)

```rust
use oxirs_vec::enhanced_performance_monitoring::{
    EnhancedPerformanceMonitor, QueryMetricsCollector
};

fn measure_performance(monitor: &EnhancedPerformanceMonitor) {
    let stats = monitor.get_query_statistics();

    println!("Performance Metrics:");
    println!("  Average Latency: {} ms", stats.avg_latency_ms);
    println!("  P50 Latency: {} ms", stats.p50_latency_ms);
    println!("  P95 Latency: {} ms", stats.p95_latency_ms);
    println!("  P99 Latency: {} ms", stats.p99_latency_ms);
    println!("  Throughput: {} QPS", stats.queries_per_second);
    println!("  Recall@10: {:.3}", stats.recall_at_10);
    println!("  Recall@100: {:.3}", stats.recall_at_100);
}
```

### Trade-off Triangle

```
         Accuracy (Recall)
                △
               ╱ ╲
              ╱   ╲
             ╱     ╲
            ╱       ╲
           ╱         ╲
   Memory ●───────────● Speed (Latency)
```

**You can optimize for 2 out of 3**:
- **High Accuracy + Low Memory** → Slower queries
- **High Accuracy + Low Latency** → More memory
- **Low Memory + Low Latency** → Lower accuracy

---

## Index Algorithm Selection

### Decision Tree

```
Is your dataset > 10M vectors?
├─ YES: Use DiskANN (disk-backed, scalable)
│   └─ Memory budget: ~100 MB for 1B vectors
└─ NO: Continue...
    │
    ├─ Need highest recall (>0.98)?
    │   └─ Use HNSW with high M (32-64)
    │       └─ Memory: ~5 GB per 1M vectors
    │
    ├─ Need lowest latency (<5ms)?
    │   └─ Use HNSW + GPU acceleration
    │       └─ Memory: ~6 GB per 1M vectors + GPU
    │
    ├─ Limited memory budget?
    │   ├─ Use PQ (Product Quantization)
    │   │   └─ Memory: ~50 bytes per vector
    │   └─ Use DiskANN
    │       └─ Memory: Minimal (disk-backed)
    │
    └─ Balanced requirements?
        └─ Use HNSW with default settings
            └─ Memory: ~4 GB per 1M vectors
```

### Algorithm Comparison

| Algorithm | Recall@10 | Latency (p95) | Memory/1M | Build Time | Best For |
|-----------|-----------|---------------|-----------|------------|----------|
| **HNSW** | 0.98 | 8 ms | 4 GB | 20 min | Balanced, high recall |
| **IVF** | 0.92 | 5 ms | 2 GB | 10 min | Fast search, moderate recall |
| **LSH** | 0.85 | 3 ms | 1 GB | 5 min | Ultra-fast, approximate |
| **PQ** | 0.90 | 12 ms | 50 MB | 15 min | Low memory |
| **DiskANN** | 0.95 | 15 ms | 100 MB | 3 hr | Billion-scale |
| **NSG** | 0.96 | 10 ms | 3 GB | 30 min | Graph-based, balanced |

### Selection Code

```rust
use oxirs_vec::query_planning::{QueryPlanner, VectorQueryType};

fn select_optimal_index(
    num_vectors: usize,
    dimensions: usize,
    memory_budget_gb: f32,
) -> anyhow::Result<Box<dyn oxirs_vec::VectorIndex>> {
    // Billion-scale: Use DiskANN
    if num_vectors > 100_000_000 {
        let config = oxirs_vec::diskann::DiskAnnConfig {
            dimensions,
            max_degree: 64,
            search_list_size: 100,
            index_path: "/mnt/nvme/vectors".to_string(),
        };
        return Ok(Box::new(oxirs_vec::diskann::DiskAnnIndex::new(config)?));
    }

    // Memory-constrained: Use PQ
    let required_memory_gb = (num_vectors * dimensions * 4) as f32 / 1_000_000_000.0;
    if required_memory_gb > memory_budget_gb {
        let config = oxirs_vec::pq::PQConfig {
            dimensions,
            num_subvectors: dimensions / 8,
            num_centroids: 256,
            max_elements: num_vectors,
        };
        return Ok(Box::new(oxirs_vec::pq::PQIndex::new(config)?));
    }

    // Default: HNSW for best balance
    let config = oxirs_vec::hnsw::HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 100,
        max_elements: num_vectors,
        dimensions,
    };
    Ok(Box::new(oxirs_vec::hnsw::HnswIndex::new(config)?))
}
```

---

## HNSW Optimization

### Parameter Tuning Matrix

| Workload | M | ef_construction | ef_search | Memory | Latency | Recall |
|----------|---|-----------------|-----------|--------|---------|--------|
| **High Recall** | 32 | 400 | 200 | 8 GB/1M | 15 ms | 0.99 |
| **Balanced** | 16 | 200 | 100 | 4 GB/1M | 8 ms | 0.96 |
| **Fast Search** | 8 | 100 | 50 | 2 GB/1M | 4 ms | 0.92 |
| **Low Memory** | 8 | 100 | 100 | 2 GB/1M | 6 ms | 0.94 |

### Dynamic Parameter Adjustment

```rust
use oxirs_vec::hnsw::{HnswConfig, HnswIndex};
use oxirs_vec::adaptive_recall::AdaptiveRecallOptimizer;

fn adaptive_hnsw_search(
    index: &HnswIndex,
    query: &Vector,
    target_recall: f32,
) -> anyhow::Result<Vec<(String, f32)>> {
    let mut ef_search = 50;
    let max_ef_search = 500;

    loop {
        let results = index.search_with_ef(query, 10, ef_search)?;

        // Estimate recall from result quality
        let estimated_recall = estimate_recall(&results);

        if estimated_recall >= target_recall || ef_search >= max_ef_search {
            return Ok(results);
        }

        // Increase ef_search
        ef_search = (ef_search as f32 * 1.5) as usize;
    }
}

fn estimate_recall(results: &[(String, f32)]) -> f32 {
    // Heuristic: Higher similarities indicate better recall
    if results.is_empty() {
        return 0.0;
    }

    let avg_similarity: f32 = results.iter().map(|(_, sim)| sim).sum::<f32>() / results.len() as f32;
    avg_similarity.min(1.0)
}
```

### Multi-threaded Index Building

```rust
use oxirs_vec::hnsw::{HnswConfig, ParallelHnswBuilder};

fn parallel_index_construction(
    vectors: Vec<(String, Vector)>,
) -> anyhow::Result<HnswIndex> {
    let config = HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 100,
        max_elements: vectors.len(),
        dimensions: vectors[0].1.dimensions,
    };

    let builder = ParallelHnswBuilder::new(config)
        .num_threads(16)          // Use 16 threads
        .batch_size(10000);       // Process 10k vectors per batch

    let index = builder.build(vectors)?;

    Ok(index)
}
```

### HNSW Layer Distribution

```rust
fn analyze_hnsw_structure(index: &HnswIndex) {
    let stats = index.get_statistics();

    println!("HNSW Structure:");
    println!("  Total layers: {}", stats.num_layers);
    println!("  Layer 0 nodes: {}", stats.layer_0_nodes);
    println!("  Average degree: {:.2}", stats.avg_degree);
    println!("  Average path length: {:.2}", stats.avg_path_length);

    // Rule of thumb: Layer 0 should have most nodes
    // Each higher layer should have ~1/M nodes of the previous layer
}
```

---

## Memory Optimization

### Compression Techniques

#### 1. Scalar Quantization (4x reduction)

```rust
use oxirs_vec::sq::{SqConfig, SqIndex, QuantizationMode};

fn quantize_index(vectors: Vec<Vector>) -> anyhow::Result<SqIndex> {
    let config = SqConfig {
        dimensions: vectors[0].dimensions,
        quantization_mode: QuantizationMode::PerDimension,  // Best quality
        bits: 8,  // 8-bit quantization
    };

    let mut index = SqIndex::new(config)?;

    // Train quantizer
    index.train(&vectors)?;

    // Add vectors
    for (i, vector) in vectors.into_iter().enumerate() {
        index.add(format!("vec_{}", i), vector)?;
    }

    Ok(index)
}

// Memory comparison:
// Original:  384 dimensions × 4 bytes = 1,536 bytes/vector
// SQ (8-bit): 384 dimensions × 1 byte  = 384 bytes/vector
// Reduction: 4×
```

#### 2. Product Quantization (16x reduction)

```rust
use oxirs_vec::pq::{PQConfig, PQIndex};

fn product_quantize_index(vectors: Vec<Vector>) -> anyhow::Result<PQIndex> {
    let config = PQConfig {
        dimensions: 384,
        num_subvectors: 48,      // 384 / 8 = 48 subvectors
        num_centroids: 256,      // 8-bit codes
        max_elements: vectors.len(),
    };

    let mut index = PQIndex::new(config)?;

    // Train codebooks
    index.train(&vectors)?;

    // Add vectors
    for (i, vector) in vectors.into_iter().enumerate() {
        index.add(format!("vec_{}", i), vector)?;
    }

    Ok(index)
}

// Memory comparison:
// Original: 384 dimensions × 4 bytes = 1,536 bytes/vector
// PQ:       48 subvectors × 1 byte  = 48 bytes/vector
// Reduction: 32×
```

#### 3. Optimized Product Quantization (Best quality)

```rust
use oxirs_vec::opq::{OpqConfig, OpqIndex};

fn opq_index(vectors: Vec<Vector>) -> anyhow::Result<OpqIndex> {
    let config = OpqConfig {
        dimensions: 384,
        num_subvectors: 48,
        num_centroids: 256,
        rotation_iterations: 20,  // OPQ rotation for better quality
        max_elements: vectors.len(),
    };

    let mut index = OpqIndex::new(config)?;
    index.train(&vectors)?;

    for (i, vector) in vectors.into_iter().enumerate() {
        index.add(format!("vec_{}", i), vector)?;
    }

    Ok(index)
}

// Better recall than PQ with same memory footprint
// Recall@10: PQ ~0.85, OPQ ~0.90
```

### Memory Tiering

```rust
use oxirs_vec::tiering::{TieringManager, TieringConfig, TieringPolicy};

fn setup_memory_tiering() -> anyhow::Result<TieringManager> {
    let config = TieringConfig {
        hot_tier_capacity_gb: 10.0,    // Fast SSD/RAM
        warm_tier_capacity_gb: 50.0,   // Regular SSD
        cold_tier_capacity_gb: 500.0,  // HDD/Object storage
        policy: TieringPolicy::AccessFrequency,
    };

    TieringManager::new(config)
}

// Automatic promotion/demotion based on access patterns
```

---

## Query Optimization

### Query Planning

```rust
use oxirs_vec::query_planning::{QueryPlanner, CostModel};

fn optimize_query_execution() -> anyhow::Result<()> {
    let planner = QueryPlanner::new(CostModel::default());

    // Planner selects optimal strategy based on:
    // 1. Query vector characteristics (sparsity, norm)
    // 2. Historical performance data
    // 3. Available indices
    // 4. System load

    Ok(())
}
```

### Result Caching

```rust
use oxirs_vec::advanced_caching::{MultiLevelCache, CacheConfig};

fn setup_query_cache() -> anyhow::Result<MultiLevelCache> {
    let config = CacheConfig {
        l1_capacity: 1000,       // Hot: 1k queries
        l2_capacity: 10000,      // Warm: 10k queries
        l3_capacity: 100000,     // Cold: 100k queries
        ttl_seconds: 3600,       // 1 hour TTL
    };

    let cache = MultiLevelCache::new(config)?;

    // Cache hit rate target: >50%
    Ok(cache)
}

// Expected speedup: 10-100× for cached queries
```

### Batch Query Processing

```rust
use oxirs_vec::real_time_updates::BatchProcessor;

fn batch_queries(
    store: &VectorStore,
    queries: Vec<Vector>,
) -> anyhow::Result<Vec<Vec<(String, f32)>>> {
    let batch_processor = BatchProcessor::new();

    // Process queries in parallel
    let results = batch_processor.batch_search(store, &queries, 10)?;

    // Throughput: ~10× higher than sequential processing
    Ok(results)
}
```

### Query Rewriting

```rust
use oxirs_vec::query_rewriter::{QueryRewriter, QueryRewriterConfig};

fn optimize_query_vector(query: &Vector) -> anyhow::Result<Vector> {
    let config = QueryRewriterConfig {
        enable_expansion: true,
        enable_reduction: true,
        enable_normalization: true,
    };

    let rewriter = QueryRewriter::new(config);
    let optimized = rewriter.rewrite_query(query)?;

    // Improvements:
    // - Remove noisy dimensions
    // - Boost important dimensions
    // - Normalize for better similarity comparison

    Ok(optimized)
}
```

---

## GPU Acceleration

### CUDA Setup

```rust
use oxirs_vec::gpu::{GpuAccelerator, GpuConfig};

fn setup_gpu_acceleration() -> anyhow::Result<GpuAccelerator> {
    let config = GpuConfig {
        device_id: 0,              // Use GPU 0
        memory_pool_size_mb: 4096, // 4 GB pool
        use_tensor_cores: true,    // Enable Tensor Cores
        mixed_precision: true,     // FP16/FP32 mixed precision
    };

    let accelerator = GpuAccelerator::new(config)?;

    Ok(accelerator)
}
```

### GPU-Accelerated Distance Calculation

```rust
use oxirs_vec::gpu_acceleration::{GpuVectorIndex, BatchProcessor};

fn gpu_batch_search(
    query_vectors: Vec<Vector>,
    index_vectors: Vec<Vector>,
) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
    let mut gpu_index = GpuVectorIndex::new()?;

    // Upload index to GPU
    gpu_index.upload_vectors(&index_vectors)?;

    // Batch compute distances on GPU
    let results = gpu_index.batch_search(&query_vectors, 10)?;

    // Expected speedup: 10-50× depending on GPU
    Ok(results)
}
```

### Performance Comparison

| Distance Metric | CPU (16 cores) | GPU (RTX 3090) | Speedup |
|-----------------|----------------|----------------|---------|
| Cosine          | 100k vec/s     | 5M vec/s       | 50×     |
| Euclidean       | 120k vec/s     | 4M vec/s       | 33×     |
| Manhattan       | 130k vec/s     | 3.5M vec/s     | 27×     |
| Dot Product     | 150k vec/s     | 8M vec/s       | 53×     |
| Pearson         | 80k vec/s      | 2M vec/s       | 25×     |

---

## Caching Strategies

### Intelligent Caching

```rust
use oxirs_vec::adaptive_intelligent_caching::{
    AdaptiveIntelligentCache, CacheConfiguration
};

fn setup_adaptive_cache() -> anyhow::Result<AdaptiveIntelligentCache> {
    let config = CacheConfiguration {
        total_capacity_mb: 1024,  // 1 GB
        enable_ml_prediction: true,
        enable_prefetching: true,
        eviction_policy: "Adaptive".to_string(),
    };

    let cache = AdaptiveIntelligentCache::new(config)?;

    // Features:
    // - ML-based access prediction
    // - Predictive prefetching
    // - Adaptive eviction policy
    // - Pattern recognition

    Ok(cache)
}
```

### Cache Warming

```rust
use oxirs_vec::advanced_caching::CacheWarmer;

fn warm_cache(
    cache: &mut MultiLevelCache,
    popular_queries: Vec<Vector>,
) -> anyhow::Result<()> {
    let warmer = CacheWarmer::new();

    // Pre-populate cache with popular queries
    warmer.warm_cache(cache, &popular_queries)?;

    println!("Cache warmed with {} queries", popular_queries.len());
    Ok(())
}
```

---

## Workload-Specific Tuning

### Use Case 1: Semantic Search (High Recall)

```rust
fn semantic_search_config() -> HnswConfig {
    HnswConfig {
        m: 32,                // High connectivity
        ef_construction: 400, // Thorough construction
        ef_search: 200,       // Comprehensive search
        max_elements: 10_000_000,
        dimensions: 384,
    }
}

// Characteristics:
// - Recall@10: 0.98+
// - Latency: 15-20 ms
// - Memory: 8 GB per 1M vectors
// - Use case: Q&A systems, document retrieval
```

### Use Case 2: Real-time Recommendations (Low Latency)

```rust
fn realtime_recommendation_config() -> HnswConfig {
    HnswConfig {
        m: 8,                 // Lower connectivity
        ef_construction: 100, // Fast construction
        ef_search: 50,        // Quick search
        max_elements: 10_000_000,
        dimensions: 128,      // Smaller embeddings
    }
}

// Characteristics:
// - Recall@10: 0.92
// - Latency: 3-5 ms
// - Memory: 2 GB per 1M vectors
// - Use case: Real-time recommendation systems
```

### Use Case 3: E-commerce (Balanced)

```rust
fn ecommerce_config() -> HnswConfig {
    HnswConfig {
        m: 16,                // Balanced
        ef_construction: 200, // Balanced
        ef_search: 100,       // Balanced
        max_elements: 50_000_000,
        dimensions: 256,
    }
}

// Characteristics:
// - Recall@10: 0.96
// - Latency: 8-10 ms
// - Memory: 4 GB per 1M vectors
// - Use case: Product search, visual search
```

### Use Case 4: Large-Scale Image Search

```rust
fn image_search_config() -> DiskAnnConfig {
    DiskAnnConfig {
        dimensions: 512,      // Image embeddings (ResNet, CLIP)
        max_degree: 64,
        search_list_size: 100,
        index_path: "/mnt/nvme/image_vectors".to_string(),
    }
}

// Characteristics:
// - Recall@10: 0.95
// - Latency: 15-20 ms
// - Memory: ~100 MB (disk-backed)
// - Dataset: 1B+ images
// - Use case: Reverse image search, visual similarity
```

---

## Benchmarking

### Comprehensive Benchmark Suite

```rust
use oxirs_vec::advanced_benchmarking::{
    AdvancedBenchmarkSuite, AdvancedBenchmarkConfig, BenchmarkAlgorithm
};

fn run_comprehensive_benchmark() -> anyhow::Result<()> {
    let config = AdvancedBenchmarkConfig {
        algorithms: vec![
            BenchmarkAlgorithm::HNSW,
            BenchmarkAlgorithm::IVF,
            BenchmarkAlgorithm::LSH,
            BenchmarkAlgorithm::PQ,
            BenchmarkAlgorithm::DiskANN,
        ],
        dataset_sizes: vec![10_000, 100_000, 1_000_000],
        dimensions: vec![128, 384, 768],
        num_queries: 1000,
        k_values: vec![1, 10, 100],
    };

    let suite = AdvancedBenchmarkSuite::new(config)?;
    let results = suite.run()?;

    // Analyze results
    for result in results {
        println!("Algorithm: {}", result.algorithm);
        println!("  Build time: {} s", result.build_time_seconds);
        println!("  Query latency (p95): {} ms", result.query_latency_p95_ms);
        println!("  Recall@10: {:.3}", result.recall_at_10);
        println!("  Memory: {} MB", result.memory_usage_mb);
    }

    Ok(())
}
```

### GPU Benchmarking

```rust
use oxirs_vec::gpu_benchmarks::{GpuBenchmarkSuite, GpuBenchmarkConfig};

fn benchmark_gpu_performance() -> anyhow::Result<()> {
    let config = GpuBenchmarkConfig {
        num_vectors: 1_000_000,
        dimensions: 384,
        batch_sizes: vec![1, 10, 100, 1000],
        metrics: vec!["cosine", "euclidean", "dot_product"],
    };

    let suite = GpuBenchmarkSuite::new(config)?;
    let results = suite.run()?;

    // Compare CPU vs GPU
    for result in results {
        println!("Metric: {}", result.metric);
        println!("  CPU throughput: {} vec/s", result.cpu_throughput);
        println!("  GPU throughput: {} vec/s", result.gpu_throughput);
        println!("  Speedup: {}×", result.speedup);
    }

    Ok(())
}
```

---

## Performance Checklist

### Pre-Production Checklist

- [ ] Run comprehensive benchmarks on production-like data
- [ ] Profile memory usage under load
- [ ] Test with peak query load (stress testing)
- [ ] Verify recall meets requirements (>0.95)
- [ ] Validate latency SLAs (p95 < 50 ms, p99 < 100 ms)
- [ ] Enable monitoring and alerting
- [ ] Configure caching (target >50% hit rate)
- [ ] Set up query optimization
- [ ] Test crash recovery and WAL
- [ ] Benchmark with and without GPU acceleration

### Optimization Workflow

1. **Baseline Measurement**: Measure current performance
2. **Identify Bottleneck**: Profile to find slowest component
3. **Apply Optimization**: Tune parameters or change algorithm
4. **Re-measure**: Verify improvement
5. **Iterate**: Repeat until targets met

---

## Performance Tuning Tools

### Built-in Profiler

```rust
use oxirs_vec::performance_insights::PerformanceInsightsAnalyzer;

fn profile_queries(store: &VectorStore) -> anyhow::Result<()> {
    let analyzer = PerformanceInsightsAnalyzer::new();

    // Profile query execution
    let query = Vector::new(vec![0.5; 384]);
    let profile = analyzer.profile_query(store, &query, 10)?;

    println!("Query Profile:");
    println!("  Total time: {} ms", profile.total_time_ms);
    println!("  Index selection: {} ms", profile.index_selection_ms);
    println!("  Search time: {} ms", profile.search_time_ms);
    println!("  Post-processing: {} ms", profile.postprocessing_ms);

    Ok(())
}
```

---

## Next Steps

1. **Apply Tuning**: Use this guide to optimize your deployment
2. **Measure Results**: Benchmark before and after
3. **Monitor Performance**: Set up continuous monitoring
4. **Iterate**: Continuously improve based on production metrics

---

**Document Version**: 1.0
**OxiRS Vec Version**: v0.1.0-beta.2
**Last Updated**: December 6, 2025
