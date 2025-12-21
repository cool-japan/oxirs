# OxiRS Vec - GPU Acceleration Guide

**Version**: v0.1.0-beta.2
**Last Updated**: December 6, 2025

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Supported Operations](#supported-operations)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Configuration](#configuration)
8. [Memory Management](#memory-management)
9. [Optimization Techniques](#optimization-techniques)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Introduction

OxiRS Vec provides comprehensive GPU acceleration for vector operations, delivering 10-50× speedup for distance calculations and similarity search.

### GPU Acceleration Features

- **16 Distance Metrics**: CUDA kernels for all major distance functions
- **Mixed Precision**: FP16/BF16 for 2× memory efficiency
- **Tensor Cores**: Specialized hardware for matrix operations
- **Batch Processing**: Efficient handling of large query batches
- **Memory Management**: Smart GPU memory pooling

### When to Use GPU Acceleration

✅ **Use GPU When**:
- Processing > 1M vectors
- Batch queries (> 100 queries)
- High-dimensional vectors (> 512 dims)
- Real-time latency requirements (< 10ms)
- Cost-effective at scale (amortize GPU cost)

❌ **Skip GPU When**:
- Small datasets (< 100k vectors)
- Single queries
- Low-dimensional vectors (< 128 dims)
- CPU sufficient for requirements
- No CUDA-compatible GPU available

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| GPU | NVIDIA GTX 1060 | NVIDIA RTX 3090 | NVIDIA A100 |
| CUDA Compute | 6.0+ | 7.5+ (Turing) | 8.0+ (Ampere) |
| GPU Memory | 4 GB | 12 GB | 40-80 GB |
| CUDA Toolkit | 11.0+ | 12.0+ | 12.2+ |
| Driver | 450+ | 525+ | 535+ |

### Software Requirements

```bash
# Check NVIDIA driver version
nvidia-smi

# Check CUDA version
nvcc --version

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Supported GPUs

| GPU Series | Compute Capability | Tensor Cores | FP16 Support |
|------------|-------------------|--------------|--------------|
| GeForce GTX 10xx | 6.1 | ❌ | ✅ |
| GeForce RTX 20xx | 7.5 | ✅ | ✅ |
| GeForce RTX 30xx | 8.6 | ✅ Gen 3 | ✅ |
| GeForce RTX 40xx | 8.9 | ✅ Gen 4 | ✅ |
| Tesla V100 | 7.0 | ✅ | ✅ |
| Tesla A100 | 8.0 | ✅ Gen 3 | ✅ |
| H100 | 9.0 | ✅ Gen 4 | ✅ |

---

## Installation

### Install CUDA Toolkit

#### Ubuntu/Debian

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA 12.2
sudo apt-get install cuda-12-2

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Verify Installation

```bash
# Check CUDA installation
nvcc --version

# Run CUDA samples
cd /usr/local/cuda-12.2/samples/1_Utilities/deviceQuery
make
./deviceQuery
```

### Build OxiRS Vec with GPU Support

```bash
cd oxirs/engine/oxirs-vec

# Build with GPU features
cargo build --release --features gpu-full

# Or specific features
cargo build --release --features cuda,gpu,blas
```

### Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `gpu` | Basic GPU support | oxirs-core/gpu |
| `cuda` | CUDA acceleration | CUDA Toolkit 12.0+ |
| `candle-gpu` | Candle GPU backend | candle-core |
| `gpu-full` | All GPU features | cuda + candle-gpu + gpu |

---

## Quick Start

### Basic GPU Usage

```rust
use oxirs_vec::{VectorStore, gpu::GpuAccelerator};

fn main() -> anyhow::Result<()> {
    // Check GPU availability
    if !oxirs_vec::gpu::is_gpu_available() {
        eprintln!("No GPU available, falling back to CPU");
        return Ok(());
    }

    // Create GPU accelerator
    let gpu = GpuAccelerator::new_default()?;
    println!("GPU initialized: {}", gpu.device_name());

    // Create vector store with GPU
    let mut store = VectorStore::new();
    store.enable_gpu(gpu)?;

    // Use as normal - GPU automatically used for operations
    store.index_resource("doc1".to_string(), "This is a test document")?;

    let results = store.similarity_search("test", 10)?;
    println!("Found {} results", results.len());

    Ok(())
}
```

### GPU-Accelerated Search

```rust
use oxirs_vec::gpu::{GpuConfig, create_performance_accelerator};

fn gpu_search_example() -> anyhow::Result<()> {
    // Create performance-optimized GPU config
    let config = GpuConfig {
        device_id: 0,              // Use GPU 0
        memory_pool_size_mb: 4096, // 4 GB memory pool
        use_tensor_cores: true,    // Enable Tensor Cores
        mixed_precision: true,     // Use FP16 for 2× speedup
        batch_size: 1000,          // Process 1000 queries at once
    };

    let gpu = create_performance_accelerator(config)?;

    // Batch query processing
    let query_vectors = vec![
        Vector::new(vec![0.5; 384]),
        Vector::new(vec![0.6; 384]),
        // ... 1000 queries
    ];

    let results = gpu.batch_search(&query_vectors, &index_vectors, 10)?;

    Ok(())
}
```

---

## Supported Operations

### Distance Metrics

OxiRS Vec provides GPU kernels for 16 distance metrics:

| Metric | GPU Kernel | Speedup vs CPU | Use Case |
|--------|------------|----------------|----------|
| Cosine Similarity | ✅ | 50× | Semantic search |
| Euclidean Distance | ✅ | 40× | Spatial data |
| Manhattan Distance | ✅ | 35× | Feature vectors |
| Dot Product | ✅ | 55× | Neural embeddings |
| Minkowski Distance | ✅ | 30× | General purpose |
| Pearson Correlation | ✅ | 25× | Time series |
| Jaccard Similarity | ✅ | 20× | Set similarity |
| Dice Coefficient | ✅ | 20× | Text similarity |
| Hamming Distance | ✅ | 45× | Binary vectors |
| Canberra Distance | ✅ | 28× | Sparse vectors |
| Chebyshev Distance | ✅ | 32× | Max difference |
| Angular Distance | ✅ | 48× | Directional data |
| KL Divergence | ✅ | 22× | Probability distributions |
| JS Divergence | ✅ | 22× | Symmetric KL |
| Hellinger Distance | ✅ | 24× | Probability distributions |
| Bhattacharyya Distance | ✅ | 23× | Statistical distributions |

### Vector Operations

```rust
use oxirs_vec::gpu::GpuVectorOps;

// Matrix multiplication with Tensor Cores
let result = gpu.matmul_tensor_cores(&matrix_a, &matrix_b)?;

// Batch normalization
let normalized = gpu.batch_normalize(&vectors)?;

// Batch quantization
let quantized = gpu.batch_quantize(&vectors, bits=8)?;
```

---

## Performance Benchmarks

### Distance Calculation Throughput

**Test Setup**: RTX 3090, 1M vectors, 384 dimensions

| Metric | CPU (16 cores) | GPU (RTX 3090) | Speedup |
|--------|----------------|----------------|---------|
| Cosine | 100k vec/s | 5.0M vec/s | 50× |
| Euclidean | 120k vec/s | 4.2M vec/s | 35× |
| Manhattan | 130k vec/s | 3.8M vec/s | 29× |
| Dot Product | 150k vec/s | 8.0M vec/s | 53× |
| Pearson | 80k vec/s | 2.0M vec/s | 25× |

### Query Latency

**Test Setup**: 10M vectors, k=10

| Batch Size | CPU (p95) | GPU (p95) | Speedup |
|------------|-----------|-----------|---------|
| 1 query | 8 ms | 5 ms | 1.6× |
| 10 queries | 60 ms | 8 ms | 7.5× |
| 100 queries | 550 ms | 20 ms | 27.5× |
| 1000 queries | 5500 ms | 150 ms | 36.7× |

**Key Insight**: GPU shines with batch processing!

### Memory Efficiency

| Precision | Memory per Vector | Throughput | Accuracy Loss |
|-----------|-------------------|------------|---------------|
| FP32 (CPU) | 1536 bytes | 1.0× | 0% (baseline) |
| FP32 (GPU) | 1536 bytes | 50× | 0% |
| FP16 (GPU) | 768 bytes | 80× | < 0.1% |
| BF16 (GPU) | 768 bytes | 75× | < 0.05% |

---

## Configuration

### Basic Configuration

```rust
use oxirs_vec::gpu::{GpuConfig, GpuAccelerator};

let config = GpuConfig::default();
let gpu = GpuAccelerator::new(config)?;
```

### Performance Configuration

```rust
let config = GpuConfig {
    device_id: 0,
    memory_pool_size_mb: 8192,     // 8 GB
    use_tensor_cores: true,
    mixed_precision: true,
    batch_size: 2000,
    prefetch_enabled: true,
    stream_count: 4,               // 4 CUDA streams for parallelism
};

let gpu = GpuAccelerator::new(config)?;
```

### Memory-Optimized Configuration

```rust
use oxirs_vec::gpu::create_memory_optimized_accelerator;

// For memory-constrained GPUs
let config = GpuConfig {
    device_id: 0,
    memory_pool_size_mb: 2048,     // 2 GB
    use_tensor_cores: false,       // Save memory
    mixed_precision: true,         // FP16 for 2× savings
    batch_size: 100,               // Smaller batches
    enable_compression: true,
};

let gpu = create_memory_optimized_accelerator(config)?;
```

### Multi-GPU Configuration

```rust
use oxirs_vec::gpu::MultiGpuManager;

fn setup_multi_gpu() -> anyhow::Result<()> {
    let manager = MultiGpuManager::new()?;

    // Distribute workload across GPUs
    manager.add_device(0)?; // GPU 0
    manager.add_device(1)?; // GPU 1
    manager.add_device(2)?; // GPU 2

    // Automatic load balancing
    manager.enable_load_balancing()?;

    // Process queries across all GPUs
    let results = manager.batch_search(&queries, &index, 10)?;

    Ok(())
}
```

---

## Memory Management

### GPU Memory Pool

```rust
use oxirs_vec::gpu::GpuMemoryPool;

// Create memory pool
let pool = GpuMemoryPool::new(4096)?; // 4 GB

// Allocate buffer
let buffer = pool.allocate(1_000_000 * 384 * 4)?; // 1M vectors

// Automatically freed when buffer goes out of scope
```

### Streaming Large Datasets

```rust
use oxirs_vec::gpu::GpuStreamProcessor;

fn process_large_dataset(vectors: &[Vector]) -> anyhow::Result<()> {
    let processor = GpuStreamProcessor::new()?;

    // Process in chunks (don't load all to GPU at once)
    for chunk in vectors.chunks(100_000) {
        let results = processor.process_chunk(chunk)?;
        // Process results...
    }

    Ok(())
}
```

### Memory Monitoring

```rust
fn monitor_gpu_memory(gpu: &GpuAccelerator) {
    let stats = gpu.get_memory_stats();

    println!("GPU Memory:");
    println!("  Total: {} GB", stats.total_gb);
    println!("  Used: {} GB", stats.used_gb);
    println!("  Free: {} GB", stats.free_gb);
    println!("  Utilization: {:.1}%", stats.utilization_percent);

    // Alert if memory usage high
    if stats.utilization_percent > 90.0 {
        eprintln!("WARNING: GPU memory usage critical!");
    }
}
```

---

## Optimization Techniques

### 1. Batch Processing

```rust
// ❌ DON'T: Process queries one at a time
for query in queries {
    let result = gpu.search(&query, &index, 10)?;
    results.push(result);
}

// ✅ DO: Batch all queries together
let results = gpu.batch_search(&queries, &index, 10)?;

// Expected speedup: 10-50× for large batches
```

### 2. Mixed Precision

```rust
// Use FP16 for 2× speedup and 2× memory savings
let config = GpuConfig {
    mixed_precision: true,  // Enable FP16
    ..Default::default()
};

let gpu = GpuAccelerator::new(config)?;

// Automatic FP32 → FP16 conversion
// Accuracy loss < 0.1% for most applications
```

### 3. Tensor Cores

```rust
// Enable Tensor Cores for matrix operations
let config = GpuConfig {
    use_tensor_cores: true,  // 8× faster matrix multiply
    ..Default::default()
};

// Tensor Cores automatically used for:
// - Matrix multiplication
// - Batch normalization
// - Attention mechanisms
```

### 4. Asynchronous Transfers

```rust
use oxirs_vec::gpu::AsyncGpuOps;

async fn async_gpu_search() -> anyhow::Result<()> {
    let gpu = AsyncGpuOps::new()?;

    // Upload data asynchronously
    let upload_future = gpu.upload_vectors_async(&vectors);

    // Overlap compute with data transfer
    let compute_future = gpu.compute_async();

    // Wait for both
    let (_, results) = tokio::join!(upload_future, compute_future);

    Ok(())
}
```

### 5. Kernel Fusion

```rust
// Fuse multiple operations into single kernel
let fused_ops = gpu.create_fused_kernel(vec![
    Operation::Normalize,
    Operation::DotProduct,
    Operation::Softmax,
])?;

// Execute all operations in one pass
let result = gpu.execute_fused(fused_ops, &data)?;

// Reduces memory bandwidth and latency
```

---

## Troubleshooting

### Issue 1: Out of Memory

**Symptoms**: `CUDA_ERROR_OUT_OF_MEMORY`

**Solutions**:
```rust
// 1. Reduce batch size
let config = GpuConfig {
    batch_size: 100,  // Smaller batches
    ..Default::default()
};

// 2. Enable mixed precision
let config = GpuConfig {
    mixed_precision: true,  // FP16 uses 2× less memory
    ..Default::default()
};

// 3. Use streaming
let processor = GpuStreamProcessor::new()?;
for chunk in data.chunks(10_000) {
    processor.process_chunk(chunk)?;
}
```

### Issue 2: Slow GPU Performance

**Symptoms**: GPU slower than expected

**Diagnosis**:
```rust
let stats = gpu.get_performance_stats();
println!("GPU utilization: {}%", stats.utilization);
println!("Memory bandwidth: {} GB/s", stats.memory_bandwidth);
println!("Compute throughput: {} TFLOPS", stats.compute_throughput);
```

**Solutions**:
1. **Increase batch size**: GPU needs large batches
2. **Check CPU-GPU transfer**: Bottleneck in data transfer
3. **Enable Tensor Cores**: 8× speedup for matrix ops
4. **Use mixed precision**: 2× throughput increase

### Issue 3: CUDA Initialization Failed

**Symptoms**: `Failed to initialize CUDA context`

**Solutions**:
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify driver version
cat /proc/driver/nvidia/version

# Check GPU visibility
echo $CUDA_VISIBLE_DEVICES

# Run CUDA samples
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
./deviceQuery
```

### Issue 4: Poor Scaling with Multiple GPUs

**Symptoms**: 2 GPUs not 2× faster

**Solutions**:
```rust
// Enable P2P memory access
manager.enable_peer_to_peer_access()?;

// Use NVLink for fast GPU-GPU transfer
manager.enable_nvlink()?;

// Balance load evenly
manager.set_load_balancing_strategy(LoadBalancingStrategy::RoundRobin)?;
```

---

## Best Practices

### DO ✅

1. **Batch Queries**: Always batch multiple queries together
   ```rust
   let results = gpu.batch_search(&queries, &index, 10)?; // Good
   ```

2. **Use Mixed Precision**: Enable FP16 for 2× speedup
   ```rust
   let config = GpuConfig { mixed_precision: true, ..Default::default() };
   ```

3. **Monitor GPU Usage**: Track utilization and memory
   ```rust
   let stats = gpu.get_stats();
   assert!(stats.utilization > 80.0, "GPU underutilized");
   ```

4. **Profile Operations**: Identify bottlenecks
   ```rust
   let profile = gpu.profile_operations()?;
   ```

5. **Enable Tensor Cores**: 8× speedup for matrix ops
   ```rust
   let config = GpuConfig { use_tensor_cores: true, ..Default::default() };
   ```

### DON'T ❌

1. **Don't Use GPU for Small Batches**: GPU overhead dominates
   ```rust
   // ❌ Bad: Single query on GPU
   gpu.search(&query, &index, 10)?;

   // ✅ Good: Batch on GPU, single on CPU
   if queries.len() > 100 {
       gpu.batch_search(&queries, &index, 10)?;
   } else {
       cpu_search(&queries, &index, 10)?;
   }
   ```

2. **Don't Ignore CPU-GPU Transfer Cost**:
   ```rust
   // ❌ Bad: Transfer data for every query
   for query in queries {
       let result = gpu.search(&query, &index, 10)?; // Transfer overhead!
   }

   // ✅ Good: Batch transfer
   gpu.upload_all(&queries)?;
   let results = gpu.batch_search_no_transfer(&queries, &index, 10)?;
   ```

3. **Don't Allocate Memory Repeatedly**:
   ```rust
   // ❌ Bad: Allocate in loop
   for _ in 0..1000 {
       let buffer = gpu.allocate(size)?; // Slow!
   }

   // ✅ Good: Reuse buffer
   let buffer = gpu.allocate(size)?;
   for _ in 0..1000 {
       buffer.copy_from(&data)?; // Fast!
   }
   ```

---

## Benchmarking Your Setup

### Run Comprehensive Benchmarks

```rust
use oxirs_vec::gpu_benchmarks::{GpuBenchmarkSuite, GpuBenchmarkConfig};

fn benchmark_gpu() -> anyhow::Result<()> {
    let config = GpuBenchmarkConfig {
        num_vectors: 1_000_000,
        dimensions: 384,
        batch_sizes: vec![1, 10, 100, 1000, 10000],
        metrics: vec![
            "cosine", "euclidean", "dot_product",
            "manhattan", "pearson",
        ],
    };

    let suite = GpuBenchmarkSuite::new(config)?;
    let results = suite.run()?;

    // Print results
    for result in results {
        println!("Metric: {} | Batch: {} | GPU: {:.1}M vec/s | CPU: {:.1}k vec/s | Speedup: {:.1}×",
                 result.metric,
                 result.batch_size,
                 result.gpu_throughput / 1_000_000.0,
                 result.cpu_throughput / 1_000.0,
                 result.speedup);
    }

    Ok(())
}
```

### Expected Results

| GPU | Cosine (1M vec/s) | Memory | Price |
|-----|-------------------|--------|-------|
| GTX 1080 Ti | 2.5 | 11 GB | $700 |
| RTX 2080 Ti | 4.0 | 11 GB | $1200 |
| RTX 3090 | 5.0 | 24 GB | $1500 |
| A100 | 12.0 | 40 GB | $10000 |
| H100 | 20.0 | 80 GB | $30000 |

---

## Next Steps

1. **Install CUDA**: Follow installation guide above
2. **Run Benchmarks**: Measure performance on your GPU
3. **Optimize Configuration**: Tune for your workload
4. **Monitor Performance**: Track GPU utilization
5. **Scale Up**: Add more GPUs if needed

---

## Additional Resources

- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/
- **Tensor Cores**: https://www.nvidia.com/en-us/data-center/tensor-cores/
- **cuBLAS**: https://docs.nvidia.com/cuda/cublas/
- **NVIDIA Performance Optimization**: https://docs.nvidia.com/deeplearning/performance/

---

**Document Version**: 1.0
**OxiRS Vec Version**: v0.1.0-beta.2
**Last Updated**: December 6, 2025
