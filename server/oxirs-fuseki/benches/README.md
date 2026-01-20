# OxiRS Fuseki - Load Testing and Benchmarking Suite

Comprehensive load testing and performance benchmarking suite for OxiRS Fuseki.

## Overview

This suite includes:

1. **Load Testing** (`load_testing.rs`) - Stress testing under various load conditions
2. **Performance Benchmarking** (`performance_benchmarks.rs`) - Detailed performance analysis of 0.1.0 optimizations

## Running Benchmarks

### All Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run with specific features
cargo bench --features production
```

### Specific Benchmark Suites

```bash
# Load testing only
cargo bench --bench load_testing

# Performance benchmarks only
cargo bench --bench performance_benchmarks
```

### Individual Benchmark Groups

```bash
# Concurrent queries benchmark
cargo bench --bench load_testing -- concurrent_queries

# Memory pool benchmark
cargo bench --bench performance_benchmarks -- memory_pool

# Work stealing scheduler
cargo bench --bench performance_benchmarks -- work_stealing
```

## Benchmark Categories

### Load Testing

#### 1. Concurrent Queries
Tests system behavior under various concurrency levels (10, 100, 1000 concurrent requests).

```bash
cargo bench --bench load_testing -- concurrent_queries
```

#### 2. Query Latency
Measures query response times across different dataset sizes.

```bash
cargo bench --bench load_testing -- query_latency
```

#### 3. Throughput (QPS)
Measures queries per second sustained over time.

```bash
cargo bench --bench load_testing -- throughput
```

#### 4. Batch Queries
Tests batch query execution performance.

```bash
cargo bench --bench load_testing -- batch_queries
```

#### 5. Result Streaming
Benchmarks memory-efficient result streaming.

```bash
cargo bench --bench load_testing -- result_streaming
```

#### 6. Dataset Operations
Tests dataset creation, backup, and snapshot operations.

```bash
cargo bench --bench load_testing -- dataset_operations
```

#### 7. Memory Pressure
Tests performance under memory constraints.

```bash
cargo bench --bench load_testing -- memory_pressure
```

#### 8. Connection Pooling
Benchmarks connection pool performance.

```bash
cargo bench --bench load_testing -- connection_pool
```

#### 9. Query Caching
Compares cache hit vs miss performance.

```bash
cargo bench --bench load_testing -- query_cache
```

### Performance Benchmarking (v0.1.0 Optimizations)

#### Concurrency (concurrent.rs)

1. **Work-Stealing Scheduler**
   ```bash
   cargo bench --bench performance_benchmarks -- work_stealing
   ```

2. **Priority Queue**
   ```bash
   cargo bench --bench performance_benchmarks -- priority_queue
   ```

3. **Load Shedding**
   ```bash
   cargo bench --bench performance_benchmarks -- load_shedding
   ```

#### Memory Management (memory_pool.rs)

4. **Memory Pool**
   ```bash
   cargo bench --bench performance_benchmarks -- memory_pool
   ```

5. **Memory Adaptation**
   ```bash
   cargo bench --bench performance_benchmarks -- memory_adaptation
   ```

6. **Chunked Arrays**
   ```bash
   cargo bench --bench performance_benchmarks -- chunked_arrays
   ```

7. **Garbage Collection**
   ```bash
   cargo bench --bench performance_benchmarks -- garbage_collection
   ```

#### Batching (batch_execution.rs)

8. **Request Batching**
   ```bash
   cargo bench --bench performance_benchmarks -- request_batching
   ```

9. **Adaptive Batching**
   ```bash
   cargo bench --bench performance_benchmarks -- adaptive_batching
   ```

10. **Parallel Execution**
    ```bash
    cargo bench --bench performance_benchmarks -- parallel_execution
    ```

#### Streaming (streaming_results.rs)

11. **Zero-Copy Streaming**
    ```bash
    cargo bench --bench performance_benchmarks -- zero_copy_streaming
    ```

12. **Compression Streaming**
    ```bash
    cargo bench --bench performance_benchmarks -- compression_streaming
    ```

13. **Backpressure**
    ```bash
    cargo bench --bench performance_benchmarks -- backpressure
    ```

#### Dataset Management (dataset_management.rs)

14. **Bulk Operations**
    ```bash
    cargo bench --bench performance_benchmarks -- bulk_operations
    ```

15. **Snapshots**
    ```bash
    cargo bench --bench performance_benchmarks -- snapshots
    ```

16. **Versioning**
    ```bash
    cargo bench --bench performance_benchmarks -- versioning
    ```

## Benchmark Output

Benchmarks generate detailed HTML reports in `target/criterion/`:

```bash
# View results
open target/criterion/report/index.html

# View specific benchmark
open target/criterion/concurrent_queries/report/index.html
```

## Performance Targets

### v0.1.0 Targets

| Metric | Target | Current |
|--------|--------|---------|
| **Concurrent Queries (100)** | < 100ms p95 | TBD |
| **Query Latency (10k triples)** | < 50ms p95 | TBD |
| **Throughput** | > 10,000 QPS | TBD |
| **Memory Pool (acquire/release)** | > 1M ops/sec | TBD |
| **Result Streaming (100k results)** | < 500ms | TBD |
| **Batch Processing (100 queries)** | < 200ms | TBD |
| **Zero-Copy Streaming** | > 1GB/sec | TBD |

## Custom Benchmarks

### Creating Custom Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn custom_benchmark(c: &mut Criterion) {
    c.bench_function("my_custom_test", |b| {
        b.iter(|| {
            // Your code here
            black_box(expensive_operation());
        });
    });
}

criterion_group!(benches, custom_benchmark);
criterion_main!(benches);
```

### Running Custom Benchmarks

```bash
# Add to benches/ directory and run
cargo bench --bench custom_benchmark
```

## Profiling

### CPU Profiling

```bash
# Install flamegraph
cargo install flamegraph

# Profile benchmark
cargo flamegraph --bench load_testing -- --bench
```

### Memory Profiling

```bash
# Install heaptrack
sudo apt-get install heaptrack  # Linux
brew install heaptrack          # macOS

# Profile memory usage
heaptrack cargo bench --bench load_testing
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run benchmarks
        run: cargo bench --workspace
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion
```

## Interpreting Results

### Key Metrics

- **Time**: Lower is better
- **Throughput**: Higher is better
- **Iterations**: More iterations = more confidence
- **R² (goodness of fit)**: Closer to 1.0 is better
- **Mean**: Average execution time
- **Std Dev**: Consistency (lower = more consistent)

### Regression Detection

Criterion automatically detects performance regressions:

```
concurrent_queries/10  time:   [10.234 ms 10.456 ms 10.678 ms]
                       change: [+15.23% +18.45% +21.67%] (p = 0.00 < 0.05)
                       Performance has regressed.
```

## Troubleshooting

### Issue: Benchmarks taking too long

```bash
# Reduce measurement time
cargo bench -- --measurement-time 5

# Reduce sample size
cargo bench -- --sample-size 10
```

### Issue: Inconsistent results

```bash
# Increase warm-up time
cargo bench -- --warm-up-time 10

# Increase sample size
cargo bench -- --sample-size 100
```

### Issue: Out of memory

```bash
# Run benchmarks sequentially
cargo bench -- --test-threads=1
```

## Best Practices

1. **Isolate System**: Run benchmarks on a dedicated machine or CI
2. **Consistent Environment**: Use same hardware/OS for comparisons
3. **Warm-up**: Let JIT and caches stabilize before measuring
4. **Multiple Runs**: Run benchmarks multiple times for confidence
5. **Baseline**: Compare against known baseline or previous versions
6. **Document**: Record environment details with results

## Further Reading

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Flamegraph Documentation](https://github.com/flamegraph-rs/flamegraph)

## Support

For issues and questions:
- **GitHub Issues**: https://github.com/cool-japan/oxirs/issues
- **Documentation**: https://github.com/cool-japan/oxirs/tree/main/docs
